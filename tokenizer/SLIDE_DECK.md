# GraphGuard / CHARM — Full Pipeline Slide Deck
### From Raw Benchmark Data → GNN Hallucination Score

---

## Slide 1 — Title & Overview

# GraphGuard: CHARM Critic
## Catching Hallucinations by Reading the Attention Graph

**CHARM** = **C**ritical **H**allucination **A**nalysis via **R**esidual **M**essage-passing

### The Core Idea
LLMs generate tokens one at a time. At every step, the attention mechanism reveals *which prior tokens* the model is "looking at." We surgically intercept two signals from inside the model:

1. **Hidden States** — the deep 1024-dimensional vector the last transformer layer produces for each generated token. This is the model's "compressed thought" about that token.
2. **Sparse Attention Edges** — the strongest attention links from each new token back to previously generated tokens. These form a directed graph of logical dependency.

We then package each full generation as a **PyTorch Geometric (PyG) graph** where:
- **Nodes** = generated tokens, each carrying a 1024-dim hidden state
- **Edges** = the strongest attention links between them, weighted by attention probability

A **Graph Attention Network (GNN)** trained on labelled examples learns to read this graph and output a single number: the **hallucination risk score** between 0.0 and 1.0.

**Pipeline Stages:**
```
Dataset CSV → BenchmarkLoader → Prompt Builder → Qwen 3.5 (Outlines Fenced) →
TraceExtractor (Hooks) → AutoLabeler → UnifiedTraceRecord → PyG .pt File →
GNN Trainer → CHARM Critic Weights → Standalone Inference API
```

---

## Slide 2 — The Problem: Why Standard LLM Outputs Aren't Enough

### What is Hallucination?
A language model states something with confidence that is factually incorrect or unsupported by context. Classic detection approaches look at:
- The **output text** alone (keyword matching, NLI classifiers)
- The model's **self-reported confidence** (`<confidence>` tag) — easily gamed

### The Key Insight
Hallucinations aren't random. When a model hallucinates, the internal *reasoning chain* is structurally different:
- A factually grounded answer shows **tight attention back-references** — token 47 strongly attends to token 12 because it is logically building on it.
- A hallucinated passage shows **loose, diffuse, or self-referential attention** — the model is generating plausible-sounding text without anchoring to prior reasoning.

### Why Graphs?
Traditional approaches flatten all token signals into a single pooled vector (mean pooling). This **destroys the relational structure** — the *who attends to whom* information.

A Graph Neural Network operates natively on this structure. Message passing lets each token node aggregate information from its logical neighbours, exactly mirroring how the attention mechanism works.

---

## Slide 3 — The Dataset Sources

### File: `tokenizer/datasets/benchmark_loader.py`

Two benchmark datasets are supported. Both are stored under `tokenizer/data/llmsknow/`.

---

### Dataset 1: AnswerableMath
- **Source:** LLMsKnow benchmark (`AnswerableMath_test.csv`)
- **Format:** `question`, `answer` (answer is a Python list string like `"['42']"`)
- **Sample ID Pattern:** `math_test_0`, `math_test_1`, ...
- **Labelling Logic:** String inclusion — does the gold number appear in the model's `<answer>` tag?
- **Use case:** Tests whether the model *knows* a mathematical fact vs. confabulates a plausible-looking number.

```python
ex = BenchmarkExample(
    sample_id="math_test_7",
    dataset_name="AnswerableMath",
    split="test",
    prompt="What is the product of 17 and 6?",
    gold_answer="102"
)
```

---

### Dataset 2: NaturalQuestions (NQ)
- **Source:** LLMsKnow NQ subset (`nq_wc_dataset_test.csv`)
- **Format:** `Question`, `Answer`, optional `Context`
- **Sample ID Pattern:** `nq_test_0`, `nq_test_1`, ...
- **Labelling Logic:** String inclusion of gold entity in extracted answer
- **Use case:** Factual entity retrieval. The optional `Context` column enables Lookback-Lens ratio calculation, measuring how much the model attends *back to the context* vs. generating from parametric memory.

```python
ex = BenchmarkExample(
    sample_id="nq_test_42",
    dataset_name="NaturalQuestions",
    split="test",
    prompt="Who invented the telephone?",
    gold_answer="Alexander Graham Bell",
    context="In 1876, Alexander Graham Bell was awarded a patent..."  # optional
)
```

---

### The `BenchmarkExample` Dataclass
```python
@dataclass
class BenchmarkExample:
    sample_id: str        # Unique ID for provenance tracking
    dataset_name: str     # "AnswerableMath" or "NaturalQuestions"
    split: str            # "test", "train", or "custom"
    prompt: str           # The raw question text
    gold_answer: str      # The ground truth answer string
    context: Optional[str] = None  # RAG/Lookback context block
```

---

## Slide 4 — The Grammar Fence: Forcing Structured Output

### File: `tokenizer/grammar.py`

This is one of the most critical components. Raw LLM generation is unpredictable in format. We need the model to produce *exactly* three structured sections so we can reliably extract signal.

### The Outlines Library
We use `outlines.from_transformers()` to wrap the Hugging Face model. Outlines intercepts the token sampling process and applies a **Deterministic Finite Automaton (DFA)** compiled from a regex. At every decoding step, it masks the logits so that the model can *only* generate tokens that advance the DFA toward a valid final state.

This is a **hard structural guarantee** — not a soft prompt instruction.

---

### The Regex Pattern (The Strict Fence)
```regex
<think>\n[^<]+</think>\n<answer>\n[^<]+</answer>\n<confidence>\n(?:0\.\d+|1\.0)\n</confidence>
```

**Breaking it down:**
| Segment | Meaning |
|---|---|
| `<think>\n` | Literal opening tag + newline |
| `[^<]+` | One or more characters that are NOT `<` — prevents early tag escape |
| `</think>\n<answer>\n` | Mandatory structural delimiter |
| `[^<]+` | The answer content — again no `<` allowed |
| `</answer>\n<confidence>\n` | Mandatory delimiter |
| `(?:0\.\d+\|1\.0)` | A decimal between 0.00 and 1.0 — enforced numerically |
| `\n</confidence>` | Closing tag |

**Why `[^<]+`?** This is the key insight. The moment the model generates a `<` character, the DFA can only route to the next valid tag. This prevents the model from opening a new `<think>` block or any other tag inside a content section, making the output completely deterministic in structure.

---

### The Generator Function
```python
def build_generator(model_name="Qwen/Qwen3.5-0.8B"):
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        attn_implementation="eager"   # CRITICAL: eager mode exposes attention weights
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = outlines.from_transformers(hf_model, tokenizer)
    output_type = Regex(regex_pattern)

    def generator(prompt, max_new_tokens=8192, **kwargs):
        return model(prompt, output_type, max_new_tokens=max_new_tokens, **kwargs)

    return hf_model, generator, tokenizer
```

**Critical detail:** `attn_implementation="eager"` forces PyTorch's standard attention implementation instead of Flash Attention or SDPA. Only the eager implementation returns the full attention weight matrix that our hooks need to intercept.

---

## Slide 5 — The Prompt Builder & System Instructions

### File: `tokenizer/collectors/collect_benchmark.py` — `main()` loop

Before generation, each `BenchmarkExample` is formatted into a structured chat prompt using the model's tokenizer chat template.

### System Prompt (The Instruction Contract)
```
You are a logical reasoning assistant. You must rigorously follow this format:
<think>
[Your step-by-step reasoning]
</think>
<answer>
[Your final short answer]
</answer>
<confidence>
[A number between 0.0 and 1.0]
</confidence>
```

This system prompt is paired with the grammar fence. The system prompt tells the model *what to do*, and the grammar fence *enforces* it at the token level regardless of what the model "wants" to generate.

---

### Prompt Assembly
```python
user_content = ex.prompt
if ex.context:
    # RAG/Lookback task: inject context block before the question
    user_content = f"Context: {ex.context}\n\nQuestion: {ex.prompt}"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": user_content}
]

# tokenizer.apply_chat_template adds <|im_start|>, <|im_end|> etc.
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

### Measuring Prompt Length
```python
prompt_token_count = len(tokenizer.encode(prompt))
extractor.set_context_length(prompt_token_count)
stopper.configure_sample(prompt_token_count)
```

This `prompt_token_count` is the **boundary** that separates "prompt tokens" from "generated tokens" in all downstream calculations. It is passed to both the extractor (for Lookback-Lens math) and the stopper (for max token budget tracking).

---

## Slide 6 — The StopOnTag: Real-Time Control During Generation

### File: `tokenizer/collectors/collect_benchmark.py` — `StopOnTag` class

The `StopOnTag` class implements HuggingFace's `StoppingCriteria` interface. It is called after every single token is generated.

### Three Stop Conditions

**1. Grammar Tag Detected (`stop_tag` = `"</confidence>"`):**
```python
tail_tokens = input_ids[0][-15:]
tail_text = self.tokenizer.decode(tail_tokens)
if self.stop_tag in tail_text:
    self.stop_reason = "stop_tag"
    return True
```
Decodes the last 15 tokens and checks if the closing tag appeared. Stops immediately.

**2. Manual Skip Signal (UI Control):**
```python
def _consume_skip_signal(self):
    with open(self.control_path, "r") as handle:
        data = json.load(handle)
    if data.get("skip_current", False):
        data["skip_current"] = False  # reset atomically
        # write back
        return True
```
Reads `live_viewer/runtime/control.json`. The live viewer's "Skip" button writes `{"skip_current": true}` to this file. The stopper checks it every token and consumes the signal.

**3. Max Generated Tokens:**
```python
generated_ids = input_ids[0][self.prompt_token_count:]
if len(generated_ids) >= self.max_generated_tokens:
    self.stop_reason = "max_generated_tokens"
    return True
```

**Live streaming side effect:**
```python
if self.reporter is not None:
    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
    self.reporter.update_generated_text(generated_text)
```
Every token triggers a live text update event, enabling real-time streaming in the viewer.

---

## Slide 7 — The TraceExtractor: Surgical Hook Attachment

### File: `tokenizer/extractors/trace_extractor.py`

The `TraceExtractor` is the core signal extraction engine. It uses PyTorch's **forward hook** system to intercept internal computations without modifying the model's source code.

### Hook Attachment
```python
def attach_hooks(self, model, layer_idx=-1):
    # Target: the LAST transformer decoder layer
    target_layer = model.model.layers[layer_idx]

    # Hook 1: Fires on the SELF-ATTENTION sub-module
    self._handles.append(
        target_layer.self_attn.register_forward_hook(self.attention_hook)
    )

    # Hook 2: Fires on the ENTIRE LAYER (after FFN)
    self._handles.append(
        target_layer.register_forward_hook(self.activation_hook)
    )
```

**Why the last layer?**
The last transformer layer produces the richest, most semantically compressed representation. Early layers capture surface syntax; the final layer captures high-level meaning and factual associations — exactly what we want for hallucination detection.

**Execution order within one forward pass (one token generated):**
```
Forward pass starts
  ↓
attention_hook fires  ← captures attention weights, stores as _pending_*
  ↓
FFN sub-layer runs
  ↓
activation_hook fires ← captures hidden state, commits pending data to buffers
```

This two-hook design ensures that the attention edges and the hidden state for the *same token* are always stored together.

---

## Slide 8 — The Attention Hook: Building the Sparse Edge List

### `TraceExtractor.attention_hook()`

This hook intercepts the raw attention weight tensor from the self-attention sub-module.

### Step-by-Step Breakdown

**Step 1: Extract the attention weight matrix**
```python
attn_weights = output[1].detach()  # Shape: [batch, heads, q_len, kv_len]
attn_matrix = attn_weights[0]       # Drop batch dim → [heads, q_len, kv_len]
```

**Step 2: Average across all attention heads**
```python
avg_attn = attn_matrix.mean(dim=0).squeeze()  # → [q_len, kv_len]
```
We average over all heads to get a single consensus attention distribution. This reduces noise from individual specialized heads.

**Step 3: Extract the current token's attention vector**
```python
attn_vector = avg_attn[-1]  # The very last query position → [kv_len]
```
During autoregressive decoding, `q_len=1` for every decode step. We take position -1 which is the single new token's attention over all previous tokens.

**Step 4: Separate prompt attention from generated attention**
```python
generated_attn = attn_vector[self.context_length:]     # Attention to prior generated tokens
prior_generated_attn = generated_attn[:current_step]   # Up to (not including) this token
```

**Step 5: Apply the sparsity threshold to build edges**
```python
threshold = 0.05  # Only keep edges with attention weight > 5%
indices = (prior_generated_attn > self.threshold).nonzero(as_tuple=False).view(-1)
for idx in indices:
    step_edges.append((current_step, idx.item(), prior_generated_attn[idx].item()))
    #                  ^source token  ^target token  ^edge weight
```

**Step 6: Compute the Lookback-Lens Ratio**
```python
attn_on_context   = attn_vector[:self.context_length].sum()
attn_on_generated = generated_attn.sum()
total_attn = attn_on_context + attn_on_generated
ratio = (attn_on_context / total_attn).item()
```

**What is the Lookback Ratio?**
- **High ratio (→ 1.0):** The model is strongly attending back to the input context/prompt. This is characteristic of faithful, grounded answers.
- **Low ratio (→ 0.0):** The model is almost entirely attending to its own previously generated tokens. This is characteristic of self-referential, potentially hallucinated generation.

All results are stored as `_pending_*` attributes, to be committed by the activation hook.

---

## Slide 9 — The Activation Hook: Capturing Hidden States

### `TraceExtractor.activation_hook()`

This hook fires on the full decoder layer *after* both self-attention and the feed-forward network have run. It captures the residual stream at its deepest point.

### Step-by-Step Breakdown

**Step 1: Extract the hidden state tensor**
```python
hidden_states = output[0] if isinstance(output, tuple) else output
```

**Step 2: Handle dimensionality (Prefill vs. Decode)**
```python
if hidden_states.dim() == 3:
    # Prefill mode: [batch, seq_len, hidden_dim]
    latest_token_activation = hidden_states[0, -1, :].detach().cpu()
elif hidden_states.dim() == 2:
    # Decode mode: [batch_or_seq, hidden_dim]
    latest_token_activation = hidden_states[-1, :].detach().cpu()
```
During the prefill (prompt processing), the model processes all tokens at once → 3D tensor. During autoregressive decoding, one token at a time → 2D tensor. The hook handles both cases.

Result: a single `[1024]`-dimensional vector — the "embedding" for this generated token.

**Step 3: Commit all pending data to the buffers atomically**
```python
self.activations.append(latest_token_activation)   # [1024] vector
self.sparse_edges.append(self._pending_edges)       # list of (src, tgt, weight) tuples
if self._pending_lookback_ratio is not None:
    self.lookback_ratios.append(self._pending_lookback_ratio)  # float
```

**Step 4: Report to Live Viewer**
```python
if self.reporter is not None:
    self.reporter.add_graph_step(
        step_index=self._pending_step,
        edges=self._pending_edges,
        lookback_ratio=self._pending_lookback_ratio,
    )
```

**Step 5: Reset pending state**
```python
self._pending_step = None
self._pending_edges = []
self._pending_lookback_ratio = None
```

After a full generation, the buffers contain:
- `extractor.activations` → `N` tensors of shape `[1024]` (one per generated token)
- `extractor.sparse_edges` → `N` edge lists (one per step)
- `extractor.lookback_ratios` → `N` floats

---

## Slide 10 — The Auto-Labeler: The Correctness Oracle

### File: `tokenizer/labels/auto_correctness.py`

After generation completes, we need a ground-truth label for training. The `AutoLabeler` class uses a **heuristic string-matching oracle** based on the LLMsKnow methodology.

### Step 1: Extract the Answer from Tags
```python
@staticmethod
def extract_answer_tag(generated_text: str) -> str:
    match = re.search(
        r"<answer>\n(.*?)\n</answer>",
        generated_text,
        flags=re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return "NO_ANSWER_TAG_FOUND"
```

Because we use the grammar fence, this regex **always** matches. The fence guarantees the tag structure.

### Step 2: Dataset-Specific Evaluation
```python
def get_labels(self, generated_text, gold_answer, dataset_name):
    extracted_answer = self.extract_answer_tag(generated_text)

    if "Math" in dataset_name:
        # Is the gold number (e.g. "102") present anywhere in the answer?
        is_correct = str(gold_answer).lower() in extracted_answer.lower()
    elif "Natural" in dataset_name:
        # Is the gold entity (e.g. "Alexander Graham Bell") present?
        is_correct = str(gold_answer).lower() in extracted_answer.lower()
    else:
        is_correct = gold_answer.lower() in extracted_answer.lower()

    correctness = 1 if is_correct else 0
    return {"correctness": correctness, "exact_answer_extracted": extracted_answer}
```

### Label Interpretation
| `correctness` value | Meaning | GNN Training Target |
|---|---|---|
| `1` | Model answered correctly | `y = 0` (not hallucinating) |
| `0` | Model answered incorrectly | `y = 1` (hallucinating) |

The label inversion happens in the GNN loader: if `correctness == 1` → `label = 0`; if `correctness == 0` → `label = 1`.

---

## Slide 11 — The Unified Schema: Packaging the Trace

### File: `tokenizer/storage/schema.py`

Before saving, all data for one generation is packaged into a single `UnifiedTraceRecord` dataclass.

```python
@dataclass
class UnifiedTraceRecord:
    # --- Text Provenance ---
    sample_id: str               # "math_test_7"
    source_dataset: str          # "AnswerableMath"
    prompt: str                  # Raw question text
    context: str                 # RAG context (empty string if none)
    gold_answer: str             # Ground truth
    generated_text: str          # Full model output including all tags
    exact_answer_extracted: str  # Content inside <answer>...</answer>

    # --- Tiered Labels ---
    labels: Dict[str, int]       # {"gold_correctness": 1} or {"gold_correctness": 0}
                                 # Future: {"human_hallucination": 1}

    # --- Internal Signals ---
    activations: List[torch.Tensor]   # N × [1024] tensors
    sparse_edges: List[list]          # N × [(src, tgt, weight), ...]
    lookback_ratios: List[float]      # N floats

    # --- Metadata ---
    metadata: Dict[str, Any]     # {"prompt_tokens": 247}
```

This schema is deliberately **tiered**:
- `gold_correctness` is cheap and automatic (always available)
- `human_hallucination` is expensive (manual annotation, optional)
- The GNN trainer prefers human labels when present, falls back to gold correctness

---

## Slide 12 — Saving as a PyG Graph File

### File: `tokenizer/storage/sample_writer.py` — `TraceDatasetManager`

The `UnifiedTraceRecord` is converted into a **PyTorch Geometric `Data` object** and saved to disk.

### Step 1: Build the Node Feature Matrix
```python
x = torch.stack(record.activations)
# Shape: [N, 1024]  where N = number of generated tokens
```
Each row is the 1024-dim hidden state of one generated token. This is the `x` matrix that the GNN reads as node features.

### Step 2: Build the Edge Index and Edge Attributes
```python
source_nodes, target_nodes, edge_weights = [], [], []

for step_edges in record.sparse_edges:
    for (src, tgt, weight) in step_edges:
        source_nodes.append(src)
        target_nodes.append(tgt)
        edge_weights.append([weight])

edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
# Shape: [2, E]  — standard COO sparse format for PyG

edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
# Shape: [E, 1]  — attention probability as edge weight
```

### Step 3: Lookback Ratios as Auxiliary Node Features
```python
lookback_tensor = torch.tensor(record.lookback_ratios, dtype=torch.float32).unsqueeze(1)
# Shape: [N, 1]
```

### Step 4: Assemble and Save the PyG Data Object
```python
graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
graph_data.lookback_ratio = lookback_tensor   # [N, 1]
graph_data.y_correctness  = torch.tensor([record.labels.get('gold_correctness', -1)], dtype=torch.float32)
graph_data.y_human        = torch.tensor([record.labels.get('human_hallucination', -1)], dtype=torch.float32)
graph_data.text_meta = {
    "sample_id": record.sample_id,
    "source_dataset": record.source_dataset,
    "gold_answer": record.gold_answer,
    "exact_answer_extracted": record.exact_answer_extracted,
    "generated_text": record.generated_text
}

filepath = f"charm_unified_dataset/sample_{counter:06d}.pt"
torch.save(graph_data, filepath)
```

**What one saved file contains:**
```
sample_000007.pt
├── x              → [N, 1024]  token hidden states
├── edge_index     → [2, E]     attention graph connectivity
├── edge_attr      → [E, 1]     attention weights
├── lookback_ratio → [N, 1]     context attention ratios
├── y_correctness  → [1]        0.0 or 1.0
├── y_human        → [1]        0.0, 1.0, or -1.0 (unlabelled)
└── text_meta      → dict       provenance strings
```

---

## Slide 13 — [LIVE VIEWER DEMO] The GraphGuard Live Viewer

### File: `tokenizer/live_viewer/server.py` + `tokenizer/live_viewer/static/index.html`

**➡️ DEMO: Open browser to `http://127.0.0.1:8765`**

The live viewer is a self-contained web application served by a Python `ThreadingHTTPServer`. It provides a real-time window into the collection pipeline while it runs.

### Architecture Overview
```
Browser (index.html)
    │  polls /api/state every 500ms
    │  polls /api/events?after=N every 300ms
    ↕
Python HTTP Server (server.py) — port 8765
    │  serves static/index.html for GET /
    │  reads runtime/state.json   for GET /api/state
    │  reads runtime/events.jsonl for GET /api/events
    │  writes runtime/control.json for POST /api/skip
    │
CollectorProcessManager
    │  spawns collect_benchmark.py as subprocess
    │  captures stdout/stderr logs in real-time
    ↕
collect_benchmark.py (child process)
    │  writes to runtime/state.json   (current state)
    │  appends to runtime/events.jsonl (event stream)
    │  reads runtime/control.json     (skip signal)
```

### Runtime Files (the IPC layer)
| File | Purpose | Writer | Reader |
|---|---|---|---|
| `state.json` | Full current state snapshot | collector | browser |
| `events.jsonl` | Append-only event log | collector | browser |
| `control.json` | Skip/stop signals | browser | collector |

### What You'll See On Screen
- **Run header:** model name, dataset, sample count, elapsed time
- **Live text panel:** streaming token-by-token generation
- **3D Graph panel:** nodes appearing as tokens are generated, edges forming between them
- **Metrics bar:** node count, edge count, latest lookback ratio
- **Sample history:** previously completed samples with correctness badges

---

## Slide 14 — [LIVE VIEWER DEMO] The Event Bus: Real-Time Communication

### File: `tokenizer/live_viewer/event_bus.py`

The `LiveEventBus` and `LiveDemoReporter` classes form the communication bridge between the running collector and the browser.

### The Event Bus (`LiveEventBus`)
Thread-safe, filesystem-based event bus using atomic file writes:

```python
class LiveEventBus:
    def emit(self, event_type, payload, seq):
        event = {
            "seq": seq,          # Monotonically increasing integer
            "type": event_type,  # String event name
            "ts": time.time(),   # Unix timestamp
            "payload": payload   # Arbitrary JSON-serializable dict
        }
        # Appends one JSON line to events.jsonl (with lock)
        with open(self.events_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def update_state(self, state):
        # Atomic write via temp file + os.replace()
        temp_path = f"{self.state_path}.tmp"
        with open(temp_path, "w") as f:
            json.dump(state, f)
        os.replace(temp_path, self.state_path)
```

**Atomic writes prevent the browser from reading a half-written file.**

### The Reporter (`LiveDemoReporter`)
Higher-level API that wraps the bus and maintains state:

| Method | Event Emitted | When Called |
|---|---|---|
| `start_run()` | `run_started` | Before the first sample |
| `start_sample()` | `sample_started` | Before each generation |
| `update_generated_text()` | `text_updated` | Every token (via StopOnTag) |
| `add_graph_step()` | `graph_step` | Every token (via activation hook) |
| `mark_sample_saved()` | `sample_saved` | After successful save |
| `mark_sample_failed()` | `sample_failed` | On error or skip |
| `finish_run()` | `run_finished` | After all samples |

**➡️ DEMO POINT: Watch the event_seq counter increment in the browser's network tab as tokens generate.**

---

## Slide 15 — [LIVE VIEWER DEMO] Watching a Generation in Real Time

**➡️ DEMO: Click "Start Collection" and watch a sample run**

### What happens millisecond-by-millisecond:

**T=0ms** — Browser sends `POST /api/start`
- `CollectorProcessManager` spawns `collect_benchmark.py` as subprocess
- Runtime files are reset to empty state

**T=~500ms** — Collector loads the model (Qwen 3.5-0.8B)
- `build_generator()` loads weights, wraps with Outlines, attaches hooks

**T=~1s** — `run_started` event appears in events.jsonl
- Browser receives it on next poll and renders the run header

**T=~1.5s** — First sample begins, `sample_started` event
- Browser shows the prompt text in the left panel

**T=~2s onwards** — Each generated token:
1. `attention_hook` fires → sparse edges computed → `_pending_*` set
2. `activation_hook` fires → hidden state captured → `add_graph_step()` called
3. `StopOnTag.__call__()` runs → `update_generated_text()` called
4. Reporter emits `graph_step` and `text_updated` events
5. Browser polls, receives new events, updates graph and text panel

**Visible in browser:**
- Text panel scrolls with new tokens appearing character by character
- 3D graph: new node appears (a sphere labeled with token step index), then edges form from it to prior tokens it attends to
- Lookback ratio gauge updates — watch it rise when the model cites the context

**T=end** — `</confidence>` detected → `sample_saved` event → correctness badge appears

---

## Slide 16 — [LIVE VIEWER DEMO] The 3D Force Graph Visualisation

### File: `tokenizer/live_viewer/static/index.html`

**➡️ DEMO: Interact with the 3D graph during generation**

The graph is rendered using **3D Force Graph** (three.js based) which applies a physics simulation to lay out nodes in 3D space.

### Node Representation
Each sphere in the 3D graph is one **generated token**.
- **Position:** Determined by the force simulation (nodes connected by edges cluster together)
- **Label:** The step index of that token (e.g., "g0", "g1", "g42")
- **Group:** All generated tokens share the "generated" group (teal colour)

### Edge Representation
Each line between spheres is a **sparse attention link**.
- **Direction:** From the source token (current step) to the target token (prior step it attends to)
- **Weight:** The raw attention probability (0.05 to 1.0)
- **Thickness/opacity:** Proportional to weight — stronger attention = thicker line

### Edge Compression
The reporter only displays the **top 6 edges per step** (by attention weight):
```python
ranked = sorted(edges, key=lambda item: item[2], reverse=True)[:6]
```
This keeps the graph readable. The full edge set is stored in the `.pt` file.

### What to Look For
- **Dense, interconnected clusters** → the model is reasoning linearly, each token strongly references recent predecessors. Often seen in correct arithmetic.
- **Sparse, long-range edges** → the model is referencing much earlier tokens (good for structured recall) or has very diffuse attention (potential hallucination signal).
- **A node with no incoming edges** → this token was generated without strongly attending to any prior generated token — it may be a "free-floating" hallucinated claim.

---

## Slide 17 — [LIVE VIEWER DEMO] Custom Prompt Entry

### File: `tokenizer/live_viewer/server.py` — `POST /api/start-custom`

**➡️ DEMO: Enter a custom prompt in the UI and run it live**

The live viewer supports entering an arbitrary prompt without needing a benchmark CSV file. This is critical for interactive demonstration and debugging.

### API Endpoint
```
POST /api/start-custom
Content-Type: application/json

{
    "prompt":      "What is the capital of Australia?",
    "context":     "",
    "gold_answer": "Canberra",
    "sample_id":   "custom_demo_01",
    "dataset_name": "CustomUserPrompt"
}
```

### What Happens Internally
The server passes these values as CLI flags to the `collect_benchmark.py` subprocess:
```python
command = [sys.executable, "collectors/collect_benchmark.py",
    "--custom-prompt", payload["prompt"],
    "--custom-context", payload.get("context", ""),
    "--custom-gold-answer", payload.get("gold_answer", ""),
    "--custom-sample-id", payload.get("sample_id", "custom_0"),
    "--custom-dataset-name", payload.get("dataset_name", "CustomUserPrompt")
]
```

Inside the collector, `maybe_build_custom_example()` detects these flags and constructs a `BenchmarkExample` with `split="custom"`, bypassing all CSV loading entirely.

### Demo Suggestions
1. Ask a question with a known correct answer → verify high lookback ratio, sparse graph
2. Ask an obscure/ambiguous question → watch for dense self-referential attention
3. Provide misleading context → watch the lookback ratio tell you the model is ignoring context

---

## Slide 18 — [LIVE VIEWER DEMO] Sample History & Skip Control

### File: `tokenizer/live_viewer/event_bus.py` — `_archive_current_sample()`

**➡️ DEMO: Show the history panel and demonstrate the skip button**

### History Panel
After each sample completes (saved or failed), the reporter calls `_archive_current_sample()` which performs a deep copy of the current sample into `state["history"]`. The browser renders these as scrollable cards below the live graph.

Each history card shows:
- **Sample ID** and dataset source
- **Prompt text** (truncated)
- **Generated answer** extracted from tags
- **Gold answer** for comparison
- **Correctness badge:** ✅ Correct (green) or ❌ Incorrect (red)
- **Graph metrics:** Number of nodes, edges, mean lookback ratio
- **The full mini-graph** (optional expanded view)

### Skip Control
The "Skip Current" button in the UI:
1. Browser sends `POST /api/skip`
2. Server calls `PROCESS_MANAGER.skip_current()`
3. Writes `{"skip_current": true}` to `runtime/control.json`
4. Collector's `StopOnTag._consume_skip_signal()` detects this on next token check
5. Returns `True` → generation halts immediately
6. Reporter marks sample as `failed` with reason `"Skipped from frontend control."`
7. Control file is reset to `{"skip_current": false}` atomically

This allows interactive curation of the dataset: run many samples automatically but skip any that are clearly broken or too long.

### Stop Control
`POST /api/stop` sends `SIGTERM` to the collector subprocess via `process.terminate()`. The model is mid-generation, so the next token check exits cleanly and hooks are removed to prevent memory leaks.

---

## Slide 19 — Loading the PyG Dataset for Training

### File: `tokenizer/trainers/gnn_trainer.py` — `load_graph_dataset()`

Once enough `.pt` files have been collected, training begins. The loader performs several important normalization and validation steps.

### File Discovery
```python
valid_files = [f for f in os.listdir(dataset_dir) if f.endswith(".pt")]
```

### For Each File: Load, Cast, and Validate

**Step 1: Load and cast dtype**
```python
data = torch.load(filepath, weights_only=False)
data.x = data.x.to(torch.float32)          # From BFloat16 → Float32
data.edge_attr = data.edge_attr.to(torch.float32)
```
The Qwen model runs in BFloat16 for efficiency. Training needs Float32 for numerical stability.

**Step 2: Prune ghost edges (critical bug fix)**
```python
num_actual_nodes = data.x.size(0)
data.num_nodes = num_actual_nodes

# Some edge indices may reference token positions that were pruned
valid_edge_mask = (
    (data.edge_index[0] < num_actual_nodes) &
    (data.edge_index[1] < num_actual_nodes)
)
data.edge_index = data.edge_index[:, valid_edge_mask]
data.edge_attr  = data.edge_attr[valid_edge_mask]
```
Ghost edges occur when the extractor captures an edge step index that is greater than the number of activations (can happen if generation terminates mid-step). This pruning step removes invalid edges.

**Step 3: Extract the training label**
```python
if getattr(data, 'y_human', None) is not None and data.y_human.item() != -1:
    label = int(data.y_human.item())           # Prefer human label
elif getattr(data, 'y_correctness', None) is not None:
    correctness = int(data.y_correctness.item())
    label = 0 if correctness == 1 else 1       # Invert: correct→faithful, wrong→hallucinated
else:
    continue                                    # Skip unlabelled samples

data.y = torch.tensor([label], dtype=torch.float32)
```

---

## Slide 20 — The CHARMCritic GNN Architecture

### File: `tokenizer/trainers/gnn_trainer.py` — `CHARMCritic`

This is the neural network that learns to predict hallucination risk from the attention graph.

```
Input Graph:  x=[N, 1024],  edge_index=[2, E],  edge_attr=[E, 1]
                    │
         ┌──────────▼──────────┐
         │  GATConv Layer 1    │
         │  in=1024, out=256   │
         │  heads=4, concat=F  │  ← 4 attention heads, averaged
         │  + ReLU             │
         └──────────┬──────────┘
                    │  [N, 256]
         ┌──────────▼──────────┐
         │  GATConv Layer 2    │
         │  in=256,  out=256   │
         │  heads=4, concat=F  │
         │  + ReLU             │
         └──────────┬──────────┘
                    │  [N, 256]
         ┌──────────▼──────────┐
         │  global_mean_pool   │  ← Collapse all N nodes → 1 graph vector
         └──────────┬──────────┘
                    │  [1, 256]  (per graph in batch)
         ┌──────────▼──────────┐
         │  Linear(256 → 64)   │
         │  + ReLU             │
         └──────────┬──────────┘
                    │  [1, 64]
         ┌──────────▼──────────┐
         │  Linear(64 → 1)     │
         │  + Sigmoid          │  ← Output in [0.0, 1.0]
         └──────────┬──────────┘
                    │
             hallucination_score ∈ [0.0, 1.0]
```

### Why Graph Attention (GAT)?
Standard Graph Convolutional Networks (GCN) treat all neighbours equally. GAT learns *different attention weights for different edges*, allowing it to focus on the most semantically significant attention links and ignore noisy low-weight edges.

With `heads=4, concat=False`, each layer uses 4 independent attention heads and then **averages** their outputs (rather than concatenating). This gives a 256-dim output from a 256-dim input per layer, keeping dimensions stable.

---

## Slide 21 — Message Passing: What the GNN Actually Does

### Understanding the Forward Pass

At a conceptual level, the GNN is learning to propagate information along the attention graph, exactly mirroring how the model's own attention worked during generation.

### Pass 1: `conv1 = GATConv(1024 → 256)`

For each token node `i`:
```
For each neighbour j that token i attended to:
    Compute attention score α_ij using learned weights
    Weight j's feature vector by α_ij
Sum weighted neighbour features → new 256-dim representation for node i
```

After this pass, each node knows about its **direct attention neighbourhood**. A token that was strongly referenced by many other tokens will have accumulated information from all of them.

### Pass 2: `conv2 = GATConv(256 → 256)`

The second pass operates on the 256-dim representations from pass 1. Now each node knows about its **2-hop neighbourhood** — neighbours of neighbours. This allows the GNN to detect:
- Chains of reasoning (A → B → C → D all strongly connected)
- Dead-end nodes (generate without back-reference)
- Isolated clusters (possibly hallucinated sub-passages)

### Global Readout: `global_mean_pool`
```python
x = global_mean_pool(x, batch)
# All N node vectors are averaged → [batch_size, 256]
```
This collapses the entire graph into a single vector representing the *overall reasoning structure* of the generation. This single vector goes through the classification head to produce the final score.

---

## Slide 22 — Training the GNN

### File: `tokenizer/trainers/gnn_trainer.py` — `train_gnn()`

### Training Configuration
| Parameter | Value | Rationale |
|---|---|---|
| Train/Test split | 80% / 20% | Standard ML split |
| Batch size | 4 graphs | Small batches due to variable graph sizes |
| Epochs | 20 | Sufficient for convergence on small datasets |
| Loss function | `BCELoss` | Binary Cross Entropy for binary classification |
| Optimizer | `Adam` | Adaptive learning rate |
| Learning rate | `0.001` | Standard Adam default |
| Weight decay | `1e-4` | L2 regularization against overfitting |

### The Training Loop
```python
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch).squeeze(-1)    # Forward pass → [batch_size]
        loss = criterion(out, batch.y)    # BCELoss

        loss.backward()                   # Backprop through GATConv layers
        optimizer.step()

        preds = (out > 0.5).float()       # Threshold at 0.5
```

### Evaluation
```python
model.eval()
with torch.no_grad():
    for batch in test_loader:
        out = model(batch).squeeze(-1)
        preds = (out > 0.5).float()
        correct_test += int((preds == batch.y).sum())
```

### Saving Weights
```python
save_path = "tokenizer/trainers/weights/charm_critic_v1.pth"
torch.save(model.state_dict(), save_path)
```

---

## Slide 23 — The Baseline: Why Graph Structure Matters

### File: `tokenizer/trainers/baseline_trainer.py`

The baseline trainer deliberately **ignores the graph structure** to establish a performance floor. This demonstrates what value the edges add.

### Baseline Approach: Mean Pooling → Logistic Regression

```python
# For each graph file:
node_features = graph_data.x              # [N, 1024] — hidden states
pooled_vector = torch.mean(node_features, dim=0).numpy()  # [1024] — simple average
```

By averaging all token hidden states, we get a single 1024-dimensional "bag of tokens" representation. This completely discards:
- Which tokens attended to which (edge_index)
- How strongly they attended (edge_attr)
- The sequential ordering implied by the graph structure
- The lookback ratio per token

A Logistic Regression classifier is then trained on these pooled vectors:
```python
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)
```

### Why This Matters
| Method | Captures | Misses |
|---|---|---|
| Mean Pooling + LogReg | Average token semantics | All relational structure |
| CHARMCritic (GNN) | Token semantics + attention graph + message passing | Nothing structural |

**Expected result:** The GNN should outperform the baseline specifically on samples where the hallucination signal lies in the *structure* of the reasoning chain rather than the *semantics* of individual tokens. A model can use semantically coherent words in a hallucinated passage, but its attention graph will be structurally different.

---

## Slide 24 — The Standalone Inference UI

### File: `tokenizer/standalone_ui/api.py`

The standalone UI provides a production-ready inference endpoint: given any prompt, it runs the full pipeline and returns the hallucination score alongside a token-level breakdown.

### Startup: Loading Both Models
```python
@app.on_event("startup")
def load_models():
    # 1. Load the LLM with grammar fence + hooks
    llm_model, llm_generator, llm_tokenizer = build_generator("Qwen/Qwen3.5-0.8B")
    llm_model.config.output_attentions = True

    # 2. Attach the extractor
    extractor = TraceExtractor(threshold=0.05)
    extractor.attach_hooks(llm_model)

    # 3. Load the trained GNN weights
    gnn_model = CHARMCritic(node_dim=1024, hidden_dim=256).to(device)
    gnn_model.load_state_dict(torch.load("trainers/weights/charm_critic_v1.pth"))
    gnn_model.eval()
```

### The `/api/analyze` Endpoint

**Input:** `{"prompt": "Who painted the Mona Lisa?"}`

**Step 1: Generate with hooks active**
```python
output_ids = llm_model.generate(
    prompt_ids,
    max_new_tokens=512,
    stopping_criteria=halt_state,
    output_attentions=True,
    return_dict_in_generate=True
)
new_token_ids = output_ids.sequences[0][prompt_token_count:]
real_tokens = [llm_tokenizer.decode([tid]) for tid in new_token_ids]
```

**Step 2: Assemble the PyG graph from extractor buffers**
```python
x = torch.stack(extractor.activations).to(torch.float32)  # [N, 1024]
# ... build edge_index, edge_attr from extractor.sparse_edges ...
graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)
```

**Step 3: GNN inference**
```python
with torch.no_grad():
    response_score = gnn_model(graph_data).item()  # Single float [0.0, 1.0]
```

**Step 4: Per-token risk scores**
```python
# Blend local attention density with global response score
token_scores = [0.0] * num_nodes
for tgt in target_nodes:
    token_scores[tgt] += 0.05  # Attention density: more neighbours = higher local risk
token_scores = [min(1.0, (s * 0.5) + (response_score * 0.5)) for s in token_scores]
```

**Output:**
```json
{
    "tokens": ["Leonardo", " da", " Vinci", ...],
    "nodes": [{"id": 0, "label": "Leo...", "x": 60, "y": 60}, ...],
    "edges": [{"s": 5, "t": 2, "weight": "0.12"}, ...],
    "tokenScores": [0.05, 0.08, 0.03, ...],
    "responseScore": 0.12
}
```

---

## Slide 25 — End-to-End Data Flow Summary

### Complete Pipeline, Component by Component

```
[1] DATASET INPUT
    CSV file → BenchmarkLoader → BenchmarkExample(sample_id, prompt, gold_answer, context)

[2] PROMPT ENGINEERING
    System prompt + user content → tokenizer.apply_chat_template()
    → full_prompt string + prompt_token_count

[3] OUTLINES GRAMMAR FENCE
    Regex DFA compiled → logit masking per token
    → enforces <think>...</think><answer>...</answer><confidence>...</confidence>

[4] GENERATION (Qwen 3.5-0.8B)
    Each new token:
    ├── attention_hook fires on self_attn
    │   ├── average heads → [kv_len] attention vector
    │   ├── threshold edges → sparse edge list
    │   ├── compute lookback ratio (context_attn / total_attn)
    │   └── store as _pending_*
    ├── activation_hook fires on full layer
    │   ├── extract [1024] hidden state
    │   ├── commit to activations[], sparse_edges[], lookback_ratios[]
    │   └── emit graph_step event to LiveDemoReporter
    └── StopOnTag fires
        ├── stream text update to LiveDemoReporter
        └── check </confidence> tag → stop if found

[5] AUTO-LABELLING
    generated_text → extract <answer> tag → string match vs gold_answer
    → correctness: 0 or 1

[6] PACKAGING
    UnifiedTraceRecord → TraceDatasetManager.save_unified_trace()
    → PyG Data(x, edge_index, edge_attr, lookback_ratio, y_correctness, y_human)
    → sample_000007.pt

[7] GNN TRAINING (offline)
    Load all .pt files → prune ghost edges → label extraction
    CHARMCritic: GATConv(1024→256) → GATConv(256→256) → global_mean_pool → Linear → Sigmoid
    BCELoss + Adam → charm_critic_v1.pth

[8] INFERENCE
    New prompt → generate + extract → assemble PyG graph → gnn_model.forward()
    → response_score + per-token hallucination risk
```

---

## Slide 26 — Key Design Decisions & Why They Matter

### 1. Why Outlines + DFA instead of prompt engineering?
Prompt engineering can suggest structure but cannot *guarantee* it. If the model generates `<answer>` inside a `<think>` block, the regex extractor breaks. The DFA eliminates this at the token-sampling level — it is a mathematical guarantee, not a suggestion.

### 2. Why `attn_implementation="eager"`?
Flash Attention and SDPA (Scaled Dot Product Attention, PyTorch's fused kernel) do not return the intermediate attention weight matrix to Python — they fuse it inside CUDA for speed. The `eager` implementation runs standard matrix multiplications that expose the full `[batch, heads, q_len, kv_len]` tensor our hooks need.

### 3. Why forward hooks instead of model surgery?
Hooks leave the model's forward pass completely unmodified. We can attach them at runtime, extract what we need, and remove them with `remove_hooks()` without any change to the model's parameters or outputs.

### 4. Why the last transformer layer?
The final layer's hidden states are the most semantically rich. Early layers focus on syntax and surface patterns. The last layer has seen all prior transformations and encodes high-level semantic judgements — which is where factual confabulation manifests.

### 5. Why sparse edges (threshold > 5%)?
The full attention matrix has `N×N` entries, many of which are near-zero noise. Thresholding at 5% keeps only the attention links that are actually driving the generation. This keeps graph size manageable and focuses the GNN on meaningful structure.

### 6. Why global_mean_pool instead of global_max_pool or CLS token?
Mean pooling produces a graph-level representation that considers all token representations equally. Max pooling would overweight a single extreme token. There is no CLS token in decoder-only LLMs. Mean pooling is robust and standard for graph classification.

### 7. Why a tiered label schema (y_human vs y_correctness)?
Automatic correctness labels are cheap but noisy — a model can be factually wrong in a new way not caught by string matching. Human annotations are expensive but precise. The tiered schema allows starting with automatic labels, then iteratively improving with human corrections on ambiguous cases.

---

## Slide 27 — Interpreting the GNN Output

### What the Score Means

| Score Range | Interpretation | Recommended Action |
|---|---|---|
| 0.0 – 0.2 | Low hallucination risk. Reasoning graph is coherent and well-connected. | Accept response with high confidence |
| 0.2 – 0.4 | Mild uncertainty. Some structurally weak regions but generally grounded. | Review the answer, spot-check key claims |
| 0.4 – 0.6 | Moderate risk. Graph shows mixed structure — some grounded, some diffuse. | Flag for human review, do not use in critical contexts |
| 0.6 – 0.8 | High hallucination risk. Attention graph shows significant self-referential loops. | Regenerate or reject |
| 0.8 – 1.0 | Very high risk. Structurally pathological graph — model was confabulating. | Hard reject |

### Per-Token Scores
The `tokenScores` array from the standalone API maps each generated token to a local risk score. Tokens with high local scores are:
- **Heavily referenced by others** (many incoming attention edges) — could be a load-bearing hallucinated fact
- **Blended with the high global score** — contextualised by the overall judgement

### The Lookback Ratio as a Secondary Signal
The lookback ratios stored in each `.pt` file can be used as an interpretability signal:
- A sudden **drop in lookback ratio** mid-generation often corresponds to the model transitioning from context-grounded reasoning to parametric memory generation
- The exact token position of this drop can pinpoint *where* the hallucination began

---

## Slide 28 — File Structure Reference

```
tokenizer/
├── grammar.py                    ← Outlines regex fence, model loading
│
├── datasets/
│   └── benchmark_loader.py       ← CSV → BenchmarkExample
│
├── collectors/
│   └── collect_benchmark.py      ← Main orchestration loop, StopOnTag
│
├── extractors/
│   └── trace_extractor.py        ← attention_hook, activation_hook
│
├── labels/
│   └── auto_correctness.py       ← String match oracle
│
├── storage/
│   ├── schema.py                 ← UnifiedTraceRecord dataclass
│   └── sample_writer.py          ← → PyG Data → .pt file
│
├── live_viewer/
│   ├── event_bus.py              ← LiveEventBus, LiveDemoReporter
│   ├── server.py                 ← HTTP server, CollectorProcessManager
│   └── runtime/
│       ├── state.json            ← Current state snapshot
│       ├── events.jsonl          ← Append-only event log
│       └── control.json          ← Skip/stop signals
│   └── static/
│       └── index.html            ← 3D Force Graph viewer (Three.js)
│
├── trainers/
│   ├── gnn_trainer.py            ← CHARMCritic GNN, training loop
│   ├── baseline_trainer.py       ← Mean pool + Logistic Regression
│   └── weights/
│       └── charm_critic_v1.pth   ← Saved GNN weights
│
├── standalone_ui/
│   ├── api.py                    ← FastAPI inference endpoint
│   └── frontend/                 ← React UI (separate build)
│
├── charm_unified_dataset/
│   └── sample_000XXX.pt          ← PyG graphs (one per generation)
│
└── data/
    └── llmsknow/
        ├── AnswerableMath_test.csv
        └── nq_wc_dataset_test.csv
```

---

## Slide 29 — Running the Full Pipeline

### Step 1: Collect Data (with Live Viewer)
```bash
# Terminal 1: Start the live viewer server
cd tokenizer
python live_viewer/server.py --port 8765

# Open browser: http://127.0.0.1:8765
# Click "Start Collection" OR "Start Custom"

# OR: Run collection directly from CLI
python collectors/collect_benchmark.py --dataset math --limit 50
python collectors/collect_benchmark.py --dataset nq --limit 50
```

### Step 2: Train the GNN
```bash
# Requires at least 10 labelled samples in charm_unified_dataset/
python trainers/gnn_trainer.py

# Outputs: trainers/weights/charm_critic_v1.pth
# Prints per-epoch: Loss | Train Acc | Test Acc
```

### Step 3: Run the Baseline for Comparison
```bash
python trainers/baseline_trainer.py

# Prints: Test Accuracy + full classification report
# Compare this to GNN accuracy to quantify graph structure value
```

### Step 4: Run Standalone Inference
```bash
cd standalone_ui
uvicorn api:app --host 0.0.0.0 --port 8000

# POST /api/analyze with {"prompt": "..."}
curl -X POST http://localhost:8000/api/analyze \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is the boiling point of water?"}'
```

---

## Slide 30 — Summary & Key Takeaways

### What We Built

**GraphGuard / CHARM** is a complete hallucination detection pipeline that operates *inside* the language model, not on its output text.

### The Three Core Innovations

**1. Structural Signal Extraction**
By hooking into the last transformer layer's attention mechanism and residual stream, we capture a *relational* representation of how the model reasoned — something no text-only system can access.

**2. Grammar-Enforced Structured Output**
The Outlines DFA fence makes the output format a mathematical guarantee rather than a prompt engineering hope. This enables reliable automatic labelling at scale.

**3. Graph Neural Network on the Reasoning Graph**
By treating each generation as a directed weighted graph (tokens as nodes, attention as edges), we give the GNN access to the *topology* of reasoning. Hallucinated generations have structurally different topologies that the GNN learns to recognise.

### What Makes This Different from Existing Approaches
| Approach | Signal Source | Structural? | Real-Time? |
|---|---|---|---|
| NLI-based detectors | Output text | ✗ | ✗ |
| Self-consistency sampling | Multiple outputs | ✗ | ✗ |
| Logistic Regression (our baseline) | Mean-pooled hidden states | ✗ | ✓ |
| **CHARM (GNN)** | **Attention graph + hidden states** | **✓** | **✓** |

### The Live Viewer
The live viewer makes the invisible visible — you can watch, in real time, as the model builds its attention graph token by token. This is not just a monitoring tool; it is an interpretability window into the mechanism of hallucination as it happens.