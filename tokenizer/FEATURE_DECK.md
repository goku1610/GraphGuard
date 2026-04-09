# GraphGuard — How Features Detect Hallucination
### From Raw LLM Signals → Attributed Graph → GNN Verdict → Live Visual Evidence

---

## Slide 1 — The Core Idea

A language model's **internal computation** during text generation leaves traces that reveal whether it is grounding its output in real knowledge or fabricating. We capture three classes of trace signal, encode them into a single graph, and train a GNN to read that graph as a hallucination detector.

| Signal class | What it captures | Where it lives in the graph |
|---|---|---|
| **Hidden-state activations** | The model's semantic representation of each token | Node features (`x`, 1024 dims) |
| **Lookback ratio** | How much attention each token pays to prompt context vs prior response tokens | Node features (concatenated, +1 dim) |
| **Sparse attention edges + weights** | Which tokens strongly attend to which, and how strongly | Graph topology (`edge_index`) + edge features (`edge_attr`, 1 dim) |

---

## Slide 2 — Signal 1: Hidden-State Activations (Node Features)

### What they are
At each decode step, the last transformer layer produces a 1024-dimensional vector — the model's internal "thought" about the token it just generated. This vector encodes:
- Semantic meaning of the token in context
- The cumulative effect of all prior layers' transformations
- Implicit confidence and factual grounding signals

### How they help detect hallucination
When a model hallucinates, its hidden states drift into regions of activation space that differ from grounded generation. Correct answers tend to cluster; fabricated answers scatter. By presenting these vectors as node features, the GNN can learn to recognise hallucination-correlated activation patterns — not just for individual tokens, but in relation to their graph neighbourhood.

### Where this appears in the live viewer
Each **blue sphere** in the 3D graph represents one generated token. Although you cannot see the 1024-dim vector directly, the GNN consumes it internally. The viewer shows the *consequence*: nodes whose activations contribute to a high hallucination score will eventually be flagged in the standalone UI's per-token risk heatmap.

---

## Slide 3 — Signal 2: Lookback Ratio (Context-Grounding Feature)

### What it is
For each generated token, we compute:

```
lookback_ratio = attention_on_prompt / (attention_on_prompt + attention_on_response)
```

This is a single scalar in [0, 1] measuring what fraction of the model's attention is directed back at the original prompt/context versus at its own previously generated tokens.

### How it helps detect hallucination
This is the core intuition behind Lookback Lens (Chuang et al., 2024). When a model is grounded, it frequently looks back at the context for facts. When it hallucinates, it becomes self-referential — recycling its own output instead of checking the source material.

A **sudden drop** in lookback ratio during generation is a strong signal that the model has transitioned from retrieval-grounded reasoning to parametric confabulation.

### Where this appears in the live viewer

**1. The Lookback Ratio bar (left panel)**
A gradient bar that fills in real-time. Green = high grounding, yellow/red = drifting away from context. Updates on every decode step.

**2. The Lookback vs Decode Step line chart (centre panel, below graph)**
A Chart.js time series that plots lookback % for every decode step. You can visually identify:
- Stable high ratio = grounded generation
- Gradual decline = model slowly drifting
- Sharp cliff = abrupt transition to hallucination

The chart supports PNG export and JSON data export for post-hoc analysis.

### How it enters the GNN
Previously stored but **unused**. Now concatenated as the 1025th dimension of each node's feature vector, so the GNN sees both the semantic embedding and the context-grounding signal together.

---

## Slide 4 — Signal 3: Sparse Attention Edges + Weights (Graph Structure + Edge Features)

### What they are
At each decode step, the attention hook captures the full attention vector from the newly generated token to all prior tokens. We:
1. Average across attention heads → single [kv_len] vector
2. Threshold at τ=0.05 (keep only links with >5% attention)
3. Store each surviving link as a directed edge (current_step → prior_step, weight)

### How the graph structure helps detect hallucination
The *topology* of the attention graph differs structurally between faithful and hallucinated generations:

| Pattern | Faithful generation | Hallucinated generation |
|---|---|---|
| Connectivity | Rich back-connections to diverse prior tokens | Sparse, local connections (self-referential loops) |
| Degree distribution | Broad — many tokens are attended to | Narrow — few tokens receive most attention |
| Path length | Short paths back to early (grounded) tokens | Long or missing paths to context-adjacent tokens |
| Clustering | Distributed clusters | Tight, isolated cliques |

The GNN's message-passing mechanism propagates information along these edges — it literally "reads" the attention structure the model created.

### How edge weights help (the `edge_dim=1` upgrade)
Previously, the attention weights were stored in `edge_attr` but **silently ignored** by GATConv (because `edge_dim` was not set). Now:

```
GATConv(..., edge_dim=1)
```

Inside each GATConv layer, the edge weight is:
1. Projected via a learned linear layer (`lin_edge`)
2. Combined with node-derived attention coefficients
3. Passed through LeakyReLU + softmax

This means the GNN doesn't treat all edges equally — it amplifies messages along high-attention edges and suppresses weak ones. Two tokens might both be connected, but the one with 0.4 attention weight sends a much stronger message than the one at 0.06.

### Where this appears in the live viewer

**1. The 3D Force-Directed Graph (centre panel)**
- Each blue sphere = a generated token node
- Each red directed edge = a surviving attention link above the 5% threshold
- Orange animated particles flow along edges showing attention direction
- The yellow sphere = the most recently generated token
- Nodes are positioned in a 3D helix, with step number determining position

**2. The Step Timeline (right panel)**
Every `graph_step` event logs: `Step X added Y edges | Lookback Z%`. You can watch:
- Steps with **many edges** = the model is broadly attending (grounded)
- Steps with **zero edges** = the model isn't strongly attending to anything prior (suspicious)
- Steps where lookback drops while edges thin out = likely hallucination onset

---

## Slide 5 — The Final Training Data (What the Graph Actually Looks Like)

Each `.pt` file saved by the pipeline is a PyTorch Geometric `Data` object:

```
sample_000042.pt
├── x              → [N, 1024]   raw hidden-state activations
├── edge_index     → [2, E]      directed attention graph (src → tgt)
├── edge_attr      → [E, 1]      attention weight per edge
├── lookback_ratio → [N, 1]      per-token context grounding score
├── y_correctness  → [1]         auto-labelled: 1=correct, 0=incorrect
└── y_human        → [1]         human label (if available): 1=hallucinated, 0=faithful
```

At training time, the data loader enriches this:

**att+act mode** (default):
```
data.x = cat(activations, lookback_ratio) → [N, 1025]
```

**att mode** (ablation):
```
data.x = lookback_ratio → [N, 1]
```

Edge features (`edge_attr`) remain `[E, 1]` in both modes.

The label `y` is derived from `y_human` (preferred) or `y_correctness` (fallback), inverted so that `1 = hallucinated, 0 = faithful`.

---

## Slide 6 — How the GNN Combines These Features

```
Input:  x=[N, node_dim]  edge_index=[2, E]  edge_attr=[E, 1]
                    │
         ┌──────────▼──────────────┐
         │  GATConv Layer 1        │
         │  in=node_dim, out=256   │
         │  heads=4, edge_dim=1    │  ← edge weights modulate attention
         │  + ReLU                 │
         └──────────┬──────────────┘
                    │  [N, 256]
         ┌──────────▼──────────────┐
         │  GATConv Layer 2        │
         │  in=256,  out=256       │
         │  heads=4, edge_dim=1    │
         │  + ReLU                 │
         └──────────┬──────────────┘
                    │  [N, 256]
         ┌──────────▼──────────────┐
         │  global_mean_pool       │  ← all N node embeddings → 1 vector
         └──────────┬──────────────┘
                    │  [1, 256]
         ┌──────────▼──────────────┐
         │  Linear(256→64) + ReLU  │
         │  Linear(64→1) + Sigmoid │
         └──────────┬──────────────┘
                    │
             hallucination_score ∈ [0.0, 1.0]
```

### What each layer does with the features

**Layer 1 (GATConv):**
- For each node i, computes learned attention α_ij over neighbours j
- α_ij incorporates: node i's projection, node j's projection, AND edge weight (i,j)
- Message from j to i = α_ij × transformed(x_j)
- Output: 256-dim embedding per token that encodes 1-hop neighbourhood

**Layer 2 (GATConv):**
- Same mechanism on the 256-dim embeddings from Layer 1
- Now each node sees its 2-hop neighbourhood
- Can detect: "my neighbour attended strongly to a token that itself had low lookback"

**Global Mean Pool:**
- Averages all N node embeddings → single graph-level vector
- Every token contributes equally to the final representation

**Prediction Head:**
- Two dense layers compress 256 → 64 → 1
- Sigmoid squashes to [0, 1] = hallucination probability

---

## Slide 7 — The Two Training Modes (Ablation)

| Mode | Command | Node dim | What it tests |
|---|---|---|---|
| `att` | `python trainers/gnn_trainer.py --mode att` | 1 | Can graph structure + edge weights + lookback alone detect hallucination? |
| `att+act` | `python trainers/gnn_trainer.py --mode att+act` | 1025 | Does adding the full semantic embedding improve detection? |

### Why this ablation matters

If **att mode performs well**, it proves the CHARM paper's thesis: the attention graph's **topology and edge weights** carry hallucination signal independent of token semantics. The GNN is reading the "shape of reasoning" rather than the "content of reasoning."

If **att+act significantly outperforms att**, it means semantic information (the 1024-dim activation) is doing the heavy lifting. The graph structure is helpful but secondary.

If **both perform similarly**, it suggests the signals are redundant for your dataset — the model's attention patterns already encode enough semantic information.

Each mode saves separate weights:
- `weights/charm_critic_att_v1.pth`
- `weights/charm_critic_att_act_v1.pth`

---

## Slide 8 — Reading Hallucination in the Live Viewer

The live viewer at `http://localhost:8765` provides three simultaneous visual channels that map directly to the GNN's input features.

### Visual Channel 1: The 3D Reasoning Graph
**What you see:** Nodes (tokens) arranged in a 3D helix. Edges (attention links) drawn as red lines with animated orange particles.

**What to look for:**
- **Dense, well-connected graph** → model is cross-referencing prior tokens → likely grounded
- **Sparse graph with isolated clusters** → model is self-referential → likely hallucinating
- **Long chains with no branching** → sequential copying without verification → suspicious

### Visual Channel 2: The Lookback Ratio Chart
**What you see:** A line chart plotting lookback % over decode steps.

**What to look for:**
- **Stable line at 60-80%** → model consistently checking context → grounded
- **Declining slope** → model gradually drifting from context → confidence should decrease
- **Sharp cliff (e.g., 70% → 20% in 5 steps)** → abrupt switch to hallucination mode
- **Low flat line from the start** → model never grounded on context → likely parametric hallucination

### Visual Channel 3: The Step Timeline + Metrics
**What you see:** A log of `Step X added Y edges | Lookback Z%` entries, plus live node/edge counts.

**What to look for:**
- Steps that add 0 edges while lookback drops → the token is being generated in isolation
- Sudden bursts of many edges → model re-engaging with context (possible self-correction)
- Node count vs edge count ratio → dense graphs (high edge/node) are more grounded

### Putting it all together
When the 3D graph goes sparse, the lookback chart drops, and the step timeline shows "0 edges" entries simultaneously — that's the visual signature of hallucination onset. The GNN sees the exact same pattern in its feature matrix: low lookback in node features, missing edges in the adjacency, and weak weights on surviving edges.

---

## Slide 9 — From Visual Evidence to GNN Score

### The connection is direct:

| What you see in the viewer | What the GNN receives | How it influences the score |
|---|---|---|
| Dense graph with many edges | Rich `edge_index`, many message-passing paths | More information flow → informed prediction |
| High lookback bar / stable chart | High values in the 1025th dim of `x` | Node features signal grounding |
| Thick animated particles (strong edges) | High values in `edge_attr` | GATConv amplifies these messages |
| Sparse graph, isolated nodes | Few edges, limited message passing | Nodes rely on their own features → less context |
| Dropping lookback chart | Low/declining 1025th dim across nodes | Node features signal drift from context |
| Step timeline: "0 edges" | Missing rows in `edge_index` | Token embedding is not updated by neighbours |

The live viewer is not just a monitoring tool — it is an **interpretability window** into the exact features the GNN is learning from. Every visual pattern you can identify by eye is a pattern the GNN can learn to detect automatically.

---

## Slide 10 — Summary: Three Features, One Graph, One Score

```
LLM generates token
       │
       ├── activation_hook → 1024-dim hidden state ──┐
       │                                              │
       ├── attention_hook → sparse edges + weights ───┤──→ PyG Data object ──→ CHARMCritic GNN
       │                  → lookback ratio ───────────┘         │
       │                                                         ▼
       │                                              hallucination_score ∈ [0, 1]
       │
       └── live_viewer ← graph_step event
                        ← text_updated event
                        ← lookback_ratio
```

The three signals are complementary:
- **Activations** tell you *what* the model is thinking
- **Lookback ratio** tells you *where* it is looking
- **Attention edges + weights** tell you *how* it is connecting its thoughts

The GNN fuses all three through learned message passing. The live viewer shows all three in real time. Together, they make the invisible mechanism of hallucination visible and measurable.

---

## Slide 11 — How the Dataset Is Generated (End-to-End)

The entire training corpus is produced by a single automated pipeline. No manual graph construction, no external annotation services — just one command.

### Step 1: Source benchmarks (the questions)

Six benchmark CSVs live under `tokenizer/data/llmsknow/`:

| Dataset | Task type | Example question | Gold answer |
|---|---|---|---|
| AnswerableMath | Arithmetic reasoning | "What is 127 × 43?" | "5461" |
| NaturalQuestions | Factual QA (with context) | "Who painted the Mona Lisa?" | "Leonardo da Vinci" |
| MovieQA | Factual knowledge recall | "Who directed Inception?" | "Christopher Nolan" |
| MNLI | Natural language inference | "Does premise entail hypothesis?" | "entailment" |
| WinoGrande | Commonsense reasoning | "The trophy doesn't fit the suitcase because it is too ___" | "large" |
| WinoBias | Coreference + bias detection | "The nurse helped the doctor because [they]..." | pronoun target |

`BenchmarkLoader` reads each CSV and produces a standardised `BenchmarkExample(sample_id, prompt, gold_answer, context)`.

### Step 2: Constrained generation (the LLM answers)

For each example, Qwen 3.5-0.8B generates a response under a **DFA-enforced regex fence** (via Outlines):

```
<think>
[step-by-step reasoning]
</think>
<answer>
[final short answer]
</answer>
<confidence>
[0.0 to 1.0]
</confidence>
```

The regex is compiled into a finite-state machine at the token-sampling level, so the model **cannot** produce malformed output. This structure is essential because it forces the model to expose its chain-of-thought before committing to an answer — giving the attention graph a longer, richer sequence to analyse.

The model is loaded with `attn_implementation="eager"` so PyTorch exposes the full `[batch, heads, q_len, kv_len]` attention matrix (Flash Attention fuses it away).

### Step 3: Trace extraction (while the LLM generates)

Two forward hooks fire on every decode step, attached to the last transformer layer:

**attention_hook** — fires on `self_attn`:
1. Reads the full attention matrix `[heads, q_len, kv_len]`
2. Averages across heads → `[kv_len]`
3. Thresholds at τ=0.05 → sparse edge list `[(src, tgt, weight), ...]`
4. Computes lookback_ratio = `attention_on_prompt / total_attention`

**activation_hook** — fires on the full layer output:
1. Extracts the `[1024]` hidden-state vector for the latest generated token
2. Commits the activation, edges, and lookback ratio to the extractor's buffers
3. Emits a `graph_step` event to the live viewer

After generation completes, the extractor holds:
- `activations` → N tensors of shape `[1024]`
- `sparse_edges` → N lists of `(src, tgt, weight)` tuples
- `lookback_ratios` → N floats

### Step 4: Auto-labelling (is the answer correct?)

`AutoLabeler` extracts the text inside the `<answer>` tags and compares it to the gold answer using a dataset-appropriate heuristic:
- **Math**: is the gold number anywhere in the extracted answer?
- **NQ / factual**: case-insensitive string inclusion
- **Fallback**: same inclusion check

Result: `correctness = 1` (correct) or `0` (incorrect). Incorrect answers are treated as hallucinations during training (label is inverted: `y = 1 - correctness`).

Optionally, `y_human` can override this with a manual annotation via the human feedback interface.

### Step 5: Packaging into a PyG graph

`TraceDatasetManager.save_unified_trace()` converts the raw buffers into a PyTorch Geometric `Data` object:

```
x              = stack(activations)           → [N, 1024]
edge_index     = tensor([sources, targets])   → [2, E]
edge_attr      = tensor(weights)              → [E, 1]
lookback_ratio = tensor(ratios).unsqueeze(1)  → [N, 1]
y_correctness  = tensor([correctness])        → [1]
y_human        = tensor([-1])                 → [1]  (or manual label if available)
```

Saved as `charm_unified_dataset/sample_000042.pt`.

### The full loop in one command

```bash
cd tokenizer
python collectors/collect_benchmark.py --dataset all --limit 50
```

This generates up to 50 samples from each of the 6 benchmarks (300 total), each with full traces, auto-labels, and packaged as `.pt` graphs ready for training.

---

## Slide 12 — How the GNN Actually Detects Hallucination (Concise Walkthrough)

Here is the exact path from a `.pt` file to a hallucination score, step by step.

### 1. Load and enrich

```
Raw .pt file:  x=[N, 1024]  lookback_ratio=[N, 1]  edge_attr=[E, 1]
                        ↓ concatenate
Enriched:      x=[N, 1025]  (1024 activation dims + 1 lookback dim)
```

The data loader also prunes invalid edges and derives the binary label from `y_human` or `y_correctness`.

### 2. First message-passing layer

For each token node `i`, GATConv computes:
- A **learned attention coefficient** α\_ij for every neighbour `j`, incorporating:
  - Node i's projected features (its activation + lookback)
  - Node j's projected features
  - The **edge weight** between them (how strongly i attended to j during generation)
- A **weighted message** from each neighbour: `msg_j = α_ij × Linear(x_j)`
- The **sum** of all incoming messages becomes the new 256-dim embedding for node i

**What this captures:** each token now "knows" about its direct attention neighbourhood. A token that attended broadly to many grounded tokens gets a rich, contextualised embedding. A token that attended to nothing (or only to itself) gets a sparse, uninformed embedding.

### 3. Second message-passing layer

Same operation, but on the 256-dim embeddings from Layer 1. Now each token sees its **2-hop neighbourhood** — neighbours of neighbours. This lets the GNN detect compound patterns like:
- "Token A attended to Token B, which itself had very low lookback" → transitive hallucination signal
- "Token C attended to many tokens that all attended to the same anchor" → grounded consensus

### 4. Global mean pool

All N node embeddings are averaged into a single 256-dim vector representing the **entire generation** as one point in embedding space.

### 5. Prediction head

Two linear layers compress 256 → 64 → 1, followed by sigmoid:
- Output ∈ [0.0, 1.0]
- Close to 0 → the graph looks like faithful generation
- Close to 1 → the graph looks like hallucinated generation

### 6. Training signal

The model is trained with BCE loss against the binary label (hallucinated or not). Over many examples, the GATConv layers learn which combinations of:
- activation patterns (what),
- lookback values (where), and
- edge weight distributions (how)

...are predictive of hallucination.

---

## Slide 13 — How This Gets You a Hallucination Score (the Easiest Way)

### For batch analysis (offline)

```bash
# 1. Collect data (one command)
python collectors/collect_benchmark.py --dataset all --limit 50

# 2. Train the GNN (one command)
python trainers/gnn_trainer.py --mode att+act

# 3. Done. Weights saved to trainers/weights/charm_critic_att_act_v1.pth
```

Every `.pt` file in `charm_unified_dataset/` now has a correctness label, and the trained GNN can score any new graph.

### For live single-prompt analysis (standalone API)

```bash
# Start the inference server
python standalone_ui/api.py
```

Then send any prompt:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Who painted the Mona Lisa?"}'
```

Response:

```json
{
  "tokens": ["<think>", "\n", "The", " Mona", " Lisa", ...],
  "responseScore": 0.12,
  "tokenScores": [0.05, 0.08, 0.03, ...],
  "nodes": [...],
  "edges": [...]
}
```

`responseScore` is the GNN's hallucination probability. That's it — one number, one API call.

### For real-time visual monitoring (live viewer)

```bash
# Terminal 1: start the viewer
python live_viewer/server.py

# Terminal 2: run collection (or click "Start Live Run" in the UI)
python collectors/collect_benchmark.py --dataset math --limit 10
```

Open `http://localhost:8765`. Watch:
- The 3D graph build token by token
- The lookback chart plot in real time
- The step timeline log each edge addition
- The correctness label appear when generation finishes

### Summary: three ways to get hallucination scores

| Method | Input | Output | Effort |
|---|---|---|---|
| **Batch collection + training** | Benchmark CSVs | Trained `.pth` weights | One-time setup |
| **Standalone API** | Any text prompt | `responseScore` + `tokenScores` JSON | Single HTTP call |
| **Live viewer** | Any prompt (typed or benchmark) | Real-time 3D graph + lookback chart + step log | Open browser |

All three use the same underlying pipeline: extract traces → build graph → run GNN (or visualise the raw features). The GNN sees exactly what the live viewer shows — the viewer is the human-readable version of the GNN's input.
