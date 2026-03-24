# Combined Approach for `tokenizer`

## Goal

Yes, these approaches can be combined, but they should not be merged by copying scripts directly into one file. The clean approach is to turn `tokenizer` into a unified data collection and training pipeline with multiple signal sources:

- graph traces from the current `tokenizer`
- automatic correctness labels and benchmark datasets from `LLMsKnow`
- attention-based lookback features and span-style faithfulness labels from `Lookback-Lens`

The result should be one collector that can build richer examples for a future critic.

## Concrete Merged Pipeline

This is the cleanest way to think about the fusion:

1. load benchmark tasks and gold answers from `LLMsKnow`
2. wrap each example in the strict `tokenizer` generation format
3. run local Qwen generation
4. extract graph traces from `tokenizer`
5. extract lookback ratios from `Lookback-Lens` style attention dynamics
6. compute automatic labels
7. package everything into one canonical multimodal `.pt` example
8. train fair baselines and the final CHARM-style GNN on the same stored data

The key idea is that `tokenizer` becomes the central runtime, while `LLMsKnow` contributes scalable benchmark supervision and `Lookback-Lens` contributes contextual attention features.

### 1. Input Logic

What changes:

- remove the dependency on manual question entry as the only input path
- keep interactive mode as an optional source, not the primary source

How it merges:

- load benchmark datasets from `LLMsKnow/data`
- iterate automatically over tasks such as math, natural questions, and other benchmark-style QA/classification datasets
- for context-grounded tasks, also load document-style inputs similar to `Lookback-Lens`
- convert every example into the same prompt contract used by `tokenizer`
- force the model to answer in:
  - `<think>`
  - `<answer>`
  - `<confidence>`

Result:

- the bottleneck moves from human typing to automatic benchmark traversal
- each benchmark example becomes one candidate multimodal training sample

### 2. Extraction Logic

What changes:

- the hook system stops being graph-only
- it becomes a multi-signal extractor

How it merges:

- keep the current node features from `tokenizer`
  - deep hidden states per generation step
- keep the current edge construction from `tokenizer`
  - sparse attention graph using thresholded attention mass
- add `Lookback-Lens` style attention dynamics
  - for tasks with a source context, split attention into:
    - attention pointing to prompt/context tokens
    - attention pointing to newly generated tokens
  - compute a lookback ratio for each layer, head, and decoding step
- optionally keep token-level score or confidence signals from generation

Result:

- each sample contains both sparse graph structure and dense temporal attention statistics

### 3. Supervision Logic

What changes:

- the system no longer depends only on manual `y/n` grading
- labels become tiered and source-aware

How it merges:

- Tier 1: factual correctness
  - reuse `LLMsKnow`-style matching against benchmark gold answers
  - parse the text inside `<answer>` and compare it to the gold answer
  - save as `y_correct`
- Tier 2: contextual faithfulness
  - for summarization or context-grounded tasks, evaluate whether the answer is supported by the provided context
  - save as `y_faithful`
- Tier 3: span-level localization
  - identify which answer tokens are unsupported
  - save a token mask or aligned span list
- Tier 4: optional human label
  - keep manual grading as a high-quality override or audit source
  - save as `y_human`

Important rule:

- do not collapse all labels into one boolean during collection
- keep correctness, faithfulness, and human judgment separate in storage

Result:

- cheap labels from benchmarks scale data generation
- contextual labels improve hallucination specificity
- manual labels remain available for high-quality evaluation

### 4. Storage Logic

What changes:

- stop saving only a minimal PyG graph
- save one canonical multimodal example object

Each saved example should include:

- text fields
  - prompt
  - context
  - gold answer
  - model answer
- graph fields
  - node features
  - sparse edge index
  - edge attributes
- dense signals
  - lookback ratios
  - token scores
  - confidence values
- labels
  - `y_correct`
  - `y_faithful`
  - `y_human`
  - span mask
- metadata
  - model name
  - dataset name
  - split
  - generation settings
  - sample source

Result:

- one `.pt` file becomes a full multimodal training example rather than only a graph snapshot

### 5. Training Logic

What changes:

- the saved schema enables direct apples-to-apples comparison between detectors

Train these models on the same stored examples:

- Baseline probe
  - hidden states only
  - mimics `LLMsKnow`
- Baseline ratios
  - lookback-ratio features only
  - mimics `Lookback-Lens`
- CHARM GNN
  - sparse graph plus optional auxiliary features
  - your main architecture
- optional fusion model
  - combines GNN embedding, lookback features, and probe score

Result:

- the dataset design itself creates fair fights between baselines and the graph critic

### Summary Architecture Flow

- load question or context-grounded sample from benchmark data
- generate constrained response with `tokenizer`
- extract hidden states and sparse attention graph
- compute lookback ratios when source context exists
- evaluate automatic correctness and faithfulness
- package everything into one canonical multimodal example
- save to disk for downstream baseline and GNN training

## What Is Present Right Now

### `tokenizer/`

Current purpose: interactive trace collection for graph-based hallucination data.

Contents:

- `main.py`
  - interactive loop
  - asks a question
  - generates a constrained answer with `<think>`, `<answer>`, `<confidence>`
  - asks the human for `y/n`
- `extractor.py`
  - forward hooks
  - captures last-layer activations
  - captures sparse attention edges above threshold
- `dataset_manager.py`
  - converts captured traces into `torch_geometric.data.Data`
  - saves `trace_*.pt`
- `grammar.py`
  - loads Qwen
  - builds the constrained generator
- `charm_dataset/`
  - currently contains `trace_0.pt`
- `venv/`
  - local Python virtual environment
- `__pycache__/`
  - Python bytecode cache

Current data product:

- node features from hidden activations
- sparse edges from attention
- one human hallucination label per saved graph

### `LLMsKnow/`

Current purpose: benchmark-driven probing of whether the model internally represents correctness.

Root contents:

- `README.md`
- `requirements.txt`
- `LICENSE`
- `data/`
- `src/`

`data/` contents:

- `AnswerableMath.csv`
- `AnswerableMath_test.csv`
- `mnli_train.csv`
- `mnli_validation.csv`
- `movie_qa_test.csv`
- `movie_qa_train.csv`
- `nq_wc_dataset.csv`
- `winobias_dev.csv`
- `winobias_test.csv`
- `winogrande_test.csv`
- `winogrande_train.csv`

`src/` contents:

- `generate_model_answers.py`
  - runs the model on benchmark questions
  - saves generations, token ids, and token scores
- `compute_correctness.py`
  - computes automatic correctness labels from gold answers
- `extract_exact_answer.py`
  - extracts the short answer span from a longer response
- `probing_utils.py`
  - loads models
  - traces internal reps using `TraceDict`
  - extracts chosen layer/token features
- `probe.py`
  - trains logistic probes on internal representations
- `probe_all_layers_and_tokens.py`
  - explores layer/token heatmaps
- `probe_choose_answer.py`
  - selects among candidate answers using a saved probe
- `probe_type_of_error.py`
  - probes error subtypes
- `resampling.py`
  - generates multiple sampled answers per prompt
- `resamples_utils.py`
  - helper utilities for resampling
- `resampling_merge_runs.py`
  - merges parallel resampling outputs
- `logprob_detection.py`
  - baseline using token log probabilities
- `p_true_detection.py`
  - baseline using truth-probability style prompting

Current data product:

- benchmark prompts and gold answers
- model generations
- token ids and score tensors
- automatic correctness labels
- hidden-state probe features

### `Lookback-Lens/`

Current purpose: attention-only hallucination detection and decoding guidance for context-faithfulness tasks.

Root contents:

- `README.md`
- `requirements.txt`
- `step01_extract_attns.py`
- `step02_eval_gpt4o.py`
- `step03_lookback_lens.py`
- `step04_run_decoding.py`
- `generation.py`
- `eval_exact_match.py`
- `lookback_lens_demo.ipynb`
- `lookback-lens.png`
- `data/`
- `classifiers/`
- `transformers-4.32.0/`

`data/` contents:

- `cnndm-1000.jsonl`
- `nq-open-10_total_documents_gold_at_4.jsonl.gz`
- `xsum-1000.jsonl`

`classifiers/` contents:

- `classifier_anno-cnndm-7b_predefined_span.pkl`
- `classifier_anno-cnndm-7b_sliding_window_8.pkl`
- `classifier_anno-nq-7b_predefined_span.pkl`
- `classifier_anno-nq-7b_sliding_window_8.pkl`

Current data product:

- lookback-ratio tensors derived from attention maps
- GPT-4o faithfulness labels
- problematic spans
- lightweight logistic classifiers

## How The Three Approaches Differ

### `tokenizer`

- strongest structural representation
- human-in-the-loop labels
- graph-ready for PyG
- low scale, high label quality

### `LLMsKnow`

- benchmark-driven
- automatic correctness labels
- probe-friendly hidden-state features
- high scale, weaker label semantics for contextual hallucination

### `Lookback-Lens`

- attention-only
- very good for context-faithfulness
- supports span-level supervision
- simple classifiers and decoding guidance

## Recommended Unified Design

The best approach is to keep `tokenizer` as the central package and absorb the useful ideas as modular subsystems.

### 1. Define One Canonical Example Schema

Every collected sample should map to one standard record before saving.

Suggested logical schema:

- `sample_id`
- `source`
  - `interactive`
  - `llmsknow`
  - `lookback`
- `task_type`
  - `qa`
  - `summarization`
  - `classification`
  - `reasoning`
- `prompt`
- `context`
- `gold_answer`
- `model_answer`
- `exact_answer`
- `labels`
  - `human_hallucination`
  - `gold_correctness`
  - `judge_faithfulness`
  - `span_labels`
- `signals`
  - `hidden_states`
  - `last_layer_trace`
  - `attention_sparse_edges`
  - `lookback_ratio`
  - `token_scores`
  - `confidence_text`
- `graph`
  - `x`
  - `edge_index`
  - `edge_attr`
- `metadata`
  - model name
  - dataset name
  - split
  - generation params

This is the key step. Without a canonical schema, the three sources stay incompatible.

### 2. Split `tokenizer` Into Clear Modules

Recommended structure:

```text
tokenizer/
  main.py
  collectors/
    interactive.py
    llmsknow_adapter.py
    lookback_adapter.py
  extractors/
    hidden_states.py
    sparse_attention.py
    lookback_ratio.py
    token_scores.py
  labels/
    human_labeler.py
    gold_correctness.py
    judge_labeler.py
    span_alignment.py
  storage/
    schema.py
    graph_builder.py
    save_sample.py
  training/
    train_gnn.py
    train_probe_baseline.py
    train_lookback_baseline.py
    train_fusion_model.py
```

Do not keep everything inside the current `main.py`.

### 3. Bring `LLMsKnow` Datasets Into `tokenizer`

Use `LLMsKnow/data/` as dataset sources, but not the old script layout.

What to reuse:

- benchmark CSV loaders
- correctness logic
- exact answer extraction logic
- optional resampling logic

How to integrate:

- create adapter loaders inside `tokenizer/collectors/llmsknow_adapter.py`
- convert each dataset row into the canonical schema
- generate model answers using your current target model
- attach:
  - `gold_answer`
  - `model_answer`
  - `gold_correctness`
  - optional `exact_answer`
  - optional `token_scores`

Important note:

`LLMsKnow` labels are mostly correctness labels, not always pure hallucination labels. A wrong answer on a benchmark is useful supervision, but it is not identical to contextual hallucination. Keep these labels separate in the schema.

### 4. Bring `Lookback-Lens` Signals Into `tokenizer`

What to reuse:

- the idea of per-layer, per-head lookback ratio
- context vs generated-token attention split
- judge-based faithfulness labels
- span-level supervision

How to integrate:

- add `lookback_ratio.py` to compute the ratio during generation
- if context exists, compute:
  - attention on source context
  - attention on generated continuation
  - ratio for each layer/head/time step
- save this into `signals.lookback_ratio`
- if using a judge, save:
  - `judge_faithfulness`
  - `span_labels`

This gives you a token-time faithfulness signal that complements your graph.

### 5. Keep The Current Graph Builder As The Core

Your current advantage is the graph object.

The graph should remain the primary training object for the CHARM-style critic:

- nodes = generation steps or token states
- node features = hidden states, optional confidence, optional probe scores
- edges = sparse attention edges
- edge attributes = attention weight, optional edge type

Then enrich each graph with extra side channels:

- graph-level labels from human or benchmark correctness
- token-level span labels from judge annotations
- auxiliary dense tensor `lookback_ratio`

This turns the graph into a multimodal training example instead of only a sparse trace.

### 6. Use A Tiered Labeling System

Do not collapse all supervision into one boolean immediately.

Recommended label tiers:

- Tier 1: `gold_correctness`
  - from benchmark answer matching
- Tier 2: `judge_faithfulness`
  - from GPT-4o or a local evaluator
- Tier 3: `human_hallucination`
  - from direct manual grading
- Tier 4: `span_labels`
  - token/span localization of unsupported content

Why this matters:

- `LLMsKnow` gives cheap scale
- `Lookback-Lens` gives contextual faithfulness
- current `tokenizer` gives highest quality manual labels

A unified collector can train on all of them with multi-task loss.

### 7. Save More Than Just `.pt` Graph Files

Right now `tokenizer` only saves graph `.pt` files. That is too narrow if you want to combine all three pipelines.

Recommended storage:

```text
tokenizer/
  unified_dataset/
    samples/
      sample_000001.pt
      sample_000002.pt
    manifests/
      train.jsonl
      valid.jsonl
      test.jsonl
    raw_exports/
      llmsknow_answers.csv
      lookback_annotations.jsonl
```

Each `.pt` sample can store:

- PyG graph
- auxiliary tensors
- metadata dict
- labels dict

The manifest should help with:

- train/val/test splits
- filtering by source
- filtering by label type
- ablation studies

### 8. Train Multiple Models, Not Just One

Once the unified dataset exists, train several models:

#### A. Graph critic

- input: graph + auxiliary tensors
- model: GNN
- target: hallucination / faithfulness / correctness

#### B. Probe baseline

- input: selected hidden state vectors
- model: logistic regression or shallow MLP
- target: correctness or hallucination

#### C. Lookback baseline

- input: lookback-ratio features
- model: logistic regression
- target: contextual faithfulness

#### D. Fusion critic

- input:
  - graph embedding from GNN
  - lookback summary features
  - probe score
  - confidence text features
- model: small MLP on top of fused features

This is the fairest way to compare whether the graph actually adds value.

### 9. Use Multi-Task Training

A good final critic should not learn from only one label source.

Possible losses:

- graph-level BCE loss for hallucination
- graph-level BCE loss for correctness
- token-level BCE loss for span labels
- auxiliary regression or classification loss for confidence mismatch

Example idea:

- main head predicts `human_hallucination`
- auxiliary head predicts `gold_correctness`
- token head predicts hallucinated span tokens

This lets cheap labels regularize the expensive human-labeled graph task.

### 10. Introduce Data Provenance Explicitly

Every saved example should record:

- which directory it came from
- which original dataset it came from
- which label type was used
- whether the label is human, gold, or judge-derived

Without provenance, later training results will be hard to trust.

## Practical Migration Plan

### Phase 1: Stabilize Current `tokenizer`

- fix generation/runtime issues
- make sample saving deterministic
- save richer metadata per trace

### Phase 2: Add Unified Sample Schema

- wrap the current graph save path in a canonical sample object
- keep backward compatibility with current `trace_*.pt`

### Phase 3: Add `LLMsKnow` Dataset Adapters

- import CSV datasets
- generate answers automatically
- compute correctness labels
- save samples without requiring human grading

### Phase 4: Add Lookback Features

- compute lookback ratios for context tasks
- save them beside graph traces
- optionally import judge-based faithfulness annotations

### Phase 5: Add Training Scripts

- `train_probe_baseline.py`
- `train_lookback_baseline.py`
- `train_gnn.py`
- `train_fusion_model.py`

### Phase 6: Compare Sources And Ablations

Train and compare:

- graph only
- lookback only
- probe only
- graph + lookback
- graph + lookback + automatic labels

## What Should Be Reused vs Rewritten

### Reuse directly

- dataset files from `LLMsKnow/data`
- benchmark loading logic
- correctness heuristics where sensible
- lookback-ratio idea and computation logic

### Rewrite cleanly

- all path handling
- sample saving format
- script orchestration
- training entry points
- shared schema

### Use carefully

- GPT-4o annotation flow from `Lookback-Lens`
  - useful, but costly
- exact-answer extraction from `LLMsKnow`
  - useful, but task-specific
- resampling code from `LLMsKnow`
  - useful for uncertainty, but not required for the first version

## Final Recommendation

Yes, combine them, but in this order:

1. keep `tokenizer` as the core package
2. import `LLMsKnow` datasets and correctness labels as scalable supervision
3. import `Lookback-Lens` attention-derived lookback features and span labels
4. store everything in one unified sample format
5. train graph, probe, lookback, and fusion baselines side by side

The best end state is not "three repos glued together". The best end state is:

- one unified collector
- one unified dataset schema
- multiple complementary labels
- multiple comparable models
- one graph-centric critic that can exploit all of the above

## Minimum Viable Unified Version

If you want the smallest useful combined system, implement only this:

- reuse `LLMsKnow` benchmark datasets
- generate answers with your `tokenizer` model runner
- collect:
  - graph traces
  - lookback ratios
  - correctness labels
- save one `.pt` sample per example
- train:
  - one logistic baseline
  - one GNN critic

That gives you a clean first version without dragging in every research feature at once.
