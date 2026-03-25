# Tokenizer Implementation Tracker

This file tracks what has been completed and what is next.

## Completed

- Added tokenizer-local dataset vendoring under:
  - `tokenizer/data/llmsknow`
  - `tokenizer/data/lookback_lens`
- Added reference code snapshots under:
  - `tokenizer/reference/llmsknow`
  - `tokenizer/reference/lookback_lens`
- Unified benchmark collector CLI with custom prompt mode.
- Added live viewer controls:
  - start
  - custom run
  - skip current
  - stop run
  - history replay + back-to-live mode
- Added early-stop guard for runaway generations (`max_generated_tokens`).
- Added human feedback module:
  - `tokenizer/human_feedback/prepare_tasks.py`
  - `tokenizer/human_feedback/annotation_api.py`
  - `tokenizer/human_feedback/README.md`
- Added all-core dataset loaders in `tokenizer/datasets/benchmark_loader.py`:
  - `math`
  - `nq`
  - `movie_qa`
  - `mnli`
  - `winogrande`
  - `winobias`
- Updated collector dataset choices to support all loader keys above.

## In Progress / Next

1. Add `--dataset all` support to collector for sequential multi-dataset runs.
2. Add split-manifest generation for reproducible train/val/test partitions.
3. Add one merge script to inject human annotations into sample metadata.
4. Add lightweight browser page for annotation (token link -> task view -> submit).
5. Add dataset balancing caps per source during collection for fair training.

## Quick Commands

- Run benchmark collector:
  - `python tokenizer/collectors/collect_benchmark.py --dataset math --split test --limit 100`
- Run custom collector:
  - `python tokenizer/collectors/collect_benchmark.py --custom-prompt "..." --custom-gold-answer "..."`
- Generate annotation tasks:
  - `python tokenizer/human_feedback/prepare_tasks.py --dataset math --split test --limit 200 --output tokenizer/human_feedback/tasks_math_test.jsonl`
