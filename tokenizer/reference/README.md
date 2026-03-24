This directory keeps the external reference material that informed the tokenizer pipeline, so `tokenizer/` can remain self-contained after removing the original sibling repos.

Included bundles:

- `llmsknow/`
  - benchmark loading and answer-labeling references
  - copied files:
    - `README.md`
    - `requirements.txt`
    - `compute_correctness.py`
    - `extract_exact_answer.py`
    - `generate_model_answers.py`
    - `resamples_utils.py`
    - `resampling.py`

- `lookback_lens/`
  - lookback-ratio extraction, evaluation, and decoding references
  - copied files:
    - `README.md`
    - `requirements.txt`
    - `step01_extract_attns.py`
    - `step02_eval_gpt4o.py`
    - `step03_lookback_lens.py`
    - `step04_run_decoding.py`
    - `eval_exact_match.py`
    - `generation.py`
    - `lookback_lens_demo.ipynb`
    - `lookback-lens.png`

These are stored for provenance and future porting reference. The active tokenizer runtime should continue to use the code under `tokenizer/` itself.
