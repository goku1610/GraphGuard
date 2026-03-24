# Run Commands

This file lists the main commands to run GraphGuard components.

## 0) Go to project root

```bash
cd /home/saksham/coding/GraphGuard
```

## 1) Environment setup (first time)

```bash
pip install -r LLMsKnow/requirements.txt
pip install -r Lookback-Lens/requirements.txt
pip install fastapi uvicorn
```

If you use a virtual environment, activate it first.

## 2) Collect traces (benchmark mode)

Default (math test, first 10):

```bash
python tokenizer/collectors/collect_benchmark.py
```

NaturalQuestions mode:

```bash
python tokenizer/collectors/collect_benchmark.py --dataset nq --split test --limit 10
```

Math with custom limit:

```bash
python tokenizer/collectors/collect_benchmark.py --dataset math --split test --limit 25
```

## 3) Collect traces (single custom prompt from CLI)

```bash
python tokenizer/collectors/collect_benchmark.py \
  --custom-prompt "What is the capital of Australia?" \
  --custom-context "" \
  --custom-gold-answer "Canberra" \
  --custom-sample-id "custom_demo_1" \
  --custom-dataset-name "CustomUserPrompt"
```

## 4) Run live viewer (3D cone frontend for collector)

```bash
python tokenizer/live_viewer/server.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

Notes:
- Use the UI buttons for start/skip/stop/custom run.
- If port is busy, change `--port` (for example `8780`).

## 5) Train baseline model (non-graph logistic regression)

```bash
python tokenizer/trainers/baseline_trainer.py
```

## 6) Train GNN critic

```bash
python tokenizer/trainers/gnn_trainer.py
```

Weights output:

```text
tokenizer/trainers/weights/charm_critic_v1.pth
```

## 7) Run standalone backend API (for standalone React UI)

```bash
cd tokenizer/standalone_ui
python api.py
```

API endpoint:

```text
http://localhost:8000/api/analyze
```

## 8) Run standalone React frontend

In a new terminal:

```bash
cd /home/saksham/coding/GraphGuard/tokenizer/standalone_ui/frontend
npm install
npm run dev
```

Then open the Vite URL shown in terminal (usually `http://localhost:5173`).

## 9) Recommended terminal layout

Use separate terminals:

1. Collector or API backend
2. Live viewer server (optional)
3. Standalone frontend (optional)

## 10) Where outputs are stored

- Trace dataset (`.pt`): `tokenizer/charm_unified_dataset/`
- Live viewer runtime files: `tokenizer/live_viewer/runtime/`
- Trained GNN weights: `tokenizer/trainers/weights/`
