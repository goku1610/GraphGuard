# Human Feedback Workflow

This module supports:

1. generate model answers first (`prepare_tasks.py`)
2. import tasks into an annotation API (`annotation_api.py`)
3. assign non-overlapping tasks to annotators
4. collect labels via token-based links
5. export labels for merge-back into training data

## 1) Generate task file (model answers first)

From project root:

```bash
python tokenizer/human_feedback/prepare_tasks.py --dataset math --split test --limit 200 --output tokenizer/human_feedback/tasks_math_test.jsonl
```

NaturalQuestions:

```bash
python tokenizer/human_feedback/prepare_tasks.py --dataset nq --split test --limit 200 --output tokenizer/human_feedback/tasks_nq_test.jsonl
```

## 2) Start annotation API

```bash
python tokenizer/human_feedback/annotation_api.py --host 0.0.0.0 --port 8100
```

## 3) Admin setup with curl

Create annotators:

```bash
curl -X POST http://localhost:8100/admin/annotators -H "Content-Type: application/json" -d '{"name":"alice"}'
curl -X POST http://localhost:8100/admin/annotators -H "Content-Type: application/json" -d '{"name":"bob"}'
```

Import tasks:

```bash
curl -X POST http://localhost:8100/admin/tasks/import -H "Content-Type: application/json" -d '{"path":"tokenizer/human_feedback/tasks_math_test.jsonl"}'
```

Assign in round-robin:

```bash
curl -X POST http://localhost:8100/admin/assign/round-robin -H "Content-Type: application/json" -d '{"max_tasks_per_annotator":100}'
```

## 4) Share links

Each annotator response includes a link:

```text
http://localhost:8100/annotate?token=<token>
```

Use your public domain/IP instead of localhost when sharing externally.

## 5) Annotator API flow

Get next task:

```bash
curl "http://localhost:8100/annotate/next?token=<token>"
```

Submit annotation:

```bash
curl -X POST http://localhost:8100/annotate/submit \
  -H "Content-Type: application/json" \
  -d '{"token":"<token>","task_id":"answerablemath_test_0","is_correct":0,"is_hallucinated":1,"severity":4,"notes":"Wrong final numeric answer"}'
```

Progress:

```bash
curl "http://localhost:8100/annotate/progress?token=<token>"
```

## 6) Export collected labels

```bash
curl "http://localhost:8100/admin/export"
```

Default output:

```text
tokenizer/human_feedback/annotations_export.jsonl
```
