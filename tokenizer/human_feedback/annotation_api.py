import argparse
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class CreateAnnotatorRequest(BaseModel):
    name: str


class ImportTasksRequest(BaseModel):
    path: str


class AssignRequest(BaseModel):
    max_tasks_per_annotator: int = 100


class SubmitRequest(BaseModel):
    token: str
    task_id: str
    is_correct: int
    is_hallucinated: int
    severity: int = 0
    notes: str = ""


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS annotators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            token TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            dataset_name TEXT,
            split TEXT,
            sample_id TEXT,
            prompt TEXT NOT NULL,
            context TEXT,
            gold_answer TEXT,
            generated_text TEXT NOT NULL,
            extracted_answer TEXT,
            auto_correctness INTEGER,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assignments (
            task_id TEXT NOT NULL,
            annotator_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            assigned_at TEXT NOT NULL,
            PRIMARY KEY (task_id, annotator_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            annotator_id INTEGER NOT NULL,
            is_correct INTEGER NOT NULL,
            is_hallucinated INTEGER NOT NULL,
            severity INTEGER NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(task_id, annotator_id)
        )
        """
    )
    conn.commit()
    conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Human feedback annotation API.")
    parser.add_argument(
        "--db",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotations.db"),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    return parser.parse_args()


args = parse_args()
init_db(args.db)
app = FastAPI(title="GraphGuard Human Feedback API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conn():
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    return conn


def get_annotator_by_token(conn, token):
    row = conn.execute("SELECT * FROM annotators WHERE token = ?", (token,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Invalid token.")
    return row


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/admin/annotators")
def create_annotator(req: CreateAnnotatorRequest):
    conn = get_conn()
    token = secrets.token_urlsafe(16)
    try:
        conn.execute(
            "INSERT INTO annotators(name, token, created_at) VALUES (?, ?, ?)",
            (req.name.strip(), token, utc_now()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=409, detail="Annotator name already exists.")
    row = conn.execute("SELECT id, name, token FROM annotators WHERE name = ?", (req.name.strip(),)).fetchone()
    conn.close()
    return {
        "annotator_id": row["id"],
        "name": row["name"],
        "token": row["token"],
        "share_link": f"http://localhost:8100/annotate?token={row['token']}",
    }


@app.post("/admin/tasks/import")
def import_tasks(req: ImportTasksRequest):
    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail="tasks file not found")
    conn = get_conn()
    inserted = 0
    with open(req.path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            sample = item.get("sample", {})
            try:
                conn.execute(
                    """
                    INSERT INTO tasks(
                        id, dataset_name, split, sample_id, prompt, context, gold_answer,
                        generated_text, extracted_answer, auto_correctness, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.get("task_id"),
                        sample.get("dataset_name"),
                        sample.get("split"),
                        sample.get("sample_id"),
                        sample.get("prompt", ""),
                        sample.get("context"),
                        sample.get("gold_answer"),
                        item.get("generated_text", ""),
                        item.get("extracted_answer"),
                        int(item.get("auto_correctness", 0)),
                        json.dumps(item, ensure_ascii=True),
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                continue
    conn.commit()
    conn.close()
    return {"inserted": inserted}


@app.post("/admin/assign/round-robin")
def assign_round_robin(req: AssignRequest):
    conn = get_conn()
    annotators = conn.execute("SELECT id FROM annotators ORDER BY id").fetchall()
    if not annotators:
        conn.close()
        raise HTTPException(status_code=400, detail="No annotators available.")

    tasks = conn.execute("SELECT id FROM tasks ORDER BY id").fetchall()
    added = 0
    for i, task in enumerate(tasks):
        annotator_id = annotators[i % len(annotators)]["id"]
        count_row = conn.execute(
            "SELECT COUNT(*) AS c FROM assignments WHERE annotator_id = ?",
            (annotator_id,),
        ).fetchone()
        if count_row["c"] >= req.max_tasks_per_annotator:
            continue
        try:
            conn.execute(
                "INSERT INTO assignments(task_id, annotator_id, status, assigned_at) VALUES (?, ?, 'pending', ?)",
                (task["id"], annotator_id, utc_now()),
            )
            added += 1
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    conn.close()
    return {"assigned": added}


@app.get("/annotate/next")
def get_next_task(token: str):
    conn = get_conn()
    annotator = get_annotator_by_token(conn, token)
    row = conn.execute(
        """
        SELECT t.*, a.status
        FROM assignments a
        JOIN tasks t ON t.id = a.task_id
        WHERE a.annotator_id = ? AND a.status = 'pending'
        ORDER BY a.assigned_at, t.id
        LIMIT 1
        """,
        (annotator["id"],),
    ).fetchone()
    if row is None:
        conn.close()
        return {"done": True}

    conn.execute(
        "UPDATE assignments SET status = 'in_progress' WHERE task_id = ? AND annotator_id = ? AND status = 'pending'",
        (row["id"], annotator["id"]),
    )
    conn.commit()
    conn.close()
    return {
        "done": False,
        "task": {
            "task_id": row["id"],
            "prompt": row["prompt"],
            "context": row["context"],
            "gold_answer": row["gold_answer"],
            "generated_text": row["generated_text"],
            "extracted_answer": row["extracted_answer"],
            "auto_correctness": row["auto_correctness"],
        },
    }


@app.post("/annotate/submit")
def submit_annotation(req: SubmitRequest):
    conn = get_conn()
    annotator = get_annotator_by_token(conn, req.token)
    assignment = conn.execute(
        "SELECT * FROM assignments WHERE task_id = ? AND annotator_id = ?",
        (req.task_id, annotator["id"]),
    ).fetchone()
    if assignment is None:
        conn.close()
        raise HTTPException(status_code=403, detail="Task is not assigned to this annotator.")

    conn.execute(
        """
        INSERT OR REPLACE INTO annotations(
            task_id, annotator_id, is_correct, is_hallucinated, severity, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            req.task_id,
            annotator["id"],
            int(req.is_correct),
            int(req.is_hallucinated),
            int(req.severity),
            req.notes.strip(),
            utc_now(),
        ),
    )
    conn.execute(
        "UPDATE assignments SET status = 'done' WHERE task_id = ? AND annotator_id = ?",
        (req.task_id, annotator["id"]),
    )
    conn.commit()
    conn.close()
    return {"saved": True}


@app.get("/annotate/progress")
def progress(token: str):
    conn = get_conn()
    annotator = get_annotator_by_token(conn, token)
    row = conn.execute(
        """
        SELECT
            SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) AS done_count,
            COUNT(*) AS total_count
        FROM assignments
        WHERE annotator_id = ?
        """,
        (annotator["id"],),
    ).fetchone()
    conn.close()
    done_count = int(row["done_count"] or 0)
    total_count = int(row["total_count"] or 0)
    return {"done": done_count, "total": total_count}


@app.get("/admin/export")
def export_annotations(path: str = ""):
    out_path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotations_export.jsonl")
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT
            an.task_id, an.annotator_id, au.name AS annotator_name,
            an.is_correct, an.is_hallucinated, an.severity, an.notes, an.created_at
        FROM annotations an
        JOIN annotators au ON au.id = an.annotator_id
        ORDER BY an.created_at
        """
    ).fetchall()
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True) + "\n")
    conn.close()
    return {"exported": len(rows), "path": out_path}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
