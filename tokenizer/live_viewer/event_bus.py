import json
import os
import threading
import time
import copy
from dataclasses import asdict, is_dataclass


class LiveEventBus:
    def __init__(self, runtime_dir=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.runtime_dir = runtime_dir or os.path.join(base_dir, "runtime")
        os.makedirs(self.runtime_dir, exist_ok=True)
        self.events_path = os.path.join(self.runtime_dir, "events.jsonl")
        self.state_path = os.path.join(self.runtime_dir, "state.json")
        self._lock = threading.Lock()

    def reset(self, initial_state):
        with self._lock:
            with open(self.events_path, "w", encoding="utf-8"):
                pass
            self._write_json(self.state_path, initial_state)

    def update_state(self, state):
        with self._lock:
            self._write_json(self.state_path, state)

    def emit(self, event_type, payload, seq):
        event = {
            "seq": seq,
            "type": event_type,
            "ts": time.time(),
            "payload": self._normalize(payload),
        }
        with self._lock:
            with open(self.events_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

    def _write_json(self, path, payload):
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(self._normalize(payload), handle)
        os.replace(temp_path, path)

    def _normalize(self, value):
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return {key: self._normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._normalize(item) for item in value]
        return value


class LiveDemoReporter:
    def __init__(self, bus, max_live_edges_per_step=6):
        self.bus = bus
        self.max_live_edges_per_step = max_live_edges_per_step
        self._event_seq = 0
        self._last_text = ""
        self._state = self._build_empty_state()

    def start_run(self, model_name, dataset_name, total_samples):
        self._event_seq = 0
        self._last_text = ""
        self._state = self._build_empty_state()
        self._state["run"] = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "total_samples": total_samples,
            "started_at": time.time(),
            "status": "running",
        }
        self.bus.reset(self._state)
        self._emit("run_started", self._state["run"])

    def start_sample(self, example, prompt, prompt_token_count):
        self._last_text = ""
        self._state["current_sample"] = {
            "sample_id": example.sample_id,
            "dataset_name": example.dataset_name,
            "split": example.split,
            "prompt": example.prompt,
            "context": example.context,
            "prompt_with_system": prompt,
            "prompt_token_count": prompt_token_count,
            "generated_text": "",
            "status": "running",
            "nodes": [],
            "edges": [],
            "lookback_ratios": [],
            "metrics": {
                "node_count": 0,
                "edge_count": 0,
                "latest_lookback_ratio": None,
            },
        }
        self._sync_state()
        self._emit(
            "sample_started",
            {
                "sample_id": example.sample_id,
                "dataset_name": example.dataset_name,
                "split": example.split,
                "prompt": example.prompt,
                "context": example.context,
            },
        )

    def update_generated_text(self, generated_text):
        current = self._current_sample()
        if current is None or generated_text == self._last_text:
            return
        self._last_text = generated_text
        current["generated_text"] = generated_text
        self._sync_state()
        self._emit("text_updated", {"generated_text": generated_text})

    def add_graph_step(self, step_index, edges, lookback_ratio=None):
        current = self._current_sample()
        if current is None:
            return

        node_id = f"g{step_index}"
        existing_ids = {node["id"] for node in current["nodes"]}
        if node_id not in existing_ids:
            current["nodes"].append(
                {
                    "id": node_id,
                    "label": str(step_index),
                    "step": step_index,
                    "group": "generated",
                }
            )

        live_edges = self._compress_edges(edges)
        existing_edge_ids = {edge["id"] for edge in current["edges"]}
        new_edges = []
        for edge in live_edges:
            if edge["id"] in existing_edge_ids:
                continue
            current["edges"].append(edge)
            new_edges.append(edge)

        if lookback_ratio is not None:
            current["lookback_ratios"].append(lookback_ratio)
            current["metrics"]["latest_lookback_ratio"] = lookback_ratio

        current["metrics"]["node_count"] = len(current["nodes"])
        current["metrics"]["edge_count"] = len(current["edges"])
        self._sync_state()
        self._emit(
            "graph_step",
            {
                "node": {
                    "id": node_id,
                    "label": str(step_index),
                    "step": step_index,
                    "group": "generated",
                },
                "edges": new_edges,
                "lookback_ratio": lookback_ratio,
                "metrics": current["metrics"],
            },
        )

    def mark_sample_failed(self, sample_id, reason):
        current = self._current_sample()
        if current is None:
            return
        current["status"] = "failed"
        current["failure_reason"] = reason
        self._archive_current_sample()
        self._sync_state()
        self._emit("sample_failed", {"sample_id": sample_id, "reason": reason})

    def mark_sample_saved(self, sample_id, path, correctness):
        current = self._current_sample()
        if current is None:
            return
        current["status"] = "saved"
        current["saved_path"] = path
        current["correctness"] = correctness
        self._archive_current_sample()
        self._sync_state()
        self._emit(
            "sample_saved",
            {
                "sample_id": sample_id,
                "path": path,
                "correctness": correctness,
                "metrics": current["metrics"],
            },
        )

    def finish_run(self):
        if self._state["run"] is None:
            return
        self._state["run"]["status"] = "finished"
        self._state["run"]["finished_at"] = time.time()
        self._sync_state()
        self._emit("run_finished", self._state["run"])

    def _compress_edges(self, edges):
        if not edges:
            return []
        ranked = sorted(edges, key=lambda item: item[2], reverse=True)[: self.max_live_edges_per_step]
        compressed = []
        for source, target, weight in ranked:
            compressed.append(
                {
                    "id": f"{source}->{target}",
                    "from": f"g{source}",
                    "to": f"g{target}",
                    "weight": round(float(weight), 4),
                }
            )
        return compressed

    def _current_sample(self):
        return self._state.get("current_sample")

    def _sync_state(self):
        self.bus.update_state(self._state)

    def _emit(self, event_type, payload):
        self._event_seq += 1
        self.bus.emit(event_type, payload, self._event_seq)

    def _build_empty_state(self):
        return {
            "run": None,
            "current_sample": None,
            "history": [],
        }

    def _archive_current_sample(self):
        current = self._current_sample()
        if current is None:
            return
        archived = copy.deepcopy(current)
        self._state["history"].append(archived)
