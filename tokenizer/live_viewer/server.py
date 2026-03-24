import argparse
import json
import os
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
STATE_PATH = os.path.join(RUNTIME_DIR, "state.json")
EVENTS_PATH = os.path.join(RUNTIME_DIR, "events.jsonl")
CONTROL_PATH = os.path.join(RUNTIME_DIR, "control.json")
TOKENIZER_DIR = os.path.dirname(BASE_DIR)


class CollectorProcessManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._process = None
        self._logs = []
        self._max_logs = 400

    def start(self, custom_payload=None):
        with self._lock:
            if self.is_running():
                return False, "Collector already running."
            self._reset_runtime_files()
            self._logs = []
            command = [sys.executable, "collectors/collect_benchmark.py"]
            if custom_payload:
                command.extend(
                    [
                        "--custom-prompt",
                        custom_payload.get("prompt", ""),
                        "--custom-context",
                        custom_payload.get("context", ""),
                        "--custom-gold-answer",
                        custom_payload.get("gold_answer", ""),
                        "--custom-sample-id",
                        custom_payload.get("sample_id", "custom_0"),
                        "--custom-dataset-name",
                        custom_payload.get("dataset_name", "CustomUserPrompt"),
                    ]
                )
            self._process = subprocess.Popen(
                command,
                cwd=TOKENIZER_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            threading.Thread(target=self._consume_output, daemon=True).start()
            threading.Thread(target=self._wait_for_exit, daemon=True).start()
            return True, "Collector started."

    def get_status(self):
        with self._lock:
            if self._process is None:
                return {"running": False, "pid": None, "logs": self._logs[-80:]}
            running = self._process.poll() is None
            return {
                "running": running,
                "pid": self._process.pid,
                "returncode": None if running else self._process.returncode,
                "logs": self._logs[-80:],
            }

    def stop(self):
        with self._lock:
            if not self.is_running():
                return False, "Collector is not running."
            self._process.terminate()
            return True, "Stop signal sent to collector."

    def skip_current(self):
        with self._lock:
            if not self.is_running():
                return False, "Collector is not running."
            control = {"skip_current": True}
            with open(CONTROL_PATH, "w", encoding="utf-8") as handle:
                json.dump(control, handle)
            return True, "Skip signal sent for current sample."

    def is_running(self):
        return self._process is not None and self._process.poll() is None

    def _consume_output(self):
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            cleaned = line.rstrip()
            if not cleaned:
                continue
            with self._lock:
                self._logs.append(cleaned)
                if len(self._logs) > self._max_logs:
                    self._logs = self._logs[-self._max_logs :]

    def _wait_for_exit(self):
        process = self._process
        if process is None:
            return
        process.wait()

    def _reset_runtime_files(self):
        os.makedirs(RUNTIME_DIR, exist_ok=True)
        with open(EVENTS_PATH, "w", encoding="utf-8"):
            pass
        with open(STATE_PATH, "w", encoding="utf-8") as handle:
            json.dump({"run": None, "current_sample": None, "history": []}, handle)
        with open(CONTROL_PATH, "w", encoding="utf-8") as handle:
            json.dump({"skip_current": False}, handle)


PROCESS_MANAGER = CollectorProcessManager()


def read_state():
    if not os.path.exists(STATE_PATH):
        return {"run": None, "current_sample": None, "process": PROCESS_MANAGER.get_status()}
    with open(STATE_PATH, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    state["process"] = PROCESS_MANAGER.get_status()
    return state


def read_events(after_seq):
    if not os.path.exists(EVENTS_PATH):
        return []
    events = []
    with open(EVENTS_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event["seq"] > after_seq:
                events.append(event)
    return events


class LiveViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._send_json(read_state())
            return
        if parsed.path == "/api/events":
            query = parse_qs(parsed.query)
            after = int(query.get("after", ["0"])[0])
            self._send_json({"events": read_events(after)})
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/start":
            started, message = PROCESS_MANAGER.start()
            status = PROCESS_MANAGER.get_status()
            code = HTTPStatus.OK if started else HTTPStatus.CONFLICT
            self._send_json({"started": started, "message": message, "process": status}, status=code)
            return
        if parsed.path == "/api/skip":
            skipped, message = PROCESS_MANAGER.skip_current()
            status = PROCESS_MANAGER.get_status()
            code = HTTPStatus.OK if skipped else HTTPStatus.CONFLICT
            self._send_json({"skipped": skipped, "message": message, "process": status}, status=code)
            return
        if parsed.path == "/api/stop":
            stopped, message = PROCESS_MANAGER.stop()
            status = PROCESS_MANAGER.get_status()
            code = HTTPStatus.OK if stopped else HTTPStatus.CONFLICT
            self._send_json({"stopped": stopped, "message": message, "process": status}, status=code)
            return
        if parsed.path == "/api/start-custom":
            payload = self._read_json_body()
            prompt = (payload.get("prompt") or "").strip()
            if not prompt:
                self._send_json(
                    {"started": False, "message": "Custom prompt cannot be empty."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            started, message = PROCESS_MANAGER.start(custom_payload=payload)
            status = PROCESS_MANAGER.get_status()
            code = HTTPStatus.OK if started else HTTPStatus.CONFLICT
            self._send_json({"started": started, "message": message, "process": status}, status=code)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format, *args):
        return

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}


def main():
    parser = argparse.ArgumentParser(description="Serve the tokenizer live graph viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    os.makedirs(RUNTIME_DIR, exist_ok=True)
    PROCESS_MANAGER._reset_runtime_files()
    server = ThreadingHTTPServer((args.host, args.port), LiveViewerHandler)
    print(f"Live viewer running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
