"""Serve the training dashboard at http://127.0.0.1:8765."""

import argparse
import os
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import psutil

from .persona import NAME

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path("runs") / NAME
DATA_DIR = None


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def load_payload(run_dir, data_dir=None):
    config = read_json(run_dir / "run_config.json")
    stats_path = Path(data_dir or config.get("data", "data/processed-qwen3"))
    if not stats_path.is_absolute():
        stats_path = ROOT / stats_path
    snapshot = run_dir / "data_stats.json"
    stats = read_json(snapshot if snapshot.exists() else stats_path / "stats.json")
    metrics = run_dir / "metrics.jsonl"
    rows = []
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A callback may still be writing the final line.
            if row.get("type") in ("train", "val"):
                row["iter_c"] = row["iteration"]
                rows.append(row)

    train = [r for r in rows if r["type"] == "train"]
    val = [r for r in rows if r["type"] == "val"]
    last = train[-1] if train else {}
    total = config.get("iters")
    current = last.get("iter_c", 0)
    cumulative_total = total
    status_file = read_json(run_dir / "status.json")
    status = status_file.get("state", "WAITING" if not rows else "STALE")
    if status == "TRAINING" and not psutil.pid_exists(status_file.get("pid", -1)):
        status = "STOPPED"
    if not status_file and train and total and last["iteration"] >= total:
        status = "COMPLETED"
    speed = last.get("iterations_per_second", 0)
    eta = (
        max(0, total - last["iteration"]) / speed
        if total and speed and status == "TRAINING"
        else None
    )
    mem = psutil.virtual_memory()
    return {
        "project": NAME,
        "run": run_dir.name,
        "ts": time.time(),
        "status": status,
        "elapsed": last.get("wall_time"),
        "config": config,
        "stats": stats,
        "train": train,
        "val": val,
        "progress": {
            "cum_iter": current,
            "cum_total": cumulative_total,
            "frac": min(1, current / cumulative_total) if cumulative_total else 0,
            "it_per_sec": speed,
            "eta_sec": eta,
        },
        "sys": {
            "cpu": psutil.cpu_percent(),
            "ram_pct": mem.percent,
            "load1": psutil.getloadavg()[0],
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self.send(
                200,
                Path(__file__).with_name("dashboard.html").read_bytes(),
                "text/html; charset=utf-8",
            )
        elif url.path == "/api/metrics":
            run = parse_qs(url.query).get("run", [str(RUN_DIR)])[0]
            run_dir = Path(run)
            if not run_dir.is_absolute():
                run_dir = ROOT / run_dir
            try:
                body = json.dumps(
                    load_payload(run_dir, DATA_DIR), ensure_ascii=False, allow_nan=False
                ).encode()
                self.send(200, body, "application/json; charset=utf-8")
            except (OSError, ValueError, KeyError) as error:
                self.send(500, json.dumps({"error": str(error)}).encode(), "application/json")
        else:
            self.send(404, b"Not found", "text/plain")


def main():
    os.chdir(ROOT)
    global RUN_DIR, DATA_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN_DIR)
    parser.add_argument(
        "--data", type=Path, help="dataset statistics fallback when the run has no snapshot"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    RUN_DIR, DATA_DIR = args.run, args.data
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"http://127.0.0.1:{args.port} ({RUN_DIR.name})", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
