"""LoRA SFT with MLX and JSONL metrics."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import load_dataset

from persona import NAME

ROOT = Path(__file__).resolve().parent


class MetricsCallback(TrainingCallback):
    def __init__(self, out):
        self.path = out / "metrics.jsonl"
        self.started = time.monotonic()

    def write(self, kind, info):
        row = {
            "project": NAME,
            "type": kind,
            **info,
            "wall_time": round(time.monotonic() - self.started, 1),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def on_train_loss_report(self, info):
        self.write("train", info)

    def on_val_loss_report(self, info):
        self.write("val", info)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(**CONFIG_DEFAULTS)
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", type=Path, default=Path("runs") / NAME)
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=400)
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", "--resume-adapter-file", dest="resume_adapter_file")
    args = parser.parse_args()
    args.project = NAME
    args.train, args.mask_prompt = True, True
    args.adapter_path = str(args.out / "adapters")
    if (args.out / "metrics.jsonl").exists() and not args.resume_adapter_file:
        parser.error("output already has metrics; use --resume or a new --out")
    return args


def main():
    os.chdir(ROOT)
    args = parse_args()
    out = args.out
    del args.out
    out.mkdir(parents=True, exist_ok=True)
    for key in ("model", "data", "adapter_path", "resume_adapter_file"):
        value = getattr(args, key)
        if value and Path(value).is_absolute():
            setattr(args, key, os.path.relpath(value, ROOT))
    config_text = json.dumps(vars(args), indent=2)
    (out / f"run_config.{time.time_ns()}.json").write_text(config_text)
    (out / "run_config.json").write_text(config_text)
    stats = Path(args.data) / "stats.json"
    if stats.exists():
        (out / "data_stats.json").write_bytes(stats.read_bytes())
    status = {"project": NAME, "state": "TRAINING", "pid": os.getpid()}

    def save_status():
        path = out / "status.json"
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(status))
        pending.replace(path)

    save_status()
    try:
        np.random.seed(args.seed)
        model, tokenizer = load(args.model)
        train_set, valid_set, _ = load_dataset(args, tokenizer)
        # mlx_lm.lora.run replaces the supplied callback with reporting callbacks.
        train_model(args, model, train_set, valid_set, MetricsCallback(out))
    except BaseException as error:
        status.update(state="STOPPED", error=type(error).__name__)
        raise
    else:
        status["state"] = "COMPLETED"
    finally:
        save_status()


if __name__ == "__main__":
    main()
