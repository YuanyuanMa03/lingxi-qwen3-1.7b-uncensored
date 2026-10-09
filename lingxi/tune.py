"""Compare a small LoRA grid on fixed validation rows and reply prompts."""

import argparse
from collections import Counter
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
from mlx_lm.tuner.trainer import evaluate
import numpy as np

from .persona import NAME, PERSONA, qwen_messages

ROOT = Path(__file__).resolve().parents[1]


def assess(model_path, adapter, data, out, validation_count=128):
    out.mkdir(parents=True, exist_ok=False)
    valid = Path(data) / "valid.jsonl"
    rows = [json.loads(line) for line in valid.read_text().splitlines()]
    indices = list(range(len(rows)))
    random.Random(42).shuffle(indices)
    if validation_count:
        indices = indices[:validation_count]
    questions = [
        (int(n), q)
        for n, q in re.findall(r"^(\d+)\.\s+(.+)$", (ROOT / "docs/questions.md").read_text(), re.M)
        if int(n) in (1, 2, 3, 4, 5, 6, 10)
    ] + [
        (11, "只输出结果：17乘23等于多少？"),
        (12, "只输出JSON，不要解释：姓名小满，年龄25，字段为name和age。"),
        (13, "用一句中文解释为什么夏天白天更长，不超过50字。"),
    ]
    config = {
        "project": NAME,
        "model": str(model_path),
        "adapter": str(adapter) if adapter else None,
        "data": str(data),
        "validation_sha256": hashlib.sha256(valid.read_bytes()).hexdigest(),
        "validation_indices": indices,
        "mask_prompt": True,
        "max_seq_length": 1536,
        "seed": 42,
        "max_tokens": 384,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "enable_thinking": False,
        "system_prompt": PERSONA,
        "questions": questions,
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    model, tokenizer = load(model_path, adapter_path=str(adapter) if adapter else None)
    dataset = CacheDataset(ChatDataset([rows[i] for i in indices], tokenizer, mask_prompt=True))
    np.random.seed(42)
    loss = evaluate(model, dataset, batch_size=1, num_batches=len(indices), max_seq_length=1536)
    answers = []
    for number, question in questions:
        prompt = tokenizer.apply_chat_template(
            qwen_messages(
                [{"role": "system", "content": PERSONA}, {"role": "user", "content": question}]
            ),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        mx.random.seed(42 + number)
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=384,
            sampler=make_sampler(temp=0.7, top_p=0.8, top_k=20),
        )
        compact = re.sub(r"\s+", "", text)
        repeats = Counter(compact[i : i + 12] for i in range(max(0, len(compact) - 11)))
        row = {
            "id": number,
            "prompt": question,
            "text": text,
            "max_repeated_12gram": max(repeats.values(), default=0),
        }
        answers.append(row)
        with (out / "answers.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(out.parent.name, "answer", number, flush=True)
    by_id = {row["id"]: row for row in answers}
    try:
        json_correct = json.loads(by_id[12]["text"]) == {"name": "小满", "age": 25}
    except json.JSONDecodeError:
        json_correct = False
    result = {
        "validation_loss": loss,
        "validation_rows": len(indices),
        "repetitive_ids": [a["id"] for a in answers if a["max_repeated_12gram"] >= 6],
        "image_ids": [a["id"] for a in answers if re.search(r"!\[.*?\]\(", a["text"])],
        "arithmetic_correct": bool(re.search(r"(?<!\d)391(?!\d)", by_id[11]["text"])),
        "arithmetic_exact_format": by_id[11]["text"].strip() == "391",
        "json_correct": json_correct,
        "short_reply_ids": [a["id"] for a in answers if a["id"] <= 10 and len(a["text"]) < 40],
        "review_scope": "diagnostics and assistant review; not a general capability benchmark",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(out.parent.name, result, flush=True)
    del model, tokenizer, dataset
    gc.collect()
    mx.clear_cache()
    return result


def train_trial(data, run, learning_rate, scale, iters=200, rank=8, schedule=False, accumulation=4):
    command = [
        sys.executable,
        "-u",
        "-m",
        "lingxi.train",
        "--data",
        str(data),
        "--out",
        str(run),
        "--iters",
        str(iters),
        "--learning-rate",
        str(learning_rate),
        "--lora-rank",
        str(rank),
        "--lora-scale",
        str(scale),
        "--grad-accumulation-steps",
        str(accumulation),
        "--steps-per-report",
        "20",
        "--steps-per-eval",
        str(iters),
        "--val-batches",
        "25",
        "--save-every",
        str(iters // 2),
    ]
    if schedule:
        command += [
            "--cosine-schedule",
            "--warmup-updates",
            str(max(1, iters // accumulation // 20)),
        ]
    with run.with_name(run.name + ".log").open("x") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    return assess("models/Qwen3-1.7B", run / "adapters", data, run / "assessment")


def complete_run(data, run, lr, scale, rank, schedule, accumulation):
    if run.exists():
        raise FileExistsError(run)
    run.parent.mkdir(parents=True, exist_ok=True)
    samples = sum(1 for _ in (data / "train.jsonl").open())
    iters = accumulation * ((samples + accumulation - 1) // accumulation)
    pipeline = run.with_name(run.name + ".pipeline.json")
    if pipeline.exists() or run.with_name(run.name + ".log").exists():
        raise FileExistsError("pipeline output already exists")

    def stage(state, **details):
        pending = pipeline.with_name(pipeline.name + ".tmp")
        pending.write_text(
            json.dumps(
                {"project": NAME, "state": state, "run": str(run), "pid": os.getpid(), **details},
                indent=2,
            )
        )
        pending.replace(pipeline)
        print(state, details, flush=True)

    command = [
        sys.executable,
        "-u",
        "-m",
        "lingxi.train",
        "--data",
        str(data),
        "--out",
        str(run),
        "--iters",
        str(iters),
        "--learning-rate",
        str(lr),
        "--lora-rank",
        str(rank),
        "--lora-scale",
        str(scale),
        "--grad-accumulation-steps",
        str(accumulation),
        "--steps-per-eval",
        "2048",
        "--val-batches",
        "-1",
        "--save-every",
        "2048",
        "--steps-per-report",
        "20",
    ]
    if schedule:
        command += [
            "--cosine-schedule",
            "--warmup-updates",
            str(max(1, iters // accumulation // 20)),
        ]
    try:
        stage(
            "TRAINING", samples=samples, microbatches=iters, optimizer_updates=iters // accumulation
        )
        with run.with_name(run.name + ".log").open("x") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        stage("EVALUATING")
        assess("models/Qwen3-1.7B", None, data, run / "base-assessment", validation_count=None)
        result = assess(
            "models/Qwen3-1.7B",
            run / "adapters",
            data,
            run / "final-assessment",
            validation_count=None,
        )
        if (
            result["repetitive_ids"]
            or result["image_ids"]
            or not result["arithmetic_correct"]
            or not result["json_correct"]
            or len(result["short_reply_ids"]) >= 4
        ):
            stage("QUALITY_REVIEW_REQUIRED", result=result)
            return
        export = Path("dist") / run.name
        if export.exists():
            raise FileExistsError(export)
        stage("EXPORTING", output=str(export))
        subprocess.run(
            [sys.executable, "-m", "lingxi.publish", "--run", str(run), "--out", str(export)],
            check=True,
            cwd=ROOT,
        )
        model, _, config = load(export / NAME, return_config=True)
        bits = {m.bits for _, m in model.named_modules() if hasattr(m, "bits")}
        if bits != {4} or config["quantization"]["bits"] != 4:
            raise ValueError("export is not MLX 4-bit")
        del model
        gc.collect()
        mx.clear_cache()
        stage("EVALUATING_4BIT")
        quantized = assess(
            str(export / NAME), None, data, run / "4bit-assessment", validation_count=None
        )
        if (
            quantized["repetitive_ids"]
            or quantized["image_ids"]
            or not quantized["arithmetic_correct"]
            or not quantized["json_correct"]
            or len(quantized["short_reply_ids"]) >= 4
        ):
            stage("QUALITY_REVIEW_REQUIRED", quantized_result=quantized, output=str(export / NAME))
            return
        stage(
            "COMPLETED",
            adapter_result=result,
            quantized_result=quantized,
            output=str(export / NAME),
            review="diagnostics only; read raw replies before release",
        )
    except BaseException as error:
        stage("STOPPED", error=type(error).__name__)
        raise


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed-qwen3"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--full", action="store_true", help="one epoch followed by evaluation and local export"
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lora-scale", type=float)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--grad-accumulation-steps", type=int, default=4)
    parser.add_argument("--cosine-schedule", action="store_true")
    args = parser.parse_args()
    if args.grad_accumulation_steps < 1:
        parser.error("gradient accumulation must be positive")
    if args.full:
        if args.learning_rate is None or args.lora_scale is None:
            parser.error("full training requires the selected --learning-rate and --lora-scale")
        complete_run(
            args.data,
            args.out,
            args.learning_rate,
            args.lora_scale,
            args.lora_rank,
            args.cosine_schedule,
            args.grad_accumulation_steps,
        )
        return
    if args.iters < 2 or args.iters % (2 * args.grad_accumulation_steps):
        parser.error("iters must be a positive multiple of twice the gradient accumulation")
    if args.learning_rate is not None or args.lora_scale is not None:
        if args.learning_rate is None or args.lora_scale is None:
            parser.error("a single trial requires both --learning-rate and --lora-scale")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        train_trial(
            args.data,
            args.out,
            args.learning_rate,
            args.lora_scale,
            args.iters,
            args.lora_rank,
            args.cosine_schedule,
            args.grad_accumulation_steps,
        )
        return
    args.out.mkdir(parents=True, exist_ok=False)
    plan = {
        "project": NAME,
        "data": str(args.data),
        "iters": args.iters,
        "learning_rates": [1e-4, 2e-5],
        "scales": [20, 4],
        "rank": args.lora_rank,
        "accumulation": args.grad_accumulation_steps,
        "cosine_schedule": args.cosine_schedule,
        "selection": "reply quality gates first; then fixed validation loss",
    }
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2))
    results = {"base": assess("models/Qwen3-1.7B", None, args.data, args.out / "base")}
    for lr, scale in itertools.product(plan["learning_rates"], plan["scales"]):
        label = f"lr{lr:g}-scale{scale}"
        print("Starting", label, flush=True)
        results[label] = train_trial(
            args.data,
            args.out / label,
            lr,
            scale,
            args.iters,
            args.lora_rank,
            args.cosine_schedule,
            args.grad_accumulation_steps,
        )
        (args.out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
