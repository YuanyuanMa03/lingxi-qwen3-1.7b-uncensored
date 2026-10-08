"""Fuse adapters and quantize locally; upload only with --upload."""

import argparse
import os
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

from persona import NAME, PERSONA

ROOT = Path(__file__).resolve().parent


def build_card(run, repo, data=None):
    config = json.loads((run / "run_config.json").read_text())
    data = Path(data or config["data"])
    if not data.is_absolute():
        data = ROOT / data
    snapshot = run / "data_stats.json"
    stats = json.loads((snapshot if snapshot.exists() else data / "stats.json").read_text())
    rows = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    train = [r for r in rows if r.get("type") == "train"]
    val = [r["val_loss"] for r in rows if r.get("type") == "val"]
    return f"""---
base_model: Qwen/Qwen3-1.7B
tags: [mlx, uncensored, roleplay, emotional-support]
language: [zh, en]
library_name: mlx
---

# {NAME}

MLX 4-bit export of Qwen3-1.7B LoRA SFT for Chinese roleplay and emotional support, using MLX on Apple Silicon.
Uncensored describes the intended behavior; refusal reduction and general capability require separate evaluation.

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load({repo!r})
messages = [{{"role": "system", "content": {PERSONA!r}}},
            {{"role": "user", "content": "我失恋了，怎么走出来"}}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=512))
```

## Training

- Model: `{Path(config["model"]).name}`
- LoRA: `{json.dumps(config["lora_parameters"])}`, {config["num_layers"]} layers
- Batch: {config["batch_size"]}, gradient accumulation: {config["grad_accumulation_steps"]}
- Learning rate: {config["learning_rate"]}, maximum sequence length: {config["max_seq_length"]}
- Requested iterations in this segment: {config["iters"]}
- Last reported training loss: {train[-1]["train_loss"] if train else "not recorded"}
- Best validation loss in this segment: {min(val) if val else "not recorded"}
- Processed samples: {stats["total"]}

## Data and limitations

Training mixes `usamakenway/unified-uncensored-qwen-chatml-sft` instruction data, `shibing624/roleplay-zh-sharegpt-gpt4-data`, and `AngelWarmSmile123/deep-emotional-support-zh` dialogues.
Source counts: `{json.dumps(stats.get("by_source", {}))}`.
For historical runs these counts precede filtering; new runs record retained counts.
The local emotional-support dataset card specifies CC-BY-NC-SA-4.0. The code's Apache-2.0 license
is not a blanket license for the mixed dataset or trained weights. Model redistribution terms need review.

Refusal probes use string matching, not an independent capability benchmark.
DPO training is not implemented in this repository. A/B preferences are collected for future experiments.
"""


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs") / NAME)
    parser.add_argument("--model", help="base model location, overrides the saved path")
    parser.add_argument(
        "--data", type=Path, help="dataset location for historical runs without a stats snapshot"
    )
    parser.add_argument("--out", type=Path, default=Path("dist"))
    parser.add_argument("--repo", default=f"myy555/{NAME}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.run / "run_config.json").read_text())
    model = args.model or config["model"]
    if Path(model).is_absolute() and not Path(model).exists():
        parser.error("saved model path is unavailable; use --model to specify the current location")
    if Path(model).is_absolute():
        model = os.path.relpath(model, ROOT)
    adapters = Path(os.path.relpath(args.run / "adapters", ROOT))
    if not (adapters / "adapters.safetensors").exists():
        parser.error(f"no final adapter weights in {adapters}")
    bf16 = args.out / "bf16"
    q4 = args.out / NAME
    commands = [
        [
            sys.executable,
            "-m",
            "mlx_lm.fuse",
            "--model",
            model,
            "--adapter-path",
            str(adapters),
            "--save-path",
            str(bf16),
        ],
        [
            sys.executable,
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            str(bf16),
            "--mlx-path",
            str(q4),
            "-q",
            "--q-bits",
            "4",
        ],
    ]
    card = build_card(args.run, args.repo, args.data)
    if args.dry_run:
        import shlex

        for command in commands:
            print(shlex.join(command))
        print(card)
        return
    for command in commands:
        subprocess.run(command, check=True)
    (q4 / "README.md").write_text(card)
    print(q4)
    if args.upload:
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=args.repo, folder_path=q4)
        api.upload_folder(repo_id=args.repo, folder_path=adapters, path_in_repo="adapters")
        print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
