"""Prepare chat datasets with token filtering and a reproducible split."""

import argparse
import os
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .persona import NAME, qwen_messages

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = Path("data/processed-qwen3")

IM_START = re.compile(r"<\|im_start\|>(\w+)\s*\n(.*?)<\|im_end\|>", re.S)


def parse_chatml_text(text: str):
    msgs = []
    for role, content in IM_START.findall(text):
        if role not in ("system", "user", "assistant"):
            return None
        content = content.strip()
        if not content:
            return None
        msgs.append({"role": role, "content": content})
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
        return None
    return msgs


def load_uncensored_chatml(quotas: dict, max_chars: int):
    shards = sorted((DATA / "uncensored-chatml/data").glob("train-*.parquet"))
    random.Random(42).shuffle(shards)
    need = dict(quotas)
    total_need = sum(need.values())
    rows = []
    stats = Counter()
    for shard in shards:
        if not any(v > 0 for v in need.values()):
            break
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=512):
            cols = {
                k: batch.column(k).to_pylist() for k in ("text", "source_family", "source_license")
            }
            for text, family, lic in zip(
                cols["text"], cols["source_family"], cols["source_license"], strict=True
            ):
                if need.get(family, 0) <= 0 or lic != "apache-2.0":
                    continue
                if len(text) > max_chars:
                    continue
                msgs = parse_chatml_text(text)
                if msgs is None:
                    stats["parse_fail"] += 1
                    continue
                rows.append({"messages": msgs, "src": f"chatml/{family}"})
                need[family] -= 1
                stats[f"chatml/{family}"] += 1
                if not any(v > 0 for v in need.values()):
                    break
            if not any(v > 0 for v in need.values()):
                break
    return rows, stats, total_need - sum(need.values())


def load_roleplay_zh():
    rows = []
    stats = Counter()
    for f in sorted((DATA / "roleplay-zh").glob("sharegpt_formatted_data-*.jsonl")):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                system = (d.get("system_prompt") or "").strip()
                msgs = [{"role": "system", "content": system}] if system else []
                for turn in d.get("conversations", []):
                    role = {"human": "user", "gpt": "assistant"}.get(turn["from"])
                    if role is None:
                        break
                    msgs.append({"role": role, "content": turn["value"].strip()})
                if len(msgs) >= 3 and msgs[-1]["role"] == "assistant":
                    rows.append({"messages": msgs, "src": "roleplay-zh"})
                    stats["roleplay-zh"] += 1
    return rows, stats


def clean_emotional_output(out: str) -> str:
    out = out.strip()
    if out.startswith("[思考过程]") and "[回复]" in out:
        out = out.split("[回复]", 1)[1]
    out = out.replace("[回复]", "\n\n")
    return out.strip()


def load_emotional_zh():
    rows = []
    stats = Counter()
    with open(DATA / "emotional-support-zh/emotional_support.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            user = d.get("instruction", "").strip()
            extra = (d.get("input") or "").strip()
            if extra:
                user = f"{user}\n\n{extra}"
            ans = clean_emotional_output(d.get("output", ""))
            if user and ans:
                rows.append(
                    {
                        "messages": [
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": ans},
                        ],
                        "src": "emotional-zh",
                    }
                )
                stats["emotional-zh"] += 1
    return rows, stats


def main():
    os.chdir(REPO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-1.7B")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dolphin", type=int, default=14000)
    ap.add_argument("--openhermes", type=int, default=0)
    ap.add_argument("--airoboros", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--valid-frac", type=float, default=0.02)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=1400,
        help="按 Qwen tokenizer 过滤: 全文 token 数上限, 防止训练时截断掉 assistant 部分产生 NaN",
    )
    args = ap.parse_args()

    if not 0 < args.valid_frac < 1:
        ap.error("--valid-frac must be between 0 and 1")
    if any((args.out / name).exists() for name in ("train.jsonl", "valid.jsonl", "stats.json")):
        ap.error("output already exists; choose a new --out to preserve existing splits")
    args.out.mkdir(parents=True, exist_ok=True)
    quotas = {"dolphin": args.dolphin, "openhermes": args.openhermes, "airoboros": args.airoboros}

    chatml_rows, s1, got = load_uncensored_chatml(quotas, args.max_chars)
    rp_rows, s2 = load_roleplay_zh()
    emo_rows, s3 = load_emotional_zh()

    all_rows = chatml_rows + rp_rows + emo_rows

    tok = AutoTokenizer.from_pretrained(args.model)
    # Keep answer tokens after prompt masking and sequence truncation.

    kept, dropped, duplicates, invalid_turns = [], 0, 0, 0
    seen = set()
    for r in all_rows:
        try:
            msgs = qwen_messages(r["messages"])
        except ValueError:
            invalid_turns += 1
            continue
        r["messages"] = msgs
        full = tok.apply_chat_template(msgs, add_generation_prompt=False, return_dict=False)
        prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, return_dict=False)
        if (
            len(full) > args.max_tokens
            or len(prompt) >= args.max_tokens - 24
            or len(full) - len(prompt) < 2
        ):
            dropped += 1
            continue
        key = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        kept.append(r)
    all_rows = kept

    if len(all_rows) < 2:
        raise ValueError("need at least two usable samples")
    rng = random.Random(args.seed)
    rng.shuffle(all_rows)

    n_valid = min(len(all_rows) - 1, max(64, int(len(all_rows) * args.valid_frac)))
    valid, train = all_rows[:n_valid], all_rows[n_valid:]

    for name, rows in (("train", train), ("valid", valid)):
        with open(args.out / f"{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

    lens = sorted(len(json.dumps(r["messages"], ensure_ascii=False)) for r in all_rows)
    stats = Counter(r["src"] for r in all_rows)
    summary = {
        "project": NAME,
        "model": os.path.relpath(args.model, REPO)
        if Path(args.model).is_absolute()
        else args.model,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "duplicates_removed": duplicates,
        "invalid_turns_removed": invalid_turns,
        "format": "qwen3-no-think",
        "before_filter": {**s1, **s2, **s3},
        "total": len(all_rows),
        "train": len(train),
        "valid": len(valid),
        "chatml_quota_filled": f"{got}/{sum(quotas.values())}",
        "dropped_too_long": dropped,
        "by_source": dict(stats),
        "chars_p50": lens[len(lens) // 2],
        "chars_p95": lens[int(len(lens) * 0.95)],
    }
    (args.out / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
