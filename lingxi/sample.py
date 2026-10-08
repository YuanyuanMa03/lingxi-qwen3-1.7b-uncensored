"""Chat with a base model, merged model, or LoRA adapter."""

import argparse
import os
from pathlib import Path

from mlx_lm import generate, load

from .persona import PERSONA

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None, help="可选: 直接挂 LoRA adapter 测试")
    ap.add_argument("--prompt", default=None, help="非交互模式: 单条提问")
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter)

    def chat(history, user_text):
        history = history + [{"role": "user", "content": user_text}]
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": PERSONA}] + history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        out = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        print(f"\n模型: {out.strip()}\n")
        return history + [{"role": "assistant", "content": out.strip()}]

    if args.prompt:
        chat([], args.prompt)
        return

    print("模型已就绪 (Ctrl-C 退出)\n")
    history = []
    while True:
        try:
            user = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        history = chat(history, user)


if __name__ == "__main__":
    main()
