"""Simple interactive chat CLI."""

from __future__ import annotations

import argparse

import torch

from .generate import load_model_from_checkpoint
from .model import generate
from .tokenizer import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
    EOS_TOKEN,
    SYSTEM_END_TOKEN,
    SYSTEM_TOKEN,
    USER_END_TOKEN,
    USER_TOKEN,
)


def build_prompt(tokenizer, system: str, history: list[tuple[str, str]], user: str) -> str:
    parts = [f"{BOS_TOKEN}{SYSTEM_TOKEN}{system}{SYSTEM_END_TOKEN}"]
    for prev_user, prev_assistant in history[-4:]:
        parts.append(f"{USER_TOKEN}{prev_user}{USER_END_TOKEN}{ASSISTANT_TOKEN}{prev_assistant}{EOS_TOKEN}")
    parts.append(f"{USER_TOKEN}{user}{USER_END_TOKEN}{ASSISTANT_TOKEN}")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with CodePlan-LM.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, tokenizer, _ = load_model_from_checkpoint(args.checkpoint, args.tokenizer, args.device)
    device = next(model.parameters()).device
    history: list[tuple[str, str]] = []

    print("CodePlan-LM chat. Type /exit to quit, /reset to clear history.")
    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history = []
            print("History cleared.")
            continue

        prompt = build_prompt(tokenizer, args.system, history, user)
        ids = tokenizer.encode(prompt)
        max_len = model.cfg.max_seq_len
        if len(ids) > max_len - args.max_new_tokens:
            ids = ids[-(max_len - args.max_new_tokens):]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        out = generate(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_id,
        )
        new_tokens = out[0, input_ids.shape[1]:].tolist()
        answer = tokenizer.decode(new_tokens, skip_special=True).strip()
        print(f"\nLM> {answer}")
        history.append((user, answer))


if __name__ == "__main__":
    main()
