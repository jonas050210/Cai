"""Train the project's BPE tokenizer from scratch."""

from __future__ import annotations

import argparse

from .data import iter_jsonl
from .tokenizer import train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CodePlan-LM tokenizer from scratch.")
    parser.add_argument("--train", required=True, help="Filtered train.jsonl")
    parser.add_argument("--out", default="artifacts/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    texts = (rec["text"] for rec in iter_jsonl(args.train))
    tok = train_tokenizer(texts, args.vocab_size, args.min_frequency, args.out)
    print(f"Saved tokenizer to {args.out} (vocab_size={tok.vocab_size})")


if __name__ == "__main__":
    main()
