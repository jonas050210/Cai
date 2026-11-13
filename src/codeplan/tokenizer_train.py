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

    texts = [rec["text"] for rec in iter_jsonl(args.train) if rec.get("text")]
    if not texts:
        raise SystemExit(
            f"No training text in {args.train}: the corpus is empty. "
            "Collect data first (bash scripts/pilot.sh)."
        )
    tok = train_tokenizer(texts, args.vocab_size, args.min_frequency, args.out)
    print(f"Saved tokenizer to {args.out} (vocab_size={tok.vocab_size})")
    if tok.vocab_size < args.vocab_size * 0.5:
        print(
            f"Warning: asked for {args.vocab_size:,} tokens but only {tok.vocab_size:,} are "
            f"possible from {len(texts):,} records. The corpus is too small or too uniform."
        )


if __name__ == "__main__":
    main()
