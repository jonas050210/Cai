"""Tokenizer fertility analysis for German/English and code.

Trains byte-level BPE tokenizers at several vocabulary sizes on a sample of the
filtered corpus and reports tokens per character / per word per domain and
language. This is how the 16k-vs-32k decision should be made: at ~50M
parameters a large vocabulary eats a big share of the budget.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .data import iter_jsonl
from .tokenizer import CodePlanTokenizer, train_tokenizer


def load_records(path: str | Path, limit: int) -> list[dict]:
    records = []
    for rec in iter_jsonl(path):
        if not rec.get("text"):
            continue
        records.append(rec)
        if limit > 0 and len(records) >= limit:
            break
    return records


def measure(tokenizer: CodePlanTokenizer, records: Iterable[dict]) -> dict:
    """Return fertility statistics grouped by ``domain/lang`` plus an overall row."""
    totals: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0])  # tokens, chars, words
    for rec in records:
        text = rec["text"]
        key = f"{rec.get('domain', 'general')}/{rec.get('lang', 'unknown')}"
        n_tokens = len(tokenizer.encode(text))
        stats = totals[key]
        stats[0] += n_tokens
        stats[1] += len(text)
        stats[2] += max(1, len(text.split()))

    groups = {}
    combined = [0, 0, 0]
    for key, (tokens, chars, words) in sorted(totals.items()):
        groups[key] = {
            "tokens": tokens,
            "chars": chars,
            "words": words,
            "tokens_per_char": tokens / max(1, chars),
            "tokens_per_word": tokens / max(1, words),
            "chars_per_token": chars / max(1, tokens),
        }
        combined[0] += tokens
        combined[1] += chars
        combined[2] += words
    overall = {
        "tokens": combined[0],
        "chars": combined[1],
        "words": combined[2],
        "tokens_per_char": combined[0] / max(1, combined[1]),
        "tokens_per_word": combined[0] / max(1, combined[2]),
        "chars_per_token": combined[1] / max(1, combined[0]),
    }
    return {"overall": overall, "groups": groups}


def non_embedding_params(d_model: int, n_layers: int, intermediate_size: int) -> int:
    """Transformer parameters without the embedding matrix."""
    per_layer = 4 * d_model**2 + 3 * d_model * intermediate_size
    return n_layers * per_layer + d_model


def recommend(
    results: list[dict],
    tolerance: float = 0.05,
    d_model: int = 512,
    n_layers: int = 10,
    intermediate_size: int = 1408,
) -> dict:
    """Pick the vocabulary with the lowest compute cost for the same text.

    Cost per character is proportional to ``total_params * tokens_per_char``:
    more tokens per character means more forward passes over the whole model,
    while a bigger vocabulary means more parameters on every one of them.
    Comparing tokens/char alone always favours the largest vocabulary.
    """
    base = non_embedding_params(d_model, n_layers, intermediate_size)
    for result in results:
        total_params = base + result["vocab_size"] * d_model
        result["total_params"] = total_params
        result["cost_index"] = total_params * result["overall"]["tokens_per_char"]

    best = min(result["cost_index"] for result in results)
    for result in sorted(results, key=lambda x: x["vocab_size"]):
        if result["cost_index"] <= best * (1.0 + tolerance):
            return {
                "vocab_size": result["vocab_size"],
                "rule": (
                    f"smallest vocab within {tolerance:.0%} of the lowest"
                    " params x tokens/char cost"
                ),
                "cost_index": result["cost_index"],
                "best_cost_index": best,
                "tokens_per_char": result["overall"]["tokens_per_char"],
            }
    return {"vocab_size": max(r["vocab_size"] for r in results), "rule": "fallback"}


def analyse(
    train_path: str | Path,
    vocab_sizes: list[int],
    d_model: int = 512,
    sample_records: int = 5000,
    eval_records: int = 2000,
    min_frequency: int = 2,
    n_layers: int = 10,
    intermediate_size: int = 1408,
) -> dict:
    if sample_records <= 0 or eval_records <= 0:
        raise ValueError("sample_records and eval_records must be positive")
    rows = load_records(train_path, sample_records + eval_records)
    if len(rows) < 2:
        raise ValueError(f"Not enough records in {train_path}")
    split = min(sample_records, len(rows) - 1)
    train_rows, eval_rows = rows[:split], rows[split:]

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for vocab_size in vocab_sizes:
            tokenizer = train_tokenizer(
                (r["text"] for r in train_rows),
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                out_path=Path(tmp) / f"tok_{vocab_size}.json",
            )
            stats = measure(tokenizer, eval_rows)
            stats["vocab_size"] = tokenizer.vocab_size
            stats["requested_vocab_size"] = vocab_size
            stats["train_records"] = len(train_rows)
            stats["eval_records"] = len(eval_rows)
            stats["embedding_params"] = tokenizer.vocab_size * d_model
            results.append(stats)

    return {
        "file": str(train_path),
        "d_model": d_model,
        "non_embedding_params": non_embedding_params(d_model, n_layers, intermediate_size),
        "results": results,
        "recommendation": recommend(results, d_model=d_model, n_layers=n_layers,
                                    intermediate_size=intermediate_size),
    }


def format_table(report: dict) -> str:
    lines = ["vocab   tok/char  char/tok  tok/word  emb params   total params   cost index"]
    for r in report["results"]:
        o = r["overall"]
        lines.append(
            f"{r['vocab_size']:>6}  {o['tokens_per_char']:8.3f}  {o['chars_per_token']:8.2f}"
            f"  {o['tokens_per_word']:8.2f}  {r['embedding_params'] / 1e6:8.2f}M"
            f"  {r.get('total_params', 0) / 1e6:11.2f}M"
            f"  {r.get('cost_index', 0) / 1e6:11.2f}M"
        )
    rec = report["recommendation"]
    lines.append(
        f"recommended vocab: {rec['vocab_size']} ({rec.get('rule', '')})"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer fertility analysis.")
    parser.add_argument("--train", required=True, help="Filtered train.jsonl")
    parser.add_argument("--vocab-sizes", default="16000,32000")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--sample-records", type=int, default=5000)
    parser.add_argument("--eval-records", type=int, default=2000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=10, help="For the parameter cost")
    parser.add_argument("--intermediate-size", type=int, default=1408)
    parser.add_argument("--out", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",") if v.strip()]
    report = analyse(
        args.train,
        vocab_sizes,
        d_model=args.d_model,
        sample_records=args.sample_records,
        eval_records=args.eval_records,
        min_frequency=args.min_frequency,
        n_layers=args.n_layers,
        intermediate_size=args.intermediate_size,
    )
    print(format_table(report))
    for result in report["results"]:
        print()
        print(f"vocab {result['vocab_size']}:")
        for group, g in result["groups"].items():
            print(
                f"  {group:<16} tok/char {g['tokens_per_char']:.3f} | "
                f"char/tok {g['chars_per_token']:.2f} | tok/word {g['tokens_per_word']:.2f}"
            )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
