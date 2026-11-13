"""Corpus statistics and token-budget check.

Answers the questions that decide whether a training run is worth starting:
how many tokens do I have, how are they split across domains and languages, how
much is duplicated, and how many epochs will a target token budget need.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from .tokenizer import CodePlanTokenizer

CHARS_PER_TOKEN_ESTIMATE = 3.2


def _repo_of(source: str) -> str:
    """``github:owner/repo/path`` -> ``owner/repo``; anything else unchanged."""
    if ":" in source:
        prefix, rest = source.split(":", 1)
        parts = rest.split("/")
        if len(parts) >= 2:
            return f"{prefix}:{parts[0]}/{parts[1]}"
    return source


def analyse(
    path: str | Path,
    target_tokens: int = 0,
    tokenizer: CodePlanTokenizer | None = None,
    sample: int = 500,
) -> dict:
    path = Path(path)
    stats: Counter = Counter()
    chars_by_domain: Counter = Counter()
    chars_by_lang: Counter = Counter()
    licenses: Counter = Counter()
    sources: Counter = Counter()
    lengths: list[int] = []
    seen: set[str] = set()
    exact_duplicates = 0
    sampled_tokens = 0
    sampled_chars = 0
    token_ready = 0

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            text = str(record.get("text", ""))
            if not text:
                stats["empty"] += 1
                continue
            stats["records"] += 1
            chars = len(text)
            stats["chars"] += chars
            lengths.append(chars)

            domain = record.get("domain", "unknown")
            lang = record.get("lang") or "unknown"
            chars_by_domain[domain] += chars
            chars_by_lang[lang] += chars
            licenses[str(record.get("license", "unknown"))] += 1
            sources[_repo_of(str(record.get("source", "unknown")))] += 1

            digest = hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower().encode()).hexdigest()
            if digest in seen:
                exact_duplicates += 1
            else:
                seen.add(digest)

            if tokenizer is not None and token_ready < sample:
                ids = len(tokenizer.encode(text[:4000]))
                sampled_tokens += ids
                sampled_chars += min(len(text), 4000)
                token_ready += 1

    total_chars = max(1, stats["chars"])
    result = {
        "file": str(path),
        "records": stats["records"],
        "unique_records": len(seen),
        "exact_duplicates": exact_duplicates,
        "chars": stats["chars"],
        "estimated_tokens": int(stats["chars"] / CHARS_PER_TOKEN_ESTIMATE),
        "domains": {
            name: {"chars": value, "share": value / total_chars}
            for name, value in chars_by_domain.most_common()
        },
        "languages": {
            name: {"chars": value, "share": value / total_chars}
            for name, value in chars_by_lang.most_common()
        },
        "licenses": dict(licenses.most_common(10)),
        "sources": {"unique": len(sources), "top": dict(sources.most_common(5))},
        "length": {
            "min": min(lengths) if lengths else 0,
            "median": int(statistics.median(lengths)) if lengths else 0,
            "p90": int(statistics.quantiles(lengths, n=10)[-1]) if len(lengths) > 10 else 0,
            "max": max(lengths) if lengths else 0,
        },
    }

    if tokenizer is not None and sampled_chars > 0:
        chars_per_token = sampled_chars / sampled_tokens
        result["tokenizer"] = {
            "sample_records": token_ready,
            "chars_per_token": chars_per_token,
            "sampled_tokens": sampled_tokens,
        }
        result["estimated_tokens"] = int(stats["chars"] / chars_per_token)

    if target_tokens > 0:
        estimated = max(1, result["estimated_tokens"])
        epochs = target_tokens / estimated
        result["budget"] = {
            "target_tokens": target_tokens,
            "available_tokens": estimated,
            "epochs_needed": epochs,
            "verdict": (
                "enough data (<= 2 epochs)"
                if epochs <= 2
                else f"thin: {epochs:.1f} epochs means visible repetition"
                if epochs <= 4
                else f"too little data: {epochs:.1f} epochs, collect more"
            ),
        }
    return result


def format_report(report: dict) -> str:
    lines = [
        f"{report['file']}",
        f"  records        {report['records']:,} ({report['unique_records']:,} unique,"
        f" {report['exact_duplicates']:,} exact duplicates)",
        f"  chars          {report['chars']:,}",
        f"  est. tokens    {report['estimated_tokens']:,}",
        "  domains        "
        + ", ".join(f"{k} {v['share']:.1%}" for k, v in report["domains"].items()),
        "  languages      "
        + ", ".join(f"{k} {v['share']:.1%}" for k, v in report["languages"].items()),
        "  lengths        "
        f"min {report['length']['min']} / median {report['length']['median']}"
        f" / p90 {report['length']['p90']} / max {report['length']['max']}",
        f"  sources        {report['sources']['unique']:,} unique",
    ]
    if "tokenizer" in report:
        tok = report["tokenizer"]
        lines.append(
            f"  tokenizer      {tok['chars_per_token']:.2f} chars/token"
            f" on {tok['sample_records']} sampled records"
        )
    if "budget" in report:
        budget = report["budget"]
        lines.append(
            f"  budget         {budget['target_tokens']:,} tokens"
            f" -> {budget['epochs_needed']:.1f} epochs: {budget['verdict']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus statistics and token budget.")
    parser.add_argument("--input", required=True, help="JSONL file (raw or filtered)")
    parser.add_argument("--tokenizer", default=None, help="Measure real tokens with this tokenizer")
    parser.add_argument("--target-tokens", type=int, default=0)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--out", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    tokenizer = CodePlanTokenizer.load(args.tokenizer) if args.tokenizer else None
    report = analyse(args.input, args.target_tokens, tokenizer, args.sample)
    print(format_report(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
