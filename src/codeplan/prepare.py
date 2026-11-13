"""Filter, deduplicate, and split JSONL records.

Hard rule for this project: natural-language records must be German or English.
Code is allowed to contain identifiers/keywords from programming languages, but
comments and docs should still be DE/EN. The filter is conservative, not perfect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from .dedup import NearDuplicateFilter, sketch

ALLOWED_LANGS = {"de", "en"}
ALLOWED_DOMAINS = {"code", "planning", "general"}
DISALLOWED_SCRIPTS = {
    "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "DEVANAGARI", "THAI",
    "HANGUL", "HIRAGANA", "KATAKANA", "CJK", "HAN", "BENGALI", "TAMIL",
}


# Unicode ranges that are rejected by the corpus policy. Checking ranges is
# ~40x faster than unicodedata.name() on every letter of every document.
_DISALLOWED_RANGES = (
    (0x0370, 0x03FF),  # Greek
    (0x0400, 0x052F),  # Cyrillic
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0B80, 0x0BFF),  # Tamil
    (0x0E00, 0x0E7F),  # Thai
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility
)


def is_disallowed_char(ch: str) -> bool:
    code = ord(ch)
    for low, high in _DISALLOWED_RANGES:
        if low <= code <= high:
            return True
    return False


def script_name(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "UNKNOWN"
    for token in DISALLOWED_SCRIPTS:
        if token in name:
            return token
    return "LATIN_OR_COMMON"


def bad_script_ratio(text: str) -> float:
    letters = 0
    bad = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        if is_disallowed_char(ch):
            bad += 1
    return bad / letters if letters else 0.0


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def heuristic_lang(text: str) -> str:
    lower = " " + text.lower() + " "
    de = sum(lower.count(w) for w in [" der ", " die ", " das ", " und ", " nicht ", " mit ", " für ", " eine ", " werden "])
    en = sum(lower.count(w) for w in [" the ", " and ", " is ", " are ", " not ", " with ", " for ", " this ", " that ", " you "])
    if de > en and de >= 2:
        return "de"
    if en > de and en >= 2:
        return "en"
    return "unknown"


def detect_lang(text: str, declared: str | None = None) -> str:
    if declared in ALLOWED_LANGS:
        return declared
    try:
        from langdetect import DetectorFactory, detect  # type: ignore

        DetectorFactory.seed = 0
        lang = detect(text[:4000])
        if lang in ALLOWED_LANGS:
            return lang
    except Exception:
        pass
    return heuristic_lang(text)


COMMENT_PATTERNS = [
    re.compile(r'"""(.*?)"""', re.DOTALL),
    re.compile(r"'''(.*?)'''", re.DOTALL),
    re.compile(r"/\*(.*?)\*/", re.DOTALL),
    re.compile(r"//\s*(.+)"),
    re.compile(r"#\s*(.+)"),
    re.compile(r"--\s*(.+)"),
]


def extract_natural_language(text: str) -> str:
    """Pull comments, docstrings and similar prose out of source code."""
    parts: list[str] = []
    for pattern in COMMENT_PATTERNS:
        parts.extend(pattern.findall(text))
    return "\n".join(part.strip() for part in parts if part.strip())


def detect_lang_for_record(text: str, declared: str | None, domain: str) -> tuple[str, str]:
    """Return ``(lang, source)`` with source in declared/detected/code-default.

    Code records are judged on their comments and docstrings only. A file that
    contains no prose is language-neutral instead of a language-filter failure;
    rejecting it would bias the corpus towards comment-heavy code.
    """
    if declared in ALLOWED_LANGS:
        return declared, "declared"

    prose = extract_natural_language(text) if domain == "code" else text
    if prose.strip():
        lang = detect_lang(prose)
        if lang in ALLOWED_LANGS:
            return lang, "detected"
        if domain != "code" or len(prose.split()) >= 8:
            return lang, "detected"
    elif domain == "code":
        return "en", "code-default"

    lang = detect_lang(text)
    if lang in ALLOWED_LANGS:
        return lang, "detected"
    if domain == "code" and len(prose.split()) < 8:
        return "en", "code-default"
    return lang, "detected"


def record_hash(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def _filter_params(
    min_chars: int,
    max_chars: int,
    max_bad_script_ratio: float,
    require_license: bool,
    sketch_config: dict | None,
) -> dict:
    return {
        "min_chars": min_chars,
        "max_chars": max_chars,
        "max_bad_script_ratio": max_bad_script_ratio,
        "require_license": require_license,
        "sketch": sketch_config,
    }


def _filter_one(payload: tuple[str, dict]) -> tuple[str, dict | None, object]:
    """Phase 1 of the pipeline: everything that does not need global state.

    Returns ``(status, record, sketch)``. Runs inside worker processes, so it
    must not touch the dedup indexes - those stay in the parent.
    """
    line, params = payload
    line = line.strip()
    if not line:
        return "skip", None, None
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return "bad_json", None, None

    text = normalize_text(str(rec.get("text", "")))
    if len(text) < params["min_chars"]:
        return "too_short", None, None
    if len(text) > params["max_chars"]:
        return "too_long", None, None
    if bad_script_ratio(text) > params["max_bad_script_ratio"]:
        return "bad_script", None, None

    domain = rec.get("domain", "general")
    if domain not in ALLOWED_DOMAINS:
        return "bad_domain", None, None

    lang, lang_source = detect_lang_for_record(text, rec.get("lang"), domain)
    if lang not in ALLOWED_LANGS:
        return "bad_lang", None, None

    license_name = rec.get("license", "unknown")
    if params["require_license"] and license_name in {"unknown", "", None}:
        return "missing_license", None, None

    h = record_hash(text)
    out = {
        "id": rec.get("id", h[:16]),
        "text": text,
        "lang": lang,
        "lang_source": lang_source,
        "domain": domain,
        "license": license_name,
        "source": rec.get("source", "unknown"),
        "hash": h,
    }
    sig = None
    config = params["sketch"]
    if config is not None and len(text) >= config["min_chars"]:
        sig = sketch(
            text,
            window=config["window"],
            num_perm=config["num_perm"],
            max_chars=config["max_chars"],
        )
    return "ok", out, sig


def _split_for(record_hash_value: str, val_ratio: float, test_ratio: float) -> str:
    bucket = int(record_hash_value[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def prepare(
    input_path: Path,
    out_dir: Path,
    min_chars: int = 40,
    max_chars: int = 200_000,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    max_bad_script_ratio: float = 0.02,
    require_license: bool = False,
    near_dedup: NearDuplicateFilter | None = None,
    workers: int = 1,
) -> dict:
    """Filter, deduplicate and split a corpus.

    ``workers > 1`` runs the expensive per-record work (normalisation, script
    check, language detection, MinHash) in a process pool. Deduplication stays in
    the parent process so the result is byte-identical to ``workers=1``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    stats = Counter()
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    sketch_config = None
    if near_dedup is not None:
        sketch_config = {
            "window": near_dedup.window,
            "num_perm": near_dedup.num_perm,
            "max_chars": near_dedup.max_chars,
            "min_chars": near_dedup.min_chars,
        }
    params = _filter_params(
        min_chars, max_chars, max_bad_script_ratio, require_license, sketch_config
    )

    def lines() -> Iterator[str]:
        with open(input_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stats["input"] += 1
                yield line

    def consume(status: str, out: dict | None, sig: object) -> None:
        if status != "ok" or out is None:
            if status != "skip":
                stats[status] += 1
            return
        h = out["hash"]
        if h in seen:
            stats["duplicate"] += 1
            return
        if near_dedup is not None:
            if sig is None and len(out["text"]) < near_dedup.min_chars:
                near_dedup.skipped += 1
            if near_dedup.is_duplicate_sketch(sig):  # type: ignore[arg-type]
                stats["near_duplicate"] += 1
                return
        seen.add(h)
        split = _split_for(h, val_ratio, test_ratio)
        splits[split].append(out)
        stats[f"kept_{split}"] += 1
        stats[f"domain_{out['domain']}"] += 1
        stats[f"lang_{out['lang']}"] += 1
        stats[f"lang_source_{out['lang_source']}"] += 1

    if workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            jobs = ((line, params) for line in lines())
            for status, out, sig in pool.imap(_filter_one, jobs, chunksize=16):
                consume(status, out, sig)
    else:
        for line in lines():
            consume(*_filter_one((line, params)))

    for split, records in splits.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats["kept_total"] = sum(len(v) for v in splits.values())
    stats_dict = dict(stats)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return stats_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and split JSONL corpus.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--max-bad-script-ratio", type=float, default=0.02)
    parser.add_argument("--require-license", action="store_true")
    parser.add_argument(
        "--near-dedup",
        action="store_true",
        help="Also drop near-duplicates (MinHash + LSH, Jaccard threshold)",
    )
    parser.add_argument("--near-dedup-threshold", type=float, default=0.8)
    parser.add_argument("--near-dedup-max-chars", type=int, default=50_000)
    parser.add_argument("--near-dedup-perm", type=int, default=128)
    parser.add_argument("--near-dedup-bands", type=int, default=32)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Processes for filtering (default: all cores except one)",
    )
    args = parser.parse_args()
    near_dedup = None
    if args.near_dedup:
        near_dedup = NearDuplicateFilter(
            threshold=args.near_dedup_threshold,
            num_perm=args.near_dedup_perm,
            bands=args.near_dedup_bands,
            max_chars=args.near_dedup_max_chars,
        )
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    stats = prepare(
        Path(args.input), Path(args.out_dir), args.min_chars, args.max_chars,
        args.val_ratio, args.test_ratio, args.max_bad_script_ratio, args.require_license,
        near_dedup=near_dedup, workers=workers if workers > 1 else 1,
    )
    if near_dedup is not None:
        stats["near_dedup"] = near_dedup.stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
