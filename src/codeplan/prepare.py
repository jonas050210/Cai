"""Filter, deduplicate, and split JSONL records.

Hard rule for this project: natural-language records must be German or English.
Code is allowed to contain identifiers/keywords from programming languages, but
comments and docs should still be DE/EN. The filter is conservative, not perfect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ALLOWED_LANGS = {"de", "en"}
ALLOWED_DOMAINS = {"code", "planning", "general"}
DISALLOWED_SCRIPTS = {
    "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "DEVANAGARI", "THAI",
    "HANGUL", "HIRAGANA", "KATAKANA", "CJK", "HAN", "BENGALI", "TAMIL",
}


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
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    bad = 0
    for c in letters:
        if script_name(c) != "LATIN_OR_COMMON":
            bad += 1
    return bad / len(letters)


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


def prepare(
    input_path: Path,
    out_dir: Path,
    min_chars: int = 40,
    max_chars: int = 200_000,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    max_bad_script_ratio: float = 0.02,
    require_license: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    stats = Counter()
    splits = {"train": [], "val": [], "test": []}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["input"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            text = normalize_text(str(rec.get("text", "")))
            if len(text) < min_chars:
                stats["too_short"] += 1
                continue
            if len(text) > max_chars:
                stats["too_long"] += 1
                continue
            if bad_script_ratio(text) > max_bad_script_ratio:
                stats["bad_script"] += 1
                continue
            domain = rec.get("domain", "general")
            if domain not in ALLOWED_DOMAINS:
                stats["bad_domain"] += 1
                continue
            lang, lang_source = detect_lang_for_record(text, rec.get("lang"), domain)
            if lang not in ALLOWED_LANGS:
                stats["bad_lang"] += 1
                continue
            license_name = rec.get("license", "unknown")
            if require_license and license_name in {"unknown", "", None}:
                stats["missing_license"] += 1
                continue
            h = record_hash(text)
            if h in seen:
                stats["duplicate"] += 1
                continue
            seen.add(h)
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
            bucket = int(h[:8], 16) / 0xFFFFFFFF
            if bucket < test_ratio:
                split = "test"
            elif bucket < test_ratio + val_ratio:
                split = "val"
            else:
                split = "train"
            splits[split].append(out)
            stats[f"kept_{split}"] += 1
            stats[f"domain_{domain}"] += 1
            stats[f"lang_{lang}"] += 1
            stats[f"lang_source_{lang_source}"] += 1

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
    args = parser.parse_args()
    stats = prepare(
        Path(args.input), Path(args.out_dir), args.min_chars, args.max_chars,
        args.val_ratio, args.test_ratio, args.max_bad_script_ratio, args.require_license,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
