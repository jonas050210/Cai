"""Convert raw local files into JSONL records for the filtered corpus.

This tool does not download anything. You own the data you point it at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CODE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".c", ".h", ".cpp", ".hpp", ".cc",
    ".rs", ".go", ".java", ".kt", ".cs", ".sh", ".bash", ".sql", ".html", ".css",
    ".json", ".yaml", ".yml", ".toml",
}
TEXT_EXTS = {".md", ".txt", ".rst", ".org"}
PLANNING_HINTS = {
    "plan", "planung", "architecture", "architektur", "design", "roadmap",
    "spec", "spezifikation", "requirements", "anforderungen", "todo", "task",
}


def guess_domain(path: Path, text: str) -> str:
    name = path.name.lower()
    if path.suffix.lower() in CODE_EXTS:
        return "code"
    if any(h in name for h in PLANNING_HINTS):
        return "planning"
    head = text[:2000].lower()
    if any(h in head for h in PLANNING_HINTS):
        return "planning"
    return "general"


def guess_lang(text: str) -> str:
    """Very small heuristic. Prefer explicit metadata when possible."""
    try:
        from langdetect import detect  # type: ignore

        lang = detect(text[:4000])
        if lang in {"de", "en"}:
            return lang
    except Exception:
        pass
    lower = text.lower()
    de_hits = sum(lower.count(w) for w in [" der ", " die ", " das ", " und ", " nicht ", " mit ", " für "])
    en_hits = sum(lower.count(w) for w in [" the ", " and ", " is ", " are ", " not ", " with ", " for "])
    if de_hits > en_hits:
        return "de"
    if en_hits > 0:
        return "en"
    return "unknown"


def ingest(input_dir: Path, out_path: Path, license_name: str = "unknown", source: str = "local") -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for path in sorted(input_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in CODE_EXTS | TEXT_EXTS:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if len(text.strip()) < 40:
                continue
            record = {
                "id": hashlib.sha256(str(path).encode()).hexdigest()[:16],
                "text": text,
                "lang": guess_lang(text),
                "domain": guess_domain(path, text),
                "license": license_name,
                "source": f"{source}:{path.relative_to(input_dir)}",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local code/text files into JSONL.")
    parser.add_argument("--input", required=True, help="Directory with raw files")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--license", default="unknown", help="License metadata for all files")
    parser.add_argument("--source", default="local", help="Source label")
    args = parser.parse_args()
    n = ingest(Path(args.input), Path(args.out), args.license, args.source)
    print(f"Wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
