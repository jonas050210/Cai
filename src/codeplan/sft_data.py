"""Turn a raw code corpus into supervised instruction pairs.

A base model can continue code but cannot answer questions. SFT needs
instruction/answer pairs, and hand-written ones do not scale. This module mines
them from the code itself:

- **implement**: a function with a docstring becomes "implement this from the
  description", the function body is the answer.
- **explain**: the same function becomes "what does this do", the docstring is
  the answer.
- **test**: a function plus tests that call it become "make these tests pass".

Everything stays grounded in real, licensed code: no model generates the targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .data import iter_jsonl

SYSTEM_PROMPT = "Antworte auf Deutsch oder Englisch. Fokus: Coding und Planung."

# file extension -> language used for parsing and for the prompt
EXTENSIONS = {
    "py": "python",
    "pyi": "python",
    "js": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "go": "go",
    "rs": "rust",
}
CODE_EXTENSIONS = set(EXTENSIONS)

PY_DEF = re.compile(r"^(?P<indent>[ \t]*)(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)\s*\(")
PY_TEST = re.compile(r"^[ \t]*(?:async\s+)?def\s+(?P<name>test_\w+)\s*\(")
JS_FUNC = re.compile(
    r"^(?P<indent>[ \t]*)(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)\s*\("
)
JS_ARROW = re.compile(
    r"^(?P<indent>[ \t]*)(?:export\s+)?const\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s*)?\([^)]*\)\s*=>"
)
GO_FUNC = re.compile(
    r"^(?P<indent>[ \t]*)func\s+(?:\([^)]*\)\s*)?(?P<name>[A-Za-z_]\w*)\s*\("
)
RUST_FUNC = re.compile(
    r"^(?P<indent>[ \t]*)(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*[<(]"
)
FUNC_PATTERNS = {
    "python": [PY_DEF],
    "javascript": [JS_FUNC, JS_ARROW],
    "typescript": [JS_FUNC, JS_ARROW],
    "go": [GO_FUNC],
    "rust": [RUST_FUNC],
}
BRACE_LANGUAGES = {"javascript", "typescript", "go", "rust"}
DOCSTRING_START = re.compile(r'^[ \t]*(?:[rRuU])?("""|\'\'\')')
COMMENT_BLOCK = re.compile(r"/\*\*(.*?)\*/", re.S)
LINE_COMMENT = re.compile(r"^\s*(?://+|#)\s?(.*)$")


@dataclass
class Function:
    name: str
    language: str
    code: str
    description: str
    source: str
    kind: str = "function"
    tests: str = field(default="")


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _block(lines: list[str], start: int, brace: bool = False) -> tuple[list[str], int]:
    """Return the lines of the block that starts at ``start`` and the next index."""
    if brace:
        return _brace_block(lines, start)
    base = _indent_of(lines[start])
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and _indent_of(line) <= base:
            break
        end += 1
    return lines[start:end], end


def _brace_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect a ``{ ... }`` block by counting braces (comments/strings ignored)."""
    depth = 0
    opened = False
    end = start
    while end < len(lines):
        line = _strip_noise(lines[end])
        depth += line.count("{") - line.count("}")
        if "{" in line:
            opened = True
        end += 1
        if opened and depth <= 0:
            break
        if end - start > 80:
            break
    return lines[start:end], end


def _strip_noise(line: str) -> str:
    """Remove string literals and line comments so braces are counted correctly."""
    line = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)
    line = re.sub(r"'(?:[^'\\]|\\.)*'", "''", line)
    line = re.sub(r"`(?:[^`\\]|\\.)*`", "``", line)
    return line.split("//", 1)[0]


def _docstring(block: list[str]) -> str:
    """Extract a Python docstring from a function block."""
    body = block[1:]
    if not body:
        return ""
    first = body[0]
    match = DOCSTRING_START.match(first)
    if not match:
        return ""
    quote = match.group(1)
    text = first.split(quote, 1)[1]
    if quote in text:
        # single line docstring: """text"""
        return text.split(quote)[0].strip()
    pieces = [text.strip()]
    for line in body[1:]:
        if quote in line:
            pieces.append(line.split(quote)[0].strip())
            break
        pieces.append(line.strip())
    return "\n".join(piece for piece in pieces if piece).strip()


def _description_from_comments(lines: list[str], start: int) -> str:
    """Collect a JSDoc block or a run of line comments directly above a line."""
    index = start - 1
    while index >= 0 and not lines[index].strip():
        index -= 1
    if index < 0:
        return ""
    candidate = lines[index].strip()
    if candidate.endswith("*/"):
        collected: list[str] = []
        while index >= 0:
            collected.append(lines[index])
            if "/*" in lines[index]:
                break
            index -= 1
        block = "\n".join(reversed(collected))
        match = COMMENT_BLOCK.search(block)
        if not match:
            return ""
        return _clean_comment(match.group(1))
    collected = []
    while index >= 0:
        match = LINE_COMMENT.match(lines[index])
        if not match:
            break
        collected.append(match.group(1))
        index -= 1
    return _clean_comment("\n".join(reversed(collected)))


def _clean_comment(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip().lstrip("*").strip()
        if line.startswith("@"):  # JSDoc tags are not natural language
            continue
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _valid_description(text: str, min_chars: int, max_chars: int) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    lowered = text.lower()
    return not any(marker in lowered for marker in ("todo", "fixme", "xxx", "type: ignore"))


def language_of(record: dict) -> str | None:
    """Guess the language from the file extension; None for non-code files."""
    for key in ("language", "lang_hint"):
        value = str(record.get(key) or "").lower()
        if value in EXTENSIONS.values():
            return value
        if value in ("ts", "tsx"):
            return "typescript"
        if value in ("js", "jsx"):
            return "javascript"
    path = str(record.get("path") or record.get("source") or "")
    if "." not in path:
        return None
    return EXTENSIONS.get(path.rsplit(".", 1)[-1].lower().strip())


def extract_functions(
    text: str,
    source: str = "",
    language: str = "python",
    min_description: int = 20,
    max_description: int = 1200,
    max_code_lines: int = 80,
) -> list[Function]:
    """Find documented functions in a source file."""
    lines = text.splitlines()
    found: list[Function] = []
    is_python = language == "python"
    patterns = FUNC_PATTERNS.get(language, [PY_DEF])
    brace = language in BRACE_LANGUAGES

    index = 0
    while index < len(lines):
        match = None
        for pattern in patterns:
            match = pattern.match(lines[index])
            if match:
                break
        if not match:
            index += 1
            continue
        block, next_index = _block(lines, index, brace=brace)
        index = max(next_index, index + 1)
        if len(block) < 3 or len(block) > max_code_lines:
            continue
        code = "\n".join(line.rstrip() for line in block).rstrip()
        if is_python:
            description = _docstring(block)
        else:
            description = _description_from_comments(lines, next_index - len(block))
        if not _valid_description(description, min_description, max_description):
            continue
        tests = _find_tests(lines, match.group("name")) if is_python else ""
        found.append(
            Function(
                name=match.group("name"),
                language=language,
                code=code,
                description=description,
                source=source,
                tests=tests,
            )
        )
    return found


def _find_tests(lines: list[str], name: str) -> str:
    """Collect test functions that call ``name``."""
    blocks: list[str] = []
    for index, line in enumerate(lines):
        if not PY_TEST.match(line):
            continue
        block, _ = _block(lines, index)
        body = "\n".join(block)
        if f"{name}(" in body and len(block) <= 40:
            blocks.append(body)
        if len(blocks) >= 2:
            break
    return "\n\n".join(blocks)


def _signature(code: str, language: str) -> str:
    first = code.splitlines()[0].strip() if code else ""
    return first


def build_pairs(function: Function, max_pairs_per_kind: int = 1) -> list[dict]:
    """Create instruction/answer records from one function."""
    pairs: list[dict] = []
    signature = _signature(function.code, function.language)
    first_sentence = function.description.splitlines()[0].strip()

    if function.description and len(function.code.splitlines()) >= 2:
        pairs.append(
            {
                "kind": "implement",
                "user": (
                    f"Implementiere diese Funktion in {function.language}.\n\n"
                    f"Signatur:\n{signature}\n\nBeschreibung:\n{function.description}"
                ),
                "plan": first_sentence,
                "code": function.code,
                "answer": function.code,
            }
        )
        pairs.append(
            {
                "kind": "explain",
                "user": (
                    f"Erkläre in 1-3 Sätzen, was dieser Code macht:\n\n"
                    f"```{function.language}\n{function.code}\n```"
                ),
                "plan": "Code lesen, Zweck und Rückgabewert benennen.",
                "answer": function.description,
            }
        )

    if function.tests:
        pairs.append(
            {
                "kind": "test",
                "user": (
                    "Diese Tests müssen bestehen. Implementiere die fehlende Funktion.\n\n"
                    f"```python\n{function.tests}\n```"
                ),
                "plan": "Tests lesen, erwartetes Verhalten ableiten, Funktion schreiben.",
                "tests": function.tests,
                "code": function.code,
                "answer": function.code,
            }
        )

    return pairs[: max(1, max_pairs_per_kind) * 3]


def _fingerprint(pair: dict) -> str:
    return hashlib.sha256((pair["user"] + "\x00" + pair.get("answer", "")).encode()).hexdigest()


def _split(key: str, val_ratio: float) -> str:
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def generate(
    input_path: str | Path,
    out_dir: str | Path,
    max_pairs: int = 50000,
    val_ratio: float = 0.02,
    min_description: int = 20,
    kinds: tuple[str, ...] = ("implement", "explain", "test"),
) -> dict:
    """Mine SFT pairs from a filtered JSONL corpus."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts = {"records": 0, "files_with_functions": 0, "pairs": 0, "duplicates": 0}
    per_kind: dict[str, int] = {}
    per_language: dict[str, int] = {}

    handles = {
        name: open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") for name in ("train", "val")
    }
    try:
        for record in iter_jsonl(input_path):
            counts["records"] += 1
            text = record.get("text") or ""
            language = language_of(record)
            if language is None or not text:
                counts["skipped_files"] = counts.get("skipped_files", 0) + 1
                continue
            functions = extract_functions(
                text,
                source=str(record.get("source", "")),
                language=language,
                min_description=min_description,
            )
            if not functions:
                continue
            counts["files_with_functions"] += 1
            for function in functions:
                for pair in build_pairs(function):
                    if pair["kind"] not in kinds:
                        continue
                    fingerprint = _fingerprint(pair)
                    if fingerprint in seen:
                        counts["duplicates"] += 1
                        continue
                    seen.add(fingerprint)
                    pair["system"] = SYSTEM_PROMPT
                    pair["source"] = function.source
                    pair["language"] = function.language
                    split = _split(fingerprint, val_ratio)
                    handles[split].write(json.dumps(pair, ensure_ascii=False) + "\n")
                    counts["pairs"] += 1
                    per_kind[pair["kind"]] = per_kind.get(pair["kind"], 0) + 1
                    per_language[function.language] = per_language.get(function.language, 0) + 1
                    if counts["pairs"] >= max_pairs:
                        raise StopIteration
    except StopIteration:
        pass
    finally:
        for handle in handles.values():
            handle.close()

    counts["per_kind"] = per_kind
    counts["per_language"] = per_language
    counts["out_dir"] = str(out_dir)
    (out_dir / "report.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine SFT pairs (implement/explain/test) from code.")
    parser.add_argument("--input", required=True, help="Filtered train.jsonl")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--min-description", type=int, default=20)
    parser.add_argument(
        "--kinds",
        default="implement,explain,test",
        help="Comma separated subset of implement,explain,test",
    )
    args = parser.parse_args()

    kinds = tuple(kind.strip() for kind in args.kinds.split(",") if kind.strip())
    unknown = [kind for kind in kinds if kind not in ("implement", "explain", "test")]
    if unknown:
        raise SystemExit(f"Unknown kinds: {unknown}")
    if not Path(args.input).exists():
        raise SystemExit(f"Input not found: {args.input}")

    report = generate(
        args.input,
        args.out_dir,
        max_pairs=args.max_pairs,
        val_ratio=args.val_ratio,
        min_description=args.min_description,
        kinds=kinds,
    )
    print(json.dumps(report, indent=2))
    if report["pairs"] == 0:
        raise SystemExit("No SFT pairs could be mined; the corpus has too few documented functions.")


if __name__ == "__main__":
    main()
