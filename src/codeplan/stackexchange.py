"""Stack Exchange questions and answers as text (CC BY-SA 4.0).

Raw code teaches syntax; questions teach *judgement* - "why is X slower",
"when should I use Y". The public API is enough, no multi-gigabyte dump.

The output uses the collector's JSONL shape, plus an answer field so
``codeplan-sft-data`` can turn the top answers into instruction pairs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LICENSE = "cc-by-sa-4.0"
API = "https://api.stackexchange.com/2.3"
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"[ \t]+")


def to_text(body: str) -> str:
    """Strip HTML to readable plain text (no parser dependency)."""
    text = body.replace("<br>", "\n").replace("<br/>", "\n").replace("</p>", "\n")
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    text = SPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _get(path: str, params: dict[str, str], timeout: int = 60) -> dict:
    url = f"{API}/{path}?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"User-Agent": "codeplan-stackexchange/0.1", "Accept-Encoding": "gzip"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return json.loads(raw)


def top_questions(site: str, tag: str, count: int) -> list[dict]:
    params = {
        "site": site,
        "tagged": tag,
        "sort": "votes",
        "order": "desc",
        "pagesize": str(min(100, count)),
        "filter": "withbody",
    }
    payload = _get("questions", params)
    questions = []
    for item in payload.get("items", []):
        questions.append(
            {
                "id": item["question_id"],
                "title": html.unescape(item["title"]),
                "body": to_text(item.get("body", "")),
                "score": item.get("score", 0),
                "link": item.get("link", ""),
                "tags": item.get("tags", []),
            }
        )
    return questions


def top_answer(site: str, question_id: int, count: int = 1) -> str:
    payload = _get(
        f"questions/{question_id}/answers",
        {
            "site": site,
            "sort": "votes",
            "order": "desc",
            "pagesize": str(count),
            "filter": "withbody",
        },
    )
    items = payload.get("items", [])
    return to_text(items[0].get("body", "")) if items else ""


def record(question: dict, answer: str, site: str) -> dict:
    text = question["title"] + "\n\n" + question["body"]
    if answer:
        text += "\n\nAnswer:\n" + answer
    digest = hashlib.sha256(f"se:{site}:{question['id']}".encode()).hexdigest()[:16]
    return {
        "id": digest,
        "text": text,
        "lang": "en",
        "domain": "planning",
        "license": LICENSE,
        "source": f"stackexchange:{site}/{question['id']}",
        "title": question["title"],
        "answer": answer,
        "score": question["score"],
    }


def fetch(
    site: str, tags: list[str], per_tag: int, out_path: Path, with_answers: bool = True
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    chars = 0
    seen: set[int] = set()
    with open(out_path, "w", encoding="utf-8") as out:
        for tag in tags:
            if per_tag <= 0:
                break
            try:
                questions = top_questions(site, tag, per_tag)
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                print(f"  {tag}: request failed ({exc}), skipping")
                continue
            for question in questions[:per_tag]:
                if question["id"] in seen:
                    continue
                seen.add(question["id"])
                answer = ""
                if with_answers:
                    try:
                        answer = top_answer(site, question["id"])
                    except (urllib.error.URLError, TimeoutError, ValueError):
                        answer = ""
                rec = record(question, answer, site)
                if len(rec["text"]) < 400:
                    continue
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                chars += len(rec["text"])
            out.flush()
            print(f"  {tag}: {written} records so far, {chars / 1e6:.1f}M chars", flush=True)
    return {"records": written, "chars": chars, "out": str(out_path), "license": LICENSE}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Stack Exchange Q&A as JSONL.")
    parser.add_argument("--out", default="data/raw/stackexchange.jsonl")
    parser.add_argument("--site", default="stackoverflow")
    parser.add_argument(
        "--tags",
        default="python,javascript,typescript,java,go,rust,git,sql,docker,linux",
        help="Comma separated tags",
    )
    parser.add_argument("--per-tag", type=int, default=60)
    parser.add_argument("--no-answers", action="store_true")
    args = parser.parse_args()

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    print(f"Fetching {args.per_tag} questions per tag from {args.site}: {', '.join(tags)}")
    result = fetch(args.site, tags, args.per_tag, Path(args.out), not args.no_answers)
    print(json.dumps(result, indent=2))
    print(
        "\nAttribution: content is CC BY-SA 4.0, see the source field of every record.\n"
        "Next: bash scripts/build_corpus.sh --knowledge\n"
    )


if __name__ == "__main__":
    main()
