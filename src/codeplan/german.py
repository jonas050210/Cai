"""German text for the bilingual model.

Two sources, both licence-clean:

* Wikipedia (CC BY-SA 4.0) via the public API - no dump download, no parsing of
  multi-gigabyte XML, just article extracts in plain text.
* Nothing else by design: German book corpora are copyrighted, and scraped
  pages have no licence we can honour.

The records use the same JSONL shape as ``codeplan-collect`` so they can be
mixed and filtered by the normal pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LICENSE = "cc-by-sa-4.0"
USER_AGENT = "codeplan-german/0.1 (corpus builder; contact: repository owner)"

# Articles that are usually lists, dates or disambiguation pages add little.
SKIP_TITLE_HINTS = (
    "list of", "liste", "begriffsklärung", "disambiguation", "timeline",
    "chronik", "jahre", "gestorben", "geboren",
)


def _api(lang: str, params: dict[str, str], timeout: int = 60) -> bytes:
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def random_articles(lang: str, count: int = 20) -> list[dict]:
    """Fetch a batch of random article extracts."""
    payload = json.loads(
        _api(
            lang,
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "generator": "random",
                "grnnamespace": "0",
                "grnlimit": str(min(count, 20)),
                "prop": "extracts",
                "explaintext": "1",
                "exintro": "0",
                "exlimit": "max",
            },
        )
    )
    pages = (payload.get("query") or {}).get("pages") or []
    articles = []
    for page in pages:
        title = page.get("title", "")
        text = page.get("extract") or ""
        if len(text) < 500:
            continue
        if any(hint in title.lower() for hint in SKIP_TITLE_HINTS):
            continue
        articles.append({"title": title, "text": text})
    return articles


def record(article: dict, lang: str) -> dict:
    digest = hashlib.sha256(f"{lang}:{article['title']}".encode()).hexdigest()[:16]
    return {
        "id": digest,
        "text": article["text"],
        "lang": lang,
        "domain": "general",
        "license": LICENSE,
        "source": f"wikipedia:{lang}/{article['title'].replace(' ', '_')}",
    }


def fetch(lang: str, pages: int, out_path: Path, sleep: float = 0.5) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen: set[str] = set()
    chars = 0
    with open(out_path, "w", encoding="utf-8") as out:
        while written < pages:
            try:
                articles = random_articles(lang, 20)
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                print(f"  request failed ({exc}); retrying in 5 s")
                time.sleep(5)
                continue
            if not articles:
                time.sleep(sleep)
                continue
            for article in articles:
                if article["title"] in seen:
                    continue
                seen.add(article["title"])
                rec = record(article, lang)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                chars += len(rec["text"])
                if written % 100 == 0:
                    print(f"  {written}/{pages} articles, {chars / 1e6:.1f}M chars", flush=True)
                if written >= pages:
                    break
            out.flush()
            time.sleep(sleep)
    return {"pages": written, "chars": chars, "out": str(out_path), "license": LICENSE}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch German Wikipedia text as JSONL.")
    parser.add_argument("--out", default="data/raw/german_wiki.jsonl")
    parser.add_argument("--lang", default="de")
    parser.add_argument("--pages", type=int, default=3000, help="How many articles")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between requests")
    args = parser.parse_args()

    print(f"Fetching {args.pages} {args.lang} articles (licence CC BY-SA 4.0)")
    result = fetch(args.lang, args.pages, Path(args.out), args.sleep)
    print(json.dumps(result, indent=2))
    print(
        "\nNext:\n"
        f"  cat data/raw/{Path(args.out).name} >> data/raw/all.jsonl   # or let build_corpus do it\n"
        "  remember: the corpus needs the same domain balance during packing\n"
    )


if __name__ == "__main__":
    main()
