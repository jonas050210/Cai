import json

from codeplan import german
from codeplan.german import fetch, random_articles, record


def _payload(titles):
    pages = []
    for title in titles:
        length = 6 if title == "Kurz" else 40  # "Kurz" must fall under the 500-char floor
        pages.append({"title": title, "extract": f"Dies ist ein Text über {title}. " * length})
    return json.dumps({"query": {"pages": pages}}).encode()


def test_random_articles_filters_short_and_list_pages(monkeypatch):
    monkeypatch.setattr(
        german, "_api", lambda lang, params: _payload(["Rust", "Liste von Städten", "Kurz"])
    )
    articles = random_articles("de", 20)
    assert [a["title"] for a in articles] == ["Rust"]


def test_record_carries_licence_and_source():
    rec = record({"title": "Bundesrepublik Deutschland", "text": "x" * 600}, "de")
    assert rec["license"] == "cc-by-sa-4.0"
    assert rec["lang"] == "de"
    assert rec["source"].startswith("wikipedia:de/")
    assert rec["domain"] == "general"


def test_fetch_writes_requested_pages(tmp_path, monkeypatch):
    batches = [["Python"], ["Rust"], ["Go"]]
    monkeypatch.setattr(german, "_api", lambda lang, params: _payload(batches.pop(0)))
    monkeypatch.setattr(german.time, "sleep", lambda s: None)

    out = tmp_path / "de.jsonl"
    result = fetch("de", 3, out, sleep=0)
    assert result["pages"] == 3
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert {line["source"].split("/")[-1] for line in lines} == {"Python", "Rust", "Go"}
    assert all(line["license"] == "cc-by-sa-4.0" for line in lines)


def test_fetch_retries_on_network_error(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky(lang, params):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("boom")
        return _payload(["Python"])

    monkeypatch.setattr(german, "_api", flaky)
    monkeypatch.setattr(german.time, "sleep", lambda s: None)
    result = fetch("de", 1, tmp_path / "de.jsonl", sleep=0)
    assert result["pages"] == 1
    assert calls["n"] >= 2


def test_cc_by_sa_is_an_allowed_licence():
    from codeplan.collect import license_ok

    assert license_ok("cc-by-sa-4.0")
