import json

from codeplan import stackexchange
from codeplan.stackexchange import fetch, record, to_text, top_questions


def test_to_text_strips_html_and_entities():
    body = "<p>Use <code>len(x)</code></p><p>it is O(1) &amp; fast</p>"
    text = to_text(body)
    assert "<" not in text
    assert "&amp;" not in text
    assert "O(1)" in text


def test_top_questions_maps_fields(monkeypatch):
    payload = {
        "items": [
            {
                "question_id": 42,
                "title": "Why is dict lookup O(1)?",
                "body": "<p>Because of hashing.</p>",
                "score": 120,
                "link": "https://example.invalid/q/42",
                "tags": ["python"],
            }
        ]
    }
    monkeypatch.setattr(stackexchange, "_get", lambda path, params: payload)
    questions = top_questions("stackoverflow", "python", 1)
    assert questions[0]["id"] == 42
    assert questions[0]["title"].startswith("Why is dict")
    assert questions[0]["score"] == 120


def test_record_contains_question_and_answer():
    question = {"id": 7, "title": "T", "body": "B", "score": 3, "tags": [], "link": ""}
    rec = record(question, "the answer", "stackoverflow")
    assert "Answer:" in rec["text"]
    assert rec["license"] == "cc-by-sa-4.0"
    assert rec["domain"] == "planning"
    assert rec["source"] == "stackexchange:stackoverflow/7"


def test_fetch_writes_records_and_skips_short_ones(tmp_path, monkeypatch):
    def fake_questions(site, tag, count):
        return [
            {"id": 1, "title": "Long enough question?", "body": "x" * 500, "score": 9,
             "tags": [tag], "link": ""},
            {"id": 2, "title": "tiny", "body": "", "score": 1, "tags": [tag], "link": ""},
        ]

    monkeypatch.setattr(stackexchange, "top_questions", fake_questions)
    monkeypatch.setattr(stackexchange, "top_answer", lambda site, qid, count=1: "y" * 200)

    out = tmp_path / "se.jsonl"
    result = fetch("stackoverflow", ["python"], 2, out)
    assert result["records"] == 1
    line = json.loads(out.read_text().splitlines()[0])
    assert line["source"].startswith("stackexchange:")
