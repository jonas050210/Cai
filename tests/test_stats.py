import json

from codeplan.stats import _repo_of, analyse, format_report
from codeplan.tokenizer import train_tokenizer


def _write(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _records():
    return [
        {"text": "def add(a, b):\n    return a + b\n" * 5, "domain": "code", "lang": "en",
         "license": "MIT", "source": "github:owner/repo/src/add.py"},
        {"text": "Projektplan: Daten sammeln und Modell trainieren. " * 5, "domain": "planning",
         "lang": "de", "license": "MIT", "source": "github:owner/repo/docs/PLAN.md"},
        {"text": "def add(a, b):\n    return a + b\n" * 5, "domain": "code", "lang": "en",
         "license": "MIT", "source": "github:other/lib/src/add.py"},
    ]


def test_repo_of_parses_sources():
    assert _repo_of("github:owner/repo/src/file.py") == "github:owner/repo"
    assert _repo_of("local") == "local"
    assert _repo_of("github:owner") == "github:owner"


def test_analyse_counts_domains_and_duplicates(tmp_path):
    path = tmp_path / "train.jsonl"
    _write(path, _records())
    report = analyse(path)
    assert report["records"] == 3
    assert report["unique_records"] == 2
    assert report["exact_duplicates"] == 1
    assert report["domains"]["code"]["share"] == pytest_approx(0.7, 0.25)
    assert report["languages"]["en"]["chars"] > 0
    assert report["licenses"]["MIT"] == 3
    assert report["sources"]["unique"] == 2
    assert "domains" in format_report(report)


def test_budget_verdict(tmp_path):
    path = tmp_path / "train.jsonl"
    _write(path, _records())
    report = analyse(path, target_tokens=10)
    assert report["budget"]["verdict"].startswith("enough data")

    report = analyse(path, target_tokens=10_000_000)
    assert "too little data" in report["budget"]["verdict"]


def test_tokenizer_measurement(tmp_path):
    texts = ["def add(a, b): return a + b"] * 8
    tokenizer = train_tokenizer(texts, 512, 1, tmp_path / "tok.json")
    path = tmp_path / "train.jsonl"
    _write(path, _records())
    report = analyse(path, tokenizer=tokenizer, sample=2)
    assert report["tokenizer"]["chars_per_token"] > 1
    assert report["estimated_tokens"] > 0


def pytest_approx(value, rel):
    import pytest

    return pytest.approx(value, rel=rel)
