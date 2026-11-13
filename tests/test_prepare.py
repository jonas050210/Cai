import json

from codeplan.prepare import prepare


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def test_filter_language_and_dedup(tmp_path):
    records = [
        {"text": "This is a valid English code planning record. " * 3, "lang": "en", "domain": "code"},
        {"text": "Dies ist ein deutscher Plan für Software und Tests. " * 3, "lang": "de", "domain": "planning"},
        {"text": "Ceci est un texte français qui ne doit pas passer. " * 3, "lang": "fr", "domain": "general"},
        {"text": "Это русский текст который не должен пройти. " * 3, "lang": "ru", "domain": "general"},
        {"text": "This is a valid English code planning record. " * 3, "lang": "en", "domain": "code"},
        {"text": "short", "lang": "en", "domain": "code"},
    ]
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, records)
    stats = prepare(inp, tmp_path / "out", val_ratio=0.2, test_ratio=0.2)
    assert stats["kept_total"] == 2
    assert stats["duplicate"] == 1
    assert stats["bad_lang"] >= 1
    assert stats["too_short"] == 1
