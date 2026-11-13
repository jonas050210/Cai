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


def test_code_without_prose_is_kept(tmp_path):
    code = "def compute(x, y):\n    result = x * 2 + y\n    return result\n\n" * 5
    records = [{"text": code, "lang": "unknown", "domain": "code", "license": "MIT"}]
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, records)
    stats = prepare(inp, tmp_path / "out")
    assert stats["kept_total"] == 1
    assert stats["lang_source_code-default"] == 1


def test_code_with_foreign_comments_is_rejected(tmp_path):
    code = "# Calcule la somme des nombres et renvoie le resultat\n" * 4
    records = [{"text": code + "def f(x):\n    return x\n" * 4, "lang": "unknown",
                "domain": "code", "license": "MIT"}]
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, records)
    stats = prepare(inp, tmp_path / "out")
    assert stats["kept_total"] == 0
    assert stats["bad_lang"] == 1


def test_german_comments_in_code_are_detected(tmp_path):
    code = (
        "# Berechne die Summe der Zahlen und gib das Ergebnis zurueck\n"
        "def summe(werte):\n    return sum(werte)\n"
    ) * 3
    records = [{"text": code, "lang": None, "domain": "code", "license": "MIT"}]
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, records)
    stats = prepare(inp, tmp_path / "out")
    assert stats["kept_total"] == 1
    assert stats["lang_de"] == 1
