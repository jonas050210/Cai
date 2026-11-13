from codeplan.tokenizer import train_tokenizer
from codeplan.tokenizer_eval import analyse, format_table, measure, recommend

TEXTS = [
    "def add(a, b):\n    return a + b\n",
    "Projektplan: Daten sammeln, Tokenizer trainieren, Modell testen.",
    "Build a small coding assistant with tests and clear plans.",
    "Die Funktion berechnet die Summe zweier Zahlen und gibt sie zurück.",
]


def test_measure_groups(tmp_path):
    tok = train_tokenizer(TEXTS * 5, vocab_size=512, min_frequency=1, out_path=tmp_path / "t.json")
    records = [
        {"text": TEXTS[0], "domain": "code", "lang": "en"},
        {"text": TEXTS[3], "domain": "general", "lang": "de"},
    ]
    stats = measure(tok, records)
    assert set(stats["groups"]) == {"code/en", "general/de"}
    assert stats["overall"]["tokens"] > 0
    assert 0 < stats["overall"]["tokens_per_char"] <= 2
    assert stats["overall"]["chars_per_token"] > 0


def test_analyse_and_recommend(tmp_path):
    path = tmp_path / "train.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for i in range(40):
            text = TEXTS[i % len(TEXTS)].replace("\n", " ")
            handle.write(
                '{"text": "%s", "domain": "code", "lang": "en"}\n' % text.replace('"', "'")
            )

    report = analyse(path, [512, 1024], d_model=64, sample_records=30, eval_records=10)
    assert len(report["results"]) == 2
    assert report["results"][0]["embedding_params"] == report["results"][0]["vocab_size"] * 64
    actual = {r["vocab_size"] for r in report["results"]}
    assert report["recommendation"]["vocab_size"] in actual
    assert "recommended vocab" in format_table(report)


def test_recommend_prefers_small_vocab_within_tolerance():
    results = [
        {"vocab_size": 16000, "overall": {"tokens_per_char": 0.26}},
        {"vocab_size": 32000, "overall": {"tokens_per_char": 0.25}},
    ]
    assert recommend(results, tolerance=0.05)["vocab_size"] == 16000
    assert recommend(results, tolerance=0.01)["vocab_size"] == 32000
