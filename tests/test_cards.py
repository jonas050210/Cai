from codeplan.cards import cards, sft_pairs


def test_cards_are_paired_and_licensed():
    records = cards()
    assert len(records) > 60
    english = [r for r in records if r["lang"] == "en"]
    german = [r for r in records if r["lang"] == "de"]
    assert english and german
    assert all(r["license"] == "cc0-1.0" for r in records)
    assert all(r["domain"] == "planning" for r in records)
    assert len({r["id"] for r in records}) == len(records)


def test_german_cards_do_not_contain_english_boilerplate():
    german = [r for r in cards() if r["lang"] == "de"]
    for record in german:
        assert "Use it when" not in record["text"]
        assert "avoid it when" not in record["text"]
        assert "operations cost" not in record["text"]


def test_sft_pairs_cover_explain_and_translate():
    pairs = sft_pairs()
    kinds = {pair["source"].split(":")[0] for pair in pairs}
    assert "translate" in kinds
    assert any(pair["source"].startswith("card:") for pair in pairs)
    for pair in pairs:
        assert pair["user"].strip()
        assert pair["assistant"].strip()
        assert pair["system"].strip()
