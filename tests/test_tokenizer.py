import pytest

from codeplan.tokenizer import CodePlanTokenizer, train_tokenizer


def test_train_and_encode(tmp_path):
    texts = [
        "def add(a, b):\n    return a + b\n",
        "Projektplan: Daten sammeln, Tokenizer trainieren, Modell testen.",
        "Build a small coding assistant. Use tests and clear plans.",
    ] * 5
    path = tmp_path / "tokenizer.json"
    tok = train_tokenizer(texts, vocab_size=512, min_frequency=1, out_path=path)
    loaded = CodePlanTokenizer.load(path)
    ids = loaded.encode("def add(a, b): return a + b", add_bos=True, add_eos=True)
    assert ids[0] == loaded.bos_id
    assert ids[-1] == loaded.eos_id
    assert loaded.vocab_size == tok.vocab_size


def test_sft_labels_mask_prompt(tmp_path):
    texts = ["hello world code plan"] * 10
    path = tmp_path / "tokenizer.json"
    tok = train_tokenizer(texts, vocab_size=512, min_frequency=1, out_path=path)
    ids, labels = tok.format_sft("system", "user question", "<plan>do it</plan>")
    assert len(ids) == len(labels)
    assert labels[0] == -100
    assert any(lab != -100 for lab in labels)


def test_vocab_size_below_byte_level_minimum_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="impossible for byte-level BPE"):
        train_tokenizer(["hello world"] * 3, vocab_size=128, min_frequency=1,
                        out_path=tmp_path / "tok.json")
