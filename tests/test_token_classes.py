import json

import numpy as np

from codeplan.config import ModelConfig
from codeplan.eval import evaluate
from codeplan.model import CodePlanLM
from codeplan.token_classes import classify_token, evaluate_classes, vocab_classes
from codeplan.tokenizer import train_tokenizer
from codeplan.utils import save_checkpoint


def test_classify_token_kinds():
    assert classify_token("def") == "keyword"
    assert classify_token("Ġreturn") == "keyword"
    assert classify_token("count") == "identifier"
    assert classify_token("ĠgetUserName") == "identifier"
    assert classify_token("42") == "number"
    assert classify_token("3.14e-5") == "number"
    assert classify_token("0xFF") == "number"
    assert classify_token("'") == "string"
    assert classify_token('"') == "string"
    assert classify_token("==") == "operator"
    assert classify_token("Ċ") == "whitespace"
    assert classify_token("ĠĠ") == "whitespace"
    assert classify_token("") == "other"


def test_vocab_classes_covers_every_id():
    vocab = {"def": 0, "x": 1, "1": 2, "Ċ": 3, "=": 4}
    classes = vocab_classes(vocab)
    assert classes == {0: "keyword", 1: "identifier", 2: "number", 3: "whitespace", 4: "operator"}


def _setup(tmp_path):
    tokenizer = train_tokenizer(
        ["def add(a, b): return a + b"] * 8,
        vocab_size=512,
        min_frequency=1,
        out_path=tmp_path / "tokenizer.json",
    )
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=16,
        d_model=16,
        n_layers=1,
        n_heads=2,
        intermediate_size=32,
        gradient_checkpointing=False,
    )
    model = CodePlanLM(cfg)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, None, None, None, 0, {"model": cfg.to_dict()})

    packed = tmp_path / "packed"
    packed.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(0, tokenizer.vocab_size, size=200, dtype=np.uint16).tofile(packed / "val_all.bin")
    (packed / "meta.json").write_text(
        json.dumps({"dtype": "uint16", "vocab_size": tokenizer.vocab_size})
    )
    return checkpoint, tokenizer, packed


def test_evaluate_token_classes_covers_class_share(tmp_path):
    checkpoint, tokenizer, packed = _setup(tmp_path)
    result = evaluate(
        str(checkpoint), str(packed), "val", 16, 1, 2, "cpu", "fp32",
        str(tmp_path / "tokenizer.json"), False, True,
    )
    table = result["classes"]
    assert table
    assert sum(row["tokens"] for row in table.values()) == result["tokens"]
    assert abs(sum(row["share"] for row in table.values()) - 1.0) < 1e-6
    for row in table.values():
        assert row["perplexity"] > 0


def test_evaluate_classes_batch_count_is_capped(tmp_path):
    import torch

    checkpoint, tokenizer, packed = _setup(tmp_path)
    from codeplan.generate import load_model_from_checkpoint

    model, tok, _ = load_model_from_checkpoint(checkpoint, tmp_path / "tokenizer.json", "cpu")
    result = evaluate_classes(
        model, packed / "val_all.bin", 16, 1, 1, torch.device("cpu"), None,
        vocab_classes(tok.tokenizer.get_vocab()),
    )
    assert result["batches"] == 1
    assert result["tokens"] == 15  # one label per position after the first
