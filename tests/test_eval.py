import json

import numpy as np

from codeplan.config import ModelConfig
from codeplan.eval import evaluate
from codeplan.model import CodePlanLM
from codeplan.tokenizer import train_tokenizer
from codeplan.utils import save_checkpoint


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
    for name in ["val_all.bin", "val_code.bin", "val_planning.bin"]:
        rng.integers(0, tokenizer.vocab_size, size=200, dtype=np.uint16).tofile(packed / name)
    (packed / "meta.json").write_text(
        json.dumps({"dtype": "uint16", "vocab_size": tokenizer.vocab_size})
    )
    return checkpoint, tmp_path / "tokenizer.json", packed


def test_evaluate_per_domain(tmp_path):
    checkpoint, tokenizer, packed = _setup(tmp_path)
    result = evaluate(
        str(checkpoint), str(packed), "val", seq_len=8, batch_size=1, max_batches=2,
        device_name="cpu", precision="fp32", tokenizer=str(tokenizer), per_domain=True,
    )
    assert set(result["domains"]) == {"code", "planning"}
    assert result["domains"]["code"]["batches"] == 2
    assert result["mean_domain_perplexity"] > 0
    assert result["vocab_size"] > 0


def test_evaluate_all_file(tmp_path):
    checkpoint, tokenizer, packed = _setup(tmp_path)
    result = evaluate(
        str(checkpoint), str(packed), "val", seq_len=8, batch_size=1, max_batches=3,
        device_name="cpu", precision="fp32", tokenizer=str(tokenizer),
    )
    assert result["file"] == "val_all.bin"
    assert result["batches"] == 3
    assert result["perplexity"] > 0


def test_per_domain_skips_too_short_files(tmp_path):
    checkpoint, tokenizer, packed = _setup(tmp_path)
    np.arange(4, dtype=np.uint16).tofile(packed / "val_general.bin")
    result = evaluate(
        str(checkpoint), str(packed), "val", seq_len=8, batch_size=1, max_batches=1,
        device_name="cpu", precision="fp32", tokenizer=str(tokenizer), per_domain=True,
    )
    assert "error" in result["domains"]["general"]
    assert result["mean_domain_perplexity"] > 0
    assert set(result["domains"]) == {"code", "planning", "general"}
