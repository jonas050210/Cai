import json

import pytest
import torch

from codeplan.config import ModelConfig
from codeplan.export import export
from codeplan.model import CodePlanLM
from codeplan.tokenizer import train_tokenizer
from codeplan.utils import save_checkpoint


def _tiny_checkpoint(tmp_path):
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
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, None, None, None, 12, {"model": cfg.to_dict()})
    return path, tmp_path / "tokenizer.json"


def test_export_safetensors(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    checkpoint, tokenizer_path = _tiny_checkpoint(tmp_path)
    out = tmp_path / "model.safetensors"
    info = export(str(checkpoint), out, dtype="fp16", tokenizer_path=str(tokenizer_path))

    assert out.exists()
    assert info["parameters"] > 0
    assert info["dtype"] == "fp16"
    tensors = load_file(str(out))
    assert "embed_tokens.weight" in tensors
    assert tensors["embed_tokens.weight"].dtype == torch.float16
    assert all(t.dtype == torch.float16 for t in tensors.values())

    config = json.loads((tmp_path / "model.config.json").read_text())
    assert config["model"]["d_model"] == 16
    assert config["dtype"] == "fp16"
    assert config["step"] == 12


def test_export_roundtrip_with_tied_embeddings(tmp_path):
    pytest.importorskip("safetensors")
    from codeplan.export import load_exported

    checkpoint, tokenizer_path = _tiny_checkpoint(tmp_path)
    out = tmp_path / "model.safetensors"
    info = export(str(checkpoint), out, dtype="fp32", tokenizer_path=str(tokenizer_path))

    assert info["aliases"] == ["lm_head.weight"]
    restored = load_exported(out, tmp_path / "model.config.json")
    original, _tok, _payload = _load_original(checkpoint, tokenizer_path)
    for (name_a, tensor_a), (name_b, tensor_b) in zip(
        original.state_dict().items(), restored.state_dict().items()
    ):
        assert name_a == name_b
        assert torch.equal(tensor_a, tensor_b)


def _load_original(checkpoint, tokenizer_path):
    from codeplan.generate import load_model_from_checkpoint

    return load_model_from_checkpoint(checkpoint, tokenizer_path, "cpu")
