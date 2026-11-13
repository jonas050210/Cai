import pytest

from codeplan.config import ModelConfig
from codeplan.model import CodePlanLM
from codeplan.tokenizer import train_tokenizer
from codeplan.utils import save_checkpoint

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def _checkpoint(tmp_path):
    tokenizer = train_tokenizer(
        ["def add(a, b): return a + b"] * 8,
        vocab_size=512,
        min_frequency=1,
        out_path=tmp_path / "tokenizer.json",
    )
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        d_model=16,
        n_layers=1,
        n_heads=2,
        intermediate_size=32,
        gradient_checkpointing=False,
    )
    model = CodePlanLM(cfg)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, None, None, None, 0, {"model": cfg.to_dict()})
    return path, tmp_path / "tokenizer.json"


def test_health_and_generate(tmp_path):
    from fastapi.testclient import TestClient

    from codeplan.server import create_app

    checkpoint, tokenizer = _checkpoint(tmp_path)
    client = TestClient(create_app(str(checkpoint), str(tokenizer), "cpu"))

    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["vocab_size"] > 0

    response = client.post("/generate", json={"prompt": "Write add(a, b)", "max_new_tokens": 4})
    assert response.status_code == 200
    body = response.json()
    assert body["completion_tokens"] <= 4
    assert body["prompt_tokens"] > 0

    assert client.post("/generate", json={"max_new_tokens": 4}).status_code == 400
    assert client.post("/generate", json={"prompt": "x", "top_p": 0}).status_code == 400
    assert client.get("/").status_code == 200
