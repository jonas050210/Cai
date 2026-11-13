import json
from pathlib import Path

from codeplan.config import TrainConfig, load_train_config


def test_load_config():
    cfg = load_train_config("configs/tiny_smoke.json")
    assert isinstance(cfg, TrainConfig)
    assert cfg.model.d_model == 64
    assert cfg.mixture["code"] > 0


def test_config_roundtrip():
    cfg = TrainConfig()
    d = cfg.to_dict()
    cfg2 = TrainConfig.from_dict(d)
    assert cfg2.model.d_model == cfg.model.d_model


def test_validate_accepts_repo_configs():
    for name in ["tiny_smoke", "tiny_sft", "pretrain_pilot", "pretrain_50m_4060ti",
                 "sft_50m_4060ti"]:
        cfg = load_train_config(f"configs/{name}.json")
        assert cfg.validate() == [], f"{name}: {cfg.validate()}"


def test_validate_catches_common_mistakes():
    cfg = TrainConfig()
    cfg.seq_len = cfg.model.max_seq_len + 1
    cfg.warmup_steps = cfg.max_steps
    cfg.batch_size = 0
    cfg.mixture = {}
    errors = cfg.validate()
    assert any("seq_len" in e for e in errors)
    assert any("warmup_steps" in e for e in errors)
    assert any("batch_size" in e for e in errors)
    assert any("mixture" in e for e in errors)


def test_token_budget_helpers():
    cfg = TrainConfig(batch_size=1, grad_accum=16, seq_len=2048, max_steps=100)
    assert cfg.tokens_per_step() == 32768
    assert cfg.total_tokens() == 3_276_800


def test_unknown_config_keys_are_reported(tmp_path, capsys):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"max_steps": 10, "mystery_key": 1, "notes": "ok"}))
    cfg = load_train_config(path)
    captured = capsys.readouterr()
    assert "mystery_key" in captured.out
    assert "notes" not in captured.out
    assert cfg.max_steps == 10


def test_configs_use_torch_compile_key():
    for name in ["pretrain_pilot", "pretrain_50m_4060ti", "quick_test", "sft_50m_4060ti"]:
        raw = json.loads(Path(f"configs/{name}.json").read_text())
        assert "torch_compile" in raw, name
        assert "compile" not in raw, name
