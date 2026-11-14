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
                 "sft_50m_4060ti", "pretrain_100m_4060ti", "pretrain_100m_fast",
                 "sft_100m_4060ti", "pretrain_bilingual_100m", "sft_bilingual_100m"]:
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


def test_cli_overrides_batch_size_and_compile(tmp_path, monkeypatch):
    """--batch-size/--grad-accum/--no-compile must win over the config file.

    Uses real (tiny) files instead of patching pathlib.Path, which is immutable
    on Python 3.14.
    """
    import numpy as np

    from codeplan import pretrain
    from codeplan.tokenizer import train_tokenizer

    tokenizer = train_tokenizer(
        ["def add(a, b):\n    return a + b"] * 20,
        vocab_size=274,
        min_frequency=1,
        out_path=tmp_path / "tokenizer.json",
    )
    packed = tmp_path / "packed"
    packed.mkdir()
    ids = tokenizer.encode("def add(a, b):\n    return a + b\n" * 200)
    for name in ("train_code.bin", "val_all.bin"):
        np.array(ids, dtype=np.uint16).tofile(packed / name)
    (packed / "meta.json").write_text(
        json.dumps({"dtype": "uint16", "vocab_size": tokenizer.vocab_size,
                    "target_mixture": {"code": 1.0, "planning": 0.0, "general": 0.0}})
    )

    raw = json.loads(Path("configs/pretrain_100m_fast.json").read_text())
    raw["data_dir"] = str(packed)
    raw["tokenizer_path"] = str(tmp_path / "tokenizer.json")
    raw["out_dir"] = str(tmp_path / "run")
    raw["max_steps"] = 2000
    raw["warmup_steps"] = 10
    raw["num_workers"] = 4
    raw["model"] = {
        "vocab_size": 274, "max_seq_len": 32, "d_model": 16, "n_layers": 1,
        "n_heads": 2, "intermediate_size": 32, "dropout": 0.0,
        "tie_embeddings": True, "gradient_checkpointing": False,
    }
    raw["seq_len"] = 32
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(raw))

    captured = {}

    def fake_train_loop(model, loader, cfg, *args, **kwargs):
        captured.update(
            batch_size=cfg.batch_size, grad_accum=cfg.grad_accum,
            compile=cfg.torch_compile, num_workers=cfg.num_workers,
        )
        return {"last_step": 0, "best_val": 0.0, "bad_evals": 0, "interrupted": False}

    monkeypatch.setattr(pretrain, "train_loop", fake_train_loop)
    monkeypatch.setattr("sys.argv", [
        "codeplan-pretrain", "--config", str(config),
        "--batch-size", "1", "--grad-accum", "16", "--no-compile", "--fresh",
        "--device", "cpu",
    ])
    pretrain.main()
    assert captured == {
        "batch_size": 1, "grad_accum": 16, "compile": False, "num_workers": 4,
    }
