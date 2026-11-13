import json

import numpy as np

from codeplan.config import ModelConfig, TrainConfig
from codeplan.doctor import ERROR, OK, WARN, run
from codeplan.tokenizer import train_tokenizer


def _project(tmp_path, max_steps=10):
    tokenizer = train_tokenizer(
        ["def add(a, b): return a + b"] * 8, 512, 1, tmp_path / "tokenizer.json"
    )
    packed = tmp_path / "packed"
    packed.mkdir()
    rng = np.random.default_rng(0)
    for name in ["train_all.bin", "train_code.bin", "val_all.bin"]:
        rng.integers(0, tokenizer.vocab_size, size=400, dtype=np.uint16).tofile(packed / name)
    (packed / "meta.json").write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "vocab_size": tokenizer.vocab_size,
                "splits": {
                    "train": {
                        "files": {"all": "train_all.bin", "code": "train_code.bin"},
                        "tokens": {"all": 400, "code": 400},
                    },
                    "val": {"files": {"all": "val_all.bin"}, "tokens": {"all": 400}},
                },
            }
        )
    )
    cfg = TrainConfig(
        data_dir=str(packed),
        tokenizer_path=str(tmp_path / "tokenizer.json"),
        out_dir=str(tmp_path / "checkpoints"),
        seq_len=32,
        batch_size=1,
        grad_accum=2,
        max_steps=max_steps,
        warmup_steps=2,
        eval_every=5,
        save_every=5,
        eval_iters=1,
        device="cpu",
        precision="fp32",
        mixture={"code": 1.0},
    )
    cfg.model = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        intermediate_size=32,
        gradient_checkpointing=False,
    )
    return cfg


def _levels(report):
    return [level for level, _ in report["results"]]


def test_doctor_passes_on_valid_project(tmp_path, capsys):
    cfg = _project(tmp_path)
    report = run(cfg, "cpu", bench_steps=0)
    assert report["counts"][ERROR] == 0
    messages = " ".join(message for _, message in report["results"])
    assert "tokenizer" in messages
    assert "parameters" in messages
    assert any(level == WARN for level in _levels(report))  # CPU warning


def test_doctor_detects_missing_data(tmp_path, capsys):
    cfg = _project(tmp_path)
    cfg.data_dir = str(tmp_path / "missing")
    report = run(cfg, "cpu", bench_steps=0)
    assert report["counts"][ERROR] >= 1
    assert any("packed data dir missing" in message for _, message in report["results"])


def test_doctor_detects_invalid_config(tmp_path, capsys):
    cfg = _project(tmp_path)
    cfg.seq_len = cfg.model.max_seq_len * 2
    report = run(cfg, "cpu", bench_steps=0)
    assert report["counts"][ERROR] >= 1
    assert any("seq_len" in message for _, message in report["results"])


def test_doctor_benchmark_reports_throughput(tmp_path, capsys):
    cfg = _project(tmp_path)
    report = run(cfg, "cpu", bench_steps=2)
    messages = " ".join(message for _, message in report["results"])
    assert "throughput:" in messages
    assert "estimated wall clock" in messages
    assert report["counts"][OK] > 0


def test_doctor_warns_about_epochs(tmp_path, capsys):
    cfg = _project(tmp_path, max_steps=5000)
    report = run(cfg, "cpu", bench_steps=0)
    assert any("epochs" in message and level == ERROR for level, message in report["results"])
