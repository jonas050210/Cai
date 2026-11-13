import json
import signal

import pytest
import torch

from codeplan.config import ModelConfig
from codeplan.model import CodePlanLM
from codeplan.tokenizer import train_tokenizer
from codeplan.trainer import InterruptGuard, train_loop


def test_guard_first_press_requests_second_press_raises(capsys):
    guard = InterruptGuard()
    guard._handle(signal.SIGINT, None)
    assert guard.requested is True
    assert guard.presses == 1
    with pytest.raises(KeyboardInterrupt):
        guard._handle(signal.SIGINT, None)
    assert "finishing this step" in capsys.readouterr().out


def test_guard_saves_and_returns_early(tmp_path, monkeypatch):
    from codeplan.config import TrainConfig

    cfg = TrainConfig(
        data_dir=str(tmp_path),
        tokenizer_path=str(tmp_path / "tokenizer.json"),
        out_dir=str(tmp_path / "run"),
        seq_len=8,
        batch_size=1,
        grad_accum=1,
        max_steps=50,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=1,
        log_every=1,
        eval_every=1000,
        save_every=1000,
        eval_iters=1,
        keep_last_checkpoints=0,
        model=ModelConfig(
            vocab_size=64, max_seq_len=8, d_model=16, n_layers=1, n_heads=2,
            intermediate_size=32, gradient_checkpointing=False,
        ),
    )

    def install_now(self):
        self.requested = True
        self.presses = 1
        return self

    monkeypatch.setattr("codeplan.trainer.InterruptGuard.install", install_now)

    class FakeLoader:
        def __iter__(self):
            while True:
                yield torch.randint(0, 64, (1, 8)), None

    model = CodePlanLM(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

    result = train_loop(
        model, FakeLoader(), cfg, torch.device("cpu"), None, optimizer, scheduler, None
    )

    assert result["interrupted"] is True
    assert result["last_step"] == 0  # stopped after the first step, not after 50
    last = tmp_path / "run" / "last.pt"
    assert last.exists()
    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert payload["step"] == 0
    assert payload["interrupted"] is True  # save_checkpoint merges extra at top level
    assert "optimizer" in payload  # resumable, not just weights


def _write_tiny_dataset(tmp_path):
    tokenizer = train_tokenizer(
        ["def add(a, b):\n    return a + b"] * 20,
        vocab_size=512,
        min_frequency=1,
        out_path=tmp_path / "tokenizer.json",
    )
    packed = tmp_path / "packed"
    packed.mkdir()
    ids = tokenizer.encode("def add(a, b):\n    return a + b\n" * 40)
    for name in ("train_code.bin", "val_all.bin"):
        with open(packed / name, "wb") as handle:
            torch.tensor(ids, dtype=torch.uint16).numpy().tofile(handle)
    (packed / "meta.json").write_text(
        json.dumps({"dtype": "uint16", "vocab_size": tokenizer.vocab_size,
                    "target_mixture": {"code": 1.0, "planning": 0.0, "general": 0.0}})
    )
    return packed


def _config(tmp_path, max_steps=2):
    from codeplan.config import TrainConfig

    cfg = TrainConfig(
        data_dir=str(tmp_path / "packed"),
        tokenizer_path=str(tmp_path / "tokenizer.json"),
        out_dir=str(tmp_path / "run"),
        seq_len=16,
        batch_size=1,
        grad_accum=1,
        max_steps=max_steps,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=1,
        log_every=1,
        eval_every=1000,
        save_every=1000,
        eval_iters=1,
        num_workers=0,
        keep_last_checkpoints=0,
        model=ModelConfig(
            vocab_size=512, max_seq_len=16, d_model=16, n_layers=1, n_heads=2,
            intermediate_size=32, gradient_checkpointing=False,
        ),
    )
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg.to_dict()))
    return path


def test_pretrain_continues_from_last_checkpoint(tmp_path, monkeypatch, capsys):
    from codeplan import pretrain

    _write_tiny_dataset(tmp_path)
    config = _config(tmp_path)

    monkeypatch.setattr("sys.argv", ["codeplan-pretrain", "--config", str(config)])
    pretrain.main()
    first = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False)
    assert first["step"] == 1
    assert "Found" not in capsys.readouterr().out

    # running the same command again must continue, not restart
    pretrain.main()
    out = capsys.readouterr().out
    assert "continuing from there" in out
    stopped = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False)
    assert stopped["step"] == 1  # already at max_steps, nothing to do

    # --fresh starts over
    monkeypatch.setattr("sys.argv", ["codeplan-pretrain", "--config", str(config), "--fresh"])
    pretrain.main()
    fresh = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False)
    assert fresh["step"] == 1
    assert "continuing from there" not in capsys.readouterr().out
