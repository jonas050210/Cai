from codeplan.trainer import EarlyStopper, prune_checkpoints


def test_prune_checkpoints_keeps_newest(tmp_path):
    for step in [0, 1000, 2000, 3000]:
        (tmp_path / f"step_{step:07d}.pt").write_bytes(b"x")
    (tmp_path / "last.pt").write_bytes(b"x")

    removed = prune_checkpoints(tmp_path, keep=2)
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert [p.name for p in removed] == ["step_0000000.pt", "step_0001000.pt"]
    assert remaining == ["step_0002000.pt", "step_0003000.pt"]
    assert (tmp_path / "last.pt").exists()


def test_prune_checkpoints_disabled(tmp_path):
    for step in [0, 1000, 2000]:
        (tmp_path / f"step_{step:07d}.pt").write_bytes(b"x")
    assert prune_checkpoints(tmp_path, keep=0) == []
    assert len(list(tmp_path.glob("step_*.pt"))) == 3


def test_early_stopper_patience():
    stopper = EarlyStopper(patience=2)
    best = 5.0
    stop, best = stopper.update(4.0, best)
    assert not stop and best == 4.0 and stopper.bad == 0

    stop, best = stopper.update(4.5, best)
    assert not stop and best == 4.0 and stopper.bad == 1

    stop, best = stopper.update(4.6, best)
    assert stop and best == 4.0


def test_early_stopper_disabled():
    stopper = EarlyStopper(patience=0)
    best = 1.0
    for _ in range(10):
        stop, best = stopper.update(2.0, best)
        assert not stop
    assert stopper.bad == 10
