import pytest
import torch

from codeplan.config import ModelConfig
from codeplan.model import CodePlanLM
from codeplan.soup import average_state_dicts, collect_paths, soup
from codeplan.utils import save_checkpoint


def _checkpoint(path, value, cfg):
    model = CodePlanLM(cfg)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(value)
    save_checkpoint(path, model, None, None, None, int(value), {"model": cfg.to_dict()})
    return path


def test_average_state_dicts_equals_mean():
    a = {"w": torch.ones(2), "n": torch.tensor(3, dtype=torch.long)}
    b = {"w": torch.full((2,), 3.0), "n": torch.tensor(7, dtype=torch.long)}
    merged = average_state_dicts([a, b])
    assert torch.allclose(merged["w"], torch.full((2,), 2.0))
    assert merged["w"].dtype == torch.float32
    # integer buffers cannot be averaged and are copied from the first checkpoint
    assert merged["n"].item() == 3


def test_average_respects_weights():
    a = {"w": torch.zeros(1)}
    b = {"w": torch.full((1,), 4.0)}
    merged = average_state_dicts([a, b], weights=[3, 1])
    assert merged["w"].item() == pytest.approx(1.0)


def test_average_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        average_state_dicts([{"w": torch.zeros(2)}, {"w": torch.zeros(3)}])
    with pytest.raises(ValueError):
        average_state_dicts([{"w": torch.zeros(2)}, {"v": torch.zeros(2)}])


def test_collect_paths_reports_missing(tmp_path):
    with pytest.raises(SystemExit):
        collect_paths([str(tmp_path / "nope.pt")], None, 5)
    with pytest.raises(SystemExit):
        collect_paths(None, None, 5)


def test_soup_end_to_end(tmp_path):
    cfg = ModelConfig(
        vocab_size=64, max_seq_len=8, d_model=16, n_layers=1, n_heads=2,
        intermediate_size=32, gradient_checkpointing=False,
    )
    a = _checkpoint(tmp_path / "a.pt", 1.0, cfg)
    b = _checkpoint(tmp_path / "b.pt", 3.0, cfg)
    payload = soup([a, b])
    assert payload["step"] == 3
    assert payload["extra"]["soup"]["sources"] == [str(a), str(b)]

    merged = CodePlanLM(cfg)
    merged.load_state_dict(payload["model"], strict=True)
    with torch.no_grad():
        assert next(merged.parameters()).flatten()[0].item() == pytest.approx(2.0)
