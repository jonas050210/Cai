"""Resuming must restore the RNG state, including on CUDA.

The original bug: ``torch.load(..., map_location=device)`` moves *every* tensor
in the payload to the device, including the RNG state. ``torch.set_rng_state``
only accepts a CPU ``ByteTensor``, so resuming on a GPU crashed with
"RNG state must be a torch.ByteTensor".
"""

import random

import numpy as np
import torch

from codeplan.utils import _rng_state, load_checkpoint, save_checkpoint


def _model():
    return torch.nn.Linear(4, 4)


def test_rng_state_normalisation():
    state = torch.get_rng_state()
    assert _rng_state(state).dtype == torch.uint8
    assert _rng_state(state).device.type == "cpu"
    # tolerate a stored numpy array (older checkpoints, weights_only variations)
    assert _rng_state(state.numpy()).dtype == torch.uint8
    assert _rng_state(list(range(16))).tolist() == list(range(16))


def test_resume_restores_random_state(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    save_checkpoint(path, model, optimizer, None, None, step=5, config={})

    expected = torch.rand(4)
    expected_python = random.random()
    expected_numpy = np.random.rand()

    # a fresh process: different state, then restore from the checkpoint
    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)
    load_checkpoint(path, _model(), optimizer, None, None, device="cpu", restore_rng=True)

    assert torch.allclose(torch.rand(4), expected)
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy


def test_resume_without_rng_does_not_crash(tmp_path):
    model = _model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, None, None, None, step=1, config={})
    load_checkpoint(path, _model(), device="cpu", restore_rng=True)


def test_resume_accepts_byte_state_from_numpy(tmp_path):
    """Simulates a payload whose RNG tensors were converted by map_location."""
    model = _model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, None, None, None, step=1, config={})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rng"]["torch"] = payload["rng"]["torch"].numpy()
    torch.save(payload, path)
    load_checkpoint(path, _model(), device="cpu", restore_rng=True)
