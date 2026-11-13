"""Same seed must give the same model, the same batch order and the same loss."""

import hashlib

import torch

from codeplan.config import ModelConfig
from codeplan.model import CodePlanLM
from codeplan.utils import set_seed


def _config():
    return ModelConfig(
        vocab_size=128,
        max_seq_len=32,
        d_model=32,
        n_layers=2,
        n_heads=2,
        intermediate_size=64,
        gradient_checkpointing=False,
    )


def _weights_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().float().numpy().tobytes())
    return digest.hexdigest()


def test_same_seed_same_weights():
    set_seed(1234)
    first = _weights_hash(CodePlanLM(_config()))
    set_seed(1234)
    second = _weights_hash(CodePlanLM(_config()))
    assert first == second

    set_seed(99)
    assert _weights_hash(CodePlanLM(_config())) != first


def test_same_seed_same_loss_and_gradients():
    set_seed(7)
    model = CodePlanLM(_config())
    batch = torch.randint(0, 127, (2, 16))
    _, loss, _ = model(batch, labels=batch)
    loss.backward()
    grad_hash = hashlib.sha256(
        b"".join(p.grad.detach().numpy().tobytes() for p in model.parameters() if p.grad is not None)
    ).hexdigest()

    set_seed(7)
    model2 = CodePlanLM(_config())
    _, loss2, _ = model2(batch, labels=batch)
    loss2.backward()
    grad_hash2 = hashlib.sha256(
        b"".join(p.grad.detach().numpy().tobytes() for p in model2.parameters() if p.grad is not None)
    ).hexdigest()

    assert torch.allclose(loss, loss2)
    assert grad_hash == grad_hash2
