import torch

from codeplan.config import ModelConfig
from codeplan.model import CodePlanLM, generate


def tiny_cfg():
    return ModelConfig(
        vocab_size=64,
        max_seq_len=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        intermediate_size=64,
        gradient_checkpointing=False,
    )


def test_forward_backward():
    model = CodePlanLM(tiny_cfg())
    x = torch.randint(0, 64, (2, 16))
    logits, loss, _ = model(x, labels=x)
    assert logits.shape == (2, 16, 64)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()


def test_attention_mask():
    model = CodePlanLM(tiny_cfg())
    x = torch.randint(0, 64, (2, 16))
    mask = torch.ones_like(x)
    mask[1, 8:] = 0
    logits, loss, _ = model(x, labels=x, attention_mask=mask)
    assert torch.isfinite(loss)


def test_generate_shape():
    model = CodePlanLM(tiny_cfg())
    x = torch.randint(0, 64, (1, 4))
    out = generate(model, x, max_new_tokens=5, temperature=0.0, eos_token_id=None)
    assert out.shape == (1, 9)


def test_gradient_checkpointing():
    cfg = tiny_cfg()
    cfg.gradient_checkpointing = True
    model = CodePlanLM(cfg)
    x = torch.randint(0, 64, (2, 16))
    _, loss, _ = model(x, labels=x)
    loss.backward()
