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


def test_gqa_shrinks_kv_heads_and_keeps_output_shape():
    import torch

    from codeplan.config import ModelConfig
    from codeplan.model import CodePlanLM

    base = dict(vocab_size=128, max_seq_len=16, d_model=32, n_layers=1, n_heads=4,
                intermediate_size=64, gradient_checkpointing=False)
    mha = CodePlanLM(ModelConfig(**base))
    gqa = CodePlanLM(ModelConfig(**base, n_kv_heads=2))
    assert gqa.num_parameters() < mha.num_parameters()

    x = torch.randint(0, 128, (2, 8))
    logits, loss, _ = gqa(x, labels=x)
    assert logits.shape == (2, 8, 128)
    assert loss.item() > 0


def test_gqa_kv_cache_has_fewer_heads_and_matches_full_forward():
    import torch

    from codeplan.config import ModelConfig
    from codeplan.model import CodePlanLM

    cfg = ModelConfig(vocab_size=128, max_seq_len=32, d_model=32, n_layers=2, n_heads=4,
                      n_kv_heads=2, qk_norm=True, intermediate_size=64,
                      gradient_checkpointing=False)
    model = CodePlanLM(cfg).eval()
    x = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        full = model(x)[0]
        _, _, kvs = model(x[:, :10], use_cache=True)
        step, _, _ = model(x[:, 10:11], start_pos=10, use_cache=True, past_kvs=kvs)
    k_head, v_head = kvs[0]
    assert k_head.shape[1] == 2  # KV heads, not query heads
    assert torch.allclose(full[:, 10], step[:, -1], atol=1e-4)


def test_z_loss_increases_the_reported_loss():
    import torch

    from codeplan.config import ModelConfig
    from codeplan.model import CodePlanLM

    base = dict(vocab_size=128, max_seq_len=16, d_model=32, n_layers=1, n_heads=4,
                intermediate_size=64, gradient_checkpointing=False)
    x = torch.randint(1, 128, (2, 10))
    plain = CodePlanLM(ModelConfig(**base))(x, labels=x)[1].item()
    with_z = CodePlanLM(ModelConfig(**base, z_loss_coef=1e-2))(x, labels=x)[1].item()
    assert with_z > plain


def test_old_checkpoint_configs_still_load():
    """Configs written before GQA/QK-Norm existed must keep working."""
    from codeplan.config import ModelConfig

    old = {"vocab_size": 1000, "max_seq_len": 128, "d_model": 64, "n_layers": 2,
           "n_heads": 4, "intermediate_size": 128, "dropout": 0.0,
           "rope_theta": 10000.0, "tie_embeddings": True, "gradient_checkpointing": True}
    cfg = ModelConfig.from_dict(old)
    assert cfg.n_kv_heads == 0  # 0 means "same as n_heads"
    assert cfg.qk_norm is False
    assert cfg.z_loss_coef == 0.0
