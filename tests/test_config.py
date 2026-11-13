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
