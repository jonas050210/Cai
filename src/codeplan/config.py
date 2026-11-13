"""Configuration helpers.

Configs are plain JSON so another AI or human can edit them without extra deps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_model: int = 512
    n_layers: int = 10
    n_heads: int = 8
    intermediate_size: int = 1408
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    gradient_checkpointing: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        allowed = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**allowed)


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/packed"
    tokenizer_path: str = "artifacts/tokenizer.json"
    out_dir: str = "checkpoints/pretrain"
    seq_len: int = 2048

    # optimization
    batch_size: int = 1
    grad_accum: int = 16
    max_steps: int = 20000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # runtime
    device: str = "auto"
    precision: str = "bf16"  # bf16 | fp16 | fp32
    num_workers: int = 2
    seed: int = 1337
    compile: bool = False
    resume: str | None = None
    init_checkpoint: str | None = None

    # logging / eval
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    eval_iters: int = 20

    # SFT only
    sft_data: str | None = None
    sft_max_len: int = 2048

    # Pretraining domain mixture. Target: 99% code + planning.
    mixture: dict[str, float] = field(default_factory=lambda: {"code": 0.65, "planning": 0.34, "general": 0.01})

    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        model_d = d.pop("model", {})
        allowed = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**allowed)
        if isinstance(model_d, ModelConfig):
            cfg.model = model_d
        elif isinstance(model_d, dict):
            cfg.model = ModelConfig.from_dict(model_d)
        return cfg


def load_train_config(path: str | Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return TrainConfig.from_dict(raw)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
