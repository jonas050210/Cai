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
    # Grouped-query attention: fewer K/V heads than Q heads shrinks the KV cache
    # and speeds up generation. 0 means "same as n_heads" (classic MHA).
    n_kv_heads: int = 0
    # Per-head RMSNorm on q and k before RoPE. Stabilises long runs and allows
    # a higher learning rate (used by modern small models, e.g. Gemma 2).
    qk_norm: bool = False
    # Auxiliary loss on logsumexp(logits): keeps the softmax from drifting.
    z_loss_coef: float = 0.0

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
    torch_compile: bool = False  # +30-80% on Ada, costs ~1 min of compile time
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

    # checkpoints / early stopping
    keep_last_checkpoints: int = 2  # 0 keeps everything
    early_stop_patience: int = 0  # 0 disables early stopping

    # SFT only
    sft_data: str | None = None
    sft_max_len: int = 2048
    sft_val_ratio: float = 0.05

    # Pretraining domain mixture. Target: 99% code + planning.
    mixture: dict[str, float] = field(default_factory=lambda: {"code": 0.65, "planning": 0.34, "general": 0.01})

    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the config is usable."""
        errors: list[str] = []
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.grad_accum < 1:
            errors.append("grad_accum must be >= 1")
        if self.seq_len < 8:
            errors.append("seq_len must be >= 8")
        if self.seq_len > self.model.max_seq_len:
            errors.append(
                f"seq_len ({self.seq_len}) exceeds model.max_seq_len ({self.model.max_seq_len})"
            )
        if self.model.d_model % self.model.n_heads != 0:
            errors.append("model.d_model must be divisible by model.n_heads")
        if self.model.n_layers < 1 or self.model.n_heads < 1:
            errors.append("model.n_layers and model.n_heads must be >= 1")
        if self.model.vocab_size < 1:
            errors.append("model.vocab_size must be >= 1")
        if not 0.0 <= self.model.dropout < 1.0:
            errors.append("model.dropout must be in [0, 1)")
        if self.lr <= 0:
            errors.append("lr must be > 0")
        if self.min_lr < 0 or self.min_lr > self.lr:
            errors.append("min_lr must be in [0, lr]")
        if self.max_steps < 1:
            errors.append("max_steps must be >= 1")
        if self.warmup_steps >= self.max_steps:
            errors.append("warmup_steps must be smaller than max_steps")
        if self.log_every < 1 or self.save_every < 1 or self.eval_every < 1:
            errors.append("log_every, save_every and eval_every must be >= 1")
        if self.eval_iters < 1:
            errors.append("eval_iters must be >= 1 (set eval_every high to skip validation)")
        if self.precision not in {"bf16", "fp16", "fp32"}:
            errors.append("precision must be bf16, fp16 or fp32")
        if self.keep_last_checkpoints < 0:
            errors.append("keep_last_checkpoints must be >= 0")
        if self.early_stop_patience < 0:
            errors.append("early_stop_patience must be >= 0")
        if not 0.0 <= self.sft_val_ratio < 0.9:
            errors.append("sft_val_ratio must be in [0, 0.9)")
        if not self.mixture:
            errors.append("mixture must not be empty")
        elif any(value < 0 for value in self.mixture.values()) or sum(self.mixture.values()) <= 0:
            errors.append("mixture weights must be >= 0 and sum to more than 0")
        return errors

    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum * self.seq_len

    def total_tokens(self) -> int:
        return self.max_steps * self.tokens_per_step()

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


ALLOWED_EXTRA_KEYS = {"notes", "description", "comment"}


def unknown_keys(raw: dict, cls: type) -> list[str]:
    """Keys that would be silently ignored by ``from_dict``."""
    return sorted(set(raw) - set(cls.__dataclass_fields__) - ALLOWED_EXTRA_KEYS)


def load_train_config(path: str | Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ignored = unknown_keys(raw, TrainConfig)
    if ignored:
        print(f"Warning: ignoring unknown keys in {path}: {', '.join(ignored)}")
    model_raw = raw.get("model") if isinstance(raw.get("model"), dict) else None
    if model_raw:
        ignored_model = unknown_keys(model_raw, ModelConfig)
        if ignored_model:
            print(
                f"Warning: ignoring unknown model keys in {path}: {', '.join(ignored_model)}"
            )
    return TrainConfig.from_dict(raw)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
