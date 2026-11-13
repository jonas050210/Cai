"""Shared runtime utilities."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def precision_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    if precision == "fp32" or device.type != "cuda":
        return None
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"Unknown precision: {precision}")


def _git_sha() -> tuple[str, bool]:
    """Return (short sha, dirty) without failing outside a repository."""
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout.strip()
        )
        return sha or "unknown", dirty
    except Exception:  # pragma: no cover - depends on the environment
        return "unknown", False


def run_metadata(config: dict, extra: dict | None = None) -> dict:
    """Everything needed to reproduce a run six months later."""
    import platform
    import socket
    import sys
    import time

    sha, dirty = _git_sha()
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": sha,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "config": config,
    }
    if extra:
        payload.update(extra)
    return payload


def configure_cpu_threads(device: torch.device | str) -> int:
    """Use every core for CPU training; leave GPU runs to the default thread pool."""
    if str(device).startswith("cuda"):
        return torch.get_num_threads()
    cpus = os.cpu_count() or 4
    threads = max(1, cpus)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, cpus // 4)))
    except RuntimeError:
        pass  # already initialised by another call
    return threads


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the eager module behind a ``torch.compile`` wrapper."""
    return getattr(model, "_orig_mod", model)


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    """Compile the model when asked; silently fall back when unsupported."""
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile is not available in this torch build; running eagerly")
        return model
    try:
        return torch.compile(model)
    except Exception as exc:  # pragma: no cover - depends on the platform
        print(f"torch.compile failed ({exc}); running eagerly")
        return model


def _rng_state(state: Any) -> torch.Tensor:
    """Normalise a stored RNG state to a CPU uint8 tensor."""
    if isinstance(state, torch.Tensor):
        return state.detach().to("cpu", dtype=torch.uint8)
    return torch.as_tensor(state, dtype=torch.uint8)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    step: int,
    config: dict,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": unwrap_model(model).state_dict(),
        "config": config,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: torch.device | str = "cpu",
    restore_rng: bool = False,
) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and "rng" in payload:
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        # map_location moved every tensor to `device`, including the RNG state.
        # torch.set_rng_state only accepts a CPU ByteTensor, so move it back.
        torch.set_rng_state(_rng_state(rng.get("torch")))
        cuda_states = rng.get("cuda")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([_rng_state(state) for state in cuda_states])
    return payload


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
