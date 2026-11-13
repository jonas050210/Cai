"""Export a checkpoint to safetensors plus a portable model config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ModelConfig
from .generate import load_model_from_checkpoint
from .model import CodePlanLM

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def export(
    checkpoint: str,
    out_path: str | Path,
    dtype: str = "fp32",
    tokenizer_path: str | None = None,
) -> dict:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit(
            "safetensors is required: pip install 'codeplan-lm[serve]' or pip install safetensors"
        ) from exc

    if dtype not in DTYPES:
        raise ValueError(f"Unknown dtype {dtype}; choose from {sorted(DTYPES)}")

    model, tokenizer, payload = load_model_from_checkpoint(checkpoint, tokenizer_path, "cpu")
    target = DTYPES[dtype]

    # Tied embeddings (and any other alias) share storage; safetensors refuses
    # duplicates. Keep the first name, drop the aliases and record them.
    seen: dict[tuple, str] = {}
    tensors = {}
    dropped: list[str] = []
    for name, tensor in model.state_dict().items():
        key = (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        if key in seen:
            dropped.append(name)
            continue
        seen[key] = name
        tensors[name] = tensor.detach().to(target).contiguous()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))

    config_path = out_path.with_suffix(".config.json")
    config = {
        "model": model.config_dict(),
        "tokenizer": str(tokenizer_path or "tokenizer.json"),
        "source_checkpoint": str(checkpoint),
        "step": payload.get("step"),
        "dtype": dtype,
        "tied_embeddings": bool(model.cfg.tie_embeddings),
        "aliases": dropped,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {
        "weights": str(out_path),
        "config": str(config_path),
        "dtype": dtype,
        "tensors": len(tensors),
        "aliases": dropped,
        "parameters": model.num_parameters(),
        "vocab_size": tokenizer.vocab_size,
    }


def load_exported(weights: str | Path, config_path: str | Path) -> CodePlanLM:
    """Rebuild a model from an exported safetensors file (inverse of :func:`export`)."""
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("safetensors is required to load an exported model") from exc

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = CodePlanLM(ModelConfig.from_dict(config["model"]))
    state = load_file(str(weights))
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"lm_head.weight"} if config.get("tied_embeddings") else set()
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"Exported weights do not match config (missing={missing}, unexpected={unexpected})"
        )
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to safetensors.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="artifacts/model.safetensors")
    parser.add_argument("--dtype", default="fp32", choices=sorted(DTYPES))
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    info = export(args.checkpoint, args.out, args.dtype, args.tokenizer)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
