"""Average checkpoints into one model ("model soup").

Averaging the last few checkpoints of a run is free quality: no extra training,
no extra data, usually 2-5% lower validation loss. All inputs must share the
same architecture.

    codeplan-soup --dir checkpoints/pretrain_100m --last 5 --out checkpoints/pretrain_100m/soup.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_payload(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError(f"{path} has no 'model' weights")
    return payload


def average_state_dicts(states: list[dict], weights: list[float] | None = None) -> dict:
    """Weighted average of state dicts, computed in float32."""
    if not states:
        raise ValueError("No state dicts given")
    if weights is None:
        weights = [1.0] * len(states)
    if len(weights) != len(states):
        raise ValueError(f"{len(weights)} weights for {len(states)} checkpoints")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights must sum to a positive number")

    reference = states[0]
    for other in states[1:]:
        if set(other) != set(reference):
            missing = set(reference) ^ set(other)
            raise ValueError(f"Checkpoints differ in parameters, e.g. {sorted(missing)[:3]}")
        for key, tensor in reference.items():
            if other[key].shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: {tuple(tensor.shape)} vs {tuple(other[key].shape)}"
                )

    averaged = {}
    for key, tensor in reference.items():
        if tensor.is_floating_point():
            acc = torch.zeros_like(tensor, dtype=torch.float32)
            for weight, state in zip(weights, states):
                acc += state[key].to(torch.float32) * weight
            averaged[key] = (acc / total).to(tensor.dtype)
        else:
            averaged[key] = tensor.clone()
    return averaged


def soup(paths: list[Path], weights: list[float] | None = None) -> dict:
    payloads = [load_payload(p) for p in paths]
    states = [p["model"] for p in payloads]
    merged = average_state_dicts(states, weights)
    steps = [int(p.get("step", 0)) for p in payloads]
    config = payloads[-1].get("config", {})
    return {
        "step": max(steps),
        "model": merged,
        "config": config,
        "extra": {
            "soup": {
                "sources": [str(p) for p in paths],
                "steps": steps,
                "weights": weights or [1.0] * len(paths),
            }
        },
    }


def collect_paths(
    checkpoints: list[str] | None, directory: str | None, last: int
) -> list[Path]:
    paths = [Path(p) for p in (checkpoints or [])]
    if directory:
        found = sorted(Path(directory).glob("step_*.pt"))
        if not found:
            raise FileNotFoundError(f"No step_*.pt in {directory}")
        paths.extend(found[-last:] if last > 0 else found)
    if not paths:
        raise SystemExit("Pass --checkpoints a.pt b.pt or --dir <run dir> [--last N]")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing checkpoint(s): {', '.join(missing)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Average checkpoints into one model.")
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Explicit checkpoint list")
    parser.add_argument("--dir", default=None, help="Run directory; uses its step_*.pt files")
    parser.add_argument("--last", type=int, default=5, help="How many of them (default 5)")
    parser.add_argument("--weights", default=None, help="Comma separated weights, e.g. 1,1,2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = collect_paths(args.checkpoints, args.dir, args.last)
    weights = None
    if args.weights:
        weights = [float(v) for v in args.weights.split(",") if v.strip()]
        if len(weights) != len(paths):
            raise SystemExit(f"--weights needs {len(paths)} values, got {len(weights)}")

    payload = soup(paths, weights)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(json.dumps(
        {
            "out": str(out),
            "checkpoints": len(paths),
            "step": payload["step"],
            "sources": [str(p) for p in paths],
        },
        indent=2,
    ))
    print(f"Verify with: codeplan-eval --checkpoint {out} --data-dir data/packed")


if __name__ == "__main__":
    main()
