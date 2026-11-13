"""Evaluate perplexity on packed validation/test data, optionally per domain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import PackedDataset, load_meta
from .generate import load_model_from_checkpoint
from .trainer import autocast_context
from .utils import get_device, precision_dtype


@torch.no_grad()
def evaluate_file(
    model,
    path: Path,
    seq_len: int,
    batch_size: int,
    max_batches: int,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict:
    ds = PackedDataset(path, seq_len, dtype=load_meta(path.parent)["dtype"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    losses = []
    for i, (x, _y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        with autocast_context(device, dtype):
            _, loss, _ = model(x, labels=x)
        losses.append(loss.item())

    mean_loss = sum(losses) / max(1, len(losses))
    return {
        "file": path.name,
        "batches": len(losses),
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
    }


def evaluate(
    checkpoint: str,
    data_dir: str,
    split: str,
    seq_len: int,
    batch_size: int,
    max_batches: int,
    device_name: str,
    precision: str,
    tokenizer: str | None = None,
    per_domain: bool = False,
) -> dict:
    device = get_device(device_name)
    model, tok, _ = load_model_from_checkpoint(checkpoint, tokenizer, device_name)
    dtype = precision_dtype(device, precision)
    model.eval()

    if not per_domain:
        path = Path(data_dir) / f"{split}_all.bin"
        if not path.exists():
            raise FileNotFoundError(path)
        result = evaluate_file(
            model, path, seq_len, batch_size, max_batches, device, dtype
        )
        result.update({"split": split, "vocab_size": tok.vocab_size})
        return result

    domains = {}
    for path in sorted(Path(data_dir).glob(f"{split}_*.bin")):
        domain = path.stem[len(split) + 1:]
        if domain == "all":
            continue
        try:
            domains[domain] = evaluate_file(
                model, path, seq_len, batch_size, max_batches, device, dtype
            )
        except ValueError as exc:  # file shorter than seq_len + 1
            domains[domain] = {"file": path.name, "error": str(exc)}
    if not domains:
        raise FileNotFoundError(f"No {split}_<domain>.bin files in {data_dir}")
    weights = {name: d["loss"] for name, d in domains.items() if "loss" in d}
    if not weights:
        raise ValueError(f"No {split}_<domain>.bin file is long enough for seq_len={seq_len}")
    result = {
        "split": split,
        "vocab_size": tok.vocab_size,
        "domains": domains,
        "mean_domain_loss": sum(weights.values()) / len(weights),
        "mean_domain_perplexity": math.exp(min(sum(weights.values()) / len(weights), 20)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CodePlan-LM perplexity.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--per-domain",
        action="store_true",
        help="Report loss/perplexity per domain file instead of the mixed _all file",
    )
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint, args.data_dir, args.split, args.seq_len, args.batch_size,
        args.max_batches, args.device, args.precision, args.tokenizer, args.per_domain,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
