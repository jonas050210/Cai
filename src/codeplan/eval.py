"""Evaluate perplexity on packed validation/test data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import PackedDataset, load_meta
from .generate import load_model_from_checkpoint
from .trainer import autocast_context
from .utils import get_device, precision_dtype


@torch.no_grad()
def evaluate(checkpoint: str, data_dir: str, split: str, seq_len: int, batch_size: int,
             max_batches: int, device_name: str, precision: str, tokenizer: str | None = None) -> dict:
    device = get_device(device_name)
    model, tok, _ = load_model_from_checkpoint(checkpoint, tokenizer, device_name)
    dtype = precision_dtype(device, precision)
    meta = load_meta(data_dir)
    path = Path(data_dir) / f"{split}_all.bin"
    if not path.exists():
        raise FileNotFoundError(path)
    ds = PackedDataset(path, seq_len, dtype=meta["dtype"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()

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
        "split": split,
        "batches": len(losses),
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "vocab_size": tok.vocab_size,
    }


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
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint, args.data_dir, args.split, args.seq_len, args.batch_size,
        args.max_batches, args.device, args.precision, args.tokenizer,
    )
    print(result)


if __name__ == "__main__":
    main()
