"""Pretrain CodePlan-LM from random initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_train_config, save_json
from .data import MixturePackedDataset, PackedDataset, load_meta
from .model import CodePlanLM
from .tokenizer import CodePlanTokenizer
from .trainer import build_optimizer, lr_lambda, train_loop, worker_init_fn
from .utils import get_device, load_checkpoint, precision_dtype, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain CodePlan-LM from scratch.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Override resume checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    if args.resume:
        cfg.resume = args.resume
    if args.device:
        cfg.device = args.device
    if args.precision:
        cfg.precision = args.precision
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    dtype = precision_dtype(device, cfg.precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = CodePlanTokenizer.load(cfg.tokenizer_path)
    cfg.model.vocab_size = tokenizer.vocab_size
    meta = load_meta(cfg.data_dir)
    if meta["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("Packed data vocab_size does not match tokenizer")

    model = CodePlanLM(cfg.model).to(device)
    print(f"Model parameters: {model.num_parameters()/1e6:.2f}M")
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_lambda(step, cfg))
    use_scaler = dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else None

    start_step = 0
    best_val = float("inf")
    bad_evals = 0
    if cfg.resume:
        payload = load_checkpoint(
            cfg.resume, model, optimizer, scheduler, scaler, device, restore_rng=True
        )
        start_step = int(payload["step"]) + 1
        best_val = float(payload.get("val_loss") or float("inf"))
        bad_evals = int(payload.get("bad_evals") or 0)
        print(f"Resumed from {cfg.resume} at step {start_step} (best_val={best_val:.4f})")
    elif cfg.init_checkpoint:
        load_checkpoint(cfg.init_checkpoint, model, device=device)
        print(f"Initialized weights from {cfg.init_checkpoint}")

    weights = cfg.mixture or meta.get("target_mixture", {"code": 0.65, "planning": 0.34, "general": 0.01})
    train_ds = MixturePackedDataset(cfg.data_dir, "train", cfg.seq_len, weights, seed=cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=worker_init_fn if cfg.num_workers > 0 else None,
    )

    val_loader = None
    val_path = Path(cfg.data_dir) / "val_all.bin"
    if val_path.exists():
        try:
            val_ds = PackedDataset(val_path, cfg.seq_len, dtype=meta["dtype"])
            val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        except ValueError:
            val_loader = None

    save_json(Path(cfg.out_dir) / "config.json", cfg.to_dict())
    train_loop(
        model, train_loader, cfg, device, dtype, optimizer, scheduler, scaler,
        start_step=start_step, val_loader=val_loader, mode="pretrain",
        best_val=best_val, bad_evals=bad_evals,
    )


if __name__ == "__main__":
    main()
