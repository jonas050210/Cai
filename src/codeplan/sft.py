"""Supervised fine-tuning for structured coding/planning answers."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_train_config, save_json
from .data import SFTDataset, sft_collate, split_sft_dataset
from .model import CodePlanLM
from .tokenizer import CodePlanTokenizer
from .trainer import build_optimizer, lr_lambda, train_loop
from .utils import get_device, load_checkpoint, precision_dtype, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT CodePlan-LM on structured examples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--init", default=None, help="Pretrained checkpoint to initialize from")
    parser.add_argument("--data", default=None, help="Override SFT JSONL path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    if args.init:
        cfg.init_checkpoint = args.init
    if args.data:
        cfg.sft_data = args.data
    if args.device:
        cfg.device = args.device
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if not cfg.sft_data:
        raise ValueError("SFT config needs sft_data or --data")

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    dtype = precision_dtype(device, cfg.precision)
    tokenizer = CodePlanTokenizer.load(cfg.tokenizer_path)
    cfg.model.vocab_size = tokenizer.vocab_size
    cfg.seq_len = cfg.sft_max_len

    model = CodePlanLM(cfg.model).to(device)
    if cfg.init_checkpoint:
        load_checkpoint(cfg.init_checkpoint, model, device=device)
        print(f"Initialized SFT from {cfg.init_checkpoint}")
    else:
        print("Warning: SFT from random init is usually not useful.")

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_lambda(step, cfg))
    use_scaler = dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else None

    start_step = 0
    best_val = float("inf")
    bad_evals = 0
    if cfg.resume:
        payload = load_checkpoint(cfg.resume, model, optimizer, scheduler, scaler, device, restore_rng=True)
        start_step = int(payload["step"]) + 1
        best_val = float(payload.get("val_loss") or float("inf"))
        bad_evals = int(payload.get("bad_evals") or 0)

    ds = SFTDataset(cfg.sft_data, tokenizer, max_len=cfg.sft_max_len)
    train_ds, val_ds = split_sft_dataset(ds, cfg.sft_val_ratio, cfg.seed)
    collate = partial(sft_collate, pad_id=tokenizer.pad_id)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("SFT dataset smaller than batch_size; reduce batch_size.")

    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
            pin_memory=(device.type == "cuda"),
        )
    print(f"SFT examples: {len(train_ds)} train / {0 if val_ds is None else len(val_ds)} val")

    save_json(Path(cfg.out_dir) / "config.json", cfg.to_dict())
    train_loop(
        model, loader, cfg, device, dtype, optimizer, scheduler, scaler,
        start_step=start_step, val_loader=val_loader, mode="sft",
        best_val=best_val, bad_evals=bad_evals,
    )


if __name__ == "__main__":
    main()
