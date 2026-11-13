"""Trainer pieces shared by pretraining and SFT.

Besides the loop itself this module holds small, testable helpers for checkpoint
retention and early stopping.
"""

from __future__ import annotations

import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .utils import append_jsonl, save_checkpoint


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    seed = 1337 + worker_id
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = random.Random(seed + info.dataset.rng.randint(0, 100000))
    random.seed(seed)
    np.random.seed(seed)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)


def lr_lambda(step: int, cfg: TrainConfig) -> float:
    min_ratio = cfg.min_lr / cfg.lr if cfg.lr > 0 else 0.0
    if step < cfg.warmup_steps:
        return (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def prune_checkpoints(out_dir: str | Path, keep: int) -> list[Path]:
    """Keep only the newest ``keep`` step_*.pt files. ``keep <= 0`` disables pruning."""
    if keep <= 0:
        return []
    files = sorted(Path(out_dir).glob("step_*.pt"))
    removed = files[:-keep] if len(files) > keep else []
    for path in removed:
        path.unlink(missing_ok=True)
    return removed


class EarlyStopper:
    """Counts non-improving validations."""

    def __init__(self, patience: int = 0):
        self.patience = patience
        self.bad = 0

    def update(self, val_loss: float, best_val: float) -> tuple[bool, float]:
        """Return ``(should_stop, best_val)``."""
        if val_loss < best_val:
            self.bad = 0
            return False, val_loss
        self.bad += 1
        if self.patience > 0 and self.bad >= self.patience:
            return True, best_val
        return False, best_val


@torch.no_grad()
def evaluate_packed(model, val_loader, device, dtype, max_batches: int) -> float:
    model.eval()
    losses = []
    it = cycle(val_loader)
    for _ in range(max_batches):
        x, _y = next(it)
        x = x.to(device, non_blocking=True)
        with autocast_context(device, dtype):
            _, loss, _ = model(x, labels=x)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def evaluate_sft(model, val_loader, device, dtype, max_batches: int) -> float:
    """Validation loss for SFT batches (input_ids, labels, attention_mask)."""
    model.eval()
    losses = []
    it = cycle(val_loader)
    for _ in range(max_batches):
        x, labels, attention_mask = next(it)
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        with autocast_context(device, dtype):
            _, loss, _ = model(x, labels=labels, attention_mask=attention_mask)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def evaluate(
    mode: str, model, val_loader, device, dtype, max_batches: int
) -> float:
    if mode == "sft":
        return evaluate_sft(model, val_loader, device, dtype, max_batches)
    return evaluate_packed(model, val_loader, device, dtype, max_batches)


def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler | None,
    start_step: int = 0,
    val_loader: DataLoader | None = None,
    mode: str = "pretrain",
    best_val: float = float("inf"),
    bad_evals: int = 0,
) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    stopper = EarlyStopper(cfg.early_stop_patience)
    stopper.bad = bad_evals
    train_it = cycle(train_loader)
    model.train()
    stopped_early = False

    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.seq_len
    t0 = time.time()
    step = start_step - 1

    for step in range(start_step, cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(cfg.grad_accum):
            batch = next(train_it)
            if mode == "sft":
                x, labels, attention_mask = batch
                x = x.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
            else:
                x, _y = batch
                x = x.to(device, non_blocking=True)
                labels = x
                attention_mask = None

            with autocast_context(device, dtype):
                _, loss, _ = model(x, labels=labels, attention_mask=attention_mask)
                loss = loss / cfg.grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item() * cfg.grad_accum

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            elapsed = time.time() - t0
            toks = (step - start_step + 1) * tokens_per_step
            lr = optimizer.param_groups[0]["lr"]
            record = {
                "step": step,
                "loss": accum_loss / cfg.grad_accum,
                "lr": lr,
                "tokens": toks,
                "tokens_per_sec": toks / max(elapsed, 1e-6),
                "mode": mode,
            }
            append_jsonl(log_path, record)
            print(
                f"step {step:6d} | loss {record['loss']:.4f} | lr {lr:.2e} | "
                f"tok/s {record['tokens_per_sec']:.0f}",
                flush=True,
            )
            t0 = time.time()

        if val_loader is not None and (
            step % cfg.eval_every == 0 or step == cfg.max_steps - 1
        ):
            val_loss = evaluate(mode, model, val_loader, device, dtype, cfg.eval_iters)
            ppl = math.exp(min(val_loss, 20))
            append_jsonl(log_path, {"step": step, "val_loss": val_loss, "val_ppl": ppl})
            print(f"step {step:6d} | val_loss {val_loss:.4f} | val_ppl {ppl:.2f}", flush=True)
            should_stop, best_val = stopper.update(val_loss, best_val)
            if val_loss <= best_val:
                save_checkpoint(
                    out_dir / "best.pt", raw_model(model), optimizer, scheduler, scaler,
                    step, cfg.to_dict(), {"val_loss": val_loss, "bad_evals": stopper.bad},
                )
            if should_stop:
                print(f"Early stopping at step {step} (patience {cfg.early_stop_patience}).")
                stopped_early = True

        if step % cfg.save_every == 0 or step == cfg.max_steps - 1:
            save_checkpoint(
                out_dir / "last.pt", raw_model(model), optimizer, scheduler, scaler,
                step, cfg.to_dict(), {"val_loss": best_val, "bad_evals": stopper.bad},
            )
            save_checkpoint(
                out_dir / f"step_{step:07d}.pt", raw_model(model), optimizer, scheduler, scaler,
                step, cfg.to_dict(),
            )
            prune_checkpoints(out_dir, cfg.keep_last_checkpoints)
            if Path(cfg.tokenizer_path).exists():
                shutil.copy(cfg.tokenizer_path, out_dir / "tokenizer.json")

        if stopped_early:
            break

    last_step = step if cfg.max_steps > start_step else start_step
    save_checkpoint(
        out_dir / "last.pt", raw_model(model), optimizer, scheduler, scaler,
        last_step, cfg.to_dict(), {"val_loss": best_val, "bad_evals": stopper.bad},
    )
    print(f"Done. Checkpoints in {out_dir}")
    return {"last_step": last_step, "best_val": best_val, "bad_evals": stopper.bad}
