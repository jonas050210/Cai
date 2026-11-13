"""Preflight check before spending GPU hours.

``codeplan-doctor`` verifies environment, config, tokenizer, packed data and
checkpoint space, and can measure real throughput/VRAM for the exact config
(``--bench-steps``). Exit code 1 means at least one hard error.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import torch

from .config import TrainConfig, load_train_config
from .data import dtype_for_vocab, load_meta
from .model import CodePlanLM
from .tokenizer import CodePlanTokenizer
from .trainer import autocast_context, build_optimizer
from .utils import get_device, precision_dtype

OK = "ok"
WARN = "warn"
ERROR = "error"


def _line(level: str, message: str) -> tuple[str, str]:
    marker = {OK: "[ok]  ", WARN: "[warn]", ERROR: "[err] "}[level]
    print(f"{marker} {message}")
    return level, message


def check_environment(device_name: str) -> list[tuple[str, str]]:
    results = []
    python = platform.python_version()
    level = OK if sys.version_info >= (3, 10) else ERROR
    results.append(_line(level, f"python {python} (requires >= 3.10)"))
    results.append(_line(OK, f"torch {torch.__version__}"))

    device = get_device(device_name)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        free = torch.cuda.mem_get_info(device)[0] / 1e9
        results.append(_line(OK, f"cuda device: {name}, {total:.1f} GB total, {free:.1f} GB free"))
        bf16 = torch.cuda.is_bf16_supported()
        results.append(_line(OK if bf16 else WARN, f"bf16 support: {bf16}"))
        results.append(
            _line(OK, f"tf32 matmul allowed: {torch.backends.cuda.matmul.allow_tf32}")
        )
    else:
        results.append(_line(WARN, "no CUDA device: training will run on CPU (slow)"))
    return results


def check_config(cfg: TrainConfig) -> list[tuple[str, str]]:
    results = []
    errors = cfg.validate()
    if errors:
        for error in errors:
            results.append(_line(ERROR, f"config: {error}"))
    else:
        results.append(_line(OK, "config passes validation"))
    results.append(
        _line(
            OK,
            f"batch {cfg.batch_size} x accum {cfg.grad_accum} x seq {cfg.seq_len}"
            f" = {cfg.tokens_per_step():,} tokens/step",
        )
    )
    tokens = cfg.total_tokens()
    if tokens >= 1e9:
        human = f"{tokens / 1e9:.2f}B"
    elif tokens >= 1e6:
        human = f"{tokens / 1e6:.0f}M"
    else:
        human = f"{tokens / 1e3:.0f}k"
    results.append(_line(OK, f"{cfg.max_steps:,} steps = {human} tokens"))
    if cfg.warmup_steps >= cfg.max_steps // 10:
        results.append(
            _line(WARN, f"warmup {cfg.warmup_steps} steps is >10% of max_steps")
        )
    return results


def check_tokenizer(cfg: TrainConfig) -> list[tuple[str, str]]:
    path = Path(cfg.tokenizer_path)
    if not path.exists():
        return [_line(ERROR, f"tokenizer not found: {path} (run codeplan-tokenizer)")]
    tokenizer = CodePlanTokenizer.load(path)
    # Training always uses the tokenizer's vocabulary, so verify against that.
    cfg.model.vocab_size = tokenizer.vocab_size
    configure = cfg.model.vocab_size
    level = OK if configure == tokenizer.vocab_size else WARN
    return [
        _line(OK, f"tokenizer {path} (vocab {tokenizer.vocab_size:,})"),
        _line(
            level,
            f"config vocab {configure:,} will be overridden by tokenizer ({tokenizer.vocab_size:,})",
        ),
    ]


def check_data(cfg: TrainConfig) -> list[tuple[str, str]]:
    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        return [_line(ERROR, f"packed data dir missing: {data_dir} (run codeplan-pack)")]
    try:
        meta = load_meta(data_dir)
    except FileNotFoundError:
        return [_line(ERROR, f"meta.json missing in {data_dir}")]

    results = [_line(OK, f"packed data {data_dir} (dtype {meta.get('dtype')})")]
    dtype = dtype_for_vocab(int(meta.get("vocab_size", 0)))
    if meta.get("dtype") != dtype.name:
        results.append(
            _line(WARN, f"meta dtype {meta.get('dtype')} vs expected {dtype.name}")
        )

    if Path(cfg.tokenizer_path).exists():
        vocab = CodePlanTokenizer.load(cfg.tokenizer_path).vocab_size
        if meta.get("vocab_size") != vocab:
            results.append(
                _line(
                    ERROR,
                    f"packed vocab {meta.get('vocab_size')} != tokenizer vocab {vocab};"
                    " re-pack the data",
                )
            )

    split_info = meta.get("splits", {}).get("train", {})
    files = split_info.get("files", {})
    tokens = split_info.get("tokens", {})
    total_tokens = int(tokens.get("all", 0))
    results.append(_line(OK, f"train tokens: {total_tokens:,} in {len(files)} domain files"))
    if total_tokens == 0:
        results.append(_line(ERROR, "train split is empty"))
        return results

    needed = cfg.total_tokens()
    epochs = needed / max(1, total_tokens)
    level = OK if epochs <= 2 else WARN if epochs <= 4 else ERROR
    results.append(
        _line(
            level,
            f"{needed / 1e6:.0f}M training tokens over {total_tokens / 1e6:.1f}M available"
            f" = {epochs:.2f} epochs",
        )
    )

    missing = [name for name in cfg.mixture if name not in files and cfg.mixture[name] > 0]
    if missing:
        results.append(
            _line(
                WARN,
                f"mixture domains without data: {missing} (weights will be renormalised)",
            )
        )
    if "val_all.bin" not in meta.get("splits", {}).get("val", {}).get("files", {}).values():
        results.append(_line(WARN, "no validation split packed; validation is disabled"))
    return results


def check_model(cfg: TrainConfig, device_name: str) -> list[tuple[str, str]]:
    device = get_device(device_name)
    model = CodePlanLM(cfg.model).to(device)
    params = model.num_parameters()
    embedding = cfg.model.vocab_size * cfg.model.d_model
    results = [
        _line(OK, f"model {params / 1e6:.2f}M parameters"),
        _line(
            OK,
            f"embedding share: {embedding / params:.1%} of parameters"
            f" ({cfg.model.vocab_size:,} x {cfg.model.d_model})",
        ),
        _line(
            OK,
            f"tokens per parameter: {cfg.total_tokens() / params:.2f} (Chinchilla ~20)",
        ),
    ]
    del model
    return results


def benchmark(cfg: TrainConfig, device_name: str, steps: int) -> list[tuple[str, str]]:
    """Measure real throughput (and VRAM) with the configured shapes."""
    device = get_device(device_name)
    dtype = precision_dtype(device, cfg.precision)
    model = CodePlanLM(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
        if device.type == "cuda"
        else None
    )
    vocab = cfg.model.vocab_size
    tokens = cfg.batch_size * cfg.seq_len

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    model.train()
    start = time.time()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randint(0, vocab, (cfg.batch_size, cfg.seq_len), device=device)
        with autocast_context(device, dtype):
            _, loss, _ = model(x, labels=x)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.time() - start
        if elapsed > 60 and step >= 2:
            steps_done = step + 1
            break
    else:
        steps_done = steps

    elapsed = time.time() - start
    micro_per_sec = steps_done / max(elapsed, 1e-9)
    tokens_per_sec = micro_per_sec * tokens
    optimizer_steps_per_sec = micro_per_sec / cfg.grad_accum
    eta_hours = cfg.max_steps / max(optimizer_steps_per_sec, 1e-9) / 3600

    results = [
        _line(OK, f"throughput: {tokens_per_sec:,.0f} tokens/s (micro-batch, {steps_done} steps)"),
        _line(OK, f"estimated wall clock for {cfg.max_steps:,} steps: {eta_hours:.1f} h"),
    ]
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        level = OK if peak < total * 0.9 else WARN
        results.append(_line(level, f"peak VRAM: {peak:.2f} GB of {total:.1f} GB"))
    return results


def check_disk(cfg: TrainConfig) -> list[tuple[str, str]]:
    params = (
        cfg.model.vocab_size * cfg.model.d_model
        + cfg.model.n_layers * (4 * cfg.model.d_model**2 + 3 * cfg.model.d_model * cfg.model.intermediate_size)
        + cfg.model.d_model
    )
    per_checkpoint = params * 12 / 1e9  # weights + Adam moments in fp32
    kept = max(1, cfg.keep_last_checkpoints) + 2
    needed = per_checkpoint * kept
    free = shutil.disk_usage(Path(cfg.out_dir).parent if Path(cfg.out_dir).parent.exists() else ".").free / 1e9
    level = OK if free > needed * 1.5 else WARN if free > needed else ERROR
    return [
        _line(
            level,
            f"disk: {free:.1f} GB free, ~{needed:.1f} GB for {kept} checkpoints"
            f" ({per_checkpoint * 1000:.0f} MB each)",
        )
    ]


def run(cfg: TrainConfig, device_name: str, bench_steps: int) -> dict:
    results = []
    results += check_environment(device_name)
    results += check_config(cfg)
    results += check_tokenizer(cfg)
    results += check_data(cfg)
    results += check_model(cfg, device_name)
    results += check_disk(cfg)
    if bench_steps > 0:
        results += benchmark(cfg, device_name, bench_steps)
    counts = {level: sum(1 for lvl, _ in results if lvl == level) for level in (OK, WARN, ERROR)}
    print(
        f"\nsummary: {counts[OK]} ok, {counts[WARN]} warnings, {counts[ERROR]} errors"
    )
    return {"results": results, "counts": counts, "config": cfg.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight check before training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--bench-steps", type=int, default=0, help="Measure throughput/VRAM with N steps"
    )
    parser.add_argument("--json", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    report = run(cfg, args.device, args.bench_steps)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"Wrote {out}")
    sys.exit(1 if report["counts"][ERROR] else 0)


if __name__ == "__main__":
    main()
