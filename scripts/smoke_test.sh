#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

codeplan-prepare \
  --input data/seed/pretrain_seed.jsonl \
  --out-dir artifacts/smoke/filtered \
  --val-ratio 0.1 \
  --test-ratio 0.1

codeplan-tokenizer \
  --train artifacts/smoke/filtered/train.jsonl \
  --out artifacts/smoke/tokenizer.json \
  --vocab-size 512 \
  --min-frequency 1

codeplan-pack \
  --filtered-dir artifacts/smoke/filtered \
  --tokenizer artifacts/smoke/tokenizer.json \
  --out-dir artifacts/smoke/packed

codeplan-pretrain --config configs/tiny_smoke.json
codeplan-sft --config configs/tiny_sft.json

codeplan-eval \
  --checkpoint artifacts/smoke/sft/last.pt \
  --data-dir artifacts/smoke/packed \
  --split val \
  --seq-len 64 \
  --batch-size 2 \
  --max-batches 2 \
  --device cpu \
  --precision fp32

codeplan-generate \
  --checkpoint artifacts/smoke/sft/last.pt \
  --prompt "Write a Python function add(a, b)." \
  --max-new-tokens 24 \
  --temperature 0.8 \
  --device cpu

echo "Smoke test completed."
