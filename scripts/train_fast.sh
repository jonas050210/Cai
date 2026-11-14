#!/usr/bin/env bash
# Training with the throughput settings for a desktop RTX 4060 (8 GB).
#
#   bash scripts/train_fast.sh                 # 12,000 steps (~390M tokens)
#   bash scripts/train_fast.sh --steps 20000
#   bash scripts/train_fast.sh --config configs/pretrain_100m_4060ti.json   # safe VRAM
#
# Differences to the plain config: micro-batch 2 instead of 1 with gradient
# checkpointing off, torch.compile on, 8 loader workers. Same 32,768 tokens per
# optimizer step, so the learning curve stays comparable.
#
# Resume works as usual: rerun the same command, or Ctrl+C.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

CONFIG="configs/pretrain_100m_fast.json"
STEPS=12000
SHORT_SEQ=512

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --short-seq) SHORT_SEQ="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Fewer thread switches in the loader, no caching allocator fragmentation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

GPU=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  echo "GPU: $GPU"
fi

bash scripts/curriculum.sh --config "$CONFIG" --max-steps "$STEPS" --short-seq "$SHORT_SEQ"

cat <<TXT

Done. Next:
  codeplan-eval --checkpoint checkpoints/pretrain_100m_fast/last.pt --data-dir data/packed --token-classes
  codeplan-soup --dir checkpoints/pretrain_100m_fast --last 3 --out checkpoints/pretrain_100m_fast/soup.pt
TXT
