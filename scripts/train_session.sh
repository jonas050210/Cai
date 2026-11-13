#!/usr/bin/env bash
# Run pretraining in short, resumable sessions with a thermal guard.
#
#   bash scripts/train_session.sh --config configs/pretrain_50m_4060ti.json
#   bash scripts/train_session.sh --session-steps 400 --max-total 2000
#
# Each session trains --session-steps more steps and exits, so the machine can
# be used between sessions. Progress is never lost: the loop resumes from
# last.pt. The loop stops when the GPU is too hot for too long.
set -euo pipefail

CONFIG="configs/pretrain_50m_4060ti.json"
SESSION_STEPS=800
MAX_TOTAL=20000
TEMP_LIMIT=80
TEMP_WAIT=600
POWER_LIMIT=0
DEVICE="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --session-steps) SESSION_STEPS="$2"; shift 2 ;;
    --max-total) MAX_TOTAL="$2"; shift 2 ;;
    --temp-limit) TEMP_LIMIT="$2"; shift 2 ;;
    --temp-wait) TEMP_WAIT="$2"; shift 2 ;;
    --power-limit) POWER_LIMIT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

have_smi() { command -v nvidia-smi >/dev/null 2>&1; }

gpu_temp() {
  if have_smi; then
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1
  else
    echo 0
  fi
}

wait_for_cool() {
  local temp waited
  temp="$(gpu_temp)"
  if (( temp < TEMP_LIMIT )); then return 0; fi
  echo "GPU at ${temp}C (limit ${TEMP_LIMIT}C): waiting up to ${TEMP_WAIT}s"
  waited=0
  while (( waited < TEMP_WAIT )); do
    sleep 30
    waited=$(( waited + 30 ))
    temp="$(gpu_temp)"
    if (( temp < TEMP_LIMIT )); then
      echo "GPU cooled to ${temp}C"
      return 0
    fi
  done
  echo "GPU still hot (${temp}C) after ${TEMP_WAIT}s; stopping for safety"
  return 1
}

if (( POWER_LIMIT > 0 )); then
  if have_smi; then
    echo "Setting power limit to ${POWER_LIMIT}W (needs sudo, failure is fine)"
    sudo -n nvidia-smi -pl "$POWER_LIMIT" || echo "  skipped: no passwordless sudo"
  fi
fi

OUT_DIR="$(python -c "from codeplan.config import load_train_config; print(load_train_config('$CONFIG').out_dir)")"
LAST="$OUT_DIR/last.pt"

current_step() {
  python - "$OUT_DIR" <<'PY'
import sys
from pathlib import Path

import torch

last = Path(sys.argv[1]) / "last.pt"
if not last.exists():
    print(0)
else:
    payload = torch.load(last, map_location="cpu", weights_only=False)
    print(int(payload["step"]) + 1)
PY
}

CURRENT="$(current_step)"
echo "Starting at step ${CURRENT} (target ${MAX_TOTAL})"

while (( CURRENT < MAX_TOTAL )); do
  wait_for_cool || exit 1
  NEXT=$(( CURRENT + SESSION_STEPS ))
  if (( NEXT > MAX_TOTAL )); then NEXT="$MAX_TOTAL"; fi
  echo "=== session: steps ${CURRENT} -> ${NEXT} ==="
  if [[ -f "$LAST" ]]; then
    codeplan-pretrain --config "$CONFIG" --resume "$LAST" --max-steps "$NEXT" --device "$DEVICE"
  else
    codeplan-pretrain --config "$CONFIG" --max-steps "$NEXT" --device "$DEVICE"
  fi
  CURRENT="$NEXT"
  echo "Session done. GPU at $(gpu_temp)C"
done

echo "Reached ${MAX_TOTAL} steps."
