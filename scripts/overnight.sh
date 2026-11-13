#!/usr/bin/env bash
# Start the data run in the background and go to sleep.
#
#   bash scripts/overnight.sh              # corpus only (~30-40 min)
#   bash scripts/overnight.sh --train      # corpus, then pretraining
#
# Everything goes to logs/overnight-<date>.log. Closing the terminal is fine;
# the process keeps running. Check progress any time with:
#
#   tail -f logs/overnight-*.log
#
# If it is still running in the morning, that is normal - rerun the same
# command later and every finished step is skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

TRAIN=0
WORKERS=12
CONFIG="configs/pretrain_100m_4060ti.json"
STEPS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) TRAIN=1; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p logs
LOG="logs/overnight-$(date +%Y%m%d-%H%M).log"

{
  echo "=== $(date) ==="
  bash scripts/build_corpus.sh --workers "$WORKERS"
  if (( TRAIN )); then
    echo "=== training $(date) ==="
    ARGS=(--config "$CONFIG")
    [[ -n "$STEPS" ]] && ARGS+=(--max-steps "$STEPS")
    bash scripts/curriculum.sh "${ARGS[@]}"
  fi
  echo "=== done $(date) ==="
} > "$LOG" 2>&1 &

PID=$!
echo "Started in the background (pid $PID)."
echo "Log: $LOG"
echo "Watch: tail -f $LOG"
echo "Stop:  kill $PID   (training saves a checkpoint first)"
if (( ! TRAIN )); then
  echo
  echo "Training is not part of this run. Start it whenever you want with:"
  echo "  bash scripts/curriculum.sh --config $CONFIG"
fi
