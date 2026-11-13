#!/usr/bin/env bash
# Two-phase curriculum: train on short windows first, then on the full context.
#
#   bash scripts/curriculum.sh --config configs/pretrain_100m_4060ti.json
#
# Short sequences in the first phase mean more samples per second while the
# model is still learning local structure; the second phase sees the real
# 2048-token context. Both phases share one checkpoint, so nothing is repeated.
#
# Why this helps: at seq 512 the same number of steps covers 4x more distinct
# text, which is where the early loss drop comes from. The model architecture
# (max_seq_len) never changes, so the resume is exact.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

CONFIG="configs/pretrain_100m_4060ti.json"
SHORT_SEQ=512
SHORT_FRACTION=0.2      # share of max_steps spent on short windows
MAX_STEPS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --short-seq) SHORT_SEQ="$2"; shift 2 ;;
    --short-fraction) SHORT_FRACTION="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p artifacts
python3 - "$CONFIG" "$SHORT_SEQ" "$SHORT_FRACTION" "$MAX_STEPS" <<'PY'
import json
import sys

config_path, short_seq, fraction, max_steps = sys.argv[1:5]
cfg = json.load(open(config_path))
total = int(max_steps) if max_steps else cfg["max_steps"]
phase1 = max(1, int(total * float(fraction)))

phase1_cfg = dict(cfg)
phase1_cfg["seq_len"] = int(short_seq)
phase1_cfg["max_steps"] = phase1
phase1_cfg["warmup_steps"] = min(cfg["warmup_steps"], max(1, phase1 // 10))
phase1_cfg["log_every"] = cfg.get("log_every", 20)
json.dump(phase1_cfg, open("artifacts/curriculum_phase1.json", "w"), indent=2)

phase2_cfg = dict(cfg)
phase2_cfg["max_steps"] = total
phase2_cfg["warmup_steps"] = 1          # already warm, keep the LR schedule
json.dump(phase2_cfg, open("artifacts/curriculum_phase2.json", "w"), indent=2)

print(f"phase 1: seq {short_seq}, steps 0-{phase1}")
print(f"phase 2: seq {cfg['seq_len']}, steps {phase1}-{total}")
PY

echo "=== phase 1 (short windows) ==="
codeplan-pretrain --config artifacts/curriculum_phase1.json

echo "=== phase 2 (full context) ==="
codeplan-pretrain --config artifacts/curriculum_phase2.json

echo "done: checkpoints in $(python3 -c "import json;print(json.load(open('artifacts/curriculum_phase2.json'))['out_dir'])")"
