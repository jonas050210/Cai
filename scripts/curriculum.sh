#!/usr/bin/env bash
# Two-phase curriculum: train on short windows first, then on the full context.
#
#   bash scripts/curriculum.sh --config configs/pretrain_100m_4060ti.json
#
# Phase 1 uses the same number of *tokens* as its share of the run, but at seq
# 512 - so it sees four times as many distinct documents while the model is still
# learning local structure. Phase 2 then spends the remaining tokens on the full
# 2048 context. Both phases share one checkpoint, so nothing is trained twice.
#
# The architecture (max_seq_len) never changes, so the resume is exact. Time is
# roughly the same as an uninterrupted run; the win is sample diversity early.
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
# same token budget as its share of the run, four times the steps
phase1 = max(1, int(total * float(fraction) * (cfg["seq_len"] / int(short_seq))))

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

tokens1 = phase1 * cfg["batch_size"] * cfg["grad_accum"] * int(short_seq)
tokens2 = (total - max(phase1, 0)) * cfg["batch_size"] * cfg["grad_accum"] * cfg["seq_len"]
print(f"phase 1: seq {short_seq}, {phase1} steps = {tokens1/1e6:.0f}M tokens")
print(f"phase 2: seq {cfg['seq_len']}, {total - phase1} steps = {max(0, tokens2)/1e6:.0f}M tokens")
PY

echo "=== phase 1 (short windows) ==="
codeplan-pretrain --config artifacts/curriculum_phase1.json

echo "=== phase 2 (full context) ==="
codeplan-pretrain --config artifacts/curriculum_phase2.json

echo "done: checkpoints in $(python3 -c "import json;print(json.load(open('artifacts/curriculum_phase2.json'))['out_dir'])")"
