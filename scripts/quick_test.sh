#!/usr/bin/env bash
# Five-minute end-to-end training test.
#
#   bash scripts/quick_test.sh
#
# Uses the real pilot corpus (data/packed + artifacts/tokenizer.json) when it
# exists, otherwise builds a tiny corpus from data/seed/ so the test always runs
# without any download. Trains configs/quick_test.json (7.5M parameters,
# 150 steps x 1,024 tokens) and prints the validation loss curve.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$ROOT/.venv/bin:$PATH"

if [[ -f artifacts/tokenizer.json && -d data/packed ]]; then
  echo "== real pilot corpus found =="
  CFG=configs/quick_test.json
else
  echo "== no pilot corpus yet: building a mini corpus from data/seed (about 1 min) =="
  echo "   run 'bash scripts/pilot.sh' later for the real 12M-token corpus"
  codeplan-prepare \
    --input data/seed/pretrain_seed.jsonl \
    --out-dir artifacts/quick/filtered \
    --val-ratio 0.1 \
    --test-ratio 0.1
  codeplan-tokenizer \
    --train artifacts/quick/filtered/train.jsonl \
    --out artifacts/quick/tokenizer.json \
    --vocab-size 1024 \
    --min-frequency 1
  codeplan-pack \
    --filtered-dir artifacts/quick/filtered \
    --tokenizer artifacts/quick/tokenizer.json \
    --out-dir artifacts/quick/packed

  CFG=artifacts/quick/config.json
  python3 - "$CFG" <<'PY'
import json
import sys

cfg = json.load(open("configs/quick_test.json"))
cfg["data_dir"] = "artifacts/quick/packed"
cfg["tokenizer_path"] = "artifacts/quick/tokenizer.json"
cfg["out_dir"] = "artifacts/quick/run"
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
PY
fi

echo "== training ($CFG) =="
codeplan-pretrain --config "$CFG"

LOG="$(python3 -c "import json,sys;print(json.load(open('$CFG'))['out_dir'])")/log.jsonl"
echo
echo "Loss curve: $LOG"
if [[ "$CFG" != "configs/quick_test.json" ]]; then
  echo "Note: the seed corpus has ~80 examples, so validation loss flattens while"
  echo "train loss memorises. That is expected; the point is that the loss falls."
fi

cat <<EOF

Next steps:
  bash scripts/pilot.sh                                   # ~12M real tokens
  codeplan-pretrain --config configs/pretrain_pilot.json  # 1-2 h
EOF
