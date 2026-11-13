#!/usr/bin/env bash
# Reproduce the pilot corpus and the pipeline test end to end.
#
#   export GITHUB_TOKEN=...          # raises the API limit from 60/h to 5000/h
#   bash scripts/pilot.sh
#
# Result: ~12M tokens of permissively licensed code, a 16k tokenizer, packed
# binaries and a doctor report. This is a *pipeline* test, not a quality run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPOS="psf/requests,pallets/flask,pallets/click,encode/httpx,expressjs/express,axios/axios,\
chalk/chalk,BurntSushi/ripgrep,gin-gonic/gin,python-poetry/poetry,pytest-dev/pytest,\
fastapi/fastapi,pydantic/pydantic,Textualize/rich,vitejs/vite,tailwindlabs/tailwindcss,\
prettier/prettier,gofiber/fiber"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "GITHUB_TOKEN is not set; the collector will be slow (60 requests/hour)."
fi

echo "== 1/6 collect =="
codeplan-collect --out data/raw/pilot.jsonl --repos "$REPOS" --sleep 0.2

echo "== 2/6 filter + near-dedup =="
codeplan-prepare \
  --input data/raw/pilot.jsonl \
  --out-dir data/filtered \
  --require-license \
  --near-dedup

echo "== 3/6 statistics =="
codeplan-stats \
  --input data/filtered/train.jsonl \
  --target-tokens 1000000000 \
  --out artifacts/stats_pilot.json

echo "== 4/6 tokenizer + fertility =="
codeplan-tokenizer \
  --train data/filtered/train.jsonl \
  --out artifacts/tokenizer.json \
  --vocab-size 16000
codeplan-fertility \
  --train data/filtered/train.jsonl \
  --vocab-sizes 8000,16000,32000 \
  --sample-records 3000 \
  --eval-records 400 \
  --out artifacts/fertility_pilot.json

echo "== 5/6 pack =="
codeplan-pack \
  --filtered-dir data/filtered \
  --tokenizer artifacts/tokenizer.json \
  --out-dir data/packed

echo "== 6/6 preflight =="
codeplan-doctor --config configs/pretrain_pilot.json

cat <<'EOF'

Next: run the pipeline test (about 1-2 h on a 4060 Ti, 3-4 h on CPU):

  codeplan-pretrain --config configs/pretrain_pilot.json

Then inspect checkpoints/pretrain_pilot/log.jsonl and run

  codeplan-eval --checkpoint checkpoints/pretrain_pilot/last.pt \
    --data-dir data/packed --split val --seq-len 256 --per-domain
EOF
