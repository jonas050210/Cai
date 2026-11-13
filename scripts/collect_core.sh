#!/usr/bin/env bash
# Collect the curated repository list and prepare it for a serious training run.
#
#   bash scripts/collect_core.sh            # 53 repos, ~45-60 min
#   bash scripts/collect_core.sh --stats    # only show what is already there
#
# The list lives in data/repos/core.txt: every entry was checked against the
# GitHub API for licence (MIT/Apache/BSD/ISC/Unlicense) and size.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

if [[ -f "$HOME/.config/codeplan/env" ]]; then
  # shellcheck disable=SC1091
  . "$HOME/.config/codeplan/env"
fi

if [[ "${1:-}" == "--stats" ]]; then
  codeplan-stats --input data/filtered/train.jsonl --target-tokens 983000000
  exit 0
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "No GITHUB_TOKEN. Run: bash scripts/install_token.sh" >&2
  exit 1
fi

echo "== collect (data/repos/core.txt) =="
codeplan-collect --out data/raw/core.jsonl --repos-file data/repos/core.txt \
  --max-repo-mb 40 --sleep 0.2

echo "== prepare =="
codeplan-prepare --input data/raw/core.jsonl --out-dir data/filtered \
  --require-license --near-dedup --syntax-check

echo "== stats =="
codeplan-stats --input data/filtered/train.jsonl --target-tokens 983000000

cat <<'TXT'

Next, if stats says <= 2 epochs:
  codeplan-tokenizer --train data/filtered/train.jsonl --out artifacts/tokenizer.json --vocab-size 16000
  codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed
  codeplan-doctor --config configs/pretrain_100m_4060ti.json --bench-steps 20
  bash scripts/curriculum.sh --config configs/pretrain_100m_4060ti.json
TXT
