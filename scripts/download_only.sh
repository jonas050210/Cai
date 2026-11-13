#!/usr/bin/env bash
# Download the corpus only - no GPU, no torch, no training venv.
#
#   bash scripts/download_only.sh            # core + wide, ~30-40 min
#   bash scripts/download_only.sh --core     # only the 53 core repos
#
# Uses a tiny venv (.venv-dl) with a single dependency (langdetect), so it runs
# on any laptop. The result is data/raw/*.jsonl - plain text you can copy to the
# training machine on a USB stick and continue with:
#
#   bash scripts/build_corpus.sh     # skips collect, does prepare -> pack -> doctor
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LISTS="core,wide"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --core) LISTS="core"; shift ;;
    --lists) LISTS="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -f "$HOME/.config/codeplan/env" ]]; then
  # shellcheck disable=SC1091
  . "$HOME/.config/codeplan/env"
fi
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "No GITHUB_TOKEN." >&2
  echo "  bash scripts/install_token.sh        # stores it once" >&2
  echo "  export GITHUB_TOKEN=ghp_...          # or for this shell only" >&2
  exit 1
fi

PY="$ROOT/.venv-dl/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "== creating the download venv (.venv-dl, ~1 min) =="
  python3 -m venv "$ROOT/.venv-dl"
  "$ROOT/.venv-dl/bin/pip" install -q -U pip
  "$ROOT/.venv-dl/bin/pip" install -q langdetect
fi

mkdir -p data/raw
IFS=',' read -r -a NAMES <<< "$LISTS"
for name in "${NAMES[@]}"; do
  out="data/raw/${name}.jsonl"
  echo "== $name =="
  PYTHONPATH=src "$PY" -m codeplan.collect --out "$out" \
    --repos-file "data/repos/${name}.txt" --max-repo-mb 40 --workers 12 --sleep 0.2
done

echo
echo "Downloaded:"
du -sh data/raw/*.jsonl 2>/dev/null || true
cat <<'TXT'

Copy data/raw/ to the training machine (USB stick), put it into the same folder
there, then run:

  bash scripts/build_corpus.sh
TXT
