#!/usr/bin/env bash
# Build the whole training corpus in one go, skipping whatever already exists.
#
#   bash scripts/build_corpus.sh                 # core + wide lists (~2-4 h)
#   bash scripts/build_corpus.sh --core-only     # only the 53 core repos (~45 min)
#   bash scripts/build_corpus.sh --stop-after stats
#   bash scripts/build_corpus.sh --workers 16     # more parallel downloads
#   bash scripts/build_corpus.sh --knowledge      # + concept cards + Stack Exchange Q&A
#
# Every step is idempotent: if the output file is already there, it is skipped.
# Interrupting is safe - rerun the same command and it continues where the
# missing files are.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

REPOS="core,wide"
STOP_AFTER="doctor"
VOCAB=16000
WORKERS=16
BILINGUAL=0
KNOWLEDGE=0
SE_TAGS="python,javascript,typescript,java,go,rust,git,sql,docker,linux"
SE_PER_TAG=40
CONFIG="configs/pretrain_100m_4060ti.json"
DE_PAGES=3000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --core-only) REPOS="core"; shift ;;
    --repos) REPOS="$2"; shift 2 ;;
    --stop-after) STOP_AFTER="$2"; shift 2 ;;
    --vocab) VOCAB="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --bilingual) BILINGUAL=1; VOCAB="${VOCAB_BILINGUAL:-24576}"; shift ;;
    --de-pages) DE_PAGES="$2"; shift 2 ;;
    --knowledge) KNOWLEDGE=1; shift ;;
    --se-per-tag) SE_PER_TAG="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -f "$HOME/.config/codeplan/env" ]]; then
  # shellcheck disable=SC1091
  . "$HOME/.config/codeplan/env"
fi

step_done() { [[ -s "$1" ]]; }
stage() { echo; echo "=== $* ==="; }
reached() { [[ "$STOP_AFTER" == "$1" ]] && { echo; echo "Stopped after $1 (as requested)."; exit 0; } || true; }

if (( BILINGUAL )); then
  CONFIG="configs/pretrain_bilingual_100m.json"
  VOCAB="${VOCAB_BILINGUAL:-24576}"
  echo "Bilingual mode: German Wikipedia + vocab ${VOCAB} + ${CONFIG}"
fi

mkdir -p data/raw data/filtered artifacts

# 1. raw data ------------------------------------------------------------
IFS=',' read -r -a LISTS <<< "$REPOS"
for list in "${LISTS[@]}"; do
  raw="data/raw/${list}.jsonl"
  if step_done "$raw"; then
    echo "skip collect: $raw exists"
    continue
  fi
  if [[ ! -f "data/repos/${list}.txt" ]]; then
    echo "missing repo list data/repos/${list}.txt" >&2
    exit 1
  fi
  if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "No GITHUB_TOKEN. Run: bash scripts/install_token.sh" >&2
    exit 1
  fi
  count=$(grep -vc '^#' "data/repos/${list}.txt" || true)
  stage "collect ${list} (${count} repos)"
  codeplan-collect --out "$raw" --repos-file "data/repos/${list}.txt" \
    --max-repo-mb 50 --workers "${WORKERS}" --sleep 0.2
done
# German text (CC BY-SA 4.0) is not a GitHub repository, so it has its own step.
if (( BILINGUAL )); then
  if step_done "data/raw/german_wiki.jsonl"; then
    echo "skip german: data/raw/german_wiki.jsonl exists"
  else
    stage "german wikipedia (${DE_PAGES} articles)"
    codeplan-german-wiki --out data/raw/german_wiki.jsonl --pages "$DE_PAGES"
  fi
fi
# Concept cards + Stack Exchange Q&A: knowledge, not more syntax.
if (( KNOWLEDGE )); then
  if step_done "data/raw/cards.jsonl"; then
    echo "skip cards: data/raw/cards.jsonl exists"
  else
    stage "concept cards"
    codeplan-cards --out data/raw/cards.jsonl --sft-out data/raw/cards_sft.jsonl
  fi
  if step_done "data/raw/stackexchange.jsonl"; then
    echo "skip stackexchange: exists"
  else
    stage "stack exchange Q&A (${SE_PER_TAG} per tag)"
    codeplan-se --out data/raw/stackexchange.jsonl --per-tag "$SE_PER_TAG" --tags "$SE_TAGS"
  fi
fi
reached collect

# 2. merge ---------------------------------------------------------------
merged="data/raw/all.jsonl"
if step_done "$merged"; then
  echo "skip merge: $merged exists"
else
  stage "merge"
  : > "$merged"
  for list in "${LISTS[@]}"; do
    [[ -f "data/raw/${list}.jsonl" ]] && cat "data/raw/${list}.jsonl" >> "$merged"
  done
  [[ -f data/raw/german_wiki.jsonl ]] && cat data/raw/german_wiki.jsonl >> "$merged"
  [[ -f data/raw/cards.jsonl ]] && cat data/raw/cards.jsonl >> "$merged"
  [[ -f data/raw/stackexchange.jsonl ]] && cat data/raw/stackexchange.jsonl >> "$merged"
  echo "$(wc -l < "$merged") records"
fi
reached merge

# 3. filter --------------------------------------------------------------
filtered="data/filtered/train.jsonl"
if step_done "$filtered"; then
  echo "skip prepare: $filtered exists"
else
  stage "prepare (licence, near-dedup, syntax gate)"
  codeplan-prepare --input "$merged" --out-dir data/filtered \
    --require-license --near-dedup --syntax-check --workers 20
fi
reached prepare

# 4. token budget --------------------------------------------------------
stage "stats"
codeplan-stats --input "$filtered" --target-tokens 983000000 --out artifacts/corpus_stats.json
reached stats

# 5. tokenizer -----------------------------------------------------------
if step_done "artifacts/tokenizer.json"; then
  echo "skip tokenizer: artifacts/tokenizer.json exists"
else
  stage "tokenizer (vocab ${VOCAB})"
  codeplan-tokenizer --train "$filtered" --out artifacts/tokenizer.json --vocab-size "$VOCAB"
fi
reached tokenizer

# 6. pack ----------------------------------------------------------------
if step_done "data/packed/meta.json"; then
  echo "skip pack: data/packed exists"
else
  stage "pack"
  codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed
fi
reached pack

# 7. sanity --------------------------------------------------------------
stage "doctor"
codeplan-doctor --config "$CONFIG" --bench-steps 20

cat <<'TXT'

Corpus ready. Training:

  tmux new -s train
  bash scripts/train_fast.sh --config "$CONFIG" --steps 12000
  # Ctrl+B then D to detach; tmux attach -t train to come back
TXT
