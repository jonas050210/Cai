# Training Guide

## 1. Smoke test

```bash
bash scripts/smoke_test.sh
```

This verifies the pipeline on CPU with tiny synthetic data.

## 2. Collect and prepare real data

From GitHub (only whitelisted licenses, tarballs, checksum manifest):

```bash
# Raise the API limit first: export GITHUB_TOKEN=...
codeplan-collect --out data/raw/github.jsonl --max-repos 200 \
  --languages python,typescript,javascript --min-stars 200

# Or curate specific repositories
codeplan-collect --out data/raw/curated.jsonl --repos psf/requests,pallets/flask
```

Star-ranked search also surfaces awesome-lists, tutorial collections and repos
with inflated stars. Check `data/raw/*.manifest.jsonl` (status and record counts per
repo) and prefer `--repos` for a curated corpus.

From local files:

```bash
codeplan-ingest --input /path/to/files --out data/raw/my_data.jsonl --license "MIT" --source "my-repos"
```

Filter and split (keeps the manifest auditable via `data/raw/*.manifest.jsonl`):

```bash
codeplan-prepare --input data/raw/github.jsonl --out-dir data/filtered \
  --require-license --near-dedup
codeplan-stats --input data/filtered/train.jsonl --target-tokens 1000000000
```

`--workers` (default: all cores minus one) parallelises the per-document work;
the output is byte-identical to a single-process run.

`--near-dedup` removes near-duplicates (MinHash 128 permutations, Jaccard 0.8,
8-byte shingles) on top of exact hashing. `codeplan-stats` tells you how many
tokens you really have and how many epochs a target budget would need; if it says
"too little data", collect more instead of training.

Code records are judged on their comments and docstrings. Code without prose is
kept as `lang_source=code-default` instead of being dropped; foreign-language
comments are still rejected.

Inspect:
```bash
cat data/filtered/stats.json
```

You want mostly `code` and `planning`, and only `de` / `en`.

## 3. Choose the vocabulary size, then train the tokenizer

```bash
codeplan-fertility --train data/filtered/train.jsonl \
  --vocab-sizes 16000,32000 --out artifacts/fertility.json
```

The report gives tokens/char and tokens/word per domain and language plus the
embedding cost for each vocabulary. Rule of thumb for a ~48M model: 32k costs
~16M parameters for embeddings, so 16k is often better unless fertility for
German or code degrades by more than a few percent.

```bash
codeplan-tokenizer --train data/filtered/train.jsonl --out artifacts/tokenizer.json --vocab-size 16000
```

Do not train the tokenizer on validation/test data.

## 4. Pack data

```bash
codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed
```

## 5. Preflight

```bash
codeplan-doctor --config configs/pretrain_50m_4060ti.json --bench-steps 20
```

Checks Python/torch/CUDA/bf16, validates the config, verifies tokenizer and packed
vocabularies, counts tokens per domain, estimates epochs, checkpoint disk usage,
and — with `--bench-steps` — measures real throughput, peak VRAM and the expected
wall clock. Exit code 1 means do not start.

## 6. Pretrain

One long run:

```bash
codeplan-pretrain --config configs/pretrain_50m_4060ti.json
```

Short sessions (recommended on a desktop GPU):

```bash
bash scripts/train_session.sh --config configs/pretrain_50m_4060ti.json \
  --session-steps 800 --max-total 20000 --temp-limit 80 --power-limit 120
```

Each session resumes from `last.pt`, so you can stop at any time. The script
checks the GPU temperature before every session and waits or aborts when the GPU
is too hot. `--power-limit` needs passwordless sudo and is skipped otherwise.

Useful overrides:
```bash
codeplan-pretrain --config configs/pretrain_50m_4060ti.json --max-steps 1000
codeplan-pretrain --config configs/pretrain_50m_4060ti.json --resume checkpoints/pretrain_50m/last.pt
```

Checkpoint hygiene: `keep_last_checkpoints: 2` keeps only the newest step files
(each is ~580 MB for the 50M model with optimizer state). `early_stop_patience`
stops training after N validations without improvement, and `best.pt`/`last.pt`
remember the best validation loss across resumes.

If you get CUDA OOM:
- keep `batch_size: 1`
- keep `gradient_checkpointing: true`
- reduce `seq_len` to 1024
- reduce `eval_iters`
- close other GPU apps

## 7. SFT data

A base model continues code; it does not answer. SFT pairs are mined from the
corpus itself, so no model generates the targets:

```bash
codeplan-sft-data --input data/filtered/train.jsonl --out-dir data/sft
```

Three kinds per documented function: `implement` (docstring -> code),
`explain` (code -> docstring) and `test` (tests -> implementation, rare). Python
docstrings, JSDoc and `//`/`///` comments in Go and Rust are understood; files are
classified by extension, so `.md`/`.json` are skipped. On the pilot corpus this
yields ~29k pairs in a few seconds. Report: `data/sft/report.json`.

## 8. SFT

Create `data/sft/train.jsonl` with structured examples.

```bash
codeplan-sft --config configs/sft_50m_4060ti.json --data data/sft/train.jsonl \
  --init checkpoints/pretrain_50m/last.pt
```

## 9. Evaluate

```bash
codeplan-eval --checkpoint checkpoints/sft_50m/last.pt --data-dir data/packed \
  --split test --seq-len 2048 --per-domain
```

`--per-domain` reports loss/perplexity per domain file, which is what you want
when training samples a 65/34/1 mixture but the val file is naturally distributed.

Perplexity is only a proxy. `codeplan-bench` generates code, extracts the
`<code>` block and runs unit tests in a limited subprocess:

```bash
codeplan-bench --checkpoint checkpoints/sft_50m/last.pt \
  --tasks data/bench/code_tasks.jsonl --k 1 --allow-exec
```

It reports pass@k and language compliance. It is **not** a real sandbox: the
code runs as your user with CPU/memory/time limits only. Use a container for
untrusted models.

## 10. Export and serve

```bash
codeplan-export --checkpoint checkpoints/sft_50m/last.pt \
  --out artifacts/model.safetensors --dtype bf16
codeplan-serve --checkpoint checkpoints/sft_50m/last.pt --host 0.0.0.0 --port 8000
```

`POST /generate {"prompt": "..."}` returns `{"text": ...}`; `GET /health` reports
model size and context. Tied embeddings are exported once (`aliases` in the config
file lists the dropped duplicate); `codeplan.export.load_exported()` rebuilds the
model from weights + config.

## 11. Chat

```bash
codeplan-chat --checkpoint checkpoints/sft_50m/last.pt
```

## Expected scale

A 50M model usually needs hundreds of millions to billions of clean tokens before it becomes interesting. The default 20k steps at effective batch ~32k tokens is roughly 650M tokens. On a single 4060 Ti this can take many hours to days depending on throughput.
