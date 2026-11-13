# 100M plan (RTX 4060 Ti, 8 GB)

Everything below is measured or computed for `configs/pretrain_100m_4060ti.json`.

## The model

| Field | Value |
|---|---|
| d_model | 768 |
| layers | 12 |
| heads | 12 |
| intermediate | 2112 |
| vocabulary | 16000 (tied embeddings) |
| context | 2048 |
| parameters | **99.0M** (12.3M embeddings, 12%) |
| tokens per step | 32,768 (batch 1 × accum 16 × 2048) |

Spot check: `codeplan-doctor --config configs/pretrain_100m_4060ti.json`.

## Memory budget (8 GB card)

| Item | Size |
|---|---|
| bf16 weights | 0.20 GB |
| fp32 master weights | 0.40 GB |
| AdamW moments (fp32) | 0.79 GB |
| gradients (fp32) | 0.40 GB |
| activations (checkpointing, seq 2048) | ~0.2 GB |
| logits (2048 × 16000, fp32) | 0.13 GB |
| **total** | **~2.1 GB** |

Roughly 6 GB of headroom. `batch_size: 2` would fit; start with 1 and raise it
after `codeplan-doctor --bench-steps 20` shows the real peak.

## Tokens and time

Chinchilla-optimal for 99M parameters is ~2B tokens. The config trains 983M
tokens (10 tokens/parameter), which is the honest compromise for one GPU:

| Throughput | 983M tokens |
|---|---|
| 20,000 tok/s | 13.6 h |
| 35,000 tok/s | 7.8 h |
| 50,000 tok/s | 5.5 h |

`codeplan-doctor --bench-steps 20` prints the number for your card; the table is
just to plan the evening.

## Data: the actual blocker

At the measured 3.65 characters per token, 983M tokens need **~3.6 GB of text**
after filtering. The pilot corpus has 18.4M tokens, i.e. 0.37 epochs — `doctor`
refuses the run on purpose.

Minimum sensible order of business:

1. collect until `codeplan-stats` says `≤ 2 epochs` (≈ 500M+ tokens, ~2 GB text),
2. only then start the 100M run,
3. keep the 40M model as the fast baseline for experiments.

## Commands

```bash
export GITHUB_TOKEN=...

# 1. data
codeplan-collect --out data/raw/more.jsonl --max-repos 400 --min-stars 2000 \
  --languages python,typescript,javascript,rust,go
codeplan-prepare --input data/raw/more.jsonl --out-dir data/filtered \
  --require-license --near-dedup --syntax-check
codeplan-stats --input data/filtered/train.jsonl --target-tokens 983000000

# 2. tokenizer on the final mix (decide once: changing it invalidates checkpoints)
codeplan-tokenizer --train data/filtered/train.jsonl --out artifacts/tokenizer.json --vocab-size 16000
codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed

# 3. preflight, then train
codeplan-doctor --config configs/pretrain_100m_4060ti.json --bench-steps 20
bash scripts/curriculum.sh --config configs/pretrain_100m_4060ti.json

# 4. instructions, then chat
codeplan-sft-data --input data/filtered/train.jsonl --out-dir data/sft
codeplan-sft --config configs/sft_100m_4060ti.json --data data/sft/train.jsonl
codeplan-serve --checkpoint checkpoints/sft_100m/last.pt --port 8000
```

## Why the curriculum

`scripts/curriculum.sh` runs the first 20% of the steps at seq 512. The same
number of steps then covers four times as many distinct documents — exactly where
the early loss drop comes from — before switching to the full 2048 context. The
architecture is unchanged, so the resume is exact.

## Checkpoints cost disk

99M parameters ≈ 1.0 GB per checkpoint (`weights + optimizer + scheduler`).
`keep_last_checkpoints: 1` plus `best.pt` and `last.pt` = ~3 GB. Plan for it.

## Honest expectations

A 99M model trained on ~1B tokens of mixed GitHub code learns syntax, common
patterns, docstring-style answers and short completions. It does not become a
useful coding assistant; that starts around 1B parameters and 50B tokens. The
value here is the pipeline: everything is measured, reproducible and swappable.
