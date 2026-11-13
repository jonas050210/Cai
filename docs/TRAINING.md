# Training Guide

## 1. Smoke test

```bash
bash scripts/smoke_test.sh
```

This verifies the pipeline on CPU with tiny synthetic data.

## 2. Prepare real data

```bash
codeplan-ingest --input /path/to/files --out data/raw/my_data.jsonl --license "MIT" --source "my-repos"
codeplan-prepare --input data/raw/my_data.jsonl --out-dir data/filtered --require-license
```

Inspect:
```bash
cat data/filtered/stats.json
```

You want mostly `code` and `planning`, and only `de` / `en`.

## 3. Train tokenizer

```bash
codeplan-tokenizer --train data/filtered/train.jsonl --out artifacts/tokenizer.json --vocab-size 32000
```

Do not train the tokenizer on validation/test data.

## 4. Pack data

```bash
codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed
```

## 5. Pretrain

```bash
codeplan-pretrain --config configs/pretrain_50m_4060ti.json
```

Useful overrides:
```bash
codeplan-pretrain --config configs/pretrain_50m_4060ti.json --max-steps 1000
codeplan-pretrain --config configs/pretrain_50m_4060ti.json --resume checkpoints/pretrain_50m/last.pt
```

If you get CUDA OOM:
- keep `batch_size: 1`
- keep `gradient_checkpointing: true`
- reduce `seq_len` to 1024
- reduce `eval_iters`
- close other GPU apps

## 6. SFT

Create `data/sft/train.jsonl` with structured examples.

```bash
codeplan-sft --config configs/sft_50m_4060ti.json --data data/sft/train.jsonl
```

## 7. Evaluate

```bash
codeplan-eval --checkpoint checkpoints/sft_50m/last.pt --data-dir data/packed --split test --seq-len 2048
```

Perplexity is only a proxy. Also test generated code manually or with sandboxed unit tests.

## 8. Chat

```bash
codeplan-chat --checkpoint checkpoints/sft_50m/last.pt
```

## Expected scale

A 50M model usually needs hundreds of millions to billions of clean tokens before it becomes interesting. The default 20k steps at effective batch ~32k tokens is roughly 650M tokens. On a single 4060 Ti this can take many hours to days depending on throughput.
