# Runbook: first test with real data

Two paths. Pick **A** first; it takes an evening and proves the whole chain on
real data. **B** is the real training run and needs a corpus ~50x larger.

## Path 0: five-minute test

Needs no data download and no GPU:

```bash
PATH="$PWD/.venv/bin:$PATH" bash scripts/smoke_test.sh   # ~25 s, synthetic seed data
```

Proves install, tokenizer, packing, train loop, checkpoint save/load, export,
import and the server. If this fails, nothing else will work.

Once the pilot corpus exists (`bash scripts/pilot.sh`), the next step up is a
real run with a small model on real data (~3-5 min on CPU, <1 min on GPU):

```bash
codeplan-pretrain --config configs/quick_test.json
```

7.5M parameters, 150 steps x 1,024 tokens, val loss every 50 steps. Loss must
fall from ~9.6 to below ~7.5; that is the "everything really works" signal.

## Path A: pipeline test (pilot corpus, ~12M tokens)

Cost: ~30 min data work + 1-2 h training on a 4060 Ti (3-4 h on CPU).

```bash
export GITHUB_TOKEN=...        # 60 req/h -> 5000 req/h
bash scripts/pilot.sh          # collect -> filter -> stats -> tokenizer -> pack -> doctor
codeplan-pretrain --config configs/pretrain_pilot.json
codeplan-eval --checkpoint checkpoints/pretrain_pilot/last.pt \
  --data-dir data/packed --split val --seq-len 256 --per-domain
```

What "success" looks like:
- `doctor` ends with `0 errors` (warnings about epochs are fine here).
- Training loss falls from ~9.7 to ~7.0 within 100-200 steps.
- Validation perplexity falls (16.8k -> ~1.2k on the pilot, CPU demo run).
- `checkpoints/pretrain_pilot/last.pt` exists and `--resume` continues the step count.
- Per-domain perplexity differs by domain instead of being identical.

Known limitation: 2.7 epochs means the model partly memorises the pilot corpus.
This run validates plumbing and speed, not quality.

## Path B: real run (~1B tokens)

### 1. Data (the actual bottleneck)

```bash
export GITHUB_TOKEN=...
# curated repos scale best; star-ranked search also returns awesome-lists
codeplan-collect --out data/raw/gh1.jsonl --repos owner/a,owner/b,...
codeplan-collect --out data/raw/gh2.jsonl --max-repos 400 --languages python,typescript,javascript
cat data/raw/gh*.jsonl > data/raw/all.jsonl
codeplan-prepare --input data/raw/all.jsonl --out-dir data/filtered --require-license --near-dedup
codeplan-stats --input data/filtered/train.jsonl --target-tokens 1000000000
```

Stop here until `codeplan-stats` says `enough data (<= 2 epochs)` or at most
`thin: 2-4 epochs`. At the pilot density (3.65 chars/token) 1B tokens need about
3.7 GB of text, i.e. several hundred mid-sized repositories.

Add German and planning content deliberately: GitHub code is ~97% English, and
`planning` was only ~11% of the pilot corpus.

### 2. Tokenizer (decide once, it is hard to change later)

```bash
codeplan-tokenizer --train data/filtered/train.jsonl --out artifacts/tokenizer.json --vocab-size 16000
codeplan-fertility --train data/filtered/train.jsonl --vocab-sizes 8000,16000,32000
codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed
```

### 3. Preflight (never skip)

```bash
codeplan-doctor --config configs/pretrain_50m_4060ti.json --bench-steps 20
```

`--bench-steps` measures throughput and peak VRAM with the exact shapes and
prints the expected wall clock. If VRAM is above ~90% of the card, lower
`seq_len` to 1024 (adjust `model.max_seq_len` too) before starting.

### 4. Train in sessions

```bash
bash scripts/train_session.sh --config configs/pretrain_50m_4060ti.json \
  --session-steps 800 --max-total 20000 --temp-limit 80 --power-limit 120
```

Watch `checkpoints/pretrain_50m/log.jsonl`: `loss` and `val_loss` must both go
down. Early stopping and `best.pt` protect against a wasted run.

### 5. SFT and evaluation

```bash
codeplan-sft --config configs/sft_50m_4060ti.json --data data/sft/train.jsonl
codeplan-eval --checkpoint checkpoints/sft_50m/last.pt --data-dir data/packed \
  --split test --seq-len 2048 --per-domain
codeplan-bench --checkpoint checkpoints/sft_50m/last.pt \
  --tasks data/bench/code_tasks.jsonl --allow-exec
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `doctor`: `packed vocab != tokenizer vocab` | tokenizer retrained after packing | re-run `codeplan-pack` |
| `doctor`: `N epochs` error | corpus too small | collect more, or lower `max_steps` |
| CUDA OOM at step 0 | seq too long / batch too big | `seq_len: 1024`, keep `batch_size: 1` |
| Loss stays ~9.7 | lr too low or data/tokenizer mismatch | check `lr`, re-pack data, watch first 50 steps |
| Loss becomes `nan` | fp16 overflow on old GPUs | `precision: bf16` (or `fp32` to debug) |
| Validation loss rises while train falls | overfitting / too many epochs | early stopping, more data |
| Throughput collapses over time | thermal throttling | lower power limit, better airflow |
| `codeplan-bench` always 0% | model too small / random init | expected below ~100M with this data budget |

## Hardware notes (RTX 4060 Ti 8 GB)

- 40-48M parameters at seq 2048, batch 1, grad-accum 16 fits with headroom
  (~2 GB used); gradient checkpointing is the safety net, not a necessity.
- Power-limit 110-120 W costs ~5-10% throughput and drops temperatures by
  roughly 10 °C; the session script pauses above 80 °C.
- The CPU (i7-12700F) does tokenizer training, packing and `--near-dedup`
  quickly; keep `num_workers: 2` so the GPU is not starved.
