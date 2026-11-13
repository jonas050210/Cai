# AI Handoff: CodePlan-LM

## Mission

Build a private, from-scratch language model for German/English coding and technical planning.

Hard constraints:
1. Do not initialize from foreign/pretrained model weights.
2. Natural-language training data should be German or English only.
3. Target training mixture: ~99% code + planning, ~1% general DE/EN language.
4. Reasoning-like behavior should be trained through structured examples:
   `<plan>...</plan><code>...</code><tests>...</tests><answer>...</answer>`.
5. Target hardware: Ubuntu, RTX 4060 Ti 8 GB, 32 GB RAM.
6. Keep the repository small. Do not commit datasets, checkpoints, or generated artifacts.

## Current state

Implemented:
- Decoder-only Transformer: RMSNorm, RoPE, SwiGLU, tied embeddings, KV cache generation.
- Byte-level BPE tokenizer trained from scratch (with a byte-level minimum-vocabulary guard).
- Tokenizer fertility analysis and vocabulary recommendation (`codeplan-fertility`).
- GitHub collection with a license whitelist, streaming size cap and a checksum manifest.
- JSONL ingestion, DE/EN filtering (comments/docstrings for code), exact dedup,
  MinHash/LSH near-dedup, deterministic split, corpus statistics + token budget check.
- Packed binary pretraining datasets with domain mixture sampling.
- Pretraining loop with bf16/fp16, gradient accumulation, cosine LR, checkpointing,
  checkpoint retention and early stopping; best/last checkpoints survive resumes.
- Resume-friendly short sessions with a GPU thermal guard (`scripts/train_session.sh`).
- SFT loop with assistant-only loss masking and a deterministic validation split.
- Perplexity evaluation overall and per domain.
- Coding benchmark with unit-test execution in a limited subprocess (pass@k, language compliance).
- safetensors export plus `load_exported()` round-trip.
- FastAPI inference server with JSON API and a minimal web UI.
- Preflight doctor: environment, config validation, tokenizer/data consistency,
  epoch check, disk estimate, and a measured throughput/VRAM benchmark.
- Pilot runbook and a reproducible pilot script (`scripts/pilot.sh`, 18 repos).
- SFT data mining from the corpus (docstrings -> implement/explain/test pairs).
- Tiny smoke config and synthetic seed data; 74 unit tests; CI (lint, tests, smoke).

Not implemented yet:
- Large-scale data curation. The pilot corpus is 12M tokens; a 40M model wants
  ~800M. `codeplan-doctor` refuses the full run until the data exists.
- Hard language enforcement at decoding time.
- RL / preference optimization.
- Distributed training.
- Real sandboxing (containers) for benchmark execution.
- Quantized inference export.
- A human-evaluated planning-quality rubric.

## Key commands

```bash
pip install -e ".[dev]"
make lint && make test && make smoke
ruff check src tests
pytest -q
```

Real pipeline:
```bash
export GITHUB_TOKEN=...
bash scripts/pilot.sh                      # reproducible 12M-token pilot + preflight
codeplan-prepare --input data/raw/github.jsonl --out-dir data/filtered \
  --require-license --near-dedup
codeplan-stats --input data/filtered/train.jsonl --target-tokens 1000000000
codeplan-fertility --train data/filtered/train.jsonl --vocab-sizes 8000,16000,32000
codeplan-doctor --config configs/pretrain_50m_4060ti.json --bench-steps 20
codeplan-tokenizer --train data/filtered/train.jsonl --out artifacts/tokenizer.json --vocab-size 16000
codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed
bash scripts/train_session.sh --config configs/pretrain_50m_4060ti.json --session-steps 800
codeplan-sft --config configs/sft_50m_4060ti.json --data data/sft/train.jsonl
codeplan-eval --checkpoint checkpoints/sft_50m/last.pt --data-dir data/packed --split test --per-domain
codeplan-bench --checkpoint checkpoints/sft_50m/last.pt --tasks data/bench/code_tasks.jsonl --allow-exec
codeplan-chat --checkpoint checkpoints/sft_50m/last.pt
codeplan-export --checkpoint checkpoints/sft_50m/last.pt --out artifacts/model.safetensors --dtype bf16
codeplan-serve --checkpoint checkpoints/sft_50m/last.pt --host 0.0.0.0 --port 8000
```

## Model config

50M target config:
- vocab: **16000** (decided with `codeplan-fertility`, see `docs/DECISIONS.md`;
  a 32k vocabulary spends a third of a 48M model on embeddings and costs ~13% more
  compute per character at only 6% better compression).
- The runtime always overrides `model.vocab_size` with the tokenizer's actual size.
- context: 2048
- d_model: 512
- layers: 10
- heads: 8
- intermediate: 1408
- tied embeddings
- gradient checkpointing

Approximate parameter count: ~40.3M with a 16k vocabulary (48.5M with 32k).

## Data format

Pretraining JSONL:
```json
{"id":"...","text":"...","lang":"de|en","lang_source":"declared|detected|code-default","domain":"code|planning|general","license":"MIT","source":"..."}
```

`lang_source=code-default` means the record is code without prose; it is kept as
language-neutral instead of being dropped by the language filter.

SFT JSONL, option A:
```json
{"system":"...","user":"...","assistant":"<plan>...</plan><code>...</code>"}
```

SFT JSONL, option B:
```json
{"user":"...","plan":"...","code":"...","tests":"...","answer":"..."}
```

SFT JSONL, option C:
```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Benchmark task JSONL:
```json
{"id":"add","prompt":"Write add(a, b).","tests":"assert add(2, 3) == 5"}
```

## Safety / quality notes

- The DE/EN filter is heuristic plus optional `langdetect`. Code is judged on its
  comments/docstrings only; that is still not perfect.
- License metadata matters. The collector refuses anything outside its whitelist and
  writes a manifest with tarball checksums. Do not train on data you do not have rights to use.
- Checkpoints are loaded with `weights_only=False`; only load checkpoints you created or trust.
- The benchmark executes model output as your user. Use a container for real evals.
- The model is small and will hallucinate. Do not deploy for high-stakes use.

## Working agreement (user instruction, 2026-11-13)

- After every merge, also run `cd` into the repository and `git pull` so the local
  checkout on `main` is up to date.
- The user keeps the working copy at `~/Desktop/Cai` on native Ubuntu
  (`git clone https://github.com/jonas050210/Cai.git ~/Desktop/Cai`, then `make setup`).

## Good next tasks

1. Collect ~800M tokens (curated repo lists beat star-ranked search) and audit the
   manifest; the current pilot has 12M.
2. Add German and planning/spec content: the pilot is 97% English, 11% planning.
3. Real sandboxing (containers) for `codeplan-bench`, plus more tasks.
4. Try lr 2e-4 vs 3e-4 for the first 1,000 steps and compare val loss.
5. Add a 150M config only after the 40M/50M baseline is stable.
6. Quantized inference and a streaming endpoint.
