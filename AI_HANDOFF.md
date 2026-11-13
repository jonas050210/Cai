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
- GitHub collection with a license whitelist, path/size filters and a checksum manifest.
- JSONL ingestion, DE/EN filtering (comments/docstrings for code), dedup, deterministic split.
- Packed binary pretraining datasets with domain mixture sampling.
- Pretraining loop with bf16/fp16, gradient accumulation, cosine LR, checkpointing,
  checkpoint retention and early stopping; best/last checkpoints survive resumes.
- Resume-friendly short sessions with a GPU thermal guard (`scripts/train_session.sh`).
- SFT loop with assistant-only loss masking and a deterministic validation split.
- Perplexity evaluation overall and per domain.
- Coding benchmark with unit-test execution in a limited subprocess (pass@k, language compliance).
- safetensors export plus `load_exported()` round-trip.
- FastAPI inference server with JSON API and a minimal web UI.
- Tiny smoke config and synthetic seed data; 39 unit tests; CI (lint, tests, smoke).

Not implemented yet:
- Large-scale data collection/curation (the collector exists, the corpus does not).
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
codeplan-collect --out data/raw/github.jsonl --max-repos 200 --languages python,typescript
codeplan-prepare --input data/raw/github.jsonl --out-dir data/filtered --require-license
codeplan-fertility --train data/filtered/train.jsonl --vocab-sizes 16000,32000
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
- vocab: 32000 in the config files, but always overridden by the trained tokenizer's
  actual size at runtime. **Run `codeplan-fertility` before deciding**: at ~48M
  parameters a 32k vocabulary spends ~16M parameters (a third of the model) on
  embeddings, so 16k is often the better trade.
- context: 2048
- d_model: 512
- layers: 10
- heads: 8
- intermediate: 1408
- tied embeddings
- gradient checkpointing

Approximate parameter count: ~48.5M with a 32k vocabulary.

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

## Good next tasks

1. Run the collector at scale and audit the manifest (licenses, per-repo record counts).
2. Improve near-duplicate detection (MinHash/SimHash) instead of exact hashing only.
3. Add coding eval tasks with real sandboxing (container) and pass@k reporting.
4. Tokenizer fertility analysis on the real corpus (DE vs EN vs code).
5. Add a 150M config only after 50M training is stable.
6. Quantized inference and a streaming endpoint.
