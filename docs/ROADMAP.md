# Roadmap

## Phase 0: Pipeline

- [x] Model architecture
- [x] Tokenizer training
- [x] Data filtering
- [x] Packing
- [x] Pretraining loop
- [x] SFT loop
- [x] Generation/chat
- [x] Smoke test
- [x] Unit tests + CI (lint, tests, smoke)

## Phase 1: Data

- [x] Licensed collection tooling with license whitelist (`codeplan-collect`)
- [x] Dataset manifest with checksums
- [x] Code-aware language filtering (comments/docstrings instead of whole file)
- [x] Tokenizer fertility analysis (`codeplan-fertility`)
- [ ] Collect the actual DE/EN corpus at scale
- [x] Near-duplicate detection (MinHash + LSH, `--near-dedup`)
- [x] Corpus statistics and token-budget check (`codeplan-stats`)
- [x] Pilot corpus with a reproducible script (`scripts/pilot.sh`)
- [ ] Manual license audit of the manifest

## Phase 2: Training

- [x] Checkpoint retention (`keep_last_checkpoints`)
- [x] Early stopping (`early_stop_patience`)
- [x] Resumable short sessions + GPU thermal guard
- [x] Validation loss restored on resume
- [x] Preflight checks with measured throughput/VRAM (`codeplan-doctor`)
- [x] 99M configuration with a memory budget (`docs/100M_PLAN.md`)
- [x] Syntax gate, curriculum scheduling, reproducible `run.json`
- [ ] Train 50M baseline (blocked on data: 18M of ~800M tokens)
- [ ] Train 99M model (blocked on data: 18M of ~983M tokens)
- [ ] Track sample quality over time
- [ ] Add learning-rate finder
- [ ] Try 1024 vs 2048 context

## Phase 3: Evaluation

- [x] Per-domain perplexity (`--per-domain`)
- [x] Coding tasks with unit-test execution (`codeplan-bench`)
- [x] pass@1 / pass@k
- [x] Language-compliance measurement
- [ ] Real sandboxing (containers) for test execution
- [ ] Planning-quality rubric

## Phase 4: Product

- [x] Export to safetensors
- [x] Local FastAPI inference server
- [x] Simple web UI
- [ ] Quantized inference
- [ ] Streaming responses
