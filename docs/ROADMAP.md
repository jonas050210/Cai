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
- [ ] Near-duplicate detection (MinHash/SimHash)
- [ ] Manual license audit of the manifest

## Phase 2: Training

- [x] Checkpoint retention (`keep_last_checkpoints`)
- [x] Early stopping (`early_stop_patience`)
- [x] Resumable short sessions + GPU thermal guard
- [x] Validation loss restored on resume
- [ ] Train 50M baseline
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
