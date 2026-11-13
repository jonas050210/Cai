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

## Phase 1: Data

- [ ] Collect licensed DE/EN code
- [ ] Collect DE/EN planning/spec/architecture text
- [ ] Improve comment-level language filtering
- [ ] Add dataset manifest and checksums
- [ ] Audit tokenizer fertility

## Phase 2: Training

- [ ] Train 50M baseline
- [ ] Track validation loss and sample quality
- [ ] Add early stopping
- [ ] Add learning-rate finder
- [ ] Try 1024 vs 2048 context

## Phase 3: Evaluation

- [ ] Add small DE/EN coding tasks
- [ ] Execute generated tests in sandbox
- [ ] Add pass@1 / pass@k
- [ ] Add language-compliance eval
- [ ] Add planning-quality rubric

## Phase 4: Product

- [ ] Export safetensors
- [ ] Local FastAPI inference server
- [ ] Simple web UI
- [ ] Quantized inference if useful
