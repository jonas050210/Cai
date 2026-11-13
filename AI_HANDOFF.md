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
- Byte-level BPE tokenizer trained from scratch.
- JSONL ingestion, DE/EN filtering, deduplication, deterministic split.
- Packed binary pretraining datasets with domain mixture sampling.
- Pretraining loop with bf16/fp16, gradient accumulation, cosine LR, checkpointing.
- SFT loop with assistant-only loss masking.
- Perplexity evaluation, one-shot generation, simple chat CLI.
- Tiny smoke config and synthetic seed data.
- Unit tests and lint config.

Not implemented yet:
- Large-scale data collection/curation.
- Strong coding benchmark evaluation, e.g. HumanEval-like pass@k.
- Hard language enforcement at decoding time.
- RL / preference optimization.
- Distributed training.
- Quantized inference export.

## Key commands

```bash
pip install -e ".[dev]"
bash scripts/smoke_test.sh
ruff check src tests
pytest -q
```

Real training:
```bash
codeplan-ingest --input DATA_DIR --out data/raw/my_data.jsonl --license LICENSE --source SOURCE
codeplan-prepare --input data/raw/my_data.jsonl --out-dir data/filtered --require-license
codeplan-tokenizer --train data/filtered/train.jsonl --out artifacts/tokenizer.json --vocab-size 32000
codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed
codeplan-pretrain --config configs/pretrain_50m_4060ti.json
codeplan-sft --config configs/sft_50m_4060ti.json --data data/sft/train.jsonl
codeplan-eval --checkpoint checkpoints/sft_50m/last.pt --data-dir data/packed --split test
codeplan-chat --checkpoint checkpoints/sft_50m/last.pt
```

## Model config

50M target config:
- vocab: 32000
- context: 2048
- d_model: 512
- layers: 10
- heads: 8
- intermediate: 1408
- tied embeddings
- gradient checkpointing

Approximate parameter count: ~48-50M.

## Data format

Pretraining JSONL:
```json
{"id":"...","text":"...","lang":"de|en","domain":"code|planning|general","license":"MIT","source":"..."}
```

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

## Safety / quality notes

- The DE/EN filter is heuristic plus optional `langdetect`. It is not perfect.
- Code comments can contain other languages; improve filtering before serious training.
- License metadata matters. Do not train on data you do not have rights to use.
- Checkpoints are loaded with `weights_only=False`; only load checkpoints you created or trust.
- The model is small and will hallucinate. Do not deploy for high-stakes use.

## Good next tasks

1. Add a real DE/EN code/planning dataset manifest and license audit.
2. Add better language detection for comments/docs inside code.
3. Add coding eval tasks with unit-test execution in a sandbox.
4. Add validation loss early stopping.
5. Add tokenizer fertility analysis for German vs English vs code.
6. Add a 150M config only after 50M training is stable.
7. Add export to safetensors and a small inference server.
