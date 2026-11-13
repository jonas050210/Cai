.PHONY: setup smoke quick test lint ci pilot collect stats sft-data prepare tokenizer fertility pack doctor pretrain session sft eval bench export serve

setup:
	pip install -e ".[dev]"

smoke:
	bash scripts/smoke_test.sh

quick:
	codeplan-pretrain --config configs/quick_test.json

test:
	pytest -q

lint:
	ruff check src tests

ci: lint test

pilot:
	bash scripts/pilot.sh

collect:
	codeplan-collect --out data/raw/github.jsonl --max-repos 200

stats:
	codeplan-stats --input data/raw/github.jsonl --target-tokens 1000000000

prepare:
	codeplan-prepare --input data/raw/my_data.jsonl --out-dir data/filtered --require-license --near-dedup

tokenizer:
	codeplan-tokenizer --train data/filtered/train.jsonl --out artifacts/tokenizer.json --vocab-size 16000

fertility:
	codeplan-fertility --train data/filtered/train.jsonl --vocab-sizes 8000,16000,32000 --out artifacts/fertility.json

pack:
	codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed

doctor:
	codeplan-doctor --config configs/pretrain_50m_4060ti.json --bench-steps 20

pretrain:
	codeplan-pretrain --config configs/pretrain_50m_4060ti.json

session:
	bash scripts/train_session.sh --config configs/pretrain_50m_4060ti.json

sft-data:
	codeplan-sft-data --input data/filtered/train.jsonl --out-dir data/sft

sft:
	codeplan-sft --config configs/sft_50m_4060ti.json --data data/sft/train.jsonl

chat:
	codeplan-chat --checkpoint checkpoints/sft_50m/last.pt

eval:
	codeplan-eval --checkpoint checkpoints/sft_50m/last.pt --data-dir data/packed --split test --seq-len 2048 --per-domain

bench:
	codeplan-bench --checkpoint checkpoints/sft_50m/last.pt --tasks data/bench/code_tasks.jsonl --allow-exec

export:
	codeplan-export --checkpoint checkpoints/sft_50m/last.pt --out artifacts/model.safetensors --dtype bf16

serve:
	codeplan-serve --checkpoint checkpoints/sft_50m/last.pt --host 0.0.0.0 --port 8000
