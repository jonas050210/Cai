.PHONY: setup smoke test lint ci collect prepare tokenizer fertility pack pretrain session sft eval bench export serve

setup:
	pip install -e ".[dev]"

smoke:
	bash scripts/smoke_test.sh

test:
	pytest -q

lint:
	ruff check src tests

ci: lint test

collect:
	codeplan-collect --out data/raw/github.jsonl --max-repos 200

prepare:
	codeplan-prepare --input data/raw/my_data.jsonl --out-dir data/filtered --require-license

tokenizer:
	codeplan-tokenizer --train data/filtered/train.jsonl --out artifacts/tokenizer.json --vocab-size 32000

fertility:
	codeplan-fertility --train data/filtered/train.jsonl --vocab-sizes 16000,32000 --out artifacts/fertility.json

pack:
	codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed

pretrain:
	codeplan-pretrain --config configs/pretrain_50m_4060ti.json

session:
	bash scripts/train_session.sh --config configs/pretrain_50m_4060ti.json

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
