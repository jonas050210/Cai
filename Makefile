.PHONY: setup smoke test lint prepare tokenizer pack pretrain sft chat eval

setup:
	pip install -e ".[dev]"

smoke:
	bash scripts/smoke_test.sh

test:
	pytest -q

lint:
	ruff check src tests

prepare:
	codeplan-prepare --input data/raw/my_data.jsonl --out-dir data/filtered --require-license

tokenizer:
	codeplan-tokenizer --train data/filtered/train.jsonl --out artifacts/tokenizer.json --vocab-size 32000

pack:
	codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed

pretrain:
	codeplan-pretrain --config configs/pretrain_50m_4060ti.json

sft:
	codeplan-sft --config configs/sft_50m_4060ti.json --data data/sft/train.jsonl

chat:
	codeplan-chat --checkpoint checkpoints/sft_50m/last.pt

eval:
	codeplan-eval --checkpoint checkpoints/sft_50m/last.pt --data-dir data/packed --split test --seq-len 2048
