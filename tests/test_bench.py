import json

import numpy as np
import pytest
import torch

from codeplan.bench import extract_code, pass_at_k, run_tests
from codeplan.data import SFTDataset, split_sft_dataset
from codeplan.tokenizer import train_tokenizer


def test_extract_code_prefers_code_tags():
    answer = "<plan>Do it</plan>\n<code>def add(a, b):\n    return a + b</code>"
    assert extract_code(answer) == "def add(a, b):\n    return a + b"


def test_extract_code_fenced_block():
    answer = "Sure:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_code(answer).strip().startswith("def add")


def test_extract_code_fallback_plain_text():
    assert extract_code("def add(a, b): return a + b") == "def add(a, b): return a + b"


def test_pass_at_k_bounds():
    assert pass_at_k(1, 0, 1) == 0.0
    assert pass_at_k(1, 1, 1) == 1.0
    assert pass_at_k(4, 4, 1) == pytest.approx(1.0)
    assert 0.0 < pass_at_k(4, 2, 2) < 1.0


def test_run_tests_pass_and_fail():
    ok, _ = run_tests("def add(a, b):\n    return a + b", "assert add(2, 3) == 5", timeout=10)
    assert ok
    ok, output = run_tests("def add(a, b):\n    return a - b", "assert add(2, 3) == 5", timeout=10)
    assert not ok
    assert "assert" in output.lower() or "AssertionError" in output


def test_run_tests_timeout_and_empty():
    ok, output = run_tests("", "assert True", timeout=5)
    assert not ok and output == "empty code"
    ok, output = run_tests("while True:\n    pass", "assert True", timeout=1)
    assert not ok
    assert any(word in output.lower() for word in ("timeout", "signal", "killed", "error"))


def test_split_sft_dataset_is_deterministic(tmp_path):
    texts = ["code plan test hallo welt"] * 10
    tok = train_tokenizer(texts, vocab_size=512, min_frequency=1, out_path=tmp_path / "t.json")
    data = tmp_path / "sft.jsonl"
    with open(data, "w", encoding="utf-8") as handle:
        for i in range(10):
            handle.write(
                json.dumps({"user": f"task {i}", "assistant": f"<code>x = {i}</code>"}) + "\n"
            )
    ds = SFTDataset(data, tok, max_len=512)

    train_a, val_a = split_sft_dataset(ds, 0.2, seed=1)
    train_b, val_b = split_sft_dataset(ds, 0.2, seed=1)
    assert len(val_a) == 2
    assert len(train_a) == 8
    assert list(val_a.indices) == list(val_b.indices)

    train_c, _ = split_sft_dataset(ds, 0.2, seed=2)
    assert list(train_c.indices) != list(train_a.indices)

    full, no_val = split_sft_dataset(ds, 0.0, seed=1)
    assert no_val is None and len(full) == 10


def test_packed_dataset_dtypes(tmp_path):
    from codeplan.data import PackedDataset

    path = tmp_path / "tokens.bin"
    np.arange(64, dtype=np.uint16).tofile(path)
    ds = PackedDataset(path, seq_len=8, dtype="uint16")
    x, y = ds[1]
    assert torch.equal(y[:-1], x[1:])
