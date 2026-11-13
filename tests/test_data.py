import json

import numpy as np

from codeplan.data import PackedDataset, SFTDataset, sft_collate
from codeplan.tokenizer import train_tokenizer


def test_packed_dataset(tmp_path):
    path = tmp_path / "tokens.bin"
    arr = np.arange(100, dtype=np.uint16)
    arr.tofile(path)
    ds = PackedDataset(path, seq_len=16, dtype=np.uint16)
    x, y = ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    assert y[0].item() == x[0].item() + 1


def test_sft_collate(tmp_path):
    texts = ["code plan test hallo welt"] * 10
    tok = train_tokenizer(texts, vocab_size=512, min_frequency=1, out_path=tmp_path / "tok.json")
    data = tmp_path / "sft.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        f.write(json.dumps({"system": "DE/EN coding.", "user": "Write add", "assistant": "<code>def add(a,b): return a+b</code>"}) + "\n")
        f.write(json.dumps({"system": "DE/EN coding.", "user": "Plan", "assistant": "<plan>Small steps.</plan>"}) + "\n")
    ds = SFTDataset(data, tok, max_len=128)
    batch = [ds[0], ds[1]]
    ids, labels, mask = sft_collate(batch, tok.pad_id)
    assert ids.shape == labels.shape == mask.shape
    assert (labels[ids == tok.pad_id] == -100).all()
