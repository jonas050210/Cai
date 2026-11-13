"""Pack filtered JSONL into binary token files for fast training."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import dtype_for_vocab, iter_jsonl
from .tokenizer import CodePlanTokenizer


def pack(split_path: Path, out_dir: Path, split: str, tokenizer: CodePlanTokenizer) -> dict:
    dtype = dtype_for_vocab(tokenizer.vocab_size)
    counts: dict[str, int] = defaultdict(int)
    files: dict[str, str] = {}
    handles = {}
    all_path = out_dir / f"{split}_all.bin"
    handles["all"] = open(all_path, "wb")
    files["all"] = all_path.name

    try:
        batch: list[dict] = []

        def flush() -> None:
            if not batch:
                return
            texts = [rec["text"] for rec in batch]
            encoded = tokenizer.encode_batch(texts, add_eos=True)
            for rec, ids in zip(batch, encoded):
                domain = rec.get("domain", "general")
                if domain not in handles:
                    path = out_dir / f"{split}_{domain}.bin"
                    handles[domain] = open(path, "wb")
                    files[domain] = path.name
                buf = np.asarray(ids, dtype=dtype).tobytes()
                handles[domain].write(buf)
                handles["all"].write(buf)
                counts[domain] += len(ids)
                counts["all"] += len(ids)
            batch.clear()

        for rec in iter_jsonl(split_path):
            domain = rec.get("domain", "general")
            if domain not in {"code", "planning", "general"}:
                continue
            batch.append(rec)
            if len(batch) >= 256:
                flush()
        flush()
    finally:
        for h in handles.values():
            h.close()

    return {"files": files, "tokens": dict(counts), "dtype": np.dtype(dtype).name}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack filtered JSONL into .bin token files.")
    parser.add_argument("--filtered-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    filtered = Path(args.filtered_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = CodePlanTokenizer.load(args.tokenizer)

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "dtype": np.dtype(dtype_for_vocab(tokenizer.vocab_size)).name,
        "splits": {},
        "target_mixture": {"code": 0.65, "planning": 0.34, "general": 0.01},
        "note": "Training samples domains by target_mixture; 99% code+planning.",
    }
    for split in ["train", "val", "test"]:
        path = filtered / f"{split}.jsonl"
        if not path.exists():
            continue
        meta["splits"][split] = pack(path, out_dir, split, tokenizer)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
