"""Dataset utilities for packed pretraining and SFT."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import DEFAULT_SYSTEM_PROMPT, CodePlanTokenizer


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16) if vocab_size <= 65535 else np.dtype(np.uint32)


def load_meta(data_dir: str | Path) -> dict:
    with open(Path(data_dir) / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


class PackedDataset(Dataset):
    """Sequential windows over one packed token file."""

    def __init__(self, bin_path: str | Path, seq_len: int, dtype: np.dtype | str = np.uint16):
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        self.tokens = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        if len(self.tokens) < seq_len + 1:
            raise ValueError(f"{self.bin_path} is too short for seq_len={seq_len}")
        self.n_windows = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


class MixturePackedDataset(Dataset):
    """Samples windows from domain-specific packed files according to weights."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        seq_len: int,
        weights: dict[str, float],
        seed: int = 1337,
        length: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        meta = load_meta(self.data_dir)
        self.dtype = np.dtype(meta["dtype"])
        self.weights = {k: float(v) for k, v in weights.items() if float(v) > 0}
        if not self.weights:
            raise ValueError("Mixture weights are empty")
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.domains: list[str] = []
        self.probs: list[float] = []
        self.arrays: dict[str, np.memmap] = {}
        for domain, weight in self.weights.items():
            path = self.data_dir / f"{split}_{domain}.bin"
            if not path.exists():
                continue
            arr = np.memmap(path, dtype=self.dtype, mode="r")
            if len(arr) < seq_len + 1:
                continue
            self.arrays[domain] = arr
            self.domains.append(domain)
            self.probs.append(weight)
        if not self.domains:
            raise ValueError(f"No usable packed files for split={split}")
        total = sum(self.probs)
        self.probs = [p / total for p in self.probs]
        self.rng = random.Random(seed)
        self.length = length or max(1, sum(len(a) for a in self.arrays.values()) // seq_len)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        domain = self.rng.choices(self.domains, weights=self.probs, k=1)[0]
        arr = self.arrays[domain]
        max_start = len(arr) - self.seq_len - 1
        start = self.rng.randint(0, max_start)
        chunk = arr[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


class SFTDataset(Dataset):
    """JSONL SFT examples.

    Supported formats:
    1. {"system": ..., "user": ..., "assistant": ...}
    2. {"user": ..., "plan": ..., "code": ..., "tests": ..., "answer": ...}
    3. {"messages": [{"role": "system"|"user"|"assistant", "content": ...}]}
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: CodePlanTokenizer,
        max_len: int = 2048,
        default_system: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.default_system = default_system
        self.examples: list[tuple[list[int], list[int]]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                system, user, assistant = self._normalize(obj)
                ids, labels = tokenizer.format_sft(system, user, assistant)
                if len(ids) > max_len:
                    ids = ids[:max_len]
                    labels = labels[:max_len]
                if len(ids) < 2:
                    continue
                if not any(lab != -100 for lab in labels):
                    continue
                self.examples.append((ids, labels))
        if not self.examples:
            raise ValueError(f"No SFT examples found in {path}")

    def _normalize(self, obj: dict) -> tuple[str, str, str]:
        if "messages" in obj:
            system = self.default_system
            user_parts: list[str] = []
            assistant_parts: list[str] = []
            for msg in obj["messages"]:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    system = content
                elif role == "user":
                    user_parts.append(content)
                elif role == "assistant":
                    assistant_parts.append(content)
            return system, "\n".join(user_parts), "\n".join(assistant_parts)

        system = obj.get("system", self.default_system)
        user = obj.get("user", "")
        if "assistant" in obj:
            assistant = obj["assistant"]
        else:
            parts: list[str] = []
            if obj.get("plan"):
                parts.append(f"<plan>{obj['plan']}</plan>")
            if obj.get("code"):
                parts.append(f"<code>{obj['code']}</code>")
            if obj.get("tests"):
                parts.append(f"<tests>{obj['tests']}</tests>")
            if obj.get("answer"):
                parts.append(f"<answer>{obj['answer']}</answer>")
            assistant = "".join(parts)
        return system, user, assistant

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, labels = self.examples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def sft_collate(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int):
    max_len = max(len(x) for x, _ in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for x, y in batch:
        pad_len = max_len - len(x)
        input_ids.append(torch.cat([x, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(len(x), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
