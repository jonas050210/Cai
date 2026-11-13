"""Per-token-class loss.

A single perplexity number hides what a model actually struggles with. This
module splits the validation loss by the *kind* of token being predicted -
identifiers, numbers, keywords, strings, operators - so a run can tell you
"it is bad at identifiers" instead of "it is bad".

The classifier works on raw BPE token strings, which use ``Ġ`` for a leading
space and ``Ċ`` for a newline (byte-level BPE).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PackedDataset, load_meta
from .trainer import autocast_context

CLASSES = ("identifier", "keyword", "number", "string", "operator", "whitespace", "other")

KEYWORDS = frozenset(
    """
    def class return if else elif for while import from as with try except finally raise
    pass break continue lambda yield global nonlocal assert del is not and or in None True
    False async await match case self cls print
    func var const let type struct interface map chan go defer select package range switch
    case default fallthrough import fn impl trait pub use mod crate super unsafe where mut
    ref static enum union extern loop dyn box move match
    function typeof instanceof new this throw catch finally extends super null void delete
    export default require module constructor getter setter public private protected readonly
    abstract implements namespace declare await yield of
    """.split()
)

_TOKEN_SPACE = "Ġ"
_TOKEN_NEWLINE = "Ċ"

_NUMBER = re.compile(r"[-+]?(0[xX][0-9a-fA-F_]+|0[bB][01_]+|\d[\d_]*(\.[\d_]*)?([eE][-+]?\d+)?)")
_IDENT = re.compile(r"[A-Za-z_$@][A-Za-z0-9_$]*")
_OPERATOR = re.compile(r"[^\w\s]+")


def classify_token(text: str) -> str:
    """Classify one BPE token string."""
    if not text:
        return "other"
    visible = text.replace(_TOKEN_SPACE, " ").replace(_TOKEN_NEWLINE, "\n")
    core = visible.strip()
    if not core:
        return "whitespace"
    if core in KEYWORDS:
        return "keyword"
    if _NUMBER.fullmatch(core):
        return "number"
    if core[0] in "'\"`" or core in ("'''", '"""'):
        return "string"
    if _IDENT.fullmatch(core):
        return "identifier"
    if _OPERATOR.fullmatch(core):
        return "operator"
    return "other"


def vocab_classes(vocab: dict[str, int]) -> dict[int, str]:
    """Map token id -> class for a tokenizer vocabulary."""
    return {idx: classify_token(token) for token, idx in vocab.items()}


@torch.no_grad()
def evaluate_classes(
    model,
    path: Path,
    seq_len: int,
    batch_size: int,
    max_batches: int,
    device: torch.device,
    dtype: torch.dtype | None,
    classes: dict[int, str] | None = None,
) -> dict:
    """Mean next-token cross-entropy split by the class of the predicted token."""
    path = Path(path)
    ds = PackedDataset(path, seq_len, dtype=load_meta(path.parent)["dtype"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    model.eval()
    batches = 0
    for i, (x, _y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        with autocast_context(device, dtype):
            logits, _loss, _ = model(x)
        logits = logits[:, :-1].float()
        targets = x[:, 1:]
        flat = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        )
        for token_id, value in zip(targets.reshape(-1).tolist(), flat.tolist()):
            name = classes.get(token_id, "other") if classes else "other"
            sums[name] += value
            counts[name] += 1
        batches += 1
    model.train()

    total = sum(counts.values())
    table = {
        name: {
            "tokens": counts[name],
            "share": counts[name] / total if total else 0.0,
            "loss": sums[name] / counts[name] if counts[name] else 0.0,
        }
        for name in CLASSES
        if counts[name]
    }
    for row in table.values():
        row["perplexity"] = float(torch.exp(torch.tensor(min(row["loss"], 20.0))))
    return {
        "file": path.name,
        "batches": batches,
        "tokens": total,
        "mean_loss": sum(sums.values()) / total if total else 0.0,
        "classes": table,
    }


def format_table(result: dict) -> str:
    lines = [
        f"{'class':<12}{'tokens':>12}{'share':>8}{'loss':>9}{'ppl':>10}",
        "-" * 51,
    ]
    for name, row in sorted(result["classes"].items(), key=lambda kv: -kv[1]["tokens"]):
        lines.append(
            f"{name:<12}{row['tokens']:>12,}{row['share'] * 100:>7.1f}%"
            f"{row['loss']:>9.3f}{row['perplexity']:>10.2f}"
        )
    lines.append("-" * 51)
    lines.append(f"{'all':<12}{result['tokens']:>12,}{100.0:>7.1f}%{result['mean_loss']:>9.3f}")
    return "\n".join(lines)
