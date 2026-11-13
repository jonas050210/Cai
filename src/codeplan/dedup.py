"""Near-duplicate detection with MinHash sketches and LSH bands.

No external dependency: shingles are byte windows, hashes are a splitmix64 mix
vectorized with NumPy, and each document keeps the ``num_perm`` smallest hashes
(bottom-k sketch). Candidate pairs come from LSH bands and are verified by
estimating Jaccard similarity from the sketches.

The exact-duplicate filter in :mod:`codeplan.prepare` only catches identical
text; boilerplate, vendored copies and lightly edited files need this pass.
Memory is roughly ``num_perm * 8`` bytes per document plus Python overhead
(~0.5 GB for one million documents at the default settings).
"""

from __future__ import annotations

import numpy as np

MIX_A = np.uint64(0xBF58476D1CE4E5B9)
MIX_B = np.uint64(0x94D049BB133111EB)
SEED = np.uint64(0x9E3779B97F4A7C15)


def _mix(values: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer, vectorized."""
    z = values + SEED
    z ^= z >> np.uint64(30)
    z *= MIX_A
    z ^= z >> np.uint64(27)
    z *= MIX_B
    z ^= z >> np.uint64(31)
    return z


def sketch(
    text: str,
    window: int = 8,
    num_perm: int = 64,
    max_chars: int = 50_000,
) -> np.ndarray | None:
    """Bottom-k MinHash sketch of ``text`` (sorted uint64 array) or ``None``."""
    if num_perm < 4:
        raise ValueError("num_perm must be >= 4")
    if not 4 <= window <= 8:
        raise ValueError("window must be between 4 and 8 bytes")
    data = text[:max_chars].encode("utf-8", errors="ignore")
    if len(data) < window + 8:
        return None
    buffer = np.frombuffer(data, dtype=np.uint8)
    windows = np.lib.stride_tricks.sliding_window_view(buffer, 8)
    grams = np.ascontiguousarray(windows).view(np.uint64).ravel()
    if window < 8:
        grams &= np.uint64((1 << (8 * window)) - 1)
    hashed = np.unique(_mix(grams))
    if hashed.size > num_perm:
        hashed = np.partition(hashed, num_perm - 1)[:num_perm]
    return np.sort(hashed)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Estimate Jaccard similarity from two bottom-k sketches."""
    if a.size == 0 or b.size == 0:
        return 0.0
    intersection = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - intersection
    return intersection / union if union else 0.0


class NearDuplicateFilter:
    """Streaming LSH deduplicator: ``is_duplicate(text)`` keeps the first copy."""

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        bands: int = 32,
        window: int = 8,
        max_chars: int = 50_000,
        min_chars: int = 500,
    ):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if bands < 1 or num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.threshold = threshold
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.window = window
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.sketches: list[np.ndarray] = []
        self.index: dict[tuple[int, int, int, int], int] = {}
        self.seen = 0
        self.duplicates = 0
        self.skipped = 0

    def sketch(self, text: str) -> np.ndarray | None:
        """Compute the sketch for ``text`` (the expensive, parallelisable part)."""
        # Very short files have too few distinct shingles for a stable estimate.
        if len(text) < self.min_chars:
            self.skipped += 1
            return None
        return sketch(text, window=self.window, num_perm=self.num_perm, max_chars=self.max_chars)

    def is_duplicate(self, text: str) -> bool:
        sig = self.sketch(text)
        return False if sig is None else self.is_duplicate_sketch(sig)

    def is_duplicate_sketch(self, sig: np.ndarray | None) -> bool:
        """Sequential part: LSH lookup plus Jaccard verification."""
        if sig is None:
            return False
        self.seen += 1

        keys = []
        for band in range(self.bands):
            chunk = sig[band * self.rows : (band + 1) * self.rows]
            keys.append((band, *chunk.tolist()))

        for key in keys:
            candidate = self.index.get(key)
            if candidate is None:
                continue
            if jaccard(sig, self.sketches[candidate]) >= self.threshold:
                self.duplicates += 1
                return True

        doc_id = len(self.sketches)
        self.sketches.append(sig)
        for key in keys:
            self.index.setdefault(key, doc_id)
        return False

    def stats(self) -> dict:
        return {
            "documents": self.seen,
            "duplicates": self.duplicates,
            "skipped_short": self.skipped,
            "threshold": self.threshold,
            "num_perm": self.num_perm,
            "bands": self.bands,
        }
