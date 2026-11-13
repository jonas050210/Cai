import pytest

from codeplan.dedup import NearDuplicateFilter, jaccard, sketch

BASE = "\n".join(
    f"def func_{i}(x, y):\n    value = x * {i} + y\n    return value + {i * 3}\n"
    for i in range(160)
)
NEAR = (
    "# Copyright 2026 Example Corp\n# SPDX-License-Identifier: MIT\n"
    + BASE
    + "\ndef extra(z):\n    return z * 42\n"
)
UNRELATED = "\n".join(
    f"class Klass{i}:\n    def method(self, a):\n        return a - {i}\n" for i in range(160)
)


def test_sketch_is_deterministic():
    first = sketch(BASE)
    second = sketch(BASE)
    assert first is not None and second is not None
    assert (first == second).all()
    assert jaccard(first, second) == 1.0


def test_sketch_rejects_bad_parameters():
    with pytest.raises(ValueError):
        sketch(BASE, window=3)
    with pytest.raises(ValueError):
        sketch(BASE, num_perm=2)


def test_jaccard_bounds():
    a = sketch(BASE)
    b = sketch(UNRELATED)
    assert 0.0 <= jaccard(a, b) < 0.2


def test_near_duplicate_is_flagged_once():
    filter_ = NearDuplicateFilter()
    assert not filter_.is_duplicate(BASE)
    assert filter_.is_duplicate(NEAR)
    assert not filter_.is_duplicate(UNRELATED)
    stats = filter_.stats()
    assert stats["duplicates"] == 1
    assert stats["documents"] == 3


def test_short_documents_are_skipped():
    filter_ = NearDuplicateFilter()
    assert not filter_.is_duplicate("x = 1\n" * 5)
    assert filter_.skipped == 1
    assert filter_.stats()["documents"] == 0


def test_filter_validates_configuration():
    with pytest.raises(ValueError):
        NearDuplicateFilter(threshold=0.0)
    with pytest.raises(ValueError):
        NearDuplicateFilter(num_perm=64, bands=7)
