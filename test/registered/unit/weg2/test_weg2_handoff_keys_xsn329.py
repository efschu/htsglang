"""xsn328/329: the dormant hold reads with P's handed-over page keys."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2.handoff_keys import first_mismatch, keys_for_span  # noqa: E402


def test_span_slices_the_handed_keys_by_page():
    keys = [f"k{i}" for i in range(10)]
    assert keys_for_span(keys, 0, 10) == keys
    assert keys_for_span(keys, 4, 3) == ["k4", "k5", "k6"]
    assert keys_for_span(keys, 4, 7) is None          # beyond the hand-off
    assert keys_for_span(keys, 0, 0) is None
    assert keys_for_span(None, 0, 3) is None
    assert keys_for_span(keys, 3, 4, page_size=2) is None   # unaligned
    assert keys_for_span(keys, 4, 4, page_size=2) == ["k2", "k3"]


def test_first_mismatch():
    assert first_mismatch(["a", "b"], ["a", "b"]) is None
    assert first_mismatch(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert first_mismatch(["a"], ["a", "b"]) == 1


def test_hit_query_and_hold_are_wired():
    from sglang.srt.managers import cache_controller as cc, scheduler as sc
    assert hasattr(cc, "WEG2_HANDOFF_PAGE_KEYS")
    src = open(cc.__file__).read()
    i = src.index("def _storage_hit_query")
    assert "WEG2_HANDOFF_PAGE_KEYS.get(operation.request_id)" in src[i:i + 3000]
    s2 = open(sc.__file__).read()
    assert "keys_for_span(_hd, int(_matched_len)" in s2
    j = s2.index("def _weg2_release_dormant_hold")
    assert "WEG2_HANDOFF_PAGE_KEYS.pop" in s2[j:j + 1500]
