"""Posten 2 (18.09.): ARENA-GET asks the store suffix once per key CLASS."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.cache_controller import weg2_suffixed_stems  # noqa: E402


class _Backend:
    def __init__(self):
        self.calls = []

    def _suffix_for_key(self, key):
        self.calls.append(key)
        if "." not in key:
            return "_kv", True
        if key.endswith(".mamba"):
            return "_mb", True
        return "_dr", False


def test_one_question_per_class_and_exact_stems():
    b = _Backend()
    keys = ["%040x" % i for i in range(1000)] + ["abc.mamba", "def.mamba", "ghi.draft-x"]
    out = weg2_suffixed_stems(b, keys)
    assert out[:2] == [keys[0] + "_kv", keys[1] + "_kv"]
    assert out[-3:] == ["abc.mamba_mb", "def.mamba_mb", "ghi.draft-x_dr"]
    assert len(b.calls) == 3, b.calls
    assert weg2_suffixed_stems(b, []) == []


def test_both_arena_get_sites_use_it():
    from sglang.srt.managers import cache_controller as cc
    src = open(cc.__file__).read()
    assert src.count("weg2_suffixed_stems(self.storage_backend, hash_values)") == 1
    assert src.count("weg2_suffixed_stems(self.storage_backend, draft_keys)") == 1
    assert "_get_suffixed_key(k) for k in hash_values" not in src
