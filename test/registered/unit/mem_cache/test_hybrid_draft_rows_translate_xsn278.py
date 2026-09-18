"""weg2xsn277/278 (18.09.2026): group D's hybrid cache controller handed the
draft pool the RAW target rows; the DFlash window pool on D has 4113 slots
and carries a slot mapper the base controller translates through. The 4k
smoke wrote 200 rows past the pool (silent), the first 98k load crashed
(illegal memory access); xsn278's WEG2-ARENA-LOAD guard named it
(dst=[1025,4314] of 4114). Both hybrid branches now translate."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hc  # noqa: E402


class _Mapper:
    def __init__(self):
        self.calls = []

    def translate_read(self, idx):
        self.calls.append(("read", idx.tolist()))
        return idx % 7

    def translate_write(self, idx):
        self.calls.append(("write", idx.tolist()))
        return idx % 5


def test_a_window_pool_translates_and_a_mirror_pool_does_not():
    m = _Mapper()
    pool = SimpleNamespace(weg2_slot_mapper=m)
    idx = torch.tensor([1025, 4314])
    assert hc._draft_device_rows(pool, idx, "load").tolist() == [1025 % 5, 4314 % 5]
    assert hc._draft_device_rows(pool, idx, "write").tolist() == [1025 % 7, 4314 % 7]
    assert m.calls == [("write", [1025, 4314]), ("read", [1025, 4314])]
    mirror = SimpleNamespace()
    assert hc._draft_device_rows(mirror, idx, "load") is idx


def test_both_hybrid_branches_translate_the_draft_rows():
    src = open(hc.__file__).read()
    i = src.index("self.mem_pool_host_draft.backup_from_device_all_layer(")
    assert '_draft_device_rows(self.mem_pool_device_draft, device_indices, "write")' in src[i:i + 400]
    j = src.index("self.mem_pool_host_draft.load_to_device_per_layer(")
    assert '_draft_device_rows(self.mem_pool_device_draft, device_indices, "load")' in src[j:j + 400]
    # the base controller's rule, verbatim: write -> translate_read, load -> translate_write
    from sglang.srt.managers import cache_controller as cc
    b = open(cc.__file__).read()
    k = b.index("def _draft_device_indices")
    assert 'if direction == "write":' in b[k:k + 800] and "translate_read(device_indices)" in b[k:k + 800]
