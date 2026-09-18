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
    def __init__(self, device="cpu"):
        self.calls = []
        self.device = device

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
    # xsn281: the load branch translates ONCE per load through _draft_rows_or_skip
    assert "draft_rows," in src[j:j + 400]
    k = src.index("_draft_rows_or_skip(\n")
    assert 'self.mem_pool_device_draft, device_indices, "load"' in src[k:k + 200]
    # the base controller's rule, verbatim: write -> translate_read, load -> translate_write
    from sglang.srt.managers import cache_controller as cc
    b = open(cc.__file__).read()
    k = b.index("def _draft_device_indices")
    assert 'if direction == "write":' in b[k:k + 800] and "translate_read(device_indices)" in b[k:k + 800]


def test_xsn279_the_rows_are_moved_to_the_mappers_device_first():
    """xsn279: 'Expected all tensors to be on the same device, cuda:0 and
    cpu' -- the hybrid path carries CPU pair lists, the mapper's table lives
    on the device. The rows are moved to mapper.device before translating."""
    seen = {}

    class _M(_Mapper):
        def translate_write(self, idx):
            seen["device"] = str(idx.device)
            return idx
    m = _M(device="cpu")
    pool = SimpleNamespace(weg2_slot_mapper=m)
    hc._draft_device_rows(pool, torch.tensor([1, 2]), "load")
    assert seen["device"] == "cpu"
    src = open(hc.__file__).read()
    i = src.index("def _draft_device_rows")
    assert 'getattr(mapper, "device", None)' in src[i:i + 1500] and "device_indices.to(device=dev)" in src[i:i + 1500]


def test_xsn281_an_exhausted_window_pool_skips_the_draft_half_by_name(caplog):
    """xsn281: the mapper raised 'DFLASH small solo pool exhausted: need 202
    more draft slots but only 0 reclaimable of 4113 total (ctx cap 2048)'
    for the 4316-token smoke and the group died. The draft half is skipped
    by name; the target load is untouched."""
    import logging

    class _M(_Mapper):
        def translate_write(self, idx):
            raise RuntimeError("DFLASH small solo pool exhausted: need 202 more draft slots "
                               "but only 0 reclaimable of 4113 total (ctx cap 2048).")
    m = _M(); m.num_draft_slots = 4113; m.ctx_cap = 2048
    pool = SimpleNamespace(weg2_slot_mapper=m)
    hc._DRAFT_SKIP_LOGGED["n"] = 0
    with caplog.at_level(logging.INFO):
        rows, skipped = hc._draft_rows_or_skip(pool, torch.tensor([1, 2, 3]), "load", nrows=4313)
    assert rows is None and skipped is True
    assert any("WEG2-DRAFT-LOAD SKIPPED rows=4313" in r.getMessage() and "ctx_cap=2048" in r.getMessage()
               for r in caplog.records)
    # any other error still raises
    class _Boom(_Mapper):
        def translate_write(self, idx):
            raise RuntimeError("something else")
    import pytest
    with pytest.raises(RuntimeError):
        hc._draft_rows_or_skip(SimpleNamespace(weg2_slot_mapper=_Boom()), torch.tensor([1]), "load", nrows=1)
    # a healthy mapper: rows come back translated, not skipped
    rows, skipped = hc._draft_rows_or_skip(SimpleNamespace(weg2_slot_mapper=_Mapper()), torch.tensor([6, 7]), "load", nrows=2)
    assert rows.tolist() == [1, 2] and skipped is False
    src = open(hc.__file__).read()
    i = src.index("draft_rows, draft_rows_skipped = None, False")
    assert "_draft_rows_or_skip(" in src[i:i + 2500] and "if draft_rows is not None:" in src[i:i + 2500]
