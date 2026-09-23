"""weg2xsn277 (18.09.2026): the first 98k-token arena->device load of the
boot died with 'CUDA error: an illegal memory access was encountered',
reported asynchronously at `_transfer` (the 4k smoke had loaded fine). The
load now refuses BY NAME before the kernel when a slot lies outside the
arena, a destination row outside the device pool, or a slot's page is not
registered, and prints the load's terms once per load (layer 0).
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402


class _Pool:
    def __init__(self, rows):
        self.k_buffer = [torch.zeros((rows, 4), dtype=torch.uint8)]


def _p(A=100, pinned=None):
    p = ap.ArenaMHAHostPool.__new__(ap.ArenaMHAHostPool)
    p.arena_slots = A
    p._pinned = pinned
    p.row_slot = {i: i * 2 for i in range(50)}       # rows 0..49 -> slots 0..98
    p.arena_k_refs = [object()]
    p.arena_v_refs = [object()]
    p._zero_k = p._zero_v = object()
    p.calls = []
    p._transfer = lambda dp, k, v, s, d, l: p.calls.append((s.tolist(), d.tolist(), l))
    return p


def test_a_valid_load_launches_with_the_resolved_slots():
    p = _p()
    ap.ArenaMHAHostPool._load_arena(p, _Pool(1000), torch.tensor([1, 2, 60]), torch.tensor([10, 11, 12]), 0)
    # row 60 has no slot -> the zero path; rows 1,2 -> slots 2,4
    assert p.calls[0] == ([0], [12], 0)
    assert p.calls[1] == ([2, 4], [10, 11], 0)


def test_a_slot_outside_the_arena_is_refused_by_name():
    p = _p(A=3)                                        # slot 4 (row 2) is outside
    with pytest.raises(RuntimeError) as exc:
        ap.ArenaMHAHostPool._load_arena(p, _Pool(1000), torch.tensor([1, 2]), torch.tensor([10, 11]), 0)
    assert "WEG2-ARENA-LOAD REFUSED" in str(exc.value) and "outside the arena of 3 slots" in str(exc.value)
    assert p.calls == []


def test_a_device_row_outside_the_pool_is_refused_by_name():
    p = _p()
    with pytest.raises(RuntimeError) as exc:
        ap.ArenaMHAHostPool._load_arena(p, _Pool(11), torch.tensor([1, 2]), torch.tensor([10, 11]), 0)
    assert "device rows [10,11] outside the pool of 11 rows" in str(exc.value)


def test_an_unregistered_slot_page_is_refused_by_name():
    bm = torch.ones(100, dtype=torch.bool)
    bm[4] = False                                      # row 2 -> slot 4 not pinned
    p = _p(pinned=bm)
    with pytest.raises(RuntimeError) as exc:
        ap.ArenaMHAHostPool._load_arena(p, _Pool(1000), torch.tensor([1, 2]), torch.tensor([10, 11]), 0)
    assert "not all registered" in str(exc.value)
    p2 = _p(pinned=torch.ones(100, dtype=torch.bool))
    ap.ArenaMHAHostPool._load_arena(p2, _Pool(1000), torch.tensor([1, 2]), torch.tensor([10, 11]), 0)
    assert p2.calls == [([2, 4], [10, 11], 0)]
