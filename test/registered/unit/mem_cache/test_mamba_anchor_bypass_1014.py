"""ANCHOR-BYPASS (NF h13 84e787ec96, D TP0 14:16:17, weg2-10-82).

The 32-slot mamba arena was full of the lane-held requests' PARK anchors;
every lane resume waited the 5 s ROOM-WAIT and missed its anchor (D's store
vote 0, W31, full prefill on P) although the blob was on disk. Now an anchor
the arena has no slot for is read into a host STAGING row: the fill is asked
without a room wait, the staging row goes into the transfer's host indices and
``None`` sends the key down the per-key read (HiCacheFile.get -> the rank's
canonical slice).
"""

import os
import threading

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.mem_cache.pool_host import arena_mamba_pool as amp
from sglang.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost

S, A = 4, 32


class _Arena:
    def __init__(self, found):
        self.found = found
        self.refs = []

    def find_slots(self, stems):
        return [self.found] * len(stems)

    def ref_slots(self, slots, delta):
        self.refs.append((list(slots), delta))
        return len(slots)


class _Backend:
    def __init__(self, fill_slot):
        self.fill_slot = fill_slot
        self.room_waits = []

    def _get_suffixed_key(self, k):
        return k + ".mamba"

    def arena_fill_from_disk(self, arena, stems, total_bytes, prefix=False, room_wait=True):
        self.room_waits.append(room_wait)
        return [self.fill_slot] * len(stems)


def _pool(found=(-1, 0)):
    p = object.__new__(ArenaMambaPoolHost)
    p.staging_rows = S
    p.arena_slots = A
    p.arena = _Arena(found)
    p._page_bytes = 4096
    p.page_size = 1
    p.size = S
    p.free_slots = torch.arange(S, dtype=torch.int64)
    p.lock = threading.RLock()
    return p


def test_a_full_arena_reads_the_anchor_into_a_staging_row(monkeypatch):
    monkeypatch.delenv(amp.ENV_ANCHOR_BYPASS, raising=False)
    p = _pool()
    be = _Backend(fill_slot=None)          # the arena has no slot
    hi = torch.tensor([S + A + 7], dtype=torch.int64)    # the read placeholder
    out = p.arena_resolve_reads(be, hi, ["k0"])
    assert out == [None]                   # -> the caller's per-key read
    assert 0 <= int(hi[0]) < S             # a staging row, not a placeholder
    assert be.room_waits == [False]        # no 5 s room wait with a row in hand
    assert len(p.free_slots) == S - 1


def test_an_arena_slot_wins_and_gives_the_staging_row_back(monkeypatch):
    monkeypatch.delenv(amp.ENV_ANCHOR_BYPASS, raising=False)
    p = _pool()
    be = _Backend(fill_slot=9)             # the fill found room after all
    hi = torch.tensor([S + A + 7], dtype=torch.int64)
    out = p.arena_resolve_reads(be, hi, ["k0"])
    assert out == [True] and int(hi[0]) == S + 9
    assert len(p.free_slots) == S


def test_no_staging_row_keeps_the_room_wait_and_the_miss(monkeypatch):
    monkeypatch.delenv(amp.ENV_ANCHOR_BYPASS, raising=False)
    p = _pool()
    p.free_slots = torch.arange(0, dtype=torch.int64)    # every staging row in use
    be = _Backend(fill_slot=None)
    hi = torch.tensor([S + A + 7], dtype=torch.int64)
    out = p.arena_resolve_reads(be, hi, ["k0"])
    assert out == [False] and be.room_waits == [True]


def test_the_switch_off_is_the_arena_only_read(monkeypatch):
    monkeypatch.setenv(amp.ENV_ANCHOR_BYPASS, "0")
    p = _pool()
    be = _Backend(fill_slot=None)
    hi = torch.tensor([S + A + 7], dtype=torch.int64)
    out = p.arena_resolve_reads(be, hi, ["k0"])
    assert out == [False] and be.room_waits == [True] and len(p.free_slots) == S
