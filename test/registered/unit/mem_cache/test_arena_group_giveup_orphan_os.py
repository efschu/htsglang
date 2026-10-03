"""OS: a claim the whole group gives up is freed; orphans are room before kept pages.

Metal (NF y3w e033a931db):

* D 01:39:13: the KV arena (6485 slots, ``complete=6472`` at 01:38:21) was
  full; a ~1900-page claim came back ``#1427 ARENA-CLAIM REFUSED statuses=
  [1, 2, 4]`` on TP0/TP2 and ``[0, 2, 4]`` on TP1. TP1 released its fresh
  claims (``#1427r CLAIM-RELEASE fresh=753 freed=0 kept=753`` -- kept because
  TP0/TP2 had joined), TP0/TP2 unclaimed their joins. Every rank resolved only
  ITS claim: 753 slots stayed CLAIMED with no open writer and no byte
  (``#1439 ... claimed=753`` on P), then 512 more at 01:40:20.
* P PP0 read weg2-1-4 at 01:39:45 (300 of 1952 pages) and 01:40:52 (764 of
  1955): each read stopped at such a slot.
* ``_evict_for_claim`` of the KV arena never reaped (#231 ran for the mamba
  arena only): claims evicted kept pages while orphans sat beside them.

The fix: the last writer out of a slot nobody wrote frees it (arena.c
``free_if_abandoned``, whichever rank that is -- one verdict from the slot's
own state); ``_evict_for_claim`` reaps orphans before it touches a page.

Hermetic: the real C arena (one mapping stands for the three ranks).
RED on a332187f28, GREEN with OS.
"""
from __future__ import annotations

import os
import shutil
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.mem_cache.pool_host import arena_pool  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

SLOT = 4096
HALF = SLOT // 2


@pytest.fixture
def arena(tmp_path):
    a = ShmArena(str(tmp_path / "arena-kv.bin"), SLOT, 8)
    yield a
    a.close()


def _group_claim(arena, stem):
    """TP1 first (fresh), TP0 and TP2 join -- the metal's statuses."""
    (s1, st1, g1), = arena.claim_slots([stem], [SLOT])
    (s0, st0, g0), = arena.claim_slots([stem], [SLOT])
    (s2, st2, g2), = arena.claim_slots([stem], [SLOT])
    assert (st1, st0, st2) == (0, 1, 1) and s0 == s1 == s2
    return s1, g1


@pytest.mark.parametrize("fresh_last", [False, True])
def test_y3w_a_claim_the_whole_group_gives_up_is_freed(arena, fresh_last):
    """RED on a332187f28: CLAIMED with no writer for ever (the metal's
    ``claimed=753``). GREEN: FREE, whichever rank gives up last."""
    s, g = _group_claim(arena, "p612")
    if fresh_last:
        arena.unclaim([s, s], [g, g])  # TP0, TP2 give their joins up
        st = arena.release_claims([s], [g], reason="claim_refused")  # TP1 last
        assert st == [0], "the last one out frees it"
    else:
        arena_pool._release_fresh(arena, [s], [g], "claim_refused")  # TP1 first: kept
        arena.unclaim([s], [g])  # TP0
        arena.unclaim([s], [g])  # TP2 -- the last one out
    assert arena.find_slots(["p612"]) == [(-1, 0)]
    assert arena.stats()["claimed"] == 0
    (_s, st2, _g), = arena.claim_slots(["p612"], [SLOT])
    assert st2 == 0, "the stem gets a fresh slot, never a join into an orphan"


def test_a_slot_with_merged_bytes_stays_for_its_writers(arena):
    """#1427r holds: TP1 merged its rows, then TP0/TP2 give up -- the bytes
    stay (the page is CLAIMED, #231's reap judges it after its age)."""
    s, g = _group_claim(arena, "tail")
    assert arena.complete_slots([s], [g], [(0, HALF)]) == [0]  # TP1's rows merged
    arena.unclaim([s], [g])  # TP0
    arena.unclaim([s], [g])  # TP2
    assert arena.find_slots(["tail"]) == [(s, 1)]


def test_an_open_writer_keeps_the_slot(arena):
    s, g = _group_claim(arena, "live")
    arena_pool._release_fresh(arena, [s], [g], "claim_refused")
    arena.unclaim([s], [g])  # one joiner still open
    assert arena.find_slots(["live"]) == [(s, 1)]


def test_the_claim_room_takes_orphans_before_any_complete_page(arena, monkeypatch):
    """RED on a332187f28: the KV claim room evicts a COMPLETE page and leaves
    the orphan (partial rows, no writer) where it is. GREEN: the orphan goes,
    the COMPLETE pages stay."""
    monkeypatch.setenv("SGLANG_WEG2_ARENA_PARTIAL_REAP_S", "0.05")
    import ctypes

    for i in range(7):
        pay = bytes([i]) * SLOT
        buf = ctypes.create_string_buffer(pay, SLOT)
        assert arena.write(["c%d" % i], [SLOT], [((0, SLOT),)], [ctypes.addressof(buf)]) == [1]
    s, g = _group_claim(arena, "orphan")
    assert arena.complete_slots([s], [g], [(0, HALF)]) == [0]  # rows merged, then all gave up
    arena.unclaim([s, s], [g, g])
    time.sleep(0.1)
    pool = object.__new__(arena_pool.ArenaMHAHostPool)
    monkeypatch.setattr(arena_pool._handoff_pending, "keep_for", lambda _p: None)
    freed = pool._evict_for_claim(arena, 1)
    assert freed >= 1
    assert arena.find_slots(["orphan"]) == [(-1, 0)]
    assert [st for _s, st in arena.find_slots(["c%d" % i for i in range(7)])] == [2] * 7
