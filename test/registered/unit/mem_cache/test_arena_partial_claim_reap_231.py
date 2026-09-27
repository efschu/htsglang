"""#231 (rc12m-dpr 09271152): the P mamba arena (32 slots) filled with slots
that could never become COMPLETE.

A slot is COMPLETE only when every writer rank merged its extents. The P
ranks decide their anchor claims locally -- PP0 reads the store and finds a
node already backed, PP1/PP2 publish it (MAMBA-ARENA weg2-0-11: PP0 written=1,
PP1/PP2 written=3) -- so a stem PP1/PP2 claimed and wrote but PP0 never joined
stays CLAIMED: no reader may use it, no evictor may take it, and it is not
FREE. Census 12:00 -> 12:15: complete 29 -> 5 while every claim found no
free slot; from 12:13:14 every END anchor was refused (end_anchor=missing),
D resumed short (weg2-18-64: deliverable 37248, deepest anchor 17536,
shortfall 19712) or not at all (#928 'no recurrent state'), W50, P prefilled
again.

The mamba arena's claim now takes the room of a slot NO writer can still
come to: every claimer merged or gave the claim up (open-writer count 0 in
the slot header), no reference, untouched for the age. A claim that is open
-- its rank asleep through a D phase, behind, mid flip -- is never taken, by
any process, for any length of time. Hermetic: the real C arena (two
mappings of one file = P and D), the real _evict_for_claim of both pools."""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost  # noqa: E402
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

SLOT = 4096
HALF = SLOT // 2


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "arena-mamba.bin")


@pytest.fixture
def arena(path):
    a = ShmArena(path, SLOT, 4)
    yield a
    a.close()


@pytest.fixture
def fast_reap(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ARENA_PARTIAL_REAP_S", "0.05")


def _mamba_room(arena, need=1):
    """The claim-time room of the MAMBA anchor arena (what _claim calls)."""
    return ArenaMambaPoolHost._evict_for_claim(object.__new__(ArenaMambaPoolHost), arena, need)


def _orphan(arena, stem):
    """PP1 claims fresh, PP2 joins, both merge their halves' first quarter and
    take the node's reader reference; PP0 never joins; the tree reset gives
    the references back. The slot stays CLAIMED with no open writer."""
    (s, st, g), = arena.claim_slots([stem], [SLOT])
    assert st == 0
    (s2, st2, g2), = arena.claim_slots([stem], [SLOT])
    assert (s2, st2) == (s, 1)
    assert arena.complete_slots([s], [g], [(SLOT // 4, SLOT // 4)]) == [0]
    assert arena.complete_slots([s], [g2], [(HALF, SLOT // 4)]) == [0]
    assert arena.ref_slots([s, s], +1) == 2
    assert arena.ref_slots([s, s], -1) == 2        # P's reset (release_tree_rows)
    assert arena.find_slots([stem]) == [(s, 1)]
    return s, g


def test_metal_shape_orphaned_claims_fill_the_arena_and_the_end_anchor_gets_a_slot(arena, fast_reap):
    """RED on 2f800cac1a: the four orphans hold every slot and the claim of
    the next request's END anchor gets status 4 even after the room making
    -- the rc12m-dpr 'end_anchor=missing' from 12:13:14 on."""
    for i in range(4):
        _orphan(arena, f"anchor-{i}")
    assert arena.stats()["claimed"] == 4 and arena.stats()["complete"] == 0
    assert arena.claim_slots(["end-anchor"], [SLOT])[0][1] == 4
    time.sleep(0.1)
    assert _mamba_room(arena) == 4
    (s, st, g), = arena.claim_slots(["end-anchor"], [SLOT])
    assert st == 0, "the END anchor gets a fresh slot"
    assert arena.complete_slots([s], [g], [(0, SLOT)]) == [1]
    assert arena.find_slots(["end-anchor"]) == [(s, 2)]
    assert arena.find_slots(["anchor-0"]) == [(-1, 0)], "an orphan is gone from the index"
    # PP0 comes for the stem after all: a FRESH slot of its own, never a refused completion
    (s0, st0, g0), = arena.claim_slots(["anchor-0"], [SLOT])
    assert st0 == 0
    assert arena.complete_slots([s0], [g0], [(0, SLOT // 4)]) == [0]


def test_p_asleep_through_a_d_phase_keeps_its_open_claim_and_completes_after_the_wake(path, fast_reap):
    """The review's mandatory case: P claims (PP1 merged its half, PP2 has
    claimed and not merged), P sleeps -- its reset drops PP1's node reference
    -- while D, awake on the SAME arena, needs room and asks for it again and
    again long past the reap age. Nothing of P's is taken; after the wake PP2
    merges and the anchor is COMPLETE."""
    p = ShmArena(path, SLOT, 4)
    d = ShmArena(path, SLOT, 4)                      # D's own mapping of the shared arena
    try:
        (s, st1, g1), = p.claim_slots(["p-anchor"], [SLOT])     # PP1
        (_, st2, g2), = p.claim_slots(["p-anchor"], [SLOT])     # PP2 joins
        assert (st1, st2) == (0, 1)
        assert p.complete_slots([s], [g1], [(0, HALF)]) == [0]  # PP1 merged
        assert p.ref_slots([s], +1) == 1
        assert p.ref_slots([s], -1) == 1                        # P's sleep reset
        # D fills the rest and keeps needing room while P sleeps
        for i in range(3):
            (ds, _, dg), = d.claim_slots([f"d-{i}"], [SLOT])
            assert d.complete_slots([ds], [dg], [(0, SLOT)]) == [1]
            assert d.ref_slots([ds], +1) == 1                    # D's live requests read them
        for _ in range(4):
            time.sleep(0.06)                                     # >> the reap age every round
            assert _mamba_room(d) == 0
            assert d.claim_slots(["d-more"], [SLOT])[0][1] == 4
        assert p.find_slots(["p-anchor"]) == [(s, 1)], "P's open claim survived D's phase"
        # P wakes, PP2 merges its half: COMPLETE
        assert p.complete_slots([s], [g2], [(HALF, HALF)]) == [1]
        assert d.find_slots(["p-anchor"]) == [(s, 2)]
    finally:
        d.close()
        p.close()


def test_a_referenced_partial_slot_is_never_reaped(arena, fast_reap):
    (s, _, g), = arena.claim_slots(["held"], [SLOT])
    assert arena.complete_slots([s], [g], [(0, HALF)]) == [0]
    assert arena.ref_slots([s], +1) == 1            # a live tree node still reads its own layers
    for i in range(3):
        _orphan(arena, f"o{i}")
    time.sleep(0.1)
    assert _mamba_room(arena) == 3
    assert arena.find_slots(["held"]) == [(s, 1)]


def test_a_writer_given_up_by_abort_makes_the_slot_an_orphan(arena, fast_reap):
    (s, _, g1), = arena.claim_slots(["gave-up"], [SLOT])
    (_, st, g2), = arena.claim_slots(["gave-up"], [SLOT])
    assert st == 1
    assert arena.complete_slots([s], [g1], [(0, HALF)]) == [0]
    time.sleep(0.1)
    assert _mamba_room(arena) == 0                  # the joiner is still open
    assert arena.unclaim([s], [g2]) == 1            # abort_write of the join
    time.sleep(0.1)
    assert _mamba_room(arena) == 1


def test_a_recent_orphan_waits_for_the_age(arena, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ARENA_PARTIAL_REAP_S", "30")
    for i in range(4):
        _orphan(arena, f"fresh{i}")
    assert _mamba_room(arena) == 0
    assert arena.stats()["claimed"] == 4


def test_the_reap_can_be_switched_off(arena, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ARENA_PARTIAL_REAP_S", "0")
    for i in range(4):
        _orphan(arena, f"off{i}")
    time.sleep(0.05)
    assert _mamba_room(arena) == 0


def test_complete_unreferenced_slots_still_go_first(arena, fast_reap):
    (s, _, g), = arena.claim_slots(["done"], [SLOT])
    assert arena.complete_slots([s], [g], [(0, SLOT)]) == [1]
    for i in range(3):
        _orphan(arena, f"x{i}")
    time.sleep(0.1)
    assert _mamba_room(arena) == 1                  # the COMPLETE one suffices
    assert arena.stats()["claimed"] == 3


def test_outside_the_mamba_anchor_arena_the_reap_is_a_no_op(arena, fast_reap):
    """27B form (host_anchor_slots, 0 MAMBA-ARENA claims) and every KV arena:
    the KV pool's claim room never reaps, orphans or not."""
    for i in range(4):
        _orphan(arena, f"kv{i}")
    time.sleep(0.1)
    kv_pool = object.__new__(ArenaMHAHostPool)
    assert ArenaMHAHostPool._evict_for_claim(kv_pool, arena, 1) == 0
    assert arena.stats()["claimed"] == 4


def test_payload_written_partials_are_never_judged(arena, fast_reap):
    """The payload path (arena.write, no direct claim) keeps its partial
    pages: the reap judges only slots entered through the direct protocol."""
    import ctypes
    buf = ctypes.create_string_buffer(HALF)
    st = arena.write(["pay"], [SLOT], [[(0, HALF)]], [ctypes.addressof(buf)])
    assert st == [0]
    for i in range(3):
        _orphan(arena, f"q{i}")
    time.sleep(0.1)
    assert _mamba_room(arena) == 3
    assert arena.find_states(["pay"]) == [1]


def test_a_join_freed_unwritten_by_the_mamba_pool_is_no_longer_open(arena, fast_reap):
    """The node that took a JOIN claim is freed before its write landed
    (ArenaMambaPoolHost.free of a pending row): the join is given up, so the
    slot is an orphan once the other writer merged -- not open forever."""
    import torch
    (s, _, g1), = arena.claim_slots(["freed-join"], [SLOT])
    (_, st, g2), = arena.claim_slots(["freed-join"], [SLOT])
    assert st == 1
    assert arena.complete_slots([s], [g1], [(0, HALF)]) == [0]
    pool = object.__new__(ArenaMambaPoolHost)
    pool.arena, pool.staging_rows, pool.arena_slots = arena, 0, 4
    pool._pending = {s: (g2, False)}
    time.sleep(0.1)
    assert _mamba_room(arena) == 0                  # the join is still open
    assert pool.free(torch.tensor([s])) == 1
    assert pool._pending == {}
    time.sleep(0.1)
    assert _mamba_room(arena) == 1
