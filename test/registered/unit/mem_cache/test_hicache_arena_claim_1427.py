"""#1427 Stufe 4 (direct writes): two layer-shard ranks claim the SAME page
key, each DMAs its own extent into the slot, and the page is COMPLETE only
when both extents are in. No payload copy through arena_write."""

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

SLOT = 4096


@pytest.fixture
def arena(tmp_path):
    a = ShmArena(str(tmp_path / "arena-4096.bin"), SLOT, 16)
    yield a
    a.close()


def test_fresh_claim_join_and_complete_by_two_ranks(arena):
    stems = ["p0", "p1"]
    r0 = arena.claim_slots(stems, [SLOT, SLOT])
    assert [st for _, st, _ in r0] == [0, 0]           # rank 0: fresh
    r1 = arena.claim_slots(stems, [SLOT, SLOT])
    assert [s for s, _, _ in r1] == [s for s, _, _ in r0]
    assert [st for _, st, _ in r1] == [1, 1]           # rank 1: joins the same slots
    assert arena.find_slots(stems) == [(s, 1) for s, _, _ in r0]   # CLAIMED
    # rank 0 writes the first half, rank 1 the second half, straight into the slot
    for s, _, _ in r0:
        arena.slot_view(s, SLOT)[: SLOT // 2] = b"A" * (SLOT // 2)
        arena.slot_view(s, SLOT)[SLOT // 2:] = b"B" * (SLOT // 2)
    c0 = arena.complete_slots([s for s, _, _ in r0], [g for _, _, g in r0], [(0, SLOT // 2)])
    assert c0 == [0, 0]                                # merged, not full
    assert arena.find_slots(stems)[0][1] == 1
    c1 = arena.complete_slots([s for s, _, _ in r1], [g for _, _, g in r1], [(SLOT // 2, SLOT // 2)])
    assert c1 == [1, 1]                                # completed now
    assert [st for _, st in arena.find_slots(stems)] == [2, 2]
    assert bytes(arena.slot_view(r0[0][0], SLOT)[:4]) == b"AAAA"
    # a third claim sees COMPLETE and has nothing to write
    assert [st for _, st, _ in arena.claim_slots(stems, [SLOT, SLOT])] == [2, 2]
    # readers can still pin it
    assert arena.ref_slots([r0[0][0]], +1) == 1


def test_a_recycled_slot_refuses_the_late_completion(arena):
    (s, st, g), = arena.claim_slots(["q"], [SLOT])
    assert st == 0
    arena.free_slots([s])
    (s2, st2, g2), = arena.claim_slots(["r"], [SLOT])
    # the generation moved on: the old writer's completion is refused
    assert arena.complete_slots([s], [g], [(0, SLOT)]) == [3]
    assert arena.complete_slots([s2], [g2], [(0, SLOT)]) == [1]


def test_too_large_and_full_arena(arena):
    assert arena.claim_slots(["big"], [SLOT + 1])[0][1] == 3
    got = arena.claim_slots([f"k{i}" for i in range(16)], [SLOT] * 16)
    assert all(st == 0 for _, st, _ in got)
    assert arena.claim_slots(["one_more"], [SLOT])[0][1] == 4
