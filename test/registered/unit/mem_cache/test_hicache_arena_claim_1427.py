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


def test_a_full_arena_refuses_in_constant_time_and_free_restores_the_counters(arena):
    import time
    got = arena.claim_slots([f"f{i}" for i in range(16)], [SLOT] * 16)
    assert all(st == 0 for _, st, _ in got)
    st = arena.stats()
    t0 = time.perf_counter()
    for _ in range(2000):
        assert arena.claim_slots(["late"], [SLOT])[0][1] == 4
    assert time.perf_counter() - t0 < 1.0, "#1431: a full arena must not walk every slot per claim"
    arena.free_slots([s for s, _, _ in got[:4]])
    assert [x for _, x, _ in arena.claim_slots(["a", "b", "c", "d"], [SLOT] * 4)] == [0, 0, 0, 0]
    assert arena.claim_slots(["e"], [SLOT])[0][1] == 4


def test_find_by_stem_hashes_in_c_and_agrees_with_the_python_key(arena):
    """#1439: arena_find_stems computes key128 in C; the answer must equal the
    Python-hashed path for present and absent stems alike."""
    from sglang.srt.mem_cache.storage.file.hicache_arena import key128
    stems = [f"cafe{i:02d}_Qwen.kv" for i in range(6)] + ["nope.mamba"]
    got = arena.claim_slots(stems[:4], [SLOT] * 4)
    arena.complete_slots([s for s, _, _ in got], [g for _, _, g in got], [(0, SLOT)])
    by_stem = arena.find_slots(stems)
    assert [st for _, st in by_stem] == [2, 2, 2, 2, 0, 0, 0]
    assert arena.find_states(stems) == [2, 2, 2, 2, 0, 0, 0]
    # the C key equals the Python key: a slot claimed by stem is found by the Python-hashed lookup too
    lo, hi = arena._keys(stems[:1])
    import ctypes
    slots = (ctypes.c_int64 * 1)(); st = (ctypes.c_int8 * 1)()
    arena._lib.arena_find_slots(arena._base, 1, lo, hi, slots, st)
    assert (int(slots[0]), int(st[0])) == (by_stem[0][0], 2)
