"""xsn342 (18.09.2026): PP1 wedged 150 s in arena claim_slot during the retain
sweep. Desk repro: after ~100 claim+abort rounds of one 4096-page node the
clock hand had wrapped and every fresh claim re-walked the 315k-slot occupied
prefix (2 us -> 2.8 ms per claim, 11.6 s per node). Second defect in the same
boot: abort_write freed slots without unlinking their index cells, so the next
claim of the same key was handed a slot another writer had meanwhile claimed
(ARENA-COMPLETE LOST 'recycled under the writer', slots 315678..315680)."""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

SB = 64


def _arena(tmp_path, slots):
    return ShmArena(str(tmp_path / f"arena-{slots}.bin"), SB, slots)


def _stems(tag, n):
    return [f"{tag}-{i:07d}" for i in range(n)]


def _claim(a, st):
    return a.claim_slots(st, [SB] * len(st))


def _complete(a, got):
    a.complete_slots([g[0] for g in got], [g[2] for g in got], [(0, SB)])


def test_fresh_claim_stays_o1_after_the_clock_hand_wrapped(tmp_path):
    slots = 65536
    a = _arena(tmp_path, slots)
    # occupied prefix: 60 % of the arena COMPLETE
    prefix = int(slots * 0.6)
    got = _claim(a, _stems("pre", prefix))
    _complete(a, got)
    assert a.stats()["complete"] == prefix
    # the sweep's churn: claim + abort of the same node until the hand wraps
    node = _stems("node", 512)
    for _ in range(slots // 512 + 4):
        g = _claim(a, node)
        a.free_slots([s for s, st, _ in g if st == 0])
    t = time.perf_counter()
    g = _claim(a, _stems("fresh", 1024))
    dt = time.perf_counter() - t
    assert all(st == 0 for _, st, _ in g)
    # before the fix: every claim walked the whole prefix (measured 11.6 s for
    # 4096 claims over a 315k prefix); 1024 claims over a 39k prefix would be
    # ~0.5-1 s. With the hand following the claim: microseconds each.
    assert dt < 0.15, f"1024 fresh claims took {dt*1000:.0f} ms after the hand wrapped"


def test_free_unlinks_the_index_cell_and_reclaim_is_a_real_claim(tmp_path):
    a = _arena(tmp_path, 4096)
    node = _stems("n", 256)
    g1 = _claim(a, node)
    assert all(st == 0 for _, st, _ in g1)
    a.free_slots([s for s, _, _ in g1])
    st = a.stats()
    assert st["claimed"] == 0
    # another writer takes the arena's next slots (they are the freed ones)
    g_other = _claim(a, _stems("other", 256))
    other_slots = {s for s, _, _ in g_other}
    # the re-claim of the freed keys must NOT be handed the other writer's slots
    g2 = _claim(a, node)
    mine = {s for s, _, _ in g2}
    assert not (mine & other_slots), "re-claim handed out slots another writer owns"
    assert all(st == 0 for _, st, _ in g2), "a fresh claim reports fresh"
    assert a.stats()["claimed"] == 512
    # and every returned slot really carries the key (find agrees)
    found = a.find_slots(node)
    assert [s for s, _ in found] == [s for s, _, _ in g2]


def test_join_of_a_live_claim_is_status_1_not_fresh(tmp_path):
    a = _arena(tmp_path, 1024)
    node = _stems("j", 64)
    g1 = _claim(a, node)
    g2 = _claim(a, node)
    assert [s for s, _, _ in g2] == [s for s, _, _ in g1]
    assert all(st == 1 for _, st, _ in g2)
    assert a.stats()["claimed"] == 64


def test_claim_by_stems_hashes_in_c_like_python(tmp_path):
    from sglang.srt.mem_cache.storage.file.hicache_arena import key128
    a = _arena(tmp_path, 2048)
    st = _stems("h", 300)
    got = _claim(a, st)
    assert all(s == 0 for _, s, _ in got)
    # the C hash agrees with the Python one: find by stem lands on the same slots
    found = a.find_slots(st)
    assert [s for s, _ in found] == [s for s, _, _ in got]
    # and the header keys equal key128
    lo, hi = a._keys(st[:1])
    assert (int(lo[0]), int(hi[0])) == key128(st[0])
