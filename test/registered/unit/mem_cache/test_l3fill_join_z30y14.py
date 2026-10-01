"""L3FILL-JOIN (z30y14, 27B image 02adfaadee, 01.10. 03:15:38Z).

Specimen: P's followers filled the SAME canonical pages from the L3 store at
the same second (``ARENA-EVICT n=2 want=256 need=88`` on PP1 and PP2,
03:15:36). ``arena_fill_from_disk`` treated a claim answered status 1 (JOIN:
another writer holds the claim) as a miss that ends the prefix, so each rank
stopped at the other's first claim -- ``#1433 L3->L2 fill: 5 of 5`` / ``12 of
12``, ``ARENA-FREE reason=l3fill_past_prefix slots=59`` / ``55`` -- and the
ranks reached 95684 / 95716 tokens where PP0 (reading alone, earlier) had
97631: ``#1400 STORE-TOLD MISMATCH told=97631 own_prefix=93858``. The store
view was the same on every rank (all 88 candidates passed the stat); only the
concurrent fill diverged.

Danger directions pinned here, on the REAL shm arena (arena.c) and real files:
  * a page another rank is filling right now ends this rank's prefix early
    (rank-dependent store depth for one told);
  * a joined slot is freed whole under its first writer (its completion comes
    back 3/4 -- the #1427r class) when this rank gives its claim back;
  * the join leaves the page CLAIMED (an open writer nobody resolves).
"""

import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096
N = 20


def _store(tmp_path, short=()):
    root = tmp_path / "store"
    root.mkdir()
    stems = ["p%04d" % i for i in range(N)]
    for i, st in enumerate(stems):
        size = TOTAL // 2 if i in short else TOTAL
        (root / (st + ".bin")).write_bytes(bytes([i + 1]) * size)

    def _path(stem):
        return os.path.join(str(root), stem + ".bin")

    be = object.__new__(HiCacheFile)
    be._existing_path = _path
    be._stat_stems = lambda ss: {s: os.path.getsize(_path(s)) for s in ss if os.path.exists(_path(s))}
    be._arena_evict_to_disk = lambda arena, want, need=0: 0
    return be, stems


def test_a_page_another_rank_is_filling_does_not_end_this_ranks_prefix(tmp_path):
    """The other follower holds fresh claims on pages 5..9 (its fill is in
    flight). This rank's prefix fill must still yield all N pages, byte-exact
    -- before the fix it ended at page 5."""
    be, stems = _store(tmp_path)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 64)
    other = arena.claim_slots(stems[5:10], [TOTAL] * 5)
    assert [st for _s, st, _g in other] == [0] * 5  # the other rank's fresh claims
    out = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True)
    assert all(s is not None for s in out), [i for i, s in enumerate(out) if s is None]
    for i, s in enumerate(out):
        assert bytes(arena.slot_view(s, TOTAL)[:4]) == bytes([i + 1]) * 4
        assert arena.find_slots([stems[i]]) == [(s, 2)]
    # the other rank completes its own claims afterwards: already COMPLETE (2),
    # never recycled (3) or freed (4) under it
    cs = arena.complete_slots([s for s, _st, _g in other], [g for _s, _st, g in other], [(0, TOTAL)])
    assert cs == [2] * 5
    assert arena.stats()["claimed"] == 0


def test_both_followers_reach_the_same_depth(tmp_path):
    """The z30y14 shape: two ranks, the same need set, interleaved claims.
    Rank B claimed every odd page first; rank A then fills, then B fills.
    Both must answer the full prefix -- the same depth for one told."""
    be, stems = _store(tmp_path)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 64)
    b_claims = arena.claim_slots(stems[1::2], [TOTAL] * (N // 2))
    a = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True)
    # B's own read of its claims lands after A completed them: status 2
    assert arena.complete_slots([s for s, _x, _g in b_claims], [g for _s, _x, g in b_claims],
                                [(0, TOTAL)]) == [2] * (N // 2)
    b = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True)
    depth = lambda out: next((i for i, s in enumerate(out) if s is None), N)  # noqa: E731
    assert depth(a) == depth(b) == N
    assert a == b


def test_a_failed_read_of_a_joined_page_leaves_the_slot_to_its_first_writer(tmp_path):
    """Page 7's file is short (the read fails). The other rank holds the claim
    and will complete it from its own source. This rank releases its join --
    it must not free the slot under the other writer (#1427r)."""
    be, stems = _store(tmp_path, short={7})
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 64)
    ((s7, st7, g7),) = arena.claim_slots([stems[7]], [TOTAL])
    assert st7 == 0
    out = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True)
    assert out[7] is None
    assert all(s is not None for i, s in enumerate(out) if i != 7)
    assert arena.find_slots([stems[7]]) == [(s7, 1)]  # still CLAIMED, still the other's
    assert arena.complete_slots([s7], [g7], [(0, TOTAL)]) == [1]


def test_a_fresh_claim_given_back_unread_is_freed(tmp_path):
    """Sole claimant, read failed: the slot goes back to the free list as
    before (no leaked CLAIMED slot)."""
    be, stems = _store(tmp_path, short={3})
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 64)
    out = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True)
    assert out[3] is None
    assert arena.find_slots([stems[3]]) == [(-1, 0)]
    assert arena.stats()["claimed"] == 0
