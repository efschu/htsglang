"""L3FILL-JOIN (z30y14, 27B image 02adfaadee, 01.10. 03:15:38Z) -- against the
NF form of the fill (L3FILL-JOINED: a JOINED claim is unclaimed at once, the
prefetch io thread waits a bounded time for its holder, a stale holder is
quarantined).

Specimen: P's followers filled the SAME canonical pages from the L3 store at
the same second (``ARENA-EVICT n=2 want=256 need=88`` on PP1 and PP2,
03:15:36). Their claims interleaved -- each held fresh claims on pages the
other had joined -- and each rank stopped at the other's first claim:
``#1433 L3->L2 fill: 5 of 5`` / ``12 of 12``, ``ARENA-FREE reason=
l3fill_past_prefix slots=59`` / ``55``; the ranks reached 95684 / 95716 tokens
where PP0 (reading alone, earlier) had 97631: ``#1400 STORE-TOLD MISMATCH
told=97631 own_prefix=93858``. The store view was the same on every rank (all
88 candidates passed the stat); only the concurrent fill diverged.

The NF wait (L3FILL-JOINED (2)) ran BEFORE the fill read its own claims: with
interleaved claims both followers waited for each other's unread pages, both
spent the bound, and both ended at the other's first claim -- the z30y14 shape
(red on f7099c0cbd). The fix reads this fill's own claims first and waits for
the joined ones afterwards (L3FILL-JOIN on the NF form).

Danger directions pinned here, on the REAL shm arena (arena.c) and real files,
the fills on prefetch io threads (``hicache-prefetch-io-<k>``) as on metal:
  * a page another rank is filling right now ends this rank's prefix early
    (rank-dependent store depth for one told);
  * two followers with interleaved claims reach different depths (z30y14);
  * a joined slot is freed whole under its first writer when this rank gives
    its join back (the #1427r class), or this rank's own completed pages past
    an unresolved join are freed although the other follower waits for them;
  * the join leaves the page CLAIMED (an open writer nobody resolves).
"""

import os
import shutil
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.mem_cache import hicache_storage as hs  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

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


def _on_prefetch_thread(k, fn, box, key):
    th = threading.Thread(target=lambda: box.__setitem__(key, fn()), name="hicache-prefetch-io-%d" % k)
    th.start()
    return th


class _InterleavedClaims:
    """One follower's view of the shared arena whose FIRST claim batch
    interleaves with the other follower's stem by stem (arena_claim takes
    one stem at a time, so two concurrent batches interleave): ``first(i)``
    says whether this follower claims stem i before the other one."""

    def __init__(self, arena, first, events):
        self._arena, self._first, self._ev, self._done = arena, first, events, False

    def __getattr__(self, name):
        return getattr(self._arena, name)

    def claim_slots(self, stems, totals, role=0):
        if self._done:
            return self._arena.claim_slots(stems, totals, role=role)
        self._done = True
        out = []
        for i, (st, tot) in enumerate(zip(stems, totals)):
            if not self._first(i):
                assert self._ev[i].wait(10), "the other follower never claimed stem %d" % i
            out += self._arena.claim_slots([st], [tot], role=role)
            if self._first(i):
                self._ev[i].set()
        return out


def test_a_page_another_rank_is_filling_does_not_end_this_ranks_prefix(tmp_path):
    """The other follower holds fresh claims on pages 5..9 and completes them
    while this rank's fill runs. This rank's prefix fill must still yield all
    N pages, byte-exact -- the joined pages through the bounded wait, never
    by writing into the other's slots."""
    be, stems = _store(tmp_path)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 64)
    other = arena.claim_slots(stems[5:10], [TOTAL] * 5)
    assert [st for _s, st, _g in other] == [0] * 5  # the other rank's fresh claims
    done = {}

    def _other_reads():
        time.sleep(0.3)
        for i, (s, _st, _g) in zip(range(5, 10), other):
            arena.slot_view(s, TOTAL)[:] = bytes([i + 1]) * TOTAL
        done["cs"] = arena.complete_slots([s for s, _st, _g in other], [g for _s, _st, g in other],
                                          [(0, TOTAL)])

    joins0 = hs._FILL_JOIN_N[1]
    box = {}
    with envs.FLLIPER_PDFLIP_L3FILL_JOIN_WAIT_MS.override(3000):
        th_o = threading.Thread(target=_other_reads)
        th_o.start()
        _on_prefetch_thread(0, lambda: HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True),
                            box, "out").join(15)
        th_o.join(15)
    out = box["out"]
    assert hs._FILL_JOIN_N[1] - joins0 == 5          # the five pages were JOINED, not raced in complete
    assert all(s is not None for s in out), [i for i, s in enumerate(out) if s is None]
    for i, s in enumerate(out):
        assert bytes(arena.slot_view(s, TOTAL)[:4]) == bytes([i + 1]) * 4
        assert arena.find_slots([stems[i]]) == [(s, 2)]
    # the other rank completed its own claims itself (status 1: this fill wrote
    # nothing into them), never recycled (3) or freed (4) under it
    assert done["cs"] == [1] * 5
    assert [out[i] for i in range(5, 10)] == [s for s, _st, _g in other]
    assert arena.stats()["claimed"] == 0


def test_both_followers_reach_the_same_depth(tmp_path):
    """The z30y14 shape: two followers, the same need set, interleaved claims
    (A claims the even pages first, B the odd ones), both fills concurrent on
    prefetch io threads. Both must answer the full prefix -- the same depth
    for one told. Red on f7099c0cbd: each waited for the other's unread pages
    and ended at the other's first claim (A at 1, B at 0)."""
    be, stems = _store(tmp_path)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 64)
    ev = [threading.Event() for _ in range(N)]
    view_a = _InterleavedClaims(arena, lambda i: i % 2 == 0, ev)
    view_b = _InterleavedClaims(arena, lambda i: i % 2 == 1, ev)
    box = {}
    with envs.FLLIPER_PDFLIP_L3FILL_JOIN_WAIT_MS.override(1500):
        ths = [_on_prefetch_thread(k, (lambda v=v: HiCacheFile.arena_fill_from_disk(
            be, v, stems, TOTAL, prefix=True)), box, k) for k, v in enumerate((view_a, view_b))]
        for th in ths:
            th.join(20)
    a, b = box[0], box[1]
    depth = lambda out: next((i for i, s in enumerate(out) if s is None), N)  # noqa: E731
    assert (depth(a), depth(b)) == (N, N)
    assert a == b
    for i, s in enumerate(a):
        assert bytes(arena.slot_view(s, TOTAL)[:4]) == bytes([i + 1]) * 4
        assert arena.find_slots([stems[i]]) == [(s, 2)]
    assert arena.stats()["claimed"] == 0


def test_a_join_given_back_leaves_the_slot_to_its_first_writer(tmp_path):
    """The other rank holds the claim on page 7 and does not complete it
    within this fill's bound. This rank gives its join back -- it must not
    free the slot under the other writer (#1427r), and its own pages past 7,
    already read and completed, stay in the L2 (the other follower may be
    waiting for exactly those); only the ANSWER past 7 is None (prefix)."""
    be, stems = _store(tmp_path)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 64)
    ((s7, st7, g7),) = arena.claim_slots([stems[7]], [TOTAL])
    assert st7 == 0
    box = {}
    with envs.FLLIPER_PDFLIP_L3FILL_JOIN_WAIT_MS.override(100):
        _on_prefetch_thread(1, lambda: HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True),
                            box, "out").join(15)
    out = box["out"]
    assert all(s is not None for s in out[:7])
    assert all(s is None for s in out[7:])
    assert arena.find_slots([stems[7]]) == [(s7, 1)]  # still CLAIMED, still the other's
    assert arena.claim_info([s7])[0][5] == 1          # one open writer: the other's (the join went)
    assert [st for _s, st in arena.find_slots(stems[8:])] == [2] * (N - 8)
    assert arena.complete_slots([s7], [g7], [(0, TOTAL)]) == [1]
    assert arena.stats()["claimed"] == 0


def test_a_fresh_claim_given_back_unread_is_freed(tmp_path):
    """Sole claimant, read failed: the slot goes back to the free list as
    before (no leaked CLAIMED slot)."""
    be, stems = _store(tmp_path, short={3})
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 64)
    out = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True)
    assert out[3] is None
    assert arena.find_slots([stems[3]]) == [(-1, 0)]
    assert arena.stats()["claimed"] == 0
