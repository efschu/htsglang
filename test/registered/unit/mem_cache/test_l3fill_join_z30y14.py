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


# 27B port 20./21.10.: the OTHER RANK is a real second PROCESS sharing the shm arena, as on the metal (PP1/PP2 are rank
# processes). The NF fill takes over a claim of ITS OWN process that no live sibling thread holds (L3-FILL SELF-JOIN);
# an in-process thread standing in for another rank was therefore taken over -- a harness artifact, not the z30y14 shape.
# The danger directions and every assertion are unchanged.
import multiprocessing as _mp  # noqa: E402

_CTX = _mp.get_context("fork")


def _other_rank_fills_5_to_9(path, stems, conn, delay):
    a = ShmArena(path, TOTAL, 64)
    other = a.claim_slots(stems[5:10], [TOTAL] * 5)
    conn.send([(int(s_), int(st), int(g)) for s_, st, g in other])
    time.sleep(delay)
    for i, (s_, _st, _g) in zip(range(5, 10), other):
        a.slot_view(s_, TOTAL)[:] = bytes([i + 1]) * TOTAL
    conn.send(list(a.complete_slots([s_ for s_, _st, _g in other], [g for _s, _st, g in other], [(0, TOTAL)])))


def test_a_page_another_rank_is_filling_does_not_end_this_ranks_prefix(tmp_path):
    """The other follower holds fresh claims on pages 5..9 and completes them
    while this rank's fill runs. This rank's prefix fill must still yield all
    N pages, byte-exact -- the joined pages through the bounded wait, never
    by writing into the other's slots."""
    be, stems = _store(tmp_path)
    path = str(tmp_path / "kv.bin")
    arena = ShmArena(path, TOTAL, 64)
    pc, cc = _CTX.Pipe()
    proc = _CTX.Process(target=_other_rank_fills_5_to_9, args=(path, stems, cc, 0.3))
    proc.start()
    other = pc.recv()
    assert [st for _s, st, _g in other] == [0] * 5  # the other rank's fresh claims
    joins0 = hs._FILL_JOIN_N[1]
    box = {}
    with envs.FLLIPER_PDFLIP_L3FILL_JOIN_WAIT_MS.override(3000):
        _on_prefetch_thread(0, lambda: HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True),
                            box, "out").join(15)
    cs = pc.recv()
    proc.join(15)
    out = box["out"]
    assert hs._FILL_JOIN_N[1] - joins0 == 5          # the five pages were JOINED, not raced in complete
    assert all(s is not None for s in out), [i for i, s in enumerate(out) if s is None]
    for i, s in enumerate(out):
        assert bytes(arena.slot_view(s, TOTAL)[:4]) == bytes([i + 1]) * 4
        assert arena.find_slots([stems[i]]) == [(s, 2)]
    # the other rank completed its own claims itself (status 1: this fill wrote
    # nothing into them), never recycled (3) or freed (4) under it
    assert cs == [1] * 5
    assert [out[i] for i in range(5, 10)] == [s for s, _st, _g in other]
    assert arena.stats()["claimed"] == 0


def _follower_b(path, root, stems, evs, conn):
    be = object.__new__(HiCacheFile)
    be._existing_path = lambda stem: os.path.join(root, stem + ".bin")
    be._stat_stems = lambda ss: {x: os.path.getsize(be._existing_path(x)) for x in ss
                                 if os.path.exists(be._existing_path(x))}
    be._arena_evict_to_disk = lambda arena, want, need=0: 0
    view = _InterleavedClaims(ShmArena(path, TOTAL, 64), lambda i: i % 2 == 1, evs)
    box = {}
    with envs.FLLIPER_PDFLIP_L3FILL_JOIN_WAIT_MS.override(1500):
        _on_prefetch_thread(1, lambda: HiCacheFile.arena_fill_from_disk(be, view, stems, TOTAL, prefix=True),
                            box, "out").join(20)
    conn.send(box.get("out"))


def test_both_followers_reach_the_same_depth(tmp_path):
    """The z30y14 shape: two followers, the same need set, interleaved claims
    (A claims the even pages first, B the odd ones), both fills concurrent on
    prefetch io threads -- of two rank PROCESSES. Both must answer the full
    prefix -- the same depth for one told. Red on f7099c0cbd: each waited for
    the other's unread pages and ended at the other's first claim (A at 1, B at 0)."""
    be, stems = _store(tmp_path)
    path = str(tmp_path / "kv.bin")
    arena = ShmArena(path, TOTAL, 64)
    ev = [_CTX.Event() for _ in range(N)]
    pc, cc = _CTX.Pipe()
    proc = _CTX.Process(target=_follower_b, args=(path, str(tmp_path / "store"), stems, ev, cc))
    proc.start()
    view_a = _InterleavedClaims(arena, lambda i: i % 2 == 0, ev)
    box = {}
    with envs.FLLIPER_PDFLIP_L3FILL_JOIN_WAIT_MS.override(1500):
        _on_prefetch_thread(0, lambda: HiCacheFile.arena_fill_from_disk(be, view_a, stems, TOTAL, prefix=True),
                            box, 0).join(20)
    b = pc.recv()
    proc.join(20)
    a = box[0]
    depth = lambda out: next((i for i, s in enumerate(out) if s is None), N)  # noqa: E731
    assert (depth(a), depth(b)) == (N, N)
    assert a == b
    for i, s in enumerate(a):
        assert bytes(arena.slot_view(s, TOTAL)[:4]) == bytes([i + 1]) * 4
        assert arena.find_slots([stems[i]]) == [(s, 2)]
    assert arena.stats()["claimed"] == 0


def _other_rank_holds_7(path, stems, conn, release):
    a = ShmArena(path, TOTAL, 64)
    ((s7, st7, g7),) = a.claim_slots([stems[7]], [TOTAL])
    conn.send((int(s7), int(st7), int(g7)))
    release.wait(30)
    conn.send(list(a.complete_slots([s7], [g7], [(0, TOTAL)])))


def test_a_join_given_back_leaves_the_slot_to_its_first_writer(tmp_path):
    """The other rank holds the claim on page 7 and does not complete it
    within this fill's bound. This rank gives its join back -- it must not
    free the slot under the other writer (#1427r), and its own pages past 7,
    already read and completed, stay in the L2 (the other follower may be
    waiting for exactly those); only the ANSWER past 7 is None (prefix)."""
    be, stems = _store(tmp_path)
    path = str(tmp_path / "kv.bin")
    arena = ShmArena(path, TOTAL, 64)
    pc, cc = _CTX.Pipe()
    release = _CTX.Event()
    proc = _CTX.Process(target=_other_rank_holds_7, args=(path, stems, cc, release))
    proc.start()
    s7, st7, g7 = pc.recv()
    assert st7 == 0
    box = {}
    # the 27B give-back shape is the FIXED bound (JOIN-LIVENESS off, the stale reap above the bound): the NF liveness
    # wait instead follows a holder that touches its claims (every NF-code rank does, S23) and reaps one that is silent
    # for the idle bound -- pinned separately below (test_nf_a_silent_holder_is_reaped_and_read_not_recomputed)
    with envs.FLLIPER_PDFLIP_L3FILL_JOIN_WAIT_MS.override(100), \
            envs.FLLIPER_PDFLIP_L3FILL_JOIN_LIVENESS.override(False), \
            envs.FLLIPER_PDFLIP_L3FILL_STALE_CLAIM_S.override(30.0):
        _on_prefetch_thread(1, lambda: HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True),
                            box, "out").join(15)
    out = box["out"]
    try:
        assert all(s is not None for s in out[:7])
        assert all(s is None for s in out[7:])
        assert arena.find_slots([stems[7]]) == [(s7, 1)]  # still CLAIMED, still the other's
        assert arena.claim_info([s7])[0][5] == 1          # one open writer: the other's (the join went)
        assert [st for _s, st in arena.find_slots(stems[8:])] == [2] * (N - 8)
    finally:
        release.set()
    assert pc.recv() == [1]                           # the other rank completes its own claim
    proc.join(15)
    assert arena.stats()["claimed"] == 0


def test_nf_a_silent_holder_is_reaped_and_read_not_recomputed(tmp_path):
    """NF STALE-CLAIM-REAP under the liveness wait (the h-series fix): a holder that keeps its claim on page 7 WITHOUT
    any progress for the idle bound is quarantined (generation kept) and this rank reads the page itself -- the full
    prefix, nothing recomputed; the silent holder's late completion is refused (never written over the new owner)."""
    be, stems = _store(tmp_path)
    path = str(tmp_path / "kv.bin")
    arena = ShmArena(path, TOTAL, 64)
    pc, cc = _CTX.Pipe()
    release = _CTX.Event()
    proc = _CTX.Process(target=_other_rank_holds_7, args=(path, stems, cc, release))
    proc.start()
    s7, _st7, _g7 = pc.recv()
    box = {}
    with envs.FLLIPER_PDFLIP_L3FILL_JOIN_WAIT_MS.override(100), envs.FLLIPER_PDFLIP_L3FILL_JOIN_LIVENESS.override(True):
        _on_prefetch_thread(1, lambda: HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True),
                            box, "out").join(15)
    release.set()
    late = pc.recv()
    proc.join(15)
    out = box["out"]
    assert all(s is not None for s in out), [i for i, s in enumerate(out) if s is None]
    for i, s in enumerate(out):
        assert bytes(arena.slot_view(s, TOTAL)[:4]) == bytes([i + 1]) * 4
    assert late != [1]                                # the silent holder's completion does not land on the page


def test_a_fresh_claim_given_back_unread_is_freed(tmp_path):
    """Sole claimant, read failed: the slot goes back to the free list as
    before (no leaked CLAIMED slot)."""
    be, stems = _store(tmp_path, short={3})
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 64)
    out = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True)
    assert out[3] is None
    assert arena.find_slots([stems[3]]) == [(-1, 0)]
    assert arena.stats()["claimed"] == 0
