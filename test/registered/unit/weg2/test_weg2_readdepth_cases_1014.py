"""READ-DEPTH 1014: the five rank-asymmetric L3 reads of NF h16h 853534a9ff
(boot _211215, P, 22:30-23:00Z), one test per case, against the mechanism the
P.log names -- and which fix closes it.

| rid | metal | mechanism | closed by |
|---|---|---|---|
| weg2-51-579 | PP0 ``LATE-COMPLETE REFUSED statuses=[3]`` 22:30:02, landed 300 of 1556 (PP1 1556, PP2 694) | the arena reap race | READ-DEPTH arena.c (this branch) |
| weg2-60-639 | PP2 ``#248e ORDERED-EVICT site=l3fill victims=[weg2-60-639 ... pages=159, lowest_page=304]``, PP0+PP2 ``JOIN-WAIT still_claimed=159 waited_ms=5004``, landed 304 | the fill evicts the read's OWN kept pages; the joiners poll vanished stems 5 s | SELF-EVICT e3c9a228ec (+ JOIN-IDLE 300384f533) |
| weg2-68-683 | PP1 ``LATE-COMPLETE ... statuses=[3] total=770`` 22:47:16; PP0+PP1 ``JOIN-WAIT still_claimed=25`` 5 s, ``#1436 ARENA-GET MISS find=(slot=-1)``, landed 0 | reap race (PP1's claim recycled, its stem vanished) + the joiner's 5-s poll | READ-DEPTH arena.c (root) + JOIN-IDLE 300384f533 (the joiner) |
| weg2-72-690 | PP0 filled 589, landed 557 of 636, no L3-FILL line; PP1 then had to fill exactly 79 = 636-557 | a landed page evicted by a peer clock before the walk's reference | REF-AT-LAND 300384f533 |
| weg2-78-784 | PP0 ``LATE-COMPLETE n=2 total=271`` (263 slots in one fill), PP2 too, both landed 761 of 1148 | the arena reap race | READ-DEPTH arena.c (this branch) |
| weg2-144-1416 (6) | PP0 ``LATE-COMPLETE n=5`` / PP2 ``n=8`` 00:19:14, landed 1061 / 1047 of 1432 | the arena reap race (+ the HELD settle re-read is not wired on P) | READ-DEPTH arena.c + P-HELD-REREAD 4c83b46672 |
| weg2-182-1661 (7, h48) | D TP2 ``LATE-COMPLETE n=14``, read 810 of 1417; P 3rd pass PP0 59 | reap race on D and P (one shared arena) + D: tail consumed at the short reap, stale witness | READ-DEPTH arena.c + TAIL-CONSUME-ON-FULL (this branch, test_weg2_tail_consume_on_full_h48_1014.py) |
| weg2-198-1695 (8) | PP0 92, PP1/PP2 76 of 483; D ``#1427 ARENA-DROP`` stage i x3 the same second, no status 3 | D's claim-room clock takes P's landed, unreferenced pages; the joiners poll vanished stems | REF-AT-LAND + JOIN-IDLE 300384f533 (the 72-690 test) |

PP0's short read is the told for the group (the followers re-read to it,
``W27-UNIFORM FOLLOWER TREE SHORT ... re-reading``), so every case cost the
group (store hit - PP0's anchor). P-HELD-REREAD 4c83b46672 (PP0 re-reads a
short read the store holds, before its told) is the generic net under all
five; its own tests are ``test_weg2_p_held_reread_1014.py``.

The port tests skip where the port is absent (this branch is cut from
b76e5fce79, the ports live on desk/nf-h18port-1014 and are integrated on
desk/nf-h16m-int-1014); the report records the runs on a merged tree."""
from __future__ import annotations

import os
import shutil
import threading
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache import hicache_storage as hs  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096
_HAS_SELF_EVICT = hasattr(hs, "fill_protect_scope")
_HAS_REF_AT_LAND = "take_ref" in HiCacheFile.arena_fill_from_disk.__code__.co_varnames
_HAS_JOIN_IDLE = hasattr(hs, "fill_join_idle_ms") or "SGLANG_WEG2_L3FILL_JOIN_IDLE_MS" in open(hs.__file__).read()


def _store(root, stems, fill=0x30):
    os.makedirs(root, exist_ok=True)
    for i, s in enumerate(stems):
        with open(os.path.join(root, s + ".bin"), "wb") as f:
            f.write(bytes([(fill + i) % 256]) * TOTAL)
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    be._existing_path = _path
    be._stat_stems = lambda ss: {s: os.path.getsize(_path(s)) for s in ss if os.path.exists(_path(s))}
    be._arena_evict_to_disk = lambda arena, want, need=None: 0
    return be


def _full_arena(path, n):
    arena = ShmArena(path, TOTAL, n)
    old = [f"old{i}" for i in range(n)]
    cl = arena.claim_slots(old, [TOTAL] * n)
    assert arena.complete_slots([c[0] for c in cl], [c[2] for c in cl], [(0, TOTAL)]) == [1] * n
    return arena


def _on_io_thread(fn):
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("r", fn()), name="hicache-prefetch-io-0")
    th.start()
    th.join(30)
    return box["r"]


def _reap_race_fill(tmp_path, monkeypatch, n_pages):
    """A live evictor (another stage) holds ``n_pages`` candidates mid copy;
    a third stage's eviction tail reaps (``arena.reap_stale()`` -- the
    production call); this stage's fill claims, reads, and the evictor's free
    lands before the completion."""
    stems = [f"p{i}" for i in range(n_pages)]
    be = _store(str(tmp_path / "store"), stems)
    arena = _full_arena(str(tmp_path / "kv.bin"), n_pages * 2)
    cands = arena.evict_candidates(n_pages)
    arena.reap_stale()
    real = hs.l3_read_pages_parallel
    once = {"x": False}

    def evictor_done():
        if not once["x"]:
            once["x"] = True
            arena.free_slots([c[0] for c in cands], reason="evict_to_disk")
            return len(cands)
        return 0

    def read(pio, paths, total, ptrs):
        rc = real(pio, paths, total, ptrs)
        evictor_done()
        return rc

    monkeypatch.setattr(hs, "l3_read_pages_parallel", read)
    be._arena_evict_to_disk = lambda a, want, need=None: evictor_done()
    out = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True)
    return out


def test_case_51_579_pp0_lands_every_page_of_the_reap_race(tmp_path, monkeypatch):
    """PP0 22:30:02: 4 recycled slots (5116-5119) ended PP0's walk at 300 of
    1556. Red on b76e5fce79 and on e3c9a228ec (no port touches the reap)."""
    out = _reap_race_fill(tmp_path, monkeypatch, 4)
    assert all(s is not None for s in out), out


def test_case_78_784_a_263_slot_recycle_in_one_fill(tmp_path, monkeypatch):
    """PP0 22:59:50 ``LATE-COMPLETE n=2 total=271``: one fill lost 263 slots
    (the evictor's want >= 256). Same race, the metal's width."""
    out = _reap_race_fill(tmp_path, monkeypatch, 263)
    assert all(s is not None for s in out)


@pytest.mark.skipif(not _HAS_JOIN_IDLE, reason="JOIN-IDLE (300384f533) not on this tree")
def test_case_68_683_the_joiner_of_a_recycled_claim_reads_it_itself(tmp_path, monkeypatch):
    """PP1's claim of page 0 was recycled under it (its stem left the index);
    PP0 had JOINED it and polled the vanished stem for the 5-s JOIN-WAIT
    (``still_claimed=25 waited_ms=5004``, ``ARENA-GET MISS find=(slot=-1)``),
    landed 0. JOIN-IDLE: a vanished stem is claimed and read by the joiner."""
    monkeypatch.delenv("SGLANG_WEG2_L3FILL_JOIN_IDLE_MS", raising=False)
    stems = ["p0", "p1", "p2"]
    be = _store(str(tmp_path / "store"), stems)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    [(slot, status, gen)] = arena.claim_slots(["p0"], [TOTAL])     # PP1's claim
    assert status == 0

    def recycled_under_pp1():
        time.sleep(0.05)
        arena.free_slots([slot], reason="evict_to_disk")              # the reap race's free

    threading.Thread(target=recycled_under_pp1).start()
    t0 = time.perf_counter()
    kw = {"take_ref": True} if _HAS_REF_AT_LAND else {}
    takeover = getattr(envs, "SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER", None)
    ctx = takeover.override(False) if takeover is not None else _null()
    with ctx, envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(5000):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, stems, TOTAL, prefix=True, **kw))
    took = time.perf_counter() - t0
    assert all(s is not None for s in out), out
    assert took < 3.0, took                                            # metal: 5.0 s and landed 0


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.mark.skipif(not _HAS_REF_AT_LAND, reason="REF-AT-LAND (300384f533) not on this tree")
def test_case_72_690_a_landed_page_survives_a_peer_clock(tmp_path):
    """PP0 filled 589 pages and walked 557: a peer stage's clock (stage (i):
    unkept, unreferenced COMPLETE pages) took the freshly landed pages before
    PP0's reference. REF-AT-LAND: every page is referenced as it lands.
    Case 8 (weg2-198-1695) is the same mechanism across groups: D's
    ``#1427 ARENA-DROP stages=i:194/934/220`` (claim room of its park) took
    PP0's landed pages (92 of 483) and the stems PP1/PP2 had joined."""
    stems = [f"p{i}" for i in range(6)]
    be = _store(str(tmp_path / "store"), stems)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(
        be, arena, stems, TOTAL, prefix=True, take_ref=True))
    got = arena.evict_candidates(64)
    assert not {int(c[0]) for c in got} & {int(s) for s in out}
    assert [int(st) for _, st in arena.find_slots(stems)] == [2] * 6


@pytest.mark.skipif(not _HAS_SELF_EVICT, reason="SELF-EVICT (e3c9a228ec) not on this tree")
def test_case_60_639_the_fill_never_evicts_the_reads_own_kept_pages(tmp_path):
    """PP2 22:39:47 ``#248e ORDERED-EVICT site=l3fill need=497 took=159
    victims=[weg2-60-639:rank=3,pages=159,lowest_page=304]`` -- the fill of
    weg2-60-639 made its room from weg2-60-639's own kept pages 304.., the
    walks of PP0 and PP2 ended at 304. With the read's stems on the clock's
    keep list the room comes from another page."""
    stems = [f"o{i}" for i in range(4)] + ["l0"]                     # 4 own pages in L2 + 1 only in L3
    be = _store(str(tmp_path / "store"), ["l0"])
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 5)
    cl = arena.claim_slots(stems[:4] + ["foreign"], [TOTAL] * 5)
    assert arena.complete_slots([c[0] for c in cl], [c[2] for c in cl], [(0, TOTAL)]) == [1] * 5

    def real_clock(a, want, need=None):
        keep = hs.fill_protected_stems()
        got = a.evict_candidates(int(need or want), keep_stems=keep)
        a.free_slots([c[0] for c in got], reason="evict_to_disk")
        return len(got)

    be._arena_evict_to_disk = real_clock
    op = types.SimpleNamespace(hash_value=stems, _weg2_protect_stems=list(stems))
    with envs.SGLANG_WEG2_L3FILL_PROTECT_READ.override(True), hs.fill_protect_scope(be, op):
        out = HiCacheFile.arena_fill_from_disk(be, arena, ["l0"], TOTAL, prefix=True)
    assert out[0] is not None
    assert [int(st) for _, st in arena.find_slots(stems[:4])] == [2] * 4, "own pages stay in L2"
    assert arena.find_slots(["foreign"])[0][0] == -1, "the room came from the foreign page"


@pytest.mark.skipif(not _HAS_SELF_EVICT, reason="SELF-EVICT (e3c9a228ec) not on this tree")
def test_mutant_case_60_639_protect_off_evicts_the_reads_own_page(tmp_path):
    stems = [f"o{i}" for i in range(4)] + ["l0"]
    be = _store(str(tmp_path / "store"), ["l0"])
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 4)              # only own pages: the clock must take one
    cl = arena.claim_slots(stems[:4], [TOTAL] * 4)
    assert arena.complete_slots([c[0] for c in cl], [c[2] for c in cl], [(0, TOTAL)]) == [1] * 4

    def real_clock(a, want, need=None):
        keep = hs.fill_protected_stems()
        got = a.evict_candidates(int(need or want), keep_stems=keep)
        a.free_slots([c[0] for c in got], reason="evict_to_disk")
        return len(got)

    be._arena_evict_to_disk = real_clock
    op = types.SimpleNamespace(hash_value=stems, _weg2_protect_stems=list(stems))
    with envs.SGLANG_WEG2_L3FILL_PROTECT_READ.override(False), hs.fill_protect_scope(be, op):
        HiCacheFile.arena_fill_from_disk(be, arena, ["l0"], TOTAL, prefix=True)
    assert [int(st) for _, st in arena.find_slots(stems[:4])] != [2] * 4, "the metal: own page taken"
