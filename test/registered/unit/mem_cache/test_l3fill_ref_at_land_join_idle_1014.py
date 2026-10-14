"""REF-AT-LAND / JOIN-IDLE (NF h18 fe799c4d6e, P 16:18:13-18, weg2-0-2;
weg2-10-25; h18 162636 weg2-6-67).

weg2-0-2: PP2 filled its 322 own claims and got the 405 PP0 had claimed
(``JOIN-WAIT completed=405``), landed 297 of 1440 pages -- the completed pages
carried no reference until the page walk, and another rank's arena clock
(``ARENA-EVICT want=405`` on every rank, weg2-0-4 reading 1644 pages) took
them. PP0/PP1 had joined PP2's 322 and polled stems that had vanished from the
arena for 5 s (``completed=0 still_claimed=322 moves=0``) and cut at 297 too:
74212 tokens recomputed.

Now (a) the fill references every slot as it lands (``take_ref``), (b) a
joined stem gone from the arena is claimed and read by the joiner, (c) a JOIN
wait without any move ends after ``SGLANG_WEG2_L3FILL_JOIN_IDLE_MS`` (1 s) and
the claims are broken and read -- never cut on a claim nobody fills; (d) the
JOIN-WAIT line names the holders of what is left.
"""

import logging
import os
import shutil
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.environ import envs
from sglang.srt.mem_cache import hicache_storage as hs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096


def _backend(root):
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    be._existing_path = _path
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}
    be._arena_evict_to_disk = lambda arena, want, need=None: 0
    return be


def _setup(tmp_path, stems=("p0", "p1", "p2"), slots=16):
    root = tmp_path / "store"
    root.mkdir()
    for s in stems:
        (root / (s + ".bin")).write_bytes(b"\x11" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, slots)
    return _backend(str(root)), arena


def _run_on(thread_name, fn):
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("r", fn()), name=thread_name)
    th.start()
    th.join(20)
    return box["r"]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_L3FILL_REF_AT_LAND", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_L3FILL_JOIN_IDLE_MS", raising=False)
    # h18port: the 'holder' of these tests is this very process (its claims carry our pid); with the SELF-JOIN
    # takeover (e937898ced) on it would be taken over at once -- off, so the holder stands for a FOREIGN rank
    with envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(False):
        yield


def _evict_all_unreferenced(arena):
    """Another rank's arena clock: every COMPLETE unreferenced slot goes."""
    got = arena.evict_candidates(64) if hasattr(arena, "evict_candidates") else []
    slots = [int(x[0]) if isinstance(x, (tuple, list)) else int(x) for x in got]
    if slots:
        arena.free_slots(slots, reason="test_peer_clock")
    return slots


def test_landed_pages_carry_a_reference_a_peer_clock_cannot_take(tmp_path):
    be, arena = _setup(tmp_path)
    out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
        be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True, take_ref=True))
    assert all(s is not None for s in out)
    taken = _evict_all_unreferenced(arena)
    assert not set(taken) & set(out), "a referenced page is never the peer clock's"
    st = arena.find_slots(["p0", "p1", "p2"])
    assert [int(x[0]) for x in st] == list(out) and all(int(x[1]) == 2 for x in st)
    arena.ref_slots(list(out), -1)


def test_without_take_ref_the_old_answer_has_no_reference(tmp_path):
    be, arena = _setup(tmp_path)
    out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
        be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True))
    assert type(out) is list
    taken = _evict_all_unreferenced(arena)
    assert set(taken) == set(out), "the h18 window: completed, unreferenced, taken"


def test_switch_off_keeps_the_old_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L3FILL_REF_AT_LAND", "0")
    be, arena = _setup(tmp_path)
    out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
        be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True, take_ref=True))
    assert type(out) is list


def test_a_slot_overwritten_or_cleared_gives_its_reference_back(tmp_path):
    be, arena = _setup(tmp_path)
    out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
        be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True, take_ref=True))
    slots = list(out)
    out[1] = None                       # past a hole: the reference goes back
    assert set(_evict_all_unreferenced(arena)) == {slots[1]}
    out.release_all()
    assert set(_evict_all_unreferenced(arena)) == {slots[0], slots[2]}


def test_a_joined_stem_that_vanished_is_claimed_and_read_by_the_joiner(tmp_path):
    """weg2-0-2: the joined holder filled the page, a peer clock evicted it --
    the joiner polled a stem that could never turn COMPLETE. Now: claimed and
    read by the joiner, well before the old 5-s bound."""
    be, arena = _setup(tmp_path)
    [(slot, status, gen)] = arena.claim_slots(["p1"], [TOTAL])   # the holder (another rank)
    assert status == 0

    def holder_fills_then_peer_evicts():
        time.sleep(0.05)
        assert arena.complete_slots([slot], [gen], [(0, TOTAL)]) == [1]
        arena.free_slots([slot], reason="test_peer_clock")       # evicted before the joiner looked

    threading.Thread(target=holder_fills_then_peer_evicts).start()
    t0 = time.perf_counter()
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(300):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True, take_ref=True))
    took = time.perf_counter() - t0
    assert all(s is not None for s in out), out
    assert took < 2.0, took


def test_a_claim_nobody_fills_is_broken_after_the_idle_bound_and_read(tmp_path, caplog):
    """weg2-10-25 / 6-67: ``still_claimed=6 waited_ms=5001 moves=0`` and the
    read was cut. Now 1 s without a move, the claim is broken, the joiner reads
    it; the JOIN-WAIT line names the holder."""
    be, arena = _setup(tmp_path)
    [(slot, status, gen)] = arena.claim_slots(["p1"], [TOTAL])   # holder that never fills
    assert status == 0
    t0 = time.perf_counter()
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(300), caplog.at_level(logging.INFO):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True, take_ref=True))
    took = time.perf_counter() - t0
    assert all(s is not None for s in out), out
    assert 0.9 < took < 3.0, took
    line = [r.getMessage() for r in caplog.records if "L3-FILL JOIN-WAIT" in r.getMessage()][-1]
    assert "still_claimed=1" in line and "holders=[" in line and "idle_bound_ms=1000" in line


def test_idle_bound_off_keeps_the_stale_age(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L3FILL_JOIN_IDLE_MS", "0")
    assert hs.join_idle_ms() == 0


def test_h16h_weg2_58_558_a_stalled_holder_costs_one_idle_bound_not_the_5s_join_wait(tmp_path, caplog):
    """h16h 853534a9ff P 19:21:06-11, weg2-58-558: PP2 read its 807 pages in 214 ms (l3fill 140 ms); PP0/PP1
    joined the same claims and sat out the 5-s bound (``JOIN-WAIT stems=217 completed=105 still_claimed=112
    waited_ms=5004`` / ``stems=161 completed=0 still_claimed=161 waited_ms=5002``), READ-STAGES l3fill_ms
    5109/5136. With the metal's JOIN bound (5000 ms): claims nobody moves are broken after the idle bound
    (1 s) and read by the joiner -- every page lands, well under the 5 s."""
    stems = tuple("q%d" % i for i in range(16))
    be, arena = _setup(tmp_path, stems=stems, slots=64)
    held = arena.claim_slots(list(stems[4:12]), [TOTAL] * 8)    # a foreign holder that never fills
    assert all(int(st) == 0 for _s, st, _g in held)
    t0 = time.perf_counter()
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(5000), \
            envs.SGLANG_WEG2_L3FILL_JOIN_LIVENESS.override(True), caplog.at_level(logging.INFO):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, list(stems), TOTAL, prefix=True, take_ref=True))
    took = time.perf_counter() - t0
    assert all(s is not None for s in out), out
    assert took < 3.0, f"{took:.2f}s -- the metal sat 5.0 s"
    line = [r.getMessage() for r in caplog.records if "L3-FILL JOIN-WAIT" in r.getMessage()][-1]
    assert "idle_bound_ms=1000" in line and "still_claimed=8" in line, line
