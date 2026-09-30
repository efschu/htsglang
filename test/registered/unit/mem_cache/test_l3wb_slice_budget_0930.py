"""L3WB-SLICE (30.09.): one L3 write-behind pass never blocks a rank for more than its slice budget.

Metal z30y10 (dcdb9ab8f9, D TP0): the P->D flip's wake was clean at 20:03:51; at 20:03:52 the
write-behind gate opened (``gate=open (was leg)``) and the next pass logged at 20:04:10

    L3-REUSE WRITE-BEHIND pass=16 pages=8 bytes=262144 ms=17599.6 cpu_ms=17574.6 arenas=2
        complete=4092 new=4092 on_disk=4084 ...

-- 17.6 s of CPU for 8 pages, and in exactly that window TP0 logged nothing from any thread and
answered no /get_server_info (front 20:04:06: ``WEG2 STOP W3 Weg2DrainWitnessUnreachable``).

``new == complete``: the baseline of what is already secured (``arena._l3wb_sec``) is the
PROCESS's; the 4084 pages P had already put on L3 are new to D TP0 until D has stat'ed them once, so
the first pass after the gate opens verifies the whole L2 -- in one uninterrupted burst. The pass
had no budget in time, only in bytes (and 8 pages were far under it).

Pinned here, on the real C arena, the real L3 index and evictor, a slow per-stem verification
(the metal cost form: the pass time grows with ``new``, not with bytes) on a VIRTUAL clock so the
budget edge is exact:

* every pass ends at the first slice edge at or past the budget -- not later (the burst is
  bounded), not earlier (the throughput is kept);
* every pass makes progress, even when one slice alone exceeds the budget;
* the continuation passes together verify / write EVERY page -- stretched, nothing dropped;
* a head of pages that never secures does not starve the tail (per-arena cursor);
* the thread continues a cut pass after the yield, and a finished cycle waits the rest of its tick.
"""
from __future__ import annotations

import os
import shutil
import sys
import time as _real_time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache import hicache_storage as HS  # noqa: E402
from sglang.srt.mem_cache.canonical_page_store import CanonicalExtentWindow  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 64
SFX = "_NF_898fe1bf"
# binary-exact virtual times, so the budget EDGE is hit exactly (5 stems = half the budget)
SLICE_S = 10 / 1024      # the budget under test (virtual seconds)
STEM_S = 1 / 1024        # the per-stem verification cost (virtual): the metal's cost form


class _KVPage:
    pass


class _Clock:
    """``time`` for hicache_storage with a virtual perf_counter/thread_time."""

    def __init__(self):
        self.t = 1024.0

    def perf_counter(self):
        return self.t

    def thread_time(self):
        return self.t

    def __getattr__(self, name):
        return getattr(_real_time, name)


def _backend(tmp_path, monkeypatch, n):
    from sglang.srt.mem_cache.storage.file.l3_index import L3Index
    from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

    monkeypatch.setenv("SGLANG_WEG2_L3_PERSIST", "1")
    root = tmp_path / "store"
    root.mkdir(parents=True, exist_ok=True)
    (root / "L3_IDENTITY.json").write_text("{}")
    be = object.__new__(HiCacheFile)
    be.file_path = str(root)
    be._known_shards = set()
    be._legacy_flat = False
    be._key_geom = {"is_mla_model": False}
    be.metadata_cache = None
    be.dcp_owner_mode = False
    be.canonical_kv_page = _KVPage()
    be._canonical_kv_extents = CanonicalExtentWindow(TOTAL, ((0, TOTAL),))
    be.canonical_qsa_page = None
    be.canonical_mamba_blob = None
    be.canonical_draft_page = None
    be.kv_config_suffix = SFX
    be._kv_config_suffix_is_group_wide = True
    be.config_suffix = SFX + "_0_1"
    be._config_suffix_is_group_wide = False
    be._evictor = LRUFileEvictor(
        str(root), SFX, tp_rank=0, writes_shared_keys=False,
        path_for_stem=be._existing_path, iter_existing=be._iter_existing_files,
    )
    idx = L3Index(str(tmp_path / "l3idx.bin"), cap=1 << 14)
    be._l3idx = idx
    be._l3idx_tried = True
    be._evictor.l3_index = idx
    adir = tmp_path / "shm"
    adir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(adir))
    arena = ShmArena(str(adir / f"arena-{TOTAL}.bin"), TOTAL, n + 16)
    be._arenas = {TOTAL: arena}
    pay = torch.zeros(TOTAL, dtype=torch.uint8)
    stems = [be._get_suffixed_key(f"{i:05d}" + "cd" * 30) for i in range(n)]
    for s in stems:
        assert arena.write([s], [TOTAL], [((0, TOTAL),)], [pay.data_ptr()]) == [1]
    return be, arena, stems


def _slow_stat(be, clock, monkeypatch):
    """The metal cost form: the verification of the new stems costs per STEM."""
    real = be._stat_stems

    def slow(stems):
        clock.t += STEM_S * len(stems)
        return real(stems)

    monkeypatch.setattr(be, "_stat_stems", slow)


def _no_quiet():
    return None


def _drive(be, max_passes=10000):
    """What the thread does: continue a cut pass until the cycle is over."""
    passes = []
    cont = False
    for _ in range(max_passes):
        tot = be.l3_write_behind_pass(budget_bytes=1 << 40, quiet=_no_quiet, cont=cont,
                                      slice_s=SLICE_S)
        passes.append(tot)
        cont = bool(tot.get("sliced"))
        if not cont:
            return passes
    raise AssertionError("the cycle never ended: %d passes" % max_passes)


def _assert_bounded(passes):
    for p in passes[:-1]:
        assert p["sliced"], p
        # (a) never later than the first slice edge at or past the budget: the last slice
        #     STARTED before it (a slice starting exactly at the budget is one too many) ...
        assert p["last_slice_at_ms"] < SLICE_S * 1e3, p
        # (b) ... and never earlier: the budget is used, the throughput kept
        assert p["ms"] >= SLICE_S * 1e3, p
    assert not passes[-1]["sliced"]


def test_the_first_pass_after_a_flip_is_sliced_and_verifies_everything(tmp_path, monkeypatch):
    """The z30y10 shape: every page COMPLETE and already on L3 (the other group wrote it), this
    process's baseline empty -> new == complete. RED before L3WB-SLICE: one pass, 1536 stems x 1 ms."""
    n = 1536
    be, arena, _ = _backend(tmp_path, monkeypatch, n)
    first = be.l3_write_behind_pass(budget_bytes=1 << 40, quiet=_no_quiet, slice_s=0)
    assert first["written"] == n                       # the other group's writes, all on disk
    arena._l3wb_sec = None                             # this process never saw them
    clock = _Clock()
    monkeypatch.setattr(HS, "time", clock)
    _slow_stat(be, clock, monkeypatch)
    passes = _drive(be)
    assert len(passes) > 1
    _assert_bounded(passes)
    assert sum(p["on_disk"] for p in passes) == n      # every page verified, none dropped
    assert sum(p["written"] for p in passes) == 0
    assert sum(p["deferred"] for p in passes[:-1]) > 0
    again = be.l3_write_behind_pass(budget_bytes=1 << 40, quiet=_no_quiet, slice_s=SLICE_S)
    assert again["new"] == 0                           # the baseline holds all of them now


def test_writes_are_stretched_not_lost(tmp_path, monkeypatch):
    n = 700
    be, arena, stems = _backend(tmp_path, monkeypatch, n)
    clock = _Clock()
    monkeypatch.setattr(HS, "time", clock)
    _slow_stat(be, clock, monkeypatch)
    passes = _drive(be)
    _assert_bounded(passes)
    assert sum(p["written"] for p in passes) == n
    assert all(be._stem_exists(s) for s in stems)


def test_every_pass_makes_progress_even_when_one_slice_exceeds_the_budget(tmp_path, monkeypatch):
    """Per-stem cost 5x the whole budget: one slice per pass, but never zero."""
    n = 64
    be, arena, _ = _backend(tmp_path, monkeypatch, n)
    clock = _Clock()
    monkeypatch.setattr(HS, "time", clock)
    monkeypatch.setattr(sys.modules[__name__], "STEM_S", 5 * SLICE_S)
    _slow_stat(be, clock, monkeypatch)
    passes = _drive(be, max_passes=200)
    assert all(p["slices"] == 1 for p in passes), [p["slices"] for p in passes]
    assert all(p["written"] > 0 for p in passes)
    assert sum(p["written"] for p in passes) == n
    assert all(p["slice_max_stems"] == 4 for p in passes[1:])   # the adaptive floor


def test_a_census_longer_than_the_budget_still_makes_progress(tmp_path, monkeypatch):
    """The census of a big arena alone can use the budget up (720896 slots on D): the pass
    still runs its first slice -- a budget check before any slice would cut every pass at
    zero work, forever."""
    n = 40
    be, arena, _ = _backend(tmp_path, monkeypatch, n)
    clock = _Clock()
    monkeypatch.setattr(HS, "time", clock)
    _slow_stat(be, clock, monkeypatch)
    real = arena.complete_census

    def slow_census():
        clock.t += 2 * SLICE_S
        return real()

    monkeypatch.setattr(arena, "complete_census", slow_census)
    passes = _drive(be, max_passes=100)
    assert all(p["slices"] >= 1 for p in passes)
    assert sum(p["written"] for p in passes) == n


def test_a_head_that_never_secures_does_not_starve_the_tail(tmp_path, monkeypatch):
    """The first 150 slots fail their write every time (status 4: stay new). Without the cursor,
    every continuation would restart at that head and spend its whole budget there."""
    from sglang.srt.mem_cache.storage.file import pageio

    n = 600
    be, arena, stems = _backend(tmp_path, monkeypatch, n)
    head = set(stems[:150])
    real_load = pageio.load

    class _Failing:
        def __init__(self, pio):
            self._pio = pio

        def __getattr__(self, name):
            return getattr(self._pio, name)

        def write_pages(self, finals, totals, extents, ptrs, fsync):
            keep = [k for k, f in enumerate(finals) if os.path.basename(f)[:-4] not in head]
            got = self._pio.write_pages([finals[k] for k in keep], [totals[k] for k in keep],
                                        [extents[k] for k in keep], [ptrs[k] for k in keep], fsync)
            out = [4] * len(finals)                    # an io error: nothing lands for the head
            for k, st in zip(keep, got):
                out[k] = st
            return out

    monkeypatch.setattr(pageio, "load", lambda: _Failing(real_load()))
    clock = _Clock()
    monkeypatch.setattr(HS, "time", clock)
    _slow_stat(be, clock, monkeypatch)
    passes = []
    cont = False
    for _ in range(60):                                # one cycle = the thread's continuation run
        tot = be.l3_write_behind_pass(budget_bytes=1 << 40, quiet=_no_quiet, cont=cont,
                                      slice_s=SLICE_S)
        passes.append(tot)
        cont = bool(tot.get("sliced"))
        if not cont:
            break
    assert not passes[-1]["sliced"], len(passes)
    written = sum(p["written"] for p in passes)
    assert written == n - len(head)
    assert all(be._stem_exists(s) for s in stems[150:])
    assert not any(be._stem_exists(s) for s in stems[:150])


def test_slice_off_is_the_pre_slice_form(tmp_path, monkeypatch):
    n = 300
    be, arena, _ = _backend(tmp_path, monkeypatch, n)
    clock = _Clock()
    monkeypatch.setattr(HS, "time", clock)
    _slow_stat(be, clock, monkeypatch)
    tot = be.l3_write_behind_pass(budget_bytes=1 << 40, quiet=_no_quiet, slice_s=0)
    assert not tot["sliced"] and tot["written"] == n and tot["deferred"] == 0


def test_the_thread_continues_a_cut_pass_after_the_yield(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L3_WRITE_BEHIND_YIELD_MS", "25")
    assert HiCacheFile._l3wb_next_wait({"sliced": True}, 2.0, 100.0, 100.3) == (True, 0.025)
    # a finished cycle waits the REST of its tick (it started at 100.0, now 100.3)
    cont, w = HiCacheFile._l3wb_next_wait({"sliced": False}, 2.0, 100.0, 100.3)
    assert cont is False and abs(w - 1.7) < 1e-9
    # a cycle longer than the tick: the next one after one yield, not a full tick later
    assert HiCacheFile._l3wb_next_wait({"sliced": False}, 2.0, 100.0, 103.0) == (False, 0.025)
    assert HiCacheFile._l3wb_next_wait(None, 2.0, 100.0, 100.0) == (False, 2.0)


def test_the_default_budget_is_25_ms():
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_L3_WRITE_BEHIND_SLICE_MS.get() == 25.0
    assert abs(HiCacheFile._l3wb_slice_s() - 0.025) < 1e-12


T2 = 128


def _put(arena, stem, total, fill=0):
    pay = torch.full((total,), fill & 0xFF, dtype=torch.uint8)
    assert arena.write([stem], [total], [((0, total),)], [pay.data_ptr()]) == [1]


def test_a_new_cycle_resets_every_arena_not_only_those_its_first_pass_reaches(tmp_path, monkeypatch):
    """NF 03ef1f6699 (port of its NF-only test to the 27B pass): a new cycle resets the cycle
    state of EVERY arena before the first one is visited. Two arenas; cycle 1 writes n pages
    in each. Cycle 2 brings m new pages in each and its FIRST pass is cut inside arena 1, so
    arena 2 is not reached by it. Arena 2 must still be visited later in cycle 2 -- in the
    82871045cd form an arena was reset only when the cycle's first pass reached it, arena 2
    kept cycle 1's ``done`` and its m new pages waited a whole cycle."""
    n, m = 200, 200
    be, _a0, _ = _backend(tmp_path, monkeypatch, 0)
    a1 = ShmArena(str(tmp_path / "shm" / f"arena-a1-{TOTAL}.bin"), TOTAL, n + m + 16)
    a2 = ShmArena(str(tmp_path / "shm" / f"arena-a2-{T2}.bin"), T2, n + m + 16)
    be._arenas = {TOTAL: a1, T2: a2}
    s1 = [be._get_suffixed_key(f"{i:05d}" + "a1" * 30) for i in range(n + m)]
    s2 = [be._get_suffixed_key(f"{i:05d}" + "b2" * 30) for i in range(n + m)]

    def fill(lo, hi):
        for i in range(lo, hi):
            _put(a1, s1[i], TOTAL, i)
            _put(a2, s2[i], T2, i)

    clock = _Clock()
    monkeypatch.setattr(HS, "time", clock)
    _slow_stat(be, clock, monkeypatch)
    fill(0, n)
    p1 = _drive(be)
    _assert_bounded(p1)
    assert all(be._stem_exists(s) for s in s1[:n] + s2[:n])
    fill(n, n + m)
    p2 = _drive(be)
    assert p2[0]["sliced"] and p2[0]["arenas"] == 1          # the first pass ends in arena 1
    _assert_bounded(p2)
    assert all(be._stem_exists(s) for s in s1[n:])
    assert all(be._stem_exists(s) for s in s2[n:])           # RED in the 82871045cd form
    assert sum(p["written"] for p in p2) == 2 * m
