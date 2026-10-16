# SPDX-License-Identifier: Apache-2.0
"""ROOM-ARBITER 1016 (F3): hold-and-wait between concurrent fills -- only the OLDEST request's fill waits for room.

MEASURED (boot ..._1f4b3e73d4_1016_103428, P, 11:27:18-11:27:43, 25 s without a prefill batch): arena pinned
6485/6485; several store reads per rank waited for room at once, each holding what it had landed (``ARENA-REF-HOLDERS
gap=1122/1109/407``), all ended together at the idle bound (``READ-ROOM SHORT waited_ms=10022..10456`` x6 at
11:27:25). Driven here through the REAL C arena and the REAL ``HiCacheFile.arena_fill_from_disk`` on two prefetch
io threads at once: the younger fill landed what the free slots allowed and waits for the rest; the older fill finds
nothing. RED on 1f4b3e73d4: the older fill gets room only when the younger one gave up at its idle bound. GREEN:
PP0's round announces the older request, the younger fill ends at once, its landed references go back (the #257 cut
at the reap -- played by the test), the older fill lands everything well inside the bound.
"""

import contextlib
import os
import shutil
import threading
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena
from sglang.srt.weg2 import pp_slot_fidelity as SF
from sglang.srt.weg2 import read_room as RR

try:
    from sglang.srt.weg2 import room_wait_arbiter as ARB
except ImportError:          # the base tree: no arbiter -- the scenario runs and goes red by behaviour
    ARB = None

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096
BOUND_MS = 3000


def _scope(rid):
    return ARB.fill_scope(rid) if ARB is not None else contextlib.nullcontext()


def _backend(root):
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    def _clock(arena, want, need=None):
        got = arena.evict_candidates(int(want))
        if got:
            arena.free_evicted([c[0] for c in got], reason="test_clock")
        return len(got)

    be._existing_path = _path
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}
    be._arena_evict_to_disk = _clock
    return be


class _Tree:
    """PP0's tree: nothing it holds is spillable (the live-holder case of 11:27, ``served_nothing_to_spill``)."""

    def __init__(self, arena, pp_rank=0, pp_size=1):
        self.arena = arena
        self.pp_rank, self.pp_size = pp_rank, pp_size
        setattr(self, SF.FLOOR_LOCAL_PP_ATTR, True)

    def _weg2_direct_pool(self):
        return types.SimpleNamespace(arena=self.arena)

    def _w3_arena_spill(self, pool, need_pages, claimer=None, released_out=None):
        return 0


class _Scheduler:
    def __init__(self, tree, period=0.02):
        self.tree, self.period, self.stop = tree, period, threading.Event()
        self.th = threading.Thread(target=self._run, name="scheduler")

    def _run(self):
        while not self.stop.is_set():
            RR.tick(self.tree)
            time.sleep(self.period)

    def __enter__(self):
        self.th.start()
        time.sleep(0.1)
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.th.join(5)


def _pin_pages(arena, n, tag="live"):
    got = arena.claim_slots([f"{tag}{i}" for i in range(n)], [TOTAL] * n)
    slots = [s for s, _st, _g in got]
    assert arena.complete_slots(slots, [g for _s, _st, g in got], [(0, TOTAL)]) == [1] * n
    arena.ref_slots(slots, +1)
    return slots


def _setup(tmp_path, stems, slots):
    root = tmp_path / "store"
    root.mkdir()
    for s in stems:
        (root / (s + ".bin")).write_bytes(b"\x33" * TOTAL)
    return _backend(str(root)), ShmArena(str(tmp_path / "kv.bin"), TOTAL, slots)


@pytest.fixture(autouse=True)
def _clean():
    RR.reset_for_tests()
    if ARB is not None:
        ARB.reset_for_tests()
    with envs.SGLANG_WEG2_L3FILL_ROOM_WAIT_MS.override(300), \
            envs.SGLANG_WEG2_READ_ROOM_WAIT_MS.override(BOUND_MS), \
            envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(False):
        yield
    RR.reset_for_tests()
    if ARB is not None:
        ARB.reset_for_tests()


class _Fill:
    """One store read on its own prefetch io thread, its request named like ``_arena_page_get`` names it."""

    def __init__(self, be, arena, rid, stems, name):
        self.box = {}
        self.t0 = self.t1 = None

        def run():
            self.t0 = time.monotonic()
            with _scope(rid):
                self.box["out"] = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True,
                                                                   take_ref=True)
            self.t1 = time.monotonic()

        self.th = threading.Thread(target=run, name=name)

    def start(self):
        self.th.start()
        return self

    def join(self, s=30):
        self.th.join(s)
        assert not self.th.is_alive()
        return self.box["out"]


def test_11_27_the_younger_fill_yields_and_the_oldest_lands_inside_the_bound(tmp_path):
    young = [f"y{i}" for i in range(12)]
    old = [f"o{i}" for i in range(8)]
    be, arena = _setup(tmp_path, young + old, slots=32)
    _pin_pages(arena, 24)                                   # live holders nothing can spill: 8 slots free
    tree = _Tree(arena)
    with _Scheduler(tree):
        fy = _Fill(be, arena, "weg2-7-12", young, "hicache-prefetch-io-1").start()
        time.sleep(0.4)                                     # the younger read landed 8 and waits for 4 more
        fo = _Fill(be, arena, "weg2-7-11", old, "hicache-prefetch-io-2").start()
        out_y = fy.join()
        # the reap's #257 cut of the younger short read: its landed references go back
        landed_y = [s for s in out_y if s is not None]
        assert len(landed_y) == 8, out_y
        arena.ref_slots(landed_y, -1)
        out_o = fo.join()
    landed_o = sum(1 for s in out_o if s is not None)
    o_wait = fo.t1 - fo.t0
    y_after_o = fy.t1 - fo.t0
    assert landed_o == len(old), (landed_o, o_wait)
    assert y_after_o < 1.5, f"the younger fill held its pages {y_after_o:.2f}s after the older one began to wait"
    assert o_wait < 2.0, f"the older fill waited {o_wait:.2f}s (bound {BOUND_MS} ms)"


def test_a_follower_acts_on_pp0s_announcement_only(tmp_path):
    """A follower never decides: without an announcement every fill waits as before; with PP0's announcement the
    named request waits and any other request's fill ends."""
    be, arena = _setup(tmp_path, ["a"], slots=8)
    tree = _Tree(arena, pp_rank=1, pp_size=3)
    RR.tick(tree)                                            # arms, resolves the arena
    assert RR.served(arena)
    ARB.register(arena, "weg2-3-40")
    ARB.register(arena, "weg2-3-41")
    RR._ST.poll_t = 0.0
    RR.tick(tree)                                            # a follower's round with waits registered
    assert ARB.decide(arena) is None and ARB.owner(arena) is None, "a follower never announces"
    assert ARB.may_wait(arena, "weg2-3-41") and ARB.may_wait(arena, "weg2-3-40"), "no announcement: both wait"
    since = ARB.now_ms()
    time.sleep(0.01)
    ARB._A.pp_rank = 0                                       # PP0's round (one process plays both)
    assert ARB.decide(arena) == (ARB.rid_hash("weg2-3-40"), 41)
    ARB._A.pp_rank = 1
    assert ARB.may_wait(arena, "weg2-3-40", since_ms=since)
    assert not ARB.may_wait(arena, "weg2-3-41", since_ms=since)
    # a newcomer that registered after the announcement is never sent away by it
    later = ARB.now_ms() + 1
    assert ARB.may_wait(arena, "weg2-3-39", since_ms=later)


def test_pp0_orders_by_the_fronts_arrival_number_not_by_registration():
    ARB._A.pp_rank = 0
    assert ARB.age_key("weg2-30-272") == 273 and ARB.age_key("weg2-31-268") == 269
    k = [ARB._order_key(ARB.rid_hash(r), ARB.age_key(r)) for r in ("weg2-30-272", "weg2-31-268", "free-form")]
    assert k[1] < k[0] < k[2], k


def test_nobody_waiting_clears_the_announcement(tmp_path):
    be, arena = _setup(tmp_path, ["a"], slots=8)
    tree = _Tree(arena)
    RR.tick(tree)
    slot = ARB.register(arena, "weg2-1-5")
    assert ARB.decide(arena) is not None and ARB.owner(arena) is not None
    ARB.unregister(arena, slot, "weg2-1-5")
    assert ARB.decide(arena) is None and ARB.owner(arena) is None
    assert ARB.may_wait(arena, "weg2-1-9")


def test_a_fill_without_a_named_request_or_off_the_served_arena_waits_as_before(tmp_path):
    be, arena = _setup(tmp_path, ["a"], slots=8)
    assert ARB.may_wait(arena, None)
    assert ARB.may_wait(arena, "weg2-1-1"), "READ-ROOM does not serve this arena: not arbitrated"


def test_the_controller_names_the_fills_request_on_the_io_thread():
    """The seam: ``_arena_page_get`` runs the fill inside ``fill_scope(operation.request_id)`` -- without it no
    production fill is arbitrated (driven through the real controller method, like test_l3_fast_read_0928)."""
    import numpy as np

    from sglang.srt.managers import cache_controller as cc

    class _PoolArena:
        def __init__(self):
            self.by_stem = {}

        def find_slots_np(self, stems):
            got = [self.by_stem.get(st, (-1, 0)) for st in stems]
            return (np.asarray([s for s, _ in got], dtype=np.int64), np.asarray([t for _, t in got], dtype=np.int8))

        def ref_slots_np(self, a, delta):
            return len(list(a))

        def ref_slots(self, slots, delta):
            return 1

    seen = []

    def fill(ar, stems, nbytes, prefix=False):
        seen.append(ARB.current_rid())
        return [None] * len(stems)

    arena = _PoolArena()
    pool = types.SimpleNamespace(arena_read=True, arena=arena, _page_bytes=TOTAL,
                                 ensure_bound=lambda be, role: True, resolve_rows=lambda hi, slots: None)
    ctl = types.SimpleNamespace(mem_pool_host=pool, storage_backend=types.SimpleNamespace(arena_fill_from_disk=fill),
                                page_size=1)
    op = types.SimpleNamespace(probe_pins=None, completed_tokens=0, n=0, request_id="weg2-9-77")
    op.increment = lambda k: setattr(op, "n", op.n + k)
    orig = cc.weg2_suffixed_stems
    cc.weg2_suffixed_stems = lambda be, hv: ["s%d" % i for i in range(len(hv))]
    try:
        cc.HiCacheController._arena_page_get(ctl, op, [0, 1, 2], None)
    finally:
        cc.weg2_suffixed_stems = orig
    assert seen and all(r == "weg2-9-77" for r in seen), seen
    assert ARB.current_rid() is None, "the scope ends with the fill"
