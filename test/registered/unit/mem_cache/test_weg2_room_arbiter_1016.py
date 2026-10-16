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

    def __init__(self, arena, pp_rank=0, pp_size=1, held=()):
        self.arena = arena
        self.pp_rank, self.pp_size = pp_rank, pp_size
        self.ongoing_prefetch = {}         # the reads in flight on this rank (a yielded read stays until its reap)
        self.held = list(held)             # spillable tree copies (the 11:12:06 host-leaf spill), empty = live holders
        setattr(self, SF.FLOOR_LOCAL_PP_ATTR, True)

    def _weg2_direct_pool(self):
        return types.SimpleNamespace(arena=self.arena)

    def _w3_arena_spill(self, pool, need_pages, claimer=None, released_out=None):
        take = self.held[: max(int(need_pages), 4)]
        del self.held[: len(take)]
        if take:
            self.arena.ref_slots(take, -1)
        return len(take)


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


def _run_pair(be, arena, tree, young, old, third=None, reap_after=0.4):
    """The younger read lands what the free slots allow and waits; the older one arrives and waits. Production
    configuration: ROOM-ARBITER and ROOM-VERDICTS both on. A yielded younger read stays in ``ongoing_prefetch`` until
    its reap, which gives its landed references back ``reap_after`` s later (a scheduler round on every rank)."""
    with _Scheduler(tree):
        tree.ongoing_prefetch["weg2-7-12"] = object()
        fy = _Fill(be, arena, "weg2-7-12", young, "hicache-prefetch-io-1").start()
        time.sleep(0.4)
        fo = _Fill(be, arena, "weg2-7-11", old, "hicache-prefetch-io-2").start()
        out_y = fy.join()
        landed_y = [s for s in out_y if s is not None]
        ft = None
        if third:
            time.sleep(0.15)
            ft = _Fill(be, arena, "weg2-7-13", third, "hicache-prefetch-io-3").start()
        time.sleep(reap_after)
        arena.ref_slots(landed_y, -1)        # the reap's #257 cut of the younger short read
        tree.ongoing_prefetch.pop("weg2-7-12", None)
        if ft is not None:
            ft.join()
        out_o = fo.join()
    return fy, fo, landed_y, sum(1 for s in out_o if s is not None)


def test_11_27_the_younger_fill_yields_and_the_oldest_lands(tmp_path, caplog):
    """Both switches on (the production default). The younger read holds 8 landed references and gains nothing:
    proven hold-and-wait -> it yields; its give-back is room on its way, so the older read is NOT futile meanwhile
    (review A2) and lands everything once the reap gave the references back."""
    young = [f"y{i}" for i in range(12)]
    old = [f"o{i}" for i in range(8)]
    be, arena = _setup(tmp_path, young + old, slots=32)
    _pin_pages(arena, 24)                                   # live holders nothing can spill: 8 slots free
    tree = _Tree(arena)
    tree._w3_spill_quiet_until = time.monotonic() + 0.6    # the younger's own wait is not futile before the older comes
    # the younger's proof of hold-and-wait is a fresh answer round after the backoff -- a repost away (300 ms here)
    with envs.SGLANG_WEG2_READ_ROOM_REPOST_MS.override(300), caplog.at_level("INFO"):
        fy, fo, landed_y, landed_o = _run_pair(be, arena, tree, young, old)
    assert len(landed_y) == 8
    assert landed_o == len(old), landed_o
    assert fy.t1 - fo.t0 < 1.5, "the younger fill held its pages after the older one began to wait"
    msgs = [r.getMessage() for r in caplog.records]
    assert any("ROOM-ARBITER YIELD" in m for m in msgs)


def test_review_P7_a_third_fill_posting_meanwhile_does_not_make_the_oldest_futile(tmp_path):
    young = [f"y{i}" for i in range(12)]
    old = [f"o{i}" for i in range(8)]
    third = [f"t{i}" for i in range(4)]
    be, arena = _setup(tmp_path, young + old + third, slots=32)
    _pin_pages(arena, 24)
    tree = _Tree(arena)
    tree._w3_spill_quiet_until = time.monotonic() + 0.45
    fy, fo, landed_y, landed_o = _run_pair(be, arena, tree, young, old, third=third, reap_after=0.25)
    assert landed_o == len(old), "the oldest's wait ended futile while the yielded references were on their way"


def test_review_A3_pairs_of_11_12_12_both_land_when_the_room_comes(tmp_path, caplog):
    """11:12:12 / 11:13:22 (boot 103428): concurrent waits that ENDED WELL (still_full=0), the room came from the
    trees' spill. The younger has progress and must not be sent away."""
    young = [f"y{i}" for i in range(12)]
    old = [f"o{i}" for i in range(8)]
    be, arena = _setup(tmp_path, young + old, slots=32)
    spillable = _pin_pages(arena, 24)
    tree = _Tree(arena, held=spillable)
    with caplog.at_level("INFO"):
        fy, fo, landed_y, landed_o = _run_pair(be, arena, tree, young, old, reap_after=0.0)
    assert len(landed_y) == len(young), len(landed_y)
    assert landed_o == len(old)
    assert not any("ROOM-ARBITER YIELD" in r.getMessage() for r in caplog.records)


def test_review_A3_trickling_room_keeps_the_younger_fill_waiting(tmp_path, caplog):
    """11:12:06 shape: the trees' spill hands room back in pieces (a few pages per serve, a serve per post): the
    younger read gains slots between long rounds without progress -- room comes, so it is no hold-and-wait."""
    young = [f"y{i}" for i in range(16)]
    old = [f"o{i}" for i in range(4)]
    be, arena = _setup(tmp_path, young + old, slots=24)
    spillable = _pin_pages(arena, 20)                       # 4 slots free; the rest comes 4 pages per serve
    tree = _Tree(arena, held=spillable)

    def spill_four(pool, need_pages, claimer=None, released_out=None):
        take = tree.held[:4]
        del tree.held[:4]
        if take:
            arena.ref_slots(take, -1)
        return len(take)

    tree._w3_arena_spill = spill_four
    with envs.SGLANG_WEG2_READ_ROOM_REPOST_MS.override(300), caplog.at_level("INFO"):
        fy, fo, landed_y, landed_o = _run_pair(be, arena, tree, young, old, reap_after=0.0)
    assert not any("ROOM-ARBITER YIELD" in r.getMessage() for r in caplog.records)
    assert len(landed_y) == len(young), len(landed_y)


def test_review_A3_an_ended_owner_sends_nobody_away(tmp_path):
    """PP0's header still names an owner whose fills ended (until PP0's next round): no fill yields to it."""
    be, arena = _setup(tmp_path, ["a"], slots=8)
    tree = _Tree(arena)
    RR.tick(tree)
    since = ARB.now_ms()
    time.sleep(0.01)
    slot = ARB.register(arena, "weg2-4-10")
    ARB.register(arena, "weg2-4-11")
    assert ARB.decide(arena)[0] == ARB.rid_hash("weg2-4-10")
    assert not ARB.may_wait(arena, "weg2-4-11", since_ms=since)
    ARB.unregister(arena, slot, "weg2-4-10")                 # the owner's fill ended; PP0 has not decided again
    assert ARB.may_wait(arena, "weg2-4-11", since_ms=since)


def test_review_A3_a_younger_fill_that_holds_nothing_is_no_hold_and_wait(tmp_path, caplog):
    young = [f"y{i}" for i in range(4)]
    old = [f"o{i}" for i in range(4)]
    be, arena = _setup(tmp_path, young + old, slots=16)
    _pin_pages(arena, 16)                                   # nothing free: the younger lands nothing
    tree = _Tree(arena)
    tree._w3_spill_quiet_until = time.monotonic() + 5.0    # no futility either: only the arbiter could end it
    with envs.SGLANG_WEG2_READ_ROOM_WAIT_MS.override(800), caplog.at_level("INFO"):
        fy, fo, landed_y, landed_o = _run_pair(be, arena, tree, young, old, reap_after=0.0)
    assert not landed_y
    assert not any("ROOM-ARBITER YIELD" in r.getMessage() for r in caplog.records)


def test_review_A4_a_terminated_operation_ends_the_wait_and_leaves_the_arbitration(tmp_path):
    stems = [f"r{i}" for i in range(4)]
    be, arena = _setup(tmp_path, stems, slots=8)
    _pin_pages(arena, 8)
    tree = _Tree(arena)
    tree._w3_spill_quiet_until = time.monotonic() + 5.0
    op = types.SimpleNamespace(flag=False)
    op.is_terminated = lambda: op.flag
    box = {}

    def run():
        with ARB.fill_scope("weg2-2-20", op):
            t0 = time.monotonic()
            box["out"] = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True, take_ref=True)
            box["s"] = time.monotonic() - t0

    with _Scheduler(tree):
        th = threading.Thread(target=run, name="hicache-prefetch-io-0")
        th.start()
        time.sleep(0.3)
        assert any(h == ARB.rid_hash("weg2-2-20") for h, _k, _p, _pr in ARB.registrations(arena))
        op.flag = True                                       # the client aborted: the read is terminated
        th.join(10)
    assert box["s"] < 1.0, box["s"]
    assert not any(h == ARB.rid_hash("weg2-2-20") for h, _k, _p, _pr in ARB.registrations(arena))


def test_review_P5_a_newcomer_registered_during_pp0s_read_is_not_sent_away(tmp_path, monkeypatch):
    be, arena = _setup(tmp_path, ["a"], slots=8)
    tree = _Tree(arena, pp_rank=0, pp_size=3)
    RR.tick(tree)
    ARB.register(arena, "weg2-3-50")
    real = ARB.registrations
    box = {}

    def regs_then_newcomer(a):
        out = real(a)
        box["slot"] = ARB.register(arena, "weg2-3-40")       # older newcomer registers right after PP0's read
        box["since"] = ARB.now_ms()
        time.sleep(0.003)
        return out

    monkeypatch.setattr(ARB, "registrations", regs_then_newcomer)
    ARB._A.pp_rank = 0
    assert ARB.decide(arena)[0] == ARB.rid_hash("weg2-3-50")
    monkeypatch.setattr(ARB, "registrations", real)
    assert ARB.may_wait(arena, "weg2-3-40", since_ms=box["since"])


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


def test_prio_1016_priority_descending_then_arrival(tmp_path):
    """PRIO = ALTER (prio_age.key): a younger request of priority 1 is OLDER than every request of priority 0; inside
    one priority the arrival number decides."""
    be, arena = _setup(tmp_path, ["a"], slots=8)
    tree = _Tree(arena)
    RR.tick(tree)
    ARB.note_priority("weg2-5-90", 0)
    ARB.note_priority("weg2-5-95", 1)
    ARB.note_priority("weg2-5-97", 1)
    for rid in ("weg2-5-90", "weg2-5-95", "weg2-5-97"):
        ARB.register(arena, rid)
    assert ARB.decide(arena) == (ARB.rid_hash("weg2-5-95"), 96)
    assert sorted(pr for _h, _k, _p, pr in ARB.registrations(arena)) == [0, 1, 1], "the slot carries the priority"


def test_prio_1016_the_intake_notes_the_requests_priority():
    """The seam: ``Scheduler._prefetch_kvcache`` names the request's priority before it registers the read."""
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._prefetch_kvcache)
    assert "_arbiter.note_priority(req.rid, _pa.lane_of_req(req))" in src
