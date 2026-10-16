# SPDX-License-Identifier: Apache-2.0
"""ROOM-VERDICTS 1016 (F2): a room wait no P rank can answer ends on that fact, not at its idle bound.

MEASURED (boot ..._1f4b3e73d4_1016_103428, P): 11:27:25 / 11:27:35 ``READ-ROOM SHORT class=served_nothing_to_spill
spilled_pages=0 waited_ms=10019..10456`` on PP1/PP2 -- every serve of the wait answered ``released_pages=0`` with
``free=0 clock=0`` and ``W3-ARENA SPILL candidates=0``; 25 s without a prefill batch with 5 requests queued, and
twelve 10-13 s P gaps since 11:19Z at the 10 s bound. Driven through the REAL C arena and the REAL
``HiCacheFile.arena_fill_from_disk`` with a scheduler thread running the REAL ``read_room.tick``/``serve``.
RED on 1f4b3e73d4: the read waits the whole bound. GREEN: it ends at the first round every rank answered.
Positive control: a host copy held by a write in flight keeps the wait (it may become spillable).
"""

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
    from sglang.srt.weg2 import room_verdicts as RV
    from sglang.srt.weg2 import room_wait_arbiter as ARB
except ImportError:          # the base tree: the scenario runs and goes red by behaviour
    RV = ARB = None

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096
BOUND_MS = 3000


def _need_module():
    if RV is None:
        pytest.fail("no weg2/room_verdicts.py on this tree")


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


class _Comp:
    def __init__(self, host_value=None, host_lock_ref=0):
        self.host_value, self.host_lock_ref, self.value = host_value, host_lock_ref, None


class _Data:
    def __init__(self, comps):
        self.comps = comps

    def __getitem__(self, _k):
        return self.comps[0]

    def __iter__(self):
        return iter(self.comps)


class _Node:
    def __init__(self, nid, host_value=None):
        self.id = nid
        self.component_data = _Data([_Comp(host_value)])
        self.write_through_pending_id = None


class _Tree:
    """A P tree whose spill finds nothing (the live-holder case); ``in_flight`` puts one host-backed node under a
    write-through that has not acked (``ongoing_write_through``)."""

    def __init__(self, arena, pp_rank=0, pp_size=1, in_flight=False):
        self.arena = arena
        self.pp_rank, self.pp_size = pp_rank, pp_size
        self.root_node = _Node(0)
        self.nodes = [self.root_node, _Node(7, host_value=[1, 2])]
        self.ongoing_write_through = {7: object()} if in_flight else {}
        self.spills = 0
        setattr(self, SF.FLOOR_LOCAL_PP_ATTR, True)

    def _collect_all_nodes(self):
        return list(self.nodes)

    def _weg2_direct_pool(self):
        return types.SimpleNamespace(arena=self.arena)

    def _w3_arena_spill(self, pool, need_pages, claimer=None, released_out=None):
        self.spills += 1
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
        (root / (s + ".bin")).write_bytes(b"\x44" * TOTAL)
    return _backend(str(root)), ShmArena(str(tmp_path / "kv.bin"), TOTAL, slots)


@pytest.fixture(autouse=True)
def _clean():
    RR.reset_for_tests()
    for mod in (RV, ARB):
        if mod is not None:
            mod.reset_for_tests()
    with envs.SGLANG_WEG2_L3FILL_ROOM_WAIT_MS.override(300), \
            envs.SGLANG_WEG2_READ_ROOM_WAIT_MS.override(BOUND_MS), \
            envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(False):
        yield
    RR.reset_for_tests()
    for mod in (RV, ARB):
        if mod is not None:
            mod.reset_for_tests()


def _timed_read(be, arena, stems):
    box = {}

    def run():
        t0 = time.monotonic()
        box["out"] = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, prefix=True, take_ref=True)
        box["s"] = time.monotonic() - t0

    th = threading.Thread(target=run, name="hicache-prefetch-io-0")
    th.start()
    th.join(30)
    assert not th.is_alive()
    return box["out"], box["s"]


def test_11_27_nothing_to_spill_nothing_in_flight_ends_at_the_first_answered_round(tmp_path, caplog):
    stems = [f"r{i}" for i in range(8)]
    be, arena = _setup(tmp_path, stems, slots=16)
    _pin_pages(arena, 16)
    tree = _Tree(arena)
    with _Scheduler(tree), caplog.at_level("INFO"):
        out, waited = _timed_read(be, arena, stems)
    assert all(s is None for s in out)                      # no room existed: the cut is the same as before
    assert tree.spills >= 1, "the rank was asked"
    assert waited < 1.5, f"waited {waited:.2f}s for room no rank could give (bound {BOUND_MS} ms)"
    assert any("READ-ROOM FUTILE" in r.getMessage() for r in caplog.records)


def test_a_host_copy_under_a_write_in_flight_keeps_the_wait(tmp_path, caplog):
    """Positive control: the write-through may land and make the copy spillable -- no futility, the bound holds."""
    stems = [f"r{i}" for i in range(4)]
    be, arena = _setup(tmp_path, stems, slots=8)
    _pin_pages(arena, 8)
    tree = _Tree(arena, in_flight=True)
    with _Scheduler(tree), caplog.at_level("INFO"):
        out, waited = _timed_read(be, arena, stems)
    assert all(s is None for s in out)
    assert waited >= BOUND_MS / 1000.0 * 0.9, waited
    assert not any("READ-ROOM FUTILE" in r.getMessage() for r in caplog.records)


def test_futility_needs_every_p_rank_and_a_looking_spill(tmp_path, monkeypatch):
    _need_module()
    be, arena = _setup(tmp_path, ["a"], slots=8)
    monkeypatch.setattr(ARB, "_alive", lambda pid: True)
    since = ARB.now_ms()
    time.sleep(0.01)
    RV.note_verdict(arena, 0, 0, False, pid=101, refs=0)
    RV.note_verdict(arena, 0, 0, False, pid=102, refs=0)
    assert RV.futile(arena, since, pp_size=3) is None, "PP2 has not answered"
    RV.note_verdict(arena, 0, 0, True, pid=103, refs=0)
    assert RV.futile(arena, since, pp_size=3) is None, "PP2's spill was in its backoff: it did not look"
    RV.note_verdict(arena, 0, 0, False, pid=103, refs=0)
    assert RV.futile(arena, since, pp_size=3) is not None
    RV.note_verdict(arena, 5, 0, False, pid=102, refs=0)
    assert RV.futile(arena, since, pp_size=3) is None, "PP1 released pages: room may come"
    RV.note_verdict(arena, 0, 1, False, pid=102, refs=0)
    assert RV.futile(arena, since, pp_size=3) is None, "PP1 has I/O in flight"
    RV.note_verdict(arena, 0, 0, False, pid=102, refs=0)
    later = ARB.now_ms() + 5
    assert RV.futile(arena, later, pp_size=3) is None, "answers older than the last progress do not count"


def test_review_P4_a_serve_that_finds_room_never_ends_a_wait(tmp_path):
    """Review A1 (probe P4): serve with deficit<=0 (8 free slots, need 4) wrote released=0 -> futile."""
    _need_module()
    be, arena = _setup(tmp_path, ["a"], slots=8)
    tree = _Tree(arena)
    RR.tick(tree)
    since = ARB.now_ms()
    time.sleep(0.01)
    RR.serve(tree, tree._weg2_direct_pool(), arena, 4, read_key=1)
    assert RV.futile(arena, since, pp_size=1) is None


def test_review_A1_only_answers_before_the_last_claim_round_count(tmp_path, monkeypatch):
    _need_module()
    be, arena = _setup(tmp_path, ["a"], slots=8)
    monkeypatch.setattr(ARB, "_alive", lambda pid: True)
    since = ARB.now_ms()
    time.sleep(0.01)
    claim_round = ARB.now_ms()
    time.sleep(0.01)
    RV.note_verdict(arena, 0, 0, False, pid=101, refs=0)          # answered AFTER the claim round started
    assert RV.futile(arena, since, pp_size=1, until_ms=claim_round) is None, "no claim round ran after the answer"
    assert RV.futile(arena, since, pp_size=1, until_ms=ARB.now_ms()) is not None


def test_review_A2_a_pending_give_back_is_room_on_its_way(tmp_path, monkeypatch):
    _need_module()
    be, arena = _setup(tmp_path, ["a"], slots=8)
    monkeypatch.setattr(ARB, "_alive", lambda pid: True)
    since = ARB.now_ms()
    time.sleep(0.01)
    RV.note_verdict(arena, 0, 0, False, pid=101, refs=0)
    RV.note_giveback(arena, "weg2-1-9", 8)
    assert RV.futile(arena, since, pp_size=1) is None, "a yielded fill's references are on their way back"
    tree = types.SimpleNamespace(ongoing_prefetch={"weg2-1-9": object()})
    assert RV.reap_givebacks(arena, tree) == 0, "not reaped yet: the record stays"
    tree.ongoing_prefetch = {}
    assert RV.reap_givebacks(arena, tree) == 1 and not RV.givebacks(arena)
    assert RV.futile(arena, since, pp_size=1) is not None


def test_review_foreign_references_block_futility(tmp_path, monkeypatch):
    """A slot frees only when EVERY holder lets go -- group D's references are no P rank's to answer for."""
    _need_module()
    be, arena = _setup(tmp_path, ["a"], slots=8)
    _pin_pages(arena, 4)
    monkeypatch.setattr(ARB, "_alive", lambda pid: True)
    since = ARB.now_ms()
    time.sleep(0.01)
    RV.note_verdict(arena, 0, 0, False, pid=101, refs=1)           # P holds 1 of the 4 references
    assert RV.futile(arena, since, pp_size=1) is None
    RV.note_verdict(arena, 0, 0, False, pid=101, refs=4)
    assert RV.futile(arena, since, pp_size=1) is not None
    RV.note_verdict(arena, 0, 0, False, pid=101, refs=-1)          # no ledger: unknown
    assert RV.futile(arena, since, pp_size=1) is None


def test_review_A7_transient_is_an_o1_census_with_running_chunks_counted(tmp_path):
    _need_module()
    t = types.SimpleNamespace(ongoing_write_through={}, ongoing_backup={}, ongoing_load_back={},
                              component_protected_size_={"full": 0})
    assert RV.transient_blocked(t) == 0
    t.ongoing_backup = {1: object()}
    assert RV.transient_blocked(t) == 1
    t.ongoing_backup = {}
    t.component_protected_size_ = {"full": 4096}                    # a running chunk's request holds a device lock
    assert RV.transient_blocked(t) == 1
    t.component_protected_size_ = {"full": 0}
    t.ongoing_load_back = {7: object()}
    assert RV.transient_blocked(t) == 1
    # the cost the review asked for, measured: O(1), independent of the tree size
    t.ongoing_load_back = {}
    t0 = time.perf_counter()
    for _ in range(20000):
        RV.transient_blocked(t)
    per_us = (time.perf_counter() - t0) / 20000 * 1e6
    assert per_us < 50, per_us


def test_review_A2_another_requests_holding_fill_blocks_futility(tmp_path, monkeypatch):
    """A waiting fill of ANOTHER request that holds landed references is room the arbitration can still free."""
    _need_module()
    be, arena = _setup(tmp_path, ["a"], slots=8)
    monkeypatch.setattr(ARB, "_alive", lambda pid: True)
    since = ARB.now_ms()
    time.sleep(0.01)
    RV.note_verdict(arena, 0, 0, False, pid=101, refs=0)
    ARB.register(arena, "weg2-8-2", held=8)
    assert RV.futile(arena, since, pp_size=1, rid="weg2-8-1") is None
    assert RV.futile(arena, since, pp_size=1, rid="weg2-8-2") is not None, "its own holding does not count"
    assert RV.futile(arena, since, pp_size=1, rid="weg2-8-1", ignore_holders=True) is not None


def test_re_review_agb_the_read_room_tick_runs_after_the_release_drain():
    """Probe P9: the reap only queues the release (append_host_mem_release); the references return in the drain."""
    import inspect

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache.check_hicache_events)
    assert src.find("_read_room_tick(self)") > src.find("drain_storage_control_queues") >= 0


def test_re_review_agb_a_give_back_stays_while_its_release_is_queued(tmp_path, monkeypatch):
    _need_module()
    import queue as _q

    be, arena = _setup(tmp_path, ["a"], slots=8)
    monkeypatch.setattr(ARB, "_alive", lambda pid: True)
    RV.note_giveback(arena, "weg2-1-9", 8)
    rq = _q.Queue()
    rq.put(object())                                     # the reaped read's release, not drained yet
    tree = types.SimpleNamespace(ongoing_prefetch={}, cache_controller=types.SimpleNamespace(host_mem_release_queue=rq))
    assert RV.reap_givebacks(arena, tree) == 0 and RV.givebacks(arena)
    rq.get()                                             # drained
    assert RV.reap_givebacks(arena, tree) == 1 and not RV.givebacks(arena)
