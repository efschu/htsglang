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
    RV.note_verdict(arena, 0, 0, False, pid=101)
    RV.note_verdict(arena, 0, 0, False, pid=102)
    assert RV.futile(arena, since, pp_size=3) is None, "PP2 has not answered"
    RV.note_verdict(arena, 0, 0, True, pid=103)
    assert RV.futile(arena, since, pp_size=3) is None, "PP2's spill was in its backoff: it did not look"
    RV.note_verdict(arena, 0, 0, False, pid=103)
    assert RV.futile(arena, since, pp_size=3) is not None
    RV.note_verdict(arena, 5, 0, False, pid=102)
    assert RV.futile(arena, since, pp_size=3) is None, "PP1 released pages: room may come"
    RV.note_verdict(arena, 0, 1, False, pid=102)
    assert RV.futile(arena, since, pp_size=3) is None, "PP1 has a write in flight"
    RV.note_verdict(arena, 0, 0, False, pid=102)
    later = ARB.now_ms() + 5
    assert RV.futile(arena, later, pp_size=3) is None, "answers older than the last progress do not count"


def test_transient_counts_only_host_backed_nodes_with_io_in_flight(tmp_path):
    _need_module()
    be, arena = _setup(tmp_path, ["a"], slots=8)
    t = _Tree(arena)
    assert RV.transient_blocked(t) == 0
    t.ongoing_write_through = {7: object()}
    assert RV.transient_blocked(t) == 1
    t.ongoing_write_through = {}
    t.nodes[1].component_data.comps[0].host_lock_ref = 1      # a load-back reads these rows
    assert RV.transient_blocked(t) == 1
    t.nodes[1].component_data.comps[0].host_lock_ref = 0
    t.nodes[1].component_data.comps[0].host_value = None       # no host copy: pins nothing
    t.ongoing_write_through = {7: object()}
    assert RV.transient_blocked(t) == 0
