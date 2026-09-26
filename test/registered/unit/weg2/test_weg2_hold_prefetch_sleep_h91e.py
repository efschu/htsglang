"""H91e: the dormant hold's own store read is not a sleep term (fnFL2h91bb3).

THE SPECIMEN, boot fnFL2h91bb3 @ 50cd2884ac (2026-09-26), D = TP3, the first
metal boot on which the H91 park path really ran:

* 15:52:39.722 front: ``WEG2 WAIT-BOUND FIRED epoch=16`` -> ``/weg2/park_running``;
  D: ``#969AD RETRACT site=weg2_park_running`` + ``WEG2-D-PARK park_running
  epoch=16 reason=wait-bound-60s: 1 running retracted ... parked=['weg2-16-23']``
  on all three ranks;
* 15:52:39.905 first sleep leg (kv_cache + cuda_graph): drain (idle), flush
  (``#1470 FLUSH-PUBLISH`` joined the park's write-throughs BEFORE the reset),
  pause, ``WEG2-DORMANT set``, then ``WEG2-D-PARK hold: 1 parked request(s) at
  the head of the dormant hold ['weg2-16-23']`` -- the ordinary intake
  registered the request's store read first (#1455), ``#1028 HICACHE-ROUND ...
  ongoing_prefetch=1``;
* 15:52:39.985 second sleep leg (weights_*): the pre-sleep drain found
  ``hicache_prefetch(1: weg2-16-)`` on every rank, 887 group polls / 10.01 s,
  ``W120 Weg2SleepDrainRefused`` 3/3 -> ``W29 Weg2FlipRankDisagree`` -> D dead.

The drain could never have ended it: a prefetch record leaves
``ongoing_prefetch`` only through ``check_prefetch_progress`` (admission, the
#1233 orphan collector -- which skips held rids, #1456 -- and the hold's own
top-up), a revoke, an abort or a re-issue; ``check_hicache_events`` calls none
of them. The hold's read is meant to span the flip (#1455/#1456), it targets
host rows only, and the KV pool is already paused -- so from the dormant point
on it is not a sleep term. Everything else still is.

The stand-in scheduler runs the REAL ``Scheduler.idle_blockers`` /
``Scheduler.is_fully_idle``, the REAL ``d_park_runtime.park_running`` /
``hold_parked`` and the REAL ``SchedulerWeightUpdaterManager`` drain over a
real three-rank MAX reduce (one thread per rank). Its intake keeps the real
order (read registered, then held) -- pinned by a ratchet below.

RED on 50cd2884ac: the two ``test_bb3_*`` cases (W120 on 3/3), the
pure-function and idle-assert cases (names absent), and the foreign-read
negative on its NAMING only (base names the held read beside the foreign one;
its W120 verdict is the same on both sides). The load-back negative and the
intake-order ratchet are green on both sides by design.
"""
from __future__ import annotations

import inspect
import os
import threading
from collections import deque
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.test_utils import maybe_stub_sgl_kernel  # noqa: E402

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.weight_updater import (  # noqa: E402
    SchedulerWeightUpdaterManager,
)
from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402

RID = "weg2-16-23"
_COLLECTIVE_TIMEOUT_S = 5.0


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.delenv(ds.RESUME_MARGIN_ENV, raising=False)


class _Group:
    """A 3-rank all_reduce over threads (the gloo shape of the drain's reduce)."""

    def __init__(self, world):
        self.world = world
        self._slots = [None] * world
        self._out = None
        self._b1 = threading.Barrier(world, timeout=_COLLECTIVE_TIMEOUT_S)
        self._b2 = threading.Barrier(world, timeout=_COLLECTIVE_TIMEOUT_S)

    def all_reduce(self, rank, values, op):
        self._slots[rank] = list(values)
        if self._b1.wait() == 0:
            self._out = [op(col) for col in zip(*self._slots)]
        self._b2.wait()
        return list(self._out)


class _RankTree:
    """One rank's HiCache terms. ``check_hicache_events`` is the real poll's
    shape for a prefetch record: a group collective that terminates nothing
    (only admission / orphan collection / the hold top-up do)."""

    enable_storage = True

    def __init__(self, group, rank):
        self.group = group
        self.rank = rank
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        self.polls = 0

    def check_hicache_events(self):
        self.polls += 1
        self.group.all_reduce(self.rank, [0], max)

    def hicache_group_max(self, values, *, label):
        return self.group.all_reduce(self.rank, values, max)


class _Batch:
    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.batch_is_full = True

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, **_kw):
        pass

    def retract_all(self, server_args, offload_kv=True, retain=False):
        out, self.reqs = self.reqs, []
        return out


class _RankSched:
    """Group D's scheduler on one rank, as far as park, hold and sleep read it."""

    idle_blockers = Scheduler.idle_blockers
    is_fully_idle = Scheduler.is_fully_idle

    def __init__(self, group, rank, running):
        self.running_batch = _Batch(running)
        self.waiting_queue = []
        self.last_batch = None
        self.enable_overlap = False
        self.result_queue = deque()
        self.chunked_req = None
        self.anchor_tails = []
        self.server_args = SimpleNamespace()
        self.weg2_dormant = False
        self.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        self.kv_session_offload = None
        self.grammar_manager = SimpleNamespace(grammar_queue=[])
        self.disaggregation_mode = DisaggregationMode.NULL
        self.enable_hisparse = False
        self.enable_hierarchical_cache = True
        self.tree_cache = _RankTree(group, rank)

    def _pp_microbatches_drained(self):
        return True

    def _969ad_note_retract(self, req, site):
        pass

    def _add_request_to_queue(self, req, is_retracted=False):
        # The REAL order (#1455, pinned below): the store read is registered
        # by the intake FIRST, then the dormant hold takes the request.
        self.tree_cache.ongoing_prefetch[req.rid] = SimpleNamespace(rid=req.rid)
        if self.weg2_dormant:
            hold = getattr(self, "weg2_dormant_hold", None)
            if hold is None:
                hold = self.weg2_dormant_hold = []
            hold.append(req)
            return
        self.waiting_queue.append(req)


def _req(rid, seq):
    return SimpleNamespace(
        rid=rid, kv_arrival_seq=seq, origin_input_ids=[0] * 3001, output_ids=[0] * 4876,
        is_fast_lane=False, spill_class=None,
    )


def _rank(group, rank, running):
    sch = _RankSched(group, rank, running)
    wu = SchedulerWeightUpdaterManager.__new__(SchedulerWeightUpdaterManager)
    wu.scheduler = sch
    wu.is_fully_idle = sch.is_fully_idle
    return sch, wu


def _bb3_rank(sch, wu, *, bound_s, extra=None):
    """One rank of group D through bb3's sequence: park, sleep leg 1
    (kv_cache: drain, pause -> dormant, hold), sleep leg 2 (weights: drain,
    idle assert). ``extra`` plants another term at the dormant point."""
    rt.park_running(
        sch, Weg2ParkRunningReqInput(epoch=16, reason="wait-bound-60s"), late_hold_armed=True
    )
    wu._weg2_drain_hicache_before_sleep(bound_s=bound_s)  # leg 1
    assert sch.is_fully_idle(), sch.idle_blockers()
    sch.weg2_dormant = True  # pause(kv_cache) -> WEG2-DORMANT set
    rt.hold_parked(sch, hold_armed=True)
    if extra is not None:
        extra(sch)
    wu._weg2_drain_hicache_before_sleep(bound_s=bound_s)  # leg 2
    return wu._weg2_sleep_idle()


def _run_group(bound_s, extra=None):
    group = _Group(3)
    ranks = [_rank(group, r, [_req(RID, 7)]) for r in range(3)]
    outcome = [None] * 3

    def main(r):
        try:
            outcome[r] = _bb3_rank(*ranks[r], bound_s=bound_s, extra=extra)
        except Exception as exc:  # noqa: BLE001 -- the outcome IS the finding
            outcome[r] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=main, args=(r,)) for r in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "a rank never left the sleep"
    return outcome, ranks


# ------------------------------------------------------------------ the death
def test_bb3_park_then_sleep_every_rank_sleeps():
    """RED on 50cd2884ac: 'Weg2SleepDrainRefused: W120 ... hicache_prefetch(1:
    weg2-16-)' on all three ranks, exactly bb3's 15:52:49 lines."""
    outcome, ranks = _run_group(bound_s=0.5)
    assert outcome == [True, True, True], outcome
    # every rank polled exactly as often (the x105 law) -- here: not at all,
    # the verdict was idle on the first reduce of each leg
    assert len({sch.tree_cache.polls for sch, _ in ranks}) == 1


def test_bb3_the_held_read_keeps_running_and_the_wake_releases_it_first():
    """The fix exempts, it does not end: the hold's read is still open after
    the sleep (the flip is when it runs, #1455), the parked request is at the
    head of the hold, and once the group is awake the same read is an ordinary
    term again (the admission, not the sleep, terminates it)."""
    outcome, ranks = _run_group(bound_s=0.5)
    assert outcome == [True, True, True], outcome
    for sch, wu in ranks:
        assert list(sch.tree_cache.ongoing_prefetch) == [RID]
        assert [r.rid for r in sch.weg2_dormant_hold] == [RID]
        assert ds.park_site(sch.weg2_dormant_hold[0]) == ds.SITE_FLIP
        assert wu._weg2_hold_owned_prefetch() == frozenset({RID})
        sch.weg2_dormant = False  # the wake
        assert wu._weg2_hold_owned_prefetch() == frozenset()
        assert sch.idle_blockers() == ["hicache_prefetch(1: weg2-16-)"]
        assert not wu._weg2_sleep_idle()


# ------------------------------------------------------------ the negatives
def test_a_read_the_hold_does_not_own_still_blocks_the_sleep():
    """A dormant group's prefetch of a request that is NOT held (here: planted
    beside the held one) is a sleep term -- W120 on every rank (on both sides),
    naming only that read (red on 50cd2884ac, which names both)."""
    def plant(sch):
        sch.tree_cache.ongoing_prefetch["weg2-9-9"] = SimpleNamespace(rid="weg2-9-9")

    outcome, _ = _run_group(bound_s=0.2, extra=plant)
    assert all(str(o).startswith("Weg2SleepDrainRefused") for o in outcome), outcome
    assert all("hicache_prefetch(1: weg2-9-9)" in str(o) for o in outcome), outcome
    assert all("weg2-16-" not in str(o).split("blocks on")[1] for o in outcome), outcome


def test_a_load_back_beside_the_held_read_still_blocks_the_sleep():
    """The device half of HiCache is never exempt."""
    def plant(sch):
        sch.tree_cache.ongoing_load_back[42] = object()

    outcome, _ = _run_group(bound_s=0.2, extra=plant)
    assert all(str(o).startswith("Weg2SleepDrainRefused") for o in outcome), outcome
    assert all("hicache_load_back(1)" in str(o) for o in outcome), outcome


def test_hold_owned_prefetch_is_empty_unless_dormant_and_held():
    from sglang.srt.managers.weg2_sleep_drain import hold_owned_prefetch

    held = [SimpleNamespace(rid="a"), SimpleNamespace(rid="b")]
    ongoing = {"a": 1, "c": 2}
    assert hold_owned_prefetch(dormant=True, hold=held, ongoing_prefetch=ongoing) == {"a"}
    assert hold_owned_prefetch(dormant=False, hold=held, ongoing_prefetch=ongoing) == frozenset()
    assert hold_owned_prefetch(dormant=True, hold=(), ongoing_prefetch=ongoing) == frozenset()
    assert hold_owned_prefetch(dormant=True, hold=held, ongoing_prefetch={}) == frozenset()


# --------------------------------------------------------------- ratchets
def test_the_release_leg_asserts_idle_with_the_same_exemption_as_its_drain():
    src = inspect.getsource(SchedulerWeightUpdaterManager.release_memory_occupation)
    i_drain = src.index("self._weg2_drain_hicache_before_sleep()")
    i_assert = src.index("self._weg2_sleep_idle()")
    assert i_drain < i_assert
    assert "self.is_fully_idle()" not in src[i_drain:i_assert + 40]


def test_the_real_intake_registers_the_read_before_the_dormant_hold():
    """Pins the stand-in's order: #1455 -- the hold sits AFTER the prefetch."""
    src = inspect.getsource(Scheduler._add_request_to_queue)
    i_read = src.index("self._prefetch_kvcache(req)")
    i_hold = src.index("hold.append(req)")
    assert i_read < i_hold
