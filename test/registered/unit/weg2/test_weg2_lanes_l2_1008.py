"""PRIORITY LANES 1008, part L2 (D side), plan deskq/PLAN-PRIO-LANES-1008.md section 2 "D (Decode)".

* ``park_running(rids=..., hold="lane")``: ONLY the named running requests leave -- retracted RETAINING the
  span, forced host write-through, the draft rows saved as the flip park does -- no sleep follows, the other
  requests keep decoding;
* the hold: a held park takes part in neither the 30-s awake re-queue nor the #248h capacity re-queue nor the
  sleep's dormant hold, and ``order_waiting`` does not prefer it;
* ``POST /weg2/lane_floor``: the floor of D's admission (``priority >= floor``), the epoch rule, the re-queue
  of the held parks in ARRIVAL order when the floor falls (age kept, marker ``WEG2-D-PARK requeue(lane)``);
* the victim choice (``displace_for_age``, ``park_youngest``) never takes a request of a higher lane;
* switch off: every path unchanged (the existing park suites are the other half of that proof).

The collaborator runs against a stand-in scheduler (``scheduler.py`` itself is READ, not imported: its import
pulls transformers -> torchao -> the Triton device list a GPU-less desk does not have).
"""
from __future__ import annotations

import ast
import os
import sys
import types
from collections import deque
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    Weg2LaneFloorReqInput,
    Weg2ParkRunningReqInput,
)
from sglang.srt.mem_cache.base_prefix_cache import FORCE_HOST_WRITE_THROUGH_ATTR  # noqa: E402
from sglang.srt.weg2 import d_lane  # noqa: E402
from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402
from sglang.srt.weg2 import lanes as LN  # noqa: E402
from sglang.srt.weg2 import resume_via_p as rvp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_park_wait_h91c2 import FakeClock  # noqa: E402

_SRT = os.path.dirname(os.path.dirname(ds.__file__))


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.delenv(ds.RESUME_MARGIN_ENV, raising=False)
    monkeypatch.setenv(ds.AWAKE_REQUEUE_ENV, "30")


@pytest.fixture
def lanes_on():
    with envs.SGLANG_WEG2_LANES.override(True):
        yield


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(rt, "time", c)
    return c


# ------------------------------------------------------------------ stand-ins
def _req(n, lane=None, seq=None):
    """A D request: rid ``weg2-1-<n>`` (the front's arrival counter n), ``priority`` = its lane."""
    return types.SimpleNamespace(
        rid=f"weg2-1-{n}", kv_arrival_seq=n if seq is None else seq, origin_input_ids=[0] * 10,
        output_ids=[0] * 3, is_fast_lane=False, spill_class=None, priority=lane,
    )


class _Batch:
    """A running batch that really removes what is retracted (release_req + filter_batch(keep_indices))."""

    def __init__(self, reqs, spec=False):
        self.reqs = list(reqs)
        self.batch_is_full = True
        self.released = []
        self.retract_all_calls = 0
        self.spec_algorithm = types.SimpleNamespace(is_none=lambda: False) if spec else None

    def is_empty(self):
        return not self.reqs

    def release_req(self, idx, remaining, server_args, retain=False):
        self.released.append((self.reqs[idx].rid, retain))

    def filter_batch(self, chunked_req_to_exclude=None, keep_indices=None):
        if keep_indices is not None:
            self.reqs = [self.reqs[i] for i in keep_indices]

    def retract_all(self, server_args, offload_kv=True, retain=False):
        self.retract_all_calls += 1
        out, self.reqs = self.reqs, []
        return out


class _Sched:
    def __init__(self, running=(), waiting=(), spec=False):
        self.running_batch = _Batch(running, spec=spec)
        self.waiting_queue = list(waiting)
        self.last_batch = None
        self.enable_overlap = False
        self.result_queue = deque()
        self.chunked_req = None
        self.anchor_tails = []
        self.server_args = types.SimpleNamespace(max_running_requests=6)
        self.weg2_dormant = False
        self.enable_hicache_storage = False
        self.noted = []
        self.sent = []
        self.ipc_channels = types.SimpleNamespace(
            send_to_tokenizer=types.SimpleNamespace(send_output=lambda o, r: self.sent.append(o)))

    def _969ad_note_retract(self, req, site):
        self.noted.append((req.rid, site))

    def _add_request_to_queue(self, req, is_retracted=False):
        if self.weg2_dormant:
            hold = getattr(self, "weg2_dormant_hold", None)
            if hold is None:
                hold = self.weg2_dormant_hold = []
            hold.append(req)
        else:
            self.waiting_queue.append(req)

    def _weg2_group_min_flags(self, flags):
        return [1 if f else 0 for f in flags]

    def uniform_min_avail(self):
        return 0


def _park_lane(s, rids, hold="lane", epoch=7):
    return rt.park_running(s, Weg2ParkRunningReqInput(epoch=epoch, reason="lane", rids=list(rids), hold=hold))


def _floor(s, floor, epoch):
    return rt.lane_floor(s, Weg2LaneFloorReqInput(floor=floor, epoch=epoch))


# ------------------------------------------------- (1) the rids park, no sleep
def test_rids_park_retains_the_span_with_write_through_and_the_others_decode_on(lanes_on, caplog):
    a, b, c, d = _req(1, 0), _req(2, 0), _req(3, 1), _req(4, 1)
    s = _Sched(running=[a, b, c, d], waiting=[_req(9, 0)])
    with caplog.at_level("INFO"):
        out = _park_lane(s, [a.rid, b.rid])
    assert out.success and out.lane is True
    assert out.parked == [a.rid, b.rid] and out.held == [] and out.lane_skipped == {}
    # exactly those two left the batch, retained (retain=True), never retract_all
    assert s.running_batch.released == [(a.rid, True), (b.rid, True)]
    assert s.running_batch.retract_all_calls == 0
    assert [r.rid for r in s.running_batch.reqs] == [c.rid, d.rid]
    # forced host write-through only on the parked; the others are untouched
    assert all(getattr(r, FORCE_HOST_WRITE_THROUGH_ATTR) for r in (a, b))
    assert not any(hasattr(r, FORCE_HOST_WRITE_THROUGH_ATTR) for r in (c, d))
    # parked in the flip site with the hold, in the park list; the queue is untouched
    assert [r.rid for r in s.weg2_d_parked] == [a.rid, b.rid]
    assert all(ds.park_site(r) == ds.SITE_FLIP and d_lane.lane_held(r) for r in (a, b))
    assert [r.rid for r in s.waiting_queue] == ["weg2-1-9"]
    assert s.noted == [(a.rid, "weg2_park_running"), (b.rid, "weg2_park_running")]
    # NO sleep follows: neither the late hold nor the flip-park window opened
    assert getattr(s, rt.LATE_HOLD_ATTR, None) is None
    assert getattr(s, rt.FLIP_PARK_OPEN_ATTR, None) is None
    assert s.running_batch.batch_is_full is False
    # the marker (names fixed in weg2/lanes.py) with rids= and ms=
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith(LN.MARK_D_PARK_PARK))
    assert f"rids=['{a.rid}', '{b.rid}']" in line and " ms=" in line and "hold=lane" in line


def test_the_draft_rows_of_exactly_the_parked_are_saved(lanes_on):
    a, b = _req(1, 0), _req(2, 1)
    s = _Sched(running=[a, b])
    with mock.patch.object(rt.d_park_draft, "save_parked", return_value=1) as save:
        _park_lane(s, [b.rid])
    (_sched, reqs), kw = save.call_args
    assert [r.rid for r in reqs] == [b.rid] and kw == {"site": ds.SITE_FLIP}


def test_unknown_queued_and_chunked_rids_are_named_not_parked(lanes_on):
    a = _req(1, 0)
    queued = _req(5, 0)
    chunk = _req(6, 0)
    # LANES FIX 1: a chunked prefill with NO chunk in flight is parked at its chunk border (test_weg2_lanes_fix1_1010);
    # one whose chunk still writes its rows (cannot be after the landing) is named, not parked
    chunk.inflight_middle_chunks = 1
    s = _Sched(running=[a], waiting=[queued])
    s.weg2_lane_floor, s.weg2_lane_epoch = 1, 1   # D's floor stands (LANES FIX 2: a queued rid is held only below it)
    s.chunked_req = chunk
    out = _park_lane(s, [a.rid, queued.rid, chunk.rid, "weg2-1-99"])
    # LANES FIX 2 (metal 180335): a rid that only waits in ``waiting_queue`` is taken into the lane hold (it used to be
    # named ``held`` and stay in the queue, where D's idle witness counts it: the D->P quiesce never saw an idle group)
    assert out.success and out.parked == [a.rid, queued.rid]
    assert out.held == []
    assert set(out.lane_skipped) == {chunk.rid, "weg2-1-99"}
    assert "chunked prefill" in out.lane_skipped[chunk.rid]
    assert s.chunked_req is chunk  # the lane park never drops the chunked request
    assert s.waiting_queue == [] and d_lane.lane_held(queued) and queued in s.weg2_d_parked


def test_nothing_running_to_park_answers_success_with_the_reasons(lanes_on):
    s = _Sched(running=[_req(1, 1)])
    out = _park_lane(s, ["weg2-1-50"])
    assert out.success and out.parked == [] and "weg2-1-50" in out.lane_skipped
    assert [r.rid for r in s.running_batch.reqs] == ["weg2-1-1"]


def test_a_flip_parked_request_asked_for_a_lane_hold_waits_for_its_floor(lanes_on):
    a = _req(1, 0)
    s = _Sched()
    rt.parked_list(s).append(a)
    ds.mark_parked(a, ds.SITE_FLIP, epoch=3, now=0.0)
    out = _park_lane(s, [a.rid])
    assert out.success and out.parked == [a.rid]
    assert d_lane.lane_held(a)


def test_spec_decoding_only_the_back_of_the_batch_may_leave(lanes_on):
    a, b, c = _req(1, 0), _req(2, 1), _req(3, 0)
    s = _Sched(running=[a, b, c], spec=True)
    out = _park_lane(s, [a.rid, c.rid])  # the middle one stays: not the tail
    assert not out.success and out.parked == [] and "back of the batch" in out.message
    assert [r.rid for r in s.running_batch.reqs] == [a.rid, b.rid, c.rid]
    assert s.running_batch.released == []
    s2 = _Sched(running=[b, a, c], spec=True)
    out2 = _park_lane(s2, [a.rid, c.rid])  # a suffix: allowed
    assert out2.success and out2.parked == [a.rid, c.rid]
    assert [r.rid for r in s2.running_batch.reqs] == [b.rid]


def test_refusals_nothing_parked(lanes_on):
    a = _req(1, 0)
    s = _Sched(running=[a])
    assert not _park_lane(s, [a.rid], hold="bogus").success
    assert not _park_lane(s, [], hold="lane").success  # hold needs rids
    s.anchor_tails = [object()]
    assert not _park_lane(s, [a.rid]).success
    s.anchor_tails = []
    with mock.patch.object(ds, "d_flip_park_active", lambda: False):
        assert not _park_lane(s, [a.rid]).success
    assert s.running_batch.released == [] and [r.rid for r in s.running_batch.reqs] == [a.rid]
    s.weg2_dormant = True
    out = _park_lane(s, [a.rid])
    assert out.success and out.parked == [] and out.lane_skipped == {a.rid: "dormant"}


def test_rids_without_the_switch_are_refused_and_nothing_is_parked():
    a = _req(1, 0)
    s = _Sched(running=[a])
    out = _park_lane(s, [a.rid])
    assert not out.success and "SGLANG_WEG2_LANES" in out.message and out.parked == []
    assert s.running_batch.released == [] and not hasattr(s, "weg2_d_parked")


def test_the_flip_park_without_rids_is_the_old_path(lanes_on):
    """No rids, no hold: park_running is the H91b flip park, switch on or off (retract_all, sleep to follow)."""
    a, b = _req(1, 0), _req(2, 1)
    s = _Sched(running=[a, b], waiting=[_req(9, 0)])
    out = rt.park_running(s, Weg2ParkRunningReqInput(epoch=5, reason="d-to-p"))
    assert out.success and out.parked == [a.rid, b.rid] and out.held == ["weg2-1-9"]
    assert out.lane is False and out.lane_skipped == {}
    assert s.running_batch.retract_all_calls == 1
    assert getattr(s, rt.FLIP_PARK_OPEN_ATTR) == 5


def test_a_flip_park_leaves_a_lane_hold_out_of_its_answer_and_clock(lanes_on):
    held, run = _req(1, 0), _req(2, 1)
    s = _Sched(running=[held, run])
    _park_lane(s, [held.rid])
    since = getattr(held, ds.SINCE_ATTR)
    out = rt.park_running(s, Weg2ParkRunningReqInput(epoch=9, reason="d-to-p"))
    assert out.parked == [run.rid]  # the hold is not a flip-parked request of this park
    assert held in s.weg2_d_parked and d_lane.lane_held(held)
    assert getattr(held, ds.SINCE_ATTR) == since


# ------------------------------------------ (2) the hold survives the requeues
def test_a_hold_survives_the_30s_requeue_and_a_plain_park_does_not(lanes_on, clock):
    a, b = _req(1, 0), _req(2, 0)
    s = _Sched(running=[a, b])
    _park_lane(s, [a.rid], hold="lane")
    _park_lane(s, [b.rid], hold="")  # same park without the hold
    clock.t += 31.0
    moved = rt.park_tick(s)
    assert moved == 1 and [r.rid for r in s.waiting_queue] == [b.rid]  # only the plain park came back
    assert [r.rid for r in s.weg2_d_parked] == [a.rid] and d_lane.lane_held(a)
    clock.t += 3600.0
    assert rt.park_tick(s) == 0 and [r.rid for r in s.weg2_d_parked] == [a.rid]  # an hour later: still held


def test_a_hold_survives_the_248h_capacity_requeue(lanes_on, clock):
    import time

    a = _req(1, 0)
    s = _Sched(running=[a])
    _park_lane(s, [a.rid])
    now = time.monotonic()
    setattr(a, rvp.CAPPARK_SINCE_ATTR, now - 30.0)  # a capacity park older than the re-read cadence
    setattr(a, rvp.CAPPARK_AT_ATTR, now - 10.0)
    assert rt.park_tick(s) == 0
    assert s.waiting_queue == [] and [r.rid for r in s.weg2_d_parked] == [a.rid]
    # the same capacity park WITHOUT the hold is re-queued (the rule the hold withdraws it from)
    b = _req(2, 0)
    rt.parked_list(s).append(b)
    ds.mark_parked(b, ds.SITE_FLIP, epoch=None, now=clock.t)
    setattr(b, rvp.CAPPARK_SINCE_ATTR, now - 30.0)
    setattr(b, rvp.CAPPARK_AT_ATTR, now - 10.0)
    assert rt.park_tick(s) == 1
    assert s.waiting_queue == [b] and s.weg2_d_parked == [a]


def test_a_hold_stays_parked_over_the_sleep(lanes_on):
    held, other = _req(1, 0), _req(2, 0)
    s = _Sched(running=[held, other])
    _park_lane(s, [held.rid])
    rt.parked_list(s).append(other)
    ds.mark_parked(other, ds.SITE_FLIP, epoch=1, now=0.0)
    s.weg2_dormant = True
    assert rt.hold_parked(s, hold_armed=True) == 1  # only the flip park enters the dormant hold
    assert [r.rid for r in s.weg2_dormant_hold] == [other.rid]
    assert s.weg2_d_parked == [held] and d_lane.lane_held(held)
    # unarmed: nothing moves either
    s2 = _Sched(running=[held])
    s2.weg2_d_parked = [held]
    s2.weg2_dormant = True
    assert rt.hold_parked(s2, hold_armed=False) == 0 and s2.weg2_d_parked == [held]
    assert getattr(s2, "_weg2_d_park_slept", False) is False


def test_order_waiting_does_not_prefer_a_held_request():
    held, parked, plain = _req(1, 0), _req(2, 0), _req(3, 0)
    ds.mark_parked(held, ds.SITE_FLIP, now=0.0)
    ds.mark_parked(parked, ds.SITE_FLIP, now=0.0)
    d_lane.mark_hold(held)
    # held is the OLDEST parked, yet it stays where it was in the queue; the flip park goes first
    assert ds.order_waiting([plain, held, parked]) == [parked, plain, held]
    d_lane.clear_hold(held)
    assert ds.order_waiting([plain, held, parked]) == [held, parked, plain]


# ---------------------------------------------------- (3) the floor admission
def test_below_floor_reads_the_priority_as_the_field_is_read():
    s = types.SimpleNamespace()
    assert not d_lane.below_floor(s, _req(1, None))  # floor 0: nobody is below
    setattr(s, d_lane.FLOOR_ATTR, 2)
    assert d_lane.below_floor(s, _req(1, None)) and d_lane.below_floor(s, _req(2, 1))
    assert not d_lane.below_floor(s, _req(3, 2)) and not d_lane.below_floor(s, _req(4, 7))
    for junk in (-3, "abc", 1.5, True):
        assert d_lane.lane_of_req(_req(5, junk)) == 0  # malformed / negative = lane 0


def _lane_skip_fn():
    """``Scheduler._weg2_lane_skip`` out of scheduler.py (the class cannot be imported on a desk)."""
    src = open(os.path.join(_SRT, "managers", "scheduler.py")).read()
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)
                and n.name == "_weg2_lane_skip")
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "scheduler.py", "exec"), ns)
    return ns["_weg2_lane_skip"]


def _admit(s, queue=None):
    """One admission pass as the scheduler loop runs it: the floor first, then the park gate."""
    skip = _lane_skip_fn()
    gate = rt.admission(s, s.running_batch)
    admitted = []
    for r in list(s.waiting_queue if queue is None else queue):
        if getattr(s, "weg2_lane_floor", 0) and skip(s, r) is not None:
            continue
        if gate is not None and gate.skip(r, admitted=[x.rid for x in admitted]) is not None:
            continue
        admitted.append(r)
    return [r.rid for r in admitted]


def test_admission_lets_in_only_lanes_at_or_above_the_floor(lanes_on):
    s = _Sched(waiting=[_req(1, 0), _req(2, 1), _req(3, 2), _req(4, None), _req(5, 2)])
    assert _admit(s) == [f"weg2-1-{n}" for n in (1, 2, 3, 4, 5)]  # floor 0: everyone (the old loop)
    assert _floor(s, 2, 1).success
    assert _admit(s) == ["weg2-1-3", "weg2-1-5"]
    assert _floor(s, 1, 2).success
    assert _admit(s) == ["weg2-1-2", "weg2-1-3", "weg2-1-5"]


def test_a_parked_request_below_the_floor_makes_no_barrier_for_the_lane_that_runs(lanes_on):
    below = _req(1, 0)
    ds.mark_parked(below, ds.SITE_FLIP, epoch=1, now=0.0)
    newcomer = _req(2, 1)
    s = _Sched(running=[_req(3, 1)], waiting=[below, newcomer])
    # floor 0: the parked one is first and holds the newcomer back (the barrier of today)
    gate = rt.admission(s, s.running_batch)
    assert gate is not None and gate.skip(newcomer) == "weg2_d_park_first"
    assert _floor(s, 1, 1).success
    assert rt.admission(s, s.running_batch) is None  # no barrier: the floor keeps `below` out, not a seat
    assert _admit(s) == [newcomer.rid]


def test_the_scheduler_loop_skips_below_the_floor_before_the_park_gate():
    src = open(os.path.join(_SRT, "managers", "scheduler.py")).read()
    floor_at = src.index("_lane_skip = self._weg2_lane_skip(req)")
    gate_at = src.index("_d_skip = _d_park_gate.skip(req, admitted=")
    assert floor_at < gate_at
    assert 'if getattr(self, "weg2_lane_floor", 0):' in src[floor_at - 200:floor_at]  # floor 0: not even called
    assert "_note_skip(_lane_skip, req.rid)" in src[floor_at:gate_at]


# ------------------------------------------------------- (4) the floor RPC
def test_the_epoch_rule(lanes_on):
    s = _Sched()
    assert _floor(s, 0, 0).message == "floor unchanged"
    out = _floor(s, 2, 1)
    assert out.success and (out.floor, out.epoch) == (2, 1)
    assert _floor(s, 2, 1).message == "floor unchanged"  # the same body again: a no-op
    stale = _floor(s, 1, 1)  # same epoch, another floor
    assert not stale.success and "stale" in stale.message and (stale.floor, stale.epoch) == (2, 1)
    assert not _floor(s, 3, 0).success  # epoch 0 with a floor above 0 is no reset: stale
    assert d_lane.floor_of(s) == 2
    assert not _floor(s, 5, 0).success  # a lower epoch than held
    assert d_lane.floor_of(s) == 2
    reset = _floor(s, 0, 0)  # the front's (re)start is always taken
    assert reset.success and (reset.floor, reset.epoch) == (0, 0)
    assert _floor(s, 4, 3).success and d_lane.floor_of(s) == 4  # the epoch may jump ahead


def test_the_floor_needs_the_switch_and_group_d(monkeypatch):
    s = _Sched()
    out = _floor(s, 1, 1)
    assert not out.success and "SGLANG_WEG2_LANES" in out.message
    assert not hasattr(s, d_lane.FLOOR_ATTR)
    with envs.SGLANG_WEG2_LANES.override(True):
        monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
        out = _floor(s, 1, 1)
        assert out.success and "not group D" in out.message and not hasattr(s, d_lane.FLOOR_ATTR)


def test_the_floor_is_written_to_the_progress_beacon(lanes_on):
    from sglang.srt.weg2 import progress_beacon

    s = _Sched()
    with mock.patch.object(progress_beacon, "beat_lane") as beat:
        _floor(s, 3, 4)
    beat.assert_called_once_with(3, 4)


def test_requeue_when_the_floor_falls_in_arrival_order_age_kept(lanes_on, caplog):
    # parked in batch order 3,1,2 -- the arrival (age) is 1,2,3
    r3, r1, r2 = _req(3, 1), _req(1, 1), _req(2, 1)
    seqs = {r.rid: r.kv_arrival_seq for r in (r1, r2, r3)}
    s = _Sched(running=[r3, r1, r2], waiting=[_req(9, 2)])
    assert _park_lane(s, [r3.rid, r1.rid, r2.rid]).success
    assert _floor(s, 2, 1).success  # floor up: the lane-1 holds stay held
    assert s.weg2_d_parked and all(d_lane.lane_held(r) for r in s.weg2_d_parked)
    with caplog.at_level("INFO"):
        out = _floor(s, 1, 2)  # LANE-EMPTY: the floor falls to lane 1
    assert out.success and out.requeued == [r1.rid, r2.rid, r3.rid] and out.held == 0
    assert [r.rid for r in s.waiting_queue] == [r1.rid, r2.rid, r3.rid, "weg2-1-9"]  # the head, oldest first
    assert s.weg2_d_parked == []
    assert not any(d_lane.lane_held(r) for r in (r1, r2, r3))
    assert {r.rid: r.kv_arrival_seq for r in (r1, r2, r3)} == seqs  # the age is untouched
    assert all(ds.park_site(r) == ds.SITE_FLIP for r in (r1, r2, r3))  # they resume first, as a flip park does
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith(LN.MARK_D_PARK_REQUEUE))
    assert " n=3 " in line and f"rids=['{r1.rid}', '{r2.rid}', '{r3.rid}']" in line


def test_requeue_takes_only_the_lanes_the_floor_now_lets_in(lanes_on):
    l0, l1, l2 = _req(1, 0), _req(2, 1), _req(3, 2)
    s = _Sched(running=[l0, l1, l2])
    assert _park_lane(s, [l0.rid, l1.rid, l2.rid]).success
    assert _floor(s, 3, 1).requeued == []  # nothing is at or above 3
    out = _floor(s, 1, 2)
    assert out.requeued == [l1.rid, l2.rid]  # arrival order, not lane order
    assert [r.rid for r in s.weg2_d_parked] == [l0.rid] and out.held == 1
    out = _floor(s, 0, 3)
    assert out.requeued == [l0.rid] and out.held == 0
    assert [r.rid for r in s.waiting_queue] == [l0.rid, l1.rid, l2.rid]


def test_after_the_requeue_the_request_resumes_first_and_the_floor_admits_it(lanes_on):
    held = _req(1, 0)
    newcomer = _req(2, 1)
    s = _Sched(running=[held], waiting=[newcomer])
    _park_lane(s, [held.rid])
    assert _floor(s, 1, 1).success
    assert _admit(s) == [newcomer.rid]
    # the lane-1 work is done: the floor falls, the held park resumes before anything that waited
    s.waiting_queue = []
    s.running_batch.reqs = []
    late = _req(7, 0)
    s.waiting_queue = [late]
    assert _floor(s, 0, 2).requeued == [held.rid]
    assert [r.rid for r in s.waiting_queue] == [held.rid, late.rid]
    assert _admit(s)[0] == held.rid


def test_abort_reaches_a_held_park(lanes_on):
    a, b = _req(1, 0), _req(2, 0)
    s = _Sched(running=[a, b])
    _park_lane(s, [a.rid, b.rid])
    n = rt.park_abort(s, types.SimpleNamespace(rid=a.rid, abort_all=False))
    assert n == 1 and [r.rid for r in s.weg2_d_parked] == [b.rid]
    assert len(s.sent) == 1


# ------------------------------------------------------------- (5) victim choice
def _dsched(running, waiting, cap=3, spec=False):
    batch = types.SimpleNamespace(
        reqs=list(running), released=[],
        spec_algorithm=None if not spec else types.SimpleNamespace(is_none=lambda: False))
    batch.release_req = lambda idx, rem, sa, retain=False: batch.released.append((batch.reqs[idx].rid, retain))

    def filt(keep_indices):
        batch.reqs = [batch.reqs[i] for i in keep_indices]

    batch.filter_batch = filt
    sched = types.SimpleNamespace(
        waiting_queue=list(waiting), server_args=types.SimpleNamespace(max_running_requests=cap),
        running_batch=batch,
        token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: 1 << 30))
    sched._add_request_to_queue = lambda req, is_retracted=False: sched.waiting_queue.append(req)
    return sched, batch


def _displace(sched, batch):
    with mock.patch.object(ds, "d_flip_park_active", lambda: True), \
            mock.patch.object(rt, "seat_cap", lambda s: None):
        return rt.displace_for_age(sched, batch)


def test_displace_for_age_never_takes_a_victim_of_a_higher_lane(lanes_on):
    # the older waiter (2) is lane 0; the youngest running one (7) is lane 1: protected
    sched, batch = _dsched([_req(5, 0), _req(6, 0), _req(7, 1)], [_req(2, 0)])
    assert _displace(sched, batch) == "weg2-1-6"  # the youngest request OF ITS LANE OR BELOW
    assert batch.released == [("weg2-1-6", True)]
    # all runners above the waiter's lane: nobody leaves
    sched2, batch2 = _dsched([_req(5, 1), _req(6, 1), _req(7, 1)], [_req(2, 0)])
    assert _displace(sched2, batch2) is None and batch2.released == []


def test_displace_for_age_is_unchanged_inside_a_lane(lanes_on):
    sched, batch = _dsched([_req(5, 1), _req(6, 1), _req(7, 1)], [_req(2, 1)])
    assert _displace(sched, batch) == "weg2-1-7"


def test_displace_for_age_ignores_a_waiter_the_floor_keeps_out(lanes_on):
    sched, batch = _dsched([_req(5, 1), _req(6, 1), _req(7, 1)], [_req(2, 0)])
    setattr(sched, d_lane.FLOOR_ATTR, 1)
    assert _displace(sched, batch) is None and batch.released == []  # lane 0 cannot be admitted: no victim for it
    # an admissible older waiter still displaces within its lane
    sched2, batch2 = _dsched([_req(5, 1), _req(6, 1), _req(7, 1)], [_req(2, 1)])
    setattr(sched2, d_lane.FLOOR_ATTR, 1)
    assert _displace(sched2, batch2) == "weg2-1-7"


def test_displace_for_age_switch_off_reads_the_priority_of_nobody():
    # lanes off: the old rule, whatever the priority field says
    sched, batch = _dsched([_req(5, 0), _req(6, 0), _req(7, 9)], [_req(2, 0)])
    assert _displace(sched, batch) == "weg2-1-7"


def _youngest(sched, rid):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput as In

    return rt.park_running(sched, In(epoch=1, youngest=rid))


def _ysched(running):
    s = _Sched(running=running)
    s.running_batch.release_req = lambda idx, rem, sa, retain=False: None
    return s


def test_park_youngest_never_takes_a_lane_above_the_floor(lanes_on):
    hi, lo = _req(7, 2), _req(6, 1)
    s = _ysched([lo, hi])
    setattr(s, d_lane.FLOOR_ATTR, 1)
    out = _youngest(s, hi.rid)
    assert not out.success and "above the floor" in out.message
    assert [r.rid for r in s.running_batch.reqs] == [lo.rid, hi.rid]
    # the floor lane itself is a legitimate victim lane (the rules inside a lane stay as they are)
    ok = _youngest(s, lo.rid)
    assert ok.success and ok.parked == [lo.rid]


def test_park_youngest_unchanged_with_the_switch_off():
    hi = _req(7, 5)
    s = _ysched([_req(6, 0), hi])
    out = _youngest(s, hi.rid)
    assert out.success and out.parked == [hi.rid]


# ----------------------------------------------------- (6) names and wiring
def test_the_names_are_the_ones_l1_fixed():
    assert LN.RPC_LANE_FLOOR == "/weg2/lane_floor" and (LN.RPC_KEY_FLOOR, LN.RPC_KEY_EPOCH) == ("floor", "epoch")
    assert LN.MARK_D_PARK_PARK == "WEG2-D-PARK park(lane)"
    assert LN.MARK_D_PARK_REQUEUE == "WEG2-D-PARK requeue(lane)"
    body = Weg2LaneFloorReqInput(**{LN.RPC_KEY_FLOOR: 3, LN.RPC_KEY_EPOCH: 4})
    assert (body.floor, body.epoch) == (3, 4)


def _read(*parts):
    return open(os.path.join(_SRT, *parts)).read()


def test_wiring_lane_floor_route_rpc_and_dispatch():
    http = _read("entrypoints", "http_server.py")
    assert '@app.api_route("/weg2/lane_floor", methods=["POST"])' in http
    assert "_global_state.tokenizer_manager.weg2_lane_floor(obj)" in http
    assert '("weg2_lane_floor", Weg2LaneFloorReqOutput)' in _read("managers", "tokenizer_control_mixin.py")
    sch = _read("managers", "scheduler.py")
    assert "(Weg2LaneFloorReqInput, self.handle_weg2_lane_floor)" in sch
    assert "return d_park_runtime.lane_floor(self, recv_req)" in sch
    # a lane park does not spend the front's collect window
    assert 'if not (getattr(recv_req, "rids", None) or getattr(recv_req, "hold", "")):' in sch


def test_the_http_body_of_a_flip_park_is_unchanged_and_a_lane_answer_adds_two_keys():
    http = _read("entrypoints", "http_server.py")
    assert 'if getattr(ret, "lane", False):' in http
    assert 'body["lane"] = True' in http and 'body["skipped"]' in http
