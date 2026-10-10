"""PRIORITY LANES fix round 2 (metal jjbbrx, boot 18:03:35Z, image int23lanes2 @ ed880c51b3), 10.10.2026.

Probe 2b stopped the server (W3 Weg2DrainWitnessDisagreement).  What the logs show (deskq/done/lanes-metall3-1010/,
boot logs ...180335.{front,P,D}.log), in order:

* 18:12:39,06 the lane-1 request (weg2-7-19, 16 000 tokens) arrives while the 98 000-token lane-0 leg (weg2-6-18) is in
  its LAST chunk on P; ``arrive`` raises the floor and awaits P's reply to the floor RPC (1667 ms: P answers at the end of
  its pass).  The lane-1 request has no Pending, no queue place.
* 18:12:40,70 the lane-0 leg ends (P-DRAIN prefilled=1 queue_at_exit=0, drain_s=18.7) -- 77 ms BEFORE the lane-1 Pending
  exists; the P phase is over, ``P->D`` flip (epoch 7), the lane-0 request is handed to D.
* 18:12:45,28 D is awake.  The lane-0 request waits in D's ``waiting_queue``: D's admission skips it (floor 1).  The front
  books it as a lane park (``_d_parked`` with the never-lapsing stamp) and so keeps it out of ``_flip_ledger``.
* 18:12:45,29 ``D->P`` flip epoch 8 (for the lane-1 request, ARRIVAL-SEAT PBOUND-FLIP-NOW): the drain is skipped (ledger
  empty), ``_wait_bound_park`` finds ``running = _flip_ledger(D) = []`` -> ``nothing-running``: no park RPC.  The quiesce
  polls ``/flush_cache`` on D: 400 ``not-idle because: waiting_queue`` (#queue-req: 1, #running-req: 0), 22 s, W3.

So the held request was in NO list D's idle verdict ignores (``Scheduler.is_fully_idle`` counts ``waiting_queue``) while the
front's two witnesses (drain, W3) had already taken it out.  The fix is on D: a request below D's floor that only waits in the
queue is taken into the LANE HOLD (``weg2_d_parked`` + ``LANE_HOLD_ATTR``, where a retracted lane park sits), per pass
(``Scheduler._weg2_d_lane_hold_queued``) and when the front's lane park RPC names it (``park_running(rids, hold="lane")``
used to answer such a rid ``held`` and leave it in the queue).  Plus, front side: the P phase does not end while a floor-lane
arrival has no place yet (the same predicate that holds the take, ``LC.take_wait``): no P->D->P pair for a request that
needs P.  P's chunk park (``WEG2-PARK (lane)``) had nothing to park: the floor reached PP0 after the last chunk (P log:
``FINISH-TRACK`` 18:12:40 before ``PR LANE-FLOOR``); the default ``LANE_PREEMPT_CHUNK_TOKENS=0`` does not prevent it
(``AdderHoldTest`` of the L3 file runs with the env unset; pinned again below).

``scheduler.py`` itself is READ, not imported (its import pulls transformers -> torchao).
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers.io_struct import Weg2LaneFloorReqInput  # noqa: E402
from sglang.srt.weg2 import d_lane  # noqa: E402
from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402
from sglang.srt.weg2 import lane_ctl as LC  # noqa: E402
from sglang.srt.weg2 import lanes as LN  # noqa: E402
from sglang.srt.weg2 import lanes_p  # noqa: E402
from sglang.srt.weg2 import phase_policy as PP  # noqa: E402
from sglang.srt.weg2.front import Front, _p_drain_pool, witness_verdict  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_lanes_l2_1008 import (  # noqa: E402,F401  (fixtures + stand-ins of the L2 file)
    _Batch, _Sched, _floor, _group_d, _park_lane, _req, clock, lanes_on,
)
from test_weg2_lanes_l3_1008 import _LanesOn, _run_chunk_decision, _stage  # noqa: E402
from test_weg2_lanes_l4_1008 import (  # noqa: E402
    NOW, LaneHarness, LaneGroup, _arrive, _e2e, _front, _note, _on, _pend,
)
from test_weg2_lanes_fix1_1010 import _pool  # noqa: E402
from test_weg2_phase_policy_h91c import _until  # noqa: E402

_SRT = os.path.dirname(os.path.dirname(ds.__file__))


def _idle(s):
    """The clauses of ``Scheduler.is_fully_idle`` / ``idle_blockers`` the metal run named (``not-idle because:
    waiting_queue``; ``#running-req: 0``): running batch, chunked request, waiting queue."""
    return s.running_batch.is_empty() and s.chunked_req is None and not s.waiting_queue


def _floored(s, floor=1, epoch=1):
    """The state D's floor RPC leaves on the scheduler."""
    setattr(s, d_lane.FLOOR_ATTR, floor)
    setattr(s, d_lane.EPOCH_ATTR, epoch)
    return s


def _pass_hook(s):
    """``Scheduler._weg2_d_lane_hold_queued`` as ``get_next_batch_to_run`` calls it (stand-in: the method body)."""
    return len(rt.hold_queued(s))


# ================================================================== A. D: a floor-held queued request is in the lane hold
def test_the_metal_state_a_floor_held_request_in_the_queue_keeps_d_from_ever_being_idle(lanes_on):
    """The probe-2b state: D awake, nothing running, the lane-0 hand-off in ``waiting_queue``, floor 1.  Without the
    hold D's idle witness says 'not idle because: waiting_queue' for as long as the floor stands (the 22 s stall)."""
    lo = _req(18, 0)
    s = _floored(_Sched(waiting=[lo]))
    assert not _idle(s) and s.waiting_queue == [lo]


def test_the_pass_hook_takes_the_floor_held_queue_into_the_lane_hold_and_d_is_idle(lanes_on, caplog):
    lo, hi, lo2 = _req(18, 0), _req(19, 1), _req(20, 0)
    s = _floored(_Sched(waiting=[lo, hi, lo2]))
    with caplog.at_level("INFO"):
        assert _pass_hook(s) == 2
    assert s.waiting_queue == [hi]                                   # the floor lane stays queued: it can run
    assert s.weg2_d_parked == [lo, lo2] and all(d_lane.lane_held(r) for r in (lo, lo2))
    assert not d_lane.lane_held(hi)
    assert any(r.getMessage().startswith(d_lane.MARK_D_HOLD_QUEUED) and "via=pass" in r.getMessage()
               for r in caplog.records)
    s.waiting_queue = []                                             # hi admitted and done
    assert _idle(s)                                                  # D can quiesce: the metal W3 does not happen
    assert _pass_hook(s) == 0                                        # nothing left: silent, nothing moves


def test_the_hold_is_oldest_first_by_arrival_not_by_queue_position(lanes_on):
    a, b = _req(30, 0), _req(10, 0)                                  # b arrived first, queued behind a
    s = _floored(_Sched(waiting=[a, b]))
    rt.hold_queued(s)
    assert s.weg2_d_parked == [b, a]


def test_the_lane_park_rpc_takes_a_queued_rid_into_the_hold_and_answers_it_parked(lanes_on):
    lo = _req(18, 0)
    s = _floored(_Sched(waiting=[lo]))
    out = _park_lane(s, [lo.rid])
    assert out.success and out.lane is True and out.parked == [lo.rid] and out.held == []
    assert s.waiting_queue == [] and s.weg2_d_parked == [lo] and d_lane.lane_held(lo)
    assert _idle(s)


def test_the_rpc_parks_the_running_and_takes_the_queued_in_one_answer(lanes_on):
    run, q = _req(1, 0), _req(2, 0)
    s = _floored(_Sched(running=[run], waiting=[q]))
    out = _park_lane(s, [run.rid, q.rid])
    assert out.success and out.parked == [run.rid, q.rid] and out.held == []
    assert s.weg2_d_parked == [run, q] and all(d_lane.lane_held(r) for r in (run, q))
    assert s.waiting_queue == [] and s.running_batch.is_empty()


def test_the_rpc_of_a_queued_rid_the_floor_lane_owns_is_not_asked_for_and_stays(lanes_on):
    """Only the rids the front names move: a queued request of the floor lane that is not named is untouched."""
    lo, hi = _req(1, 0), _req(2, 1)
    s = _floored(_Sched(waiting=[lo, hi]))
    out = _park_lane(s, [lo.rid])
    assert out.parked == [lo.rid] and s.waiting_queue == [hi] and not d_lane.lane_held(hi)
    out2 = _park_lane(s, [hi.rid])                                   # a named rid of the floor lane is never held
    assert out2.parked == [] and out2.held == [hi.rid] and s.waiting_queue == [hi]


def test_the_rpc_never_holds_a_request_d_s_own_floor_lets_in_the_pass_rule_takes_it_with_the_floor(lanes_on):
    """The floor RPC and the park RPC travel side by side: a park served BEFORE the floor must not put a request into a
    hold that only a floor CHANGE ends (D's floor 0 admits it).  It stays queued, named ``held``; once D's floor stands the
    pass rule moves it."""
    lo = _req(18, 0)
    s = _Sched(waiting=[lo])                                          # D's floor is still 0
    out = _park_lane(s, [lo.rid])
    assert out.success and out.parked == [] and out.held == [lo.rid]
    assert s.waiting_queue == [lo] and not d_lane.lane_held(lo) and not getattr(s, "weg2_d_parked", None)
    _floored(s)                                                       # the floor RPC lands
    assert _pass_hook(s) == 1 and s.waiting_queue == [] and d_lane.lane_held(lo)


def test_a_park_without_hold_names_a_queued_rid_held_as_before(lanes_on):
    """No ``hold``: the ordinary flip-site rids park; a queued rid stays only queued and is named (the L2 contract)."""
    q = _req(2, 0)
    s = _Sched(waiting=[q])
    out = _park_lane(s, [q.rid], hold="")
    assert out.success and out.parked == [] and out.held == [q.rid]
    assert s.waiting_queue == [q] and not getattr(s, "weg2_d_parked", None)


def test_the_hold_ends_where_every_lane_hold_ends_the_floor_falling_to_its_lane(lanes_on, caplog):
    lo, hi = _req(18, 0), _req(19, 1)
    s = _floored(_Sched(waiting=[lo, hi]))
    rt.hold_queued(s)
    assert _floor(s, 1, 1).success is True                           # same epoch + floor: a no-op
    assert d_lane.lane_held(lo) and s.weg2_d_parked == [lo]
    with caplog.at_level("INFO"):
        out = _floor(s, 0, 2)                                        # the lane-1 request ended
    assert out.success and out.requeued == [lo.rid] and out.held == 0
    assert not d_lane.lane_held(lo) and not s.weg2_d_parked
    assert s.waiting_queue[0] is lo                                  # back at the head of the queue, age untouched


def test_the_hold_survives_the_sleep_of_the_next_flip_like_every_lane_hold(lanes_on):
    lo = _req(18, 0)
    s = _floored(_Sched(waiting=[lo]))
    rt.hold_queued(s)
    assert rt.hold_parked(s, hold_armed=True) == 0                   # a lane hold waits for its floor, not for the wake
    assert s.weg2_d_parked == [lo] and d_lane.lane_held(lo)


def test_the_pass_rule_does_nothing_without_a_floor_off_the_switch_or_asleep(lanes_on):
    lo = _req(18, 0)
    s = _Sched(waiting=[lo])
    assert rt.hold_queued(s) == [] and s.waiting_queue == [lo]       # floor 0
    _floored(s)
    s.weg2_dormant = True
    assert rt.hold_queued(s) == [] and s.waiting_queue == [lo]       # D asleep: its hold is the dormant hold
    s.weg2_dormant = False
    s.waiting_queue = []
    assert rt.hold_queued(s) == []                                   # empty queue


def test_switch_off_nothing_moves_even_with_a_floor_on_the_scheduler():
    lo = _req(18, 0)
    s = _floored(_Sched(waiting=[lo]))
    assert rt.hold_queued(s) == [] and rt.hold_queued(s, rids=[lo.rid]) == []
    assert s.waiting_queue == [lo] and not hasattr(s, "weg2_d_parked")


def test_off_group_d_or_without_the_park_nothing_moves(lanes_on, monkeypatch):
    from unittest import mock

    lo = _req(18, 0)
    s = _floored(_Sched(waiting=[lo]))
    with mock.patch.object(ds, "d_flip_park_active", lambda: False):
        assert rt.hold_queued(s) == []
    assert s.waiting_queue == [lo]


def test_every_tp_rank_moves_the_same_requests_the_same_way(lanes_on):
    outs = []
    for _rank in range(3):
        reqs = [_req(18, 0), _req(19, 1), _req(20, 0)]
        s = _floored(_Sched(waiting=reqs))
        rt.hold_queued(s)
        out = _park_lane(s, [reqs[1].rid])                           # a rid the floor lane owns, named to the rpc
        outs.append(([r.rid for r in s.weg2_d_parked], [r.rid for r in s.waiting_queue], out.parked, out.held))
    assert outs[0] == outs[1] == outs[2]


def test_mutant_without_the_hold_d_stays_not_idle_for_the_witness_the_metal_w3(lanes_on, monkeypatch):
    """CAN-FAIL twin of the pass-hook test: the pass step does nothing -> the queue keeps the held request."""
    monkeypatch.setattr(rt, "hold_queued", lambda *a, **k: [])
    lo = _req(18, 0)
    s = _floored(_Sched(waiting=[lo]))
    _pass_hook(s)
    assert not _idle(s)


def test_the_scheduler_calls_the_pass_step_only_with_a_floor_after_the_park_tick():
    src = open(os.path.join(_SRT, "managers", "scheduler.py")).read()
    head = src[src.index("    def get_next_batch_to_run("):]
    head = head[:head.index("        _weg2_resume_warm_tick(self)")]
    assert head.index("self._weg2_d_park_tick()  # H91b") < head.index("self._weg2_d_lane_hold_queued()")
    assert 'if getattr(self, "weg2_lane_floor", 0):\n' in head       # floor 0 / switch off: one attribute read
    body = src[src.index("    def _weg2_d_lane_hold_queued("):]
    assert "d_park_runtime.hold_queued(self)" in body[:400]


# ================================================================== B. the witnesses agree
def test_the_drain_witness_agrees_when_d_holds_the_lane_request_in_the_hold(lanes_on):
    """Probe 2b end to end at the two witnesses: the front books the lane-0 hand-off as a park (ledger empty), D's
    scheduler runs its pass step.  Agreement = ``witness_verdict(len(ledger), idle) is None``."""
    f = _front("D")
    D = f.groups["D"]
    D.outstanding["weg2-6-18"] = NOW
    f._d_parked["weg2-6-18"] = PP.LANE_PARK_STAMP                     # LC._park_d's booking
    ledger = Front._flip_ledger(f, D)
    assert ledger == []                                              # the front side: nothing to wait for
    lo = _req(18, 0)
    s = _floored(_Sched(waiting=[lo]))
    assert witness_verdict(len(ledger), _idle(s)) == "front drained, rank NOT idle"      # before the pass step: W3
    _pass_hook(s)
    assert witness_verdict(len(ledger), _idle(s)) is None            # after it: the two witnesses agree


def test_canfail_the_witness_message_is_the_one_the_metal_stop_wrote(lanes_on):
    assert witness_verdict(0, False) == "front drained, rank NOT idle"


def test_flip_begin_names_the_lane_held_share_of_outstanding_and_the_ledger(lanes_on):
    f = _front("D")
    D = f.groups["D"]
    assert LC.flip_begin_note(f, D) == ""                            # no lane booking: the old line, unchanged
    D.outstanding["weg2-6-18"] = NOW
    D.outstanding["weg2-7-19"] = NOW
    f._d_parked["weg2-6-18"] = PP.LANE_PARK_STAMP
    LC.ctl(f).dormant["weg2-6-18"] = NOW                              # `_lane_parked_set` reads the register
    assert LC.flip_begin_note(f, D) == " lane_held=1 ledger=1"


def test_the_flip_begin_line_gets_the_note_only_with_the_switch_on():
    src = open(os.path.join(_SRT, "weg2", "front.py")).read()
    assert '_lctl.flip_begin_note(self, S) if _lanes.enabled() else "")' in src
    assert 'logger.info("WEG2-FLIP begin epoch=%d sleep=%s wake=%s outstanding=%d queue=%d%s"' in src


# ================================================================== C. the P phase stays open for a floor-lane arrival
@_on
async def test_the_p_drain_stays_open_while_a_floor_lane_arrival_has_no_place_and_then_dispatches_its_leg():
    f = _front("P")
    queue, gates, done = collections.deque(), {}, []
    ls = f._lane_state()
    _note(f, "hi", 1)
    LC.ctl(f).arriving["hi"] = time.time()                            # arrive(): the preempt awaits P's floor reply
    ls.set_floor(1)
    task = _pool(f, queue, gates, done)
    await asyncio.sleep(0.45)                                         # several rounds: nothing in flight, queue empty
    assert not task.done()                                            # the phase is NOT over (probe 2b: it was)
    hi = _pend(f, "hi", 1, NOW)
    LC.stamp(f, hi)                                                   # the Pending exists ...
    queue.append(hi)                                                  # ... and is queued
    await asyncio.sleep(0.25)
    assert "hi" in gates and not task.done()                          # dispatched by the same drain, on P
    gates["hi"].set()
    assert await asyncio.wait_for(task, 3) >= 1
    assert done == ["hi"]


@_on
async def test_mutant_without_the_wait_the_empty_drain_ends_at_once_the_probe_2b_flip():
    f = _front("P")
    queue, gates, done = collections.deque(), {}, []
    _note(f, "hi", 1)
    LC.ctl(f).arriving["hi"] = time.time()
    f._lane_state().set_floor(1)
    task = _pool(f, queue, gates, done, with_wait=False)
    assert await asyncio.wait_for(task, 1) == 0                       # returns: the P->D flip follows without hi


@_on
async def test_an_arrival_that_never_gets_a_place_cannot_hold_the_phase_for_good(monkeypatch):
    f = _front("P")
    _note(f, "hi", 1)
    LC.ctl(f).arriving["hi"] = time.time() - LC.ARRIVING_MAX_S - 1.0
    f._lane_state().set_floor(1)
    assert LC.take_wait(f) is False
    LC.ctl(f).arriving["hi"] = time.time()
    assert LC.take_wait(f) is True


@_on
async def test_a_default_pool_without_the_predicate_is_unchanged_and_a_non_arrival_does_not_hold():
    async def one(p):
        return p
    queue = collections.deque()
    assert await asyncio.wait_for(_p_drain_pool(queue, 2, one, lambda p: None, lambda: True), 1) == 0
    f = _front("P")
    _note(f, "lo", 0)
    LC.ctl(f).arriving["lo"] = time.time()
    f._lane_state().set_floor(1)                                      # a lane below the floor waits at the gate: no hold
    assert await asyncio.wait_for(_p_drain_pool(queue, 2, one, lambda p: None, lambda: True,
                                                lane_take_wait=lambda: LC.take_wait(f)), 1) == 0


@_on
async def test_the_phase_stays_closed_to_the_arrival_when_the_phase_is_leaving():
    """``may_dispatch()`` False (the front is leaving ``serving``): a registered arrival does not keep the pool."""
    f = _front("P")
    _note(f, "hi", 1)
    LC.ctl(f).arriving["hi"] = time.time()
    f._lane_state().set_floor(1)

    async def one(p):
        return p
    assert await asyncio.wait_for(_p_drain_pool(collections.deque(), 2, one, lambda p: None, lambda: False,
                                                lane_take_wait=lambda: LC.take_wait(f)), 1) == 0


class _SlowFloorGroup(LaneGroup):
    """P answers the floor RPC late, as it does at the end of its pass (1667 ms in probe 2b)."""

    floor_delay = 0.0
    floor_arrived = False

    async def _lane_floor(self, request):
        self.floor_arrived = True                                     # the RPC is HERE; the answer is late
        await asyncio.sleep(self.floor_delay)
        return await super()._lane_floor(request)


async def _probe_2b(h, caplog, lo_done_after=0.25):
    """lane-0 LONG on P; lane-1 LONG arrives (its floor RPC to P is slow); the lane-0 leg ends INSIDE that window."""
    h.p.hold = {}
    h.d.hold = {}
    lo = h.post_lane("lo", chars=9000)
    assert await _until(lambda: h.p.running, 10)
    h.lo_rid = next(iter(h.p.running))
    h.p.floor_delay = 1.0
    hi = h.post_lane("hi", priority=1, chars=9000)
    assert await _until(lambda: h.p.floor_arrived, 10)               # the floor RPC is at P (its answer is slow)
    await asyncio.sleep(lo_done_after)
    h.p.release("lo")                                                 # the lane-0 leg ends: nothing in flight on P
    return lo, hi


@_e2e
async def test_e2e_probe_2b_the_lane1_leg_runs_on_p_in_the_same_phase_without_a_flip_pair(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="weg2.front")
    # the lane loop's own sweep must not be what holds the lane-0 leg: its tick (5 s here) is far behind the leg's end, so
    # the hold asserted below can only come from the synchronous sweep in ``_on_leg1_done`` (mutant M4 of the gate)
    monkeypatch.setattr(LC, "TICK_S", 5.0)
    h = LaneHarness(awake="P", d_bs=4, tp_prefill_max_tokens=1000, p_phase_max_requests=6, d_wait_bound_s=0.4,
                    idle_layout="D")
    h.p = _SlowFloorGroup("P")
    async with h:
        lo, hi = await _probe_2b(h, caplog)
        assert await _until(lambda: "gen:hi" in h.p.timeline, 15)    # the lane-1 leg is dispatched on P ...
        # ... and P did not sleep first: no P->D flip stood between the lane-0 leg's end and the lane-1 leg
        assert "rpc:release_memory_occupation" not in h.p.timeline[:h.p.timeline.index("gen:hi")]
        assert not any("W3" in r.getMessage() and "STOP" in r.getMessage() for r in caplog.records)
        assert h.lo_rid in h.front._lane_ctl_obj.held_rids()         # the lane-0 leg ended below the floor: held NOW
        assert "gen:lo" not in h.d.timeline and h.lo_rid not in h.front.groups["D"].outstanding
        h.p.release("hi")
        assert await _until(lambda: "gen:hi" in h.d.timeline, 20)    # now the P->D flip: the lane-1 request decodes
        assert "gen:lo" not in h.d.timeline                          # the held lane-0 request waits for its lane
        h.d.release("hi")
        await asyncio.wait_for(hi, 10)
        assert await _until(lambda: h.front._lane_state().lane_floor == 0, 10)
        assert await _until(lambda: "gen:lo" in h.d.timeline, 20)    # resumed after the lane ended, then it decodes
        h.p.release_all()
        h.d.release_all()
        (s, _) = await asyncio.wait_for(lo, 20)
        assert s == 200


@_e2e
async def test_e2e_mutant_without_the_wait_the_p_to_d_flip_stands_in_front_of_the_lane1_leg(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="weg2.front")
    monkeypatch.setattr(LC, "take_wait", lambda fr: False)
    h = LaneHarness(awake="P", d_bs=4, tp_prefill_max_tokens=1000, p_phase_max_requests=6, d_wait_bound_s=0.4,
                    idle_layout="D")
    h.p = _SlowFloorGroup("P")
    async with h:
        lo, hi = await _probe_2b(h, caplog)
        assert await _until(lambda: "gen:hi" in h.p.timeline, 30)
        assert "rpc:release_memory_occupation" in h.p.timeline[:h.p.timeline.index("gen:hi")]   # the flip pair
        h.p.release_all()
        h.d.release_all()
        await asyncio.wait_for(asyncio.gather(lo, hi), 30)


@_e2e
async def test_e2e_mutant_without_the_sweep_at_the_legs_end_the_held_request_is_handed_to_d(caplog, monkeypatch):
    """CAN-FAIL twin of the hold at the leg's end: the sweep called from ``_on_leg1_done`` does nothing (the lane loop's
    own sweep is kept, but its tick comes after the D admitter): the lane-0 request the front books as held reaches D
    (probe 2b: D-ADMIT 18:12:40,730 -> D's waiting_queue -> the W3)."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    real = LC.sweep

    def sweep(fr):
        if sys._getframe(1).f_code.co_name == "_on_leg1_done":
            return 0
        return real(fr)
    monkeypatch.setattr(LC, "sweep", sweep)
    monkeypatch.setattr(LC, "TICK_S", 5.0)                            # the loop does not rescue it inside the test window
    h = LaneHarness(awake="P", d_bs=4, tp_prefill_max_tokens=1000, p_phase_max_requests=6, d_wait_bound_s=0.4,
                    idle_layout="D")
    h.p = _SlowFloorGroup("P")
    async with h:
        lo, hi = await _probe_2b(h, caplog)
        assert await _until(lambda: "gen:hi" in h.p.timeline, 20)
        # the leg ended (``_probe_2b``) and nothing held it: the lane loop's tick is 5 s away -- booked held only by the sweep
        assert h.lo_rid not in h.front._lane_ctl_obj.held_rids()
        h.p.release_all()
        h.d.release_all()
        await asyncio.wait_for(asyncio.gather(lo, hi), 30)


# ================================================================== D. the marker, the P chunk park at the default
@_on
async def test_the_preempt_marker_says_what_parked_p_counts(caplog):
    caplog.set_level(logging.WARNING, logger="weg2.front")
    f = _front("P")
    f.groups["P"].outstanding["lo"] = NOW
    _note(f, "lo", 0)
    await _arrive(f, "hi", 1)
    msgs = [r.getMessage() for r in caplog.records if r.getMessage().startswith(LN.MARK_PREEMPT)]
    assert len(msgs) == 1 and "parked_p=1" in msgs[0]
    assert "NOT a confirmation" in msgs[0] and "WEG2-PARK (lane)" in msgs[0]


def test_the_default_chunk_env_does_not_prevent_the_p_chunk_park_and_a_set_one_does_not_either():
    """Metal 18:03:35Z: no 'WEG2-PARK (lane)'.  The cause was timing (the lane-0 leg was in its LAST chunk when the floor
    reached PP0: P log FINISH-TRACK before PR LANE-FLOOR), not ``SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS=0``: with the
    env unset (0) a continuation of a lane below the floor gets no chunk and parks in place; a cap changes only the size."""
    with _LanesOn() as on:                                            # the env popped: the default
        assert LN.preempt_chunk_tokens() == 0
        s = _stage(0)
        stt = lanes_p.state_of(s, create=True)
        stt.offer(1, 1)
        stt.promote()
        members, req, out = _run_chunk_decision(s)
        assert members == [] and out is req and req.weg2_lane_parked and req.weg2_pool_parked
        on.chunk(4096)
        assert LN.preempt_chunk_tokens() == 4096
        members, req, out = _run_chunk_decision(s)
        assert members == [] and req.weg2_lane_parked                 # the park is the floor's, not the cap's
