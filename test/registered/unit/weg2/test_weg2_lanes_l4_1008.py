"""PRIORITY LANES 1008, part L4 (front controller), plan deskq/PLAN-PRIO-LANES-1008.md sections 1 and 2.

Hermetic: no GPU, no network beyond loopback aiohttp test servers, every async body under ``asyncio.run``.
The groups are stubbed: the RPC names and bodies are the ones ``weg2/lanes.py`` (L1) fixes for L2 / L3
(``POST /weg2/park_running`` with ``rids`` + ``hold="lane"``, ``POST /weg2/lane_floor`` ``{floor, epoch}``);
whether the real endpoints answer that way is L5's proof, not this file's.

What is pinned:

* the transition: a higher lane preempts (floor, epoch, D's lower-lane rids parked with hold=lane, waiting ones
  held out of queue / ``_ready_for_d``, floor RPC to P and D, the marker with its counts); during a flip the
  preempt is DEFERRED to ``WEG2-FLIP done``; two requests of the lane batch (one preempt); the cascade 2 -> 1 -> 0
  resumes in the order of the ORIGINAL arrival; a request that arrives below the floor waits at the gate, with
  the client-gone exit;
* the clocks: held time counts against no wait bound (``clock_t``), the order stays the original arrival;
* P: a leg held for a higher lane stands beside the drain pool and ends no drain; the P->D flip does not wait
  for it; its stall check sees evidence; the lane park never lapses (30-s PARK-LAPSED);
* the keepalive: ``: lane-hold floor=N`` into a held open stream at an event border, never otherwise;
* the watchers: park_stuck lists no held rid, the beacon reads a lane hold;
* end to end with the real controller and fake groups: lane 1 while D decodes lane 0, a lane-1 LONG request
  through a flip pair (the lane park survives the wake), a lane-0 arrival held at the gate;
* switch off: no lane branch, no RPC, no key.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import functools
import json
import logging
import os
import sys
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import lane_ctl as LC  # noqa: E402
from sglang.srt.weg2 import lanes as LN  # noqa: E402
from sglang.srt.weg2 import park_stuck as PS  # noqa: E402
from sglang.srt.weg2 import phase_policy as PP  # noqa: E402
from sglang.srt.weg2 import progress_beacon as PB  # noqa: E402
from sglang.srt.weg2.front import Front, Pending, _p_drain_pool  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_phase_policy_h91c import FakeGroup, Harness, _until  # noqa: E402

X = 4096
NOW = 1_000_000.0


# ------------------------------------------------------------------ helpers
class Rpc:
    """The front's ``rpc`` stub: records every call, answers like L2 / L3 are specified to."""

    def __init__(self, front, park_status=200, floor_status=None):
        self.calls = []
        self.front = front
        self.park_status = park_status
        self.floor_status = floor_status or {}
        self.d_running = None  # None = D parks every rid it is asked for

    async def __call__(self, g, path, body, timeout):
        self.calls.append((g.name, path, body))
        if path == PP.PARK_PATH:
            if self.park_status != 200:
                return self.park_status, "no"
            asked = list((body or {}).get("rids") or [])
            got = asked if self.d_running is None else [r for r in asked if r in self.d_running]
            return 200, json.dumps({"parked": got, "held": []})
        if path == LN.RPC_LANE_FLOOR:
            return self.floor_status.get(g.name, 200), "{}"
        return 200, "{}"

    def paths(self, path):
        return [(g, b) for g, p, b in self.calls if p == path]


def _front(awake="D"):
    f = Front("http://p", "http://d", awake, "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    f.state = "serving"
    f.admit_d = True
    f.rpc = Rpc(f)
    return f


def _note(f, rid, lane):
    f._lane_state().note(rid, lane)


def _pend(f, rid, lane, t_arrive, **kw):
    loop = asyncio.get_event_loop()
    p = Pending(rid, "/generate", {"rid": rid}, "x", t_arrive, loop.create_future(), est_prompt=5000,
                est_uncached=5000, lane=lane, **kw)
    _note(f, rid, lane)
    return p


@contextlib.contextmanager
def _switch(on=True):
    """The switch, restored on EVERY exit (``Env.override`` restores only when the body did not raise: a failed
    test would leave the lane controller on for every test after it)."""
    had = LN.ENV_LANES in os.environ
    old = os.environ.get(LN.ENV_LANES)
    os.environ[LN.ENV_LANES] = "1" if on else "0"
    try:
        yield
    finally:
        if had:
            os.environ[LN.ENV_LANES] = old
        else:
            os.environ.pop(LN.ENV_LANES, None)


@contextlib.contextmanager
def _env(name, value):
    had = name in os.environ
    old = os.environ.get(name)
    os.environ[name] = str(value)
    try:
        yield
    finally:
        if had:
            os.environ[name] = old
        else:
            os.environ.pop(name, None)


def _on(fn):
    """Run an async test body with the switch on."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with _switch(True):
            return asyncio.run(fn(*a, **k))
    return wrapper


async def _arrive(f, rid, lane, client_gone=lambda r: False):
    _note(f, rid, lane)
    return await LC.arrive(f, types.SimpleNamespace(), rid, client_gone, _Web)


class _Web:
    @staticmethod
    def json_response(data, status=200):
        return ("json", data, status)


def _marks(caplog, mark):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(mark)]


# ------------------------------------------------------------------ the pure terms
def test_clock_t_is_the_original_arrival_plus_the_hold_and_zero_without():
    p = types.SimpleNamespace(t_arrive=100.0)
    assert LC.clock_t(p) == 100.0
    p.lane_held_s = 40.0
    assert LC.clock_t(p) == 140.0
    assert LC.clock_t(types.SimpleNamespace(), default=7.0) == 7.0
    assert LC.clock_t(types.SimpleNamespace(t_arrive=0.0), default=7.0) == 7.0
    assert Pending("r", "/g", {}, "t", 5.0, None).lane_held_s == 0.0


def test_the_lane_park_never_lapses_and_the_flip_park_still_does():
    assert PP.parked_lapsed(NOW, NOW + 31.0) is True
    assert PP.parked_lapsed(PP.LANE_PARK_STAMP, NOW + 1e9) is False
    assert PP.parked_lapsed(PP.LANE_PARK_STAMP, float("inf")) is False


def test_lane_park_body_is_the_park_body_plus_rids_hold_and_the_floor():
    b = PP.lane_park_body(7, ["a", "b"], 2, 5)
    assert b == {"epoch": 7, "reason": "lane-preempt", "rids": ["a", "b"], "hold": "lane", "floor": 2,
                 "lane_epoch": 5}
    assert PP.park_body(7, 60.0) == {"epoch": 7, "reason": "wait-bound-60s"}  # the flip park is unchanged


def test_park_stuck_lists_no_held_rid_and_a_release_starts_the_streak_again():
    ps = PS.ParkStuck()
    for _ in range(3):
        ps.note_park(["a"])
    assert ps.block(3)["stuck"] == 1 and "lane_held" not in ps.block(3)
    ps.hold(["a"])
    blk = ps.block(3)
    assert blk["stuck"] == 0 and blk["lane_held"] == 1
    ps.release(["a"])
    assert ps.block(3)["stuck"] == 0 and "lane_held" not in ps.block(3)
    ps.note_park(["a"])
    assert ps.streak["a"] == 1


def test_beacon_reads_a_lane_hold_from_the_trailer_or_the_acknowledged_rpc():
    assert PB.lane_hold(0, 0, {}, True) is None                       # no floor: no hold
    assert PB.lane_hold(2, 4, {1: (2, 4)}, False) is None             # nothing of this group is held
    assert "floor=2 epoch=4" in PB.lane_hold(2, 4, {1: (2, 4)}, True)
    assert PB.lane_hold(2, 4, {1: (2, 3)}, True) is None              # a rank that saw an older epoch is no proof
    assert "acknowledged" in PB.lane_hold(2, 4, {}, True, acked=True)  # 32-byte files: the RPC alone
    assert PB.lane_hold(2, 4, {}, True, acked=False) is None


# ------------------------------------------------------------------ preempt: the four scenarios
@_on
async def test_lane1_while_d_decodes_parks_exactly_the_lower_lane_rids_and_holds_the_waiting(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("D")
    D = f.groups["D"]
    for i in range(3):
        D.outstanding[f"d{i}"] = NOW
        _note(f, f"d{i}", 0)
    q = _pend(f, "q0", 0, NOW - 50)            # waits for P
    r = _pend(f, "r0", 0, NOW - 60)            # prefilled, waits for a D seat
    f.queue.append(q)
    f._ready_for_d.append(r)
    f._sync_batch_gate()
    assert not f._batch_gate.is_set()

    assert await _arrive(f, "hi", 1) is None
    ls = f._lane_state()
    assert (ls.lane_floor, ls.lane_epoch) == (1, 1)
    park = f.rpc.paths(PP.PARK_PATH)
    assert len(park) == 1 and park[0][0] == "D"
    assert park[0][1]["rids"] == ["d0", "d1", "d2"] and park[0][1]["hold"] == "lane"
    assert park[0][1]["floor"] == 1 and park[0][1]["lane_epoch"] == 1
    floors = sorted(g for g, _b in f.rpc.paths(LN.RPC_LANE_FLOOR))
    assert floors == ["D", "P"] and all(b == {"floor": 1, "epoch": 1} for _g, b in f.rpc.paths(LN.RPC_LANE_FLOOR))
    # the lane-0 rids are parked for the front: never lapse, out of the flip ledger, still open streams
    assert all(f._d_parked[f"d{i}"] == PP.LANE_PARK_STAMP for i in range(3))
    assert f._flip_ledger(D) == [] and sorted(D.outstanding) == ["d0", "d1", "d2"]
    # waiting ones are out of both deques; the gate of the batch is open (nothing waits for D)
    assert list(f.queue) == [] and list(f._ready_for_d) == [] and f._batch_gate.is_set()
    assert sorted(h.p.rid for h in f._lane_ctl_obj.held.values()) == ["q0", "r0"]
    line = _marks(caplog, LN.MARK_PREEMPT)[0]
    assert line.startswith("WEG2 LANE-PREEMPT floor=1 epoch=1 parked_d=3 parked_p=0 held=2 ")
    assert f.counters["lane_preempt"] == 1 and f.counters["lane_parked_d"] == 3


@_on
async def test_a_lane_park_gives_the_front_seat_back_so_the_higher_lane_can_run_on_a_bs1_d():
    """NF's D is bs1: the displaced request's seat must go back, or the lane waits for a seat it holds."""
    from sglang.srt.weg2.front import Seat

    f = _front("D")
    f.d_bs = 1
    f._d_seat = asyncio.Semaphore(1)
    await f._d_seat.acquire()
    f.groups["D"].outstanding["d0"] = NOW
    _note(f, "d0", 0)
    seat = Seat(f, "d0", "short")
    assert f._d_seat._value == 0 and seat.held
    assert await _arrive(f, "hi", 1) is None
    assert f._d_seat._value == 1 and not seat.held          # the semaphore got it back
    assert "d0" in f._park_stuck().lane_held                # and the hold survives Seat.release's streak reset
    seat.release("leg2-end")                                # the leg2 finally later: idempotent, no double release
    assert f._d_seat._value == 1


@_on
async def test_a_failing_reconcile_never_stops_the_controllers_pass(caplog):
    f = _front("D")

    async def boom(*a, **k):
        raise RuntimeError("lane layer bug")
    orig = LC.reconcile
    LC.reconcile = boom
    try:
        await f._lane_reconcile("controller")                # logged, swallowed
    finally:
        LC.reconcile = orig
    assert any("LANE reconcile" in r.getMessage() for r in caplog.records)


@_on
async def test_lane1_while_p_prefills_sends_the_floor_and_counts_the_legs_it_stops(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("P")
    P = f.groups["P"]
    for i in range(2):
        P.outstanding[f"p{i}"] = NOW
        _note(f, f"p{i}", 0)
    q = _pend(f, "q0", 0, NOW - 5)
    f.queue.append(q)
    assert await _arrive(f, "hi", 1) is None
    assert f.rpc.paths(PP.PARK_PATH) == []      # D sleeps and holds nothing: nothing to park
    assert sorted(g for g, _b in f.rpc.paths(LN.RPC_LANE_FLOOR)) == ["D", "P"]
    assert _marks(caplog, LN.MARK_PREEMPT)[0].startswith(
        "WEG2 LANE-PREEMPT floor=1 epoch=1 parked_d=0 parked_p=2 held=1 ")
    # the P->D flip's drain does not wait for the legs held for the lane, but for the floor lane's own
    P.outstanding["hi"] = NOW
    assert f._flip_ledger(P) == ["hi"]


@_on
async def test_a_sleeping_d_is_not_asked_to_park_its_dormant_requests_the_front_books_the_hold_unconfirmed():
    """Review L4 FR1 finding 4: a booking for a sleeping D is UNCONFIRMED (``dormant``) until a park RPC after the
    wake is answered; only that answer moves it to ``parked_d``."""
    f = _front("P")
    f.groups["D"].outstanding["dd"] = NOW
    _note(f, "dd", 0)
    assert await _arrive(f, "hi", 1) is None
    assert f.rpc.paths(PP.PARK_PATH) == []
    lc = f._lane_ctl_obj
    assert f._d_parked["dd"] == PP.LANE_PARK_STAMP and "dd" in lc.dormant and "dd" not in lc.parked_d
    assert f.counters["lane_parked_d_dormant"] == 1 and f.counters["lane_parked_d"] == 0
    assert "dd" in f._lane_parked_set()                       # still out of the flip ledger / the wake counts
    assert f._flip_ledger(f.groups["D"]) == []
    # nothing asks a sleeping D, however often the loop ticks
    await LC.step(f, time.time() + 10)
    assert f.rpc.paths(PP.PARK_PATH) == []


@_on
async def test_after_the_wake_the_dormant_booking_is_parked_by_rid_and_booked_only_on_d_s_answer(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("P")
    f.groups["D"].outstanding["dd"] = NOW
    _note(f, "dd", 0)
    assert await _arrive(f, "hi", 1) is None
    lc = f._lane_ctl_obj
    assert _marks(caplog, LN.MARK_PREEMPT)[0].startswith("WEG2 LANE-PREEMPT floor=1 epoch=1 parked_d=0 parked_p=0 held=0 dormant=1 ")
    # the P->D wake: the lane parks stay booked over it (the flip's PARK-RESUME must not resume them)
    parked = dict(f._d_parked)
    assert f._lane_unpark_keep(parked) == {"dd": PP.LANE_PARK_STAMP} and parked == {}
    f.awake = "D"
    # D answers the park without it ("dd" is not running there): still dormant, asked again
    f.rpc.d_running = set()
    await LC.step(f, time.time() + 1)
    assert [b["rids"] for _g, b in f.rpc.paths(PP.PARK_PATH)] == [["dd"]]
    assert "dd" in lc.dormant and "dd" not in lc.parked_d
    f.rpc.d_running = None
    await LC.step(f, time.time() + 3)
    assert [b["rids"] for _g, b in f.rpc.paths(PP.PARK_PATH)] == [["dd"], ["dd"]] and f.rpc.paths(PP.PARK_PATH)[1][1]["hold"] == "lane"
    assert "dd" in lc.parked_d and "dd" not in lc.dormant and f.counters["lane_parked_d"] == 1
    await LC.step(f, time.time() + 6)                                    # confirmed: not asked a third time
    assert len(f.rpc.paths(PP.PARK_PATH)) == 2
    # the lane ends: the confirmed park resumes
    f._lane_end("hi")
    await LC.reconcile(f, "end")
    assert "dd" not in lc.parked_d and "dd" not in f._d_parked


@_on
async def test_a_dormant_booking_ends_with_the_lane_if_the_floor_falls_before_the_wake():
    f = _front("P")
    f.groups["D"].outstanding["dd"] = NOW
    _note(f, "dd", 0)
    assert await _arrive(f, "hi", 1) is None
    f._lane_end("hi")
    await LC.reconcile(f, "end")
    lc = f._lane_ctl_obj
    assert "dd" not in lc.dormant and "dd" not in f._d_parked and f.counters["lane_resumed_d"] == 1


@_on
async def test_lane1_arriving_during_a_flip_is_deferred_until_flip_done(caplog):
    """User decision 2: the flip runs to its end, then the lane displaces; no flip is aborted."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("D")
    f.state = "flipping"
    f._flip_dst = "P"
    D = f.groups["D"]
    D.outstanding["d0"] = NOW
    _note(f, "d0", 0)
    assert await _arrive(f, "hi", 1) is None      # not held: it goes its normal way (queues)
    ls = f._lane_state()
    assert ls.lane_floor == 0 and f.rpc.calls == []
    assert len(_marks(caplog, LN.MARK_DEFER)) == 1
    await LC.reconcile(f, "tick")                 # still flipping: one DEFER line per arrival, not per pass
    assert len(_marks(caplog, LN.MARK_DEFER)) == 1
    f.state = "serving"                           # WEG2-FLIP done
    await LC.step(f)
    assert ls.lane_floor == 1 and f.rpc.paths(PP.PARK_PATH)[0][1]["rids"] == ["d0"]
    assert _marks(caplog, LN.MARK_PREEMPT)[0].startswith("WEG2 LANE-PREEMPT floor=1 epoch=1 parked_d=1")


@_on
async def test_the_once_per_phase_latches_do_not_block_a_lane_change_inside_the_phase():
    """Plan 6 / task (4): ``_park_attempt_epoch`` (one flip park per D phase) and ``admit_d=False`` (closed for the
    phase) are the flip park's latches; the lane park is its own RPC and never reads or writes them."""
    f = _front("D")
    f.epoch = 7
    f._park_attempt_epoch = 7            # the phase's flip park was spent ...
    f.admit_d = False                    # ... and admission to D is closed for the phase
    f.groups["D"].outstanding["d0"] = NOW
    _note(f, "d0", 0)
    assert await _arrive(f, "hi", 1) is None
    assert f.rpc.paths(PP.PARK_PATH)[0][1]["rids"] == ["d0"]      # parked all the same
    assert (f._park_attempt_epoch, f.admit_d) == (7, False)       # latches untouched by the preempt
    f._lane_end("hi")
    await LC.reconcile(f, "end")
    assert f._lane_state().lane_floor == 0 and (f._park_attempt_epoch, f.admit_d) == (7, False)
    assert await _arrive(f, "hi2", 1) is None                     # a second lane change in the same phase
    assert len(f.rpc.paths(PP.PARK_PATH)) == 2


@_on
async def test_two_lane1_requests_batch_and_the_lane_ends_with_the_last_of_them(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("D")
    f.groups["D"].outstanding["d0"] = NOW
    _note(f, "d0", 0)
    assert await _arrive(f, "hi1", 1) is None
    assert await _arrive(f, "hi2", 1) is None            # same lane: normal path, no second preempt
    assert len(_marks(caplog, LN.MARK_PREEMPT)) == 1 and len(f.rpc.paths(PP.PARK_PATH)) == 1
    f._lane_end("hi1")
    await LC.reconcile(f, "end")
    assert f._lane_state().lane_floor == 1               # one of the lane is left
    f._lane_end("hi2")
    await LC.reconcile(f, "end")
    assert f._lane_state().lane_floor == 0 and len(_marks(caplog, LN.MARK_RESUME)) == 1


@_on
async def test_cascade_2_1_0_resumes_in_the_order_of_the_original_arrival(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("D")
    D = f.groups["D"]
    D.outstanding["d0"] = NOW
    _note(f, "d0", 0)
    a = _pend(f, "A", 0, NOW - 100)
    c = _pend(f, "C", 0, NOW - 300)
    b = _pend(f, "B", 1, NOW - 200)
    for p in (a, c, b):
        f.queue.append(p)
    assert await _arrive(f, "one", 1) is None            # floor 1: lane 0 held
    assert sorted(p.rid for p in f.queue) == ["B"] and f._lane_state().lane_floor == 1
    D.outstanding["one"] = NOW
    assert await _arrive(f, "two", 2) is None            # floor 2: lane 1 held too, "one" parked
    assert f._lane_state().lane_floor == 2 and list(f.queue) == []
    assert f.rpc.paths(PP.PARK_PATH)[-1][1]["rids"] == ["one"]
    assert sorted(f._lane_ctl_obj.parked_d) == ["d0", "one"]
    # lane 2 ends: floor falls to 1, lane 1 resumes (B back, "one" runs again), lane 0 stays held
    f._lane_end("two")
    await LC.reconcile(f, "end")
    assert f._lane_state().lane_floor == 1
    assert [p.rid for p in f.queue] == ["B"] and "one" not in f._d_parked and f._d_parked["d0"] == PP.LANE_PARK_STAMP
    assert sorted(h.p.rid for h in f._lane_ctl_obj.held.values()) == ["A", "C"]
    res = _marks(caplog, LN.MARK_RESUME)
    assert res[0].startswith("WEG2 LANE-RESUME floor=2->1 epoch=3 resumed=")
    # lane 1 ends: floor 0, lane 0 back ORDERED BY ITS ORIGINAL ARRIVAL (C, A), then the D park goes
    f._lane_end("one")
    f._lane_end(b.rid)
    f.queue.clear()
    await LC.reconcile(f, "end")
    assert f._lane_state().lane_floor == 0
    assert [p.rid for p in f.queue] == ["C", "A"] and f._d_parked == {} and f._lane_ctl_obj.parked_d == {}
    assert res and _marks(caplog, LN.MARK_RESUME)[1].startswith("WEG2 LANE-RESUME floor=1->0 epoch=4 resumed=3 ")
    assert [floor for _g, b_ in f.rpc.paths(LN.RPC_LANE_FLOOR) for floor in [b_["floor"]]][-2:] == [0, 0]


@_on
async def test_held_time_counts_against_no_clock_and_the_order_is_the_original_arrival():
    f = _front("D")
    f.groups["D"].outstanding["d0"] = NOW
    _note(f, "d0", 0)
    old = _pend(f, "old", 0, time.time() - 200)
    f.queue.append(old)
    assert await _arrive(f, "hi", 1) is None
    h = next(iter(f._lane_ctl_obj.held.values()))
    h.t_held = time.time() - 150                           # it has been held for 150 s
    f._lane_end("hi")
    await LC.reconcile(f, "end")
    assert list(f.queue) == [old] and 149 < old.lane_held_s < 152
    # ORDER: the original arrival; CLOCKS: counted from the resume, 150 s of hold taken off
    assert time.time() - old.t_arrive > 199
    assert 45 < time.time() - LC.clock_t(old) < 52
    # the wait bound of the D phase reads the clock, not the arrival
    wait = PP.oldest_d_phase_wait((LC.clock_t(p) for p in f.queue), time.time() - 1000, time.time())
    assert not PP.wait_bound_fired(wait, 60.0)
    assert PP.wait_bound_fired(PP.oldest_d_phase_wait((p.t_arrive for p in f.queue), time.time() - 1000,
                                                      time.time()), 60.0)
    # a newcomer older than the resumed one in clock but younger in order sorts behind it
    new = _pend(f, "new", 0, time.time() - 10)
    f.queue.append(new)
    assert [p.rid for p in sorted(f.queue, key=lambda q: q.t_arrive)] == ["old", "new"]


# ------------------------------------------------------------------ the gate
@_on
async def test_an_arrival_below_the_floor_waits_at_the_gate_and_is_released_after_the_parked(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("D")
    held_q = _pend(f, "A", 0, NOW - 100)
    f.queue.append(held_q)
    assert await _arrive(f, "hi", 1) is None

    order = []

    async def late_arrival():
        out = await _arrive(f, "late", 0)
        order.append(("late", [p.rid for p in f.queue]))
        return out

    task = asyncio.ensure_future(late_arrival())
    await asyncio.sleep(0.05)
    assert not task.done() and "late" in f._lane_ctl_obj.gates          # waits, nothing routed
    assert f._lane_state().counts(pending=[*f._lane_ctl_obj.gates])["0"]["pending"] == 1
    f._lane_end("hi")
    await LC.reconcile(f, "end")
    assert await asyncio.wait_for(task, 2) is None
    # the parked first: A was back in the queue when the gate let "late" go
    assert order == [("late", ["A"])]
    # the stamp: original arrival as the order, the hold as the clock offset
    p = _pend(f, "late", 0, time.time())
    LC.stamp(f, p)
    assert p.lane_held_s > 0.04 and p.t_arrive < time.time() - 0.04
    assert any("WEG2 LANE-ARRIVE rid=late lane=0 floor=1->1" in m for m in
               [r.getMessage() for r in caplog.records])


@_on
async def test_a_client_that_leaves_while_held_at_the_gate_is_answered_499_and_leaves_the_register():
    f = _front("D")
    assert await _arrive(f, "hi", 1) is None
    gone = {"v": False}
    task = asyncio.ensure_future(_arrive(f, "late", 0, client_gone=lambda r: gone["v"]))
    await asyncio.sleep(0.05)
    assert not task.done()
    gone["v"] = True
    out = await asyncio.wait_for(task, LC.GATE_TICK_S + 1.5)
    assert out[0] == "json" and out[2] == 499 and "state=lane-hold" in out[1]["error"]
    assert "late" not in f._lane_ctl_obj.gates and f.counters["client_gone_lane_hold"] == 1


@_on
async def test_the_lane_requests_client_gone_empties_the_lane_through_its_handler_end():
    """H102 / plan 8: the floor falls when the handler of the last lane request ends (any reason)."""
    f = _front("D")
    f.groups["D"].outstanding["d0"] = NOW
    _note(f, "d0", 0)
    assert await _arrive(f, "hi", 1) is None
    assert f._lane_state().lane_floor == 1
    f._lane_end("hi")                  # what ipc_out_wrap's finally does when the handler ended on a hang-up
    await LC.step(f)
    assert f._lane_state().lane_floor == 0 and "d0" not in f._d_parked


# ------------------------------------------------------------------ park failures never block the lane
@_on
async def test_an_old_d_without_the_park_endpoint_never_blocks_the_higher_lane(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("D")
    f.rpc.park_status = 404
    f.groups["D"].outstanding["d0"] = NOW
    _note(f, "d0", 0)
    assert await _arrive(f, "hi", 1) is None
    assert f._lane_state().lane_floor == 1 and f._lane_ctl_obj.park_unsupported
    assert "d0" not in f._d_parked
    assert any("LANE-PARK-UNSUPPORTED" in r.getMessage() for r in caplog.records)
    n = len(f.rpc.paths(PP.PARK_PATH))
    await LC.step(f, NOW + 100)
    assert len(f.rpc.paths(PP.PARK_PATH)) == n          # no retry against an endpoint that is not there


@_on
async def test_a_straggler_below_the_floor_is_parked_by_the_next_pass_and_a_failed_park_is_retried():
    f = _front("D")
    f.rpc.park_status = 500
    f.groups["D"].outstanding["d0"] = NOW
    _note(f, "d0", 0)
    assert await _arrive(f, "hi", 1) is None
    assert f._lane_ctl_obj.parked_d == {}                 # the park failed; the lane was not blocked
    f.rpc.park_status = 200
    await LC.step(f, time.time() + 5)
    assert "d0" in f._lane_ctl_obj.parked_d and f._d_parked["d0"] == PP.LANE_PARK_STAMP
    # a hand-off that reaches D after the park is a straggler of the same kind
    f.groups["D"].outstanding["late"] = NOW
    _note(f, "late", 0)
    await LC.step(f, time.time() + 10)
    assert "late" in f._lane_ctl_obj.parked_d


@_on
async def test_a_d_that_ignores_the_rids_filter_parks_the_floor_lane_too_and_it_is_not_booked_as_a_lane_park():
    f = _front("D")
    f.groups["D"].outstanding.update({"d0": NOW, "hi": NOW})
    _note(f, "d0", 0)
    _note(f, "hi", 1)

    async def rpc(g, path, body, timeout):
        f.rpc.calls.append((g.name, path, body))
        if path == PP.PARK_PATH:
            return 200, json.dumps({"parked": ["d0", "hi"], "held": []})
        return 200, "{}"
    f.rpc = rpc
    f.rpc.calls = []
    LC.ctl(f)
    f._lane_state().note("x", 1)
    await LC.reconcile(f, "tick")
    assert f._d_parked["d0"] == PP.LANE_PARK_STAMP
    assert f._d_parked["hi"] != PP.LANE_PARK_STAMP and "hi" not in f._lane_ctl_obj.parked_d
    assert f.counters["lane_park_overreach"] == 1


@_on
async def test_a_group_without_the_floor_endpoint_is_named_once_and_not_asked_again():
    f = _front("D")
    f.rpc.floor_status = {"P": 404}
    assert await _arrive(f, "hi", 1) is None
    assert f._lane_ctl_obj.unsupported == {"P"} and f._lane_ctl_obj.acked == {"D": (1, 1)}
    n = len(f.rpc.paths(LN.RPC_LANE_FLOOR))
    await LC.step(f, time.time() + 10)
    assert len(f.rpc.paths(LN.RPC_LANE_FLOOR)) == n


@_on
async def test_an_unacknowledged_floor_is_asked_again_and_a_wake_message_carries_it():
    f = _front("D")
    f.rpc.floor_status = {"D": 503}
    assert await _arrive(f, "hi", 1) is None
    assert f._lane_ctl_obj.acked == {"P": (1, 1)}
    f.rpc.floor_status = {}
    await LC.step(f, time.time() + 2)
    assert f._lane_ctl_obj.acked == {"P": (1, 1), "D": (1, 1)}
    assert f._lane_wake_fields() == {"lane_floor": 1, "lane_epoch": 1}


# ------------------------------------------------------------------ the P side hooks
@_on
async def test_the_p_drain_pool_dispatches_the_lane_beside_held_legs_and_ends_without_them():
    f = _front("P")
    queue = collections.deque()
    done_order = []
    gates = {}

    async def one(p):
        gates.setdefault(p.rid, asyncio.Event())
        await gates[p.rid].wait()
        return p

    def on_done(p):
        done_order.append(p.rid)

    loop = asyncio.get_event_loop()
    lo = [_pend(f, f"lo{i}", 0, NOW + i) for i in range(2)]
    queue.extend(lo)
    ls = f._lane_state()
    task = asyncio.ensure_future(_p_drain_pool(
        queue, 2, one, on_done, lambda: True, lane_held=lambda p: p.lane < ls.lane_floor))
    await asyncio.sleep(0.05)
    assert sorted(gates) == ["lo0", "lo1"]                   # limit 2: both in flight
    hi = _pend(f, "hi", 1, NOW + 10)
    ls.set_floor(1)
    queue.append(hi)
    await asyncio.sleep(0.25)
    assert "hi" in gates                                     # dispatched beside the two held legs
    assert not task.done()
    gates["hi"].set()
    assert await asyncio.wait_for(task, 2) >= 1              # the drain ends: only held legs are left
    assert done_order == ["hi"]
    # the held legs finish later: their hand-over runs from the callback
    ls.set_floor(0)
    gates["lo0"].set()
    gates["lo1"].set()
    await asyncio.sleep(0.05)
    assert sorted(done_order) == ["hi", "lo0", "lo1"]
    assert loop is not None


def test_the_p_drain_pool_is_the_old_pool_without_lane_held():
    async def body():
        queue = collections.deque([types.SimpleNamespace(rid=f"r{i}") for i in range(3)])
        seen = []

        async def one(p):
            await asyncio.sleep(0.01)
            return p

        await _p_drain_pool(queue, 2, one, lambda p: seen.append(p.rid), lambda: True)
        assert sorted(seen) == ["r0", "r1", "r2"]

    asyncio.run(body())


@_on
async def test_a_leg_held_for_a_higher_lane_is_no_stall():
    f = _front("P")
    f._lane_state().set_floor(2)
    held = _pend(f, "p0", 0, NOW)
    live = _pend(f, "p1", 2, NOW)
    assert f._lane_p_held(held) is True and f._lane_p_held(live) is False
    # the bounded POST: with evidence refreshed every second the leg is never named stalled
    f.p_leg1_stall_s = 0.4

    async def post():
        await asyncio.sleep(1.6)
        return 200, b"{}"

    async def stalled_progress(g):
        return {"forward_ct": 1}
    f._weg2_decode_progress = stalled_progress
    out = await f._leg1_bounded(held, post())
    assert out == (200, b"{}")
    f.p_leg1_stall_s = 0.4
    with pytest.raises(RuntimeError, match="INTAKE-STALL|intake"):
        await f._leg1_bounded(live, post())


@_on
async def test_the_wake_keeps_the_lane_parks_and_counts_only_the_flip_parks():
    f = _front("D")
    f.groups["D"].outstanding.update({"lane": NOW, "flip": NOW, "run": NOW})
    for r in ("lane", "flip", "run"):
        _note(f, r, 0)
    f._d_parked["lane"] = PP.LANE_PARK_STAMP
    f._d_parked["flip"] = time.time()
    f._lane_state().set_floor(1)
    LC.ctl(f).parked_d["lane"] = NOW
    fields = f._wake_handoff_fields("D")
    assert fields["parked_n"] == 1 and fields["handoff_n"] == 1     # "run"; the lane park is neither
    assert fields["lane_floor"] == 1 and fields["lane_epoch"] == 1
    keep = f._lane_unpark_keep(f._d_parked)
    assert list(keep) == ["lane"] and list(f._d_parked) == ["flip"]


# ------------------------------------------------------------------ keepalive
class _Resp:
    def __init__(self, fail=False):
        self.written = []
        self.fail = fail

    async def write(self, b):
        if self.fail:
            raise ConnectionResetError("gone")
        self.written.append(bytes(b))


@_on
async def test_keepalive_writes_one_sse_comment_into_a_held_open_stream_at_an_event_border():
    f = _front("D")
    f._lane_state().set_floor(3)
    _note(f, "held", 1)
    _note(f, "live", 3)
    r_held, r_live, r_mid = _Resp(), _Resp(), _Resp()
    e_held = f._lane_stream_open("held", None, r_held)
    f._lane_stream_open("live", None, r_live)
    _note(f, "mid", 0)
    e_mid = f._lane_stream_open("mid", None, r_mid)
    e_mid["boundary"] = False                                  # stopped inside an SSE event
    with _env(LN.ENV_KEEPALIVE_S, 5):
        t0 = time.time()
        assert await LC.keepalive_tick(f, t0 + 1) == 0          # quiet for 1 s of 5
        assert await LC.keepalive_tick(f, t0 + 6) == 1
        assert r_held.written == [b": lane-hold floor=3\n\n"] and r_live.written == [] and r_mid.written == []
        assert await LC.keepalive_tick(f, t0 + 7) == 0          # the period counts from the last write
        assert await LC.keepalive_tick(f, t0 + 12) == 1 and len(r_held.written) == 2
    with _env(LN.ENV_KEEPALIVE_S, 0):
        assert await LC.keepalive_tick(f, t0 + 100) == 0        # 0 = no keepalive
    assert f.counters["lane_keepalives"] == 2 and e_held["resp"] is r_held


@_on
async def test_keepalive_to_a_client_that_left_drops_the_stream_and_the_floor_off_writes_nothing():
    f = _front("D")
    f._lane_state().set_floor(2)
    _note(f, "held", 0)
    f._lane_stream_open("held", None, _Resp(fail=True))
    with _env(LN.ENV_KEEPALIVE_S, 1):
        assert await LC.keepalive_tick(f, time.time() + 5) == 0
    assert "held" not in f._lane_ctl_obj.streams
    f._lane_state().set_floor(0)
    f._lane_stream_open("held", None, _Resp())
    with _env(LN.ENV_KEEPALIVE_S, 1):
        assert await LC.keepalive_tick(f, time.time() + 5) == 0


# ------------------------------------------------------------------ watchers and state.json
@_on
async def test_state_block_counts_held_requests_in_their_lane_and_shows_the_hold():
    f = _front("D")
    f.groups["D"].outstanding["d0"] = NOW
    _note(f, "d0", 0)
    q = _pend(f, "q0", 0, NOW - 5)
    f.queue.append(q)
    assert f._lane_block()["lanes"]["0"] == {"pending": 1, "running_p": 0, "running_d": 1, "parked": 0}
    assert "lane_hold" not in f._lane_block()
    assert await _arrive(f, "hi", 1) is None
    blk = f._lane_block()
    assert blk["lane_floor"] == 1 and blk["lane_epoch"] == 1
    assert blk["lanes"]["0"] == {"pending": 1, "running_p": 0, "running_d": 0, "parked": 1}
    assert blk["lane_hold"]["held"] == 1 and blk["lane_hold"]["parked_d"] == 1
    assert blk["lane_hold"]["acked"] == {"D": [1, 1], "P": [1, 1]}


@_on
async def test_the_group_health_poll_reads_a_lane_hold_as_progress():
    f = _front("D")
    LC.ctl(f)
    g = f.groups["D"]
    g.outstanding["d0"] = NOW
    _note(f, "d0", 0)
    assert f._lane_health_hold(g, "") is None                # floor 0: no hold
    assert await _arrive(f, "hi", 1) is None
    why = f._lane_health_hold(g, "")
    assert why is not None and why.startswith("lane-hold floor=1 epoch=1")


@_on
async def test_lane_reprefill_counts_a_resume_without_prefix_of_a_held_rid_only(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("D")
    f.groups["D"].outstanding["d0"] = NOW
    _note(f, "d0", 0)
    assert await _arrive(f, "hi", 1) is None
    f._lane_reprefill("never-held")
    assert f.counters["lane_reprefill"] == 0
    f._lane_reprefill("d0")
    assert f.counters["lane_reprefill"] == 1
    assert _marks(caplog, LN.MARK_REPREFILL)[0].startswith("WEG2 LANE-REPREFILL rid=d0 lane=0 floor=1")


# ------------------------------------------------------------------ switch off
def test_switch_off_no_lane_branch_no_rpc_no_key_no_register(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")

    async def body():
        f = _front("D")
        f.groups["D"].outstanding["d0"] = NOW
        assert await f._lane_arrive(types.SimpleNamespace(), "hi", {"priority": 5}) is None
        await f._lane_reconcile("controller")
        await LC.step(f)
        assert f.rpc.calls == [] and "_lane_ctl_obj" not in f.__dict__ and "_lane_state_obj" not in f.__dict__
        assert f._lane_block() == {} and f._lane_wake_fields() == {}
        assert f._lane_stream_open("r", None, _Resp()) is None
        assert f._flip_ledger(f.groups["P"]) == []
        p = Pending("r", "/g", {}, "t", 5.0, None)
        assert f._lane_stamp(p) is p and p.t_arrive == 5.0 and p.lane_held_s == 0.0
        f._lane_reprefill("d0")
        assert f.counters["lane_reprefill"] == 0
        assert f._lane_held_pendings() == [] and f._lane_parked_set() == set()
        assert f._lane_unpark_keep({"a": 1.0}) == {}

    asyncio.run(body())
    assert not [r for r in caplog.records if "LANE" in r.getMessage()]


def test_switch_off_the_state_of_a_front_is_byte_identical_and_park_stuck_has_no_new_key():
    f = _front("D")
    assert "lane_hold" not in f._lane_block() and "lane_hold" not in PS.ParkStuck().block(3)
    assert "lane_floor" not in f.state_dict()


# ------------------------------------------------------------------ end to end: the real controller, fake groups
class LaneGroup(FakeGroup):
    """FakeGroup that answers the two lane RPCs like L2 / L3 are specified to and can stream SSE."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.floor_bodies = []

    async def _lane_floor(self, request):
        body = await request.json()
        self.floor_bodies.append(body)
        self.timeline.append(f"rpc:lane_floor={body['floor']}")
        return __import__("aiohttp").web.json_response({"ok": True})

    async def _park(self, request):
        body = await request.json()
        self.park_bodies.append(body)
        self.timeline.append("rpc:weg2/park_running")
        rids = body.get("rids")
        got = sorted(r for r in self.running if rids is None or r in rids)
        return __import__("aiohttp").web.json_response({"parked": got})

    async def start(self):
        await super().start()
        # the aiohttp app is frozen once the server runs: a fresh app carries the extra route
        await self.server.close()
        web = __import__("aiohttp").web
        from aiohttp.test_utils import TestServer
        app = web.Application()
        app.router.add_post("/generate", self._generate)
        app.router.add_post("/flush_cache", self._ok)
        app.router.add_post("/release_memory_occupation", self._ok)
        app.router.add_post("/resume_memory_occupation", self._resume)
        app.router.add_post("/abort_request", self._ok)
        app.router.add_get("/get_server_info", self._info)
        app.router.add_post(PP.PARK_PATH, self._park)
        app.router.add_post(LN.RPC_LANE_FLOOR, self._lane_floor)
        self.server = TestServer(app)
        await self.server.start_server()
        self.url = str(self.server.make_url("")).rstrip("/")


class LaneHarness(Harness):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.p = LaneGroup("P")
        self.d = LaneGroup("D")

    async def __aenter__(self):
        await super().__aenter__()
        self.tasks.append(asyncio.create_task(LC.loop(self.front)))
        return self

    def post_lane(self, mark, priority=None, chars=300):
        body = {"prompt": f"MARK{mark} " + mark * chars}
        if priority is not None:
            body["priority"] = priority
        t = asyncio.create_task(self._post(body))
        self.posts.append(t)
        return t


def _e2e(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        # the served front wraps the generate handler in ipc_out_wrap (main()): its finally takes the rid out of
        # the lane register -- the harness registers the bare handler, so the same wrap is put on the class
        orig = Front.handle_generate

        async def wrapped(self, request):
            return await self.ipc_out_wrap(functools.partial(orig, self))(request)

        Front.handle_generate = wrapped
        try:
            with _switch(True):
                asyncio.run(fn(*a, **k))
        finally:
            Front.handle_generate = orig
    return wrapper


@_e2e
async def test_e2e_lane1_while_d_decodes_lane0_parks_runs_alone_and_the_parked_resume():
    async with LaneHarness(awake="D", d_bs=4, tp_prefill_max_tokens=100_000) as h:
        h.d.hold = {}
        lo = [h.post_lane(f"lo{i}") for i in range(3)]
        assert await _until(lambda: len(h.d.running) == 3, 10)
        lo_rids = sorted(h.d.running)
        # the front's own ledger of D must hold all three too before lane 1 arrives (D's fake sees the POST a moment
        # before the front books it): a park of two of them is the harness race, not the controller
        assert await _until(lambda: sorted(h.front.groups["D"].outstanding) == lo_rids, 10)
        hi = h.post_lane("hi", priority=1)
        assert await _until(lambda: "gen:hi" in h.d.timeline, 10)
        # the park (exactly the three lane-0 rids, hold=lane) came BEFORE the lane-1 request reached D
        assert h.d.park_bodies and sorted(h.d.park_bodies[0]["rids"]) == lo_rids and h.d.park_bodies[0]["hold"] == "lane"
        assert h.d.timeline.index("rpc:weg2/park_running") < h.d.timeline.index("gen:hi")
        assert {"floor": 1, "epoch": 1} in h.d.floor_bodies and {"floor": 1, "epoch": 1} in h.p.floor_bodies
        assert all(h.front._d_parked.get(r) == PP.LANE_PARK_STAMP for r in lo_rids)
        # the lane-0 streams are open and un-re-posted: no second leg 1, no second leg 2
        assert not any(t.done() for t in lo) and [m for m in h.d.gen_marks if m.startswith("lo")].count("lo0") == 1
        h.d.release("hi")
        (s, _), = await asyncio.wait_for(asyncio.gather(hi), 10)
        assert s == 200
        # the lane ends with its handler: floor falls to 0, D is told, the lane-0 parks end
        assert await _until(lambda: h.front._lane_state().lane_floor == 0 and not h.front._d_parked, 10)
        assert await _until(lambda: {"floor": 0, "epoch": 2} in h.d.floor_bodies, 10)   # D is told, then resumes
        h.d.release_all()
        res = await asyncio.wait_for(asyncio.gather(*lo), 10)
        assert [s for s, _ in res] == [200, 200, 200]
        assert h.front.counters["lane_preempt"] == 1 and h.front.counters["lane_resume"] == 1


@_e2e
async def test_e2e_a_lane0_arrival_waits_at_the_gate_until_the_lane1_request_is_done():
    async with LaneHarness(awake="D", d_bs=4, tp_prefill_max_tokens=100_000) as h:
        h.d.hold = {}
        hi = h.post_lane("hi", priority=1)
        assert await _until(lambda: "gen:hi" in h.d.timeline, 10)
        lo = h.post_lane("lo")
        await asyncio.sleep(0.4)
        assert "gen:lo" not in h.d.timeline and "lo" not in h.p.gen_marks     # held at the gate: not routed
        assert len(h.front._lane_ctl_obj.gates) == 1
        h.d.release("hi")
        await asyncio.wait_for(hi, 10)
        assert await _until(lambda: "gen:lo" in h.d.timeline, 10)
        assert h.d.timeline.index("done:hi") < h.d.timeline.index("gen:lo")
        h.d.release_all()
        (s, _) = await asyncio.wait_for(lo, 10)
        assert s == 200


@_e2e
async def test_e2e_a_lane1_long_request_flips_without_waiting_for_the_parked_and_the_wake_keeps_the_park():
    async with LaneHarness(awake="D", d_bs=4, tp_prefill_max_tokens=1000, d_wait_bound_s=0.4,
                           p_phase_max_requests=6) as h:
        h.d.hold = {}
        lo = h.post_lane("lo", chars=300)                       # SHORT: decodes on D
        assert await _until(lambda: h.d.running, 10)
        lo_rid = next(iter(h.d.running))
        hi = h.post_lane("hi", priority=1, chars=9000)          # LONG: needs P
        # the lane park, then the flip D->P is NOT held up by the parked lane-0 decode, P prefills hi
        assert await _until(lambda: "gen:hi" in h.p.timeline, 20)
        assert h.d.park_bodies[0]["rids"] == [lo_rid] and h.d.park_bodies[0]["hold"] == "lane"
        assert h.front._d_parked[lo_rid] == PP.LANE_PARK_STAMP
        # ... and the flip back to D: the wake carries hi alone (handoff_n 1), the lane park is not "resumed"
        assert await _until(lambda: "gen:hi" in h.d.timeline, 20)
        # (the leg 2 may reach D a moment before the wake RPC is recorded: wait for the wake, never read it early --
        # the same test flaked 2 in 8 on 572f86f3f3 for exactly that)
        assert await _until(lambda: any("kv_cache" in (b.get("tags") or []) for b in h.d.resume_bodies), 10)
        kv = [b for b in h.d.resume_bodies if "kv_cache" in (b.get("tags") or [])]
        assert kv and kv[-1].get("parked_n") == 0 and kv[-1].get("handoff_n") == 1, kv
        assert kv[-1].get("lane_floor") == 1
        assert h.front._d_parked.get(lo_rid) == PP.LANE_PARK_STAMP       # survived the P->D wake
        assert not lo.done()
        h.d.release("hi")
        await asyncio.wait_for(hi, 10)
        assert await _until(lambda: h.front._lane_state().lane_floor == 0 and lo_rid not in h.front._d_parked, 10)
        h.d.release_all()
        (s, _) = await asyncio.wait_for(lo, 10)
        assert s == 200
        assert h.d.gen_marks.count("lo") == 1                            # never re-posted: a resume, not a redo


# ------------------------------------------------------------------ review FR1: held P legs leave P before the P->D flip
@_on
async def test_the_pool_takes_the_held_legs_off_p_when_only_they_are_left_and_returns_without_them(caplog):
    """Review L4 FR1 finding 1: the held legs must not stay behind on P (they would sit in P's waiting queue, the
    P->D witness would read 'rank not idle' = W3, and nothing would run them after the floor fell)."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("P")
    queue = collections.deque()
    done_order = []
    gates = {}

    async def one(p):
        ev = gates.setdefault(p.rid, asyncio.Event())
        try:
            await ev.wait()
        except asyncio.CancelledError:
            if p.lane_p_taken:
                return p                                          # what the controller's ``one`` does
            raise
        return p

    def on_done(p):
        if not p.lane_p_taken:
            done_order.append(p.rid)

    lo = [_pend(f, f"lo{i}", 0, NOW + i) for i in range(2)]
    queue.extend(lo)
    ls = f._lane_state()
    task = asyncio.ensure_future(_p_drain_pool(
        queue, 2, one, on_done, lambda: True, lane_held=lambda p: p.lane < ls.lane_floor,
        lane_take=lambda its, cancel: LC.p_take(f, its, cancel)))
    await asyncio.sleep(0.05)
    assert sorted(gates) == ["lo0", "lo1"]
    hi = _pend(f, "hi", 1, NOW + 10)
    ls.set_floor(1)
    queue.append(hi)
    await asyncio.sleep(0.25)
    assert "hi" in gates and not task.done()                    # the lane runs beside the two held legs
    gates["hi"].set()
    assert await asyncio.wait_for(task, 3) >= 1                 # only held legs left: they are taken, the drain ends
    assert done_order == ["hi"]
    lc = f._lane_ctl_obj
    assert sorted(h.p.rid for h in lc.held.values()) == ["lo0", "lo1"]
    assert sorted(b["rid"] for _g, b in f.rpc.paths("/abort_request")) == ["lo0", "lo1"]
    assert all(p.lane_p_taken for p in lo) and not any(p.leg1_done for p in lo)
    assert f.counters["lane_p_taken"] == 2
    # a pool whose callback fails falls back to the late hand-over (nothing is lost)
    queue2 = collections.deque([_pend(f, "lo9", 0, NOW + 20)])
    ls.set_floor(1)
    seen = []

    async def boom(its, cancel):
        raise RuntimeError("take failed")

    async def one3(p):
        await asyncio.sleep(0.2)
        return p
    t2 = asyncio.ensure_future(_p_drain_pool(
        queue2, 1, one3, lambda p: seen.append(p.rid), lambda: True, lane_held=lambda p: p.lane < ls.lane_floor,
        lane_take=boom))
    assert await asyncio.wait_for(t2, 2) >= 0
    ls.set_floor(0)
    await asyncio.sleep(0.4)
    assert seen == ["lo9"]
    assert any("LANE-P-TAKE failed" in r.getMessage() for r in caplog.records)


@_on
async def test_p_take_marks_aborts_by_rid_keeps_the_pendings_and_the_resume_puts_them_back_by_arrival(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front("P")
    ls = f._lane_state()
    lo = [_pend(f, f"lo{i}", 0, NOW + i) for i in range(2)]
    cancelled = []
    _note(f, "hi", 1)
    ls.set_floor(1)
    for p in lo:
        f.groups["P"].outstanding[p.rid] = NOW
    n = await LC.p_take(f, lo, lambda: cancelled.append(True))
    assert n == 2 and cancelled == [True]
    assert all(p.lane_p_taken and p.lane_state == "held" for p in lo)
    aborts = f.rpc.paths("/abort_request")
    assert sorted(b["rid"] for _g, b in aborts) == ["lo0", "lo1"] and all(g == "P" for g, _b in aborts)
    lc = f._lane_ctl_obj
    assert sorted(h.p.rid for h in lc.held.values()) == ["lo0", "lo1"] and list(f.queue) == []
    assert "lo0" in f._park_stuck().lane_held
    assert _marks(caplog, LC.MARK_P_TAKE)[0].startswith("WEG2 LANE-P-TAKE rid=lo0,lo1 floor=1 epoch=")
    assert f.counters["lane_p_taken"] == 2
    # a newer lane-0 request waits in the queue meanwhile; the lane ends: the taken legs come back BEFORE it
    f.queue.append(_pend(f, "newer", 0, NOW + 50))
    f._lane_end("hi")
    await LC.reconcile(f, "end")
    assert [p.rid for p in f.queue] == ["lo0", "lo1", "newer"]
    assert not any(p.lane_p_taken for p in lo) and not any(p.leg1_done for p in lo)
    assert all(p.lane_held_s >= 0.0 and p.t_arrive == NOW + i for i, p in enumerate(lo))


@_on
async def test_p_take_when_the_floor_fell_during_the_aborts_puts_the_legs_straight_back():
    f = _front("P")
    ls = f._lane_state()
    lo = _pend(f, "lo0", 0, NOW)
    ls.set_floor(0)                                             # the lane ended while the aborts were on the wire
    await LC.p_take(f, [lo], lambda: None)
    assert [p.rid for p in f.queue] == ["lo0"] and not f._lane_ctl_obj.held and lo.lane_p_taken is False


@_on
async def test_a_take_abort_that_fails_is_counted_and_the_leg_is_kept_all_the_same(caplog):
    f = _front("P")
    f._lane_state().set_floor(1)
    _note(f, "hi", 1)

    async def bad(g, path, body, timeout):
        return 500, "no"
    f.rpc = bad
    lo = _pend(f, "lo0", 0, NOW)
    assert await LC.p_take(f, [lo], lambda: None) == 1
    assert f.counters["lane_p_take_abort_failed"] == 1 and len(f._lane_ctl_obj.held) == 1
    assert any("abort on P failed" in r.getMessage() for r in caplog.records)


def _p_hold_group(group):
    """P that keeps every leg until released; ``/abort_request`` is only recorded (the leg is the front's to cancel)."""
    group.hold = {}
    return group


@_e2e
async def test_e2e_lane1_during_a_lane0_p_prefill_runs_through_the_p_to_d_flip_and_the_taken_leg_prefills_again(caplog):
    """The plan's 'Spur 1 waehrend P-Prefill' through the P->D flip and the floor fall (review FR1 finding 1):
    lane 0 LONG held on P, lane 1 LONG arrives -> the floor RPC, the lane runs beside the held leg, only the held
    leg is left -> it is taken off P by rid BEFORE the flip (no W3), the flip goes through, the lane runs on D,
    the floor falls, the taken leg is queued by its original arrival, flips back to P and finishes."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    async with LaneHarness(awake="P", d_bs=4, tp_prefill_max_tokens=1000, p_phase_max_requests=6,
                           d_wait_bound_s=0.4, idle_layout="D") as h:
        h.p.hold = {}
        h.d.hold = {}
        lo = h.post_lane("lo", chars=9000)                      # LONG lane 0: P prefills it
        assert await _until(lambda: h.p.running, 10)
        lo_rid = next(iter(h.p.running))
        hi = h.post_lane("hi", priority=1, chars=9000)          # LONG lane 1
        assert await _until(lambda: "gen:hi" in h.p.timeline, 10)   # dispatched beside the held leg
        assert {"floor": 1, "epoch": 1} in h.p.floor_bodies
        h.p.release("hi")                                       # the lane's own leg 1 ends
        # only the held leg is left in the pool: it is taken off P (abort by rid) and the P->D flip follows
        assert await _until(lambda: h.front._lane_ctl_obj.held or h.front.counters["lane_p_taken"], 15)
        assert h.front.counters["lane_p_taken"] == 1
        assert "rpc:abort_request" in h.p.timeline
        assert await _until(lambda: "gen:hi" in h.d.timeline, 20)   # the flip went through: hi runs on D
        assert not any("W3" in r.getMessage() and "STOP" in r.getMessage() for r in caplog.records)
        assert h.front.state == "serving" and lo_rid not in h.front._d_parked
        assert not lo.done()
        h.d.release("hi")
        await asyncio.wait_for(hi, 10)
        # the floor falls: the taken leg is queued again (original arrival) and the front goes back to P for it
        assert await _until(lambda: h.front._lane_state().lane_floor == 0, 10)
        assert await _until(lambda: h.p.gen_marks.count("lo") == 2, 30)     # its leg 1 runs on P again
        h.p.release_all()
        h.d.release_all()
        (s, _) = await asyncio.wait_for(lo, 30)
        assert s == 200
        assert h.front.counters["lane_p_taken"] == 1


# ------------------------------------------------------------------ review FR1: waiters, X-SOLO, SK under the lane filter
def _asr_front():
    """A front with D awake and one seat, for ``_arrival_seat_wait`` (the seat is 'taken' by a flag)."""
    f = _front("D")
    f.d_bs = 1
    f._d_phase_n = 1
    f.d_wait_bound_s = 60.0
    f.t_awake = time.time() - 1.0
    f.__dict__["_seat_taken"] = 1

    def taken():
        return f.__dict__["_seat_taken"], 1
    f._arrival_seat_taken = taken

    async def kv(est, max_tokens, rid, uncached):
        return True, "kv ok", 0
    f._arrival_seat_kv = kv
    return f


@_on
async def test_an_arrival_seat_waiter_below_the_floor_takes_no_seat_counts_no_clock_and_keeps_its_place(caplog):
    """Review L4 FR1 finding 2: a lane-0 SHORT already in ``_arrival_seat_wait`` when the preempt hits is held --
    out of ``waiters`` (no wait bound, no head, no displacement victim), never granted the freed seat; it re-enters
    with its ORIGINAL order stamp and the held time as clock offset."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _asr_front()
    st = f._asr_st()
    _note(f, "lo", 0)
    _note(f, "lo2", 0)
    w = asyncio.ensure_future(f._arrival_seat_wait("lo", 100, None, 100))
    await asyncio.sleep(0.2)
    stamp = st["waiters"]["lo"]
    w2 = asyncio.ensure_future(f._arrival_seat_wait("lo2", 100, None, 100))   # a NEWER lane-0 waiter
    await asyncio.sleep(0.1)
    assert await _arrive(f, "hi", 1) is None                  # the floor rises
    lc = f._lane_ctl_obj
    assert await _until(lambda: set(lc.waiter_held) == {"lo", "lo2"}, 5)     # held at each waiter's next tick
    assert "lo" not in st["waiters"] and "lo2" not in st["waiters"] and set(lc.waiter_held) == {"lo", "lo2"}
    assert lc.waiter_held["lo"][0] == stamp                    # the original order stamp is kept
    assert f._asr_waiter_clocks(st) == []                      # no wait clock for the bound
    assert f.counters["lane_waiter_held"] == 2 and not w.done() and not w2.done()
    f.__dict__["_seat_taken"] = 0                              # a seat frees up (the lane-1 request has not taken it)
    await asyncio.sleep(0.3)
    assert not w.done() and not w2.done() and not st["granted"]    # held waiters never take the freed seat
    assert set(f.__dict__["_lane_ctl_obj"].waiter_held) == {"lo", "lo2"}
    f.__dict__["_seat_taken"] = 1
    f._lane_end("hi")
    await LC.reconcile(f, "end")                               # the floor falls: both re-enter, 'lo' first
    assert await _until(lambda: set(st["waiters"]) == {"lo", "lo2"}, 5)
    assert sorted(st["waiters"], key=st["waiters"].get) == ["lo", "lo2"] and st["waiters"]["lo"] == stamp
    off = st["lane_off"]["lo"]
    assert off >= 0.3 and f._asr_waiter_clocks(st)[0] == pytest.approx(stamp + off)
    assert "WEG2 LANE-RESUME" in " ".join(r.getMessage() for r in caplog.records)
    assert any("waiters=2" in r.getMessage() for r in caplog.records)
    f.__dict__["_seat_taken"] = 0
    assert await asyncio.wait_for(w, 3) is True                # the OLDEST waiter is granted first
    assert not w2.done() or st["granted"].get("lo") is not None
    f.__dict__["_seat_taken"] = 1
    w2.cancel()
    await asyncio.gather(w2, return_exceptions=True)
    assert "lo2" not in lc.waiter_held and "lo2" not in st.get("lane_off", {})


@_on
async def test_a_held_waiter_that_leaves_takes_its_registers_with_it():
    f = _asr_front()
    _note(f, "lo", 0)
    w = asyncio.ensure_future(f._arrival_seat_wait("lo", 100, None, 100))
    await asyncio.sleep(0.1)
    assert await _arrive(f, "hi", 1) is None
    await asyncio.sleep(0.2)
    assert "lo" in f._lane_ctl_obj.waiter_held
    w.cancel()
    await asyncio.gather(w, return_exceptions=True)
    assert "lo" not in f._lane_ctl_obj.waiter_held and "lo" not in f._asr_st()["waiters"]
    assert f._lane_block()["lane_hold"]["waiters"] == 0


@_on
async def test_a_waiter_released_from_the_gate_keeps_the_original_arrival_as_its_order_and_the_hold_as_offset():
    f = _asr_front()
    _note(f, "lo", 0)
    lc = LC.ctl(f)
    lc.gate_done["lo"] = (NOW - 100.0, 7.0)
    w = asyncio.ensure_future(f._arrival_seat_wait("lo", 100, None, 100))
    await asyncio.sleep(0.1)
    st = f._asr_st()
    assert st["waiters"]["lo"] == NOW - 100.0 and st["lane_off"]["lo"] == 7.0
    w.cancel()
    await asyncio.gather(w, return_exceptions=True)


def test_arrival_seat_waiter_clocks_are_the_raw_stamps_with_the_switch_off():
    f = _front("D")
    st = f._asr_st()
    st["waiters"].update({"a": 5.0, "b": 9.0})
    assert f._asr_waiter_clocks(st) == [5.0, 9.0]
    with _switch(False):
        assert LC.waiter_hold(f, st, "a") is False and "_lane_ctl_obj" not in f.__dict__


@_on
async def test_x_solo_decides_p_for_a_band_request_below_a_floor_that_rose_inside_its_window():
    f = _front("D")
    f.tp_prefill_max_tokens = X
    _note(f, "lo", 0)
    with _env("SGLANG_WEG2_X_SOLO_WINDOW_MS", "300"):
        t = asyncio.ensure_future(f._x_solo_admits("lo", 5000))
        await asyncio.sleep(0.1)
        f._lane_state().set_floor(1)                          # no arrival counted in the window: only the floor
        assert await asyncio.wait_for(t, 3) is False          # P (queue, then the lane sweep holds it)
        _note(f, "hi", 1)
        t2 = asyncio.ensure_future(f._x_solo_admits("hi", 5000))
        assert await asyncio.wait_for(t2, 3) is True          # the lane itself is a singleton as before


@_on
async def test_short_kept_requests_below_the_floor_are_held_in_queue_and_ready_and_come_back_by_arrival():
    f = _front("D")
    sk_q = _pend(f, "sk_q", 0, NOW - 30, short_kept=True)
    sk_d = _pend(f, "sk_d", 0, NOW - 40, short_kept=True, d_direct=True, skip_leg1=True)
    f.queue.append(sk_q)
    f._ready_for_d.append(sk_d)
    f._sync_batch_gate()
    assert await _arrive(f, "hi", 1) is None
    assert list(f.queue) == [] and list(f._ready_for_d) == [] and f._batch_gate.is_set()
    lc = f._lane_ctl_obj
    assert sorted(h.p.rid for h in lc.held.values()) == ["sk_d", "sk_q"]
    f._lane_end("hi")
    await LC.reconcile(f, "end")
    assert [p.rid for p in f.queue] == ["sk_q"] and [p.rid for p in f._ready_for_d] == ["sk_d"]
    assert sk_q.short_kept and sk_d.d_direct and sk_q.t_arrive == NOW - 30


# ------------------------------------------------------------------ review FR1: lane 1 during a lane-0 D prefill (X route)
@_e2e
async def test_e2e_lane1_during_a_lane0_d_prefill_on_the_x_route_parks_it_runs_alone_and_the_parked_resume(caplog):
    """The plan's fourth scenario (review FR1 finding 3): a lane-0 SHORT (uncached <= X) is prefilled by D itself
    (ARRIVAL-SEAT verdict d_prefill / the X route, no leg 1, no flip); lane 1 arrives while that prefill runs."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    async with LaneHarness(awake="D", d_bs=4, tp_prefill_max_tokens=100_000) as h:
        h.d.hold = {}
        lo = h.post_lane("lo", chars=300)
        assert await _until(lambda: len(h.d.running) == 1, 10)
        lo_rid = next(iter(h.d.running))
        # the route really was the X route: D prefills it itself (no leg 1 on P), the verdict is logged
        assert "lo" not in h.p.gen_marks
        assert any(("verdict=d_prefill" in r.getMessage() and lo_rid in r.getMessage())
                   or ("WEG2-ROUTE rid=%s SHORT" % lo_rid) in r.getMessage() for r in caplog.records), \
            [r.getMessage()[:120] for r in caplog.records if lo_rid in r.getMessage()][:12]
        hi = h.post_lane("hi", priority=1, chars=300)
        assert await _until(lambda: "gen:hi" in h.d.timeline, 10)
        assert h.d.park_bodies[0]["rids"] == [lo_rid] and h.d.park_bodies[0]["hold"] == "lane"
        assert h.d.timeline.index("rpc:weg2/park_running") < h.d.timeline.index("gen:hi")
        assert h.front._d_parked[lo_rid] == PP.LANE_PARK_STAMP and h.front.state == "serving"
        assert not lo.done() and "hi" not in h.p.gen_marks
        h.d.release("hi")
        await asyncio.wait_for(hi, 10)
        assert await _until(lambda: h.front._lane_state().lane_floor == 0 and lo_rid not in h.front._d_parked, 10)
        h.d.release_all()
        (s, _) = await asyncio.wait_for(lo, 10)
        assert s == 200 and h.d.gen_marks.count("lo") == 1     # a resume, never a second prefill


@_e2e
async def test_e2e_a_lane0_arrival_seat_waiter_is_not_granted_the_seat_the_lane1_park_frees(caplog):
    """Review FR1 finding 2 end to end, on a bs1 D: lo1 runs, lo2 waits for the seat (ARRIVAL-SEAT), lane 1 arrives.
    lo1 is parked and gives its seat back; lo2 must NOT take it (priority inversion) -- the lane-1 request does,
    and after it the lane-0 requests resume in the order of their arrival."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    async with LaneHarness(awake="D", d_bs=1, tp_prefill_max_tokens=100_000, d_wait_bound_s=60.0) as h:
        h.d.hold = {}
        lo1 = h.post_lane("lo1", chars=300)
        assert await _until(lambda: len(h.d.running) == 1, 10)
        lo2 = h.post_lane("lo2", chars=300)
        assert await _until(lambda: bool(h.front._asr_st()["waiters"]), 10)       # lo2 waits for the seat
        hi = h.post_lane("hi", priority=1, chars=300)
        assert await _until(lambda: "gen:hi" in h.d.timeline, 15)
        assert "gen:lo2" not in h.d.timeline                    # the freed seat went to the lane
        assert await _until(lambda: len(h.front._lane_ctl_obj.waiter_held) == 1, 5)     # held at its next tick
        assert "gen:lo2" not in h.d.timeline and not h.front._asr_st()["waiters"]
        h.d.release("hi")
        await asyncio.wait_for(hi, 10)
        assert await _until(lambda: h.front._lane_state().lane_floor == 0, 10)
        h.d.release_all()
        res = await asyncio.wait_for(asyncio.gather(lo1, lo2), 20)
        assert [s for s, _ in res] == [200, 200]
        assert h.d.timeline.index("gen:hi") < h.d.timeline.index("gen:lo2")
        assert h.d.gen_marks.count("lo1") == 1 and h.d.gen_marks.count("lo2") == 1
