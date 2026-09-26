"""H91c2: the park and wait stages of the NF standard form H91 at a desk.

The run fnFL2h91v1 proved only bs2 and stau6; park (the youngest parks) and
wait (60 s bound, then back to P) never ran on metal. This file plays their
cases through with fake clocks and stand-in groups -- no GPU, no model -- and
holds the three defects the play-through found:

* H91c2-1 (front): D's park answer names the requests it holds QUEUED behind
  the parked ones (``held``, never started: an X-route request behind the
  H95c seat cap n, a newcomer behind a pressure-park barrier). The front read
  only ``parked``, so the held rids stayed in its flip ledger: the D->P drain
  waited the whole drain window (120 s) for requests D keeps in its park list
  by design, and then W1b aborted them per rid -- a client stream cut by the
  wait bound, and a D->P flip 120 s late (a wait far beyond 60 s).
* H91c2-2 (D): ``park_running`` re-lists a request that decode pressure had
  parked earlier WITHOUT restarting its park clock. ``park_tick`` reads the
  OLDEST stamp, so a pressure park older than 30 s made the very next pass
  re-queue the whole flip park: D decoded the parked requests again while the
  front, told they were parked, flipped -- quiesce never idle -> W3 STOP
  (or, for short decodes, the park undone and the bound blown).
* H91c2-3 (D): a park site is never cleared. A request resumed from a flip
  park kept SITE_FLIP; when decode pressure retracted it later,
  ``note_retracted`` kept the stale FLIP mark, and the admission gate let a
  flip park resume "as soon as it fits" -- no "older one first" rule: it was
  re-admitted on the next pass and retracted again (thrash) instead of
  waiting for the older request as the user design says.

The three defect tests are red on c8bf488fe7 and green with the fix; the
case tests (a)-(c) are green on both and document the play-through; case (d)
(ranks) is red on the base too, through H91c2-2 (the stale pressure stamp
makes every rank due right after the park).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import types
from collections import deque
from typing import Dict

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from aiohttp import web  # noqa: E402

from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402
from sglang.srt.weg2 import phase_policy as pp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_phase_policy_h91c import FakeGroup, Harness, _until  # noqa: E402


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.delenv(ds.RESUME_MARGIN_ENV, raising=False)
    monkeypatch.setenv(ds.AWAKE_REQUEUE_ENV, "30")


# ------------------------------------------------------------ scheduler fake
class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = float(t)

    def monotonic(self) -> float:
        return self.t

    def time(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(rt, "time", c)
    return c


def _req(rid, seq):
    return types.SimpleNamespace(
        rid=rid, kv_arrival_seq=seq, origin_input_ids=[0] * 10, output_ids=[0] * 3,
        is_fast_lane=False, spill_class=None,
    )


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


class _Sched:
    """The scheduler state d_park_runtime moves, nothing more."""

    def __init__(self, running=(), waiting=()):
        self.running_batch = _Batch(running)
        self.waiting_queue = list(waiting)
        self.last_batch = None
        self.enable_overlap = False
        self.result_queue = deque()
        self.chunked_req = None
        self.anchor_tails = []
        self.server_args = types.SimpleNamespace(max_running_requests=6)
        self.weg2_dormant = False
        self.enable_hicache_storage = False
        self.sent = []
        self.min_flag_calls = 0
        self.ipc_channels = types.SimpleNamespace(
            send_to_tokenizer=types.SimpleNamespace(send_output=lambda o, r: self.sent.append(o))
        )

    def _969ad_note_retract(self, req, site):
        pass

    def _add_request_to_queue(self, req, is_retracted=False):
        if self.weg2_dormant:
            hold = getattr(self, "weg2_dormant_hold", None)
            if hold is None:
                hold = self.weg2_dormant_hold = []
            hold.append(req)
        else:
            self.waiting_queue.append(req)

    def _weg2_group_min_flags(self, flags):
        self.min_flag_calls += 1
        return [1 if f else 0 for f in flags]

    def uniform_min_avail(self):
        return 0


def _park(s, epoch=5):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    return rt.park_running(s, Weg2ParkRunningReqInput(epoch=epoch, reason="wait-bound-60s"))


def _pressure(s, victims):
    """What _retract_decode_and_requeue does on D: the victims leave the
    running batch, join the queue, and are noted as a pressure park."""
    ids = {id(r) for r in victims}
    s.running_batch.reqs = [r for r in s.running_batch.reqs if id(r) not in ids]
    for r in victims:
        s._add_request_to_queue(r, is_retracted=True)
    return rt.note_retracted(s, victims)


def _admit(s):
    """One admission pass: the gate's verdict per waiting request; admitted
    requests move to the running batch (the adder always has room here)."""
    gate = rt.admission(s, s.running_batch)
    admitted = []
    for r in list(s.waiting_queue):
        if gate is not None and gate.skip(r) is not None:
            continue
        admitted.append(r)
    for r in admitted:
        s.waiting_queue.remove(r)
        s.running_batch.reqs.append(r)
    return [r.rid for r in admitted]


# ============================================================ the defects
def test_h91c2_1_park_verdict_carries_the_held_rids():
    """D answers parked + held; both stay on D and resume after its wake,
    so both are 'parked' for the front's drain, W3 and W1b."""
    body = json.dumps({"success": True, "parked": ["r"], "held": ["h1", "h2"],
                       "epoch": 3, "message": "parked 1, queued behind them 2"})
    verdict, rids, why = pp.park_verdict(200, body)
    assert verdict == pp.PARK_PARKED and why == ""
    assert rids == ["r", "h1", "h2"]
    # unchanged without held, and a malformed held is never read as "none"
    assert pp.park_verdict(200, json.dumps({"parked": ["a"]}))[1] == ["a"]
    assert pp.park_verdict(200, json.dumps({"parked": ["a"], "held": "x"}))[0] == pp.PARK_FAILED
    assert pp.park_verdict(200, json.dumps({"parked": ["a"], "held": [1]}))[0] == pp.PARK_FAILED


class HeldD(FakeGroup):
    """D as part B answers: requests marked ``queued`` never start (an X-route
    request behind the seat cap n); the park lists them under ``held``."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.queued_marks = set()
        self.queued: Dict[str, str] = {}

    async def _generate(self, request):
        payload = await request.json()
        mark = self._mark(payload)
        rid = payload.get("rid") or mark
        queued = mark in self.queued_marks
        self.timeline.append(f"{'queued' if queued else 'gen'}:{mark}")
        self.gen_marks.append(mark)
        (self.queued if queued else self.running)[rid] = mark
        try:
            await self.hold.setdefault(mark, asyncio.Event()).wait()
        finally:
            (self.queued if queued else self.running).pop(rid, None)
            self.timeline.append(f"done:{mark}")
        return web.json_response({
            "choices": [{"text": "x"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 1,
                      "prompt_tokens_details": {"cached_tokens": 100}},
        })

    async def _park(self, request):
        body = await request.json()
        self.park_bodies.append(body)
        self.timeline.append("rpc:weg2/park_running")
        return web.json_response({"success": True, "parked": sorted(self.running),
                                  "held": sorted(self.queued), "epoch": body.get("epoch"),
                                  "message": ""})


def test_h91c2_1_a_held_request_is_parked_for_the_front_not_drained_and_aborted():
    """Red on c8bf488fe7: the drain waited the whole window for the held
    X-route request and W1b then aborted it on D by rid."""

    async def body():
        hh = Harness(awake="D", d_bs=4, tp_prefill_max_tokens=100_000,
                     d_wait_bound_s=0.6, p_phase_max_requests=6, drain_deadline_s=3.0)
        hh.d = HeldD("D")
        hh.d.queued_marks = {"s1"}
        async with hh as h:
            h.d.hold = {}
            t0 = h.post("s0")                                   # X route, decoding on D
            assert await _until(lambda: h.d.running, 10)
            t1 = h.post("s1")                                   # X route, queued on D
            assert await _until(lambda: h.d.queued, 10)
            assert h.front.counters["route_short"] == 2
            tl = h.post("L0", chars=400_000)                    # LONG: waits for P
            assert await _until(lambda: h.d.park_bodies, 10)    # the bound fired
            t_park = time.time()
            assert await _until(lambda: "rpc:release_memory_occupation" in h.d.timeline, 10)
            t_sleep = time.time()
            # the flip followed the park at once: nothing D still runs held it
            assert t_sleep - t_park < 2.0, t_sleep - t_park
            assert "rpc:abort_request" not in h.d.timeline, h.d.timeline
            # both stay in flight for the front, and D's next wake names both
            assert await _until(lambda: "gen:L0" in h.p.timeline, 10)
            assert await _until(lambda: len([b for b in h.d.resume_bodies
                                             if b.get("tags") == ["kv_cache"]]) >= 1, 20)
            kv = [b for b in h.d.resume_bodies if b.get("tags") == ["kv_cache"]]
            assert kv[-1].get("parked_n") == 2, kv
            assert not t0.done() and not t1.done()
            h.d.release_all()
            results = await asyncio.wait_for(asyncio.gather(t0, t1, tl), 20)
            assert [s for s, _ in results] == [200, 200, 200]
            assert h.d.gen_marks.count("s0") == 1               # never re-posted

    asyncio.run(body())


def test_h91c2_2_a_flip_park_restarts_the_requeue_clock_of_an_earlier_pressure_park(clock):
    """Red on c8bf488fe7: the pressure park's 40-s-old stamp made the first
    awake pass after park_running re-queue the whole park (D ran them again
    while the front flipped: W3)."""
    old, young, new = _req("old", 1), _req("young", 2), _req("new", 3)
    s = _Sched(running=[old, young], waiting=[new])
    assert _pressure(s, [young]) == 1                   # decode pressure: young parks
    assert ds.park_site(young) == ds.SITE_PRESSURE
    clock.t += 40.0                                     # it waits for old ... 40 s
    out = _park(s)                                      # the wait bound fires
    assert out.parked == ["old", "young"] and out.held == ["new"]
    clock.t += 0.2                                      # the next scheduler pass
    assert rt.park_tick(s) == 0, "the flip park was undone before D's sleep"
    assert [r.rid for r in s.weg2_d_parked] == ["old", "young", "new"]
    assert s.waiting_queue == []
    clock.t += 29.0
    assert rt.park_tick(s) == 0
    clock.t += 1.0                                      # 30 s after THIS park: the net
    assert rt.park_tick(s) == 3


def test_h91c2_3_a_resumed_flip_park_retracted_by_pressure_waits_for_the_older(clock):
    """Red on c8bf488fe7: the stale FLIP mark let the retracted request back
    in on the very next pass although the older one still decodes."""
    older, younger = _req("older", 1), _req("younger", 2)
    s = _Sched(running=[older, younger])
    _park(s)                                            # wait bound: both flip-parked
    s.weg2_dormant = True
    assert rt.hold_parked(s, hold_armed=True) == 2      # the sleep
    s.weg2_dormant = False                              # the wake releases the hold
    s.waiting_queue = list(s.weg2_dormant_hold)
    s.weg2_dormant_hold = []
    assert _admit(s) == ["older", "younger"]            # both resume, oldest first
    clock.t += 5.0
    assert _pressure(s, [younger]) == 1                 # later: the pool is full
    assert ds.park_site(younger) == ds.SITE_PRESSURE
    for _ in range(3):                                  # older still decodes
        assert _admit(s) == [], "the younger came back while the older still runs"
    s.running_batch.reqs.remove(older)                  # the older finishes
    assert _admit(s) == ["younger"]


# ============================================================ the cases
def test_case_a_d_full_the_youngest_parks_then_resumes_after_the_older(clock):
    """(a) D full (n = --d-bs = 6): under spec only the batch's back may
    leave, and it IS the youngest; the pressure retraction parks exactly it at
    the queue head, the newcomer waits behind the park, the park resumes when
    no older request is live, then the newcomer. (The H91d draft rows of the
    victim ride along at the same retraction: test_weg2_d_park_draft_h91d.)"""
    reqs = [_req(f"r{i}", i) for i in range(6)]
    new = _req("new", 9)
    s = _Sched(running=reqs, waiting=[new])
    order = ds.retraction_order(reqs, spec_active=True)
    assert order == list(range(6)) and reqs[order[-1]].rid == "r5"
    assert _pressure(s, [reqs[5]]) == 1
    assert [r.rid for r in s.waiting_queue] == ["r5", "new"]
    assert _admit(s) == []                                # r0..r4 live: r5 and new wait
    for r in reqs[:4]:
        s.running_batch.reqs.remove(r)
    assert _admit(s) == []                                # r4 is still older and live
    s.running_batch.reqs.remove(reqs[4])
    assert _admit(s) == ["r5"]
    assert _admit(s) == ["new"]


def test_case_c_two_parks_in_one_round_resume_oldest_first(clock):
    """(c) two victims of one retract_decode: each waits for every older
    live request, so they come back one at a time, oldest first."""
    a, b, c = _req("a", 1), _req("b", 2), _req("c", 3)
    s = _Sched(running=[a, b, c])
    assert _pressure(s, [c, b]) == 2
    assert [r.rid for r in s.waiting_queue] == ["b", "c"]
    assert _admit(s) == []
    s.running_batch.reqs.remove(a)
    assert _admit(s) == ["b"]
    s.running_batch.reqs.remove(b)
    assert _admit(s) == ["c"]


def test_case_c_the_last_seat_parks_and_resumes_as_the_oldest(clock):
    """(c) the solo-OOM retraction of the only request is a park too; it is
    the oldest live request, so it resumes on the next pass, ahead of a
    newcomer that waits behind it."""
    only, new = _req("only", 1), _req("new", 5)
    s = _Sched(running=[only], waiting=[new])
    assert _pressure(s, [only]) == 1
    assert [r.rid for r in s.waiting_queue] == ["only", "new"]
    assert _admit(s) == ["only"]
    assert _admit(s) == ["new"]


def test_case_c_resume_after_the_flip_back_with_fewer_seats_than_parked_plus_new():
    """(c) six parked + three new at the wake: n clamps to --d-bs = 6
    (H95), the parked ones take the seats first, oldest first; the new ones
    wait on D until a parked one ends."""
    seats = ds.phase_seats(3, 6, cap=6, epoch="9")
    assert seats.n == 6 and seats.clamped
    parked = [_req(f"p{i}", i) for i in range(6)]
    for r in parked:
        ds.mark_parked(r, ds.SITE_FLIP, epoch=8)
    new = [_req(f"n{i}", 10 + i) for i in range(3)]
    waiting = ds.order_waiting(new + list(reversed(parked)))
    assert [r.rid for r in waiting][:6] == [f"p{i}" for i in range(6)]
    gate = ds.admission_gate(waiting, running=[])
    assert all(gate.skip(r) is None for r in parked)
    assert all(gate.skip(r) == "weg2_d_park_first" for r in new)


def test_case_c_a_park_during_a_running_flip_lists_the_pressure_parked_too(clock):
    """(c) park_running while a pressure park waits (the front's flip is
    about to start): the pressure-parked request is in the answer's parked
    list, the queued newcomer in held, nothing is left on D's queue."""
    old, young, new = _req("old", 1), _req("young", 2), _req("new", 3)
    s = _Sched(running=[old, young], waiting=[new])
    _pressure(s, [young])
    out = _park(s)
    assert out.parked == ["old", "young"] and out.held == ["new"]
    assert s.waiting_queue == [] and s.running_batch.reqs == []
    # the parked list keeps each request's site: young still waits for old
    assert ds.park_site(young) == ds.SITE_PRESSURE and ds.park_site(old) == ds.SITE_FLIP


def test_case_b_the_bound_is_a_trigger_not_a_latency_cap():
    """(b) the wait is counted from max(arrival, D phase start): a request the
    6-cap left behind in a P phase fires the bound 60 s into the NEXT D phase,
    its end-to-end wait is the P-phase rest + flip + 60 s + park + flip. So a
    wait of 103 s (R28) is reachable with H91 too; the bound itself fires on
    the first controller tick at >= 60 s (0.2 s tick), never later."""
    t_arrive, t_d_awake = 0.0, 35.0           # arrived early in the P phase
    assert not pp.wait_bound_fired(pp.d_phase_wait_s(t_arrive, t_d_awake, 94.9), 60.0)
    assert pp.wait_bound_fired(pp.d_phase_wait_s(t_arrive, t_d_awake, 95.0), 60.0)
    # an arrival during the D phase: its own arrival is the origin
    assert pp.wait_bound_fired(pp.d_phase_wait_s(50.0, 35.0, 110.0), 60.0)
    assert not pp.wait_bound_fired(pp.d_phase_wait_s(50.0, 35.0, 109.9), 60.0)


def test_case_d_the_park_decisions_are_replicated_across_ranks(clock):
    """(d) three ranks, the same broadcast park request and the same
    replicated queues, rank-local clocks skewed by up to 0.4 s: every rank
    parks the same rids, holds the same list, and park_tick's verdict is the
    group MIN -- no rank requeues alone."""
    ranks = []
    for skew in (0.0, 0.2, 0.4):
        old, young, new = _req("old", 1), _req("young", 2), _req("new", 3)
        s = _Sched(running=[old, young], waiting=[new])
        clock.t = 1000.0 + skew
        _pressure(s, [young])
        clock.t = 1040.0 + skew
        out = _park(s)
        ranks.append((s, out))
    assert len({tuple(o.parked) for _, o in ranks}) == 1
    assert len({tuple(o.held) for _, o in ranks}) == 1
    assert len({tuple(r.rid for r in s.weg2_d_parked) for s, _ in ranks}) == 1
    # the rank-local stamps differ by the ranks' processing times; at an
    # instant inside that spread the local verdicts differ -- park_tick acts
    # only on the group MIN, so no rank requeues alone
    local = [ds.awake_requeue_due(s.weg2_d_parked, now=1070.1, bound_s=30.0) for s, _ in ranks]
    assert local == [True, False, False] and min(local) is False
    assert all(ds.awake_requeue_due(s.weg2_d_parked, now=1070.5, bound_s=30.0) for s, _ in ranks)
    # and none of them is due right after the park (H91c2-2)
    assert not any(ds.awake_requeue_due(s.weg2_d_parked, now=1040.5, bound_s=30.0)
                   for s, _ in ranks)
