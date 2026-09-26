"""H91c3: the three rests the H91c2 report named at the park/wait stages of
the NF standard form H91 (desk play-through, no GPU, no model).

* H91c3-1 (front): ``_wait_bound_park`` stamped ``_d_parked`` only AFTER the
  park RPC answered. D stamps its park while serving the RPC and re-queues it
  after PARK_REQUEUE_S (30 s) on its own clock, so the front's 30-s clock ran
  late by the RPC's duration: for that long the front still counted as parked
  what D was running again (drain and W3 blind to it). The stamp is now taken
  before the RPC and written only for the rids D confirms.
* H91c3-2 (D + front): a hand-off already in ``D.outstanding`` whose leg 2
  had not reached D's scheduler when the park came (tokenizer / HTTP pipe)
  was in neither of D's lists. D's park barrier did NOT catch it: the
  admission gate reads the waiting queue and the pending-outside lists, never
  ``weg2_d_parked`` -- so it was admitted after the park and decoded to its
  end, while the front's D->P drain waited for it (at worst the whole drain
  window, then W1b aborted it). D now holds every new arrival between the
  park and the sleep behind the park and says so (``late_hold``); the front
  counts its in-flight hand-offs as parked.
* H91c3-3 (front): an X-route request during a D phase does not count in the
  phase's n (H95). With n < --d-bs the front still had a seat for it, so it
  went to D and waited there behind the H95c seat cap -- in neither the
  front's queue nor anything the wait bound reads (with the P queue empty
  the controller never even evaluated the bound). With the wait bound armed
  and the phase's n seats taken, it now falls through to route BATCH: the P
  queue, where its arrival counts in ``d_phase_wait_s``.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Dict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
from aiohttp import web  # noqa: E402

from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402
from sglang.srt.weg2 import phase_policy as pp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_park_wait_h91c2 import FakeClock, _admit, _req, _Sched  # noqa: E402
from test_weg2_phase_policy_h91c import FakeGroup, Harness, _until  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", "..", ".."))


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.delenv(ds.RESUME_MARGIN_ENV, raising=False)
    monkeypatch.setenv(ds.AWAKE_REQUEUE_ENV, "30")


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(rt, "time", c)
    return c


# ================================================================ H91c3-1
class SlowParkD(FakeGroup):
    """D whose park RPC takes ``park_s``: it parks (and stamps) on receipt,
    the answer reaches the front ``park_s`` later. ``park_status`` != 200 =
    a failed park."""

    def __init__(self, *a, park_s: float = 0.5, park_status: int = 200, **kw):
        super().__init__(*a, **kw)
        self.park_s = park_s
        self.park_status = park_status
        self.t_park_received = None

    async def _park(self, request):
        body = await request.json()
        self.t_park_received = time.time()
        self.park_bodies.append(body)
        self.timeline.append("rpc:weg2/park_running")
        parked = sorted(self.running)
        await asyncio.sleep(self.park_s)
        if self.park_status != 200:
            return web.json_response({"error": "boom"}, status=self.park_status)
        return web.json_response({"success": True, "parked": parked, "held": [],
                                  "epoch": body.get("epoch"), "message": ""})


def _park_harness(park_s: float, park_status: int = 200) -> Harness:
    hh = Harness(awake="D", d_bs=4, tp_prefill_max_tokens=100_000,
                 d_wait_bound_s=0.4, p_phase_max_requests=6, drain_deadline_s=3.0)
    hh.d = SlowParkD("D", park_s=park_s, park_status=park_status)
    return hh


def test_h91c3_1_the_front_stamps_the_park_before_the_rpc():
    """Red on 636502eb97: the stamp was the answer's arrival, 0.5 s after D
    had parked (and started its own 30-s clock)."""

    async def body():
        async with _park_harness(park_s=0.5) as h:
            h.d.hold = {}
            t0 = h.post("s0")                                   # X route: decodes on D
            assert await _until(lambda: h.d.running, 10)
            rid0 = next(iter(h.d.running))
            tl = h.post("L0", chars=400_000)                    # LONG: waits for P
            assert await _until(lambda: rid0 in h.front._d_parked, 10)
            stamp = h.front._d_parked[rid0]
            # the front's clock starts no later than D's own park stamp
            assert stamp <= h.d.t_park_received, (stamp, h.d.t_park_received)
            assert h.d.t_park_received - stamp < 0.4
            h.d.release_all()
            results = await asyncio.wait_for(asyncio.gather(t0, tl), 20)
            assert [s for s, _ in results] == [200, 200]

    asyncio.run(body())


def test_h91c3_1_a_failed_park_leaves_no_stamp():
    """The stamp only counts for rids D confirmed: a failed park RPC writes
    none (the pre-H91 fallback runs, nothing is parked for the front)."""

    async def body():
        async with _park_harness(park_s=0.1, park_status=500) as h:
            h.d.hold = {}
            t0 = h.post("s0")
            assert await _until(lambda: h.d.running, 10)
            tl = h.post("L0", chars=400_000)
            assert await _until(lambda: h.front.counters["park_failed"] == 1, 10)
            assert h.front._d_parked == {}
            h.d.release_all()
            results = await asyncio.wait_for(asyncio.gather(t0, tl), 20)
            assert [s for s, _ in results] == [200, 200]

    asyncio.run(body())


# ================================================================ H91c3-2
class _IntakeSched(_Sched):
    """The scheduler's intake order: the #1443 dormant hold first, then (a
    new request, never a re-queue) the late hold behind an open park, then
    the waiting queue -- Scheduler._add_request_to_queue's tail."""

    def _add_request_to_queue(self, req, is_retracted=False):
        hook = getattr(rt, "hold_late_arrival", None)
        if (not self.weg2_dormant and not is_retracted and hook is not None
                and hook(self, req)):
            return
        super()._add_request_to_queue(req, is_retracted)


def _park_late(s, epoch=5, armed=True):
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    return rt.park_running(s, Weg2ParkRunningReqInput(epoch=epoch, reason="wait-bound-60s"),
                           late_hold_armed=armed)


def test_h91c3_2_d_holds_a_hand_off_that_arrives_after_the_park(clock):
    """Red on 636502eb97: the late hand-off entered the waiting queue and the
    next pass admitted it (no barrier: weg2_d_parked is outside the gate)."""
    a = _req("a", 1)
    s = _IntakeSched(running=[a])
    out = _park_late(s)
    assert out.parked == ["a"] and out.held == [] and out.late_hold is True
    clock.t += 0.05
    late = _req("late", 2)
    s._add_request_to_queue(late)                       # the leg 2 lands now
    assert s.waiting_queue == [], "the late hand-off is not in D's queue"
    assert [r.rid for r in s.weg2_d_parked] == ["a", "late"]
    assert ds.park_site(late) is None                   # held, not parked
    assert _admit(s) == [] and s.running_batch.reqs == []
    # its requeue clock is the park's: the net re-queues both together
    clock.t += 29.9
    assert rt.park_tick(s) == 0
    clock.t += 0.1
    assert rt.park_tick(s) == 2
    assert [r.rid for r in s.waiting_queue] == ["a", "late"]
    # the re-queue closed the late hold: the next arrival is ordinary
    new = _req("new", 3)
    s._add_request_to_queue(new)
    assert [r.rid for r in s.waiting_queue] == ["a", "late", "new"]


def test_h91c3_2_the_sleep_takes_the_late_hand_off_into_the_hold_behind_the_park(clock):
    a, b = _req("a", 1), _req("b", 2)
    s = _IntakeSched(running=[a, b])
    _park_late(s)
    late = _req("late", 3)
    s._add_request_to_queue(late)
    s.weg2_dormant = True                               # the sleep leg's dormant point
    assert rt.hold_parked(s, hold_armed=True) == 3
    assert [r.rid for r in s.weg2_dormant_hold] == ["a", "b", "late"]
    assert s.weg2_d_parked == [] and getattr(s, rt.LATE_HOLD_ATTR, None) is None
    s.weg2_dormant = False                              # after the wake: ordinary intake
    new = _req("new", 4)
    s._add_request_to_queue(new)
    assert [r.rid for r in s.waiting_queue] == ["new"]


def test_h91c3_2_without_the_dormant_hold_d_neither_holds_nor_promises(clock):
    """#1443 off: an arrival after the sleep is refused (W25), so D promises
    nothing and the front keeps waiting for its in-flight hand-offs."""
    a = _req("a", 1)
    s = _IntakeSched(running=[a])
    out = _park_late(s, armed=False)
    assert out.late_hold is False
    late = _req("late", 2)
    s._add_request_to_queue(late)
    assert s.waiting_queue == [late]


def test_h91c3_2_a_pending_post_wake_settle_withdraws_the_promise(clock):
    """#1471 settle releases straight into the queue, not through the intake:
    with one pending D promises nothing (the front drains as before)."""
    s = _IntakeSched(running=[_req("a", 1)])
    s.weg2_post_wake_settle = [_req("settling", 2)]
    out = _park_late(s)
    assert out.late_hold is False
    s._add_request_to_queue(_req("late", 3))
    assert [r.rid for r in s.waiting_queue] == ["late"]


def test_h91c3_2_the_late_hold_is_replicated_across_ranks(clock):
    """Three ranks, the same broadcast park and the same intake order: every
    rank holds the same late arrival at the same position."""
    views = []
    for skew in (0.0, 0.2, 0.4):
        clock.t = 1000.0 + skew
        s = _IntakeSched(running=[_req("a", 1)], waiting=[_req("q", 2)])
        _park_late(s)
        s._add_request_to_queue(_req("late", 3))
        views.append(tuple(r.rid for r in s.weg2_d_parked))
    assert set(views) == {("a", "q", "late")}


def test_h91c3_2_the_answer_carries_late_hold_and_the_front_reads_it_strictly():
    assert pp.park_late_hold(200, '{"parked": [], "late_hold": true}') is True
    assert pp.park_late_hold(200, '{"parked": []}') is False           # old D
    assert pp.park_late_hold(200, '{"parked": [], "late_hold": "yes"}') is False
    assert pp.park_late_hold(409, '{"late_hold": true}') is False
    assert pp.park_late_hold(200, "not json") is False


def test_h91c3_2_the_wiring_reaches_the_live_scheduler_and_the_endpoint():
    """The hook sits in Scheduler._add_request_to_queue after the #1443
    dormant hold and before the waiting queue, only for new requests; the
    park handler passes the dormant-admit switch; the endpoint returns the
    field."""
    sched = open(os.path.join(_ROOT, "python/sglang/srt/managers/scheduler.py")).read()
    body = sched[sched.index("    def _add_request_to_queue(self"):]
    body = body[:body.index("\n    def ", 10)]
    i_dormant = body.index("hold.append(req)")
    i_late = body.index("if not is_retracted and self._weg2_d_park_hold_late(req):")
    i_queue = body.index("self.waiting_queue.append(req)")
    assert i_dormant < i_late < i_queue
    assert "late_hold_armed=_weg2_dormant_admit_armed()" in sched
    http = open(os.path.join(_ROOT, "python/sglang/srt/entrypoints/http_server.py")).read()
    assert '"late_hold": bool(getattr(ret, "late_hold", False))' in http


class LateD(FakeGroup):
    """D where the leg 2 of a mark in ``late_marks`` reaches the scheduler only
    once ``arrive`` is set (the tokenizer / HTTP pipe). Arriving after a park
    it is held (``queued``), as part B does with ``late_hold``."""

    def __init__(self, *a, late_hold: bool = True, **kw):
        super().__init__(*a, **kw)
        self.late_marks = set()
        self.late_hold = late_hold
        self.arrive = asyncio.Event()
        self.queued: Dict[str, str] = {}

    async def _generate(self, request):
        payload = await request.json()
        mark = self._mark(payload)
        rid = payload.get("rid") or mark
        if mark in self.late_marks:
            await self.arrive.wait()
        held = bool(self.park_bodies) and self.late_hold
        self.timeline.append(f"{'held' if held else 'gen'}:{mark}")
        self.gen_marks.append(mark)
        (self.queued if held else self.running)[rid] = mark
        try:
            if self.hold is not None:
                await self.hold.setdefault(mark, asyncio.Event()).wait()
            else:                                       # released: serve at once
                await asyncio.sleep(self.delay)
        finally:
            (self.queued if held else self.running).pop(rid, None)
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
        ans = {"success": True, "parked": sorted(self.running), "held": sorted(self.queued),
               "epoch": body.get("epoch"), "message": ""}
        if self.late_hold:
            ans["late_hold"] = True
        return web.json_response(ans)


def _late_harness(late_hold: bool) -> Harness:
    hh = Harness(awake="D", d_bs=4, tp_prefill_max_tokens=100_000,
                 d_wait_bound_s=0.6, p_phase_max_requests=6, drain_deadline_s=3.0)
    hh.d = LateD("D", late_hold=late_hold)
    hh.d.late_marks = {"late"}
    return hh


def test_h91c3_2_a_hand_off_in_flight_at_the_park_is_parked_not_drained():
    """Red on 636502eb97: the in-flight hand-off stayed in the flip ledger,
    the drain waited the whole window and W1b aborted it on D."""

    async def body():
        async with _late_harness(late_hold=True) as h:
            h.d.hold = {}
            t0 = h.post("s0")                                   # X route, decoding on D
            assert await _until(lambda: h.d.running, 10)
            tlate = h.post("late")                              # X route, still in the pipe
            assert await _until(lambda: len(h.front.groups["D"].outstanding) == 2, 10)
            assert "gen:late" not in h.d.timeline
            tl = h.post("L0", chars=400_000)                    # LONG: waits for P
            assert await _until(lambda: h.d.park_bodies, 10)    # the bound fired
            t_park = time.time()
            assert await _until(lambda: "rpc:release_memory_occupation" in h.d.timeline, 10)
            assert time.time() - t_park < 2.0, "the drain waited for the in-flight hand-off"
            assert h.front.counters["d_parked_in_flight"] == 1
            h.d.arrive.set()                                    # it lands now: D holds it
            assert await _until(lambda: "held:late" in h.d.timeline, 10)
            assert "rpc:abort_request" not in h.d.timeline, h.d.timeline
            assert await _until(lambda: "gen:L0" in h.p.timeline, 10)
            assert await _until(lambda: len([b for b in h.d.resume_bodies
                                             if b.get("tags") == ["kv_cache"]]) >= 1, 20)
            kv = [b for b in h.d.resume_bodies if b.get("tags") == ["kv_cache"]]
            assert kv[-1].get("parked_n") == 2, kv
            h.d.release_all()
            results = await asyncio.wait_for(asyncio.gather(t0, tlate, tl), 20)
            assert [s for s, _ in results] == [200, 200, 200]
            assert h.d.gen_marks.count("late") == 1             # never re-posted

    asyncio.run(body())


def test_h91c3_2_an_old_d_without_late_hold_is_drained_as_before():
    """No promise from D: the in-flight hand-off stays running for the front,
    the flip waits for it (the pre-H91c3 path, unchanged)."""

    async def body():
        async with _late_harness(late_hold=False) as h:
            h.d.hold = {}
            t0 = h.post("s0")
            assert await _until(lambda: h.d.running, 10)
            tlate = h.post("late")
            assert await _until(lambda: len(h.front.groups["D"].outstanding) == 2, 10)
            tl = h.post("L0", chars=400_000)
            assert await _until(lambda: h.d.park_bodies, 10)
            await asyncio.sleep(1.0)
            assert "rpc:release_memory_occupation" not in h.d.timeline
            assert h.front.counters["d_parked_in_flight"] == 0
            h.d.arrive.set()
            assert await _until(lambda: "gen:late" in h.d.timeline, 10)
            h.d.release("late")
            assert await _until(lambda: "rpc:release_memory_occupation" in h.d.timeline, 10)
            h.d.release_all()
            results = await asyncio.wait_for(asyncio.gather(t0, tlate, tl), 20)
            assert [s for s, _ in results] == [200, 200, 200]

    asyncio.run(body())


# ================================================================ H91c3-3
def _kv_resumes(d):
    return [b for b in d.resume_bodies if b.get("tags") == ["kv_cache"]]


def _seat_harness(bound_s: float = 0.6) -> Harness:
    from test_weg2_park_wait_h91c2 import HeldD

    hh = Harness(awake="P", p_concurrency=4, d_bs=4, tp_prefill_max_tokens=100_000,
                 d_wait_bound_s=bound_s, p_phase_max_requests=6, drain_deadline_s=3.0)
    hh.d = HeldD("D")
    hh.d.queued_marks = {"s1"}          # D's H95c cap n = 1 keeps s1 queued (emulated)
    return hh


def test_h91c3_3_an_x_route_past_the_phase_seats_waits_where_the_bound_sees_it():
    """Red on 636502eb97: s1 went to D and waited there behind the seat cap
    n = 1; the P queue was empty, the wait bound never fired."""

    async def body():
        async with _seat_harness() as h:
            h.d.hold = {}
            tl = h.post("L0", chars=400_000)                    # LONG: the phase's one hand-off
            assert await _until(lambda: h.d.running, 20)
            assert _kv_resumes(h.d)[-1].get("handoff_n") == 1   # -> n = 1
            assert h.front._d_phase_n == 1
            t1 = h.post("s1")                                   # X route, the n seat is taken
            assert await _until(lambda: h.front.counters["wait_bound_fired"] == 1, 5)
            assert h.front.counters["route_short"] == 0
            assert h.front.counters["short_phase_seats_full"] >= 1
            h.d.queued_marks = set()                            # next phase: n = 2
            assert await _until(lambda: "gen:s1" in h.p.timeline, 10)   # served via P
            assert await _until(lambda: h.front.awake == "D" and not h.front._d_parked, 20)
            kv = _kv_resumes(h.d)[-1]
            assert kv.get("handoff_n") == 1 and kv.get("parked_n") == 1, kv
            assert h.front._d_phase_n == 2
            # s1 reaches D in the new phase, then both end (release per mark:
            # HeldD, unlike FakeGroup, cannot serve a request after release_all)
            assert await _until(lambda: "gen:s1" in h.d.timeline, 10)
            h.d.release("L0")
            h.d.release("s1")
            results = await asyncio.wait_for(asyncio.gather(tl, t1), 20)
            assert [s for s, _ in results] == [200, 200]
            assert h.d.gen_marks.count("L0") == 1               # parked, never re-posted

    asyncio.run(body())


def test_h91c3_3_a_free_phase_seat_still_takes_the_x_route_on_d():
    """The phase's hand-off is done: its seat is free, the X-route request
    goes to D directly as before."""

    async def body():
        async with _seat_harness() as h:
            h.d.queued_marks = set()
            h.d.hold = {}
            tl = h.post("L0", chars=400_000)
            assert await _until(lambda: h.d.running, 20)
            h.d.release("L0")
            (s0, _) = await asyncio.wait_for(tl, 10)
            assert s0 == 200 and h.front.awake == "D"
            t1 = h.post("s1")
            assert await _until(lambda: h.d.running, 10)
            assert h.front.counters["route_short"] == 1
            assert h.front.counters["short_phase_seats_full"] == 0
            h.d.release_all()
            (s1, _) = await asyncio.wait_for(t1, 10)
            assert s1 == 200 and "gen:s1" not in h.p.timeline

    asyncio.run(body())


def test_h91c3_3_without_the_wait_bound_the_x_route_is_untouched():
    """--d-wait-bound-s 0 (H91 part C rule 3 off): the X route as before,
    whatever n is."""

    async def body():
        async with _seat_harness(bound_s=0.0) as h:
            h.d.queued_marks = set()
            h.d.hold = {}
            tl = h.post("L0", chars=400_000)
            assert await _until(lambda: h.d.running, 20)
            t1 = h.post("s1")
            assert await _until(lambda: len(h.d.running) == 2, 10)
            assert h.front.counters["route_short"] == 1
            assert h.front.counters["short_phase_seats_full"] == 0
            h.d.release_all()
            results = await asyncio.wait_for(asyncio.gather(tl, t1), 20)
            assert [s for s, _ in results] == [200, 200]

    asyncio.run(body())


def test_h91c3_3_the_front_derives_n_with_ds_own_function():
    assert pp.d_phase_seats(1, 0, 6) == ds.phase_seats(1, 0, cap=6).n == 1
    assert pp.d_phase_seats(3, 6, 6) == 6          # clamped to --d-bs
    assert pp.d_phase_seats(0, 0, 6) == 1          # at least one seat
