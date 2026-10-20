"""S87 (rc19 boot ..._10e9b2b492_1019_230333, front.log): the front's seat count against what D holds.

THE METAL, reconstructed rid by rid:
* No seat leaks: 3206 ``D-ADMIT`` and 3200 ``D-REFILL``, balanced per rid; the 6 open ones are live at the log's end.
  The open leg 2s rebuilt from acquire/release equal the logged ``d_outstanding``, the P-served hand-offs not yet
  admitted equal ``ready_for_d`` (every DP-WAIT sample) -- ``taken`` counts no dead rid. Its excess over ``n`` after
  06:00 is the live D line (4-11 hand-offs waiting for a seat) and legs D holds without a seat.
* The seat-count defect: ``SEAT-ROTATE`` (54x, 47 in hour 06) released the deferred parked rid's seat
  (``D-REFILL freed_by=rotate``) and passed its permit on, while D resumed the rid anyway (``PARK-RESUME`` lists it
  beside ``PARK-RESUME-DEFERRED``) and decoded it: up to 3 legs at a time without a seat next to 6 seated ones
  (``PARK-RUNNING parked=7``). pdflip-166-3095: rotated 4 times (06:25:50, 06:26:23, 06:26:52, 06:27:15), released
  once, seatless on D until ``PDFLIP-SERVED`` 06:27:23.

THE FIX: a rotated rid keeps its seat as a DEBT (``Seat.to_credit``): its permit goes on to the older request the
rotation chose, the next freed permit settles the debt before a younger request is admitted -- unless an older one
waits (PRIO=ALTER) -- and the seat is released only at leg 2's end, as every seat. ``S87 SEAT-LEG-UNSEATED`` names a
leg 2 D holds without a seat (counted, never healed).

Part B drives the real ``leg2`` against a fake D (in-process aiohttp, as test_pdflip_leg2_client_close_h61): every
end path returns its seat exactly once -- client abort, D's stream broken (the 05:45:29/05:54:16
``ClientPayloadError`` class), reroute (before the re-queue)."""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import aiohttp  # noqa: E402
import pytest  # noqa: E402
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front import Front  # noqa: E402

D_BS = 6
OUT = "pdflip-166-3095"                 # the metal's rotated rid (youngest parked)
PARKED_OLDER = ["pdflip-150-3050", "pdflip-150-3051", "pdflip-150-3052", "pdflip-150-3053", "pdflip-150-3054"]


def _front():
    f = object.__new__(F.Front)
    f.epoch, f.d_bs = 183, D_BS
    f.counters = collections.Counter()
    f._d_seat = asyncio.Semaphore(0)
    f._d_seats_live = set()
    f._ready_for_d = collections.deque()
    return f


def _waiter(loop, rid):
    p = F.Pending(rid, "/v1/messages", {}, "t", time.time() - 300, loop.create_future(),
                  est_prompt=1000, est_uncached=1000, span_known=True)
    p.leg1_done, p.t_ready = True, time.time() - 250
    return p


def _seats_of(f, rid):
    return [s for s in f._d_seats_live if s.rid == rid]


def _full(f):
    """Six normal seats held (permits 0): five older parked, the youngest is OUT."""
    seats = {r: F.Seat(f, r, "batch") for r in PARKED_OLDER + [OUT]}
    assert f._d_seat._value == 0
    return seats


def _admit(f, rid):
    """The d_admitter's step for the rotation's taker: the permit, then its Seat (the hand-off leaves the D line)."""
    assert f._d_seat._value >= 1, "no permit for the older request the rotation chose"
    f._d_seat._value -= 1
    for q in list(f._ready_for_d):
        if q.rid == rid:
            f._ready_for_d.remove(q)
            if not q.fut.done():
                q.fut.set_result(True)
    return F.Seat(f, rid, "batch")


def _rotate(f, loop, parked, rid_in):
    """One P->D wake's rotation: the taker waits in the D line (``_ready_for_d``, where the plan's waiters come
    from), the plan defers the youngest parked one."""
    w = _waiter(loop, rid_in)
    f._ready_for_d.append(w)
    return f._seat_rotate_plan(parked, [w])


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


# ---- (A) the rotation keeps the seat -------------------------------------------------------------------------------

def test_rc19_166_3095_holds_exactly_one_seat_through_four_rotations(loop):
    """RED on rc19: after the first rotation the rid holds NO seat while D runs it (D-REFILL freed_by=rotate)."""
    f = _front()
    seats = _full(f)
    parked = PARKED_OLDER + [OUT]
    takers = ["pdflip-140-3001", "pdflip-141-3002", "pdflip-142-3003", "pdflip-143-3004"]
    for i, rid_in in enumerate(takers):
        out = _rotate(f, loop, parked, rid_in)
        assert out == [OUT]
        mine = _seats_of(f, OUT)
        assert len(mine) == 1, "rotation %d: %s holds %d seats while D runs it" % (i + 1, OUT, len(mine))
        assert mine[0].held and mine[0].on_credit
        if i == 0:
            _admit(f, rid_in)                       # its permit went on: the older taker is seated
        else:
            assert f._d_seat._value == 0, "a second rotation of the same rid minted a permit"
            f._ready_for_d.clear()
    assert not Front._seat_leg_unseated(f, {r: 0 for r in parked + [takers[0]]}, ())
    seats[OUT].release("leg2_finished")             # its leg ends: the seat goes, a debt mints no permit
    assert not _seats_of(f, OUT) and f._d_seat._value == 0
    for s in list(f._d_seats_live):
        s.release("leg2_finished")
    assert f._d_seat._value == D_BS, "the permits drifted"


def test_the_next_freed_seat_settles_the_debt_before_a_younger_request(loop):
    f = _front()
    seats = _full(f)
    _rotate(f, loop, PARKED_OLDER + [OUT], "pdflip-140-3001")
    _admit(f, "pdflip-140-3001")
    f._ready_for_d.append(_waiter(loop, "pdflip-170-3110"))     # YOUNGER than OUT: it does not pass the debt
    seats["pdflip-150-3050"].release("leg2_finished")
    assert _seats_of(f, OUT)[0].on_credit is False, "the debt of the rid D runs was not settled"
    assert f._d_seat._value == 0, "a younger request got a permit while D still runs the rotated rid"
    assert f.counters["d_seat_credit_settled"] == 1


def test_prio_alter_an_older_waiter_is_not_passed_over_by_the_settlement(loop):
    """An OLDER request waits in the D line when a seat frees: the permit goes to it (semaphore), the debt stays for
    the next freed seat."""
    f = _front()
    seats = _full(f)
    _rotate(f, loop, PARKED_OLDER + [OUT], "pdflip-140-3001")
    _admit(f, "pdflip-140-3001")
    older = _waiter(loop, "pdflip-160-3080")                     # older than OUT (3080 < 3095)
    f._ready_for_d.append(older)
    seats["pdflip-150-3050"].release("leg2_finished")
    assert f._d_seat._value == 1, "the settlement overtook an older waiting request"
    assert _seats_of(f, OUT)[0].on_credit is True
    assert f.counters["d_seat_credit_yield_older"] == 1
    _admit(f, older.rid)                                         # the older one is seated (it leaves the line)
    seats["pdflip-150-3051"].release("leg2_finished")            # nobody older waits now: the debt is settled
    assert _seats_of(f, OUT)[0].on_credit is False and f._d_seat._value == 0


def test_the_invariant_names_a_leg_d_holds_without_a_seat(caplog):
    f = _front()
    F.Seat(f, "pdflip-1-1", "batch")
    caplog.set_level(logging.ERROR)
    bad = Front._seat_leg_unseated(f, {"pdflip-1-1": 0, OUT: 0}, ())
    assert bad == [OUT]
    assert any("S87 SEAT-LEG-UNSEATED n=1 rids=['%s']" % OUT in m for m in caplog.messages)
    n = len(caplog.messages)
    Front._seat_leg_unseated(f, {"pdflip-1-1": 0, OUT: 0}, ())    # unchanged set: named once
    assert len(caplog.messages) == n
    assert Front._seat_leg_unseated(f, {"pdflip-1-1": 0, OUT: 0}, {OUT}) == []   # a RESUME-VIA-P hold runs no D work


# ---- (B) every end path returns its seat exactly once ---------------------------------------------------------------

PROMPT = "a prompt"


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _fake_d(mode):
    async def chat(request):
        if mode == "w50":
            # D's named X refusal (W31 PdFlipTpPrefillExceeded), the non-stream 503 shape
            return web.json_response({"error": {"message": "W31 " + F.X_REFUSAL_MARKER + " rid=r1 extent=20000 > "
                                                "riegel=4096"}}, status=503)
        if mode == "reroute":
            return web.json_response({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                                      "usage": {"prompt_tokens": 20000, "completion_tokens": 1,
                                                "prompt_tokens_details": {"cached_tokens": 0}}})
        resp = web.StreamResponse(status=200)
        resp.content_type = "text/event-stream"
        if mode == "broken":
            resp.headers["Content-Length"] = "100000"          # D dies mid-body: ClientPayloadError at the front
        await resp.prepare(request)
        try:
            for i in range(3):
                await resp.write(_sse({"choices": [{"index": 0, "delta": {"content": f"t{i}"},
                                                    "finish_reason": None}]}))
                await asyncio.sleep(0.2)
            if mode == "broken":
                request.transport.close()
                return resp
            await resp.write(_sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
            await resp.write(_sse({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                                                            "prompt_tokens_details": {"cached_tokens": 9}}}))
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
        except (ConnectionResetError, RuntimeError):
            pass
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/get_server_info", lambda r: web.json_response({}))
    return app


async def _leg(mode, hang_up=False, state=None):
    d = TestServer(_fake_d(mode))
    await d.start_server()
    url = str(d.make_url("")).rstrip("/")
    front = Front("http://p", url, "D", "t", "", 0, 0, {}, 45.0, carrier_max_tokens=27466,
                  tp_prefill_max_tokens=4096)
    front.groups["D"].url = url
    front.session = aiohttp.ClientSession()
    seat = F.Seat(front, "r1", "batch")
    v0 = front._d_seat._value
    state = {} if state is None else state
    state["front"], state["seat"], state["v0"] = front, seat, v0
    stream = mode not in ("reroute", "w50")
    pending = None
    if mode in ("reroute", "w50"):
        pending = F.Pending("r1", "/v1/chat/completions", {}, PROMPT, time.time(),
                            asyncio.get_running_loop().create_future(), est_prompt=20000, est_uncached=100)
        pending.seat = seat

    async def handler(request):
        payload = await request.json()
        try:
            return await front.leg2(request, "r1", payload, PROMPT, stream, pending, seat=seat)
        finally:
            state["done"] = True

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    try:
        async with aiohttp.ClientSession() as cs:
            task = asyncio.ensure_future(cs.post(f"http://{host}:{port}/v1/chat/completions",
                                                 json={"messages": [{"role": "user", "content": PROMPT}],
                                                       "stream": stream}))
            if mode in ("reroute", "w50"):
                for _ in range(200):                      # the re-queue: the seat must be back BEFORE it waits
                    if pending in front.queue:
                        break
                    await asyncio.sleep(0.02)
                state["at_requeue"] = (seat.held, front._d_seat._value - v0, pending.seat)
                if not pending.fut.done():
                    pending.fut.set_exception(F.PdFlipStop("W0", "test end"))
                try:   # the re-queued leg's own end is the controller's; bounded here, its finally runs at cleanup
                    await asyncio.wait_for(task, 3.0)
                except Exception:  # noqa: BLE001
                    task.cancel()
            else:
                resp = await task

                async def _read():
                    async for chunk in resp.content.iter_any():
                        if hang_up and b"t0" in chunk:
                            return
                try:   # a broken D leaves the client's stream open: read until the front's leg 2 is over
                    await asyncio.wait_for(_read(), 5.0)
                except (asyncio.TimeoutError, aiohttp.ClientError):
                    pass
                resp.close()
        for _ in range(100 if mode not in ("reroute", "w50") else 10):
            if state.get("done"):
                break
            await asyncio.sleep(0.02)
    finally:
        await runner.cleanup()
        await front.session.close()
        await d.close()
    return front, seat, v0, state


def test_end_path_client_abort_returns_the_seat_once():
    front, seat, v0, st = asyncio.run(_leg("normal", hang_up=True))
    assert st.get("done")
    assert not seat.held and seat not in front._d_seats_live
    assert front._d_seat._value - v0 == 1


def test_end_path_d_stream_broken_returns_the_seat_once():
    front, seat, v0, st = asyncio.run(_leg("broken"))
    assert st.get("done")
    assert front.counters["leg2_failures"] >= 1 or front.groups["D"].served == 0
    assert not seat.held and front._d_seat._value - v0 == 1


def test_end_path_served_returns_the_seat_once():
    front, seat, v0, st = asyncio.run(_leg("normal"))
    assert not seat.held and front._d_seat._value - v0 == 1


def test_end_path_reroute_returns_the_seat_before_the_requeue():
    front, seat, v0, st = asyncio.run(_leg("reroute"))
    held, gained, pseat = st["at_requeue"]
    assert held is False and gained == 1 and pseat is None, st["at_requeue"]
    assert front._d_seat._value - v0 == 1, "the reroute's seat came back twice or not at all"


def test_end_path_on_credit_victim_settles_the_head_and_the_permits_add_up(loop):
    """ON-CREDIT: the head is seated on credit, the displaced victim keeps its normal seat; the victim's end settles
    the head's debt (nobody older waits) and the permits come back to d_bs exactly."""
    f = _front()
    seats = {r: F.Seat(f, r, "short") for r in PARKED_OLDER + ["pdflip-116-2759"]}
    head = F.Seat(f, "pdflip-116-2762", "short", on_credit=True)      # the metal's seat=7/6
    seats["pdflip-116-2759"].release("leg2_finished")               # the victim ends (abort or served)
    assert head.on_credit is False and f._d_seat._value == 0
    for s in list(f._d_seats_live):
        s.release("leg2_finished")
    assert f._d_seat._value == D_BS


def test_metal_park_no_seat_pattern_the_waiting_short_gets_its_seat_in_age_order(loop):
    """rc19 06:53:19 (``SET-VERDICT ... pdflip-226-3228:23 ... seats_free=0 verdict=PARK_NO_SEAT why=R5``): D held 6
    seats and 3 legs without one -- the rotated pdflip-190-3140, -214-3200, -230-3244. RED on rc19: those three hold
    no seat, the count under-reads D by 3. GREEN: each holds exactly one (a debt); the kurz pdflip-226-3228 waiting at
    the arrival seat is not passed over by the YOUNGER debtor 3244 -- the freed seats go 3140, 3200 (older), then the
    short, never a younger request first. Nothing here asserts where a seatless kurz request goes: under SOLL R1a it
    may ride a flip decided by the rules; the 06:53 flip's own defects (p_only by state counted as big, R7; the R6
    rest of a prefill not really running) are S2b's (planner correction 20.10. 07:45Z)."""
    f = _front()
    seated = ["pdflip-150-3050", "pdflip-150-3051", "pdflip-150-3052"]
    rotated = ["pdflip-190-3140", "pdflip-214-3200", "pdflip-230-3244"]
    seats = {r: F.Seat(f, r, "batch") for r in seated + rotated}
    takers = ["pdflip-140-3001", "pdflip-141-3002", "pdflip-142-3003"]
    for k, rid_in in enumerate(takers):          # one rotation per wake: the youngest still-seated parked one
        parked = seated + takers[:k] + rotated[:3 - k]
        assert _rotate(f, loop, parked, rid_in) == [rotated[2 - k]]
        _admit(f, rid_in)
    legs = {r: 0 for r in seated + rotated + takers}
    assert Front._seat_leg_unseated(f, legs, ()) == [], "a leg D holds has no seat (the count under-reads D)"
    assert all(len(_seats_of(f, r)) == 1 and _seats_of(f, r)[0].on_credit for r in rotated)
    assert Front._seats_in_use(f) == len(legs), "the seats in use are not what D holds"
    short = "pdflip-226-3228"
    f._asr_state = {"waiters": {short: time.time()}, "fits": {}, "age_plan": {}}
    order = []
    for r in seated:                            # three seats free, one after the other
        seats[r].release("leg2_finished")
        if f._d_seat._value:
            order.append(short)
            f._d_seat._value -= 1
            F.Seat(f, short, "short")
            f._asr_state["waiters"].pop(short, None)
        else:
            order.append(next(x for x in rotated if not _seats_of(f, x)[0].on_credit and x not in order))
    assert order == ["pdflip-190-3140", "pdflip-214-3200", short], order
    assert _seats_of(f, "pdflip-230-3244")[0].on_credit, "the younger debtor took the seat before the older short"


# ---- S87r (review done/nf-review-s87-1020.md): A1 the permit is never lost, A3 the three untested end paths,
# ---- A4 the D-line waiters the admitter skips do not hold a debt's settlement back.


def test_a1_exception_in_age_check_must_not_lose_a_permit(loop, monkeypatch):
    """A1 (reviewer test test_danger_exception_in_age_check_must_not_lose_a_permit, RED on 69835d46f1: permits 0):
    release() runs in leg 2's finally and has set held=False / left _d_seats_live BEFORE _pass_permit; a raising age
    check (_older_waiter_than / _sa_key / _asr_verdict_stale) must still return the permit."""
    f = _front()
    s1 = F.Seat(f, "pdflip-1-10", "batch")
    s2 = F.Seat(f, "pdflip-1-20", "short", on_credit=True)

    def boom(*a, **k):
        raise KeyError("x")
    monkeypatch.setattr(Front, "_older_waiter_than", boom)
    s1.release("leg2_finished")          # must not raise out of leg 2's finally either
    s2.release("leg2_finished")
    assert f._d_seat._value == 1, "the permit of s1 was lost (permits %d)" % f._d_seat._value
    assert f.counters["d_seat_pass_permit_raised"] == 1


def test_a3_end_path_w50_requeue_returns_the_seat_before_the_requeue():
    """A3: D refuses by name (W31 -> W50 requeue): the seat is back BEFORE the request waits in P's queue, once."""
    front, seat, v0, st = asyncio.run(_leg("w50"))
    held, gained, pseat = st["at_requeue"]
    assert held is False and gained == 1 and pseat is None, st["at_requeue"]
    assert front.counters["W50_PdFlipTpPrefillExceeded"] == 1
    assert front._d_seat._value - v0 == 1, "the W50 requeue's seat came back twice or not at all"


def _bare_front():
    front = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, carrier_max_tokens=27466,
                  tp_prefill_max_tokens=4096)
    return front


def test_a3_end_path_park_handback_returns_the_seat_before_the_requeue():
    """A3: PARK-HANDBACK closes D's unstarted prefill leg; the seat is back before the request joins P's queue."""
    async def run():
        front = _bare_front()

        async def rpc(*a, **k):
            raise ConnectionError("no D in this test")
        front.rpc = rpc
        seat = F.Seat(front, "r1", "batch")
        v0 = front._d_seat._value
        p = F.Pending("r1", "/v1/chat/completions", {}, PROMPT, time.time(),
                      asyncio.get_running_loop().create_future(), est_prompt=20000, est_uncached=100)
        p.seat = seat
        task = asyncio.ensure_future(front._requeue_park_handback(None, "r1", {}, PROMPT, False, p, seat))
        for _ in range(200):
            if p in front.queue:
                break
            await asyncio.sleep(0.01)
        at_requeue = (p in front.queue, seat.held, front._d_seat._value - v0, p.seat)
        p.fut.set_exception(F.PdFlipStop("W0", "test end"))
        resp = await asyncio.wait_for(task, 3.0)
        return front, seat, v0, at_requeue, resp

    front, seat, v0, at_requeue, resp = asyncio.run(run())
    assert at_requeue == (True, False, 1, None), at_requeue
    assert resp.status == 503
    assert front._d_seat._value - v0 == 1 and seat not in front._d_seats_live


def test_a3_end_path_client_gone_before_d_returns_the_seat(monkeypatch):
    """A3: H102 -- the client left before its leg 2 was posted: D is not asked, the seat goes straight back, once."""
    monkeypatch.setattr(F, "client_gone", lambda request: True)
    monkeypatch.setattr(F, "response_complete", lambda request: False)

    async def run():
        front = _bare_front()
        seat = F.Seat(front, "r1", "batch")
        v0 = front._d_seat._value
        resp = await front.leg2(object(), "r1", {}, PROMPT, False, None, seat=seat)
        return front, seat, v0, resp

    front, seat, v0, resp = asyncio.run(run())
    assert resp.status == 499
    assert front.counters["client_gone_pre_d"] == 1
    assert not seat.held and seat not in front._d_seats_live
    assert front._d_seat._value - v0 == 1
    assert "r1" not in front.groups["D"].outstanding, "D must not be asked for it"


HEAD_OLD = "pdflip-1-10"      # the D line's head, older than the debtor
DEBTOR = "pdflip-1-30"        # an S46 F5 AGE PLAN head seated on credit
YOUNG = "pdflip-1-50"         # a younger hand-off the SA backfill would seat


@pytest.mark.parametrize("why", ["kv_blocked", "token_budget", "s46_f4_hold", "s60_f3_yield"])
def test_a4_skipped_older_d_line_entry_does_not_hold_the_settlement_back(loop, why):
    """A4: the admitter skipped the older head on its last pass (``_d_admit_skip``) -- it would not take a passed
    permit, so the freed permit settles the debt (as rc19 did) instead of going to the semaphore, where the admitter
    would seat a YOUNGER request and leave the debt standing (D longer over d_bs, exactly under KV pressure)."""
    f = _front()
    victim = F.Seat(f, "pdflip-1-5", "batch")
    head = F.Seat(f, DEBTOR, "short", on_credit=True)
    f._ready_for_d.append(_waiter(loop, HEAD_OLD))
    Front._d_admit_skip_note(f, HEAD_OLD, why)
    assert Front._older_waiter_than(f, DEBTOR) is False
    victim.release("leg2_finished")
    assert head.on_credit is False and f._d_seat._value == 0, (head.on_credit, f._d_seat._value)


def test_a4_an_older_head_the_admitter_takes_still_wins_the_permit(loop):
    """A4, the other direction (S87 PRIO=ALTER unchanged): an older head the admitter would seat (no skip verdict,
    or a cleared one) takes the permit; the debt waits."""
    f = _front()
    victim = F.Seat(f, "pdflip-1-5", "batch")
    head = F.Seat(f, DEBTOR, "short", on_credit=True)
    f._ready_for_d.append(_waiter(loop, HEAD_OLD))
    Front._d_admit_skip_note(f, HEAD_OLD, "kv_blocked")
    Front._d_admit_skip_note(f, HEAD_OLD, None)          # the next pass found it admissible
    assert Front._older_waiter_than(f, DEBTOR) is True
    victim.release("leg2_finished")
    assert head.on_credit is True and f._d_seat._value == 1


def _admitter_front(loop_entries, fits, monkeypatch, token_blocks=False, dormant=False, yields=False):
    front = _bare_front()
    front._d_seat = asyncio.Semaphore(0)       # every seat held: the admitter parks at the acquire
    front._d_accepts_leg2 = lambda: True

    async def reading():
        return {}
    front._d_reading_if_armed = reading
    front._d_budget_fits = lambda est, rd, l1: fits(est)
    front._d_token_budget_blocks = (lambda rid, est, rd, l1: token_blocks(est)) if callable(token_blocks) else (
        lambda rid, est, rd, l1: token_blocks)
    front._hl_check = lambda p, where: False
    monkeypatch.setattr(Front, "_dormant_hold_for_older", lambda self, p: dormant and p.rid == HEAD_OLD)
    monkeypatch.setattr(Front, "_d_line_admit_yields", lambda self, p: yields and p.rid == HEAD_OLD)
    for q in loop_entries:
        front._ready_for_d.append(q)
    return front


def _pend_est(rid, est):
    p = F.Pending(rid, "/v1/messages", {}, "t", time.time() - 300, asyncio.get_running_loop().create_future(),
                  est_prompt=est, est_uncached=est, span_known=True)
    p.leg1_done, p.t_ready = True, time.time() - 250
    return p


@pytest.mark.parametrize("case", ["kv_blocked", "token_budget", "s46_f4_hold", "s60_f3_yield"])
def test_a4_the_real_d_admitter_records_its_skip_and_the_debt_settles(monkeypatch, case):
    """A4 against the REAL ``d_admitter`` loop: the older head is skipped (KV-blocked with a fitting younger
    backfill / token budget / S46 F4 hold / S60 F3 yield). A normal seat then frees: the debt of the AGE PLAN head
    DEBTOR (younger than the skipped head, older than the backfill) is settled -- the permit does not go to the
    semaphore for the younger one. Before A4 the skipped head counted as an older waiter: permit to the semaphore,
    the admitter seats YOUNG, the debt stands."""
    async def run():
        head = _pend_est(HEAD_OLD, 5000)
        young = _pend_est(YOUNG, 500)
        front = _admitter_front(
            [head, young], fits=lambda est: est <= (1000 if case == "kv_blocked" else 10 ** 9),
            monkeypatch=monkeypatch, token_blocks=(case == "token_budget"),
            dormant=(case == "s46_f4_hold"), yields=(case == "s60_f3_yield"))
        victim = F.Seat(front, "pdflip-1-5", "batch")
        debt = F.Seat(front, DEBTOR, "short", on_credit=True)
        task = asyncio.ensure_future(front.d_admitter())
        for _ in range(40):
            await asyncio.sleep(0.05)
            if HEAD_OLD in (front.__dict__.get("_d_admit_skip") or {}):
                break
        skip = dict(front.__dict__.get("_d_admit_skip") or {})
        victim.release("leg2_finished")
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        return front, skip, debt

    front, skip, debt = asyncio.run(run())
    assert skip.get(HEAD_OLD) == case, skip
    assert debt.on_credit is False, "the skipped older head held the debt's settlement back (S87r A4)"
    assert front.counters["d_seat_credit_settled"] == 1
    assert not any(s.rid == YOUNG for s in front._d_seats_live), "the younger one took the permit past the debt"


def test_a4_an_older_done_entry_does_not_hold_the_settlement_back(loop):
    """A resolved / failed / cancelled D-line entry takes no seat (the admitter pops it, d_admit_skipped_done): it
    is no older waiter, the freed permit settles the debt (review mutant ML)."""
    f = _front()
    victim = F.Seat(f, "pdflip-1-5", "batch")
    head = F.Seat(f, DEBTOR, "short", on_credit=True)
    gone = _waiter(loop, HEAD_OLD)
    gone.fut.cancel()
    f._ready_for_d.append(gone)
    victim.release("leg2_finished")
    assert head.on_credit is False and f._d_seat._value == 0


def test_a4_the_real_d_admitter_clears_its_skip_when_the_head_becomes_admissible(monkeypatch):
    """A4, the clearing half against the REAL ``d_admitter``: the older head is skipped on one pass (no room: no
    younger fits, its token budget blocks), then the room is there. The admitter clears its verdict, so the head counts again as the older waiter: the
    freed permit goes to the semaphore and the HEAD takes it -- the younger debt waits (PRIO=ALTER)."""
    async def run():
        room = {"fits": False}
        head = _pend_est(HEAD_OLD, 5000)
        young = _pend_est(YOUNG, 6000)
        # the gate prices with the same room the probe reads: no room -> no backfill fits AND the head's token
        # budget blocks (FIX 4a), the head is skipped; room -> it passes both and takes the next permit
        front = _admitter_front([head, young], fits=lambda est: room["fits"], monkeypatch=monkeypatch,
                                token_blocks=lambda est: not room["fits"])
        victim = F.Seat(front, "pdflip-1-5", "batch")
        debt = F.Seat(front, DEBTOR, "short", on_credit=True)
        task = asyncio.ensure_future(front.d_admitter())
        for _ in range(40):
            await asyncio.sleep(0.05)
            if (front.__dict__.get("_d_admit_skip") or {}).get(HEAD_OLD) == "token_budget":
                break
        blocked = dict(front.__dict__.get("_d_admit_skip") or {})
        room["fits"] = True
        for _ in range(40):
            await asyncio.sleep(0.05)
            if HEAD_OLD not in (front.__dict__.get("_d_admit_skip") or {}):
                break
        cleared = dict(front.__dict__.get("_d_admit_skip") or {})
        victim.release("leg2_finished")
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        return front, blocked, cleared, debt

    front, blocked, cleared, debt = asyncio.run(run())
    assert blocked.get(HEAD_OLD) == "token_budget", blocked
    assert HEAD_OLD not in cleared, cleared
    assert debt.on_credit is True, "the debt was settled past the older admissible head"
    assert front.counters["d_seat_credit_yield_older"] == 1
