"""S87 (rc19 boot ..._10e9b2b492_1019_230333, front.log): the front's seat count against what D holds.

THE METAL, reconstructed rid by rid:
* No seat leaks: 3206 ``D-ADMIT`` and 3200 ``D-REFILL``, balanced per rid; the 6 open ones are live at the log's end.
  The open leg 2s rebuilt from acquire/release equal the logged ``d_outstanding``, the P-served hand-offs not yet
  admitted equal ``ready_for_d`` (every DP-WAIT sample) -- ``taken`` counts no dead rid. Its excess over ``n`` after
  06:00 is the live D line (4-11 hand-offs waiting for a seat) and legs D holds without a seat.
* The seat-count defect: ``SEAT-ROTATE`` (43x, 41 in hour 06) released the deferred parked rid's seat
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
    stream = mode != "reroute"
    pending = None
    if mode == "reroute":
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
            if mode == "reroute":
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
        for _ in range(100 if mode != "reroute" else 10):
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
