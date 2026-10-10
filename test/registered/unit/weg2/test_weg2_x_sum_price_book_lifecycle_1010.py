"""X-SUM-PRICE: the book `DPrefillInflight` is written on every way a SHORT
reaches D and emptied on every way it does not (review xsum-review2-1009, B1/B4).

A line that is booked and never freed is a phantom: for GRANT_TTL_S (30 s) it
is added to every collected set's sum, so a small SHORT is sent to P's batch
(and a flip follows) for work D never had. A line that is not booked is the
y6 defect itself: D prefills more than X because the burst read an empty book.

One test per way, each red against the mutant that removes its line:
* the admitter pops a `d_direct` hand-over  (M3)           -> booked
* the client is gone before leg 2           (M2)           -> freed
* the handler is cancelled in the seat wait (cancel guard) -> freed
* the seat comes back None                                 -> freed
* the admitter's POST barrier expires (W36)                -> freed
* D-SHORT-DRAIN with the switch off reads no book (m1)     -> int23 behaviour

int24 follow-up (review xsum-review3-1010, R3-1), same rule, read with `live={rid}`:
* PARK-HANDBACK / W50 requeue / D-verdict REROUTE            -> freed before the wait on P
* cancel in `_x_exact_backfill`                              -> freed
* caller's grant at WINDOW_S=0 (the only booking there)      -> booked
* first content (main release) and `enter_leg2` (> TTL)      -> row ends at the first content,
                                                                lives while D prefills
"""
from __future__ import annotations

import asyncio
import os
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2.front import Front  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

X = 4096
TOKENS = 3000


def _front(drain_tokens=0):
    return Front("http://p", "http://d", "D", "xsum-lifecycle", "", 0, 0, {}, 45.0,
                 tp_prefill_max_tokens=X, d_short_drain_tokens=drain_tokens)


def _pending(f, rid, tokens=TOKENS, d_direct=True):
    fut = asyncio.get_running_loop().create_future()
    p = front_mod.Pending(rid, "/generate", {}, "x", time.time(), fut,
                          est_prompt=tokens, est_uncached=tokens, d_eligible=True)
    p.d_direct = d_direct
    return p


def _booked(f):
    return f._d_pf_book().pending_tokens(now=time.time(), live={})


async def _drop(*tasks):
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


class _Req:
    """What handle_generate reads from an aiohttp request."""

    def __init__(self, payload):
        self._payload = payload
        self.path = "/generate"

    async def json(self):
        return self._payload


def _payload(tokens):
    return {"text": "x" * int(tokens * front_mod.CHARS_PER_TOKEN),
            "sampling_params": {"max_new_tokens": 8}}


# ---- M3: the admitter's popleft -------------------------------------------------
def test_the_admitter_books_a_d_direct_hand_over_at_the_popleft():
    """Six ways hand a SHORT to D through `_ready_for_d` (SHORT-KEPT, SHORT-IN-FLIP,
    X-IDLE-REGRANT, hand-over, queue-take, D-SHORT-DRAIN). While it waits there it
    counts as `handed`; the admitter's popleft is the one place it leaves the
    deque for D, and from there it is D's prefill: its tokens must stay in
    `carried` until D's first content, or the next arrival reads D as idle."""

    async def body():
        f = _front()
        p = _pending(f, "h0")
        f._ready_for_d.append(p)
        assert f._d_carried_tokens() == TOKENS, "waiting in the deque: counted as handed"
        adm = asyncio.ensure_future(f.d_admitter())
        try:
            await asyncio.wait_for(p.fut, 3.0)
            assert not f._ready_for_d
            assert f._d_carried_tokens() == TOKENS, (
                "the admitter popped it and nothing booked it: carried fell to 0 while D prefills it")
        finally:
            await _drop(adm)

    asyncio.run(body())


def test_the_admitter_frees_the_booking_when_the_post_never_comes(monkeypatch):
    """W36: the client behind the rid never reached its POST. The seat is
    released; the booking made at the popleft goes with it (it would stand for
    the rest of GRANT_TTL_S)."""
    monkeypatch.setattr(front_mod, "POST_BARRIER_S", 0.2)

    async def body():
        f = _front()
        p = _pending(f, "w36")
        f._ready_for_d.append(p)
        adm = asyncio.ensure_future(f.d_admitter())
        try:
            await asyncio.wait_for(p.fut, 3.0)
            assert f._d_carried_tokens() == TOKENS
            await asyncio.sleep(0.6)
            assert f.counters["W36_Weg2AdmitterBarrierExpired"] == 1
            assert f._d_carried_tokens() == 0
        finally:
            await _drop(adm)

    asyncio.run(body())


# ---- M2: client gone before leg 2 ------------------------------------------------
def test_a_client_gone_before_leg_2_frees_the_booking(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_CLIENT_GONE_ABORT", "1")

    async def body():
        f = _front()
        f._d_pf_book().grant(rid="gone", tokens=TOKENS, now=time.time())
        assert _booked(f) == TOKENS
        req = types.SimpleNamespace(transport=None, path="/generate")  # tr is None: the client left
        resp = await f.leg2(req, "gone", {}, "x", False, None, seat=None)
        assert resp.status == 499
        assert _booked(f) == 0, "the request never reached D; its booking must not count 30 s as a phantom"

    asyncio.run(body())


# ---- the collect gate books before the seat; the handler frees on every way out ---
async def _handler_in_the_seat_wait(f, seat_wait):
    """Run the REAL handle_generate with `_acquire_short_seat` replaced by a
    stand-in that books like the collect gate does and then behaves as `seat_wait`."""
    seen = {}

    async def stand_in(rid, est_prompt, why=None, **kw):
        seen["rid"] = rid
        f._d_pf_book().grant(rid=rid, tokens=TOKENS, now=time.time())  # the gate's provisional booking
        return await seat_wait()

    f._acquire_short_seat = stand_in
    task = asyncio.ensure_future(f.handle_generate(_Req(_payload(500))))
    for _ in range(500):
        await asyncio.sleep(0)
        if "rid" in seen:
            break
    assert "rid" in seen, "handle_generate never reached the short seat"
    return seen["rid"], task


def test_a_cancel_in_the_seat_wait_frees_the_booking():
    """The gate booked, the handler was cancelled before the caller's grant:
    nothing else would ever free the line."""

    async def body():
        f = _front()

        async def forever():
            await asyncio.sleep(3600)

        rid, task = await _handler_in_the_seat_wait(f, forever)
        assert _booked(f) == TOKENS
        await _drop(task)
        assert _booked(f) == 0

    asyncio.run(body())


def test_a_seat_that_comes_back_none_frees_the_provisional_booking():
    """D refused (budget / phase): the SHORT falls through to P's batch. The
    gate's provisional booking is not D's work."""

    async def body():
        f = _front()

        async def refused():
            return None

        rid, task = await _handler_in_the_seat_wait(f, refused)
        for _ in range(50):
            await asyncio.sleep(0)
        try:
            assert _booked(f) == 0
        finally:
            await _drop(task)

    asyncio.run(body())


# ---- m1: D-SHORT-DRAIN with the switch off is int23 -------------------------------
@pytest.mark.parametrize("switch,moved", [("0", 1), ("1", 0)])
def test_d_short_drain_prices_the_hand_over_only_with_the_switch_on(monkeypatch, switch, moved):
    """A d_direct hand-over of 3000 waits in `_ready_for_d`; a queued SHORT of 3000
    asks to be drained (N = X = 4096). On: 3000 + 3000 > 4096, nothing moves.
    Off: the drain prices the queue alone, exactly as int23 did (3000 <= 4096)."""
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_DECODE_COLLECT_PREFILL_BUSY", switch)

    async def body():
        f = _front(drain_tokens=X)
        f._ready_for_d.append(_pending(f, "waiting"))
        q = _pending(f, "queued", d_direct=False)
        f.queue.append(q)
        assert f._d_short_drain(time.time()) == moved
        assert (q in f._ready_for_d) == bool(moved)

    asyncio.run(body())


# ---- review xsum-review3-1010 R3-1: the rest of the book's writers ------------------
# `_booked(f)` reads with `live={}`: a row in leg 2 is then always dead, which hides
# a row that a missing `done` leaves behind. The tests below read with
# `live={rid}`: D still holds the rid (the running decode, or the inner leg 2 that
# sets it in `outstanding` again), so a row that was not removed counts again.
def _live(f, rid):
    return f._d_pf_book().pending_tokens(now=time.time(), live={rid})


def _rpc_stub(f):
    async def rpc(group, path, body, timeout=0):
        return 200, "ok"

    f.rpc = rpc


async def _until_queued(f, p):
    for _ in range(500):
        await asyncio.sleep(0)
        if p in f.queue:
            return
    raise AssertionError("the request never reached P's queue")


def test_park_handback_frees_the_booking_before_it_waits_on_p():
    """PARK-HANDBACK: D held the leg unstarted at a park; the request joins P's
    batch. Its row is freed there -- the inner leg 2 sets the rid in `outstanding`
    again and would bring a row that was left standing back to life."""

    async def body():
        f = _front()
        _rpc_stub(f)
        p = _pending(f, "hb")
        f._d_pf_book().grant(rid="hb", tokens=TOKENS, now=time.time())
        f._d_pf_book().enter_leg2(rid="hb")
        assert _live(f, "hb") == TOKENS
        req = types.SimpleNamespace(path="/generate")
        task = asyncio.ensure_future(f._requeue_park_handback(req, "hb", {}, "x", False, p, None))
        try:
            await _until_queued(f, p)
            assert _live(f, "hb") == 0, "handed back to P's batch, the row must be gone"
        finally:
            await _drop(task)

    asyncio.run(body())


def test_a_w50_requeue_frees_the_booking_before_it_waits_on_p():
    """W50: D refused the SHORT (X gate); back on P's batch, same rule as above."""

    async def body():
        f = _front()
        p = _pending(f, "w50", d_direct=False)
        f._d_pf_book().grant(rid="w50", tokens=TOKENS, now=time.time())
        f._d_pf_book().enter_leg2(rid="w50")
        assert _live(f, "w50") == TOKENS
        req = types.SimpleNamespace(path="/generate")
        refusal = b'{"error": "W31 Weg2TpPrefillExceeded uncached=9000 X=4096"}'
        task = asyncio.ensure_future(
            f._requeue_after_x_refusal(req, "w50", {}, "x", False, p, None, refusal))
        try:
            await _until_queued(f, p)
            assert _live(f, "w50") == 0, "refused by D, back on P's batch: the row must be gone"
        finally:
            await _drop(task)

    asyncio.run(body())


def test_a_cancel_in_the_exact_backfill_frees_the_booking():
    """leg 2 has entered the book (`enter_leg2`) and is cancelled in
    `_x_exact_backfill`, before the try whose finally ends the leg: this `except`
    is the only place that frees the row, and D never got the request."""

    async def body():
        f = _front()

        async def cancelled(*a, **k):
            raise asyncio.CancelledError()

        f._x_exact_backfill = cancelled
        f._d_pf_book().grant(rid="bf", tokens=TOKENS, now=time.time())
        req = types.SimpleNamespace(path="/generate")
        with pytest.raises(asyncio.CancelledError):
            await f.leg2(req, "bf", {}, "x", False, None, seat=None)
        # `outstanding` keeps the rid on this path (the finally is not reached),
        # so the row is read as live: only the `done` makes it go away.
        assert _live(f, "bf") == 0

    asyncio.run(body())


def test_the_window_zero_grant_of_the_caller_books_the_short(monkeypatch):
    """K135 (int24; K132 in the xsum branch): with SGLANG_WEG2_DECODE_COLLECT_WINDOW_S=0 the
    collect gate returns before it books, so the caller's grant after the seat is
    the ONLY booking of a granted SHORT -- and it prices the queue-take and the
    D-SHORT-DRAIN."""
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", "0")
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_DECODE_COLLECT_PREFILL_BUSY", "1")

    async def body():
        f = _front()
        seen = {}

        async def seat_without_booking(rid, est_prompt, why=None, **kw):
            seen["rid"] = rid
            seen["uncached"] = kw.get("uncached")
            return types.SimpleNamespace(release=lambda *a, **k: None)  # a seat; the gate did not book

        async def stop_at_leg2(request, rid, payload, text, stream, pending=None, seat=None, **kw):
            seen["booked_at_leg2"] = _live(f, rid)
            return types.SimpleNamespace(status=200)

        f._acquire_short_seat = seat_without_booking
        f.leg2 = stop_at_leg2
        await f.handle_generate(_Req(_payload(500)))
        assert seen.get("uncached"), "handle_generate never took the short seat"
        assert seen["booked_at_leg2"] == seen["uncached"], (
            "the SHORT was granted on D and nothing booked it: the next arrival, the queue-take "
            "and the D-SHORT-DRAIN read an idle D")

    asyncio.run(body())


# ---- leg 2 against a D that streams: the two ends of the row's life ----------------
class _FakeStreamBody:
    """D's stream as leg 2 reads it (`readany`, also under ROS's polled read and
    `iter_any`): the first chunk waits for `gate` (D prefills until then), the
    next never comes (D decodes: the rid stays in `outstanding`)."""

    def __init__(self, gate, first):
        self.gate, self.first, self.reads, self.reading_on = gate, first, 0, asyncio.Event()

    async def readany(self):
        self.reads += 1
        if self.reads == 1:
            await self.gate.wait()
            return self.first
        self.reading_on.set()  # past the first chunk: D decodes
        await asyncio.sleep(3600)
        return b""

    async def iter_any(self):
        yield await self.readany()


class _FakeD:
    """What leg 2 reads of aiohttp's `session.post(...)` and its response."""

    def __init__(self, body):
        self.status, self.content_type, self.content = 200, "text/event-stream", body

    def post(self, *a, **k):
        d = self

        class _Cm:
            async def __aenter__(self):
                return d

            async def __aexit__(self, *exc):
                return False

        return _Cm()


def test_the_row_lives_while_d_prefills_and_ends_at_the_first_content():
    """enter_leg2: the row is kept as long as D holds the rid, however long D
    prefills (a prefill over GRANT_TTL_S must still count). First content: D's
    prefill is over, the row ends there and not with the whole decode (D holds the
    rid for all of it, so a row left standing would be a phantom the entire decode)."""
    from aiohttp.test_utils import make_mocked_request

    from sglang.srt.weg2.d_prefill_inflight import GRANT_TTL_S

    async def body():
        f = _front()
        gate = asyncio.Event()
        stream = _FakeStreamBody(gate, b'data: {"text": "hi"}\n\n')
        f.session = _FakeD(stream)
        f._d_pf_book().grant(rid="fc", tokens=TOKENS, now=time.time())
        req = make_mocked_request("POST", "/generate")
        task = asyncio.ensure_future(f.leg2(req, "fc", {"stream": True}, "x", True, None, seat=None))
        try:
            for _ in range(500):  # leg 2 is at D, waiting for the first chunk
                await asyncio.sleep(0)
                if "fc" in f.groups["D"].outstanding:
                    break
            await asyncio.sleep(0.05)
            late = time.time() + GRANT_TTL_S + 5
            assert f._d_pf_book().pending_tokens(now=late, live={"fc"}) == TOKENS, (
                "D still prefills: the row must outlive the grant TTL while D holds the rid")
            gate.set()
            await asyncio.wait_for(stream.reading_on.wait(), 5.0)  # first content passed, D decodes
            assert "fc" in f.groups["D"].outstanding
            assert _live(f, "fc") == 0, "first content = prefill over: the row ends there"
        finally:
            await _drop(task)

    asyncio.run(body())


def test_a_d_verdict_reroute_frees_the_booking_before_it_waits_on_p():
    """leg 2 priced D's answer over X: the request rejoins P's batch (WEG2-REROUTE).
    The row is freed there; the inner leg 2 books anew."""

    async def body():
        f = _front()
        _rpc_stub(f)
        p = _pending(f, "rr", d_direct=False)

        async def read():
            return b'{"text": "x", "meta_info": {"prompt_tokens": 20000, "cached_tokens": 0, "completion_tokens": 1}}'

        d = _FakeD(None)
        d.content_type, d.read = "application/json", read
        f.session = d
        f._d_pf_book().grant(rid="rr", tokens=TOKENS, now=time.time())
        req = types.SimpleNamespace(path="/generate")
        task = asyncio.ensure_future(f.leg2(req, "rr", {}, "x", False, p, seat=None))
        try:
            await _until_queued(f, p)
            assert f.counters["reroute"] == 1
            assert _live(f, "rr") == 0
        finally:
            await _drop(task)

    asyncio.run(body())
