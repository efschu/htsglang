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
