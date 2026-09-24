# SPDX-License-Identifier: Apache-2.0
"""RC2 review (24.09.): the idle-policy SHORT drain must not take back what
D's own #915 budget just refused.

`--d-short-drain-tokens N` (idle policy b, 95049973e0) hands a queued
SHORT-only backlog to D. Its field contract (`Pending.d_eligible`) is a SHORT
"queued only because of the PHASE". The BATCH site set it for EVERY SHORT that
fell through to BATCH, including the one `_acquire_short_seat` refused on D's
#915 budget (FIX 4a: "falls through to route BATCH rather than overcommitting
the staging pool"). The next controller pass then drained it straight back
into `_ready_for_d`, where the same gate held it at the head (law 2 never skips
the head) -- and a non-empty `_ready_for_d` closes the batch gate (C5/R-16), so
every SHORT arrival behind it waited, also the ones that fit, until D's running
decodes ended (up to --drain-deadline-s each). With the switch off the refused
request waits in the queue and the next SHORT that fits is served at once.

Measured on the desk against RC2 head 4aded781ab (a live seat, a pool reading
with available=1000, the refused SHORT at 3000, the next one at 500):
switch off -> the 500 gets its seat in 0.0 s; switch on -> still waiting at
the batch gate after 2.0 s.

Hermetic: a real Front, the real handle_generate for the route, the real
controller and admitter; no GPU, no HTTP.
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2.front import Front, Seat  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

X = 4096
REFUSED_TOKENS = 3000   # SHORT (<= X) but above the pool reading below
FITS_TOKENS = 500       # SHORT and inside it
AVAILABLE = 1000


class _Req:
    """The two things handle_generate reads from an aiohttp request."""

    def __init__(self, payload, path="/generate"):
        self._payload = payload
        self.path = path

    async def json(self):
        return self._payload


def _text_payload(tokens):
    # the front prices a prompt at CHARS_PER_TOKEN chars per token (no tokenizer)
    return {"text": "x" * int(tokens * front_mod.CHARS_PER_TOKEN),
            "sampling_params": {"max_new_tokens": 8}}


def _front(awake="D", d_short_drain_tokens=0):
    f = Front("http://p", "http://d", awake, "rc2rev", "", 0, 0, {}, 45.0,
              tp_prefill_max_tokens=X, d_short_drain_tokens=d_short_drain_tokens)
    f.stops = []
    f.do_stop = lambda name, detail: f.stops.append((name, detail))
    return f


async def _arm_the_budget(f):
    """One decode running on D (a live seat, so the #915 gate is armed) and
    D's pool reading with room for AVAILABLE rows."""
    await f._d_seat.acquire()
    seat = Seat(f, "running", "short", tokens=2000)
    f.groups["D"].outstanding["running"] = time.time()

    async def reading():
        return {"t": time.time(), "available": AVAILABLE, "limit": 27466, "occupied": 0}

    f._d_pool_reading = reading
    return seat


async def _route(f, tokens):
    """Run the REAL handle_generate until the request is queued (or answered);
    return (queued Pending or None, the handler task -- still running)."""
    n0 = len(f.queue)
    task = asyncio.ensure_future(f.handle_generate(_Req(_text_payload(tokens))))
    for _ in range(500):
        await asyncio.sleep(0)
        if len(f.queue) > n0 or task.done():
            break
    return (f.queue[-1] if len(f.queue) > n0 else None), task


async def _drop(*tasks):
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


# ------------------------------------------------------------------ the route
def test_a_short_that_ds_budget_refused_is_queued_but_not_drain_eligible():
    async def body():
        f = _front(awake="D", d_short_drain_tokens=4096)
        await _arm_the_budget(f)
        p, task = await _route(f, REFUSED_TOKENS)
        await _drop(task)
        assert p is not None, "the #915 refusal must fall through to route BATCH (FIX 4a)"
        assert p.est_uncached <= X, "it IS a SHORT by its own verdict"
        assert p.d_eligible is False, (
            "D just refused it on its own budget: it is queued because of D, not the phase, "
            "so the SHORT drain must not hand it back to D's admission line")

    asyncio.run(body())


def test_a_short_queued_by_the_phase_stays_drain_eligible():
    async def body():
        f = _front(awake="P", d_short_drain_tokens=4096)
        p, task = await _route(f, REFUSED_TOKENS)
        await _drop(task)
        assert p is not None and p.d_eligible is True, (
            "a SHORT that arrived while P was awake is exactly what the drain is for")

    asyncio.run(body())


def test_the_refusal_is_named_for_the_budget_only():
    async def body():
        f = _front(awake="D")
        await _arm_the_budget(f)
        why = []
        assert await f._acquire_short_seat("big", REFUSED_TOKENS, why) is None
        assert why == ["d_budget"]
        why = []
        seat = await f._acquire_short_seat("small", FITS_TOKENS, why)
        assert seat is not None and why == []
        seat.release("test")
        g = _front(awake="P")
        why = []
        assert await g._acquire_short_seat("phase", FITS_TOKENS, why) is None
        assert why == [], "a phase fall-through is not D's refusal"
        h = _front(awake="D")
        h.drain_deadline_s = 0.05
        h._batch_gate.clear()
        why = []
        assert await h._acquire_short_seat("gate", FITS_TOKENS, why) is None
        assert why == [], "a held batch gate is not D's budget"
        assert await f._acquire_short_seat("old-callers", REFUSED_TOKENS) is None, (
            "the reason list is optional: the carrier path calls without it")

    asyncio.run(body())


# ------------------------------------------------------------- the controller
async def _after_a_refusal(d_short_drain_tokens):
    f = _front(awake="D", d_short_drain_tokens=d_short_drain_tokens)
    running = await _arm_the_budget(f)
    p, handler = await _route(f, REFUSED_TOKENS)
    assert p is not None
    ctl = asyncio.ensure_future(f.controller())
    adm = asyncio.ensure_future(f.d_admitter())
    await asyncio.sleep(0.6)   # three controller ticks, a dozen admitter polls
    state = {"queue": [q.rid for q in f.queue],
             "ready_for_d": [q.rid for q in f._ready_for_d],
             "batch_gate_open": f._batch_gate.is_set()}
    try:
        seat = await asyncio.wait_for(f._acquire_short_seat("fits", FITS_TOKENS), 0.5)
        state["next_short"] = "seat" if seat is not None else "batch"
        if seat is not None:
            seat.release("test")
    except asyncio.TimeoutError:
        state["next_short"] = "waiting at the batch gate"
    running.release("test")
    await _drop(ctl, adm, handler)
    return p.rid, state


def test_switch_on_keeps_the_refused_short_off_ds_line_and_the_gate_open():
    async def body():
        rid, s = await _after_a_refusal(4096)
        assert s["ready_for_d"] == [], (
            f"the drain handed the budget-refused {rid} back to D's admission line: {s}")
        assert s["queue"] == [rid]
        assert s["batch_gate_open"] is True, "a held head in _ready_for_d closes the gate"
        assert s["next_short"] == "seat", (
            f"a SHORT that fits must not wait behind the refused one: {s}")

    asyncio.run(body())


def test_switch_off_is_todays_path():
    async def body():
        rid, s = await _after_a_refusal(0)
        assert s == {"queue": [rid], "ready_for_d": [], "batch_gate_open": True,
                     "next_short": "seat"}, s

    asyncio.run(body())
