"""ARRIVAL-SEAT: every queue has an owner (NF-Operator 30.09., y4y 17:05:40Z).

Befund (analysis seat): two SHORT requests (weg2-45-64/65, ~775 tokens each)
arrived during the flip P->D (17:05:48). ``self.awake != "D"`` -> queued BATCH
for P. Then FLIP-ECONOMICS held 300 s ("1551 < 4096, fairness=False"):
ARRIVAL-SEAT's step overwrites wait_fired/fairness_fired and its candidates are
only the entries that need P -- a SHORT is none; ``_d_short_drain`` is off on
NF (default 0). A pure SHORT queue had no owner; both ran into the 300 s
timeout with an anchor stream served beside them.

Fix (ARRIVAL-SEAT on):
1. a SHORT arriving during a flip TO D is queued for D (the flip's target, not
   only ``awake``);
2. a queued d_eligible SHORT goes to D's admission line like an arrival
   (verdict d_prefill) in ``_arrival_seat_step``;
3. net: the wait bound fires for EVERY queued entry, non-candidates too.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import arrival_seat_rule as _asr  # noqa: E402
from sglang.srt.weg2.front import Front, Pending  # noqa: E402

X = 4096


class _Req:
    def __init__(self, path, payload):
        self.path = path
        self._payload = payload
        self._d = {}

    async def json(self):
        return self._payload

    def __setitem__(self, k, v):
        self._d[k] = v

    def get(self, k, default=None):
        return self._d.get(k, default)


def _front(awake):
    f = Front("http://p", "http://d", awake, "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    f.state = "serving"
    f.admit_d = True
    return f


def _asr_on():
    return envs.SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE.override(True)


def test_y4y_a_short_arriving_during_the_flip_to_d_is_queued_for_d(caplog):
    """(1) weg2-45-64: ~775 tokens, the flip P->D runs -> D's admission line."""
    caplog.set_level(logging.INFO)

    async def go():
        f = _front("P")
        f.state = "flipping"
        f._flip_dst = "D"

        async def fake_leg2(*a, **k):
            return "served"
        f.leg2 = fake_leg2
        task = asyncio.ensure_future(f.handle_generate(_Req("/generate", {"text": "x" * 2300})))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except asyncio.TimeoutError:
            pass
        queued = list(f.queue)
        ready = list(f._ready_for_d)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return f, queued, ready

    with _asr_on():
        f, queued, ready = asyncio.run(go())
    assert queued == [], "not BATCH for P: nothing owned that queue (y4y)"
    assert len(ready) == 1 and ready[0].d_direct and not ready[0].leg1_done
    assert f.counters["arrival_seat_short_in_flip"] == 1
    assert any("SHORT-IN-FLIP" in m and "verdict=d_prefill" in m for m in caplog.messages)


def test_the_flip_to_p_keeps_the_old_path():
    """A SHORT arriving while D flips to P still goes to P's queue (P is next)."""
    async def go():
        f = _front("D")
        f.state = "flipping"
        f._flip_dst = "P"

        async def fake_leg2(*a, **k):
            return "served"
        f.leg2 = fake_leg2
        task = asyncio.ensure_future(f.handle_generate(_Req("/generate", {"text": "x" * 2300})))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except asyncio.TimeoutError:
            pass
        out = (list(f.queue), list(f._ready_for_d))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return out

    with _asr_on():
        queued, ready = asyncio.run(go())
    assert len(queued) == 1 and ready == []


def _pending(rid, age_s, d_eligible=True, uncached=775):
    loop = asyncio.get_event_loop()
    return Pending(rid=rid, path="/generate", payload={}, text="x", t_arrive=time.time() - age_s,
                   fut=loop.create_future(), est_prompt=uncached, est_uncached=uncached,
                   d_eligible=d_eligible)


def test_y4y_then_an_empty_d_takes_the_queued_shorts_as_d_prefill(caplog):
    """(2) the two SHORTs sit in the queue, D is awake and EMPTY: the step hands
    them to D's admission line (verdict d_prefill) -- no FLIP-ECONOMICS hold,
    no flip to P for 1551 tokens."""
    caplog.set_level(logging.INFO)

    async def go():
        f = _front("D")
        f.t_awake = time.time() - 2.0
        a, b = _pending("weg2-45-64", 1.0), _pending("weg2-45-65", 0.9)
        f.queue.extend([a, b])
        got = await f._arrival_seat_step(f.groups["D"], time.time())
        return f, got, a, b

    with _asr_on():
        f, got, a, b = asyncio.run(go())
    assert got == (False, False, None)
    assert list(f.queue) == [] and list(f._ready_for_d) == [a, b] and a.d_direct and b.d_direct
    assert f.counters["arrival_seat_queue_to_d"] == 2
    assert sum("QUEUED-SHORT" in m and "verdict=d_prefill" in m for m in caplog.messages) == 2


def test_a_long_head_is_still_a_candidate_and_the_short_behind_goes_to_d():
    async def go():
        f = _front("D")
        f.t_awake = time.time() - 2.0
        long_ = _pending("L", 1.0, d_eligible=False, uncached=20000)
        short = _pending("S", 0.5)
        f.queue.extend([long_, short])
        with mock.patch.object(Front, "_arrival_seat_taken", lambda self: (0, 4)), \
                mock.patch.object(Front, "_arrival_seat_kv",
                                  lambda self, *a, **k: _async((False, "kv", 0))), \
                mock.patch.object(Front, "_arrival_seat_kv_displace", lambda self, *a, **k: _async(None)), \
                mock.patch.object(_asr, "age_plan_enabled", lambda env=None: False):
            await f._arrival_seat_step(f.groups["D"], time.time())
        return f, long_, short

    with _asr_on():
        f, long_, short = asyncio.run(go())
    assert list(f.queue) == [long_] and list(f._ready_for_d) == [short]


async def _async(v):
    return v


def test_net_the_wait_bound_fires_for_a_non_candidate(caplog):
    """(3) a queued entry no rule takes (here: a SHORT D refused at arrival,
    d_eligible False) past the bound fires it -> the flip to P takes it."""
    caplog.set_level(logging.WARNING)

    async def go():
        f = _front("D")
        f.t_awake = time.time() - 200.0
        f.queue.append(_pending("weg2-45-66", 100.0, d_eligible=False))
        with mock.patch.object(_asr, "age_plan_enabled", lambda env=None: True):
            got = await f._arrival_seat_step(f.groups["D"], time.time())
            got2 = await f._arrival_seat_step(f.groups["D"], time.time())
        return f, got, got2

    with _asr_on():
        f, got, got2 = asyncio.run(go())
    assert got == (True, True, None) and got2 == (True, True, None)
    assert len(f.queue) == 1
    lines = [m for m in caplog.messages if "WAIT-BOUND-ANY" in m]
    assert len(lines) == 1 and "weg2-45-66" in lines[0]


def test_below_the_bound_a_non_candidate_waits():
    async def go():
        f = _front("D")
        f.t_awake = time.time() - 200.0
        f.queue.append(_pending("w", 5.0, d_eligible=False))
        return await f._arrival_seat_step(f.groups["D"], time.time())

    with _asr_on():
        assert asyncio.run(go()) == (False, False, None)
