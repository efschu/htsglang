"""Q-712 DECODE-COLLECT counts the UNCACHED rest of a SHORT also without ARRIVAL-SEAT (INT8 y8vb 03.10.).

METAL (boot_weg2_dkr27browauthoritybar1fs10031930_76f5bbde19_1003_193056.front.log, desk 870 F3):
``DECODE-COLLECT dcheck n=6 tokens=137934 ... over X`` -- 137934 is the sum of the six est_prompt
(7743+36262+25+36423+35690+21791), the uncached rests sum to 12413. ARRIVAL-SEAT is off in that
profile, so ``_acquire_short_seat`` was called WITHOUT ``uncached=`` and
``_decode_collect_short(rid, est_tokens)`` booked the whole prompt: "a set <= X goes to D" never
fired for a SHORT with a large cached prefix (all 10 releases of y8vb route=P, 27 of 30 on y8va).
"""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import arrival_seat_rule as _asr  # noqa: E402
from sglang.srt.weg2.front import Front  # noqa: E402

X = 4096
PREFIX = "a" * 110016   # chars/3 -> 36673 tokens; D measured 36672 of it
TEXT = PREFIX + "b" * 621  # +207 tokens


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


def _run(arrival_seat):
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    f.state = "serving"
    f.admit_d = True
    f.spans.record_presence(PREFIX, 36672)
    seen = {}

    async def fake_seat(rid, est_tokens=0, refused=None, max_tokens=None, uncached=None):
        seen.update(est_tokens=est_tokens, uncached=uncached)
        return None  # falls through to BATCH: the test only reads the arguments

    f._acquire_short_seat = fake_seat

    async def go():
        async def fake_leg2(*a, **k):
            return "served"
        f.leg2 = fake_leg2
        task = asyncio.ensure_future(f.handle_generate(_Req("/generate", {"text": TEXT})))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except asyncio.TimeoutError:
            pass
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    orig = _asr.enabled
    _asr.enabled = lambda *a, **k: arrival_seat
    try:
        asyncio.run(go())
    finally:
        _asr.enabled = orig
    return seen


def test_without_arrival_seat_the_collect_window_gets_the_uncached_rest():
    """RED on 1de15e82ca: uncached stayed None -> est_prompt (36673) was booked."""
    seen = _run(arrival_seat=False)
    assert seen, "the SHORT reached the D seat path"
    assert seen["uncached"] is not None and seen["uncached"] <= 208
    assert seen["est_tokens"] > 30000, "the whole prompt is still the seat's KV need"


def test_with_arrival_seat_the_call_is_unchanged():
    seen = _run(arrival_seat=True)
    assert seen and seen["uncached"] is not None and seen["uncached"] <= 208
