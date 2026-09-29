# SPDX-License-Identifier: Apache-2.0
"""W3-STOP (kvs2 boot 09291223, 2026-09-29): a STOP answers the requests in
flight, not only the queued ones.

MEASURED (front log boot_weg2_dkrnfh91dprsavisadoptstcutvsyncodx2bswre2cut
z30x2kvdemandbar1dauer09291223_424346f693_0929_122349.front.log:677-836):
``WEG2 STOP W3 Weg2DrainWitnessDisagreement`` at 12:32:58 failed the queue and
wrote the A5 stop request -- but rid weg2-0-2 (the host acceptance's needle
probe, sent 12:29:34) was on its leg 2 to a D the stalled flip never woke. Its
handler waited until the container was stopped by hand at 12:39:32 (503, 598 s),
the host script sat in that probe (timeout 1800 s), never reached the hold
loop that reads ``stop_request.json``, and the boot stood in ``flipping`` for
6.5 minutes after its STOP.

Guarded here: a handler in flight when the STOP falls answers 503 with the
STOP's name at once; a stream that already sent its status line is closed.
Hermetic: no GPU, no HTTP.
"""
from __future__ import annotations

import asyncio
import json
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

W3 = "W3 Weg2DrainWitnessDisagreement"
W3_DETAIL = ("front drained, rank NOT idle: front ledger [] vs group(P) flush_cache "
             "(reduced over every rank, #1268) -> 'Flush cache failed.\\n'")


def _front():
    f = front_mod.Front(
        prefill="http://p", decode="http://d", awake="P", tag="w3stop",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, flip_min_work_tokens=1,
    )

    async def rpc(g, path, body, timeout):
        return 200, "{}"

    f.rpc = rpc
    return f


def _handler(f):
    """The route's handler as ``main`` binds it (``Front.stop_guard`` around
    ``handle_generate``); a tree without the guard serves the bare handler --
    the base's behaviour, which the tests below show red."""
    guard = getattr(f, "stop_guard", None)
    return guard(f.handle_generate) if callable(guard) else f.handle_generate


class _Req(dict):
    """The handler's request as far as this path reads it: a mapping (aiohttp's
    Request is one) whose body never arrives -- the stand-in for a leg 2 on a
    D the stalled flip never woke."""

    def __init__(self, gate: asyncio.Event):
        super().__init__()
        self.gate = gate
        self.path = "/v1/chat/completions"

    async def json(self):
        await self.gate.wait()
        return {}


class TestStopAnswersInFlight(CustomTestCase):
    def setUp(self):
        os.environ.pop("WEG2_STATE_DIR", None)

    def test_kvs2_shape_a_request_on_a_leg_is_answered_503_by_the_stop(self):
        """RED on 424346f693: the handler keeps waiting after the STOP (the
        wait below times out). GREEN: 503 with the STOP's name, in time."""

        async def scenario():
            f = _front()
            gate = asyncio.Event()
            t = asyncio.ensure_future(_handler(f)(_Req(gate)))
            await asyncio.sleep(0.02)
            f.do_stop(W3, W3_DETAIL)
            done, _ = await asyncio.wait({t}, timeout=2.0)
            if t not in done:
                t.cancel()
                await asyncio.gather(t, return_exceptions=True)
                return None, f
            return t.result(), f

        resp, f = asyncio.run(scenario())
        self.assertIsNotNone(resp, "the in-flight request was not answered by the STOP")
        self.assertEqual(resp.status, 503)
        self.assertIn("W3", json.loads(resp.body)["error"])
        self.assertEqual(f.counters["stop_inflight_answered"], 1)
        self.assertEqual(f.state, "STOP")

    def test_a_started_stream_is_closed_not_answered_twice(self):
        async def scenario():
            f = _front()
            gate = asyncio.Event()
            req = _Req(gate)
            req["weg2_prepared"] = True     # the status line of a stream went out
            t = asyncio.ensure_future(_handler(f)(req))
            await asyncio.sleep(0.02)
            f.do_stop(W3, W3_DETAIL)
            done, _ = await asyncio.wait({t}, timeout=2.0)
            if t not in done:
                t.cancel()
                await asyncio.gather(t, return_exceptions=True)
            return t, done

        t, done = asyncio.run(scenario())
        self.assertIn(t, done)
        self.assertTrue(t.cancelled())

    def test_after_the_stop_a_new_request_is_refused_and_nothing_stays_registered(self):
        async def scenario():
            f = _front()
            gate = asyncio.Event()
            t = asyncio.ensure_future(_handler(f)(_Req(gate)))
            await asyncio.sleep(0.02)
            f.do_stop(W3, W3_DETAIL)
            done, _ = await asyncio.wait({t}, timeout=2.0)
            if t not in done:
                t.cancel()
                await asyncio.gather(t, return_exceptions=True)
            late = await _handler(f)(_Req(asyncio.Event()))
            return f, late

        f, late = asyncio.run(scenario())
        self.assertEqual(late.status, 503)
        self.assertEqual(len(f.__dict__.get("_generate_tasks", ())), 0)

    def test_a_stop_outside_the_loop_still_stops(self):
        f = _front()
        f.do_stop(W3, W3_DETAIL)
        self.assertEqual(f.state, "STOP")


if __name__ == "__main__":
    unittest.main()
