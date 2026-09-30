"""NF-STAU-KV Wurzel 3 im 27B-Slot (NF 98218cc745, y3v weg2-28-58: 66,2 s unsichtbar).

Ein kept SHORT lief mit skip_leg1 durch den P-Drain (leg1_done=True, kein Leg 1). Nach D's
x_refusal wird er "durch P (Leg 1, dann Leg 2)" neu eingereiht -- mit altem leg1_done=True sagte
needs_p() nein, und kein Flip kam fuer ihn. Der Requeue setzt leg1_done jetzt zurueck.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as F  # noqa: E402

N = 86598
W50 = (b'{"error": {"message": "W50 Weg2TpPrefillExceeded: this group may prefill at '
       b'most 3246 uncached tokens itself (--tp-prefill-max-tokens); this request\'s '
       b'extent after prefix matching is 86598. Refused by name so the caller re-routes '
       b'it through the prefill group -- never prefilled here silently."}}')


def test_a_rerouted_kept_short_runs_leg1_again():
    from sglang.srt.weg2.front_tokens import TokenSpans

    async def run():
        f = F.Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                    carrier_max_tokens=524288, tp_prefill_max_tokens=3246)
        f.epoch = 30
        ids = np.arange(N, dtype=np.int32)
        f.x_exact = True
        f.ftok = SimpleNamespace(ids_for=lambda text: ids)
        f.tspans = TokenSpans(agent_span=False)
        loop = asyncio.get_event_loop()
        p = F.Pending("weg2-28-58", "/v1/messages", {}, "t" * 10, 0.0, loop.create_future(),
                      est_prompt=N, est_uncached=454, span_known=True, store_span_est=86144,
                      skip_leg1=True, short_kept=True, price_epoch=29)
        p.leg1_done = True    # P-Drain: skip_leg1 -> leg1_done, kein Leg 1 lief
        task = asyncio.create_task(f._requeue_after_x_refusal(
            SimpleNamespace(path="/v1/messages"), p.rid, {}, "t" * 10, False, p, None, W50))
        for _ in range(200):
            if task.done() or f.queue:
                break
            await asyncio.sleep(0.005)
        assert list(f.queue) == [p]
        assert p.leg1_done is False and p.skip_leg1 is False
        task.cancel()

    asyncio.run(run())
