"""H91c3: the three rests the H91c2 report named at the park/wait stages of
the NF standard form H91 (desk play-through, no GPU, no model).

* H91c3-1 (front): ``_wait_bound_park`` stamped ``_d_parked`` only AFTER the
  park RPC answered. D stamps its park while serving the RPC and re-queues it
  after PARK_REQUEUE_S (30 s) on its own clock, so the front's 30-s clock ran
  late by the RPC's duration: for that long the front still counted as parked
  what D was running again (drain and W3 blind to it). The stamp is now taken
  before the RPC and written only for the rids D confirms.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from aiohttp import web  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_phase_policy_h91c import FakeGroup, Harness, _until  # noqa: E402


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
