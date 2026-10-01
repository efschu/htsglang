"""D->P flip time in the USER's definition (FLIPZEIT-VERLAUF-0929.md, Folgepunkt 30.09.):
"Decode-Ende -> erstes Prefill" -- the last D decode round (the park RPC's send, else D's last
served leg 2), no earlier than the oldest waiter's arrival -> the start of P's first prefill.

Before: ``flip_first_work`` measured D->P as ``WEG2-FLIP begin`` -> the first leg-1 DISPATCH.
Missing: the park RPC before the begin (27B z30j median 0.66 s, p90 1.15, max 3.31 s; NF z30w
0.40 s) and P's queue until its prefill starts. Now one ``flip_user_time`` event per D->P flip
with the span and its parts (park_rpc, pre_begin, legs, first_chunk).
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2 import front_state_ipc as fsi  # noqa: E402
from sglang.srt.weg2 import host_ledger  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_dashipc_dp", os.path.join(os.path.dirname(__file__), "test_weg2_dashboard_ipc_0929.py"))
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)


# ---------------------------------------------------------------- the clock

def test_the_park_rpc_starts_the_span_and_p_prefill_ends_it():
    c = fsi.DpFlipClock()
    c.note_d_served(90.0)
    c.note_park(5, 100.0, 660.0)                 # 27B z30j median park RPC
    c.begin(5, 100.7, oldest_waiter_ts=95.0)     # the waiter arrived before the park: the park counts
    c.done(103.2)
    ev = c.first_prefill("weg2-6-1", t_dispatch=103.25, t_end=106.0, p_prefill_s=2.5)
    assert (ev["epoch"], ev["dir"], ev["start_source"], ev["prefill_start_source"]) == \
        (6, "D>P", "park_rpc_sent", "leg1_end_minus_p_prefill_s")
    assert ev["flip_user_ms"] == 3500                        # 100.0 -> 103.5
    assert ev["parts"] == {"park_rpc_ms": 660, "pre_begin_ms": 700, "legs_ms": 2500, "first_chunk_ms": 300}
    assert c.first_prefill("weg2-6-2", 107.0, 108.0, 0.5) is None   # one per flip


def test_without_a_park_the_last_served_leg2_is_the_decode_end():
    c = fsi.DpFlipClock()
    c.note_d_served(50.0)
    c.begin(3, 60.0, oldest_waiter_ts=58.0)
    c.done(62.0)
    ev = c.first_prefill("r", 62.1, 64.0, None)              # P's body carried no prefill time
    assert (ev["start_source"], ev["prefill_start_source"], ev["flip_user_ms"]) == \
        ("oldest_waiter_arrival", "leg1_dispatch", 4100)
    assert ev["parts"]["park_rpc_ms"] is None and ev["idle_flip"] is False
    c.begin(5, 70.0, oldest_waiter_ts=40.0)                  # the waiter came first: D's served leg 2 starts it
    c.done(71.0)
    ev = c.first_prefill("r2", 71.1, 72.0, None)
    assert (ev["start_source"], ev["flip_user_ms"]) == ("last_d_served", 21100)


def test_an_idle_flip_starts_at_the_first_dispatch_not_at_the_decode_end():
    # z30y14 epoch 4->5 (01.10. 02:44): the idle-layout swap began with outstanding=0 queue=0
    # at 02:44:28.9, done 02:44:30.8; the first leg 1 came 26 s later (02:44:56.3, wall 2.56 s).
    # The old clock read decode end -> P prefill start = 37.4 s, and M2 alarmed FLIP-SLOW.
    c = fsi.DpFlipClock()
    c.note_d_served(1000.0)
    c.begin(4, 1010.9, oldest_waiter_ts=None)                # nobody waits for this flip
    c.done(1012.8)
    ev = c.first_prefill("weg2-5-5", t_dispatch=1038.3, t_end=1040.8, p_prefill_s=2.4)
    assert ev["idle_flip"] is True
    assert ev["start_source"] == "first_dispatch_after_idle_flip"
    assert ev["flip_user_ms"] == 100                          # 1038.3 -> 1038.4: P's own queue only
    assert ev["parts"]["pre_begin_ms"] is None                # the user came after the flip


def test_the_span_starts_no_earlier_than_the_oldest_waiter():
    c = fsi.DpFlipClock()
    c.note_d_served(10.0)                                     # D idle since 10 s
    c.begin(2, 30.0, oldest_waiter_ts=28.0)                   # the request came at 28 s
    c.done(32.0)
    ev = c.first_prefill("r", 32.0, 33.0, 0.5)
    assert (ev["start_source"], ev["flip_user_ms"]) == ("oldest_waiter_arrival", 4500)


def test_a_park_of_another_phase_does_not_count():
    c = fsi.DpFlipClock()
    c.note_park(1, 5.0, 100.0)
    c.begin(4, 40.0, None)
    assert c._armed["start_source"] == "flip_begin"


def test_no_flip_no_event():
    assert fsi.DpFlipClock().first_prefill("r", 1.0, 2.0, 0.5) is None


# ---------------------------------------------------------------- the front

class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def test_the_front_publishes_one_flip_user_time_after_a_dp_flip():
    sd = _d._boot(tempfile.mkdtemp(prefix="flipzeit-dp-"))
    f = _d._front()
    f.rpc = _d._rpc
    f.p_leg1_stall_s = 0.0
    body = json.dumps({"usage": {"prompt_tokens": 5000, "completion_tokens": 1,
                                 "prompt_tokens_details": {"cached_tokens": 0}},
                       "sglext": {"weg2_prefill_s": 0.25}}).encode()
    f.session = SimpleNamespace(post=lambda url, json=None: _Resp(200, body))
    p = front_mod.Pending(rid="weg2-1-1", path="/v1/chat/completions", payload={"messages": []}, text="x",
                          t_arrive=time.time(), fut=None)

    async def run():
        f._ipc_dp_clock().note_park(f.epoch, time.time() - 0.4, 400.0)   # the park RPC before the flip
        await f.flip("D", "P")
        await f.leg1(p)
        await asyncio.sleep(0.2)

    with mock.patch.dict(os.environ, {"WEG2_STATE_DIR": sd}), \
            mock.patch.object(host_ledger, "read_cgroup_pressure", return_value=dict(_d.Z30U)), \
            mock.patch.object(host_ledger, "read_cgroup", return_value={"max": 84 * _d.GIB}):
        asyncio.run(run())
        deadline = time.time() + 3.0
        while time.time() < deadline and not _d._of(sd, "flip_user_time"):
            time.sleep(0.02)
    ev = _d._of(sd, "flip_user_time")
    assert len(ev) == 1
    w = ev[0]["data"]
    assert (w["dir"], w["start_source"], w["prefill_start_source"], w["rid"]) == \
        ("D>P", "park_rpc_sent", "leg1_end_minus_p_prefill_s", "weg2-1-1")
    assert w["parts"]["park_rpc_ms"] == 400 and w["flip_user_ms"] >= 400
    assert w["parts"]["legs_ms"] is not None and w["parts"]["first_chunk_ms"] is not None


def test_the_park_rpc_and_the_served_leg2_feed_the_clock():
    src = inspect.getsource(front_mod.Front._wait_bound_park)
    assert "self._ipc_dp_clock().note_park(self.epoch, t_park" in src
    assert "note_d_served" in inspect.getsource(front_mod.Front._ipc_note_served_d)
