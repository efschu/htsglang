"""#246b (30.09., NF y4c ...dauer09300427, 768cb0b257, front 04:39:57-04:42:08):
weg2-22-37 (4447 tokens) and weg2-22-38 (4191) arrived while D decoded
weg2-0-5 alone (5 of 6 seats free)::

    X-SOLO rid=weg2-22-37 uncached=4447 X_live=4964 verdict=p reason=d_outstanding=2
    WEG2-ROUTE rid=weg2-22-37 LONG -> P leg 1 (uncached=4447 > X=4096 ...)
    LATE-BATCH rid=weg2-22-37 deferred_to_epoch=23
    ... ARRIVAL-SEAT YOUNGEST-PARK rid=weg2-0-5 oldest_wait_s=60.0 / 120.0
    DP-WAIT rid=weg2-22-37 wait_s=130.9 hold_s=127.9 hold_by=d-work

The front routed them to P on the X-SOLO band floor (X_busy 4096, D busy),
but the ARRIVAL-SEAT step asked ``needs_p()`` against the LIVE X 4964: no
candidate, no verdict, no flip -- the bound (c) parked the only decode twice
for nothing, and the flip came only when weg2-0-5 ended (04:42:05). User
rules violated: arrival with a free seat flips now (1), the collect window
(2), the 60 s bound (3).

Now a queued request needs P above the X it was ROUTED on."""
from __future__ import annotations

import asyncio
import inspect
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import _front, _on, _pending  # noqa: E402

from sglang.srt.weg2 import arrival_seat_rule as asr  # noqa: E402
from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2 import phase_policy  # noqa: E402

X_LIVE = 4964
X_BUSY = 4096


def _band(rid, uncached, t, x_routed=X_BUSY):
    p = _pending(rid, uncached, t, payload={"max_tokens": 256})
    p.x_routed = x_routed
    return p


def test_needs_p_reads_the_routed_x():
    assert phase_policy.needs_p(4447, X_LIVE, x_routed=X_BUSY)          # base: False
    assert not phase_policy.needs_p(4447, X_LIVE)                      # not routed long: as before
    assert not phase_policy.needs_p(4000, X_LIVE, x_routed=X_BUSY)
    assert not phase_policy.needs_p(4447, X_LIVE, x_routed=X_BUSY, leg1_done=True)
    assert phase_policy.needs_p(6000, X_LIVE, x_routed=0)


def test_y4c_the_band_arrival_flips_now_with_free_seats(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["weg2-0-5"], n=6, kv={"available": 300000, "evictable": 0},
               admit_t={"weg2-0-5": 1.0})
    f.tp_prefill_max_tokens = X_LIVE
    now = time.time()
    a, b = _band("weg2-22-37", 4447, now - 0.1), _band("weg2-22-38", 4191, now - 0.1)
    f.queue = [a, b]
    wait_fired, fairness_fired, immediate = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert immediate is a                          # base: None -- 128 s until weg2-0-5 ended
    assert (wait_fired, fairness_fired) == (True, True)
    assert f.counters["arrival_seat_flip_now"] == 1
    assert f.rpc_calls == []                       # no youngest park: a flip, at once


def test_the_collect_window_trigger_sees_it_too():
    now = time.time()
    q = [_band("weg2-22-37", 4447, now)]
    assert phase_policy.immediate_park_trigger(q, X_LIVE) is q[0]      # base: None


def test_a_request_not_routed_long_stays_d_work(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a"], n=6, kv={"available": 300000, "evictable": 0})
    f.tp_prefill_max_tokens = X_LIVE
    p = _band("short-behind", 4447, time.time(), x_routed=0)
    f.queue = [p]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], time.time()))[2] is None


def test_the_route_stamps_the_routed_x_on_the_pending():
    src = inspect.getsource(front_mod.Front.handle_generate)
    assert "x_routed=(int(x_route) if route == \"long\"" in src
    assert "x_routed" in front_mod.Pending.__dataclass_fields__
    assert asr.verdict(True, True, 4447, X_BUSY) == asr.FLIP_NOW
