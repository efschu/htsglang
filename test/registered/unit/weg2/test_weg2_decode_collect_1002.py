"""DECODE-COLLECT (user rule 02.10. ~19:07Z, 27B and NF).

"wenn prefill requests reinkommen und noch decoded wird, dann wird erstmal noch
15 sekunden weiterdecoded und prefill requests gesammelt, erst dann ... je nach
anzahl in P oder D" -- and "auf den alten weg ... indem man die zeit auf 0
stellt". Base d019aa8e1e: an arrival on a decoding D was served at once
(PBOUND-FLIP-NOW / ARRIVAL-SEAT flip_now, the SHORT took its D seat at once).
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import X, _front, _pending  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401


def _on(monkeypatch, window="15"):
    monkeypatch.setenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "1")
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_D_CHECK_S", raising=False)  # default 7.5
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_AGE_PLAN", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_PBOUND_FLIP_NOW", raising=False)
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", window)


def test_the_code_default_is_15_seconds(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_WINDOW_S", raising=False)
    assert float(envs.SGLANG_WEG2_DECODE_COLLECT_WINDOW_S.get()) == 15.0


def test_a_p_bound_arrival_on_a_decoding_d_is_held_not_flipped(monkeypatch):
    """Base: PBOUND-FLIP-NOW flips at once -> (True, True, p)."""
    _on(monkeypatch)
    f = _front(running=["a", "b"], n=6, kv={"available": 400000, "evictable": 0})
    now = time.time()
    f.queue = [_pending("big", 16448, now - 1.0)]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (False, False, None)
    assert f.counters["decode_collect_hold"] == 1
    assert f.counters["arrival_seat_flip_now"] == 0


def test_after_the_window_the_set_over_x_flips_to_p(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a"], n=6, kv={"available": 400000, "evictable": 0})
    now = time.time()
    f.t_awake = now - 60.0
    a, b = _pending("s1", 3000, now - 16.0), _pending("s2", 3000, now - 10.0)
    f.queue = [b, a]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, a)
    assert f.counters["decode_collect_release_P"] == 1


def test_at_d_check_a_small_set_goes_to_d(monkeypatch):
    """User ~19:12Z: at 7.5 s a set <= X is prefilled on D at once."""
    _on(monkeypatch)
    f = _front(running=["a"], n=6, kv={"available": 400000, "evictable": 0})
    now = time.time()
    f.t_awake = now - 60.0
    s = _pending("s1", 800, now - 8.0)
    s.d_eligible = True
    f.queue = [s]
    wait_fired, _fair, immediate = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert (wait_fired, immediate) == (False, None), "no flip for a set <= X"
    assert f.counters["decode_collect_release_D"] == 1
    assert f._ready_for_d == [s], "the SHORT goes to D's admission line"


def test_at_d_check_a_set_over_x_collects_on(monkeypatch):
    _on(monkeypatch)
    f = _front(running=["a"], n=6, kv={"available": 400000, "evictable": 0})
    now = time.time()
    f.t_awake = now - 60.0
    f.queue = [_pending("s1", 3000, now - 8.0), _pending("s2", 3000, now - 5.0)]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (False, False, None)
    assert f.counters["decode_collect_dcheck_hold"] == 1
    assert f.counters["decode_collect_release_D"] == f.counters["decode_collect_release_P"] == 0


def test_without_d_check_the_window_routes_a_small_set_to_d(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("SGLANG_WEG2_DECODE_COLLECT_D_CHECK_S", "0")
    f = _front(running=["a"], n=6, kv={"available": 400000, "evictable": 0})
    now = time.time()
    f.t_awake = now - 60.0
    s = _pending("s1", 800, now - 8.0)
    s.d_eligible = True
    f.queue = [s]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (False, False, None), "held to 15 s"
    f._dc_state["t_open"] = now - 16.0
    asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert f.counters["decode_collect_release_D"] == 1


def test_the_d_check_default_is_7_5_seconds(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_DECODE_COLLECT_D_CHECK_S", raising=False)
    assert float(envs.SGLANG_WEG2_DECODE_COLLECT_D_CHECK_S.get()) == 7.5


def test_an_idle_d_serves_at_once(monkeypatch):
    _on(monkeypatch)
    f = _front(running=[], n=6, kv={"available": 400000, "evictable": 0})
    p = _pending("big", 16448, time.time())
    f.queue = [p]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], time.time())) == (True, True, p)
    assert f.counters["decode_collect_hold"] == 0


def test_zero_is_the_old_path(monkeypatch):
    _on(monkeypatch, window="0")
    f = _front(running=["a"], n=6, kv={"available": 400000, "evictable": 0})
    p = _pending("big", 16448, time.time())
    f.queue = [p]
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], time.time())) == (True, True, p)
    assert "_dc_state" not in f.__dict__


def test_a_short_waits_the_window_then_takes_its_d_seat(monkeypatch):
    """Base: the SHORT took its seat at once."""
    _on(monkeypatch, window="0.3")
    f = _front(running=["a"], n=6, kv={"available": 400000, "evictable": 0})
    t0 = time.time()
    seat = asyncio.run(f._acquire_short_seat("short", 500, uncached=500))
    assert seat is not None and time.time() - t0 >= 0.25
    assert f.counters["decode_collect_release_D"] == 1


def test_a_short_beside_a_p_bound_arrival_rides_the_flip(monkeypatch):
    _on(monkeypatch, window="0.3")
    f = _front(running=["a"], n=6, kv={"available": 400000, "evictable": 0})
    f.queue = [_pending("big", 16448, time.time())]
    assert asyncio.run(f._acquire_short_seat("short", 500, uncached=500)) is None
    assert f.counters["decode_collect_release_P"] == 1
