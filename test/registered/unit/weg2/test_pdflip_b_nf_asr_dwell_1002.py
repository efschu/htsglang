# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-B on NF (02.10.2026): the ARRIVAL-SEAT flip_now passes K7's derived
min-dwell before it parks D.

27B's park trigger (_immediate_park_due) checks the dwell first, so PDFLIP-B
labels an admission closed by the phase's own park ``overridden_by=park``.
NF decides D->P in the ARRIVAL-SEAT step, which parked at any awake time; the
override then let the flip through after the 2 s fairness floor -- the 27B
N5d shape (overridden_by=fairness after 6.6 s awake). Red on b034139189 +
PDFLIP-A/B (d7fc96d994), green with the K7-DWELL hold.
"""

from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_weg2_arrival_seat_rule_0929 import _front, _on, _pending  # noqa: E402

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


def _asr(monkeypatch, awake_s):
    _on(monkeypatch)
    f = _front(running=["a"], n=6, kv={"available": 400000, "evictable": 0}, admit_t={"a": 1.0})
    now = time.time()
    f.t_awake = now - awake_s
    f.min_dwell_ms = None
    # warm K7: the boot's first flip is excluded, the median of the later D->P ones
    f.flip_log = [{"sleep": "D", "wake": "P", "flip_ms": 24600.0},
                  {"sleep": "D", "wake": "P", "flip_ms": 2300.0}]
    big = _pending("long", 16448, now - 0.1)
    f.queue = [big]
    return f, big, now


def test_red_flip_now_waits_for_the_k7_dwell_before_the_park(monkeypatch):
    f, big, now = _asr(monkeypatch, awake_s=1.0)
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (False, False, None)
    assert f.counters["arrival_seat_k7_dwell_hold"] == 1
    assert f.counters["arrival_seat_flip_now"] == 0 and f.rpc_calls == []


def test_past_the_dwell_it_flips_now(monkeypatch):
    f, big, now = _asr(monkeypatch, awake_s=2.4)
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, big)
    assert f.counters["arrival_seat_k7_dwell_hold"] == 0


def test_switch_off_is_the_old_trigger(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_X_BAND_FOLLOWS_PRICE", "0")
    f, big, now = _asr(monkeypatch, awake_s=1.0)
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, big)


def test_the_first_flip_of_a_boot_has_no_dwell(monkeypatch):
    f, big, now = _asr(monkeypatch, awake_s=1.0)
    f.flip_log = []
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, big)
