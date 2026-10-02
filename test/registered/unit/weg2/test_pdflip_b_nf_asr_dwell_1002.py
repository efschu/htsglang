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


def _asr(monkeypatch, awake_s, running=("a",)):
    _on(monkeypatch)
    f = _front(running=list(running), n=6, kv={"available": 400000, "evictable": 0},
               admit_t={r: 1.0 for r in running})
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


# ---- IDLE D NEVER HOLDS (user law 02.10.: Flipzeit = last token -> first token; a
# dwell hold while D sits idle is pure Vorlauf -- 27B D>P Vorlauf 1.6-2.2 s) ----------

def test_red_idle_d_never_holds_the_flip_for_the_dwell(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO)
    f, big, now = _asr(monkeypatch, awake_s=1.0, running=())
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, big)
    assert f.counters["arrival_seat_k7_dwell_hold"] == 0
    assert f.counters["arrival_seat_k7_dwell_skip_idle"] == 1
    assert any("K7-DWELL skip rid=long d_running=0" in r.getMessage()
               and "(idle D: the dwell would be pure Vorlauf)" in r.getMessage() for r in caplog.records)


def test_d_decoding_holds_until_the_min_dwell_and_names_d_running(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO)
    f, big, now = _asr(monkeypatch, awake_s=1.0, running=("a", "b"))
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (False, False, None)
    assert f.counters["arrival_seat_k7_dwell_hold"] == 1
    assert f.counters["arrival_seat_k7_dwell_skip_idle"] == 0
    assert any("K7-DWELL hold rid=long d_running=2 " in r.getMessage() for r in caplog.records)
    # the same D past the dwell: the flip goes
    f.t_awake = now - 2.4
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, big)


def test_a_parked_decode_is_no_running_decode(monkeypatch):
    f, big, now = _asr(monkeypatch, awake_s=1.0, running=("a",))
    f._d_parked = {"a": now}          # D no longer runs it: the ledger drops it
    assert asyncio.run(f._arrival_seat_step(f.groups["D"], now)) == (True, True, big)
    assert f.counters["arrival_seat_k7_dwell_skip_idle"] == 1


def test_the_resumed_min_dwell_never_holds_an_idle_d(monkeypatch):
    """The ARRIVAL-SEAT MIN-DWELL (resumed decodes) holds only for a resumed decode
    still RUNNING on D: idle D -> no hold, whatever the round-trip price."""
    f, big, now = _asr(monkeypatch, awake_s=1.0, running=())
    f._seat_resumed_epoch = {"gone": f.epoch}           # resumed in this phase, finished since
    f._flip_round_trip_price = lambda: (30.0, "test")
    assert f._asr_min_dwell_held(big, f.groups["D"], now) is False
