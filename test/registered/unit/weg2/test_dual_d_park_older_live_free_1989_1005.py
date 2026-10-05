# SPDX-License-Identifier: Apache-2.0
"""#1989 D-PARK OLDER-LIVE-FREE (dual D, SGLANG_WEG2_DUAL_D_PARK_OLDER_LIVE_FREE, default off).

METAL ``boot_weg2_dkr27bnvfp4dual1sstreambar1fs10051150_0a17db10c3_1005_115046`` (pt2), D:

  11:56:02 SEAT-AGE DISPLACE rid_out=weg2-0-7 older_waiting=weg2-0-5 trigger=kv running=2 cap=6
  11:56:04 Prefill batch ... #cached-token: 71390        (0-5 admitted; 0-5 decodes 12648 tok, 243 s)
  11:56:24-11:59:51  83/83 TP0 'Decode batch, #running-req: 1 ... #queue-req: 5', 0 X-GATE and
                     0 Prefill batch on TP0-TP2 -- 0-7 parked (older-live: 0-5 runs), 0-8, 0-11,
                     0-12, 0-18 'weg2_d_park_first' behind it, ~100k rows free
  12:00:05 WEG2-SERVED group=D rid=weg2-0-5 -> the same second X-GATE admit 0-7, 0-18, 0-8 ...,
           'Prefill batch, #new-seq: 4'; weg2-0-18 (1132 tok) LEG2-FIRST-CONTENT leg2_ms=233292

Fix: when EVERY parked request still waiting is a pressure park blocked only by an older RUNNING
request (it cannot resume this pass), newcomers take the free seats beside it -- at most
``seat_cap - running - parked`` this pass, its own seat stays held. Not when the victim is still
held for an older WAITING request (Q-698), not with an F3 deferral, not with a park outside the
queue, never in the flip form or with the switch off.
"""
from __future__ import annotations

import logging
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import d_park_runtime as DPR  # noqa: E402
from sglang.srt.weg2 import d_seats as DS  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ON = "SGLANG_WEG2_DUAL_D_PARK_OLDER_LIVE_FREE"
DUAL_D_ON = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D", ON: "1"}
KEYS = ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP", ON, "SGLANG_WEG2_SEAT_ROTATE",
        "SGLANG_WEG2_D_PARK_BARRIER_ADMITTED", DS.RESUME_MARGIN_ENV)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)


def _env(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def _req(rid, span=1000, out=0):
    return types.SimpleNamespace(rid=rid, origin_input_ids=[0] * span, output_ids=[0] * out,
                                 prefix_indices=[], kv_arrival_seq=int(rid.rsplit("-", 1)[1]))


def _parked(rid="weg2-0-7", held_for=None, site=DS.SITE_PRESSURE):
    v = _req(rid, span=38144, out=39)
    DS.mark_parked(v, site)
    if held_for is not None:
        setattr(v, DS.DISPLACED_FOR_ATTR, held_for)
    return v


def _pt2(held_for=None):
    """The pt2 queue at 11:56:24: 0-5 runs, 0-7 parked, four younger newcomers."""
    running = [_req("weg2-0-5", span=71391, out=200)]
    v7 = _parked(held_for=held_for)
    newcomers = [_req("weg2-0-8", 37998), _req("weg2-0-11", 9819), _req("weg2-0-12", 9819),
                 _req("weg2-0-18", 1132)]
    return running, v7, newcomers


def _admit_pass(gate, waiting):
    """The scheduler loop's use of the gate: skip() before each candidate, the
    admitted rids accumulate (KV/X/adder admit everything here)."""
    admitted, skipped = [], {}
    for r in waiting:
        why = gate.skip(r, admitted=list(admitted))
        if why is None:
            admitted.append(str(r.rid))
        else:
            skipped[str(r.rid)] = why
    return admitted, skipped


# -- the metal case -------------------------------------------------------------------------

def test_pt2_newcomers_take_the_free_seats_beside_the_older_live_victim(monkeypatch, caplog):
    _env(monkeypatch, DUAL_D_ON)
    caplog.set_level(logging.INFO, logger=DS.logger.name)
    running, v7, newcomers = _pt2()
    waiting = DS.order_waiting([v7] + newcomers)
    gate = DS.admission_gate(waiting, running=running, seat_cap=6)
    assert gate.barrier is True and "weg2-0-7" in gate.blocked
    assert gate.seat_room == 4, gate.note
    admitted, skipped = _admit_pass(gate, waiting)
    assert skipped == {"weg2-0-7": "weg2_d_park_older_live"}, (
        "pt2: 0-7 blocked behind the running 0-5 held 0-8/0-11/0-12/0-18 behind the barrier for 209 s")
    assert admitted == ["weg2-0-8", "weg2-0-11", "weg2-0-12", "weg2-0-18"]
    assert "older_live_free=1 seat_room=4" in gate.note
    assert ("WEG2-D-PARK OLDER-LIVE-FREE rid=weg2-0-7 older_running=weg2-0-5 seat_room=4 "
            "running=1 parked=1 cap=6") in caplog.text


def test_the_parked_seat_stays_held(monkeypatch):
    """A fifth newcomer would take the seat 0-7 comes back to: it waits."""
    _env(monkeypatch, DUAL_D_ON)
    running, v7, newcomers = _pt2()
    extra = _req("weg2-0-19", 500)
    waiting = DS.order_waiting([v7] + newcomers + [extra])
    gate = DS.admission_gate(waiting, running=running, seat_cap=6)
    admitted, skipped = _admit_pass(gate, waiting)
    assert len(admitted) == 4 and skipped["weg2-0-19"] == "weg2_d_park_first"
    # seats already full (cap - running - parked == 0): the barrier as before
    gate0 = DS.admission_gate(waiting, running=running + [_req("weg2-0-2")] * 4, seat_cap=6)
    assert gate0.seat_room is None
    assert gate0.skip(newcomers[0], admitted=[]) == "weg2_d_park_first"


# -- where the barrier must stand -----------------------------------------------------------

def test_q698_victim_held_for_an_older_waiting_request_keeps_the_barrier(monkeypatch):
    """11:56:02: 0-7 parked FOR 0-5 while 0-5 still waits for KV -- 0-5 gets it first."""
    _env(monkeypatch, DUAL_D_ON)
    _, v7, newcomers = _pt2(held_for="weg2-0-5")
    older = _req("weg2-0-5", span=71391)
    waiting = DS.order_waiting([v7, older] + newcomers)
    gate = DS.admission_gate(waiting, running=[_req("weg2-0-3")], seat_cap=6)
    assert gate.seat_room is None and "held_for_older=1" in gate.note
    assert gate.skip(older, admitted=[]) is None                  # older than 0-7: SA lets it go
    for r in newcomers:
        assert gate.skip(r, admitted=[]) == "weg2_d_park_first"


def test_a_resumable_parked_request_keeps_the_barrier(monkeypatch):
    """Nothing older runs: 0-7 comes back THIS pass -- the barrier is for it."""
    _env(monkeypatch, DUAL_D_ON)
    _, v7, newcomers = _pt2()
    gate = DS.admission_gate([v7] + newcomers, running=[_req("weg2-0-9")], seat_cap=6)
    assert "weg2-0-7" not in gate.blocked and gate.seat_room is None
    assert gate.skip(v7, admitted=[]) is None
    # AP is on by default: once 0-7 is admitted this pass the newcomers follow (unchanged)
    assert gate.skip(newcomers[0], admitted=[]) == "weg2_d_park_first"
    assert gate.skip(newcomers[0], admitted=["weg2-0-7"]) is None


def test_a_flip_park_or_a_park_outside_the_queue_keeps_the_barrier(monkeypatch):
    _env(monkeypatch, DUAL_D_ON)
    running, v7, newcomers = _pt2()
    flip = _parked("weg2-0-6", site=DS.SITE_FLIP)
    g1 = DS.admission_gate([flip, v7] + newcomers, running=running, seat_cap=6)
    assert g1.seat_room is None
    outside = _parked("weg2-0-4")
    g2 = DS.admission_gate([v7] + newcomers, running=running, pending_outside=[outside], seat_cap=6)
    assert g2.seat_room is None
    for g in (g1, g2):
        assert g.skip(newcomers[-1], admitted=[]) == "weg2_d_park_first"


def test_an_f3_deferral_keeps_the_barrier(monkeypatch):
    _env(monkeypatch, DUAL_D_ON)
    running, v7, newcomers = _pt2()
    df = types.SimpleNamespace(defer=lambda pw, w, b: frozenset({"weg2-0-7"}), wake_seq=3)
    gate = DS.admission_gate([v7] + newcomers, running=running, seat_cap=6, decode_first=df)
    assert gate.seat_room is None


# -- default off / flip unchanged -----------------------------------------------------------

@pytest.mark.parametrize("env", [
    {},                                                                 # flip form
    {"SGLANG_WEG2_GROUP": "D"},                                         # flip D
    {"SGLANG_WEG2_GROUP": "D", ON: "1"},                                # flip D, switch alone
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"},         # dual D, default off
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P", ON: "1"},  # dual P
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D", ON: "0"},
])
def test_off_and_flip_form_keep_the_pt2_barrier(monkeypatch, env):
    _env(monkeypatch, env)
    assert DS.older_live_free_armed() is False
    running, v7, newcomers = _pt2()
    waiting = DS.order_waiting([v7] + newcomers)
    gate = DS.admission_gate(waiting, running=running, seat_cap=6)
    assert gate.seat_room is None and "older_live_free" not in gate.note
    admitted, skipped = _admit_pass(gate, waiting)
    assert admitted == [] and set(skipped.values()) == {"weg2_d_park_older_live", "weg2_d_park_first"}


def test_off_the_gate_equals_the_call_without_a_seat_cap(monkeypatch):
    """Byte-for-byte: with the switch off a seat cap changes nothing in the verdict."""
    _env(monkeypatch, {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"})
    running, v7, newcomers = _pt2()
    waiting = DS.order_waiting([v7] + newcomers)
    a = DS.admission_gate(waiting, running=running, seat_cap=6)
    b = DS.admission_gate(waiting, running=running)
    assert a == b


def _sched(running, waiting, cap=6):
    batch = types.SimpleNamespace(reqs=list(running), spec_algorithm=None)
    sched = types.SimpleNamespace(waiting_queue=list(waiting),
                                  server_args=types.SimpleNamespace(max_running_requests=cap))
    return sched, batch


def _admission(sched, batch, seat_cap_calls):
    def _cap(s):
        seat_cap_calls.append(1)
        return 6

    with mock.patch.object(DS, "d_flip_park_active", lambda: True), \
         mock.patch.object(DS, "d_park_active", lambda: False), \
         mock.patch.object(DPR, "seat_cap", _cap), \
         mock.patch.object(DPR, "displace_for_age", lambda s, b: None), \
         mock.patch.object(DPR, "_apply_park_defer", lambda s: 0), \
         mock.patch.object(DPR, "decode_first_facts", lambda s, b: None):
        return DPR.admission(sched, batch)


def test_runtime_passes_the_seat_cap_only_when_armed(monkeypatch):
    running, v7, newcomers = _pt2()
    _env(monkeypatch, {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"})
    calls = []
    gate = _admission(*_sched(running, [v7] + newcomers), calls)
    assert calls == [] and gate.seat_room is None, "default off read the seat cap"
    monkeypatch.setenv(ON, "1")
    gate = _admission(*_sched(running, [v7] + newcomers), calls)
    assert calls == [1] and gate.seat_room == 4
