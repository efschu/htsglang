# SPDX-License-Identifier: Apache-2.0
"""#1540 D-SIGNAL-SEATS: d_signal (D's id-space / Mamba-arena reading) is pressure only while D has
work that needs the rows -- a live seat or a leg-1-done request waiting for one.

Metal B9e (deskq/done/1520, boot ...fs10041810_d172504597, 18:14-18:22Z): with the front tag fix the
ladder SAW D's id space at 0.93-0.99 (D.log: 'full token usage', running-req 0, queue 0) as pressure in
EVERY tick. P stopped, lent (stage 1) and slept 0.6 s later (stage 2, sleep_after=3 ticks); the wake needs
``pressure <= 0 AND seats_done > seat_mark AND room_ok``: 7 seats finished after the sleep, but the
standing id-space reading kept pressure > 0, so P never woke (P queue 11-32). Workaround was
FLLIPER_PDFLIP_DUAL_D_ID_PRESSURE=1.0.

DANGER DIRECTION: P keeps running while D really is short. The gate only mutes the reading while there is
NO live seat and NOTHING waiting for one; a live seat or a waiting leg-1-done request leaves the reading
exactly as before. Default OFF = the B9e reading byte for byte (asserted as such).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.environ import envs
from flliper.srt.pdflip import card_kv_ledger as K
from flliper.srt.pdflip import dual_d_priority as DP
from flliper.srt.pdflip import front as FR
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GiB = 1 << 30
STEP_B = 64 << 20
AIR_B = 32 << 20
W_B = 10 * GiB
TAG = "d-signal-seats-1540-%d" % os.getpid()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("FLLIPER_PDFLIP_DUAL_LAYOUT", "FLLIPER_PDFLIP_GROUP", "FLLIPER_PDFLIP_DUAL_P_SLEEP",
              "FLLIPER_PDFLIP_DUAL_D_ID_PRESSURE", "FLLIPER_PDFLIP_TAG"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_KV_TAG", TAG)
    tmp = tempfile.mkdtemp(prefix="dsig1540")
    monkeypatch.setattr(DP, "d_signal_file", lambda tag, root="/dev/shm": os.path.join(tmp, "wkvd-%s.json" % tag))
    return tmp


def _flag(monkeypatch, on):
    if on:
        monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_SEATS", "1")
    else:
        monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_SEATS", raising=False)


def _publish(id_frac=0.93, arena_complete=118643, arena_slots=720896):
    # D's tick writes this (dual_d_kv_stage.publish_d_signal): the B9e reading, no arena pressure
    DP.publish_d_signal(DP.d_signal_file(TAG), id_frac=id_frac, arena_complete=arena_complete,
                        arena_slots=arena_slots, now=time.time())


def _ledger():
    path = os.path.join(tempfile.mkdtemp(prefix="wkvc1540"), "card")
    d = K.CardKvLedger(path, "D")
    d.contribute(2 * GiB, committed=0)                  # free ledger bytes, no D demand (B9e)
    p = K.CardKvLedger(path, "P")
    p.contribute(0)
    return path


def _front(seats=0, ready=0, with_attrs=True):
    f = types.SimpleNamespace(dual_kv_ledgers=[_ledger()])
    if with_attrs:
        f._d_seats_live = set(range(seats))
        f._ready_for_d = list(range(ready))
    return f


def _reading(f):
    return FR.Front._dual_p_stage_reading(f)


# -- the pure gate --------------------------------------------------------------

def test_gate_is_a_pass_through_when_disarmed():
    assert DP.d_signal_seat_gate(STEP_B, "id_space=0.93", d_has_seats=False, armed=False) == (STEP_B, "id_space=0.93")
    assert DP.d_signal_seat_gate(STEP_B, "id_space=0.93", d_has_seats=True, armed=False) == (STEP_B, "id_space=0.93")
    assert DP.d_signal_seat_gate(0, "", d_has_seats=False, armed=False) == (0, "")


def test_gate_mutes_only_without_seats():
    assert DP.d_signal_seat_gate(STEP_B, "id_space=0.93", d_has_seats=False, armed=True) == (0, "")
    assert DP.d_signal_seat_gate(STEP_B, "id_space=0.93", d_has_seats=True, armed=True) == (STEP_B, "id_space=0.93")
    assert DP.d_signal_seat_gate(0, "", d_has_seats=False, armed=True) == (0, "")


def test_env_switch_exists_and_defaults_off(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_SEATS", raising=False)
    assert envs.FLLIPER_PDFLIP_DUAL_D_SIGNAL_SEATS.get() is False
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_SEATS", "1")
    assert envs.FLLIPER_PDFLIP_DUAL_D_SIGNAL_SEATS.get() is True


# -- the front's reading (the B9e condition) ----------------------------------------

def test_default_off_the_standing_id_space_presses_with_or_without_seats(monkeypatch):
    _flag(monkeypatch, False)
    _publish(0.93)
    assert _reading(_front(seats=0, ready=0))["d_short"] > 0, "B9e: the old reading presses with no D work at all"
    assert _reading(_front(seats=2, ready=1))["d_short"] > 0
    # the stub of the older tests carries neither attribute: off must not touch them
    assert _reading(_front(with_attrs=False))["d_short"] > 0


def test_on_standing_id_space_without_d_work_is_no_pressure(monkeypatch):
    _flag(monkeypatch, True)
    _publish(0.93)
    assert _reading(_front(seats=0, ready=0))["d_short"] == 0
    _publish(0.99)
    assert _reading(_front(seats=0, ready=0))["d_short"] == 0


def test_on_a_live_seat_or_a_waiting_request_leaves_the_reading_unchanged(monkeypatch):
    _flag(monkeypatch, True)
    _publish(0.93)
    assert _reading(_front(seats=1, ready=0))["d_short"] > 0, "a live D seat"
    assert _reading(_front(seats=0, ready=1))["d_short"] > 0, "a leg-1-done request waits for a seat"
    off = _reading(_front(seats=1, ready=0))
    _flag(monkeypatch, False)
    assert _reading(_front(seats=1, ready=0))["d_short"] == off["d_short"], "same bytes as the old reading"


def test_on_the_arena_reading_is_gated_the_same_way(monkeypatch):
    _flag(monkeypatch, True)
    _publish(0.10, arena_complete=112, arena_slots=112)           # Mamba arena full, id space free
    assert _reading(_front(seats=0, ready=0))["d_short"] == 0
    assert _reading(_front(seats=1, ready=0))["d_short"] > 0


def test_on_ledger_demand_is_never_muted(monkeypatch):
    _flag(monkeypatch, True)
    _publish(0.93)
    f = _front(seats=0, ready=0)
    d = K.CardKvLedger(f.dual_kv_ledgers[0], "D")
    d.request(4 * GiB)                                            # D asks more than the card has: a real demand
    assert _reading(f)["d_short"] > 0, "the ledger bytes are not the d_signal; the gate must not touch them"


# -- the B9e stall as a stage-machine run -------------------------------------------

def _tick(st, d_short, seats_done, p_committed=0):
    return st.tick(pressure=d_short, p_committed=p_committed, free_min=GiB, p_grant_bytes=STEP_B,
                   d_air_bytes=AIR_B, seats_done=seats_done, weights_bytes=W_B, host_ok=True,
                   card_room=[(GiB, GiB // 2)] * 3)


def _run_b9e(monkeypatch, on):
    """stop -> lend -> sleep with a standing id-space; 7 seats finish after the sleep; does P wake?"""
    _flag(monkeypatch, on)
    _publish(0.93)
    st = DP.PressureStages(sleep_capable=True, sleep_after=3)
    seats = {"live": 1}                                            # D has a seat when the burst hits
    done = 0

    def ticks(n):
        out = []
        for _ in range(n):
            f = _front(seats=seats["live"], ready=0)
            out.append(_tick(st, _reading(f)["d_short"], done, p_committed=0))
        return out

    acts = [a for a, _ in ticks(8)]
    assert "stop" in acts and "lend" in acts and "sleep" in acts, acts
    assert st.p_state == "sleeping"
    seats["live"] = 0                                              # the 7 seats finish after the sleep
    done += 7
    acts = [a for a, _ in ticks(6)]
    return st, acts


def test_b9e_stall_off_p_never_wakes(monkeypatch):
    st, acts = _run_b9e(monkeypatch, on=False)
    assert "wake" not in acts and st.p_state == "sleeping", "the B9e stall, reproduced with the default"


def test_b9e_stall_on_p_wakes_after_the_seats_end(monkeypatch):
    st, acts = _run_b9e(monkeypatch, on=True)
    assert "wake" in acts and st.p_state == "serving", acts


def test_on_a_seat_that_is_still_live_keeps_p_down(monkeypatch):
    """The danger direction: D really works -> P stays stopped/asleep as before."""
    _flag(monkeypatch, True)
    _publish(0.93)
    st = DP.PressureStages(sleep_capable=True, sleep_after=3)
    acts = [_tick(st, _reading(_front(seats=2, ready=0))["d_short"], 0)[0] for _ in range(8)]
    assert "sleep" in acts
    assert [_tick(st, _reading(_front(seats=2, ready=0))["d_short"], 0)[0] for _ in range(6)] == [None] * 6
    assert st.p_state == "sleeping"
