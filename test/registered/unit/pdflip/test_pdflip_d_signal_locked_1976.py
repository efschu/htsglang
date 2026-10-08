# SPDX-License-Identifier: Apache-2.0
"""#1976 D-SIGNAL-LOCKED: d_signal reads D's LOCKED rows and the Mamba arena's PINNED slots, not the table fill.

Metal f11 (docker-acceptance 27b boot fs10050941_325a123a15, front/P/D logs 10:00-10:14Z, deskq/done/1976):
the 262k request pdflip-0-85 was paused ten times ('PDFLIP DUAL P-PAUSE ... pressure=92274688 B'). 92274688 B is
NOT a measured shortage but ``unit`` -- one P grant step (4096 tok x 22528 B/tok) -- which
``dual_d_priority.d_signal_short`` returns for each reason it finds. The reasons were permanent:

  * id_space 0.91-1.00 in all 224 non-gated readings: ``used = size - available - evictable`` counts the
    ~1M rows D never mapped (D_KV_MAX_TOKENS=1048576, D maps 8192-45056 of them; D.log 'full token usage
    0.98' with ONE running request);
  * arena 111-112/112 COMPLETE slots (a full cache; ARENA-REF-CENSUS pinned=4..18).

With #1540 D-SIGNAL-SEATS on, the reading counts as soon as D has a seat: 56 of 67 stage-1 events came within
1 s of a D-ADMIT (median 0.12 s), stage 2 put P to sleep (54 sleeps, median 11 s, 69 % of 10:00-10:14:50 asleep).

DANGER DIRECTION: P keeps running although D really is short of rows. The switch changes only WHICH D reading
is compared with the thresholds: locked rows >= 0.90 of the id space or pinned slots >= slots - 2 still press
exactly as before; a reading without the new keys (an older D) falls back to the old fields; ledger demand
(bytes) is untouched. Default OFF = the old reading byte for byte (asserted).
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
from flliper.srt.pdflip import dual_d_kv_stage as DK
from flliper.srt.pdflip import dual_d_priority as DP
from flliper.srt.pdflip import front as FR
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GiB = 1 << 30
UNIT = 92274688                       # f11: one P grant step, 4096 tok x 22528 B/tok
TAG = "d-signal-locked-1976-%d" % os.getpid()
NOW = 1000.0


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("FLLIPER_PDFLIP_DUAL_LAYOUT", "FLLIPER_PDFLIP_GROUP", "FLLIPER_PDFLIP_DUAL_P_SLEEP",
              "FLLIPER_PDFLIP_DUAL_D_ID_PRESSURE", "FLLIPER_PDFLIP_TAG", "FLLIPER_PDFLIP_DUAL_D_SIGNAL_SEATS",
              "FLLIPER_PDFLIP_DUAL_D_SIGNAL_LOCKED"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_KV_TAG", TAG)
    tmp = tempfile.mkdtemp(prefix="dsig1976")
    monkeypatch.setattr(DP, "d_signal_file", lambda tag, root="/dev/shm": os.path.join(tmp, "wkvd-%s.json" % tag))
    return tmp


# the f11 reading at a D-ADMIT: D mapped 8192 of 1048576 rows, 1 running request; arena 112/112 complete, 18 pinned
F11 = {"ts": NOW, "id_frac": 0.99, "arena_complete": 112, "arena_slots": 112,
       "locked_frac": 0.027, "arena_pinned": 18}


def _short(sig, locked, **kw):
    return DP.d_signal_short(sig, now=NOW, unit=UNIT, locked=locked, **kw)


# -- the pure reading -----------------------------------------------------------------

def test_env_switch_exists_and_defaults_off(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_LOCKED", raising=False)
    assert envs.FLLIPER_PDFLIP_DUAL_D_SIGNAL_LOCKED.get() is False
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_LOCKED", "1")
    assert envs.FLLIPER_PDFLIP_DUAL_D_SIGNAL_LOCKED.get() is True


def test_default_off_the_f11_reading_presses_as_before():
    """The measured fault, reproduced with the default: both table-fill reasons, d_need = one grant step."""
    assert _short(F11, locked=False) == (UNIT, "id_space=0.99,arena=112/112")
    # the keyword is optional: the old call shape is the old answer
    assert DP.d_signal_short(F11, now=NOW, unit=UNIT) == (UNIT, "id_space=0.99,arena=112/112")


def test_on_the_f11_reading_is_no_pressure():
    assert _short(F11, locked=True) == (0, "")


def test_on_a_really_locked_id_space_still_presses():
    sig = dict(F11, locked_frac=0.93)
    short, why = _short(sig, locked=True)
    assert short == UNIT and why == "id_locked=0.93"
    assert _short(dict(F11, locked_frac=0.90), locked=True)[0] == UNIT          # the threshold is inclusive, as before
    assert _short(dict(F11, locked_frac=0.899), locked=True)[0] == 0


def test_on_a_really_pinned_arena_still_presses():
    sig = dict(F11, arena_pinned=110)                                           # slots - 2: the old margin
    short, why = _short(sig, locked=True)
    assert short == UNIT and why == "arena_pinned=110/112"
    assert _short(dict(F11, arena_pinned=109), locked=True)[0] == 0


def test_on_both_reasons_name_both():
    sig = dict(F11, locked_frac=0.95, arena_pinned=112)
    assert _short(sig, locked=True) == (UNIT, "id_locked=0.95,arena_pinned=112/112")


def test_on_an_older_d_without_the_keys_falls_back_to_the_old_fields():
    """Danger direction: a missing field never mutes the reading."""
    old = {"ts": NOW, "id_frac": 0.99, "arena_complete": 112, "arena_slots": 112}
    assert _short(old, locked=True) == (UNIT, "id_space=0.99,arena=112/112")
    only_id = dict(old, arena_pinned=3)                                         # pinned known, locked_frac absent
    assert _short(only_id, locked=True) == (UNIT, "id_space=0.99")
    only_arena = dict(old, locked_frac=0.02)
    assert _short(only_arena, locked=True) == (UNIT, "arena=112/112")


def test_on_a_stale_or_missing_reading_is_still_nothing():
    assert _short(None, locked=True) == (0, "")
    assert _short(dict(F11, ts=NOW - 6.0, locked_frac=0.99, arena_pinned=112), locked=True) == (0, "")


# -- what D publishes ------------------------------------------------------------------

def test_publish_without_the_new_fields_writes_the_old_keys_only(tmp_path):
    p = str(tmp_path / "sig.json")
    DP.publish_d_signal(p, id_frac=0.5, arena_complete=3, arena_slots=112, now=NOW)
    assert sorted(json.load(open(p))) == ["arena_complete", "arena_slots", "id_frac", "ts"]


def test_publish_with_the_new_fields_adds_two_keys(tmp_path):
    p = str(tmp_path / "sig.json")
    DP.publish_d_signal(p, id_frac=0.5, arena_complete=3, arena_slots=112, now=NOW, locked_frac=0.02,
                        arena_pinned=7)
    d = json.load(open(p))
    assert d["locked_frac"] == 0.02 and d["arena_pinned"] == 7 and d["id_frac"] == 0.5


class _Alloc:
    def __init__(self, size, avail):
        self.size = size
        self._avail = avail

    def available_size(self):
        return self._avail


def _d_actor(size=1048576, mapped=8192, avail=3000):
    return types.SimpleNamespace(allocator=_Alloc(size, avail), mapped_tokens=mapped)


def _d_sched(evictable=0):
    return types.SimpleNamespace(tp_rank=0, tree_cache=types.SimpleNamespace(evictable_size=lambda: evictable))


def test_d_tick_publishes_locked_frac_next_to_the_old_id_frac(monkeypatch, tmp_path):
    """f11 shape: D mapped 8192 of the 1M-row id space, 5192 rows in use. The OLD id_frac reads 0.995 (the
    unmapped rows count as used); the locked rows are 5192 / 1048576."""
    p = str(tmp_path / "sig.json")
    monkeypatch.setattr(DP, "d_signal_file", lambda tag, root="/dev/shm": p)
    monkeypatch.setattr(DK, "_arena_census", lambda: (112, 112))
    monkeypatch.setattr(DK, "_arena_pinned", lambda: 18)
    actor = _d_actor(mapped=8192, avail=3000)
    DK.publish_d_signal(_d_sched(), actor)
    d = json.load(open(p))
    assert d["id_frac"] == pytest.approx((1048576 - 3000) / 1048576)            # unchanged
    assert d["id_frac"] > 0.99
    assert d["locked_frac"] == pytest.approx((8192 - 3000) / 1048576)
    assert d["locked_frac"] < 0.01
    assert d["arena_complete"] == 112 and d["arena_slots"] == 112 and d["arena_pinned"] == 18


def test_d_tick_evictable_cache_is_not_locked(monkeypatch, tmp_path):
    p = str(tmp_path / "sig.json")
    monkeypatch.setattr(DP, "d_signal_file", lambda tag, root="/dev/shm": p)
    monkeypatch.setattr(DK, "_arena_census", lambda: (0, 0))
    monkeypatch.setattr(DK, "_arena_pinned", lambda: None)
    actor = _d_actor(mapped=45056, avail=1000)
    DK.publish_d_signal(_d_sched(evictable=28845), actor)                       # CACHE-YIELD shape of f11 10:00:52
    d = json.load(open(p))
    assert d["locked_frac"] == pytest.approx((45056 - 1000 - 28845) / 1048576)
    assert "arena_pinned" not in d                                              # no arena: key absent -> old reading


def test_d_tick_full_mapped_pool_is_locked_as_before(monkeypatch, tmp_path):
    """y8v shape (the reason of Q-660): every row of the id space mapped and locked -> both readings agree."""
    p = str(tmp_path / "sig.json")
    monkeypatch.setattr(DP, "d_signal_file", lambda tag, root="/dev/shm": p)
    monkeypatch.setattr(DK, "_arena_census", lambda: (112, 112))
    monkeypatch.setattr(DK, "_arena_pinned", lambda: 112)
    actor = _d_actor(size=1048576, mapped=1048576, avail=40000)
    DK.publish_d_signal(_d_sched(), actor)
    d = json.load(open(p))
    assert d["id_frac"] == pytest.approx(d["locked_frac"]) and d["locked_frac"] > 0.96
    assert _short(d, locked=True)[0] == UNIT and _short(d, locked=False)[0] == UNIT


# -- the front's reading ---------------------------------------------------------------

def _publish(sig):
    json.dump(sig, open(DP.d_signal_file(TAG), "w"))


def _front(seats=1):
    path = os.path.join(tempfile.mkdtemp(prefix="wkvc1976"), "card")
    d = K.CardKvLedger(path, "D")
    d.contribute(2 * GiB, committed=0)
    p = K.CardKvLedger(path, "P")
    p.contribute(0)
    return types.SimpleNamespace(dual_kv_ledgers=[path], _d_seats_live=set(range(seats)), _ready_for_d=[])


def _reading(f):
    return FR.Front._dual_p_stage_reading(f)


def test_front_default_off_f11_reading_presses_with_a_seat(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_SEATS", "1")                  # the f11 boot had #1540 on
    _publish(dict(F11, ts=time.time()))
    assert _reading(_front(seats=1))["d_short"] > 0, "f11 reproduced: a D seat plus table fill = pressure"


def test_front_on_f11_reading_does_not_press_but_real_demand_does(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_SEATS", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_LOCKED", "1")
    _publish(dict(F11, ts=time.time()))
    f = _front(seats=1)
    assert _reading(f)["d_short"] == 0
    d = K.CardKvLedger(f.dual_kv_ledgers[0], "D")
    d.request(4 * GiB)                                                          # D asks more than the card has
    assert _reading(f)["d_short"] > 0, "ledger bytes are never muted by this switch"


def test_front_on_a_really_locked_d_still_presses(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_LOCKED", "1")
    _publish(dict(F11, ts=time.time(), locked_frac=0.97))
    assert _reading(_front(seats=0, ))["d_short"] > 0


# -- the stage machine: the f11 loop ---------------------------------------------------

def _tick(st, d_short, seats_done):
    return st.tick(pressure=d_short, p_committed=0, free_min=GiB, p_grant_bytes=UNIT, d_air_bytes=32 << 20,
                   seats_done=seats_done, weights_bytes=10 * GiB, host_ok=True, card_room=[(GiB, GiB // 2)] * 3)


def _seat_burst(monkeypatch, on):
    """One short D seat (f11: pdflip-0-83, 3.8 s) while a leg 1 runs: does P leave 'serving'?"""
    if on:
        monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_LOCKED", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_SEATS", "1")
    _publish(dict(F11, ts=time.time()))
    st = DP.PressureStages(sleep_capable=True, sleep_after=3)
    acts = [_tick(st, _reading(_front(seats=1))["d_short"], 0)[0] for _ in range(8)]
    return st, acts


def test_f11_loop_off_a_short_seat_stops_lends_and_sleeps_p(monkeypatch):
    st, acts = _seat_burst(monkeypatch, on=False)
    assert "stop" in acts and "lend" in acts and "sleep" in acts and st.p_state == "sleeping", acts


def test_f11_loop_on_a_short_seat_leaves_p_serving(monkeypatch):
    st, acts = _seat_burst(monkeypatch, on=True)
    assert set(acts) == {None} and st.p_state == "serving", acts


# -- flip unchanged ----------------------------------------------------------------------

def test_flip_form_runs_no_d_signal_reading_even_with_the_switch_on(monkeypatch):
    """The flip form has no dual ledgers: the pump takes neither the pressure nor the stage tick, so the
    switch has nothing to read; D's tick without the dual D actor publishes no signal."""
    import collections
    from unittest import mock

    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_D_SIGNAL_LOCKED", "1")
    sched = types.SimpleNamespace(tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace()))
    with mock.patch.object(DK, "publish_d_signal", side_effect=AssertionError("published")):
        assert DK.tick(sched) is None
    f = types.SimpleNamespace(dual_kv_ledgers=[], queue=collections.deque(), counters=collections.Counter(),
                              _dual_task=None, _dual_backoff_until=0.0)
    with mock.patch.object(FR.Front, "_dual_pressure_tick", side_effect=AssertionError("pressure")), \
            mock.patch.object(FR.Front, "_dual_stage_tick", side_effect=AssertionError("stage")), \
            mock.patch.object(DP, "d_signal_short", side_effect=AssertionError("d_signal")):
        FR.Front._dual_pump(f, lambda: None)
