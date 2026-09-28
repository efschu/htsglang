"""PARK-CYCLE DWELL (27B rc12k27 b23, 27.09.): a D phase that began by resuming
parked requests is priced at the whole park cycle (D->P + P->D flip), not at
one flip, before the next immediate park may fire (weg2/phase_policy.py
``park_cycle_dwell_ms``, front ``_immediate_park_due``).

Metal (front log 10:02:38-10:04:24): 6 immediate parks, each fired by a
DIFFERENT fresh over-X arrival (weg2-2-23, 4-24, 6-26, 8-29, 10-35, 12-39),
D awake at the fire 3.0 / 3.2 / 10.2 / 2.6 / 18.7 / 20.5 s against K7's
one-flip dwell of 2.2-3.3 s; P then drained 1-3 requests (0.6-8.9 s) and
flipped straight back; the same six rids were parked up to 5 times in 80 s."""
from __future__ import annotations

import os
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import phase_policy as pp  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(pp.PARK_CYCLE_DWELL_ENV, raising=False)
    # these pin the WAKE clock; the decode clock (PARK-DECODE-DWELL) has its own file
    monkeypatch.setenv(pp.PARK_DECODE_DWELL_ENV, "0")


def test_the_cycle_price_applies_only_to_a_resumed_phase_and_the_switch():
    assert pp.park_cycle_dwell_ms(2231, 2064, resumed_phase=True) == 4295
    assert pp.park_cycle_dwell_ms(2231, 2064, resumed_phase=False) == 2231  # K7 unchanged
    assert pp.park_cycle_dwell_ms(2231, 2064, resumed_phase=True, on=False) == 2231
    assert pp.park_cycle_dwell_ms(-5, -1, resumed_phase=True) == 0.0
    assert pp.park_cycle_dwell_on({}) and not pp.park_cycle_dwell_on({pp.PARK_CYCLE_DWELL_ENV: "0"})


# the metal parks: (awake_s at the fire, K7's D->P price ms, the last P->D price ms)
METAL = [
    ("weg2-2-23", 3.0, 2231, 2064),
    ("weg2-4-24", 3.2, 3071, 1843),
    ("weg2-6-26", 10.2, 3268, 1880),
    ("weg2-8-29", 2.6, 2473, 2036),
    ("weg2-10-35", 18.7, 2920, 1935),
    ("weg2-12-39", 20.5, 2506, 1816),
]


def test_the_metal_parks_right_after_a_resume_are_held_the_late_ones_fire():
    held_before = [rid for rid, a, dp, pd in METAL
                   if not pp.immediate_park_dwell_ok(a, dp, 2000.0)]
    held_after = [rid for rid, a, dp, pd in METAL
                  if not pp.immediate_park_dwell_ok(a, pp.park_cycle_dwell_ms(dp, pd, True), 2000.0)]
    assert held_before == []  # K7 let every one of them fire
    assert held_after == ["weg2-2-23", "weg2-4-24", "weg2-8-29"]
    # held means DELAYED, never dropped: each fires once D decoded the cycle's price
    for rid, a, dp, pd in METAL:
        need = pp.park_cycle_dwell_ms(dp, pd, True)
        assert pp.immediate_park_dwell_ok(need / 1000.0, need, 2000.0)
        assert need / 1000.0 - a <= 2.5  # the added delay is at most one P->D flip


# -- the front's gate --------------------------------------------------------


def _front(epoch, resume_epoch, awake_s, dp_ms, pd_ms):
    from sglang.srt.weg2 import front as F

    ns = types.SimpleNamespace(
        epoch=epoch, _park_attempt_epoch=-1, _park_resume_epoch=resume_epoch,
        _park_immediate_dwell_epoch=-1, t_awake=time.time() - awake_s,
        queue=[types.SimpleNamespace(rid="weg2-8-29", est_uncached=91755, t_arrive=time.time(),
                                     p_only=False, x_requeues=0, leg1_done=False, skip_leg1=False)],
        tp_prefill_max_tokens=4096, counters={"park_immediate_dwell_holds": 0},
        _flip_ledger=lambda g: ["weg2-0-1"],
        _derived_min_dwell_ms=lambda s, d: ((dp_ms, "last-flip-D->P") if (s, d) == ("D", "P")
                                            else (pd_ms, "last-flip-P->D")),
    )
    return F.Front._immediate_park_due, ns


def test_front_holds_the_resumed_phase_and_names_the_cycle(caplog):
    fn, ns = _front(epoch=10, resume_epoch=10, awake_s=2.6, dp_ms=2473, pd_ms=2036)
    with caplog.at_level("INFO"):
        assert fn(ns, None, time.time()) is None
    assert ns.counters["park_immediate_dwell_holds"] == 1
    assert any("PARK-IMMEDIATE-DWELL" in m and "+cycle:last-flip-P->D=2036ms" in m for m in caplog.messages)
    ns.t_awake = time.time() - 4.6  # 4510 ms of cycle decoded
    assert fn(ns, None, time.time()).rid == "weg2-8-29"


def test_front_fresh_phase_and_switch_off_keep_k7(monkeypatch):
    fn, ns = _front(epoch=10, resume_epoch=8, awake_s=2.6, dp_ms=2473, pd_ms=2036)
    assert fn(ns, None, time.time()).rid == "weg2-8-29"  # not a resumed phase: K7
    fn, ns = _front(epoch=10, resume_epoch=10, awake_s=2.6, dp_ms=2473, pd_ms=2036)
    monkeypatch.setenv(pp.PARK_CYCLE_DWELL_ENV, "0")
    assert fn(ns, None, time.time()).rid == "weg2-8-29"


def test_wiring_the_resume_stamps_the_phase():
    src = open(os.path.join(os.path.dirname(pp.__file__), "front.py"), encoding="utf-8").read()
    resume = src.index('self.counters["d_parked_resumed"] += len(_h91_parked)')
    stamp = src.index("self._park_resume_epoch = self.epoch", resume)
    clear = src.index("_h91_parked.clear()", resume)
    assert resume < stamp < clear
