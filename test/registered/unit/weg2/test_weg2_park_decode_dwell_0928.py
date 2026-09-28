"""PARK-DECODE-DWELL (NF rc12z22, boot ...dauer09281447_7d507357b9, 14:52:02-
14:58:16): 22 flips in 374 s, D awake 38 %, P 42 %, flipping 20 %, and D
DECODING only 1-3 s of each 7-16 s D phase -- the first decode round of a phase
came 4.2-7.9 s after the wake (the resumed requests are reloaded store->device,
WEG2-LOAD-DEVICE weg2-0-5 43520 tok 1065 ms each, serially), while the
immediate park's dwell (K7 + cycle, 6.4 s) ran from the WAKE and fired just as
decode began. 2088 completion tokens in 374 s = 335 tok/min for every agent.

Pinned: the dwell counts from D's first streamed chunk of the phase and prices
both flips PLUS this phase's resume (decode duty >= 50 %); with no chunk it
holds for twice the cycle on the wake clock (a phase that streams nothing must
not wait for the fairness bound); "0" restores the wake clock exactly."""
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
    monkeypatch.delenv(pp.PARK_DECODE_DWELL_ENV, raising=False)


def test_policy_counts_decode_not_wake():
    # NF 14:56:23: cycle ~6.5 s, resume 5.2 s -> need 11.7 s of DECODE
    assert not pp.park_decode_dwell_ok(awake_s=6.5, decode_s=1.3, cycle_ms=11700, floor_ms=2000)
    assert pp.park_decode_dwell_ok(awake_s=17.0, decode_s=11.8, cycle_ms=11700, floor_ms=2000)
    # no decoded chunk yet: twice the cycle on the wake clock, then it fires
    assert not pp.park_decode_dwell_ok(awake_s=10.0, decode_s=None, cycle_ms=6000, floor_ms=2000)
    assert pp.park_decode_dwell_ok(awake_s=12.0, decode_s=None, cycle_ms=6000, floor_ms=2000)
    # the fairness floor still binds
    assert not pp.park_decode_dwell_ok(awake_s=5.0, decode_s=1.0, cycle_ms=100, floor_ms=2000)
    assert pp.park_decode_dwell_on({}) and not pp.park_decode_dwell_on({pp.PARK_DECODE_DWELL_ENV: "0"})


def test_duty_is_configurable_and_clamped():
    assert pp.park_decode_duty({}) == 0.5
    assert pp.park_decode_duty({pp.PARK_DECODE_DUTY_ENV: "0.75"}) == 0.75
    assert pp.park_decode_duty({pp.PARK_DECODE_DUTY_ENV: "7"}) == 0.9
    assert pp.park_decode_duty({pp.PARK_DECODE_DUTY_ENV: "x"}) == 0.5
    # 75 % duty: decode 3x the cycle
    assert not pp.park_decode_dwell_ok(30, 17.0, 6000, 2000, duty=0.75)
    assert pp.park_decode_dwell_ok(30, 18.1, 6000, 2000, duty=0.75)


def test_nf_1456_resume_of_five_parked_park_waits_for_decode():
    """14:56:23-14:56:24: the wake re-read and reloaded 5 parked requests
    (WEG2-LOAD-DEVICE ~1 s each, serially), first decode ~5 s after the wake;
    an over-X arrival during the resume. Before: the wake clock let the park
    fire at the cycle (6.5 s) -- 1.5 s after decode began. After: it waits
    for the cycle + resume of DECODE."""
    cycle_ms, resume_s = 3544 + 2935, 5.0
    assert pp.immediate_park_dwell_ok(6.5, cycle_ms, 2000)          # before: fires
    assert not pp.park_decode_dwell_ok(6.5, 1.5, cycle_ms + resume_s * 1000, 2000)  # after: held
    assert pp.park_decode_dwell_ok(17.0, 11.6, cycle_ms + resume_s * 1000, 2000)


def _front(epoch, resume_epoch, awake_s, first_chunk_after_s, dp_ms, pd_ms):
    from sglang.srt.weg2 import front as F

    now = time.time()
    t_awake = now - awake_s
    ns = types.SimpleNamespace(
        epoch=epoch, _park_attempt_epoch=-1, _park_resume_epoch=resume_epoch,
        _park_immediate_dwell_epoch=-1, t_awake=t_awake,
        _d_decode_epoch=(epoch if first_chunk_after_s is not None else -1),
        _d_decode_t0=(t_awake + first_chunk_after_s if first_chunk_after_s is not None else 0.0),
        queue=[types.SimpleNamespace(rid="weg2-17-22", est_uncached=70000, t_arrive=now,
                                     p_only=False, x_requeues=0, leg1_done=False, skip_leg1=False)],
        tp_prefill_max_tokens=4096, counters={"park_immediate_dwell_holds": 0},
        _flip_ledger=lambda g: ["weg2-0-5", "weg2-4-11"],
        _derived_min_dwell_ms=lambda s, d: ((3544, "last-flip-D->P") if (s, d) == ("D", "P")
                                            else (2935, "last-flip-P->D")),
    )
    return F.Front._immediate_park_due, ns


def test_front_nf_metal_park_at_6_4s_is_now_held(caplog):
    # NF epoch 18: resumed phase, first decode 5.2 s after the wake, park fired at 6.4 s
    fn, ns = _front(epoch=18, resume_epoch=18, awake_s=6.4, first_chunk_after_s=5.2,
                    dp_ms=3544, pd_ms=2935)
    with caplog.at_level("INFO"):
        assert fn(ns, None, time.time()) is None
    assert any("PARK-IMMEDIATE-DWELL" in m and "decode-clock" in m for m in caplog.messages)
    # it fires once D decoded cycle (3544+2935) + resume (5200) = 11.7 s
    ns.t_awake = time.time() - (5.2 + 11.8)
    ns._d_decode_t0 = ns.t_awake + 5.2
    assert fn(ns, None, time.time()).rid == "weg2-17-22"


def test_front_fresh_phase_prices_both_flips_and_switch_off_is_the_wake_clock(monkeypatch):
    # a fresh (not resumed) phase: both flips + resume from the decode clock
    fn, ns = _front(epoch=10, resume_epoch=8, awake_s=4.0, first_chunk_after_s=1.0,
                    dp_ms=3544, pd_ms=2935)
    assert fn(ns, None, time.time()) is None           # 3 s of decode < 7.5 s
    monkeypatch.setenv(pp.PARK_DECODE_DWELL_ENV, "0")
    fn, ns = _front(epoch=10, resume_epoch=8, awake_s=4.0, first_chunk_after_s=1.0,
                    dp_ms=3544, pd_ms=2935)
    assert fn(ns, None, time.time()).rid == "weg2-17-22"  # K7 on the wake clock, as before


def test_front_stamps_the_first_d_chunk_once_per_phase():
    src = open(os.path.join(os.path.dirname(pp.__file__), "front.py"), encoding="utf-8").read()
    i = src.index("PARK-DECODE-DWELL: the first chunk D streams")
    blk = src[i:i + 400]
    assert 'self.awake == "D" and self._d_decode_epoch != self.epoch' in blk
    assert "self._d_decode_t0 = time.time()" in blk
