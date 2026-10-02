# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-X (02.10.2026): X's price is the LONG excursion the request waits,
not divided by k.

User: "bei 15 sekunden flipzeit plus rechnung ist die schwelle aber nicht bei
6k sondern eher bei 35k oder sowas???"

N4p (...f405217a61_1002_095319.front.log) 10:06:26: X COST-LINE RE-SOLVE
X=6223, price=5.27 s (warm gather legs + resume 472 ms) / k=1.33 -> 3.96 s.
N4q (...58361d5471_1002_101341) 10:18:40: price 8.14 s / k=3.00 -> 2.71 s,
X=2918 -- k counted the acceptance probe's manual flips (10:17:15, 10:17:21).
The excursion the user waits, from the same N4p front log: D->P = R28 DP-WAIT
wait_s of LONG arrivals (cold epoch-1 excluded: 2.4, 2.4 s), P->D =
LEG2-FIRST-CONTENT via=after_p (cold first excluded: 3.089, 3.099, 3.169,
3.004, 2.960, 8.089, 4.307 s) -> 2.40 + 3.96 = 6.36 s, undivided.

weg2-14-69 (7691 uncached, 4096 in the store) went LONG at X=6404/6223; a
TEXT request of that size is SHORT under the corrected X even without the
store credit. (weg2-14-69 itself carries an image: W102 routes it to P,
which runs the transient tower -- vision only before P's prefill, a user law
-- whatever X says.)

Hermetic, CPU. Each test named red_* is red on 64ff550b21.
"""

from __future__ import annotations

import collections
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import phase_policy as pp  # noqa: E402

# N4p 10:06:26 COST-LINE RE-SOLVE inputs
D_LINE = {"a_ms": 155.0, "b_ms": 0.649, "c_ms": 0.0016, "n": 75, "n_lo": 64, "n_hi": 3367, "prefix_med": 67584}
P_LINE = {"a_ms": 53.0, "b_ms": 0.102, "c_ms": 0.0005, "n": 46, "n_lo": 64, "n_hi": 4096}
PREFIX = 67584
X_PREV = 6223
UNCACHED_14_69 = 7691
# N4p excursion samples (epoch, s); epoch 1 / 2 = the boot's first (cold) excursion
DP = [(1, 4.0), (1, 3.2), (15, 2.4), (17, 2.4)]
PD = [(2, 6.723), (8, 3.089), (10, 3.099), (12, 3.169), (14, 3.004), (16, 2.960), (18, 8.089), (20, 4.307)]
FIRST = {"dp": 1, "pd": 2}


def _x(price_s, k):
    x_star, why = pp.solve_x_cost_line(price_s=price_s, k=k, line=D_LINE, r_p=None,
                                       prefix_tokens=PREFIX, line_p=P_LINE)
    assert why == "ok"
    return x_star


def test_the_old_price_over_k_reproduces_n4p_and_n4q():
    assert abs(_x(5.27, 1.33) - 6224) < 15          # N4p 10:06:26 X*=6224
    assert _x(5.27, 1.33) < UNCACHED_14_69          # -> LONG


def test_red_excursion_price_is_the_warm_mean_of_both_waits():
    price, src = pp.excursion_price_s(DP, PD, (), FIRST)
    assert price == pytest.approx(2.4 + (3.089 + 3.099 + 3.169 + 3.004 + 2.960 + 8.089 + 4.307) / 7, abs=1e-6)
    assert "dp=2.40s(n=2)" in src and "pd=3.96s(n=7)" in src


def test_red_a_text_request_like_weg2_14_69_is_short_without_the_store_credit():
    price, _ = pp.excursion_price_s(DP, PD, (), FIRST)
    x_star = _x(price, 1.0)                           # undivided
    assert x_star > UNCACHED_14_69 and 9500 < x_star < 10600
    x, _limited = pp.x_step_limited(min(x_star, 12288), X_PREV, envs.SGLANG_WEG2_X_COST_MAX_STEP.get())
    assert x > UNCACHED_14_69                         # even the first hysteresis step admits it


def test_red_manual_probe_epochs_never_enter_the_price():
    dp = DP + [(30, 9.0)]
    pd = PD + [(31, 21.4)]                            # the probe flip's 21 s "vorlauf" class
    clean, _ = pp.excursion_price_s(DP, PD, (), FIRST)
    price, src = pp.excursion_price_s(dp, pd, {30, 31}, FIRST)
    assert price == pytest.approx(clean) and "2 manual epochs excluded" in src


def test_no_samples_names_why():
    price, src = pp.excursion_price_s([(1, 4.0)], [], (), {"dp": 1, "pd": None})
    assert price is None and src.startswith("excursion:none(dp n=0 pd n=0")


# ---------------------------------------------------------------- through the front

def _front(**kw):
    from sglang.srt.weg2 import front as F

    ns = types.SimpleNamespace(
        _d_cost_rows=collections.deque(maxlen=64), _d_cost_all=collections.deque(maxlen=4096),
        _x_cost_seed=dict(D_LINE, boot_tag="n4p", at="2026-10-02 10:06:26,626"),
        _park_rt_seed={"round_trip_s": 5.27, "boot_tag": "n4p", "at": "x"},
        flip_log=[], _resume_ms_log=[], p_phase_max_requests=6,
        _p_phase_k=collections.deque([1, 1, 2, 1, 2, 1], maxlen=32), X_SAMPLE_WINDOW=32, epoch=40)
    for k, v in kw.items():
        setattr(ns, k, v)
    ns._park_warm_legs_ms = lambda: F.Front._park_warm_legs_ms(ns)
    ns._x_cost_line_of = lambda *a: F.Front._x_cost_line_of(ns, *a)
    return F.Front, ns


def test_red_front_inputs_take_the_excursion_undivided():
    F, ns = _front(_x_exc_dp=collections.deque(DP), _x_exc_pd=collections.deque(PD),
                   _x_exc_first=dict(FIRST), _x_manual_epochs=set())
    with envs.SGLANG_WEG2_X_EXCURSION_PRICE.override(True):
        _line, _src, price, price_src, k, k_src = F._x_cost_inputs(ns)
    assert price == pytest.approx(6.36, abs=0.01) and price_src.startswith("excursion:")
    assert k == 1.0 and k_src.startswith("not-divided(measured k=1.33")


def test_front_without_excursion_samples_falls_back_to_the_ski_price_but_never_divides():
    F, ns = _front()
    with envs.SGLANG_WEG2_X_EXCURSION_PRICE.override(True):
        _l, _s, price, price_src, k, _ks = F._x_cost_inputs(ns)
    assert price == 5.27 and "->fallback ski-record:n4p" in price_src and k == 1.0


def test_switch_off_is_the_ski_price_over_k():
    F, ns = _front(_x_exc_dp=collections.deque(DP), _x_exc_pd=collections.deque(PD),
                   _x_exc_first=dict(FIRST), _x_manual_epochs=set())
    # RELEASE-INTEG: the ski price / k (mean) form is X-K-FLIP off as well (release head: on)
    with envs.SGLANG_WEG2_X_EXCURSION_PRICE.override(False), envs.SGLANG_WEG2_X_K_FLIP.override(False):
        _l, _s, price, price_src, k, k_src = F._x_cost_inputs(ns)
    assert price == 5.27 and price_src.startswith("ski-record") and k == pytest.approx(1.333, abs=0.01)


def test_red_the_front_records_samples_and_the_manual_flip_marks_its_epochs():
    import asyncio

    from sglang.srt.weg2 import front as F

    ns = types.SimpleNamespace(epoch=7, X_SAMPLE_WINDOW=32)
    F.Front._note_x_excursion(ns, "dp", 2.4)
    ns.epoch = 8
    F.Front._note_x_excursion(ns, "pd", 3.1)
    assert list(ns._x_exc_dp) == [(7, 2.4)] and list(ns._x_exc_pd) == [(8, 3.1)]
    assert ns._x_exc_first == {"dp": 7, "pd": 8}

    flips = []

    async def _flip(src, dst):
        flips.append((src, dst))
        ns.epoch += 1
        ns.awake = dst

    ns.update = None
    ns.dual_layout, ns.state, ns.awake, ns.admit_d = False, "serving", "D", True
    ns.epoch = 20
    ns._manual_flip_refusal = lambda: None
    ns.flip = _flip
    ns.state_dict = lambda: {"epoch": ns.epoch}

    async def _park(*a, **k):  # RELEASE-INTEG: the release head parks every manual flip from D
        return "parked"

    ns._wait_bound_park = _park
    asyncio.run(F.Front.handle_manual_flip(ns, None))
    assert flips == [("D", "P"), ("P", "D")]
    assert ns._x_manual_epochs == {21, 22}            # the probe's P phase and the D phase after it


def test_red_vision_excursions_never_price_x():
    """weg2-14-69 / weg2-16-73 (N4p) carry an image: W102 binds them to P,
    whose tower runs before the prefill -- a vision excursion is no text
    excursion. Their DP-WAIT / LEG2 waits are not sampled."""
    import inspect

    from sglang.srt.weg2 import front as F

    ns = types.SimpleNamespace(X_VISION_RIDS_MAX=2)
    F.Front._x_note_vision(ns, "weg2-14-69")
    assert F.Front._x_is_vision(ns, "weg2-14-69") and not F.Front._x_is_vision(ns, "weg2-14-70")
    F.Front._x_note_vision(ns, "a")
    F.Front._x_note_vision(ns, "b")
    assert not F.Front._x_is_vision(ns, "weg2-14-69")        # bounded, oldest first
    src = inspect.getsource(F.Front)
    assert 'a.origin == "long" and not Front._x_is_vision(self, p.rid)' in src
    assert '_via == "after_p" and not Front._x_is_vision(self, rid)' in src
    assert "Front._x_note_vision(self, rid)" in src


def test_n4p_text_only_samples_still_make_7691_short():
    """N4p's two warm D->P samples (epochs 15, 17) are weg2-14-69 and weg2-16-73 --
    both vision. Text only, the D->P side has no warm sample yet, the price falls
    back to the ski price -- undivided: 5.27 s -> X* ~8.3k > 7691, still SHORT.
    With the P->D text samples (3.089 3.099 3.169 3.004 4.307 s) and a first
    text D->P wait the excursion price takes over."""
    text_dp = [(1, 4.0), (1, 3.2)]
    text_pd = [(2, 6.723), (8, 3.089), (10, 3.099), (12, 3.169), (14, 3.004), (20, 4.307)]
    price, src = pp.excursion_price_s(text_dp, text_pd, (), FIRST)
    assert price is None and "dp n=0" in src
    x_star = _x(5.27, 1.0)
    assert 8200 < x_star < 8450 and x_star > UNCACHED_14_69


def test_red_the_cost_line_resolve_never_waits_for_the_launcher_import(monkeypatch):
    """N4p first D->P (09:57:42): kv wake answered 09:57:44.787, FLIP-TIMELINE done 46.310,
    yet `X COST-LINE RE-SOLVE` / `WEG2-FLIP done` only at 46.986 -- `resolve_x_live`
    imported sglang.srt.weg2.launcher first and sat on the import lock until the H75
    prewarm thread finished ("launcher prewarmed off the event loop in 5.8 s", 46.989):
    677 ms between the flip's end and P's first chunk (VM D>P nachlauf=677). The
    cost-line solve (default) needs nothing from the launcher."""
    import builtins

    from sglang.srt.weg2 import front as F

    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name == "sglang.srt.weg2.launcher":
            raise AssertionError("the cost-line resolve imported the launcher")
        return real_import(name, *a, **k)

    ns = types.SimpleNamespace(_resolve_x_cost_line=lambda: 4242)
    monkeypatch.setattr(builtins, "__import__", guard)
    with envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True):
        assert F.Front.resolve_x_live(ns) == 4242
