# SPDX-License-Identifier: Apache-2.0
"""#363 act -- the decision rule reads the measured band AND the flip cost.

THE RULE THIS SUITE PINS, verbatim from ``regime_ms_clock``:

    signal(C)    = 100 * (mean_total_ms - predicted_ms(C)) / mean_total_ms
    threshold(C) = max(enter_margin_pct, band(C) + flip_cost_pct(C))
    band(C)      = sqrt(band(I)^2 + band(C)^2)
    flip_cost_pct(C) = 100 * flip_cost_s(C) / flip_payback_s

Before this slice the clock compared ``signal`` against the band and the enter
margin and NEVER against the flip cost -- the instrumented cost existed on the
stage, was used by ``DwellGate`` to decide how long a flip must be HELD, and
had no bearing on whether the flip was WORTH MAKING. So a stage that is 6 %
faster and takes four seconds to move into was adopted on the same evidence as
one that is 6 % faster and moves in 80 ms.

The discriminating test is ``test_an_expensive_flip_is_refused_where_a_cheap
_one_is_taken``: the two candidates differ in NOTHING except ``flip_cost_s``,
so it fails the moment the cost stops entering the decision. The can-fail arm
recorded with this suite reverts the threshold to the pre-slice ``max(band,
enter_margin)`` and that test goes red on its own.

The second duty is that the rule must not become an excuse never to move: a
cheap flip with a real gain is still taken, and the suite pins that direction
too.
"""

from __future__ import annotations

import math

import pytest

from sglang.srt.managers.regime_classifier import REGIME_MIXED, Stage
from sglang.srt.managers.regime_ms_clock import (
    DEFAULT_FLIP_PAYBACK_S,
    MsClockError,
    MsStageDecider,
    combined_band_pct,
    decision_threshold_pct,
    flip_cost_pct,
    improvement_pct,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_stage(name, *, gain=0.0, band=1.0, flip_cost_s=1.0, unmeasured=False):
    if unmeasured:
        gain = band = flip_cost_s = 0.0
    return Stage(
        name=name,
        regime=REGIME_MIXED,
        weight_vector=None,
        kv_token_vector=(1, 1),
        vram_budget_mib=(8000, 8000),
        max_total_num_tokens=100_000,
        measured_gain_pct=gain,
        measured_band_pct=band,
        flip_cost_s=flip_cost_s,
        unmeasured=unmeasured,
    )


def run(decider, current, candidates, split, rounds):
    """Feed ``rounds`` identical boundaries and return the last decision."""
    compute, wait = split
    last = None
    for i in range(rounds):
        for _ in range(decider.window.min_samples):
            decider.observe_round(i, compute, wait)
        last = decider.decide(current, candidates)
    return last


# -- the two new functions ---------------------------------------------------


def test_the_flip_cost_is_priced_over_the_payback_horizon():
    stage = make_stage("c", gain=10.0, flip_cost_s=1.2)
    assert flip_cost_pct(stage, payback_s=60.0) == pytest.approx(2.0)
    assert flip_cost_pct(stage) == pytest.approx(100.0 * 1.2 / DEFAULT_FLIP_PAYBACK_S)


def test_pricing_an_unmeasured_stages_flip_is_refused_not_read_as_free():
    with pytest.raises(MsClockError) as exc:
        flip_cost_pct(make_stage("c", unmeasured=True))
    assert "UNMEASURED" in str(exc.value)


def test_a_non_positive_horizon_is_refused():
    with pytest.raises(MsClockError):
        flip_cost_pct(make_stage("c"), payback_s=0.0)


def test_the_threshold_is_the_max_of_the_margin_and_band_plus_cost():
    cur = make_stage("i", gain=0.0, band=4.0, flip_cost_s=0.0)
    cand = make_stage("c", gain=20.0, band=3.0, flip_cost_s=3.0)
    band = combined_band_pct(cur, cand)
    assert band == pytest.approx(math.sqrt(16.0 + 9.0))
    got = decision_threshold_pct(cur, cand, enter_margin_pct=5.0, payback_s=60.0)
    assert got == pytest.approx(band + 5.0)  # 5.0 = 100 * 3 / 60


def test_the_enter_margin_is_a_floor_under_the_threshold_not_a_third_term():
    cur = make_stage("i", band=0.1, flip_cost_s=0.0)
    cand = make_stage("c", gain=20.0, band=0.1, flip_cost_s=0.06)
    # band ~0.14 %, cost 0.1 % -> both far under the 5 % policy floor.
    assert decision_threshold_pct(cur, cand, enter_margin_pct=5.0) == pytest.approx(5.0)


# -- the rule in the decider -------------------------------------------------

#: Half the round is spent at a barrier, so the stage axes have something to
#: win. With a 30 % measured gain that is 11.5 % of the round.
WAIT_BOUND = (50.0, 50.0)


def test_a_cheap_flip_with_a_real_gain_is_still_taken():
    cur = make_stage("balanced", gain=0.0, band=1.0, flip_cost_s=0.05)
    fast = make_stage("split-heavy", gain=30.0, band=1.0, flip_cost_s=0.05)
    decision = run(MsStageDecider(), cur, [fast], WAIT_BOUND, 6)
    assert decision.target == "split-heavy"
    assert decision.flip_cost_pct == pytest.approx(100.0 * 0.05 / 60.0)
    assert decision.threshold_pct == pytest.approx(5.0)


def test_an_expensive_flip_is_refused_where_a_cheap_one_is_taken():
    """The discriminating test: the ONLY difference is ``flip_cost_s``."""
    cur = make_stage("balanced", gain=0.0, band=1.0, flip_cost_s=0.05)
    cheap = make_stage("cheap", gain=30.0, band=1.0, flip_cost_s=0.05)
    dear = make_stage("dear", gain=30.0, band=1.0, flip_cost_s=8.0)

    assert run(MsStageDecider(), cur, [cheap], WAIT_BOUND, 6).target == "cheap"

    refused = run(MsStageDecider(), cur, [dear], WAIT_BOUND, 6)
    assert refused.target is None
    assert refused.signal_pct == pytest.approx(11.5, abs=0.1)
    # 8 s over a 60 s horizon = 13.3 % of a round, above the 11.5 % signal.
    assert refused.threshold_pct == pytest.approx(1.414 + 13.333, abs=0.01)
    assert "would not repay what it costs" in refused.reason


def test_the_refusal_names_the_cost_the_horizon_and_the_band_separately():
    cur = make_stage("balanced", gain=0.0, band=1.0, flip_cost_s=0.05)
    dear = make_stage("dear", gain=30.0, band=1.0, flip_cost_s=8.0)
    reason = run(MsStageDecider(), cur, [dear], WAIT_BOUND, 6).reason
    assert "instrumented flip cost" in reason
    assert "payback horizon" in reason
    assert "enter margin" in reason


def test_a_longer_horizon_makes_the_same_flip_affordable():
    """The horizon is policy and it is the ONLY judgement in the term."""
    cur = make_stage("balanced", gain=0.0, band=1.0, flip_cost_s=0.05)
    dear = make_stage("dear", gain=30.0, band=1.0, flip_cost_s=8.0)
    taken = run(MsStageDecider(flip_payback_s=600.0), cur, [dear], WAIT_BOUND, 6)
    assert taken.target == "dear"
    assert taken.flip_cost_pct == pytest.approx(100.0 * 8.0 / 600.0)


def test_the_band_case_and_the_cost_case_are_reported_differently():
    """Same refusal in outcome, different remedy, so different reason."""
    cur = make_stage("balanced", gain=0.0, band=12.0, flip_cost_s=0.05)
    inside_band = make_stage("noise", gain=1.0, band=12.0, flip_cost_s=0.05)
    banded = run(MsStageDecider(), cur, [inside_band], WAIT_BOUND, 6)
    assert banded.target is None
    assert "does not clear its combined A-vs-A band" in banded.reason
    assert "would not repay" not in banded.reason


def test_a_non_positive_payback_horizon_is_refused_at_construction():
    with pytest.raises(MsClockError) as exc:
        MsStageDecider(flip_payback_s=0.0)
    assert "removes the instrumented flip cost" in str(exc.value)


def test_the_decision_carries_the_rule_into_the_trace():
    cur = make_stage("balanced", gain=0.0, band=1.0, flip_cost_s=0.05)
    fast = make_stage("split-heavy", gain=30.0, band=1.0, flip_cost_s=0.05)
    row = run(MsStageDecider(), cur, [fast], WAIT_BOUND, 6).as_dict()
    assert row["flip_cost_pct"] is not None
    assert row["threshold_pct"] is not None
    assert row["signal_pct"] > row["threshold_pct"]


def test_the_summary_reports_the_horizon_it_priced_with():
    assert MsStageDecider().summary()["flip_payback_s"] == DEFAULT_FLIP_PAYBACK_S


def test_candidates_are_ranked_by_improvement_not_by_margin_over_threshold():
    """A cheap flip must not win the comparison by being cheap."""
    cur = make_stage("balanced", gain=0.0, band=1.0, flip_cost_s=0.05)
    big = make_stage("big-win", gain=30.0, band=1.0, flip_cost_s=1.0)
    small = make_stage("small-win", gain=8.0, band=1.0, flip_cost_s=0.01)
    assert improvement_pct(50.0, 50.0, cur, big) > improvement_pct(
        50.0, 50.0, cur, small
    )
    decision = run(MsStageDecider(), cur, [small, big], WAIT_BOUND, 6)
    assert decision.candidate == "big-win"
