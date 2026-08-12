"""#363 intra-phase axis -- the ms/round stage clock. Hermetic, no GPU.

What this suite is for: the controller already flips between stages on a
REGIME LABEL. This module adds the measurement to the decision, and the two
duties of the suite are symmetric.

* IT MUST FLIP. A sustained shift into a wait-bound regime, where the stage
  axes have something to win, produces a proposal.
* IT MUST NOT FLIP. Noise, a single-boundary spike, a signal inside the
  combined A-vs-A band, and an oscillating load all produce nothing. A
  controller that cannot be provoked is as broken as one that cannot be
  calmed, so both directions are pinned here rather than one.

The load-bearing test is ``test_improvement_depends_on_the_measured_split``.
The first version of this module scaled the whole round by the ratio of the
two measured gains, which cancels the measured round out of the verdict
algebraically -- an "ms/round-driven" controller that never reads a
millisecond. That test is what keeps the arithmetic honest; delete it and the
regression is invisible.
"""

import unittest

from sglang.srt.managers.regime_classifier import REGIME_MIXED, Stage
from sglang.srt.managers.regime_ms_clock import (
    MsClockError,
    MsRoundWindow,
    MsStageDecider,
    combined_band_pct,
    improvement_pct,
    pack_ms_sample,
    predicted_round_ms,
    unpack_ms_sample,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_stage(
    name,
    *,
    gain=0.0,
    band=1.0,
    flip_cost_s=1.0,
    kv=(1, 1),
    vram=(8000, 8000),
    tokens=100_000,
    unmeasured=False,
):
    if unmeasured:
        gain = band = flip_cost_s = 0.0
    return Stage(
        name=name,
        regime=REGIME_MIXED,
        weight_vector=None,
        kv_token_vector=tuple(kv),
        vram_budget_mib=tuple(vram),
        max_total_num_tokens=tokens,
        measured_gain_pct=gain,
        measured_band_pct=band,
        flip_cost_s=flip_cost_s,
        unmeasured=unmeasured,
    )


#: The incumbent and a stage measured 30 % faster. With band 1.0 each the
#: combined band is sqrt(2) = 1.41 %, well under the signal a wait-bound
#: window produces and well over the one a compute-bound window produces.
CUR = make_stage("balanced", gain=0.0, band=1.0)
FAST = make_stage("split-heavy", gain=30.0, band=1.0)

#: A wait-bound window: half the round is spent at a barrier, so the axes
#: that move the split have something to win. 100*50*(1-1/1.3)/100 = 11.5 %.
WAIT_BOUND = (50.0, 50.0)
#: A compute-bound window: 5 % of the round is wait. 1.15 %, under the exit
#: watermark, so it does not even start the leave streak.
COMPUTE_BOUND = (95.0, 5.0)


def feed(decider, split, rounds=8, start=0):
    """Fill the sliding window with ``rounds`` identical samples."""
    compute, wait = split
    for i in range(rounds):
        decider.observe_round(start + i, compute, wait)


def run_boundaries(decider, current, split, n, candidates=(FAST,)):
    """Drive ``n`` boundaries at a fixed split; return (flips, decisions).

    Mirrors what the observer does: on a proposal the stage in force changes
    and the decider is told, which resets its streaks and clears the window.
    """
    flips = []
    decisions = []
    stage = current
    by_name = {s.name: s for s in (current,) + tuple(candidates)}
    for b in range(n):
        feed(decider, split, start=b * 8)
        others = [s for s in by_name.values() if s.name != stage.name]
        d = decider.decide(stage, others)
        decisions.append(d)
        if d.wants_flip:
            flips.append((b, d.target))
            stage = by_name[d.target]
            decider.note_flip()
    return flips, decisions


class TestConstructionRefusals(unittest.TestCase):
    """Every watermark law is a constructor check, not a convention."""

    def test_exit_margin_above_enter_is_refused(self):
        with self.assertRaises(MsClockError) as cm:
            MsStageDecider(enter_margin_pct=2.0, exit_margin_pct=5.0)
        self.assertIn("hysteresis", str(cm.exception))

    def test_equal_margins_are_refused(self):
        with self.assertRaises(MsClockError):
            MsStageDecider(enter_margin_pct=5.0, exit_margin_pct=5.0)

    def test_exit_window_not_longer_than_enter_is_refused(self):
        with self.assertRaises(MsClockError) as cm:
            MsStageDecider(enter_window=4, exit_window=4)
        self.assertIn("asymmetry is the contract", str(cm.exception))

    def test_zero_enter_margin_is_refused(self):
        with self.assertRaises(MsClockError):
            MsStageDecider(enter_margin_pct=0.0, exit_margin_pct=0.0)

    def test_min_samples_above_capacity_is_refused(self):
        with self.assertRaises(MsClockError) as cm:
            MsRoundWindow(capacity=4, min_samples=8)
        self.assertIn("never become ready", str(cm.exception))

    def test_negative_sample_is_refused_not_clamped(self):
        w = MsRoundWindow()
        with self.assertRaises(MsClockError):
            w.observe(0, -1.0, 0.0)
        with self.assertRaises(MsClockError):
            w.observe(0, 0.0, -1.0)


class TestArithmetic(unittest.TestCase):
    def test_improvement_depends_on_the_measured_split(self):
        """THE regression guard. Same two stages, two different windows.

        If the improvement is computed by scaling the whole round, these two
        assertions collapse to the same number and the controller stops being
        ms-driven. The compute-bound case must be strictly smaller.
        """
        wait_bound = improvement_pct(*WAIT_BOUND, CUR, FAST)
        compute_bound = improvement_pct(*COMPUTE_BOUND, CUR, FAST)
        self.assertGreater(wait_bound, compute_bound)
        self.assertAlmostEqual(wait_bound, 11.538, places=2)
        self.assertAlmostEqual(compute_bound, 1.154, places=2)

    def test_a_window_with_no_wait_offers_nothing_to_win(self):
        self.assertAlmostEqual(improvement_pct(100.0, 0.0, CUR, FAST), 0.0)

    def test_predicted_ms_rescales_only_the_wait_term(self):
        predicted = predicted_round_ms(50.0, 50.0, CUR, FAST)
        self.assertAlmostEqual(predicted, 50.0 + 50.0 / 1.3, places=6)

    def test_reverse_direction_is_negative(self):
        self.assertLess(improvement_pct(*WAIT_BOUND, FAST, CUR), 0.0)

    def test_zero_round_is_refused(self):
        with self.assertRaises(MsClockError) as cm:
            predicted_round_ms(0.0, 0.0, CUR, FAST)
        self.assertIn("set the prediction by the substitution", str(cm.exception))

    def test_unmeasured_stage_is_refused_in_either_role(self):
        """#578 placeholder zeros are not a measured gain of zero."""
        unmeasured = make_stage("planner-solved", unmeasured=True)
        with self.assertRaises(MsClockError) as cm:
            improvement_pct(*WAIT_BOUND, CUR, unmeasured)
        self.assertIn("UNMEASURED", str(cm.exception))
        with self.assertRaises(MsClockError):
            improvement_pct(*WAIT_BOUND, unmeasured, CUR)

    def test_combined_band_is_quadrature_and_exceeds_either(self):
        a = make_stage("a", gain=0.0, band=3.0)
        b = make_stage("b", gain=10.0, band=4.0)
        self.assertAlmostEqual(combined_band_pct(a, b), 5.0, places=6)
        self.assertGreater(combined_band_pct(a, b), 4.0)


class TestWindow(unittest.TestCase):
    def test_not_ready_below_min_samples(self):
        w = MsRoundWindow(min_samples=8)
        for i in range(7):
            w.observe(i, 10.0, 10.0)
        self.assertFalse(w.ready)
        w.observe(7, 10.0, 10.0)
        self.assertTrue(w.ready)

    def test_means_and_wait_share(self):
        w = MsRoundWindow(min_samples=1)
        w.observe(0, 75.0, 25.0)
        self.assertAlmostEqual(w.mean_total_ms, 100.0)
        self.assertAlmostEqual(w.mean_wait_share, 0.25)

    def test_window_slides(self):
        w = MsRoundWindow(capacity=4, min_samples=1)
        for i in range(4):
            w.observe(i, 100.0, 0.0)
        for i in range(4):
            w.observe(4 + i, 0.0, 100.0)
        self.assertAlmostEqual(w.mean_wait_ms, 100.0)
        self.assertEqual(len(w), 4)


class TestDecisionMustFlip(unittest.TestCase):
    def test_sustained_wait_bound_regime_flips(self):
        d = MsStageDecider()
        flips, decisions = run_boundaries(d, CUR, WAIT_BOUND, 8)
        self.assertEqual(len(flips), 1, [x.reason for x in decisions])
        self.assertEqual(flips[0][1], "split-heavy")
        # exit_window=4 is the binding half: not before the 4th boundary.
        self.assertEqual(flips[0][0], 3)

    def test_the_regime_change_is_what_moves_it(self):
        """Compute-bound first: nothing. Then wait-bound: it moves.

        This is the intra-phase axis in one test -- the stage table, the
        candidate and the thresholds are all identical across the two halves,
        and the ONLY thing that changed is the measured split.
        """
        d = MsStageDecider()
        flips_a, _ = run_boundaries(d, CUR, COMPUTE_BOUND, 10)
        self.assertEqual(flips_a, [])
        flips_b, _ = run_boundaries(d, CUR, WAIT_BOUND, 10)
        self.assertEqual(len(flips_b), 1)


class TestDecisionMustNotFlip(unittest.TestCase):
    def test_compute_bound_never_flips(self):
        """40 boundaries of a compute-bound rig move nothing.

        The incumbent never even starts its leave streak: at 1.15 % the best
        candidate is under the 2 % exit watermark AND inside the 1.41 %
        combined band, so both halves refuse independently.
        """
        d = MsStageDecider()
        flips, decisions = run_boundaries(d, CUR, COMPUTE_BOUND, 40)
        self.assertEqual(flips, [])
        self.assertEqual(d.leave_streak, 0)
        self.assertEqual(d.candidate_streak, 0)
        self.assertIn("does not clear its combined", decisions[-1].reason)

    def test_one_boundary_of_clean_signal_is_not_enough(self):
        """A full window of wait-bound samples, but only ONE boundary.

        The signal is well over the enter watermark and over its band, so the
        only thing refusing is the sustain requirement. That is the property
        under test: hysteresis, not a weak signal.
        """
        d = MsStageDecider()
        feed(d, WAIT_BOUND, rounds=64)
        spike = d.decide(CUR, [FAST])
        self.assertFalse(spike.wants_flip)
        self.assertGreater(spike.signal_pct, d.enter_margin_pct)
        self.assertGreater(spike.signal_pct, spike.band_pct)
        self.assertEqual(spike.leave_streak, 1)
        self.assertEqual(spike.candidate_streak, 1)
        # The challenger's own streak is checked first, so that is the reason
        # reported at boundary 1; the incumbent's slower leave window becomes
        # the binding one from boundary 2 on.
        self.assertIn("for 1 of 2 required boundaries", spike.reason)

    def test_signal_inside_the_band_never_flips_however_long(self):
        """A gain inside its own A-vs-A band is not a gain."""
        noisy_cur = make_stage("balanced", gain=0.0, band=12.0)
        noisy_fast = make_stage("split-heavy", gain=30.0, band=12.0)
        d = MsStageDecider()
        flips, decisions = run_boundaries(
            d, noisy_cur, WAIT_BOUND, 60, candidates=(noisy_fast,)
        )
        self.assertEqual(flips, [])
        self.assertIn("does not clear its combined", decisions[-1].reason)
        self.assertGreater(decisions[-1].band_pct, decisions[-1].signal_pct)

    def test_alternating_leader_resets_the_challenger_streak(self):
        """Two near-tied candidates trading places must not accumulate."""
        a = make_stage("cand-a", gain=30.0, band=1.0)
        b = make_stage("cand-b", gain=30.0, band=1.0)
        d = MsStageDecider()
        stage = CUR
        flips = []
        for i in range(40):
            feed(d, WAIT_BOUND, start=i * 8)
            # The leader alternates: each boundary a different stage is best.
            leader, other = (a, b) if i % 2 == 0 else (b, a)
            bumped = make_stage(leader.name, gain=31.0, band=1.0)
            dec = d.decide(stage, [bumped, other])
            if dec.wants_flip:
                flips.append(dec.target)
                d.note_flip()
        self.assertEqual(flips, [])
        self.assertLessEqual(d.candidate_streak, 1)

    def test_unmeasured_candidates_are_skipped_not_proposed(self):
        d = MsStageDecider()
        planner_stage = make_stage("planner-solved", unmeasured=True)
        flips, decisions = run_boundaries(
            d, CUR, WAIT_BOUND, 12, candidates=(planner_stage,)
        )
        self.assertEqual(flips, [])
        self.assertIn("nothing to compare", decisions[-1].reason)

    def test_unmeasured_incumbent_makes_the_axis_abstain(self):
        d = MsStageDecider()
        unmeasured_cur = make_stage("planner-solved", unmeasured=True)
        feed(d, WAIT_BOUND)
        dec = d.decide(unmeasured_cur, [FAST])
        self.assertFalse(dec.wants_flip)
        self.assertIn("UNMEASURED", dec.reason)

    def test_cold_window_proposes_nothing(self):
        d = MsStageDecider()
        d.observe_round(0, 50.0, 50.0)
        dec = d.decide(CUR, [FAST])
        self.assertFalse(dec.wants_flip)
        self.assertIn("not ready", dec.reason)


class TestThrash(unittest.TestCase):
    def test_oscillating_load_produces_bounded_flips(self):
        """The thrash guard: a load alternating every single boundary.

        The correct outcome is BOUNDED, not zero, and the distinction is the
        point of the test. The sliding window spans several boundaries, so a
        load flipping faster than the window resolves is not seen as two
        regimes taking turns -- it is seen as ONE steady mixed regime whose
        time-averaged split is moderately wait-heavy. Settling once onto the
        stage that suits that mix is right; doing it repeatedly is thrash.

        So the assertion is a small bound over 60 boundaries. A regression
        that reopens per-boundary flipping shows up as a count in the tens.
        """
        d = MsStageDecider()
        stage = CUR
        by_name = {CUR.name: CUR, FAST.name: FAST}
        flips = []
        for i in range(60):
            split = WAIT_BOUND if i % 2 == 0 else COMPUTE_BOUND
            feed(d, split, start=i * 8)
            others = [s for s in by_name.values() if s.name != stage.name]
            dec = d.decide(stage, others)
            if dec.wants_flip:
                flips.append(i)
                stage = by_name[dec.target]
                d.note_flip()
        self.assertLessEqual(len(flips), 3, f"thrash: flipped at {flips}")

    def test_slow_oscillation_flips_but_stays_far_below_boundary_count(self):
        """A genuinely alternating workload at a period the hysteresis admits.

        Some flips are correct here -- the regime really is changing. What
        must hold is that the count stays bounded well below one per
        boundary; the dwell gate (regime_classifier.DwellGate, deliberately a
        separate mechanism) is what bounds it further in production.
        """
        d = MsStageDecider()
        stage = CUR
        by_name = {CUR.name: CUR, FAST.name: FAST}
        flips = []
        for i in range(120):
            split = WAIT_BOUND if (i // 12) % 2 == 0 else COMPUTE_BOUND
            feed(d, split, start=i * 8)
            others = [s for s in by_name.values() if s.name != stage.name]
            dec = d.decide(stage, others)
            if dec.wants_flip:
                flips.append(i)
                stage = by_name[dec.target]
                d.note_flip()
        self.assertGreater(len(flips), 0)
        self.assertLess(len(flips), 12)

    def test_note_flip_clears_the_window_and_the_streaks(self):
        """Samples from the old stage must not judge the new one."""
        d = MsStageDecider()
        feed(d, WAIT_BOUND)
        d.decide(CUR, [FAST])
        self.assertGreater(len(d.window), 0)
        d.note_flip()
        self.assertEqual(len(d.window), 0)
        self.assertEqual(d.candidate_streak, 0)
        self.assertEqual(d.leave_streak, 0)
        self.assertIsNone(d.candidate)


class TestGroupReduction(unittest.TestCase):
    """The ms axis must consume a GROUP statistic, never this rank's own.

    On a real rig every rank measures a different number. A controller that
    decided from its own would have every rank propose a different stage --
    permanently vetoed at best, a hang at worst. These pin the reduction that
    makes the input uniform, and the conservative direction of each term.
    """

    @staticmethod
    def group_min(payloads):
        """Element-wise MIN, i.e. what the collective channel does."""
        return [min(col) for col in zip(*payloads)]

    def test_round_length_is_the_slowest_rank(self):
        fast = pack_ms_sample(40.0, 10.0)  # total 50
        slow = pack_ms_sample(60.0, 20.0)  # total 80
        compute, wait = unpack_ms_sample(self.group_min([fast, slow]))
        self.assertAlmostEqual(compute + wait, 80.0, places=4)

    def test_wait_is_the_least_any_rank_paid(self):
        a = pack_ms_sample(40.0, 10.0)
        b = pack_ms_sample(60.0, 20.0)
        _, wait = unpack_ms_sample(self.group_min([a, b]))
        self.assertAlmostEqual(wait, 10.0, places=4)

    def test_a_single_rank_round_trips(self):
        compute, wait = unpack_ms_sample(pack_ms_sample(70.0, 30.0))
        self.assertAlmostEqual(compute, 70.0, places=4)
        self.assertAlmostEqual(wait, 30.0, places=4)

    def test_one_blind_rank_makes_the_group_blind(self):
        """A partial group is not a group -- absence must survive the MIN."""
        seen = pack_ms_sample(50.0, 50.0)
        blind = pack_ms_sample(None, None)
        self.assertIsNone(unpack_ms_sample(self.group_min([seen, blind])))

    def test_the_conservative_bias_understates_the_gain(self):
        """Long round, small addressable term: both bias the same way."""
        a = pack_ms_sample(40.0, 10.0)
        b = pack_ms_sample(60.0, 20.0)
        compute, wait = unpack_ms_sample(self.group_min([a, b]))
        group_signal = improvement_pct(compute, wait, CUR, FAST)
        # Either rank's own numbers would have promised more than the group
        # statistic does.
        self.assertLess(group_signal, improvement_pct(40.0, 10.0, CUR, FAST))
        self.assertLess(group_signal, improvement_pct(60.0, 20.0, CUR, FAST))

    def test_malformed_payload_is_refused(self):
        with self.assertRaises(MsClockError):
            unpack_ms_sample([1, 2, 3])


class TestDeterminism(unittest.TestCase):
    def test_tie_break_is_by_name_not_by_iteration_order(self):
        """A tie broken by dict order is a rank-local decision in disguise."""
        a = make_stage("aaa", gain=30.0, band=1.0)
        z = make_stage("zzz", gain=30.0, band=1.0)
        forward = MsStageDecider()
        reverse = MsStageDecider()
        feed(forward, WAIT_BOUND)
        feed(reverse, WAIT_BOUND)
        self.assertEqual(
            forward.decide(CUR, [a, z]).candidate,
            reverse.decide(CUR, [z, a]).candidate,
        )
        self.assertEqual(forward.decide(CUR, [a, z]).candidate, "aaa")


if __name__ == "__main__":
    unittest.main()
