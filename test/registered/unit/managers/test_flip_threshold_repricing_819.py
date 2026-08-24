"""#819: the flip threshold is PRICED FROM THE MEASURED SEAM, not frozen at boot.

RED-FIRST, against the window-3 specimen. The live boot emitted, on 49
decision lines, a bar that never moved:

    PHASE-POLICY holding in tp: pending prefill 9962 tok > N=7004 but
    <= 12608 with 4 req decoding: too short for the round trip to beat
    prefilling it in tp

while the SAME process, on the same boot, also emitted (three times, once
per rank):

    #777 N IS STALE AND STAYS STALE. The live threshold is N=7004 tok,
    priced at boot off the UNMEASURED seam seed 3.2s. Measured flips put
    the seam at 3.60287s, which prices the same break-even at 7886 tok
    (1.13x). N is built once, in config_from_env, and nothing reprices it.

Both lines are true at once, and that is the whole defect: the estimator
followed the regime and the THRESHOLD did not. #777 named the gap and
deliberately left the actuator to the planner. This is that actuator.

WHY REPRICING ON ONE RANK IS SAFE, and why it is not a #616g divergence.
The three ranks measured DIFFERENT prices (3.60287 / 3.66144 / 4.62869 s),
which price N at 7886 / 8014 / 10131 -- so a bar that followed each rank's
own estimate would differ per rank. It does not diverge the group because
the policy is not evaluated per rank: `recv_requests` runs the hook only on
the request-origin rank and BROADCASTS the resulting arm
(request_receiver.py:169-193, whose own comment records the measured
alternative -- "1/2/3 arms on PP0/PP1/PP2, a 12765-line capture-census
flood, and a self-kill"). One rank decides; every rank obeys the same
broadcast arm.

KNOWN AND NAMED LIMITATION: the deciding rank prices off ITS OWN leg, and
phase_flip_boot's own docstring states "the flip's cost is the SLOWEST
rank's copy". When the deciding rank is the fast one the bar is priced
low. That is strictly better than pricing off an UNMEASURED constant --
the error shrinks from 3.2s-vs-reality to fast-rank-vs-slowest-rank -- but
it is an error, and closing it needs a group reduce that no existing
collective carries today. Not silently assumed away here.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers import phase_policy as pp
from sglang.srt.managers.phase_policy import (
    PHASE_TP,
    PhasePolicyConfig,
    PhasePolicyInputs,
    PhasePolicyState,
    break_even_tokens,
    decide,
    effective_flip_threshold,
    live_flip_tokens,
)
from sglang.test.test_utils import CustomTestCase

# The window-3 boot's own numbers, so every assertion below is a live one.
N_BOOT = 7004
SEED_C = 3.2
#: The three per-rank seam measurements from the #777 lines in
#: boot_window3_0823_1733.log, and the N each one prices.
MEASURED = ((3.60287, 7886), (3.66144, 8014), (4.62869, 10131))
#: The throughputs config_from_env priced N_BOOT with. break_even_tokens
#: inverts to exactly 7004 at the 3.2 s seed, which is what pins these as the
#: live pair rather than a plausible one -- asserted below.
TP_TOK_S = 1681.0
PP_TOK_S = 7245.5


def _cfg(**kw):
    base = dict(
        enabled=True,
        flip_tokens=N_BOOT,
        min_dwell_s=0.0,
        pp_window_s=15.0,
        tp_decode_floor_s=0.0,
        prefill_runs_in_tp=True,
        flip_cost_s=SEED_C,
    )
    base.update(kw)
    return PhasePolicyConfig(**base)


class _EstimatorFixture(CustomTestCase):
    """Save/restore the process-global estimator and the boot pricing."""

    def setUp(self):
        self._saved_est = pp._FLIP_COST_ESTIMATOR
        self._saved_boot = pp._FLIP_TOKENS_AT_BOOT
        self._saved_pricing = pp._FLIP_TOKENS_PRICING
        self._saved_said = pp._FLIP_TOKENS_STALE_SAID
        pp._FLIP_COST_ESTIMATOR = None
        pp.note_flip_tokens_pricing(N_BOOT, TP_TOK_S, PP_TOK_S, False)

    def tearDown(self):
        pp._FLIP_COST_ESTIMATOR = self._saved_est
        pp._FLIP_TOKENS_AT_BOOT = self._saved_boot
        pp._FLIP_TOKENS_PRICING = self._saved_pricing
        pp._FLIP_TOKENS_STALE_SAID = self._saved_said

    def _measure(self, seconds):
        """Calibrate the estimator to exactly `seconds` of ROUND TRIP.

        The first observation REPLACES the seed outright (FlipCostEstimator's
        own documented rule), so one call pins the value with no EMA lag.

        #856: an UNDIRECTED reading is a round trip and is split evenly across
        the two legs, so `value()` returns what was observed -- except below
        two leg-minimums, where each leg floors at `MIN_ESTIMATE_S` and the
        round trip therefore floors at twice it. That floor is the point of
        the constant (it stops the multiplicative band collapsing to zero),
        and it applies per leg because each leg carries its own band.
        """
        pp._FLIP_COST_ESTIMATOR = pp.RoundTripFlipCost(seed_s=SEED_C)
        pp.observe_flip_cost(seconds)
        floor = 2.0 * pp.FlipCostEstimator.MIN_ESTIMATE_S
        self.assertAlmostEqual(
            pp._FLIP_COST_ESTIMATOR.value(), max(seconds, floor), places=6
        )


class TestTheSpecimenIsReproduced(_EstimatorFixture):
    """The window-3 numbers, before anything is repriced."""

    def test_the_boot_pricing_really_does_yield_7004(self):
        # If this drifts, every other number in this file is about a
        # different deployment and the specimen no longer applies.
        self.assertEqual(break_even_tokens(SEED_C, TP_TOK_S, PP_TOK_S), N_BOOT)

    def test_the_specimen_ceiling_is_reproduced_exactly(self):
        # "> N=7004 but <= 12608 with 4 req decoding", and its two siblings
        # at 2 and 5 decoding reqs, all from the same boot.
        cfg = _cfg(decode_contention=1.0)
        self.assertEqual(effective_flip_threshold(cfg, 2), 11674)
        self.assertEqual(effective_flip_threshold(cfg, 4), 12608)
        self.assertEqual(effective_flip_threshold(cfg, 5), 12841)


class TestTheBarFollowsTheMeasurement(_EstimatorFixture):
    """THE #819 ACTUATOR. This is what #777 declined to build."""

    def test_an_uncalibrated_estimator_leaves_the_bar_byte_identical(self):
        # Cold start: no measurement, so the seed stands and a deployment
        # that never flips behaves exactly as before.
        cfg = _cfg()
        self.assertEqual(live_flip_tokens(cfg), N_BOOT)
        self.assertEqual(effective_flip_threshold(cfg, 0), N_BOOT)

    def test_each_measured_rank_price_reprices_the_bar(self):
        # The three #777 lines, each repricing N to the value that line
        # itself computed. Today the bar is 7004 for all three.
        cfg = _cfg()
        for seconds, expected_n in MEASURED:
            with self.subTest(seam_s=seconds):
                self._measure(seconds)
                self.assertEqual(live_flip_tokens(cfg), expected_n)

    def test_the_ceiling_moves_with_the_repriced_bar(self):
        # The scaled ceiling is derived from N, so repricing N must carry
        # the whole ladder -- otherwise the lower bar and the upper band
        # would be priced off different seams.
        cfg = _cfg(decode_contention=1.0)
        # Uncalibrated, the ladder is the specimen's own, to the token.
        self.assertEqual(
            [effective_flip_threshold(cfg, b) for b in (2, 4, 5)],
            [11674, 12608, 12841],
        )
        self._measure(4.62869)
        # The whole ladder reprices, not just its floor. This is the term
        # that stayed frozen when only N was repriced: the differential
        # solve derives from C directly, so C had to move too.
        self.assertEqual(
            [effective_flip_threshold(cfg, b) for b in (2, 4, 5)],
            [16886, 18236, 18574],
        )
        self.assertGreater(
            effective_flip_threshold(cfg, 4),
            12608,
            "a seam measured MORE expensive than the seed must RAISE the bar",
        )

    def test_halving_the_measured_price_halves_the_bar(self):
        # THE COUPLING #834 EXISTS FOR. break-even is linear in the seam
        # cost, so a seam shrunk by SGLANG_SEAM_SHRINK must be visible in
        # the threshold without anyone editing a constant.
        cfg = _cfg()
        self._measure(8.0)
        expensive = live_flip_tokens(cfg)
        self._measure(4.0)
        cheap = live_flip_tokens(cfg)
        # Within one token: break-even is linear in C but returns a ROUNDED
        # integer, so 8.0 s -> 17511 and 4.0 s -> 8755 differ from an exact
        # halving by the rounding alone. Asserting exact equality here would
        # be pinning a rounding artefact, not the coupling.
        self.assertLessEqual(
            abs(expensive - 2 * cheap),
            1,
            f"halving the seam must halve the bar, got {expensive} -> {cheap}",
        )


class TestTheWholeLegIsPriced(_EstimatorFixture):
    """#819: the estimator is fed a LEG, not one step of a leg."""

    def _stats(self, **kw):
        base = dict(direction="tp_to_pp", movers_ms=2722.0, cutover_ms=50.0)
        base.update(kw)
        return base

    def test_the_leg_total_becomes_the_price(self):
        # #819's property, RESTATED PER LEG BY #856 rather than weakened. The
        # whole leg total (not just the refill step) still becomes the price
        # of THAT LEG -- which is what this test was always about. What
        # changed is that C is a round trip, so the OTHER leg is still on its
        # seed half until it too is measured, and the deliberate arithmetic
        # is 11.4901 + 3.2/2 = 13.0901. Asserting 11.4901 here would be
        # asserting that one leg is the whole round trip, which is exactly
        # the defect #856 removes.
        pp._FLIP_COST_ESTIMATOR = pp.RoundTripFlipCost(seed_s=SEED_C)
        pp.observe_flip_leg(self._stats(total_ms=11490.1))
        est = pp._FLIP_COST_ESTIMATOR
        self.assertAlmostEqual(est.leg("tp_to_pp").value(), 11.4901, places=4)
        self.assertAlmostEqual(est.value(), 11.4901 + SEED_C / 2.0, places=4)

    def test_a_shrinking_cutover_lowers_the_bar(self):
        # THE #834 COUPLING, end to end. Same flip, same movers, only the
        # cutover shrinks -- the term SGLANG_SEAM_SHRINK moves. W13 measured
        # that term at 3449 ms with HiCache on and 50 ms with it off, against
        # movers of ~2722 ms, so these are the shape of real numbers.
        cfg = _cfg()
        pp._FLIP_COST_ESTIMATOR = pp.RoundTripFlipCost(seed_s=SEED_C)
        pp.observe_flip_leg(self._stats(cutover_ms=3449.0, total_ms=6171.0))
        wide_seam = live_flip_tokens(cfg)
        pp._FLIP_COST_ESTIMATOR = pp.RoundTripFlipCost(seed_s=SEED_C)
        pp.observe_flip_leg(self._stats(cutover_ms=50.0, total_ms=2772.0))
        shrunk_seam = live_flip_tokens(cfg)
        self.assertLess(
            shrunk_seam,
            wide_seam,
            "shrinking the cutover must lower the bar; with only the refill "
            "leg fed, the cutover was not in the price at all",
        )

    def test_a_leg_without_a_total_is_not_a_reading(self):
        # Absent or malformed stats must leave the estimate untouched rather
        # than clamp a zero into it -- the module's rule for a non-reading.
        pp._FLIP_COST_ESTIMATOR = pp.RoundTripFlipCost(seed_s=SEED_C)
        for bad in (None, {}, {"total_ms": None}, {"total_ms": "n/a"}, 17.0):
            with self.subTest(stats=bad):
                pp.observe_flip_leg(bad)
                self.assertFalse(pp._FLIP_COST_ESTIMATOR.calibrated)


class TestTheSeedIsAPriorAndNotAPin(_EstimatorFixture):
    """#770 family: no magic pin survives a measurement."""

    def test_a_measurement_supersedes_the_seed_outright(self):
        cfg = _cfg()
        self.assertEqual(live_flip_tokens(cfg), N_BOOT)
        self._measure(4.62869)
        self.assertNotEqual(
            live_flip_tokens(cfg),
            N_BOOT,
            "the seed must not survive a measurement that contradicts it",
        )

    def test_an_operator_pin_is_never_overridden(self):
        # An explicitly set N is an ASSERTION, not a derivation, and the
        # estimator must not quietly move it.
        pp.note_flip_tokens_pricing(N_BOOT, TP_TOK_S, PP_TOK_S, True)
        cfg = _cfg()
        self._measure(4.62869)
        self.assertEqual(live_flip_tokens(cfg), N_BOOT)


class TestTheThrashGuardSurvives(_EstimatorFixture):
    """#759/#677 economics must not break backwards under repricing."""

    def test_a_cheap_seam_can_never_collapse_the_bar_to_zero(self):
        # A near-zero reading is #748's churn signature from the other end:
        # every pending prefill would fund a flip. MIN_ESTIMATE_S floors it.
        cfg = _cfg()
        self._measure(pp.FlipCostEstimator.MIN_ESTIMATE_S)
        self.assertGreater(live_flip_tokens(cfg), 0)

    def test_a_small_backlog_still_holds_after_repricing(self):
        # The specimen backlog (9962 tok, 4 req decoding) sat UNDER the
        # ceiling and was held. A repricing that made the seam dearer must
        # keep holding it -- the guard may tighten, never loosen.
        cfg = _cfg(decode_contention=1.0)
        self._measure(4.62869)
        self.assertLess(9962, effective_flip_threshold(cfg, 4))


class TestTheDecisionLineReadsOnce(_EstimatorFixture):
    """#713 rule: verdict and its inputs in ONE reading, and they must agree."""

    def _hold_reason(self, pending, running):
        cfg = _cfg(decode_contention=1.0, min_dwell_s=0.0)
        state = PhasePolicyState()
        d = decide(
            cfg,
            state,
            PhasePolicyInputs(
                phase=PHASE_TP,
                pending_prefill_tokens=pending,
                running_bs=running,
                now=1000.0,
            ),
        )
        self.assertIsNone(d.direction, f"expected a hold, got {d.reason}")
        return d.reason

    def test_the_specimen_line_is_reproduced_before_repricing(self):
        # The window-3 line, verbatim in its load-bearing parts.
        reason = self._hold_reason(9962, 4)
        self.assertIn("pending prefill 9962 tok > N=7004 but <= 12608", reason)
        self.assertIn("4 req decoding", reason)
        self.assertIn("too short for the round trip", reason)

    def test_the_line_reports_the_bar_it_actually_applied(self):
        # THE FAILURE THIS PREVENTS: a repriced bar printed against the frozen
        # seed, so the log says 7004 while the policy compared 10131. The
        # number in the text and the number in the comparison are one value.
        self._measure(4.62869)
        reason = self._hold_reason(17000, 4)
        self.assertIn("N=10131", reason)
        self.assertIn("<= 18236", reason)
        self.assertNotIn("N=7004", reason)

    def test_the_line_names_its_provenance(self):
        # Seed and measurement are distinguishable without reading the code.
        self.assertIn("seed", self._hold_reason(9962, 4))
        self._measure(4.62869)
        self.assertIn("measured", self._hold_reason(17000, 4))


class TestHysteresisIsUnchanged(_EstimatorFixture):
    """#759/#677: repricing must not weaken the anti-thrash guards."""

    def test_min_dwell_still_blocks_a_second_flip_after_repricing(self):
        # The dwell is a TIMER and owes nothing to the price. A repricing
        # that made flips look cheap must not buy a second flip inside it.
        self._measure(0.5)  # a cheap seam: the bar drops a long way
        cfg = _cfg(min_dwell_s=10.0)
        state = PhasePolicyState()
        state.last_flip_at = 1000.0
        d = decide(
            cfg,
            state,
            PhasePolicyInputs(
                phase=PHASE_TP,
                pending_prefill_tokens=10**6,
                running_bs=2,
                now=1001.0,
            ),
        )
        self.assertIsNone(d.direction)
        self.assertIn("min dwell", d.reason)


if __name__ == "__main__":
    unittest.main()
