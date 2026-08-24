"""#856(b): C is a ROUND TRIP, so it is priced from BOTH legs.

RED-FIRST, against the W25 specimen (boot_w25_0824_1125.log, pin 63c3c0dd00).

THE DEFECT. This module's own model says ``C = round-trip flip cost,
seconds``, and ``break_even_tokens`` refuses a bad premise by saying flipping
"never repays the {flip_cost_s}s round trip". ``observe_flip_leg`` fed ONE LEG
per sample -- and its own docstring computes the round trip it was not
feeding, "tp_to_pp 11490 + pp_to_tp 5681 = 17171 ms". Both directions went
into ONE EMA.

The two legs are not the same quantity. On the binding rank W25 measured
tp_to_pp 10466-13181 ms against pp_to_tp 5078-6545 ms. One EMA over both
converges to NEITHER: the policy spent 8.50 s while the legs were 11.6 and
6.4 and the round trip was 18.06 s. #819's own closing sentence is the rule
it broke one level up -- "a component and its container are different
quantities and an EMA fed both alternately converges to neither".

WHY THE ARITHMETIC IS NOT IN DOUBT. ``test_the_w25_blend_is_reproduced``
replays PP0's eleven real PHASE-FLIP DONE totals through a single estimator
and reproduces the four bars the boot actually printed (15853 / 18110 / 18464
/ 18614) to the token. That is what makes the blend a measured defect rather
than a reading of a log.

DIRECTION OF THE CORRECTION, STATED PLAINLY. Fixing this RAISES the bar
(C 8.50 -> 18.06 s, N 18614 -> ~39500), so it makes TP-stickiness on 16-20k
prompts MORE correct, not less. That is the honest consequence and it is not
softened here: the remedy for a bar that is too high is to make the seam
cheaper, not to keep under-pricing it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers import phase_policy as pp
from sglang.srt.managers.phase_policy import (
    PhasePolicyConfig,
    RoundTripFlipCost,
    break_even_tokens,
    live_flip_cost_s,
    live_flip_tokens,
)
from sglang.test.test_utils import CustomTestCase

SEED_C = 3.2
N_BOOT = 7004
TP_TOK_S = 1681.0
PP_TOK_S = 7245.5

#: PP0's eleven `PHASE-FLIP DONE ... in N ms` totals from
#: boot_w25_0824_1125.log, in epoch order. Odd epochs are pp_to_tp, even are
#: tp_to_pp -- the alternation is what the single EMA blended away.
W25_LEGS_MS = (
    ("pp_to_tp", 5077.9),
    ("tp_to_pp", 10466.4),
    ("pp_to_tp", 5211.0),
    ("tp_to_pp", 10567.9),
    ("pp_to_tp", 6537.0),
    ("tp_to_pp", 13180.9),
    ("pp_to_tp", 6523.8),
    ("tp_to_pp", 11513.0),
    ("pp_to_tp", 6545.4),
    ("tp_to_pp", 11646.8),
    ("pp_to_tp", 6416.0),
)

#: The bars boot_w25_0824_1125.log actually printed, in order.
W25_PRINTED_BARS = (15853, 18110, 18464, 18614)


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


class _Fixture(CustomTestCase):
    """Save/restore the process-global estimator and the boot pricing."""

    def setUp(self):
        self._saved = (
            pp._FLIP_COST_ESTIMATOR,
            pp._FLIP_TOKENS_AT_BOOT,
            pp._FLIP_TOKENS_PRICING,
            pp._FLIP_TOKENS_STALE_SAID,
        )
        pp._FLIP_COST_ESTIMATOR = RoundTripFlipCost(seed_s=SEED_C)
        pp.note_flip_tokens_pricing(N_BOOT, TP_TOK_S, PP_TOK_S, False)

    def tearDown(self):
        (
            pp._FLIP_COST_ESTIMATOR,
            pp._FLIP_TOKENS_AT_BOOT,
            pp._FLIP_TOKENS_PRICING,
            pp._FLIP_TOKENS_STALE_SAID,
        ) = self._saved

    @staticmethod
    def _leg(direction, total_ms):
        return {"direction": direction, "total_ms": total_ms}


class TestTheBlendWasReal(_Fixture):
    """The defect, reproduced from the boot's own numbers before it is fixed."""

    def test_the_w25_blend_is_reproduced(self):
        # A single EMA over the alternating legs, at the shipped ALPHA. This
        # is the OLD behaviour, modelled directly, and it must reproduce the
        # bars the boot printed -- otherwise this file is about a defect that
        # was never on metal.
        ema = None
        bars = []
        for _direction, ms in W25_LEGS_MS:
            x = ms / 1000.0
            alpha = pp.FlipCostEstimator.ALPHA
            ema = x if ema is None else ema + alpha * (x - ema)
            bars.append(break_even_tokens(ema, TP_TOK_S, PP_TOK_S))
        for printed in W25_PRINTED_BARS:
            self.assertIn(
                printed,
                bars,
                f"the boot printed N={printed}; a single EMA over the legs "
                f"must reproduce it or this specimen is not what it claims",
            )
        # And the blend lands between the legs, belonging to neither.
        self.assertAlmostEqual(ema, 8.5041, places=3)

    def test_the_blend_converged_to_neither_leg(self):
        tp = [ms for d, ms in W25_LEGS_MS if d == "tp_to_pp"]
        ptp = [ms for d, ms in W25_LEGS_MS if d == "pp_to_tp"]
        blended_s = 8.5041
        self.assertLess(blended_s * 1000.0, min(tp))
        self.assertGreater(blended_s * 1000.0, max(ptp))


class TestTheRoundTripIsPriced(_Fixture):
    """The new contract: C is the SUM of the two legs."""

    def test_two_legs_price_their_sum(self):
        pp.observe_flip_leg(self._leg("tp_to_pp", 11646.8))
        pp.observe_flip_leg(self._leg("pp_to_tp", 6416.0))
        self.assertAlmostEqual(live_flip_cost_s(_cfg()), 18.0628, places=3)

    def test_the_w25_round_trip_is_not_the_blend(self):
        for direction, ms in W25_LEGS_MS:
            pp.observe_flip_leg(self._leg(direction, ms))
        c = live_flip_cost_s(_cfg())
        self.assertGreater(
            c,
            15.0,
            "the round trip of an 11.6s and a 6.4s leg cannot be priced at "
            "the 8.50s a single blended EMA produced",
        )
        # And the bar follows it, well above the 18614 the blend produced.
        self.assertGreater(live_flip_tokens(_cfg()), 30000)

    def test_each_leg_keeps_its_own_estimate(self):
        # THE SEPARATION IS THE FIX. Feeding one direction repeatedly must
        # not drag the other leg with it.
        for _ in range(6):
            pp.observe_flip_leg(self._leg("tp_to_pp", 11646.8))
        est = pp._FLIP_COST_ESTIMATOR
        self.assertAlmostEqual(est.leg("tp_to_pp").value(), 11.6468, places=3)
        self.assertFalse(est.leg("pp_to_tp").calibrated)
        # The unmeasured leg still contributes its seed HALF, so C is the
        # measured leg plus half the round-trip seed -- never the seed twice.
        self.assertAlmostEqual(est.value(), 11.6468 + SEED_C / 2.0, places=3)


class TestProvenanceNamesTheHalfMeasuredState(_Fixture):
    """A boolean cannot say "one leg". Three states, three words."""

    def test_no_leg_measured_reads_seed(self):
        self.assertEqual(pp.flip_cost_provenance(), "seed")

    def test_one_leg_measured_reads_half(self):
        pp.observe_flip_leg(self._leg("tp_to_pp", 11646.8))
        self.assertEqual(pp.flip_cost_provenance(), "half-measured (tp_to_pp only)")

    def test_both_legs_measured_reads_measured(self):
        pp.observe_flip_leg(self._leg("tp_to_pp", 11646.8))
        pp.observe_flip_leg(self._leg("pp_to_tp", 6416.0))
        self.assertEqual(pp.flip_cost_provenance(), "measured")

    def test_the_half_measured_state_is_not_printed_as_measured(self):
        # THE CAN-FAIL DIRECTION for this whole section: a provenance that
        # collapsed the middle state into "measured" would satisfy every
        # other assertion here and restore exactly the ambiguity #856 removes.
        pp.observe_flip_leg(self._leg("pp_to_tp", 6416.0))
        self.assertNotEqual(pp.flip_cost_provenance(), "measured")
        self.assertIn("pp_to_tp", pp.flip_cost_provenance())


class TestTheDetectorGateIsReconciled(_Fixture):
    """A third state was introduced, so its consumers must be reconciled.

    The #838 economy detector refuses to question a bar whose price is a seed,
    on the stated ground that "an assumption is not the policy's own claim".
    A HALF-measured round trip is still half assumption. Leaving the detector
    on the two-state boolean would have let one measured leg promote a
    half-seed price to evidence -- hardening one module and quietly changing
    another module's premise, which is the miss this build keeps naming.
    """

    def test_one_leg_does_not_make_the_price_evidence(self):
        pp.observe_flip_leg(self._leg("tp_to_pp", 11646.8))
        self.assertTrue(pp.flip_cost_measured(), "repricing still engages")
        self.assertFalse(
            pp.flip_cost_fully_measured(),
            "one measured leg is not a measured ROUND TRIP",
        )

    def test_both_legs_make_the_price_evidence(self):
        pp.observe_flip_leg(self._leg("tp_to_pp", 11646.8))
        pp.observe_flip_leg(self._leg("pp_to_tp", 6416.0))
        self.assertTrue(pp.flip_cost_fully_measured())

    def test_the_seed_is_never_evidence(self):
        self.assertFalse(pp.flip_cost_fully_measured())

    def test_the_two_predicates_are_not_the_same_function(self):
        # THE CAN-FAIL DIRECTION: if `fully_measured` were an alias for
        # `measured`, every assertion above except this one would still pass.
        pp.observe_flip_leg(self._leg("tp_to_pp", 11646.8))
        self.assertNotEqual(pp.flip_cost_measured(), pp.flip_cost_fully_measured())


class TestTheUndirectedReadingIsARoundTrip(_Fixture):
    """A sample with no direction is not filed under a guessed leg."""

    def test_an_undirected_sample_prices_itself(self):
        # Split evenly across the legs, so value() returns exactly what was
        # observed -- the pre-#856 single-number semantics, preserved for
        # every caller that really did measure a round trip.
        pp.observe_flip_cost(17.171)
        self.assertAlmostEqual(pp._FLIP_COST_ESTIMATOR.value(), 17.171, places=4)

    def test_an_unknown_direction_is_refused_not_filed(self):
        # THE CAN-FAIL DIRECTION: filing a leg this class does not know under
        # one it does would corrupt both estimates while looking calibrated.
        pp.observe_flip_cost(9.9, "pp_to_dcp")
        self.assertFalse(pp._FLIP_COST_ESTIMATOR.calibrated)
        self.assertEqual(pp.flip_cost_provenance(), "seed")

    def test_a_leg_without_a_total_is_still_not_a_reading(self):
        for bad in (None, {}, {"total_ms": None}, {"total_ms": "n/a"}, 17.0):
            with self.subTest(stats=bad):
                pp.observe_flip_leg(bad)
                self.assertFalse(pp._FLIP_COST_ESTIMATOR.calibrated)


class TestTheSeedPathIsUnchanged(_Fixture):
    """An uncalibrated round trip must value EXACTLY the round-trip seed."""

    def test_an_uncalibrated_instance_values_the_seed(self):
        self.assertAlmostEqual(RoundTripFlipCost(seed_s=SEED_C).value(), SEED_C)

    def test_the_seed_is_never_counted_twice(self):
        # The half-split is what makes this true; a per-leg full seed would
        # double C on every unmeasured boot and silently double the bar.
        self.assertAlmostEqual(RoundTripFlipCost(seed_s=9.0).value(), 9.0, places=6)

    def test_a_leg_still_tracks_a_regime_change_downward(self):
        # #677's property, per leg: the estimator is not a ratchet, so a
        # genuine seam improvement LOWERS the bar rather than latching high.
        for _ in range(4):
            pp.observe_flip_leg(self._leg("tp_to_pp", 12000.0))
        high = pp._FLIP_COST_ESTIMATOR.leg("tp_to_pp").value()
        for _ in range(4):
            pp.observe_flip_leg(self._leg("tp_to_pp", 3000.0))
        low = pp._FLIP_COST_ESTIMATOR.leg("tp_to_pp").value()
        self.assertLess(low, high / 2.0)


if __name__ == "__main__":
    unittest.main()
