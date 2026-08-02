"""The adaptive chain planner, and the tipping points between its modes (#451).

Falsifier-first. Every mode boundary is pinned by a *pair* of runs over the
same synthetic frontier table: one where the boundary has not been crossed and
one where it has, with nothing between them but the perturbation. A test that
only asserted the mode it expected would pass just as well against a planner
that always returned that mode, so each boundary test asserts the flip in both
directions and is therefore able to fail.

The tables here are synthetic and anchored on the real P1 records in
``docs/dev/TASK_333_M2_MEASUREMENTS.md`` -- the 5090 column is the measured
one, the 3080 columns use the per-stage ratios measured in §9.4, and the
decode and encode rows are invented because the shipped P1 report has none.
That last point is not a detail: it is exactly why the real table is used in
:class:`TheShippedMeasurementIsNotEnoughTest` to show what the planner does
when a stage was never measured, which is refuse rather than guess.

**BOOT-PENDING.** These tipping points are correct for the 4.6 / fp32-parity
era table. The user's operating point is the fp16/bf16 TensorRT engine and
RIFE 4.26, which nobody has measured, so the *positions* of the boundaries
will move. The planner's behaviour at a boundary will not, which is what these
tests pin.

Everything is CPU. ``plan_job`` and the whole policy are pure arithmetic.
"""

import unittest
from fractions import Fraction
from pathlib import Path

from sglang.srt.planner.cost_model import Provenance
from sglang.srt.video_enhance.chain_policy import (
    REQUIRES_FRAME_DECIMATION,
    REQUIRES_SCALED_DECODE,
    ChainMode,
    ChainPolicyError,
    PolicyInputs,
    PolicyRequest,
    SourceProbe,
    candidate_chains,
    choose_chain,
    require_chain,
    sr_entry_points,
)
from sglang.srt.video_enhance.frame_math import MIB, Resolution
from sglang.srt.video_enhance.tenant import TenantConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


#: The 5090 column. SR, resize and RIFE are the measured P1 numbers
#: (docs/dev/measurements/333-m2/p1_5090.json and p1_rife_5090.json); the
#: 1920x1080 resize row, both decode rows and both encode rows are synthetic,
#: because the shipped report has no such measurement.
BASE_MS = {
    ("decode", "960x540"): 0.8,
    ("decode", "1920x1080"): 2.0,
    ("decode", "3840x2160"): 5.0,
    ("sr", "960x540"): 35.19,
    ("sr", "1280x720"): 64.53,
    ("sr", "1920x1080"): 146.29,
    ("resize", "1920x1080"): 1.6,
    ("resize", "3840x2160"): 6.27,
    ("resize", "5120x2880"): 13.57,
    ("resize", "7680x4320"): 24.37,
    ("rife", "1920x1080"): 5.98,
    ("rife", "3840x2160"): 20.68,
    ("encode", "1920x1080"): 1.56,
    ("encode", "3840x2160"): 3.10,
}

#: §9.4 measured the 3080 at 2.55x the 5090 on SR. One ratio for every stage
#: keeps the synthetic table simple; the ratio per stage does not matter to a
#: tipping point, only the totals do.
CARD_RATIO = {"5090": 1.0, "3080a": 2.55, "3080b": 2.55}

#: 1185.4 MiB, the measured P4 value for a 1080p pair at scale 1.0. A 4K pair
#: measured 4740.7 MiB; the tenant config carries one scalar and every
#: candidate here interpolates at the same target, so one value is exact for
#: this comparison -- see the module docstring of ``chain_policy``.
RIFE_PAIR_BYTES = int(1185.4 * MIB)


def frontier(*, scale=None, override=None, cards=CARD_RATIO):
    """A probe-sample list in ``ProbeReport.samples`` shape."""
    rows = []
    for card, ratio in cards.items():
        for (stage, resolution), ms in BASE_MS.items():
            value = (override or {}).get((stage, resolution), ms) * ratio
            value *= (scale or {}).get(stage, 1.0)
            rows.append(
                {
                    "post": "P1",
                    "stage": stage,
                    "card": card,
                    "resolution": resolution,
                    "dtype": "fp16",
                    "options": {},
                    "ms_per_frame": value,
                    "ms_stdev": 0.0,
                    "iterations": 10,
                }
            )
    return rows


def inputs(rows=None, **kwargs):
    return PolicyInputs.from_samples(
        frontier() if rows is None else rows,
        budgets_mib={"5090": 30000, "3080a": 18000, "3080b": 18000},
        **kwargs,
    )


CONFIG = TenantConfig(budget_mib=20000, rife_measured_bytes_per_pair=RIFE_PAIR_BYTES)

#: The user's target scenario: 1080p at 25 fps in, 2160p at 50 fps out.
SOURCE_1080P25 = SourceProbe(Resolution(1920, 1080), Fraction(25), duration_s=600.0)
TARGET_4K50 = dict(target=Resolution(3840, 2160), target_frame_rate=Fraction(50))

#: 4K at 24 fps in, 4K at 48 fps out. SR has nothing to add.
SOURCE_4K24 = SourceProbe(Resolution(3840, 2160), Fraction(24), duration_s=600.0)
TARGET_4K48 = dict(target=Resolution(3840, 2160), target_frame_rate=Fraction(48))


def decide(probe, rows, **request_kwargs):
    request = PolicyRequest(**request_kwargs)
    return choose_chain(probe, request, inputs(rows), CONFIG)


def card_price(priced, card="5090"):
    """One named card's column. ``per_card`` is in the rate table's order,
    which is alphabetical, so indexing it by position picks a 3080."""
    return next(c for c in priced.per_card if c.card == card)


def stage_price(priced, stage, card="5090"):
    return next(s for s in card_price(priced, card).stages if s.stage == stage)


class FullWinsWheneverItFitsTest(CustomTestCase):
    """Tier 0 is not a preference the caller can express; it is the ranking."""

    def test_full_is_chosen_when_the_frontier_carries_it(self):
        fast = frontier(scale={k: 0.1 for k in ("sr", "resize", "rife", "encode", "decode")})
        decision = decide(SOURCE_1080P25, fast, **TARGET_4K50)
        self.assertTrue(decision.feasible)
        self.assertIs(decision.mode, ChainMode.FULL)
        self.assertEqual(decision.request.source, Resolution(1920, 1080))
        self.assertTrue(decision.request.enable_sr)

    def test_full_wins_even_when_a_cheaper_shape_has_more_headroom(self):
        """The falsifier for "the planner just maximises throughput".

        On the fast table both ``full`` and the pre-downscale ladder are
        feasible and the ladder is between two and eleven times faster. A
        planner ranking by headroom would take the fastest; this one takes
        the one that keeps every source pixel.
        """
        fast = frontier(scale={k: 0.1 for k in ("sr", "resize", "rife", "encode", "decode")})
        decision = decide(SOURCE_1080P25, fast, **TARGET_4K50)
        chosen = decision.price
        faster = [
            p
            for p in decision.considered
            if p.feasible and p.headroom > chosen.headroom
        ]
        self.assertIs(decision.mode, ChainMode.FULL)
        self.assertTrue(
            faster, "the test is vacuous unless a faster feasible shape existed"
        )
        for entry in faster:
            self.assertIs(entry.candidate.mode, ChainMode.PRE_DOWNSCALE)


class PreDownscaleBoundaryTest(CustomTestCase):
    """Which SR entry point the frontier can carry, and that it is solved."""

    def test_the_ladder_is_the_measured_sr_resolutions(self):
        ladder = sr_entry_points(inputs())
        self.assertEqual(
            ladder,
            (
                Resolution(1920, 1080),
                Resolution(1280, 720),
                Resolution(960, 540),
            ),
        )

    def test_candidates_are_only_entry_points_sr_can_still_reach_the_target_from(self):
        """A x4 model must land at or above the target, or SR is pointless."""
        request = PolicyRequest(**TARGET_4K50)
        entries = {
            c.request.source
            for c in candidate_chains(SOURCE_1080P25, request, inputs())
            if c.mode is ChainMode.PRE_DOWNSCALE
        }
        # 960x540 x4 is exactly 3840x2160; 1280x720 x4 overshoots and resizes
        # back down. Anything smaller than 540p could not reach 4K at all and
        # is not offered.
        self.assertEqual(entries, {Resolution(960, 540), Resolution(1280, 720)})

    def test_the_boundary_moves_with_the_table_in_both_directions(self):
        """Can-fail proof: perturb only the 720p SR row and watch the entry move.

        With the measured 64.53 ms the 720p entry is short of the input rate
        and the planner drops to 540p, discarding more detail. Make that one
        row fast enough and the planner climbs back up the ladder -- nothing
        else in the table changes.
        """
        slow = decide(SOURCE_1080P25, frontier(), **TARGET_4K50)
        self.assertIs(slow.mode, ChainMode.PRE_DOWNSCALE)
        self.assertEqual(slow.request.source, Resolution(960, 540))

        fast = decide(
            SOURCE_1080P25,
            frontier(override={("sr", "1280x720"): 20.0}),
            **TARGET_4K50,
        )
        self.assertIs(fast.mode, ChainMode.PRE_DOWNSCALE)
        self.assertEqual(fast.request.source, Resolution(1280, 720))

    def test_a_pre_downscale_names_the_executor_capability_it_needs(self):
        decision = decide(SOURCE_1080P25, frontier(), **TARGET_4K50)
        self.assertEqual(decision.requires, (REQUIRES_SCALED_DECODE,))
        self.assertTrue(decision.feasible)
        self.assertFalse(decision.runnable)

    def test_require_runnable_excludes_it_and_the_refusal_says_why(self):
        decision = decide(
            SOURCE_1080P25, frontier(), require_runnable=True, **TARGET_4K50
        )
        self.assertFalse(decision.feasible)
        self.assertIn(REQUIRES_SCALED_DECODE, decision.reason)
        self.assertIn("require_runnable", decision.reason)

    def test_the_decoder_is_priced_at_the_source_size_not_the_entry_size(self):
        """A pre-downscale saves SR work, not decode work.

        The falsifier for the easy mistake: the chain's ``source`` is the
        entry resolution, so pricing every stage off the chain would credit
        the mode with a decode that runs at 540p when the decoder is still
        reading a 1080p stream.
        """
        decision = decide(SOURCE_1080P25, frontier(), **TARGET_4K50)
        decode = stage_price(decision.price, "decode")
        self.assertEqual(decode.resolution, "1920x1080")
        self.assertAlmostEqual(decode.ms, BASE_MS[("decode", "1920x1080")], places=6)


class RifeOnlyBoundaryTest(CustomTestCase):
    """Skipping SR is geometry, not economy."""

    def test_a_source_at_the_target_takes_the_rife_only_chain(self):
        decision = decide(SOURCE_4K24, frontier(), **TARGET_4K48)
        self.assertTrue(decision.feasible)
        self.assertIs(decision.mode, ChainMode.RIFE_ONLY)
        self.assertFalse(decision.request.enable_sr)

    def test_one_step_below_the_target_brings_sr_back(self):
        """Can-fail proof for the rife-only boundary.

        The source shrinks by one macroblock row and column and the same
        table now produces an SR chain, because a Lanczos upscale cannot
        stand in for what SR recovers.
        """
        just_below = SourceProbe(
            Resolution(3840 - 2, 2160 - 2), Fraction(24), duration_s=600.0
        )
        modes = {
            c.mode
            for c in candidate_chains(
                just_below, PolicyRequest(**TARGET_4K48), inputs()
            )
        }
        self.assertNotIn(ChainMode.RIFE_ONLY, modes)
        self.assertIn(ChainMode.FULL, modes)

    def test_a_source_above_the_target_still_skips_sr(self):
        source = SourceProbe(Resolution(3840, 2160), Fraction(24), duration_s=600.0)
        modes = {
            c.mode
            for c in candidate_chains(
                source,
                PolicyRequest(
                    target=Resolution(1920, 1080), target_frame_rate=Fraction(48)
                ),
                inputs(),
            )
        }
        self.assertEqual(modes, {ChainMode.RIFE_ONLY})


class DecimationBoundaryTest(CustomTestCase):
    """The one mode that discards real input, and its opt-in.

    The source here is 540p at 50 fps to 4K at 50 fps, which is the shape the
    user's own example describes: the x4 model lands exactly on the target, so
    there is no pre-downscale ladder below it -- 540p is already the smallest
    entry point that reaches 4K -- and dropping frames is the only lever left.
    On this table the full rate misses the gate by 4.4 fps and halving the
    input clears it, which is what makes the boundary a boundary rather than a
    preference.
    """

    SOURCE = SourceProbe(Resolution(960, 540), Fraction(50), duration_s=600.0)
    TARGET = dict(target=Resolution(3840, 2160), target_frame_rate=Fraction(50))

    def decide(self, **kwargs):
        return decide(self.SOURCE, frontier(), **{**self.TARGET, **kwargs})

    def test_the_full_rate_misses_the_gate_on_this_table(self):
        """Without this the opt-in test below would prove nothing."""
        decision = self.decide()
        self.assertFalse(decision.feasible)
        full = next(
            p for p in decision.considered if p.candidate.mode is ChainMode.FULL
        )
        self.assertLess(full.aggregate_chain_fps, full.required_chain_fps)
        self.assertEqual(
            [p.candidate.mode for p in decision.considered], [ChainMode.FULL]
        )

    def test_it_is_never_reached_without_the_opt_in(self):
        decision = self.decide()
        self.assertFalse(decision.feasible)
        self.assertIsNone(decision.mode)
        self.assertIn("must be requested by name", decision.reason)

    def test_the_same_table_with_the_opt_in_finds_a_chain(self):
        """Can-fail proof: the only difference is ``allow_decimation``."""
        decision = self.decide(allow_decimation=True, max_decimation=3)
        self.assertTrue(decision.feasible)
        self.assertIs(decision.mode, ChainMode.DECIMATE_RESYNTH)
        self.assertEqual(decision.price.candidate.decimation, 2)

    def test_it_names_the_decode_stride_the_executor_does_not_have(self):
        decision = self.decide(allow_decimation=True, max_decimation=3)
        self.assertIn(REQUIRES_FRAME_DECIMATION, decision.requires)
        self.assertFalse(decision.runnable)
        excluded = self.decide(
            allow_decimation=True, max_decimation=3, require_runnable=True
        )
        self.assertFalse(excluded.feasible)
        self.assertIn(REQUIRES_FRAME_DECIMATION, excluded.reason)

    def test_the_smallest_decimation_that_fits_is_the_one_chosen(self):
        decision = self.decide(allow_decimation=True, max_decimation=4)
        chosen = decision.price.candidate
        feasible = sorted(
            p.candidate.decimation
            for p in decision.considered
            if p.candidate.mode is ChainMode.DECIMATE_RESYNTH and p.feasible
        )
        self.assertIs(decision.mode, ChainMode.DECIMATE_RESYNTH)
        # Several factors clear the gate here, and the coarser ones clear it by
        # more. Picking the smallest is therefore a quality decision the
        # ranking makes, not an artefact of only one option fitting.
        self.assertGreater(len(feasible), 1)
        self.assertEqual(chosen.decimation, min(feasible))

    def test_the_response_is_honest_about_what_was_thrown_away(self):
        decision = self.decide(allow_decimation=True, max_decimation=3)
        chosen = decision.price.candidate
        payload = decision.as_dict()["chosen"]
        self.assertGreater(payload["discarded_input_fraction"], 0.0)
        self.assertGreater(payload["synthetic_output_fraction"], 0.0)
        self.assertIn("discarded", payload["quality_note"])
        self.assertIn("every 2nd frame", payload["quality_note"])
        self.assertAlmostEqual(
            payload["discarded_input_fraction"],
            (chosen.decimation - 1) / chosen.decimation,
            places=4,
        )

    def test_the_decoder_is_charged_for_the_frames_that_are_thrown_away(self):
        """A decoder cannot skip frames; decimation must not pretend it can.

        Without this the mode would look cheaper than it is by exactly the
        decode column, and the decode column is what a high input rate makes
        expensive.
        """
        decision = self.decide(allow_decimation=True, max_decimation=3)
        decode = stage_price(decision.price, "decode")
        self.assertEqual(decode.invocations, float(decision.price.candidate.decimation))

    def test_interpolating_further_costs_more_per_pair(self):
        """A P1 RIFE cell is per interpolated frame, not per pair invocation.

        ``shard_plan.stage_stream_factors`` returns one invocation per pair at
        any multiplier, which is right at x2 and would make every decimated
        candidate look free at the RIFE stage. The planner charges
        ``arity_out`` instead, so raising the multiplier raises the bill.
        """
        decision = self.decide(allow_decimation=True, max_decimation=3)
        by_multiplier = {}
        for entry in decision.considered:
            if entry.candidate.mode is not ChainMode.DECIMATE_RESYNTH:
                continue
            rife = stage_price(entry, "rife")
            by_multiplier[entry.candidate.request.fps_multiplier] = rife.ms
        self.assertEqual(sorted(by_multiplier), [2, 3])
        self.assertLess(by_multiplier[2], by_multiplier[3])


class RefusalNamesTheNumbersTest(CustomTestCase):
    def test_no_chain_fits_and_every_candidate_is_accounted_for(self):
        glacial = frontier(scale={k: 40.0 for k in ("sr", "rife", "resize", "encode")})
        decision = decide(SOURCE_1080P25, glacial, **TARGET_4K50)
        self.assertFalse(decision.feasible)
        self.assertIsNone(decision.mode)
        self.assertIn("no chain reaches 3840x2160 at 50 fps", decision.reason)
        # Each candidate contributes a line carrying its own arithmetic.
        for entry in decision.considered:
            self.assertIn(entry.reason, decision.reason)
            self.assertIn("chain fps", entry.reason)
        self.assertIn("required", decision.reason)

    def test_require_chain_raises_the_same_text(self):
        glacial = frontier(scale={k: 40.0 for k in ("sr", "rife", "resize", "encode")})
        with self.assertRaises(ChainPolicyError) as caught:
            require_chain(
                SOURCE_1080P25,
                PolicyRequest(**TARGET_4K50),
                inputs(glacial),
                CONFIG,
            )
        self.assertIn("no chain reaches", str(caught.exception))

    def test_a_budget_that_cannot_hold_one_frame_excludes_the_card_by_name(self):
        tiny = PolicyInputs.from_samples(
            frontier(), budgets_mib={"5090": 30000, "3080a": 1200, "3080b": 1200}
        )
        decision = choose_chain(
            SOURCE_4K24, PolicyRequest(**TARGET_4K48), tiny, CONFIG
        )
        excluded = [c for c in decision.price.per_card if not c.admitted]
        self.assertEqual({c.card for c in excluded}, {"3080a", "3080b"})
        for card in excluded:
            self.assertIn("MiB", card.blocker)
        self.assertEqual(decision.cards, ("5090",))

    def test_a_target_rate_that_is_not_a_whole_multiple_is_refused(self):
        decision = decide(
            SOURCE_1080P25,
            frontier(),
            target=Resolution(3840, 2160),
            target_frame_rate=Fraction(60),
        )
        self.assertFalse(decision.feasible)
        self.assertIn("whole multiple", decision.reason)
        self.assertIn("25", decision.reason)

    def test_a_shortfall_with_an_unknown_duration_is_not_waved_through(self):
        """The lead a shortfall needs scales with the duration.

        With no duration the computed lead is 0 s for any deficit, which a
        naive ``lead <= budget`` test would read as "fits". It does not.
        """
        no_duration = SourceProbe(Resolution(1920, 1080), Fraction(25))
        decision = choose_chain(
            no_duration,
            PolicyRequest(max_watch_ahead_s=120.0, **TARGET_4K50),
            inputs(),
            CONFIG,
        )
        full = next(
            p for p in decision.considered if p.candidate.mode is ChainMode.FULL
        )
        self.assertFalse(full.feasible)
        self.assertIn("declares no duration", full.reason)


class WatchAheadTest(CustomTestCase):
    def test_a_bounded_shortfall_is_absorbed_by_a_granted_lead(self):
        short = frontier(scale={k: 1.05 for k in ("sr", "resize", "rife", "encode")})
        strict = choose_chain(
            SOURCE_4K24, PolicyRequest(**TARGET_4K48), inputs(short), CONFIG
        )
        self.assertTrue(strict.feasible)  # 4K rife-only clears 24 fps outright

        # Now make it genuinely short and grant a lead. Same table both times.
        tight = frontier(scale={"rife": 2.5, "encode": 2.5, "decode": 2.5})
        without = choose_chain(
            SOURCE_4K24, PolicyRequest(**TARGET_4K48), inputs(tight), CONFIG
        )
        with_lead = choose_chain(
            SOURCE_4K24,
            PolicyRequest(max_watch_ahead_s=600.0, **TARGET_4K48),
            inputs(tight),
            CONFIG,
        )
        self.assertFalse(without.feasible)
        self.assertTrue(with_lead.feasible)
        self.assertGreater(with_lead.price.watch_ahead_s, 0.0)
        self.assertIn("lead", with_lead.reason)


class ProvenanceTest(CustomTestCase):
    """Measured, estimate, absent -- and never a fourth quiet case."""

    def test_a_missing_row_makes_the_candidate_unpriceable(self):
        rows = [r for r in frontier() if r["resolution"] != "3840x2160" or r["stage"] != "encode"]
        decision = decide(SOURCE_4K24, rows, **TARGET_4K48)
        self.assertFalse(decision.feasible)
        self.assertIs(decision.provenance, Provenance.ABSENT)
        self.assertIn("unpriceable", decision.reason)
        self.assertIn("encode", decision.reason)

    def test_the_same_gap_is_priced_and_labelled_when_estimates_are_allowed(self):
        """Can-fail proof for the estimate flag: one boolean, two outcomes."""
        rows = [r for r in frontier() if r["resolution"] != "3840x2160" or r["stage"] != "encode"]
        decision = decide(SOURCE_4K24, rows, allow_estimates=True, **TARGET_4K48)
        self.assertTrue(decision.feasible)
        self.assertIs(decision.provenance, Provenance.ESTIMATE)
        self.assertIn("extrapolation", decision.reason)
        encode = stage_price(decision.price, "encode")
        self.assertIs(encode.provenance, Provenance.ESTIMATE)
        self.assertIn("1920x1080", encode.source)

    def test_a_stage_with_no_row_at_all_stays_absent_even_with_estimates(self):
        """An extrapolation needs a measured row to extrapolate *from*."""
        rows = [r for r in frontier() if r["stage"] != "encode"]
        decision = decide(SOURCE_4K24, rows, allow_estimates=True, **TARGET_4K48)
        self.assertFalse(decision.feasible)
        self.assertIs(decision.provenance, Provenance.ABSENT)

    def test_the_extrapolation_is_linear_in_pixels(self):
        rows = [r for r in frontier() if r["resolution"] != "1920x1080" or r["stage"] != "sr"]
        decision = decide(
            SOURCE_1080P25, rows, allow_estimates=True, **TARGET_4K50
        )
        full = next(
            p for p in decision.considered if p.candidate.mode is ChainMode.FULL
        )
        sr = stage_price(full, "sr")
        # Nearest measured row by pixel ratio is 1280x720 at 64.53 ms.
        expected = 64.53 * (1920 * 1080) / (1280 * 720)
        self.assertAlmostEqual(sr.ms, expected, places=6)
        self.assertIs(sr.provenance, Provenance.ESTIMATE)

    def test_the_two_colour_stages_are_reported_as_unpriced_not_as_free(self):
        decision = decide(SOURCE_4K24, frontier(), **TARGET_4K48)
        self.assertEqual(
            set(decision.price.unpriced_stages), {"color_to_rgb", "color_to_yuv"}
        )


class TheShippedMeasurementIsNotEnoughTest(CustomTestCase):
    """What the planner does with the real P1 report that exists today.

    The record has SR, resize and RIFE on one card and no decode or encode row
    at all. The honest answer to "how fast is the full chain" against it is
    therefore "nobody measured two of its stages", and that is the answer the
    planner gives -- including with ``allow_estimates``, because an
    extrapolation still needs a row of the same stage to start from.
    """

    REPORTS = Path(__file__).resolve().parents[3] / "docs/dev/measurements/333-m2"

    def test_the_real_report_loads_and_prices_nothing_end_to_end(self):
        if not self.REPORTS.is_dir():
            self.skipTest(f"no measurement records under {self.REPORTS}")
        real = PolicyInputs.from_probe_dir(self.REPORTS)
        self.assertIn("sr", real.rates.stages)
        self.assertNotIn("encode", real.rates.stages)
        decision = choose_chain(
            SOURCE_4K24, PolicyRequest(**TARGET_4K48), real, CONFIG
        )
        self.assertFalse(decision.feasible)
        self.assertIn("unpriceable", decision.reason)
        for missing in ("decode", "encode"):
            self.assertIn(missing, decision.reason)

    def test_estimates_do_not_rescue_a_stage_nobody_ever_measured(self):
        if not self.REPORTS.is_dir():
            self.skipTest(f"no measurement records under {self.REPORTS}")
        real = PolicyInputs.from_probe_dir(self.REPORTS)
        decision = choose_chain(
            SOURCE_4K24,
            PolicyRequest(allow_estimates=True, **TARGET_4K48),
            real,
            CONFIG,
        )
        self.assertFalse(decision.feasible)

    def test_an_absent_measurement_directory_refuses_rather_than_invents(self):
        empty = PolicyInputs.from_probe_dir(None)
        decision = choose_chain(
            SOURCE_4K24, PolicyRequest(**TARGET_4K48), empty, CONFIG
        )
        self.assertFalse(decision.feasible)
        self.assertIn("no card can run this chain", decision.reason)


class TheRifeFootprintGateStillAppliesTest(CustomTestCase):
    def test_a_tenant_without_a_p4_value_cannot_admit_any_rife_chain(self):
        unmeasured = TenantConfig(budget_mib=30000)
        decision = choose_chain(
            SOURCE_4K24, PolicyRequest(**TARGET_4K48), inputs(), unmeasured
        )
        self.assertFalse(decision.feasible)
        self.assertIn("measurement post P4", decision.reason)


class TheClientVisibleAnswerTest(CustomTestCase):
    def test_the_response_carries_the_mode_the_reason_and_the_provenance(self):
        decision = decide(SOURCE_4K24, frontier(), **TARGET_4K48)
        payload = decision.as_dict()
        self.assertEqual(payload["mode"], "rife_only")
        self.assertEqual(payload["provenance"], "measured")
        self.assertTrue(payload["feasible"])
        self.assertTrue(payload["runnable"])
        self.assertIn("chain fps", payload["reason"])
        self.assertEqual(payload["chosen"]["mode"], "rife_only")
        self.assertEqual(payload["cards"], ["3080a", "3080b", "5090"])
        self.assertTrue(payload["considered"])

    def test_a_refusal_serialises_with_every_candidate_it_looked_at(self):
        glacial = frontier(scale={k: 40.0 for k in ("sr", "rife", "resize", "encode")})
        payload = decide(SOURCE_1080P25, glacial, **TARGET_4K50).as_dict()
        self.assertFalse(payload["feasible"])
        self.assertIsNone(payload["mode"])
        self.assertEqual(len(payload["considered"]), 3)
        for entry in payload["considered"]:
            self.assertFalse(entry["feasible"])
            self.assertIn("cards", entry)


if __name__ == "__main__":
    unittest.main()
