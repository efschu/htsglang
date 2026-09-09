# SPDX-License-Identifier: Apache-2.0
"""#1286b: a POOL FLOOR under the objective, and the curve it is read from.

WHY THIS EXISTS, AND WHY ONLY NOW.  #1286 repriced every candidate against the
boot's own sizing formula (``pp_cut.pp_phase_pool`` over the runtime's own
budget-post list) and closed a +64.1 % over-pricing.  The repricing did not
move the makespan arm's CHOICE -- that is argued in #1286 F9 and asserted
there -- but it moved its PRICE, and that is what makes this slice necessary:
the cut ``--pp-solve-objective makespan`` ships now prices at **304,946**
tokens against the incumbent's **715,089**.  The speed arm buys TTFT with 57 %
of the capacity, and before this flag the only way to put a bound under that
number was to pin a layer cut by hand -- which is precisely the act the solver
exists to replace, and which #1236/#1240 spent two boots proving is where
un-priced layouts come from.

THE SHAPE IS A CONSTRAINT, NOT A SECOND OBJECTIVE.  The standing user law is
that trades live behind ONE objective knob (memory
``kv-vs-perf-waehlbar-fast-getrennt``), and this file asserts that shape rather
than merely the arithmetic: ``--pp-solve-pool-floor`` does not rank anything.
It narrows the set, and ``--pp-solve-objective`` ranks over what is left.
"fastest above F" is the same ranking on a smaller set, which is why it can be
a constraint and not a knob.

NO SILENT FALLBACK IS THE LOAD-BEARING HALF.  A floor that degrades to "the
fastest cut below it" when nothing clears is worse than no floor: every line
of the boot still says the floor was honoured.  So the empty case is a W40
refusal that prints the FRONTIER -- the alternatives that do clear it, and the
best pool anywhere in the field -- and the mutants below are aimed at exactly
that direction.

THE FIXTURE IS THE FIVE PRICED CUTS OF SECTION 1am-fix
(``/spinning/gpu-arb/weg2/WEG2_BUILD_DECISIONS_0906.md``), and it is
SELF-VERIFYING rather than transcribed: every pool below is re-derived here by
the shipped ``pp_phase_pool`` from the shipped launcher constants, and asserted
against the figure the record published.  A fixture that only quoted the record
would go on passing after the model drifted away from it.

    cut        attn     priced pool     realised (metal)
    44,10,10   11,2,3       304,946     304,655  (weg2sb5f, +0.10 %)
    43,11,10   10,3,3       354,047     --
    42,11,11   10,3,3       374,249     --
    32,18,14    8,4,4       715,089     714,788  (weg2rg6/sb4, +0.04 %)
    31,17,16    7,5,4       614,078     --

Hermetic: no GPU, no NVML, no checkpoint, no server, no launcher run.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner.pp_cut import (
    LAYER_FAMILY_ATTENTION,
    LAYER_FAMILY_LINEAR,
    PhasePoolModel,
    attention_counts,
    family_costs_from_measurement,
    pp_phase_pool,
    stage_pp_capacities,
)
from sglang.srt.planner.pp_cut_launch import (
    FRONTIER_MAX_POINTS,
    CutCandidate,
    CutDecision,
    PPCutRefused,
    pareto_frontier,
    solve_launch_cut,
)
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# THE POOL AXIS -- boot weg2sb5f's own inputs, the same fixture as
# test_pp_cut_boot_sizing_1286.py and cited to the same two boots.
# ---------------------------------------------------------------------------

#: ``PP-CUT inputs: ... free=[27960, 17064, 16552] MiB`` (weg2sb5f; identical
#: on weg2rg6).
BUDGETS_MIB = (27960.0, 17064.0, 16552.0)
#: ``weights attn 355.1 / linear 366.2 MiB per layer -> mean 363.4 used``.
MEAN_LAYER_MIB = 363.4
#: ``kv=2048 B/token/attn-layer (from config, fp8_e4m3)``.
KV_MIB_PER_TOKEN_PER_ATTN_LAYER = 2048.0 / (1024.0 * 1024.0)
ARMING_FLOOR_MIB = 1229.0
CORRIDOR_HOLDBACK_MIB = 1024.0
ACTIVATION_RESERVE_MIB = 1024.0
MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT = 1.5588
MAMBA_SLOTS = 20
STAGE_FIXED_MIB = (2342.0, 1105.5, 3518.0)
ZERO_POSTS_ACKNOWLEDGED = (
    "mamba pre-capture reserve",
    "speculative intermediate state",
    "GGUF dequant scratch",
)

#: 64 layers, every 4th a full-attention layer -- the checkpoint's own
#: ``layer_types``.
FAMILIES = tuple(
    LAYER_FAMILY_ATTENTION if (i % 4) == 3 else LAYER_FAMILY_LINEAR
    for i in range(64)
)

# ---------------------------------------------------------------------------
# THE TIME AXIS -- the launcher's own defaults, so the ranking under test is
# the ranking a boot would get and not a ranking invented for a test.
# ---------------------------------------------------------------------------

#: ``--pp-cut-measured-ms-per-layer`` default (boot bsscale, BSSCALE_0907.md).
MEASURED_MS_PER_LAYER = (8.10, 35.16, 33.59)
INCUMBENT = (32, 18, 14)
CHUNK_TOKENS = 4096
#: ``--pp-cut-calibration-prefix-tokens`` / ``--pp-cut-design-prefix-tokens``
#: fallback.  Equal, so the depth factor is exactly 1.0 and the ranking under
#: test is not also a test of the depth extrapolation.
CALIBRATION_PREFIX_TOKENS = 4096.0
DESIGN_PREFIX_TOKENS = 4096
#: ``--pp-cut-attn-anchor-ms`` / ``--pp-cut-attn-anchor-prefix-tokens``.
ATTN_ANCHOR_MS = 400.0
ATTN_ANCHOR_PREFIX_TOKENS = 262144.0

#: THE ANCHOR STAGE IS NOT A CHOICE MADE HERE.  The launcher takes the first
#: card that is not the 5090 (``launcher.py``: ``next((i for i, c in
#: enumerate(cards) if "5090" not in c.name), len(cards) - 1)``), and on group
#: P's layout stage 0 IS the 5090 -- which its own measured rate says out loud:
#: 8.10 ms/layer against 35.16 and 33.59.  So the anchor is stage 1.
#:
#: It matters, and that is stated rather than hidden: with the anchor on stage
#: 0 the family split makes 43,11,10 and 42,11,11 cost the SAME to the last
#: digit (390.65 ms both), and the fixture's ordering would rest on a
#: tie-break.  On stage 1 -- the launcher's own answer -- and on stage 2 they
#: separate, in the same order.  A test whose expected value depends on a
#: parameter must name the parameter.
ANCHOR_STAGE = 1

#: Crossing prices for the two adjacent pairs.  ZERO on purpose: a contiguous
#: three-stage cut always makes exactly the same two crossings, so
#: ``crossing_ms`` is cut-INDEPENDENT here and cannot enter the ranking.
#: Pricing it would make this file a second test of #1240's crossing model
#: instead of a test of the floor; leaving the map EMPTY would be worse still,
#: because ``crossing_price`` then refuses every candidate as UNPRICED.
PAIR_MS = {(0, 1): 0.0, (1, 2): 0.0}

#: Card names no measured rate library can answer for, so
#: ``ms_per_layer_from_card_library`` returns None and the cost model is the
#: measured list above -- deterministic on any box, whether or not a
#: card_rate_pass library happens to be on it.
HERMETIC_CARDS = ("HERMETIC-TEST-CARD-A", "HERMETIC-TEST-CARD-B", "HERMETIC-TEST-CARD-C")

#: ``--max-kv-per-request``: the PHYSICAL floor, one full-context prompt.
CAP_TOKENS = 262144

# ---------------------------------------------------------------------------
# THE FIVE PRICED CUTS of SECTION 1am-fix, with the pools the record published.
# ---------------------------------------------------------------------------

SB5F = (44, 10, 10)
CUT_43 = (43, 11, 10)
CUT_42 = (42, 11, 11)
RG6 = (32, 18, 14)
CUT_31 = (31, 17, 16)

PRICED_POOL = {
    SB5F: 304946,
    CUT_43: 354047,
    CUT_42: 374249,
    RG6: 715089,
    CUT_31: 614078,
}
#: The per-rank halves of two of them, from the same table -- kept because the
#: WORLD figure is a min and a min can be right for the wrong reason.
PRICED_PER_RANK = {
    SB5F: (304946, 2566935, 1217496),
    RG6: (716350, 887403, 715089),
}
#: The attention vectors the record names beside each cut.  Re-derived below;
#: kept here so a drift in ``attention_counts`` fails as itself.
PRICED_ATTN = {
    SB5F: (11, 2, 3),
    CUT_43: (10, 3, 3),
    CUT_42: (10, 3, 3),
    RG6: (8, 4, 4),
    CUT_31: (7, 5, 4),
}

#: THE ANSWERS THIS FILE PINS, in the order the field ranks them by time.
#: Derived from the constants above, not chosen: see
#: ``test_the_makespan_order_of_the_five_is_the_cost_models_own``.
FIVE_BY_MAKESPAN = (SB5F, CUT_43, CUT_42, CUT_31, RG6)


def funded_model(budgets=BUDGETS_MIB) -> PhasePoolModel:
    """The pool model with EVERY post the boot charges, funded (#1286)."""
    return PhasePoolModel(
        free_mib=tuple(budgets),
        weight_mib_per_layer=MEAN_LAYER_MIB,
        kv_mib_per_token_per_attn_layer=KV_MIB_PER_TOKEN_PER_ATTN_LAYER,
        arming_floor_mib=tuple(ARMING_FLOOR_MIB for _ in budgets),
        mamba_mib_per_linear_layer_per_slot=MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT,
        mamba_slots=MAMBA_SLOTS,
        stage_fixed_mib=STAGE_FIXED_MIB,
        activation_reserve_mib=ACTIVATION_RESERVE_MIB,
        corridor_holdback_mib=CORRIDOR_HOLDBACK_MIB,
        zero_posts_acknowledged=ZERO_POSTS_ACKNOWLEDGED,
        page_size=1,
    )


def family_cost():
    """The launcher's own two-depth family split, from its own defaults."""
    cost, _prov = family_costs_from_measurement(
        measured_ms_per_layer=MEASURED_MS_PER_LAYER,
        measured_counts=INCUMBENT,
        measured_attn_counts=attention_counts(FAMILIES, INCUMBENT),
        chunk_tokens=CHUNK_TOKENS,
        ref_prefix_tokens=CALIBRATION_PREFIX_TOKENS,
        anchor_stage=ANCHOR_STAGE,
        anchor_attn_ms_per_layer=ATTN_ANCHOR_MS,
        anchor_prefix_tokens=ATTN_ANCHOR_PREFIX_TOKENS,
    )
    return cost


def makespan_of(counts) -> float:
    """The contiguous makespan: the SLOWEST stage, the rule ``price()`` uses."""
    attn = attention_counts(FAMILIES, tuple(counts))
    return max(family_cost().stage_ms(tuple(counts), attn, DESIGN_PREFIX_TOKENS))


def solve(**kw) -> CutDecision:
    """``solve_launch_cut`` over the REAL contiguous field of this rig."""
    params = dict(
        layer_families=FAMILIES,
        incumbent_layers=INCUMBENT,
        measured_ms_per_layer=MEASURED_MS_PER_LAYER,
        measured_provenance="hermetic fixture: the launcher's own defaults",
        card_names=HERMETIC_CARDS,
        pool_model=funded_model(),
        cap_tokens=CAP_TOKENS,
        family_cost=family_cost(),
        design_prefix_tokens=DESIGN_PREFIX_TOKENS,
        per_pair_crossing_ms=PAIR_MS,
        # The gapped family is #753-gated and its own slice's subject; keeping
        # it out makes this file's field the CONTIGUOUS one, which is what the
        # five fixture cuts are and what a boot of this form actually ships.
        enumerate_gapped=False,
        objective="makespan",
    )
    params.update(kw)
    return solve_launch_cut(**params)


def cand(counts, pool, ms) -> CutCandidate:
    """A bare candidate, for the pure ``pareto_frontier`` cases."""
    return CutCandidate(
        layers=tuple(counts),
        attn=attention_counts(FAMILIES, tuple(counts)),
        makespan_ms=float(ms),
        pool_tokens=float(pool),
    )


class TestTheFixtureIsTheShippedModelsOwnArithmetic(CustomTestCase):
    """Before anything is asserted ABOUT the five cuts, the five are checked.

    The record's figures are the falsifiable half here: they were computed by
    this code on 2026-09-09 and written down, and if the model drifts the
    fixture must fail rather than quietly become a description of an older
    model.  This is the same discipline #1286 used against the two boots.
    """

    def test_every_fixture_pool_is_pp_phase_pools_own_answer(self):
        model = funded_model()
        for counts, expected in sorted(PRICED_POOL.items()):
            attn = attention_counts(FAMILIES, counts)
            self.assertEqual(
                attn,
                PRICED_ATTN[counts],
                msg="the attention vector of %s drifted from the record" % (counts,),
            )
            self.assertEqual(
                int(pp_phase_pool(counts, attn, model)),
                int(expected),
                msg=(
                    "cut %s prices at a different world pool than SECTION "
                    "1am-fix published; the fixture and the model disagree, "
                    "and the model is what a boot runs" % (counts,)
                ),
            )

    def test_the_per_rank_halves_match_too_so_the_min_is_right_for_the_right_reason(self):
        model = funded_model()
        for counts, expected in sorted(PRICED_PER_RANK.items()):
            got = tuple(
                int(c)
                for c in stage_pp_capacities(
                    counts, attention_counts(FAMILIES, counts), model
                )
            )
            self.assertEqual(got, expected, msg="per-rank capacities of %s" % (counts,))

    def test_the_makespan_order_of_the_five_is_the_cost_models_own(self):
        """The ORDER the floor tests depend on, derived rather than assumed."""
        order = tuple(sorted(PRICED_POOL, key=makespan_of))
        self.assertEqual(
            order,
            FIVE_BY_MAKESPAN,
            msg=(
                "the five fixture cuts no longer rank in the order this file's "
                "floor expectations are built on; every expectation below is a "
                "statement about THIS order"
            ),
        )
        # ... and the two axes genuinely oppose each other over the five, which
        # is the whole reason a floor is a meaningful thing to ask for.
        by_pool = tuple(sorted(PRICED_POOL, key=lambda c: PRICED_POOL[c]))
        self.assertEqual(by_pool[0], SB5F)
        self.assertEqual(by_pool[-1], RG6)
        self.assertEqual(FIVE_BY_MAKESPAN[0], SB5F)


class TestTheFloorConstrainsTheObjective(CustomTestCase):
    """floor None / 350k / 500k, over the rig's real contiguous field."""

    def test_no_floor_is_the_behaviour_before_1286b(self):
        """DEFAULT None = byte-identical.  Two independent statements of it."""
        base = solve()
        self.assertIsNone(base.pool_floor)
        self.assertEqual(base.chosen.layers, SB5F)
        self.assertEqual(int(base.chosen.pool_tokens), PRICED_POOL[SB5F])
        # (a) passing the argument explicitly as None changes NOTHING -- every
        #     field of the decision, not merely the chosen row.
        explicit = solve(pool_floor=None)
        for field in ("chosen", "kv_floor", "makespan", "ranked", "cap_tokens",
                      "objective", "unpriced", "pinned", "design_prefix_tokens"):
            self.assertEqual(
                getattr(base, field),
                getattr(explicit, field),
                msg="pool_floor=None moved %s" % field,
            )
        self.assertEqual(base.provenance_line(), explicit.provenance_line())
        self.assertEqual(base.table_lines(), explicit.table_lines())
        self.assertEqual(base.trade_line(), explicit.trade_line())
        # (b) and the chosen row is what the PRE-#1286b rule says it is, with
        #     that rule restated here independently of the code under test.
        feasible = [
            c for c in base.ranked if float(c.pool_tokens) >= float(CAP_TOKENS)
        ]
        want = min(feasible, key=lambda c: (c.total_ms, -c.pool_tokens))
        self.assertEqual((want.layers, want.attn), (base.chosen.layers, base.chosen.attn))

    def test_floor_350k_picks_the_fastest_cut_that_clears_it(self):
        d = solve(pool_floor=350_000)
        self.assertEqual(d.pool_floor, 350_000)
        self.assertEqual(d.chosen.layers, CUT_43)
        self.assertEqual(int(d.chosen.pool_tokens), PRICED_POOL[CUT_43])
        # It is NOT the unconstrained winner, and it IS slower than it: the
        # floor cost something, and the test says how much rather than only
        # that the answer changed.
        free = solve()
        self.assertNotEqual(d.chosen.layers, free.chosen.layers)
        self.assertGreater(d.chosen.total_ms, free.chosen.total_ms)
        self.assertGreater(d.chosen.pool_tokens, free.chosen.pool_tokens)

    def test_floor_500k_picks_the_fastest_above_the_floor_whatever_it_is(self):
        """The FIELD answers; the test asserts the PROPERTY, not a favourite.

        The briefing's expectation is stated as a property on purpose: which
        cut wins at 500,000 is a fact about this checkpoint's enumeration, and
        pinning a name would make the test a transcript of one solve.  What
        must hold is that the winner clears the floor, is the fastest thing
        that does, and is not slower than the incumbent -- which clears it.
        """
        floor = 500_000
        d = solve(pool_floor=floor)
        self.assertGreaterEqual(int(d.chosen.pool_tokens), floor)
        above = [c for c in d.ranked if float(c.pool_tokens) >= float(floor)]
        self.assertTrue(above, msg="fixture invalid: nothing clears 500k")
        best = min(above, key=lambda c: (c.total_ms, -c.pool_tokens))
        self.assertEqual((d.chosen.layers, d.chosen.attn), (best.layers, best.attn))
        # The incumbent clears 500k, so the winner can never be slower than it.
        rg6 = next(c for c in d.ranked if c.layers == RG6)
        self.assertGreaterEqual(int(rg6.pool_tokens), floor)
        self.assertLessEqual(d.chosen.total_ms, rg6.total_ms)

    def test_the_floor_is_never_degraded_to_a_faster_cut_below_it(self):
        """THE DANGER DIRECTION, asserted over a whole ladder of floors.

        One floor value proves one point of a step function.  Every floor from
        below the field to above it is swept here, and at each the answer is
        either a cut that clears it or a refusal -- never a fast cut below it,
        which is the failure a silent fallback produces and the one an operator
        cannot see from any log line.
        """
        for floor in (0, 100_000, 262_144, 304_946, 304_947, 350_000, 400_000,
                      500_000, 600_000, 715_089):
            try:
                d = solve(pool_floor=floor)
            except PPCutRefused:
                continue
            self.assertGreaterEqual(
                int(d.chosen.pool_tokens),
                floor,
                msg="floor %d shipped a cut below itself" % floor,
            )

    def test_the_same_floor_binds_the_maxkv_arm_and_the_printed_makespan_row(self):
        """Not "the makespan arm honours it": the SHIPPED row of every arm does.

        ``pick_shipped_cut`` returns ``decision.makespan`` for the makespan arm
        and ``decision.kv_floor`` for maxkv, so both rows are shipping
        surfaces, and a floor honoured only by ``chosen`` would be a floor one
        flag walks past.
        """
        floor = 500_000
        d = solve(pool_floor=floor, objective="maxkv")
        self.assertGreaterEqual(int(d.chosen.pool_tokens), floor)
        self.assertGreaterEqual(int(d.kv_floor.pool_tokens), floor)
        self.assertIsNotNone(d.makespan)
        self.assertGreaterEqual(int(d.makespan.pool_tokens), floor)
        # maxkv is still maxkv: the floor narrowed the set, it did not re-rank.
        self.assertEqual(
            int(d.chosen.pool_tokens),
            max(int(c.pool_tokens) for c in d.ranked),
        )


class TestTheRefusalIsNamedAndCarriesTheFrontier(CustomTestCase):
    def test_a_floor_above_the_whole_field_refuses_by_name(self):
        ceiling = max(PRICED_POOL.values())
        with self.assertRaises(PPCutRefused) as ctx:
            solve(pool_floor=ceiling + 5_000_000)
        msg = str(ctx.exception)
        self.assertIn("W40 Weg2PPCutRefused", msg)
        self.assertIn("--pp-solve-pool-floor", msg)
        # It must NOT be answerable by moving --max-kv-per-request: that is a
        # different floor and naming it here would send the operator to the
        # wrong flag (the #1286 F5 shape -- the right sentence for the wrong
        # fault is more expensive than no sentence).
        self.assertNotIn("--max-kv-per-request", msg)

    def test_that_refusal_prints_the_frontier_and_its_denominator(self):
        with self.assertRaises(PPCutRefused) as ctx:
            solve(pool_floor=9_000_000)
        msg = str(ctx.exception)
        self.assertIn("FRONTIER:", msg)
        self.assertIn("NONE of", msg)
        self.assertIn("servable candidates clears the floor", msg)
        self.assertIn("Best pool anywhere in the field", msg)
        self.assertIn("short of the floor by", msg)
        # The ceiling it names is the field's real maximum -- not the largest
        # of the five landmarks, the largest ANYWHERE -- so the operator reads
        # "this floor is unreachable, and by how much" off the sentence.
        best = max(int(c.pool_tokens) for c in solve().servable)
        self.assertIn("pool %d" % best, msg)
        self.assertGreater(best, max(PRICED_POOL.values()))

    def test_a_pin_below_the_floor_is_refused_and_the_alternatives_are_named(self):
        """The three-best half of the frontier note, on the path it is for.

        Here the field DOES clear the floor and one pinned layout does not, so
        the useful sentence is not "impossible" but "these three would".
        """
        floor = 500_000
        with self.assertRaises(PPCutRefused) as ctx:
            solve(pool_floor=floor, pinned_layers=SB5F)
        msg = str(ctx.exception)
        self.assertIn("--pp-solve-pool-floor", msg)
        self.assertIn("the pinned layer cut 44,10,10", msg)
        self.assertIn("short by %d" % (floor - PRICED_POOL[SB5F]), msg)
        self.assertIn("clear the floor", msg)
        self.assertIn("fastest are", msg)
        self.assertIn("Best pool anywhere in the field", msg)

    def test_the_physical_floor_keeps_its_own_flag_and_its_own_sentence(self):
        """Two floors, two refusals; the older one must not be re-labelled."""
        with self.assertRaises(PPCutRefused) as ctx:
            solve(pinned_layers=(4, 32, 28))
        msg = str(ctx.exception)
        self.assertIn("--max-kv-per-request", msg)
        self.assertNotIn("--pp-solve-pool-floor", msg)

    def test_a_pin_that_clears_both_floors_is_not_refused(self):
        d = solve(pool_floor=500_000, pinned_layers=RG6)
        self.assertTrue(d.pinned)
        self.assertEqual(d.chosen.layers, RG6)
        self.assertEqual(d.pool_floor, 500_000)


class TestThePareto(CustomTestCase):
    """The curve itself: what is on it, what is not, and what the line says."""

    def test_a_dominated_candidate_is_not_on_the_frontier(self):
        fast_small = cand((44, 10, 10), 304_946, 359.0)
        slow_big = cand((32, 18, 14), 715_089, 632.9)
        # Beaten on BOTH axes by fast_small: slower and smaller.
        dominated = cand((43, 11, 10), 300_000, 400.0)
        got = pareto_frontier([slow_big, dominated, fast_small])
        self.assertEqual(
            [c.layers for c in got], [fast_small.layers, slow_big.layers]
        )

    def test_a_candidate_beaten_on_one_axis_only_stays(self):
        a = cand((44, 10, 10), 304_946, 359.0)
        b = cand((32, 18, 14), 715_089, 632.9)
        self.assertEqual(len(pareto_frontier([a, b])), 2)

    def test_two_points_equal_on_both_axes_collapse_to_one(self):
        a = cand((43, 11, 10), 354_047, 368.3)
        b = cand((42, 11, 11), 354_047, 368.3)
        self.assertEqual(len(pareto_frontier([a, b])), 1)

    def test_equal_time_keeps_only_the_larger_pool(self):
        small = cand((43, 11, 10), 354_047, 368.3)
        big = cand((42, 11, 11), 374_249, 368.3)
        got = pareto_frontier([small, big])
        self.assertEqual([c.layers for c in got], [big.layers])

    def test_the_frontier_of_the_real_field_is_a_monotone_staircase(self):
        d = solve()
        front = d.frontier
        self.assertTrue(front)
        for a, b in zip(front, front[1:]):
            self.assertLess(
                a.total_ms, b.total_ms, msg="frontier is not sorted by time"
            )
            self.assertLess(
                a.pool_tokens, b.pool_tokens, msg="frontier pool is not monotone"
            )
        # Both ENDS are the field's own extremes, which is what makes the curve
        # readable as a curve: everything faster than the first point does not
        # exist, and nothing anywhere holds more than the last.
        self.assertEqual(
            front[0].total_ms, min(c.total_ms for c in d.servable)
        )
        self.assertEqual(
            front[-1].pool_tokens, max(c.pool_tokens for c in d.servable)
        )
        # No point on it is dominated by any candidate of the field.
        for p in front:
            for c in d.servable:
                if c.total_ms <= p.total_ms and c.pool_tokens >= p.pool_tokens:
                    self.assertTrue(
                        c.total_ms == p.total_ms and c.pool_tokens == p.pool_tokens,
                        msg="%s is dominated by %s" % (p.layers, c.layers),
                    )

    def test_the_four_landmark_cuts_are_on_it_and_the_fifth_is_dominated(self):
        """The record's five, sorted onto the curve -- and 31,17,16 is NOT on it.

        This is the finding worth pinning: 31,17,16 was the OLD model's
        pool-maximal row (988,897 before #1286), and against the whole field it
        is not even Pareto-optimal -- some cut is both faster AND larger.  A
        frontier that listed it would be listing a layout no operator should
        ever choose, which is exactly what "dominated" means.
        """
        d = solve()
        on = {c.layers for c in d.frontier}
        for counts in (SB5F, CUT_43, CUT_42, RG6):
            self.assertIn(counts, on, msg="%s fell off the frontier" % (counts,))
        self.assertNotIn(CUT_31, on)
        # ... and the domination is NAMED, not merely asserted: some real
        # candidate is at least as fast and holds at least as much.
        c31 = next(c for c in d.ranked if c.layers == CUT_31)
        self.assertEqual(int(c31.pool_tokens), PRICED_POOL[CUT_31])
        strict = [
            c
            for c in d.servable
            if c.layers != CUT_31
            and c.total_ms <= c31.total_ms
            and c.pool_tokens >= c31.pool_tokens
            and (c.total_ms < c31.total_ms or c.pool_tokens > c31.pool_tokens)
        ]
        self.assertTrue(
            strict,
            msg="31,17,16 is off the frontier but nothing dominates it",
        )
        best = min(strict, key=lambda c: c.total_ms)
        self.assertLess(best.total_ms, c31.total_ms)
        self.assertGreater(best.pool_tokens, c31.pool_tokens)

    def test_the_frontier_line_carries_the_floor_the_counts_and_both_ends(self):
        d = solve(pool_floor=500_000)
        line = d.frontier_line()
        self.assertTrue(line.startswith("PP-CUT FRONTIER: "))
        self.assertIn("pool_floor=500000", line)
        self.assertIn("non-dominated of", line)
        self.assertIn("servable (of", line)
        self.assertIn("priced)", line)
        # Every point says whether it clears the floor, so the cost of the
        # floor is a subtraction between two rows of ONE line.
        self.assertIn(" BELOW", line)
        self.assertIn(" CLEARS", line)
        self.assertIn("pool=%d" % int(d.frontier[-1].pool_tokens), line)

    def test_the_frontier_line_says_none_when_no_floor_is_set(self):
        line = solve().frontier_line()
        self.assertIn("pool_floor=none", line)
        self.assertNotIn(" BELOW", line)
        self.assertNotIn(" CLEARS", line)

    def test_truncation_bounds_resolution_and_never_range(self):
        """A bounded line must not be able to hide the curve's right-hand end."""
        field = [cand((44, 10, 10), 100_000 + 1000 * i, 300.0 + i) for i in range(60)]
        decision = CutDecision(
            chosen=field[0],
            kv_floor=field[-1],
            cap_tokens=CAP_TOKENS,
            pinned=False,
            cost_provenance="hermetic",
            ranked=tuple(field),
            frontier=pareto_frontier(field),
            servable=tuple(field),
        )
        line = decision.frontier_line(top=5)
        self.assertIn("further non-dominated point(s)", line)
        self.assertIn("pool=%d" % int(field[0].pool_tokens), line)
        self.assertIn("pool=%d" % int(field[-1].pool_tokens), line)
        # and the default bound is generous enough not to bite on this rig
        self.assertGreaterEqual(FRONTIER_MAX_POINTS, len(solve().frontier))

    def test_an_empty_field_says_so_rather_than_printing_an_empty_curve(self):
        decision = CutDecision(
            chosen=cand((44, 10, 10), 1, 1.0),
            kv_floor=cand((44, 10, 10), 1, 1.0),
            cap_tokens=CAP_TOKENS,
            pinned=False,
            cost_provenance="hermetic",
        )
        self.assertIn("EMPTY", decision.frontier_line())


class TestTheLauncherSeam(CustomTestCase):
    """The flag, its default, and the two lines that publish the decision.

    Source-level where a boot would be needed otherwise -- the same choice
    ``test_pp_cut_boot_sizing_1286.py`` makes for ``PP-POOL-JOIN``: these
    assertions are about what the emitter WRITES, and the emitter cannot be
    run here without a checkpoint and three cards.
    """

    def _parser(self):
        from sglang.srt.weg2.launcher import build_parser

        return build_parser()

    def _launcher_source(self):
        import sglang.srt.weg2.launcher as mod

        with open(mod.__file__, "r", encoding="utf-8") as fh:
            return fh.read()

    def test_the_flag_exists_and_defaults_to_no_floor(self):
        ns = self._parser().parse_args(["--tree", "/t", "--tag", "x"])
        self.assertIsNone(ns.pp_solve_pool_floor)
        ns = self._parser().parse_args(
            ["--tree", "/t", "--tag", "x", "--pp-solve-pool-floor", "500000"]
        )
        self.assertEqual(ns.pp_solve_pool_floor, 500000)

    def test_it_is_one_flag_and_not_a_second_objective(self):
        act = next(
            a for a in self._parser()._actions if a.dest == "pp_solve_pool_floor"
        )
        self.assertIsNone(getattr(act, "choices", None))
        objective = next(
            a for a in self._parser()._actions if a.dest == "pp_solve_objective"
        )
        self.assertEqual(set(objective.choices), {"maxkv", "makespan", "incumbent"})

    def test_the_help_states_the_no_fallback_rule_and_the_other_floor(self):
        # Flattened: argparse re-wraps the paragraph at the terminal width, so
        # a two-word phrase can land across a line break whenever the text
        # BEFORE it changes length (measured 2026-09-09: the #1305 rewrite of
        # the sentence head split 'NO SILENT' as 'NO\nSILENT'). Asserting on
        # the raw text tests the line width, not the presence of the rule.
        help_text = " ".join(self._parser().format_help().split())
        self.assertIn("--pp-solve-pool-floor", help_text)
        self.assertIn("NO SILENT", help_text)
        self.assertIn("--max-kv-per-request", help_text)

    def test_the_launcher_hands_the_floor_to_the_solver(self):
        """The flag reaches the solver -- since 2026-09-09 via the resolver.

        It used to read ``pool_floor=ns.pp_solve_pool_floor`` at the call. The
        shipped default (the 39,13,12 order) put ``resolve_pool_floor`` in
        between, so the assertion follows the seam rather than the old spelling:
        the flag goes INTO the resolver and the resolver's value goes into the
        solve. Both halves are asserted, because either one alone would still
        pass with the floor dropped on the floor between them.
        """
        src = self._launcher_source()
        self.assertIn("resolve_pool_floor(ns.pp_solve_pool_floor)", src)
        self.assertIn("pool_floor=pool_floor", src)

    def test_the_shipped_line_renders_and_publishes_the_floor(self):
        """RENDERED, not grepped: twelve substitutions is the failure class.

        A wrong arity in this format raises TypeError at EMIT time -- after the
        weights are loaded, which is a spent window -- and a source scan cannot
        see it. ``shipped_line`` is pure for exactly this reason.
        """
        from sglang.srt.weg2.launcher import (
            incumbent_candidate,
            pick_shipped_cut,
            shipped_line,
        )

        for floor, shown in ((None, "pool_floor=none"), (350_000, "pool_floor=350000")):
            d = solve(pool_floor=floor)
            chosen, why = pick_shipped_cut(d, INCUMBENT, PRICED_ATTN[RG6], "makespan")
            line = shipped_line(
                d,
                chosen,
                why,
                incumbent_candidate(d, INCUMBENT, PRICED_ATTN[RG6]),
                d.makespan or d.chosen,
                # PASSED IN, never read from P_PP_STAGE_RATIO_SCORES inside the
                # formatter: #1233 pins the census of that vector's readers.
                # An obviously fake value, asserted ABSENT below, because the
                # incumbent IS ranked in this field and the fallback must not
                # fire.
                incumbent_fallback="FALLBACK-MUST-NOT-APPEAR",
                # REQUIRED since 2026-09-09: once the floor has a shipped
                # default, the number alone no longer says whether the boot
                # obeyed a standing order or an operator. This file drives the
                # solver directly, so its floors are all "flag" by definition.
                floor_source="flag (test drives solve_launch_cut directly)",
            )
            self.assertNotIn("FALLBACK-MUST-NOT-APPEAR", line)
            print("\nRECORD %s" % line)
            self.assertTrue(line.startswith("PP-CUT SHIPPED: "))
            self.assertIn(shown, line)
            self.assertIn("chosen_pool=%d" % int(chosen.pool_tokens), line)
            self.assertIn("makespan_ms=", line)
            # the two arms it did NOT take stay priced beside it (#1254)
            self.assertIn("pool-maximal (kv-floor)", line)
            self.assertIn("makespan-optimal", line)
            # and the floor it reports is the one the shipped row actually met
            if floor is not None:
                self.assertGreaterEqual(int(chosen.pool_tokens), floor)

    def test_the_frontier_line_of_this_fixture_is_recorded(self):
        """Prints the curve, so the record carries the line a boot will emit."""
        for floor in (None, 500_000):
            print("\nRECORD %s" % solve(pool_floor=floor).frontier_line())
        d = solve()
        for point in d.frontier:
            print(
                "RECORD FRONTIER-POINT %s attn %s total_ms=%.2f pool=%d"
                % (
                    ",".join(str(n) for n in point.layers),
                    ",".join(str(a) for a in point.attn),
                    point.total_ms,
                    int(point.pool_tokens),
                )
            )
        print(
            "RECORD FIELD priced=%d servable=%d frontier=%d"
            % (len(d.ranked), len(d.servable), len(d.frontier))
        )
        self.assertTrue(d.frontier)

    def test_the_frontier_line_is_emitted_on_every_boot(self):
        src = self._launcher_source()
        self.assertIn("log(decision.frontier_line())", src)

    def test_the_shipped_candidate_is_checked_against_the_floor(self):
        """`incumbent` names a candidate, so it bypasses the narrowed set."""
        src = self._launcher_source()
        self.assertIn("decision.refuse_shipped_below_floors(", src)
        # and the check sits AFTER the arm is picked, not before it
        self.assertLess(
            src.index("chosen, ship_why = pick_shipped_cut("),
            src.index("decision.refuse_shipped_below_floors("),
        )

    def test_that_check_actually_refuses_a_shipped_row_below_the_floor(self):
        """The guard, exercised rather than only located."""
        d = solve()
        below = d.chosen
        object.__setattr__(d, "pool_floor", int(below.pool_tokens) + 1)
        with self.assertRaises(PPCutRefused) as ctx:
            d.refuse_shipped_below_floors(below, "the SHIPPED cut (test)")
        self.assertIn("--pp-solve-pool-floor", str(ctx.exception))
        # and passes a row that clears it
        object.__setattr__(d, "pool_floor", int(below.pool_tokens))
        d.refuse_shipped_below_floors(below, "the SHIPPED cut (test)")


if __name__ == "__main__":
    unittest.main()
