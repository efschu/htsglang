# SPDX-License-Identifier: Apache-2.0
"""The P cut ships 39,13,12 BY DEFAULT -- solved, not pinned, and the floor is
READ OFF THE BOOT'S OWN FRONTIER (#1305).

USER ORDER, 2026-09-09, verbatim: "39,13,12 mit bs2 im pp layout soll standard
werden vorerst."

THE DISTINCTION THIS FILE EXISTS TO PROTECT.  The cheap way to obey that order
is ``--pp-stage-ratio 39,13,12`` -- a hand pin -- and it is the wrong way,
because a pin keeps shipping 39,13,12 after the thing that made it right has
changed.  The solver stays the authority.  What the order changes is a
CONSTRAINT under ``--pp-solve-objective makespan``: a floor under the shipped
pool, so the rule reads "the FASTEST cut that still holds F tokens".

WHAT THE PREVIOUS FORM OF THIS FILE PINNED, AND WHY IT WAS GREEN ON A WRONG
BOOT.  The floor used to be a CONSTANT, 448,027, the midpoint of the interval
(414654, 481400] on which "fastest above F" selects 39,13,12 -- read off ONE
boot's frontier (#1286b, tip 68308f7fae).  This file drove ``choose_under_
floor`` with that constant against THAT recorded frontier and was green.  On
boot weg2sn5pre (2026-09-09, tip 319e76d60b, the merged serve tree) the census
budgets had moved the whole frontier up in pool, the selecting interval had
become (482768, 578199], and the launcher printed the order on its provenance
line and SHIPPED 42,11,11.  A test that pins a constant against a stale
frontier cannot see a moved frontier: the desk stayed green while the metal
shipped the wrong cut (record SECTION 1bg).

THE RULE NOW (operator ruling, #1305): the order names a CUT
(``DEFAULT_PP_ORDERED_CUT``), and the solver derives the floor from its OWN
frontier at boot: ``floor = pool(ordered cut on this frontier)``.  Under
"fastest above F" that floor selects the ordered cut EXACTLY -- every faster
frontier point holds strictly less pool -- and it cannot drift, because the
floor and the frontier come out of one solve on one set of inputs.  A cut that
is not on the frontier is ``W67 Weg2PPCutOrderedCutOffFrontier``, never a
neighbour.  ``--pp-solve-pool-floor N`` stays the operator's override, ``0``
turns the floor off.  No constant replaces 448,027.

WHAT IS DRIVEN.  Three layers, from cheapest to closest to the metal:

1. ``derive_pool_floor_from_cut`` + ``choose_under_floor`` -- the shipped rule
   -- against TWO recorded frontiers: #1286b's fifteen points, and
   weg2sn5pre's fourteen (the boot that shipped 42,11,11 under the constant).
   The rule ships 39,13,12 on BOTH; the constant ships 42,11,11 on the second
   -- the defect reproduced against the rule that replaced it.
2. ``solve_launch_cut`` on THIS BOOT'S OWN INPUTS -- weg2sn5pre's per-rank
   census budgets, posts and cost inputs transcribed from its ``PP-CUT
   inputs`` / ``PP-CUT budget posts`` lines -- with the derivation armed.  The
   POOL axis is the pool model's exact arithmetic (the boot's own
   ``PP-POOL-JOIN`` priced 463,763 for 42,11,11 and group P sized 463,289,
   0.1 %); the TIME axis on the desk is the measured-ms fallback rather than
   the card-rate library, so ms figures are not asserted, pools and the
   chosen cut are.  This is the "desk solver path with the tree's census
   budgets" the ruling asked for: the budgets are an INPUT here, transcribed,
   and what the test proves is that the RULE ships the ordered cut on them --
   the next census change cannot ship a neighbour any more, it can only
   refuse by name.
3. The launcher seam: ``resolve_pool_floor`` hands the cut to the solver, the
   ``PP-CUT POOL FLOOR:`` line is rendered from the decision AFTER the solve
   (the number does not exist before the frontier does), ``PP-CUT SHIPPED``
   carries ``pool_floor_source=default-from-ordered-cut``, the help quotes the
   order and names W67, and the AST pin: the cut is assigned exactly once, the
   integer 448027 appears nowhere in either package, no ``pool_floor``
   signature default is a bare int, the argparse default stays None.

Plus #1305 item 2, the same boot's second finding: group P's
``--max-total-tokens`` was a constant 428,000 (boot weg2zr2's numbers) and cut
the REALISED pool 7.7 % below the priced one -- below the floor the boot
claimed to hold.  It is now the shipped cut's priced pool, handed to ``argv_p``
by the caller that holds ``PCutFacts``, and the ``PP-CUT P-CAP:`` line prints
the cap and the floor side by side.

Hermetic: no GPU, no NVML, no checkpoint, no server, no launcher run.
"""

import ast
import os
import pathlib
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner.pp_cut import (
    LAYER_FAMILY_ATTENTION,
    LAYER_FAMILY_LINEAR,
    PhasePoolModel,
    attention_counts,
    family_costs_from_measurement,
)
from sglang.srt.planner.pp_cut_launch import (
    CutCandidate,
    PPCutRefused,
    choose_under_floor,
    derive_pool_floor_from_cut,
    pareto_frontier,
    solve_launch_cut,
)
from sglang.srt.weg2 import DEFAULT_PP_ORDERED_CUT, launcher
from sglang.srt.weg2.launcher import (
    floor_source_for_shipped_line,
    pool_floor_line,
    resolve_pool_floor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

WEG2_DIR = pathlib.Path(launcher.__file__).resolve().parent
PLANNER_DIR = WEG2_DIR.parent / "planner"

#: The order, in the one spelling every assertion below searches for.
ORDER = "39,13,12 mit bs2 im pp layout soll standard werden vorerst"
ORDERED_CUT = (39, 13, 12)
#: The constant this file used to pin, kept ONLY as the defect's reproduction.
THE_OLD_CONSTANT = 448027

# ---------------------------------------------------------------------------
# FRONTIER A: #1286b (SECTION 1am-b, tip 68308f7fae) -- fifteen points.
# ---------------------------------------------------------------------------
FRONTIER_1286B = (
    ((44, 10, 10), (11, 2, 3), 359.0, 304946),
    ((43, 11, 10), (10, 3, 3), 368.3, 354047),
    ((42, 11, 11), (10, 3, 3), 374.3, 374249),
    ((41, 12, 11), (10, 3, 3), 410.8, 394452),
    ((40, 12, 12), (10, 3, 3), 417.6, 414654),
    ((39, 13, 12), (9, 4, 3), 420.1, 481400),
    ((38, 13, 13), (9, 3, 4), 453.4, 503847),
    ((37, 13, 14), (9, 3, 4), 470.3, 526294),
    ((36, 14, 14), (9, 3, 4), 495.9, 548741),
    ((35, 15, 14), (8, 4, 4), 505.2, 640591),
    ((34, 15, 15), (8, 4, 4), 513.5, 664584),
    ((34, 16, 14), (8, 4, 4), 547.8, 665844),
    ((33, 17, 14), (8, 4, 4), 590.3, 691097),
    ((32, 18, 14), (8, 4, 4), 632.9, 715089),
    ((32, 19, 13), (8, 4, 4), 675.4, 716350),
)

# ---------------------------------------------------------------------------
# FRONTIER B: boot weg2sn5pre (2026-09-09, tip 319e76d60b), transcribed from
# its own ``PP-CUT FRONTIER`` line (launcher log weg2sn5pre_sb5d_0909_175833,
# line 51): "pool_floor=448027 objective=makespan 14 non-dominated of 1154
# servable (of 1274 priced)".  The boot SHIPPED 42,11,11 on it.
# ---------------------------------------------------------------------------
FRONTIER_SN5PRE = (
    ((44, 10, 10), (11, 2, 3), 367.2, 387411),
    ((43, 11, 10), (10, 3, 3), 372.3, 444758),
    ((42, 11, 11), (10, 3, 3), 378.2, 463763),
    ((41, 12, 11), (10, 3, 3), 414.9, 482768),
    ((39, 13, 12), (9, 4, 3), 421.4, 578199),
    ((38, 13, 13), (9, 3, 4), 457.4, 599315),
    ((37, 13, 14), (9, 3, 4), 469.9, 620432),
    ((36, 14, 14), (9, 3, 4), 500.0, 641549),
    ((35, 15, 14), (8, 4, 4), 505.1, 745000),
    ((34, 15, 15), (8, 4, 4), 513.1, 768756),
    ((33, 16, 15), (8, 4, 4), 547.7, 792512),
    ((32, 16, 16), (8, 4, 4), 556.4, 802573),
    ((32, 17, 15), (8, 4, 4), 590.2, 816269),
    ((31, 18, 15), (7, 5, 4), 595.4, 841729),
)


def _field(frontier):
    return [
        CutCandidate(layers=layers, attn=attn, makespan_ms=ms, pool_tokens=float(pool))
        for layers, attn, ms, pool in frontier
    ]


def _pool_of(frontier, cut):
    return next(pool for layers, _a, _m, pool in frontier if layers == cut)


def _choose(frontier, floor, objective="makespan"):
    field = _field(frontier)
    return choose_under_floor(
        field,
        objective=objective,
        pool_floor=floor,
        choosable=field,
        cost_provenance="fixture",
    )


def _derive(frontier, cut=ORDERED_CUT):
    field = _field(frontier)
    return derive_pool_floor_from_cut(pareto_frontier(field), field, cut, "fixture")


def _flat(text):
    """Collapse wrapping so a quoted ORDER can be searched for as one string."""
    return " ".join(text.replace("#:", " ").split())


# --------------------------------------------------------------- the fixtures


class TheTwoFixturesAreFrontiers(CustomTestCase):
    def _check(self, frontier):
        field = _field(frontier)
        self.assertEqual(
            [c.layers for c in pareto_frontier(field)], [c.layers for c in field],
            "the transcription must be non-dominated and fastest-first",
        )
        for a, b in zip(frontier, frontier[1:]):
            self.assertLess(a[2], b[2])
            self.assertLess(a[3], b[3])

    def test_1286b(self):
        self._check(FRONTIER_1286B)

    def test_sn5pre(self):
        self._check(FRONTIER_SN5PRE)

    def test_the_interval_that_selects_the_order_moved_between_them(self):
        """The whole finding, in two numbers: 448027 sat inside the first
        interval and BELOW the second one."""
        lo_a = _pool_of(FRONTIER_1286B, (40, 12, 12))
        hi_a = _pool_of(FRONTIER_1286B, ORDERED_CUT)
        lo_b = _pool_of(FRONTIER_SN5PRE, (41, 12, 11))
        hi_b = _pool_of(FRONTIER_SN5PRE, ORDERED_CUT)
        self.assertTrue(lo_a < THE_OLD_CONSTANT <= hi_a)
        self.assertTrue(THE_OLD_CONSTANT <= lo_b < hi_b)


# --------------------------------------------- the rule, on both frontiers


class TheDefaultIsReadOffEachFrontier(CustomTestCase):
    def test_on_1286b_the_floor_is_481400_and_ships_the_order(self):
        floor, source, cut = _derive(FRONTIER_1286B)
        self.assertEqual((floor, source, cut), (481400, "default-from-ordered-cut", ORDERED_CUT))
        self.assertEqual(_choose(FRONTIER_1286B, floor).layers, ORDERED_CUT)

    def test_on_sn5pre_the_floor_is_578199_and_ships_the_order(self):
        floor, source, cut = _derive(FRONTIER_SN5PRE)
        self.assertEqual((floor, source, cut), (578199, "default-from-ordered-cut", ORDERED_CUT))
        self.assertEqual(_choose(FRONTIER_SN5PRE, floor).layers, ORDERED_CUT)

    def test_the_old_constant_ships_42_11_11_on_sn5pre(self):
        """The metal's defect, reproduced against the rule that replaced it."""
        self.assertEqual(_choose(FRONTIER_SN5PRE, THE_OLD_CONSTANT).layers, (42, 11, 11))
        self.assertEqual(_choose(FRONTIER_1286B, THE_OLD_CONSTANT).layers, ORDERED_CUT)

    def test_the_derived_floor_selects_the_order_under_any_uniform_repricing(self):
        """No interval, no tolerance: scale every pool by s and the rule
        still lands on the ordered cut, because the floor scales with it."""
        for s in (0.5, 0.93, 1.0, 1.08, 1.5, 3.0):
            scaled = tuple((l, a, ms, int(pool * s)) for l, a, ms, pool in FRONTIER_SN5PRE)
            floor, _src, _cut = _derive(scaled)
            self.assertEqual(floor, _pool_of(scaled, ORDERED_CUT))
            self.assertEqual(_choose(scaled, floor).layers, ORDERED_CUT, s)

    def test_a_flag_outranks_the_derivation(self):
        """An explicit floor is the operator's number; the cut it selects is
        whatever the frontier says, including a neighbour."""
        self.assertEqual(_choose(FRONTIER_SN5PRE, 500000).layers, ORDERED_CUT)
        self.assertEqual(_choose(FRONTIER_SN5PRE, 460000).layers, (42, 11, 11))
        self.assertEqual(_choose(FRONTIER_SN5PRE, None).layers, (44, 10, 10))


# --------------------------------------------------- off the frontier: W67


class OffTheFrontierIsW67(CustomTestCase):
    def test_dominated_is_refused_by_name_and_names_the_dominator(self):
        # A synthetic cut that holds more pool in less time than 39,13,12:
        # 39,13,12 is then priced, servable and DOMINATED.
        frontier = FRONTIER_SN5PRE + (((40, 12, 12), (10, 3, 3), 400.0, 600000),)
        with self.assertRaises(PPCutRefused) as cm:
            _derive(frontier)
        msg = str(cm.exception)
        self.assertIn("W67 Weg2PPCutOrderedCutOffFrontier", msg)
        self.assertIn("DOMINATED", msg)
        self.assertIn("40,12,12", msg)
        self.assertIn("--pp-solve-pool-floor 0", msg)
        self.assertIn(ORDER, _flat(msg))

    def test_unpriced_is_refused_by_name(self):
        frontier = tuple(p for p in FRONTIER_SN5PRE if p[0] != ORDERED_CUT)
        with self.assertRaises(PPCutRefused) as cm:
            _derive(frontier)
        msg = str(cm.exception)
        self.assertIn("W67 Weg2PPCutOrderedCutOffFrontier", msg)
        self.assertIn("UNPRICED", msg)
        self.assertIn("FRONTIER", msg)

    def test_it_never_ships_a_neighbour(self):
        """The refusal is the whole answer: no floor, no cut comes back."""
        frontier = tuple(p for p in FRONTIER_SN5PRE if p[0] != ORDERED_CUT)
        self.assertRaises(PPCutRefused, _derive, frontier)
        # ... while an explicit floor on the same field still works: the flag
        # is the way past the refusal, and it is named in the message.
        self.assertEqual(_choose(frontier, 500000).layers, (38, 13, 13))

    def test_the_code_is_the_next_free_one_and_lives_in_the_planner(self):
        src = (PLANNER_DIR / "pp_cut_launch.py").read_text()
        self.assertEqual(src.count("W67 Weg2PPCutOrderedCutOffFrontier"), 1)


# ------------------------------------- the solver on THIS boot's own inputs


#: weg2sn5pre, launcher log weg2sn5pre_sb5d_0909_175833 lines 46-48, verbatim:
#: ``PP-CUT inputs: layers=64 attn=16 kv=2048 B/token/attn-layer ... mean 363.4
#: used; free=[27936, 16992, 16720] MiB ... arming floor 1229.0 MiB/rank`` and
#: ``PP-CUT budget posts ...: stage_fixed 2342.0,1105.5,3518.0; corridor
#: holdback 0.0; mamba 1.5588/linear-layer/slot x 5 slots; speculative
#: intermediate 0.0; prefill activation reserve 1024.0; mamba pre-capture
#: reserve 0.0; page_size 1``.  The budgets are the corridor law's MEASURED-P
#: output for that boot (floors 1055/1095/858) -- an INPUT here, transcribed.
SN5PRE_BUDGETS_MIB = (27936.0, 16992.0, 16720.0)
SN5PRE_MEAN_LAYER_MIB = 363.4
SN5PRE_STAGE_FIXED_MIB = (2342.0, 1105.5, 3518.0)
SN5PRE_MAMBA_SLOTS = 5
SN5PRE_CORRIDOR_HOLDBACK_MIB = 0.0
SN5PRE_ACTIVATION_RESERVE_MIB = 1024.0
KV_MIB_PER_TOKEN_PER_ATTN_LAYER = 2048.0 / (1024.0 * 1024.0)
ARMING_FLOOR_MIB = 1229.0
MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT = 1.5588
CAP_TOKENS = 262144
FAMILIES = tuple(
    LAYER_FAMILY_ATTENTION if (i % 4) == 3 else LAYER_FAMILY_LINEAR for i in range(64)
)
MEASURED_MS_PER_LAYER = (8.10, 35.16, 33.59)
INCUMBENT = (32, 18, 14)
HERMETIC_CARDS = ("HERMETIC-TEST-CARD-A", "HERMETIC-TEST-CARD-B", "HERMETIC-TEST-CARD-C")
PAIR_MS = {(0, 1): 0.0, (1, 2): 0.0}
#: The boot's OWN priced pools for three landmark cuts (PP-CUT FRONTIER line).
SN5PRE_PRICED = {(42, 11, 11): 463763, (41, 12, 11): 482768, (39, 13, 12): 578199}
#: THE TRANSCRIPTION'S RESOLUTION, stated: the log prints the mean layer
#: weight rounded to 0.1 MiB (``mean 363.4 used``; the per-family figures on
#: the same boot's RING-CKPT line, 355.13 / 366.15, give 363.395), and
#: 0.05 MiB x 42 layers on the binding rank is ~11 tokens at 20480 B/token.
#: Measured on the remote desk (368e85278d): 463758 vs 463763, 578194 vs
#: 578199 -- 5 tokens, the rounding and nothing else.  So pools are asserted
#: to within this many tokens; the CUT and the floor-equals-chosen-pool
#: identity are asserted exactly.
POOL_TRANSCRIPTION_TOL = 16


def _sn5pre_model():
    return PhasePoolModel(
        free_mib=SN5PRE_BUDGETS_MIB,
        weight_mib_per_layer=SN5PRE_MEAN_LAYER_MIB,
        kv_mib_per_token_per_attn_layer=KV_MIB_PER_TOKEN_PER_ATTN_LAYER,
        arming_floor_mib=tuple(ARMING_FLOOR_MIB for _ in SN5PRE_BUDGETS_MIB),
        mamba_mib_per_linear_layer_per_slot=MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT,
        mamba_slots=SN5PRE_MAMBA_SLOTS,
        stage_fixed_mib=SN5PRE_STAGE_FIXED_MIB,
        activation_reserve_mib=SN5PRE_ACTIVATION_RESERVE_MIB,
        corridor_holdback_mib=SN5PRE_CORRIDOR_HOLDBACK_MIB,
        zero_posts_acknowledged=(
            "mamba pre-capture reserve",
            "speculative intermediate state",
            "GGUF dequant scratch",
        ),
        page_size=1,
    )


def _family_cost():
    cost, _prov = family_costs_from_measurement(
        measured_ms_per_layer=MEASURED_MS_PER_LAYER,
        measured_counts=INCUMBENT,
        measured_attn_counts=attention_counts(FAMILIES, INCUMBENT),
        chunk_tokens=4096,
        ref_prefix_tokens=4096.0,
        anchor_stage=1,
        anchor_attn_ms_per_layer=400.0,
        anchor_prefix_tokens=262144.0,
    )
    return cost


def _solve(**kw):
    params = dict(
        layer_families=FAMILIES,
        incumbent_layers=INCUMBENT,
        measured_ms_per_layer=MEASURED_MS_PER_LAYER,
        measured_provenance="hermetic fixture: weg2sn5pre's own PP-CUT inputs",
        card_names=HERMETIC_CARDS,
        pool_model=_sn5pre_model(),
        cap_tokens=CAP_TOKENS,
        family_cost=_family_cost(),
        design_prefix_tokens=4096,
        per_pair_crossing_ms=PAIR_MS,
        enumerate_gapped=False,
        objective="makespan",
    )
    params.update(kw)
    return solve_launch_cut(**params)


class TheSolverOnThisBootsInputs(CustomTestCase):
    """Layer 2: the desk solver path on weg2sn5pre's own census budgets."""

    def test_the_pool_axis_is_the_boots_own_arithmetic(self):
        """The three landmark pools the boot printed, to the token."""
        decision = _solve(pool_floor=None, pool_floor_from_cut=None)
        by_cut = {c.layers: int(c.pool_tokens) for c in decision.servable if c.kind == "contiguous"}
        for cut, pool in SN5PRE_PRICED.items():
            self.assertIsNotNone(by_cut.get(cut), cut)
            self.assertLessEqual(abs(by_cut[cut] - pool), POOL_TRANSCRIPTION_TOL, (cut, by_cut[cut], pool))

    def test_the_default_derives_the_floor_from_the_order_and_ships_it(self):
        decision = _solve(pool_floor=None, pool_floor_from_cut=DEFAULT_PP_ORDERED_CUT)
        self.assertEqual(tuple(decision.chosen.layers), ORDERED_CUT)
        self.assertLessEqual(abs(decision.pool_floor - SN5PRE_PRICED[ORDERED_CUT]), POOL_TRANSCRIPTION_TOL)
        self.assertEqual(decision.pool_floor, int(decision.chosen.pool_tokens))
        self.assertEqual(decision.pool_floor_source, "default-from-ordered-cut")
        self.assertEqual(decision.pool_floor_cut, ORDERED_CUT)
        self.assertIn(ORDERED_CUT, [tuple(c.layers) for c in decision.frontier])

    def test_the_old_constant_ships_42_11_11_on_the_same_inputs(self):
        """The boot's defect on the desk: same inputs, the constant, 42,11,11."""
        decision = _solve(pool_floor=THE_OLD_CONSTANT, pool_floor_from_cut=None)
        self.assertEqual(tuple(decision.chosen.layers), (42, 11, 11))
        self.assertEqual(decision.pool_floor_source, "flag")

    def test_off_is_the_unfloored_makespan_winner(self):
        decision = _solve(pool_floor=None, pool_floor_from_cut=None)
        self.assertEqual(decision.pool_floor, None)
        self.assertEqual(decision.pool_floor_source, "none")
        self.assertNotEqual(tuple(decision.chosen.layers), ORDERED_CUT)
        self.assertLess(int(decision.chosen.pool_tokens), SN5PRE_PRICED[ORDERED_CUT] - POOL_TRANSCRIPTION_TOL)

    def test_a_pin_is_judged_against_the_derived_floor_too(self):
        """A pinned cut below the order's floor is the W40 it always was."""
        with self.assertRaises(PPCutRefused) as cm:
            _solve(
                pool_floor=None,
                pool_floor_from_cut=DEFAULT_PP_ORDERED_CUT,
                pinned_layers=(44, 10, 10),
            )
        self.assertIn("W40", str(cm.exception))
        self.assertIn("--pp-solve-pool-floor", str(cm.exception))


# -------------------------------------------------------- the launcher seam


class TheLauncherSeam(CustomTestCase):
    def test_absent_hands_the_cut_to_the_solver_and_states_the_rule(self):
        floor, cut, rule = resolve_pool_floor(None)
        self.assertIsNone(floor)
        self.assertEqual(cut, ORDERED_CUT)
        self.assertEqual(cut, tuple(DEFAULT_PP_ORDERED_CUT))
        self.assertIn("source=default-from-ordered-cut 39,13,12", rule)
        self.assertIn(ORDER, _flat(rule))
        self.assertIn("W67", rule)

    def test_zero_is_off_and_negative_is_off(self):
        for v in (0, -1):
            floor, cut, prov = resolve_pool_floor(v)
            self.assertIsNone(floor)
            self.assertIsNone(cut)
            self.assertIn("source=flag", prov)
            self.assertIn("is NOT in force", prov)

    def test_a_positive_flag_is_the_operators_number(self):
        floor, cut, prov = resolve_pool_floor(500000)
        self.assertEqual((floor, cut), (500000, None))
        self.assertIn("source=flag", prov)

    class _Decision:
        def __init__(self, floor, source, cut, row):
            self.pool_floor = floor
            self.pool_floor_source = source
            self.pool_floor_cut = cut
            self.kv_floor = row
            self.chosen = row

    class _Row:
        layers = ORDERED_CUT
        attn = (9, 4, 3)
        pool_tokens = 578199
        makespan_ms = 421.4

        def fmt(self):
            return "39,13,12 attn 9,4,3"

    def test_the_floor_line_is_rendered_from_the_decision(self):
        row = self._Row()
        line = pool_floor_line(self._Decision(578199, "default-from-ordered-cut", ORDERED_CUT, row))
        self.assertIn(
            "PP-CUT POOL FLOOR: pool_floor=578199 source=default-from-ordered-cut "
            "39,13,12 (frontier of this boot)",
            line,
        )
        self.assertIn(ORDER, _flat(line))
        flag = pool_floor_line(self._Decision(500000, "flag", None, row))
        self.assertIn("PP-CUT POOL FLOOR: pool_floor=500000 source=flag", flag)
        off = pool_floor_line(self._Decision(None, "none", None, row))
        self.assertIn("PP-CUT POOL FLOOR: pool_floor=none source=none", off)

    def test_the_shipped_line_carries_the_source(self):
        row = self._Row()
        decision = self._Decision(578199, "default-from-ordered-cut", ORDERED_CUT, row)
        source = floor_source_for_shipped_line(decision)
        self.assertTrue(source.startswith("default-from-ordered-cut 39,13,12"))
        line = launcher.shipped_line(
            decision, row, "the floored makespan cut", row, row,
            incumbent_fallback="n/a", floor_source=source,
        )
        self.assertIn("layers=39,13,12", line)
        self.assertIn("pool_floor=578199", line)
        self.assertIn("pool_floor_source=default-from-ordered-cut 39,13,12", line)
        self.assertIn(ORDER, _flat(line))

    def test_the_source_is_required_not_defaulted(self):
        row = self._Row()
        with self.assertRaises(TypeError):
            launcher.shipped_line(
                self._Decision(578199, "flag", None, row), row, "why", row, row,
                incumbent_fallback="n/a",
            )

    def test_the_rule_is_logged_before_the_solve_and_the_number_after(self):
        import inspect

        src = inspect.getsource(launcher.solve_p_cut)
        self.assertLess(src.index("PP-CUT POOL FLOOR RULE"), src.index("solve_launch_cut("))
        self.assertGreater(src.index("log(pool_floor_line(decision))"), src.index("solve_launch_cut("))
        self.assertIn("pool_floor_from_cut=pool_floor_from_cut", src)
        self.assertIn("resolve_pool_floor(", src)
        self.assertIn("floor_source_for_shipped_line(decision)", src)

    def test_the_flag_is_unset_by_default_so_the_source_is_knowable(self):
        ns = launcher.build_parser().parse_args(["--tree", "/t", "--tag", "x"])
        self.assertIsNone(ns.pp_solve_pool_floor)

    def test_the_help_quotes_the_order_the_cut_and_the_refusal(self):
        h = _flat(launcher.build_parser().format_help())
        self.assertIn(ORDER, h)
        self.assertIn("DEFAULT_PP_ORDERED_CUT", h)
        self.assertIn("W67", h)
        self.assertNotIn("UNSET (the default) = no floor", h)
        self.assertNotIn("448027", h)


# ---------------------------------------------- item 2: the cap is the pool


class TheCapIsTheShippedPool(CustomTestCase):
    def _argv(self, **kw):
        return launcher.argv_p(
            "py", "/nonexistent/model", [1, 1, 1], 1, 1,
            launcher.RING_FORM_SENTINEL_STORE_CFG, [], **kw,
        )

    def test_the_cap_is_the_number_handed_in(self):
        argv = self._argv(draft_kv_on_p=True, p_max_total_tokens=578199)
        self.assertEqual(argv[argv.index("--max-total-tokens") + 1], "578199")
        # it rides right behind the head's flags, head-scoped as before
        self.assertLess(argv.index("--speculative-draft-kv-only"), argv.index("--max-total-tokens"))

    def test_no_pool_no_cap_and_no_head_no_cap(self):
        self.assertNotIn("--max-total-tokens", self._argv(draft_kv_on_p=True))
        self.assertNotIn("--max-total-tokens", self._argv(draft_kv_on_p=False, p_max_total_tokens=578199))

    def test_main_hands_the_shipped_pool_to_both_argv_builds(self):
        import inspect

        src = inspect.getsource(launcher.main)
        self.assertEqual(src.count("p_max_total_tokens=int(cut.pool_tokens)"), 2)

    def test_the_provenance_line_prints_cap_and_floor(self):
        import inspect

        src = inspect.getsource(launcher.solve_p_cut)
        self.assertIn("PP-CUT P-CAP: --max-total-tokens=%d", src)
        self.assertIn("pool_floor=%s", src)

    def test_the_constant_is_gone(self):
        src = (WEG2_DIR / "launcher.py").read_text()
        tree = ast.parse(src)
        ints = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == 428000]
        self.assertEqual(ints, [], f"the weg2zr2 cap is restated at {ints}")
        names = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id in ("P_MAX_TOTAL_TOKENS", "derive_p_max_total_tokens")
        ]
        self.assertEqual(names, [])


# ------------------------------------------------------ the singleness pin


class TheCutIsWrittenExactlyOnce(CustomTestCase):
    """The AST pin, in the shape ``DEFAULT_P_BS`` uses on its own branch."""

    def _sources(self):
        for d in (WEG2_DIR, PLANNER_DIR):
            for path in sorted(d.glob("*.py")):
                yield path, ast.parse(path.read_text(), filename=str(path))

    def test_the_cut_is_assigned_exactly_once(self):
        sites = []
        for path, tree in self._sources():
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id == "DEFAULT_PP_ORDERED_CUT":
                        sites.append(f"{path.name}:{node.lineno}")
        self.assertEqual(sites, ["__init__.py:%d" % _constant_lineno()], sites)

    def test_the_old_constant_and_its_name_are_gone_from_both_packages(self):
        found = []
        for path, tree in self._sources():
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and node.value == THE_OLD_CONSTANT:
                    found.append(f"{path.name}:{node.lineno}")
                if isinstance(node, ast.Name) and node.id == "DEFAULT_PP_SOLVE_POOL_FLOOR":
                    found.append(f"{path.name}:{node.lineno} (name)")
        self.assertEqual(found, [], f"the floor is restated at {found}")

    def test_no_pool_floor_default_anywhere_is_a_bare_int_literal(self):
        bad = []
        for path, tree in self._sources():
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                args = node.args
                pairs = list(
                    zip(args.args[len(args.args) - len(args.defaults):], args.defaults)
                ) + [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
                for arg, default in pairs:
                    if arg.arg not in ("pool_floor", "pool_floor_from_cut"):
                        continue
                    if isinstance(default, ast.Constant) and isinstance(default.value, int):
                        bad.append(f"{path.name}:{node.lineno} {node.name}")
        self.assertEqual(bad, [], f"bare numeric pool_floor defaults at {bad}")

    def test_the_argparse_default_stays_none(self):
        found = None
        tree = ast.parse((WEG2_DIR / "launcher.py").read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "add_argument":
                continue
            flags = {a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
            if "--pp-solve-pool-floor" not in flags:
                continue
            found = next((k.value for k in node.keywords if k.arg == "default"), "<absent>")
        self.assertIsNotNone(found, "the flag disappeared")
        self.assertTrue(isinstance(found, ast.Constant) and found.value is None)

    def test_the_order_and_the_rule_are_recorded_where_the_cut_lives(self):
        src = (WEG2_DIR / "__init__.py").read_text()
        head = src[: src.index("DEFAULT_PP_ORDERED_CUT = ")]
        block = _flat(head[head.rindex("#: K3 -- THE ORDERED P CUT"):])
        self.assertIn(ORDER, block)
        self.assertIn("W67", block)
        self.assertIn("weg2sn5pre", block)


def _constant_lineno():
    tree = ast.parse((WEG2_DIR / "__init__.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "DEFAULT_PP_ORDERED_CUT":
                    return node.lineno
    raise AssertionError("DEFAULT_PP_ORDERED_CUT is gone")


if __name__ == "__main__":
    unittest.main()
