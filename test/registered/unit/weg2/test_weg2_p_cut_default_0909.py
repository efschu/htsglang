# SPDX-License-Identifier: Apache-2.0
"""The P cut ships 39,13,12 BY DEFAULT -- and it is still SOLVED, not pinned.

USER ORDER, 2026-09-09, verbatim: "39,13,12 mit bs2 im pp layout soll standard
werden vorerst."

THE DISTINCTION THIS WHOLE FILE EXISTS TO PROTECT.  The cheap way to obey that
order is ``--pp-stage-ratio 39,13,12`` -- a hand pin -- and it is the wrong
way, because a pin keeps shipping 39,13,12 after the thing that made it right
has changed: another card, another checkpoint, another measured ms/layer.  The
solver stays the authority.  What the order changes is a CONSTRAINT: the
shipped default of ``--pp-solve-pool-floor`` (``DEFAULT_PP_SOLVE_POOL_FLOOR``,
448027 world KV tokens), under which ``--pp-solve-objective makespan`` -- itself
a user default since 2026-09-08 -- reads "the FASTEST cut that still holds
448,027 tokens".  On this rig's frontier that is 39,13,12; on a re-priced
frontier it is whatever the solver then says, which is the point.

THE FLOOR'S RULE, so a re-measure can redo it instead of copying it.  From the
``PP-CUT FRONTIER`` line of #1286b (SECTION 1am-b), the two points bracketing
the order are 40,12,12 at pool 414654 (417.6 ms, the next FASTER cut) and
39,13,12 at pool 481400 (420.1 ms).  Under "fastest above F" the order is
selected for any F in (414654, 481400], and the shipped floor is that
interval's MIDPOINT -- the value furthest from both boundaries it must not
cross (8.05 % of headroom down, 6.93 % up).  The obvious alternative, "the
chosen pool minus the priced-vs-realised tolerance", is REJECTED and the
rejection is asserted below: that tolerance is 0.10 % measured, so such a floor
would carry a downward margin exactly the size of its own measurement error.

WHY THE FIFTEEN-POINT FRONTIER IS A SOUND STAND-IN for the 932-candidate field,
which is the one step in this file that is an argument and not a lookup: under
"fastest above F" the winner is ALWAYS on the Pareto frontier.  If a candidate
c clears F and is dominated by c' (pool at least as large, total_ms no larger),
then c' clears F too and is at least as fast, so the argmin's (ms, pool) pair
is never improved by looking outside the frontier.  Restricting the fixture to
the 15 non-dominated points therefore cannot change the answer to THIS
question.  It would not be sound for a question about the dominated field --
none is asked here.

WHAT IS DRIVEN IS THE SHIPPED RULE, not a restatement of it.  ``choose_under_
floor`` is the function the boot calls; this file hands it the real frontier
and reads back the cut.  A test that re-implemented "min total_ms over pool >=
F" would pass against a launcher that had stopped doing it.

Hermetic: no GPU, no NVML, no checkpoint, no server, no launcher run.  The AST
half reads source text and imports nothing at all.
"""

import ast
import os
import pathlib
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner.pp_cut_launch import (
    CutCandidate,
    PPCutRefused,
    choose_under_floor,
)
from sglang.srt.weg2 import launcher
from sglang.srt.weg2.launcher import (
    DEFAULT_PP_SOLVE_POOL_FLOOR,
    resolve_pool_floor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WEG2_DIR = pathlib.Path(launcher.__file__).resolve().parent
PLANNER_DIR = WEG2_DIR.parent / "planner"

# ---------------------------------------------------------------------------
# THE FRONTIER, transcribed from the ``PP-CUT FRONTIER`` line #1286b published
# in SECTION 1am-b of /spinning/gpu-arb/weg2/WEG2_BUILD_DECISIONS_0906.md:
#
#   PP-CUT FRONTIER: pool_floor=none objective=makespan 15 non-dominated of 932
#   servable (of 932 priced), fastest first: ...
#
# ``total_ms`` there is makespan + crossing.  The candidates below carry it as
# ``makespan_ms`` with ``crossing_ms`` left at its 0.0 default, so ``total_ms``
# -- the only column the ranking reads -- equals the published figure.  The
# decomposition is not under test here; the ORDERING RULE is.
# ---------------------------------------------------------------------------

FRONTIER = (
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

#: THE ORDER, and the cut immediately faster than it -- the two that define the
#: interval the floor has to land inside.
ORDERED_CUT = (39, 13, 12)
NEXT_FASTER_CUT = (40, 12, 12)
#: The unfloored makespan winner: what ships with the floor OFF.
UNFLOORED_WINNER = (44, 10, 10)
#: The largest pool anywhere on the frontier -- a floor above it cannot be met.
FIELD_MAX_POOL = 716350


def _field():
    return [
        CutCandidate(
            layers=layers, attn=attn, makespan_ms=ms, pool_tokens=float(pool)
        )
        for layers, attn, ms, pool in FRONTIER
    ]


def _pool_of(cut):
    return next(pool for layers, _a, _m, pool in FRONTIER if layers == cut)


def _choose(floor, objective="makespan"):
    field = _field()
    return choose_under_floor(
        field,
        objective=objective,
        pool_floor=floor,
        choosable=field,
        cost_provenance="fixture: SECTION 1am-b PP-CUT FRONTIER",
    )


# --------------------------------------------------------------- the fixture


class TheFixtureIsAFrontier(CustomTestCase):
    """Before anything is asserted with the 15 points, the 15 are checked.

    A transcribed fixture that had lost a digit would still let every selection
    test below pass for the wrong reason, so the two properties the argument in
    the module docstring rests on are asserted rather than assumed.
    """

    def test_it_is_sorted_fastest_first_and_pool_rises_with_ms(self):
        for a, b in zip(FRONTIER, FRONTIER[1:]):
            self.assertLess(a[2], b[2], f"{a[0]} should be faster than {b[0]}")
            self.assertLess(a[3], b[3], f"{a[0]} should hold less than {b[0]}")

    def test_no_point_dominates_another(self):
        """The monotone chain above IS non-domination; stated as its own claim."""
        for i, (li, _ai, mi, pi) in enumerate(FRONTIER):
            for j, (lj, _aj, mj, pj) in enumerate(FRONTIER):
                if i == j:
                    continue
                self.assertFalse(
                    pj >= pi and mj <= mi,
                    f"{lj} would dominate {li}, so this is not a frontier",
                )

    def test_the_two_bracketing_pools_are_the_ones_the_rule_used(self):
        self.assertEqual(_pool_of(NEXT_FASTER_CUT), 414654)
        self.assertEqual(_pool_of(ORDERED_CUT), 481400)


# ------------------------------------------------------------- the selection


class TheDefaultSelectsTheOrderedCut(CustomTestCase):
    def test_the_shipped_default_ships_39_13_12(self):
        """THE ORDER, driven through the shipped selection function."""
        chosen = _choose(DEFAULT_PP_SOLVE_POOL_FLOOR)
        self.assertEqual(
            chosen.layers,
            ORDERED_CUT,
            "user order 2026-09-09: '39,13,12 mit bs2 im pp layout soll "
            "standard werden vorerst'",
        )
        self.assertEqual(chosen.attn, (9, 4, 3))
        self.assertEqual(int(chosen.pool_tokens), 481400)

    def test_the_default_is_the_midpoint_of_the_selecting_interval(self):
        """The RULE, not just the number: any F in (414654, 481400] works."""
        low, high = _pool_of(NEXT_FASTER_CUT), _pool_of(ORDERED_CUT)
        self.assertGreater(DEFAULT_PP_SOLVE_POOL_FLOOR, low)
        self.assertLessEqual(DEFAULT_PP_SOLVE_POOL_FLOOR, high)
        self.assertEqual(DEFAULT_PP_SOLVE_POOL_FLOOR, (low + high) // 2)

    def test_both_interval_edges_are_live_boundaries(self):
        """One token either side of the interval selects a DIFFERENT cut.

        This is what makes the midpoint rule worth anything: the boundaries are
        real, so distance from them is real margin and not decoration.
        """
        self.assertEqual(_choose(_pool_of(NEXT_FASTER_CUT)).layers, NEXT_FASTER_CUT)
        self.assertEqual(_choose(_pool_of(NEXT_FASTER_CUT) + 1).layers, ORDERED_CUT)
        self.assertEqual(_choose(_pool_of(ORDERED_CUT)).layers, ORDERED_CUT)
        self.assertEqual(_choose(_pool_of(ORDERED_CUT) + 1).layers, (38, 13, 13))

    def test_the_rejected_rule_is_named_and_is_worse(self):
        """"chosen pool minus the 0.10 % tolerance" -- it selects, but barely.

        It is not wrong today; it is BRITTLE, and the assertion is exactly
        that: its downward margin is the same size as the measurement error it
        was derived from, while the shipped rule's is two orders larger.
        """
        tolerance_rule = int(_pool_of(ORDERED_CUT) * 0.999)
        self.assertEqual(_choose(tolerance_rule).layers, ORDERED_CUT)
        low = _pool_of(NEXT_FASTER_CUT)
        margin_tolerance = (tolerance_rule - low) / low
        margin_shipped = (DEFAULT_PP_SOLVE_POOL_FLOOR - low) / low
        self.assertGreater(margin_shipped, 20 * margin_tolerance)

    def test_a_lower_default_would_ship_a_different_cut(self):
        """THE MUTANT, pinned as a test rather than only run by hand.

        Lower the floor below 40,12,12's pool and the order stops being
        obeyed -- so a future edit that "rounds the constant down a bit"
        cannot pass this file.
        """
        self.assertNotEqual(_choose(_pool_of(NEXT_FASTER_CUT) - 1).layers, ORDERED_CUT)
        self.assertNotEqual(_choose(400000).layers, ORDERED_CUT)


class TheFloorOffIsExactlyTheOldBehaviour(CustomTestCase):
    def test_no_floor_selects_the_unfloored_makespan_winner(self):
        self.assertEqual(_choose(None).layers, UNFLOORED_WINNER)

    def test_zero_resolves_to_no_floor_at_all(self):
        """0 is the OFF switch, and it resolves to None -- not to a floor of 0.

        Both would select the same cut, so the selection alone cannot tell them
        apart; the difference is that a floor of 0 would print
        ``pool_floor=0``, i.e. an instrument claiming a floor is armed when
        none is.
        """
        floor, prov = resolve_pool_floor(0)
        self.assertIsNone(floor)
        self.assertIn("pool_floor=none", prov)
        self.assertIn("source=flag", prov)
        self.assertEqual(_choose(floor).layers, UNFLOORED_WINNER)

    def test_off_is_byte_identical_to_the_pre_1286b_choice(self):
        """Not just the same cut: the same candidate, on every priced axis."""
        off = _choose(None)
        pre_1286b = min(_field(), key=lambda c: (c.total_ms, -c.pool_tokens))
        self.assertEqual(off.layers, pre_1286b.layers)
        self.assertEqual(off.attn, pre_1286b.attn)
        self.assertEqual(off.total_ms, pre_1286b.total_ms)
        self.assertEqual(off.pool_tokens, pre_1286b.pool_tokens)

    def test_negative_is_off_too_rather_than_a_floor_below_zero(self):
        self.assertIsNone(resolve_pool_floor(-1)[0])


class AFloorNothingClearsIsARefusal(CustomTestCase):
    def test_above_the_whole_field_it_refuses_by_name(self):
        with self.assertRaises(PPCutRefused) as caught:
            _choose(FIELD_MAX_POOL + 1)
        msg = str(caught.exception)
        self.assertIn("W40", msg)
        self.assertIn("--pp-solve-pool-floor", msg)
        self.assertIn(str(FIELD_MAX_POOL), msg)

    def test_it_never_falls_back_to_the_fastest_cut_below(self):
        """The load-bearing direction: no silent degradation."""
        with self.assertRaises(PPCutRefused):
            _choose(FIELD_MAX_POOL + 1)

    def test_the_pool_maximal_candidate_is_the_one_it_names(self):
        with self.assertRaises(PPCutRefused) as caught:
            _choose(FIELD_MAX_POOL + 1)
        self.assertIn("32,19,13", str(caught.exception))


# ------------------------------------------------------- value and provenance


class TheDefaultReachesTheBootThroughTheParser(CustomTestCase):
    def test_the_flag_is_unset_by_default_so_the_source_is_knowable(self):
        """``default=None`` is deliberate: it is how "flag absent" stays visible.

        The VALUE default lives in ``resolve_pool_floor``, not in argparse,
        because a boot has to publish whether 448027 came from the order or
        from an operator -- and an argparse default erases that distinction.
        """
        ns = launcher.build_parser().parse_args(["--tree", "/t", "--tag", "x"])
        self.assertIsNone(ns.pp_solve_pool_floor)

    def test_absent_resolves_to_the_constant_and_says_default(self):
        floor, prov = resolve_pool_floor(None)
        self.assertEqual(floor, DEFAULT_PP_SOLVE_POOL_FLOOR)
        self.assertEqual(floor, 448027)
        self.assertIn("source=default", prov)
        self.assertIn("39,13,12 mit bs2 im pp layout soll standard werden vorerst", prov)

    def test_an_operator_value_outranks_the_default_and_says_flag(self):
        floor, prov = resolve_pool_floor(500000)
        self.assertEqual(floor, 500000)
        self.assertIn("source=flag", prov)
        self.assertIn("is NOT in force", prov)

    def test_the_order_is_quoted_in_the_flag_help(self):
        h = launcher.build_parser().format_help()
        self.assertIn("39,13,12 mit bs2 im pp layout soll standard werden vorerst", h)
        self.assertIn(str(DEFAULT_PP_SOLVE_POOL_FLOOR), h)

    def test_the_help_no_longer_claims_unset_means_no_floor(self):
        """Instrument-text-lies, class A -- the sentence that #1286b left true
        and this branch made false."""
        h = launcher.build_parser().format_help()
        self.assertNotIn("UNSET (the default) = no floor", h)


class TheBootRecordPublishesTheFloorAndItsSource(CustomTestCase):
    class _Row:
        layers = ORDERED_CUT
        attn = (9, 4, 3)
        pool_tokens = 481400
        makespan_ms = 420.1

        def fmt(self):
            return "39,13,12 attn 9,4,3"

    class _Decision:
        pool_floor = 448027

        def __init__(self, row):
            self.kv_floor = row
            self.chosen = row

    def test_the_shipped_line_renders_and_carries_both(self):
        row = self._Row()
        line = launcher.shipped_line(
            self._Decision(row),
            row,
            "the floored makespan cut",
            row,
            row,
            incumbent_fallback="n/a",
            floor_source="default (user order 2026-09-09)",
        )
        self.assertIn("layers=39,13,12", line)
        self.assertIn("pool_floor=448027", line)
        self.assertIn("pool_floor_source=default", line)
        self.assertIn("user order 2026-09-09", line)

    def test_the_source_is_required_not_defaulted(self):
        """A caller that forgets it must fail at the call site, loudly."""
        row = self._Row()
        with self.assertRaises(TypeError):
            launcher.shipped_line(
                self._Decision(row), row, "why", row, row, incumbent_fallback="n/a"
            )

    def test_solve_p_cut_logs_the_floor_line_before_it_solves(self):
        """A W40 must name the floor it measured against, so the line is
        emitted whether the solve then succeeds or refuses."""
        import inspect

        src = inspect.getsource(launcher.solve_p_cut)
        self.assertIn("PP-CUT POOL FLOOR", src)
        self.assertLess(
            src.index("PP-CUT POOL FLOOR"),
            src.index("solve_launch_cut("),
            "the provenance line must be logged BEFORE the solve that can refuse",
        )
        self.assertIn("resolve_pool_floor(ns.pp_solve_pool_floor)", src)


# ------------------------------------------------------ the singleness pin


class TheNumberIsWrittenExactlyOnce(CustomTestCase):
    """The AST pin, in the shape ``DEFAULT_P_BS`` uses on its own branch.

    The order says "vorerst" -- provisional, explicitly re-measurable -- so the
    cost of the eventual re-measure is exactly the number of places that carry
    the value.  This keeps that number at one.  It is an AST read of the actual
    default EXPRESSION, not a grep, so a literal cannot hide behind formatting,
    a line break or a comment.
    """

    def _sources(self):
        for d in (WEG2_DIR, PLANNER_DIR):
            for path in sorted(d.glob("*.py")):
                yield path, ast.parse(path.read_text(), filename=str(path))

    def test_the_constant_is_assigned_exactly_once(self):
        sites = []
        for path, tree in self._sources():
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for t in targets:
                    if isinstance(t, ast.Name) and t.id == "DEFAULT_PP_SOLVE_POOL_FLOOR":
                        sites.append(f"{path.name}:{node.lineno}")
        self.assertEqual(sites, ["launcher.py:%d" % _constant_lineno()], sites)

    def test_the_literal_appears_nowhere_else_in_either_package(self):
        """448027 as a bare number anywhere but its own assignment is a copy."""
        found = []
        for path, tree in self._sources():
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Constant) and node.value == 448027):
                    continue
                if path.name == "launcher.py" and node.lineno == _constant_lineno():
                    continue
                found.append(f"{path.name}:{node.lineno}")
        self.assertEqual(found, [], f"the floor is restated at {found}")

    def test_no_pool_floor_default_anywhere_is_a_bare_int_literal(self):
        """Signature defaults for a ``pool_floor`` parameter must be None.

        The value default belongs to ``resolve_pool_floor`` alone; a function
        that quietly defaulted its own ``pool_floor`` to a number would be a
        second, unattributable floor.
        """
        bad = []
        for path, tree in self._sources():
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                args = node.args
                pairs = list(
                    zip(args.args[len(args.args) - len(args.defaults) :], args.defaults)
                ) + [
                    (a, d)
                    for a, d in zip(args.kwonlyargs, args.kw_defaults)
                    if d is not None
                ]
                for arg, default in pairs:
                    if arg.arg != "pool_floor":
                        continue
                    if isinstance(default, ast.Constant) and isinstance(
                        default.value, int
                    ):
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
            flags = {
                a.value
                for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            }
            if "--pp-solve-pool-floor" not in flags:
                continue
            found = next(
                (k.value for k in node.keywords if k.arg == "default"), "<absent>"
            )
        self.assertIsNotNone(found, "the flag disappeared")
        self.assertTrue(
            isinstance(found, ast.Constant) and found.value is None,
            "argparse must keep default=None so 'flag absent' stays detectable; "
            "the VALUE default lives in resolve_pool_floor",
        )

    def test_the_order_is_recorded_where_the_number_lives(self):
        src = (WEG2_DIR / "launcher.py").read_text()
        head = src[: src.index("DEFAULT_PP_SOLVE_POOL_FLOOR = ")]
        block = head[head.rindex("#: THE SHIPPED FLOOR") :]
        self.assertIn(
            "39,13,12 mit bs2 im pp layout soll standard werden vorerst", block
        )
        self.assertIn("414654", block)
        self.assertIn("481400", block)


def _constant_lineno():
    tree = ast.parse((WEG2_DIR / "launcher.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "DEFAULT_PP_SOLVE_POOL_FLOOR":
                    return node.lineno
    raise AssertionError("DEFAULT_PP_SOLVE_POOL_FLOOR is gone")


if __name__ == "__main__":
    unittest.main()
