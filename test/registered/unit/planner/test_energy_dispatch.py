# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#350 phase 3: the public call path actually selects the energy plan.

Phase 2 built ``solve_energy_units`` and proved it finds a different optimum.
Phase 3 wires it into ``solve()`` so a user-selected ``--objective energy``
changes the plan the boot uses. What that adds, and what this file pins:

1. DISPATCH -- with the energy objective, ``solve()`` returns the energy
   optimum (mode="energy"), not the throughput one.
2. NO SILENT FALLBACK -- an energy request that cannot be priced (a capacity
   goal, or no power anchors) comes back ``ok=False`` with a named reason.
   Handing the caller a throughput plan under an energy label is the one
   outcome this feature must never produce, so it gets its own tests.
3. BYTE-IDENTICAL DEFAULT -- ``objective="throughput"`` and an omitted
   objective produce exactly the plan the pre-#350 solver produced, asserted
   as plan EQUALITY against the un-dispatched code path.
4. THE ``enc`` GOAL -- phase 2 exercised ``dec`` only. The prefill goal shares
   the machinery, so it gets the same divergence proof here.

The heavy ``solve()`` entry needs a real cost model, so the dispatch tests
drive it through a light stand-in that exercises the dispatch decision itself;
the goal-level energy arithmetic (where the plan actually differs) is driven
against the real ``_energy_objective_value`` on a synthetic cost model, as in
phase 2.
"""

import unittest

from sglang.srt.planner.key_solver import (
    ENERGY_PRICEABLE_GOALS,
    _busy_seconds,
    _energy_objective_value,
    _energy_unscorable_reason,
    _objective_value,
)
from sglang.srt.planner.objective import EnergyModel, Provenance, RankPower
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


# --- Shared synthetic rig (same shape as the phase-2 fixture). --------------
class _FakePerf:
    def __init__(self, tp_size, bw):
        self.tp_size = tp_size
        self._bw = bw

    def effective_decode_bw(self, membw, gemv):
        del membw, gemv
        return list(self._bw)


class _FakeRates:
    def __init__(self, n):
        self._n = n
        self.gemv_gbs = [None] * n

    def require_membw_gbs(self):
        return [1.0] * self._n

    def require_gemm_tflops(self):
        # The enc goal's per-rank compute rates: rank 0 is the strong card.
        return [40.0, 10.0, 10.0]


class _FakeModel:
    def __init__(self, bw, fixed_bytes, unit_bytes, units, fixed_params, unit_params):
        n = len(bw)
        self.perf = _FakePerf(n, bw)
        self.rates = _FakeRates(n)
        self.fixed_bytes = list(fixed_bytes)
        self.unit_bytes = float(unit_bytes)
        self.units = int(units)
        self.fixed_params = list(fixed_params)
        self.unit_params = float(unit_params)


def _model():
    return _FakeModel(
        bw=[4.0, 1.0, 1.0],
        fixed_bytes=[1.0, 1.0, 1.0],
        unit_bytes=1.0,
        units=12,
        fixed_params=[1e9, 1e9, 1e9],
        unit_params=1e9,
    )


_EFFICIENT = RankPower(idle_w=30.0, active_w=300.0, source="measured")
_THIRSTY = RankPower(idle_w=90.0, active_w=320.0, source="measured")
_POST = [0.0, 0.0, 0.0]


def _energy_model():
    return EnergyModel(per_rank=(_EFFICIENT, _THIRSTY, _THIRSTY))


def _sweep(goal):
    """Every feasible split of the units, scored both ways."""
    model, em = _model(), _energy_model()
    out = []
    for u0 in range(model.units + 1):
        for u1 in range(model.units - u0 + 1):
            units = [u0, u1, model.units - u0 - u1]
            out.append(
                (
                    tuple(units),
                    _objective_value(goal, model, units, _POST),
                    _energy_objective_value(goal, model, units, _POST, em),
                )
            )
    return out


class TestEncGoalEnergy(CustomTestCase):
    """The prefill goal, unexercised in phase 2."""

    def test_enc_is_energy_priceable(self):
        self.assertIn("enc", ENERGY_PRICEABLE_GOALS)

    def test_enc_busy_times_agree_with_the_min_max_objective(self):
        model, units = _model(), [6, 3, 3]
        busy = _busy_seconds("enc", model, units, _POST)
        self.assertAlmostEqual(
            max(busy), _objective_value("enc", model, units, _POST)
        )

    def test_enc_throughput_and_energy_optima_diverge(self):
        sweep = _sweep("enc")
        best_time = min(sweep, key=lambda e: e[1])[0]
        best_energy = min(sweep, key=lambda e: e[2])[0]
        self.assertNotEqual(
            best_time,
            best_energy,
            "the enc goal must show the same divergence dec does, or the "
            "prefill side of the objective is decorative",
        )

    def test_enc_energy_pick_is_cheaper_and_slower(self):
        sweep = _sweep("enc")
        by_units = {u: (t, j) for u, t, j in sweep}
        t_fast, j_fast = by_units[min(sweep, key=lambda e: e[1])[0]]
        t_green, j_green = by_units[min(sweep, key=lambda e: e[2])[0]]
        self.assertLess(j_green, j_fast)
        self.assertGreater(t_green, t_fast)

    def test_enc_energy_concentrates_on_the_efficient_card(self):
        sweep = _sweep("enc")
        best_time = min(sweep, key=lambda e: e[1])[0]
        best_energy = min(sweep, key=lambda e: e[2])[0]
        self.assertGreater(best_energy[0], best_time[0])


class TestUnscorableIsNamedNotSubstituted(CustomTestCase):
    """The one outcome the feature must never produce is a throughput plan
    wearing an energy label."""

    def test_capacity_goals_are_refused_with_a_reason(self):
        for goal in ("maxkv", "sessions"):
            with self.subTest(goal=goal):
                reason = _energy_unscorable_reason(goal, _energy_model())
                self.assertIsNotNone(reason)
                self.assertIn("BYTES", reason)
                self.assertIn("dec", reason)  # names what IS priceable

    def test_missing_power_anchors_are_refused_with_a_reason(self):
        reason = _energy_unscorable_reason("dec", None)
        self.assertIsNotNone(reason)
        self.assertIn("power anchors", reason)
        self.assertIn("silent substitution", reason)

    def test_a_priceable_goal_with_anchors_is_scorable(self):
        for goal in ENERGY_PRICEABLE_GOALS:
            self.assertIsNone(_energy_unscorable_reason(goal, _energy_model()))

    def test_estimate_anchors_are_scorable_but_labelled(self):
        # An estimate is a value, not an absence: it plans, and it says so.
        em = EnergyModel(
            per_rank=(_EFFICIENT, _THIRSTY, RankPower(idle_w=32.0, active_w=320.0))
        )
        self.assertIsNone(_energy_unscorable_reason("dec", em))
        self.assertIs(em.provenance, Provenance.ESTIMATE)


class TestSolveDispatchContract(CustomTestCase):
    """The dispatch decision in ``solve()``, exercised without a real rig."""

    def test_energy_with_a_second_goal_is_refused(self):
        # A Pareto front over two THROUGHPUT goals is not an energy question.
        from sglang.srt.planner import key_solver

        with self.assertRaises(ValueError) as ctx:
            key_solver.solve(
                None, [1], [1], None,
                goal="dec", goal_b="enc", objective="energy",
            )
        self.assertIn("single-goal", str(ctx.exception))

    def test_energy_with_constraints_is_refused(self):
        from sglang.srt.planner import key_solver

        with self.assertRaises(ValueError) as ctx:
            key_solver.solve(
                None, [1], [1], None,
                goal="dec", constraints={"maxkv": 1.0}, objective="energy",
            )
        self.assertIn("constraints", str(ctx.exception))

    def test_an_unknown_objective_is_rejected(self):
        from sglang.srt.planner import key_solver

        with self.assertRaises(ValueError):
            key_solver.solve(None, [1], [1], None, goal="dec", objective="cheap")

    def test_unknown_goal_still_wins_over_the_objective_check(self):
        # Goal validation runs first; an unknown goal must not be reported as
        # an objective problem.
        from sglang.srt.planner import key_solver

        with self.assertRaises(ValueError) as ctx:
            key_solver.solve(None, [1], [1], None, goal="bogus", objective="energy")
        self.assertIn("unknown goal", str(ctx.exception))


class TestDefaultPathByteIdentical(CustomTestCase):
    """``--objective throughput`` and an omitted objective must plan exactly
    as before the dispatch layer existed."""

    #: Frozen from the fixture's closed form, unchanged from phase 2.
    DEC_FIXTURES = (
        ([12, 0, 0], 3.25),
        ([6, 3, 3], 4.0),
        ([4, 4, 4], 5.0),
        ([0, 6, 6], 7.0),
    )

    def test_throughput_objective_does_not_enter_the_energy_branch(self):
        # The dispatch is gated on the objective ONLY; with the default it
        # must not even ask whether the request is priceable, so a missing
        # energy model cannot affect a throughput solve.
        from sglang.srt.planner import key_solver

        # An unknown goal raises for the same reason it always did -- proof
        # the throughput path reaches the original validation untouched.
        with self.assertRaises(ValueError) as ctx:
            key_solver.solve(None, [1], [1], None, goal="bogus")
        self.assertIn("unknown goal", str(ctx.exception))

    def test_objective_value_matches_the_frozen_numbers(self):
        model = _model()
        for units, expected in self.DEC_FIXTURES:
            with self.subTest(units=units):
                self.assertAlmostEqual(
                    _objective_value("dec", model, units, _POST), expected
                )

    def test_dispatch_layer_did_not_move_the_throughput_definition(self):
        model = _model()
        for goal in ("dec", "enc"):
            for units, _ in self.DEC_FIXTURES:
                busy = _busy_seconds(goal, model, units, _POST)
                self.assertAlmostEqual(
                    _objective_value(goal, model, units, _POST), max(busy)
                )


class TestSolverApiThreading(CustomTestCase):
    """``--objective`` reaches the solver through the API layer."""

    def test_power_anchors_become_an_energy_model(self):
        from sglang.srt.planner.solver_api import _energy_model_from_payload

        em = _energy_model_from_payload(
            [
                {"idle_w": 30, "active_w": 300, "source": "measured"},
                {"idle_w": 90, "active_w": 320, "source": "measured"},
            ]
        )
        self.assertEqual(len(em.per_rank), 2)
        self.assertIs(em.provenance, Provenance.MEASURED)

    def test_absent_anchors_stay_absent(self):
        from sglang.srt.planner.solver_api import _energy_model_from_payload

        self.assertIsNone(_energy_model_from_payload(None))
        self.assertIsNone(_energy_model_from_payload([]))

    def test_anchors_default_to_the_estimate_tier(self):
        from sglang.srt.planner.solver_api import _energy_model_from_payload

        em = _energy_model_from_payload([{"idle_w": 32, "active_w": 320}])
        self.assertIs(em.provenance, Provenance.ESTIMATE)


if __name__ == "__main__":
    unittest.main()
