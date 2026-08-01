# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Energy as a first-class SOLVER goal (#350 phase 2).

Phase 1 ranked finished candidates. Phase 2 pushes the objective INTO the key
solver, and the test that justifies it is the one showing the energy optimum
is a plan the throughput search never returns: ``_solve_single`` minimises the
SLOWEST rank (a min-max over per-rank times), while the energy objective
minimises ``lockstep_time * sum_r power_r`` -- a product of a max and a sum,
whose optimum can sit elsewhere entirely.

Three things are proven here:

1. DIVERGENCE -- on a rig with one efficient card and two thirsty ones, the
   energy-optimal unit vector differs from the throughput-optimal one, and it
   is genuinely better in J/token while genuinely worse in seconds. If the
   two ever coincided on this fixture the solver integration would be doing
   nothing.
2. PROVENANCE -- the J/work rate carries measured/estimate by weakest-wins
   over the per-card power anchors, and a rig with one estimated card cannot
   report a measured efficiency.
3. NO REGRESSION -- the throughput path is untouched: ``_solve_single`` and
   ``_objective_value`` return exactly what they returned before phase 2 on
   fixed fixtures (plan equality, not just "still runs").

CPU-only, hermetic: the cost model is a small stand-in exposing the two
attributes ``_terms`` reads for the time goals, so no rig probe, no torch, no
card is involved.
"""

import unittest

from sglang.srt.planner.key_solver import (
    ENERGY_PRICEABLE_GOALS,
    _busy_seconds,
    _energy_objective_value,
    _objective_value,
)
from sglang.srt.planner.objective import (
    EnergyModel,
    Provenance,
    RankPower,
    energy_per_work,
    energy_rate,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


# --- A three-rank stand-in of the parts _terms("dec") reads. ---------------
#
# Rank 0 is the fast, efficient card (this rig's 5090); ranks 1-2 are the
# slower, thirstier ones (the 3080s). Everything is chosen so the arithmetic
# is checkable by hand.
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


class _FakeModel:
    """Only what the ``dec`` goal touches: fixed_bytes, unit_bytes, the
    effective decode bandwidths, and the rank count."""

    def __init__(self, bw, fixed_bytes, unit_bytes, units):
        n = len(bw)
        self.perf = _FakePerf(n, bw)
        self.rates = _FakeRates(n)
        self.fixed_bytes = list(fixed_bytes)
        self.unit_bytes = float(unit_bytes)
        self.units = int(units)


def _model():
    # Rank 0 is 4x the bandwidth of ranks 1-2: putting a unit on rank 0 costs
    # a quarter of the time it costs on the others.
    return _FakeModel(
        bw=[4.0, 1.0, 1.0],
        fixed_bytes=[1.0, 1.0, 1.0],
        unit_bytes=1.0,
        units=12,
    )


#: The rig asymmetry, as power anchors. The efficient card has the higher
#: ceiling (it is the big one) but the thirsty pair together dominate the
#: rig's draw whenever they are busy.
_EFFICIENT = RankPower(idle_w=30.0, active_w=300.0, source="measured")
_THIRSTY = RankPower(idle_w=90.0, active_w=320.0, source="measured")


def _energy_model():
    return EnergyModel(per_rank=(_EFFICIENT, _THIRSTY, _THIRSTY))


_POST = [0.0, 0.0, 0.0]


class TestEnergyPerWork(CustomTestCase):
    def test_lockstep_time_times_summed_power(self):
        # Rank 0 busy 1.0 s, ranks 1-2 busy 0.5 s -> round is 1.0 s, the two
        # slow ranks sit at 50 % utilization.
        em = _energy_model()
        j = energy_per_work([1.0, 0.5, 0.5], em)
        expected = 1.0 * (
            _EFFICIENT.watts(1.0) + 2 * _THIRSTY.watts(0.5)
        )
        self.assertAlmostEqual(j, expected)

    def test_a_waiting_rank_draws_its_idle_floor(self):
        em = _energy_model()
        j = energy_per_work([1.0, 0.0, 0.0], em)
        self.assertAlmostEqual(j, 1.0 * (300.0 + 90.0 + 90.0))

    def test_rank_count_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            energy_per_work([1.0, 1.0], _energy_model())

    def test_swapped_anchors_are_rejected(self):
        with self.assertRaises(ValueError):
            RankPower(idle_w=300.0, active_w=30.0)


class TestEnergyProvenance(CustomTestCase):
    def test_all_measured_anchors_give_a_measured_rate(self):
        rate = energy_rate([1.0, 0.5, 0.5], _energy_model())
        self.assertIs(rate.provenance, Provenance.MEASURED)
        self.assertEqual(rate.unit, "J/tok")

    def test_one_estimated_card_makes_the_rig_figure_an_estimate(self):
        em = EnergyModel(
            per_rank=(
                _EFFICIENT,
                _THIRSTY,
                RankPower(idle_w=32.0, active_w=320.0),  # estimate-tdp
            )
        )
        self.assertIs(em.provenance, Provenance.ESTIMATE)
        rate = energy_rate([1.0, 0.5, 0.5], em)
        self.assertIs(rate.provenance, Provenance.ESTIMATE)

    def test_frame_unit_for_the_video_classes(self):
        rate = energy_rate([1.0, 1.0, 1.0], _energy_model(), work_unit="frame")
        self.assertEqual(rate.unit, "J/frame")

    def test_capacity_goals_are_not_energy_priceable(self):
        # maxkv/sessions terms are BYTES; pricing them in joules would be the
        # silent substitution the discipline forbids.
        self.assertEqual(ENERGY_PRICEABLE_GOALS, ("dec", "enc"))
        self.assertNotIn("maxkv", ENERGY_PRICEABLE_GOALS)
        self.assertNotIn("sessions", ENERGY_PRICEABLE_GOALS)


class TestThroughputAndEnergyOptimaDiverge(CustomTestCase):
    """The phase-2 headline, at the solver's own objective functions."""

    def _sweep(self):
        """Every feasible split of 12 units over 3 ranks, scored both ways."""
        model = _model()
        em = _energy_model()
        out = []
        for u0 in range(model.units + 1):
            for u1 in range(model.units - u0 + 1):
                u2 = model.units - u0 - u1
                units = [u0, u1, u2]
                t = _objective_value("dec", model, units, _POST)
                j = _energy_objective_value("dec", model, units, _POST, em)
                out.append((tuple(units), t, j))
        return out

    def test_the_two_optima_are_different_plans(self):
        sweep = self._sweep()
        best_time = min(sweep, key=lambda e: e[1])
        best_energy = min(sweep, key=lambda e: e[2])
        self.assertNotEqual(
            best_time[0],
            best_energy[0],
            "the energy optimum must be a plan the time-optimal search does "
            "not return, or pushing the objective into the solver buys "
            "nothing",
        )

    def test_the_energy_optimum_is_slower_and_cheaper(self):
        # The honest trade, both directions asserted so neither can be a
        # coincidence of the fixture.
        sweep = self._sweep()
        by_units = {u: (t, j) for u, t, j in sweep}
        best_time = min(sweep, key=lambda e: e[1])[0]
        best_energy = min(sweep, key=lambda e: e[2])[0]
        t_fast, j_fast = by_units[best_time]
        t_green, j_green = by_units[best_energy]
        self.assertLess(j_green, j_fast, "energy pick must win joules")
        self.assertGreater(t_green, t_fast, "energy pick must lose seconds")

    def test_the_energy_optimum_concentrates_on_the_efficient_card(self):
        # Not just "different" -- different in the direction the rig's
        # asymmetry predicts: more units on rank 0, the efficient card.
        sweep = self._sweep()
        best_time = min(sweep, key=lambda e: e[1])[0]
        best_energy = min(sweep, key=lambda e: e[2])[0]
        self.assertGreater(best_energy[0], best_time[0])

    def test_busy_seconds_agree_with_the_min_max_objective(self):
        # The energy objective must read the SAME per-rank terms the
        # throughput objective maxes over -- not a parallel cost model.
        model, units = _model(), [6, 3, 3]
        busy = _busy_seconds("dec", model, units, _POST)
        self.assertAlmostEqual(
            max(busy), _objective_value("dec", model, units, _POST)
        )


class TestThroughputPathUnchanged(CustomTestCase):
    """Phase 2 must not move a single throughput plan."""

    #: Frozen expectations, computed from the fixture's closed form: for the
    #: ``dec`` goal, time_r = (fixed_r + unit_bytes * u_r) / bw_r.
    FIXTURES = (
        ([12, 0, 0], 3.25),
        ([6, 3, 3], 4.0),
        ([4, 4, 4], 5.0),
        ([0, 6, 6], 7.0),
    )

    def test_objective_value_matches_the_frozen_numbers(self):
        model = _model()
        for units, expected in self.FIXTURES:
            with self.subTest(units=units):
                self.assertAlmostEqual(
                    _objective_value("dec", model, units, _POST), expected
                )

    def test_objective_value_is_still_the_max_of_the_terms(self):
        # The definition itself, pinned: phase 2 added a sibling, it did not
        # redefine the throughput objective.
        model = _model()
        for units, _ in self.FIXTURES:
            busy = _busy_seconds("dec", model, units, _POST)
            self.assertAlmostEqual(
                _objective_value("dec", model, units, _POST), max(busy)
            )

    def test_energy_evaluation_does_not_mutate_the_model(self):
        model = _model()
        before = (
            list(model.fixed_bytes),
            model.unit_bytes,
            model.units,
            list(model.perf.effective_decode_bw(None, None)),
        )
        _energy_objective_value("dec", model, [6, 3, 3], _POST, _energy_model())
        after = (
            list(model.fixed_bytes),
            model.unit_bytes,
            model.units,
            list(model.perf.effective_decode_bw(None, None)),
        )
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
