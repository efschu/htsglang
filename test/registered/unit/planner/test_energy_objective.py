# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Energy as a selectable planner objective (#350, ANALYSE_347 §6).

The point of the objective is that it CHANGES the ranking: a config ranked
best for throughput and a config ranked best for energy must be able to
DIFFER, or the objective does nothing. The headline test builds two candidate
placements on this rig's asymmetry -- one that spreads work across all three
cards (more tok/s), one that concentrates on the efficient 5090 (more tok/J)
-- and asserts the two objectives pick different winners.

Everything is CPU-only, hermetic: the energy source is injected as a plain
function returning J/token Rates, exactly the way ``roofline_energy`` takes an
injectable power profile. No torch, no server, no card.
"""

import unittest

from sglang.srt.planner.cost_model import Provenance, Rate
from sglang.srt.planner.objective import (
    Objective,
    combine_provenance,
    objective_value,
    rank,
    resolve_objective,
    work_per_joule,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestWorkPerJoule(CustomTestCase):
    def test_efficiency_is_the_inverse_of_j_per_work(self):
        tput = Rate.measured(100.0, "probe", unit="tok/s")
        jpt = Rate.measured(2.0, "harness", unit="J/tok")
        wpj = work_per_joule(tput, jpt)
        self.assertAlmostEqual(wpj.require(), 0.5)  # 1 / 2 J per tok
        self.assertEqual(wpj.unit, "tok/J")
        self.assertIs(wpj.provenance, Provenance.MEASURED)

    def test_measured_over_estimate_is_an_estimate(self):
        # The #359 house rule: a measured tok/s over an ESTIMATE J/tok is an
        # estimate tok/J, never a measurement.
        tput = Rate.measured(100.0, "probe", unit="tok/s")
        jpt = Rate.estimate(4.0, "roofline", unit="J/tok")
        wpj = work_per_joule(tput, jpt)
        self.assertAlmostEqual(wpj.require(), 0.25)
        self.assertIs(wpj.provenance, Provenance.ESTIMATE)

    def test_absent_energy_makes_the_efficiency_absent(self):
        tput = Rate.measured(100.0, "probe", unit="tok/s")
        jpt = Rate.absent("no energy measured and no TDP for this card")
        wpj = work_per_joule(tput, jpt)
        self.assertTrue(wpj.is_absent)
        self.assertIn("no energy measured", wpj.source)

    def test_absent_throughput_makes_the_efficiency_absent(self):
        tput = Rate.absent("no measured decode baseline for this checkpoint")
        jpt = Rate.measured(2.0, "harness", unit="J/tok")
        wpj = work_per_joule(tput, jpt)
        self.assertTrue(wpj.is_absent)

    def test_non_positive_j_per_work_is_absent_not_infinite(self):
        tput = Rate.measured(100.0, "probe", unit="tok/s")
        wpj = work_per_joule(tput, Rate.measured(0.0, "bug", unit="J/tok"))
        self.assertTrue(wpj.is_absent)

    def test_frame_unit_for_video_classes(self):
        tput = Rate.measured(24.0, "probe", unit="frame/s")
        jpf = Rate.measured(3.0, "harness", unit="J/frame")
        wpj = work_per_joule(tput, jpf, work_unit="frame")
        self.assertAlmostEqual(wpj.require(), 1.0 / 3.0)
        self.assertEqual(wpj.unit, "frame/J")


class TestCombineProvenance(CustomTestCase):
    def test_weakest_wins(self):
        M, E, A = Provenance.MEASURED, Provenance.ESTIMATE, Provenance.ABSENT
        self.assertIs(combine_provenance(M, M), M)
        self.assertIs(combine_provenance(M, E), E)
        self.assertIs(combine_provenance(E, M), E)
        self.assertIs(combine_provenance(M, A), A)
        self.assertIs(combine_provenance(A, E), A)


class TestObjectiveValue(CustomTestCase):
    def test_throughput_passes_the_rate_through(self):
        tput = Rate.measured(123.0, "probe", unit="tok/s")
        self.assertIs(objective_value(Objective.THROUGHPUT, tput), tput)

    def test_energy_needs_a_j_per_work_rate(self):
        tput = Rate.measured(123.0, "probe", unit="tok/s")
        with self.assertRaises(ValueError):
            objective_value(Objective.ENERGY, tput, None)


# --- The rig: a 5090 and two 3080s, with this rig's efficiency asymmetry. ---
#
# Two candidate placements over the same three cards:
#
#   spread   -- work across all three cards. Highest aggregate tok/s, but the
#               3080s are less efficient per joule and the lockstep wait wastes
#               power on the fast card -> higher J/token.
#   concentrate -- most work on the efficient 5090. Lower tok/s (fewer cards
#               doing useful work) but the best J/token.
#
# The numbers are illustrative but ordered exactly as the rig behaves: spread
# wins tok/s, concentrate wins tok/J. If the objective did nothing, both
# objectives would pick the same candidate and this test would fail.
_CANDIDATES = {
    "spread": {"tok_s": 180.0, "j_per_tok": 3.0},       # 60 tok/J
    "concentrate": {"tok_s": 120.0, "j_per_tok": 1.5},  # 80 tok/J
}


def _throughput(cand: str) -> Rate:
    return Rate.measured(_CANDIDATES[cand]["tok_s"], "fixture", unit="tok/s")


def _j_per_tok(cand: str) -> Rate:
    return Rate.measured(_CANDIDATES[cand]["j_per_tok"], "fixture", unit="J/tok")


class TestThroughputVsEnergyDiverge(CustomTestCase):
    """The headline: the two objectives pick DIFFERENT winners."""

    def test_throughput_best_and_energy_best_differ(self):
        cands = list(_CANDIDATES)

        tput_ranked, _, tput_floor = rank(
            cands, Objective.THROUGHPUT, throughput_fn=_throughput
        )
        energy_ranked, _, energy_floor = rank(
            cands,
            Objective.ENERGY,
            throughput_fn=_throughput,
            j_per_work_fn=_j_per_tok,
        )

        self.assertEqual(tput_ranked[0].key, "spread")
        self.assertEqual(energy_ranked[0].key, "concentrate")
        # The whole point: the winners are not the same candidate.
        self.assertNotEqual(tput_ranked[0].key, energy_ranked[0].key)
        # Both rankings are measured-grade here (no estimate mixed in).
        self.assertIs(tput_floor, Provenance.MEASURED)
        self.assertIs(energy_floor, Provenance.MEASURED)

    def test_energy_winner_really_has_the_better_efficiency(self):
        energy_ranked, _, _ = rank(
            list(_CANDIDATES),
            Objective.ENERGY,
            throughput_fn=_throughput,
            j_per_work_fn=_j_per_tok,
        )
        self.assertAlmostEqual(energy_ranked[0].score.require(), 1.0 / 1.5)
        self.assertAlmostEqual(energy_ranked[-1].score.require(), 1.0 / 3.0)


class TestRankingProvenanceAndAbsence(CustomTestCase):
    def test_absent_energy_candidate_is_unscorable_not_dropped(self):
        # A candidate with no energy data must surface as unscorable, with its
        # reason, not vanish or be ranked on a substituted number.
        def jpw(cand):
            if cand == "concentrate":
                return Rate.absent("no TDP in the card library for this card")
            return _j_per_tok(cand)

        ranked, unscorable, floor = rank(
            list(_CANDIDATES),
            Objective.ENERGY,
            throughput_fn=_throughput,
            j_per_work_fn=jpw,
        )
        self.assertEqual([s.key for s in ranked], ["spread"])
        self.assertEqual([s.key for s in unscorable], ["concentrate"])
        self.assertIn("no TDP", unscorable[0].score.source)
        self.assertIs(floor, Provenance.MEASURED)

    def test_ranking_floor_is_the_weakest_provenance(self):
        # spread measured, concentrate estimate -> the ranking is only
        # estimate-grade and must say so.
        def jpw(cand):
            if cand == "concentrate":
                return Rate.estimate(1.5, "roofline", unit="J/tok")
            return _j_per_tok(cand)

        _, _, floor = rank(
            list(_CANDIDATES),
            Objective.ENERGY,
            throughput_fn=_throughput,
            j_per_work_fn=jpw,
        )
        self.assertIs(floor, Provenance.ESTIMATE)

    def test_all_absent_ranking_floor_is_absent(self):
        _, unscorable, floor = rank(
            list(_CANDIDATES),
            Objective.ENERGY,
            throughput_fn=_throughput,
            j_per_work_fn=lambda c: Rate.absent("no energy anywhere"),
        )
        self.assertEqual(len(unscorable), 2)
        self.assertIs(floor, Provenance.ABSENT)

    def test_energy_ranking_requires_the_energy_source(self):
        with self.assertRaises(ValueError):
            rank(list(_CANDIDATES), Objective.ENERGY, throughput_fn=_throughput)


class TestObjectiveSelection(CustomTestCase):
    def test_default_is_throughput_byte_identical(self):
        from types import SimpleNamespace

        # Absent field -> today's behaviour.
        self.assertIs(resolve_objective(SimpleNamespace()), Objective.THROUGHPUT)
        self.assertIs(
            resolve_objective(SimpleNamespace(objective="throughput")),
            Objective.THROUGHPUT,
        )

    def test_energy_selected(self):
        from types import SimpleNamespace

        self.assertIs(
            resolve_objective(SimpleNamespace(objective="energy")),
            Objective.ENERGY,
        )

    def test_unknown_objective_is_rejected(self):
        from types import SimpleNamespace

        with self.assertRaises(ValueError):
            resolve_objective(SimpleNamespace(objective="cheapest"))

    def test_server_args_default_and_validation(self):
        from sglang.srt.server_args import ServerArgs

        field = ServerArgs.__dataclass_fields__["objective"]
        self.assertEqual(field.default, "throughput")

        args = ServerArgs.__new__(ServerArgs)
        args.objective = "energy"
        args._handle_objective()  # canonicalises, no raise
        self.assertEqual(args.objective, "energy")

        args.objective = "nonsense"
        with self.assertRaises(ValueError):
            args._handle_objective()


if __name__ == "__main__":
    unittest.main()
