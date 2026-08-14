# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#363 defect 4: the planner feed's per-rank KV split, resolved not derived.

HOW THIS ONE WAS FOUND. It was MASKED by defect 3. While no rank could find
the card probe, the feed failed early with "no card probe on disk". With the
probe reachable (R14), the solver gets further on metal and hits the next
wall -- one all three ranks logged on the first boot after that fix:

    PlannerFeedUnavailable("the solver API returns no per-rank KV vector
    (Stage.kv_token_vector). key_solver.capacity() computes it as cap['p']
    but key_solver_payload does not surface it; exposing it is the remaining
    wiring step. A split derived here would be a fabricated reshard target")

So #363's "0 flip targets" had THREE independent causes, not the two the ACT
window found: the missing per-stage measurements, the unreachable card probe,
and this. Fixing one reveals the next, which is why each was invisible until
the one before it moved.

THE WIRING, AND THE LINE IT DOES NOT CROSS.

`key_solver.capacity()` computes the per-rank split as `cap["p"]` and threw it
away; every caller read only the SUM (`cells["maxkv"]["value"]`). It is now
surfaced, unmodified, as `per_rank_kv_tokens` -- absolute tokens, the solver's
own numbers.

`Stage.kv_token_vector` lives in the #297 reshard RATIO space, so the absolute
split has to be resolved into it. It is RESOLVED, never DERIVED: the solver's
proportions are matched against the vectors the OPERATOR declared in
`--kv-reshard-vectors`, and the answer is always one of those or nothing.
Rounding the absolute split into a small ratio would produce exactly what the
old refusal warned about -- a vector nothing has backed with pool rows that
nonetheless reads as the solver's answer.

When no declared vector is close enough the feed refuses AND NAMES the
proportions it wanted next to every declared vector's deviation, so the
operator can declare the right one instead of guessing.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers.regime_runtime import (
    _VECTOR_MATCH_TOLERANCE,
    _match_declared_vector,
    _shares,
    _shares_str,
)


class TestSharesAreScaleFree(unittest.TestCase):
    """Two vectors that differ by a common factor are the same layout."""

    def test_a_common_factor_does_not_change_the_shares(self):
        self.assertEqual(_shares([2, 11, 10]), _shares([20, 110, 100]))

    def test_a_zero_total_has_no_shares(self):
        self.assertIsNone(_shares([0, 0, 0]))

    def test_the_description_carries_tokens_and_percentages(self):
        s = _shares_str([100, 300, 100])
        self.assertIn("[100, 300, 100] tokens", s)
        self.assertIn("20.0%", s)
        self.assertIn("60.0%", s)


class TestOnlyADeclaredVectorCanBeTheAnswer(unittest.TestCase):
    def test_an_exact_proportional_match_resolves(self):
        """The solver's absolute split scaled by ~2600; same layout."""
        vec, dev, _ = _match_declared_vector(
            [5200, 28600, 26000], [(2, 11, 10), (3, 10, 10)]
        )
        self.assertEqual(vec, (2, 11, 10))
        self.assertAlmostEqual(dev, 0.0, places=9)

    def test_the_closest_declared_vector_wins(self):
        """Slightly nearer 3,10,10 than 2,11,10 -- and it must pick, not
        average the two into something neither of them is."""
        vec, dev, _ = _match_declared_vector(
            [13.2, 43.3, 43.5], [(2, 11, 10), (3, 10, 10)]
        )
        self.assertEqual(vec, (3, 10, 10))
        self.assertLessEqual(dev, _VECTOR_MATCH_TOLERANCE)

    def test_nothing_close_enough_is_a_refusal_not_a_snap(self):
        """A split nowhere near either declared vector must NOT be rounded
        onto the nearest one. That is the fabricated target."""
        vec, dev, ranked = _match_declared_vector(
            [90, 5, 5], [(2, 11, 10), (3, 10, 10)]
        )
        self.assertIsNone(vec)
        self.assertGreater(dev, _VECTOR_MATCH_TOLERANCE)
        self.assertTrue(ranked, "the refusal must still rank the alternatives")

    def test_the_refusal_names_every_declared_vector_and_its_deviation(self):
        """So the operator can declare the right one instead of guessing."""
        _, _, ranked = _match_declared_vector([90, 5, 5], [(2, 11, 10), (3, 10, 10)])
        text = "; ".join(ranked)
        self.assertIn("[2, 11, 10]", text)
        self.assertIn("[3, 10, 10]", text)
        self.assertIn("off by", text)

    def test_the_answer_is_always_a_declared_vector_verbatim(self):
        declared = [(2, 11, 10), (3, 10, 10)]
        vec, _, _ = _match_declared_vector([5200, 28600, 26000], declared)
        self.assertIn(vec, declared)

    def test_no_declared_vectors_is_a_refusal(self):
        vec, _, ranked = _match_declared_vector([1, 1, 1], [])
        self.assertIsNone(vec)
        self.assertEqual(ranked, [])

    def test_a_wrong_length_vector_is_ignored_not_matched(self):
        """A two-rank vector cannot describe a three-rank split."""
        vec, _, _ = _match_declared_vector([10, 10, 10], [(1, 1)])
        self.assertIsNone(vec)

    def test_an_infeasible_split_has_no_answer(self):
        vec, _, _ = _match_declared_vector([0, 0, 0], [(2, 11, 10)])
        self.assertIsNone(vec)


class TestTheToleranceIsNotAccidental(unittest.TestCase):
    def test_the_tolerance_admits_the_granularity_a_small_vector_can_express(self):
        """One unit of a 23-unit vector is 4.3 % of the pool, so a tolerance
        tighter than the vector's own granularity would refuse every vector an
        operator could declare. 2 % sits below one unit and above float
        noise."""
        self.assertLess(_VECTOR_MATCH_TOLERANCE, 1.0 / 23.0)
        self.assertGreater(_VECTOR_MATCH_TOLERANCE, 1e-6)

    def test_a_split_one_whole_unit_away_is_refused(self):
        """2,11,10 vs 2,12,9 differ by one unit on two ranks -- visibly a
        different layout, and it must not resolve."""
        vec, _, _ = _match_declared_vector([2, 12, 9], [(2, 11, 10)])
        self.assertIsNone(vec)


class TestTheSolverSurfacesItsOwnSplit(unittest.TestCase):
    """The other half: key_solver must actually hand the split over."""

    def test_candidate_to_json_carries_per_rank_kv_tokens(self):
        from sglang.srt.planner.key_solver import Candidate

        c = Candidate(
            units=[1, 1, 1],
            mlp_ratio=[1, 1, 1],
            roles=["shard"] * 3,
            feasible=True,
            reasons=[],
            predictions={"maxkv": {"value": 300, "per_rank_tokens": [90, 110, 100]}},
            raw={},
            tradeoff={},
            remeasure={},
        )
        self.assertEqual(c.to_json()["per_rank_kv_tokens"], [90, 110, 100])

    def test_an_infeasible_candidate_carries_none_not_a_guess(self):
        from sglang.srt.planner.key_solver import Candidate

        c = Candidate(
            units=[1, 1, 1],
            mlp_ratio=[1, 1, 1],
            roles=["shard"] * 3,
            feasible=False,
            reasons=["no pool"],
            predictions={"maxkv": {"value": None, "per_rank_tokens": None}},
            raw={},
            tradeoff={},
            remeasure={},
        )
        self.assertIsNone(c.to_json()["per_rank_kv_tokens"])

    def test_predict_all_fills_the_cell_from_capacity(self):
        """The plumbing at its source: cells["maxkv"]["per_rank_tokens"] is
        cap["p"], rounded, and nothing else."""
        import inspect

        from sglang.srt.planner import key_solver

        src = inspect.getsource(key_solver._predict_all)
        self.assertIn('cells["maxkv"]["per_rank_tokens"]', src)
        self.assertIn('cap["p"]', src)


if __name__ == "__main__":
    unittest.main()
