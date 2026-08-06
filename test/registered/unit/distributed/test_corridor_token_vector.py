"""Corridor-constrained uneven-DCP token-vector solver (#602).

CPU only: the solver is a pure function of integer capacities. No device, no
process group, no model.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        python -m pytest -q test/registered/unit/distributed/test_corridor_token_vector.py
"""

import itertools
import unittest

from sglang.srt.distributed.corridor_vector import (
    CORRIDOR_GRAIN,
    CorridorInfeasible,
    RankCapacity,
    context_budget,
    corridor_pool_bytes,
    solve_corridor_vector,
    solve_token_vector,
)
from sglang.srt.distributed.utils import partition_units
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

MIB = 1 << 20

# Measured on the tp3 rig (2026-08-06 window), profiled per-rank token
# capacities P_r for the three cards.
MEASURED_P = [301435, 117912, 158474]


def brute_force_best(capacities, grain):
    """Exhaustive argmax of C(v) over sum(v) <= grain -- the oracle."""
    n = len(capacities)
    best_c, best = -1, None
    for total in range(n, grain + 1):
        # compositions of `total` into n positive parts
        for cuts in itertools.combinations(range(1, total), n - 1):
            vec = []
            prev = 0
            for c in cuts:
                vec.append(c - prev)
                prev = c
            vec.append(total - prev)
            c = context_budget(vec, capacities)
            if c > best_c:
                best_c, best = c, vec
    return best_c, best


class TestObjective(CustomTestCase):
    def test_objective_is_the_documented_formula(self):
        caps = [100, 50, 25]
        vec = [4, 2, 1]
        # min(100//4, 50//2, 25//1) = 25 ; sum = 7
        self.assertEqual(context_budget(vec, caps), 25 * 7)

    def test_degenerate_vector_scores_zero(self):
        self.assertEqual(context_budget([1, 0, 1], [10, 10, 10]), 0)
        self.assertEqual(context_budget([], []), 0)
        self.assertEqual(context_budget([1, 1], [10]), 0)


class TestFloorIsStructural(CustomTestCase):
    """The floor is a hard constraint, never an objective term: no rank may
    ever be asked to hold more tokens than its capacity."""

    def test_no_rank_exceeds_its_capacity(self):
        cases = [
            MEASURED_P,
            [301435, 117912, 158474, 90000],
            [1000, 1000, 1000],
            [7, 7],
            [64, 1],
            [999983, 13, 500],
            [100, 100, 100, 100, 100],
        ]
        for caps in cases:
            with self.subTest(caps=caps):
                sol = solve_token_vector(caps)
                for r, (held, cap) in enumerate(
                    zip(sol.per_rank_tokens, sol.capacities)
                ):
                    self.assertLessEqual(
                        held,
                        cap,
                        f"rank {r} holds {held} > capacity {cap} (vector {sol.vector})",
                    )
                self.assertEqual(sol.context_tokens, sol.unit * sum(sol.vector))
                self.assertEqual(
                    sol.context_tokens, context_budget(sol.vector, sol.capacities)
                )

    def test_corridor_capacity_clamps_the_profiled_one(self):
        ranks = [
            RankCapacity(0, profiled_tokens=300000, corridor_tokens=200000),
            RankCapacity(1, profiled_tokens=100000, corridor_tokens=None),
            RankCapacity(2, profiled_tokens=100000, corridor_tokens=400000),
        ]
        self.assertEqual([r.effective_tokens for r in ranks], [200000, 100000, 100000])
        self.assertTrue(ranks[0].corridor_binds)
        self.assertFalse(ranks[1].corridor_binds)
        self.assertFalse(ranks[2].corridor_binds)
        sol = solve_corridor_vector(ranks)
        self.assertLessEqual(sol.per_rank_tokens[0], 200000)


class TestOptimality(CustomTestCase):
    def test_matches_brute_force_on_small_grain(self):
        grain = 12
        for caps in ([40, 25, 10], [7, 7, 7], [100, 3, 50], [64, 64, 1]):
            with self.subTest(caps=caps):
                sol = solve_token_vector(caps, grain=grain)
                best_c, _ = brute_force_best(caps, grain)
                self.assertEqual(sol.context_tokens, best_c)

    def test_never_worse_than_the_proportional_heuristic(self):
        """The pre-#602 hint was partition_units(64, P) gcd-reduced."""
        cases = [
            MEASURED_P,
            [301435, 117912, 158474, 90000],
            [123457, 65537, 32771],
            [50000, 50000, 50000],
            [11, 13, 17],
        ]
        for caps in cases:
            with self.subTest(caps=caps):
                legacy = partition_units(CORRIDOR_GRAIN, caps)
                sol = solve_token_vector(caps)
                self.assertGreaterEqual(
                    sol.context_tokens, context_budget(legacy, caps)
                )

    def test_beats_the_proportional_heuristic_where_rounding_bites(self):
        caps = [10**5, 3 * 10**4 + 7, 4 * 10**4 + 11]
        legacy = context_budget(partition_units(CORRIDOR_GRAIN, caps), caps)
        sol = solve_token_vector(caps)
        self.assertGreater(sol.context_tokens, legacy)

    def test_solution_never_exceeds_total_capacity(self):
        for caps in ([301435, 117912, 158474], [5, 5, 5], [97, 89, 83]):
            with self.subTest(caps=caps):
                sol = solve_token_vector(caps)
                self.assertLessEqual(sol.context_tokens, sum(caps))
                self.assertGreaterEqual(sol.total_waste_tokens, 0)

    def test_equal_capacities_reduce_to_the_uniform_vector(self):
        sol = solve_token_vector([50000, 50000, 50000])
        self.assertEqual(sol.vector, [1, 1, 1])

    def test_grain_is_respected(self):
        for grain in (3, 8, 64, 128):
            with self.subTest(grain=grain):
                sol = solve_token_vector(MEASURED_P, grain=grain)
                self.assertLessEqual(sum(sol.vector), grain)
                self.assertTrue(all(v >= 1 for v in sol.vector))


class TestDeterminism(CustomTestCase):
    def test_pure_function_of_its_inputs(self):
        a = solve_token_vector(MEASURED_P)
        b = solve_token_vector(list(MEASURED_P))
        self.assertEqual(a.vector, b.vector)
        self.assertEqual(a.context_tokens, b.context_tokens)

    def test_rank_order_comes_from_dcp_rank_not_list_order(self):
        ranks = [
            RankCapacity(2, 158474),
            RankCapacity(0, 301435),
            RankCapacity(1, 117912),
        ]
        sol = solve_corridor_vector(ranks)
        self.assertEqual(sol.capacities, MEASURED_P)


class TestInfeasible(CustomTestCase):
    def test_zero_capacity_is_named_not_silently_absorbed(self):
        with self.assertRaises(CorridorInfeasible) as ctx:
            solve_token_vector([100000, 0, 50000])
        self.assertIn("[1]", str(ctx.exception))

    def test_negative_capacity_is_named(self):
        with self.assertRaises(CorridorInfeasible):
            solve_token_vector([-5, 10, 10])

    def test_grain_too_small_for_the_rank_count(self):
        with self.assertRaises(CorridorInfeasible):
            solve_token_vector([10] * 5, grain=4)

    def test_incomplete_rank_cover_is_refused(self):
        with self.assertRaises(CorridorInfeasible):
            solve_corridor_vector([RankCapacity(0, 10), RankCapacity(2, 10)])


class TestCorridorPoolBytes(CustomTestCase):
    def test_reserve_and_post_sizing_are_both_subtracted(self):
        got = corridor_pool_bytes(
            free_bytes=8000 * MIB, reserve_mib=1024, post_sizing_mib=2000
        )
        self.assertEqual(got, (8000 - 1024 - 2000) * MIB)

    def test_colocated_ranks_split_the_remainder(self):
        one = corridor_pool_bytes(
            free_bytes=8000 * MIB, reserve_mib=1024, post_sizing_mib=2000
        )
        two = corridor_pool_bytes(
            free_bytes=8000 * MIB,
            reserve_mib=1024,
            post_sizing_mib=2000,
            colocated_ranks=2,
        )
        self.assertEqual(two, one // 2)

    def test_overcommitted_card_reports_zero_not_negative(self):
        self.assertEqual(
            corridor_pool_bytes(
                free_bytes=1000 * MIB, reserve_mib=1024, post_sizing_mib=500
            ),
            0,
        )

    def test_zero_bytes_makes_the_solver_refuse(self):
        with self.assertRaises(CorridorInfeasible):
            solve_corridor_vector(
                [
                    RankCapacity(0, 100000, corridor_tokens=0),
                    RankCapacity(1, 100000),
                ]
            )

    def test_bad_colocation_count_is_refused(self):
        with self.assertRaises(ValueError):
            corridor_pool_bytes(1 << 30, 1024, 0, colocated_ranks=0)


if __name__ == "__main__":
    unittest.main()
