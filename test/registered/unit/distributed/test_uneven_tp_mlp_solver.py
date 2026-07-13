"""Unit tests for the uneven-TP self-calibration solver
(solve_unit_rebalance / suggest_unit_rebalance in
sglang.srt.distributed.utils) — pure functions, no GPU, no torch.

The solver models the KV-pool maximin problem: rank r's token capacity is
(free_bytes + shed_mlp_units * bytes_per_unit) / bytes_per_token, and the
scheduler pool is the MIN over ranks. Includes the real measurement that
motivated the feature (Qwen3.6-27B FP8, TP=3 auto on 32+20+20 GB, fp8 KV:
max_total_num_tokens=594999, leftover free 2.38/1.15/2.29 GB -> TP1 pins,
~3.5 GB stranded): the suggested shift must move units AWAY from TP1 and
raise the projected minimum.
"""

import unittest

from sglang.srt.distributed.utils import (
    solve_unit_rebalance,
    solve_unit_rebalance_multi,
    suggest_unit_rebalance,
    suggest_unit_rebalance_multi,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GB = 1 << 30


def _capacities(free, bpt, units0, units, bpu):
    return [
        (free[r] + (units0[r] - units[r]) * bpu[r]) / bpt[r]
        for r in range(len(units))
    ]


def _real_measurement():
    """The measured baseline, reconstructed as solver inputs.

    TP=3 auto (5090 32GB + 2x3080 20GB), agreed pool 594999 tokens =
    TP1's capacity; TP0/TP2 had 2.38/2.29 GB free vs TP1's 1.15 GB.
    bytes/token proportional to the kv-head share [4,2,2]; MLP units
    partitioned ~[5,3,3]; one MLP unit ~ 48 layers x 3 mats x 5120
    hidden x 1 byte (fp8).
    """
    bpt = [26000.0, 13000.0, 13000.0]
    pinned = 594999
    free = [
        pinned * bpt[0] + (2.38 - 1.15) * GB,
        pinned * bpt[1],
        pinned * bpt[2] + (2.29 - 1.15) * GB,
    ]
    units = [4608, 2765, 2765]
    bytes_per_unit = [48 * 3 * 5120.0] * 3
    family_bytes = [units[r] * bytes_per_unit[r] for r in range(3)]
    return free, bpt, units, bytes_per_unit, family_bytes


class TestSolveUnitRebalance(CustomTestCase):
    def test_balanced_input_is_unchanged(self):
        free = [1000.0 * 100, 1000.0 * 100]
        bpt = [100.0, 100.0]
        units = [8, 8]
        bpu = [50.0, 50.0]
        new_units, projected = solve_unit_rebalance(free, bpt, units, bpu)
        self.assertEqual(new_units, units)
        self.assertEqual(projected, 1000)

    def test_real_measurement_shifts_away_from_tp1(self):
        free, bpt, units, bpu, _ = _real_measurement()
        new_units, projected = solve_unit_rebalance(free, bpt, units, bpu)
        # Units move AWAY from the pinned rank 1 ...
        self.assertLess(new_units[1], units[1])
        # ... total unit count is conserved ...
        self.assertEqual(sum(new_units), sum(units))
        # ... and the projected minimum rises meaningfully above 594999.
        self.assertGreater(projected, 594999)
        self.assertGreater(projected, 630000)
        # The projection is consistent with the capacity model.
        caps = _capacities(free, bpt, units, new_units, bpu)
        self.assertEqual(projected, int(min(caps)))

    def test_min_never_decreases(self):
        # Invariant sweep: the greedy result never falls below the
        # initial minimum capacity, and every rank keeps >= 1 unit.
        cases = [
            ([9e9, 4e9, 8e9], [20000.0, 10000.0, 10000.0], [10, 6, 6], [7e5] * 3),
            ([5e9, 5e9], [10000.0, 10000.0], [4, 12], [1e6] * 2),
            ([1e9, 30e9], [8000.0, 8000.0], [64, 64], [1e5] * 2),
            ([2e9, 2e9, 2e9, 20e9], [9000.0] * 4, [8, 8, 8, 8], [2e5] * 4),
        ]
        for free, bpt, units, bpu in cases:
            initial_min = min(free[r] / bpt[r] for r in range(len(units)))
            new_units, projected = solve_unit_rebalance(free, bpt, units, bpu)
            self.assertGreaterEqual(projected, int(initial_min) - 1)
            self.assertEqual(sum(new_units), sum(units))
            self.assertTrue(all(u >= 1 for u in new_units))

    def test_one_unit_floor_is_respected(self):
        # The pinned rank cannot shed below 1 unit, however large the
        # imbalance — partition_units requires >= 1 unit per rank.
        free = [0.0, 100e9]
        bpt = [10000.0, 10000.0]
        units = [2, 2]
        bpu = [1e6, 1e6]
        new_units, _ = solve_unit_rebalance(free, bpt, units, bpu)
        self.assertGreaterEqual(new_units[0], 1)
        self.assertEqual(sum(new_units), 4)

    def test_quant_block_coarse_units(self):
        # Quant-block-coarsened families have FEW units of LARGE byte
        # size; the greedy must still improve in whole units only.
        free = [10e9, 2e9]
        bpt = [10000.0, 10000.0]
        units = [6, 6]
        bpu = [0.7e9, 0.7e9]  # 0.7 GB per unit
        initial_min = min(free[r] / bpt[r] for r in range(2))
        new_units, projected = solve_unit_rebalance(free, bpt, units, bpu)
        self.assertEqual(sum(new_units), 12)
        self.assertGreater(projected, int(initial_min))
        # Rank 1 sheds, rank 0 receives.
        self.assertLess(new_units[1], 6)
        self.assertGreater(new_units[0], 6)

    def test_length_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            solve_unit_rebalance([1.0, 2.0], [1.0], [1, 1], [1.0, 1.0])

    def test_invalid_inputs_raise(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            solve_unit_rebalance([1e9, 1e9], [0.0, 1.0], [4, 4], [1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "must own"):
            solve_unit_rebalance([1e9, 1e9], [1.0, 1.0], [0, 4], [1.0, 1.0])


class TestSuggestUnitRebalance(CustomTestCase):
    def test_balanced_returns_none(self):
        # max/min <= 1.10: no hint (in particular an ACTIVE vector that
        # already balances the ranks stays silent).
        free = [105.0 * 1e6, 100.0 * 1e6]
        bpt = [1000.0, 1000.0]
        self.assertIsNone(
            suggest_unit_rebalance(free, bpt, [8, 8], [1e8, 1e8])
        )

    def test_exactly_at_threshold_returns_none(self):
        free = [110.0 * 1e6, 100.0 * 1e6]
        bpt = [1000.0, 1000.0]
        self.assertIsNone(
            suggest_unit_rebalance(free, bpt, [8, 8], [1e8, 1e8])
        )

    def test_real_measurement_yields_hint(self):
        free, bpt, units, _, family_bytes = _real_measurement()
        result = suggest_unit_rebalance(free, bpt, units, family_bytes)
        self.assertIsNotNone(result)
        new_units, cur_min, projected = result
        self.assertEqual(cur_min, 594999)
        self.assertLess(new_units[1], units[1])  # shift away from TP1
        self.assertGreater(projected, cur_min)

    def test_uncalibratable_inputs_return_none(self):
        # Degenerate inputs must never produce a hint.
        self.assertIsNone(
            suggest_unit_rebalance([1e9], [1000.0], [8], [1e8])
        )  # single rank
        self.assertIsNone(
            suggest_unit_rebalance([1e9, 2e9], [1000.0, 1000.0], [0, 8], [1e8, 1e8])
        )  # zero units
        self.assertIsNone(
            suggest_unit_rebalance([1e9, 2e9], [1000.0, 1000.0], [8, 8], [0.0, 1e8])
        )  # zero family bytes
        self.assertIsNone(
            suggest_unit_rebalance([0.0, 2e9], [1000.0, 1000.0], [8, 8], [1e8, 1e8])
        )  # pinned rank has zero capacity

    def test_no_gain_returns_none(self):
        # Imbalance above threshold but bytes_per_unit too small for a
        # whole-token gain: the solver finds no strictly better partition.
        free = [200.0 * 1e6, 100.0 * 1e6]
        bpt = [1000.0, 1000.0]
        # 2 units per rank, each worth ~1 token: shedding the single
        # allowed unit gains < 1 token after int() truncation.
        self.assertIsNone(
            suggest_unit_rebalance(free, bpt, [2, 2], [800.0, 800.0])
        )

    def test_hint_matches_solver(self):
        free, bpt, units, bpu, family_bytes = _real_measurement()
        expected_units, expected_projected = solve_unit_rebalance(
            free, bpt, units, bpu
        )
        result = suggest_unit_rebalance(free, bpt, units, family_bytes)
        self.assertEqual(result[0], expected_units)
        self.assertEqual(result[2], expected_projected)


class TestMultiFamilyRebalance(CustomTestCase):
    """Joint solving over the "mlp" and "moe" families: more families do
    not raise the sum(free)/sum(bpt) ceiling, they supply the shiftable
    mass needed to reach it."""

    def test_single_family_matches_wrapper(self):
        free, bpt, units, bpu, _ = _real_measurement()
        single = solve_unit_rebalance(free, bpt, units, bpu)
        multi = solve_unit_rebalance_multi(free, bpt, {"mlp": (units, bpu)})
        self.assertEqual(multi[0]["mlp"], single[0])
        self.assertEqual(multi[1], single[1])

    def test_moe_supplies_mass_when_mlp_is_too_small(self):
        # The dense-MLP family alone cannot unpin rank 1 (tiny bytes per
        # unit and a 1-unit floor); adding the byte-heavy moe family must
        # push the projected minimum decisively further.
        free, bpt, units, _, _ = _real_measurement()
        tiny_mlp_bpu = [512.0] * 3
        moe_bpu = [8 * 48 * 3 * 5120.0] * 3
        mlp_only, proj_mlp_only = solve_unit_rebalance_multi(
            free, bpt, {"mlp": (units, tiny_mlp_bpu)}
        )
        _, proj_joint = solve_unit_rebalance_multi(
            free,
            bpt,
            {"mlp": (units, tiny_mlp_bpu), "moe": (units, moe_bpu)},
        )
        self.assertGreater(proj_joint, proj_mlp_only)
        self.assertGreater(proj_joint, 630_000)

    def test_ceiling_is_conserved(self):
        # sum(free)/sum(bpt) bounds the projection regardless of how many
        # families are shiftable (weight moves conserve total free bytes).
        free, bpt, units, bpu, _ = _real_measurement()
        ceiling = sum(free) / sum(bpt)
        for families in (
            {"mlp": (units, bpu)},
            {"mlp": (units, bpu), "moe": (units, [b * 8 for b in bpu])},
        ):
            _, projected = solve_unit_rebalance_multi(free, bpt, families)
            self.assertLessEqual(projected, int(ceiling) + 1)

    def test_units_conserved_per_family(self):
        free, bpt, units, bpu, _ = _real_measurement()
        new_units, _ = solve_unit_rebalance_multi(
            free,
            bpt,
            {"mlp": (units, bpu), "moe": (units, [b * 8 for b in bpu])},
        )
        for name in ("mlp", "moe"):
            self.assertEqual(sum(new_units[name]), sum(units))
            self.assertTrue(all(u >= 1 for u in new_units[name]))

    def test_suggest_multi_reports_only_changed_families(self):
        free, bpt, units, _, _ = _real_measurement()
        # moe dominates; the tiny mlp family may stay untouched.
        families = {
            "mlp": (units, [u * 512.0 for u in units]),
            "moe": (units, [u * 8 * 48 * 3 * 5120.0 for u in units]),
        }
        result = suggest_unit_rebalance_multi(free, bpt, families)
        self.assertIsNotNone(result)
        changed, cur_min, projected = result
        self.assertEqual(cur_min, 594999)
        self.assertGreater(projected, cur_min)
        self.assertIn("moe", changed)
        for name, vec in changed.items():
            self.assertNotEqual(vec, units)
            self.assertEqual(sum(vec), sum(units))
        # The byte-dominant moe family sheds from the pinned TP1 (the
        # tiny mlp family may legitimately flow the other way once moe
        # moves have unpinned TP1).
        self.assertLess(changed["moe"][1], units[1])

    def test_suggest_multi_drops_degenerate_family(self):
        # A degenerate family (zero bytes) must not block calibration of
        # the healthy one.
        free, bpt, units, bpu, family_bytes = _real_measurement()
        families = {
            "mlp": (units, family_bytes),
            "moe": (units, [0.0] * 3),
        }
        result = suggest_unit_rebalance_multi(free, bpt, families)
        self.assertIsNotNone(result)
        changed, _, _ = result
        self.assertEqual(set(changed), {"mlp"})

    def test_suggest_multi_balanced_returns_none(self):
        free = [100.0 * 1e6, 100.0 * 1e6]
        bpt = [1000.0, 1000.0]
        self.assertIsNone(
            suggest_unit_rebalance_multi(
                free, bpt, {"mlp": ([8, 8], [1e8, 1e8])}
            )
        )


if __name__ == "__main__":
    unittest.main()
