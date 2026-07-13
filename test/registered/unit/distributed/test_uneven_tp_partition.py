"""Unit tests for the uneven-TP shard-plan helpers in
sglang.srt.distributed.utils — pure functions, no GPU, no server."""

import unittest

from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    partition_offsets,
    partition_sizes,
    partition_units,
    set_tp_partition_ratios,
    tp_partition_offset,
    tp_partition_size,
    tp_partition_sizes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPartitionUnits(CustomTestCase):
    def _check_invariants(self, units, weights):
        sizes = partition_units(units, weights)
        self.assertEqual(sum(sizes), units)
        self.assertTrue(all(s >= 1 for s in sizes))
        self.assertEqual(len(sizes), len(weights))
        # Deterministic pure function.
        self.assertEqual(sizes, partition_units(units, weights))
        return sizes

    def test_even_weights_reproduce_even_split(self):
        self.assertEqual(partition_units(8, [1, 1]), [4, 4])
        self.assertEqual(partition_units(24, [1, 1, 1, 1]), [6, 6, 6, 6])

    def test_proportional_split(self):
        self.assertEqual(partition_units(24, [2, 1, 1]), [12, 6, 6])
        self.assertEqual(partition_units(48, [3, 1]), [36, 12])

    def test_remainder_distributed_deterministically(self):
        # 10 over (2, 1, 1): quotas 5.0/2.5/2.5 -> [5, 2, 2] + 1 remainder;
        # ties in the fractional parts break toward the lower rank index.
        self.assertEqual(partition_units(10, [2, 1, 1]), [5, 3, 2])

    def test_minimum_one_unit_per_rank(self):
        # Extreme imbalance: small ranks still get one unit each.
        sizes = self._check_invariants(3, [100, 1, 1])
        self.assertEqual(sizes, [1, 1, 1])
        sizes = self._check_invariants(4, [1000, 1, 1])
        self.assertEqual(sizes, [2, 1, 1])

    def test_single_weight(self):
        self.assertEqual(partition_units(7, [3]), [7])

    def test_large_imbalance(self):
        sizes = self._check_invariants(64, [97, 3])
        self.assertEqual(sizes[0] + sizes[1], 64)
        self.assertGreater(sizes[0], sizes[1])
        self.assertGreaterEqual(sizes[1], 1)

    def test_fewer_units_than_ranks_raises(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            partition_units(2, [1, 1, 1])

    def test_invariants_sweep(self):
        for units in (3, 4, 7, 16, 24, 40, 64, 96):
            for weights in (
                [1, 1],
                [2, 1],
                [2, 1, 1],
                [5, 3, 2],
                [7, 1, 1, 1],
                [3622, 2247, 2247],
            ):
                if units < len(weights):
                    continue
                self._check_invariants(units, weights)


class TestPartitionSizes(CustomTestCase):
    def test_exact_division_by_weight_sum(self):
        self.assertEqual(partition_sizes(48, [2, 1, 1]), [24, 12, 12])
        self.assertEqual(partition_sizes(30, [3, 2]), [18, 12])

    def test_indivisible_raises_and_names_dimension(self):
        with self.assertRaisesRegex(ValueError, "50"):
            partition_sizes(50, [2, 1, 1])

    def test_units_mode_rounds_whole_units(self):
        # 8 heads of 16 elements each over (3, 1): partition_units(8, [3, 1])
        # = [6, 2], scaled by 16.
        self.assertEqual(partition_sizes(128, [3, 1], units=8), [96, 32])

    def test_units_mode_requires_multiple_of_units(self):
        with self.assertRaisesRegex(ValueError, "130"):
            partition_sizes(130, [3, 1], units=8)

    def test_sum_and_positivity(self):
        for total, weights, units in (
            (48, [2, 1, 1], None),
            (128, [3, 1], 8),
            (24, [5, 3], 24),
        ):
            sizes = partition_sizes(total, weights, units=units)
            self.assertEqual(sum(sizes), total)
            self.assertTrue(all(s >= 1 for s in sizes))


class TestPartitionOffsets(CustomTestCase):
    def test_prefix_sums(self):
        # Offsets must be the prefix sums of the sizes: contiguous,
        # monotonically increasing, covering [0, total).
        weights = [2, 1, 1]
        total = 48
        sizes = partition_sizes(total, weights)
        expected_offset = 0
        for rank in range(len(weights)):
            offset, size = partition_offsets(total, weights, rank)
            self.assertEqual(offset, expected_offset)
            self.assertEqual(size, sizes[rank])
            expected_offset += size
        self.assertEqual(expected_offset, total)

    def test_monotone_offsets_units_mode(self):
        weights = [5, 3, 2]
        total = 160  # 16 units of 10
        prev_end = 0
        for rank in range(3):
            offset, size = partition_offsets(total, weights, rank, units=16)
            self.assertEqual(offset, prev_end)
            self.assertGreaterEqual(size, 1)
            prev_end = offset + size
        self.assertEqual(prev_end, total)


class TestGlobalShardPlan(CustomTestCase):
    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_default_path_is_even_split(self):
        # No installed plan: identical to divide()-based behavior.
        self.assertIsNone(get_tp_partition_ratios())
        self.assertEqual(tp_partition_sizes(48, 4), [12, 12, 12, 12])
        self.assertEqual(tp_partition_size(48, 4, 3), 12)
        self.assertEqual(tp_partition_offset(48, 4, 3), 36)

    def test_default_path_keeps_divisibility_assert(self):
        with self.assertRaises(AssertionError):
            tp_partition_sizes(50, 4)

    def test_installed_plan_is_used(self):
        set_tp_partition_ratios([2, 1, 1])
        self.assertEqual(get_tp_partition_ratios(), [2, 1, 1])
        self.assertEqual(tp_partition_sizes(48, 3), [24, 12, 12])
        self.assertEqual(tp_partition_size(48, 3, 0), 24)
        self.assertEqual(tp_partition_offset(48, 3, 2), 36)

    def test_plan_ignored_for_other_tp_size(self):
        # Layers running with their own tp_size (e.g. disable_tp layers use
        # tp_size=1) must keep the classic even split.
        set_tp_partition_ratios([2, 1, 1])
        self.assertEqual(tp_partition_sizes(48, 1), [48])
        self.assertEqual(tp_partition_sizes(48, 4), [12, 12, 12, 12])

    def test_unset_plan_restores_default(self):
        set_tp_partition_ratios([2, 1, 1])
        set_tp_partition_ratios(None)
        self.assertIsNone(get_tp_partition_ratios())
        self.assertEqual(tp_partition_sizes(48, 3), [16, 16, 16])

    def test_empty_plan_treated_as_none(self):
        set_tp_partition_ratios([])
        self.assertIsNone(get_tp_partition_ratios())


if __name__ == "__main__":
    unittest.main()
