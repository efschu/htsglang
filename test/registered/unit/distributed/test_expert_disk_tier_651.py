"""#651: residency logic for the expert-to-disk tier.

The eviction path is the part that decides whether the feature is cheap or
ruinous, and it is invisible in a serving run -- a wrong LRU shows up as a
throughput number nobody can attribute. So it is tested here, on the CPU, where
it can be driven deterministically.
"""

import unittest

from sglang.srt.mem_cache.expert_disk_tier import (
    NOT_RESIDENT,
    ExpertDiskResidency,
    LayerResidency,
    bytes_saved,
    plan_hot_sets,
)


class TestLayerResidency(unittest.TestCase):
    def _layer(self, hot=(0, 1), staging=2, num_experts=8):
        return LayerResidency(
            num_experts=num_experts, hot_experts=hot, staging_rows=staging
        )

    def test_hot_experts_own_stable_rows(self):
        layer = self._layer()
        self.assertEqual(layer.row_of(0), 0)
        self.assertEqual(layer.row_of(1), 1)
        # Faulting cold experts must never move a hot expert's row, or the
        # remap table handed to the kernel would go stale mid-forward.
        for e in (4, 5, 6, 7):
            layer.acquire(e)
        self.assertEqual(layer.row_of(0), 0)
        self.assertEqual(layer.row_of(1), 1)

    def test_cold_expert_is_not_resident_until_acquired(self):
        layer = self._layer()
        self.assertEqual(layer.row_of(5), NOT_RESIDENT)
        res = layer.acquire(5)
        self.assertTrue(res.copied_from_disk)
        self.assertNotEqual(layer.row_of(5), NOT_RESIDENT)

    def test_hot_acquire_is_a_hit_and_copies_nothing(self):
        layer = self._layer()
        res = layer.acquire(1)
        self.assertFalse(res.copied_from_disk)
        self.assertIsNone(res.evicted)
        self.assertEqual(layer.hits, 1)
        self.assertEqual(layer.faults, 0)

    def test_second_use_of_a_staged_expert_is_a_hit(self):
        layer = self._layer()
        layer.acquire(5)
        res = layer.acquire(5)
        self.assertFalse(res.copied_from_disk)
        self.assertEqual(layer.faults, 1)
        self.assertEqual(layer.hits, 1)

    def test_staging_pool_evicts_least_recently_used(self):
        layer = self._layer(hot=(0, 1), staging=2)
        layer.acquire(4)  # staged
        layer.acquire(5)  # staged, pool now full
        layer.acquire(4)  # refresh 4, so 5 is now the LRU
        res = layer.acquire(6)
        self.assertEqual(res.evicted, 5)
        self.assertEqual(layer.row_of(5), NOT_RESIDENT)
        self.assertNotEqual(layer.row_of(4), NOT_RESIDENT)

    def test_eviction_frees_the_row_for_reuse(self):
        layer = self._layer(hot=(0,), staging=1)
        r1 = layer.acquire(3).row
        r2 = layer.acquire(4).row
        self.assertEqual(r1, r2)
        self.assertEqual(layer.row_of(3), NOT_RESIDENT)

    def test_total_rows_is_hot_plus_staging(self):
        self.assertEqual(self._layer(hot=(0, 1, 2), staging=4).total_rows, 7)

    def test_zero_staging_rows_is_refused(self):
        # Without a staging row a cold expert could never be served at all.
        with self.assertRaises(ValueError):
            LayerResidency(num_experts=8, hot_experts=(0,), staging_rows=0)

    def test_out_of_range_expert_is_refused(self):
        layer = self._layer()
        with self.assertRaises(ValueError):
            layer.acquire(99)
        with self.assertRaises(ValueError):
            LayerResidency(num_experts=4, hot_experts=(9,), staging_rows=1)

    def test_duplicate_hot_experts_refused(self):
        with self.assertRaises(ValueError):
            LayerResidency(num_experts=8, hot_experts=(1, 1), staging_rows=1)

    def test_refresh_hot_changes_residency(self):
        layer = self._layer(hot=(0, 1), staging=1)
        layer.acquire(7)
        layer.refresh_hot([6, 7])
        self.assertEqual(layer.row_of(7), 1)
        # 0 was hot and is not any more; it must now be a fault, not a stale row.
        self.assertEqual(layer.row_of(0), NOT_RESIDENT)

    def test_miss_rate_reports_faults_over_lookups(self):
        layer = self._layer(hot=(0,), staging=1)
        layer.acquire(0)
        layer.acquire(0)
        layer.acquire(5)
        self.assertAlmostEqual(layer.miss_rate(), 1 / 3)


class TestPlanning(unittest.TestCase):
    def test_plan_keeps_the_busiest_experts(self):
        counts = [[0, 50, 0, 7]]
        plan = plan_hot_sets(counts, hot_fraction=0.5)
        self.assertEqual(plan[0], [1, 3])

    def test_plan_keeps_at_least_one_expert(self):
        counts = [[1, 2, 3, 4]]
        plan = plan_hot_sets(counts, hot_fraction=0.01)
        self.assertEqual(len(plan[0]), 1)

    def test_bad_fraction_refused(self):
        for bad in (0.0, -0.5, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plan_hot_sets([[1, 2]], hot_fraction=bad)

    def test_bytes_saved_charges_for_staging_rows(self):
        # 10 experts, keep 2 hot + 2 staging -> only 6 rows are actually absent.
        saved = bytes_saved(
            num_layers=1,
            num_experts=10,
            bytes_per_expert=100,
            hot_fraction=0.2,
            staging_rows=2,
        )
        self.assertEqual(saved, 600)

    def test_bytes_saved_is_zero_when_everything_is_resident(self):
        saved = bytes_saved(
            num_layers=40,
            num_experts=10,
            bytes_per_expert=100,
            hot_fraction=1.0,
            staging_rows=2,
        )
        self.assertEqual(saved, 0)

    def test_measured_census_shape_is_cheap(self):
        """The #651 census said the coldest 20% take 0.03% of lookups.

        Reproduced as a planning check: with a matching synthetic
        distribution, keeping 80% hot must leave the cold traffic tiny.
        """
        # 100 experts: 80 busy, 20 nearly dead -- the measured shape.
        counts = [[1000] * 80 + [1] * 20]
        plan = plan_hot_sets(counts, hot_fraction=0.8)
        self.assertEqual(len(plan[0]), 80)
        self.assertNotIn(99, plan[0])


class TestExpertDiskResidency(unittest.TestCase):
    def test_layers_are_independent(self):
        res = ExpertDiskResidency(
            num_layers=2,
            num_experts=8,
            hot_per_layer={0: (0, 1), 1: (2, 3)},
            staging_rows=1,
        )
        self.assertEqual(res.row_of(0, 0), 0)
        self.assertEqual(res.row_of(1, 0), NOT_RESIDENT)
        self.assertEqual(res.row_of(1, 2), 0)

    def test_totals_aggregate_across_layers(self):
        res = ExpertDiskResidency(
            num_layers=2, num_experts=8, hot_per_layer={0: (0,), 1: (0,)},
            staging_rows=1,
        )
        res.acquire(0, 0)  # hit
        res.acquire(1, 5)  # fault
        totals = res.totals()
        self.assertEqual(totals["hits"], 1)
        self.assertEqual(totals["faults"], 1)
        self.assertAlmostEqual(totals["miss_rate"], 0.5)

    def test_refresh_from_counts_follows_live_traffic(self):
        """Residency must track the workload, not the one-off census."""
        res = ExpertDiskResidency(
            num_layers=1, num_experts=4, hot_per_layer={0: (0, 1)}, staging_rows=1
        )
        self.assertEqual(res.row_of(0, 3), NOT_RESIDENT)
        # Live counters say 2 and 3 are the busy ones now.
        res.refresh_from_counts([[0, 0, 90, 80]], hot_fraction=0.5)
        self.assertNotEqual(res.row_of(0, 3), NOT_RESIDENT)
        self.assertEqual(res.row_of(0, 0), NOT_RESIDENT)


if __name__ == "__main__":
    unittest.main()
