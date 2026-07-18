"""Unit tests for the kv-boundary-aware Q-dimension split (task #116).

Pure functions, no GPU, no server. The auto planner (--rank-tp-ratio auto)
derives a memory-proportional weight vector; under REPLICATED-KV geometry
(TP > num_kv_heads) that vector can otherwise produce a q-head split whose
per-rank packets STRADDLE a global kv-head-group boundary, which the #105
current-chunk ragged kernel cannot represent (it fails fast in
_replicated_kv_ragged_reindex). Passing ``groups`` (= kv_total) constrains
the split so no rank straddles, while staying byte-identical whenever the
raw split is already aligned.
"""

import itertools
import random
import unittest

from sglang.srt.distributed.utils import (
    _partition_units_raw,
    attn_q_partition_groups,
    partition_sizes,
    partition_units,
    set_tp_partition_ratios,
    tp_partition_sizes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _straddles(sizes, units, groups):
    """True if any rank's contiguous unit range crosses a kv-group boundary
    (boundaries at multiples of units // groups)."""
    per = units // groups
    boundaries = {per * k for k in range(1, groups)}
    lo = 0
    for s in sizes:
        hi = lo + s  # rank owns units [lo, hi)
        # a boundary b strictly inside (lo, hi) means this rank straddles it
        if any(lo < b < hi for b in boundaries):
            return True
        lo = hi
    return False


class TestKvAlignedSplit(CustomTestCase):
    # ---- byte-identity: groups=None must equal the pre-#116 raw split ----

    def test_groups_none_is_byte_identical(self):
        random.seed(1)
        for _ in range(4000):
            tp = random.randint(2, 8)
            units = random.randint(tp, 32)
            w = [random.randint(1, 12) for _ in range(tp)]
            self.assertEqual(
                partition_units(units, w, None),
                _partition_units_raw(units, w),
            )

    def test_kv_ge_tp_falls_back_to_raw(self):
        # groups >= number of ranks: every rank already fits in one group,
        # no alignment repair possible/needed -> raw split (byte-identical).
        for units, tp, groups in [(8, 2, 4), (12, 3, 4), (16, 4, 8), (6, 2, 3)]:
            w = [random.randint(1, 9) for _ in range(tp)]
            self.assertEqual(
                partition_units(units, w, groups),
                _partition_units_raw(units, w),
            )

    def test_non_divisible_groups_falls_back_to_raw(self):
        # kv**2 does not divide q (units % groups != 0): alignment impossible,
        # return raw (the #105 guard then correctly rejects it downstream).
        for units, tp, groups in [(7, 5, 4), (10, 6, 3), (9, 5, 2)]:
            if units % groups == 0:
                continue
            w = [random.randint(1, 9) for _ in range(tp)]
            self.assertEqual(
                partition_units(units, w, groups),
                _partition_units_raw(units, w),
            )

    # ---- alignment invariant: no straddle under kv < tp ----

    def test_no_straddle_over_full_sweep(self):
        random.seed(2)
        checked = 0
        for _ in range(6000):
            tp = random.randint(3, 8)
            groups = random.randint(2, tp - 1)  # kv < tp
            per = random.randint(1, 5)
            units = groups * per
            if units < tp:
                continue  # partition_units needs >= 1 unit per rank
            w = [random.randint(1, 12) for _ in range(tp)]
            sizes = partition_units(units, w, groups)
            self.assertEqual(sum(sizes), units)
            self.assertTrue(all(s >= 1 for s in sizes))
            self.assertFalse(
                _straddles(sizes, units, groups),
                msg=f"straddle units={units} groups={groups} w={w} -> {sizes}",
            )
            checked += 1
        self.assertGreater(checked, 1000)

    def test_already_aligned_inputs_unchanged(self):
        # Explicit kv-aligned ratio (the #82 workaround 2,2,4,4,4 -> gcd
        # [1,1,2,2,2]) and even splits already respect the boundary: the
        # aligned path must return the IDENTICAL raw partition.
        cases = [
            (8, [1, 1, 2, 2, 2], 2),  # 2,2,4,4,4 reduced
            (8, [1, 1, 1, 1], 2),  # even, tp=4
            (12, [1, 1, 1, 2, 1], 3),
        ]
        for units, w, groups in cases:
            raw = _partition_units_raw(units, w)
            if _straddles(raw, units, groups):
                continue  # not an "already aligned" case; skip
            self.assertEqual(partition_units(units, w, groups), raw)

    # ---- the concrete A3B TP=5 3+1+1 co-location case ----

    def test_a3b_tp5_repair(self):
        # A3B: q=16, kv=2 -> units = q // kv = 8, groups = kv = 2 (per=4).
        # Memory-proportional weights [5,5,5,9,9] (3 ranks on the 5090 sharing
        # its budget, 2 ranks each on a 3080). The raw split straddles the
        # boundary at unit 4; the aligned split must not.
        units, groups = 8, 2
        for w in ([5, 5, 5, 9, 9], [1, 1, 1, 3, 2], [2, 2, 2, 6, 4]):
            sizes = partition_units(units, w, groups)
            self.assertFalse(_straddles(sizes, units, groups))
            # In head space (unit = kv_total = 2 heads): a clean aligned split.
            heads = [s * 2 for s in sizes]
            self.assertEqual(sum(heads), 16)

    def test_attn_q_partition_groups_gates_on_replication(self):
        # Only engages under an INSTALLED uneven plan (tp_plan_active) AND
        # kv < tp (REPLICATED-KV). Otherwise None -> byte-identical default.
        self.assertIsNone(attn_q_partition_groups(2, 5))  # no plan installed
        set_tp_partition_ratios([5, 5, 5, 9, 9])
        try:
            self.assertEqual(attn_q_partition_groups(2, 5), 2)  # kv < tp
            self.assertIsNone(attn_q_partition_groups(8, 5))  # kv >= tp
        finally:
            set_tp_partition_ratios(None)
        set_tp_partition_ratios([2, 1, 1])
        try:
            self.assertEqual(attn_q_partition_groups(2, 3), 2)  # kv < tp
            self.assertIsNone(attn_q_partition_groups(4, 3))  # kv >= tp
        finally:
            set_tp_partition_ratios(None)

    # ---- consistency through the installed-ratio wrappers ----

    def test_tp_partition_sizes_threads_groups(self):
        # With a ratio plan installed, tp_partition_sizes(...groups=kv) and
        # partition_sizes must agree, and the per-rank head split must be
        # straddle-free. Mirrors what qkv/o_proj/backends compute.
        set_tp_partition_ratios([5, 5, 5, 9, 9])
        try:
            q_total, kv, tp = 16, 2, 5
            units = q_total // kv  # 8
            groups = attn_q_partition_groups(kv, tp)  # 2
            sizes = tp_partition_sizes(q_total, tp, units=units, groups=groups)
            self.assertEqual(
                sizes, partition_sizes(q_total, [5, 5, 5, 9, 9], units, groups)
            )
            self.assertEqual(sum(sizes), q_total)
            # head-space straddle check (each unit = kv heads)
            self.assertFalse(_straddles([s // (q_total // units) for s in sizes], units, groups))
        finally:
            set_tp_partition_ratios(None)


if __name__ == "__main__":
    unittest.main()
