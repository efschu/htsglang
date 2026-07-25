"""Weighted uneven-DCP owner rule: read/write inverse, disjointness, packing.

This is the property that decides whether uneven DCP is CORRECT, and it is pure
integer arithmetic -- so it is testable on CPU, with no device and no
collectives. Worth pinning hard, because every failure mode here is SILENT: a
token gets fetched from a slot it was never written to, and the model emits
fluent nonsense instead of crashing.

Four properties:
  1. INVERSE      -- the slot the writer stores a token in is exactly the slot
                     the reader fetches it from.
  2. PARTITION    -- every cache slot is owned by exactly one rank (no gaps, no
                     double ownership).
  3. NO COLLISION -- within a rank, distinct owned slots map to distinct
                     physical slots (else tokens overwrite each other).
  4. COMPACTNESS  -- the physical slots a rank uses are densely packed and
                     proportional to its weight, which is the entire point of
                     weighting.
"""

import unittest

import torch

from sglang.srt.distributed.utils import (
    get_cp_token_ratios,
    set_cp_token_ratios,
)
from sglang.srt.layers.dcp.owner import (
    dcp_weighted_owner_bounds,
    dcp_weighted_write_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Non-uniform token plans worth exercising. [1,1,1] is included on purpose:
# the weighted rule must degenerate to the even modulo layout.
PLANS = ([2, 1, 1], [3, 2, 1], [4, 2, 2], [1, 1, 1], [5, 3])

# A multiple of lcm(sum(plan)) over all PLANS -- lcm(4,6,8,3,8) == 24 -- so the
# slot range covers a WHOLE number of virtual blocks for every plan. That
# matters for the compactness/proportionality property only: on a partial
# trailing block the low-offset ranks legitimately own one slot more than their
# exact share (e.g. 512 slots with cp_S=6 leaves 2 spare offsets, giving rank 0
# 257 rather than 256). The inverse/partition/collision properties hold for ANY
# slot count and are exercised over the same range.
N_SLOTS = 480


def _read_compact(loc: torch.Tensor, cp_S, cp_lo, cp_ratio):
    """Read-side mapping, transcribed from build_dcp_weighted_kv_indices."""
    off = loc % cp_S
    return (loc // cp_S) * cp_ratio + (off - cp_lo)


class TestDcpWeightedOwnerRule(CustomTestCase):
    def setUp(self):
        self._saved = get_cp_token_ratios()

    def tearDown(self):
        set_cp_token_ratios(self._saved)

    def test_read_and_write_are_exact_inverses(self):
        """Property 1. The one that makes the cache coherent."""
        locs = torch.arange(N_SLOTS, dtype=torch.int64)
        for plan in PLANS:
            set_cp_token_ratios(plan)
            for rank in range(len(plan)):
                cp_S, cp_lo, cp_hi, cp_ratio = dcp_weighted_owner_bounds(
                    len(plan), rank
                )
                write_loc, mask = dcp_weighted_write_slots(
                    locs, cp_S, cp_lo, cp_hi, cp_ratio
                )
                read_loc = _read_compact(locs, cp_S, cp_lo, cp_ratio)
                self.assertTrue(
                    torch.equal(write_loc[mask], read_loc[mask]),
                    f"plan={plan} rank={rank}: write and read slots disagree",
                )

    def test_every_slot_owned_by_exactly_one_rank(self):
        """Property 2. A gap loses tokens; an overlap corrupts them."""
        locs = torch.arange(N_SLOTS, dtype=torch.int64)
        for plan in PLANS:
            set_cp_token_ratios(plan)
            owners = torch.zeros(N_SLOTS, dtype=torch.int64)
            for rank in range(len(plan)):
                cp_S, cp_lo, cp_hi, cp_ratio = dcp_weighted_owner_bounds(
                    len(plan), rank
                )
                _, mask = dcp_weighted_write_slots(locs, cp_S, cp_lo, cp_hi, cp_ratio)
                owners += mask.to(torch.int64)
            self.assertTrue(
                torch.equal(owners, torch.ones(N_SLOTS, dtype=torch.int64)),
                f"plan={plan}: slots owned {sorted(set(owners.tolist()))} times, "
                "expected exactly once each",
            )

    def test_no_two_owned_slots_share_a_physical_slot(self):
        """Property 3. A collision silently overwrites a live token."""
        locs = torch.arange(N_SLOTS, dtype=torch.int64)
        for plan in PLANS:
            set_cp_token_ratios(plan)
            for rank in range(len(plan)):
                cp_S, cp_lo, cp_hi, cp_ratio = dcp_weighted_owner_bounds(
                    len(plan), rank
                )
                write_loc, mask = dcp_weighted_write_slots(
                    locs, cp_S, cp_lo, cp_hi, cp_ratio
                )
                owned = write_loc[mask]
                self.assertEqual(
                    owned.unique().numel(),
                    owned.numel(),
                    f"plan={plan} rank={rank}: physical slot collision",
                )

    def test_physical_slots_are_compact_and_weight_proportional(self):
        """Property 4. Weighting is pointless if the packing is not dense."""
        locs = torch.arange(N_SLOTS, dtype=torch.int64)
        for plan in PLANS:
            set_cp_token_ratios(plan)
            total = sum(plan)
            for rank, weight in enumerate(plan):
                cp_S, cp_lo, cp_hi, cp_ratio = dcp_weighted_owner_bounds(
                    len(plan), rank
                )
                write_loc, mask = dcp_weighted_write_slots(
                    locs, cp_S, cp_lo, cp_hi, cp_ratio
                )
                owned = write_loc[mask]
                # Dense: 0..n-1 with no holes.
                self.assertEqual(int(owned.min()), 0)
                self.assertEqual(
                    int(owned.max()),
                    owned.numel() - 1,
                    f"plan={plan} rank={rank}: physical slots are not dense",
                )
                # Proportional: this rank holds weight/total of the slots.
                self.assertEqual(
                    owned.numel(),
                    N_SLOTS * weight // total,
                    f"plan={plan} rank={rank}: expected a {weight}/{total} share",
                )

    def test_uniform_plan_degenerates_to_the_even_modulo_layout(self):
        """An all-ones plan must reproduce the classic even rule exactly:
        own L iff L % dcp_size == rank, stored at L // dcp_size."""
        locs = torch.arange(N_SLOTS, dtype=torch.int64)
        dcp = 3
        set_cp_token_ratios([1] * dcp)
        for rank in range(dcp):
            cp_S, cp_lo, cp_hi, cp_ratio = dcp_weighted_owner_bounds(dcp, rank)
            write_loc, mask = dcp_weighted_write_slots(
                locs, cp_S, cp_lo, cp_hi, cp_ratio
            )
            self.assertTrue(torch.equal(mask, (locs % dcp) == rank))
            self.assertTrue(torch.equal(write_loc[mask], (locs // dcp)[mask]))


if __name__ == "__main__":
    unittest.main()
