# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#771: the seam's staging ask must be a POST IN THE POOL SOLVE.

WHAT GOES WRONG WITHOUT IT. The KV backing relief rung is deliberately
all-or-nothing: ``collective_kv_backing_relief`` grants nothing when the
MANDATORY ask exceeds what the rung can return above its admission floor,
because laundering a short ask into a seam that then does not fit is worse
than a clean abandon (phase_flip_spill.py, "THE MANDATORY HALF IS NEVER
BOUNDED"). That rule is right. What was wrong is the SIZING underneath it: a
pool solved without the staging term leaves a mandatory ask nothing can fund,
so the rung reports it could return 1902-2037 MiB, grants 0, and the seam
abandons -- measured 72 times in one boot with ``evicted 0 rows over 0
shrinks so far`` while the flip never happened.

This is the same class as the arming floor before it (see
``arming_floor_target_bytes``): a term the planner had every input for and did
not charge, producing a healthy-looking boot that simply never flips, with no
runtime recovery because the pool is fixed at boot.

SPECIMEN NUMBERS are this rig's boot projection: 840 MiB on rank 0 and
1617 MiB on rank 1 ("needs N MiB free" at the ladder's arming threshold).
"""

from __future__ import annotations

import unittest

from sglang.srt.managers import phase_flip_seam_reserve as seam

MIB = 1 << 20


class TestTheStagingAskIsCharged(unittest.TestCase):
    def test_the_specimen_asks_are_charged_in_full(self):
        # Rank 0 and rank 1 from this rig's boot-time STAGING PROJECTION.
        self.assertGreaterEqual(seam.staging_post_bytes(840 * MIB), 840 * MIB)
        self.assertGreaterEqual(seam.staging_post_bytes(1617 * MIB), 1617 * MIB)

    def test_an_unknown_ask_charges_nothing_rather_than_guessing(self):
        # A projection that could not be computed must not become a silent
        # fabricated number; it charges 0 and the caller says so out loud.
        self.assertEqual(seam.staging_post_bytes(0), 0)
        self.assertEqual(seam.staging_post_bytes(None), 0)

    def test_a_negative_ask_is_not_a_credit(self):
        self.assertEqual(seam.staging_post_bytes(-5 * MIB), 0)

    def test_the_entry_margin_rides_on_top_when_given(self):
        # The gate compares against staging + entry margin, so the pool must
        # reserve for the number the gate will actually enforce -- the same
        # reasoning the arming floor uses.
        got = seam.staging_post_bytes(840 * MIB, entry_margin_bytes=512 * MIB)
        self.assertEqual(got, (840 + 512) * MIB)


class TestItComposesWithTheArmingFloor(unittest.TestCase):
    """The two terms are different quantities and must both be charged."""

    def test_the_staging_post_is_additive_to_the_floor(self):
        floor = seam.arming_floor_target_bytes(configured_mib=1536)
        post = seam.staging_post_bytes(1617 * MIB)
        total = seam.pool_flip_posts_bytes(
            arming_floor_bytes=floor, staging_post_bytes_=post
        )
        self.assertEqual(total, floor + post)

    def test_a_cold_projection_leaves_the_floor_intact(self):
        floor = seam.arming_floor_target_bytes(configured_mib=1536)
        total = seam.pool_flip_posts_bytes(
            arming_floor_bytes=floor, staging_post_bytes_=0
        )
        self.assertEqual(total, floor)


if __name__ == "__main__":
    unittest.main()
