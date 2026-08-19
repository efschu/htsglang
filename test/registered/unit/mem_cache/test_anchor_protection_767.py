# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#767: anchor protection may not depend on --mamba-checkpoint-interval.

THE PREMISE THAT WAS WRONG. ``protect_deepest_anchors`` returned False whenever
no interval was configured, on the reasoning "no interval means no grid, so
there are no anchors either way". There are. The ``no_buffer`` path donates a
checkpoint for every finished request at ``cache_len = len(token_ids)``,
interval or not, and ``evict_mamba``'s own docstring says what losing the
deepest one costs: it "silently moves the resume point of identical requests
and re-introduces run-to-run drift".

MEASURED, one commit, short-prompt determinism gate (identical temp-0 requests,
which must all return the same bytes):

    48 slots, idle, 10 probes ->  1 distinct   PASS
    48 slots, idle, 20 probes ->  2 distinct
    48 slots, under 4-way load -> 7 distinct + 1 degenerate
    12 slots, idle, 10 probes ->  3 distinct

Divergence scales with slot pressure, which is what eviction responds to. The
interval only adds MORE anchors on top; it is not what makes the deepest one
worth keeping.

WHAT STAYS. The host-tier branch is untouched and is the one real exemption: a
node whose device value is dropped is still matchable and is loaded back, so
the resume point does not move and there is nothing to protect against.
"""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.mamba_ckpt_utils import protect_deepest_anchors


class TestProtectionDoesNotDependOnTheInterval(unittest.TestCase):
    def test_a_device_only_pool_protects_without_an_interval(self):
        # THE #767 REGRESSION. This returned False, so nothing was spared and
        # eviction under pressure moved the resume point run to run.
        self.assertTrue(protect_deepest_anchors(None, host_tier_present=False))

    def test_an_interval_still_protects(self):
        self.assertTrue(protect_deepest_anchors(8192, host_tier_present=False))

    def test_zero_is_not_a_special_case(self):
        # 0 is not "no anchors"; it is a degenerate grid. Either way the
        # deepest checkpoint of a path is still the resume point.
        self.assertTrue(protect_deepest_anchors(0, host_tier_present=False))


class TestTheHostTierExemptionSurvives(unittest.TestCase):
    """Untouched: a spilled anchor is matchable and reloads, so it may go."""

    def test_a_host_tier_does_not_protect_with_an_interval(self):
        self.assertFalse(protect_deepest_anchors(8192, host_tier_present=True))

    def test_a_host_tier_does_not_protect_without_an_interval(self):
        self.assertFalse(protect_deepest_anchors(None, host_tier_present=True))


if __name__ == "__main__":
    unittest.main()
