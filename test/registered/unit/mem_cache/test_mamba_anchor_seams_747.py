"""#747 seam parity: one grid rule, two cache lineages.

`MambaRadixCache` honours `--mamba-checkpoint-interval`; the unified tree's
mamba component does not know the flag exists. That divergence is why
`server_args.py`'s refusal was written, and why its stated reason names a class
(`HiMambaRadixCache`) that has no construction site: the two lineages drifted
and nobody could see it. See NOTE_747.

So the decisions live in ONE place -- `mem_cache/mamba_ckpt_utils.py`, pure
integer arithmetic -- and both lineages call it. These tests pin the rules
against the `mamba_radix_cache.py` lines they mirror, so a future edit to
either lineage that changes an anchor decision has to change this file too.

THE ONE GENUINELY NEW DECISION is the eviction protection, because the premise
behind it changes when a host tier exists:

* device-only: an evicted anchor is a DEAD anchor, so `MambaRadixCache` spares
  the deepest anchors of every path -- "losing the deepest one silently moves
  the resume point of identical requests and re-introduces run-to-run drift"
  (`mamba_radix_cache.py:1092-1097`);
* host tier present: an evicted anchor still matches and loads back
  (`mamba_component.py:71-74`, `:139-144`), so it is not lost and the
  protection is not needed to keep the resume point stable.

That relaxation is deliberate and is implemented as an explicit branch, never
as a silent difference -- both directions are pinned below.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import unittest

from sglang.srt.mem_cache.mamba_ckpt_utils import (
    floor_to_interval,
    is_on_interval,
    protect_deepest_anchors,
)
from sglang.test.test_utils import CustomTestCase

#: The cadence #747 documents: 8192 tokens = 16 x chunked_prefill_size(512).
CADENCE = 8192
CHUNK = 512


class TestTheEvictionProtectionBranch(CustomTestCase):
    """The new decision. Both directions pinned."""

    def test_device_only_keeps_todays_protection(self):
        self.assertTrue(protect_deepest_anchors(CADENCE, host_tier_present=False))

    def test_a_host_tier_relaxes_it(self):
        self.assertFalse(protect_deepest_anchors(CADENCE, host_tier_present=True))

    def test_no_interval_means_no_protection_either_way(self):
        """Matches `protect = self.mamba_checkpoint_interval is not None`
        (mamba_radix_cache.py:1101): without the grid there are no anchors to
        protect, host tier or not."""
        for host in (False, True):
            with self.subTest(host_tier_present=host):
                self.assertFalse(protect_deepest_anchors(None, host_tier_present=host))

    def test_the_relaxation_is_documented_where_it_happens(self):
        """A silent relaxation is the failure mode; the reason must sit at the
        decision, not only in a note."""
        src = inspect.getsource(protect_deepest_anchors)
        low = src.lower()
        self.assertIn("host", low)
        self.assertIn("load", low)
        self.assertTrue(
            "drift" in low or "resume point" in low,
            "must say WHAT the protection preserves",
        )


class TestRetentionRefusesOffGrid(CustomTestCase):
    """Both retention sites REFUSE an off-grid length rather than floor it --
    `mamba_radix_cache.py:654` and `:796`. Rounding the key down would pair a
    deeper state with a shorter key, which is silent corruption."""

    def test_on_grid_lengths_are_retainable(self):
        for k in range(1, 6):
            with self.subTest(pos=k * CADENCE):
                self.assertTrue(is_on_interval(k * CADENCE, CADENCE))

    def test_off_grid_lengths_are_not(self):
        for pos in (1, CADENCE - 1, CADENCE + 1, 3 * CADENCE + 7):
            with self.subTest(pos=pos):
                self.assertFalse(is_on_interval(pos, CADENCE))

    def test_flooring_is_NOT_the_retention_rule(self):
        """Guards against a future 'fix' that floors instead of refusing."""
        pos = CADENCE + 7
        self.assertEqual(floor_to_interval(pos, CADENCE), CADENCE)
        self.assertFalse(is_on_interval(pos, CADENCE))

    def test_the_interval_is_a_multiple_of_the_prefill_chunk(self):
        """#747 cadence contract: anchors can only exist at chunk boundaries,
        so the grid must be a multiple of the chunk size."""
        self.assertEqual(CADENCE % CHUNK, 0)
        self.assertEqual(CADENCE // CHUNK, 16)


class TestTheGridIsOffByDefault(CustomTestCase):
    def test_interval_none_accepts_every_position(self):
        for pos in (0, 1, 7, 512, 8191, 8192):
            with self.subTest(pos=pos):
                self.assertTrue(is_on_interval(pos, None))

    def test_interval_none_floors_to_identity(self):
        for pos in (0, 1, 7, 8191):
            with self.subTest(pos=pos):
                self.assertEqual(floor_to_interval(pos, None), pos)


class TestBothLineagesCallTheSameRule(CustomTestCase):
    """Seam parity by construction: neither lineage may re-implement the rule.
    This is the pin that would have caught the original drift."""

    def test_mamba_radix_cache_imports_the_shared_rules(self):
        from sglang.srt.mem_cache import mamba_radix_cache

        src = inspect.getsource(mamba_radix_cache)
        self.assertIn("from sglang.srt.mem_cache.mamba_ckpt_utils import", src)

    def test_neither_lineage_reimplements_the_modulo(self):
        """`pos % interval == 0` must appear only in the shared helper."""
        from sglang.srt.mem_cache import mamba_ckpt_utils, mamba_radix_cache
        from sglang.srt.mem_cache.unified_cache_components import mamba_component

        for mod in (mamba_radix_cache, mamba_component):
            with self.subTest(module=mod.__name__):
                src = inspect.getsource(mod)
                self.assertNotIn("% self.mamba_checkpoint_interval", src)
        self.assertIn("% interval", inspect.getsource(mamba_ckpt_utils))

    def test_the_device_lineage_calls_the_eviction_rule(self):
        """Pin the CALL inside the evicting function, not the import -- an
        import alone satisfied two pins earlier today."""
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

        src = inspect.getsource(MambaRadixCache.evict_mamba)
        self.assertIn("protect_deepest_anchors(", src)

    def test_the_unified_branching_seam_uses_the_checkpoint_grid(self):
        """mamba_radix_cache.py:1600 prefers the interval over the chunk size;
        the unified component must make the same choice, or a configured grid
        is silently ignored on that lineage -- the #747 defect itself."""
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        src = inspect.getsource(MambaComponent.finalize_match_result)
        self.assertIn("mamba_checkpoint_interval", src)
        self.assertIn("floor_to_interval(", src)


if __name__ == "__main__":
    unittest.main()
