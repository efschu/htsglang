"""#824 -- a retention cap must never fall BELOW the already-protected prefix.

THE CRASH THIS CLOSES, measured on metal 2026-08-23 06:07:06 (boot
boot_816_core_0823_0601, rank PP0, which took the whole instance with it --
PP1 and PP2 died seconds later of gloo "Connection closed by peer")::

    File "python/sglang/srt/mem_cache/unified_radix_cache.py", line 1204,
      in cache_unfinished_req
    assert req.cache_protected_len <= len(new_indices) + self.page_size - 1
    AssertionError: req.cache_protected_len=16384, len(new_indices)=8192,
                    page_aligned_len=8192

Exactly 2x, and both numbers are on the documented 8192 checkpoint cadence.

HOW THE PAIR IS PRODUCED. ``cache_unfinished_req`` ends by publishing
``req.cache_protected_len = len(new_indices)`` (unified_radix_cache.py:1234,
mamba_radix_cache.py:1013), so the protected prefix is whatever the LAST call
retained. The NEXT call recomputes what to retain as a ``min`` across
components (unified_radix_cache.py:1148-1157), and the mamba component returns
``req.mamba_last_track_seqlen`` -- a checkpoint-grid position. That position is
NOT monotone:

  * ``mamba_track_seqlen_aligned = req.mamba_branching_seqlen``
    (schedule_batch.py:2660) can set it BELOW an earlier track, and
  * it is reset to ``None`` outright in several places
    (schedule_batch.py:1601, mamba_component.py:893).

So a request that already published protected=16384 can come back with a
tracked position of 8192. The ``min`` then caps the retention at 8192, the
re-match returns 8192 indices, and the assert -- which is correct, and is the
last thing standing between this and a silently truncated protected range --
kills the rank.

THE RULE, and it is deliberately the same shape as #816: a length that is
already COMMITTED cannot be retroactively reduced. #816 said the exposed id
space may never exceed the committed backing; #824 says a retention cap may
never fall below the committed protected prefix. Both are "never shrink below
what you already promised", one on ids, one on tokens.

WHY A PREDICATE AND NOT A CORRECTED LENGTH. Raising the retention back up to
``protected`` is NOT available: the donated mamba state sits at exactly the
tracked position, and mamba_component.py:684-687 spells out the consequence --
"Never floor: rounding the retained key down while donating a deeper state
would pair state and key at different positions (silent corruption)". Handing
back any number invites that pairing. So the helper only ANSWERS THE QUESTION,
and the caller routes a True into the decline machinery it already has.

That machinery is ``_decline_retention``, and reusing it matters, because the
right answer is NOT the same for both callers (#783): an UNFINISHED step
answers ``0`` ("cache nothing this step" -- ``cache_unfinished_req`` would
otherwise hand the tree a key it cannot own), while a FINISHED one answers
``None`` ("no constraint" -- the request is over and the full-attention KV is
retained under a mamba tombstone). A bare ``None`` from the shrink check would
have been wrong for the unfinished caller, which is the one that crashed.
Capacity is lost either way; correctness is not.

ONE RULE, TWO LINEAGES -- the #747 pattern. The same assert exists in BOTH
cache lineages (unified_radix_cache.py:1204 and mamba_radix_cache.py:991), so
the decision lives in ``mamba_ckpt_utils.py`` as pure integer arithmetic and
both call it, exactly as #747 did for the grid rules.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.mem_cache.mamba_ckpt_utils import retention_shrinks_protected
from sglang.test.test_utils import CustomTestCase

#: The pair measured on metal, 2026-08-23 06:07:06.
PROTECTED = 16384
TRACKED = 8192

#: The documented cadence (#747): 8192 tokens = 16 x chunked_prefill_size(512).
CADENCE = 8192


class TestTheMetalPair(CustomTestCase):
    """The exact numbers from the crash. Red before the fix existed."""

    def test_the_measured_regression_is_caught(self):
        """8192 tracked under 16384 protected is a truncation."""
        self.assertTrue(retention_shrinks_protected(TRACKED, PROTECTED))

    def test_a_cap_at_exactly_the_protected_length_is_not_a_shrink(self):
        """The boundary is <, not <=: equal keeps every committed token."""
        self.assertFalse(retention_shrinks_protected(PROTECTED, PROTECTED))

    def test_a_deeper_cap_is_not_a_shrink(self):
        """Growing is the normal case and must not be disturbed."""
        self.assertFalse(retention_shrinks_protected(PROTECTED + CADENCE, PROTECTED))

    def test_one_token_below_protected_is_a_shrink(self):
        """The boundary is exact; a near-miss still truncates."""
        self.assertTrue(retention_shrinks_protected(PROTECTED - 1, PROTECTED))


class TestTheDegenerateInputs(CustomTestCase):
    """``None`` and 0 are different answers here and must not be conflated."""

    def test_an_existing_decline_is_not_re_judged(self):
        """``None`` means the component already declined: nothing to shrink."""
        self.assertFalse(retention_shrinks_protected(None, PROTECTED))

    def test_nothing_protected_yet_can_never_shrink(self):
        """The first chunk has protected=0; every cap grows from there."""
        self.assertFalse(retention_shrinks_protected(TRACKED, 0))

    def test_a_zero_cap_under_nothing_protected_is_not_a_shrink(self):
        """0 < 0 is false, and 0 must not be confused with ``None``."""
        self.assertFalse(retention_shrinks_protected(0, 0))

    def test_a_zero_cap_under_a_protected_prefix_is_a_shrink(self):
        """The #783 collapse, and the worst case: it would drop everything."""
        self.assertTrue(retention_shrinks_protected(0, PROTECTED))

    def test_a_missing_protected_length_is_nothing_protected(self):
        """``cache_protected_len`` is Optional at several call sites."""
        self.assertFalse(retention_shrinks_protected(TRACKED, None))


class TestTheRuleIsAboutShrinkingOnly(CustomTestCase):
    """Off-grid-ness is a DIFFERENT decision and stays with its own helper."""

    def test_an_off_grid_cap_above_protected_is_not_this_helpers_business(self):
        """This helper judges shrinkage. ``is_on_interval`` judges the grid.

        Conflating them would silently re-implement the #747 grid rule in a
        second place, which is the exact drift #747 was written to end.
        """
        self.assertFalse(retention_shrinks_protected(PROTECTED + 1, PROTECTED))


if __name__ == "__main__":
    unittest.main()
