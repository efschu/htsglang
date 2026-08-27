"""#943b: a prefix crossing a cutover survives as a RE-FETCH, never as a revival.

THE COVERAGE THIS FILLS, named when the bisection closed. #943 asked why the
anchors died and the answer was #937 refusing to publish prefetch spans whose
host tier had been replaced -- correctly, since that refusal is the garbage fix.
The gap the bisection then exposed is that NO test pinned what happens to a
prefix across a cutover at all: the store path is unit-tested without one, and
the only evidence either way came from boots.

So this pins the property the re-issue world is supposed to have, and it is
deliberately a property about TWO things at once:

  * the prefix is RECOVERABLE -- the request is recorded as owed a fresh fetch,
    and an agreeing round hands that request back to the caller, so the prefix
    can be served without a full recompute; and
  * the old span is UNRECOVERABLE -- nothing retained about it can be used to
    republish it, and the operation that carried it cannot be re-stamped into
    looking current.

Pinning only the first would be satisfied by the very defect #937 removed.
"""

import unittest

import torch

from sglang.srt.managers.cache_controller import StorageOperation
from sglang.srt.mem_cache.hicache_phase_binding import (
    binding_state,
    current_generation,
    write_back_stamp_is_current,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# ~1s: plain objects and one single-rank vote. No pool, no accelerator, no boot.
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _cache_with(pending):
    """A stand-in carrying the REAL gate. Single rank, so the reduce is a no-op
    and the AGREEING path is what runs."""
    import types

    cache = types.SimpleNamespace(
        _reissue_pending=dict(pending),
        _reissue_taken=0,
        _reissue_disagreements=0,
        attn_cp_group=None,
        attn_tp_group=None,
        tp_world_size=1,
        _req_id_digest=UnifiedRadixCache._req_id_digest,
    )
    cache._all_reduce_attn_groups = types.MethodType(
        UnifiedRadixCache._all_reduce_attn_groups, cache
    )
    return cache


class TestACutoverIsWhatMakesAStampStale(CustomTestCase):
    """The cutover, in the only terms this layer knows it: the binding
    generation advances, and a stamp taken before it stops being current."""

    def test_a_stamp_taken_before_the_cutover_is_stale_after_it(self):
        op = StorageOperation(torch.zeros(1, dtype=torch.int64), [1, 2, 3], "h")
        self.assertTrue(write_back_stamp_is_current(op.binding_generation))

        binding_state().advance("tp")  # the cutover
        try:
            self.assertFalse(
                write_back_stamp_is_current(op.binding_generation),
                "a prefetch opened before the cutover must not read as current "
                "after it -- that is the #937 publish gate",
            )
        finally:
            binding_state().reset()

    def test_the_stale_operation_cannot_be_made_current_again(self):
        """THE UNRECOVERABLE HALF. If this ever becomes possible, the prefix
        'survives' by republishing bytes from a replaced pool, which is the 2j
        garbage wearing the fix's clothes."""
        from sglang.srt.managers import cache_controller

        err = getattr(cache_controller, "StaleStampRewrite", None)
        self.assertIsNotNone(
            err,
            "the binding stamp is not write-once, so a stale span is one "
            "assignment away from looking publishable (#943)",
        )
        op = StorageOperation(torch.zeros(1, dtype=torch.int64), [1, 2, 3], "h")
        binding_state().advance("tp")
        try:
            with self.assertRaises(err):
                op.binding_generation = current_generation()
        finally:
            binding_state().reset()


class TestThePrefixIsRecoverableAsAFreshFetch(CustomTestCase):
    """THE RECOVERABLE HALF."""

    def test_an_agreeing_round_hands_the_request_back_to_be_refetched(self):
        cache = _cache_with({"req-A": 1})
        got = UnifiedRadixCache.take_agreed_reissue(cache, ["req-A"])
        self.assertEqual(got, "req-A")
        self.assertEqual(cache._reissue_taken, 1)

    def test_the_entry_is_consumed_so_the_refetch_does_not_loop(self):
        cache = _cache_with({"req-A": 1})
        UnifiedRadixCache.take_agreed_reissue(cache, ["req-A"])
        self.assertNotIn("req-A", cache._reissue_pending)
        self.assertIsNone(UnifiedRadixCache.take_agreed_reissue(cache, ["req-A"]))

    def test_a_request_not_in_the_waiting_queue_is_never_nominated(self):
        """The candidate set is the INTERSECTION of owed and present. An
        agreement naming a request some rank cannot act on would leave that rank
        sitting out the collective its peers entered."""
        cache = _cache_with({"req-gone": 1})
        self.assertIsNone(UnifiedRadixCache.take_agreed_reissue(cache, ["req-here"]))
        self.assertIn("req-gone", cache._reissue_pending)


class TestNothingRevivableIsRetained(CustomTestCase):
    """What is kept about a refused prefetch is a NAME and a COUNT. If it ever
    grows to hold the operation, the host span or the indices, the revival this
    whole design forbids becomes reachable from local state."""

    def test_the_pending_record_holds_only_a_count(self):
        cache = _cache_with({"req-A": 2})
        for key, value in cache._reissue_pending.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(
                value,
                int,
                "the pending entry holds something other than a count; a span "
                "or an operation kept here is a revival waiting to happen",
            )
            self.assertNotIsInstance(value, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
