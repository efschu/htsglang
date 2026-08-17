"""#694: eviction is SKIPPED before the alloc_token_slots OOM, not ineffective.

Specimen (2026-08-16, two occurrences, scheduler death): 512 tokens refused
with ~167k evictable. F4-r4's 1f594e702e instrument separates "eviction ran and
found nothing" from "eviction was skipped because avail >= want" -- and this is
the second.

ROOT. ``evict_from_tree_cache`` gates the eviction on
``uniform_avail_for_evict(...) < num_tokens``. That reads
``tree_cache.uniform_avail_floor``, which the scheduler publishes ONCE per
iteration (``scheduler.py:4142-4144``) as the group MIN of ``available_size()``.
Allocations made later in the same iteration are not charged against it, so late
in an iteration the published number is stale-optimistic: it still reports the
availability the pool had at publish time. With ``floor >= num_tokens`` the
eviction is skipped entirely, the alloc then fails against the LIVE pool, and
the raise reports a tree full of evictable tokens that nothing ever asked for.

This is not a new class. The HOST sibling has exactly this bug fixed already:
``uniform_host_avail_for_backup`` subtracts a per-iteration ledger
(``uniform_host_admitted_since_floor``) and its comment states the reasoning --
"a stale floor over-admits ... charging admissions against the floor removes the
staleness without a second collective". The DEVICE sibling never got the ledger.

Sufficiency of the same argument here: this rank's live availability is at least
``avail_at_publish - admitted_since``, and ``avail_at_publish >= floor``, so a
request that clears ``floor - admitted`` fits the real pool too. Rank-uniform by
construction: ``num_tokens`` comes from the replicated batch, so every rank
charges the same amount at the same allocation, and the eviction predicate stays
identical across ranks (which is the #616g invariant this must not break).

Hermetic: mocks only, no CUDA.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.common import (
    evict_from_tree_cache,
    note_uniform_admitted,
    uniform_avail_for_evict,
)

SPECIMEN_EVICTABLE = 167_000
SPECIMEN_WANT = 512
SPECIMEN_LIVE_AVAIL = 392


def _tree(*, floor=None, admitted=0, evictable=SPECIMEN_EVICTABLE):
    tc = MagicMock()
    tc.is_chunk_cache.return_value = False
    tc.uniform_avail_floor = floor
    tc.uniform_admitted_since_floor = admitted
    tc.evictable_size.return_value = evictable
    tc.evict.return_value = MagicMock(num_tokens_evicted=evictable)
    alloc = MagicMock()
    alloc.available_size.return_value = SPECIMEN_LIVE_AVAIL
    # Not a SWATokenToKVPoolAllocator -> the standard branch.
    tc.token_to_kv_pool_allocator = alloc
    return tc, alloc


class TheStaleFloorSkipsEviction(unittest.TestCase):
    """THE FALSIFIER: the specimen shape must still attempt an eviction."""

    def test_the_specimen_shape_attempts_eviction(self):
        """512 wanted, 392 live, 167k evictable, and a floor gone stale.

        The floor was published when the pool held plenty; allocations since
        have drawn it down to 392. Without charging them the gate reads the
        stale number, skips, and the caller raises on a full tree.
        """
        tc, _ = _tree(floor=SPECIMEN_EVICTABLE, admitted=SPECIMEN_EVICTABLE - 100)
        evict_from_tree_cache(tc, SPECIMEN_WANT)
        self.assertTrue(
            tc.evict.called,
            "eviction was SKIPPED at the specimen shape -- this is #694",
        )

    def test_an_uncharged_stale_floor_does_skip(self):
        """CAN-FAIL PROOF: without the ledger the gate really does skip.

        Kept deliberately: it shows the guard above is load-bearing rather than
        passing for an unrelated reason.
        """
        tc, _ = _tree(floor=SPECIMEN_EVICTABLE, admitted=0)
        evict_from_tree_cache(tc, SPECIMEN_WANT)
        self.assertFalse(tc.evict.called)


class TheLedgerMirrorsTheHostSibling(unittest.TestCase):
    def test_the_charge_lowers_the_evict_predicate(self):
        tc, _ = _tree(floor=10_000, admitted=0)
        self.assertEqual(
            uniform_avail_for_evict(tc, tc.token_to_kv_pool_allocator), 10_000
        )
        note_uniform_admitted(tc, 9_600)
        self.assertEqual(
            uniform_avail_for_evict(tc, tc.token_to_kv_pool_allocator), 400
        )

    def test_the_charged_floor_never_goes_negative(self):
        tc, _ = _tree(floor=100, admitted=0)
        note_uniform_admitted(tc, 5_000)
        self.assertEqual(uniform_avail_for_evict(tc, tc.token_to_kv_pool_allocator), 0)

    def test_no_floor_is_byte_identical_to_the_live_local_value(self):
        """Single rank / agreeing pools must behave exactly as before."""
        tc, alloc = _tree(floor=None, admitted=0)
        self.assertEqual(uniform_avail_for_evict(tc, alloc), SPECIMEN_LIVE_AVAIL)
        note_uniform_admitted(tc, 4_096)
        self.assertEqual(uniform_avail_for_evict(tc, alloc), SPECIMEN_LIVE_AVAIL)

    def test_charging_without_a_floor_is_a_noop(self):
        """No floor means the attribute is never read; do not grow a ledger."""
        tc, _ = _tree(floor=None, admitted=0)
        note_uniform_admitted(tc, 1_234)
        self.assertEqual(getattr(tc, "uniform_admitted_since_floor", 0), 0)

    def test_the_predicate_stays_rank_uniform(self):
        """#616g: two ranks with different LIVE pools but the same floor and
        the same charges must decide identically."""
        a, _ = _tree(floor=50_000, admitted=49_800)
        b, b_alloc = _tree(floor=50_000, admitted=49_800)
        b_alloc.available_size.return_value = 999_999  # roomy rank
        self.assertEqual(
            uniform_avail_for_evict(a, a.token_to_kv_pool_allocator),
            uniform_avail_for_evict(b, b_alloc),
        )


if __name__ == "__main__":
    unittest.main()
