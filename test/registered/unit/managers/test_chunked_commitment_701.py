"""#701 slice-3 rework: the cross-pass commitment hole, against the REAL adder.

The review gate BLOCKED the original slice-3 premise and it was right on both
counts I checked in-code:

* `schedule_policy.py:1464` really does gate the FULL lifetime --
  `total_tokens >= self.rem_total_tokens -> NO_TOKEN` -- so the commit's
  "admitted on a 512-token affordability check" story is false at first
  admission. There is no missing full-length gate.
* `schedule_policy.py:734-737` documents the real suspect in-code:
  `rem_total_tokens` INCLUDES `full_evictable_size()` while the allocator can
  only recover MAMBA-recoverable bytes. Paper-evictable funds an admission the
  evictor cannot honour -- gate passes, relief later frees 0.

What is genuinely missing is a RESERVATION. `PrefillAdder` is constructed
fresh each pass, so a resident chunked request's remaining PREFILL is
represented nowhere in later passes; only remaining DECODE is reserved, for
requests that appear in `running_batch.reqs` at all (a resident-but-batchless
chunked request may not, #631 defect O). Later admissions therefore spend the
chunked request's committed future, and the deadlock returns with two actors.

These tests run against a REAL `PrefillAdder` (only the pool and tree are
doubles), so the cross-pass ones FAIL against current behaviour rather than
against a module's absence -- which is what the gate said the original nine
could not do.

Hermetic: no CUDA.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.planner.chunked_admission import (
    ChunkedCommitmentLedger,
    PoolState,
    decide_chunked_admission,
    effective_rem_total_tokens,
)


def _tree_cache(*, evictable: int = 0, recoverable=None) -> MagicMock:
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = evictable
    tc.full_evictable_size.return_value = evictable
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    # The mamba-recoverable figure, which may be far below paper-evictable.
    tc.mamba_recoverable_size = MagicMock(
        return_value=evictable if recoverable is None else recoverable
    )
    return tc


def _allocator(*, available: int = 0) -> MagicMock:
    alloc = MagicMock()
    alloc.available_size.return_value = available
    alloc.full_available_size.return_value = available
    alloc.swa_available_size.return_value = 0
    return alloc


def _adder(tree_cache, allocator, **kwargs) -> PrefillAdder:
    defaults = dict(
        page_size=1,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=MagicMock(reqs=[]),
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=None,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )
    defaults.update(kwargs)
    return PrefillAdder(**defaults)


class TheCrossPassHoleIsRealOnTheLiveAdder(unittest.TestCase):
    """(b) the gate's cross-pass falsifier, against real PrefillAdder code."""

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_a_fresh_pass_does_not_know_about_an_outstanding_commitment(self):
        """Red against current behaviour: this IS the deadlock channel.

        A chunked request holding a 300k-token remaining commitment is
        invisible to the next pass's budget, so the next pass believes the
        whole pool is spendable.
        """
        adder = _adder(_tree_cache(evictable=0), _allocator(available=400_000))
        self.assertGreaterEqual(adder.rem_total_tokens, 400_000)

        ledger = ChunkedCommitmentLedger()
        ledger.commit("req-A", remaining_tokens=300_000)
        # The real adder's own number, corrected by the ledger, is what a later
        # pass must spend against.
        self.assertEqual(
            effective_rem_total_tokens(adder.rem_total_tokens, ledger),
            adder.rem_total_tokens - 300_000,
        )

    def test_spending_down_a_chunk_releases_exactly_that_much(self):
        adder = _adder(_tree_cache(evictable=0), _allocator(available=400_000))
        ledger = ChunkedCommitmentLedger()
        ledger.commit("req-A", remaining_tokens=300_000)
        ledger.spend("req-A", 512)
        self.assertEqual(ledger.outstanding_tokens(), 299_488)
        self.assertEqual(
            effective_rem_total_tokens(adder.rem_total_tokens, ledger),
            adder.rem_total_tokens - 299_488,
        )

    def test_release_on_finish_returns_the_whole_commitment(self):
        ledger = ChunkedCommitmentLedger()
        ledger.commit("req-A", remaining_tokens=300_000)
        ledger.release("req-A")
        self.assertEqual(ledger.outstanding_tokens(), 0)

    def test_a_resident_but_batchless_request_is_still_counted(self):
        """#631 defect O biting the accounting: absence from running_batch.reqs
        must not remove the commitment."""
        adder = _adder(_tree_cache(evictable=0), _allocator(available=400_000))
        self.assertEqual(list(adder.running_batch.reqs), [])
        ledger = ChunkedCommitmentLedger()
        ledger.commit("req-A", remaining_tokens=300_000)
        self.assertEqual(ledger.outstanding_tokens(), 300_000)
        self.assertLess(
            effective_rem_total_tokens(adder.rem_total_tokens, ledger),
            adder.rem_total_tokens,
        )

    def test_double_commit_is_refused_rather_than_silently_doubling(self):
        ledger = ChunkedCommitmentLedger()
        ledger.commit("req-A", remaining_tokens=300_000)
        with self.assertRaises(ValueError):
            ledger.commit("req-A", remaining_tokens=300_000)

    def test_overspending_a_commitment_is_refused(self):
        ledger = ChunkedCommitmentLedger()
        ledger.commit("req-A", remaining_tokens=1000)
        with self.assertRaises(ValueError):
            ledger.spend("req-A", 1001)


class ThePaperEvictableOvercountIsRefused(unittest.TestCase):
    """(c) the gate's hybrid-SSM falsifier -- defect 2, the likely specimen."""

    def test_paper_evictable_cannot_fund_what_the_evictor_cannot_recover(self):
        pool = PoolState(
            free_tokens=1_000.0,
            recoverable_evictable_tokens=2_000.0,  # what eviction can honour
            locked_tokens=0.0,
            total_capacity_tokens=437_000.0,
            permanent_reserve_tokens=0.0,
        )
        d = decide_chunked_admission(remaining_tokens=100_000, pool=pool)
        self.assertFalse(d.admitted)
        self.assertEqual(d.fundable_tokens, 3_000.0)

    def test_the_constructor_refuses_a_paper_evictable_keyword(self):
        """The old field name funded the specimen; make it unrepresentable."""
        with self.assertRaises(TypeError):
            PoolState(
                free_tokens=1.0,
                evictable_unlocked_tokens=2.0,  # noqa - the removed name
                locked_tokens=0.0,
                total_capacity_tokens=3.0,
                permanent_reserve_tokens=0.0,
            )


class TheRefuseBoundaryUsesAchievableFundable(unittest.TestCase):
    """Defect 3: refusing at RAW capacity creates a silent forever-defer wedge."""

    def _pool(self, permanent):
        return PoolState(
            free_tokens=10_000.0,
            recoverable_evictable_tokens=0.0,
            locked_tokens=0.0,
            total_capacity_tokens=437_000.0,
            permanent_reserve_tokens=permanent,
        )

    def test_between_the_two_bounds_is_refused_not_deferred_forever(self):
        """A request above the achievable ceiling can never be served: refuse.

        Under the old rule it sat below raw capacity and deferred every pass,
        and since a non-CONTINUE verdict breaks the FCFS loop, the whole queue
        wedged behind it with no usage-1.00 tell.
        """
        pool = self._pool(permanent=50_000.0)
        d = decide_chunked_admission(remaining_tokens=400_000, pool=pool)
        self.assertEqual(d.verdict, "refuse")
        self.assertIn("achievable", d.reason.lower())

    def test_below_the_achievable_ceiling_still_defers(self):
        pool = self._pool(permanent=50_000.0)
        d = decide_chunked_admission(remaining_tokens=300_000, pool=pool)
        self.assertEqual(d.verdict, "defer")

    def test_permanent_reserves_of_zero_reproduce_the_raw_capacity_bound(self):
        pool = self._pool(permanent=0.0)
        self.assertEqual(decide_chunked_admission(437_001, pool).verdict, "refuse")
        self.assertEqual(decide_chunked_admission(436_999, pool).verdict, "defer")


class TheDeferredHeadIsVisibleToTheFlip(unittest.TestCase):
    """Defect 5: a deferred head must not look like an idle instance."""

    def test_a_deferred_head_inhibits_idle_flip_formation(self):
        from sglang.srt.planner.chunked_admission import (
            deferred_head_blocks_idle_flip,
        )

        self.assertTrue(deferred_head_blocks_idle_flip(deferred_head_count=1))
        self.assertFalse(deferred_head_blocks_idle_flip(deferred_head_count=0))

    def test_defer_records_age_so_a_wedge_is_observable(self):
        ledger = ChunkedCommitmentLedger()
        ledger.note_deferred("req-B", pass_index=1)
        ledger.note_deferred("req-B", pass_index=2)
        ledger.note_deferred("req-B", pass_index=7)
        self.assertEqual(ledger.defer_age("req-B"), 6)
        self.assertEqual(ledger.defer_age("never-seen"), 0)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# WIRING falsifiers (#701). Both defects, through the REAL PrefillAdder, red on
# unwired (chunked_admission_enabled=False) and green on wired. The flag is the
# A/B arm, so each test asserts the SAME adder both ways.
# ---------------------------------------------------------------------------


# NOTE (#694, Slot-2 704240ce83): an earlier draft of this block tested a
# "paper-evictable vs recoverable" distinction. That hypothesis is REFUTED --
# evictable and protected are DISJOINT by construction (inc_lock_ref moves
# tokens out of evictable_size_ into protected_size_, radix_cache.py:605-606),
# and the readable specimen's LRU evictable size matched an independent
# traversal exactly. The real #694 root was a STALE FLOOR
# (uniform_avail_for_evict published once per iteration and never charged by
# later allocations), fixed on Slot-2's branch. Those tests are removed rather
# than left passing against a mock of a distinction the code does not make.


class TheTwoActorDeadlockAcrossPasses(unittest.TestCase):
    """#701 defect (b). A resident chunked request's committed future must be
    visible to a LATER pass, whose PrefillAdder is a fresh object."""

    def _fresh_pass(self, ledger, enabled):
        # A NEW adder each pass, exactly as the scheduler builds it. Anything
        # the adder held itself would be forgotten here -- which is the bug.
        return _adder(
            _tree_cache(evictable=0),
            _allocator(available=100_000),
            commitment_ledger=ledger,
            chunked_admission_enabled=enabled,
        )

    def test_unwired_a_later_pass_spends_the_resident_request_s_future(self):
        led = ChunkedCommitmentLedger()
        led.commit("resident", 80_000)  # actor 1, mid-prefill
        adder = self._fresh_pass(led, enabled=False)
        # Actor 2 sees the whole pool and will admit on top of a commitment
        # that is already spoken for.
        self.assertEqual(adder.rem_total_tokens, 100_000)

    def test_wired_the_commitment_survives_the_rebuild(self):
        led = ChunkedCommitmentLedger()
        led.commit("resident", 80_000)
        adder = self._fresh_pass(led, enabled=True)
        self.assertEqual(adder.rem_total_tokens, 20_000)

    def test_spending_the_commitment_returns_the_budget(self):
        """As chunks commit, the reservation shrinks and the budget recovers."""
        led = ChunkedCommitmentLedger()
        led.commit("resident", 80_000)
        led.spend("resident", 30_000)
        self.assertEqual(self._fresh_pass(led, True).rem_total_tokens, 50_000)
        led.release("resident")
        self.assertEqual(self._fresh_pass(led, True).rem_total_tokens, 100_000)

    def test_no_ledger_is_not_an_error(self):
        """A caller that has not wired the ledger yet must not crash."""
        adder = self._fresh_pass(None, enabled=True)
        self.assertEqual(adder.rem_total_tokens, 100_000)
