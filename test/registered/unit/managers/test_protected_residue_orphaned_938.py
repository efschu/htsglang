"""#938 -- the cutover drop cannot see the rows it is about to orphan.

THE SPECIMEN (window 2j, /spinning/evidence-665-f1/SPECIMEN-2026-08-27T1125Z-
2j-UNOWNED-FORENSIK.txt): an unowned block growing ~283 rows per cutover, which
is one retracted request's FULL allocation (``kv_allocated_len ==
kv_committed_len == cache_protected_len``, no gap -- so it is not #935's
interval between two lengths). The endproof is an on_idle deficit of 7012
identical to the #919 UNOWNED peak of 7012 while the #935 guard stayed silent.

THE BLIND SPOT THIS FILE PINS. ``drop_prefix_tree_returning_rows`` evicts
``tree_evictable_full_rows`` and then re-reads THE SAME metric to check for
residue. That metric excludes locked rows by definition, so the check designed
to catch "the tree still holds rows ``reset()`` is about to orphan" is blind to
the one class of row guaranteed to still be held: ``evict`` walks leaves and
refuses locked nodes, and ``_reset_full`` then zeroes
``component_protected_size_`` and installs a fresh root without freeing a
single device row.

That is why the seam log looks healthy while the block grows. On the 2j boot
the drop reported ``tree_rows_returned`` of 4412-4418 per cutover -- plainly not
the W29 "returned zero" defect -- and lost ~283 rows anyway.

WHAT THIS CHANGE IS, AND IS NOT. An instrument. It measures the protected
residue and names it; it frees nothing and releases no lock. A node is still
locked at this point mainly because a write-through is IN FLIGHT against it, a
live reader copying those very device rows to the host; releasing and evicting
mid-flight would free the copy's source underneath it, a use-after-free in the
#913 IMA family and strictly worse than the leak. The real fix hangs the
release off the write-through ACK, and is a decision for after this line has
reported a number from metal.

WHAT THIS FILE DRIVES. The real ``drop_prefix_tree_returning_rows`` against a
real ``UnifiedRadixCache`` with a real locked node -- deliberately NOT a double.
This module's own W29 lesson is that "the suite's own double had the attribute
and not the method, exactly backwards from production, which is how ten green
tests survived the boot this killed", so a hand-shaped stub is exactly the
instrument that cannot be trusted here.
"""

import unittest

import torch

from sglang.srt.managers import phase_flip_runtime
from sglang.srt.managers.phase_flip_runtime import (
    drop_prefix_tree_returning_rows,
    tree_evictable_full_rows,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


def tree_protected_full_rows(tree):
    """Resolved at CALL time, not import time, on purpose.

    The point of this file is that the drop is SILENT about protected residue
    today. If the module-level import of the new reader were what failed, the
    red would be an ImportError -- which proves a symbol is missing, not that
    the guard is blind. Resolving late keeps collection working against the
    unfixed tree so the behavioural assertions below are what go red.
    """
    reader = getattr(phase_flip_runtime, "tree_protected_full_rows", None)
    if reader is None:
        raise AssertionError(
            "phase_flip_runtime has no tree_protected_full_rows: the drop "
            "cannot read the protected half of the BasePrefixCache contract"
        )
    return reader(tree)


# ~3s: one tiny CPU-only radix tree, no accelerator, no group, no boot.
register_cpu_ci(est_time=3, suite="base-a-test-cpu")

LOCKED = 6  # rows held by the locked node -- the retract-donated anchor
FREE_ROWS = 5  # rows on a second, unlocked branch the drop CAN evict
PAGE_SIZE = 1


def _build_cache() -> UnifiedRadixCache:
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    return UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=ReqToTokenPool(
                size=8, max_context_len=64, device="cpu", enable_memory_saver=False
            ),
            token_to_kv_pool_allocator=TokenToKVPoolAllocator(
                size=64,
                dtype=torch.float16,
                device="cpu",
                kvcache=MHATokenToKVPool(
                    size=64,
                    page_size=PAGE_SIZE,
                    dtype=torch.float16,
                    head_num=2,
                    head_dim=4,
                    layer_num=2,
                    device="cpu",
                    enable_memory_saver=False,
                ),
                need_sort=False,
            ),
            page_size=PAGE_SIZE,
            disable=False,
            tree_components=(ComponentType.FULL,),
        )
    )


def _tree_with_a_locked_branch():
    """A tree holding one LOCKED branch and one evictable branch.

    The locked branch stands in for the anchor the retraction just donated
    while its write-through is still in flight -- the thing `evict` refuses
    and `reset()` then orphans.
    """
    cache = _build_cache()

    # Branch A: locked. Distinct first token so it is a separate child of root.
    cache.insert(
        InsertParams(
            key=RadixKey(list(range(100, 100 + LOCKED))),
            value=torch.arange(LOCKED, dtype=torch.int64),
        )
    )
    match = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(list(range(100, 100 + LOCKED))))
    )
    cache.inc_lock_ref(match.last_device_node)

    # Branch B: left evictable, so the drop has real work to report.
    cache.insert(
        InsertParams(
            key=RadixKey(list(range(200, 200 + FREE_ROWS))),
            value=torch.arange(LOCKED, LOCKED + FREE_ROWS, dtype=torch.int64),
        )
    )
    return cache


class TestTheHarnessReallyHoldsProtectedRows(CustomTestCase):
    """Before asserting anything about the guard, prove the scenario is the one
    claimed: production-shaped readers, and a genuinely locked branch."""

    def test_the_cache_answers_both_halves_of_the_contract_by_method(self):
        cache = _build_cache()
        # The W29 trap, inverted on purpose: METHODS, as production has.
        self.assertTrue(callable(getattr(cache, "full_evictable_size", None)))
        self.assertTrue(callable(getattr(cache, "full_protected_size", None)))

    def test_the_locked_branch_is_protected_and_not_evictable(self):
        cache = _tree_with_a_locked_branch()
        self.assertEqual(tree_protected_full_rows(cache), LOCKED)
        self.assertEqual(tree_evictable_full_rows(cache), FREE_ROWS)


class TestTheDropNamesWhatItOrphans(CustomTestCase):
    """THE RED TEST. Today the drop evicts the free branch, reports a healthy
    row count, and says nothing at all about the locked rows it leaves for
    reset() to orphan."""

    def setUp(self):
        phase_flip_runtime._PROTECTED_RESIDUE_ORPHANED_ROWS = 0
        phase_flip_runtime._PROTECTED_RESIDUE_ORPHANED_DROPS = 0

    def test_the_protected_residue_is_reported(self):
        cache = _tree_with_a_locked_branch()

        with self.assertLogs(phase_flip_runtime.logger, level="ERROR") as caught:
            drop_prefix_tree_returning_rows(cache)

        line = "\n".join(caught.output)
        self.assertIn("#938 PROTECTED RESIDUE ORPHANED", line)
        self.assertIn(f"{LOCKED} row(s)", line)

    def test_the_orphaned_rows_are_counted_across_drops(self):
        for _ in range(2):
            drop_prefix_tree_returning_rows(_tree_with_a_locked_branch())

        self.assertEqual(
            phase_flip_runtime._PROTECTED_RESIDUE_ORPHANED_ROWS, 2 * LOCKED
        )
        self.assertEqual(phase_flip_runtime._PROTECTED_RESIDUE_ORPHANED_DROPS, 2)

    def test_a_healthy_returned_count_does_not_imply_a_clean_drop(self):
        """The 2j shape: 4412-4418 rows returned per cutover AND ~283 lost.
        A nonzero return is not evidence of a complete drop."""
        cache = _tree_with_a_locked_branch()
        with self.assertLogs(phase_flip_runtime.logger, level="ERROR"):
            returned = drop_prefix_tree_returning_rows(cache)
        self.assertEqual(returned, FREE_ROWS)  # looks healthy...
        self.assertGreater(LOCKED, 0)  # ...while these were orphaned


class TestTheInstrumentFreesNothing(CustomTestCase):
    """THE SAFETY PIN. Releasing a lock here and evicting would free the source
    of an in-flight write-through: a use-after-free in the #913 IMA family,
    strictly worse than the leak. This stage measures only."""

    def test_the_allocator_gets_back_only_the_evictable_branch(self):
        cache = _tree_with_a_locked_branch()
        allocator = cache.token_to_kv_pool_allocator
        before = int(allocator.available_size())

        with self.assertLogs(phase_flip_runtime.logger, level="ERROR"):
            drop_prefix_tree_returning_rows(cache)

        # Exactly the unlocked branch came back. The locked rows are still
        # held -- leaked, and deliberately so until the ACK route exists.
        self.assertEqual(int(allocator.available_size()) - before, FREE_ROWS)

    def test_the_locked_node_keeps_its_lock(self):
        cache = _tree_with_a_locked_branch()
        node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(100, 100 + LOCKED))))
        ).last_device_node
        locks_before = [int(cd.lock_ref) for cd in node.component_data]

        with self.assertLogs(phase_flip_runtime.logger, level="ERROR"):
            drop_prefix_tree_returning_rows(cache)

        self.assertEqual([int(cd.lock_ref) for cd in node.component_data], locks_before)


class TestTheNegativeReadingIsLoggedToo(CustomTestCase):
    """A zero has to be visible, or it cannot be compared against a growing
    unowned block on the next boot."""

    def test_a_clean_tree_reports_zero_protected_residue(self):
        cache = _build_cache()
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(200, 200 + FREE_ROWS))),
                value=torch.arange(FREE_ROWS, dtype=torch.int64),
            )
        )

        with self.assertLogs(phase_flip_runtime.logger, level="INFO") as caught:
            drop_prefix_tree_returning_rows(cache)

        self.assertIn(
            "#938 protected residue at drop: 0 row(s)", "\n".join(caught.output)
        )


class TestAnUnreadableCountIsNotZero(CustomTestCase):
    """The W29 abstention rule, applied to the protected half."""

    def test_a_tree_without_the_reader_abstains_rather_than_reporting_zero(self):
        class _NoProtectedReader:
            full_evictable_size = None

        self.assertIsNone(tree_protected_full_rows(_NoProtectedReader()))


if __name__ == "__main__":
    unittest.main()
