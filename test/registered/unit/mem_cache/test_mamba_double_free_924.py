# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#924: a Mamba state slot returned twice, and a ledger that could not say so.

THE SPECIMEN, 2026-08-27 boot 1 of the 2d acceptance, all three ranks
(/spinning/evidence-665-f1/boot_accept2d0827_0827_0253.log:32825, 03:01:20Z)::

    pool memory leak detected!
    [mamba] total=20, available=23, evictable=0, protected=0, session_held=0,
            uncached=0, withheld=0, double_owned=0,
            leaked_full_pages={...}, leaked_mamba_pages=None

+3 AVAILABLE ON A 20-SLOT POOL. The growth is monotone and slow -- the
scheduler's own ``mamba usage`` field, which is ``1 - available/total`` and so
prints a surplus as a NEGATIVE, went -0.05 (=+1) at 02:57:59Z on all three
ranks in the same second, then -0.10, then -0.15 at the fatal reading.

TWO THINGS WERE BROKEN, AND EACH HID THE OTHER.

1. ``MambaSlotAllocator.free`` (``allocator/mamba.py``) was an unconditional
   ``torch.cat((self.free_slots, free_index))``. No membership question, so a
   slot released twice is REPRESENTABLE, and since ``available_size()`` is
   ``len(free_slots)`` it inflates the count by exactly the duplicate count --
   the shape of the specimen. Its sibling with the same job,
   ``HostKVCache.free`` (``pool_host/base.py:359-380``), has carried a
   ``slot_used`` ledger and an assertion since #905.

2. ``_check_mamba_pool``'s own diagnosis builds
   ``set(mamba_allocator.free_slots.tolist())``, and the ``set()`` collapses
   precisely the duplicates that caused the surplus. That is why the SAME line
   that says "leak" also says ``leaked_mamba_pages=None``: nothing is missing,
   something is present twice, and the checker had no term for it. The full
   pool has had one since #912 (``double_owned``, subtracted, with
   ``double_owned_src=census|live``); the mamba pool had none.

WHAT IS DELIBERATELY *NOT* DONE. Deduplicating inside ``free`` would keep the
free list sound and leave the second releaser running -- the surplus would stop
being observable while the ownership defect went on. So the allocator SAYS it,
with the releasing caller's stack (twenty-odd call sites reach this free, and
the specimen's death came minutes later when nothing still knew which), and
raises. And in the checker the duplicate term is subtracted so the reader is
not sent hunting a leak that is not there, but the check stays FATAL on a
duplicate: a Mamba slot handed to two requests answers both, which is the
wrong-answer-without-a-crash direction.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.mem_cache.allocator import mamba as mamba_allocator_module
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.test.ci.ci_register import register_cpu_ci

#: Resolved rather than imported so that a tree WITHOUT the wache fails these
#: tests one by one -- a collection error would collapse the allocator half and
#: the checker half of this ticket into a single unreadable red.
MambaSlotDoubleFree = getattr(
    mamba_allocator_module, "MambaSlotDoubleFree", RuntimeError
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

#: The measured firing, identical on PP0/PP1/PP2.
SPECIMEN_TOTAL = 20
SPECIMEN_AVAILABLE = 23


def _slots(*ids) -> torch.Tensor:
    return torch.tensor(list(ids), dtype=torch.int64)


class TestTheAllocatorRefusesTheSecondRelease(unittest.TestCase):
    def test_a_normal_alloc_free_cycle_is_unchanged(self):
        alloc = MambaSlotAllocator(size=SPECIMEN_TOTAL, device="cpu")
        self.assertEqual(alloc.available_size(), SPECIMEN_TOTAL)
        taken = alloc.alloc(3)
        self.assertEqual(alloc.available_size(), SPECIMEN_TOTAL - 3)
        alloc.free(taken)
        self.assertEqual(alloc.available_size(), SPECIMEN_TOTAL)

    def test_slot_zero_is_reserved_and_never_handed_out(self):
        alloc = MambaSlotAllocator(size=SPECIMEN_TOTAL, device="cpu")
        taken = alloc.alloc(SPECIMEN_TOTAL)
        self.assertNotIn(0, taken.tolist())
        self.assertEqual(sorted(taken.tolist()), list(range(1, SPECIMEN_TOTAL + 1)))

    def test_the_second_release_is_refused_and_names_the_slot(self):
        # THE REGRESSION: this used to append the id a second time and return,
        # leaving available_size() one over the pool for the rest of the run.
        alloc = MambaSlotAllocator(size=SPECIMEN_TOTAL, device="cpu")
        taken = alloc.alloc(1)
        alloc.free(taken)
        with self.assertRaises(MambaSlotDoubleFree) as caught:
            alloc.free(taken)
        self.assertIn(str(int(taken[0])), str(caught.exception))
        self.assertEqual(alloc.available_size(), SPECIMEN_TOTAL)

    def test_the_specimen_shape_cannot_be_reached_any_more(self):
        # Three separate slots released twice each is exactly the +3 the metal
        # showed, arrived at one at a time.
        alloc = MambaSlotAllocator(size=SPECIMEN_TOTAL, device="cpu")
        taken = alloc.alloc(3)
        alloc.free(taken)
        for slot in taken.split(1):
            with self.assertRaises(MambaSlotDoubleFree):
                alloc.free(slot)
        self.assertLessEqual(alloc.available_size(), SPECIMEN_TOTAL)

    def test_the_first_offender_is_recorded_for_attribution(self):
        alloc = MambaSlotAllocator(size=SPECIMEN_TOTAL, device="cpu")
        taken = alloc.alloc(1)
        alloc.free(taken)
        with self.assertRaises(MambaSlotDoubleFree):
            alloc.free(taken)
        # The whole point: ONE boot attributes it, not three.
        self.assertIsNotNone(alloc._first_double_free_trace)
        self.assertIn(
            "test_the_first_offender_is_recorded_for_attribution",
            alloc._first_double_free_trace,
        )
        self.assertEqual(alloc._double_free_count, 1)

    def test_the_refusal_is_logged_before_it_raises(self):
        alloc = MambaSlotAllocator(size=SPECIMEN_TOTAL, device="cpu")
        taken = alloc.alloc(1)
        alloc.free(taken)
        with self.assertLogs(
            "sglang.srt.mem_cache.allocator.mamba", level="ERROR"
        ) as logs:
            with self.assertRaises(MambaSlotDoubleFree):
                alloc.free(taken)
        line = "\n".join(logs.output)
        self.assertIn("#924", line)
        self.assertIn("DOUBLE FREE", line)

    def test_an_alloc_group_returns_its_remainder_without_tripping_the_wache(self):
        alloc = MambaSlotAllocator(size=SPECIMEN_TOTAL, device="cpu")
        alloc.alloc_group_begin(4)
        alloc.alloc(1)
        alloc.alloc(1)
        alloc.alloc_group_end()  # two prefetched-but-unused slots come back
        self.assertEqual(alloc.available_size(), SPECIMEN_TOTAL - 2)

    def test_an_empty_release_is_still_a_no_op(self):
        alloc = MambaSlotAllocator(size=SPECIMEN_TOTAL, device="cpu")
        alloc.free(torch.empty((0,), dtype=torch.int64))
        self.assertEqual(alloc.available_size(), SPECIMEN_TOTAL)

    def test_a_sentinel_id_does_not_wrap_the_ledger(self):
        # -1 is the ping-pong track buffer's empty marker. It is filtered at
        # memory_pool.py:2213 but not at every site, and a bool tensor would
        # answer the membership question about the LAST slot instead.
        alloc = MambaSlotAllocator(size=SPECIMEN_TOTAL, device="cpu")
        last = alloc.slot_used.numel() - 1
        alloc.slot_used[last] = True
        alloc.free(_slots(-1))
        self.assertTrue(bool(alloc.slot_used[last]))

    def test_clear_resets_the_ledger_with_the_free_list(self):
        alloc = MambaSlotAllocator(size=SPECIMEN_TOTAL, device="cpu")
        alloc.alloc(5)
        alloc.clear()
        self.assertEqual(alloc.available_size(), SPECIMEN_TOTAL)
        self.assertFalse(bool(alloc.slot_used.any()))
        # And a release after a clear is a double free, because the clear
        # already returned everything.
        with self.assertRaises(MambaSlotDoubleFree):
            alloc.free(_slots(1))


class _Tree:
    def __init__(self, mamba_rows=(), full_rows=(), protected=0):
        self._mamba = list(mamba_rows)
        self._full = list(full_rows)
        self._protected = protected

    def mamba_protected_size(self):
        return self._protected

    def all_mamba_values_flatten(self):
        return torch.tensor(self._mamba, dtype=torch.int64)

    def all_values_flatten(self):
        return torch.tensor(self._full, dtype=torch.int64)


class _Observer:
    def session_held_mamba_slots(self):
        return 0


class TestTheMambaLedgerNamesTheDuplicate(unittest.TestCase):
    """The real ``_check_mamba_pool``, driven."""

    @staticmethod
    def _checker(free_list, *, tree, evictable=0, published=None):
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            PoolStats,
        )

        mamba_allocator = SimpleNamespace(
            free_slots=torch.tensor(free_list, dtype=torch.int64),
            size=SPECIMEN_TOTAL,
        )
        if published is not None:
            mamba_allocator.double_owned_slots = published
        req_to_token_pool = SimpleNamespace(
            mamba_allocator=mamba_allocator,
            mamba_pool=SimpleNamespace(size=SPECIMEN_TOTAL),
        )
        full_alloc = SimpleNamespace(
            free_pages=torch.arange(1, 11, dtype=torch.int64),
            release_pages=torch.empty((0,), dtype=torch.int64),
            size=10,
            page_size=1,
        )
        checker = SchedulerInvariantChecker(
            is_hybrid_swa=False,
            is_hybrid_ssm=True,
            disaggregation_mode=None,
            page_size=1,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=10,
            server_args=SimpleNamespace(dcp_size=1),
            tree_cache=tree,
            token_to_kv_pool_allocator=full_alloc,
            req_to_token_pool=req_to_token_pool,
            pool_stats_observer=_Observer(),
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
        )
        ps = PoolStats(
            full_num_used=0,
            full_token_usage=0.0,
            full_available_size=10,
            full_evictable_size=0,
            is_hybrid_ssm=True,
            mamba_available_size=len(free_list),
            mamba_evictable_size=evictable,
        )
        return checker, ps

    def test_a_healthy_pool_is_unchanged_and_says_census(self):
        checker, ps = self._checker(
            list(range(1, SPECIMEN_TOTAL + 1)), tree=_Tree(full_rows=range(1, 11))
        )
        leak, msg = checker._check_mamba_pool(ps)
        self.assertFalse(leak, msg)
        self.assertIn("double_owned=0", msg)
        self.assertIn("double_owned_src=census", msg)

    def test_the_specimen_is_named_as_a_double_free_and_stays_fatal(self):
        # available=23 on total=20: three slots present twice.
        free_list = list(range(1, SPECIMEN_TOTAL + 1)) + [4, 9, 17]
        checker, ps = self._checker(free_list, tree=_Tree(full_rows=range(1, 11)))
        self.assertEqual(len(free_list), SPECIMEN_AVAILABLE)
        leak, msg = checker._check_mamba_pool(ps)
        self.assertTrue(leak, msg)
        self.assertIn("double_owned=3", msg)
        self.assertIn("double_owned_src=live", msg)
        self.assertIn("free_list_duplicates=3", msg)
        self.assertIn("duplicate_slot_ids=[4, 9, 17]", msg)

    def test_the_pre_924_diagnosis_could_not_name_it(self):
        # The half that made the specimen unreadable: the leak branch's own
        # set() collapses the duplicates, so nothing is "leaked".
        free_list = list(range(1, SPECIMEN_TOTAL + 1)) + [4, 9, 17]
        checker, ps = self._checker(free_list, tree=_Tree(full_rows=range(1, 11)))
        _, msg = checker._check_mamba_pool(ps)
        self.assertIn("leaked_mamba_pages=None", msg)
        self.assertIn("duplicate_slot_ids=", msg)

    def test_a_free_and_cached_slot_is_the_912_population_and_only_balances(self):
        # A slot the tree also holds is NOT a double free: nothing will be
        # handed out twice. It is subtracted, named, and NOT made fatal.
        free_list = list(range(1, SPECIMEN_TOTAL + 1))
        checker, ps = self._checker(
            free_list,
            tree=_Tree(mamba_rows=[7], full_rows=range(1, 11)),
            evictable=1,
        )
        leak, msg = checker._check_mamba_pool(ps)
        self.assertFalse(leak, msg)
        self.assertIn("double_owned=1", msg)
        self.assertIn("free_list_duplicates=0", msg)
        self.assertIn("double_owned_src=live", msg)

    def test_a_published_census_keeps_precedence(self):
        free_list = list(range(1, SPECIMEN_TOTAL + 1))
        checker, ps = self._checker(
            free_list, tree=_Tree(full_rows=range(1, 11)), published=2
        )
        leak, msg = checker._check_mamba_pool(ps)
        self.assertIn("double_owned=2", msg)
        self.assertIn("double_owned_src=census", msg)

    def test_mutant_a_genuine_deficit_stays_fatal(self):
        # Slots owned by nobody: the opposite sign, and the new term is 0.
        free_list = list(range(1, SPECIMEN_TOTAL - 2))
        checker, ps = self._checker(free_list, tree=_Tree(full_rows=range(1, 11)))
        leak, msg = checker._check_mamba_pool(ps)
        self.assertTrue(leak, msg)
        self.assertIn("double_owned=0", msg)

    def test_mutant_the_term_must_not_absolve_a_wider_surplus(self):
        # One duplicate, but four slots too many: subtracting the duplicate
        # still leaves a residue and the check still raises.
        free_list = list(range(1, SPECIMEN_TOTAL + 1)) + [4, 21, 22, 23]
        checker, ps = self._checker(free_list, tree=_Tree(full_rows=range(1, 11)))
        leak, msg = checker._check_mamba_pool(ps)
        self.assertTrue(leak, msg)
        self.assertIn("free_list_duplicates=1", msg)


if __name__ == "__main__":
    unittest.main()
