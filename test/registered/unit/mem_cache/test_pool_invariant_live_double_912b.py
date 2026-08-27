"""#912 REST: the on-idle ledger's one-row surplus, and what it is NOT.

Boot 2 of the 2c acceptance was driven to a deliberate three-minute idle and
died on all three ranks, verbatim
(/spinning/evidence-665-f1/boot_accept2c0827_0827_0049.log:21372, 00:57:49Z)::

    ValueError: pool memory leak detected! [full] total=432089,
      available=133120, evictable=1, protected=0, session_held=0, uncached=0,
      withheld=298969, double_owned=0

    133120 + 1 + 0 + 0 + 0 + 298969 - 0 = 432090   vs   total = 432089

A SURPLUS OF EXACTLY ONE ROW, with #912's correction term reading zero.

THE FRAMING THIS FILE REFUTES FIRST
===================================

The acceptance note's own hypothesis was that the 1 is structural: "``total``
is the reservation while available/withheld enumerate over reservation +
page_size", the same ``+1`` that observable B2 measured as
``store_bound_rows - reserved_backing_rows == page_size`` on 435 of 435 dial
lines. It was recorded there as "a strong hypothesis worth a red-first test",
and the red-first test refutes it:

``TokenToKVPoolAllocator.clear()`` builds ``torch.arange(1, self.size + 1)``
(``mem_cache/allocator/token.py:44-45``), so the allocator's id space is
``1 .. size`` -- exactly ``size`` ids -- while ``total`` for this branch is
``self.token_to_kv_pool_allocator.size`` (``invariant_checker.py:187``). Row 0
is never in a free list and is never handed out. Both sides of the ledger are
the same id space; there is no ``page_size`` padding on either. The pool
tensor's extra row is real and is what ``store_bound_rows`` describes, but it
is not an id the ledger ever counts. ``TestNoPageSizePaddingInTheLedger912b``
pins that, so the hypothesis cannot be re-run as a new finding.

WHAT THE 1 ACTUALLY IS
======================

The same boot's post-cutover census, one second earlier (:21367), partitions
the id space with nothing left over::

    size=432089 free=133120 cached=0 withheld=298969 unaccounted=0

``free + withheld == size`` EXACTLY. Every id already has exactly one owner
and the tree holds none. The ``evictable=1`` the checker reads a second later
is therefore a row the tree took on in the round BETWEEN the two readings --
the boot's own OUTTRACE names the arrival, ``HEALTH_C n=1 (new) off=1
tail=[49276]`` (:21340), and 49276 is below the 133120 cap, i.e. inside the
free range. One row, two owners.

THE ROOT IS THAT THE CORRECTION TERM IS A SNAPSHOT AND THE REST IS LIVE.
``double_owned`` arrives as ``allocator.double_owned_slots``, published only
by the phase-flip census's ownership audit and cleared to ``None`` at every
cutover's id-space retirement (``phase_flip_runtime.py:6255-6264``, whose own
comment accepts "a possible false leak flag in that window"). ``available``
and ``evictable`` are read at the instant of the check. So a double claim that
arises after the last census is structurally invisible to the only term that
could name it -- and the flip is exactly what makes such a window.

THE FIX. ``SchedulerInvariantChecker._live_double_claimed_rows`` enumerates
the intersection of the free reading and the tree's values at the instant of
the check, and ``_check_full_pool`` uses it when the published snapshot is
absent or empty. It is not a tolerance: it subtracts the size of a set whose
members are known by id, the residue still raises, and a DEFICIT -- the
opposite sign, #832/#856's shape -- is untouched. The raising line now also
says which reading paid, ``double_owned_src=census|live``.
"""

import unittest

import torch

from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_row_ownership import read_free_rows
from sglang.test.test_utils import CustomTestCase

#: The measured firing, identical on PP0/PP1/PP2.
SPECIMEN_TOTAL = 432089
SPECIMEN_AVAILABLE = 133120
SPECIMEN_EVICTABLE = 1
SPECIMEN_WITHHELD = 298969
#: The census one second earlier, same ranks.
CENSUS_FREE = 133120
CENSUS_CACHED = 0
#: The row the OUTTRACE names as arriving between the two readings.
HEALTH_CHECK_ROW = 49276


class _Alloc(BaseTokenToKVPoolAllocator):
    """The real free-list machinery over CPU tensors, as #912's own harness."""

    def __init__(self, size: int):
        self.size = size
        self.page_size = 1
        self.device = "cpu"
        self.dtype = torch.int64
        self._free_listeners = []
        self.clear()

    def clear(self):
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)
        self.release_pages = torch.empty((0,), dtype=torch.int64)
        self.is_not_in_free_group = True
        self.free_group = []
        self._notify_clear()

    def alloc(self, need_size: int):
        out = self.free_pages[:need_size].clone()
        self.free_pages = self.free_pages[need_size:]
        return out

    def free(self, free_index: torch.Tensor):
        self.free_pages = torch.cat((free_index, self.free_pages))


class _Tree:
    def __init__(self, rows):
        self._rows = list(rows)

    def all_values_flatten(self):
        return torch.tensor(self._rows, dtype=torch.int64)


class _UnreadableTree:
    pass


class TestNoPageSizePaddingInTheLedger912b(CustomTestCase):
    """Refutation, red-first, of the ``reservation + page_size`` framing."""

    def test_the_free_list_spans_exactly_size_ids_starting_at_one(self):
        alloc = _Alloc(1000)
        self.assertEqual(int(alloc.free_pages.numel()), 1000)
        self.assertEqual(int(alloc.free_pages.min()), 1)
        self.assertEqual(int(alloc.free_pages.max()), 1000)

    def test_row_zero_is_not_an_id_the_ledger_counts(self):
        alloc = _Alloc(1000)
        reading = read_free_rows(alloc)
        self.assertTrue(reading.is_enumerable)
        self.assertNotIn(0, set(reading.rows))
        self.assertEqual(
            reading.count,
            alloc.size,
            "the enumerated free space and `total` are the SAME id space, so "
            "no page_size term can be missing from one side of the ledger",
        )

    def test_the_surplus_is_one_and_page_size_cannot_account_for_it(self):
        """The specimen arithmetic, stated so the refutation is concrete."""
        accounted = SPECIMEN_AVAILABLE + SPECIMEN_EVICTABLE + SPECIMEN_WITHHELD
        self.assertEqual(accounted - SPECIMEN_TOTAL, 1)
        # And the census a second earlier leaves NO room for an unclaimed id:
        self.assertEqual(CENSUS_FREE + SPECIMEN_WITHHELD, SPECIMEN_TOTAL)
        self.assertEqual(CENSUS_CACHED, 0)


class TestLiveDoubleClaimDerivation912b(CustomTestCase):
    """The term itself: an enumerated intersection, or ``None``."""

    @staticmethod
    def _derive(alloc, tree):
        return SchedulerInvariantChecker._live_double_claimed_rows(
            read_free_rows(alloc), tree
        )

    def test_the_specimen_shape_yields_exactly_one_row(self):
        """A free list that still lists the row the tree took on."""
        alloc = _Alloc(1000)
        tree = _Tree([HEALTH_CHECK_ROW % 1000 or 1])
        self.assertEqual(self._derive(alloc, tree), 1)

    def test_a_tree_row_outside_the_free_list_is_not_double_claimed(self):
        alloc = _Alloc(1000)
        alloc.free_pages = alloc.free_pages[alloc.free_pages != 7]
        tree = _Tree([7])
        self.assertEqual(
            self._derive(alloc, tree),
            0,
            "the ordinary, correct state must derive nothing -- otherwise the "
            "term would subtract on every healthy check",
        )

    def test_the_release_buffer_counts_as_the_free_side(self):
        """``read_free_rows`` unions both lists; the derivation must follow it."""
        alloc = _Alloc(1000)
        alloc.free_pages = alloc.free_pages[alloc.free_pages != 7]
        alloc.release_pages = torch.tensor([7], dtype=torch.int64)
        tree = _Tree([7])
        self.assertEqual(self._derive(alloc, tree), 1)

    def test_unanswerable_is_none_and_never_zero(self):
        alloc = _Alloc(100)
        self.assertIsNone(self._derive(alloc, _UnreadableTree()))

        class _NotEnumerable:
            is_enumerable = False
            rows = ()

        self.assertIsNone(
            SchedulerInvariantChecker._live_double_claimed_rows(
                _NotEnumerable(), _Tree([1])
            )
        )
        self.assertIsNone(
            SchedulerInvariantChecker._live_double_claimed_rows(None, _Tree([1]))
        )


class TestTheLedgerClosesOnTheSpecimen912b(CustomTestCase):
    """The equation, at the measured tuple, with and without the term."""

    @staticmethod
    def _check(double_owned):
        return SchedulerInvariantChecker._check_pool_invariant(
            "full",
            SPECIMEN_AVAILABLE,
            SPECIMEN_EVICTABLE,
            0,
            0,
            SPECIMEN_TOTAL,
            0,
            SPECIMEN_WITHHELD,
            double_owned,
        )

    def test_red_with_the_snapshot_reading_the_boot_actually_had(self):
        leak, msg = self._check(0)
        self.assertTrue(leak, f"the field crash must reproduce: {msg}")

    def test_green_with_the_one_row_the_live_derivation_names(self):
        leak, msg = self._check(1)
        self.assertFalse(leak, f"the invariant must close: {msg}")

    def test_mutant_the_term_must_not_close_a_two_row_surplus(self):
        """One named row explains one row and no more."""
        leak, _msg = SchedulerInvariantChecker._check_pool_invariant(
            "full",
            SPECIMEN_AVAILABLE + 1,
            SPECIMEN_EVICTABLE,
            0,
            0,
            SPECIMEN_TOTAL,
            0,
            SPECIMEN_WITHHELD,
            1,
        )
        self.assertTrue(leak)

    def test_mutant_a_deficit_stays_fatal(self):
        """The opposite sign -- rows with NO owner -- must never be masked."""
        for derived in (0, 1, 22):
            with self.subTest(double_owned=derived):
                leak, _msg = SchedulerInvariantChecker._check_pool_invariant(
                    "full",
                    SPECIMEN_AVAILABLE - 5,
                    SPECIMEN_EVICTABLE,
                    0,
                    0,
                    SPECIMEN_TOTAL,
                    0,
                    SPECIMEN_WITHHELD,
                    derived,
                )
                self.assertTrue(
                    leak,
                    "a subtracted term can only make a surplus smaller; it "
                    "must never turn a deficit into a pass",
                )


class TestCheckFullPoolEndToEnd912b(CustomTestCase):
    """Execution-proven wiring: the real ``_check_full_pool``, driven.

    The specimen's shape is reproduced at a small scale -- an allocator whose
    free list still lists a row the tree also holds, with no cap and no
    published census reading, which is exactly the state the boot was in one
    round after its post-cutover census.
    """

    @staticmethod
    def _checker(alloc, tree, evictable):
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            PoolStats,
        )

        class _Observer:
            def session_held_tokens(self):
                return 0

        class _Args:
            dcp_size = 1

        checker = SchedulerInvariantChecker(
            is_hybrid_swa=False,
            is_hybrid_ssm=True,
            disaggregation_mode=None,
            page_size=1,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=alloc.size,
            server_args=_Args(),
            tree_cache=tree,
            token_to_kv_pool_allocator=alloc,
            req_to_token_pool=None,
            pool_stats_observer=_Observer(),
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
        )
        ps = PoolStats(
            full_num_used=0,
            full_token_usage=0.0,
            full_available_size=int(alloc.free_pages.numel()),
            full_evictable_size=evictable,
            is_hybrid_ssm=True,
        )
        return checker, ps

    @staticmethod
    def _tree(rows):
        class _T(_Tree):
            def supports_mamba(self):
                return True

            def full_protected_size(self):
                return 0

        return _T(rows)

    def test_the_specimen_shape_closes_and_names_the_live_reading(self):
        alloc = _Alloc(100)  # ids 1..100, all free
        tree = self._tree([7])  # 7 is still in the free list
        checker, ps = self._checker(alloc, tree, evictable=1)
        leak, msg = checker._check_full_pool(ps)
        self.assertFalse(leak, msg)
        self.assertIn("double_owned_src=live", msg)
        self.assertIn("double_owned=1", msg)

    def test_a_healthy_pool_is_unchanged_and_says_census(self):
        alloc = _Alloc(100)
        alloc.free_pages = alloc.free_pages[alloc.free_pages != 7]
        tree = self._tree([7])
        checker, ps = self._checker(alloc, tree, evictable=1)
        leak, msg = checker._check_full_pool(ps)
        self.assertFalse(leak, msg)
        self.assertIn("double_owned_src=census", msg)
        self.assertIn("double_owned=0", msg)

    def test_a_published_snapshot_keeps_precedence(self):
        alloc = _Alloc(100)
        alloc.free_pages = alloc.free_pages[alloc.free_pages != 7]
        alloc.double_owned_slots = 3
        tree = self._tree([7])
        checker, ps = self._checker(alloc, tree, evictable=1)
        leak, msg = checker._check_full_pool(ps)
        self.assertIn("double_owned_src=census", msg)
        self.assertIn("double_owned=3", msg)
        self.assertTrue(
            leak,
            "a census reading of 3 over-corrects a 1-row surplus into a "
            "deficit; the wider authority still wins and the check still "
            "raises rather than being quietly replaced by the narrow reading",
        )

    def test_mutant_a_genuine_deficit_survives_the_live_reading(self):
        """Rows with NO owner: the live intersection is empty, still fatal."""
        alloc = _Alloc(100)
        alloc.free_pages = alloc.free_pages[:90]  # 10 ids owned by nobody
        tree = self._tree([])
        checker, ps = self._checker(alloc, tree, evictable=0)
        leak, msg = checker._check_full_pool(ps)
        self.assertTrue(leak, msg)
        self.assertIn("double_owned=0", msg)


class TestPrecedenceAndWiring912b(CustomTestCase):
    def test_the_raising_line_says_which_reading_paid(self):
        leak, msg = SchedulerInvariantChecker._check_pool_invariant(
            "full", 10, 0, 0, 0, 10, 0, 0, 0
        )
        self.assertFalse(leak)
        self.assertNotIn(
            "double_owned_src",
            msg,
            "the primitive stays a pure equation; the source label is the "
            "caller's, so this test fails if the two ever get merged",
        )


if __name__ == "__main__":
    unittest.main()
