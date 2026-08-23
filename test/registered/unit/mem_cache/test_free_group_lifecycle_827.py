"""#827 W10: the free-group window must apply its staged frees exactly once,
and staged rows must never survive into an idle ledger check.

SOURCE, and what it does and does not establish
===============================================

/spinning/evidence-665-f1/boot_window1_0823_1204.log, one boot of
integ/808-739-810 @ 7aedf2413881122537f940ae59dc388a28aef841. One second after
the last phase-flip cutover all three ranks raised out of the strict idle
invariant (``invariant_checker.py:127-129``, ``available + evictable +
protected + session_held + uncached + withheld == total``) and the group died
six seconds later::

    PP0  total=471314 available=115840 evictable=2944 protected=16384 withheld=344338
    PP1  total=471314 available= 99456 evictable=19328 protected=    0 withheld=344338
    PP2  total=471314 available= 99456 evictable=19328 protected=    0 withheld=344338

The group cap that boot was 126976 rows (``KV-BACKING cap agreement``, log line
2576) and ``cap + withheld == total`` exactly (126976 + 344338 == 471314), so
the withheld term was sound and every rank owed exactly 126976 circulating
slots. PP1 and PP2 accounted 118784 -- short exactly one 8192 granule, the same
number on two ranks that had nothing else wrong with them.

WHAT THIS FILE PINS is the mechanism by which rows can go missing from that
ledger while every other bucket looks right: a free-group window whose staged
frees are dropped, or applied twice. It does NOT claim to be the boot's proven
root cause, and an earlier draft of this file that did claim so was wrong on
its own evidence -- see the two corrections below, both of which came from
re-reading the log rather than from theory.

CORRECTION 1, the two ranks that matter. ``_pool_census``
(phase_flip_runtime.py:3870, 3935) prints ``free=`` as ``len(set(free_pages) |
set(release_pages))`` -- DISTINCT ids -- and ``available=`` from
``available_size()``, which is ``len(free_pages) + len(release_pages)``, a raw
length. Their difference is therefore the number of duplicate free-list
entries, and across all 45 census lines of that boot it is zero on PP1 and PP2
at every single checkpoint, while on PP0 it runs 0 -> 16384 -> 16202 -> 8396.
Duplicates are a PP0-only phenomenon; the uniform -8192 that killed all three
ranks is not explained by them.

CORRECTION 2, PP0's surplus is a different defect. At 12:09:14, while PP0's
duplicate count was still zero and before any cap had engaged, PP0 already
logged ``KV-OWNERSHIP VIOLATION (at-arm pp_to_tp) [exclusivity] 16384 rows:
16384 row id(s) are claimed by more than one owner [('free_list',
'radix_cache')] sample=[1, 2, 3, 4, 5, 6, 7, 8]``. Rows owned by the free list
and the radix tree at once are counted in ``available`` AND in
``evictable``/``protected``, which is a surplus, not a shortfall. That is the
#822 ownership authority's own finding and is filed separately; it is not the
free-group window and nothing here should be read as fixing it.

THE MECHANISM, in ``BaseTokenToKVPoolAllocator``. ``free_group_end`` freed
``self.free_group`` without ever clearing it, so a closed window stayed armed;
``free_group_begin`` cleared it unconditionally, so an abandoned window's
staged rows were dropped. Neither production window is wrapped in try/finally
(batch_result_processor.py:100-113 and :761-855, dllm/mixin/scheduler.py:86-159
-- verified: no try, no early return between the paired calls), so a raise
between the two is what abandons one.

Staged rows are the worst kind of missing, because they are missing from every
bucket at once: out of the radix tree (``release_kv_cache`` has run), in
neither free list (so ``available_size`` cannot see them), and held by no
request (so neither protected nor session-held). And ``is_fully_idle``
(scheduler.py:8947-8972) never consults ``free_group``, so an abandoned window
that then goes quiet reaches ``on_idle`` and raises before any next
``free_group_begin`` could have recovered it. That asymmetry is why the reclaim
is wired into the idle path and not only into ``free_group_begin``.
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

#: One granule, the shortfall each rank showed in the specimen.
GRANULE = 8192


class _Alloc(BaseTokenToKVPoolAllocator):
    """The base class's free-group machinery, verbatim, over a CPU free list.

    Deliberately NOT a mock of the window: ``free_group_begin``,
    ``free_group_end``, ``flush_free_group`` and
    ``reclaim_abandoned_free_group`` are inherited unmodified, which is the
    whole point -- the defect lives in them. Only ``alloc``/``free``/``clear``
    are supplied, and ``free`` concatenates without checking membership exactly
    as ``paged.py:300-302`` does, because that is what turns a double
    application into duplicate ids rather than a no-op.
    """

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
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.free_pages = torch.cat((free_index, self.free_pages))
            self._notify_free(free_index)
        else:
            self.free_group.append(free_index)

    # -- the two readings the pool census takes, and their difference --------

    def distinct_ids(self) -> int:
        """What the census prints as ``free=``: the free lists as a SET."""
        return len(set(self.free_pages.tolist()) | set(self.release_pages.tolist()))

    def duplicates(self) -> int:
        """What the census's ``available=`` minus ``free=`` measures."""
        return self.available_size() - self.distinct_ids()


class FreeGroupAppliesExactlyOnceTest(unittest.TestCase):
    """A closed window must not stay armed and pay twice."""

    def test_a_second_end_does_not_free_the_same_rows_again(self):
        alloc = _Alloc(4 * GRANULE)
        held = alloc.alloc(GRANULE)

        alloc.free_group_begin()
        alloc.free(held)
        alloc.free_group_end()

        self.assertEqual(alloc.available_size(), 4 * GRANULE)
        self.assertEqual(alloc.duplicates(), 0, "the first end must be exact")

        # The outer window closes after the inner one, or a retry path runs the
        # same close twice. On the pre-fix code this freed `held` a second time.
        alloc.free_group_end()

        self.assertEqual(
            alloc.duplicates(),
            0,
            "a second free_group_end double-freed the staged rows: the same "
            "ids are in the free list twice, so available_size over-counts "
            "and the pool can hand one row to two requests",
        )
        self.assertEqual(alloc.available_size(), 4 * GRANULE)

    def test_the_staged_list_is_empty_once_the_window_has_closed(self):
        """The state that made the double free possible, asserted directly.

        Named separately from the symptom above because a fix that dropped the
        duplicates some other way -- a unique() belt over the free list, say --
        would leave the window armed and this defect intact one layer down.
        """
        alloc = _Alloc(4 * GRANULE)
        alloc.free_group_begin()
        alloc.free(alloc.alloc(GRANULE))
        alloc.free_group_end()

        self.assertEqual(
            alloc.free_group,
            [],
            "free_group_end left the staged frees in place, so the window is "
            "still armed after it has closed",
        )

    def test_end_without_a_begin_is_a_no_op(self):
        alloc = _Alloc(4 * GRANULE)
        before = alloc.available_size()
        alloc.free_group_end()
        self.assertEqual(alloc.available_size(), before)
        self.assertEqual(alloc.duplicates(), 0)


class AnAbandonedWindowForfeitsNoRowsTest(unittest.TestCase):
    """Rows staged by a window nobody closed must come back, not vanish."""

    @staticmethod
    def _abandon(alloc):
        """Stage one granule inside a window and never close it.

        This is what a raise between ``free_group_begin`` and
        ``free_group_end`` leaves behind; the production windows carry no
        try/finally, so the unwind simply skips the close.
        """
        held = alloc.alloc(GRANULE)
        alloc.free_group_begin()
        alloc.free(held)
        return held

    def test_staged_rows_are_in_no_bucket_at_all(self):
        """The premise the ledger arithmetic rests on, pinned on its own.

        If this ever stops holding -- if staged rows become visible to
        ``available_size`` -- the idle check could no longer read them as a
        leak and the rest of this file would be testing a dead hazard.
        """
        alloc = _Alloc(4 * GRANULE)
        self._abandon(alloc)
        self.assertEqual(
            alloc.available_size(),
            3 * GRANULE,
            "staged rows must be invisible to available_size; that is why "
            "the idle ledger reports them as missing",
        )
        self.assertEqual(alloc.distinct_ids(), 3 * GRANULE)

    def test_the_idle_reclaim_recovers_an_abandoned_window(self):
        """The path that actually saves the boot.

        The group went quiet with a window open and died in ``on_idle``. The
        next ``free_group_begin`` never ran, so recovery THERE would not have
        helped; recovery on the idle path is what closes this.
        """
        alloc = _Alloc(4 * GRANULE)
        held = self._abandon(alloc)

        recovered = alloc.reclaim_abandoned_free_group()

        self.assertEqual(recovered, GRANULE, "the reclaim must report its work")
        self.assertEqual(
            alloc.available_size(),
            4 * GRANULE,
            "the abandoned window's rows were forfeited: they are in neither "
            "free list nor tree, which the idle ledger reports as a pool "
            "memory leak of exactly one granule",
        )
        self.assertEqual(alloc.duplicates(), 0, "recovery must not double-book")
        self.assertEqual(
            sorted(alloc.free_pages.tolist())[: len(held)],
            sorted(held.tolist()),
            "the recovered ids must be the staged ones",
        )
        self.assertTrue(alloc.is_not_in_free_group, "the window must be closed")

    def test_the_idle_reclaim_is_silent_and_free_on_the_ordinary_path(self):
        """The other direction: no closed or empty window is ever touched."""
        alloc = _Alloc(4 * GRANULE)
        self.assertEqual(alloc.reclaim_abandoned_free_group(), 0)

        alloc.free_group_begin()
        alloc.free(alloc.alloc(GRANULE))
        alloc.free_group_end()
        self.assertEqual(
            alloc.reclaim_abandoned_free_group(),
            0,
            "a window that closed properly has nothing to reclaim",
        )
        self.assertEqual(alloc.available_size(), 4 * GRANULE)
        self.assertEqual(alloc.duplicates(), 0)

        # An OPEN window with nothing staged yet is not abandoned either, and
        # must keep its flag -- reclaiming here would close a live window under
        # its owner.
        alloc.free_group_begin()
        self.assertEqual(alloc.reclaim_abandoned_free_group(), 0)
        self.assertFalse(alloc.is_not_in_free_group)

    def test_reopening_an_open_window_applies_its_rows_instead_of_dropping_them(
        self,
    ):
        """The second line of defence, for the case that does reach a reopen."""
        alloc = _Alloc(4 * GRANULE)
        held = self._abandon(alloc)

        alloc.free_group_begin()

        self.assertEqual(alloc.available_size(), 4 * GRANULE)
        self.assertEqual(alloc.duplicates(), 0)
        self.assertEqual(
            sorted(alloc.free_pages.tolist())[: len(held)], sorted(held.tolist())
        )
        self.assertFalse(alloc.is_not_in_free_group, "the NEW window must be left open")
        self.assertEqual(alloc.free_group, [], "and it must start empty")

    def test_a_clean_reopen_stages_nothing_and_recovers_nothing(self):
        alloc = _Alloc(4 * GRANULE)
        alloc.free_group_begin()
        alloc.free(alloc.alloc(GRANULE))
        alloc.free_group_end()
        alloc.free_group_begin()
        self.assertEqual(alloc.free_group, [])
        self.assertEqual(alloc.duplicates(), 0)
        self.assertEqual(alloc.available_size(), 4 * GRANULE)


class TheLedgerShortfallIsExactlyOneGranuleTest(unittest.TestCase):
    """The specimen's own arithmetic, restricted to what it establishes."""

    TOTAL = 471314
    CAP = 126976
    WITHHELD = 344338

    def test_the_withheld_term_was_sound_and_the_shortfall_was_circulating(self):
        self.assertEqual(
            self.CAP + self.WITHHELD,
            self.TOTAL,
            "withheld accounted for exactly the capped-off id space, so the "
            "missing rows were in the circulating range, not the withheld one",
        )
        pp1 = 99456 + 19328 + 0  # available + evictable + protected
        pp2 = 99456 + 19328 + 0
        self.assertEqual(pp1 - self.CAP, -GRANULE)
        self.assertEqual(pp2 - self.CAP, -GRANULE)

    def test_an_abandoned_window_reproduces_that_shortfall_exactly(self):
        """One granule staged and forfeited == the two clean ranks' reading."""
        alloc = _Alloc(self.CAP)
        held = alloc.alloc(GRANULE)
        alloc.free_group_begin()
        alloc.free(held)
        # Pre-fix, the next begin dropped these. The ledger reading at that
        # moment -- which is what on_idle evaluates -- is the CAP short one
        # granule, on a pool with nothing else wrong with it.
        self.assertEqual(alloc.available_size() - self.CAP, -GRANULE)
        self.assertEqual(alloc.duplicates(), 0, "and with no duplicates, as PP1/PP2")


class TheGuardsCanStillFailTest(unittest.TestCase):
    """Can-fail proofs: each assertion above is shown to fail on a mutant.

    Desk-written-never-executed: an instrument that cannot fail is not an
    instrument. One dying mutant per call edge of the fix.
    """

    def test_mutant_end_does_not_clear_the_staged_list(self):
        """Revert the `end` half: the duplicate assertion must go red."""

        class Mutant(_Alloc):
            def free_group_end(self):
                self.is_not_in_free_group = True
                if self.free_group:  # pre-fix: reads without clearing
                    self.free(torch.cat(self.free_group))

        alloc = Mutant(4 * GRANULE)
        alloc.free_group_begin()
        alloc.free(alloc.alloc(GRANULE))
        alloc.free_group_end()
        alloc.free_group_end()
        self.assertEqual(
            alloc.duplicates(),
            GRANULE,
            "the mutant must reproduce the double free, or the guard above "
            "proves nothing",
        )

    def test_mutant_begin_drops_an_abandoned_windows_rows(self):
        """Revert the `begin` half: the forfeit assertion must go red."""

        class Mutant(_Alloc):
            def free_group_begin(self):
                self.is_not_in_free_group = False
                self.free_group = []  # pre-fix: unconditional drop

        alloc = Mutant(4 * GRANULE)
        alloc.free_group_begin()
        alloc.free(alloc.alloc(GRANULE))
        alloc.free_group_begin()
        self.assertEqual(
            alloc.available_size(),
            3 * GRANULE,
            "the mutant must reproduce the forfeited granule, or the guard "
            "above proves nothing",
        )

    def test_mutant_idle_reclaim_does_nothing(self):
        """Remove the idle reclaim: the abandoned granule stays lost."""

        class Mutant(_Alloc):
            def reclaim_abandoned_free_group(self) -> int:
                return 0

        alloc = Mutant(4 * GRANULE)
        alloc.alloc(GRANULE)
        alloc.free_group_begin()
        alloc.free(torch.arange(1, GRANULE + 1, dtype=torch.int64))
        alloc.reclaim_abandoned_free_group()
        self.assertEqual(
            alloc.available_size(),
            3 * GRANULE,
            "the mutant must leave the granule staged, or the reclaim test "
            "above proves nothing",
        )

    def test_flush_still_leaves_the_window_open_for_its_owner(self):
        """flush_free_group keeps its own contract -- apply now, do NOT close.

        It is deliberately not routed through the shared helper: its own
        regression suite binds the PRODUCTION body onto a duck-typed stand-in
        (test_deferred_free_eviction_681.py:68), which would break if this
        method started calling another private attribute.
        """
        alloc = _Alloc(4 * GRANULE)
        held = alloc.alloc(GRANULE)
        alloc.free_group_begin()
        alloc.free(held)
        applied = alloc.flush_free_group()
        self.assertEqual(applied, GRANULE)
        self.assertFalse(alloc.is_not_in_free_group, "the window must stay open")
        self.assertEqual(alloc.available_size(), 4 * GRANULE)
        # And the owner's own end-call is then a safe no-op, not a double free.
        alloc.free_group_end()
        self.assertEqual(alloc.duplicates(), 0)
        self.assertEqual(alloc.available_size(), 4 * GRANULE)


if __name__ == "__main__":
    unittest.main()
