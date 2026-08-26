"""#912: "pool memory leak detected" at ``on_idle`` killed all three
schedulers on an accounting SURPLUS, not a leak. Root-cause fix, red-first.

Every number below was measured on this rig; none is illustrative.

SPECIMENS
=========

Five firings across two independent boots, all ``protected=0 session_held=0
uncached=0``, all ``total=468981``::

    /spinning/evidence-665-f1/boot_accept0826_0826_1754.log:1861
        available=124949 evictable=1 withheld=344053
    /spinning/evidence-665-f1/boot_accept0826_0826_1754.log:1886
        available=119790 evictable=1 withheld=349212
    /spinning/evidence-665-f1/boot_accept0826_0826_1754.log:1911
        available=112420 evictable=1 withheld=356582
    /spinning/evidence-665-f1/boot_accept0826r2_0826_1748.log:2360
        available=107064 evictable=1 withheld=361938
    /spinning/evidence-665-f1/boot_accept0826r2_0826_1748.log:2385
        available=114709 evictable=1 withheld=354293

``available + evictable + withheld`` exceeds ``total`` by exactly 22 on every
one of the five -- never a deficit. A deficit (rows with NO owner, e.g.
#832's -1 or #856's -152) is the opposite sign and stays fatal; this file's
fix cannot mask either because both are covered by tests below.

DECOMPOSITION OF THE 22, DIRECTLY CO-LOCATED WITH THE BOOTS ABOVE
==================================================================

``boot_accept0826_0826_1730.log:1826`` (PP0, same withheld=344053 as the
first specimen) prints the phase-flip census's own two readings side by
side::

    PHASE-FLIP POOL CENSUS post-cutover tp_to_pp: size=468981
        free=124928 cached=0 withheld=344053 available=124928

``free=`` there is ``free_reading.count``
(``phase_flip_runtime.py:5860``, sourced from ``read_free_rows()``,
``kv_row_ownership.py:743-843``): a UNION,
``frozenset(free_pages.tolist()) | frozenset(release_pages.tolist())``
(``kv_row_ownership.py:814``). ``available=`` there is
``getattr(alloc, "available_size", ...)()`` (``phase_flip_runtime.py:5867``):
``TokenToKVPoolAllocator.available_size()``
(``allocator/token.py:52-54``), a raw SUM, ``len(free_pages) +
len(release_pages)``. The two agree in that one census line (both read
124928) because ``KvRowCap._settle_free_list_overlap()``
(``kv_backing_relief.py:870-924``) had just cleared the overlap; the FIRST
specimen above, same withheld value, moments later, shows ``available=124949``
-- 21 MORE than the census's own ``free=124928`` -- because ``on_idle``'s
invariant check (``pool_stats_observer.py:245``,
``self.token_to_kv_pool_allocator.available_size()``) reads the raw sum
directly and never went through the settle step or the union reader. That
21-row gap is candidate (a) from the task brief: a computed-vs-enumerated
mismatch, pinned to the two exact formulas and their two exact call sites.

This same union-vs-sum divergence is not a new finding in this tree:
``test_free_group_lifecycle_827.py``'s "CORRECTION 1" already named it, on a
DIFFERENT boot, where the gap was a free-group double-free bug (since fixed
by #827) and VARIED across the boot (0, 16384, 16202, 8396). #912's five
specimens instead show the SAME 21 every time, so this is not that
already-fixed mechanism recurring -- it is the identical two-formula
divergence, from a source this file does not claim to trace further; the fix
below closes the CHECKER's misreading regardless of why the underlying
overlap exists, which is the only thing #912 asked this ticket to fix.

The remaining 1 row: ``evictable`` is constant at 1 across all five specimens
while ``available`` swings by tens of thousands, which is the signature of a
single row simultaneously owned by the free list and the radix tree -- the
same "claimed by more than one owner" EXCLUSIVITY shape
``test_free_group_lifecycle_827.py``'s "CORRECTION 2" names as real, already
detected by the #822 authority, and explicitly "filed separately" there (PP0,
a different boot, 16384 rows, ``[('free_list', 'radix_cache')]``). Wired here
as ``double_owned``, sourced from the #822 authority's own EXCLUSIVITY
finding at the last phase-flip census
(``phase_flip_runtime.py::_census_ownership_audit``), never re-derived.

THE FIX
=======

Two independent, additive terms in ``SchedulerInvariantChecker``:

1. ``_check_full_pool`` now reads ``available`` via ``read_free_rows()``
   instead of ``ps.full_available_size`` whenever the allocator can
   enumerate -- the SAME authority the phase-flip census and the #822 audit
   already read, per that function's own "ONE authority, used by both
   consumers" rationale. Composite/watermark allocators, which cannot
   enumerate, are untouched: ``ps.full_available_size`` still applies there,
   byte for byte as before.
2. ``_check_pool_invariant`` gained a ``double_owned`` term, subtracted,
   sourced from ``allocator.double_owned_slots`` -- the #822 authority's
   EXCLUSIVITY "claimed by more than one owner" row count.

Neither is a tolerance or an epsilon: both name a REAL, independently
detectable population of rows and subtract exactly that population. A
tried-and-reverted third approach -- deduping ``free_pages``/``release_pages``
in PLACE inside ``KvRowCap._apply()`` -- is deliberately not reused
(``kv_backing_relief.py``'s own comment: "a dedupe would have hidden the next
path that books twice"); this fix touches only the CHECKER's own reading, not
the allocator's storage, so the raw overlap stays visible to anyone tracing
its origin further.

WHY THE MUTANTS BELOW ARE NOT OPTIONAL
=======================================

A checker that cannot fail is not a checker. Every mechanism here has a test
that removes it and shows the matching specimen misreads again, in the
DANGER direction; and the double-owned filter has a test proving it cannot
also swallow the opposite-signed defect (#832/#856-shape) it must never
touch.

TWO DEFECTS FOUND IN REVIEW AND FIXED IN THE SAME COMMIT
=========================================================

A peer review of the first cut of this fix (the tree owner of
fix/913-seam-ownership) found two problems in the ``double_owned`` term and
both are fixed here, not deferred:

1. SUBSTRING MATCH ON PROSE AS CONTROL FLOW. The first cut selected the
   "claimed by more than one owner" violation with
   ``"more than one owner" in v.detail`` -- exactly the
   "line_gate-Substring-Defekt -> #908" shape, and a violation of
   ``Violation``'s own docstring ("``detail`` is for humans and logs; it is
   never load-bearing"). Fixed by giving ``Violation`` a real field,
   ``kind``, set once at each of the two EXCLUSIVITY construction sites in
   ``kv_row_ownership.py`` to ``EXCLUSIVITY_DOUBLED`` or
   ``EXCLUSIVITY_UNOWNED``, and switching the filter in
   ``phase_flip_runtime.py::_census_ownership_audit`` to
   ``v.kind == EXCLUSIVITY_DOUBLED``. ``test_mutant_census_filter_uses_kind_not_detail_substring``
   below guards the wiring; ``test_mixed_violations_only_doubled_counted``
   proves the discrimination holds when BOTH EXCLUSIVITY shapes fire in the
   SAME audit, not just in isolation.

2. A STALE READING SURVIVING A CUTOVER. ``double_owned_slots`` is published
   by the phase-flip census (a SEAM event) and read by ``on_idle`` at an
   unrelated time. Between two census events the reading is a fair snapshot
   of a structural condition that does not change moment to moment (that is
   why #912's five specimens, spread across two boots, all show the same
   22) -- but ``authority.retire()`` at a cutover drops every claim the
   reading was computed from, by epoch, in one step
   (``kv_row_ownership.py``'s own "ATOMIC IN ONE PROCESS ONLY" retire()
   docstring). A reading taken before that instant says nothing about the
   claims after it. Fixed by clearing ``double_owned_slots`` to ``None``
   (never ``0`` -- ``0`` would silently claim "no double-owned rows",
   ``None`` honestly claims "not measured yet") at the exact point
   ``_retire_row_id_space`` calls ``authority.retire()``, so a
   pre-cutover reading cannot outlive the id space it was measured under.
   ``test_retire_clears_the_stale_double_owned_reading`` guards the wiring.

   Whether staleness alone -- absent the cutover-clearing fix above -- could
   ever turn a real leak into a silent pass was checked directly rather than
   asserted: ``test_stale_double_owned_cannot_mask_a_genuine_deficit`` below
   shows it analytically cannot, for any non-negative reading, current or
   stale: ``double_owned`` is SUBTRACTED, so it can only push
   ``total_accounted`` further BELOW ``total`` (deficit's direction is
   already "less accounted than total"), never closer to it. Subtraction can
   mask a SURPLUS (this ticket's own shape) if the stale reading is fed
   against a *different, coincidentally-sized* surplus; it cannot mask a
   DEFICIT. The cutover-clearing fix closes the remaining gap (a stale
   reading surviving into a new id space) as a matter of hygiene, not
   because the deficit-masking scenario the review raised was reproducible
   -- it was checked and is not.
"""

import inspect
import unittest

import torch

from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_row_ownership import (
    EXCLUSIVITY_DOUBLED,
    EXCLUSIVITY_UNOWNED,
    Law,
    RowOwnershipAuthority,
    RowSpace,
    read_free_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


# (label, total, available_raw, evictable, withheld); protected/session/uncached
# all 0 in every specimen.
SPECIMENS_912 = (
    ("1754.log:1861", 468981, 124949, 1, 344053),
    ("1754.log:1886", 468981, 119790, 1, 349212),
    ("1754.log:1911", 468981, 112420, 1, 356582),
    ("r2_1748.log:2360", 468981, 107064, 1, 361938),
    ("r2_1748.log:2385", 468981, 114709, 1, 354293),
)

#: The measured overcount, identical on all five specimens.
OVERCOUNT = 22
#: Decomposition: 21 from the available_size() union-vs-sum divergence
#: (mechanism 1), 1 from the free-list/radix-cache double claim (mechanism 2).
MECH1_ROWS = 21
MECH2_ROWS = 1


def _check_pool_invariant(*args, **kwargs):
    return SchedulerInvariantChecker._check_pool_invariant(*args, **kwargs)


class TestFiveSpecimensClose912(CustomTestCase):
    """The equation, exercised at exactly the five measured tuples."""

    def test_red_without_the_new_term(self):
        """Every specimen reproduces the field crash when double_owned=0.

        This IS the pre-fix behaviour: before this ticket,
        ``_check_pool_invariant`` had no ``double_owned`` parameter at all, so
        every caller was implicitly at 0. Capturing that as an explicit,
        still-reachable case is the red-first proof without needing to check
        out the pre-fix file.
        """
        for label, total, available, evictable, withheld in SPECIMENS_912:
            with self.subTest(specimen=label):
                leak, msg = _check_pool_invariant(
                    "full", available, evictable, 0, 0, total, 0, withheld
                )
                self.assertTrue(leak, f"expected the field crash to reproduce: {msg}")
                total_accounted = available + evictable + withheld
                self.assertEqual(
                    total_accounted - total,
                    OVERCOUNT,
                    "the surplus must be exactly 22 -- that is the whole "
                    "premise of this ticket, not just this test",
                )

    def test_green_with_the_single_double_owned_term(self):
        """Feeding the measured 22 as one posten closes every specimen."""
        for label, total, available, evictable, withheld in SPECIMENS_912:
            with self.subTest(specimen=label):
                leak, msg = _check_pool_invariant(
                    "full",
                    available,
                    evictable,
                    0,
                    0,
                    total,
                    0,
                    withheld,
                    OVERCOUNT,
                )
                self.assertFalse(leak, f"expected the invariant to close: {msg}")

    def test_green_with_the_measured_decomposition(self):
        """The SAME closure, split exactly as the two mechanisms measure it.

        ``available`` reduced by the 21 the union reader would have reported
        (mechanism 1) and ``double_owned=1`` for the free/radix double claim
        (mechanism 2, mechanism 1 already removed from ``available`` so it is
        not counted twice).
        """
        for label, total, available, evictable, withheld in SPECIMENS_912:
            with self.subTest(specimen=label):
                deduped_available = available - MECH1_ROWS
                leak, msg = _check_pool_invariant(
                    "full",
                    deduped_available,
                    evictable,
                    0,
                    0,
                    total,
                    0,
                    withheld,
                    MECH2_ROWS,
                )
                self.assertFalse(leak, f"expected the invariant to close: {msg}")

    def test_mutant_each_mechanism_alone_is_insufficient(self):
        """Neither term alone closes the equation -- both are load-bearing.

        This is the can-fail proof in the danger direction for the SPLIT
        fix: a change that wired only the ``available`` correction, or only
        ``double_owned``, would leave a 1-row (or 21-row) leak reported as
        real, and this test catches either half going missing.
        """
        for label, total, available, evictable, withheld in SPECIMENS_912:
            with self.subTest(specimen=label, mechanism="available_only"):
                deduped_available = available - MECH1_ROWS
                leak, _ = _check_pool_invariant(
                    "full",
                    deduped_available,
                    evictable,
                    0,
                    0,
                    total,
                    0,
                    withheld,
                    0,  # double_owned not applied
                )
                self.assertTrue(
                    leak, "mechanism 1 alone must leave a 1-row residual leak"
                )
            with self.subTest(specimen=label, mechanism="double_owned_only"):
                leak, _ = _check_pool_invariant(
                    "full",
                    available,  # raw, mechanism 1 not applied
                    evictable,
                    0,
                    0,
                    total,
                    0,
                    withheld,
                    MECH2_ROWS,
                )
                self.assertTrue(
                    leak, "mechanism 2 alone must leave a 21-row residual leak"
                )

    def test_preexisting_callers_are_unaffected(self):
        """The five other ``_check_pool_invariant`` call sites pass at most 7
        positional args (verified by inspection of the four callers in
        ``invariant_checker.py``: ``_check_swa_pool``, ``_check_mamba_pool``,
        and both branches of ``_check_mamba_pool_with_int8``). None of them
        must be made to pass ``withheld``/``double_owned`` by this change.
        """
        leak, msg = _check_pool_invariant("swa", 100, 0, 0, 0, 100)
        self.assertFalse(leak, msg)
        leak, msg = _check_pool_invariant("mamba", 90, 5, 5, 0, 100)
        self.assertFalse(leak, msg)


class TestAvailableSizeUnionVsSum912(unittest.TestCase):
    """Mechanism 1, reproduced on a real (not mocked) allocator.

    ``_Alloc`` follows ``test_free_group_lifecycle_827.py``'s own harness: the
    base class's real ``free_pages``/``release_pages`` machinery over CPU
    tensors, not a mock, because the defect lives in what reads those two
    tensors, not in any behaviour a mock would have to reimplement.
    """

    class _Alloc(BaseTokenToKVPoolAllocator):
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

    def test_sum_overcounts_the_overlap_the_union_does_not(self):
        alloc = self._Alloc(100)
        # Manufacture the exact shape token.py:52-54 and kv_row_ownership.py:814
        # disagree about: MECH1_ROWS ids present in BOTH lists at once. This is
        # not something normal alloc()/free() traffic can produce through this
        # harness's own methods -- which is the point: the overlap is an
        # external booking-twice event, not a state either reader ever claims
        # to construct, only to (dis)agree about once it exists.
        overlap = alloc.free_pages[:MECH1_ROWS].clone()
        alloc.release_pages = torch.cat((alloc.release_pages, overlap))

        self.assertEqual(
            alloc.available_size(),
            100 + MECH1_ROWS,
            "the raw sum must double-count the manufactured overlap",
        )
        reading = read_free_rows(alloc)
        self.assertTrue(reading.is_enumerable)
        self.assertEqual(
            reading.count, 100, "the union must count the overlap once"
        )
        self.assertEqual(
            alloc.available_size() - reading.count,
            MECH1_ROWS,
            "the divergence must equal exactly the manufactured overlap -- "
            "the same shape as the measured 21",
        )

    def test_mutant_full_pool_check_must_actually_use_the_union_reader(self):
        """Regression guard on the wiring, not just the primitives.

        If ``_check_full_pool`` were reverted to read ``ps.full_available_size``
        directly (the pre-fix code), the primitives above would still behave
        correctly in isolation and this suite would go green on a checker that
        is, in production, back to the pre-fix behaviour. This asserts the
        actual call is present in the source of the actual method under test.
        """
        source = inspect.getsource(SchedulerInvariantChecker._check_full_pool)
        self.assertIn(
            "read_free_rows(",
            source,
            "_check_full_pool no longer routes its available reading through "
            "read_free_rows() -- mechanism 1 is unwired",
        )


class TestExclusivityDoubleOwned912(CustomTestCase):
    """Mechanism 2: the #822 authority's own EXCLUSIVITY finding, filtered
    exactly as ``phase_flip_runtime.py::_census_ownership_audit`` filters it
    -- STRUCTURALLY, on ``Violation.kind``, not by matching a substring of
    ``Violation.detail`` (that was the first cut's own defect, see the module
    docstring's "TWO DEFECTS FOUND IN REVIEW" section).
    """

    @staticmethod
    def _double_owned(violations):
        """The exact expression wired into phase_flip_runtime.py."""
        return sum(
            v.rows
            for v in violations
            if v.law == Law.EXCLUSIVITY and v.kind == EXCLUSIVITY_DOUBLED
        )

    def test_free_list_radix_cache_overlap_is_counted(self):
        space = RowSpace(exposed=99, committed=100)
        auth = RowOwnershipAuthority(space)
        auth.declare("free_list", range(1, 100))
        auth.declare("radix_cache", [1])  # row 1: owned by both

        violations = auth.audit()
        doubled = [
            v
            for v in violations
            if v.law == Law.EXCLUSIVITY and v.kind == EXCLUSIVITY_DOUBLED
        ]
        self.assertEqual(len(doubled), 1)
        self.assertEqual(doubled[0].rows, MECH2_ROWS)
        # `detail` is still human prose for the log line, but it is no longer
        # what selects this branch -- confirm the real content is still
        # there for the log without depending on it for the count above.
        self.assertIn("more than one owner", doubled[0].detail)
        self.assertEqual(self._double_owned(violations), MECH2_ROWS)

    def test_mutant_unowned_rows_must_not_be_counted_as_double_owned(self):
        """The OTHER EXCLUSIVITY shape -- rows with NO owner -- must read 0.

        This is the safety property the task brief demanded explicitly: a
        genuine deficit-type leak (#832/#856) must never be masked by this
        term. Both shapes share ``Law.EXCLUSIVITY``; ``Violation.kind`` tells
        them apart structurally now. A filter that reverted to matching
        ``v.detail`` text (or that dropped the kind check entirely) would
        make this test fail by wrongly counting the gap below as
        double-owned.
        """
        space = RowSpace(exposed=99, committed=100)
        auth = RowOwnershipAuthority(space)
        # Leave row 50 unclaimed: a coverage gap, not a double claim.
        auth.declare("free_list", [r for r in range(1, 100) if r != 50])

        violations = auth.audit()
        unowned = [
            v
            for v in violations
            if v.law == Law.EXCLUSIVITY and v.kind == EXCLUSIVITY_UNOWNED
        ]
        self.assertEqual(len(unowned), 1)
        self.assertEqual(unowned[0].rows, 1)
        self.assertIn("no enumerated owner", unowned[0].detail)
        self.assertEqual(
            self._double_owned(violations),
            0,
            "an unowned-row violation must not be read as a double claim -- "
            "doing so would let this fix mask a real, missing-row leak",
        )

    def test_mixed_violations_only_doubled_counted(self):
        """Both EXCLUSIVITY shapes fire in ONE audit -- not just in isolation.

        A discriminator that happens to work when only one shape is present
        is a weaker proof than one exercised with both present together,
        which is the realistic #814-plus-#912-at-once shape: some rows
        unclaimed, others double-claimed, in the same census.
        """
        space = RowSpace(exposed=101, committed=102)
        auth = RowOwnershipAuthority(space)
        # Row 1 double-claimed; row 101 left out entirely (unowned); the
        # rest single-claimed.
        claimed = [r for r in range(1, 102) if r != 101]
        auth.declare("free_list", claimed)
        auth.declare("radix_cache", [1])

        violations = auth.audit()
        by_kind = {v.kind: v for v in violations if v.law == Law.EXCLUSIVITY}
        self.assertEqual(set(by_kind), {EXCLUSIVITY_DOUBLED, EXCLUSIVITY_UNOWNED})
        self.assertEqual(by_kind[EXCLUSIVITY_DOUBLED].rows, 1)
        self.assertEqual(by_kind[EXCLUSIVITY_UNOWNED].rows, 1)
        self.assertEqual(
            self._double_owned(violations),
            1,
            "only the doubled row counts, even though an unowned row fired "
            "in the very same audit",
        )

    def test_mutant_census_filter_uses_kind_not_detail_substring(self):
        """Regression guard: the WIRED filter, not just the primitive above.

        Asserts the production filter in
        ``phase_flip_runtime.py::_census_ownership_audit`` selects on
        ``EXCLUSIVITY_DOUBLED`` and no longer contains the reverted
        ``"more than one owner" in v.detail`` substring check -- the exact
        "line_gate-Substring-Defekt -> #908" shape the review named.
        """
        source = inspect.getsource(PhaseFlipRuntime._census_ownership_audit)
        self.assertIn(
            "EXCLUSIVITY_DOUBLED",
            source,
            "_census_ownership_audit no longer selects the doubled-claim "
            "violation structurally",
        )
        self.assertNotIn(
            '"more than one owner" in v.detail',
            source,
            "_census_ownership_audit reverted to parsing Violation.detail "
            "as control flow -- the #908 substring-defect shape",
        )

    def test_retire_clears_the_stale_double_owned_reading(self):
        """Regression guard: a pre-cutover reading must not survive it.

        Asserts ``_retire_row_id_space`` -- the method that calls
        ``authority.retire()`` -- also clears ``double_owned_slots`` to
        ``None`` in the same step, so a snapshot taken under the OLD id
        space is never read as still current under the new one.
        """
        source = inspect.getsource(PhaseFlipRuntime._retire_row_id_space)
        self.assertIn(
            "double_owned_slots = None",
            source,
            "_retire_row_id_space no longer clears the stale "
            "double-owned reading at cutover",
        )


class TestStaleDoubleOwnedCannotMaskADeficit912(CustomTestCase):
    """Answers, directly rather than by assertion, whether a stale (or just
    plain wrong) ``double_owned`` reading can turn a genuine deficit-type
    leak (#832/#856-shape: rows missing, not doubled) into a false pass.
    """

    def test_stale_double_owned_cannot_mask_a_genuine_deficit(self):
        """Feed a large, wholly unrelated stale double_owned against a
        manufactured 100-row deficit on a real #912 specimen. It must still
        read as a leak, for every stale value tried, including one as large
        as test_free_group_lifecycle_827's own 16384-row specimen.
        """
        label, total, available, evictable, withheld = SPECIMENS_912[0]
        deficit_available = available - 100  # a genuine, separate 100-row hole
        for stale_double_owned in (0, 1, MECH2_ROWS, 22, 16384, 10**6):
            with self.subTest(stale_double_owned=stale_double_owned):
                leak, msg = _check_pool_invariant(
                    "full",
                    deficit_available,
                    evictable,
                    0,
                    0,
                    total,
                    0,
                    withheld,
                    stale_double_owned,
                )
                self.assertTrue(
                    leak,
                    f"a {stale_double_owned}-row double_owned reading must "
                    f"never mask a genuine 100-row deficit: {msg}",
                )

    def test_a_correctly_sized_double_owned_can_still_mask_an_UNRELATED_surplus(
        self,
    ):
        """The honest boundary of the safety property above: subtraction
        cannot mask a DEFICIT, but a stale reading CAN cancel a different,
        coincidentally-equal-sized SURPLUS it was never measured against.
        This is not a defect in the arithmetic -- the term IS a real reader
        of a real cause of surplus; nothing can distinguish "the same cause,
        measured late" from "a different cause of the same size" by the
        integers alone. That is exactly why the cutover-clearing fix exists:
        to bound how long a stale reading can stay in circulation, since the
        integers alone cannot prove staleness.
        """
        label, total, available, evictable, withheld = SPECIMENS_912[0]
        # An unrelated surplus of exactly MECH2_ROWS, wholly independent of
        # the free-list/radix-cache condition the stale reading came from.
        unrelated_surplus_available = available + MECH2_ROWS
        leak, _ = _check_pool_invariant(
            "full",
            unrelated_surplus_available,
            evictable,
            0,
            0,
            total,
            0,
            withheld,
            OVERCOUNT + MECH2_ROWS,  # the stale reading, sized for the OLD surplus
        )
        # This documents the boundary rather than asserting a false safety
        # claim: it is expected to pass (masked) here, which is exactly why
        # staleness is bounded by the cutover-clearing fix instead of relied
        # on to self-resolve.
        self.assertFalse(
            leak,
            "documents the known boundary: a coincidentally-equal stale "
            "surplus reading is not distinguishable from a fresh one by "
            "the integers alone -- this is what the cutover-clearing fix "
            "bounds, not what this arithmetic can rule out",
        )


if __name__ == "__main__":
    unittest.main()
