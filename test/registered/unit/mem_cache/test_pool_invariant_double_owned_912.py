"""#912: "pool memory leak detected" at ``on_idle`` killed all three
schedulers on an accounting SURPLUS, not a leak. Root-cause fix, red-first.

Every number below was measured on this rig; none is illustrative.

READ THIS FIRST: MECHANISM 2's LEDGER TERM WAS DELETED BY #969 CUT C
====================================================================

#912 shipped TWO mechanisms (see "THE FIX" below). Mechanism 1 -- reading
``available`` through ``read_free_rows()``'s UNION instead of
``available_size()``'s raw SUM -- IS ALIVE and is tested here. Mechanism 2 --
a ``double_owned`` parameter on ``_check_pool_invariant``, subtracted from the
ledger equation -- IS GONE, deliberately, and this file no longer tests it:

* ``77b42d6d0a`` "[#969 CUT C] Delete the double-claim census term: it was
  GIL-spinning the rank that had to post the proxy" -- deleted
  ``_live_double_claimed_rows``, the ``double_owned`` parameter, its
  subtraction, its message field, the census/live resolution block and the
  ``double_owned_src`` suffix on the full pool, as ONE change (producer, term
  and consumer are coupled; deleting the producer alone silently changes the
  equation instead of raising).
* ``7f88b49a08`` "[#969 CUT C fixup]" -- restored the truncated ledger call and
  swept the mamba twins ``_check_mamba_pool`` / ``_check_mamba_pool_with_int8``.

THE EVIDENCE FOR THE DELETION was measured, not argued: py-spy on the three
ranks of ``boot_969cut_55fdfa5e7a_0829_133212.log``, taken while the group was
wedged, put PP0 GIL-bound in this very term's generator expression
(``invariant_checker.py:207``, ``_live_double_claimed_rows`` <-
``_check_full_pool`` <- ``_check_all_pools`` <- ``on_idle``) while PP1 and PP2
waited in ``_pp_recv_proxy_tensors`` for a proxy PP0 never returned to post.
Second independent measurement of that class (first:
SPECIMEN-2026-08-27T0611Z-CENSUS-AUDIT-CPU-WEDGE.txt). Upstream's
``_check_pool_invariant`` is count-based and carries no such term, so under the
upstream-minimal law the fork-own compensation layer was a deletion candidate,
not a repair order.

WHAT STILL COVERS THE #912 CRASH CLASS, and is tested below:

1. mechanism 1, the union reader (``TestAvailableSizeUnionVsSum912``) -- 21 of
   the measured 22;
2. #927's per-access allocator resolution
   (``test_checker_reads_bound_pool_927.py``) -- the flip was manufacturing the
   remaining overlap by auditing the BOOT phase's pool, so it is a false
   positive removed at its root rather than subtracted;
3. the #822 authority's own EXCLUSIVITY discrimination
   (``TestExclusivityDoubleOwned912``), which is ALIVE: the census still
   reports doubly-claimed rows separately in
   ``phase_flip_runtime.py::_census_ownership_audit``. What #969 CUT C removed
   is only the SUBTRACTION of that population from the ledger equation, never
   its detection;
4. on the mamba side, ``_mamba_double_claimed`` still sets ``leak = True`` on a
   genuine free-list duplicate (the #924 double-free family).

ALSO FOLDED IN HERE: the ``reservation + page_size`` refutation that used to
live in ``test_pool_invariant_live_double_912b.py``
(``TestNoPageSizePaddingInTheLedger912``, bottom of this file). That file was
deleted with the same cut -- every one of its other classes drove
``_live_double_claimed_rows`` or the deleted parameter -- but its refutation is
a property of the ALLOCATOR's id space, still true, and worth keeping so the
hypothesis is not re-run as a new finding.

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
   EXCLUSIVITY "claimed by more than one owner" row count. **DELETED BY #969
   CUT C -- see the section at the top of this docstring. The paragraphs below
   describe mechanism 2 as it stood while it existed; they are kept because
   they are the record of WHY it was built and of the two review defects it
   went through, and because the census half they describe is still live.**

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
   asserted, in a class this file used to carry: it analytically cannot, for
   any non-negative reading, current or stale, because ``double_owned`` was
   SUBTRACTED and could only push ``total_accounted`` further BELOW ``total``
   (a deficit's direction is already "less accounted than total"), never
   closer to it. Subtraction could mask a SURPLUS if a stale reading were fed
   against a *different, coincidentally-sized* one; it could not mask a
   DEFICIT. Those two tests went with the term in #969 CUT C -- with no
   subtracted posten left in the equation there is nothing to mask with, which
   is a stronger guarantee than the tests were making. What remains is
   ``test_the_signature_carries_no_double_owned_term_969`` below, which pins
   the deletion itself so the term cannot be re-added without a decision.
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
    """The equation, exercised at exactly the five measured tuples.

    Since #969 CUT C this class no longer shows the term CLOSING the five --
    there is no term. It pins the specimen premise (the raw sum overcounts by
    exactly 22, which is what mechanism 1 and #927 between them account for)
    and the shape of the surviving signature.
    """

    def test_the_raw_sum_reproduces_the_field_crash(self):
        """Every specimen reads as a leak when ``available`` is the RAW sum.

        This is the premise of the whole ticket, and it is also the behaviour
        of the shipped equation today: nothing is subtracted, so a specimen fed
        the raw ``available_size()`` reading raises. What stops it in
        production is mechanism 1 -- ``_check_full_pool`` does not pass the raw
        sum, it passes ``read_free_rows().count`` -- and #927, not a correction
        term inside this function.
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

    def test_mechanism_one_alone_leaves_the_one_row_927_removes(self):
        """The 22 decomposes 21 + 1, and only the 21 is this function's to fix.

        Feeding the union reader's ``available`` (mechanism 1) still leaves a
        one-row surplus on every specimen. That row is #927's false positive --
        the checker auditing the BOOT phase's allocator while load-back
        allocated from the incoming one -- and it is removed at its root by the
        per-access resolution in ``_allocator()``, not by a subtracted term
        here. This test is what makes the split explicit now that the term that
        used to absorb the 1 is gone.
        """
        for label, total, available, evictable, withheld in SPECIMENS_912:
            with self.subTest(specimen=label):
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
                )
                self.assertTrue(
                    leak, "mechanism 1 alone must leave a 1-row residual"
                )
                self.assertEqual(
                    deduped_available + evictable + withheld - total,
                    MECH2_ROWS,
                    "the residual must be exactly the one double-claimed row",
                )

    def test_preexisting_callers_are_unaffected(self):
        """The other ``_check_pool_invariant`` call sites pass at most 7
        positional args (verified by inspection of the callers in
        ``invariant_checker.py``: ``_check_swa_pool``, ``_check_mamba_pool``,
        and both branches of ``_check_mamba_pool_with_int8``). None of them
        must be made to pass ``withheld`` by this change.
        """
        leak, msg = _check_pool_invariant("swa", 100, 0, 0, 0, 100)
        self.assertFalse(leak, msg)
        leak, msg = _check_pool_invariant("mamba", 90, 5, 5, 0, 100)
        self.assertFalse(leak, msg)

    def test_the_signature_carries_no_double_owned_term_969(self):
        """#969 CUT C, pinned so it cannot be undone by accident.

        The term is not merely unused -- it is GONE from the signature, so a
        caller that tries to pass it raises instead of silently changing the
        equation, which is the coupling ``77b42d6d0a`` names (producer, term
        and consumer go together). This is the same assertion that commit's own
        one-tool check made.
        """
        params = inspect.signature(
            SchedulerInvariantChecker._check_pool_invariant
        ).parameters
        self.assertNotIn(
            "double_owned",
            params,
            "the #969 CUT C deletion has been reverted; if that is intended, "
            "the GIL-spin measurement in this file's docstring has to be "
            "answered first",
        )
        self.assertIn("withheld", params, "the #656 posten must survive")
        self.assertFalse(
            hasattr(SchedulerInvariantChecker, "_live_double_claimed_rows"),
            "the deleted producer is back; the term's consumer will follow",
        )
        with self.assertRaises(TypeError):
            _check_pool_invariant("full", 1, 0, 0, 0, 1, 0, 0, 0)


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


class TestADeficitStaysFatal912(CustomTestCase):
    """The #832/#856 direction -- rows with NO owner -- must never pass.

    This used to need a class of its own arguing that a stale ``double_owned``
    reading could not mask a deficit. #969 CUT C removed the subtracted term,
    so the equation is a plain sum and the property is structural rather than
    argued; one test is enough to keep it pinned.
    """

    def test_a_manufactured_deficit_reads_as_a_leak(self):
        label, total, available, evictable, withheld = SPECIMENS_912[0]
        deficit_available = available - 100  # a genuine, separate 100-row hole
        leak, msg = _check_pool_invariant(
            "full", deficit_available, evictable, 0, 0, total, 0, withheld
        )
        self.assertTrue(leak, f"a 100-row deficit must stay fatal: {msg}")


class TestNoPageSizePaddingInTheLedger912(CustomTestCase):
    """Refutation, red-first, of the ``reservation + page_size`` framing.

    MOVED HERE from ``test_pool_invariant_live_double_912b.py``, which #969
    CUT C deleted along with the ``_live_double_claimed_rows`` producer its
    other four classes drove. The refutation itself survives the cut because it
    is a property of the ALLOCATOR's id space, not of the deleted term.

    THE FRAMING IT REFUTES. Boot 2 of the 2c acceptance died on all three ranks
    at ``/spinning/evidence-665-f1/boot_accept2c0827_0827_0049.log:21372`` with
    a surplus of exactly ONE row (``total=432089, available=133120,
    evictable=1, withheld=298969``). The acceptance note's hypothesis was that
    the 1 is structural -- "``total`` is the reservation while
    available/withheld enumerate over reservation + page_size", the same ``+1``
    observable B2 measured as ``store_bound_rows - reserved_backing_rows ==
    page_size`` on 435 of 435 dial lines. It is not:
    ``TokenToKVPoolAllocator.clear()`` builds ``torch.arange(1, self.size + 1)``
    (``mem_cache/allocator/token.py:44-45``), so the id space is ``1 .. size``,
    exactly ``size`` ids, and ``total`` for that branch is the same
    ``allocator.size``. Row 0 is never in a free list and never handed out.
    Both sides are the same id space; there is no ``page_size`` padding on
    either. The pool tensor's extra row is real and is what
    ``store_bound_rows`` describes, but it is not an id the ledger ever counts.

    WHAT THE 1 ACTUALLY WAS: the same boot's post-cutover census one second
    earlier (:21367) partitions the id space with nothing left over --
    ``size=432089 free=133120 cached=0 withheld=298969 unaccounted=0`` -- so
    ``free + withheld == size`` exactly and the ``evictable=1`` read a second
    later is a row the tree took on between the two readings (the boot's own
    OUTTRACE names the arrival, ``HEALTH_C n=1 (new) off=1 tail=[49276]`` at
    :21340, and 49276 is inside the free range). One row, two owners -- which
    is #927's shape, and #927 answered it by resolving the allocator per
    access rather than by subtracting the row.
    """

    #: The measured firing, identical on PP0/PP1/PP2.
    SPECIMEN_TOTAL = 432089
    SPECIMEN_AVAILABLE = 133120
    SPECIMEN_EVICTABLE = 1
    SPECIMEN_WITHHELD = 298969
    #: The census one second earlier, same ranks.
    CENSUS_FREE = 133120
    CENSUS_CACHED = 0

    def test_the_free_list_spans_exactly_size_ids_starting_at_one(self):
        alloc = TestAvailableSizeUnionVsSum912._Alloc(1000)
        self.assertEqual(int(alloc.free_pages.numel()), 1000)
        self.assertEqual(int(alloc.free_pages.min()), 1)
        self.assertEqual(int(alloc.free_pages.max()), 1000)

    def test_row_zero_is_not_an_id_the_ledger_counts(self):
        alloc = TestAvailableSizeUnionVsSum912._Alloc(1000)
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
        accounted = (
            self.SPECIMEN_AVAILABLE + self.SPECIMEN_EVICTABLE + self.SPECIMEN_WITHHELD
        )
        self.assertEqual(accounted - self.SPECIMEN_TOTAL, 1)
        # And the census a second earlier leaves NO room for an unclaimed id:
        self.assertEqual(
            self.CENSUS_FREE + self.SPECIMEN_WITHHELD, self.SPECIMEN_TOTAL
        )
        self.assertEqual(self.CENSUS_CACHED, 0)


if __name__ == "__main__":
    unittest.main()
