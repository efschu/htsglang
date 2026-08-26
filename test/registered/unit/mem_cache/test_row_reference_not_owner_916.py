"""#916: a REFERENCE to a KV row is not one of that row's owners.

THE SPECIMEN. Rerun acceptance window 2026-08-26, boot #2
(``/spinning/evidence-665-f1/boot_rerun0826_0826_2149.log``), 21:53:13, the
same three lines on all three ranks::

    PHASE-FLIP POOL CENSUS at-arm pp_to_tp: size=468981 free=443648
      cached=12281 withheld=0 ... resident_reqs=2 unaccounted=13052
    KV-OWNERSHIP VIOLATION (at-arm pp_to_tp) [exclusivity] 12280 rows:
      12280 row id(s) are claimed by more than one owner
      [('radix_cache', 'resident:requests')]; two owners writing the same KV
      row is silent corruption, not a crash sample=[2, 3, 4, 5, 6, 7, 8, 9]

33 such lines in one boot, six of them this shape (12280 x3, 4096 x3).

THE ARITHMETIC NAMES IT BEFORE ANY CODE IS READ. ``cached=12281`` and the
violation is ``12280`` -- every cached row but one is also claimed by a
resident request -- and ``sample`` is the LOWEST ids in the space, the head of
a sequence. That is not a corruption pattern; it is a shared prefix.

AND THE CODE SAYS IT OUTRIGHT. ``RadixCache.cache_unfinished_req`` inserts the
request's rows into the tree, takes one ref in the memory pool
(``radix_cache.py:509``), and then writes the TREE's ids back into
``req_to_token`` (``:534-537``). From that instant the tree and the request
name the same ids ON PURPOSE, refcounted by ``inc_lock_ref``. The census reads
``tree.all_values_flatten()`` for ``radix_cache`` and
``req_to_token[idx, :seqlen]`` for ``resident:requests``, so on any stack with
prefix caching on, those two sets OVERLAP BY CONSTRUCTION.

VERDICT: CENSUS ARTEFACT, of the same family as #832's ~94000 false unowned
rows and #822 root A's 122-row false leak -- an instrument that had no term
for a legitimate relation and reported it as a defect. It is a real finding
and a DIFFERENT fix from the one a genuine double claim would need.

WHAT THE FIX IS NOT. It is not silencing the law. The pair that WAS dangerous
was being printed in the same sentence, with the same severity, as this one:
``('free_list', 'resident:requests')`` -- a live request still naming a row
the allocator has put back on its free list, which the next ``alloc`` hands to
a second writer. Window 2 printed it 8 times and nobody could triage it out of
the noise. It now has its own violation kind and its own message.

THE CLASS, not the instance. ``Claim`` carries a ROLE. ``free_list``,
``radix_cache`` and ``cap_withheld`` are mutually exclusive row STATES;
``resident:*`` is a REFERENCE to whatever state holds the row. LAW.EXCLUSIVITY
partitions the states and leaves references to the two checks that are
actually about them: they still answer the coverage question (a resident row
is not unowned -- that is the whole of #822 root A), and they must never
overlap the free list.

THE RETRACTION SUSPECTS ARE FALSIFIED BY ORDERING, not by argument. #856 (an
empty flip plan retracting residents), #842 (the cutover cold-starting the
downstream radix) and #888b (carrier seats resident in the forbidden layout)
all act inside ``_execute`` -- ``_release_residents_for_cutover`` is reached at
``phase_flip_runtime.py:11094``. The at-arm census is taken in ``arm()``
(``phase_flip_runtime.py:4788``), before any of them runs. Whatever a
retraction does to the claim ledger, it cannot be what the at-arm line reports.

SIBLING SWEEP, recorded because the absence of a second instance is itself a
result. The tp_to_pp arm runs the identical code and printed NO doubled
violation in this boot -- its at-arm census read ``cached=4096`` with the live
request holding 4097..5964, a request whose prefix had not been cached yet, so
the overlap happened to be empty. Same defect, different luck. The tp_to_pp
census is pinned below so a future boot cannot be read as "only pp_to_tp is
affected".
"""

import inspect
import unittest

import torch

from sglang.srt.mem_cache.kv_row_ownership import (
    EXCLUSIVITY_DOUBLED,
    EXCLUSIVITY_FREED_REFERENCE,
    EXCLUSIVITY_UNOWNED,
    ROLE_EXCLUSIVE,
    ROLE_FREE,
    ROLE_REFERENCE,
    Law,
    RowOwnershipAuthority,
    RowSpace,
    audit_pool_census,
)
from sglang.srt.mem_cache.memory_pool import (
    CpuCopyIdsUnreadable,
    CpuCopyUnmappedRows,
    check_cpu_copy_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# The measured boot-#2 numbers, used as-is so the tests name the specimen.
SIZE = 468981
FREE = 443648
CACHED = 12281
DOUBLED_ROWS = 12280
COMMITTED = 485366


def _excl(violations, kind):
    return [v for v in violations if v.law == Law.EXCLUSIVITY and v.kind == kind]


class TestTheSharedPrefixIsNotACorruption(CustomTestCase):
    """The specimen, reproduced at its own scale and refused as a violation."""

    @staticmethod
    def _specimen_authority():
        """One request whose whole cached prefix is in the tree.

        Scaled down from the boot's 12281/12280 but built the same way:
        ``cache_unfinished_req`` put rows 2..12281 in the tree AND wrote those
        same ids into ``req_to_token``, so both readers see them.
        """
        auth = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        auth.declare("free_list", range(30, 100), role=ROLE_FREE)
        auth.declare("radix_cache", range(1, 21), role=ROLE_EXCLUSIVE)
        # rows 1..20 are the tree's; the request references all of them and
        # additionally holds 21..29, its uncached tail.
        auth.declare("resident:requests", range(1, 30), role=ROLE_REFERENCE)
        return auth

    def test_the_field_shape_is_no_longer_a_violation(self):
        found = self._specimen_authority().audit()
        self.assertEqual(
            _excl(found, EXCLUSIVITY_DOUBLED),
            [],
            "the tree and the request that references its ids are not two "
            "owners writing the same row -- cache_unfinished_req makes them "
            "the same ids on purpose (radix_cache.py:534-537)",
        )

    def test_the_reference_still_answers_the_coverage_question(self):
        """#822 root A must not come back. The request's UNCACHED rows 21..29
        are in no state at all, and they must still read as owned."""
        found = self._specimen_authority().audit()
        self.assertEqual(
            _excl(found, EXCLUSIVITY_UNOWNED),
            [],
            "a row held by a live request is owned by that request; reporting "
            "it as belonging to nobody is #822 root A, the defect the fourth "
            "owner was added to close",
        )

    def test_mutant_declaring_the_reference_as_a_state_reproduces_the_field_line(self):
        """THE DYING MUTANT. Revert the role and the boot's own violation
        comes straight back, at exactly the overlap size."""
        auth = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        auth.declare("free_list", range(30, 100), role=ROLE_FREE)
        auth.declare("radix_cache", range(1, 21), role=ROLE_EXCLUSIVE)
        auth.declare("resident:requests", range(1, 30), role=ROLE_EXCLUSIVE)

        doubled = _excl(auth.audit(), EXCLUSIVITY_DOUBLED)
        self.assertEqual(len(doubled), 1)
        self.assertEqual(doubled[0].rows, 20)
        self.assertIn("radix_cache", doubled[0].detail)
        self.assertIn("resident:requests", doubled[0].detail)

    def test_the_specimen_ratio_holds_at_the_measured_scale(self):
        """Not a scaled analogue this time: the boot's own numbers.

        ``cached=12281`` with ``12280`` doubled means the overlap was the whole
        tree bar one row. Built that way, the old law fires at 12280 and the
        new one is silent -- which is the two-state discrimination the census
        integer could never make.
        """
        tree = range(2, 2 + CACHED)
        resident = range(2, 2 + DOUBLED_ROWS)  # every cached row but the last

        old = RowOwnershipAuthority(RowSpace(exposed=SIZE, committed=COMMITTED))
        old.declare("radix_cache", tree, role=ROLE_EXCLUSIVE)
        old.declare("resident:requests", resident, role=ROLE_EXCLUSIVE)
        self.assertEqual(
            _excl(old.audit(expect_full_coverage=False), EXCLUSIVITY_DOUBLED)[0].rows,
            DOUBLED_ROWS,
        )

        new = RowOwnershipAuthority(RowSpace(exposed=SIZE, committed=COMMITTED))
        new.declare("radix_cache", tree, role=ROLE_EXCLUSIVE)
        new.declare("resident:requests", resident, role=ROLE_REFERENCE)
        self.assertEqual(
            _excl(new.audit(expect_full_coverage=False), EXCLUSIVITY_DOUBLED), []
        )


class TestTheLawKeepsItsTeeth(CustomTestCase):
    """A fix that only removed a log line would be the wrong fix."""

    def test_a_reference_over_a_freed_row_is_its_own_named_violation(self):
        """The shape that WAS dangerous and was lost in the noise.

        ('free_list', 'resident:requests') -- window 2 printed it 8 times,
        indistinguishable from the benign pair. It is a use-after-free: the
        row is back in the free list and the next alloc hands it to a second
        writer while this request is still reading it.
        """
        auth = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        auth.declare("free_list", range(50, 100), role=ROLE_FREE)
        auth.declare("radix_cache", range(1, 50), role=ROLE_EXCLUSIVE)
        auth.declare("resident:requests", [55, 56, 57], role=ROLE_REFERENCE)

        found = auth.audit()
        freed = _excl(found, EXCLUSIVITY_FREED_REFERENCE)
        self.assertEqual(len(freed), 1)
        self.assertEqual(freed[0].rows, 3)
        self.assertEqual(freed[0].sample, (55, 56, 57))
        self.assertIn("use-after-free", freed[0].detail)
        self.assertEqual(
            _excl(found, EXCLUSIVITY_DOUBLED),
            [],
            "the dangerous shape must be its OWN kind, not folded back into "
            "the one this ticket just made benign",
        )

    def test_the_two_reference_shapes_are_told_apart_in_one_audit(self):
        """Both at once, which is the realistic census: a shared prefix AND a
        row that was freed under the request."""
        auth = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        auth.declare("free_list", range(50, 100), role=ROLE_FREE)
        auth.declare("radix_cache", range(1, 50), role=ROLE_EXCLUSIVE)
        auth.declare(
            "resident:requests", list(range(1, 20)) + [77], role=ROLE_REFERENCE
        )

        found = auth.audit()
        self.assertEqual(_excl(found, EXCLUSIVITY_DOUBLED), [])
        freed = _excl(found, EXCLUSIVITY_FREED_REFERENCE)
        self.assertEqual(len(freed), 1)
        self.assertEqual(
            freed[0].rows,
            1,
            "only the freed row counts; the 19 shared prefix rows in the SAME "
            "claim must not inflate it",
        )

    def test_the_genuine_state_double_claim_still_fires(self):
        """#912 mechanism 2's own specimen: free_list and radix_cache. Two row
        STATES, mutually exclusive, and this one is still a violation."""
        auth = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        auth.declare("free_list", range(1, 100), role=ROLE_FREE)
        auth.declare("radix_cache", [1], role=ROLE_EXCLUSIVE)

        doubled = _excl(auth.audit(), EXCLUSIVITY_DOUBLED)
        self.assertEqual(len(doubled), 1)
        self.assertEqual(doubled[0].rows, 1)

    def test_default_role_is_exclusive_so_an_unupdated_caller_cannot_go_quiet(self):
        """A caller that says nothing means what every caller meant before
        #916. Defaulting to REFERENCE would silently disarm the law for any
        owner someone forgot to classify."""
        auth = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        auth.declare("free_list", [1, 2, 3])
        auth.declare("radix_cache", [3])
        self.assertEqual(
            len(_excl(auth.audit(expect_full_coverage=False), EXCLUSIVITY_DOUBLED)), 1
        )


class TestTheLedgerTermSurvivesTheVerdictChange(CustomTestCase):
    """#912's correction term asks a different question and still needs an
    answer. The law says the share is lawful; the on-idle ledger still counts
    those rows twice (once evictable, once session_held) and must still
    subtract them, or #822 root A returns through the invariant checker."""

    def test_shared_reference_rows_counts_the_overlap_the_law_now_permits(self):
        auth = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        auth.declare("radix_cache", range(1, 21), role=ROLE_EXCLUSIVE)
        auth.declare("resident:requests", range(1, 30), role=ROLE_REFERENCE)
        self.assertEqual(auth.shared_reference_rows(), 20)

    def test_a_row_shared_with_no_state_is_not_a_ledger_surplus(self):
        """The request's uncached tail is counted ONCE in the ledger, so it
        must not be subtracted."""
        auth = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        auth.declare("radix_cache", [], role=ROLE_EXCLUSIVE)
        auth.declare("resident:requests", range(1, 30), role=ROLE_REFERENCE)
        self.assertEqual(auth.shared_reference_rows(), 0)

    def test_it_is_not_derived_from_the_violation_list(self):
        """A correction term sourced from a verdict changes whenever the
        verdict does -- which is exactly what #916 did to it."""
        auth = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        auth.declare("radix_cache", range(1, 21), role=ROLE_EXCLUSIVE)
        auth.declare("resident:requests", range(1, 21), role=ROLE_REFERENCE)
        self.assertEqual(
            _excl(auth.audit(expect_full_coverage=False), EXCLUSIVITY_DOUBLED), []
        )
        self.assertEqual(auth.shared_reference_rows(), 20)

    def test_a_retired_claim_is_not_a_ledger_surplus(self):
        """#796's epoch rule, applied to the ledger term too: ids from a dead
        id space are not rows and cannot be double-counted.

        BUILT AS A #796 SURVIVOR, not by calling ``retire()``. ``retire()``
        drops every claim, so a test that used it would pass against an
        authority with no epoch filter at all -- measured: that mutant survived
        the first cut of this test. The shape that actually needs the filter is
        a claim carrying a PRE-cutover stamp while the space has moved on,
        which is what ``audit`` reports as LAW.RETIREMENT and what a replayed
        census reconstructs.
        """
        auth = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        auth.retire(exposed=99, committed=100)  # space epoch is now 1
        auth.declare("radix_cache", range(1, 21), role=ROLE_EXCLUSIVE)
        auth.declare("resident:requests", range(1, 21), role=ROLE_REFERENCE)
        self.assertEqual(auth.shared_reference_rows(), 20)

        stale = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        stale.retire(exposed=99, committed=100)
        stale.declare("radix_cache", range(1, 21), role=ROLE_EXCLUSIVE)
        stale.declare("resident:requests", range(1, 21), role=ROLE_REFERENCE, epoch=0)
        self.assertEqual(
            stale.shared_reference_rows(),
            0,
            "a claim from the retired id space must not correct a ledger it "
            "was never measured against -- #912's own carried lesson",
        )
        self.assertTrue(
            any(
                v.law == Law.RETIREMENT for v in stale.audit(expect_full_coverage=False)
            )
        )


class TestTheCensusBridgeDeclaresTheRole(CustomTestCase):
    """PRESENT-AND-VERDRAHTET. The role is only a fix if the production census
    path declares it -- the middle delivery state is the expensive one."""

    def test_audit_pool_census_end_to_end_on_the_field_shape(self):
        auth = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        found = audit_pool_census(
            auth,
            exposed=99,
            committed=100,
            free_rows=range(30, 100),
            cached_rows=range(1, 21),
            withheld_rows=(),
            resident_rows={"requests": range(1, 30)},
            why="at-arm pp_to_tp",
        )
        self.assertEqual(
            found,
            [],
            "the production bridge must produce a CLEAN census for the "
            "ordinary shared-prefix state that killed 33 log lines",
        )
        self.assertEqual(auth.shared_reference_rows(), 20)

    def test_the_bridge_still_reports_a_freed_reference(self):
        auth = RowOwnershipAuthority(RowSpace(exposed=99, committed=100))
        found = audit_pool_census(
            auth,
            exposed=99,
            committed=100,
            free_rows=range(30, 100),
            cached_rows=range(1, 30),
            withheld_rows=(),
            resident_rows={"requests": [1, 2, 44]},
            why="at-arm pp_to_tp",
        )
        freed = _excl(found, EXCLUSIVITY_FREED_REFERENCE)
        self.assertEqual(len(freed), 1)
        self.assertEqual(freed[0].rows, 1)
        self.assertEqual(freed[0].sample, (44,))

    def test_observe_census_declares_resident_as_a_reference(self):
        from sglang.srt.mem_cache import kv_row_ownership

        src = inspect.getsource(kv_row_ownership.RowOwnershipAuthority.observe_census)
        self.assertIn("role=ROLE_REFERENCE", src)
        self.assertIn("role=ROLE_FREE", src)

    def test_the_ratchet_every_declared_owner_states_a_role(self):
        """THE CLASS RATCHET, not the instance.

        ``declare`` defaults to ROLE_EXCLUSIVE, which is the safe default -- a
        forgotten owner keeps the law's teeth rather than losing them. But
        "safe" and "right" are different: a FIFTH owner that is really a
        reference (a host-tier holder, a spilled-rows register) would be added
        the way the fourth was, silently, and republish the same false
        corruption. So the bridge is held to stating a role for every owner it
        declares -- the count is checked, not a list of names, because a name
        list is the thing that goes stale when the fifth owner arrives.
        """
        from sglang.srt.mem_cache import kv_row_ownership

        src = inspect.getsource(kv_row_ownership.RowOwnershipAuthority.observe_census)
        declares = src.count("self.declare(")
        roles = src.count("role=ROLE_")
        self.assertEqual(
            declares,
            roles,
            f"observe_census makes {declares} declare() calls but states only "
            f"{roles} roles -- an owner was added without saying whether it is "
            f"a row STATE or a REFERENCE, which is exactly how #916 happened",
        )
        self.assertGreaterEqual(declares, 4)

    def test_the_ledger_term_is_wired_into_the_census_audit(self):
        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        src = inspect.getsource(PhaseFlipRuntime._census_ownership_audit)
        self.assertIn(
            "shared_reference_rows()",
            src,
            "the #912 correction term no longer picks up the share the #916 "
            "law stopped reporting -- the on-idle ledger would read the "
            "working set as a leak again",
        )
        # #912's own structural discriminator must survive this change.
        self.assertIn("EXCLUSIVITY_DOUBLED", src)


class TestTheSiblingArm(CustomTestCase):
    """tp_to_pp runs the identical code. It printed no doubled line in boot #2
    only because that arm's live request had no cached prefix yet -- same
    defect, different luck. Pinned so a later boot cannot be read as
    'pp_to_tp only'."""

    def test_a_request_with_no_cached_prefix_never_showed_the_defect(self):
        """The measured at-arm tp_to_pp census: cached=4096 (an earlier
        request's), the live request holding 4097..5964, overlap zero."""
        auth = RowOwnershipAuthority(RowSpace(exposed=SIZE, committed=COMMITTED))
        auth.declare("radix_cache", range(1, 4097), role=ROLE_EXCLUSIVE)
        auth.declare("resident:requests", range(4097, 5965), role=ROLE_EXCLUSIVE)
        self.assertEqual(
            _excl(auth.audit(expect_full_coverage=False), EXCLUSIVITY_DOUBLED),
            [],
            "the pre-fix law was silent on this arm only because the overlap "
            "was empty, not because the arm was sound",
        )

    def test_the_same_arm_fires_the_moment_that_request_is_cached(self):
        auth = RowOwnershipAuthority(RowSpace(exposed=SIZE, committed=COMMITTED))
        auth.declare("radix_cache", range(1, 4097), role=ROLE_EXCLUSIVE)
        auth.declare("resident:requests", range(1, 5965), role=ROLE_EXCLUSIVE)
        self.assertEqual(
            _excl(auth.audit(expect_full_coverage=False), EXCLUSIVITY_DOUBLED)[0].rows,
            4096,
            "cache_unfinished_req on the tp_to_pp arm produces the identical "
            "shape; the fix must cover both arms and does, because the role "
            "is declared once in observe_census",
        )


# ----------------------------------------------------------------------
# POSTEN 2 -- the #913 guard could not decline, because it died on its own
# first device sync.
# ----------------------------------------------------------------------
class _UnreadableIndices:
    """A tensor whose ids cannot be read -- a CUDA context already faulted.

    Boot #2, 21:53:36: the SAME rank had logged "#760 quiesce ... synchronizing
    load_stream failed (CUDA error: an illegal memory access was encountered)"
    one line before the traceback. Every subsequent device call on that context
    raises. ``indices.min()`` is a device reduction, so the guard raised
    ``AcceleratorError`` from inside itself and the rank died with a traceback
    naming ``memory_pool.py:664 lo = int(indices.min())`` -- an instrument
    reported as a cause. ``SEAM COPY DECLINED: 0`` in a boot that died on the
    seam copy path.
    """

    def __init__(self, n=64):
        self._n = n

    def numel(self):
        return self._n

    def min(self):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    def max(self):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


class TestTheGuardDeclinesInsteadOfDying(CustomTestCase):
    def test_an_unreadable_id_set_is_a_decline_not_an_escape(self):
        with self.assertRaises(CpuCopyIdsUnreadable):
            check_cpu_copy_rows(
                _UnreadableIndices(), 485366, "offload", "row", backed_rows=114688
            )

    def test_the_decline_reaches_the_seam_through_the_existing_except(self):
        """A SUBCLASS on purpose: `seam_copy_state` catches
        `CpuCopyUnmappedRows` and already answers it with recompute. A sibling
        type would have needed a second except clause somewhere, and the one
        that got forgotten would be the one that killed a rank."""
        self.assertTrue(issubclass(CpuCopyIdsUnreadable, CpuCopyUnmappedRows))

    def test_it_says_it_is_not_a_verdict_about_the_ids(self):
        try:
            check_cpu_copy_rows(
                _UnreadableIndices(), 485366, "offload", "row", backed_rows=114688
            )
        except CpuCopyIdsUnreadable as exc:
            self.assertIn("could not be read", str(exc))
            self.assertIn("not a verdict", str(exc))
        else:  # pragma: no cover - the assertion above is the test
            self.fail("the guard did not decline")

    def test_a_zero_backing_is_refused_with_no_device_access_at_all(self):
        """The one mapped verdict that IS answerable sync-free.

        ``backed_rows`` is a host int. Zero from a pool that HAS an arena means
        every page is released, so no id can be mapped and none needs reading.
        The stub below raises on any device call, which is the proof that the
        refusal took none.
        """
        with self.assertRaises(CpuCopyUnmappedRows) as caught:
            check_cpu_copy_rows(
                _UnreadableIndices(), 485366, "offload", "row", backed_rows=0
            )
        self.assertNotIsInstance(caught.exception, CpuCopyIdsUnreadable)
        self.assertIn("without", str(caught.exception))

    def test_none_backing_still_means_the_question_does_not_apply(self):
        """A pool with no arena answers None, not zero. Refusing every copy off
        the dial lane would be the opposite defect."""
        check_cpu_copy_rows(
            torch.tensor([1, 2, 3]), 10, "offload", "row", backed_rows=None
        )

    def test_the_913_verdicts_are_unchanged(self):
        """The reordering must not move either existing refusal."""
        with self.assertRaises(CpuCopyUnmappedRows):
            check_cpu_copy_rows(
                torch.tensor([122898]), 485366, "offload", "row", backed_rows=114688
            )
        with self.assertRaises(ValueError):
            check_cpu_copy_rows(
                torch.tensor([485366]), 485366, "offload", "row", backed_rows=114688
            )
        with self.assertRaises(ValueError):
            check_cpu_copy_rows(
                torch.tensor([-1]), 485366, "offload", "row", backed_rows=114688
            )
        check_cpu_copy_rows(
            torch.tensor([114687]), 485366, "offload", "row", backed_rows=114688
        )

    def test_the_seam_counts_the_two_declines_apart(self):
        from sglang.srt.managers.schedule_batch import _SEAM_STATE_COUNTS

        self.assertIn("declined_unreadable", _SEAM_STATE_COUNTS)
        self.assertIn("declined_unmapped", _SEAM_STATE_COUNTS)

    def test_the_seam_routes_the_subclass_to_its_own_counter(self):
        from sglang.srt.managers import schedule_batch

        src = inspect.getsource(schedule_batch.seam_copy_state)
        self.assertIn("CpuCopyIdsUnreadable", src)
        self.assertIn("declined_unreadable", src)
        # #913's narrowness must survive: a bare ValueError would swallow the
        # #783b addressing defect.
        self.assertNotIn("except ValueError", src)


if __name__ == "__main__":
    unittest.main()
