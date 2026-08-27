"""#919 -- name the owner the census did not enumerate, three-valued.

WHAT THE LINE ALREADY SAID, and what the ticket read instead. Measured on the
0826 rerun, three ranks, both boots::

    KV-OWNERSHIP VIOLATION (pre-cutover tp_to_pp) [exclusivity] 4096 rows:
    4096 committed row id(s) of 228897 (1.8%) belong to NO ENUMERATED OWNER;
    on this stack that has meant an UN-ENUMERATED SECOND POOL OBJECT, NOT A
    LEAK sample=[1, 2, 3, 4, 5, 6, 7, 8]

#919 as filed read that as "the tree LOSES 4096 rows without free" and hung
#842 on it. The line says the other thing, and it says it in its own text.

THE READING IS NOT STALE, checked first and negative: the source is
`_pool_census` (phase_flip_runtime.py:5990-5993), which resolves
`token_to_kv_pool_allocator` PER ACCESS from the scheduler -- exactly the form
the #927 fix had to install at its three construction sites. So #919 is not
the fourth false positive of that family; the number holds, only its
interpretation did not.

THREE STRUCTURAL FACTS that make a second pool the leading candidate:
  * always `sample=[1..8]` and always exactly 4096 -- a fixed block at the
    BOTTOM of the id space, independent of the total (228897/140961/148289);
  * the withheld set is by construction the TOP
    (`range(size - withheld_n + 1, size + 1)`), so the block cannot be that;
  * the same shape in the 2g boot at 2047-of-2048, i.e. the entire committed
    backing was unowned.

WHY THREE VALUES. "A second pool covers it" and "a second pool exists but does
not" are both "not a leak" but only the first explains the block; "no second
pool" is the one that sends the hunt to the release/retirement paths. Folding
them into a boolean would lose exactly the distinction the probe exists to
draw.

ADDITIVE ONLY. The probe adds one log line per unowned violation. It does not
add, remove or reclassify a violation, and it cannot raise -- an instrument
that changes a verdict is not an instrument.

WHAT EACH TEST HOLDS DOWN
  1. a covering candidate is named                 -- the defect's explanation;
  2. a present-but-disjoint candidate is NOT named as covering -- the mutant
     guard: a probe that answers COVERS whenever any pool exists explains
     nothing and would retire a real hunt;
  3. no candidate at all reads NO-SECOND-POOL      -- the answer that keeps the
     hunt alive;
  4. an empty sample cannot be tested              -- honest, rather than a
     COVERS by vacuous truth over zero rows;
  5. the rig's measured shape resolves             -- the 0826 sample against a
     draft-sized pool.
"""

import unittest

from sglang.srt.mem_cache.kv_row_ownership import (
    CANDIDATE_ABSENT,
    CANDIDATE_COVERS,
    CANDIDATE_DISJOINT,
    OwnerCandidate,
    unenumerated_owner_verdict,
)

#: The block measured on all three ranks of the 0826 rerun.
_RERUN_SAMPLE = (1, 2, 3, 4, 5, 6, 7, 8)


class TestUnenumeratedOwnerProbe919(unittest.TestCase):
    def test_a_covering_candidate_is_named(self):
        cands = [OwnerCandidate(name="draft_allocator", lo=1, hi=4097)]
        verdict, detail = unenumerated_owner_verdict(_RERUN_SAMPLE, cands)
        self.assertEqual(verdict, CANDIDATE_COVERS)
        self.assertIn("draft_allocator", detail)
        self.assertIn("did not enumerate", detail)

    def test_a_disjoint_candidate_is_not_reported_as_covering(self):
        """MUTANT GUARD. A probe that says COVERS whenever any pool exists
        explains nothing and would retire a hunt that is still needed."""
        cands = [OwnerCandidate(name="tp_worker_allocator", lo=500_000, hi=600_000)]
        verdict, detail = unenumerated_owner_verdict(_RERUN_SAMPLE, cands)
        self.assertEqual(verdict, CANDIDATE_DISJOINT)
        self.assertIn("none covers the sample", detail)
        self.assertIn("tp_worker_allocator", detail)

    def test_no_candidate_reads_as_genuinely_ownerless(self):
        verdict, detail = unenumerated_owner_verdict(_RERUN_SAMPLE, [])
        self.assertEqual(verdict, CANDIDATE_ABSENT)
        self.assertIn("no second pool", detail)

    def test_an_empty_sample_cannot_be_tested(self):
        """`all()` over an empty sample is True, so a naive probe would answer
        COVERS for a violation that carried no evidence at all."""
        cands = [OwnerCandidate(name="draft_allocator", lo=1, hi=4097)]
        verdict, detail = unenumerated_owner_verdict((), cands)
        self.assertEqual(verdict, CANDIDATE_DISJOINT)
        self.assertIn("no sample", detail)

    def test_the_first_covering_candidate_wins_over_a_later_disjoint_one(self):
        cands = [
            OwnerCandidate(name="far_pool", lo=900_000, hi=910_000),
            OwnerCandidate(name="draft_allocator", lo=1, hi=4097),
        ]
        verdict, detail = unenumerated_owner_verdict(_RERUN_SAMPLE, cands)
        self.assertEqual(verdict, CANDIDATE_COVERS)
        self.assertIn("draft_allocator", detail)

    def test_the_range_is_one_based_and_hi_exclusive(self):
        """The census's id space is `range(1, size + 1)`. An off-by-one here
        turns a covering candidate into a disjoint one -- wrong in the
        expensive direction, because it sends the hunt to reset_tree for a
        block that was explained all along."""
        exact = OwnerCandidate(name="p", lo=1, hi=4097)
        self.assertTrue(exact.covers([1, 4096]))
        self.assertFalse(exact.covers([4097]))
        self.assertFalse(OwnerCandidate(name="p", lo=2, hi=4097).covers([1]))


class _Alloc:
    def __init__(self, size):
        self.size = size


class _Runner:
    def __init__(self, alloc):
        self.token_to_kv_pool_allocator = alloc


class _Worker:
    def __init__(self, alloc):
        self.model_runner = _Runner(alloc)


class _Stacks:
    def __init__(self, tp_alloc):
        self.tp_worker = _Worker(tp_alloc)


class _Sched:
    """A scheduler whose census allocator is NOT the draft pool -- the shape
    the rig actually runs."""

    def __init__(self, *, draft_size=4096, tp_size=None):
        self.token_to_kv_pool_allocator = _Alloc(228897)
        self.draft_token_to_kv_pool_allocator = (
            _Alloc(draft_size) if draft_size else None
        )
        self.phase_flip_stacks = _Stacks(_Alloc(tp_size)) if tp_size else None


class TestTheCensusPrintsTheCandidate919(unittest.TestCase):
    """The probe must run where the violation is RAISED, or it is a desk
    artefact. A pure verdict function nobody calls answers nothing."""

    def _runtime(self, sched):
        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        rt = object.__new__(PhaseFlipRuntime)
        rt._census_scheduler = sched
        return rt

    def _unowned(self, rows=4096, sample=_RERUN_SAMPLE):
        from sglang.srt.mem_cache.kv_row_ownership import (
            EXCLUSIVITY_UNOWNED,
            Law,
            Violation,
        )

        return Violation(
            law=Law.EXCLUSIVITY,
            rows=rows,
            detail="belong to no enumerated owner",
            sample=tuple(sample),
            kind=EXCLUSIVITY_UNOWNED,
        )

    def test_a_synthetic_census_prints_the_covers_line(self):
        rt = self._runtime(_Sched(draft_size=4096))
        with self.assertLogs(
            "sglang.srt.managers.phase_flip_runtime", level="WARNING"
        ) as cm:
            rt._note_unenumerated_owner([self._unowned()])
        blob = "\n".join(cm.output)
        self.assertIn("#919 UNOWNED-BLOCK", blob)
        self.assertIn(CANDIDATE_COVERS, blob)
        self.assertIn("draft_allocator", blob)

    def test_with_no_second_pool_it_says_so(self):
        """The answer that keeps the reset_tree hunt alive."""
        rt = self._runtime(_Sched(draft_size=None))
        with self.assertLogs(
            "sglang.srt.managers.phase_flip_runtime", level="WARNING"
        ) as cm:
            rt._note_unenumerated_owner([self._unowned()])
        self.assertIn(CANDIDATE_ABSENT, "\n".join(cm.output))

    def test_a_doubled_violation_is_not_probed(self):
        """MUTANT GUARD on reach: the probe is about the UNOWNED shape only.
        #916 showed the doubled shape is a different question with a different
        (already amended) answer."""
        from sglang.srt.mem_cache.kv_row_ownership import (
            EXCLUSIVITY_DOUBLED,
            Law,
            Violation,
        )

        doubled = Violation(
            law=Law.EXCLUSIVITY,
            rows=12280,
            detail="claimed by more than one owner",
            sample=(2, 3, 4),
            kind=EXCLUSIVITY_DOUBLED,
        )
        rt = self._runtime(_Sched(draft_size=4096))
        with self.assertNoLogs(
            "sglang.srt.managers.phase_flip_runtime", level="WARNING"
        ):
            rt._note_unenumerated_owner([doubled])

    def test_the_probe_never_raises(self):
        """An instrument that can kill a seam is not an instrument."""
        rt = self._runtime(None)
        rt._note_unenumerated_owner([self._unowned()])
        rt._note_unenumerated_owner(None)


if __name__ == "__main__":
    unittest.main()
