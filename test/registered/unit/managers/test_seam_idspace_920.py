"""#920: the cutover copied resident KV out of a pool that does not store it.

Every number here is measured on this rig; none is illustrative.

THE SPECIMEN
============

Boot 2c, 2026-08-27 00:45:53Z, a ``tp_to_pp`` cutover, inside
``_release_residents_for_cutover`` -- past the seam's no-return point, so
there is no abort left and the rank dies::

    /spinning/evidence-665-f1/boot_accept2c0827_0827_0029.log:75388  (PP1)
      ValueError: HiCache CPU copy (offload): row 144956 does not address
        this pool, which has 140588 rows.
    /spinning/evidence-665-f1/boot_accept2c0827_0827_0029.log:75449  (PP2)
      ValueError: HiCache CPU copy (offload): row 148793 does not address
        this pool, which has 147894 rows.

PP0 did not die. That is luck: its own TP pool is 228260 rows and the ids in
play stayed under it.

THE ROOT, DERIVED FROM THE TREE AND NOT GUESSED
===============================================

The three row counts above are RANK-LOCAL, and the same boot prints all three
against ONE shared id space of 467565 (:1153, :1154, :1173, the seam sizer's
``src.num_rows`` = the DIRECTION'S SOURCE pool,
``phase_flip_seam_reserve.py:1489``)::

    rank 0  pp_to_tp [holds 483951 rows]   tp_to_pp [holds 228260 rows]
    rank 1  pp_to_tp [holds 483951 rows]   tp_to_pp [holds 140588 rows]
    rank 2  pp_to_tp [holds 483951 rows]   tp_to_pp [holds 147894 rows]

The PP-side pool spans the whole id space on every rank; the TP-side pool is
roughly ``T/3`` of it, per rank, and the sizer says why in as many words:
"under the TP layout that is its token SHARE of the id space"
(``phase_flip_seam_reserve.py:1515-1518``).

``req_to_token`` holds GLOBAL slot ids in BOTH layouts. Under PP that IS the
pool row -- ``local_pp = my_slots.clone()  # PP row of slot L is L itself
(dcp_size=1)`` (``layers/dcp/phase_flip_plan.py:573``). Under TP it is not:
``local_tp = rows_of(my_slots, vec, rank)`` (:574), the weighted compaction
``(L // S) * ratio + (L % S - lo)`` (``layers/dcp/reshard_plan.py:155-166``),
which is the same expression the hot read path applies per access
(``dcp_weighted_read_slots``, ``layers/dcp/owner.py:440``, whose docstring
names its input as "a flat list of GLOBAL cache slots (typically the
``req_to_token`` rows of a paged read)").

``Req.offload_kv_cache`` (``schedule_batch.py:1789-1793``) hands the raw
``req_to_token`` slice to ``get_cpu_copy`` with no compaction, so under the TP
layout it addresses the wrong pool rows. TWO SIGNS, and the fatal one is the
lesser: an id ABOVE the rank's row count is the ``ValueError`` above; an id
BELOW it is a different row's KV copied out under this request's name -- no
crash, wrong prefix. A bounds check on the ids would pass exactly the silent
half, which is why the fix asks about the LAYOUT.

WHAT THE FIX IS
===============

``seam_copy_addresses_the_bound_pool(scheduler)`` (phase_flip_runtime.py) and
one call to it in ``build_cutover_release``, which is the only site in the
tree that turns ``copy_state`` on. Two independent readings, both must say yes:

  1. the resident layout is one in which a ``req_to_token`` entry IS a pool row
     (``SEAM_COPY_GLOBAL_ROW_LAYOUTS``);
  2. the bound pool physically addresses the allocator's whole id space --
     arithmetic, layout-agnostic, and abstaining on a paged lane where the two
     numbers are not the same quantity.

``check_cpu_copy_rows``'s refusal (``mem_cache/memory_pool.py:753``) is NOT
touched and stays fatal for anything that still reaches it. This keeps the
seam from being that path; it does not soften the guard.
"""

import unittest

from sglang.srt.managers.phase_flip_runtime import (
    PHASE_PP,
    PHASE_TP,
    SEAM_COPY_GLOBAL_ROW_LAYOUTS,
    build_cutover_release,
    seam_copy_addresses_the_bound_pool,
)
from sglang.test.test_utils import CustomTestCase

#: (label, layout tag, pool rows, allocator id space), measured 2026-08-27.
#: The PP row count is the boot's ``pp_to_tp [this rank holds ...]`` reading,
#: the TP row counts are the three ``tp_to_pp`` ones.
SPECIMENS_920 = (
    ("PP1 tp_to_pp (died, row 144956)", PHASE_TP, 140588, 467565),
    ("PP2 tp_to_pp (died, row 148793)", PHASE_TP, 147894, 467565),
    ("PP0 tp_to_pp (survived on luck)", PHASE_TP, 228260, 467565),
    ("any rank pp_to_tp (lawful)", PHASE_PP, 483951, 467565),
)


class _Pool:
    def __init__(self, rows):
        self.store_bound_rows = rows


class _Alloc:
    def __init__(self, size, rows, page_size=1):
        self.size = size
        self.page_size = page_size
        self._kvcache = _Pool(rows)


class _Tree:
    def reset(self):
        return 0

    def all_values_flatten(self):  # pragma: no cover - not exercised here
        raise AssertionError("the cutover release must not enumerate the tree")


class _Scheduler:
    def __init__(self, layout, rows, size, page_size=1):
        self.phase_flip_active_stack = layout
        self.token_to_kv_pool_allocator = _Alloc(size, rows, page_size)
        self.tree_cache = _Tree()
        self.server_args = object()
        self.req_to_token_pool = object()
        self.hisparse_coordinator = None
        self.phase_flip_runtime = None


class TestSpecimens920(CustomTestCase):
    """The predicate, at exactly the four measured configurations."""

    def test_the_two_deaths_and_the_one_lucky_survivor_are_all_refused(self):
        for label, layout, rows, size in SPECIMENS_920[:3]:
            with self.subTest(specimen=label):
                sched = _Scheduler(layout, rows, size)
                self.assertFalse(
                    seam_copy_addresses_the_bound_pool(sched),
                    f"{label}: the copy must be refused",
                )

    def test_the_lucky_survivor_is_refused_for_the_same_reason_as_the_deaths(
        self,
    ):
        """PP0 lived only because 228260 > the ids that happened to be live.

        A fix keyed on "did it crash" would leave PP0 taking a copy whose rows
        are silently the wrong ones -- the sign that does not raise. Pinned
        separately because it is the case a bounds check would get wrong.
        """
        _, layout, rows, size = SPECIMENS_920[2]
        sched = _Scheduler(layout, rows, size)
        self.assertFalse(seam_copy_addresses_the_bound_pool(sched))
        # And it would ALSO be refused with a pool big enough to address every
        # id -- reading 1 alone decides it, which is the whole point.
        sched_wide = _Scheduler(layout, size + 1, size)
        self.assertFalse(
            seam_copy_addresses_the_bound_pool(sched_wide),
            "a TP-layout pool that happens to span the id space is still "
            "indexed by compacted rows; the copy stays unlawful",
        )

    def test_the_pp_layout_stays_lawful(self):
        label, layout, rows, size = SPECIMENS_920[3]
        sched = _Scheduler(layout, rows, size)
        self.assertTrue(
            seam_copy_addresses_the_bound_pool(sched),
            f"{label}: this direction's copy has never been the defect and "
            "must not be disabled by the fix",
        )


class TestBothReadingsAreNecessary920(CustomTestCase):
    """Neither reading alone covers the specimens; each has its own case."""

    def test_reading_two_alone_catches_the_measured_deaths(self):
        """The arithmetic half, exercised where the layout half is silent."""
        for label, _layout, rows, size in SPECIMENS_920[:2]:
            with self.subTest(specimen=label):
                # Same pool arithmetic, but presented under the layout the
                # first reading accepts: the second reading must still refuse.
                sched = _Scheduler(PHASE_PP, rows, size)
                self.assertFalse(
                    seam_copy_addresses_the_bound_pool(sched),
                    f"{label}: a pool that cannot address the id space must "
                    "be refused whatever the layout tag says",
                )

    def test_reading_one_alone_catches_what_arithmetic_cannot(self):
        """The silent sign: rows >= size, compaction still applies."""
        sched = _Scheduler(PHASE_TP, 999999, 467565)
        self.assertFalse(seam_copy_addresses_the_bound_pool(sched))

    def test_an_unreadable_layout_is_refused_not_assumed(self):
        sched = _Scheduler(None, 483951, 467565)
        self.assertFalse(
            seam_copy_addresses_the_bound_pool(sched),
            "LAYOUT_TAG_UNKNOWN means 'cannot rule it out'; for a copy that "
            "can kill the rank the safe direction is to decline",
        )
        self.assertFalse(seam_copy_addresses_the_bound_pool(None))

    def test_a_paged_lane_abstains_instead_of_guessing(self):
        """A paged pool's leading dimension counts PAGES, not tokens.

        Comparing it to an allocator ``size`` in tokens would decline a lane
        that addresses every id it can hand out, so reading 2 abstains there
        and reading 1 decides alone.
        """
        sched = _Scheduler(PHASE_PP, 1024, 65536, page_size=64)
        self.assertTrue(seam_copy_addresses_the_bound_pool(sched))
        sched_tp = _Scheduler(PHASE_TP, 1024, 65536, page_size=64)
        self.assertFalse(
            seam_copy_addresses_the_bound_pool(sched_tp),
            "abstaining on the arithmetic must not also abstain on the layout",
        )

    def test_an_unreadable_pool_does_not_manufacture_a_refusal(self):
        """``None`` rows is 'unanswerable', never 'a pool of zero rows'."""
        sched = _Scheduler(PHASE_PP, 483951, 467565)
        sched.token_to_kv_pool_allocator._kvcache = object()
        self.assertTrue(seam_copy_addresses_the_bound_pool(sched))
        sched.token_to_kv_pool_allocator = None
        self.assertFalse(
            seam_copy_addresses_the_bound_pool(sched),
            "no allocator at all is a different state from an unreadable pool",
        )


class TestLayoutListIsExhaustiveByIntent920(CustomTestCase):
    """Zukunfts-Check: a new layout must be classified, not defaulted in."""

    def test_only_pp_is_declared_to_carry_global_rows(self):
        self.assertEqual(SEAM_COPY_GLOBAL_ROW_LAYOUTS, (PHASE_PP,))

    def test_an_unlisted_layout_is_refused(self):
        sched = _Scheduler("some-future-layout", 999999, 467565)
        self.assertFalse(seam_copy_addresses_the_bound_pool(sched))


class TestTheSeamActuallyStopsCopying920(CustomTestCase):
    """Execution-proven wiring, not ``inspect.getsource``.

    ``build_cutover_release`` is the one site that sets ``copy_state``. This
    drives its real retract closure and reads the keyword it actually passed
    to ``retract_all``.
    """

    @staticmethod
    def _copy_state_passed(sched):
        import sglang.srt.managers.schedule_batch as sb

        seen = {}
        original = sb.retract_all

        def _spy(**kwargs):
            seen.update(kwargs)
            return list(kwargs.get("reqs") or ())

        sb.retract_all = _spy
        try:
            built = build_cutover_release(sched)
            assert built is not None, "the harness must supply a resettable tree"
            retract, _reset = built
            retract([object()])
        finally:
            sb.retract_all = original
        return seen

    def test_tp_layout_retracts_without_copying(self):
        sched = _Scheduler(PHASE_TP, 140588, 467565)
        seen = self._copy_state_passed(sched)
        self.assertIs(
            seen["copy_state"],
            False,
            "the measured PP1 configuration must retract with no host copy -- "
            "that copy is what reached check_cpu_copy_rows and killed the rank",
        )
        self.assertIs(
            seen["offload_kv"],
            False,
            "the pre-existing decode-disagg skip must be unchanged",
        )

    def test_pp_layout_still_copies(self):
        sched = _Scheduler(PHASE_PP, 483951, 467565)
        seen = self._copy_state_passed(sched)
        self.assertIs(
            seen["copy_state"],
            True,
            "the fix must not disable the direction that was never the defect",
        )

    def test_mutant_a_constant_true_reproduces_the_death_configuration(self):
        """If ``copy_state`` went back to a constant, this is what returns.

        The mutant is not applied to the product: it is stated as the
        arithmetic the product now avoids, at the measured numbers, so a
        reviewer can see the test would go red on a revert rather than take
        that on trust.
        """
        _, _layout, rows, size = SPECIMENS_920[0]
        sched = _Scheduler(PHASE_TP, rows, size)
        self.assertFalse(seam_copy_addresses_the_bound_pool(sched))
        # The id that killed PP1, against the pool the copy would have indexed.
        self.assertGreater(
            144956,
            rows,
            "row 144956 is outside the 140588-row pool -- the ValueError",
        )
        seen = self._copy_state_passed(sched)
        self.assertIsNot(seen["copy_state"], True)


if __name__ == "__main__":
    unittest.main()
