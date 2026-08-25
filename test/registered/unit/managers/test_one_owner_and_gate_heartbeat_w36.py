"""W36: two owners for one row, and a counter that could not say "checked".

PART 1 -- THE 22-ROW DOUBLE-COUNT. Measured on metal, all three ranks:

    ValueError: pool memory leak detected! [full] total=468981,
      available=108565, evictable=1, protected=0, session_held=0,
      uncached=0, withheld=360437          -> 469003, TWENTY-TWO OVER

The census names it exactly one line earlier:

    size=468981 free=108544 withheld=360437 available=108566

`withheld + free == size` EXACTLY -- the id space is fully owned -- while
`available` is `free + 22`, because `available_size()` is
`len(free_pages) + len(release_pages)` and 22 ids had re-entered the RELEASE
buffer while `_withheld` still counted them. Two owners, one row.

W29 was the SAME detector with the opposite sign (one row, zero owners). Both
signs prove it measures, which is why it is never softened.

WHICH OWNER WINS, decided by reading: the ids are ABOVE THE CAP, and `_apply`'s
own docstring is "move ids above the cap out of every free list". The withhold
is authoritative; the free list is the stale owner.

PART 2 -- RUNG 3. Every stale-generation gate logged only on REFUSAL, so W36's
eight cutovers with zero refusals were indistinguishable from eight cutovers
whose gates were never reached. The rung was lost to an ambiguity created by
the very lines meant to detect it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.srt.managers.cache_controller import gate_heartbeat
from sglang.test.test_utils import CustomTestCase


class _Alloc:
    page_size = 1

    def __init__(self, free, release):
        self.free_pages = torch.tensor(free, dtype=torch.int64)
        self.release_pages = torch.tensor(release, dtype=torch.int64)
        self.residency_withheld_slots = 0

    def available_size(self):
        return len(self.free_pages) + len(self.release_pages)


def _cap_with(alloc, withheld):
    from sglang.srt.managers.kv_backing_relief import KvRowCap

    cap = KvRowCap.__new__(KvRowCap)
    cap._alloc = alloc
    cap._cap = 100
    cap._withheld = torch.tensor(withheld, dtype=torch.int64)
    return cap


class TestTheSpecimenArithmetic(CustomTestCase):
    def test_a_row_in_both_is_removed_from_the_free_side(self):
        # 101..104 are above the cap and withheld; two of them also sit in the
        # release buffer. The free side is the stale owner.
        alloc = _Alloc(free=[1, 2, 3], release=[101, 102])
        cap = _cap_with(alloc, [101, 102, 103, 104])
        cap._publish()
        self.assertEqual(alloc.release_pages.tolist(), [])
        self.assertEqual(cap.withheld, 4, "the withhold keeps them")

    def test_the_identity_holds_afterwards(self):
        # The invariant the checker enforces: available never overlaps withheld.
        alloc = _Alloc(free=[1, 2, 3], release=[101, 102])
        cap = _cap_with(alloc, [101, 102, 103, 104])
        cap._publish()
        held = set(cap._withheld.tolist())
        avail = set(alloc.free_pages.tolist()) | set(alloc.release_pages.tolist())
        self.assertEqual(held & avail, set(), "one row, one owner")

    def test_without_the_settle_the_overlap_is_what_the_checker_sees(self):
        # RED-FIRST, modelled on the specimen: 22 rows in both buckets make
        # available exceed the id space by exactly 22.
        size = 1000
        withheld = list(range(size - 400, size))  # 400 held
        free = list(range(0, size - 400))  # the rest free
        release = withheld[:22]  # 22 double-owned
        alloc = _Alloc(free=free, release=release)
        counted = alloc.available_size() + len(withheld)
        self.assertEqual(counted, size + 22, "the W36 signature")

    def test_no_overlap_is_left_untouched(self):
        alloc = _Alloc(free=[1, 2], release=[3])
        cap = _cap_with(alloc, [101, 102])
        cap._publish()
        self.assertEqual(alloc.free_pages.tolist(), [1, 2])
        self.assertEqual(alloc.release_pages.tolist(), [3])
        self.assertEqual(cap.withheld, 2)

    def test_the_reclaim_is_counted_by_name(self):
        alloc = _Alloc(free=[], release=[101])
        cap = _cap_with(alloc, [101])
        cap._publish()
        self.assertEqual(getattr(cap, "_double_owned_reclaimed", 0), 1)


class TestTheCheckerIsNotSoftened(CustomTestCase):
    """It has now fired correctly in BOTH directions; softening it would blind
    the one instrument that has been right every time."""

    def _check(self, **kw):
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker,
        )

        return SchedulerInvariantChecker._check_pool_invariant("full", **kw)

    def test_the_w36_over_count_still_aborts(self):
        leak, _ = self._check(
            available=108565,
            evictable=1,
            protected=0,
            session_held=0,
            total=468981,
            uncached=0,
            withheld=360437,
        )
        self.assertTrue(leak, "22 rows OVER is a leak")

    def test_the_w29_under_count_still_aborts(self):
        leak, _ = self._check(
            available=107041,
            evictable=1,
            protected=0,
            session_held=0,
            total=469733,
            uncached=0,
            withheld=362690,
        )
        self.assertTrue(leak, "1 row SHORT is a leak")

    def test_a_balanced_pool_is_still_clean(self):
        leak, _ = self._check(
            available=108543,
            evictable=1,
            protected=0,
            session_held=0,
            total=468981,
            uncached=0,
            withheld=360437,
        )
        self.assertFalse(leak, "the checker must be able to pass")


class TestTheGateHeartbeat(CustomTestCase):
    def test_it_reports_both_numbers_and_resets(self):
        ctl = types.SimpleNamespace(_gate_checked=7, _gate_refused=2)
        self.assertEqual(gate_heartbeat(ctl), "checked=7 refused=2")
        self.assertEqual(gate_heartbeat(ctl), "checked=0 refused=0", "per epoch")

    def test_an_unreached_gate_reads_checked_zero(self):
        # THE CAN-FAIL. W36's gates were silent and the rung was inconclusive;
        # an unreachable gate must now be VISIBLE, not absent.
        self.assertEqual(gate_heartbeat(types.SimpleNamespace()), "checked=0 refused=0")

    def test_the_seam_emits_it_every_cutover(self):
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        src = inspect.getsource(PhaseFlipRuntime._release_residents_for_cutover)
        self.assertIn("gate_heartbeat", src)
        self.assertIn("STALE-GATE HEARTBEAT", src)

    def test_the_gates_count_checks_not_only_refusals(self):
        import inspect

        from sglang.srt.managers.cache_controller import (
            consume_gate,
            operation_is_stale,
        )

        for fn in (consume_gate, operation_is_stale):
            self.assertIn("_gate_checked", inspect.getsource(fn))


if __name__ == "__main__":
    unittest.main()
