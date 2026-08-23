# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#770/#812 -- an under-backed rank is NAMED, and never silently binds the group.

Specimen: boot_816_core_0823_0608.log 06:32:05, all three ranks. The live set
is genuinely replicated under PP (a request's tokens occupy KV on every stage),
so the floor is the SAME on every rank -- while the caps are unequal by design:

    PP0  backed 212992  floor 128549  ->  60.4% of its own cap
    PP1  backed 124928  floor 128549  -> 102.9%   <- the defect
    PP2  backed 133120  floor 128549  ->  96.6%

PP1's 102.9% clamps to 100% in `_floor_ppm`, the group MAX takes that, and
nobody shrinks -- vetoing PP0's fully fundable 84443-row plan on the rank that
needed it.

What this file pins is the DETECTION and the fact that the floor itself is NOT
lowered to make the arithmetic tidy: see TestFloorRowsDetectsButDoesNotLower.

Hermetic: no CUDA, no pool, no collectives.
"""

import unittest

from sglang.srt.managers.kv_backing_relief import (
    _SHRINK_SCALE,
    _floor_ppm,
    _shrink_ppm,
    floor_exceeds_local_cap,
)

# The specimen, verbatim.
PP_BACKED = (212992, 124928, 133120)
SPEC_FLOOR = 128549
PP0_SLACK_ROWS = 84443


class TestTheSpecimenNumbers(unittest.TestCase):
    """Pin the arithmetic before anything leans on it."""

    def test_pp1_floor_is_above_its_own_cap(self):
        self.assertGreater(SPEC_FLOOR, PP_BACKED[1])
        pct = 100.0 * SPEC_FLOOR / PP_BACKED[1]
        self.assertAlmostEqual(pct, 102.9, places=1)

    def test_the_other_two_ranks_are_under_their_caps(self):
        self.assertLess(SPEC_FLOOR, PP_BACKED[0])
        self.assertLess(SPEC_FLOOR, PP_BACKED[2])

    def test_pp0_slack_is_what_the_log_said(self):
        self.assertEqual(PP_BACKED[0] - SPEC_FLOOR, PP0_SLACK_ROWS)


class TestFloorExceedsLocalCap(unittest.TestCase):
    """The DEFECT case must be distinguishable from the HEALTHY one.

    Both used to return _SHRINK_SCALE from _floor_ppm and were therefore
    indistinguishable to every caller -- which is how an under-backed rank's
    local problem became a group-wide freeze.
    """

    def test_pp1_is_flagged(self):
        self.assertTrue(floor_exceeds_local_cap(SPEC_FLOOR, PP_BACKED[1]))

    def test_the_healthy_ranks_are_not_flagged(self):
        self.assertFalse(floor_exceeds_local_cap(SPEC_FLOOR, PP_BACKED[0]))
        self.assertFalse(floor_exceeds_local_cap(SPEC_FLOOR, PP_BACKED[2]))

    def test_exactly_at_the_cap_is_HEALTHY_not_a_defect(self):
        """floor == cap is a full pool, not an under-backed one. The whole
        point of the split is that these two are different."""
        self.assertFalse(floor_exceeds_local_cap(124928, 124928))

    def test_an_unknown_cap_is_never_a_defect_report(self):
        self.assertFalse(floor_exceeds_local_cap(128549, 0))
        self.assertFalse(floor_exceeds_local_cap(128549, -1))


class TestFloorPpmNeverExceedsFullScale(unittest.TestCase):
    """A proportion above 100% would ask a peer to grow, which is not what a
    floor means."""

    def test_pp1_proportion_is_capped_at_full_scale(self):
        self.assertEqual(_floor_ppm(SPEC_FLOOR, PP_BACKED[1]), _SHRINK_SCALE)

    def test_pp0_proportion_is_a_real_fraction(self):
        ppm = _floor_ppm(SPEC_FLOOR, PP_BACKED[0])
        self.assertLess(ppm, _SHRINK_SCALE)
        # 128549/212992 = 60.4%
        self.assertAlmostEqual(ppm / _SHRINK_SCALE, 0.604, places=3)

    def test_no_input_can_produce_more_than_full_scale(self):
        for floor in (0, 1, 124927, 124928, 128549, 10**9):
            self.assertLessEqual(_floor_ppm(floor, PP_BACKED[1]), _SHRINK_SCALE)

    def test_shrink_ppm_neutral_element_is_unchanged(self):
        """The sibling function's contract must not drift while I edit next
        to it: 'no change' is exactly the neutral element of the MIN."""
        self.assertEqual(_shrink_ppm(999999, 124928), _SHRINK_SCALE)
        self.assertEqual(_shrink_ppm(124928, 124928), _SHRINK_SCALE)


class TestFloorRowsDetectsButDoesNotLower(unittest.TestCase):
    """The floor is DETECTED as under-backed and deliberately NOT lowered.

    A first version clamped it down to the cap. That is wrong in the dangerous
    direction: the floor is `live set + 1 + margin + reserve`, so lowering it
    to a smaller cap authorises a cap BELOW rows that are still in use.
    ``test_residency_cap_flip_levelling_792::TheLevellingMustNotCapBelowTheLiveSet``
    failed on the clamp and passes without it -- the tree already carried the
    invariant, and it is older than this ticket.

    A real rung needs a pool and an allocator, so the method is exercised
    UNBOUND against a stub, the technique the #717 suites already use.
    """

    def _rung(self, cap, page=1, margin=0, reserve=512):
        from sglang.srt.managers.kv_backing_relief import KvBackingRelief

        class _Stub:
            _margin_rows = margin
            _admission_reserve_rows = reserve

            class _pool:
                page_size = page

            def _current_rows(self):
                return cap

        return KvBackingRelief._floor_rows, _Stub()

    def test_an_under_backed_rank_keeps_its_floor(self):
        """THE LOAD-BEARING CASE, and the assert is the opposite of my first
        draft: the floor must SURVIVE, because the live set is still live."""
        fn, stub = self._rung(cap=PP_BACKED[1])
        max_live = SPEC_FLOOR - 1 - 0 - 512
        floor = fn(stub, max_live)
        self.assertEqual(floor, SPEC_FLOOR)
        self.assertGreater(floor, PP_BACKED[1])
        # ...and it is still recognisable as the defect it is.
        self.assertTrue(floor_exceeds_local_cap(floor, PP_BACKED[1]))

    def test_a_roomy_rank_is_unaffected(self):
        fn, stub = self._rung(cap=PP_BACKED[0])
        max_live = SPEC_FLOOR - 1 - 0 - 512
        floor = fn(stub, max_live)
        self.assertEqual(floor, SPEC_FLOOR)
        self.assertFalse(floor_exceeds_local_cap(floor, PP_BACKED[0]))

    def test_an_unreadable_cap_leaves_the_floor_alone(self):
        from sglang.srt.managers.kv_backing_relief import KvBackingRelief

        class _Broken:
            _margin_rows = 0
            _admission_reserve_rows = 512

            class _pool:
                page_size = 1

            def _current_rows(self):
                raise RuntimeError("no pool")

        self.assertEqual(
            KvBackingRelief._floor_rows(_Broken(), SPEC_FLOOR - 513), SPEC_FLOOR
        )

    def test_page_alignment_is_preserved(self):
        cap = 124900
        self.assertNotEqual(cap % 64, 0, "fixture must be non-divisible")
        fn, stub = self._rung(cap=cap, page=64)
        floor = fn(stub, SPEC_FLOOR)
        self.assertEqual(floor % 64, 0)
        self.assertGreaterEqual(floor, SPEC_FLOOR)

    def test_the_floor_never_drops_below_the_live_set(self):
        """The invariant 792 protects, asserted here directly so this file
        cannot drift back toward the clamp."""
        for cap in (1, 1000, PP_BACKED[1], PP_BACKED[0]):
            fn, stub = self._rung(cap=cap)
            max_live = 128036
            self.assertGreater(fn(stub, max_live), max_live)


if __name__ == "__main__":
    unittest.main()
