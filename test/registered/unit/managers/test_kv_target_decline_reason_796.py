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
"""#796: the group's shrink decision must say WHY it declined.

THE DEFECT THIS PINS, measured on metal 2026-08-22 (boot_798_0822_0543.log).
Three PP ranks, one instance. PP0 is short of seam staging and its rung has a
fundable plan -- ``current=204800 rows, floor=116737, slack=89119,
deficit=+1740 MiB -> SHRINK to 149126``. The shrink never happens. Across the
whole boot there is not one occurrence of ``runtime_set_backing_rows``, of
``the eviction did not deliver the mark``, or of ``ABSTAIN on device``:
:func:`kv_backing_relief.collective_kv_target` returned ``None`` and
``apply_target`` was never called at all.

It returned None because ``target = max(desire, max_floor)`` must clear EVERY
rank's floor, and PP2 -- a rank under no memory pressure whatsoever, sitting
on 2693 MiB spendable -- reported ``fundable_bytes() == 0`` on all eight of
its logged asks. ``fundable_bytes`` is ``max(0, current - floor)``, so PP2's
floor was at or above its own cap, which is at or above ``min_current``. One
rank that needs no shrink therefore vetoes the shrink for the rank that does,
and PP0's 89119 rows of slack stay unreachable while the flip abandons eight
times and phase purity yields.

THAT DECLINE IS ENTIRELY SILENT. ``collective_kv_target`` returns a bare
``Optional[int]``; the caller logs "returned NOTHING" and names three possible
causes without saying which one held. The vetoing rank's floor is computed on
every rank, retained in ``_last_proposal_terms``, and printed nowhere -- a
rank that "fits" never reports its terms. So the one number that decided the
outcome for the whole group is the one number no log carries.

These tests are hermetic: no CUDA, no distributed, no scheduler. The reduction
is a plain element-wise MIN over four-field proposals, which is exactly the
contract ``default_collective_min`` implements.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers import kv_backing_relief as kbr


def _min_reduce(proposals):
    """Element-wise MIN, the contract ``default_collective_min`` implements."""
    return [min(fields) for fields in zip(*proposals)]


def _proposal(*, desire: int, floor: int, current: int):
    """One rank's four-field proposal, as :meth:`KvBackingRelief.propose` builds it."""
    return (int(desire), -int(floor), int(current), -int(current))


#: The metal numbers from boot_798_0822_0543.log, arm 7 at 05:54:54. PP0's
#: terms are logged verbatim; PP1's and PP2's floors are the ONLY unlogged
#: terms, reconstructed from their ``fundable_bytes()`` asks (PP1 "can return
#: 608 MiB", PP2 "can return 0 MiB") against the shared 204800-row cap.
_PP0 = _proposal(desire=149126, floor=115681, current=204800)
_PP1 = _proposal(desire=204800, floor=185000, current=204800)
_PP2 = _proposal(desire=204800, floor=204800, current=204800)


class TestPeerFloorVetoIsReported(unittest.TestCase):
    def test_the_metal_shape_declines(self):
        """The regression itself: a fundable plan on PP0, no shrink for anyone."""
        reduced = _min_reduce([_PP0, _PP1, _PP2])
        self.assertIsNone(
            kbr.collective_kv_target(reduced[:4]),
            "the metal shape must still decline -- this test pins the REASON, "
            "not a behaviour change",
        )

    def test_decline_names_all_three_terms(self):
        """A reader must get desire, max_floor and min_current, not a bare None."""
        reduced = _min_reduce([_PP0, _PP1, _PP2])
        why = kbr.explain_kv_target(reduced[:4])
        self.assertTrue(why, "a decline must explain itself")
        for term in ("149126", "204800"):
            self.assertIn(
                term,
                why,
                f"the decline must name the term {term} that decided it; got {why!r}",
            )

    def test_decline_identifies_the_peer_floor_as_the_binding_term(self):
        """The fix direction differs per cause, so the cause must be named.

        A peer-floor veto is answered by lowering that peer's floor (evicting
        its recomputable prefix). An abstention is answered by repairing the
        abstaining rank. Reporting "declined" for both sends the reader to the
        wrong half of the mechanism, which is what cost this chain its shifts.
        """
        reduced = _min_reduce([_PP0, _PP1, _PP2])
        why = kbr.explain_kv_target(reduced[:4]).lower()
        self.assertIn("floor", why)
        self.assertNotIn(
            "abstain",
            why,
            "no rank abstained in the metal shape; calling this an abstention "
            "sends the reader to the wrong defect",
        )

    def test_abstention_is_named_as_an_abstention(self):
        """The other decline path must be distinguishable from the floor veto."""
        reduced = _min_reduce([_PP0, kbr.ABSTAIN])
        self.assertIsNone(kbr.collective_kv_target(reduced[:4]))
        why = kbr.explain_kv_target(reduced[:4]).lower()
        self.assertIn("abstain", why)

    def test_a_granted_target_explains_itself_too(self):
        """A shrink that DOES happen names the target and the rank that set it."""
        healthy = _min_reduce(
            [
                _proposal(desire=149126, floor=115681, current=204800),
                _proposal(desire=204800, floor=120000, current=204800),
            ]
        )
        self.assertEqual(kbr.collective_kv_target(healthy[:4]), 149126)
        why = kbr.explain_kv_target(healthy[:4])
        self.assertIn("149126", why)

    def test_floor_veto_is_reported_even_when_it_only_raises_the_target(self):
        """A partial veto is the same defect one degree weaker, and it is silent too.

        Here a peer's floor does not cancel the shrink, it merely lifts the
        target above what the pressed rank asked for -- so the pressed rank
        wins less than it priced and nothing says why.
        """
        partial = _min_reduce(
            [
                _proposal(desire=149126, floor=115681, current=204800),
                _proposal(desire=204800, floor=180000, current=204800),
            ]
        )
        self.assertEqual(kbr.collective_kv_target(partial[:4]), 180000)
        why = kbr.explain_kv_target(partial[:4])
        self.assertIn("180000", why, f"the raising floor must be named; got {why!r}")


if __name__ == "__main__":
    unittest.main()
