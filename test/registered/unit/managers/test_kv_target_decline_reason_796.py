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
"""#796: the seam's funder must work on a fleet whose pools are UNEVEN.

MEASURED ON METAL, boot_798_0822_0629.log under live load, 114 identical
declines and this decline reason only::

    DECLINED because no rank asked to go below the group's smallest cap
    (deepest desire 112640 rows against cap 112640)

The per-rank terms behind it, with the pools DIFFERENT SIZES by a factor of
1.8 -- which is not a fault but this fork's whole design (uneven TP shards,
uneven DCP tokens, the KV ratio)::

    PP0 cur=204800 floor=19529 slack=185271 deficit=+1576 MiB -> SHRINK to 154376
    PP1 cur=135168 floor=19529 slack=115639 deficit=  +36 MiB -> SHRINK to 126976
    PP2 cur=112640 floor=19529 slack= 93111 deficit=  -55 MiB -> no change

PP0 was 295 MiB short of seam staging while its own rung offered 1576 MiB --
five times what was needed -- out of 185271 rows of slack above a floor of
19529. Nothing refused it. The shrink was simply not REPRESENTABLE.

TWO FAULTS, STACKED, and repairing either alone changes nothing:

1. ``propose`` encoded "no change" as ``desire = current``, an absolute row
   id. PP2 needed nothing and proposed 112640; because its pool is the
   SMALLEST, that was the smallest number in the group and won the MIN which
   this module documents as "the most-pressed rank sets the ambition". It did
   the opposite -- the least-pressed rank with the smallest pool set it.
2. Even repaired, PP0's genuine ambition of 154376 lies ABOVE PP2's entire
   pool, so it can never fall below ``min_current``, and the decline fires
   again for a different reason. One absolute row id cannot express a shrink
   across pools of different sizes.

THE NATURAL EXPERIMENT that isolates it, and the falsifier any fix must keep
passing: the same boot GRANTED three times, and every grant is in the round
where all three ranks report ``cur=450560`` -- the EVEN-pool layout. Even
pools: the machinery works. Uneven pools: 114 declines. So the fix must keep
the even-pool grant AND grant PP0's shrink.

Hermetic: no CUDA, no distributed, no scheduler. The reduction is a plain
element-wise MIN over four-field proposals, which is exactly the contract
``default_collective_min`` implements.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers import kv_backing_relief as kbr
from sglang.srt.managers.kv_backing_relief import (
    _floor_ppm,
    _rows_for_ppm,
    _shrink_ppm,
)


def _min_reduce(proposals):
    """Element-wise MIN, the contract ``default_collective_min`` implements."""
    return [min(fields) for fields in zip(*proposals)]


def _proposal(*, desire: int, floor: int, current: int):
    """One rank's four-field proposal, as :meth:`KvBackingRelief.propose` builds it.

    #796: the first two fields are PARTS PER MILLION of this rank's own cap.
    Rows are a rank's private unit on an uneven fleet; a proportion is the same
    statement on every rank.
    """
    return (
        _shrink_ppm(desire, current),
        -_floor_ppm(floor, current),
        int(current),
        -int(current),
    )


# The metal numbers, verbatim, from 06:35:28Z.
_PP0 = _proposal(desire=154376, floor=19529, current=204800)
_PP1 = _proposal(desire=126976, floor=19529, current=135168)
_PP2 = _proposal(desire=112640, floor=19529, current=112640)  # no change

# The even-pool pp_to_tp round from the same boot, which already worked.
_EVEN = [
    _proposal(desire=450560, floor=4219, current=450560),
    _proposal(desire=450560, floor=4219, current=450560),
    _proposal(desire=402344, floor=4219, current=450560),
]


class TestTheUnevenFleetIsFundable(unittest.TestCase):
    def test_the_metal_shape_now_grants(self):
        """The regression itself. Red before the currency change: this declined."""
        reduced = _min_reduce([_PP0, _PP1, _PP2])
        ppm = kbr.collective_kv_shrink_ppm(reduced[:4])
        self.assertIsNotNone(
            ppm,
            "PP0 offered 1576 MiB against a 295 MiB shortfall out of 185271 "
            "rows of slack; a group that cannot express that is the defect",
        )
        self.assertEqual(
            _rows_for_ppm(ppm, 204800),
            154376,
            "the pressed rank must reach the row target it actually asked for",
        )

    def test_a_peer_that_needs_nothing_does_not_set_the_ambition(self):
        """PP2's 'no change' must be neutral, not the smallest number in the group."""
        self.assertEqual(_PP2[0], 1_000_000, "no change is the neutral element")
        with_peer = kbr.collective_kv_shrink_ppm(_min_reduce([_PP0, _PP1, _PP2])[:4])
        alone = kbr.collective_kv_shrink_ppm(_min_reduce([_PP0])[:4])
        self.assertEqual(
            with_peer,
            alone,
            "a rank asking for nothing must not change the group's decision",
        )

    def test_each_rank_converts_against_its_own_cap(self):
        """One proportion, three different row counts. That is the whole point."""
        ppm = kbr.collective_kv_shrink_ppm(_min_reduce([_PP0, _PP1, _PP2])[:4])
        rows = [_rows_for_ppm(ppm, c) for c in (204800, 135168, 112640)]
        self.assertEqual(len(set(rows)), 3, f"uneven pools, uneven targets: {rows}")
        for got, cap in zip(rows, (204800, 135168, 112640)):
            self.assertLess(got, cap, "every rank actually shrinks")
            self.assertGreater(got, 19529, "and none is driven below its floor")

    def test_the_even_pool_grant_still_works(self):
        """Strand 8's falsifier: the pp_to_tp round that already worked must keep working."""
        ppm = kbr.collective_kv_shrink_ppm(_min_reduce(_EVEN)[:4])
        self.assertIsNotNone(ppm)
        self.assertEqual(_rows_for_ppm(ppm, 450560), 402344)

    def test_a_calm_group_still_declines(self):
        """No ambition anywhere means no shrink. The tier law must keep working."""
        calm = _min_reduce(
            [
                _proposal(desire=204800, floor=19529, current=204800),
                _proposal(desire=112640, floor=19529, current=112640),
            ]
        )
        self.assertIsNone(kbr.collective_kv_shrink_ppm(calm[:4]))

    def test_an_abstention_still_cancels_for_everyone(self):
        """One rank that cannot take part still stops the group. Unchanged law."""
        reduced = _min_reduce([_PP0, kbr.ABSTAIN])
        self.assertIsNone(kbr.collective_kv_shrink_ppm(reduced[:4]))
        self.assertIn("abstain", kbr.explain_kv_target(reduced[:4]).lower())

    def test_a_floor_still_raises_the_target(self):
        """The limit still wins; it is simply expressed as a proportion now."""
        reduced = _min_reduce(
            [
                _proposal(desire=20000, floor=19529, current=204800),
                _proposal(desire=112640, floor=100000, current=112640),
            ]
        )
        ppm = kbr.collective_kv_shrink_ppm(reduced[:4])
        self.assertIsNotNone(ppm)
        self.assertGreaterEqual(
            _rows_for_ppm(ppm, 112640),
            100000,
            "the peer's floor must not be unmapped through",
        )


class TestTheDeclineStillExplainsItself(unittest.TestCase):
    def test_a_grant_names_the_proportion(self):
        why = kbr.explain_kv_target(_min_reduce([_PP0, _PP1, _PP2])[:4])
        self.assertIn("GRANTED", why)
        self.assertIn("%", why, f"the currency is a proportion now; got {why!r}")

    def test_a_calm_decline_names_the_tier_law(self):
        calm = _min_reduce([_proposal(desire=204800, floor=19529, current=204800)])
        why = kbr.explain_kv_target(calm[:4])
        self.assertIn("tier law", why)


if __name__ == "__main__":
    unittest.main()
