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
"""The floor clamp silently undoes the granularity round-up.

MEASURED, boot_798_0822_0737.log: 15 shrinks reported
"0 MiB but the driver's free column did not move".

``_release_rows`` rounds the ask UP to one release granularity, because a
shrink smaller than one commit chunk per buffer clears no extent anywhere
(the 2026-08-11 measurement in that comment). The very next line then
clamps the target to the eviction floor:

    rows_wanted = max(rows_wanted, self._min_release_rows())
    target = max(floor, current - rows_wanted)

When the floor binds, the clamp gives back a target whose distance from
``current`` is BELOW one granule again -- and nothing re-checks it. The
shrink is attempted anyway: the cap engages, ``decommit_range`` clears no
extent, zero bytes are returned, and the rank has lost capacity for
nothing.

PP2's shape from that boot: current=126976, floor=88945, granule=229376.
Round-up asks 229376; the clamp yields 88945; the actual distance is
38031 rows, one sixth of a granule. Note the granule EXCEEDS the whole
pool, so on this rank no shrink can pay at all at this chunk size.

The fix is in the safe direction and cannot revive #717: it shrinks LESS
(not at all), so the cap is never lowered further toward the live set.
A shrink that cannot cross a granule must not engage the cap.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers import kv_backing_relief as kbr

# PP2's measured shape.
CURRENT = 126_976
FLOOR = 88_945
GRANULE = 229_376


class _Pool:
    backing_commit_chunk_bytes = 8 * 1024 * 1024

    def runtime_set_backing_rows(self, target):  # pragma: no cover - guarded
        raise AssertionError(
            "the cap must not be engaged for a shrink that cannot cross a "
            "release granule -- that costs capacity and returns nothing"
        )


class _Relief:
    """Only the surface the release-rows arithmetic touches."""

    def __init__(self):
        self._pool = _Pool()
        self.engaged = []

    def _min_release_rows(self):
        return GRANULE

    def _shrink_to(self, target, current):
        self.engaged.append((int(target), int(current)))
        return 0


class TestFloorClampDefeatsGranule(unittest.TestCase):
    def test_sub_granule_shrink_must_not_engage_the_cap(self):
        """The measured defect, stated as a test."""
        r = _Relief()
        released = kbr.KvBackingRelief.release_rows_after_floor(
            r, rows_wanted=107_049, current=CURRENT, floor=FLOOR, page=1
        )
        self.assertEqual(
            released,
            0,
            "a shrink below one granule returns nothing; it must report 0",
        )
        self.assertEqual(
            r.engaged,
            [],
            "and it must not have engaged the cap to discover that",
        )

    def test_a_shrink_that_does_cross_a_granule_still_proceeds(self):
        """The guard must not disable the rung on pools that CAN pay."""
        r = _Relief()
        big = CURRENT + 4 * GRANULE
        kbr.KvBackingRelief.release_rows_after_floor(
            r, rows_wanted=GRANULE, current=big, floor=0, page=1
        )
        self.assertEqual(len(r.engaged), 1, "a payable shrink must still run")
        target, _current = r.engaged[0]
        self.assertLessEqual(
            big - target,
            big,
            "sanity: the target lies below current",
        )
        self.assertGreaterEqual(
            big - target, GRANULE, "and it crosses at least one granule"
        )

    def test_guard_never_deepens_the_shrink(self):
        """#717 direction: the guard may only shrink LESS, never more.

        #717 killed boots by pulling backing below the highest live row.
        Any change here must be provably unable to do that. This guard only
        ever converts a shrink into no shrink.
        """
        r = _Relief()
        kbr.KvBackingRelief.release_rows_after_floor(
            r, rows_wanted=107_049, current=CURRENT, floor=FLOOR, page=1
        )
        for target, _c in r.engaged:
            self.assertGreaterEqual(
                target, FLOOR, "the target may never fall below the floor"
            )


if __name__ == "__main__":
    unittest.main()
