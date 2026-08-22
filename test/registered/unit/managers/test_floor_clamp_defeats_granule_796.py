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
"""The release granule, characterised against the geometry that shipped.

RETRACTION. The first version of this file asserted that PP2's release
granule was 229376 rows and therefore EXCEEDED its whole pool of 126976,
so that "on this rank no shrink can pay at all at this chunk size". That
was wrong, and the error was invisible because the fixture STUBBED
``_min_release_rows`` to return the number it wanted. A characterisation
test that supplies its own answer cannot fail when the answer is wrong.

What the boot actually recorded:

  * boot_798_0822_0737.log:1325 -- "32768 B/row over 32 arena buffers"
  * the same boot's server_args: flip_seam_chunk_mib=8,
    enable_vram_dial=False, so the arena commit chunk is 8 MiB
  * boot_798_0822_0737.log:3382 -- current=126976 floor=88945 slack=38031

    _min_release_rows() = ceil(8 MiB * 32 / 32 KiB) = 8192 rows = 256 MiB

8192, not 229376. The granule is 6.4% of the pool, and PP2's post-floor
distance of 38031 rows is 4.64 granules. The string "229376" does not
occur anywhere in that boot log; it is 28 * 8192, and it is the fixture
constant from ``test_shrink_cannot_pay_reason_796.py`` (256 MiB * 28 /
32 KiB), which was carried into a commit message and a code comment as
if it had been measured.

So "the floor clamp defeats the granule" is NOT what happened on PP2.
Every one of that boot's 15 zero-byte shrinks asked at least three whole
granules deep, and the guard this file pins would not have prevented a
single one of them. Why those shrinks released nothing is a different
question, and it is still open.

The guard itself stays, and stays tested: a shrink whose post-clamp
distance really is below one granule must not engage the cap, because
discovering that by attempting it costs the rank its capacity and
returns nothing. That shape is constructed here rather than claimed to
be measured.

DIRECTION OF SAFETY, unchanged: the guard only ever turns a shrink into
NO shrink. It never deepens one, so it cannot pull backing below the
highest live row -- the #717 fault that reverted c4e557963e.
"""

from __future__ import annotations

import math
import unittest

from sglang.srt.managers import kv_backing_relief as kbr

# PP2's measured shape, boot_798_0822_0737.log:3382.
CURRENT = 126_976
FLOOR = 88_945
SLACK = CURRENT - FLOOR  # 38031, and the log's own "slack=38031" agrees.

# PP2's measured geometry, boot_798_0822_0737.log:1325 plus that boot's
# server_args. These are the three terms _min_release_rows is made of.
CHUNK_BYTES = 8 * 1024 * 1024
BUFFERS = 32
BYTES_PER_ROW = 32_768

# Derived from the three terms above, not asserted independently of them.
GRANULE = math.ceil(CHUNK_BYTES * BUFFERS / BYTES_PER_ROW)  # 8192


class _Pool:
    backing_commit_chunk_bytes = CHUNK_BYTES

    def __init__(self):
        self.shrunk_to = []

    def runtime_set_backing_rows(self, target):
        self.shrunk_to.append(int(target))
        return 0


class _Relief:
    """Only the surface the release-rows arithmetic touches.

    ``_min_release_rows`` is the REAL implementation, bound off the class.
    The previous fixture stubbed it, which is why a granule that no chunk
    and buffer count in this file could produce survived review.
    """

    def __init__(self, *, buffers=BUFFERS, bytes_per_row=BYTES_PER_ROW):
        self._pool = _Pool()
        self._buffers = buffers
        self._bytes_per_row = bytes_per_row
        self.engaged = []

    _min_release_rows = kbr.KvBackingRelief._min_release_rows

    def _shrink_to(self, target, current):
        self.engaged.append((int(target), int(current)))
        return 0


class TestReleaseGranuleOnTheShippedGeometry(unittest.TestCase):
    def test_the_granule_is_computed_from_the_measured_terms(self):
        """The retraction, stated as a test.

        This is the assertion the old fixture could not make, because it
        returned its granule instead of computing one.
        """
        r = _Relief()
        self.assertEqual(
            r._min_release_rows(),
            8192,
            "8 MiB in each of 32 buffers at 32 KiB/row is 8192 rows",
        )
        self.assertEqual(r._min_release_rows(), GRANULE)

    def test_the_granule_does_not_exceed_the_pool(self):
        """The specific claim that was wrong, pinned so it cannot return."""
        r = _Relief()
        granule = r._min_release_rows()
        self.assertLess(
            granule,
            CURRENT,
            "the granule is 6.4% of PP2's pool, not larger than it",
        )
        self.assertLess(
            granule,
            SLACK,
            "and it fits inside the post-floor distance 4.64 times, so the "
            "floor clamp does NOT defeat it on this rank",
        )

    def test_pp2s_measured_shape_engages_a_payable_shrink(self):
        """At PP2's real numbers the guard must NOT refuse.

        The zero-byte shrinks in that boot therefore need another
        explanation; refusing here would hide the real one.
        """
        r = _Relief()
        kbr.KvBackingRelief.release_rows_after_floor(
            r, rows_wanted=107_049, current=CURRENT, floor=FLOOR, page=1
        )
        self.assertEqual(
            len(r.engaged), 1, "PP2's measured shape is payable and must run"
        )
        target, _current = r.engaged[0]
        self.assertEqual(target, FLOOR, "the clamp binds at the eviction floor")
        self.assertGreaterEqual(
            CURRENT - target,
            3 * GRANULE,
            "and the surviving distance is more than three whole granules, "
            "which is why the granule cannot explain a zero release",
        )


class TestGuardOnASubGranuleShape(unittest.TestCase):
    """CONSTRUCTED shapes. Not measured -- no boot has shown this on PP2."""

    def test_sub_granule_shrink_must_not_engage_the_cap(self):
        r = _Relief()
        near_floor = CURRENT - (GRANULE // 8)
        released = kbr.KvBackingRelief.release_rows_after_floor(
            r, rows_wanted=107_049, current=CURRENT, floor=near_floor, page=1
        )
        self.assertEqual(
            released, 0, "a shrink below one granule returns nothing"
        )
        self.assertEqual(
            r.engaged, [], "and must not engage the cap to discover that"
        )

    def test_a_shrink_that_does_cross_a_granule_still_proceeds(self):
        r = _Relief()
        big = CURRENT + 4 * GRANULE
        kbr.KvBackingRelief.release_rows_after_floor(
            r, rows_wanted=GRANULE, current=big, floor=0, page=1
        )
        self.assertEqual(len(r.engaged), 1, "a payable shrink must still run")
        target, _current = r.engaged[0]
        self.assertGreaterEqual(
            big - target, GRANULE, "and it crosses at least one granule"
        )

    def test_a_chunkless_arena_yields_no_granule(self):
        """The belt: chunk 0 means no extent can clear at any depth."""
        r = _Relief()
        r._pool.backing_commit_chunk_bytes = 0
        self.assertEqual(r._min_release_rows(), 0)

    def test_guard_never_deepens_the_shrink(self):
        """#717 direction: the guard may only shrink LESS, never more."""
        for floor in (FLOOR, CURRENT - (GRANULE // 8), CURRENT - 1):
            r = _Relief()
            kbr.KvBackingRelief.release_rows_after_floor(
                r, rows_wanted=107_049, current=CURRENT, floor=floor, page=1
            )
            for target, _c in r.engaged:
                self.assertGreaterEqual(
                    target, floor, "the target may never fall below the floor"
                )
                self.assertLess(target, CURRENT)


if __name__ == "__main__":
    unittest.main()
