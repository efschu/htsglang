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
"""#796: a rung that priced NO eviction must say which branch refused it.

WHY THIS IS THE DECIDING NUMBER, not a nicety. The group's shrink target is
``max(desire, max_floor)`` over every rank's floor, so a single rank whose
floor sits at or above its own cap cancels the shrink for the whole group --
including for a rank holding a fully fundable plan. Measured on metal
2026-08-22 (boot_798_0822_0543.log): PP0 could have returned +1740 MiB from
89119 rows of slack, PP2 reported ``fundable_bytes() == 0`` on all eight of
its asks, and the flip abandoned eight times.

``fundable_bytes`` is ``max(0, current - floor)``, so PP2's zero means its
floor was at or above its cap. Its floor came from ``_evict_floor_rows``,
which has EIGHT distinct ways to return the plain, un-evicted floor:
eviction disabled, an unreadable parked extent, no tree cache, an unknown
resident half, a mark pinned by work in flight, a priced floor no better than
the plain one, a pricing exception, and nothing evictable above the reserve.

Over a SPARSE live set the plain floor is routinely ABOVE the cap -- #714
established that ``max_live`` is a high-water ID in the id space, not a count
of backed rows -- so every one of those eight branches produces the same
observable: floor >= current, slack 0, and a group-wide veto. They want
completely different responses. Three of them are healthy (the pool is
genuinely live), one is a configuration (eviction off), and the rest are
defects. The log distinguishes none of them, which is why this took a boot
per hypothesis.

Hermetic: no CUDA, no scheduler, no distributed. The rung carries only the
fields these methods read (the #717 fixture idiom).
"""

from __future__ import annotations

import unittest
from unittest import mock

from sglang.srt.managers import kv_backing_relief as kbr

#: A high-water ID over a sparse set: far above any plausible backed-row
#: count, which is exactly the #714 shape that makes the plain floor a veto.
MAX_LIVE = 336_694
CURRENT = 204_800


def _relief(split, *, pending=None, armed=None, tree=True, evict_enabled=True):
    """A rung carrying only what these methods read (the #717 fixture idiom)."""
    r = kbr.KvBackingRelief.__new__(kbr.KvBackingRelief)
    r._pool = type("P", (), {"page_size": 1})()
    r._margin_rows = 1
    r._admission_reserve_rows = 511
    r._last_live_split = split
    r._tree_cache_fn = (lambda: object()) if tree else (lambda: None)
    r._evictable_rows = 0
    r._flip_pending_fn = pending
    r._flip_armed_fn = armed
    r.evicted_rows_total = 0
    r.evict_count = 0
    r._device = 0
    r._device_index = 0
    r._bytes_per_row = 1024
    r._evict_enabled = lambda: evict_enabled
    return r


def _reason(r) -> str:
    return str(getattr(r, "_last_evict_floor_reason", "") or "")


class TestEveryPlainFloorBranchNamesItself(unittest.TestCase):
    def test_eviction_disabled_says_so(self):
        r = _relief({"req_max": -1, "req_rows": 0}, evict_enabled=False)
        floor, rows = r._evict_floor_rows(MAX_LIVE)
        self.assertEqual(rows, 0)
        self.assertGreater(
            floor, CURRENT, "the sparse plain floor vetoes by construction"
        )
        self.assertIn("evict", _reason(r).lower())

    def test_no_tree_cache_says_so(self):
        r = _relief({"req_max": -1, "req_rows": 0}, tree=False)
        floor, rows = r._evict_floor_rows(MAX_LIVE)
        self.assertEqual(rows, 0)
        self.assertTrue(_reason(r), "a plain floor must always carry a reason")
        self.assertIn("tree", _reason(r).lower())

    def test_unknown_resident_half_says_so(self):
        """req_max < 0 with rows resident: unreadable, and a defect if it persists."""
        r = _relief({"req_max": -1, "req_rows": 7})
        floor, rows = r._evict_floor_rows(MAX_LIVE)
        self.assertEqual(rows, 0)
        self.assertIn("resident", _reason(r).lower())

    def test_mark_pinned_by_work_in_flight_says_so(self):
        r = _relief({"req_max": MAX_LIVE + 10, "req_rows": 5})
        floor, rows = r._evict_floor_rows(MAX_LIVE)
        self.assertEqual(rows, 0)
        self.assertIn("pinned", _reason(r).lower())

    def test_nothing_evictable_says_so(self):
        """The healthy cause: the pool is genuinely live. Still must be named."""
        r = _relief({"req_max": -1, "req_rows": 0})
        with mock.patch(
            "sglang.srt.managers.kv_radix_watermark.evictable_rows_above",
            return_value=(0, 0),
        ):
            floor, rows = r._evict_floor_rows(MAX_LIVE)
        self.assertEqual(rows, 0)
        self.assertIn("evictable", _reason(r).lower())

    def test_a_floor_above_the_high_water_is_not_reported_as_liveness(self):
        """The zero that means NOTHING, and used to be read as health.

        `_floor_rows(x) == x + 1 + margin + reserve`, so a resident ceiling
        within (margin + reserve) rows of the high-water lifts the PRICED floor
        past the high-water row. `evictable_rows_above` is then asked for rows
        in a region the tree cannot hold anything in, and returns zero for that
        reason alone. Branch 8 reported that zero as "the pool is genuinely
        live" -- a claim about liveness drawn from a query that measured none.

        Measured on metal, boot_798_0822_0737.log 07:50:19Z: priced floor
        167440 against high-water row 164055, plain floor 168152, reported as
        health while the flip stayed wedged in TP.
        """
        # req_max just under the high-water: past the PINNED guard, below the
        # plain floor, but far enough up that the priced floor clears max_live.
        r = _relief({"req_max": MAX_LIVE - 100, "req_rows": 3})
        with mock.patch(
            "sglang.srt.managers.kv_radix_watermark.evictable_rows_above",
            return_value=(0, 0),
        ):
            floor, rows = r._evict_floor_rows(MAX_LIVE)
        self.assertEqual(rows, 0)
        reason = _reason(r)
        self.assertNotIn(
            "genuinely live",
            reason,
            "an empty query region is not evidence of liveness; this is the "
            f"mislabelling that cost a boot to diagnose. Got: {reason!r}",
        )
        self.assertIn("above the high-water", reason.lower())

    def test_a_real_band_with_nothing_priced_stays_a_named_suspicion(self):
        """The OTHER zero: a genuine band the tree prices at nothing.

        Sample A on the same boot -- priced floor 97643, high-water row 134148,
        so 36506 rows lie above the resident ceiling and at or below the
        high-water and the tree prices none of them. That may be health, or it
        may be the ~94000 unaccounted rows the POOL CENSUS reports. The message
        must not decide which; it must point at the reading that settles it.
        """
        r = _relief({"req_max": 1_000, "req_rows": 3})
        with mock.patch(
            "sglang.srt.managers.kv_radix_watermark.evictable_rows_above",
            return_value=(0, 0),
        ):
            floor, rows = r._evict_floor_rows(MAX_LIVE)
        self.assertEqual(rows, 0)
        reason = _reason(r).lower()
        self.assertIn("evictable", reason)
        self.assertIn("census", reason, "name where the answer is read")

    def test_the_two_zero_branches_return_identically(self):
        """This change is instrumentation. It must not move the ladder.

        Both branches return `(plain, 0)` exactly as the single branch did, so
        no shrink decision anywhere changes as a result of distinguishing them.
        """
        results = []
        for split in ({"req_max": MAX_LIVE - 100, "req_rows": 3},
                      {"req_max": 1_000, "req_rows": 3}):
            r = _relief(split)
            with mock.patch(
                "sglang.srt.managers.kv_radix_watermark.evictable_rows_above",
                return_value=(0, 0),
            ):
                results.append(r._evict_floor_rows(MAX_LIVE))
        plain = MAX_LIVE + 1 + 1 + 511
        self.assertEqual(results, [(plain, 0), (plain, 0)])

    def test_a_priced_eviction_reports_that_it_priced_one(self):
        """The success path is named too, or absence of a reason is ambiguous."""
        r = _relief({"req_max": -1, "req_rows": 0})
        with mock.patch(
            "sglang.srt.managers.kv_radix_watermark.evictable_rows_above",
            return_value=(50_000, 3),
        ):
            floor, rows = r._evict_floor_rows(MAX_LIVE)
        self.assertEqual(rows, 50_000)
        self.assertLess(floor, CURRENT, "a priced eviction must beat the cap")
        self.assertTrue(_reason(r))
        self.assertIn("priced", _reason(r).lower())


class TestTheReasonReachesTheRefusalLine(unittest.TestCase):
    """A reason recorded and never printed is the defect this ticket is about."""

    def test_summary_carries_the_reason_when_the_rung_has_no_slack(self):
        r = _relief({"req_max": -1, "req_rows": 0})
        with mock.patch(
            "sglang.srt.managers.kv_radix_watermark.evictable_rows_above",
            return_value=(0, 0),
        ):
            floor, _rows = r._evict_floor_rows(MAX_LIVE)
        r._last_proposal_terms = {
            "current": CURRENT,
            "floor_rows": floor,
            "deficit": 0,
            "desire": CURRENT,
            "skipped": "",
        }
        summary = r.last_proposal_summary()
        self.assertIn(
            "evictable",
            summary.lower(),
            "a rung whose floor vetoes the group must say WHY its floor is "
            f"where it is; got {summary!r}",
        )


if __name__ == "__main__":
    unittest.main()
