"""#717: "nothing resident" and "unknown" are opposite states with one encoding.

``_resident_ceiling`` returns -1 for BOTH, and its own docstring says so:
"Highest row a RESIDENT REQUEST pins, or -1 when none/unknown."
``build_flip_live_slots_fn`` sets ``req_max`` to -1 when it has no request
parts, so an idle box is encoded exactly like an unreadable split.

The two demand opposite behaviour:
  * unreadable  -> evict NOTHING (unmapping a row a live request reads is the
    one unrecoverable error);
  * none resident -> the whole live set is tree-only and MAXIMALLY evictable.

The rung took the conservative branch precisely when it had the most to win,
so the floor pinned to max_live + reserve on an idle box:

    BOTH BLOCKED ... 0 req resident
    KV rung: current=137216 rows, floor=398471, slack=0

That is the 2.9x-over-cap shape (397,958 max_live against a 137,216 cap), and
it makes slack permanently 0, which kills the #688 evict-rung funding path and
leaves every flip funded by the raw seam budget alone.
"""

import unittest

from sglang.test.test_utils import CustomTestCase

# the specimen, verbatim
MAX_LIVE = 397_958
CURRENT_CAP = 137_216
OBSERVED_FLOOR = 398_471  # == MAX_LIVE + 1 + margin + reserve, as logged


def _relief(split, evict_rows=50_000):
    """A KvBackingRelief with only the attributes these two methods read."""
    from sglang.srt.managers import kv_backing_relief as m

    r = m.KvBackingRelief.__new__(m.KvBackingRelief)
    r._pool = type("P", (), {"page_size": 1})()
    r._margin_rows = 1
    r._admission_reserve_rows = 511  # 1 + 1 + 511 = 513 above max_live
    r._last_live_split = split
    r._tree_cache_fn = lambda: object()
    r._evictable_rows = evict_rows
    return r


class TestNothingResidentIsNotUnknown717(CustomTestCase):
    def setUp(self):
        from sglang.srt.managers import kv_backing_relief as m

        # price the watermark deterministically; the real one walks a tree
        self._orig = m.__dict__.get("_TEST_PATCHED", None)
        import sglang.srt.managers.kv_radix_watermark as w

        self._w_orig = w.evictable_rows_above
        w.evictable_rows_above = lambda tree, floor: (50_000, 1)

    def tearDown(self):
        import sglang.srt.managers.kv_radix_watermark as w

        w.evictable_rows_above = self._w_orig

    def test_the_specimen_shape_reproduces(self):
        """Sanity: the plain floor really is the logged 398471."""
        r = _relief({"req_max": -1, "req_rows": 0})
        self.assertEqual(r._floor_rows(MAX_LIVE), OBSERVED_FLOOR)
        self.assertGreater(
            OBSERVED_FLOOR,
            CURRENT_CAP,
            "the floor must exceed the cap: slack pins to 0",
        )

    def test_nothing_resident_prices_the_eviction(self):
        """THE FIX. Zero resident rows means the whole live set is evictable,
        so the floor must collapse to the reserve rather than to max_live."""
        r = _relief({"req_max": -1, "req_rows": 0})
        floor, rows = r._evict_floor_rows(MAX_LIVE)
        self.assertLess(
            floor, OBSERVED_FLOOR, f"floor must drop below the plain floor, got {floor}"
        )
        self.assertLess(
            floor,
            CURRENT_CAP,
            f"floor must fall under the cap so slack > 0, got {floor}",
        )
        self.assertGreater(rows, 0, "an eviction must be priced")

    def test_unreadable_split_still_refuses(self):
        """CAN-FAIL. The conservative branch must survive for its real case.
        A fix that simply treated -1 as 'evict everything' passes the test
        above and fails this one -- and that mistake unmaps live rows."""
        for split in (None, {}, {"req_max": -1}):
            with self.subTest(split=split):
                r = _relief(split)
                floor, rows = r._evict_floor_rows(MAX_LIVE)
                self.assertEqual(floor, r._floor_rows(MAX_LIVE))
                self.assertEqual(rows, 0, "no eviction may be priced when unknown")

    def test_resident_rows_present_is_unchanged(self):
        """A pinned mark still refuses: nothing to win when work is in flight."""
        r = _relief({"req_max": MAX_LIVE, "req_rows": 4096})
        floor, rows = r._evict_floor_rows(MAX_LIVE)
        self.assertEqual(floor, r._floor_rows(MAX_LIVE))
        self.assertEqual(rows, 0)

    def test_nothing_resident_predicate_is_exact(self):
        self.assertTrue(_relief({"req_max": -1, "req_rows": 0})._nothing_resident())
        self.assertFalse(_relief({"req_max": -1, "req_rows": 5})._nothing_resident())
        self.assertFalse(_relief({"req_max": -1})._nothing_resident())
        self.assertFalse(_relief(None)._nothing_resident())
        self.assertFalse(_relief({})._nothing_resident())


if __name__ == "__main__":
    unittest.main()
