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


class TestBothSidesAgreeOnTheBranch717(CustomTestCase):
    """The half the first attempt missed, and the reason it crashed.

    ``_evict_floor_rows`` and ``_lower_watermark_to`` both key on
    ``_resident_ceiling() < 0``. c4e557963e taught the PRICING side that
    nothing-resident is priceable and left the EVICTION side refusing it, so
    the rung priced a target it then declined to collect: nothing was
    evicted, and the cap engaged over a full live set.

    A disagreement between these two is not a lost optimisation, it is an
    illegal address. These pins hold them to the same reading.
    """

    def setUp(self):
        import sglang.srt.managers.kv_radix_watermark as w

        self._orig_evict = w.evict_rows_above
        self.calls = []

        def _spy(tree, target_row, *, resident_ceiling=-1):
            self.calls.append((int(target_row), int(resident_ceiling)))
            return 4096

        w.evict_rows_above = _spy

    def tearDown(self):
        import sglang.srt.managers.kv_radix_watermark as w

        w.evict_rows_above = self._orig_evict

    def _armed(self, split):
        r = _relief(split)
        r.evicted_rows_total = 0
        r.evict_count = 0
        r._device = 0
        r._device_index = 0
        r._bytes_per_row = 1024
        return r

    def test_nothing_resident_actually_evicts(self):
        """The eviction must RUN on the branch pricing just opened."""
        r = self._armed({"req_max": -1, "req_rows": 0})

        freed = r._lower_watermark_to(1000)

        self.assertGreater(freed, 0, "the priced eviction was never collected")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(
            self.calls[0][1],
            -1,
            "no resident row pins anything, so nothing may be withheld from "
            "the eviction on that grounds",
        )

    def test_unreadable_split_evicts_nothing(self):
        """CAN-FAIL, and the one that matters: the conservative branch must
        survive for its real case. A fix that simply dropped the guard would
        pass the test above and fail this one -- and that unmaps live rows."""
        for split in (None, {}, {"req_max": -1}):
            with self.subTest(split=split):
                self.calls.clear()
                r = self._armed(split)

                freed = r._lower_watermark_to(1000)

                self.assertEqual(freed, 0)
                self.assertEqual(
                    self.calls, [], "nothing may be evicted on an unknown split"
                )


if __name__ == "__main__":
    unittest.main()
