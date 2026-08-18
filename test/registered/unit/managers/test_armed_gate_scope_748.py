"""#748 regression: the armed gate still wedges an IDLE box.

My own #748 refinement turned the wholesale armed gate into an exclusion set --
rows inside the parked extent are protected, rows above are funding. It left
exactly ONE wholesale case, documented at the time as "an UNKNOWN extent while
a flip is armed, where there is no boundary to name".

That case has an INVERTED failure mode, and comp4 reproduced it on the
composite (2026-08-18 06:36Z: "no KV provider" x9, "IDLE-LOCK" x5, and the
vacuous relief form on pp_to_tp -- "returned NOTHING ... evicted 0 rows over 0
shrinks"). The mechanism:

* ``_flip_pending`` answers from ``live_fn.last_req_extent``;
* ``phase_flip_runtime`` sets that attribute ONLY when
  ``split["req_rows"] > 0``;
* on an IDLE box no enumeration ever sees a resident request, so the attribute
  is NEVER set;
* so the probe answers UNKNOWN, the ceiling answers -2, and the rung refuses
  wholesale.

It refuses hardest in exactly the case where nothing is parked and there is
nothing to protect. 407k tokens pending, 0 running, flip armed, no funding,
wedge.

THE DISTINCTION THAT FIXES IT is the #717 family's own lesson one level out:
``req_rows == 0`` is two opposite states with one encoding -- "nothing has ever
been resident" (an idle box: nothing to protect) and "resident but quiesced for
the flip" (parked: protect it). They are separable from data already on hand:
if any enumeration ever saw a resident request the sticky extent EXISTS, and if
an enumeration has run at all the split is readable. Only a split that was
never readable is genuinely unknown.

The can-fail direction is #744's: the armed flip's own pack rows must STILL be
protected. A fix that simply evicts everything while armed must fail that pin.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PARKED_ROWS = 127_182
PARKED_TOP = 183_998
MAX_LIVE = 397_958
EVICT_TARGET = 61_303


def _relief(*, pending, armed, split=None, evict_rows=50_000):
    from sglang.srt.managers import kv_backing_relief as m

    r = m.KvBackingRelief.__new__(m.KvBackingRelief)
    r._pool = type("P", (), {"page_size": 1})()
    r._margin_rows = 1
    r._admission_reserve_rows = 511
    r._last_live_split = split if split is not None else {"req_max": -1, "req_rows": 0}
    r._tree_cache_fn = lambda: object()
    r._evictable_rows = evict_rows
    r._flip_pending_fn = pending
    r._flip_armed_fn = armed
    r.evicted_rows_total = 0
    r.evict_count = 0
    r._device = 0
    r._device_index = 0
    r._bytes_per_row = 1024
    return r


def _pending_probe(live_fn, armed_fn):
    """The production closure's logic, exercised directly.

    Reproduced here rather than reaching into ``build_kv_backing_relief``
    because that factory needs a pool, an allocator and a live-set function;
    the decision this file is about is entirely in these few lines.
    """
    from sglang.srt.managers.kv_backing_relief import flip_pending_from_live_fn

    return flip_pending_from_live_fn(live_fn, armed_fn)


class _LiveFn:
    """Stands in for ``build_flip_live_slots_fn``'s closure."""

    def __init__(self, *, last_split=None, last_req_extent=None):
        if last_split is not None:
            self.last_split = last_split
        if last_req_extent is not None:
            self.last_req_extent = last_req_extent

    def __call__(self):
        return None


class _WatermarkStub:
    """The win is priced through kv_radix_watermark; the #717/#744 suites stub
    it the same way. Without it the rung prices 0 for a reason that has nothing
    to do with the gate under test."""

    def setUp(self):
        import sglang.srt.managers.kv_radix_watermark as w

        self._v_orig = w.evictable_rows_above
        self._e_orig = w.evict_rows_above
        self.calls = []
        w.evictable_rows_above = lambda tree, floor: (50_000, 1)

        def _spy(tree, target, resident_ceiling=None):
            self.calls.append((target, resident_ceiling))
            return 50_000

        w.evict_rows_above = _spy

    def tearDown(self):
        import sglang.srt.managers.kv_radix_watermark as w

        w.evictable_rows_above = self._v_orig
        w.evict_rows_above = self._e_orig


class TestTheIdleBoxSpecimen(_WatermarkStub, CustomTestCase):
    """0 running, flip armed, 407k pending: the IDLE-LOCK shape."""

    def test_an_idle_box_reports_NOTHING_PARKED_not_unknown(self):
        """The root. An enumeration ran and saw zero resident requests -- that
        is positive evidence, not absence of evidence."""
        live = _LiveFn(last_split={"req_rows": 0, "req_max": -1})
        self.assertEqual(_pending_probe(live, lambda: True), (0, -1))

    def test_the_idle_box_rung_still_prices_and_collects(self):
        """The wedge, at the rung: it must fund while a flip is armed."""
        r = _relief(
            pending=lambda: _pending_probe(
                _LiveFn(last_split={"req_rows": 0, "req_max": -1}), lambda: True
            ),
            armed=lambda: True,
        )
        floor, won = r._evict_floor_rows(MAX_LIVE)
        self.assertGreater(won, 0, "an idle box with a flip armed must still fund")
        freed = r._lower_watermark_to(EVICT_TARGET)
        self.assertGreater(
            freed, 0, "priced but never collected -- the vacuous relief form"
        )

    def test_a_split_that_never_ran_is_still_UNKNOWN(self):
        """CAN-FAIL: the fix must not turn every absence into 'nothing parked'.

        No enumeration has run at all, so there is genuinely no evidence. That
        must still refuse.
        """
        live = _LiveFn()
        self.assertEqual(_pending_probe(live, lambda: True), (-1, -1))


class TestTheUnderLoadSpecimen(_WatermarkStub, CustomTestCase):
    """Requests resident and quiesced for the flip: the 'no KV provider' shape."""

    def test_a_parked_extent_is_reported_and_protected(self):
        live = _LiveFn(
            last_split={"req_rows": 0, "req_max": -1},
            last_req_extent=(PARKED_ROWS, PARKED_TOP),
        )
        self.assertEqual(_pending_probe(live, lambda: True), (PARKED_ROWS, PARKED_TOP))

    def test_CAN_FAIL_744_the_armed_flips_own_pack_rows_stay_protected(self):
        """#744's regression direction, pinned.

        A fix that unblocked the funder by evicting everything while armed
        would pass every test above and re-open the illegal-access window this
        gate exists for. The evictor's ceiling must not reach into the extent.
        """
        r = _relief(
            pending=lambda: (PARKED_ROWS, PARKED_TOP),
            armed=lambda: True,
        )
        floor, won = r._evict_floor_rows(MAX_LIVE)
        self.assertGreaterEqual(
            floor,
            r._floor_rows(PARKED_TOP),
            "the floor dropped below the parked extent: pack rows unprotected",
        )

    def test_outside_a_flip_nothing_is_parked(self):
        live = _LiveFn(last_req_extent=(PARKED_ROWS, PARKED_TOP))
        self.assertEqual(_pending_probe(live, lambda: False), (0, -1))


if __name__ == "__main__":
    unittest.main()
