"""#744: a request PARKED for a phase flip is not an idle box.

THE SPECIMEN, 2026-08-17 19:30:40, from
``boot_bundle.log.20260817T193224Z`` (see ``ANALYSE_741``):

    19:30:31  PHASE-FLIP armed: pp_to_tp
    19:30:31  FLIP EXTENT PROBE: seqlen=51311 kv_allocated_len=51310
    19:30:31  CENSUS at-arm: cached=127182 cur_slot_reqs=4, backing=309464,
              highest live row=183998
    19:30:40  KV-BACKING EVICTED 127731 recomputable row(s) to bring the
              high-water mark below 61303 (resident ceiling -1)
    19:30:40  backing 61303 instead of 116736, highest live row 0
    19:30:4x  CUDA error: an illegal memory access was encountered

Twenty-four log lines from the eviction to the fault.

WHY BOTH HALVES WERE BLIND AT ONCE. ``_nothing_resident`` asks the live split
for ``req_rows``, and ``_shrink_to``'s safety net re-measures through
``_max_live_row`` -- which calls the SAME ``_live_slots_fn``. A flip quiesces
its requests before packing them, and a quiesced request sits in none of the
batch structures ``_live_reqs`` enumerates, so that one enumeration reports
zero to both. Fixing only the predicate would leave the net equally blind to
any other caller that shrinks during a park, which is why #744 feeds both from
one side channel instead.

Two independent lines, because the failure is a SILENT eviction followed by a
DELAYED fault -- nothing between the two says anything is wrong:
  1. the parked extent is visible to the predicate AND to the net;
  2. the rung refuses outright while a flip is armed.

The can-fail proofs cut both ways on purpose. Removing the extent visibility
must let the eviction happen again (or the test proves nothing), and the gate
must NOT make the rung dead outside flips -- #688's evict-rung funding path
depends on it staying live, and a fix that quietly disabled the rung would
"pass" the specimen while costing the thing #717 was built to win.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The specimen's numbers, verbatim.
PARKED_ROWS = 127_182  # cached rows the flip was about to pack
PARKED_TOP = 183_998  # highest live row at arm
EVICT_TARGET = 61_303  # the cap the rung shrank to
MAX_LIVE = 397_958


def _relief(
    split,
    *,
    pending=None,
    armed=None,
    evict_rows=50_000,
    live_max=None,
):
    """A rung carrying only what these methods read (the #717 idiom)."""
    from sglang.srt.managers import kv_backing_relief as m

    r = m.KvBackingRelief.__new__(m.KvBackingRelief)
    r._pool = type("P", (), {"page_size": 1})()
    r._margin_rows = 1
    r._admission_reserve_rows = 511
    r._last_live_split = split
    r._tree_cache_fn = lambda: object()
    r._evictable_rows = evict_rows
    r._flip_pending_fn = pending
    r._flip_armed_fn = armed
    r.evicted_rows_total = 0
    r.evict_count = 0
    r._device = 0
    r._device_index = 0
    r._bytes_per_row = 1024
    if live_max is not None:
        class _L:
            def numel(self):
                return 1

            def max(self):
                return live_max

        r._live_slots_fn = lambda: _L()
        r._live_slots_fn.last_split = split
    return r


class TestTheParkedExtentIsVisible(CustomTestCase):
    """Line 1: the extent reaches BOTH the predicate and the net."""

    def test_parked_rows_defeat_nothing_resident(self):
        """THE SPECIMEN. req_rows==0 while the flip holds 127182 rows."""
        r = _relief(
            {"req_max": -1, "req_rows": 0},
            pending=lambda: (PARKED_ROWS, PARKED_TOP),
        )
        self.assertFalse(
            r._nothing_resident(),
            "a parked flip extent must not read as an idle box",
        )

    def test_without_the_extent_it_still_reads_as_idle(self):
        """CAN-FAIL PROOF for line 1.

        Same split, extent channel removed: the predicate returns to its
        pre-#744 answer. If this ever fails, the test above is passing for
        some reason other than the fix.
        """
        r = _relief({"req_max": -1, "req_rows": 0}, pending=None)
        self.assertTrue(r._nothing_resident())

    def test_unknown_extent_blocks_rather_than_reads_empty(self):
        """(-1, -1) is UNKNOWN. Unknown is never treated as empty."""
        r = _relief(
            {"req_max": -1, "req_rows": 0}, pending=lambda: (-1, -1)
        )
        self.assertFalse(r._nothing_resident())

    def test_a_raising_probe_is_unknown_not_empty(self):
        def _boom():
            raise RuntimeError("probe down")

        r = _relief({"req_max": -1, "req_rows": 0}, pending=_boom)
        self.assertFalse(r._nothing_resident())

    def test_the_net_folds_the_parked_top_into_max_live_row(self):
        """The half a predicate-only fix would have missed.

        ``_shrink_to`` turns its cap into a fact by re-reading this. With the
        parked rows invisible it re-read 0 and confirmed a cap under 127k
        live rows.
        """
        r = _relief(
            {"req_max": -1, "req_rows": 0},
            pending=lambda: (PARKED_ROWS, PARKED_TOP),
            live_max=0,
        )
        self.assertEqual(r._max_live_row(), PARKED_TOP)

    def test_the_net_takes_the_higher_of_the_two(self):
        r = _relief(
            {"req_max": -1, "req_rows": 0},
            pending=lambda: (10, 5_000),
            live_max=200_000,
        )
        self.assertEqual(r._max_live_row(), 200_000)

    def test_unknown_extent_refuses_to_shrink_at_all(self):
        r = _relief(
            {"req_max": -1, "req_rows": 0},
            pending=lambda: (-1, -1),
            live_max=1_000,
        )
        self.assertEqual(
            r._max_live_row(), -1, "unknown parked extent must refuse, not cap"
        )

    def test_without_the_extent_the_net_reads_zero(self):
        """CAN-FAIL PROOF for the net: this is the pre-#744 blindness."""
        r = _relief({"req_max": -1, "req_rows": 0}, pending=None, live_max=0)
        self.assertEqual(r._max_live_row(), 0)


class TestTheArmedGate(CustomTestCase):
    """Line 2: refuse while armed, and ONLY while armed."""

    def setUp(self):
        import sglang.srt.managers.kv_radix_watermark as w

        self._w_orig = w.evictable_rows_above
        w.evictable_rows_above = lambda tree, floor: (50_000, 1)
        self.calls = []
        self._e_orig = w.evict_rows_above

        def _spy(tree, target, resident_ceiling=None):
            self.calls.append((target, resident_ceiling))
            return 50_000

        w.evict_rows_above = _spy

    def tearDown(self):
        import sglang.srt.managers.kv_radix_watermark as w

        w.evictable_rows_above = self._w_orig
        w.evict_rows_above = self._e_orig

    def test_armed_refuses_to_price(self):
        r = _relief({"req_max": -1, "req_rows": 0}, armed=lambda: True)
        floor, won = r._evict_floor_rows(MAX_LIVE)
        self.assertEqual(won, 0, "no eviction may be priced during a flip")
        self.assertEqual(floor, r._floor_rows(MAX_LIVE))

    def test_armed_refuses_to_collect(self):
        r = _relief({"req_max": -1, "req_rows": 0}, armed=lambda: True)
        self.assertEqual(r._lower_watermark_to(EVICT_TARGET), 0)
        self.assertEqual(self.calls, [], "nothing may be evicted while armed")

    def test_a_raising_armed_probe_is_treated_as_armed(self):
        def _boom():
            raise RuntimeError("controller gone")

        r = _relief({"req_max": -1, "req_rows": 0}, armed=_boom)
        self.assertEqual(r._lower_watermark_to(EVICT_TARGET), 0)

    def test_THE_RUNG_IS_NOT_DEAD_OUTSIDE_FLIPS(self):
        """CAN-FAIL PROOF for line 2, and the one that guards #688.

        A gate that always refused would satisfy every test above while
        silently killing the evict-rung funding path #717 exists to open. Not
        armed, nothing parked -> the eviction must still price AND collect.
        """
        r = _relief(
            {"req_max": -1, "req_rows": 0},
            armed=lambda: False,
            pending=lambda: (0, -1),
        )
        floor, won = r._evict_floor_rows(MAX_LIVE)
        self.assertGreater(won, 0, "the rung must stay live outside flips")
        freed = r._lower_watermark_to(EVICT_TARGET)
        self.assertGreater(freed, 0, "the priced eviction was not collected")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(
            self.calls[0][1], -1, "nothing resident pins anything here"
        )

    def test_default_construction_leaves_the_rung_unchanged(self):
        """No channels wired -> exactly the pre-#744 behaviour.

        The rung is built in several places; a fix that changed behaviour for
        callers that never opted in would be a silent regression elsewhere.
        """
        r = _relief({"req_max": -1, "req_rows": 0})
        self.assertFalse(r._flip_armed())
        self.assertEqual(r._flip_pending(), (0, -1))
        floor, won = r._evict_floor_rows(MAX_LIVE)
        self.assertGreater(won, 0)


class TestTheChannelsAreActuallyWired(CustomTestCase):
    """A fix wired into nothing is the defect class this repo keeps finding.

    Both channels default to inert, which is what keeps every existing caller
    unchanged -- and it is exactly what would let this ship doing nothing at
    all. Parsed from the source rather than grepped: the module's prose names
    both kwargs while explaining them, so a substring check would pass for the
    wrong reason.
    """

    def _factory_call(self):
        import ast
        import inspect

        from sglang.srt.managers import kv_backing_relief as m

        tree = ast.parse(inspect.getsource(m))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "KvBackingRelief"
            ):
                return node
        return None

    def test_the_production_factory_passes_both_channels(self):
        call = self._factory_call()
        self.assertIsNotNone(call, "no KvBackingRelief(...) construction found")
        kwargs = {k.arg for k in call.keywords}
        self.assertIn("flip_armed_fn", kwargs, "line 2 is not wired")
        self.assertIn("flip_pending_fn", kwargs, "line 1 is not wired")

    def test_the_pending_probe_is_inert_outside_a_flip(self):
        """The #688 guarantee, at the wiring level rather than the rung's."""

        class _Rt:
            def is_armed(self):
                return False

        live_fn = lambda: None  # noqa: E731 - stub
        live_fn.last_req_extent = (999, 999)

        def armed():
            return bool(_Rt().is_armed())

        def pending():
            if not armed():
                return (0, -1)
            return getattr(live_fn, "last_req_extent", None) or (-1, -1)

        self.assertEqual(
            pending(),
            (0, -1),
            "a stale extent must not survive outside a flip and pin the rung",
        )


class TestBothLinesTogether(CustomTestCase):
    def test_either_line_alone_stops_the_specimen(self):
        """Belt and suspenders, stated as a test rather than as a claim."""
        split = {"req_max": -1, "req_rows": 0}
        extent_only = _relief(
            split, pending=lambda: (PARKED_ROWS, PARKED_TOP), armed=lambda: False
        )
        gate_only = _relief(split, pending=None, armed=lambda: True)
        for name, r in (("extent", extent_only), ("gate", gate_only)):
            with self.subTest(line=name):
                self.assertEqual(
                    r._lower_watermark_to(EVICT_TARGET),
                    0,
                    f"{name} alone must stop the specimen",
                )

    def test_neither_line_reproduces_the_specimen(self):
        """CAN-FAIL for the pair: with both removed, the bug is back."""
        r = _relief({"req_max": -1, "req_rows": 0}, pending=None, armed=None)
        import sglang.srt.managers.kv_radix_watermark as w

        orig_e, orig_v = w.evict_rows_above, w.evictable_rows_above
        w.evictable_rows_above = lambda tree, floor: (50_000, 1)
        w.evict_rows_above = lambda tree, target, resident_ceiling=None: 50_000
        try:
            self.assertGreater(
                r._lower_watermark_to(EVICT_TARGET),
                0,
                "without either line the specimen's eviction still runs",
            )
        finally:
            w.evict_rows_above, w.evictable_rows_above = orig_e, orig_v


if __name__ == "__main__":
    unittest.main()
