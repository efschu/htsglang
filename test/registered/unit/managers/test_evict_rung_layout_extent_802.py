"""#802: a parked extent is a row id, and a row id belongs to ONE pool.

THE DEADLOCK. ``build_flip_live_slots_fn`` remembers the last enumeration
that saw resident requests on ``live_fn.last_req_extent`` (#744, so the rung
cannot evict rows a quiesced flip is about to read). There is exactly ONE
writer and NO clearer, and its safety argument reads:

    "The rung consults this ONLY while a flip is armed, so a stale value
     cannot outlive the flip and cannot leave the rung dead outside one."

That covers TIME. It does not cover LAYOUT. The extent also outlives the
CUTOVER, so one enumerated in the PP phase is still on the function when
tp_to_pp arms later -- by which point the resident pool is the TP one and the
ids come from a different, larger space.

MEASURED ON METAL (boot_802_staged1_0822_1528, 43 occurrences):

    KV rung: current=212992 rows, floor=348106, slack=0
    FLOOR UNREACHABLE: it exceeds the current cap by 135114 rows, so this
    rung can never fund and every flip depends on the raw seam fund alone.

``floor = max_live + 1 + margin + reserve``, and the healthy samples solve
the constant: parked ceiling 160661 priced floor 164758, so margin+reserve is
4096. Applying it to the latched floor gives max_live = 344009 -- against a
TP pool whose ENTIRE cap is 212992 rows. A live row at id 344009 cannot exist
in a 212992-row pool, so the ceiling was not describing the resident layout.

The tell is that the SAME floor was reported against three different caps
(212992 / 133120 / 124928): it had stopped tracking the active pool
altogether. Unlatched samples priced floor 148253/164758 with 48234-64739
rows of genuine slack, i.e. the pool could have funded the seam every time.

Consequence: rung dead -> seam unfundable -> flip abandoned -> re-armed ->
13 consecutive refusals, limiter at its 60 s cap, DRAIN-MODE SUPPRESSION
YIELDED, and prefill stuck in TP at 1335 tok/s.

THE RULE, which ``_active_layout_pool`` already states one call away: the two
layouts are two pools and only one is backed, so a value bound to the
released one cannot describe the resident one.

WHAT IS PINNED HERE. The discard is deliberately NARROW, because the danger
direction is unmapping a row a flip is about to read (#722/#744 cost us that
twice). Only a POSITIVE mismatch -- both sides known AND different -- may
discard. Every ambiguous case keeps protecting, and each one is asserted
below in the direction that would go RED if the discard were widened.
"""

import unittest

from sglang.srt.managers.kv_backing_relief import flip_pending_from_live_fn
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The metal numbers, used as the fixture so the test reads against the
# specimen rather than against invented values.
PP_SPACE_ROW = 344009  # priced floor 348106 = this + 1 + 4096
TP_POOL_CAP = 212992
PARKED_ROWS = 8


class _Live:
    """Stands in for the live-slots function, which carries the attributes."""


def _live(extent=None, split=None):
    fn = _Live()
    if extent is not None:
        fn.last_req_extent = extent
    if split is not None:
        fn.last_split = split
    return fn


class TestLayoutTaggedExtent(CustomTestCase):
    def test_extent_from_the_other_layout_is_discarded(self):
        """THE FIX. A PP-space extent must not price the TP pool's floor.

        Without it this returns (8, 344009) and the caller prices a floor of
        348106 against a 212992-row cap -- the measured deadlock.
        """
        fn = _live(extent=(PARKED_ROWS, PP_SPACE_ROW, "pp"))
        self.assertEqual(
            flip_pending_from_live_fn(fn, lambda: True, lambda: "tp"),
            (0, -1),
            "an extent from the released layout still priced the resident pool",
        )

    def test_same_layout_extent_STILL_PROTECTS(self):
        """THE DANGER DIRECTION, and the reason the discard is narrow.

        This is the #744 guarantee: while armed in the layout the extent was
        measured in, those rows must stay backed. Widen the discard by one
        step -- drop the tag comparison and always return (0, -1) -- and the
        rung is free to evict rows the flip is about to read, which is the
        crash #722/#744 already paid for. That mutation turns this RED.
        """
        fn = _live(extent=(PARKED_ROWS, 160661, "tp"))
        self.assertEqual(
            flip_pending_from_live_fn(fn, lambda: True, lambda: "tp"),
            (PARKED_ROWS, 160661),
            "the rung stopped protecting rows parked in the RESIDENT layout",
        )

    def test_untagged_extent_still_protects(self):
        """An extent written before #802 has no tag; it must not be discarded.

        Ambiguity resolves toward protection, never toward eviction.
        """
        fn = _live(extent=(PARKED_ROWS, 160661))
        self.assertEqual(
            flip_pending_from_live_fn(fn, lambda: True, lambda: "tp"),
            (PARKED_ROWS, 160661),
        )

    def test_unreadable_active_layout_still_protects(self):
        """A layout probe that RAISES must not authorise eviction."""

        def boom():
            raise RuntimeError("no stacks installed")

        fn = _live(extent=(PARKED_ROWS, PP_SPACE_ROW, "pp"))
        self.assertEqual(
            flip_pending_from_live_fn(fn, lambda: True, boom),
            (PARKED_ROWS, PP_SPACE_ROW),
        )

    def test_unknown_active_layout_still_protects(self):
        """None means 'could not tell', which is not 'the other one'."""
        fn = _live(extent=(PARKED_ROWS, PP_SPACE_ROW, "pp"))
        self.assertEqual(
            flip_pending_from_live_fn(fn, lambda: True, lambda: None),
            (PARKED_ROWS, PP_SPACE_ROW),
        )

    def test_omitted_layout_fn_keeps_the_748_two_arg_contract(self):
        """#748 calls this with two arguments; that form must be unchanged."""
        fn = _live(extent=(PARKED_ROWS, PP_SPACE_ROW, "pp"))
        self.assertEqual(
            flip_pending_from_live_fn(fn, lambda: True),
            (PARKED_ROWS, PP_SPACE_ROW),
        )

    def test_not_armed_is_still_nothing_parked(self):
        """#688/#748: outside a flip the rung stays fully live."""
        fn = _live(extent=(PARKED_ROWS, PP_SPACE_ROW, "pp"))
        self.assertEqual(
            flip_pending_from_live_fn(fn, lambda: False, lambda: "tp"), (0, -1)
        )

    def test_unreadable_armed_state_is_UNKNOWN_not_idle(self):
        """#748: an unreadable flip blocks; it is not an idle box."""

        def boom():
            raise RuntimeError("unreadable")

        fn = _live(extent=(PARKED_ROWS, PP_SPACE_ROW, "pp"))
        self.assertEqual(flip_pending_from_live_fn(fn, boom, lambda: "tp"), (-1, -1))

    def test_discarded_extent_puts_the_floor_back_under_the_cap(self):
        """THE LATCH, stated as the arithmetic the metal log printed.

        floor = max_live + 1 + margin + reserve, margin+reserve = 4096 (solved
        from the healthy sample: ceiling 160661 -> floor 164758). With the
        PP-space extent in play the floor clears the whole TP cap; with it
        discarded the floor comes from the resident live set and leaves real
        slack -- which is what "the rung may fund the seam" means.
        """
        margin_plus_reserve = 4096
        resident_live_max = 160661

        latched_floor = PP_SPACE_ROW + 1 + margin_plus_reserve
        self.assertEqual(latched_floor, 348106, "fixture drifted from the specimen")
        self.assertGreater(
            latched_floor, TP_POOL_CAP, "the specimen's floor did not exceed the cap"
        )

        healthy_floor = resident_live_max + 1 + margin_plus_reserve
        self.assertEqual(healthy_floor, 164758)
        self.assertLess(healthy_floor, TP_POOL_CAP)
        self.assertEqual(TP_POOL_CAP - healthy_floor, 48234, "slack lost its meaning")


if __name__ == "__main__":
    unittest.main()
