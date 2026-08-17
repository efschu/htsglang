"""#552: a spilled session may not be starved by continuous fast-lane traffic.

THE SHAPE. ``_maybe_restore_flow`` defers a spilled session's restore while
ANY fast-lane request sits in the waiting queue, and the reason is good --
restoring into fast-lane pressure only re-triggers the spill, one full D2H+H2D
per cycle. But the deferral was UNBOUNDED: it called
``slot.hysteresis.reset()`` and returned, so no progress accumulated, and
under continuous fast-lane traffic an older spilled session never restored.
"Fast beats FCFS" is a tie-break; it was acting as an indefinite hold.

THE PRECEDENT, which is why this is a bound and not a redesign: the scheduler
already solved the same shape for the other lane. ``fast_lane_heavy_aging_ms``
promotes a heavy request that has waited too long AHEAD of the fast tier for
one admission. This is that rule in the units this loop has -- iterations, not
milliseconds.

FAILURE DIRECTION, stated because every anti-starvation bound has one: when
the bound fires, one restore happens while a fast request is waiting, and that
fast request may pay a re-spill. That is the price of not stranding a session
forever, and it is bounded to one restore per aged-out session.

These pins are on ``RestoreHysteresis`` -- the counter and the streak live in
one object precisely because the thing that zeroes the streak (a deferral) is
the thing that must be counted. Pure state machine, no scheduler, no CUDA.
"""

import unittest

from sglang.srt.managers.kv_session_offload import (
    DEFAULT_RESTORE_DEFER_LIMIT,
    RestoreHysteresis,
)


class TestTheDeferralIsCounted(unittest.TestCase):
    def test_a_deferral_still_zeroes_the_streak(self):
        """The anti-flutter behaviour must be unchanged: a deferral wipes
        accumulated readiness exactly as the bare reset() did."""
        h = RestoreHysteresis(3)
        self.assertFalse(h.update(True))
        self.assertFalse(h.update(True))
        h.defer()
        self.assertFalse(h.update(True), "the streak was not zeroed")

    def test_deferrals_accumulate(self):
        h = RestoreHysteresis(1, defer_limit=5)
        for i in range(3):
            h.defer()
            self.assertEqual(h.deferrals, i + 1)

    def test_the_bound_is_not_exceeded_early(self):
        h = RestoreHysteresis(1, defer_limit=4)
        self.assertFalse(h.defer())
        self.assertFalse(h.defer())
        self.assertFalse(h.defer())

    def test_the_bound_fires_at_the_limit(self):
        """THE PIN. Without it the session is stranded for as long as
        fast-lane traffic continues."""
        h = RestoreHysteresis(1, defer_limit=4)
        for _ in range(3):
            self.assertFalse(h.defer())
        self.assertTrue(h.defer(), "the deferral is unbounded: session stranded")

    def test_it_keeps_firing_once_past_the_limit(self):
        """Not a one-shot latch that a single missed check would lose."""
        h = RestoreHysteresis(1, defer_limit=2)
        h.defer()
        self.assertTrue(h.defer())
        self.assertTrue(h.defer())


class TestOnlyARealRestoreClearsTheCount(unittest.TestCase):
    """The subtle half. If a deferral cleared the count, the bound would be
    unreachable -- the code would look bounded and starve exactly as before."""

    def test_reset_does_not_clear_the_deferral_count(self):
        h = RestoreHysteresis(2, defer_limit=3)
        h.defer()
        h.defer()
        h.reset()
        self.assertEqual(
            h.deferrals,
            2,
            "reset() cleared the anti-starvation count, so the bound can "
            "never be reached and the fix is cosmetic",
        )

    def test_clear_deferrals_resets_it(self):
        h = RestoreHysteresis(2, defer_limit=3)
        h.defer()
        h.defer()
        h.clear_deferrals()
        self.assertEqual(h.deferrals, 0)

    def test_after_a_restore_the_session_gets_the_full_budget_again(self):
        h = RestoreHysteresis(1, defer_limit=3)
        h.defer()
        h.defer()
        h.clear_deferrals()
        self.assertFalse(h.defer())
        self.assertFalse(h.defer())
        self.assertTrue(h.defer())


class TestTheBoundIsDisableable(unittest.TestCase):
    """<= 0 restores the pre-#552 behaviour exactly, so an operator who
    prefers the old indefinite hold can have it and the change is reversible
    without a revert."""

    def test_zero_never_fires(self):
        h = RestoreHysteresis(1, defer_limit=0)
        for _ in range(1000):
            self.assertFalse(h.defer())

    def test_negative_never_fires(self):
        h = RestoreHysteresis(1, defer_limit=-5)
        for _ in range(100):
            self.assertFalse(h.defer())

    def test_it_still_counts_while_disabled(self):
        """Observability survives the switch: the count is still visible even
        when it does not act, so a starving session is diagnosable."""
        h = RestoreHysteresis(1, defer_limit=0)
        h.defer()
        h.defer()
        self.assertEqual(h.deferrals, 2)


class TestTheDefaultIsGenerous(unittest.TestCase):
    """The default must not change normal behaviour -- it fires only in the
    pathological case, or it would trade a rare starvation for a common
    re-spill."""

    def test_the_default_is_large_enough_to_be_pathological_only(self):
        self.assertGreaterEqual(DEFAULT_RESTORE_DEFER_LIMIT, 50)

    def test_the_default_is_on(self):
        """Default-OFF would ship the starvation. A bound that must be
        switched on is not a fix."""
        self.assertGreater(DEFAULT_RESTORE_DEFER_LIMIT, 0)
        self.assertFalse(RestoreHysteresis(1).defer())
        self.assertEqual(RestoreHysteresis(1).defer_limit, DEFAULT_RESTORE_DEFER_LIMIT)


class TestTheCallSiteIsWired(unittest.TestCase):
    """Source pin: a correct state machine nobody calls is the #421 class."""

    def setUp(self):
        import inspect

        from sglang.srt.managers import kv_session_offload as m

        self.src = inspect.getsource(m.KVSessionOffloadManager._maybe_restore_flow)

    def test_the_deferral_site_counts_instead_of_bare_reset(self):
        self.assertIn("slot.hysteresis.defer()", self.src)

    def test_the_site_falls_through_rather_than_always_returning(self):
        """`if not ...defer(): return` is the shape that lets the bound act;
        an unconditional return would count and still strand."""
        self.assertIn("if not slot.hysteresis.defer():", self.src)

    def test_a_forced_restore_is_logged(self):
        self.assertIn("fast-lane deferrals", self.src)


if __name__ == "__main__":
    unittest.main()
