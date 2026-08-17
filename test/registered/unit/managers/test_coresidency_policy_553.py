"""#553 Cut 3: hot/cold events ACTUATE, and the two directions fail differently.

Cuts 1+2 built the bridge and its probes; nothing called them — verified by
grep before building this: no importer of ``coresidency_registry`` existed
outside its own module, so the answer to "does a cold event actuate today" was
**no, it only prices**.

NO NEW ACTUATOR. Every move composes an existing authority:
``vram_dial.apply_budget_request`` (replicated grow/shrink, already enforces
the floor, "rejections carry the exact floor arithmetic and change nothing"),
``vram_dial.verify_pool_reached_capacity`` (the read-back),
``GdnSlotRuntime.unbind`` (#364), and the Cut 1/2 bridge.

THE ASYMMETRY IS THE DESIGN:

  * **Shrink must not strand bytes.** A source asked for bytes that reports
    nothing is carried as STRANDED, never counted as zero and never dropped.
    Bytes that left one ledger and entered none are the shape that goes
    unnoticed for weeks.
  * **Grow must not exceed the floor.** Not enforced here on purpose — the
    dial enforces it. This module's duty is to not paper over the refusal.
  * **#217:** "came back" is a measurement, not an assertion. A grow reports
    what a read-back MEASURED. The victim that stayed at 23% post-restore is
    why ``reached_bytes`` exists and why it is never the request.
  * **#694:** a count is a promise only if the same call delivered it.
    Totals come from actuator reports, never from the plan.
"""

import unittest

from sglang.srt.managers.coresidency_policy import cold_event, hot_event
from sglang.srt.managers.coresidency_registry import (
    ORIGIN_ASSET,
    ORIGIN_DIAL,
    ReclaimSource,
    ReclaimView,
)


def _view(*sources, unavailable=()):
    return ReclaimView(tuple(sources), tuple(unavailable))


def _src(name, nbytes, origin=ORIGIN_DIAL, rank=0):
    return ReclaimSource(
        name=name,
        origin=origin,
        reclaimable_bytes=nbytes,
        restorable=True,
        cost_rank=rank,
    )


class TestColdActuatesInRankedOrder(unittest.TestCase):
    def test_it_calls_the_release_for_each_planned_source(self):
        calls = []

        def _release(source):
            calls.append(source.name)
            return True, source.reclaimable_bytes, "ok"

        result = cold_event(
            3000,
            view=_view(_src("dial0", 2000), _src("experts", 2000, ORIGIN_ASSET, 1)),
            release_fn=_release,
        )
        self.assertTrue(result.ok)
        self.assertEqual(calls, ["dial0", "experts"])
        self.assertEqual(result.delivered_bytes, 4000)

    def test_it_stops_once_the_ask_is_covered(self):
        calls = []

        def _release(source):
            calls.append(source.name)
            return True, source.reclaimable_bytes, "ok"

        cold_event(
            1000,
            view=_view(_src("dial0", 2000), _src("experts", 2000, ORIGIN_ASSET, 1)),
            release_fn=_release,
        )
        self.assertEqual(calls, ["dial0"], "took more sources than the ask needed")


class TestColdRefusesRatherThanPartiallyDrawing(unittest.TestCase):
    def test_an_unfundable_ask_actuates_nothing(self):
        calls = []

        result = cold_event(
            9000,
            view=_view(_src("dial0", 1000)),
            release_fn=lambda s: calls.append(s.name) or (True, 0, ""),
        )
        self.assertFalse(result.ok)
        self.assertIn("refused", result.refused)
        self.assertEqual(calls, [], "a refused event still actuated")

    def test_the_refusal_says_unavailable_bytes_do_not_count(self):
        from sglang.srt.managers.coresidency_registry import Unavailable

        result = cold_event(
            9000,
            view=_view(
                _src("dial0", 1000),
                unavailable=(Unavailable("experts", ORIGIN_ASSET, "VA-stable"),),
            ),
            release_fn=lambda s: (True, s.reclaimable_bytes, ""),
        )
        self.assertIn("do not count toward the ask", result.refused)


class TestShrinkNeverStrandsBytes(unittest.TestCase):
    """A source asked and silent is STRANDED, not zero and not dropped."""

    def test_a_source_that_reports_nothing_is_flagged_stranded(self):
        result = cold_event(
            1000,
            view=_view(_src("dial0", 2000)),
            release_fn=lambda s: (True, None, "released, did not measure"),
        )
        self.assertEqual(len(result.stranded_steps), 1)
        self.assertEqual(result.stranded_steps[0].name, "dial0")

    def test_stranded_bytes_are_not_counted_as_delivered(self):
        """#694: a count is a promise only if the same call delivered it."""
        result = cold_event(
            1000,
            view=_view(_src("dial0", 2000)),
            release_fn=lambda s: (True, None, ""),
        )
        self.assertEqual(result.delivered_bytes, 0)

    def test_a_delivered_zero_is_not_stranded(self):
        """Zero is an accounting; silence is not."""
        result = cold_event(
            1000,
            view=_view(_src("dial0", 2000)),
            release_fn=lambda s: (True, 0, "nothing to give"),
        )
        self.assertEqual(result.stranded_steps, ())
        self.assertEqual(result.delivered_bytes, 0)

    def test_a_failing_release_is_recorded_not_dropped(self):
        result = cold_event(
            1000,
            view=_view(_src("dial0", 2000)),
            release_fn=lambda s: (False, None, "device busy"),
        )
        self.assertFalse(result.ok)
        self.assertEqual(len(result.steps), 1)
        self.assertIn("device busy", result.steps[0].detail)

    def test_a_raising_release_is_recorded_not_dropped(self):
        def _boom(source):
            raise RuntimeError("actuator exploded")

        result = cold_event(
            1000, view=_view(_src("dial0", 2000)), release_fn=_boom
        )
        self.assertFalse(result.ok)
        self.assertIn("actuator exploded", result.steps[0].detail)


class TestGrowRespectsTheDialsRefusal(unittest.TestCase):
    """The floor is the dial's to enforce; this must not paper over it."""

    def test_a_floor_refusal_is_returned_as_is(self):
        result = hot_event(
            1 << 30,
            grow_fn=lambda n: (False, "would fall below the corridor floor by 512 MiB"),
        )
        self.assertFalse(result.ok)
        self.assertIn("corridor floor", result.refused)

    def test_a_refused_grow_is_never_retried_smaller(self):
        calls = []

        hot_event(
            1000,
            grow_fn=lambda n: calls.append(n) or (False, "below floor"),
        )
        self.assertEqual(
            calls, [1000], "a floor refusal is a statement about the rig, not "
            "a negotiation"
        )


class TestHotMeasuresRatherThanAsserts(unittest.TestCase):
    """#217: the victim stayed at 23% post-restore. 'Came back' is measured."""

    def test_reached_reports_the_measurement_not_the_request(self):
        result = hot_event(
            10_000,
            grow_fn=lambda n: (True, "accepted"),
            measure_fn=lambda: 2_300,
        )
        self.assertTrue(result.ok)
        self.assertEqual(
            result.reached_bytes,
            2_300,
            "reporting the request would have claimed a restore that did not "
            "happen -- the #217 shape exactly",
        )

    def test_an_unmeasured_grow_reads_as_unmeasured_not_as_zero(self):
        """#606: absent measurement and measured-nothing are different."""
        result = hot_event(10_000, grow_fn=lambda n: (True, "accepted"))
        self.assertIsNone(result.reached_bytes)

    def test_a_failing_readback_refuses_rather_than_reporting_warm(self):
        def _boom():
            raise RuntimeError("nvml down")

        result = hot_event(
            10_000, grow_fn=lambda n: (True, "accepted"), measure_fn=_boom
        )
        self.assertFalse(result.ok)
        self.assertIn("must not be reported warm", result.refused)

    def test_a_measured_zero_is_reported_as_zero(self):
        result = hot_event(
            10_000, grow_fn=lambda n: (True, "ok"), measure_fn=lambda: 0
        )
        self.assertEqual(result.reached_bytes, 0)
        self.assertTrue(result.ok, "a measured zero is a fact, not a failure")


class TestNoNewActuator(unittest.TestCase):
    """Source pin: every move must be an existing authority's own API."""

    def test_the_module_introduces_no_actuator_of_its_own(self):
        import inspect

        from sglang.srt.managers import coresidency_policy as m

        src = inspect.getsource(m)
        for forbidden in ("torch.cuda", "cuMem", "set_active_prefix", "runtime_set_"):
            with self.subTest(token=forbidden):
                self.assertNotIn(
                    forbidden,
                    src,
                    "this module must compose existing actuators, never move "
                    "memory itself",
                )


if __name__ == "__main__":
    unittest.main()
