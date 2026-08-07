"""#614 (d): graph-coverage attribution on the spill-tick path.

Hermetic: the classifier is pure, and the wiring is pinned against the module
source in the same style as ``test_kv_session_offload_unit.py``'s AST pins. No
card, no server, no CUDA graph.

WHAT WAS MISSING. Arms H (``H_ps2_prefill_spill``) and I (``I_dflash_shards``)
of the #550 window ran under full CUDA graphs -- ``boot_matrix/arms.py``
``BASE_EXPECT`` declares ``graphs=True`` and both inherit it -- while the only
per-tick observable, the ``SGLANG_KVSO_TICK_TRACE`` line, reported an interval,
a ``tick_cost`` and a host-tail size with nothing saying whether that cost came
from a graph-covered or an eager segment. ``ANALYSE_spill_matrix_20260804.md``
S13 recorded the same hole as "the spill-tick decomposition instrument does not
exist ... ms/step, but not the three-way split".

THE ORDER-DEPENDENCE THAT SHAPES THIS. ``SplitDeviceTimer`` hands the SAME
``Slot`` object to every reporter registered on the timer, and
``CollectiveClock.harvest_detail`` is DESTRUCTIVE -- it returns the events to
the pool and clears ``slot.pairs``. kvso registers its reporter with
``add_reporter`` (``_install_regulator_device_timer``), i.e. last, so any ms it
read would be an order-dependent zero. ``graph_capture_skipped`` is the one
field harvesting does not touch. ``test_harvest_does_not_disturb_the_signal``
pins that property against the real clock, because the whole design rests on
it.

CAN-FAIL PROOF: change ``tick_graph_state_from_slot`` to return
``TICK_GRAPH_EAGER`` for a ``None`` slot -> ``test_absent_slot_is_unattributed``
goes red. Observed red before restoring.
"""

import inspect
import unittest

from sglang.srt.managers import kv_session_offload
from sglang.srt.managers.kv_session_offload import (
    TICK_GRAPH_COVERED,
    TICK_GRAPH_EAGER,
    TICK_GRAPH_UNATTRIBUTED,
    tick_graph_state_from_slot,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Slot:
    def __init__(self, skipped):
        self.graph_capture_skipped = skipped


class TestTickGraphClassifier(CustomTestCase):
    def test_absent_slot_is_unattributed(self):
        """No slot reached the reporter -- no SplitDeviceTimer is installed on
        this runner. Reporting that as "eager" is the wrong zero the
        CollectiveClock docstring (utils/collective_clock.py:41-44) exists to
        refuse; the trace must say it does not know."""
        self.assertEqual(tick_graph_state_from_slot(None), TICK_GRAPH_UNATTRIBUTED)

    def test_capture_skipped_is_covered(self):
        self.assertEqual(tick_graph_state_from_slot(_Slot(True)), TICK_GRAPH_COVERED)

    def test_nothing_skipped_is_eager(self):
        self.assertEqual(tick_graph_state_from_slot(_Slot(False)), TICK_GRAPH_EAGER)

    def test_the_three_states_are_distinct(self):
        states = {TICK_GRAPH_UNATTRIBUTED, TICK_GRAPH_COVERED, TICK_GRAPH_EAGER}
        self.assertEqual(len(states), 3)

    def test_a_slot_without_the_attribute_is_not_covered(self):
        """Defensive on purpose: a future Slot variant that drops the flag must
        degrade to "eager", never silently to "covered"."""

        class _Bare:
            pass

        self.assertEqual(tick_graph_state_from_slot(_Bare()), TICK_GRAPH_EAGER)


class TestSignalSurvivesHarvest(CustomTestCase):
    """The property the whole design rests on: reading the graph flag is
    order-independent because harvesting does not touch it."""

    def test_harvest_does_not_disturb_the_signal(self):
        from sglang.srt.utils.collective_clock import CollectiveClock

        clock = CollectiveClock()
        clock.arm()
        slot = clock.disarm()
        slot.graph_capture_skipped = True
        # An empty slot harvests to a zero result without needing a card; the
        # point is what harvesting LEAVES BEHIND.
        self.assertIsNotNone(clock.harvest_detail(slot))
        self.assertEqual(tick_graph_state_from_slot(slot), TICK_GRAPH_COVERED)

    def test_harvest_clears_the_pairs_it_owns(self):
        """States the destructive half explicitly, so the reason kvso must not
        harvest is pinned rather than only commented."""
        from sglang.srt.utils.collective_clock import CollectiveClock

        clock = CollectiveClock()
        clock.arm()
        slot = clock.disarm()
        clock.harvest_detail(slot)
        self.assertEqual(slot.pairs, [])


class TestWiring(CustomTestCase):
    """The classifier is only worth anything if it is actually called and its
    result actually printed."""

    def setUp(self):
        self.src = inspect.getsource(kv_session_offload)

    def test_reporter_reads_the_collective_slot(self):
        report = inspect.getsource(
            kv_session_offload.KVSessionOffloadManager._device_timer_report
        )
        self.assertIn("collective_slot", report)
        self.assertIn("tick_graph_state_from_slot", report)

    def test_reporter_does_not_harvest(self):
        """If this ever fails, the metrics reporter's reading has been stolen
        -- two consumers of one destructive harvest, decided by registration
        order."""
        report = inspect.getsource(
            kv_session_offload.KVSessionOffloadManager._device_timer_report
        )
        # The two destructive entry points, by call shape -- the word
        # "harvest" itself appears in the prose that explains why they are
        # not called.
        self.assertNotIn("harvest_detail(", report)
        self.assertNotIn(".harvest(", report)

    def test_trace_line_carries_the_state(self):
        self.assertIn("tick_graph=%s", self.src)

    def test_trace_line_still_carries_its_original_fields(self):
        """The addition must not have displaced the existing time series --
        S13 named tick_cost and host_tail as what K2 could deliver."""
        for field in ("tick_cost=%.3fms", "host_tail=%d", "interval=%d"):
            with self.subTest(field=field):
                self.assertIn(field, self.src)

    def test_state_defaults_to_unattributed(self):
        """Initialised before any tick is reported, so an absent value never
        reads as a measurement."""
        self.assertIn("self._tick_graph_state = TICK_GRAPH_UNATTRIBUTED", self.src)


if __name__ == "__main__":
    unittest.main()
