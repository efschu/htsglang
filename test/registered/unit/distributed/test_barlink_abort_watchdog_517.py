"""#517 phase 2: the abort guard stops charging the serving hot path.

Task #600 sampled a live bs=1 decode round with py-spy: 11 of 14 samples sat
in ``check_aborted``, ~7 ms of a 46.5 ms round. The cause is not the check
itself but its BOUND. ``ENV_MAX_LAG`` counts checks, and a check-count bound
presumes that "the host is far ahead of the device" is a fault precursor --
while under the overlap scheduler at bs=1 it is the design. The host is always
more than four checks ahead, so the bound bound every round and forced a
stream synchronization, i.e. the guard spent the run-ahead the scheduler
exists to build.

The fix moves the read off the serving path entirely: the barlink watchdog
thread -- which already exists and is already awake on a timer -- reads the
sticky word on a PRIVATE stream and publishes a plain flag. The hot path reads
the flag. The bound becomes TIME (``ENV_POLL_MS``), which is the question the
guard actually promises to answer.

These tests pin: the hot path is provably free (a counter, not a stopwatch),
the guard still raises, the latency bound is stated in time, the capture
exclusion holds, and every kill switch still works.
"""

import os
import types
import unittest
from unittest import mock

from sglang.srt.distributed.device_communicators import barlink_abort_gate as gate
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Word:
    """A stand-in for the device abort word that COUNTS its reads.

    The point of the suite: "the hot path does not read the device" is a
    claim about a call count, so it is asserted as one. A stopwatch would
    measure the test host's mood instead.
    """

    def __init__(self, value=0):
        self.value = value
        self.reads = 0
        self.is_cuda = True

    def __getitem__(self, idx):
        self.reads += 1
        return self

    def item(self):
        self.reads += 1
        return self.value


def _transport(**over):
    """A BarlinkDeviceTransport-shaped stub carrying the REAL methods."""
    from sglang.srt.distributed.device_communicators.barlink_device import (
        BarlinkDeviceTransport,
    )

    word = over.pop("word", _Word())
    stub = types.SimpleNamespace(
        _seq_dev=word,
        _abort_poll_active=True,
        _abort_code_seen=0,
        _abort_poll_stream=None,
        _abort_poll_dst=None,
        _unchecked_launches=1,
        _captured_launches=False,
        _boundary_checks=0,
        rank=0,
        world_size=3,
        _TIMEOUT_CYCLES=60_000_000_000,
    )
    stub.__dict__.update(over)
    stub.check_aborted = BarlinkDeviceTransport.check_aborted.__get__(stub)
    stub._raise_aborted = BarlinkDeviceTransport._raise_aborted.__get__(stub)
    stub.word = word
    return stub


class TestTheHotPathIsCheckFree(unittest.TestCase):
    def test_a_thousand_checks_read_the_device_zero_times(self):
        t = _transport()
        for _ in range(1000):
            t.check_aborted("host-path collective")
        self.assertEqual(t.word.reads, 0)

    def test_it_does_not_ask_the_environment_either(self):
        """An os.environ lookup per collective is not free at 8+ per round,
        and the enable switch is only needed on the path that raises."""
        t = _transport()
        with mock.patch.object(
            gate, "abort_check_enabled", side_effect=AssertionError("hot path")
        ):
            for _ in range(100):
                t.check_aborted("host-path collective")

    def test_it_does_not_ask_whether_a_capture_is_running(self):
        """``graph_capture_running`` is a CUDA API call. The clean hot path
        must not reach it -- there is no device read to protect."""
        import sglang.srt.distributed.device_communicators.barlink as bl

        t = _transport()
        with mock.patch.object(
            bl, "graph_capture_running", side_effect=AssertionError("hot path")
        ):
            for _ in range(100):
                t.check_aborted("host-path collective")

    def test_the_fallback_path_still_reads_in_line(self):
        """The can-fail twin: with no watchdog the transport keeps the
        pre-#517 in-line read, so the counter above is measuring the new
        path and not a broken stub."""
        t = _transport(_abort_poll_active=False)
        import sglang.srt.distributed.device_communicators.barlink as bl

        with mock.patch.object(bl, "graph_capture_running", return_value=False):
            t.check_aborted("host-path collective")
        self.assertGreater(t.word.reads, 0)


class TestTheGuardStillGuards(unittest.TestCase):
    def test_a_tripped_word_raises_on_the_next_check(self):
        from sglang.srt.distributed.device_communicators.barlink_device import (
            DeviceCollectiveAborted,
        )

        t = _transport(_abort_code_seen=2)
        with self.assertRaises(DeviceCollectiveAborted) as cm:
            t.check_aborted("cuda-graph replay")
        self.assertIn("rank 0/3", str(cm.exception))
        self.assertIn("cuda-graph replay", str(cm.exception))

    def test_the_report_is_the_same_one_definition_as_the_inline_path(self):
        """Both callers go through ``_raise_aborted``; a second copy of this
        diagnostic would drift on the first edit."""
        import inspect

        from sglang.srt.distributed.device_communicators.barlink_device import (
            BarlinkDeviceTransport,
        )

        src = inspect.getsource(BarlinkDeviceTransport.check_aborted)
        self.assertEqual(src.count("raise DeviceCollectiveAborted("), 0)
        self.assertEqual(src.count("self._raise_aborted("), 2)

    def test_the_kill_switch_still_silences_it(self):
        t = _transport(_abort_code_seen=2)
        with mock.patch.dict(os.environ, {gate.ENV_ENABLE: "0"}):
            t.check_aborted("host-path collective")

    def test_the_poll_is_one_way_and_sticky(self):
        from sglang.srt.distributed.device_communicators.barlink_device import (
            BarlinkDeviceTransport,
        )

        t = _transport(_abort_code_seen=7)
        t.poll_status_word = BarlinkDeviceTransport.poll_status_word.__get__(t)
        self.assertTrue(t.poll_status_word())
        self.assertEqual(t.word.reads, 0)  # already tripped: no read at all
        self.assertEqual(t._abort_code_seen, 7)


class TestTheBoundIsTimeNotCheckCount(unittest.TestCase):
    def test_the_default_poll_interval_is_ten_milliseconds(self):
        self.assertAlmostEqual(gate.poll_interval_s(), 0.010)

    def test_it_is_settable_and_is_read_per_call(self):
        with mock.patch.dict(os.environ, {gate.ENV_POLL_MS: "2.5"}):
            self.assertAlmostEqual(gate.poll_interval_s(), 0.0025)

    def test_a_nonsense_value_falls_back_loudly_rather_than_to_zero(self):
        with mock.patch.dict(os.environ, {gate.ENV_POLL_MS: "soon"}):
            self.assertAlmostEqual(gate.poll_interval_s(), 0.010)

    def test_the_documented_bound_names_the_unit_change(self):
        import inspect

        src = inspect.getsource(gate)
        self.assertIn("ENV_POLL_MS", src)
        self.assertIn("check-count bound", src)
        self.assertIn("overlap scheduler", src)


class TestTheCaptureExclusion(unittest.TestCase):
    """Torch captures in global mode: a synchronizing call in ANY thread
    invalidates the capture, so the poll must be locked out, not merely
    asked to be polite."""

    def setUp(self):
        gate.reset_for_test()

    def tearDown(self):
        gate.reset_for_test()

    def test_polling_is_paused_inside_the_capture_context(self):
        self.assertFalse(gate.polling_paused())
        with gate.pause_polling():
            self.assertTrue(gate.polling_paused())
        self.assertFalse(gate.polling_paused())

    def test_a_paused_poll_reads_nothing(self):
        polled = []
        gate.register(types.SimpleNamespace(poll_status_word=lambda: polled.append(1)))
        with gate.pause_polling():
            self.assertEqual(gate.poll_status_words(), 0)
        self.assertEqual(polled, [])
        gate.poll_status_words()
        self.assertEqual(polled, [1])

    def test_the_capture_context_manager_enters_it(self):
        """One definition of 'a capture is happening': the process-wide
        ``graph_capture``. A per-backend guess is what drifts."""
        import inspect

        from sglang.srt.distributed import parallel_state

        src = inspect.getsource(parallel_state.graph_capture)
        self.assertIn("pause_polling()", src)

    def test_the_disable_switch_stops_the_poll_too(self):
        polled = []
        gate.register(types.SimpleNamespace(poll_status_word=lambda: polled.append(1)))
        with mock.patch.dict(os.environ, {gate.ENV_ENABLE: "0"}):
            gate.poll_status_words()
        self.assertEqual(polled, [])

    def test_a_transport_that_raises_in_the_poll_does_not_kill_the_watchdog(self):
        def boom():
            raise RuntimeError("device gone")

        gate.register(types.SimpleNamespace(poll_status_word=boom))
        self.assertEqual(gate.poll_status_words(), 0)


class TestTheWatchdogRunsBothDuties(unittest.TestCase):
    def test_the_thread_polls_the_abort_words(self):
        import inspect

        from sglang.srt.distributed.device_communicators import barlink_liveness

        src = inspect.getsource(barlink_liveness.PeerWatchdog._run)
        self.assertIn("poll_abort_words()", src)
        self.assertIn("poll_interval_s()", src)

    def test_the_liveness_probe_keeps_its_own_slower_cadence(self):
        """Folding the two would tie each duty's latency to the other's
        natural interval -- /proc at 100 Hz, or an abort report at 1 Hz."""
        import inspect

        from sglang.srt.distributed.device_communicators import barlink_liveness

        src = inspect.getsource(barlink_liveness.PeerWatchdog._run)
        self.assertIn("next_probe", src)

    def test_a_transport_only_arms_the_poll_when_a_watchdog_exists(self):
        self.assertTrue(gate.should_poll_status(True, True))
        self.assertFalse(gate.should_poll_status(True, False))
        self.assertFalse(gate.should_poll_status(False, True))


if __name__ == "__main__":
    unittest.main()
