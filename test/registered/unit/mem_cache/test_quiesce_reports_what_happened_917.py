"""#917: the #760 drain must not report a drain it did not perform.

THE SPECIMEN. `boot_rerun0826_0826_2149.log`:

    L2189  PP1  #760 quiesce (phase flip pp_to_tp): synchronizing write_stream failed (IMA)
    L2195  PP1  #760 quiesce (phase flip pp_to_tp): synchronizing load_stream failed (IMA)
    L2201  PP1  #760 device-tier I/O quiesced for phase flip pp_to_tp in 0.2 ms
                (write and load streams drained while their pools are still live).

Both streams failed and the summary line said "drained" anyway. PP2 did the
same at L2464/L2470/L2476. A `grep -c "device-tier I/O quiesced"` over that
boot returns 3 -- one per rank -- in a boot where exactly ONE rank drained
cleanly. The counter-specimen makes the cost concrete:
`boot_accept0826r7fix_0826_1817.log` has 96 of that line and 0 failures, and
the two logs were compared on that line.

THE CLASS. An exception handler on the no-return path that cannot tell a
survivable failure from a context kill, feeding an instrument that then
reports the survivable reading -- #867 in `barlink_abort_gate`,
`SeamCensus.mark` in `phase_flip_seam_census`, and this. Third instance.

RED-FIRST. On the parent commit the success line is emitted unconditionally
and nothing is registered with the poison record; every assertion below fails
there except the two that pin the pre-existing clean path.
"""

import types
import unittest

from sglang.srt.distributed.device_communicators import barlink_abort_gate
from sglang.srt.managers.cache_controller import HiCacheController

IMA = "CUDA error: an illegal memory access was encountered"


class _Stream:
    """A stream that records its synchronize, or raises what the metal raised."""

    def __init__(self, log, name, raises=None):
        self._log = log
        self._name = name
        self._raises = raises

    def synchronize(self):
        self._log.append(self._name)
        if self._raises is not None:
            raise self._raises


def _controller(calls, write_raises=None, load_raises=None):
    return types.SimpleNamespace(
        write_stream=_Stream(calls, "write", write_raises),
        load_stream=_Stream(calls, "load", load_raises),
    )


class TestTheDrainReportsWhatHappened(unittest.TestCase):
    def setUp(self):
        barlink_abort_gate.clear_poison_record()

    def tearDown(self):
        barlink_abort_gate.clear_poison_record()

    def test_a_clean_drain_still_reports_a_clean_drain(self):
        """The pre-existing success path, byte-identical.

        The corpus is grepped for this line. Changing it for a healthy flip
        would invalidate every existing reading of it.
        """
        calls = []
        with self.assertLogs("sglang.srt.managers.cache_controller", "INFO") as cm:
            HiCacheController.quiesce_device_io(_controller(calls), "test seam")
        self.assertEqual(calls, ["write", "load"])
        joined = "\n".join(cm.output)
        self.assertIn("device-tier I/O quiesced for test seam", joined)
        self.assertNotIn("NOT quiesced", joined)

    def test_a_failed_stream_suppresses_the_success_line(self):
        """The defect, stated as a pin: no "quiesced" line when nothing drained."""
        calls = []
        c = _controller(
            calls, write_raises=RuntimeError(IMA), load_raises=RuntimeError(IMA)
        )
        with self.assertLogs("sglang.srt.managers.cache_controller", "ERROR") as cm:
            HiCacheController.quiesce_device_io(c, "phase flip pp_to_tp")
        joined = "\n".join(cm.output)
        self.assertIn("NOT quiesced", joined)
        self.assertIn("write_stream and load_stream did not drain", joined)
        # The success wording must be absent, not merely accompanied.
        self.assertNotIn("streams drained while their pools are still live", joined)

    def test_one_failed_stream_is_enough_to_withhold_the_claim(self):
        """A half-drain is not a drain.

        The seam proceeds either way -- there is no abort left at the no-return
        point -- so the only thing the instrument controls is whether the
        reader is told.
        """
        calls = []
        c = _controller(calls, load_raises=RuntimeError(IMA))
        with self.assertLogs("sglang.srt.managers.cache_controller", "ERROR") as cm:
            HiCacheController.quiesce_device_io(c, "phase flip tp_to_pp")
        joined = "\n".join(cm.output)
        self.assertIn("NOT quiesced", joined)
        self.assertIn("load_stream did not drain", joined)
        self.assertNotIn("write_stream and", joined)
        # Both streams are still ATTEMPTED: a failed write must not skip the load.
        self.assertEqual(calls, ["write", "load"])

    def test_the_drain_still_returns_its_elapsed_time(self):
        """#690: the wait is attributable whether or not it succeeded."""
        calls = []
        c = _controller(calls, write_raises=RuntimeError(IMA))
        elapsed = HiCacheController.quiesce_device_io(c, "seam")
        self.assertGreaterEqual(elapsed, 0.0)


class TestTheDrainRegistersPoisonWithTheProcessRecord(unittest.TestCase):
    def setUp(self):
        barlink_abort_gate.clear_poison_record()

    def tearDown(self):
        barlink_abort_gate.clear_poison_record()

    def test_a_poison_class_failure_names_this_boundary(self):
        calls = []
        c = _controller(calls, write_raises=RuntimeError(IMA))
        HiCacheController.quiesce_device_io(c, "phase flip pp_to_tp")
        record = barlink_abort_gate.poison_record()
        self.assertIsNotNone(record)
        self.assertIn("#760 device-tier quiesce", record["source"])
        self.assertIn("write_stream", record["source"])

    def test_an_earlier_reporter_keeps_the_record(self):
        """FIRST-WINS. In both specimens the watchdog saw it first.

        A drain that overwrote the record would re-create exactly the
        misattribution #867 was built to end -- the newest site claiming the
        origin.
        """
        barlink_abort_gate.record_poison("barlink poll_status_word", RuntimeError(IMA))
        calls = []
        c = _controller(calls, write_raises=RuntimeError(IMA))
        HiCacheController.quiesce_device_io(c, "phase flip pp_to_tp")
        self.assertEqual(
            barlink_abort_gate.poison_record()["source"], "barlink poll_status_word"
        )

    def test_a_non_poison_failure_records_nothing(self):
        """A stream that failed for a survivable reason is not a context kill."""
        calls = []
        c = _controller(calls, load_raises=RuntimeError("CUDA out of memory"))
        HiCacheController.quiesce_device_io(c, "seam")
        self.assertIsNone(barlink_abort_gate.poison_record())


class TestTheOutcomeReachesTheSecondEmitter(unittest.TestCase):
    """The runtime prints its OWN success claim off this call's return value.

    `phase_flip_runtime._quiesce_hicache` emits the direction-tagged,
    prefix-carrying line -- the one a three-rank log is actually read by -- and
    a float return carries no outcome. Fixing only the controller's line would
    have left the more-read line still announcing a drain that did not happen.
    """

    def setUp(self):
        barlink_abort_gate.clear_poison_record()

    def tearDown(self):
        barlink_abort_gate.clear_poison_record()

    def test_a_clean_drain_publishes_an_empty_failure_set(self):
        calls = []
        c = _controller(calls)
        HiCacheController.quiesce_device_io(c, "seam")
        self.assertEqual(c.last_quiesce_failed, ())

    def test_a_failed_drain_publishes_which_streams_failed(self):
        calls = []
        c = _controller(calls, load_raises=RuntimeError(IMA))
        HiCacheController.quiesce_device_io(c, "seam")
        self.assertEqual(c.last_quiesce_failed, ("load_stream",))

    def _runtime_shell(self, controller):
        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        # The runtime reaches the drain through `getattr(controller,
        # "quiesce_device_io")`, so the fake has to expose the REAL method
        # bound to itself -- a fake that only carries streams makes the
        # runtime return 0.0 without logging anything, which is a green test
        # that exercised nothing.
        if not hasattr(controller, "quiesce_device_io"):
            controller.quiesce_device_io = lambda reason: (
                HiCacheController.quiesce_device_io(controller, reason)
            )
        rt = object.__new__(PhaseFlipRuntime)
        rt._census_scheduler = types.SimpleNamespace(
            tree_cache=types.SimpleNamespace(cache_controller=controller)
        )
        return rt

    def test_the_runtime_line_reports_the_failure_not_a_drain(self):
        calls = []
        c = _controller(calls, write_raises=RuntimeError(IMA))
        rt = self._runtime_shell(c)
        with self.assertLogs("sglang.srt.managers.phase_flip_runtime", "ERROR") as cm:
            rt._quiesce_hicache("pp_to_tp")
        joined = "\n".join(cm.output)
        self.assertIn("did NOT", joined)
        self.assertIn("write_stream", joined)
        self.assertNotIn("streams quiesced in", joined)

    def test_the_runtime_line_is_unchanged_on_a_clean_drain(self):
        calls = []
        c = _controller(calls)
        rt = self._runtime_shell(c)
        with self.assertLogs("sglang.srt.managers.phase_flip_runtime", "WARNING") as cm:
            rt._quiesce_hicache("tp_to_pp")
        joined = "\n".join(cm.output)
        self.assertIn("device-tier streams quiesced in", joined)
        self.assertNotIn("did NOT", joined)

    def test_a_controller_that_publishes_nothing_reads_as_clean(self):
        """An older controller, or one that never reached the publish.

        Absence of the reading is not evidence of failure -- and turning it
        into one would make every pre-#917 controller look broken.
        """
        rt = self._runtime_shell(
            types.SimpleNamespace(quiesce_device_io=lambda reason: 0.001)
        )
        with self.assertLogs("sglang.srt.managers.phase_flip_runtime", "WARNING") as cm:
            rt._quiesce_hicache("pp_to_tp")
        self.assertIn("device-tier streams quiesced in", "\n".join(cm.output))


class TestTheDrainCannotBeTheReasonAFlipDies(unittest.TestCase):
    """It runs past the no-return point. Every path must return, not raise."""

    def setUp(self):
        barlink_abort_gate.clear_poison_record()

    def tearDown(self):
        barlink_abort_gate.clear_poison_record()

    def test_a_missing_stream_is_skipped_not_faulted(self):
        calls = []
        c = types.SimpleNamespace(write_stream=None, load_stream=_Stream(calls, "load"))
        HiCacheController.quiesce_device_io(c, "seam")
        self.assertEqual(calls, ["load"])

    def test_an_unreachable_poison_gate_does_not_escape(self):
        calls = []
        c = _controller(calls, write_raises=RuntimeError(IMA))
        original = barlink_abort_gate.is_poison_error
        barlink_abort_gate.is_poison_error = lambda exc: (_ for _ in ()).throw(
            RuntimeError("gate unavailable")
        )
        try:
            elapsed = HiCacheController.quiesce_device_io(c, "seam")  # must not raise
        finally:
            barlink_abort_gate.is_poison_error = original
        self.assertGreaterEqual(elapsed, 0.0)

    def test_the_helper_is_a_module_function_not_a_method(self):
        """The harness shape this must survive.

        `quiesce_device_io` is exercised unbound against a `SimpleNamespace`
        (`test_flip_seam_guard_760.py`). A `self.`-looked-up helper would raise
        `AttributeError` out of an exception handler on the no-return path --
        turning an instrument into the thing that kills the flip.
        """
        from sglang.srt.managers import cache_controller

        self.assertTrue(callable(cache_controller._record_quiesce_poison))
        self.assertFalse(
            hasattr(HiCacheController, "_record_quiesce_poison"),
            "the helper must not be reachable through `self`",
        )


if __name__ == "__main__":
    unittest.main()
