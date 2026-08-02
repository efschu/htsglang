"""#431 -- the BAR1 deadline reaches the kernel, and its expiry is loud.

The 2026-08-02 repro window
(``docs/dev/ANALYSE_431_fp8_bar1_dcp_deadlock.md``) killed the dispatch-
divergence theory for the fp8/BAR1/uneven-weighted-DCP arm: all three ranks
recorded byte-identical ``(op, nbytes, path, rounds)`` sequences while the run
crawled at one collective per ~30-40 s. What it exposed instead were two
code-level defects, and this file is the falsifier for both plus the logging
gap that made the third reading unfalsifiable.

1. THE DEADLINE NEVER GREW. ``barlink_liveness`` has documented since #312
   that the BAR1 device cap "is multiplied by up to 40x inside the JIT
   cold-build window". ``barlink_device.py:1239`` does that.
   ``barlink_bar1.py`` passed ``int(self.cap_cycles)`` raw at all three of its
   launch sites and never called ``resolve_timeout_cycles`` at all -- so the
   one transport whose kernels spin on a device deadline was the one
   transport that did not extend it during the window the mechanism exists
   for. The window was open for the entire 22-minute stall.

2. THE EXPIRY WAS SILENT. A tripped kernel writes ``ctlStatus`` and returns,
   leaving its output buffer partially written. Nothing on a production path
   read that word -- ``raise_if_aborted`` was reachable only from three
   bring-up proofs -- so ``abort_fp8_bar1_decode.txt`` is EMPTY for a run in
   which, by the measured rate, essentially every collective tripped.

3. OPEN WITHOUT CLOSE. The window's accounting read 6 OPEN / 0 CLOSE, which
   looks exactly like a leak. ``cold_build_window`` had simply never logged a
   close line, so zero was the only number that grep could return.

CPU only: no CUDA context, no process group, no transport bring-up. The
transports are ``__new__`` stand-ins carrying the REAL methods under test.
"""

import os
import unittest
from unittest import mock

import torch

from sglang.srt.distributed.device_communicators import barlink_abort_gate
from sglang.srt.distributed.device_communicators.barlink import BarlinkCommunicator
from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    Bar1CollectiveAborted,
    BarlinkBar1Transport,
)
from sglang.srt.utils import jit_cold_build
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

#: The shipped default, spelled out rather than read from the module: the
#: point of the assertions below is the RELATION between the raw constant and
#: what the kernel receives, and a test that read both from the same source
#: could not tell them apart.
CAP = 60_000_000_000


def _transport(*, cap=CAP, world=3, rank=0, group="dcp:0", aborted=0):
    """A BAR1 transport carrying the real deadline and abort methods.

    ``__new__`` on purpose: ``_deadline_cycles`` and ``check_aborted`` are the
    code under test and must not be re-implemented here, but constructing the
    transport would map BAR1 apertures. Only the attributes those two methods
    (and the round planners they call at raise time) actually read are set.
    """
    t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
    t.cap_cycles = cap
    t.rank = rank
    t.world = world
    t.group = group
    t.threads = 256
    t.load_shape = 2
    t.read_flush = 0
    t.grid_from = 4 << 20
    t.graph_grid = False
    t._graph_grid_reported = True
    t._geo = {"chunk_max": 1 << 20, "a2a_slot": 1 << 20, "off_mesh": 0,
              "off_ring": 0}
    t._peers = {}
    t._own = (0x1000, 0, 0)
    t._own_flag = (0x2000, 0, 0)
    t._round_dev = torch.zeros(1, dtype=torch.int64)
    t._ctl_dev = torch.tensor([aborted, 0], dtype=torch.int32)
    t._abort_window = None
    t._last_op = ""
    t._last_nbytes = 0
    t._unchecked_launches = 0
    t._captured_launches = False
    t._registered_in_gate = False
    t._up = True
    return t


class _RecordingExt:
    """Stands in for the JIT extension: records the launch arguments."""

    def __init__(self):
        self.calls = []

    def bar1_all_reduce(self, *args):
        self.calls.append(args)

    @property
    def cap_of_last_call(self):
        # Positional layout of bar1_all_reduce, counted off the call site in
        # barlink_bar1._all_reduce_one_round: the cycle budget is the
        # argument right after (round_dev, ctl_dev).
        return self.calls[-1][14]


class TestDeadlineReachesTheKernel(CustomTestCase):
    """#431 fix 1. Can-fail: unfixed, every assertion below reads ``CAP``."""

    def test_outside_the_window_the_deadline_is_the_bare_constant(self):
        """The default and steady-state paths must stay byte-identical."""
        self.assertFalse(jit_cold_build.in_cold_build_window())
        self.assertEqual(_transport()._deadline_cycles(), CAP)

    def test_inside_the_window_the_deadline_is_multiplied(self):
        t = _transport()
        with jit_cold_build.cold_build_window("test"):
            self.assertEqual(
                t._deadline_cycles(),
                CAP * jit_cold_build.cold_build_timeout_mult(),
            )
        self.assertEqual(t._deadline_cycles(), CAP)

    def test_the_multiplier_is_honoured_from_the_environment(self):
        t = _transport()
        with mock.patch.dict(
            os.environ, {"SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT": "7"}, clear=False
        ):
            with jit_cold_build.cold_build_window("test"):
                self.assertEqual(t._deadline_cycles(), CAP * 7)

    def test_the_all_reduce_launch_carries_the_extended_budget(self):
        """The behavioural falsifier: what the KERNEL ARGUMENT actually is.

        Drives the real ``_all_reduce_one_round`` with a recording stand-in
        for the extension. Unfixed, the recorded budget is ``CAP`` both
        inside and outside the window -- that is the whole defect -- and this
        test fails on the second assertion.
        """
        t = _transport()
        t._ext = _RecordingExt()
        t.algorithm_for = lambda nbytes: "mesh"
        t._kernel = lambda moved, threshold, where: 0
        inp = torch.zeros(64, dtype=torch.float32)

        t._all_reduce_one_round(inp)
        self.assertEqual(t._ext.cap_of_last_call, CAP)

        with jit_cold_build.cold_build_window("cuda-graph capture warmup"):
            t._all_reduce_one_round(inp)
        self.assertEqual(
            t._ext.cap_of_last_call,
            CAP * jit_cold_build.cold_build_timeout_mult(),
            "the BAR1 kernel must receive the extended deadline while the "
            "cold-build window is open -- this is #431 fix 1",
        )

    def test_no_launch_site_passes_the_raw_cap_any_more(self):
        """The structural half, for the two sites a CPU test cannot launch.

        ``bar1_mesh_pipe`` and ``bar1_all_to_all`` need a mapped window and
        real peer pointers to drive, so their fix is pinned at the source:
        the raw ``int(self.cap_cycles)`` argument must not appear at a launch
        site any more. Unfixed there are three such lines.
        """
        import inspect

        from sglang.srt.distributed.device_communicators import barlink_bar1

        src = inspect.getsource(barlink_bar1)
        self.assertNotIn("int(self.cap_cycles), int(self.threads)", src)
        # ...and all three sites reach the resolver instead.
        self.assertEqual(
            src.count("self._deadline_cycles(), int(self.threads)"), 3
        )


class TestTrippedKernelIsLoud(CustomTestCase):
    """#431 fix 2. Can-fail: unfixed, ``check_aborted`` does not exist."""

    def setUp(self):
        barlink_abort_gate.reset_for_test()

    def tearDown(self):
        barlink_abort_gate.reset_for_test()

    def test_a_clean_status_word_raises_nothing(self):
        t = _transport(aborted=0)
        t._note_launch("all_reduce", 4096)
        t.check_aborted("unit test")

    def test_nothing_launched_means_nothing_is_read(self):
        """The free path at a boundary: no traffic, no device read."""
        t = _transport(aborted=1)

        def _boom():
            raise AssertionError("status() must not be read with no launches")

        t.status = _boom
        t.check_aborted("boundary with no traffic")

    def test_a_tripped_kernel_raises_with_rank_op_and_rounds(self):
        t = _transport(aborted=1, rank=2, world=3, group="dcp:0")
        t._note_launch("all_reduce", 4 << 20)
        with self.assertRaises(Bar1CollectiveAborted) as cm:
            t.check_aborted("all_reduce on group dcp:0")
        exc = cm.exception
        self.assertEqual(exc.rank, 2)
        self.assertEqual(exc.op, "all_reduce")
        self.assertEqual(exc.nbytes, 4 << 20)
        self.assertEqual(exc.rounds, t.ar_rounds(4 << 20))
        self.assertEqual(exc.launches, 1)
        text = str(exc)
        self.assertIn("rank 2/3", text)
        self.assertIn("dcp:0", text)
        self.assertIn("all_reduce", text)
        self.assertIn("partially written", text)
        # The effective budget is named, not just the constant: with fix 1 in
        # place those two numbers differ inside the cold-build window, and
        # the reader needs the one the kernel actually got.
        self.assertIn(str(CAP), text)

    def test_the_kill_switch_restores_the_silent_behaviour(self):
        t = _transport(aborted=1)
        t._note_launch("all_reduce", 4096)
        with mock.patch.dict(
            os.environ, {barlink_abort_gate.ENV_ENABLE: "0"}, clear=False
        ):
            t.check_aborted("kill switch set")

    def test_the_check_interval_is_honoured(self):
        """``..._CHECK_EVERY`` trades reporting latency for synchronizations."""
        t = _transport(aborted=1)
        with mock.patch.dict(
            os.environ, {barlink_abort_gate.ENV_EVERY: "3"}, clear=False
        ):
            t._note_launch("all_reduce", 4096)
            t.check_aborted("1st")
            t._note_launch("all_reduce", 4096)
            t.check_aborted("2nd")
            t._note_launch("all_reduce", 4096)
            with self.assertRaises(Bar1CollectiveAborted) as cm:
                t.check_aborted("3rd")
        self.assertEqual(cm.exception.launches, 3)


class TestCaptureSafety(CustomTestCase):
    """The design constraint: no host read of a device word inside a capture.

    Not a style rule. Reading ``ctlStatus`` is a D2H copy plus a stream
    synchronization; issued while the current stream is being captured it is
    illegal, and the failure would land on the graph runner rather than here.
    """

    def setUp(self):
        barlink_abort_gate.reset_for_test()

    def tearDown(self):
        barlink_abort_gate.reset_for_test()

    def _capturing(self, value):
        return mock.patch(
            "sglang.srt.distributed.device_communicators.barlink."
            "graph_capture_running",
            return_value=value,
        )

    def test_no_device_read_happens_under_capture(self):
        t = _transport(aborted=1)
        t._unchecked_launches = 5  # as if the counter had somehow advanced

        def _boom():
            raise AssertionError("no host-side device read inside a capture")

        t.status = _boom
        with self._capturing(True):
            t.check_aborted("during capture")

    def test_a_captured_launch_does_not_advance_the_host_counter(self):
        """A recorded kernel has not RUN; only a replay runs it."""
        t = _transport()
        with self._capturing(True):
            t._note_launch("all_reduce", 4096)
        self.assertEqual(t._unchecked_launches, 0)
        self.assertTrue(t._captured_launches)

    def test_the_replay_boundary_is_what_covers_a_captured_kernel(self):
        t = _transport(aborted=1)
        with self._capturing(True):
            t._note_launch("all_reduce", 4096)
        barlink_abort_gate.register(t)
        with self.assertRaises(Bar1CollectiveAborted):
            barlink_abort_gate.check_after_graph_replay()

    def test_the_replay_check_is_separately_switchable(self):
        t = _transport(aborted=1)
        with self._capturing(True):
            t._note_launch("all_reduce", 4096)
        barlink_abort_gate.register(t)
        with mock.patch.dict(
            os.environ, {barlink_abort_gate.ENV_REPLAY: "0"}, clear=False
        ):
            barlink_abort_gate.check_after_graph_replay()

    def test_an_empty_registry_is_the_whole_default_path(self):
        """No BAR1 transport in the process: one truth test, nothing else."""
        with mock.patch.object(
            barlink_abort_gate, "abort_check_enabled"
        ) as enabled:
            barlink_abort_gate.check_after_graph_replay()
            barlink_abort_gate.check_aborts("boundary")
            enabled.assert_not_called()

    def test_both_graph_backends_check_at_their_replay_boundary(self):
        """Pinned at the source: importing the backends needs a CUDA build.

        A replayed graph runs the BAR1 kernels with no host code between
        them, so if this call is ever dropped from a ``replay()`` the
        captured decode path goes silent again -- exactly the pre-#431 state.
        """
        import pathlib

        root = (
            pathlib.Path(__file__).resolve().parents[4]
            / "python/sglang/srt/model_executor/runner_backend"
        )
        for name in ("full_cuda_graph_backend.py", "breakable_cuda_graph_backend.py"):
            src = (root / name).read_text()
            self.assertIn("barlink_abort_gate.check_after_graph_replay()", src, name)


class TestTheDispatchSitesActuallyCheck(CustomTestCase):
    """The wiring. A check nothing calls is a comment (#431 fix 2).

    Drives the real ``BarlinkCommunicator`` collectives against a transport
    whose ``check_aborted`` raises, and asserts the exception escapes the
    collective. Unfixed, every one of these returns the result of a
    collective whose output buffer is partially written.
    """

    class _Aborting:
        BARLINK_OPS = frozenset(
            {"all_reduce", "all_gather", "reduce_scatter", "broadcast"}
        )

        def handles(self, op, nbytes):
            return True

        def name(self):
            return "bar1"

        def check_aborted(self, where):
            raise Bar1CollectiveAborted(f"tripped, seen at {where}", op="all_reduce")

        def barlink_all_reduce(self, comm, inp):
            return inp

        def barlink_all_gather(self, comm, inp, dim=-1):
            return inp

        def barlink_reduce_scatter(self, comm, inp, dim=-1):
            return inp

        def barlink_broadcast(self, comm, tensor, src=0):
            return tensor

    def _comm(self):
        c = BarlinkCommunicator.__new__(BarlinkCommunicator)
        c.transport = self._Aborting()
        c.group = "dcp:0"
        c.disabled = False
        c.rank = 0
        c.world_size = 3
        c._path_dispatcher = None
        c._fallback_reported = set()
        return c

    def test_every_transport_collective_is_followed_by_the_check(self):
        comm = self._comm()
        x = torch.zeros(64, dtype=torch.float32)
        for call in (
            lambda: comm.all_reduce(x),
            lambda: comm.all_gather(x),
            lambda: comm.reduce_scatter(x),
            lambda: comm.broadcast(x),
        ):
            with self.assertRaises(Bar1CollectiveAborted):
                call()

    def test_a_transport_without_the_method_is_left_alone(self):
        """gloo, shm, device, ucx: none of them has a device deadline."""

        class _Plain(self._Aborting):
            check_aborted = None

        comm = self._comm()
        comm.transport = _Plain()
        comm.transport.check_aborted = None
        x = torch.zeros(64, dtype=torch.float32)
        self.assertIs(comm.all_reduce(x), x)


class TestColdBuildWindowLogsItsClose(CustomTestCase):
    """#431 fix 3. Can-fail: unfixed there is no close line to find.

    The repro window's ``jit/coldwindow_*.txt`` was collected with
    ``grep -aE 'JIT cold-build window (open|close)'`` and read 6 OPEN /
    0 CLOSE across a 22-minute stall. That reads as a leaked window; in fact
    ``cold_build_window`` logged only one direction, so the count could never
    have been anything else. An instrument that can only count opens cannot
    falsify a leak.
    """

    LOGGER = "sglang.srt.utils.jit_cold_build"

    def test_open_and_close_are_both_logged(self):
        with self.assertLogs(self.LOGGER, level="INFO") as cm:
            with jit_cold_build.cold_build_window("unit test"):
                pass
        text = "\n".join(cm.output)
        self.assertIn("JIT cold-build window open (unit test)", text)
        self.assertIn("JIT cold-build window close (unit test)", text)

    def test_the_close_is_logged_even_when_the_block_raises(self):
        with self.assertLogs(self.LOGGER, level="INFO") as cm:
            with self.assertRaises(ValueError):
                with jit_cold_build.cold_build_window("failing block"):
                    raise ValueError("boom")
        self.assertIn(
            "JIT cold-build window close (failing block)", "\n".join(cm.output)
        )

    def test_nesting_logs_exactly_one_open_and_one_close(self):
        """Otherwise the counts would not balance for a re-entrant window."""
        with self.assertLogs(self.LOGGER, level="INFO") as cm:
            with jit_cold_build.cold_build_window("outer"):
                with jit_cold_build.cold_build_window("inner"):
                    pass
        text = "\n".join(cm.output)
        self.assertEqual(text.count("JIT cold-build window open"), 1)
        self.assertEqual(text.count("JIT cold-build window close"), 1)
        self.assertFalse(jit_cold_build.in_cold_build_window())


if __name__ == "__main__":
    unittest.main()
