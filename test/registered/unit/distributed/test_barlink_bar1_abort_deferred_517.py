"""#517 -- the BAR1 abort guard gets cheap without going blind.

The #476 window
(``/spinning/gpu-battery-results/2026-08-03_w4_t2_476_bar1_floor/RESULTS.md``)
measured what the #431 guard costs on the decode path: -9.22 % on code
decode_TPS against the same-tree NCCL baseline, split 6.64 pp on Seam A (the
host-path collectives) and 5.26 pp on Seam B (the CUDA-graph replay
boundary). Removing both seams reproduces #424's pre-#431 BAR1 advantage of
+2.5 %. The same window's A2 arm CRASHED with a real, intermittent
``Bar1CollectiveAborted`` -- a spin kernel took its abort path, observed at a
replay boundary, last collective ``all_to_all`` (8 bytes, 0 rounds) -- so the
guard is not optional and a cheaper guard that cannot see that event is a
regression, not an optimization.

What this file pins:

1. THE SHAPE OF THE CHEAPENING. The status read is STAGED: a non-blocking
   D2H onto the current stream plus a ``cudaEventQuery``, reading the value
   an earlier check staged. ``status()`` -- the blocking ``.item()`` -- is
   never called on the deferred path.

2. THE §3 EVENT IS STILL CAUGHT. Word set, abort observed at a replay
   boundary, 8-byte ``all_to_all``: the guard raises, with the same
   structured attributes the crash line carried.

3. THE BOUND IS LOAD-BEARING. A deferred read whose event never resolves is
   the naive version of this cheapening, and it is blind. With
   ``..._ABORT_MAX_LAG`` raised out of the way the guard misses the abort
   for as long as the test cares to look; with the shipped default it forces
   one wait and raises. That is the binds-proof for the default.

4. THE DEFAULT PATH IS UNTOUCHED. A CPU status word does not defer at all
   (there is no synchronization to avoid), ``..._ABORT_DEFER=0`` restores the
   pre-#517 blocking read exactly, and ``..._CHECK_EVERY=1`` still checks at
   every boundary.

CPU only: no CUDA context, no process group, no transport bring-up. The
transport is a ``__new__`` stand-in carrying the REAL methods under test.
"""

import os
import unittest
from unittest import mock

import torch

os.environ.setdefault("SGLANG_BARLINK_BAR1_ABORT_SYNC_DEADLINE_MS", "0")

from sglang.srt.distributed.device_communicators import barlink_abort_gate
from sglang.srt.distributed.device_communicators.barlink import BarlinkCommunicator
from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    Bar1CollectiveAborted,
    BarlinkBar1Transport,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ScriptedEvent:
    """Stands in for ``torch.cuda.Event`` with a scripted completion.

    ``query()`` is what the deferred read costs in the steady state, and
    ``synchronize()`` is the wait it exists to avoid -- so both are counted,
    and a test that claims "no synchronization happened" asserts on the
    counter rather than on the absence of a symptom.
    """

    def __init__(self, ready_after: int = 0, never: bool = False):
        #: queries that must pass before the copy reports complete
        self.ready_after = ready_after
        self.never = never
        self.queries = 0
        self.records = 0
        self.syncs = 0

    def record(self, *args, **kwargs):
        self.records += 1

    def query(self) -> bool:
        if self.syncs > 0:
            return True
        self.queries += 1
        if self.never:
            return False
        return self.queries > self.ready_after

    def synchronize(self):
        self.syncs += 1


def _transport(*, aborted=0, world=3, rank=0, group="tp:0", defer=True, event=None):
    """A BAR1 transport carrying the real abort methods, staged read armed.

    ``__new__`` on purpose: ``check_aborted`` / ``_read_status_for_check``
    are the code under test and must not be re-implemented here, but
    constructing the transport would map BAR1 apertures. The staged read is
    wired by hand to exactly what ``_arm_status_stage`` builds on a CUDA
    word -- a persistent 1-element view, a host destination, an event --
    with the event scripted so a CPU-only test can drive the in-flight case
    at all.
    """
    t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
    t.cap_cycles = 60_000_000_000
    t.rank = rank
    t.world = world
    t.group = group
    t.threads = 256
    t.load_shape = 2
    t.read_flush = 0
    t.grid_from = 4 << 20
    t.graph_grid = False
    t._graph_grid_reported = True
    t._geo = {"chunk_max": 1 << 20, "a2a_slot": 1 << 20, "off_mesh": 0, "off_ring": 0}
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
    t._ctl_defer = bool(defer)
    t._ctl_src = t._ctl_dev[0:1]
    t._ctl_stage = torch.zeros(1, dtype=torch.int32)
    t._ctl_event = event if event is not None else _ScriptedEvent()
    t._ctl_inflight = False
    t._ctl_lag = 0
    t._deferred_launches = 0
    t._boundary_checks = 0
    # --- missing from original builder, defaults from real __init__ ---
    t._last_op_captured = False         # barlink_bar1.py:1606, default False
    t._abort_poll_stream = None         # barlink_bar1.py:1658, default None
    t._abort_poll_dst = None            # barlink_bar1.py:1659, default None
    t._abort_poll_active = False        # barlink_bar1.py:1660, default False
    t._ctl_sync_timeouts = 0           # barlink_bar1.py:1664, default 0
    t._abort_code_seen = 0             # barlink_bar1.py:1661, default 0
    t._ctl_stall_run = 0               # barlink_bar1.py:1668, default 0
    t._ctl_build_deferred_s = 0.0      # barlink_bar1.py:1675, default 0.0
    t._up = True
    return t


def _trip(t):
    """What a spin kernel does when it exceeds its deadline: one sticky bit."""
    t._ctl_dev[0] = 1


def _boundary(t, where="cuda-graph replay"):
    """One CUDA-graph replay boundary check on this transport."""
    t.check_aborted(where)


class TestTheStagedReadNeverBlocks(CustomTestCase):
    """Seam B's cost was the synchronization, so the synchronization goes."""

    def setUp(self):
        barlink_abort_gate.reset_for_test()

    def tearDown(self):
        barlink_abort_gate.reset_for_test()

    def test_the_deferred_path_never_calls_the_blocking_read(self):
        ev = _ScriptedEvent()
        t = _transport(event=ev)
        t._captured_launches = True

        def _boom():
            raise AssertionError(
                "status() is the blocking .item(); the staged path must not "
                "call it"
            )

        t.status = _boom
        for _ in range(10):
            _boundary(t)
        self.assertEqual(ev.syncs, 0, "no stream synchronization in steady state")
        self.assertGreater(ev.records, 0, "the staged copy is actually issued")

    def test_a_boundary_with_no_traffic_is_still_completely_free(self):
        """The pre-#431 free path is unchanged: nothing launched, nothing read."""
        ev = _ScriptedEvent()
        t = _transport(aborted=1, event=ev)
        _boundary(t)
        self.assertEqual(ev.records, 0)
        self.assertEqual(ev.queries, 0)

    def test_one_staged_copy_is_in_flight_at_a_time(self):
        """A second copy is not issued while the first is unresolved."""
        ev = _ScriptedEvent(never=True)
        t = _transport(event=ev)
        t._captured_launches = True
        with mock.patch.dict(
            os.environ, {barlink_abort_gate.ENV_MAX_LAG: "1000000"}, clear=False
        ):
            for _ in range(6):
                _boundary(t)
        self.assertEqual(ev.records, 1)


class TestTheDeferredGuardStillSeesTheCrash(CustomTestCase):
    """#476 §3, reconstructed: the event the guard exists for."""

    def setUp(self):
        barlink_abort_gate.reset_for_test()

    def tearDown(self):
        barlink_abort_gate.reset_for_test()

    def test_the_476_section_3_abort_raises_at_the_next_boundary(self):
        """Word set during a replayed graph, 8-byte all_to_all, tp:0.

        The staged copy issued at the boundary AFTER the aborting graph
        observes the word (it is ordered behind that graph on the same
        stream); the check that follows reads it. One boundary of latency,
        and the sticky bit means nothing is lost in between.
        """
        ev = _ScriptedEvent()
        t = _transport(rank=0, world=3, group="tp:0", event=ev)
        with mock.patch(
            "sglang.srt.distributed.device_communicators.barlink."
            "graph_capture_running",
            return_value=True,
        ):
            t._note_launch("all_to_all", 8)
        self.assertTrue(t._captured_launches)
        barlink_abort_gate.register(t)

        # Boundary of a clean replay: stages a zero.
        barlink_abort_gate.check_after_graph_replay()
        # The spin kernel of the NEXT replayed graph takes its abort path.
        _trip(t)
        # That graph's boundary: the staged copy issued here sees the bit.
        barlink_abort_gate.check_after_graph_replay()
        with self.assertRaises(Bar1CollectiveAborted) as cm:
            barlink_abort_gate.check_after_graph_replay()

        exc = cm.exception
        self.assertEqual(exc.rank, 0)
        self.assertEqual(exc.world, 3)
        self.assertEqual(exc.group, "tp:0")
        self.assertEqual(exc.op, "all_to_all")
        self.assertEqual(exc.nbytes, 8)
        text = str(exc)
        self.assertIn("rank 0/3", text)
        self.assertIn("a spin kernel took its abort path", text)
        self.assertIn("observed at cuda-graph replay", text)
        self.assertIn("all_to_all (8 bytes", text)
        # #583 reworded this. The old text was "0 collective(s) ran since the
        # previous check, so the abort is in that window and the named one is
        # its most recent member" -- which is false exactly here: nothing ran
        # on the host path, so the named collective is a graph-capture
        # artefact and not a member of anything. The empty window must now
        # say so, and must not claim membership.
        self.assertIn("No collective ran on the host path", text)
        self.assertIn("GRAPH-REPLAY window", text)
        self.assertNotIn("its most recent member", text)
        # The staged read announces itself, so a post-mortem does not read
        # the boundary it names as the boundary it happened on.
        self.assertIn("STAGED", text)

    def test_a_host_path_collective_window_is_accumulated_not_lost(self):
        """Collectives that ran while the value was in flight are named.

        With a blocking read the window is always the current one. With a
        staged read the checks whose value had not arrived yet still ran
        collectives, and the raise has to name all of them or it understates
        which buffers are suspect.
        """
        ev = _ScriptedEvent(ready_after=2)
        t = _transport(aborted=1, event=ev)
        with self.assertRaises(Bar1CollectiveAborted) as cm:
            for _ in range(8):
                t._note_launch("all_reduce", 4096)
                t.check_aborted("all_reduce")
        self.assertGreater(cm.exception.launches, 1)


class TestTheLagBoundIsLoadBearing(CustomTestCase):
    """The naive cheapening, and why the shipped default is not it.

    "Read it later" without a bound is not a guard: an overlap-scheduled
    host queues work faster than the device drains it, and a staged value
    can stay in flight for as long as that lasts. ``..._ABORT_MAX_LAG`` is
    the bound, and this is its can-fail proof.
    """

    def setUp(self):
        barlink_abort_gate.reset_for_test()

    def tearDown(self):
        barlink_abort_gate.reset_for_test()

    def test_without_the_bound_the_deferred_read_is_blind(self):
        ev = _ScriptedEvent(never=True)
        t = _transport(aborted=1, event=ev)
        t._captured_launches = True
        with mock.patch.dict(
            os.environ, {barlink_abort_gate.ENV_MAX_LAG: "1000000"}, clear=False
        ):
            for _ in range(200):
                _boundary(t)  # 200 boundaries, tripped word, not one raise
        self.assertEqual(ev.syncs, 0)

    def test_with_the_shipped_default_the_bound_forces_the_read(self):
        ev = _ScriptedEvent(never=True)
        t = _transport(aborted=1, event=ev)
        t._captured_launches = True
        self.assertEqual(barlink_abort_gate.max_lag(), 4)
        with self.assertRaises(Bar1CollectiveAborted):
            for _ in range(barlink_abort_gate.max_lag() + 2):
                _boundary(t)
        self.assertEqual(ev.syncs, 1, "exactly one forced wait, then the raise")

    def test_the_bound_is_configurable_and_lower_is_stricter(self):
        ev = _ScriptedEvent(never=True)
        t = _transport(aborted=1, event=ev)
        t._captured_launches = True
        with mock.patch.dict(
            os.environ, {barlink_abort_gate.ENV_MAX_LAG: "1"}, clear=False
        ):
            _boundary(t)  # issues the copy
            with self.assertRaises(Bar1CollectiveAborted):
                _boundary(t)  # unresolved once -> forced wait -> raise

    def test_a_non_integer_bound_falls_back_to_the_default(self):
        with mock.patch.dict(
            os.environ, {barlink_abort_gate.ENV_MAX_LAG: "soon"}, clear=False
        ):
            self.assertEqual(barlink_abort_gate.max_lag(), 4)
        with mock.patch.dict(
            os.environ, {barlink_abort_gate.ENV_MAX_LAG: "0"}, clear=False
        ):
            self.assertEqual(barlink_abort_gate.max_lag(), 1)


class TestCheckEveryNowReachesTheReplayBoundary(CustomTestCase):
    """#476 §4 candidate 2 / TICKET_476 §1.4: the knob that did not reach.

    Before #517 a replay boundary was entered with ``_unchecked_launches ==
    0``, i.e. through ``_captured_launches``, BELOW the interval test -- so
    the documented latency knob throttled Seam A only and Seam B synced
    every single time. Unfixed, the first boundary below raises.
    """

    def setUp(self):
        barlink_abort_gate.reset_for_test()

    def tearDown(self):
        barlink_abort_gate.reset_for_test()

    def test_k_greater_than_one_throttles_the_boundary(self):
        t = _transport(aborted=1, defer=False)
        t._captured_launches = True
        with mock.patch.dict(
            os.environ, {barlink_abort_gate.ENV_EVERY: "5"}, clear=False
        ):
            for _ in range(4):
                _boundary(t)  # four boundaries, no device read at all
            with self.assertRaises(Bar1CollectiveAborted):
                _boundary(t)

    def test_the_default_still_checks_at_every_boundary(self):
        """K = 1 is behaviour-identical to pre-#517."""
        t = _transport(aborted=1, defer=False)
        t._captured_launches = True
        with self.assertRaises(Bar1CollectiveAborted):
            _boundary(t)


class TestTheDefaultAndTheKillSwitches(CustomTestCase):
    """Every cheapening has to be reversible by name."""

    def setUp(self):
        barlink_abort_gate.reset_for_test()

    def tearDown(self):
        barlink_abort_gate.reset_for_test()

    def test_defer_off_restores_the_blocking_read_exactly(self):
        t = _transport(aborted=1, defer=False)
        t._note_launch("all_reduce", 4096)
        reads = []
        real = t.status

        def _counting():
            reads.append(1)
            return real()

        t.status = _counting
        with self.assertRaises(Bar1CollectiveAborted) as cm:
            t.check_aborted("all_reduce")
        self.assertEqual(len(reads), 1, "one blocking read, at the first check")
        self.assertNotIn("STAGED", str(cm.exception))

    def test_a_host_status_word_never_defers(self):
        """The arming predicate, at its single definition.

        Deferring exists to avoid a device synchronization. A word that is
        not on a device has none to avoid, so reading it directly is both
        cheaper AND stricter -- which is also why every pre-#517 hermetic
        test keeps its exact meaning.
        """
        self.assertFalse(barlink_abort_gate.should_defer_status(False, True))
        self.assertFalse(barlink_abort_gate.should_defer_status(True, False))
        self.assertTrue(barlink_abort_gate.should_defer_status(True, True))

    def test_arming_is_driven_by_that_predicate_and_by_nothing_else(self):
        """``_arm_status_stage`` asks the predicate; a CPU word gets nothing."""
        t = _transport(defer=False)
        t._ctl_src = None
        t._ctl_stage = None
        t._ctl_event = None
        t._arm_status_stage()
        self.assertFalse(t._ctl_defer)
        self.assertIsNone(t._ctl_stage)

    def test_defer_env_default_is_on_and_switchable(self):
        self.assertTrue(barlink_abort_gate.defer_enabled())
        with mock.patch.dict(
            os.environ, {barlink_abort_gate.ENV_DEFER: "0"}, clear=False
        ):
            self.assertFalse(barlink_abort_gate.defer_enabled())

    def test_the_master_kill_switch_still_wins(self):
        t = _transport(aborted=1)
        t._captured_launches = True
        with mock.patch.dict(
            os.environ, {barlink_abort_gate.ENV_ENABLE: "0"}, clear=False
        ):
            for _ in range(5):
                _boundary(t)


class TestTheLabelIsNoLongerBuiltEagerly(CustomTestCase):
    """#476 §4 candidate 4. Unfixed, the label is an f-string per collective."""

    class _Recording:
        BARLINK_OPS = frozenset(
            {"all_reduce", "all_gather", "reduce_scatter", "broadcast"}
        )

        def __init__(self):
            self.labels = []

        def handles(self, op, nbytes):
            return True

        def name(self):
            return "bar1"

        def check_aborted(self, where):
            self.labels.append(where)

        def barlink_all_reduce(self, comm, inp):
            return inp

        def barlink_all_gather(self, comm, inp, dim=-1):
            return inp

        def barlink_reduce_scatter(self, comm, inp, dim=-1):
            return inp

        def barlink_broadcast(self, comm, tensor, src=0):
            return tensor

    def _comm(self, transport):
        c = BarlinkCommunicator.__new__(BarlinkCommunicator)
        c.transport = transport
        c.group = "tp:0"
        c.disabled = False
        c.rank = 0
        c.world_size = 3
        c._path_dispatcher = None
        c._fallback_reported = set()
        return c

    def test_the_label_is_the_bare_op_from_the_call_site(self):
        t = self._Recording()
        comm = self._comm(t)
        x = torch.zeros(64, dtype=torch.float32)
        comm.all_reduce(x)
        comm.all_gather(x)
        comm.broadcast(x)
        self.assertEqual(t.labels, ["all_reduce", "all_gather", "broadcast"])
        for label in t.labels:
            self.assertNotIn("group", label)

    def test_the_group_is_still_in_the_raised_message(self):
        """Nothing is lost: the transport names its own group at raise time."""
        t = _transport(aborted=1, group="dcp:0", defer=False)
        t._note_launch("broadcast", 8)
        with self.assertRaises(Bar1CollectiveAborted) as cm:
            t.check_aborted("broadcast")
        self.assertIn("group dcp:0", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
