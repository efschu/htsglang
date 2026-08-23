"""#818 -- a dead peer must end the wait, not extend it.

THE INCIDENT. On 2026-08-22/23 rank PP1 died of a device-side assert three
times. Each time the two SURVIVORS stayed in barlink's post-transport abort
gate, and the instance was lost -- not to the assert, but to the wait:

    _wait_ctl_event        barlink_bar1.py
    _read_status_for_check barlink_bar1.py
    check_aborted          barlink_bar1.py
    _after_transport       barlink.py
    all_reduce / all_gather

in every usable py-spy generation, while PP1 appeared in none. The only line
the survivors emitted said the compute stream "has not retired the copy" --
true, and useless, because the reason it had not was that the peer was dead.

WHY THE EXISTING LADDER COULD NOT CATCH IT. Two gates stand between an expiry
and an escalation, and a dead peer walks past both:

  * ``Bar1CollectiveStalled`` fires after N CONSECUTIVE expiries, and
    ``_ctl_stall_run`` is reset to 0 by EVERY resolved read. A dead peer
    produces INTERMITTENT expiries: the specimen logged 20 CUMULATIVE
    expiries in ~36 s and never escalated.
  * ``defer_stall_for_building_peer`` can extend the wait to
    ``SGLANG_BARLINK_BUILD_WINDOW_CAP_S`` (900 s) off a published build
    marker. A process that no longer exists is not building anything.

So the fix is not a shorter timeout. It is a DIFFERENT QUESTION, asked at the
three points that can wait: is the peer still there?

WHAT THIS FILE PINS, one case per call edge, each with a mutant that only it
kills:

 1. the expiry path raises ``Bar1PeerLost`` on the FIRST expiry with a dead
    peer, far below the consecutive-expiry ceiling;
 2. a LIVE peer with the same never-resolving event does NOT raise -- the
    other direction, without which case 1 proves only that the gate fires,
    not that it measures anything;
 3. the error names the peer (rank, pid, host); a message without identity
    passes case 1 and dies here;
 4. a published build window does NOT forgive a dead peer -- placing the
    check after ``defer_stall_for_building_peer`` passes case 1 and dies here;
 5. ``SGLANG_BARLINK_PEER_LIVENESS=0`` restores the previous behaviour
    exactly, so the escape hatch the message advertises actually exists;
 6. the deadline-0 branch, which blocks in the driver with no timeout at all,
    asks before it enters;
 7. with a long deadline the gate answers within the PROBE interval, not the
    deadline -- i.e. the in-loop probe edge exists and is not dead code.

CPU only: no CUDA context, no process group, no BAR1 aperture. The transport
is a ``__new__`` stand-in carrying the REAL methods under test, following
``test_barlink_bar1_abort_deferred_517.py``; the peer table is a REAL
``PeerTable`` over a REAL killed child process, following
``test_barlink_peer_liveness.py``.
"""

import multiprocessing
import os
import time
import unittest
from unittest import mock

import torch

from sglang.srt.distributed.device_communicators import barlink_liveness as live
from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    Bar1CollectiveStalled,
    Bar1PeerLost,
    BarlinkBar1Transport,
    raise_if_peer_lost,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _sleep_forever():
    time.sleep(3600)


class _ScriptedEvent:
    """``torch.cuda.Event`` stand-in with a scripted completion.

    ``never=True`` is the case that matters here: the staged copy never
    retires, which is exactly what a peer dying mid-collective leaves behind.
    ``syncs`` is counted because "did it enter the unbounded branch" is an
    assertion on a counter, not on the absence of a symptom.
    """

    def __init__(self, ready_after: int = 0, never: bool = False):
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


def _transport(*, world=3, rank=0, group="flip_tp:0", event=None, table=None):
    """A BAR1 transport carrying the real ``_wait_ctl_event``.

    ``__new__`` on purpose, for the reason the #517 suite records: the methods
    under test must not be re-implemented here, but constructing the transport
    would map BAR1 apertures.
    """
    t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
    t.rank = rank
    t.world = world
    t.group = group
    t._ctl_dev = torch.tensor([0, 0], dtype=torch.int32)
    t._abort_window = None
    t._last_op = "all_reduce"
    t._last_nbytes = 10485760
    t._ctl_defer = True
    t._ctl_src = t._ctl_dev[0:1]
    t._ctl_stage = torch.zeros(1, dtype=torch.int32)
    t._ctl_event = event if event is not None else _ScriptedEvent(never=True)
    t._ctl_inflight = False
    t._ctl_lag = 0
    t._deferred_launches = 0
    t._boundary_checks = 0
    t._unchecked_launches = 0
    t._captured_launches = False
    t._ctl_sync_timeouts = 0
    t._ctl_stall_run = 0
    t._ctl_build_deferred_s = 0.0
    t._expiry_census_fired = True  # the census is not what this file measures
    t._peer_table = table
    t._up = True
    return t


class AbortGateLivenessTest(CustomTestCase):
    """Every case drives the REAL ``_wait_ctl_event``."""

    def setUp(self):
        super().setUp()
        self._procs = []
        live.reset_for_test()
        # A short deadline keeps the suite fast; the DEFAULT is 2000 ms and
        # case 7 pins the behaviour that matters when it is long.
        self._env = mock.patch.dict(
            os.environ,
            {
                "SGLANG_BARLINK_BAR1_ABORT_SYNC_DEADLINE_MS": "40",
                "SGLANG_BARLINK_PEER_PROBE_S": "0.01",
                "SGLANG_BARLINK_PEER_LIVENESS": "1",
            },
        )
        self._env.start()

    def tearDown(self):
        self._env.stop()
        for p in self._procs:
            if p.is_alive():
                p.kill()
            p.join(timeout=5)
        live.reset_for_test()
        super().tearDown()

    # ---------------------------------------------------------------- helpers
    def _child(self):
        p = multiprocessing.Process(target=_sleep_forever, daemon=True)
        p.start()
        self._procs.append(p)
        return p

    def _table(self, *, dead: bool):
        """A real PeerTable whose rank-1 entry is a real child process.

        Killed for the dead case, left running for the live one, so the two
        directions differ ONLY in whether that pid still exists.
        """
        child = self._child()
        if dead:
            child.kill()
            child.join(timeout=5)
            # os.kill(pid, 0) can still succeed on a zombie; join() above
            # reaps it, and this asserts the precondition rather than
            # assuming it -- a "dead peer" test over a live pid measures
            # nothing.
            self.assertFalse(live.pid_alive(child.pid))
        else:
            self.assertTrue(live.pid_alive(child.pid))
        me = live.local_identity(0)
        peer = live.PeerIdentity(
            rank=1, host=me.host, pid=child.pid, boot=live._boot_marker(child.pid)
        )
        return live.PeerTable([me, peer], self_rank=0)

    # ------------------------------------------------------------------ cases
    def test_dead_peer_raises_on_the_first_expiry(self):
        """Case 1 -- the EXPIRY edge, isolated from the in-loop probe edge.

        The probe interval is deliberately set LONGER than the deadline, so
        the in-loop probe cannot fire and the only edge left is the one on the
        expiry path. Without this the two edges mask each other and neither
        mutant dies -- which is how the first version of this case failed:
        it asserted one expiry had happened while the loop probe had already
        raised at 10 ms, and the counter was 0.
        """
        t = _transport(table=self._table(dead=True))
        with mock.patch.dict(os.environ, {"SGLANG_BARLINK_PEER_PROBE_S": "10"}):
            with self.assertRaises(Bar1PeerLost):
                t._wait_ctl_event()
        # The point is not merely that it raised, but that it raised WITHOUT
        # needing the consecutive-expiry ceiling: one expiry was enough.
        self.assertEqual(t._ctl_sync_timeouts, 1)
        self.assertLess(t._ctl_stall_run, 30)

    def test_live_peer_does_not_raise_peer_lost(self):
        """Case 2 -- the other direction, and the reason case 1 means anything.

        Same never-resolving event, same code path, peer ALIVE. The gate must
        stay silent and hand control back to the existing stall ladder. A
        mutant that raises unconditionally passes case 1 and dies here.
        """
        t = _transport(table=self._table(dead=False))
        for _ in range(3):
            self.assertFalse(t._wait_ctl_event())
        self.assertEqual(t._ctl_sync_timeouts, 3)

    def test_error_names_the_dead_peer(self):
        """Case 3 -- identity. Kills: a message that says only 'peer lost'."""
        table = self._table(dead=True)
        dead_pid = table.dead_peers()[0].pid
        t = _transport(table=table)
        with self.assertRaises(Bar1PeerLost) as ctx:
            t._wait_ctl_event()
        text = str(ctx.exception)
        self.assertIn(str(dead_pid), text)
        self.assertIn("rank=1", text.replace("rank 1", "rank=1"))
        self.assertIn(live._hostname(), text)
        # the op and byte count the operator needs to compare across ranks
        self.assertIn("all_reduce", text)
        self.assertIn("10485760", text)
        self.assertEqual(ctx.exception.rank, 0)
        self.assertEqual(ctx.exception.group, "flip_tp:0")

    def test_build_window_cannot_forgive_a_dead_peer(self):
        """Case 4 -- ordering. Kills: the check placed AFTER the deferral.

        ``defer_stall_for_building_peer`` returns True here, which is what it
        would do for a peer that published a JIT build marker and then died.
        The old ladder would forgive the run and keep waiting; a dead peer
        must be fatal regardless.
        """
        t = _transport(table=self._table(dead=True))
        t._ctl_stall_run = 100  # already past any ceiling
        # Probe interval longer than the deadline, so this case reaches the
        # deferral's own gate on the EXPIRY path rather than being short-cut
        # by the in-loop probe -- otherwise it would pass for the wrong reason.
        with mock.patch.dict(os.environ, {"SGLANG_BARLINK_PEER_PROBE_S": "10"}):
            with mock.patch(
                "sglang.srt.distributed.device_communicators.barlink_bar1"
                ".defer_stall_for_building_peer",
                return_value=True,
            ) as deferral:
                with self.assertRaises(Bar1PeerLost):
                    t._wait_ctl_event()
        deferral.assert_not_called()
        self.assertEqual(t._ctl_sync_timeouts, 1)

    def test_liveness_disabled_restores_previous_behaviour(self):
        """Case 5 -- the escape hatch the error message advertises.

        With the knob off the dead peer is invisible to the gate and the old
        ceiling is what decides, so this also pins that #818 added no second
        way to fail.
        """
        t = _transport(table=self._table(dead=True))
        t._ctl_stall_run = 0
        with mock.patch.dict(os.environ, {"SGLANG_BARLINK_PEER_LIVENESS": "0"}):
            self.assertFalse(t._wait_ctl_event())
            # and the pre-existing escalation still works untouched
            t._ctl_stall_run = 10_000
            with self.assertRaises(Bar1CollectiveStalled):
                t._wait_ctl_event()

    def test_deadline_zero_branch_asks_before_blocking(self):
        """Case 6 -- the unbounded edge. Kills: no check before synchronize().

        A deadline of 0 blocks in the CUDA driver with no timeout at all. The
        assertion is on the SYNC COUNTER: proving it never entered the
        blocking call, not merely that something raised.
        """
        ev = _ScriptedEvent(never=True)
        t = _transport(event=ev, table=self._table(dead=True))
        with mock.patch.dict(
            os.environ, {"SGLANG_BARLINK_BAR1_ABORT_SYNC_DEADLINE_MS": "0"}
        ):
            with self.assertRaises(Bar1PeerLost):
                t._wait_ctl_event()
        self.assertEqual(ev.syncs, 0)

    def test_answer_is_bounded_by_the_probe_not_the_deadline(self):
        """Case 7 -- the in-loop probe edge. Kills: removing the loop probe.

        With a 30 s deadline and a 10 ms probe interval, a gate that only
        checks on expiry would take 30 s to answer. This asserts it answers in
        well under a second, which no expiry-only implementation can do.
        """
        t = _transport(table=self._table(dead=True))
        with mock.patch.dict(
            os.environ, {"SGLANG_BARLINK_BAR1_ABORT_SYNC_DEADLINE_MS": "30000"}
        ):
            t0 = time.monotonic()
            with self.assertRaises(Bar1PeerLost):
                t._wait_ctl_event()
            waited = time.monotonic() - t0
        self.assertLess(waited, 5.0)
        # and it did NOT get there by the expiry path
        self.assertEqual(t._ctl_sync_timeouts, 0)


class RaiseIfPeerLostUnitTest(CustomTestCase):
    """The helper's own contract, independent of the wait loop."""

    def setUp(self):
        super().setUp()
        live.reset_for_test()

    def tearDown(self):
        live.reset_for_test()
        super().tearDown()

    def test_no_table_is_a_silent_no_op(self):
        """A transport built before ``install()`` ran must not be broken by it."""
        t = _transport(table=None)
        self.assertIsNone(raise_if_peer_lost(t, 0.0))

    def test_empty_table_is_a_silent_no_op(self):
        me = live.local_identity(0)
        t = _transport(table=live.PeerTable([me], self_rank=0))
        self.assertIsNone(raise_if_peer_lost(t, 0.0))

    def test_a_broken_table_does_not_hold_the_wedge_open(self):
        """A diagnostic that raises would turn a stall into an AttributeError.

        The module's own rule, inherited from
        ``defer_stall_for_building_peer``: never let the guard's helper be the
        thing that fails.
        """

        class _Exploding:
            def dead_peers(self):
                raise RuntimeError("peer table is broken")

        t = _transport(table=_Exploding())
        self.assertIsNone(raise_if_peer_lost(t, 0.0))


if __name__ == "__main__":
    unittest.main()
