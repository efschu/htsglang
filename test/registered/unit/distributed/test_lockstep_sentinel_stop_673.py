"""#673: stop the lockstep sentinel, and stop it BEFORE the groups it uses.

INVENTORY ROW (evidence-665-f1/DIAG_673_TEARDOWN_ABORT.md:209):

    | lockstep-sentinel | distributed/device_communicators/lockstep_sentinel.py:173
      | stop() at :247, **zero callers** | dist.all_gather_object on a
      dedicated gloo group, every 0.5 s |

TWO PROBLEMS, and the second is the one that makes this more than a one-line
caller fix.

1. ORPHANED. ``stop()`` had no callers, so on any boot with
   ``SGLANG_LOCKSTEP_SENTINEL=1`` the thread ran until the interpreter died.

2. ``stop()`` DOES NOT WAIT. Its whole body was ``self._stop.set()``. The loop
   is ``while not stop: sleep(interval); compare_once()``, and
   ``compare_once`` -> ``_gather`` -> ``dist.all_gather_object(group=...)``. So
   when ``stop()`` RETURNED, the thread could still be inside a collective on
   the gloo group. Setting a flag is not an ordering guarantee.

WHY THAT MATTERS NOW. #673's other fix destroys the process groups
(``scheduler_teardown.release_distributed``). Destroying a group while a live
thread is inside a collective on it is its own abort -- so the sentinel must be
STOPPED AND JOINED before that destroy runs. Today the two never meet, because
the destroy ships gated default-off and the sentinel is opt-in. An
armed-together boot must be safe BY CONSTRUCTION, not by one of the two flags
staying off, which is why the ordering is enforced inside ``release_distributed``
itself rather than only by the order of calls in the scheduler's ``finally``.
"""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.test_utils import CustomTestCase

SENTINEL_MOD = "sglang.srt.distributed.device_communicators.lockstep_sentinel"


class _Blocked:
    """Stands in for a thread parked inside all_gather_object on a peer that
    is gone: it never observes the stop event."""

    def __init__(self):
        self.release = threading.Event()

    def __call__(self):
        self.release.wait()


def _sentinel(worker_target=None):
    """A real LockstepSentinel carrying only what the stop path touches."""
    from sglang.srt.distributed.device_communicators.lockstep_sentinel import (
        LockstepSentinel,
    )

    s = LockstepSentinel.__new__(LockstepSentinel)
    s._stop = threading.Event()
    if worker_target is None:

        def _cooperative():
            while not s._stop.is_set():
                time.sleep(0.01)

        worker_target = _cooperative
    s._thread = threading.Thread(
        target=worker_target, name="lockstep-sentinel", daemon=True
    )
    s._thread.start()
    return s


class TestStopActuallyWaits(CustomTestCase):
    def test_stop_JOINS_rather_than_only_setting_a_flag(self):
        """RED-FIRST. ``stop()`` used to be ``self._stop.set()`` and nothing
        else, so it could return while the thread was still inside
        ``all_gather_object`` -- which is exactly the window in which the group
        must not be destroyed."""
        s = _sentinel()
        thread = s._thread
        self.assertEqual(s.stop(timeout_s=2.0), "joined")
        self.assertFalse(thread.is_alive(), "stop() returned before the join")
        self.assertIsNone(s._thread, "a clean join clears the handle")

    def test_a_blocked_sentinel_is_DETACHED_loudly_not_waited_on_forever(self):
        blocked = _Blocked()
        s = _sentinel(worker_target=blocked)
        try:
            t0 = time.perf_counter()
            with self.assertLogs(SENTINEL_MOD, level="WARNING") as cap:
                outcome = s.stop(timeout_s=0.05)
            self.assertEqual(outcome, "detached")
            self.assertLess(time.perf_counter() - t0, 1.0, "the join was unbounded")
            self.assertIn("detach", "\n".join(cap.output).lower())
        finally:
            blocked.release.set()
            s._thread.join(timeout=2.0)

    def test_a_detached_sentinel_KEEPS_its_handle(self):
        """Same rule as the two siblings: clearing the handle for a thread that
        is still running deletes the evidence of the leak."""
        blocked = _Blocked()
        s = _sentinel(worker_target=blocked)
        try:
            s.stop(timeout_s=0.05)
            self.assertIsNotNone(s._thread)
            self.assertTrue(s._thread.is_alive())
        finally:
            blocked.release.set()
            s._thread.join(timeout=2.0)

    def test_a_clean_stop_is_QUIET(self):
        s = _sentinel()
        with self.assertNoLogs(SENTINEL_MOD, level="WARNING"):
            s.stop(timeout_s=2.0)

    def test_it_is_idempotent(self):
        s = _sentinel()
        self.assertEqual(s.stop(timeout_s=2.0), "joined")
        self.assertEqual(s.stop(timeout_s=2.0), "already stopped")

    def test_a_sentinel_that_never_started_a_thread_is_safe(self):
        from sglang.srt.distributed.device_communicators.lockstep_sentinel import (
            LockstepSentinel,
        )

        s = LockstepSentinel.__new__(LockstepSentinel)
        s._stop = threading.Event()
        s._thread = None
        self.assertEqual(s.stop(timeout_s=0.05), "already stopped")
        self.assertTrue(s._stop.is_set(), "the flag must still be set")


class TestTheModuleLevelStop(CustomTestCase):
    def test_it_stops_the_installed_sentinel(self):
        from sglang.srt.distributed.device_communicators import lockstep_sentinel as ls

        s = _sentinel()
        thread = s._thread
        with mock.patch.object(ls, "_SENTINEL", s):
            self.assertEqual(ls.stop_sentinel(timeout_s=2.0), "joined")
        self.assertFalse(thread.is_alive())

    def test_no_sentinel_installed_is_a_quiet_no_op(self):
        from sglang.srt.distributed.device_communicators import lockstep_sentinel as ls

        with mock.patch.object(ls, "_SENTINEL", None):
            self.assertIsNone(ls.stop_sentinel(timeout_s=0.05))

    def test_it_NEVER_RAISES(self):
        from sglang.srt.distributed.device_communicators import lockstep_sentinel as ls

        class Boom:
            def stop(self, timeout_s=None):
                raise RuntimeError("nope")

        with mock.patch.object(ls, "_SENTINEL", Boom()):
            self.assertIsNone(ls.stop_sentinel(timeout_s=0.05))


class TestTheORDERINGIsEnforcedByConstruction(CustomTestCase):
    """THE POINT OF THIS TICKET.

    Destroying the process groups while the sentinel is inside
    ``all_gather_object`` on its gloo group is its own abort. The guarantee
    must not depend on the order of two calls in the scheduler's ``finally``,
    because that order is one careless edit away from inverting.
    """

    def _armed_scheduler(self):
        return SimpleNamespace(
            server_args=SimpleNamespace(scheduler_distributed_teardown=True)
        )

    def test_release_distributed_STOPS_THE_SENTINEL_FIRST(self):
        """CAN-FAIL by reordering: records the real call order and fails if the
        destroy happens before the sidecar is down.

        Patches the FUNCTIONS, not sys.modules: `from X import Y` resolves the
        package ATTRIBUTE, so a sys.modules patch is silently bypassed -- the
        exact trap that let an earlier #673 test run the real destroy and pass
        by accident.
        """
        from sglang.srt.managers import scheduler_teardown as td

        order = []
        with mock.patch(
            "sglang.srt.distributed.device_communicators.lockstep_sentinel"
            ".stop_sentinel",
            side_effect=lambda *a, **k: (order.append("stop_sentinel"), "joined")[1],
        ), mock.patch(
            "sglang.srt.distributed.parallel_state.destroy_model_parallel",
            side_effect=lambda: order.append("destroy_model_parallel"),
        ), mock.patch(
            "sglang.srt.distributed.parallel_state.destroy_distributed_environment",
            side_effect=lambda: order.append("destroy_world"),
        ):
            td.release_distributed(self._armed_scheduler(), graceful=True)

        self.assertIn("stop_sentinel", order, "the sentinel was never stopped")
        self.assertIn("destroy_model_parallel", order)
        self.assertLess(
            order.index("stop_sentinel"),
            order.index("destroy_model_parallel"),
            "the groups were destroyed while the sentinel could still be "
            "inside a collective on them -- the abort this ordering prevents",
        )

    def test_the_scheduler_calls_the_sentinel_release_BEFORE_the_distributed_one(self):
        """Source-order pin, so a reorder in the finally is caught too."""
        import inspect

        from sglang.srt.managers import scheduler as sched_mod

        src = inspect.getsource(sched_mod)
        i_sent = src.index("release_lockstep_sentinel(scheduler")
        i_dist = src.index("release_distributed(scheduler")
        self.assertLess(
            i_sent,
            i_dist,
            "the sentinel must be stopped before the groups are destroyed",
        )


class TestTheTeardownWiring(CustomTestCase):
    def test_graceful_stops_it(self):
        from sglang.srt.distributed.device_communicators import lockstep_sentinel as ls
        from sglang.srt.managers import scheduler_teardown as td

        s = _sentinel()
        thread = s._thread
        with mock.patch.object(ls, "_SENTINEL", s):
            self.assertEqual(
                td.release_lockstep_sentinel(SimpleNamespace(), graceful=True),
                "joined",
            )
        self.assertFalse(thread.is_alive())

    def test_the_exception_path_leaves_it_alone(self):
        from sglang.srt.distributed.device_communicators import lockstep_sentinel as ls
        from sglang.srt.managers import scheduler_teardown as td

        s = _sentinel()
        try:
            with mock.patch.object(ls, "_SENTINEL", s):
                self.assertIsNone(
                    td.release_lockstep_sentinel(SimpleNamespace(), graceful=False)
                )
            self.assertTrue(s._thread.is_alive())
        finally:
            s.stop(timeout_s=2.0)

    def test_it_is_UNGATED(self):
        """The sentinel leaks on every opt-in boot, whether or not the
        process-group destroy is armed, so it must not inherit that gate."""
        from sglang.srt.distributed.device_communicators import lockstep_sentinel as ls
        from sglang.srt.managers import scheduler_teardown as td

        s = _sentinel()
        sched = SimpleNamespace(
            server_args=SimpleNamespace(scheduler_distributed_teardown=False)
        )
        with mock.patch.object(ls, "_SENTINEL", s):
            self.assertEqual(td.release_lockstep_sentinel(sched, graceful=True), "joined")

    def test_it_NEVER_RAISES(self):
        from sglang.srt.managers import scheduler_teardown as td

        with mock.patch.dict(
            "sys.modules",
            {"sglang.srt.distributed.device_communicators.lockstep_sentinel": None},
        ):
            self.assertIsNone(
                td.release_lockstep_sentinel(SimpleNamespace(), graceful=True)
            )


class TestTheStopPathIsACTUALLYCALLED(CustomTestCase):
    def test_the_scheduler_teardown_calls_it(self):
        import ast
        import inspect

        from sglang.srt.managers import scheduler as sched_mod

        tree = ast.parse(inspect.getsource(sched_mod))
        called = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        self.assertIn("release_lockstep_sentinel", called)


if __name__ == "__main__":
    unittest.main()
