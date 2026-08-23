"""#673: the kvso destination IO thread must be stopped, and bounded.

FROM THE #673 BACKGROUND-THREAD INVENTORY
(evidence-665-f1/DIAG_673_TEARDOWN_ABORT.md:208):

    | kvso-dest-io | kv_session_spill_destination.py:646 | **none exists at
      all** | evt.synchronize() at :815 = cudaEventSynchronize |

Same teardown class as #673's root candidate, and worse in one respect: the
NCCL fix ships gated, while this thread is created UNCONDITIONALLY in
``SpillDestinationController.__init__`` on every boot that uses kvso and is
never stopped by anything. ``self._worker`` appears exactly twice in the whole
file -- the constructor and ``.start()`` -- with no ``join()``, no stop method
and no ``close``/``shutdown`` on the class.

Its body calls ``evt.synchronize()`` (``:815``), i.e. ``cudaEventSynchronize``.
A thread sitting in a CUDA call while the process tears down is the shape that
aborts, and ``daemon=True`` does not help: the interpreter stops waiting for
it, which is precisely how it ends up mid-C++-call at exit.

THE LOOP ALREADY KNOWS HOW TO STOP. ``_worker_loop`` returns on a ``None``
sentinel (``:805-806``). What is missing is a caller for it and a join. So the
fix supplies the missing half rather than reshaping the worker.

THE DEADLINE IS NOT DECORATION. The thread may be inside
``cudaEventSynchronize`` on a wedged device, where the sentinel will not be
read at all. Blocking teardown on it would trade an abort for a hang, which is
worse -- the abort at least ends the process. So the join is bounded, and when
the deadline passes the thread is detached EXPLICITLY and loudly, so that the
choice appears in the log instead of being implied by a daemon flag.
"""

import queue
import threading
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase


def _controller(worker_target=None, name="kvso-dest-io-test"):
    """A real SpillDestinationController carrying only what the stop path
    touches, with a real thread on the REAL worker loop unless overridden."""
    from sglang.srt.managers.kv_session_spill_destination import (
        SpillDestinationController,
    )

    c = SpillDestinationController.__new__(SpillDestinationController)
    c._jobs = queue.Queue()
    c._worker = threading.Thread(
        target=worker_target or c._worker_loop, name=name, daemon=True
    )
    c._worker.start()
    return c


class _StuckWorker:
    """A worker that never reads the sentinel -- stands in for a thread parked
    inside cudaEventSynchronize on a wedged device."""

    def __init__(self):
        self.release = threading.Event()

    def __call__(self):
        self.release.wait()


class TestTheThreadCanBeStopped(CustomTestCase):
    def test_the_controller_HAS_a_stop_path(self):
        """RED-FIRST. The inventory row in one assertion."""
        from sglang.srt.managers.kv_session_spill_destination import (
            SpillDestinationController,
        )

        self.assertTrue(
            hasattr(SpillDestinationController, "stop_worker"),
            "kvso-dest-io has no way to be stopped; it is leaked on every boot "
            "that uses kvso",
        )

    def test_stop_worker_JOINS_the_real_loop(self):
        c = _controller()
        self.assertTrue(c._worker.is_alive())
        outcome = c.stop_worker(timeout_s=2.0)
        self.assertFalse(c._worker.is_alive(), "the worker was not joined")
        self.assertEqual(outcome, "joined")

    def test_it_is_idempotent(self):
        """Teardown paths get called twice -- a retry, a test, a second call
        site added later. The second call must be a no-op, not an error."""
        c = _controller()
        self.assertEqual(c.stop_worker(timeout_s=2.0), "joined")
        self.assertEqual(c.stop_worker(timeout_s=2.0), "already stopped")

    def test_stopping_a_never_started_controller_is_safe(self):
        from sglang.srt.managers.kv_session_spill_destination import (
            SpillDestinationController,
        )

        c = SpillDestinationController.__new__(SpillDestinationController)
        self.assertEqual(c.stop_worker(timeout_s=0.05), "already stopped")


class TestTheJoinIsBounded(CustomTestCase):
    """A stuck CUDA event must not hold the process open."""

    def test_a_stuck_worker_does_not_hang_teardown(self):
        stuck = _StuckWorker()
        c = _controller(worker_target=stuck)
        try:
            t0 = threading.Event()
            start = __import__("time").perf_counter()
            outcome = c.stop_worker(timeout_s=0.05)
            elapsed = __import__("time").perf_counter() - start
            self.assertEqual(outcome, "detached")
            self.assertLess(elapsed, 1.0, "the join was not bounded")
            self.assertTrue(c._worker.is_alive(), "it really is still running")
            del t0
        finally:
            stuck.release.set()
            c._worker.join(timeout=2.0)

    def test_the_detach_is_LOGGED_not_implied(self):
        """CAN-FAIL. A silent detach is indistinguishable from a clean join in
        the log, which is how this class of leak stayed invisible."""
        stuck = _StuckWorker()
        c = _controller(worker_target=stuck)
        try:
            with self.assertLogs(
                "sglang.srt.managers.kv_session_spill_destination", level="WARNING"
            ) as cap:
                c.stop_worker(timeout_s=0.05)
            joined = "\n".join(cap.output)
            self.assertIn("detach", joined.lower())
        finally:
            stuck.release.set()
            c._worker.join(timeout=2.0)

    def test_a_clean_join_is_QUIET(self):
        """CAN-FAIL counterweight: a teardown that warns on every healthy exit
        trains readers to ignore it."""
        c = _controller()
        with self.assertNoLogs(
            "sglang.srt.managers.kv_session_spill_destination", level="WARNING"
        ):
            c.stop_worker(timeout_s=2.0)


class TestTheTeardownWiring(CustomTestCase):
    """The stop path must be reached by the scheduler's shutdown."""

    def _sched(self, dest):
        return SimpleNamespace(
            server_args=SimpleNamespace(),
            _kvso_manager_for_test=SimpleNamespace(_dest=dest),
        )

    def test_graceful_teardown_stops_the_worker(self):
        from sglang.srt.managers import scheduler_teardown as td

        c = _controller()
        out = td.release_kv_session_offload_io(
            self._sched(c), graceful=True, _manager=SimpleNamespace(_dest=c)
        )
        self.assertEqual(out, "joined")
        self.assertFalse(c._worker.is_alive())

    def test_the_exception_path_does_NOT_touch_it(self):
        """Same guard as release_distributed and release_host_resources: on the
        exception path the device may be wedged, and a teardown that hangs is
        worse than the abort it prevents."""
        from sglang.srt.managers import scheduler_teardown as td

        c = _controller()
        try:
            self.assertIsNone(
                td.release_kv_session_offload_io(
                    self._sched(c), graceful=False, _manager=SimpleNamespace(_dest=c)
                )
            )
            self.assertTrue(c._worker.is_alive(), "it must be left alone")
        finally:
            c.stop_worker(timeout_s=2.0)

    def test_no_kvso_manager_is_a_quiet_no_op(self):
        """kvso is off by default; the default boot must not change."""
        from sglang.srt.managers import scheduler_teardown as td

        self.assertIsNone(
            td.release_kv_session_offload_io(
                SimpleNamespace(server_args=SimpleNamespace()),
                graceful=True,
                _manager=None,
            )
        )

    def test_a_manager_without_destinations_is_a_no_op(self):
        from sglang.srt.managers import scheduler_teardown as td

        self.assertIsNone(
            td.release_kv_session_offload_io(
                SimpleNamespace(server_args=SimpleNamespace()),
                graceful=True,
                _manager=SimpleNamespace(_dest=None),
            )
        )

    def test_it_NEVER_RAISES(self):
        """It runs in a finally during shutdown: raising would replace a clean
        exit with a traceback, or mask the failure that caused the exit."""
        from sglang.srt.managers import scheduler_teardown as td

        class Hostile:
            @property
            def _dest(self):
                raise RuntimeError("boom")

        self.assertIsNone(
            td.release_kv_session_offload_io(
                SimpleNamespace(server_args=SimpleNamespace()),
                graceful=True,
                _manager=Hostile(),
            )
        )

    def test_a_failing_stop_is_swallowed_and_reported(self):
        from sglang.srt.managers import scheduler_teardown as td

        class Boom:
            def stop_worker(self, timeout_s=None):
                raise RuntimeError("nope")

        self.assertIsNone(
            td.release_kv_session_offload_io(
                SimpleNamespace(server_args=SimpleNamespace()),
                graceful=True,
                _manager=SimpleNamespace(_dest=Boom()),
            )
        )


if __name__ == "__main__":
    unittest.main()


class TestTheStopPathIsACTUALLYCALLED(CustomTestCase):
    """The failure mode this ticket is FULL of: a stop method nobody calls.

    ``dual_group_lane.stop_worker`` (the other inventory row) has existed all
    along with ZERO callers, which is why its thread leaks too. A fix that
    added ``stop_worker`` here and forgot the call site would reproduce that
    exact defect while looking finished.
    """

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
        self.assertIn(
            "release_kv_session_offload_io",
            called,
            "the stop path is never invoked -- an orphaned stop method is the "
            "dual-group-lane defect, not a fix",
        )

    def test_it_is_called_on_the_graceful_flag_not_unconditionally(self):
        """#815: THE ASSERTION IS THE SAME, THE INSTRUMENT CHANGED.

        This used to locate the call by `src.index("release_kv_session_offload_
        io(\\n")` -- a literal that required the call to be SPLIT across lines.
        86d1cb1384 [#794] reflowed it onto one line to match its three sibling
        `release_*` calls, a pure formatting change, and the search raised
        ValueError. The test was pinning a line break, not a contract.

        What it is actually for is unchanged, and it is NOT covered by the AST
        existence check above: that the graceful flag threaded into the teardown
        call is the scheduler's REAL exit state, rather than hardcoded True or
        dropped to the parameter default. A regression doing either would leave
        every other test in this file green -- the behaviour tests above drive
        `release_kv_session_offload_io` directly and never see the call site.

        So it is read off the syntax tree now, where a reflow cannot reach it.
        """
        import ast
        import inspect

        from sglang.srt.managers import scheduler as sched_mod

        tree = ast.parse(inspect.getsource(sched_mod))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "release_kv_session_offload_io"
        ]
        self.assertEqual(len(calls), 1, "expected exactly one teardown call site")
        graceful = [k for k in calls[0].keywords if k.arg == "graceful"]
        self.assertEqual(
            len(graceful),
            1,
            "the teardown must pass `graceful` explicitly, not fall back to the "
            "parameter default -- the default is not the scheduler's exit state",
        )
        self.assertEqual(
            ast.unparse(graceful[0].value),
            "scheduler.gracefully_exit",
            "the flag must be the scheduler's real exit state, not a constant",
        )
