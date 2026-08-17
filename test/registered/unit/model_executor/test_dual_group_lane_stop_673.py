"""#673: the dual-group lane worker must be stopped, and the detach must be loud.

INVENTORY ROW (evidence-665-f1/DIAG_673_TEARDOWN_ABORT.md:210):

    | dual-group-lane-{id} | model_executor/dual_group_lane.py:2154 |
      stop_worker() at :2172, **zero callers** | owns a live torch.cuda.Stream,
      launches kernels |

TWO DEFECTS, not one. The inventory records the orphaned caller; working on the
kvso sibling turned up the second, which is worse because it is silent:

    def stop_worker(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=10.0)
        self._thread = None          # <-- UNCONDITIONAL

After a join that TIMED OUT, the handle is cleared anyway. There is no
``is_alive()`` check and no log, so a thread that is still running -- with its
own CUDA stream, launching kernels -- is detached silently AND the object
forgets it exists. A second stop_worker() then returns immediately on
``self._thread is None``, reporting success for a thread it abandoned. That is
the #673 abort shape with the evidence deleted.

Wiring a caller without fixing that would have installed a silent 10 s stall
followed by a silent leak, which is why the order here is: fix the method,
then wire it.
"""

import threading
import time
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase

LANE_MOD = "sglang.srt.model_executor.dual_group_lane"


class _StuckWorker:
    """Stands in for a lane thread parked in a kernel launch / stream sync:
    it never observes the stop event."""

    def __init__(self):
        self.release = threading.Event()

    def __call__(self):
        self.release.wait()


def _lane(worker_target=None, lane_id=7):
    """A real DualGroupLane carrying only what the stop path touches."""
    from sglang.srt.model_executor.dual_group_lane import DualGroupLane

    lane = DualGroupLane.__new__(DualGroupLane)
    lane.lane_id = lane_id
    lane._wake = threading.Event()
    lane._stop = threading.Event()
    if worker_target is None:
        # A cooperative worker with the real contract: wake, observe stop, exit.
        def _cooperative():
            while not lane._stop.is_set():
                lane._wake.wait(timeout=0.05)
                lane._wake.clear()

        worker_target = _cooperative
    lane._thread = threading.Thread(
        target=worker_target, name=f"dual-group-lane-{lane_id}", daemon=True
    )
    lane._thread.start()
    return lane


class TestTheDetachIsNoLongerSilent(CustomTestCase):
    def test_a_TIMED_OUT_join_KEEPS_the_handle(self):
        """RED-FIRST, and the defect the inventory row does not record.

        Clearing the handle after a failed join destroys the only evidence
        that a CUDA-stream thread is still running.
        """
        stuck = _StuckWorker()
        lane = _lane(worker_target=stuck)
        try:
            lane.stop_worker(timeout_s=0.05)
            self.assertTrue(lane._thread.is_alive(), "it really is still running")
            self.assertIsNotNone(
                lane._thread,
                "the handle was cleared for a thread that is still alive: the "
                "leak is now invisible and a second stop reports success",
            )
        finally:
            stuck.release.set()
            lane._thread.join(timeout=2.0)

    def test_the_detach_is_LOGGED_and_names_the_thread(self):
        stuck = _StuckWorker()
        lane = _lane(worker_target=stuck, lane_id=3)
        try:
            with self.assertLogs(LANE_MOD, level="WARNING") as cap:
                lane.stop_worker(timeout_s=0.05)
            joined = "\n".join(cap.output)
            self.assertIn("detach", joined.lower())
            self.assertIn("dual-group-lane-3", joined)
        finally:
            stuck.release.set()
            lane._thread.join(timeout=2.0)

    def test_it_REPORTS_what_it_did(self):
        """A stop path that returns None cannot be audited by its caller."""
        stuck = _StuckWorker()
        lane = _lane(worker_target=stuck)
        try:
            self.assertEqual(lane.stop_worker(timeout_s=0.05), "detached")
        finally:
            stuck.release.set()
            lane._thread.join(timeout=2.0)

    def test_a_second_stop_after_a_detach_does_NOT_claim_success(self):
        """The consequence of keeping the handle: the leak stays reportable."""
        stuck = _StuckWorker()
        lane = _lane(worker_target=stuck)
        try:
            self.assertEqual(lane.stop_worker(timeout_s=0.05), "detached")
            self.assertEqual(lane.stop_worker(timeout_s=0.05), "detached")
        finally:
            stuck.release.set()
            lane._thread.join(timeout=2.0)


class TestTheCleanPath(CustomTestCase):
    def test_a_cooperative_worker_is_JOINED_and_the_handle_cleared(self):
        lane = _lane()
        self.assertEqual(lane.stop_worker(timeout_s=2.0), "joined")
        self.assertIsNone(lane._thread)

    def test_a_clean_join_is_QUIET(self):
        """CAN-FAIL counterweight: warning on every healthy exit trains the
        reader to ignore the warning that matters."""
        lane = _lane()
        with self.assertNoLogs(LANE_MOD, level="WARNING"):
            lane.stop_worker(timeout_s=2.0)

    def test_it_is_idempotent(self):
        lane = _lane()
        self.assertEqual(lane.stop_worker(timeout_s=2.0), "joined")
        self.assertEqual(lane.stop_worker(timeout_s=2.0), "already stopped")

    def test_a_lane_that_never_started_is_safe(self):
        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        lane = DualGroupLane.__new__(DualGroupLane)
        lane._thread = None
        self.assertEqual(lane.stop_worker(timeout_s=0.05), "already stopped")


class TestTheDeadline(CustomTestCase):
    def test_the_default_deadline_is_the_two_second_class(self):
        """10 s was never measured; teardown budgets are seconds, not tens of
        seconds, and the kvso sibling settled on 2 s."""
        import inspect

        from sglang.srt.model_executor.dual_group_lane import DualGroupLane

        default = inspect.signature(DualGroupLane.stop_worker).parameters[
            "timeout_s"
        ].default
        self.assertLessEqual(float(default), 2.0)
        self.assertGreater(float(default), 0.0)

    def test_a_stuck_worker_does_not_hang_teardown(self):
        stuck = _StuckWorker()
        lane = _lane(worker_target=stuck)
        try:
            t0 = time.perf_counter()
            lane.stop_worker(timeout_s=0.05)
            self.assertLess(time.perf_counter() - t0, 1.0, "the join was unbounded")
        finally:
            stuck.release.set()
            lane._thread.join(timeout=2.0)


class TestTheComponentSeam(CustomTestCase):
    """The component owns HOW to stop what it built; it had a builder and no
    counterpart."""

    def test_stop_dual_group_lanes_stops_every_lane(self):
        from sglang.srt.model_executor.dual_group_lane import stop_dual_group_lanes

        lanes = [_lane(lane_id=i) for i in range(3)]
        out = stop_dual_group_lanes(SimpleNamespace(dual_group_lanes=lanes))
        self.assertEqual(out, ["joined", "joined", "joined"])
        for lane in lanes:
            self.assertIsNone(lane._thread)

    def test_one_failing_lane_does_not_strand_the_others(self):
        from sglang.srt.model_executor.dual_group_lane import stop_dual_group_lanes

        class Boom:
            def stop_worker(self, timeout_s=None):
                raise RuntimeError("nope")

        good = _lane()
        out = stop_dual_group_lanes(
            SimpleNamespace(dual_group_lanes=[Boom(), good])
        )
        self.assertIsNone(good._thread, "the second lane must still be stopped")
        self.assertIn("joined", out)

    def test_no_lanes_is_a_quiet_no_op(self):
        from sglang.srt.model_executor.dual_group_lane import stop_dual_group_lanes

        self.assertEqual(stop_dual_group_lanes(SimpleNamespace()), [])
        self.assertEqual(
            stop_dual_group_lanes(SimpleNamespace(dual_group_lanes=[])), []
        )


class TestTheTeardownWiring(CustomTestCase):
    def test_graceful_teardown_stops_the_lanes(self):
        from sglang.srt.managers import scheduler_teardown as td

        lanes = [_lane()]
        out = td.release_dual_group_lanes(
            SimpleNamespace(dual_group_lanes=lanes), graceful=True
        )
        self.assertEqual(out, "joined")
        self.assertIsNone(lanes[0]._thread)

    def test_the_exception_path_leaves_them_alone(self):
        from sglang.srt.managers import scheduler_teardown as td

        lanes = [_lane()]
        try:
            self.assertIsNone(
                td.release_dual_group_lanes(
                    SimpleNamespace(dual_group_lanes=lanes), graceful=False
                )
            )
            self.assertIsNotNone(lanes[0]._thread)
        finally:
            lanes[0].stop_worker(timeout_s=2.0)

    def test_no_lanes_is_a_no_op(self):
        from sglang.srt.managers import scheduler_teardown as td

        self.assertIsNone(
            td.release_dual_group_lanes(SimpleNamespace(), graceful=True)
        )

    def test_it_NEVER_RAISES(self):
        from sglang.srt.managers import scheduler_teardown as td

        class Hostile:
            @property
            def dual_group_lanes(self):
                raise RuntimeError("boom")

        self.assertIsNone(td.release_dual_group_lanes(Hostile(), graceful=True))


class TestTheStopPathIsACTUALLYCALLED(CustomTestCase):
    """The defect this whole row IS: a stop method with zero callers."""

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
            "release_dual_group_lanes",
            called,
            "an orphaned stop method is exactly the defect being fixed",
        )

    def test_it_is_gated_on_the_graceful_flag(self):
        import inspect

        from sglang.srt.managers import scheduler as sched_mod

        src = inspect.getsource(sched_mod)
        idx = src.index("release_dual_group_lanes(")
        self.assertIn("graceful=scheduler.gracefully_exit", src[idx : idx + 200])


if __name__ == "__main__":
    unittest.main()
