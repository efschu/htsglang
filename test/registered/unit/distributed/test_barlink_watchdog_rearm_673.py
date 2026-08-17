"""#673 Option B: stopping the watchdog must GIVE THE READ BACK, not remove it.

THE BLINDNESS (proved in bdbbaf49e1, fixed here).

Since #517 phase 2 the watchdog thread is the only reader of a
``BarlinkDeviceTransport``'s abort word:

* ``_arm_status_poll`` (barlink_device.py:1440) runs ONCE at bring-up and
  latches ``_abort_poll_active = True`` (:1478);
* that latch is ONE-WAY -- the only other assignments are the class default
  (:993) and the constructor (:1071);
* ``check_aborted`` (:1540) short-circuits on it::

      if self._abort_poll_active:
          if not self._abort_code_seen:
              return          # no device read at all

* and ``_abort_code_seen`` is written only by ``poll_status_word`` (:1504),
  whose only caller is the watchdog.

So stopping the watchdog while the latch is armed leaves the word unread by
anyone and ``check_aborted`` answers "not aborted" forever. ``should_poll_status``
already names the intended degradation -- *"the guard degrades to the
#517-phase-1 behaviour rather than to blindness"* -- but it is evaluated only at
bring-up, so it protects "no watchdog at boot", not "watchdog stopped later".

Option B closes that: stopping the watchdog CLEARS the latch on every
registered transport, so ``check_aborted`` falls back to the pre-#517 in-line
device read. The abort-word semantics are untouched; only who reads it changes.

SCOPE PRECISION, verified rather than assumed: this blindness is specific to
the ``barlink_device`` family. ``BarlinkBar1Transport.check_aborted``
(barlink_bar1.py:5122) never consults ``_abort_poll_active`` -- it always goes
through ``_read_status_for_check`` (:4889) -- so bar1 keeps its own read either
way. The re-arm still covers it, because clearing a latch that only
short-circuits the watchdog's own poll is a no-op once the watchdog is going
away. A test below pins that asymmetry so a future edit cannot quietly make
bar1 depend on the latch too.
"""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.distributed.device_communicators import (
    barlink_abort_gate,
    barlink_liveness,
)
from sglang.srt.distributed.device_communicators.barlink_device import (
    BarlinkDeviceTransport,
    DeviceCollectiveAborted,
)

LIVENESS_MOD = "sglang.srt.distributed.device_communicators.barlink_liveness"


def _device_transport(code: int = 0, *, launches: int = 8, armed: bool = True):
    """A device transport shaped for check_aborted, with the poll latch armed.

    Mirrors ``_fake_transport`` in test_barlink_device_abort_583.py and adds
    the #517 fields, so the REAL methods are exercised.
    """
    t = object.__new__(BarlinkDeviceTransport)
    t._seq_dev = torch.zeros(2, dtype=torch.int64)
    t._seq_dev[1] = code
    t.rank = 1
    t.world_size = 3
    t._unchecked_launches = launches
    t._captured_launches = False
    t._boundary_checks = 0
    t._registered_in_gate = False
    t._abort_poll_active = bool(armed)
    t._abort_code_seen = 0
    t._abort_poll_stream = None
    t._abort_poll_dst = None
    return t


class _Cooperative:
    """A worker that exits on the stop event, and records what the latch said
    at the moment it noticed."""

    def __init__(self, watchdog, probe):
        self.watchdog = watchdog
        self.probe = probe
        self.latch_when_stopped = None

    def __call__(self):
        while not self.watchdog._stop.is_set():
            time.sleep(0.001)
        self.latch_when_stopped = self.probe()


class _Blocked:
    def __init__(self):
        self.release = threading.Event()

    def __call__(self):
        self.release.wait()


def _watchdog(worker_target=None):
    """A watchdog whose worker STAYS ALIVE until stopped, like the real one.

    A target that returns immediately would make every stop report "already
    stopped" and quietly hollow out the tests that assert a real join.
    """
    w = barlink_liveness.PeerWatchdog.__new__(barlink_liveness.PeerWatchdog)
    w._interval = 1.0
    w._stop = threading.Event()
    w.trips = 0

    def _cooperative():
        while not w._stop.is_set():
            time.sleep(0.001)

    w._thread = threading.Thread(
        target=worker_target or _cooperative,
        name="barlink-peer-watchdog",
        daemon=True,
    )
    w._thread.start()
    return w


class _Base(unittest.TestCase):
    def setUp(self):
        barlink_abort_gate.reset_for_test()

    def tearDown(self):
        barlink_abort_gate.reset_for_test()


# --- STEP 1: the re-arm ------------------------------------------------------


class TestTheReadIsGivenBack(_Base):
    def test_AN_ABORT_AFTER_THE_STOP_IS_STILL_SEEN(self):
        """RED-FIRST, and the whole point of Option B.

        The device word trips AFTER the watchdog is stopped -- exactly the
        teardown window #673 is about. With the latch left armed this raises
        nothing at all, which is the blindness.
        """
        t = _device_transport(code=0)
        barlink_abort_gate.register(t)
        w = _watchdog()
        w.stop(timeout_s=0.25)

        t._seq_dev[1] = 1  # a kernel takes its abort path, after the stop
        with self.assertRaises(DeviceCollectiveAborted):
            t.check_aborted("after-teardown-stop")

    def test_the_latch_is_cleared_on_every_registered_transport(self):
        ts = [_device_transport() for _ in range(3)]
        for t in ts:
            barlink_abort_gate.register(t)
        _watchdog().stop(timeout_s=0.25)
        for t in ts:
            self.assertFalse(t._abort_poll_active)

    def test_the_latch_is_cleared_BEFORE_THE_THREAD_IS_GONE(self):
        """No window where the reader has exited but the latch still says
        'the watchdog reads for you'. The worker records the latch at the
        moment it observes the stop event; it must already be False.
        """
        t = _device_transport()
        barlink_abort_gate.register(t)
        w = barlink_liveness.PeerWatchdog.__new__(barlink_liveness.PeerWatchdog)
        w._interval = 1.0
        w._stop = threading.Event()
        w.trips = 0
        coop = _Cooperative(w, lambda: t._abort_poll_active)
        w._thread = threading.Thread(target=coop, name="barlink-peer-watchdog", daemon=True)
        w._thread.start()
        w.stop(timeout_s=1.0)
        self.assertIs(
            coop.latch_when_stopped,
            False,
            "the latch was still armed while the reader was shutting down",
        )

    def test_a_transport_without_the_latch_is_left_alone(self):
        """The registry is heterogeneous; the re-arm must not invent state."""
        foreign = SimpleNamespace(name="not-a-device-transport")
        barlink_abort_gate.register(foreign)
        _watchdog().stop(timeout_s=0.25)
        self.assertFalse(hasattr(foreign, "_abort_poll_active"))

    def test_a_raising_transport_does_not_strand_the_others(self):
        class Hostile:
            @property
            def _abort_poll_active(self):
                raise RuntimeError("boom")

        good = _device_transport()
        barlink_abort_gate.register(Hostile())
        barlink_abort_gate.register(good)
        _watchdog().stop(timeout_s=0.25)
        self.assertFalse(good._abort_poll_active)

    def test_rearm_with_no_transports_is_a_no_op(self):
        self.assertEqual(barlink_abort_gate.rearm_inline_reads(), 0)

    def test_rearm_is_idempotent(self):
        t = _device_transport()
        barlink_abort_gate.register(t)
        self.assertEqual(barlink_abort_gate.rearm_inline_reads(), 1)
        self.assertEqual(barlink_abort_gate.rearm_inline_reads(), 0)

    def test_the_abort_word_SEMANTICS_are_untouched(self):
        """Only who reads it changes. A clean word still raises nothing."""
        t = _device_transport(code=0)
        barlink_abort_gate.register(t)
        _watchdog().stop(timeout_s=0.25)
        t.check_aborted("clean")  # must not raise

    def test_bar1_NEVER_depended_on_the_latch(self):
        """Precision pin for the scope claim in the module docstring: the
        blindness is device-family only, because bar1's check_aborted reads
        through _read_status_for_check instead of the latch. If a future edit
        makes bar1 short-circuit on the latch, this fails and the re-arm must
        be re-verified for that family."""
        import inspect

        from sglang.srt.distributed.device_communicators.barlink_bar1 import (
            BarlinkBar1Transport,
        )

        src = inspect.getsource(BarlinkBar1Transport.check_aborted)
        self.assertNotIn("_abort_poll_active", src)


# --- STEP 2: the stop method's own defect ------------------------------------


class TestStopJoinsBeforeClearing(_Base):
    def test_a_TIMED_OUT_join_KEEPS_the_handle(self):
        """RED-FIRST against `thread, self._thread = self._thread, None`
        placed BEFORE the join: a live thread with no record of it."""
        blocked = _Blocked()
        w = _watchdog(blocked)
        try:
            outcome = w.stop(timeout_s=0.05)
            self.assertEqual(outcome, "detached")
            self.assertIsNotNone(w._thread, "the handle was dropped for a live thread")
            self.assertTrue(w._thread.is_alive())
        finally:
            blocked.release.set()
            w._thread.join(timeout=2.0)

    def test_the_detach_is_LOGGED(self):
        blocked = _Blocked()
        w = _watchdog(blocked)
        try:
            with self.assertLogs(LIVENESS_MOD, level="WARNING") as cap:
                w.stop(timeout_s=0.05)
            self.assertIn("detach", "\n".join(cap.output).lower())
        finally:
            blocked.release.set()
            w._thread.join(timeout=2.0)

    def test_a_second_stop_after_a_detach_does_not_claim_success(self):
        blocked = _Blocked()
        w = _watchdog(blocked)
        try:
            self.assertEqual(w.stop(timeout_s=0.05), "detached")
            self.assertEqual(w.stop(timeout_s=0.05), "detached")
        finally:
            blocked.release.set()
            w._thread.join(timeout=2.0)

    def test_a_clean_join_clears_the_handle_and_is_quiet(self):
        w = _watchdog()
        with self.assertNoLogs(LIVENESS_MOD, level="WARNING"):
            self.assertEqual(w.stop(timeout_s=1.0), "joined")
        self.assertIsNone(w._thread)

    def test_it_is_idempotent(self):
        w = _watchdog()
        self.assertEqual(w.stop(timeout_s=1.0), "joined")
        self.assertEqual(w.stop(timeout_s=1.0), "already stopped")

    def test_a_watchdog_that_never_started_is_safe(self):
        w = barlink_liveness.PeerWatchdog.__new__(barlink_liveness.PeerWatchdog)
        w._interval = 1.0
        w._stop = threading.Event()
        w.trips = 0
        w._thread = None
        self.assertEqual(w.stop(timeout_s=0.05), "already stopped")
        self.assertTrue(w._stop.is_set())

    def test_the_deadline_is_derived_from_the_POLL_CADENCE(self):
        """Not copied from the 2.0 s siblings. The loop waits ~10 ms
        (poll_interval_s default), so the default here is a small multiple of
        that, not of a second."""
        import inspect

        default = inspect.signature(barlink_liveness.PeerWatchdog.stop).parameters[
            "timeout_s"
        ].default
        self.assertLessEqual(float(default), 0.5)
        self.assertGreaterEqual(float(default), 20 * 0.010)


# --- STEP 3: the caller ------------------------------------------------------


class TestTheTeardownWiring(_Base):
    def test_graceful_stops_it_and_re_arms(self):
        from sglang.srt.managers import scheduler_teardown as td

        t = _device_transport()
        barlink_abort_gate.register(t)
        w = _watchdog()
        with mock.patch.object(barlink_liveness, "_watchdog", w):
            self.assertEqual(
                td.release_barlink_watchdog(SimpleNamespace(), graceful=True), "joined"
            )
        self.assertFalse(t._abort_poll_active)

    def test_the_exception_path_leaves_it_alone(self):
        from sglang.srt.managers import scheduler_teardown as td

        w = _watchdog()
        try:
            with mock.patch.object(barlink_liveness, "_watchdog", w):
                self.assertIsNone(
                    td.release_barlink_watchdog(SimpleNamespace(), graceful=False)
                )
            self.assertIsNotNone(w._thread)
        finally:
            w.stop(timeout_s=1.0)

    def test_it_is_ALWAYS_STOP_with_no_gate(self):
        """A stop caller has no reason to be opt-in; always-stop is the safe
        direction, and the destroy's flag must not leak into it."""
        from sglang.srt.managers import scheduler_teardown as td

        w = _watchdog()
        sched = SimpleNamespace(
            server_args=SimpleNamespace(scheduler_distributed_teardown=False)
        )
        with mock.patch.object(barlink_liveness, "_watchdog", w):
            self.assertEqual(td.release_barlink_watchdog(sched, graceful=True), "joined")

    def test_no_watchdog_installed_is_a_quiet_no_op(self):
        from sglang.srt.managers import scheduler_teardown as td

        with mock.patch.object(barlink_liveness, "_watchdog", None):
            self.assertIsNone(
                td.release_barlink_watchdog(SimpleNamespace(), graceful=True)
            )

    def test_it_NEVER_RAISES(self):
        from sglang.srt.managers import scheduler_teardown as td

        class Boom:
            def stop(self, timeout_s=None):
                raise RuntimeError("nope")

        with mock.patch.object(barlink_liveness, "_watchdog", Boom()):
            self.assertIsNone(
                td.release_barlink_watchdog(SimpleNamespace(), graceful=True)
            )


class TestTheOrderingIsByConstruction(_Base):
    def test_release_distributed_STOPS_THE_WATCHDOG_BEFORE_DESTROYING(self):
        """CAN-FAIL by reordering. Patches the FUNCTIONS, not sys.modules --
        `from X import Y` resolves the package attribute, so a sys.modules
        patch is silently bypassed."""
        from sglang.srt.managers import scheduler_teardown as td

        order = []
        with mock.patch(
            "sglang.srt.distributed.device_communicators.barlink_liveness"
            ".stop_watchdog",
            side_effect=lambda *a, **k: (order.append("stop_watchdog"), "joined")[1],
        ), mock.patch(
            "sglang.srt.distributed.parallel_state.destroy_model_parallel",
            side_effect=lambda: order.append("destroy_model_parallel"),
        ), mock.patch(
            "sglang.srt.distributed.parallel_state.destroy_distributed_environment",
            side_effect=lambda: order.append("destroy_world"),
        ):
            td.release_distributed(
                SimpleNamespace(
                    server_args=SimpleNamespace(scheduler_distributed_teardown=True)
                ),
                graceful=True,
            )

        self.assertIn("stop_watchdog", order, "the watchdog was never stopped")
        self.assertLess(
            order.index("stop_watchdog"),
            order.index("destroy_model_parallel"),
            "the groups were destroyed while the watchdog was still polling",
        )

    def test_the_scheduler_calls_it_before_release_distributed(self):
        import inspect

        from sglang.srt.managers import scheduler as sched_mod

        src = inspect.getsource(sched_mod)
        self.assertLess(
            src.index("release_barlink_watchdog(scheduler"),
            src.index("release_distributed(scheduler"),
        )

    def test_the_call_site_EXISTS(self):
        import ast
        import inspect

        from sglang.srt.managers import scheduler as sched_mod

        called = {
            n.func.id
            for n in ast.walk(ast.parse(inspect.getsource(sched_mod)))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        self.assertIn("release_barlink_watchdog", called)


if __name__ == "__main__":
    unittest.main()
