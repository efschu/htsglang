"""Peer death must end a collective wait, not extend it (task #312).

WHY THIS EXISTS
---------------
Twice in one fair window a rank was SIGKILLed by the OOM killer during CUDA
graph capture, and both times the surviving ranks kept spinning: 100 % SM at
zero PCIe, unbounded, until somebody intervened by hand. Two mechanisms
produce that picture and neither can see a dead process:

* the gloo ``cpu_group`` every barlink handshake runs on is built with a
  hardcoded 7200 s timeout, which is a hang as far as any operator is
  concerned;
* the BAR1 spin kernels do carry a cycle deadline, but it is rank-local, it
  is multiplied 40x inside the JIT cold-build window -- which IS the capture
  window -- and its expiry writes a status word that no production path
  reads.

The tests below are the CPU-side falsifiers for the fix. Each one names the
property that would have to break for the endless spin to come back:

1. a wait whose predicate never becomes true ends at the deadline, loudly;
2. a wait whose peer is gone ends BEFORE the deadline, naming the rank, the
   host and the pid;
3. a recycled pid does not read as "still alive";
4. a cross-host peer is reported undecidable, not guessed dead;
5. the success path pays nothing -- no clock read, no liveness probe;
6. the kill switch restores the previous, unbounded call exactly;
7. the watchdog turns a dead peer into a tripped abort word, which is the
   only channel that can end a DEVICE-side spin during graph replay;
8. the abort word degrades to "absent" rather than refusing to boot when the
   runtime will not map it.

Everything here is hermetic: no CUDA, no torch.distributed, no second
process except the two the liveness tests spawn deliberately and reap.
"""

import multiprocessing
import os
import time
import unittest
from unittest import mock

from sglang.srt.distributed.device_communicators import barlink_liveness as live
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


def _sleep_forever() -> None:  # pragma: no cover - runs in a child process
    time.sleep(3600)


class _EnvGuard:
    """Set env vars for the duration of a block and restore them exactly.

    The feature switch is PINNED rather than inherited unless the test names
    it. A test that asserts a deadline fires must not silently take the
    disabled path when the ambient environment happens to carry
    ``SGLANG_BARLINK_PEER_LIVENESS=0`` -- that path is an unbounded loop, so
    such a test does not fail, it hangs the whole run. That is the same
    "wedge instead of a verdict" shape this module exists to remove, and it
    is worth spending three lines to keep out of the test harness too.
    """

    def __init__(self, **values):
        values.setdefault(live.ENV_ENABLE, "1")
        self._values = {
            k: (str(v) if v is not None else None) for k, v in values.items()
        }
        self._saved = {}

    def __enter__(self):
        for key, value in self._values.items():
            self._saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return self

    def __exit__(self, *exc):
        for key, old in self._saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        return False


class LivenessTestBase(CustomTestCase):
    def setUp(self):
        live.reset_for_test()
        self._children = []
        # Pin the feature ON for the whole case, whatever the ambient
        # environment says. Individual tests that are ABOUT the off state
        # re-set it through their own _EnvGuard, which wins for its block.
        self._saved_enable = os.environ.get(live.ENV_ENABLE)
        os.environ[live.ENV_ENABLE] = "1"

    def tearDown(self):
        for proc in self._children:
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=5)
        live.reset_for_test()
        if self._saved_enable is None:
            os.environ.pop(live.ENV_ENABLE, None)
        else:
            os.environ[live.ENV_ENABLE] = self._saved_enable

    def spawn_child(self):
        """A real, live process whose pid the table can be pointed at."""
        proc = multiprocessing.Process(target=_sleep_forever, daemon=True)
        proc.start()
        self._children.append(proc)
        return proc

    @staticmethod
    def table_with(entries, self_rank=0):
        return live.PeerTable(entries, self_rank=self_rank)

    @staticmethod
    def me(rank=0):
        return live.local_identity(rank)


class TestPidLiveness(LivenessTestBase):
    """The one fact the whole fix rests on: does that process still exist?"""

    def test_own_pid_is_alive(self):
        self.assertTrue(live.pid_alive(os.getpid()))

    def test_absurd_pid_is_dead(self):
        # 2**22 is above the default pid_max on Linux, so it cannot exist.
        self.assertFalse(live.pid_alive(4194305))

    def test_nonpositive_pid_is_dead(self):
        self.assertFalse(live.pid_alive(0))
        self.assertFalse(live.pid_alive(-1))

    def test_a_killed_child_reads_as_dead(self):
        proc = self.spawn_child()
        self.assertTrue(live.pid_alive(proc.pid))
        proc.kill()
        proc.join(timeout=5)
        self.assertFalse(live.pid_alive(proc.pid))

    def test_permission_error_counts_as_alive(self):
        """A process owned by somebody else still exists.

        Reading EPERM as "gone" would abort a healthy group the moment a rank
        ran under a different uid.
        """
        with mock.patch("os.kill", side_effect=PermissionError):
            self.assertTrue(live.pid_alive(12345))


class TestPeerTable(LivenessTestBase):
    def test_self_is_never_dead(self):
        table = self.table_with([self.me(0)], self_rank=0)
        self.assertEqual(table.state(table.entries[0]), live.ALIVE)
        self.assertEqual(table.dead_peers(), [])

    def test_dead_peer_is_named_with_host_and_pid(self):
        gone = live.PeerIdentity(rank=1, host=live._hostname(), pid=4194305, boot="")
        table = self.table_with([self.me(0), gone], self_rank=0)
        dead = table.dead_peers()
        self.assertEqual([e.rank for e in dead], [1])
        self.assertIn("rank 1", table.describe_dead())
        self.assertIn("4194305", table.describe_dead())

    def test_live_child_is_alive(self):
        proc = self.spawn_child()
        peer = live.PeerIdentity(
            rank=1,
            host=live._hostname(),
            pid=proc.pid,
            boot=live._boot_marker(proc.pid),
        )
        table = self.table_with([self.me(0), peer], self_rank=0)
        self.assertEqual(table.dead_peers(), [])
        proc.kill()
        proc.join(timeout=5)
        self.assertEqual([e.rank for e in table.dead_peers()], [1])

    def test_recycled_pid_does_not_read_as_alive(self):
        """Same pid, different process start time = the original is gone.

        Without this, a busy box that recycles a pid between rendezvous and
        probe would silently make the whole check useless.
        """
        peer = live.PeerIdentity(
            rank=1,
            host=live._hostname(),
            pid=os.getpid(),
            boot="0",  # our real /proc start time is never "0"
        )
        table = self.table_with([self.me(0), peer], self_rank=0)
        self.assertEqual([e.rank for e in table.dead_peers()], [1])

    def test_cross_host_peer_is_undecidable_not_dead(self):
        """A remote pid means nothing locally, and guessing would be worse.

        The deadline carries this case; a false "dead" would abort a healthy
        group.
        """
        peer = live.PeerIdentity(rank=1, host="some-other-box", pid=4194305, boot="")
        table = self.table_with([self.me(0), peer], self_rank=0)
        self.assertEqual(table.state(peer), live.UNKNOWN)
        self.assertEqual(table.dead_peers(), [])

    def test_opaque_peer_is_undecidable(self):
        """A peer on an older build publishes nothing; that is not death."""
        peer = live.PeerIdentity(rank=1, host="", pid=0, boot="")
        table = self.table_with([self.me(0), peer], self_rank=0)
        self.assertEqual(table.state(peer), live.UNKNOWN)


class TestBoundedPollDeadline(LivenessTestBase):
    def test_deadline_fires_and_names_the_wait(self):
        with _EnvGuard(SGLANG_BARLINK_PEER_TIMEOUT_S=0.3, SGLANG_BARLINK_PEER_PROBE_S=0.05):
            started = time.monotonic()
            with self.assertRaises(live.CollectiveTimeoutError) as ctx:
                live.bounded_poll(lambda: False, "never-ready")
            waited = time.monotonic() - started
        self.assertIn("never-ready", str(ctx.exception))
        self.assertIn("no peer could be proven dead", str(ctx.exception))
        self.assertLess(waited, 5.0, "the deadline must actually bound the wait")
        self.assertGreaterEqual(waited, 0.25)

    def test_deadline_error_names_the_knob_to_change(self):
        with _EnvGuard(SGLANG_BARLINK_PEER_TIMEOUT_S=0.2):
            with self.assertRaises(live.CollectiveTimeoutError) as ctx:
                live.bounded_poll(lambda: False, "never-ready")
        self.assertIn(live.ENV_TIMEOUT_S, str(ctx.exception))
        self.assertIn(live.ENV_ENABLE, str(ctx.exception))

    def test_a_wait_that_resolves_late_still_succeeds(self):
        deadline = time.monotonic() + 0.2
        with _EnvGuard(
            SGLANG_BARLINK_PEER_TIMEOUT_S=10.0, SGLANG_BARLINK_PEER_PROBE_S=0.05
        ):
            live.bounded_poll(lambda: time.monotonic() >= deadline, "slow-but-healthy")

    def test_peer_census_is_in_the_timeout_message(self):
        proc = self.spawn_child()
        peer = live.PeerIdentity(
            rank=1,
            host=live._hostname(),
            pid=proc.pid,
            boot=live._boot_marker(proc.pid),
        )
        table = self.table_with([self.me(0), peer], self_rank=0)
        with _EnvGuard(SGLANG_BARLINK_PEER_TIMEOUT_S=0.2, SGLANG_BARLINK_PEER_PROBE_S=0.05):
            with self.assertRaises(live.CollectiveTimeoutError) as ctx:
                live.bounded_poll(lambda: False, "wedged", table=table)
        self.assertIn("Peer census", str(ctx.exception))
        self.assertIn("alive", str(ctx.exception))


class TestBoundedPollPeerLoss(LivenessTestBase):
    """The success criterion: a killed rank ends the wait well inside 60 s."""

    def test_dead_peer_beats_the_deadline(self):
        proc = self.spawn_child()
        peer = live.PeerIdentity(
            rank=1,
            host=live._hostname(),
            pid=proc.pid,
            boot=live._boot_marker(proc.pid),
        )
        table = self.table_with([self.me(0), peer], self_rank=0)
        proc.kill()
        proc.join(timeout=5)

        # A deadline far beyond the 60 s criterion, so only the liveness
        # check can end this wait.
        with _EnvGuard(
            SGLANG_BARLINK_PEER_TIMEOUT_S=600.0, SGLANG_BARLINK_PEER_PROBE_S=0.05
        ):
            started = time.monotonic()
            with self.assertRaises(live.PeerLostError) as ctx:
                live.bounded_poll(lambda: False, "all-reduce chunk 3", table=table)
            waited = time.monotonic() - started

        message = str(ctx.exception)
        self.assertLess(waited, 60.0, "a dead peer must be caught inside 60 s")
        self.assertIn("all-reduce chunk 3", message)
        self.assertIn("rank 1", message)
        self.assertIn(str(proc.pid), message)
        self.assertIn(live._hostname(), message)
        self.assertIn("7200s", message, "the message must say what it saved you from")

    def test_peer_loss_trips_registered_abort_windows(self):
        """The host wait and the device spin end together, not separately."""
        window = _FakeAbortWindow()
        live.register_abort_window(window)
        peer = live.PeerIdentity(rank=1, host=live._hostname(), pid=4194305, boot="")
        table = self.table_with([self.me(0), peer], self_rank=0)
        with _EnvGuard(
            SGLANG_BARLINK_PEER_TIMEOUT_S=600.0, SGLANG_BARLINK_PEER_PROBE_S=0.0
        ):
            with self.assertRaises(live.PeerLostError):
                live.bounded_poll(lambda: False, "mesh barrier", table=table)
        self.assertTrue(window.tripped)
        self.assertIn("rank 1", window.reason)

    def test_on_abort_hook_runs_before_the_raise(self):
        calls = []
        peer = live.PeerIdentity(rank=1, host=live._hostname(), pid=4194305, boot="")
        table = self.table_with([self.me(0), peer], self_rank=0)
        with _EnvGuard(
            SGLANG_BARLINK_PEER_TIMEOUT_S=600.0, SGLANG_BARLINK_PEER_PROBE_S=0.0
        ):
            with self.assertRaises(live.PeerLostError):
                live.bounded_poll(
                    lambda: False,
                    "ring step 1",
                    table=table,
                    on_abort=lambda: calls.append("aborted"),
                )
        self.assertEqual(calls, ["aborted"])

    def test_check_peers_is_a_one_shot_version_of_the_same_verdict(self):
        peer = live.PeerIdentity(rank=2, host=live._hostname(), pid=4194305, boot="")
        table = self.table_with([self.me(0), peer], self_rank=0)
        with self.assertRaises(live.PeerLostError) as ctx:
            live.check_peers("rendezvous broadcast", table=table)
        self.assertIn("rank 2", str(ctx.exception))
        self.assertIn("rendezvous broadcast", str(ctx.exception))

    def test_check_peers_is_silent_on_a_healthy_group(self):
        table = self.table_with([self.me(0)], self_rank=0)
        live.check_peers("rendezvous broadcast", table=table)

    def test_the_global_registry_is_consulted_when_no_table_is_passed(self):
        """Deep wait sites must not have to thread a table through."""
        peer = live.PeerIdentity(rank=1, host=live._hostname(), pid=4194305, boot="")
        live.register_peer_table(self.table_with([self.me(0), peer], self_rank=0))
        with self.assertRaises(live.PeerLostError):
            live.check_peers("some deep spin")


class TestSuccessPathCost(LivenessTestBase):
    """No extra synchronization on a collective that completes.

    The cost ceiling being asserted: an already-satisfied wait reads no
    clock, probes no peer and sleeps not at all -- it is one predicate call.
    """

    def test_an_already_ready_wait_reads_no_clock(self):
        with mock.patch("time.monotonic", side_effect=AssertionError("clock read")):
            live.bounded_poll(lambda: True, "hot path")

    def test_an_already_ready_wait_probes_no_peer(self):
        with mock.patch.object(
            live, "pid_alive", side_effect=AssertionError("liveness probed")
        ):
            peer = live.PeerIdentity(rank=1, host=live._hostname(), pid=1, boot="")
            table = self.table_with([self.me(0), peer], self_rank=0)
            live.bounded_poll(lambda: True, "hot path", table=table)

    def test_a_wait_inside_the_spin_budget_reads_no_clock(self):
        """512 bare predicate calls before the first ``time.monotonic()``."""
        state = {"n": 0}

        def ready():
            state["n"] += 1
            return state["n"] > 100

        with mock.patch("time.monotonic", side_effect=AssertionError("clock read")):
            live.bounded_poll(ready, "briefly-not-ready")

    def test_the_probe_is_rate_limited_not_per_iteration(self):
        """Cost ceiling: at most one census per SGLANG_BARLINK_PEER_PROBE_S.

        A per-iteration ``kill(pid, 0)`` would put a syscall in the middle of
        a spin loop; this pins the rate down instead of trusting the comment.
        """
        proc = self.spawn_child()
        peer = live.PeerIdentity(
            rank=1,
            host=live._hostname(),
            pid=proc.pid,
            boot=live._boot_marker(proc.pid),
        )
        table = self.table_with([self.me(0), peer], self_rank=0)
        probes = []
        real = live.pid_alive

        def counting(pid):
            probes.append(pid)
            return real(pid)

        with mock.patch.object(live, "pid_alive", counting):
            with _EnvGuard(
                SGLANG_BARLINK_PEER_TIMEOUT_S=0.6, SGLANG_BARLINK_PEER_PROBE_S=0.2
            ):
                with self.assertRaises(live.CollectiveTimeoutError):
                    live.bounded_poll(lambda: False, "wedged", table=table)
        # 0.6 s at one census per 0.2 s is at most 4 probes for the poll plus
        # the census the timeout message builds. Anything per-iteration would
        # be in the thousands.
        self.assertLessEqual(len(probes), 8, f"probe rate not bounded: {len(probes)}")


class TestKillSwitch(LivenessTestBase):
    def test_disabled_restores_the_unbounded_loop(self):
        state = {"n": 0}

        def ready():
            state["n"] += 1
            return state["n"] > 50

        with _EnvGuard(SGLANG_BARLINK_PEER_LIVENESS=0, SGLANG_BARLINK_PEER_TIMEOUT_S=0.001):
            self.assertFalse(live.liveness_enabled())
            # No deadline can fire: the disabled path is the old bare loop.
            live.bounded_poll(ready, "disabled")
        self.assertEqual(state["n"], 51)

    def test_disabled_never_raises_on_a_dead_peer(self):
        peer = live.PeerIdentity(rank=1, host=live._hostname(), pid=4194305, boot="")
        table = self.table_with([self.me(0), peer], self_rank=0)
        with _EnvGuard(SGLANG_BARLINK_PEER_LIVENESS=0):
            live.check_peers("disabled", table=table)

    def test_zero_timeout_also_means_unbounded(self):
        state = {"n": 0}

        def ready():
            state["n"] += 1
            return state["n"] > 10

        with _EnvGuard(SGLANG_BARLINK_PEER_TIMEOUT_S=0):
            live.bounded_poll(ready, "no-deadline")
        self.assertEqual(state["n"], 11)

    def test_install_returns_none_when_disabled(self):
        with _EnvGuard(SGLANG_BARLINK_PEER_LIVENESS=0):
            self.assertIsNone(live.install(object()))
        self.assertEqual(live.registered_tables(), [])

    def test_watchdog_is_separately_switchable(self):
        with _EnvGuard(SGLANG_BARLINK_PEER_LIVENESS=1, SGLANG_BARLINK_PEER_WATCHDOG=0):
            self.assertTrue(live.liveness_enabled())
            self.assertFalse(live.watchdog_enabled())
            self.assertIsNone(live.ensure_watchdog())


class TestColdBuildWindow(LivenessTestBase):
    """A first boot must not be mistaken for a dead peer.

    The ranks arrive at the pre-capture warmup minutes apart while each one
    sits in nvcc. The device deadline already scales there; the host deadline
    has to scale with it or the fix would turn a cold cache into a crash.
    """

    def test_deadline_is_scaled_inside_the_window(self):
        from sglang.srt.utils import jit_cold_build

        with _EnvGuard(
            SGLANG_BARLINK_PEER_TIMEOUT_S=100.0, SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT=7
        ):
            self.assertEqual(live.wait_timeout_s(), 100.0)
            with jit_cold_build.cold_build_window("test"):
                self.assertEqual(live.wait_timeout_s(), 700.0)
            self.assertEqual(live.wait_timeout_s(), 100.0)

    def test_a_dead_peer_is_still_caught_inside_the_window(self):
        """Scaling the deadline must not scale the liveness verdict.

        Death is a fact, not a duration -- the whole point of the check is
        that it is decidable while the deadline is not.
        """
        from sglang.srt.utils import jit_cold_build

        peer = live.PeerIdentity(rank=1, host=live._hostname(), pid=4194305, boot="")
        table = self.table_with([self.me(0), peer], self_rank=0)
        with _EnvGuard(
            SGLANG_BARLINK_PEER_TIMEOUT_S=600.0,
            SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT=40,
            SGLANG_BARLINK_PEER_PROBE_S=0.0,
        ):
            with jit_cold_build.cold_build_window("test"):
                started = time.monotonic()
                with self.assertRaises(live.PeerLostError):
                    live.bounded_poll(lambda: False, "capture barrier", table=table)
                self.assertLess(time.monotonic() - started, 60.0)


class _FakeAbortWindow:
    """The AbortWindow contract without a CUDA runtime."""

    def __init__(self):
        self.device_ptr = 0
        self.reason = None
        self.word = 0

    @property
    def tripped(self):
        return self.reason is not None

    def trip(self, reason):
        if self.reason is None:
            self.reason = reason
            self.word = 1

    def value(self):
        return self.word


class TestAbortWindow(LivenessTestBase):
    def test_trip_is_idempotent_and_keeps_the_first_reason(self):
        window = _FakeAbortWindow()
        live.register_abort_window(window)
        live.trip_all_abort_windows("first")
        live.trip_all_abort_windows("second")
        self.assertEqual(window.reason, "first")

    def test_unregistered_windows_are_left_alone(self):
        window = _FakeAbortWindow()
        live.register_abort_window(window)
        live.unregister_abort_window(window)
        live.trip_all_abort_windows("reason")
        self.assertFalse(window.tripped)

    def test_a_real_window_degrades_when_the_runtime_will_not_map_it(self):
        """No CUDA is not a reason to refuse to boot.

        The kernels see a null pointer and skip their probe; the host-side
        bounds are unaffected. That is a smaller fix, not a broken one.
        """
        with mock.patch.object(
            live, "_map_host_word", side_effect=RuntimeError("no runtime")
        ):
            window = live.AbortWindow()
        self.assertEqual(window.device_ptr, 0)
        self.assertFalse(window.tripped)
        window.trip("still works host-side")
        self.assertTrue(window.tripped)
        window.close()


class TestWatchdog(LivenessTestBase):
    """The only channel that can end a device-side spin during graph replay.

    No host code runs inside a captured collective, so a thread outside it
    has to write the word the kernels poll.
    """

    def test_probe_trips_the_window_on_a_dead_peer(self):
        window = _FakeAbortWindow()
        live.register_abort_window(window)
        peer = live.PeerIdentity(rank=1, host=live._hostname(), pid=4194305, boot="")
        live.register_peer_table(self.table_with([self.me(0), peer], self_rank=0))

        watchdog = live.PeerWatchdog(interval_s=0.05)
        dead = watchdog.probe_once()

        self.assertEqual([e.rank for e in dead], [1])
        self.assertTrue(window.tripped)
        self.assertIn("peer rank gone", window.reason)
        self.assertEqual(window.value(), 1)

    def test_probe_is_silent_on_a_healthy_group(self):
        window = _FakeAbortWindow()
        live.register_abort_window(window)
        live.register_peer_table(self.table_with([self.me(0)], self_rank=0))
        watchdog = live.PeerWatchdog(interval_s=0.05)
        self.assertEqual(watchdog.probe_once(), [])
        self.assertFalse(window.tripped)

    def test_the_thread_catches_a_kill_within_a_probe_interval(self):
        proc = self.spawn_child()
        peer = live.PeerIdentity(
            rank=1,
            host=live._hostname(),
            pid=proc.pid,
            boot=live._boot_marker(proc.pid),
        )
        live.register_peer_table(self.table_with([self.me(0), peer], self_rank=0))
        window = _FakeAbortWindow()
        live.register_abort_window(window)

        watchdog = live.PeerWatchdog(interval_s=0.05).start()
        try:
            time.sleep(0.2)
            self.assertFalse(window.tripped, "must not trip on a live peer")
            proc.kill()
            proc.join(timeout=5)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not window.tripped:
                time.sleep(0.02)
            self.assertTrue(window.tripped, "watchdog missed the kill")
            self.assertIn(str(proc.pid), window.reason)
        finally:
            watchdog.stop()

    def test_a_window_registered_after_the_death_is_still_tripped(self):
        """The thread keeps probing, so ordering at bring-up cannot lose it."""
        peer = live.PeerIdentity(rank=1, host=live._hostname(), pid=4194305, boot="")
        live.register_peer_table(self.table_with([self.me(0), peer], self_rank=0))
        watchdog = live.PeerWatchdog(interval_s=0.05).start()
        try:
            time.sleep(0.15)
            late = _FakeAbortWindow()
            live.register_abort_window(late)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not late.tripped:
                time.sleep(0.02)
            self.assertTrue(late.tripped)
        finally:
            watchdog.stop()

    def test_a_failing_probe_does_not_kill_the_thread(self):
        watchdog = live.PeerWatchdog(interval_s=0.02)
        with mock.patch.object(
            live, "any_dead_peers", side_effect=RuntimeError("boom")
        ):
            watchdog.start()
            time.sleep(0.15)
            alive = watchdog._thread is not None and watchdog._thread.is_alive()
        watchdog.stop()
        self.assertTrue(alive, "a watchdog that dies is worse than none")

    def test_ensure_watchdog_is_idempotent(self):
        first = live.ensure_watchdog()
        second = live.ensure_watchdog()
        self.assertIs(first, second)


class TestConfigSurface(LivenessTestBase):
    def test_every_knob_is_registered_in_environ(self):
        """A knob the operator cannot find is a knob that does not exist."""
        from sglang.srt.environ import envs

        for name in (
            "SGLANG_BARLINK_PEER_LIVENESS",
            "SGLANG_BARLINK_PEER_TIMEOUT_S",
            "SGLANG_BARLINK_PEER_PROBE_S",
            "SGLANG_BARLINK_PEER_WATCHDOG",
        ):
            self.assertTrue(hasattr(envs, name), f"{name} missing from environ.py")

    def test_env_names_match_the_module_constants(self):
        self.assertEqual(live.ENV_ENABLE, "SGLANG_BARLINK_PEER_LIVENESS")
        self.assertEqual(live.ENV_TIMEOUT_S, "SGLANG_BARLINK_PEER_TIMEOUT_S")
        self.assertEqual(live.ENV_PROBE_S, "SGLANG_BARLINK_PEER_PROBE_S")
        self.assertEqual(live.ENV_WATCHDOG, "SGLANG_BARLINK_PEER_WATCHDOG")

    def test_defaults_are_the_documented_ones(self):
        with _EnvGuard(
            SGLANG_BARLINK_PEER_LIVENESS=None,
            SGLANG_BARLINK_PEER_TIMEOUT_S=None,
            SGLANG_BARLINK_PEER_PROBE_S=None,
            SGLANG_BARLINK_PEER_WATCHDOG=None,
        ):
            self.assertTrue(live.liveness_enabled())
            self.assertEqual(live.wait_timeout_s(), 120.0)
            self.assertEqual(live.probe_interval_s(), 1.0)
            self.assertTrue(live.watchdog_enabled())

    def test_a_garbage_value_falls_back_to_the_default(self):
        with _EnvGuard(SGLANG_BARLINK_PEER_TIMEOUT_S="not-a-number"):
            self.assertEqual(live.wait_timeout_s(), 120.0)

    def test_describe_config_names_all_four_knobs(self):
        text = live.describe_config()
        for name in (
            live.ENV_ENABLE,
            live.ENV_TIMEOUT_S,
            live.ENV_PROBE_S,
            live.ENV_WATCHDOG,
        ):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
