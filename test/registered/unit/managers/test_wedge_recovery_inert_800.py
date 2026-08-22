"""#800: the admission-wedge recovery must not be able to do nothing quietly.

THE SPECIMEN. Boot r8 @ 4c20c5b0d3, evidence-665-f1/wedge_1122_112408/boot.log.
The #699 detector emitted 46 alarm lines, correctly. The #788 recovery rung
announced four times and reported a result three times, each result reading:

    RECOVERY: forced-admission attempt returned None (None means the gate is
    off or inert on this boot -- ...)

Every clause of that line is a defect this file guards:

  * "None means the gate is off or inert" -- it did not. The same boot logged
    ``[#656 CORRIDOR-ADMISSION] ARMED on device 0`` on all three ranks at
    10:44:01Z and cleared four prefill admissions at 11:19:47Z reclaiming
    232 / 1112 / 1126 / 1216 MiB. The ``None`` came from the
    ``free - want >= floor`` exit: the corridor was HEALTHY. Six exits return
    ``None``; the message named one of them and was wrong.
  * "the attempt ran" -- on PP0 it is not established that it returned. Four
    announcements produced three results; the missing one is PP0's. PP0
    announced at 11:23:09 and emitted ZERO log lines of any kind over the
    remaining 88 s, while PP1 and PP2 emitted 18 in that window.
  * Nothing consumed any of it. The result went to ``logger.error`` and
    stopped there, so the code reported its own ineffectiveness three times
    and no state anywhere changed.

WHAT MAKES THIS FILE GO RED. Each test names the mutant it kills. The
load-bearing ones are the two that a "simplification" would reintroduce:

  * ``test_specimen_shape_names_the_exit_and_does_not_say_gate_is_off`` --
    red the moment an exit stops writing its reason and the caller is back to
    inferring a cause from ``None``.
  * ``test_scheduler_thread_silence_becomes_its_own_state_and_escalates`` --
    red the moment the recovery can be posted, ignored and forgotten. This is
    the specimen reproduced end to end: gate healthy, scheduler thread not
    looping, and the system must now say so in a distinct, greppable line.

Hermetic: no CUDA, no scheduler boot. Every clock is injected except in the one
test that must prove the real watchdog THREAD calls the poller.
"""

import time
import unittest
from types import SimpleNamespace

from sglang.srt.managers.corridor_admission import (
    ACTUATING_REASONS,
    REASON_CLEARED,
    REASON_COOLDOWN,
    REASON_FREE_PROBE_FAILED,
    REASON_HEADROOM_SUFFICIENT,
    REASON_LADDER_RAISED,
    REASON_NEVER_CALLED,
    REASON_NO_GUARD,
    REASON_NO_SCHEDULER,
    REASON_PHASE_FLIP_OFF,
    REASON_SHORT,
    PrefillAdmissionGate,
    guard_prefill_admission_explained,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    AdmissionWedgeRecovery,
    create_admission_wedge_watchdog,
    make_admission_wedge_poller,
)
from sglang.srt.managers.wedge_recovery import (
    RECOVERY_CHANNEL_ATTR,
    STATE_ACTUATED,
    STATE_INERT,
    STATE_PENDING,
    STATE_UNCONSUMED,
    WedgeRecoveryChannel,
    drain_recovery_request,
    get_recovery_channel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1024 * 1024


class _FakeVerdict:
    def __init__(self, ok=True, law_breached=False, reclaimed=0, detail="d"):
        self.ok = ok
        self.law_breached = law_breached
        self.reclaimed = reclaimed
        self.detail = detail


class _FakeGuard:
    """The corridor guard, reduced to the three things the gate reads."""

    def __init__(self, free_mib=4096, floor_mib=1331, verdict=None, raises=None):
        self._free = free_mib * MIB
        self.floor_bytes = floor_mib * MIB
        self._verdict = verdict or _FakeVerdict()
        self._raises = raises
        self.providers = ["allocator-cache"]
        self.delta_bytes = 256 * MIB
        self.device_index = 0

    def free_bytes(self):
        if self._raises == "free":
            raise RuntimeError("nvml gone")
        return self._free

    def ensure_headroom(self, want, reason="", refusal_is_fatal=False):
        if self._raises == "ladder":
            raise RuntimeError("ladder tore")
        return self._verdict


def _gate(guard, *, cooldown_s=0.0, clock=None):
    """A PrefillAdmissionGate whose guard lookup is pinned to ``guard``."""
    scheduler = SimpleNamespace(server_args=SimpleNamespace(enable_phase_flip=True))
    gate = PrefillAdmissionGate(
        scheduler, cooldown_s=cooldown_s, clock=clock or (lambda: 0.0)
    )
    gate._guard = lambda: guard
    gate._maybe_lend = lambda *a, **k: None
    gate._announce_once = lambda *a, **k: None
    return gate


class _FakeSessionController:
    def maybe_reap(self, now):
        return None


class _FakeFlushWrapper:
    def check_pending(self):
        return None


class _FakeScheduler:
    """Only the attributes the wedge path reads. Never a scheduler boot."""

    def __init__(self, *, queued=2, running=0, age=200.0, enable_phase_flip=True):
        self.is_initializing = False
        self.waiting_queue = [object()] * queued
        self.running_batch = SimpleNamespace(reqs=[object()] * running)
        self.last_first_token_progress_time = -age
        self.server_args = SimpleNamespace(enable_phase_flip=enable_phase_flip)
        self.session_controller = _FakeSessionController()
        self.flush_wrapper = _FakeFlushWrapper()
        self.external_corpus_manager = None
        self.return_health_check_ipcs = []


# --- part 1: every exit of the gate names itself -------------------------


class TestGateExitsAreNamed(CustomTestCase):
    """MUTANT KILLED: delete any ``self.last_reason = ...`` assignment.

    Each exit below is reached by construction, so removing the assignment on
    that edge leaves the previous reason (or ``never-called``) in place and
    the corresponding assertion fails. One dying mutant per call edge.
    """

    def test_headroom_sufficient(self):
        gate = _gate(_FakeGuard(free_mib=4096, floor_mib=1331))
        self.assertIsNone(gate.before_admission(0))
        self.assertEqual(gate.last_reason, REASON_HEADROOM_SUFFICIENT)

    def test_no_guard(self):
        gate = _gate(None)
        self.assertIsNone(gate.before_admission(0))
        self.assertEqual(gate.last_reason, REASON_NO_GUARD)

    def test_free_probe_failed(self):
        gate = _gate(_FakeGuard(raises="free"))
        self.assertIsNone(gate.before_admission(0))
        self.assertEqual(gate.last_reason, REASON_FREE_PROBE_FAILED)

    def test_cooldown(self):
        ticks = iter([0.0, 0.0, 0.01, 0.01])
        gate = _gate(
            _FakeGuard(free_mib=100, floor_mib=1331),
            cooldown_s=10.0,
            clock=lambda: next(ticks),
        )
        self.assertIsNotNone(gate.before_admission(0))
        self.assertEqual(gate.last_reason, REASON_CLEARED)
        self.assertIsNone(gate.before_admission(0))
        self.assertEqual(gate.last_reason, REASON_COOLDOWN)

    def test_ladder_raised(self):
        gate = _gate(_FakeGuard(free_mib=100, floor_mib=1331, raises="ladder"))
        self.assertIsNone(gate.before_admission(0))
        self.assertEqual(gate.last_reason, REASON_LADDER_RAISED)

    def test_cleared_and_short(self):
        gate = _gate(_FakeGuard(free_mib=100, floor_mib=1331))
        self.assertIsNotNone(gate.before_admission(0))
        self.assertEqual(gate.last_reason, REASON_CLEARED)

        breached = _FakeVerdict(ok=True, law_breached=True)
        gate = _gate(_FakeGuard(free_mib=100, floor_mib=1331, verdict=breached))
        self.assertIsNotNone(gate.before_admission(0))
        self.assertEqual(gate.last_reason, REASON_SHORT)

    def test_phase_flip_off_and_no_scheduler(self):
        off = SimpleNamespace(server_args=SimpleNamespace(enable_phase_flip=False))
        self.assertEqual(
            guard_prefill_admission_explained(off, 0).reason, REASON_PHASE_FLIP_OFF
        )
        self.assertEqual(
            guard_prefill_admission_explained(None, 0).reason, REASON_NO_SCHEDULER
        )

    def test_only_a_ladder_run_counts_as_actuation(self):
        """``verdict is not None`` is NOT the actuation bit, and must not become it."""
        self.assertEqual(ACTUATING_REASONS, frozenset({REASON_CLEARED, REASON_SHORT}))
        for reason in (
            REASON_HEADROOM_SUFFICIENT,
            REASON_NO_GUARD,
            REASON_COOLDOWN,
            REASON_FREE_PROBE_FAILED,
            REASON_LADDER_RAISED,
            REASON_PHASE_FLIP_OFF,
            REASON_NO_SCHEDULER,
            REASON_NEVER_CALLED,
        ):
            self.assertNotIn(reason, ACTUATING_REASONS, reason)

    def test_every_exit_is_distinguishable(self):
        """The whole point: no two exits may share a name."""
        reasons = [
            REASON_CLEARED,
            REASON_SHORT,
            REASON_PHASE_FLIP_OFF,
            REASON_NO_GUARD,
            REASON_FREE_PROBE_FAILED,
            REASON_HEADROOM_SUFFICIENT,
            REASON_COOLDOWN,
            REASON_LADDER_RAISED,
            REASON_NO_SCHEDULER,
            REASON_NEVER_CALLED,
        ]
        self.assertEqual(len(reasons), len(set(reasons)))


class TestSpecimenShape(CustomTestCase):
    def test_specimen_shape_names_the_exit_and_does_not_say_gate_is_off(self):
        """The 2026-08-22 11:22Z reading, reproduced.

        Gate ARMED, corridor healthy (free 4096 MiB over a 1331 MiB arming
        floor -- the specimen's own PP0/PP1 floor), tokens=0 exactly as the
        recovery calls it. The old path turned this into "the gate is off or
        inert on this boot". It must now be named for what it is, and must
        NOT be confusable with the genuinely-inert reading.

        MUTANT KILLED: any change that makes the healthy-corridor exit and
        the no-guard exit report the same thing again.
        """
        healthy = _gate(_FakeGuard(free_mib=4096, floor_mib=1331))
        self.assertIsNone(healthy.before_admission(0))

        genuinely_inert = _gate(None)
        self.assertIsNone(genuinely_inert.before_admission(0))

        self.assertEqual(healthy.last_reason, REASON_HEADROOM_SUFFICIENT)
        self.assertEqual(genuinely_inert.last_reason, REASON_NO_GUARD)
        self.assertNotEqual(healthy.last_reason, genuinely_inert.last_reason)


# --- part 2: the production drain edge -----------------------------------


class TestSchedulerDrainsOnItsOwnThread(CustomTestCase):
    def test_process_input_requests_is_the_drain_edge(self):
        """The REAL ``Scheduler.process_input_requests`` must consume a request.

        Called unbound on a duck-typed scheduler, so this asserts about the
        production function every loop family reaches once per iteration --
        not about a helper that resembles it. A test that called
        ``drain_recovery_request`` directly would prove the drain works and
        say nothing about whether anything calls it, which is precisely the
        gap that let #788's recovery ship inert.

        MUTANT KILLED: remove the ``drain_recovery_request(self)`` line from
        ``process_input_requests``, or move it into a loop family that this
        boot does not run.
        """
        sched = _FakeScheduler()
        channel = get_recovery_channel(sched)
        gate = _gate(_FakeGuard(free_mib=4096, floor_mib=1331))
        setattr(sched, "phase_flip_corridor_admission", gate)

        seq = channel.post(0.0, tokens=0)
        self.assertEqual(channel.acked_seq, 0, "nothing may ack before the drain")

        Scheduler.process_input_requests(sched, [])

        self.assertEqual(channel.acked_seq, seq)
        self.assertEqual(channel.acked_reason, REASON_HEADROOM_SUFFICIENT)

    def test_drain_is_a_no_op_when_nothing_was_posted(self):
        """The cost on a boot that never wedges is one attribute read."""
        sched = _FakeScheduler()
        self.assertIsNone(drain_recovery_request(sched))
        self.assertIsNone(getattr(sched, RECOVERY_CHANNEL_ATTR, None))

    def test_drain_survives_an_actuator_that_raises(self):
        sched = _FakeScheduler()
        channel = get_recovery_channel(sched)
        setattr(
            sched,
            "phase_flip_corridor_admission",
            _gate(_FakeGuard(free_mib=100, floor_mib=1331, raises="ladder")),
        )
        seq = channel.post(0.0, tokens=0)
        Scheduler.process_input_requests(sched, [])
        self.assertEqual(channel.acked_seq, seq)
        self.assertEqual(channel.acked_reason, REASON_LADDER_RAISED)


# --- part 3: the watchdog thread only posts ------------------------------


class TestWatchdogThreadDoesNotActuate(CustomTestCase):
    def test_poller_posts_and_never_runs_the_ladder_itself(self):
        """The watchdog poll must not touch the gate.

        This is the PP0 finding as a test: #788 ran the CUDA-touching relief
        ladder on the watchdog thread, and on the specimen the thread that
        made that call never spoke again. The poller may post; it may not
        actuate.

        MUTANT KILLED: put ``guard_prefill_admission(scheduler, 0)`` back into
        the watchdog thread.
        """
        sched = _FakeScheduler(queued=2, running=0, age=200.0)
        touched = []

        class _TattlingGate:
            last_reason = REASON_HEADROOM_SUFFICIENT

            def before_admission(self, tokens):
                touched.append(tokens)
                return None

        setattr(sched, "phase_flip_corridor_admission", _TattlingGate())

        make_admission_wedge_poller(sched)()

        channel = get_recovery_channel(sched)
        self.assertEqual(channel.requested_seq, 1, "the poll must post a request")
        self.assertEqual(touched, [], "the watchdog thread must not run the ladder")
        self.assertEqual(channel.acked_seq, 0, "only the scheduler thread may ack")

    def test_poller_does_not_post_while_the_box_is_serving(self):
        sched = _FakeScheduler(queued=2, running=4, age=200.0)
        make_admission_wedge_poller(sched)()
        self.assertIsNone(getattr(sched, RECOVERY_CHANNEL_ATTR, None))

    def test_the_watchdog_thread_actually_calls_the_poller(self):
        """The LAST untested call edge: thread -> poller.

        Everything else in this file proves the poller and the driver do the
        right thing when called. This proves the thread calls them at all --
        without it, ``create_admission_wedge_watchdog`` could spawn a loop
        that sleeps and does nothing and every other test would stay green,
        which is the shape of the defect this whole file is about.

        MUTANT KILLED: drop the ``poll()`` call from ``_loop``.
        """
        sched = _FakeScheduler(queued=2, running=0, age=200.0)
        thread = create_admission_wedge_watchdog(sched, poll_interval=0.01)
        self.assertTrue(thread.is_alive())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            channel = getattr(sched, RECOVERY_CHANNEL_ATTR, None)
            if channel is not None and channel.requested_seq > 0:
                break
            time.sleep(0.01)
        channel = getattr(sched, RECOVERY_CHANNEL_ATTR, None)
        self.assertIsNotNone(channel, "the watchdog thread never ran a poll")
        self.assertGreater(channel.requested_seq, 0)


# --- part 4: outcomes are consumed, and silence escalates ----------------


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class TestOutcomesAreConsumed(CustomTestCase):
    def _driver(self, sched, clock, **kw):
        return AdmissionWedgeRecovery(sched, clock=clock, **kw)

    def test_scheduler_thread_silence_becomes_its_own_state_and_escalates(self):
        """THE HEADLINE CAN-FAIL TEST -- the specimen, end to end.

        Wedge alarming, corridor healthy, and the scheduler thread never
        reaches ``process_input_requests``. The old path produced seven
        identical ``returned None`` lines and no state. The new path must
        produce UNCONSUMED -- a statement about the scheduler thread, which
        the old path could not make at all -- and, after two of them, a single
        distinct escalation.

        MUTANT KILLED: drop the ack grace check, drop the escalation counter,
        or let a settled outcome go unrecorded. Any of those and this is red.
        """
        clock = _Clock()
        sched = _FakeScheduler(queued=2, running=0, age=200.0)
        driver = self._driver(sched, clock, grace_s=45.0, retry_s=30.0)

        self.assertIsNone(driver.step(alarm=True))  # posts #1
        channel = get_recovery_channel(sched)
        self.assertEqual(channel.requested_seq, 1)

        # Still inside the grace window: not yet a finding.
        clock.advance(10.0)
        self.assertEqual(driver.step(alarm=True).state, STATE_PENDING)

        # Past the grace window and nobody drained it.
        clock.advance(40.0)
        first = driver.step(alarm=True)
        self.assertEqual(first.state, STATE_UNCONSUMED)
        self.assertFalse(channel.escalated, "one is not a run")

        # Second attempt, same silence -> escalation.
        clock.advance(30.0)
        self.assertIsNone(driver.step(alarm=True))  # posts #2
        clock.advance(50.0)
        second = driver.step(alarm=True)
        self.assertEqual(second.state, STATE_UNCONSUMED)
        self.assertTrue(channel.escalated)
        self.assertEqual(channel.consecutive_non_actuating, 2)

    def test_inert_carries_the_named_exit_not_a_bare_none(self):
        clock = _Clock()
        sched = _FakeScheduler(queued=2, running=0, age=200.0)
        setattr(
            sched,
            "phase_flip_corridor_admission",
            _gate(_FakeGuard(free_mib=4096, floor_mib=1331)),
        )
        driver = self._driver(sched, clock, grace_s=45.0, retry_s=30.0)

        driver.step(alarm=True)
        Scheduler.process_input_requests(sched, [])
        clock.advance(1.0)
        outcome = driver.step(alarm=True)

        self.assertEqual(outcome.state, STATE_INERT)
        self.assertEqual(outcome.reason, REASON_HEADROOM_SUFFICIENT)
        self.assertFalse(outcome.actuated)

        # The result must be a READABLE FACT, not only a log line. #788 had
        # no such field, which is why it could report its own ineffectiveness
        # three times without anything acting on it.
        # MUTANT KILLED: stop storing the settled outcome on the channel.
        channel = get_recovery_channel(sched)
        self.assertIsNotNone(channel.last_outcome)
        self.assertEqual(channel.last_outcome.state, STATE_INERT)
        self.assertEqual(channel.last_outcome.reason, REASON_HEADROOM_SUFFICIENT)

    def test_an_actuating_recovery_clears_the_run(self):
        clock = _Clock()
        sched = _FakeScheduler(queued=2, running=0, age=200.0)
        setattr(
            sched,
            "phase_flip_corridor_admission",
            _gate(_FakeGuard(free_mib=100, floor_mib=1331)),
        )
        driver = self._driver(sched, clock, grace_s=45.0, retry_s=30.0)
        channel = get_recovery_channel(sched)

        # One unconsumed attempt first, so there is a run to clear.
        driver.step(alarm=True)
        clock.advance(50.0)
        self.assertEqual(driver.step(alarm=True).state, STATE_UNCONSUMED)
        self.assertEqual(channel.consecutive_non_actuating, 1)

        clock.advance(30.0)
        driver.step(alarm=True)
        Scheduler.process_input_requests(sched, [])
        clock.advance(1.0)
        outcome = driver.step(alarm=True)

        self.assertEqual(outcome.state, STATE_ACTUATED)
        self.assertEqual(outcome.reason, REASON_CLEARED)
        self.assertEqual(channel.consecutive_non_actuating, 0)

    def test_a_cleared_wedge_resets_the_episode(self):
        clock = _Clock()
        sched = _FakeScheduler(queued=2, running=0, age=200.0)
        driver = self._driver(sched, clock, grace_s=45.0, retry_s=30.0)
        channel = get_recovery_channel(sched)

        driver.step(alarm=True)
        clock.advance(50.0)
        driver.step(alarm=True)
        self.assertEqual(channel.consecutive_non_actuating, 1)

        driver.step(alarm=False)
        self.assertEqual(channel.consecutive_non_actuating, 0)
        self.assertFalse(channel.escalated)

    def test_retry_interval_bounds_the_attempt_rate(self):
        """Attempts are spaced by ``retry_s``, measured from the last POST.

        Grace (45 s) is deliberately shorter than retry (60 s) here so the
        settle happens strictly inside the retry window and the bound is what
        is being tested, not the settle.
        """
        clock = _Clock()
        sched = _FakeScheduler(queued=2, running=0, age=200.0)
        driver = self._driver(sched, clock, grace_s=45.0, retry_s=60.0)

        driver.step(alarm=True)  # posts #1 at t0
        channel = get_recovery_channel(sched)
        clock.advance(50.0)
        self.assertEqual(driver.step(alarm=True).state, STATE_UNCONSUMED)
        clock.advance(1.0)  # t0+51, inside the 60 s retry window
        driver.step(alarm=True)
        self.assertEqual(channel.requested_seq, 1, "must not retry inside the window")
        clock.advance(10.0)  # t0+61
        driver.step(alarm=True)
        self.assertEqual(channel.requested_seq, 2)


class TestChannelUnit(CustomTestCase):
    def test_ack_seq_is_written_last(self):
        """The watchdog gates on ``acked_seq``, so it must be the last write."""
        channel = WedgeRecoveryChannel()
        sched = _FakeScheduler()
        setattr(
            sched,
            "phase_flip_corridor_admission",
            _gate(_FakeGuard(free_mib=4096, floor_mib=1331)),
        )
        seq = channel.post(0.0, tokens=0)
        channel.drain(sched)
        self.assertEqual(channel.acked_seq, seq)
        self.assertEqual(channel.acked_reason, REASON_HEADROOM_SUFFICIENT)

    def test_drain_is_idempotent_per_sequence(self):
        channel = WedgeRecoveryChannel()
        sched = _FakeScheduler()
        calls = []

        class _CountingGate:
            last_reason = REASON_HEADROOM_SUFFICIENT

            def before_admission(self, tokens):
                calls.append(tokens)
                return None

        setattr(sched, "phase_flip_corridor_admission", _CountingGate())
        channel.post(0.0, tokens=0)
        channel.drain(sched)
        channel.drain(sched)
        channel.drain(sched)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
