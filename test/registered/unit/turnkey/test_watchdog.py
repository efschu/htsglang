# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for the #604 watchdog state machine.

Hermetic: no server, no socket, no clock, no GPU. Time is an argument.

The FALSIFIER these tests exist for is
``test_http_200_alone_never_reaches_healthy``: the incident being fixed was a
server that answered ``/health`` with 200 while every generation hung
(#622/#649). Any watchdog that treats an HTTP 200 as liveness reports that
server as healthy forever. Make ``step`` return HEALTHY on ``api_ok`` without
a generation verdict and that test goes red.

The second falsifier is ``test_watchdog_never_emits_a_spawn_action``: #638
proved that a watchdog which STARTS serving leaks its cgroup into it. The
action vocabulary has no spawn verb, and that test pins it.
"""

import unittest

from sglang.srt.turnkey import watchdog as W
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _pol(**kw):
    base = dict(poll_s=10, generation_probe_s=100, wedge_confirmations=3,
                backoff_s=(30, 60, 120), max_restarts=3,
                restart_window_s=1000, boot_grace_s=500)
    base.update(kw)
    return W.Policy(**base)


def _healthy(state, policy, t):
    """Drive the machine to HEALTHY via a successful generation probe."""
    d = W.step(state, W.Observation(True, True, True), t, policy)
    assert d.state.phase == W.HEALTHY, d.reason
    return d.state


class TestBootGrace(CustomTestCase):
    def test_initial_state_is_booting_not_healthy(self):
        p = _pol()
        s = W.initial(0.0, p)
        self.assertEqual(s.phase, W.BOOTING)
        self.assertEqual(s.boot_deadline, p.boot_grace_s)

    def test_unreachable_inside_grace_does_nothing(self):
        p = _pol()
        s = W.initial(0.0, p)
        d = W.step(s, W.Observation(False, False), 100.0, p)
        self.assertEqual(d.action, W.ACT_NONE)
        self.assertEqual(d.state.phase, W.BOOTING)
        self.assertIn("grace", d.reason)

    def test_port_open_but_api_silent_is_still_booting(self):
        # The realistic cold-boot middle: bound, not yet serving.
        p = _pol()
        d = W.step(W.initial(0.0, p), W.Observation(True, False), 100.0, p)
        self.assertEqual(d.action, W.ACT_NONE)
        self.assertIn("port is open", d.reason)

    def test_grace_expiry_restarts(self):
        p = _pol()
        d = W.step(W.initial(0.0, p), W.Observation(False, False), 600.0, p)
        self.assertEqual(d.action, W.ACT_RESTART)
        self.assertIn("boot grace", d.reason)

    def test_long_jit_boot_is_not_killed(self):
        # #172/#615: a legitimate cold cache boot takes minutes. With a
        # 1800 s grace, a 20 min boot must survive untouched.
        p = _pol(boot_grace_s=1800)
        s = W.initial(0.0, p)
        for t in range(0, 1200, 60):
            d = W.step(s, W.Observation(True, False), float(t), p)
            self.assertEqual(d.action, W.ACT_NONE, f"killed at t={t}")
            s = d.state


class TestWedgeDetection(CustomTestCase):
    def test_http_200_alone_never_reaches_healthy_WHEN_PROBING(self):
        """THE falsifier, now scoped to the probing configuration.

        CONTRACT CHANGED 2026-08-15 (standing order: retire the generation
        prober). This still holds WHEN the probe is enabled, which is why the
        policy here opts in. Under the shipped retired default the opposite is
        true and is asserted in TestProberRetired below -- deliberately, with
        the blind spot named.
        """
        p = _pol(generation_probe_enabled=True)
        s = W.initial(0.0, p)
        # Reachable, no generation verdict, forever.
        for t in (10.0, 200.0, 5000.0, 100000.0):
            d = W.step(s, W.Observation(True, True, None), t, p)
            self.assertNotEqual(
                d.state.phase, W.HEALTHY,
                f"HTTP 200 alone was accepted as healthy at t={t}")
            s = d.state

    def test_api_back_after_boot_demands_a_generation_probe(self):
        p = _pol(generation_probe_enabled=True)
        d = W.step(W.initial(0.0, p), W.Observation(True, True, None), 10.0, p)
        self.assertEqual(d.action, W.ACT_PROBE_GENERATION)
        self.assertEqual(d.state.phase, W.SUSPECT)

    def test_one_failed_probe_is_not_a_wedge(self):
        p = _pol(generation_probe_enabled=True)
        s = _healthy(W.initial(0.0, p), p, 10.0)
        d = W.step(s, W.Observation(True, True, False), 200.0, p)
        self.assertEqual(d.action, W.ACT_NONE)
        self.assertEqual(d.state.phase, W.SUSPECT)
        self.assertEqual(d.state.gen_failures, 1)

    def test_confirmations_convict_and_restart(self):
        p = _pol(wedge_confirmations=3)
        s = _healthy(W.initial(0.0, p), p, 10.0)
        t = 200.0
        for i in (1, 2):
            d = W.step(s, W.Observation(True, True, False), t, p)
            self.assertEqual(d.action, W.ACT_NONE, f"convicted too early at {i}")
            s, t = d.state, t + 100
        d = W.step(s, W.Observation(True, True, False), t, p)
        self.assertEqual(d.action, W.ACT_RESTART)
        self.assertIn("WEDGED", d.reason)
        self.assertIn("200", d.reason)

    def test_success_resets_the_suspicion_counter(self):
        p = _pol(generation_probe_enabled=True)
        s = _healthy(W.initial(0.0, p), p, 10.0)
        s = W.step(s, W.Observation(True, True, False), 200.0, p).state
        s = W.step(s, W.Observation(True, True, False), 300.0, p).state
        self.assertEqual(s.gen_failures, 2)
        s = W.step(s, W.Observation(True, True, True), 400.0, p).state
        self.assertEqual(s.phase, W.HEALTHY)
        self.assertEqual(s.gen_failures, 0)

    def test_probe_is_due_on_cadence_not_every_tick(self):
        p = _pol(generation_probe_s=100, generation_probe_enabled=True)
        s = _healthy(W.initial(0.0, p), p, 10.0)
        self.assertEqual(W.step(s, W.Observation(True, True), 50.0, p).action,
                         W.ACT_NONE)
        self.assertEqual(W.step(s, W.Observation(True, True), 111.0, p).action,
                         W.ACT_PROBE_GENERATION)



class TestProberRetired(CustomTestCase):
    """The shipped default: passive evidence is the verdict.

    WHAT THIS GIVES UP, stated rather than discovered later: with generation
    retired there is no periodic check that can see a #622 wedge (HTTP 200,
    no tokens), because seeing it requires generating. That is covered by the
    one-shot real generation at teardown/restore and by boot-log age, not by
    the watchdog.
    """

    def test_default_is_retired(self):
        self.assertFalse(W.Policy(
            poll_s=10, generation_probe_s=100, wedge_confirmations=3,
            backoff_s=(30, 60, 120), max_restarts=3,
            restart_window_s=1000, boot_grace_s=500).generation_probe_enabled)

    def test_passive_evidence_alone_is_healthy(self):
        p = _pol()
        s = W.initial(0.0, p)
        for t in (10.0, 200.0, 5000.0, 100000.0):
            d = W.step(s, W.Observation(True, True, None), t, p)
            self.assertEqual(d.state.phase, W.HEALTHY)
            self.assertEqual(d.action, W.ACT_NONE)
            s = d.state

    def test_no_tick_ever_schedules_a_generation(self):
        p = _pol(generation_probe_s=100)
        s = _healthy(W.initial(0.0, p), p, 10.0)
        for t in (50.0, 111.0, 10_000.0):
            self.assertNotEqual(
                W.step(s, W.Observation(True, True), t, p).action,
                W.ACT_PROBE_GENERATION,
                f"generation scheduled at t={t} despite retirement")

    def test_crash_detection_survives_retirement(self):
        """The half that must NOT be given up."""
        p = _pol()
        s = _healthy(W.initial(0.0, p), p, 10.0)
        d = W.step(s, W.Observation(False, False), 60.0, p)
        self.assertNotEqual(d.state.phase, W.HEALTHY)


class TestCrashPath(CustomTestCase):
    def test_healthy_to_unreachable_restarts(self):
        p = _pol()
        s = _healthy(W.initial(0.0, p), p, 10.0)
        d = W.step(s, W.Observation(False, False), 200.0, p)
        self.assertEqual(d.action, W.ACT_RESTART)
        self.assertIn("API stopped answering", d.reason)

    def test_alive_process_shut_api_is_named_distinctly(self):
        p = _pol()
        s = _healthy(W.initial(0.0, p), p, 10.0)
        d = W.step(s, W.Observation(True, False), 200.0, p)
        self.assertIn("port stays open", d.reason)


class TestBackoffAndGiveUp(CustomTestCase):
    def test_backoff_ladder_delays_successive_restarts(self):
        # boot_grace deliberately SHORTER than the backoff here; see
        # test_grace_dominates_a_shorter_backoff for why that matters.
        p = _pol(backoff_s=(30, 60, 120), max_restarts=9, boot_grace_s=10)
        s = W.initial(0.0, p)
        d = W.step(s, W.Observation(False, False), 600.0, p)
        self.assertEqual(d.action, W.ACT_RESTART)
        # ladder index 1 -> 60 s
        self.assertEqual(d.state.next_restart_at, 660.0)
        # Past the (short) grace but inside the backoff: hold, do not restart.
        d2 = W.step(d.state, W.Observation(False, False), 640.0, p)
        self.assertEqual(d2.action, W.ACT_NONE)
        self.assertIn("backoff", d2.reason)
        # Past both: restart is allowed again.
        d3 = W.step(d.state, W.Observation(False, False), 700.0, p)
        self.assertEqual(d3.action, W.ACT_RESTART)

    def test_grace_dominates_a_shorter_backoff(self):
        """The effective hold between restarts is max(boot_grace, backoff).

        Pinned because it is a real and slightly surprising interaction: a
        generous boot grace (needed for JIT cold caches, #172/#615) silently
        makes a shorter backoff ladder inert. That is the CORRECT behaviour --
        a lane still inside its boot window has not failed yet, so there is
        nothing to back off from -- but an operator tuning backoff_s down and
        seeing no change deserves this documented rather than mysterious.
        """
        p = _pol(backoff_s=(30, 60), max_restarts=9, boot_grace_s=500)
        d = W.step(W.initial(0.0, p), W.Observation(False, False), 600.0, p)
        self.assertEqual(d.action, W.ACT_RESTART)
        self.assertEqual(d.state.next_restart_at, 660.0)   # backoff says 660
        self.assertEqual(d.state.boot_deadline, 1100.0)    # grace says 1100
        # At t=800 the backoff has expired but the grace has not: still BOOTING.
        d2 = W.step(d.state, W.Observation(False, False), 800.0, p)
        self.assertEqual(d2.action, W.ACT_NONE)
        self.assertEqual(d2.state.phase, W.BOOTING)
        self.assertIn("grace", d2.reason)

    def test_backoff_last_value_repeats(self):
        p = _pol(backoff_s=(30, 60))
        self.assertEqual(p.backoff_for(0), 30)
        self.assertEqual(p.backoff_for(1), 60)
        self.assertEqual(p.backoff_for(7), 60)

    def test_gives_up_after_max_restarts(self):
        p = _pol(backoff_s=(0,), max_restarts=3, boot_grace_s=10,
                 restart_window_s=100000)
        s = W.initial(0.0, p)
        t = 100.0
        restarts = 0
        for _ in range(40):
            d = W.step(s, W.Observation(False, False), t, p)
            if d.action == W.ACT_RESTART:
                restarts += 1
            s, t = d.state, t + 100
            if s.phase == W.GIVEN_UP:
                break
        self.assertEqual(s.phase, W.GIVEN_UP)
        self.assertEqual(restarts, 3)

    def test_given_up_is_loud_not_silent(self):
        p = _pol()
        s = W.WatchdogState(phase=W.GIVEN_UP)
        d = W.step(s, W.Observation(False, False), 10.0, p)
        self.assertEqual(d.action, W.ACT_ALARM)
        self.assertTrue(d.alarming)

    def test_given_up_never_restarts_again(self):
        p = _pol()
        s = W.WatchdogState(phase=W.GIVEN_UP)
        for t in range(0, 100000, 5000):
            d = W.step(s, W.Observation(False, False), float(t), p)
            self.assertNotEqual(d.action, W.ACT_RESTART)
            s = d.state

    def test_given_up_recovers_only_on_a_real_generation(self):
        p = _pol()
        s = W.WatchdogState(phase=W.GIVEN_UP)
        # HTTP 200 alone does not lift it.
        d = W.step(s, W.Observation(True, True, None), 10.0, p)
        self.assertEqual(d.state.phase, W.GIVEN_UP)
        # A generated token does.
        d = W.step(s, W.Observation(True, True, True), 20.0, p)
        self.assertEqual(d.state.phase, W.HEALTHY)

    def test_window_expiry_forgives_old_restarts(self):
        p = _pol(max_restarts=2, restart_window_s=1000, backoff_s=(0,),
                 boot_grace_s=1)
        s = W.WatchdogState(phase=W.HEALTHY, restarts=(0.0, 1.0))
        # Inside the window: budget exhausted.
        d = W.step(s, W.Observation(False, False), 500.0, p)
        self.assertEqual(d.state.phase, W.GIVEN_UP)
        # Outside it: the old restarts no longer count.
        d = W.step(s, W.Observation(False, False), 5000.0, p)
        self.assertEqual(d.action, W.ACT_RESTART)


class TestStructuralGuarantees(CustomTestCase):
    def test_watchdog_never_emits_a_spawn_action(self):
        """#638: the watchdog must not be able to start serving itself.

        A serving process spawned by the watchdog inherits the watchdog's
        cgroup, so stopping the watchdog kills production. The vocabulary has
        no spawn verb; restarts go through systemd.
        """
        self.assertEqual(
            set(W.ACTIONS),
            {W.ACT_NONE, W.ACT_PROBE_GENERATION, W.ACT_RESTART, W.ACT_ALARM})
        for verb in ("spawn", "exec", "launch", "start", "fork", "popen"):
            self.assertNotIn(verb, set(W.ACTIONS))

    def test_step_is_pure_and_does_not_mutate_input(self):
        p = _pol()
        s = W.initial(0.0, p)
        before = (s.phase, s.gen_failures, s.restarts, s.next_restart_at)
        W.step(s, W.Observation(False, False), 9999.0, p)
        after = (s.phase, s.gen_failures, s.restarts, s.next_restart_at)
        self.assertEqual(before, after)

    def test_every_decision_carries_a_reason(self):
        p = _pol()
        states = [W.initial(0.0, p),
                  W.WatchdogState(phase=W.HEALTHY),
                  W.WatchdogState(phase=W.SUSPECT, gen_failures=2),
                  W.WatchdogState(phase=W.GIVEN_UP)]
        obss = [W.Observation(a, b, g)
                for a in (True, False) for b in (True, False)
                for g in (None, True, False)]
        for s in states:
            for o in obss:
                for t in (0.0, 1000.0, 100000.0):
                    d = W.step(s, o, t, p)
                    self.assertTrue(d.reason.strip(), (s, o, t))
                    self.assertIn(d.action, W.ACTIONS)
                    self.assertIn(d.state.phase, W.PHASES)


if __name__ == "__main__":
    unittest.main()
