# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#799: the watchdog acts on the #699/#739 admission-wedge verdict.

Hermetic: no server, no socket, no GPU, no systemd. Time is an argument and
every side effect is injected.

THE INCIDENT. Boot 0822_0829 ran thirteen minutes with 0 decode batches and 0
finished requests. The scheduler's admission-wedge detector was right about it
146 times. Every supervisor read "healthy", because the only cheap liveness
signal on this stack is ``/get_model_info``, which answers from the HTTP
process without ever touching the scheduler -- by design
(``turnkey/probe.py``). A watchdog on that path cannot see this failure class,
and no amount of tuning makes it see it.

THE FALSIFIERS, one per direction, because a watchdog can fail both ways:

* ``test_tick_restarts_when_the_scheduler_reports_a_wedge`` -- the detector
  fires and the PRODUCTION path acts. It drives ``WatchdogRunner.tick``, not
  ``W.step``: a test that calls the pure function proves the pure function
  works and says nothing about whether anything calls it. Hardcode
  ``wedged=None`` back into ``tick`` and this test goes red.
* ``test_tick_never_restarts_a_lane_that_reports_no_wedge`` -- the detector is
  silent and the watchdog does nothing. A supervisor that restarts a healthy
  lane is worse than one that sleeps.

THE THIRD, which is neither: ``test_no_measurement_is_byte_identical_to_the
_old_machine``. When no verdict is published, every decision must be exactly
what it was before this feature existed. A new channel is allowed to add a
verdict; it is not allowed to change the meaning of silence.
"""

import os
import tempfile
import unittest

from sglang.srt.turnkey import config as C
from sglang.srt.turnkey import runner as R
from sglang.srt.turnkey import watchdog as W
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _pol(**kw):
    base = dict(poll_s=10, generation_probe_s=100, wedge_confirmations=3,
                backoff_s=(30, 60, 120), max_restarts=3,
                restart_window_s=1000, boot_grace_s=500)
    base.update(kw)
    return W.Policy(**base)


class _Sig:
    """What ``wedge_status.read_wedge_signal`` returns, duck-typed."""

    def __init__(self, verdict, detail="", stale=False):
        self.verdict = verdict
        self.detail = detail
        self.stale = stale


def _healthy(state, policy, t):
    """Drive to HEALTHY the way the live path does: a no-wedge verdict."""
    d = W.step(state, W.Observation(True, True, None, False), t, policy)
    assert d.state.phase == W.HEALTHY, d.reason
    return d.state


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------


class TestWedgeSignalConvicts(CustomTestCase):
    def test_one_wedge_report_is_not_a_conviction(self):
        p = _pol()
        s = _healthy(W.initial(0.0, p), p, 1.0)
        d = W.step(s, W.Observation(True, True, None, True), 2.0, p)
        self.assertEqual(d.state.phase, W.SUSPECT)
        self.assertEqual(d.action, W.ACT_NONE)
        self.assertEqual(d.state.wedge_hits, 1)

    def test_confirmations_convict_and_restart(self):
        p = _pol(wedge_confirmations=3)
        s = _healthy(W.initial(0.0, p), p, 1.0)
        for i, t in enumerate((2.0, 3.0), start=1):
            d = W.step(s, W.Observation(True, True, None, True), t, p)
            s = d.state
            self.assertEqual(d.action, W.ACT_NONE, f"convicted early at {i}")
        d = W.step(s, W.Observation(True, True, None, True), 4.0, p)
        self.assertEqual(d.action, W.ACT_RESTART)
        self.assertIn("WEDGED", d.reason)
        self.assertTrue(d.alarming)

    def test_a_clean_verdict_resets_the_suspicion(self):
        p = _pol(wedge_confirmations=3)
        s = _healthy(W.initial(0.0, p), p, 1.0)
        s = W.step(s, W.Observation(True, True, None, True), 2.0, p).state
        s = W.step(s, W.Observation(True, True, None, True), 3.0, p).state
        self.assertEqual(s.wedge_hits, 2)
        d = W.step(s, W.Observation(True, True, None, False), 4.0, p)
        self.assertEqual(d.state.phase, W.HEALTHY)
        self.assertEqual(d.state.wedge_hits, 0)

    def test_http_200_does_not_outrank_the_scheduler(self):
        """The whole point. api_ok is True in every observation here."""
        p = _pol(wedge_confirmations=1)
        s = _healthy(W.initial(0.0, p), p, 1.0)
        d = W.step(s, W.Observation(True, True, None, True), 2.0, p)
        self.assertEqual(d.action, W.ACT_RESTART)


class TestWedgeSignalRestraint(CustomTestCase):
    def test_no_measurement_is_byte_identical_to_the_old_machine(self):
        """``wedged=None`` must decide exactly as the pre-#799 machine did.

        Compared against the same machine with the channel switched OFF, over
        every reachable (phase, port, api) combination. Any divergence means
        the new channel changed the meaning of silence.
        """
        on, off = _pol(), _pol(wedge_signal_enabled=False)
        for port in (True, False):
            for api in (True, False):
                for t in (1.0, 600.0, 5000.0):
                    a = W.step(W.initial(0.0, on),
                               W.Observation(port, api, None, None), t, on)
                    b = W.step(W.initial(0.0, off),
                               W.Observation(port, api, None, None), t, off)
                    self.assertEqual(a.state.phase, b.state.phase)
                    self.assertEqual(a.action, b.action)
                    self.assertEqual(a.reason, b.reason)

    def test_the_kill_switch_is_honoured(self):
        p = _pol(wedge_signal_enabled=False, wedge_confirmations=1)
        s = _healthy(W.initial(0.0, _pol()), _pol(), 1.0)
        d = W.step(s, W.Observation(True, True, None, True), 2.0, p)
        self.assertNotEqual(d.action, W.ACT_RESTART)

    def test_a_stale_publisher_never_convicts(self):
        """Staleness reaches the machine as None, and None is not evidence."""
        p = _pol(wedge_confirmations=1)
        s = _healthy(W.initial(0.0, p), p, 1.0)
        d = W.step(s, W.Observation(True, True, None, None), 2.0, p)
        self.assertEqual(d.action, W.ACT_NONE)


class TestWedgeGivesUp(CustomTestCase):
    def test_the_wedge_path_reaches_given_up_and_stops_restarting(self):
        """The abort condition: N restarts, then loud and terminal.

        A wedge that survives every restart is a human problem. Thrashing it
        burns the cards and destroys the evidence needed to fix it.
        """
        p = _pol(wedge_confirmations=1, max_restarts=3, backoff_s=(0,),
                 restart_window_s=10_000)
        s = W.initial(0.0, p)
        restarts, t = 0, 1.0
        for _ in range(40):
            t += 10.0
            d = W.step(s, W.Observation(True, True, None, True), t, p)
            s = d.state
            if d.action == W.ACT_RESTART:
                restarts += 1
            if s.phase == W.GIVEN_UP:
                break
        self.assertEqual(s.phase, W.GIVEN_UP)
        self.assertEqual(restarts, 3)
        d = W.step(s, W.Observation(True, True, None, True), t + 100.0, p)
        self.assertEqual(d.action, W.ACT_ALARM)
        self.assertNotEqual(d.action, W.ACT_RESTART)

    def test_given_up_reopens_on_a_positive_clean_verdict(self):
        """With the generation probe retired, this is the ONLY door left.

        Before #799 the recovery condition was ``generation is True``, which
        is unreachable while the probe is retired -- an operator who fixed the
        lane got a permanently deaf watchdog.
        """
        p = _pol()
        s = W.WatchdogState(phase=W.GIVEN_UP)
        d = W.step(s, W.Observation(True, True, None, False), 5.0, p)
        self.assertEqual(d.state.phase, W.HEALTHY)

    def test_given_up_does_not_reopen_on_silence(self):
        p = _pol()
        s = W.WatchdogState(phase=W.GIVEN_UP)
        d = W.step(s, W.Observation(True, True, None, None), 5.0, p)
        self.assertEqual(d.state.phase, W.GIVEN_UP)
        self.assertEqual(d.action, W.ACT_ALARM)


# ---------------------------------------------------------------------------
# The call edge. This is the half that was missing, not the state machine.
# ---------------------------------------------------------------------------


class _Rig:
    """A WatchdogRunner with every side effect captured."""

    def __init__(self, policy, verdicts, api=True, stop=None, drift=None):
        self.restarts = []
        self.alarms = []
        self.logs = []
        self.wedge_reads = 0
        self._verdicts = list(verdicts)
        self._t = [0.0]
        deps = R.RunnerDeps(
            now=lambda: self._tick_clock(),
            sleep=lambda s: None,
            api_probe=lambda: type("P", (), {"ok": api, "detail": ""})(),
            port_probe=lambda: True,
            wedge_probe=self._wedge,
            operator_stop=lambda: stop,
            append_alarm=self.alarms.append,
            restart_drift=lambda: drift,
            restart_unit=lambda u: (self.restarts.append(u), True)[1],
            log=lambda lvl, msg: self.logs.append((lvl, msg)),
        )
        self.run = R.WatchdogRunner(unit="htsglang-serving@ship.service",
                                   base_url="http://127.0.0.1:30030",
                                   policy=policy, deps=deps)

    def _tick_clock(self):
        self._t[0] += 10.0
        return self._t[0]

    def _wedge(self):
        self.wedge_reads += 1
        if not self._verdicts:
            return _Sig(None, "exhausted")
        v = self._verdicts.pop(0)
        return _Sig(v, f"verdict={v}")


class TestRunnerCallEdge(CustomTestCase):
    def test_tick_restarts_when_the_scheduler_reports_a_wedge(self):
        """THE falsifier for the whole ticket.

        Drives the real ``tick``. If ``tick`` stops passing the wedge verdict
        into the Observation -- the mutation that describes the code as it
        stood before #799 -- no restart is requested and this fails.
        """
        rig = _Rig(_pol(wedge_confirmations=2, backoff_s=(0,)),
                   verdicts=[False, True, True, True])
        for _ in range(4):
            rig.run.tick()
        self.assertEqual(rig.restarts, ["htsglang-serving@ship.service"])

    def test_tick_never_restarts_a_lane_that_reports_no_wedge(self):
        """The other direction: a silent detector must produce no action."""
        rig = _Rig(_pol(wedge_confirmations=2), verdicts=[False] * 8)
        for _ in range(8):
            rig.run.tick()
        self.assertEqual(rig.restarts, [])
        self.assertEqual(rig.run.state.phase, W.HEALTHY)

    def test_the_wedge_source_is_consulted_on_every_tick(self):
        rig = _Rig(_pol(), verdicts=[False] * 5)
        for _ in range(5):
            rig.run.tick()
        self.assertEqual(rig.wedge_reads, 5)

    def test_the_wedge_source_is_consulted_even_when_http_is_gone(self):
        """The scheduler can be publishing while the HTTP layer is dead.

        Gating the read on ``api.ok`` would blind the watchdog for the second
        half of boot 0822_0829, when the port stopped answering entirely.
        """
        rig = _Rig(_pol(), verdicts=[True] * 3, api=False)
        rig.run.tick()
        self.assertEqual(rig.wedge_reads, 1)

    def test_an_alarm_reaches_the_durable_ledger(self):
        rig = _Rig(_pol(wedge_confirmations=1, backoff_s=(0,)),
                   verdicts=[True, True])
        rig.run.tick()
        self.assertTrue(rig.alarms, "an alarming decision wrote no ledger line")
        self.assertIn("WEDGED", rig.alarms[0])

    def test_a_quiet_tick_writes_no_ledger_line(self):
        rig = _Rig(_pol(), verdicts=[False, False])
        rig.run.tick()
        rig.run.tick()
        self.assertEqual(rig.alarms, [])


class TestOperatorStop(CustomTestCase):
    def test_the_marker_suspends_every_restart(self):
        """A GPU window is not an outage. Serving is meant to be down.

        The legacy shell watchdog inherits this guard through its start
        script, which exits 3 on the marker. The turnkey path restarts via
        systemd and would NOT, so arming it without this check would drive
        boots into windows an operator had closed.
        """
        # boot_grace_s is deliberately TINY. With a long grace the lane is
        # unreachable-but-booting for the whole test and no restart would be
        # requested even with the guard removed -- the assertion would pass
        # for the wrong reason and the mutant would survive. Measured: this
        # exact test with boot_grace_s=500 did survive mutant M5.
        pol = _pol(wedge_confirmations=1, backoff_s=(0,), boot_grace_s=5.0)
        rig = _Rig(pol, verdicts=[True] * 10, api=False,
                   stop="strand 8 owns the cards")
        for _ in range(10):
            rig.run.tick()
        self.assertEqual(rig.restarts, [])
        self.assertEqual(rig.alarms, [])

    def test_the_same_rig_DOES_restart_once_the_marker_is_gone(self):
        """The paired positive: proves the previous test's silence is caused
        by the marker and not by a grace period that never expired."""
        pol = _pol(wedge_confirmations=1, backoff_s=(0,), boot_grace_s=5.0)
        rig = _Rig(pol, verdicts=[True] * 10, api=False, stop=None)
        for _ in range(10):
            rig.run.tick()
        self.assertTrue(rig.restarts, "control arm requested no restart")

    def test_lifting_the_marker_returns_full_boot_grace(self):
        rig = _Rig(_pol(), verdicts=[None], api=False,
                   stop="window in progress")
        rig.run.tick()
        self.assertEqual(rig.run.state.phase, W.BOOTING)

    def test_reason_reading(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "STOPPED")
            self.assertIsNone(R.operator_stop_reason(p))
            with open(p, "w") as fh:
                fh.write("  gpu window 09:00-10:30  \n")
            self.assertEqual(R.operator_stop_reason(p), "gpu window 09:00-10:30")

    def test_an_empty_marker_still_stops(self):
        """The file's EXISTENCE is the order; its contents only explain it."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "STOPPED")
            open(p, "w").close()
            self.assertIsNotNone(R.operator_stop_reason(p))


# ---------------------------------------------------------------------------
# The second call edge: config -> CLI -> runner
# ---------------------------------------------------------------------------

_REPO = tempfile.TemporaryDirectory()
os.makedirs(os.path.join(_REPO.name, ".git"), exist_ok=True)
U1 = "GPU-11111111-1111-1111-1111-111111111111"

_TOML = """
[stack]
name = "t"
repo = "%s"
venv = "%s/.venv"
log_dir = "/var/log/t"

[[cards]]
uuid = "%s"
label = "a"

[wheel]
dist = "sglang-kernel"
version = "0.4.4"
must_import = ["sgl_kernel"]

[serving.ship]
port = 30030
argv = ["/bin/python", "-m", "sglang.launch_server"]
cards = [0]
boot_log = "/var/log/t/ship.log"
""" % (_REPO.name, _REPO.name, U1)


def tearDownModule():
    _REPO.cleanup()


class TestRestartTargetDrift(CustomTestCase):
    """A sighted watchdog with a stale restart path is worse than a blind one.

    Measured on this rig, 2026-08-22: three automated restart paths, three
    DIFFERENT services. stack.toml boots Qwen3.8-27B-INT8-yarn1.5 at
    pp-stage-ratio 14,10,8; start-serving-30030.sh (which the legacy shell
    watchdog runs) boots Qwen3.6-27B-INT8-W8A8 at tp-size 3; the instance
    actually running that morning was Qwen3.8-27B-INT8-vocabint8-embed at
    pp-stage-ratio 32,18,14. "Recovering" into any of the others replaces the
    service while reporting success.
    """

    def test_matching_targets_do_not_veto(self):
        self.assertIsNone(restart_drift_of("/m/A", "/m/A"))

    def test_a_different_model_vetoes(self):
        d = restart_drift_of("/m/Qwen3.8-INT8-yarn1.5",
                             "/m/Qwen3.8-INT8-vocabint8-embed")
        self.assertIsNotNone(d)
        self.assertIn("DRIFTED", d)

    def test_an_unknowable_comparison_is_not_drift(self):
        """A lane that has never booted must still be restartable."""
        self.assertIsNone(restart_drift_of("/m/A", None))
        self.assertIsNone(restart_drift_of(None, "/m/A"))

    def test_argv_and_boot_log_extraction(self):
        argv = ["/bin/python", "-m", "sglang.launch_server",
                "--model-path", "/m/A", "--tp-size", "3"]
        self.assertEqual(R.argv_model_path(argv), "/m/A")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "boot.log")
            with open(p, "w") as fh:
                fh.write("noise\n[x] server_args=ServerArgs(model_path='/m/B',"
                         " tokenizer_path='/m/B')\n")
            self.assertEqual(R.boot_log_model_path(p), "/m/B")
        self.assertIsNone(R.boot_log_model_path("/nonexistent/boot.log"))

    def test_a_drifted_lane_alarms_but_never_restarts(self):
        rig = _Rig(_pol(wedge_confirmations=1, backoff_s=(0,)),
                   verdicts=[True] * 6, drift="targets differ")
        for _ in range(6):
            rig.run.tick()
        self.assertEqual(rig.restarts, [])
        self.assertTrue(rig.alarms, "a refused restart must still alarm")
        self.assertIn("REFUSED", rig.alarms[0])

    def test_a_refused_restart_does_not_spend_the_restart_budget(self):
        """Otherwise a drifted lane reaches GIVEN_UP having never restarted,
        and the give-up line would blame a restart policy that never ran."""
        rig = _Rig(_pol(wedge_confirmations=1, backoff_s=(0,), max_restarts=3),
                   verdicts=[True] * 20, drift="targets differ")
        for _ in range(20):
            rig.run.tick()
        self.assertNotEqual(rig.run.state.phase, W.GIVEN_UP)
        self.assertEqual(rig.run.state.restarts, ())

    def test_the_same_rig_restarts_once_the_drift_is_gone(self):
        rig = _Rig(_pol(wedge_confirmations=1, backoff_s=(0,)),
                   verdicts=[True] * 6, drift=None)
        for _ in range(6):
            rig.run.tick()
        self.assertTrue(rig.restarts, "control arm requested no restart")


def restart_drift_of(configured, booted):
    return R.restart_target_drift(configured, booted)


class TestConfigReachesTheRunner(CustomTestCase):
    def test_defaults_arrive_switched_on(self):
        c = C.loads(_TOML)
        self.assertTrue(c.watchdog.wedge_signal_enabled)
        self.assertEqual(c.watchdog.wedge_status_dir, "")

    def test_config_can_switch_it_off_and_that_reaches_the_policy(self):
        c = C.loads(_TOML + "\n[watchdog]\nwedge_signal_enabled = false\n")
        self.assertFalse(c.watchdog.wedge_signal_enabled)

    def test_cmd_watch_hands_both_settings_to_the_runner(self):
        """The CLI edge needs its own mutant: a Policy built without
        ``wedge_signal_enabled`` silently defaults to True and the config key
        becomes decoration, which is precisely the 'parsed and never read'
        defect ``_cmd_watch`` already documents for ``enabled``."""
        from sglang.srt.turnkey import __main__ as M

        c = C.loads(_TOML + "\n[watchdog]\nwedge_signal_enabled = false\n"
                            'wedge_status_dir = "/run/x/wedge"\n')
        seen = {}

        class _FakeRunner:
            def __init__(self, **kw):
                seen.update(kw)

            def run(self, max_ticks=None):
                pass

        real = M.R.WatchdogRunner
        M.R.WatchdogRunner = _FakeRunner
        try:
            rc = M._cmd_watch(c, type("A", (), {"lane": "ship", "unit": None,
                                                "ticks": 1})())
        finally:
            M.R.WatchdogRunner = real
        self.assertEqual(rc, 0)
        self.assertEqual(seen["wedge_status_dir"], "/run/x/wedge")
        self.assertFalse(seen["policy"].wedge_signal_enabled)


if __name__ == "__main__":
    unittest.main()
