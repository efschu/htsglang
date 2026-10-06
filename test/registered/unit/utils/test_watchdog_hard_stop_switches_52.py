"""deskq 52 (user decision 06.10.): the two watchdogs that stop the whole process get an on/off switch each.

  * Scheduler watchdog (WatchdogRaw, soft=False): after the dump it sends SIGQUIT to the parent process.
    Until now the only way to avoid that was a very large --watchdog-timeout.
    SGLANG_ENABLE_SCHEDULER_WATCHDOG_KILL (default 1 = today's kill) = 0: dump and line stay, no signal.
  * SubprocessWatchdog: a scheduler/detokenizer child that exited non-zero makes it SIGQUIT its own process.
    SGLANG_ENABLE_SUBPROCESS_WATCHDOG_KILL (default 1) = 0: the death is reported, nothing is signalled.

Red on the base (d10a4c3d60): the env is unknown there, so the "off" cases still signal. Green with the change.
The env is set through ``mock.patch.dict(os.environ)`` and not ``envs.X.override`` on purpose: the same file must be
runnable against the base, where the descriptor does not exist yet, and fail on the BEHAVIOUR, not on an
AttributeError.
"""

import os
import signal
import time
import types
import unittest
from unittest import mock

from sglang.srt.utils import watchdog as W
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

SCHED = "SGLANG_ENABLE_SCHEDULER_WATCHDOG_KILL"
SUB = "SGLANG_ENABLE_SUBPROCESS_WATCHDOG_KILL"

_REAL_SLEEP = time.sleep


def _env(**kv):
    """Both switches cleared, then ``kv`` applied (an unset env = the default)."""
    clean = {k: v for k, v in os.environ.items() if k not in (SCHED, SUB)}
    clean.update(kv)
    return mock.patch.dict(os.environ, clean, clear=True)


def _hard_watchdog(timeout=0.2):
    wd = W.WatchdogRaw.__new__(W.WatchdogRaw)  # no thread: the check is driven by hand
    wd.debug_name, wd.watchdog_timeout, wd.soft = "Scheduler", timeout, False
    wd.get_counter, wd.is_active = (lambda: 7), (lambda: True)  # active, counter frozen
    wd.dump_info = wd.describe_arm = None
    wd.parent_process = mock.Mock()
    return wd


def _trip(wd):
    """Run the check to its verdict; the 5 s pre-SIGQUIT sleep is shortened."""
    with mock.patch.object(W, "pyspy_dump_schedulers", lambda *a, **k: None), mock.patch.object(
        W.time, "sleep", side_effect=lambda s: _REAL_SLEEP(min(s, 0.02))
    ), mock.patch.object(W, "logger") as log:
        wd._watchdog_once()
    return log


class SchedulerWatchdogKill(unittest.TestCase):
    def test_default_still_sends_sigquit_to_the_parent(self):
        """Without the env the behaviour is today's: a frozen counter ends in SIGQUIT to the parent."""
        wd = _hard_watchdog()
        with _env():
            _trip(wd)
        wd.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)

    def test_switch_on_explicitly_is_the_default(self):
        wd = _hard_watchdog()
        with _env(**{SCHED: "1"}):
            _trip(wd)
        wd.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)

    def test_switch_off_keeps_the_dump_and_the_line_but_sends_nothing(self):
        """BUG-CLASS (the user's decision): the hard stop could not be switched off. Off = the timeout line is
        still written, the process lives on."""
        wd = _hard_watchdog()
        with _env(**{SCHED: "0"}):
            log = _trip(wd)
        wd.parent_process.send_signal.assert_not_called()
        lines = [str(c.args[0]) for c in log.error.call_args_list]
        self.assertTrue(any("watchdog timeout" in m for m in lines), lines)
        self.assertTrue(any("SGLANG_ENABLE_SCHEDULER_WATCHDOG_KILL=0" in m for m in lines), lines)

    def test_soft_watchdog_is_unaffected_by_the_switch(self):
        """soft=True never signalled; the switch must not change that (nor start signalling)."""
        wd = _hard_watchdog()
        wd.soft = True
        with _env(**{SCHED: "1"}):
            _trip(wd)
        wd.parent_process.send_signal.assert_not_called()


def _dead(pid, code):
    return types.SimpleNamespace(pid=pid, exitcode=code, is_alive=lambda: False)


def _sub_watchdog(*procs):
    return W.SubprocessWatchdog(processes=list(procs), process_names=[f"p{i}" for i in range(len(procs))])


class SubprocessWatchdogKill(unittest.TestCase):
    def test_default_still_sigquits_its_own_process_on_a_crash(self):
        sw = _sub_watchdog(_dead(101, 1))
        with _env(), mock.patch("os.kill") as kill, mock.patch.object(W, "logger"):
            stop = sw._check_processes()
        self.assertTrue(stop)
        kill.assert_called_once_with(os.getpid(), signal.SIGQUIT)

    def test_switch_off_reports_the_death_but_signals_nothing_and_keeps_watching(self):
        sw = _sub_watchdog(_dead(102, 1))
        with _env(**{SUB: "0"}), mock.patch("os.kill") as kill, mock.patch.object(W, "logger") as log:
            stop = sw._check_processes()
        kill.assert_not_called()
        self.assertFalse(stop, "the loop must go on watching: a later death must still be reported")
        lines = [str(c.args[0]) for c in log.error.call_args_list]
        self.assertTrue(any("Subprocess p0" in m and "is gone" in m for m in lines), lines)
        self.assertTrue(any("SGLANG_ENABLE_SUBPROCESS_WATCHDOG_KILL=0" in m for m in lines), lines)

    def test_switch_off_names_the_suppression_once_not_every_poll(self):
        sw = _sub_watchdog(_dead(103, 1))
        with _env(**{SUB: "0"}), mock.patch("os.kill"), mock.patch.object(W, "logger") as log:
            for _ in range(5):
                sw._check_processes()
        n = sum("SGLANG_ENABLE_SUBPROCESS_WATCHDOG_KILL=0" in str(c.args[0]) for c in log.error.call_args_list)
        self.assertEqual(n, 1)

    def test_clean_exit_never_signals_in_either_setting(self):
        """Unchanged policy: exit code 0 is reported, not signalled; the switch must not make it so."""
        for val in ("1", "0"):
            sw = _sub_watchdog(_dead(104, 0))
            with _env(**{SUB: val}), mock.patch("os.kill") as kill, mock.patch.object(W, "logger"):
                self.assertFalse(sw._check_processes())
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
