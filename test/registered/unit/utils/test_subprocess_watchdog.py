# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for SubprocessWatchdog in watchdog.py"""

import multiprocessing as mp
import os
import signal
import threading
import time
import unittest.mock

from sglang.srt.utils.watchdog import SubprocessWatchdog
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")
register_cpu_ci(est_time=9, suite="base-c-test-cpu")


def healthy_worker():
    time.sleep(10)


def crashing_worker():
    os._exit(1)


def slow_crash_worker(delay: float = 0.5):
    time.sleep(delay)
    os._exit(42)


def noop_worker():
    pass


class TestSubprocessWatchdog(CustomTestCase):
    def setUp(self):
        self.sigquit_triggered = threading.Event()
        self._procs = []
        self._monitor = None

        original_kill = os.kill

        def mock_kill(pid, sig):
            if sig == signal.SIGQUIT:
                self.sigquit_triggered.set()
            else:
                original_kill(pid, sig)

        self._patcher = unittest.mock.patch("os.kill", side_effect=mock_kill)
        self._patcher.start()

    def tearDown(self):
        if self._monitor is not None:
            self._monitor.stop()
        self._patcher.stop()
        for p in self._procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

    def _spawn(self, target, args=()):
        proc = mp.Process(target=target, args=args)
        proc.start()
        self._procs.append(proc)
        return proc

    def _watch(self, procs, names=None, interval=0.1):
        if not isinstance(procs, list):
            procs = [procs]
        self._monitor = SubprocessWatchdog(
            processes=procs,
            process_names=names,
            interval=interval,
        )
        self._monitor.start()
        return self._monitor

    def test_healthy_processes_no_sigquit(self):
        proc = self._spawn(healthy_worker)
        self._watch(proc)
        time.sleep(0.5)
        self.assertFalse(self.sigquit_triggered.is_set())

    def test_crashed_process_triggers_sigquit(self):
        proc = self._spawn(slow_crash_worker, args=(0.2,))
        self._watch(proc)
        self.assertTrue(
            self.sigquit_triggered.wait(timeout=5.0),
            "SIGQUIT was not triggered within timeout",
        )

    def test_immediate_crash_detection(self):
        proc = self._spawn(crashing_worker)
        self._watch(proc, interval=0.05)
        self.assertTrue(
            self.sigquit_triggered.wait(timeout=5.0),
            "Immediate crash was not detected",
        )

    def test_multiple_processes_one_crashes(self):
        healthy = self._spawn(healthy_worker)
        crashing = self._spawn(slow_crash_worker, args=(0.2,))
        self._watch([healthy, crashing], names=["healthy", "crashing"])
        self.assertTrue(
            self.sigquit_triggered.wait(timeout=5.0),
            "Crash was not detected when one of multiple processes crashed",
        )

    def test_empty_processes_list(self):
        self._watch([], interval=0.1)
        time.sleep(0.3)
        self.assertFalse(self.sigquit_triggered.is_set())

    def test_normal_exit_no_sigquit(self):
        proc = self._spawn(noop_worker)
        proc.join(timeout=2)
        self._watch(proc)
        time.sleep(0.3)
        self.assertFalse(
            self.sigquit_triggered.is_set(),
            "SIGQUIT should not be triggered for normal exit (exitcode=0)",
        )


class TestTheWatchdogSaysHowTheChildDied(CustomTestCase):
    """#485/C40: a launcher that cannot say HOW its child died is broken.

    A rank of a PP boot was SIGKILLed mid-decode and the incident was written
    up as a "silent death" for a whole shift. The exit code WAS in the log --
    as the bare integer -9, which reads as noise unless you already know that
    a negative exitcode is a signal number. Nothing named the signal, and
    nothing connected it to the only mechanism on this rig that delivers an
    unattributable SIGKILL: the kernel's cgroup OOM killer, whose report goes
    to a kernel ring buffer that a container cannot read.

    These tests are about the RECORD, not the trigger policy. The SIGQUIT
    behaviour above is deliberately unchanged.
    """

    def setUp(self):
        self._procs = []
        self._monitor = None
        original_kill = os.kill

        def mock_kill(pid, sig):
            if sig != signal.SIGQUIT:
                original_kill(pid, sig)

        self._patcher = unittest.mock.patch("os.kill", side_effect=mock_kill)
        self._patcher.start()

    def tearDown(self):
        if self._monitor is not None:
            self._monitor.stop()
        self._patcher.stop()
        for p in self._procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

    def _spawn(self, target, args=()):
        proc = mp.Process(target=target, args=args)
        proc.start()
        self._procs.append(proc)
        return proc

    def _sweep(self, procs, names):
        """One synchronous sweep, so the assertion is on the RECORD it wrote."""
        self._monitor = SubprocessWatchdog(processes=procs, process_names=names)
        with self.assertLogs("sglang.srt.utils.watchdog", level="ERROR") as cm:
            self._monitor._check_processes()
        return "\n".join(cm.output)

    def test_a_signal_death_is_named_not_just_numbered(self):
        # THE C40 REGRESSION. -9 must read as SIGKILL in the log line itself.
        proc = self._spawn(healthy_worker)
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=5)
        self.assertEqual(proc.exitcode, -9)

        text = self._sweep([proc], ["scheduler_0"])
        self.assertIn("SIGKILL", text)
        self.assertIn("scheduler_0", text)
        self.assertIn(str(proc.pid), text)

    def test_a_sigkill_is_attributed_to_the_out_of_memory_killer(self):
        # SIGKILL is never sent by this codebase to a healthy rank, so the
        # report must name the mechanism that does send it, and say where the
        # evidence lives -- a container cannot read the kernel ring buffer.
        proc = self._spawn(healthy_worker)
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=5)

        text = self._sweep([proc], ["scheduler_0"])
        lowered = text.lower()
        self.assertIn("oom", lowered)
        self.assertTrue(
            "cgroup" in lowered or "memory.events" in lowered,
            f"the SIGKILL report must point at the cgroup evidence: {text}",
        )

    def test_an_unexpected_exit_zero_is_still_recorded(self):
        # The old sweep skipped exitcode == 0 outright, so a rank that fell out
        # of its serving loop and returned cleanly left NO record at all. The
        # trigger policy stays as it was; the silence does not.
        proc = self._spawn(noop_worker)
        proc.join(timeout=5)
        self.assertEqual(proc.exitcode, 0)

        text = self._sweep([proc], ["scheduler_0"])
        self.assertIn("scheduler_0", text)
        self.assertIn("exit code 0", text)

    def test_every_dead_process_is_named_in_one_sweep(self):
        # The old sweep returned at the FIRST dead process, so when a rank
        # death took its peers down the record named exactly one of them.
        a = self._spawn(healthy_worker)
        b = self._spawn(healthy_worker)
        os.kill(a.pid, signal.SIGKILL)
        os.kill(b.pid, signal.SIGABRT)
        a.join(timeout=5)
        b.join(timeout=5)

        text = self._sweep([a, b], ["scheduler_0", "scheduler_1"])
        self.assertIn("scheduler_0", text)
        self.assertIn("scheduler_1", text)
        self.assertIn("SIGKILL", text)
        self.assertIn("SIGABRT", text)

    def test_a_live_process_is_never_reported(self):
        proc = self._spawn(healthy_worker)
        self._monitor = SubprocessWatchdog(processes=[proc], process_names=["live"])
        with unittest.mock.patch.object(
            __import__("sglang.srt.utils.watchdog", fromlist=["logger"]), "logger"
        ) as log:
            self.assertFalse(self._monitor._check_processes())
            log.error.assert_not_called()

    def test_a_death_is_reported_once_not_every_poll(self):
        # The sweep runs at 1 Hz forever; a dead process must not produce a
        # line per second for the rest of the boot.
        proc = self._spawn(noop_worker)
        proc.join(timeout=5)
        self._monitor = SubprocessWatchdog(processes=[proc], process_names=["s0"])
        with self.assertLogs("sglang.srt.utils.watchdog", level="ERROR") as cm:
            self._monitor._check_processes()
            first = len(cm.output)
            self._monitor._check_processes()
            self._monitor._check_processes()
            self.assertEqual(
                len(cm.output), first, "the same death was reported more than once"
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
