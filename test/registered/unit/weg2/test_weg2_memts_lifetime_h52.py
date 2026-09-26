# SPDX-License-Identifier: Apache-2.0
"""H52: the host memory time series must not outlive its boot.

MEASURED 24.09. 14:00Z: 107 mem_timeseries.sh loggers (memts_weg2_fnFL2x33 ..
x153b) still running, 3397 process starts/s instead of 258, %sys 15.7 instead
of 4.0. Root: the logger was stopped only by teardown(), which the normal arm
end never runs (the arm TERMs front/launcher and kill -9s the schedulers; the
logger sits in its own session). `$!` was the logger's pid and the state json
carried it (x157: 3358575 in the log and in helper_pids) -- the signal was
never sent.

Pinned here:
  * start_memts() persists the state WITH the pid (mutant: persist before the
    append -> TestStartMemtsPersistsThePid red);
  * teardown() stops the logger by process group and prints the verdict line;
  * a pre-spawn refusal stops it too (teardown() is not run there);
  * the anchored pgrep pattern hits this tag only, the owner pattern never
    matches the supervisor's own argv;
  * real processes: the supervisor ends the logger when its owner is gone, and
    stop_memts() leaves another tag's logger alive.
"""

import contextlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.registry import nvml as nvml_registry
from sglang.srt.weg2 import launcher as L
from sglang.test.test_utils import CustomTestCase


def _state(tag="h52t", pids=None) -> "L.BootState":
    st = L.BootState(tag=tag, tip="deadbeef", tree="/nonexistent-h52", stamp="2026-09-24T00:00:00Z")
    st.pids = pids or {}
    st.helper_pids = []
    st.ring_dir = ""
    st.admin_key_file = ""
    return st


class _TmpArb(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="weg2-h52-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        p = mock.patch.object(L, "GPU_ARB", self.tmp)
        p.start()
        self.addCleanup(p.stop)
        # #135: start_memts() also writes the helper registry under EVIDENCE_DIR
        e = mock.patch.object(L, "EVIDENCE_DIR", self.tmp)
        e.start()
        self.addCleanup(e.stop)


class TestPatterns(_TmpArb):
    def _logger_argv(self, tag):
        return f"bash {L.MEMTS} {L.memts_csv_path(tag)} 5"

    def _supervisor_argv(self, tag):
        return (f"bash -c {L._MEMTS_SUPERVISOR_SH} memts-guard {L.MEMTS} {L.memts_csv_path(tag)} 5 "
                f"{L.memts_owner_pattern(tag)} 5")

    def test_pgrep_pattern_hits_own_logger_and_supervisor(self):
        pat = L.memts_pgrep_pattern("fnFL2x157")
        self.assertRegex(self._logger_argv("fnFL2x157"), pat)
        self.assertRegex(self._supervisor_argv("fnFL2x157"), pat)

    def test_pgrep_pattern_misses_foreign_tags(self):
        pat = L.memts_pgrep_pattern("fnFL2x15")
        for other in ("fnFL2x157", "fnFL2x15b", "weg2xsn422", "xfnFL2x15"):
            self.assertIsNone(re.search(pat, self._logger_argv(other)), other)
            self.assertIsNone(re.search(pat, self._supervisor_argv(other)), other)

    def test_pgrep_pattern_misses_readers_of_the_csv(self):
        pat = L.memts_pgrep_pattern("fnFL2x157")
        csv = L.memts_csv_path("fnFL2x157")
        for argv in (f"tail -f {csv}", f"python3 plot.py {csv}",
                     f"pkill -f ^bash {L.MEMTS} {re.escape(csv)}"):
            self.assertIsNone(re.search(pat, argv), argv)

    def test_owner_pattern(self):
        own = L.memts_owner_pattern("fnFL2x157")
        py = "/spinning/htsglang-gpu/.venv/bin/python -m"
        self.assertRegex(f"{py} sglang.srt.weg2.launcher --tree /t --tag fnFL2x157 --profile nextflash", own)
        self.assertRegex(f"{py} sglang.srt.weg2.front --prefill http://x --tag fnFL2x157", own)
        self.assertIsNone(re.search(own, f"{py} sglang.srt.weg2.front --tag fnFL2x1570 --x 1"))
        self.assertIsNone(re.search(own, f"{py} sglang.srt.weg2.front --tag weg2xsn422"))
        # The supervisor carries the pattern in its own argv; it must not keep
        # itself alive.
        self.assertIsNone(re.search(own, self._supervisor_argv("fnFL2x157")))


class TestStartMemtsPersistsThePid(_TmpArb):
    def test_state_on_disk_carries_the_pid_right_after_start(self):
        st = _state("h52persist")
        launcher_argv = "python -m sglang.srt.weg2.launcher --tree /t --tag h52persist"
        with (
            mock.patch.object(L.subprocess, "Popen", return_value=mock.Mock(pid=424242)) as popen,
            mock.patch.object(L, "_proc_cmdline", return_value=launcher_argv),
        ):
            pid = L.start_memts(st, lambda s: None)
        self.assertEqual(pid, 424242)
        on_disk = json.load(open(L.state_path(st)))
        self.assertEqual(on_disk["memts_pid"], 424242)
        self.assertIn(424242, on_disk["helper_pids"])
        kw = popen.call_args.kwargs
        self.assertTrue(kw.get("start_new_session"), "logger must lead its own process group")
        argv = popen.call_args.args[0]
        self.assertIn(L.memts_owner_pattern("h52persist"), argv, "owner watch armed")

    def test_owner_watch_off_when_the_launcher_argv_does_not_match(self):
        st = _state("h52off")
        with (
            mock.patch.object(L.subprocess, "Popen", return_value=mock.Mock(pid=424243)) as popen,
            mock.patch.object(L, "_proc_cmdline", return_value="python -m pytest"),
        ):
            L.start_memts(st, lambda s: None)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[7], "", "an owner that can never match would kill the logger at once")


class TestTeardownStopsTheLogger(_TmpArb):
    def _teardown(self, st, cmdline, pgrep_out=""):
        path = L.state_path(st)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(st.__dict__, f, default=str)
        buf = io.StringIO()
        with (
            mock.patch("os.killpg") as killpg,
            mock.patch("os.kill", side_effect=ProcessLookupError("gone")) as kill,
            mock.patch("time.sleep"),
            mock.patch("subprocess.run", return_value=mock.Mock(stdout=pgrep_out)),
            mock.patch.object(L, "_proc_cmdline", return_value=cmdline),
            mock.patch.object(nvml_registry, "memory_snapshot", return_value=[]),
            contextlib.redirect_stdout(buf),
        ):
            L.teardown(path)
        return buf.getvalue(), killpg, kill

    def test_teardown_kills_the_logger_group_and_says_so(self):
        st = _state("h52td")
        st.memts_pid = 777001
        st.helper_pids = [777001, 777002]
        out, killpg, kill = self._teardown(st, f"bash -c ... {L.memts_csv_path('h52td')} 5")
        killpg.assert_called_once_with(777001, signal.SIGTERM)
        self.assertIn("WEG2-TEARDOWN memts pid=777001 killed", out)
        termed = [c.args[0] for c in kill.call_args_list]
        self.assertIn(777002, termed, "the other helpers still get their TERM")
        self.assertNotIn(777001, termed, "the logger is stopped by group, not twice by pid")

    def test_recycled_pid_is_never_signalled(self):
        st = _state("h52rec")
        st.memts_pid = 777003
        st.helper_pids = [777003]
        out, killpg, _ = self._teardown(st, "/usr/sbin/sshd -D")
        killpg.assert_not_called()
        self.assertIn("WEG2-TEARDOWN memts pid=777003 already-gone", out)

    def test_legacy_state_without_memts_pid_is_caught_by_the_net(self):
        st = _state("h52leg")
        st.helper_pids = [777004]
        del st.__dict__["memts_pid"]  # a state json written by the old launcher
        out, killpg, kill = self._teardown(st, "", pgrep_out="777004\n")
        killpg.assert_not_called()
        self.assertIn(777004, [c.args[0] for c in kill.call_args_list])
        self.assertIn("WEG2-TEARDOWN memts pid=0", out)


class TestPreSpawnRefusalStopsTheLogger(CustomTestCase):
    def setUp(self):
        orig = L._ACTIVE_BOOT_STATE
        self.addCleanup(setattr, L, "_ACTIVE_BOOT_STATE", orig)
        L.set_admin_key(None)
        self.addCleanup(lambda: L.set_admin_key(None))

    def test_refusal_before_group_p_stops_memts_without_teardown(self):
        def fake_main(argv=None):
            st = _state("h52pre", pids={"P": 0, "D": 0})
            st.memts_pid = 777005
            L._ACTIVE_BOOT_STATE = st
            raise L.Weg2LaunchRefused("W1 fake pre-spawn refusal")

        with (
            mock.patch.object(L, "teardown") as td,
            mock.patch.object(L, "stop_memts", return_value="WEG2-TEARDOWN memts pid=777005 killed") as sm,
            mock.patch.object(L, "main", side_effect=fake_main),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc = L.cli(["--fake"])
        self.assertEqual(rc, 2)
        td.assert_not_called()
        sm.assert_called_once_with("h52pre", 777005)


def _pgrep(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]


class TestRealProcesses(_TmpArb):
    """Real bash/pgrep, a fake logger script, no GPU, a few seconds."""

    def setUp(self):
        super().setUp()
        fake = os.path.join(self.tmp, "fake_memts.sh")
        with open(fake, "w") as f:
            f.write('#!/usr/bin/env bash\nwhile :; do echo x >> "$1"; sleep 1; done\n')
        os.chmod(fake, 0o755)
        for p in (mock.patch.object(L, "MEMTS", fake), mock.patch.object(L, "MEMTS_OWNER_POLL_S", 1)):
            p.start()
            self.addCleanup(p.stop)
        self.tag = f"h52r{os.getpid()}"
        self.owner = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)",
             "sglang.srt.weg2.front", "--tag", self.tag])
        self.addCleanup(self._reap, self.owner)
        real = L._proc_cmdline
        launcher_argv = f"python -m sglang.srt.weg2.launcher --tree /t --tag {self.tag}"
        p = mock.patch.object(L, "_proc_cmdline",
                              side_effect=lambda pid: launcher_argv if pid == os.getpid() else real(pid))
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self._kill_pattern, L.memts_pgrep_pattern(self.tag))

    @staticmethod
    def _reap(p):
        if p.poll() is None:
            p.kill()
        p.wait()

    @staticmethod
    def _kill_pattern(pat):
        for q in _pgrep(pat):
            with contextlib.suppress(OSError):
                os.kill(q, signal.SIGKILL)

    def _wait_gone(self, pat, deadline_s):
        t0 = time.time()
        while time.time() - t0 < deadline_s:
            if not _pgrep(pat):
                return True
            time.sleep(0.2)
        return False

    def _start(self):
        st = _state(self.tag)
        pid = L.start_memts(st, lambda s: None)
        pat = L.memts_pgrep_pattern(self.tag)
        t0 = time.time()
        while len(_pgrep(pat)) < 2 and time.time() - t0 < 5:
            time.sleep(0.1)
        found = _pgrep(pat)
        self.assertIn(pid, found, "pgrep net must see the supervisor")
        self.assertGreaterEqual(len(found), 2, "supervisor + logger")
        return pid, pat

    def test_logger_ends_when_its_owner_is_gone(self):
        _pid, pat = self._start()
        time.sleep(1.5)
        self.assertTrue(_pgrep(pat), "owner alive -> logger alive")
        self._reap(self.owner)
        self.assertTrue(self._wait_gone(pat, 6), "owner gone -> supervisor TERMs its group")

    def test_stop_memts_kills_own_tag_only(self):
        pid, pat = self._start()
        foreign_tag = self.tag + "x"
        foreign_csv = L.memts_csv_path(foreign_tag)
        foreign = subprocess.Popen(["bash", L.MEMTS, foreign_csv, "5"], start_new_session=True)
        self.addCleanup(self._reap, foreign)
        line = L.stop_memts(self.tag, pid)
        self.assertIn(f"WEG2-TEARDOWN memts pid={pid} killed", line)
        self.assertTrue(self._wait_gone(pat, 3))
        self.assertIsNone(foreign.poll(), "another tag's logger must survive")
        self.assertIn("already-gone", L.stop_memts(self.tag, pid))


if __name__ == "__main__":
    unittest.main()
