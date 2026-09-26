# SPDX-License-Identifier: Apache-2.0
"""#135: every launcher helper leaves a process-group registry entry the arm can stop.

MEASURED 26.09. (fnFL2h91v1): the launcher-armed front deadman lived five
minutes past the arm's teardown (teardown 08:00:50Z, verdict 08:05:58Z); the
arm never knew its pid. The fix writes
``{EVIDENCE_DIR}/helpers_{tag}/{name}.pid`` = ``PGID START OWNER OWNER_START
NAME`` -- the format boot_helpers.sh reads -- and teardown() stops deadmen by
process group.

Pinned here:
  * the entry format (five fields, START from /proc stat field 22, OWNER = the
    launcher's parent);
  * start_memts() registers the supervisor (mutant: drop the call -> red);
  * teardown() group-TERMs a deadman that leads its group, falls back to the
    pid for anything else, and removes the entries;
  * a real session-leader helper is stopped with its child by a group TERM.
"""

import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.registry import nvml as nvml_registry
from sglang.srt.weg2 import launcher as L
from sglang.test.test_utils import CustomTestCase


def _state(tag="h135t"):
    st = L.BootState(tag=tag, tip="deadbeef", tree="/nonexistent-h135", stamp="2026-09-26T00:00:00Z")
    st.pids = {}
    st.helper_pids = []
    st.ring_dir = ""
    st.admin_key_file = ""
    return st


class _Tmp(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="weg2-h135-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for attr in ("GPU_ARB", "EVIDENCE_DIR"):
            p = mock.patch.object(L, attr, self.tmp)
            p.start()
            self.addCleanup(p.stop)


class TestEntryFormat(_Tmp):
    def test_entry_is_the_boot_helpers_line(self):
        L.register_helper("h135f", "deadman_front", os.getpid())
        with open(f"{self.tmp}/helpers_h135f/deadman_front.pid") as f:
            fields = f.read().split()
        self.assertEqual(len(fields), 5)
        pgid, start, owner, ostart, name = fields
        self.assertEqual(int(pgid), os.getpid())
        self.assertEqual(int(start), L._proc_start_ticks(os.getpid()))
        self.assertGreater(int(start), 0)
        self.assertEqual(int(owner), os.getppid())
        self.assertEqual(int(ostart), L._proc_start_ticks(os.getppid()))
        self.assertEqual(name, "deadman_front")

    def test_unwritable_registry_never_raises(self):
        with mock.patch.object(L, "EVIDENCE_DIR", "/proc/nonexistent-h135"):
            msg = L.register_helper("h135u", "memts", 1)
        self.assertIn("NOT registered", msg)

    def test_start_ticks_survive_parentheses_in_comm(self):
        self.assertEqual(L._proc_start_ticks(987654321), 0)
        self.assertGreater(L._proc_start_ticks(os.getpid()), 0)


class TestStartMemtsRegisters(_Tmp):
    def test_supervisor_entry_written(self):
        st = _state("h135m")
        with (
            mock.patch.object(L.subprocess, "Popen", return_value=mock.Mock(pid=os.getpid())),
            mock.patch.object(L, "_proc_cmdline", return_value=""),
        ):
            L.start_memts(st, lambda s: None)
        self.assertTrue(os.path.exists(f"{self.tmp}/helpers_h135m/memts.pid"))


class TestTeardownStopsHelpersByGroup(_Tmp):
    def _teardown(self, st, cmdline, getpgid):
        path = L.state_path(st)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(st.__dict__, f, default=str)
        L.register_helper(st.tag, "deadman_front", 1)
        buf = io.StringIO()
        with (
            mock.patch("os.killpg") as killpg,
            mock.patch("os.kill") as kill,
            mock.patch("os.getpgid", side_effect=getpgid),
            mock.patch("time.sleep"),
            mock.patch("subprocess.run", return_value=mock.Mock(stdout="")),
            mock.patch.object(L, "_proc_cmdline", return_value=cmdline),
            mock.patch.object(L, "_alive", return_value=False),
            mock.patch.object(nvml_registry, "memory_snapshot", return_value=[]),
            contextlib.redirect_stdout(buf),
        ):
            L.teardown(path)
        return buf.getvalue(), killpg, kill

    def test_deadman_leader_is_group_termed_and_unregistered(self):
        st = _state("h135g")
        st.helper_pids = [880001]
        out, killpg, kill = self._teardown(
            st, "bash /spinning/gpu-arb/devtools/boot_deadman.sh x.front.log 30030", lambda p: p)
        killpg.assert_any_call(880001, signal.SIGTERM)
        self.assertIn("880001:group", out)
        self.assertFalse(os.path.exists(f"{self.tmp}/helpers_h135g/deadman_front.pid"))

    def test_foreign_argv_falls_back_to_pid(self):
        st = _state("h135p")
        st.helper_pids = [880002]
        out, killpg, kill = self._teardown(st, "/usr/sbin/sshd -D", lambda p: p)
        self.assertNotIn(mock.call(880002, signal.SIGTERM), killpg.call_args_list)
        self.assertIn(mock.call(880002, signal.SIGTERM), kill.call_args_list)
        self.assertIn("880002:pid", out)


class TestRealGroupStop(CustomTestCase):
    def test_group_term_takes_the_child(self):
        # a stand-in deadman: session leader with a sleeping child
        p = subprocess.Popen(["bash", "-c", "exec -a boot_deadman_h135 bash -c 'sleep 3135 & wait'"],
                             start_new_session=True)
        self.addCleanup(lambda: (p.poll() is None) and os.killpg(p.pid, signal.SIGKILL))
        t0 = time.time()
        while time.time() - t0 < 5:
            kids = subprocess.run(["pgrep", "-g", str(p.pid)], capture_output=True, text=True).stdout.split()
            if len(kids) >= 2:
                break
            time.sleep(0.05)
        self.assertEqual(L._stop_helper_pid(p.pid, expect="boot_deadman_h135"), "group")
        p.wait(timeout=5)
        t0 = time.time()
        while time.time() - t0 < 5:
            if not subprocess.run(["pgrep", "-g", str(p.pid)], capture_output=True, text=True).stdout.split():
                break
            time.sleep(0.05)
        self.assertEqual(
            subprocess.run(["pgrep", "-g", str(p.pid)], capture_output=True, text=True).stdout.split(), [],
            "the sleeping child dies with its group")


if __name__ == "__main__":
    unittest.main()
