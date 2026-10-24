"""Review of S7 D-SURVIVES-P (desk/27b-d-survives-p-1024 @ 804fbbb450): danger-direction tests the
author's 14 do not pin.

(R1) A hung MPS control daemon (NVIDIA forum 382783: terminate_client "intermittently never returns")
     must not hang the dying rank or kill_process_tree: every control call carries a finite timeout,
     and a TimeoutExpired is a refusal (None), never a raise.
(R2) The bounded sync runs in a DAEMON thread -- a non-daemon thread blocked in cudaDeviceSynchronize
     is joined at interpreter shutdown and the rank never exits.
(R3) Without an injected pid the dying rank terminates ITSELF (os.getpid()), never its parent or
     anyone else.
(R4) End to end through kill_process_tree with the real terminate_clients: a pid the daemon lists as
     its client but that is NOT a child of this tree (D's ranks under --dual-mps) is never terminated.
"""

from __future__ import annotations

import os
import subprocess
import threading
import types
import unittest
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import psutil  # noqa: E402

from flliper.srt.pdflip import mps_exit_quiesce as Q  # noqa: E402
from flliper.srt.utils import common as C  # noqa: E402

MPS = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/pdflip-dual-mps-r/pipe"}


class ControlIsBounded(unittest.TestCase):
    def test_every_control_call_carries_a_finite_timeout(self):
        seen = []

        class R:
            returncode, stdout, stderr = 0, "", ""

        def run(argv, **kw):
            seen.append(kw.get("timeout"))
            return R()

        Q.control("get_server_list", env=dict(MPS), run=run)
        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(seen[0])
        self.assertGreater(float(seen[0]), 0.0)
        self.assertLessEqual(float(seen[0]), Q.CONTROL_TIMEOUT_S)

    def test_a_hung_daemon_is_a_refusal_not_a_raise(self):
        def run(argv, **kw):
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))

        self.assertIsNone(Q.control("terminate_client 1 2", env=dict(MPS), run=run))


class SyncThread(unittest.TestCase):
    def test_the_sync_runs_in_a_daemon_thread(self):
        seen = {}
        release = threading.Event()

        def sync():
            t = threading.current_thread()
            seen["daemon"] = t.daemon
            seen["main"] = t is threading.main_thread()
            release.wait(2.0)

        try:
            Q.quiesce_self("x", env=dict(MPS), trip=lambda r: None, sync=sync,
                           terminate=lambda pids, why="": [], pid=5, bound=0.2)
        finally:
            release.set()
        self.assertIs(seen.get("main"), False)
        self.assertIs(seen.get("daemon"), True)

    def test_without_an_injected_pid_the_rank_terminates_itself(self):
        asked = []
        release = threading.Event()

        def terminate(pids, why=""):
            asked.append(list(pids))
            release.set()
            return list(pids)

        v = Q.quiesce_self("x", env=dict(MPS), trip=lambda r: None, sync=lambda: release.wait(2.0),
                           terminate=terminate, bound=0.02)
        self.assertEqual(asked, [[os.getpid()]])
        self.assertEqual(v, "terminated")


class KillProcessTreeNeverTerminatesAStranger(unittest.TestCase):
    def test_a_daemon_client_outside_this_tree_is_never_terminated(self):
        ev = []

        class Child:
            def __init__(self, pid):
                self.pid = pid

            def kill(self):
                ev.append(("kill", self.pid))

        class Parent:
            def __init__(self, pid):
                self.pid = pid

            def children(self, recursive=True):
                return [Child(1238), Child(1240), Child(1242)]   # P PP0/PP1/PP2

        fake_psutil = types.SimpleNamespace(Process=Parent, NoSuchProcess=psutil.NoSuchProcess)

        def ctl(cmd):
            ev.append(("ctl", cmd))
            if cmd == "get_server_list":
                return "4242\n"
            if cmd == "get_client_list 4242":
                return "1238\n1239\n1240\n1241\n1242\n1243\n"   # P and D, one daemon
            if cmd.startswith("terminate_client"):
                return ""
            return None

        real = Q.terminate_clients   # the real intersection, only the daemon is faked

        def terminate(pids, why=""):
            return real(pids, why=why, ctl=ctl)

        with mock.patch.object(C, "psutil", fake_psutil), mock.patch.object(Q, "terminate_clients", terminate), \
                mock.patch.dict(os.environ, dict(MPS), clear=False):
            os.environ.pop(Q.ENV, None)
            C.kill_process_tree(902, include_parent=False)
        terms = [c for k, c in ev if k == "ctl" and c.startswith("terminate_client")]
        self.assertEqual(sorted(terms), ["terminate_client 4242 1238", "terminate_client 4242 1240",
                                         "terminate_client 4242 1242"])
        for d in (1239, 1241, 1243):
            self.assertNotIn(f"terminate_client 4242 {d}", terms)
        first_kill = next(i for i, e in enumerate(ev) if e[0] == "kill")
        last_term = max(i for i, e in enumerate(ev) if e[0] == "ctl" and e[1].startswith("terminate_client"))
        self.assertLess(last_term, first_kill)


if __name__ == "__main__":
    unittest.main()
