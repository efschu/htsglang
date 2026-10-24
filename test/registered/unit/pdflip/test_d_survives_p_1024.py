"""S7 D-SURVIVES-P (24.10.): a dying MPS client rank leaves no GPU work on the card it shares.

Specimen: 27B NVFP4 dual rc39a (b9cddabda1), boot dkr27bnvfp4dualbar1drei10240856. P and D are
clients of ONE MPS daemon on cards 0,1,2 (--dual-mps on). 09:09:47 P PP1 (card 1) dies on the closed
gloo pair with its last prefill chunk still on the card and exits without a device synchronize; host
kernel log the same second: 'NVRM: Xid (PCI:0000:18:00): 31, pid=<D TP1> ... FAULT_PDE
ACCESS_TYPE_VIRT_READ' -> D TP1 'illegal memory access' (W113 status copy, dual_d_kv_stage.tick ->
max_live_id) -> D dead. Three P deaths at a host-side scheduling point (10210031, 10220215,
10240650): no Xid, D kept decoding. NVIDIA MPS: ending a client by exception/signal with work in
flight is undefined for the server and ALL its clients; terminate_client is the safe way.

Danger directions: (1) the specimen -- a rank exits / is SIGKILLed with work on the card (no drain,
no terminate, or terminate AFTER the kill); (2) a non-MPS boot or the switch off changes behaviour;
(3) terminate_client aimed at a pid that is not this daemon's client, or a refused terminate counted
as done; (4) a drained rank terminated anyway (terminate_client is the fallback, it has a field
report of hitting untouched clients).
"""

from __future__ import annotations

import inspect
import os
import threading
import types
import unittest
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import psutil  # noqa: E402

from flliper.srt.pdflip import mps_exit_quiesce as Q  # noqa: E402
from flliper.srt.utils import common as C  # noqa: E402

MPS = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/pdflip-dual-mps-t/pipe"}


class Armed(unittest.TestCase):
    def test_armed_only_for_an_mps_client_with_the_switch_on(self):
        self.assertFalse(Q.armed({}))
        self.assertFalse(Q.armed({"CUDA_MPS_PIPE_DIRECTORY": "  "}))
        self.assertTrue(Q.armed(dict(MPS)))
        self.assertTrue(Q.armed(dict(MPS, **{Q.ENV: "1"})))
        for off in ("0", "false", "off", "no"):
            self.assertFalse(Q.armed(dict(MPS, **{Q.ENV: off})), off)

    def test_bound(self):
        self.assertEqual(Q.bound_s({}), Q.DEFAULT_BOUND_S)
        self.assertEqual(Q.bound_s({Q.BOUND_ENV: "2.5"}), 2.5)
        self.assertEqual(Q.bound_s({Q.BOUND_ENV: "x"}), Q.DEFAULT_BOUND_S)


class QuiesceSelf(unittest.TestCase):
    def _fakes(self, ev, sync_impl=None):
        def trip(reason):
            ev.append(("trip", reason))

        def sync():
            ev.append(("sync",))
            if sync_impl:
                sync_impl()

        def terminate(pids, why=""):
            ev.append(("terminate", list(pids)))
            return list(pids)

        return trip, sync, terminate

    def test_off_without_mps_touches_nothing(self):
        ev = []
        trip, sync, terminate = self._fakes(ev)
        self.assertEqual(Q.quiesce_self("x", env={}, trip=trip, sync=sync, terminate=terminate, pid=7), "off")
        self.assertEqual(Q.quiesce_self("x", env=dict(MPS, **{Q.ENV: "0"}), trip=trip, sync=sync,
                                        terminate=terminate, pid=7), "off")
        self.assertEqual(ev, [])

    def test_a_rank_that_drains_is_not_terminated(self):
        ev = []
        trip, sync, terminate = self._fakes(ev)
        v = Q.quiesce_self("AssertionError", env=dict(MPS), trip=trip, sync=sync, terminate=terminate, pid=7,
                           bound=2.0)
        self.assertEqual(v, "drained")
        self.assertEqual([e[0] for e in ev], ["trip", "sync"])   # spin kernels released before the sync waits

    def test_specimen_work_left_on_the_card_is_terminated_before_exit(self):
        # PP1 09:09:47: the chunk / the PP traffic to the dead PP2 never completes on its own -- the
        # synchronize would wait forever; only the daemon's terminate_client ends it.
        ended = threading.Event()
        ev = []
        trip, _sync, _t = self._fakes(ev)

        def sync():
            ev.append(("sync",))
            ended.wait(5.0)

        def terminate(pids, why=""):
            ev.append(("terminate", list(pids)))
            ended.set()
            return list(pids)

        v = Q.quiesce_self("RuntimeError", env=dict(MPS), trip=trip, sync=sync, terminate=terminate, pid=1240,
                           bound=0.05)
        self.assertEqual(v, "terminated")
        self.assertEqual(ev[-1], ("terminate", [1240]))

    def test_a_refused_terminate_is_named_undrained(self):
        hang = threading.Event()
        v = Q.quiesce_self("x", env=dict(MPS), trip=lambda r: None, sync=lambda: hang.wait(2.0),
                           terminate=lambda pids, why="": [], pid=5, bound=0.02)
        hang.set()
        self.assertEqual(v, "undrained")

    def test_a_sync_error_is_drained_error_and_never_raises(self):
        def sync():
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        def trip(r):
            raise OSError("no window")

        v = Q.quiesce_self("x", env=dict(MPS), trip=trip, sync=sync, terminate=lambda p, why="": [], pid=5,
                           bound=2.0)
        self.assertEqual(v, "drained-error")


class TerminateClients(unittest.TestCase):
    def _ctl(self, cmds, refuse_terminate=False):
        def ctl(cmd):
            cmds.append(cmd)
            if cmd == "get_server_list":
                return "4242\n"
            if cmd == "get_client_list 4242":
                return "1238\n1239\n1240\n1241\n"
            if cmd.startswith("terminate_client"):
                return None if refuse_terminate else ""
            return None

        return ctl

    def test_only_this_daemons_clients_are_terminated(self):
        cmds = []
        done = Q.terminate_clients([1240, 9999], why="t", ctl=self._ctl(cmds))
        self.assertEqual(done, [1240])
        self.assertEqual([c for c in cmds if c.startswith("terminate_client")], ["terminate_client 4242 1240"])

    def test_a_refused_terminate_is_not_counted(self):
        self.assertEqual(Q.terminate_clients([1240], ctl=self._ctl([], refuse_terminate=True)), [])

    def test_control_reads_the_daemon_answer(self):
        class R:
            def __init__(self, rc, out):
                self.returncode, self.stdout, self.stderr = rc, out, ""

        calls = []

        def run(argv, **kw):
            calls.append((argv, kw.get("input"), (kw.get("env") or {}).get("CUDA_MPS_PIPE_DIRECTORY")))
            return R(0, "4242\n")

        self.assertEqual(Q.control("get_server_list", env=dict(MPS), run=run), "4242\n")
        self.assertEqual(calls[0], ([Q.CONTROL], "get_server_list\n", MPS["CUDA_MPS_PIPE_DIRECTORY"]))
        self.assertIsNone(Q.control("terminate_client 1 2", env=dict(MPS), run=lambda a, **k: R(0, "Error: no such")))
        self.assertIsNone(Q.control("x", env=dict(MPS), run=lambda a, **k: R(1, "")))

        def boom(a, **k):
            raise FileNotFoundError(Q.CONTROL)

        self.assertIsNone(Q.control("x", env=dict(MPS), run=boom))


class KillProcessTree(unittest.TestCase):
    def _tree(self, ev):
        class Child:
            def __init__(self, pid):
                self.pid = pid

            def kill(self):
                ev.append(("kill", self.pid))

        class Parent:
            def __init__(self, pid):
                self.pid = pid

            def children(self, recursive=True):
                return [Child(1238), Child(1240), Child(1300)]

        return types.SimpleNamespace(Process=Parent, NoSuchProcess=psutil.NoSuchProcess)

    def _run(self, env):
        ev = []

        def terminate(pids, why=""):
            ev.append(("terminate", sorted(pids)))
            return list(pids)

        with mock.patch.object(C, "psutil", self._tree(ev)), mock.patch.object(Q, "terminate_clients", terminate), \
                mock.patch.dict(os.environ, env, clear=False):
            if "CUDA_MPS_PIPE_DIRECTORY" not in env:
                os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
            C.kill_process_tree(123, include_parent=False, skip_pid=1300)
        return ev

    def test_mps_children_are_terminated_before_the_sigkill(self):
        ev = self._run(dict(MPS))
        self.assertEqual(ev[0], ("terminate", [1238, 1240]))       # skip_pid stays out
        self.assertEqual(ev[1:], [("kill", 1238), ("kill", 1240)])

    def test_without_mps_the_kill_is_unchanged(self):
        ev = self._run({})
        self.assertEqual(ev, [("kill", 1238), ("kill", 1240)])

    def test_switch_off_restores_the_old_kill(self):
        ev = self._run(dict(MPS, **{Q.ENV: "0"}))
        self.assertEqual(ev, [("kill", 1238), ("kill", 1240)])


class SchedulerDeathPath(unittest.TestCase):
    def test_the_dying_rank_quiesces_before_it_signals_and_before_the_hold_decision(self):
        from flliper.srt.managers import scheduler as SC

        src = inspect.getsource(SC.run_scheduler_process)
        death = src.index("Scheduler hit an exception")
        q = src.index("_s7q.quiesce_self(", death)
        self.assertLess(q, src.index("_1223_hold_on = False", death))
        self.assertLess(q, src.index("parent_process.send_signal(signal.SIGQUIT)", death))
        self.assertLess(q, src.index("os.killpg(pgid, signal.SIGKILL)", death))


class TerminateBudget(unittest.TestCase):
    """Review SOLL 1: a hung daemon costs the budget, not 3 x 10 s / (2 + N) x 10 s."""

    def test_nothing_is_asked_once_the_budget_is_spent(self):
        clock = [0.0]
        cmds = []

        def ctl(cmd):
            cmds.append(cmd)
            clock[0] += 5.0                        # every daemon call takes 5 s (a slow / hanging daemon)
            if cmd == "get_server_list":
                return "4242\n"
            if cmd == "get_client_list 4242":
                return "1238\n1240\n1242\n"
            return ""

        done = Q.terminate_clients([1238, 1240, 1242], why="t", ctl=ctl, budget_s=12.0, now=lambda: clock[0])
        self.assertEqual(done, [1238])             # 0 s, 5 s, 10 s asked; at 15 s the 12 s budget is spent
        self.assertEqual(len(cmds), 3)
        self.assertLessEqual(clock[0], 12.0 + 5.0)

    def test_the_real_control_gets_the_rest_of_the_budget_as_its_timeout(self):
        clock = [0.0]
        seen = []

        def fake_control(cmd, timeout_s=None, **kw):
            seen.append((cmd, timeout_s))
            clock[0] += 9.0
            return {"get_server_list": "4242\n", "get_client_list 4242": "1240\n"}.get(cmd, "")

        with mock.patch.object(Q, "control", fake_control):
            done = Q.terminate_clients([1240], why="t", budget_s=12.0, now=lambda: clock[0])
        self.assertEqual([t for _c, t in seen], [Q.CONTROL_TIMEOUT_S, 3.0])
        self.assertEqual(done, [])                 # the terminate would start at 18 s > 12 s: not asked


class SchedulerBaseExceptionPath(unittest.TestCase):
    def test_keyboardinterrupt_and_systemexit_quiesce_then_propagate(self):
        from flliper.srt.managers import scheduler as SC

        src = inspect.getsource(SC.run_scheduler_process)
        b = src.index("except BaseException as scheduler_base_exc:")
        fin = src.index("    finally:", b)
        body = src[b:fin]
        self.assertIn("_s7q.quiesce_self(", body)
        self.assertIn("scheduler_base_exc.code in (0, None)", body)   # a graceful SystemExit stays untouched
        self.assertTrue(body.rstrip().endswith("raise"))
        self.assertLess(src.index("except Exception as scheduler_exc:"), b)


if __name__ == "__main__":
    unittest.main()
