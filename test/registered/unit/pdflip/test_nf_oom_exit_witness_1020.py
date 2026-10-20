# SPDX-License-Identifier: Apache-2.0
"""NF rc26, boot ...dauer10201307, flip 51 (P->D), 20.10. 14:18:29Z: D TP0
stopped writing after ``[core.cpp] WEG2-RING LEG tag=weights_14 ... allocs=21
bytes=1264582656`` -- no traceback, no terminate, no "Fatal Python error".

(A) WURZEL (host journal, proxmox, 16:18:29 local = 14:18:29Z)::

    ZMQbg/IO/0 invoked oom-killer: gfp_mask=0x100cca(GFP_HIGHUSER_MOVABLE)
    memory: usage 109051904kB, limit 109051904kB, failcnt 38250
    oom-kill:constraint=CONSTRAINT_MEMCG ... oom_memcg=/system.slice/docker-e75b9129....scope
    Memory cgroup out of memory: Killed process 1798627 (flliper::schedu)
        total-vm:249923888kB, anon-rss:6678972kB, file-rss:66872kB,
        shmem-rss:72993760kB, UID:0 pgtables:330668kB oom_score_adj:500

The container's 104 GiB ``memory.max`` was reached; the kernel SIGKILLed the
largest process, D TP0. A SIGKILLed rank writes nothing.

(B) WITNESS GAP 1 -- state.json. The FIRST cause came from a SURVIVOR: TP2's
``RuntimeError: [gloo ... pair.cc:547] Connection closed by peer`` filed as
RANK_EXCEPTION tp2pp0. Now every rank arms the cgroup ``oom_kill`` baseline at
process start and files KERNEL_OOM_KILL when the counter rose -- same rule,
same counter, every rank (rank-unity), the own exception kept in the detail.

(C) WITNESS GAP 2 -- the D log. The parent's SubprocessWatchdog polls once a
second; TP1/TP2's SIGQUIT arrived before the next poll, and the SIGQUIT
handler's first act was ``stop()``: "killed by SIGKILL ... oom_kill 0 -> 1"
was never written. Now the handler sweeps every child before it stops the watch.

CPU only, no GPU, no CUDA context.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import state_file as sf  # noqa: E402
from flliper.srt.utils import watchdog as wd  # noqa: E402


def _write_cgroup(root: str, oom_kill: int, current: int = 109051904 * 1024,
                  mx: int = 109051904 * 1024) -> None:
    with open(os.path.join(root, "memory.events"), "w") as f:
        f.write("low 0\nhigh 0\nmax 38250\noom %d\noom_kill %d\noom_group_kill 0\n"
                % (oom_kill, oom_kill))
    with open(os.path.join(root, "memory.current"), "w") as f:
        f.write("%d\n" % current)
    with open(os.path.join(root, "memory.max"), "w") as f:
        f.write("%d\n" % mx)


def _state_dir(tmp: str) -> str:
    d = sf.init(tmp, "nfoomwitness-boot-1020", "boot", {})
    sf.transition(d, "launching", writer="host")
    sf.transition(d, "loading", writer="host")
    sf.transition(d, "serving", writer="host")
    return d


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cg = os.path.join(self._tmp.name, "cg")
        os.makedirs(self.cg)
        self._saved = dict(sf._OOM_WITNESS)
        sf._OOM_WITNESS.update({"baseline": None, "armed": False})
        self._p = mock.patch.object(sf, "_cgroup_memory_root", return_value=self.cg)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        sf._OOM_WITNESS.clear()
        sf._OOM_WITNESS.update(self._saved)
        self._tmp.cleanup()


class StateFileNamesTheKernelOomKill(_Base):
    def _gloo(self):
        return RuntimeError("[/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:547] "
                            "Connection closed by peer [172.17.0.3]:19353")

    def test_1418_the_survivor_files_kernel_oom_kill_not_its_own_consequence(self):
        """RED on rc26/rc27: cause.code RANK_EXCEPTION, detail = the gloo line."""
        _write_cgroup(self.cg, 0)
        self.assertEqual(sf.arm_oom_witness(), 0)
        d = _state_dir(os.path.join(self._tmp.name, "st"))
        _write_cgroup(self.cg, 1)                       # the kernel killed D TP0
        exc = self._gloo()
        self.assertTrue(sf.note_rank_death(d, group="D", tp_rank=2, pp_rank=0, exc=exc,
                                           detail="RuntimeError: " + str(exc)))
        cause = sf.read(d)["cause"]
        self.assertEqual(cause["code"], sf.KERNEL_OOM_KILL_CODE)
        self.assertEqual(cause["origin"], "rank")
        self.assertEqual(cause["rank"], "tp2pp0")
        self.assertEqual(cause["exception_type"], "RuntimeError")
        self.assertIn("oom_kill 0 -> 1", cause["detail_full"])
        self.assertIn("Connection closed by peer", cause["detail_full"])   # nothing overwritten
        self.assertEqual(sf.last_rank_death_code(), sf.KERNEL_OOM_KILL_CODE)

    def test_no_kill_keeps_rank_exception(self):
        _write_cgroup(self.cg, 3)
        sf.arm_oom_witness()
        d = _state_dir(os.path.join(self._tmp.name, "st"))
        self.assertIsNone(sf.oom_witness())
        sf.note_rank_death(d, group="D", tp_rank=0, pp_rank=0, exc=ValueError("x"))
        self.assertEqual(sf.read(d)["cause"]["code"], sf.RANK_EXCEPTION_CODE)
        self.assertEqual(sf.last_rank_death_code(), sf.RANK_EXCEPTION_CODE)

    def test_unarmed_or_unreadable_is_never_a_verdict(self):
        _write_cgroup(self.cg, 5)
        self.assertIsNone(sf.oom_witness())             # never armed
        os.remove(os.path.join(self.cg, "memory.events"))
        self.assertIsNone(sf.arm_oom_witness())         # unreadable baseline
        _write_cgroup(self.cg, 9)
        self.assertIsNone(sf.oom_witness())

    def test_first_arm_wins(self):
        _write_cgroup(self.cg, 0)
        sf.arm_oom_witness()
        _write_cgroup(self.cg, 1)
        self.assertEqual(sf.arm_oom_witness(), 0)       # a re-arm cannot hide the kill
        self.assertIn("oom_kill 0 -> 1", sf.oom_witness())

    def test_rank_unity_every_survivor_decides_the_same(self):
        """D TP1, D TP2, P PP1, P PP2 all died of the consequence: whichever
        writes first, the boot carries one answer, and each rank's own
        decision (its log line) is the same."""
        _write_cgroup(self.cg, 0)
        sf.arm_oom_witness()
        _write_cgroup(self.cg, 1)
        codes = []
        for i, (g, tp, pp, exc) in enumerate([
                ("D", 1, 0, RuntimeError("Connection closed by peer")),
                ("D", 2, 0, RuntimeError("Connection closed by peer")),
                ("P", 0, 1, BrokenPipeError(32, "Broken pipe")),
                ("P", 0, 2, ConnectionResetError(104, "Connection reset by peer"))]):
            d = _state_dir(os.path.join(self._tmp.name, "st%d" % i))
            sf.note_rank_death(d, group=g, tp_rank=tp, pp_rank=pp, exc=exc)
            codes.append(sf.read(d)["cause"]["code"])
        self.assertEqual(set(codes), {sf.KERNEL_OOM_KILL_CODE})

    def test_witness_line_names_memory_against_max(self):
        _write_cgroup(self.cg, 0)
        sf.arm_oom_witness()
        _write_cgroup(self.cg, 1)
        line = sf.oom_witness()
        self.assertIn("memory.current=106496 MiB", line)
        self.assertIn("memory.max=106496 MiB", line)


def _sleeper():
    time.sleep(60)


class SigquitHandlerWitnessesTheDeadChild(unittest.TestCase):
    """The 14:18:29 race: child SIGKILLed, a peer's SIGQUIT arrives before
    the watchdog's next poll, the handler stops the watch."""

    def setUp(self):
        self.ctx = mp.get_context("spawn")
        self.victim = self.ctx.Process(target=_sleeper)
        self.alive = self.ctx.Process(target=_sleeper)
        self.victim.start()
        self.alive.start()

    def tearDown(self):
        for p in (self.victim, self.alive):
            if p.is_alive():
                p.kill()
            p.join(5)

    def _kill_victim(self):
        os.kill(self.victim.pid, signal.SIGKILL)
        # Wait until the exit is REPORTABLE without reaping it (WNOWAIT): a
        # leader shown as "Z" in /proc can still have threads finishing their
        # exit, and waitpid answers (0, 0) until the last one is gone.
        deadline = time.time() + 30
        while time.time() < deadline:
            info = os.waitid(os.P_PID, self.victim.pid,
                             os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if info is not None:
                return
            time.sleep(0.01)
        self.fail("victim not reportable within 30 s")

    def _watchdog(self):
        # interval 3600 s: the poll never runs inside this test -- exactly the
        # position of the 1 s poll that the peers' SIGQUIT overtook.
        return wd.SubprocessWatchdog(processes=[self.victim, self.alive],
                                     process_names=["scheduler_0", "scheduler_1"],
                                     interval=3600.0)

    def test_sweep_reports_the_sigkilled_child_with_the_oom_delta(self):
        with mock.patch.object(wd, "_oom_kill_count", side_effect=[0, 1]):
            w = self._watchdog()
            self._kill_victim()
            with self.assertLogs(wd.logger, level=logging.ERROR) as cm:
                n = w.report_dead_children("SIGQUIT")
        text = "\n".join(cm.output)
        self.assertEqual(n, 1)
        self.assertIn("scheduler_0", text)
        self.assertIn("killed by SIGKILL (exit code -9)", text)
        self.assertIn("oom_kill went 0 -> 1", text)
        self.assertNotIn("scheduler_1 (pid", text)
        self.assertIn("EXIT-WITNESS at SIGQUIT: 1 of 2", text)

    def test_sweep_right_after_the_kill_waits_for_the_dying_child(self):
        """Production timing: the sweep runs while the SIGKILLed rank may
        still be tearing down (leader "Z", waitpid (0, 0)). No test-side
        wait here -- the sweep's own bounded DYING wait must read -9."""
        with mock.patch.object(wd, "_oom_kill_count", side_effect=[0, 1]):
            w = self._watchdog()
            os.kill(self.victim.pid, signal.SIGKILL)
            with self.assertLogs(wd.logger, level=logging.ERROR) as cm:
                n = w.report_dead_children("SIGQUIT")
        text = "\n".join(cm.output)
        self.assertEqual(n, 1)
        self.assertTrue("killed by SIGKILL (exit code -9)" in text or "is DYING" in text, text)
        self.assertIn("oom_kill went 0 -> 1", text)

    def test_a_child_still_dying_after_the_wait_is_named_dying(self):
        fake = mock.Mock()
        fake.pid = 4242
        fake.is_alive.return_value = True
        with mock.patch.object(wd, "_oom_kill_count", side_effect=[0, 1]), \
                mock.patch.object(wd, "_proc_state", return_value="Z"), \
                mock.patch.object(wd.SubprocessWatchdog, "DYING_WAIT_S", 0.05):
            w = wd.SubprocessWatchdog(processes=[fake], process_names=["scheduler_0"],
                                      interval=3600.0)
            with self.assertLogs(wd.logger, level=logging.ERROR) as cm:
                self.assertEqual(w.report_dead_children("SIGQUIT"), 1)
        text = "\n".join(cm.output)
        self.assertIn("scheduler_0 (pid=4242) is DYING", text)
        self.assertIn("oom_kill went 0 -> 1", text)

    def test_sweep_with_every_child_alive_says_so(self):
        w = self._watchdog()
        with self.assertLogs(wd.logger, level=logging.ERROR) as cm:
            self.assertEqual(w.report_dead_children("SIGQUIT"), 0)
        self.assertIn("all 2 tracked children still alive", "\n".join(cm.output))

    def test_sweep_does_not_report_a_child_twice(self):
        with mock.patch.object(wd, "_oom_kill_count", return_value=0):
            w = self._watchdog()
            self._kill_victim()
            with self.assertLogs(wd.logger, level=logging.ERROR) as cm:
                w.report_dead_children("poll")
                w.report_dead_children("SIGQUIT")
        self.assertEqual(sum("scheduler_0 (pid" in l for l in cm.output), 1)

    def test_sigquit_handler_sweeps_before_it_stops_the_watch(self):
        """RED on rc26/rc27: the handler only calls stop(); no line names
        scheduler_0 and its SIGKILL."""
        from flliper.srt.managers import tokenizer_manager as tm

        with mock.patch.object(wd, "_oom_kill_count", side_effect=[0, 1]):
            w = self._watchdog()
            self._kill_victim()
            order = []
            real_sweep, real_stop = w.report_dead_children, w.stop
            w.report_dead_children = lambda why: (order.append("sweep"), real_sweep(why))[1]
            w.stop = lambda: (order.append("stop"), real_stop())[1]
            fake_tm = mock.Mock()
            fake_tm._subprocess_watchdog = w
            with mock.patch.object(tm, "kill_process_tree") as kpt, \
                    self.assertLogs(wd.logger, level=logging.ERROR) as cm:
                tm.SignalHandler(fake_tm).running_phase_sigquit_handler()
        self.assertEqual(order, ["sweep", "stop"])
        kpt.assert_called_once()
        text = "\n".join(cm.output)
        self.assertIn("scheduler_0", text)
        self.assertIn("killed by SIGKILL", text)
        self.assertIn("oom_kill went 0 -> 1", text)


class SchedulerArmsAndLogsTheWitness(unittest.TestCase):
    """Wiring, by source: the arm runs at the top of run_scheduler_process
    (before any rank can have died) and the death path logs the witness and
    the code note_rank_death really chose."""

    def test_wiring(self):
        import inspect

        from flliper.srt.managers import scheduler as sch

        src = inspect.getsource(sch.run_scheduler_process)
        arm = src.index("arm_oom_witness()")
        self.assertLess(arm, src.index("configure_scheduler_process("))
        self.assertLess(src.index("_rd_state.oom_witness()"), src.index("_rd_state.note_rank_death("))
        self.assertIn("_rd_state.last_rank_death_code()", src)


if __name__ == "__main__":
    unittest.main()
