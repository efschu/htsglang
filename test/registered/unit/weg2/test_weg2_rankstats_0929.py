"""DASHBOARD-AUS-IPC, the open sources (29.09.): rank-side ``rankstats`` and the
front's ``errors`` + ``group_health``.

27B's conditions for rankstats (review of the plan, 29.09.):
* never written from the decode/forward path -- one timer thread reads counters
  the round path increments anyway, no lock in the path, atomic tmp+replace;
* switch off -> no thread;
* a mutant without the thread writes 0 times per round;
* display only, it replaces no deadman riegel.

RED on 481cba2a3c: weg2/rankstats.py, the env switch, group_health_verdict and the
front's errors/group_health do not exist.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import front_health as _fh
from sglang.srt.weg2 import front_state_ipc as fsi
from sglang.srt.weg2 import rank_state as rs
from sglang.srt.weg2 import rankstats
from sglang.srt.weg2 import state_file
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SCHED_PY = os.path.join(os.path.dirname(front_mod.__file__), "..", "managers", "scheduler.py")


def _fake_scheduler():
    mr = SimpleNamespace(prefill_tokens_total=0, gen_tokens_total=0,
                         spec_total_num_accept_tokens=0, spec_total_num_forward_ct=0)
    return SimpleNamespace(forward_ct=0, metrics_reporter=mr, waiting_queue=[],
                           running_batch=SimpleNamespace(reqs=[]))


def _round(s, n=1):
    """What the round path does anyway: plain int increments."""
    s.forward_ct += n
    s.metrics_reporter.gen_tokens_total += n


def _threads():
    return [t for t in threading.enumerate() if t.name == rankstats.THREAD_NAME and t.is_alive()]


class TestSwitch(CustomTestCase):
    def tearDown(self):
        if rankstats._CURRENT is not None:
            rankstats._CURRENT.stop()
            rankstats._CURRENT = None

    def test_switch_off_starts_no_thread_and_writes_nothing(self):
        d = tempfile.mkdtemp(prefix="rkst-off-")
        before = len(_threads())
        with envs.SGLANG_WEG2_ENABLE_RANKSTATS.override(False), envs.SGLANG_WEG2_RANK_STATE_DIR.override(d):
            self.assertIsNone(rankstats.maybe_start(_fake_scheduler(), tp_rank=0, pp_rank=0))
        self.assertEqual(len(_threads()), before)
        self.assertEqual(os.listdir(d), [])

    def test_no_rank_state_dir_starts_no_thread(self):
        before = len(_threads())
        with envs.SGLANG_WEG2_ENABLE_RANKSTATS.override(True), envs.SGLANG_WEG2_RANK_STATE_DIR.override(None):
            self.assertIsNone(rankstats.maybe_start(_fake_scheduler(), tp_rank=0, pp_rank=0))
        self.assertEqual(len(_threads()), before)

    def test_switch_on_timer_writes_the_rank_file(self):
        d = tempfile.mkdtemp(prefix="rkst-on-")
        s = _fake_scheduler()
        _round(s, 7)
        s.metrics_reporter.prefill_tokens_total = 16384
        s.waiting_queue.extend([1, 2])
        with envs.SGLANG_WEG2_ENABLE_RANKSTATS.override(True), envs.SGLANG_WEG2_RANK_STATE_DIR.override(d), \
                envs.SGLANG_WEG2_RANKSTATS_PERIOD_S.override(0.2), \
                mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "d"}):
            r = rankstats.maybe_start(s, tp_rank=1, pp_rank=0)
            self.assertIsNotNone(r)
            self.assertEqual(len(_threads()) >= 1, True)
            rankstats.note_post_wake({"n": 0, "run_ms": 31.0})
            path = os.path.join(d, "D.tp1pp0.rankstats")
            deadline = time.time() + 5
            while time.time() < deadline and not os.path.exists(path):
                time.sleep(0.05)
            with open(path) as f:
                rec = json.load(f)
        self.assertEqual(rec["schema"], rankstats.SCHEMA)
        self.assertEqual((rec["group"], rec["tp_rank"], rec["pp_rank"]), ("D", 1, 0))
        self.assertEqual(rec["work"]["forward_ct"], 7)
        self.assertEqual(rec["tokens"], {"prefill_total": 16384, "decode_total": 7})
        # §3 (29.09.) extends sched with queue_req/running_req/pending_tokens
        self.assertEqual({k: rec["sched"][k] for k in ("waiting", "running")},
                         {"waiting": 2, "running": 0})
        self.assertEqual(rec["last_post_wake"], {"n": 0, "run_ms": 31.0})
        self.assertIn("errors", rec)
        self.assertFalse([n for n in os.listdir(d) if ".tmp." in n])
        # the RankState reader / W7-W10 gate never sees it; a relaunch clears it
        states, bad = rs.read_group_states(d)
        self.assertEqual((states, bad), ([], []))
        self.assertEqual(rs.clear_rank_state_dir(d), 1)


class TestNeverFromTheRoundPath(CustomTestCase):
    def test_mutant_without_thread_writes_zero_times_per_round(self):
        """RankStats never started (= the timer removed): the round path's own
        increments and the post-wake note write nothing, however many rounds."""
        d = tempfile.mkdtemp(prefix="rkst-mut-")
        s = _fake_scheduler()
        r = rankstats.RankStats(state_dir=d, group="D", tp_rank=0, pp_rank=0,
                                read_counters=lambda: rankstats.scheduler_counters(s), period=0.2)
        rankstats._CURRENT = r
        try:
            with mock.patch("os.replace") as rep, mock.patch("builtins.open") as op:
                for _ in range(5000):
                    _round(s)
                    rankstats.note_post_wake({"n": 1})
                self.assertEqual((rep.call_count, op.call_count), (0, 0))
            self.assertEqual(r.writes, 0)
            self.assertEqual(os.listdir(d), [])
        finally:
            rankstats._CURRENT = None

    def test_writes_follow_the_clock_not_the_rounds(self):
        d = tempfile.mkdtemp(prefix="rkst-clk-")
        s = _fake_scheduler()
        r = rankstats.RankStats(state_dir=d, group="D", tp_rank=0, pp_rank=0,
                                read_counters=lambda: rankstats.scheduler_counters(s), period=0.5).start()
        try:
            for _ in range(20000):
                _round(s)
            self.assertLessEqual(r.writes, 1)  # 20k rounds inside < 1 period
        finally:
            r.stop()

    def test_only_the_process_start_and_the_post_wake_census_name_rankstats(self):
        """AST: in scheduler.py only run_scheduler_process (start) and the
        post-wake census (one assignment) touch rankstats -- no round function."""
        with open(SCHED_PY) as f:
            tree = ast.parse(f.read())
        users = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if "rankstats" in ast.dump(node):
                    users.add(node.name)
        self.assertEqual(users, {"run_scheduler_process", "_weg2_post_wake_pass_log"})


class TestErrorTally(CustomTestCase):
    def test_counts_and_keeps_the_last_eight_structured(self):
        lg = logging.getLogger("sglang.test.rankstats.tally")
        h = rankstats.ErrorTally()
        lg.addHandler(h)
        try:
            lg.warning("not an error")
            for i in range(10):
                lg.error("boom %d", i)
            try:
                raise ValueError("x")
            except ValueError:
                lg.exception("with exc")
        finally:
            lg.removeHandler(h)
        snap = h.snapshot()
        self.assertEqual(snap["n"], 11)
        self.assertEqual(len(snap["last"]), 8)
        self.assertEqual(snap["last"][-1]["exc"], "ValueError")
        self.assertEqual(snap["last"][-2]["text"], "boom 9")

    def test_install_is_idempotent(self):
        a = rankstats.install_error_tally("sglang.test.rankstats.idem")
        b = rankstats.install_error_tally("sglang.test.rankstats.idem")
        self.assertIs(a, b)


def _front():
    return front_mod.Front(
        prefill="http://p", decode="http://d", awake="D", tag="rkst",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, flip_min_work_tokens=1)


class TestFrontGroupHealthAndErrors(CustomTestCase):
    def test_verdict_words(self):
        v = fsi.group_health_verdict
        self.assertEqual([v(True, True, 0, False), v(False, True, 0, False), v(False, True, 2, False),
                          v(True, True, 0, True), v(False, False, 3, False)],
                         ["ok", "busy", "failing", "held", "dead"])

    def test_one_event_per_change_not_per_poll(self):
        sd = state_file.init(tempfile.mkdtemp(prefix="rkst-gh-"), "rkst-boot-20260929T130000Z-beef", "boot", {})
        f = _front()
        g = f.groups["D"]
        seq = [(True, True, 0), (True, True, 0), (False, True, 1), (False, True, 2), (True, True, 0)]
        with mock.patch.dict(os.environ, {"WEG2_STATE_DIR": sd}):
            for ok, alive, streak in seq:
                g.health_facts = _fh.GroupFacts(ok, alive, streak, None, time.time())
                f._ipc_group_health_observe(g)
            deadline = time.time() + 3
            while time.time() < deadline and len([e for e in state_file.events(sd) if e["type"] == "group_health"]) < 3:
                time.sleep(0.02)
        ev = [e["data"] for e in state_file.events(sd) if e["type"] == "group_health"]
        self.assertEqual([(e["prev"], e["verdict"]) for e in ev],
                         [(None, "ok"), ("ok", "failing"), ("failing", "ok")])
        self.assertEqual(ev[1]["streak"], 1)

    def test_front_fields_carry_errors_once_the_writer_runs(self):
        sd = state_file.init(tempfile.mkdtemp(prefix="rkst-er-"), "rkst-boot-20260929T130001Z-beef", "boot", {})
        f = _front()

        async def body():
            t = asyncio.create_task(f.ipc_front_writer(period_s=0.01))
            await asyncio.sleep(0.05)
            logging.getLogger("sglang.srt.weg2.front").error("W-test front error")
            await asyncio.sleep(0.1)
            t.cancel()

        with mock.patch.dict(os.environ, {"WEG2_STATE_DIR": sd}):
            asyncio.run(body())
        errs = f._ipc_front_fields()["errors"]
        self.assertGreaterEqual(errs["n"], 1)
        self.assertIn("W-test front error", [e["text"] for e in errs["last"]])


if __name__ == "__main__":
    unittest.main()
