# SPDX-License-Identifier: Apache-2.0
"""FH: the front's /health answers from the health poller's facts (27B rc12k27 b1).

b1 (dkr27breleasedraftbar1w109270932): PP1 died 09:45:58 and was HELD (#1223 DEBUG-HOLD, dump
written, process alive); the group's HTTP /health (tokenizer manager, never asks a scheduler) said
200 until the scheduler watchdog's SIGQUIT at 09:53:22, so the first PDFLIP-HEALTH failure was
09:53:47 and W17 09:54:27; the front's /health said 200 for 8.5 min.

Drives the real ``Front.health_poll_once`` / ``handle_health`` / ``group_dead_should_stop`` on a
front built with ``object.__new__``; the switch-off cases are the metal behaviour.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import front as front_mod  # noqa: E402
from flliper.srt.pdflip import front_health as FH  # noqa: E402

P_SID, D_SID = 4711, 4712
HELD_PID = 458
EXC = "PPWidthDivergenceRefused: #1233 W27 PP WIDTH DIVERGENCE REFUSED: received hidden_states with 1024 row(s)"


def _env(on: bool):
    return mock.patch.dict(os.environ, {FH.ENV: "1" if on else "0"})


def _write_dump(d, pid=HELD_PID, tag="rc12gb1092709", mtime=None):
    path = os.path.join(d, f"{tag}_rank1_pid{pid}_port5001_20260927T094559Z.txt")
    with open(path, "w") as fh:
        fh.write(f"#1223 DEBUG-HOLD rank=1\nutc: x\npid: {pid}\nexception: {EXC}\n\n=== TRACEBACK ===\n")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _sessions(mapping):
    return lambda pid: mapping.get(int(pid))


def _front(state="serving"):
    f = object.__new__(front_mod.Front)
    f.groups = {"P": front_mod.Group("P", "http://p", P_SID), "D": front_mod.Group("D", "http://d", D_SID)}
    f.state, f.awake, f.epoch, f.stop = state, "P", 11, None
    f.t0 = time.time() - 600
    f.counters = collections.Counter()
    f.queue = collections.deque()
    f._ready_for_d = collections.deque()
    f._batch_gate = asyncio.Event()
    return f


def _http(status):
    """_probe_group_health stand-in: {group name: ok}."""
    async def probe(self, g, timeout_s):
        return status[g.name]
    return probe


class _Resp200:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session200:
    """aiohttp session stand-in for the old live probe: every group says 200 (b1)."""

    def get(self, url, timeout=None):
        return _Resp200()


def _run(coro):
    return asyncio.run(coro)


class FindHold(unittest.TestCase):
    def test_a_dump_of_a_live_process_in_the_session_is_the_hold(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_dump(d)
            h = FH.find_hold(P_SID, time.time() - 60, [d], _sessions({HELD_PID: P_SID}))
        self.assertEqual(h, FH.HoldFact(HELD_PID, path, EXC))

    def test_another_group_or_a_dead_pid_is_not_this_groups_hold(self):
        with tempfile.TemporaryDirectory() as d:
            _write_dump(d)
            self.assertIsNone(FH.find_hold(D_SID, time.time() - 60, [d], _sessions({HELD_PID: P_SID})))
            self.assertIsNone(FH.find_hold(P_SID, time.time() - 60, [d], _sessions({})))

    def test_a_dump_older_than_the_front_is_another_boots(self):
        with tempfile.TemporaryDirectory() as d:
            _write_dump(d, mtime=time.time() - 3600)
            self.assertIsNone(FH.find_hold(P_SID, time.time() - 60, [d], _sessions({HELD_PID: P_SID})))

    def test_no_session_no_lookup(self):
        with tempfile.TemporaryDirectory() as d:
            _write_dump(d)
            self.assertIsNone(FH.find_hold(0, 0, [d], _sessions({HELD_PID: 0})))

    def test_pid_session_reads_proc(self):
        self.assertEqual(FH.pid_session(os.getpid()), os.getsid(0))
        self.assertIsNone(FH.pid_session(2 ** 30))

    def test_hold_dirs(self):
        self.assertEqual(FH.hold_dirs({"FLLIPER_DEBUG_HOLD_DIR": "/var/lib/htsglang/evidence/debug_hold"}),
                         ["/var/lib/htsglang/evidence/debug_hold", "/spinning/gpu-arb/debug_hold"])


class Verdicts(unittest.TestCase):
    H = FH.HoldFact(HELD_PID, "/x", EXC)

    def f(self, ok=True, alive=True, streak=0, hold=None):
        return FH.GroupFacts(ok, alive, streak, hold, time.time())

    def test_truth_table(self):
        U = FH.unhealthy_reason
        self.assertIsNone(U(self.f(), "serving"))
        self.assertIn("debug-hold", U(self.f(hold=self.H), "serving"))
        self.assertIn("debug-hold", U(self.f(hold=self.H), "flipping"))      # dead in every state
        self.assertEqual(U(self.f(alive=False, streak=1), "flipping"), "process_alive=False")
        self.assertIsNone(U(self.f(ok=False, streak=1), "serving"))           # one miss is transport
        self.assertIn("streak=2", U(self.f(ok=False, streak=2), "serving"))
        self.assertIsNone(U(self.f(ok=False, streak=9), "flipping"))          # asleep behind a leg

    def test_the_w17_gate_takes_the_hold_in_every_state(self):
        f = object.__new__(front_mod.Front)
        for state in ("serving", "flipping", "idle"):
            self.assertTrue(f.group_dead_should_stop(state=state, ok=True, alive=True, streak=1, hold=True))
        self.assertFalse(f.group_dead_should_stop(state="flipping", ok=False, alive=True, streak=5))

    def test_intervals(self):
        with _env(True):
            self.assertEqual((FH.poll_interval_s(), FH.probe_timeout_s()), (5.0, 8.0))
        with _env(False):
            self.assertEqual((FH.poll_interval_s(), FH.probe_timeout_s()), (15.0, 25.0))


class TheB1HoldIsSeenAtTheFirstPoll(unittest.TestCase):
    def _poll(self, front, http, sessions, d, alive=None):
        alive = alive or {P_SID: True, D_SID: True}
        with mock.patch.object(front_mod.Front, "_probe_group_health", _http(http)), \
             mock.patch.object(front_mod, "_sid_alive", lambda sid: alive.get(sid, True)), \
             mock.patch.object(FH, "pid_session", _sessions(sessions)):
            _run(front.health_poll_once([d]))

    def _health(self, front, http=None):
        with mock.patch.object(front_mod.Front, "_probe_group_health", _http(http or {"P": True, "D": True})):
            r = _run(front.handle_health(None))
        return r.status, json.loads(r.body)

    def test_b1_held_rank_http_200_session_alive_is_503_and_w17(self):
        with _env(True), tempfile.TemporaryDirectory() as d:
            front = _front()
            _write_dump(d)
            self._poll(front, {"P": True, "D": True}, {HELD_PID: P_SID}, d)
            status, body = self._health(front)
        self.assertEqual(front.state, "STOP")
        self.assertIn("W17 PdFlipGroupDead", str(front.stop))
        self.assertIn("DEBUG-HOLD pid=458", str(front.stop))
        self.assertEqual(status, 503)
        self.assertIn("debug-hold pid=458", body["unhealthy"]["P"])
        self.assertEqual(body["facts"]["P"]["hold"]["exception"], EXC)
        self.assertNotIn("D", body["unhealthy"])

    def test_the_metal_behaviour_switch_off_answers_200(self):
        with _env(False), tempfile.TemporaryDirectory() as d:
            front = _front()
            front.session = _Session200()
            _write_dump(d)
            status, body = self._health(front)          # old live probe: both 200
        self.assertEqual(status, 200)
        self.assertEqual((body["P"], body["D"]), (200, 200))
        self.assertIsNone(front.groups["P"].health_facts)

    def test_http_streak_two_is_503_one_is_not(self):
        with _env(True), tempfile.TemporaryDirectory() as d:
            front = _front()
            self._poll(front, {"P": False, "D": True}, {}, d)
            self.assertEqual(self._health(front)[0], 200)
            self._poll(front, {"P": False, "D": True}, {}, d)
            status, body = self._health(front)
        self.assertEqual(status, 503)
        self.assertEqual(body["unhealthy"], {"P": "http_ok=False streak=2"})

    def test_a_dead_session_is_503_at_once(self):
        with _env(True), tempfile.TemporaryDirectory() as d:
            front = _front()
            self._poll(front, {"P": False, "D": True}, {}, d, alive={P_SID: False, D_SID: True})
            status, body = self._health(front)
        self.assertEqual(status, 503)
        self.assertEqual(body["unhealthy"], {"P": "process_alive=False"})

    def test_flip_sleeping_group_is_not_503_the_dead_one_is(self):
        with _env(True), tempfile.TemporaryDirectory() as d:
            front = _front("flipping")
            for _ in range(4):  # D silent behind its flip leg, alive
                self._poll(front, {"P": True, "D": False}, {}, d)
            status, body = self._health(front)
            self.assertEqual((status, front.state), (200, "flipping"))
            self.assertEqual(body["unhealthy"], {})
            _write_dump(d)
            self._poll(front, {"P": True, "D": False}, {HELD_PID: P_SID}, d)
            status, body = self._health(front)
        self.assertEqual(status, 503)
        self.assertIn("debug-hold pid=458", body["unhealthy"]["P"])

    def test_no_or_stale_facts_probe_live(self):
        with _env(True):
            front = _front()
            status, body = self._health(front, {"P": True, "D": False})
            self.assertEqual(status, 503)
            self.assertEqual(body["facts"]["D"]["source"], "live")
            front.groups["P"].health_facts = FH.GroupFacts(True, True, 0, None, time.time() - 3600)
            front.groups["D"].health_facts = FH.GroupFacts(True, True, 0, None, time.time())
            status, body = self._health(front, {"P": True, "D": False})
        self.assertEqual(status, 200)
        self.assertEqual((body["facts"]["P"]["source"], body["facts"]["D"]["source"]), ("live", "poller"))

    def test_probes_run_concurrently(self):
        seen = []

        async def slow(self, g, timeout_s):
            seen.append(("start", g.name))
            await asyncio.sleep(0.2)
            seen.append(("end", g.name))
            return True

        with _env(True), tempfile.TemporaryDirectory() as d:
            front = _front()
            with mock.patch.object(front_mod.Front, "_probe_group_health", slow), \
                 mock.patch.object(front_mod, "_sid_alive", lambda sid: True):
                t = time.time()
                _run(front.health_poll_once([d]))
                dt = time.time() - t
        self.assertLess(dt, 0.35)
        self.assertEqual([k for k, _ in seen[:2]], ["start", "start"])

    def test_the_poller_is_armed_with_the_new_cadence(self):
        calls = []

        async def once(self, dirs):
            calls.append(dirs)
            raise asyncio.CancelledError

        async def fake_sleep(s):
            calls.append(s)

        with _env(True):
            front = _front()
            with mock.patch.object(front_mod.Front, "health_poll_once", once), \
                 mock.patch.object(front_mod.asyncio, "sleep", fake_sleep):
                with self.assertRaises(asyncio.CancelledError):
                    _run(front.health_poller())
        self.assertEqual(calls[0], FH.POLL_S)


if __name__ == "__main__":
    unittest.main()
