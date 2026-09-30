# SPDX-License-Identifier: Apache-2.0
"""FP (NF rc12p 14:13:39): a /health that fails while the group runs a long forward (91k extend)
is not a dead group. The front reads a per-rank 32-byte mmap beacon (forward_ct, t_start, t_done);
progress -> PDFLIP-HEALTH-BUSY, no streak. Hold / process_alive=False stay fatal at once."""

from __future__ import annotations

import asyncio
import collections
import json
import os
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import front as front_mod  # noqa: E402
from flliper.srt.pdflip import front_health as FH  # noqa: E402
from flliper.srt.pdflip import progress_beacon as FP  # noqa: E402

P_SID, D_SID = 4711, 4712


def _front(tmp):
    f = object.__new__(front_mod.Front)
    f.groups = {"P": front_mod.Group("P", "http://p", P_SID), "D": front_mod.Group("D", "http://d", D_SID)}
    f.state, f.awake, f.epoch, f.stop, f.tag = "serving", "D", 3, None, "t"
    f.t0 = time.time() - 600
    f.counters = collections.Counter()
    f.queue = collections.deque()
    f._ready_for_d = collections.deque()
    f._batch_gate = asyncio.Event()
    return f


def _http(status):
    async def probe(self, g, timeout_s):
        return status[g.name]
    return probe


class Beacon(unittest.TestCase):
    def test_writer_and_reader_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_HICACHE_ARENA_DIR": d,
                                          FP.ENV: "1"}):
            w = FP._Writer()
            w.beat(7, True)
            cur = FP.read_group(os.path.join(d, "progress"), "D", 99, lambda pid: 99)
            (ct, ts, td), = cur.values()
            self.assertEqual(ct, 7)
            self.assertGreater(ts, td)
            self.assertIn("in forward 7", FP.progress({}, cur))
            w.beat(7, False)
            cur2 = FP.read_group(os.path.join(d, "progress"), "D", 99, lambda pid: 99)
            self.assertIsNone(FP.progress(cur2, cur2), "idle and unchanged: no progress")
            w.beat(8, True)
            w.beat(8, False)
            cur3 = FP.read_group(os.path.join(d, "progress"), "D", 99, lambda pid: 99)
            self.assertIn("forward_ct 7->8", FP.progress(cur2, cur3))
            self.assertEqual(FP.read_group(os.path.join(d, "progress"), "D", 99, lambda pid: 1), {},
                             "another session's file is not this group's")

    def test_switch_off_writes_nothing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_HICACHE_ARENA_DIR": d,
                                          FP.ENV: "0"}):
            w = FP._Writer()
            w.beat(1, True)
            self.assertFalse(os.path.exists(os.path.join(d, "progress")))

    def test_a_stuck_forward_is_not_progress(self):
        now = time.time_ns()
        cur = {1: (5, now - int(200e9), now - int(300e9))}
        self.assertIsNone(FP.progress(cur, cur, now_ns=now))


class TheLongExtendIsNotADeadGroup(unittest.TestCase):
    def _poll(self, f, http, beacon, alive=None):
        alive = alive or {P_SID: True, D_SID: True}
        with mock.patch.object(front_mod.Front, "_probe_group_health", _http(http)), \
             mock.patch.object(front_mod, "_sid_alive", lambda sid: alive.get(sid, True)), \
             mock.patch.object(FP, "read_group", lambda d, g, sid, session_of=None: beacon.get(g, {})), \
             mock.patch.object(FP, "beacon_dir", lambda tag="", env=None: "/x"):
            asyncio.run(f.health_poll_once([]))

    def _health(self, f):
        with mock.patch.object(front_mod.Front, "_probe_group_health", _http({"P": True, "D": True})):
            r = asyncio.run(f.handle_health(None))
        return r.status, json.loads(r.body)

    def test_91k_extend_health_slow_twice_progress_runs_200(self):
        with mock.patch.dict(os.environ, {FP.ENV: "1", FH.ENV: "1"}):
            f = _front(None)
            t = time.time_ns()
            in_fwd = {"D": {11: (4120, t - int(20e9), t - int(21e9))}, "P": {}}
            for _ in range(3):  # three slow /health while D is 20+ s into one extend
                self._poll(f, {"P": True, "D": False}, in_fwd)
            status, body = self._health(f)
        self.assertEqual(status, 200, body)
        self.assertEqual(f.state, "serving")
        self.assertEqual(f.groups["D"].health_fail_streak, 0)
        self.assertGreaterEqual(f.counters["health_busy"], 3)

    def test_the_metal_behaviour_switch_off_is_503(self):
        with mock.patch.dict(os.environ, {FP.ENV: "0", FH.ENV: "1"}):
            f = _front(None)
            t = time.time_ns()
            in_fwd = {"D": {11: (4120, t - int(20e9), t - int(21e9))}}
            self._poll(f, {"P": True, "D": False}, in_fwd)
            self._poll(f, {"P": True, "D": False}, in_fwd)
            status, _ = self._health(f)
        self.assertEqual(status, 503)

    def test_no_progress_still_counts(self):
        with mock.patch.dict(os.environ, {FP.ENV: "1", FH.ENV: "1"}):
            f = _front(None)
            idle = {"D": {11: (4120, 1, 2)}}
            self._poll(f, {"P": True, "D": False}, idle)
            self._poll(f, {"P": True, "D": False}, idle)
            status, _ = self._health(f)
        self.assertEqual(status, 503)

    def test_a_dead_session_is_fatal_despite_a_fresh_beacon(self):
        with mock.patch.dict(os.environ, {FP.ENV: "1", FH.ENV: "1"}):
            f = _front(None)
            t = time.time_ns()
            in_fwd = {"D": {11: (4120, t - int(5e9), t - int(6e9))}}
            self._poll(f, {"P": True, "D": False}, in_fwd, alive={P_SID: True, D_SID: False})
            status, body = self._health(f)
        self.assertEqual(status, 503)
        self.assertEqual(body["unhealthy"], {"D": "process_alive=False"})

    def test_wiring(self):
        from flliper.srt.managers import scheduler as S

        src = open(S.__file__).read()
        i = src.index("self.forward_ct += 1\n        batch.forward_iter = self.forward_ct")
        self.assertIn("_pdflip_beacon.beat_start(self.forward_ct)", src[i:i + 400])
        j = src.index("def process_batch_result(")
        self.assertIn("_pdflip_beacon.beat_done(", src[j:j + 400])


if __name__ == "__main__":
    unittest.main()
