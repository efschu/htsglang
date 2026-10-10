# SPDX-License-Identifier: Apache-2.0
"""int24e: the W17 Weg2GroupDead verdict lines carry the host-memory fact (log text only).

NF deaths 10.10. 06:09Z and 13:14Z were host-RAM exhaustion -> swap storm -> ~50 s without CPU ->
W17. The verdict line now explains itself ("host stall" vs "group hangs"): `` mem: <host_mem_line()>``
(PSI memory some/full avg10, swap used, swap/compaction deltas) as a suffix, the same format the
WEG2-HEALTH line prints. NO switch, NO behaviour change: the stop fires exactly as before, and a
failure of the measurement leaves the suffix off, never the stop.

Drives the real ``Front.health_poll_once`` (hold site and dead-process site) and
``Front._health_poller_old`` on a front built with ``object.__new__`` (pattern of
test_weg2_front_health_facts_b1_0927).
"""

from __future__ import annotations

import asyncio
import collections
import os
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2 import front_health as FH  # noqa: E402

P_SID, D_SID = 4711, 4712
HELD_PID = 458
EXC = "PPWidthDivergenceRefused: x"
FAKE_MEM = (
    "psi_mem_some_avg10=41.50 psi_mem_full_avg10=33.20 swap_used_mib=7001 "
    "pswpin_d=100 pswpout_d=900 compact_stall_d=3 pgmajfault_d=40"
)
TAIL = "(a 200 alone is a transport fact)"


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
    async def probe(self, g, timeout_s):
        return status[g.name]
    return probe


def _write_dump(d):
    path = os.path.join(d, f"rc12gb1092709_rank1_pid{HELD_PID}_port5001_20260927T094559Z.txt")
    with open(path, "w") as fh:
        fh.write(f"#1223 DEBUG-HOLD rank=1\nutc: x\npid: {HELD_PID}\nexception: {EXC}\n\n=== TRACEBACK ===\n")
    return path


def _poll(front, http, d, alive, sessions=None, n=1):
    with mock.patch.object(front_mod.Front, "_probe_group_health", _http(http)), \
         mock.patch.object(front_mod, "_sid_alive", lambda sid: alive.get(sid, True)), \
         mock.patch.object(FH, "pid_session", lambda pid: (sessions or {}).get(int(pid))), \
         mock.patch.dict(os.environ, {FH.ENV: "1"}):
        for _ in range(n):
            asyncio.run(front.health_poll_once([d]))


def _dead_process_stop(mem_patch, n=2):
    """P's process is gone and /health silent: W17 fires at streak 2. Returns (front, detail)."""
    with tempfile.TemporaryDirectory() as d:
        front = _front()
        with mem_patch:
            _poll(front, {"P": False, "D": True}, d, {P_SID: False, D_SID: True}, n=n)
    return front, str(front.stop)


def _hold_stop(mem_patch):
    with tempfile.TemporaryDirectory() as d:
        front = _front()
        _write_dump(d)
        with mem_patch:
            _poll(front, {"P": True, "D": True}, d, {}, sessions={HELD_PID: P_SID})
    return front, str(front.stop)


def _mem(value=FAKE_MEM):
    return mock.patch.object(FH, "host_mem_line", lambda: value)


def _mem_raises():
    return mock.patch.object(FH, "host_mem_line", mock.Mock(side_effect=RuntimeError("psi gone")))


class TheVerdictLineExplainsItself(unittest.TestCase):
    def test_dead_process_site_carries_mem_with_psi_and_swap_fields(self):
        front, detail = _dead_process_stop(_mem())
        self.assertEqual(front.state, "STOP")
        self.assertIn("W17 Weg2GroupDead", detail)
        self.assertIn(f"group P: /health failed 2x and process_alive=False {TAIL}", detail)
        self.assertTrue(detail.endswith(" mem: " + FAKE_MEM), detail)
        for field in ("psi_mem_some_avg10=", "psi_mem_full_avg10=", "swap_used_mib=", "pswpout_d="):
            self.assertIn(field, detail)

    def test_hold_site_carries_mem(self):
        front, detail = _hold_stop(_mem())
        self.assertEqual(front.state, "STOP")
        self.assertIn("W17 Weg2GroupDead", detail)
        self.assertIn("DEBUG-HOLD pid=458", detail)
        self.assertTrue(detail.endswith(" mem: " + FAKE_MEM), detail)

    def test_real_host_mem_line_reaches_the_verdict(self):
        # no patch: the real instrument (reads /proc here; "unavailable(..)" is also a fact, never absent)
        front, detail = _dead_process_stop(mock.patch.object(FH, "_HOST_MEM", None))
        self.assertEqual(front.state, "STOP")
        self.assertIn(" mem: ", detail)
        self.assertTrue("psi_mem_some_avg10=" in detail or "unavailable(" in detail, detail)

    def test_format_is_identical_to_the_weg2_health_line(self):
        # ONE grep reads both: the text after ` mem: ` is the same string on WEG2-HEALTH and on the verdict.
        with tempfile.TemporaryDirectory() as d, _mem(), self.assertLogs("weg2.front", level="WARNING") as cm:
            front = _front()
            _poll(front, {"P": False, "D": True}, d, {P_SID: False, D_SID: True}, n=2)
        health = [r for r in cm.output if "WEG2-HEALTH group=P" in r]
        self.assertTrue(health)
        self.assertTrue(all(r.endswith(" mem: " + FAKE_MEM) for r in health), health)
        self.assertTrue(str(front.stop).endswith(" mem: " + FAKE_MEM))

    def test_old_poller_site_carries_mem(self):
        front = _front()

        class _Fail:
            def get(self, url, timeout=None):
                raise OSError("no route")

        front.session = _Fail()
        calls = {"n": 0}

        async def sleep(_s):
            calls["n"] += 1
            if calls["n"] > 2:
                raise asyncio.CancelledError

        async def run():
            with mock.patch.object(front_mod.asyncio, "sleep", sleep):
                try:
                    await front._health_poller_old()
                except asyncio.CancelledError:
                    pass

        with _mem(), mock.patch.object(front_mod, "_sid_alive", lambda sid: False):
            asyncio.run(run())
        self.assertEqual(front.state, "STOP")
        self.assertIn("W17 Weg2GroupDead", str(front.stop))
        self.assertTrue(str(front.stop).endswith(" mem: " + FAKE_MEM), str(front.stop))

    def test_mutant_without_the_suffix_is_red(self):
        # The assertion above must detect a removed suffix: with the helper neutered the same check fails.
        with mock.patch.object(front_mod, "_w17_mem_suffix", lambda mem=None: ""):
            _front_, detail = _dead_process_stop(_mem())
        self.assertFalse(detail.endswith(" mem: " + FAKE_MEM))
        self.assertNotIn("psi_mem_some_avg10=", detail)
        with mock.patch.object(front_mod, "_w17_mem_suffix", lambda mem=None: ""):
            _front_, detail = _hold_stop(_mem())
        self.assertNotIn("psi_mem_some_avg10=", detail)


class AFailedMeasurementNeverPreventsTheStop(unittest.TestCase):
    def test_dead_process_stop_still_happens_without_suffix(self):
        front, detail = _dead_process_stop(_mem_raises())
        self.assertEqual(front.state, "STOP")
        self.assertTrue(detail.endswith(TAIL), detail)
        self.assertNotIn(" mem:", detail)

    def test_hold_stop_still_happens_without_suffix(self):
        front, detail = _hold_stop(_mem_raises())
        self.assertEqual(front.state, "STOP")
        self.assertIn("say nothing about it", detail)
        self.assertNotIn(" mem:", detail)

    def test_the_suffix_helper_itself_never_raises(self):
        with _mem_raises():
            self.assertEqual(front_mod._w17_mem_suffix(), "")        # measures now -> fails -> empty
        self.assertEqual(front_mod._w17_mem_suffix(""), "")           # empty reading -> no suffix
        self.assertEqual(front_mod._w17_mem_suffix("x=1"), " mem: x=1")
        with _mem("measured now"):
            self.assertEqual(front_mod._w17_mem_suffix(), " mem: measured now")

    def test_old_poller_stop_still_happens_when_the_measurement_raises(self):
        front = _front()

        class _Fail:
            def get(self, url, timeout=None):
                raise OSError("no route")

        front.session = _Fail()
        calls = {"n": 0}

        async def sleep(_s):
            calls["n"] += 1
            if calls["n"] > 2:
                raise asyncio.CancelledError

        async def run():
            with mock.patch.object(front_mod.asyncio, "sleep", sleep):
                try:
                    await front._health_poller_old()
                except asyncio.CancelledError:
                    pass

        with _mem_raises(), mock.patch.object(front_mod, "_sid_alive", lambda sid: False):
            asyncio.run(run())
        self.assertEqual(front.state, "STOP")
        self.assertNotIn(" mem:", str(front.stop))


class NoBehaviourChange(unittest.TestCase):
    def test_a_healthy_poll_logs_and_stops_nothing(self):
        with tempfile.TemporaryDirectory() as d, _mem():
            front = _front()
            _poll(front, {"P": True, "D": True}, d, {P_SID: True, D_SID: True}, n=3)
        self.assertEqual(front.state, "serving")
        self.assertIsNone(front.stop)

    def test_one_miss_is_still_no_stop(self):
        front, _ = None, None
        with tempfile.TemporaryDirectory() as d, _mem():
            front = _front()
            _poll(front, {"P": False, "D": True}, d, {P_SID: False, D_SID: True}, n=1)
        self.assertEqual(front.state, "serving")

    def test_stop_switch_off_still_suppresses(self):
        with tempfile.TemporaryDirectory() as d, _mem(), \
             mock.patch.dict(os.environ, {"SGLANG_WEG2_ENABLE_GROUP_DEAD_STOP": "0"}):
            front = _front()
            _poll(front, {"P": False, "D": True}, d, {P_SID: False, D_SID: True}, n=3)
        self.assertEqual(front.state, "serving")


if __name__ == "__main__":
    unittest.main()
