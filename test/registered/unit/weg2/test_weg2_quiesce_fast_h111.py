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
"""fnFL2 H111 (Tail-Buchhaltung): the P quiesce answers within two fast polls
of the last write-through, and a poll during a lap never spoils that lap.

THE MEASUREMENT (x177/x178/h91v1, all 15 P->D flips, H111 quiesce_gap.py): the
last write-through of P lands +19..+189 ms after P-end, /flush_cache answers
200 another 95..189 ms later. Two causes, both protocol, not work:

* the front sleeps 50 ms between polls, and PP0 needs one poll to stamp a lap
  and the next to read it;
* a poll that finds the lap still on the ring WANTS another one. PP0's
  pass-top harvests the landed lap and then stamps the wanted one in the same
  pass, so the next poll reads the harvested lap as
  ``#1268 IDLE-ROUND stale ... round id mismatch`` (8 of 15 flips, e.g.
  x177 P-Log 04:51:52 epoch=2, one more poll). At a poll FASTER than the lap
  this drops every lap -- the fast poll alone would livelock the quiesce into
  its 90 s W3 deadline.

THE FIX (switch SGLANG_WEG2_QUIESCE_FAST, default off = byte-identical): the
front polls every SGLANG_WEG2_QUIESCE_FAST_POLL_MS (10) and PP0's pending
answer does not re-want while its lap is outstanding.

WHAT IS REAL AND WHAT IS MODELLED: the P group is the H77 harness
(``test_weg2_idle_round_fresh_1268.build_p_group``) -- every real
``Scheduler._weg2_vote_*`` hook, the real ``group_idle_verdict``,
``flush_cache`` and flush wrapper; modelled are the wire, idleness and the
clock. The front half runs the real ``Front.quiesce`` against a scripted RPC.
"""

import asyncio
import importlib.util
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.weg2 import front as front_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

SCHED_LOGGER = "sglang.srt.managers.scheduler"


def _harness():
    sibling = Path(__file__).with_name("test_weg2_idle_round_fresh_1268.py")
    spec = importlib.util.spec_from_file_location("_h77_harness_h111", sibling)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _harness()


class _PCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.clock = H.FakeClock()
        patcher = mock.patch("time.monotonic", new=self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.g = H.build_p_group(self.clock)

    def pass_pp0(self):
        H.run_pass(self.g, 0)
        self.clock.advance(0.002)

    def lap_home(self):
        """PP1 and PP2 turn until every list PP0 forwarded is through (the
        followers consume one list per pass), so a stamped lap is home; PP0
        has not harvested yet (that is PP0's next pass)."""
        g = self.g
        while g.wire.inbox[1] or g.wire.inbox[2]:
            H.run_pass(g, 1)
            H.run_pass(g, 2)
        self.clock.advance(0.002)

    def fast_cycle(self):
        """A lap slower than the poll: poll, PP0 stamps, poll AGAIN while the
        lap is on the ring, the lap comes home, PP0 harvests. Returns the two
        poll answers."""
        a = H.poll(self.g)
        self.pass_pp0()
        b = H.poll(self.g)
        self.lap_home()
        self.pass_pp0()
        return a, b


class PendingDuringLap(_PCase):
    def _poll_during_lap(self):
        g = self.g
        self.assertFalse(H.poll(g), "first poll: no lap yet")
        self.pass_pp0()  # stamps the wanted lap
        self.assertIsNotNone(g.ranks[0]._weg2_vote_outstanding)
        self.assertFalse(g.ranks[0]._weg2_vote_wanted)
        self.assertFalse(H.poll(g), "lap still on the ring: pending")

    def test_off_poll_during_lap_still_wants_byte_identical(self):
        with envs.SGLANG_WEG2_QUIESCE_FAST.override(False):
            self._poll_during_lap()
            self.assertTrue(
                self.g.ranks[0]._weg2_vote_wanted,
                "switch off must keep the pre-H111 unconditional want",
            )

    def test_on_poll_during_lap_does_not_rewant(self):
        with envs.SGLANG_WEG2_QUIESCE_FAST.override(True):
            with self.assertLogs(SCHED_LOGGER, level="INFO") as logs:
                self._poll_during_lap()
            self.assertFalse(
                self.g.ranks[0]._weg2_vote_wanted,
                "a poll during the lap wanted a second lap (the stale-lap cause)",
            )
            self.assertTrue(
                any("H111 QUIESCE-FAST lap epoch=" in m for m in logs.output),
                "no metal marker for the suppressed re-want",
            )

    def test_on_first_poll_without_lap_still_wants(self):
        """The guard binds only while a lap is OUTSTANDING -- a poll with no
        lap on the ring must still ask for one, or the quiesce never starts."""
        with envs.SGLANG_WEG2_QUIESCE_FAST.override(True):
            self.assertFalse(H.poll(self.g))
            self.assertIsNone(getattr(self.g.ranks[0], "_weg2_vote_outstanding", None))
            self.assertTrue(self.g.ranks[0]._weg2_vote_wanted)


class FastPollAgainstSlowLap(_PCase):
    """The core: polls faster than the lap. Off: every harvested lap is
    stale at its read (the x177 epoch=2 shape, repeated) -- the fast poll
    WITHOUT the guard would ride the 90 s deadline into W3. On: 200 on the
    first poll after the lap came home."""

    CYCLES = 6

    def test_off_fast_polls_drop_every_lap(self):
        with envs.SGLANG_WEG2_QUIESCE_FAST.override(False):
            with self.assertLogs(H.VOTE_LOGGER, level="INFO") as logs:
                answers = []
                for _ in range(self.CYCLES):
                    answers.extend(self.fast_cycle())
            self.assertFalse(any(answers), "off: a fast poll must never reach 200 here")
            stale = [m for m in logs.output if "#1268 IDLE-ROUND stale" in m and "round id mismatch" in m]
            self.assertGreaterEqual(len(stale), self.CYCLES - 1, logs.output)
            self.assertEqual(self.g.resets.n, 0, "no flush may run on a stale lap")

    def test_on_fast_polls_reach_200_on_the_next_poll(self):
        with envs.SGLANG_WEG2_QUIESCE_FAST.override(True):
            with self.assertLogs(H.VOTE_LOGGER, level="INFO") as logs:
                first = self.fast_cycle()
                self.assertEqual(first, (False, False))
                ok = H.poll(self.g)
            self.assertTrue(ok, "on: the landed lap must answer the next poll")
            self.assertFalse(
                [m for m in logs.output if "#1268 IDLE-ROUND stale" in m],
                "on: no lap may be dropped as stale",
            )
            self.assertEqual(self.g.resets.n, 1, "exactly one group flush")

    def test_on_busy_rank_still_blocks(self):
        """Faster is not looser: a follower with an in-flight write-through
        keeps the group not idle, the drained group then reaches its 200."""
        g = self.g
        with envs.SGLANG_WEG2_QUIESCE_FAST.override(True):
            H.busy(g, [2], ["hicache_backup(4)"])
            self.fast_cycle()
            self.assertFalse(H.poll(g), "PP2 backs up: GROUP NOT IDLE")
            self.assertEqual(g.resets.n, 0)
            H.idle(g, [2])
            self.pass_pp0()  # stamps the lap the NOT-IDLE read wanted
            self.lap_home()
            self.pass_pp0()  # harvests it
            self.assertTrue(H.poll(g), "the drained group must reach its 200")
            self.assertEqual(g.resets.n, 1)


class FrontPoll(CustomTestCase):
    def test_interval_off_is_the_1455_value(self):
        with envs.SGLANG_WEG2_QUIESCE_FAST.override(False):
            self.assertEqual(front_mod.quiesce_poll_s(), (0.05, False))

    def test_interval_on_default_10ms_and_floor(self):
        with envs.SGLANG_WEG2_QUIESCE_FAST.override(True):
            self.assertEqual(front_mod.quiesce_poll_s(), (0.01, True))
            with envs.SGLANG_WEG2_QUIESCE_FAST_POLL_MS.override(0):
                self.assertEqual(front_mod.quiesce_poll_s(), (0.001, True))
            with envs.SGLANG_WEG2_QUIESCE_FAST_POLL_MS.override(20):
                self.assertEqual(front_mod.quiesce_poll_s(), (0.02, True))

    def _run_quiesce(self, fast: bool, answers):
        codes = list(answers)
        sleeps = []

        async def rpc(_g, path, _body, _timeout):
            assert path == "/flush_cache"
            return codes.pop(0), "body"

        async def fake_sleep(s):
            sleeps.append(s)

        fake = SimpleNamespace(rpc=rpc, _health_inflight={})
        g = SimpleNamespace(name="P")
        with envs.SGLANG_WEG2_QUIESCE_FAST.override(fast), mock.patch.object(
            front_mod.asyncio, "sleep", new=fake_sleep
        ):
            ok = asyncio.run(front_mod.Front.quiesce(fake, g))
        return ok, sleeps

    def test_quiesce_off_sleeps_50ms_no_new_line(self):
        with self.assertNoLogs(front_mod.logger, level="INFO"):
            ok, sleeps = self._run_quiesce(False, [400, 400, 200])
        self.assertEqual(ok, (True, "body"))
        self.assertEqual(sleeps, [0.05, 0.05])

    def test_quiesce_on_sleeps_fast_and_names_itself(self):
        with self.assertLogs(front_mod.logger, level="INFO") as logs:
            ok, sleeps = self._run_quiesce(True, [400, 400, 200])
        self.assertEqual(ok, (True, "body"))
        self.assertEqual(sleeps, [0.01, 0.01])
        self.assertTrue(any("WEG2-QUIESCE-FAST group=P polls=3" in m for m in logs.output))


register_cpu_ci(est_time=3, suite="base-a-test-cpu")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
