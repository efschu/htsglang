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
"""fnFL2 H111b: the sleeper's pair lanes one tag ahead of the pause.

THE MEASUREMENT (x177/x178/h91v1, 15 P->D flips, H111 legsplit.py): PP0's
deposit chain 1.55-1.76 s against a 1.25 s copy-engine floor (one D2H engine,
H22: 9.34 GB/s for 11.69 GB). Above the floor: the per-tag lockstep (the
faster lane idles until the slower one ends the tag; x177 97k weights_1:
p0 181 ms, p1 63 ms -- sum over the flip 42-156 ms) and pause + credit + loop
step of every tag between two tags' copies (~55 ms).

WHAT IS PINNED HERE (``weg2/deposit_lookahead.LaneLookahead`` real, the
deposits modelled by functions with their own sleeps and an event log):

* THE LAW PER TAG: every lane of tag j has deposited before the loop pauses
  tag j (xchg-lane-ordnung deposit -> pause -> credit);
* THE BOUND: a lane never starts tag j before the loop paused tag j-1-ahead;
* THE GAIN: with lanes that alternate being slow, the lookahead chain is
  shorter than the lockstep chain of the same deposits;
* refusals travel to the loop thread by name; a stuck lane is refused by
  name at its budget; the worker takes the rank's device first;
* switch off / other group: the loop's scope yields None (the lockstep).
"""

import threading
import time
import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.weg2 import deposit_lookahead as dl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

TAGS = [f"weights_{i}" for i in range(6)]


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = []  # (kind, tag_index, lane, t)

    def add(self, kind, j, lane=None):
        with self.lock:
            self.events.append((kind, j, lane, time.perf_counter()))


def run_loop(look, rec, n_tags, pause_s=0.0, diag_s=0.0):
    """The sleep loop's shape under H111b: diag deposit, join the lanes of the
    tag, pause, credit, advance."""
    look.start()
    try:
        for j in range(n_tags):
            if diag_s:
                time.sleep(diag_s)
            look.join_tag(j)
            rec.add("pause", j)
            if pause_s:
                time.sleep(pause_s)
            look.advance(j)
    finally:
        look.close()


def run_lockstep(deposit, lanes, rec, n_tags, pause_s=0.0, diag_s=0.0):
    """The pre-H111b form: every tag's lanes in parallel, joined, then pause."""
    for j in range(n_tags):
        if diag_s:
            time.sleep(diag_s)
        ths = [threading.Thread(target=deposit, args=(TAGS[j], lane)) for lane in lanes]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        rec.add("pause", j)
        if pause_s:
            time.sleep(pause_s)


def alternating(rec, slow=0.03, fast=0.005):
    """p0 slow on even tags, p1 slow on odd ones (the x177 w0/w1 shape)."""

    def deposit(tag, lane):
        j = TAGS.index(tag)
        rec.add("start", j, lane)
        slow_lane = 0 if j % 2 == 0 else 1
        time.sleep(slow if lane == slow_lane else fast)
        rec.add("end", j, lane)

    return deposit


class TheLaw(CustomTestCase):
    def test_every_lane_deposits_before_its_tags_pause(self):
        rec = Recorder()
        look = dl.LaneLookahead(TAGS, [0, 1], alternating(rec), ahead=1, budget_s=10)
        run_loop(look, rec, len(TAGS))
        pause_t = {j: t for k, j, _l, t in rec.events if k == "pause"}
        for k, j, lane, t in rec.events:
            if k == "end":
                self.assertLess(t, pause_t[j], f"lane p{lane} ended tag {j} after its pause")
        self.assertEqual(len(pause_t), len(TAGS))

    def test_no_lane_runs_more_than_ahead_beyond_the_pause(self):
        for ahead in (0, 1, 2):
            rec = Recorder()
            look = dl.LaneLookahead(TAGS, [0, 1], alternating(rec), ahead=ahead, budget_s=10)
            run_loop(look, rec, len(TAGS), pause_s=0.002)
            pause_t = {j: t for k, j, _l, t in rec.events if k == "pause"}
            for k, j, lane, t in rec.events:
                if k == "start" and j - 1 - ahead >= 0:
                    self.assertGreater(
                        t, pause_t[j - 1 - ahead],
                        f"ahead={ahead}: p{lane} started tag {j} before tag "
                        f"{j - 1 - ahead} was paused")

    def test_per_lane_tag_order_is_the_loop_order(self):
        rec = Recorder()
        look = dl.LaneLookahead(TAGS, [0, 1], alternating(rec), ahead=1, budget_s=10)
        run_loop(look, rec, len(TAGS))
        for lane in (0, 1):
            order = [j for k, j, l, _t in rec.events if k == "start" and l == lane]
            self.assertEqual(order, list(range(len(TAGS))))


class TheGain(CustomTestCase):
    def test_lookahead_chain_shorter_than_lockstep(self):
        """Lanes alternate being slow: the lockstep pays max(p0, p1) per tag,
        the lookahead approaches max(sum p0, sum p1)."""
        rec_a, rec_b = Recorder(), Recorder()
        t0 = time.perf_counter()
        run_lockstep(alternating(rec_a), [0, 1], rec_a, len(TAGS), pause_s=0.004, diag_s=0.002)
        lockstep = time.perf_counter() - t0
        look = dl.LaneLookahead(TAGS, [0, 1], alternating(rec_b), ahead=1, budget_s=10)
        t0 = time.perf_counter()
        run_loop(look, rec_b, len(TAGS), pause_s=0.004, diag_s=0.002)
        ahead = time.perf_counter() - t0
        # model: lockstep ~6 x (30 + 6) = 216 ms, lookahead ~ 3 x 35 + tail
        self.assertLess(ahead, lockstep * 0.85, f"lookahead {ahead:.3f}s vs lockstep {lockstep:.3f}s")


class Refusals(CustomTestCase):
    def test_lane_refusal_reaches_the_loop_at_its_tag(self):
        class W68(RuntimeError):
            pass

        def deposit(tag, lane):
            if lane == 1 and tag == TAGS[2]:
                raise W68("W68 Weg2XchgPlanDisagree: lane p1 tag weights_2")

        look = dl.LaneLookahead(TAGS, [0, 1], deposit, ahead=1, budget_s=10)
        look.start()
        try:
            look.join_tag(0)
            look.advance(0)
            look.join_tag(1)
            look.advance(1)
            with self.assertRaises(W68):
                look.join_tag(2)
        finally:
            look.close()

    def test_stuck_lane_is_refused_by_name(self):
        gate = threading.Event()

        def deposit(tag, lane):
            if lane == 0:
                gate.wait(2.0)

        look = dl.LaneLookahead(TAGS[:2], [0, 1], deposit, ahead=1, budget_s=0.2)
        look.start()
        try:
            with self.assertRaises(dl.LaneLookaheadError) as cm:
                look.join_tag(0)
            self.assertIn("p0", str(cm.exception))
            self.assertIn(TAGS[0], str(cm.exception))
        finally:
            gate.set()
            look.close()

    def test_worker_takes_the_device_before_its_first_deposit(self):
        seen = []
        local = threading.local()

        def init():
            local.dev = 1

        def deposit(tag, lane):
            seen.append(getattr(local, "dev", None))

        look = dl.LaneLookahead(TAGS[:2], [0, 1], deposit, ahead=1, budget_s=10,
                                thread_init=init)
        look.start()
        try:
            look.join_tag(0)
            look.advance(0)
            look.join_tag(1)
        finally:
            look.close()
        self.assertEqual(seen, [1] * 4)


class Switch(CustomTestCase):
    def test_off_by_default(self):
        self.assertFalse(dl.lookahead_on())
        self.assertFalse(dl.lookahead_on("P"))

    def test_on_only_for_the_named_groups(self):
        with envs.SGLANG_WEG2_DEPOSIT_LANE_LOOKAHEAD.override(True):
            self.assertTrue(dl.lookahead_on("P"))
            self.assertFalse(dl.lookahead_on("D"))
            with envs.SGLANG_WEG2_DEPOSIT_LANE_LOOKAHEAD_GROUPS.override("P,D"):
                self.assertTrue(dl.lookahead_on("D"))

    def _scope(self, group):
        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager as M,
        )

        calls = []
        stub = SimpleNamespace(
            _weg2_group_name=lambda: group,
            _weg2_xchg_deposit_pair_lanes=lambda fi: calls.append(fi) or [0, 1],
        )
        scope = M._weg2_h111b_scope.__get__(stub)
        with scope(SimpleNamespace(epoch=None), TAGS) as look:
            return look, calls

    def test_scope_off_yields_none_and_reads_no_plan(self):
        look, calls = self._scope("P")
        self.assertIsNone(look)
        self.assertEqual(calls, [], "switch off must not even read the plan")

    def test_scope_on_for_group_d_yields_none(self):
        with envs.SGLANG_WEG2_DEPOSIT_LANE_LOOKAHEAD.override(True):
            look, calls = self._scope("D")
        self.assertIsNone(look)
        self.assertEqual(calls, [])


register_cpu_ci(est_time=5, suite="base-a-test-cpu")

if __name__ == "__main__":
    unittest.main()
