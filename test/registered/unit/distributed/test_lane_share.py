# Copyright 2026 SGLang Team
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
"""#274 slice D, S1: the online card-equivalent estimator.

Every test drives the meter with a SYNTHETIC counter trace whose true E is
known by construction, so a failure names the arithmetic rather than the rig.
The four traps the estimator exists to avoid each get their own falsifier:
mixed arms, missing floors, rung changes, and floor re-learning under load.
"""

from __future__ import annotations

import unittest

from sglang.srt.model_executor.lane_share import (
    ClassSample,
    LaneShareMeter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Trace:
    """A pair of monotone counters advanced at chosen rates."""

    def __init__(self, meter, keys=("serving", "lane0")):
        self.meter = meter
        self.t = 100.0
        self.counters = {k: {"prefill_tokens": 0.0, "decode_tokens": 0.0} for k in keys}

    def step(self, rates, *, seconds=1.0, rung="static"):
        """Advance ``seconds`` at ``rates`` = {key: (arm, work_per_second)}."""
        self.t += seconds
        for key, spec in rates.items():
            if spec is None:
                continue
            arm, rate = spec
            self.counters[key][arm] += rate * seconds
        samples = [ClassSample(k, dict(v)) for k, v in self.counters.items()]
        return self.meter.observe(self.t, samples, rung=rung)

    def run(self, rates, *, windows, **kw):
        out = []
        for _ in range(windows):
            win = self.step(rates, **kw)
            if win is not None:
                out.append(win)
        return out


class TestWindowing(CustomTestCase):
    def test_first_observation_only_anchors(self):
        m = LaneShareMeter()
        self.assertIsNone(
            m.observe(1.0, [ClassSample("serving", {"decode_tokens": 0})])
        )

    def test_a_window_shorter_than_window_s_stays_open(self):
        m = LaneShareMeter(window_s=1.0)
        m.observe(1.0, [ClassSample("serving", {"decode_tokens": 0})])
        self.assertIsNone(
            m.observe(1.4, [ClassSample("serving", {"decode_tokens": 40})])
        )
        self.assertIsNotNone(
            m.observe(2.1, [ClassSample("serving", {"decode_tokens": 80})])
        )

    def test_idle_windows_are_named_not_counted_as_solo(self):
        m = LaneShareMeter()
        tr = _Trace(m)
        tr.step({})
        wins = tr.run({}, windows=3)
        self.assertTrue(all(w.kind == "idle" for w in wins))
        self.assertEqual(m.counts["solo"], 0)
        self.assertEqual(m.floors(), {})


class TestFloorsAndShare(CustomTestCase):
    """The happy path: solo floors first, then a shared window."""

    def _meter(self):
        return LaneShareMeter(window_s=1.0, ema_s=1.0, floor_min_windows=3)

    def test_solo_windows_set_the_floor_and_shared_windows_divide_by_it(self):
        m = self._meter()
        tr = _Trace(m)
        tr.step({})
        tr.run({"serving": ("decode_tokens", 40.0)}, windows=4)
        tr.run({"lane0": ("decode_tokens", 25.0)}, windows=4)
        self.assertAlmostEqual(m.floors()["serving/decode_tokens"].rate, 40.0, places=6)
        self.assertAlmostEqual(m.floors()["lane0/decode_tokens"].rate, 25.0, places=6)

        # 0.84 * 40 and 0.704 * 25: the round-6 concurrent point, E = 1.544.
        wins = tr.run(
            {
                "serving": ("decode_tokens", 33.6),
                "lane0": ("decode_tokens", 17.6),
            },
            windows=4,
        )
        self.assertTrue(all(w.kind == "shared" for w in wins))
        self.assertAlmostEqual(wins[-1].e, 0.84 + 0.704, places=6)
        by_key = {c.key: c for c in wins[-1].classes}
        self.assertAlmostEqual(by_key["serving"].share, 0.84, places=6)
        self.assertAlmostEqual(by_key["lane0"].share, 0.704, places=6)

    def test_serial_zero_sum_split_lands_at_or_below_one(self):
        """The instrument's own validation, and the reason the offline number
        was trusted: serial tick sharing is a zero-sum split of ONE wall clock,
        so it has to measure E <= 1.  The slice-C prefill arm measured 0.974
        (shares 0.342 and 0.632) -- feed exactly those rates back in."""
        m = self._meter()
        tr = _Trace(m)
        tr.step({})
        tr.run({"serving": ("decode_tokens", 40.0)}, windows=4)
        tr.run({"lane0": ("decode_tokens", 25.0)}, windows=4)
        wins = tr.run(
            {"serving": ("decode_tokens", 13.68), "lane0": ("decode_tokens", 15.8)},
            windows=3,
        )
        self.assertLess(wins[-1].e, 1.0)
        self.assertAlmostEqual(wins[-1].e, 0.974, places=3)

    def test_ema_tracks_but_lags(self):
        m = self._meter()
        tr = _Trace(m)
        tr.step({})
        tr.run({"serving": ("decode_tokens", 40.0)}, windows=4)
        tr.run({"lane0": ("decode_tokens", 25.0)}, windows=4)
        first = tr.run(
            {"serving": ("decode_tokens", 33.6), "lane0": ("decode_tokens", 17.6)},
            windows=1,
        )[0]
        self.assertAlmostEqual(first.e_ema, first.e, places=6)
        later = tr.run(
            {"serving": ("decode_tokens", 20.0), "lane0": ("decode_tokens", 12.5)},
            windows=1,
        )[0]
        self.assertLess(later.e_ema, first.e_ema)
        self.assertGreater(later.e_ema, later.e)

    def test_share_of_an_undisturbed_class_is_one(self):
        m = self._meter()
        tr = _Trace(m)
        tr.step({})
        tr.run({"serving": ("decode_tokens", 40.0)}, windows=4)
        tr.run({"lane0": ("decode_tokens", 25.0)}, windows=4)
        win = tr.run(
            {"serving": ("decode_tokens", 40.0), "lane0": ("decode_tokens", 25.0)},
            windows=1,
        )[0]
        self.assertAlmostEqual(win.e, 2.0, places=6)


class TestTraps(CustomTestCase):
    def test_two_arms_in_one_window_is_dropped_not_averaged(self):
        """A prefill-shaped and a decode-shaped step have different floors, so
        a window carrying both has no defined share."""
        m = LaneShareMeter(floor_min_windows=1)
        tr = _Trace(m)
        tr.step({})
        tr.run({"lane0": ("prefill_tokens", 3500.0)}, windows=2)
        tr.run({"serving": ("decode_tokens", 40.0)}, windows=2)
        tr.t += 1.0
        tr.counters["lane0"]["prefill_tokens"] += 3500.0
        tr.counters["lane0"]["decode_tokens"] += 25.0
        tr.counters["serving"]["decode_tokens"] += 40.0
        win = tr.meter.observe(
            tr.t, [ClassSample(k, dict(v)) for k, v in tr.counters.items()]
        )
        self.assertEqual(win.kind, "dropped")
        self.assertTrue(win.dropped.startswith("mixed_arms:"))
        self.assertIn("lane0", win.dropped)
        self.assertIsNone(win.e)

    def test_a_shared_window_without_a_floor_reports_no_e(self):
        m = LaneShareMeter(floor_min_windows=3)
        tr = _Trace(m)
        tr.step({})
        tr.run({"serving": ("decode_tokens", 40.0)}, windows=4)
        win = tr.run(
            {"serving": ("decode_tokens", 33.6), "lane0": ("decode_tokens", 17.6)},
            windows=1,
        )[0]
        self.assertEqual(win.kind, "shared")
        self.assertIsNone(win.e)
        self.assertTrue(win.dropped.startswith("no_floor:"))
        self.assertIn("lane0/decode_tokens", win.dropped)
        self.assertEqual(m.counts["shared_without_floor"], 1)

    def test_a_floor_needs_min_windows_before_it_counts(self):
        m = LaneShareMeter(floor_min_windows=3)
        tr = _Trace(m)
        tr.step({})
        tr.run({"lane0": ("decode_tokens", 25.0)}, windows=2)
        self.assertEqual(m.floors()["lane0/decode_tokens"].windows, 2)
        self.assertIsNone(m._floor_for("lane0", "decode_tokens"))
        tr.run({"lane0": ("decode_tokens", 25.0)}, windows=1)
        self.assertIsNotNone(m._floor_for("lane0", "decode_tokens"))

    def test_a_rung_change_inside_a_window_drops_it(self):
        """E is only defined per controller state (addendum 12 (4))."""
        m = LaneShareMeter(floor_min_windows=1)
        tr = _Trace(m)
        tr.step({})
        tr.run({"serving": ("decode_tokens", 40.0)}, windows=2)
        tr.run({"lane0": ("decode_tokens", 25.0)}, windows=2)
        win = tr.step(
            {"serving": ("decode_tokens", 33.6), "lane0": ("decode_tokens", 17.6)},
            seconds=0.4,
            rung="duty=0.5",
        )
        self.assertEqual(win.kind, "dropped")
        self.assertEqual(win.dropped, "rung_change")
        self.assertIsNone(win.e)

    def test_shared_windows_never_move_a_floor(self):
        """Self-conditioning: the denominator may not learn from the load the
        controller is steering."""
        m = LaneShareMeter(floor_min_windows=1)
        tr = _Trace(m)
        tr.step({})
        tr.run({"serving": ("decode_tokens", 40.0)}, windows=2)
        tr.run({"lane0": ("decode_tokens", 25.0)}, windows=2)
        before = m.floors()["lane0/decode_tokens"].rate
        tr.run(
            {"serving": ("decode_tokens", 33.6), "lane0": ("decode_tokens", 17.6)},
            windows=5,
        )
        self.assertAlmostEqual(m.floors()["lane0/decode_tokens"].rate, before, places=9)

    def test_frozen_floors_survive_a_solo_window(self):
        m = LaneShareMeter(floor_min_windows=1)
        tr = _Trace(m)
        tr.step({})
        tr.run({"lane0": ("decode_tokens", 25.0)}, windows=2)
        m.freeze_floors()
        tr.run({"lane0": ("decode_tokens", 5.0)}, windows=3)
        self.assertAlmostEqual(m.floors()["lane0/decode_tokens"].rate, 25.0, places=6)
        self.assertTrue(m.floors()["lane0/decode_tokens"].frozen)

    def test_loaded_floors_are_used_verbatim(self):
        m = LaneShareMeter(floor_min_windows=3)
        m.load_floors({"serving/decode_tokens": 40.0, "lane0/decode_tokens": 25.0})
        tr = _Trace(m)
        tr.step({})
        win = tr.run(
            {"serving": ("decode_tokens", 33.6), "lane0": ("decode_tokens", 17.6)},
            windows=1,
        )[0]
        self.assertAlmostEqual(win.e, 1.544, places=6)


class TestSnapshot(CustomTestCase):
    def test_snapshot_is_json_shaped(self):
        import json

        m = LaneShareMeter(floor_min_windows=1)
        tr = _Trace(m)
        tr.step({})
        tr.run({"serving": ("decode_tokens", 40.0)}, windows=2)
        tr.run({"lane0": ("decode_tokens", 25.0)}, windows=2)
        tr.run(
            {"serving": ("decode_tokens", 33.6), "lane0": ("decode_tokens", 17.6)},
            windows=2,
        )
        snap = m.snapshot()
        json.dumps(snap)
        self.assertAlmostEqual(snap["e_ema"], 1.544, places=3)
        self.assertIn("serving/decode_tokens", snap["floors"])
        self.assertEqual(snap["last"]["kind"], "shared")
        self.assertGreaterEqual(snap["counts"]["shared"], 2)


class TestSchedulerWiring(CustomTestCase):
    """The seam: the scheduler must sample BOTH classes on ONE clock, and it
    must do nothing at all when there are no lanes."""

    @staticmethod
    def _scheduler(meter, lanes, gen=0, prefill=0):
        from types import SimpleNamespace

        return SimpleNamespace(
            lane_share_meter=meter,
            _lane_share_next_t=0.0,
            dual_group_lanes=lanes,
            metrics_reporter=SimpleNamespace(
                gen_tokens_total=gen,
                prefill_tokens_total=prefill,
                log_lane_share=lambda w: None,
            ),
            _lane_rung=lambda: "static",
        )

    @staticmethod
    def _lane(lane_id, prefill, decode):
        from types import SimpleNamespace

        return SimpleNamespace(
            lane_id=lane_id,
            work_total={"prefill_tokens": prefill, "decode_tokens": decode},
        )

    def test_a_boot_without_lanes_never_touches_the_meter(self):
        from sglang.srt.managers.scheduler import Scheduler

        sched = self._scheduler(None, [])
        Scheduler._lane_share_sample(sched)  # must not raise, must not build one
        self.assertIsNone(sched.lane_share_meter)

    def test_both_classes_and_both_arms_reach_the_meter(self):
        from sglang.srt.managers.scheduler import Scheduler

        m = LaneShareMeter(window_s=0.0, floor_min_windows=1)
        sched = self._scheduler(m, [self._lane(0, 0, 0)])
        Scheduler._lane_share_sample(sched)
        # The rate limiter in front of the sampler is keyed on window_s; a
        # zero window means "sample every call", which is what this test wants.
        sched.metrics_reporter.gen_tokens_total = 40
        sched.metrics_reporter.prefill_tokens_total = 0
        sched.dual_group_lanes = [self._lane(0, 0, 25)]
        Scheduler._lane_share_sample(sched)
        last = m.history()[-1]
        self.assertEqual({c.key for c in last.classes}, {"serving", "lane0"})
        self.assertEqual(m.counts["shared_without_floor"] + m.counts["shared"], 1)

    def test_a_failing_meter_disables_itself_instead_of_killing_the_loop(self):
        from sglang.srt.managers.scheduler import Scheduler

        class _Boom:
            window_s = 1.0

            def observe(self, *a, **kw):
                raise RuntimeError("boom")

        sched = self._scheduler(_Boom(), [self._lane(0, 0, 0)])
        Scheduler._lane_share_sample(sched)
        self.assertIsNone(sched.lane_share_meter)


class TestGaugePublication(CustomTestCase):
    """Only a COMPLETE shared window may move the share/E gauges: publishing a
    partial sum as if it were E is how a controller ends up steering on noise.
    """

    class _Gauge:
        def __init__(self):
            self.values = {}
            self._pending = None

        def labels(self, **kw):
            self._pending = tuple(sorted(kw.items()))
            return self

        def set(self, v):
            self.values[self._pending] = v

    def _collector(self):
        from types import SimpleNamespace

        from sglang.srt.observability.metrics_collector import (
            SchedulerMetricsCollector,
        )

        c = SimpleNamespace(
            labels={},
            lane_share=self._Gauge(),
            lane_share_e=self._Gauge(),
            lane_share_floor=self._Gauge(),
        )
        c.log_lane_share = lambda w: SchedulerMetricsCollector.log_lane_share(c, w)
        return c

    def _meter_windows(self):
        m = LaneShareMeter(floor_min_windows=1)
        tr = _Trace(m)
        tr.step({})
        solo = tr.run({"serving": ("decode_tokens", 40.0)}, windows=2)
        shared = tr.run(
            {"serving": ("decode_tokens", 33.6), "lane0": ("decode_tokens", 17.6)},
            windows=1,
        )
        return solo, shared

    def test_a_solo_window_publishes_a_floor_but_no_e(self):
        c = self._collector()
        solo, _ = self._meter_windows()
        c.log_lane_share(solo[-1])
        self.assertEqual(c.lane_share_e.values, {})
        self.assertEqual(c.lane_share.values, {})
        self.assertIn(
            (("arm", "decode_tokens"), ("lane_class", "serving")),
            c.lane_share_floor.values,
        )

    def test_a_floorless_shared_window_publishes_no_e(self):
        c = self._collector()
        _, shared = self._meter_windows()
        win = shared[-1]
        self.assertIsNone(win.e)  # lane0 has no floor yet
        c.log_lane_share(win)
        self.assertEqual(c.lane_share_e.values, {})

    def test_a_complete_shared_window_publishes_share_and_e(self):
        m = LaneShareMeter(floor_min_windows=1)
        tr = _Trace(m)
        tr.step({})
        tr.run({"serving": ("decode_tokens", 40.0)}, windows=2)
        tr.run({"lane0": ("decode_tokens", 25.0)}, windows=2)
        win = tr.run(
            {"serving": ("decode_tokens", 33.6), "lane0": ("decode_tokens", 17.6)},
            windows=1,
        )[0]
        c = self._collector()
        c.log_lane_share(win)
        self.assertAlmostEqual(c.lane_share_e.values[()], 1.544, places=6)
        self.assertAlmostEqual(
            c.lane_share.values[(("lane_class", "serving"),)], 0.84, places=6
        )
        self.assertAlmostEqual(
            c.lane_share.values[(("lane_class", "lane0"),)], 0.704, places=6
        )


if __name__ == "__main__":
    unittest.main()
