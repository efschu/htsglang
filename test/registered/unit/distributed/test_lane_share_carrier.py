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
"""#284: the device-time axis of the card-equivalent estimator, and its gate.

Round 8 left one number unexplained: the lane keeps 100 % of its solo rate
under load in round 4 and 30 % in round 8.  A rate ratio cannot say why,
because two entirely different things produce the same number -- a class that
was denied the card, and a class that had the card and ran slower on it.  This
file tests the arithmetic that separates them (occupancy vs cost), the clock
that supplies it without putting a synchronize in the lane's round, and the
standing gate that turns the result into a criterion.

Every test is hermetic: fake CUDA events with a hand-driven retirement flag,
synthetic counter traces whose true decomposition is known by construction.
"""

from __future__ import annotations

import unittest

from sglang.srt.model_executor.lane_device_clock import LaneDeviceClock
from sglang.srt.model_executor.lane_share import (
    CARRIER_SM,
    CARRIER_STARVED,
    CARRIER_SUBMISSION,
    ClassSample,
    LaneShareGate,
    LaneShareMeter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# fakes: a CUDA event whose retirement the test decides
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self, owner: "_FakeDevice") -> None:
        self.owner = owner
        self.t = 0.0
        self.done = False
        self.records = 0
        self.syncs = 0

    def record(self, stream=None) -> None:
        self.t = self.owner.device_now
        self.done = False
        self.records += 1
        self.owner.recorded.append(self)
        self.owner.last_stream = stream

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.syncs += 1
        self.done = True

    def elapsed_time(self, other: "_FakeEvent") -> float:
        return other.t - self.t


class _FakeDevice:
    """Hands out fake events and holds the fake device clock."""

    def __init__(self) -> None:
        self.device_now = 0.0
        self.created: list = []
        self.recorded: list = []
        self.last_stream = None

    def event(self) -> _FakeEvent:
        ev = _FakeEvent(self)
        self.created.append(ev)
        return ev

    def advance(self, ms: float) -> None:
        self.device_now += ms

    def retire_all(self) -> None:
        for ev in self.recorded:
            ev.done = True


class TestLaneDeviceClock(CustomTestCase):
    def _clock(self, **kw):
        dev = _FakeDevice()
        wall = {"t": 0.0}
        clk = LaneDeviceClock(
            dev.event, stream="lane-stream", clock=lambda: wall["t"], **kw
        )
        return dev, wall, clk

    def test_a_span_is_not_read_until_the_pair_has_retired(self):
        """Rule 1.  Reading an in-flight pair blocks, and a blocking read in
        the lane's round measures the wait it caused."""
        dev, _, clk = self._clock()
        with clk.span():
            dev.advance(12.0)
        self.assertEqual(clk.snapshot().device_ms, 0.0)
        self.assertEqual(clk.snapshot().pending, 1)
        self.assertEqual(clk.snapshot().forced_reads, 0)
        dev.retire_all()
        self.assertEqual(clk.harvest(), 1)
        self.assertAlmostEqual(clk.snapshot().device_ms, 12.0)
        self.assertEqual(clk.snapshot().pending, 0)

    def test_harvest_stops_at_the_first_pair_still_in_flight(self):
        dev, _, clk = self._clock()
        pairs = []
        for ms in (5.0, 7.0, 9.0):
            with clk.span():
                dev.advance(ms)
            pairs.append(len(dev.recorded))
        # Retire the FIRST pair only.  Pairs on one stream retire in order, so
        # a harvest that skipped ahead would be reading an unfinished event.
        dev.recorded[0].done = True
        dev.recorded[1].done = True
        self.assertEqual(clk.harvest(), 1)
        self.assertAlmostEqual(clk.snapshot().device_ms, 5.0)
        self.assertEqual(clk.snapshot().pending, 2)

    def test_events_are_recycled_rather_than_reallocated(self):
        dev, _, clk = self._clock()
        for _ in range(20):
            with clk.span():
                dev.advance(1.0)
            dev.retire_all()
            clk.harvest()
        self.assertEqual(clk.snapshot().spans, 20)
        self.assertAlmostEqual(clk.snapshot().device_ms, 20.0)
        # Two events for the first span, reused ever after.
        self.assertEqual(len(dev.created), 2)

    def test_at_the_ring_cap_the_oldest_pair_is_waited_for_never_dropped(self):
        """The alternative to a bounded, counted block is silent
        under-reporting of device time exactly when the lane is busiest."""
        dev, _, clk = self._clock(max_pending=2)
        for _ in range(5):
            with clk.span():
                dev.advance(3.0)
        snap = clk.snapshot()
        self.assertGreater(snap.forced_reads, 0)
        self.assertLessEqual(snap.pending, 3)
        # Nothing was lost: what is folded plus what is pending is all of it.
        dev.retire_all()
        clk.harvest()
        self.assertAlmostEqual(clk.snapshot().device_ms, 15.0)
        self.assertEqual(clk.snapshot().spans, 5)

    def test_already_measured_device_time_is_folded_without_a_second_pair(self):
        dev, _, clk = self._clock()
        clk.add_device_ms(21.5)
        clk.add_device_ms(None)
        self.assertAlmostEqual(clk.snapshot().device_ms, 21.5)
        self.assertEqual(clk.snapshot().spans, 1)
        self.assertEqual(dev.created, [])

    def test_busy_wall_counts_the_open_interval_but_device_time_does_not(self):
        dev, wall, clk = self._clock()
        clk.mark_busy()
        wall["t"] = 0.5
        clk.mark_busy()  # idempotent: must not restart the interval
        wall["t"] = 2.0
        snap = clk.snapshot()
        self.assertAlmostEqual(snap.busy_wall_ms, 2000.0)
        # An unretired span has no measured duration and none is invented.
        with clk.span():
            dev.advance(50.0)
        self.assertEqual(clk.snapshot().device_ms, 0.0)
        clk.mark_idle()
        wall["t"] = 3.0
        self.assertAlmostEqual(clk.snapshot().busy_wall_ms, 2000.0)

    def test_mark_idle_without_a_busy_interval_is_a_no_op(self):
        _, wall, clk = self._clock()
        clk.mark_idle()
        wall["t"] = 9.0
        self.assertEqual(clk.snapshot().busy_wall_ms, 0.0)

    def test_drain_blocks_on_whatever_is_left(self):
        dev, _, clk = self._clock()
        for _ in range(3):
            with clk.span():
                dev.advance(4.0)
        self.assertEqual(clk.drain(), 3)
        self.assertAlmostEqual(clk.snapshot().device_ms, 12.0)

    def test_the_stream_is_the_one_it_was_bound_to(self):
        dev, _, clk = self._clock()
        clk.bind_stream("other-stream")
        with clk.span():
            pass
        self.assertEqual(dev.last_stream, "other-stream")


# ---------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------


class _DeviceTrace:
    """A counter trace that carries work AND device time per class.

    ``rates`` maps a key to ``(arm, work_per_s, occupancy, duty)`` where
    occupancy is the fraction of the window the class's kernels executed and
    duty the fraction it held work.  Device ms then follow from the definition,
    which is the point: the test states the physics and the meter has to
    recover it, not the other way round.
    """

    def __init__(self, meter, keys=("serving", "lane0")):
        self.meter = meter
        self.t = 100.0
        self.work = {k: {"prefill_tokens": 0.0, "decode_tokens": 0.0} for k in keys}
        self.dev = {k: {"device_ms": 0.0, "busy_wall_ms": 0.0} for k in keys}
        self.no_device = set()

    def step(self, rates, *, seconds=1.0, rung="static"):
        self.t += seconds
        for key, spec in rates.items():
            if spec is None:
                continue
            arm, rate, occupancy, duty = spec
            self.work[key][arm] += rate * seconds
            self.dev[key]["device_ms"] += occupancy * seconds * 1000.0
            self.dev[key]["busy_wall_ms"] += duty * seconds * 1000.0
        samples = [
            ClassSample(
                k,
                dict(v),
                device=None if k in self.no_device else dict(self.dev[k]),
            )
            for k, v in self.work.items()
        ]
        return self.meter.observe(self.t, samples, rung=rung)

    def run(self, rates, *, windows, **kw):
        out = []
        for _ in range(windows):
            win = self.step(rates, **kw)
            if win is not None:
                out.append(win)
        return out


def _row(win, key):
    for row in win.classes:
        if row.key == key:
            return row
    raise AssertionError(f"{key} not in window")


class TestCarrierDecomposition(CustomTestCase):
    """share = occupancy_ratio / cost_ratio, and which term carries a loss."""

    def _meter(self):
        return LaneShareMeter(window_s=1.0, ema_s=1.0, floor_min_windows=1)

    def _floors(self, tr):
        # Lane solo: 50 tok/s, kernels on the card 90 % of the time, always
        # holding work -> 18 ms of device time per token.
        tr.step({})
        tr.run({"lane0": ("decode_tokens", 50.0, 0.90, 1.0)}, windows=2)
        tr.run({"serving": ("decode_tokens", 40.0, 0.95, 1.0)}, windows=2)

    def test_the_share_is_exactly_occupancy_over_cost(self):
        m = self._meter()
        tr = _DeviceTrace(m)
        self._floors(tr)
        win = tr.run(
            {
                "serving": ("decode_tokens", 34.0, 0.80, 1.0),
                "lane0": ("decode_tokens", 15.0, 0.45, 1.0),
            },
            windows=1,
        )[0]
        row = _row(win, "lane0")
        self.assertAlmostEqual(row.share, 15.0 / 50.0, places=9)
        self.assertAlmostEqual(
            row.share, row.occupancy_ratio / row.cost_ratio, places=9
        )

    def test_slower_kernels_on_the_same_card_time_is_sm_competition(self):
        m = self._meter()
        tr = _DeviceTrace(m)
        self._floors(tr)
        # Same 90 % of the card, half the tokens: every token cost twice the
        # device ms it did solo.
        win = tr.run(
            {
                "serving": ("decode_tokens", 30.0, 0.90, 1.0),
                "lane0": ("decode_tokens", 25.0, 0.90, 1.0),
            },
            windows=1,
        )[0]
        row = _row(win, "lane0")
        self.assertAlmostEqual(row.occupancy_ratio, 1.0, places=9)
        self.assertAlmostEqual(row.cost_ratio, 2.0, places=9)
        self.assertEqual(row.carrier, CARRIER_SM)

    def test_the_same_cost_on_less_card_time_is_a_submission_gap(self):
        m = self._meter()
        tr = _DeviceTrace(m)
        self._floors(tr)
        # Held work the whole window (duty 1.0) but its kernels were only on
        # the card for 45 % of it, at unchanged 18 ms per token.
        win = tr.run(
            {
                "serving": ("decode_tokens", 30.0, 0.90, 1.0),
                "lane0": ("decode_tokens", 25.0, 0.45, 1.0),
            },
            windows=1,
        )[0]
        row = _row(win, "lane0")
        self.assertAlmostEqual(row.cost_ratio, 1.0, places=9)
        self.assertAlmostEqual(row.occupancy_ratio, 0.5, places=9)
        self.assertEqual(row.carrier, CARRIER_SUBMISSION)

    def test_a_lane_that_did_not_hold_work_is_starved_not_denied(self):
        m = self._meter()
        tr = _DeviceTrace(m)
        self._floors(tr)
        win = tr.run(
            {
                "serving": ("decode_tokens", 30.0, 0.90, 1.0),
                "lane0": ("decode_tokens", 25.0, 0.45, 0.5),
            },
            windows=1,
        )[0]
        row = _row(win, "lane0")
        self.assertAlmostEqual(row.duty_ratio, 0.5, places=9)
        self.assertEqual(row.carrier, CARRIER_STARVED)

    def test_a_class_that_lost_nothing_has_no_carrier(self):
        m = self._meter()
        tr = _DeviceTrace(m)
        self._floors(tr)
        win = tr.run(
            {
                "serving": ("decode_tokens", 30.0, 0.90, 1.0),
                "lane0": ("decode_tokens", 50.0, 0.90, 1.0),
            },
            windows=1,
        )[0]
        self.assertIsNone(_row(win, "lane0").carrier)

    def test_a_class_without_a_device_clock_still_gets_a_share(self):
        """No occupancy is not zero occupancy: the absence of an instrument
        must not be reported as a measurement of denial."""
        m = self._meter()
        tr = _DeviceTrace(m)
        tr.no_device.add("lane0")
        self._floors(tr)
        win = tr.run(
            {
                "serving": ("decode_tokens", 30.0, 0.90, 1.0),
                "lane0": ("decode_tokens", 25.0, 0.45, 1.0),
            },
            windows=1,
        )[0]
        row = _row(win, "lane0")
        self.assertAlmostEqual(row.share, 0.5, places=9)
        self.assertIsNone(row.occupancy)
        self.assertIsNone(row.carrier)
        self.assertNotIn("occupancy", row.to_json())

    def test_a_window_without_device_counters_does_not_erase_a_floor(self):
        m = self._meter()
        tr = _DeviceTrace(m)
        self._floors(tr)
        before = m.floors()["lane0/decode_tokens"]
        self.assertIsNotNone(before.occupancy)
        tr.no_device.add("lane0")
        tr.run({"lane0": ("decode_tokens", 50.0, 0.90, 1.0)}, windows=1)
        after = m.floors()["lane0/decode_tokens"]
        self.assertAlmostEqual(after.occupancy, before.occupancy, places=9)
        self.assertAlmostEqual(after.cost, before.cost, places=9)

    def test_the_window_json_names_the_carrier(self):
        m = self._meter()
        tr = _DeviceTrace(m)
        self._floors(tr)
        win = tr.run(
            {
                "serving": ("decode_tokens", 30.0, 0.90, 1.0),
                "lane0": ("decode_tokens", 25.0, 0.90, 1.0),
            },
            windows=1,
        )[0]
        js = win.to_json()
        lane = [c for c in js["classes"] if c["key"] == "lane0"][0]
        self.assertEqual(lane["carrier"], CARRIER_SM)
        self.assertIn("occupancy_ratio", lane)
        self.assertIn("cost_ratio", lane)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


class TestLaneShareGate(CustomTestCase):
    def _shared(self, share, carrier=CARRIER_SM):
        """One synthetic shared window carrying a chosen share and carrier."""
        m = LaneShareMeter(window_s=1.0, floor_min_windows=1)
        tr = _DeviceTrace(m)
        tr.step({})
        tr.run({"lane0": ("decode_tokens", 50.0, 0.90, 1.0)}, windows=2)
        tr.run({"serving": ("decode_tokens", 40.0, 0.95, 1.0)}, windows=2)
        occ = 0.90 if carrier == CARRIER_SM else 0.90 * share
        duty = 0.5 if carrier == CARRIER_STARVED else 1.0
        return tr.run(
            {
                "serving": ("decode_tokens", 30.0, 0.90, 1.0),
                "lane0": ("decode_tokens", 50.0 * share, occ, duty),
            },
            windows=1,
        )[0]

    def test_below_min_windows_the_verdict_is_insufficient_never_pass(self):
        g = LaneShareGate("lane0", 0.3, min_windows=5)
        for _ in range(4):
            g.observe(self._shared(0.9))
        self.assertEqual(g.verdict, "insufficient")

    def test_the_median_decides_not_the_mean(self):
        """One window in which the lane was between jobs is a long left tail
        on a bounded quantity; a mean lets that tail decide a gate."""
        g = LaneShareGate("lane0", 0.30, min_windows=5)
        for share in (0.35, 0.36, 0.34, 0.35, 0.01):
            g.observe(self._shared(share))
        self.assertAlmostEqual(g.median_share, 0.35, places=6)
        self.assertEqual(g.verdict, "pass")

    def test_a_lane_below_the_threshold_fails_and_the_carrier_is_named(self):
        g = LaneShareGate("lane0", 0.50, load="4 concurrent 128-token requests")
        for _ in range(6):
            g.observe(self._shared(0.30, carrier=CARRIER_SM))
        self.assertEqual(g.verdict, "fail")
        self.assertEqual(g.carrier, CARRIER_SM)
        self.assertEqual(g.failed, 6)
        self.assertIn("4 concurrent", g.describe())

    def test_the_carrier_follows_the_windows_not_the_threshold(self):
        g = LaneShareGate("lane0", 0.50)
        for _ in range(6):
            g.observe(self._shared(0.30, carrier=CARRIER_SUBMISSION))
        self.assertEqual(g.carrier, CARRIER_SUBMISSION)

    def test_solo_and_idle_windows_are_not_judged(self):
        """A lane must not pass a load gate by not being loaded."""
        m = LaneShareMeter(window_s=1.0, floor_min_windows=1)
        g = LaneShareGate("lane0", 0.9, min_windows=1)
        m.attach_gate(g)
        tr = _DeviceTrace(m)
        tr.step({})
        tr.run({"lane0": ("decode_tokens", 50.0, 0.90, 1.0)}, windows=3)
        tr.run({}, windows=2)
        self.assertEqual(g.judged, 0)
        self.assertEqual(g.verdict, "insufficient")

    def test_an_attached_gate_sees_the_meters_windows(self):
        m = LaneShareMeter(window_s=1.0, floor_min_windows=1)
        g = LaneShareGate("lane0", 0.30, min_windows=1, load="probe")
        m.attach_gate(g)
        tr = _DeviceTrace(m)
        tr.step({})
        tr.run({"lane0": ("decode_tokens", 50.0, 0.90, 1.0)}, windows=2)
        tr.run({"serving": ("decode_tokens", 40.0, 0.95, 1.0)}, windows=2)
        tr.run(
            {
                "serving": ("decode_tokens", 30.0, 0.90, 1.0),
                "lane0": ("decode_tokens", 10.0, 0.90, 1.0),
            },
            windows=2,
        )
        self.assertEqual(g.judged, 2)
        self.assertEqual(g.verdict, "fail")
        snap = m.snapshot()
        self.assertEqual(len(snap["gates"]), 1)
        self.assertEqual(snap["gates"][0]["verdict"], "fail")
        self.assertEqual(snap["gates"][0]["load"], "probe")

    def test_a_floor_that_moved_while_judging_voids_the_verdict(self):
        """Measured on the #284 boot: a driver that ran three lane
        configurations through one meter blended their solo rates into one
        floor, and the gate reported a comfortable pass for a lane that was
        keeping barely half of what the number said."""
        g = LaneShareGate("lane0", 0.30, min_windows=3)
        m = LaneShareMeter(window_s=1.0, floor_min_windows=1)
        m.attach_gate(g)
        tr = _DeviceTrace(m)
        tr.step({})
        tr.run({"serving": ("decode_tokens", 40.0, 0.95, 1.0)}, windows=2)
        # A fast lane, then a slow one: the floor is re-learned in between.
        tr.run({"lane0": ("decode_tokens", 50.0, 0.90, 1.0)}, windows=2)
        tr.run(
            {
                "serving": ("decode_tokens", 30.0, 0.90, 1.0),
                "lane0": ("decode_tokens", 20.0, 0.90, 1.0),
            },
            windows=2,
        )
        tr.run({"lane0": ("decode_tokens", 15.0, 0.30, 1.0)}, windows=3)
        tr.run(
            {
                "serving": ("decode_tokens", 30.0, 0.90, 1.0),
                "lane0": ("decode_tokens", 10.0, 0.60, 1.0),
            },
            windows=2,
        )
        self.assertGreater(g.floor_span, g.floor_tolerance)
        self.assertEqual(g.verdict, "insufficient")
        self.assertEqual(g.insufficient_reason, "floor_moved")

    def test_frozen_floors_give_the_verdict_back(self):
        g = LaneShareGate("lane0", 0.30, min_windows=3)
        m = LaneShareMeter(window_s=1.0, floor_min_windows=1)
        m.attach_gate(g)
        tr = _DeviceTrace(m)
        tr.step({})
        tr.run({"serving": ("decode_tokens", 40.0, 0.95, 1.0)}, windows=2)
        tr.run({"lane0": ("decode_tokens", 50.0, 0.90, 1.0)}, windows=2)
        m.freeze_floors()
        tr.run({"lane0": ("decode_tokens", 15.0, 0.30, 1.0)}, windows=2)
        tr.run(
            {
                "serving": ("decode_tokens", 30.0, 0.90, 1.0),
                "lane0": ("decode_tokens", 10.0, 0.60, 1.0),
            },
            windows=4,
        )
        self.assertEqual(g.floor_span, 0.0)
        self.assertEqual(g.verdict, "fail")
        self.assertAlmostEqual(g.median_share, 0.2, places=6)

    def test_the_reason_names_which_of_the_two_holes_it_is(self):
        g = LaneShareGate("lane0", 0.30, min_windows=99)
        self.assertEqual(g.insufficient_reason, "too_few_windows")

    def test_a_boot_without_a_threshold_carries_no_gate(self):
        m = LaneShareMeter()
        self.assertEqual(m.snapshot()["gates"], [])

    def test_the_gate_cannot_move_the_number_it_judges(self):
        """It is a consumer.  A meter with and without a gate has to produce
        identical windows, or the instrument is steering its own reading."""
        wins = []
        for attach in (False, True):
            m = LaneShareMeter(window_s=1.0, floor_min_windows=1)
            if attach:
                m.attach_gate(LaneShareGate("lane0", 0.3))
            tr = _DeviceTrace(m)
            tr.step({})
            tr.run({"lane0": ("decode_tokens", 50.0, 0.90, 1.0)}, windows=2)
            tr.run({"serving": ("decode_tokens", 40.0, 0.95, 1.0)}, windows=2)
            wins.append(
                [
                    w.to_json()
                    for w in tr.run(
                        {
                            "serving": ("decode_tokens", 30.0, 0.90, 1.0),
                            "lane0": ("decode_tokens", 10.0, 0.90, 1.0),
                        },
                        windows=2,
                    )
                ]
            )
        for a, b in zip(*wins):
            a.pop("t_end"), b.pop("t_end")
            self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
