"""The PP inter-forward BUBBLE instrument, and why it is not the wait term.

BSSCALE_0907.md (tip 37c884b0b0) measured group P at 4 792 tok/s against a
binding-stage bound of 6 472, with 26 % unattributed -- and 861/861
``Prefill rank batch`` lines reading exactly ``wait 0.0``, because that term
sums collectives INSIDE a forward and a tp_size=1 stage has none. The gap
lives BETWEEN forwards. These tests pin that the new term measures that gap,
that it reaches both the per-batch line and a per-window summary, and that it
is arithmetically independent of ``wait`` -- a reader must not be able to get
one from the other.

Pure CPU: the clock is faked, no CUDA events, no device.
"""

import logging
import unittest

import pytest

try:
    from sglang.srt.managers.scheduler_components.metrics_reporter import RankPrefillLog
    from sglang.srt.managers.scheduler_components.pp_bubble import PPBubbleMeter
    from sglang.srt.utils.collective_clock import CollectiveClock, Slot
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(
        f"#249 default-device collection leak broke the import chain: {_import_err}",
        allow_module_level=True,
    )

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LOGGER_NAME = "sglang.srt.managers.scheduler_components.metrics_reporter"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeTimer:
    """Stands in for DeviceTimer: durations complete when the test says so."""

    def __init__(self, log):
        self._log = log
        self.completed = []

    def _report(self):
        while self.completed:
            item = self.completed.pop(0)
            if isinstance(item, tuple):
                self._log._on_duration(item[0], collective_slot=item[1])
            else:
                self._log._on_duration(item)


class TestBubbleShare(unittest.TestCase):
    def test_share_is_gap_over_gap_plus_forward(self):
        """The share is computed from the GAPS, never from anything else."""
        clock = FakeClock()
        meter = PPBubbleMeter(rank=1, clock=clock, window_s=1e9)
        # Three forwards of 100 ms each, separated by gaps of 300 ms.
        for i in range(3):
            if i:
                clock.advance(0.300)
            meter.begin(i)
            clock.advance(0.100)
            meter.end()
        # gaps 2 x 300 ms = 600, forwards 3 x 100 ms = 300.
        self.assertAlmostEqual(meter.share, 600.0 / 900.0, places=9)
        self.assertAlmostEqual(meter.mean_gap_ms, 300.0, places=6)

    def test_share_is_not_derivable_from_wait(self):
        """bubble and wait are DIFFERENT terms, and the log carries both.

        This is the indicator-law half: an instrument is a finding only once
        it has been checked that it measures what it claims. Here the wait
        term is exactly zero (the tp_size=1 shape) while the bubble is large,
        so any formula deriving one from the other is refuted by one line.
        """
        clock = FakeClock()
        log = RankPrefillLog()
        log.bubble = PPBubbleMeter(rank=0, clock=clock, window_s=1e9)
        timer = FakeTimer(log)
        log.timer = timer
        clock_obj = CollectiveClock()
        log.clock = clock_obj

        # Forward 0: no preceding gap, so its line carries no bubble field.
        log.bubble.begin(0)
        clock.advance(0.050)
        log.bubble.end()
        log.record(new_tokens=4096, cached_tokens=0, timed=True, graphed=False)
        timer.completed.append((0.050, Slot()))
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as first:
            log.flush()
        self.assertNotIn("bubble_ms=", "\n".join(first.output))

        # 400 ms of pipeline bubble, then forward 1. An EMPTY slot is exactly
        # the tp_size=1 shape: no intra-forward collective was recorded, so
        # the wait term harvests 0.0 while the bubble is 400 ms.
        clock.advance(0.400)
        log.bubble.begin(1)
        clock.advance(0.050)
        log.bubble.end()
        log.record(new_tokens=4096, cached_tokens=0, timed=True, graphed=False)
        timer.completed.append((0.050, Slot()))

        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as captured:
            log.flush()
        line = "\n".join(captured.output)
        self.assertIn("bubble_ms=400.0 (between forwards, mb=1)", line)
        # ...and the wait half of the same line is 0.0. The two numbers are
        # in the same line and disagree: 400.0 cannot come from 0.0.
        self.assertIn("wait 0.0", line)

    def test_window_summary_format_and_reset(self):
        clock = FakeClock()
        meter = PPBubbleMeter(rank=2, clock=clock, window_s=5.0)
        lines = []
        # 15 s of wall: forwards of 200 ms with 300 ms gaps -> 30 forwards.
        for i in range(30):
            if i:
                clock.advance(0.300)
            meter.begin(i % 3)
            clock.advance(0.200)
            out = meter.end()
            if out:
                lines.append(out)
        self.assertTrue(lines, "a 15 s run at window_s=5 must emit summaries")
        first = lines[0]
        self.assertTrue(first.startswith("PP-BUBBLE rank=2 share="), first)
        self.assertIn("of wall, mean=", first)
        self.assertIn("n=", first)
        # The share is computable from the log: the line carries its own
        # numerator and denominator, so a reader never has to reconstruct
        # them from wall time (the denominator law).
        self.assertIn("forward_ms=", first)
        self.assertIn("bubble_ms=", first)

    def test_pending_is_taken_once(self):
        """A record that folds no forward must not inherit the last gap."""
        clock = FakeClock()
        meter = PPBubbleMeter(clock=clock, window_s=1e9)
        meter.begin(0)
        clock.advance(0.010)
        meter.end()
        clock.advance(0.100)
        meter.begin(1)
        self.assertEqual(meter.take_pending(), (100.0, 1))
        self.assertIsNone(meter.take_pending())

    def test_line_without_a_forward_boundary_is_unchanged(self):
        """A boot that never drives the meter emits the byte-identical line."""
        log = RankPrefillLog()
        timer = FakeTimer(log)
        log.timer = timer
        log.record(new_tokens=17, cached_tokens=0, timed=True, graphed=False)
        timer.completed.append(0.020)
        with self.assertLogs(LOGGER_NAME, level=logging.INFO) as captured:
            log.flush()
        line = "\n".join(captured.output)
        self.assertIn("Prefill rank batch, #new-token: 17", line)
        self.assertNotIn("bubble_ms=", line)


if __name__ == "__main__":
    unittest.main()
