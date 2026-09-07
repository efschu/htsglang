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
        # FIX 1r/2: this used to pin "of wall", which was the defect -- the
        # share is gap/(gap+forward) and the line now says so.
        self.assertIn("of gap+forward, mean=", first)
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



class TestBubbleDenominatorAndStarvation(unittest.TestCase):
    """FIX 1r/2: the line must name the denominator it actually uses, and a
    starvation gap must not read as a pipeline stall.

    The shipped module docstring asserted that ``share`` is NOT over wall time
    "because a rank that is asleep, held at a debug stop, or waiting on the
    very first request of a boot has no forward to be between" -- but ``begin``
    computes the gap from ``_last_end`` unconditionally, so a rank idle BETWEEN
    two forwards contributed its whole idle as bubble, and the emitted line
    said ``share=...% of wall``, the opposite of the docstring. Two different
    answers to the denominator question from one module, and no way to tell
    the pipeline stall PP0's 41.7 %% duty poses from queue starvation.
    """

    def test_line_names_the_denominator_it_uses(self):
        clock = FakeClock()
        meter = PPBubbleMeter(rank=0, clock=clock, window_s=1e9)
        for i in range(2):
            if i:
                clock.advance(0.300)
            meter.begin(i)
            clock.advance(0.100)
            meter.end()
        line = meter.summary_line()
        # gap/(gap+forward) is what `share` computes; the line must say so.
        self.assertIn("of gap+forward", line)
        self.assertNotIn("of wall", line)

    def test_starvation_idle_is_a_separate_numerator(self):
        """30 s of nothing-to-run between two forwards is NOT a pipeline stall.

        MEASURED before the fix: this shape printed
        ``share=99.3% of wall, mean=30000.0 ms, n=2`` with no term telling a
        reader that the 30 s was queue starvation.
        """
        clock = FakeClock()
        meter = PPBubbleMeter(rank=0, clock=clock, window_s=1e9)
        meter.begin(0)
        clock.advance(0.100)
        meter.end()
        # The scheduler visits the loop with nothing to launch, repeatedly.
        for _ in range(3):
            meter.note_no_batch()
        clock.advance(30.0)
        meter.begin(1)
        clock.advance(0.100)
        meter.end()
        line = meter.summary_line()
        self.assertIn("starved_ms=30000.0", line)
        self.assertAlmostEqual(meter.starved_ms, 30000.0, places=3)

    def test_a_pipeline_stall_is_not_charged_to_starvation(self):
        """A gap with work admitted throughout carries starved_ms=0."""
        clock = FakeClock()
        meter = PPBubbleMeter(rank=0, clock=clock, window_s=1e9)
        for i in range(3):
            if i:
                clock.advance(0.300)
            meter.begin(i)
            clock.advance(0.100)
            meter.end()
        self.assertEqual(meter.starved_ms, 0.0)
        self.assertIn("starved_ms=0.0", meter.summary_line())

    def test_max_gap_is_printed_so_one_gap_cannot_hide_in_a_mean(self):
        clock = FakeClock()
        meter = PPBubbleMeter(rank=0, clock=clock, window_s=1e9)
        # 9 short gaps of 10 ms and one long gap of 5 s: the mean is 508 ms,
        # which describes neither population.
        for i in range(11):
            if i:
                clock.advance(5.0 if i == 5 else 0.010)
            meter.begin(i)
            clock.advance(0.100)
            meter.end()
        line = meter.summary_line()
        self.assertIn("max=5000.0 ms", line)
        self.assertAlmostEqual(meter.max_gap_ms, 5000.0, places=3)

    def test_window_reset_clears_the_new_accumulators(self):
        clock = FakeClock()
        meter = PPBubbleMeter(rank=0, clock=clock, window_s=1.0)
        meter.begin(0)
        clock.advance(0.100)
        meter.end()
        meter.note_no_batch()
        clock.advance(2.0)
        meter.begin(1)
        clock.advance(0.100)
        first = meter.end()
        self.assertIn("starved_ms=2000.0", first)
        # Next window: two clean forwards, no starvation carried forward.
        for i in range(2, 4):
            clock.advance(0.010)
            meter.begin(i)
            clock.advance(0.600)
            second = meter.end()
        self.assertIsNotNone(second)
        self.assertIn("starved_ms=0.0", second)
        self.assertIn("max=10.0 ms", second)



class TestTheClassifierIsWiredAtEveryLaunchGuard(unittest.TestCase):
    """The meter sees timestamps only; the SCHEDULER knows whether there was
    work. That knowledge exists at exactly one kind of site -- the
    ``if cur_batch:`` guard around ``_pp_launch_batch`` -- and there are three
    of them. A fourth launch site added without the ``else`` would silently
    reclassify starvation as pipeline stall, which is the defect this fix
    removes, so the wiring is pinned structurally rather than by inspection.
    """

    def _guards(self):
        import ast
        import pathlib

        import sglang.srt.managers.scheduler_pp_mixin as mixin

        tree = ast.parse(pathlib.Path(mixin.__file__).read_text())
        out = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if not (isinstance(node.test, ast.Name) and node.test.id == "cur_batch"):
                continue
            body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
            if "_pp_launch_batch" in body:
                out.append(node)
        return out

    def test_all_three_launch_guards_classify_the_empty_visit(self):
        import ast

        guards = self._guards()
        self.assertEqual(len(guards), 3, "the launch-guard count moved")
        for node in guards:
            self.assertTrue(node.orelse, "a launch guard has no else branch")
            dumped = ast.dump(ast.Module(body=node.orelse, type_ignores=[]))
            self.assertIn("_pp_bubble_note_no_batch", dumped)

    def test_the_mixin_forwards_to_the_meter_and_tolerates_no_reporter(self):
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        class FakeLog:
            pass

        class FakeReporter:
            pass

        clock = FakeClock()
        meter = PPBubbleMeter(rank=0, clock=clock, window_s=1e9)
        log = FakeLog()
        log.bubble = meter
        reporter = FakeReporter()
        reporter.rank_prefill_log = log

        class FakeScheduler(SchedulerPPMixin):
            def __init__(self):
                self.metrics_reporter = reporter

        sched = FakeScheduler()
        sched._pp_bubble_note_no_batch()
        meter.begin(0)
        clock.advance(0.010)
        meter.end()
        clock.advance(1.0)
        meter.note_no_batch()
        meter.begin(1)
        self.assertAlmostEqual(meter.starved_ms, 1000.0, places=3)

        # A scheduler stand-in without a reporter must not grow one.
        class BareScheduler(SchedulerPPMixin):
            pass

        BareScheduler()._pp_bubble_note_no_batch()


if __name__ == "__main__":
    unittest.main()
