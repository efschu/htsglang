"""#1241 slice (1): the DECODE half of the per-rank compute/wait clock.

Hermetic. No CUDA, no GPU, no model: the device sits behind ``ClockBackend``
(utils/collective_clock.py) and every test here injects a fake one whose
events are plain integers. Run with ``CUDA_VISIBLE_DEVICES=''``.

Named red-first: each test name states the WRONG behaviour it exists to
catch, because the failure mode of a timing instrument is not a crash -- it
is a plausible number. The four that carry mutants are marked MUTANT-N.
"""

from __future__ import annotations

import logging
import unittest

from sglang.srt.debug_utils.rank_phase_summary import (
    parse_rank_batch_line,
    parse_unsplit_line,
)
from sglang.srt.managers.scheduler_components.decode_round_log import DecodeRoundLog
from sglang.srt.utils.collective_clock import ClockBackend, CollectiveClock


# ---------------------------------------------------------------------------
# The fake device. ONE adapter, per the module's own contract.
# ---------------------------------------------------------------------------


class FakeClockState:
    def __init__(self) -> None:
        self.now = 0.0
        #: An event is readable once its record time is <= this. DEFAULT is
        #: "immediately readable", which is the ADVERSARIAL default: a clock
        #: that reads the round it is inside would then succeed, and the
        #: ordering tests have to hold events back explicitly to make the
        #: deferral observable.
        self.readable_from = float("inf")
        self.synchronize_calls = 0

    def advance(self, ms: float) -> None:
        self.now += float(ms)


class FakeEvent:
    def __init__(self, state: FakeClockState) -> None:
        self._state = state
        self.t = None

    def record(self) -> None:
        self.t = self._state.now

    def query(self) -> bool:
        return self.t is not None and self.t <= self._state.readable_from

    def elapsed_time(self, other: FakeEvent) -> float:
        return other.t - self.t

    def synchronize(self) -> None:  # pragma: no cover - must never be called
        self._state.synchronize_calls += 1
        raise AssertionError(
            "the decode round clock synchronized the device; the whole point "
            "of reading one round late is that it never does"
        )


class FakeBackend(ClockBackend):
    def __init__(self, state: FakeClockState) -> None:
        self.state = state
        self.capturing = False

    def event(self):
        return FakeEvent(self.state)

    def is_capturing(self) -> bool:
        return self.capturing


class Harness:
    """A clock, a log, a captured logger, and a way to run a round."""

    def __init__(self, rank: int = 1) -> None:
        self.state = FakeClockState()
        self.backend = FakeBackend(self.state)
        self.clock = CollectiveClock(backend=self.backend)
        self.log = DecodeRoundLog(clock=self.clock, rank=rank)

    def ready_now(self) -> None:
        """Release everything held back."""
        self.state.readable_from = float("inf")

    def hold(self) -> None:
        """Everything recorded from now on stays unreadable until released."""
        self.state.readable_from = self.state.now - 1e-9


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class DecodeRoundClockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness()
        self.cap = _Capture()
        self.logger = logging.getLogger(
            "sglang.srt.managers.scheduler_components.decode_round_log"
        )
        self.logger.addHandler(self.cap)
        self.logger.setLevel(logging.INFO)
        self.addCleanup(self.logger.removeHandler, self.cap)

    # -- helpers ---------------------------------------------------------

    def run_forward(self, category="decode", graphed=False, compute_ms=10.0, waits=()):
        """One bracketed forward: `waits` is a list of (family, ms)."""
        with self.h.log.segment(category, graphed):
            for family, ms in waits:
                with self.h.clock.span(family):
                    self.h.state.advance(ms)
                    compute_ms -= ms
            self.h.state.advance(max(compute_ms, 0.0))

    def decode_lines(self):
        return [l for l in self.cap.lines if l.startswith("Decode rank batch")]

    # -- the tests -------------------------------------------------------

    def test_MUTANT_1_compute_is_the_round_MINUS_the_wait_never_the_wait_itself(self):
        """Swapping compute and wait produces a line of the same shape."""
        self.h.log.begin_round(round_id=7, bs=1, rows=1)
        self.run_forward(compute_ms=20.0, waits=[("tp.all_reduce", 5.0)])
        self.h.ready_now()
        self.h.log.begin_round(round_id=8, bs=1, rows=1)
        line = self.decode_lines()[0]
        parsed = parse_rank_batch_line("[2026-09-09 00:00:00 TP1] " + line)
        self.assertIsNotNone(parsed, line)
        self.assertAlmostEqual(parsed["gpu_ms"], 20.0, places=1)
        self.assertAlmostEqual(parsed["wait_ms"], 5.0, places=1)
        self.assertAlmostEqual(parsed["compute_ms"], 15.0, places=1)
        # The asymmetry is what a swap cannot survive.
        self.assertGreater(parsed["compute_ms"], parsed["wait_ms"])

    def test_MUTANT_2_round_N_is_NOT_emitted_while_round_N_is_open(self):
        """A clock that reads the current round has to synchronize to do it."""
        self.h.log.begin_round(round_id=11, bs=1, rows=1)
        self.run_forward()
        self.h.ready_now()
        self.assertEqual(self.decode_lines(), [], "round 11 emitted inside round 11")
        self.h.log.begin_round(round_id=12, bs=1, rows=1)
        self.assertEqual(len(self.decode_lines()), 1)
        self.assertIn("#round: 11", self.decode_lines()[0])
        self.assertNotIn("#round: 12", self.decode_lines()[0])
        self.assertEqual(self.h.state.synchronize_calls, 0)

    def test_MUTANT_3_the_line_names_the_rank_as_a_FIELD_not_via_the_prefix(self):
        """The two Weg-2 groups write different log prefixes; a cross-rank
        join that depends on the formatter is not a join."""
        self.h.log.begin_round(round_id=1, bs=2, rows=2)
        self.run_forward()
        self.h.ready_now()
        self.h.log.begin_round(round_id=2, bs=2, rows=2)
        line = self.decode_lines()[0]
        self.assertIn("rank: 1,", line)
        # Parsed with NO TPn in the prefix at all.
        parsed = parse_rank_batch_line("[2026-09-09 00:00:00] " + line)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["rank"], "TP1")

    def test_MUTANT_4_a_graph_replayed_round_REFUSES_the_split_never_reports_zero(self):
        """A replayed collective records no span. An empty slot is not a
        wait of 0.0; it is an unknown wait.

        #1241b: a graph that carries event NODES now yields the split (see
        test_decode_graph_event_nodes_1241b.py). This harness declares no
        graph, which is exactly the shape of a graph captured before the
        nodes existed -- and the reason now NAMES the missing thing instead
        of naming the mechanism, because "graph-replay" as a reason claims
        graphs cannot be split, which is false since #1241b."""
        self.h.log.begin_round(round_id=3, bs=6, rows=24)
        self.run_forward(graphed=True, compute_ms=12.9, waits=())
        self.h.ready_now()
        self.h.log.begin_round(round_id=4, bs=6, rows=24)
        line = self.decode_lines()[0]
        self.assertIn("split unavailable: graph-replay-no-event-nodes", line)
        self.assertIn("graphed-fwd 1/1", line)
        self.assertNotIn("wait 0.0", line)
        self.assertNotIn("compute", line)
        self.assertIsNone(parse_rank_batch_line("[x TP1] " + line))
        un = parse_unsplit_line("[2026-09-09 00:00:00 TP1] " + line)
        self.assertIsNotNone(un)
        self.assertEqual(un["reason"], "graph-replay-no-event-nodes")
        self.assertFalse(un["split_known"])
        self.assertAlmostEqual(un["gpu_ms"], 12.9, places=1)

    def test_ONE_grep_finds_the_prefill_and_the_decode_family(self):
        self.h.log.begin_round(round_id=5, bs=1, rows=1)
        self.run_forward()
        self.h.ready_now()
        self.h.log.begin_round(round_id=6, bs=1, rows=1)
        decode = self.decode_lines()[0]
        prefill = (
            "Prefill rank batch, #new-token: 53, #cached-token: 0, #chunks: 1, "
            "gpu-ms: 111.0 (compute 32.8, wait 78.1)"
        )
        for line in (decode, prefill):
            self.assertIn(" rank batch, ", line)

    def test_a_speculative_round_folds_draft_and_verify_into_ONE_line(self):
        self.h.log.begin_round(round_id=20, bs=6, rows=24)
        self.run_forward("extend", compute_ms=6.0, waits=[("tp.all_reduce", 2.0)])
        self.run_forward(
            "target_verify", compute_ms=14.0, waits=[("tp.all_reduce", 4.0)]
        )
        self.h.ready_now()
        self.h.log.begin_round(round_id=21, bs=6, rows=24)
        lines = self.decode_lines()
        self.assertEqual(len(lines), 1, lines)
        line = lines[0]
        self.assertIn("#fwd: 2", line)
        self.assertIn("bs: 6", line)
        self.assertIn("#rows: 24", line)
        self.assertIn("gpu-ms: 20.0", line)
        self.assertIn("spec_draft:tp.all_reduce 2.0/1x", line)
        self.assertIn("spec_verify:tp.all_reduce 4.0/1x", line)

    def test_the_phase_prefix_KEEPS_the_dispatch_family_it_does_not_replace_it(self):
        self.h.log.begin_round(round_id=30, bs=1, rows=4)
        self.run_forward(
            "target_verify",
            compute_ms=9.0,
            waits=[("tp.all_reduce", 1.0), ("dcp.all_gather", 2.0)],
        )
        self.h.ready_now()
        self.h.log.begin_round(round_id=31, bs=1, rows=4)
        line = self.decode_lines()[0]
        self.assertIn("spec_verify:dcp.all_gather", line)
        self.assertIn("spec_verify:tp.all_reduce", line)

    def test_an_unreadable_round_BLOCKS_the_drain_and_never_reorders(self):
        self.h.log.begin_round(round_id=40, bs=1, rows=1)
        self.h.hold()
        self.run_forward()
        # round 40's events stay unreadable
        self.h.log.begin_round(round_id=41, bs=1, rows=1)
        self.run_forward()
        self.h.ready_now()
        self.h.log.begin_round(round_id=42, bs=1, rows=1)
        lines = self.decode_lines()
        self.assertEqual(len(lines), 2)
        self.assertIn("#round: 40", lines[0])
        self.assertIn("#round: 41", lines[1])

    def test_a_decode_step_with_no_timed_forward_emits_NOTHING(self):
        """An empty round is not a zero round."""
        self.h.log.begin_round(round_id=50, bs=1, rows=1)
        self.h.ready_now()
        self.h.log.begin_round(round_id=51, bs=1, rows=1)
        self.assertEqual(self.decode_lines(), [])

    def test_outside_a_round_a_segment_is_a_NOOP(self):
        """Prefill forwards must pass through untouched."""
        with self.h.log.segment("extend", False):
            with self.h.clock.span("tp.all_reduce"):
                self.h.state.advance(3.0)
        self.h.ready_now()
        self.assertEqual(self.decode_lines(), [])
        # And the round counter never opened: no pending round exists to be
        # emitted later either.
        self.h.log.begin_round(round_id=60, bs=1, rows=1)
        self.h.log.begin_round(round_id=61, bs=1, rows=1)
        self.assertEqual(self.decode_lines(), [])

    def test_the_wait_can_never_exceed_the_round(self):
        self.h.log.begin_round(round_id=70, bs=1, rows=1)
        with self.h.log.segment("decode", False):
            with self.h.clock.span("tp.all_reduce"):
                self.h.state.advance(5.0)
        self.h.ready_now()
        self.h.log.begin_round(round_id=71, bs=1, rows=1)
        parsed = parse_rank_batch_line("[x TP1] " + self.decode_lines()[0])
        self.assertGreaterEqual(parsed["compute_ms"], 0.0)
        self.assertLessEqual(parsed["wait_ms"], parsed["gpu_ms"] + 1e-6)

    def test_the_overhead_is_stated_ONCE_with_its_denominator(self):
        self.h.log.OVERHEAD_ROUNDS = 3
        for i in range(6):
            self.h.log.begin_round(round_id=100 + i, bs=1, rows=1)
            self.run_forward()
            self.h.ready_now()
        self.h.log.begin_round(round_id=999, bs=1, rows=1)
        over = [l for l in self.cap.lines if l.startswith("Decode rank clock overhead")]
        self.assertEqual(len(over), 1, over)
        self.assertIn("us/round host-side over", over[0])
        self.assertIn("% of the mean round gpu-ms", over[0])
        self.assertIn("Dropped rounds", over[0])

    def test_the_summary_parser_reads_the_decode_JOIN_KEYS(self):
        self.h.log.begin_round(round_id=412, bs=6, rows=24)
        self.run_forward(compute_ms=31.4, waits=[("tp.all_reduce", 11.6)])
        self.h.ready_now()
        self.h.log.begin_round(round_id=413, bs=6, rows=24)
        parsed = parse_rank_batch_line(
            "[2026-09-09 00:00:00 TP1] " + self.decode_lines()[0]
        )
        self.assertEqual(parsed["round"], 412)
        self.assertEqual(parsed["bs"], 6)
        self.assertEqual(parsed["rows"], 24)
        self.assertEqual(parsed["phase"], "Decode")
        self.assertEqual(parsed["rank"], "TP1")
        self.assertIn("wall", parsed)

    # -- the round's LIFETIME (review R1 / refuter MF-1, MF-8) -------------

    def test_MUTANT_9_a_round_left_OPEN_swallows_the_forwards_after_it(self):
        """The defect the first version shipped: a round opened at the funnel
        and retired only at the NEXT decode batch stays open across the batch
        boundary, so the following prefill forward folds into it -- more
        ``#fwd``, more ``gpu-ms``, and its collectives labelled
        ``spec_draft:``. ``end_round`` is what the funnel calls for every
        non-decode batch."""
        self.h.log.begin_round(round_id=200, bs=6, rows=24)
        self.run_forward(compute_ms=20.0, waits=[("tp.all_reduce", 4.0)])
        self.h.log.end_round()
        # A PREFILL forward now runs. It must not be bracketed at all.
        self.run_forward("extend", compute_ms=90.0, waits=[("tp.all_reduce", 30.0)])
        self.h.ready_now()
        self.h.log.begin_round(round_id=201, bs=6, rows=24)
        lines = self.decode_lines()
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("#fwd: 1", lines[0])
        self.assertIn("gpu-ms: 20.0", lines[0])
        self.assertNotIn("spec_draft", lines[0])

    def test_MUTANT_10_the_LAST_round_of_a_burst_is_emitted_not_left_open(self):
        """A decode burst that ends has no next batch to close its last
        round. Without a retire at the idle tick that round -- the last of
        every measurement window and every pre-flip drain -- is never
        emitted, which the module docstring already claimed it was."""
        self.h.log.begin_round(round_id=300, bs=1, rows=1)
        self.run_forward()
        self.h.ready_now()
        self.assertEqual(self.decode_lines(), [])
        self.assertTrue(self.h.log.has_pending)
        self.h.log.end_round()
        self.assertEqual(len(self.decode_lines()), 1)
        self.assertIn("#round: 300", self.decode_lines()[0])
        self.assertFalse(self.h.log.has_pending)

    def test_end_round_is_IDEMPOTENT_and_emits_nothing_a_second_time(self):
        self.h.log.begin_round(round_id=301, bs=1, rows=1)
        self.run_forward()
        self.h.ready_now()
        self.h.log.end_round()
        n = len(self.decode_lines())
        self.h.log.end_round()
        self.h.log.end_round()
        self.assertEqual(len(self.decode_lines()), n)

    # -- ONE SLOT, TWO ARMERS (review R1 / refuter MF-2) -------------------

    def test_MUTANT_11_a_decode_round_NEVER_steals_the_armed_PREFILL_slot(self):
        """The silent half of the round-lifetime defect. ``open_round`` used
        to overwrite ``_slot`` unconditionally, so a round opened while
        ``SplitDeviceTimer`` held the slot made the SHIPPED #252 prefill line
        lose its compute/wait column -- a regression of the other half of
        this same instrument, visible nowhere."""
        self.h.clock.arm()
        self.h.log.begin_round(round_id=400, bs=1, rows=1)
        with self.h.log.segment("decode", False):
            with self.h.clock.span("tp.all_reduce"):
                self.h.state.advance(3.0)
        prefill_slot = self.h.clock.disarm()
        self.assertIsNotNone(
            prefill_slot, "the decode round disarmed the prefill timer"
        )
        self.assertEqual(
            len(prefill_slot.pairs),
            1,
            "the collective went into the round's slot, not the prefill's",
        )
        self.assertEqual(self.h.clock.contention_counts, (0, 1))

    def test_MUTANT_12_a_CONTENDED_round_refuses_its_split_never_reports_zero(self):
        self.h.clock.arm()
        self.h.log.begin_round(round_id=401, bs=1, rows=1)
        self.run_forward(compute_ms=8.0, waits=[("tp.all_reduce", 2.0)])
        self.h.clock.disarm()
        self.h.ready_now()
        self.h.log.end_round()
        line = self.decode_lines()[0]
        self.assertIn("split unavailable: slot-contended-with-prefill", line)
        self.assertNotIn("wait 0.0", line)

    def test_a_prefill_arming_INSIDE_an_open_round_is_refused_not_granted(self):
        """The same theft in the other direction: ``arm`` overwriting a
        round's slot would empty it, and the round would report ``wait 0.0``
        -- a fabrication with the shape of a measurement."""
        self.h.log.begin_round(round_id=402, bs=1, rows=1)
        with self.h.log.segment("decode", False):
            self.h.clock.arm()
            self.assertIsNone(
                self.h.clock.disarm(),
                "a refused arming handed back somebody else's slot",
            )
            with self.h.clock.span("tp.all_reduce"):
                self.h.state.advance(4.0)
            self.h.state.advance(6.0)
        self.h.ready_now()
        self.h.log.end_round()
        line = self.decode_lines()[0]
        self.assertIn("(compute 6.0, wait 4.0)", line)
        self.assertEqual(self.h.clock.contention_counts, (1, 0))

    # -- the LADDER JOIN AXIS (review R3) ----------------------------------

    def test_MUTANT_13_the_join_axis_is_WALL_TIME_and_is_ON_the_line(self):
        """``#round`` is the scheduler's ``forward_ct``; the ladder probe's
        own ``round`` is an arm repeat index. Joining them is a category
        error, so the line carries an epoch stamp that both sides can read."""
        import time as _time

        before = _time.time()
        self.h.log.begin_round(round_id=4120, bs=6, rows=24)
        self.run_forward(compute_ms=31.4, waits=[("tp.all_reduce", 11.6)])
        after = _time.time()
        self.h.ready_now()
        self.h.log.end_round()
        parsed = parse_rank_batch_line("[x TP1] " + self.decode_lines()[0])
        self.assertIn("wall", parsed)
        # Stamped when the round OPENED, not when the line was written. The
        # field is printed to MILLISECONDS, so the readback can sit up to one
        # tick below the stamp -- that quantisation is the join's resolution
        # and is named here rather than papered over with a loose bound.
        tick = 0.001
        self.assertGreaterEqual(parsed["wall"], before - tick)
        self.assertLessEqual(parsed["wall"], after + tick)

    def test_the_row_count_is_rows_SUBMITTED_and_the_field_says_so(self):
        """``#tokens`` invited the reading that it is comparable to the
        ladder's ``completion_tokens``. Under MTP they differ by the
        acceptance rate."""
        self.h.log.begin_round(round_id=500, bs=6, rows=18)
        self.run_forward()
        self.h.ready_now()
        self.h.log.end_round()
        line = self.decode_lines()[0]
        self.assertIn("#rows: 18", line)
        self.assertNotIn("#tokens", line)

    # -- the SUMMARY's denominator (review R4) -----------------------------

    def test_MUTANT_14_the_summary_COUNTS_the_rounds_it_could_not_split(self):
        """A summary that silently drops withheld rounds reports a mean over
        the readable minority of a graph-covered boot and labels it the
        boot's."""
        from sglang.srt.debug_utils.rank_phase_summary import report, summarize

        self.h.log.begin_round(round_id=600, bs=6, rows=24)
        self.run_forward(compute_ms=10.0, waits=[("tp.all_reduce", 2.0)])
        self.h.ready_now()
        self.h.log.begin_round(round_id=601, bs=6, rows=24)
        self.run_forward(graphed=True, compute_ms=12.0)
        self.h.ready_now()
        self.h.log.end_round()
        lines = ["[2026-09-09 00:00:00 TP1] " + l for l in self.decode_lines()]
        summary = summarize(lines)
        self.assertEqual(summary["TP1"]["count"], 1)
        self.assertEqual(summary["TP1"]["withheld"], 1)
        self.assertEqual(
            summary["TP1"]["withheld_reasons"], {"graph-replay-no-event-nodes": 1}
        )
        text = report(summary)
        self.assertIn("WITHHELD 1/2 rounds", text)
        self.assertIn("graph-replay-no-event-nodes x1", text)

    def test_a_window_of_MOSTLY_withheld_rounds_is_named_not_evidence(self):
        from sglang.srt.debug_utils.rank_phase_summary import report, summarize

        for i in range(3):
            self.h.log.begin_round(round_id=700 + i, bs=6, rows=24)
            self.run_forward(graphed=True, compute_ms=12.0)
            self.h.ready_now()
        self.h.log.begin_round(round_id=799, bs=6, rows=24)
        self.run_forward(compute_ms=10.0, waits=[("tp.all_reduce", 2.0)])
        self.h.ready_now()
        self.h.log.end_round()
        lines = ["[2026-09-09 00:00:00 TP1] " + l for l in self.decode_lines()]
        text = report(summarize(lines))
        self.assertIn("WINDOW NOT EVIDENCE", text)

    def test_the_overhead_line_states_the_CONTENTION_counts_and_the_rate(self):
        self.h.log.OVERHEAD_ROUNDS = 2
        for i in range(4):
            self.h.log.begin_round(round_id=800 + i, bs=1, rows=1)
            self.run_forward()
            self.h.ready_now()
        self.h.log.end_round()
        over = [l for l in self.cap.lines if l.startswith("Decode rank clock overhead")]
        self.assertEqual(len(over), 1, over)
        self.assertIn("prefill armings refused 0", over[0])
        self.assertIn("decode rounds opened contended 0", over[0])
        self.assertIn("lines/s on this rank", over[0])

    # -- the WIRING, which no unit test of this module can reach ------------

    def test_the_funnel_CLOSES_the_round_for_every_non_decode_batch(self):
        """`end_round` with no caller is the defect MY-M1 proved survivable:
        the suite passed with `_retire_open` gutted. Assert the call site."""
        import inspect

        import sglang.srt.managers.scheduler as sched

        src = inspect.getsource(sched.Scheduler._run_batch_forward)
        self.assertIn("_drl.end_round()", src)
        self.assertIn("if not batch.forward_mode.is_decode():", src)
        # The boundary must sit OUTSIDE the is_decode branch, i.e. the
        # `_drl` lookup is at the method's own indent level (8 spaces).
        self.assertIn("\n        _drl = getattr(", src)

    def test_the_decode_log_is_FLUSHED_where_the_prefill_log_is_flushed(self):
        import inspect

        import sglang.srt.managers.scheduler_components.metrics_reporter as mr

        for fn, call in (
            (mr.SchedulerMetricsReporter.report_prefill_stats, "_drl.flush()"),
            (mr.SchedulerMetricsReporter.report_decode_stats, "_drl.flush()"),
            (
                mr.SchedulerMetricsReporter._maybe_log_idle_metrics,
                "_drl.end_round()",
            ),
        ):
            src = inspect.getsource(fn)
            self.assertIn('getattr(self, "decode_round_log", None)', src, fn.__qualname__)
            self.assertIn(call, src, fn.__qualname__)

    def test_the_flush_sites_survive_a_reporter_STAND_IN(self):
        """The three flush sites run BEFORE the logging-rank gate, and that
        pre-gate region is driven in tests by SimpleNamespace stand-ins that
        carry only the fields it reads. An unconditional attribute there is
        an AttributeError on every silent-rank test -- which is how this was
        found, not by reasoning about it."""
        from types import SimpleNamespace

        import sglang.srt.managers.scheduler_components.metrics_reporter as mr

        stub = SimpleNamespace(
            rank_prefill_log=mr.RankPrefillLog(),
            is_stats_logging_rank=False,
            current_scheduler_metrics_enabled=False,
            prefill_tokens_total=0,
        )
        mr.SchedulerMetricsReporter.report_prefill_stats(
            stub,
            batch=None,
            prefill_stats=SimpleNamespace(log_input_tokens=1, log_hit_tokens=0),
            can_run_cuda_graph=False,
        )
        mr.SchedulerMetricsReporter._maybe_log_idle_metrics(stub)

    def test_the_weightless_worker_decode_path_is_BRACKETED_too(self):
        """An unbracketed rank retires an empty round and emits no line, and
        a three-rank comparison silently missing a rank is worse than none."""
        import inspect

        import sglang.srt.model_executor.model_runner as mrun

        src = inspect.getsource(mrun.ModelRunner)
        # The graph-replay branch of the weightless worker path...
        head = src.split("return self._forward_weightless_worker")[0]
        tail = head.rsplit("if worker_can_run_graph:", 1)[-1]
        self.assertIn("_decode_round_segment", tail)
        # ...and the eager fallback right after it.
        self.assertIn(
            'with self._decode_round_segment("decode", graphed=False):\n'
            "                    return self._forward_weightless_worker",
            src,
        )

    def test_the_prefill_line_still_parses_byte_identically(self):
        prefill = (
            "[2026-08-06 19:15:53 TP0] Prefill rank batch, #new-token: 53, "
            "#cached-token: 0, #chunks: 1, gpu-ms: 111.0 (compute 32.8, wait 78.1) "
            "(wait by family: tp.all_reduce 62.5/129x, dcp.all_gather 5.3/16x)"
        )
        parsed = parse_rank_batch_line(prefill)
        self.assertEqual(parsed["rank"], "TP0")
        self.assertEqual(parsed["phase"], "Prefill")
        self.assertAlmostEqual(parsed["gpu_ms"], 111.0)
        self.assertAlmostEqual(parsed["compute_ms"], 32.8)
        self.assertAlmostEqual(parsed["wait_ms"], 78.1)
        self.assertEqual(parsed["wait_by_family"]["tp.all_reduce"], (62.5, 129))
        self.assertNotIn("round", parsed)

    def test_the_clock_backend_defaults_to_torch_and_is_the_only_cuda_seam(self):
        import inspect

        import sglang.srt.utils.collective_clock as cc

        src = inspect.getsource(cc)
        body = src.split("class TorchCudaBackend", 1)[1].split("@dataclasses", 1)[0]
        self.assertIn("torch.cuda.Event", body)
        self.assertIn("is_current_stream_capturing", body)
        after = src.split("class CollectiveClock:", 1)[1]
        self.assertNotIn("torch.cuda.Event(", after)
        self.assertNotIn("torch.cuda.is_current_stream_capturing", after)


if __name__ == "__main__":
    unittest.main()
