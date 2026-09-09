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
        self.h.log.begin_round(round_id=7, bs=1, tokens=1)
        self.run_forward(compute_ms=20.0, waits=[("tp.all_reduce", 5.0)])
        self.h.ready_now()
        self.h.log.begin_round(round_id=8, bs=1, tokens=1)
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
        self.h.log.begin_round(round_id=11, bs=1, tokens=1)
        self.run_forward()
        self.h.ready_now()
        self.assertEqual(self.decode_lines(), [], "round 11 emitted inside round 11")
        self.h.log.begin_round(round_id=12, bs=1, tokens=1)
        self.assertEqual(len(self.decode_lines()), 1)
        self.assertIn("#round: 11", self.decode_lines()[0])
        self.assertNotIn("#round: 12", self.decode_lines()[0])
        self.assertEqual(self.h.state.synchronize_calls, 0)

    def test_MUTANT_3_the_line_names_the_rank_as_a_FIELD_not_via_the_prefix(self):
        """The two Weg-2 groups write different log prefixes; a cross-rank
        join that depends on the formatter is not a join."""
        self.h.log.begin_round(round_id=1, bs=2, tokens=2)
        self.run_forward()
        self.h.ready_now()
        self.h.log.begin_round(round_id=2, bs=2, tokens=2)
        line = self.decode_lines()[0]
        self.assertIn("rank: 1,", line)
        # Parsed with NO TPn in the prefix at all.
        parsed = parse_rank_batch_line("[2026-09-09 00:00:00] " + line)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["rank"], "TP1")

    def test_MUTANT_4_a_graph_replayed_round_REFUSES_the_split_never_reports_zero(self):
        """A replayed collective records no span. An empty slot is not a
        wait of 0.0; it is an unknown wait."""
        self.h.log.begin_round(round_id=3, bs=6, tokens=24)
        self.run_forward(graphed=True, compute_ms=12.9, waits=())
        self.h.ready_now()
        self.h.log.begin_round(round_id=4, bs=6, tokens=24)
        line = self.decode_lines()[0]
        self.assertIn("split unavailable: graph-replay", line)
        self.assertIn("graphed-fwd 1/1", line)
        self.assertNotIn("wait 0.0", line)
        self.assertNotIn("compute", line)
        self.assertIsNone(parse_rank_batch_line("[x TP1] " + line))
        un = parse_unsplit_line("[2026-09-09 00:00:00 TP1] " + line)
        self.assertIsNotNone(un)
        self.assertEqual(un["reason"], "graph-replay")
        self.assertFalse(un["split_known"])
        self.assertAlmostEqual(un["gpu_ms"], 12.9, places=1)

    def test_ONE_grep_finds_the_prefill_and_the_decode_family(self):
        self.h.log.begin_round(round_id=5, bs=1, tokens=1)
        self.run_forward()
        self.h.ready_now()
        self.h.log.begin_round(round_id=6, bs=1, tokens=1)
        decode = self.decode_lines()[0]
        prefill = (
            "Prefill rank batch, #new-token: 53, #cached-token: 0, #chunks: 1, "
            "gpu-ms: 111.0 (compute 32.8, wait 78.1)"
        )
        for line in (decode, prefill):
            self.assertIn(" rank batch, ", line)

    def test_a_speculative_round_folds_draft_and_verify_into_ONE_line(self):
        self.h.log.begin_round(round_id=20, bs=6, tokens=24)
        self.run_forward("extend", compute_ms=6.0, waits=[("tp.all_reduce", 2.0)])
        self.run_forward(
            "target_verify", compute_ms=14.0, waits=[("tp.all_reduce", 4.0)]
        )
        self.h.ready_now()
        self.h.log.begin_round(round_id=21, bs=6, tokens=24)
        lines = self.decode_lines()
        self.assertEqual(len(lines), 1, lines)
        line = lines[0]
        self.assertIn("#fwd: 2", line)
        self.assertIn("bs: 6", line)
        self.assertIn("#tokens: 24", line)
        self.assertIn("gpu-ms: 20.0", line)
        self.assertIn("spec_draft:tp.all_reduce 2.0/1x", line)
        self.assertIn("spec_verify:tp.all_reduce 4.0/1x", line)

    def test_the_phase_prefix_KEEPS_the_dispatch_family_it_does_not_replace_it(self):
        self.h.log.begin_round(round_id=30, bs=1, tokens=4)
        self.run_forward(
            "target_verify",
            compute_ms=9.0,
            waits=[("tp.all_reduce", 1.0), ("dcp.all_gather", 2.0)],
        )
        self.h.ready_now()
        self.h.log.begin_round(round_id=31, bs=1, tokens=4)
        line = self.decode_lines()[0]
        self.assertIn("spec_verify:dcp.all_gather", line)
        self.assertIn("spec_verify:tp.all_reduce", line)

    def test_an_unreadable_round_BLOCKS_the_drain_and_never_reorders(self):
        self.h.log.begin_round(round_id=40, bs=1, tokens=1)
        self.h.hold()
        self.run_forward()
        # round 40's events stay unreadable
        self.h.log.begin_round(round_id=41, bs=1, tokens=1)
        self.run_forward()
        self.h.ready_now()
        self.h.log.begin_round(round_id=42, bs=1, tokens=1)
        lines = self.decode_lines()
        self.assertEqual(len(lines), 2)
        self.assertIn("#round: 40", lines[0])
        self.assertIn("#round: 41", lines[1])

    def test_a_decode_step_with_no_timed_forward_emits_NOTHING(self):
        """An empty round is not a zero round."""
        self.h.log.begin_round(round_id=50, bs=1, tokens=1)
        self.h.ready_now()
        self.h.log.begin_round(round_id=51, bs=1, tokens=1)
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
        self.h.log.begin_round(round_id=60, bs=1, tokens=1)
        self.h.log.begin_round(round_id=61, bs=1, tokens=1)
        self.assertEqual(self.decode_lines(), [])

    def test_the_wait_can_never_exceed_the_round(self):
        self.h.log.begin_round(round_id=70, bs=1, tokens=1)
        with self.h.log.segment("decode", False):
            with self.h.clock.span("tp.all_reduce"):
                self.h.state.advance(5.0)
        self.h.ready_now()
        self.h.log.begin_round(round_id=71, bs=1, tokens=1)
        parsed = parse_rank_batch_line("[x TP1] " + self.decode_lines()[0])
        self.assertGreaterEqual(parsed["compute_ms"], 0.0)
        self.assertLessEqual(parsed["wait_ms"], parsed["gpu_ms"] + 1e-6)

    def test_the_overhead_is_stated_ONCE_with_its_denominator(self):
        self.h.log.OVERHEAD_ROUNDS = 3
        for i in range(6):
            self.h.log.begin_round(round_id=100 + i, bs=1, tokens=1)
            self.run_forward()
            self.h.ready_now()
        self.h.log.begin_round(round_id=999, bs=1, tokens=1)
        over = [l for l in self.cap.lines if l.startswith("Decode rank clock overhead")]
        self.assertEqual(len(over), 1, over)
        self.assertIn("us/round host-side over", over[0])
        self.assertIn("% of the mean round gpu-ms", over[0])
        self.assertIn("Dropped rounds", over[0])

    def test_the_summary_parser_reads_the_decode_JOIN_KEYS(self):
        self.h.log.begin_round(round_id=412, bs=6, tokens=24)
        self.run_forward(compute_ms=31.4, waits=[("tp.all_reduce", 11.6)])
        self.h.ready_now()
        self.h.log.begin_round(round_id=413, bs=6, tokens=24)
        parsed = parse_rank_batch_line(
            "[2026-09-09 00:00:00 TP1] " + self.decode_lines()[0]
        )
        self.assertEqual(parsed["round"], 412)
        self.assertEqual(parsed["bs"], 6)
        self.assertEqual(parsed["phase"], "Decode")
        self.assertEqual(parsed["rank"], "TP1")

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
