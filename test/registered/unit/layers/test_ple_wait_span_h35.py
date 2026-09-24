"""fnFL2 H35: ``ple.wait`` -- the PLE layer's join on its HMM gather as a
family of the per-rank decode clock.

Hermetic, CPU only: the clock is the fake of the #1241/H23 tests (events are
plain numbers on a fake timeline, ``synchronize`` raises), the stream that
waits is a fake whose ``wait_stream`` advances the timeline by the stall.
Run with ``CUDA_VISIBLE_DEVICES=''``.

The numbers are x144's code@10k on D-TP0 (24.09.): verify compute median
16.4 ms against a floor of 10.0 (mid2/needle2, fnFA23 code@10k) -- a PLE
stall of 6.4 ms per round that today sits in ``compute``.

Cases:
* switch on, round open: the stall is its own family
  ``spec_verify:ple.wait`` on ``Decode rank batch``, ``compute`` drops by
  exactly the stall, ``gpu-ms`` is unchanged (the Zaehlprobe of the split);
* switch off: byte-identical split to before (no family, compute carries it);
* unarmed clock: nothing is recorded even with the switch on;
* DECODE-ROUND-COST carries ``ple_ms`` (``-`` when not recorded, never 0.0);
* the source wiring: the PLE layer's join and its non-prefetch lookup go
  through the helper (a writer-less family would print nothing forever).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import logging
import pathlib
import unittest

from sglang.srt.environ import envs
from sglang.srt.layers import ple_wait_span as pws
from sglang.srt.managers.scheduler_components.decode_round_log import DecodeRoundLog
from sglang.srt.managers.scheduler_components.wake_round_census import WakeRoundCensus
from sglang.srt.utils.collective_clock import ClockBackend, CollectiveClock

PLE_STALL_MS = 6.4
FLOOR_MS = 10.0
FETCH_MS = 8.8
AR_MS = 8.2


class _State:
    def __init__(self) -> None:
        self.now = 0.0
        self.syncs = 0


class _Event:
    def __init__(self, state: _State) -> None:
        self._state = state
        self.t = None

    def record(self) -> None:
        self.t = self._state.now

    def query(self) -> bool:
        return self.t is not None

    def elapsed_time(self, other: "_Event") -> float:
        return other.t - self.t

    def synchronize(self) -> None:  # pragma: no cover - must never run
        self._state.syncs += 1
        raise AssertionError("the instrument synchronized the device")


class _Backend(ClockBackend):
    def __init__(self, state: _State) -> None:
        self.state = state

    def event(self):
        return _Event(self.state)

    def is_capturing(self) -> bool:
        return False


class _Stream:
    """The main stream: its join on the prefetch stream stalls ``stall_ms``."""

    def __init__(self, state: _State, stall_ms: float) -> None:
        self.state = state
        self.stall_ms = stall_ms
        self.joined = []

    def wait_stream(self, other) -> None:
        self.joined.append(other)
        self.state.now += self.stall_ms


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class PleWaitSpanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _State()
        self.clock = CollectiveClock(backend=_Backend(self.state))
        self.log = DecodeRoundLog(clock=self.clock, rank=0)
        self.cap = _Capture()
        for name in (
            "sglang.srt.managers.scheduler_components.decode_round_log",
            "sglang.srt.managers.scheduler_components.wake_round_census",
        ):
            lg = logging.getLogger(name)
            lg.addHandler(self.cap)
            lg.setLevel(logging.INFO)
            self.addCleanup(lg.removeHandler, self.cap)
        self.next_round = 500
        self.side = object()

    def _round(self) -> _Stream:
        """One x144 code@10k verify round: fetch + all-reduce spans, the PLE
        join on the prefetch stream, the compute floor."""
        rid = self.next_round
        self.next_round += 1
        main = _Stream(self.state, PLE_STALL_MS)
        self.log.begin_round(round_id=rid, bs=1, rows=4)
        with self.log.segment("target_verify", False):
            with self.clock.span("pool.fetch"):
                self.state.now += FETCH_MS
            pws.wait_for_ple_prefetch(self.side, clock=self.clock, current_stream=main)
            with self.clock.span("tp.all_reduce"):
                self.state.now += AR_MS
            self.state.now += FLOOR_MS
        return main

    def _rank_lines(self):
        return [l for l in self.cap.lines if l.startswith("Decode rank batch")]

    def test_switch_on_names_the_stall_and_keeps_gpu_ms(self):
        with envs.SGLANG_DEBUG_DECODE_PLE_WAIT.override(True):
            main = self._round()
            self._round()
        self.log.end_round()
        self.assertEqual(main.joined, [self.side])  # the join itself still happens
        line = self._rank_lines()[0]
        gpu = FETCH_MS + PLE_STALL_MS + AR_MS + FLOOR_MS
        self.assertIn("gpu-ms: %.1f (compute %.1f," % (gpu, FLOOR_MS), line)
        self.assertIn("spec_verify:ple.wait %.1f/1x" % PLE_STALL_MS, line)
        self.assertEqual(self.state.syncs, 0)

    def test_switch_off_is_the_old_split(self):
        with envs.SGLANG_DEBUG_DECODE_PLE_WAIT.override(False):
            main = self._round()
            self._round()
        self.log.end_round()
        self.assertEqual(main.joined, [self.side])
        line = self._rank_lines()[0]
        gpu = FETCH_MS + PLE_STALL_MS + AR_MS + FLOOR_MS
        self.assertIn(
            "gpu-ms: %.1f (compute %.1f," % (gpu, FLOOR_MS + PLE_STALL_MS), line
        )
        self.assertNotIn("ple.wait", line)

    def test_unarmed_clock_records_nothing(self):
        """Outside a round/capture the span must lay nothing (the dispatch
        sites' own guard); with the switch on a stray call is still free."""
        main = _Stream(self.state, PLE_STALL_MS)
        calls = []
        real_span = self.clock.span

        def spy(*a, **k):  # ``span``'s contract: the caller checked ``armed``
            calls.append(a)
            return real_span(*a, **k)

        self.clock.span = spy
        with envs.SGLANG_DEBUG_DECODE_PLE_WAIT.override(True):
            self.assertFalse(self.clock.armed)
            pws.wait_for_ple_prefetch(self.side, clock=self.clock, current_stream=main)
        self.assertEqual(calls, [])
        self.assertEqual(main.joined, [self.side])
        self.assertIsNone(self.clock._slot)
        self.assertEqual(self.clock._pool, [])

    def test_round_cost_carries_ple_ms_and_never_a_fake_zero(self):
        census = WakeRoundCensus(rank=0)
        census.arm(wake_mono=None)
        census.note_open(round_id=1, mono=1.0)
        census.note_open(round_id=2, mono=1.1)
        on = census.on_round(
            round_id=1, gpu_ms=33.4, compute_ms=FLOOR_MS,
            families={"spec_verify:pool.fetch": FETCH_MS,
                      "spec_verify:ple.wait": PLE_STALL_MS,
                      "spec_verify:tp.all_reduce": AR_MS},
            graphed=True, now_mono=1.2)
        off = census.on_round(
            round_id=2, gpu_ms=33.4, compute_ms=FLOOR_MS + PLE_STALL_MS,
            families={"spec_verify:pool.fetch": FETCH_MS,
                      "spec_verify:tp.all_reduce": AR_MS},
            graphed=True, now_mono=1.2)
        self.assertAlmostEqual(on.ple_ms, PLE_STALL_MS)
        self.assertTrue(on.line().endswith("ple_ms=6.4"), on.line())
        self.assertIsNone(off.ple_ms)
        self.assertTrue(off.line().endswith("ple_ms=-"), off.line())
        # the existing fields keep their place (H23 readers match substrings)
        self.assertIn("compute_ms=10.0 fetch_ms=8.8 allreduce_ms=8.2", on.line())

    def test_the_ple_layer_routes_its_waits_through_the_helper(self):
        """Source wiring (qwen4_exp is not importable without a device
        stack): the prefetch join calls ``wait_for_ple_prefetch`` and no bare
        ``wait_stream`` is left in it; the non-prefetch lookup sits under
        ``ple_wait_scope``."""
        src = pathlib.Path(pws.__file__).resolve().parents[1] / "models" / "qwen4_exp.py"
        tree = ast.parse(src.read_text())
        funcs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpPLELayer":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        funcs[item.name] = ast.unparse(item)
        consume = funcs["_consume_prefetched_embeddings"]
        self.assertIn("wait_for_ple_prefetch(self._prefetch_stream)", consume)
        self.assertNotIn(".wait_stream(", consume)
        forward = funcs["forward"]
        self.assertRegex(
            forward,
            r"with ple_wait_scope\(\):\s+embeddings = self\.ple_embedding\(batch, forward_batch\)",
        )


if __name__ == "__main__":
    unittest.main()
