"""Register #52 (26.09.): the collective clock's graph reader is opt-in.

Until rc2.1l every NF boot captured its decode graphs with the clock's
event-record nodes, bound K event sets per graph (x176: ``bound: 192 pairs x
8 event sets``, fnFL2h91v1: 288 pairs) and, before EVERY replay, swapped the
set through ``cudaGraphExecEventRecordNodeSetEvent`` (two host calls per
pair) and recorded a launch fence -- a measuring instrument inside the
production decode (20.09.: Runden/s 26,0 -> 24,4..27,1).

Contract pinned here:

* default (``SGLANG_DEBUG_COLLECTIVE_CLOCK_GRAPH_NODES`` unset) = OFF: no
  node, no event created, nothing bound, ZERO SetEvent calls and zero event
  records per replay;
* ON = today's behaviour, same swap count as test_decode_graph_event_sets_0920;
* a graphed round with the reader off is refused BY NAME
  (``graph-replay-reader-off``), parseable by rank_phase_summary;
* one boot line names the state; the AR census gets a named warning.

Hermetic: the cudart seam is the fake of test_decode_graph_event_sets_0920.
Run with ``CUDA_VISIBLE_DEVICES=''``. The first test is red on d1c7094ba6.
"""
from __future__ import annotations

import logging
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_decode_graph_event_sets_0920 import (  # noqa: E402
    Backend,
    FakeBinder,
    FakeGraph,
    State,
)

from sglang.srt.debug_utils.rank_phase_summary import parse_unsplit_line  # noqa: E402
from sglang.srt.utils.collective_clock import CollectiveClock  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=2, suite="stage-a-cpu")

ENV = "SGLANG_DEBUG_COLLECTIVE_CLOCK_GRAPH_NODES"


class CountingBackend(Backend):
    """Counts every event record (fence or pair) the clock issues."""

    def __init__(self, st):
        super().__init__(st)
        self.records = 0

    def event(self):
        ev = super().event()
        orig = ev.record

        def record():
            self.records += 1
            orig()

        ev.record = record
        return ev


class H:
    def __init__(self, ring=2):
        self.st = State()
        self.be = CountingBackend(self.st)
        # No graph_nodes argument: the switch is read from the env, exactly
        # as the process-global clock does.
        self.clock = CollectiveClock(backend=self.be, graph_ring=ring)
        self.graph = FakeGraph()
        self.be.binder = FakeBinder(self.graph, self.be.by_handle)

    def capture(self, key, fams):
        with self.clock.capture_scope(key, phase="spec_verify"):
            self.st.capturing = True
            try:
                for f in fams:
                    if self.clock.armed:
                        with self.clock.span(f):
                            pass
            finally:
                self.st.capturing = False
        nodes = self.clock.captured_graph(key)
        if nodes is not None:
            n = 100
            for pre, post, _ in nodes.pairs:
                self.graph.nodes[n] = pre.handle
                self.graph.nodes[n + 1] = post.handle
                n += 2
        return nodes

    def cycle(self, replays=5):
        """capture -> bind -> N replays, the runner's order."""
        self.capture("g", ["all_reduce", "a2a"])
        bound = self.clock.bind_graph("g", self.graph)
        for _ in range(replays):
            self.clock.note_graph_replay("g")
        return bound


def _env_unset():
    env = dict(os.environ)
    env.pop(ENV, None)
    return mock.patch.dict(os.environ, env, clear=True)


class ReaderSwitchTest(unittest.TestCase):
    def test_default_off_no_set_event_no_event_no_node(self):
        """RED on d1c7094ba6: 20 SetEvent calls, 5 swaps, 22 events."""
        with _env_unset():
            h = H(ring=2)
            bound = h.cycle(replays=5)
        self.assertEqual(h.be.binder.set_calls, 0, "a replay swapped event sets")
        self.assertEqual(h.clock._swap_calls, 0)
        self.assertFalse(bound, "an event set was bound")
        self.assertIsNone(h.clock.captured_graph("g"), "the graph carries nodes")
        self.assertEqual(h.be.created, 0, "the off form created events")
        self.assertEqual(h.be.records, 0, "the off form recorded a fence/pair")
        self.assertEqual(h.clock.graph_node_counts[:2], (0, 0))

    def test_explicit_off_via_env(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_DEBUG_COLLECTIVE_CLOCK_GRAPH_NODES.override(False):
            h = H(ring=2)
            h.cycle(replays=3)
        self.assertEqual(h.be.binder.set_calls, 0)
        self.assertEqual(h.clock._graph_reader_off_captures, 1)

    def test_on_is_todays_behaviour(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_DEBUG_COLLECTIVE_CLOCK_GRAPH_NODES.override(True):
            h = H(ring=2)
            bound = h.cycle(replays=5)
        self.assertTrue(bound)
        # identical to test_swap_allocates_nothing_and_is_counted:
        # 5 swaps x 2 pairs x 2 events
        self.assertEqual(h.be.binder.set_calls, 5 * 4)
        self.assertEqual(h.clock._swap_calls, 5)
        self.assertEqual(h.clock.graph_node_counts[:2], (1, 4))

    def test_state_is_fixed_at_first_capture(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_DEBUG_COLLECTIVE_CLOCK_GRAPH_NODES.override(False):
            h = H(ring=2)
            h.capture("a", ["all_reduce"])
        with envs.SGLANG_DEBUG_COLLECTIVE_CLOCK_GRAPH_NODES.override(True):
            h.capture("b", ["all_reduce"])  # same process, same form
        self.assertIsNone(h.clock.captured_graph("b"))

    def test_graphed_round_refused_by_name_when_off(self):
        with _env_unset():
            h = H(ring=2)
            h.capture("g", ["all_reduce"])
            span = h.clock.open_round()
            span.graph_replayed = True
            h.clock.note_graph_replay("g")
            h.clock.close_round(span)
            r = h.clock.harvest_round(span)
        self.assertIsNone(r.wait_ms)
        self.assertIsNone(r.compute_ms)
        self.assertEqual(r.split_refused, "graph-replay-reader-off")
        line = (
            "[2026-09-26 12:00:00 TP1] Decode rank batch, rank: 1, #round: 7, t: 1.0, bs: 1, #rows: 3, "
            "#fwd: 1, gpu-ms: 30.0 (split unavailable: "
            f"{r.split_refused}, graphed-fwd 1/1)"
        )
        parsed = parse_unsplit_line(line)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["reason"], "graph-replay-reader-off")

    def test_undecided_clock_keeps_the_structural_reason(self):
        """No capture scope ever entered: the old reason, unchanged."""
        with _env_unset():
            h = H(ring=2)
            span = h.clock.open_round()
            span.graph_replayed = True
            h.clock.close_round(span)
            r = h.clock.harvest_round(span)
        self.assertEqual(r.split_refused, "graph-replay-no-event-nodes")

    def test_one_boot_line_names_the_state_and_the_census(self):
        from sglang.srt.environ import envs

        records = []

        class Cap(logging.Handler):
            def emit(self, record):
                records.append((record.levelno, record.getMessage()))

        lg = logging.getLogger("sglang.srt.utils.collective_clock")
        handler = Cap()
        lg.addHandler(handler)
        old = lg.level
        lg.setLevel(logging.INFO)
        try:
            with _env_unset(), envs.SGLANG_WEG2_AR_ROUND_CENSUS.override(True):
                h = H(ring=2)
                h.capture("a", ["all_reduce"])
                h.capture("b", ["all_reduce"])
        finally:
            lg.removeHandler(handler)
            lg.setLevel(old)
        state = [m for _, m in records if "graph reader" in m]
        self.assertEqual(len(state), 2, records)  # one state line + one census warning
        self.assertIn("graph reader OFF", state[0])
        self.assertIn("graph-replay-reader-off", state[0])
        self.assertTrue(
            any(lv == logging.WARNING and "BARLINK-ROUND-CENSUS" in m for lv, m in records)
        )


if __name__ == "__main__":
    unittest.main()
