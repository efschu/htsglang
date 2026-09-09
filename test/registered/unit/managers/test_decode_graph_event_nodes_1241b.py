"""#1241b: the decode compute/wait split survives a CUDA-graph REPLAY.

Hermetic. No CUDA, no GPU, no capture: the device sits behind
``ClockBackend`` (utils/collective_clock.py) and this file adds the second
half of that seam -- a fake STREAM-CAPTURE and GRAPH-REPLAY layer that
implements exactly the three CUDA properties the design leans on and nothing
else. Run with ``CUDA_VISIBLE_DEVICES=''``.

    1. an event recorded while the stream is CAPTURING stamps no time; it
       becomes a NODE of the graph (``FakeEvent.record`` under
       ``state.capturing``);
    2. a REPLAY re-executes those nodes in order and stamps them, OVERWRITING
       whatever the previous replay left (``FakeGraphLayer.replay``);
    3. ``elapsed_time`` between two such nodes is only meaningful once the
       replay that executed them has completed (``query``), and reading it is
       never allowed to block.

Named red-first: each test name states the WRONG behaviour it exists to
catch. The failure mode of a timing instrument is not a crash -- it is a
plausible number -- so four tests carry mutants, all on the DANGER
direction: allocate on the replay path, price the whole replay as wait,
share one event pair across two regions, and read timestamps a later replay
has already overwritten.
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
# The fake device + the fake capture/replay layer. ONE adapter.
# ---------------------------------------------------------------------------


class FakeState:
    def __init__(self) -> None:
        self.now = 0.0
        #: An event is readable once its stamp is <= this. "Immediately
        #: readable" is the ADVERSARIAL default: a clock that read the round
        #: it sits inside would succeed, so the deferral tests hold events
        #: back explicitly rather than relying on the default to hide a bug.
        self.readable_from = float("inf")
        self.capturing = False
        self.synchronize_calls = 0
        #: Called on every ``query()``, with the number of queries so far.
        #: THE FORWARD THREAD, modelled. The read loop is not atomic against
        #: it on the real device either, and a hook is the only way a
        #: single-threaded fake can put a replay INSIDE the loop rather than
        #: politely before or after it.
        self.on_query = None
        self.queries = 0

    def advance(self, ms: float) -> None:
        self.now += float(ms)


class FakeEvent:
    def __init__(self, state: FakeState) -> None:
        self._state = state
        self.t = None
        self.records = 0
        self.became_node = False

    def record(self) -> None:
        self.records += 1
        if self._state.capturing:
            # CUDA fact 1: a record on a capturing stream stamps nothing. It
            # becomes an event-record node of the graph being built.
            self.became_node = True
            self.t = None
            return
        self.t = self._state.now

    def stamp(self, t: float) -> None:
        """What a REPLAY of this node does. Overwrites, every time."""
        self.t = t

    def query(self) -> bool:
        self._state.queries += 1
        hook = self._state.on_query
        if hook is not None:
            hook(self._state.queries)
        return self.t is not None and self.t <= self._state.readable_from

    def elapsed_time(self, other: "FakeEvent") -> float:
        assert self.t is not None and other.t is not None, (
            "elapsed_time on an unstamped event: the clock read a node the "
            "device had not executed"
        )
        if not self._complete() or not other._complete():
            # CUDA fact 3, the failure half: cudaEventElapsedTime returns
            # cudaErrorNotReady when either event has not completed, and
            # torch raises that as RuntimeError. Reachable here only for a
            # node a replay re-recorded BETWEEN the clock's query and its
            # read -- which is precisely the window the two-sided generation
            # check exists for.
            raise RuntimeError("cudaErrorNotReady")
        return other.t - self.t

    def _complete(self) -> bool:
        return self.t is not None and self.t <= self._state.readable_from

    def synchronize(self) -> None:  # pragma: no cover - must never be called
        self._state.synchronize_calls += 1
        raise AssertionError("the decode round clock synchronized the device")


class FakeBackend(ClockBackend):
    def __init__(self, state: FakeState) -> None:
        self.state = state
        self.events_created = 0
        self.materialized = 0

    def event(self):
        self.events_created += 1
        return FakeEvent(self.state)

    def is_capturing(self) -> bool:
        return self.state.capturing

    def materialize(self, event) -> None:
        assert not self.state.capturing, (
            "an event was materialized INSIDE the capture; the pairs a "
            "capture uses are created before it"
        )
        self.materialized += 1
        event.record()


class Harness:
    def __init__(self, rank: int = 1) -> None:
        self.state = FakeState()
        self.backend = FakeBackend(self.state)
        self.clock = CollectiveClock(backend=self.backend)
        self.log = DecodeRoundLog(clock=self.clock, rank=rank)

    # -- the fake graph layer -------------------------------------------

    def capture(self, key, regions, phase=None):
        """Capture one graph. ``regions`` is a list of family names, in the
        order the captured forward issues them (repeats are repeats).

        ORDER IS THE REAL RUNNER'S ORDER: ``capture_scope`` is entered while
        the stream is NOT yet capturing (decode_cuda_graph_runner wraps
        ``backend.capture_one``, and the stream starts capturing inside it),
        so the scope's pre-materialization runs outside the capture -- which
        ``FakeBackend.materialize`` asserts, because materializing a lazy
        ``torch.cuda.Event`` inside a capture is exactly the allocation the
        design promises not to make."""
        with self.clock.capture_scope(key, phase=phase):
            self.state.capturing = True
            try:
                for family in regions:
                    with self.clock.span(family):
                        # Device work inside the region stamps nothing under
                        # capture; the shape is what is being recorded.
                        pass
            finally:
                self.state.capturing = False
        return self.clock.captured_graph(key)

    def replay(self, key, per_region_ms):
        """CUDA fact 2: re-execute the nodes, in order, overwriting them."""
        nodes = self.clock.captured_graph(key)
        assert nodes is not None, "replayed a graph that carries no nodes"
        for (pre, post, _family), ms in zip(nodes.pairs, per_region_ms):
            pre.stamp(self.state.now)
            self.state.advance(ms)
            post.stamp(self.state.now)

    def ready_now(self) -> None:
        self.state.readable_from = float("inf")

    def hold(self) -> None:
        self.state.readable_from = self.state.now - 1e-9


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class GraphEventNodeTest(unittest.TestCase):
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

    def graph_round(self, round_id, key, per_region_ms, extra_compute_ms=6.0,
                    bs=6, rows=24, category="decode", declare=True):
        """One decode round whose single forward is a graph replay."""
        self.h.log.begin_round(round_id=round_id, bs=bs, rows=rows)
        with self.h.log.segment(category, graphed=True):
            if declare:
                self.h.clock.note_graph_replay(key)
            if self.h.clock.captured_graph(key) is not None:
                self.h.replay(key, per_region_ms)
            self.h.state.advance(extra_compute_ms)

    def eager_round(self, round_id, waits, compute_ms=20.0, bs=1, rows=1):
        self.h.log.begin_round(round_id=round_id, bs=bs, rows=rows)
        with self.h.log.segment("decode", graphed=False):
            for family, ms in waits:
                with self.h.clock.span(family):
                    self.h.state.advance(ms)
                    compute_ms -= ms
            self.h.state.advance(max(compute_ms, 0.0))

    def lines(self):
        return [ln for ln in self.cap.lines if ln.startswith("Decode rank batch")]

    # -- the tests -------------------------------------------------------

    def test_a_captured_graph_replayed_three_times_yields_three_split_lines(self):
        """The whole point. Before #1241b every one of these was
        `split unavailable: graph-replay` and the ladder had no per-rank
        wait on the form it actually runs."""
        self.h.capture("k8", ["tp.all_reduce", "dcp.all_gather", "tp.all_reduce"])
        for i, ms in enumerate([[1.0, 2.0, 1.5], [1.2, 2.2, 1.3], [0.8, 2.4, 1.1]]):
            self.graph_round(100 + i, "k8", ms, extra_compute_ms=6.0)
        self.h.ready_now()
        self.h.log.end_round()

        got = self.lines()
        self.assertEqual(len(got), 3, got)
        expect = [(4.5, 10.5), (4.7, 10.7), (4.3, 10.3)]
        for line, (wait, span) in zip(got, expect):
            parsed = parse_rank_batch_line("[2026-09-09 00:00:00 TP1] " + line)
            self.assertIsNotNone(parsed, line)
            self.assertAlmostEqual(parsed["wait_ms"], wait, places=1)
            self.assertAlmostEqual(parsed["gpu_ms"], span, places=1)
            self.assertAlmostEqual(parsed["compute_ms"], span - wait, places=1)
            self.assertNotIn("split unavailable", line)

    def test_the_per_family_waits_are_the_replay_s_own_values_not_the_bracket(self):
        """MUTANT-1, the danger direction: attribute the WHOLE replay to
        wait. The families then sum to the round and compute reads 0.0 --
        which is a number, and looks like a measurement."""
        self.h.capture("k8", ["tp.all_reduce", "dcp.all_gather", "tp.all_reduce"])
        self.graph_round(1, "k8", [1.0, 2.0, 1.5], extra_compute_ms=6.0)
        self.h.ready_now()
        self.h.log.end_round()

        line = self.lines()[0]
        self.assertIn("wait by family: tp.all_reduce 2.5/2x", line)
        self.assertIn("dcp.all_gather 2.0/1x", line)
        parsed = parse_rank_batch_line("[x TP1] " + line)
        # 10.5 ms of bracket, 4.5 ms of it in collectives. The mutant makes
        # these two equal; the asymmetry is what it cannot survive.
        self.assertAlmostEqual(parsed["wait_ms"], 4.5, places=1)
        self.assertGreater(parsed["compute_ms"], parsed["wait_ms"])

    def test_every_region_of_a_graph_owns_its_own_pair(self):
        """MUTANT-2: one pair per FAMILY instead of per REGION. The graph
        still replays and the line still prints a number -- a wrong one,
        spanning the compute between the first and the last occurrence."""
        nodes = self.h.capture(
            "k8", ["tp.all_reduce", "dcp.all_gather", "tp.all_reduce"]
        )
        self.assertEqual(len(nodes.pairs), 3)
        ids = [id(e) for pre, post, _ in nodes.pairs for e in (pre, post)]
        self.assertEqual(len(set(ids)), 6, "an event pair is shared by two regions")
        self.assertTrue(all(pre.became_node and post.became_node
                            for pre, post, _ in nodes.pairs))

    def test_the_replay_path_allocates_no_event(self):
        """MUTANT-3: create the pair inside the round. On the real device
        that is a cudaEventCreate on the hot stream in the middle of the span
        being measured -- the instrument then measures itself."""
        self.h.capture("k8", ["tp.all_reduce", "tp.all_reduce"])
        self.graph_round(1, "k8", [1.0, 1.0])
        self.h.ready_now()
        self.graph_round(2, "k8", [1.0, 1.0])
        before = self.h.backend.events_created
        for i in range(3, 8):
            self.graph_round(i, "k8", [1.0, 1.0])
        self.assertEqual(
            self.h.backend.events_created, before,
            "the replay path created an event",
        )
        self.assertEqual(self.h.state.synchronize_calls, 0)

    def test_a_round_whose_nodes_a_later_replay_overwrote_is_refused_by_name(self):
        """MUTANT-4, the quietest of the four: read the nodes anyway. They
        hold a plausible number -- somebody else's round."""
        self.h.capture("k8", ["tp.all_reduce"])
        # Round 1's events are held unreadable PAST round 2's begin-round
        # flush, so round 1 is still pending when round 2 replays the same
        # graph and overwrites its nodes. Only then does everything become
        # readable -- the adversarial order, because the friendly one (round
        # 1 read at round 2's boundary, before the overwrite) succeeds and
        # is separately pinned by the three-replays test above.
        self.h.hold()
        self.graph_round(1, "k8", [1.0])
        self.graph_round(2, "k8", [9.0])
        self.h.ready_now()
        self.h.log.end_round()

        got = self.lines()
        self.assertEqual(len(got), 2, got)
        un = parse_unsplit_line("[2026-09-09 00:00:00 TP1] " + got[0])
        self.assertIsNotNone(un, got[0])
        self.assertEqual(un["reason"], "graph-replay-nodes-overwritten")
        self.assertFalse(un["split_known"])
        self.assertNotIn("wait 0.0", got[0])
        # and the round that DID own the nodes is still split.
        self.assertIsNotNone(parse_rank_batch_line("[x TP1] " + got[1]))
        counts = self.h.clock.graph_node_counts
        self.assertEqual(counts[3], 1, "the stale read was not counted")

    def test_a_graph_captured_without_nodes_names_what_is_missing(self):
        """The reason survives -- but `graph-replay` claimed the MECHANISM
        was the obstacle, which since #1241b is false. It has to name the
        missing thing instead, or a reader concludes graphs cannot be split."""
        self.graph_round(1, "never-captured", [], declare=True)
        self.h.ready_now()
        self.h.log.end_round()
        un = parse_unsplit_line("[2026-09-09 00:00:00 TP1] " + self.lines()[0])
        self.assertEqual(un["reason"], "graph-replay-no-event-nodes")
        self.assertIn("graphed-fwd 1/1", self.lines()[0])

    def test_an_unexecuted_node_is_refused_not_blocked_on(self):
        """A round is read one round late; a node the device has not reached
        is a refusal, never a wait and never a synchronize."""
        self.h.capture("k8", ["tp.all_reduce"])
        self.h.log.begin_round(round_id=1, bs=1, rows=1)
        with self.h.log.segment("decode", graphed=True):
            self.h.clock.note_graph_replay("k8")
            # The bracket completes but the node inside it does not: the
            # replay's own end is readable while a node is still unstamped.
            self.h.state.advance(5.0)
        self.h.ready_now()
        self.h.log.end_round()
        un = parse_unsplit_line("[2026-09-09 00:00:00 TP1] " + self.lines()[0])
        self.assertEqual(un["reason"], "graph-replay-nodes-unread")
        self.assertEqual(self.h.state.synchronize_calls, 0)
        self.assertEqual(self.h.clock.graph_node_counts[4], 1)

    def test_a_replayed_verify_spells_its_families_the_way_an_eager_one_does(self):
        """Two spellings of one family is the quiet way a decomposition stops
        being comparable across the graphed and the eager arm."""
        self.h.capture("kv", ["tp.all_reduce"], phase="spec_verify")
        self.graph_round(1, "kv", [2.0], category="target_verify")
        self.h.ready_now()
        self.h.log.end_round()
        self.assertIn("spec_verify:tp.all_reduce 2.0/1x", self.lines()[0])

    def test_the_second_capture_creates_no_event_inside_the_capture(self):
        """The pairs are pre-created. The first capture of a process has no
        hint and grows the pool; every later one must not allocate inside."""
        first = self.h.capture("k4", ["tp.all_reduce"] * 5)
        second = self.h.capture("k8", ["tp.all_reduce"] * 5)
        self.assertEqual(first.late_created, 10)
        self.assertEqual(second.late_created, 0)
        self.assertGreaterEqual(self.h.backend.materialized, 10)

    def test_a_recapture_of_one_key_replaces_the_destroyed_graph_s_nodes(self):
        """Pairs of a graph that no longer exists still hold plausible
        timestamps."""
        first = self.h.capture("k8", ["tp.all_reduce", "tp.all_reduce"])
        second = self.h.capture("k8", ["tp.all_reduce"])
        self.assertIsNot(first, second)
        self.assertIs(self.h.clock.captured_graph("k8"), second)
        self.assertEqual(self.h.clock.graph_node_counts[0], 1)

    def test_a_nested_capture_is_refused_not_merged(self):
        self.h.state.capturing = True
        with self.h.clock.capture_scope("outer"):
            with self.h.clock.capture_scope("inner"):
                with self.h.clock.span("tp.all_reduce"):
                    pass
        self.h.state.capturing = False
        self.assertIsNone(self.h.clock.captured_graph("inner"))
        self.assertEqual(len(self.h.clock.captured_graph("outer").pairs), 1)

    def test_the_overhead_line_prints_the_node_denominator(self):
        self.h.capture("k8", ["tp.all_reduce", "dcp.all_gather"])
        self.h.log.OVERHEAD_ROUNDS = 2
        for i in range(4):
            self.graph_round(i, "k8", [1.0, 1.0])
        self.h.ready_now()
        self.h.log.end_round()
        over = [ln for ln in self.cap.lines if ln.startswith("Decode rank clock overhead")]
        self.assertEqual(len(over), 1, over)
        self.assertIn("1 graphs carry 4 nodes (4 created inside a capture)", over[0])
        self.assertIn("overwritten by a later replay", over[0])
        self.assertIn("not yet complete", over[0])
        self.assertIn("rounds that replayed one graph twice", over[0])


class ConcurrentReplayTest(unittest.TestCase):
    """The read is NOT atomic against the forward thread (review finding 2).

    ``harvest_round`` runs on the SCHEDULER thread, out of the metrics
    flush; ``note_graph_replay`` and the graph launch run on the FORWARD
    thread. Between the clock's generation check and its last ``query()``
    lie ~two host calls per pair, and the overlap scheduler routinely has
    the next replay of the same key already queued. A check taken only
    BEFORE the read therefore proves nothing about the read.

    Both tests drive that window explicitly, through ``FakeState.on_query``:
    the replay lands between pair 1 and pair 2, never politely outside the
    loop. The friendly orders are already pinned by ``GraphEventNodeTest``.
    """

    def setUp(self) -> None:
        self.h = Harness()
        self.cap = _Capture()
        self.logger = logging.getLogger(
            "sglang.srt.managers.scheduler_components.decode_round_log"
        )
        self.logger.addHandler(self.cap)
        self.logger.setLevel(logging.INFO)
        self.addCleanup(self.logger.removeHandler, self.cap)

    def lines(self):
        return [ln for ln in self.cap.lines if ln.startswith("Decode rank batch")]

    def _pending_round(self, key, per_region_ms):
        """One graphed round, left unread: the flush happens in the test."""
        self.h.log.begin_round(round_id=1, bs=6, rows=24)
        with self.h.log.segment("decode", graphed=True):
            self.h.clock.note_graph_replay(key)
            self.h.replay(key, per_region_ms)
            self.h.state.advance(6.0)

    def test_a_replay_landing_MID_read_is_refused_not_averaged_in(self):
        """MUTANT-5, and the quietest defect in this file: check the
        generation ONCE, before the read. The replay that lands inside the
        loop re-stamps the early nodes; the loop then sums pair 1 from
        replay N+1 and pair 2 from replay N and prints their mixture as a
        measurement. Nothing raises, and the stale counter -- which the boot
        ticket leans on to decide whether the instrument works -- stays 0,
        so the boot would report the trap as a clean read."""
        self.h.capture("k8", ["tp.all_reduce", "dcp.all_gather"])
        self._pending_round("k8", [1.0, 2.0])

        def land_a_replay(n):
            # query 1 = the round bracket's end, 2 = pair 1's post,
            # 3 = pair 2's post. Fire on 3: pair 1 has been read, pair 2
            # has not.
            if n != 3:
                return
            self.h.state.on_query = None
            self.h.clock.note_graph_replay("k8")
            self.h.replay("k8", [50.0, 60.0])

        self.h.state.on_query = land_a_replay
        self.h.log.end_round()

        un = parse_unsplit_line("[2026-09-09 00:00:00 TP1] " + self.lines()[0])
        self.assertIsNotNone(un, self.lines()[0])
        self.assertEqual(un["reason"], "graph-replay-nodes-overwritten")
        self.assertEqual(
            self.h.clock.graph_node_counts[3], 1, "the mid-read replay was not counted"
        )
        self.assertEqual(self.h.state.synchronize_calls, 0)

    def test_a_node_re_recorded_mid_read_is_a_refusal_not_an_exception(self):
        """MUTANT-6: let ``cudaErrorNotReady`` out. ``elapsed_time`` raises
        RuntimeError for an event a concurrent replay re-recorded and the
        device has not reached; uncaught it rides the flush into the
        scheduler tick, on a form that emits ~5695 graphed rounds per rank
        per 23 minutes. A crash is not the honest form of 'unknown'."""
        self.h.capture("k8", ["tp.all_reduce", "dcp.all_gather"])
        self._pending_round("k8", [1.0, 2.0])

        def re_record_in_flight(n):
            if n != 3:
                return
            self.h.state.on_query = None
            # The device is executing the next replay: pair 2's PRE node has
            # been re-recorded and its POST has not been reached yet.
            nodes = self.h.clock.captured_graph("k8")
            self.h.state.readable_from = self.h.state.now
            nodes.pairs[1][0].stamp(self.h.state.now + 1.0)

        self.h.state.on_query = re_record_in_flight
        self.h.log.end_round()

        un = parse_unsplit_line("[2026-09-09 00:00:00 TP1] " + self.lines()[0])
        self.assertIsNotNone(un, self.lines()[0])
        self.assertEqual(un["reason"], "graph-replay-nodes-unread")
        self.assertEqual(self.h.clock.graph_node_counts[4], 1)
        self.assertEqual(self.h.state.synchronize_calls, 0)


class MultiGraphRoundTest(unittest.TestCase):
    """More than one declared replay under ONE bracket (review finding 3).

    The shipped form kept a single ``(nodes, generation)`` on the span and
    OVERWROTE it, which prices the last replay and drops the rest with no
    refusal and no counter -- an undercounted wait that prints as a
    measurement. Two different graphs sum; one graph twice cannot, because
    the second replay overwrites the nodes the first would be read from.
    """

    def setUp(self) -> None:
        self.h = Harness()
        self.cap = _Capture()
        self.logger = logging.getLogger(
            "sglang.srt.managers.scheduler_components.decode_round_log"
        )
        self.logger.addHandler(self.cap)
        self.logger.setLevel(logging.INFO)
        self.addCleanup(self.logger.removeHandler, self.cap)

    def lines(self):
        return [ln for ln in self.cap.lines if ln.startswith("Decode rank batch")]

    def test_two_different_graphs_in_one_bracket_are_summed_not_dropped(self):
        """MUTANT-7: keep one entry per span. The draft graph's wait then
        vanishes from a round that ran it, and the line still prints."""
        self.h.capture("kdraft", ["tp.all_reduce"])
        self.h.capture("ktarget", ["dcp.all_gather"])
        self.h.log.begin_round(round_id=1, bs=6, rows=24)
        with self.h.log.segment("decode", graphed=True):
            self.h.clock.note_graph_replay("kdraft")
            self.h.replay("kdraft", [1.0])
            self.h.clock.note_graph_replay("ktarget")
            self.h.replay("ktarget", [2.0])
            self.h.state.advance(7.0)
        self.h.ready_now()
        self.h.log.end_round()

        line = self.lines()[0]
        self.assertNotIn("split unavailable", line)
        parsed = parse_rank_batch_line("[x TP1] " + line)
        self.assertAlmostEqual(parsed["wait_ms"], 3.0, places=1)
        self.assertIn("tp.all_reduce 1.0/1x", line)
        self.assertIn("dcp.all_gather 2.0/1x", line)

    def test_one_graph_declared_twice_in_one_bracket_is_refused_by_name(self):
        """The second replay re-executed the very nodes the first
        declaration would have been read from. That wait is not smaller, it
        is destroyed -- and summing the two declarations would count the
        second replay twice."""
        self.h.capture("k8", ["tp.all_reduce"])
        self.h.log.begin_round(round_id=1, bs=6, rows=24)
        with self.h.log.segment("decode", graphed=True):
            self.h.clock.note_graph_replay("k8")
            self.h.replay("k8", [1.0])
            self.h.clock.note_graph_replay("k8")
            self.h.replay("k8", [9.0])
            self.h.state.advance(5.0)
        self.h.ready_now()
        self.h.log.end_round()

        un = parse_unsplit_line("[2026-09-09 00:00:00 TP1] " + self.lines()[0])
        self.assertIsNotNone(un, self.lines()[0])
        self.assertEqual(un["reason"], "graph-replay-key-replayed-twice")
        self.assertFalse(un["split_known"])
        self.assertEqual(self.h.clock.graph_node_counts[5], 1)


class RunnerGraphKeyTest(unittest.TestCase):
    """The clock's registry is process-global; a ShapeKey is not (finding 1).

    ``shape_key.py`` states in its own docstring that a ShapeKey identifies
    a shape "across all runners" -- correct for a graph backend that one
    runner owns, wrong for the clock's one dict. A speculative draft runner
    and the target runner both capture ``bs=8``; under the bare shape the
    second capture REPLACES the first's node list, the first runner's next
    replay reads a graph it never ran, and the generation check passes
    because nobody bumped it in between.

    The runner's own method is borrowed here rather than reimplemented, so
    this cannot pass against a copy that has drifted from the call sites.
    """

    @staticmethod
    def _runner_class():
        # Imported inside the test: everything above in this file is
        # torch-free, and the runner module pulls the whole executor chain.
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        class _BareRunner:
            _clock_graph_key = DecodeCudaGraphRunner._clock_graph_key

        return _BareRunner

    def test_a_runner_s_key_for_one_shape_is_stable_across_calls(self):
        """A tag re-drawn per call would make the capture and the replay
        disagree, and every graphed round would refuse for no reason."""
        runner = self._runner_class()()
        first = runner._clock_graph_key(("shape", 8))
        second = runner._clock_graph_key(("shape", 8))
        self.assertEqual(first, second)

    def _capture_log(self):
        cap = _Capture()
        logger = logging.getLogger(
            "sglang.srt.managers.scheduler_components.decode_round_log"
        )
        logger.addHandler(cap)
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.removeHandler, cap)
        return cap

    def test_two_runners_capturing_one_shape_do_not_share_a_node_list(self):
        """MUTANT-8: pass the bare ShapeKey. Both captures land on one
        registry entry, and the draft runner's replay is priced with the
        target runner's nodes -- a fabricated split, with no refusal, in the
        exact shape of a measurement."""
        cls = self._runner_class()
        draft, target = cls(), cls()
        shape = ("shape", 8)
        kd, kt = draft._clock_graph_key(shape), target._clock_graph_key(shape)
        self.assertNotEqual(kd, kt)

        cap = self._capture_log()
        h = Harness()
        h.capture(kd, ["tp.all_reduce"])
        h.capture(kt, ["dcp.all_gather", "dcp.all_gather"])
        self.assertIsNotNone(h.clock.captured_graph(kd))
        self.assertEqual(len(h.clock.captured_graph(kd).pairs), 1)
        self.assertEqual(len(h.clock.captured_graph(kt).pairs), 2)

        h.log.begin_round(round_id=1, bs=6, rows=24)
        with h.log.segment("decode", graphed=True):
            h.clock.note_graph_replay(kd)
            h.replay(kd, [1.0])
            h.state.advance(4.0)
        h.ready_now()
        h.log.end_round()
        line = [ln for ln in cap.lines if ln.startswith("Decode rank batch")][0]
        self.assertIn("tp.all_reduce 1.0/1x", line)
        self.assertNotIn("dcp.all_gather", line)


class EagerPathUnchangedTest(unittest.TestCase):
    """#1241b must be invisible to the eager arm, byte for byte."""

    def setUp(self) -> None:
        self.h = Harness()
        self.cap = _Capture()
        self.logger = logging.getLogger(
            "sglang.srt.managers.scheduler_components.decode_round_log"
        )
        self.logger.addHandler(self.cap)
        self.logger.setLevel(logging.INFO)
        self.addCleanup(self.logger.removeHandler, self.cap)

    def test_the_eager_round_line_is_the_1241_line_character_for_character(self):
        self.h.log.begin_round(round_id=7, bs=1, rows=1)
        with self.h.log.segment("decode", graphed=False):
            with self.h.clock.span("tp.all_reduce"):
                self.h.state.advance(5.0)
            self.h.state.advance(15.0)
        self.h.log.begin_round(round_id=8, bs=1, rows=1)
        line = [ln for ln in self.cap.lines if ln.startswith("Decode rank batch")][0]
        head, _, tail = line.partition(", t: ")
        _, _, rest = tail.partition(", bs: ")
        self.assertEqual(head, "Decode rank batch, rank: 1, #round: 7")
        self.assertEqual(
            "bs: " + rest,
            "bs: 1, #rows: 1, #fwd: 1, gpu-ms: 20.0 (compute 15.0, wait 5.0) "
            "(wait by family: tp.all_reduce 5.0/1x)",
        )

    def test_an_eager_round_never_touches_the_graph_registry(self):
        self.h.log.begin_round(round_id=1, bs=1, rows=1)
        with self.h.log.segment("decode", graphed=False):
            with self.h.clock.span("tp.all_reduce"):
                self.h.state.advance(1.0)
        self.h.log.end_round()
        self.assertEqual(self.h.clock.graph_node_counts, (0, 0, 0, 0, 0, 0))

    def test_the_clock_is_armed_by_a_capture_scope_or_no_node_is_ever_laid(self):
        """The dispatch sites gate on `armed`, not on `span`. A capture scope
        that leaves `armed` False instruments nothing at all, silently."""
        self.assertFalse(self.h.clock.armed)
        with self.h.clock.capture_scope("k8"):
            self.assertTrue(self.h.clock.armed)
        self.assertFalse(self.h.clock.armed)


if __name__ == "__main__":
    unittest.main()
