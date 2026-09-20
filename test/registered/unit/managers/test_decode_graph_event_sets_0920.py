"""Task #52 (20.09.): K event sets per captured graph.

fn8r5 read ``graph-replay-nodes-overwritten-by-1`` on every decode round:
under the overlap scheduler replay G+1 is launched before the scheduler
thread reads round G, and a static graph re-stamps its event-record nodes
on every replay. The fix points the exec graph's event-record nodes at a
different event set before every replay (cudaGraphExecEventRecordNode-
SetEvent), so generation G stays readable until replay G+K.

Hermetic: the cudart seam (``ClockBackend.graph_binder``) is a fake that
models node identity and the exec update; the replay stamps whatever
events the nodes are CURRENTLY bound to, which is the CUDA fact the design
leans on. Run with ``CUDA_VISIBLE_DEVICES=''``.
"""
from __future__ import annotations

import unittest

from sglang.srt.utils.collective_clock import ClockBackend, CollectiveClock
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu")


class State:
    def __init__(self):
        self.now = 0.0
        self.capturing = False


class Ev:
    _ids = 0

    def __init__(self, st):
        self.st = st
        self.t = None
        Ev._ids += 1
        self.handle = Ev._ids  # the cudaEvent_t, modelled

    def record(self):
        if self.st.capturing:
            self.t = None
            return
        self.t = self.st.now

    def query(self):
        return self.t is not None

    def elapsed_time(self, other):
        if self.t is None or other.t is None:
            raise RuntimeError("cudaErrorNotReady")
        return other.t - self.t


class FakeGraph:
    """The captured graph: nodes (ints) -> event handle they record."""

    def __init__(self):
        self.nodes = {}  # node -> handle
        self.exec = 0xE

    def raw_cuda_graph(self):
        return id(self)

    def raw_cuda_graph_exec(self):
        return self.exec


class FakeBinder:
    def __init__(self, graph: FakeGraph, events_by_handle):
        self.graph = graph
        self.events = events_by_handle
        self.set_calls = 0

    def graph_handles(self, cuda_graph):
        return cuda_graph.raw_cuda_graph(), cuda_graph.raw_cuda_graph_exec()

    def event_record_nodes(self, raw):
        return list(self.graph.nodes.items())

    def set_event(self, exec_handle, node, event_handle):
        assert exec_handle == self.graph.exec
        assert node in self.graph.nodes
        self.graph.nodes[node] = event_handle
        self.set_calls += 1

    @staticmethod
    def event_handle(ev):
        return ev.handle


class Backend(ClockBackend):
    def __init__(self, st):
        self.st = st
        self.created = 0
        self.by_handle = {}
        self.binder = None

    def event(self):
        self.created += 1
        e = Ev(self.st)
        self.by_handle[e.handle] = e
        return e

    def is_capturing(self):
        return self.st.capturing

    def materialize(self, ev):
        assert not self.st.capturing
        ev.record()

    def graph_binder(self):
        return self.binder


class H:
    def __init__(self, ring=2):
        self.st = State()
        self.be = Backend(self.st)
        self.clock = CollectiveClock(backend=self.be, graph_ring=ring)
        self.graph = FakeGraph()
        self.be.binder = FakeBinder(self.graph, self.be.by_handle)

    def capture(self, key, fams):
        with self.clock.capture_scope(key, phase="spec_verify"):
            self.st.capturing = True
            try:
                for f in fams:
                    with self.clock.span(f):
                        pass
            finally:
                self.st.capturing = False
        nodes = self.clock.captured_graph(key)
        # the capture laid event-record nodes: one per pair event
        n = 100
        for pre, post, _ in nodes.pairs:
            self.graph.nodes[n] = pre.handle
            self.graph.nodes[n + 1] = post.handle
            n += 2
        return nodes

    def replay(self, key, ms_per_region):
        """CUDA fact: a replay stamps the events the nodes are bound to NOW."""
        self.clock.note_graph_replay(key)
        nodes = self.clock.captured_graph(key)
        handles = [self.graph.nodes[n] for n in sorted(self.graph.nodes)]
        for i, ms in enumerate(ms_per_region):
            pre = self.be.by_handle[handles[2 * i]]
            post = self.be.by_handle[handles[2 * i + 1]]
            pre.t = self.st.now
            self.st.now += ms
            post.t = self.st.now
        return nodes.generation


class EventSetsTest(unittest.TestCase):
    def test_bind_matches_every_pair_and_creates_k_minus_1_sets(self):
        h = H(ring=3)
        nodes = h.capture("g", ["all_reduce", "all_reduce", "a2a"])
        created_before = h.be.created
        self.assertTrue(h.clock.bind_graph("g", h.graph))
        self.assertTrue(nodes.bound)
        self.assertEqual(len(nodes.event_sets), 3)
        self.assertEqual(len(nodes.node_handles), 3)
        # 2 spare sets x 3 pairs x 2 events, all materialized outside capture
        self.assertEqual(h.be.created - created_before, 12)

    def test_bind_refuses_when_a_pair_event_is_not_a_node(self):
        h = H(ring=2)
        nodes = h.capture("g", ["all_reduce"])
        h.graph.nodes.clear()  # the graph carries no event-record nodes
        self.assertFalse(h.clock.bind_graph("g", h.graph))
        self.assertFalse(nodes.bound)

    def test_generation_g_survives_replay_g_plus_1(self):
        """The fn8r5 shape: G+1 replayed before G is read. With K=2 the
        reading of G must be the numbers G stamped, not a refusal."""
        h = H(ring=2)
        nodes = h.capture("g", ["all_reduce", "a2a"])
        h.clock.bind_graph("g", h.graph)
        g1 = h.replay("g", [1.0, 2.0])
        g2 = h.replay("g", [10.0, 20.0])  # overwrites set g2 % 2, not g1's
        fams, why = h.clock._graph_reading(nodes, g1, count=True)
        self.assertIsNone(why)
        self.assertAlmostEqual(fams["spec_verify:all_reduce"].total_ms, 1.0)
        self.assertAlmostEqual(fams["spec_verify:a2a"].total_ms, 2.0)
        fams2, why2 = h.clock._graph_reading(nodes, g2, count=True)
        self.assertIsNone(why2)
        self.assertAlmostEqual(fams2["spec_verify:a2a"].total_ms, 20.0)

    def test_generation_g_is_refused_by_name_after_replay_g_plus_k(self):
        h = H(ring=2)
        nodes = h.capture("g", ["all_reduce"])
        h.clock.bind_graph("g", h.graph)
        g1 = h.replay("g", [1.0])
        h.replay("g", [2.0])
        # fences are recorded eagerly: complete once stamped (Ev.record)
        h.replay("g", [3.0])  # G+2 re-executes G's set
        fams, why = h.clock._graph_reading(nodes, g1, count=True)
        self.assertIsNone(fams)
        self.assertEqual(why, "graph-replay-nodes-overwritten-by-2")

    def test_swap_allocates_nothing_and_is_counted(self):
        h = H(ring=2)
        h.capture("g", ["all_reduce", "a2a"])
        h.clock.bind_graph("g", h.graph)
        created = h.be.created
        for _ in range(5):
            h.replay("g", [1.0, 1.0])
        self.assertEqual(h.be.created, created, "the replay path allocated")
        # 5 replays, set changes every replay with K=2: 5 swaps x 2 pairs x 2
        self.assertEqual(h.be.binder.set_calls, 5 * 4)
        self.assertEqual(h.clock._swap_calls, 5)

    def test_unbound_graph_keeps_the_ring_semantics(self):
        h = H(ring=2)
        h.be.binder = None
        nodes = h.capture("g", ["all_reduce"])
        self.assertFalse(h.clock.bind_graph("g", h.graph))
        g1 = h.replay("g", [1.0])
        h.replay("g", [2.0])
        fams, why = h.clock._graph_reading(nodes, g1, count=True)
        self.assertIsNone(fams)
        self.assertEqual(why, "graph-replay-nodes-overwritten-by-1")


if __name__ == "__main__":
    unittest.main()


class ExternalEventTest(unittest.TestCase):
    def test_torch_backend_creates_external_timing_events(self):
        """fn8t 20.09.: without external=True a captured record is a
        cross-stream dependency, not an event-record node -- the graph then
        carries nothing the reader could read."""
        import torch

        from sglang.srt.utils.collective_clock import TorchCudaBackend

        seen = {}

        class FakeEvent:
            def __init__(self, **kw):
                seen.update(kw)

        orig = torch.cuda.Event
        torch.cuda.Event = FakeEvent
        try:
            TorchCudaBackend().event()
        finally:
            torch.cuda.Event = orig
        self.assertTrue(seen.get("enable_timing"))
        self.assertTrue(seen.get("external"), seen)
