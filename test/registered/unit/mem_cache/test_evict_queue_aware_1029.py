"""27b-queue-evict-1029: queue-aware eviction (Belady over the waiting queue).

Derived properties of QueueAwareStrategy (python/flliper/srt/mem_cache/evict_policy.py):
  P1  a node no waiting request needs is evicted before any node a waiting request needs
  P2  among waiters' nodes, the one the queue needs LATEST (largest earliest-position) goes first
  P3  equal keys fall back to LRU; with an empty queue the order IS LRU's order
  P4  a shared prefix carries the position of its EARLIEST waiting user
  P5  keys are pure: get_priority mutates nothing on the node (eviction order only, no computed value)
and the wiring that makes it reach the 27B path: SchedulePolicy.calc_priority hands the queue to the
tree, RadixCache.evict / UnifiedRadixCache.evict (FullComponent._peel_heap) open an eviction pass.
The mutation block proves the property checks are discriminating: each mutant fails at least one.
"""

import types
import unittest

import torch

from flliper.srt.environ import envs
from flliper.srt.managers.schedule_policy import SchedulePolicy
from flliper.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
from flliper.srt.mem_cache.cache_init_params import CacheInitParams
from flliper.srt.mem_cache.evict_policy import (
    LRUStrategy,
    QueueAwareStrategy,
    build_queue_marks,
)
from flliper.srt.mem_cache.memory_pool import ReqToTokenPool
from flliper.srt.mem_cache.radix_cache import RadixCache, RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from flliper.srt.mem_cache.utils import get_eviction_strategy
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# ------------------------------------------------------------------------------------- helpers
class _Node:
    def __init__(self, *, nid, parent, t):
        self.id = nid
        self.parent = parent
        self.last_access_time = t
        self.hit_count = 0


class _Req:
    def __init__(self, *, node):
        self.last_node = node
        self.best_match_node = None


def _chain_tree():
    """root -> s(shared) -> {a, b}; root -> c; root -> d.  Access times: a 1, b 2, c 3, d 4, s 5."""
    root = _Node(nid=0, parent=None, t=0)
    s = _Node(nid=1, parent=root, t=5)
    a = _Node(nid=2, parent=s, t=1)
    b = _Node(nid=3, parent=s, t=2)
    c = _Node(nid=4, parent=root, t=3)
    d = _Node(nid=5, parent=root, t=4)
    return root, s, a, b, c, d


def _order(strategy, nodes, queue):
    strategy.note_waiting_queue(queue)
    strategy.note_eviction_run()
    return [n.id for n in sorted(nodes, key=strategy.get_priority)]


def check_properties(strategy_factory):
    """Returns the list of violated properties (empty = all hold)."""
    bad = []
    _, s, a, b, c, d = _chain_tree()
    leaves = [a, b, c, d]
    # P1 + P2: waiters on c (pos 0) and a (pos 1); b, d unneeded -> b(2), d(4) by LRU, then a (later), then c
    if _order(strategy_factory(), leaves, [_Req(node=c), _Req(node=a)]) != [b.id, d.id, a.id, c.id]:
        bad.append("P1/P2")
    # P3: empty queue == LRU order
    lru = [n.id for n in sorted(leaves, key=LRUStrategy().get_priority)]
    if _order(strategy_factory(), leaves, []) != lru:
        bad.append("P3-empty")
    # P3: two nodes of the SAME earliest waiter tie on position -> LRU between them (a 1 before s 5)
    st = strategy_factory()
    st.note_waiting_queue([_Req(node=a)])
    st.note_eviction_run()
    if not st.get_priority(a) < st.get_priority(s):
        bad.append("P3-tie")
    # P4: s is shared by a (pos 1) and b (pos 0) -> s carries pos 0, i.e. outlives a's leaf key
    st = strategy_factory()
    st.note_waiting_queue([_Req(node=b), _Req(node=a)])
    st.note_eviction_run()
    if not st.get_priority(a) < st.get_priority(s):
        bad.append("P4")
    # P5: no field of the node moves
    st = strategy_factory()
    st.note_waiting_queue([_Req(node=a)])
    st.note_eviction_run()
    before = [(n.id, n.last_access_time, n.hit_count, n.parent) for n in leaves]
    for n in leaves:
        st.get_priority(n)
    if before != [(n.id, n.last_access_time, n.hit_count, n.parent) for n in leaves]:
        bad.append("P5")
    return bad


# ------------------------------------------------------------------------------ strategy alone
class TestQueueAwareProperties(CustomTestCase):
    def test_all_properties_hold(self):
        self.assertEqual(check_properties(QueueAwareStrategy), [])

    def test_marks_walk_stops_at_earlier_waiter_and_cap_degrades_to_lru(self):
        root, s, a, b, _, _ = _chain_tree()
        marks = build_queue_marks(waiting_queue=[_Req(node=a), _Req(node=b)], max_visits=100)
        # a, s from req 0; b from req 1 -- the walk of req 1 stops at s (already marked), root never
        self.assertEqual(marks.first_use, {a.id: 0, s.id: 0, b.id: 1})
        self.assertEqual(marks.visits, 3)
        self.assertFalse(marks.capped)
        capped = build_queue_marks(waiting_queue=[_Req(node=a), _Req(node=b)], max_visits=1)
        self.assertTrue(capped.capped)
        self.assertEqual(capped.first_use, {a.id: 0})  # the rest stays unmarked = LRU keys

    def test_best_match_node_is_preferred_over_last_node(self):
        """A host hit (best_match_node deeper than the device match) must protect the host part too."""
        _, s, a, _, _, _ = _chain_tree()
        req = _Req(node=s)
        req.best_match_node = a
        marks = build_queue_marks(waiting_queue=[req], max_visits=100)
        self.assertIn(a.id, marks.first_use)

    def test_rebuild_only_when_queue_changes(self):
        _, _, a, b, _, _ = _chain_tree()
        st = QueueAwareStrategy()
        q = [_Req(node=a)]
        st.note_waiting_queue(q)
        st.note_eviction_run()
        st.note_eviction_run()
        self.assertEqual(st.stats.rebuilds, 1)  # same list, same ends -> no walk
        q.append(_Req(node=b))  # in-place append is seen without a new hand-over
        st.note_eviction_run()
        self.assertEqual(st.stats.rebuilds, 2)
        self.assertIn(b.id, st._first_use)

    def test_counter_and_log_line_every_n_runs(self):
        _, _, a, b, c, d = _chain_tree()
        st = QueueAwareStrategy(log_every_runs=2)
        st.note_waiting_queue([_Req(node=a)])  # a is the OLDEST (t=1): LRU would take it first
        with self.assertLogs("flliper.srt.mem_cache.evict_policy", level="INFO") as cm:
            for _ in range(2):
                st.note_eviction_run()
                for n in (a, b, c, d):
                    st.get_priority(n)
            st.note_eviction_run()  # closes run 2 -> log
        self.assertEqual(st.stats.runs, 2)
        self.assertEqual(st.stats.runs_inverted, 2)
        self.assertTrue(any("EVICT-QUEUE-AWARE runs=2 lru_order_inverted=2" in m for m in cm.output))


class TestMutationProof(CustomTestCase):
    """Each mutant breaks one derivation; the property check must name it."""

    def test_mutants_are_caught(self):
        class LruOnly(QueueAwareStrategy):
            def get_priority(self, node):
                return (0, 0, node.last_access_time)

        class EarliestFirst(QueueAwareStrategy):  # sign flipped: evicts the HEAD's prefix first
            def get_priority(self, node):
                g, p, t = super().get_priority(node)
                return (g, -p, t)

        class NoLruTie(QueueAwareStrategy):
            def get_priority(self, node):
                g, p, _ = super().get_priority(node)
                return (g, p, -node.id)

        class Touching(QueueAwareStrategy):  # writes the node (like a refresh would)
            def get_priority(self, node):
                node.last_access_time += 0
                node.hit_count += 1
                return super().get_priority(node)

        for mutant in (LruOnly, EarliestFirst, NoLruTie, Touching):
            with self.subTest(mutant=mutant.__name__):
                self.assertNotEqual(check_properties(mutant), [])


# ------------------------------------------------------------------------------ real caches
def _mock_allocator(freed):
    return types.SimpleNamespace(device="cpu", free=lambda v: freed.append(v))


def _insert(cache, tokens, value):
    cache.insert(InsertParams(key=RadixKey(list(tokens)), value=value))


def _leaf(cache, tokens):
    return cache.match_prefix(MatchPrefixParams(key=RadixKey(list(tokens)))).last_device_node


A, B, C = range(100, 104), range(200, 204), range(300, 304)


class _RadixFixture:
    def _setup(self, *, queue_aware: bool):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
        self.freed = []
        with envs.FLLIPER_EVICT_POLICY_QUEUE_AWARE.override(queue_aware):
            cache = self._make_cache()
        for i, toks in enumerate((A, B, C)):
            _insert(cache, toks, value=self._value(n=len(toks), base=10 * i))
        self.nodes = {k: _leaf(cache, t) for k, t in (("A", A), ("B", B), ("C", C))}
        for t, k in enumerate(("A", "B", "C")):  # A oldest: plain LRU would evict A first
            self.nodes[k].last_access_time = t + 1
        return cache

    def _evict_one_through_policy(self, cache, waiter_tokens):
        """The scheduler path: calc_priority hands the queue over, then the admission evicts."""
        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        waiter = _Req(node=_leaf(cache, waiter_tokens))
        policy.calc_priority([waiter])
        for k, t in enumerate(("A", "B", "C")):
            self.nodes[t].last_access_time = k + 1  # the match above touched the waiter's path
        cache.evict(EvictParams(num_tokens=4))
        return [k for k, n in self.nodes.items() if self._evicted(cache, n)]


class TestRadixCacheIntegration(_RadixFixture, CustomTestCase):
    def _make_cache(self):
        return RadixCache.create_simulated(mock_allocator=_mock_allocator(self.freed))

    def _value(self, *, n, base):
        return torch.arange(base, base + n)

    @staticmethod
    def _evicted(cache, node):
        return node.evicted or node not in node.parent.children.values()

    def test_switch_selects_strategy(self):
        self.assertIsInstance(self._setup(queue_aware=True).eviction_strategy, QueueAwareStrategy)
        self.assertIsInstance(self._setup(queue_aware=False).eviction_strategy, LRUStrategy)
        with envs.FLLIPER_EVICT_POLICY_QUEUE_AWARE.override(True):  # explicit non-LRU stays as chosen
            self.assertNotIsInstance(get_eviction_strategy("slru"), QueueAwareStrategy)

    def test_waiter_protects_its_prefix(self):
        cache = self._setup(queue_aware=True)
        self.assertEqual(self._evict_one_through_policy(cache, A), ["B"])

    def test_switch_off_is_plain_lru(self):
        cache = self._setup(queue_aware=False)
        self.assertEqual(self._evict_one_through_policy(cache, A), ["A"])

    def test_no_waiter_evicts_exactly_what_lru_evicts(self):
        """Bit-identity of the decision set when the queue is empty: same rows freed, same count."""
        got = {}
        for qa in (True, False):
            cache = self._setup(queue_aware=qa)
            cache.note_waiting_queue([])
            cache.evict(EvictParams(num_tokens=8))
            got[qa] = (sorted(torch.cat(self.freed).tolist()), cache.evictable_size())
        self.assertEqual(got[True], got[False])


class TestUnifiedRadixCacheIntegration(_RadixFixture, CustomTestCase):
    """The 27B path: hybrid models under HiCache build UnifiedRadixCache; FULL eviction = _peel_heap."""

    def _make_cache(self):
        from flliper.srt.mem_cache.allocator import TokenToKVPoolAllocator
        from flliper.srt.mem_cache.memory_pool import MHATokenToKVPool

        kv = MHATokenToKVPool(size=64, page_size=1, dtype=torch.float16, head_num=1, head_dim=4,
                              layer_num=1, device="cpu", enable_memory_saver=False)
        self.alloc = TokenToKVPoolAllocator(size=64, dtype=torch.float16, device="cpu", kvcache=kv,
                                            need_sort=False)
        return UnifiedRadixCache(params=CacheInitParams(
            req_to_token_pool=ReqToTokenPool(size=8, max_context_len=64, device="cpu",
                                             enable_memory_saver=False),
            token_to_kv_pool_allocator=self.alloc, page_size=1, disable=False,
            tree_components=(ComponentType.FULL,)))

    def _value(self, *, n, base):
        return self.alloc.alloc(n).to(dtype=torch.int64, copy=True)

    @staticmethod
    def _evicted(cache, node):
        return node.evicted or node not in node.parent.children.values()

    def test_waiter_protects_its_prefix(self):
        cache = self._setup(queue_aware=True)
        self.assertEqual(self._evict_one_through_policy(cache, A), ["B"])

    def test_switch_off_is_plain_lru(self):
        cache = self._setup(queue_aware=False)
        self.assertEqual(self._evict_one_through_policy(cache, A), ["A"])


if __name__ == "__main__":
    unittest.main()
