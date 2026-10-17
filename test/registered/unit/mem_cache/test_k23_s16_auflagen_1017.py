"""K23 review conditions A1-A4 (S16, 17.10.) on top of S14's TP0 host-only leaf spill.

A1  the host-only round gets the REST of the FORMA synchronous-L3 budget (need ~1107, cap 128).
A2  a raise after the first drop still sends the STATE of the dropped nodes; ``_r12_rec`` is restored.
A3  a worker names every STATE(exists=0) it cannot delete (``K23 SPILL-DIVERGED``), counter included.
A4  replica: TP0 spills + attach(); a worker tree consumes -> same node keys, unreconcilable=0.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache import form_a_host_shadow as R12
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.weg2 import host_only_spill as HOS
from sglang.srt.weg2 import pp_slot_fidelity as SF
from sglang.test.test_utils import CustomTestCase

from test_w3_spill_host_leaves_int22_1008 import SPAN, _AnchorPool, _death_shape
from test_unified_radix_cache_unittest import CacheConfig, build_fixture


def _spill(cache, pool, claimer, role, need=SPAN, **kw):
    setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, False)
    with mock.patch.object(R12, "role", return_value=role):
        return cache._w3_arena_spill(pool, need, claimer=claimer, **kw)


def _keys(cache):
    out = set()
    for n in cache._collect_all_nodes():
        if n is not cache.root_node:
            k = R12.node_key(n)
            if k is not None:
                out.add(k)
    return out


def _many_leaves(n_leaves, pages=SPAN):
    """A device claimer with ``n_leaves`` host-only sibling leaves of ``pages`` pages each."""
    cache, allocator, _ = build_fixture(
        CacheConfig(page_size=1, components=(ComponentType.FULL,), kv_size=4096))
    cache.insert(InsertParams(key=RadixKey(list(range(1, 1 + pages)), None),
                              value=allocator.alloc(pages).to(dtype=torch.int64)))
    top = [n for n in cache._collect_all_nodes() if n is not cache.root_node][0]
    leaves = []
    for i in range(n_leaves):
        key = list(range(1, 1 + pages)) + [1000 + i * pages + j for j in range(pages)]
        cache.insert(InsertParams(key=RadixKey(key, None), value=allocator.alloc(len(key)).to(dtype=torch.int64)))
    for n in cache._collect_all_nodes():
        if n is not cache.root_node and n is not top:
            leaves.append(n)
    for i, node in enumerate(leaves):
        node.component_data[ComponentType.FULL].host_value = torch.arange(
            100 + i * pages, 100 + (i + 1) * pages, dtype=torch.int64)
        node.hash_value = ["h%d-%d" % (i, j) for j in range(pages)]
        cache._evict_to_host(node)
    return cache, top, leaves


class TestA1Cap(CustomTestCase):
    def setUp(self):
        R12._reset_state_for_tests()

    def tearDown(self):
        R12._reset_state_for_tests()

    def test_need_1107_with_cap_128_secures_at_most_128_pages(self):
        cache, top, leaves = _many_leaves(70, pages=16)       # 70 * 16 = 1120 host-only pages
        pool = _AnchorPool()
        got = _spill(cache, pool, top, "host", need=1107, max_secure_pages=128)
        self.assertLessEqual(sum(pool.secured), 128, "the synchronous L3 write is bounded (was ~1000 pages)")
        self.assertEqual(sum(pool.secured), 128)
        self.assertEqual(got, 128)
        self.assertEqual(cache._w3_spill_cap_used, 128, "FORMA line counts the host-only pages in secure_pages")
        self.assertGreater(cache._w3_spill_cap_skipped, 0)

    def test_cap_zero_is_unbounded(self):
        cache, top, leaves = _many_leaves(10, pages=16)
        pool = _AnchorPool()
        got = _spill(cache, pool, top, "host", need=160, max_secure_pages=0)
        self.assertEqual(got, 160)

    def test_a_leaf_that_does_not_fit_the_rest_is_passed_over(self):
        cache, top, leaves = _many_leaves(4, pages=16)
        pool = _AnchorPool()
        r = HOS.spill_host_only(cache, pool, 64, 1, set(), max_secure_pages=8)
        self.assertEqual(r["secure_pages"], 0, "a 16-page leaf does not fit a rest of 8")
        self.assertEqual(r["over_cap"], 4)
        self.assertEqual(pool.secured, [])


class TestA2ExceptionPath(CustomTestCase):
    def setUp(self):
        R12._reset_state_for_tests()

    def tearDown(self):
        R12._reset_state_for_tests()

    def test_raise_after_first_drop_still_sends_state_and_restores_rec(self):
        cache, top, mid, leaf = _death_shape()
        pool = _AnchorPool()
        real = cache._evict_host_leaf
        calls = {"n": 0}

        def boom(node, tracker):
            r = real(node, tracker)
            calls["n"] += 1
            raise RuntimeError("after first drop")

        sentinel = ["outer"]
        cache._r12_rec = sentinel
        with mock.patch.object(cache, "_evict_host_leaf", side_effect=boom):
            with self.assertRaises(RuntimeError):
                _spill(cache, pool, top, "host")
        self.assertEqual(calls["n"], 1)
        self.assertIs(cache._r12_rec, sentinel, "_r12_rec restored, not set to None")
        sent = {(e[1], int(e[2])) for e in R12._S.ledger if e[0] == R12.STATE and e[3] == 0}
        self.assertEqual(len(sent), 1, "the node TP0 already dropped reaches the workers")
        self.assertNotIn(R12.node_key(leaf), _keys(cache))
        self.assertEqual(sent, {R12.node_key(leaf)})


class TestA3A4Divergence(CustomTestCase):
    def setUp(self):
        R12._reset_state_for_tests()

    def tearDown(self):
        R12._reset_state_for_tests()

    def test_a4_replica_trees_equal_after_tp0_spill(self):
        tp0, top0, mid0, leaf0 = _death_shape()
        worker, topw, midw, leafw = _death_shape()
        self.assertEqual(_keys(tp0), _keys(worker))
        got = _spill(tp0, _AnchorPool(), top0, "host")
        self.assertEqual(got, 2 * SPAN)
        with mock.patch.object(R12, "role", return_value="host"):
            out = R12.attach([])
        self.assertEqual(len(out), 1)
        self.assertTrue(isinstance(out[0], R12.FormAHostVerdict))
        with mock.patch.object(R12, "role", return_value="worker"), \
                mock.patch.object(R12, "cut_role", return_value=None):
            with self.assertLogs(R12.logger, level="INFO") as lg:
                rest = R12.consume(out, tree=worker)
                R12.logger.info("sentinel")
        self.assertEqual(rest, [])
        self.assertEqual(_keys(tp0), _keys(worker), "node keys of both trees equal")
        self.assertFalse([m for m in lg.output if "SPILL-DIVERGED" in m or "UNRECONCILABLE" in m])
        self.assertNotIn(leafw, worker._collect_all_nodes())
        self.assertNotIn(midw, worker._collect_all_nodes())
        self.assertEqual(R12._S.counts.get("k23_state_diverged", 0), 0)

    def test_a3_worker_names_every_non_deleted_outcome(self):
        # the worker's node is device-resident (not host-only): TP0's exists=0 cannot be applied
        worker, topw, midw, leafw = _death_shape()
        # make the worker's mid an interior, device-evicted node: leaf stays (has child) -> unreconcilable
        ev = (R12.STATE, R12.node_key(midw)[0], R12.node_key(midw)[1], 0, 0, 0)
        with mock.patch.object(R12, "role", return_value="worker"), \
                mock.patch.object(R12, "cut_role", return_value=None):
            with self.assertLogs(R12.logger, level="WARNING") as lg:
                c = R12.apply(worker, [ev], seq=7)
        self.assertEqual(c["unreconcilable"], 1)
        lines = [m for m in lg.output if "K23 SPILL-DIVERGED role=worker key=" in m]
        self.assertEqual(len(lines), 1)
        self.assertIn("outcome=unreconcilable", lines[0])
        self.assertEqual(R12._S.counts["k23_state_diverged"], 1)

    def test_a3_noop_outcome_is_named_too(self):
        worker, topw, midw, leafw = _death_shape()
        # a node without host rows on the worker: TP0 says gone, reconcile = noop, node stays
        for n in (leafw, midw):
            n.component_data[ComponentType.FULL].host_value = None
        ev = (R12.STATE, "h0-%d" % (SPAN - 1), 3 * SPAN, 0, 0, 0)
        with mock.patch.object(R12, "role", return_value="worker"), \
                mock.patch.object(R12, "cut_role", return_value=None):
            with self.assertLogs(R12.logger, level="WARNING") as lg:
                R12.apply(worker, [ev], seq=8)
        self.assertTrue([m for m in lg.output if "K23 SPILL-DIVERGED" in m and "outcome=noop" in m])


if __name__ == "__main__":
    unittest.main()
