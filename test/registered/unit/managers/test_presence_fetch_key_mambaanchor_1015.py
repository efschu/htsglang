"""MAMBA-ANCHOR 1015 (weg2/presence_fetch_key.py): the #950 presence probe asks
the store with the FETCH's key form (bigram RadixKey) over the fetch's span
(the whole span for the anchor-clamped v2 question).

SPECIMEN: NF boot ..._1015_071520, P PP0 07:20:27, weg2-0-6 (prompt 211346):
device head 3008 with state, the store holding the shared prefix up to the fork
211136 with its deepest recurrent anchor at 208320 (the same prompt read
``anchored=208320`` as weg2-5-30 in boot 070103). The probe answered
``H108 PRESENCE-PROBE keys=own pages=0 present=False`` / ``#1035 R13 EMPTY KV
PREFIX (keys=1024)`` -> ``#1400 STORE-TOLD INTAKE verdict=declined:store_absent``
-> 208338 tokens recomputed.

RED on 364ab5de5d: ``_prefetch_kvcache`` returns ``declined:store_absent`` on
the weg2-0-6 geometry (unigram keys) -- and with the key form alone fixed it
still would (the anchor is page 3208 of the span, the probe asked 1024).
"""

import os
import types
import unittest
from unittest import mock

from sglang.srt.managers import cache_controller as cc
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache import hicache_phase_binding as hpb
from sglang.srt.mem_cache.hicache_storage import STORAGE_BATCH_SIZE
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.utils import compute_node_hash_values, get_hash_str
from sglang.srt.weg2 import handoff as ho
from sglang.srt.weg2 import presence_fetch_key as pfk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PS = 64


class _Store:
    """File-backend semantics: leading KV hits; v2 clamped to the deepest page
    of the asked batch that also carries a mamba anchor (trailing pool)."""

    def __init__(self, kv_keys, anchor_keys=()):
        self.kv = set(kv_keys)
        self.anchors = set(anchor_keys)
        self.asked = []

    def _lead(self, batch):
        n = 0
        for k in batch:
            if k not in self.kv:
                break
            n += 1
        return n

    def batch_exists(self, batch, extra_info=None):
        self.asked.append(list(batch))
        return self._lead(batch)

    def batch_exists_v2(self, batch, pool_transfers, extra_info=None):
        self.asked.append(list(batch))
        hit = 0
        for i in range(self._lead(batch)):
            if batch[i] in self.anchors:
                hit = i + 1
        return types.SimpleNamespace(kv_hit_pages=hit)


def _controller(store, hybrid=True):
    ctl = types.SimpleNamespace(
        get_hash_str=get_hash_str,
        page_size=PS,
        storage_backend=store,
        _presence_pool_transfers=(lambda: [object()]) if hybrid else (lambda: None),
    )
    ctl.store_presence_pages = HiCacheController.store_presence_pages.__get__(ctl)
    return ctl


class _Geo:
    """weg2-0-6: head ``dev`` on the device (state at its end), the store holds
    the bigram chain up to ``fork`` (what P's write-through keyed), anchors at
    the head and at ``anchor``."""

    def __init__(self, dev=3008, fork=211136, prompt=211346, anchor=208320):
        self.dev, self.fork, self.n, self.anchor = dev, fork, prompt, anchor
        self.ids = [(7 * i + 11) % 248000 for i in range(prompt)]
        # P's store keys: the tree's bigram chain from the root (#1233)
        self.chain = get_hash_str(RadixKey(self.ids[: fork + 1], is_bigram=True), None, page_size=PS)
        node = types.SimpleNamespace(
            key=RadixKey(self.ids[: dev + 1], is_bigram=True), parent=None)
        self.d_last_hash = compute_node_hash_values(node, PS)[dev // PS - 1]
        anchors = [self.chain[dev // PS - 1]]
        if anchor:
            anchors.append(self.chain[anchor // PS - 1])
        self.store = _Store(self.chain[: fork // PS], anchors)

    def span(self):
        return self.ids[self.dev : self.n - 1]  # _compute_max_prefix_len = n - 1


class TestTheMechanism(CustomTestCase):
    def test_chain_is_prefix_consistent(self):
        # the device node's last hash continues exactly into the store chain
        g = _Geo()
        self.assertEqual(g.d_last_hash, g.chain[g.dev // PS - 1])
        cont = get_hash_str(RadixKey(g.ids[g.dev : g.fork + 1], is_bigram=True),
                            g.d_last_hash, page_size=PS)
        self.assertTrue(cont == g.chain[g.dev // PS :], "chain continues")

    def test_unigram_ids_meet_no_store_key(self):
        # true on both sides of the fix: the pre-1015 probe input
        g = _Geo()
        own = get_hash_str(list(g.span()), g.d_last_hash, page_size=PS)
        self.assertFalse(set(own) & g.store.kv)

    def test_probe_key_equals_the_fetch_key(self):
        # parity: span_key hashes exactly as prefetch_from_storage keys the read
        g = _Geo()
        tree = types.SimpleNamespace(is_eagle=True, page_size=PS)
        req = types.SimpleNamespace(extra_key=None, last_host_node=None)
        probe = get_hash_str(pfk.span_key(tree, req, g.span()), g.d_last_hash, page_size=PS)
        fetch_key = RadixKey(g.span(), extra_key=None, is_bigram=True).page_aligned(PS)
        # (plain bool: a 3300-key list diff under the retry wrapper takes minutes)
        self.assertTrue(probe == get_hash_str(fetch_key, g.d_last_hash, page_size=PS),
                        "the probe hashes what the fetch reads")


class TestStorePresencePages(CustomTestCase):
    def _key(self, g):
        return RadixKey(g.span(), is_bigram=True).page_aligned(PS)

    def test_weg2_0_6_anchor_beyond_page_1024_is_seen(self):
        g = _Geo()
        pages = _controller(g.store).store_presence_pages(self._key(g), g.d_last_hash, None)
        self.assertEqual(pages, (g.anchor - g.dev) // PS)  # 3208 pages -> 208320
        self.assertGreater(len(g.store.asked[-1]), STORAGE_BATCH_SIZE)

    def test_key_form_alone_matters(self):
        # anchor INSIDE the first batch (STORAGE_BATCH_SIZE: 128 by default, NF
        # SGLANG_HICACHE_STORAGE_BATCH=1024), so the span term plays no part:
        # the key form alone decides (M2 leaves this green, the id list is 0)
        k = STORAGE_BATCH_SIZE // 2
        g = _Geo(anchor=3008 + k * PS)
        ctl = _controller(g.store)
        self.assertEqual(ctl.store_presence_pages(self._key(g), g.d_last_hash, None), k)
        self.assertEqual(ctl.store_presence_pages(g.span(), g.d_last_hash, None), 0)

    def test_dense_v1_question_keeps_the_first_batch(self):
        g = _Geo()
        ctl = _controller(g.store, hybrid=False)
        self.assertEqual(ctl.store_presence_pages(self._key(g), g.d_last_hash, None),
                         STORAGE_BATCH_SIZE)
        self.assertEqual(len(g.store.asked[-1]), STORAGE_BATCH_SIZE)

    def test_switch_off_is_the_pre_1015_question(self):
        g = _Geo()
        with mock.patch.dict(os.environ, {pfk.ENV: "0"}):
            tree = types.SimpleNamespace(is_eagle=True, page_size=PS)
            req = types.SimpleNamespace(extra_key=None, last_host_node=None)
            span = pfk.span_key(tree, req, g.span())
            self.assertIs(type(span), list)
            pages = _controller(g.store).store_presence_pages(span, g.d_last_hash, None)
        self.assertEqual(pages, 0)
        self.assertTrue(g.store.asked[-1]
                        == get_hash_str(list(g.span()), g.d_last_hash, page_size=PS)[:STORAGE_BATCH_SIZE],
                        "switch off: the unigram chain's first batch, as before")

    def test_no_anchor_on_the_path_stays_absent(self):
        # KV without state is no hit (S1/FIX-3: they stand or fall together)
        g = _Geo(anchor=0)
        g.store.anchors = set()
        self.assertEqual(_controller(g.store).store_presence_pages(self._key(g), g.d_last_hash, None), 0)


class _Node:
    def __init__(self, last_hash):
        self.backuped = False
        self.parent = None
        self.key = None
        self._h = last_hash

    def get_last_hash_value(self):
        return self._h

    def get_prefix_hash_values(self, parent):
        return []


class _Tree:
    def __init__(self, ctl, collective):
        self.root_node = object()
        self.cache_controller = ctl
        self.hicache_storage_pass_prefix_keys = False
        self.ongoing_prefetch = {}
        self.is_eagle = True
        self.page_size = PS
        self.calls = []
        self._collective = collective

    def prefetch_participation_is_collective(self):
        return self._collective

    def prefetch_from_storage(self, rid, node, tokens, last_hash, prefix_keys, **kw):
        self.calls.append((len(tokens), kw.get("locally_eligible", True)))


class _Req:
    def __init__(self, g, rid):
        self.rid = rid
        self.extra_key = None
        self.prefix_indices = list(range(g.dev))
        self.host_hit_length = 0
        self.full_untruncated_fill_ids = list(g.ids)
        self.last_host_node = _Node(g.d_last_hash)

    def init_next_round_input(self, tree_cache, cow_mamba=False):
        pass

    def _compute_max_prefix_len(self, n):
        return n - 1


class _Sched:
    _prefetch_kvcache = Scheduler._prefetch_kvcache
    _note_prefetch_unregistered = Scheduler._note_prefetch_unregistered
    _weg2_local_store_matches = Scheduler._weg2_local_store_matches

    def __init__(self, g, collective=False):
        self.enable_hicache_storage = True
        self.page_size = PS
        self.tree_cache = _Tree(_controller(g.store), collective)
        self.waiting_queue = []


class TestTheSchedulerAsksWithTheFetchKey(CustomTestCase):
    def setUp(self):
        self._p = mock.patch.object(ho, "read", return_value=None)
        self._p.start()
        self._g = mock.patch.object(hpb, "current_generation", return_value=1)
        self._g.start()

    def tearDown(self):
        self._p.stop()
        self._g.stop()
        cc.WEG2_HANDOFF_PAGE_KEYS.clear()

    def test_weg2_0_6_is_not_store_absent(self):
        # P (group_decides False): the pre-1015 probe returned declined:store_absent
        g = _Geo()
        s = _Sched(g)
        req = _Req(g, "weg2-0-6")
        verdict = s._prefetch_kvcache(req)
        self.assertNotEqual(verdict, "declined:store_absent")
        self.assertTrue(req._pp_store_presence_cache[1])
        self.assertEqual(s.tree_cache.calls[-1][0], g.n - 1 - g.dev)

    def test_d_vote_enters_with_the_span(self):
        # D (group_decides True): the vote carried nothing before
        g = _Geo()
        s = _Sched(g, collective=True)
        req = _Req(g, "weg2-0-6")
        s._prefetch_kvcache(req)
        n, eligible = s.tree_cache.calls[-1]
        self.assertTrue(eligible)
        self.assertEqual(n, g.n - 1 - g.dev)

    def test_x_store_vote_credits_the_anchored_depth(self):
        # #1317 design A: the second caller of the probe (mutant M4)
        g = _Geo()
        s = _Sched(g)
        req = _Req(g, "weg2-0-6")
        s.waiting_queue = [req]
        out = s._weg2_local_store_matches(["weg2-0-6"])
        self.assertEqual(out["weg2-0-6"], g.anchor)


if __name__ == "__main__":
    unittest.main()
