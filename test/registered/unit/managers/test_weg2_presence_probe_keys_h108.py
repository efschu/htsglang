"""H108 (rc12z25 D 16:56:02, weg2-72-124): the #950 presence probe asks with
the keys the fetch reads with.

THE DEFECT: `_prefetch_kvcache` asks `store_presence_pages` whether the store
holds a span only when `last_host_node` is not backuped -- on D a D-own node,
whose hash is the tree's convention (bigram on the EAGLE tree). The probe
chained its own page hashes from that hash, the fetch reads with P's hand-off
keys (#1442), and the two never met: the probe answered 0 with every page and
P's end anchor in the store. TP0 entered the #580 vote with nothing
(`#915 PREFETCH REFUSED reason=vote_negative ... need=0`, `anchor=` +1 on TP0
only), the group recomputed the P leg's 4317-token tail on D, W16. All 7 short
after-P legs of rc12z25 (lost 4288 / 192 / 192 / 128 x4) have this shape.

RED on 85386b1df1: `store_presence_pages` takes no hand-off keys, the real
`_prefetch_kvcache` enters the vote ineligible with an empty span, the late
hand-off case is served from the cache, and #915 REFUSED names no key source.
"""

import logging
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
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Store:
    """File-backend semantics: leading KV hits, v2 min-clamped to the last page
    that also carries the mamba anchor."""

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
        lead = self._lead(batch)
        hit = 0
        for i in range(lead):
            if batch[i] in self.anchors:
                hit = i + 1
        return types.SimpleNamespace(kv_hit_pages=hit)


def _controller(store, page_size, hybrid):
    ctl = types.SimpleNamespace(
        get_hash_str=get_hash_str,
        page_size=page_size,
        storage_backend=store,
        _presence_pool_transfers=(lambda: [object()]) if hybrid else (lambda: None),
    )
    ctl.store_presence_pages = HiCacheController.store_presence_pages.__get__(ctl)
    return ctl


class _Geo:
    """A P leg handed to D: ``dev`` tokens already on D's device (a D-own node),
    P wrote the pages above it and its end anchor."""

    def __init__(self, page_size, dev_pages, p_pages, tail, bigram):
        P = page_size
        self.P = P
        self.dev = dev_pages * P
        self.n = p_pages * P + tail
        self.ids = list(range(5000, 5000 + self.n))
        # P's chain: plain convention from the root (what P publishes, #1442).
        self.chain = get_hash_str(self.ids[: p_pages * P], None, page_size=P)
        # D's own node over the device prefix, hashed as D's tree hashes it.
        node = types.SimpleNamespace(
            key=RadixKey(self.ids[: self.dev + (1 if bigram else 0)], is_bigram=bigram),
            parent=None,
        )
        self.d_last_hash = compute_node_hash_values(node, P)[dev_pages - 1]
        self.store = _Store(self.chain, anchor_keys=[self.chain[-1]])


NF = dict(page_size=64, dev_pages=396, p_pages=463, tail=29, bigram=True)  # weg2-72-124
B27 = dict(page_size=1, dev_pages=2000, p_pages=2600, tail=1, bigram=False)


class TestTheProbeAsksWithTheFetchKeys(CustomTestCase):
    def _span(self, g):
        return g.ids[g.dev : g.n - 1]

    def test_own_chain_from_a_d_node_misses_p_pages(self):
        # The mechanism, true on both sides of the fix: D's bigram node hash
        # starts a chain that meets none of P's keys.
        g = _Geo(**NF)
        ctl = _controller(g.store, g.P, hybrid=True)
        self.assertEqual(ctl.store_presence_pages(self._span(g), g.d_last_hash, None), 0)

    def test_nf_geometry_handoff_keys_see_pages_and_end_anchor(self):
        g = _Geo(**NF)
        ctl = _controller(g.store, g.P, hybrid=True)
        keys = g.chain[g.dev // g.P :]
        pages = ctl.store_presence_pages(
            self._span(g), g.d_last_hash, None, page_keys=keys
        )
        self.assertEqual(pages, len(keys))  # 67 pages up to the end anchor 29632
        self.assertEqual(g.store.asked[-1][: len(keys)], keys)

    def test_27b_geometry_dense_page1_tp3(self):
        # dense (no mamba transfer: plain batch_exists), page_size 1, three
        # symmetric D ranks asking the same span.
        g = _Geo(**B27)
        keys = g.chain[g.dev :]
        for _rank in range(3):
            ctl = _controller(g.store, g.P, hybrid=False)
            self.assertEqual(
                ctl.store_presence_pages(self._span(g), g.d_last_hash, None, page_keys=keys),
                min(len(keys), STORAGE_BATCH_SIZE),
            )

    def test_27b_geometry_without_keys_unchanged(self):
        # 27B's plain tree hash already chains onto P's keys: no hand-off record,
        # same answer as before.
        g = _Geo(**B27)
        ctl = _controller(g.store, g.P, hybrid=False)
        self.assertEqual(
            ctl.store_presence_pages(self._span(g), g.d_last_hash, None),
            min(len(g.chain) - g.dev, STORAGE_BATCH_SIZE),
        )


class _Node:
    def __init__(self, last_hash):
        self.backuped = False  # a D-own node: never written through
        self.parent = None
        self.key = None
        self._h = last_hash

    def get_last_hash_value(self):
        return self._h

    def get_prefix_hash_values(self, parent):
        return []


class _Tree:
    def __init__(self, ctl):
        self.root_node = object()
        self.cache_controller = ctl
        self.hicache_storage_pass_prefix_keys = False
        self.ongoing_prefetch = {}
        self.calls = []

    def prefetch_participation_is_collective(self):
        return True

    def prefetch_from_storage(self, rid, node, tokens, last_hash, prefix_keys, **kw):
        self.calls.append((list(tokens), kw.get("locally_eligible")))


class _Req:
    def __init__(self, g, rid):
        self.rid = rid
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

    def __init__(self, g):
        self.enable_hicache_storage = True
        self.page_size = g.P
        self.tree_cache = _Tree(_controller(g.store, g.P, hybrid=True))


class TestTP0EntersTheVoteWithTheSpan(CustomTestCase):
    """The real `_prefetch_kvcache` on the rc12z25 TP0 shape."""

    def setUp(self):
        self.records = {}
        self._p = mock.patch.object(ho, "read", side_effect=lambda rid: self.records.get(rid))
        self._p.start()
        self._g = mock.patch.object(hpb, "current_generation", return_value=3)
        self._g.start()

    def tearDown(self):
        self._p.stop()
        self._g.stop()
        cc.WEG2_HANDOFF_PAGE_KEYS.clear()

    def test_handoff_span_makes_tp0_eligible(self):
        g = _Geo(**NF)
        s = _Sched(g)
        req = _Req(g, "weg2-72-124")
        self.records[req.rid] = {"page_keys": list(g.chain)}
        s._prefetch_kvcache(req)
        tokens, eligible = s.tree_cache.calls[-1]
        self.assertTrue(eligible, "P's pages and end anchor are in the store")
        self.assertEqual(len(tokens), g.n - 1 - g.dev)  # 4316, not the empty vote
        self.assertTrue(req._pp_store_presence_cache[1])

    def test_late_handoff_is_asked_again(self):
        # xsn331: the first registration can race P's hand-off write.
        g = _Geo(**NF)
        s = _Sched(g)
        req = _Req(g, "weg2-72-124")
        s._prefetch_kvcache(req)
        self.assertEqual(s.tree_cache.calls[-1], ([], False))
        asked = len(g.store.asked)
        self.records[req.rid] = {"page_keys": list(g.chain)}
        s._prefetch_kvcache(req)
        self.assertEqual(len(g.store.asked), asked + 1, "new coverage: probe again")
        self.assertTrue(s.tree_cache.calls[-1][1])
        s._prefetch_kvcache(req)
        self.assertEqual(len(g.store.asked), asked + 1, "same coverage: cached")

    def test_non_weg2_rid_keeps_the_old_question(self):
        g = _Geo(**NF)
        s = _Sched(g)
        seen = []
        s.tree_cache.cache_controller.store_presence_pages = (
            lambda tokens, last_hash, prefix_keys: seen.append(1) or 0
        )
        req = _Req(g, "plain-rid")
        s._prefetch_kvcache(req)
        self.assertEqual(seen, [1])
        self.assertEqual(s.tree_cache.calls[-1], ([], False))


class TestRefusedLineNamesTheKeySource(CustomTestCase):
    def test_keys_term(self):
        from sglang.srt.mem_cache import unified_radix_cache as urc

        tree = types.SimpleNamespace(
            _prefetch_line_terms=lambda need: dict(
                need=need, available=0, threshold=256, occupied=0, limit=0,
                pool_id=0, epoch=0, phase="tp", generation=0,
            ),
        )
        setattr(tree, urc.PRESENCE_SRC_ATTR, {"weg2-72-124": "handoff"})
        with self.assertLogs(urc.logger, level=logging.WARNING) as cm:
            urc.UnifiedRadixCache._log_prefetch_refused(tree, "vote_negative", "weg2-72-124", 0)
            urc.UnifiedRadixCache._log_prefetch_refused(tree, "vote_negative", "weg2-9-9", 0)
        self.assertIn("keys=handoff", cm.output[0])
        self.assertIn("keys=-", cm.output[1])


if __name__ == "__main__":
    unittest.main()
