"""NF-H16P PRESENCE 1015: the MAMBA-ANCHOR presence-probe fix (d1941d1014) on
the h16p line (226aad7105), replayed with the numbers of TODAY's two cases of
its class on h16p boots.

SPECIMEN 1 (boot ..._226aad7105_1015_124245, P PP0 12:48:22, rid weg2-0-5,
prompt 185038): device match 65536 (with state), the store holding the shared
prefix with its end anchor at 184768 (front ``L3-INDEX-PRESENCE depth=184768
kv_pages=2887``; weg2-0-3 had just read exactly that depth on every stage).
PP0 asked ``H108 PRESENCE-PROBE keys=own span_tokens=119501 matched=65536
pages=0 present=False`` / ``#1035 R13 EMPTY KV PREFIX (keys=1024)`` ->
``#1400 STORE-TOLD INTAKE verdict=declined:store_absent`` -> P recomputed
119502 tokens (front ``WEG2-USAGE-TRUE p_computed=119502``). The anchor is page
1863 of the span: BOTH differences of d1941d1014 bite (unigram key, 1024-page
batch).

SPECIMEN 2 (boot ..._226aad7105_1015_122948, P PP0 12:38:50, rid weg2-3-19,
prompt 88078): device 3008, store anchor 86784 (front depth), ``H108 ...
span_tokens=85069 matched=3008 pages=0 present=False`` -> 85070 recomputed.
Anchor page 1309 of the span.

RED on 226aad7105 (``declined:store_absent``, presence 0), GREEN with the port.
The module is imported lazily so the red run fails on the assertion, not on a
missing import.
"""

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

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PS = 64
# NF runs SGLANG_HICACHE_STORAGE_BATCH=1024 (the "keys=1024" of #1035 R13); the
# CPU default is 128. Pinned here so the replay asks exactly what the boot asked.
NF_BATCH = 1024


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
        self.asked.append(len(batch))
        return self._lead(batch)

    def batch_exists_v2(self, batch, pool_transfers, extra_info=None):
        self.asked.append(len(batch))
        hit = 0
        for i in range(self._lead(batch)):
            if batch[i] in self.anchors:
                hit = i + 1
        return types.SimpleNamespace(kv_hit_pages=hit)


def _controller(store):
    ctl = types.SimpleNamespace(
        get_hash_str=get_hash_str,
        page_size=PS,
        storage_backend=store,
        _presence_pool_transfers=lambda: [object()],
    )
    ctl.store_presence_pages = HiCacheController.store_presence_pages.__get__(ctl)
    return ctl


class _Case:
    """``dev`` on the device with state; the store holds P's bigram chain up to
    ``store_end`` with anchors at the device head and at ``anchor``."""

    def __init__(self, rid, dev, prompt, anchor, store_end=None, seed=13):
        self.rid, self.dev, self.n, self.anchor = rid, dev, prompt, anchor
        store_end = anchor if store_end is None else store_end
        self.ids = [(seed * i + 5) % 248000 for i in range(prompt)]
        self.chain = get_hash_str(RadixKey(self.ids[: store_end + 1], is_bigram=True), None, page_size=PS)
        node = types.SimpleNamespace(key=RadixKey(self.ids[: dev + 1], is_bigram=True), parent=None)
        self.d_last_hash = compute_node_hash_values(node, PS)[dev // PS - 1]
        self.store = _Store(self.chain[: store_end // PS],
                            [self.chain[dev // PS - 1], self.chain[anchor // PS - 1]])

    def span(self):
        return self.ids[self.dev : self.n - 1]  # _compute_max_prefix_len = n - 1

    def span_pages(self):
        return (self.anchor - self.dev) // PS


def weg2_0_5():
    return _Case("weg2-0-5", dev=65536, prompt=185038, anchor=184768)


def weg2_3_19():
    return _Case("weg2-3-19", dev=3008, prompt=88078, anchor=86784)


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
    def __init__(self, ctl):
        self.root_node = object()
        self.cache_controller = ctl
        self.hicache_storage_pass_prefix_keys = False
        self.ongoing_prefetch = {}
        self.is_eagle = True  # P: SGLANG_HICACHE_BIGRAM_KEYS=1 (#1233)
        self.page_size = PS
        self.calls = []

    def prefetch_participation_is_collective(self):
        return False  # P: PP0's told decides, no TP vote

    def prefetch_from_storage(self, rid, node, tokens, last_hash, prefix_keys, **kw):
        self.calls.append(len(tokens))


class _Req:
    def __init__(self, c):
        self.rid = c.rid
        self.extra_key = None
        self.prefix_indices = list(range(c.dev))
        self.host_hit_length = 0
        self.full_untruncated_fill_ids = list(c.ids)
        self.last_host_node = _Node(c.d_last_hash)

    def init_next_round_input(self, tree_cache, cow_mamba=False):
        pass

    def _compute_max_prefix_len(self, n):
        return n - 1


class _Sched:
    _prefetch_kvcache = Scheduler._prefetch_kvcache
    _note_prefetch_unregistered = Scheduler._note_prefetch_unregistered

    def __init__(self, c):
        self.enable_hicache_storage = True
        self.page_size = PS
        self.tree_cache = _Tree(_controller(c.store))
        self.waiting_queue = []


class _Base(CustomTestCase):
    def setUp(self):
        for p in (mock.patch.object(ho, "read", return_value=None),
                  mock.patch.object(hpb, "current_generation", return_value=1),
                  mock.patch.object(cc, "STORAGE_BATCH_SIZE", NF_BATCH)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(cc.WEG2_HANDOFF_PAGE_KEYS.clear)


class TestTodaysCasesAreNotStoreAbsent(_Base):
    def _intake(self, c):
        s = _Sched(c)
        req = _Req(c)
        return s, req, s._prefetch_kvcache(req)

    def test_geometry_is_the_boot_geometry(self):
        # the replay asks what the boot asked: span_tokens and the page of the
        # anchor beyond the 1024-page batch (both differences of d1941d1014)
        for c, span_tokens in ((weg2_0_5(), 119501), (weg2_3_19(), 85069)):
            self.assertEqual(len(c.span()), span_tokens, c.rid)
            self.assertGreater(c.span_pages(), NF_BATCH, c.rid)

    def test_weg2_0_5_reads_instead_of_recomputing_119502(self):
        c = weg2_0_5()
        s, req, verdict = self._intake(c)
        self.assertNotEqual(verdict, "declined:store_absent")
        self.assertTrue(req._pp_store_presence_cache[1], "present=True")
        self.assertEqual(s.tree_cache.calls[-1], c.n - 1 - c.dev)  # the read asks the whole span

    def test_weg2_3_19_reads_instead_of_recomputing_85070(self):
        c = weg2_3_19()
        s, req, verdict = self._intake(c)
        self.assertNotEqual(verdict, "declined:store_absent")
        self.assertTrue(req._pp_store_presence_cache[1], "present=True")

    def test_presence_reaches_the_store_anchor(self):
        # the anchored depth the probe answers: weg2-0-5 -> 184768 (1863 pages
        # above the device head), weg2-3-19 -> 86784 (1309 pages)
        for c in (weg2_0_5(), weg2_3_19()):
            key = RadixKey(c.span(), is_bigram=True).page_aligned(PS)
            pages = _controller(c.store).store_presence_pages(key, c.d_last_hash, None)
            self.assertEqual(c.dev + pages * PS, c.anchor, c.rid)
            self.assertGreater(c.store.asked[-1], NF_BATCH, c.rid)


class TestEveryStageAnswersAlike(_Base):
    def test_pp0_and_followers_reach_the_same_verdict(self):
        # raenge-nie-uneins: PP0 (intake) and both followers (#1400
        # FOLLOWER-EARLY-READ calls the same _prefetch_kvcache) hash the same
        # content in the same key form -- one answer, PP0's told stays the
        # decision. Three independent schedulers over one store content.
        c = weg2_0_5()
        out = set()
        for _stage in range(3):
            s = _Sched(c)
            req = _Req(c)
            v = s._prefetch_kvcache(req)
            calls = s.tree_cache.calls
            out.add((v, bool(req._pp_store_presence_cache[1]), calls[-1] if calls else None))
        # (alike on the base too -- alike absent; the port makes them alike present)
        self.assertEqual(len(out), 1, out)
        self.assertTrue(next(iter(out))[1], "present on every stage")


class TestNotItsClass(_Base):
    def test_store_without_anchor_beyond_the_head_stays_absent_of_depth(self):
        # KV and state stand or fall together: the store holds KV to 184768
        # but no anchor above the device head -> the probe credits nothing
        # above the head (the told-fallback / arena_full cases of today are
        # NOT this class and are not changed by this port)
        c = weg2_0_5()
        c.store.anchors = {c.chain[c.dev // PS - 1]}
        key = RadixKey(c.span(), is_bigram=True).page_aligned(PS)
        self.assertEqual(_controller(c.store).store_presence_pages(key, c.d_last_hash, None), 0)


if __name__ == "__main__":
    unittest.main()
