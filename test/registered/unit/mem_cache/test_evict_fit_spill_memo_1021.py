# SPDX-License-Identifier: Apache-2.0
"""EVICT-FIT + SPILL-MEMO (NF rc27 reboot 1021_023225; nf-rc29-pphase-1021 W-D / W-C).

EVICT-FIT (pdflip/evict_fit.py). Metal: PP0 18254 ``SF LOADBACK-ROOM SAME-PASS rid=pdflip-0-19 kv_tokens=113408
floor=85760 evicted=165632`` and PP1 18600 ``LB-VERDICT ROW-ROOM ... evicted=165632`` for a need of 27648 / 27520 --
the peel demoted the backed node 9 / 18 (151360 tokens) WHOLE (P 18252 / 18598 ``#1469 EVICT node=.. backuped=True``).
Now a backed leaf longer than the remaining need is split and only the tail leaves the device. Tested on a REAL
UnifiedRadixCache (CPU fixture), so ``_split_node`` and the demotion run as on the metal.

SPILL-MEMO (pdflip/spill_memo.py). Metal: PP1 18488/18507, PP2 18561/18605 -- RETAIN-PUBLISH issue_ms 498-827 for 20
SKIPPED nodes, every skip re-running the W3 room-set spill walk with the same answer 0. Now a decided empty spill is
not walked again until the arena/tree state changes.
"""

import os
import types
import unittest
from unittest import mock

import torch

from flliper.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.pdflip import evict_fit as FIT
from flliper.srt.pdflip import pp_slot_fidelity as SF
from flliper.srt.pdflip import spill_memo as SM
from flliper.srt.pdflip import ud_last as UDL

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

FULL = ComponentType.FULL


def _env(**kv):
    return mock.patch.dict(os.environ, {k: str(v) for k, v in kv.items()})


class _Tree:
    """A real CPU tree on the local-PP floor, write_back policy, every new backup refused (arena full)."""

    def __init__(self):
        cfg = CacheConfig(page_size=1, components=(ComponentType.FULL, ComponentType.MAMBA), mamba_cache_size=8,
                          kv_size=1024, max_context_len=1024)
        self.cache, self.alloc, _ = build_fixture(cfg)
        c = self.cache
        setattr(c, SF.FLOOR_LOCAL_PP_ATTR, True)
        c.cache_controller = types.SimpleNamespace(write_policy="write_back")
        c.write_backup = lambda node, write_back=False, kv_only_if_mamba_refused=False: 0
        c.writing_check = lambda write_back=False: None

    def insert(self, tokens):
        v = self.alloc.alloc(len(tokens))
        assert v is not None
        self.cache.insert(InsertParams(key=RadixKey(list(tokens), None), value=v.to(dtype=torch.int64)))

    def node_at(self, first_token, depth):
        """The node whose key starts the path at ``first_token`` and covers ``depth`` tokens from the root."""
        node, d = self.cache.root_node, 0
        while d < depth:
            node = next(n for n in node.children.values() if (d > 0 or n.key.token_ids[0] == first_token))
            d += len(node.key)
        return node

    def back(self, node):
        """Mark ``node`` host-backed for its whole length (the host rows the arena would hold)."""
        n = len(node.key)
        node.component_data[FULL].host_value = torch.arange(5000, 5000 + n, dtype=torch.int64)

    def evict(self, n):
        return int(self.cache.evict(EvictParams(num_tokens=n)).num_tokens_evicted)


class EvictFit(unittest.TestCase):
    def test_backed_leaf_is_split_only_the_need_is_demoted(self):
        """DANGER DIRECTION 'more evicted than needed': a 200-token backed leaf pays a 30-token need with 30."""
        with _env(**{SF.ENV: 1}):
            t = _Tree()
            t.insert(range(1, 201))
            leaf = t.node_at(1, 200)
            t.back(leaf)
            ev0 = t.cache.evictable_size()
            got = t.evict(30)
        self.assertEqual(got, 30, "the peel evicted more than the request needed")
        self.assertEqual(len(leaf.key), 30, "the node object is the demoted tail")
        self.assertTrue(leaf.evicted)
        head = leaf.parent
        self.assertEqual(len(head.key), 170)
        self.assertIsNotNone(head.component_data[FULL].value, "the head stays on the device")
        self.assertIsNotNone(head.component_data[FULL].host_value, "and keeps its host rows")
        self.assertEqual(t.cache.evictable_size(), ev0 - 30)

    def test_switch_off_demotes_the_whole_node(self):
        with _env(**{SF.ENV: 1, FIT.ENV: 0}):
            t = _Tree()
            t.insert(range(1, 201))
            t.back(t.node_at(1, 200))
            self.assertEqual(t.evict(30), 200)

    def test_group_floor_unchanged(self):
        with _env(**{SF.ENV: 1}):
            t = _Tree()
            setattr(t.cache, SF.FLOOR_LOCAL_PP_ATTR, False)
            t.insert(range(1, 201))
            t.back(t.node_at(1, 200))
            self.assertEqual(t.evict(30), 200, "off the local-PP floor (TP group / Form A) the node goes whole")

    def test_metal_chain_pass_two_then_fit(self):
        """The metal shape scaled: backed 151 <- un-backed 14 (the fresh tail), need 27. No other leaf: pass 2 drops
        the tail (UD-LAST DROP-AFTER), the backed parent is split for the remaining 13 -- 27 evicted, not 165."""
        with _env(**{SF.ENV: 1}):
            t = _Tree()
            t.insert(range(1, 152))
            t.insert(list(range(1, 152)) + list(range(1000, 1014)))
            a = t.node_at(1, 151)
            t.back(a)
            got = t.evict(27)
        self.assertEqual(got, 27)
        self.assertEqual(len(a.key), 13)
        self.assertTrue(a.evicted)
        self.assertEqual(len(a.parent.key), 138)
        self.assertIsNotNone(a.parent.component_data[FULL].value)

    def test_metal_chain_with_a_backed_leaf_elsewhere_drops_nothing(self):
        """UD-LAST + EVICT-FIT: the un-backed tail is held, the other backed leaf pays exactly the need."""
        with _env(**{SF.ENV: 1}):
            t = _Tree()
            t.insert(range(1, 152))
            t.insert(list(range(1, 152)) + list(range(1000, 1014)))
            t.insert(range(2000, 2040))
            t.back(t.node_at(1, 151))
            other = t.node_at(2000, 40)
            t.back(other)
            tail = t.node_at(1, 165)
            got = t.evict(27)
        self.assertEqual(got, 27)
        self.assertFalse(tail.evicted, "the un-backed fresh tail stays on the device")
        self.assertIsNotNone(tail.component_data[FULL].value)
        self.assertEqual(len(other.key), 27)
        self.assertTrue(other.evicted)
        self.assertEqual(len(other.parent.key), 13)

    def test_locked_or_partly_backed_leaf_is_not_split(self):
        with _env(**{SF.ENV: 1}):
            t = _Tree()
            t.insert(range(1, 201))
            leaf = t.node_at(1, 200)
            leaf.component_data[FULL].host_value = torch.arange(0, 100, dtype=torch.int64)   # partly backed
            self.assertFalse(FIT.fit(t.cache, leaf, 30))
            t.back(leaf)
            leaf.component_data[FULL].host_lock_ref = 1
            self.assertFalse(FIT.fit(t.cache, leaf, 30), "a load reads its host rows")
            leaf.component_data[FULL].host_lock_ref = 0
            self.assertFalse(FIT.fit(t.cache, leaf, 200), "nothing to keep")
            self.assertTrue(FIT.fit(t.cache, leaf, 30))

    def test_switch_default_on(self):
        self.assertTrue(FIT.on({}))
        self.assertTrue(SM.on({}))
        self.assertTrue(UDL.on({}))


class _Arena:
    def __init__(self):
        self.slots, self.complete, self.claimed, self.referenced, self.refs = 6485, 6485, 0, 6485, 18172
        self.path = None

    def stats(self):
        return {"slots": self.slots, "complete": self.complete, "claimed": self.claimed}

    def ref_census(self):
        return self.referenced, self.refs, self.complete


def _follower():
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    class C:
        W3_SPILL_MIN_PAGES = UnifiedRadixCache.W3_SPILL_MIN_PAGES
        W3_SPILL_EMPTY_BACKOFF_S = UnifiedRadixCache.W3_SPILL_EMPTY_BACKOFF_S
        _w3_arena_spill = UnifiedRadixCache._w3_arena_spill

    c = C()
    c.pp_size, c.pp_rank = 3, 1
    c.component_evictable_size_ = {FULL: 419520}
    c.component_protected_size_ = {FULL: 0}
    c.evictable_device_leaves = {1, 2}
    c.evictable_host_leaves = set()
    pool = types.SimpleNamespace(arena=_Arena(), secure_rows_to_l3=lambda *a, **k: None)
    return c, pool


class SpillMemo(unittest.TestCase):
    def _run(self, c, pool, n, released=0, undecided=False):
        from flliper.srt.pdflip import w3_room_set as W

        calls = []

        def spill(cache, p, need, want, released_out=None, claimer=None):
            calls.append(need)
            cache._w3_room_undecided = undecided
            return released

        with mock.patch.object(W, "spill", spill), mock.patch.object(W, "armed", lambda cache: True), \
                mock.patch.object(W, "_debts", lambda cache: {}):
            for _ in range(n):
                c._w3_arena_spill(pool, 64)
        return calls

    def test_refused_spill_is_not_walked_again_without_a_state_change(self):
        """DANGER DIRECTION 'refused attempt repeated without a state change': 20 skips of one retain publish walk
        the spill once."""
        with _env(**{SM.ENV: 1}):
            c, pool = _follower()
            calls = self._run(c, pool, 20)
        self.assertEqual(len(calls), 1, "the identical empty walk was repeated on the scheduler thread")
        walks, skips, ms = SM.counters(c)
        self.assertEqual((walks, skips), (1, 19))

    def test_any_state_change_walks_again(self):
        with _env(**{SM.ENV: 1}):
            for change in ("unref", "free", "set", "lock", "leaf"):
                c, pool = _follower()
                self.assertEqual(len(self._run(c, pool, 1)), 1)
                a = pool.arena
                if change == "unref":
                    a.referenced -= 1
                    a.refs -= 1
                elif change == "free":
                    a.complete -= 1
                elif change == "set":
                    c._w3_room_seen = (7, 2)
                elif change == "lock":
                    c.component_protected_size_[FULL] = 64
                else:
                    c.evictable_device_leaves.add(3)
                self.assertEqual(len(self._run(c, pool, 1)), 1, change)

    def test_undecided_and_released_are_never_memoised(self):
        with _env(**{SM.ENV: 1}):
            c, pool = _follower()
            self.assertEqual(len(self._run(c, pool, 3, undecided=True)), 3, "an undecided spill resolves by acks")
            c, pool = _follower()
            self.assertEqual(len(self._run(c, pool, 3, released=5)), 3)

    def test_switch_off_walks_every_time(self):
        with _env(**{SM.ENV: 0}):
            c, pool = _follower()
            self.assertEqual(len(self._run(c, pool, 5)), 5)


if __name__ == "__main__":
    unittest.main()
