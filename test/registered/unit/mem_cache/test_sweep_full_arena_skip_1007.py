"""SWEEP-FULL-ARENA (NF boot 1007_2142, P->D flip epoch 2: 54.3 s, cards idle).

THE MEASUREMENT (P.log 21:47:43-21:48:34). After a 540,884-token P phase the
KV arena (4.75 GiB, 6485 slots) was full of pages P's own tree referenced
(``ARENA-REF-HOLDERS at=reset arena_complete=6485``, ``RESET-RELEASE
released=6494``). Every publish sweep on every PP rank -- the bubble publisher
on each idle loop pass, the quiesce's ``#1470 FLUSH-PUBLISH`` on each
``/flush_cache`` poll, the sleep leg's flush -- walked 58 un-backed nodes and
made one claim per node: ``ARENA-CLAIM REFUSED statuses=[4]``, then
``_evict_for_claim`` over the hand-off keep list (``ARENA-DROP need=64
freed=0``, 291 of them on PP0 in 20 s), ~70 ms each: ``PUBLISH-SWEEP
unbacked=58 issued=0 refused=58 sweep_ms=4143``. A PP loop pass took 4.1 s,
the #1268 idle vote crawled round the ring, PP0 answered every quiesce poll
after 8.3 s with 400 (five polls: drain+quiesce=41730 ms), and the sleep leg
paid the same sweep on PP1 and PP2 in turn (sleep-kv 8883 ms).

After this sweep's first ``#1421 arena_claim`` refusal, a node whose claim the
arena provably cannot serve now (``ArenaMHAHostPool.claim_would_refuse``) is
counted refused without the claim. The same nodes are backed, the same ones
refused; only the futile claims go. Switch
``FLLIPER_PDFLIP_ENABLE_SWEEP_FULL_ARENA_SKIP`` (default on).
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import types
import unittest
from unittest import mock

import torch

from flliper.srt.environ import envs
from flliper.srt.mem_cache import form_a_host_shadow
from flliper.srt.mem_cache.base_prefix_cache import InsertParams
from flliper.srt.mem_cache.memory_pool_host import HostPoolGroup
from flliper.srt.mem_cache.pool_host import arena_pool
from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from flliper.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

CHAIN = 3      # nodes per chain: root -> head -> mid -> tail
SPAN = 16      # tokens per node (page_size 1: one page hash per token)


class _Controller:
    write_policy = "write_back"

    def __init__(self):
        self.copied = []
        self.mem_pool_host = types.SimpleNamespace(available_size=lambda: 1 << 20)

    def write(self, device_indices, node_id=None, extra_pools=None, host_indices=None):
        self.copied.append(node_id)
        return host_indices


class _DirectPool:
    """The KV arena as the sweep sees it: a claim is refused unless the node's
    first page hash is in ``room`` (a full arena with no unreferenced slot);
    ``claim_would_refuse`` answers the same question without claiming."""

    def __init__(self, room=()):
        self.room = set(room)
        self.arena = object()   # bound; no secure_rows_to_l3 -> the W3 spill is inert
        self.claims = []
        self.asked = []
        self._next = 10_000

    def alloc_write(self, hashes):
        self.claims.append(hashes[0])
        if hashes[0] not in self.room:
            return None
        rows = torch.arange(self._next, self._next + len(hashes), dtype=torch.int64)
        self._next += len(hashes)
        return rows

    def claim_would_refuse(self, hashes):
        self.asked.append(hashes[0])
        return hashes[0] not in self.room


def _tree():
    """Two chains of CHAIN un-backed device nodes, every node with its page hashes."""
    cache, allocator, _ = build_fixture(CacheConfig(page_size=1, components=(ComponentType.FULL,)))
    for first in (1, 5000):
        for depth in range(1, CHAIN + 1):
            n = depth * SPAN
            cache.insert(InsertParams(
                key=RadixKey(list(range(first, first + n)), None),
                value=allocator.alloc(n).to(dtype=torch.int64),
            ))
    for node in cache._collect_all_nodes():
        if node is not cache.root_node:
            node.hash_value = ["h%d-%d" % (node.id, i) for i in range(len(node.key))]
    cache.cache_controller = _Controller()
    return cache


def _chains(cache):
    heads = list(cache.root_node.children.values())
    out = []
    for head in heads:
        chain = [head]
        while chain[-1].children:
            chain.append(next(iter(chain[-1].children.values())))
        out.append(chain)
    return out


def _sweep(cache, pool, *, on=True):
    with envs.FLLIPER_PDFLIP_ENABLE_SWEEP_FULL_ARENA_SKIP.override(on), \
            mock.patch.object(UnifiedRadixCache, "_pdflip_direct_pool", lambda self: pool), \
            mock.patch.object(UnifiedRadixCache, "_mamba_write_through_pin_admissible",
                              lambda self, node, write_back=False: True):
        return cache.publish_unbacked_sweep(max_issue=256)


def _backed_ids(cache):
    """Backed nodes by POSITION in the tree: node ids come from a process-wide
    counter, so two trees built one after the other never share ids."""
    nodes = [n for n in cache._collect_all_nodes() if n is not cache.root_node]
    base = min(n.id for n in nodes)
    return sorted(n.id - base for n in nodes if n.backuped)


class TestAFullArenaIsClaimedOncePerSweep(CustomTestCase):
    def test_1007_2142_one_claim_not_one_per_unbacked_node(self):
        """RED before the fix: 6 claims (one per un-backed node, each one the
        ~70 ms keep-list walk of the metal), all refused."""
        cache = _tree()
        pool = _DirectPool(room=())
        stats = _sweep(cache, pool, on=True)
        self.assertEqual(len(pool.claims), 1, "a full arena is asked once by a claim, then by its room")
        self.assertEqual((stats["unbacked"], stats["issued"], stats["refused"]), (2 * CHAIN, 0, 2 * CHAIN),
                         "the sweep's census is unchanged: every un-backed node counted, every one refused")
        self.assertEqual(stats["arena_full_skipped"], 2 * CHAIN - 1)
        self.assertEqual(cache.cache_controller.copied, [])
        self.assertEqual(_backed_ids(cache), [])

    def test_switch_off_claims_every_node_as_before(self):
        cache = _tree()
        pool = _DirectPool(room=())
        stats = _sweep(cache, pool, on=False)
        self.assertEqual(len(pool.claims), 2 * CHAIN)
        self.assertEqual(pool.asked, [])
        self.assertNotIn("arena_full_skipped", stats)
        self.assertEqual((stats["unbacked"], stats["issued"], stats["refused"]), (2 * CHAIN, 0, 2 * CHAIN))

    def test_a_chain_the_arena_can_still_serve_is_backed_exactly_as_before(self):
        """Exactness: room for the second chain's pages -> it is backed with the
        switch on just as with it off; only the first chain's futile claims go."""
        results = {}
        for on in (False, True):
            cache = _tree()
            (_, second) = _chains(cache)
            pool = _DirectPool(room={n.hash_value[0] for n in second})
            stats = _sweep(cache, pool, on=on)
            results[on] = (_backed_ids(cache), stats["issued"], stats["refused"], stats["unbacked"])
        self.assertEqual(results[True], results[False])
        self.assertEqual(results[True][1], CHAIN)

    def test_nothing_is_asked_while_the_arena_has_room(self):
        """Flip unchanged: no arena_claim refusal in the sweep -> no room question at all."""
        cache = _tree()
        pool = _DirectPool(room={n.hash_value[0] for c in _chains(cache) for n in c})
        stats = _sweep(cache, pool, on=True)
        self.assertEqual(pool.asked, [])
        self.assertEqual(stats["issued"], 2 * CHAIN)
        self.assertNotIn("arena_full_skipped", stats)

    def test_a_recording_d_park_and_the_form_a_shadow_keep_every_claim(self):
        cache = _tree()
        pool = _DirectPool(room=())
        cache._pdflip_park_track = {}
        _sweep(cache, pool, on=True)
        self.assertEqual(len(pool.claims), 2 * CHAIN, "HY records every refused park backup")
        cache2 = _tree()
        pool2 = _DirectPool(room=())
        with mock.patch.object(form_a_host_shadow, "role", lambda: "host"):
            _sweep(cache2, pool2, on=True)
        self.assertEqual(len(pool2.claims), 2 * CHAIN, "R12 mirrors every refusal TP0 records")

    def test_a_w3_spill_that_releases_room_gets_the_claim(self):
        """A spill that frees room keeps every node's claim, exactly as with the
        switch off (a claim and its retry after the spill, per node)."""
        claims = {}
        for on in (False, True):
            cache = _tree()
            pool = _DirectPool(room=())
            with mock.patch.object(UnifiedRadixCache, "_w3_arena_spill", lambda self, p, n, claimer=None: 1):
                _sweep(cache, pool, on=on)
            claims[on] = len(pool.claims)
        self.assertEqual(claims[True], claims[False])
        self.assertGreaterEqual(claims[True], 2 * CHAIN)


class _Arena:
    """ShmArena's read side: states by stem, stats, ref census, partial reap."""

    def __init__(self, *, slots, complete, claimed, referenced, present=(), orphans=0):
        self.slots, self.complete, self.claimed, self.referenced = slots, complete, claimed, referenced
        self.present = set(present)
        self.orphans = orphans
        self.reaped = 0

    def find_states(self, stems):
        return [2 if s in self.present else 0 for s in stems]

    def stats(self):
        return {"slots": self.slots, "complete": self.complete, "claimed": self.claimed}

    def ref_census(self):
        return (self.referenced, self.referenced, self.complete)

    def reap_partial(self, age):
        freed, self.orphans = list(range(self.orphans)), 0
        self.claimed -= len(freed)
        self.reaped += len(freed)
        return freed


def _pool(arena):
    pool = ArenaMHAHostPool.__new__(ArenaMHAHostPool)
    pool.arena = arena
    pool._backend = types.SimpleNamespace(_get_suffixed_key=lambda h: "s-" + h)
    return pool


class TestClaimWouldRefuse(CustomTestCase):
    def test_full_and_every_slot_referenced(self):
        arena = _Arena(slots=8, complete=8, claimed=0, referenced=8)
        self.assertTrue(_pool(arena).claim_would_refuse(["a", "b"]))

    def test_free_slots_serve_the_absent_pages_without_room_making(self):
        arena = _Arena(slots=8, complete=6, claimed=0, referenced=6, orphans=1)
        self.assertFalse(_pool(arena).claim_would_refuse(["a", "b"]))
        self.assertEqual(arena.reaped, 0, "no claim would have reaped: nothing is taken here either")

    def test_unreferenced_complete_slots_are_room(self):
        arena = _Arena(slots=8, complete=8, claimed=0, referenced=6)
        self.assertFalse(_pool(arena).claim_would_refuse(["a", "b"]))
        self.assertTrue(_pool(arena).claim_would_refuse(["a", "b", "c"]))

    def test_pages_already_in_the_arena_need_no_room(self):
        arena = _Arena(slots=8, complete=8, claimed=0, referenced=8, present={"s-a", "s-b"})
        self.assertFalse(_pool(arena).claim_would_refuse(["a", "b"]))

    def test_orphan_claims_the_claim_would_reap_are_room(self):
        arena = _Arena(slots=8, complete=6, claimed=2, referenced=6, orphans=2)
        with mock.patch.object(arena_pool, "_partial_reap_age_s", lambda: 30.0):
            self.assertFalse(_pool(arena).claim_would_refuse(["a", "b"]))
        self.assertEqual(arena.reaped, 2)

    def test_the_bound_tier_wrapper_hands_it_through(self):
        group = HostPoolGroup.__new__(HostPoolGroup)
        host = types.SimpleNamespace(claim_would_refuse=lambda hashes: "asked")
        object.__setattr__(group, "anchor_entry", types.SimpleNamespace(host_pool=host))
        self.assertEqual(group.claim_would_refuse(["a"]), "asked")


if __name__ == "__main__":
    unittest.main()
