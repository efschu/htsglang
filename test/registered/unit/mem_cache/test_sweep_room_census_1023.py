"""SWEEPCENSUS-1023 (27B int4ar boot 1023_223241, D TP0/TP1 23:26:52-23:35:26Z).

THE MEASUREMENT. At 23:26:43 the FULL KV arena was full (``ARENA-REF-CENSUS
arena-32768.bin slots=720896 complete=720771``). From then on every finished
request's S108 turn end (``_s108_turn_end_l2`` -> ``publish_unbacked_sweep``
over the request's chain, parents first) ran into the SWEEP-FULL-ARENA room
question once PER NODE: ``_pdflip_full_arena_refuses`` -> ``claim_would_refuse``
-> ``_reap_orphan_claims`` + ``ref_census``, a strided read of all 720896 slot
headers each time. The chain's nodes below a refused top get the SAME top,
so the same question was asked ~240 times per turn end. The D decode gaps
carried it in their ``result`` span (process_batch_result): 0.0-0.5 s per
5 min before 23:25, then 116.7 s and 218.4 s; 10 of 10 WEDGE-STACK lines
(FORWARD-HANG mode=DECODE bs=3) stood in ref_census / find_states /
reap_partial.

THE FIX. One room memo per sweep: a top refused under the current room
generation stays refused for its children, and the census room is read
once; every claim, release or spill inside the sweep bumps the generation
and voids both (never a stale permission).

Danger directions tested here: (1) a sweep over 240 chain nodes at a full
arena reads the census O(1) times, not once per node; (2) a room change in
the middle of the sweep changes the answer (a spill that frees room gets the
claims, a claim that uses the room up gets the next node refused without a
futile claim).
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import types
import unittest
from unittest import mock

import torch

from flliper.srt.environ import envs
from flliper.srt.mem_cache.base_prefix_cache import InsertParams
from flliper.srt.mem_cache.pool_host import arena_pool
from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache import unified_radix_cache as urc
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from flliper.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

CHAIN_LEN = 240   # nodes of one finished turn's chain (60k tokens / 256 on the metal)


class _Controller:
    write_policy = "write_back"

    def __init__(self):
        self.copied = []
        self.mem_pool_host = types.SimpleNamespace(available_size=lambda: 1 << 20)

    def write(self, device_indices, node_id=None, extra_pools=None, host_indices=None):
        self.copied.append(node_id)
        return host_indices


class _Arena:
    """ShmArena's read side, with the census reads counted. Room = free +
    unreferenced COMPLETE slots, as the claim's eviction clock sees it."""

    def __init__(self, *, slots, complete, referenced):
        self.slots, self.complete, self.claimed, self.referenced = slots, complete, 0, referenced
        self.census_reads = 0

    def find_states(self, stems):
        return [0 for _ in stems]

    def stats(self):
        return {"slots": self.slots, "complete": self.complete, "claimed": self.claimed}

    def ref_census(self):
        self.census_reads += 1
        return (self.referenced, self.referenced, self.complete)

    def reap_partial(self, age):
        return []

    def room(self):
        return self.slots - self.complete - self.claimed + self.complete - self.referenced

    def take(self, n):
        free = self.slots - self.complete - self.claimed
        fresh = min(free, n)
        self.complete += fresh
        self.referenced += n


class _Pool:
    """The KV direct pool the sweep sees: claims against the counted arena, the
    room question through the REAL ``ArenaMHAHostPool.claim_would_refuse``."""

    def __init__(self, arena):
        self.arena = arena
        self.claims = []
        self.asked = 0
        self._real = ArenaMHAHostPool.__new__(ArenaMHAHostPool)
        self._real.arena = arena
        self._real._backend = types.SimpleNamespace(_get_suffixed_key=lambda h: "s-" + h)
        self._real._pdflip_reaps_orphan_claims = False
        self._next = 10_000

    def alloc_write(self, hashes):
        self.claims.append(hashes[0])
        if len(hashes) > self.arena.room():
            return None
        self.arena.take(len(hashes))
        rows = torch.arange(self._next, self._next + len(hashes), dtype=torch.int64)
        self._next += len(hashes)
        return rows

    def claim_would_refuse(self, hashes, room_cache=None):
        self.asked += 1
        return self._real.claim_would_refuse(hashes, room_cache=room_cache)


def _chain_tree(n_nodes):
    """One chain root -> n_1 -> ... -> n_N of one-token device nodes (page_size 1:
    one page hash per node), none backed: a finished turn's chain."""
    cache, allocator, _ = build_fixture(
        CacheConfig(page_size=1, components=(ComponentType.FULL,), kv_size=4 * n_nodes + 64))
    values = allocator.alloc(n_nodes).to(dtype=torch.int64)
    for d in range(1, n_nodes + 1):
        cache.insert(InsertParams(key=RadixKey(list(range(1, d + 1)), None), value=values[:d]))
    for node in cache._collect_all_nodes():
        if node is not cache.root_node:
            node.hash_value = ["h%d-%d" % (node.id, i) for i in range(len(node.key))]
    cache.cache_controller = _Controller()
    chain = []
    node = cache.root_node
    while node.children:
        node = next(iter(node.children.values()))
        chain.append(node)
    return cache, chain


def _heads_tree(sizes):
    """Independent heads under the root (each its own top), one per size."""
    total = sum(sizes)
    cache, allocator, _ = build_fixture(
        CacheConfig(page_size=1, components=(ComponentType.FULL,), kv_size=total + 64))
    heads = []
    first = 1
    for n in sizes:
        cache.insert(InsertParams(key=RadixKey(list(range(first, first + n)), None),
                                  value=allocator.alloc(n).to(dtype=torch.int64)))
        first += 10_000
    for node in cache._collect_all_nodes():
        if node is not cache.root_node:
            node.hash_value = ["h%d-%d" % (node.id, i) for i in range(len(node.key))]
    cache.cache_controller = _Controller()
    by_first = {}
    for node in cache.root_node.children.values():
        by_first[node.key.token_ids[0] if hasattr(node.key, "token_ids") else node.key[0]] = node
    for i, n in enumerate(sizes):
        heads.append(by_first[1 + 10_000 * i])
    return cache, heads


def _sweep(cache, pool, first, spill=None):
    spill = spill or (lambda self, p, n, claimer=None: 0)
    with envs.FLLIPER_PDFLIP_ENABLE_SWEEP_FULL_ARENA_SKIP.override(True), \
            mock.patch.object(UnifiedRadixCache, "_pdflip_direct_pool", lambda self: pool), \
            mock.patch.object(UnifiedRadixCache, "_w3_arena_spill", spill), \
            mock.patch.object(UnifiedRadixCache, "_mamba_write_through_pin_admissible",
                              lambda self, node, write_back=False: True):
        return cache.publish_unbacked_sweep(max_issue=10_000, clock=None, first=list(first), chain_only=True)


class TestOneCensusPerSweep(CustomTestCase):
    def test_1023_223241_full_arena_chain_reads_the_census_once(self):
        """RED before the fix: 239 census reads (one per node after the first
        refused claim), each a strided read of every slot header on the metal."""
        arena = _Arena(slots=720_896, complete=720_896, referenced=720_896)
        pool = _Pool(arena)
        cache, chain = _chain_tree(CHAIN_LEN)
        census0 = arena_pool._ROOM_CENSUS_N[0]
        stats = _sweep(cache, pool, chain)
        self.assertEqual(len(pool.claims), 1, "one futile claim, then the room question")
        self.assertLessEqual(arena.census_reads, 1, "O(1) header census reads per sweep, not one per node")
        self.assertEqual(arena_pool._ROOM_CENSUS_N[0] - census0, arena.census_reads)
        self.assertLessEqual(pool.asked, 2, "the chain below a refused top is answered from the memo")
        self.assertEqual((stats["unbacked"], stats["issued"], stats["refused"]), (CHAIN_LEN, 0, CHAIN_LEN),
                         "the same nodes counted and refused as before")
        self.assertEqual(stats["arena_full_skipped"], CHAIN_LEN - 1)
        self.assertGreaterEqual(stats["room_memo_hits"], CHAIN_LEN - 2)
        self.assertNotIn("_pdflip_sweep_room_memo", cache.__dict__, "the memo dies with its sweep")

    def test_the_memo_does_not_outlive_its_sweep(self):
        """A second sweep reads the census again (the room may have moved between turn ends)."""
        arena = _Arena(slots=64, complete=64, referenced=64)
        pool = _Pool(arena)
        cache, chain = _chain_tree(8)
        _sweep(cache, pool, chain)
        first_reads = arena.census_reads
        _sweep(cache, pool, chain)
        self.assertEqual(arena.census_reads, 2 * first_reads)
        self.assertGreaterEqual(first_reads, 1)

    def test_without_the_memo_every_node_asks(self):
        """The fix's own baseline: with the memo absent (as before), the chain asks per node."""
        arena = _Arena(slots=64, complete=64, referenced=64)
        pool = _Pool(arena)
        cache, chain = _chain_tree(12)
        with mock.patch.object(urc, "_sweep_room_memo", lambda: None):
            stats = _sweep(cache, pool, chain)
        self.assertEqual(arena.census_reads, 11)
        self.assertEqual(stats["refused"], 12)


class TestARoomChangeChangesTheAnswer(CustomTestCase):
    def test_a_spill_that_frees_room_mid_sweep_gets_the_claims(self):
        """The first room question's spill releases room: the memo's refusal and
        census are void, and the chain is claimed from there (never a stale
        refusal of a freed room)."""
        arena = _Arena(slots=64, complete=64, referenced=64)
        pool = _Pool(arena)
        cache, chain = _chain_tree(10)
        state = {"n": 0}

        def spill(self, p, n, claimer=None):
            state["n"] += 1
            if state["n"] == 1:
                arena.referenced -= 32   # the spill let 32 COMPLETE slots go
                return 32
            return 0

        stats = _sweep(cache, pool, chain, spill=spill)
        self.assertEqual(stats["issued"], len(chain) - 1, "the room the spill freed serves the rest of the chain")
        self.assertEqual(stats["refused"], 1, "only the first claim, before the spill, was refused")
        self.assertGreaterEqual(arena.census_reads, 1, "after the room moved, the census is read again")

    def test_a_claim_that_uses_the_room_up_voids_the_cached_room(self):
        """Heads: big (64 pages, refused: arms the skip), A (16 pages, the room is
        exactly 16: permission, claimed), B (16 pages: must be refused WITHOUT a
        claim -- a stale cached room of 16 would have let B claim)."""
        arena = _Arena(slots=1000, complete=1000, referenced=984)   # room 16
        pool = _Pool(arena)
        cache, (big, a, b) = _heads_tree([64, 16, 16])
        stats = _sweep(cache, pool, [big, a, b])
        self.assertEqual(pool.claims, [big.hash_value[0], a.hash_value[0]],
                         "big refused by its claim, A claimed with the room, B refused by the room question")
        self.assertEqual(arena.census_reads, 2, "A's claim moved the room: B read the census again")
        self.assertEqual(stats["issued"], 1)
        self.assertEqual(stats["refused"], 2)


if __name__ == "__main__":
    unittest.main()
