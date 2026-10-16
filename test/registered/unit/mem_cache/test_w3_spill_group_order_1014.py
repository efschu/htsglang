"""NF arena fix 1 (14.10.2026, base f4afb10f28): the W3 spill is group-equal.

THE MEASUREMENT (nf-arena-voll-analyse-1014, P PP0 03:05:11-34, P.log:15554-15602):
every ``W3-ARENA SPILL`` of PP0 handed 64 pages of references back and the
claim's next line was ``ARENA-DROP need=41 freed=0 stages=i:0,ii:0,iii:0``.
A slot is free only when EVERY rank's reference is gone; the spill ordered its
candidates by node id and trusted the PP ranks' trees to be replicas, but they
are not (PP0 candidates=31 x 64 pages, PP1 candidates=9 x 208-256 pages), so
each rank let go of other slots and no slot ever reached zero references.

THE FIX: the order is read from the arena all ranks share -- fewest reader
references first (the pages a peer already let go), then the slot id (the
canonical page identity), then the node id.

THE FAKE: one arena (a reference array), three "PP ranks" = three trees with
their own host rows over the SAME canonical slots, each with its own pool object
on that arena; releasing a node's host rows takes one reference per slot off,
as ``HostPoolGroup.free`` does. ``freed`` = slots with 0 references after the
three ranks spilled, once each, in the order PP0, PP1, PP2.
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import unittest
from unittest import mock

import torch

from flliper.srt.environ import envs
from flliper.srt.mem_cache.base_prefix_cache import InsertParams
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from flliper.srt.pdflip import pp_slot_fidelity as SF
from flliper.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

SPAN = 64          # pages per canonical chunk (= W3_SPILL_MIN_PAGES: one chunk per spill)
SLOTS = 400
X, Y, Z = 100, 100 + SPAN, 100 + 2 * SPAN      # first slot of the three canonical chunks


class _Arena:
    """The shared header: one reader-reference count per slot, over all ranks."""

    def __init__(self, held):
        self.refs = [0] * SLOTS
        for start in held:
            for s in range(start, start + SPAN):
                self.refs[s] = 3

    def slot_refs(self, slots):
        return [self.refs[int(s)] for s in slots]

    def free_slots(self):
        return sorted(s for s in range(SLOTS) if s >= X and s < Z + SPAN and self.refs[s] == 0)


class _RankPool:
    """One rank's host pool on the shared arena (P == 1: row = slot)."""

    staging_rows = 0
    arena_slots = SLOTS

    def __init__(self, arena):
        self.arena = arena
        self.secured = 0

    def secure_rows_to_l3(self, host_indices):
        n = int(host_indices.numel())
        self.secured += n
        return {"pages": n, "on_disk": n, "written": 0, "lost": 0}


def _tree(chains):
    """A rank's tree: ``chains`` = [(first_slot, n_pages), ...] in INSERTION
    order (so the node ids run in that order); each node is device-resident and
    its Full host value names the canonical slots [first_slot, first_slot+n)."""
    cache, allocator, _ = build_fixture(CacheConfig(page_size=1, components=(ComponentType.FULL,)))
    rows = {}
    for first, n in chains:
        tokens = list(range(first * 1000, first * 1000 + n))   # distinct first token per chain
        cache.insert(InsertParams(key=RadixKey(tokens, None),
                                  value=allocator.alloc(n).to(dtype=torch.int64)))
        rows[tokens[0]] = first
    for node in cache._collect_all_nodes():
        if node is cache.root_node:
            continue
        first = rows[node.key.token_ids[0]]
        node.component_data[ComponentType.FULL].host_value = torch.arange(
            first, first + len(node.key), dtype=torch.int64)
    return cache


def _spill(cache, pool):
    """One W3 spill of a rank; its release takes this rank's reference off every
    slot of the node (the arena's own count, shared with the peers)."""
    released = []

    def _release(self, node, comp, target=None, tracker=None):
        hv = node.component_data[ComponentType.FULL].host_value
        refs = getattr(pool.arena, "refs", None)
        for s in hv.tolist():
            if refs is not None:
                refs[s] -= 1
        released.append((int(hv.min()), int(hv.numel())))
        return 0, int(hv.numel())

    with mock.patch.object(UnifiedRadixCache, "_evict_component_and_detach_lru", _release):
        got = cache._w3_arena_spill(pool, SPAN)
    return got, released


def _three_ranks(per_rank_chains, held=(X, Y, Z), group_order=True):
    arena = _Arena(held)
    out = []
    with envs.FLLIPER_PDFLIP_W3_SPILL_GROUP_ORDER.override(group_order):
        for chains in per_rank_chains:
            cache = _tree(chains)
            out.append(_spill(cache, _RankPool(arena)))
    return arena, out


# PP trees that are NOT replicas: the same canonical pages, other node splits and
# other insertion orders (node ids)
REPLICA_BREAKING = [
    [(Z, SPAN), (Y, SPAN), (X, SPAN)],     # PP0: Z has the lowest id
    [(Y, SPAN), (Z, SPAN), (X, SPAN)],     # PP1: Y has the lowest id
    [(X, SPAN), (Y, SPAN), (Z, SPAN)],     # PP2: X has the lowest id
]


class TestW3SpillGroupOrder(CustomTestCase):
    def test_old_node_id_order_frees_nothing(self):
        """RED on f4afb10f28 (and with the switch off): each rank releases its
        lowest-id node = three different chunks, every slot keeps 2 references,
        the claim finds freed=0 (P.log:15554-15602)."""
        arena, out = _three_ranks(REPLICA_BREAKING, group_order=False)
        self.assertEqual([r[1][0][0] for r in out], [Z, Y, X], "three ranks, three different chunks")
        self.assertEqual(arena.free_slots(), [], "no slot reached zero references: ARENA-DROP freed=0")
        self.assertEqual(sorted(set(arena.refs[X:Z + SPAN])), [2])

    def test_group_equal_order_frees_a_slot_on_all_ranks(self):
        """GREEN: every rank releases the SAME chunk (the lowest slot), so after
        the third rank its slots are referenced by nobody -- the claim's clock
        takes them (``ARENA-DROP freed>0``)."""
        arena, out = _three_ranks(REPLICA_BREAKING)
        self.assertEqual([r[1] for r in out], [[(X, SPAN)]] * 3, "all three ranks let go of chunk X")
        self.assertEqual(arena.free_slots(), list(range(X, X + SPAN)),
                         "chunk X is referenced by NO rank any more")
        self.assertEqual([r[0] for r in out], [SPAN] * 3)

    def test_different_node_splits_still_converge(self):
        """The trees differ in the split, not only in the ids: PP1 holds the three
        chunks as ONE node (an old read's whole span). Old order: PP0 releases Z,
        PP1 everything, PP2 X -> Z and X keep a reference. New: PP0 releases X,
        PP1 (one node) everything, PP2 prefers the slots a peer already began to
        release (X: 1 reference left) -> X reaches zero."""
        ranks = [
            [(Z, SPAN), (Y, SPAN), (X, SPAN)],
            [(X, 3 * SPAN)],
            [(X, SPAN), (Y, SPAN), (Z, SPAN)],
        ]
        old, _ = _three_ranks(ranks, group_order=False)
        self.assertEqual(old.free_slots(), [], "old order: the 192-page node and the id order never meet")
        new, out = _three_ranks(ranks)
        self.assertEqual(new.free_slots(), list(range(X, X + SPAN)))
        self.assertEqual(out[1][1], [(X, 3 * SPAN)], "PP1 had no other node to give")

    def test_a_slot_only_this_rank_holds_goes_first(self):
        """A page whose last other reference is gone is the cheapest room there
        is: it is released before an older / lower-slot page the peers still
        hold (releasing it frees the slot at once)."""
        arena = _Arena(held=(X, Y, Z))
        for s in range(Z, Z + SPAN):
            arena.refs[s] = 1                      # only this rank holds Z
        with envs.FLLIPER_PDFLIP_W3_SPILL_GROUP_ORDER.override(True):
            got, rel = _spill(_tree([(X, SPAN), (Y, SPAN), (Z, SPAN)]), _RankPool(arena))
        self.assertEqual(rel, [(Z, SPAN)])
        self.assertEqual(arena.free_slots(), list(range(Z, Z + SPAN)), "freed at once, on the first release")

    def test_the_node_with_the_most_advanced_slot_goes_first(self):
        """The key is the node's LOWEST reference count over its slots: a node
        that holds one page a peer already let go (refs 1) goes before a node
        whose pages are all at 2 -- the one release that frees a slot at once
        beats an even spread (a max-based key would pick the all-2 node)."""
        arena = _Arena(held=(X, Y, Z))
        for s in range(X, X + SPAN):
            arena.refs[s] = 2                      # X: every page at 2
        arena.refs[Y] = 1                          # Y: one page only this rank holds, the rest at 3
        with envs.FLLIPER_PDFLIP_W3_SPILL_GROUP_ORDER.override(True):
            _got, rel = _spill(_tree([(X, SPAN), (Y, SPAN), (Z, SPAN)]), _RankPool(arena))
        self.assertEqual(rel, [(Y, SPAN)])
        self.assertEqual(arena.free_slots(), [Y], "the one page that only this rank held is free at once")

    def test_without_header_read_the_slot_is_still_the_order(self):
        """A pool whose arena cannot read its headers (no ``slot_refs``) still
        orders by the canonical slot, not by the node id."""
        arena = _Arena(held=(X, Y, Z))
        pool = _RankPool(arena)
        pool.arena = type("Hdrless", (), {"refs": arena.refs})()      # no slot_refs
        with envs.FLLIPER_PDFLIP_W3_SPILL_GROUP_ORDER.override(True):
            _got, rel = _spill(_tree([(Z, SPAN), (Y, SPAN), (X, SPAN)]), pool)
        self.assertEqual(rel, [(X, SPAN)])

    def test_unresolvable_rows_keep_the_node_id_order(self):
        """A pool whose rows name no arena slot (the existing fixtures) keeps
        the old order byte for byte."""
        class _Plain:
            staging_rows = 0

            def __init__(self):
                self.arena = object()

            def secure_rows_to_l3(self, hv):
                n = int(hv.numel())
                return {"pages": n, "on_disk": n, "written": 0, "lost": 0}

        _got, rel = _spill(_tree([(Z, SPAN), (Y, SPAN), (X, SPAN)]), _Plain())
        self.assertEqual(rel, [(Z, SPAN)], "lowest node id first, as before")

    def test_log_names_how_many_slots_became_free(self):
        arena = _Arena(held=(X, Y, Z))
        for s in range(X, X + SPAN):
            arena.refs[s] = 1
        with envs.FLLIPER_PDFLIP_W3_SPILL_GROUP_ORDER.override(True), \
                self.assertLogs("flliper.srt.mem_cache.unified_radix_cache", level="INFO") as cm:
            _spill(_tree([(X, SPAN), (Y, SPAN)]), _RankPool(arena))
        line = [m for m in cm.output if "W3-ARENA SPILL" in m][0]
        self.assertIn("ref_free=%d/%d" % (SPAN, SPAN), line)


class TestRealArenaThreeRanks(CustomTestCase):
    """The same scenario on the REAL objects: the C arena on a temp file, a
    bound ArenaMHAHostPool per rank, and one reference ledger per rank (another
    PROCESS each, as in the NF boot). ``release_tree_rows`` is the tree's own
    release (``HostPoolGroup.free`` -> ledger -> ``arena_ref_slots`` -1), so the
    header the next rank reads is the C header. The proof that a slot is free:
    the arena clock (``evict_candidates``, ``arena.c:597-599`` takes only
    refcount 0) hands it out."""

    SLOTS = 256
    CHUNK = 64

    def _ranks(self):
        import os
        import tempfile

        from test_arena_queue_refs_27b import S, _Rank

        fd, path = tempfile.mkstemp(prefix="w3-group-order-", suffix=".bin")
        os.close(fd)
        os.unlink(path)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        ranks = [_Rank(path, self.SLOTS, own_ledger=(i > 0)) for i in range(3)]
        for r in ranks:
            r.pool._backend.arena_secure_to_disk = (
                lambda arena, items: {"on_disk": len(items), "written": 0, "lost": 0})
        names = ["page-%d" % j for j in range(3 * self.CHUNK)]
        slots = [int(s) for s in ranks[0].publish(names).tolist()]
        for r in ranks:                           # every rank's tree references every page
            r.arena.ref_slots_np(slots, +1)
        return ranks, slots, S

    def _tree_on(self, rank, S, chunk_slots_in_insert_order):
        cache, allocator, _ = build_fixture(CacheConfig(page_size=1, components=(ComponentType.FULL,)))
        rows_of = {}
        for slots in chunk_slots_in_insert_order:
            tokens = list(range(slots[0] * 1000, slots[0] * 1000 + len(slots)))
            cache.insert(InsertParams(key=RadixKey(tokens, None),
                                      value=allocator.alloc(len(slots)).to(dtype=torch.int64)))
            rows_of[tokens[0]] = slots
        for node in cache._collect_all_nodes():
            if node is cache.root_node:
                continue
            sl = rows_of[node.key.token_ids[0]]
            node.component_data[ComponentType.FULL].host_value = torch.tensor(
                [S + s for s in sl], dtype=torch.int64)
        return cache

    def _spill_real(self, cache, rank):
        def _release(self_, node, comp, target=None, tracker=None):
            hv = node.component_data[ComponentType.FULL].host_value
            rank.pool.release_tree_rows(hv)
            return 0, int(hv.numel())

        with mock.patch.object(UnifiedRadixCache, "_evict_component_and_detach_lru", _release):
            return cache._w3_arena_spill(rank.pool, SPAN)

    def _scenario(self, group_order):
        from test_arena_queue_refs_27b import _hdr

        ranks, slots, S = self._ranks()
        X_, Y_, Z_ = (slots[i * self.CHUNK:(i + 1) * self.CHUNK] for i in range(3))
        inserts = [[Z_, Y_, X_], [Y_, Z_, X_], [X_, Y_, Z_]]          # three different node-id orders
        with envs.FLLIPER_PDFLIP_W3_SPILL_GROUP_ORDER.override(group_order):
            for rank, order in zip(ranks, inserts):
                self.assertEqual(self._spill_real(self._tree_on(rank, S, order), rank), self.CHUNK)
        _state, ref = _hdr(ranks[0].arena)
        zero = [s for s in slots if int(ref[s]) == 0]
        return ranks[0].arena, zero, X_

    def test_old_order_the_clock_finds_nothing(self):
        arena, zero, _X = self._scenario(group_order=False)
        self.assertEqual(zero, [], "three ranks, three chunks: every slot still has a reference")
        self.assertEqual(arena.evict_candidates(self.CHUNK), [], "ARENA-DROP need=64 freed=0")

    def test_group_equal_order_the_clock_takes_the_chunk(self):
        arena, zero, X_ = self._scenario(group_order=True)
        self.assertEqual(sorted(zero), sorted(X_), "chunk X: no rank references it any more")
        cands = arena.evict_candidates(self.CHUNK)
        self.assertEqual(sorted(c[0] for c in cands), sorted(X_), "the arena clock hands exactly chunk X out")


class TestHostOnlySpillOrder(CustomTestCase):
    """spill_host_only (the host-only H-leaf round) orders the same way."""

    def _host_only_tree(self):
        cache, allocator, _ = build_fixture(CacheConfig(page_size=1, components=(ComponentType.FULL,)))
        nodes = {}
        for first in (Z, X):                               # Z inserted first: the lower node id
            n = 16
            toks = list(range(first * 1000, first * 1000 + n))
            cache.insert(InsertParams(key=RadixKey(toks, None),
                                      value=allocator.alloc(n).to(dtype=torch.int64)))
        for node in cache._collect_all_nodes():
            if node is cache.root_node:
                continue
            first = int(node.key.token_ids[0]) // 1000
            node.component_data[ComponentType.FULL].host_value = torch.arange(first, first + 16, dtype=torch.int64)
            node.hash_value = ["h%d-%d" % (first, j) for j in range(16)]
            cache._evict_to_host(node)
            nodes[first] = node
        setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, True)
        return cache, nodes

    def _run(self, group_order):
        cache, nodes = self._host_only_tree()
        arena = _Arena(held=(X, Y, Z))
        pool = _RankPool(arena)
        with envs.FLLIPER_PDFLIP_W3_SPILL_GROUP_ORDER.override(group_order), \
                envs.FLLIPER_PDFLIP_ENABLE_W3_SPILL_HOST_LEAVES.override(True):
            got = cache._w3_arena_spill(pool, 16, claimer=None)
        present = {first for first, n in nodes.items() if n in cache._collect_all_nodes()}
        return got, present

    def test_group_equal_order_picks_the_lowest_slot_leaf(self):
        got, present = self._run(True)
        self.assertEqual(got, 32, "two host-only leaves of 16 pages: the round (W3_SPILL_MIN_PAGES = 64) takes both")
        self.assertEqual(present, set(), "both leaves left the tree with their L3 copy")

    def test_first_leaf_is_the_lowest_slot_not_the_lowest_id(self):
        cache, nodes = self._host_only_tree()
        arena = _Arena(held=(X, Y, Z))
        pool = _RankPool(arena)
        order = []
        real = pool.secure_rows_to_l3

        def spy(hv):
            order.append(int(hv.min()))
            return real(hv)

        pool.secure_rows_to_l3 = spy
        with envs.FLLIPER_PDFLIP_W3_SPILL_GROUP_ORDER.override(True), \
                envs.FLLIPER_PDFLIP_ENABLE_W3_SPILL_HOST_LEAVES.override(True):
            from flliper.srt.pdflip import host_only_spill as hos

            hos.spill_host_only(cache, pool, 16, 1, set())
        self.assertEqual(order[0], X, "the leaf on the lowest canonical slot first (old: the lowest node id = Z)")
        with envs.FLLIPER_PDFLIP_W3_SPILL_GROUP_ORDER.override(False), \
                envs.FLLIPER_PDFLIP_ENABLE_W3_SPILL_HOST_LEAVES.override(True):
            cache2, _ = self._host_only_tree()
            pool2 = _RankPool(_Arena(held=(X, Y, Z)))
            order2 = []
            real2 = pool2.secure_rows_to_l3
            pool2.secure_rows_to_l3 = lambda hv: (order2.append(int(hv.min())), real2(hv))[1]
            hos.spill_host_only(cache2, pool2, 16, 1, set())
        self.assertEqual(order2[0], Z, "switch off: the old node-id order")


if __name__ == "__main__":
    unittest.main()
