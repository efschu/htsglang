"""#681: what ``full_evictable_size_`` promises, the frontier must be able to pay.

CONTEXT, AND A CORRECTION. Commit 7752dc88cd diagnosed the 01:46:10 crash as
"tokens behind a locked chain are counted and unreachable", on the model that
``evict`` walks the leaf frontier while ``full_evictable_size_`` counts unlocked
tokens ANYWHERE in the tree. The first half is right; the conclusion is not, and
``TestLockRefsAreAncestorClosed`` below is why:

    ``inc_lock_ref`` / ``dec_lock_ref`` walk from a node to the ROOT, and
    ``_split_node`` copies ``full_lock_ref`` onto the new upper half. So
    ``full_lock_ref(parent) >= full_lock_ref(child)`` holds for every edge, at
    every moment. An UNLOCKED node therefore cannot have a LOCKED descendant --
    its whole subtree is unlocked -- so it always has an unlocked leaf below it
    and the peel always reaches it.

The locked-chain term is structurally ZERO. Lowering the counter by it would
have moved the admission budget by nothing.

THE GAP THAT IS REAL is the other end of the frontier: which unlocked leaves the
actuator can consume. ``evict_full`` selects with ``get_leaf_lru_no_lock`` --
unlocked and childless -- but ``_evict_leaf_node`` requires ``mamba_value is not
None``. An UNLOCKED TOMBSTONE LEAF satisfies the selector and violates the
consumer, and the code creates that state itself:

    ``_iteratively_delete_tombstone_leaf`` breaks on
    ``node.parent.full_lock_ref > 0``. A tombstone that loses its last child
    while a request holds it survives as a LOCKED tombstone leaf; when that
    request finishes, nothing revisits it. It is now unlocked, childless,
    counted in ``full_evictable_size_``, in the full LRU list -- and the next
    ``evict_full`` selects it and dies on the assert.

The 01:46 tree had exactly one: node 5937 (fr=0, mv=None, childless, fll=True),
alongside a single payable leaf 5959. Replaying that dumped tree through the
deployed code selects 5937 and raises. Reconstruction of the dump is checked
against two numbers the process printed independently -- 140683 total tokens and
65766 evictable -- and both match to the token.

THE REPAIR IS ON THE ACTUATOR, NOT THE COUNTER. An unlocked tombstone leaf is
precisely what ``_iteratively_delete_tombstone_leaf`` already deletes one step
earlier; the only reason this one survived is that it was locked at that
instant. Teaching the frontier to delete it makes the counter's promise TRUE
rather than making the counter smaller, which is the stronger closure of "a
counter must never promise what the actuator cannot pay".

GROUP-UNIFORMITY: the repair is a pure function of replicated tree state, so
every rank takes the same branch on the same iteration. No new collective, and
no new channel beside the existing ``uniform_avail_floor``.
"""

import unittest
from array import array
from typing import List, Tuple

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, TreeNode
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)

NUM_LAYERS = 8
GLOBAL_INTERVAL = 4
KV_POOL_SIZE = 4096
MAX_CONTEXT_LEN = 1024
CHUNK = 64


# --------------------------------------------------------------------------
# CPU pools (same shape as test_mamba_lock_ref_pairing_581)
# --------------------------------------------------------------------------


def _build_pools(mamba_size: int = 8, max_num_reqs: int = 8):
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    set_global_server_args_for_scheduler(server_args)

    device = "cpu"
    full_attention_layer_ids = [
        i for i in range(GLOBAL_INTERVAL - 1, NUM_LAYERS, GLOBAL_INTERVAL)
    ]
    mamba_layers = [i for i in range(NUM_LAYERS) if i not in full_attention_layer_ids]
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=512,
            n_groups=4,
            num_heads=8,
            head_dim=64,
            state_size=32,
            conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

    req_to_token_pool = HybridReqToTokenPool(
        size=max_num_reqs,
        mamba_size=mamba_size,
        mamba_spec_state_size=max_num_reqs,
        max_context_len=MAX_CONTEXT_LEN,
        device=device,
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=mamba_layers,
        enable_mamba_extra_buffer=False,
        enable_linear_replayssm=False,
    )
    kv_pool = HybridLinearKVPool(
        size=KV_POOL_SIZE,
        dtype=torch.bfloat16,
        page_size=1,
        head_num=2,
        head_dim=64,
        full_attention_layer_ids=full_attention_layer_ids,
        device=device,
        enable_memory_saver=False,
        mamba_pool=req_to_token_pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=KV_POOL_SIZE,
        dtype=torch.bfloat16,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )
    return req_to_token_pool, allocator


def _cache_init_params(req_to_token_pool, allocator) -> CacheInitParams:
    return CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        disable=False,
        enable_kv_cache_events=False,
        enable_mamba_extra_buffer=False,
    )


def _build_tree(mamba_size: int = 8):
    pool, allocator = _build_pools(mamba_size)
    return MambaRadixCache(_cache_init_params(pool, allocator)), allocator, pool


def _key(token_ids) -> RadixKey:
    return RadixKey(array("q", token_ids))


def _insert(tree, allocator, pool, token_ids) -> TreeNode:
    slot = pool.mamba_allocator.alloc(1)
    assert slot is not None, "test setup: mamba pool exhausted"
    tree.insert(
        InsertParams(
            key=_key(token_ids),
            value=allocator.alloc(len(token_ids)),
            mamba_value=slot,
        )
    )
    return tree.match_prefix(MatchPrefixParams(key=_key(token_ids))).last_device_node


# --------------------------------------------------------------------------
# Shared tree walkers
# --------------------------------------------------------------------------


def _walk(root: TreeNode) -> List[TreeNode]:
    out, stack = [], [root]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(node.children.values())
    return out


def _lock_attr(tree) -> str:
    return "full_lock_ref" if isinstance(tree, MambaRadixCache) else "lock_ref"


def _closure_violations(tree) -> List[Tuple[int, int]]:
    """Edges where the child is locked harder than its parent.

    Empty on every state the public API can produce; non-empty is the shape
    #681 assumed -- an unlocked node with a locked descendant, whose tokens the
    peel could never reach.
    """
    attr = _lock_attr(tree)
    bad = []
    for node in _walk(tree.root_node):
        for child in node.children.values():
            if getattr(node, attr) < getattr(child, attr):
                bad.append((node.id, child.id))
    return bad


def _unreachable_behind_locks(tree) -> int:
    """Tokens in unlocked nodes that have a LOCKED descendant.

    This is the quantity repair B was specified to subtract from the admission
    budget. It is structurally zero -- see ``TestLockRefsAreAncestorClosed``.
    """
    attr = _lock_attr(tree)
    stranded = 0
    for node in _walk(tree.root_node):
        if node is tree.root_node or getattr(node, attr) != 0:
            continue
        stack = list(node.children.values())
        while stack:
            cur = stack.pop()
            if getattr(cur, attr) > 0:
                stranded += len(node.key)
                break
            stack.extend(cur.children.values())
    return stranded


def _stage_three_nodes(tree, allocator, pool):
    """P -> CHILD on one branch, OTHER on a second, with a pinned LRU order.

    OTHER is pushed to the most-recently-used end of BOTH lists so the full
    peel is forced onto CHILD; without that the fixture's outcome would depend
    on insertion order rather than on the property under test.
    """
    p = _insert(tree, allocator, pool, list(range(1000, 1000 + CHUNK)))
    child = _insert(tree, allocator, pool, list(range(1000, 1000 + 2 * CHUNK)))
    other = _insert(tree, allocator, pool, list(range(5000, 5000 + CHUNK)))
    tree.full_lru_list.reset_node_mru(other)
    tree.mamba_lru_list.reset_node_mru(other)
    return p, child, other


def _stage_unlocked_tombstone_leaf(tree, allocator, pool):
    """Drive the tree, through public transitions only, into the 01:46 state.

    Returns the node that ends up an unlocked, childless MAMBA TOMBSTONE that
    is still counted in ``full_evictable_size_``.
    """
    p, child, other = _stage_three_nodes(tree, allocator, pool)

    # Make P the least recently used mamba entry so the mamba pressure lands on
    # it. It is an INTERNAL node, so evict_mamba tombstones it in place.
    for node in (child, other):
        tree.mamba_lru_list.reset_node_mru(node)
    tree.evict(EvictParams(num_tokens=0, mamba_num=1))
    assert p.mamba_value is None, "test setup: P was not tombstoned"
    assert len(p.children) == 1, "test setup: P is not internal"

    # A request whose last_node IS the tombstone holds it. `inc_lock_ref` takes
    # only the full ref -- the mamba ref is skipped, P has no mamba value.
    tree.inc_lock_ref(p)

    # Its child is evicted. The tombstone cleanup breaks on P's lock, so P
    # survives as a LOCKED tombstone leaf.
    tree.evict(EvictParams(num_tokens=CHUNK))
    assert len(p.children) == 0, "test setup: P did not become a leaf"
    assert p.mamba_value is None, "test setup: P is no longer a tombstone"

    # The request finishes. Nothing revisits P.
    tree.dec_lock_ref(p)
    assert p.full_lock_ref == 0
    return p


class TestLockRefsAreAncestorClosed(unittest.TestCase):
    """The premise #681 rested on cannot occur, so its remedy would be a no-op.

    Both radix classes lock a PATH to the root, so locking is ancestor-closed
    and 'tokens behind a locked chain' is the empty set.
    """

    def test_mamba_tree_has_no_unlocked_node_with_a_locked_descendant(self):
        tree, allocator, pool = _build_tree()
        deep = None
        for i in range(1, 5):
            deep = _insert(tree, allocator, pool, list(range(1000, 1000 + i * CHUNK)))
        _insert(tree, allocator, pool, list(range(5000, 5000 + CHUNK)))
        shallow = tree.match_prefix(
            MatchPrefixParams(key=_key(list(range(1000, 1000 + CHUNK))))
        ).last_device_node

        for locked in (deep, shallow):
            tree.inc_lock_ref(locked)
            self.assertEqual([], _closure_violations(tree))
            self.assertEqual(
                0,
                _unreachable_behind_locks(tree),
                "repair B's quantity must be structurally zero",
            )
            tree.dec_lock_ref(locked)

    def test_a_split_under_a_lock_keeps_the_closure(self):
        tree, allocator, pool = _build_tree()
        node = _insert(tree, allocator, pool, list(range(1000, 1000 + 2 * CHUNK)))
        tree.inc_lock_ref(node)
        # A shorter sequence sharing the prefix splits the locked node in two.
        _insert(tree, allocator, pool, list(range(1000, 1000 + CHUNK)) + [77])
        self.assertEqual([], _closure_violations(tree))
        self.assertEqual(0, _unreachable_behind_locks(tree))

    def test_base_radix_cache_is_ancestor_closed_too(self):
        tree = RadixCache.create_simulated()
        tree.insert(InsertParams(key=_key([1, 2, 3, 4])))
        tree.insert(InsertParams(key=_key([1, 2, 5, 6])))
        node = tree.match_prefix(
            MatchPrefixParams(key=_key([1, 2, 3, 4]))
        ).last_device_node
        tree.inc_lock_ref(node)
        self.assertEqual([], _closure_violations(tree))
        self.assertEqual(0, _unreachable_behind_locks(tree))

    def test_the_detector_can_fail(self):
        """Mutation proof: hand-build the shape #681 assumed and see it caught.

        Without this, the two assertions above would be satisfied by a detector
        that always returns empty.
        """
        tree, allocator, pool = _build_tree()
        parent = _insert(tree, allocator, pool, list(range(1000, 1000 + CHUNK)))
        child = _insert(tree, allocator, pool, list(range(1000, 1000 + 2 * CHUNK)))
        # Not reachable through the API: lock the child WITHOUT its ancestors.
        child.full_lock_ref = 1
        self.assertEqual([(parent.id, child.id)], _closure_violations(tree))
        self.assertEqual(len(parent.key), _unreachable_behind_locks(tree))
        child.full_lock_ref = 0


class TestUnlockedTombstoneLeafIsPayable(unittest.TestCase):
    """The frontier must be able to consume every unlocked leaf it selects."""

    def test_the_state_the_crash_tree_was_in_is_reachable(self):
        tree, allocator, pool = _build_tree()
        p = _stage_unlocked_tombstone_leaf(tree, allocator, pool)
        # Exactly node 5937's signature in the 01:46 dump.
        self.assertEqual(0, p.full_lock_ref)
        self.assertEqual(0, len(p.children))
        self.assertIsNone(p.mamba_value)
        self.assertTrue(tree.full_lru_list.in_list(p))
        self.assertIs(p, tree.full_lru_list.get_leaf_lru_no_lock())

    def test_the_frontier_pays_the_tombstone_leaf_instead_of_asserting(self):
        """RED before the repair: ``AssertionError: leaf node mamba value ...``.

        The selector offers the node, so the consumer must take it. Raising
        here kills every rank at once, which is how a recoverable cache state
        became a group-wide crash.
        """
        tree, allocator, pool = _build_tree()
        p = _stage_unlocked_tombstone_leaf(tree, allocator, pool)
        before = allocator.available_size()
        freed = tree.evict(EvictParams(num_tokens=len(p.key)))
        self.assertEqual(len(p.key), freed.num_tokens_evicted)
        self.assertEqual(before + len(p.key), allocator.available_size())

    def test_the_counter_never_over_promises(self):
        """The #681 invariant, stated as the property it actually is.

        Asking for everything ``full_evictable_size_`` claims must deliver
        everything it claims -- no matter which of the reachable states the
        tree is in.
        """
        tree, allocator, pool = _build_tree()
        _stage_unlocked_tombstone_leaf(tree, allocator, pool)
        promised = tree.full_evictable_size()
        self.assertGreater(promised, 0, "test setup: nothing to evict")
        freed = tree.evict(EvictParams(num_tokens=promised))
        self.assertEqual(
            promised,
            freed.num_tokens_evicted,
            "eviction under-delivered against its own counter",
        )
        self.assertEqual(0, tree.full_evictable_size())

    def test_a_locked_tombstone_leaf_is_still_refused(self):
        """The repair must not reach past a lock -- that is repair A, unbuilt.

        Mutation proof for the branch: with the lock still held the same node
        must NOT be freed, so the new path is gated on the lock and not on the
        tombstone alone.
        """
        tree, allocator, pool = _build_tree()
        p, child, other = _stage_three_nodes(tree, allocator, pool)
        for node in (child, other):
            tree.mamba_lru_list.reset_node_mru(node)
        tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        tree.inc_lock_ref(p)
        # Drain every payable leaf: first P's child, then the other branch.
        tree.evict(EvictParams(num_tokens=CHUNK))
        tree.evict(EvictParams(num_tokens=CHUNK))

        # P is now a LOCKED tombstone leaf and the only node left. It is not
        # counted as evictable, and the frontier must not take it.
        self.assertEqual(1, p.full_lock_ref)
        self.assertIsNone(p.mamba_value)
        self.assertEqual(0, len(p.children))
        self.assertEqual(0, tree.full_evictable_size())
        before = allocator.available_size()
        freed = tree.evict(EvictParams(num_tokens=len(p.key)))
        self.assertEqual(0, freed.num_tokens_evicted)
        self.assertEqual(before, allocator.available_size())
        self.assertIn(
            p,
            list(tree.root_node.children.values()),
            "the locked tombstone must stay in the tree",
        )

    def test_the_ordinary_path_is_untouched(self):
        """No tombstone leaf anywhere: the peel behaves exactly as before."""
        tree, allocator, pool = _build_tree()
        for i in range(1, 5):
            _insert(tree, allocator, pool, list(range(1000, 1000 + i * CHUNK)))
        promised = tree.full_evictable_size()
        before = allocator.available_size()
        freed = tree.evict(EvictParams(num_tokens=CHUNK))
        self.assertEqual(CHUNK, freed.num_tokens_evicted)
        self.assertEqual(before + CHUNK, allocator.available_size())
        self.assertEqual(promised - CHUNK, tree.full_evictable_size())


if __name__ == "__main__":
    unittest.main()
