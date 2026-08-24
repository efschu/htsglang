"""#841 -- the contiguous-backup law of the host tier, and the two crash
shapes that follow from breaking it.

WHAT HAPPENED (window 5, integ/round5 @ 54e69ca2af, reproduced 2 of 2 boots).
Both boots died within seconds of going IDLE, never under load, on all three
ranks at once::

    scheduler.py:2567    event_loop_normal -> on_idle()
    scheduler.py:9451    on_idle -> invariant_checker._check_tree_cache()
    invariant_checker.py:534         -> tree_cache.sanity_check()
    unified_radix_cache.py:4009      -> raise AssertionError

with two different violation sets in the same ledger -- the HOST tier::

    boot 5b   node 144 backed up but parent 11 not backed up
    boot 5a   H-leaf extra: [18, 19]
              2 stale nodes in host_leaves: [18, 19]
              mamba host LRU: +S3=set(), +lru={18}

They are not two defects. They are one state and its consequence, and this
file proves both against the real ledger, with no GPU.

THE LAW. A node may carry a host copy only if its parent does -- backed-up
nodes form a contiguous prefix from the root. ``write_backup`` has always
upheld it (``unified_radix_cache.py:1943-1948``: back the parent up first,
refuse when that fails). The sanity check asserts it on every idle tick,
suppressed only when the server's write policy is ``write_back``.

THE SECOND WRITER. ``check_prefetch_progress`` completes a storage prefetch by
calling ``_insert_helper_host``, which walks down from ``last_host_node`` --
allowed to be the ROOT by ``scheduler.py:4420`` -- through whatever existing
children match the fetched key, DEVICE-ONLY ones included, and attaches the
fetched tail wherever the walk stops. No parent gate anywhere on that path.
Window 5's ``matched=45`` is exactly root -> node 12 (1 token) -> node 11
(44 tokens), both device-only: that is where the backed child was hung.

WHY THE ORPHAN FOLLOWS. ``_is_device_leaf`` qualifies a node whose children
hold host data but no device data, and ``_evict_device_leaf``'s write-through
branch DELETES an un-backed device leaf. So the illegal parent is deletable
while it still has children: ``_remove_leaf_from_parent`` pops the edge and
the backed subtree below it stays in ``evictable_host_leaves`` and in the aux
host LRUs while no longer being reachable from the root. Boot 5a caught the
tree one eviction after boot 5b did.

ATTRIBUTION, stated as evidence rather than as a story. The specimen logs say
``[#703 flip-writeback] ... staged=0`` on every one of its 54 (5a) / 153 (5b)
invocations and log no staging exception, so the flip lane committed nothing
to the ledger on these boots. The prefetch lane did: the only rank with an
extra prefetch on 5b (PP1, 23:11:57, ``loaded=16339``) is the only rank with
an extra violation (node 145 AND node 14, against node 144 alone on PP0/PP2).

The flip lane is nonetheless fixed here too. It reached ``write_backup`` with
``write_back=True``, which disarms that function's own parent gate, on a
server running ``write_through`` -- where the sanity check stays armed. It is
the same law, broken the same way, one refusal away from firing.

THE CHECKER IS CORRECT AND STAYS ARMED. Nothing here relaxes it.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import (
    BASE_COMPONENT_TYPE,
    UnifiedRadixCache,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# ~10s: a handful of tiny CPU-only radix trees; no pools beyond a 1024-slot
# bfloat16 MHA pool, no accelerator, no subprocess.
register_cpu_ci(est_time=10, suite="base-a-test-cpu")


#: The window-5 prefix geometry: root -> (1 token) -> (44 tokens) = 45, the
#: `matched=45` the specimen logs report on every rank of boot 5b.
MATCHED_PREFIX_LEN = 45


def build_cache(size: int = 1024) -> UnifiedRadixCache:
    """A real UnifiedRadixCache on the CPU.

    ``get_device()`` raises without an accelerator, so every pool is pinned to
    "cpu" explicitly. ``cache_controller`` stays None, which is what arms the
    parent-backed invariant in ``sanity_check`` -- it is suppressed only for
    the ``write_back`` write policy, and the window-5 boots ran
    ``--hicache-write-policy write_through``.
    """
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    req_to_token_pool = ReqToTokenPool(
        size=8, max_context_len=size, device="cpu", enable_memory_saver=False
    )
    kv_pool = MHATokenToKVPool(
        size=size,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=8,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=size,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    return UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
            tree_components=(ComponentType.FULL,),
        )
    )


def insert_device_prefix(cache: UnifiedRadixCache, tokens: list) -> None:
    """A perfectly ordinary device-tier insert: no host copy anywhere."""
    cache.insert(
        InsertParams(
            key=RadixKey(list(tokens)),
            value=torch.arange(len(tokens), dtype=torch.int64),
        )
    )


def prefetch_host_insert(cache: UnifiedRadixCache, fetched: list, base: int = 1000):
    """Exactly what ``check_prefetch_progress`` does at the completion of a
    storage prefetch (``unified_radix_cache.py``): walk down from the root and
    attach the fetched tail as a host-only node."""
    return cache._insert_helper_host(
        cache.root_node,
        RadixKey(list(fetched)),
        torch.arange(base, base + len(fetched), dtype=torch.int64),
        [f"h{i}" for i in range(len(fetched))],
    )


def stale_host_leaves(cache: UnifiedRadixCache) -> set:
    """Nodes the host ledger still tracks that the tree can no longer reach.

    This is the quantity behind "H-leaf extra" and "stale nodes in
    host_leaves" -- computed here directly rather than parsed out of the
    assertion text, so the test measures the ledger and not the message.
    """
    reachable = set(cache._collect_all_nodes())
    return {node.id for node in cache.evictable_host_leaves - reachable}


class TestPrefetchHostInsertUpholdsTheLaw(CustomTestCase):
    """The window-5 state, form (b): child backed, parent not."""

    def test_host_insert_under_device_only_parent_is_declined(self):
        cache = build_cache()
        insert_device_prefix(cache, range(1, MATCHED_PREFIX_LEN + 1))
        cache.sanity_check()

        result = prefetch_host_insert(cache, range(1, MATCHED_PREFIX_LEN + 11))

        # The walk consumed the whole device-only prefix, reproducing the
        # specimen's `matched=45` -- so this IS the production landing site,
        # not a contrived one.
        self.assertEqual(result.prefix_len, MATCHED_PREFIX_LEN)
        # ... and then refused to hang a backed child off an un-backed parent.
        self.assertIsNone(result.inserted_host_node)
        self.assertTrue(result.host_span_unclaimed)
        self.assertEqual(cache._host_insert_refused_unbacked_parent, 1)

        # The idle-path check that killed both boots.
        cache.sanity_check()

    def test_the_declined_state_is_the_one_that_crashed(self):
        """Falsifier for the fix: re-create the illegal state by hand and
        confirm the checker still calls it out.

        Without this, a fix that merely stopped the tree from reaching the
        state would be indistinguishable from one that quietly disarmed the
        invariant. The checker is correct and must stay armed.
        """
        cache = build_cache()
        insert_device_prefix(cache, range(1, MATCHED_PREFIX_LEN + 1))

        # Bypass the gate the way the pre-#841 code path did.
        parent = cache.root_node.children[RadixKey([1]).child_key(cache.page_size)]
        self.assertFalse(parent.backuped)
        child = cache._add_new_node(
            parent,
            RadixKey(list(range(MATCHED_PREFIX_LEN + 1, MATCHED_PREFIX_LEN + 11))),
            torch.arange(10, dtype=torch.int64),
        )
        child.component_data[BASE_COMPONENT_TYPE].value = None
        child.component_data[BASE_COMPONENT_TYPE].host_value = torch.arange(
            10, dtype=torch.int64
        )
        cache._update_evictable_leaf_sets(child)

        with self.assertRaises(AssertionError) as caught:
            cache.sanity_check()
        self.assertIn("backed up but parent", str(caught.exception))

    def test_host_insert_under_the_root_still_lands(self):
        """The law refuses a gap, not the feature.

        A prefetch into an empty tree attaches under the root, which is
        exempt by definition -- if this went red the fix would have bought
        correctness by disabling the storage tier.
        """
        cache = build_cache()
        result = prefetch_host_insert(cache, range(1, 11))

        self.assertIsNotNone(result.inserted_host_node)
        self.assertFalse(result.host_span_unclaimed)
        self.assertTrue(result.inserted_host_node.backuped)
        self.assertIn(result.inserted_host_node, cache.evictable_host_leaves)
        cache.sanity_check()

    def test_write_back_policy_still_adopts_the_tail(self):
        """The gate is armed exactly where the invariant is.

        Condition taken from upstream PR 31902's second commit (f198ebf97f).
        Under `write_back` the checker suppresses the parent invariant
        (`sanity_check` :3899-3903) AND the orphan hazard is absent --
        `_evict_device_leaf` writes an un-backed leaf back instead of deleting
        it, so no edge is ever popped above a backed child. Declining there
        would cost retention and buy nothing.
        """
        cache = build_cache()
        insert_device_prefix(cache, range(1, MATCHED_PREFIX_LEN + 1))

        class _WriteBackController:
            write_policy = "write_back"

        cache.cache_controller = _WriteBackController()
        try:
            result = prefetch_host_insert(cache, range(1, MATCHED_PREFIX_LEN + 11))
            self.assertIsNotNone(result.inserted_host_node)
            self.assertFalse(result.host_span_unclaimed)
            self.assertEqual(cache._host_insert_refused_unbacked_parent, 0)
            # The child is backed and its parent is not -- legal under this
            # policy, and the checker agrees.
            self.assertTrue(result.inserted_host_node.backuped)
            self.assertFalse(result.inserted_host_node.parent.backuped)
            cache.sanity_check()
        finally:
            cache.cache_controller = None

    def test_host_insert_extends_a_backed_chain(self):
        """And a second prefetch deepening an already-backed chain lands too,
        because every node on the path to it carries a host copy."""
        cache = build_cache()
        first = prefetch_host_insert(cache, range(1, 11))
        self.assertIsNotNone(first.inserted_host_node)

        second = prefetch_host_insert(cache, range(1, 21), base=2000)
        self.assertIsNotNone(second.inserted_host_node)
        self.assertEqual(second.prefix_len, 10)
        self.assertFalse(second.host_span_unclaimed)
        self.assertIs(second.inserted_host_node.parent, first.inserted_host_node)
        cache.sanity_check()


class TestOrphanedHostSubtree(CustomTestCase):
    """The window-5 consequence, form (a): the backed subtree outlives the
    edge that reached it."""

    def test_device_eviction_over_a_prefetched_node_leaves_no_stale_leaves(self):
        """The full boot-5a sequence, end to end.

        device insert -> storage prefetch extends it -> device insert over the
        prefetched span -> device eviction. Pre-#841 the last step deleted the
        un-backed parent out from under a backed child and produced
        "H-leaf extra" + "stale nodes in host_leaves".
        """
        cache = build_cache()
        insert_device_prefix(cache, range(1, MATCHED_PREFIX_LEN + 1))
        prefetch_host_insert(cache, range(1, MATCHED_PREFIX_LEN + 11))
        cache.sanity_check()

        insert_device_prefix(cache, range(1, MATCHED_PREFIX_LEN + 11))
        cache.sanity_check()
        self.assertEqual(stale_host_leaves(cache), set())

        cache.evict(EvictParams(num_tokens=60))
        self.assertEqual(stale_host_leaves(cache), set())
        cache.sanity_check()

    def test_delete_guard_refuses_to_orphan_a_backed_subtree(self):
        """Falsifier for the second guard, exercised directly.

        Under the law this state is unreachable, so the guard would otherwise
        be code that never runs and never fails -- untested by construction.
        Build the state by hand and prove the delete branch refuses instead of
        popping the edge.
        """
        cache = build_cache()
        insert_device_prefix(cache, range(1, MATCHED_PREFIX_LEN + 1))
        parent = cache.root_node.children[RadixKey([1]).child_key(cache.page_size)]
        child = cache._add_new_node(
            parent,
            RadixKey(list(range(MATCHED_PREFIX_LEN + 1, MATCHED_PREFIX_LEN + 11))),
            torch.arange(10, dtype=torch.int64),
        )
        # Child: host only. Parent: device only. The illegal shape.
        child.component_data[BASE_COMPONENT_TYPE].value = None
        child.component_data[BASE_COMPONENT_TYPE].host_value = torch.arange(
            10, dtype=torch.int64
        )
        cache._update_evictable_leaf_sets(child)
        cache._update_evictable_leaf_sets(parent)

        # The parent now qualifies as a device leaf despite having a child,
        # which is the precondition the delete branch was missing.
        self.assertFalse(parent.backuped)
        self.assertTrue(cache._is_device_leaf(parent))

        tracker = {ct: 0 for ct in cache.tree_components}
        cache._evict_device_leaf(parent, tracker)

        # Refused: the edge survives, so the backed child is still reachable.
        self.assertIn(child, set(cache._collect_all_nodes()))
        self.assertEqual(stale_host_leaves(cache), set())


class TestFlipWritebackUpholdsTheLaw(CustomTestCase):
    """The third writer: ``--phase-flip-writeback``.

    It staged nothing on either window-5 boot (``staged=0``, 54 and 153
    invocations, no staging exception), so it is not this crash's producer.
    It breaks the same law by the same means, though: ``write_back=True``
    disarms ``write_backup``'s parent gate while the server runs
    ``write_through``, where the checker stays armed.
    """

    def _tree(self, backuped_flags):
        """A root -> A -> B chain with the given backup states."""

        class Node:
            def __init__(self, node_id, parent, hash_value):
                self.id = node_id
                self.parent = parent
                self.hash_value = hash_value
                self.children = {}
                self.backuped = False

        root = Node(0, None, None)
        node_a = Node(1, root, ["ha"])
        node_b = Node(2, node_a, ["hb"])
        root.children["a"] = node_a
        node_a.children["b"] = node_b
        node_a.backuped, node_b.backuped = backuped_flags
        return root, node_a, node_b

    def _run(self, root, refuse_ids):
        from sglang.srt.mem_cache import hicache_flip_writeback as fw

        staged = []

        class FakeTree:
            root_node = root
            ongoing_backup = None

            def write_backup(self, node, write_back=False):
                if node.id in refuse_ids:
                    # The three real refusals: the mamba pin budget, the
                    # rank-uniform host floor, the staging ring. All return 0
                    # without raising, which is why the loop cannot infer a
                    # parent's state from its position in the walk.
                    return 0
                node.backuped = True
                staged.append(node.id)
                return 1

            def writing_check(self, write_back=False):
                return None

        tree = FakeTree()
        # The canonical-store gate is about page-key geometry across the flip,
        # not about the ledger; satisfy it so the staging loop is what runs.
        with mock.patch.object(fw, "require_canonical_store", lambda _t: None):
            report = fw.flip_writeback(tree, deadline_s=0.0)
        return report, staged

    def test_child_is_not_staged_over_an_unbacked_parent(self):
        root, node_a, node_b = self._tree((False, False))
        report, staged = self._run(root, refuse_ids={1})

        # A was refused (as the host floor refuses on an uneven rig), so B
        # must not be staged over the gap A leaves behind.
        self.assertEqual(staged, [])
        self.assertFalse(node_a.backuped)
        self.assertFalse(node_b.backuped)
        self.assertEqual(report.staged, 0)

    def test_a_contiguous_chain_still_stages_in_full(self):
        """The law refuses a gap, not the feature: with nothing refusing, the
        whole parent-first chain still reaches the host tier."""
        root, node_a, node_b = self._tree((False, False))
        report, staged = self._run(root, refuse_ids=set())

        self.assertEqual(staged, [1, 2])
        self.assertEqual(report.staged, 2)


if __name__ == "__main__":
    unittest.main()
