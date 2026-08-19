from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the hybrid (full and Mamba) KV cache.
"""

import heapq
from array import array
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from numpy import float64

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
    requests_forced_host_write_through,
)
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.utils import split_node_hash_value
from sglang.srt.runtime_context import get_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

import logging
import os

from sglang.srt.environ import envs
from sglang.srt.mem_cache.mamba_ckpt_utils import (
    floor_to_interval,
    is_on_interval,
    is_resume_candidate,
    protect_deepest_anchors,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def _skipped_ids(params: Optional[DecLockRefParams], component: ComponentType):
    """Node ids whose `component` lock the paired acquire did NOT take."""
    if params is None:
        return ()
    return params.skip_lock_node_ids.get(component, ())


def _radix_debug_dump() -> bool:
    """Per-node radix detail on the error path, off by default (#695).

    Read from the environment on each call rather than cached at import: this
    is an error path, it runs at most once per failure, and an operator who
    sets the variable while chasing a live incident should not have to restart
    the instance to see the detail.
    """
    return os.environ.get("SGLANG_RADIX_DEBUG_DUMP", "") not in ("", "0", "false", "False")


class TreeNode:

    counter = 0
    last_access_time_counter_float = float64(1.0)

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.mamba_value: Optional[torch.Tensor] = None
        self.mamba_host_value: Optional[torch.Tensor] = None
        # invariant: for any node, if mamba_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, mamba_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= mamba_lock_ref.
        # for full_lock, once it is locked, its parent must be locked as well
        # for mamba_lock, it only need lock node itself
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        self.last_access_time = get_last_access_time()

        self.hit_count = 0
        self.host_ref_counter = 0
        self.host_mamba_ref_counter = 0
        # store the host indices of KV cache
        self.host_value = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        self.prev = None
        self.next = None
        self.mamba_prev = None
        self.mamba_next = None
        self.host_mamba_prev = None
        self.host_mamba_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def mamba_evicted(self):
        return self.mamba_value is None

    @property
    def backuped(self):
        return self.host_value is not None

    @property
    def mamba_backuped(self):
        return self.mamba_host_value is not None

    def protect_host(self):
        """Protect the host KV value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host KV value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def protect_host_mamba(self):
        """Protect the host mamba value from eviction."""
        self.host_mamba_ref_counter += 1

    def release_host_mamba(self):
        """Release the host mamba value, allowing it to be evicted."""
        if self.host_mamba_ref_counter > 0:
            self.host_mamba_ref_counter -= 1
        else:
            raise RuntimeError("Host mamba reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []
        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


class LRUList:
    def __init__(self, mamba: bool = False):
        self.mamba = mamba
        if self.mamba:
            self.prv = "mamba_prev"
            self.nxt = "mamba_next"
            self.lock_ref = "mamba_lock_ref"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"
        # Initialize dummy head and tail nodes
        self.head = TreeNode()  # Most recently used side
        self.tail = TreeNode()  # Least recently used side
        setattr(self.head, self.nxt, self.tail)  # self.head.next = self.tail
        setattr(self.tail, self.prv, self.head)  # self.tail.prev = self.head
        self.cache = {}

    def _add_node(self, node):
        """Helper to add node right after head (most recently used)"""
        self._add_node_after(self.head, node)

    def _add_node_after(self, old_node, new_node):
        """Helper to add node right after old_node"""
        setattr(new_node, self.prv, old_node)  # new_node.prev = old_node
        setattr(
            new_node, self.nxt, getattr(old_node, self.nxt)
        )  # new_node.next = old_node.next
        setattr(
            getattr(old_node, self.nxt), self.prv, new_node
        )  # old_node.next.prev = new_node
        setattr(old_node, self.nxt, new_node)  # old_node.next = new_node

    def _remove_node(self, node):
        """Helper to remove node from linked list"""
        setattr(
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )  # node.prev.next = node.next
        setattr(
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )  # node.next.prev = node.prev
        # Clear self pointers to break reference cycles among evicted nodes.
        setattr(node, self.prv, None)
        setattr(node, self.nxt, None)

    def _get_lru(self) -> Optional[TreeNode]:
        """
        Get the least recently used node
        """
        if len(self.cache) == 0:
            return None
        return getattr(self.tail, self.prv)

    def reset_node_mru(self, node):
        """
        Move a (existing) node to most recently used position
        """
        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Resetting mamba tombstone node in mamba lru list: {node.id=}"
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(self, node, root_node):
        """
        Move an (existing) node and its parents to most recently used position. Child node is
        more recently used than parent node.
        """
        prev_node = self.head
        while node != root_node:
            if not self.mamba or node.mamba_value is not None:
                assert (
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru"
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def insert_mru(self, node):
        """
        Insert a (new) node as most recently used
        """
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Inserting mamba tombstone node in mamba lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        """
        Remove node from lru list
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Removing mamba tombstone node from mamba lru list: {node.id=}"
        del self.cache[node.id]
        self._remove_node(node)

    def get_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used node that is not locked
        """
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used leaf node that is not locked
        """
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_prev_no_lock(
        self, node: TreeNode, check_id: bool = True
    ) -> Optional[TreeNode]:
        """
        Get the previous (i.e. more recently used) node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no node in the lru list without lock
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):
        """
        Get the previous (i.e. more recently used) leaf node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no leaf node in the lru list without lock
        if x == self.head:
            return None
        return x

    def in_list(self, node: Optional[TreeNode]):
        """
        Check if the node is in the lru list
        """
        if not node:
            return False
        return node.id in self.cache

    def pretty_print(self, tree_cache: Optional[MambaRadixCache] = None):
        """
        Pretty print the lru list
        """
        # #695: bounded, and to the logger. This is called from the same OOM
        # path as MambaRadixCache.pretty_print; an unbounded chain walk printed
        # to stdout there is what buried the RuntimeError at 13:58:37.
        LIMIT = 24
        head = []
        x_lru = self._get_lru()
        seen = 0
        while x_lru is not None and x_lru.id in self.cache:
            seen += 1
            if len(head) < LIMIT:
                head.append(f"[{x_lru.id}] {x_lru.last_access_time:f}")
            x_lru = getattr(x_lru, self.prv)
        tail = "" if seen <= LIMIT else f" ... (+{seen - LIMIT} more)"
        logger.error(
            "%s LRU list: %d entries, oldest first: %s%s",
            f"{self.mamba=}",
            seen,
            " -> ".join(head),
            tail,
        )

        if not tree_cache:
            return
        # #695: the sibling of the walk above, bounded the same way. This one
        # heapifies EVERY node and concatenates one entry per node into a
        # single string, so on the tree that killed the 13:58:37 instance it
        # built a multi-megabyte line at the moment of an OOM.
        if self.mamba:
            nodes = tree_cache._collect_nontombstone_nodes()
        else:
            nodes = tree_cache._collect_all_nodes()
        total = len(nodes)
        heapq.heapify(nodes)
        LIMIT = 24
        oldest = []
        while nodes and len(oldest) < LIMIT:
            x = heapq.heappop(nodes)
            oldest.append(f"[{x.id}] {x.last_access_time:f}")
        tail = "" if total <= LIMIT else f" ... (+{total - LIMIT} more)"
        logger.error(
            "%s Nodes by last_access_time: %d total, oldest first: %s%s",
            f"{self.mamba=}",
            total,
            " -> ".join(oldest),
            tail,
        )

    # Note: this is expensive, only use for debug
    def sanity_check_evictable_size(self):
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)
        """
        node = self.get_lru_no_lock()
        evictable_size = 0
        while self.in_list(node):
            evictable_size += (
                len(node.value) if not self.mamba else len(node.mamba_value)
            )
            node = self.get_prev_no_lock(node)
        return evictable_size

    # Note: this is expensive, only use for debug or idle check
    def sanity_check(self, tree_cache: MambaRadixCache):
        """
        Check if the lru list is valid by rebuilding the lru list from the tree, heapifying it, and
        checking if the lru list is valid.
        """
        try:
            if self.mamba:
                nodes = tree_cache._collect_nontombstone_nodes()
            else:
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes)
            total_lru = len(self.cache)
            # heapify based on last_access_time
            heapq.heapify(nodes)
            # the root node is not in the lru list
            assert len(nodes) == (
                total_lru + (0 if self.mamba else 1)
            ), f"len(nodes): {len(nodes)}, total_lru: {total_lru}"

            x_lru = self._get_lru()
            while len(nodes):
                x = heapq.heappop(nodes)
                if x == tree_cache.root_node:
                    # root node is not in the lru list
                    continue
                assert (
                    x_lru is not None and x_lru.id in self.cache
                ), f"Incorrect LRU list, x_lru is None or not in cache: {x_lru=}, {x.id=}"

                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.mamba=}, x: {x.id=} != x_lru: {x_lru.id=}, {x.last_access_time=}, {x_lru.last_access_time=}"
                assert (
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.id=}"
                assert (
                    x_lru.mamba_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.mamba_lock_ref=}, {x_lru.id=}"
                x_lru = getattr(x, self.prv)

            if self.mamba:
                evictable_size = tree_cache.mamba_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()
            else:
                evictable_size = tree_cache.full_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.mamba=}, total nodes: {total_nodes}, total lru: {total_lru}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"
        except Exception as e:
            if get_parallel().tp_rank == 0:
                msg = f"Mamba Radix tree sanity check failed, ping @yizhang2077: {e}"
                logger.error(msg)
                tree_cache.pretty_print()
                tree_cache.full_lru_list.pretty_print(tree_cache)
                tree_cache.mamba_lru_list.pretty_print(tree_cache)
                raise Exception(msg)


class MambaRadixCache(KVCacheEventMixin, BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        assert (
            isinstance(params.token_to_kv_pool_allocator, TokenToKVPoolAllocator)
            or isinstance(
                params.token_to_kv_pool_allocator, PagedTokenToKVPoolAllocator
            )
            or isinstance(
                params.token_to_kv_pool_allocator, UnifiedMambaTokenToKVPoolAllocator
            )
        )
        self.req_to_token_pool: HybridReqToTokenPool = params.req_to_token_pool
        # #581: let the pool evict cached checkpoints from its own REQUIRED
        # allocation sites (active state / ping-pong buffers) instead of
        # asserting when the free list is empty.
        if hasattr(self.req_to_token_pool, "bind_tree_cache"):
            self.req_to_token_pool.bind_tree_cache(self)
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.mamba_cache_chunk_size = get_server_args().mamba_cache_chunk_size
        # --mamba-checkpoint-interval: when set, checkpoints live only at
        # absolute multiples of this token count; the prefix-resume point is
        # capped to that grid and the deepest checkpoints per path are
        # shielded from mamba eviction (see evict_mamba). None = upstream
        # behavior, byte-identical.
        self.mamba_checkpoint_interval = get_server_args().mamba_checkpoint_interval
        # Live window: how many of the deepest on-grid checkpoints per
        # root-to-leaf path evict_mamba tries to keep (best effort; a
        # second pass ignores the window when the pool must yield a slot).
        self.mamba_ckpt_window = envs.SGLANG_MAMBA_CKPT_WINDOW.get()
        # Strict resume: only resume at the DEEPEST interval boundary of the
        # full-KV match; if that exact checkpoint is missing, recompute from
        # scratch instead of a shallower (survivor-dependent) checkpoint.
        self.mamba_ckpt_strict_resume = envs.SGLANG_MAMBA_CKPT_STRICT_RESUME.get()
        # Per-request resume/insert attribution logging (GPU debugging).
        self.ckpt_debug = envs.SGLANG_MAMBA_CKPT_DEBUG.get()

        self.page_size = params.page_size
        self.disable = params.disable
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer
        # #755: the lock reorder. Read from the SAME predicate that decides the
        # floor, so the runtime and mamba_pool_floor cannot disagree about how
        # many slots a request may hold -- a disagreement in that direction is
        # the #581 late assert.
        self.mamba_slot_reorder = bool(
            getattr(params, "mamba_slot_reorder", False)
        )
        #: Counted, because a persistent refusal stream means write-through is
        #: not keeping up and the reduced floor is buying nothing.
        self._mamba_reorder_refusals = 0
        self.enable_mamba_extra_buffer_lazy = params.enable_mamba_extra_buffer_lazy
        self.kv_event_queue = []

        if not self.enable_mamba_extra_buffer:
            assert (
                self.page_size == 1
            ), f"Page size must be 1 for MambaRadixCache v1, got {self.page_size}"

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        self.reset()

    ##### Public API #####

    def supports_mamba(self) -> bool:
        return True

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(array("q"), None)
        self.root_node.value = []
        self.root_node.hash_value = []
        self.root_node.full_lock_ref = 1
        self.root_node.mamba_lock_ref = 1
        self.full_evictable_size_ = 0
        self.mamba_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.mamba_protected_size_ = 0
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        self.full_lru_list = LRUList(mamba=False)
        self.mamba_lru_list = LRUList(mamba=True)
        self._record_all_cleared_event()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            params: MatchPrefixParams containing key and optional Mamba-specific parameters.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        key = self._match_pre_processor(params)
        if key is None:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                best_match_node=self.root_node,
            )

        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0, mamba_exist=False)

        key = params.key
        value = params.value
        mamba_value = params.mamba_value
        prev_prefix_len = params.prev_prefix_len

        if value is None:
            value = torch.tensor([x for x in key.raw_token_ids()], dtype=torch.int64)
        prefix_len, mamba_exist = self._insert_helper(
            self.root_node,
            key,
            value,
            mamba_value,
            params.chunked,
            prev_prefix_len,
            params.force_host_write_through,
        )
        return InsertResult(prefix_len=prefix_len, mamba_exist=mamba_exist)

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        """Cache request when it finishes."""
        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free_mamba_cache(req)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        if is_insert:
            if self.enable_mamba_extra_buffer:
                cache_len = req.mamba_last_track_seqlen
                # --mamba-checkpoint-interval: the tracked position is on the
                # grid by construction (prefill targets and decode tracking
                # both use the interval); enforce it so an off-grid state can
                # never enter the tree.
                if cache_len is not None and not is_on_interval(
                    cache_len, self.mamba_checkpoint_interval
                ):
                    logger.warning(
                        "mamba checkpoint interval: dropping off-grid tracked "
                        "state at %d (interval %d), rid=%s",
                        cache_len,
                        self.mamba_checkpoint_interval,
                        req.rid,
                    )
                    cache_len = 0
            else:
                cache_len = len(token_ids)
                # ReplaySSM (no_buffer): `temporal[slot]` lags the live state by
                # the slot's unflushed ring depth (`write_pos`), so cap the
                # donate to the last flush boundary (where temporal is current)
                # and reset the cursor, keeping the donated checkpoint consistent
                # with its key length. page_size is asserted == 1, so no realign.
                write_pos_buf = self.req_to_token_pool.mamba_pool.replayssm_write_pos
                if write_pos_buf is not None:
                    cache_len -= int(write_pos_buf[req.mamba_pool_idx].item())
                    write_pos_buf[req.mamba_pool_idx] = 0
                # --mamba-checkpoint-interval (no_buffer): the donated state
                # sits exactly at `cache_len`; there is no mechanism to
                # snapshot an earlier position, so an off-grid finish is NOT
                # cached at all (rounding the key down would pair a deeper
                # state with a shorter key). Prefill-step checkpoints (grid-
                # clipped chunk ends) remain the resume points.
                if not is_on_interval(cache_len, self.mamba_checkpoint_interval):
                    cache_len = 0
            if cache_len is None:
                cache_len = 0
            if cache_len != len(token_ids):
                cache_end_idx = max(cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[cache_end_idx:])
                token_ids = token_ids[:cache_len]
                kv_indices = kv_indices[:cache_len]

            if self.page_size != 1:
                page_aligned_len = len(kv_indices) // self.page_size * self.page_size
                page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                    dtype=torch.int64, copy=True
                )
            else:
                page_aligned_len = len(kv_indices)
                page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

            assert (
                cache_len == page_aligned_len
            ), f"It is required {cache_len=}, {page_aligned_len=}, {kv_committed_len=}, {len(req.origin_input_ids)=}, {len(req.output_ids)=} ping @yizhang2077 if you see this"

            # Radix Cache takes one ref in memory pool
            # insert the token_ids and kv_indices into the radix tree
            if self.enable_mamba_extra_buffer:
                mamba_ping_pong_track_buffer_to_keep = (
                    self.req_to_token_pool.get_mamba_ping_pong_keep_idx(req)
                )
                src_active = req.mamba_ping_pong_track_buffer[
                    mamba_ping_pong_track_buffer_to_keep
                ].unsqueeze(-1)
                assert src_active.item() != -1, (
                    f"Cached mamba slot is -1: keep_idx={mamba_ping_pong_track_buffer_to_keep}, "
                    f"buf={req.mamba_ping_pong_track_buffer.tolist()}, "
                    f"next_track_idx={req.mamba_next_track_idx}, "
                    f"last_track_seqlen={req.mamba_last_track_seqlen}, "
                    f"rid={req.rid}"
                )
                if self.int8_ckpt_pool is not None:
                    mamba_value = self._commit_int8_checkpoint(src_active)
                    # quantized -> no ping-pong slot needs keeping
                    mamba_ping_pong_track_buffer_to_keep = None
                else:
                    mamba_value = src_active.clone()
            else:
                if self.int8_ckpt_pool is not None:
                    mamba_value = self._commit_int8_checkpoint(
                        req.mamba_pool_idx.unsqueeze(-1)
                    )
                else:
                    mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
                mamba_ping_pong_track_buffer_to_keep = None

            if mamba_value is None:
                # CACHE-INSERT degradation: the int8 checkpoint pool could not
                # yield a slot (see `_alloc_mamba_slot`). Skip the insert and
                # release the KV exactly like the `is_insert=False` path does --
                # the state is simply not cached, the request still finishes.
                self.token_to_kv_pool_allocator.free(
                    page_aligned_kv_indices[req.cache_protected_len :]
                )
                # `mamba_exist=True` + no ping-pong slot to keep, i.e. the
                # request's active mamba slot goes back to the pool.
                self.req_to_token_pool.free_mamba_cache(
                    req, mamba_ping_pong_track_buffer_to_keep=None
                )
                self.dec_lock_ref(req.last_node)
                return

            result = self.insert(
                InsertParams(
                    key=RadixKey(token_ids[:page_aligned_len], req.extra_key),
                    value=page_aligned_kv_indices,
                    mamba_value=mamba_value,
                    prev_prefix_len=req.cache_protected_len,
                    force_host_write_through=requests_forced_host_write_through(req),
                )
            )
            mamba_exist = result.mamba_exist
            if self.ckpt_debug:
                logger.info(
                    "mamba-ckpt cache_finished: rid=%s cache_len=%d slot=%s "
                    "mamba_exist=%s last_track=%s total_len=%d",
                    req.rid,
                    cache_len,
                    int(mamba_value[0].item()),
                    mamba_exist,
                    req.mamba_last_track_seqlen,
                    kv_committed_len,
                )
            if mamba_exist and self.int8_ckpt_pool is not None:
                # state already cached -> the int8 slot we just allocated is a duplicate
                self.int8_ckpt_pool.free(mamba_value)
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])
            mamba_exist = True

        if mamba_exist:
            mamba_ping_pong_track_buffer_to_keep = None

        # With int8 checkpoints the radix owns an int8 slot (not the request's active
        # slot), so the active mamba slot must always be returned to the active pool.
        free_mamba_cache = (
            True
            if (self.enable_mamba_extra_buffer or self.int8_ckpt_pool is not None)
            else mamba_exist
        )

        if free_mamba_cache:
            self.req_to_token_pool.free_mamba_cache(
                req,
                mamba_ping_pong_track_buffer_to_keep=mamba_ping_pong_track_buffer_to_keep,
            )

        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""

        def _skip_cache_unfinished_req(req: Req) -> None:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.extend_range.end
            ]

            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        token_ids = req.get_fill_ids()
        cache_len = (
            req.mamba_last_track_seqlen
            if self.enable_mamba_extra_buffer
            else len(token_ids)
        )
        if self.disable or cache_len is None:
            return _skip_cache_unfinished_req(req)
        # --mamba-checkpoint-interval: only grid positions may carry a
        # checkpoint. extra_buffer targets are on-grid by construction;
        # no_buffer donates the live state at the chunk end, which the
        # scheduler clips to the grid — an off-grid end (edge paths) is
        # simply not cached this step.
        if not is_on_interval(cache_len, self.mamba_checkpoint_interval):
            if self.enable_mamba_extra_buffer:
                logger.warning(
                    "mamba checkpoint interval: off-grid unfinished track "
                    "position %d (interval %d), skipping cache, rid=%s",
                    cache_len,
                    self.mamba_checkpoint_interval,
                    req.rid,
                )
            return _skip_cache_unfinished_req(req)

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        # kv_indices is the kv indices to be cached
        kv_indices = kv_indices_orig[:cache_len]
        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        assert page_aligned_len == len(
            kv_indices
        ), f"page_aligned_len != len(kv_indices), {page_aligned_len=}, {len(kv_indices)=}, {cache_len=}, {self.page_size=}, {self.mamba_cache_chunk_size=}"

        page_aligned_token_ids = token_ids[:page_aligned_len]

        # Donate the mamba index to the radix cache instead of copying.
        # This avoids a data copy that would race with the forward stream.
        # Checkpoint donation is a CACHE-INSERT path: a mamba slot that cannot be
        # served degrades to "do not cache this state this step" (the request
        # keeps computing and simply misses the cache later). Every allocation
        # that can fail is done BEFORE any request-visible mutation, so a skip
        # never leaves the ping-pong buffer half-swapped.
        # #755: only the no_buffer arm below performs the lock reorder. The
        # other arms keep the original order, so the release stays at the tail
        # for them -- initialised here so the tail reads one variable rather
        # than re-deriving which arm ran.
        early_release = False

        if self.int8_ckpt_pool is not None:
            # int8 path: quantize the to-be-cached active state into an int8 slot
            # (strategy-agnostic donate hook).
            if self.enable_mamba_extra_buffer:
                new_slot = self._alloc_mamba_slot()
                ckpt_slot = None if new_slot is None else self._alloc_int8_ckpt_slot()
                if ckpt_slot is None:
                    if new_slot is not None:
                        self.req_to_token_pool.mamba_allocator.free(new_slot)
                    return _skip_cache_unfinished_req(req)
                src_active = self.req_to_token_pool.donate_mamba_ping_pong_slot(
                    req, new_slot
                )
                self.int8_ckpt_pool.store_from_active(
                    self.req_to_token_pool.mamba_pool, src_active, ckpt_slot
                )
                mamba_value_donated = ckpt_slot
                self.req_to_token_pool.mamba_allocator.free(src_active)
            else:
                mamba_value_donated = self._commit_int8_checkpoint(
                    req.mamba_pool_idx.view(-1)
                )
                if mamba_value_donated is None:
                    return _skip_cache_unfinished_req(req)
        elif self.enable_mamba_extra_buffer:
            new_slot = self._alloc_mamba_slot()
            if new_slot is None:
                return _skip_cache_unfinished_req(req)
            mamba_value_donated = self.req_to_token_pool.donate_mamba_ping_pong_slot(
                req, new_slot
            )
        else:
            # #755: THE LOCK REORDER. Default order is
            #   alloc -> copy -> insert -> dec(old) -> inc(new)
            # which holds the OLD pin across the alloc, so a request owns
            # active + donated + pinned = 3 slots at this instant. The donated
            # slot BECOMES the next pin, so the double-count exists only
            # because of the overlap. Releasing first makes it 2.
            #
            # SAFE ONLY WITH A HOST-BACKED ANCHOR, per node and at THIS moment
            # (NOTE_755 section 3). Between the release and the new pin the old
            # node is evictable; with a host copy that degrades to load_back or
            # re-prefill, without one it is a DEAD anchor. So the config
            # predicate is not enough -- the node is asked too.
            early_release = self._mamba_early_release_admissible(req.last_node)
            if early_release:
                self.dec_lock_ref(req.last_node)
            elif self.mamba_slot_reorder:
                # The config promised the reduced floor, but THIS anchor is not
                # backed. Reverting to the 3-slot order here would claim a slot
                # the pool no longer reserves -- the #581 late-assert class. So
                # skip the insert instead: that path holds active + old pin = 2
                # and stays inside the budget. Loud, and counted.
                self._mamba_reorder_refusals += 1
                if self._mamba_reorder_refusals in (1, 10) or (
                    self._mamba_reorder_refusals % 100 == 0
                ):
                    logger.warning(
                        "#755: skipping the mamba cache insert for rid=%s -- "
                        "the resume anchor is not host-backed at release time, "
                        "and the reduced floor has no third slot to fall back "
                        "on. The request keeps computing; only this checkpoint "
                        "is not cached. (%d so far; a persistent count means "
                        "write-through is not keeping up.)",
                        getattr(req, "rid", "?"),
                        self._mamba_reorder_refusals,
                    )
                return _skip_cache_unfinished_req(req)

            mamba_value_donated = self._alloc_mamba_slot()
            if mamba_value_donated is None:
                if early_release:
                    # The window opened and the alloc failed. The anchor is
                    # host-backed (that is what admitted the release), so the
                    # request resumes via load_back rather than losing state.
                    logger.warning(
                        "#755: mamba slot alloc failed inside the release "
                        "window for rid=%s; the old anchor was host-backed, so "
                        "the resume path is load_back / re-prefill.",
                        getattr(req, "rid", "?"),
                    )
                return _skip_cache_unfinished_req(req)
            # mamba_pool is a pure PHYSICAL store; translate both slot ids
            # virtual->physical (identity for the non-unified memory pool) before the copy.
            translate = self.req_to_token_pool.translate_mamba_indices
            self.req_to_token_pool.mamba_pool.copy_from(
                translate(req.mamba_pool_idx.unsqueeze(0)),
                translate(mamba_value_donated),
            )

        result = self.insert(
            InsertParams(
                key=RadixKey(page_aligned_token_ids, req.extra_key),
                value=page_aligned_kv_indices,
                mamba_value=mamba_value_donated,
                prev_prefix_len=req.cache_protected_len,
                chunked=chunked,
            )
        )
        new_prefix_len, mamba_exist = result.prefix_len, result.mamba_exist
        if self.ckpt_debug:
            logger.info(
                "mamba-ckpt cache_unfinished: rid=%s cache_len=%d slot=%s "
                "mamba_exist=%s fill_len=%d",
                req.rid,
                cache_len,
                int(mamba_value_donated[0].item()),
                mamba_exist,
                len(token_ids),
            )
        if mamba_exist:
            self._free_mamba_value(mamba_value_donated)

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(
            MatchPrefixParams(key=RadixKey(page_aligned_token_ids, req.extra_key))
        )
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        if not mamba_exist:
            assert torch.equal(new_last_node.mamba_value, mamba_value_donated)

        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {len(page_aligned_token_ids)=}, {mamba_exist=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # #755: the release already happened before the alloc when the reorder
        # was admitted. Releasing again here would double-decrement the ref.
        if not early_release:
            self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # NOTE: this is needed for both page_size == 1 and page_size > 1
        req.prefix_indices = torch.cat(
            [new_indices, kv_indices_orig[len(new_indices) :]]
        )
        req.cache_protected_len = len(new_indices)
        req.mamba_last_track_seqlen = None
        req.last_node = new_last_node

    def pretty_print(self) -> None:
        """One bounded record about the tree's SHAPE, at ERROR level.

        #695. This runs on the production OOM path (mem_cache/common.py's
        alloc handlers call it between logging the error and raising it), and
        what it used to do there was print one bare line PER NODE. At the
        2026-08-16 13:58:37 crash that was ~370 lines of `print()` after the
        line that mattered:

            RuntimeError: Out of memory. Try to allocate 512 tokens.
            Available full tokens: 167743 (available=392 + evictable=167351)

        The first diagnosis read the visible tail, found "terminate called
        without an active exception" with no Python frame, and filed a
        teardown abort. The RuntimeError was 300 lines above. The dump did not
        just add noise, it MOVED THE CAUSE OUT OF VIEW.

        WHAT AN OOM READER ACTUALLY NEEDS is the shape, not the tree: how many
        nodes, how many tokens, and HOW MUCH IS LOCKED -- because the question
        at an allocation failure with six figures of "evictable" is whether the
        counter was promising tokens the eviction frontier could not reach
        (#681: correct as a count, wrong as a capability). The locked share is
        that question in one number.

        The per-node detail is not deleted, only moved behind
        SGLANG_RADIX_DEBUG_DUMP, and it goes to the logger there too.
        """
        summary = self._shape_summary()
        if _radix_debug_dump():
            lines = []
            self._print_helper(self.root_node, 0, sink=lines.append)
            logger.error(
                "%s\nper-node detail (SGLANG_RADIX_DEBUG_DUMP):\n%s",
                summary,
                "\n".join(lines),
            )
        else:
            logger.error("%s", summary)

    def _shape_summary(self) -> str:
        """Walk once; report shape. NEVER RAISES.

        A reporter on an error path that can raise will replace the error it
        was called to explain -- and this one could: ``_print_helper`` asserts
        the child-key invariant mid-walk, so a tree that is BOTH out of memory
        AND structurally surprising reported the AssertionError and lost the
        OOM. Structural surprises are counted and named here instead.
        """
        nodes = tokens = mamba_tokens = 0
        locked_nodes = locked_tokens = 0
        mismatched = 0
        max_depth = 0
        try:
            stack = [(self.root_node, 0)]
            while stack:
                node, depth = stack.pop()
                nodes += 1
                max_depth = max(max_depth, depth)
                # `node.value` is a TENSOR. `tensor or ()` evaluates
                # bool(tensor), which RAISES for any multi-element tensor --
                # so this reporter died on the second real node it touched.
                # Measured 2026-08-17 02:18:09, on all three ranks, inside the
                # crash this walk exists to explain:
                #   RADIX SHAPE: walk failed after 3 nodes (RuntimeError(
                #   'Boolean value of Tensor with more than one value is
                #   ambiguous')). Partial: tokens=1, locked_nodes=1.
                # A diagnostic that only works on an empty tree is not a
                # diagnostic. Length is asked for directly, never via truthiness.
                val = getattr(node, "value", None)
                n = 0 if val is None else len(val)
                tokens += n
                mv = getattr(node, "mamba_value", None)
                if mv is not None:
                    mamba_tokens += len(mv)
                if getattr(node, "full_lock_ref", 0) or getattr(
                    node, "mamba_lock_ref", 0
                ):
                    locked_nodes += 1
                    locked_tokens += n
                for key, child in (getattr(node, "children", {}) or {}).items():
                    try:
                        if key != child.key.child_key(self.page_size):
                            mismatched += 1
                    except Exception:  # noqa: BLE001 - counted, never fatal
                        mismatched += 1
                    stack.append((child, depth + 1))
        except Exception as exc:  # noqa: BLE001 - a reporter must not raise
            return (
                f"RADIX SHAPE: walk failed after {nodes} nodes ({exc!r}). "
                f"Partial: tokens={tokens}, locked_nodes={locked_nodes}."
            )
        pct = (100.0 * locked_tokens / tokens) if tokens else 0.0
        return (
            f"RADIX SHAPE: {nodes} nodes, {tokens} full tokens, "
            f"{mamba_tokens} mamba tokens, max depth {max_depth}; "
            f"LOCKED {locked_nodes} nodes / {locked_tokens} tokens "
            f"({pct:.1f}% of tokens). "
            f"child-key mismatches: {mismatched}. "
            f"At an allocation failure the locked share is the question: "
            f"tokens counted evictable but sitting behind a locked chain are "
            f"not reachable by the eviction frontier (#681). "
            f"Set SGLANG_RADIX_DEBUG_DUMP=1 for per-node detail."
        )

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def _evict_leaf_node(
        self, x: TreeNode, is_evict_mamba: bool
    ) -> Tuple[int, int, TreeNode, TreeNode]:
        assert (
            x.full_lock_ref == 0 and x.mamba_lock_ref == 0
        ), f"evict leaf node invalid with {x.id=} {x.full_lock_ref=} {x.mamba_lock_ref=}"

        assert x.mamba_value is not None, f"leaf node mamba value is not None, {x.id=}"
        # 1. a leaf node, free full tokens and mamba
        self._record_remove_event(x)
        self.token_to_kv_pool_allocator.free(x.value)
        full_num_evicted = len(x.value)
        self._free_mamba_value(x.mamba_value)
        mamba_num_evicted = len(x.mamba_value)

        # 2. get the next node, update the lru lists
        if is_evict_mamba:
            x_next = self.mamba_lru_list.get_prev_no_lock(x)
        else:
            x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
        self.full_lru_list.remove_node(x)
        self.mamba_lru_list.remove_node(x)

        # 3. delete the leaf node
        self._delete_leaf(x)

        # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
        x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
        full_num_evicted += leaf_full_num_evicted
        return full_num_evicted, mamba_num_evicted, x, x_next

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        full_num_evicted = 0
        mamba_num_evicted = 0

        if params.num_tokens > 0:
            full_num_evicted = self.evict_full(params.num_tokens)
        if params.mamba_num > 0:
            mamba_num_evicted = self.evict_mamba(params.mamba_num)

        return EvictResult(
            num_tokens_evicted=full_num_evicted, mamba_num_evicted=mamba_num_evicted
        )

    def evict_mamba(self, mamba_num: int) -> int:
        """Evict mamba states. Returns the number of mamba states evicted.

        With --mamba-checkpoint-interval a first pass spares the deepest
        ``mamba_ckpt_window`` checkpoints of every path (the prefix-resume
        anchors: losing the deepest one silently moves the resume point of
        identical requests and re-introduces run-to-run drift). The window
        is best effort — a second pass ignores it when the pool must yield.
        """
        if self.disable or mamba_num <= 0:
            return 0
        # #743 INSTRUMENT: a SUCCESSFUL eviction was the silent event. The
        # starvation emitter below covers the pool failing to yield; nothing
        # covered it yielding by destroying a cached anchor, which is the
        # event that costs prefix reuse. The per-node anchor depths cost a
        # walk to the root each, so they are collected ONLY when the rate
        # limiter will actually print them.
        from sglang.srt.mem_cache.mamba_slot_observer import (
            clock,
            emit_lines,
            observer_of,
            probe_available,
        )

        obs = observer_of(self)
        now = clock()
        trace = [] if obs.would_emit(now) else None
        # #747: one rule, both lineages. MambaRadixCache is device-only, so
        # an evicted anchor is a lost anchor and the protection stays on.
        protect = protect_deepest_anchors(
            self.mamba_checkpoint_interval, host_tier_present=False
        )
        mamba_num_evicted = self._evict_mamba_pass(
            mamba_num, protect_window=protect, trace=trace
        )
        self._mamba_evict_pass1 = getattr(self, "_mamba_evict_pass1", 0) + int(
            mamba_num_evicted
        )
        if protect and mamba_num_evicted < mamba_num:
            second = self._evict_mamba_pass(
                mamba_num - mamba_num_evicted, protect_window=False, trace=trace
            )
            mamba_num_evicted += second
            # #767 ACCEPTANCE COUNTER. The second pass is the one that drops a
            # protected anchor, which is what moves a resume point. It exists so
            # protection can never deadlock an allocation and is correct; but at
            # the acceptance load it must not be REACHED, because every anchor it
            # takes is a determinism loss the first pass was supposed to prevent.
            # A non-zero count here says the pool is too small for the protected
            # set -- a SIZING answer -- or that the protected set is too broad.
            if second > 0:
                self._mamba_evict_pass2 = getattr(self, "_mamba_evict_pass2", 0) + int(
                    second
                )
                n = self._mamba_evict_pass2
                if n <= 3 or n % 200 == 0:
                    # INFO, not WARNING: the second pass is documented as
                    # legitimate ("best effort -- a second pass ignores it when
                    # the pool must yield"), and anchor eviction was later
                    # FALSIFIED as the cause of the drift this counter was added
                    # to chase. It stays as an accounting line, not an alarm.
                    logger.info(
                        "#767 SECOND-PASS EVICTION: dropped %d protected "
                        "anchor(s) because the first pass freed %d of %d "
                        "needed (pass1 total %d, pass2 total %d, evictable %d). "
                        "Every one of these moves a resume point.",
                        second,
                        mamba_num_evicted - second,
                        mamba_num,
                        getattr(self, "_mamba_evict_pass1", 0),
                        n,
                        self.mamba_evictable_size(),
                    )
        emit_lines(
            logger,
            obs.note_eviction(
                now=now,
                requested=mamba_num,
                evicted=mamba_num_evicted,
                nodes=trace or (),
                evictable=self.mamba_evictable_size(),
                protected=self.mamba_protected_size(),
                available=probe_available(
                    getattr(self.req_to_token_pool, "mamba_allocator", None)
                ),
                lineage="device",
            ),
        )
        return mamba_num_evicted

    def _evict_mamba_pass(
        self, mamba_num: int, protect_window: bool, trace: Optional[List] = None
    ) -> int:
        # get the least recently used node that is not locked, doesn't have to be a leaf
        x = self.mamba_lru_list.get_lru_no_lock()
        mamba_num_evicted = 0
        # evict lru leaf nodes until mamba_num_tokens is reached
        while mamba_num_evicted < mamba_num and (self.mamba_lru_list.in_list(x)):
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert (
                len(x.mamba_value) == 1
            ), f"node has abnormal mamba length, {x.id=}, {len(x.mamba_value)=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.mamba_lock_ref == 0, f"node is in use by mamba kv indices, {x.id=}"

            if protect_window and self._in_ckpt_live_window(x):
                x = self.mamba_lru_list.get_prev_no_lock(x)
                continue

            if trace is not None:
                # BEFORE the node is destroyed. A leaf is removed from the
                # tree by `_evict_leaf_node`, so its depth is unrecoverable
                # afterwards; recording both branches here keeps the two
                # cases reporting the same quantity.
                from sglang.srt.mem_cache.mamba_slot_observer import (
                    anchor_depth_tokens,
                )

                trace.append((x.id, anchor_depth_tokens(x)))

            if len(x.children) > 0:
                # 1. an internal node, free mamba tokens.
                self._free_mamba_value(x.mamba_value)
                mamba_num_evicted += len(x.mamba_value)

                # 2. get the next node, update the lru lists
                x_next = self.mamba_lru_list.get_prev_no_lock(x)
                self.mamba_lru_list.remove_node(x)

                # 3. tombstone the node
                self._tombstone_internal_node(x)
            else:
                _, mamba_evicted_delta, _, x_next = self._evict_leaf_node(x, True)
                mamba_num_evicted += mamba_evicted_delta

            x = x_next

        return mamba_num_evicted

    def _in_ckpt_live_window(self, node: TreeNode) -> bool:
        """True if ``node`` is one of the deepest ``mamba_ckpt_window``
        checkpoints on some root-to-leaf path through it (i.e. fewer than
        ``mamba_ckpt_window`` mamba-valued nodes exist strictly below it
        along at least one descendant path)."""
        window = self.mamba_ckpt_window
        if window <= 0:
            return False
        if len(node.children) == 0:
            return True  # a leaf is always the deepest checkpoint of its path
        # DFS with the running count of checkpoints below `node`; early out
        # as soon as one path stays under the window.
        stack = [(child, 0) for child in node.children.values()]
        while stack:
            cur, cnt = stack.pop()
            if cur.mamba_value is not None:
                cnt += 1
            if cnt >= window:
                continue
            if len(cur.children) == 0:
                return True
            stack.extend((child, cnt) for child in cur.children.values())
        return False

    def evict_full(self, full_num_tokens: int) -> int:
        """Evict full KV cache. Returns the number of tokens evicted."""
        if self.disable or full_num_tokens <= 0:
            return 0

        full_num_evicted = 0
        # get the least recently used leaf node that is not locked
        x = self.full_lru_list.get_leaf_lru_no_lock()

        while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):
            assert (
                x != self.root_node
            ), f"root node should not exist in full lru list, {x.id=}"
            if x.mamba_value is None:
                # #681: AN UNLOCKED TOMBSTONE LEAF, WHICH THE SELECTOR OFFERS
                # AND THE OLD CONSUMER REFUSED.
                #
                # `get_leaf_lru_no_lock` asks for unlocked and childless;
                # `_evict_leaf_node` additionally demands a mamba value and
                # asserted when it was missing. That combination is reachable
                # WITHOUT any invariant being broken:
                # `_iteratively_delete_tombstone_leaf` breaks on
                # `node.parent.full_lock_ref > 0`, so a tombstone that loses
                # its last child while a request holds it survives as a LOCKED
                # tombstone leaf -- and when that request finishes, nothing
                # revisits it. It is then unlocked, childless, counted in
                # `full_evictable_size_`, and first in line at the frontier.
                #
                # Measured 2026-08-16 01:46:10: the tree all three ranks
                # printed held exactly one (node 5937 -- fr=0, mv=None,
                # childless, in the full LRU list). Replaying that dumped tree
                # through this function selects it and dies on the assert,
                # which kills the whole group over a state the cache itself
                # produced and can perfectly well pay.
                #
                # Freeing it here is not a new capability: it is the same
                # deletion `_iteratively_delete_tombstone_leaf` would have
                # performed one step earlier, taken now that the lock which
                # deferred it is gone. Deliberately NOT extended past a lock --
                # a LOCKED tombstone leaf is still refused, because reaching
                # behind a live reference is a different repair.
                x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
                full_num_evicted += self._free_tombstone_leaf(x)
                x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
                full_num_evicted += leaf_full_num_evicted
            else:
                full_num_evicted_delta, _, x, x_next = self._evict_leaf_node(x, False)
                full_num_evicted += full_num_evicted_delta

            # if parent has no more children, it is a leaf. It is possible that this node is lru, so
            # we need to get the first leaf node in the lru list
            if len(x.parent.children) == 0:
                x_next = self.full_lru_list.get_leaf_lru_no_lock()

            x = x_next

        return full_num_evicted

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        """
        Increment the lock reference count for the node.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return IncLockRefResult()

        result = IncLockRefResult()

        # protect mamba value in current node if it exists
        if node.mamba_value is not None:
            if node.mamba_lock_ref == 0:
                self.mamba_evictable_size_ -= len(node.mamba_value)
                self.mamba_protected_size_ += len(node.mamba_value)
            node.mamba_lock_ref += 1
        else:
            # INVARIANT: a release may only decrement refs this acquire took.
            # A mamba TOMBSTONE takes no mamba ref here, and the node can gain
            # a mamba value while this lock is held (an insert re-attaches a
            # checkpoint; in the hierarchical subclass a load-back revives one
            # and takes its OWN pin). Recording the skip lets the paired
            # release leave that value's ref alone instead of consuming it.
            result.skip_lock_node_ids.setdefault(ComponentType.MAMBA, set()).add(
                node.id
            )

        while node != self.root_node:
            # lock full from node to root
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
            node.full_lock_ref += 1
            node = node.parent
        return result

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return DecLockRefResult()

        # A node in the skip set was a mamba tombstone when the paired acquire
        # ran, so no mamba ref was taken for it and none may be released here.
        skipped_mamba = node.id in _skipped_ids(params, ComponentType.MAMBA)
        if node.mamba_value is not None and not skipped_mamba:
            assert (
                node.mamba_lock_ref > 0
            ), f"dec_lock_ref on node with {node.mamba_lock_ref=}, {node.id=}"
            if node.mamba_lock_ref == 1:
                self.mamba_evictable_size_ += len(node.mamba_value)
                self.mamba_protected_size_ -= len(node.mamba_value)
            node.mamba_lock_ref -= 1

        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
            node.full_lock_ref -= 1
            node = node.parent

        return DecLockRefResult()

    def sanity_check(self):
        if self.disable:
            return
        self.full_lru_list.sanity_check(self)
        self.mamba_lru_list.sanity_check(self)

    def evictable_size(self) -> Tuple[int, int]:
        # Note: use full_evictable_size() and mamba_evictable_size() instead.
        raise NotImplementedError

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable_size_

    def protected_size(self) -> Tuple[int, int]:
        # Note: use full_protected_size() and mamba_protected_size() instead.
        raise NotImplementedError

    def full_protected_size(self) -> int:
        # protected size refers to the size of the full cache that is locked
        return self.full_protected_size_

    def mamba_protected_size(self) -> int:
        # protected size refers to the size of the mamba cache that is locked
        return self.mamba_protected_size_

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def all_mamba_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            if node.mamba_value is not None:
                values.append(node.mamba_value)
            for _, child in node.children.items():
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def available_and_evictable_str(self) -> str:
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = self.full_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Full LRU list evictable size: {self.full_lru_list.sanity_check_evictable_size()}\n"
        )

    ##### Internal Helper Functions #####

    def _mamba_early_release_admissible(self, node) -> bool:
        """#755: may this node's pin be released BEFORE the donation alloc?

        Two questions, both required, and deliberately asked separately:

        * is the reorder configured at all (``mamba_slot_reorder``) -- the
          config-level predicate that also decided the floor;
        * is THIS node host-backed RIGHT NOW (``mamba_backuped``) -- because
          the release opens a window in which the node is evictable, and the
          whole safety argument is that an eviction there costs a
          ``load_back`` rather than the anchor.

        A config-only check would be the formula-only edit NOTE_755 refuses:
        it would release pins for nodes whose backup does not exist yet.

        Also required: the copy must be COMPLETE, not merely queued (#767).
        ``mamba_backuped`` reads ``mamba_host_value is not None``, and the
        write-through path sets that the moment the transfer is HANDED to the
        cache controller -- the same block then records the node in
        ``ongoing_write_through``. Between those two facts the anchor exists
        as an intention only, so a release there is precisely the dead anchor
        this predicate exists to prevent.
        """
        if not self.mamba_slot_reorder:
            return False
        if node is None or node is getattr(self, "root_node", None):
            # The root is never evicted and carries no mamba value to lose;
            # releasing early buys nothing and the pin bookkeeping is simpler
            # kept uniform.
            return False
        if not bool(getattr(node, "mamba_backuped", False)):
            return False
        return self._mamba_host_copy_complete(node)

    def _mamba_host_copy_complete(self, node) -> bool:
        """#767: has this node's host copy actually LANDED?

        The device-only pool has no asynchronous write-through, so a node that
        carries a host value carries a finished one and the answer is True.
        The hierarchical subclass overrides this, because there the value is
        published when the transfer is queued.
        """
        return True

    def _alloc_mamba_slot(self) -> Optional[torch.Tensor]:
        """Allocate one mamba pool slot, evicting if necessary.

        Returns ``None`` when the pool is exhausted AND eviction cannot free a
        slot. That happens when every cached mamba state belongs to a running
        request: `evict_mamba` walks the LRU through `get_lru_no_lock`, which
        skips every node with `mamba_lock_ref > 0`, so it evicts nothing and
        returns 0. This is a legitimate transient runtime state, not a bug --
        it used to `assert` and kill the scheduler process. Callers MUST handle
        `None`; for cache-insert paths the correct degradation is to skip
        caching this state (a later cache miss), never to crash.

        NOT rank-uniform on its own (#639b). This docstring used to claim it
        was -- "rank-uniform without a collective: `max_mamba_cache_size` is
        min-reduced across ranks at startup (see `_sync_uneven_mamba_cache_size`),
        the schedulers are replicated and see the identical request stream, so
        the pool reaches exhaustion on every rank in the same step" -- and that
        reasoning is wrong. It was part of the 2026-08-07 07:45 / 10:04
        `PrefixLensRankDivergence` crashes (rank 0 sum=19711, rank 1 sum=16957).

        A uniform pool SIZE is not a uniform eviction OUTCOME. The startup
        min-reduce equalises how many slots each rank HAS; it says nothing
        about which slots are FREE. Occupancy is what this branch tests, and
        it diverges from two rank-local sources that the size reduce cannot
        touch:

        * WHICH node gets tombstoned is chosen by a rank-local LRU walk
          (`evict_mamba` / `MambaComponent.drive_eviction` use
          `get_lru_no_lock`, which skips nodes with `mamba_lock_ref > 0`), so
          equal-sized pools at different occupancy tombstone different nodes.
        * the degrade branch itself is rank-local: returning `None` makes the
          caller SKIP a cache insert on that rank only, so the trees stop being
          replicas and the divergence sustains itself.

        And the tombstone is visible to the matcher -- it clears the mamba
        state while leaving the node's KV in place, so the node still exists
        on both ranks but only one of them will match through it.

        The eviction trigger is therefore pinned to the group MIN of the mamba
        pool's availability, published once per iteration by the scheduler from
        the reduce it already runs (`_publish_uniform_mamba_floor`). No new
        collective, and no floor at all on a single rank or on ranks whose
        occupancy agrees.
        """
        # Imported here rather than at module scope: this file's import block
        # is already an E402 region, and a local import keeps the new
        # dependency out of it without adding a finding.
        from sglang.srt.mem_cache.common import peer_needs_mamba_evict

        slot = self.req_to_token_pool.mamba_allocator.alloc(1)
        if slot is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.req_to_token_pool.mamba_allocator.alloc(1)
            if slot is None:
                self._log_mamba_slot_starvation("mamba")
        elif peer_needs_mamba_evict(self):
            # #639b: this rank had a slot, a peer did not; match its tombstone.
            # Unreachable unless a floor was published, so a single-rank or
            # even-occupancy boot takes exactly the pre-#639b path.
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
        return slot

    def _log_mamba_slot_starvation(self, pool_name: str) -> None:
        """Rate-limited warning for an unservable mamba slot request."""
        self._mamba_starvation_count = getattr(self, "_mamba_starvation_count", 0) + 1
        count = self._mamba_starvation_count
        # Log the first few, then powers of ten -- a starved pool can hit this
        # every step and must not flood the scheduler log.
        if count <= 3 or count % 1000 == 0:
            logger.warning(
                "%s slot pool exhausted and nothing evictable (all cached states "
                "are locked by running requests): skipping this cache insert. "
                "occurrence=%d mamba_evictable=%d mamba_protected=%d",
                pool_name,
                count,
                self.mamba_evictable_size(),
                self.mamba_protected_size(),
            )

    @property
    def int8_ckpt_pool(self):
        """The int8 checkpoint pool, or None when --enable-int8-mamba-checkpoint is off.
        When enabled, radix-cached mamba states live HERE (int8), not in the active
        bf16 pool -> ~2x cached-prefix capacity at fixed memory."""
        return getattr(self.req_to_token_pool, "mamba_ckpt_pool", None)

    def _alloc_int8_ckpt_slot(self) -> Optional[torch.Tensor]:
        """Allocate one int8 checkpoint slot, evicting cached states if the pool is full.

        Returns ``None`` when exhausted with nothing evictable (see
        `_alloc_mamba_slot`); callers skip the cache insert instead of crashing.
        """
        slot = self.int8_ckpt_pool.alloc(1)
        if slot is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.int8_ckpt_pool.alloc(1)
            if slot is None:
                self._log_mamba_slot_starvation("int8 mamba checkpoint")
        return slot

    def _commit_int8_checkpoint(
        self, active_slots: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Quantize the active-pool state at ``active_slots`` into a fresh int8
        checkpoint slot and return that slot. Strategy-agnostic donate hook: both
        no_buffer (copy_from) and extra_buffer (ping-pong) converge here. The caller
        frees ``active_slots`` separately.

        Returns ``None`` when the checkpoint pool cannot yield a slot; the caller
        must then skip the cache insert."""
        ckpt_slot = self._alloc_int8_ckpt_slot()
        if ckpt_slot is None:
            return None
        self.int8_ckpt_pool.store_from_active(
            self.req_to_token_pool.mamba_pool, active_slots, ckpt_slot
        )
        return ckpt_slot

    def _free_mamba_value(self, mamba_value: torch.Tensor) -> None:
        """Free a node's mamba_value to the right allocator (int8 ckpt pool or the
        active mamba allocator)."""
        if self.int8_ckpt_pool is not None:
            self.int8_ckpt_pool.free(mamba_value)
        else:
            self.req_to_token_pool.mamba_allocator.free(mamba_value)

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """
        Mamba prefix matching helper. It factors in the sliding window size such that
        the matched node is guaranteed to either 1. connected to root without mamba tombstone,
        or 2. the number of matching tokens from the matched node to the last mamba tombstone
        node is greater than or equal to the sliding window size.
        """
        node = self.root_node
        child_key = key.child_key(self.page_size)

        value: List[torch.Tensor] = []
        # Token depth of `node` (sum of matched value lengths so far). With
        # --mamba-checkpoint-interval only checkpoints at absolute interval
        # multiples are resume candidates: off-grid checkpoints (legacy
        # entries or unaligned edge paths) would re-introduce
        # traffic-dependent resume points.
        cum_tokens = 0
        best_value_len = 0
        best_last_node = node
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            # update best_value_len and best_last_node if needed
            # #747: one anchor rule, both lineages (see mamba_ckpt_utils).
            if is_resume_candidate(
                cum_tokens,
                self.mamba_checkpoint_interval,
                has_device_value=node.mamba_value is not None,
            ):
                best_value_len = len(value)
                best_last_node = node

            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                cum_tokens += len(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)
        # handle best_value_len and best_last_node, for the case that last node is fully matched
        if is_resume_candidate(
            cum_tokens,
            self.mamba_checkpoint_interval,
            has_device_value=node.mamba_value is not None,
        ):
            best_value_len = len(value)
            best_last_node = node

        return value, best_last_node, best_value_len

    def _match_pre_processor(self, params: MatchPrefixParams) -> Optional[RadixKey]:
        """Preprocess the key before matching."""
        key = params.key

        if self.disable or len(key) == 0:
            return None

        return key

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: List[torch.Tensor],
        last_node: TreeNode,
        best_value_len: int,
    ) -> MatchResult:
        """Post-process the matched result."""
        cow_mamba = params.cow_mamba
        req = params.req

        # update time for matched nodes, and make nodes closer to root to be least recently used
        # this allows mamba to evict nodes closer to root first
        node_update = last_node
        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
        self.mamba_lru_list.reset_node_and_parents_mru(node_update, self.root_node)

        # This last_access_time is for sanity check, can be deleted after validation in production
        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= (
                0.00001  # assuming less than 100000 nodes in a branch of the tree
            )
            node_update = node_update.parent

        # --mamba-checkpoint-interval strict resume: identical requests must
        # resume at the DEEPEST interval boundary of their match or not at
        # all — a shallower surviving checkpoint depends on which entries
        # the LRU churn spared and would vary run-to-run. The branching
        # logic below then re-establishes the missing deep checkpoint.
        if (
            self.mamba_checkpoint_interval is not None
            and self.mamba_ckpt_strict_resume
            and best_value_len > 0
        ):
            total_match_tokens = sum(len(v) for v in value)
            best_depth = sum(len(v) for v in value[:best_value_len])
            if best_depth != floor_to_interval(
                total_match_tokens, self.mamba_checkpoint_interval
            ):
                best_value_len = 0
                last_node = self.root_node

        # Calculate the branching point. It is defined as the last aligned position that
        # does not have a mamba value. With --mamba-checkpoint-interval the
        # alignment grid is the (coarser) checkpoint interval, so
        # re-established checkpoints stay on the deterministic grid.
        if len(value) > best_value_len:
            branch_grid = self.mamba_checkpoint_interval or self.mamba_cache_chunk_size
            matched_tokens = sum(len(v) for v in value)
            chunk_aligned_seqlen = (matched_tokens // branch_grid) * branch_grid
            mamba_branching_seqlen = (
                chunk_aligned_seqlen if chunk_aligned_seqlen > 0 else None
            )
            # #743 INSTRUMENT: this branch IS the truncation. `value` is what
            # the radix matched; `value[:best_value_len]` is what a surviving
            # mamba state can back. Until now the difference was invisible,
            # so a cache doing its job behind a slot pool that is too small
            # read in the logs exactly like a cache that never had the
            # prefix. Costs nothing on the healthy path: the whole block is
            # already conditional on a truncation having happened.
            from sglang.srt.mem_cache.mamba_slot_observer import (
                clock,
                emit_lines,
                observer_of,
            )

            emit_lines(
                logger,
                observer_of(self).note_truncation(
                    now=clock(),
                    rid=getattr(req, "rid", None),
                    matched_tokens=matched_tokens,
                    usable_tokens=sum(len(v) for v in value[:best_value_len]),
                    node_id=getattr(last_node, "id", None),
                    lineage="device",
                ),
            )
        else:
            mamba_branching_seqlen = None

        # Defer COW to forward stream: record source index, allocate destination
        if cow_mamba and last_node.mamba_value is not None:
            if req.mamba_pool_idx is None:
                dst_index = self.req_to_token_pool.mamba_allocator.alloc(1)
                if dst_index is None:
                    self.inc_lock_ref(last_node)
                    self.evict(EvictParams(num_tokens=0, mamba_num=1))
                    dst_index = self.req_to_token_pool.mamba_allocator.alloc(1)
                    self.dec_lock_ref(last_node)
                else:
                    # #639b: sibling of `_alloc_mamba_slot`'s pin. This rank's
                    # COW slot came out of the pool while a peer had to
                    # tombstone a node for its own; match that tombstone, or
                    # the replicas disagree about which nodes carry mamba
                    # state. Locked as above so eviction cannot reclaim the
                    # node this request resumes from.
                    from sglang.srt.mem_cache.common import peer_needs_mamba_evict

                    if peer_needs_mamba_evict(self):
                        self.inc_lock_ref(last_node)
                        self.evict(EvictParams(num_tokens=0, mamba_num=1))
                        self.dec_lock_ref(last_node)
                if dst_index is None:
                    # REQUIRED-allocation path: this slot would hold the
                    # request's OWN resumed state, so "skip caching" is not an
                    # option. Degrade to a full cache MISS instead of killing the
                    # scheduler: report a zero-length match so the request
                    # re-prefills from scratch and gets its mamba slot through
                    # the normal admission path, which already defers a request
                    # when `rem_mamba_slots` is exhausted. Reusing the KV prefix
                    # without the matching mamba state would be silently wrong,
                    # so the full-KV match is dropped as well.
                    self._log_mamba_slot_starvation("mamba (prefix-resume COW)")
                    return MatchResult(
                        device_indices=torch.empty(
                            (0,), dtype=torch.int64, device=self.device
                        ),
                        last_device_node=self.root_node,
                        last_host_node=self.root_node,
                        best_match_node=self.root_node,
                        mamba_branching_seqlen=None,
                    )
                req.mamba_pool_idx = dst_index[0]
            req.mamba_cow_src_index = last_node.mamba_value
            req.mamba_needs_clear = False

        if self.ckpt_debug:
            # Per-request resume attribution (see debug env): with the
            # interval set, resume_tokens MUST be an interval multiple.
            full_match_tokens = sum(len(v) for v in value)
            resume_tokens = sum(len(v) for v in value[:best_value_len])
            slot = (
                int(last_node.mamba_value[0].item())
                if last_node.mamba_value is not None
                else None
            )
            logger.info(
                "mamba-ckpt match: rid=%s full_match=%d resume=%d node=%s "
                "slot=%s branching=%s strict=%s",
                getattr(req, "rid", None),
                full_match_tokens,
                resume_tokens,
                last_node.id,
                slot,
                mamba_branching_seqlen,
                self.mamba_ckpt_strict_resume,
            )
            if not is_on_interval(resume_tokens, self.mamba_checkpoint_interval):
                logger.error(
                    "mamba-ckpt match: OFF-GRID RESUME %d (interval %s), rid=%s",
                    resume_tokens,
                    self.mamba_checkpoint_interval,
                    getattr(req, "rid", None),
                )

        value = value[:best_value_len]
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
            mamba_branching_seqlen=mamba_branching_seqlen,
        )

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.mamba_value = None  # mamba cache can not be split
        new_node.full_lock_ref = child.full_lock_ref
        new_node.mamba_lock_ref = 0
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()

        # child time should be later than parent's time for mamba tombstone
        child.last_access_time = get_last_access_time()

        self.full_lru_list.remove_node(child)
        if child.mamba_value is not None:
            self.mamba_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        # insert the new node and child into the lru lists, insert
        # parent first so that parent is after child in the lru list
        self.full_lru_list.insert_mru(new_node)
        self.full_lru_list.insert_mru(child)
        if child.mamba_value is not None:
            self.mamba_lru_list.insert_mru(child)
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        mamba_value,
        chunked: bool = False,
        prev_prefix_len: int = 0,
        force_host_write_through: bool = False,
    ) -> Tuple[int, bool]:
        # ``force_host_write_through`` is inert here (no host tier); the
        # hierarchical subclass consumes it.
        # Update the last access time from root to leaf, so that
        # mamba will tombstone the node closer to root first
        assert mamba_value is not None, "Mamba value should not be None here."
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)
        if len(key) == 0:
            return 0, True

        child_key = key.child_key(self.page_size)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            if node.mamba_value is not None:
                self.mamba_lru_list.reset_node_mru(node)
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prev_prefix_len < total_prefix_length + prefix_len:
                start = max(0, prev_prefix_len - total_prefix_length)
                self.token_to_kv_pool_allocator.free(value[start:prefix_len])

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = key.child_key(self.page_size)

        mamba_value_exist = False
        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            new_node.mamba_value = mamba_value
            self.full_lru_list.insert_mru(new_node)
            self.mamba_lru_list.insert_mru(new_node)
            node.children[child_key] = new_node
            self.full_evictable_size_ += len(value)
            self.mamba_evictable_size_ += len(mamba_value)
            self._record_store_event(new_node)
        elif node.mamba_value is None:  # add for mamba tombstone
            node.mamba_value = mamba_value
            self.full_lru_list.reset_node_mru(node)
            self.mamba_lru_list.insert_mru(node)
            self.mamba_evictable_size_ += len(mamba_value)
            node.last_access_time = get_last_access_time()
        else:  # mamba value already exists
            mamba_value_exist = True
            self.full_lru_list.reset_node_mru(node)
            self.mamba_lru_list.reset_node_mru(node)
            node.last_access_time = get_last_access_time()

        return total_prefix_length, mamba_value_exist

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0
        while node.parent.mamba_value is None and len(node.parent.children) == 0:
            # root node is not evictable
            if node.parent == self.root_node:
                break
            # if locked, means node is in use, skip
            if node.parent.full_lock_ref > 0:
                break
            # delete tombstone node evicts full tokens
            full_num_evicted += self._free_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _free_tombstone_leaf(self, node: TreeNode) -> int:
        """Free one UNLOCKED, childless mamba tombstone. Returns tokens freed.

        The single place a tombstone leaf is paid for, shared by the cleanup
        walk above and by the ``evict_full`` frontier (#681), so both routes
        keep ``full_evictable_size_`` and the LRU list in step by construction.
        """
        assert (
            node.mamba_value is None
        ), f"not a tombstone, {node.id=}, {len(node.mamba_value)=}"
        assert len(node.children) == 0, f"tombstone is not a leaf, {node.id=}"
        assert (
            node.full_lock_ref == 0
        ), f"tombstone is locked, {node.id=}, {node.full_lock_ref=}"
        assert (
            node.mamba_lock_ref == 0
        ), f"tombstone mamba_lock_ref should always be 0, {node.full_lock_ref=}, {node.mamba_lock_ref=}, {node.id=}"
        self._record_remove_event(node)
        self.token_to_kv_pool_allocator.free(node.value)
        freed = len(node.value)
        self.full_lru_list.remove_node(node)
        self._delete_tombstone_leaf(node)
        return freed

    def _delete_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is not None
        ), f"Invariant violated: leaf node is a tombstone, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)
        self.mamba_evictable_size_ -= len(node.mamba_value)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        self.mamba_evictable_size_ -= len(node.mamba_value)
        node.mamba_value = None

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is None
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if cur_node.mamba_value is not None:
                ret_list.append(cur_node)
            stack.extend(cur_node.children.values())

        return ret_list

    def _collect_all_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            ret_list.append(cur_node)
            stack.extend(cur_node.children.values())
        return ret_list

    def _print_helper(self, node: TreeNode, indent: int, sink=None) -> None:
        """Render the tree per node into ``sink`` (default: the logger).

        #695: ``sink`` replaces the bare ``print``. The child-key invariant is
        no longer asserted HERE -- this function is reached from the OOM path,
        and an assertion raised while explaining an out-of-memory error
        replaces that error with a structural one. ``_shape_summary`` counts
        the same mismatch and reports it without raising.
        """
        out = sink if sink is not None else (lambda line: logger.error("%s", line))
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            out(
                " " * current_indent
                + f"[{current_node.id}] {len(current_node.key)} "
                + f"fr={current_node.full_lock_ref} "
                + f"mr={current_node.mamba_lock_ref} "
                + f"fll={self.full_lru_list.in_list(current_node)} "
                + f"mll={self.mamba_lru_list.in_list(current_node)} "
                + f"mv={current_node.mamba_value}"
            )
            for _key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

    def _total_size_helper(self) -> Tuple[int, int]:
        total_size = 0
        total_mamba_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            if current_node.mamba_value is not None:
                total_mamba_size += len(current_node.mamba_value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size, total_mamba_size
