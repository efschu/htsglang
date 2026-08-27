from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedTreeNode,
    )


class FullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def __init__(self, cache, params):
        super().__init__(cache, params)
        # HiCache state: set to host KV pool when HiCache enabled
        self._full_kv_pool_host = None

    def _free_full(self, values) -> None:
        """Give rows back to the CURRENTLY BOUND allocator, resolved per access.

        #941: THIS WAS A BOUND METHOD CAPTURED IN ``__init__``::

            allocator = cache.token_to_kv_pool_allocator
            self._free_full = allocator.free

        and a bound method carries its instance. ``hicache_phase_binding``
        states the law this broke -- "THREE READERS, AND WHY A PARTIAL REBIND IS
        WORSE THAN NONE ... If one moves and another does not, the readers
        disagree about which pool a row id names, and the disagreement is
        invisible: every call still succeeds, against different memory". Its
        ``_stamp`` moves ``token_to_kv_pool_allocator`` on the scheduler, the
        tree and the controller; it could not reach this capture, and
        ``coherence_check`` could not see it either, because that check compares
        ``hicache_binding_generation`` over the three NAMED readers and this
        component is a fourth. Green indicator, stale free path.

        MEASURED (window 2k, boot_2k_dd0e3bc224_0827_1237.log, PP0). At a
        ``tp_to_pp`` drop::

            at-arm      free=470650 cached=267 unaccounted=332
            RESIDENTS RELEASED ... the prefix tree dropped returning 267 row(s)
            pre-cutover free=470650 cached=0   unaccounted=599

        267 rows leave the tree, the bound allocator's free count does not move,
        and ``unaccounted`` rises by exactly 267 -- five cycles, five exact
        matches. The rows are not destroyed: they land on the OTHER phase's free
        list as DUPLICATES of ids it already calls free, which is why that
        allocator's ``available_size`` (a raw length) drifts above its own
        enumerated free set by the same running total. A leak on one side and a
        second writer waiting on the other.

        ONLY ONE OF THE TWO DROPS COULD SHOW IT. The seam retracts and drops
        BEFORE the cutover rebinds, so at a ``pp_to_tp`` drop the live binding is
        still the phase that minted the rows and the stale capture agrees by
        accident. At a ``tp_to_pp`` drop it does not, which is why the loss is
        once per cutover PAIR and is sized by the TP phase's allocation.

        RESOLVED PER ACCESS, NOT RE-STAMPED. Re-stamping would add a fifth thing
        to keep in step; reading the binding at the moment of use cannot go
        stale, and it is the rule this codebase already applies to rebindable
        pools elsewhere ("Candidates are resolved PER ACCESS, never held -- a
        construction reference to a rebindable pool is the #927 class",
        phase_flip_runtime.py). ``tree_components`` is fixed at construction, so
        the SWA branch below is a constant test, not a second binding.

        THIS CHANGES WHERE A FREE IS ADDRESSED, NEVER WHETHER ONE HAPPENS. No
        row is freed here that was not already being freed by this same call at
        this same moment, so there is no new release to pair with a later drop
        and no double-free to build.
        """
        allocator = self.cache.token_to_kv_pool_allocator
        # When SWA is present, only free full-attention KV here;
        # SWA KV will be freed by cascade via SWAComponent.evict_component.
        if ComponentType.SWA in self.cache.tree_components:
            allocator = allocator.full_attn_allocator
        allocator.free(values)

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode, int], bool]:
        # `depth` (#747) is unused: full KV validity does not depend on the
        # node's absolute position.
        if match_device_only:
            return lambda node, depth: (
                node.component_data[self.component_type].value is not None
            )

        # HiCache: evicted + backuped nodes are valid match boundaries.
        return lambda node, depth: (
            node.component_data[self.component_type].value is not None or node.backuped
        )

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        # Compute Full KV host hit length: walk from last_host_node up to
        # last_device_node, summing host_value lengths of evicted nodes.
        ct = self.component_type
        kv_host_hit = 0
        node = result.best_match_node
        root_node = self.cache.root_node
        while node is not result.last_device_node and node is not root_node:
            full_host = node.component_data[ct].host_value
            if full_host is not None:
                kv_host_hit += len(full_host)
            node = node.parent
        if kv_host_hit > 0:
            return result._replace(
                host_hit_length=max(result.host_hit_length, kv_host_hit)
            )
        return result

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        ct = self.component_type
        new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref
        child_cd = child.component_data[ct]
        split_len = len(new_parent.key)
        if child_cd.value is not None:
            new_parent.component_data[ct].value = child_cd.value[:split_len].clone()
            child_cd.value = child_cd.value[split_len:].clone()
        if child_cd.host_value is not None:
            new_parent.component_data[ct].host_value = child_cd.host_value[
                :split_len
            ].clone()
            child_cd.host_value = child_cd.host_value[split_len:].clone()

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            self._free_full(cd.value)
            freed = len(cd.value)
            self.cache.component_evictable_size_[self.component_type] -= freed
            # NOTE: cd.value = None is deferred to _cascade_evict (Full as trigger)
            # because SWA's free_swa still needs to read Full.value.
            # cd.value = None

        # Host layer
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            if self._full_kv_pool_host is not None:
                self._full_kv_pool_host.free(cd.host_value)
            cd.host_value = None
        return freed, host_freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 2

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        request = params.num_tokens
        heap = [
            (self.cache.eviction_strategy.get_priority(n), n)
            for n in self.cache.evictable_device_leaves
        ]
        heapq.heapify(heap)
        ct = self.component_type
        while tracker[ct] < request and heap:
            _, x = heapq.heappop(heap)
            if x not in self.cache.evictable_device_leaves:
                continue
            self.cache._evict_device_leaf(x, tracker)
            if x.parent is not None and x.parent in self.cache.evictable_device_leaves:
                heapq.heappush(
                    heap,
                    (self.cache.eviction_strategy.get_priority(x.parent), x.parent),
                )

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict host leaves to free KV host pool space."""
        heap = [
            (self.cache.eviction_strategy.get_priority(n), n)
            for n in self.cache.evictable_host_leaves
        ]
        heapq.heapify(heap)
        ct = self.component_type
        while tracker[ct] < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x not in self.cache.evictable_host_leaves:
                continue
            self.cache._evict_host_leaf(x, tracker)
            if x.parent is not None and x.parent in self.cache.evictable_host_leaves:
                heapq.heappush(
                    heap,
                    (self.cache.eviction_strategy.get_priority(x.parent), x.parent),
                )

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type

        # Only the last host node needs to be protected.
        if lock_host:
            cd = node.component_data[ct]
            if cd.host_value is None:
                return result
            cd.host_lock_ref += 1
            self.cache._update_evictable_leaf_sets(node)
            return result

        root = self.cache.root_node
        cur = node

        # Skip the bottom evicted segment
        while cur is not root and cur.component_data[ct].value is None:
            result.skip_lock_node_ids.setdefault(ct, set()).add(cur.id)
            cur = cur.parent

        # Lock the device-on segment up to root
        delta = 0
        while cur is not root:
            cd = cur.component_data[ct]
            assert (
                cd.value is not None
            ), f"FULL invariant broken: evicted ancestor {cur.id} above device-on segment"
            if cd.lock_ref == 0:
                key_len = len(cd.value)
                self.cache.component_evictable_size_[ct] -= key_len
                self.cache.component_protected_size_[ct] += key_len
                delta += key_len
            cd.lock_ref += 1
            self.cache.evictable_device_leaves.discard(cur)
            cur = cur.parent
        result.delta = delta
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        if lock_host:
            cd = node.component_data[ct]
            if cd.host_value is None or cd.host_lock_ref == 0:
                return
            cd.host_lock_ref -= 1
            self.cache._update_evictable_leaf_sets(node)
            return

        root = self.cache.root_node
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        cur = node
        hops = 0
        while cur != root:
            # #827: THE WALK'S ONLY EXIT IS REACHING ROOT.
            #
            # A node whose ancestor chain is broken -- detached, evicted, or
            # left behind by a rebuilt root -- walks PAST the top, `cur`
            # becomes None, and the next line dereferences it. Measured
            # 2026-08-23 08:55:45 on all three ranks:
            #   AttributeError: 'NoneType' object has no attribute 'id'
            # which names the wrong thing: the fault is a broken parent chain
            # several frames earlier, not this attribute.
            #
            # NAMED, NOT SWALLOWED. A `break` here would leave
            # `component_protected_size_` overcounted for the life of the
            # process -- a loud fault traded for a slow leak. This raises with
            # the node, the component and the distance walked, so the next
            # occurrence is diagnosable from the message alone.
            if cur is None:
                raise RuntimeError(
                    f"release_component_lock: parent chain broke after {hops} "
                    f"hops from node {node.id} (component {ct.name}) without "
                    f"reaching root {getattr(root, 'id', None)}. An ancestor "
                    f"was detached or evicted while this lock was held."
                )
            hops += 1
            if cur.id in skip_lock_node_ids:
                cur = cur.parent
                continue
            cd = cur.component_data[ct]
            assert cd.value is not None
            assert cd.lock_ref > 0

            if cd.lock_ref == 1:
                key_len = len(cd.value)
                self.cache.component_evictable_size_[ct] += key_len
                self.cache.component_protected_size_[ct] -= key_len
            cd.lock_ref -= 1
            if cd.lock_ref == 0:
                self.cache._update_evictable_leaf_sets(cur)
            cur = cur.parent

    # ---- HiCache Hooks ----

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        req: Optional[Req] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            # Full KV backup is handled by the main flow
            # (write_backup → cache_controller.write on host_value directly).
            # No extra PoolTransfer needed.
            return None

        if phase == CacheTransferPhase.LOAD_BACK:
            # `node` is best_match_node. FULL device evict only from leaves,
            # so once we hit a device-on node, everything above is also device-on
            backed_up: list[torch.Tensor] = []
            nodes: list = []
            cur = node
            while cur.evicted:
                cd = cur.component_data[ct]
                assert cd.host_value is not None
                backed_up.append(cd.host_value)
                nodes.append(cur)
                cur = cur.parent
            backed_up.reverse()
            nodes.reverse()
            return [
                PoolTransfer(
                    name=PoolName.KV,
                    host_indices=(
                        torch.cat(backed_up)
                        if backed_up
                        else torch.empty((0,), dtype=torch.int64, device="cpu")
                    ),
                    device_indices=None,
                    nodes_to_load=nodes,
                )
            ]

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                node.component_data[ct].host_value = transfers[0].host_indices.clone()

        elif phase == CacheTransferPhase.LOAD_BACK:
            if not transfers or transfers[0].device_indices is None:
                self.cache._update_evictable_leaf_sets(node)
                return

            xfer = transfers[0]
            device_indices = xfer.device_indices
            offset = 0
            for n in xfer.nodes_to_load or []:
                cd = n.component_data[ct]
                n_len = len(cd.host_value)
                cd.value = device_indices[offset : offset + n_len].clone()
                offset += n_len
                # Full uses leaf sets, not LRU
                self.cache.component_evictable_size_[ct] += n_len
                self.cache._update_evictable_leaf_sets(n)

            self.cache._update_evictable_leaf_sets(node)
