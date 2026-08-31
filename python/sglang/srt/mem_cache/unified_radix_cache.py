from __future__ import annotations

import json
import logging
import sys
import threading
import time
from array import array
from collections import Counter, defaultdict
from functools import partial
from queue import Empty, Queue
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
)

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.distributed.utils import uneven_dcp_active
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
    requests_forced_host_write_through,
)
from sglang.srt.mem_cache.common import (
    note_uniform_host_admitted,
    uniform_host_avail_for_backup,
    uniform_host_floor_active,
)
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.hicache_collective import (
    COLLECTIVE_POLL_MAX_S,
    COLLECTIVE_POLL_MIN_S,
    COLLECTIVE_POLL_SPINS,
    HiCacheCollectiveDesyncError,
    HiCacheCollectiveError,
    HiCacheCollectiveTimeoutError,
    bounded_recv,
    bounded_wait,
    collective_rank_desc,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.match_refusal_census import (
    emit as census_emit,
    format_prefetch_gate as _format_prefetch_gate,
    new_match_census,
    note_prefetch_gate as _note_prefetch_gate,
    prefetch_gate_due as _prefetch_gate_due,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaLoadBackUnservable,
)
from sglang.srt.mem_cache.unified_cache_components import (
    _NUM_COMPONENT_TYPES,
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentData,
    ComponentType,
    EvictLayer,
    FullComponent,
    LRURefreshPhase,
    MambaComponent,
    SWAComponent,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    get_eviction_strategy,
    split_node_hash_value,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.session.streaming_session import StreamingSession

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        PrefetchOperation,
    )
    from sglang.srt.server_args import ServerArgs


T = TypeVar("T")


# The HiCacheCollective* error types and the bounded-wait mechanism used to be
# defined here. They moved to hicache_collective so that HiRadixCache shares ONE
# implementation with this class instead of a second, drifting copy (#630 was
# exactly that drift). They stay importable from this module because that is
# their established import surface -- this tuple is what keeps that re-export
# explicit rather than an import that looks unused and invites deletion.
_COLLECTIVE_REEXPORTS = (
    HiCacheCollectiveError,
    HiCacheCollectiveDesyncError,
    HiCacheCollectiveTimeoutError,
)


# Rank-invariant slot layout for the sidecar-pool entries of a control
# collective. The reduced vector's LENGTH AND ORDER must never be derived from
# rank-local state (``cc.extra_host_mem_release_queues``, ``comp_xfers``): those
# are built from this rank's host-pool entries, which are asymmetric under
# uneven DCP, and gloo does not reject an all_reduce whose numel differs across
# ranks -- it wedges. Every rank therefore reduces the full PoolName universe;
# a pool this rank does not own contributes 0, the MIN identity for "drain /
# claim nothing", so a set divergence degrades to a no-op instead of a hang.
_POOL_SLOT_ORDER: tuple[PoolName, ...] = tuple(PoolName)
_POOL_SLOT_COUNT: int = len(_POOL_SLOT_ORDER)

#: #939: how many times one request's storage prefetch may be re-issued across
#: cutovers before the situation is reported as a recompute. A REPORT, not a
#: gate: refusing the re-issue would be a rank-local decision taken inside the
#: #580 participation region, which is the one thing this whole path may not
#: do. The line exists so a cutover cadence faster than a fetch completes is
#: named in the log instead of showing up only as an unexplained `cached=0`.
_MAX_PREFETCH_REISSUES: int = 3

# Poll schedule for _wait_bounded. The spin window covers the latency of a
# healthy CPU collective between local ranks, so the sleep path is only ever
# reached once something is actually wrong.
#: Self-identifying head of the prefetch participation vote (#580). Any value
#: works as long as it is the same on every rank and not a plausible payload
#: value; it is reduced as a (tag, -tag) pair so one MIN yields min and max.
_PREFETCH_VOTE_TAG = 580

_COLLECTIVE_POLL_SPINS = COLLECTIVE_POLL_SPINS
_COLLECTIVE_POLL_MIN_S = COLLECTIVE_POLL_MIN_S
_COLLECTIVE_POLL_MAX_S = COLLECTIVE_POLL_MAX_S


def _pool_slot(pool_name, offset: int) -> int:
    """Index of ``pool_name`` inside a reduced vector starting at ``offset``."""
    for i, known in enumerate(_POOL_SLOT_ORDER):
        if known == pool_name:
            return offset + i
    raise HiCacheCollectiveError(
        f"host pool {pool_name!r} is not a member of PoolName, so it has no "
        "rank-invariant slot in the HiCache control collective. Add it to "
        "PoolName instead of keying the collective off rank-local pool names."
    )


class UnifiedTreeNode:
    counter = 0

    def __init__(self, tree_components: tuple[ComponentType, ...], priority: int = 0):
        self.children = defaultdict(partial(UnifiedTreeNode, tree_components))
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.tree_components = tree_components
        # list indexed by ComponentType (int enum 0..N-1)
        self.component_data: list[ComponentData] = [
            ComponentData() for _ in range(_NUM_COMPONENT_TYPES)
        ]
        self.last_access_time = get_and_increase_time_counter()
        self.creation_time = get_and_increase_time_counter()
        self.hash_value = None
        self.hit_count = 0
        self.priority = priority
        self.lru_prev: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.lru_next: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.id = UnifiedTreeNode.counter
        UnifiedTreeNode.counter += 1
        self.write_through_pending_id: Optional[int] = None

    def component(self, component_type: ComponentType) -> ComponentData:
        return self.component_data[component_type]

    @property
    def backuped(self) -> bool:
        """Tree-level: Full KV present on host."""
        return self.component_data[ComponentType.FULL].host_value is not None

    @property
    def evicted(self) -> bool:
        """Tree-level: Full KV not on device (non-root with value=None)."""
        return (
            self.parent is not None
            and self.component_data[ComponentType.FULL].value is None
        )

    def __lt__(self, other: UnifiedTreeNode):
        return self.last_access_time < other.last_access_time

    def get_last_hash_value(self) -> Optional[str]:
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: UnifiedTreeNode) -> list[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value


class UnifiedLRUList:
    def __init__(
        self,
        component_type: ComponentType,
        tree_components: tuple[ComponentType, ...],
        use_host_ptr: bool = False,
    ):
        self.component_type = component_type
        # Pointer slot: host LRU uses offset slots so device/host pointers
        # never collide on the same node.
        self._pt: int = component_type + (_NUM_COMPONENT_TYPES if use_host_ptr else 0)
        self.head = UnifiedTreeNode(tree_components)
        self.tail = UnifiedTreeNode(tree_components)
        self.head.lru_next[self._pt] = self.tail
        self.tail.lru_prev[self._pt] = self.head
        self.cache: dict[int, UnifiedTreeNode] = {}

    def _add_node_after(self, prev_node: UnifiedTreeNode, new_node: UnifiedTreeNode):
        pt = self._pt
        new_node.lru_prev[pt] = prev_node
        new_node.lru_next[pt] = prev_node.lru_next[pt]
        prev_node.lru_next[pt].lru_prev[pt] = new_node
        prev_node.lru_next[pt] = new_node

    def _add_node(self, node: UnifiedTreeNode):
        self._add_node_after(self.head, node)

    def _remove_node(self, node: UnifiedTreeNode):
        pt = self._pt
        node.lru_prev[pt].lru_next[pt] = node.lru_next[pt]
        node.lru_next[pt].lru_prev[pt] = node.lru_prev[pt]
        # Clear self pointers to break reference cycles among evicted nodes.
        node.lru_prev[pt] = None
        node.lru_next[pt] = None

    def insert_mru(self, node: UnifiedTreeNode):
        assert node.id not in self.cache
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        del self.cache[node.id]
        self._remove_node(node)

    def reset_node_mru(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        should_include,
    ):
        prev_node = self.head
        while node != root_node:
            if should_include(node):
                assert node.id in self.cache
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def reset_node_and_window_ancestors_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        window_size: int,
        should_include,
    ):
        prev_node = self.head
        accumulated = 0
        while node != root_node and accumulated < window_size:
            if should_include(node):
                assert node.id in self.cache
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            accumulated += len(node.key)
            node = node.parent

    def in_list(self, node: Optional[UnifiedTreeNode]):
        return node is not None and node.id in self.cache

    def get_prev_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        while x.component_data[ct].lock_ref > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        while x.component_data[ct].lock_ref > 0 or len(x.children) > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_prev_no_host_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        """Host-LRU walker: skip nodes whose component host_lock_ref > 0."""
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        while x.component_data[ct].host_lock_ref > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_lru_no_lock(self):
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self):
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_lru_no_host_lock(self):
        return self.get_prev_no_host_lock(self.tail, check_id=False)


COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
    ComponentType.MAMBA: MambaComponent,
    ComponentType.SWA: SWAComponent,
}

logger = logging.getLogger(__name__)


class _OngoingWriteThrough(NamedTuple):
    """Tracks an in-flight D→H write-through operation."""

    node: UnifiedTreeNode
    lock_params: Optional[DecLockRefParams]
    publish_nodes: list[UnifiedTreeNode]


class _OngoingLoadBack(NamedTuple):
    """Tracks an in-flight H→D load-back operation."""

    node: UnifiedTreeNode
    lock_params: DecLockRefParams
    host_lock_params: DecLockRefParams


class _OngoingPrefetch(NamedTuple):
    """Tracks an in-flight storage→host prefetch operation."""

    anchor_node: UnifiedTreeNode
    prefetch_key: RadixKey
    host_indices: torch.Tensor
    operation: PrefetchOperation
    anchor_lock_params: DecLockRefParams
    comp_xfers: dict[ComponentType, list[PoolTransfer]]


class UnifiedRadixCache(KVCacheEventMixin, BasePrefixCache):
    def __init__(
        self,
        params: CacheInitParams,
    ):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.is_eagle = params.is_eagle
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.kv_event_queue = []
        self.eviction_policy = params.eviction_policy.lower()
        self.eviction_strategy = get_eviction_strategy(self.eviction_policy)

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()
        self._enable_metrics_flag = params.enable_metrics
        self.enable_storage_metrics = False
        self.storage_metrics_collector: Optional[StorageMetricsCollector] = None
        self.extra_metric_labels = None

        assert params.tree_components is not None
        self.tree_components = tuple(params.tree_components)
        component_registry = COMPONENT_REGISTRY
        if params.component_registry_override:
            component_registry = {
                **COMPONENT_REGISTRY,
                **params.component_registry_override,
            }
        self.components: dict[ComponentType, TreeComponent] = {
            ct: component_registry[ct](self, params) for ct in self.tree_components
        }
        self._components_tuple: tuple[TreeComponent, ...] = tuple(
            self.components.values()
        )
        self.sidecar_pool_specs: list[SidecarPoolSpec] = []

        # Streaming session: embedded StreamingSession with self as inner.
        # Always on -- zero overhead when no streaming session is open (the
        # try_* entries short-circuit on non-streaming reqs / real TreeNodes).
        # Dispatch methods below pre-check conditions so the session's
        # internal fall-through to self.inner.xxx never fires -- no recursion.
        self.session = StreamingSession(inner=self)

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.pp_group = params.pp_cache_group
        self.tp_world_size = (
            1
            if self.tp_group is None
            else torch.distributed.get_world_size(group=self.tp_group)
        )
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.work_list: list[torch.distributed.Work] = []
        # Deadline for every cross-rank control collective issued from this
        # cache. See _wait_bounded: without it a dead peer parks this rank in
        # all_reduce until the two-hour gloo group timeout expires.
        self.collective_timeout_s = envs.SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S.get()

        # HiCache D↔H defaults (overridden by init_hicache)
        self.cache_controller: Optional[HybridCacheController] = None
        self.write_through_threshold = 256
        self.prefetch_stop_policy = "best_effort"
        self.prefetch_threshold = 256
        self.prefetch_timeout_base = 1.0
        self.prefetch_timeout_per_page = 0.25
        self.hicache_storage_pass_prefix_keys = False

        self.reset()
        # #581/#773: give the request pool a handle back, so a REQUIRED mamba
        # allocation can evict cached checkpoints before failing. `MambaRadixCache`
        # has always done this at construction; this lineage never did, which
        # left `_alloc_mamba_slots_or_evict`'s evict-then-retry (and #639b's
        # rank-parity tombstone branch) unreachable on every hybrid-SSM boot
        # routed here -- i.e. the pool reported exhaustion while cached,
        # evictable checkpoints were sitting in the tree.
        if hasattr(self.req_to_token_pool, "bind_tree_cache"):
            self.req_to_token_pool.bind_tree_cache(self)
        logger.info(f"Init Unified RadixTree with components {self.tree_components}")
        self._log_mamba_floor_posture()

    def _log_mamba_floor_posture(self) -> None:
        """State the floor, the pool and the retention budget once, at boot.

        #773: none of these three numbers was observable on a healthy boot.
        `_validate_max_mamba_cache_size` returns SILENTLY for any pool at or
        above the floor, so the only way the floor ever reached a log was by
        refusing a boot. That is why a pool sitting exactly ON the floor --
        i.e. with no room at all for cache retention -- looked identical to a
        comfortable one, and why the reduction taken by a lineage that does
        not implement it went unnoticed.

        A budget of 0 is not an error: it is the honest statement that this
        pool is fully committed to the running set. It does mean every mamba
        write-through backup will be declined, so it is worth saying out loud
        rather than leaving to be inferred from a silent absence of host
        anchors.
        """
        if ComponentType.MAMBA not in self.tree_components:
            return
        mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
        if mamba_pool is None:
            return
        try:
            from sglang.srt.mem_cache.mamba_pool_floor import (
                describe_mamba_floor,
                mamba_hard_floor,
            )
            from sglang.srt.runtime_context import get_server_args

            server_args = get_server_args()
            mrr = server_args.max_running_requests or 1
            floor = mamba_hard_floor(server_args, mrr)
            budget = self._mamba_pin_budget
            logger.info(
                "MAMBA-FLOOR pool=%d floor=%d retention_budget=%d (%s)%s",
                mamba_pool.size,
                floor,
                budget,
                describe_mamba_floor(server_args, mrr),
                (
                    " -- the pool is fully committed to the running set, so"
                    " every mamba host backup will be declined; raise"
                    " --max-mamba-cache-size to buy cache retention"
                    if budget <= 0
                    else ""
                ),
            )
        except Exception:  # noqa: BLE001 -- an instrument never breaks a boot
            logger.debug("MAMBA-FLOOR posture unavailable", exc_info=True)

    def _wait_bounded(self, work, label: str) -> None:
        """Wait for ``work`` with a deadline, or raise a named error.

        These control collectives run on the gloo ``cpu_group``s, whose default
        timeout is two hours and which nothing on this path shortens. A peer
        that dies without closing its socket (an OOM rank stuck in teardown)
        therefore parks the survivor indefinitely. Polling ``is_completed()``
        against a deadline is backend-agnostic and turns that wedge into an
        exception the scheduler's own handler reports and dies on.

        The healthy case costs nothing measurable: a CPU all_reduce of a few
        ints between local ranks completes inside the initial spin window, so
        no sleep is ever reached.

        Thin wrapper over the shared mechanism in ``hicache_collective`` -- see
        that module for the poll schedule and the rationale.
        """
        bounded_wait(
            work,
            label,
            self.collective_timeout_s,
            collective_rank_desc(self),
        )

    def _attn_reduce_world(self) -> int:
        """#1028: how many ranks `_all_reduce_attn_groups` ACTUALLY reduces over.

        A REDUCE THAT COVERS ONE RANK IS NOT AN AGREEMENT, AND TWO MARKERS
        PRINTED IT AS ONE FOR THE WHOLE CAMPAIGN. `drain_retired_prefetch` and
        `take_agreed_reissue` pack `[digest, -digest]` so that one MIN reduce
        yields both min and max, and then read `min == max` as "every rank
        agrees". Under `--tp-size 1 --pp-size 3` every group this helper
        touches has world size 1, the reduce is a no-op, and `min == max`
        holds BY ARITHMETIC on every round. So

            #943 STALE-REFUSAL VERDICT AGREES ACROSS RANKS
            #943b PREFETCH RE-ISSUED: req=... agreed by every rank

        were affirmative rank-agreement claims covering nobody, on every boot
        of this strand -- and §AG quoted the first of them as evidence. That is
        the indicator law exactly: a counter is a finding only once you have
        checked THAT it measures what it claims.

        Returned rather than asserted, so the two call sites can say which
        world they agreed in instead of implying one.
        """
        world = 1
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None:
                try:
                    world = max(world, torch.distributed.get_world_size(group=group))
                except Exception:  # noqa: BLE001 - a census may abstain
                    pass
        if world == 1:
            world = max(world, int(getattr(self, "tp_world_size", 1) or 1))
        return world

    def _all_reduce_attn_groups(self, tensor: torch.Tensor, op, label: str = "hicache"):
        reduced = False
        for name, group in (
            ("attn_cp", self.attn_cp_group),
            ("attn_tp", self.attn_tp_group),
        ):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                self._wait_bounded(
                    torch.distributed.all_reduce(
                        tensor, op=op, group=group, async_op=True
                    ),
                    f"{label}/all_reduce/{name}",
                )
                reduced = True
        if not reduced and self.tp_world_size > 1:
            self._wait_bounded(
                torch.distributed.all_reduce(
                    tensor, op=op, group=self.tp_group, async_op=True
                ),
                f"{label}/all_reduce/tp",
            )

    def _hicache_prefetch_symmetric(self) -> bool:
        """Uneven-DCP (a non-uniform token vector installed) makes the per-rank
        HiCache host pools ASYMMETRIC (e.g. 214598 / 103838 / 124606 tokens), so
        the storage-prefetch registration and rate-limit decisions can diverge
        across ranks. A rank that early-returns never registers
        ``ongoing_prefetch[req_id]`` and so SKIPS the ``_all_reduce_attn_groups``
        collectives in can_terminate_prefetch / check_prefetch_progress, while the
        ranks that did register ENTER them -> collective desync -> NCCL deadlock.

        The two symmetrization mechanisms (participation consensus in
        prefetch_from_storage; capacity floor in _symmetrize_prefetch_capacity)
        are gated on this predicate. Stock even-TP HiCache (uniform host pools)
        never trips it -> that path is byte-identical. Uses the same attn/TP
        groups the existing prefetch collectives run on."""
        return (
            getattr(self, "enable_storage", False)
            and self.cache_controller is not None
            and self.tp_world_size > 1
            and uneven_dcp_active()
        )

    def prefetch_participation_is_collective(self) -> bool:
        """True when whether to prefetch is decided by the GROUP, not per rank.

        The scheduler asks before applying its own local prefetch gate: in this
        mode every rank must call ``prefetch_from_storage`` so that every rank
        enters the participation vote, and a rank whose local gate says no
        passes ``locally_eligible=False`` instead of skipping the call (#580).
        Absent on the other tree-cache classes, which stay on the local gate.
        """
        return self._hicache_prefetch_symmetric()

    def _symmetrize_prefetch_capacity(self) -> None:
        """Mechanism (2): derive the speculative-prefetch capacity limit from the
        MIN host-pool size across the DCP/TP ranks.

        Under weighted DCP the host pools differ per rank, so the stock per-rank
        ``int(0.5 * mem_pool_host.size)`` limit makes ``prefetch_rate_limited()``
        answer differently on different ranks -> ranks diverge on whether to even
        start a prefetch, BEFORE the participation consensus in
        prefetch_from_storage is reached (that divergence would desync the
        consensus all-reduce itself). Using the shared MIN makes the rate-limit
        gate trip in lockstep on every rank. Gated so the general (even-TP) path
        keeps its per-rank limit unchanged."""
        if not self._hicache_prefetch_symmetric():
            return
        cc = self.cache_controller
        if getattr(cc, "mem_pool_host", None) is None:
            # The gate above is rank-uniform (config + uneven_dcp_active), so a
            # rank-local early return HERE would leave the other ranks in the
            # all_reduce below with no partner. A HybridCacheController always
            # owns a host pool, so this is a structural break, not a state the
            # collective may be skipped for: name it instead of hanging.
            raise HiCacheCollectiveError(
                "cache controller has no mem_pool_host while prefetch "
                "symmetrization is active; the peer ranks are entering the "
                "capacity all_reduce and this rank cannot."
            )
        local_size = int(cc.mem_pool_host.size)
        size_tensor = torch.tensor([local_size], dtype=torch.long)
        self._all_reduce_attn_groups(
            size_tensor,
            torch.distributed.ReduceOp.MIN,
            label="symmetrize_prefetch_capacity",
        )
        min_size = int(size_tensor.item())
        cc.prefetch_capacity_limit = int(0.5 * min_size)
        logger.info(
            "[uneven-dcp hicache] prefetch_capacity_limit symmetrized to %d "
            "(min host-pool %d across attn groups; local host-pool %d)",
            cc.prefetch_capacity_limit,
            min_size,
            local_size,
        )

    def _barrier_attn_groups(self, label: str = "hicache"):
        waited = False
        for name, group in (
            ("attn_cp", self.attn_cp_group),
            ("attn_tp", self.attn_tp_group),
        ):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                self._wait_bounded(
                    torch.distributed.barrier(group=group, async_op=True),
                    f"{label}/barrier/{name}",
                )
                waited = True
        if not waited and self.tp_world_size > 1:
            self._wait_bounded(
                torch.distributed.barrier(group=self.tp_group, async_op=True),
                f"{label}/barrier/tp",
            )

    def _drain_async_work(self):
        """
        Block until all outstanding async sends are consumed, then clear.

        Called at the start of each event round, so work_list holds the sends
        accumulated since the last round. This bounds it and applies
        backpressure when a downstream PP rank lags. Scheduler thread only.

        The wait is bounded for the same reason every other collective on this
        path is (#630): these are isends on the PP gloo ``cpu_group``, and a
        downstream PP rank that never posts the matching receive parks this
        rank here until the group's two-hour timeout expires. Under PP + disk
        HiCache that shows up as a warmup that never finishes and a health
        endpoint stuck at 503 with nothing in the log.
        """
        for i, work in enumerate(self.work_list):
            self._wait_bounded(work, f"pp_sync/isend[{i}]->pp{self.pp_rank + 1}")
        self.work_list.clear()

    def _all_reduce(
        self,
        data: torch.Tensor,
        tp_reduce_op: torch.distributed.ReduceOp,
        label: str = "hicache",
    ):
        """
        Synchronize data across all TP and PP ranks.

        In particular, "tp_reduce_op" is performed on all TP ranks of the first PP rank,
        and then the result is propagated to all following PP ranks.

        Must be called in the scheduler thread.
        """
        if self.pp_rank == 0:
            self._all_reduce_attn_groups(data, tp_reduce_op, label=label)
        self._pp_sync(data)

    def _pp_sync(self, data: torch.Tensor) -> None:
        """
        Synchronize data across the PP pipeline, where PPn (n>0) will receive PP0's data.
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return
        if self.pp_rank > 0:
            # Bounded via irecv rather than recv: recv has no async form and no
            # timeout, so every PP rank above the first would otherwise block
            # here without a deadline. See hicache_collective.bounded_recv.
            bounded_recv(
                data,
                group=self.pp_group,
                group_src=self.pp_rank - 1,
                tag=P2PTag.HIRADIX_PP_SYNC,
                label=f"pp_sync/recv<-pp{self.pp_rank - 1}",
                timeout_s=self.collective_timeout_s,
                rank_desc=collective_rank_desc(self),
            )
        if self.pp_rank + 1 < self.pp_size:
            copy_of_data = data.clone()
            send_work = torch.distributed.isend(
                copy_of_data,
                group_dst=self.pp_rank + 1,
                group=self.pp_group,
                tag=P2PTag.HIRADIX_PP_SYNC,
            )
            self.work_list.append(send_work)

    def reset(self) -> None:
        self._reset_full()

    def _reset_full(self) -> None:
        """Full reset: destroy entire tree and all state."""
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.root_node.priority = -sys.maxsize
        self.root_node.key = RadixKey(array("q"), None)
        self.root_node.component_data[BASE_COMPONENT_TYPE].value = []
        self.root_node.hash_value = []
        for ct in self.tree_components:
            self.root_node.component_data[ct].lock_ref = 1
        self.component_evictable_size_ = {ct: 0 for ct in self.tree_components}
        self.component_protected_size_ = {ct: 0 for ct in self.tree_components}

        self.lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components) for ct in self.tree_components
        }
        self.session.slots.clear()

        self.evictable_device_leaves: set[UnifiedTreeNode] = set()
        self.evictable_host_leaves: set[UnifiedTreeNode] = set()
        self.host_lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components, use_host_ptr=True)
            for ct in self.tree_components
        }
        self.ongoing_write_through: dict[int, _OngoingWriteThrough] = {}
        self.ongoing_load_back: dict[int, _OngoingLoadBack] = {}
        # #773: resolved lazily -- the pool and the server args are both in
        # place by the first backup, but not necessarily at construction.
        self._mamba_pin_budget_cached: Optional[int] = None
        self._mamba_pin_skipped = 0
        # #1028 chunk publish (see `_inc_hit_count`).
        self._chunk_publish_n = 0
        # #811: lazily resolved for the same reason as the pin budget above.
        self._anchor_ack_release_armed_cached: Optional[bool] = None
        # #841: host-only inserts declined for breaking the contiguous-backup
        # law. Counted rather than silent, so a collapsing storage hit rate is
        # attributable to the law and not to the backend.
        self._host_insert_refused_unbacked_parent = 0
        self.enable_storage = False
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.ongoing_prefetch: dict[str, _OngoingPrefetch] = {}
        # #939: prefetch records displaced by a RE-ISSUE of the same req_id.
        # They are terminated but NOT freed here -- see `_retire_ongoing_prefetch`.
        # TWO release paths, and a new holder like this one needs BOTH or it
        # becomes unfreeable (#966): `drain_retired_prefetch` from the per-round
        # reap, which is collective and lives under the enable_hicache_storage
        # gate, and `_release_retired_prefetch_local` from `detach_storage_backend`,
        # which is what runs once that gate is cleared and the reap is therefore
        # unreachable.
        self._retired_prefetch: list[_OngoingPrefetch] = []
        self._retired_prefetch_attempts: dict[str, int] = {}
        self._retired_prefetch_reaped = 0
        #: #943b: requests whose prefetch #937 refused to publish because its
        #: binding generation had gone stale, and which are therefore owed a
        #: FRESH fetch under the current binding. req_id -> times refused.
        #: Never a span and never an operation: reviving either is the corruption
        #: `StaleStampRewrite` exists to forbid.
        self._reissue_pending: dict[str, int] = {}
        self._reissue_taken = 0
        self._reissue_disagreements = 0
        self._retired_prefetch_recompute = 0
        self.ongoing_backup: dict[int, tuple[UnifiedTreeNode, DecLockRefParams]] = {}
        # #810: built in `init_hicache`, once the controller and the
        # symmetrized prefetch reservation exist. None here and for the whole
        # of `--hicache-host-role retention`, which is the default.
        self.staging_write_ring = None
        self._init_pin_trace()

        if self.cache_controller is not None:
            self.cache_controller.reset()
            self.cache_controller.mem_pool_host.clear()
            self.enable_storage = self.cache_controller.enable_storage

        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
        )
        self._record_all_cleared_event()

    def init_hicache(self, server_args: ServerArgs, params: CacheInitParams) -> None:
        """Initialize HiCache infrastructure."""
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            attach_hybrid_pool_to_unified_cache,
        )

        # Direct IO layout fixup (must happen before pool creation)
        if server_args.hicache_io_backend == "direct":
            if server_args.hicache_mem_layout == "page_first":
                server_args.override(
                    "hicache.mem_layout_force", hicache_mem_layout="page_first_direct"
                )
                logger.warning(
                    "Page first layout is not supported with direct IO backend, "
                    "switching to page first direct layout"
                )

        self.load_cache_event = threading.Event()
        self.sidecar_pool_specs.clear()
        self.extra_metric_labels = server_args.extra_metric_labels

        # Parse storage config once, share with assembler and tree
        storage_backend = server_args.hicache_storage_backend
        storage_extra_config = None
        storage_prefetch_threshold = 256
        prefetch_timeout_base = 1.0
        prefetch_timeout_per_ki_token = 0.25
        hicache_storage_pass_prefix_keys = False
        if storage_backend is not None:
            (
                storage_extra_config,
                storage_prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = HybridCacheController.parse_storage_backend_extra_config(
                server_args.hicache_storage_backend_extra_config
            )

        attach_hybrid_pool_to_unified_cache(
            self,
            params,
            server_args,
            load_cache_event=self.load_cache_event,
            attn_cp_group=params.attn_cp_cache_group,
            attn_tp_group=params.attn_tp_cache_group,
            storage_backend=storage_backend,
            storage_extra_config=storage_extra_config,
            storage_prefetch_threshold=storage_prefetch_threshold,
        )

        # State initialization
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        if storage_backend is not None:
            self._apply_storage_runtime_config(
                storage_backend=storage_backend,
                prefetch_threshold=storage_prefetch_threshold,
                prefetch_timeout_base=prefetch_timeout_base,
                prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
                enable_storage=self.cache_controller.enable_storage,
                enable_storage_metrics=self._enable_metrics_flag,
                extra_metric_labels=self.extra_metric_labels,
            )
            # Uneven-DCP: make the storage-prefetch handshake rank-symmetric so
            # concurrent bursts don't desync the prefetch collectives (deadlock).
            self._symmetrize_prefetch_capacity()

            # #810: bound the write-through consumer of a STAGING host tier,
            # AFTER the symmetrization above so the capacity is the complement
            # of the GROUP-agreed prefetch reservation. A later runtime attach
            # re-derives that reservation; it is not the staging shape (the
            # role requires a backend at boot) and the ring keeps its boot
            # capacity there rather than dropping live admissions.
            from sglang.srt.mem_cache.staging_write_ring import (
                build_staging_write_ring,
            )

            self.staging_write_ring = build_staging_write_ring(
                server_args, self.cache_controller
            )

    def register_sidecar_pool(self, spec: SidecarPoolSpec) -> None:
        self.sidecar_pool_specs.append(spec)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result

        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return self._empty_match_result
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
            key_match_depth,
        ) = self._match_prefix_helper(key)
        return self._match_post_processor(
            params,
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
            key_match_depth,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        result = self._insert_helper(self.root_node, key, value, params)
        return result

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        for component in self._components_tuple:
            component.drive_eviction(params=params, tracker=tracker)

        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            self.writing_check(write_back=True)

        self.update_eviction_metrics(sum(tracker.values()), start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        result = self.session.try_inc_lock_ref(node)
        if result is not None:
            return result
        if self.disable:
            return IncLockRefResult()
        if self._pin_trace_every:
            self._pin_trace_begin("inc")
        result = IncLockRefResult()
        for component in self._components_tuple:
            result = component.acquire_component_lock(node=node, result=result)

        self._update_evictable_leaf_sets(node)
        return result

    def dec_lock_ref(
        self,
        node: Any,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        result = self.session.try_dec_lock_ref(node, params)
        if result is not None:
            return result
        if self.disable:
            return DecLockRefResult()
        if self._pin_trace_every:
            self._pin_trace_begin("dec")
        for component in self._components_tuple:
            if skip_swa and component.component_type == ComponentType.SWA:
                continue
            component.release_component_lock(node=node, params=params)

        self._update_evictable_leaf_sets(node)
        # TODO: delta is not aggregated from components; no caller uses it yet.
        return DecLockRefResult()

    def dec_swa_lock_only(
        self,
        node: UnifiedTreeNode,
        swa_uuid_for_lock: Optional[int] = None,
    ) -> None:
        """Early-release the SWA portion of a request's tree lock, plus any
        strictly-lower-priority locks (e.g. Mamba) co-located on `node`.
        """
        if self.disable:
            return
        swa_component = self.components.get(ComponentType.SWA)
        if swa_component is None:
            return
        swa_component.release_window_lock(node, swa_uuid_for_lock)

        # Drop strictly-lower-priority locks (e.g. Mamba) co-located on `node`.
        swa_priority = swa_component.eviction_priority(is_leaf=False)
        dec_params = DecLockRefParams(swa_uuid_for_lock=swa_uuid_for_lock)
        for comp in self._components_tuple:
            if comp.eviction_priority(is_leaf=False) < swa_priority:
                comp.release_component_lock(node, dec_params)

    def dec_mamba_lock_only(self, node: UnifiedTreeNode) -> bool:
        """Release ONLY the mamba portion of a request's tree lock on `node`.

        #773. The per-component lock model makes this lineage's #755 reorder
        strictly safer than the original: `MambaRadixCache` releases the whole
        node lock early, which also drops the FULL component's KV lock on
        every ancestor up to the root, so the request's own matched prefix is
        evictable inside the window. Here only the MAMBA component's lock is
        dropped -- the KV path stays protected, and the only thing made
        evictable is the one state slot the reorder is trying not to
        double-count.

        `dec_swa_lock_only` is the same shape for the SWA component.
        """
        if self.disable or node is None or node is self.root_node:
            return False
        comp = self.components.get(ComponentType.MAMBA)
        if comp is None:
            return False
        comp.release_component_lock(node, None)
        return True

    def _note_protected_beyond_retention(self, req, effective_cache_len: int) -> None:
        """#935: name the rows that neither the tree nor the allocator will own.

        `cache_protected_len` means "the tree owns the KV below this". The
        finished path frees from it and inserts only up to
        `effective_cache_len`, so a cpl ABOVE the retention leaves
        [effective_cache_len, cache_protected_len) in neither -- a silent,
        per-request loss that scales with the prompt.

        Counted and rate-limited rather than raised: this runs at request
        finish, and an exception here would kill a rank over an accounting
        fault. The number is what was missing, not the crash.
        """
        cpl = int(getattr(req, "cache_protected_len", 0) or 0)
        orphaned = cpl - int(effective_cache_len)
        if orphaned <= 0:
            return
        self._protected_beyond_retention_rows = (
            getattr(self, "_protected_beyond_retention_rows", 0) + orphaned
        )
        n = getattr(self, "_protected_beyond_retention_count", 0) + 1
        self._protected_beyond_retention_count = n
        if n <= 3 or n % 100 == 0:
            logger.warning(
                "#935 PROTECTED-BEYOND-RETENTION rid=%s: cache_protected_len="
                "%d exceeds the retained length %d, so %d row(s) are neither "
                "inserted (the key is truncated to the retention) nor freed "
                "(the free starts at cache_protected_len). They are owned by "
                "nobody from here on. cpl means 'the tree owns the KV below "
                "this' and it does not here. occurrence=%d cumulative_rows=%d",
                getattr(req, "rid", None),
                cpl,
                int(effective_cache_len),
                orphaned,
                n,
                self._protected_beyond_retention_rows,
            )

    def _mamba_anchor_early_release(self, req) -> bool:
        """Decide, per request, whether to take the #755 reorder this step.

        Returns True when the old anchor's mamba lock was released early, so
        the caller must not release it a second time at the normal site.

        A config that promised the reduced floor but meets a node that is not
        host-backed does NOT silently revert to the three-slot order -- the
        floor no longer reserves that slot, and claiming it is the #581 late
        failure. It takes the reduced-budget path instead: the old pin simply
        stays held, which is `active + old pin = 2` and still fits.
        """
        if ComponentType.MAMBA not in self.tree_components:
            return False
        from sglang.srt.mem_cache.mamba_pool_floor import mamba_slot_reorder_active
        from sglang.srt.runtime_context import get_server_args

        if not mamba_slot_reorder_active(get_server_args()):
            return False
        comp = self.components.get(ComponentType.MAMBA)
        if comp is None or not comp.anchor_release_admissible(req.last_node):
            return False
        return self.dec_mamba_lock_only(req.last_node)

    # ---- #811: release the admission anchor pin at the write-through ack ----

    @property
    def _anchor_ack_release_armed(self) -> bool:
        """Config-level gate for the #811 mechanism, resolved once.

        The per-node question -- may THIS pin be released NOW -- is always
        `MambaComponent.anchor_release_admissible` (host copy present AND the
        ack landed); this property never substitutes for it.
        """
        if self._anchor_ack_release_armed_cached is None:
            from sglang.srt.mem_cache.mamba_pool_floor import (
                mamba_anchor_ack_release_active,
            )
            from sglang.srt.runtime_context import get_server_args

            self._anchor_ack_release_armed_cached = mamba_anchor_ack_release_active(
                get_server_args()
            ) and (ComponentType.MAMBA in self.tree_components)
        return self._anchor_ack_release_armed_cached

    @staticmethod
    def _mamba_ref_taken(result: IncLockRefResult, node: Any) -> bool:
        """Did the inc_lock_ref that produced `result` take a MAMBA ref on
        `node`? False for a tombstone (the component recorded the skip)."""
        skipped = result.skip_lock_node_ids.get(ComponentType.MAMBA)
        return not (skipped and node.id in skipped)

    def note_anchor_pin(
        self, req, lock_result: IncLockRefResult, settle: bool = True
    ) -> None:
        """#811: record (and optionally settle) the anchor pin just taken on
        `req.last_node` by an inc_lock_ref whose result is `lock_result`.

        Armed only. `settle=False` is the ADMISSION site: the matched
        anchor's state is the request's deferred-COW SOURCE
        (`req.mamba_cow_src_index`, copied only at the start of the first
        extend forward pass -- model_runner's
        `_maybe_execute_deferred_mamba_cow_and_clear`), so the pin MUST
        survive until that copy has provably executed. Only the sweep, which
        runs over the running (decode) batch -- i.e. over requests whose
        extend result has been processed -- may release it.

        `settle=True` is the CHUNK-BOUNDARY site: the freshly pinned node
        holds a donated COPY of the request's own state, which the request
        never reads back. Three outcomes for a pin that was actually taken:

        * backup ACKED (`anchor_release_admissible`) -- the persistent copy
          exists; release here, exactly as the sweep would later;
        * backup IN FLIGHT -- keep the pin; the sweep releases it after
          `_finish_write_through_ack` retires the ack;
        * NO backup and none in flight (pin budget refused it, or the
          write-through threshold has not admitted it) -- the pin is not
          allowed to persist: it is given back in the same step, before
          anything could rely on it. This is the #581 half of the floor
          argument: with the per-request pinned-checkpoint term dropped, a
          persistent pin may only exist while the retention budget bounds
          it. The state itself stays cached on the device and stays
          evictable -- the same soft degradation the pin budget already
          chose in `write_backup` -- and it has no host copy a resume could
          half-read, so the #767 failure shape is structurally absent here.
        """
        if not self._anchor_ack_release_armed:
            return
        node = req.last_node
        req.mamba_anchor_pin_released = False
        req.mamba_anchor_pin_held = (
            node is not None
            and node is not self.root_node
            and len(node.component_data) > int(ComponentType.MAMBA)
            and node.component_data[ComponentType.MAMBA].value is not None
            and self._mamba_ref_taken(lock_result, node)
        )
        if not req.mamba_anchor_pin_held or not settle:
            return
        comp = self.components.get(ComponentType.MAMBA)
        if comp is None:
            return
        if comp.anchor_release_admissible(node):
            # Acked backup: release now rather than waiting for the sweep.
            if self.dec_mamba_lock_only(node):
                req.mamba_anchor_pin_held = False
                req.mamba_anchor_pin_released = True
        elif node.id not in self.ongoing_write_through:
            # No backup and none in flight: the pin must not persist.
            if self.dec_mamba_lock_only(node):
                req.mamba_anchor_pin_held = False
                req.mamba_anchor_pin_released = True

    def release_acked_anchor_pin(self, req) -> bool:
        """#811 sweep primitive: release `req`'s anchor pin iff its backup is
        acked. Returns True when a ref was actually given back.

        Safe to call every scheduler tick: unarmed, pin-less, and
        already-released requests fall through on attribute checks. The
        admissibility gate is `anchor_release_admissible` -- an in-flight
        backup (node still in `ongoing_write_through`) is refused, which is
        the #767 invariant: no release before the persistent copy exists.
        """
        if not self._anchor_ack_release_armed:
            return False
        if not getattr(req, "mamba_anchor_pin_held", False):
            return False
        if req.mamba_anchor_pin_released:
            return False
        node = req.last_node
        comp = self.components.get(ComponentType.MAMBA)
        if node is None or comp is None:
            return False
        if not comp.anchor_release_admissible(node):
            # A held pin whose node has no host copy and no backup in flight
            # would otherwise stay pinned until the request finishes -- the
            # per-request term the armed floor no longer reserves. Re-issue
            # the backup (still bounded by the write-through pin budget; a
            # refusal is retried at the next sweep), so the ack that permits
            # this release eventually arrives.
            cd = (
                node.component_data[ComponentType.MAMBA]
                if len(node.component_data) > int(ComponentType.MAMBA)
                else None
            )
            if (
                cd is not None
                and cd.value is not None
                and cd.host_value is None
                and node.id not in self.ongoing_write_through
            ):
                self.write_backup(node)
            return False
        if not self.dec_mamba_lock_only(node):
            return False
        req.mamba_anchor_pin_held = False
        req.mamba_anchor_pin_released = True
        return True

    def release_acked_anchor_pins(self, reqs) -> int:
        """Sweep `release_acked_anchor_pin` over `reqs`; returns releases."""
        if not self._anchor_ack_release_armed:
            return 0
        released = 0
        for req in reqs:
            if self.release_acked_anchor_pin(req):
                released += 1
        return released

    def _anchor_dec_skip(self, req, dec_params: DecLockRefParams) -> None:
        """#811: make `dec_params` skip the MAMBA ref of `req.last_node` when
        that ref was already given back early, and consume the marker."""
        if not self._anchor_ack_release_armed:
            return
        if req.mamba_anchor_pin_released and req.last_node is not None:
            dec_params.skip_lock_node_ids.setdefault(ComponentType.MAMBA, set()).add(
                req.last_node.id
            )
        req.mamba_anchor_pin_held = False
        req.mamba_anchor_pin_released = False

    def inc_host_lock_ref(self, node: Any) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult()
        result = IncLockRefResult()
        for component in self._components_tuple:
            result = component.acquire_component_lock(
                node=node, result=result, lock_host=True
            )

        self._update_evictable_leaf_sets(node)
        return result

    def dec_host_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult()
        for component in self._components_tuple:
            component.release_component_lock(node=node, params=params, lock_host=True)

        self._update_evictable_leaf_sets(node)
        return DecLockRefResult()

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs) -> None:
        if self.session.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
            return

        kv_committed_len = req.pop_committed_kv_cache()
        # #969L: THE VALUE AT THE PARK INSERT. §S proved this insert IS reached
        # for a retracted request (is_insert=True, skip=False) and that nothing
        # declines it (#991=0, #969H EMPTY=0), which leaves only a ZERO-LENGTH
        # span. This reads the number and names which of the two park-path
        # writers wrote it last (#969L stamps in schedule_batch: "extend" sets
        # it to seq_len, "reset_for_retract" zeroes it). A missing stamp means
        # one of the OTHER 16 writers (§T: streaming_session 7, dual_group_lane
        # 3, disagg 1) owns it, which would be its own finding.
        # Grep: "#969L COMMIT-AT-INSERT".
        try:
            from sglang.srt.managers.phase_purity import SEAM_READMIT_ATTR as _SRA

            if getattr(req, _SRA, None) is not None or kv_committed_len == 0:
                _n = getattr(UnifiedRadixCache, "_969l_n", 0) + 1
                UnifiedRadixCache._969l_n = _n
                if _n <= 40 or _n % 256 == 0:
                    logger.warning(
                        "#969L COMMIT-AT-INSERT n=%d rid=%s committed=%s src=%s "
                        "is_insert=%s origin=%d out=%d readmit=%s",
                        _n,
                        str(getattr(req, "rid", "?"))[:8],
                        kv_committed_len,
                        getattr(req, "_kvc_src", "UNSTAMPED"),
                        is_insert,
                        len(getattr(req, "origin_input_ids", ()) or ()),
                        len(getattr(req, "output_ids", ()) or ()),
                        getattr(req, _SRA, None),
                    )
        except Exception:  # noqa: BLE001
            logger.warning("#969L COMMIT-AT-INSERT PROBE RAISED", exc_info=True)


        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(req, is_finished=True)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        result = None
        insert_params = None

        if is_insert:
            insert_params = InsertParams(
                prev_prefix_len=req.cache_protected_len,
                priority=req.priority or 0,
                force_host_write_through=requests_forced_host_write_through(req),
            )

            # components prepare insert data + return effective cache_len
            effective_cache_len = len(token_ids)
            for comp in self._components_tuple:
                cl = comp.prepare_for_caching_req(
                    req=req,
                    insert_params=insert_params,
                    token_ids_len=len(token_ids),
                    is_finished=True,
                )
                if cl is not None:
                    effective_cache_len = min(effective_cache_len, cl)

            # Truncate if needed
            if effective_cache_len < len(token_ids):
                free_start = max(effective_cache_len, req.cache_protected_len)
                # #935: THE FINISHED PATH TRUSTS cache_protected_len AND NEVER
                # CHECKS IT, while the unfinished path ASSERTS on it (:1231,
                # the #824 guard). Both free-sites here start at
                # `cache_protected_len` on one premise, stated by
                # `retention_shrinks_protected`: "that length is COMMITTED: the
                # tree owns the KV below it". When that is true the max() is a
                # correct optimisation and nothing leaks.
                #
                # When it is FALSE -- cpl carried forward above what was
                # actually retained -- the rows in
                # [effective_cache_len, cache_protected_len) are neither
                # inserted (token_ids is truncated to effective_cache_len just
                # below) nor freed (the free starts at cpl). They are then
                # owned by nobody: exactly the census's "belong to no
                # enumerated owner", and exactly the per-request deficit #935
                # measures.
                #
                # NAMED, NOT FREED, and the direction is deliberate. Freeing
                # the range would be right if the tree does not own it and a
                # DOUBLE FREE if it does, and this site cannot tell the two
                # apart -- that is the whole reason the premise is trusted
                # here. A wrong free is a use-after-free; a named leak is a
                # number in a log. Whoever lets cpl go stale is the repair
                # (#930's PP-admission truncation is one such producer); this
                # is the guard that stops the loss from being SILENT, which is
                # what let it accumulate unattributed.
                self._note_protected_beyond_retention(req, effective_cache_len)
                self.token_to_kv_pool_allocator.free(kv_indices[free_start:])
                token_ids = token_ids[:effective_cache_len]
                kv_indices = kv_indices[:effective_cache_len]

            radix_key = RadixKey(
                token_ids, req.extra_key, is_bigram=self.is_eagle
            ).page_aligned(self.page_size)
            page_aligned_len = len(radix_key)
            values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

            insert_params.key = radix_key
            insert_params.value = values
            result = self.insert(insert_params)

            # Free unaligned tail
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])

        finish_dec_params = DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock)
        # #811: an anchor pin already released at the write-through ack must
        # not be decremented a second time here. Covers retraction too --
        # release_kv_cache funnels into this method.
        self._anchor_dec_skip(req, finish_dec_params)
        self.dec_lock_ref(
            req.last_node,
            finish_dec_params,
            skip_swa=getattr(req, "swa_prefix_lock_released", False),
        )

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req, is_finished=True, insert_result=result, insert_params=insert_params
            )

    def cache_unfinished_req(self, req: Req, chunked: bool = False, **kwargs) -> None:
        if self.session.try_cache_unfinished_req(req, chunked=chunked, **kwargs):
            return

        token_ids = req.get_fill_ids()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            req.prefix_indices = kv_indices
            return

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # components prepare insert data + return effective cache_len
        insert_params = InsertParams(
            prev_prefix_len=req.cache_protected_len,
            chunked=chunked,
            priority=req.priority or 0,
        )
        # #773/#755: THE LOCK REORDER. Release the OLD anchor's mamba pin
        # BEFORE the donation alloc below, so the old and new anchors never
        # coexist and the request holds `active + donated` rather than
        # `active + donated + old pin`. Only admissible for a node whose host
        # copy has actually landed -- see MambaComponent.anchor_release_admissible.
        #
        # #811 (armed only): the pin may ALREADY be gone -- released at the
        # write-through ack by the sweep, or never persisted (note_anchor_pin).
        # Then the reorder must not release a ref this request no longer
        # holds; the dec below still needs the skip either way.
        if self._anchor_ack_release_armed:
            released_this_call = (
                not req.mamba_anchor_pin_released
                and req.mamba_anchor_pin_held
                and self._mamba_anchor_early_release(req)
            )
            if released_this_call:
                req.mamba_anchor_pin_held = False
                req.mamba_anchor_pin_released = True
            mamba_anchor_released = req.mamba_anchor_pin_released
        else:
            released_this_call = mamba_anchor_released = (
                self._mamba_anchor_early_release(req)
            )

        effective_cache_len = len(token_ids)
        for comp in self._components_tuple:
            cl = comp.prepare_for_caching_req(
                req=req,
                insert_params=insert_params,
                token_ids_len=len(token_ids),
                is_finished=False,
            )
            if cl is not None:
                effective_cache_len = min(effective_cache_len, cl)

        if envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
            for comp in self._components_tuple:
                comp.free_out_of_window_slots(
                    req, effective_cache_len - 1, insert_params
                )

        if effective_cache_len <= 0:
            if released_this_call:
                # #773: nothing was inserted, so no new anchor will take over
                # the pin we dropped -- and `req.last_node` is unchanged, so
                # the NEXT dec_lock_ref for it would decrement a mamba ref
                # this step already released (#583's lock-ref pairing bug).
                # Restore the exact pre-call state instead of carrying the
                # imbalance forward. Re-acquiring a node whose state was
                # evicted inside the window is still correct: the component
                # records the skip and the paired release honours it.
                #
                # #811: only for a release performed IN THIS CALL. A pin the
                # ack already released stays released -- `req.last_node` is
                # unchanged and the request's release marker keeps steering
                # every later dec around it.
                self.components[ComponentType.MAMBA].acquire_component_lock(
                    node=req.last_node, result=IncLockRefResult()
                )
                if self._anchor_ack_release_armed:
                    req.mamba_anchor_pin_held = True
                    req.mamba_anchor_pin_released = False
            req.prefix_indices = kv_indices_orig.to(dtype=torch.int64, copy=True)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(
                    req, is_finished=False, insert_params=insert_params
                )
            return

        kv_indices = kv_indices_orig[:effective_cache_len]

        radix_key = RadixKey(
            token_ids[:effective_cache_len],
            req.extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        page_aligned_len = len(radix_key)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

        insert_params.key = radix_key
        insert_params.value = values
        result = self.insert(insert_params)

        # Match prefix
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        new_prefix_len = result.prefix_len
        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {page_aligned_len=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        dec_params = DecLockRefParams(swa_uuid_for_lock=req.swa_uuid_for_lock)
        if mamba_anchor_released:
            # #773: the mamba half of this lock was already released above, so
            # only the FULL/SWA halves are still outstanding. The skip set is
            # the mechanism the components already use for exactly this
            # question (a ref that was never taken must not be given back).
            dec_params.skip_lock_node_ids.setdefault(ComponentType.MAMBA, set()).add(
                req.last_node.id
            )
        self.dec_lock_ref(req.last_node, dec_params)
        lock_result = self.inc_lock_ref(new_last_node)

        # Update req fields
        if len(new_indices) < len(kv_indices_orig):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices_orig[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.cache_protected_len = len(new_indices)
        req.last_node = new_last_node
        req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock
        # #811: the request now anchors at `new_last_node`, a donated COPY of
        # its own state it never reads back -- settle the fresh pin (release
        # if acked, keep only while a backup is in flight).
        self.note_anchor_pin(req, lock_result, settle=True)

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req,
                is_finished=False,
                insert_result=result,
                insert_params=insert_params,
            )

    # ---- Internal Helpers ----

    def _state_anchor_depth(self, best_match_node: UnifiedTreeNode) -> Optional[int]:
        """Absolute depth of the deepest STATE-BEARING boundary on this match.

        #1040 / the user's round-down grant. KV is DIVISIBLE -- every prefix
        length is a legal place to stop -- while the recurrent (mamba/GDN) state
        is POINTLIKE: it exists only at the exact positions a checkpoint was
        committed to, and it cannot be trimmed or rewound to any other one. A
        length chosen in the KV's coordinate system therefore lands, for mamba,
        on a boundary with no state behind it, and reusing the KV prefix there
        resumes a scan from a state that has consumed different tokens than the
        prefix covers. That is silently wrong, not loudly wrong.

        THE PREDICATE IS NOT RE-DERIVED HERE, AND THAT IS THE POINT.
        `best_match_node` is set in `_match_prefix_helper` only where
        `_all_valid` accepted, and mamba's validator IS `is_resume_candidate`
        (mamba_component.py `create_match_validator`) -- the same call
        `MambaRadixCache._match_prefix_helper` makes. Writing a second "is this
        an anchor" rule here would be exactly the second bookkeeping beside an
        existing truth that the upstream-minimal law forbids, and #747 already
        records what happens when two anchor lineages drift apart. So this
        function only READS OFF the depth of the node that predicate already
        chose: the sum of key lengths from `best_match_node` back to the root,
        which reproduces the walk's `cum_tokens` at the moment that node was
        accepted (the same units the validators were handed, bigram view
        included).

        Returns ``None`` when no component in this cache makes a state-bearing
        claim. A pure-KV cache has nothing to align to, every length is valid
        for it, and every caller must then leave its lengths exactly as they
        were -- that is what keeps the upstream path byte-for-byte unchanged.
        ZERO is a different answer from ``None`` and means something real: the
        walk found no acceptable anchor at all, so the honest extent is "load
        back nothing". Under leaf-only mamba data (a node split NULLS the
        parent's `host_value`, mamba_component.py's split path) the candidate
        set along one path is typically ONE point or NONE, so zero is an
        ordinary outcome and never an error.
        """
        if not any(
            comp.component_type == ComponentType.MAMBA
            for comp in self._components_tuple
        ):
            return None
        depth = 0
        node = best_match_node
        root = self.root_node
        while node is not None and node is not root:
            depth += len(node.key)
            node = node.parent
        return depth

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> tuple[list[torch.Tensor], UnifiedTreeNode, UnifiedTreeNode, int, int]:
        # Non-HiCache mode has only device-resident matches, so the scheduler
        # device anchor follows the best match. In HiCache mode, host-backed
        # nodes can also match, so we separately track the best device-resident
        # match for scheduler prefix indices and locking.
        node = self.root_node
        child_key = key.child_key(self.page_size)
        value: list[torch.Tensor] = []
        best_match_node = node
        best_match_device_node = node
        best_match_device_value_len = 0
        separate_device_match = self.cache_controller is not None
        if separate_device_match:
            validators = tuple(
                comp.create_match_validator() for comp in self._components_tuple
            )
            device_validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in self._components_tuple
            )
        else:
            validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in self._components_tuple
            )

        # #904 (g)/(h): the read-side discriminator. None unless armed, and
        # when it is None the walk below does not build or feed anything --
        # see match_refusal_census for why a zero hit is three worlds, not
        # one, and why only two of them are defects.
        census = new_match_census()
        if census is not None:
            # An armed walk always counts as observed, even when it traverses
            # nothing: "the root had no matching child" is NOT_PRESENT, a real
            # verdict, and must not be reported as "I did not measure".
            census.note_reached(0)
        component_names = tuple(type(comp).__name__ for comp in self._components_tuple)

        def _all_valid(validators, node, depth):
            return all([v(node, depth) for v in validators])

        def _census_refusals(node, depth, tokens):
            """Attribute a refusal to the component(s) that declined it.

            Runs ONLY on the armed path and ONLY for a node that was already
            found invalid, so the extra validator calls cost nothing in the
            default configuration and nothing on an accepted node.

            #913/W42: the component NAME alone was not actionable. A validator
            that is a conjunction hides one defect per term behind its single
            bit, and the 0826 window spent its whole census budget learning
            only that ``MambaComponent`` refused -- not which of its two terms,
            whose fixes are in different files. ``explain_match_refusal`` is
            asked here, beside the predicate that already said no, so the
            reason travels on the same line as the blame instead of having to
            be inferred from a second log.

            The component is asked with the SAME ``match_device_only`` this
            walk built its validators with, or the two would answer about
            different rules -- a host-backed node is admissible to one and not
            the other, and an explanation drawn from the wrong one would name a
            term the predicate never evaluated.
            """
            for name, component, validator in zip(
                component_names, self._components_tuple, validators
            ):
                if not validator(node, depth):
                    try:
                        reason = component.explain_match_refusal(
                            node, depth, match_device_only=not separate_device_match
                        )
                    except Exception:  # noqa: BLE001 - never break a match walk
                        reason = None
                    census.note_refused(name, tokens, reason)

        def _update_best_if_valid(node, depth, key_tokens=0):
            nonlocal best_match_node
            nonlocal best_match_device_value_len, best_match_device_node
            matched = _all_valid(validators, node, depth)
            if census is not None:
                census.note_reached(key_tokens)
                if matched:
                    census.note_accepted(key_tokens)
                else:
                    _census_refusals(node, depth, key_tokens)
            if matched:
                best_match_node = node

            if not separate_device_match:
                if matched:
                    best_match_device_value_len = len(value)
                    best_match_device_node = node
                return
            if _all_valid(device_validators, node, depth):
                best_match_device_value_len = len(value)
                best_match_device_node = node

        # #747: absolute token depth of `node`, accumulated over matched KEY
        # tokens (evicted-but-backuped nodes included -- their tokens are part
        # of the position even when their KV chunk is not in `value`). The
        # mamba validator gates resume anchors on this depth being a
        # --mamba-checkpoint-interval multiple; mirrors `cum_tokens` in
        # MambaRadixCache._match_prefix_helper.
        cum_tokens = 0
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]

            # HiCache: dead node (evicted + not backuped) — stop traversal
            if child.evicted and not child.backuped:
                # #904: the tree kept this node's SHAPE without its bytes.
                # The key matched, so the prefix was stored once; recording
                # the tokens it would have contributed is what separates
                # "loaded then invalidated" from "never written".
                if census is not None:
                    census.note_dead_stop(
                        child.key.match(key, page_size=self.page_size)
                    )
                break

            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                node = self._split_node(child.key, child, prefix_len)
                if not node.evicted:
                    value.append(node.component_data[BASE_COMPONENT_TYPE].value)
                cum_tokens += prefix_len
                _update_best_if_valid(node, cum_tokens, prefix_len)
                break

            if not child.evicted:
                value.append(child.component_data[BASE_COMPONENT_TYPE].value)
            node = child
            cum_tokens += prefix_len
            _update_best_if_valid(node, cum_tokens, prefix_len)
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

        census_emit(census, logger)
        # #915 THE OTHER HALF, wired at last. The reason a prefetch declined has
        # been recorded on every attempt since #915 landed and was never once
        # printed, because this call did not exist. See `prefetch_gate_due`.
        if _prefetch_gate_due():
            logger.info("%s", _format_prefetch_gate())
        # #1040: THE KEY DEPTH IS RETURNED BESIDE THE ANCHOR, NOT INSTEAD OF IT.
        # `cum_tokens` is how far the KEY matched, with no validator consulted;
        # `best_match_node` is how far a node was accepted BY the validators,
        # mamba's `is_resume_candidate` among them. The two are different
        # questions and only their PAIR can say which of the two #1039 worlds a
        # cold load-back is in: "the prefix was never stored" (key depth small)
        # or "the prefix is here and its recurrent anchor died with an evicted
        # node" (key depth large, anchor shallow or zero). A single hit number
        # collapses them, which is the shape #904 already paid for once.
        return (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
            cum_tokens,
        )

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        best_match_node: UnifiedTreeNode,
        best_match_device_node: UnifiedTreeNode,
        best_match_device_value_len: int,
        key_match_depth: int = 0,
    ) -> MatchResult:
        node_update = best_match_node
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue  # Full uses last_access_time, not LRU
            comp.refresh_lru(LRURefreshPhase.MATCH_END, node_update, self.root_node)

        cur_time = get_and_increase_time_counter()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001
            node_update = node_update.parent

        # last_host_node will be used as the starting node for the subsequent
        # `prefetch_from_storage` flow. We directly use best_match_node here,
        # because best_match_node represents the node where all components
        # have reached consensus on both device & host availability.
        last_host_node = (
            best_match_node
            if self.cache_controller is not None
            else best_match_device_node
        )

        if best_match_device_value_len > 0:
            device_indices = torch.cat(value[:best_match_device_value_len])
        else:
            device_indices = self._empty_match_result.device_indices
        result = MatchResult(
            device_indices=device_indices,
            last_device_node=best_match_device_node,
            last_host_node=last_host_node,
            best_match_node=best_match_node,
            host_hit_length=0,
            state_anchor_depth=self._state_anchor_depth(best_match_node),
            key_match_depth=int(key_match_depth),
        )

        for component in self._components_tuple:
            result = component.finalize_match_result(
                result=result,
                params=params,
                value_chunks=value,
                best_value_len=best_match_device_value_len,
            )
        return result

    def _split_node(
        self, key: RadixKey, child: UnifiedTreeNode, split_len: int
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components, priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count
        new_node.creation_time = child.creation_time

        self._for_each_component_lru(child, UnifiedLRUList.remove_node)

        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        for component in self._components_tuple:
            component.redistribute_on_node_split(new_parent=new_node, child=child)
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        if child.backuped:
            self._replace_pending_write_through_node(child, [new_node, child])

        self._for_each_component_lru(
            new_node, UnifiedLRUList.insert_mru, skip_existing=True
        )
        self._for_each_component_lru(
            child, UnifiedLRUList.insert_mru, skip_existing=True
        )
        child.last_access_time = get_and_increase_time_counter()

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(child)
        return new_node

    def _touch_node(self, node: UnifiedTreeNode):
        node.last_access_time = get_and_increase_time_counter()
        if node != self.root_node:
            for comp in self._components_tuple:
                if comp.component_type == BASE_COMPONENT_TYPE:
                    continue
                comp.refresh_lru(LRURefreshPhase.WALKDOWN, node, self.root_node)

    def _add_new_node(
        self,
        parent: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        priority: int = 0,
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components, priority=priority)
        new_node.parent = parent
        new_node.key = key
        new_node.component_data[BASE_COMPONENT_TYPE].value = value.clone()
        parent.children[key.child_key(self.page_size)] = new_node
        self.component_evictable_size_[BASE_COMPONENT_TYPE] += len(value)
        if self.enable_storage:
            new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(parent)
        self._record_store_event(new_node)
        return new_node

    def _unevict_node_on_insert(
        self, node: UnifiedTreeNode, fresh_value: torch.Tensor
    ) -> None:
        """Restore an evicted node's Full device value from fresh KV indices
        during insert."""
        ct = BASE_COMPONENT_TYPE
        cd = node.component_data[ct]
        assert cd.value is None
        n = len(fresh_value)
        cd.value = fresh_value.clone()
        self.component_evictable_size_[ct] += n
        self._update_evictable_leaf_sets(node)
        if node.parent is not None:
            self._update_evictable_leaf_sets(node.parent)
        self._record_store_event(node, medium=StorageMedium.GPU)

    def _insert_helper(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        params: InsertParams,
    ) -> InsertResult:
        priority = params.priority
        if priority is None:
            priority = 0
        self._touch_node(node)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        child_key = key.child_key(self.page_size)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)
            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)
            node.priority = max(node.priority, priority)

            if node.evicted:
                self._unevict_node_on_insert(node, value[:prefix_len])
                # FULL was restored from the request's fresh KV. Aux
                # components (e.g. SWA) may still hold tombstones and need
                # to rebuild their value from the same slice.
                for component in self._components_tuple:
                    if component.component_type == BASE_COMPONENT_TYPE:
                        continue
                    component.recover_after_unevict(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        params=params,
                    )
            else:
                value_slice = value[:prefix_len]
                consumed_from = prefix_len
                # Let each component claim ownership of overlapping KV slots
                for component in self._components_tuple:
                    comp_consumed_from = component.update_component_on_insert_overlap(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        value_slice=value_slice,
                        params=params,
                    )
                    consumed_from = min(consumed_from, comp_consumed_from)

                dup_start = max(0, params.prev_prefix_len - total_prefix_length)
                if dup_start < consumed_from:
                    self.token_to_kv_pool_allocator.free(
                        value_slice[dup_start:consumed_from]
                    )

            self._inc_hit_count(node, params.chunked, params.force_host_write_through)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

        is_new_leaf = False
        # Create new leaf for remaining suffix. A leaf survives on its Full
        # value alone; auxiliary components (SWA, Mamba) may legitimately hold
        # only a tombstone for this span (e.g. the whole leaf is outside the SWA
        # window). Materialize it anyway so the Full KV stays cacheable.
        if len(key):
            target_node = self._add_new_node(node, key, value, priority=priority)
            is_new_leaf = True
        else:
            target_node = node

        # Finalize: let each component attach its data to the target node.
        # e.g. Mamba attaches mamba_value to the leaf node
        result = InsertResult(prefix_len=total_prefix_length)
        for component in self._components_tuple:
            component.commit_insert_component_data(
                node=target_node,
                is_new_leaf=is_new_leaf,
                params=params,
                result=result,
            )

        if target_node is not self.root_node:
            for component in self._components_tuple:
                if component.component_type == BASE_COMPONENT_TYPE:
                    continue
                component.refresh_lru(
                    LRURefreshPhase.INSERT_END, target_node, self.root_node
                )

        if is_new_leaf:
            self._inc_hit_count(
                target_node, params.chunked, params.force_host_write_through
            )
        return result

    def _insert_helper_host(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        host_value: torch.Tensor,
        hash_value: list[str],
    ) -> InsertResult:
        total_len = len(key)
        self._touch_node(node)
        if total_len == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        child_key = key.child_key(self.page_size)
        matched_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)

            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)

            if len(key):
                child_key = key.child_key(self.page_size)

        result = InsertResult(prefix_len=matched_length, total_len=total_len)
        if len(key) == 0:
            if (
                node is not self.root_node
                and node.component_data[BASE_COMPONENT_TYPE].host_value is not None
            ):
                result.inserted_host_node = node
            return result

        # #841: THE CONTIGUOUS-BACKUP LAW, enforced here because this is the
        # second writer into the host tier and it used to be exempt.
        #
        # The law: a node may carry a host copy only if its parent does. It is
        # not decorative. `_evict_device_leaf`'s write-through branch DELETES
        # an un-backed device leaf outright, and `_is_device_leaf` qualifies a
        # node whose children hold host data but no device data. So an
        # un-backed parent above a backed child is a node that can be deleted
        # while it still has children -- `_remove_leaf_from_parent` pops the
        # edge and the backed subtree below it is orphaned: still in
        # `evictable_host_leaves`, still in the aux host LRUs, no longer
        # reachable from the root. That is precisely the window-5 idle-path
        # crash, in both of its shapes:
        #
        #   "node 144 backed up but parent 11 not backed up"   (the state)
        #   "H-leaf extra: [18, 19]" + "stale nodes in host_leaves: [18, 19]"
        #   + "mamba host LRU: +S3=set(), +lru={18}"           (after the delete)
        #
        # `write_backup` has always upheld the law, at :1943-1948, by backing
        # the parent up first and refusing when that fails. This path had no
        # such gate: `check_prefetch_progress` walks down from
        # `last_host_node` -- which `Scheduler._prefetch_kvcache` allows to be
        # the ROOT (scheduler.py:4420) -- through whatever children match the
        # fetched key, device-only ones included, and attached the fetched tail
        # wherever the walk stopped. Window 5's `matched=45` is exactly
        # root -> node 12 (1 token) -> node 11 (44 tokens), both device-only.
        #
        # Refusing costs one storage hit. Attaching costs the scheduler, on the
        # next idle check, on every rank at once. The same trade `write_backup`
        # already makes one function up.
        #
        # WHY REFUSE RATHER THAN BACK THE PARENT UP HERE. Calling
        # `write_backup(node)` would satisfy the law and keep the hit -- it
        # recurses up the chain and is the authority for exactly this. It is
        # also a rank-LOCAL tree edit (`evict_host` deletes host leaves) issued
        # immediately after the prefetch's own all_reduce, i.e. at a collective
        # seam. That is the #639/#645 wedge shape, four specimens deep, and it
        # is not worth re-opening to recover a prefetch tail.
        #
        # RESIDUAL RISK, named rather than assumed away: `backuped` is not
        # provably rank-uniform (the mamba pin budget at
        # `_mamba_write_through_pin_admissible` is rank-local), so ranks could
        # in principle decline on different nodes and diverge. This predicate
        # already drives rank-divergent tree edits today -- `_evict_device_leaf`
        # DELETES an un-backed node and DEMOTES a backed one -- and #645
        # uniformized the host admission behind it for that reason. This gate
        # rides on that same uniformization; it does not add a new class of
        # divergence. If a boot ever shows the ranks declining different nodes,
        # that is #645's admission drifting, not this gate.
        #
        # UPSTREAM CORROBORATION, found after this fix was written and worth
        # recording as such: sgl-project PR 31902 (Yiqi Yang, 30043ca7eb
        # 2026-07-21 and f198ebf97f 2026-07-22, "drop prefetched host refill
        # under an un-backed-up parent") diagnoses the same defect and installs
        # the same gate at the same line, with a 150-test regression file. It is
        # NOT merged: it is absent from `upstream/main` at 95f5ecd3d2
        # (2026-08-24), so the defect is live upstream today and no OSS sync
        # could have carried the repair into this tree.
        #
        # The write-policy condition below is taken from that PR's second
        # commit rather than invented here. It is right: `sanity_check` arms
        # this invariant only when the policy is not `write_back`
        # (:3899-3903), and under `write_back` the orphan hazard is absent too
        # -- `_evict_device_leaf` writes an un-backed leaf back and returns
        # rather than deleting it, so no edge is ever popped above a backed
        # child. Declining under `write_back` would refuse a harmless insert
        # and cost retention for nothing.
        write_back_policy = (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        )
        parent_backed = node is self.root_node or node.backuped
        if not parent_backed and not write_back_policy:
            self._host_insert_refused_unbacked_parent += 1
            # #1035 R11: THIS WAS logger.debug AND THEREFORE INVISIBLE.
            # The chain census found it as a BLIND row: this is the #841 law
            # itself -- it silently zeroes a host insert and marks the whole
            # fetched tail unclaimed -- and on an INFO boot it printed NOTHING,
            # so a boot where this fired and a boot where storage was simply
            # cold looked identical in every log of the campaign. A guard that
            # can zero the read path must be audible at the level the boots
            # actually run at. Rate-limited like the campaign's other
            # instruments so a hot path cannot flood the log.
            if (
                self._host_insert_refused_unbacked_parent <= 40
                or self._host_insert_refused_unbacked_parent % 256 == 0
            ):
                logger.warning(
                    "#841 host-only insert declined (n=%d): parent node %s "
                    "carries no host copy, so a backed child under it could be "
                    "orphaned by a device eviction. matched=%d declined=%d "
                    "-- the declined tail is released, NOT published.",
                    self._host_insert_refused_unbacked_parent,
                    node.id,
                    matched_length,
                    total_len - matched_length,
                )
            # The caller reserved the tail for a node that will not exist.
            # Nothing in the tree can free it, so say so.
            result.host_span_unclaimed = True
            return result

        new_node = UnifiedTreeNode(self.tree_components, priority=node.priority)
        new_node.parent = node
        new_node.key = key
        new_node.hash_value = hash_value
        new_node.component_data[BASE_COMPONENT_TYPE].host_value = host_value.clone()
        node.children[child_key] = new_node
        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(node)
        result.inserted_host_node = new_node
        return result

    # ---- Evict Helpers ----

    def _cascade_evict(
        self,
        node: UnifiedTreeNode,
        trigger: TreeComponent,
        tracker: dict[ComponentType, int],
        target: EvictLayer = EvictLayer.DEVICE,
    ):
        """Cascade eviction from trigger to lower-or-equal priority components."""

        is_leaf = False
        if target == EvictLayer.DEVICE:
            is_leaf = node in self.evictable_device_leaves
        elif target == EvictLayer.HOST:
            is_leaf = node in self.evictable_host_leaves

        trigger_priority = trigger.eviction_priority(is_leaf)

        for comp in self._components_tuple:
            if comp.eviction_priority(is_leaf) <= trigger_priority:
                if comp is not trigger and comp.node_has_component_data(node, target):
                    cd = node.component_data[comp.component_type]
                    # A comp whose TRUE internal priority outranks the trigger
                    # is only in this loop because leaf-collapse flattened
                    # priorities; a lock on it is a legit pin and must be
                    # spared. A lock on a strictly-lower-priority tier is a
                    # real strand — fall through to the assert below.
                    if comp.eviction_priority(
                        is_leaf=False
                    ) >= trigger.eviction_priority(is_leaf=False):
                        if EvictLayer.DEVICE in target and cd.lock_ref != 0:
                            continue
                        if EvictLayer.HOST in target and cd.host_lock_ref != 0:
                            continue
                    if EvictLayer.DEVICE in target:
                        assert cd.lock_ref == 0
                    if EvictLayer.HOST in target:
                        assert cd.host_lock_ref == 0
                    self._evict_component_and_detach_lru(
                        node, comp, target=target, tracker=tracker
                    )

        # Now that all components (including SWA which depends on Full.value)
        # have been freed, we can safely tombstone Full.value.
        # This is deferred from evict_component because free_swa needs it.
        #
        # #927 INVESTIGATED AND LEFT ALONE -- the trigger test is not the
        # narrower question it looks like, it is EQUIVALENT here, and the
        # priority lattice is why. On the DEVICE target this function is
        # reached with a non-BASE trigger from exactly two places, both on
        # INTERNAL nodes (`mamba_component.py:529`, `swa_component.py:441`),
        # and internal priorities are "full=2 > swa=1 > mamba=0"
        # (tree_component.py:292). The cascade loop admits a component only
        # when `eviction_priority(is_leaf) <= trigger_priority`, so Full at 2
        # is unreachable from a trigger at 0 or 1. The leaf path does not use
        # this function at all (`_evict_device_leaf` loops the components
        # directly), and `_evict_to_host` -- the only path that leaves a node
        # in the tree -- passes the BASE component as the trigger explicitly
        # (`:2065-2071`). So "Full was the trigger" and "Full's rows were
        # freed in this cascade" name the same set of cascades.
        #
        # Recorded because a rewrite to `base_rows_freed` was built, shipped
        # in 698cd396ce and reverted here as a NO-OP: it changed no behaviour,
        # and a one-line mutant back to this condition leaves a behavioural
        # suite green precisely because the two are equivalent. #927's real
        # root is elsewhere; the doubly-claimed rows are not produced here.
        if (
            target is EvictLayer.DEVICE
            and trigger.component_type == BASE_COMPONENT_TYPE
        ):
            node.component_data[trigger.component_type].value = None

        self._update_evictable_leaf_sets(node)

    def _remove_leaf_from_parent(self, node: UnifiedTreeNode):
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node

    def _evict_component_and_detach_lru(
        self,
        node: UnifiedTreeNode,
        comp: TreeComponent,
        target: EvictLayer = EvictLayer.DEVICE,
        tracker: Optional[dict[ComponentType, int]] = None,
    ) -> tuple[int, int]:
        # #904: MAKE THE ORDERING ENFORCED, NOT ASSUMED.
        #
        # Every caller of this funnel already selects an unlocked node --
        # `drive_eviction` walks `lru.get_lru_no_lock()`, `_evict_device_leaf`
        # asserts `_is_device_leaf` (which excludes `lock_ref > 0`), and
        # `_evict_to_host` is reached only from those. But that is a property
        # of the CALLERS, and a component's rows are freed HERE. A pin is
        # precisely what covers the window between "the H2D copy is enqueued"
        # and "a reader has consumed it"; freeing under one is the
        # load-then-invalidate half of #904, and it would be silent -- the
        # slot is recycled and the next reader gets someone else's state, with
        # no assertion anywhere.
        #
        # The family's own convention is an ACT-time check, not a selection-
        # time one: mamba_radix_cache.py:1149-1178 and :1286-1299,
        # swa_radix_cache.py:600 and :638, hi_mamba_radix_cache.py:1137 all
        # assert at the free. This funnel is the one place that did not, so
        # it joins them rather than relying on six callers staying correct.
        if EvictLayer.DEVICE in target:
            cd = node.component_data[comp.component_type]
            if cd.value is not None and cd.lock_ref > 0:
                raise ValueError(
                    f"#904: refusing to free {comp.component_type.name} device "
                    f"rows of node {node.id} while lock_ref={cd.lock_ref}. The "
                    "pin covers a load-back or a running request; the caller "
                    "must select an unlocked node, never free under the pin."
                )
        device_freed, host_freed = comp.evict_component(node, target=target)
        if tracker is not None:
            if EvictLayer.DEVICE in target:
                tracker[comp.component_type] += device_freed
            elif EvictLayer.HOST in target:
                tracker[comp.component_type] += host_freed

        # Detach from the appropriate LRU list(s)
        ct = comp.component_type
        for layer, lru_lists in (
            (EvictLayer.DEVICE, self.lru_lists),
            (EvictLayer.HOST, self.host_lru_lists),
        ):
            if layer in target:
                lru = lru_lists[ct]
                if lru.in_list(node):
                    lru.remove_node(node)
        return device_freed, host_freed

    def _iteratively_delete_tombstone_leaf(
        self, deleted_node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ):
        """Walk up from *deleted_node* and cascade-delete childless ancestors.

        Only the Full (base) component decides whether a node survives:
          - Full device present  → keep as D-leaf
          - Full host present    → keep as H-leaf
          - neither              → evict all remaining data, delete, continue up
        """
        ct = BASE_COMPONENT_TYPE
        cur = deleted_node.parent
        while cur != self.root_node and len(cur.children) == 0:
            if any(
                cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in cur.component_data
            ):
                break

            has_device = cur.component_data[ct].value is not None
            has_host = cur.component_data[ct].host_value is not None

            if has_device:
                self._update_evictable_leaf_sets(cur)
                break

            # Full device absent — clean up orphaned aux device data.
            for comp in self.components.values():
                if comp.node_has_component_data(cur):
                    self._evict_component_and_detach_lru(
                        cur, comp, target=EvictLayer.DEVICE, tracker=tracker
                    )

            if has_host:
                self._update_evictable_leaf_sets(cur)
                break

            # Full absent on both layers — evict remaining host data, delete.
            for comp in self.components.values():
                if comp.node_has_component_data(cur, target=EvictLayer.HOST):
                    self._evict_component_and_detach_lru(
                        cur, comp, target=EvictLayer.HOST, tracker=tracker
                    )

            self.evictable_host_leaves.discard(cur)
            self._remove_leaf_from_parent(cur)
            parent = cur.parent
            self._update_evictable_leaf_sets(parent)
            cur = parent

    def _for_each_component_lru(
        self,
        node: UnifiedTreeNode,
        lru_op,
        target: EvictLayer = EvictLayer.DEVICE,
        skip_existing: bool = False,
    ):
        """Apply lru_op to each aux component's LRU that has data on this node.
        If skip_existing=True, skip components already in the target LRU list."""
        lru_dict = self.host_lru_lists if target is EvictLayer.HOST else self.lru_lists
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue  # Full uses leaf sets, not LRU
            cd = node.component_data[ct]
            if (cd.host_value if target is EvictLayer.HOST else cd.value) is not None:
                lru = lru_dict[ct]
                if skip_existing and lru.in_list(node):
                    continue
                lru_op(lru, node)

    def evict_host(
        self, num_tokens: int, component_type: ComponentType = BASE_COMPONENT_TYPE
    ) -> int:
        """Evict host resources for a specific component to free host pool space."""
        tracker: dict[ComponentType, int] = {ct: 0 for ct in self.tree_components}
        comp = self.components.get(component_type)
        if comp is not None:
            comp.drive_host_eviction(num_tokens, tracker)
        return tracker[component_type]

    def _is_device_leaf(self, node: UnifiedTreeNode) -> bool:
        """D-leaf: Full device value present, no child with Full KV on device,
        unlocked, not root.

        Only the Full (base) component is required; auxiliary components
        (Mamba, SWA) are not mandatory for D-leaf membership."""
        ct = BASE_COMPONENT_TYPE
        if node is self.root_node or node.evicted:
            return False
        if any(cd.lock_ref > 0 for cd in node.component_data):
            return False
        if any(
            child.component_data[ct].value is not None
            for child in node.children.values()
        ):
            return False
        return True

    def _is_host_leaf(self, node: UnifiedTreeNode) -> bool:
        """H-leaf: evicted, Full host value present, no children, unlocked, not root.

        Only the Full (base) component host_value is required; auxiliary
        components are not mandatory for H-leaf membership."""
        if node is self.root_node or not node.evicted:
            return False
        if not node.backuped:
            return False
        if any(cd.host_lock_ref > 0 for cd in node.component_data):
            return False
        if len(node.children) > 0:
            return False
        return True

    def _update_evictable_leaf_sets(self, node: UnifiedTreeNode) -> None:
        """Update both device and host leaf sets for a node."""
        if self._is_device_leaf(node):
            self.evictable_device_leaves.add(node)
        else:
            self.evictable_device_leaves.discard(node)

        if self._is_host_leaf(node):
            self.evictable_host_leaves.add(node)
        else:
            self.evictable_host_leaves.discard(node)

    def _evict_to_host(
        self, node: UnifiedTreeNode, tracker: Optional[dict[ComponentType, int]] = None
    ) -> None:
        """GPU→CPU demotion: release all device resources, node stays in tree."""
        assert not node.evicted and node.backuped
        trigger = self.components[BASE_COMPONENT_TYPE]
        self._evict_component_and_detach_lru(
            node, trigger, target=EvictLayer.DEVICE, tracker=tracker
        )
        self._cascade_evict(node, trigger, tracker)
        self._record_remove_event(node, medium=StorageMedium.GPU)

        # after device eviction, insert aux components into host LRU.
        self._for_each_component_lru(
            node, UnifiedLRUList.insert_mru, target=EvictLayer.HOST, skip_existing=True
        )
        self._update_evictable_leaf_sets(node.parent)

    def _evict_device_leaf(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict a device leaf node, choosing the right strategy:

        - backuped: demote to host via _evict_to_host (node stays in tree)
        - not backuped + write_back: write_backup first, then demote
        - not backuped + write_through: Cascade evict all components

        All freed device tokens are accumulated into *tracker*.
        """
        assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"
        if not node.backuped:
            if (
                self.cache_controller is not None
                and self.cache_controller.write_policy == "write_back"
            ):
                written = self.write_backup(node, write_back=True)
                if written == 0:
                    return
                self.writing_check(write_back=True)
                self._evict_to_host(node, tracker)
                return
            else:
                # #841: this branch DELETES, and `_is_device_leaf` qualifies a
                # node whose children hold host data but no device data -- so
                # without this guard the delete pops the edge above a
                # host-backed subtree and orphans it in `evictable_host_leaves`
                # and the aux host LRUs. Under the contiguous-backup law
                # (upheld by `write_backup` and, since #841, by
                # `_insert_helper_host`) a backed child implies a backed
                # parent, so an UN-backed node cannot have backed children and
                # this state is unreachable. Keep the check anyway: it is the
                # falsifier for the law. Refusing the eviction leaves the
                # node's device rows in place, which is what keeps the law
                # true; dropping a caller's eviction request costs tokens,
                # while the delete costs every rank's scheduler at the next
                # idle check.
                if node.children:
                    logger.error(
                        "#841 refusing to delete un-backed node %s: it still "
                        "has %d child node(s), and deleting it would orphan "
                        "any host-backed subtree below it. The "
                        "contiguous-backup law has been broken upstream of "
                        "this point -- find the writer, do not relax this "
                        "guard.",
                        node.id,
                        len(node.children),
                    )
                    return
                # Write-through: node has no backup, delete entirely.
                self._record_remove_event(node, medium=StorageMedium.GPU)
                for comp in self._components_tuple:
                    self._evict_component_and_detach_lru(
                        node, comp, target=EvictLayer.ALL, tracker=tracker
                    )
                self.evictable_device_leaves.discard(node)
                parent = node.parent
                self._remove_leaf_from_parent(node)
                self._update_evictable_leaf_sets(parent)
                self._iteratively_delete_tombstone_leaf(node, tracker)
                return
        self._evict_to_host(node, tracker)

    def _evict_host_leaf(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ) -> None:
        """Atomically evict all components on a host leaf.

        All freed tokens are accumulated into *tracker*."""
        assert self._is_host_leaf(node), f"node {node.id} is not an H-leaf"

        self._record_remove_event(node, medium=StorageMedium.CPU)
        for comp in self._components_tuple:
            _, hf = self._evict_component_and_detach_lru(
                node, comp, target=EvictLayer.ALL, tracker=None
            )
            tracker[comp.component_type] += hf
        self.evictable_host_leaves.discard(node)
        self._remove_leaf_from_parent(node)
        self._iteratively_delete_tombstone_leaf(node, tracker)

    # ---- HiCache: Backup / LoadBack ----

    def write_backup(self, node: UnifiedTreeNode, write_back: bool = False) -> int:
        """Backup a node's data from device to host (D->H)."""
        if self.cache_controller is None:
            return 0

        # #581 pin budget, #773: a write-through pin on a node that carries a
        # mamba checkpoint makes its STATE SLOT unevictable until the ack
        # drains. Beyond the budget, skip the backup entirely rather than pin
        # another slot -- the state stays cached on the device and stays
        # EVICTABLE, so the pool can always serve the running set's required
        # allocations. Backing off here costs at most a host-tier miss; not
        # backing off costs the scheduler.
        #
        # Checked BEFORE the parent recursion and before any transfer is
        # built, so a refusal neither forces a parent backup this node will
        # not use nor strands a host allocation.
        if not self._mamba_write_through_pin_admissible(node, write_back=write_back):
            self._note_mamba_pin_skipped()
            return 0

        # Backup invariant (write-through): parent must be backuped first
        if not write_back and (
            node.parent is not self.root_node and not node.parent.backuped
        ):
            if self.write_backup(node.parent) <= 0:
                return 0

        device_value = node.component_data[BASE_COMPONENT_TYPE].value
        kv_xfer = PoolTransfer(name=PoolName.KV, device_indices=device_value)

        # Build aux transfers, keyed per component.
        comp_xfers: dict[ComponentType, list] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            t = comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_HOST)
            if t:
                comp_xfers[comp.component_type] = t
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_HOST, kv_xfer, comp_xfers
        )

        # Pre-evict host if insufficient.
        #
        # #639: RANK-UNIFORM host admission. `kv_tokens` is replicated (it is
        # this node's own length); the host pool is this rank's own shard --
        # 359652 / 287722 / 273336 slots on the crashing boot. Decided
        # locally, the roomy rank backs the node up and the tight ranks return
        # 0, and under `write_through` that verdict is a TREE EDIT rather than
        # bookkeeping: `_evict_device_leaf` DEMOTES a backed-up node (it stays
        # in the tree, matchable, and `load_back` can restore it) and DELETES
        # one without a backup. The radix replicas then diverge, `match_prefix`
        # returns a rank-dependent prefix, and `prepare_for_extend` turns it
        # into a rank-dependent `extend_num_tokens` -- the BAR1 stall with the
        # CLEAN abort word, four specimens, roomiest rank always LOW.
        #
        # This is one tier BELOW the #616g device-side floors and upstream of
        # them, which is why those are deployed and did not bind: `load_back`'s
        # floor decides whether host content is loaded back, not whether it
        # was ever written. None (host pools agree, single rank, no host tier)
        # keeps the live local value exactly as before.
        #
        # #645: and the eviction BELOW the compare is where that floor was
        # still not enough. Making `host_avail` replicated made the `if` a
        # rank-uniform branch -- every rank now enters the eviction together,
        # with the same `needed` -- but `evict_host` selects its victims from
        # `evictable_host_leaves`, and `_is_host_leaf` gates candidacy on
        # `node.evicted`, i.e. on a rank-sized DEVICE pool. So the ranks
        # delete different nodes (`_evict_host_leaf` ->
        # `_remove_leaf_from_parent`, a tree edit) and some raise `needed`
        # while others do not (`evicted < needed` -> no backup, and under
        # write_through that node is deleted at its next device eviction).
        # Two production specimens on 2026-08-07, one per direction: 13:26
        # rank 0 [22014] vs peers [19967] (kept a chunk node its peers
        # deleted, difference exactly one 2047-token chunk) and 12:15 rank 0
        # [2047] vs peers [10238] (lost the new node its peers backed up).
        #
        # A floor repairs an ADMISSION, not a SELECTION: no arithmetic on a
        # published scalar makes two ranks pick the same nodes out of two
        # different candidate sets. So under an active floor this path does
        # not evict at all and refuses uniformly instead -- a refusal every
        # rank reaches on the same replicated compare, leaving every tree
        # untouched.
        #
        # THE COST, stated plainly: this is the only host-eviction trigger the
        # base component has when `enable_storage` is False (the prefetch
        # trigger returns early without a storage backend). So on an UNEVEN
        # rig -- the only case where a floor is published at all -- a
        # saturated host tier now stops accepting new backups instead of
        # recycling itself, and the host-tier hit rate decays to whatever was
        # cached before saturation. That is a throughput regression traded
        # for a correctness one, deliberately: the alternative is the wedge
        # above. Recovering the recycling needs a group-agreed victim list,
        # which is a scheduler-side design, not a fix to this line.
        #
        # With no floor (pools agree, single rank, no host tier) the eviction
        # retry runs exactly as before.
        kv_tokens = len(device_value)
        host_avail = uniform_host_avail_for_backup(
            self, self.cache_controller.mem_pool_host
        )
        if host_avail < kv_tokens:
            if uniform_host_floor_active(self):
                return 0
            needed = kv_tokens - host_avail
            evicted = self.evict_host(needed)
            if evicted < needed:
                return 0

        # #810: the STAGING bound, taken BEFORE the allocation rather than
        # after it fails. Under `--hicache-host-role staging` the tier is a
        # drain buffer, so the undrained write-through set must leave room for
        # the read consumer; a refusal costs one un-backed-up node -- what an
        # exhausted tier costs today -- but it is COUNTED and it is reached
        # without the rank-local `evict_host` above. `None` under the default
        # role skips the gate entirely.
        ring = self.staging_write_ring
        if ring is not None and not ring.admit(node.id, kv_tokens):
            return 0

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        host_indices = self.cache_controller.write(
            device_value, node_id=node.id, extra_pools=aux_xfers or None
        )
        if host_indices is None:
            # #810: the write failed after the ring admitted it, so the page
            # never reaches the drain and its admission must not stay charged.
            if ring is not None:
                ring.abort(node.id)
            return 0

        # #645: charge the admission against the published floor, so the next
        # backup in THIS iteration decides against what is left rather than
        # against the iteration-start snapshot. No-op when no floor is active.
        note_uniform_host_admitted(self, kv_tokens)

        # Commit
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            node,
            CacheTransferPhase.BACKUP_HOST,
            transfers=[kv_xfer],
        )
        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                node,
                CacheTransferPhase.BACKUP_HOST,
                transfers=xfers,
            )

        lock_params = None
        if not write_back:
            lock_params = self.inc_lock_ref(node).to_dec_params()
        self._track_write_through_node(node, lock_params)
        return len(host_indices)

    def _track_write_through_node(
        self,
        node: UnifiedTreeNode,
        lock_params: Optional[DecLockRefParams],
    ) -> None:
        node.write_through_pending_id = node.id
        self.ongoing_write_through[node.id] = _OngoingWriteThrough(
            node, lock_params, [node]
        )

    def _replace_pending_write_through_node(
        self, old_node: UnifiedTreeNode, new_nodes: list[UnifiedTreeNode]
    ) -> None:
        ack_id = old_node.write_through_pending_id
        if ack_id is None:
            return

        pending = self.ongoing_write_through.get(ack_id)
        if pending is None:
            return

        lock_node, lock_params, publish_nodes = pending
        updated_nodes = []
        replaced = False
        for node in publish_nodes:
            if node is old_node:
                updated_nodes.extend(new_nodes)
                replaced = True
            else:
                updated_nodes.append(node)

        if not replaced:
            return

        for node in new_nodes:
            node.write_through_pending_id = ack_id
        self.ongoing_write_through[ack_id] = _OngoingWriteThrough(
            lock_node,
            lock_params,
            updated_nodes,
        )

    def _finish_write_through_ack(self, ack_id: int) -> None:
        lock_node, lock_params, publish_nodes = self.ongoing_write_through.pop(ack_id)
        for node in publish_nodes:
            if node.write_through_pending_id == ack_id:
                node.write_through_pending_id = None
            self._record_store_event(node, medium=StorageMedium.CPU)
        if lock_params is not None:
            self.dec_lock_ref(lock_node, lock_params)
        # #810: end of the ADMITTED phase. The device->host copy has landed, so
        # the admission taken in `write_backup` is retired here -- before the
        # storage hand-off below, which takes its own charge per operation.
        # Releasing first keeps the two phases from double-counting one page,
        # and retiring the node-keyed charge at exactly one site keeps a node
        # SPLIT (one ack, several storage backups) from stranding it.
        if self.staging_write_ring is not None:
            self.staging_write_ring.release(ack_id)
        if self.enable_storage:
            # Back up each fragment: after a split, lock_node only holds the
            # suffix; the prefix fragment must be persisted as well.
            for node in publish_nodes:
                self.write_backup_storage(node)

    def load_back(
        self,
        best_match_node: UnifiedTreeNode,
        mem_quota: Optional[int] = None,
        req=None,
    ) -> bool:
        """Load evicted KV data from host back to device (H→D)."""
        if self.cache_controller is None:
            return False

        start_time = time.perf_counter()
        host_anchor_params = self.inc_host_lock_ref(best_match_node).to_dec_params()
        # Build KV transfer
        kv_xfer = self.components[BASE_COMPONENT_TYPE].build_hicache_transfers(
            best_match_node, CacheTransferPhase.LOAD_BACK
        )[0]

        # Lock path & pre-evict if device pool is insufficient
        result = self.inc_lock_ref(best_match_node)
        ancestor_lock_params = result.to_dec_params()
        kv_tokens = len(kv_xfer.host_indices)

        # Build aux transfers, keyed per component.
        comp_xfers: dict[ComponentType, list] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            try:
                t = comp.build_hicache_transfers(
                    best_match_node, CacheTransferPhase.LOAD_BACK, req=req
                )
            except MambaLoadBackUnservable:
                # #968 FIX-4: a component that cannot serve its half of this
                # load-back kills the WHOLE load-back. Nothing has been
                # transferred yet at this point -- `cache_controller.load` is
                # still below -- so unwinding the two lock refs taken above
                # leaves the tree exactly as it was found, the same shape as
                # every other `return False` in this function.
                self.dec_lock_ref(best_match_node, ancestor_lock_params)
                self.dec_host_lock_ref(best_match_node, host_anchor_params)
                return False
            if t:
                comp_xfers[comp.component_type] = t
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.LOAD_BACK, kv_xfer, comp_xfers
        )

        # Skip if there is nothing to load, or if the Full-KV transfer is too
        # small / exceeds memory quota. Aux transfers should still run even
        # when the Full-KV load is skipped by thresholding.
        if (kv_tokens < self.load_back_threshold and not comp_xfers) or (
            mem_quota is not None and kv_tokens > mem_quota + result.delta
        ):
            self.dec_lock_ref(best_match_node, ancestor_lock_params)
            self.dec_host_lock_ref(best_match_node, host_anchor_params)
            return False

        # #616g: RANK-UNIFORM load-back admission. This is the same defect as
        # the eviction trigger in `evict_from_tree_cache`, in the class this
        # rig actually runs (UnifiedRadixCache, not HiRadixCache): load-back
        # EXTENDS this rank's device prefix, and whether it happens is decided
        # from this rank's own free space. Under uneven pools the roomy rank
        # loads the prefix back while the tight rank gives up, the device trees
        # stop being replicas, and `match_prefix` -> `extend_num_tokens` turns
        # that into TP collectives entered with rank-dependent token counts --
        # the 21:52:25 wedge (rank 0, the largest pool, reducing 1690 tokens
        # against its peers' 1818).
        #
        # Decide from the group floor: if the BINDING rank cannot hold the
        # load-back, no rank attempts it. Note the floor makes the local
        # eviction retry unreachable rather than merely unlikely -- the floor
        # is a MIN, so `floor >= kv_tokens` implies this rank's own
        # availability is >= kv_tokens too. None (single rank, or pools that
        # agree) keeps the live local path exactly as it was.
        # #1045 ONE BRANCH. The memory-axis decision -- "is there room for this
        # load-back" -- is now taken from the GROUP floor and from nothing else.
        #
        # WHAT WAS HERE AND WHY IT HAD TO GO. A second, rank-local branch ran
        # whenever `uniform_avail_floor` was None, guarded only by
        # `pp_load_back_told is not None`. That made a correctness watchman on
        # the memory axis depend on the DELIVERY signal of a different axis: the
        # moment the extent stopped being delivered, this fell back to deciding
        # from THIS rank's own free space -- the quantity
        # `rank_gpu_memory_mib=[31800,18800,19800]` makes unequal by
        # configuration. The roomy rank loads the prefix back, the tight rank
        # gives up, the device trees stop being replicas, and the next TP
        # collective is entered with rank-dependent token counts. That is the
        # 21:52:25 wedge, and it is measured, not hypothetical.
        #
        # The floor is now published on EVERY iteration and on every path
        # (scheduler.py `_publish_uniform_evict_floor`, including the
        # single-rank path, where it carries the local value because that IS
        # the group min for a group of one). So `floor is None` here can no
        # longer mean "pools happened to agree" or "single rank" -- it can only
        # mean nobody published, which is a broken construction.
        #
        # CRASH, NOT FALLBACK. A rank that cannot answer this question from a
        # group number must not answer it from a local one: that is precisely
        # how the ranks stop agreeing, and a compensating local answer is the
        # defect class, not the remedy. Both numbers are named so the crash is
        # a measurement.
        floor = getattr(self, "uniform_avail_floor", None)
        if floor is None:
            raise RuntimeError(
                "#1045 UNIFORM FLOOR ABSENT AT LOAD-BACK: this rank must decide "
                f"whether {kv_tokens} token(s) of host load-back fit, and the "
                "group availability floor was never published for this "
                "iteration. Deciding from this rank's own free space is exactly "
                "the rank-divergent path that killed the 21:52:25 boot "
                "(rank_gpu_memory_mib makes local availability unequal by "
                "configuration), so this refuses loudly instead. The floor is "
                "published unconditionally by Scheduler._publish_uniform_evict_"
                "floor; its absence here is a construction violation, not a "
                "runtime condition. "
                f"rid={getattr(req, 'rid', None)} kv_tokens={kv_tokens} "
                f"local_available={self.token_to_kv_pool_allocator.available_size()}"
            )
        if floor < kv_tokens:
            self.dec_lock_ref(best_match_node, ancestor_lock_params)
            self.dec_host_lock_ref(best_match_node, host_anchor_params)
            return False

        # Load H→D
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        device_indices = self.cache_controller.load(
            host_indices=kv_xfer.host_indices,
            node_id=best_match_node.id,
            extra_pools=aux_xfers or None,
        )

        self.dec_lock_ref(best_match_node, ancestor_lock_params)
        if device_indices is None:
            self.dec_host_lock_ref(best_match_node, host_anchor_params)
            return False

        # Commit: each component gets only its own transfers
        kv_xfer.device_indices = device_indices
        self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            best_match_node,
            CacheTransferPhase.LOAD_BACK,
            [kv_xfer],
        )
        for node in kv_xfer.nodes_to_load or ():
            self._record_store_event(node, medium=StorageMedium.GPU)
        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                best_match_node,
                CacheTransferPhase.LOAD_BACK,
                xfers,
            )

        self._update_evictable_leaf_sets(best_match_node)
        self.ongoing_load_back[best_match_node.id] = _OngoingLoadBack(
            best_match_node,
            self.inc_lock_ref(best_match_node).to_dec_params(),
            host_anchor_params,
        )

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return True

    def _build_sidecar_transfers(
        self,
        phase: CacheTransferPhase,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list[PoolTransfer]:
        transfers: list[PoolTransfer] = []
        for spec in self.sidecar_pool_specs:
            if spec.indices_from_pool == PoolName.KV:
                indices_source = kv_xfer
            else:
                source_component = {
                    PoolName.SWA: ComponentType.SWA,
                    PoolName.MAMBA: ComponentType.MAMBA,
                }.get(spec.indices_from_pool)
                if source_component is None:
                    raise AssertionError(
                        f"Unsupported sidecar indices source pool "
                        f"{spec.indices_from_pool}."
                    )
                matching_sources = comp_xfers.get(source_component, ())
                if not matching_sources:
                    continue
                indices_source = matching_sources[0]
                if indices_source.name != spec.indices_from_pool:
                    raise AssertionError(
                        f"Sidecar indices source pool {spec.indices_from_pool} "
                        f"resolved to {indices_source.name} during {phase}."
                    )

            indices = (
                indices_source.device_indices
                if phase == CacheTransferPhase.BACKUP_HOST
                else indices_source.host_indices
            )
            if indices is None or len(indices) == 0:
                continue
            transfers.append(
                PoolTransfer(
                    name=spec.pool_name,
                    keys=indices_source.keys,
                    hit_policy=spec.hit_policy,
                    indices_from_pool=spec.indices_from_pool,
                )
            )
        return transfers

    def _inc_hit_count(
        self,
        node: UnifiedTreeNode,
        chunked: bool = False,
        force_host_write_through: bool = False,
    ) -> None:
        """Increment hit count; trigger write_backup when threshold reached.

        ``force_host_write_through`` marks a hand-off insert (see
        ``requests_forced_host_write_through``): the node reaches the host tier
        regardless of hit count and of the write policy, because the donated
        device slots are freed by the same finish.
        """
        if node.evicted:
            return
        write_back_policy = (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        )
        # #1028 CHUNK PUBLISH -- A DELIBERATE DEVIATION FROM UPSTREAM'S
        # "skip the hit count update for chunked requests", NAMED AS SUCH.
        #
        # Upstream (`hiradix_cache.py:_inc_hit_count`) skips the host backup of
        # every chunked-prefill node and publishes at `cache_finished_req`
        # instead. On pure attention that loses nothing: the finished request's
        # node covers every chunk it was split into.
        #
        # It loses EVERYTHING here, and it was measured losing it (boot
        # `boot_855_1028fence`, 2026-08-30). Two fork properties break the
        # upstream premise:
        #   1. Drain-and-flip: a chunked prefill interrupted by a phase flip
        #      never reaches `cache_finished_req` at all (#856 removed the
        #      carry, so the flip DISCARDS it). Nothing is ever published, so
        #      the re-admission recomputes the whole prompt.
        #   2. GDN hybrid: a mamba/recurrent state is only valid AT one token
        #      position. The per-chunk node created here already carries a
        #      donated state (`MambaComponent.prepare_for_caching_req`, the
        #      `is_finished=False` branch) -- the anchor EXISTS on the device
        #      and this early return is the only reason it never reaches the
        #      host or the storage tier.
        # Measured consequence of the upstream form on that boot: exactly 11
        # host backups per rank in the whole run (`#969H BACKUP` n=1..11 on
        # PP0/PP1/PP2 at identical timestamps), all 11 with a mamba value, so
        # storage held 11 `.mamba` anchors -- one per FINISHED request -- and
        # a 13179-token prompt found its deepest anchor at 3072 and recomputed
        # the remaining 10107 tokens.
        #
        # Publishing the chunk restores the standing no-double-prefill bound
        # (`kein-doppel-prefill`, verbatim user order: at most ONE HiCache
        # chunk of loss) and is what `mamba-per-knoten-nicht-gitter` asks for
        # in its own words -- "states per radix node/chunk like KV pages" --
        # at the one link where the per-chunk state was being dropped. That
        # law also waives write volume as a counter-argument explicitly.
        #
        # Gated structurally, not by a flag: without a mamba tier or without a
        # storage tier there is no anchor to publish and the path stays
        # byte-identical to upstream.
        chunk_publish = (
            chunked
            and not write_back_policy
            and self._chunk_anchor_publish_enabled()
        )
        if (chunked and not chunk_publish) or write_back_policy:
            if not force_host_write_through:
                return
        else:
            node.hit_count += 1
            if chunk_publish:
                self._note_chunk_publish()
        if (
            self.cache_controller is not None
            and not node.backuped
            and (
                force_host_write_through
                or node.hit_count >= self.write_through_threshold
            )
        ):
            self.write_backup(node)

    def write_backup_storage(self, node: UnifiedTreeNode) -> None:
        if (
            not self.enable_storage
            or self.cache_controller is None
            or not node.backuped
        ):
            return

        # Weighted uneven-DCP owner mode (task #60): each rank's host pool only
        # holds real data for the tokens it OWNED at backup time (owner rule on
        # the GLOBAL device slot). Pass a per-page owner mask so _page_backup
        # writes exactly those pages to the rank-shared L3 page files.
        kv_page_owner_mask = None
        owner_ctx = self.cache_controller._dcp_owner_ctx()
        if owner_ctx is not None:
            device_value = node.component_data[BASE_COMPONENT_TYPE].value
            if device_value is None:
                # Ownership can no longer be derived (node already device-
                # evicted). Skipping is safe: batch_exists prefix semantics
                # just truncate at the first missing page, identically on
                # every rank (rank-shared file names).
                logger.warning(
                    "[uneven-dcp hicache] skipping storage backup of node %d: "
                    "device indices gone, page ownership unknown.",
                    node.id,
                )
                return
            S, lo, hi = owner_ctx
            off = device_value.to(torch.int64) % S
            # page_size == 1 is enforced at storage attach; index per page.
            kv_page_owner_mask = (
                ((off >= lo) & (off < hi))[:: self.page_size].contiguous().cpu()
            )

        prefix_keys = None
        if self.hicache_storage_pass_prefix_keys:
            prefix_keys = node.get_prefix_hash_values(node.parent)

        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            transfers = comp.build_hicache_transfers(
                node,
                CacheTransferPhase.BACKUP_STORAGE,
            )
            if transfers:
                comp_xfers[comp.component_type] = transfers

        kv_xfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=node.component_data[BASE_COMPONENT_TYPE].host_value,
            keys=node.hash_value,
        )
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_STORAGE, kv_xfer, comp_xfers
        )
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)

        operation_id = self.cache_controller.write_storage(
            node.component_data[BASE_COMPONENT_TYPE].host_value,
            node.key.token_ids,
            node.hash_value,
            prefix_keys,
            extra_pools=aux_xfers or None,
            kv_page_owner_mask=kv_page_owner_mask,
        )
        self.ongoing_backup[operation_id] = (
            node,
            self.inc_host_lock_ref(node).to_dec_params(),
        )
        # #810: the DRAIN phase begins here. The host lock keeps these tokens
        # resident until the backup acks, so they are the bytes a staging tier
        # is sized from; the charge cannot be refused (the page is already on
        # the host) but it must be counted, or the next admission decides
        # against an occupancy that hides the whole drain queue. Keyed by the
        # STORAGE OPERATION, so a split node's fragments -- several backups out
        # of one write-through ack -- each carry and retire their own charge.
        if self.staging_write_ring is not None:
            self.staging_write_ring.occupy(
                operation_id,
                len(node.component_data[BASE_COMPONENT_TYPE].host_value),
            )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: UnifiedTreeNode,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
        locally_eligible: bool = True,
    ) -> None:
        if not self.enable_storage or self.cache_controller is None:
            return

        # #580: decide the MODE before any rank-local predicate runs. Under
        # `symmetric` the participation vote below is the group's decision
        # point, so nothing between here and it may `return` -- see the
        # eligibility comment further down.
        symmetric = self._hicache_prefetch_symmetric()

        extra_key = last_host_node.key.extra_key if last_host_node.key else None
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        # RANK-LOCAL predicates, every one of them. `locally_eligible` carries
        # the caller's own local gate (Scheduler._prefetch_kvcache tests
        # `last_host_node.backuped`, i.e. "full KV in THIS rank's host pool");
        # `prefetch_rate_limited()` reads the per-rank prefetch_tokens_occupied
        # counter. Under uneven DCP the host pools -- and therefore both -- drift
        # apart across ranks.
        #
        # #580: these MUST NOT gate entry into the collective. They did until
        # 2026-08-05, on the (false) assumption recorded here that the gate was
        # rank-symmetric because _symmetrize_prefetch_capacity symmetrizes the
        # capacity LIMIT. It symmetrizes the limit, never the occupancy counter
        # and never `backuped`. A rank that tripped one of them returned before
        # the vote, so its peers stood in a collective it never entered: TP0
        # posted the 4-byte vote while TP1/TP2 had moved on to the 64-byte
        # kv-pressure consensus on the same gloo group, and gloo aborted TP0
        # with "op.preamble.length <= op.nbytes. 16 vs 4".
        #
        # So under `symmetric` local ineligibility only LOWERS THIS RANK'S VOTE;
        # participation itself is unconditional and the payload shape is fixed.
        # #915: SAY WHICH TERM DECLINED, because this conjunction is three
        # different verdicts wearing one boolean and none of them was counted.
        #
        # The 0826 window attempted a prefetch on 264 of 675 census-sampled
        # match walks. The other 411 declined HERE, silently: there is no
        # counter, no log line, and no way to tell from the boot which of the
        # three terms fired. That is the #914 shape one module over -- blame
        # without a defect -- and the three point at unrelated remedies:
        #   ANCHOR    the caller's local gate (`last_host_node.backuped`, with
        #             the root admitted on purpose at scheduler.py:4933)
        #   TOO_SHORT fewer than `prefetch_threshold` (256) new tokens to fetch
        #   RATE      `prefetch_tokens_occupied >= prefetch_capacity_limit`,
        #             and that limit is `0.5 * mem_pool_host.size`
        #             (cache_controller.py:729) -- which across a phase flip is
        #             not one number. #905 measured the two host tiers at
        #             703472 rows (PP) and 30518 (TP), a 23x asymmetry, so the
        #             TP-phase budget is ~15259 tokens, under four prefetches
        #             of the 4096 this window actually completed.
        # A counter here decides that on the next boot instead of re-arguing it.
        #
        # COUNTED, NOT GATED. Nothing below changes which prefetches run; this
        # records the verdict the code was already reaching. Order matters only
        # for attribution, so the terms are evaluated in the order written and
        # the FIRST failing one is named -- a request can trip several, and
        # summing them would double-count the way `refused_tokens_by_component`
        # is documented to.
        if not locally_eligible:
            reason = "anchor"
        elif prefetch_length < self.prefetch_threshold:
            reason = "too_short"
        elif self.cache_controller.prefetch_rate_limited():
            reason = "rate_limited"
        else:
            reason = None
        eligible = reason is None
        _note_prefetch_gate(reason, prefetch_length)
        if not eligible and not symmetric:
            return

        anchor_lock_params = None
        host_indices = None
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        sidecar_xfers: list[PoolTransfer] = []
        alloc_failed = True
        if eligible:
            anchor_lock_params = self.inc_host_lock_ref(last_host_node).to_dec_params()
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
            if host_indices is None:
                self.evict_host(prefetch_length)
                host_indices = self.cache_controller.mem_pool_host.alloc(
                    prefetch_length
                )
            if host_indices is None and not symmetric:
                available_size = self.cache_controller.mem_pool_host.available_size()
                prefetch_length = available_size - (available_size % self.page_size)
                if prefetch_length >= self.prefetch_threshold:
                    prefetch_key = prefetch_key[:prefetch_length]
                    host_indices = self.cache_controller.mem_pool_host.alloc(
                        prefetch_length
                    )
                else:
                    self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                    return
            if host_indices is None and not symmetric:
                self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                return
            # NOTE: under `symmetric` we deliberately SKIP the truncation-retry so
            # prefetch_key length (hence prefetch_tokens_occupied and the
            # min_completed_tokens reduce) stays identical across ranks; a failed
            # full alloc leaves host_indices None and becomes a negative consensus
            # vote below rather than a per-rank early-return.

            alloc_failed = host_indices is None
            if host_indices is not None:
                for comp in self._components_tuple:
                    if comp.component_type == BASE_COMPONENT_TYPE:
                        continue
                    transfers = comp.build_hicache_transfers(
                        last_host_node,
                        CacheTransferPhase.PREFETCH,
                        token_ids=prefetch_key.token_ids,
                        prefetch_tokens=len(prefetch_key),
                        last_hash=last_hash,
                    )
                    if transfers == []:
                        # #1035 -- THE PREFETCH THAT VANISHES WITHOUT A WORD.
                        #
                        # An empty (not None) list is a component saying "I could
                        # not acquire my host resource" -- for MambaComponent,
                        # `_mamba_pool_host.alloc(1)` returned None twice, once
                        # before and once after an eviction pass, i.e. the host
                        # ANCHOR pool is exhausted. `alloc_failed` then votes this
                        # rank out of the prefetch entirely and the request is
                        # recomputed from scratch.
                        #
                        # That is the right SEMANTIC (a KV span without its anchor
                        # is unmatchable, so not prefetching beats publishing an
                        # unmatchable node), but until now it was completely
                        # SILENT: no counter, no line, indistinguishable in every
                        # log from "there was nothing in storage". Read-through
                        # measured 0 for the whole campaign while this branch,
                        # if hot, was the reason -- and no instrument could say
                        # so. Name it, count it, and print the pool occupancy
                        # that caused it, so the anchor-DENSITY step that follows
                        # is dimensioned against a measured exhaustion rate
                        # instead of an assumed one.
                        alloc_failed = True
                        self._1035_n = getattr(self, "_1035_n", 0) + 1
                        if self._1035_n <= 40 or self._1035_n % 256 == 0:
                            _hp = getattr(comp, "_mamba_pool_host", None)
                            try:
                                _avail = (
                                    _hp.available_size() if _hp is not None else -1
                                )
                            except Exception:  # noqa: BLE001 - diagnostic only
                                _avail = -1
                            try:
                                _size = int(getattr(_hp, "size", -1))
                            except Exception:  # noqa: BLE001 - diagnostic only
                                _size = -1
                            logger.warning(
                                "#1035 PREFETCH DROPPED (host anchor pool "
                                "exhausted) n=%d comp=%s req=%s prefetch_tokens=%d "
                                "host_anchor_avail=%s host_anchor_size=%s -- this "
                                "rank votes the prefetch DOWN; the prompt is "
                                "recomputed in full. Not a storage miss.",
                                self._1035_n,
                                comp.component_type,
                                req_id,
                                len(prefetch_key),
                                _avail,
                                _size,
                            )
                        break
                    if transfers:
                        comp_xfers[comp.component_type] = transfers
            if host_indices is not None:
                kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
                sidecar_xfers = self._build_sidecar_transfers(
                    CacheTransferPhase.PREFETCH, kv_xfer, comp_xfers
                )

        if symmetric:
            # Mechanism (1) -- participation symmetry. Per-rank host pools are
            # asymmetric under weighted DCP, so a raw alloc can succeed on some
            # ranks and fail on others. Registering ongoing_prefetch on only a
            # SUBSET makes check_prefetch_progress enter its _all_reduce_attn_groups
            # collectives (can_terminate_prefetch's MAX reduce and the
            # min_completed_tokens MIN reduce) on a mismatched set of ranks ->
            # NCCL deadlock. All-reduce "am I eligible AND did I fully allocate"
            # with MIN (logical AND): every rank registers the prefetch iff ALL
            # ranks could; otherwise none do, keeping the downstream collectives
            # matched. No async prefetch op exists yet, so a negative vote is a
            # clean local release with nothing to tear down.
            #
            # The (tag, -tag) head makes the payload SELF-IDENTIFYING: after a
            # MIN reduce, slot 0 is min(tag) and slot 1 is -max(tag), so
            # slot0 != -slot1 proves the ranks were not all in this collective.
            # Same-width traffic from another site is then a named error on
            # every rank instead of a short read some ranks silently accept.
            local_ok = 1 if (eligible and not alloc_failed) else 0
            vote = torch.tensor(
                [_PREFETCH_VOTE_TAG, -_PREFETCH_VOTE_TAG, local_ok], dtype=torch.int
            )
            self._all_reduce_attn_groups(
                vote,
                torch.distributed.ReduceOp.MIN,
                label="prefetch_participation_vote",
            )
            tag_lo, tag_hi = int(vote[0].item()), -int(vote[1].item())
            if tag_lo != tag_hi or tag_lo != _PREFETCH_VOTE_TAG:
                raise HiCacheCollectiveDesyncError(
                    "prefetch_participation_vote returned a foreign payload "
                    f"(tag min={tag_lo} max={tag_hi}, expected "
                    f"{_PREFETCH_VOTE_TAG} on every rank): the TP ranks were "
                    "not all inside this collective. Some rank issued a "
                    "different collective on the same group -- the #580 "
                    "failure -- and continuing would corrupt the vote."
                )
            if int(vote[2].item()) == 0:
                if host_indices is not None:
                    self.cache_controller.append_host_mem_release(
                        host_indices=host_indices,
                        extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
                    )
                if anchor_lock_params is not None:
                    self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                return
            # Positive consensus: every rank allocated -> all fall through to register.
        elif alloc_failed:
            self.cache_controller.append_host_mem_release(
                host_indices=host_indices,
                extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
            )
            self.dec_host_lock_ref(last_host_node, anchor_lock_params)
            return

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            prefetch_key,
            last_hash,
            prefix_keys,
            extra_pools=aux_xfers or None,
        )
        # DIAGNOSTIC ONLY (#905 window): stamp the host pool identity and its
        # clear-epoch AT REGISTRATION, i.e. at the instant these host slots were
        # allocated. The completion path below compares them against the pool it
        # actually frees against. Stamped on the operation rather than in
        # `_OngoingPrefetch` because that record is a NamedTuple and is unpacked
        # positionally in five places.
        try:
            _p = self.cache_controller.mem_pool_host
            _p = getattr(getattr(_p, "anchor_entry", None), "host_pool", None) or _p
            operation._host_pool_id_at_reg = id(_p)
            operation._host_pool_epoch_at_reg = int(getattr(_p, "_clear_epoch", 0))
        except Exception:  # noqa: BLE001 - a diagnostic may never break a path
            operation._host_pool_id_at_reg = None
            operation._host_pool_epoch_at_reg = None
        # #939: A RE-ISSUE MUST NOT CLOBBER THE RECORD IT REPLACES.
        #
        # `_prefetch_kvcache` re-runs this whole path when a retracted request
        # is re-queued (`_add_request_to_queue`), which is exactly what a phase
        # cutover produces -- so a pre-cutover record can still be registered
        # under this req_id when the post-cutover one arrives. The assignment
        # below used to overwrite it outright: its `host_indices` were lost to
        # every owner (nothing else holds them) and its `anchor_lock_params`
        # lock ref was never decremented, leaving the node PROTECTED forever.
        # The next cutover's drop then orphans exactly that node's device rows,
        # which is the #938 shape -- one request's allocation per cutover --
        # without any in-flight write-through being needed to explain it.
        #
        # Retire, do not reap. See `_retire_ongoing_prefetch`.
        self._retire_ongoing_prefetch(req_id)
        self.ongoing_prefetch[req_id] = _OngoingPrefetch(
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        )
        self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _retire_ongoing_prefetch(self, req_id: str) -> bool:
        """Displace the record under ``req_id``, terminated but NOT freed.

        FREEING HERE WOULD BE A USE-AFTER-FREE. Whether this operation's host
        span is safe to release is answered by `can_terminate_prefetch`, which
        checks `pool_transfers_done` and is COLLECTIVE (an all_reduce over the
        attention groups). `HiCacheController.terminate_prefetch` is only
        `operation.mark_terminate()` -- a flag, not a join -- so a span freed
        at this point can still be the destination the prefetch transfer thread
        is writing into. And running that collective here is not an option
        either: whether a clobber happens is a RANK-LOCAL condition, and this
        function already sits inside the #580 participation region, so a
        collective conditional on it is the #580 desync verbatim.

        So the record moves to `_retired_prefetch`, where the per-round drain --
        which is allowed to run collectives, and does -- reaps it.

        THE ORDERING QUESTION IS ANSWERED BY CONSTRUCTION, not by timing: the
        re-issue owns the `req_id` slot from here on, so a late completion for
        the displaced operation resolves through `check_prefetch_progress` to
        the NEW record and never touches the old one. The old one is reachable
        only from the retired list, i.e. only from the reap.
        """
        record = self.ongoing_prefetch.pop(req_id, None)
        if record is None:
            return False
        try:
            record.operation.mark_terminate()
        except Exception:  # noqa: BLE001 - a terminated flag may never break this
            pass
        self._retired_prefetch.append(record)
        attempts = self._retired_prefetch_attempts.get(req_id, 0) + 1
        self._retired_prefetch_attempts[req_id] = attempts
        logger.info(
            "#939 PREFETCH RE-ISSUED: req=%s displaced record retired (attempt "
            "%d); its host span is released by the reap, not here, because the "
            "transfer may still be reading it.",
            req_id,
            attempts,
        )
        if attempts >= _MAX_PREFETCH_REISSUES:
            self._retired_prefetch_recompute += 1
            logger.warning(
                "#939 RE-FETCH BUDGET SPENT: req=%s had %d re-fetches crossed "
                "by cutovers, recomputing. The prefix is not being served from "
                "storage for this request; a cutover cadence faster than a "
                "fetch completes will do this every time.",
                req_id,
                attempts,
            )
        return True

    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation) -> bool:
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation: PrefetchOperation) -> bool:
        if self.prefetch_stop_policy == "best_effort":
            return True

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self._prefetch_timeout_check_linear_func(
                operation
            )
        else:
            return True
        if (
            completed
            and getattr(operation, "pool_transfers", None)
            and not getattr(operation, "pool_transfers_done", True)
        ):
            can_terminate = False

        operation_terminated = operation.is_terminated()
        states = torch.tensor(
            [1 - int(can_terminate), int(operation_terminated)],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(
            states,
            torch.distributed.ReduceOp.MAX,
            label="can_terminate_prefetch",
        )
        can_terminate = states[0].item() == 0
        operation_terminated = states[1].item() == 1
        return can_terminate or operation_terminated

    def _rehome_stale_prefetch_span(
        self,
        req_id: str,
        stale_indices,
        n_tokens: int,
        stale_generation,
        hash_values=None,
        prefix_keys=None,
    ):
        """#939: re-home an already-fetched span into the CURRENT generation's pool.

        WHY THIS EXISTS. A prefetch opened under binding generation N completes
        after the request's own prefill flip, so it lands under N+1/N+2. That is
        not a race that can be tuned away: BOOT-MEASURED 2026-08-30 on
        boot_855_1039ingen, the request that triggers the prefetch also triggers
        the flips, so under drain-and-flip EVERY request spans a flip pair
        (generations 0,2,4,6,8 across five requests; a min_dwell of 60 s refused
        117581 flips and changed the pattern not at all). #937 therefore refused
        100 % of non-zero fetches, five of five, zero counterexamples.

        WHAT IS ACTUALLY STALE, and what is not. The fetched BYTES are keyed by
        content hash and are generation-independent -- they are as valid now as
        when they were read. What is stale is only the HOST SLOTS they were
        written into, which were minted by the pool of `stale_generation` and
        will be freed against it. So the correct move is not to discard the
        bytes and re-read them (#943b's re-issue, which loses the race with
        admission anyway), but to mint fresh slots under the CURRENT binding and
        copy the bytes across.

        THIS IS NOT A BYPASS OF #937 -- IT IS A NEW LEGITIMATE EXIT FROM IT.
        The form #937 was built against is measured and specific (0827 soak:
        every prompt >= 256 tokens returned non-deterministic garbage at
        temperature 0 because a span was adopted WHOLE out of a pool that had
        already been replaced). That form stays structurally impossible here:
        the caller adopts ONLY the tensor this function returns, and that tensor
        can only ever come from `dst_pool.alloc` on the pool bound RIGHT NOW.
        `stale_indices` is never returned, never adopted, and still travels the
        generation-stamped release to the pool that minted it. Every failure
        path below returns None, which leaves the #937 refusal exactly as it was.

        Returns the freshly minted indices, or None to decline (caller refuses).
        """
        # EVERY DECLINE BELOW CARRIES A NAMED REASON. Measured 2026-08-30
        # (boot_855_939reread): one refusal per rank (req=e1ad3aac, 4618 tokens)
        # could not be attributed to ANY counter, because several of these
        # branches used to `return None` in silence. A silent exit is the class
        # this campaign has already paid for twice; the reason is now always on
        # the record, even when the outcome is simply "nothing to do".
        def _decline(reason: str, detail: str = "") -> None:
            attr = f"_rehome_declined_{reason}"
            n = getattr(self, attr, 0) + 1
            setattr(self, attr, n)
            if n <= 5 or n % 256 == 0:
                logger.warning(
                    "#939 RE-HOME DECLINED (%s) req=%s tokens=%s "
                    "from_generation=%s%s -- #937 refusal stands. (%d so far.)",
                    reason,
                    req_id,
                    n_tokens,
                    stale_generation,
                    f" {detail}" if detail else "",
                    n,
                )
            return None

        if n_tokens <= 0:
            return _decline("empty_span")
        cc = self.cache_controller
        dst_pool = getattr(cc, "mem_pool_host", None)
        if dst_pool is None:
            return _decline("no_destination_pool")
        try:
            from sglang.srt.mem_cache.hicache_phase_binding import (
                current_generation,
                host_pool_for_generation,
            )

            src_pool = host_pool_for_generation(stale_generation)
        except Exception:  # noqa: BLE001
            logger.warning(
                "#939 RE-HOME DECLINED (binding_lookup_raised) req=%s: could not "
                "resolve the pool for generation %s.",
                req_id,
                stale_generation,
                exc_info=True,
            )
            return _decline("binding_lookup_raised")
        if src_pool is None:
            # The generation that minted the span no longer resolves to a pool.
            return _decline("source_pool_gone")
        if src_pool is dst_pool:
            # The stamp moved but the tier did not: nothing to re-home, and a
            # copy onto itself would be a no-op at best. Not an error -- but it
            # must still be VISIBLE, or it looks exactly like a lost span.
            return _decline("same_pool_object", "tier did not move")
        page = int(getattr(dst_pool, "page_size", 1) or 1)
        if int(getattr(src_pool, "page_size", page) or page) != page:
            # Different page geometry between the two tiers: a page-wise copy
            # would silently reinterpret bytes. Refuse; a wrong prefix is worse
            # than a missing one, which is the whole lesson of #937.
            return None

        # PER-PAGE GEOMETRY, BOOT-MEASURED (boot_855_939rehome, 2026-08-30):
        # equal `page_size` is NOT enough. All 15 copy attempts of that boot
        # raised
        #     RuntimeError: shape '[2, 16, 1, 4, 256]' is invalid for input of
        #     size 16384   (and 8192)
        # -- the source page carried HALF, or a QUARTER, of the destination's
        # elements. The PP-phase and TP-phase host tiers shard differently under
        # uneven DCP, so carrying a span across a flip is a RESHARD, not a
        # relocation. Compare the geometry up front and decline cheaply, rather
        # than allocating a span and then catching a reshape deep inside the
        # pool. The exception handler below stays as the backstop; this makes
        # the common case explicit and free.
        try:
            _src_page = src_pool.get_dummy_flat_data_page()
            _dst_page = dst_pool.get_dummy_flat_data_page()
            _geom_ok = (
                _src_page.numel() == _dst_page.numel()
                and _src_page.dtype == _dst_page.dtype
            )
        except Exception:  # noqa: BLE001
            _geom_ok = False
            _src_page = _dst_page = None
        # NOT a decline any more: when the geometry differs the span is RE-READ
        # by content key into the fresh slots instead of copied. That is the
        # #939 order's own form ("re-issue the read under the current
        # generation"), and it keeps the layout-correct reader as the only
        # writer of a host page -- a reshard mover beside it would be the
        # bespoke second mover `ein-job-ein-mover` forbids.
        if not _geom_ok:
            self._rehome_geometry_reshard = (
                getattr(self, "_rehome_geometry_reshard", 0) + 1
            )
            if self._rehome_geometry_reshard <= 5:
                logger.warning(
                    "#939 RE-HOME VIA RE-READ req=%s tokens=%d from_generation=%s: "
                    "source page %s elems vs destination page %s elems -- the "
                    "tiers shard differently, so this is a RESHARD, not a copy. "
                    "Re-reading by content key into the current tier. (%d so far.)",
                    req_id,
                    int(n_tokens),
                    stale_generation,
                    None if _src_page is None else _src_page.numel(),
                    None if _dst_page is None else _dst_page.numel(),
                    self._rehome_geometry_reshard,
                )
            if not hash_values:
                # No content keys for this span -> nothing to re-read against.
                self._rehome_declined_no_keys = (
                    getattr(self, "_rehome_declined_no_keys", 0) + 1
                )
                return None

        t0 = time.perf_counter()
        new_indices = dst_pool.alloc(n_tokens)
        if new_indices is None:
            self.evict_host(n_tokens)
            new_indices = dst_pool.alloc(n_tokens)
        if new_indices is None:
            self._rehome_declined_no_room = (
                getattr(self, "_rehome_declined_no_room", 0) + 1
            )
            logger.warning(
                "#939 RE-HOME DECLINED (no room) req=%s tokens=%d: the current "
                "generation's host pool could not seat the span even after "
                "evict_host. Falling through to the #937 refusal. (%d so far.)",
                req_id,
                int(n_tokens),
                self._rehome_declined_no_room,
            )
            return None
        if not _geom_ok:
            # RESHARD PATH: the canonical reader writes the pages, in the
            # destination tier's own layout, from the geometry-neutral store.
            n_pages = (n_tokens + page - 1) // page
            got = cc.reread_pages_into(
                list(hash_values)[:n_pages],
                new_indices,
                prefix_keys=prefix_keys,
                label=f"#939-reread:{req_id[:8]}",
            )
            if int(got) < int(n_tokens):
                # A partial re-read cannot be published: the caller inserts
                # `n_tokens` and a short span would advertise pages that were
                # never written. Hand the slots back and let #937 stand.
                try:
                    dst_pool.free(new_indices)
                except Exception:  # noqa: BLE001
                    pass
                self._rehome_reread_short = (
                    getattr(self, "_rehome_reread_short", 0) + 1
                )
                logger.warning(
                    "#939 RE-READ SHORT req=%s wanted=%d got=%d from_generation=%s: "
                    "slots returned, #937 refusal stands. (%d so far.)",
                    req_id,
                    int(n_tokens),
                    int(got),
                    stale_generation,
                    self._rehome_reread_short,
                )
                return None
            _via = "reread"
        else:
            _via = "copy"
        try:
            n_pages = 0 if not _geom_ok else 0
            if _geom_ok:
                for off in range(0, n_tokens, page):
                    dst_pool.set_from_flat_data_page(
                        int(new_indices[off]),
                        src_pool.get_data_page(int(stale_indices[off]), flat=True),
                    )
                    n_pages += 1
            else:
                n_pages = (n_tokens + page - 1) // page
        except Exception:  # noqa: BLE001
            # Hand the fresh slots straight back and decline. NEVER fall through
            # to adopting `stale_indices` -- that is precisely the #937 form.
            try:
                dst_pool.free(new_indices)
            except Exception:  # noqa: BLE001
                pass
            self._rehome_copy_failed = getattr(self, "_rehome_copy_failed", 0) + 1
            logger.warning(
                "#939 RE-HOME COPY FAILED req=%s tokens=%d from_generation=%s: "
                "fresh slots returned, #937 refusal stands. (%d so far.)",
                req_id,
                int(n_tokens),
                stale_generation,
                self._rehome_copy_failed,
                exc_info=True,
            )
            return None

        self._prefetch_span_rehomed = getattr(self, "_prefetch_span_rehomed", 0) + 1
        # THE RECONCILIATION INSTRUMENT (#939 guardrail 2): after this change,
        # `refused_stale + re_homed` must equal what `refused_stale` alone used
        # to be. If the two do not add up, a THIRD exit is running silently and
        # that is the finding, not the ratio.
        logger.warning(
            "#939 PREFETCH SPAN RE-HOMED n=%d via=%s req=%s tokens=%d pages=%d "
            "from_generation=%s to_generation=%s copy_ms=%.1f | reconcile: "
            "refused_stale=%d re_homed=%d sum=%d",
            self._prefetch_span_rehomed,
            _via,
            req_id,
            int(n_tokens),
            n_pages,
            stale_generation,
            current_generation(),
            (time.perf_counter() - t0) * 1e3,
            getattr(self, "_prefetch_insert_refused_stale", 0),
            self._prefetch_span_rehomed,
            getattr(self, "_prefetch_insert_refused_stale", 0)
            + self._prefetch_span_rehomed,
        )
        return new_indices

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            return True

        (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return True
        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        min_completed_tokens = completed_tokens
        hit_pages = operation.pool_storage_result.extra_pool_hit_pages
        if self.tp_world_size > 1:
            # Reduce full completed tokens together with the sidecar pools that
            # this prefetch actually transferred, in one all_reduce. The vector
            # spans the fixed PoolName universe, not the rank-local comp_xfers
            # set: build_hicache_transfers can yield a different set of pools
            # per rank, and a rank-dependent numel wedges the collective. Only
            # the pools this rank transferred are read back, so the local
            # semantics are unchanged whenever the sets do agree.
            sidecar_pools = [t.name for xfers in comp_xfers.values() for t in xfers]
            packed_list = [completed_tokens] + [0] * _POOL_SLOT_COUNT
            for p in sidecar_pools:
                packed_list[_pool_slot(p, 1)] = int(hit_pages.get(p, 0))
            packed = torch.tensor(packed_list, dtype=torch.int)
            self._all_reduce_attn_groups(
                packed,
                torch.distributed.ReduceOp.MIN,
                label="check_prefetch_progress",
            )
            min_completed_tokens = int(packed[0].item())
            for p in sidecar_pools:
                hit_pages[p] = int(packed[_pool_slot(p, 1)].item())

        fetched_key = prefetch_key[:min_completed_tokens]
        # #937: DO NOT PUBLISH A SPAN WHOSE TIER NO LONGER EXISTS.
        #
        # The two frees below are stamped with `operation.binding_generation`
        # (#905/#719) precisely because a prefetch opened under binding N can
        # complete after a cutover has rebound the host tier to N+1. This
        # insert sat between them consulting none of it, so the completion that
        # could not safely FREE its own slots was still allowed to ADVERTISE
        # them to every later match walk. One side of the axis was closed; this
        # is the other side, and it is the side that reaches the model.
        #
        # MEASURED (2j soak boot boot_2h_4e855cc80a_0827_1056.log, all three
        # ranks): every prompt at or above 256 tokens -- the HiCache storage
        # prefetch gate, `prefetch_threshold` in TOKENS -- returned garbage,
        # non-deterministic at temperature 0, while 255 and below were correct.
        # The prefetch completions enumerate the failing lengths exactly
        # (256, 257, 258, 260, 272, 284, 300) and every one reports
        # `matched=0 loaded=N refused=0` beside a `MOVED=True` free-site line:
        # the span was adopted whole, out of a pool that had already been
        # replaced. Under `phase_flip_purity=strict:1` every request crosses a
        # pp_to_tp cutover between prefill and re-admission, so this is the
        # common case on this rig, not a rare race.
        #
        # The refusal reuses #841's existing shape rather than inventing a
        # second one: `host_span_unclaimed` makes `unclaimed_to` the WHOLE
        # completed span, which routes it through the generation-stamped free
        # below to the pool that actually minted it -- so the span is neither
        # leaked nor freed against the wrong tier -- and makes
        # `loaded_from_storage` 0, so the metric reports the retention that
        # actually happened. Same authority as the free, no second stamp
        # scheme: `write_back_stamp_is_current` is the predicate
        # `append_host_mem_release` already consults.
        from sglang.srt.mem_cache.hicache_phase_binding import (
            write_back_stamp_is_current,
        )

        _stamp = getattr(operation, "binding_generation", None)
        _stale = not write_back_stamp_is_current(_stamp)
        # #939 RE-HOMING, tried BEFORE the refusal. The bytes are content-keyed
        # and still valid; only the slots holding them belong to a superseded
        # generation. `_rehome_stale_prefetch_span` mints fresh slots under the
        # CURRENT binding and copies the bytes there, returning None to decline
        # -- and on None the #937 refusal below runs exactly as it always did.
        # The stale indices are never adopted on either branch.
        _rehomed = (
            self._rehome_stale_prefetch_span(
                req_id,
                host_indices,
                int(min_completed_tokens),
                _stamp,
                hash_values=hash_value,
                prefix_keys=getattr(operation, "prefix_keys", None),
            )
            if _stale
            else None
        )
        if _stale and _rehomed is None:
            self._prefetch_insert_refused_stale = (
                getattr(self, "_prefetch_insert_refused_stale", 0) + 1
            )
            logger.warning(
                "#937 STALE PREFETCH INSERT REFUSED: req=%s %d token(s) fetched "
                "under binding generation %s, which is no longer current; the "
                "span is released to that generation's pool instead of being "
                "published to the tree. (%d so far.)",
                req_id,
                int(min_completed_tokens),
                _stamp,
                self._prefetch_insert_refused_stale,
            )
            insert_result = InsertResult(prefix_len=0, host_span_unclaimed=True)
            # #943b: OWED A FRESH FETCH, NOT A REVIVED ONE.
            #
            # The span above is released to the generation that minted it and is
            # gone. What the request has lost is its PREFIX, and the only
            # correct way to get it back is to fetch again under the binding
            # that is current now -- a NEW operation, a NEW stamp, NEW host
            # slots, out of the content-keyed store. Recording the req_id here
            # is the whole of the state this needs; nothing about the dead
            # operation is kept, because keeping it is what would tempt the
            # re-stamp that `StaleStampRewrite` refuses.
            #
            # The COUNT is reported, never gated (#943b design point 4). A cap
            # that refused a fourth attempt would be a rank-local predicate
            # deciding participation, which is the #580 shape this whole path
            # is arranged to avoid; the natural terminator is admission, which
            # takes the request out of the waiting queue the drain reads.
            self._reissue_pending[req_id] = self._reissue_pending.get(req_id, 0) + 1
            _n = self._reissue_pending[req_id]
            if _n > _MAX_PREFETCH_REISSUES:
                logger.info(
                    "#943b RE-FETCH BUDGET EXCEEDED: req=%s has been refused %d "
                    "time(s), above the reporting budget of %d. Reported, NOT "
                    "refused: a cap here would be a rank-local predicate "
                    "deciding collective participation (#580).",
                    req_id,
                    _n,
                    _MAX_PREFETCH_REISSUES,
                )
        else:
            # `_rehomed` is None on the normal (current-stamp) path and carries
            # freshly minted current-generation slots on the re-homed path. The
            # tree therefore only ever adopts slots that belong to the binding
            # that is live right now -- never `host_indices` when it is stale.
            _adopt = _rehomed if _rehomed is not None else host_indices
            insert_result = self._insert_helper_host(
                last_host_node,
                fetched_key,
                _adopt[:min_completed_tokens],
                hash_value[: min_completed_tokens // self.page_size],
            )
            if _rehomed is not None:
                # Whatever the tree did NOT adopt out of the re-homed span
                # belongs to the CURRENT generation's pool -- not to the stale
                # one that `host_indices` is routed to further below. Sending
                # these two spans down one route is exactly the head/tail drift
                # #905 was written to stop.
                from sglang.srt.mem_cache.hicache_phase_binding import (
                    current_generation as _cur_gen,
                )

                _new_unclaimed_to = (
                    min_completed_tokens
                    if insert_result.host_span_unclaimed
                    else insert_result.prefix_len
                )
                self.cache_controller.append_host_mem_release(
                    host_indices=_rehomed[:_new_unclaimed_to],
                    generation=_cur_gen(),
                )

        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                last_host_node,
                CacheTransferPhase.PREFETCH,
                xfers,
                insert_result=insert_result,
                pool_storage_result=operation.pool_storage_result,
            )

        # #841: the matched head was never adopted (the tree already had it),
        # and when the contiguous-backup law declined the insert the fetched
        # TAIL was not adopted either. Both are this rank's to release: no
        # tree node references them, so nothing else ever will.
        # #939: when the span was RE-HOMED, the tree references the fresh slots
        # and the ENTIRE stale span is unreferenced -- so all of it goes back to
        # the generation that minted it, regardless of how much of the re-homed
        # copy the tree adopted. Reading `insert_result` here (which now
        # describes the FRESH span) would under-free the stale one and leak it.
        unclaimed_to = (
            min_completed_tokens
            if (insert_result.host_span_unclaimed or _rehomed is not None)
            else insert_result.prefix_len
        )
        # DIAGNOSTIC ONLY (#905 window): the decisive datum. If the pool object
        # or its clear-epoch moved between registration and here, the span being
        # freed was minted under a bookkeeping state that no longer exists, and
        # the double-free is a CUTOVER LIFETIME defect rather than a
        # free-it-twice defect. Logged unconditionally (one line per completion,
        # this is a diagnostic boot) so the NEGATIVE case is visible too.
        try:
            _p = self.cache_controller.mem_pool_host
            _p = getattr(getattr(_p, "anchor_entry", None), "host_pool", None) or _p
            logger.warning(
                "#905 PREFETCH-COMPLETE free-site: req=%s unclaimed_to=%d "
                "completed=%d min_synced=%d | pool id now=%d epoch now=%d | "
                "at registration id=%s epoch=%s | MOVED=%s",
                req_id,
                int(unclaimed_to),
                int(completed_tokens),
                int(min_completed_tokens),
                id(_p),
                int(getattr(_p, "_clear_epoch", 0)),
                getattr(operation, "_host_pool_id_at_reg", None),
                getattr(operation, "_host_pool_epoch_at_reg", None),
                (
                    id(_p) != getattr(operation, "_host_pool_id_at_reg", id(_p))
                    or int(getattr(_p, "_clear_epoch", 0))
                    != int(getattr(operation, "_host_pool_epoch_at_reg", 0) or 0)
                ),
            )
        except Exception:  # noqa: BLE001 - a diagnostic may never break a path
            pass
        # #905 FIX. MEASURED MECHANISM (R6 diagnostic boot, 2026-08-26 18:15Z,
        # all three ranks, six of six completions):
        #
        #   free-site: unclaimed_to=49 | pool id now=138604341884544 epoch 2
        #              size 30518 | at registration id=138604342978576 epoch 3
        #              size 703472 | MOVED=True
        #   -> Double-free: 49 of 49 in range but not allocated, span [0, 48],
        #      free_slots=30518 (i.e. the pool being freed against holds NOTHING)
        #
        # The prefetch allocates its host slots from the PP-phase host tier
        # (703472 rows) and completes after a cutover has rebound
        # `mem_pool_host` to the TP-phase tier (30518 rows) -- a DIFFERENT pool
        # object. A raw `.free()` here therefore returns slots to a pool that
        # never handed them out. Because 49 < 30518 the indices are IN RANGE,
        # so the #718 index-axis guard (628d9705b1, orphaned off this train)
        # cannot see them: same root, the other side of the same axis.
        #
        # The route already exists and is already used by `_drain_revoke`
        # (below): `append_host_mem_release(..., generation=...)` sends a span
        # stamped with a superseded binding to `host_pool_for_generation`, i.e.
        # to the pool that minted it, instead of to whatever is bound now.
        # `PrefetchOperation` inherits that stamp from `StorageOperation`
        # (cache_controller.py:177, "the binding this operation was OPENED
        # under"), so nothing new has to be carried.
        #
        # The tail below moves to the same call for the same reason: head and
        # tail of one prefetch belong to one pool, and having them take two
        # different routes is how they drifted apart in the first place.
        #
        # REACHABILITY, and why this fired now and not before: `unclaimed_to`
        # is 0 for a prefetch whose fetched span is adopted whole, and
        # `free([])` is a no-op. It is 49 exactly when a component validator
        # declines the fetched head -- the #904 census on this same run reports
        # `verdict=refused refusers=MambaComponent:49`. The refusal does not
        # cause this defect; it is what makes a latent lifetime defect
        # reachable, which is why a quiet single-stream probe never hit it.
        _binding_generation = getattr(operation, "binding_generation", None)
        self.cache_controller.append_host_mem_release(
            host_indices=host_indices[:unclaimed_to],
            generation=_binding_generation,
        )
        self.cache_controller.append_host_mem_release(
            host_indices=host_indices[min_completed_tokens:completed_tokens],
            generation=_binding_generation,
        )
        self.dec_host_lock_ref(last_host_node, anchor_lock_params)
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        # #841: a declined insert loaded NOTHING into the tree. Reporting the
        # fetched tail as `loaded` would make the metric measure the transfer
        # instead of the retention, and a refusal would read as a hit.
        loaded_from_storage = (
            0
            if insert_result.host_span_unclaimed
            else min_completed_tokens - insert_result.prefix_len
        )
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage
        # #843: `refused` separates the TWO reasons this line can say loaded=0,
        # which are not the same finding and were indistinguishable at INFO.
        #
        #   refused=0  nothing was there to load -- the storage tier missed, or
        #              the tree already held the whole fetched span
        #              (matched == completed_synced). Arithmetic, not a defect.
        #   refused=1  the tail WAS fetched and the tree declined to adopt it,
        #              because the walk landed on a node carrying no host copy
        #              (#841's contiguous-backup law).
        #
        # Window 6 needed exactly this and could not get it: 339 prefetches all
        # reported loaded=0, and the only refusal marker was a logger.debug on
        # a boot running log_level='info', so it could never appear. That made
        # W-841's acceptance criterion 2 -- "loaded=0 everywhere AND the decline
        # line frequent" -- unmeasurable by construction. 315 of those 339 were
        # arithmetic and 24 were refusals; without this field that split costs
        # a log-and-code archaeology pass.
        logger.info(
            "HiCache prefetch success req=%s completed_local=%d completed_synced=%d matched=%d loaded=%d refused=%d tail_release=%d occupied=%d",
            req_id,
            completed_tokens,
            min_completed_tokens,
            insert_result.prefix_len,
            loaded_from_storage,
            int(insert_result.host_span_unclaimed),
            completed_tokens - min_completed_tokens,
            self.cache_controller.prefetch_tokens_occupied,
        )
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)
        return True

    @staticmethod
    def _req_id_digest(req_id: str) -> int:
        """A rank-independent 63-bit digest of a req_id, never 0.

        Deterministic across processes on purpose -- Python's own ``hash`` is
        salted per process and would make every rank disagree. 0 is reserved
        for "this rank has nothing to drain".
        """
        import hashlib

        raw = hashlib.sha256(req_id.encode("utf-8")).digest()[:8]
        return (int.from_bytes(raw, "big") & 0x7FFF_FFFF_FFFF_FFFF) or 1

    def drain_retired_prefetch(self) -> int:
        """Reap ONE agreed retired prefetch record. Call every round.

        UNIFORMITY IS ENFORCED HERE, NOT INFERRED. It would be easy to argue
        that the retired list is rank-uniform -- retirement follows a
        registration that only happens under the #580 positive consensus -- but
        this module refuses arguments of exactly that shape elsewhere
        (`_prefetch_done_for` RAISES rather than assume a request is in the
        replicated set), and a wrong inference here is a three-rank hang rather
        than a wrong answer. So both halves are enforced:

        * ORDER: candidates are taken in sorted req_id order, a canonical
          sequence every rank computes identically from the ids themselves.
          Insertion order is never consulted.
        * MEMBERSHIP: the ranks AGREE on the candidate before anyone reaps it,
          via one all_reduce. Each rank contributes ``[d, -d]`` for its own
          first candidate's digest (``[0, 0]`` when it has none) and reduces
          with MIN, which yields ``min(d)`` and ``-max(d)`` in a single
          collective. Reaping proceeds only when ``min == max != 0``, i.e.
          every rank named the same record. A rank that has not retired it yet
          contributes 0, the agreement fails, and the record simply waits for
          the next round -- it never hangs, and it is never reaped on one rank
          only.

        THE COLLECTIVE IS UNCONDITIONAL. Running it "only in rounds with a
        non-empty list" would make participation depend on a rank-local
        predicate, which is the #580 failure this design exists to avoid. An
        empty list contributes zeros and the reduce is a no-op, so the common
        case costs one two-element all_reduce per round. Measure before
        optimising: correctness first, and piggy-backing this onto an existing
        round collective is a micro-optimisation, not a fix.

        Returns the number of records reaped (0 or 1).
        """
        if self.cache_controller is None:
            return 0

        # Sort by the req_id we can recover from the operation, not by position.
        candidates = sorted(
            self._retired_prefetch,
            key=lambda rec: getattr(rec.operation, "request_id", "") or "",
        )
        local = candidates[0] if candidates else None
        digest = (
            self._req_id_digest(getattr(local.operation, "request_id", "") or "")
            if local is not None
            else 0
        )

        # #943: RIDE THIS REDUCE TO ANSWER THE ONE QUESTION THE RE-ISSUE NEEDS.
        #
        # Making #937's refusal reachable for a re-issue means re-registering a
        # prefetch from a per-round hook, and that is only legal if the refusal
        # VERDICT is rank-uniform. It looks uniform -- the stamp is taken at a
        # registration that happens for the same request on every rank, and the
        # generation is advanced by the cutover rebind, which runs on every rank
        # -- but that is reasoning, and `_prefetch_done_for` already spells out
        # what reasoning costs here: a collective entered on a subset of ranks
        # is the #580 failure, "refused rather than risked".
        #
        # So it is MEASURED, and measured for free: this reduce already runs
        # every round, so the running refusal total rides along as two more
        # elements. All ranks build the same numel because they run this same
        # line, and MIN over [n, -n] yields min and max in one pass -- the shape
        # the digest half already uses.
        #
        # AN INSTRUMENT, NOT A GATE. It never changes what the drain does; a
        # divergence is reported and the drain proceeds exactly as before. The
        # re-issue is NOT built on top of this yet, deliberately: this line has
        # to report from metal first, which is the same discipline #938 Stage A
        # used before its own hypothesis was allowed to become a fix.
        stale_refusals = int(getattr(self, "_prefetch_insert_refused_stale", 0))
        vote = torch.tensor(
            [digest, -digest, stale_refusals, -stale_refusals], dtype=torch.int64
        )
        self._all_reduce_attn_groups(
            vote, torch.distributed.ReduceOp.MIN, label="drain_retired_prefetch"
        )
        agreed_min = int(vote[0].item())
        agreed_max = -int(vote[1].item())
        refusals_min = int(vote[2].item())
        refusals_max = -int(vote[3].item())
        if refusals_min != refusals_max:
            self._stale_refusal_divergences = (
                getattr(self, "_stale_refusal_divergences", 0) + 1
            )
            logger.error(
                "#943 STALE-REFUSAL VERDICT DIVERGES ACROSS RANKS: this rank has "
                "refused %d stale prefetch insert(s), the group spans [%d, %d]. "
                "The #937 verdict is therefore NOT rank-uniform, and a re-issue "
                "driven from it would enter a collective on a subset of ranks -- "
                "the #580 failure. Do not wire the re-issue on this reading. "
                "(divergence %d)",
                stale_refusals,
                refusals_min,
                refusals_max,
                self._stale_refusal_divergences,
            )
        elif stale_refusals > 0 and not getattr(self, "_stale_refusal_agreed_once", 0):
            self._stale_refusal_agreed_once = 1
            _world = self._attn_reduce_world()
            logger.info(
                "#943 STALE-REFUSAL VERDICT %s at %d refusal(s) -- %s",
                (
                    "AGREES ACROSS RANKS"
                    if _world > 1
                    else "SINGLE-RANK (NO PEERS REDUCED)"
                ),
                stale_refusals,
                (
                    "the first affirmative reading that the #937 verdict is "
                    "rank-uniform. One agreeing round is not the property; the "
                    "absence of a DIVERGES line over a whole boot is."
                    if _world > 1
                    else "#1028: the reduce behind this comparison covered ONE "
                    "rank, so min == max holds by arithmetic and says NOTHING "
                    "about the peers. Not an agreement -- read it as unmeasured."
                ),
            )
        if agreed_min == 0 or agreed_min != agreed_max:
            # No agreement this round (or nothing anywhere to drain). The
            # record keeps its host span and its lock ref until every rank
            # names it -- visible meanwhile as #938 protected residue, which
            # is why that counter is expected to RETURN rather than grow.
            return 0
        if local is None or digest != agreed_min:
            return 0

        self._retired_prefetch.remove(local)
        if not self.can_terminate_prefetch(local.operation):
            # Still in flight: put it back and try again next round. This is
            # the one question that decides whether the span is safe to free,
            # and it is asked here -- where running a collective is legal --
            # rather than at the retire site, where it is not.
            self._retired_prefetch.append(local)
            return 0

        req_id = getattr(local.operation, "request_id", "") or ""
        completed_tokens, _ = self.cache_controller.terminate_prefetch(local.operation)
        generation = getattr(local.operation, "binding_generation", None)
        # The whole span, by the #905/#911 route: this record published
        # NOTHING to the tree, so every row it holds is unclaimed and goes back
        # to the pool that minted it, not to whatever is bound now.
        self.cache_controller.append_host_mem_release(
            host_indices=local.host_indices,
            generation=generation,
        )
        self.dec_host_lock_ref(local.anchor_node, local.anchor_lock_params)
        self.cache_controller.prefetch_tokens_occupied -= len(local.prefetch_key)
        self._retired_prefetch_reaped += 1
        logger.info(
            "#939 RETIRED PREFETCH REAPED: req=%s %d row(s) returned to "
            "binding generation %s and its anchor lock released (%d reaped, "
            "%d still retired).",
            req_id,
            int(local.host_indices.numel()),
            generation,
            self._retired_prefetch_reaped,
            len(self._retired_prefetch),
        )
        return 1

    def _release_retired_prefetch_local(self) -> int:
        """Release every retired prefetch record on THIS rank. Detach only.

        #966. `drain_retired_prefetch` is the steady-state reap and it is
        collective by necessity: it runs while the peers run, so WHICH record is
        reaped has to be agreed before anyone acts. A detach can neither use it
        nor imitate it -- a detach may not depend on a collective its peers may
        already have left, which is the rule `detach_storage_backend` above
        already states for its own control-queue drain.

        WHAT REPLACES THE AGREEMENT IS NOT AN ARGUMENT ABOUT UNIFORMITY. The
        retire site (`_retire_ongoing_prefetch`) refuses to free because
        `mark_terminate()` is a flag and not a join, so the prefetch transfer
        thread may still be writing into the span; the collective exists to
        answer that one question. By the time this runs, the caller has already
        JOINED those threads (`HiCacheController.detach_storage_backend` ->
        `_stop_storage_threads`, which raises rather than returning on failure),
        so there is no writer left to race and nothing a peer could tell this
        rank that it does not already know.

        ONLY THE RETIRED LIST, AND THIS IS THE LOAD-BEARING RESTRICTION.
        `ongoing_prefetch` is deliberately untouched. A retired record published
        NOTHING to the tree, so every row it holds is unclaimed (the same
        argument `drain_retired_prefetch` makes at its own free); a LIVE record's
        span may still be adopted into the tree by `check_prefetch_progress`, so
        freeing it here would be a use-after-free now and a double free later --
        strictly worse than the leak this repairs.

        Returns the number of records released.
        """
        cc = self.cache_controller
        if cc is None:
            return 0

        released = 0
        rows = 0
        while self._retired_prefetch:
            record = self._retired_prefetch.pop()
            # The whole span, by the #905/#911 route, exactly as the reap does
            # it: a record retired before a cutover names slots from the pool
            # that MINTED them, not from whatever is bound now.
            cc.append_host_mem_release(
                host_indices=record.host_indices,
                generation=getattr(record.operation, "binding_generation", None),
            )
            self.dec_host_lock_ref(record.anchor_node, record.anchor_lock_params)
            cc.prefetch_tokens_occupied -= len(record.prefetch_key)
            if cc.prefetch_tokens_occupied < 0:
                cc.prefetch_tokens_occupied = 0
            released += 1
            rows += int(record.host_indices.numel())

        total = getattr(self, "_retired_prefetch_released_detach", 0) + released
        self._retired_prefetch_released_detach = total
        # UNCONDITIONAL, including the zero. A line emitted only on a find
        # cannot distinguish "nothing was stranded" from "this code never ran",
        # which is the #962a blind-probe shape; the absence of this line in a
        # boot that detached is then read as an all-clear. Not reset by
        # `_reset_full`, so the total spans a whole boot across flips.
        logger.info(
            "#966 RETIRED PREFETCH RELEASED AT DETACH: released=%d rows=%d "
            "(cumulative=%d). These records' only reap sits under the "
            "enable_hicache_storage gate this detach clears, so anything left "
            "here would have no reachable free path at all.",
            released,
            rows,
            total,
        )
        return released

    def take_agreed_reissue(self, local_candidates: Sequence[str]) -> Optional[str]:
        """One req_id every rank agrees is owed a fresh fetch, or ``None``.

        #943b. The re-issue itself runs through the scheduler's ordinary
        ``_prefetch_kvcache``, so it inherits the whole #580 participation
        machinery instead of re-inventing it. What this method owns is the ONE
        decision that machinery cannot make for it: WHICH request, agreed by
        every rank before any of them acts.

        THE HAZARD, AND WHY A LIVE MEASUREMENT IS NOT ENOUGH. Boot a810ef69ec
        measured the #937 refusal verdict as rank-uniform (DIVERGES 0, AGREES 3,
        over 111 cutovers and 48 refusals). That is evidence about one boot on
        one rig, and building a collective on it would make the uniformity a
        LOAD-BEARING ASSUMPTION verified nowhere in the code. So the agreement
        is taken here, every time, by the same shape ``drain_retired_prefetch``
        already uses: MIN over ``[d, -d]`` yields the group min and max in one
        pass, and only ``min == max != 0`` is an agreement.

        The candidate set is the intersection the caller can see -- requests
        that are BOTH owed a re-fetch AND present in this rank's waiting queue.
        Voting on "owed" alone would agree on a request some rank cannot act on,
        and that rank would then sit out the collective its peers entered, which
        is the failure this is arranged to prevent rather than a smaller version
        of it.

        DISAGREEMENT IS A WAIT, NOT A LOSS. The pending entry survives, so the
        next round votes again. A request whose peers never agree simply keeps
        its recompute -- today's behaviour -- and the disagreement is counted so
        that "the re-issue never fires" and "the re-issue fires and does not
        help" stay distinguishable.
        """
        pending = [r for r in local_candidates if r in self._reissue_pending]
        pending.sort()
        local = pending[0] if pending else ""
        digest = self._req_id_digest(local) if local else 0

        vote = torch.tensor([digest, -digest], dtype=torch.int64)
        self._all_reduce_attn_groups(
            vote, torch.distributed.ReduceOp.MIN, label="take_agreed_reissue"
        )
        agreed_min = int(vote[0].item())
        agreed_max = -int(vote[1].item())

        if agreed_min == 0:
            # Nothing owed anywhere, or some rank has nothing to offer. Not a
            # disagreement worth counting: an empty round is the common case.
            return None
        if agreed_min != agreed_max:
            self._reissue_disagreements += 1
            if (
                self._reissue_disagreements <= 3
                or self._reissue_disagreements % 100 == 0
            ):
                logger.warning(
                    "#943b RE-ISSUE NOT AGREED: this rank offered %r (digest %d) "
                    "and the group spans [%d, %d], so NO rank re-issues this "
                    "round. The request keeps its recompute and the entry keeps "
                    "its place in the queue. Entering the re-registration on a "
                    "split verdict is the #580 failure and is the one thing this "
                    "gate exists to refuse. (disagreement %d)",
                    local,
                    digest,
                    agreed_min,
                    agreed_max,
                    self._reissue_disagreements,
                )
            return None
        if digest != agreed_min:
            # Agreement reached on a req_id this rank did not nominate. It
            # cannot act, so it must not: returning None here keeps it out of a
            # collective its peers are entering under a different key.
            return None

        self._reissue_pending.pop(local, None)
        self._reissue_taken += 1
        logger.info(
            "#943b PREFETCH RE-ISSUED: req=%s %s; fetching "
            "again under the CURRENT binding generation (a new operation, new "
            "host slots, fresh bytes from the content-keyed store -- the "
            "refused span was released to its minting generation and is not "
            "revived). (%d re-issued, %d still owed.)",
            local,
            (
                "agreed by every rank"
                if self._attn_reduce_world() > 1
                else "chosen rank-locally (#1028: the agreement reduce covered "
                "ONE rank, so this is NOT a cross-rank agreement)"
            ),
            self._reissue_taken,
            len(self._reissue_pending),
        )
        return local

    def terminate_prefetch(self, req_id: str) -> None:
        if req_id not in self.ongoing_prefetch:
            return
        operation = self.ongoing_prefetch[req_id].operation
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def release_aborted_request(self, rid: str) -> None:
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        if rid not in self.ongoing_prefetch:
            return

        (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        self._barrier_attn_groups(label="release_aborted_request")
        self.dec_host_lock_ref(last_host_node, anchor_lock_params)
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(
            host_indices=host_indices[:completed_tokens],
            extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
        )
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

    def _drain_storage_control_queues_local(self) -> None:
        """Drain the storage control queues on THIS rank, no TP agreement.

        #872: this wrapper was missing here while both sibling caches
        (``HiRadixCache``, ``HiMambaRadixCache``) carried it, and the flip
        writeback fence finds its drain by exactly this name
        (``hicache_flip_writeback._await_storage_acks``). Duck-typed, so its
        absence was not an error: the fence returned "nothing acknowledged,
        everything outstanding" at once and waited none of its deadline. Boot
        ``w40_857strict``: ``acked=0`` on all 21 fence reports, three of them
        claiming ``deadline reached`` at ``elapsed=0.000s/2.000s`` -- a
        deadline reached after none of its two seconds, which is the early
        return and not a slow backend.

        ``None`` limits mean "drain everything on this rank". The steady-state
        path derives its counts from an all_reduce; a fence, a detach and a
        shutdown may not depend on a collective their peers may already have
        left -- issuing one inside the flip seam is the #630 wedge shape.
        """
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_backup=None,
            n_release=None,
            extra_release_counts=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        extra_release_counts: Optional[dict[PoolName, int]],
        log_metrics: bool,
    ) -> None:
        cc = self.cache_controller

        def _drain_queue(q: Queue[T], limit: Optional[int]) -> Iterator[T]:
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            drained = 0
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is None:
                    continue
                drained += 1
                (
                    last_host_node,
                    prefetch_key,
                    _host_indices,
                    _operation,
                    anchor_lock_params,
                    comp_xfers,
                ) = info
                # W35: this revoke may be drained AFTER a cutover, so the
                # slots it releases can belong to the binding the prefetch was
                # OPENED under. The operation carries that generation.
                cc.append_host_mem_release(
                    extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
                    generation=getattr(_operation, "binding_generation", None),
                )
                self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                cc.prefetch_tokens_occupied -= len(prefetch_key)
                if cc.prefetch_tokens_occupied < 0:
                    cc.prefetch_tokens_occupied = 0
            return drained

        def _drain_backup():
            drained = 0
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                drained += 1
                entry = self.ongoing_backup.pop(operation.id, None)
                if entry is not None:
                    node, lock_params = entry
                    self.dec_host_lock_ref(node, lock_params)
                # #810: the storage write acked -- this is the drain the
                # staging ring measures its residency against. Outside the
                # `entry is not None` arm on purpose: the charge is keyed by
                # the operation, so it retires whenever the operation does.
                if self.staging_write_ring is not None:
                    self.staging_write_ring.release(operation.id)
                if (
                    log_metrics
                    and self.enable_storage_metrics
                    and self.storage_metrics_collector is not None
                ):
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )
            return drained

        def _drain_release():
            host_indices_list = []
            released_tokens = 0
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
                released_tokens += len(host_indices)
            if host_indices_list:
                # #989 OWNERSHIP AT THE GIVE-BACK: FREE ONLY WHAT IS OWNED.
                #
                # This queue has FIFTEEN producers and this is their ONE
                # give-back, with no dedup and no ownership test -- the
                # fourth instance today of "many producers, one give-back,
                # no provenance" (#990 lock_ref, #991 mamba slot, #993
                # req-pool row, now the host region). Measured, boot 18
                # (ecedf3f791, 2026-08-28 22:57:26, plain PP=3 with the flip
                # PROVABLY off -- arming 0, flipdone 0): PP0 died on
                # `Double-free detected: slots not currently allocated:
                # [60934…60971+]` from this exact call.
                #
                # Two shapes reach here and both are handled: the same span
                # queued TWICE (duplicate within the batch) and a span queued
                # again AFTER it was already freed (already-free in the pool).
                # Freeing a slot that is not allocated is never correct, so
                # dropping those is not a heuristic -- it is the give-back
                # doing what it always should have done.
                #
                # THE PROVENANCE IS THE POINT, not the containment. Each
                # offender is named with the `module:lineno` that queued it
                # (#989 stamp in `append_host_mem_release`), so the NEXT
                # 90-second boot identifies the producer PAIR itself instead
                # of us choosing the likeliest of fifteen candidates.
                _idx = torch.cat(host_indices_list, dim=0)
                _cpu = _idx.detach().to("cpu", copy=False).flatten()
                _uniq, _counts = torch.unique(_cpu, return_counts=True)
                _dupes = _uniq[_counts > 1]
                _pool = cc.mem_pool_host
                _slot_used = getattr(_pool, "slot_used", None)
                if _slot_used is not None:
                    _unowned = _uniq[~_slot_used[_uniq]]
                else:
                    _unowned = _uniq[:0]
                if _dupes.numel() or _unowned.numel():
                    _n = getattr(self, "_989_release_conflicts", 0) + 1
                    self._989_release_conflicts = _n
                    if _n <= 8 or _n % 256 == 0:
                        _site = getattr(cc, "host_release_site", None)
                        _names = []
                        for _t in (_dupes, _unowned):
                            for _s in _t[:4].tolist():
                                _names.append(
                                    f"{_s}<-{_site(_s) if _site else '?'}"
                                )
                        logger.error(
                            "#989 HOST RELEASE CONFLICT occurrence=%d: %d slot(s) "
                            "queued twice, %d slot(s) already free. Freeing only "
                            "the owned, unique set (%d of %d). Offenders with the "
                            "site that queued them: %s. This queue has fifteen "
                            "producers and one give-back; a slot arriving here "
                            "unowned means two of them claimed the same span. "
                            "The W35 generation guard cannot see this -- it "
                            "routes by BINDING GENERATION, and these are two "
                            "current-generation batches.",
                            _n,
                            int(_dupes.numel()),
                            int(_unowned.numel()),
                            int((_uniq.numel() - _unowned.numel())),
                            int(_cpu.numel()),
                            ", ".join(_names) or "-",
                        )
                _to_free = _uniq if _slot_used is None else _uniq[_slot_used[_uniq]]
                if _to_free.numel():
                    cc.mem_pool_host.free(_to_free.to(_idx.device))
            return len(host_indices_list), released_tokens

        def _drain_extra_release():
            drained: dict[PoolName, tuple[int, int]] = {}
            if not extra_release_counts:
                return drained
            for pool_name, limit in extra_release_counts.items():
                release_queue = cc.extra_host_mem_release_queues.get(pool_name)
                if release_queue is None:
                    continue
                host_indices_list = []
                released_tokens = 0
                for host_indices in _drain_queue(release_queue, limit):
                    host_indices_list.append(host_indices)
                    released_tokens += len(host_indices)
                if host_indices_list:
                    # #718/#847: resolve through the queue's OWNING entry, not
                    # only through the currently bound tier. A phase rebind onto
                    # a narrower tier used to make this lookup return None, and
                    # the slots -- already dequeued -- were then dropped on the
                    # floor instead of freed. The drain is the last holder, so a
                    # miss here is a permanent host-slot leak, not a retry.
                    entry = cc.entry_for_extra_release(pool_name)
                    if entry is not None:
                        entry.host_pool.free(torch.cat(host_indices_list, dim=0))
                    else:
                        # Rate-limited: the drain runs every iteration, so an
                        # unbounded emitter here is the log-flood class.
                        n = getattr(self, "_drain_orphaned_releases", 0) + 1
                        self._drain_orphaned_releases = n
                        if n <= 3 or n % 200 == 0:
                            logger.error(
                                "#718/#847 DRAIN ORPHANED: %d host slot(s) of "
                                "pool '%s' were dequeued but name no known "
                                "entry; they cannot be freed and are leaked. "
                                "(%d so far.)",
                                released_tokens,
                                pool_name,
                                n,
                            )
                drained[pool_name] = (len(host_indices_list), released_tokens)
            return drained

        _drain_revoke()
        _drain_backup()
        _drain_release()
        _drain_extra_release()

    def drain_storage_control_queues(self) -> None:
        cc = self.cache_controller
        extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
        # The sidecar slots are laid out over the fixed PoolName universe, NOT
        # over list(extra_release_queues): that dict mirrors this rank's host
        # pools, which are asymmetric under uneven DCP, and a rank-dependent
        # numel is exactly what wedges the all_reduce below. _pool_slot raises
        # rather than silently dropping a pool that has no invariant slot.
        local_qsize_list = [
            cc.prefetch_revoke_queue.qsize(),
            cc.ack_backup_queue.qsize(),
            cc.host_mem_release_queue.qsize(),
        ] + [0] * _POOL_SLOT_COUNT
        for pool_name, release_queue in extra_release_queues.items():
            local_qsize_list[_pool_slot(pool_name, 3)] = release_queue.qsize()
        qsizes = torch.tensor(
            local_qsize_list,
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(
            qsizes,
            torch.distributed.ReduceOp.MIN,
            label="drain_storage_control_queues",
        )
        qsize_list = list(map(int, qsizes.tolist()))
        n_revoke, n_backup, n_release = qsize_list[:3]
        extra_release_counts = {
            pool_name: qsize_list[_pool_slot(pool_name, 3)]
            for pool_name in extra_release_queues
        }
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            extra_release_counts=extra_release_counts,
            log_metrics=True,
        )

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[dict[str, str]],
    ) -> None:
        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            attn_cp_rank, attn_cp_size = (
                self.cache_controller.get_attn_cp_rank_and_size()
            )
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
                "attn_cp_rank": attn_cp_rank,
                "attn_cp_size": attn_cp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            existing_collector = self.storage_metrics_collector
            if existing_collector is None:
                from sglang.srt.runtime_context import get_server_args

                storage_cls = resolve_collector_class(
                    get_server_args(),
                    STAT_LOGGER_ROLE_STORAGE,
                    StorageMetricsCollector,
                )
                self.storage_metrics_collector = storage_cls(labels=labels)
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                existing_collector.labels = labels
            else:
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )
        else:
            self.storage_metrics_collector = None

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) the storage backend at runtime (#545).

        Mirrors ``HiRadixCache.attach_storage_backend`` -- same validation, same
        named refusals, same "already attached to this backend is success,
        attached to a DIFFERENT backend is failure" rule -- with one addition
        this cache needs and HiRadixCache does not: the prefetch capacity must
        be re-symmetrized across ranks afterwards.

        THE CALLER IS THE WHOLE GROUP, and that is what makes this safe.
        ``_symmetrize_prefetch_capacity`` enters an all_reduce over the DCP/TP
        ranks, and its own guard says a rank-local early return "would leave
        the other ranks in the all_reduce with no partner". A single-rank
        attach would therefore hang or raise. It does not happen because
        ``attach_hicache_storage`` fans out through ``FanOutCommunicator`` to
        every rank and merges the results, so every rank runs this method or
        none does. The scheduler additionally refuses a non-idle scheduler by
        name before reaching here.
        """
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: "
                    f"{hicache_storage_prefetch_policy!r}. Expected one of {allowed}.",
                )
        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if self.cache_controller is None:
            return (
                False,
                "UnifiedRadixCache has no cache controller: hierarchical "
                "caching was not enabled at startup, and enabling it at "
                "runtime is a separate capability (the host pool and its "
                "threads are built at boot).",
            )

        # Already attached: same backend is success (policies may still be
        # updated), a DIFFERENT backend is a refusal rather than a silent
        # swap -- swapping would strand every page written under the old one.
        if self.enable_storage:
            current = getattr(self.cache_controller, "storage_backend_type", None)
            if current == storage_backend:
                return True, (
                    f"HiCache storage backend {storage_backend!r} is already attached."
                )
            return False, (
                f"A different HiCache storage backend is already attached "
                f"({current!r}); detach it before attaching {storage_backend!r}."
            )

        extra_config = None
        if storage_backend_extra_config_json:
            try:
                extra_config = json.loads(storage_backend_extra_config_json)
            except Exception as e:
                logger.exception("Failed to parse storage_backend_extra_config_json")
                return False, (
                    f"Failed to parse storage_backend_extra_config_json "
                    f"{storage_backend_extra_config_json!r}: {e}"
                )

        prefetch_threshold = getattr(self, "prefetch_threshold", 256)
        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                **self._get_hybrid_storage_attach_kwargs(),
            )
        except Exception as e:
            logger.exception(f"Failed to attach storage backend {storage_backend!r}")
            return False, f"Failed to attach storage backend {storage_backend!r}: {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=getattr(self, "prefetch_timeout_base", 3.0),
            prefetch_timeout_per_ki_token=getattr(
                self, "prefetch_timeout_per_ki_token", 0.25
            ),
            hicache_storage_pass_prefix_keys=getattr(
                self, "hicache_storage_pass_prefix_keys", False
            ),
            enable_storage=True,
            enable_storage_metrics=getattr(self, "_enable_metrics_flag", False),
            extra_metric_labels=getattr(self, "extra_metric_labels", None),
        )
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
        if hicache_write_policy is not None:
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )

        # AFTER the config is applied, and on every rank: the capacity limit is
        # derived from the MIN host-pool size across ranks, so it can only be
        # computed once each rank knows storage is on.
        self._symmetrize_prefetch_capacity()
        return True, "Attached HiCache storage backend successfully."

    def _get_hybrid_storage_attach_kwargs(self) -> dict:
        """Host pools for a multi-component (hybrid/GDN) controller.

        Mirrors the HiRadixCache helper of the same name. The MAMBA/state
        component is covered here rather than refused: the controller's
        ``mem_pool_host.entries`` spans every component it owns, which is the
        same set the boot-time attach passes.
        """
        cc = self.cache_controller
        entries = getattr(getattr(cc, "mem_pool_host", None), "entries", None)
        return {"host_pools": entries} if entries is not None else {}

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) the storage backend at runtime (#545).

        ORDER IS THE CONTRACT, and it is HiRadixCache's: drain the control
        queues BEFORE tearing the controller's threads down, or acks and
        releases can no longer be matched to their nodes and host pages and
        locks leak. Drained again afterwards to sweep whatever the shutdown
        itself produced.

        The drain here is LOCAL -- ``None`` limits mean "drain everything on
        this rank". The queue-size path used in the steady state derives its
        counts from an all_reduce, and a detach may not depend on a collective
        that its peers may already have left.

        A detached tier's pages become MISSES, never corruption: the keys are
        content-addressed, so a page that is no longer reachable is simply not
        found -- the same argument #703 uses for its drops.
        """
        if self.cache_controller is None:
            return False, "UnifiedRadixCache has no cache controller to detach."

        try:
            self._drain_storage_control_queues_impl(
                n_revoke=None,
                n_backup=None,
                n_release=None,
                extra_release_counts=None,
                log_metrics=False,
            )
            # Idempotent: ask the controller to clean up even when
            # enable_storage is already False, since that may be leftover
            # state from a partially-failed previous detach.
            self.cache_controller.detach_storage_backend()
            # #966: THE HOLDER THIS DOCSTRING'S CONTRACT NAMES BUT NEVER DRAINED.
            #
            # `_retired_prefetch` (#939) holds a host span, an anchor host lock
            # ref and a `prefetch_tokens_occupied` charge. Its ONLY release is
            # `drain_retired_prefetch`, whose only caller (scheduler.py:8400)
            # sits below `if not self.enable_hicache_storage: return {}`
            # (scheduler.py:8386) -- and THIS operation is what clears that flag
            # (scheduler.py:10820). So a record still retired when the detach
            # returns has no reachable free path at all: precisely the "host
            # pages and locks leak" this method's own docstring exists to
            # prevent. The #939 holder was added after that sentence was written
            # and was never folded into it.
            #
            # Nor does anything else close it: `_reset_full` REBINDS the list to
            # `[]` without releasing, and `cache_controller.reset()` zeroes the
            # prefetch charge only `if self.enable_storage`, which the call
            # above has just set False -- so the charge would outlive even a
            # full reset and throttle prefetch admission for the rest of the boot.
            #
            # PLACED HERE, between the two existing drains, because that is the
            # only slot that is both safe and effective. AFTER the controller
            # teardown: the retire site refuses to free while a transfer thread
            # may still be writing into the span, and the join above is what
            # ends that. BEFORE the second control-queue drain: a
            # current-generation span is QUEUED rather than freed on the spot,
            # and that drain -- already here to "sweep whatever the shutdown
            # itself produced" -- is what turns it into an actual free.
            self._release_retired_prefetch_local()
            self._drain_storage_control_queues_impl(
                n_revoke=None,
                n_backup=None,
                n_release=None,
                extra_release_counts=None,
                log_metrics=False,
            )
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # An admin operation must not kill the server.
            return False, f"Failed to detach HiCache storage backend: {e}"

        self.enable_storage = False
        self.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def storage_capacity_stats(self) -> Optional[dict]:
        """Capacity limits and on-disk usage of the attached backend, if any."""
        if not self.enable_storage or self.cache_controller is None:
            return None
        backend = getattr(self.cache_controller, "storage_backend", None)
        if backend is None or not hasattr(backend, "capacity_stats"):
            return None
        try:
            return backend.capacity_stats()
        except Exception:
            logger.exception("Failed to read HiCache storage capacity stats.")
            return None

    def resize_storage_backend(
        self,
        max_size_bytes: Optional[int] = None,
        min_free_bytes: Optional[int] = None,
    ) -> tuple[bool, str, Optional[dict]]:
        """Re-cap the attached storage backend without detaching it.

        Runtime attach/detach are refused on this cache (see above) because
        they would have to build or tear down the hybrid controller's storage
        threads and per-component host-pool registrations while the tree is
        live. Resize has neither problem: it only moves the backend evictor's
        own byte counters and unlinks files, under the evictor's lock, so it is
        supported here exactly as it is on HiRadixCache. Semantics -- grow is
        immediate, shrink evicts LRU inline -- are documented on
        ``LRUFileEvictor.set_limits``.
        """
        if not self.enable_storage or self.cache_controller is None:
            return False, "HiCache storage backend is not enabled.", None

        backend = getattr(self.cache_controller, "storage_backend", None)
        if backend is None:
            return False, "No HiCache storage backend is attached.", None

        if not hasattr(backend, "resize"):
            return (
                False,
                f"Storage backend {type(backend).__name__} does not support resize.",
                None,
            )

        try:
            stats = backend.resize(
                max_size_bytes=max_size_bytes, min_free_bytes=min_free_bytes
            )
        except Exception as e:
            logger.exception("Failed to resize HiCache storage backend.")
            return False, f"Failed to resize HiCache storage backend: {e}", None

        if stats is None:
            return (
                False,
                f"Storage backend {type(backend).__name__} has no resizable "
                f"capacity accounting.",
                None,
            )
        return True, "Resized HiCache storage backend successfully.", stats

    def clear_storage_backend(self) -> bool:
        try:
            ok = self.cache_controller.clear_storage_backend()
        except Exception as e:
            logger.error("Failed to clear hierarchical cache storage backend: %s", e)
            return False
        if ok:
            logger.info("Hierarchical cache storage backend cleared successfully!")
        return ok

    # ---- HiCache: Async Event Management ----

    #: Contribution of a rank with an EMPTY transfer queue to the cross-rank
    #: MIN: "I impose no constraint", not "drain nothing".
    _NO_ACK_CONSTRAINT = 1 << 30

    def _count_ready_acks(self, ack_queue, label: str) -> int:
        """How many leading entries of `ack_queue` this rank may drain now.

        RANK-LOCAL SINCE #737. This used to MIN-reduce a ready count across the
        group, and that reduction was the deadlock: it sat inside
        `check_hicache_events` -> `_get_new_batch_prefill_raw`, i.e. inside the
        PER-MICROBATCH path of a pipeline, and a pipeline keeps its stages at
        DIFFERENT offsets by construction. Measured 2026-08-17: PP0/PP1 inside
        this drain while PP2 was blocked in `_pp_recv_proxy_tensors` waiting for
        data PP1 would only send after leaving the collective PP1 could not
        leave without PP2. A circular wait between adjacent stages -- the #633
        shape, one level up, with a collective in place of a handler.

        WHAT THE OLD REDUCTION PROVIDED, and where each part went:

        * OP-SEQUENCE UNIFORMITY ("every rank must enter, or NCCL desyncs").
          That concerned the TP all_reduce, which `_all_reduce` ran only on
          `pp_rank == 0`. With no reduction there is no op to keep in sequence.
        * THE EMPTY-QUEUE SENTINEL, which existed because a MIN conflates
          "nothing to drain" with "not finished yet" -- one idle rank froze the
          drain on ALL ranks and `protected` ratcheted until the state pool died
          (#581). Rank-local counting removes that failure mode by construction:
          an idle rank cannot freeze a peer that no longer waits on it.
        * THE THROTTLE ("ranks never run further apart than the slowest live
          transfer"). This was PACING, not correctness, and it is the one thing
          genuinely lost. It is deliberately NOT replaced by a bound here; see
          the drain-depth line below for why, and the filed pacing task.

        WHAT NOW OWNS CORRECTNESS -- do not resurrect the collective for it.
        The cross-rank hazard was publishing a shared content key whose bytes
        are incomplete on some stage. That is bounded per PAGE by #706's
        completeness marker (`canonical_kv_page.py`, `PageCompleteness`):
        production is layer-sharded while storage is token-sharded, so a page's
        slots arrive from several PP stages and `is_complete()` gates use.
        Ranks arbitrarily far apart in ack processing therefore yield an
        INCOMPLETE page, which reads as a MISS -- never as wrong bytes. Ack
        lockstep was never what made this safe.

        Everything this function now touches is this rank's own: `ack_queue`,
        and downstream `_finish_write_through_ack` popping this rank's
        `ongoing_write_through` and `dec_lock_ref`-ing this rank's own node.
        """
        ready = 0
        for _, finish_event, _ack_list in ack_queue:
            if not finish_event.query():
                break
            ready += 1

        # #737 OBSERVABILITY, deliberately not a bound. The throttle that used
        # to fall out of the MIN is gone (see the docstring), and its loss shows
        # up as a FAST rank running ahead in host-tier pressure. No backpressure
        # limit is shipped here: a bound chosen without an operating point is
        # the #505c class of shipped-number-without-evidence. This line is what
        # makes the first real specimen attributable when it appears.
        if ready and self._drain_depth_every:
            now = time.perf_counter()
            if now - getattr(self, "_drain_depth_at", 0.0) >= self._drain_depth_every:
                self._drain_depth_at = now
                logger.info(
                    "#737 drain depth [%s]: pp_rank=%s ready=%d pending=%d "
                    "(rank-local; no group agreement is taken here)",
                    label,
                    self.pp_rank,
                    ready,
                    len(ack_queue) - ready,
                )
        return ready

    def writing_check(self, write_back: bool = False) -> None:
        """Poll write-through completions."""
        cc = self.cache_controller
        if cc is None:
            return

        if write_back:
            # Blocking: wait for all pending write-backs
            while self.ongoing_write_through:
                for _, finish_event, ack_list in cc.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        if ack_id in self.ongoing_write_through:
                            self._finish_write_through_ack(ack_id)
                cc.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        finish_count = self._count_ready_acks(cc.ack_write_queue, "writing_check")

        # Process completed acks
        while finish_count > 0:
            _, finish_event, ack_list = cc.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                self._finish_write_through_ack(ack_id)
            finish_count -= 1

    def loading_check(self) -> None:
        """Poll load-back completions."""
        cc = self.cache_controller
        if cc is None:
            return
        # Same rank-local-queue rule as the write side: a rank with nothing
        # loading must not hold the others' load-back pins (which also make
        # mamba checkpoints unevictable).
        finish_count = self._count_ready_acks(cc.ack_load_queue, "loading_check")

        while finish_count > 0:
            _, finish_event, ack_list = cc.ack_load_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                node, lock_params, host_lock_params = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(node, lock_params)
                self.dec_host_lock_ref(node, host_lock_params)
            finish_count -= 1

    # ---- HiCache: Scheduler Entry Points ----

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> tuple[torch.Tensor, UnifiedTreeNode]:
        """Prepare KV cache loading from host to device.
        Returns (device_indices, last_node) tuple."""
        best_match_node = params.best_match_node
        mem_quota = params.mem_quota
        req = params.req
        assert req is not None
        last_best_match_device_node = req.last_node

        def _collect_new_prefix_indices() -> torch.Tensor:
            prefix_chunks: list[torch.Tensor] = []
            node = best_match_node
            while node is not last_best_match_device_node:
                value = node.component_data[BASE_COMPONENT_TYPE].value
                assert value is not None
                prefix_chunks.append(value)
                node = node.parent
            if not prefix_chunks:
                return self._empty_match_result.device_indices
            prefix_chunks.reverse()
            return torch.cat(prefix_chunks)

        if (
            best_match_node.evicted
            or params.host_hit_length > 0
            or (
                req is not None
                and (req.swa_host_hit_length > 0 or req.mamba_host_hit_length > 0)
            )
        ):
            if self.load_back(best_match_node, mem_quota, req=req):
                new_indices = _collect_new_prefix_indices()
                if new_indices.numel() == 0:
                    return (
                        self._empty_match_result.device_indices,
                        last_best_match_device_node,
                    )

                logger.debug(
                    "init_load_back success: loaded %d tokens for node %d",
                    len(new_indices),
                    best_match_node.id,
                )
                return new_indices, best_match_node

        return (
            self._empty_match_result.device_indices,
            last_best_match_device_node,
        )

    # ---- #581 mamba pin trace (SGLANG_MAMBA_PIN_TRACE, default off) -------

    def _init_pin_trace(self) -> None:
        """Arm the per-tick mamba pin trace from the environment.

        `SGLANG_MAMBA_PIN_TRACE=N` emits one line per rank every N scheduler
        ticks; 0 (default) leaves every traced path unentered.
        """
        self._pin_trace_every = int(envs.SGLANG_MAMBA_PIN_TRACE.get())
        # #737: seconds between drain-depth lines, 0 = off. Rate-limited rather
        # than per-drain because the point is to catch a SUSTAINED divergence in
        # host-tier pressure between ranks, not to narrate healthy rounds.
        self._drain_depth_every = 30.0
        self._drain_depth_at = 0.0
        self._pin_trace_ticks = 0
        self._pin_trace_site = "?"
        self._pin_trace_ops: Counter = Counter()

    def _pin_trace_begin(self, op: str) -> None:
        """Attribute the lock call about to run to its immediate caller.

        Resolved ONCE per inc/dec_lock_ref rather than once per component, so
        the components can tag their own mamba accounting with the same site.
        """
        self._pin_trace_site = sys._getframe(2).f_code.co_name
        self._pin_trace_ops[(op, self._pin_trace_site)] += 1

    def record_pin_trace_mamba(self, op: str, host: bool = False) -> None:
        """Called by the MAMBA component when a mamba ref actually moves."""
        tag = f"{op}_mamba_host" if host else f"{op}_mamba"
        self._pin_trace_ops[(tag, self._pin_trace_site)] += 1

    def _mamba_pins_in(self, registry) -> int:
        """Registry entries whose stored lock actually took a MAMBA ref.

        An entry's `lock_params` carries the skip set from the paired acquire,
        so an entry that skipped MAMBA (tombstone at acquire time) holds no
        mamba pin and must not be counted as one.
        """
        count = 0
        for entry in registry.values():
            params = entry.lock_params
            if params is None:
                continue
            if entry.node.id in params.skip_lock_node_ids.get(ComponentType.MAMBA, ()):
                continue
            count += 1
        return count

    def _mamba_pins_held(self) -> int:
        """Mamba state slots currently pinned by in-flight write-throughs."""
        return self._mamba_pins_in(self.ongoing_write_through)

    @property
    def _mamba_pin_budget(self) -> int:
        """How many state slots retention may hold pinned at once.

        The pool above the hard floor -- see
        :func:`mamba_pool_floor.mamba_retention_pin_budget` for why that is
        the right number and what happens without it (#581).
        """
        if self._mamba_pin_budget_cached is None:
            from sglang.srt.mem_cache.mamba_pool_floor import (
                mamba_retention_pin_budget,
            )
            from sglang.srt.runtime_context import get_server_args

            server_args = get_server_args()
            mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
            if mamba_pool is None:
                # No mamba pool on this tree: nothing to protect, and the
                # guard must never refuse a backup it cannot be about. -1
                # reads as "unbounded" at the single comparison below.
                self._mamba_pin_budget_cached = -1
            else:
                self._mamba_pin_budget_cached = mamba_retention_pin_budget(
                    server_args,
                    server_args.max_running_requests or 1,
                    mamba_pool.size,
                )
        return self._mamba_pin_budget_cached

    def _mamba_write_through_pin_admissible(
        self, node: UnifiedTreeNode, write_back: bool = False
    ) -> bool:
        """May this backup take a write-through pin on a mamba checkpoint?

        Three ways to be admissible without consulting the budget at all, and
        each is a case where the backup costs no state slot:

        * ``write_back=True`` -- the demotion path never calls
          ``inc_lock_ref`` (see ``write_backup``), so it pins nothing;
        * the node carries no mamba value -- ``acquire_component_lock`` takes
          a MAMBA ref only on a node that has one, so a KV-only node's pin
          costs no state slot and refusing it would lose a host copy for
          nothing;
        * there is no mamba pool on this tree.

        Otherwise the pin is charged against the budget. Refusing here costs
        at most a host-tier miss: the state stays cached on the device and
        stays EVICTABLE, so the pool can always serve the running set. Not
        refusing costs the scheduler.
        """
        if write_back:
            return True
        budget = self._mamba_pin_budget
        if budget < 0:
            return True
        if ComponentType.MAMBA not in self.tree_components:
            return True
        if node is self.root_node:
            return True
        if len(node.component_data) <= int(ComponentType.MAMBA):
            return True
        if node.component_data[ComponentType.MAMBA].value is None:
            return True
        return self._mamba_pins_held() < budget

    def _note_mamba_pin_skipped(self) -> None:
        """Count and (rate-limited) announce a backup declined by the budget.

        A persistent count is a real signal, not noise: it means the
        write-through ack drain is not keeping up with insert pressure, so
        the host tier is being fed slower than the tree is producing
        checkpoints. The pool is safe either way -- that is the point of the
        budget -- but the host-tier hit rate is paying for it.
        """
        self._mamba_pin_skipped += 1
        if self._mamba_pin_skipped <= 3 or self._mamba_pin_skipped % 1000 == 0:
            mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
            logger.warning(
                "mamba write-through pin budget reached (%d in flight, "
                "budget=%d, pool=%s): skipping the host backup of this "
                "checkpoint to keep its state slot evictable. occurrence=%d",
                self._mamba_pins_held(),
                self._mamba_pin_budget,
                "?" if mamba_pool is None else mamba_pool.size,
                self._mamba_pin_skipped,
            )

    def _chunk_anchor_publish_enabled(self) -> bool:
        """May a chunked-prefill node reach the host tier? (#1028)

        Structural, not a flag: the deviation from upstream's chunked skip is
        only meaningful where there is a per-chunk recurrent anchor to publish
        AND a storage tier to publish it to. Without either, this returns
        False and `_inc_hit_count` behaves byte-identically to upstream.
        """
        # DELIBERATELY NOT CACHED. `self.enable_storage` is False at
        # `__init__` time and only becomes True inside `init_hicache`
        # (:813 vs :846). A value memoised by an early caller would pin this
        # to False for the process and leave a WIRED-BUT-INERT write path --
        # the #742/#745 failure class this exact area has produced before
        # (prior art: "silent-inert write path on the composite"), and the
        # most expensive kind of bug here because it looks like a clean boot
        # that simply does not help. Three attribute reads against a host
        # backup is not a cost worth that risk.
        return bool(
            self.cache_controller is not None
            and getattr(self, "enable_storage", False)
            and ComponentType.MAMBA in self.tree_components
        )

    def _note_chunk_publish(self) -> None:
        """Count and (rate-limited) announce a chunk publish. (#1028)

        THIS COUNTER IS THE RANK-UNANIMITY PROOF, and that is why it prints
        `n` rather than only a total. The publish decision is taken at the
        scheduler's chunk boundary (`scheduler.py`'s
        `maybe_cache_unfinished_req(..., chunked=True)`), which every rank
        runs for the same request at the same split -- so it is unanimous BY
        CONSTRUCTION, not by agreement, and this path holds no reduce (the
        one at `check_prefetch_progress` is TP-scoped and this boot runs
        tp_size=1, pp_size=3, so it is structurally skipped).

        `raenge-nie-uneins` is therefore satisfied by construction and this
        line is how the claim gets CHECKED instead of asserted: identical `n`
        at identical timestamps across PP0/PP1/PP2 is the same evidence shape
        that `#969H BACKUP` gave for the 11 finish-time backups of boot
        `boot_855_1028fence` (n=1..11 on all three ranks, same seconds).
        Divergent `n` between ranks means a rank-local input entered this
        path and is a STOP-class finding, not a tuning observation.

        The one rank-local input reachable from here is the #581/#773
        write-through pin budget (`_mamba_write_through_pin_admissible`,
        rank-local by its own comment at the `write_backup` head). It fired
        ZERO times in that boot (trap-safe count of "write-through pin budget"
        = bare 0, genuine 0, whole file) because 11 backups never approached
        it. Publishing per chunk raises pin pressure, so it becomes REACHABLE
        here for the first time -- `_note_mamba_pin_skipped` is the paired
        instrument, and a nonzero count there next to a divergent `n` here is
        the divergence, named in advance.
        """
        # PER-RANK, NOT PER-INSTANCE. The first version of this counter was
        # `self._chunk_publish_n`, and that made it useless for the one job it
        # was built for: a rank runs SEVERAL UnifiedRadixCache instances (the
        # PP stack, the TP stack, the flip stacks), so the counter reset per
        # instance and the log filled with `n=1` lines. Boot 1 printed 4/4/4
        # across the ranks and that LOOKED like the unanimity proof; it was a
        # line-count coincidence, and boot 2 printed 7/6/6 from the same
        # healthy state. A class attribute is per PROCESS, and one process is
        # one rank, which is the population the `raenge-nie-uneins` check
        # actually asks about. Same shape as the `#969H` probe, which is why
        # that one could carry its claim and this one could not.
        n = getattr(type(self), "_chunk_publish_n_rank", 0) + 1
        type(self)._chunk_publish_n_rank = n
        self._chunk_publish_n += 1
        if n <= 40 or n % 256 == 0:
            logger.warning(
                "#1028P CHUNK-PUBLISH n=%d: publishing a chunked-prefill node "
                "to the host tier (upstream skips this; see _inc_hit_count). "
                "pin_skipped=%d",
                n,
                self._mamba_pin_skipped,
            )

    def _emit_pin_trace(self) -> None:
        """One line per rank per N ticks; counters are since the previous line."""
        self._pin_trace_ticks += 1
        if self._pin_trace_ticks % self._pin_trace_every:
            return

        ops = " ".join(
            f"{op}@{site}={count}"
            for (op, site), count in sorted(self._pin_trace_ops.items())
        )
        controller = self.cache_controller
        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        logger.info(
            "MAMBA-PIN-TRACE impl=unified tick=%d ack_write=%s ack_load=%s "
            "wt_mamba_pins=%d lb_mamba_pins=%d ongoing_wt=%d ongoing_lb=%d "
            "ongoing_backup=%d protected=%d evictable=%d mamba_avail=%s ops[%s]",
            self._pin_trace_ticks,
            "?" if controller is None else len(controller.ack_write_queue),
            "?" if controller is None else len(controller.ack_load_queue),
            self._mamba_pins_in(self.ongoing_write_through),
            self._mamba_pins_in(self.ongoing_load_back),
            len(self.ongoing_write_through),
            len(self.ongoing_load_back),
            len(self.ongoing_backup),
            self.component_protected_size_.get(ComponentType.MAMBA, 0),
            self.component_evictable_size_.get(ComponentType.MAMBA, 0),
            "?" if mamba_allocator is None else mamba_allocator.available_size(),
            ops,
        )
        self._pin_trace_ops.clear()

    def check_hicache_events(self) -> None:
        """Called per scheduler step to poll async HiCache events."""
        # #1028 ROUND CENSUS -- the instrument that decides whether a per-ROUND
        # carrier is viable at all, taken BEFORE any such carrier is built.
        #
        # §AH established that `_pp_sync` cannot carry the prefetch agreement
        # because it is a rendezvous per OPERATION and entry is not rank-
        # uniform. The proposed replacement is a rendezvous per ROUND here,
        # on the premise that every rank reaches this point exactly once per
        # scheduler step. THAT PREMISE IS UNTESTED, and the operator's own
        # reading of the last boot is evidence against it: PP0 ran a fifth
        # prefill batch its peers never ran, so the ranks did not execute the
        # same amount of work in the same window.
        #
        # If the round counts diverge across ranks, a per-round ring send
        # inherits the identical unmatched-entry wedge one level up, and the
        # carrier has to be the existing per-pass proxy message instead. This
        # counter answers that with data. Purely local, no collective; the
        # comparison is made across the three ranks' logs afterwards.
        try:
            self._1028_round = int(getattr(self, "_1028_round", 0)) + 1
            _every = 25
            if self._1028_round % _every == 0:
                logger.info(
                    "#1028 HICACHE-ROUND n=%d pp_rank=%s pp_size=%s "
                    "attn_reduce_world=%d ongoing_prefetch=%d",
                    self._1028_round,
                    getattr(self, "pp_rank", -1),
                    getattr(self, "pp_size", -1),
                    self._attn_reduce_world(),
                    len(getattr(self, "ongoing_prefetch", {}) or {}),
                )
        except Exception:  # noqa: BLE001 - a probe may never break the round
            pass
        # Reap the previous round's PP-sync sends before issuing new ones.
        self._drain_async_work()
        self.writing_check()
        self.loading_check()
        if self._pin_trace_every:
            self._emit_pin_trace()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def flush_write_through_acks(self) -> None:
        """Flush pending write-through acknowledgements."""
        self.writing_check()

    def ready_to_load_host_cache(self) -> int:
        """Notify the cache controller to start the KV cache loading."""
        if self.cache_controller is not None:
            return self.cache_controller.start_loading()
        return 0

    # ---- Query / Inspection APIs ----
    # These APIs exist for compatibility with other RadixTree implementations.
    # TODO: simplify and consolidate in a future refactor.

    @property
    def sliding_window_size(self):
        swa = self.components.get(ComponentType.SWA)
        return swa.sliding_window_size if swa else None

    def swa_reprefill_tail_tokens(self) -> int:
        """
        Only unified_kv + HiCache needs this: SWA lives in a per-request ring
        (state_slot/pos), not content-stable and never offloaded to host, so a
        reused prefix's trailing sliding window would read another request's
        stale ring slots. Re-prefilling that window rewrites this request's ring
        (what plain radix reuse does via its SWA match gate). 0 for every other
        layout.
        """
        swa = self.components.get(ComponentType.SWA)
        unified_compress_only_hicache = (
            self.cache_controller is not None
            and swa is not None
            and swa._swa_kv_pool_host is None
        )
        return swa.sliding_window_size if unified_compress_only_hicache else 0

    def supports_swa(self) -> bool:
        return ComponentType.SWA in self.components

    def supports_mamba(self) -> bool:
        return ComponentType.MAMBA in self.components

    # ---- Streaming session API (delegates to composed StreamingSession) ----

    def supports_streaming_session(self) -> bool:
        return True

    def release_session(self, session_id: str) -> None:
        self.session.release_session(session_id)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_tokens(active_pool_idxs)

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_full_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_swa_tokens(active_pool_idxs)

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_req_count(active_pool_idxs)

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_mamba_slots(active_pool_idxs)

    def evictable_size(self) -> int:
        return self.component_evictable_size_.get(BASE_COMPONENT_TYPE, 0)

    def protected_size(self) -> int:
        return self.component_protected_size_.get(BASE_COMPONENT_TYPE, 0)

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.SWA, 0)

    def mamba_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.MAMBA, 0)

    def swa_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.SWA, 0)

    def mamba_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.MAMBA, 0)

    def total_size(self):
        total_size = 0
        total_aux_size = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            if full_value is not None:
                total_size += len(full_value)
            for ct in self.tree_components:
                if ct == BASE_COMPONENT_TYPE:
                    continue
                value = node.component_data[ct].value
                if value is not None:
                    total_aux_size += len(value)
            for child in node.children.values():
                stack.append(child)
        return total_size, total_aux_size

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs(node: UnifiedTreeNode):
            for child in node.children.values():
                v = child.component_data[BASE_COMPONENT_TYPE].value
                if v is not None:
                    values.append(v)
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def _all_component_values_flatten(
        self, component_type: ComponentType
    ) -> torch.Tensor:
        if component_type not in self.components:
            return torch.tensor([], dtype=torch.int64, device=self.device)

        values = []

        def _dfs(node: UnifiedTreeNode):
            value = node.component_data[component_type].value
            if value is not None:
                values.append(value)
            for child in node.children.values():
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentType.MAMBA)

    def available_and_evictable_str(self) -> str:
        if self.supports_swa():
            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        else:
            full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable = self.component_evictable_size_[BASE_COMPONENT_TYPE]
        lines = [
            f"Available full tokens: {full_available_size + full_evictable} "
            f"(full_available_size={full_available_size} + full_evictable_size_={full_evictable})"
        ]
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            if ct.is_swa:
                available_size = self.token_to_kv_pool_allocator.swa_available_size()
            elif ct.is_mamba:
                available_size = self.req_to_token_pool.mamba_allocator.available_size()
            else:
                continue

            lines.append(
                f"Available {ct}: {available_size + self.component_evictable_size_[ct]} "
                f"(available_size={available_size} + component_evictable_size_={self.component_evictable_size_[ct]})"
            )
        return "\n".join(lines) + "\n"

    def leak_census_str(self) -> str:
        """What the TREE holds, recomputed from the nodes, for a pool leak.

        The pool ledger only ever says how big the ``evictable`` TERM is. When
        rows go missing that leaves two indistinguishable worlds, and they have
        opposite fixes:

          (a) the tree HOLDS the rows and the term does not follow -- a
              bookkeeping defect at the handover point;
          (b) the tree holds NOTHING and the rows were orphaned outside it --
              a missing free / a lost owner.

        ``sanity_check`` PART 4 already recomputes the per-component sizes from
        a full node walk, but it runs after the pool check, which raises first,
        so its answer never reached a log. This is the same recomputation,
        callable at the moment the leak is found and cheap enough to run once
        on a fatal path: node count, the recomputed evictable/protected sums
        per component against the tracked terms, and the first few nodes with
        their row counts, so "steht der Knoten im Baum?" is answered by the
        error itself rather than by the next boot.
        """
        try:
            all_nodes = self._collect_all_nodes()
        except Exception as exc:  # noqa: BLE001 - diagnostics must not mask the leak
            return f"tree census unavailable: {exc!r}"

        parts = [f"nodes={len(all_nodes)}"]
        for ct in self.tree_components:
            evictable = 0
            protected = 0
            for n in all_nodes:
                if n is self.root_node:
                    continue
                cd = n.component_data[ct]
                if cd.value is not None:
                    if cd.lock_ref > 0:
                        protected += len(cd.value)
                    else:
                        evictable += len(cd.value)
            parts.append(
                f"{ct.name}: tracked_evictable={self.component_evictable_size_[ct]} "
                f"recomputed_evictable={evictable} "
                f"tracked_protected={self.component_protected_size_[ct]} "
                f"recomputed_protected={protected}"
            )
        held = []
        for n in all_nodes:
            if n is self.root_node:
                continue
            cd = n.component_data[BASE_COMPONENT_TYPE]
            held.append(
                f"id={n.id} keylen={len(n.key) if n.key is not None else '-'} "
                f"full_dev={0 if cd.value is None else len(cd.value)} "
                f"full_host={0 if cd.host_value is None else len(cd.host_value)} "
                f"lock={cd.lock_ref}"
            )
            if len(held) >= 8:
                break
        parts.append("held=[" + "; ".join(held) + "]")
        return "TREE CENSUS " + " | ".join(parts)

    def reclaim_rows_for_drop(self) -> dict:
        """Return EVERY row the tree still holds -- locked ones included (#1050).

        THE CONTRACT `drop_prefix_tree_returning_rows` COULD NOT KEEP. That
        function promises "empty the prefix tree AND return its rows" and pays
        the allocator back through `evict`, whose leaf walk REFUSES a locked
        node. `_reset_full` then installs a fresh root and zeroes
        `component_protected_size_` without freeing one device row, so every
        row that was locked at the drop belongs to nobody from that moment on.
        #938 has measured exactly that since 2026-08-27 and deliberately did
        not free, on a premise this function is allowed to act on only because
        the premise is now CHECKED rather than assumed.

        THE PREMISE, AND WHY IT IS CHECKED AND NOT TRUSTED. #938's comment says
        a node is still locked here "mainly because a write-through is IN
        FLIGHT against it -- a live reader that is copying those exact device
        rows to the host", and freeing under a live reader is a use-after-free
        in the #913 IMA family, strictly worse than the leak. That reasoning is
        correct and is why the release is gated, not removed: this function
        frees ONLY when `ongoing_write_through` is empty, i.e. when the seam's
        own post-retract writeback fence has already reported every copy
        settled. Then no reader exists that could be freed underneath, and a
        lock that survives a fence with nothing outstanding is not a live
        claim -- it is a claim whose holder is about to cease to exist with
        the tree.

        MEASURED, and it is why this is not speculative (boot_855_1049n9,
        2026-08-31, all 13 drops, PP0): every `#792 post-retract writeback
        fence` line reports `outstanding=0`, and two of those same drops
        orphaned 5834 and 4618 rows. The in-flight explanation did not hold on
        a single drop of that boot. The gate keeps the other case correct
        anyway -- if a copy ever IS outstanding here, this refuses and says so
        with a number, which is the state #938 already handles.

        NEVER FREES BLIND. A row that is already on the free list must not be
        freed twice: that is silent corruption, the one outcome worse than the
        leak. The candidate set is therefore differenced against the
        allocator's OWN enumerated free set (`read_free_rows`, the same
        authority the idle invariant and the #822 census read, a UNION over
        free+release rather than a double-counting sum). If that reading is not
        enumerable, this refuses entirely -- a watermark allocator can say HOW
        MANY rows are free and genuinely cannot say WHICH, and freeing against
        a count would be inventing membership.

        THE FREE IS ADDRESSED PER ACCESS, never through a captured binding:
        `FullComponent._free_full` resolves `token_to_kv_pool_allocator` at
        call time for the #941 reason (a bound method captured at construction
        carries its instance, and the seam rebinds the pool underneath it --
        the rows then land on the other phase's free list as duplicates, which
        is the same symptom with a different root, documented in that method).

        Returns a dict, ALWAYS -- the caller logs it unconditionally including
        the all-zero reading, because a negative reading is what makes the
        comparison decisive rather than suggestive.
        """
        out = {
            "reclaimed": False,
            "reason": "",
            "full_rows": 0,
            "mamba_slots": 0,
            "full_held": 0,
            "mamba_held": 0,
            "already_free": 0,
        }
        try:
            outstanding = len(self.ongoing_write_through)
        except Exception:  # noqa: BLE001 - a reclaim may never break a seam
            out["reason"] = "ongoing_write_through unreadable"
            return out
        if outstanding:
            # THE DEFERRAL IS INSTRUMENTED, NOT SILENT. Size and reason are
            # printed by the caller on every drop, so a reclaim that never
            # runs cannot become the same ratchet one level up.
            out["reason"] = f"write-through still outstanding ({outstanding})"
            return out

        try:
            nodes = [n for n in self._collect_all_nodes() if n is not self.root_node]
        except Exception:  # noqa: BLE001
            out["reason"] = "tree walk failed"
            return out

        # `component_data` is a LIST indexed by the ComponentType int enum, not
        # a dict -- a `.get` here would raise on every drop. Membership is
        # asked of `tree_components`, which is fixed at construction.
        has_mamba = ComponentType.MAMBA in self.tree_components
        full_vals = []
        mamba_vals = []
        for node in nodes:
            val = getattr(node.component_data[BASE_COMPONENT_TYPE], "value", None)
            if val is not None and len(val) > 0:
                full_vals.append(val)
            if has_mamba:
                mval = getattr(node.component_data[ComponentType.MAMBA], "value", None)
                if mval is not None and len(mval) > 0:
                    mamba_vals.append(mval)

        out["full_held"] = int(sum(len(v) for v in full_vals))
        out["mamba_held"] = int(sum(len(v) for v in mamba_vals))
        if not full_vals and not mamba_vals:
            out["reclaimed"] = True
            out["reason"] = "tree held nothing"
            return out

        full_comp = self.components.get(BASE_COMPONENT_TYPE)
        if full_vals and full_comp is not None:
            try:
                from sglang.srt.mem_cache.kv_row_ownership import read_free_rows

                reading = read_free_rows(self.token_to_kv_pool_allocator)
                if not reading.is_enumerable:
                    out["reason"] = (
                        "allocator free set is not enumerable; refusing to free "
                        "against a count"
                    )
                    return out
                already = reading.rows
                keep = []
                dup = 0
                for v in full_vals:
                    sel = [int(x) for x in v.tolist()] if hasattr(v, "tolist") else list(v)
                    fresh = [x for x in sel if x not in already]
                    dup += len(sel) - len(fresh)
                    if fresh:
                        keep.append(torch.tensor(fresh, dtype=torch.int64))
                out["already_free"] = int(dup)
                if keep:
                    merged = torch.cat(keep)
                    full_comp._free_full(merged)
                    out["full_rows"] = int(merged.numel())
            except Exception as exc:  # noqa: BLE001 - never abort a flip
                out["reason"] = f"full reclaim failed: {type(exc).__name__}"
                return out

        mamba_comp = self.components.get(ComponentType.MAMBA)
        if mamba_vals and mamba_comp is not None:
            for mval in mamba_vals:
                try:
                    mamba_comp._free_mamba_value(mval)
                    out["mamba_slots"] += int(len(mval))
                except Exception:  # noqa: BLE001 - one slot may not break a flip
                    continue

        out["reclaimed"] = True
        return out

    def _collect_all_nodes(self) -> list[UnifiedTreeNode]:
        nodes = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        return nodes

    def sanity_check(self):
        """Verify tree invariants.

        TODO(hzh): This method has relatively high latency; simplify the
        check logic once the tree implementation stabilizes.
        """
        # Skip when streaming sessions hold tree locks: the check asserts
        # all nodes are unlocked during idle, which streaming sessions break
        # by design (they hold a first-turn lock across turns).
        if self.session.any_holding_kv():
            return

        write_back = (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        )

        errors: list[str] = []
        E = errors.append
        all_nodes = self._collect_all_nodes()
        all_node_set = set(all_nodes)
        FCT = BASE_COMPONENT_TYPE

        # ── PART 1: Tree Structure ──
        # Root state
        if self.root_node.component_data[FCT].value is None:
            E("[Root] root missing Full device value")
        if self.root_node.component_data[FCT].lock_ref <= 0:
            E(
                f"[Root] root Full lock_ref={self.root_node.component_data[FCT].lock_ref}"
            )
        if self.root_node.parent is not None:
            E("[Root] root has a parent pointer")
        # Parent ↔ child bidirectional consistency
        for node in all_nodes:
            for child in node.children.values():
                if child.parent is not node:
                    pid = child.parent.id if child.parent else None
                    E(f"[Tree] child {child.id} parent={pid}, expected {node.id}")
                if child.key is None:
                    E(f"[Tree] node {child.id} has no key")

        # ── PART 2: Per-node state machine and leaf qualification ──
        expected_dev_leaves: set[UnifiedTreeNode] = set()
        expected_hst_leaves: set[UnifiedTreeNode] = set()

        for node in all_nodes:
            if node is self.root_node:
                continue
            nid = node.id
            full_dev = node.component_data[FCT].value is not None
            full_hst = node.component_data[FCT].host_value is not None

            # Full is the tree backbone, so aux data requires Full data.
            for ct in self.tree_components:
                if ct == FCT:
                    continue
                cd = node.component_data[ct]
                if cd.value is not None and not full_dev:
                    E(f"node {nid} {ct} device present but Full.value=None")
                if cd.host_value is not None and not full_hst:
                    E(f"node {nid} {ct} host present but Full.host_value=None")

            # Every node must keep Full data on at least one layer.
            if not full_dev and not full_hst:
                E(f"node {nid} dead: no Full device and no Full host")

            # Parent prefixes must keep data whenever the child does.
            if node.parent is not None and node.parent is not self.root_node:
                p_dev = node.parent.component_data[FCT].value is not None
                p_hst = node.parent.component_data[FCT].host_value is not None
                if full_dev and not p_dev:
                    E(f"node {nid} device present but parent {node.parent.id} evicted")
                if full_hst and not p_hst and not write_back:
                    E(f"node {nid} backed up but parent {node.parent.id} not backed up")

            # Lock hierarchy and counters must stay sane.
            fl = node.component_data[FCT].lock_ref
            for ct in self.tree_components:
                cd = node.component_data[ct]
                if cd.lock_ref < 0:
                    E(f"node {nid} {ct} lock_ref={cd.lock_ref}")
                if cd.host_lock_ref < 0:
                    E(f"node {nid} {ct} host_lock_ref={cd.host_lock_ref}")
                if ct != FCT and fl < cd.lock_ref:
                    E(f"node {nid} full_lock={fl} < {ct}_lock={cd.lock_ref}")
                if cd.value is None and cd.lock_ref > 0:
                    E(f"node {nid} {ct} evicted but lock_ref={cd.lock_ref}")

            # Collect expected leaf qualification (single pass)
            if self._is_device_leaf(node):
                expected_dev_leaves.add(node)
            if self._is_host_leaf(node):
                expected_hst_leaves.add(node)

        # ── PART 3: Tracking structures ──

        # Device leaf set must match the expected leaves.
        if self.evictable_device_leaves != expected_dev_leaves:
            extra = self.evictable_device_leaves - expected_dev_leaves
            missing = expected_dev_leaves - self.evictable_device_leaves
            if extra:
                E(f"D-leaf extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"D-leaf missing: {[n.id for n in list(missing)[:5]]}")

        # Host leaf set must match the expected leaves.
        if self.evictable_host_leaves != expected_hst_leaves:
            extra = self.evictable_host_leaves - expected_hst_leaves
            missing = expected_hst_leaves - self.evictable_host_leaves
            if extra:
                E(f"H-leaf extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"H-leaf missing: {[n.id for n in list(missing)[:5]]}")

        # D-leaf ∩ H-leaf = ∅
        overlap = self.evictable_device_leaves & self.evictable_host_leaves
        if overlap:
            E(
                f"[Leaf] {len(overlap)} in both sets: {[n.id for n in list(overlap)[:5]]}"
            )

        # Stale nodes: leaf sets must only contain tree-reachable nodes
        stale = self.evictable_device_leaves - all_node_set
        if stale:
            E(
                f"{len(stale)} stale nodes in device_leaves: {[n.id for n in list(stale)[:5]]}"
            )
        stale = self.evictable_host_leaves - all_node_set
        if stale:
            E(
                f"{len(stale)} stale nodes in host_leaves: {[n.id for n in list(stale)[:5]]}"
            )

        # Per-component LRU tracking
        for ct in self.tree_components:
            lru = self.lru_lists[ct]
            if ct == FCT:
                # Full uses leaf sets, not LRU
                if len(lru.cache) > 0:
                    E(f"Full device LRU not empty: {len(lru.cache)}")
                if len(self.host_lru_lists[ct].cache) > 0:
                    E(f"Full host LRU not empty: {len(self.host_lru_lists[ct].cache)}")
            else:
                # Aux device values must match the device LRU.
                tree_ids = {
                    n.id
                    for n in all_nodes
                    if n is not self.root_node
                    and n.component_data[ct].value is not None
                }
                lru_ids = set(lru.cache.keys())
                if tree_ids != lru_ids:
                    E(
                        f"{ct} device LRU: "
                        f"+tree={tree_ids - lru_ids}, +lru={lru_ids - tree_ids}"
                    )
                # Aux host-only states must match the host LRU.
                host_lru = self.host_lru_lists[ct]
                s3_ids = {
                    n.id
                    for n in all_nodes
                    if n is not self.root_node
                    and n.component_data[ct].value is None
                    and n.component_data[ct].host_value is not None
                }
                host_lru_ids = set(host_lru.cache.keys())
                if s3_ids != host_lru_ids:
                    E(
                        f"{ct} host LRU: "
                        f"+S3={s3_ids - host_lru_ids}, +lru={host_lru_ids - s3_ids}"
                    )
                # The same aux node must not appear in both device and host LRU.
                inv5_overlap = lru_ids & host_lru_ids
                if inv5_overlap:
                    E(f"{ct} in both device and host LRU: {inv5_overlap}")
                # Linked-list integrity
                self._check_lru_linked_list(lru, ct, "device", errors)
                self._check_lru_linked_list(host_lru, ct, "host", errors)

        # ── PART 4: Size Accounting ──
        for ct in self.tree_components:
            evictable = 0
            protected = 0
            for n in all_nodes:
                if n is self.root_node:
                    continue
                cd = n.component_data[ct]
                if cd.value is not None:
                    toks = len(cd.value)
                    if cd.lock_ref > 0:
                        protected += toks
                    else:
                        evictable += toks
            if self.component_evictable_size_[ct] != evictable:
                E(
                    f"[Size] {ct} evictable={self.component_evictable_size_[ct]} "
                    f"!= recomputed={evictable}"
                )
            if self.component_protected_size_[ct] != protected:
                E(
                    f"[Size] {ct} protected={self.component_protected_size_[ct]} "
                    f"!= recomputed={protected}"
                )

        # ── PART 5: Ongoing Operations ──
        for nid, (n, _, _) in self.ongoing_write_through.items():
            if n not in all_node_set:
                E(f"[Ongoing] write_through node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] write_through node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )
        for nid, (n, _, _) in self.ongoing_load_back.items():
            if n not in all_node_set:
                E(f"[Ongoing] load_back node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] load_back node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )

        # ── Result ──
        if errors:
            msg = (
                f"Sanity check FAILED ({len(errors)} violations "
                f"across {len(all_nodes)} nodes):\n"
                + "\n".join(f"  {e}" for e in errors)
            )
            logger.error(msg)
            self.pretty_print()
            raise AssertionError(msg)

    def _check_lru_linked_list(
        self,
        lru: UnifiedLRUList,
        ct: ComponentType,
        label: str,
        errors: list[str],
    ) -> None:
        """Walk a LRU doubly-linked list, collect integrity errors."""
        pt = lru._pt  # use LRU's own pointer slot
        visited: set[int] = set()
        x = lru.head.lru_next[pt]
        prev = lru.head
        while x is not None and x != lru.tail:
            if x.lru_prev[pt] != prev:
                errors.append(f"[{label}][{ct}] broken prev at node {x.id}")
            if x.id not in lru.cache:
                errors.append(f"[{label}][{ct}] node {x.id} in list not cache")
            if x.id in visited:
                errors.append(f"[{label}][{ct}] cycle at node {x.id}")
                break
            visited.add(x.id)
            prev = x
            x = x.lru_next[pt]
        if x is None:
            errors.append(
                f"[{label}][{ct}] broken chain: lru_next is None "
                f"after node {prev.id if hasattr(prev, 'id') else 'head'}"
            )
        if len(visited) != len(lru.cache):
            errors.append(
                f"[{label}][{ct}] list={len(visited)} != cache={len(lru.cache)}"
            )

    def pretty_print(self) -> None:
        stack = [(self.root_node, 0)]
        while stack:
            node, indent = stack.pop()
            component_str = " ".join(
                f"{ct}={'yes' if node.component_data[ct].value is not None else 'no'}"
                for ct in self.tree_components
            )
            print(
                " " * indent,
                f"[{node.id}]",
                len(node.key),
                f"full_lock={node.component_data[BASE_COMPONENT_TYPE].lock_ref}",
                component_str,
            )
            for child in node.children.values():
                stack.append((child, indent + 2))
