from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_RADIX_CACHE,
    RadixCacheMetricsCollector,
    resolve_collector_class,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (
        ComponentType,
    )


@runtime_checkable
class PrefixCacheTrait(Protocol):
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int
    disable: bool


@dataclasses.dataclass
class MatchPrefixParams:
    """Unified parameters for match_prefix across different cache types"""

    key: RadixKey

    # Mamba specific
    cow_mamba: bool = False
    req: Optional[Req] = None


@dataclasses.dataclass
class InsertParams:
    """Unified parameters for insert across different cache types"""

    key: Optional[RadixKey] = None
    value: Optional[torch.Tensor] = None

    # Mamba specific
    mamba_value: Optional[torch.Tensor] = None

    # SWA specific
    prev_prefix_len: int = 0
    swa_evicted_seqlen: int = 0

    # General
    chunked: bool = False
    priority: int = 0

    # Hierarchical-cache specific: hand every inserted node to the host tier
    # regardless of the hit-count write-through heuristic. See
    # ``requests_forced_host_write_through`` below.
    force_host_write_through: bool = False


#: Attribute a caller sets on a ``Req`` to demand that the request's final
#: insert reaches the host tier in full.
FORCE_HOST_WRITE_THROUGH_ATTR = "force_host_write_through"


def requests_forced_host_write_through(req: Req) -> bool:
    """Whether this request's finishing insert must be written through to the
    host tier in full, bypassing the hit-count heuristic.

    The hierarchical cache normally writes a node to the host tier only once it
    has been hit ``write_through_threshold`` times -- a hit-rate heuristic, and
    a correct one for ordinary caching: a node that never gets a second hit is
    not worth the DMA, and dropping it costs nothing but a recompute.

    A HAND-OFF is the opposite situation. When ``--enable-kv-session-offload``
    demotes a session (spill budget exhausted, see ``_budget_demote``), the
    donated prefix is not a caching opportunity but the session's only surviving
    copy: the device slots are freed on the same finish. Leaving the heuristic
    in charge silently drops exactly the leaves under the threshold -- the
    newest tokens, the ones the session just produced. Requests marked here are
    written through in full instead.

    Unmarked requests -- everything that is not a hand-off -- keep the stock
    heuristic byte for byte.
    """
    return bool(getattr(req, FORCE_HOST_WRITE_THROUGH_ATTR, False))


@dataclasses.dataclass
class InsertResult:
    """Result of an insert operation"""

    prefix_len: int
    total_len: int = 0
    last_device_node: Any = None
    mamba_exist: bool = False
    inserted_host_node: Any = None

    #: #841: the host-only insert declined to adopt the fetched tail because
    #: attaching it would have broken the contiguous-backup law (see
    #: ``UnifiedRadixCache._insert_helper_host``). The span the caller had
    #: reserved for it therefore has no owner and must be released by the
    #: caller -- nothing in the tree will ever free it.
    host_span_unclaimed: bool = False


@dataclasses.dataclass
class EvictParams:
    """Unified parameters for evict across different cache types"""

    num_tokens: int = 0
    swa_num_tokens: int = 0
    mamba_num: int = 0


@dataclasses.dataclass
class EvictResult:
    """Result of an evict operation"""

    num_tokens_evicted: int = 0
    swa_num_tokens_evicted: int = 0
    mamba_num_evicted: int = 0


@dataclasses.dataclass
class IncLockRefResult:
    """Result of an inc_lock_ref operation."""

    delta: Optional[int] = None
    swa_uuid_for_lock: Optional[int] = None
    swa_uuid_for_host_lock: Optional[int] = None
    # Component nodes that were tombstones at acquire time. Replaying this set
    # at release prevents a short-lived lock from consuming a later load-back or
    # request lock after that tombstone becomes a valid device value.
    skip_lock_node_ids: dict[ComponentType, set[int]] = dataclasses.field(
        default_factory=dict
    )

    def to_dec_params(self) -> DecLockRefParams:
        """Convert to the corresponding DecLockRefParams for dec_lock_ref."""
        return DecLockRefParams(
            swa_uuid_for_lock=self.swa_uuid_for_lock,
            swa_uuid_for_host_lock=self.swa_uuid_for_host_lock,
            skip_lock_node_ids={
                component_type: set(node_ids)
                for component_type, node_ids in self.skip_lock_node_ids.items()
            },
        )


@dataclasses.dataclass
class DecLockRefParams:
    """Parameters for dec_lock_ref operation."""

    swa_uuid_for_lock: Optional[int] = None
    swa_uuid_for_host_lock: Optional[int] = None
    skip_lock_node_ids: dict[ComponentType, set[int]] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class DecLockRefResult:
    """Result of an dec_lock_ref operation."""

    delta: Optional[int] = None


@dataclasses.dataclass
class InitLoadBackParams:
    """Unified parameters for init_load_back across different cache types."""

    best_match_node: Any
    host_hit_length: int
    mem_quota: Optional[int] = None
    req: Optional[Req] = None


class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices  :   Indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
                            Reserved for L3 storage prefetch anchoring; L2 load_back
                            uses `best_match_node` instead.
        best_match_node :   Deepest node accepted by all component validators
                            during match_prefix. Anchor for every L2 host->device
                            load_back walk (FULL / SWA / ...). For legacy caches
                            that don't run multi-component validation, set this
                            equal to `last_host_node`.
        host_hit_length :   Number of Full-KV tokens that hit on host (CPU) and need to be
                            loaded back to device. Pure-KV cache semantics;
        swa_host_hit_length  :   Number of SWA tokens that hit on host (within the sliding
                            window) and will be load-back into the SWA device pool.
        mamba_host_hit_length:   Number of Mamba slots that hit on host and will be load-back
                            into the Mamba device pool. Typically 0 or 1.
        mamba_branching_seqlen: The mamba radix cache branching point, which is the longest
                                page-aligned position that could've been cache hit if there
                                exists a mamba state.
        state_anchor_depth: ABSOLUTE depth (same units as the match walk's `cum_tokens`)
                            of `best_match_node` -- i.e. of the deepest position at which
                            EVERY component accepted, the mamba resume predicate
                            (`is_resume_candidate`) included. This is the deepest
                            STATE-BEARING boundary on the matched path, and the only
                            position at which a KV prefix may end without the recurrent
                            state being ahead of it. ``None`` when no component makes a
                            state-bearing claim (pure-KV caches): callers must then leave
                            every length untouched, which keeps upstream's path
                            byte-for-byte what it was.
        key_match_depth:    How far the KEY matched, INDEPENDENT of any validator -- the
                            walk's final `cum_tokens`. The pair (`key_match_depth`,
                            `state_anchor_depth`) is the measurement #1039 needs: the KEY
                            reached this far, a STATE survived only that far. Their gap is
                            "the tree kept the shape and lost the anchor", which no single
                            number can express. ``None`` when not measured.
    """

    device_indices: torch.Tensor
    last_device_node: Any
    last_host_node: Any
    best_match_node: Any
    host_hit_length: int = 0
    swa_host_hit_length: int = 0
    mamba_host_hit_length: int = 0
    mamba_branching_seqlen: Optional[int] = None
    cache_protected_len: Optional[int] = None
    state_anchor_depth: Optional[int] = None
    key_match_depth: Optional[int] = None


def zero_match_result(tree_cache, match_result: MatchResult) -> MatchResult:
    if tree_cache.is_chunk_cache():
        # Chunk caches' match_prefix already returns a miss; no root_node to walk back to.
        return match_result
    root = tree_cache.root_node
    return match_result._replace(
        # [:0] keeps dtype and device of the original tensor (e.g. CUDA int64)
        # without allocating a fresh empty tensor.
        device_indices=match_result.device_indices[:0],
        last_device_node=root,
        last_host_node=root,
        best_match_node=root,
        host_hit_length=0,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
        # #1018: A ZEROED MATCH PROTECTS NOTHING, AND THIS FIELD HAD TO SAY SO.
        # Of MatchResult's nine fields this function replaced seven. It left
        # `mamba_branching_seqlen` deliberately -- the #928 refusals re-attach
        # it on purpose, so the re-prefill re-establishes the anchor at that
        # grid position -- and it left `cache_protected_len` by omission.
        #
        # That omission is a contradiction inside one object: `device_indices`
        # is emptied here, so `prefix_indices` is empty downstream, while
        # `cache_protected_len` still carries the PRE-refusal claim that the
        # tree owns N tokens. Its consumer takes the claim at face value
        # (`schedule_policy.py:231-234`, mirrored `schedule_batch.py:1410`):
        #     if match_result.cache_protected_len is not None:
        #         req.cache_protected_len = match_result.cache_protected_len
        #     else:
        #         req.cache_protected_len = len(req.prefix_indices)
        # so the request is re-admitted asserting a protected prefix it just
        # been told it does not have.
        #
        # MEASURED, boot_pp3solo_871178b77e_0829_121138: one second before the
        # first idle, all three ranks log the #928 refusal for rid bbeeeae6
        # ("re-prefilling", occurrence=1, rank-UNIFORM so not a told/local
        # divergence), and the announced re-prefill never produces a prefill
        # batch. The request's rows stay allocated and the census reports them:
        # 129 rows for a ~129-token prompt, and 11 / 10 / 6 for the shorter
        # smokes -- the deficit is the prompt, i.e. an ordinary admission whose
        # launch never happens.
        #
        # `None`, not 0, and the difference matters: None means "this match
        # makes no protection claim", which routes the consumer into the same
        # else-branch an ordinary miss uses and lets it derive the length from
        # the (now empty) prefix. Hardcoding 0 would state a second fact here
        # instead of deferring to the one derivation both paths already share.
        #
        # This is the zeroing CONTRACT, not a patch at the refusal: every
        # caller of this function -- strict-resume, both #928 arms, the
        # slot-starvation zeroing -- gets the same repair from one line.
        cache_protected_len=None,
        # #1040, same contract, same reason: a zeroed match reaches back to the
        # ROOT, so there is no accepted anchor left to point at. Leaving the
        # pre-refusal depth here would hand the extent chooser a state-bearing
        # boundary that this very call has just declared unusable -- the exact
        # shape the paragraph above describes for `cache_protected_len`. ZERO,
        # not None: None means "this cache has no state to align to at all"
        # (pure-KV), while the caller here HAS such a component and is saying
        # its anchor is gone. Collapsing the two would make an honest refusal
        # read as an upstream-shaped no-op.
        state_anchor_depth=(
            None if match_result.state_anchor_depth is None else 0
        ),
    )


class BasePrefixCache(ABC, PrefixCacheTrait):
    """Cache can be indexed by either rid or key."""

    metrics_collector: Optional[RadixCacheMetricsCollector] = (
        None  # metrics collector for the cache
    )

    # #616g: this iteration's rank-uniform availability floor (group MIN of
    # `available_size()`), published by the scheduler once per iteration when
    # the ranks' pools are uneven. Cache-MUTATION triggers -- eviction and
    # hicache load-back -- decide from it instead of from this rank's own
    # pool, which is what keeps the radix replicas identical and therefore
    # keeps `match_prefix` (and the extend token count it feeds) rank-uniform.
    # None means "pools agree, or a single rank": read the live local value,
    # exactly as before the fix. DECLARED HERE, not left to getattr on the
    # instance, so the attribute is visible in the type rather than only in
    # the path that happens to set it (#606).
    uniform_avail_floor: Optional[int] = None

    # #694: device allocations admitted since that floor was published. The
    # floor is a snapshot; without charging against it the evict trigger reads
    # a stale-optimistic number late in an iteration, SKIPS the eviction, and
    # the alloc raises on a tree full of evictable tokens. Reset by the
    # scheduler in the same call that publishes the next floor, so it never
    # outlives the number it corrects. DECLARED HERE for the #606 reason.
    uniform_admitted_since_floor: int = 0

    # #639: the same pin one tier down, for the HOST pool. The host KV pools
    # are rank-sized too (359652 / 287722 / 273336 slots on the crashing
    # boot), and `write_backup`'s admission is decided against them from a
    # replicated node length. Under `write_through` that verdict is a TREE
    # edit, not bookkeeping: `_evict_device_leaf` demotes a backed-up node
    # (stays in the tree) and DELETES one without a backup, so a rank-local
    # backup verdict makes the radix replicas diverge one tier BELOW the
    # #616g pins -- which is why those did not bind on it. Group MIN of the
    # host `available_size()`, published by the scheduler once per iteration
    # when the host pools are uneven. None means "pools agree, single rank,
    # or no host tier": read the live local value, exactly as before.
    uniform_host_avail_floor: Optional[int] = None

    # #645: host tokens this rank has admitted to the host tier SINCE the
    # floor above was published. The floor is computed once per iteration but
    # read at the end of it, so it counts as free every slot the iteration's
    # own backups have already taken; charging admissions against it makes
    # the gate current without a second collective. Rank-uniform by
    # construction -- only this gate's admissions are charged, and the gate
    # compares replicated numbers against a replicated node length. Reset by
    # the scheduler in the same call that publishes the next floor, so it
    # never outlives the number it corrects. DECLARED HERE for the same
    # reason as the floor itself (#606).
    uniform_host_admitted_since_floor: int = 0

    # #645: how many backups this rank has refused since that floor was
    # published rather than fall back to a rank-local host eviction. Exists
    # only to keep the warning to one line per floor instead of one per
    # insert; reset on the same cadence as the ledger above.
    uniform_host_refusals_since_floor: int = 0

    # #639b: the same pin for the MAMBA/SSM state pool. The two floors above
    # cover the KV token axis (device and host); neither reaches the mamba
    # slot pool, whose eviction is still decided rank-locally by
    # `alloc(1) is None`. That verdict is a TREE EDIT of exactly the kind the
    # other two close: `MambaComponent.evict_component` tombstones the mamba
    # component of a node (`cd.value = None`) and leaves its KV in place, so
    # the node survives on both ranks but stops satisfying
    # `create_match_validator` on the rank that evicted. `_all_valid` in
    # `UnifiedRadixCache._match_prefix_helper` advances the match only while
    # ALL components are resident, so that rank's match stops short and the
    # extend token axis diverges with it.
    #
    # A uniform POOL SIZE is not a uniform eviction OUTCOME. The mamba pool
    # IS min-reduced at startup, but occupancy is not: which node is
    # tombstoned depends on rank-local LRU order and lock_ref history, and
    # `_alloc_mamba_slot` returning None makes the caller SKIP a cache insert
    # rank-locally, so the divergence is self-sustaining once seeded.
    #
    # Group MIN of the mamba allocator's `available_size()`, published by the
    # scheduler once per iteration when the ranks' mamba occupancy is uneven.
    # None means "occupancy agrees, single rank, or no mamba pool": read the
    # live local value, exactly as before.
    uniform_mamba_avail_floor: Optional[int] = None

    def init_metrics_collector(self):
        from sglang.srt.runtime_context import get_server_args

        server_args = get_server_args()
        labels = {"cache_type": self.__class__.__name__}
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)
        radix_cache_cls = resolve_collector_class(
            server_args,
            STAT_LOGGER_ROLE_RADIX_CACHE,
            RadixCacheMetricsCollector,
        )
        self.metrics_collector = radix_cache_cls(labels=labels)

    def update_eviction_metrics(self, num_evicted: int, start_time: float):
        if self.metrics_collector is not None and num_evicted > 0:
            self.metrics_collector.observe_eviction_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_eviction_num_tokens(num_evicted)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        pass

    def supports_fast_match_prefix(self) -> bool:
        return False

    @abstractmethod
    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult:
        pass

    @abstractmethod
    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        pass

    @abstractmethod
    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        pass

    def evictable_size(self):
        return 0

    def full_evictable_size(self):
        return 0

    def swa_evictable_size(self):
        return 0

    def protected_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def swa_protected_size(self):
        return 0

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Preparing KV cache loading from host to device.
        """
        raise NotImplementedError()

    def ready_to_load_host_cache(self) -> Any:
        """
        Notify the cache controller to start the KV cache loading
        """
        raise NotImplementedError()

    def flush_write_through_acks(self) -> None:
        """Release lock_ref on radix-tree nodes whose write-through has completed.

        Lightweight operation that only processes finished write acks.
        No-op for caches without hierarchical write-through support.
        """
        pass

    def check_hicache_events(self) -> Any:
        """
        Check HiCache related activities to update radix tree and synchronize across TP workers if needed
        """
        raise NotImplementedError()

    def take_events(self):
        return []

    def supports_swa(self) -> bool:
        return False

    def swa_reprefill_tail_tokens(self) -> int:
        # Only the unified_kv compress-only HiCache layout needs to hold back a
        # trailing sliding window for re-prefill; every other cache keeps SWA
        # content-stable and overrides this where relevant.
        return 0

    def supports_mamba(self) -> bool:
        return False

    def supports_streaming_session(self) -> bool:
        return False

    def release_session(self, session_id: str) -> None:
        pass

    def release_radix_session(self, session_id: str) -> None:
        pass

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"
