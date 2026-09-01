from __future__ import annotations

import atexit
import heapq
import json
import logging
import os
import threading
import time
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import (
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
)
from sglang.srt.mem_cache.hicache_collective import (
    HiCacheCollectiveError,
    bounded_recv,
    bounded_wait,
    collective_rank_desc,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PrefetchTimeoutConfig,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation as HybridPrefetchOperation,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    attach_hybrid_dsa_pool_to_hiradix_cache,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

from sglang.srt.mem_cache import hicache_demotion as _demotion

logger = logging.getLogger(__name__)


class HiRadixCache(RadixCache):
    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = get_mha_host_pool_cls(self.kv_cache)(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, DSATokenToKVPool):
            # Filled by attach_hybrid_dsa_pool_to_hiradix_cache after storage extra_config is parsed.
            self.token_to_kv_pool_host = None
        elif isinstance(self.kv_cache, MiniMaxSparseKVPool):
            # Filled by attach_hybrid_minimax_sparse_pool_to_hiradix_cache.
            self.token_to_kv_pool_host = None
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            raise ValueError("HiRadixCache only supports MHA, MLA, DSA, and MSA models")

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.pp_group = params.pp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        # Deadline for every cross-rank control collective issued from this
        # cache (#630). Set before the first collective below
        # (_symmetrize_prefetch_capacity) can run. Without it a dead TP peer or
        # a PP rank that never posts the matching receive parks this rank until
        # the gloo group's two-hour default timeout expires -- the PP + disk
        # HiCache warmup wedge, where health stays 503 with nothing logged.
        self.collective_timeout_s = envs.SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S.get()
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_config,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: support more timeout check functions
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.load_cache_event = threading.Event()
        if isinstance(self.kv_cache, DSATokenToKVPool):
            attach_hybrid_dsa_pool_to_hiradix_cache(
                self,
                params,
                server_args,
                extra_config=extra_config,
                prefetch_threshold=prefetch_threshold,
                enable_storage_metrics=self.enable_storage_metrics,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
            )
        elif isinstance(self.kv_cache, MiniMaxSparseKVPool):
            from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
                attach_hybrid_minimax_sparse_pool_to_hiradix_cache,
            )

            attach_hybrid_minimax_sparse_pool_to_hiradix_cache(
                self,
                params,
                server_args,
                extra_config=extra_config,
                prefetch_threshold=prefetch_threshold,
                enable_storage_metrics=self.enable_storage_metrics,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
            )
        else:
            self.cache_controller = HiCacheController(
                params.token_to_kv_pool_allocator,
                self.token_to_kv_pool_host,
                self.page_size,
                self.tp_group,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
                pp_group=self.pp_group,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
                enable_storage_metrics=self.enable_storage_metrics,
            )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_config=prefetch_timeout_config,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        # #610: must run after the controller exists and before any request is
        # served, so every rank enters the capacity reduce from the same point.
        self._symmetrize_prefetch_capacity()

        # #810: bound the write-through consumer of a STAGING host tier. Built
        # AFTER `_symmetrize_prefetch_capacity` above, so the capacity is the
        # complement of the group-agreed prefetch reservation rather than of a
        # rank-local one -- a rank-dependent admission bound on this path is
        # exactly the #645 defect. None under `--hicache-host-role retention`,
        # which is the default and leaves this path unchanged.
        from sglang.srt.mem_cache.staging_write_ring import build_staging_write_ring

        self.staging_write_ring = build_staging_write_ring(
            server_args, self.cache_controller
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        # track per-request tokens loaded from storage (L3 hits)
        # key: request_id, value: number of tokens actually loaded from storage
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.work_list: List[torch.distributed.Work] = []
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

        # Detach storage backend automatically on process shutdown
        atexit.register(self.shutdown)

        self.evictable_host_leaves = set()

        super().__init__(params=params)

    def _wait_bounded(self, work, label: str) -> None:
        """Wait for ``work`` with a deadline, or raise a named error.

        Same mechanism and rationale as ``UnifiedRadixCache._wait_bounded``;
        both delegate to ``hicache_collective.bounded_wait`` so there is one
        implementation rather than two that drift (#630 was that drift: this
        class kept raw blocking calls after #259 bounded the unified one).

        The healthy case costs nothing measurable: a CPU collective between
        local ranks completes inside the initial spin window, so no sleep is
        ever reached.
        """
        bounded_wait(work, label, self.collective_timeout_s, collective_rank_desc(self))

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
        """True when the per-rank host pools are asymmetric and the prefetch
        control path must therefore be decided by the GROUP rather than per rank.

        #610. HiRadixCache never received the #580 symmetrization that
        UnifiedRadixCache carries (`unified_radix_cache.py:531`), yet its
        prefetch control path runs the same collectives: the storage-hit MIN
        reduce in `query_storage_hit_length`, and `check_prefetch_progress`'s
        can_terminate MAX reduce plus min_completed_tokens MIN reduce. Under
        weighted DCP the host pool is sized from the per-rank DEVICE pool
        (`memory_pool_host.py:126`, `size = int(device_pool.size * ratio)`), so
        allocation success and `prefetch_rate_limited()` answer differently per
        rank and those collectives were entered by a SUBSET of the group.

        Same predicate as the unified cache: storage on, real controller,
        multi-rank, non-uniform token vector. Stock even-TP HiCache -- uniform
        host pools, identical alloc/free histories -- never trips it, so that
        path stays byte-identical.
        """
        from sglang.srt.distributed.utils import uneven_dcp_active

        return (
            self.enable_storage
            and getattr(self, "cache_controller", None) is not None
            and self.tp_world_size > 1
            and uneven_dcp_active()
        )

    def prefetch_participation_is_collective(self) -> bool:
        """True when whether to prefetch is decided by the GROUP, not per rank.

        `Scheduler._prefetch_kvcache` (scheduler.py:2907) and
        `DecodeHiCachePreallocMixin._start_hicache_prefetch`
        (decode_hicache_mixin.py:134) both probe for this method before applying
        their own rank-local gate: when it answers True they call
        `prefetch_from_storage` unconditionally and pass their local verdict as
        `locally_eligible` instead of returning early. Those call sites were
        already written for #580/#607-E; defining the predicate here is what
        connects HiRadixCache to them.
        """
        return self._hicache_prefetch_symmetric()

    def _symmetrize_prefetch_capacity(self) -> None:
        """Derive the speculative-prefetch capacity limit from the MIN host-pool
        size across the group (#610, mirroring unified_radix_cache.py:543).

        Under weighted DCP the host pools differ per rank, so the stock per-rank
        `int(0.5 * mem_pool_host.size)` limit (cache_controller.py:477) makes
        `prefetch_rate_limited()` trip on different iterations on different
        ranks -- a divergence UPSTREAM of the participation vote, which would
        desync the vote itself. The shared MIN makes the rate-limit gate trip in
        lockstep. Gated so the even-TP path keeps its per-rank limit unchanged.
        """
        if not self._hicache_prefetch_symmetric():
            return
        cc = self.cache_controller
        if getattr(cc, "mem_pool_host", None) is None:
            # The gate above is rank-uniform (config + uneven_dcp_active), so a
            # rank-local early return HERE would leave the peers alone in the
            # all_reduce below. Name it instead of hanging.
            raise HiCacheCollectiveError(
                "cache controller has no mem_pool_host while prefetch "
                "symmetrization is active; the peer ranks are entering the "
                "capacity all_reduce and this rank cannot."
            )
        size_tensor = torch.tensor([int(cc.mem_pool_host.size)], dtype=torch.long)
        self._all_reduce_attn_groups(
            size_tensor,
            torch.distributed.ReduceOp.MIN,
            label="symmetrize_prefetch_capacity",
        )
        # #968/#1065: one authority for the halved-with-floor budget -- see
        # `prefetch_capacity_limit_for` (a floor applied at only one caller
        # would be undone by this symmetrize pass).
        from sglang.srt.managers.cache_controller import (
            prefetch_capacity_limit_for,
        )

        cc.prefetch_capacity_limit = prefetch_capacity_limit_for(
            int(size_tensor[0].item())
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

        Bounded for the same reason as every other collective here (#630):
        these are isends on the PP gloo ``cpu_group``, and a downstream PP rank
        that never posts the matching receive parks this rank until the group's
        two-hour timeout expires.
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

        The following diagram illustrates the behavior of _pp_sync.

        time  | pp0                     | pp1                     | pp2
        ------|-------------------------|-------------------------|-----------------------------
        0     | _pp_sync(data=1) starts | _pp_sync(data=?) starts | _pp_sync(data=?) starts
        1     | _pp_sync(data=1) ends   |                         |
        2     |                         | _pp_sync(data=1) ends   |
        3     |                         |                         | _pp_sync(data=1) ends

        _pp_sync requires no synchronization point among ranks. The following case may also happen.

        time  | pp0                     | pp1                     | pp2
        ------|-------------------------|-------------------------|-----------------------------
        0     | _pp_sync(data=1) starts |                         |
        1     | _pp_sync(data=1) ends   |                         |
        2     |                         | _pp_sync(data=?) starts |
        3     |                         | _pp_sync(data=1) ends   |
        4     |                         |                         | _pp_sync(data=?) starts
        5     |                         |                         | _pp_sync(data=1) ends
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
            # Make a copy of data, so that the caller is safe to modify `data` after this call.
            # This is cheap, as _pp_sync is not to be used for transmitting large data.
            copy_of_data = data.clone()
            send_work = torch.distributed.isend(
                copy_of_data,
                group_dst=self.pp_rank + 1,
                group=self.pp_group,
                tag=P2PTag.HIRADIX_PP_SYNC,
            )
            self.work_list.append(send_work)

    def shutdown(self):
        """Best-effort auto-detach of storage backend on process shutdown.

        This keeps startup and runtime behavior consistent: if a backend was attached
        (either via CLI args or via admin API), we attempt to detach it on exit.
        """
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_config: PrefetchTimeoutConfig,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_config = prefetch_timeout_config
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
            existing_collector = getattr(self, "storage_metrics_collector", None)
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
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to "
                    "avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) storage backend at runtime.

        This will start storage threads inside `HiCacheController` and enable
        prefetch/backup paths. Caller must ensure there are no running/queued
        requests to avoid races.
        """
        # Validate inputs first (no side effects).
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        # If already enabled:
        # - backend unchanged: treat as success, update policies only.
        # - backend changed: treat as failure, do NOT update policies.
        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type

            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                    logger.info(
                        f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
                    )
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                    logger.info(f"Set hicache_write_policy to {hicache_write_policy}")
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )

            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        # Not enabled: update policies before controller attach so storage threads observe new values.
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
            )

        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_config,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json '{storage_backend_extra_config_json}': {e}",
            )

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                **self._get_hybrid_storage_attach_kwargs(),
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_config=prefetch_timeout_config,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) storage backend at runtime.

        Caller must ensure there are no running/queued requests to avoid races.
        """
        try:
            # Drain any pending control queues before tearing down storage threads/backend.
            # IMPORTANT: this must happen before we clear `ongoing_*`, otherwise acks/releases
            # cannot be matched to nodes and may leak host pages / locks.
            self._drain_storage_control_queues_local()
            # Idempotent detach: always ask controller to best-effort cleanup, even if
            # `self.enable_storage` is already False (may be leftover state from a
            # previous partial detach).
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # Do NOT crash the server for admin operations. Return failure with detail.
            return False, f"Failed to detach HiCache storage backend: {e}"

        # Best-effort cleanup of any leftover bookkeeping.
        self._drain_storage_control_queues_local()
        # After controller threads are fully stopped, it's safe to force-release any
        # leftover pending ops (e.g., async prefetch/backup that didn't get a revoke/ack).
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def _force_release_pending_storage_ops(self):
        """Force release any leftover pending prefetch/backup bookkeeping.

        This is a safety net for detach/shutdown paths. It assumes storage threads
        have been stopped already (via controller.detach), so no concurrent access
        to these structures should happen.
        """
        cc = self.cache_controller

        # Force release leftover prefetch ops: free pre-allocated host pages and
        # drop the host protection on the matched prefix node.
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, token_ids, host_indices, _operation = info
                except Exception:
                    # Unexpected shape; just drop it.
                    self.ongoing_prefetch.pop(req_id, None)
                    continue

                try:
                    if host_indices is not None:
                        cc.mem_pool_host.free(host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )

                try:
                    last_host_node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )

                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass

                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        # Force release leftover backup ops: drop host protection on nodes.
        try:
            for ack_id, node in list(self.ongoing_backup.items()):
                try:
                    node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                # #810: the page leaves the drain here, so its ring charge does
                # too. A forced release that skipped this would shrink the ring
                # for the rest of the process's life.
                if self.staging_write_ring is not None:
                    self.staging_write_ring.release(ack_id)
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _drain_storage_control_queues_local(self):
        """Drain storage control queues without TP synchronization.

        This is intended for shutdown/detach paths where we want to make best-effort
        cleanup even if queue sizes temporarily differ across ranks.
        """
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_backup=None,
            n_release=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is not None:
                    last_host_node, token_ids, _, _ = info
                    last_host_node.release_host()
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0

        def _drain_backup():
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    entry.release_host()
                # #810: the storage write acked -- this is the drain the
                # staging ring measures its residency against. Outside the
                # `entry is not None` arm on purpose: the charge is keyed by
                # the operation, so it is retired whenever the operation
                # retires, whether or not the node survived to be found.
                if self.staging_write_ring is not None:
                    self.staging_write_ring.release(ack_id)
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_backup()
        _drain_release()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_config, hicache_storage_pass_prefix_keys)
        """
        # Parse extra config if provided. Extra config can be a JSON string or a json/toml/yaml file path prefixed with "@".
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    # Read config from a json/toml/yaml file
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    # read config from JSON string
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        defaults = PrefetchTimeoutConfig()
        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop(
            "prefetch_timeout_base", defaults.base
        )  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", defaults.per_ki_token
        )  # seconds per 1024 tokens
        prefetch_timeout_max = extra_config.pop(
            "prefetch_timeout_max", defaults.max
        )  # seconds, upper bound for the linear timeout
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(prefetch_timeout_max, (int, float)):
            raise ValueError(
                f"prefetch_timeout_max must be number, got {type(prefetch_timeout_max).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        prefetch_timeout_config = PrefetchTimeoutConfig(
            base=float(prefetch_timeout_base),
            per_ki_token=float(prefetch_timeout_per_ki_token),
            max=float(prefetch_timeout_max),
        )

        return (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_config,
            hicache_storage_pass_prefix_keys,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        # Clear per-request tracking dicts
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.evictable_host_leaves.clear()
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def _get_extra_pools(self) -> dict:
        if not isinstance(self.cache_controller, HybridCacheController):
            return {}
        if isinstance(self.kv_cache, DSATokenToKVPool) or (
            isinstance(self.kv_cache, MiniMaxSparseKVPool)
            and self.kv_cache.index_k_pool is not None
        ):
            pool = PoolTransfer(
                name=PoolName.INDEXER,
                hit_policy=PoolHitPolicy.ALL_PAGES,
                indices_from_pool=PoolName.KV,
            )
            return {"extra_pools": [pool]}
        else:
            return {}

    def _get_hybrid_storage_attach_kwargs(self) -> dict:
        """Extra kwargs for attach_storage_backend when controller is HybridCacheController."""
        if isinstance(self.cache_controller, HybridCacheController):
            return {"host_pools": self.cache_controller.mem_pool_host.entries}
        return {}

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def storage_capacity_stats(self) -> Optional[dict]:
        """Capacity limits and on-disk usage of the attached backend, if any."""
        if not self.enable_storage:
            return None
        backend = self.cache_controller.storage_backend
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

        Growing takes effect immediately for subsequent writes. Shrinking
        evicts LRU victims inline before returning, so the call blocks for as
        long as the unlinks take; see ``LRUFileEvictor.set_limits`` for the
        exact post-shrink target and the in-flight-write carve-out.

        Unlike attach/detach this does not start or stop any thread and does
        not touch the device or host tiers, so it does not require an idle
        scheduler -- the evictor's own lock serializes it against the backup
        and prefetch threads.
        """
        if not self.enable_storage:
            return False, "HiCache storage backend is not enabled.", None

        backend = self.cache_controller.storage_backend
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

    def write_backup(self, node: TreeNode, write_back=False) -> int:
        # Backup invariant (for write-through mode): backed-up nodes must form a
        # contiguous prefix from root — no gaps.  Skip if parent isn't backed
        # up yet;
        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            return 0

        # #639: RANK-UNIFORM host admission, the same pin the sibling class
        # carries. This class asks the question by attempting the write and
        # reading a None back rather than by testing `available_size()`
        # first, but the quantity that decides the answer is the same
        # rank-sized host pool, and the consequence is the same: under
        # `write_through` a node that fails to back up is DELETED from the
        # tree at its next device eviction while a backed-up one is demoted
        # and stays matchable, so a rank-local verdict makes the radix
        # replicas diverge and the extend token count with them.
        #
        # Refuse up front when the group floor cannot hold the node, so every
        # rank reaches the same verdict before any rank allocates. None (host
        # pools agree, single rank, no host tier) leaves the path exactly as
        # it was -- this class is not the one the wedging rig instantiates,
        # and it is pinned here because #616g's load-back fix spent a boot
        # sitting in the class that deployment never built.
        from sglang.srt.mem_cache.common import (
            note_uniform_host_admitted,
            note_uniform_host_refusal,
            uniform_host_avail_for_backup,
            uniform_host_floor_active,
        )

        mem_pool_host = getattr(self.cache_controller, "mem_pool_host", None)
        floor_active = uniform_host_floor_active(self) and mem_pool_host is not None
        if floor_active:
            host_avail = uniform_host_avail_for_backup(self, mem_pool_host)
            if host_avail < len(node.value):
                return 0

        # #810: the STAGING bound, taken BEFORE the allocation rather than
        # after it fails. Under `--hicache-host-role staging` the tier is a
        # drain buffer, so the undrained write-through set must leave room for
        # the read consumer; a refusal here costs one un-backed-up node, the
        # same thing an exhausted tier costs today, but it is COUNTED and it
        # never reaches the rank-local `evict_host` below. `None` under the
        # default role skips the whole gate.
        ring = self.staging_write_ring
        if ring is not None and not ring.admit(node.id, len(node.value)):
            return 0

        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            **self._get_extra_pools(),
        )
        if host_indices is None:
            # #645: the retry's `evict_host` is a RANK-LOCAL tree edit, and
            # it is reached on a RANK-LOCAL condition -- this rank's write
            # failed. Its victims are host leaves chosen by this rank's own
            # `eviction_strategy` out of this rank's own
            # `evictable_host_leaves`, and it deletes them from the tree
            # (`x.parent.children.pop(key)` below), so a rank that takes this
            # branch while its peers do not ends up with a shorter
            # `match_prefix` than they have. That is the divergence the
            # detector reports.
            #
            # Under an active floor the gate above already refused every node
            # the group budget cannot hold, so a failure here means this
            # rank's live pool fell below the group's own agreed budget --
            # something the ledger makes arithmetically impossible for the
            # backups this gate admitted, and which otherwise signals an
            # allocation from OUTSIDE this gate (prefetch). Refusing is the
            # only answer available without a collective: it costs one
            # un-backed-up node, while evicting costs a divergent tree. The
            # sibling class carries the same guard for the same reason; see
            # `uniform_host_floor_active`.
            #
            # No floor (pools agree, single rank, no host tier) keeps the
            # eviction retry exactly as it was.
            if floor_active:
                if note_uniform_host_refusal(self) == 1:
                    logger.warning(
                        "#645: host backup refused after a write failure under "
                        "an active rank-uniform floor (%d tokens). Skipping "
                        "the rank-local host eviction, which would diverge the "
                        "radix replicas. Logged once per published floor.",
                        len(node.value),
                    )
                if ring is not None:
                    ring.abort(node.id)
                return 0
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                **self._get_extra_pools(),
            )
        if host_indices is not None:
            # #645: charge the admission against the published floor, so the
            # next backup in THIS iteration decides against what is left
            # rather than against the iteration-start snapshot. No-op when no
            # floor is active.
            note_uniform_host_admitted(self, len(node.value))
            node.host_value = host_indices.clone()
            assert len(node.host_value) > 0
            self._track_write_through_node(node, len(node.key))
            if not write_back:
                self.inc_lock_ref(node)
        else:
            # #810: the write failed after the ring admitted it, so the page
            # never reaches the drain and its admission must not stay charged.
            if ring is not None:
                ring.abort(node.id)
            return 0

        return len(host_indices)

    def _track_write_through_node(self, node: TreeNode, backup_len: int) -> None:
        node.write_through_pending_id = node.id
        self.ongoing_write_through[node.id] = (node, backup_len, [node])

    def _replace_pending_write_through_node(
        self, old_node: TreeNode, new_nodes: List[TreeNode]
    ) -> None:
        ack_id = old_node.write_through_pending_id
        if ack_id is None:
            return

        pending = self.ongoing_write_through.get(ack_id)
        if pending is None:
            return

        lock_node, backup_len, publish_nodes = pending
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
        self.ongoing_write_through[ack_id] = (lock_node, backup_len, updated_nodes)

    def _finish_write_through_ack(self, ack_id: int, *, release_lock: bool) -> None:
        lock_node, backup_len, publish_nodes = self.ongoing_write_through.pop(ack_id)
        for node in publish_nodes:
            if node.write_through_pending_id == ack_id:
                node.write_through_pending_id = None
            # DMA confirmed -- block is now on host.
            self._record_store_event(node, medium=StorageMedium.CPU)
        # #810: end of the ADMITTED phase. The device->host copy has landed, so
        # the admission taken in `write_backup` is retired here -- before the
        # storage hand-off below, which takes its own charge keyed by the
        # storage operation id. Releasing first keeps the two phases from
        # double-counting the same page, and retiring the node-keyed charge at
        # exactly one site keeps a node SPLIT (one ack fanning out into several
        # storage backups) from stranding it.
        if self.staging_write_ring is not None:
            self.staging_write_ring.release(ack_id)
        if self.enable_storage:
            self.write_backup_storage(lock_node, backup_len)
        if release_lock:
            self.dec_lock_ref(lock_node)

    def write_backup_storage(self, node: TreeNode, backup_len: Optional[int] = None):
        # Recover pre-split data via walk-and-concat if node was split.
        # prefix_keys anchored at chain top to avoid double-counting.
        if backup_len is None or len(node.key) == backup_len:
            top, key, hash_value, host_value = (
                node,
                node.key,
                node.hash_value,
                node.host_value,
            )
        else:
            top, key, hash_value, host_value = self._concat_split_chain(
                node, backup_len
            )

        prefix_keys = (
            top.get_prefix_hash_values(top.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        operation_id = self.cache_controller.write_storage(
            host_value, key, hash_value, prefix_keys, **self._get_extra_pools()
        )
        self.ongoing_backup[operation_id] = node
        node.protect_host()
        # #810: the DRAIN phase begins here. `protect_host` keeps these tokens
        # resident until the backup acks, so they are the bytes a staging tier
        # is sized from; the charge cannot be refused (the page is already on
        # the host) but it must be counted, or the next admission decides
        # against an occupancy that hides the whole drain queue.
        if self.staging_write_ring is not None:
            self.staging_write_ring.occupy(operation_id, len(host_value))

    def _concat_split_chain(self, node: TreeNode, backup_len: int):
        """Recover enqueue-time key/hash/host by walking the split chain."""
        chain, accumulated = [], 0
        current = node
        while current is not self.root_node and accumulated < backup_len:
            chain.append(current)
            accumulated += len(current.key)
            current = current.parent
        assert accumulated == backup_len, (
            f"backup chain length mismatch for node {node.id}: "
            f"expected {backup_len}, got {accumulated}"
        )
        chain.reverse()  # parent-first
        top = chain[0]
        if top.key.is_bigram:
            # Bigram segments share boundary tokens; drop overlap after first.
            token_ids = list(chain[0].key.token_ids)
            for n in chain[1:]:
                token_ids.extend(n.key.token_ids[1:])
        else:
            token_ids = []
            for n in chain:
                token_ids.extend(n.key.token_ids)
        key = RadixKey(token_ids, top.key.extra_key, top.key.is_bigram)

        if all(n.hash_value is not None for n in chain):
            hash_value = []
            for n in chain:
                hash_value.extend(n.hash_value)
        else:
            hash_value = None
        host_value = torch.cat([n.host_value for n in chain])
        return top, key, hash_value, host_value

    def _inc_hit_count(
        self, node: TreeNode, chunked=False, force_host_write_through: bool = False
    ):
        # skip the hit count update for chunked requests
        if self.cache_controller.write_policy == "write_back" or chunked:
            # A hand-off (see requests_forced_host_write_through) still has to
            # reach the host tier under write_back: that policy stages nodes
            # only at eviction time and drops them when the host pool is full,
            # which for a donated session is the same silent loss.
            if not force_host_write_through:
                return
        else:
            node.hit_count += 1

        if not node.backuped and (
            force_host_write_through or node.hit_count >= self.write_through_threshold
        ):
            # write to host if the node is not backuped
            self.write_backup(node)

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        self._finish_write_through_ack(ack_id, release_lock=False)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # Every rank must enter the all_reduce below; ongoing_write_through can
        # diverge across ranks (e.g. write_backup returning 0 on a subset under
        # host memory pressure), so a conditional skip desyncs the NCCL op
        # sequence and deadlocks under TP > 1. (Matches UnifiedRadixCache.)
        finish_count = 0
        if self.pp_rank == 0:
            for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                if not finish_event.query():
                    break
                finish_count += 1
        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(
            finish_count_tensor, torch.distributed.ReduceOp.MIN, label="writing_check"
        )
        finish_count = finish_count_tensor.item()

        if finish_count > 0:
            logger.debug(f"Process {finish_count} write back operations")
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                self._finish_write_through_ack(ack_id, release_lock=True)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        if self.pp_rank == 0:
            for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
                if not finish_event.query():
                    break
                finish_count += 1
        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(
            finish_count_tensor, torch.distributed.ReduceOp.MIN, label="loading_check"
        )
        finish_count = finish_count_tensor.item()

        if finish_count > 0:
            logger.debug(f"Process {finish_count} load operations")
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_load_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node)
            finish_count -= 1

    def is_load_back_event_done(self, consumer_index: int) -> bool:
        """Return True after the local load-back event is complete."""
        if consumer_index < 0:
            return True

        finish_event = self.cache_controller.layer_done_counter.events[
            consumer_index
        ].finish_event
        if not finish_event.query():
            return False

        self.loading_check()
        return True

    def evictable_size(self):
        return self.evictable_size_

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def _update_host_leaf_status(self, node: TreeNode):
        if not node.evicted or node.lock_ref > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return

        for child in node.children.values():
            if child.backuped:
                if node in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(node)
                return

        if node not in self.evictable_host_leaves:
            self.evictable_host_leaves.add(node)

    def evict(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        if self.cache_controller.write_policy == "write_back":
            num_evicted = self._evict_write_back(num_tokens)
        else:
            num_evicted = self._evict_write_through(num_tokens)
        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _make_eviction_heap(self):
        heap = [
            (self.eviction_strategy.get_priority(node), node)
            for node in self.evictable_leaves
        ]
        heapq.heapify(heap)
        return heap

    def _promote_parent(self, node: TreeNode, heap) -> None:
        # Once all of a node's children are evicted, it becomes a device leaf.
        p = node.parent
        if p is not self.root_node and all(c.evicted for c in p.children.values()):
            heapq.heappush(heap, (self.eviction_strategy.get_priority(p), p))

    def _evict_write_through(self, num_tokens: int) -> int:
        """write_through / write_through_selective: drop non-backuped leaves,
        demote already-backuped ones. Nothing is staged to host during eviction,
        so this is a plain on-the-fly pass.
        """
        heap = self._make_eviction_heap()
        num_evicted = 0
        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x.lock_ref > 0:
                continue
            if x.backuped:
                num_evicted += self._evict_backuped(x)
            else:
                num_evicted += self._evict_regular(x)
            self._promote_parent(x, heap)
        return num_evicted

    def _evict_write_back(self, num_tokens: int) -> int:
        """eviction for write_back mode: demote already-backuped leaves, stage non-backuped ones to host if possible, otherwise drop them.
        note this path will be deprecated in the future.
        """
        heap = self._make_eviction_heap()
        num_evicted = 0
        staged: List[Tuple[TreeNode, torch.Tensor]] = []

        def flush_staged() -> None:
            if not staged:
                return
            self.writing_check(write_back=True)
            for node, device_indices in staged:
                self.cache_controller.evict_device(device_indices)
                node.release_host()
            staged.clear()

        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x.lock_ref > 0:
                continue
            if x.backuped:
                num_evicted += self._evict_backuped(x)
            elif self.write_backup(x, write_back=True) > 0:
                x.protect_host()
                staged.append((x, x.value))
                num_evicted += self._detach_backuped(x)
            else:
                flush_staged()
                num_evicted += self._drop_subtree_no_host(x)
            self._promote_parent(x, heap)
        flush_staged()
        return num_evicted

    def _detach_backuped(self, node: TreeNode) -> int:
        # detach nodes from tree while keeping device slots, for write-back eviction
        self._record_remove_event(node, medium=StorageMedium.GPU)
        num_evicted = len(node.value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        # update leaf status for the parent because the node is evicted
        self._update_leaf_status(node.parent)
        return num_evicted

    def _evict_backuped(self, node: TreeNode):
        # #703: the insert-time ack writes storage once; a node whose storage
        # write was refused then is never retried, and this demotion is where
        # that is noticed. No-op when the node is already stored or the
        # feature is off.
        _demotion.demote_on_device_evict(self, node)
        device_indices = node.value
        num_evicted = self._detach_backuped(node)
        self.cache_controller.evict_device(device_indices)
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host -- emit BlockRemoved
        assert len(node.children) == 0, f"non-leaf, {node.id=}"

        self._record_remove_event(node)
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def _drop_subtree_no_host(self, root: TreeNode) -> int:
        nodes = []
        stack = [root]
        while stack:
            n = stack.pop()
            nodes.append(n)
            stack.extend(n.children.values())

        if any(n.host_ref_counter > 0 for n in nodes):
            return 0

        logger.warning(
            "write_back: KV cache on device are dropped without backup due to host memory pressure, subtree root %d, num_nodes %d",
            root.id,
            len(nodes),
        )

        freed_device = 0
        for n in nodes:
            if n.host_value is not None:
                self._record_remove_event(n, medium=StorageMedium.CPU)
                self.cache_controller.evict_host(n.host_value)
                n.host_value = None
            if n.value is not None:
                self._record_remove_event(n, medium=StorageMedium.GPU)
                self.cache_controller.mem_pool_device_allocator.free(n.value)
                freed_device += len(n.value)
                self.evictable_size_ -= len(n.value)
                n.value = None
            self.ongoing_write_through.pop(n.id, None)
            self.evictable_leaves.discard(n)
            self.evictable_host_leaves.discard(n)

        key = root.key.child_key(self.page_size)
        root.parent.children.pop(key, None)
        self._update_leaf_status(root.parent)
        self._update_host_leaf_status(root.parent)
        return freed_device

    def evict_host(self, num_tokens: int):
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            if x.host_ref_counter > 0:
                continue

            # Block deleted entirely (GPU already evicted, now CPU freed) --
            # emit remove(CPU) so the router drops the host-tier entry.
            self._record_remove_event(x, medium=StorageMedium.CPU)
            # #703: last chance. evict_host frees the very bytes a storage
            # write reads, and nothing else on this path persists them, so a
            # prefix that never reached disk dies here. Enqueue-only and
            # bounded; off by default.
            _demotion.demote_before_host_evict(self, x)
            num_evicted += self.cache_controller.evict_host(x.host_value)

            key = x.key.child_key(self.page_size)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._update_host_leaf_status(x.parent)

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:

        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        result = self.inc_lock_ref(ancester_node)
        delta = result.delta

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        # Protect the nodes being loaded from host eviction.
        for n in nodes_to_load:
            n.protect_host()

        # #616g: the SECOND rank-local source of radix divergence, and the one
        # that needs no eviction at all to fire. Load-back EXTENDS this rank's
        # device prefix; whether it succeeds depends on this rank's own free
        # device space. Under uneven pools the roomy rank loads a prefix back
        # while the tight ranks fail and skip it, so the device trees diverge
        # directly -- and the roomy rank then matches MORE and computes FEWER
        # extend tokens, which is the observed direction of the 21:52:25
        # specimen (rank 0, the largest pool at 179825 tokens, reducing 1690
        # against its peers' 1818).
        #
        # Decide from the group floor instead: if the BINDING rank cannot hold
        # the load-back, no rank attempts it. Uniform in both outcomes, and the
        # floor being a MIN means a rank that clears it has the room.
        floor = getattr(self, "uniform_avail_floor", None)
        if floor is not None and floor < len(host_indices):
            self.dec_lock_ref(ancester_node)
            for n in nodes_to_load:
                n.release_host()
            return None
        device_indices = self.cache_controller.load(
            host_indices=host_indices,
            node_id=last_hit_node.id,
            **self._get_extra_pools(),
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                **self._get_extra_pools(),
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            for n in nodes_to_load:
                n.release_host()
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        for n in nodes_to_load:
            n.release_host()
        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)].clone()
            offset += len(node.host_value)
            # Block promoted from host to GPU -- emit store(GPU) so downstream
            # indexers see it as device-local again.
            self._record_store_event(node, medium=StorageMedium.GPU)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return device_indices

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        last_node = params.best_match_node
        mem_quota = params.mem_quota
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            self._empty_match_result.device_indices,
            last_node,
        )

    def query_storage_hit_length(
        self,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        locally_eligible: bool = True,
    ) -> int:
        # #610: decide the MODE before any rank-local predicate runs. Under
        # `symmetric` the MIN reduce below is the group's decision point and
        # nothing between here and it may `return`.
        symmetric = self._hicache_prefetch_symmetric()

        # RANK-LOCAL, all of them. `locally_eligible` carries the caller's own
        # gate (`_build_decode_prefix_match` tests `last_host_node.backuped`,
        # i.e. "full KV in THIS rank's host pool"); `prefetch_rate_limited()`
        # reads the per-rank prefetch_tokens_occupied counter against a limit
        # derived from the per-rank host pool. Under weighted DCP both drift
        # apart across ranks, and returning on either one left the peers alone
        # in the reduce -- the #580 shape, still live here after f081654e8d
        # fixed Scheduler._prefetch_kvcache and af5e0c947e the decode side.
        eligible = (
            locally_eligible
            and self.enable_storage
            and not self.cache_controller.prefetch_rate_limited()
        )
        prefetch_key = None
        if eligible:
            prefetch_key = RadixKey(
                new_input_tokens,
                extra_key=last_host_node.key.extra_key,
                is_bigram=self.is_eagle,
            ).page_aligned(self.page_size)
            if len(prefetch_key) < self.prefetch_threshold:
                eligible = False
        if not eligible and not symmetric:
            return 0

        if eligible:
            prefetch_op_cls = (
                HybridPrefetchOperation
                if isinstance(self.cache_controller, HybridCacheController)
                else PrefetchOperation
            )
            extra_kwargs = {}
            if prefetch_op_cls is HybridPrefetchOperation:
                extra_kwargs["pool_transfers"] = self._get_extra_pools().get(
                    "extra_pools"
                )
            operation = prefetch_op_cls(
                "__storage_hit_query__",
                self.cache_controller.mem_pool_host.get_dummy_flat_data_page()[:0],
                prefetch_key,
                last_hash,
                prefix_keys,
                **extra_kwargs,
            )
            hash_values, storage_hit_count = self.cache_controller._storage_hit_query(
                operation
            )
        else:
            # Ineligible under `symmetric`: enter the reduce carrying 0. The op
            # is a MIN, so a single ineligible rank pulls the group answer to 0
            # and NO rank sees an L3 hit -- the local verdict only LOWERS the
            # ballot, it never decides participation.
            storage_hit_count = 0

        storage_hit_count_tensor = torch.tensor(storage_hit_count, dtype=torch.int)
        self._all_reduce_attn_groups(
            storage_hit_count_tensor,
            torch.distributed.ReduceOp.MIN,
            label="query_storage_hit_length",
        )
        storage_hit_count = storage_hit_count_tensor.item()
        storage_hit_count = storage_hit_count - (storage_hit_count % self.page_size)
        return storage_hit_count

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        return self.cache_controller.start_loading()

    def flush_write_through_acks(self) -> None:
        self.writing_check()

    def check_hicache_events(self):
        # Reap the previous round's PP-sync sends before issuing new ones.
        self._drain_async_work()
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
            ],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(
            qsizes,
            torch.distributed.ReduceOp.MIN,
            label="drain_storage_control_queues",
        )

        n_revoke, n_backup, n_release = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )

    # Timeout is linearly increasing with the number of pages
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        cfg = self.prefetch_timeout_config
        num_tokens = len(operation.hash_value) * self.page_size
        timeout = min(cfg.max, cfg.base + cfg.per_ki_token * num_tokens / 1024)
        return time.monotonic() - operation.start_time > timeout

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # unknown prefetch stop policy, just return True
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
            states, torch.distributed.ReduceOp.MAX, label="can_terminate_prefetch"
        )
        can_terminate = states[0].item() == 0
        operation_terminated = states[1].item() == 1
        # the operation should be terminated if it is already terminated on any TP worker
        # or it meets the termination condition on all TP workers
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return True

        # todo: more policies for prefetch progress such as timeout
        # the current policy is to prefetch with best effort and terminate when queuing is over
        last_host_node, prefetch_key, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        if operation.host_indices is None:
            # prefetch has not been issued due to insufficient host memory
            return True

        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        # Synchronize workers before mutating host cache tree state.
        completed_tokens_tensor = torch.tensor(min_completed_tokens, dtype=torch.int)
        self._all_reduce_attn_groups(
            completed_tokens_tensor,
            torch.distributed.ReduceOp.MIN,
            label="check_prefetch_progress",
        )
        min_completed_tokens = completed_tokens_tensor.item()
        fetched_key = prefetch_key[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            last_host_node,
            fetched_key,
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
        )

        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        last_host_node.release_host()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        # Track tokens actually loaded from storage for this request (L3 hits)
        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)

        return True

    def terminate_prefetch(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            return

        _, _, _, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """
        Pop and return the number of tokens loaded from storage for a request.
        Returns 0 if no prefetch was done or was revoked.
        This should be called after check_prefetch_progress() returns True.
        """
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def match_prefix(self, params: MatchPrefixParams):
        if self.disable:
            return self._empty_match_result

        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = self._empty_match_result.device_indices

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            # TODO(ispobock): use best_match_node as start node for load_back
            best_match_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        locally_eligible: bool = True,
    ):
        # #610: MODE first, before any rank-local predicate. See
        # `_hicache_prefetch_symmetric`.
        symmetric = self._hicache_prefetch_symmetric()

        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=last_host_node.key.extra_key,
            is_bigram=self.is_eagle,
        )
        # align the number of fetching tokens to the page size
        prefetch_key = prefetch_key.page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        # RANK-LOCAL predicates. `locally_eligible` is the caller's own gate
        # (`last_host_node.backuped`); `prefetch_rate_limited()` reads the
        # per-rank occupancy counter. Under `symmetric` they may not gate entry
        # into the vote below -- they only lower this rank's ballot.
        eligible = (
            locally_eligible
            and self.enable_storage
            and prefetch_length >= self.prefetch_threshold
            and not self.cache_controller.prefetch_rate_limited()
        )
        if not eligible and not symmetric:
            return

        host_indices = None
        protected = False
        if eligible:
            last_host_node.protect_host()
            protected = True
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
                    if host_indices is None:
                        last_host_node.release_host()
                        return
                else:
                    last_host_node.release_host()
                    # no sufficient host memory for prefetch
                    return
            # NOTE: under `symmetric` the truncation-retry above is deliberately
            # SKIPPED so `len(prefetch_key)` -- hence prefetch_tokens_occupied
            # and the min_completed_tokens reduce in check_prefetch_progress --
            # stays identical across ranks. A failed full alloc becomes a
            # negative vote below instead of a per-rank early return.

        if symmetric:
            # Participation symmetry, the #580 mechanism ported from
            # UnifiedRadixCache (:2243). Registering `ongoing_prefetch` on only
            # a SUBSET of ranks makes `check_prefetch_progress` enter its
            # can_terminate MAX reduce (:1546) and min_completed_tokens MIN
            # reduce (:1580) on a mismatched set of ranks, because its
            # `req_id not in self.ongoing_prefetch` early return (:1555) is
            # rank-local exactly when registration is. MIN over "am I eligible
            # AND did I fully allocate" is a logical AND: every rank registers
            # iff ALL could, otherwise none do, and that early return becomes
            # rank-SYMMETRIC. No async op exists yet, so a negative vote is a
            # clean local release with nothing to tear down.
            vote = torch.tensor(
                [1 if (eligible and host_indices is not None) else 0], dtype=torch.int
            )
            self._all_reduce_attn_groups(
                vote,
                torch.distributed.ReduceOp.MIN,
                label="prefetch_participation_vote",
            )
            if int(vote[0].item()) == 0:
                if host_indices is not None:
                    self.cache_controller.append_host_mem_release(host_indices)
                if protected:
                    last_host_node.release_host()
                return
            # Positive consensus: every rank allocated -> all fall through.
        elif host_indices is None:
            last_host_node.release_host()
            return

        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            prefetch_key,
            last_hash,
            prefix_keys,
            **self._get_extra_pools(),
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _insert_helper_host(
        self, node: TreeNode, key: RadixKey, host_value, hash_value
    ):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = node.key.match(key, page_size=self.page_size)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value.clone()
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            # Publish the newly materialized host suffix immediately so downstream
            # cache indexers can resolve descendants that extend this L2-only prefix.
            self._record_store_event(new_node, medium=StorageMedium.CPU)

        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = key.child_key(self.page_size)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode(priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len].clone()
            child.value = child.value[split_len:].clone()
        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        if child.backuped:
            self._replace_pending_write_through_node(child, [new_node, child])

        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority
        # Hand-off insert: every node on this chain goes to the host tier, the
        # hit-count heuristic does not get a vote. Parent-first order below
        # keeps write_backup's contiguity invariant intact.
        force = params.force_host_write_through

        if priority is None:
            priority = 0

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]

        if len(key) == 0:
            return InsertResult(prefix_len=0)

        node = self.root_node
        child_key = key.child_key(self.page_size)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_host_leaf_status(node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(node.parent)
                else:
                    self._inc_hit_count(node, chunked, force)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(new_node.parent)
                else:
                    self._inc_hit_count(new_node, chunked, force)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            # Compute hash_value if storage or kv events are enabled
            if self.enable_storage or self.enable_kv_cache_events:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            # Emit BlockStored so the router indexes this block.
            self._record_store_event(new_node)

            if force or self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked, force)
        return InsertResult(prefix_len=total_prefix_length)

    def release_aborted_request(self, rid: str):
        # Clean up storage hit tracking for aborted request
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, prefetch_key, host_indices, operation = self.ongoing_prefetch[
            rid
        ]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        self._barrier_attn_groups(label="release_aborted_request")
        last_host_node.release_host()
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)
