from __future__ import annotations

import json
import logging
import os
import threading
import time
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch

from sglang.srt.managers.cache_controller import CacheOperation as BaseCacheOperation
from sglang.srt.managers.cache_controller import consume_gate
from sglang.srt.managers.cache_controller import (
    HiCacheAck,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.managers.cache_controller import (
    LayerDoneCounter,
)
from sglang.srt.managers.cache_controller import (
    StorageOperation as BaseStorageOperation,
)
from sglang.srt.mem_cache.hicache_phase_guard import device_tier_disarmed
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import PoolEntry
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)
device_module = get_device_module()


class CacheOperation(BaseCacheOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, device_indices, node_id, priority)
        self.pool_transfers = pool_transfers

    @staticmethod
    def merge_pool_transfers(
        ops: List[CacheOperation],
    ) -> Optional[list[PoolTransfer]]:
        grouped: dict[tuple[PoolName, Optional[PoolName]], list[PoolTransfer]] = {}
        for op in ops:
            for t in op.pool_transfers or []:
                grouped.setdefault((t.name, t.indices_from_pool), []).append(t)
        if not grouped:
            return None

        def cat_or_none(tensors):
            parts = [x for x in tensors if x is not None]
            return torch.cat(parts) if parts else None

        return [
            PoolTransfer(
                name=ts[0].name,
                host_indices=cat_or_none(t.host_indices for t in ts),
                device_indices=cat_or_none(t.device_indices for t in ts),
                keys=[k for t in ts if t.keys for k in t.keys] or None,
                hit_policy=ts[0].hit_policy,
                indices_from_pool=ts[0].indices_from_pool,
            )
            for ts in grouped.values()
        ]

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        if len(ops) == 1:
            return ops[0]
        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged = CacheOperation(
            host_indices,
            device_indices,
            -1,
            priority,
            pool_transfers=CacheOperation.merge_pool_transfers(ops),
        )
        merged.node_ids = node_ids
        return merged


class StorageOperation(BaseStorageOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys)
        self.pool_transfers = pool_transfers
        self.pool_storage_result = PoolTransferResult.empty()


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        self.request_id = request_id
        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()
        super().__init__(
            host_indices,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=pool_transfers,
        )
        self.pool_transfers_done = not bool(pool_transfers)

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HybridCacheController(BaseHiCacheController):
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: Any,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        transfer_layer_num: Optional[int] = None,
        enable_storage_metrics: bool = False,
    ):
        startup_storage_backend = storage_backend
        self.extra_host_mem_release_queues: dict[PoolName, Queue[torch.Tensor]] = {}
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=pp_group,
            write_policy=write_policy,
            io_backend=io_backend,
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        # Override layer_num: hybrid models transfer all layers (For example, Linear Model (KV + Mamba)),
        # not just the full attention layers reported by full_kv_pool.
        if transfer_layer_num is not None and transfer_layer_num != self.layer_num:
            self.layer_num = transfer_layer_num
            self.layer_done_counter = LayerDoneCounter(self.layer_num)

        if startup_storage_backend is not None:
            self.attach_storage_backend(
                storage_backend=startup_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=model_name,
                storage_backend_extra_config=storage_backend_extra_config,
                host_pools=getattr(mem_pool_host, "entries", None),
            )

    def _start_storage_threads(self):
        super()._start_storage_threads()
        self._init_extra_host_mem_release_queues()

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        host_pools: Optional[list[PoolEntry]] = None,
    ):
        super().attach_storage_backend(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
        )

        for entry in host_pools or []:
            self.storage_backend.register_mem_host_pool_v2(entry.host_pool, entry.name)

    @staticmethod
    def parse_storage_backend_extra_config(
        storage_backend_extra_config: Optional[str],
    ) -> tuple[dict, int, float, float, bool]:
        extra_config = {}
        if storage_backend_extra_config:
            if storage_backend_extra_config.startswith("@"):
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
                extra_config = json.loads(storage_backend_extra_config)

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )
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
                "prefetch_timeout_per_ki_token must be number, got "
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def clear_storage_backend(self) -> bool:
        if not self.enable_storage:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False
        if not hasattr(self.storage_backend, "clear"):
            logger.warning(
                "Storage backend %s does not support clear operation.",
                type(self.storage_backend).__name__,
            )
            return False
        self.storage_backend.clear()
        return True

    def _init_extra_host_mem_release_queues(self) -> None:
        self.extra_host_mem_release_queues = {}
        # #718/#847: the OWNING entry, captured next to its queue.
        # `extra_host_mem_release_queues` outlives a phase rebind (it is built
        # once, at storage-thread start), but the lookup that used to resolve a
        # release went through the CURRENTLY bound `entry_map` -- so after a
        # rebind onto a narrower tier every extra-pool release fell through an
        # `entry is None -> continue` and those host slots were neither queued
        # nor freed. Silent leak, one per load-back, for the whole phase.
        # Keeping the entry beside the queue makes the pair inseparable, which
        # is what the anchor path achieves with its generation lookup.
        self.extra_host_mem_release_entries = {}
        entries = getattr(self.mem_pool_host, "entries", None) or []
        anchor_entry = getattr(self.mem_pool_host, "anchor_entry", None)
        for entry in entries:
            if entry is anchor_entry or entry.is_primary_index_anchor:
                continue
            self.extra_host_mem_release_queues[entry.name] = Queue()
            self.extra_host_mem_release_entries[entry.name] = entry

    def entry_for_extra_release(self, pool_name):
        """The entry that OWNS an extra pool's host slots, rebind or not.

        Prefers the currently bound tier and falls back to the entry captured
        when the release queue was created. Returns None only when neither
        knows the pool, which is the caller's cue to say so rather than to drop
        the slots.
        """
        entry = self.mem_pool_host.entry_map.get(pool_name)
        if entry is not None:
            return entry
        return getattr(self, "extra_host_mem_release_entries", {}).get(pool_name)

    def _append_host_mem_release_pages(
        self, release_queue: Queue, host_indices: torch.Tensor, page_size: int
    ) -> None:
        if host_indices.numel() == 0:
            return
        for page in host_indices.split(page_size):
            release_queue.put(page)

    def append_host_mem_release(
        self,
        host_indices: Optional[torch.Tensor] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
        generation=None,
    ):
        """W35: the override must route stale releases too, or the fix is inert.

        THIS SIGNATURE SHADOWS THE BASE ONE, and this is the live path on the
        mamba/hybrid rig. The base `append_host_mem_release` gained
        produce-time routing so a release opened under an older binding is
        freed against the pool it came from; an override that silently dropped
        the `generation` argument would leave that fix installed and
        unreachable on the only lane that runs -- the same shape that cost W31
        (below the drain gate), W32 (a second copy) and W33 (unreachable
        writer). Caught at the desk this time rather than on metal.
        """
        if host_indices is not None and generation is not None:
            from sglang.srt.mem_cache.hicache_phase_binding import (
                host_pool_for_generation,
                write_back_stamp_is_current,
            )

            if not write_back_stamp_is_current(generation):
                owner = host_pool_for_generation(generation)
                self._stale_release_routed = (
                    getattr(self, "_stale_release_routed", 0) + 1
                )
                if owner is None:
                    self._stale_release_orphaned = (
                        getattr(self, "_stale_release_orphaned", 0) + 1
                    )
                    logger.error(
                        "#719/W35 STALE RELEASE ORPHANED (hybrid): %d host "
                        "slot(s) from binding generation %s name no known "
                        "pool; neither queued nor freed. (%d so far.)",
                        int(host_indices.numel()),
                        generation,
                        self._stale_release_orphaned,
                    )
                else:
                    n = self._stale_release_routed
                    if n <= 3 or n % 200 == 0:
                        logger.warning(
                            "#719/W35 STALE RELEASE ROUTED (hybrid): %d host "
                            "slot(s) from binding generation %s freed against "
                            "THAT generation's pool. (%d so far.)",
                            int(host_indices.numel()),
                            generation,
                            n,
                        )
                    owner.free(host_indices)
                host_indices = None
        if host_indices is not None:
            self._append_host_mem_release_pages(
                self.host_mem_release_queue,
                host_indices,
                self.mem_pool_host.page_size,
            )
        for transfer in extra_pools or []:
            if transfer.host_indices is None or transfer.host_indices.numel() == 0:
                continue
            if transfer.indices_from_pool is not None:
                # A derived transfer borrows another pool's indices; the owner
                # releases them, and releasing them twice is the real bug.
                continue
            entry = self.entry_for_extra_release(transfer.name)
            if entry is None:
                n = getattr(self, "_orphaned_extra_release", 0) + 1
                self._orphaned_extra_release = n
                # Rate-limited on the same cadence as the stale-release routing
                # above: this fires per release, so an unbounded emitter here is
                # the log-flood class. The COUNT is the finding, not the line.
                if n <= 3 or n % 200 == 0:
                    logger.error(
                        "#718/#847 EXTRA RELEASE ORPHANED: %d host slot(s) of "
                        "pool '%s' name no known entry in the bound tier (%s) "
                        "nor in the queue registry; neither queued nor freed. "
                        "(%d so far.)",
                        int(transfer.host_indices.numel()),
                        transfer.name,
                        sorted(str(name) for name in self.mem_pool_host.entry_map),
                        n,
                    )
                continue
            if entry.is_primary_index_anchor:
                continue
            release_queue = self.extra_host_mem_release_queues.get(transfer.name)
            if release_queue is None:
                continue
            self._append_host_mem_release_pages(
                release_queue, transfer.host_indices, entry.host_pool.page_size
            )

    def reset(self):
        super().reset()
        if self.enable_storage:
            self.host_mem_release_queue.queue.clear()
            for release_queue in self.extra_host_mem_release_queues.values():
                release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        # #760: THE OVERRIDE IS THE HOLE THE CRASH WENT THROUGH. The base
        # class asks device_tier_disarmed at enqueue; this override did not,
        # so the guard's every metal reading on a Unified/hybrid deployment
        # was vacuously zero while TP-phase inserts (cache_finished_req ->
        # insert -> _inc_hit_count -> write_backup -> here) enqueued copies
        # against the PP-bound pools -- the exact stack of the 2026-08-19
        # 20:40 specimen, SIGSEGV in transfer_kv_direct seconds after a
        # pp_to_tp cutover, with zero refusal lines. Same contract as the
        # base: a refused write is a prefix that misses later, which is the
        # cheap failure.
        if device_tier_disarmed("write"):
            return None
        # #923: THE SAME OMISSION AS #760's, one question further on. The base
        # class now also asks whether the row this copy would READ is a row of
        # this rank's device pool; this override must ask it too, because on a
        # hybrid deployment this override IS the write path. The specimen is a
        # global allocator slot reaching the TP phase's compact pool after a
        # cutover: below the row count it copies another token's KV under this
        # request's key, above it the kernel's slice clamps to empty and the
        # copy dies with a bare tensor-size RuntimeError.
        if self._refuse_unaddressable_kv_rows(device_indices, "write"):
            return None
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=True,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            self.mem_pool_host.free(host_indices)
            return None

        self.write_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if not self.write_queue:
            return
        # #760/W35: THE CONSUME HALF, WHICH THIS SUBCLASS NEVER HAD.
        # `write()` above carries the ENQUEUE-time checks; the base class also
        # re-asks them at consume, and this override did not. That made the
        # mamba/hybrid path -- the live path on this rig -- the one lane where
        # a write-back queued before a cutover is consumed after it, which is
        # the shape this file's own #760 note describes.
        if not consume_gate(self, "write_queue", "write"):
            return
        op = CacheOperation.merge_ops(self.write_queue)
        # Page-first write-back JIT kernels can keep destination host indices on CPU.
        if (
            self.io_backend == "kernel"
            and self.mem_pool_host.layout == "page_first"
            and getattr(self.mem_pool_host, "can_use_write_back_jit", False)
        ):
            host_indices = op.host_indices
            device_indices = op.device_indices
            resolved_pool_transfers = op.pool_transfers
        else:
            host_indices, device_indices, resolved_pool_transfers = (
                self.move_hybrid_indices(op)
            )
        self.write_queue.clear()
        # Weighted uneven-DCP: back up only this rank's owned tokens, from
        # their COMPACT device slots (identity when the gate is off). Only the
        # anchor KV transfer is token-sharded; pool_transfers (mamba/SWA) use
        # their own per-rank indices and the draft pool holds the full token
        # context with global indices -- both keep the raw pair list.
        kv_host_indices, kv_device_indices = self._dcp_kv_transfer_pairs(
            host_indices, device_indices
        )
        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                kv_host_indices,
                kv_device_indices,
                self.io_backend,
                pool_transfers=resolved_pool_transfers,
            )
            if self.draft_tier_armed("write") and host_indices.numel() > 0:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            self._record_transfer_indices_on_stream(
                self.write_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
            self._record_transfer_indices_on_stream(
                self.write_stream, kv_host_indices, kv_device_indices
            )
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        # #760: mirror of write() above -- the base class refuses a load while
        # the active phase is not the one the bindings belong to (it would
        # fill rows the model does not read while the tree marks the prefix
        # resident); this override skipped that question entirely. A refused
        # load is a prefetch that does not land: a miss now, recomputed.
        if device_tier_disarmed("load"):
            return None
        need_load_kv = host_indices.numel() > 0

        full_allocator = getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )
        if not need_load_kv:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            device_indices = full_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None
            # #923: sibling of the write refusal. Asked before any extra pool
            # allocation, and the KV allocation is handed back so a refusal
            # costs nothing but the prefetch.
            if self._refuse_unaddressable_kv_rows(device_indices, "load"):
                full_allocator.free(device_indices)
                return None

        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=False,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            if need_load_kv:
                full_allocator.free(device_indices)
            return None

        self.load_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        return device_indices

    def start_loading(self) -> int:
        if not self.load_queue:
            return -1
        # #760/W35: same consume half, load side. A stale load fills device
        # rows from host slots this phase does not own and the tree then marks
        # the prefix RESIDENT, so attention reads KV nobody wrote -- a silent
        # wrong answer, with no assertion anywhere to catch it. Checked before
        # a producer is allocated.
        if not consume_gate(self, "load_queue", "load"):
            return -1
        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices, resolved_pool_transfers = (
            self.move_hybrid_indices(op)
        )
        self.load_queue.clear()
        # Weighted uneven-DCP: load only this rank's owned tokens into their
        # COMPACT device slots (identity when the gate is off). pool_transfers
        # and the draft pool keep the raw pair list (see start_writing).
        kv_host_indices, kv_device_indices = self._dcp_kv_transfer_pairs(
            host_indices, device_indices
        )
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()
        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    kv_host_indices,
                    kv_device_indices,
                    i,
                    self.io_backend,
                    pool_transfers=resolved_pool_transfers,
                )
                if (
                    self.draft_tier_armed("load")
                    and host_indices.numel() > 0
                    and i < self.mem_pool_host_draft.layer_num
                ):
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                producer_event.complete(i)
            self._record_transfer_indices_on_stream(
                self.load_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
            self._record_transfer_indices_on_stream(
                self.load_stream, kv_host_indices, kv_device_indices
            )
        self.ack_load_queue.append(
            HiCacheAck(
                producer_event.start_event,
                producer_event.finish_event,
                op.node_ids,
            )
        )
        return producer_id

    def _record_transfer_indices_on_stream(
        self,
        stream: torch.Stream,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> None:
        if host_indices.is_cuda:
            host_indices.record_stream(stream)
        if device_indices.is_cuda:
            device_indices.record_stream(stream)
        for transfer in pool_transfers or []:
            if transfer.host_indices is not None and transfer.host_indices.is_cuda:
                transfer.host_indices.record_stream(stream)
            if transfer.device_indices is not None and transfer.device_indices.is_cuda:
                transfer.device_indices.record_stream(stream)

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> PrefetchOperation:
        operation = PrefetchOperation(
            request_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.prefetch_queue.put(operation)
        return operation

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
        kv_page_owner_mask: Optional[torch.Tensor] = None,
    ) -> int:
        operation = StorageOperation(
            host_indices,
            token_ids,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        operation.kv_page_owner_mask = kv_page_owner_mask
        self.backup_queue.put(operation)
        return operation.id

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        hash_value = self.get_hash_str(
            operation.token_ids, operation.last_hash, page_size=self.page_size
        )

        extra_info = HiCacheStorageExtraInfo(
            prefix_keys=operation.prefix_keys.copy() if operation.prefix_keys else None
        )
        if operation.pool_transfers:
            self._hitq_v2_n = getattr(self, "_hitq_v2_n", 0) + 1
            _arm = "v2"
            hit_result = self.storage_backend.batch_exists_v2(
                hash_value, operation.pool_transfers, extra_info
            )
        elif getattr(self, "extra_host_mem_release_entries", None):
            # #1035 -- THE SILENT DEGRADATION, NAMED AND REFUSED.
            #
            # The `else` below is upstream's path for a controller that has NO
            # non-KV pools at all: nothing to cap with, so an uncapped KV answer
            # is the correct answer. On THIS controller non-KV pools ARE
            # registered (`extra_host_mem_release_entries` is populated in
            # __init__ from `mem_pool_host.entries`, one per non-anchor pool),
            # so reaching it means a component DECLINED to build its transfer
            # list -- e.g. `MambaComponent.build_hicache_transfers` returning an
            # empty list on an exhausted host anchor pool.
            #
            # Answering that request through `batch_exists` returns the KV span
            # WITHOUT the component boundary that makes it usable, and the node
            # then published carries KV and no anchor. The conjunctive match walk
            # (`unified_radix_cache.py`, mamba validator) rejects such a node for
            # good, so the uncapped answer is not a generous answer -- it is an
            # unmatchable one, bought at full host-memory price. Both counters
            # ("success") stay green while read-through stays dead: the
            # `instrument-text-luegt` shape.
            #
            # A refusal is the honest answer: report 0 pages, exactly as a real
            # miss does, and SAY SO with a counter so a zero here is readable as
            # "never happened" rather than as "not instrumented". Nothing is
            # published, nothing leaks, and the cost shows up as a named number
            # instead of as a mysteriously dead read path.
            self._hitq_v1_refused_n = getattr(self, "_hitq_v1_refused_n", 0) + 1
            if self._hitq_v1_refused_n <= 40 or self._hitq_v1_refused_n % 256 == 0:
                logger.warning(
                    "#1035 UNCAPPED-PUBLISH REFUSED n=%d: %d non-KV pool(s) are "
                    "registered (%s) but this prefetch carries NO pool transfers "
                    "-- a component could not build its list (exhausted host "
                    "anchor pool is the known cause). Answering uncapped would "
                    "publish a KV-only node the match walk can never match. "
                    "Refusing with 0 pages. v2_queries=%d",
                    self._hitq_v1_refused_n,
                    len(self.extra_host_mem_release_entries),
                    ",".join(str(n) for n in self.extra_host_mem_release_entries),
                    getattr(self, "_hitq_v2_n", 0),
                )
            _arm = "refused-uncapped-publish"
            hit_result = PoolTransferResult(
                kv_hit_pages=0,
                extra_pool_hit_pages={},
                keys_asked=len(hash_value),
            )
        else:
            _arm = "v1-uncapped"
            kv_hit_count = self.storage_backend.batch_exists(hash_value, extra_info)
            hit_result = PoolTransferResult(
                kv_hit_pages=kv_hit_count,
                extra_pool_hit_pages={},
                keys_asked=len(hash_value),
                kv_uncapped=kv_hit_count,
            )

        kv_hit_pages = hit_result.kv_hit_pages
        operation.pool_storage_result.update_kv_hit_pages(kv_hit_pages)

        # #1035c: RESOLVE "ANSWERED ZERO" INTO ITS THREE CAUSES.
        #
        # WHAT WAS INVISIBLE. `len(hash_value)` is the number of keys this probe
        # was GIVEN, it is computed on the line above, and it died at the return
        # -- so `([], 0)` from an empty key set and `([], 0)` after asking about
        # 8564 keys were byte-identical to every reader. Downstream that zero
        # reaches `completed_local`, whose own value at that point is its
        # `__init__` default, so a zero there is an INITIALISATION and not a
        # measurement. Three different states, one indistinguishable output.
        #
        # THE PARTITION, and each term names the field that decides it:
        #   NEVER-ASKED  keys_asked == 0     -- nothing was put to the store.
        #   ASKED-AND-NO kv_uncapped == 0    -- the store holds no leading page.
        #   CAPPED       kv_uncapped > 0     -- pages EXIST and a component pool
        #                                       cut the claim; `by=` names it.
        # `by=` reads `zero_capped_pools`, NOT `extra_pool_hit_pages`: that dict
        # records only non-zero boundaries, so the pool that capped to exactly 0
        # is ABSENT from it and its emptiness reads as "uncapped" while meaning
        # "capped to nothing". That misreading is the reason this line exists.
        #
        # THIS IS NOT A SECOND COUNTER. The prohibition at
        # `phase_flip_runtime.py:8807-8810` governs a quantity measured TWICE;
        # this one is measured ZERO times -- the terms are already computed and
        # then discarded, and nothing below changes a decision.
        #
        # Emitted only on the zero answer, which is the ambiguous case; a
        # non-zero claim is already attributable from `#1028B`. Rate-limited
        # WITH its suppressed count, because a bounded emitter that hides its
        # own suppression is how a zero becomes unreadable a second time.
        #
        # NO SECOND CLASSIFIER. `by=` says WHICH pool capped the claim to zero
        # and stops there. Whether that pool's shortfall is a SPAN problem (an
        # anchor exists but not in range) or a DENSITY problem (no anchor at
        # all) is already answered by `#1035b anchors_in_range(count,
        # deepest_idx)` on the `#1028B` line -- `(0, -1)` is "none in range",
        # `(1, 4095)` is "one, and here it ends". Read that field for the
        # cause; this line only partitions the zero.
        if kv_hit_pages == 0:
            self._1035c_n = getattr(self, "_1035c_n", 0) + 1
            n = self._1035c_n
            if n <= 40 or n % 256 == 0:
                asked = hit_result.keys_asked
                uncapped = hit_result.kv_uncapped
                if asked == 0:
                    cause = "NEVER-ASKED (no keys were put to the store)"
                elif uncapped == 0:
                    cause = "ASKED-AND-NO (store holds no leading page)"
                else:
                    cause = (
                        f"CAPPED (store holds {uncapped} leading KV page(s); a "
                        f"component pool cut the claim to 0)"
                    )
                logger.warning(
                    "#1035c ZERO-ANSWER PARTITION n=%d arm=%s cause=%s "
                    "asked=%d kv_uncapped=%d claimed=%d by=%s "
                    "(suppressed_so_far=%d). `by` lists pools capped to EXACTLY "
                    "0 -- they are absent from extra_pool_hit_pages by "
                    "construction, so an empty caps dict there means 'capped to "
                    "nothing', never 'uncapped'.",
                    n,
                    _arm,
                    cause,
                    asked,
                    uncapped,
                    kv_hit_pages,
                    ",".join(hit_result.zero_capped_pools) or "-",
                    max(0, n - min(n, 40) - (n // 256)),
                )

        return (
            hash_value[:kv_hit_pages],
            kv_hit_pages * self.page_size,
        )

    def move_hybrid_indices(
        self, operation: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        host_indices, device_indices = self.move_indices(
            operation.host_indices, operation.device_indices
        )
        resolved_pool_transfers = None
        if operation.pool_transfers:
            resolved_pool_transfers = []
            for transfer in operation.pool_transfers:
                transfer_host_indices, transfer_device_indices = self.move_indices(
                    transfer.host_indices, transfer.device_indices
                )
                # Keep the original PoolTransfer unchanged because tree-owned
                # transfers may still reference radix-tree host state. The
                # controller only needs a normalized execution-time copy.
                resolved_pool_transfers.append(
                    PoolTransfer(
                        name=transfer.name,
                        host_indices=transfer_host_indices,
                        device_indices=transfer_device_indices,
                        keys=transfer.keys,
                        hit_policy=transfer.hit_policy,
                        indices_from_pool=transfer.indices_from_pool,
                    )
                )
        return host_indices, device_indices, resolved_pool_transfers

    def _page_transfer(self, operation):
        # KV pools first — determines actual completed page count
        super()._page_transfer(operation)

        # Extra pools only after KV fully completes. If KV terminated early
        # (IO failure, timeout, TP mismatch), skip extra IO entirely to avoid
        # data misalignment.
        kv_completed_pages = operation.completed_tokens // self.page_size
        if operation.pool_transfers and kv_completed_pages == len(operation.hash_value):
            self._sync_trailing_keys(
                operation.pool_transfers, operation.hash_value, kv_completed_pages
            )
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_get_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)
        operation.pool_transfers_done = True

    def _page_backup(self, operation):
        # Backup extra pools
        if operation.pool_transfers:
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_set_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)

        # Backup kv pools
        super()._page_backup(operation)

    def _resolve_sidecar_derived_pool_transfers(self, operation):
        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool is None:
                continue
            if transfer.indices_from_pool != PoolName.KV:
                source = next(
                    (
                        t
                        for t in operation.pool_transfers
                        if t.indices_from_pool is None
                        and t.name == transfer.indices_from_pool
                    ),
                    None,
                )
                if source is None:
                    raise AssertionError(
                        "Storage sidecar derived pool source missing: "
                        f"{transfer.name} from {transfer.indices_from_pool}."
                    )
                transfer.host_indices = source.host_indices
                if transfer.keys is None:
                    transfer.keys = source.keys
            else:
                transfer.host_indices = operation.host_indices
                if transfer.keys is None:
                    transfer.keys = operation.hash_value

    def _sync_trailing_keys(
        self,
        pool_transfers: list[PoolTransfer],
        all_hashes: list[str],
        kv_hit_pages: int,
    ) -> None:
        """Re-align trailing-page sidecar keys after KV hit truncation.

        When the storage hit is shorter than the original target prefix, each
        pool transfer's keys must be updated to the last N hashes of the actual
        hit range instead of the last N hashes of the original target range.
        For mamba (N=1) this is just the last hit page hash; for SWA (N>1) it
        is a sliding window of the last N hit pages.
        """
        for transfer in pool_transfers:
            if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES:
                continue
            trailing_n = len(transfer.keys) if transfer.keys else 1
            transfer.keys = all_hashes[max(0, kv_hit_pages - trailing_n) : kv_hit_pages]

    def _resolve_pool_transfers_allocation(
        self,
        extra_pools: Optional[list[PoolTransfer]],
        alloc_host: bool,
        kv_device_indices: Optional[torch.Tensor] = None,
        kv_host_indices: Optional[torch.Tensor] = None,
    ) -> Optional[list[PoolTransfer]]:
        """Auto-alloc host or device indices for PoolTransfers where they are None."""
        if not extra_pools:
            return None
        # (pool, free_fn, indices) for atomic rollback on failure.
        newly_allocated: list[tuple[PoolTransfer, Callable, torch.Tensor]] = []
        derived_transfers: list[PoolTransfer] = []

        def rollback_allocated() -> None:
            for prev_pool, prev_free_fn, prev_indices in newly_allocated:
                prev_free_fn(prev_indices)
                if alloc_host:
                    prev_pool.host_indices = None
                else:
                    prev_pool.device_indices = None

        for pool in extra_pools:
            if pool.indices_from_pool is not None:
                derived_transfers.append(pool)
                continue
            entry = self.mem_pool_host.entry_map.get(pool.name)
            if entry is None:
                # #718/#847: THE BOUND TIER CANNOT DESCRIBE THIS POOL. Both
                # other answers are wrong. Returning the transfer unresolved
                # hands a None index set to `move_indices` (the W38-B crash at
                # cache_controller.py:1217, and its write-side twin one line
                # below); dropping it silently moves the KV while this pool's
                # state stays behind and the tree calls the prefix RESIDENT --
                # a wrong ANSWER, which is worse than a crash. Refusing costs
                # one recompute, which is merely slow.
                #
                # This is reachable because a phase rebind REPLACES
                # `mem_pool_host` (hicache_phase_binding._stamp), and the TP
                # tier built at phase_flip_boot.py:2019-2032 carries KV alone.
                # `check_pool_coverage` refuses that rebind now; this stays as
                # the class guard for every other way a tier can be narrower
                # than the transfers built against it.
                self._refuse_unresolvable_transfer(pool)
                rollback_allocated()
                return None
            if alloc_host:
                if pool.host_indices is not None:
                    continue
                if pool.device_indices is None:
                    self._refuse_unresolvable_transfer(pool)
                    rollback_allocated()
                    return None
                alloc_fn = entry.host_pool.alloc
                free_fn = entry.host_pool.free
                evict_fn = entry.host_evict_fn
                size = len(pool.device_indices)
            else:
                if pool.device_indices is not None:
                    continue
                if pool.host_indices is None:
                    self._refuse_unresolvable_transfer(pool)
                    rollback_allocated()
                    return None
                # device_alloc_fn / device_free_fn override entry.device_pool's
                # methods for pools whose device_pool is a raw KV pool (layout)
                # rather than an allocator (e.g. SWA).
                alloc_fn = entry.device_alloc_fn or entry.device_pool.alloc
                free_fn = entry.device_free_fn or entry.device_pool.free
                evict_fn = entry.device_evict_fn
                size = len(pool.host_indices)
            indices = alloc_fn(size)
            if indices is None and evict_fn:
                evict_fn(size)
                indices = alloc_fn(size)
            if indices is None:
                # Atomic rollback: free everything we successfully allocated.
                rollback_allocated()
                return None
            if alloc_host:
                pool.host_indices = indices
            else:
                pool.device_indices = indices
            newly_allocated.append((pool, free_fn, indices))

        # Assign indices to deferred pools from their source.
        for pool in derived_transfers:
            if pool.indices_from_pool == PoolName.KV:
                pool.host_indices = kv_host_indices
                pool.device_indices = kv_device_indices
                continue

            source = next(
                (
                    transfer
                    for transfer in extra_pools
                    if transfer.indices_from_pool is None
                    and transfer.name == pool.indices_from_pool
                ),
                None,
            )
            if source is None:
                rollback_allocated()
                return None
            pool.host_indices = source.host_indices
            pool.device_indices = source.device_indices

        # THE POST-CONDITION IS THE POINT OF THIS FUNCTION, so it is stated
        # here rather than discovered at cache_controller.py:1217 on boot
        # second 20. Contract: a returned list contains no unresolved index
        # set. Anything else is refused, and the callers (write():439,
        # load():554) already know how to give back what they took.
        for pool in extra_pools:
            if pool.host_indices is None or pool.device_indices is None:
                self._refuse_unresolvable_transfer(pool)
                rollback_allocated()
                return None
        return extra_pools

    def _refuse_unresolvable_transfer(self, pool: PoolTransfer) -> None:
        """Log one refusal, rate-limited.

        `match_prefix` re-derives the host hit every scheduler tick, so a
        refused load-back is RETRIED every tick for as long as the state stays
        on the host. The request still makes progress (it re-prefills the
        segment); what must not happen is one line per tick -- that is the
        449 MB/20 min flood class. First three, then every 200th, which is the
        cadence the stale-release routing above already uses.
        """
        n = getattr(self, "_unresolvable_transfer_refusals", 0) + 1
        self._unresolvable_transfer_refusals = n
        if n <= 3 or n % 200 == 0:
            logger.error(
                "#718/#847 POOL-SET MISMATCH: cannot resolve the '%s' transfer "
                "against the bound host tier (it describes %s; host=%s "
                "device=%s). Refusing the whole transfer set: moving the KV "
                "while this pool's state stays behind would leave the tree "
                "reporting a prefix as resident that is not, which is a wrong "
                "answer. These tokens are recomputed. (%d refusal(s) so far.)",
                pool.name,
                sorted(str(name) for name in self.mem_pool_host.entry_map),
                "set" if pool.host_indices is not None else "None",
                "set" if pool.device_indices is not None else "None",
                n,
            )
