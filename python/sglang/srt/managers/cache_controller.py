from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
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

import logging
import threading
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.hicache_phase_guard import device_tier_disarmed
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
    compute_model_identity_hash,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.pool_host import HostKVCache

from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)

device_module = get_device_module()


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[self.producer_index].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1


class CacheOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        # #760: EVERY op carries the binding it was built against, stamped at
        # construction rather than at one enqueue site -- an op created by any
        # other path would otherwise be unstamped, and an unstamped op must be
        # refused, which would silently drop legitimate write-backs.
        from sglang.srt.mem_cache.hicache_phase_binding import current_generation

        self.binding_generation = current_generation()
        self.data = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]

        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(host_indices, device_indices, -1, priority)
        merged_op.node_ids = node_ids
        return merged_op

    def __lt__(self, other: CacheOperation):
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]


class StorageOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys
        # W35: the binding this operation was OPENED under. Stamped here, at
        # construction, because that is when its host slots were allocated --
        # so every consumer downstream (the backup/prefetch threads, the
        # revoke drain, the release queue) can ask which pool the slots came
        # from instead of assuming the currently-bound one. One authority: the
        # value comes from #719's generation and nowhere else.
        try:
            from sglang.srt.mem_cache.hicache_phase_binding import current_generation

            self.binding_generation = current_generation()
        except Exception:  # noqa: BLE001 - a stamp may never break an op
            self.binding_generation = None

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    def __lt__(self, other: StorageOperation):
        return self.id < other.id


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.request_id = request_id

        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()

        super().__init__(host_indices, token_ids, last_hash, prefix_keys=prefix_keys)

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


def canonical_identity_hash_for(server_args, canonical_page: bool) -> str:
    """#706 remainder: the identity hash for a key that must be geometry-FREE.

    Under ``--phase-flip-canonical-kv-page`` the stored page holds EVERY
    attention layer at full width, so no parallel split can change its bytes.
    The key already drops the tp and pp suffixes for exactly that reason. The
    identity hash was not given the same treatment: its
    ``include_parallel_vectors`` defaults to True, so ``rank_tp_ratio`` and
    ``rank_kv_ratio`` -- which say how the KV is SPLIT across ranks, not what a
    canonical page contains -- were still moving the key.

    Live, not hypothetical: the harvest boot ran ``rank_tp_ratio=None``
    (falsy, skipped) but ``rank_kv_ratio='coupled'`` (truthy, appended), so a
    geometry term was in the key today. Two boots of the same model and
    kv-dtype with different kv-ratios write byte-identical canonical pages and
    miss each other -- which is precisely the cross-boot retention the whole
    format exists to provide.

    WHAT IS NOT DROPPED, and must never be: revision, dtype, quantization and
    kv_cache_dtype. Those describe the BYTES, and confusing two byte formats is
    the silent wrong hit the hash was introduced to turn into a clean miss.
    Only the parallel tail goes, and only when the canonical format is on --
    without it a stage's file really does hold just that stage's layers and
    geometry belongs in the key.
    """
    return compute_model_identity_hash(
        server_args, include_parallel_vectors=not canonical_page
    )


def consume_gate(controller, queue_attr: str, direction: str) -> bool:
    """May the batch queued in ``queue_attr`` be CONSUMED now? (#760/#719)

    ONE AUTHORITY, FOUR CALLERS. The write path had this logic inline and the
    load path had none; the hybrid subclass had neither. That is four sites
    that must agree about one question, which is exactly the shape that cost
    W32 (a second copy of a rule silently overriding the first) -- so the
    question is asked in one place and the four consume points call it.

    THE TWO CHECKS ARE NOT REDUNDANT, and #760 records why both are needed:

    * ``device_tier_disarmed(direction)`` -- the phase predicate. ``write()``
      and ``load()`` ask it at ENQUEUE and get the right answer there: the copy
      is queued while the model computes in the phase these pools are bound to.
      Nothing re-asked afterwards, and the cutover lands in between. Both #760
      crash specimens died three seconds AFTER a pp_to_tp cutover completed.
    * the binding generation stamp -- which cannot cover it alone: with
      ``--phase-flip-rebind-hicache`` off the binding never advances, so every
      stamp matches by construction and the check is dead code.

    WHY THE LOAD SIDE MATTERS AT LEAST AS MUCH AS THE WRITE SIDE, and why its
    absence was worse: a stale WRITE corrupts the host copy and is caught by
    the pool's own double-free/ownership assertions or simply persists wrong
    bytes. A stale LOAD fills device rows from host slots the incoming phase
    does not own and the tree then marks that prefix RESIDENT -- attention
    reads KV nobody wrote, with no assertion anywhere. A silent wrong answer,
    which this codebase ranks worse than a crash.

    ``check_shapes`` cannot substitute for either: under ``layer_first`` a
    stale binding is shape-IDENTICAL to the live one, which is why #760 records
    the shape guard armed on three ranks, refusing zero, and a SIGSEGV anyway.

    Refusing costs a cache MISS later -- the same cheap failure the #718 disarm
    already accepts. Proceeding costs the scheduler.

    Returns True to proceed. Returns False after CLEARING the queue: the batch
    is refused, loudly and counted by name.
    """
    from sglang.srt.mem_cache.hicache_phase_binding import (
        write_back_stamp_is_current,
    )

    queue = getattr(controller, queue_attr, None)
    if not queue:
        return False

    label = direction.upper()
    if device_tier_disarmed(direction):
        attr = f"_{direction}_phase_refusals"
        setattr(controller, attr, getattr(controller, attr, 0) + len(queue))
        n = getattr(controller, attr)
        if n <= 3 or n % 200 == 0:
            logger.warning(
                "#760 %s REFUSED AT CONSUME: the phase moved after these %d "
                "operation(s) were queued, so their device indices name the "
                "pool of a phase that is no longer computing. Dropping them; "
                "those prefixes miss later. (%d so far.)",
                label,
                len(queue),
                n,
            )
        queue.clear()
        return False

    fresh = [
        o
        for o in queue
        if write_back_stamp_is_current(getattr(o, "binding_generation", None))
    ]
    if len(fresh) != len(queue):
        dropped = len(queue) - len(fresh)
        attr = f"_{direction}_stamp_refusals"
        setattr(controller, attr, getattr(controller, attr, 0) + dropped)
        n = getattr(controller, attr)
        if n <= 3 or n % 200 == 0:
            logger.warning(
                "#760 %s REFUSED: %d queued operation(s) were stamped against "
                "an older binding generation and are dropped rather than "
                "consumed after the rebind. Those prefixes miss later. "
                "(%d so far.)",
                label,
                dropped,
                n,
            )
        setattr(controller, queue_attr, fresh)
    return bool(getattr(controller, queue_attr))


def operation_is_stale(controller, operation, kind: str) -> bool:
    """Was ``operation`` opened under a binding that has since moved? (#719/W35)

    THE SIBLING OF ``consume_gate``, and deliberately in the same module: that
    one answers the question for a QUEUED BATCH at a consume point, this one
    for a SINGLE operation on a background thread. One authority, two shapes --
    the alternative is a third copy of the rule, which is what cost W32.

    REFUSAL IS THE ONLY SAFE VERB HERE, unlike the release path. A stale
    RELEASE can be routed to the pool its generation names, because that pool
    still owns those slots. A stale BACKUP cannot: it would read
    ``mem_pool_host.get_data_page(...)`` for host slots whose pool may since
    have been repurposed, and persist those bytes to a CONTENT-ADDRESSED store
    under a hash computed from the tokens it was opened with. The hash would
    not match the payload, every later reader would trust it, and the
    corruption OUTLIVES THE PROCESS. There is no version of routing that makes
    that safe, so the operation is declined and nothing is written.

    A DECLINED BACKUP IS A CORRECT NON-PERSIST, not a loss: the prefix simply
    misses later and is recomputed -- the same cheap failure the #718 disarm
    and the #760 write refusal already accept.

    THREAD BOUNDARY: both generations are read EXACTLY ONCE, here, at the
    decision point. The consumers are always-running background threads and
    the current generation is mutated by the cutover on another thread; a
    second read mid-persist could straddle a rebind and answer two different
    questions about one operation. One read, one answer, one operation.
    """
    stamped = getattr(operation, "binding_generation", None)
    if stamped is None:
        return False
    from sglang.srt.mem_cache.hicache_phase_binding import current_generation

    now = current_generation()
    if int(stamped) == int(now):
        return False
    attr = f"_{kind}_stale_refusals"
    setattr(controller, attr, getattr(controller, attr, 0) + 1)
    n = getattr(controller, attr)
    if n <= 5 or n % 100 == 0:
        logger.warning(
            "#719/W35 STALE %s REFUSED: request %s was opened under binding "
            "generation %s and the binding is now %s. Declining it -- its host "
            "slots belong to a pool that may have been repurposed, and "
            "persisting them would write bytes that do not match the "
            "content-addressed hash they would be stored under. The prefix "
            "misses later instead. (%d so far.)",
            kind.upper(),
            getattr(operation, "request_id", "?"),
            stamped,
            now,
            n,
        )
    return True


class HiCacheController:
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
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
        enable_storage_metrics: bool = False,
    ):
        self.tp_group = tp_group
        self.attn_cp_group = attn_cp_group
        self.attn_tp_group = attn_tp_group
        self.pp_group = pp_group
        self.prefetch_sync_groups: List[torch.distributed.ProcessGroup] = []
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        # #706 keeps the WRAPPER too: only the hybrid pool knows its layers by
        # GLOBAL id (full_attention_layer_id_mapping / mamba_map). The unwrapped
        # full_kv_pool reports start_layer 0 and a bare layer count on every
        # stage, which cannot distinguish stage 1 from stage 0.
        self.mem_pool_device_hybrid = mem_pool_device
        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage_metrics = enable_storage_metrics

        # Draft KV pool support (best-effort piggyback on target L2/L3 ops).
        self.has_draft = False
        self.mem_pool_device_draft = None
        self.mem_pool_host_draft = None
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        # #861: the phase that owns the drafter (None on a single-phase
        # instance), the #719 binding generation the registration was minted
        # at, and the drafter-identity suffix its persisted pages carry. All
        # three are read by ``draft_tier_armed`` and nowhere else.
        self.draft_owner_phase = None
        self.draft_binding_generation = None
        self.draft_identity = None
        self._draft_disarm_warned = set()

        # Default storage page IO functions (may be overridden by attach).
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # Dedicated stop event for storage background threads (prefetch/backup).
        self.storage_stop_event = threading.Event()

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()

        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                raise ValueError(f"Failed to create storage backend: {e}") from e

    def get_attn_cp_rank_and_size(self) -> tuple[int, int]:
        """Derive CP rank/size from the attn_cp process group."""
        if self.attn_cp_group is not None:
            return (
                torch.distributed.get_rank(group=self.attn_cp_group),
                torch.distributed.get_world_size(group=self.attn_cp_group),
            )
        return 0, 1

    def _create_prefetch_sync_groups(self) -> None:
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        self.prefetch_sync_groups = []
        seen_rank_sets = set()

        if self.attn_cp_group is not None or self.attn_tp_group is not None:
            base_groups = [self.attn_cp_group, self.attn_tp_group]
        else:
            base_groups = [self.tp_group]

        for group in base_groups:
            if group is None or torch.distributed.get_world_size(group=group) == 1:
                continue
            group_ranks = tuple(torch.distributed.get_process_group_ranks(group))
            if group_ranks in seen_rank_sets:
                continue
            seen_rank_sets.add(group_ranks)
            self.prefetch_sync_groups.append(
                create_custom_parallel_group(
                    group_ranks=list(group_ranks), backend="gloo"
                )
            )

    def _destroy_prefetch_sync_groups(self) -> None:
        for group in self.prefetch_sync_groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass
        self.prefetch_sync_groups = []

    def _all_reduce_prefetch_groups(self, tensor: torch.Tensor, op) -> None:
        for group in self.prefetch_sync_groups:
            torch.distributed.all_reduce(tensor, op=op, group=group)

    def _start_storage_threads(self):
        """Start storage prefetch/backup threads and their queues.

        This is used by runtime attach, and also by reset when storage is enabled.
        """
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()

        self.prefetch_revoke_queue: Queue[str] = Queue()
        self.ack_backup_queue: Queue[StorageOperation] = Queue()
        self.host_mem_release_queue: Queue[torch.Tensor] = Queue()

        self.prefetch_thread.start()
        self.backup_thread.start()

    def _stop_storage_threads(self):
        """Stop storage prefetch/backup threads and drain internal queues.

        Caller should ensure no in-flight requests.
        """
        # Always request stop. This is safe even when storage is already disabled,
        # and makes detach truly idempotent (previous partial detach may have left
        # threads alive).
        # NOTE: do NOT clear storage_stop_event unless threads have fully stopped; otherwise
        # a still-alive thread may resume and touch released state.
        self.storage_stop_event.set()

        # Best-effort wakeups so threads exit promptly even if blocked on queues.
        try:
            if hasattr(self, "prefetch_queue"):
                self.prefetch_queue.put_nowait(None)
            if hasattr(self, "backup_queue"):
                self.backup_queue.put_nowait(None)
            if hasattr(self, "prefetch_buffer"):
                self.prefetch_buffer.put_nowait(None)
        except Exception:
            pass

        # Best-effort joins (threads are daemon, but join keeps state clean).
        threads = []
        if hasattr(self, "prefetch_thread"):
            threads.append(self.prefetch_thread)
        if hasattr(self, "backup_thread"):
            threads.append(self.backup_thread)
        if hasattr(self, "prefetch_io_aux_thread"):
            threads.append(self.prefetch_io_aux_thread)

        for t in threads:
            try:
                t.join(timeout=10)
            except Exception:
                pass

        alive = [t for t in threads if getattr(t, "is_alive", lambda: False)()]
        if alive:
            logger.error(
                "Failed to stop HiCache storage threads cleanly: %s",
                [getattr(t, "name", repr(t)) for t in alive],
            )
            raise RuntimeError("Failed to stop HiCache storage threads cleanly.")

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        """Attach (enable) storage backend at runtime.

        Requirement: no in-flight requests. This call is expected to run on the scheduler
        thread (control path), not concurrently with prefetch/backup.
        """
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")

        # Defensive: a previous partial detach may have flipped `enable_storage` but
        # left background threads alive. Attaching on top of them is unsafe.
        try:
            self._stop_storage_threads()
        except Exception as e:
            raise RuntimeError(
                "Cannot attach storage backend: previous detach did not stop storage threads cleanly."
            ) from e

        # Rollback-safe init: if creation fails, keep controller state consistent
        # for future attach attempts.
        self.storage_backend_type = storage_backend
        from sglang.srt.mem_cache.utils import get_hash_str

        self.get_hash_str = get_hash_str
        self.storage_config = self._generate_storage_config(
            model_name, storage_backend_extra_config
        )
        # Weighted uneven-DCP owner mode: page files are owner-written and
        # rank-shared; only the file backend implements that key scheme, and
        # the per-page owner rule needs page_size == 1 (a multi-token page
        # would span owner ranks). Fail fast instead of silently writing an
        # allocation-dependent (corrupt) store (task #60).
        if self.storage_config.dcp_owner_mode:
            if storage_backend != "file":
                raise NotImplementedError(
                    "Weighted uneven-DCP HiCache storage currently supports "
                    f"only the 'file' backend, got '{storage_backend}'."
                )
            if self.page_size != 1:
                raise NotImplementedError(
                    "Weighted uneven-DCP HiCache storage requires page_size == 1, "
                    f"got {self.page_size}."
                )
        # #706 whole-page protocol: same two conditions, for the layer axis.
        # Only the file backend can assemble one page from several stages
        # (partial byte-range writes), and a canonical page is ONE token's
        # attention layers. The boot-time twin of this check lives in
        # ServerArgs._handle_phase_flip; both exist because this one also
        # covers a storage backend attached at RUNTIME, where no ServerArgs
        # validation runs.
        if self.storage_config.canonical_kv_page is not None:
            if storage_backend != "file":
                raise NotImplementedError(
                    "The #706 canonical KV page currently supports only the "
                    f"'file' backend, got '{storage_backend}'."
                )
            if self.page_size != 1:
                raise NotImplementedError(
                    "The #706 canonical KV page requires page_size == 1, got "
                    f"{self.page_size}."
                )
        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
        )

        # Use storage backend factory for dynamic backend creation
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, self.mem_pool_host
            )
            self.storage_backend.register_mem_pool_host(self.mem_pool_host)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            self.prefetch_threshold = max(prefetch_threshold, self.page_size)
            # Budget speculative prefetch at half the host pool, leaving the rest for the write-back staging path.
            self.prefetch_capacity_limit = int(0.5 * self.mem_pool_host.size)
            # tracking the number of tokens locked in prefetching, updated by the main scheduler thread
            self.prefetch_tokens_occupied = 0

            # Use dedicated gloo groups so storage prefetch sync is isolated
            # from other collectives and consistent across CPxTP participants.
            self._create_prefetch_sync_groups()

            # Select the get and set functions
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (
                self.storage_backend_type
                in ["hf3fs", "mooncake", "eic", "nixl", "simm", "mori"]
            ) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

            self._maybe_register_draft_with_storage()

            # Ensure stop_event is clear before starting threads.
            self.storage_stop_event.clear()
            self._start_storage_threads()
        except Exception:
            # Best-effort cleanup for partial init.
            try:
                self._stop_storage_threads()
            except Exception:
                pass
            self._destroy_prefetch_sync_groups()
            try:
                if (
                    hasattr(self, "storage_backend")
                    and self.storage_backend is not None
                ):
                    if hasattr(self.storage_backend, "close"):
                        self.storage_backend.close()
            except Exception:
                pass
            self.storage_backend = None
            self.storage_backend_type = None
            self.enable_storage = False
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set
            self.draft_page_get_func = None
            self.draft_page_set_func = None
            raise

    def detach_storage_backend(self):
        """Detach (disable) storage backend at runtime.

        Requirement: no in-flight requests. This will stop storage threads and release
        the backend instance (best-effort close).
        """
        # Idempotent cleanup: even if `enable_storage` is already False,
        # we may still have leftover resources (threads/backend/process group) from a
        # previous partial detach. We attempt cleanup whenever possible.
        try:
            self._stop_storage_threads()
        except Exception as e:
            # Do not proceed tearing down backend/process group if threads are not
            # fully stopped; otherwise still-alive threads may touch released state.
            # Caller can retry detach.
            logger.exception("Stop storage threads failed: %s", e)
            # IMPORTANT: Do not silently succeed. Upper layers rely on exceptions here
            # to avoid flipping `enable_storage` flags while threads are still alive.
            raise RuntimeError("Stop storage threads failed; detach aborted.") from e

        # Best-effort destroy process groups created for storage ops.
        self._destroy_prefetch_sync_groups()

        # Best-effort close (some backends rely on GC/destructor).
        try:
            if (
                hasattr(self, "storage_backend")
                and self.storage_backend is not None
                and hasattr(self.storage_backend, "close")
            ):
                self.storage_backend.close()
        except Exception:
            logger.exception("Failed to close storage backend cleanly.")

        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage = False
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        # Now it's safe to clear the stop event for future re-attach.
        self.storage_stop_event.clear()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        if storage_backend_extra_config is None:
            storage_backend_extra_config = {}

        if is_dp_attention_enabled():
            self.tp_rank = get_parallel().attn_tp_rank
            self.tp_size = get_parallel().attn_tp_size
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_parallel().tp_rank
            self.tp_size = get_parallel().tp_size
            self.dp_rank = 0

        self.pp_rank = get_parallel().pp_rank
        self.pp_size = get_parallel().pp_size

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        # DeepSeekV4TokenToKVPool has compressed MLA-style rank-replicated cache
        # data. storage only needs rank 0 to write it back.
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        is_mla_model = isinstance(self.mem_pool_device, MLATokenToKVPool)
        is_compressed_mla_model = isinstance(
            self.mem_pool_device, DeepSeekV4TokenToKVPool
        )
        is_rank_replicated = is_mla_model or is_compressed_mla_model
        # Least Common Multiple among heterogeneous tp size
        tp_lcm_size = storage_backend_extra_config.pop("tp_lcm_size", None)
        should_split_heads = False

        if tp_lcm_size:
            assert tp_lcm_size % self.tp_size == 0, (
                "tp_lcm_size must be divisible by tp_size."
            )
            should_split_heads = (
                not is_rank_replicated
                and self.mem_pool_host.layout == "page_head"
                and tp_lcm_size > self.tp_size
            )

        attn_cp_rank, attn_cp_size = self.get_attn_cp_rank_and_size()

        # Page hashes cover token ids only and the backend suffix covers
        # served_model_name + parallel geometry; the KV byte format
        # (dtype/quantization/kv_cache_dtype) and the weights revision are in
        # neither. Persistent-tier entries outlive the process, so without
        # this hash a later run sharing served_model_name and storage location
        # could silently read pages written in another byte format. The
        # scheduler process publishes ServerArgs before any storage attach;
        # the fallback only covers bare-controller unit tests.
        try:
            server_args = get_server_args()
        except ValueError:
            server_args = None
        model_identity_hash = (
            canonical_identity_hash_for(
                server_args,
                bool(getattr(server_args, "phase_flip_canonical_kv_page", False)),
            )
            if server_args is not None
            else None
        )

        # #706: this rank's window in the geometry-neutral page. Built only
        # when the format is switched on, and built from the pools that KNOW
        # their global layer ids -- never inferred from a layer count, because
        # the host pool's own start_layer is 0 on every PP stage.
        canonical_kv_page = None
        canonical_mamba_blob = None
        if server_args is not None and getattr(
            server_args, "phase_flip_canonical_kv_page", False
        ):
            from sglang.srt.mem_cache.canonical_page_store import (
                build_page_window,
                resolve_attn_layer_ids,
            )

            model_config = server_args.get_model_config()
            # #706: the id list is RESOLVED, not guessed. The previous
            # `full_attention_layer_ids or range(n)` read an empty SWA-scoped
            # list as proof of a dense model and cut this GDN hybrid's page
            # against 64 slots instead of 16. See resolve_attn_layer_ids for
            # the ladder and why the fix is not in get_hybrid_layer_ids.
            attn_layer_ids = resolve_attn_layer_ids(model_config)
            canonical_kv_page = build_page_window(
                attn_layer_ids, self.mem_pool_device_hybrid, self.mem_pool_host
            )
            logger.info(
                "#706 canonical KV page active: slots [%d, %d) of %d, %d B per "
                "slot; KV keys carry content only (no tp/pp suffix).",
                canonical_kv_page.first_slot,
                canonical_kv_page.first_slot + canonical_kv_page.num_slots,
                canonical_kv_page.spec.num_attn_layers,
                canonical_kv_page.cell_bytes,
            )
            canonical_mamba_blob = self._canonical_mamba_window(
                server_args, model_config
            )

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            # TODO(hzh): Rename is_mla_model to is_rank_replicated.
            is_mla_model=is_rank_replicated,
            enable_storage_metrics=self.enable_storage_metrics,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            model_identity_hash=model_identity_hash,
            tp_lcm_size=tp_lcm_size,
            should_split_heads=should_split_heads,
            extra_config=storage_backend_extra_config,
            # #810: a staging host tier makes this backend the retention tier.
            host_role=getattr(server_args, "hicache_host_role", "retention"),
            # Weighted uneven-DCP: KV pages are token-sharded with FULL
            # replicated kv-heads -> owner-written, rank-shared page files.
            dcp_owner_mode=self._dcp_owner_ctx() is not None,
            # #706: full-width pages, stage-offset writes, suffix-free KV keys.
            canonical_kv_page=canonical_kv_page,
            canonical_mamba_blob=canonical_mamba_blob,
        )

    def _canonical_mamba_window(self, server_args, model_config):
        """This rank's window in the canonical GDN blob (#706 slice 2).

        ``None`` only when the model has NO linear/GDN layers -- then there is
        no blob and the KV page alone is the whole prefix. For a hybrid model
        this must succeed or attach fails: a canonical KV page beside a
        phase-local GDN blob delivers ZERO usable prefix, because
        ``batch_exists_v2`` takes the minimum across pools and the mamba pool is
        registered TRAILING_PAGES. Silently running KV-only would look like the
        feature was on while every cross-phase lookup missed.
        """
        from sglang.srt.mem_cache.canonical_page_store import (
            build_mamba_window,
            derive_mamba_blob_spec,
            local_mamba_layer_range,
        )

        cache_params = getattr(model_config, "mamba2_cache_params", None)
        mamba_layer_ids = list(getattr(cache_params, "layers", None) or [])
        if not mamba_layer_ids:
            return None

        hybrid = self.mem_pool_device_hybrid
        mamba_pool = getattr(hybrid, "mamba_pool", None)
        if mamba_pool is None:
            raise NotImplementedError(
                "#706: this model has GDN/linear layers but the KV pool exposes "
                "no mamba pool, so the GDN blob cannot be made phase-uniform. "
                "Refusing rather than serving a store whose KV pages are "
                "geometry-free while its GDN blobs are not -- that combination "
                "hits nothing at all."
            )
        spec = derive_mamba_blob_spec(
            model_config, mamba_pool, num_linear_layers=len(mamba_layer_ids)
        )
        layer_lo, layer_hi = local_mamba_layer_range(hybrid, mamba_layer_ids)
        # Head sharding: the uneven-TP vector when set, equal shares otherwise.
        # Under the PP prefill phase tp_size is 1, so this is [1] and the window
        # carries full heads -- exactly the layer-only cut.
        raw_ratio = getattr(server_args, "rank_tp_ratio", None)
        ratios = (
            [int(x) for x in str(raw_ratio).split(",")]
            if raw_ratio
            else [1] * max(1, self.tp_size)
        )
        window = build_mamba_window(
            spec,
            ratios=ratios,
            rank=self.tp_rank,
            layer_lo=layer_lo,
            layer_hi=layer_hi,
        )
        logger.info(
            "#706 canonical GDN blob active: layers [%d, %d) of %d, %d of %d "
            "blob bytes on this rank, %d extent(s).",
            layer_lo,
            layer_hi,
            spec.num_layers,
            window.payload_bytes,
            window.total_bytes,
            len(window.extents),
        )
        return window

    def reset(self):
        self.storage_stop_event.set()

        self.write_queue.clear()
        self.load_queue.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            self.prefetch_thread.join()
            self.backup_thread.join()
            self.prefetch_queue.queue.clear()
            self.backup_queue.queue.clear()
            self.prefetch_revoke_queue.queue.clear()
            self.ack_backup_queue.queue.clear()
            self.host_mem_release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

        self.storage_stop_event.clear()

        if self.enable_storage:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_thread_func, daemon=True
            )
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.prefetch_thread.start()
            self.backup_thread.start()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        # #718: this controller is bound to the pool it was BUILT with. While
        # the flip routes to its TP stack, that is not the pool the model
        # writes into, so this copy would persist another row's bytes under a
        # content-addressed key. Refuse: a prefix that is not staged is a miss
        # later, which is the cheap failure.
        if device_tier_disarmed("write"):
            return None
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        # #760/W35: ONE AUTHORITY, FOUR CALLERS. This block used to live here
        # inline while the load path had no equivalent and the hybrid subclass
        # had neither -- four consume points that must agree about one
        # question. The question now lives in `consume_gate` and this is one of
        # its callers; see that function for the full #760 argument and for why
        # the phase predicate and the generation stamp are BOTH required.
        if not consume_gate(self, "write_queue", "write"):
            return

        op = CacheOperation.merge_ops(self.write_queue)
        # Kernel write-back keeps host indices on CPU only for page_first AND only
        # when the staged JIT write-back kernel is available (it stages through
        # device memory and accepts CPU destination indices). Otherwise we fall back
        # to the plain transfer kernel, whose CUDA/HIP implementation requires
        # device-resident destination indices -- so the indices must be moved to the
        # device first. Without the can_use_write_back_jit check this crashes on
        # backends where the JIT kernel is unavailable, with
        # "Destination indices must be a CUDA tensor".
        if (
            self.io_backend == "kernel"
            and self.mem_pool_host.layout == "page_first"
            and getattr(self.mem_pool_host, "can_use_write_back_jit", False)
        ):
            host_indices, device_indices = op.host_indices, op.device_indices
        else:
            host_indices, device_indices = self.move_indices(
                op.host_indices, op.device_indices
            )
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        # Weighted uneven-DCP: back up only this rank's owned tokens, from
        # their COMPACT device slots (identity when the gate is off). The
        # draft pool below is NOT DCP-token-sharded (full token context,
        # global indices) and keeps the raw pair list.
        kv_host_indices, kv_device_indices = self._dcp_kv_transfer_pairs(
            host_indices, device_indices
        )

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                kv_host_indices,
                kv_device_indices,
                self.io_backend,
            )
            if self.draft_tier_armed("write"):
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            for indices in (
                host_indices,
                device_indices,
                kv_host_indices,
                kv_device_indices,
            ):
                if indices.is_cuda:
                    indices.record_stream(self.write_stream)

        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        # #718: the mirror hazard. This copy would fill rows in the bound (PP)
        # pool while the model reads the flip stack's, and the tree would
        # report the prefix resident -- so attention would read rows nobody
        # filled. Refuse: a prefetch that does not land is a miss now.
        if device_tier_disarmed("load"):
            return None
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def _dcp_owner_ctx(self) -> Optional[tuple]:
        """(S, lo, hi) of this rank's weighted uneven-DCP owner range, or None.

        Cached: the token vector is installed once at engine init, before the
        cache controller is constructed. When active, the radix tree hands the
        controller GLOBAL allocator indices while the device KV pool only holds
        this rank's COMPACT owned slots -- every device-side KV transfer must
        go through _dcp_kv_transfer_pairs (task #60)."""
        if not hasattr(self, "_dcp_owner_ctx_cache"):
            from sglang.srt.distributed.utils import uneven_dcp_owner_bounds

            self._dcp_owner_ctx_cache = uneven_dcp_owner_bounds()
        return self._dcp_owner_ctx_cache

    def _dcp_kv_transfer_pairs(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ):
        """Translate a (host page, GLOBAL device slot) pair list into the
        (host page, COMPACT device slot) pairs this rank actually owns.

        Weighted uneven-DCP stores token L's KV on the rank whose owner range
        [lo, hi) contains L % S, at compact physical slot
        (L // S) * (hi - lo) + (L % S - lo). Tokens outside [lo, hi) do not
        exist on this rank's device pool: their host pages are neither backed
        up nor loaded here (their L3 page file is written by the owner rank
        and holds the FULL replicated kv-heads). Gate off -> identity, keeping
        the stock path byte-identical."""
        ctx = self._dcp_owner_ctx()
        if ctx is None:
            return host_indices, device_indices
        S, lo, hi = ctx
        dev = device_indices.to(torch.int64)
        off = dev % S
        owned = (off >= lo) & (off < hi)
        compact = (dev // S) * (hi - lo) + (off - lo)
        host_owned = host_indices[owned.to(host_indices.device)]
        return host_owned, compact[owned]

    def move_indices(self, host_indices: torch.Tensor, device_indices: torch.Tensor):
        # move indices to GPU if using kernels, to host if using direct indexing
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            elif self.mem_pool_host.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
            else:
                raise ValueError(
                    f"Unsupported layout {self.mem_pool_host.layout!r} for io backend 'direct'"
                )
        elif self.io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        else:
            raise ValueError(f"Unsupported io backend")

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        # #760/W35: THE CONSUME-TIME CHECKS THE LOAD PATH NEVER HAD.
        # `load()` asks the phase question at ENQUEUE and is right there; the
        # cutover lands between enqueue and here, and `load()` is a SEPARATE
        # call from this one, so the gap is real rather than theoretical. A
        # stale load fills device rows from host slots this phase does not own
        # and the tree marks the prefix RESIDENT -- attention then reads KV
        # nobody wrote, with no assertion anywhere. Checked before a producer
        # is allocated, so a refused batch costs nothing downstream.
        if not consume_gate(self, "load_queue", "load"):
            return -1

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        # Weighted uneven-DCP: load only this rank's owned tokens into their
        # COMPACT device slots (identity when the gate is off). Draft pool is
        # not DCP-token-sharded and keeps the raw pair list.
        kv_host_indices, kv_device_indices = self._dcp_kv_transfer_pairs(
            host_indices, device_indices
        )

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    kv_host_indices,
                    kv_device_indices,
                    i,
                    self.io_backend,
                )
                if (
                    self.draft_tier_armed("load")
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
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the load stream is executing.
            for indices in (
                host_indices,
                device_indices,
                kv_host_indices,
                kv_device_indices,
            ):
                if indices.is_cuda:
                    indices.record_stream(self.load_stream)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id

    def evict_device(self, device_indices: torch.Tensor) -> int:
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def quiesce_device_io(self, reason: str) -> float:
        """#760: finish every in-flight device-tier copy, bounded by the copies.

        Called by the flip runtime at the seam's no-return point, BEFORE any
        pool byte moves. The copies on ``write_stream`` / ``load_stream`` were
        enqueued while the pools they name were live, so FINISHING them is
        correct -- they become durable cache entries -- while letting them ride
        into the seam races the release of the outgoing phase's backing, which
        is the SIGSEGV both #760 crash specimens died of (3 s after a
        pp_to_tp cutover, inside backup_from_device_all_layer, below the
        Python seam). No Python-side predicate can close that window: the
        enqueue-time and consume-time checks both pass legitimately, the copy
        is torn by the STREAM's asynchrony.

        The wait is bounded by construction: a stream synchronize completes
        when the already-enqueued copies do (PCIe transfer time of the
        backlog), and this thread is the only producer of device-tier I/O, so
        nothing refills the streams while it waits. Not a collective -- each
        rank drains its own streams -- so it cannot wedge the group (#630).

        Returns the wait in seconds; the caller logs it into the seam record
        so a slow drain is attributable instead of vanishing into the flip's
        residual (#690).
        """
        t0 = time.perf_counter()
        for name in ("write_stream", "load_stream"):
            stream = getattr(self, name, None)
            if stream is None:
                continue
            try:
                stream.synchronize()
            except Exception as e:  # noqa: BLE001 - a dead stream must be loud
                logger.error(
                    "#760 quiesce (%s): synchronizing %s failed (%s). "
                    "In-flight copies on it may still race the seam.",
                    reason,
                    name,
                    e,
                )
        elapsed = time.perf_counter() - t0
        logger.info(
            "#760 device-tier I/O quiesced for %s in %.1f ms (write and load "
            "streams drained while their pools are still live).",
            reason,
            elapsed * 1000.0,
        )
        return elapsed

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        self.mem_pool_host.free(host_indices)
        return len(host_indices)

    def draft_tier_armed(self, direction: str) -> bool:
        """THE ONE GATE for draft-half I/O. Six consume points, one answer.

        #861, and the shape is 2bf0f53498's ("give all four consume points one
        gate") applied to the participant that never had one. Before this, six
        sites each read ``self.has_draft`` directly -- four here and two in
        ``HybridCacheController``'s overrides, which is the lane this rig
        actually runs (the W31/W32/W33 shape: a correct mechanism a second copy
        overrides is a mechanism that never runs).

        Three conditions, and each one is a corruption if it is skipped:

        1. REGISTERED AT ALL. The pre-#861 state on a flip boot, and the
           acceptance collapse this fixes.
        2. THE ACTIVE PHASE OWNS THE DRAFTER. There is one drafter in a
           phase-flip process and it lives on the TP stack; the PP prefill
           phase has none. A draft backup taken in PP would persist rows no
           drafter ever wrote under a content-addressed key, for the TP phase
           to load as valid -- strictly worse than the missing registration.
           ``None`` means "single-phase instance", which skips this term and
           keeps every non-flip deployment byte-identical.
        3. THE BINDING HAS NOT MOVED UNDER US. Draft host indices are 1-to-1
           with the TARGET host pool's, and ``hicache_phase_binding._stamp``
           re-points ``mem_pool_host`` at every rebind. A registration minted
           at generation g indexes generation g's slot space; consumed at g+1
           it addresses a different pool. Same authority as the releases and
           the write-backs (``write_back_stamp_is_current``), one more
           consumer -- not a second stamp scheme.

        The TARGET tier's own ``device_tier_disarmed`` is NOT re-asked here:
        every device-side consume point already sits behind it, and asking
        twice would let the two answers drift. The L3 points are storage-side
        and correctly outside it.
        """
        if not self.has_draft:
            return False
        if self.draft_owner_phase is not None:
            from sglang.srt.mem_cache.hicache_phase_guard import active_phase

            if active_phase() != self.draft_owner_phase:
                self._warn_draft_disarmed(
                    direction,
                    f"the active phase is not '{self.draft_owner_phase}', which "
                    f"is the only phase that owns a drafter",
                )
                return False
        if self.draft_binding_generation is not None:
            from sglang.srt.mem_cache.hicache_phase_binding import (
                current_generation,
            )

            if int(self.draft_binding_generation) != int(current_generation()):
                self._warn_draft_disarmed(
                    direction,
                    f"registered at binding generation "
                    f"{self.draft_binding_generation}, current is "
                    f"{current_generation()}",
                )
                return False
        return True

    def _warn_draft_disarmed(self, direction: str, reason: str) -> None:
        """One line per (direction, reason) per process; the state is phase-long."""
        key = f"{direction}:{reason}"
        if key in self._draft_disarm_warned:
            return
        self._draft_disarm_warned.add(key)
        logger.warning(
            "#861 draft-half HiCache %s DISARMED: %s. Target-tier I/O is "
            "unaffected; a prefix restored while this holds carries NO draft "
            "rows, so requests admitted on it must be marked draft-cold "
            "(phase_flip_draft_bootstrap.mark_draft_cold) rather than allowed "
            "to speculate over rows nothing wrote.",
            direction,
            reason,
        )

    def set_draft_kv_pool(
        self,
        draft_device_pool,
        draft_host_pool,
        *,
        owner_phase=None,
        binding_generation=None,
        drafter_identity=None,
    ) -> None:
        """Register draft KV pools so L2/L3 ops piggyback draft transfers.

        Idempotent by design: ``rebind_hicache_draft_for_phase`` calls this on
        every pp->tp cutover with the same pools and a fresh generation, so
        re-registration is the normal case rather than an error.
        """
        self.has_draft = True
        self.mem_pool_device_draft = draft_device_pool
        self.mem_pool_host_draft = draft_host_pool
        self.draft_owner_phase = owner_phase
        self.draft_binding_generation = binding_generation
        self.draft_identity = drafter_identity
        logger.info(
            "HiCache draft KV registered: %s (host %d slots), owner_phase=%s, "
            "binding_generation=%s, drafter=%s",
            type(draft_device_pool).__name__,
            draft_host_pool.size,
            owner_phase,
            binding_generation,
            drafter_identity,
        )

        # If storage is already attached, wire up the draft I/O path now.
        # Otherwise this will be deferred until attach_storage_backend().
        self._maybe_register_draft_with_storage()

    def disarm_draft_kv_pool(self, reason: str) -> None:
        """#861: leave the draft half unarmed for the phase being entered.

        Not a teardown: the pools stay referenced so the next cutover into the
        drafter's own phase re-arms by re-stamping rather than re-allocating
        (a pinned host pool per flip would charge the host budget every time,
        and on this box that budget binds -- DESIGN_706 C1).

        Called unconditionally on the leg into a phase without a drafter, so
        "armed" is never a latch. The gate would refuse anyway on the phase
        term, and that redundancy is deliberate: a latched True that only the
        gate contradicts is one refactor away from being trusted.
        """
        if self.has_draft:
            logger.info(
                "#861 draft-half HiCache DISARMED: %s. The pools are kept for "
                "the next cutover into the drafter's phase.",
                reason,
            )
        self.has_draft = False
        self.draft_page_get_func = None
        self.draft_page_set_func = None

    def _maybe_register_draft_with_storage(self) -> None:
        """Pick the draft L3 IO implementation."""
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        if not self.has_draft or not self.enable_storage:
            return

        backend = self.storage_backend_type

        # Multi-pool zero-copy backends.
        if backend == "mooncake":
            # #861 GUARD. The v2 route keys a page by the POOL NAME
            # (`register_mem_host_pool_v2(pool, PoolName.DRAFT)` ->
            # `_get_component_key`), so the drafter-identity suffix the generic
            # route carries cannot ride along without changing the registered
            # pool identity itself. Until task #861 item (a) puts the drafter
            # into `compute_model_identity_hash` -- where every backend and both
            # key routes pick it up -- a v2 draft page would be readable by a
            # different drafter as valid. Refused by name rather than left
            # write-only: a page nobody may read is still a page the NEXT boot
            # may read.
            logger.warning(
                "HiCache draft L3 disabled on the mooncake v2 route (#861): a "
                "v2 page is keyed by pool name, so it cannot carry the drafter "
                "identity the generic route puts in its component key, and a "
                "draft page readable across a drafter change is a silently "
                "wrong draft KV. The L2 host tier is unaffected. Lift this "
                "once #861 (a) folds the drafter into the model identity hash."
            )
            return

        # TODO: support "hf3fs", "eic", "nixl", "simm"
        if backend in {"hf3fs", "eic", "nixl", "simm"}:
            logger.warning(
                "HiCache draft L3 disabled: backend %s does not yet support "
                "draft pool registration.",
                backend,
            )
            return

        # Generic backends.
        self.draft_page_get_func = self._draft_page_get_generic
        self.draft_page_set_func = self._draft_page_set_generic

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.
        """
        operation = PrefetchOperation(
            request_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.prefetch_queue.put(operation)
        return operation

    def terminate_prefetch(self, operation):
        operation.mark_terminate()
        return operation.completed_tokens, operation.hash_value

    def append_host_mem_release(self, host_indices: torch.Tensor, generation=None):
        """Queue host slots for release, ROUTED BY THE BINDING THEY CAME FROM.

        W35. `_drain_release` frees whatever is on this queue against
        `self.mem_pool_host` -- the pool bound NOW. Three producers can put
        entries here after a cutover that name slots from the pool bound
        BEFORE it: `_drain_revoke`, the prefetch transfer thread, and the
        direct path. Freeing those against the current pool is the measured
        W35 double-free ("slots not currently allocated"); dropping them leaks
        slots in a pool that returns on the very next flip, because the flip
        alternates and nothing tears the outgoing pool down.

        ROUTED AT PRODUCE TIME, which is the only point where the generation is
        still known without changing what this queue carries. A stale batch is
        freed immediately against the pool its generation names -- the queue is
        a batching convenience, not a correctness mechanism, so settling one
        batch early costs nothing. Only current-generation slots are queued, so
        by construction nothing on the queue can outlive its binding.

        A stale batch whose pool is UNKNOWN is refused loudly and NOT queued:
        freeing it here would corrupt and queueing it would corrupt later.
        """
        if host_indices.numel() == 0:
            return
        if generation is not None:
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
                        "#719/W35 STALE RELEASE ORPHANED: %d host slot(s) were "
                        "opened under binding generation %s, which names no "
                        "known pool. Not queued and not freed -- freeing them "
                        "against the current pool is the W35 double-free and "
                        "queueing them defers the same crash. (%d so far.)",
                        int(host_indices.numel()),
                        generation,
                        self._stale_release_orphaned,
                    )
                    return
                n = self._stale_release_routed
                if n <= 3 or n % 200 == 0:
                    logger.warning(
                        "#719/W35 STALE RELEASE ROUTED: %d host slot(s) opened "
                        "under binding generation %s are freed against THAT "
                        "generation's pool rather than the one bound now. "
                        "(%d so far.)",
                        int(host_indices.numel()),
                        generation,
                        n,
                    )
                owner.free(host_indices)
                return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            self.host_mem_release_queue.put(page)

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ):
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            inc += self.page_size
        operation.increment(inc)

    # todo: deprecate
    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):
        dummy_page_dst = [
            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values
        ]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            return
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            # Must set the data before increasing the completed tokens.
            # Otherwise this page may be read before being set.
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.page_size],
                page_data[i],
            )
            if not operation.increment(self.page_size):
                break  # Operation terminated by controller

    def _page_transfer(self, operation):
        # Transfer batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]

            # Best-effort draft L3 read before publishing target completion.
            # Otherwise wait_complete can race and load back target KV before
            # draft KV reaches host memory.
            if self.draft_tier_armed("l3-load"):
                self._draft_page_get(batch_hashes, batch_host_indices)

            prev_completed_tokens = operation.completed_tokens
            # Get one batch token, and update the completed_tokens if succeed
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
            # Check termination
            if (
                operation.completed_tokens
                != prev_completed_tokens + len(batch_hashes) * self.page_size
            ):
                operation.mark_terminate()
                break  # Some operations fail or operation terminated by controller

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                if operation is None:
                    continue
                self._page_transfer(operation)
                # operation terminated by controller, release pre-allocated memory
                # W35: this thread runs across cutovers, so the slots it
                # releases may have been opened under an older binding.
                self.append_host_mem_release(
                    operation.host_indices[operation.completed_tokens :],
                    generation=getattr(operation, "binding_generation", None),
                )
            except Empty:
                continue

    def prefetch_rate_limited(self) -> bool:
        """
        Rate limit the prefetching operations to avoid overwhelming the storage backend.
        """
        # cancel prefetch if too much memory is occupied
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        # todo: more sophisticated rate limiting based on storage backend performance
        return False

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        tokens_to_fetch = operation.token_ids
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        storage_query_count = 0
        hash_value = []
        page_hashes = self.get_hash_str(
            tokens_to_fetch, last_hash, page_size=self.page_size
        )

        for start in range(0, len(page_hashes), STORAGE_BATCH_SIZE):
            batch_hashes = page_hashes[start : start + STORAGE_BATCH_SIZE]
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
            hash_value.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size
            if hit_page_num < len(batch_hashes):
                break
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        return hash_value, storage_query_count

    def prefetch_thread_func(self):
        """
        Manage prefetching operations from storage backend to host memory.
        """
        self.prefetch_buffer = Queue()
        self.prefetch_io_aux_thread = threading.Thread(
            target=self.prefetch_io_aux_func, daemon=True
        )
        self.prefetch_io_aux_thread.start()
        while (not self.storage_stop_event.is_set()) or not self.prefetch_queue.empty():
            try:
                operation = self.prefetch_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                hash_value, storage_hit_count = self._storage_hit_query(operation)
                storage_hit_count_tensor = torch.tensor(
                    storage_hit_count, dtype=torch.int
                )
                self._all_reduce_prefetch_groups(
                    storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
                )
                storage_hit_count = storage_hit_count_tensor.item()

                if storage_hit_count < self.prefetch_threshold:
                    # not to prefetch if not enough benefits
                    self.prefetch_revoke_queue.put(operation.request_id)
                    self.append_host_mem_release(operation.host_indices)
                    logger.debug(
                        f"Revoking prefetch for request {operation.request_id} due to insufficient hits ({storage_hit_count})."
                    )
                else:
                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    # free the pre-allocated memory for pages that are not hit
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                    operation.host_indices = operation.host_indices[:storage_hit_count]
                    logger.debug(
                        f"Prefetching {len(operation.hash_value)} pages for request {operation.request_id}."
                    )
                    self.prefetch_buffer.put(operation)

            except Empty:
                continue

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        kv_page_owner_mask: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        operation.kv_page_owner_mask = kv_page_owner_mask
        self.backup_queue.put(operation)
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        data = [
            self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    def _draft_page_set(self, hash_values, host_indices) -> None:
        """Best-effort write draft KV pages to L3 alongside the target backup."""
        if self.draft_page_set_func is None:
            return
        try:
            self.draft_page_set_func(hash_values, host_indices)
        except Exception:
            logger.debug(
                "Draft L3 write failed (best-effort), skipping.", exc_info=True
            )

    def _draft_page_get(self, hash_values, host_indices) -> None:
        """Best-effort read draft KV pages from L3 (mirrors `_draft_page_set`)."""
        if self.draft_page_get_func is None:
            return
        try:
            self.draft_page_get_func(hash_values, host_indices)
        except Exception:
            logger.debug("Draft L3 read failed (best-effort), skipping.", exc_info=True)

    def _draft_page_set_v2(self, hash_values, host_indices) -> None:
        self.storage_backend.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_page_get_v2(self, hash_values, host_indices) -> None:
        self.storage_backend.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_component_name(self) -> str:
        """The component name a persisted draft page is keyed under.

        #861 GUARD, and it is the cheapest correct form rather than the full
        fix. Fix (0) newly makes these pages READABLE, and every HiCache key
        suffix is built from ``compute_model_identity_hash``, which covers the
        TARGET (model_path, revision, dtype, quantization, kv_cache_dtype) and
        carries nothing about the drafter. Two boots agreeing on the target and
        differing in drafter -- another NEXTN checkpoint, MTP<->EAGLE, the #156
        cross-algorithm switch -- would read each other's draft KV as valid,
        with blob length the only accidental guard and equal geometry the
        common case.

        Folding the drafter into the component name makes such a page simply
        NOT EXIST for the other drafter: a clean miss, which is the same
        argument ``HiCacheFile`` already makes for the identity hash it does
        carry. Task #861 item (a) -- drafter identity inside
        ``compute_model_identity_hash`` itself, so EVERY backend and both key
        routes carry it -- is filed and deliberately not built here.

        Falls back to the bare pool name when no identity was supplied, which
        keeps a caller that predates this parameter writing exactly the keys it
        wrote before.
        """
        if not self.draft_identity:
            return str(PoolName.DRAFT)
        return f"{PoolName.DRAFT}-{self.draft_identity}"

    def _draft_page_set_generic(self, hash_values, host_indices) -> None:
        # `{hash}.draft-{drafter}` mirrors HiCacheStorage._get_component_key's
        # `{key}.{pool_name}` convention so target/draft pages never collide,
        # and never collide across drafters either (#861).
        component = self._draft_component_name()
        draft_keys = [f"{h}.{component}" for h in hash_values]
        draft_data = [
            self.mem_pool_host_draft.get_data_page(host_indices[i * self.page_size])
            for i in range(len(draft_keys))
        ]
        self.storage_backend.batch_set(draft_keys, draft_data)

    def _draft_page_get_generic(self, hash_values, host_indices) -> None:
        component = self._draft_component_name()
        draft_keys = [f"{h}.{component}" for h in hash_values]
        draft_dummy = [
            self.mem_pool_host_draft.get_dummy_flat_data_page() for _ in draft_keys
        ]
        draft_pages = self.storage_backend.batch_get(draft_keys, draft_dummy)
        if draft_pages is None:
            return
        for i, p in enumerate(draft_pages):
            if p is not None:
                self.mem_pool_host_draft.set_from_flat_data_page(
                    host_indices[i * self.page_size], p
                )

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        prefix_keys = operation.prefix_keys
        # Weighted uneven-DCP owner mode: only the pages this rank owned at
        # backup time carry real data in the host pool -- write exactly those
        # (the rank-shared page file is complete: full replicated kv-heads).
        # Pages owned by other ranks are persisted by their owners; they still
        # count as completed here so ack/host-release semantics are unchanged.
        owner_mask = getattr(operation, "kv_page_owner_mask", None)
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            # Set one batch token, and record if success.
            # todo: allow partial success
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            if owner_mask is not None:
                owned_pos = [
                    j for j in range(len(batch_hashes)) if bool(owner_mask[i + j])
                ]
                if owned_pos:
                    owned_hashes = [batch_hashes[j] for j in owned_pos]
                    owned_host_indices = torch.cat(
                        [
                            batch_host_indices[
                                j * self.page_size : (j + 1) * self.page_size
                            ]
                            for j in owned_pos
                        ]
                    )
                    success = self.page_set_func(
                        owned_hashes, owned_host_indices, extra_info
                    )
                else:
                    success = True
            else:
                success = self.page_set_func(
                    batch_hashes, batch_host_indices, extra_info
                )
            if not success:
                logger.warning(
                    f"Write page to storage: {len(batch_hashes)} pages failed."
                )
                break

            # Best-effort draft L3 write alongside target.
            if self.draft_tier_armed("l3-write"):
                self._draft_page_set(batch_hashes, batch_host_indices)

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                # W35 CLASS 4: the durable-corruption gate. Checked HERE, on
                # the consumer thread, immediately before the persist and
                # after the operation has been dequeued -- the one point where
                # this operation's fate is decided. See `operation_is_stale`
                # for why a stale backup is REFUSED rather than routed.
                # Acked either way: an unacked operation stalls the queue, and
                # a declined backup is a correct non-persist, exactly as
                # `backup_skip` already is.
                if operation_is_stale(self, operation, "backup"):
                    self.ack_backup_queue.put(operation)
                    continue

                if not self.backup_skip:
                    # Capacity watchdog: rate-limited inside the backend, so this
                    # is cheap per operation. A backend that has stopped writing
                    # (out of disk) still runs the backup: the individual page
                    # writes turn into refusals, which the ack path already
                    # treats as a partial backup, so the caller sees a cache
                    # miss rather than a stalled queue.
                    self.storage_backend.check_disk_space()
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)

            except Empty:
                # Idle: an idle backend is exactly when a filesystem filling up
                # from other writers would otherwise go unnoticed.
                if self.enable_storage and self.storage_backend is not None:
                    try:
                        self.storage_backend.check_disk_space()
                    except Exception as e:  # never kill the worker over a probe
                        logger.warning(f"HiCache capacity watchdog failed: {e}")
                continue
