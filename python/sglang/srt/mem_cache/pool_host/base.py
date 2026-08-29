from __future__ import annotations

import abc
import logging
import threading
from functools import wraps
from typing import Optional

import torch

from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.pinned_host_budget import (
    check_and_register_pinned_post,
    unregister_pinned_post,
)
from sglang.srt.mem_cache.pool_host.common import (
    _cuda_host_unregister,
    get_allocator_from_storage,
)
from sglang.srt.utils import is_cuda, is_hip
from sglang.srt.mem_cache.pinned_host_budget import revert_pinned_posts_on_failure

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()

# Host RAM to leave free when sizing HiCache pools (OS, other processes).
HICACHE_HOST_MEMORY_RESERVE_BYTES: int = 10 * (1024**3)

_WRITE_BACK_STAGING_PAGE_CHUNK = 64


def sync_fixed_hicache_size(size: int, host_size: int) -> int:
    """Sync fixed-size HiCache token capacity across ranks with unequal
    bytes/token.

    A fixed --hicache-size is specified in GB, but the bytes/token can
    differ per rank: PP stages own different layers, and uneven-TP ranks
    (--rank-tp-ratio) own different kv-head/GDN-state shares. Use the
    global minimum token capacity within the affected group so all ranks
    expose the same host-cache capacity (the lockstep schedulers and the
    host radix index must agree on one slot count; each rank still
    allocates its own per-rank-sized host buffer for that count).
    Ratio-based sizing already derives from the synced device pool size.
    """
    if host_size <= 0 or not torch.distributed.is_available():
        return size

    if not torch.distributed.is_initialized():
        return size

    group = None
    try:
        from sglang.srt.distributed.utils import get_tp_partition_ratios

        if get_tp_partition_ratios():
            # Uneven TP: per-token bytes differ across the TP ranks, so
            # the min must span the whole (pure-TP) world group.
            from sglang.srt.distributed.parallel_state import get_world_group

            world_group = get_world_group()
            if world_group.world_size > 1:
                group = world_group.cpu_group
    except AssertionError:
        pass

    if group is None:
        try:
            from sglang.srt.distributed.parallel_state import get_pp_group

            pp_group = get_pp_group()
        except AssertionError:
            return size

        if pp_group.world_size <= 1:
            return size
        group = pp_group.cpu_group

    tensor = torch.tensor(size, dtype=torch.int64)
    torch.distributed.all_reduce(
        tensor,
        op=torch.distributed.ReduceOp.MIN,
        group=group,
    )
    synced_size = int(tensor.item())

    if synced_size != size:
        logger.info(
            "Sync fixed-size HiCache host token capacity from %d to %d.",
            size,
            synced_size,
        )
    return synced_size


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class HostKVCache(abc.ABC):
    @revert_pinned_posts_on_failure  # #729
    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
        allocator_type: str = "default",
        budget_label: Optional[str] = None,
        budget_flag: str = "--hicache-size / --hicache-ratio",
    ):
        # ``budget_label``/``budget_flag`` name this pool's post in the joint
        # pinned-host-RAM guard (#550). They are parameters rather than class
        # properties because the SAME MHATokenToKVPoolHost class backs both
        # claimants: HiCache's L2 tier and kv-session-offload's spill pool, and
        # a refusal has to tell the operator WHICH flag to lower. The defaults
        # are HiCache's, so every pre-existing caller is unchanged.
        self.budget_label = budget_label or type(self).__name__
        self.budget_flag = budget_flag
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.can_use_write_back_jit = False

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = sync_fixed_hicache_size(
                int(host_size * 1e9 // self.size_per_token), host_size
            )
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align up the host memory pool size to the page size
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        if self.size <= device_pool.size:
            logger.warning(
                "HiCache host KV pool (%d tokens) is smaller than the device pool (%d tokens);"
                "L2 cache effectiveness is reduced."
                "Consider increasing --hicache-ratio (or --hicache-size) for higher L2 cache hit rate.",
                self.size,
                device_pool.size,
            )

        # Verify there is enough available host memory -- for THIS pool plus
        # every pinned pool already allocated in this process (#550). The
        # figure comes from the #407 memtier profile, not psutil: under lxcfs
        # /proc/meminfo does not describe what this container may have.
        requested_bytes = self.size * self.size_per_token
        check_and_register_pinned_post(
            name=self.budget_label,
            flag=self.budget_flag,
            requested_bytes=requested_bytes,
            reserve_bytes=HICACHE_HOST_MEMORY_RESERVE_BYTES,
        )
        # Name the POST, not the class of feature. This line used to say
        # "hierarchical KV cache" unconditionally, so a kv-session-offload
        # spill pool announced itself as HiCache -- with both features live
        # (#550) that reading is simply wrong, and the allocation log is the
        # first place an operator looks when the joint budget refuses.
        logger.info(
            "Allocating %.2f GB pinned host memory for %s.",
            requested_bytes / 1e9,
            self.budget_label,
        )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()

    def destroy(self):
        """Unregister pinned host buffers in userspace before process exit.

        Large cudaHostRegister'd buffers are otherwise unpinned by the kernel
        during SIGKILL reclaim, which can stall teardown in uninterruptible
        sleep for tens of seconds. Idempotent. (Only the host_register path
        needs this; npu/musa pin_memory buffers are freed by torch.)
        """
        if getattr(self, "_destroyed", False):
            return
        self._destroyed = True
        # The joint budget must stop charging for a buffer that is gone (#550),
        # or a re-init inside one process would be refused for RAM nothing is
        # holding any more.
        unregister_pinned_post(getattr(self, "budget_label", type(self).__name__))
        buffers = getattr(self, "kv_buffer", None)
        if buffers is not None and self.pin_memory and (_is_cuda or _is_hip):
            if not isinstance(buffers, (list, tuple)):
                buffers = [buffers]
            for buf in buffers:
                if buf is not None:
                    _cuda_host_unregister(buf)
        self.kv_buffer = None

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    def _is_device_layer_sharded(self, device_pool=None) -> bool:
        device_pool = device_pool or self.device_pool
        return bool(device_pool.layer_shard_enabled)

    def _device_owned_layer_range(self, device_pool=None) -> tuple[int, int]:
        """Contiguous ``[start, end)`` local device layers this rank stores.

        ``(0, layer_num)`` when the device pool is not layer-sharded.
        """
        device_pool = device_pool or self.device_pool
        if not self._is_device_layer_sharded(device_pool):
            return 0, device_pool.layer_num
        return device_pool._owned_local_layer_range()

    def _effective_host_layer_num(self, device_pool=None) -> int:
        """Number of layers the host pool allocates for this rank."""
        device_pool = device_pool or self.device_pool
        if not self._is_device_layer_sharded(device_pool):
            return device_pool.layer_num
        shard_size = device_pool.layer_shard_size
        return (device_pool.layer_num + shard_size - 1) // shard_size

    def _is_device_layer_owned(self, device_pool, layer_id: int) -> bool:
        start, end = self._device_owned_layer_range(device_pool)
        return start <= layer_id < end

    def _host_layer_index(self, layer_id: int, device_pool=None) -> int:
        """Map a full local device layer id to its compacted host-buffer slot."""
        start, _ = self._device_owned_layer_range(device_pool)
        return layer_id - start

    def _owned_device_layer_ids(self, device_pool) -> list[int]:
        start, end = self._device_owned_layer_range(device_pool)
        return list(range(start, end))

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        """
        Load KV data from the host memory pool to the device memory pool for a specific layer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from the device memory pool to the host memory pool for all layers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a flat data page from the host memory pool.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """
        Get a dummy flat data page from the host memory pool.
        This is used for prefetching or initializing empty pages.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to the host memory pool.
        """
        raise NotImplementedError()

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        """Return True if per-page strides are multiples of *page_size_bytes*.

        Subclasses should override this with a layout-specific stride formula.
        This base implementation logs a warning and returns False (safe default).
        """
        logger.warning(
            "%s does not implement is_stride_page_aligned(); assuming not aligned. "
            "O_DIRECT with a file-based NIXL backend will fall back to copy mode for this pool.",
            type(self).__name__,
        )
        return False

    @synchronized
    def clear(self):
        # DIAGNOSTIC ONLY (#905 window, 2026-08-26): count how many times this
        # pool's slot bookkeeping has been wiped. A `clear()` makes EVERY index
        # handed out before it look "not currently allocated" afterwards, which
        # is exactly what the double-free assertion below reports. Comparing the
        # epoch a span was allocated under against the epoch it is freed under
        # turns that assertion from a symptom into a mechanism. No behaviour
        # change: a counter and a log line.
        self._clear_epoch = getattr(self, "_clear_epoch", 0) + 1
        logger.warning(
            "#905 HOST-POOL CLEAR: pool %r id=%d size=%d -> clear_epoch=%d. "
            "Every index allocated before this instant now reads as "
            "'not currently allocated'.",
            getattr(self, "pool_name", "?"),
            id(self),
            int(getattr(self, "size", -1)),
            self._clear_epoch,
        )
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)
        # Per-slot flag used to detect double-free.
        # slot_used[k] is true if slot k is allocated.
        self.slot_used = torch.zeros(self.size, dtype=torch.bool)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert need_size % self.page_size == 0, (
            "The requested size should be a multiple of the page size."
        )
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        assert not self.slot_used[select_index].any(), (
            f"Double-alloc detected: slots already allocated: "
            f"{select_index[self.slot_used[select_index]].tolist()}."
        )
        self.slot_used[select_index] = True

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.cpu()
        # DIAGNOSTIC ONLY (#905 window): say WHICH pool object and WHICH clear
        # epoch refused, before the assertion below fires. Same raise, same
        # control flow -- only the story is added.
        if not bool(self.slot_used[indices_cpu].all()):
            _stale = indices_cpu[~self.slot_used[indices_cpu]]
            logger.error(
                # #1034: this line said "about to raise" and no longer does --
                # the raise became the named refusal below. Left in place for
                # its pool/epoch/free_slots detail, corrected so the log does
                # not promise a crash that will not come.
                "#905 HOST-POOL DOUBLE-FREE detected (REFUSED by #1034, not "
                "raised): pool %r id=%d "
                "size=%d clear_epoch=%d | %d of %d index(es) are in range but "
                "not allocated, span [%d, %d] | free_slots=%d",
                getattr(self, "pool_name", "?"),
                id(self),
                int(getattr(self, "size", -1)),
                int(getattr(self, "_clear_epoch", 0)),
                int(_stale.numel()),
                int(indices_cpu.numel()),
                int(_stale.min()) if _stale.numel() else -1,
                int(_stale.max()) if _stale.numel() else -1,
                int(len(self.free_slots)),
            )
        # #1034: IDEMPOTENT-REFUSE THE ALREADY-FREE SUBSET, LOUDLY.
        #
        # THE CENSUS THAT FORCED THIS (§AQ1). Two give-back routes reach this
        # pool and they cannot see each other:
        #   A  append_host_mem_release -> host_mem_release_queue ->
        #      _drain_release, SEVEN on-path producers, and #989 already keys
        #      it by first-slot -> "module:lineno" precisely because it is
        #      many-producers-one-give-back.
        #   B  DIRECT free out of a tree node's `host_value`
        #      (full_component.py:180 via evict_component, and
        #      unified_radix_cache.py:3995) -- NO provenance at all.
        # A span returned through A whose node still carries `host_value` is
        # freeable a second time through B. That is the measured crash
        # (boot 0d3ea263c6, §AP7): a contiguous slot run from 0, reached via
        # _drain_prefetch_progress -> _prefetch_kvcache -> prefetch_from_storage
        # -> evict_host -> ... -> free.
        #
        # WHY REFUSE AND NOT RAISE. Returning a slot that is already free is a
        # NO-OP on this pool's state: the second giver-back has nothing to
        # give. Killing the instance for a no-op costs every in-flight request
        # and, as measured, takes the whole PP group down behind it with gloo
        # peer-closed cascades. The allocated remainder is still freed, so no
        # slot leaks.
        #
        # AND IT IS NOT A SWALLOW. The refusal is counted and named on every
        # occurrence, with the provenance table #989 already maintains, so the
        # OTHER producer is identified instead of guessed at -- which is the
        # one thing the static census could not answer. A silent
        # `try/except` here would convert a loud stop into exactly the
        # wrong-answer-with-no-crash class this campaign exists to remove;
        # this keeps the stop's information and drops only the kill.
        _live_mask = self.slot_used[indices_cpu]
        if not bool(_live_mask.all()):
            _stale = indices_cpu[~_live_mask]
            self._double_free_refused = getattr(self, "_double_free_refused", 0) + 1
            _prov = getattr(self, "_free_provenance", None)
            _who = "no earlier free recorded for that slot"
            if isinstance(_prov, dict) and _stale.numel():
                _hit = _prov.get(int(_stale[0].item()))
                if _hit:
                    _who = f"slot {int(_stale[0].item())} was ALREADY freed by {_hit}"
            logger.error(
                "#1034 HOST-POOL DOUBLE-FREE REFUSED (%d so far): pool %r "
                "id=%d clear_epoch=%d | %d of %d index(es) are in range but "
                "ALREADY FREE, span [%d, %d] | %s | the allocated remainder "
                "IS freed and no slot leaks; this give-back had nothing to "
                "give. Two routes reach this pool (queue with provenance, "
                "direct evict without) -- see ANALYSE-969 §AQ1.",
                self._double_free_refused,
                getattr(self, "pool_name", "?"),
                id(self),
                int(getattr(self, "_clear_epoch", 0)),
                int(_stale.numel()),
                int(indices_cpu.numel()),
                int(_stale.min()),
                int(_stale.max()),
                _who,
            )
            indices_cpu = indices_cpu[_live_mask]
            if indices_cpu.numel() == 0:
                return 0
        self.slot_used[indices_cpu] = False
        self.free_slots = torch.cat([self.free_slots, indices_cpu])
        # #1034: PROVENANCE FOR *EVERY* ROUTE, RECORDED HERE RATHER THAN AT
        # THE CALL SITES. #989 stamps only the queue route, which is why the
        # census could not name route B's freer. Taken inside `free` itself,
        # so a route added later is covered without being taught to -- the
        # failure mode #989's own note describes ("a future fourth kind").
        # First slot only and capped: this is a breadcrumb for the refusal
        # above, not a ledger.
        try:
            _p = getattr(self, "_free_provenance", None)
            if _p is None:
                _p = self._free_provenance = {}
            if len(_p) < 65536 and indices_cpu.numel():
                import sys as _sys

                # Walk PAST this module's own frames: `free` is wrapped by
                # `@synchronized`, so frame 1 is the wrapper and naming it
                # would make every breadcrumb read "base.py:101" -- caught in
                # the desk smoke, and exactly the useless-provenance shape
                # this is meant to end.
                _f = _sys._getframe(1)
                _here = __name__
                _hops = 0
                while (
                    _f is not None
                    and _f.f_globals.get("__name__") == _here
                    and _hops < 8
                ):
                    _f = _f.f_back
                    _hops += 1
                if _f is not None:
                    _p[int(indices_cpu[0].item())] = (
                        f"{_f.f_globals.get('__name__', '?')}:{_f.f_lineno}"
                    )
        except Exception:  # noqa: BLE001 - a breadcrumb may never break a free
            pass
        return int(indices_cpu.numel())
