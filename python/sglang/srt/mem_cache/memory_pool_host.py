from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

import numpy as np
import torch

from sglang.jit_kernel.hicache import (
    can_use_write_back_jit_kernel,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf as jit_transfer_hicache_all_layer_mla_staged_lf_pf,
)
from sglang.jit_kernel.hisparse import transfer_cache_dsv4_mla
from sglang.srt.mem_cache.pinned_host_budget import check_and_register_pinned_post
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MambaPool,
)
from sglang.srt.utils import is_cuda, is_hip, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_mps = is_mps()
_has_sgl_kvcacheio = False
if _is_cuda or _is_hip:
    # Turing (sm75) / gfx900: no sgl-kernel code for these archs. These are the
    # HiCache host<->device KV transfer kernels; a rank without them simply
    # cannot use HiCache, but must still be able to start a plain server, and
    # this module is on the ordinary import path.
    try:
        from sgl_kernel.kvcacheio import (
            transfer_kv_all_layer_direct_lf_pf,
            transfer_kv_all_layer_mla,
            transfer_kv_all_layer_mla_lf_pf,
            transfer_kv_direct,
            transfer_kv_per_layer_direct_pf_lf,
            transfer_kv_per_layer_mla,
            transfer_kv_per_layer_mla_pf_lf,
        )

        _has_sgl_kvcacheio = True
    except ImportError:
        transfer_kv_all_layer_direct_lf_pf = None
        transfer_kv_all_layer_mla = None
        transfer_kv_all_layer_mla_lf_pf = None
        transfer_kv_direct = None
        transfer_kv_per_layer_direct_pf_lf = None
        transfer_kv_per_layer_mla = None
        transfer_kv_per_layer_mla_pf_lf = None

logger = logging.getLogger(__name__)


def _bind_phase_tag(pool) -> None:
    """#760: record which flip phase a host pool's device binding was taken in.

    Host pools capture ``device_buffers`` (and derive ``layer_num`` from it) ONCE
    at construction, and nothing rebinds them. Under ``--enable-phase-flip`` the
    flip snapshots the outgoing layout's weights and FREES THEIR DEVICE STORAGES,
    so after a flip those pointers name memory the allocator has taken back. The
    host write path hands them straight to ``cudaMemcpyBatchAsync`` as src_ptrs.

    That killed the ARM I harvest boot (2026-08-18 07:48:42Z): SIGSEGV in
    ``transfer_kv_all_layer_direct_lf_pf``, 100 GiB host memory free, ``phase=tp``
    on all three ranks and NO flip in progress -- a stable TP phase writing
    through a binding taken in PP. The layer counts differ too (7/5/4 per PP stage
    against 16 in TP), so the extent is wrong as well as the pointers.
    """
    pool._bound_tp_phase = None
    pool._stale_binding_refusals = 0
    why = "unknown"
    try:
        from sglang.srt.runtime_context import get_server_args

        if not getattr(get_server_args(), "enable_phase_flip", False):
            why = "phase flip disabled"
        else:
            from sglang.srt.distributed.parallel_state import (
                phase_flip_tp_routing_active,
            )

            pool._bound_tp_phase = bool(phase_flip_tp_routing_active())
    except Exception as exc:  # noqa: BLE001 - never break pool construction
        pool._bound_tp_phase = None
        why = f"{type(exc).__name__}: {exc}"[:120]
    # SAY WHETHER THIS GUARD IS ARMED. On 2026-08-18 the ARM I respin segfaulted
    # on exactly the write this protects and emitted ZERO refusals -- and I could
    # not tell whether the binding was genuinely fine or the guard was simply
    # inert, because a None tag disables the phase check SILENTLY. That is the
    # #742 silently-inert-flag class, which I had just spent the morning
    # instrumenting in other people's code. A guard that cannot be seen arming is
    # not evidence of anything.
    logger.info(
        "HICACHE-BINDING-TAG pool=%r armed=%s bound_phase=%s layers=%d%s",
        getattr(pool, "pool_name", "?"),
        pool._bound_tp_phase is not None,
        "tp"
        if pool._bound_tp_phase
        else ("pp" if pool._bound_tp_phase is False else "-"),
        getattr(pool, "layer_num", -1),
        "" if pool._bound_tp_phase is not None else f" (not armed: {why})",
    )


def _host_binding_is_stale(pool) -> bool:
    """Would writing through this binding hand freed device memory to CUDA?

    Two independent tells, either sufficient:

    1. THE EXTENT MOVED -- ``len(device_buffers)`` no longer matches the
       ``layer_num`` derived from it at construction. This is the #719 shape
       mismatch and it makes the copy walk past the end of the vector even where
       the pointers happen to survive.
    2. THE PHASE MOVED -- the binding was taken in one layout and
       ``phase_flip_tp_routing_active()`` (the same authority the model uses to
       route a batch, so it cannot drift from reality) now reports the other.

    Refusing costs a cache miss, which this tier is built to tolerate: an
    unwritten page is simply re-read from the device. Not refusing cost a SIGSEGV.
    That asymmetry is the whole argument, and it is why this is a refusal rather
    than an attempt to rebind a live pool mid-flight.

    Logged once per pool then counted -- a torn binding is a steady state, not an
    event, so it would otherwise print on every page of every write.
    Safe on any pool: without a recorded tag it returns False and changes nothing.
    """
    bufs = getattr(pool, "device_buffers", None) or ()
    layer_num = getattr(pool, "layer_num", -1)
    name = getattr(pool, "pool_name", "?")
    if layer_num >= 0 and len(bufs) != layer_num:
        pool._stale_binding_refusals = getattr(pool, "_stale_binding_refusals", 0) + 1
        if pool._stale_binding_refusals == 1:
            logger.error(
                "HICACHE-WRITE REFUSED (#760): host pool %r was bound to %d device "
                "layer buffer(s) but now sees %d. Writing would run "
                "cudaMemcpyBatchAsync past the end of the vector. Refusing the host "
                "write -- the page stays a device-tier hit.",
                name,
                layer_num,
                len(bufs),
            )
        return True
    bound = getattr(pool, "_bound_tp_phase", None)
    if bound is None:
        return False
    try:
        from sglang.srt.distributed.parallel_state import phase_flip_tp_routing_active

        active = bool(phase_flip_tp_routing_active())
    except Exception:  # noqa: BLE001 - if we cannot tell, do not block IO
        return False
    if active == bound:
        return False
    pool._stale_binding_refusals = getattr(pool, "_stale_binding_refusals", 0) + 1
    if pool._stale_binding_refusals == 1:
        logger.error(
            "HICACHE-WRITE REFUSED (#760): host pool %r bound its device buffers in "
            "phase=%s and the active phase is now %s. The flip frees the outgoing "
            "layout's device storages, so these src_ptrs name memory the allocator "
            "has taken back -- this is the write that SIGSEGV'd in "
            "transfer_kv_all_layer_direct_lf_pf on 2026-08-18. Refusing the host "
            "write; the page stays a device-tier hit until the pool is rebound.",
            name,
            "tp" if bound else "pp",
            "tp" if active else "pp",
        )
    return True


def _split_host_indices_by_binding(pool, indices, op: str):
    """Split host indices into those this pool can still describe, and strays.

    THE SIBLING AXIS OF `_host_binding_is_stale`. That guard protects the
    POINTER axis (device_buffers captured at construction). This one protects
    the INDEX axis, which has the same root and had no guard at all.

    `HostPoolGroup.anchor_entry` is fixed at construction (:__init__), so the
    anchor never moves inside one group -- the GROUP is rebuilt onto a narrower
    tier at a phase rebind. In-flight prefetch state does not move with it:
    `unified_radix_cache.check_prefetch_progress` holds `host_indices` minted
    against the previous, wider tier and frees them after the rebind. The
    indices are then applied to a pool whose `slot_used` is shorter, and
    `pool_host/base.py:free` indexes past its end.

    Measured on the W38 acceptance boot (2026-08-26 12:54:28Z, all three ranks):
    ``IndexError: index 76997 is out of bounds for dimension 0 with size 30518``.

    This is the class `_entry_for_transfer` (:1735) names for #718/#847 -- "a
    future producer that bypasses the resolver is LOUD rather than wrong". 322f33159a
    fixed the TRANSFER path and did not sweep the index-taking accessors one
    level up, so `free()` remained such a producer.

    DROPPING A STRAY IS CORRECT, NOT A PAPER-OVER: a stray names a slot in a
    tier that has already been torn down wholesale, so there is no slot left to
    return and nothing leaks. The alternative -- indexing anyway -- is the crash
    above. Refusing costs a cache miss, which this tier is built to tolerate.
    Logged once per pool then counted: a retired tier is a steady state, not an
    event.
    """
    size = int(getattr(pool, "size", -1))
    if size < 0 or indices is None or len(indices) == 0:
        return indices, 0
    stray_mask = (indices < 0) | (indices >= size)
    n_stray = int(stray_mask.sum())
    if n_stray == 0:
        return indices, 0
    pool._stale_index_refusals = getattr(pool, "_stale_index_refusals", 0) + 1
    if pool._stale_index_refusals == 1:
        logger.error(
            "HICACHE-INDEX REFUSED (#718 class, index axis): %s on host pool %r "
            "was handed %d index(es) outside [0, %d) -- highest %d. These were "
            "minted against a wider host tier that a phase rebind has since "
            "retired; the slots they name no longer exist, so they are dropped "
            "rather than applied. Applying them is the W38 IndexError at "
            "pool_host/base.py:344. Counting further occurrences silently.",
            op,
            getattr(pool, "pool_name", "?"),
            n_stray,
            size,
            int(indices.max()),
        )
    return indices[~stray_mask], n_stray


class StrayHostIndexError(IndexError):
    """A host index outlived the tier that minted it. See #718 class, index axis."""


def _refuse_stray_host_index(pool, index, op: str) -> None:
    """Loud form of `_split_host_indices_by_binding` for the data accessors.

    `free()` can drop a stray because a dropped free is a no-op on a tier that
    no longer exists. An accessor that RETURNS or WRITES a page cannot: dropping
    would shorten a result or skip a store, and the caller would proceed on data
    nobody wrote -- "a wrong ANSWER, with no assertion anywhere", which is the
    exact failure `_entry_for_transfer` was hardened against for #718/#847.

    Subclasses IndexError so existing handlers still catch it, but carries the
    tier story instead of `index 76997 is out of bounds for dimension 0`.
    """
    size = int(getattr(pool, "size", -1))
    if size < 0 or index is None:
        return
    if torch.is_tensor(index):
        if index.numel() == 0:
            return
        lo, hi = int(index.min()), int(index.max())
    else:
        lo = hi = int(index)
    if lo >= 0 and hi < size:
        return
    raise StrayHostIndexError(
        f"HICACHE-INDEX REFUSED (#718 class, index axis): {op} on host pool "
        f"{getattr(pool, 'pool_name', '?')!r} was handed index range "
        f"[{lo}, {hi}] outside [0, {size}). These indices were minted against a "
        f"wider host tier that a phase rebind has since retired. Unlike free(), "
        f"this call returns or writes a page, so it cannot drop them without "
        f"producing a wrong answer with no crash -- refusing by name instead."
    )


from sglang.srt.mem_cache.pool_host import HostKVCache
from sglang.srt.mem_cache.pool_host.base import (
    _WRITE_BACK_STAGING_PAGE_CHUNK,
    HICACHE_HOST_MEMORY_RESERVE_BYTES,
    sync_fixed_hicache_size,
    synchronized,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin
from sglang.srt.mem_cache.pinned_host_budget import (  # noqa: E402
    revert_pinned_posts_on_failure,
)


class MambaPoolHost(HostKVCache):
    @revert_pinned_posts_on_failure  # #729
    def __init__(
        self,
        device_pool: MambaPool,
        host_to_device_ratio: float,
        host_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        layout: str = "layer_first",
    ):
        self.device_pool = device_pool
        self.page_size = 1

        # The restriction here is the WRITE-BACK STAGING kernel, and that is the
        # whole of it. `_init_write_back_staging_buffers` returns immediately
        # unless the layout is exactly 'page_first', so 'page_first' is the one
        # value that would arm the staging path this pool cannot drive.
        #
        # 'layer_first' IS SUPPORTED and was already implemented on every method
        # of this class -- init_kv_buffer builds (num_layers, size, *shape)
        # buffers, _iter_page_tensors, load_to_device_per_layer and
        # backup_from_device_all_layer each carry a layer-first branch, and that
        # branch dispatches to `transfer_kv_direct`, a straight per-layer copy.
        # The blanket 'page_first_direct only' refusal predated those branches
        # and locked the pool onto the ONE route that is known broken: under
        # 'direct', page_first_direct sends this pool through
        # transfer_kv_all_layer_direct_lf_pf -> cudaMemcpyBatchAsync, the same
        # symbols that took three SIGSEGVs on 2026-08-18 (#760). Leaving the
        # refusal in place would mean a hybrid-mamba model could not have a
        # hicache host tier at all once that kernel is gated in server_args.
        #
        # page_head is not offered: its transfers reach the lf_ph kernel that
        # #441a refuses on its own evidence.
        if layout not in ("page_first_direct", "layer_first"):
            raise ValueError(
                f"MambaPoolHost supports layout='page_first_direct' or "
                f"'layer_first', got '{layout}'. 'page_first' would arm the "
                f"write-back staging kernel, which this pool cannot drive."
            )

        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.num_mamba_layers = device_pool.num_mamba_layers

        self.conv_state_shapes = [
            conv_state.shape[2:] for conv_state in device_pool.mamba_cache.conv
        ]
        self.temporal_state_shape = device_pool.mamba_cache.temporal.shape[2:]
        self.temporal_state_elem_size = int(np.prod(self.temporal_state_shape))
        self.conv_state_elem_sizes = [
            int(np.prod(conv_shape)) for conv_shape in self.conv_state_shapes
        ]
        self.conv_dtype = device_pool.mamba_cache.conv[0].dtype
        self.temporal_dtype = device_pool.mamba_cache.temporal.dtype
        self.dtype = self.conv_dtype
        self.size_per_token = self.get_size_per_token()

        if host_size > 0:
            self.size = sync_fixed_hicache_size(
                int(host_size * 1e9 // self.size_per_token), host_size
            )
        else:
            self.size = int(device_pool.size * host_to_device_ratio)

        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size

        if self.size <= device_pool.size:
            logger.warning(
                "HiCache host KV pool (%d tokens) is smaller than the device pool (%d tokens);"
                "L2 cache effectiveness is reduced."
                "Consider increasing --hicache-ratio (or --hicache-size) for higher L2 cache hit rate.",
                self.size,
                device_pool.size,
            )

        requested_bytes = self.size * self.size_per_token
        check_and_register_pinned_post(
            name="HiCache Mamba host pool",
            flag="--hicache-size / --hicache-ratio",
            requested_bytes=requested_bytes,
            reserve_bytes=HICACHE_HOST_MEMORY_RESERVE_BYTES,
        )
        logger.info(
            "Allocating %.2f GB host memory for hierarchical Mamba cache (layout=%s).",
            requested_bytes / 1e9,
            self.layout,
        )

        self.temporal_device_ptrs = torch.tensor(
            [
                device_pool.mamba_cache.temporal[i].data_ptr()
                for i in range(self.num_mamba_layers)
            ],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.conv_device_ptrs = [
            torch.tensor(
                [conv_state[i].data_ptr() for i in range(self.num_mamba_layers)],
                dtype=torch.uint64,
                device=self.device_pool.device,
            )
            for conv_state in device_pool.mamba_cache.conv
        ]

        self.init_kv_buffer()
        self._init_write_back_staging_buffers()
        self.lock = threading.RLock()
        self.clear()

    def init_kv_buffer(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]

        if self.layout in ["page_first", "page_first_direct"]:
            # page-first: (page_num, num_layers, 1, *shape) — per-page data is contiguous
            temporal_dims = (
                self.size,
                self.num_mamba_layers,
                1,
            ) + self.temporal_state_shape
            self.temporal_buffer = alloc_func(
                temporal_dims,
                dtype=self.temporal_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.conv_buffer = []
            for conv_shape in self.conv_state_shapes:
                conv_dims = (self.size, self.num_mamba_layers, 1) + conv_shape
                self.conv_buffer.append(
                    alloc_func(
                        conv_dims,
                        dtype=self.conv_dtype,
                        device=self.device,
                        pin_memory=self.pin_memory,
                        allocator=self.allocator,
                    )
                )
        else:
            # layer-first: (num_layers, size, *shape)
            temporal_dims = (
                self.num_mamba_layers,
                self.size,
            ) + self.temporal_state_shape
            self.temporal_buffer = alloc_func(
                temporal_dims,
                dtype=self.temporal_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.conv_buffer = []
            for conv_shape in self.conv_state_shapes:
                conv_dims = (self.num_mamba_layers, self.size) + conv_shape
                self.conv_buffer.append(
                    alloc_func(
                        conv_dims,
                        dtype=self.conv_dtype,
                        device=self.device,
                        pin_memory=self.pin_memory,
                        allocator=self.allocator,
                    )
                )

    def _init_write_back_staging_buffers(self):
        self.temporal_staging_buffer = None
        self.conv_staging_buffers = [None] * len(self.conv_buffer)
        self.can_use_write_back_jit = False
        self._temporal_can_use_jit = False
        self._conv_can_use_jit = [False] * len(self.conv_buffer)
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        self._temporal_can_use_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self._item_size_per_index(self.temporal_buffer[0]),
        )
        self._conv_can_use_jit = [
            _is_cuda
            and can_use_write_back_jit_kernel(
                element_size=self._item_size_per_index(buf[0]),
            )
            for buf in self.conv_buffer
        ]
        self.can_use_write_back_jit = self._temporal_can_use_jit and all(
            self._conv_can_use_jit
        )
        self.staging_page_capacity = min(self.page_num, _WRITE_BACK_STAGING_PAGE_CHUNK)
        self.staging_token_capacity = self.staging_page_capacity * self.page_size
        self.temporal_staging_buffer = torch.empty(
            (
                self.staging_token_capacity,
                self.num_mamba_layers,
                1,
                *self.temporal_state_shape,
            ),
            dtype=self.temporal_dtype,
            device=self.device_pool.device,
        )
        self.conv_staging_buffers = [
            torch.empty(
                (
                    self.staging_token_capacity,
                    self.num_mamba_layers,
                    1,
                    *conv_shape,
                ),
                dtype=self.conv_dtype,
                device=self.device_pool.device,
            )
            for conv_shape in self.conv_state_shapes
        ]

    def get_hybrid_pool_buffer(self):
        # Expose all mamba host tensors that need Mooncake buffer registration.
        return [self.temporal_buffer, *self.conv_buffer]

    def _iter_page_tensors(self, index: int):
        if self.layout in ["page_first", "page_first_direct"]:
            yield self.temporal_buffer[index]
            for conv_buf in self.conv_buffer:
                yield conv_buf[index]
        else:
            yield self.temporal_buffer[:, index : index + self.page_size]
            for conv_buf in self.conv_buffer:
                yield conv_buf[:, index : index + self.page_size]

    @staticmethod
    def _flatten_tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().view(torch.uint8).reshape(-1)

    @synchronized
    def clear(self):
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

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
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)

    def get_size_per_token(self):
        conv_total_size = sum(
            conv_elem_size * self.conv_dtype.itemsize
            for conv_elem_size in self.conv_state_elem_sizes
        )
        temporal_size = self.temporal_state_elem_size * self.temporal_dtype.itemsize
        return (conv_total_size + temporal_size) * self.num_mamba_layers

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    @staticmethod
    def _item_size_per_index(tensor: torch.Tensor) -> int:
        if tensor.shape[0] == 0:
            return 0
        return int(tensor[0].numel() * tensor.element_size())

    @staticmethod
    def _copy_tensor(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            # TODO: Rename the interface for clarity.
            # Here, transfer_kv_per_layer_mla is reused to transfer the Mamba state.
            # This has nothing to do with MLA; it's only reused because this interface happens to transfer a single Pool.
            transfer_kv_per_layer_mla(
                src=src,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                item_size=MambaPoolHost._item_size_per_index(src),
            )
        elif io_backend == "direct":
            transfer_kv_direct(
                src_layers=[src],
                dst_layers=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    @staticmethod
    def _copy_tensor_pf_lf(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        layer_id: int,
        num_layers: int,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            item_size = MambaPoolHost._item_size_per_index(dst)
            transfer_kv_per_layer_mla_pf_lf(
                src=src,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                layer_id=layer_id,
                item_size=item_size,
                src_layout_dim=item_size * num_layers,
            )
        elif io_backend == "direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[src],
                dst_ptrs=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                layer_id=layer_id,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    @staticmethod
    def _copy_tensor_all_layers_lf_pf(
        src_layers: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        num_layers: int,
        io_backend: str,
        src_ptrs: torch.Tensor,
        staging: Optional[torch.Tensor] = None,
        can_use_jit: bool = False,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            item_size = MambaPoolHost._item_size_per_index(src_layers[0])
            if can_use_jit:
                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                    ptr_src=src_ptrs,
                    src_indices=src_indices,
                    dst_indices=dst_indices,
                    staging=staging,
                    dst=dst,
                    page_size=1,
                    element_size=item_size,
                )
            else:
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=src_ptrs,
                    dst=dst,
                    src_indices=src_indices,
                    dst_indices=dst_indices,
                    item_size=item_size,
                    dst_layout_dim=item_size * num_layers,
                    num_layers=num_layers,
                )
        elif io_backend == "direct":
            src_ptrs = [src_layers[i] for i in range(num_layers)]
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=src_ptrs,
                dst_ptrs=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend="kernel",
    ):
        if io_backend != "direct":
            raise ValueError(
                f"MambaPoolHost only supports io_backend='direct', got '{io_backend}'."
            )
        if self.layout in ["page_first", "page_first_direct"]:
            self._copy_tensor_pf_lf(
                src=self.temporal_buffer,
                dst=device_pool.mamba_cache.temporal[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=layer_id,
                num_layers=self.num_mamba_layers,
                io_backend=io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor_pf_lf(
                    src=self.conv_buffer[conv_idx],
                    dst=device_pool.mamba_cache.conv[conv_idx][layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    num_layers=self.num_mamba_layers,
                    io_backend=io_backend,
                )
        else:
            self._copy_tensor(
                self.temporal_buffer[layer_id],
                device_pool.mamba_cache.temporal[layer_id],
                host_indices,
                device_indices,
                io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor(
                    self.conv_buffer[conv_idx][layer_id],
                    device_pool.mamba_cache.conv[conv_idx][layer_id],
                    host_indices,
                    device_indices,
                    io_backend,
                )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ):
        if io_backend != "direct":
            raise ValueError(
                f"MambaPoolHost only supports io_backend='direct', got '{io_backend}'."
            )
        if self.layout in ["page_first", "page_first_direct"]:
            self._copy_tensor_all_layers_lf_pf(
                src_layers=device_pool.mamba_cache.temporal,
                dst=self.temporal_buffer,
                src_indices=device_indices,
                dst_indices=host_indices,
                num_layers=self.num_mamba_layers,
                io_backend=io_backend,
                staging=self.temporal_staging_buffer,
                can_use_jit=self._temporal_can_use_jit,
                src_ptrs=self.temporal_device_ptrs,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor_all_layers_lf_pf(
                    src_layers=device_pool.mamba_cache.conv[conv_idx],
                    dst=self.conv_buffer[conv_idx],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    num_layers=self.num_mamba_layers,
                    io_backend=io_backend,
                    staging=self.conv_staging_buffers[conv_idx],
                    can_use_jit=self._conv_can_use_jit[conv_idx],
                    src_ptrs=self.conv_device_ptrs[conv_idx],
                )
        else:
            for layer_id in range(self.num_mamba_layers):
                self._copy_tensor(
                    device_pool.mamba_cache.temporal[layer_id],
                    self.temporal_buffer[layer_id],
                    device_indices,
                    host_indices,
                    io_backend,
                )
                for conv_idx in range(len(self.conv_state_shapes)):
                    self._copy_tensor(
                        device_pool.mamba_cache.conv[conv_idx][layer_id],
                        self.conv_buffer[conv_idx][layer_id],
                        device_indices,
                        host_indices,
                        io_backend,
                    )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        data_page = torch.cat(
            [
                self._flatten_tensor_bytes(tensor)
                for tensor in self._iter_page_tensors(index)
            ]
        )
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.page_size * self.size_per_token,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def set_from_flat_data_page(
        self,
        index: int,
        data_page: torch.Tensor,
    ) -> None:
        flat_bytes = data_page.contiguous().view(torch.uint8).reshape(-1)
        start = 0
        for tensor in self._iter_page_tensors(index):
            num_bytes = tensor.numel() * tensor.element_size()
            tensor_bytes = flat_bytes[start : start + num_bytes]
            start += num_bytes
            restored = tensor_bytes.view(dtype=tensor.dtype).reshape(tensor.shape)
            tensor.copy_(restored)

    def get_page_buffer_meta(self, indices):
        """Meta data for zero-copy storage I/O.

        Only page-first layouts are supported for mamba storage zero-copy because
        each page slot in temporal/conv buffers is directly addressable.
        """
        assert len(indices) % self.page_size == 0
        if self.layout not in ["page_first", "page_first_direct"]:
            raise ValueError(
                f"Mamba storage zero-copy requires page_first layout, got {self.layout}"
            )
        indices = indices.tolist()
        ptr_list = []
        element_size_list = []

        # Compute base pointers once; each page pointer is offset from these bases.
        temporal_base_ptr = self.temporal_buffer.data_ptr()
        conv_base_ptrs = [buf.data_ptr() for buf in self.conv_buffer]
        # Component sizes are constant across pages, so precompute once as well.
        temporal_element_size = (
            self.page_size
            * self.num_mamba_layers
            * self.temporal_dtype.itemsize
            * self.temporal_state_elem_size
        )
        conv_element_sizes = [
            (
                self.page_size
                * self.num_mamba_layers
                * self.conv_dtype.itemsize
                * self.conv_state_elem_sizes[i]
            )
            for i in range(len(self.conv_state_shapes))
        ]

        for i in range(0, len(indices), self.page_size):
            # Emit component pointers in stable order:
            # temporal first, then conv_0..conv_n for this page.
            temporal_ptr = (
                temporal_base_ptr
                + indices[i]
                * self.num_mamba_layers
                * self.temporal_state_elem_size
                * self.temporal_dtype.itemsize
            )
            ptr_list.append(temporal_ptr)
            element_size_list.append(temporal_element_size)
            for j in range(len(self.conv_buffer)):
                conv_ptr = (
                    conv_base_ptrs[j]
                    + indices[i]
                    * self.num_mamba_layers
                    * self.conv_state_elem_sizes[j]
                    * self.conv_dtype.itemsize
                )
                ptr_list.append(conv_ptr)
                element_size_list.append(conv_element_sizes[j])
        return ptr_list, element_size_list


# ---- V4 Compressed KV Host Pools ----


class LogicalHostPool:
    """Pure-logical anchor pool for V4 HiCache.

    The pool manages page-aligned token slots but holds no KV tensor. V4
    compressed side pools use these logical FULL indices as stable page anchors.
    """

    def __init__(self, size: int, page_size: int, layout: str = "layer_first"):
        if size % page_size != 0:
            raise ValueError(
                "LogicalHostPool size must be page-aligned, "
                f"got size={size}, page_size={page_size}"
            )
        self.size = size
        self.page_size = page_size
        self.device = "cpu"
        self.layout = layout
        self.dtype = torch.uint8
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.kv_buffer = None
        self.size_per_token = 0
        self.allocator = None
        self.can_use_write_back_jit = True
        self.lock = threading.RLock()
        self.clear()

    @synchronized
    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool allocation must be page-aligned, "
                f"got need_size={need_size}, page_size={self.page_size}"
            )
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        if len(indices) % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool free must be page-aligned, "
                f"got len(indices)={len(indices)}, page_size={self.page_size}"
            )
        self.free_slots = torch.cat(
            [self.free_slots, indices.to(dtype=torch.int64, device="cpu").flatten()]
        )
        return len(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        pass

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        pass

    def get_data_page(self, index, flat=True):
        return torch.empty(0, dtype=torch.uint8)

    def get_dummy_flat_data_page(self):
        return torch.empty(0, dtype=torch.uint8)

    def set_from_flat_data_page(self, index, data_page):
        pass

    def get_page_buffer_meta(self, indices):
        return None

    def get_ksize_per_token(self):
        return 0


class DeepSeekV4PagedHostPool(HiSparseHostPoolMixin, HostKVCache):
    """Host mirror for a DeepSeek V4 paged KV/indexer sub-pool."""

    @revert_pinned_posts_on_failure  # #729
    def __init__(
        self,
        pool_name: str,
        device_buffers: list[torch.Tensor],
        item_bytes: int,
        num_host_pages: int,
        slot_page_size: int,
        layout: str = "layer_first",
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
    ):
        self.pool_name = pool_name
        self.layer_num = len(device_buffers)
        self.item_bytes = item_bytes
        self.num_host_pages = num_host_pages
        self.slot_page_size = slot_page_size
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.page_size = slot_page_size
        self.size = num_host_pages * slot_page_size
        self.layout = layout
        self.size_per_token = item_bytes
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.device_buffers = device_buffers
        _bind_phase_tag(self)
        self.gpu_device = device_buffers[0].device if device_buffers else device

        requested_bytes = self.layer_num * num_host_pages * self.item_bytes
        check_and_register_pinned_post(
            name=f"V4 paged host pool {pool_name}",
            flag="--hicache-size / --hicache-ratio",
            requested_bytes=requested_bytes,
            reserve_bytes=HICACHE_HOST_MEMORY_RESERVE_BYTES,
        )

        alloc_func = ALLOC_MEMORY_FUNCS[self.gpu_device]
        self.data_refs = []
        if self.layout == "layer_first":
            self.kv_buffer = [
                alloc_func(
                    (num_host_pages, self.item_bytes),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        elif self.layout == "page_first":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, self.item_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, 1, self.item_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

        logger.info(
            "Allocating %.2f GB host memory for V4 paged pool '%s' "
            "(layers=%d, pages=%d, item_bytes=%d, layout=%s).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.item_bytes,
            self.layout,
        )

        self.device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_buffers],
            dtype=torch.uint64,
            device=self.gpu_device,
        )
        self.data_ptrs = (
            torch.tensor(
                [x.data_ptr() for x in self.data_refs],
                dtype=torch.uint64,
                device=self.gpu_device,
            )
            if self.data_refs
            else None
        )
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self._init_write_back_staging_buffers()
        self.clear()

    def _init_write_back_staging_buffers(self):
        self.staging_buffer = None
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self.item_bytes * self.dtype.itemsize,
        )
        staging_page_capacity = min(self.num_host_pages, _WRITE_BACK_STAGING_PAGE_CHUNK)
        self.staging_buffer = torch.empty(
            (staging_page_capacity, self.layer_num, self.item_bytes),
            dtype=self.dtype,
            device=self.gpu_device,
        )

    def get_contiguous_buf_infos(self):
        """Return per-layer page-row buffers for PD direct-to-host transfer."""
        data_ptrs = [int(self.data_ptrs[i].item()) for i in range(self.layer_num)]
        data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.item_bytes * self.dtype.itemsize] * self.layer_num
        return data_ptrs, data_lens, item_lens

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return indices.reshape(-1, self.slot_page_size)[:, 0] // self.slot_page_size

    def _has_transfer_indices(
        self, host_indices: torch.Tensor | None, device_indices: torch.Tensor | None
    ) -> bool:
        if host_indices is None or device_indices is None:
            return False
        if host_indices.numel() != device_indices.numel():
            raise ValueError(
                f"{self.pool_name} transfer index size mismatch: "
                f"host={host_indices.numel()}, device={device_indices.numel()}"
            )
        return host_indices.numel() > 0

    def get_size_per_token(self):
        return self.item_bytes

    def get_ksize_per_token(self):
        return self.item_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]

    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        need_size = (
            (need_size + self.slot_page_size - 1) // self.slot_page_size
        ) * self.slot_page_size
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat(
            [self.free_slots, indices.to(dtype=torch.int64, device="cpu").flatten()]
        )
        return len(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        if (
            host_indices.numel() % self.slot_page_size != 0
            or device_indices.numel() % self.slot_page_size != 0
        ):
            # Whole C4 pages can use the normal HiCache page-row copy below.
            # Token-granular DSV4 C4 copy needs this helper because a token is
            # not one contiguous byte range in the paged row:
            # [value0..value63][scale0..scale63].
            transfer_cache_dsv4_mla(
                src_ptrs=self.device_ptrs,
                dst_ptrs=self.data_ptrs,
                src_indices=device_indices.to(dtype=torch.int64),
                dst_indices=host_indices.to(dtype=torch.int64),
            )
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_all_layer_mla(
                src_layers=self.device_ptrs,
                dst_layers=self.data_ptrs,
                src_indices=device_rows,
                dst_indices=host_rows,
                item_size=self.item_bytes,
                num_layers=self.layer_num,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            if self.can_use_write_back_jit:
                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                    ptr_src=self.device_ptrs,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    staging=self.staging_buffer,
                    dst=self.kv_buffer,
                    page_size=1,
                    element_size=self.item_bytes,
                )
            else:
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.device_ptrs,
                    dst=self.kv_buffer,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    item_size=self.item_bytes,
                    dst_layout_dim=self.layer_num * self.item_bytes,
                    num_layers=self.layer_num,
                )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=self.device_buffers,
                dst_layers=self.data_refs,
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            if _host_binding_is_stale(self):
                return
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=self.device_buffers,
                dst_ptrs=[self.kv_buffer],
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        else:
            raise ValueError(
                f"Unsupported V4 paged host layout/backend: {self.layout}/{io_backend}"
            )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        if (
            host_indices.numel() % self.slot_page_size != 0
            or device_indices.numel() % self.slot_page_size != 0
        ):
            # Same DSV4 C4 layout issue as backup: this is token-granular
            # preload, so it cannot use the normal HiCache page-row copy.
            transfer_cache_dsv4_mla(
                src_ptrs=self.data_ptrs[layer_id : layer_id + 1],
                dst_ptrs=self.device_ptrs[layer_id : layer_id + 1],
                src_indices=host_indices.to(dtype=torch.int64),
                dst_indices=device_indices.to(dtype=torch.int64),
            )
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)

        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_per_layer_mla(
                src=self.data_refs[layer_id],
                dst=self.device_buffers[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                item_size=self.item_bytes,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            transfer_kv_per_layer_mla_pf_lf(
                src=self.kv_buffer,
                dst=self.device_buffers[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                item_size=self.item_bytes,
                src_layout_dim=self.layer_num * self.item_bytes,
            )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=[self.data_refs[layer_id]],
                dst_layers=[self.device_buffers[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.kv_buffer],
                dst_ptrs=[self.device_buffers[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                page_size=1,
            )
        else:
            raise ValueError(
                f"Unsupported V4 paged host layout/backend: {self.layout}/{io_backend}"
            )

    def get_data_page(self, index, flat=True):
        index = int(index) // self.slot_page_size
        if self.layout == "layer_first":
            data_page = torch.stack(
                [self.kv_buffer[i][index] for i in range(self.layer_num)]
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            data_page = self.kv_buffer[index]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (self.layer_num, self.item_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        index = int(index) // self.slot_page_size
        if self.layout == "layer_first":
            data = data_page.view(self.dtype).reshape(self.layer_num, self.item_bytes)
            for i in range(self.layer_num):
                self.kv_buffer[i][index].copy_(data[i])
        elif self.layout == "page_first":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(self.layer_num, self.item_bytes)
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(self.layer_num, 1, self.item_bytes)
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        if self.layout == "layer_first":
            for row in rows:
                page_index = int(row)
                for layer_id in range(self.layer_num):
                    ptr = (
                        self.kv_buffer[layer_id].data_ptr()
                        + page_index * self.item_bytes * self.dtype.itemsize
                    )
                    ptr_list.append(ptr)
            element_size = self.item_bytes * self.dtype.itemsize
            return ptr_list, [element_size] * len(ptr_list)
        if self.layout in ["page_first", "page_first_direct"]:
            page_bytes = self.layer_num * self.item_bytes * self.dtype.itemsize
            for row in rows:
                ptr_list.append(self.kv_buffer[int(row)].data_ptr())
            return ptr_list, [page_bytes] * len(ptr_list)
        raise ValueError(f"Unsupported layout: {self.layout}")


class DeepSeekV4StateHostPool(HostKVCache):
    """Host pool for V4 CompressStatePool page rows."""

    @revert_pinned_posts_on_failure  # #729
    def __init__(
        self,
        pool_name: str,
        state_pools: list,
        num_host_pages: int,
        swa_page_size: int,
        layout: str = "layer_first",
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
    ):
        if any(pool is None for pool in state_pools):
            raise ValueError(f"{pool_name} state_pools must not contain None")

        self.pool_name = pool_name
        self.state_pools = state_pools
        self.layer_num = len(state_pools)
        self.num_host_pages = num_host_pages
        self.swa_page_size = swa_page_size
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.page_size = swa_page_size
        self.size = num_host_pages * swa_page_size
        self.layout = layout
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.ring_size = 0
        self.state_page_bytes = 0
        self.device_page_views = []
        self.gpu_device = device
        self._init_device_page_views()
        self.size_per_token = self.state_page_bytes

        requested_bytes = self.layer_num * num_host_pages * self.state_page_bytes
        check_and_register_pinned_post(
            name=f"V4 state host pool {pool_name}",
            flag="--hicache-size / --hicache-ratio",
            requested_bytes=requested_bytes,
            reserve_bytes=HICACHE_HOST_MEMORY_RESERVE_BYTES,
        )

        alloc_func = ALLOC_MEMORY_FUNCS[self.gpu_device]
        self.data_refs = []
        if self.layout == "layer_first":
            self.kv_buffer = [
                alloc_func(
                    (num_host_pages, self.state_page_bytes),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        elif self.layout == "page_first":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, self.state_page_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, 1, self.state_page_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        logger.info(
            "Allocating %.2f GB host memory for V4 state pool '%s' "
            "(layers=%d, pages=%d, state_page_bytes=%d, layout=%s).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.state_page_bytes,
            self.layout,
        )
        self.device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_page_views],
            dtype=torch.uint64,
            device=self.gpu_device,
        )
        self.data_ptrs = (
            torch.tensor(
                [x.data_ptr() for x in self.data_refs],
                dtype=torch.uint64,
                device=self.gpu_device,
            )
            if self.data_refs
            else None
        )
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self._init_write_back_staging_buffers()

    def _init_device_page_views(self) -> None:
        expected_ring_size = None
        expected_state_page_bytes = None
        for pool in self.state_pools:
            state_tensor = pool.kv_score_buffer.kv_score
            if not state_tensor.is_contiguous():
                raise ValueError(f"{self.pool_name} state tensor must be contiguous")
            ring_size = pool.ring_size
            slot_bytes = state_tensor[0].nbytes
            state_page_bytes = ring_size * slot_bytes
            if expected_ring_size is None:
                expected_ring_size = ring_size
                expected_state_page_bytes = state_page_bytes
                self.gpu_device = state_tensor.device
            elif (
                expected_ring_size != ring_size
                or expected_state_page_bytes != state_page_bytes
            ):
                raise ValueError(
                    f"{self.pool_name} state pools must share ring size and slot bytes"
                )

            state_bytes = state_tensor.view(torch.uint8).reshape(
                state_tensor.shape[0], -1
            )
            usable_slots = (state_tensor.shape[0] // ring_size) * ring_size
            self.device_page_views.append(
                state_bytes[:usable_slots].reshape(-1, state_page_bytes)
            )

        self.ring_size = expected_ring_size or 0
        self.state_page_bytes = expected_state_page_bytes or 0

    def _init_write_back_staging_buffers(self):
        self.staging_buffer = None
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self.state_page_bytes * self.dtype.itemsize,
        )
        staging_page_capacity = min(self.num_host_pages, _WRITE_BACK_STAGING_PAGE_CHUNK)
        self.staging_buffer = torch.empty(
            (staging_page_capacity, self.layer_num, self.state_page_bytes),
            dtype=self.dtype,
            device=self.gpu_device,
        )

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() % self.swa_page_size != 0:
            raise ValueError(
                f"{self.pool_name} transfer indices must be SWA-page-aligned, "
                f"got numel={indices.numel()}, swa_page_size={self.swa_page_size}"
            )
        return indices.reshape(-1, self.swa_page_size)[:, 0] // self.swa_page_size

    def get_size_per_token(self):
        return self.state_page_bytes

    def get_ksize_per_token(self):
        return self.state_page_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]

    def clear(self):
        pass

    def available_size(self):
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no free list"
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if io_backend == "kernel" and self.layout == "layer_first":
            assert self.data_ptrs is not None
            transfer_kv_all_layer_mla(
                src_layers=self.device_ptrs,
                dst_layers=self.data_ptrs,
                src_indices=device_rows,
                dst_indices=host_rows,
                item_size=self.state_page_bytes,
                num_layers=self.layer_num,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            if self.can_use_write_back_jit:
                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                    ptr_src=self.device_ptrs,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    staging=self.staging_buffer,
                    dst=self.kv_buffer,
                    page_size=1,
                    element_size=self.state_page_bytes,
                )
            else:
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.device_ptrs,
                    dst=self.kv_buffer,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    item_size=self.state_page_bytes,
                    dst_layout_dim=self.layer_num * self.state_page_bytes,
                    num_layers=self.layer_num,
                )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=self.device_page_views,
                dst_layers=self.data_refs,
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=self.device_page_views,
                dst_ptrs=[self.kv_buffer],
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        else:
            raise ValueError(
                f"Unsupported V4 state host layout/backend: {self.layout}/{io_backend}"
            )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_per_layer_mla(
                src=self.data_refs[layer_id],
                dst=self.device_page_views[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                item_size=self.state_page_bytes,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            transfer_kv_per_layer_mla_pf_lf(
                src=self.kv_buffer,
                dst=self.device_page_views[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                item_size=self.state_page_bytes,
                src_layout_dim=self.layer_num * self.state_page_bytes,
            )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=[self.data_refs[layer_id]],
                dst_layers=[self.device_page_views[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.kv_buffer],
                dst_ptrs=[self.device_page_views[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                page_size=1,
            )
        else:
            raise ValueError(
                f"Unsupported V4 state host layout/backend: {self.layout}/{io_backend}"
            )

    def get_data_page(self, index, flat=True):
        index = int(index) // self.swa_page_size
        if self.layout == "layer_first":
            data_page = torch.stack(
                [self.kv_buffer[i][index] for i in range(self.layer_num)]
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            data_page = self.kv_buffer[index]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (self.layer_num, self.state_page_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        index = int(index) // self.swa_page_size
        if self.layout == "layer_first":
            data = data_page.view(self.dtype).reshape(
                self.layer_num, self.state_page_bytes
            )
            for i in range(self.layer_num):
                self.kv_buffer[i][index].copy_(data[i])
        elif self.layout == "page_first":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(
                    self.layer_num, self.state_page_bytes
                )
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(
                    self.layer_num, 1, self.state_page_bytes
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        if self.layout == "layer_first":
            for row in rows:
                page_index = int(row)
                for layer_id in range(self.layer_num):
                    ptr = (
                        self.kv_buffer[layer_id].data_ptr()
                        + page_index * self.state_page_bytes * self.dtype.itemsize
                    )
                    ptr_list.append(ptr)
            element_size = self.state_page_bytes * self.dtype.itemsize
            return ptr_list, [element_size] * len(ptr_list)
        if self.layout in ["page_first", "page_first_direct"]:
            page_bytes = self.layer_num * self.state_page_bytes * self.dtype.itemsize
            for row in rows:
                ptr_list.append(self.kv_buffer[int(row)].data_ptr())
            return ptr_list, [page_bytes] * len(ptr_list)
        raise ValueError(f"Unsupported layout: {self.layout}")


@dataclass
class PoolEntry:
    name: PoolName
    host_pool: Any
    device_pool: Any
    layer_mapper: Callable[[int], Optional[int]]
    is_primary_index_anchor: bool = False
    # Optional eviction callbacks for auto-alloc in HybridCacheController.
    # host_evict_fn(n): evict n slots from the host pool (used by write()).
    # device_evict_fn(n): evict n slots from the device pool (used by load()).
    host_evict_fn: Optional[Callable] = None
    device_evict_fn: Optional[Callable] = None
    # Optional alloc/free overrides for the device side, used by
    # _resolve_pool_transfers_allocation. Set when entry.device_pool is the
    # raw KV/state pool (layout) rather than an allocator (e.g. SWA/Mamba,
    # where alloc lives on a separate allocator object).
    # When None, fall back to entry.device_pool.alloc/free.
    device_alloc_fn: Optional[Callable] = None
    device_free_fn: Optional[Callable] = None


class HostPoolGroup:
    def __init__(self, entries: list[PoolEntry]):
        if not entries:
            raise ValueError("HostPoolGroup requires at least one pool entry.")
        self.entries = entries
        self.entry_map = {entry.name: entry for entry in entries}
        self.anchor_entry = next(
            (entry for entry in entries if entry.is_primary_index_anchor),
            entries[0],
        )

        self.layout = self.anchor_entry.host_pool.layout
        self.page_size = self.anchor_entry.host_pool.page_size
        self.device = self.anchor_entry.host_pool.device
        self.size = self.anchor_entry.host_pool.size
        self.can_use_write_back_jit = all(
            getattr(entry.host_pool, "can_use_write_back_jit", False)
            for entry in entries
        )

    @property
    def kv_buffer(self):
        return self.anchor_entry.host_pool.kv_buffer

    @property
    def size_per_token(self):
        return self.anchor_entry.host_pool.size_per_token

    @property
    def allocator(self):
        return self.anchor_entry.host_pool.allocator

    @property
    def dtype(self):
        return self.anchor_entry.host_pool.dtype

    @property
    def start_layer(self):
        return self.anchor_entry.host_pool.start_layer

    @property
    def end_layer(self):
        return self.anchor_entry.host_pool.end_layer

    @property
    def layer_num(self):
        """W34: the anchor's layer count, delegated like every neighbour here.

        THIS PROPERTY'S ABSENCE MADE THE #718 REBIND UNARMABLE ON THIS TREE,
        and it predates the rebind work. `hicache_phase_binding.check_shapes`
        compares `device_pool.layer_num` against `host_pool.layer_num` and
        refuses when either is None -- "refusing to rebind onto pools whose
        shapes cannot be compared". A plain host pool has `layer_num`; this
        COMPOSITE did not, although it already delegates `dtype`,
        `start_layer`, `end_layer`, `kv_buffer`, `size_per_token` and
        `allocator` to the same anchor. So on every boot whose host tier is a
        `HostPoolGroup` -- which is this fork's live shape -- the shape check
        could only ever read None and refuse, host pool present or not.

        Measured indirectly across W32/W33/W34: the device tier stayed
        disarmed for the whole TP phase (6, 6 and 3 `#718 hicache-phase-guard`
        warnings) and every read-through missed with `#cached-token: 0`.

        Delegating is the correct fix and not a widening: the anchor is what
        `start_layer`/`end_layer` already describe, so a group whose anchor
        covers N layers describes N layers. A group is exactly as comparable
        as its anchor, which is the property `check_shapes` needs.
        """
        return self.anchor_entry.host_pool.layer_num

    def get_ksize_per_token(self):
        return self.anchor_entry.host_pool.get_ksize_per_token()

    def get_pool(self, name: PoolName):
        return self.entry_map[name].host_pool

    def get_page_buffer_meta(self, indices):
        # #718 class, index axis. Unlike free(), this RETURNS DATA: a stray
        # cannot be dropped without silently shortening the caller's result,
        # which is the wrong-answer-with-no-crash outcome the sibling refusals
        # exist to prevent. So it is loud.
        _refuse_stray_host_index(
            self.anchor_entry.host_pool, indices, "get_page_buffer_meta"
        )
        return self.anchor_entry.host_pool.get_page_buffer_meta(indices)

    def clear(self) -> None:
        for entry in self.entries:
            entry.host_pool.clear()

    def available_size(self):
        return self.anchor_entry.host_pool.available_size()

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        return self.anchor_entry.host_pool.alloc(need_size)

    def free(self, indices: torch.Tensor) -> int:
        # #718 class, index axis: strays from a tier this rebind retired are
        # dropped by name (see _split_host_indices_by_binding). Freeing what
        # remains is the whole point -- an empty result is still a valid free.
        pool = self.anchor_entry.host_pool
        live, n_stray = _split_host_indices_by_binding(pool, indices, "free")
        if n_stray and len(live) == 0:
            return 0
        return pool.free(live)

    def get_data_page(self, index, flat: bool = True):
        # #718 class, index axis. Returns data -- loud, for get_page_buffer_meta's reason.
        _refuse_stray_host_index(self.anchor_entry.host_pool, index, "get_data_page")
        return self.anchor_entry.host_pool.get_data_page(index, flat)

    def get_dummy_flat_data_page(self):
        return self.anchor_entry.host_pool.get_dummy_flat_data_page()

    def set_from_flat_data_page(self, index: int, data_page) -> None:
        # #718 class, index axis. A stray here WRITES into a slot the retired
        # tier owned -- corruption of whatever now lives at that offset, or an
        # out-of-bounds store. Loud, never dropped.
        _refuse_stray_host_index(
            self.anchor_entry.host_pool, index, "set_from_flat_data_page"
        )
        return self.anchor_entry.host_pool.set_from_flat_data_page(index, data_page)

    def _entry_for_transfer(self, transfer, direction: str):
        """The entry this transfer names, or a refusal.

        #718/#847: THIS USED TO SKIP SILENTLY. `entry is None or
        transfer.host_indices is None -> continue` meant that after a phase
        rebind onto a narrower tier the anchor KV moved and the extra pool's
        state did NOT, while the radix tree went on reporting the prefix as
        RESIDENT. Attention (or, for MAMBA, the recurrent step) then reads
        state nobody wrote: a wrong ANSWER, with no assertion anywhere.

        A crash is the cheaper failure and it is the honest one -- by the time
        execution reaches this loop the copy is already in flight, so there is
        no correct way to continue. The refusal that costs nothing lives one
        tier up, at `_resolve_pool_transfers_allocation`, which now returns
        None instead of a half-resolved list; this raise exists so that a
        future producer that bypasses the resolver is LOUD rather than wrong.

        Note the reachability: the `kernel`/`page_first`/write-back-JIT path
        hands `op.pool_transfers` here RAW (hybrid_cache_controller.py:468-475),
        bypassing `move_hybrid_indices` entirely -- so on that configuration
        this skip was the LIVE failure and no crash would ever have exposed it.
        """
        entry = self.entry_map.get(transfer.name)
        if entry is None:
            raise ValueError(
                f"HiCache {direction}: the bound host tier describes "
                f"{sorted(str(n) for n in self.entry_map)} and cannot describe "
                f"pool '{transfer.name}'. Skipping this transfer would move the "
                "KV while this pool's state stayed behind, with the tree still "
                "reporting the prefix resident -- a wrong answer. This is a "
                "phase-binding defect (see check_pool_coverage), not a "
                "transient."
            )
        if transfer.host_indices is None or transfer.device_indices is None:
            raise ValueError(
                f"HiCache {direction}: pool '{transfer.name}' reached the "
                f"executor unresolved (host="
                f"{'set' if transfer.host_indices is not None else 'None'}, "
                f"device="
                f"{'set' if transfer.device_indices is not None else 'None'}). "
                "_resolve_pool_transfers_allocation must refuse such a set "
                "before it is enqueued."
            )
        return entry

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        pool_transfers: Optional[list] = None,
    ) -> None:
        # 1. Anchor (KV) transfer
        anchor = self.anchor_entry
        local_layer_id = anchor.layer_mapper(layer_id)
        if local_layer_id is not None and host_indices.numel() > 0:
            anchor.host_pool.load_to_device_per_layer(
                anchor.device_pool,
                host_indices,
                device_indices,
                local_layer_id,
                io_backend,
            )

        # 2. Extra pool transfers
        for transfer in pool_transfers or []:
            entry = self._entry_for_transfer(transfer, "load")
            local_layer_id = entry.layer_mapper(layer_id)
            if local_layer_id is None:
                # A layer this pool does not cover. The ONLY legitimate skip
                # here, and it is per-layer, not per-pool.
                continue
            entry.host_pool.load_to_device_per_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                local_layer_id,
                io_backend,
            )

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        pool_transfers: Optional[list] = None,
    ) -> None:
        # 1. Anchor (KV) backup
        self.anchor_entry.host_pool.backup_from_device_all_layer(
            self.anchor_entry.device_pool,
            host_indices,
            device_indices,
            io_backend,
        )
        # 2. Extra pool backup
        for transfer in pool_transfers or []:
            entry = self._entry_for_transfer(transfer, "backup")
            entry.host_pool.backup_from_device_all_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                io_backend,
            )


class DSAIndexerPoolHost(HostKVCache):
    """Host-side DSA index buffers only. Slot layout matches the anchor MLA host pool."""

    device_pool: DSATokenToKVPool

    @revert_pinned_posts_on_failure  # #729
    def __init__(
        self,
        device_pool: DSATokenToKVPool,
        anchor_host: MLATokenToKVPoolHost,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = anchor_host.page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.layer_num = self._effective_host_layer_num()

        self.index_head_dim = device_pool.index_head_dim
        self.indexer_quant_block_size = device_pool.quant_block_size
        self.indexer_dtype = DSATokenToKVPool.index_k_with_scale_buffer_dtype
        self.indexer_size_per_token = (
            self.index_head_dim
            + self.index_head_dim // self.indexer_quant_block_size * 4
        )
        self.size = anchor_host.size
        self.page_num = anchor_host.page_num

        self.indexer_page_stride_size = (
            self.indexer_size_per_token * self.page_size * self.indexer_dtype.itemsize
        )
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        self.size_per_token = (
            self.indexer_size_per_token * self.layer_num * self.indexer_dtype.itemsize
        )

        buf_elem_size = self.page_num * self.layer_num * self.indexer_page_stride_size
        requested_bytes = buf_elem_size * self.indexer_dtype.itemsize
        check_and_register_pinned_post(
            name="DSA indexer host pool",
            flag="--hicache-size / --hicache-ratio",
            requested_bytes=requested_bytes,
            reserve_bytes=HICACHE_HOST_MEMORY_RESERVE_BYTES,
        )
        logger.info(
            "Allocating %.2f GB host memory for DSA indexer (layout=%s).",
            requested_bytes / 1e9,
            layout,
        )
        self.init_kv_buffer()
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self._init_write_back_staging_buffers()
        self.lock = threading.RLock()
        self.clear()

    def get_size_per_token(self):
        return (
            self.indexer_size_per_token * self.layer_num * self.indexer_dtype.itemsize
        )

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_pool.index_k_with_scale_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.layout == "layer_first":
            self.index_k_with_scale_buffer = alloc_func(
                (self.layer_num, self.indexer_page_num, self.indexer_page_stride_size),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.index_k_data_refs = [
                self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
            ]
            self.index_k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in self.index_k_data_refs],
                dtype=torch.uint64,
                device=self.device_pool.device,
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            self.index_k_with_scale_buffer = alloc_func(
                (
                    self.indexer_page_num,
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                ),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def _init_write_back_staging_buffers(self):
        self.staging_buffer = None
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self.indexer_page_stride_size * self.indexer_dtype.itemsize,
        )
        staging_page_capacity = min(
            self.indexer_page_num, _WRITE_BACK_STAGING_PAGE_CHUNK
        )
        self.staging_buffer = torch.empty(
            (
                staging_page_capacity,
                self.layer_num,
                1,
                self.indexer_page_stride_size,
            ),
            dtype=self.indexer_dtype,
            device=self.device_pool.device,
        )

    def get_hybrid_pool_buffer(self):
        return [self.index_k_with_scale_buffer]

    def _get_indexer_page_indices(self, host_indices, device_indices):
        if host_indices.numel() == 0:
            return host_indices, device_indices
        if host_indices.numel() % self.page_size != 0:
            raise ValueError(
                "Index buffer transfer expects page-aligned indices for DSA."
            )
        host_page_indices = (
            host_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        device_page_indices = (
            device_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        return host_page_indices, device_page_indices

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if not self._is_device_layer_owned(device_pool, layer_id):
            return
        host_layer = self._host_layer_index(layer_id)

        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.index_k_with_scale_buffer[host_layer],
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.index_k_with_scale_buffer,
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=host_layer,
                    item_size=self.indexer_page_stride_size,
                    src_layout_dim=self.indexer_layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.index_k_with_scale_buffer[host_layer]],
                    dst_layers=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.index_k_with_scale_buffer],
                    dst_ptrs=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=host_layer,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_from_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        host_layer = self._host_layer_index(layer_id)
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=device_pool.index_k_with_scale_buffer[layer_id],
                    dst=self.index_k_with_scale_buffer[host_layer],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                raise ValueError(
                    "Layer-sharded DSA indexer HiCache backup with page_first "
                    "layout is not supported without a per-layer LF->PF kernel."
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[device_pool.index_k_with_scale_buffer[layer_id]],
                    dst_layers=[self.index_k_with_scale_buffer[host_layer]],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            else:
                raise ValueError(
                    "Layer-sharded direct DSA indexer backup only supports "
                    f"layer_first layout, got {self.layout}"
                )
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if self._is_device_layer_sharded(device_pool):
            for layer_id in self._owned_device_layer_ids(device_pool):
                self._backup_from_device_per_layer(
                    device_pool, host_indices, device_indices, layer_id, io_backend
                )
            return

        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
                    src_layers=self.index_k_device_ptrs,
                    dst_layers=self.index_k_data_ptrs,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                if self.can_use_write_back_jit:
                    jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                        ptr_src=self.index_k_device_ptrs,
                        src_indices=device_page_indices,
                        dst_indices=host_page_indices,
                        staging=self.staging_buffer,
                        dst=self.index_k_with_scale_buffer,
                        page_size=1,
                        element_size=self.indexer_page_stride_size,
                    )
                else:
                    transfer_kv_all_layer_mla_lf_pf(
                        src_layers=self.index_k_device_ptrs,
                        dst=self.index_k_with_scale_buffer,
                        src_indices=device_page_indices,
                        dst_indices=host_page_indices,
                        item_size=self.indexer_page_stride_size,
                        dst_layout_dim=self.indexer_layout_dim,
                        num_layers=self.layer_num,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.index_k_with_scale_buffer,
                    dst_layers=self.index_k_data_refs,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.index_k_with_scale_buffer,
                    dst_ptrs=[self.index_k_with_scale_buffer],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        page_idx = int(index) // self.page_size
        if self.layout == "layer_first":
            data_page = self.index_k_with_scale_buffer[:, page_idx : page_idx + 1, :]
        elif self.layout in ["page_first", "page_first_direct"]:
            data_page = self.index_k_with_scale_buffer[page_idx : page_idx + 1, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (self.layer_num, self.indexer_page_stride_size),
            dtype=self.indexer_dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        page_idx = int(index) // self.page_size
        if self.layout == "layer_first":
            self.index_k_with_scale_buffer[:, page_idx : page_idx + 1, :] = (
                data_page.reshape(
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                )
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            self.index_k_with_scale_buffer[page_idx : page_idx + 1, :, :, :] = (
                data_page.reshape(
                    1,
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """Meta data for zero-copy storage I/O."""
        assert len(indices) % self.page_size == 0
        if self.layout not in ["page_first", "page_first_direct"]:
            raise ValueError(f"Unsupported layout: {self.layout}")
        ptr_list = []
        indices = indices.tolist()
        page_stride_bytes = (
            self.layer_num * self.indexer_page_stride_size * self.indexer_dtype.itemsize
        )
        base_ptr = self.index_k_with_scale_buffer.data_ptr()
        for i in range(0, len(indices), self.page_size):
            page_index = int(indices[i]) // self.page_size
            ptr_list.append(base_ptr + page_index * page_stride_bytes)
        return ptr_list, [page_stride_bytes] * len(ptr_list)
