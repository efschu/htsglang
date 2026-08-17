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

Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

from __future__ import annotations

import abc
import dataclasses
import logging
import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.jit_kernel.kvcache import can_use_store_cache, store_cache
from sglang.kernels.ops.kvcache.cache_move import (
    copy_all_layer_kv_cache_func,
    set_kv_buffer_prefix_valid_tiled,
    store_cache_4d,
)
from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import index_buf_accessor
from sglang.srt.layers.attention.dsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner
from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    build_page_major_mha_views,
    mamba_entry_bytes,
    mha_entry_bytes,
)
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    maybe_init_custom_mem_pool,
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
    set_mla_kv_scale_buffer_triton,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_cpu_has_amx_support = cpu_has_amx_support()
_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()
# `SGLANG_AITER_KV_CACHE_LAYOUT` is only meaningful on the ROCm AITER backend
# (HIP + --enable-aiter / SGLANG_USE_AITER=1). On any other platform / backend
# the SHUFFLE 5D pool layout has no consumer kernels, so the env var is
# silently ignored and the legacy NHD layout is used.
_use_aiter = bool(envs.SGLANG_USE_AITER.get()) and _is_hip


def conv_window_dedup_enabled(
    is_npu: bool, is_cpu: bool, speculative_eagle_topk: Optional[int]
) -> bool:
    """Whether the deduplicated sliding-window conv-intermediate layout is safe.

    It is only correct for a *linear* draft chain (``speculative_eagle_topk <= 1``,
    i.e. NEXTN / MTP): consecutive draft tokens then form a true sliding window, so
    the overlapping physical columns hold identical values. Under EAGLE *tree*
    verify (``topk > 1``) the conv kernel walks per-token tree ancestors, so aliased
    columns can need different values from different parent chains -> fall back to
    the dense layout. NPU/CPU also keep the dense layout (their kernels assume
    contiguous per-step windows). See ``MambaPool.__init__``.
    """
    return (
        not is_npu
        and not is_cpu
        and (speculative_eagle_topk is None or speculative_eagle_topk <= 1)
    )


def get_tensor_size_bytes(t: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(t, list):
        return sum(get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize


def graph_safe_store_bound(live_limit: int, cache_rows: int) -> int:
    """The slot bound to hand ``store_cache`` (#352).

    ``store_kvcache`` takes ``size_limit`` as a by-value kernel parameter, so a
    CUDA graph records the number it was given AT CAPTURE TIME and replays that
    number forever. The pool's live bound (``size + page_size``) is therefore
    only correct for an eager launch: after a #330 capacity growth the pool's
    ``size`` rises, the allocator hands out the new slot ids, and a graph
    captured before the growth rejects them with
    ``index >= 0 && index < size_limit`` -- a device assert on a LEGAL id.
    That is the whole of #352: >= 10 x 30k concurrent sessions are the first
    load that holds more than the boot ceiling at once, so they are the first
    to reach an id above it (low ids are handed out first).

    A bound that may end up inside a graph has to hold for the graph's whole
    lifetime, and exactly one number does: the KV buffer's ROW COUNT. The VMM
    lane reserves the virtual address range once at boot and never moves or
    reshapes it (that is what keeps captured graphs replayable at all), and
    ``KvVmmBufferOwner.finalize`` refuses any capacity above that reservation
    -- so ``cache_rows`` is an upper bound on every ``size + page_size`` the
    pool can ever reach.

    NOT gated on capture mode, deliberately. Gating would re-create the bug
    class one level up: ``get_is_capture_mode()`` is set by each graph runner
    individually and ``PrefillCudaGraphRunner.capture()`` does not set it at
    all (it is covered today only because the CUDA prefill default backend is
    the breakable graph, which ``get_is_capture_mode`` happens to detect). A
    bound whose correctness depends on every present and future capture site
    remembering to raise a flag is the same "one consumer never got the
    treatment" defect that #345 and #352 both were.

    Widening costs no safety. The band this newly admits is
    ``(size + page_size, cache_rows]``, i.e. rows above the committed span,
    which on the VMM lane are UNMAPPED: an id escaping into them still fails
    hard (illegal access at a named address) instead of touching a live row.
    Every MAPPED row was already inside the old bound. Off the dial lane the
    buffers are allocated at exactly ``size + page_size`` rows, so
    ``live_limit == cache_rows`` and the bound is unchanged -- byte-identical
    behaviour on the default path, and the strict live check stays available
    host-side via ``maybe_detect_oob``.
    """
    return max(int(live_limit), int(cache_rows))


def kv_store_bound(live_limit: int, cache: torch.Tensor, row_dim: int) -> int:
    """The single slot bound every KV writer in this module hands its kernel.

    ``graph_safe_store_bound`` picks WHICH number is graph-stable; this wrapper
    adds the one geometry conversion all the writers share, so that no writer
    has to spell it out again. A KV buffer's addressable row count under a
    writer that strides by ``row_dim`` is ``numel // row_dim`` -- true whether
    the buffer is stored 3-D ``[rows, H, D]`` (per-layer pool) or already
    flattened, and ``row_dim`` is deliberately the stride the CALLING KERNEL
    uses, not ``self.row_dim``, so the bound describes the memory that kernel
    can actually reach.

    #352 gave the raw ``store_cache`` path a bound; #355 gave the masked Triton
    writers the same one. They route through this function so the two cannot
    drift: a writer growing its own bound expression is exactly the "one
    consumer never got the treatment" defect that #345 and #352 both were.
    ``test_kv_store_bound_unity.py`` fails if a writer stops using it.
    """
    return graph_safe_store_bound(live_limit, cache.numel() // row_dim)


def kv_bound_check_enabled() -> bool:
    """Whether the masked KV writers compile their in-kernel bound check (#355).

    Passed to Triton as the per-launch ``debug`` option: on (the default) the
    ``tl.device_assert`` lowers to a real device assert, off it lowers to
    nothing at all, so the off-switch leaves no residual instruction to pay
    for.

    Deliberately NOT cached in a module global, and the price is measured
    rather than assumed: 683 ns/call (200k reps, CPU). The masked writer runs
    once per full-attention layer per step -- 16 of the 64 layers on
    Qwen3.6-27B -- so this is ~11 us of host time per decode step against a
    step of 15-25 ms on this rig, i.e. below 0.1%. Caching it would buy that
    back and cost the ability to flip the switch after import, which is what
    the falsifier test and any live A/B need.
    """
    return not envs.SGLANG_DISABLE_KV_MASKED_BOUND_CHECK.get()


def _set_kv_buffer_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,  # head_num * head_dim
    store_dtype: torch.dtype,
    device_module: Any,
    size_limit: int,
    alt_stream: Optional[torch.cuda.Stream] = None,
    same_kv_dim: bool = True,
) -> None:
    row_bytes = row_dim * store_dtype.itemsize
    if (_is_cuda or _is_hip) and same_kv_dim and can_use_store_cache(row_bytes):
        k_cache_rows = k_cache.view(-1, row_dim)
        try:
            return store_cache(
                k.view(-1, row_dim),
                v.view(-1, row_dim),
                k_cache_rows,
                v_cache.view(-1, row_dim),
                indices,
                row_bytes=row_bytes,
                size_limit=kv_store_bound(size_limit, k_cache, row_dim),
            )
        except RuntimeError as e:
            # The jit kernel's TensorMatcher error names only the offending
            # tensor; re-raise with the full call geometry so a row/index
            # count divergence (e.g. a mis-shaped pool under a derived
            # geometry) is diagnosable from the log alone.
            raise RuntimeError(
                f"store_cache rejected: k{tuple(k.shape)} v{tuple(v.shape)} "
                f"row_dim={row_dim} -> k_rows={k.numel() // row_dim}, "
                f"k_cache{tuple(k_cache.shape)}, indices"
                f"{tuple(indices.shape)}:{indices.dtype}, "
                f"size_limit={size_limit}: {e}"
            ) from e

    if _is_cpu and _cpu_has_amx_support:
        return torch.ops.sgl_kernel.store_cache_cpu(
            k,
            v,
            k_cache,
            v_cache,
            indices,
            row_dim,
        )

    from sglang.srt.model_executor.runner import get_is_capture_mode

    if get_is_capture_mode() and alt_stream is not None:
        current_stream = device_module.current_stream()
        alt_stream.wait_stream(current_stream)
        k_cache[indices] = k
        with device_module.stream(alt_stream):
            v_cache[indices] = v
        current_stream.wait_stream(alt_stream)
    else:  # fallback to naive implementation
        k_cache[indices] = k
        v_cache[indices] = v


def _set_kv_buffer_prefix_valid_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    row_dim: int,
    store_dtype: torch.dtype,
    size_limit: int,
) -> None:
    if k.numel() == 0 or loc_2d.numel() == 0 or commit_lens.numel() == 0:
        return

    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not loc_2d.is_contiguous():
        loc_2d = loc_2d.contiguous()
    if not commit_lens.is_contiguous():
        commit_lens = commit_lens.contiguous()

    row_bytes = row_dim * store_dtype.itemsize
    if row_bytes <= 0:
        return

    if row_bytes >= 8192:
        bytes_per_tile = 512
        num_warps = 8
    elif row_bytes >= 4096:
        bytes_per_tile = 256
        num_warps = 4
    else:
        bytes_per_tile = 128
        num_warps = 4

    grid = (
        int(loc_2d.shape[0]),
        int(loc_2d.shape[1]),
        triton.cdiv(row_bytes, bytes_per_tile),
    )

    # #355: the kernel strides both destination buffers by their own row stride,
    # so bound each by that stride and take the smaller -- one ``loc`` addresses
    # both. Same helper as the raw store_cache path (#352), so no drift.
    bound = min(
        kv_store_bound(size_limit, k_cache, int(k_cache.stride(0))),
        kv_store_bound(size_limit, v_cache, int(v_cache.stride(0))),
    )

    set_kv_buffer_prefix_valid_tiled[grid](
        k,
        v,
        k_cache,
        v_cache,
        loc_2d,
        commit_lens,
        int(k.stride(0) * k.element_size()),
        int(v.stride(0) * v.element_size()),
        int(k_cache.stride(0) * k_cache.element_size()),
        int(v_cache.stride(0) * v_cache.element_size()),
        int(loc_2d.shape[1]),
        bound,
        ROW_BYTES=row_bytes,
        BYTES_PER_TILE=bytes_per_tile,
        num_warps=num_warps,
        num_stages=2,
        debug=kv_bound_check_enabled(),
    )


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    enable_mamba_extra_buffer_lazy: bool = False

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        # +1 padding row at index 0: cuda-graph padded batches default
        # req_pool_indices to 0, so dummy reads/writes land here harmlessly.
        self._alloc_size = size + 1
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation = torch.zeros(self._alloc_size, dtype=torch.int64)

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        # NOTE: this check is relaxed temporarily
        # https://github.com/sgl-project/sglang/pull/20476
        # if not any(r.is_dllm() for r in reqs):
        #     assert (
        #         sum(1 for i in reusing if reqs[i].inflight_middle_chunks > 0) <= 1
        #     ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                self.req_generation[r.req_pool_idx] += 1
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free_slot(self, idx: int, *, owner: str = "request", prepend: bool = False):
        """Return one row to the free list, refusing a corrupt return loudly.

        Every row handed out is named in exactly two places -- the request that
        holds it and, for a streaming session, the ``SessionSlot`` that parked
        it between turns. If both return the same row the free list carries it
        twice and the pool hands one row to two concurrent requests: the two
        then share a ``req_to_token`` row and a mamba mapping entry, and
        ``available_size()`` reports more rows than the pool owns, so admission
        over-fills against buffers sized by ``max_num_reqs`` (#616).

        A double return is a bookkeeping defect, never something to absorb: it
        is refused by name here rather than silently corrupting the pool for a
        later device-side assert to surface somewhere unrelated. The membership
        scan is O(free list), which is bounded by the pool size and runs once
        per finished request.
        """
        if idx is None:
            raise ValueError(f"{owner} returned a null req_pool_idx to the pool")
        if not 0 <= idx < self._alloc_size:
            raise ValueError(
                f"{owner} returned req_pool_idx={idx} to a pool holding rows "
                f"0..{self._alloc_size - 1}"
            )
        if idx in self.free_slots:
            raise ValueError(
                f"{owner} returned req_pool_idx={idx} which is already free -- "
                "the row was returned twice, so it was still owned by someone "
                "when one of the two returns happened"
            )
        if prepend:
            # Rollback puts rows back at the head so the retry reuses them.
            self.free_slots.insert(0, idx)
        else:
            self.free_slots.append(idx)

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slot(req.req_pool_idx, owner="request")
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation.zero_()
        # Flush invariant: post-flush state must equal a fresh boot. The
        # token table starts zeroed at init; stale mappings would otherwise
        # survive a flush (including the padding row 0, which CUDA-graph
        # padded batches deliberately read).
        self.req_to_token.zero_()


class MambaPool:
    @dataclass(frozen=True, kw_only=True)
    class State:
        conv: List[torch.Tensor]
        temporal: torch.Tensor
        # GDN ReplaySSM ring buffers (slice 1a). Only allocated when
        # `--enable-linear-replayssm` is set; otherwise None so the legacy path is
        # byte-identical. Per-layer layout: [num_layers, num_slots, ...].
        #   replayssm_d: [num_layers, num_slots, HV, L, V]
        #   replayssm_k: [num_layers, num_slots, H,  L, K]
        #   replayssm_g: [num_layers, num_slots, HV, L]  (fp32)
        replayssm_d: Optional[torch.Tensor] = None
        replayssm_k: Optional[torch.Tensor] = None
        replayssm_g: Optional[torch.Tensor] = None

        def at_layer_idx(self, layer: int):
            kwargs = {}
            # Use fields instead of vars to avoid torch.compile graph break
            for f in fields(self):
                name = f.name
                v = getattr(self, name)
                if v is None:
                    kwargs[name] = None
                elif name in ("conv", "intermediate_conv_window"):
                    kwargs[name] = [conv[layer] for conv in v]
                else:
                    kwargs[name] = v[layer]

            return type(self)(**kwargs)

        def mem_usage_bytes(self):
            return sum(
                get_tensor_size_bytes(getattr(self, f.name))
                for f in dataclasses.fields(self)
                if getattr(self, f.name) is not None
            )

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        intermediate_ssm: torch.Tensor
        intermediate_conv_window: List[torch.Tensor]

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        envelope_layout: bool = False,
    ):
        conv_state_shape = cache_params.shape.conv
        temporal_state_shape = cache_params.shape.temporal
        conv_dtype = cache_params.dtype.conv
        ssm_dtype = cache_params.dtype.temporal
        # Kept for PD disaggregation: uneven-TP state slice offsets need the
        # unsharded totals + the partition unit family (get_state_dim_offsets).
        self.cache_params = cache_params
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        num_mamba_layers = len(mamba_layer_ids)

        self.size = size
        self.device = device
        self.debug_memory_pool = envs.SGLANG_DEBUG_MEMORY_POOL.get()
        self.enable_linear_replayssm = enable_linear_replayssm
        self.linear_replayssm_cache_len = linear_replayssm_cache_len

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        with (
            self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            if envelope_layout:
                # Page-granularity envelope layout (page_size==1 for state): all
                # mamba layers/slots share one contiguous byte buffer; conv and
                # temporal are strided views into it (see mem_cache/layout/
                # page_major.py). Only the standard CUDA Triton path is supported.
                assert not _is_npu and not (
                    _is_cpu and _cpu_has_amx_support
                ), "envelope_layout mamba is only supported on the CUDA path"
                max_slots = size + 1
                entry_bytes = mamba_entry_bytes(
                    layer_num=num_mamba_layers,
                    conv_state_shapes=conv_state_shape,
                    conv_dtype=conv_dtype,
                    temporal_state_shape=temporal_state_shape,
                    temporal_dtype=ssm_dtype,
                )
                self._raw = torch.zeros(
                    max_slots * entry_bytes, dtype=torch.uint8, device=device
                )
                conv_state, temporal_state = build_page_major_mamba_views(
                    self._raw,
                    layer_num=num_mamba_layers,
                    conv_state_shapes=conv_state_shape,
                    conv_dtype=conv_dtype,
                    temporal_state_shape=temporal_state_shape,
                    temporal_dtype=ssm_dtype,
                    max_slots=max_slots,
                )
            else:
                conv_state = [
                    torch.zeros(
                        size=(num_mamba_layers, size + 1) + conv_shape,
                        dtype=conv_dtype,
                        device=device,
                    )
                    for conv_shape in conv_state_shape
                ]

                if _is_npu:
                    from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                        _init_npu_conv_state,
                    )

                    conv_state = _init_npu_conv_state(
                        conv_state[0], conv_state_shape, speculative_num_draft_tokens
                    )

                if _is_cpu and _cpu_has_amx_support:
                    from sglang.srt.layers.amx_utils import _init_amx_conv_state

                    # CPU uses a different layout of conv_state for kernel optimization
                    conv_state = _init_amx_conv_state(conv_state)

                temporal_state = torch.zeros(
                    size=(num_mamba_layers, size + 1) + temporal_state_shape,
                    dtype=ssm_dtype,
                    device=device,
                )

            # GDN ReplaySSM ring buffers (slice 1a). Allocated only when the
            # flag is on; otherwise left as None so the legacy State is
            # byte-identical. temporal_state_shape == (HV, V, K).
            replayssm_d = replayssm_k = replayssm_g = None
            if enable_linear_replayssm:
                hv, v_dim, k_dim = temporal_state_shape
                h_k = getattr(cache_params.shape, "num_k_heads_per_tp", hv)
                L = linear_replayssm_cache_len
                num_slots = size + 1
                # Ring records live in the SSM dtype (bf16/fp32) except g (fp32).
                replayssm_d = torch.zeros(
                    size=(num_mamba_layers, num_slots, hv, L, v_dim),
                    dtype=ssm_dtype,
                    device=device,
                )
                replayssm_k = torch.zeros(
                    size=(num_mamba_layers, num_slots, h_k, L, k_dim),
                    dtype=ssm_dtype,
                    device=device,
                )
                # The log-decay gate ring (fp32): per-head SCALAR for the GDN
                # gate -> [.., L]; per-K VECTOR for the KDA gate -> [.., L, K]
                # (k_dim == temporal_state_shape[-1] for both).
                g_shape = (
                    (num_mamba_layers, num_slots, hv, L, k_dim)
                    if cache_params.is_kda
                    else (num_mamba_layers, num_slots, hv, L)
                )
                replayssm_g = torch.zeros(
                    size=g_shape,
                    dtype=torch.float32,
                    device=device,
                )

            if speculative_num_draft_tokens is not None:
                if _is_npu:
                    temporal_state = temporal_state.transpose(-1, -2)
                    temporal_state_shape = (
                        *temporal_state_shape[:-2],
                        temporal_state_shape[-1],
                        temporal_state_shape[-2],
                    )
                # Cache intermediate SSM states per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, K, V]
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device=device,
                )
                # Cache intermediate conv windows (last K-1 inputs) per draft token
                # during target verify.
                #
                # On CUDA (Triton conv kernel + Triton scatter) we use a
                # *deduplicated sliding-window* layout: consecutive draft tokens'
                # (K-1)-wide windows overlap by (K-2), so instead of D separate
                # [dim, K-1] windows we store one shared [dim, D+K-2] buffer per
                # (layer, slot) and expose an overlapping `as_strided` view of
                # logical shape [num_layers, size+1, draft_tokens, dim, K-1] where
                # step `t`'s window is the slice shared[..., :, t:t+K-1]. This
                # halves the conv-intermediate footprint (D*(K-1) -> D+K-2 columns)
                # with no numerical change: both the conv kernel write (idempotent
                # overlapping stores) and `fused_conv_window_scatter_with_mask`
                # consume the view through its strides.
                #
                # Dedup the sliding-window conv-intermediate only when it is safe:
                # CUDA + a linear draft chain (topk <= 1). NPU/CPU and EAGLE tree
                # verify (topk > 1) keep the dense layout -- see
                # `conv_window_dedup_enabled` for the full rationale. The
                # `fused_conv_window_scatter_with_mask` scatter is layout-agnostic,
                # so the dense fallback reads correctly through the same code path.
                dedup_conv_window = conv_window_dedup_enabled(
                    _is_npu, _is_cpu, speculative_eagle_topk
                )
                self._intermediate_conv_window_phys = []
                if dedup_conv_window:
                    intermediate_conv_window_cache = []
                    for conv_shape in conv_state_shape:
                        conv_dim, win = conv_shape  # win == conv_kernel - 1 == K-1
                        shared_win = (
                            speculative_num_draft_tokens + win - 1
                        )  # D + (K-1) - 1
                        phys = torch.zeros(
                            size=(
                                num_mamba_layers,
                                spec_state_size + 1,
                                conv_dim,
                                shared_win,
                            ),
                            dtype=conv_dtype,
                            device=device,
                        )
                        # view[l, s, step, d, w] = phys[l, s, d, step + w]
                        view = phys.as_strided(
                            (
                                phys.shape[0],
                                phys.shape[1],
                                speculative_num_draft_tokens,
                                conv_dim,
                                win,
                            ),
                            (
                                phys.stride(0),
                                phys.stride(1),
                                phys.stride(3),  # step -> shared-win axis (stride 1)
                                phys.stride(2),  # dim
                                phys.stride(3),  # win -> shared-win axis (stride 1)
                            ),
                        )
                        self._intermediate_conv_window_phys.append(phys)
                        intermediate_conv_window_cache.append(view)
                else:
                    # Original dense layout (NPU/CPU, or EAGLE tree verify): one
                    # [dim, K-1] window per draft token.
                    # Shape: [num_layers, size+1, draft_tokens, dim, K-1]
                    intermediate_conv_window_cache = [
                        torch.zeros(
                            size=(
                                num_mamba_layers,
                                spec_state_size + 1,
                                speculative_num_draft_tokens,
                                conv_shape[0],
                                conv_shape[1],
                            ),
                            dtype=conv_dtype,
                            device=device,
                        )
                        for conv_shape in conv_state_shape
                    ]
                    self._intermediate_conv_window_phys = intermediate_conv_window_cache
                self.mamba_cache = self.SpeculativeState(
                    conv=conv_state,
                    temporal=temporal_state,
                    intermediate_ssm=intermediate_ssm_state_cache,
                    intermediate_conv_window=intermediate_conv_window_cache,
                    replayssm_d=replayssm_d,
                    replayssm_k=replayssm_k,
                    replayssm_g=replayssm_g,
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                    f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                    # Report the deduplicated PHYSICAL conv-window buffers (the view
                    # over-reports its logical, un-deduplicated size).
                    f"intermediate_conv_window_cache size: {get_tensor_size_bytes(self._intermediate_conv_window_phys) / GB:.2f}GB "
                )
            else:
                self.mamba_cache = self.State(
                    conv=conv_state,
                    temporal=temporal_state,
                    replayssm_d=replayssm_d,
                    replayssm_k=replayssm_k,
                    replayssm_g=replayssm_g,
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                )
            if enable_linear_replayssm:
                logger.info(
                    f"GDN ReplaySSM ring buffers allocated (L="
                    f"{linear_replayssm_cache_len}): "
                    f"d={get_tensor_size_bytes(replayssm_d) / GB:.3f}GB, "
                    f"k={get_tensor_size_bytes(replayssm_k) / GB:.3f}GB, "
                    f"g={get_tensor_size_bytes(replayssm_g) / GB:.3f}GB "
                )
            # Gate granularity of the linear-attn layers (drives the kernel's
            # IS_KDA path + the g_cache layout). Read by the backend metadata to
            # decide the per-K (KDA) vs scalar (GDN) flush/advance handling.
            self.replayssm_is_kda = bool(
                enable_linear_replayssm and cache_params.is_kda
            )
            # Persistent per-slot decode-position cursor for ReplaySSM. Shared
            # across all linear-attn layers; advanced once per decode forward by
            # the backend metadata build. Index 0..size; reset on slot (re)alloc.
            self.replayssm_write_pos = (
                torch.zeros((size + 1,), dtype=torch.int32, device=device)
                if enable_linear_replayssm
                else None
            )
            mem_usage_bytes = self.mamba_cache.mem_usage_bytes()
            if isinstance(self.mamba_cache, self.SpeculativeState):
                # `intermediate_conv_window` is an as_strided view whose logical
                # shape over-reports its real footprint; charge the physical buffers
                # instead. No-op for the dense layout, where the view and the
                # physical tensors coincide.
                mem_usage_bytes -= get_tensor_size_bytes(
                    self.mamba_cache.intermediate_conv_window
                )
                mem_usage_bytes += get_tensor_size_bytes(
                    self._intermediate_conv_window_phys
                )
            self.mem_usage = mem_usage_bytes / GB
            self.num_mamba_layers = num_mamba_layers

        # Diagnostic read-before-write falsifier (no-op unless the env is set).
        # Data buffers only; replayssm_write_pos is a semantic cursor and must
        # stay zero.
        poison_tensors = list(self.mamba_cache.conv) + [
            self.mamba_cache.temporal,
            getattr(self.mamba_cache, "replayssm_d", None),
            getattr(self.mamba_cache, "replayssm_k", None),
            getattr(self.mamba_cache, "replayssm_g", None),
        ]
        if isinstance(self.mamba_cache, self.SpeculativeState):
            poison_tensors.append(self.mamba_cache.intermediate_ssm)
            poison_tensors.extend(
                getattr(self, "_intermediate_conv_window_phys", None)
                or self.mamba_cache.intermediate_conv_window
            )
        maybe_poison_pool_data(poison_tensors, "mamba pool")

        # #286 Erg. 8: book each session state SET (one slot across all GDN
        # layers) as a gdn_state_sets offload-register item. Gated behind
        # SGLANG_OFFLOAD_REGISTER (default off => no-op, the default path is
        # byte-identical); registration + size bookkeeping only, no movement.
        self._maybe_register_offload_state_sets()

    def _maybe_register_offload_state_sets(self):
        """Thin Erg.-8 adapter: registration is delegated to
        offload_gdn_states (per-set live size from the real tensor shapes,
        hot = active session via the allocator probe, va_stable required,
        session ladder from --gdn-state-set-ladder when configured)."""
        from sglang.srt.model_executor.offload_register import (
            offload_register_enabled,
        )

        if not offload_register_enabled():
            return
        from sglang.srt.model_executor.offload_gdn_states import (
            register_mamba_state_sets,
        )

        register_mamba_state_sets(self)

    def get_speculative_mamba2_params_all_layers(self) -> SpeculativeState:
        assert isinstance(self.mamba_cache, self.SpeculativeState)
        return self.mamba_cache

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_cache.at_layer_idx(layer_id)

    def reset_state(self):
        """Return every device buffer of the pool to its freshly-initialized
        (all-zero) contents: conv/temporal states, spec intermediate caches,
        GDN ReplaySSM rings and their per-slot write cursors.

        Used by the idle-time cache flush: freeing the slots alone is not
        enough — recycled slots whose per-slot metadata (most notably
        ``replayssm_write_pos``) or state bytes survive a flush can leak
        stale-state effects into later requests, so a flushed server would
        not behave like a freshly started one.

        Ordering: this is launched from the scheduler thread (default
        stream) while model forwards run on their own stream. Synchronize
        the device on BOTH sides of the memsets — (a) so they cannot overlap
        the async tail of pre-flush work and (b) so no post-flush prefill
        can start while the reset is still zeroing buffers underneath it (a
        half-finished reset corrupts the first post-flush request
        nondeterministically, which is worse than not resetting at all).
        """
        self._sync_device()
        raw = getattr(self, "_raw", None)
        if raw is not None:
            # Envelope layout: conv/temporal are views into one byte buffer.
            raw.zero_()
        else:
            for conv in self.mamba_cache.conv:
                conv.zero_()
            self.mamba_cache.temporal.zero_()
        for name in ("replayssm_d", "replayssm_k", "replayssm_g"):
            t = getattr(self.mamba_cache, name, None)
            if t is not None:
                t.zero_()
        if isinstance(self.mamba_cache, self.SpeculativeState):
            self.mamba_cache.intermediate_ssm.zero_()
            # The logical conv-window views may be as_strided over shared
            # physical buffers; zero the physical storage where it exists
            # (UnifiedMambaPool holds only the dense logical tensors).
            phys = getattr(self, "_intermediate_conv_window_phys", None)
            for t in (
                phys
                if phys is not None
                else (self.mamba_cache.intermediate_conv_window)
            ):
                t.zero_()
        if self.replayssm_write_pos is not None:
            self.replayssm_write_pos.zero_()
        self._sync_device()

    def _sync_device(self):
        """Device-wide barrier for the (rare, idle-time) reset path. No-op on
        CPU pools (the accelerator platform may be unavailable there)."""
        if str(self.device).startswith("cpu"):
            return
        current_platform.synchronize()

    def clear_slots(self, indices: torch.Tensor):
        """Zero out mamba state at the given pool indices. Must run on forward stream."""
        if not _is_npu:
            need_size = len(indices)
            for i in range(len(self.mamba_cache.conv)):
                t = self.mamba_cache.conv[i]
                z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                    t.shape[0], need_size, *t.shape[2:]
                )
                t[:, indices] = z
            t = self.mamba_cache.temporal
            z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                t.shape[0], need_size, *t.shape[2:]
            )
            t[:, indices] = z
        else:
            for i in range(len(self.mamba_cache.conv)):
                t = self.mamba_cache.conv[i]
                t[:, indices] = 0
            t = self.mamba_cache.temporal
            t[:, indices] = 0

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
        """Clone mamba state (conv + temporal) from src slots into dst slots.

        ReplaySSM invariant: the SOURCE must be a fully-flushed checkpoint
        (``write_pos[src] == 0``). Only ``temporal`` is copied, not the ring, so
        an un-flushed source would drop its last ``write_pos`` updates. Callers
        comply: COW copies radix checkpoints; ``cache_unfinished_req`` copies an
        active slot only during prefill (ring empty); ``cache_finished_req``
        caps the donate to the last flush boundary. The dst cursor is reset to 0
        (the copied checkpoint has no pending ring entries).
        """
        if self.replayssm_write_pos is not None and self.debug_memory_pool:
            # Debug-only (syncs): catch any copy of an active, un-flushed slot.
            src_wp = self.replayssm_write_pos[src_indices]
            assert bool((src_wp == 0).all().item()), (
                "copy_from requires a fully-flushed ReplaySSM source "
                f"(write_pos==0), got {src_wp.tolist()} for src "
                f"{src_indices.tolist()}"
            )
        for i in range(len(self.mamba_cache.conv)):
            self.mamba_cache.conv[i][:, dst_indices] = self.mamba_cache.conv[i][
                :, src_indices
            ]
        self.mamba_cache.temporal[:, dst_indices] = self.mamba_cache.temporal[
            :, src_indices
        ]
        if self.replayssm_write_pos is not None:
            self.replayssm_write_pos[dst_indices] = 0

    def export_state_blob(self, slot: int):
        """One session's GDN/Mamba state as an OPAQUE blob (#364).

        The blob is the unit the slot ladder vacates: every PERSISTENT
        per-slot field of the pool, copied out of VRAM so the slot can be
        handed back to the allocator and reused by another session. It is
        opaque on purpose -- nothing between here and ``import_state_blob``
        interprets it. In particular it does NOT travel the HiCache radix
        route (#212): a recurrent state is positional, not prefix-shareable,
        and a cache eviction of it would mean a silent full re-prefill.

        Covered: every field of ``mamba_cache`` except the transient
        speculative scratch, plus the ReplaySSM decode cursor. The exclusion
        list is imported from the offload register (#286 Erg. 8) rather than
        restated, so the blob and the register's per-set byte model cannot
        disagree about what a "state set" is. ``intermediate_ssm`` /
        ``intermediate_conv_window`` are verify scratch keyed to SPEC slots
        and rebuilt inside every decode step -- carrying them would make the
        blob bigger and no more correct.

        Unlike ``get_cpu_copy`` (which covers conv + temporal only, enough
        for its hibernate caller) this is complete, which is what makes the
        round-trip bit-identical.
        """
        from sglang.srt.model_executor.offload_gdn_states import (
            _TRANSIENT_SPEC_FIELDS,
        )

        idx = int(slot)
        if idx < 1 or idx > self.size:
            raise ValueError(
                f"state slot {idx} out of range; slots 1..{self.size} are "
                "session slots (slot 0 is the pool's dummy padding target "
                "and never vacates)"
            )
        current_platform.synchronize()
        blob = {}
        for f in fields(self.mamba_cache):
            if f.name in _TRANSIENT_SPEC_FIELDS:
                continue
            value = getattr(self.mamba_cache, f.name, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                blob[f.name] = [t[:, idx].to("cpu", copy=True) for t in value]
            else:
                blob[f.name] = value[:, idx].to("cpu", copy=True)
        if self.replayssm_write_pos is not None:
            blob["replayssm_write_pos"] = self.replayssm_write_pos[idx].to(
                "cpu", copy=True
            )
        current_platform.synchronize()
        return blob

    def import_state_blob(self, slot: int, blob) -> None:
        """Restore a blob produced by :meth:`export_state_blob` into ``slot``.

        The slot need not be the one it was exported from -- that is the
        point of vacating: a resumed session takes whichever resident slot is
        free. Field-by-field by NAME, so a pool built with a different
        optional-field set (ReplaySSM off) refuses the blob instead of
        silently restoring a partial state.
        """
        idx = int(slot)
        if idx < 1 or idx > self.size:
            raise ValueError(
                f"state slot {idx} out of range; slots 1..{self.size} are session slots"
            )
        expected = set(self.state_blob_fields())
        got = set(blob.keys())
        if expected != got:
            raise ValueError(
                "GDN state blob does not match this pool's state layout: "
                f"blob has {sorted(got)}, pool expects {sorted(expected)}. "
                "A partially restored recurrent state is silently wrong "
                "output, so this is a hard error."
            )
        current_platform.synchronize()
        for name, value in blob.items():
            if name == "replayssm_write_pos":
                self.replayssm_write_pos[idx] = value.to(
                    self.replayssm_write_pos.device
                )
                continue
            target = getattr(self.mamba_cache, name)
            if isinstance(target, (list, tuple)):
                for i, t in enumerate(target):
                    t[:, idx] = value[i].to(t.device)
            else:
                target[:, idx] = value.to(target.device)
        current_platform.synchronize()

    def state_blob_fields(self):
        """Names a blob of this pool carries, in a stable order. Used by the
        round-trip check and by callers sizing a blob tier."""
        from sglang.srt.model_executor.offload_gdn_states import (
            _TRANSIENT_SPEC_FIELDS,
        )

        names = [
            f.name
            for f in fields(self.mamba_cache)
            if f.name not in _TRANSIENT_SPEC_FIELDS
            and getattr(self.mamba_cache, f.name, None) is not None
        ]
        if self.replayssm_write_pos is not None:
            names.append("replayssm_write_pos")
        return names

    def get_cpu_copy(self, indices):
        current_platform.synchronize()
        conv_cpu = [
            conv[:, indices].to("cpu", non_blocking=True)
            for conv in self.mamba_cache.conv
        ]
        temporal_cpu = self.mamba_cache.temporal[:, indices].to(
            "cpu", non_blocking=True
        )
        current_platform.synchronize()
        return conv_cpu, temporal_cpu

    def load_cpu_copy(self, mamba_cache_cpu, indices):
        conv_cpu, temporal_cpu = mamba_cache_cpu
        current_platform.synchronize()
        for i, conv in enumerate(self.mamba_cache.conv):
            conv[:, indices] = conv_cpu[i].to(conv.device, non_blocking=True)
        self.mamba_cache.temporal[:, indices] = temporal_cpu.to(
            self.mamba_cache.temporal.device, non_blocking=True
        )
        current_platform.synchronize()

    def get_contiguous_buf_infos(self):
        """
        Get buffer info for RDMA registration.
        Only returns conv and temporal state buffers, excluding intermediate buffers
        used for speculative decoding (intermediate_ssm, intermediate_conv_window).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            # Skip intermediate buffers used only for speculative decoding
            # These buffers have different size (spec_state_size + 1) and should not be transferred
            if field in ("intermediate_ssm", "intermediate_conv_window"):
                continue
            # Skip GDN ReplaySSM ring buffers: they are derived/transient decode
            # scratch, not part of the persistent transferable state.
            if field in ("replayssm_d", "replayssm_k", "replayssm_g"):
                continue
            value = getattr(self.mamba_cache, field)
            if value is None:
                continue
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)
        data_ptrs, data_lens, item_lens = [], [], []

        for _, state_tensor in enumerate(state_tensors):
            data_ptrs += [
                state_tensor[i].data_ptr() for i in range(self.num_mamba_layers)
            ]
            data_lens += [state_tensor[i].nbytes for i in range(self.num_mamba_layers)]
            item_lens += [
                state_tensor[i][0].nbytes for i in range(self.num_mamba_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each state tensor.

        For mamba state, the layout is:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension (index 2) is the one that gets sliced by TP.
        Returns the size of this dimension for each tensor (repeated for each layer).
        """
        state_tensors = []
        for field in vars(self.mamba_cache):
            # Mirror the exclusions in get_contiguous_buf_infos so the returned
            # dims line up element-wise with the RDMA buffer list.
            if field in (
                "intermediate_ssm",
                "intermediate_conv_window",
                "replayssm_d",
                "replayssm_k",
                "replayssm_g",
            ):
                continue
            value = getattr(self.mamba_cache, field)
            if value is None:
                continue
            if isinstance(value, list):
                state_tensors.extend(value)
            else:
                state_tensors.append(value)

        dim_per_tensor = []
        for state_tensor in state_tensors:
            # state_tensor shape: [num_layers, size+1, sliceable_dim, ...]
            # The sliceable dimension is at index 2 (after num_layers and size)
            sliceable_dim = state_tensor.shape[2]
            # Repeat for each layer since we have per-layer data_ptrs
            dim_per_tensor += [sliceable_dim] * self.num_mamba_layers
        return dim_per_tensor

    def get_state_dim_offsets(self):
        """Per-tensor start offset (in dim-2 units) of THIS rank's slice within
        the unsharded state, element-wise parallel to get_state_dim_per_tensor.

        Needed by PD disaggregation when prefill and decode run different
        attn_tp_size under an uneven-TP (--rank-tp-ratio) plan: the generic
        `(rank * dim) % src_dim` formula in the transfer layer only holds for
        uniform shards. Returns None when no uneven plan is active (the caller
        then keeps the legacy uniform formula) or when the shape does not carry
        the partition metadata (non-Mamba2 linear-state families).
        """
        from sglang.srt.distributed.utils import tp_partition_size, tp_plan_active
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        tp_size = parallel.attn_tp_size
        if not tp_plan_active(tp_size):
            return None
        shape = getattr(self.cache_params, "shape", None)
        units = getattr(shape, "partition_units", None)
        conv_dim_total = getattr(shape, "conv_dim", None)
        num_heads_total = getattr(shape, "num_heads", None)
        if units is None or conv_dim_total is None or num_heads_total is None:
            return None
        rank = parallel.attn_tp_rank

        def _offset(total: int) -> int:
            return sum(tp_partition_size(total, tp_size, r, units) for r in range(rank))

        conv_offset = _offset(conv_dim_total)
        temporal_offset = _offset(num_heads_total)

        offsets = []
        for field in vars(self.mamba_cache):
            if field in (
                "intermediate_ssm",
                "intermediate_conv_window",
                "replayssm_d",
                "replayssm_k",
                "replayssm_g",
            ):
                continue
            value = getattr(self.mamba_cache, field)
            if value is None:
                continue
            is_conv = "conv" in field
            n_tensors = len(value) if isinstance(value, list) else 1
            per_tensor = conv_offset if is_conv else temporal_offset
            offsets += [per_tensor] * (n_tensors * self.num_mamba_layers)
        return offsets

    def get_conv_subblock_spec(self):
        """Conv-state sub-block structure, for TP-mismatched PD state transfer.

        The GDN conv1d operates on the concatenation [query | key | value]
        along conv_dim (qwen3_next Qwen3GatedDeltaNet: conv_dim =
        key_dim*2 + value_dim; conv1d.weight is loaded by
        mamba_v2_sharded_weight_loader with the sub-block spec
        [query_key, query_key, value]). Each sub-block is sharded
        INDEPENDENTLY by heads, so a rank's conv_state is
        [q_shard | k_shard | v_shard] concatenated -- NOT a contiguous slice
        of the full conv_dim. A PD transfer between different attn_tp_size must
        therefore move each sub-block separately (see _send_mamba_state_slice);
        a single flat slice delivers the wrong channels.

        Returns (sub_block_full_sizes, units, conv_dim), or None when the shape
        does not expose the GDN [q|k|v] conv structure -- then the caller keeps
        the legacy single-slice path (correct for the same-tp case, where no
        dim slicing happens at all).
        """
        shape = getattr(self.cache_params, "shape", None)
        conv_dim = getattr(shape, "conv_dim", None)
        num_heads = getattr(shape, "num_heads", None)
        head_dim = getattr(shape, "head_dim", None)
        units = getattr(shape, "partition_units", None)
        if conv_dim is None or num_heads is None or head_dim is None:
            return None
        value_dim = num_heads * head_dim
        key_dim = (conv_dim - value_dim) // 2
        # Only the [query_key | query_key | value] GDN layout is handled; if the
        # arithmetic does not reconstruct conv_dim, bail out and keep the legacy
        # flat slice rather than risk mis-slicing an unknown conv layout.
        if key_dim <= 0 or key_dim * 2 + value_dim != conv_dim:
            return None
        return ([key_dim, key_dim, value_dim], units, conv_dim)

    def get_conv_transfer_segments(self):
        """THIS rank's conv-state [q|k|v] sub-block segments for a
        TP-mismatched PD transfer, computed with THIS rank's own (possibly
        weighted --rank-tp-ratio) partition plan -- the prefill sender does NOT
        have the decode's plan, so the decode must compute the slice itself.

        Returns a list parallel to get_state_dim_per_tensor (one entry per
        conv/temporal tensor x layer): each CONV tensor's entry is a flat
        [src0, len0, src1, len1, ...] of per-sub-block (source offset in the
        FULL/unsharded conv tensor, length) for this rank's head shard of each
        sub-block; non-conv tensors get []. The destination layout is the
        rank's COMPACT conv shard = the sub-blocks concatenated in order, so the
        sender derives dst offsets by prefix-summing the lengths. Returns None
        when the GDN conv structure is unavailable (caller keeps flat slice).
        """
        spec = self.get_conv_subblock_spec()
        if spec is None:
            return None
        sub_sizes, units, _conv_dim = spec

        from sglang.srt.distributed.utils import (
            tp_partition_offset,
            tp_partition_size,
        )
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        tp_size = parallel.attn_tp_size
        rank = parallel.attn_tp_rank

        # (src offset in full tensor, len) per sub-block, using this rank's plan.
        pairs = []
        src_base = 0
        for sub_full in sub_sizes:
            seg_len = tp_partition_size(sub_full, tp_size, rank, units)
            seg_src = src_base + tp_partition_offset(sub_full, tp_size, rank, units)
            pairs += [seg_src, seg_len]
            src_base += sub_full

        segments = []
        for field in vars(self.mamba_cache):
            if field in (
                "intermediate_ssm",
                "intermediate_conv_window",
                "replayssm_d",
                "replayssm_k",
                "replayssm_g",
            ):
                continue
            value = getattr(self.mamba_cache, field)
            if value is None:
                continue
            is_conv = "conv" in field
            n_tensors = len(value) if isinstance(value, list) else 1
            entry = list(pairs) if is_conv else []
            segments += [list(entry) for _ in range(n_tensors * self.num_mamba_layers)]
        return segments


class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        *,
        size: int,
        mamba_size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        enable_mamba_extra_buffer_lazy: bool = False,
        speculative_num_draft_tokens: int = None,
        speculative_eagle_topk: Optional[int] = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        mamba_envelope_layout: bool = False,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        # Bound by the radix cache at construction (`bind_tree_cache`) so the
        # allocation sites below can evict cached checkpoints before declaring
        # the pool exhausted. None until then (and for pool-only unit setups),
        # in which case the sites degrade without an evict attempt.
        self.tree_cache = None
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_mamba_extra_buffer_lazy = enable_mamba_extra_buffer_lazy
        self.enable_memory_saver = enable_memory_saver
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            mamba_size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
            enable_linear_replayssm=enable_linear_replayssm,
            linear_replayssm_cache_len=linear_replayssm_cache_len,
            mamba_envelope_layout=mamba_envelope_layout,
        )

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
        speculative_eagle_topk: Optional[int] = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        mamba_envelope_layout: bool = False,
    ):
        self.mamba_pool = MambaPool(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
            enable_linear_replayssm=enable_linear_replayssm,
            linear_replayssm_cache_len=linear_replayssm_cache_len,
            envelope_layout=mamba_envelope_layout,
        )
        self.mamba_allocator = MambaSlotAllocator(
            size=mamba_size,
            device=device,
        )
        # #286 Erg. 8: wire the state-set hot criterion to the slot
        # allocator (active session = unparkable set). Flag-gated no-op;
        # until attached every set reports hot -- the safe direction.
        from sglang.srt.model_executor.offload_gdn_states import (
            attach_state_set_activity_probe,
        )

        attach_state_set_activity_probe(self.mamba_pool, self.mamba_allocator)
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}

        # Optional int8 checkpoint pool: the radix caches states here (int8) instead
        # of holding them in the active bf16 pool -> ~2x cached-prefix capacity at
        # fixed memory. Strategy-agnostic (no_buffer / extra_buffer / spec).
        from sglang.srt.mem_cache.mamba_checkpoint_pool import (
            maybe_init_int8_mamba_checkpoint_pool,
        )

        self.mamba_ckpt_pool = maybe_init_int8_mamba_checkpoint_pool(
            mamba_size=mamba_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
        )

        self.device = device
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            )

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def bind_tree_cache(self, tree_cache) -> None:
        """Give the pool a handle on the radix cache that owns its cached
        checkpoints, so `alloc` can evict before failing (#581)."""
        self.tree_cache = tree_cache

    def _alloc_mamba_slots_or_evict(self, n: int) -> Optional[torch.Tensor]:
        """Allocate `n` mamba slots, evicting cached checkpoints if needed.

        Returns None only when the pool is exhausted AND eviction frees
        nothing -- i.e. every cached state is locked by a running request.
        Callers must handle None; these are REQUIRED allocations (the
        request's own active state / ping-pong buffers), so the correct
        degradation is to fail the whole `alloc` cleanly and let admission
        defer the request, never to assert mid-mutation.
        """
        slots = self.mamba_allocator.alloc(n)
        if slots is None and self.tree_cache is not None:
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams

            self.tree_cache.evict(EvictParams(num_tokens=0, mamba_num=n))
            slots = self.mamba_allocator.alloc(n)
        elif slots is not None and self.tree_cache is not None:
            # #639b: this rank's alloc succeeded, a peer's did not. The peer is
            # tombstoning `n` mamba checkpoints right now, and a tombstone is a
            # tree edit the matcher reads (`create_match_validator` refuses a
            # node whose mamba value is None, and `_match_prefix_helper` stops
            # the match there). Not matching it here is what makes the two
            # radix replicas disagree about the prefix -- the 2026-08-07
            # `PrefixLensRankDivergence` crashes. Gated on a floor the
            # scheduler only publishes when the ranks' occupancy actually
            # differs, so a single-rank or even boot takes the pre-#639b path.
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams
            from sglang.srt.mem_cache.common import peer_needs_mamba_evict

            if peer_needs_mamba_evict(self.tree_cache, n):
                self.tree_cache.evict(EvictParams(num_tokens=0, mamba_num=n))
        return slots

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        # Requests that arrive without a row are the ones this call allocates
        # for. `super().alloc` returns a row for EVERY request in the batch
        # (chunked continuations included), so it cannot be used to decide what
        # a rollback may hand back -- see `_rollback_alloc` (#616).
        needs_row = [r for r in reqs if r.req_pool_idx is None]
        select_index = super().alloc(reqs)
        if select_index is None:
            return None
        fresh_rows = [r.req_pool_idx for r in needs_row]

        mamba_indices: list[torch.Tensor] = []
        mamba_ping_pong_track_buffers: list[torch.Tensor] = []
        # Requests this call gave a fresh mamba state / ping-pong buffer to.
        # On a mid-loop exhaustion every one of them must be handed back, or a
        # deferred batch silently leaks the slots it already took (#581).
        fresh_state_reqs: list[Req] = []
        fresh_pingpong_reqs: list[Req] = []
        for req in reqs:
            if req.mamba_pool_idx is not None:  # for radix cache / continuing chunked
                pass
            else:
                mid = self._alloc_mamba_slots_or_evict(1)
                if mid is None:
                    # REQUIRED allocation, and eviction could free nothing:
                    # every cached state is locked by a running request. Roll
                    # this call back completely and report failure so the
                    # caller defers the batch -- this used to be a bare assert
                    # that killed the scheduler AFTER partially mutating the
                    # pool.
                    self._rollback_alloc(
                        reqs, fresh_rows, fresh_state_reqs, fresh_pingpong_reqs
                    )
                    self._log_mamba_alloc_failure("mamba state", len(reqs))
                    return None
                req.mamba_pool_idx = mid[0]
                req.mamba_needs_clear = True
                fresh_state_reqs.append(req)
                # GDN ReplaySSM: a freshly (re)assigned slot starts an empty
                # ring. write_pos=0 means "ring empty", so the decode kernel
                # ignores ring contents and reads only the checkpoint state
                # (the post-prefill state that prefill wrote into this slot).
                if self.mamba_pool.replayssm_write_pos is not None:
                    self.mamba_pool.replayssm_write_pos[req.mamba_pool_idx] = 0
            mamba_indices.append(req.mamba_pool_idx)
            if self.enable_mamba_extra_buffer:
                if req.mamba_ping_pong_track_buffer is None:
                    if not self._alloc_ping_pong_buffer(req):
                        # Same REQUIRED-allocation degradation as the active
                        # state above: this is the exact site that killed
                        # rank 0 in the #581 boot-6 crash.
                        self._rollback_alloc(
                            reqs, fresh_rows, fresh_state_reqs, fresh_pingpong_reqs
                        )
                        self._log_mamba_alloc_failure("mamba ping-pong", len(reqs))
                        return None
                    fresh_pingpong_reqs.append(req)
                mamba_ping_pong_track_buffers.append(req.mamba_ping_pong_track_buffer)
        assert len(select_index) == len(
            mamba_indices
        ), "Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        if self.enable_mamba_extra_buffer:
            assert len(select_index) == len(
                mamba_ping_pong_track_buffers
            ), "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
        mamba_index_tensor = torch.stack(mamba_indices).to(dtype=torch.int32)
        self.req_index_to_mamba_index_mapping[select_index] = mamba_index_tensor
        if self.enable_mamba_extra_buffer:
            ping_pong_tensor = torch.stack(mamba_ping_pong_track_buffers)
            self.req_index_to_mamba_ping_pong_track_buffer_mapping[select_index] = (
                ping_pong_tensor
            )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def translate_mamba_indices(self, mamba_indices: torch.Tensor) -> torch.Tensor:
        """Virtual->physical mamba-slot translate. Identity for a static pool
        (slots are physical); UnifiedHybridReqToTokenPool overrides it for the
        unified memory pool, where mamba slot ids are virtual. Callers translate
        before calling the pool's physical-id state ops (copy_from / clear_slots
        / get_cpu_copy / load_cpu_copy)."""
        return mamba_indices

    def mamba2_layer_cache(self, layer_id: int):
        assert layer_id in self.mamba_map
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    def get_speculative_mamba2_params_all_layers(self) -> MambaPool.SpeculativeState:
        return self.mamba_pool.get_speculative_mamba2_params_all_layers()

    def get_state_buf_infos(self):
        return self.mamba_pool.get_contiguous_buf_infos()

    def get_state_dim_per_tensor(self):
        return self.mamba_pool.get_state_dim_per_tensor()

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - mamba_next_track_idx
        else:
            return mamba_next_track_idx

    def get_mamba_ping_pong_keep_idx(self, req: Req) -> int:
        """Return the ping-pong index holding the most recent tracked state.

        In lazy mode the valid state stays at next_track_idx (no eager swap).
        In normal mode it is at the "other" index (swapped after each track).
        """
        if self.enable_mamba_extra_buffer_lazy:
            return req.mamba_next_track_idx
        return self.get_mamba_ping_pong_other_idx(req.mamba_next_track_idx)

    def _rollback_alloc(
        self,
        reqs: List[Req],
        fresh_rows: List[int],
        fresh_state_reqs: List[Req],
        fresh_pingpong_reqs: List[Req],
    ) -> None:
        """Undo a partially completed `alloc` so a deferred batch leaks nothing.

        Frees exactly what THIS call handed out -- fresh mamba states, fresh
        ping-pong buffers and the request-pool rows it took -- and nothing
        else. A chunked continuation that arrived already owning its active
        state slot must keep it; freeing that would hand a live request's
        state to someone else.
        """
        for req in fresh_pingpong_reqs:
            if req.mamba_ping_pong_track_buffer is not None:
                buf = req.mamba_ping_pong_track_buffer
                self.mamba_allocator.free(buf[buf != -1])
                req.mamba_ping_pong_track_buffer = None
                req.mamba_next_track_idx = None
                req.mamba_pingpong_clear_indices = None
        for req in fresh_state_reqs:
            if req.mamba_pool_idx is not None:
                self.mamba_allocator.free(req.mamba_pool_idx.unsqueeze(0))
                req.mamba_pool_idx = None
            req.mamba_needs_clear = False
        # Hand the request-pool rows back (mirrors ReqToTokenPool.free).
        #
        # ONLY the rows this call took. `super().alloc` returns a row for every
        # request in the batch, including chunked continuations that arrived
        # already owning one, so rolling back that list returned live rows to
        # the free list and cleared their owners' `req_pool_idx` -- the pool
        # then handed one row to two requests (#616). `fresh_rows` is built
        # from the requests that arrived without a row, which is exactly the
        # set this call allocated.
        taken = set(fresh_rows)
        for req in reqs:
            if req.req_pool_idx is not None and int(req.req_pool_idx) in taken:
                req.req_pool_idx = None
        for idx in fresh_rows:
            self.free_slot(idx, owner="alloc rollback", prepend=True)

    def _log_mamba_alloc_failure(self, what: str, num_reqs: int) -> None:
        """Rate-limited warning for a REQUIRED mamba allocation that could not
        be served even after eviction."""
        self._mamba_alloc_failures = getattr(self, "_mamba_alloc_failures", 0) + 1
        count = self._mamba_alloc_failures
        if count <= 3 or count % 1000 == 0:
            evictable = protected = -1
            if self.tree_cache is not None:
                evictable = self.tree_cache.mamba_evictable_size()
                protected = self.tree_cache.mamba_protected_size()
            logger.warning(
                "%s slot pool exhausted and nothing evictable (all cached "
                "states are locked by running requests): deferring this batch "
                "of %d request(s). occurrence=%d pool=%d available=%d "
                "mamba_evictable=%d mamba_protected=%d",
                what,
                num_reqs,
                count,
                self.mamba_pool.size,
                self.mamba_allocator.available_size(),
                evictable,
                protected,
            )

    def _alloc_ping_pong_buffer(self, req: Req) -> bool:
        """Allocate the ping-pong track buffer for a new request.

        Lazy mode allocates 1 slot with the second set to -1 (allocated
        on demand at track boundaries). Normal mode allocates all slots upfront.

        Returns False when the pool is exhausted and eviction frees nothing;
        the caller then rolls the whole `alloc` back and defers the batch. It
        used to assert here, which killed the scheduler process (#581).
        """
        n = (
            1
            if self.enable_mamba_extra_buffer_lazy
            else self.mamba_ping_pong_track_buffer_size
        )
        slots = self._alloc_mamba_slots_or_evict(n)
        if slots is None:
            return False
        buf = torch.full(
            (self.mamba_ping_pong_track_buffer_size,),
            -1,
            dtype=slots.dtype,
            device=slots.device,
        )
        buf[:n] = slots
        req.mamba_ping_pong_track_buffer = buf
        req.mamba_next_track_idx = 0
        # Claimed slots must be zeroed before any kernel can observe them
        # (deferred to the forward stream, alongside the active-slot clear):
        # recycled slots otherwise expose the previous occupant's state to
        # any premature or partial read.
        req.mamba_pingpong_clear_indices = slots.clone()
        return True

    def set_mamba_ping_pong_slot(self, req: Req, idx: int, value):
        """Update a ping-pong slot value and sync the device-side mapping.

        The req holds the authoritative buffer; this keeps the
        req_index_to_mamba_ping_pong_track_buffer_mapping in sync so that
        set_mamba_track_indices_from_reqs reads correct slot indices.
        """
        req.mamba_ping_pong_track_buffer[idx] = value
        self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx] = (
            req.mamba_ping_pong_track_buffer
        )

    def donate_mamba_ping_pong_slot(
        self, req: Req, new_slot: torch.Tensor
    ) -> torch.Tensor:
        """Donate the tracked-state ping-pong slot to the radix cache.

        Returns the old slot index (shape [1]) for cache insertion and
        replaces it with new_slot so the request can continue tracking.
        In lazy mode the valid state is at next_track_idx; in normal mode
        it is at the "other" index.
        """
        donate_idx = self.get_mamba_ping_pong_keep_idx(req)
        mamba_value_donated = (
            req.mamba_ping_pong_track_buffer[donate_idx].unsqueeze(-1).clone()
        )
        assert mamba_value_donated.item() != -1, (
            f"Donated mamba slot is -1: donate_idx={donate_idx}, "
            f"buf={req.mamba_ping_pong_track_buffer.tolist()}, "
            f"next_track_idx={req.mamba_next_track_idx}, "
            f"rid={req.rid}"
        )
        self.set_mamba_ping_pong_slot(req, donate_idx, new_slot[0])
        # The replacement slot is a fresh claim: queue it for the deferred
        # forward-stream clear (see _alloc_ping_pong_buffer).
        fresh = new_slot.clone()
        if req.mamba_pingpong_clear_indices is None:
            req.mamba_pingpong_clear_indices = fresh
        else:
            req.mamba_pingpong_clear_indices = torch.cat(
                [req.mamba_pingpong_clear_indices, fresh]
            )
        return mamba_value_donated

    def free_mamba_cache(
        self, req: Req, mamba_ping_pong_track_buffer_to_keep: Optional[int] = None
    ):
        mamba_index = req.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.mamba_allocator.free(mamba_index.unsqueeze(0))
        req.mamba_pool_idx = None

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_free = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]
            )
            if mamba_ping_pong_track_buffer_to_keep is not None:
                assert mamba_ping_pong_track_buffer_to_keep in [
                    0,
                    1,
                ], f"mamba_ping_pong_track_buffer_to_keep must be 0 or 1, {mamba_ping_pong_track_buffer_to_keep=}"
                # Avoid Python-list advanced indexing on a device tensor.
                # The ping-pong buffer size is either 2 (normal) or 1 (spec decode).
                if self.mamba_ping_pong_track_buffer_size == 2:
                    idx_to_free = 1 - mamba_ping_pong_track_buffer_to_keep
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[
                            idx_to_free : idx_to_free + 1
                        ]
                    )
                else:
                    assert self.mamba_ping_pong_track_buffer_size == 1, (
                        f"Unexpected mamba_ping_pong_track_buffer_size="
                        f"{self.mamba_ping_pong_track_buffer_size}"
                    )
                    assert mamba_ping_pong_track_buffer_to_keep == 0, (
                        "mamba_ping_pong_track_buffer_to_keep must be 0 when "
                        "mamba_ping_pong_track_buffer_size is 1"
                    )
                    # Keep the only slot, so free nothing.
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[0:0]
                    )
            if self.enable_mamba_extra_buffer_lazy:
                mamba_ping_pong_track_buffer_to_free = (
                    mamba_ping_pong_track_buffer_to_free[
                        mamba_ping_pong_track_buffer_to_free != -1
                    ]
                )
            self.mamba_allocator.free(mamba_ping_pong_track_buffer_to_free)
            # Match the req.mamba_pool_idx=None clear above so the next
            # alloc() doesn't see a stale ping-pong reference on the req
            # and skip allocation (which would silently reuse a freed
            # tensor on the req side while the new pool slot leaks).
            req.mamba_ping_pong_track_buffer = None
            req.mamba_next_track_idx = None

    def clear(self):
        logger.info("Reset HybridReqToTokenPool")
        super().clear()
        self.mamba_allocator.clear()
        # A flush must also reset the mamba pool's device-side state
        # (conv/temporal, ReplaySSM rings + write cursors, spec
        # intermediates): freed slot ids get recycled, and any surviving
        # per-slot metadata would make a flushed server behave differently
        # from a freshly started one.
        self.mamba_pool.reset_state()
        # The int8 checkpoint pool holds radix-cached states in its own slots; a
        # flush/reset drops the radix tree, so its slots must be released too,
        # otherwise the (now unreferenced) slots leak and break the int8-pool
        # invariant (int8_available + radix_cached != int8_total).
        if self.mamba_ckpt_pool is not None:
            self.mamba_ckpt_pool.clear()
        self.req_index_to_mamba_index_mapping.zero_()
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


@dataclass
class KVWriteLoc:
    """Write target(s) for ``KVCache.set_kv_buffer``.

    All location info lives here (in the attention metadata), NOT in the pool:
    - ``loc``: the generic per-token write location (the allocated
      ``out_cache_loc``). VIRTUAL under the unified memory pool (it indexes the
      virtual slot space); already physical for a non-unified memory pool.
    - ``swa_loc``: the pre-translated SWA-sub-pool PHYSICAL location for hybrid
      SWA pools (``None`` otherwise).
    - ``full_loc``: the pre-translated full-attention-sub-pool PHYSICAL location
      for the unified memory pool (``None`` otherwise), computed once per forward in
      attention metadata (``ForwardMetadata.out_cache_loc_full_physical``). The
      shared full pool writes it directly; the pool never translates (replacing
      the former per-layer v2p gather / ``set_full_loc`` pin).

    ``swa_loc`` and ``full_loc`` are the parallel pair (each a pre-resolved
    PHYSICAL loc into its sub-pool, mirroring ``swa_kv_pool`` / ``full_kv_pool``);
    ``loc`` is the generic, possibly-virtual fallback. Bundling them lets a
    backend issue one ``set_kv_buffer`` call regardless of pool type.
    """

    loc: torch.Tensor
    swa_loc: Optional[torch.Tensor] = None
    full_loc: Optional[torch.Tensor] = None

    def __post_init__(self):
        # swa_loc / full_loc are resolved once at metadata-init from the full
        # (padded) out_cache_loc; piecewise/DP-padded paths later narrow loc per
        # layer, so slice these pre-resolved locs to match (same per-token order).
        if self.swa_loc is not None and self.swa_loc.shape[0] != self.loc.shape[0]:
            self.swa_loc = self.swa_loc[: self.loc.shape[0]]
        if self.full_loc is not None and self.full_loc.shape[0] != self.loc.shape[0]:
            self.full_loc = self.full_loc[: self.loc.shape[0]]


def unwrap_write_loc(loc_info):
    """Return ``(loc, swa_loc, full_loc)`` from a ``KVWriteLoc`` or a bare loc."""
    if isinstance(loc_info, KVWriteLoc):
        return loc_info.loc, loc_info.swa_loc, loc_info.full_loc
    return loc_info, None, None


class KvBufferDesc:
    """Byte-span math for one KV buffer laid out as rows of ``row_bytes`` holding
    ``tokens_per_row`` tokens each (a row = one token slot, or one whole page)."""

    __slots__ = ("name", "shape", "row_bytes", "tokens_per_row")

    def __init__(self, name: str, shape: tuple, *, row_bytes: int, tokens_per_row: int):
        self.name = name
        self.shape = tuple(shape)
        self.row_bytes = int(row_bytes)
        self.tokens_per_row = int(tokens_per_row)

    def _rows(self, num_tokens: int) -> int:
        n = max(int(num_tokens), 0)
        return (n + self.tokens_per_row - 1) // self.tokens_per_row

    def reserved_span_bytes(self, itemsize: int) -> int:
        """Full upper-bound byte size of the buffer (its whole tensor)."""
        return math.prod(self.shape) * itemsize

    def prefix_span_bytes(self, num_tokens: int, page_size: int) -> int:
        """Bytes to back to make the first ``num_tokens`` tokens usable."""
        return self._rows(num_tokens) * self.row_bytes

    def final_span_bytes(self, num_tokens: int, page_size: int) -> int:
        """Bytes of the final advertised span (adds the padded page). CEIL, not floor:
        an unaligned count must still cover its partial last page (e.g. n=17, page=16
        -> 3 pages, not 2)."""
        return self._rows(max(int(num_tokens), 0) + page_size) * self.row_bytes

    def item_len_bytes(self, page_size: int) -> int:
        """Per-page transfer chunk (one page's worth of this buffer)."""
        return (page_size // self.tokens_per_row) * self.row_bytes


def maybe_poison_pool_data(tensors, source: str) -> None:
    """SGLANG_POISON_POOL_DATA: fill freshly allocated pool DATA buffers with
    NaN instead of zeros (index/cursor tensors must stay semantic zeros and
    are not passed here).

    Diagnostic falsifier for read-before-write bugs: a fresh boot normally
    reads zeros from first-touch pages, which can HIDE a kernel that consumes
    pool bytes never written for the current request. With poison, any such
    read propagates NaN into the output loudly and immediately, instead of a
    silent, traffic-dependent divergence later. Claimed slots that go through
    the regular clear paths (deferred clear / full-state scatter) behave
    exactly as in production.

    uint8 buffers are fp8 KV data in disguise (MHATokenToKVPool stores
    float8_e5m2 / float8_e4m3fn as torch.uint8 because index_put lacks fp8
    support) — fill those with 0xFF, which decodes to NaN in BOTH fp8
    formats. Without this the fp8-KV configuration silently skipped the KV
    buffers and the falsifier reported a false pass for the KV read paths.
    (For the packed-fp4 pool 0xFF is not NaN but a saturated extreme value —
    still a loud, deterministic poison.)
    """
    if not envs.SGLANG_POISON_POOL_DATA.get():
        return
    poisoned = 0
    for t in tensors:
        if t is None:
            continue
        if t.is_floating_point():
            t.fill_(float("nan"))
            poisoned += 1
        elif t.dtype == torch.uint8:
            t.fill_(0xFF)
            poisoned += 1
    logger.warning(
        "SGLANG_POISON_POOL_DATA: poisoned %d %s buffers with NaN "
        "(0xFF for fp8-as-uint8 storage) — any read-before-write of pool "
        "data will now surface as NaN output.",
        poisoned,
        source,
    )


def zero_kv_data_buffers(kvcache) -> int:
    """Zero the KV data buffers of ``kvcache`` (and its sub-pools) in place;
    returns the number of buffers zeroed.

    Flush invariant helper: freshly booted pools are torch.zeros, so an
    idle-time flush must also return the DATA bytes to zero — otherwise a
    kernel that folds residual bytes beyond the valid sequence region into
    its result behaves differently (and run-dependently) after a flush.
    Covers the dense-attribute layouts (k_buffer / v_buffer / kv_buffer,
    tensor or list-of-tensors) plus the hybrid sub-pools; exotic pools
    without these attributes are simply skipped (best effort).
    """
    pools = [kvcache]
    for attr in ("full_kv_pool", "swa_kv_pool"):
        sub = getattr(kvcache, attr, None)
        if sub is not None:
            pools.append(sub)
    zeroed = 0
    for pool in pools:
        # A KV BUFFER IS VA-SIZED; ITS BACKING IS NOT.
        #
        # On a swappable pool the tensor spans the whole reservation while
        # only the rows below the watermark are mapped, so ``t.zero_()``
        # writes into unmapped address space -- cudaErrorIllegalAddress,
        # which kills every rank. That was unreachable while the only shrink
        # path (the #330 dial) destroyed the live set and never ran under
        # load; the #656 residency controller shrinks backing during normal
        # serving, and the corridor procedure itself calls /flush_cache
        # before every idle reading, so the two would have met.
        limit = getattr(pool, "safe_zero_rows", None)
        for name in ("k_buffer", "v_buffer", "kv_buffer"):
            bufs = getattr(pool, name, None)
            if bufs is None:
                continue
            if isinstance(bufs, torch.Tensor):
                bufs = [bufs]
            for t in bufs:
                if limit is None:
                    t.zero_()
                else:
                    t[: int(limit)].zero_()
                zeroed += 1
    return zeroed


def mark_as_sub_pool(pool) -> None:
    """Declare that `pool` is addressed in its OWN layer frame, not the global
    PP frame, and drop any global ownership map it picked up.

    `KVCache.__init__` attaches a map keyed by GLOBAL layer ids, which is right
    for a pool the model addresses with global ids. A sub-pool is different: its
    wrapper re-indexes first. `HybridLinearKVPool` maps a global id through
    `_transfer_full_attention_id` into a DENSE full-attention index before
    calling into `full_kv_pool`, so under SGLANG_PP_LAYER_SET the id arriving is
    0..N-1 while the map holds e.g. {35: 0, 39: 1, ...} and every lookup misses.

    Dropping the map makes the sub-pool's accessor degenerate to the plain
    subtraction inside its own dense frame, which is what its buffers are laid
    out for. On the contiguous path there is no map to drop and this is a no-op.
    """
    pool._local_slot_of = None


def _owned_layers_for_pool():
    """This stage's owned layer ids, or ``None`` on the contiguous path.

    Reads the SAME parser that built the model's layer list, so ownership has
    exactly one derivation. Returns None whenever the layer-set form is unused,
    which is what makes every accessor below degenerate to the arithmetic it
    replaces.
    """
    try:
        from sglang.srt.distributed import get_pp_group
        from sglang.srt.distributed.utils import get_pp_layer_set
    except Exception:  # pragma: no cover - import shape varies in unit tests
        return None
    try:
        group = get_pp_group()
        num_layers = getattr(group, "num_hidden_layers", None)
        if num_layers is None:
            return None
        return get_pp_layer_set(num_layers, group.rank_in_group, group.world_size)
    except Exception:  # pragma: no cover - no process group in unit tests
        return None


class KVCache(abc.ABC):
    layer_shard_enabled: bool = False
    post_capture_active: bool = False

    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1

        # NON-CONTIGUOUS OWNERSHIP: global layer id -> LOCAL buffer slot.
        #
        # Every per-layer buffer here is indexed by a stage-local slot, and the
        # translation was written inline as `layer_id - self.start_layer` in
        # dozens of places. That subtraction is correct only while a stage owns
        # a contiguous RANGE. Under SGLANG_PP_LAYER_SET a stage owns a set --
        # e.g. the full-attention layers {3, 7, 11, ...} of a hybrid model --
        # and then layer 7's local slot is 1, while the subtraction says 4.
        # Reading slot 4 of an 8-slot buffer is not a crash; it is another
        # layer's KV, returned confidently.
        #
        # So the translation goes through ONE accessor, built once here. When
        # ownership is contiguous the map is None and the accessor degenerates
        # to exactly the old subtraction -- that is what keeps the default path
        # byte-identical, and it is pinned.
        self._local_slot_of: Optional[Dict[int, int]] = None
        owned = _owned_layers_for_pool()
        if owned is not None:
            self._local_slot_of = {
                layer: slot for slot, layer in enumerate(sorted(owned))
            }
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

    def local_slot(self, layer_id: int) -> int:
        """Global layer id -> this stage's LOCAL buffer slot.

        Contiguous ownership: identical to the ``layer_id - self.start_layer``
        this replaces, by construction.

        Set ownership: the RANK of ``layer_id`` among the sorted owned ids. A
        layer this stage does not own is a caller bug, not a slot -- it is
        REFUSED rather than translated, because the subtraction's failure mode
        was to return a plausible index into ANOTHER layer's KV.
        """
        if self._local_slot_of is None:
            return layer_id - self.start_layer
        slot = self._local_slot_of.get(layer_id)
        if slot is None:
            raise KeyError(
                f"layer {layer_id} is not owned by this stage (owns "
                f"{sorted(self._local_slot_of)}); it has no local KV slot"
            )
        return slot

    def _finalize_allocation_log(self, num_tokens: int):
        """Common logging and mem_usage computation for KV cache allocation.
        Supports both tuple (K, V) size returns and single KV size returns.
        """
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"KV Cache is allocated. dtype: {self.dtype}, #tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"KV Cache is allocated. dtype: {self.dtype}, #tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    def get_kv_buffer_shape(self) -> Tuple[torch.Size, torch.Size]:
        k_buffer, v_buffer = self.get_kv_buffer(self.start_layer)
        return k_buffer.shape, v_buffer.shape

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError()

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool


def seam_chunk_and_retention(chunk_mib: int, swappable: bool, retain_env: str):
    """Resolve the #631 seam's commit chunk and handle retention SEPARATELY.

    Returns ``(commit_chunk_bytes_or_None, retain_handles)``.

    A COMMIT CHUNK IS NOW LOAD-BEARING ON ITS OWN, which is why the two
    came apart. Section 2.1's row-blocked seam calls ``commit_span`` /
    ``decommit_span``, and those RAISE on an unchunked arena
    (``KvVmmArena._require_chunk``): a monolithic per-buffer extent can
    only be released all-or-nothing, because ``cuMemUnmap`` takes whole
    mappings. So the streamed seam needs the chunk whether or not anyone
    wants retention.

    RETENTION DEFAULTS OFF, and this is a fix rather than a preference.
    Parked handles live in ``KvVmmArena._retained``, which is PER ARENA,
    and the flip's two layouts are two pools with two arenas. The PP
    arena parks precisely the pages the TP arena then asks the driver
    for, and the TP arena's ``_take_retained`` cannot see another arena's
    park. Both layouts stay resident, which defeats the exclusive backing
    the VA-backed pool exists to provide -- the pool sizing then has to
    fit both layouts at once. ``SGLANG_FLIP_SEAM_RETAIN_HANDLES=1``
    restores it for a single-arena user (the #330 grow/shrink dial does
    recycle its own handles).

    Only swappable pools qualify for either: a pool that never releases
    has nothing to park and nothing to stream.
    """
    if not swappable or int(chunk_mib) <= 0:
        return None, False
    retain = str(retain_env).strip().lower() in ("1", "true", "yes", "on")
    return (int(chunk_mib) << 20), retain


class MHATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
        kv_cache_layout: Optional[str] = None,
        post_capture_active: bool = False,
        vmm_commit_chunk_bytes: Optional[int] = None,
        swappable_backing: bool = False,
    ):
        if post_capture_active:
            # Reserved upper bound only (unbacked VA): page-align UP so
            # (size + page_size) % page_size == 0 holds for paged layouts.
            size = (size + page_size - 1) // page_size * page_size
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.post_capture_active = post_capture_active
        # #631: allocate on a VA reservation so the physical pages can be
        # swapped between the two phase layouts, WITHOUT the deferred
        # post-capture sizing (self.size is final at construction).
        self.swappable_backing = swappable_backing
        self._backing_released = False
        #: #631 waved seam: layer ordinals whose backing is currently
        #: unmapped. Empty = fully resident; a full set = fully released.
        self._released_layers = set()
        self._post_capture_owner = None
        # #330 dial: chunked physical commits make the tail releasable at
        # runtime; None keeps one handle per extension (stock post-capture).
        self._vmm_commit_chunk_bytes = vmm_commit_chunk_bytes
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )

        # Layout: NHD (default) | HND (SGLANG_USE_HND_KVCACHE) | vectorized_5d (ROCm AITER).
        # HND folds (page, head) into one paged index for per-kv-head sparse page tables
        # (paged backends like trtllm_mha consume directly). vectorized_5d SHUFFLE 5D:
        #   K: (num_blocks, H, D_k // X, page, X)  V: (num_blocks, H, page // X, D_v, X),
        #   X = 16 / dtype_bytes — AITER-only (ignored elsewhere, no consumer kernel).
        # HND and vectorized_5d are mutually exclusive; HND takes precedence.
        self.use_hnd = envs.SGLANG_USE_HND_KVCACHE.get()
        self.use_native_move_kv_cache = envs.SGLANG_NATIVE_MOVE_KV_CACHE.get()
        if kv_cache_layout is not None:
            # Explicit physical-layout selector wins over the platform default.
            # This is a label only; layouts that change buffer identity (e.g. the
            # page-granularity envelope) live in a dedicated pool subclass
            # (PageMajorMHATokenToKVPool) rather than in branches here.
            self.use_hnd = False
            self.kv_cache_layout = kv_cache_layout
        elif self.use_hnd:
            total_slots = self.size + self.page_size
            assert total_slots % self.page_size == 0, (
                f"HND KV cache needs (size+page_size) divisible by page_size, got "
                f"size={self.size}, page_size={self.page_size}"
            )
            self.num_pages = total_slots // self.page_size
            self.kv_cache_layout = "hnd"
        else:
            self.kv_cache_layout = "nhd"
            if _use_aiter:
                layout = envs.SGLANG_AITER_KV_CACHE_LAYOUT.get().lower()
                if layout not in ("nhd", "vectorized_5d"):
                    raise ValueError(
                        f"Unsupported SGLANG_AITER_KV_CACHE_LAYOUT={layout!r}; "
                        "expected 'nhd' or 'vectorized_5d'."
                    )
                self.kv_cache_layout = layout
                if layout == "vectorized_5d":
                    # X = 16 / storage itemsize: sized by the STORAGE dtype (not compute
                    # dtype) since it tiles the 16-byte on-pool vector.
                    self._kv_vector_x = 16 // self.store_dtype.itemsize
                    assert (self.size + self.page_size) % self.page_size == 0
                    assert self.page_size % self._kv_vector_x == 0, (
                        f"page_size={self.page_size} must be divisible by "
                        f"X={self._kv_vector_x} for vectorized_5d layout"
                    )
                    assert self.head_dim % self._kv_vector_x == 0
                    assert self.v_head_dim % self._kv_vector_x == 0

        self._create_buffers()
        # Diagnostic read-before-write falsifier (no-op unless the env is set).
        maybe_poison_pool_data(
            list(getattr(self, "k_buffer", []) or [])
            + list(getattr(self, "v_buffer", []) or []),
            "MHA KV pool",
        )

        self.device_module = torch.get_device_module(self.device)

        _use_alt_stream = _is_cuda or current_platform.is_cuda_alike()
        self.alt_stream = (
            self.device_module.Stream()
            if _use_alt_stream and enable_alt_stream
            else None
        )

        if enable_kv_cache_copy and not self.use_hnd:
            # The tiled byte copy assumes NHD slot-rows; HND uses a (page, off)
            # gather in move_kv_cache instead, so skip the slot-row copy config.
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        self._finalize_allocation_log(size)

        # for store_cache JIT kernel
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = self.head_dim == self.v_head_dim

    def _init_kv_copy_and_warmup(self):
        # Zero-layer pool (e.g. all-SWA model's full sub-pool) has no buffers.
        if self.layer_num == 0:
            self._kv_copy_config = None
            return

        # Heuristics for KV copy tiling
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        # Calculate num_locs_upper to avoid large Triton specialization (e.g. 8192)
        chunk_upper = 128 if bytes_per_tile >= _KV_COPY_TILE_SIZE_LARGE else 256

        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
            "num_locs_upper": chunk_upper,
        }

        dummy_loc = torch.zeros(chunk_upper, dtype=torch.int64, device=self.device)
        copy_all_layer_kv_cache_func(
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            chunk_upper,
            self._kv_copy_config,
        )

    def _create_buffers(self):
        if self.post_capture_active or self.swappable_backing:
            self._alloc_post_capture_buffers()
            if self.swappable_backing and not self.post_capture_active:
                # Sizing is already final; back the whole span now. Only the
                # BACKING is dynamic afterwards (#631 cross-phase swap).
                self._post_capture_owner.finalize(int(self.size))
        else:
            self._create_buffers_normal()
        self._kv_buffer_descs = self._build_kv_buffer_descs()
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _kv_buffer_shapes(self):
        """(k_shape, v_shape)"""
        if self.use_hnd:
            return (
                (self.num_pages, self.head_num, self.page_size, self.head_dim),
                (self.num_pages, self.head_num, self.page_size, self.v_head_dim),
            )
        rows = self.size + self.page_size
        return (
            (rows, self.head_num, self.head_dim),
            (rows, self.head_num, self.v_head_dim),
        )

    def _create_buffers_normal(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # The padded page (slot 0's page) absorbs dummy padded-token writes.
                if self.kv_cache_layout == "vectorized_5d":
                    total_slots = self.size + self.page_size
                    num_blocks = total_slots // self.page_size
                    x = self._kv_vector_x
                    # K: (num_blocks, H, D_k // X, page, X)
                    self.k_buffer = [
                        torch.zeros(
                            (
                                num_blocks,
                                self.head_num,
                                self.head_dim // x,
                                self.page_size,
                                x,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    # V: (num_blocks, H, page // X, D_v, X)
                    self.v_buffer = [
                        torch.zeros(
                            (
                                num_blocks,
                                self.head_num,
                                self.page_size // x,
                                self.v_head_dim,
                                x,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                else:
                    k_shape, v_shape = self._kv_buffer_shapes()
                    self.k_buffer = [
                        torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(v_shape, dtype=self.store_dtype, device=self.device)
                        for _ in range(self.layer_num)
                    ]

    # -- post-capture VA backing (opt-in; overridable per layout) --------------

    def _build_kv_buffer_descs(self):
        """Per-buffer layout descriptors, k0..k(L-1) then v0..v(L-1). Drives both the
        CUDA-VMM post-capture backing and PD-transfer registration
        (get_contiguous_buf_infos). Override per layout."""
        itemsize = self.store_dtype.itemsize
        # Derive from the real buffers when they exist (covers arbitrary layouts,
        # e.g. vectorized_5d); fall back to _kv_buffer_shapes for the pre-allocation
        # post-capture call, which only runs for NHD/HND.
        if getattr(self, "k_buffer", None) and getattr(self, "v_buffer", None):
            k_shape = tuple(self.k_buffer[0].shape)
            v_shape = tuple(self.v_buffer[0].shape)
        else:
            k_shape, v_shape = self._kv_buffer_shapes()
        # A row is a whole page when the leading dim is pages (hnd, vectorized_5d),
        # a single token slot for the plain NHD [slots, ...] layout.
        num_slots = self.size + self.page_size
        tokens_per_row = (
            self.page_size if k_shape[0] * self.page_size == num_slots else 1
        )
        descs = []
        for prefix, shape in (("k", k_shape), ("v", v_shape)):
            row_bytes = int(np.prod(shape[1:])) * itemsize
            for layer in range(self.layer_num):
                descs.append(
                    KvBufferDesc(
                        f"{prefix}{layer}",
                        shape,
                        row_bytes=row_bytes,
                        tokens_per_row=tokens_per_row,
                    )
                )
        return descs

    def _assign_post_capture_tensors(self, tensors):
        """Map owner tensors (in ``_build_kv_buffer_descs`` order) to k/v_buffer."""
        self.k_buffer = tensors[: self.layer_num]
        self.v_buffer = tensors[self.layer_num :]

    def _alloc_post_capture_buffers(self):
        dev = torch.device(self.device)
        device_id = dev.index if dev.index is not None else torch.cuda.current_device()
        # DEFAULT 8 MiB, MEASURED UNDER LOAD 2026-08-16 (#688).
        #
        # It was 0, which made the shipped row-blocking machinery INERT:
        # `_effective_row_blocks` returns 1 when the arena cannot do span ops,
        # and without a commit chunk it cannot. The 16-block default sat
        # unreachable in every fresh deployment.
        #
        # Priced by three boots of 7b706e8b89, same argv, same 4x25625-token
        # load, one variable, at matched live-slot counts:
        #
        #   chunk 0 -> 16   staging reserved per rank, 131 / ~25.7k slots
        #     PP0   1438.25 / 1510.21  ->  1061.53 / 1133.43   -377 MiB (-26%)
        #     PP1   1229.24 / 1301.20  ->   959.67 / 1031.58   -270 MiB (-22%)
        #     PP2   1870.46 / 1942.42  ->  1654.92 / 1726.83   -216 MiB (-12%)
        #
        # A CONSTANT OFFSET, not a slope: the same absolute saving at 131 and
        # at 25.7k live slots, which is what makes it worth a default -- it is
        # margin the seam gets back at every size. Corridor law warnings over
        # the same load fell 6 -> 3, and 216-377 MiB is the range the 06:47:48
        # wedge was short by (PP1 missed by 293).
        #
        # WHY 8 AND NOT 16. Measured separately, 8 -> 16 changes nothing:
        # slopes 0.00195 vs 0.00206 MiB/slot and intercepts within 3%, because
        # the arena chunk floor binds at both. The knob's own comment predicted
        # exactly this ("B=32 returns EXACTLY the B=16 numbers"). 8 is the
        # cheaper value that buys the whole shrink, and it is the value this
        # rig's operator env has been running all along -- so the default now
        # matches proven practice instead of trailing it.
        seam_chunk, retain = seam_chunk_and_retention(
            int(os.environ.get("SGLANG_FLIP_SEAM_CHUNK_MIB", "8") or 0),
            swappable=bool(self.swappable_backing),
            retain_env=os.environ.get("SGLANG_FLIP_SEAM_RETAIN_HANDLES", ""),
        )
        self._post_capture_owner = KvVmmBufferOwner(
            device=self.device,
            device_id=device_id,
            store_dtype=self.store_dtype,
            page_size=self.page_size,
            reserved_num_tokens=self.size,
            buffer_descs=self._build_kv_buffer_descs(),
            commit_chunk_bytes=self._vmm_commit_chunk_bytes or seam_chunk,
            retain_handles=retain,
        )
        self._assign_post_capture_tensors(self._post_capture_owner.tensors)

    def finalize_backing(self, config) -> None:
        """After capture+sizing: back the final span and set serving capacity.
        ``config`` is a MemoryPoolConfig (duck-typed); each pool family reads the
        fields it needs, so the finalizer stays pool-agnostic.

        #592: ``max_total_num_tokens`` IS the physical row count here, and that
        identity holds only at ``dcp_size == 1``. On the weighted uneven-DCP
        lane a rank holds ``dcp_compact_pool_rows(C, cp_S, ratio_r)`` rows for
        a global context of C -- which is why all five pool-CONSTRUCTION sites
        go through ``ModelRunner._dcp_token_sharded_pool_rows`` and this
        RESIZE path does not. It does not have to today, because
        ``post_capture_kv_sizing_planned()`` is False whenever ``dcp_size >
        1``; that gate is what makes the assumption true rather than merely
        usually true. Anyone lifting the gate (see the #592 note there) must
        translate the row count here first.
        """
        self._finalize_backing_tokens(config.max_total_num_tokens)

    def _finalize_backing_tokens(self, final_num_tokens: int) -> None:
        """Token-count primitive shared by composite pools (e.g. SWA sub-pools)."""
        self._post_capture_owner.finalize(final_num_tokens)
        self.size = int(final_num_tokens)

    # -- #631 cross-phase backing swap ---------------------------------------
    #
    # The phase flip runs TWO layouts over the SAME ranks but only ever one at
    # a time, and until now both layouts' KV pools held physical pages for the
    # whole process life. That is what halved the pool: each layout could only
    # be sized against half the per-rank budget.
    #
    # These two calls make the backing exclusive instead. The VA reservation --
    # and therefore every address a captured CUDA graph baked in -- is
    # untouched; only the physical pages behind it are unmapped and remapped,
    # which is precisely the property KvVmmBufferOwner exists for
    # ("addresses never move, so captured CUDA graphs keep replaying").
    #
    # SIZING IS NOT TOUCHED. ``self.size`` stays what the constructor computed
    # (already DCP-translated through ModelRunner._dcp_token_sharded_pool_rows),
    # so the #592 resize-translation gap on finalize_backing does not apply
    # here: the span released and restored is always this pool's OWN row count.

    def _layer_buffer_indices(self, layers):
        """Owner-buffer indices of a layer subset, or None for "all".

        ``_build_kv_buffer_descs`` emits ``k0..k(L-1)`` then ``v0..v(L-1)``,
        so layer ``i`` owns buffers ``i`` and ``layer_num + i``. Both halves
        move together: a layer whose K pages are mapped and whose V pages
        are not is a layout no reader can use.
        """
        if layers is None:
            return None
        indices = []
        for layer in layers:
            i = int(layer)
            if not (0 <= i < self.layer_num):
                raise ValueError(
                    f"layer {i} outside [0, {self.layer_num}) for this pool"
                )
            indices.append(i)
            indices.append(self.layer_num + i)
        return indices

    def release_backing(self, layers=None) -> int:
        """Unmap this pool's physical pages; return bytes handed back.

        Caller must hold an idle boundary for this pool: every row it owes
        the other layout must already have been READ (the flip's read region
        completes before its write region for exactly this reason), and no
        kernel may touch these rows until :meth:`restore_backing`.

        ``layers`` releases only that subset (#631 waved seam). The flip
        walks the seam one layer wave at a time so that the bytes staged
        between a release and its matching restore are one wave's share
        rather than the whole live set -- the term whose unboundedness
        wedged a single 270k-token request (HANDOFF_666).
        """
        if self._post_capture_owner is None:
            raise RuntimeError(
                "release_backing on a pool that was not allocated on a VA "
                "reservation; construct it with swappable_backing=True"
            )
        indices = self._layer_buffer_indices(layers)
        if indices is None:
            if self._backing_released:
                return 0
            # page_size is the floor the owner accepts; the residue is one
            # page plus at most one commit chunk per buffer, not the pool.
            released = self._post_capture_owner.shrink(self.page_size)
            self._released_layers = set(range(self.layer_num))
            self._backing_released = True
            return released
        wanted = [i for i in layers if int(i) not in self._released_layers]
        if not wanted:
            return 0
        released = self._post_capture_owner.shrink(
            self.page_size, self._layer_buffer_indices(wanted)
        )
        self._released_layers.update(int(i) for i in wanted)
        self._backing_released = len(self._released_layers) >= self.layer_num
        return released

    def restore_backing(self, layers=None) -> None:
        """Remap this pool's span at its unchanged addresses.

        ``layers`` restores only that subset; see :meth:`release_backing`.
        """
        if self._post_capture_owner is None:
            raise RuntimeError(
                "restore_backing on a pool that was not allocated on a VA "
                "reservation; construct it with swappable_backing=True"
            )
        self._post_capture_owner.finalize(
            int(self.size), self._layer_buffer_indices(layers)
        )
        if layers is None:
            self._released_layers = set()
        else:
            self._released_layers.difference_update(int(i) for i in layers)
        self._backing_released = len(self._released_layers) >= self.layer_num

    # -- #631 row-range backing, for the STREAMED seam ------------------------
    #
    # The waved seam releases a whole layer at a time, which bounds staging
    # by the wave count but leaves an irreducible transient of ONE
    # destination layer. In the tp_to_pp direction that layer spans the
    # FULL pool (a PP stage holds all tokens of its layers), i.e. 1.953 MiB
    # per 1000 pool tokens, and no ordering removes it: a peer cannot
    # release a source layer until its owner has written it, and the owner
    # cannot write until its destination layer is backed. These two calls
    # cut the unit from a layer to a ROW RANGE so the transient becomes a
    # block instead of a layer.
    #
    # A layer touched by either call is marked NOT fully backed until a
    # restore covers the whole pool, because ``backing_is_resident`` is
    # read as "may a kernel touch this pool" and a partially backed pool
    # must answer no.

    def release_backing_span(self, layers, lo_row: int, hi_row: int) -> int:
        """Hand back rows ``[lo_row, hi_row)`` of ``layers``. Rounds INWARD."""
        if self._post_capture_owner is None:
            raise RuntimeError(
                "release_backing_span on a pool that was not allocated on a "
                "VA reservation; construct it with swappable_backing=True"
            )
        indices = self._layer_buffer_indices(layers)
        released = self._post_capture_owner.release_token_span(
            int(lo_row), int(hi_row), indices
        )
        # Any partial release makes these layers non-resident; only a full
        # restore clears them again.
        self._released_layers.update(int(i) for i in layers)
        self._backing_released = len(self._released_layers) >= self.layer_num
        return released

    def restore_backing_span(self, layers, lo_row: int, hi_row: int) -> int:
        """Re-map rows ``[lo_row, hi_row)`` of ``layers``. Rounds OUTWARD.

        Deliberately does NOT clear ``_released_layers``: a span restore is
        one step of a stream, and the pool is only resident again once the
        whole span is back. ``restore_backing(layers)`` is what completes
        it.
        """
        if self._post_capture_owner is None:
            raise RuntimeError(
                "restore_backing_span on a pool that was not allocated on a "
                "VA reservation; construct it with swappable_backing=True"
            )
        indices = self._layer_buffer_indices(layers)
        return self._post_capture_owner.back_token_span(
            int(lo_row), int(hi_row), indices
        )

    @property
    def supports_backing_spans(self) -> bool:
        """Can the two span calls above actually run, or only be called?

        They need a CHUNKED arena (``KvVmmArena._require_chunk``); on an
        unchunked one they raise. The seam's gate asks THIS rather than
        ``hasattr``, because the raise would happen inside the flip's
        no-return region. See ``WavedBackingSwap.is_span_swappable``.
        """
        owner = self._post_capture_owner
        return bool(owner is not None and owner.has_commit_chunk)

    @property
    def backing_commit_chunk_bytes(self) -> int:
        """The arena's commit granule, 0 if it has none.

        The seam's staging reservation needs it as a FLOOR: a row block can
        never commit less than one chunk per buffer, however fine the
        blocking gets.
        """
        owner = self._post_capture_owner
        arena = getattr(owner, "_arena", None) if owner is not None else None
        return int(getattr(arena, "commit_chunk_bytes", 0) or 0)

    @property
    def backing_is_resident(self) -> bool:
        """False from the first layer released until the last is restored.

        PARTIAL is not resident. The waved seam (#631) unmaps one layer
        wave at a time, so between waves some layers are backed and some
        are not; every caller of this property is asking "may a kernel
        touch this pool", and the answer for a partially mapped pool is
        no.
        """
        return not self._released_layers

    @property
    def safe_zero_rows(self):
        """Rows of each K/V buffer a kernel may legally write, or None.

        None means "the whole tensor is backed" -- the ordinary,
        non-swappable case, where a caller should not pay for a slice.

        On a swappable pool the tensor spans the VA reservation and only the
        rows below the current watermark are mapped, so anything that writes
        the buffer wholesale (``zero_kv_data_buffers``) must stop here or
        fault. The minimum across buffers is deliberate: layouts differ in
        tokens-per-row, and zeroing FEWER rows than a given buffer allows
        leaves stale bytes in a region no sequence reads, while zeroing more
        touches unmapped memory and kills the process. Those two errors are
        not comparable, so the conservative one is chosen on purpose.
        """
        owner = self._post_capture_owner
        specs = getattr(owner, "_specs", None) if owner is not None else None
        if not specs:
            return None
        rows = []
        for spec in specs:
            desc = getattr(spec, "desc", None)
            if desc is None:
                return None
            rows.append(desc._rows(int(self.size) + int(self.page_size)))
        return min(rows) if rows else None

    @property
    def post_capture_backed_bytes(self) -> int:
        return self._post_capture_owner.backed_bytes if self._post_capture_owner else 0

    @property
    def uniform_backed_rows(self) -> int:
        """Rows backed in EVERY buffer. See KvVmmBufferOwner.uniform_backed_tokens."""
        owner = self._post_capture_owner
        return int(owner.uniform_backed_tokens) if owner is not None else 0

    @property
    def reserved_backing_rows(self) -> int:
        """The arena's IMMUTABLE row ceiling, or 0 when there is no arena.

        0 means "no reservation to clamp against", never "a reservation of
        zero": a pool with no VMM owner is not backed by an arena at all, so a
        caller must treat 0 as absent. See KvVmmBufferOwner.reserved_rows for
        why this has to be readable (#684).
        """
        owner = self._post_capture_owner
        return int(owner.reserved_rows) if owner is not None else 0

    @property
    def store_bound_rows(self) -> int:
        """Rows the K/V buffers physically address = the largest
        ``size + page_size`` this pool can ever reach (#352).

        Off the #330 dial lane the buffers are allocated at exactly
        ``size + page_size``, so this IS the live bound. On the dial lane it is
        the boot VA reservation, which ``KvVmmBufferOwner.finalize`` refuses to
        exceed -- which is what makes it safe for a CUDA graph to bake it in.
        """
        buffers = getattr(self, "k_buffer", None)
        if not buffers:
            return int(self.size) + int(self.page_size)
        buf = buffers[0]
        row_dim = int(self.row_dim)
        return int(buf.numel() // row_dim) if row_dim > 0 else int(buf.shape[0])

    def runtime_set_backing_tokens(self, num_tokens: int) -> int:
        """#330 dial: converge the VMM backing to ``num_tokens`` slots at
        runtime (grow maps + zeroes new rows; shrink decommits the tail back
        to the driver). Addresses never move, so captured CUDA graphs keep
        replaying. Returns the bytes actually RELEASED (0 on grow/no-op).
        Caller must hold an idle boundary and must have lowered the allocator
        ceiling first on shrink (rows above ``num_tokens`` must be dead).
        Newly grown rows are zeroed: a fresh boot's pools are torch.zeros and
        the flush identity (zero_kv_data_buffers rationale) must keep holding.
        """
        owner = self._post_capture_owner
        if owner is None:
            raise RuntimeError(
                "runtime_set_backing_tokens needs the VMM-backed pool "
                "(post_capture_active); this pool was allocated via "
                "torch.zeros and cannot release or grow physical backing."
            )
        prev = int(self.size)
        n = int(num_tokens)
        if n == prev:
            return 0
        if n > prev:
            owner.finalize(n)
            buffers = list(self.k_buffer) + list(self.v_buffer)
            for desc, buf in zip(self._kv_buffer_descs, buffers):
                lo = desc._rows(prev + self.page_size)
                hi = desc._rows(n + self.page_size)
                if hi > lo:
                    buf[lo:hi].zero_()
            self.size = n
            return 0
        released = owner.shrink(n)
        self.size = n
        return released

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        if self._post_capture_owner is not None:
            self._post_capture_owner.close()
            self._post_capture_owner = None

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def _pd_registerable_tensors(self):
        """Buffers to register for PD KV transfer, in ``_kv_buffer_descs`` order.
        Override when the registerable storage differs from k/v_buffer."""
        return self.k_buffer + self.v_buffer

    def get_contiguous_buf_infos(self):
        """(ptrs, lens, item_lens) for PD KV transfer, derived from the descriptors.
        ``lens`` is the final span at the CURRENT serving size -- for a post-capture
        pool that is the physically-backed span, not the reserved VA upper bound."""
        assert not self.use_hnd, (
            "PD-disaggregation KV transfer assumes NHD slot-row layout; "
            "HND KV cache (SGLANG_USE_HND_KVCACHE) is not supported with disagg yet."
        )
        tensors = self._pd_registerable_tensors()
        ptrs = [t.data_ptr() for t in tensors]
        lens = [
            d.final_span_bytes(self.size, self.page_size) for d in self._kv_buffer_descs
        ]
        item_lens = [d.item_len_bytes(self.page_size) for d in self._kv_buffer_descs]
        return ptrs, lens, item_lens

    def get_cpu_copy(self, indices, mamba_indices=None):
        assert not self.use_hnd, (
            "CPU KV offload indexes by slot (NHD); HND KV cache "
            "(SGLANG_USE_HND_KVCACHE) is not supported with CPU offload yet."
        )
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        assert not self.use_hnd, (
            "CPU KV offload indexes by slot (NHD); HND KV cache "
            "(SGLANG_USE_HND_KVCACHE) is not supported with CPU offload yet."
        )
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        current_platform.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[self.local_slot(layer_id)].view(self.dtype)
        return self.k_buffer[self.local_slot(layer_id)]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[self.local_slot(layer_id)].view(self.dtype)
        return self.v_buffer[self.local_slot(layer_id)]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        # Catch stale slot ids here instead of as illegal-addr / silent KV
        # corruption in the store_kvcache write (gated on SGLANG_ENABLE_ASYNC_ASSERT).
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA)")
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if dcp_kv_mask is not None:
            N, H, D = cache_k.shape
            k_buf = self.k_buffer[self.local_slot(layer_id)]
            v_buf = self.v_buffer[self.local_slot(layer_id)]
            # #355: the kernel addresses the buffer flat at ``loc * H * D``, so
            # its bound is the row count under exactly that stride -- taken from
            # the same helper the raw store_cache path uses so the two paths
            # cannot drift apart.
            masked_set_kv_buffer_kernel[(N,)](
                cache_k,
                cache_v,
                k_buf,
                v_buf,
                loc,
                dcp_kv_mask,
                kv_store_bound(self.size + self.page_size, k_buf, H * D),
                N,
                H,
                D,
                128,
                cache_k.stride(0),
                cache_k.stride(1),
                cache_v.stride(0),
                cache_v.stride(1),
                debug=kv_bound_check_enabled(),
            )
            return

        if self.use_hnd:
            # A slot is [page, :, off, :] (not a contiguous row), so scatter by (page, off).
            k_buf = self.k_buffer[self.local_slot(layer_id)]
            v_buf = self.v_buffer[self.local_slot(layer_id)]
            pages = loc // self.page_size
            offs = loc % self.page_size
            k_buf[pages, :, offs, :] = cache_k
            v_buf[pages, :, offs, :] = cache_v
            return

        self._store_kv_layer(self.local_slot(layer_id), loc, cache_k, cache_v)

    def _store_kv_layer(
        self,
        layer_idx: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        # Per-layer physical write into K/V buffer ``layer_idx``. Override for
        # layouts that change buffer identity (e.g. PageMajorMHATokenToKVPool's
        # 4-D strided views). ``loc`` and the cache tensors are already dtype-cast
        # and viewed as ``store_dtype`` by ``set_kv_buffer``.
        if self.kv_cache_layout == "vectorized_5d":
            # Late-import to keep the NHD path import-clean.
            from sglang.srt.layers.attention.utils import (
                launch_reshape_and_cache_shuffle_5d,
            )

            # The writer kernel uses key.stride(0) directly as the source
            # token stride; head/dim are assumed contiguous within each
            # token (stride(1)=head_size, stride(2)=1). Both hold for K/V
            # produced by QKV split + RoPE in upstream attention even when
            # the outer per-token stride is non-canonical, so we skip the
            # protective .contiguous() copies that would otherwise fire
            # large per-layer elementwise kernels.
            launch_reshape_and_cache_shuffle_5d(
                cache_k,
                cache_v,
                self.k_buffer[layer_idx],
                self.v_buffer[layer_idx],
                loc,
            )
            return

        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_idx],
            self.v_buffer[layer_idx],
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,
            # size + page_size = real slots + the reserved padding slot (padded /
            # dummy tokens write there); valid index range is [0, size + page_size).
            size_limit=self.size + self.page_size,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def set_kv_buffer_prefix_valid(
        self,
        layer: RadixAttention,
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id

        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got shape={tuple(loc_2d.shape)}.")
        if commit_lens.ndim != 1 or commit_lens.shape[0] != loc_2d.shape[0]:
            raise ValueError(
                "commit_lens must match loc_2d batch size: "
                f"{tuple(commit_lens.shape)=} {tuple(loc_2d.shape)=}."
            )

        num_rows = int(loc_2d.numel())
        if cache_k.shape[0] != num_rows or cache_v.shape[0] != num_rows:
            raise ValueError(
                "dense KV rows must match loc_2d size: "
                f"{tuple(cache_k.shape)=} {tuple(cache_v.shape)=} {tuple(loc_2d.shape)=}."
            )

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.contiguous().view(self.store_dtype)
            cache_v = cache_v.contiguous().view(self.store_dtype)
        else:
            cache_k = cache_k.contiguous()
            cache_v = cache_v.contiguous()

        if loc_2d.device != self.k_buffer[0].device:
            loc_2d = loc_2d.to(device=self.k_buffer[0].device, non_blocking=True)
        if commit_lens.device != self.k_buffer[0].device:
            commit_lens = commit_lens.to(
                device=self.k_buffer[0].device, non_blocking=True
            )
        if loc_2d.dtype != torch.int64:
            loc_2d = loc_2d.to(torch.int64)
        if commit_lens.dtype != torch.int32:
            commit_lens = commit_lens.to(torch.int32)

        if not (_is_cuda or _is_hip):
            row_offsets = torch.arange(loc_2d.shape[1], device=loc_2d.device)
            valid_mask = row_offsets[None, :] < commit_lens.to(torch.int64)[:, None]
            valid_idx = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                return
            self.set_kv_buffer(
                layer,
                loc_2d.reshape(-1).index_select(0, valid_idx),
                cache_k.index_select(0, valid_idx),
                cache_v.index_select(0, valid_idx),
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
            return

        _set_kv_buffer_prefix_valid_impl(
            cache_k,
            cache_v,
            self.k_buffer[self.local_slot(layer_id)],
            self.v_buffer[self.local_slot(layer_id)],
            loc_2d,
            commit_lens,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            # size + page_size = real slots + the reserved padding slot; the
            # helper widens it to the buffer's VA row count where they differ.
            size_limit=self.size + self.page_size,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # Zero-layer pool (e.g. all-SWA model's full sub-pool) has no buffers.
        if self.layer_num == 0:
            return

        # Catch stale indices here instead of as illegal-addr or silent KV corruption.
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")

        if self.use_hnd:
            pages_t, offs_t = tgt_loc // self.page_size, tgt_loc % self.page_size
            pages_s, offs_s = src_loc // self.page_size, src_loc % self.page_size
            for kb, vb in zip(self.k_buffer, self.v_buffer):
                kb[pages_t, :, offs_t, :] = kb[pages_s, :, offs_s, :]
                vb[pages_t, :, offs_t, :] = vb[pages_s, :, offs_s, :]
            return

        self._move_kv_cache_impl(tgt_loc, src_loc)

    def _move_kv_cache_impl(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # Physical move strategy. Override for layouts that change buffer identity
        # (e.g. PageMajorMHATokenToKVPool always uses the native move). The 3-D
        # per-layer buffers here ignore page_size in move_kv_cache_native.
        if self.use_native_move_kv_cache:
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
            return

        N = tgt_loc.numel()
        if N == 0:
            return

        assert (
            self._kv_copy_config is not None
        ), "KV copy not initialized. Set enable_kv_cache_copy=True in __init__"

        cfg = self._kv_copy_config
        cap = int(cfg.get("num_locs_upper", 256))

        if N <= cap:
            copy_all_layer_kv_cache_func(
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                N,
                next_power_of_2(N),
                cfg,
            )
            return

        # Huge N: chunk, but each chunk's upper is still pow2(<= cap)
        for start in range(0, N, cap):
            end = min(start + cap, N)
            chunk_len = end - start
            copy_all_layer_kv_cache_func(
                self.data_ptrs,
                self.data_strides,
                tgt_loc[start:end],
                src_loc[start:end],
                chunk_len,
                next_power_of_2(chunk_len),
                cfg,
            )


class NoOpMHATokenToKVPool(MHATokenToKVPool):
    """KV cache pool that skips physical K/V buffer allocation.

    Used in embedding-mode prefill-only workloads with the FA
    fa_skip_kv_cache path, where no layer reads or writes KV cache because
    attention uses raw K/V via flash_attn_varlen_func. Other prefill-only paths
    such as scoring/MIS may benefit from the same idea later, but some still
    stage K/V through paged cache today.

    This class keeps the scheduler's view of pool capacity (self.size is
    honored for admission) but allocates only (page_size, head_num, head_dim)
    placeholder tensors per layer to satisfy any code paths that dereference
    the buffers.

    Callers MUST ensure no real set_kv_buffer/get_*_buffer calls happen against
    this pool; those paths raise loudly so misuse is visible.
    """

    def _create_buffers(self):
        # No-op pool keeps tiny NHD placeholders regardless of SGLANG_USE_HND_KVCACHE
        # (no real KV is stored), so force NHD here to keep the store/move fast paths.
        self.use_hnd = False
        self.kv_cache_layout = "nhd"
        # Allocate minimal placeholder buffers. They exist purely so that code
        # paths holding `k_buffer` / `v_buffer` references (pointer tables,
        # layer-transfer counters, stride arithmetic) keep working without
        # None-guards scattered across the codebase. Shape is
        # [page_size, head_num, head_dim] per layer so that the unconditional
        # `key_cache.view(-1, page_size, head_num, head_dim)` in the FA backend
        # at the top of forward_extend succeeds regardless of --page-size.
        # Total footprint is still on the order of KB vs GBs for a real pool.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.v_head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _finalize_allocation_log(self, num_tokens: int):
        self.mem_usage = 0.0
        placeholder_bytes = (
            2
            * self.layer_num
            * self.page_size
            * self.head_num
            * max(self.head_dim, self.v_head_dim)
            * self.store_dtype.itemsize
        )
        logger.info(
            f"KV Cache skipped (no-op pool). Logical #tokens: {num_tokens}, "
            f"physical K/V size: ~{placeholder_bytes / 1024:.1f} KB placeholder"
        )

    def get_kv_size_bytes(self):
        # Report zero so downstream memory accounting matches reality.
        return (0, 0)

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError(
            "NoOpMHATokenToKVPool.set_kv_buffer was called. This pool is only "
            "valid in prefill-only modes (e.g. --is-embedding, scoring) with "
            "the FA backend's fa_skip_kv_cache path active; the attention "
            "backend must never write to it. Check that the workload truly "
            "performs no decode and that the FA backend's fa_skip_kv_cache "
            "preconditions are met."
        )

    def get_key_buffer(self, layer_id: int):
        # Return the placeholder. The FA backend reads this before taking the
        # fa_skip_kv_cache branch (which does not use it); the placeholder shape
        # is (page_size, head_num, head_dim) so downstream .view() calls succeed.
        return self.k_buffer[self.local_slot(layer_id)]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[self.local_slot(layer_id)]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # no-op; embedding mode has no KV cache to move
        return


class MHATokenToKVPoolFP4(MHATokenToKVPool):
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = self.head_num
                k = self.head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8
                self.k_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.k_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.k_buffer[self.local_slot(layer_id)].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.k_scale_buffer[self.local_slot(layer_id)]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k_nope_fp4_dequant = BlockFP4KVQuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant
        return self.k_buffer[self.local_slot(layer_id)]

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_v_nope_fp4 = self.v_buffer[self.local_slot(layer_id)].view(
                torch.uint8
            )
            cache_v_nope_fp4_sf = self.v_scale_buffer[self.local_slot(layer_id)]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_v_nope_fp4_dequant = BlockFP4KVQuantizeUtil.batched_dequantize(
                cache_v_nope_fp4, cache_v_nope_fp4_sf
            )
            return cache_v_nope_fp4_dequant
        return self.v_buffer[self.local_slot(layer_id)]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA-FP4)")
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k, cache_k_fp4_sf = BlockFP4KVQuantizeUtil.batched_quantize(cache_k)
            cache_v, cache_v_fp4_sf = BlockFP4KVQuantizeUtil.batched_quantize(cache_v)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

            cache_k_fp4_sf = cache_k_fp4_sf.view(self.store_dtype)
            cache_v_fp4_sf = cache_v_fp4_sf.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[self.local_slot(layer_id)][loc] = cache_k

            self.k_scale_buffer[self.local_slot(layer_id)][loc] = cache_k_fp4_sf
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[self.local_slot(layer_id)][loc] = cache_v

                self.v_scale_buffer[self.local_slot(layer_id)][loc] = cache_v_fp4_sf
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[self.local_slot(layer_id)][loc] = cache_k
            self.v_buffer[self.local_slot(layer_id)][loc] = cache_v

            self.k_scale_buffer[self.local_slot(layer_id)][loc] = cache_k_fp4_sf
            self.v_scale_buffer[self.local_slot(layer_id)][loc] = cache_v_fp4_sf


class PageMajorMHATokenToKVPool(MHATokenToKVPool):
    """MHA pool with the page-major (layer-major within a page) page-granularity envelope layout.

    All layers/slots share one contiguous ``uint8`` ``_raw`` buffer; per-layer K/V
    are 4-D strided views ``(num_pages, page_size, head_num, head_dim*)`` built by
    ``mem_cache/layout/page_major.py``. Token id ``t`` -> page ``t // page_size``,
    slot ``t % page_size``; the reserved padding slot 0 lives in page 0. At
    ``page_size == 1`` a page is a single slot (token-granularity envelope).

    Supported: the standard CUDA Triton attention + native move path. The tiled KV
    copy kernel, CPU offloading, and the spec-decode prefix-commit kernel all assume
    the per-layer contiguous 3-D layout; here they fail loudly rather than silently
    mis-indexing the strided views.
    """

    def __init__(
        self,
        *args,
        kv_cache_layout: Optional[str] = None,
        enable_kv_cache_copy: bool = False,
        **kwargs,
    ):
        assert kv_cache_layout in (
            None,
            "page_major_layer_major",
        ), f"PageMajorMHATokenToKVPool fixes its layout; got {kv_cache_layout!r}"
        # The tiled copy kernel assumes stride == row bytes, which the strided 4-D
        # views violate, so the copy path is never available here regardless of
        # what the caller requested (the spec-decode call sites pass
        # enable_kv_cache_copy=True). Always fall back to the native move.
        super().__init__(
            *args,
            kv_cache_layout="page_major_layer_major",
            enable_kv_cache_copy=False,
            **kwargs,
        )

    def _create_buffers(self):
        # One contiguous byte buffer holds all layers/slots; per-layer K/V are
        # 4-D strided views in the page-granularity envelope layout (see
        # mem_cache/layout/page_major.py).
        total_slots = self.size + self.page_size
        assert total_slots % self.page_size == 0, (
            f"page_major_layer_major needs (size + page_size) divisible by "
            f"page_size; got size={self.size}, page_size={self.page_size}"
        )
        num_pages = total_slots // self.page_size
        entry_bytes = mha_entry_bytes(
            layer_num=self.layer_num,
            head_num=self.head_num,
            head_dim=self.head_dim,
            v_head_dim=self.v_head_dim,
            itemsize=self.store_dtype.itemsize,
        )
        total_bytes = num_pages * self.page_size * entry_bytes
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # Unset slots read as zeros (matches the per-layer pool).
                self._raw = torch.zeros(
                    total_bytes, dtype=torch.uint8, device=self.device
                )
        self.k_buffer, self.v_buffer = build_page_major_mha_views(
            self._raw,
            layer_num=self.layer_num,
            head_num=self.head_num,
            head_dim=self.head_dim,
            v_head_dim=self.v_head_dim,
            store_dtype=self.store_dtype,
            page_size=self.page_size,
            num_pages=num_pages,
        )
        # stride(0) * itemsize is the per-page byte stride; for these strided
        # views np.prod(shape[1:]) would not equal it, so compute it directly.
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [x.stride(0) * x.dtype.itemsize for x in (self.k_buffer + self.v_buffer)],
            device=self.device,
        )

    def _store_kv_layer(
        self,
        layer_idx: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        # Single-launch Triton write into the 4-D envelope view. The parent's
        # view(-1, row_dim) path can't merge the strided 4-D dims.
        store_cache_4d(
            self.k_buffer[layer_idx],
            self.v_buffer[layer_idx],
            cache_k,
            cache_v,
            loc,
            page_size=self.page_size,
        )

    def _move_kv_cache_impl(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # Strided 4-D views: the tiled copy kernel assumes stride == row bytes, so
        # always take the native move (it splits token ids into
        # (page_id, slot_in_page) for the 4-D advanced index).
        move_kv_cache_native(
            self.k_buffer,
            self.v_buffer,
            tgt_loc,
            src_loc,
            page_size=self.page_size,
        )

    # The methods below assume the per-layer contiguous 3-D layout. The 4-D
    # strided envelope views have no per-layer contiguous region (their bytes are
    # interleaved layer-major within each page) and index page-major, not
    # token-major. Inheriting them would silently mis-index; fail loudly instead.

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "page-major layout has no per-layer contiguous regions; KV transfer / "
            "disaggregation is unsupported (TODO: expose the single _raw buffer "
            "with a page-aware transfer scheme)."
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError(
            "CPU offloading is unsupported under the page-major layout "
            "(TODO: split token ids into page/slot for the 4-D index)."
        )

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError(
            "CPU offloading is unsupported under the page-major layout "
            "(TODO: split token ids into page/slot for the 4-D index)."
        )

    def set_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise NotImplementedError(
            "prefix-valid commit is unsupported under the page-major layout "
            "(_set_kv_buffer_prefix_valid_impl assumes 3-D contiguous + row_dim)."
        )


class HybridLinearKVPool(KVCache):
    """KV cache with separate pools for full and linear attention layers."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        device: str,
        mamba_pool: MambaPool,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        # TODO: refactor mla related args
        use_mla: bool = False,
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        # When provided (shared-KV-pool path), use this pool for the
        # full-attention layers instead of constructing one internally.
        full_kv_pool: Optional[KVCache] = None,
        post_capture_active: bool = False,
        vmm_commit_chunk_bytes: Optional[int] = None,
        swappable_backing: bool = False,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self.head_num = head_num
        self.head_dim = head_dim
        self.mamba_pool = mamba_pool
        # virtual->physical mamba-slot translate for the HiCache offload path;
        # identity for a static pool, the allocator's `translate` for the unified pool.
        self._mamba_translate = lambda ids: ids
        self.use_mla = use_mla
        if full_kv_pool is not None:
            # Shared-KV-pool path: the caller built a UnifiedMHATokenToKVPool
            # aliasing the shared byte buffer.
            self.full_kv_pool = full_kv_pool
        elif not use_mla:
            TokenToKVPoolClass = MHATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mha_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMHATokenToKVPool
            elif full_kv_pool_class is not None:
                # Caller-selected MHA layout variant (e.g. the page-major
                # PageMajorMHATokenToKVPool). NPU / out-of-tree classes keep
                # priority since they don't understand alternate layouts.
                TokenToKVPoolClass = full_kv_pool_class

            post_capture_kwargs = (
                {
                    "post_capture_active": True,
                    "vmm_commit_chunk_bytes": vmm_commit_chunk_bytes,
                }
                if post_capture_active
                else {}
            )
            # #631: VA-backed so the two phase layouts can hold physical
            # pages exclusively. Independent of post-capture SIZING, which
            # is gated off for this config anyway.
            if swappable_backing:
                post_capture_kwargs["swappable_backing"] = True
            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=enable_memory_saver,
                enable_kv_cache_copy=enable_kv_cache_copy,
                **post_capture_kwargs,
            )
        else:
            TokenToKVPoolClass = MLATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mla_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMLATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                layer_num=self.full_layer_nums,
                device=device,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                enable_memory_saver=enable_memory_saver,
            )
        # The sub-pool is addressed with the DENSE index below, not a global
        # layer id, so it must not carry a global ownership map.
        mark_as_sub_pool(self.full_kv_pool)
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        if use_mla:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    @property
    def post_capture_active(self) -> bool:
        return getattr(self.full_kv_pool, "post_capture_active", False)

    # -- #631 cross-phase backing swap, forwarded to the full-attention pool --
    #
    # Only the full-attention KV rows are swapped here. The mamba/GDN state
    # pool lives on req_to_token_pool and is handled separately.

    def release_backing(self, layers=None) -> int:
        return self.full_kv_pool.release_backing(layers)

    def restore_backing(self, layers=None) -> None:
        self.full_kv_pool.restore_backing(layers)

    # THE SPAN SURFACE MUST BE FORWARDED TOO, and its absence is why the
    # streamed seam (#631 section 2.1) was dead code on every boot this rig
    # has taken: the object the flip actually holds is this wrapper, not the
    # full-attention pool, so ``is_span_swappable`` looked straight past the
    # capability and silently took the whole-wave branch. A capability probe
    # that a wrapper can drop is a capability that turns itself off.

    def release_backing_span(self, layers, lo_row: int, hi_row: int) -> int:
        return self.full_kv_pool.release_backing_span(layers, lo_row, hi_row)

    def restore_backing_span(self, layers, lo_row: int, hi_row: int) -> int:
        return self.full_kv_pool.restore_backing_span(layers, lo_row, hi_row)

    @property
    def supports_backing_spans(self) -> bool:
        return bool(getattr(self.full_kv_pool, "supports_backing_spans", False))

    @property
    def backing_commit_chunk_bytes(self) -> int:
        return int(getattr(self.full_kv_pool, "backing_commit_chunk_bytes", 0))

    @property
    def backing_is_resident(self) -> bool:
        return getattr(self.full_kv_pool, "backing_is_resident", True)

    @property
    def backed_bytes(self) -> int:
        """Physically committed bytes -- the number the exclusive-backing
        falsifier asserts on, so it measures allocation rather than intent."""
        return getattr(self.full_kv_pool, "post_capture_backed_bytes", 0)

    @property
    def post_capture_backed_bytes(self) -> int:
        return getattr(self.full_kv_pool, "post_capture_backed_bytes", 0)

    @property
    def uniform_backed_rows(self) -> int:
        return int(getattr(self.full_kv_pool, "uniform_backed_rows", 0))

    @property
    def reserved_backing_rows(self) -> int:
        """#684: the full-attention arena's immutable row ceiling, or 0.

        Forwarded for the same reason ``uniform_backed_rows`` is: the relief
        holds THIS pool, and the number lives one layer down.
        """
        return int(getattr(self.full_kv_pool, "reserved_backing_rows", 0))

    def finalize_backing(self, config) -> None:
        # Only the attention KV is resized; the mamba state cache is fixed pre-capture.
        self.full_kv_pool._finalize_backing_tokens(config.max_total_num_tokens)
        self.size = int(config.max_total_num_tokens)

    def initial_backing_rows(self, rows: int) -> None:
        """#330 dial lane: back the full-attention sub-pool to its boot row
        count right after construction (the ctor reserved the VA upper bound
        only). Does NOT touch ``self.size`` — on this pool family the hybrid
        ``size`` keeps the stock rows semantics of the non-VMM ctor path."""
        self.full_kv_pool._finalize_backing_tokens(rows)
        # Fresh VMM pages are relied on to read as zeros elsewhere in this
        # lane; make it explicit for the whole boot span so the dial boot is
        # byte-identical to the stock torch.zeros construction.
        for buf in list(self.full_kv_pool.k_buffer) + list(self.full_kv_pool.v_buffer):
            hi = rows + self.page_size
            buf[:hi].zero_()

    def runtime_set_backing_rows(self, rows: int) -> int:
        """#330 dial: converge the full-attention sub-pool's physical backing
        to ``rows`` at runtime; returns bytes released to the driver."""
        return self.full_kv_pool.runtime_set_backing_tokens(rows)

    @property
    def vmm_backed_bytes(self) -> int:
        return self.post_capture_backed_bytes

    @property
    def full_pool_backed_rows(self) -> int:
        return int(self.full_kv_pool.size)

    @property
    def store_bound_rows(self) -> int:
        """Rows the full-attention KV buffers physically address (#352).

        The lifetime ceiling of every ``store_kvcache`` bound this pool can
        ever pass -- including the ones a CUDA graph baked in at capture time.
        The dial's growth must stay at or below it; see
        ``graph_safe_store_bound``.
        """
        return int(self.full_kv_pool.store_bound_rows)

    def get_kv_size_bytes(self):
        return self.full_kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self):
        mamba_data_ptrs, mamba_data_lens, mamba_item_lens = (
            self.mamba_pool.get_contiguous_buf_infos()
        )
        return mamba_data_ptrs, mamba_data_lens, mamba_item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each mamba state tensor."""
        return self.mamba_pool.get_state_dim_per_tensor()

    def get_state_dim_offsets(self):
        """Uneven-TP slice start offsets, parallel to get_state_dim_per_tensor
        (None when no uneven plan is active). See MambaPool.get_state_dim_offsets."""
        return self.mamba_pool.get_state_dim_offsets()

    def get_conv_subblock_spec(self):
        """Conv-state [q|k|v] sub-block structure for TP-mismatched PD transfer.
        See MambaPool.get_conv_subblock_spec."""
        return self.mamba_pool.get_conv_subblock_spec()

    def get_conv_transfer_segments(self):
        """This rank's conv-state [q|k|v] sub-block transfer segments.
        See MambaPool.get_conv_transfer_segments."""
        return self.mamba_pool.get_conv_transfer_segments()

    def maybe_get_custom_mem_pool(self):
        return self.full_kv_pool.maybe_get_custom_mem_pool()

    def _transfer_full_attention_id(self, layer_id: int):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter
        # The layer-wise wait logic is executed at the Hybrid LinearPool level;
        # no additional wait is needed in the full_kv_pool
        self.full_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))

    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    @contextmanager
    def _transfer_id_context(self, layer: RadixAttention):
        @contextmanager
        def _patch_layer_id(layer):
            original_layer_id = layer.layer_id
            layer.layer_id = self._transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        # Write-location info lives in the metadata (`KVWriteLoc`). `full_loc` is the
        # unified pool's pre-translated PHYSICAL loc (None for a static pool, where
        # `loc` is already physical) — either way the pool writes a PHYSICAL loc.
        loc, _, full_loc = unwrap_write_loc(loc)
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        if not self.use_mla:
            write_loc = full_loc if full_loc is not None else loc
            self.full_kv_pool.set_kv_buffer(
                None,
                write_loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
                dcp_kv_mask=dcp_kv_mask,
            )
        else:
            with self._transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    loc,
                    cache_k,
                    cache_v,
                )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def get_cpu_copy(self, indices, mamba_indices=None):
        kv_cpu = self.full_kv_pool.get_cpu_copy(indices)
        # mamba_pool stores PHYSICAL ids; translate the (unified-pool virtual) ids first.
        mamba_cpu = (
            self.mamba_pool.get_cpu_copy(self._mamba_translate(mamba_indices))
            if mamba_indices is not None
            else None
        )
        return kv_cpu, mamba_cpu

    def load_cpu_copy(self, cache_cpu, indices, mamba_indices=None):
        kv_cpu, mamba_cpu = cache_cpu
        self.full_kv_pool.load_cpu_copy(kv_cpu, indices)
        if mamba_cpu is not None and mamba_indices is not None:
            self.mamba_pool.load_cpu_copy(
                mamba_cpu, self._mamba_translate(mamba_indices)
            )

    def get_v_head_dim(self):
        return self.full_kv_pool.get_value_buffer(0).shape[-1]

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_dsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_dsa = use_dsa
        self.dsa_kv_cache_store_fp8 = (
            use_dsa
            and dtype == torch.float8_e4m3fn
            and override_kv_cache_dim is not None
        )
        # When override_kv_cache_dim is provided with dsa model, we assume the
        # override kv cache dim is correct and use it directly.
        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.dsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        self._create_buffers()

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_dsa:
            # DSA will allocate indexer KV cache later and then log the total size
            self._finalize_allocation_log(size)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))

        if self.store_dtype != self.dtype:
            return self.kv_buffer[self.local_slot(layer_id)].view(self.dtype)

        return self.kv_buffer[self.local_slot(layer_id)]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))

        if self.store_dtype != self.dtype:
            return self.kv_buffer[self.local_slot(layer_id)][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[self.local_slot(layer_id)][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA)")
        layer_id = layer.layer_id
        assert not self.dsa_kv_cache_store_fp8
        parallel = get_parallel()
        if parallel.dcp_enabled:
            valid_mask = loc % parallel.attn_dcp_size == parallel.attn_dcp_rank
            if not valid_mask.all():
                loc = loc[valid_mask]
                cache_k = cache_k[valid_mask]
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            self.kv_buffer[self.local_slot(layer_id)][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[self.local_slot(layer_id)][loc] = cache_k

    def _write_mla_kv_buffer(
        self,
        dst_buffer: torch.Tensor,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        if _is_hip and self.use_dsa and self.dtype == fp8_dtype:
            # HIP FP8 path uses raw MLA KV layout (nope + rope) without per-block scales.
            # Fuse BF16/FP16 -> FP8 cast with paged KV write.
            set_mla_kv_buffer_triton_fp8_quant(
                dst_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
                fp8_dtype,
            )
        elif self.dsa_kv_cache_store_fp8:
            # OPTIMIZATION: Quantize k_nope and k_rope separately to avoid concat overhead
            # This also enables reuse of set_mla_kv_buffer_triton two-tensor write path
            # quantize_k_cache_separate returns (nope_part, rope_part) as uint8 bytes
            cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                cache_k_nope, cache_k_rope
            )

            # Reuse existing two-tensor write kernel (works with FP8 byte layout)
            # cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512) | scales(16)]
            # cache_k_rope_fp8: (num_tokens, 1, 128) uint8 [rope_bf16_bytes(128)]
            set_mla_kv_buffer_triton(
                dst_buffer,
                loc,
                cache_k_nope_fp8,
                cache_k_rope_fp8,
            )
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                dst_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
            )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA)")
        layer_id = layer.layer_id
        self._write_mla_kv_buffer(
            self.kv_buffer[self.local_slot(layer_id)],
            loc,
            cache_k_nope,
            cache_k_rope,
        )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Relocate accepted-token combined MLA KV (latent + rope) per layer."""
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")

        if tgt_loc.numel() == 0:
            return

        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        for kv_cache in self.kv_buffer:
            kv_cache[tgt_loc_flat] = kv_cache[src_loc_flat]

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        current_platform.synchronize()


class MLATokenToKVPoolFP4(MLATokenToKVPool):
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = 1  # head_num
                k = self.kv_cache_dim  # head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8

                self.kv_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.kv_scale_buffer = [
                    torch.zeros(
                        (m, k // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer
        del self.kv_scale_buffer

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))

        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.kv_buffer[self.local_slot(layer_id)].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.kv_scale_buffer[self.local_slot(layer_id)]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k_nope_fp4_dequant = BlockFP4KVQuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant

        return self.kv_buffer[self.local_slot(layer_id)]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        # loc_info may be a KVWriteLoc; MLA pools have no SWA target.
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA-FP4)")
        layer_id = layer.layer_id
        assert not self.dsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k_fp4, cache_k_fp4_sf = BlockFP4KVQuantizeUtil.batched_quantize(
                cache_k
            )

        if self.store_dtype != self.dtype:
            self.kv_buffer[self.local_slot(layer_id)][loc] = cache_k_fp4.view(
                self.store_dtype
            )
            self.kv_scale_buffer[self.local_slot(layer_id)][loc] = cache_k_fp4_sf.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[self.local_slot(layer_id)][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        maybe_detect_oob(
            loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA-FP4)"
        )
        layer_id = layer.layer_id

        if self.dsa_kv_cache_store_fp8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            # TODO no need to cat
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[self.local_slot(layer_id)][loc] = cache_k
        else:
            if cache_k_nope.dtype != self.dtype:
                from sglang.srt.layers.quantization.kvfp4_tensor import (
                    BlockFP4KVQuantizeUtil,
                )

                cache_k_nope_fp4, cache_k_nope_fp4_sf = (
                    BlockFP4KVQuantizeUtil.batched_quantize(cache_k_nope)
                )
                cache_k_rope_fp4, cache_k_rope_fp4_sf = (
                    BlockFP4KVQuantizeUtil.batched_quantize(cache_k_rope)
                )

            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[self.local_slot(layer_id)],
                loc,
                cache_k_nope_fp4,
                cache_k_rope_fp4,
            )
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[self.local_slot(layer_id)],
                loc,
                cache_k_nope_fp4_sf,
                cache_k_rope_fp4_sf,
            )


class DSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8
    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16

    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_buf_size: Optional[int] = None,
    ):
        override_dim = (
            kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
        )

        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_dsa=True,
            override_kv_cache_dim=override_dim,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim
        if index_buf_size is None:
            index_buf_size = size
        self.index_buf_size = index_buf_size
        # num head == 1 and head dim == 128 for index_k in DSA
        assert index_head_dim == 128

        if _is_hip:
            if aiter_can_use_preshuffle_paged_mqa():
                assert (
                    self.page_size % 16 == 0
                ), f"HIP preshuffle requires page_size to be a multiple of 16, got {self.page_size}"
            else:
                assert (
                    self.page_size == 1
                ), f"HIP legacy DSA path requires page_size == 1, got {self.page_size}"
        else:
            assert self.page_size == 64
        self._create_index_buffers()
        self._finalize_allocation_log(size)

    def _index_buffer_shape(self, num_pages: int) -> tuple[int, int]:
        return (
            num_pages,
            self.page_size
            * (self.index_head_dim + self.index_head_dim // self.quant_block_size * 4),
        )

    def _create_index_buffers(self):
        num_pages = (self.index_buf_size + self.page_size + 1) // self.page_size
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    # Layout:
                    #     ref: test_attention.py :: kv_cache_cast_to_fp8
                    #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                    #     data: for page i,
                    #         * buf[i, :page_size * head_dim] for fp8 data
                    #         * buf[i, page_size * head_dim:].view(float32) for scale
                    self._index_buffer_shape(num_pages),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def _clear_buffers(self):
        super()._clear_buffers()
        del self.index_k_with_scale_buffer

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Move latent KV and the DSA indexer cache (key + scale) in lockstep."""
        super().move_kv_cache(tgt_loc, src_loc)

        if tgt_loc.numel() == 0:
            return

        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        for index_k in self.index_k_with_scale_buffer:
            index_k[tgt_loc_flat] = index_k[src_loc_flat]

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))
        return self.index_k_with_scale_buffer[self.local_slot(layer_id)]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))
        buf = self.index_k_with_scale_buffer[self.local_slot(layer_id)]
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))
        buf = self.index_k_with_scale_buffer[self.local_slot(layer_id)]
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        """
        Fused method to get both index K and scale data in a single call using Triton.
        More efficient than calling get_index_k_continuous and get_index_k_scale_continuous separately.

        :param layer_id: Layer index
        :param seq_len: Sequence length
        :param page_indices: Page indices tensor
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))
        buf = self.index_k_with_scale_buffer[self.local_slot(layer_id)]
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[self.local_slot(layer_id)]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        # DSA keeps a page-indexed index_k_with_scale_buffer alongside kv_buffer.
        # Retract frees the slots/pages and they get reused by other reqs'
        # set_index_k_scale_buffer, so we must offload it here too -- otherwise
        # resume restores kv_buffer but leaves foreign index/scale in place and
        # DSA attention reads garbage at those token positions.
        kv_cache_cpu = super().get_cpu_copy(indices, mamba_indices=mamba_indices)

        page_indices = indices[:: self.page_size] // self.page_size
        torch.cuda.synchronize()
        index_k_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.layer_num):
            index_k_cpu.append([])
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = self.index_k_with_scale_buffer[layer_id][
                    chunk_page_indices
                ].to("cpu", non_blocking=True)
                index_k_cpu[-1].append(idx_cpu)
        torch.cuda.synchronize()

        return {"kv": kv_cache_cpu, "index_k": index_k_cpu}

    def load_cpu_copy(self, kv_cache_cpu_dict, indices, mamba_indices=None):
        super().load_cpu_copy(
            kv_cache_cpu_dict["kv"], indices, mamba_indices=mamba_indices
        )

        page_indices = indices[:: self.page_size] // self.page_size
        index_k_cpu = kv_cache_cpu_dict["index_k"]
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.layer_num):
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cpu[layer_id][i // page_chunk_size]
                assert idx_cpu.shape[0] == len(chunk_page_indices)
                idx_chunk = idx_cpu.to(
                    self.index_k_with_scale_buffer[0].device, non_blocking=True
                )
                self.index_k_with_scale_buffer[layer_id][chunk_page_indices] = idx_chunk
        torch.cuda.synchronize()

    def get_state_buf_infos(self):
        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)
        ]
        item_lens = [
            self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes


def move_kv_cache_native(
    k_buffer: List[torch.Tensor],
    v_buffer: List[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
    page_size: int = 1,
):
    """Move token-granular K/V rows from ``src_loc`` to ``tgt_loc``.

    Supports two buffer shapes:

    - 3-D ``[max_slots, head_num, head_dim]`` (per-layer pool): direct advanced
      indexing on dim 0; ``page_size`` is ignored.
    - 4-D ``[num_pages, page_size, head_num, head_dim]`` (envelope layout): split
      each token id into ``(page_id, slot_in_page)`` and use 2-D advanced
      indexing. PyTorch resolves the strided byte address via the view's strides.
    """
    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        if k_cache.ndim == 4:
            if page_size == 1:
                # Degenerate (num_pages, 1, head, dim): token id == page id.
                k_cache[tgt_loc_flat, 0] = k_cache[src_loc_flat, 0]
                v_cache[tgt_loc_flat, 0] = v_cache[src_loc_flat, 0]
            else:
                tgt_page = tgt_loc_flat // page_size
                tgt_tok = tgt_loc_flat % page_size
                src_page = src_loc_flat // page_size
                src_tok = src_loc_flat % page_size
                k_cache[tgt_page, tgt_tok] = k_cache[src_page, src_tok]
                v_cache[tgt_page, tgt_tok] = v_cache[src_page, src_tok]
        else:
            k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
            v_cache[tgt_loc_flat] = v_cache[src_loc_flat]


@triton.jit
def masked_set_kv_buffer_kernel(
    k_ptr,
    v_ptr,
    k_buffer_ptr,
    v_buffer_ptr,
    loc_ptr,
    mask_ptr,
    bound,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    k_stride_B: tl.constexpr,
    k_stride_H: tl.constexpr,
    v_stride_B: tl.constexpr,
    v_stride_H: tl.constexpr,
):
    """Owner-masked KV write: rank writes row ``pid`` iff ``mask[pid]``.

    ``bound`` (#355) is the number of rows this kernel may address, i.e.
    ``k_buffer.numel() // (H * D)``, supplied by ``kv_store_bound`` -- the same
    helper the raw ``store_cache`` path uses, so the two bounds cannot drift.
    It is a BY-VALUE scalar and therefore frozen into any CUDA graph that
    captures this launch, which is why it has to be the buffer's VA row count
    (stable for the graph's whole lifetime) and not the pool's live ``size``;
    see ``graph_safe_store_bound`` for the full argument.

    The check is a scalar compare against a register, guarding a write of
    ``H * D`` elements -- but it is a ``tl.device_assert`` and therefore only
    lowered when the launch passes ``debug=True`` (``kv_bound_check_enabled``,
    default on). Without it an escaped compact row corrupts a live KV row
    SILENTLY; with it the process dies naming this kernel.

    Placement note: the assert sits after two early returns, and both are
    BLOCK-UNIFORM (``pid >= N`` and a scalar mask load, identical for every
    thread in the program). That matters because Triton's assert lowering may
    emit a barrier, and a barrier reached by only some threads of a block
    hangs. Any future early return above this line has to stay uniform.
    """
    pid = tl.program_id(0)
    if pid >= N:
        return

    do_write = tl.load(mask_ptr + pid) != 0
    if not do_write:
        return

    loc = tl.load(loc_ptr + pid)
    tl.device_assert(
        (loc >= 0) & (loc < bound), "masked_set_kv_buffer: loc out of range"
    )
    total = H * D
    num_chunks = tl.cdiv(total, CHUNK)

    for c in range(num_chunks):
        offs = tl.arange(0, CHUNK)
        idx = c * CHUNK + offs
        mask = idx < total
        row = idx // D
        col = idx % D

        key = tl.load(k_ptr + pid * k_stride_B + row * k_stride_H + col, mask=mask)
        tl.store(k_buffer_ptr + loc * H * D + idx, key, mask=mask)

        value = tl.load(v_ptr + pid * v_stride_B + row * v_stride_H + col, mask=mask)
        tl.store(v_buffer_ptr + loc * H * D + idx, value, mask=mask)


class MHATokenToKOnlyPool(KVCache):
    """K-only pool for MiniMax sparse layers whose index branch never reads V
    (``sparse_disable_index_value``); allocating V would waste memory."""

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                self.k_buffer = [
                    torch.zeros(
                        (size + page_size, head_num, head_dim),
                        dtype=self.store_dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]
        self._finalize_allocation_log(size)

    def _get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[self.local_slot(layer_id)].view(self.dtype)
        return self.k_buffer[self.local_slot(layer_id)]

    def register_layer_transfer_counter(
        self, layer_transfer_counter: LayerDoneCounter
    ) -> None:
        self.layer_transfer_counter = layer_transfer_counter

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))
        return self._get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("MHATokenToKOnlyPool does not allocate V")

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("MHATokenToKOnlyPool does not allocate V")

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ) -> None:
        # Routed through MiniMaxSparseKVPool.set_index_k_buffer instead.
        raise NotImplementedError(
            "MHATokenToKOnlyPool: use set_index_k_buffer on the parent "
            "MiniMaxSparseKVPool — this pool does not store V"
        )

    def get_kv_size_bytes(self):
        k_size_bytes = sum(get_tensor_size_bytes(k) for k in self.k_buffer)
        return k_size_bytes, 0


class MiniMaxSparseKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        idx_head_dim: int,
        dense_layer_ids: List[int],
        sparse_layer_ids: List[int],
        device: str,
        disable_value_sparse_layer_ids: Optional[List[int]] = None,
        enable_memory_saver: bool = False,
        index_dtype: Optional[torch.dtype] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        # Do not call super().__init__() — delegate to sub-pools instead.
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.use_minimax_fused_kv_index_store = (
            envs.SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE.get()
        )

        local_dense_layer_ids = [
            lid for lid in dense_layer_ids if start_layer <= lid < end_layer
        ]
        local_sparse_layer_ids = [
            lid for lid in sparse_layer_ids if start_layer <= lid < end_layer
        ]

        index_dtype = index_dtype if index_dtype is not None else dtype

        # Split sparse layers by V policy: kv_sparse (index_kv_pool holds K+V) vs
        # k_only_sparse (index_k_pool holds only K; V is never read).
        disable_set = set(disable_value_sparse_layer_ids or [])
        local_kv_sparse_layer_ids = [
            g for g in local_sparse_layer_ids if g not in disable_set
        ]
        local_k_only_sparse_layer_ids = [
            g for g in local_sparse_layer_ids if g in disable_set
        ]

        # Membership check across all sparse layers, regardless of split.
        self.sparse_layer_id_mapping: dict[int, int] = {
            gid: i for i, gid in enumerate(local_sparse_layer_ids)
        }
        # Per-sub-pool local indices.
        self.index_kv_layer_id_mapping: dict[int, int] = {
            gid: i for i, gid in enumerate(local_kv_sparse_layer_ids)
        }
        self.index_k_layer_id_mapping: dict[int, int] = {
            gid: i for i, gid in enumerate(local_k_only_sparse_layer_ids)
        }

        self.main_pool = MHATokenToKVPool(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=len(local_dense_layer_ids) + len(local_sparse_layer_ids),
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.index_kv_pool: Optional[MHATokenToKVPool] = (
            MHATokenToKVPool(
                size=size,
                page_size=page_size,
                dtype=index_dtype,
                head_num=1,
                head_dim=idx_head_dim,
                layer_num=len(local_kv_sparse_layer_ids),
                device=device,
                enable_memory_saver=enable_memory_saver,
            )
            if local_kv_sparse_layer_ids
            else None
        )

        self.index_k_pool: Optional[MHATokenToKOnlyPool] = (
            MHATokenToKOnlyPool(
                size=size,
                page_size=page_size,
                dtype=index_dtype,
                head_num=1,
                head_dim=idx_head_dim,
                layer_num=len(local_k_only_sparse_layer_ids),
                device=device,
                enable_memory_saver=enable_memory_saver,
            )
            if local_k_only_sparse_layer_ids
            else None
        )

        self.mem_usage = self.main_pool.mem_usage
        if self.index_kv_pool is not None:
            self.mem_usage += self.index_kv_pool.mem_usage
        if self.index_k_pool is not None:
            self.mem_usage += self.index_k_pool.mem_usage

        # HiCacheController reads these from the top-level KV pool wrapper.
        self.layer_num = self.main_pool.layer_num
        self.start_layer = self.main_pool.start_layer
        self.end_layer = self.main_pool.end_layer
        # PD disaggregation reads these directly (no fallback) off the wrapper.
        self.head_num = self.main_pool.head_num
        self.head_dim = self.main_pool.head_dim
        self.layer_transfer_counter = None

    def register_layer_transfer_counter(
        self, layer_transfer_counter: LayerDoneCounter
    ) -> None:
        self.layer_transfer_counter = layer_transfer_counter

    def _wait_for_layer(self, layer_id: int) -> None:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.local_slot(layer_id))

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.main_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.main_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._wait_for_layer(layer_id)
        return self.main_pool.get_kv_buffer(layer_id)

    def get_index_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._wait_for_layer(layer_id)
        mapped_id = self.index_kv_layer_id_mapping.get(layer_id)
        if mapped_id is None:
            raise ValueError(
                f"layer_id={layer_id} does not have an index V cache "
                f"(either dense, or in the K-only group). "
                f"index_kv layers: {list(self.index_kv_layer_id_mapping.keys())}"
            )
        return self.index_kv_pool.get_kv_buffer(mapped_id)

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        # First try the K-only pool; fall back to the index_kv pool's K side
        # so callers that just need K work for both sparse subgroups.
        mapped_id = self.index_k_layer_id_mapping.get(layer_id)
        if mapped_id is not None:
            return self.index_k_pool.get_key_buffer(mapped_id)
        mapped_id = self.index_kv_layer_id_mapping.get(layer_id)
        if mapped_id is not None:
            return self.index_kv_pool.get_key_buffer(mapped_id)
        raise ValueError(
            f"layer_id={layer_id} is not a sparse attention layer; "
            f"sparse layers: {list(self.sparse_layer_id_mapping.keys())}"
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        """Write main K/V at `loc`. Works for any layer (dense or sparse)."""
        self.main_pool.set_kv_buffer(
            layer,
            loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
        )

    def set_index_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_idx_k: torch.Tensor,
        cache_idx_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        mapped_id = self.index_kv_layer_id_mapping.get(layer.layer_id)
        if mapped_id is None:
            raise ValueError(
                f"layer.layer_id={layer.layer_id} does not have an index V "
                f"cache (either dense, or in the K-only group). "
                f"index_kv layers: {list(self.index_kv_layer_id_mapping.keys())}"
            )
        self.index_kv_pool.set_kv_buffer(
            layer,
            loc,
            cache_idx_k,
            cache_idx_v,
            k_scale,
            v_scale,
            layer_id_override=mapped_id,
        )

    def set_index_k_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_idx_k: torch.Tensor,
    ) -> None:
        mapped_id = self.index_k_layer_id_mapping.get(layer.layer_id)
        if mapped_id is None:
            raise ValueError(
                f"layer.layer_id={layer.layer_id} is not in the K-only "
                f"sparse group. K-only layers: "
                f"{list(self.index_k_layer_id_mapping.keys())}"
            )
        sub_pool = self.index_k_pool
        if cache_idx_k.dtype != sub_pool.dtype:
            cache_idx_k = cache_idx_k.to(sub_pool.dtype)
        if sub_pool.store_dtype != sub_pool.dtype:
            cache_idx_k = cache_idx_k.view(sub_pool.store_dtype)
        sub_pool.k_buffer[mapped_id][loc] = cache_idx_k

    def _can_fuse_kv_index_store(
        self,
        index_pool: MHATokenToKVPool,
        cache_k: torch.Tensor,
        cache_idx_k: torch.Tensor,
    ) -> bool:
        """Fast-path precondition: CUDA, no per-store quantization, and a uniform
        head byte size shared by main and index caches."""
        main = self.main_pool
        return (
            self.use_minimax_fused_kv_index_store
            and _is_cuda
            # No dtype conversion / fp8 scaling on either side (the fused kernel
            # is a raw byte copy, it does not quantize).
            and main.store_dtype == main.dtype
            and index_pool.store_dtype == index_pool.dtype
            and cache_k.dtype == main.dtype
            and cache_idx_k.dtype == index_pool.dtype
            # Uniform head byte size collapses head_dim + dtype into one constant.
            and main.dtype == index_pool.dtype
            and main.head_dim == index_pool.head_dim
            # 128-bit vector copy requires a 16-byte-aligned head size.
            and (main.head_dim * main.dtype.itemsize) % 16 == 0
        )

    def set_fused_kv_index_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_idx_k: torch.Tensor,
        cache_idx_v: Optional[torch.Tensor],
    ) -> None:
        """Store main K/V + index K (+ optional index V) for a sparse layer in
        one fused JIT launch, falling back to separate stores when not applicable."""
        disable_value = cache_idx_v is None
        index_pool = self.index_k_pool if disable_value else self.index_kv_pool

        if index_pool is not None and self._can_fuse_kv_index_store(
            index_pool, cache_k, cache_idx_k
        ):
            from sglang.jit_kernel.minimax_store_kv_index import store_kv_index

            main = self.main_pool
            head_bytes = main.head_dim * main.dtype.itemsize
            if disable_value:
                idx_k_cache = self.get_index_k_buffer(layer.layer_id).flatten(1)
                idx_v_cache = None
            else:
                ik, iv = self.get_index_kv_buffer(layer.layer_id)
                idx_k_cache, idx_v_cache = ik.flatten(1), iv.flatten(1)
            store_kv_index(
                cache_k.flatten(1),
                cache_v.flatten(1),
                main.get_key_buffer(layer.layer_id).flatten(1),
                main.get_value_buffer(layer.layer_id).flatten(1),
                cache_idx_k.flatten(1),
                idx_k_cache,
                None if disable_value else cache_idx_v.flatten(1),
                idx_v_cache,
                loc,
                num_kv_heads=main.head_num,
                head_bytes=head_bytes,
            )
            return

        # Fallback: separate stores (identical semantics).
        self.set_kv_buffer(layer, loc, cache_k, cache_v)
        if disable_value:
            self.set_index_k_buffer(layer, loc, cache_idx_k)
        else:
            self.set_index_kv_buffer(layer, loc, cache_idx_k, cache_idx_v)

    def get_kv_size_bytes(self):
        sub_pools = [self.main_pool, self.index_kv_pool, self.index_k_pool]
        sizes = [p.get_kv_size_bytes() for p in sub_pools if p is not None]
        return sum(k for k, _ in sizes), sum(v for _, v in sizes)

    def get_contiguous_buf_infos(self):
        # Main K/V only; index buffers ride the state-buffer channel.
        return self.main_pool.get_contiguous_buf_infos()

    def get_index_k_state_buf_infos(self):
        # Per-page item_len (MHATokenToKVPool convention); index rows share the
        # main-KV `loc`, so the transfer reuses the same page-ids.
        pool = self.index_k_pool
        n = pool.layer_num
        data_ptrs = [pool.k_buffer[i].data_ptr() for i in range(n)]
        data_lens = [pool.k_buffer[i].nbytes for i in range(n)]
        item_lens = [pool.k_buffer[i][0].nbytes * pool.page_size for i in range(n)]
        return data_ptrs, data_lens, item_lens

    def maybe_get_custom_mem_pool(self):
        return self.main_pool.maybe_get_custom_mem_pool()

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # TODO: spec-decode needs sub-pools built with enable_kv_cache_copy=True,
        # then delegate to main_pool/index_pool.move_kv_cache.
        raise NotImplementedError(
            "move_kv_cache is not yet supported for MiniMaxSparseKVPool: "
            "sub-pools must be built with enable_kv_cache_copy=True first."
        )

    def get_v_head_dim(self):
        return self.main_pool.get_value_buffer(0).shape[-1]
