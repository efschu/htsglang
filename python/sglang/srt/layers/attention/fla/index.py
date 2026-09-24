# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import functools
from typing import Callable, Dict, Tuple

import torch
import triton

from sglang.srt.layers.attention.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


# ---------------------------------------------------------------------------
# GRAPH-STATIC cu_seqlens (P prefill graph, full backend)
#
# prepare_chunk_indices syncs the host (``.tolist()``) and
# prepare_chunk_offsets does a pageable H2D copy (``new_tensor([0])``); neither
# may run inside a CUDA-graph capture. Under the full prefill graph the GDN
# kernels receive ONE static query_start_loc tensor (the linear-attention
# backend's _extend_graph_buffers), and its first use -- a warmup forward,
# outside the capture -- computes both tables. ``tensor_cache`` would then
# serve the capture from its 4-entry cache, but it also EVICTS: the next
# eager forwards pass fresh cu_seqlens objects, the entry rotates out, and the
# table the captured graph still reads by address is freed and reused.
#
# A pinned tensor therefore gets its own PERMANENT table store: computed
# once, from the content the tensor holds at its first use (the capture
# bucket), never recomputed and never released for the process lifetime. The
# content is right for every later replay because the captured kernels bound
# themselves by the LIVE cu_seqlens on device and only take the chunk GRID
# from these tables (see _extend_graph_metadata). An unpinned tensor takes the
# unchanged tensor_cache path.
# ---------------------------------------------------------------------------
_GRAPH_STATIC_CU_SEQLENS: Dict[int, Tuple[torch.Tensor, Dict[tuple, torch.Tensor]]] = {}


def pin_graph_static_cu_seqlens(cu_seqlens: torch.Tensor) -> None:
    """Declare ``cu_seqlens`` a capture-stable buffer: its chunk tables are
    kept forever (see the block comment above). Idempotent."""
    entry = _GRAPH_STATIC_CU_SEQLENS.get(id(cu_seqlens))
    if entry is not None and entry[0] is cu_seqlens:
        return
    _GRAPH_STATIC_CU_SEQLENS[id(cu_seqlens)] = (cu_seqlens, {})


def graph_static_tables(cu_seqlens: torch.Tensor) -> Dict[tuple, torch.Tensor]:
    """The permanent tables of a pinned tensor ({} when not pinned)."""
    entry = _GRAPH_STATIC_CU_SEQLENS.get(id(cu_seqlens))
    if entry is None or entry[0] is not cu_seqlens:
        return {}
    return entry[1]


def _graph_static(name: str, fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(cu_seqlens, chunk_size):
        if _GRAPH_STATIC_CU_SEQLENS:
            entry = _GRAPH_STATIC_CU_SEQLENS.get(id(cu_seqlens))
            if entry is not None and entry[0] is cu_seqlens:
                key = (name, int(chunk_size))
                table = entry[1].get(key)
                if table is None:
                    table = fn(cu_seqlens, chunk_size)
                    entry[1][key] = table
                return table
        return fn(cu_seqlens, chunk_size)

    return wrapper


@tensor_cache
def _prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def _prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


prepare_chunk_indices = _graph_static("chunk_indices", _prepare_chunk_indices)
prepare_chunk_offsets = _graph_static("chunk_offsets", _prepare_chunk_offsets)
