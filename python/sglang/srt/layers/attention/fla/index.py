# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import functools
from typing import Callable, Dict, Tuple

import torch
import triton

from sglang.srt.layers.attention.fla.utils import tensor_cache

# ---------------------------------------------------------------------------
# HOST-KNOWN LENGTHS (P-NOSYNC, eager forwards; managers/weg2_p_overlap.py)
#
# The eager chunk tables below cost the host two waits per table and forward:
# ``.tolist()`` of the per-sequence chunk counts (device->host) and ``.to(
# cu_seqlens)`` / ``new_tensor([0])`` (pageable host->device, non_blocking=
# False). Each drains the forward stream at the first GDN layer, i.e. waits
# for the forward still running before this one. The linear-attention
# backend knows the lengths on the host (``extend_seq_lens_cpu``) and hands
# them in per cu_seqlens OBJECT (``note_host_seq_lens``); the eager
# computation then builds the SAME table on the host and sends it pinned,
# non-blocking. Unregistered tensors -- and every graph-static one, which the
# pin wrapper below serves first -- keep the stock computation.
# ---------------------------------------------------------------------------
_HOST_SEQ_LENS: dict = {}
_HOST_SEQ_LENS_CAP = 8


def note_host_seq_lens(cu_seqlens: torch.Tensor, seq_lens) -> bool:
    """Register the host lengths of ``cu_seqlens`` (one per sequence, so
    ``len == numel - 1``). Refused (False) when they cannot be the lengths of
    that tensor; bounded, oldest first."""
    try:
        lens = tuple(int(n) for n in seq_lens)
    except Exception:  # noqa: BLE001 - a non-iterable mirror: stock path
        return False
    if len(lens) != cu_seqlens.numel() - 1 or (lens and min(lens) < 0):
        return False
    _HOST_SEQ_LENS.pop(id(cu_seqlens), None)
    while len(_HOST_SEQ_LENS) >= _HOST_SEQ_LENS_CAP:
        _HOST_SEQ_LENS.pop(next(iter(_HOST_SEQ_LENS)))
    _HOST_SEQ_LENS[id(cu_seqlens)] = (cu_seqlens, lens)
    return True


def _host_seq_lens(cu_seqlens: torch.Tensor):
    entry = _HOST_SEQ_LENS.get(id(cu_seqlens))
    if entry is None or entry[0] is not cu_seqlens:
        return None
    return entry[1]


def _to_like(values, dtype, like: torch.Tensor) -> torch.Tensor:
    """``values`` onto ``like``'s device without a host wait (pinned,
    non-blocking) -- a plain tensor where there is no CUDA device."""
    t = torch.tensor(values, dtype=dtype, pin_memory=like.is_cuda)
    return t.to(like.device, non_blocking=True)


def _host_chunk_indices(lens, chunk_size: int, like: torch.Tensor) -> torch.Tensor:
    """The eager ``_prepare_chunk_indices`` table from host lengths: rows
    ``[seq, chunk]``, where ``seq`` counts the sequences that HAVE a chunk
    (``indices.eq(0).cumsum(0) - 1``), in ``like``'s dtype."""
    flat = []
    seq = -1
    for n in lens:
        for j in range((n + chunk_size - 1) // chunk_size):
            if j == 0:
                seq += 1
            flat.append(seq)
            flat.append(j)
    return _to_like(flat, like.dtype, like).view(-1, 2)


def _host_chunk_offsets(lens, chunk_size: int, like: torch.Tensor) -> torch.Tensor:
    """The eager ``_prepare_chunk_offsets`` vector from host lengths: 0 then
    the running chunk count, int64 (``cumsum`` promotes the integer input)."""
    acc = [0]
    for n in lens:
        acc.append(acc[-1] + (n + chunk_size - 1) // chunk_size)
    return _to_like(acc, torch.int64, like)


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
    host = _host_seq_lens(cu_seqlens)  # P-NOSYNC: registered host lengths
    if host is not None:
        return _host_chunk_indices(host, int(chunk_size), cu_seqlens)
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
    host = _host_seq_lens(cu_seqlens)  # P-NOSYNC: registered host lengths
    if host is not None:
        return _host_chunk_offsets(host, int(chunk_size), cu_seqlens)
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


prepare_chunk_indices = _graph_static("chunk_indices", _prepare_chunk_indices)
prepare_chunk_offsets = _graph_static("chunk_offsets", _prepare_chunk_offsets)
