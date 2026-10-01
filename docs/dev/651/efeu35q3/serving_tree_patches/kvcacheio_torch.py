"""Torch-native fallbacks for the sgl_kernel.kvcacheio "direct" transfers.

efeu-TP14 2026-10-01. The laptop (ROCm, gfx1103) has no sgl_kernel at all, so
the HiCache host<->device transfer ops were bound to None and the first backup
died with "'NoneType' object is not callable" (pool_host/mha.py:434). The three
DIRECT variants are, in sgl-kernel itself, nothing but torch copies driven from
the host (sgl-kernel/csrc/kvcacheio/transfer.cu: transfer_page_direct,
transfer_kv_direct, transfer_kv_page_first_direct_impl -- the ROCm build always
takes the per-page `fallback_to_page_copy` branch). They are mirrored here with
the SAME semantics, vectorised over pages (index_select/index_copy_ instead of
one slice copy per page: at page_size 1 a 17k-token prefix would otherwise be
~340k Python-level copies).

Semantics (identical to the C++):
  transfer_kv_direct(src_layers, dst_layers, src_idx, dst_idx, page_size)
      dst_layers[j][dst_idx] = src_layers[j][src_idx]   for every layer j
  transfer_kv_all_layer_direct_lf_pf(src_ptrs, dst_ptrs, src_idx, dst_idx, ps)
      layer-first device -> page-first host. Per page i (start s = src_idx[i*ps],
      host page d = dst_idx[i*ps] // ps): dst_ptrs[0][d, j, :ps] = src_ptrs[j][s:s+ps]
      (and dst_ptrs[1] from src_ptrs[j + L] unless MLA: len(dst_ptrs) == 1)
  transfer_kv_per_layer_direct_pf_lf(src_ptrs, dst_ptrs, src_idx, dst_idx, layer_id, ps)
      page-first host -> layer-first device for one layer: per page i (host page
      s = src_idx[i*ps] // ps, device start d = dst_idx[i*ps]):
      dst_ptrs[j][d:d+ps] = src_ptrs[0][s, layer_id + j, :ps]  (MLA: len(src_ptrs) == 1)

On this APU "host" and "device" are the same DDR5; every copy is a memcpy at
memory bandwidth, so a torch copy loses nothing to a custom kernel here.
"""

from __future__ import annotations

from typing import List

import torch


def _to(vals: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Move the gathered rows to dst's device.

    Device->host is SYNCHRONOUS on purpose: `vals` is a fresh temporary, and a
    non_blocking D2H into it would let the following host-side index_copy_
    read it before the copy landed (measured: every row stale). The C++ path
    copies straight into the final slice, which is why it may be non_blocking.
    Host->device stays non_blocking: the device-side index_copy_ is queued
    behind it on the same stream.
    """
    if vals.device == dst.device:
        return vals
    return vals.to(dst.device, non_blocking=dst.device.type != "cpu")


def _page_starts(indices: torch.Tensor, page_size: int) -> torch.Tensor:
    assert indices.numel() % page_size == 0, "indices must be page-aligned"
    return indices.view(-1, page_size)[:, 0].to(torch.int64)


def _expand(starts: torch.Tensor, page_size: int) -> torch.Tensor:
    if page_size == 1:
        return starts
    offs = torch.arange(page_size, device=starts.device, dtype=starts.dtype)
    return (starts[:, None] + offs).reshape(-1)


def transfer_kv_direct(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> None:
    assert len(src_layers) == len(dst_layers)
    assert src_indices.numel() == dst_indices.numel()
    if src_indices.numel() == 0:
        return
    for src, dst in zip(src_layers, dst_layers):
        si = src_indices.to(src.device, torch.int64)
        di = dst_indices.to(dst.device, torch.int64)
        vals = src.index_select(0, si)
        dst.index_copy_(0, di, _to(vals, dst))


def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> None:
    assert src_indices.numel() == dst_indices.numel()
    if src_indices.numel() == 0:
        return
    is_mla = len(dst_ptrs) == 1
    num_layers = len(src_ptrs) if is_mla else len(src_ptrs) // 2
    s_tok = _expand(_page_starts(src_indices, page_size), page_size)
    d_page = _page_starts(dst_indices, page_size) // page_size
    npages = d_page.numel()
    targets = [(0, 0)] if is_mla else [(0, 0), (1, num_layers)]
    for j in range(num_layers):
        for dst_slot, src_off in targets:
            src = src_ptrs[j + src_off]
            dst = dst_ptrs[dst_slot]
            vals = src.index_select(0, s_tok.to(src.device))
            vals = vals.view(npages, page_size, *src.shape[1:])
            dst[:, j].index_copy_(
                0, d_page.to(dst.device), _to(vals, dst)
            )


def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> None:
    assert src_indices.numel() == dst_indices.numel()
    if src_indices.numel() == 0:
        return
    is_mla = len(src_ptrs) == 1
    num_layers = len(dst_ptrs) if is_mla else len(dst_ptrs) // 2
    s_page = _page_starts(src_indices, page_size) // page_size
    d_tok = _expand(_page_starts(dst_indices, page_size), page_size)
    sources = [(0, 0)] if is_mla else [(0, 0), (1, num_layers)]
    for j in range(num_layers):
        for src_slot, dst_off in sources:
            src = src_ptrs[src_slot]
            dst = dst_ptrs[j + dst_off]
            vals = src[:, layer_id + j].index_select(0, s_page.to(src.device))
            vals = vals.reshape(-1, *vals.shape[2:])
            dst.index_copy_(
                0, d_tok.to(dst.device), _to(vals, dst)
            )
