"""Task #53 / Sitz 5 (20.09.): the QSA top-k rows resolve + owner compaction
as ONE Triton kernel per query row.

The graph decode path resolved every layer's top-k positions through a
chain of ~30 tiny torch kernels (index_select x2, compare x4, clamp, gather,
where x2, int64 mod/div, and/or, a radixSort for the owned-first compaction
and a reduce for the counts): measured on a 3080 (prof_fn8ad, TP1) ~100 us
per QSA layer of 1-2 us kernels plus a 19 us sort, twelve QSA layers per
round. This kernel does the same math in one launch:

    logical position -> req_to_token slot (validity: 0 <= pos < seq_len)
    slot -> (owned?, compact row) by the DCP owner rule
    rows: owned first (any order), -1 trailing; counts = #owned

Attention over the rows is permutation-invariant, so the owned rows need no
particular order (compact_owned_rows sorted descending; this kernel keeps
lane order). Owner rules: WEIGHTED (uneven DCP, layers/dcp/owner.py
dcp_weighted_read_slots), EVEN (slot % world == rank), NONE (dcp_size 1:
every valid slot is a row).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MODE_NONE = 0
MODE_EVEN = 1
MODE_WEIGHTED = 2


@triton.jit
def _qsa_rows_resolve_kernel(
    logical_ptr,  # [Tq, K] top-k logical positions
    seq_ids_ptr,  # [Tq] query row -> batch index
    seq_lens_ptr,  # [B] sequence length per batch row
    req_pool_ptr,  # [B] batch row -> req_to_token row
    req_to_token_ptr,  # [R, C] global KV slots
    rows_ptr,  # out [Tq, K] int32, owned first, -1 trailing
    counts_ptr,  # out [Tq] int32
    K,
    req_stride,
    max_col,  # C - 1
    cp_S,
    cp_lo,
    cp_hi,
    cp_ratio,
    world,
    rank,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0)
    sid = tl.load(seq_ids_ptr + q).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + sid).to(tl.int32)
    req = tl.load(req_pool_ptr + sid).to(tl.int64)

    offs = tl.arange(0, BLOCK)
    lane = offs < K
    pos = tl.load(logical_ptr + q * K + offs, mask=lane, other=-1).to(tl.int32)
    valid = lane & (pos >= 0) & (pos < seq_len)
    safe = tl.minimum(tl.maximum(pos, 0), max_col).to(tl.int64)
    slot = tl.load(req_to_token_ptr + req * req_stride + safe, mask=lane, other=0).to(tl.int64)

    if MODE == 2:
        off = slot % cp_S
        owned = (off >= cp_lo) & (off < cp_hi)
        compact = (slot // cp_S) * cp_ratio + (off - cp_lo)
    elif MODE == 1:
        owned = (slot % world) == rank
        compact = slot // world
    else:
        owned = slot >= 0
        compact = slot
    keep = valid & owned
    row = tl.where(keep, compact, -1).to(tl.int32)

    # owned-first permutation: owned lane i -> (#owned before i), unowned
    # lane i -> count + (#unowned before i); every lane writes one slot.
    incl = tl.cumsum(keep.to(tl.int32), axis=0)
    count = tl.sum(keep.to(tl.int32), axis=0)
    dest = tl.where(keep, incl - 1, count + (offs - incl))
    tl.store(rows_ptr + q * K + dest, row, mask=lane)
    tl.store(counts_ptr + q, count)


def qsa_rows_resolve(
    logical: torch.Tensor,
    seq_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool: torch.Tensor,
    req_to_token: torch.Tensor,
    *,
    mode: int = MODE_NONE,
    cp_S: int = 1,
    cp_lo: int = 0,
    cp_hi: int = 1,
    cp_ratio: int = 1,
    world: int = 1,
    rank: int = 0,
):
    """(rows [Tq, K] int32 owned-first / -1 trailing, counts [Tq] int32).

    ``logical`` [Tq, K] any int dtype; ``seq_ids`` [Tq]; ``seq_lens`` and
    ``req_pool`` [B]; ``req_to_token`` [R, C] (row stride taken from the
    tensor). Pure device ops on device tensors -- capturable in a CUDA graph
    and replayed against the live tables.
    """
    if logical.dim() != 2:
        raise ValueError("logical must be [Tq, K]")
    Tq, K = logical.shape
    if seq_ids.numel() != Tq:
        raise ValueError("QSA top-k rows do not match query rows")
    if req_to_token.dim() != 2:
        raise ValueError("req_to_token must be [R, C]")
    logical = logical.contiguous()
    rows = torch.empty((Tq, K), dtype=torch.int32, device=logical.device)
    counts = torch.empty((Tq,), dtype=torch.int32, device=logical.device)
    if Tq == 0 or K == 0:
        return rows, counts.zero_()
    block = max(16, triton.next_power_of_2(K))
    warps = 1 if block <= 256 else (4 if block <= 2048 else 8)
    _qsa_rows_resolve_kernel[(Tq,)](
        logical,
        seq_ids,
        seq_lens,
        req_pool,
        req_to_token,
        rows,
        counts,
        K,
        req_to_token.stride(0),
        req_to_token.shape[1] - 1,
        int(cp_S),
        int(cp_lo),
        int(cp_hi),
        int(cp_ratio),
        int(world),
        int(rank),
        MODE=int(mode),
        BLOCK=block,
        num_warps=warps,
    )
    return rows, counts
