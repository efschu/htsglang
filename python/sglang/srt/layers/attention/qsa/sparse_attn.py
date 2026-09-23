"""Validated sparse GQA operators migrated from the QSA reference branch."""

from typing import Optional

import torch
import triton
import triton.language as tl

_H20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
_L20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]


def _get_best_config(total_q: int):
    table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS
    return next(cfg for limit, cfg in table if total_q <= limit)


@triton.jit
def _sparse_gqa_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_seqlens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    seq_start = tl.load(cu_seqlens + batch).to(tl.int64)
    seq_end = tl.load(cu_seqlens + batch + 1).to(tl.int64)
    query_relative = tl.program_id(0).to(tl.int64)
    query = seq_start + query_relative
    if query >= seq_end:
        return

    row_topk = tl.minimum(topk, query_relative + 1)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + seq_start * sk_n + group * sk_h
    v_base = v + seq_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_prefill[(max_seqlen_k, (cu_seqlens.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_seqlens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        # The chunk-prefill K/V tensors are gathered from the KV pool and can
        # therefore carry the FP8 storage dtype, which Triton's dot rejects
        # (`Unsupported rhs dtype fp8e4nv`). Convert to Q's dtype; the QSA
        # backend writes the pool without per-tensor k/v scales, so this is a
        # plain cast (no-op for BF16 pools).
        keys = keys.to(q_values.dtype)
        values = values.to(q_values.dtype)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton_ck(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    max_q = int((cu_q[1:] - cu_q[:-1]).max().item())
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _fa2_valid_counts(
    seq_lens,
    indices,
    counts,
    topk: tl.constexpr,
    stride_i: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + row)
    positions = tl.load(
        indices + row * stride_i + cols,
        mask=cols < topk,
        other=-1,
    )
    valid = (positions >= 0) & (positions < length)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _fa2_prefix_sum(counts, cu_k, batch, BLOCK_B: tl.constexpr):
    rows = tl.arange(0, BLOCK_B)
    valid_rows = rows < batch
    row_counts = tl.load(counts + rows, mask=valid_rows, other=0)
    tl.store(cu_k, 0)
    tl.store(cu_k + rows + 1, tl.cumsum(row_counts, 0), mask=valid_rows)


def qwen_sparse_fa2_cu_seqlens_triton(
    seq_lens, indices, counts, cu_k, batch, topk, block_b: Optional[int] = None
):
    block_b = block_b or triton.next_power_of_2(batch)
    # One request per program: Triton caps a tile at 1M elements,
    # which [next_pow2(topk), next_pow2(batch)] exceeds at topk=2051, batch=512.
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )
    # Prefix sum is only over the batch dimension and remains a small 1-D
    # tensor, including during CUDA graph capture.
    _fa2_prefix_sum[(1,)](
        counts,
        cu_k,
        batch,
        BLOCK_B=block_b,
        num_warps=8,
    )


@triton.jit
def _compact_kv(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    pad_cols,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ZERO_FILL: tl.constexpr,
    KV_FP8: tl.constexpr = False,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    # 64-bit element offsets: slot * heads * dim exceeds int32 once the pool holds
    # more than 2^31 / (heads * dim) tokens (~4.2M for 2 x 256), which an FP8 pool
    # on one GPU does reach.
    src = slots.to(tl.int64)[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (
        (pack_start + cols).to(tl.int64)[:, None] * heads * dim
        + head * dim
        + dims[None, :]
    )
    load_mask = valid[:, None] & (dims[None, :] < dim)
    if ZERO_FILL:
        # Strided (page-aligned) packing: the paged decode kernel reads whole pages,
        # so every slot in [valid_count, pad_cols) must hold zeros, never stale bytes.
        # `valid_count` here is the row's page-aligned stride, not its valid count, so
        # the store covers the full region while the load stays limited to valid rows.
        store_mask = (cols < pad_cols)[:, None] & (dims[None, :] < dim)
    else:
        store_mask = load_mask
    # Dequantize while gathering: the scratch is allocated in the query dtype, so an
    # FP8 pool is read as fp8 and stored as bf16. The QSA backend writes the pool
    # without per-tensor k/v scales (see set_kv_buffer calls in
    # qwen_sparse_attn_backend.py), so no scale is applied here either.
    out_dtype = out_k.dtype.element_ty
    # fn5d 2026-09-16 (PP=3, 3080 stage): Triton refuses fp8e4nv below sm89, so
    # an FP8 pool arrives as uint8 bytes (KV_FP8) and is decoded here -- the
    # same helper the DCP rows kernel uses (WP3b).
    if KV_FP8:
        k_vals = _fp8_e4m3_bytes_to_f32(tl.load(k + src, mask=load_mask, other=0)).to(out_dtype)
        v_vals = _fp8_e4m3_bytes_to_f32(tl.load(v + src, mask=load_mask, other=0)).to(out_dtype)
    else:
        k_vals = tl.load(k + src, mask=load_mask, other=0.0).to(out_dtype)
        v_vals = tl.load(v + src, mask=load_mask, other=0.0).to(out_dtype)
    tl.store(out_k + dst, k_vals, mask=store_mask)
    tl.store(out_v + dst, v_vals, mask=store_mask)


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
    """Valid-count pass alone, without the packed cu_seqlens prefix sum."""
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )


def qwen_sparse_kv_extraction_compact_triton(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    zero_fill_cols: int = 0,
):
    """Gather the selected K/V rows into ``out_k``/``out_v``.

    ``zero_fill_cols`` > 0 selects the strided (page-aligned) layout used by the paged
    decode kernel: row ``b`` owns ``[cu_k[b], cu_k[b] + zero_fill_cols)`` and every slot
    past its valid rows is zero-filled. Paged kernels read whole pages and multiply the
    masked probabilities into V, so stale or uninitialized bytes there (NaN/Inf bit
    patterns) would otherwise leak into the output. ``0`` keeps the compact layout for
    the varlen fallback, whose rows are packed back-to-back.

    ``out_k``/``out_v`` may use a wider dtype than the pool (bf16 scratch for an FP8
    pool); rows are converted while gathering.

    Both layouts assume the valid entries of each ``indices`` row are contiguous at
    the front (``expand_qsa_block_indices`` sorts them that way): ``valid_count`` is a
    count, not a mask, so a ``-1`` in the middle of a row would shift the packing.
    """
    _, heads, dim = k.shape
    block_topk = 16
    zero_fill = zero_fill_cols > 0
    num_cols = zero_fill_cols if zero_fill else topk
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    if kv_fp8:
        # 1 byte per element either way; decoded in-kernel (fn5d, sm86 stages)
        k, v = k.view(torch.uint8), v.view(torch.uint8)
    _compact_kv[(batch, heads, triton.cdiv(num_cols, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        num_cols,
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        ZERO_FILL=zero_fill,
        KV_FP8=kv_fp8,
        num_warps=8,
    )


@triton.jit
def _fp8_e4m3_bytes_to_f32(x):
    """float8_e4m3fn stored as uint8 -> float32 by bit arithmetic. Triton
    refuses the fp8e4nv type below sm89 (the 3080 ranks, fn1w boot
    2026-09-16), so the pool bytes are loaded as uint8 and decoded here on
    every architecture alike: sign(1) exp(4, bias 7) mant(3); exp==0 is
    subnormal (2^-6 * m/8); 0x7F/0xFF (exp 15, mant 7) is NaN."""
    xi = x.to(tl.int32)
    sign = (xi >> 7) & 1
    exp = (xi >> 3) & 0xF
    man = xi & 0x7
    normal = tl.where(
        exp > 0, tl.math.exp2((exp - 7).to(tl.float32)) * (1.0 + man.to(tl.float32) / 8.0), man.to(tl.float32) / 512.0
    )
    val = tl.where(sign == 1, -normal, normal)
    nan = (exp == 15) & (man == 7)
    return tl.where(nan, float("nan"), val)


@triton.jit
def _sparse_attn_rows_fwd(
    q,
    k,
    v,
    out,
    lse,
    rows,
    scale,
    topk,
    counts,
    tail_k,
    tail_v,
    tail_map,
    tail_read,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    sl_m: tl.constexpr,
    sl_h: tl.constexpr,
    sr_m: tl.constexpr,
    sr_n: tl.constexpr,
    stk_n: tl.constexpr,
    stk_h: tl.constexpr,
    stk_d: tl.constexpr,
    stv_n: tl.constexpr,
    stv_h: tl.constexpr,
    stv_d: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_FP8: tl.constexpr,
    USE_COUNTS: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    """WP3b: sparse GQA over ABSOLUTE K/V rows, with the LSE.

    ``rows[query, :]`` are row indices straight into ``k``/``v`` (the KV pool
    as this rank stores it), ``-1`` = not attended. That is what makes one
    kernel serve every mode: prefill with a prefix, decode and the
    speculative paged modes hand it the rows they selected; under uneven DCP
    the rows are this rank's OWNED subset of the selection and the natural-log
    ``lse`` this kernel emits is what the group's LSE merge combines. A query
    whose rows are all ``-1`` (this rank owns none of its keys) gets out=0 and
    lse=-inf, which the merge weighs as exp(-inf)=0.

    PRECISION TAIL (#1243 slice 2q, ``HAS_TAIL``): ``tail_map[row]`` is the
    bf16 ring row that shadows pool row ``row`` (-1 = none). A lane whose row
    has one reads K/V from ``tail_k``/``tail_v`` in 16 bit; every other lane
    reads the pool exactly as without the tail. So each selected token is read
    ONCE, from ONE source -- no second pass, no merge. Group 0 of each query
    adds its tail lanes to ``tail_read`` (the READ fact, counted on the
    device so a graph replay counts too). ``HAS_TAIL=False`` keeps the masks
    of the tail-less kernel exactly.
    """
    query = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q + query * sq_m + (head_start + offs_h[:, None]) * sq_h + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + group * sk_h
    v_base = v + group * sv_h
    tk_base = tail_k + group * stk_h
    tv_base = tail_v + group * stv_h
    row_ptr = rows + query * sr_m
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    tail_lanes = tl.zeros([BLOCK_N], tl.int32)
    # Owned-row compaction (Task #42, 19.09.): under DCP every rank holds only
    # its OWN subset of a query's top-k rows (the rest are -1). Looping to
    # ``topk`` and masking the foreign lanes still runs the tile math for every
    # lane, so each rank paid the whole query x head x top-k work regardless of
    # its share -- the Ampere ranks at ~210 ms per layer against 33 ms on the
    # Blackwell rank, and the chunk waited for them. With the rows sorted so
    # the owned ones lead, ``counts[query]`` bounds the loop to the owned rows.
    if USE_COUNTS:
        limit = tl.load(counts + query).to(tl.int32)
    else:
        limit = topk
    for start in range(0, limit, BLOCK_N):
        current = start + offs_n
        row = tl.load(row_ptr + current * sr_n, mask=current < topk, other=-1).to(tl.int64)
        valid = row >= 0
        safe_row = tl.where(valid, row, 0)
        if HAS_TAIL:
            ring_row = tl.load(tail_map + safe_row, mask=valid, other=-1).to(tl.int64)
            in_tail = valid & (ring_row >= 0)
            in_body = valid & (ring_row < 0)
            safe_ring = tl.where(in_tail, ring_row, 0)
            tail_lanes += in_tail.to(tl.int32)
        else:
            in_body = valid
        keys_raw = tl.load(
            k_base + safe_row[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=in_body[None, :],
            other=0,
        )
        values_raw = tl.load(
            v_base + safe_row[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=in_body[:, None],
            other=0,
        )
        if KV_FP8:
            keys = _fp8_e4m3_bytes_to_f32(keys_raw).to(q_values.dtype)
            values = _fp8_e4m3_bytes_to_f32(values_raw).to(q_values.dtype)
        else:
            keys = keys_raw.to(q_values.dtype)
            values = values_raw.to(q_values.dtype)
        if HAS_TAIL:
            tail_keys = tl.load(
                tk_base + safe_ring[None, :] * stk_n + offs_d[:, None] * stk_d,
                mask=in_tail[None, :],
                other=0,
            ).to(q_values.dtype)
            tail_values = tl.load(
                tv_base + safe_ring[:, None] * stv_n + offs_d[None, :] * stv_d,
                mask=in_tail[:, None],
                other=0,
            ).to(q_values.dtype)
            keys = tl.where(in_tail[None, :], tail_keys, keys)
            values = tl.where(in_tail[:, None], tail_values, values)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        # A block with no valid key leaves next_max at -inf; exp2(-inf - -inf)
        # is nan, so the running max is only moved by finite scores.
        next_max = tl.where(next_max == -float("inf"), max_value, next_max)
        alpha = tl.where(
            max_value == -float("inf"), 1.0, tl.math.exp2(max_value - next_max)
        )
        probabilities = tl.where(
            next_max[:, None] == -float("inf"),
            0.0,
            tl.math.exp2(scores - next_max[:, None]),
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    has_keys = normalizer > 0
    safe_norm = tl.where(has_keys, normalizer, 1.0)
    output = accumulator / safe_norm[:, None]
    tl.store(
        out + query * so_m + (head_start + offs_h[:, None]) * so_h + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )
    lse_value = tl.where(
        has_keys, (max_value + tl.math.log2(safe_norm)) * 0.6931471805599453, -float("inf")
    )
    tl.store(
        lse + query * sl_m + (head_start + offs_h) * sl_h,
        lse_value,
        mask=offs_h < GROUP_SIZE,
    )
    if HAS_TAIL:
        # One count per (query, row): every head group reads the same rows.
        tl.atomic_add(tail_read, tl.sum(tail_lanes, axis=0).to(tl.int64), mask=group == 0)


def compact_owned_rows(rows):
    """Sort each query's rows so the owned ones (>= 0) lead and the -1 lanes
    trail, and count the owned ones: (rows_sorted [Tq, topk] int32,
    counts [Tq] int32). Attention is permutation-invariant over the rows, so
    the order change is exact up to fp32 summation order."""
    rows = rows.to(torch.int32)
    if rows.numel() == 0:
        return rows.contiguous(), torch.zeros((rows.shape[0],), dtype=torch.int32, device=rows.device)
    rows_sorted, _ = torch.sort(rows, dim=-1, descending=True)
    counts = (rows_sorted >= 0).sum(dim=-1, dtype=torch.int32)
    return rows_sorted.contiguous(), counts.contiguous()


def sparse_attn_rows_triton(q, k_pool, v_pool, rows, scale, row_counts=None, tail=None):
    """(out [Tq, Hq, D] in q.dtype, lse [Tq, Hq] fp32 natural log) for the
    selected absolute rows; see ``_sparse_attn_rows_fwd``.

    ``row_counts`` ([Tq] int32, from ``compact_owned_rows``) bounds every
    query's loop to its leading valid rows; without it the loop runs to
    ``topk`` and masks.

    ``tail`` (#1243 slice 2q) is ``(tail_k, tail_v, tail_map, tail_read)``:
    the precision tail's bf16 ring buffers for this layer, the int32 pool-row
    -> ring-row mapping (-1 = none) and the int64 device scalar the kernel
    adds its tail lanes to. ``None`` is the tail-less kernel."""
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k_pool.shape[1]
    group_size = num_q_heads // num_kv_heads
    rows = rows.to(torch.int32).contiguous()
    use_counts = row_counts is not None
    if use_counts:
        row_counts = row_counts.to(torch.int32).contiguous()
        if row_counts.shape[0] != rows.shape[0]:
            raise ValueError("row_counts must have one entry per query row")
    else:
        row_counts = rows  # unused pointer; USE_COUNTS=False never reads it
    has_tail = tail is not None
    if has_tail:
        tail_k, tail_v, tail_map, tail_read = tail
        if tail_k.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"sparse rows kernel: the tail ring must be 16/32 bit, got {tail_k.dtype}")
        if tuple(tail_k.shape[1:]) != tuple(k_pool.shape[1:]) or tuple(
            tail_v.shape[1:]
        ) != tuple(v_pool.shape[1:]):
            raise ValueError(
                "sparse rows kernel: the tail ring's (heads, head_dim) "
                f"{tuple(tail_k.shape[1:])} differ from the pool's {tuple(k_pool.shape[1:])}"
            )
        if tail_map.dtype != torch.int32 or not tail_map.is_contiguous():
            raise TypeError("sparse rows kernel: the tail mapping must be contiguous int32")
        if tail_read.dtype != torch.int64 or tail_read.numel() != 1:
            raise TypeError("sparse rows kernel: the tail read counter must be one int64")
    out = torch.empty_like(q)
    lse = torch.empty((total_q, num_q_heads), dtype=torch.float32, device=q.device)
    if total_q == 0:
        return out, lse
    kv_fp8 = k_pool.dtype == torch.float8_e4m3fn
    if kv_fp8:
        # Same strides (1 byte per element either way); decoded in-kernel.
        k_pool, v_pool = k_pool.view(torch.uint8), v_pool.view(torch.uint8)
    elif k_pool.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"sparse rows kernel: unsupported KV dtype {k_pool.dtype}")
    if not has_tail:
        # Unused pointers; HAS_TAIL=False never dereferences them.
        tail_k, tail_v, tail_map, tail_read = k_pool, v_pool, rows, rows
        tail_strides = (0, 0, 0, 0, 0, 0)
    else:
        tail_strides = (
            tail_k.stride(0), tail_k.stride(1), tail_k.stride(2),
            tail_v.stride(0), tail_v.stride(1), tail_v.stride(2),
        )
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    _sparse_attn_rows_fwd[(total_q, num_kv_heads)](
        q,
        k_pool,
        v_pool,
        out,
        lse,
        rows,
        scale,
        rows.shape[-1],
        row_counts,
        tail_k,
        tail_v,
        tail_map,
        tail_read,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_pool.stride(0),
        k_pool.stride(1),
        k_pool.stride(2),
        v_pool.stride(0),
        v_pool.stride(1),
        v_pool.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        rows.stride(0),
        rows.stride(1),
        *tail_strides,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        KV_FP8=kv_fp8,
        USE_COUNTS=use_counts,
        HAS_TAIL=has_tail,
        num_warps=warps,
        num_stages=stages,
    )
    return out, lse


def fp8_e4m3_bytes_to_f32_reference(x: torch.Tensor) -> torch.Tensor:
    """Torch twin of the in-kernel decode, for the pin against torch's own
    float8_e4m3fn conversion (all 256 codes)."""
    xi = x.to(torch.int32)
    sign, exp, man = (xi >> 7) & 1, (xi >> 3) & 0xF, xi & 0x7
    normal = torch.where(
        exp > 0, torch.exp2((exp - 7).float()) * (1.0 + man.float() / 8.0), man.float() / 512.0
    )
    val = torch.where(sign == 1, -normal, normal)
    return torch.where((exp == 15) & (man == 7), torch.full_like(val, float("nan")), val)


def sparse_attn_rows_reference(q, k_pool, v_pool, rows, scale):
    """Device-agnostic twin of ``sparse_attn_rows_triton`` (same contract,
    fp32 math): the pin for the kernel and for the LSE merge."""
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k_pool.shape[1]
    repeats = num_q_heads // num_kv_heads
    out = torch.zeros((total_q, num_q_heads, head_dim), dtype=torch.float32, device=q.device)
    lse = torch.full((total_q, num_q_heads), -float("inf"), dtype=torch.float32, device=q.device)
    for t in range(total_q):
        valid = rows[t] >= 0
        if not bool(valid.any()):
            continue
        sel = rows[t][valid].long()
        keys = k_pool[sel].float()  # [n, Hkv, D]
        values = v_pool[sel].float()
        for h in range(num_q_heads):
            g = h // repeats
            scores = keys[:, g, :] @ q[t, h].float() * scale  # [n]
            m = scores.max()
            p = torch.exp(scores - m)
            l = p.sum()
            out[t, h] = (p @ values[:, g, :]) / l
            lse[t, h] = m + torch.log(l)
    return out.to(q.dtype), lse


def merge_partial_attention(outs, lses):
    """The group LSE merge (the math of ``cp_lse_ag_out_ar_mha_uneven``) over a
    list of per-rank partials: ``sum_r exp(lse_r - lse) * out_r`` with
    ``lse = logsumexp_r(lse_r)``. A rank with lse=-inf contributes nothing."""
    lses = torch.stack([l.float() for l in lses])  # [R, T, H]
    global_lse = torch.logsumexp(lses, dim=0)
    total = None
    for out, l in zip(outs, lses):
        w = torch.exp(l - global_lse).unsqueeze(-1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        part = torch.nan_to_num(out.float(), nan=0.0) * w
        total = part if total is None else total + part
    return total, global_lse


__all__ = [
    "qwen_sparse_fa2_cu_seqlens_triton",
    "qwen_sparse_valid_counts_triton",
    "qwen_sparse_kv_extraction_compact_triton",
    "sparse_gqa_fwd_interface_triton",
    "sparse_gqa_fwd_interface_triton_ck",
    "sparse_attn_rows_triton",
    "sparse_attn_rows_reference",
    "merge_partial_attention",
]
