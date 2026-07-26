"""Fused fp8 dequant-GEMV for small-batch decode (#179 candidate 3, design B).

The dequant fallback (used wherever no fp8 GEMM exists) expands the whole
weight to the compute dtype ONCE PER FORWARD and then runs F.linear over it.
That cost is paid per forward, not per token, which makes it cheap in prefill
and ruinous in batch-1 decode. Measured on Qwen3.6-27B block-fp8, TP=3, forced
fallback:

    decode   91.53 -> 27.59 tok/s   (-69.9%; 10.93 -> 36.25 ms per token)
    prefill  1472.5 -> 1353.1 tok/s (-8.1%)

This kernel removes the materialisation for the decode case: it reads the fp8
bytes DIRECTLY, dequantises in-register, and accumulates -- so it moves half
the bytes of a bf16 expansion and allocates nothing but the output. Measured
against materialise+F.linear on the real bs=1 decode shapes:

    RTX 3080  (sm86, bf16)   2.77 - 4.19x
    RTX 2080 Ti (sm75, fp16) 2.51 - 5.46x
    Radeon RX Vega 64 (gfx900, fp16) 2.10 - 5.34x

RAW BYTE DECODE, deliberately. Triton's native ``fp8e4nv`` type is REJECTED on
sm86 ("type fp8e4nv not supported in this architecture") and is equally
unavailable on gfx900 and sm70 -- i.e. on every card that needs this path. So
the e4m3 fields are unpacked by hand from uint8, the same way the GGUF
MMVQ/K-quant kernels handle their formats. The decode is verified bit-exact
against ``tensor.to(torch.float32)``.

Accuracy note: this kernel is MORE accurate than the path it replaces, because
it accumulates in fp32 where materialisation rounds the weight to bf16/fp16
first. Against an fp32 reference: mean relative error 0.0014 here versus 0.0133
for materialise+F.linear. Equality against the old path is therefore the wrong
gate -- see the tests, which use greedy token-ID equality plus an error band
against fp32.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001 - absence is a supported state, not an error
    _HAS_TRITON = False


# Above this many rows the materialise+GEMM path wins: the expansion is
# amortised over the batch and cuBLAS/rocBLAS beats a hand GEMV. The kernel is
# a DECODE optimisation, and prefill deliberately keeps the existing path --
# that is the measured asymmetry (-69.9% decode vs -8.1% prefill) expressed as
# code rather than as a comment.
FUSED_GEMV_MAX_ROWS = 8


if _HAS_TRITON:

    @triton.jit
    def _fp8_block_dequant_gemv(
        X, W, S, Y,
        M, K, N,
        stride_xm, stride_ym,
        stride_wn, stride_wk,
        stride_sn, stride_sk,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        BN: tl.constexpr, BK: tl.constexpr,
    ):
        """y[n] = sum_k x[k] * (w[n,k] * s[n//BN, k//BK]), w stored as e4m3 bytes.

        e4m3fn layout: 1 sign | 4 exponent (bias 7) | 3 mantissa, no inf.
            e > 0 : (-1)^s * 2^(e-7) * (1 + m/8)
            e == 0: (-1)^s * 2^(-6)  * (m/8)      [subnormal]
        The scale is a per-(BN, BK) tile value, so it is read per k-tile instead
        of being expanded into a full (n, k) tensor.
        """
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        mask_n = offs_n < N
        mask_m = offs_m < M
        # (BLOCK_M, BLOCK_N): small M is the point -- speculative decode runs
        # several draft rows per step, so a strictly 1-row kernel would decline
        # exactly when it is most needed.
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            m2 = mask_n[:, None] & mask_k[None, :]

            b = tl.load(
                W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=m2, other=0,
            ).to(tl.int32)
            sgn = 1.0 - 2.0 * ((b >> 7) & 1).to(tl.float32)
            e = (b >> 3) & 0xF
            m = (b & 0x7).to(tl.float32)
            w = sgn * tl.where(
                e == 0,
                m * (0.015625 / 8.0),                    # 2^-6 / 8
                (1.0 + m / 8.0) * tl.exp2((e - 7).to(tl.float32)),
            )
            s = tl.load(
                S + (offs_n[:, None] // BN) * stride_sn
                  + (offs_k[None, :] // BK) * stride_sk,
                mask=m2, other=0.0,
            ).to(tl.float32)
            wq = w * s                                    # (BLOCK_N, BLOCK_K)

            x = tl.load(
                X + offs_m[:, None] * stride_xm + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :], other=0.0,
            ).to(tl.float32)                              # (BLOCK_M, BLOCK_K)
            # tl.dot, NOT a broadcast product: an earlier version used
            # tl.sum(x[:, None, :] * wq[None, :, :], axis=2), which materialises
            # a (BLOCK_M, BLOCK_N, BLOCK_K) intermediate in registers and
            # measured 4x SLOWER end to end than not fusing at all. BLOCK_M is
            # padded to 16 because tl.dot requires it.
            acc += tl.dot(x, tl.trans(wq))

        tl.store(
            Y + offs_m[:, None] * stride_ym + offs_n[None, :],
            acc.to(Y.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def fused_gemv_applicable(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Cheap dispatch predicate: is this the small-batch decode case, on a
    shape where the kernel actually wins?

    The aspect test is not a guess. The grid parallelises over N only, so a
    small N starves occupancy while a long K loop runs serially. Measured on
    the 27B's real weighted shape mix (RTX 3080, materialise vs fused):

        N=17408 K= 5120  N/K=3.40   1.41x  WIN
        N=12288 K= 5120  N/K=2.40   1.37x  WIN
        N=10240 K= 5120  N/K=2.00   1.16x  WIN
        N= 6144 K= 5120  N/K=1.20   1.06x  WIN
        N= 5120 K= 6144  N/K=0.83   0.86x  LOSE
        N= 5120 K=17408  N/K=0.29   0.90x  LOSE

    Fusing everywhere gives 1.151x on the weighted mix; fusing only where it
    wins gives 1.204x AND removes the regressions on a third of the layers.
    N >= K separates the two groups cleanly on every shape measured.
    """
    if not _HAS_TRITON or not x.is_cuda:
        return False
    if weight.dtype != torch.float8_e4m3fn:
        return False
    rows = 1 if x.dim() == 1 else x.shape[0]
    if rows > FUSED_GEMV_MAX_ROWS:
        return False
    N, K = weight.shape[-2], weight.shape[-1]
    return N >= K


def fused_block_dequant_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Fused path for block-scaled fp8. Returns None if it does not apply.

    ``weight`` is (N, K) fp8; ``scale`` is (ceil(N/bn), ceil(K/bk)).
    Returning None rather than raising keeps the caller's fallback trivial.
    """
    if not _HAS_TRITON:
        return None
    N, K = weight.shape[-2], weight.shape[-1]
    bn, bk = int(block_size[0]), int(block_size[1])
    # Ragged (partial trailing block) is left to the existing path: the index
    # arithmetic below assumes the scale tile covers every element it is asked
    # for, and correctness matters more than covering an uncommon shape.
    if N % bn or K % bk:
        return None

    squeeze = x.dim() == 1
    x2 = x.reshape(1, -1) if squeeze else x.reshape(-1, x.shape[-1])
    if x2.shape[-1] != K:
        return None
    M = x2.shape[0]
    if M > FUSED_GEMV_MAX_ROWS:
        return None
    x2 = x2.contiguous()

    w_u8 = weight.view(torch.uint8)
    y = torch.empty((M, N), device=x.device, dtype=out_dtype)
    BLOCK_M = 16  # tl.dot minimum; covers spec decode's draft rows
    BLOCK_N, BLOCK_K = 64, 128
    grid = (triton.cdiv(N, BLOCK_N),)
    _fp8_block_dequant_gemv[grid](
        x2, w_u8, scale, y,
        M, K, N,
        x2.stride(0), y.stride(0),
        w_u8.stride(0), w_u8.stride(1),
        scale.stride(0), scale.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BN=bn, BK=bk,
        num_warps=4,
    )
    if squeeze:
        return y.reshape(N)
    return y.reshape(*x.shape[:-1], N)
