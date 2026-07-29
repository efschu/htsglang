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

TWO SCALE LAYOUTS, ONE DESIGN. The block variant above was the first one built,
because the 27B on the main rig is block-scaled. But the checkpoints that
actually LAND on the cards this path exists for are per-channel
(compressed-tensors ``strategy: channel``, one fp32 scale per output row) -- the
4B fp8 on the 2080 Ti is exactly that, so for it the block kernel was never even
compiled. ``fused_channel_dequant_gemv`` closes that gap and is the SIMPLER of
the two:

* the scale does not vary along K, so it factors straight out of the k loop --
  one vector load per program at the end instead of a tile load per k step, and
  no per-element multiply inside the loop;
* there is consequently no block geometry to be ragged about, so unlike the
  block variant it never declines a shape for divisibility.

``SGLANG_FP8_FUSED_GEMV=0`` disables BOTH variants, for A/B measurement. It is
an off-switch on the fallback lane, not a second enable: the fused path is still
reachable only where ``fp8_needs_dequant_fallback()`` already put us.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
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
#
# 8 -> 16 (#274 round 7c): a DFLASH drafter proposes a whole BLOCK per round,
# and its block size is 16 (``block_size`` in the draft config, and the value
# every released Qwen3.6 DFLASH export carries). At 8 the drafter's own decode
# rounds fell off the fused path entirely and took the materialise+GEMM
# fallback -- so a measurement of "DFLASH on an fp8 card" would have been a
# measurement of the fallback. 16 is a draft BLOCK, not a serving batch: it is
# still decode-shaped work, one request deep, which is what the kernel's
# asymmetry is about.
FUSED_GEMV_MAX_ROWS = 16

# Tile geometry for the PER-CHANNEL kernel, chosen by measurement on the two
# healthy cards available (RTX 3080 sm86, RTX 5090 sm120) over the real 4B shape
# mix, weighted by multiplicity. The block variant's inline (64, 128) is NOT the
# right choice here: on the dominant 9216x2560 (64 of 128 weights) it leaves 25%
# on the table, and -- more importantly -- it is what made wide weights look like
# losers. See fused_channel_gemv_applicable for what that changed.
# Named rather than inlined so a benchmark can sweep the SHIPPED code path
# instead of a copy of it that then drifts.
CHANNEL_BLOCK_N = 32
CHANNEL_BLOCK_K = 64
CHANNEL_NUM_WARPS = 4


@lru_cache(maxsize=1)
def fused_gemv_enabled() -> bool:
    """``SGLANG_FP8_FUSED_GEMV=0`` turns the fused kernels off; default on.

    This exists so the fused path can be measured against the materialisation it
    replaces IN THE SAME BUILD -- a same-session A/B, which is the only kind this
    project trusts for a decode number. It switches BOTH the block and the
    per-channel variant, because a flag that covered only one would silently
    compare two different things on two different checkpoints.

    Deliberately NOT a second enable condition. The fused path is reachable only
    where ``fp8_needs_dequant_fallback()`` has already established that no fp8
    GEMM exists; this flag can only subtract from that, never add to it. One
    gate, one lane -- a second independent gate is a thing that drifts.
    """
    return os.environ.get("SGLANG_FP8_FUSED_GEMV", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


if _HAS_TRITON:

    @triton.jit
    def _fp8_block_dequant_gemv(
        X,
        W,
        S,
        Y,
        M,
        K,
        N,
        stride_xm,
        stride_ym,
        stride_wn,
        stride_wk,
        stride_sn,
        stride_sk,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
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
                mask=m2,
                other=0,
            ).to(tl.int32)
            sgn = 1.0 - 2.0 * ((b >> 7) & 1).to(tl.float32)
            e = (b >> 3) & 0xF
            m = (b & 0x7).to(tl.float32)
            w = sgn * tl.where(
                e == 0,
                m * (0.015625 / 8.0),  # 2^-6 / 8
                (1.0 + m / 8.0) * tl.exp2((e - 7).to(tl.float32)),
            )
            s = tl.load(
                S
                + (offs_n[:, None] // BN) * stride_sn
                + (offs_k[None, :] // BK) * stride_sk,
                mask=m2,
                other=0.0,
            ).to(tl.float32)
            wq = w * s  # (BLOCK_N, BLOCK_K)

            x = tl.load(
                X + offs_m[:, None] * stride_xm + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(
                tl.float32
            )  # (BLOCK_M, BLOCK_K)
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

    @triton.jit
    def _fp8_channel_dequant_gemv(
        X,
        W,
        S,
        Y,
        M,
        K,
        N,
        stride_xm,
        stride_ym,
        stride_wn,
        stride_wk,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """y[m,n] = s[n] * sum_k x[m,k] * w[n,k], w stored as e4m3 bytes.

        Same raw-byte e4m3 decode as the block kernel (Triton's ``fp8e4nv`` does
        not exist on sm75/sm70/sm86/gfx900, i.e. on every card that needs this
        path), same fp32 accumulator, same tl.dot with BLOCK_M padded to 16.

        The ONE structural difference, and the reason this variant is cheaper:
        the scale depends on n only, so it comes out of the k loop entirely.
        The block kernel must load an (n, k) scale tile and multiply it into the
        weights on every iteration; here a single (BLOCK_N,) vector is loaded
        once, after the loop, and applied to the accumulator. Factoring it out
        is exact -- it is the same product, reassociated -- and it keeps the
        inner loop down to load, decode, dot.
        """
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        mask_n = offs_n < N
        mask_m = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            b = tl.load(
                W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0,
            ).to(tl.int32)
            # e4m3fn: 1 sign | 4 exponent (bias 7) | 3 mantissa, no inf.
            #   e > 0 : (-1)^s * 2^(e-7) * (1 + m/8)
            #   e == 0: (-1)^s * 2^(-6)  * (m/8)      [subnormal]
            sgn = 1.0 - 2.0 * ((b >> 7) & 1).to(tl.float32)
            e = (b >> 3) & 0xF
            m = (b & 0x7).to(tl.float32)
            w = sgn * tl.where(
                e == 0,
                m * (0.015625 / 8.0),  # 2^-6 / 8
                (1.0 + m / 8.0) * tl.exp2((e - 7).to(tl.float32)),
            )

            x = tl.load(
                X + offs_m[:, None] * stride_xm + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(x, tl.trans(w))

        s = tl.load(S + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc * s[None, :]

        tl.store(
            Y + offs_m[:, None] * stride_ym + offs_n[None, :],
            acc.to(Y.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def _small_batch_fp8(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """The conditions BOTH variants share: fused path live, fp8 weight, decode."""
    if not _HAS_TRITON or not fused_gemv_enabled() or not x.is_cuda:
        return False
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        return False
    rows = 1 if x.dim() == 1 else x.shape[0]
    return rows <= FUSED_GEMV_MAX_ROWS


def _as_uint8_nk(weight: torch.Tensor) -> Optional[torch.Tensor]:
    """Reinterpret an (N, K) fp8 weight as uint8, keeping its layout.

    The kernel indexes through explicit strides, so either memory order works --
    but it must be handed a UINT8 tensor, because a float8_e4m3fn pointer makes
    Triton type the load as ``fp8e4nv``, the very type these cards reject.

    Two layouts occur in the wired call sites and both are covered:
      * (N, K) row-major -- compressed-tensors W8A16 keeps the loaded layout;
      * a ``.t()`` view of a (K, N) row-major tensor -- Fp8LinearMethod stores
        the weight transposed for F.linear.
    ``Tensor.view(dtype)`` is applied to whichever of the two is contiguous, so
    no copy is made and no assumption about non-contiguous ``view`` is needed.
    """
    if weight.dim() != 2:
        return None
    if weight.is_contiguous():
        return weight.view(torch.uint8)
    t = weight.t()
    if t.is_contiguous():
        return t.view(torch.uint8).t()
    return None


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

    This aspect test belongs to the BLOCK variant specifically -- see
    ``fused_channel_gemv_applicable``, where it was measured to be wrong.
    """
    if not _small_batch_fp8(x, weight):
        return False
    N, K = weight.shape[-2], weight.shape[-1]
    return N >= K


def fused_channel_gemv_applicable(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Dispatch predicate for the PER-CHANNEL variant: no aspect test.

    The block variant declines N < K, and inheriting that here was the obvious
    thing to do. It is wrong, and the measurement says so plainly. Fused vs
    materialise+F.linear on the real 4B per-channel mix, bf16, tiles as shipped:

    | shape (N x K) | count | N/K  | RTX 3080 sm86 | RTX 5090 sm120 |
    |---|---|---|---|---|
    | 9216 x 2560 | 64 | 3.60 | 5.54x | 2.75x |
    | 2560 x 9216 | 32 | 0.28 | 3.52x | 2.33x |
    | 1024 x 2560 | 16 | 0.40 | 1.48x | 1.30x |
    | 2560 x 4096 |  8 | 0.62 | 3.72x | 1.22x |
    | 8192 x 2560 |  8 | 3.20 | 6.04x | 2.32x |

    EVERY shape wins, wide ones included -- the worst case is 1.22x, not a loss.
    Count-weighted, keeping the N >= K gate would cost 4.44x -> 3.58x on the 3080
    and 2.34x -> 1.96x on the 5090; time-weighted (which is what a forward feels)
    it is 4.52x -> 2.18x, i.e. the gate throws away more than half the gain.

    Why the block variant's guard does not transfer: there, a long-K weight pays a
    scale-tile load and a full-tile multiply on EVERY k step, so a wide shape is
    hit twice -- starved occupancy and a heavier inner loop. Here the loop is only
    load, decode, dot; the scale is applied once to the accumulator. Wide shapes
    lose occupancy and nothing else, and the materialisation they are racing is so
    expensive that they still win.

    NOT verified on sm75: the 2080 Ti was locked to a 300 MHz SM clock by a driver
    power-cap fault throughout this work (memory clock normal), which distorts
    exactly the ALU-vs-bandwidth balance this trade-off turns on. The two cards
    above agree, so the gate is dropped; if an sm75 measurement on a healthy card
    later contradicts them, this is the place to look.
    """
    return _small_batch_fp8(x, weight)


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
    if not _HAS_TRITON or not fused_gemv_enabled():
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
        x2,
        w_u8,
        scale,
        y,
        M,
        K,
        N,
        x2.stride(0),
        y.stride(0),
        w_u8.stride(0),
        w_u8.stride(1),
        scale.stride(0),
        scale.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BN=bn,
        BK=bk,
        num_warps=4,
    )
    if squeeze:
        return y.reshape(N)
    return y.reshape(*x.shape[:-1], N)


def fused_channel_dequant_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Fused path for PER-CHANNEL fp8. Returns None if it does not apply.

    ``weight`` is (N, K) fp8 -- either row-major or a ``.t()`` view of a (K, N)
    row-major tensor, both of which occur among the callers. ``scale`` is one
    fp32 value per OUTPUT channel: (N,) or (N, 1), which is what
    compressed-tensors ``strategy: channel`` produces and what
    ``convert_to_channelwise`` turns a per-tensor scale into. A scale of any
    other shape is declined rather than guessed at.

    Note what is NOT here, compared with the block variant: no block geometry,
    hence no divisibility check and no ragged decline. Every (N, K) is masked
    correctly by the kernel, so this variant covers shapes the block one has to
    hand back.
    """
    if not _HAS_TRITON or not fused_gemv_enabled():
        return None
    N, K = weight.shape[-2], weight.shape[-1]

    s = scale
    if s.dim() == 2 and s.shape[1] == 1:
        s = s.reshape(-1)
    if s.dim() != 1 or s.shape[0] != N:
        return None
    s = s.contiguous().to(torch.float32)

    w_u8 = _as_uint8_nk(weight)
    if w_u8 is None:
        return None

    squeeze = x.dim() == 1
    x2 = x.reshape(1, -1) if squeeze else x.reshape(-1, x.shape[-1])
    if x2.shape[-1] != K:
        return None
    M = x2.shape[0]
    if M > FUSED_GEMV_MAX_ROWS:
        return None
    x2 = x2.contiguous()

    y = torch.empty((M, N), device=x.device, dtype=out_dtype)
    BLOCK_M = 16  # tl.dot minimum; covers spec decode's draft rows
    BLOCK_N, BLOCK_K = CHANNEL_BLOCK_N, CHANNEL_BLOCK_K
    grid = (triton.cdiv(N, BLOCK_N),)
    _fp8_channel_dequant_gemv[grid](
        x2,
        w_u8,
        s,
        y,
        M,
        K,
        N,
        x2.stride(0),
        y.stride(0),
        w_u8.stride(0),
        w_u8.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=CHANNEL_NUM_WARPS,
    )
    if squeeze:
        return y.reshape(N)
    return y.reshape(*x.shape[:-1], N)
