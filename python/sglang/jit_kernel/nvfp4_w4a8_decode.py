"""NVFP4 W4A8-g16 DECODE GEMV for sm_86 (small M) on the INT8 tensor cores, reading the NATIVE NVFP4 layout.

Agent N4D (user order 2026-09-25). Companion of N4A's GEMM (``sglang.jit_kernel.nvfp4_w4a8``): same tensors, same
arithmetic (E2M1 x2 -> INT8 exact, INT32 block sum -> FP32 via the magic accumulator, x E4M3 block scale exact,
FP32 accumulation, x s_x[m] * weight_scale_2 at the end). N4A stages 32-B row pieces through shared memory, which
turns every DRAM access into one scattered sector; this kernel streams 64-B row segments with 128-bit loads and
splits K over the warps of a CTA (deterministic shared-memory reduction). See the .cuh header for the modes.

* mode 0 ("diag", M <= 8):  block-diagonal B on m16n8k32, natural int8 activation layout;
* mode 1 ("xpose", M <= 48): quad register transpose + m16n8k16 per block, segment-permuted activation layout.

Requirements (else the caller falls back to N4A's GEMM): Kp % 128 == 0 and K == Kp (no K padding), 16-B aligned
rows. All Qwen3.8-27B shards satisfy both (uneven-TP block [128, 128]).
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.jit_kernel.utils import cache_once_per_arch, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

#: largest M served by the decode GEMV (mode 1); above it N4A's tiled GEMM runs
DECODE_MAX_M = 48
#: largest M served by mode 0 (diag)
DIAG_MAX_M = 8


@cache_once_per_arch
def _jit_module() -> Module:
    return load_jit(
        "nvfp4_w4a8_decode_sm86",
        cuda_files=["gemm/nvfp4_w4a8_decode_sm86.cuh"],
        cuda_wrappers=[
            ("gemm", "nvfp4_w4a8_decode_gemm"),
            ("quant", "nvfp4_w4a8_decode_quant"),
        ],
        # same numerics pins as N4A: IEEE denormals (E4M3 scales enter as value * 2^-120), exact division
        extra_cuda_cflags=[
            "-ftz=false",
            "-prec-div=true",
            "-prec-sqrt=true",
            "-lineinfo",
        ],
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def decode_max_m() -> int:
    """``SGLANG_W4A8_DECODE_MAX_M`` (default 48, 0 disables the decode GEMV)."""
    return max(0, min(DECODE_MAX_M, _env_int("SGLANG_W4A8_DECODE_MAX_M", DECODE_MAX_M)))


def mode_for_m(m: int) -> int:
    diag = max(0, min(DIAG_MAX_M, _env_int("SGLANG_W4A8_DECODE_DIAG_MAX_M", 4)))
    return 0 if m <= diag else 1


def _env_cfg() -> Optional[Tuple[int, int, int, int]]:
    env = os.environ.get("SGLANG_W4A8_DECODE_CFG", "")
    if not env:
        return None
    md, kw, rw, u = (int(v) for v in env.split(","))
    return md, kw, rw, u


@functools.lru_cache(maxsize=None)
def config_for(m: int, n_rows: int, kp: int) -> Tuple[int, int, int, int]:
    """(mode, kw, rw, u) for M tokens on a [n_rows, kp] weight.

    kw warps split K of one 16-row tile, rw row tiles share a CTA, u consecutive 128-element segments per warp step.
    The rules are the fastest configs of the RTX 3080 sweep (n4d_sweep.py, 25.09., evidence n4d_decode_0925/s1+s2),
    cited per branch as shape M -> us. ``SGLANG_W4A8_DECODE_CFG="mode,kw,rw,u"`` overrides (tuning only).
    """
    env = _env_cfg()
    if env is not None:
        return env
    tiles = -(-n_rows // 128) * 8
    nseg = kp // 128
    if mode_for_m(m) == 0:  # diag
        if tiles >= 4096:
            return 0, 4, 1, 2  # D.lm_head M1 347.7
        if nseg >= 96:
            return 0, 8, 1, 2 if m == 1 else 1  # P.down M1 81.9, M4 86.2
        return 0, 8 if (m == 1 and tiles >= 512) else 4, 1, 1  # D.gate_up M1 39.8 M4 42.0; D.down M1 20.5 M4 22.1
    if m <= 16:
        if tiles >= 4096:
            return 1, 1, 1, 2  # D.lm_head M16 400.6
        if nseg >= 96 or tiles <= 384:
            return 1, 8 if m > 8 else 4, 1, 1  # P.down M16 105.4; D.down M8 22.5, M16 27.6
        return 1, 2, 1, 2 if m <= 8 else 1  # D.gate_up M8 44.2, M16 48.4; P.gate_up M8 170.6
    if tiles <= 384 and m <= 32 and nseg < 96:
        return 1, 2, 1, 1  # D.down M32 38.4
    return 1, 1, 1, 4 if m <= 32 else 2  # D.gate_up M32 67.8 M48 85.9; P.gate_up M32 224.6; D.down/P.down M48


def eligible(x_k: int, weight: torch.Tensor) -> bool:
    kp = weight.shape[1] * 2
    return (
        kp % 128 == 0
        and x_k == kp
        and weight.stride(-1) == 1
        and weight.stride(0) % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


def quantize_activation(x: torch.Tensor, permuted: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16 [M, K] -> int8 [M, K] (natural or segment-permuted) + fp32 [M] (per_token_quant_int8 arithmetic)."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(-1) == 1
    m, k = x.shape
    xq = torch.empty((m, k), dtype=torch.int8, device=x.device)
    xs = torch.empty((m,), dtype=torch.float32, device=x.device)
    _jit_module().quant(x, xq, xs, int(bool(permuted)))
    return xq, xs


def decode_gemm(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    out_features: int,
    cfg: Optional[Tuple[int, int, int, int]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """int8 activations (layout of ``cfg[0]``) x native NVFP4 -> bf16 [M, out_features]."""
    m, kp = xq.shape
    if out is None:
        out = torch.empty((m, out_features), dtype=torch.bfloat16, device=xq.device)
    if cfg is None:
        cfg = config_for(m, weight.shape[0], kp)
    mode, kw, rw, u = cfg
    gs = weight_global_scale.reshape(1)
    if gs.dtype != torch.float32:
        gs = gs.to(torch.float32)
    ws = weight_scale.view(torch.uint8) if weight_scale.dtype != torch.uint8 else weight_scale
    _jit_module().gemm(out, xq, xs, weight, ws, gs, int(out_features), int(mode), int(kw), int(rw), int(u))
    return out


def nvfp4_w4a8_decode_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    out_features: int,
    *,
    cfg: Optional[Tuple[int, int, int, int]] = None,
) -> torch.Tensor:
    """bf16 [M, K] (M <= 48) -> bf16 [M, out_features]; caller checks ``eligible``."""
    m = x.shape[0]
    if cfg is None:
        cfg = config_for(m, weight.shape[0], weight.shape[1] * 2)
    xq, xs = quantize_activation(x, permuted=(cfg[0] == 1))
    return decode_gemm(xq, xs, weight, weight_scale, weight_global_scale, out_features, cfg)
