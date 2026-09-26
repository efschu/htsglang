# SPDX-License-Identifier: Apache-2.0
"""sm_12x NVFP4 kernel choice: W4A16 on the NATIVE layout for small M.

flashinfer #5242 (``mm_bf16_fp4(backend="cute-dsl-native")``, merged at
2f3bc5ac, flashinfer 0.7.0 source build) multiplies BF16 activations with the
very bytes the fork already keeps for the sm_120 W4A4 path -- ``weight`` uint8
[N, K/2] (E2M1, element 2j in the low nibble) and ``weight_scale_interleaved``
e4m3 in the 128x4 swizzle (docs/NVFP4_NATIVE_LAYOUT_CONTRACT.md) -- without a
second weight copy and without any preparation. Decode-sized M are
weight-bandwidth bound, and W4A16 skips the activation FP4 quantization; large
M stay on the W4A4 FP4 tensor cores.

This module is ONE decision and ONE call, nothing else:

* :func:`choose_sm12x_fp4_kernel` -- pure: ``"w4a16_native"`` or ``"w4a4"``.
* :func:`maybe_apply_sm12x_w4a16` -- the seam for ``ModelOptFp4LinearMethod.apply``
  (and the ``native-mixed`` dispatch): returns the output, or ``None`` = "not mine,
  run the rank's W4A4 path as before".

Default OFF. ``SGLANG_FP4_SM12X_W4A16_MAX_M=<M>`` (read once per process) turns
it on for ``1 <= M <= <M>``; unset/0 = every call returns ``None`` after one
integer compare. sm_8x/sm_10x ranks never take it (the 3080 keeps whatever its
backend is -- Marlin/W4A8); an sm_12x rank with the switch on but without the
#5242 kernel (e.g. the PyPI 0.7.0 wheel) is a HARD error, never a silent
fallback: the switch was set for a kernel that is not there.

MEASURED (5090 @ 400 W, flashinfer 2f3bc5ac, CUDA graph, weights rotated past
L2, autotuned; /spinning/evidence-665-f1/f_fi_next_bench_0925_1408): the
cute-dsl-native kernel is weight-bandwidth bound up to M = 16 and collapses
right after (P.down M=16 -> 24: 38 -> 111 us, gate_up 74 -> 102 us), so a
value above :data:`MEASURED_MAX_M` only makes it slower. Within M <= 16 it is
0.95-1.16x the time of flashinfer's W4A4 CUTLASS (mm_fp4, incl. activation
quant): it wins only on the down shapes at M <= 8 and loses 4-16 % on gate_up
and lm_head -- but it is 1.3-1.6x faster than the fork's W4A4 CUTLASS that
``auto`` resolves on sm_120 today, and its error vs an fp32 dequant reference is
1.6e-3 against 9.5e-2 for every W4A4 lane (the activation FP4 quantization).
So the switch buys accuracy at speed parity, not speed.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

MAX_M_ENV = "SGLANG_FP4_SM12X_W4A16_MAX_M"
NATIVE_BACKEND = "cute-dsl-native"
W4A16_NATIVE = "w4a16_native"
W4A4 = "w4a4"
#: Largest M at which the kernel stays bandwidth-bound on the 5090 (measured).
MEASURED_MAX_M = 16

#: Backends whose weights are the plain native layout (weight [N, K/2],
#: 128x4-swizzled scale). flashinfer_trtllm shuffles both; marlin repacks;
#: w4a8_int8 is an sm_8x-only resolution.
_NATIVE_LAYOUT_BACKENDS = frozenset(
    {"cutlass", "flashinfer_cutlass", "flashinfer_cudnn", "flashinfer_cutedsl", "auto"}
)

_MAX_M: Optional[int] = None
_RANK_ON: Optional[bool] = None


def _read_max_m() -> int:
    raw = str(os.environ.get(MAX_M_ENV, "") or "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        raise ValueError(f"{MAX_M_ENV}={raw!r} is not an integer") from None


def sm12x_w4a16_max_m() -> int:
    global _MAX_M
    if _MAX_M is None:
        _MAX_M = _read_max_m()
    return _MAX_M


def native_w4a16_kernel_present() -> bool:
    """flashinfer carries #5242 (located, not imported)."""
    try:
        return importlib.util.find_spec("flashinfer.gemm.kernels.native_bf16_fp4") is not None
    except (ImportError, ValueError):
        return False


def choose_sm12x_fp4_kernel(m: int, max_m: int, capability_major: int) -> str:
    """Pure kernel choice for one NVFP4 linear call."""
    if int(capability_major) != 12 or max_m <= 0:
        return W4A4
    return W4A16_NATIVE if 1 <= int(m) <= int(max_m) else W4A4


def _rank_on() -> bool:
    """Resolved once per process: switch set AND this rank is sm_12x AND the
    kernel exists (else a hard error on sm_12x)."""
    global _RANK_ON
    if _RANK_ON is not None:
        return _RANK_ON
    max_m = sm12x_w4a16_max_m()
    if max_m <= 0:
        _RANK_ON = False
        return False
    from sglang.srt.utils.common import get_device_capability

    cap = get_device_capability()
    major = int(cap[0]) if cap and cap[0] is not None else -1
    if major != 12:
        logger.info("%s=%d ignored on this rank (sm_%s): W4A16 native is sm_12x only", MAX_M_ENV, max_m, major)
        _RANK_ON = False
        return False
    if not native_w4a16_kernel_present():
        raise RuntimeError(
            f"{MAX_M_ENV}={max_m} asks for flashinfer mm_bf16_fp4(backend='{NATIVE_BACKEND}') "
            "but the installed flashinfer has no gemm.kernels.native_bf16_fp4 (#5242, commit "
            "2f3bc5ac; the PyPI 0.7.0 wheel predates it). Install the source build or unset the switch."
        )
    if max_m > MEASURED_MAX_M:
        logger.warning(
            "%s=%d is above the measured bandwidth-bound range of the kernel (M <= %d on the "
            "5090); M in (%d, %d] will run 1.5-3x slower than W4A4.",
            MAX_M_ENV, max_m, MEASURED_MAX_M, MEASURED_MAX_M, max_m,
        )
    logger.info("NVFP4 sm_12x: W4A16 %s for M <= %d, W4A4 above", NATIVE_BACKEND, max_m)
    _RANK_ON = True
    return True


def _reset_for_tests() -> None:
    global _MAX_M, _RANK_ON
    _MAX_M = None
    _RANK_ON = None


def w4a16_alpha(layer: torch.nn.Module) -> torch.Tensor:
    """The W4A16 output scale = the global WEIGHT scale (weight_scale_2), float32 (1,).

    W4A4's ``alpha`` is input_scale * weight_scale_2; the activation half does
    not exist in W4A16. ``weight_global_scale`` (bound by the Marlin and the
    native-mixed paths) is used when present, else ``alpha * input_scale_inv``.
    Cached on the layer, keyed by the source tensors' storage and version, so a
    weight reload / flip that rewrites the scalars in place re-derives it."""
    src = getattr(layer, "weight_global_scale", None)
    srcs = (src,) if src is not None else (layer.alpha, layer.input_scale_inv)
    key = tuple((t.data_ptr(), t._version) for t in srcs)
    cached = getattr(layer, "_w4a16_alpha_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if src is not None:
        a = src.detach().to(torch.float32).reshape(-1)[:1].contiguous()
    else:
        a = (layer.alpha.detach().to(torch.float32) * layer.input_scale_inv.detach().to(torch.float32)).reshape(1)
    layer._w4a16_alpha_cache = (key, a)
    return a


def _layer_has_native_layout(layer: torch.nn.Module, backend_value: str) -> bool:
    w = getattr(layer, "weight", None)
    s = getattr(layer, "weight_scale_interleaved", None)
    return (
        backend_value in _NATIVE_LAYOUT_BACKENDS
        and w is not None
        and s is not None
        and w.dtype == torch.uint8
        and w.dim() == 2
        and w.numel() > 0  # swiglu-interleave frees the plain weight
        and s.numel() > 0
        and getattr(layer, "weights_padding_cols", 0) == 0
    )


def maybe_apply_sm12x_w4a16(
    layer: torch.nn.Module,
    x,
    bias: Optional[torch.Tensor],
    backend_value: str,
) -> Optional[torch.Tensor]:
    """W4A16 native output for this call, or None (caller runs its W4A4 path)."""
    max_m = sm12x_w4a16_max_m()
    if max_m <= 0 or isinstance(x, tuple):
        return None
    if x.dtype != torch.bfloat16 or x.dim() < 1:
        return None
    m = x.numel() // x.shape[-1] if x.shape[-1] else 0
    if not 1 <= m <= max_m:
        return None
    if not _rank_on() or not _layer_has_native_layout(layer, backend_value):
        return None
    from flashinfer import mm_bf16_fp4

    x2 = x.reshape(m, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    out = mm_bf16_fp4(
        x2,
        layer.weight,
        layer.weight_scale_interleaved,
        w4a16_alpha(layer),
        backend=NATIVE_BACKEND,
        out_dtype=x.dtype,
    )
    n = int(getattr(layer, "output_size_per_partition", out.shape[-1]))
    if out.shape[-1] != n:
        out = out[:, :n]
    if bias is not None:
        out = out + bias
    return out.reshape(*x.shape[:-1], n)
