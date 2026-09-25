"""sm_86 W4A8 kernel for ``--fp4-gemm-backend native-mixed`` (Backlog #38).

The seam lives in ``nvfp4_native_mixed`` (agent N4B): it imports THIS module by name and expects it to call
``register_w4a8_kernel(fn, name)`` with

    fn(x, weight, weight_scale_swizzled, weight_global_scale, out_features) -> out

x bf16 [M, K]; weight uint8 [N, K/2] (native E2M1); weight_scale_swizzled float8_e4m3fn
[ceil128(N), ceil4(K/16)] in the 128x4 swizzle (a K-shard of a row-parallel layer arrives as the native swizzle
of its own [N, K_r/16] shard -- the exchange's tile view is geometry only, never the parameter's shape);
weight_global_scale fp32 0-dim (max(weight_scale_2)); out bf16 [M, out_features].

Importing this module in a tree without the seam does nothing (the kernel itself lives in
``sglang.jit_kernel.nvfp4_w4a8`` and is usable directly).

Small M (decode, M <= ``SGLANG_W4A8_DECODE_MAX_M``, default 48) runs agent N4D's decode GEMV
(``sglang.jit_kernel.nvfp4_w4a8_decode``) on the same bytes with the same arithmetic; larger M and shapes it does
not take (K not a multiple of 128, K padding) run N4A's tiled GEMM. ``SGLANG_W4A8_DECODE_MAX_M=0`` turns the
decode GEMV off. Both only ever run when native-mixed resolved ``w4a8_int8`` for the rank -- the
sm_8x default under ``--fp4-gemm-backend native-mixed`` (``SGLANG_FP4_NATIVE_MIXED_SM8X=marlin`` opts
out); boots without native-mixed never import a kernel from here.
"""

from __future__ import annotations

import torch

KERNEL_NAME = "nvfp4_w4a8_int8_sm86+decode"


def nvfp4_w4a8_int8_apply(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"{KERNEL_NAME}: activations must be bf16 (got {x.dtype}); the 27B line serves bf16"
        )
    from sglang.jit_kernel import nvfp4_w4a8_decode as dec

    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    if 0 < x2.shape[0] <= dec.decode_max_m() and dec.eligible(k, weight):
        if x2.stride(-1) != 1 or x2.data_ptr() % 16 != 0 or x2.stride(0) % 8 != 0:
            x2 = x2.contiguous()
        y = dec.nvfp4_w4a8_decode_linear(
            x2, weight, weight_scale_swizzled, weight_global_scale, int(out_features)
        )
        return y.view(*x.shape[:-1], int(out_features))

    from sglang.jit_kernel.nvfp4_w4a8 import nvfp4_w4a8_linear

    return nvfp4_w4a8_linear(
        x, weight, weight_scale_swizzled, weight_global_scale, int(out_features)
    )


try:  # the seam exists only in trees that carry native-mixed
    from sglang.srt.layers.quantization.nvfp4_native_mixed import register_w4a8_kernel
except ImportError:  # pragma: no cover - tree without the seam
    register_w4a8_kernel = None

if register_w4a8_kernel is not None:
    register_w4a8_kernel(nvfp4_w4a8_int8_apply, KERNEL_NAME)
