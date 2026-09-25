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
"""

from __future__ import annotations

import torch

KERNEL_NAME = "nvfp4_w4a8_int8_sm86"


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
