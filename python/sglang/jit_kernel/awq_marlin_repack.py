from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once_per_arch, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once_per_arch
def _jit_awq_marlin_repack_module() -> Module:
    return load_jit(
        "awq_marlin_repack",
        cuda_files=["gemm/marlin/awq_marlin_repack.cuh"],
        cuda_wrappers=[("awq_marlin_repack", "awq_marlin_repack")],
    )


@debug_kernel_api
def awq_marlin_repack(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    tile_size = 16
    pack_factor = 32 // num_bits
    out = torch.empty(
        (size_k // tile_size, size_n * tile_size // pack_factor),
        dtype=b_q_weight.dtype,
        device=b_q_weight.device,
    )
    module = _jit_awq_marlin_repack_module()
    module.awq_marlin_repack(out, b_q_weight, size_k, size_n, num_bits)
    return out


@debug_kernel_api
def awq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty(
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = awq_marlin_repack(b_q_weight[e], size_k, size_n, num_bits)
    return output


# =============================================================================
# H88-A (2026-10-07): AWQ (uint4 + zero point) repack to the Marlin W4A8
# (int8-activation) layout. New; the A16 repack above is unchanged. Source:
# csrc/gemm/marlin_a8/ (vendored from vLLM PR #24722, Apache-2.0).
# =============================================================================


@cache_once_per_arch
def _jit_awq_marlin_repack_a8_module() -> Module:
    return load_jit(
        "awq_marlin_repack_a8",
        cuda_files=["gemm/marlin_a8/awq_marlin_repack_a8.cuh"],
        cuda_wrappers=[("awq_marlin_repack_a8", "awq_marlin_repack_a8")],
    )


@debug_kernel_api
def awq_marlin_repack_w4a8(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int = 4,
) -> torch.Tensor:
    """AWQ-packed int32 [size_k, size_n / 8] -> W4A8 Marlin layout int32
    [size_k / 16, size_n * 16 / 8]. size_k must be a multiple of 32, size_n of 64."""
    if num_bits != 4:
        raise ValueError(f"awq_marlin_repack_w4a8 supports num_bits == 4 only, got {num_bits}")
    if size_k % 32 != 0:
        raise ValueError(f"size_k={size_k} must be a multiple of 32 for the int8-activation layout")
    tile_size = 16
    pack_factor = 32 // num_bits
    out = torch.empty(
        (size_k // tile_size, size_n * tile_size // pack_factor),
        dtype=b_q_weight.dtype,
        device=b_q_weight.device,
    )
    module = _jit_awq_marlin_repack_a8_module()
    module.awq_marlin_repack_a8(b_q_weight, out, size_k, size_n, num_bits)
    return out


def awq_marlin_moe_repack_w4a8(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int = 4,
) -> torch.Tensor:
    """Per-expert awq_marlin_repack_w4a8 over a leading expert dimension."""
    num_experts = b_q_weight.shape[0]
    output = torch.empty(
        (num_experts, size_k // 16, size_n * 16 // (32 // num_bits)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = awq_marlin_repack_w4a8(b_q_weight[e], size_k, size_n, num_bits)
    return output
