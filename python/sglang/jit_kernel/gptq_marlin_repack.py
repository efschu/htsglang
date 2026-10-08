from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once_per_arch, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Constants matching device::marlin:: in marlin.cuh
_TILE_SIZE = 16


@cache_once_per_arch
def _jit_gptq_marlin_repack_module() -> Module:
    return load_jit(
        "gptq_marlin_repack",
        cuda_files=["gemm/marlin/gptq_marlin_repack.cuh"],
        cuda_wrappers=[("gptq_marlin_repack", "gptq_marlin_repack")],
    )


@debug_kernel_api
def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    pack_factor = 32 // num_bits

    # Allocate output tensor
    out = torch.empty(
        (size_k // _TILE_SIZE, size_n * _TILE_SIZE // pack_factor),
        dtype=b_q_weight.dtype,
        device=b_q_weight.device,
    )

    module = _jit_gptq_marlin_repack_module()
    module.gptq_marlin_repack(b_q_weight, perm, out, size_k, size_n, num_bits)
    return out


# =============================================================================
# H88-A (2026-10-07): repack to the Marlin W4A8 (int8-activation) weight layout.
# New; the A16 repack above is unchanged. Source: csrc/gemm/marlin_a8/
# (vendored from vLLM PR #24722, Apache-2.0). CPU reference of the layout:
# marlin_w4a8_utils.marlin_weights(..., is_a_8bit=True).
# =============================================================================


@cache_once_per_arch
def _jit_gptq_marlin_repack_a8_module() -> Module:
    return load_jit(
        "gptq_marlin_repack_a8",
        cuda_files=["gemm/marlin_a8/gptq_marlin_repack_a8.cuh"],
        cuda_wrappers=[("gptq_marlin_repack_a8", "gptq_marlin_repack_a8")],
    )


@debug_kernel_api
def gptq_marlin_repack_w4a8(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int = 4,
) -> torch.Tensor:
    """GPTQ-packed int32 [size_k / 8, size_n] (no act-order permutation) ->
    W4A8 Marlin layout int32 [size_k / 16, size_n * 16 / 8]. size_k must be a
    multiple of 32 (the int8 tile is 32 x 32), size_n of 64."""
    if num_bits != 4:
        raise ValueError(f"gptq_marlin_repack_w4a8 supports num_bits == 4 only, got {num_bits}")
    if size_k % 32 != 0:
        raise ValueError(f"size_k={size_k} must be a multiple of 32 for the int8-activation layout")
    pack_factor = 32 // num_bits
    out = torch.empty(
        (size_k // _TILE_SIZE, size_n * _TILE_SIZE // pack_factor),
        dtype=b_q_weight.dtype,
        device=b_q_weight.device,
    )
    module = _jit_gptq_marlin_repack_a8_module()
    module.gptq_marlin_repack_a8(b_q_weight, out, size_k, size_n, num_bits)
    return out


def gptq_marlin_moe_repack_w4a8(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int = 4,
) -> torch.Tensor:
    """Per-expert gptq_marlin_repack_w4a8 over a leading expert dimension."""
    num_experts = b_q_weight.shape[0]
    pack_factor = 32 // num_bits
    out = torch.empty(
        (num_experts, size_k // _TILE_SIZE, size_n * _TILE_SIZE // pack_factor),
        dtype=b_q_weight.dtype,
        device=b_q_weight.device,
    )
    for e in range(num_experts):
        out[e] = gptq_marlin_repack_w4a8(b_q_weight[e], size_k, size_n, num_bits)
    return out
