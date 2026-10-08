from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once_per_arch, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from sgl_kernel.scalar_type import ScalarType
    from tvm_ffi.module import Module

# Constants matching device::marlin:: in marlin.cuh
_MAX_THREAD_N = 256


@cache_once_per_arch
def _jit_gptq_marlin_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "gptq_marlin",
        *args,
        cuda_files=["gemm/marlin/gptq_marlin.cuh"],
        cuda_wrappers=[("gptq_marlin_gemm", f"gptq_marlin_gemm<{args}>")],
    )


def _or_empty(
    t: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return t if t is not None else torch.empty(0, device=device, dtype=dtype)


@debug_kernel_api
def gptq_marlin_gemm(
    a: torch.Tensor,
    c: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    b_zeros: Optional[torch.Tensor],
    g_idx: Optional[torch.Tensor],
    perm: Optional[torch.Tensor],
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    device = a.device

    # Allocate output if not provided
    if c is None:
        c = torch.empty((size_m, size_n), dtype=a.dtype, device=device)

    # Early return for zero-size M
    if size_m == 0:
        return c

    # Determine activation ordering
    has_act_order = (
        g_idx is not None
        and perm is not None
        and g_idx.numel() > 0
        and perm.numel() > 0
    )

    # Allocate c_tmp for fp32 reduce
    if use_fp32_reduce:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_m_block = min(((size_m + 15) // 16) * 16, 64)
        c_tmp = torch.empty(
            sms * max_m_block * _MAX_THREAD_N,
            dtype=torch.float32,
            device=device,
        )
    else:
        c_tmp = torch.empty(0, dtype=torch.float32, device=device)

    # Allocate a_tmp for act_order column permutation
    if has_act_order:
        a_tmp = torch.empty((size_m, size_k), dtype=a.dtype, device=device)
    else:
        a_tmp = torch.empty(0, dtype=a.dtype, device=device)

    # Convert Optional tensors to empty tensors
    global_scale_t = _or_empty(global_scale, device, a.dtype)
    b_zeros_t = _or_empty(b_zeros, device, torch.int32)
    g_idx_t = _or_empty(g_idx, device, torch.int32)
    perm_t = _or_empty(perm, device, torch.int32)

    module = _jit_gptq_marlin_module(a.dtype)
    module.gptq_marlin_gemm(
        a,
        b_q_weight,
        b_scales,
        global_scale_t,
        b_zeros_t,
        g_idx_t,
        perm_t,
        c,
        c_tmp,
        a_tmp,
        workspace,
        b_q_type.id,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )

    return c


# =============================================================================
# H88-A (2026-10-07): Marlin W4A8 -- int4 weights x int8 activations (dense).
#
# Everything BELOW this line is new; everything above is the unchanged A16
# path. The kernel is a separate JIT module (csrc/gemm/marlin_a8/, vendored
# from vLLM PR #24722, Apache-2.0) so the A16 build hash, JIT cache entries and
# call sites are untouched. Contract and CPU reference: marlin_w4a8_utils.py.
# =============================================================================


@cache_once_per_arch
def _jit_gptq_marlin_a8_module(dtype: torch.dtype) -> Module:
    # dtype = output dtype (fp16 or bf16); one module per dtype and per arch
    # (sm86 builds the sm80-capable source natively, sm120 builds as 12.0).
    args = make_cpp_args(dtype)
    return load_jit(
        "gptq_marlin_a8",
        *args,
        cuda_files=["gemm/marlin_a8/gptq_marlin_a8.cuh"],
        cuda_wrappers=[("gptq_marlin_gemm_a8", f"gptq_marlin_gemm_a8<{args}>")],
    )


def _b_q_type_name(b_q_type) -> str:
    # ScalarType.__str__ names: "uint4b8" (GPTQ, symmetric) / "uint4" (AWQ, zp)
    return str(b_q_type)


@debug_kernel_api
def gptq_marlin_gemm_w4a8(
    a: torch.Tensor,
    a_scales: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    b_zeros: Optional[torch.Tensor],
    b_bias: Optional[torch.Tensor],
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    c: Optional[torch.Tensor] = None,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = True,
) -> torch.Tensor:
    """out[m, n] = a_scales[m] * sum_k a[m, k] * dequant(W)[k, n]  (+ bias).

    a          int8 [M, K] per-token quantised (see marlin_w4a8_utils.per_token_quant_int8_ref
               or int8_kernel.per_token_quant_int8), row stride % 16 == 0
    a_scales   float32 [M]; for group scales (num_groups > 1) already multiplied
               with the factor returned by marlin_act_int8_process_scales
    b_q_weight int32, Marlin W4A8 layout (gptq_marlin_repack_w4a8 /
               awq_marlin_repack_w4a8)
    b_scales   fp16/bf16 [groups, N], permuted by marlin_permute_scales(is_a_8bit
               =True); int16 x4096-encoded when groups > 1
    b_zeros    int32 [groups, N/8] (asymmetric, b_q_type uint4) or None
    b_q_type   uint4b8 (symmetric) or uint4 (with b_zeros)
    Output dtype = b_scales dtype (fp16 or bf16). Group sizes -1/32/64/128.
    """
    from sglang.jit_kernel.marlin_w4a8_utils import (
        MARLIN_TILE,
        check_w4a8_gemm_args,
        w4a8_arch_support,
    )

    device = a.device
    cap = torch.cuda.get_device_capability(device)
    ok, why = w4a8_arch_support(*cap)
    if not ok:
        raise RuntimeError(f"gptq_marlin_gemm_w4a8: {why}")
    check_w4a8_gemm_args(
        a, a_scales, b_q_weight, b_scales, b_zeros, b_bias,
        _b_q_type_name(b_q_type), size_m, size_n, size_k,
    )  # fmt: skip
    out_dtype = b_scales.dtype

    if c is None:
        c = torch.empty((size_m, size_n), dtype=out_dtype, device=device)
    if size_m == 0:
        return c

    if use_fp32_reduce:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_m_block = min(((size_m + 15) // 16) * 16, 64)
        c_tmp = torch.empty(sms * max_m_block * _MAX_THREAD_N, dtype=torch.float32, device=device)
    else:
        c_tmp = torch.empty(0, dtype=torch.float32, device=device)

    zeros_t = _or_empty(b_zeros, device, torch.int32)
    bias_t = _or_empty(b_bias, device, out_dtype)

    module = _jit_gptq_marlin_a8_module(out_dtype)
    module.gptq_marlin_gemm_a8(
        a,
        a_scales.reshape(-1),
        b_q_weight,
        b_scales,
        zeros_t,
        bias_t,
        c,
        c_tmp,
        workspace,
        b_q_type.id,
        use_atomic_add,
        use_fp32_reduce,
    )
    return c
