"""Marlin W4A8 (int4 weights, int8 activations): host-side format contract.

Pure torch/numpy, no JIT import and no CUDA call, so it can be imported and
tested on a machine without a GPU. It holds

* the shape/dtype contract of the W4A8 kernels (`w4a8_expected_shapes`,
  `check_w4a8_gemm_args`, `check_w4a8_moe_args`),
* the scale / zero-point preparation the kernels expect
  (`marlin_permute_scales`, `marlin_act_int8_process_scales`,
  `marlin_zero_points`),
* a CPU reference for the weight layout (`get_weight_perm`,
  `marlin_permute_weights`, `marlin_weights`) -- the independent definition the
  CUDA repack kernels (`gptq_marlin_repack_a8` / `awq_marlin_repack_a8`) are
  tested against on the GPU,
* a CPU reference GEMM (`reference_w4a8_gemm`, `emulate_w4a8_kernel`) and the
  tolerance the GPU numerics test uses.

Provenance (H88-A, 2026-10-07): the permutations, the int16 x4096 scale trick
and the test reference follow vLLM (local fork /spinning/shvllm,
vllm/model_executor/layers/quantization/utils/marlin_utils.py and
marlin_utils_test.py; PR #24722, jinzhen-lin; Apache-2.0, Marlin (C) 2024 Elias
Frantar, IST-DASLab). They are re-implemented here, not imported.

The A16 Marlin helpers in srt/layers/quantization/marlin_utils.py are not
touched; nothing in this module changes the A16 path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: Marlin tile (k rows per repacked row of b_q_weight)
MARLIN_TILE = 16
#: smallest n the kernels accept (device::marlin min_thread_n)
MARLIN_MIN_THREAD_N = 64
#: group sizes with an int8-activation kernel: -1 = channelwise
W4A8_GROUP_SIZES = (-1, 32, 64, 128)
#: group scales are stored as round(s / s.max() * 4096) in an int16
W4A8_SCALE_INT_RANGE = 4096
#: moe_block_size values with an int8 kernel (moe_block_size 8 has none)
W4A8_MOE_BLOCK_SIZES = (16, 32, 48, 64)
#: weight types: uint4b8 = GPTQ/symmetric (bias 8), uint4 = AWQ/asymmetric + zp
W4A8_WEIGHT_TYPES = ("uint4b8", "uint4")
#: compute capabilities the module is built for / meant for. sm86 runs the
#: sm80 SASS of the same source, sm120 gets its own build (12.0).
W4A8_TESTED_ARCHS = ((8, 6), (12, 0))
#: GPU numerics tolerance (upstream vLLM test_marlin_gemm): mean|out-ref|/mean|ref|
W4A8_GPU_MAX_DIFF = 0.04


def w4a8_arch_support(major: int, minor: int) -> Tuple[bool, str]:
    """(supported, reason). The int8-activation kernels need mma.sync m16n8k32
    with s8 and cp.async: compute capability >= 8.0. sm86 and sm120 are the two
    architectures this port is meant for; sm80/sm89 are accepted as well, older
    ones are refused (the A16 path still serves them)."""
    cc = major * 10 + minor
    if cc < 80:
        return False, f"compute capability {major}.{minor} < 8.0: no int8-activation Marlin kernel"
    if (major, minor) in W4A8_TESTED_ARCHS:
        return True, f"sm{cc}: built from the shared source (sm86 uses the sm80 SASS path)"
    return True, f"sm{cc}: accepted but not part of the H88-A test matrix (sm86, sm120)"


# ---------------------------------------------------------------------------
# shape contract
# ---------------------------------------------------------------------------


def w4a8_group_count(size_k: int, group_size: int) -> int:
    """Number of scale rows for a group size (-1 = channelwise -> 1)."""
    if group_size == -1:
        return 1
    if group_size not in W4A8_GROUP_SIZES:
        raise ValueError(
            f"group_size must be one of {W4A8_GROUP_SIZES} for Marlin W4A8, got {group_size}"
        )
    if size_k % group_size != 0:
        raise ValueError(f"size_k={size_k} is not divisible by group_size={group_size}")
    return size_k // group_size


def w4a8_expected_shapes(
    size_k: int, size_n: int, group_size: int, has_zp: bool, num_experts: Optional[int] = None
) -> dict:
    """Shapes of the tensors a W4A8 GEMM (dense, or MoE with a leading expert
    dimension) takes. The repacked weight has the same shape as the A16 layout;
    only its internal tile order differs (32x32 tiles instead of 16x64)."""
    if size_k % MARLIN_TILE != 0:
        raise ValueError(f"size_k={size_k} must be a multiple of {MARLIN_TILE}")
    if size_n % MARLIN_MIN_THREAD_N != 0:
        raise ValueError(f"size_n={size_n} must be a multiple of {MARLIN_MIN_THREAD_N}")
    pack_factor = 8  # int4 in int32
    groups = w4a8_group_count(size_k, group_size)
    lead = () if num_experts is None else (num_experts,)
    shapes = {
        "b_q_weight": lead + (size_k // MARLIN_TILE, size_n * MARLIN_TILE // pack_factor),
        "b_scales": lead + (groups, size_n),
        "b_zeros": (lead + (groups, size_n // pack_factor)) if has_zp else (0,),
        "bias": (lead + (size_n,)) if num_experts is None else (num_experts, size_n),
    }
    return shapes


def _is_fp16_bf16(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16)


def check_w4a8_gemm_args(
    a: torch.Tensor,
    a_scales: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    b_zeros: Optional[torch.Tensor],
    b_bias: Optional[torch.Tensor],
    b_q_type_name: str,
    size_m: int,
    size_n: int,
    size_k: int,
) -> int:
    """Validate the dense contract on shapes/dtypes only (no device check).
    Returns the group size. Raises ValueError with the offending quantity."""
    if b_q_type_name not in W4A8_WEIGHT_TYPES:
        raise ValueError(f"b_q_type must be one of {W4A8_WEIGHT_TYPES}, got {b_q_type_name}")
    has_zp = b_zeros is not None and b_zeros.numel() > 0
    if has_zp and b_q_type_name != "uint4":
        raise ValueError("b_zeros given: b_q_type must be uint4 (asymmetric)")
    if not has_zp and b_q_type_name != "uint4b8":
        raise ValueError("no b_zeros: b_q_type must be uint4b8 (symmetric)")
    if a.dtype != torch.int8:
        raise ValueError(f"a must be int8 (per-token quantised), got {a.dtype}")
    if tuple(a.shape) != (size_m, size_k):
        raise ValueError(f"a shape {tuple(a.shape)} != (size_m, size_k) = {(size_m, size_k)}")
    if a.stride(-1) != 1 or a.stride(0) % 16 != 0:
        raise ValueError("a rows must be contiguous and 16-element aligned (a.stride(0) % 16 == 0)")
    if a_scales.dtype != torch.float32 or tuple(a_scales.reshape(-1).shape) != (size_m,):
        raise ValueError(
            f"a_scales must be float32 with {size_m} entries (one per token), got "
            f"{a_scales.dtype} {tuple(a_scales.shape)}"
        )
    if not _is_fp16_bf16(b_scales.dtype):
        raise ValueError(f"b_scales must be float16 or bfloat16 (= output dtype), got {b_scales.dtype}")
    if b_scales.dim() != 2 or b_scales.shape[1] != size_n:
        raise ValueError(f"b_scales shape {tuple(b_scales.shape)} must be (groups, size_n={size_n})")
    groups = b_scales.shape[0]
    group_size = -1 if groups == 1 else size_k // groups
    if groups > 1 and size_k % groups != 0:
        raise ValueError(f"size_k={size_k} not divisible by number of scale groups {groups}")
    exp = w4a8_expected_shapes(size_k, size_n, group_size, has_zp)
    if b_q_weight.dtype != torch.int32 or tuple(b_q_weight.shape) != exp["b_q_weight"]:
        raise ValueError(
            f"b_q_weight must be int32 {exp['b_q_weight']}, got {b_q_weight.dtype} {tuple(b_q_weight.shape)}"
        )
    if has_zp:
        if b_zeros.dtype != torch.int32 or tuple(b_zeros.shape) != exp["b_zeros"]:
            raise ValueError(
                f"b_zeros must be int32 {exp['b_zeros']}, got {b_zeros.dtype} {tuple(b_zeros.shape)}"
            )
    if b_bias is not None and b_bias.numel() > 0:
        if b_bias.dtype != b_scales.dtype or tuple(b_bias.shape) != (size_n,):
            raise ValueError(
                f"b_bias must be {b_scales.dtype} ({size_n},), got {b_bias.dtype} {tuple(b_bias.shape)}"
            )
    return group_size


def check_w4a8_moe_args(
    a: torch.Tensor,
    a_scales: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    b_zeros: Optional[torch.Tensor],
    b_bias: Optional[torch.Tensor],
    b_q_type_name: str,
    moe_block_size: int,
    top_k: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> int:
    """MoE counterpart of check_w4a8_gemm_args (leading expert dimension on the
    weight tensors, C is [size_m * top_k, size_n]). Returns the group size."""
    if moe_block_size not in W4A8_MOE_BLOCK_SIZES:
        raise ValueError(
            f"moe_block_size must be one of {W4A8_MOE_BLOCK_SIZES} for Marlin W4A8 "
            f"(no int8 kernel for 8), got {moe_block_size}"
        )
    if b_q_type_name not in W4A8_WEIGHT_TYPES:
        raise ValueError(f"b_q_type must be one of {W4A8_WEIGHT_TYPES}, got {b_q_type_name}")
    has_zp = b_zeros is not None and b_zeros.numel() > 0
    if has_zp != (b_q_type_name == "uint4"):
        raise ValueError("b_zeros must be given exactly for b_q_type uint4 (asymmetric)")
    if a.dtype != torch.int8 or tuple(a.shape) != (size_m, size_k) or not a.is_contiguous():
        raise ValueError(f"a must be contiguous int8 ({size_m}, {size_k}), got {a.dtype} {tuple(a.shape)}")
    if a_scales.dtype != torch.float32 or tuple(a_scales.reshape(-1).shape) != (size_m,):
        raise ValueError(f"a_scales must be float32 with {size_m} entries, got {a_scales.dtype} {tuple(a_scales.shape)}")
    if not _is_fp16_bf16(b_scales.dtype) or b_scales.dim() != 3:
        raise ValueError("b_scales must be float16/bfloat16 [experts, groups, size_n]")
    num_experts, groups, n = b_scales.shape
    if n != size_n:
        raise ValueError(f"b_scales last dim {n} != size_n {size_n}")
    group_size = -1 if groups == 1 else size_k // groups
    exp = w4a8_expected_shapes(size_k, size_n, group_size, has_zp, num_experts)
    if b_q_weight.dtype != torch.int32 or tuple(b_q_weight.shape) != exp["b_q_weight"]:
        raise ValueError(
            f"b_q_weight must be int32 {exp['b_q_weight']}, got {b_q_weight.dtype} {tuple(b_q_weight.shape)}"
        )
    if has_zp and (b_zeros.dtype != torch.int32 or tuple(b_zeros.shape) != exp["b_zeros"]):
        raise ValueError(f"b_zeros must be int32 {exp['b_zeros']}, got {b_zeros.dtype} {tuple(b_zeros.shape)}")
    if b_bias is not None and b_bias.numel() > 0:
        if b_bias.dtype != b_scales.dtype or tuple(b_bias.shape) != (num_experts, size_n):
            raise ValueError(f"b_bias must be {b_scales.dtype} ({num_experts}, {size_n})")
    return group_size


# ---------------------------------------------------------------------------
# scales, zero points
# ---------------------------------------------------------------------------


def get_scale_perms():
    scale_perm: list = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int, group_size: int, is_a_8bit: bool = True) -> torch.Tensor:
    """Permute [groups, size_n] scales into the kernel's fragment order. With
    8-bit activations the 'single' permutation is used for every group size."""
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def marlin_permute_bias(b: torch.Tensor) -> torch.Tensor:
    shape = b.shape
    _, scale_perm_single = get_scale_perms()
    b = b.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return b.reshape(*shape).contiguous()


def marlin_act_int8_process_scales(s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """int16 x4096 trick for GROUP scales (not for channelwise scales).

    Returns (s_int16_in_fp16_container, factor): s_int = round(s / s.max() *
    4096) as int16, its bit pattern viewed as s.dtype; factor = s.max() / 4096
    (float32 scalar) which the caller multiplies into the per-token a_scales.
    One factor per scale TENSOR (for MoE: per layer tensor over all experts,
    exactly as vLLM does)."""
    # Computed in float32 (vLLM divides in s.dtype: for bf16 that costs ~0.4 %
    # of scale precision before the int16 rounding even starts; the kernel only
    # sees the resulting integers, so the better quotient is free).
    sf = s.float()
    smax = sf.max()
    factor = smax / W4A8_SCALE_INT_RANGE
    s_int = (sf / smax * W4A8_SCALE_INT_RANGE).round().to(torch.int16).view(s.dtype)
    return s_int, factor


def marlin_act_int8_decode_scales(s_int_view: torch.Tensor, factor: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """Inverse of marlin_act_int8_process_scales (value = int16 * factor)."""
    return s_int_view.view(torch.int16).to(dtype) * factor.to(dtype)


def get_pack_factor(num_bits: int) -> int:
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def pack_cols(q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int) -> torch.Tensor:
    assert q_w.shape == (size_k, size_n)
    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0
    q = q_w.cpu().numpy().astype(np.uint32)
    res = np.zeros((size_k, size_n // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        res |= q[:, i::pack_factor] << num_bits * i
    return torch.from_numpy(res.astype(np.int32)).contiguous()


def unpack_cols(packed: torch.Tensor, num_bits: int, size_k: int, size_n: int) -> torch.Tensor:
    pack_factor = get_pack_factor(num_bits)
    assert packed.shape == (size_k, size_n // pack_factor)
    p = packed.cpu().numpy().astype(np.uint32)
    res = np.zeros((size_k, size_n), dtype=np.uint32)
    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        res[:, i::pack_factor] = p & mask
        p = p >> num_bits
    return torch.from_numpy(res.astype(np.int32))


def marlin_zero_points(zp: torch.Tensor, size_k: int, size_n: int, num_bits: int = 4, is_a_8bit: bool = True) -> torch.Tensor:
    """Permute + pack AWQ-style zero points [groups, size_n] to int32
    [groups, size_n / 8]. With 8-bit activations the column interleave is
    skipped (the int8 dequant reads nibbles in natural order)."""
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]
    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")
    if not is_a_8bit:
        zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    return pack_cols(zp, num_bits, zp.shape[0], size_n)


# ---------------------------------------------------------------------------
# weight layout reference (what the CUDA repack kernels must produce)
# ---------------------------------------------------------------------------


def get_weight_perm(num_bits: int = 4, is_a_8bit: bool = True) -> torch.Tensor:
    perm_list: list = []
    if is_a_8bit:
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    4 * (i % 4),
                    4 * (i % 4) + 1,
                    4 * (i % 4) + 2,
                    4 * (i % 4) + 3,
                    4 * (i % 4 + 4),
                    4 * (i % 4 + 4) + 1,
                    4 * (i % 4 + 4) + 2,
                    4 * (i % 4 + 4) + 3,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(2):
                perm_list.extend([p + 512 * j for p in perm1])
    else:
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm_list.extend([p + 256 * j for p in perm1])
    perm = np.array(perm_list)
    if num_bits == 4:
        interleave = np.array([0, 4, 1, 5, 2, 6, 3, 7]) if is_a_8bit else np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 1, 2, 3]) if is_a_8bit else np.array([0, 2, 1, 3])
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")
    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    return torch.from_numpy(perm)


def marlin_permute_weights(q_w: torch.Tensor, size_k: int, size_n: int, perm: torch.Tensor, tile: int = MARLIN_TILE, is_a_8bit: bool = True) -> torch.Tensor:
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_n = {size_n}, tile = {tile}"
    if is_a_8bit:
        # 32x32 marlin tiles
        assert size_k % (tile * 2) == 0, f"size_k = {size_k} must be a multiple of {tile * 2} for the int8 layout"
        q_w = q_w.reshape((size_k // (tile * 2), tile * 2, size_n // tile, tile))
    else:
        q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))
    return q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)


def marlin_weights(q_w: torch.Tensor, size_k: int, size_n: int, num_bits: int, perm: torch.Tensor, is_a_8bit: bool = True) -> torch.Tensor:
    """Permute + pack unpacked int weights [size_k, size_n] (values 0..15 for
    4 bit, bias already added) to the kernel layout int32 [size_k/16, size_n*16/8]."""
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm, is_a_8bit=is_a_8bit)
    pack_factor = get_pack_factor(num_bits)
    q = q_w.cpu().numpy().astype(np.uint32)
    packed = np.zeros((q.shape[0], q.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        packed |= q[:, i::pack_factor] << num_bits * i
    return torch.from_numpy(packed.astype(np.int32))


def marlin_unpack_weights(packed: torch.Tensor, size_k: int, size_n: int, num_bits: int, perm: torch.Tensor, is_a_8bit: bool = True) -> torch.Tensor:
    """Inverse of marlin_weights (used by the CPU round-trip test)."""
    pack_factor = get_pack_factor(num_bits)
    p = packed.cpu().numpy().astype(np.uint32)
    rows, cols = p.shape[0], p.shape[1] * pack_factor
    q = np.zeros((rows, cols), dtype=np.uint32)
    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        q[:, i::pack_factor] = (p >> (num_bits * i)) & mask
    q = torch.from_numpy(q.astype(np.int64))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
    q = q.reshape((-1, perm.numel()))[:, inv].reshape(q.shape)
    tile = MARLIN_TILE
    if is_a_8bit:
        q = q.reshape((size_k // (tile * 2), size_n // tile, tile * 2, tile)).permute((0, 2, 1, 3))
    else:
        q = q.reshape((size_k // tile, size_n // tile, tile, tile)).permute((0, 2, 1, 3))
    return q.reshape((size_k, size_n))


# ---------------------------------------------------------------------------
# quantisation + reference GEMM
# ---------------------------------------------------------------------------


def quantize_weights_ref(w: torch.Tensor, group_size: int, asym: bool):
    """Reference 4-bit weight quantisation of w [size_k, size_n] (float).

    sym  (asym=False): uint4b8 -- q in [-8, 7] stored as q + 8; returns
         (w_ref, q_stored [0..15], s [groups, n], None)
    asym (asym=True):  uint4 with zero points -- q in [0, 15], w ~ (q - zp) * s;
         returns (w_ref, q_stored, s, zp [groups, n])
    group_size -1 = channelwise (one group)."""
    size_k, size_n = w.shape
    gs = size_k if group_size == -1 else group_size
    assert size_k % gs == 0
    wg = w.reshape(size_k // gs, gs, size_n)
    if asym:
        mx = wg.amax(dim=1, keepdim=True)
        mn = wg.amin(dim=1, keepdim=True)
        s = (mx - mn).clamp(min=1e-5) / 15.0
        zp = torch.round(torch.abs(mn / s)).clamp(0, 15)
        q = (torch.round(wg / s) + zp).clamp(0, 15)
        w_ref = (q - zp) * s
        zp_out = zp.reshape(-1, size_n).to(torch.int32)
        q_stored = q
    else:
        s = torch.maximum(wg.amax(dim=1, keepdim=True).abs() / 7.0, wg.amin(dim=1, keepdim=True).abs() / 8.0)
        s = s.clamp(min=1e-8)
        q = torch.round(wg / s).clamp(-8, 7)
        w_ref = q * s
        q_stored = q + 8
        zp_out = None
    return (
        w_ref.reshape(size_k, size_n),
        q_stored.reshape(size_k, size_n).to(torch.int32),
        s.reshape(-1, size_n),
        zp_out,
    )


def per_token_quant_int8_ref(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token int8: scale = absmax / 127, q = round(x / scale).
    Returns (int8 [M, K], float32 scales [M])."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1).clamp(min=1e-10)
    scale = amax / 127.0
    q = torch.round(xf / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    return q, scale.float()


def reference_w4a8_gemm(a_q: torch.Tensor, a_scales: torch.Tensor, w_ref: torch.Tensor) -> torch.Tensor:
    """fp64 reference: (a_q * a_scale) @ w_ref -- dequantised int8 activations
    times the dequantised weights (the quantity the GPU numerics test uses)."""
    a_deq = a_q.double() * a_scales.double().reshape(-1, 1)
    return a_deq @ w_ref.double()


def emulate_w4a8_kernel(
    a_q: torch.Tensor,
    a_scales: torch.Tensor,
    q_stored: torch.Tensor,
    s: torch.Tensor,
    zp: Optional[torch.Tensor],
    group_size: int,
    use_int16_scales: bool,
) -> torch.Tensor:
    """What the kernel computes, in fp64: per group an exact integer dot
    sum_k a_q * (q - zp_or_bias), multiplied by the group scale. With
    use_int16_scales the scale is the int16 x4096 approximation and the
    per-token scale carries the s.max()/4096 factor (the GPU path); without it
    the original scale is used (channelwise path). Matches reference_w4a8_gemm
    up to the int16 rounding of the scales."""
    size_k, size_n = q_stored.shape
    gs = size_k if group_size == -1 else group_size
    groups = size_k // gs
    if zp is None:
        centered = q_stored.double() - 8.0
    else:
        centered = q_stored.double() - zp.double().repeat_interleave(gs, dim=0)
    if use_int16_scales:
        s_int, factor = marlin_act_int8_process_scales(s.float())
        s_eff = s_int.view(torch.int16).double()
        a_eff = a_scales.double() * float(factor)
    else:
        s_eff = s.double()
        a_eff = a_scales.double()
    out = torch.zeros((a_q.shape[0], size_n), dtype=torch.float64)
    for g in range(groups):
        ks = slice(g * gs, (g + 1) * gs)
        part = a_q[:, ks].double() @ centered[ks, :]
        out += part * s_eff[g].reshape(1, -1)
    return out * a_eff.reshape(-1, 1)


def make_workspace(device, max_blocks_per_sm: int = 4) -> torch.Tensor:
    """Zero-initialised int32 lock buffer: the dense kernel needs >= #SMs
    entries, the MoE kernel min(n_tiles * blocks, sms * 4). The kernels restore
    the zeros, so one buffer per device can be reused."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int32, device=device)
