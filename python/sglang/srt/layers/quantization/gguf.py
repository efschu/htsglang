# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/ab3e80042eac24dd362408e6d63ad98768046359/vllm/model_executor/layers/quantization/gguf.py
from __future__ import annotations

import bisect
import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, List, Optional

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter, UninitializedParameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt import knob_resolution as _knob
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_is_musa = is_musa()
_is_npu = is_npu()

_has_sgl_gguf_kernels = False
if _is_cuda:
    # Turing (sm75): sgl-kernel is cubin-only with a gencode floor of sm_80 and
    # no PTX, so it holds no code a 2080/2080 Ti/T4 can execute. Guarding lets
    # the quantization package import on such a card; a rank that actually
    # loads a GGUF checkpoint fails loudly at GGUFConfig instead of dying on a
    # missing symbol mid-forward.
    try:
        from sgl_kernel import moe_sum
        from sgl_kernel.quantization import (
            ggml_dequantize,
            ggml_moe_a8,
            ggml_moe_a8_vec,
            ggml_moe_get_block_size,
            ggml_mul_mat_a8,
            ggml_mul_mat_vec_a8,
        )

        _has_sgl_gguf_kernels = True
    except ImportError:
        moe_sum = None
        ggml_dequantize = ggml_moe_a8 = ggml_moe_a8_vec = None
        ggml_moe_get_block_size = ggml_mul_mat_a8 = ggml_mul_mat_vec_a8 = None

    from sglang.jit_kernel.activation import gelu_and_mul, silu_and_mul
elif _is_musa:
    from sgl_kernel import gelu_and_mul, moe_sum, silu_and_mul
    from sgl_kernel.quantization import (
        ggml_dequantize,
        ggml_moe_a8,
        ggml_moe_a8_vec,
        ggml_moe_get_block_size,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )
elif _is_npu:
    from gguf import dequantize as gguf_dequantize
else:
    if not _is_hip:
        warnings.warn(f"Only CUDA, MUSA and NPU support GGUF quantization currently.")

logger = logging.getLogger(__name__)


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, modules_to_not_convert: list[str] | None = None) -> None:
        super().__init__()
        if _is_hip:
            warnings.warn(f"Only CUDA and MUSA support GGUF quantization currently.")
        self.modules_to_not_convert = modules_to_not_convert or []
        # Uneven-TP (--rank-tp-ratio) block alignment, consumed by
        # linear._quant_block_aligned_units([out_block, in_block]).
        # GGUF packs each row as `in_elems // ggml_block * type_size` opaque
        # bytes ALONG THE INPUT dim, so only input (block_idx=1) shard
        # boundaries must land on a quant block; use 256 = the K-quant
        # superblock (also a multiple of the 32-block standard quants), the
        # worst case since the per-tensor ggml type is unknown at layer build.
        # Both dims use 256 so that a column-parallel OUTPUT split (e.g. MLP
        # gate_up, GDN in_proj) lands on the SAME unit boundary as the coupled
        # row-parallel INPUT split (down_proj, out_proj) — otherwise the two
        # ends of the residual disagree on the per-rank intermediate/value size
        # under uneven TP (activation "hidden size must be divisible by vector
        # size" / matmul shape mismatch). Output rows are byte-whole regardless,
        # so 256 on block_idx=0 only affects the uneven-split granularity.
        self.weight_block_size = [256, 256]

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> str:
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # #269: sgl-kernel's GGUF cubins are gencode-restricted to sm_80+ (no
        # PTX fallback), so Turing (sm75; 2080/2080 Ti/T4) holds no code they
        # can execute (#212). This was previously 60 (Pascal) -- stale, and
        # low enough that a real sm75 card (capability 75) sailed past it
        # unnoticed in _enforce_capability_floor's NVIDIA-namespace numeric
        # path (model_loader/loader.py), so GGUF was only ever caught later,
        # deep in a kernel launch. supports_current_device() below answers
        # the same question functionally (via _has_sgl_gguf_kernels) and
        # takes priority when it has an opinion; this number is the fallback
        # for when it does not (e.g. that hook returns None off CUDA).
        return 80 if not _is_musa else 21

    def supports_current_device(self) -> Optional[bool]:
        """Vendor-first, functional answer for _enforce_capability_floor
        (#269): does a GGUF cubin actually exist for the card this process is
        bound to, rather than "is the capability number big enough".

        ``_has_sgl_gguf_kernels`` already records this precisely -- it is
        False iff importing the sgl-kernel GGUF cubins failed at process
        start (module import time, above), which is exactly the sm75 case
        this exists to catch (#212). Reusing it here means the fail-fast gate
        and the reason the import guard exists can never drift apart.

        None off CUDA (MUSA/NPU/CPU take their own path in
        _enforce_capability_floor / _get_quantization_config and are not this
        hook's concern; returning None there defers to get_min_capability()'s
        MUSA-specific floor above instead of reporting a false negative).
        """
        if _is_cuda:
            return _has_sgl_gguf_kernels
        return None

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GGUFConfig:
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            if _is_npu:
                return GGUFLinearAscendMethod(self)
            return GGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            # Allow skipping (dense weight) for embedding/lm_head too: returning
            # None makes VocabParallelEmbedding fall back to
            # UnquantizedEmbeddingMethod (a plain `.weight`). Needed for NEXTN
            # spec decode, which shares the target's lm_head.weight with the
            # draft (get_embed_and_head) — a quantized lm_head has no `.weight`.
            if is_layer_skipped_gguf(prefix, self.modules_to_not_convert):
                return None
            if _is_npu:
                return GGUFEmbeddingAscendMethod(self)
            return GGUFEmbeddingMethod(self)
        elif isinstance(layer, FusedMoE):
            if _is_npu:
                return GGUFMoEAscendMethod(self)
            return GGUFMoEMethod(self)
        return None


def is_layer_skipped_gguf(prefix: str, modules_to_not_convert: list[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}


def _weight_type_for_dtype(dtype: torch.dtype) -> WeightType:
    """ggml WeightType tag matching a torch floating dtype (dense tensors)."""
    return {
        torch.float32: WeightType.F32,
        torch.float16: WeightType.F16,
        torch.bfloat16: WeightType.BF16,
    }[dtype]
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
#: OCP microscaling FP4 (ggml type 39), native since #398. The kernel set is
#: complete -- dequantize (``dequantize_row_mxfp4_cuda``), MMVQ
#: (``vec_dot_mxfp4_q8_1``, dense + MoE) and MMQ (``mul_mat_mxfp4`` /
#: ``moe_mxfp4``) -- so MXFP4 joins all three sets rather than only DEQUANT.
#: A wheel built before #398 has none of them: see ``_mxfp4_kernels_present``.
MXFP4_QUANT_TYPES = {WeightType.MXFP4}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES


def _mxfp4_kernels_present() -> bool:
    """Whether the INSTALLED sgl-kernel wheel carries the #398 MXFP4 kernels.

    The wheel ships prebuilt and is pinned by sha256 (rig-runbook 2.1), so a
    tree that HAS this source change can still be running an older wheel; the
    type sets must follow the wheel, not the tree. Detected by EXISTENCE of the
    ``ggml_mxfp4_native`` op, the #73 pattern: its registration alone
    identifies the build, so the probe never has to call it. That was
    originally a necessity -- until #518 the op was registered for the CUDA
    dispatch key only and, taking no tensor, could not be routed at all
    ("no dispatchable fallback" on every call, every arch) -- and it stays
    the right shape afterwards, because existence is exactly the question and
    an existence check works on old and new wheels alike. Older wheels lack
    the op -> MXFP4 stays out of every set and the load-time MXFP4 -> Q5_0
    repack remains the only executable route, unchanged.

    ``SGLANG_GGUF_MXFP4_NATIVE=0`` forces the old behaviour on a new wheel --
    the A/B lever for the repack-vs-native comparison, and the fallback if the
    native path ever has to be taken out of a running configuration.
    """
    if os.environ.get("SGLANG_GGUF_MXFP4_NATIVE", "1")[:1] == "0":
        return False
    try:
        # Deliberately NOT gated on _is_cuda: the question is what the WHEEL
        # contains, and a host without a visible device never reaches a GGUF
        # kernel anyway (the ops are not even imported above). Gating on the
        # device would make the type sets untestable off-GPU.
        return hasattr(torch.ops.sgl_kernel, "ggml_mxfp4_native")
    except Exception:
        return False


MXFP4_NATIVE = _mxfp4_kernels_present()
if MXFP4_NATIVE:
    DEQUANT_TYPES = DEQUANT_TYPES | MXFP4_QUANT_TYPES
    MMVQ_QUANT_TYPES = MMVQ_QUANT_TYPES | MXFP4_QUANT_TYPES
    MMQ_QUANT_TYPES = MMQ_QUANT_TYPES | MXFP4_QUANT_TYPES

#: ggml types the GGUF MoE expert-offload half (#123) covers.
#:
#: ``fused_moe_gguf`` has exactly two real expert kernels -- ``ggml_moe_a8``
#: (MMQ, prefill) and ``ggml_moe_a8_vec`` (MMVQ, decode) -- plus a per-token
#: Python fallback loop that is unusable at scale. MMQ is a strict subset of
#: MMVQ, so MMVQ membership is the coverage test for both. Types outside it
#: are NOT covered: the offload would tier bytes that nothing can then read
#: at expert granularity. The #268 guard refuses those layers -- see
#: ``gguf_moe_offload_covered_type``.
MOE_OFFLOAD_SUPPORTED_TYPES = MMVQ_QUANT_TYPES


def gguf_moe_offload_covered_type(qweight_type) -> bool:
    """True when a ggml type has a GGUF MoE kernel the offload can feed.

    Takes the raw int the loader parked on ``w13_qweight_type.weight_type``
    (an unknown/unnamed type id must answer False, not raise).
    """
    try:
        wt = WeightType(int(qweight_type))
    except (ValueError, TypeError):
        return False
    return wt in MOE_OFFLOAD_SUPPORTED_TYPES


# MMQ (quantized GEMM) is weight-bandwidth-bound and does NOT scale with batch,
# so it only wins for tiny token counts; above this threshold a one-shot
# dequantize + cuBLAS fp16 GEMM is far faster (measured on RTX 3080 / Q6_K:
# MMQ wins <=~8 tokens, dequant+cuBLAS up to ~13x faster at 2048 tokens). The
# stock path used MMQ for K-quants at ALL batch sizes, crippling prefill.
# Env-tunable; MMVQ still owns the batch<=mmvq_safe decode path below.
_MMQ_MAX_TOKENS = int(os.environ.get("SGLANG_GGUF_MMQ_MAX_TOKENS", "8"))

# Per-device MMVQ->MMQ crossover (see the long note in fused_mul_mat_gguf). The
# crossover M scales with memory bandwidth; we proxy that by compute capability
# (major >= 9 == Hopper/Blackwell high-bandwidth parts). Override per rank with
# SGLANG_GGUF_MMVQ_SAFE. Cached by device index (each TP rank is its own process).
_mmvq_safe_by_dev: dict = {}

# Batched-MMVQ (ncols_dst<=8, llama.cpp-style column amortization in
# sgl_kernel's mul_mat_vec_q): with the rebuilt kernel one weight stream
# serves up to 8 activation columns, so MMVQ dominates MMQ for the whole
# spec-decode verify range (M<=8) on EVERY device class.
#
# Default: ON iff the installed sgl_kernel is the batched build, detected
# via the `ggml_dequantize(..., out=)` schema, which ships in the SAME
# build as the ncols<=8 MMVQ dispatcher. Old wheels (per-column, linear-in-M
# MMVQ) keep the measured per-device crossover -- batched dispatch there
# would be a regression. Env override: SGLANG_GGUF_BATCHED_MMVQ=0|1;
# SGLANG_GGUF_MMVQ_SAFE still wins over both.
#
# Numerics note: enabling this moves M=3..8 shapes from MMQ onto MMVQ on
# consumer devices. Both kernels sit ~5e-3 (relmax) from an fp32 reference
# but differ from EACH OTHER by ~1e-4 (different reduction order), so
# greedy outputs are NOT bit-identical to the old dispatch at near-ties --
# equally valid rounding, verified quality/accept-neutral (T66 stage 1c).
_BATCHED_MMVQ_ENV = os.environ.get("SGLANG_GGUF_BATCHED_MMVQ")


def _batched_mmvq_enabled() -> bool:
    if _BATCHED_MMVQ_ENV is not None:
        return _BATCHED_MMVQ_ENV == "1"
    return _dequant_supports_out()


# Task #73: probe for the tuned K-quant batched MMVQ kernel. Returns True only
# on a wheel that carries the ggml_mmvq_kq_tuned op (new build) AND with the
# runtime kill switch SGLANG_GGUF_KQ_KERNEL != 0 (the C++ op consults it). Old
# wheels lack the op -> False -> the legacy #72 crossover is kept unchanged.
_kq_tuned_cached: bool | None = None


def _kq_kernel_tuned() -> bool:
    global _kq_tuned_cached
    if _kq_tuned_cached is None:
        # Detect via op EXISTENCE (hasattr), not by calling it: the probe takes
        # no tensor args, so torch's dispatcher has no key to route .default()
        # and raises NotImplementedError. Its mere registration already marks a
        # tuned-kernel wheel (old wheels lack the op). The runtime kill switch
        # SGLANG_GGUF_KQ_KERNEL is then mirrored here exactly as the C++ side
        # reads it (unset or non-'0' == enabled), so disabling the kernel also
        # restores the legacy #72 MMVQ->MMQ crossover.
        try:
            has_op = hasattr(torch.ops.sgl_kernel, "ggml_mmvq_kq_tuned")
        except Exception:
            has_op = False
        env = os.environ.get("SGLANG_GGUF_KQ_KERNEL", "1")
        _kq_tuned_cached = has_op and (len(env) == 0 or env[0] != "0")
    return _kq_tuned_cached


# ---------------------------------------------------------------------------
# Shape-aware K-quant dispatch on consumer (sm < 9) devices (task #72, from
# the t69/t72 microbenches on RTX 3080 at the REAL Qwen3.6-27B TP=2 shard
# shapes, kernel_bench_3080.txt + t72_decision_bench.py):
#
# The batched-MMVQ ncols<=8 kernel amortizes the K-quant superblock decode
# well only up to M~4. Beyond that, MMQ overtakes -- earliest on "wide-K"
# shapes (row-parallel down_proj class: few output rows, long dot products,
# so MMVQ's per-column K-loop cost dominates while MMQ's flat GEMM floor
# does not move):
#   Q6_K down TP2-shard (N=5120, K=8704):  M=4 MMVQ 0.153 vs MMQ 0.163 ms;
#     M=5 0.219 vs 0.179 (MMQ 1.22x); M=6 0.297 vs 0.180; M=8 0.408 vs 0.180.
#   Q6_K down FULL (N=5120, K=17408):      MMQ wins from M=4 (0.318 vs 0.383).
#   At M=8 MMQ wins on EVERY measured K-quant shard shape, including
#     narrow-K: gate TP2-shard 0.185 vs 0.225, down K=5120 0.111 vs 0.137.
# Q8_0 shapes are NOT affected (their ncols kernel holds near-peak BW through
# M=6, e.g. lm_head M=8: MMVQ 2.09 vs MMQ 2.23 ms) and the 5090 (sm120) is
# NOT affected either (MMVQ wins everywhere there, e.g. down FULL M=4:
# 0.072 vs 0.137 ms), hence the K-quant-only + sm<9 gating.
#
# Dispatch-only change: MMVQ and MMQ are both ~5e-3 (relmax) from an fp32
# reference but differ from each other ~1e-4 (reduction order), so outputs
# are NOT bit-identical at near-ties -- same class as the batched-MMVQ
# default (see note above), quality proven via logprob-ID identity instead.
#
# SGLANG_GGUF_MMVQ_SAFE (explicit uniform cap) disables this routing;
# SGLANG_GGUF_SHAPE_DISPATCH=0 is the dedicated kill switch.
_WIDEK_MMQ_MIN_M = 5  # wide-K shapes: MMQ from M>=5 (measured crossover)
_WIDEK_MIN_K = 8192  # "wide-K" = K >= 8192 elems (down shards: 8704/17408)
_KQUANT_MMQ_ALL_MIN_M = 8  # any K-quant shape: MMQ at M>=8 (measured)
_SHAPE_DISPATCH_ENV = os.environ.get("SGLANG_GGUF_SHAPE_DISPATCH")
_sm_major_by_dev: dict = {}


def _device_sm_major() -> int:
    dev = torch.cuda.current_device()
    v = _sm_major_by_dev.get(dev)
    if v is None:
        try:
            v = torch.cuda.get_device_capability(dev)[0]
        except Exception:
            v = 0
        _sm_major_by_dev[dev] = v
    return v


def _kquant_prefers_mmq(m: int, qweight: torch.Tensor, qweight_type: int) -> bool:
    """Shape-aware MMVQ->MMQ rerouting for K-quants on sm<9 (see note above).

    Only consulted for M <= mmvq_safe (larger M never reaches the MMVQ
    branch), only in batched-MMVQ mode, and never when the user pinned an
    explicit uniform cap via SGLANG_GGUF_MMVQ_SAFE.
    """
    if qweight_type not in KQUANT_TYPES:
        return False
    if _SHAPE_DISPATCH_ENV == "0" or not _is_cuda:
        return False
    if os.environ.get("SGLANG_GGUF_MMVQ_SAFE") is not None:
        return False
    if not _batched_mmvq_enabled():
        return False
    if _device_sm_major() >= 9:
        return False
    # Task #73: with the tuned K-quant MMVQ kernel active, MMVQ beats MMQ at
    # every measured M<=8 on sm86 (RTX 3080, t73_kq_bench: Q6_K gate TP2-shard
    # M=8 = 367 vs MMQ 172 GB/s; Q6_K down TP2 M=8 = 371 vs 116; Q4_K down TP2
    # M=8 = 223 vs 138). The #72 MMVQ->MMQ reroute was calibrated on the SLOW
    # legacy kernel and is now a pessimization, so keep everything on MMVQ.
    if _kq_kernel_tuned():
        return False
    if m >= _KQUANT_MMQ_ALL_MIN_M:
        return True
    if m >= _WIDEK_MMQ_MIN_M:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        k = qweight.shape[1] // type_size * block_size
        return k >= _WIDEK_MIN_K
    return False


def _mmvq_safe_for_device() -> int:
    override = os.environ.get("SGLANG_GGUF_MMVQ_SAFE")
    if override is not None:
        return int(override)
    if _batched_mmvq_enabled():
        return 8
    dev = torch.cuda.current_device()
    v = _mmvq_safe_by_dev.get(dev)
    if v is None:
        try:
            major = torch.cuda.get_device_capability(dev)[0]
        except Exception:
            major = 0
        # High-bandwidth (Hopper sm90, Blackwell sm100/sm120): MMVQ's linear cost
        # stays under MMQ's flat floor through the small MTP verify batch (M<=6).
        # Consumer Ampere/Ada (RTX 3080/4090): MMQ overtakes at M=3.
        v = 6 if major >= 9 else 2
        _mmvq_safe_by_dev[dev] = v
    return v


# ---------------------------------------------------------------------------
# Task #163: bucket-coupled MMVQ->MMQ decode threshold. OPT-IN, default OFF.
#
# THE PROBLEM. `_batched_mmvq_enabled()` pins `mmvq_safe = 8`, so `M <= 8`
# always takes the matrix-VECTOR kernel, while `_MMQ_MAX_TOKENS` is also 8 --
# which makes the MMQ branch below unreachable for every MMQ-capable type. The
# tiled kernel `ggml_mul_mat_a8` exists and Q8_0 is in MMQ_QUANT_TYPES, but
# nothing ever dispatches to it. Measured on the real per-rank decode shapes
# (task163_xover_probe.py, Q8_0, correctness-gated against MMVQ at the
# pre-registered bar 4*ULP_bf16(max|ref|) and rel_Fro <= 1e-3; median of 3):
#
#   RTX 5090 (sm120)         M=4    M=7    M=8    (us/call, MMVQ -> MMQ)
#     qkv/o_proj  5120x5120  15.4   25.4   46.3 -> 26.4   MMQ 1.75x
#     gate_up     5120x2688   9.4   14.8   25.4 -> 15.0   MMQ 1.70x
#     down        2688x5120   8.8   13.4   26.6 -> 22.7   MMQ 1.17x
#   RTX 3080 (sm86)
#     qkv/o_proj  5120x5120  51.0   81.1   88.0 -> 89.2   MMQ 0.99x (LOSES)
#     gate_up     5120x2688  29.8   36.8   39.4 -> 50.5   MMQ 0.78x (LOSES)
#     down        2688x5120  30.7   46.0   47.0 -> 53.0   MMQ 0.89x (LOSES)
#
# So the crossover is DEVICE-dependent, not shape-dependent: on sm120 MMVQ
# falls off a cliff at exactly M=8 and MMQ wins on every shape class; on sm86
# MMVQ still wins at M=8 and its cliff sits above the MMQ band (M=12: MMVQ
# 470us vs MMQ 139us, but 12 > _MMQ_MAX_TOKENS, so that range belongs to the
# dequant branch and is out of scope here). The table below is therefore keyed
# by (device capability, shape class) and carries only MEASURED entries; an
# unmeasured device gets no threshold at all rather than an extrapolation.
#
# ---- WHY OPT-IN, AND WHAT YOU TRADE ---------------------------------------
# The threshold is coupled to the CUDA-graph decode buckets, not to the raw
# token count. That is necessary: a captured graph replays at its bucket size,
# so the kernel choice must be constant across the whole bucket or a replay
# would run a kernel other than the one it was captured with.
#
# What the coupling BUYS beyond replay consistency: a batch of 5 real tokens
# padded into captured bucket 8 and the SAME batch run eagerly (graphs off, or
# a batch too large to be captured) pick the same kernel. What it COSTS: an
# eager forward at a token count strictly between two buckets is rounded UP,
# so a genuine 5-token extend on sm120 runs MMQ (26.1us) where MMVQ would have
# been faster (19.2us) -- about 1.36x on that call. With the usual ladder
# (1,2,4,8) the exposure is exactly M=5..7 on non-replayed forwards, which is
# why the coupling is worth its price here; it is a real cost, not zero.
#
# It is NOT sufficient for run-to-run reproducibility. WHICH bucket a request
# lands in depends on the other traffic in the batch, so the same request can
# take MMVQ in one run and MMQ in another. MMVQ and MMQ are both `*_a8` (both
# quantise activations to int8) and agree to well inside the bf16 output
# resolution, but they are not bit-identical -- different reduction order. So
# enabling this introduces a NEW run-to-run variation that does not exist
# today, and that is exactly why it is opt-in with the default OFF: with the
# flag off, `_mmq_threshold_prefers_mmq` returns False before touching
# anything and the dispatch is byte-identical to the previous behaviour.
#
# `dequant_cublas` is NEVER a target of this threshold at any tolerance: it
# does not quantise the activations, so it sits ~6e-3 away in a different
# PRECISION CLASS (TOLERANCE_task163.md). Rerouting into it would silently
# change the precision class as a side effect of a speed optimisation. The
# `bucket > _MMQ_MAX_TOKENS` guard below keeps this path strictly inside the
# MMVQ<->MMQ pair.
# ---------------------------------------------------------------------------
_MMQ_THRESHOLD_ENV = "SGLANG_GGUF_MMQ_DECODE_THRESHOLD"

# (major, minor) -> {shape class -> smallest decode BUCKET that takes MMQ}.
# `None` means "measured on this device, MMQ never wins inside the MMQ band".
# A device absent from this table is unmeasured and gets no rerouting.
_MMQ_BUCKET_MIN: dict = {
    (12, 0): {"square": 8, "tall": 8, "wide_k": 8},  # RTX 5090, measured
    (8, 6): {"square": None, "tall": None, "wide_k": None},  # RTX 3080, measured
}

# Shape classes over K/N (N = output rows, K = dot-product length). The three
# classes separate the real decode shapes cleanly (down 1.91, qkv/o 1.00,
# gate_up 0.53). NOTE, so nobody reads more into this axis than it carries:
# on BOTH measured devices the crossover M came out identical across the three
# classes -- what differs by class is the SIZE of the win (1.17x on wide_k vs
# 1.75x on square), not where it starts. The axis is kept because it is the
# place a class can be excluded when a future device or shape flips one of
# them, and because #72's dispatch already reasons per shape class.
_WIDE_K_RATIO = 1.25
_TALL_RATIO = 0.8

_dev_cap_by_dev: dict = {}
_mmq_threshold_cached: Optional[bool] = None
_mmq_threshold_logged: bool = False
#: #894 S5: one line per process when the env override -- not the CLI flag --
#: decided the threshold. Separate from `_mmq_threshold_logged`, which fires on
#: the first actual reroute and therefore never fires for the configuration
#: this warning is about.
#: #901: the hand-rolled module-level bool and its reset are the authority's
#: ``Announcer`` now -- the same latch three of the four migrated sites had
#: written out by hand, with the reset contract #894's own suite pins.
_mmq_announcer = _knob.Announcer("gguf.mmq_decode_threshold")

# Decode token-count buckets, registered by the decode CUDA-graph runner(s):
# bs * num_tokens_per_bs for every captured bs. Target and draft runners both
# register; the union is used, which keeps the invariant that matters -- every
# token count a graph actually replays at IS itself a bucket, so
# `_decode_bucket_for` is the identity during a replay and the replay runs the
# kernel it was captured with.
_decode_token_buckets: tuple = ()


def set_decode_token_buckets(token_counts) -> None:
    """Register captured decode token counts (idempotent, union-merging)."""
    global _decode_token_buckets
    merged = set(_decode_token_buckets)
    merged.update(int(t) for t in token_counts if int(t) > 0)
    _decode_token_buckets = tuple(sorted(merged))


def _decode_bucket_for(m: int) -> int:
    """Smallest registered decode bucket >= m (m itself if none applies).

    With no buckets registered (CUDA graphs disabled) this is the identity, so
    eager serving simply thresholds on the raw token count.
    """
    buckets = _decode_token_buckets
    if not buckets or m > buckets[-1]:
        return m
    return buckets[bisect.bisect_left(buckets, m)]


def _reset_mmq_threshold_cache() -> None:
    """Test hook: drop the cached enable decision (and registered buckets)."""
    global _mmq_threshold_cached, _decode_token_buckets, _mmq_threshold_logged
    _mmq_threshold_cached = None
    _decode_token_buckets = ()
    _mmq_threshold_logged = False
    # #894: the override announcement is latched to the same lifetime as the
    # decision it describes. A hook that cleared one but not the other would
    # leave the second decision silent again.
    _mmq_announcer.reset()


def _announce_mmq_env_override(env: str, enabled: bool) -> None:
    """Say that the env var, not the CLI flag, decided this (#894 S5).

    THE DEFECT THIS EXISTS TO END. ``_mmq_decode_threshold_enabled``
    short-circuits on the PRESENCE of ``SGLANG_GGUF_MMQ_DECODE_THRESHOLD``, so
    a stale ``=0`` from an old A/B run beat ``--gguf-mmq-decode-threshold``
    without consulting it. The only log on the whole path fires when a reroute
    HAPPENS, so the losing configuration produced zero reroutes and therefore
    zero output: flag on, nothing rerouted, nothing said. That is #889's shape
    in a different module.

    WARNING, NOT REFUSAL, decided on the danger direction:

    * This is the GGUF matmul dispatch, reached on the first quantized matmul
      of a forward pass -- not a parse-time gate. Raising here does not refuse a
      configuration, it kills a model mid-forward, on precisely the processes
      that carry a stale env var.
    * The defect's blast radius is a wrong belief about which kernel ran (and a
      measurement filed under the wrong arm). A refusal's is the instance.
    * Nor may the precedence be flipped to "flag wins". The env override is
      documented in the flag's own help text ("wins over this flag") and exists
      so an A/B run can change the kernel choice without re-parsing ServerArgs.
      MMVQ and MMQ are not bit-identical, so quietly moving that selection as a
      side effect of a logging fix would be a bigger change than the one being
      fixed. State the truth; do not change it.

    Both spellings of the failure are covered: a value that DISAGREES with the
    flag, and a value that is not ``0`` or ``1`` at all (``env == "1"`` reads
    ``"true"`` as OFF, which was the second silence in the same three lines).

    #901: the ladder, the equivalence test, the latch and the printed skeleton
    all come from ``srt/knob_resolution``. Two things about this site are NOT
    generic and stay here: the presence rule is ``is not None`` rather than
    "non-empty" (``_mmq_decode_threshold_enabled`` short-circuits on presence
    and then tests ``== "1"``, so ``FOO=`` IS a source here and reads as OFF),
    and the equivalence test normalises an env STRING against a flag BOOLEAN.
    Both are declared as parameters instead of being re-decided in prose.
    """
    if _mmq_announcer.said:
        return
    recognised = env in ("0", "1")
    flag = None
    try:
        from sglang.srt.runtime_context import get_server_args

        flag = bool(getattr(get_server_args(), "gguf_mmq_decode_threshold", False))
    except Exception:
        # Standalone kernel use, unit tests, or a matmul that beat
        # ModelRunner's publish. The comparison is impossible; the
        # supersession is not, and the callers here are the ones most likely
        # to be surprised by it.
        flag = None

    resolution = _knob.resolve_knob(
        [
            _knob.KnobSource(
                source=_knob.env_source(_MMQ_THRESHOLD_ENV),
                value=enabled,
                present=True,  # the caller only reaches here on presence
                kind=_knob.KIND_ENV,
                label=f"{_MMQ_THRESHOLD_ENV}={env!r}",
            ),
            _knob.KnobSource(
                source=_knob.flag_source("gguf_mmq_decode_threshold"),
                value=flag,
                present=flag is not None,
                kind=_knob.KIND_FLAG,
                label=f"--gguf-mmq-decode-threshold={flag}",
                cost="it was never consulted, so the flag is INERT",
            ),
        ],
        normalize=bool,
    )

    # An UNRECOGNISED value is a defect on its own, independent of who won:
    # 'true' and 'yes' are read as OFF in silence. So the line fires either
    # when something was superseded, or when the value was not 0|1 at all.
    if recognised and flag is not None and not resolution.lost_anything:
        return  # never ambiguous -- do not train readers to skip the line
    if flag is None:
        loss = _knob.unreadable_loss_clause(
            "--gguf-mmq-decode-threshold",
            "could not be read (ServerArgs not published at this call), so it "
            "was not compared",
        )
    else:
        loss = _knob.loss_clause(
            f"--gguf-mmq-decode-threshold={flag}", resolution.ladder[1].cost
        )
    note = (
        ""
        if recognised
        else (
            " The value is not one of 0|1 and is therefore read as OFF; only "
            "the exact string '1' enables the reroute."
        )
    )
    _mmq_announcer.say(
        logger,
        _knob.supersession_line(
            "894",
            winner=f"{_MMQ_THRESHOLD_ENV}={env!r}",
            subject="the GGUF MMQ decode threshold",
            effective="this boot gets %s" % ("ON" if enabled else "OFF"),
            loss=loss,
            note=note,
            presence_rule=(
                "The env override wins on PRESENCE, not on value, so a stale "
                "=0 from an earlier A/B run silences the flag and produces no "
                "reroute and no other log line at all."
            ),
            # #901 CONTRACT CHANGE, deliberate: the old remedy said "unset
            # SGLANG_GGUF_MMQ_DECODE_THRESHOLD", which does not close this
            # site's own trap -- the presence rule here is `is not None`, so
            # `export SGLANG_GGUF_MMQ_DECODE_THRESHOLD=` leaves the override
            # PRESENT and still reading as OFF. The shared remedy says REMOVE
            # and names the empty-string trap.
            remedy=_knob.removal_remedy(_MMQ_THRESHOLD_ENV),
        ),
    )


def _mmq_decode_threshold_enabled() -> bool:
    """Opt-in gate: --gguf-mmq-decode-threshold, or the env override.

    #901: the PRESENCE RULE is asked of the authority rather than spelled
    ``os.environ.get(...) is not None`` here. It is the dangerous one of the
    two -- ``FOO=`` is a source and reads as OFF -- and having it written out
    inline is how it came to differ from the rule two modules over without
    anyone comparing them. The precedence itself does not move: the env still
    wins on presence, which is the documented, restart-free A/B override.
    """
    global _mmq_threshold_cached
    if _mmq_threshold_cached is not None:
        return _mmq_threshold_cached
    if _knob.env_present(_MMQ_THRESHOLD_ENV):
        env = _knob.env_value(_MMQ_THRESHOLD_ENV)
        enabled = env == "1"
        # #894: the env still wins -- it just no longer wins in silence.
        _announce_mmq_env_override(env, enabled)
        _mmq_threshold_cached = enabled
        return _mmq_threshold_cached
    try:
        from sglang.srt.runtime_context import get_server_args

        _mmq_threshold_cached = bool(
            getattr(get_server_args(), "gguf_mmq_decode_threshold", False)
        )
        return _mmq_threshold_cached
    except Exception:
        # No published ServerArgs yet (unit tests, standalone kernel use, or a
        # call that beat ModelRunner's publish). Answer False but do NOT latch
        # it -- latching here would permanently disable the flag for anyone
        # whose first GGUF matmul ran before the publish.
        return False


def _device_cap() -> tuple:
    try:
        dev = torch.cuda.current_device()
    except Exception:
        return (0, 0)
    v = _dev_cap_by_dev.get(dev)
    if v is None:
        try:
            v = tuple(torch.cuda.get_device_capability(dev))
        except Exception:
            v = (0, 0)
        _dev_cap_by_dev[dev] = v
    return v


def _gguf_shape_class(qweight: torch.Tensor, qweight_type: int) -> str:
    """Classify a GGUF weight shard by K/N (see _MMQ_BUCKET_MIN)."""
    n = qweight.shape[0]
    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    k = qweight.shape[1] // type_size * block_size
    ratio = k / max(n, 1)
    if ratio > _WIDE_K_RATIO:
        return "wide_k"
    if ratio < _TALL_RATIO:
        return "tall"
    return "square"


def _mmq_threshold_prefers_mmq(m: int, qweight: torch.Tensor, qweight_type: int) -> bool:
    """Task #163 opt-in reroute of a decode BUCKET from MMVQ to MMQ.

    Returns False before touching anything when the flag is off, which is what
    makes the default path byte-identical.
    """
    if not _mmq_decode_threshold_enabled():
        return False
    if not _is_cuda or qweight_type not in MMQ_QUANT_TYPES:
        return False
    bucket = _decode_bucket_for(m)
    # Stay strictly inside the MMVQ<->MMQ pair. Above _MMQ_MAX_TOKENS the
    # dequant + cuBLAS branch owns the range, and it is a different PRECISION
    # class (no activation quantisation) -- moving that boundary must be a
    # declared decision, never a side effect of a speed threshold.
    if bucket > _MMQ_MAX_TOKENS:
        return False
    cap = _device_cap()
    table = _MMQ_BUCKET_MIN.get(cap)
    if not table:
        return False
    shape_class = _gguf_shape_class(qweight, qweight_type)
    min_bucket = table.get(shape_class)
    if min_bucket is None or bucket < min_bucket:
        return False
    global _mmq_threshold_logged
    if not _mmq_threshold_logged:
        # One-shot, so a run can be PROVEN to have taken the new path instead
        # of it being inferred from the throughput number.
        _mmq_threshold_logged = True
        logger.info(
            "GGUF MMQ decode threshold ACTIVE (#163): device sm%d%d, first "
            "reroute at bucket %d (raw M=%d, shape class %s, min %d), decode "
            "buckets %s. MMVQ->MMQ is not bit-identical run-to-run; see "
            "--gguf-mmq-decode-threshold.",
            cap[0],
            cap[1],
            bucket,
            m,
            shape_class,
            min_bucket,
            _decode_token_buckets or "(none: eager)",
        )
    return True


# ---------------------------------------------------------------------------
# #63: persistent dequant workspace.
#
# The large-batch (prefill) path below dequantizes each weight shard into a
# fresh `torch.empty` every forward (e.g. 86 MiB for one ffn shard/rank of
# Qwen3.6-27B at TP=2). Under a tightly sized KV pool that per-forward
# allocation is exactly what OOMed first (M27c: server healthy, then
# "Tried to allocate 86.00 MiB" inside ggml_dequantize on the first real
# forward). Route it through one persistent, grow-only per-(device,dtype)
# buffer instead: it is allocated during weight finalization, BEFORE the KV
# budget is profiled, so the scratch is accounted for up front and the
# runtime path performs zero allocations. Dequants larger than the cap
# (default 512 MiB, e.g. a full lm_head logprob dequant) intentionally fall
# back to a fresh allocation instead of pinning that much VRAM forever.
#
# Requires the sgl_kernel `ggml_dequantize(..., out=)` schema (rebuilt
# in-tree kernel). With an older installed wheel this degrades byte-
# identically to the previous fresh-allocation behavior.
# ---------------------------------------------------------------------------
_DEQUANT_WS: dict = {}  # (lane_id, device_index, dtype) -> 1-D buffer
# (lane_id, device_index, dtype) -> largest dequant TARGET this rank can be
# asked for, in bytes, INCLUDING the over-cap ones the workspace deliberately
# refuses to hold. See gguf_dequant_scratch_residual_bytes.
_DEQUANT_PEAK_TARGET: dict = {}
_dequant_out_supported: bool | None = None


def _dequant_ws_cap_bytes() -> int:
    return int(os.environ.get("SGLANG_GGUF_DEQUANT_WS_CAP_MIB", "512")) * (1 << 20)


def _dequant_ws_key(dtype: torch.dtype, device) -> tuple:
    """Workspace key: LANE, device, dtype (#274 slice C).

    The workspace is a single reused buffer, and its safety argument is
    sequential: "within one forward the dequant -> GEMM pairs run back-to-back
    on the same CUDA stream, so each GEMM consumes the dequant before the next
    one overwrites it". Both premises break the moment two lanes forward at
    the same time on one device: two streams, interleaved pairs, one buffer --
    lane B's dequant overwrites the weight lane A's GEMM is still reading.
    That is silent numerical corruption, not a crash.

    Keying by lane restores the premise per lane (each lane keeps its own
    strictly sequential dequant->GEMM chain on its own stream). The falsifier
    for the shared form is in test_dual_group_concurrency.py.
    """
    from sglang.srt.runtime_context import current_lane_id

    return (
        current_lane_id(),
        device.index if device.type == "cuda" else device,
        dtype,
    )


def gguf_dequant_scratch_residual_bytes(device_index: int | None = None) -> int:
    """GGUF dequant scratch that is still UNALLOCATED at KV-profiling time.

    #257. The workspace below is sized at weight-load time, i.e. before the
    KV budget is profiled, so whatever it already holds is charged against
    the budget automatically (it shows up as used memory). What does NOT
    show up is the part the workspace refuses: a dequant target larger than
    ``SGLANG_GGUF_DEQUANT_WS_CAP_MIB`` falls back to a FRESH allocation of
    its full size on the eager path -- and the pre-capture warmup forwards
    are eager. For Qwen3.6-27B Q3_K_M at TP=1 the lm_head shard is 2.37 GiB
    (1.19 GiB at TP=2, #70), which is exactly the allocation that OOMed the
    decode-graph warmup at --mem-fraction-static 0.90 while the KV pool had
    already been sized as if that memory were free.

    This is not a safety margin: it is the arithmetic difference between the
    largest dequant this rank will perform and the buffer it already owns.
    Returns 0 for every non-GGUF model (nothing ever records a peak), which
    keeps the default budget path byte-identical.

    MAX, not sum, across dtypes: over-cap dequants are transient (dequant ->
    GEMM -> free, one weight at a time on one stream), so the simultaneous
    demand is the largest single one, not their total.
    """
    residual = 0
    for key, peak in _DEQUANT_PEAK_TARGET.items():
        _lane, idx, _dtype = key
        if device_index is not None and idx != device_index:
            continue
        buf = _DEQUANT_WS.get(key)
        have = buf.numel() * buf.element_size() if buf is not None else 0
        residual = max(residual, peak - have)
    return max(residual, 0)


def _dequant_supports_out() -> bool:
    global _dequant_out_supported
    if _dequant_out_supported is None:
        try:
            schema = str(torch.ops.sgl_kernel.ggml_dequantize.default._schema)
            _dequant_out_supported = "out" in schema
        except Exception:
            _dequant_out_supported = False
    return _dequant_out_supported


def _reserve_dequant_workspace(numel: int, dtype: torch.dtype, device) -> None:
    """Grow the persistent workspace (load time, pre-KV-profiling)."""
    key = _dequant_ws_key(dtype, device)
    nbytes = numel * dtype.itemsize
    # Record the peak target BEFORE either early return (#257): both of them
    # mean "this dequant fresh-allocates its full size at forward time",
    # which is precisely the demand the KV budget has to know about.
    if nbytes > _DEQUANT_PEAK_TARGET.get(key, 0):
        _DEQUANT_PEAK_TARGET[key] = nbytes
    if not _dequant_supports_out():
        # Old wheel: the runtime path cannot write into the workspace, so
        # holding one would only STEAL headroom. Keep legacy behavior.
        return
    if nbytes > _dequant_ws_cap_bytes():
        return
    buf = _DEQUANT_WS.get(key)
    if buf is None or buf.numel() < numel:
        _DEQUANT_WS[key] = torch.empty(numel, dtype=dtype, device=device)


def _ggml_dequantize_ws(
    qweight: torch.Tensor, qweight_type: int, rows: int, cols: int, dtype: torch.dtype
) -> torch.Tensor:
    """ggml_dequantize into the persistent workspace when possible.

    Sequential-stream safety: within one forward the dequant -> GEMM pairs run
    back-to-back on the same CUDA stream, so a single reused buffer is safe
    (each GEMM consumes the dequant before the next dequant overwrites it).
    A static buffer is also strictly friendlier to CUDA-graph capture than a
    per-replay allocation.
    """
    numel = rows * cols
    if _dequant_supports_out() and numel * dtype.itemsize <= _dequant_ws_cap_bytes():
        key = _dequant_ws_key(dtype, qweight.device)
        buf = _DEQUANT_WS.get(key)
        if buf is None or buf.numel() < numel:
            buf = torch.empty(numel, dtype=dtype, device=qweight.device)
            _DEQUANT_WS[key] = buf
        out = buf.narrow(0, 0, numel).view(rows, cols)
        return torch.ops.sgl_kernel.ggml_dequantize.default(
            qweight, qweight_type, rows, cols, dtype, out
        )
    return ggml_dequantize(qweight, qweight_type, rows, cols, dtype)


def _mul_mat_dequant_chunked(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int, rows: int, cols: int
) -> torch.Tensor:
    """#70: capture-safe over-cap dequant matmul (x @ dequant(qweight).T).

    Row-chunks the dequant target through the persistent #63 workspace so a
    CUDA-graph capture never materializes the full dequantized weight (which
    for the Qwen3.6-27B lm_head shard is 1.19 GiB > the 512 MiB workspace
    cap, and as a fresh allocation is retained per-graph in the capture's
    private pool -> capture OOM without --max-running-requests caps).

    Numerics: splitting along OUTPUT rows leaves every output element's
    K-dot-product intact -- each chunk GEMM computes exactly the elements it
    owns. (cuBLAS kernel selection may differ from the single big GEMM the
    eager fallback runs, so this is dispatch-level equivalent, not
    bit-identical to the eager path; the eager path itself is unchanged.)

    Stream safety mirrors _ggml_dequantize_ws: dequant->GEMM pairs run
    back-to-back on one stream, so chunk i's GEMM consumes the workspace
    before chunk i+1's dequant overwrites it. The per-chunk GEMM outputs are
    small (M x rows total) and land in the graph pool during capture.
    """
    key = _dequant_ws_key(x.dtype, qweight.device)
    buf = _DEQUANT_WS.get(key)
    if buf is None or buf.numel() < cols:
        # No (or too small a) workspace exists on this rank. Allocating up
        # to the cap here is bounded and happens at most once; during
        # capture it lands in the graph pool, which is still strictly
        # smaller than the full-dequant fresh allocation this path replaces.
        numel = max(_dequant_ws_cap_bytes() // x.dtype.itemsize, cols)
        buf = torch.empty(numel, dtype=x.dtype, device=qweight.device)
        _DEQUANT_WS[key] = buf
    rows_per_chunk = max(buf.numel() // cols, 1)
    outs = []
    for r0 in range(0, rows, rows_per_chunk):
        rn = min(rows_per_chunk, rows - r0)
        out = buf.narrow(0, 0, rn * cols).view(rn, cols)
        wchunk = torch.ops.sgl_kernel.ggml_dequantize.default(
            qweight.narrow(0, r0, rn), qweight_type, rn, cols, x.dtype, out
        )
        outs.append(x @ wchunk.T)
    return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]


def fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    if qweight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
    else:
        # MMVQ (ggml_mul_mat_vec_a8) is a mat-VEC kernel whose cost scales ~LINEARLY
        # with the token count M (it iterates over the M activation columns),
        # whereas MMQ (ggml_mul_mat_a8) is a weight-bandwidth-bound GEMM whose cost
        # is ~flat up to ~M=16. Their crossover M is therefore GPU-dependent: it
        # sits where MMVQ's rising line meets MMQ's flat floor, which moves out on
        # higher-memory-bandwidth cards. Measured on this system / Q6_K:
        #   RTX 3080 (sm86, ~760 GB/s):  MMVQ wins M<=2, MMQ faster from M=3
        #   RTX 5090 (sm120, ~1.8 TB/s): MMVQ wins through M>=6 (M=4: MMVQ 0.033ms
        #                                vs MMQ 0.049ms)
        # This matters for the MTP / speculative-decode VERIFY forward, which runs
        # at M = num_draft_tokens (typically 4): a single wrong global cap either
        # sends the 5090's verify onto the slower MMQ (cap=2) or the 3080's verify
        # onto the slower MMVQ (cap=6). Pick per physical GPU (each TP rank is a
        # separate process bound to one device) once, cached by device index.
        mmvq_safe = _mmvq_safe_for_device()
    # HACK: when doing chunked prefill we don't generate output tokens
    # so input to logits generator is empty which causes invalid parameter
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    # there is no need to call any kernel for fp16/bf16
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    # enable MMVQ in contiguous batching with batch_size=1
    if (
        x.shape[0] <= mmvq_safe
        and qweight_type in MMVQ_QUANT_TYPES
        # #72 shape-aware dispatch: reroute wide-K / M>=8 K-quant shapes to
        # MMQ on sm<9 devices (see _kquant_prefers_mmq).
        and not _kquant_prefers_mmq(x.shape[0], qweight, qweight_type)
        # #163 bucket-coupled MMQ threshold: OPT-IN (--gguf-mmq-decode-
        # threshold), default OFF -> returns False without side effects, so
        # the dispatch below is byte-identical when the flag is not set.
        and not _mmq_threshold_prefers_mmq(x.shape[0], qweight, qweight_type)
    ):
        y = ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    # MMQ (quantized GEMM) only for small batches (standard + k-quants): it is
    # weight-bandwidth-bound and does not scale with token count, so above
    # _MMQ_MAX_TOKENS the dequant + cuBLAS branch below is far faster.
    elif qweight_type in MMQ_QUANT_TYPES and x.shape[0] <= _MMQ_MAX_TOKENS:
        y = ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    # Large batch (or a type without an MMQ kernel): dequantize once, then a
    # single fp16 cuBLAS GEMM. All MMQ types are also in DEQUANT_TYPES, so
    # large-batch K-quants land here.
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        # #70: an over-cap dequant during CUDA-graph capture (draft-extend /
        # verify at bs>=4 hits the lm_head shard at M=9..16: 1.19 GiB for
        # Qwen3.6-27B) must not fresh-allocate -- the allocation is retained
        # in the graph's private pool and OOMs the capture under a tightly
        # sized KV budget (t69/boot4_oom.log). Chunk it through the
        # persistent workspace instead; outside capture the fresh-allocation
        # fallback stays byte-identical (e.g. large logprob prefills).
        if (
            _is_cuda
            and shape[0] * shape[1] * x.dtype.itemsize > _dequant_ws_cap_bytes()
            and _dequant_supports_out()
            and torch.cuda.is_current_stream_capturing()
        ):
            y = _mul_mat_dequant_chunked(x, qweight, qweight_type, *shape)
        else:
            # #63: dequantize into the persistent per-rank workspace (no
            # per-forward scratch allocation); see _ggml_dequantize_ws.
            weight = _ggml_dequantize_ws(qweight, qweight_type, *shape, x.dtype)
            y = x @ weight.T
    else:
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    return y


# ggml_moe_get_block_size(int type) -> int takes NO tensor argument. Until #518
# it was registered in sgl-kernel exclusively for the CUDA dispatch key, so
# torch had no tensor from which to infer a device and could not route the call
# -> it raised "no dispatchable fallback" NotImplementedError on EVERY call, on
# every arch. That aborted the MMQ MoE path (large-M / prefill / needle), while
# the tensor-carrying MMVQ decode path was unaffected. #518 registers it (and
# the two sibling probes) catch-all, so on a wheel built from this tree or later
# the real op answers. The kernel body is a pure compile-time lookup: it returns
# the MMQ tile width MOE_X_Q* for the ggml type, which is 8 on ROCm and 4 on
# CUDA/others for every supported quant type (see moe.cuh). The mirror below
# STAYS -- the wheel ships prebuilt and is pinned by sha256, so a tree carrying
# the #518 registration can still be running a pre-#518 wheel -- and the real op
# is preferred whenever it is dispatchable.
# Supported ggml types: Q4_0=2 Q4_1=3 Q5_0=6 Q5_1=7 Q8_0=8 Q2_K=10 Q3_K=11
# Q4_K=12 Q5_K=13 Q6_K=14, and MXFP4=39 on a #398 wheel (MOE_X_MXFP4, same 4/8
# tile width as the rest -- it is only in the set when the wheel has the
# kernel, otherwise the mirror would promise a tile the build cannot serve).
_GGML_MOE_MMQ_TYPES = frozenset(
    {2, 3, 6, 7, 8, 10, 11, 12, 13, 14} | ({39} if MXFP4_NATIVE else set())
)


def _ggml_moe_get_block_size(qtype: int) -> int:
    try:
        return ggml_moe_get_block_size(qtype)
    except NotImplementedError:
        if int(qtype) not in _GGML_MOE_MMQ_TYPES:
            return 0
        return 8 if _is_hip else 4


def fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
) -> torch.Tensor:
    def act(x: torch.Tensor):
        if activation == "silu":
            return silu_and_mul(x)
        elif activation == "gelu":
            return gelu_and_mul(x)
        raise ValueError(f"Unsupported activation: {activation}")

    out_hidden_states = torch.empty_like(x)
    # unless we decent expert reuse we are better off running moe_vec kernel
    if (
        qweight_type2 in MMQ_QUANT_TYPES
        and qweight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        BLOCK_SIZE = _ggml_moe_get_block_size(qweight_type)

        # Use the Python wrapper (allocates the alignment buffers and returns
        # the three tensors) rather than the raw sgl_kernel op, whose current
        # signature takes pre-allocated output buffers in-place. Imported
        # lazily to avoid a circular import at module load.
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size,
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, BLOCK_SIZE, E
        )
        # #109: expert_ids is allocated with torch.empty and the align kernel
        # only fills entries for blocks up to num_tokens_post_padded; the
        # TRAILING entries are uninitialized garbage. The MMQ device kernel
        # (sgl-kernel moe.cuh moe_q) guards them with a HARDCODED
        # `exp_idx > 255 || exp_idx < 0` check — exact for a full 256-row
        # expert table (garbage >= 256 is rejected), but under uneven-TP
        # expert-dim sharding the LOCAL table has only n_local+1 rows
        # (owned experts + the zero-pad expert), so garbage ids in
        # [E, 255] pass the guard and the kernel reads
        # `vx + exp_idx * exp_stride` BEYOND the weight tensor -> illegal
        # memory access (seen live: Qwen3.6-35B-A3B GGUF, TP=3, any
        # prefill > 64 tokens; the MMVQ decode path takes no expert_ids
        # blocks and is unaffected). Same latent OOB exists for any GGUF
        # MoE with < 256 experts even without sharding. Sanitize
        # out-of-range block ids to -1, which the kernel's `exp_idx < 0`
        # guard rejects; every VALID block id is < E by construction, so
        # only garbage/trailing blocks are touched. Python-side on purpose:
        # sgl_kernel ships prebuilt, and this mirrors the
        # _ggml_moe_get_block_size fallback above (usable without a kernel
        # rebuild); the durable fix belongs in moe.cuh (bound by nrows of
        # W_local, not the hardcoded 255) -> backlog #112.
        # masked_fill with a SCALAR is a device-only op: no host->device
        # copy, so it is CUDA-graph-capture safe. (A torch.where against a
        # freshly constructed host scalar -- e.g. expert_ids.new_tensor(-1)
        # -- is NOT: it aborts MTP/NEXTN graph capture with "Cannot copy
        # between CPU and CUDA tensors during CUDA graph capture".)
        expert_ids = expert_ids.masked_fill(expert_ids >= E, -1)
        out = ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        # TODO(FlamingoPg): maybe we can use moe_sum_reduce here?
        moe_sum(out, out_hidden_states)
    elif qweight_type2 in MMVQ_QUANT_TYPES and qweight_type in MMVQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]

        out = ggml_moe_a8_vec(x, w1, topk_ids, top_k, qweight_type, N, num_tokens)
        out = act(out)

        out = ggml_moe_a8_vec(
            out, w2, topk_ids, 1, qweight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        moe_sum(out, out_hidden_states)
    else:
        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                expert_up = w1[ii]

                out = fused_mul_mat_gguf(inp, expert_up, qweight_type)
                out = act(out)

                expert_down = w2[ii]
                current_state = fused_mul_mat_gguf(
                    out, expert_down, qweight_type2
                ).mul_(ww)
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def apply_gguf_embedding(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return torch.embedding(qweight, x)
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        x_flat = x.flatten()
        assert hidden_size == qweight.shape[1] // type_size * block_size
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        dequant = ggml_dequantize(
            quant, qweight_type, hidden_size, x_flat.shape[0], dtype
        )
        return dequant.view(*x.shape, hidden_size)
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        # For MergedColumnParallelLinear and QKVParallelLinear, we need to
        # materialize a single flat weight parameter (static storage for CUDA
        # Graph compatibility) plus zero-copy per-shard views.
        self._create_flat_weight_param(layer)
        # UD checkpoints ship some tensors dense (F16) inside otherwise
        # quantized families: normalize single dense qweights to params_dtype.
        self._cast_dense_qweight(layer)
        # #63: pre-size the persistent dequant workspace at LOAD time so the
        # KV-budget profiling that runs afterwards accounts for it.
        self._reserve_dequant_scratch(layer)

    def _create_flat_weight_param(self, layer: torch.nn.Module):
        """Materialize merged GGUF shards into ONE flat byte buffer + views.

        The previous layout concatenated the shards along dim0 and padded
        dim1 to the widest shard's row-byte width. With MIXED quant types in
        one merged module (e.g. unsloth UD's GDN in_proj_qkv=Q6_K next to
        in_proj_z=Q8_0: 4200 vs 5440 bytes/row) every narrower shard's slice
        `qweight[start:end, :size]` was non-contiguous, so apply() paid a
        `.contiguous()` COPY of the shard's full quantized bytes on EVERY
        forward -- including inside captured CUDA graphs (~0.96 GiB read +
        0.96 GiB write per rank per target forward for the 48 GDN layers of
        Qwen3.6-27B Q6 at TP=2). Storing the shards back-to-back in a flat
        1-D BYTE buffer makes each shard a contiguous, statically-addressed
        VIEW: zero per-forward copies, zero padding waste, unchanged
        numerics. Graph-safe by construction (views are created once at load
        time and alias the registered parameter's storage).

        Mixed DTYPES are supported too (the UD-Q8_K_XL "mixed-dtype qkvz"
        INFEASIBLE case, task #64): dense F16/BF16 shards (which the GGUF
        iterator renames to `.qweight` next to genuinely packed siblings,
        e.g. F16 in_proj_z beside Q8_0 in_proj_qkv, or F16 attn_q/k beside
        Q8_0 attn_v) are cast ONCE to params_dtype here, their shard type is
        updated accordingly, and each shard's view reinterprets the byte
        buffer in its own dtype (offsets 16-byte aligned).
        """
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            params_dtype = getattr(self, "params_dtype", torch.get_default_dtype())
            shard_types = layer.qweight_type.shard_weight_type
            tensors = []
            for idx in shard_id:
                src = data_container[shard_id_map[idx]]
                qtype = shard_types.get(idx)
                if (
                    qtype in UNQUANTIZED_TYPES
                    and src.dtype != params_dtype
                    and src.is_floating_point()
                ):
                    # Load-time normalization of dense shards: apply()'s
                    # unquantized matmul must be dtype-clean against the
                    # activations (mirrors transform_stream's `.to(dtype)`
                    # for dense modules).
                    src = src.to(params_dtype)
                    shard_types[idx] = int(_weight_type_for_dtype(params_dtype))
                tensors.append(src.contiguous())
            align = 16  # byte alignment so typed views stay legal
            offsets = []
            total = 0
            for t in tensors:
                total = -(-total // align) * align
                offsets.append(total)
                total += t.numel() * t.element_size()
            flat_data = torch.zeros(total, dtype=torch.uint8, device=qweight.device)
            # (flat_byte_offset, rows, dim1_in_shard_dtype) per shard id.
            shard_offset_map = dict[str, tuple[int, int, int]]()
            for idx, t, off in zip(shard_id, tensors, offsets):
                nbytes = t.numel() * t.element_size()
                flat_data.narrow(0, off, nbytes).copy_(
                    t.reshape(-1).view(torch.uint8)
                )
                shard_offset_map[idx] = (off, t.size(0), t.size(1))
            qweight.data_container.clear()
            flat_param = Parameter(flat_data, requires_grad=False)
            set_weight_attrs(flat_param, vars(qweight))
            set_weight_attrs(flat_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", flat_param)
            # Pre-built contiguous 2-D typed views into the flat parameter,
            # used by apply() instead of slice+contiguous. Plain attribute on
            # the layer (not a parameter): same storage, static pointers.
            layer._gguf_shard_views = {
                idx: flat_param.data.narrow(0, off, t.numel() * t.element_size())
                .view(t.dtype)
                .view(t.size(0), t.size(1))
                for (idx, t, off) in zip(shard_id, tensors, offsets)
            }

    def _cast_dense_qweight(self, layer: torch.nn.Module):
        """Single-tensor DENSE-typed `.qweight` (e.g. UD-Q8_K_XL ships 5
        full F16 MLPs, an F16 down_proj, F16 token_embd/output): cast once
        to params_dtype so the unquantized matmul/embedding path is
        dtype-clean, and record the matching weight type."""
        qweight = layer.qweight
        if getattr(layer, "_gguf_shard_views", None) or isinstance(
            qweight, UninitializedParameter
        ):
            return
        params_dtype = getattr(self, "params_dtype", None)
        if (
            params_dtype is None
            or not qweight.is_floating_point()
            or qweight.dtype == params_dtype
            or layer.qweight_type.weight_type not in UNQUANTIZED_TYPES
        ):
            return
        new_param = Parameter(qweight.data.to(params_dtype), requires_grad=False)
        set_weight_attrs(new_param, vars(qweight))
        layer.register_parameter("qweight", new_param)
        layer.qweight_type.weight_type = int(_weight_type_for_dtype(params_dtype))

    def _reserve_dequant_scratch(self, layer: torch.nn.Module):
        """Grow the #63 workspace to this layer's largest dequant target."""
        import gguf as _gguf

        params_dtype = getattr(self, "params_dtype", None)
        if params_dtype is None:
            return
        if isinstance(layer.qweight, UninitializedParameter):
            # Never-loaded module (e.g. the NEXTN draft's embed/lm_head,
            # which are replaced by module-level sharing right after the
            # loader finishes): nothing to size, and `.shape` would raise.
            return
        shard_views = getattr(layer, "_gguf_shard_views", None)
        if shard_views:
            items = [
                (layer.qweight_type.shard_weight_type[idx], view.shape)
                for idx, view in shard_views.items()
            ]
        else:
            items = [(layer.qweight_type.weight_type, tuple(layer.qweight.shape))]
        for qtype, shape in items:
            if qtype in UNQUANTIZED_TYPES or qtype not in DEQUANT_TYPES:
                continue
            block_size, type_size = _gguf.GGML_QUANT_SIZES[qtype]
            numel = shape[0] * (shape[1] // type_size * block_size)
            _reserve_dequant_workspace(numel, params_dtype, layer.qweight.device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shard_id = layer.qweight.shard_id

        if shard_id:
            # dequantize shard weights respectively
            if "q" in shard_id:
                shard_id = ["q", "k", "v"]
            else:
                # Append order == GGUF tensor order, which may NOT match the
                # fused output-partition order: a Qwen3.5 GGUF stores attn_gate
                # (z, shard 3) BEFORE attn_qkv (shard (0,1,2)), so concatenating
                # in append order yields [z|q|k|v] and the consumer's
                # [q|k|v|z] split reads garbage. Order shards by their logical
                # partition index (tuple -> its min) so the cat is correct.
                # (Also makes gate_up robust to checkpoint tensor order.)
                shard_id = sorted(
                    shard_id,
                    key=lambda s: min(s) if isinstance(s, tuple) else s,
                )
            # Zero-copy contiguous shard views (built once at load time by
            # _create_flat_weight_param): no per-forward slice+contiguous.
            shard_views = getattr(layer, "_gguf_shard_views", None)
            if shard_views is None:
                raise RuntimeError(
                    "GGUF merged layer has shard ids but no finalized shard "
                    "views; process_weights_after_loading did not run or the "
                    "layer was loaded with fewer than 2 shards."
                )
            result = []
            for idx in shard_id:
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                result.append(fused_mul_mat_gguf(x, shard_views[idx], qweight_type))
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = fused_mul_mat_gguf(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config
        # GGUF MoE has no fused-experts override path; define the attribute so
        # the `assert self.fused_experts is None` guard in apply() evaluates
        # instead of raising AttributeError on the (previously untested) MoE
        # forward.
        self.fused_experts = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        # gate up proj
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        # gate down proj
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )

        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # Assemble the GGUF MoE UninitializedParameters from their loaded
        # data_containers. On the CUDA path this is the ONLY hook that runs
        # materialize_gguf_weights (the Ascend method has its own copy), and
        # it is required for the uneven-TP expert-dim sharding path: it also
        # appends the trailing zero-pad expert and registers the global->
        # local topk remap buffer used by FusedMoE.forward_impl. Safe no-op
        # for already-materialized / non-GGUF params.
        if hasattr(layer, "materialize_gguf_weights"):
            layer.materialize_gguf_weights()

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        assert self.fused_experts is None

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        topk_weights, topk_ids, _ = topk_output
        output = fused_moe_gguf(
            x=x,
            w1=layer.w13_qweight,
            w2=layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            qweight_type=layer.w13_qweight_type.weight_type,
            qweight_type2=layer.w2_qweight_type.weight_type,
            activation=moe_runner_config.activation,
        )
        return StandardCombineInput(hidden_states=output)


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type
        hidden_size = qweight.tensor_shape[1]

        return apply_gguf_embedding(
            x, qweight, qweight_type, hidden_size, dtype=self.params_dtype
        )


class GGUFUninitializedParameter(UninitializedParameter):
    cls_to_become = Parameter
    data_container: list[torch.Tensor]


# =============================================================================
# NPU-specific implementations for Ascend hardware
# =============================================================================
def ggml_dequantize_ascend(
    qweight: torch.Tensor,
    qweight_type: int,
    rows: int,
    cols: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize GGML quantized weights for NPU.

    Uses gguf library's reference implementation which supports all GGML formats
    and is guaranteed to be correct. The dequantization runs on CPU during model
    loading, then the dequantized weights are transferred to NPU for inference.
    """

    # Move to CPU for dequantization using gguf library
    qweight_cpu = qweight.cpu().numpy()

    # Use gguf library's dequantize (supports all GGML formats)
    dequant_np = gguf_dequantize(qweight_cpu, qweight_type)

    # Convert to torch and move to target device
    result = torch.from_numpy(dequant_np).to(dtype=dtype, device=qweight.device)
    result = result.reshape(rows, cols)

    return result


class GGUFLinearAscendMethod(LinearMethodBase):
    """Linear method for GGUF on Ascend NPU."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            raise ValueError(
                f"Unsupported GGUF quantization type {WeightType(qweight_type)} in layer."
            )
        self._create_padded_weight_param(layer)
        # Pre-dequantize weights for faster inference
        self._pre_dequantize_weights(layer)

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()
            for idx in shard_id:
                id_in_container = shard_id_map[idx]
                start = sum(x.size(0) for x in data_container[:id_in_container])
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    def _pre_dequantize_weights(self, layer: torch.nn.Module):
        """Pre-dequantize GGML weights to FP16 for faster inference.

        This eliminates runtime dequantization overhead at the cost of more memory.
        """
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type

        if qweight_type in UNQUANTIZED_TYPES and qweight.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            layer.dequantized_weight = qweight
            return

        shard_id = getattr(qweight, "shard_id", None)
        has_shard_offset = hasattr(qweight, "shard_offset_map")

        if shard_id and has_shard_offset:
            # Handle sharded weights (QKV merged)
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            dequant_shards = []
            for idx in shard_id:
                start, end, offset = qweight.shard_offset_map[idx]
                shard_qtype = layer.qweight_type.shard_weight_type[idx]
                shard_data = qweight[start:end, :offset].contiguous()

                block_size, type_size = gguf.GGML_QUANT_SIZES[shard_qtype]
                shape = (
                    shard_data.shape[0],
                    shard_data.shape[1] // type_size * block_size,
                )
                dequant = ggml_dequantize_ascend(
                    shard_data, shard_qtype, *shape, self.params_dtype
                )
                dequant_shards.append(dequant)

            dequant_weight = torch.cat(dequant_shards, dim=0)
        else:
            # Handle single weight
            block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
            shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
            dequant_weight = ggml_dequantize_ascend(
                qweight, qweight_type, *shape, self.params_dtype
            )

        layer.dequantized_weight = dequant_weight

        if hasattr(layer, "qweight"):
            del layer.qweight
        if hasattr(layer, "qweight_type"):
            del layer.qweight_type

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Use pre-dequantized weight (always available after process_weights_after_loading)
        weight = layer.dequantized_weight
        out = x @ weight.T
        if bias is not None:
            out.add_(bias)
        return out


class GGUFMoEAscendMethod(FusedMoEMethodBase):
    """MoE method for GGUF on Ascend NPU."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

        # Store params_dtype for pre-dequantization
        self.params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module):
        """Pre-dequantize MoE weights to FP16 for faster inference."""

        if hasattr(layer, "materialize_gguf_weights"):
            layer.materialize_gguf_weights()

        # Check if weights are actually loaded (not still UninitializedParameter/empty)
        w13_qweight = layer.w13_qweight
        w13_qtype = layer.w13_qweight_type.weight_type

        # Pre-dequantize w13 weights (gate+up projections)
        if w13_qtype not in UNQUANTIZED_TYPES:
            num_experts = w13_qweight.shape[0]
            w13_dequant_list = []

            block_size, type_size = gguf.GGML_QUANT_SIZES[w13_qtype]

            for e in range(num_experts):
                qweight_cpu = w13_qweight[e].cpu().numpy()
                rows = w13_qweight[e].shape[0]
                cols = w13_qweight[e].shape[1] // type_size * block_size

                dequant_np = gguf_dequantize(qweight_cpu.flatten(), w13_qtype)
                dequant = (
                    torch.from_numpy(dequant_np)
                    .to(dtype=self.params_dtype, device=w13_qweight.device)
                    .reshape(rows, cols)
                    .transpose(-1, -2)
                    .contiguous()
                )
                w13_dequant_list.append(dequant)

            w13_full = torch.stack(w13_dequant_list, dim=0)

            layer.register_buffer("w13_dequant", w13_full, persistent=False)
        else:
            layer.register_buffer("w13_dequant", w13_qweight.data, persistent=False)

        # Pre-dequantize w2 weights (down projection)
        w2_qweight = layer.w2_qweight
        w2_qtype = layer.w2_qweight_type.weight_type

        if w2_qtype not in UNQUANTIZED_TYPES:
            num_experts = w2_qweight.shape[0]
            w2_dequant_list = []

            block_size, type_size = gguf.GGML_QUANT_SIZES[w2_qtype]

            for e in range(num_experts):
                qweight_cpu = w2_qweight[e].cpu().numpy()
                rows = w2_qweight[e].shape[0]
                cols = w2_qweight[e].shape[1] // type_size * block_size

                dequant_np = gguf_dequantize(qweight_cpu.flatten(), w2_qtype)
                dequant = (
                    torch.from_numpy(dequant_np)
                    .to(dtype=self.params_dtype, device=w2_qweight.device)
                    .reshape(rows, cols)
                    .transpose(-1, -2)
                    .contiguous()
                )
                w2_dequant_list.append(dequant)

            w2_full = torch.stack(w2_dequant_list, dim=0)

            layer.register_buffer("w2_dequant", w2_full, persistent=False)
        else:
            layer.register_buffer("w2_dequant", w2_qweight.data, persistent=False)

        if hasattr(layer, "w2_qweight"):
            del layer.w2_qweight
        if hasattr(layer, "w13_qweight"):
            del layer.w13_qweight

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        """Apply MoE forward pass on NPU using npu_grouped_matmul for maximum performance."""
        from sglang.srt.distributed.communication_op import (
            tensor_model_parallel_all_gather,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Check if pre-dequantized weights are available
        use_pre_dequant = hasattr(layer, "w13_dequant") and hasattr(layer, "w2_dequant")

        if not use_pre_dequant:
            raise RuntimeError(
                "GGUF MoE on NPU requires pre-dequantization (FusedMoE fix). Please report if this occurs."
            )

        w13 = layer.w13_dequant
        w2 = layer.w2_dequant

        num_experts = w13.shape[0]

        tp_size = getattr(layer, "moe_tp_size", 1)

        original_dtype = x.dtype
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]

        # Ensure correct dtypes for NPU ops
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        #  MoE routing initialization - reorder tokens by expert
        row_idx_len = num_tokens * top_k
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=x.device)
            .view(top_k, -1)
            .permute(1, 0)
            .contiguous()
        )

        sorted_hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                x, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
            )
        )

        # Compute tokens per expert
        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts
        )
        expert_tokens = expert_tokens.to(torch.int64)

        w13_gmm = w13  # No transpose needed

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[sorted_hidden_states],
            weight=[w13_gmm],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        #  Activation (SwiGLU)
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)

        # TP all-gather for intermediate dimension if needed
        if tp_size > 1:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=-1)

        w2_gmm = w2

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2_gmm],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        # Finalize routing - reorder back and apply weights
        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )

        if tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, dim=-1
            )

        # Ensure output matches input dtype
        final_hidden_states = final_hidden_states.to(dtype=original_dtype)

        return StandardCombineInput(hidden_states=final_hidden_states)


class GGUFEmbeddingAscendMethod(GGUFLinearAscendMethod):
    """Embedding method for GGUF on Ascend NPU."""

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.embedding(layer.dequantized_weight, x)
