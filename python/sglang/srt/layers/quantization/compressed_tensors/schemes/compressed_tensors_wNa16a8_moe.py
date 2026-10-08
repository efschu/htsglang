"""compressed-tensors MoE, int4 expert weights x int8 activations (W4A8), H88-B.

Sits next to ``compressed_tensors_wNa16_moe.py`` (the W4A16 Marlin scheme, left
untouched) and is selected by ``CompressedTensorsConfig.get_moe_scheme`` only
behind the switch ``moe_act_int8_requested()`` (layers/quantization/moe_act_int8.py,
H88-E: flag ``--moe-act-int8 on`` OR env ``SGLANG_MOE_ACT_INT8=1``, default OFF) on CUDA. With the switch off nothing in
this module runs.

What it does differently from ``CompressedTensorsWNA16MoE`` (whose
``create_weights`` it inherits: same checkpoint tensors, same loader contract,
``is_transposed=True``):

* the expert weights are repacked to the Marlin W4A8 layout (32x32 tiles, nibble
  order of the int8 dequant) with ``gptq_marlin_moe_repack_w4a8`` (H88-A);
* the group scales are permuted with the "single" permutation and converted to
  ``int16 = round(s / s.max() * 4096)`` (bit pattern viewed as the model dtype);
  ``s.max() / 4096`` is kept as a float32 0-d tensor per layer tensor
  (``layer.w13_act_scale_factor`` / ``layer.w2_act_scale_factor``) and is
  multiplied into the per-token activation scale at run time. Channelwise scales
  (one group) stay plain and have no factor;
* zero points (asymmetric checkpoints; prepared, no NF checkpoint uses them) are
  permuted without the column interleave of the A16 layout;
* ``apply_weights`` quantises the activations per token to int8 and runs
  ``fused_marlin_moe_w4a8``.

REFUSALS (loud, at load time, never a silent fallback to a wrong layout) for the
things this AP does not carry; H88-C removes them:

* weights are placeholders (store adoption: the repacked rows, the int16 scales
  and the per-tensor factor of the origin process would have to travel together);
* the shared expert store (``SGLANG_MOE_EXPERT_STORE_DIR``) is on: its identity
  has no layout tag, a W4A8 layer would adopt/serve W4A16 rows;
* only a subset of the expert rows is valid on the card (``repack_rows`` /
  ``_h2d_cut_rows``): the per-tensor factor and the uint4 repack must see every
  row.
Also refused: act-order (g_idx), expert parallelism / a2a backends (the kernel has
no -1 expert ids), group sizes other than -1/32/64/128, K/N not divisible by 64.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16MoE,
    _rang_karte,
)

if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationArgs

    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

__all__ = [
    "CompressedTensorsWNA16A8MoE",
    "validate_w4a8_moe_dims",
    "w4a8_process_moe_scales",
    "w4a8_moe_zero_points",
    "w4a8_repack_moe_weights",
]

logger = logging.getLogger(__name__)

#: group sizes the int8 kernels are built for (-1 = channelwise)
W4A8_GROUP_SIZES = (-1, 32, 64, 128)
#: kernel thread tile: K and N of every GEMM must be a multiple of this
#: (marlin_a8: min_thread_k = min_thread_n = 64)
W4A8_DIM_MULTIPLE = 64

MARKER = "MOE-ACT-INT8"


# ---------------------------------------------------------------------------
# boot marker
# ---------------------------------------------------------------------------

_MARK = {"layers": 0, "groups": set(), "lock": threading.Lock()}


def _mark_scheme_built(group_size: int) -> Tuple[int, str]:
    """Count one W4A8 MoE layer scheme; returns (layers so far, groups text)."""
    with _MARK["lock"]:
        _MARK["layers"] += 1
        _MARK["groups"].add(int(group_size))
        return _MARK["layers"], ",".join(str(g) for g in sorted(_MARK["groups"]))


def _reset_marker_for_tests() -> None:
    with _MARK["lock"]:
        _MARK["layers"] = 0
        _MARK["groups"].clear()


# ---------------------------------------------------------------------------
# pure helpers (CPU-testable; the kernel-bound repack is injected)
# ---------------------------------------------------------------------------


def validate_w4a8_moe_dims(
    hidden_size: int, intermediate_size: int, group_size: int
) -> None:
    """The shape contract of the two grouped GEMMs of one MoE layer.

    GEMM 1: K = hidden, N = 2 * intermediate; GEMM 2: K = intermediate,
    N = hidden. The kernel picks a thread tile (k, n) in {(128,128), (64,128),
    (128,64), (64,256)} that divides the problem, so K and N must be multiples of
    64; a group size must divide K."""
    if group_size not in W4A8_GROUP_SIZES:
        raise ValueError(
            f"{MARKER}: group_size {group_size} has no int8 kernel "
            f"(supported: {W4A8_GROUP_SIZES})"
        )
    dims = {
        "hidden (K of GEMM 1, N of GEMM 2)": hidden_size,
        "2*intermediate (N of GEMM 1)": 2 * intermediate_size,
        "intermediate (K of GEMM 2)": intermediate_size,
    }
    for name, v in dims.items():
        if v % W4A8_DIM_MULTIPLE != 0:
            raise ValueError(
                f"{MARKER}: {name} = {v} is not a multiple of {W4A8_DIM_MULTIPLE} "
                f"(hidden={hidden_size}, intermediate per partition={intermediate_size}); "
                f"the Marlin W4A8 kernel needs it"
            )
    if group_size != -1:
        for name, k in (("hidden", hidden_size), ("intermediate", intermediate_size)):
            if k % group_size != 0:
                raise ValueError(
                    f"{MARKER}: {name} = {k} is not divisible by group_size {group_size}"
                )


def w4a8_process_moe_scales(
    s: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Group scales [E, G, N] (model dtype) -> kernel scales.

    Permutation "single" (8-bit activations use it for every group size), then,
    for G > 1, the int16 x4096 trick over the WHOLE tensor (one factor per layer
    tensor, all experts, as vLLM). Returns (scales [E, G, N] in s.dtype -- for
    G > 1 the int16 bit pattern --, factor float32 0-d or None for G == 1)."""
    from sglang.jit_kernel import marlin_w4a8_utils as U

    E, G, N = s.shape
    if E == 0:
        return torch.empty_like(s), None
    flat = s.reshape(E * G, N)
    # size_k is only used by the A16 branch of the permutation; the a8 branch
    # ignores it and group_size only has to differ from "channelwise" cases
    perm = U.marlin_permute_scales(
        flat, size_k=G, size_n=N, group_size=-1 if G == 1 else 1, is_a_8bit=True
    ).reshape(E, G, N)
    if G == 1:
        return perm.contiguous(), None
    if float(perm.float().abs().max()) == 0.0:
        # all-zero scales (e.g. a layer of pad experts): nothing to scale, avoid 0/0
        return torch.zeros_like(perm), torch.ones((), dtype=torch.float32, device=s.device)
    out, factor = U.marlin_act_int8_process_scales(perm)
    return out.contiguous(), factor.to(torch.float32).reshape(())


def w4a8_moe_zero_points(q_zp_packed: torch.Tensor, size_n: int) -> torch.Tensor:
    """AWQ-style packed zero points [E, G, N/8] int32 -> kernel layout.

    unpack, undo the AWQ column interleave, apply the scale permutation, repack
    WITHOUT the A16 column interleave (the int8 dequant reads nibbles in natural
    order). Pure torch on the tensor's own device; matches
    ``marlin_w4a8_utils.marlin_zero_points`` (tested)."""
    import numpy

    from sglang.srt.layers.quantization.marlin_utils import (
        _pack_cols_torch,
        _unpack_cols_torch,
        get_scale_perms,
    )

    E, G, W = q_zp_packed.shape
    if E == 0:
        return torch.empty_like(q_zp_packed)
    assert W * 8 == size_n, f"zero-point width {W}*8 != size_n {size_n}"
    dev = q_zp_packed.device
    flat = q_zp_packed.reshape(E * G, W)
    q = _unpack_cols_torch(flat, 4, E * G, size_n)
    undo = torch.tensor(numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7])), device=dev)
    q = q.reshape((-1, undo.numel()))[:, undo].reshape((-1, size_n)).contiguous()
    scale_perm, _ = get_scale_perms()
    perm = torch.tensor(scale_perm, device=dev)
    zp = q.reshape((-1, perm.numel()))[:, perm].reshape((-1, size_n)).contiguous()
    out = _pack_cols_torch(zp, 4, E * G, size_n)
    return out.reshape(E, G, W).contiguous()


def w4a8_repack_moe_weights(
    packed: torch.Tensor,
    packed_factor: int,
    num_bits: int,
    repack_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """GPTQ/CT-packed expert stack int32 [E, K/8, N] -> Marlin W4A8 layout
    int32 [E, K/16, N*16/8] (``gptq_marlin_moe_repack_w4a8``; ``repack_fn`` lets
    a test inject the CPU reference)."""
    if repack_fn is None:
        from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_moe_repack_w4a8

        repack_fn = gptq_marlin_moe_repack_w4a8
    size_k = packed.shape[1] * packed_factor
    size_n = packed.shape[2]
    return repack_fn(packed, size_k, size_n, num_bits)


# ---------------------------------------------------------------------------
# the scheme
# ---------------------------------------------------------------------------


class CompressedTensorsWNA16A8MoE(CompressedTensorsWNA16MoE):
    """int4 (g32/g64/g128/channel, sym or asym) experts x dynamic per-token int8
    activations on the Marlin W4A8 kernels (CUDA, compute capability >= 8.0)."""

    def __init__(
        self,
        quant_config: "CompressedTensorsConfig",
        weight_quant: "QuantizationArgs",
        num_gpu_experts: int = -1,
    ):
        super().__init__(quant_config, weight_quant, num_gpu_experts)
        if self.num_bits != 4:
            raise ValueError(
                f"{MARKER}: only int4 expert weights have an int8-activation Marlin "
                f"kernel, got num_bits={self.num_bits}"
            )
        if self.actorder == "group":
            raise NotImplementedError(
                f"{MARKER}: group act-order (g_idx) is not supported by the "
                f"Marlin W4A8 kernel (no g_idx path)"
            )
        if self.strategy != "channel" and self.group_size not in W4A8_GROUP_SIZES:
            raise ValueError(
                f"{MARKER}: group_size {self.group_size} has no int8 kernel "
                f"(supported: {W4A8_GROUP_SIZES})"
            )
        self.w4a8_group_size = -1 if self.strategy == "channel" else int(self.group_size)
        layers, groups = _mark_scheme_built(self.w4a8_group_size)
        logger.info(
            "%s active layers=%d groups=%s (int4 experts x int8 activations, Marlin W4A8; "
            "layers = MoE layer schemes built so far in this process, groups = group sizes seen)",
            MARKER,
            layers,
            groups,
        )

    # -- guards ---------------------------------------------------------

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    def _refuse_until_h88c(self, layer) -> None:
        from sglang.srt.layers.moe import expert_store

        why = None
        try:
            from sglang.srt.weg2 import adopt as _adopt

            if _adopt.weights_are_placeholder():
                why = (
                    "this rank holds placeholder weights (store adoption / dummy "
                    "load): the repacked W4A8 rows, their int16 scales and the "
                    "per-tensor factor of the origin process would have to travel "
                    "together"
                )
        except ImportError:
            pass
        if why is None and expert_store.store_enabled():
            why = (
                "the shared expert store is on (SGLANG_MOE_EXPERT_STORE_DIR): its "
                "identity has no layout tag, so P/D could adopt W4A16 rows into a "
                "W4A8 layer"
            )
        if why is None:
            num_experts = layer.w13_weight_g_idx.shape[0]
            from sglang.srt.layers.moe.store_adopt import repack_rows

            if (
                repack_rows(layer, num_experts) is not None
                or getattr(layer, "_h2d_cut_rows", None) is not None
            ):
                why = (
                    "only a subset of the expert rows is valid on the card "
                    "(vetoed/cut rows): the per-tensor scale factor and the repack "
                    "must see every row"
                )
        if why is not None:
            raise RuntimeError(
                f"{MARKER} REFUSED (H88-C pending): {why}. Switch SGLANG_MOE_ACT_INT8 "
                f"off for this boot or wait for the loader/offload/store AP (H88-C)."
            )

    # -- weights --------------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"{MARKER}: the kernel writes in the scale dtype, which must be "
                f"float16 or bfloat16, got {params_dtype}"
            )
        validate_w4a8_moe_dims(
            hidden_size, intermediate_size_per_partition, self.w4a8_group_size
        )
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        layer.w13_act_scale_factor = None
        layer.w2_act_scale_factor = None

    def _repack_to_marlin(self, layer: torch.nn.Module) -> None:
        from sglang.srt.managers.weg2_memory_saver import back_into_tag_pool

        self._refuse_until_h88c(layer)

        zero_pad = getattr(layer, "zero_expert_shard_pad", None)
        if callable(zero_pad):
            zero_pad()

        if not hasattr(layer, "_original_shapes"):
            layer._original_shapes = {}

        def replace_tensor(name, new_t):
            target_attr = getattr(layer, name)
            if name not in layer._original_shapes:
                layer._original_shapes[name] = tuple(target_attr.shape)
            with back_into_tag_pool():
                target_attr.resize_(new_t.shape)
            target_attr.copy_(new_t)
            del new_t

        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device
        # act-order is refused in __init__: g_idx stay empty, as in the A16 scheme
        for name in (
            "w13_weight_g_idx",
            "w2_weight_g_idx",
            "w13_g_idx_sort_indices",
            "w2_g_idx_sort_indices",
        ):
            setattr(
                layer,
                name,
                torch.nn.Parameter(
                    torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                    requires_grad=False,
                ),
            )

        # weights -> W4A8 layout (K/N from the CT-packed shape [E, K/8, N])
        replace_tensor(
            "w13_weight_packed",
            w4a8_repack_moe_weights(
                layer.w13_weight_packed, self.packed_factor, self.num_bits
            ),
        )
        replace_tensor(
            "w2_weight_packed",
            w4a8_repack_moe_weights(
                layer.w2_weight_packed, self.packed_factor, self.num_bits
            ),
        )

        # scales -> single permutation, int16 x4096, one factor per layer tensor
        s13, f13 = w4a8_process_moe_scales(layer.w13_weight_scale)
        replace_tensor("w13_weight_scale", s13)
        s2, f2 = w4a8_process_moe_scales(layer.w2_weight_scale)
        replace_tensor("w2_weight_scale", s2)
        with back_into_tag_pool():
            layer.w13_act_scale_factor = None if f13 is None else f13.clone()
            layer.w2_act_scale_factor = None if f2 is None else f2.clone()

        if not self.sym:
            replace_tensor(
                "w13_weight_zero_point",
                w4a8_moe_zero_points(
                    layer.w13_weight_zero_point,
                    layer.w13_weight_zero_point.shape[2] * self.packed_factor,
                ),
            )
            replace_tensor(
                "w2_weight_zero_point",
                w4a8_moe_zero_points(
                    layer.w2_weight_zero_point,
                    layer.w2_weight_zero_point.shape[2] * self.packed_factor,
                ),
            )

        with back_into_tag_pool():
            from sglang.srt.layers.quantization.marlin_utils import (
                marlin_make_workspace,
            )

            layer.workspace = marlin_make_workspace(_rang_karte(layer), 4)
        layer.is_marlin_converted = True

        from sglang.srt.layers.moe.expert_offload import (
            presplit_expert_offload_after_repack,
        )

        presplit_expert_offload_after_repack(layer)

    def restore_weights_before_loading(self, layer: torch.nn.Module):
        super().restore_weights_before_loading(layer)
        layer.w13_act_scale_factor = None
        layer.w2_act_scale_factor = None

    # -- runtime --------------------------------------------------------

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        from sglang.srt.layers.moe import get_moe_a2a_backend

        if not get_moe_a2a_backend().is_none():
            raise NotImplementedError(
                f"{MARKER}: the Marlin W4A8 MoE kernel has no expert-parallel / a2a "
                f"path (expert ids must not contain -1); got a2a backend "
                f"{get_moe_a2a_backend()}"
            )
        # the runner object only carries the config: apply_weights below calls
        # fused_marlin_moe_w4a8 directly, like the A16 scheme calls fused_marlin_moe
        self.moe_runner_config = moe_runner_config
        self.runner = None

    def get_marlin_quant_info(self, layer):
        raise NotImplementedError(
            f"{MARKER}: MarlinMoeQuantInfo describes the W4A16 layout; the W4A8 "
            f"layer is served by fused_marlin_moe_w4a8 only"
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe_w4a8 import (
            fused_marlin_moe_w4a8,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert self.moe_runner_config.activation in (
            "silu",
            "gelu",
        ), "Only SiLU/GeLU activations are supported."

        expert_map = None
        if hasattr(layer, "dispatcher") and hasattr(
            layer.dispatcher, "local_expert_mapping"
        ):
            expert_map = layer.dispatcher.local_expert_mapping
        if expert_map is not None:
            raise NotImplementedError(
                f"{MARKER}: expert parallelism (expert_map) is not supported by the "
                f"Marlin W4A8 MoE kernel"
            )

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _router_logits = dispatch_output.topk_output

        output = fused_marlin_moe_w4a8(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            w1_act_factor=layer.w13_act_scale_factor,
            w2_act_factor=layer.w2_act_scale_factor,
            w1_zeros=layer.w13_weight_zero_point if not self.sym else None,
            w2_zeros=layer.w2_weight_zero_point if not self.sym else None,
            workspace=layer.workspace,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
            clamp_limit=self.moe_runner_config.swiglu_limit,
            activation=self.moe_runner_config.activation,
            is_gated=self.moe_runner_config.is_gated,
        )
        return StandardCombineInput(hidden_states=output)
