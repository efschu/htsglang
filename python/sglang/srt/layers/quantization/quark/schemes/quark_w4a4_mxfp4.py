# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Callable, Optional

import torch

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.quark.schemes import QuarkLinearScheme
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import direct_register_custom_op, mxfp_supported

_is_hip = is_hip()

# aiter backs the MXFP4 path on AMD. It supports gfx90a/908/942/950/1100 --
# NOT gfx900 (Vega), where it is simply not installed.
#
# This matters far beyond MXFP4: nearly every core layer imports
# layers/quantization/__init__.py, which pulls in quark -> this module. An
# unguarded aiter import therefore takes down the ENTIRE forward path on such a
# rank, and because it raises first it MASKS every other blocker behind it
# (measured: it hid the sgl_kernel imports in activation.py and spec_utils.py).
#
# Guarding it is the correct behaviour, not a workaround: without aiter this
# quantization scheme cannot run, and a rank that does not select MXFP4 has no
# reason to care. `get_min_capability`/scheme selection already decides whether
# it is used; the import must not decide it for the whole process.
_HAS_AITER_MXFP4 = False
if _is_hip:
    try:
        from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import (
            fused_gemm_afp4wfp4_split_cat as _fused_gemm_afp4wfp4_split_cat_orig,
        )
        from aiter.ops.triton.gemm_afp4wfp4 import (
            gemm_afp4wfp4 as _gemm_afp4wfp4_orig,
        )
        from aiter.ops.triton.gemm_afp4wfp4_pre_quant_atomic import (
            gemm_afp4wfp4_pre_quant as _gemm_afp4wfp4_pre_quant_orig,
        )
        from aiter.ops.triton.quant import (
            dynamic_mxfp4_quant as _dynamic_mxfp4_quant_orig,
        )

        _HAS_AITER_MXFP4 = True
    except ImportError:
        # No aiter (e.g. gfx900): leave the scheme undefined-but-importable.
        # QuarkW4A4MXFP4.__init__ raises a clear error if anything tries to
        # actually select it.
        pass

if _is_hip and _HAS_AITER_MXFP4:

    def _aiter_gemm_afp4wfp4(
        x: torch.Tensor,
        w: torch.Tensor,
        x_scales: torch.Tensor,
        w_scales: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        _gemm_afp4wfp4_orig(x, w, x_scales, w_scales, y.dtype, y)

    def _aiter_gemm_afp4wfp4_fake(
        x: torch.Tensor,
        w: torch.Tensor,
        x_scales: torch.Tensor,
        w_scales: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        return None

    direct_register_custom_op(
        op_name="aiter_gemm_afp4wfp4",
        op_func=_aiter_gemm_afp4wfp4,
        mutates_args=["y"],
        fake_impl=_aiter_gemm_afp4wfp4_fake,
    )

    def gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, y):
        torch.ops.sglang.aiter_gemm_afp4wfp4(x, w, x_scales, w_scales, y)

    def _aiter_gemm_afp4wfp4_pre_quant(
        x: torch.Tensor,
        w: torch.Tensor,
        w_scales: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        _gemm_afp4wfp4_pre_quant_orig(x, w, w_scales, y.dtype, y)

    def _aiter_gemm_afp4wfp4_pre_quant_fake(
        x: torch.Tensor,
        w: torch.Tensor,
        w_scales: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        return None

    direct_register_custom_op(
        op_name="aiter_gemm_afp4wfp4_pre_quant",
        op_func=_aiter_gemm_afp4wfp4_pre_quant,
        mutates_args=["y"],
        fake_impl=_aiter_gemm_afp4wfp4_pre_quant_fake,
    )

    def gemm_afp4wfp4_pre_quant(x, w, w_scales, dtype, y):
        torch.ops.sglang.aiter_gemm_afp4wfp4_pre_quant(x, w, w_scales, y)

    def _aiter_dynamic_mxfp4_quant(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _dynamic_mxfp4_quant_orig(x)

    def _aiter_dynamic_mxfp4_quant_fake(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M, N = x.shape
        x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
        blockscale = torch.empty(
            (M, (N + 31) // 32), dtype=torch.uint8, device=x.device
        )
        return x_fp4, blockscale

    direct_register_custom_op(
        op_name="aiter_dynamic_mxfp4_quant",
        op_func=_aiter_dynamic_mxfp4_quant,
        mutates_args=[],
        fake_impl=_aiter_dynamic_mxfp4_quant_fake,
    )

    def dynamic_mxfp4_quant(x):
        return torch.ops.sglang.aiter_dynamic_mxfp4_quant(x)

    def _aiter_fused_gemm_split_cat(
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        S1: int,
        S2: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _fused_gemm_afp4wfp4_split_cat_orig(
            x=x,
            w=w,
            y=y,
            x_scale=x_scale,
            w_scale=w_scale,
            S1=S1,
            S2=S2,
            dtype=y.dtype,
        )

    def _aiter_fused_gemm_split_cat_fake(
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        S1: int,
        S2: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M = x.shape[0]
        D = y.shape[1]
        S3 = y.shape[2]
        c1 = torch.empty((M, D, S1 + S3), dtype=y.dtype, device=x.device)
        c2 = torch.empty((M, D, S2), dtype=y.dtype, device=x.device)
        return c1, c2

    direct_register_custom_op(
        op_name="aiter_fused_gemm_split_cat",
        op_func=_aiter_fused_gemm_split_cat,
        mutates_args=[],
        fake_impl=_aiter_fused_gemm_split_cat_fake,
    )

    def fused_gemm_afp4wfp4_split_cat(x, w, y, x_scale, w_scale, S1, S2, dtype):
        return torch.ops.sglang.aiter_fused_gemm_split_cat(
            x, w, y, x_scale, w_scale, S1, S2
        )


__all__ = ["QuarkW4A4MXFP4"]
logger = logging.getLogger(__name__)

OCP_MX_BLOCK_SIZE = 32


class QuarkW4A4MXFP4(QuarkLinearScheme):

    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any],
        is_checkpoint_mxfp4_serialized: bool = True,
    ):
        # The module-scope aiter import is guarded so that a rank without
        # aiter (e.g. gfx900) can still IMPORT this module -- otherwise it
        # takes down every core layer that transitively reaches quark. Selecting
        # the scheme without aiter is a different matter and must fail loudly
        # here, rather than as a NameError deep inside create_weights().
        if _is_hip and not _HAS_AITER_MXFP4:
            raise NotImplementedError(
                "QuarkW4A4MXFP4 requires the aiter kernel library, which is "
                "not available on this AMD device (aiter supports "
                "gfx90a/908/942/950/1100; gfx900/Vega is not among them). "
                "Use a different quantization scheme on this rank."
            )
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized

        if not self.is_checkpoint_mxfp4_serialized:
            if not mxfp_supported():
                raise NotImplementedError(
                    "Online MXFP4 quantization requires an AMD ROCm device with "
                    "FP4 hardware support (gfx95x, e.g. MI355x)."
                )
            logger.info_once(
                "Using online MXFP4 quantization from a higher precision checkpoint. Beware that this optimization may degrade prediction quality - please validate your model accuracy. More details at https://docs.sglang.io/advanced_features/quantization.html#online-quantization."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not self.is_checkpoint_mxfp4_serialized:
            assert layer.weight.dtype == torch.uint8
            assert layer.weight_scale.dtype == torch.uint8

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        self.input_size_per_partition = input_size_per_partition

        output_size_per_partition = sum(output_partition_sizes)
        self.output_size_per_partition = output_size_per_partition

        layer.logical_widths = output_partition_sizes

        original_weight_loader = weight_loader
        if not self.is_checkpoint_mxfp4_serialized:
            weight_loader = self.get_online_mxfp4_weight_loader(layer, weight_loader)

        # WEIGHT
        # Both serialized and online quantization use packed uint8 format
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=original_weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def get_online_mxfp4_weight_loader(
        self,
        layer,
        original_weight_loader: Callable,
    ) -> Callable:
        """
        Wrap the original weight loader to perform online MXFP4 quantization.
        """

        def online_mxfp4_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            shard_id: int | str | None = None,
        ):
            # Materialize on device the loaded weight.
            loaded_weight = loaded_weight.to(param.device)

            # Quantize the loaded weight shard immediately. Since MXFP4 uses per-group quantization, there is no need to load all shards (e.g. q_proj, k_proj, v_proj) before doing online quantization.
            qweight, weight_scale = dynamic_mxfp4_quant(loaded_weight)

            # Required e.g. for q_proj, k_proj, v_proj.
            kwargs = {}
            if shard_id is not None:
                kwargs["loaded_shard_id"] = shard_id

            # Use the original weight loader to handle the loading logic
            # (e.g. qkv sharding, etc.)
            original_weight_loader(param, qweight, **kwargs)

            layer.weight_scale.weight_loader(layer.weight_scale, weight_scale, **kwargs)

        return online_mxfp4_weight_loader

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Bias will be added after the GEMM if provided
        three_d = False
        fused_gemm_split_cat = False
        x_s = None
        y = None

        if isinstance(x, tuple):
            assert len(x) in [
                2,
                3,
                5,
            ], "For tuple input, only (x, x_s), (x, x_s, y), or (x, y, S1, S2, out_dtype) formats are accepted"
            if len(x) == 2:
                x, x_s = x
            elif len(x) == 3:
                x, x_s, y = x
            elif len(x) == 5:
                x, y, S1, S2, out_dtype = x
                fused_gemm_split_cat = True

        use_fused_quant_gemm = (
            not fused_gemm_split_cat
            and x_s is None
            and y is not None
            and layer.weight.shape[0] == y.shape[1]
        )

        if x.dim() == 3:
            three_d = True
            x = x.view(-1, x.shape[-1])
            output_shape = [*x.shape[:-1], layer.weight.shape[0]]

        # use_fused_quant_gemm = true, x_q is a bf16/fp16 num
        # x_s is not None = true, x_q is uint8 num
        if use_fused_quant_gemm or x_s is not None:
            x_q = x
        else:
            x_q, x_s = dynamic_mxfp4_quant(x)

        if y is None:
            y = torch.empty(
                x_q.shape[0],
                layer.weight.shape[0],
                device=x_q.device,
                dtype=self.out_dtype,
            )

        if use_fused_quant_gemm:
            gemm_afp4wfp4_pre_quant(x_q, layer.weight, layer.weight_scale, y.dtype, y)
            y = y.to(x.dtype)
        elif fused_gemm_split_cat:
            k, v = fused_gemm_afp4wfp4_split_cat(
                x=x_q,
                w=layer.weight,
                y=y,
                x_scale=x_s,
                w_scale=layer.weight_scale,
                S1=S1,
                S2=S2,
                dtype=out_dtype,
            )
        else:
            gemm_afp4wfp4(x_q, layer.weight, x_s, layer.weight_scale, self.out_dtype, y)

        if bias is not None:
            y = y + bias

        if fused_gemm_split_cat:
            return k, v
        elif three_d:
            return y.view(*output_shape)
        else:
            return y
