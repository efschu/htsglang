"""H68 (port of upstream sglang PR #38092, open): the NVFP4 fused-MoE Marlin
path no longer allocates the swizzled block-scale copies.

``ModelOptNvFp4FusedMoEMethod.create_weights`` built ``w13/w2_blockscale_swizzled``
from the still-empty scale buffers for every backend except TRT-LLM. On the
Marlin path nothing ever reads them -- ``process_weights_after_loading``
returns right after ``prepare_moe_nvfp4_layer_for_marlin`` -- and
``swizzle_blockscale`` ends in ``.cuda()``: the dead copy (~10 % of every
NVFP4 expert) sat on the CARD even for experts allocated on the host.

Hermetic, CPU only: a bare method instance, a bare layer module; the native
branch runs with a CPU stand-in for ``swizzle_blockscale``.
"""

import types
from unittest import mock

import pytest

try:
    import torch

    from sglang.srt.layers.quantization import modelopt_quant as M
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _create(use_marlin):
    meth = object.__new__(M.ModelOptNvFp4FusedMoEMethod)
    meth.quant_config = types.SimpleNamespace(is_checkpoint_nvfp4_serialized=True, group_size=16)
    meth.use_marlin_fallback = use_marlin
    meth.enable_flashinfer_trtllm_moe = False
    layer = torch.nn.Module()
    layer.num_local_experts = 2
    layer.num_experts = 2
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True)
    meth.create_weights(layer, num_experts=2, hidden_size=64,
                        intermediate_size_per_partition=32,
                        params_dtype=torch.bfloat16, weight_loader=lambda *a, **k: None)
    return layer


def test_marlin_path_skips_swizzled_blockscales():
    calls = []

    def cpu_swizzle(t):
        calls.append(tuple(t.shape))
        return t.clone()

    with mock.patch.object(M, "swizzle_blockscale", cpu_swizzle):
        marlin = _create(True)
        assert calls == []
        native = _create(False)
    assert marlin.w13_blockscale_swizzled is None
    assert marlin.w2_blockscale_swizzled is None
    # the scales the Marlin repack DOES read are still created
    assert tuple(marlin.w13_weight_scale.shape) == (2, 64, 4)
    assert tuple(marlin.w2_weight_scale.shape) == (2, 64, 2)
    assert len(calls) == 2
    assert isinstance(native.w13_blockscale_swizzled, torch.nn.Parameter)
    assert isinstance(native.w2_blockscale_swizzled, torch.nn.Parameter)


def test_subclass_without_the_flag_keeps_old_behaviour():
    """A method object that never ran this __init__ (no attribute) keeps
    allocating: the skip needs the flag, never guesses it."""
    calls = []
    meth = object.__new__(M.ModelOptNvFp4FusedMoEMethod)
    meth.quant_config = types.SimpleNamespace(is_checkpoint_nvfp4_serialized=True, group_size=16)
    meth.enable_flashinfer_trtllm_moe = False
    layer = torch.nn.Module()
    layer.num_local_experts = layer.num_experts = 2
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True)
    with mock.patch.object(M, "swizzle_blockscale", lambda t: calls.append(1) or t.clone()):
        meth.create_weights(layer, num_experts=2, hidden_size=64, intermediate_size_per_partition=32,
                            params_dtype=torch.bfloat16, weight_loader=lambda *a, **k: None)
    assert len(calls) == 2
