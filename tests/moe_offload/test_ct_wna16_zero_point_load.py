# SPDX-License-Identifier: Apache-2.0
"""Loader gate: compressed-tensors WNA16 MoE zero points are stored flipped
like the packed weights and must be transposed on load (Qwen3.8-Flash-Next
AWQ, cyankiwi checkpoint, measured 16.09.2026):

    down_proj.weight_packed      [2560, 80]   -> param w2_weight_packed  [E, in/pack, out]
    down_proj.weight_scale       [2560, 20]   -> param w2_weight_scale   [E, groups, out]
    down_proj.weight_zero_point  [320, 20]    -> param w2_weight_zero_point [E, groups, out/pack]
    gate_proj.weight_zero_point  [80, 80]     (square: an untransposed load is silently wrong)

Upstream transposed everything except names containing "zero"; w2 died on
"320 vs 20" and w13 would have loaded the wrong axis without a sound.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

HIDDEN = 2560
INTER = 640
GROUP = 32
PACK = 8
E = 2


def _ct_layer(monkeypatch, sym=False):
    """A FusedMoE shell wired for the compressed-tensors WNA16 loader path."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
        CompressedTensorsWNA16MoE,
    )

    scheme = CompressedTensorsWNA16MoE.__new__(CompressedTensorsWNA16MoE)
    scheme.quant_config = type(
        "C", (), {"quant_format": "pack-quantized", "get_name": lambda self: "compressed-tensors"}
    )()
    scheme.num_bits = 4
    scheme.packed_factor = PACK
    scheme.strategy = "group"
    scheme.group_size = GROUP
    scheme.actorder = None
    scheme.sym = sym
    scheme.num_gpu_experts = -1

    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.num_local_experts = E
    layer.moe_tp_size = 1
    layer.moe_tp_rank = 0
    layer.moe_ep_size = 1
    layer.quant_method = scheme
    layer.scheme = scheme
    layer.use_padded_loading = False
    layer.use_presharded_weights = False
    layer.use_triton_kernels = False
    layer.use_flashinfer_trtllm_moe = False
    layer.use_flashinfer_mxfp4_moe = False
    layer.use_deep_gemm = False
    layer._has_fused_shared = False
    layer.num_fused_shared_experts = 0
    layer.quant_config = scheme.quant_config
    layer.moe_runner_config = type("R", (), {"is_gated": True})()
    layer.layer_id = 0
    layer.moe_tp_family = None
    layer.moe_tp_units = INTER // 128
    monkeypatch.delenv("SGLANG_MOE_RESIDENT_EXPERT_FRACTION", raising=False)
    with torch.device("cpu"):
        scheme.create_weights(
            layer,
            num_experts=E,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTER,
            params_dtype=torch.bfloat16,
            weight_loader=layer.weight_loader,
        )
    return layer


def _ckpt_tensors():
    """Checkpoint-shaped tensors with position-coded values."""
    def coded(*shape, dtype):
        n = 1
        for d in shape:
            n *= d
        return torch.arange(n, dtype=torch.int64).reshape(shape).to(dtype)

    return {
        "down_zp": coded(HIDDEN // PACK, INTER // GROUP, dtype=torch.int32),  # [320, 20]
        "gate_zp": coded(INTER // PACK, HIDDEN // GROUP, dtype=torch.int32),  # [80, 80]
        "down_scale": coded(HIDDEN, INTER // GROUP, dtype=torch.bfloat16),  # [2560, 20]
    }


def test_w2_zero_point_is_transposed_like_the_weight(monkeypatch):
    layer = _ct_layer(monkeypatch)
    t = _ckpt_tensors()
    layer._weight_loader_impl(
        param=layer.w2_weight_zero_point,
        loaded_weight=t["down_zp"],
        weight_name="experts.0.down_proj.weight_zero_point",
        shard_id="w2",
        expert_id=1,
    )
    assert layer.w2_weight_zero_point.shape == (E, INTER // GROUP, HIDDEN // PACK)
    assert torch.equal(layer.w2_weight_zero_point[1], t["down_zp"].t())


def test_w13_zero_point_lands_on_the_group_axis(monkeypatch):
    """Square [80, 80]: only a value check can tell the axes apart."""
    layer = _ct_layer(monkeypatch)
    t = _ckpt_tensors()
    layer._weight_loader_impl(
        param=layer.w13_weight_zero_point,
        loaded_weight=t["gate_zp"],
        weight_name="experts.0.gate_proj.weight_zero_point",
        shard_id="w1",
        expert_id=0,
    )
    assert layer.w13_weight_zero_point.shape == (E, HIDDEN // GROUP, 2 * INTER // PACK)
    # gate half = the first INTER//PACK columns, rows = groups
    got = layer.w13_weight_zero_point[0][:, : INTER // PACK]
    assert torch.equal(got, t["gate_zp"].t())


def test_w2_scale_keeps_its_transposed_load(monkeypatch):
    layer = _ct_layer(monkeypatch)
    t = _ckpt_tensors()
    layer._weight_loader_impl(
        param=layer.w2_weight_scale,
        loaded_weight=t["down_scale"],
        weight_name="experts.0.down_proj.weight_scale",
        shard_id="w2",
        expert_id=0,
    )
    assert torch.equal(layer.w2_weight_scale[0], t["down_scale"].t())
