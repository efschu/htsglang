# SPDX-License-Identifier: Apache-2.0
"""Host-side gate for the fp8 MoE load-time presplit (#256).

Before #256 the fp8 expert stack was committed on the default device by
``Fp8MoEMethod.create_weights`` and only split at first forward, so a 31 GB fp8
checkpoint OOM'd a 32 GB card before the offload could do anything. These tests
pin the two halves of the fix without loading a model or running a kernel:

  * ``create_weights`` allocates the expert stack on the host when the offload
    fraction is set, and on the default device when it is not;
  * ``process_weights_after_loading`` ends in the presplit;
  * the presplit itself keeps [R+C] slots, spills [E-R] to a pinned host pool,
    and puts every staged tensor of one expert -- weight, scale, bias -- on the
    SAME pool row, which is what makes the shared fetch plan correct.

``pin_memory()`` needs a CUDA context, so the file is skipped without one; no
GPU memory is allocated and no kernel runs (all tensors stay on the host).

Run:
  PYTHONPATH=python python -m pytest tests/moe_offload/test_fp8_presplit.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the pinned spill pool needs a CUDA context (host tensors only)",
)

from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    MoEExpertOffloadCache,
    presplit_expert_offload_after_repack,
    reset_expert_offload_release,
    resident_slot_count,
    scratch_slot_count,
)

FRACTION_ENV = "SGLANG_MOE_RESIDENT_EXPERT_FRACTION"

# Qwen3.6-35B-A3B-FP8 geometry, scaled down on the expert axis so the test is
# cheap: hidden 2048, moe_intermediate 512, weight_block_size [128, 128].
E = 32
HIDDEN = 256
INTER = 128
BLOCK = 128


@pytest.fixture
def fraction_025(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    reset_expert_offload_release()
    yield 0.25
    reset_expert_offload_release()


def _block_fp8_layer(num_experts=E, with_bias=False):
    """A stub FusedMoE layer carrying block-fp8 expert tensors on the host.

    Rows are filled with the expert id so a misrouted row is visible as a value,
    not just as a shape.
    """
    layer = torch.nn.Module()
    layer.num_local_experts = num_experts

    def _p(*shape, dtype=torch.float32):
        t = torch.empty(shape, dtype=dtype)
        for e in range(num_experts):
            t[e] = e
        return torch.nn.Parameter(t, requires_grad=False)

    layer.register_parameter(
        "w13_weight", _p(num_experts, 2 * INTER, HIDDEN, dtype=torch.float8_e4m3fn)
    )
    layer.register_parameter(
        "w2_weight", _p(num_experts, HIDDEN, INTER, dtype=torch.float8_e4m3fn)
    )
    layer.register_parameter(
        "w13_weight_scale_inv",
        _p(num_experts, 2 * (INTER // BLOCK), HIDDEN // BLOCK),
    )
    layer.register_parameter(
        "w2_weight_scale_inv",
        _p(num_experts, HIDDEN // BLOCK, INTER // BLOCK),
    )
    if with_bias:
        layer.register_parameter("w13_weight_bias", _p(num_experts, 2 * INTER))
        layer.register_parameter("w2_weight_bias", _p(num_experts, HIDDEN))
    return layer


def _staged_attrs(layer):
    return [
        a
        for a in MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS
        if getattr(layer, a, None) is not None
    ]


# --------------------------------------------------------------------------
# presplit mechanics
# --------------------------------------------------------------------------


def test_presplit_allocates_r_plus_c_not_the_full_stack(fraction_025):
    layer = _block_fp8_layer()
    before = {a: getattr(layer, a).data.clone() for a in _staged_attrs(layer)}

    presplit_expert_offload_after_repack(layer)

    R = resident_slot_count(E, 0.25)
    C = scratch_slot_count(R)
    assert (R, C) == (8, 8)
    presplit = layer._moe_offload_presplit
    assert layer._moe_offload_full_experts == E
    assert set(presplit) == set(before)

    for attr, (buf, spill) in presplit.items():
        assert buf.shape[0] == R + C, attr
        assert spill.shape[0] == E - R, attr
        assert spill.is_pinned(), attr
        # resident block is the first R experts, byte for byte
        assert torch.equal(
            buf[:R].view(torch.uint8), before[attr][:R].view(torch.uint8)
        )
        # the registered param is a placeholder: it must not keep the stack alive
        assert getattr(layer, attr).shape[0] == 0, attr


def test_scales_and_bias_share_the_expert_row_with_their_weight(fraction_025):
    """The spill pool is indexed by one shared fetch plan (expert -> row), so a
    spill expert's weight, its block scales and its bias must all sit at the
    same row. This is the fp8-specific half of #256."""
    layer = _block_fp8_layer(with_bias=True)
    attrs = _staged_attrs(layer)
    assert "w13_weight_scale_inv" in attrs and "w13_weight_bias" in attrs
    before = {a: getattr(layer, a).data.clone() for a in attrs}

    presplit_expert_offload_after_repack(layer)

    R = resident_slot_count(E, 0.25)
    presplit = layer._moe_offload_presplit
    for expert in range(R, E):
        row = expert - R  # the index _fetch uses, identically for every attr
        for attr, (_buf, spill) in presplit.items():
            assert torch.equal(
                spill[row].view(torch.uint8), before[attr][expert].view(torch.uint8)
            ), f"{attr}: expert {expert} did not land on pool row {row}"


def test_presplit_is_a_noop_without_the_offload_flag(monkeypatch):
    monkeypatch.delenv(FRACTION_ENV, raising=False)
    layer = _block_fp8_layer(with_bias=True)
    ptrs = {a: getattr(layer, a).data_ptr() for a in _staged_attrs(layer)}
    shapes = {a: getattr(layer, a).shape for a in _staged_attrs(layer)}

    presplit_expert_offload_after_repack(layer)

    assert not hasattr(layer, "_moe_offload_presplit")
    for attr, ptr in ptrs.items():
        assert getattr(layer, attr).data_ptr() == ptr, attr
        assert getattr(layer, attr).shape == shapes[attr], attr


# --------------------------------------------------------------------------
# Fp8MoEMethod wiring
# --------------------------------------------------------------------------


def _fp8_method(monkeypatch, block_quant=True):
    """An Fp8MoEMethod with just the fields create_weights /
    process_weights_after_loading read, and no distributed group behind it."""
    from sglang.srt.layers.quantization import fp8 as fp8_mod

    class _Cfg:
        is_checkpoint_fp8_serialized = True
        weight_block_size = [BLOCK, BLOCK] if block_quant else None
        activation_scheme = "dynamic"
        is_fp4_experts = False
        dequant_fp4_to_fp8 = False

    method = fp8_mod.Fp8MoEMethod.__new__(fp8_mod.Fp8MoEMethod)
    method.quant_config = _Cfg()
    method.use_mxfp8 = False
    method.block_quant = block_quant
    method.convert_mxfp8_to_block = False
    method.weight_block_size = _Cfg.weight_block_size
    method.is_fp4_expert = False
    method.dequant_fp4_to_fp8 = False
    method.with_bias = False
    method.use_marlin = False

    monkeypatch.setattr(
        fp8_mod, "get_parallel", lambda: type("_P", (), {"tp_size": 1})()
    )
    return method


def _create_weights(method, num_experts=E):
    layer = torch.nn.Module()
    layer.num_local_experts = num_experts
    method.create_weights(
        layer,
        num_experts=num_experts,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTER,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    return layer


def test_create_weights_puts_the_expert_stack_on_the_host_when_offloading(
    monkeypatch, fraction_025
):
    method = _fp8_method(monkeypatch)
    layer = _create_weights(method)
    assert layer.w13_weight.device.type == "cpu"
    assert layer.w2_weight.device.type == "cpu"
    # shapes are the stock ones -- only the residence changed
    assert layer.w13_weight.shape == (E, 2 * INTER, HIDDEN)
    assert layer.w2_weight.shape == (E, HIDDEN, INTER)


def test_create_weights_default_path_uses_the_default_device(monkeypatch):
    monkeypatch.delenv(FRACTION_ENV, raising=False)
    method = _fp8_method(monkeypatch)
    reference = torch.empty(1)
    layer = _create_weights(method)
    for attr in ("w13_weight", "w2_weight", "w13_weight_scale_inv"):
        assert getattr(layer, attr).device == reference.device, attr


def test_process_weights_after_loading_ends_in_the_presplit(monkeypatch):
    """The call must be the LAST statement: the Marlin fallback and the fnuz /
    aiter branches replace w13/w2_weight, and the presplit has to stage the
    tensors the kernel ends up reading."""
    from sglang.srt.layers.moe import expert_offload as eo_mod

    method = _fp8_method(monkeypatch)
    layer = _block_fp8_layer()
    seen = {}

    def _record(lyr):
        seen["layer"] = lyr
        seen["w13_at_call"] = lyr.w13_weight.data_ptr()

    monkeypatch.setattr(eo_mod, "presplit_expert_offload_after_repack", _record)
    monkeypatch.setattr(
        method, "process_weights_after_loading_block_quant", lambda lyr: None
    )

    method.process_weights_after_loading(layer)

    assert seen["layer"] is layer
    assert seen["w13_at_call"] == layer.w13_weight.data_ptr()


def test_marlin_fallback_runs_before_the_presplit(monkeypatch):
    """Order gate for the sm<89 path: _prepare_marlin_moe rewrites the expert
    tensors, so it must be finished when the presplit stages them."""
    from sglang.srt.layers.moe import expert_offload as eo_mod

    method = _fp8_method(monkeypatch)
    method.use_marlin = True
    layer = _block_fp8_layer()
    order = []

    monkeypatch.setattr(
        method, "process_weights_after_loading_block_quant", lambda lyr: None
    )
    monkeypatch.setattr(
        method, "_prepare_marlin_moe", lambda lyr: order.append("marlin")
    )
    monkeypatch.setattr(
        eo_mod,
        "presplit_expert_offload_after_repack",
        lambda lyr: order.append("presplit"),
    )

    method.process_weights_after_loading(layer)
    assert order == ["marlin", "presplit"]


# --------------------------------------------------------------------------
# offload cache reads the presplit stash
# --------------------------------------------------------------------------


def test_cache_reads_the_full_expert_count_from_the_stash(fraction_025):
    """After the presplit, layer.w13_weight is a 0-row placeholder; the cache
    must size itself from _moe_offload_full_experts, not from the placeholder."""
    layer = _block_fp8_layer()
    layer.moe_runner_config = None
    presplit_expert_offload_after_repack(layer)
    layer.num_local_experts = 0  # what a shape-derived reader would see

    cache = MoEExpertOffloadCache(layer, 0.25)
    assert cache.num_local_experts == E
    assert cache.resident_count == resident_slot_count(E, 0.25)
    assert cache.planner.buffer_size == cache.resident_count + cache.scratch
