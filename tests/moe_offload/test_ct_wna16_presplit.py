# SPDX-License-Identifier: Apache-2.0
"""Host-side gate for the compressed-tensors WNA16 MoE load-time offload (WP2,
Qwen3.8-Flash-Next).

The AWQ/GPTQ INT4 checkpoints of Qwen3.8-Flash-Next are written by
llm-compressor in the compressed-tensors ``pack-quantized`` format, so the
FusedMoE layer takes ``CompressedTensorsWNA16MoE`` -- which, unlike
``awq_moe`` / ``gptq_moe`` / ``fp8``, had neither half of the per-expert
offload: ``create_weights`` committed the whole ``[E, ...]`` stack on the
card (72 GB of experts on a 32 GB card OOM at load, the #256 shape) and
``process_weights_after_loading`` never called the presplit. These tests pin
both halves without loading a model or running a marlin kernel:

  * ``create_weights`` puts the expert-major tensors (packed weights, group
    scales and -- asymmetric checkpoints only -- zero points) on the host when
    the offload fraction is set, and on the default device when it is not;
  * the compressed-tensors names are staged by the offload cache;
  * ``process_weights_after_loading`` runs the repack first and ends in the
    presplit;
  * the presplit stages every tensor of one expert on the SAME spill row.

Run:
  PYTHONPATH=python python -m pytest tests/moe_offload/test_ct_wna16_presplit.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    MoEExpertOffloadCache,
    presplit_expert_offload_after_repack,
    reset_expert_offload_release,
    resident_slot_count,
)

FRACTION_ENV = "SGLANG_MOE_RESIDENT_EXPERT_FRACTION"

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the pinned spill pool needs a CUDA context (host tensors only)",
)

# Qwen3.8-Flash-Next geometry scaled down: 512 experts x (2560 -> 640), AWQ
# group 32 asymmetric, 4-bit int32 packing (pack factor 8).
E = 32
HIDDEN = 256
INTER = 128
GROUP = 32
PACK = 8

CT_ATTRS_ASYM = (
    "w13_weight_packed",
    "w2_weight_packed",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_zero_point",
    "w2_weight_zero_point",
)
CT_ATTRS_SYM = CT_ATTRS_ASYM[:4]


class _StubCTConfig:
    quant_format = "pack-quantized"


def _scheme(sym: bool):
    """A CompressedTensorsWNA16MoE with just the fields create_weights reads
    (``__new__`` skips the constructor's config-group validation)."""
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
        CompressedTensorsWNA16MoE,
    )

    scheme = CompressedTensorsWNA16MoE.__new__(CompressedTensorsWNA16MoE)
    scheme.quant_config = _StubCTConfig()
    scheme.num_bits = 4
    scheme.packed_factor = PACK
    scheme.strategy = "group"
    scheme.group_size = GROUP
    scheme.actorder = None
    scheme.sym = sym
    scheme.num_gpu_experts = -1
    return scheme


def _create_weights(scheme, num_experts=E, ambient="meta"):
    """Run create_weights under a non-CPU ambient device (see the AWQ twin for
    why ``meta``: it makes the host/default distinction observable without a
    card)."""
    layer = torch.nn.Module()
    layer.num_local_experts = num_experts
    layer.moe_tp_size = 1
    with torch.device(ambient):
        scheme.create_weights(
            layer,
            num_experts=num_experts,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTER,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **k: None,
        )
    return layer


def _repacked_ct_layer(num_experts=E):
    """A stub FusedMoE layer with post-repack marlin tensors under the
    compressed-tensors names. Rows carry the expert id so a misrouted row is
    visible as a value."""
    layer = torch.nn.Module()
    layer.num_local_experts = num_experts

    def _p(*shape, dtype=torch.bfloat16):
        t = torch.empty(shape, dtype=dtype)
        for e in range(num_experts):
            t[e] = e
        return torch.nn.Parameter(t, requires_grad=False)

    layer.register_parameter(
        "w13_weight_packed",
        _p(num_experts, HIDDEN // 16, 2 * INTER * 2, dtype=torch.int32),
    )
    layer.register_parameter(
        "w2_weight_packed", _p(num_experts, INTER // 16, HIDDEN * 2, dtype=torch.int32)
    )
    layer.register_parameter(
        "w13_weight_scale", _p(num_experts, HIDDEN // GROUP, 2 * INTER)
    )
    layer.register_parameter("w2_weight_scale", _p(num_experts, INTER // GROUP, HIDDEN))
    layer.register_parameter(
        "w13_weight_zero_point",
        _p(num_experts, HIDDEN // GROUP, 2 * INTER // PACK, dtype=torch.int32),
    )
    layer.register_parameter(
        "w2_weight_zero_point",
        _p(num_experts, INTER // GROUP, HIDDEN // PACK, dtype=torch.int32),
    )
    # post-repack g_idx of a non-actorder checkpoint: [E, 0], never staged
    layer.register_parameter(
        "w13_weight_g_idx",
        torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32), requires_grad=False
        ),
    )
    return layer


@pytest.fixture
def fraction_025(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "0.25")
    reset_expert_offload_release()
    yield 0.25
    reset_expert_offload_release()


# --------------------------------------------------------------------------
# create_weights residence
# --------------------------------------------------------------------------


def test_create_weights_puts_the_expert_stack_on_the_host_when_offloading(
    fraction_025,
):
    layer = _create_weights(_scheme(sym=False))
    for attr in CT_ATTRS_ASYM:
        assert getattr(layer, attr).device.type == "cpu", attr
    # shapes are the stock ones -- only the residence changed
    assert layer.w13_weight_packed.shape == (E, HIDDEN // PACK, 2 * INTER)
    assert layer.w2_weight_packed.shape == (E, INTER // PACK, HIDDEN)
    assert layer.w13_weight_scale.shape == (E, HIDDEN // GROUP, 2 * INTER)
    assert layer.w2_weight_scale.shape == (E, INTER // GROUP, HIDDEN)
    assert layer.w13_weight_zero_point.shape == (E, HIDDEN // GROUP, 2 * INTER // PACK)


def test_symmetric_checkpoint_has_no_zero_points_and_still_offloads(fraction_025):
    """Minachist-style INT4 g128 symmetric experts: no zero-point tensors at
    all; the four remaining expert-major tensors go to the host."""
    layer = _create_weights(_scheme(sym=True))
    for attr in CT_ATTRS_SYM:
        assert getattr(layer, attr).device.type == "cpu", attr
    assert not hasattr(layer, "w13_weight_zero_point")


def test_create_weights_default_path_uses_the_default_device(monkeypatch):
    monkeypatch.delenv(FRACTION_ENV, raising=False)
    layer = _create_weights(_scheme(sym=False))
    for attr in CT_ATTRS_ASYM:
        assert getattr(layer, attr).device.type == "meta", attr


def test_create_weights_at_fraction_one_is_the_stock_path(monkeypatch):
    monkeypatch.setenv(FRACTION_ENV, "1.0")
    layer = _create_weights(_scheme(sym=False))
    for attr in CT_ATTRS_ASYM:
        assert getattr(layer, attr).device.type == "meta", attr


def test_weight_attrs_survive_the_device_change(fraction_025):
    layer = _create_weights(_scheme(sym=False))
    for attr in CT_ATTRS_ASYM:
        p = getattr(layer, attr)
        assert getattr(p, "is_transposed", None) is True, attr
        assert getattr(p, "quant_method", None) == "group", attr
        assert getattr(p, "weight_loader", None) is not None, attr


def test_compressed_tensors_names_are_staged_by_the_cache():
    for attr in CT_ATTRS_ASYM:
        assert attr in MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS, attr


def test_scheme_is_not_refused_by_the_quant_guard():
    """CompressedTensorsWNA16MoE now has both offload halves; the fail-fast
    guard must let it through (the NVFP4 scheme that shares the
    'w13_weight_packed' name stays refused)."""
    from sglang.srt.layers.moe.expert_offload import (
        assert_expert_offload_quant_supported,
    )

    assert_expert_offload_quant_supported(None, 0, scheme=_scheme(sym=False))

    class CompressedTensorsW4A4Nvfp4MoE:  # same class name as the real scheme
        pass

    with pytest.raises(RuntimeError):
        assert_expert_offload_quant_supported(
            None, 0, scheme=CompressedTensorsW4A4Nvfp4MoE()
        )


# --------------------------------------------------------------------------
# process_weights_after_loading wiring
# --------------------------------------------------------------------------


def test_repack_runs_before_the_presplit(monkeypatch):
    """Order gate: the marlin repack replaces every expert tensor in place, so
    it has to be finished (``is_marlin_converted``) when the presplit stages
    them. The repack kernels are stubbed with identity transforms."""
    from sglang.srt.layers.moe import expert_offload as eo_mod
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_wNa16_moe as ct_mod,
    )

    order = []
    monkeypatch.setattr(ct_mod, "gptq_marlin_moe_repack", lambda w, *a, **k: w.clone())
    monkeypatch.setattr(
        ct_mod, "marlin_moe_permute_scales", lambda s, *a, **k: s.clone()
    )
    monkeypatch.setattr(
        ct_mod, "moe_awq_to_marlin_zero_points", lambda z, *a, **k: z.clone()
    )
    monkeypatch.setattr(ct_mod, "marlin_make_workspace", lambda *a, **k: None)

    def _presplit(layer):
        assert layer.is_marlin_converted is True
        order.append("presplit")

    monkeypatch.setattr(eo_mod, "presplit_expert_offload_after_repack", _presplit)

    scheme = _scheme(sym=False)
    monkeypatch.delenv(FRACTION_ENV, raising=False)
    layer = _create_weights(scheme, ambient="cpu")
    # the loader's device_loading_context normally provides these
    for attr in ("w2_weight_g_idx", "w13_g_idx_sort_indices", "w2_g_idx_sort_indices"):
        assert hasattr(layer, attr), attr
    scheme.process_weights_after_loading(layer)
    assert order == ["presplit"]
    assert layer.is_marlin_converted is True
    # a second call is the documented no-op (no double repack, no second presplit)
    scheme.process_weights_after_loading(layer)
    assert order == ["presplit"]


# --------------------------------------------------------------------------
# presplit mechanics on the compressed-tensors names
# --------------------------------------------------------------------------


@needs_cuda
def test_presplit_stages_all_six_ct_tensors(fraction_025):
    layer = _repacked_ct_layer()
    before = {a: getattr(layer, a).data.clone() for a in CT_ATTRS_ASYM}

    presplit_expert_offload_after_repack(layer)

    R = resident_slot_count(E, 0.25)
    presplit = layer._moe_offload_presplit
    assert set(presplit) == set(CT_ATTRS_ASYM)
    assert layer._moe_offload_full_experts == E
    # the empty g_idx stays where it is: it is not an expert-major payload
    assert layer.w13_weight_g_idx.shape == (E, 0)

    for attr, (buf, spill) in presplit.items():
        assert spill.shape[0] == E - R, attr
        assert spill.is_pinned(), attr
        assert torch.equal(
            buf[:R].view(torch.uint8), before[attr][:R].view(torch.uint8)
        ), attr
        assert getattr(layer, attr).shape[0] == 0, attr


@needs_cuda
def test_zero_points_share_the_expert_row_with_their_weight(fraction_025):
    layer = _repacked_ct_layer()
    before = {a: getattr(layer, a).data.clone() for a in CT_ATTRS_ASYM}

    presplit_expert_offload_after_repack(layer)

    R = resident_slot_count(E, 0.25)
    presplit = layer._moe_offload_presplit
    for expert in range(R, E):
        row = expert - R
        for attr, (_buf, spill) in presplit.items():
            assert torch.equal(
                spill[row].view(torch.uint8), before[attr][expert].view(torch.uint8)
            ), f"{attr}: expert {expert} did not land on pool row {row}"
