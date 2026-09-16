"""Minachist INT4/INT6 mixed Qwen3.8-Flash-Next: ``embed_tokens`` and ``lm_head``
are pack-quantized INT8 g128 (config group_3), the PLE/indexer projections
too. ``embed_tokens`` is gathered and dequantized per row
(CompressedTensorsPackedEmbeddingMethod); ``lm_head`` takes the Marlin
linear scheme and the logits gate must accept a head that carries
``weight_packed`` instead of ``.weight``."""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

from sglang.srt.layers.quantization.compressed_tensors.ct_embedding import (
    CompressedTensorsPackedEmbeddingMethod,
)

ROWS, DIM, GROUP = 12, 256, 128


def _quantized_table(seed=3):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(ROWS, DIM, generator=g)
    scale = w.abs().view(ROWS, DIM // GROUP, GROUP).amax(-1) / 127.0  # [rows, groups]
    q = torch.round(w.view(ROWS, DIM // GROUP, GROUP) / scale.unsqueeze(-1)).clamp(-128, 127)
    q = q.view(ROWS, DIM).to(torch.int8)
    packed = pack_to_int32(q, 8, packed_dim=1)  # [rows, DIM/4], offset 128
    ref = (q.view(ROWS, DIM // GROUP, GROUP).float() * scale.unsqueeze(-1)).view(ROWS, DIM)
    return packed, scale.to(torch.float16), ref


def _layer(packed, scale):
    m = CompressedTensorsPackedEmbeddingMethod(num_bits=8, group_size=GROUP, params_dtype=torch.float32)
    layer = torch.nn.Module()
    m.create_weights(layer, DIM, [ROWS], DIM, ROWS, params_dtype=torch.float32)
    assert layer.weight_packed.shape == (ROWS, DIM // 4)
    assert layer.weight_scale.shape == (ROWS, DIM // GROUP)
    layer.weight_packed.data.copy_(packed)
    layer.weight_scale.data.copy_(scale)
    return m, layer


def test_gather_dequantizes_exactly_the_rows_it_looked_up():
    packed, scale, ref = _quantized_table()
    m, layer = _layer(packed, scale)
    ids = torch.tensor([0, 5, 11, 5])
    out = m.embedding(layer, ids)
    assert out.shape == (4, DIM)
    assert torch.allclose(out, ref[ids].to(torch.float16).float(), atol=1e-2, rtol=1e-2)


def test_dense_apply_matches_the_gather():
    packed, scale, ref = _quantized_table()
    m, layer = _layer(packed, scale)
    x = torch.randn(3, DIM)
    logits = m.apply(layer, x)
    assert torch.allclose(logits, x @ ref.to(torch.float16).float().t(), atol=1e-1, rtol=1e-2)


def test_vocab_loader_attrs_shard_rows_not_the_packed_axis():
    packed, scale, _ = _quantized_table()
    _, layer = _layer(packed, scale)
    assert layer.weight_packed.output_dim == 0 and layer.weight_packed.packed_dim == 1
    assert layer.weight_scale.output_dim == 0
    assert not hasattr(layer.weight_shape, "output_dim")


def test_logits_gate_accepts_a_packed_head():
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    head = SimpleNamespace(weight_packed=torch.zeros(2, 2, dtype=torch.int32))
    method = SimpleNamespace(apply=lambda *a, **k: None)
    assert should_apply_lm_head_quant_method(head, method) is True
    dense = SimpleNamespace(weight=torch.zeros(2, 2))
    assert should_apply_lm_head_quant_method(dense, None) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="scheme selection probes the device capability")
def test_scheme_selection_by_explicit_target_names():
    """AutoRound lists module NAMES as targets; a 6-bit symmetric group must
    resolve to the widened WNA16 scheme, an INT8 g128 target to the 8-bit one."""
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsWNA16,
    )

    cfg = CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "quantization_status": "compressed",
            "kv_cache_scheme": None,
            "ignore": ["model.norm"],
            "config_groups": {
                "group_0": {
                    "targets": ["model.layers.0.linear_attn.out_proj"],
                    "weights": {"num_bits": 6, "group_size": 64, "symmetric": True, "strategy": "group", "type": "int"},
                },
                "group_3": {
                    "targets": ["lm_head", "model.embed_tokens"],
                    "weights": {"num_bits": 8, "group_size": 128, "symmetric": True, "strategy": "group", "type": "int"},
                },
            },
        }
    )
    six = cfg.get_linear_scheme(layer=torch.nn.Module(), layer_name="model.layers.0.linear_attn.out_proj")
    assert isinstance(six, CompressedTensorsWNA16) and six.src_num_bits == 6
    assert six.quant_type.size_bits == 8
    head = cfg.get_linear_scheme(layer=torch.nn.Module(), layer_name="lm_head")
    assert isinstance(head, CompressedTensorsWNA16) and head.src_num_bits == 8
    assert head.group_size == 128


def test_explicit_target_wins_over_a_parent_entry_in_the_ignore_list(monkeypatch):
    """Minachist's AutoRound export lists the GDN parent module
    ``...layers.0.linear_attn`` in ``ignore`` while naming its child
    ``...linear_attn.in_proj_qkv`` as an INT6 target. The ignore match is a
    substring match, so the parent used to swallow the child and the layer
    came up unquantized with ``weight_packed`` tensors to load. An explicit
    target name must resolve to its scheme; a name only covered by the
    ignore list stays unquantized."""
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsWNA16,
    )

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 6), raising=False)
    parent = "model.language_model.layers.0.linear_attn"
    child = parent + ".in_proj_qkv"
    cfg = CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "quantization_status": "compressed",
            "kv_cache_scheme": None,
            "ignore": [parent, "model.language_model.layers.0.mlp.gate"],
            "config_groups": {
                "group_0": {
                    "targets": [child],
                    "weights": {"num_bits": 6, "group_size": 64, "symmetric": True, "strategy": "group", "type": "int"},
                },
            },
        }
    )
    six = cfg.get_linear_scheme(layer=torch.nn.Module(), layer_name=child)
    assert isinstance(six, CompressedTensorsWNA16) and six.src_num_bits == 6 and six.group_size == 64
    assert cfg.get_linear_scheme(layer=torch.nn.Module(), layer_name=parent + ".in_proj_b") is None
    assert cfg.get_linear_scheme(layer=torch.nn.Module(), layer_name="model.language_model.layers.0.mlp.gate") is None


def test_fused_module_resolves_through_its_explicitly_targeted_shards(monkeypatch):
    """The GDN in-projection is one fused module ``in_proj_qkvz`` built from
    the checkpoint's ``in_proj_qkv`` + ``in_proj_z`` (both INT6 targets)
    while their parent ``linear_attn`` sits in the ignore list; the sibling
    ``in_proj_ba`` (shards in_proj_b/in_proj_a) is bf16 in the checkpoint
    and listed in ignore. Shards with different schemes are refused."""
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsWNA16,
    )

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 6), raising=False)
    la = "model.language_model.layers.0.linear_attn"
    six = {"num_bits": 6, "group_size": 64, "symmetric": True, "strategy": "group", "type": "int"}
    eight = {"num_bits": 8, "group_size": 64, "symmetric": True, "strategy": "group", "type": "int"}
    base = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "kv_cache_scheme": None,
        "packed_modules_mapping": {
            "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
            "in_proj_ba": ["in_proj_b", "in_proj_a"],
        },
        "ignore": [la, la + ".in_proj_b", la + ".in_proj_a"],
    }
    cfg = CompressedTensorsConfig.from_config(
        dict(base, config_groups={"group_0": {"targets": [la + ".in_proj_qkv", la + ".in_proj_z"], "weights": six}})
    )
    fused = cfg.get_linear_scheme(layer=torch.nn.Module(), layer_name=la + ".in_proj_qkvz")
    assert isinstance(fused, CompressedTensorsWNA16) and fused.src_num_bits == 6
    assert cfg.get_linear_scheme(layer=torch.nn.Module(), layer_name=la + ".in_proj_ba") is None

    mixed = CompressedTensorsConfig.from_config(
        dict(
            base,
            config_groups={
                "group_0": {"targets": [la + ".in_proj_qkv"], "weights": six},
                "group_1": {"targets": [la + ".in_proj_z"], "weights": eight},
            },
        )
    )
    with pytest.raises(ValueError, match="different quantization schemes"):
        mixed.get_linear_scheme(layer=torch.nn.Module(), layer_name=la + ".in_proj_qkvz")


def test_dequantize_pack_quantized_weight_roundtrips_int8_g64():
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
        dequantize_pack_quantized_weight,
    )

    out, inp, group = 6, 256, 64
    g = torch.Generator().manual_seed(11)
    q = torch.randint(-128, 128, (out, inp), generator=g, dtype=torch.int8)
    scale = (torch.rand(out, inp // group, generator=g) + 0.5) / 127
    packed = pack_to_int32(q, 8, packed_dim=1)
    assert packed.shape == (out, inp // 4)
    ref = (q.view(out, inp // group, group).float() * scale.unsqueeze(-1)).view(out, inp)
    got = dequantize_pack_quantized_weight(packed, scale.to(torch.float16), torch.Size((out, inp)))
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-2)
    with pytest.raises(ValueError, match="does not fit"):
        dequantize_pack_quantized_weight(packed, scale, torch.Size((out + 1, inp)))


def test_load_packed_hc_linear_widens_into_the_dense_parameter():
    """Packed and scale arrive as two separate tensors (any order); the
    weight_shape tensor is absorbed; a foreign name is not claimed."""
    from sglang.srt.models.qwen4_exp import load_packed_hc_linear

    out, inp, group = 6, 256, 64
    g = torch.Generator().manual_seed(5)
    q = torch.randint(-128, 128, (out, inp), generator=g, dtype=torch.int8)
    scale = (torch.rand(out, inp // group, generator=g) + 0.5) / 127
    packed = pack_to_int32(q, 8, packed_dim=1)
    ref = (q.view(out, inp // group, group).float() * scale.unsqueeze(-1)).view(out, inp)
    mod = "model.layers.3.attn_hyper_connection.input_mix_weight_down"
    param = torch.nn.Parameter(torch.zeros(out, inp, dtype=torch.bfloat16))
    params = {mod + ".weight": param}
    pending = {}
    assert load_packed_hc_linear(pending, mod + ".weight_shape", torch.tensor([out, inp]), params)
    assert load_packed_hc_linear(pending, mod + ".weight_scale", scale.to(torch.float16), params)
    assert pending and torch.count_nonzero(param) == 0  # waits for the payload
    assert load_packed_hc_linear(pending, mod + ".weight_packed", packed, params)
    assert not pending
    assert torch.allclose(param.float(), ref, atol=2e-2, rtol=2e-2)
    assert not load_packed_hc_linear(pending, "model.layers.3.linear_attn.out_proj.weight_packed", packed, params)
    assert not load_packed_hc_linear(pending, mod + ".weight", ref, params)  # a bf16 export loads the plain way
