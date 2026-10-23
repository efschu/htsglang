"""tools/quant/autoround_gptq_to_ct.py: the repack must dequantize bit-identically.

GPTQ reference dequant (sym, auto_gptq v1, zero 8) against compressed-tensors'
own unpack of the repacked tensor; plus the refusals (asymmetric zeros,
act-order g_idx).
"""
import importlib.util
import os
import sys

import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
_SPEC = importlib.util.spec_from_file_location(
    "autoround_gptq_to_ct", os.path.join(_ROOT, "tools", "quant", "autoround_gptq_to_ct.py"))
conv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(conv)


def _gptq_pack(q: torch.Tensor) -> torch.Tensor:
    """q uint4 values [K, N] -> GPTQ qweight int32 [K/8, N] (nibble j = row 8i+j)."""
    k, n = q.shape
    q = q.to(torch.int64).reshape(k // 8, 8, n)
    word = torch.zeros(k // 8, n, dtype=torch.int64)
    for j in range(8):
        word |= q[:, j, :] << (4 * j)
    return (word - ((word >> 31) << 32)).to(torch.int32)  # wrap to signed int32


def _gptq_dequant(q: torch.Tensor, scales: torch.Tensor, group: int) -> torch.Tensor:
    k, n = q.shape
    s = scales.to(torch.float32).repeat_interleave(group, dim=0)
    return ((q.to(torch.float32) - 8.0) * s).t()  # [N, K]


def _make(k=256, n=128, group=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(0, 16, (k, n), generator=g)
    scales = (torch.rand(k // group, n, generator=g) * 0.02 + 0.001).to(torch.float16)
    qz = torch.full((k // group, n // 8), 0x77777777, dtype=torch.int32)
    t = {"x.qweight": _gptq_pack(q), "x.scales": scales, "x.qzeros": qz}
    return q, scales, t


def test_repack_is_transpose_and_bit_identical_dequant():
    ct_utils = pytest.importorskip("compressed_tensors.compressors.pack_quantized.helpers")
    q, scales, t = _make()
    out = conv.convert_group("x", t, 128, torch.float32)
    wp = out["x.weight_packed"]
    assert wp.shape == (128, 256 // 8) and wp.dtype == torch.int32
    assert torch.equal(wp, t["x.qweight"].t())
    assert out["x.weight_shape"].tolist() == [128, 256]
    # compressed-tensors' own unpack: signed int4 values v = q - 8
    unpack = getattr(ct_utils, "unpack_from_int32")
    v = unpack(wp, 4, torch.Size([128, 256]))
    assert torch.equal(v.to(torch.int64), (q.t() - 8).to(torch.int64))
    # and compressed-tensors' own packer produces the very same words
    assert torch.equal(ct_utils.pack_to_int32((q.t() - 8).to(torch.int8), 4), wp)
    deq_ct = v.to(torch.float32) * out["x.weight_scale"].repeat_interleave(128, dim=1)
    assert torch.equal(deq_ct, _gptq_dequant(q, scales, 128))


def test_scale_cast_matches_loader_cast():
    _, scales, t = _make(seed=1)
    out = conv.convert_group("x", t, 128, torch.bfloat16)
    # the GPTQ-Marlin loader copies fp16 checkpoint scales into a bf16 param: same cast
    assert torch.equal(out["x.weight_scale"], scales.t().to(torch.bfloat16))


def test_refuses_asymmetric_zero_points():
    _, _, t = _make()
    t["x.qzeros"][0, 0] = 0x77777776
    with pytest.raises(SystemExit, match="qzeros"):
        conv.convert_group("x", t, 128, torch.bfloat16)


def test_refuses_act_order_g_idx():
    _, _, t = _make()
    gi = torch.arange(256, dtype=torch.int32) // 128
    t["x.g_idx"] = gi.flip(0)
    with pytest.raises(SystemExit, match="g_idx"):
        conv.convert_group("x", t, 128, torch.bfloat16)
    t["x.g_idx"] = gi
    conv.convert_group("x", t, 128, torch.bfloat16)  # trivial g_idx accepted


def test_config_refuses_wrong_source():
    base = {"bits": 4, "group_size": 128, "sym": True, "packing_format": "auto_round:auto_gptq"}
    cfg = conv.ct_quant_config(base)
    w = cfg["config_groups"]["group_0"]["weights"]
    assert (w["num_bits"], w["group_size"], w["strategy"], w["symmetric"]) == (4, 128, "group", True)
    assert cfg["format"] == "pack-quantized" and "lm_head" in cfg["ignore"]
    for bad in ({"sym": False}, {"bits": 8}, {"packing_format": "auto_round:auto_awq"}, {"desc_act": True}):
        with pytest.raises(SystemExit):
            conv.ct_quant_config({**base, **bad})


def _write_src(tmp, vision_split: bool):
    """A tiny AutoRound-shaped checkpoint: one quantized linear, a bf16 weight and a
    vision tower split over two shards (as in the 27B AutoRound exports)."""
    import json
    from safetensors.torch import save_file
    _, _, t = _make()
    s1 = {"model.language_model.layers.0.mlp.down_proj." + k.split(".", 1)[1]: v for k, v in t.items()}
    s1["model.visual.blocks.0.attn.qkv.weight"] = torch.ones(4, 4, dtype=torch.bfloat16)
    s2 = {"model.visual.blocks.1.attn.qkv.weight": torch.full((4, 4), 2.0, dtype=torch.bfloat16),
          "lm_head.weight": torch.zeros(8, 4, dtype=torch.bfloat16)}
    s3 = {"model.visual.merger.linear_fc1.weight": torch.full((4, 4), 3.0, dtype=torch.bfloat16)}
    files = {"model-00001-of-00003.safetensors": s1, "model-00002-of-00003.safetensors": s2,
             "model-00003-of-00003.safetensors": s3}
    wmap = {}
    for fn, d in files.items():
        save_file(d, os.path.join(tmp, fn))
        wmap.update({k: fn for k in d})
    json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(tmp, "model.safetensors.index.json"), "w"))
    json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"],
               "quantization_config": {"bits": 4, "group_size": 128, "sym": True, "quant_method": "auto-round",
                                       "packing_format": "auto_round:auto_gptq"}},
              open(os.path.join(tmp, "config.json"), "w"))
    open(os.path.join(tmp, "tokenizer.json"), "w").write("{}")


def test_main_puts_the_tower_into_one_shard_the_vision_loader_accepts(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(_ROOT, "python"))
    vsl = pytest.importorskip("flliper.srt.planner.vision_stage_load")
    from safetensors import safe_open
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _write_src(str(src), True)
    # the source layout is exactly what the loader refuses
    with pytest.raises(vsl.VisionStageLoadRefused, match="spans"):
        vsl.find_tower_shard(str(src))
    monkeypatch.setattr(sys, "argv", ["x", str(src), str(dst)])
    assert conv.main() == 0
    shard = vsl.find_tower_shard(str(dst))
    assert os.path.basename(shard) == "model-visual.safetensors"
    ext = vsl.tower_extent(shard)
    assert ext.pieces == 3 and ext.byte_sum == 3 * 4 * 4 * 2
    assert ext.last_end - ext.first_offset == ext.byte_sum  # contiguous single extent
    # the shard that held only tower tensors is gone from the index and the disk
    assert not os.path.exists(dst / "model-00003-of-00003.safetensors")
    import json
    wmap = json.load(open(dst / "model.safetensors.index.json"))["weight_map"]
    assert set(wmap.values()) == {"model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
                                  "model-visual.safetensors"}
    with safe_open(str(dst / "model-visual.safetensors"), "pt") as f:
        assert torch.equal(f.get_tensor("model.visual.blocks.1.attn.qkv.weight"),
                           torch.full((4, 4), 2.0, dtype=torch.bfloat16))
    assert "model.language_model.layers.0.mlp.down_proj.weight_packed" in wmap
    assert (dst / "tokenizer.json").exists()
    cfg = json.load(open(dst / "config.json"))
    assert cfg["quantization_config"]["quant_method"] == "compressed-tensors"
