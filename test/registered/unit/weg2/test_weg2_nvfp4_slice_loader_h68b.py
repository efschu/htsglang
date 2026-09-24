"""H68b (NF line): does the loader tolerate a SHORTENED layer count?

The slice smoke (4 layers, one card, no offload) needs sglang to read a
4-layer view of nvidia/Qwen3.8-Flash-Next-NVFP4 and to read ONLY layers 0-3.
``--json-model-override-args`` cannot do it (``PretrainedConfig.update``
replaces the nested ``text_config``), so ``weg2/tools/nvfp4_slice_dir`` builds a
symlink view with a rewritten config.json. Pinned here against the REAL
download (opt-in, ``H68_NVFP4_CKPT=<dir>``; metadata only):

* sglang's config path reads the view: 4 text layers, 4 layer types, the PLE
  layer inside, quant method modelopt_mixed;
* the REAL ``Qwen4ExpForConditionalGeneration.weight_name_needed`` (stand-in
  ``self``: layers [0, 4), PLE checkpoint backend) reads 24,679 of 299,545
  tensors (8.28 GiB), hands the 128 PLE shards over as meta, skips the rest;
  every layer-indexed tensor it reads belongs to layers 0-3;
* the routed experts it reads resolve to NVFP4.

The hermetic half (a tiny fake checkpoint) runs everywhere.
"""

import json
import os
import re
import types

import pytest

try:
    from sglang.srt.weg2.tools import nvfp4_slice_dir as S
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

CKPT = os.environ.get("H68_NVFP4_CKPT", "")


def test_slice_dir_is_symlinks_plus_one_config(tmp_path):
    src = tmp_path / "ckpt"
    src.mkdir()
    (src / "model-00001-of-00001.safetensors").write_bytes(b"x")
    (src / "tokenizer.json").write_text("{}")
    cfg = {"text_config": {"num_hidden_layers": 8, "layer_types": ["linear_attention"] * 7 + ["full_attention"]}}
    (src / "config.json").write_text(json.dumps(cfg))
    dst = tmp_path / "slice"
    types_ = S.make_slice_dir(str(src), str(dst), 4)
    assert types_ == ["linear_attention"] * 4
    assert os.path.islink(dst / "model-00001-of-00001.safetensors")
    assert not os.path.islink(dst / "config.json")
    out = json.loads((dst / "config.json").read_text())
    assert out["text_config"]["num_hidden_layers"] == 4
    assert out["_h68b_slice"]["full_layers"] == 8
    S.make_slice_dir(str(src), str(dst), 4)  # idempotent
    with pytest.raises(ValueError):
        S.make_slice_dir(str(src), str(tmp_path / "s1"), 1)  # the PLE needs layer 1


@pytest.mark.skipif(not CKPT or not os.path.isdir(CKPT), reason="H68_NVFP4_CKPT not set (opt-in)")
def test_the_loader_reads_only_the_slice(tmp_path):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration as Q
    from sglang.srt.utils.hf_transformers.config import get_config
    from sglang.srt.weg2.nvfp4_ckpt_plan import _read_header

    sl = str(tmp_path / "slice4")
    S.make_slice_dir(CKPT, sl, 4)
    cfg = get_config(sl, trust_remote_code=True, model_override_args={"language_model_only": True})
    tc = cfg.text_config
    assert tc.num_hidden_layers == 4 and len(tc.layer_types) == 4
    assert tc.short_conv_layer_ids == [1]  # the PLE layer is inside the slice
    q = ModelConfig._parse_quant_hf_config(types.SimpleNamespace(
        hf_config=cfg, model_path=sl,
        _parse_modelopt_quant_config=lambda d: ModelConfig._parse_modelopt_quant_config(None, d)))
    assert q["quant_method"] == "modelopt_mixed"

    fake = types.SimpleNamespace(
        language_model_only=True,
        model=types.SimpleNamespace(start_layer=0, end_layer=4),
        config=types.SimpleNamespace(num_hidden_layers=4),
        _EXPERT_ID_RE=Q._EXPERT_ID_RE,
        _ple_ngram_embedding=lambda: types.SimpleNamespace(_ckpt_backend=True),
        _owned_expert_range=lambda: None,
        _num_routed_experts_for_form_a=lambda: None,
    )
    idx = json.load(open(os.path.join(sl, "model.safetensors.index.json")))["weight_map"]
    size = {}
    for f in sorted(set(idx.values())):
        hdr, _n = _read_header(os.path.join(sl, f), True)
        size.update({k: m["data_offsets"][1] - m["data_offsets"][0] for k, m in hdr.items()})
    read = meta = 0
    read_bytes = 0
    layers = set()
    experts = set()
    for name in idx:
        v = Q.weight_name_needed(fake, name)
        if v == "meta":
            meta += 1
        elif v:
            read += 1
            read_bytes += size[name]
            m = re.search(r"\.layers\.(\d+)\.", name)
            if m:
                layers.add(int(m.group(1)))
            if ".experts." in name:
                experts.add(re.sub(r"\.experts\.\d+\..*$", ".experts", name.replace("model.language_model.", "model.")))
    assert (read, meta) == (24679, 128)
    assert round(read_bytes / 2**30, 2) == 8.28
    assert layers == {0, 1, 2, 3}
    mp = ModelOptMixedPrecisionConfig.from_config(
        json.load(open(os.path.join(sl, "config.json")))["quantization_config"])
    assert {mp.resolve_quant_algo(p) for p in experts} == {"NVFP4"} and len(experts) == 4
