"""H68: NVFP4 FORMAT KEY -- which NVFP4 a checkpoint is, from its two JSON
files alone (``weg2/nvfp4_ckpt_plan.detect_nvfp4_format``).

"NVFP4" names two incompatible families -- ModelOpt (nvidia/*, RadixArk/*:
``quant_algo``, ``weight_scale_2``/``input_scale``) and compressed-tensors
(unsloth "Dynamic": ``nvfp4-pack-quantized``/``mixed-precision``) -- and a
ModelOpt MIXED_PRECISION export can mix NVFP4, block FP8 and per-tensor FP8
per module. The key names all of it in one greppable string; the two
quantization files of one export are compared and their disagreement named
(nvidia/Qwen3.8-Flash-Next-NVFP4 @ fc694b54fb: FP8_PB_WO vs FP8_BLOCK_SCALES).

Hermetic: the quantization sections are written into a temp dir; no torch
tensor, no GPU.
"""

import json
import os
import tempfile

import pytest

try:
    from sglang.srt.weg2 import nvfp4_ckpt_plan as P
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def modelopt_mixed_config(n_layers=4, mtp_algo="FP8_PB_WO", with_ple=True):
    ql = {
        f"model.language_model.layers.{i}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16}
        for i in range(n_layers)
    }
    if mtp_algo:
        ql["mtp.layers.0.mlp.experts"] = {"quant_algo": mtp_algo, "group_size": 128}
    if with_ple:
        ql["model.language_model.layers.1.ple.ple_embedding.ngram_embedding"] = {"quant_algo": "FP8"}
    ignore = ["lm_head", "model.language_model.embed_tokens", "model.visual*"]
    for i in range(n_layers):
        ignore += [f"model.language_model.layers.{i}.mlp.shared_expert*",
                   f"model.language_model.layers.{i}.mlp.gate"]
    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "producer": {"name": "modelopt", "version": "0.46.0"},
        "ignore": ignore,
        "quantized_layers": ql,
    }


def write_ckpt(tmp, qc=None, hf=None):
    cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"]}
    if qc is not None:
        cfg["quantization_config"] = qc
    with open(os.path.join(tmp, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    if hf is not None:
        with open(os.path.join(tmp, "hf_quant_config.json"), "w") as fh:
            json.dump(hf, fh)
    return tmp


def test_modelopt_mixed_key_names_every_scheme():
    with tempfile.TemporaryDirectory() as tmp:
        qc = modelopt_mixed_config()
        hf = {"producer": qc["producer"], "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "exclude_modules": qc["ignore"],
            "quantized_layers": dict(qc["quantized_layers"], **{
                "mtp.layers.0.mlp.experts": {"quant_algo": "FP8_BLOCK_SCALES", "group_size": 128}
            }),
        }}
        key = P.detect_nvfp4_format(write_ckpt(tmp, qc, hf))
    assert key is not None
    assert key.flavour == P.FLAVOUR_MODELOPT
    assert key.key == (
        "modelopt:MIXED_PRECISION|draft.moe.routed=fp8blk128;"
        "moe.routed=nvfp4a4g16;ple.table=fp8pt;*=bf16"
    )
    assert key.group_size == 16 and key.activation_bits == 4
    assert key.draft_algo == P.ALGO_FP8_BLOCK and key.ple_algo == P.ALGO_FP8
    # the two files disagree on the SPELLING only -> named, verdict "aliases"
    assert len(key.inconsistencies) == 1
    assert "FP8_PB_WO" in key.inconsistencies[0]
    assert "FP8_BLOCK_SCALES" in key.inconsistencies[0]
    assert "aliases -> FP8_BLOCK g128" in key.inconsistencies[0]
    assert key.line().startswith("NVFP4-FORMAT key=modelopt:MIXED_PRECISION|")


def test_real_conflict_is_called_conflict():
    with tempfile.TemporaryDirectory() as tmp:
        qc = modelopt_mixed_config()
        hf_layers = dict(qc["quantized_layers"])
        hf_layers["model.language_model.layers.0.mlp.experts"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        key = P.detect_nvfp4_format(
            write_ckpt(tmp, qc, {"quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": hf_layers}})
        )
    assert any("CONFLICT" in i for i in key.inconsistencies)


def test_hf_quant_config_only_export_is_read():
    """ModelOpt legacy: no quantization_config in config.json at all."""
    with tempfile.TemporaryDirectory() as tmp:
        hf = {"producer": {"name": "modelopt", "version": "0.3"},
              "quantization": {"quant_algo": "W4A16_NVFP4", "group_size": 16,
                               "exclude_modules": ["lm_head"]}}
        key = P.detect_nvfp4_format(write_ckpt(tmp, None, hf))
    assert key.key == "modelopt:W4A16_NVFP4|*=nvfp4a16g16"
    assert key.activation_bits == 16


def test_radixark_27b_shape_fp8_linear_plus_nvfp4_mlp():
    """The 27B line's format (2): MIXED_PRECISION with static per-tensor FP8
    on GDN/attention projections and NVFP4 W4A4 on every MLP + lm_head."""
    ql = {}
    for i in range(2):
        ql[f"model.language_model.layers.{i}.linear_attn.in_proj_qkv"] = {"quant_algo": "FP8"}
        ql[f"model.language_model.layers.{i}.mlp.gate_proj"] = {"quant_algo": "NVFP4", "group_size": 16}
    ql["lm_head"] = {"quant_algo": "NVFP4", "group_size": 16}
    qc = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": ql,
          "kv_cache_quant_algo": "FP8"}
    with tempfile.TemporaryDirectory() as tmp:
        key = P.detect_nvfp4_format(write_ckpt(tmp, qc))
    assert key.key == (
        "modelopt:MIXED_PRECISION|attn.gdn=fp8pt;mlp.dense=nvfp4a4g16;"
        "vocab.lm_head=nvfp4a4g16;*=bf16"
    )
    assert key.kv_cache_quant_algo == "FP8"
    assert key.draft_algo is None


def test_compressed_tensors_nvfp4_is_the_other_flavour():
    """unsloth "Dynamic": nvfp4 MLP + fp8 rest, compressed-tensors."""
    qc = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {"targets": ["re:.*mlp.*"],
                        "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                        "input_activations": {"num_bits": 4, "type": "float", "group_size": 16}},
            "group_1": {"targets": ["re:.*self_attn.*"],
                        "weights": {"num_bits": 8, "type": "float"},
                        "input_activations": {"num_bits": 8, "type": "float"}},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        key = P.detect_nvfp4_format(write_ckpt(tmp, qc))
    assert key.flavour == P.FLAVOUR_COMPRESSED_TENSORS
    assert key.key.startswith("compressed-tensors:mixed-precision|")
    assert "nvfp4a4g16" in key.key and "fp8pt" in key.key


def test_int_compressed_tensors_is_not_nvfp4():
    """The NF production shape (INT6/INT4/INT8 groups) carries no NVFP4."""
    def grp(bits, gs, targets):
        return {"targets": targets, "weights": {"num_bits": bits, "group_size": gs, "type": "int"}}
    qc = {"quant_method": "compressed-tensors", "format": "pack-quantized",
          "config_groups": {"g0": grp(6, 64, ["model.language_model.layers.0.linear_attn.in_proj_z"]),
                            "g1": grp(4, 128, ["re:.*mlp\\.experts.*"]),
                            "g2": grp(8, 128, ["re:.*lm_head"])}}
    with tempfile.TemporaryDirectory() as tmp:
        assert P.detect_nvfp4_format(write_ckpt(tmp, qc)) is None


@pytest.mark.parametrize(
    "raw,norm",
    [("FP8_PB_WO", "FP8_BLOCK"), ("fp8_block_scales", "FP8_BLOCK"), ("NVFP4", "NVFP4"),
     ("W4A16_NVFP4", "W4A16_NVFP4"), ("FP8", "FP8"), ("", "BF16"), ("NVFP5", "UNKNOWN:NVFP5")],
)
def test_normalize_algo_never_guesses(raw, norm):
    assert P.normalize_algo(raw) == norm


@pytest.mark.parametrize(
    "module,role",
    [
        ("model.language_model.layers.3.mlp.experts", "moe.routed"),
        ("model.language_model.layers.3.mlp.experts.17.gate_proj", "moe.routed"),
        ("model.language_model.layers.3.mlp.shared_expert.up_proj", "moe.shared"),
        ("model.language_model.layers.3.mlp.shared_expert_gate", "moe.shared"),
        ("model.language_model.layers.3.mlp.gate", "moe.router"),
        ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding", "ple.table"),
        ("model.language_model.layers.1.ple.key_proj", "ple"),
        ("model.language_model.layers.3.self_attn.q_proj", "attn.full"),
        ("model.language_model.layers.2.linear_attn.in_proj_qkv", "attn.gdn"),
        ("model.language_model.layers.2.attn_hyper_connection.input_mix_weight_up", "hyper"),
        ("model.language_model.layers.2.mlp.gate_proj", "mlp.dense"),
        ("mtp.layers.0.mlp.experts.3.down_proj", "draft.moe.routed"),
        ("mtp.fc_hidden", "draft.other"),
        ("lm_head", "vocab.lm_head"),
        ("re:(model|language_model)\\..*\\.mlp\\.experts\\..*", "moe.routed"),
    ],
)
def test_roles(module, role):
    assert P.module_role(module) == role
