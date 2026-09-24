"""H68: the planner's compute-format key (``uneven_perf.checkpoint_compute_format``,
ANALYSE_321 sec. 8.2) for a ModelOpt MIXED_PRECISION MoE export.

THE DEFECT (measured 24.09. on nvidia/Qwen3.8-Flash-Next-NVFP4 @ fc694b54fb):
``checkpoint_compute_format`` returned ``bf16`` with no families -- the MoE
family (routed experts NVFP4) was "mixed" because the draft head's FP8-block
experts and the excluded BF16 shared experts landed in it, was dropped, and
the checkpoint-wide key fell through to the router/hyper "mlp" evidence.

Hermetic: the quantization sections are written into a temp dir; no torch
tensor, no GPU. The compressed-tensors guard pins the NF production checkpoint
shape (Minachist: INT6/INT4/INT8 groups) to its old result -- A/B'd against
10e6aca0ce's uneven_perf.py, the real Minachist config.json included.
"""

import json
import os
import tempfile

import pytest

try:
    from sglang.srt import uneven_perf as U
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layer_types(n=8):
    return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n)]


def modelopt_mixed_config(n_layers=4, mtp_algo="FP8_PB_WO", with_ple=True):
    """ModelOpt MIXED_PRECISION in the shape of the nvidia Qwen3.8-Flash-Next
    export (routed experts NVFP4 g16, MTP experts block FP8, PLE table FP8,
    everything else excluded)."""
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
        attn = "self_attn*" if (i + 1) % 4 == 0 else "linear_attn*"
        ignore += [
            f"model.language_model.layers.{i}.{attn}",
            f"model.language_model.layers.{i}.mlp.gate",
            f"model.language_model.layers.{i}.mlp.shared_expert*",
            f"model.language_model.layers.{i}.mlp.shared_expert_gate",
            f"model.language_model.layers.{i}.mlp_hyper_connection*",
        ]
    return {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
            "ignore": ignore, "quantized_layers": ql}


def minachist_like_config():
    """compressed-tensors shape of the NF production checkpoint (group 0 INT6
    dense, group 1 INT4 routed experts, group 2/3 INT8, ignore list)."""
    def grp(bits, gs, targets):
        return {"format": "pack-quantized", "input_activations": None, "output_activations": None,
                "targets": targets,
                "weights": {"group_size": gs, "num_bits": bits, "strategy": "group",
                            "symmetric": True, "type": "int"}}
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": grp(6, 64, ["model.language_model.layers.0.linear_attn.in_proj_qkv",
                                   "model.language_model.layers.0.mlp.shared_expert.down_proj",
                                   "model.language_model.layers.3.self_attn.q_proj"]),
            "group_1": grp(4, 128, ["re:(model|language_model)\\..*\\.mlp\\.experts\\..*"]),
            "group_2": grp(8, 64, ["model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_up"]),
            "group_3": grp(8, 128, ["re:.*embed_tokens", "re:.*lm_head",
                                    "model.language_model.layers.3.self_attn.indexer.index_qk_proj"]),
        },
        "ignore": ["model.language_model.layers.0.linear_attn",
                   "model.language_model.layers.0.mlp.gate",
                   "model.language_model.layers.0.mlp.shared_expert_gate",
                   "model.language_model.layers.0.mlp_hyper_connection"],
    }


def write_ckpt(tmp, qc):
    cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"],
           "quantization_config": qc, "text_config": {"layer_types": _layer_types(4)}}
    with open(os.path.join(tmp, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    return tmp


def test_planner_key_names_the_moe_family_nvfp4():
    with tempfile.TemporaryDirectory() as tmp:
        write_ckpt(tmp, modelopt_mixed_config())
        fmt, desc = U.checkpoint_compute_format(tmp)
        fmt2, _desc2, fams = U.checkpoint_compute_format_families(tmp)
    assert fmt == fmt2 == "nvfp4_a4", desc
    assert fams[U.GEMM_FAMILY_MOE] == "nvfp4_a4"
    assert fams[U.GEMM_FAMILY_ATTN_GDN] == "bf16"
    assert fams[U.GEMM_FAMILY_VOCAB] == "bf16"


def test_draft_scheme_never_describes_the_target():
    """The MTP experts alone (fp8 block) must not turn the target's MoE into fp8."""
    qc = modelopt_mixed_config()
    qc["quantized_layers"] = {k: v for k, v in qc["quantized_layers"].items() if k.startswith("mtp.")}
    assert U._per_family_formats(qc).get(U.GEMM_FAMILY_MOE) != "fp8"


def test_bf16_routed_entry_keeps_moe_unresolved():
    """Narrowness: bf16 evidence that is NOT a shared expert keeps the family
    out, exactly like before."""
    qc = modelopt_mixed_config()
    qc["ignore"] = qc["ignore"] + ["model.language_model.layers.0.mlp.experts.7"]
    assert U.GEMM_FAMILY_MOE not in U._per_family_formats(qc)


def test_routed_experts_must_agree():
    qc = modelopt_mixed_config()
    qc["quantized_layers"]["model.language_model.layers.0.mlp.experts"] = {
        "quant_algo": "W4A16_NVFP4", "group_size": 16}
    assert U.GEMM_FAMILY_MOE not in U._per_family_formats(qc)


@pytest.mark.parametrize("name,draft", [("mtp.layers.0.mlp.experts", True),
                                        ("model.mtp.layers.0.fc", True),
                                        ("model.language_model.layers.0.mlp.experts", False),
                                        ("model.layers.3.self_attn.q_proj", False)])
def test_draft_namespace(name, draft):
    assert U._is_draft_module(name) is draft


def test_compressed_tensors_result_is_unchanged():
    """GUARD for the NF line: a compressed-tensors config never reaches the
    H68 resolution (it has no quantized_layers), so its families are exactly
    what the unmodified rule returned."""
    qc = minachist_like_config()
    assert U._per_family_formats(qc, layer_split={"gdn": 36, "full": 12}) == {
        "vocab": "int8_a16",
        "attn_gdn": "bf16",
    }
    assert U._per_family_formats(qc) == {"vocab": "int8_a16"}
    with tempfile.TemporaryDirectory() as tmp:
        write_ckpt(tmp, qc)
        fmt, _desc = U.checkpoint_compute_format(tmp)
    assert fmt == "compressed-tensors"
