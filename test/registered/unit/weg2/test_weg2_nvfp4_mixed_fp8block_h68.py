"""H68: ModelOpt MIXED_PRECISION learns 2-D block FP8, and the MTP head of a
MIXED_PRECISION export is built quantized when (and only when) the export
lists ``mtp.*`` layers.

Why (nvidia/Qwen3.8-Flash-Next-NVFP4 @ fc694b54fb): its MTP experts are
128x128 block FP8, spelled ``FP8_PB_WO`` in config.json and
``FP8_BLOCK_SCALES`` in hf_quant_config.json. Before this,
``ModelOptMixedPrecisionConfig.get_quant_method`` returned NO method for that
FusedMoE and ``_mtp_quant_config`` returned ``None`` for every
``modelopt_mixed`` checkpoint -- the head would have been built in bf16 over
E4M3 weights + ``weight_scale_inv``. Upstream sglang #39126 (cebca698e2,
"Not carried on this line" per b113aea441bf) and vLLM #55513 fix the same.

Hermetic, CPU only: the fp8 method classes are replaced by recorders (their
constructors read the MoE-runner globals), layers are bare instances.
"""

import types
from unittest import mock

import pytest

try:
    import torch

    from sglang.srt.layers.quantization import fp8 as fp8_mod
    from sglang.srt.layers.quantization import modelopt_quant as M
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def mixed_qc(mtp_algo="FP8_PB_WO", mtp_group=128, extra=None):
    ql = {
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": {"quant_algo": "FP8"},
    }
    if mtp_algo:
        ql["mtp.layers.0.mlp.experts"] = {"quant_algo": mtp_algo, "group_size": mtp_group}
    ql.update(extra or {})
    return {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
            "ignore": ["lm_head"], "quantized_layers": ql}


@pytest.mark.parametrize("spelling", ["FP8_PB_WO", "FP8_BLOCK_SCALES"])
def test_block_fp8_config_is_built_for_both_spellings(spelling):
    cfg = M.ModelOptMixedPrecisionConfig.from_config(mixed_qc(spelling))
    assert cfg.fp8_block_config is not None
    assert cfg.fp8_block_config.weight_block_size == [128, 128]
    assert cfg.fp8_block_config.is_checkpoint_fp8_serialized
    assert cfg.fp8_block_config.activation_scheme == "dynamic"


def test_no_block_layer_builds_no_fp8_config():
    cfg = M.ModelOptMixedPrecisionConfig.from_config(mixed_qc(mtp_algo=None))
    assert cfg.fp8_block_config is None


def test_two_block_edges_are_refused():
    qc = mixed_qc(extra={"mtp.layers.1.mlp.experts": {"quant_algo": "FP8_BLOCK_SCALES", "group_size": 64}})
    with pytest.raises(ValueError, match="one block size"):
        M.ModelOptMixedPrecisionConfig.from_config(qc)


def _bare(cls):
    obj = object.__new__(cls)
    torch.nn.Module.__init__(obj)
    return obj


class _Rec:
    def __init__(self, quant_config):
        self.quant_config = quant_config


def test_block_fp8_moe_and_linear_get_the_fp8_block_methods():
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

    cfg = M.ModelOptMixedPrecisionConfig.from_config(mixed_qc())
    # the model installs its fused-module mapping before any layer is built
    cfg.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    with mock.patch.object(fp8_mod, "Fp8MoEMethod", _Rec), mock.patch.object(
        fp8_mod, "Fp8LinearMethod", _Rec
    ):
        moe = cfg.get_quant_method(_bare(FusedMoE), "mtp.layers.0.mlp.experts")
        cfg.quantized_layers["mtp.layers.0.self_attn.qkv_proj"] = {"quant_algo": "FP8_PB_WO", "group_size": 128}
        lin = cfg.get_quant_method(_bare(LinearBase), "mtp.layers.0.self_attn.qkv_proj")
    assert isinstance(moe, _Rec) and moe.quant_config is cfg.fp8_block_config
    assert isinstance(lin, _Rec) and lin.quant_config is cfg.fp8_block_config


def test_routed_nvfp4_dispatch_is_unchanged():
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

    cfg = M.ModelOptMixedPrecisionConfig.from_config(mixed_qc())
    with mock.patch.object(M, "ModelOptNvFp4FusedMoEMethod", _Rec):
        got = cfg.get_quant_method(_bare(FusedMoE), "model.layers.0.mlp.experts")
    assert isinstance(got, _Rec) and got.quant_config is cfg.nvfp4_config


def _mtp_quant_config():
    try:
        from sglang.srt.models.qwen3_5_mtp import _mtp_quant_config as f
    except Exception as ex:  # pragma: no cover - import-chain dependent
        pytest.skip(f"qwen3_5_mtp not importable on this host: {ex}")
    return f


def test_mtp_head_quantized_iff_listed():
    f = _mtp_quant_config()
    listed = M.ModelOptMixedPrecisionConfig.from_config(mixed_qc())
    unlisted = M.ModelOptMixedPrecisionConfig.from_config(mixed_qc(mtp_algo=None))
    assert f(listed) is listed
    assert f(unlisted) is None


def test_serialized_nvfp4_mtp_stays_bf16():
    f = _mtp_quant_config()
    fp4 = types.SimpleNamespace(get_name=lambda: "modelopt_fp4")
    assert f(fp4) is None
