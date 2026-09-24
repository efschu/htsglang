"""H68 (NF line): the dry run against the REAL nvidia/Qwen3.8-Flash-Next-NVFP4
download -- metadata only (index + 11 safetensors headers, ~40 MB of JSON) plus
the 24576 gate/up ``weight_scale_2`` scalars (4 bytes each).

OPT-IN: runs only when ``H68_NVFP4_CKPT`` names the checkpoint directory, e.g.

    H68_NVFP4_CKPT=/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-NVFP4-nvidia

The numbers are the checkpoint's (HF revision fc694b54fb0174e0913e6adf86691ef85a4ead47,
verified 24.09. against the HF API sizes and range-fetched headers); they are
Next-Flash values and therefore live here, not in the model-agnostic basis
tests.
"""

import json
import os

import pytest

try:
    from sglang.srt import uneven_perf as U
    from sglang.srt.weg2 import nvfp4_ckpt_plan as P
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

CKPT = os.environ.get("H68_NVFP4_CKPT", "")
pytestmark = pytest.mark.skipif(
    not CKPT or not os.path.isdir(CKPT), reason="H68_NVFP4_CKPT not set (opt-in)"
)

NF_KEY = (
    "modelopt:MIXED_PRECISION|draft.moe.routed=fp8blk128;"
    "moe.routed=nvfp4a4g16;ple.table=fp8pt;*=bf16"
)


@pytest.fixture(scope="module")
def plan():
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptMixedPrecisionConfig,
    )

    qc = json.load(open(os.path.join(CKPT, "config.json")))["quantization_config"]
    cfg = ModelOptMixedPrecisionConfig.from_config(qc)
    return P.plan_from_checkpoint(CKPT, read_scalars=True, resolver=cfg.resolve_quant_algo)


def test_format_key_and_alias_inconsistency(plan):
    assert plan.fmt.key == NF_KEY
    assert plan.fmt.producer == "modelopt 0.46.0.dev281+g73d778422"
    assert plan.fmt.inconsistencies == (
        "mtp.layers.0.mlp.experts: config.json=FP8_PB_WO/g128 "
        "hf_quant_config.json=FP8_BLOCK_SCALES/g128 (aliases -> FP8_BLOCK g128)",
    )


def test_every_tensor_fits_the_plan(plan):
    assert (plan.files, plan.tensors) == (11, 299545)
    assert plan.total_bytes == 132639846394
    assert plan.anomalies == []  # includes the loader drift guard


def test_routed_expert_row(plan):
    row = plan.expert_row
    assert (row.algo, row.layers, row.experts_per_layer, row.uniform) == ("NVFP4", 48, 512, True)
    assert row.ckpt_bytes == 2764824
    assert row.marlin_slot_bytes == 2764804
    assert round(row.row_mib_all_layers, 2) == 126.56
    # Marlin keeps ONE w13 global scale: exact for every expert of this export
    assert row.gate_up_scale2_equal == (24576, 24576)


def test_draft_and_ple(plan):
    assert plan.draft_expert_row.algo == "FP8_BLOCK"
    assert plan.draft_expert_row.experts_per_layer == 512
    assert plan.census[("ple.table", "FP8")][1] == 51200245762
    assert plan.bytes_for("moe.routed", "NVFP4") == 67948314624


def test_planner_names_the_moe_family():
    fmt, _desc, fams = U.checkpoint_compute_format_families(CKPT)
    assert fmt == "nvfp4_a4"
    assert fams[U.GEMM_FAMILY_MOE] == "nvfp4_a4"
