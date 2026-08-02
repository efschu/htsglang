# SPDX-License-Identifier: Apache-2.0
"""#442 / upstream sgl-project/sglang#33276: the DSpark draft namespace.

A hybrid DeepSeek-V4 NVFP4 checkpoint declares ``quant_method=fp8`` with
``moe_quant_algo=NVFP4``. Its draft experts are NOT NVFP4 -- they stay in the
source MXFP4 layout -- so ``_get_quantization_config`` appends the draft
namespace to the NVFP4 exclusion list before building ``ModelOptFp4Config``.

That list named only ``model.decoder.*``, which is how NEXTN exposes the draft
block. DSpark exposes it as ``stages.<stage_id>.*``
(``deepseek_v4_dspark.py::_remap_dspark_weight_name`` rewrites every ``mtp.N.*``
checkpoint name to ``stages.N.*``), so a DSpark draft MoE layer fell through to
``ModelOptNvFp4FusedMoEMethod`` and was read as NVFP4.

Pinned here:

* a DSpark draft expert prefix is excluded (the fix),
* a NEXTN draft expert prefix stays excluded (no regression),
* a target expert prefix stays NVFP4 (the exclusion did not widen).
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import HybridFp8NvFp4Config
from sglang.srt.model_loader.loader import _get_quantization_config

# One DSpark stage, one NEXTN decoder block, one ordinary target layer. The
# first two must be excluded from NVFP4, the third must not.
DSPARK_DRAFT_EXPERTS = "stages.0.mlp.experts"
NEXTN_DRAFT_EXPERTS = "model.decoder.0.mlp.experts"
TARGET_EXPERTS = "model.layers.3.mlp.experts"


def _hybrid_nvfp4_quant_config() -> HybridFp8NvFp4Config:
    """Drive the real loader branch that builds the exclusion list."""

    model_config = types.SimpleNamespace(
        quantization="fp8",
        is_fp4_experts=True,
        # What config.json:quantization_config carries for this family. An
        # empty checkpoint-side exclusion list is the interesting case: the
        # draft namespaces are added by the loader, not by the checkpoint.
        nvfp4_moe_meta={"group_size": 16, "exclude_modules": []},
        dtype="auto",
    )
    fp8_config = Fp8Config(is_checkpoint_fp8_serialized=True)
    model_class = types.SimpleNamespace(
        packed_modules_mapping={},
        remap_prefix=None,
        hf_to_sglang_mapper=None,
    )

    loader = "sglang.srt.model_loader.loader"
    with (
        mock.patch(
            f"{loader}.get_model_architecture", return_value=(model_class, "arch")
        ),
        mock.patch(f"{loader}.get_quant_config", return_value=fp8_config),
        mock.patch(f"{loader}._enforce_capability_floor"),
        mock.patch.object(Fp8Config, "get_supported_act_dtypes", return_value=["auto"]),
    ):
        quant_config = _get_quantization_config(model_config, load_config=None)

    assert isinstance(quant_config, HybridFp8NvFp4Config), quant_config
    return quant_config


class TestDsparkNvfp4Exclusion(unittest.TestCase):
    def setUp(self) -> None:
        self.nvfp4_config = _hybrid_nvfp4_quant_config().nvfp4_config

    def test_dspark_draft_experts_are_excluded_from_nvfp4(self) -> None:
        # Fails on the unfixed tree: the list held only "model.decoder.*".
        self.assertTrue(
            self.nvfp4_config.is_layer_excluded(DSPARK_DRAFT_EXPERTS),
            "DSpark draft experts must not be read as NVFP4; the checkpoint "
            "stores them in the source MXFP4 layout.",
        )

    def test_nextn_draft_experts_remain_excluded(self) -> None:
        self.assertTrue(
            self.nvfp4_config.is_layer_excluded(NEXTN_DRAFT_EXPERTS),
            "The NEXTN exclusion predates this change and must survive it.",
        )

    def test_target_experts_are_still_nvfp4(self) -> None:
        self.assertFalse(
            self.nvfp4_config.is_layer_excluded(TARGET_EXPERTS),
            "Widening the exclusion to the target experts would silently drop "
            "NVFP4 for the whole MoE stack.",
        )

    def test_dspark_name_mapping_really_produces_the_stages_prefix(self) -> None:
        """The pattern is only correct if DSpark still emits ``stages.N.*``."""

        from sglang.srt.models.deepseek_v4_dspark import (
            DeepseekV4ForCausalLMDSpark,
        )

        mapper = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name
        mapped = mapper(
            types.SimpleNamespace(confidence_head=None),
            "mtp.0.ffn.experts.0.w1.weight",
        )
        self.assertIsNotNone(mapped)
        self.assertTrue(
            mapped.startswith("stages.0."),
            f"DSpark draft namespace changed to {mapped!r}; the NVFP4 "
            "exclusion pattern must follow it.",
        )
        self.assertTrue(self.nvfp4_config.is_layer_excluded("stages.0.mlp.experts"))


if __name__ == "__main__":
    unittest.main()
