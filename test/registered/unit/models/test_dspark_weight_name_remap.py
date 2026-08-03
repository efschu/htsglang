"""Hermetic unit test for the DSpark draft checkpoint-name remapping.

The ``.scale`` rename must be anchored at the suffix. Packed checkpoints
(gptq / awq / auto_round) store their scales as ``.scales``, and an
unanchored substring replace turns that into ``.weight_scale_invs`` --
a name that matches no parameter, is dropped with only a warning
(``load_weights``: "DSpark V4 draft: unexpected weight"), and leaves the
draft with uninitialised scales and a speculative accept rate silently
pinned to zero. ``.scale_inv`` degrades the same way, into
``.weight_scale_inv_inv``.

Same failure family as the #113 GGUF draft-MTP namespace bug: a name
rule written for one format, applied unanchored to a second format's
namespace, failing by silent drop rather than by exception.

Pure Python, no GPU, no weights.
"""

import unittest

from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_remap = DeepseekV4ForCausalLMDSpark._remap_mtp_rest


class TestDSparkWeightNameRemap(CustomTestCase):
    def test_packed_scales_suffix_is_preserved(self):
        self.assertEqual(_remap("attn.wo_b.scales"), "self_attn.wo_b.scales")
        self.assertEqual(_remap("ffn.w1.scales"), "mlp.gate_proj.scales")

    def test_packed_scale_inv_suffix_is_preserved(self):
        self.assertEqual(_remap("ffn.w1.scale_inv"), "mlp.gate_proj.scale_inv")

    def test_fp8_scale_suffix_still_renamed(self):
        self.assertEqual(_remap("attn.wo_b.scale"), "self_attn.wo_b.weight_scale_inv")

    def test_other_packed_tensors_untouched(self):
        self.assertEqual(_remap("ffn.w2.qweight"), "mlp.down_proj.qweight")
        self.assertEqual(_remap("ffn.w3.qzeros"), "mlp.up_proj.qzeros")

    def test_existing_renames_intact(self):
        self.assertEqual(_remap("attn_norm.weight"), "input_layernorm.weight")
        self.assertEqual(_remap("ffn.gate.bias"), "mlp.gate.e_score_correction_bias")
        self.assertEqual(_remap("ffn.gate.tid2eid"), "mlp.topk.tid2eid")


if __name__ == "__main__":
    unittest.main()
