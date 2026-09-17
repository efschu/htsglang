"""#1483: a literal ignore entry that names an ANCESTOR module does not ignore
its children; substring matching stays for every other shape."""
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer

LUED_IGNORE = ["model.visual.blocks.0.attn.qkv", "model.language_model.layers.0.linear_attn",
               "re:.*visual.*", "re:.*mtp.*", "re:.*linear_attn[.]in_proj_a$", "re:.*linear_attn[.]in_proj_b$"]
FM = {"in_proj_qkvz": ["in_proj_qkv", "in_proj_z"], "in_proj_ba": ["in_proj_b", "in_proj_a"],
      "qkv_proj": ["q_proj", "k_proj", "v_proj"]}


class Test1483(unittest.TestCase):
    def test_children_of_an_ancestor_literal_are_quantized(self):
        for name in ("model.language_model.layers.0.linear_attn.out_proj",
                     "model.language_model.layers.0.linear_attn.in_proj_qkvz"):
            self.assertFalse(should_ignore_layer(name, ignore=LUED_IGNORE, fused_mapping=FM), name)

    def test_regex_and_exact_literals_still_ignore(self):
        self.assertTrue(should_ignore_layer("model.language_model.layers.0.linear_attn.in_proj_ba", ignore=LUED_IGNORE, fused_mapping=FM))
        self.assertTrue(should_ignore_layer("model.visual.blocks.0.attn.qkv", ignore=LUED_IGNORE, fused_mapping=FM))
        self.assertTrue(should_ignore_layer("model.language_model.layers.0.linear_attn", ignore=LUED_IGNORE, fused_mapping=FM))
        self.assertTrue(should_ignore_layer("mtp.layers.0.mlp.down_proj", ignore=LUED_IGNORE, fused_mapping=FM))

    def test_partial_literal_substring_still_matches(self):
        self.assertTrue(should_ignore_layer("model.layers.3.self_attn.qkv_proj", ignore=["layers.3.self_attn"], fused_mapping=FM))
        self.assertFalse(should_ignore_layer("model.layers.3.self_attn.qkv_proj", ignore=["model.layers.3"], fused_mapping=FM))


if __name__ == "__main__":
    unittest.main()
