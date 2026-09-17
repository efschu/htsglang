"""#1482: the vocab is quantized iff the checkpoint carries <leaf>.weight_scale;
the ignore list is only the fallback without a readable index."""
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.quantization.compressed_tensors import ct_embedding as ce

CFG = {"config_groups": {"g": {"targets": ["Linear"]}}, "ignore": ["re:.*visual.*"]}


def _ckpt(keys):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model-00001.safetensors" for k in keys}}, f)
    return d


class Test1482(unittest.TestCase):
    def test_bf16_vocab_in_original_checkpoint_is_not_quantized(self):
        d = _ckpt(["lm_head.weight", "model.language_model.embed_tokens.weight",
                   "model.language_model.layers.0.mlp.down_proj.weight_scale"])
        self.assertFalse(ce.vocab_is_quantized(CFG, "model.embed_tokens", d))
        self.assertFalse(ce.vocab_is_quantized(CFG, "lm_head", d))

    def test_int8_vocab_with_scale_is_quantized(self):
        d = _ckpt(["model.language_model.embed_tokens.weight",
                   "model.language_model.embed_tokens.weight_scale", "lm_head.weight"])
        self.assertTrue(ce.vocab_is_quantized(CFG, "model.embed_tokens", d))
        self.assertFalse(ce.vocab_is_quantized(CFG, "lm_head", d))

    def test_no_index_falls_back_to_ignore_list(self):
        d = tempfile.mkdtemp()  # nothing readable
        self.assertTrue(ce.vocab_is_quantized(CFG, "model.embed_tokens", d))
        self.assertFalse(ce.vocab_is_quantized({"ignore": ["re:.*embed.*"]}, "model.embed_tokens", d))
        self.assertTrue(ce.vocab_is_quantized(CFG, "model.embed_tokens", None))


if __name__ == "__main__":
    unittest.main()
