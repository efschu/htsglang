"""SGLANG_GGUF_DENSE_VOCAB must reach lm_head, not just embed_tokens (#197).

The flag has two halves and only one of them was wired:

* the LOADER honours it -- ``model_loader/gguf_qwen35.py`` puts ``lm_head.``
  into ``dequant_prefixes`` and into the dense-prefix set, so the GGUF
  ``output`` tensor is dequantized on the fly into a plain ``.weight``;
* the MODULE did not -- ``Qwen3VLForConditionalGeneration.__init__`` built
  ``ParallelLMHead`` with the GGUF ``quant_config`` unconditionally, so it
  still expected packed ``qweight``/``qweight_type`` parameters.

``embed_tokens`` (``Qwen3Model.__init__``) already carried exactly the gate
lm_head was missing. This pins the pair.

CPU only: the constructor is run with the vision tower, the language model,
the head and the process groups stubbed out -- what is under test is which
``quant_config`` reaches ``ParallelLMHead``, nothing numeric.
"""

import os
import types
import unittest
from unittest.mock import patch

import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeQuantConfig:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class _StubModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.init_kwargs = kwargs


def _text_config():
    return types.SimpleNamespace(
        vocab_size=151936,
        hidden_size=4096,
        tie_word_embeddings=False,
        rope_scaling={},
        num_hidden_layers=2,
    )


def _config():
    """The Qwen3.5 shape: a wrapper config carrying a separate text_config.

    Qwen3_5ForConditionalGeneration passes language_model_cls=Qwen3_5ForCausalLM,
    so __init__ takes the `config.text_config` branch — the one the 27B uses.
    """
    return types.SimpleNamespace(
        text_config=_text_config(),
        tie_word_embeddings=False,
        vision_config=types.SimpleNamespace(deepstack_visual_indexes=[8, 16, 24]),
    )


class GgufDenseVocabLmHeadTest(CustomTestCase):
    def setUp(self):
        self._saved = os.environ.get("SGLANG_GGUF_DENSE_VOCAB")
        os.environ.pop("SGLANG_GGUF_DENSE_VOCAB", None)

    def tearDown(self):
        os.environ.pop("SGLANG_GGUF_DENSE_VOCAB", None)
        if self._saved is not None:
            os.environ["SGLANG_GGUF_DENSE_VOCAB"] = self._saved

    def _build(self, quant_name):
        """Run the real __init__ and report the quant_config lm_head got."""
        from sglang.srt.models import qwen3_vl

        seen = {}

        class _SpyLMHead(_StubModule):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                seen["quant_config"] = kwargs.get("quant_config")

        quant_config = _FakeQuantConfig(quant_name) if quant_name else None
        with patch.multiple(
            qwen3_vl,
            Qwen3VLMoeVisionModel=_StubModule,
            ParallelLMHead=_SpyLMHead,
            LogitsProcessor=_StubModule,
            Pooler=_StubModule,
            get_pp_group=lambda: types.SimpleNamespace(
                is_first_rank=True, is_last_rank=True, world_size=1
            ),
            get_server_args=lambda: types.SimpleNamespace(
                mm_enable_dp_encoder=False, enable_dp_lm_head=False
            ),
        ):
            qwen3_vl.Qwen3VLForConditionalGeneration(
                config=_config(),
                quant_config=quant_config,
                language_model_cls=_StubModule,
            )
        return seen["quant_config"]

    def test_gguf_default_keeps_the_quantized_resident_head(self):
        """Flag unset: unchanged, packed qweight head (the default lane)."""
        quant_config = self._build("gguf")
        self.assertIsNotNone(quant_config)
        self.assertEqual(quant_config.get_name(), "gguf")

    def test_gguf_dense_vocab_builds_a_dense_lm_head(self):
        """Flag set: the module must drop the quant_config, like the loader."""
        os.environ["SGLANG_GGUF_DENSE_VOCAB"] = "1"
        self.assertIsNone(
            self._build("gguf"),
            "lm_head was still built with the GGUF quant_config while the "
            "loader dequantizes `lm_head.` into a dense .weight",
        )

    def test_flag_matches_the_loader_gate(self):
        """The module gate and the loader gate must be the same predicate."""
        from sglang.srt.model_loader.gguf_qwen35 import gguf_dense_vocab

        os.environ["SGLANG_GGUF_DENSE_VOCAB"] = "1"
        self.assertTrue(gguf_dense_vocab())
        self.assertIsNone(self._build("gguf"))

        os.environ["SGLANG_GGUF_DENSE_VOCAB"] = "0"
        self.assertFalse(gguf_dense_vocab())
        self.assertIsNotNone(self._build("gguf"))

    def test_non_gguf_quantization_is_untouched_by_the_flag(self):
        """Do-no-harm: the flag is GGUF-only, exactly like the embed gate."""
        os.environ["SGLANG_GGUF_DENSE_VOCAB"] = "1"
        quant_config = self._build("compressed-tensors")
        self.assertIsNotNone(quant_config)
        self.assertEqual(quant_config.get_name(), "compressed-tensors")

    def test_unquantized_model_is_untouched_by_the_flag(self):
        os.environ["SGLANG_GGUF_DENSE_VOCAB"] = "1"
        self.assertIsNone(self._build(None))


if __name__ == "__main__":
    unittest.main()
