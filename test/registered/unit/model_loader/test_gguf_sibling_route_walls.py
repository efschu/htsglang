# SPDX-License-Identifier: Apache-2.0
"""The two sibling-route boot walls a bespoke-family GGUF hits (#402).

Both come from the same structural fact: a bespoke-arch GGUF (qwen35,
gemma4, deepseek4, dflash-draft) is unreadable by transformers' GGUF metadata
reader, so its config AND its tokenizer are read from the plain HF files
sitting next to the .gguf shards. Those sibling files come from the UPSTREAM
repo, and the upstream repo describes the upstream weights.

WALL 1 -- the sibling config.json carries the upstream checkpoint's
``quantization_config``. For DeepSeek-V4-Flash-0731 that is fp8 block-quant,
while the file being loaded is a GGUF and the CLI says ``--quantization
gguf``. ``ModelConfig._verify_quantization`` sees the two disagree and aborts::

    ValueError: Quantization method specified in the model config (fp8) does
    not match the quantization method specified in the `quantization`
    argument (gguf).

which names no real conflict: not one fp8 tensor is being read. The GGUF's
ggml types are the ground truth, so ``reconcile_sibling_config`` drops the
block (one log line) on the bespoke-GGUF route only.

WALL 2 -- the tokenizer took the other branch. With no sibling tokenizer next
to the shards, ``_resolve_tokenizer_name`` fell back to
``AutoTokenizer(..., gguf_file=...)``, which is the very reader that cannot
read the arch::

    ValueError: GGUF model with architecture deepseek4 is not supported yet.
    (transformers/modeling_gguf_pytorch_utils.py:650)

That fallback cannot work for a bespoke arch -- there is nothing on the other
side of it. It is now a refusal that names the directory and the files that
would fix it, and the sibling route is taken whenever those files exist.

Header-only fixtures, so this runs anywhere; the real 128 GiB V4-Flash export
is exercised by the last class when it happens to be on the machine (config
and tokenizer only -- no weights, no GPU).
"""

import json
import pathlib
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


_GEOMETRY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_hidden_layers": 2,
}

#: The upstream fp8 block exactly as DeepSeek-V4-Flash-0731 ships it.
_UPSTREAM_QUANT_CONFIG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
}

_V4_TREE = pathlib.Path(
    "/spinning/llm_stuff/club-3090/models-cache/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL"
)
_V4_SHARD = _V4_TREE / "DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf"


def _write_gguf(
    directory: pathlib.Path,
    arch: str = "dflash-draft",
    *,
    quantization_config=None,
    tokenizer_files=(),
) -> pathlib.Path:
    """A header-only GGUF plus the sibling files this test wants next to it."""
    import gguf
    import numpy as np

    path = directory / "tiny.gguf"
    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_block_count(_GEOMETRY["num_hidden_layers"])
    writer.add_embedding_length(_GEOMETRY["hidden_size"])
    writer.add_feed_forward_length(_GEOMETRY["intermediate_size"])
    writer.add_head_count(_GEOMETRY["num_attention_heads"])
    writer.add_head_count_kv(_GEOMETRY["num_key_value_heads"])
    writer.add_key_length(_GEOMETRY["head_dim"])
    writer.add_tensor("dflash_fc.weight", np.zeros((64, 32), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    config = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "vocab_size": 100,
        "num_target_layers": 8,
        "dflash_config": {"mask_token_id": 7, "target_layer_ids": [0, 1]},
        **_GEOMETRY,
    }
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (directory / "config.json").write_text(json.dumps(config))
    for name in tokenizer_files:
        (directory / name).write_text("{}")
    return path


class TestWall1SiblingQuantizationConfig(CustomTestCase):
    """The upstream quantization_config must not survive the GGUF route."""

    def test_quantization_config_is_dropped(self):
        from sglang.srt.utils.hf_transformers_utils import get_config

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(
                pathlib.Path(tmp), quantization_config=_UPSTREAM_QUANT_CONFIG
            )
            config = get_config(str(path), trust_remote_code=True)

        # Gone entirely, not merely falsy: ModelConfig probes it with both
        # getattr(...) and hasattr(...).
        self.assertFalse(hasattr(config, "quantization_config"))
        self.assertIsNone(getattr(config, "quantization_config", None))
        # Everything else the sibling config carries is untouched.
        self.assertEqual(config.architectures, ["DFlashDraftModel"])
        self.assertEqual(config.num_target_layers, 8)
        for attr, want in _GEOMETRY.items():
            self.assertEqual(getattr(config, attr), want, msg=attr)

    def test_the_reported_abort_is_what_the_drop_prevents(self):
        """Can-fail: feed the pre-fix config into the frame that aborted.

        ``_verify_quantization`` is the whole of the reported failure. With
        the block present it raises the boot's exact message; the dropped
        config passes the same frame.
        """
        from sglang.srt.configs.model_config import ModelConfig

        pre_fix = ModelConfig.__new__(ModelConfig)
        pre_fix.hf_config = type("C", (), {})()
        pre_fix.hf_config.quantization_config = dict(_UPSTREAM_QUANT_CONFIG)
        pre_fix.hf_text_config = pre_fix.hf_config
        pre_fix.quantization = "gguf"
        pre_fix.is_draft_model = False
        pre_fix.quantization_inherited = False
        pre_fix.quantization_explicitly_unset = False
        pre_fix.model_path = "<fixture>"
        with self.assertRaises(ValueError) as cm:
            pre_fix._verify_quantization()
        self.assertIn("(fp8) does not match", str(cm.exception))
        self.assertIn("(gguf)", str(cm.exception))

        post_fix = ModelConfig.__new__(ModelConfig)
        post_fix.hf_config = type("C", (), {})()
        post_fix.hf_text_config = post_fix.hf_config
        post_fix.quantization = "gguf"
        post_fix.is_draft_model = False
        post_fix.quantization_inherited = False
        post_fix.quantization_explicitly_unset = False
        post_fix.model_path = "<fixture>"
        post_fix._verify_quantization()
        self.assertEqual(post_fix.quantization, "gguf")

    def test_a_config_without_the_block_is_unchanged(self):
        """Negative control: nothing is invented where nothing was declared."""
        from sglang.srt.utils.hf_transformers_utils import get_config

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(pathlib.Path(tmp))
            config = get_config(str(path), trust_remote_code=True)
        self.assertIsNone(getattr(config, "quantization_config", None))

    def test_non_gguf_checkpoints_keep_their_quantization_config(self):
        """The drop is gated on the bespoke-GGUF route.

        A plain HF directory goes nowhere near reconcile_sibling_config, so an
        fp8 / awq / gptq checkpoint still declares what it is.
        """
        from sglang.srt.utils.hf_transformers_utils import get_config

        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            (directory / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen2ForCausalLM"],
                        "model_type": "qwen2",
                        "vocab_size": 100,
                        "quantization_config": _UPSTREAM_QUANT_CONFIG,
                        **_GEOMETRY,
                    }
                )
            )
            config = get_config(str(directory), trust_remote_code=True)

        self.assertEqual(
            config.quantization_config["quant_method"],
            "fp8",
            msg="a non-GGUF checkpoint must keep its own quantization_config",
        )


class TestWall2BespokeTokenizerRoute(CustomTestCase):
    """AutoTokenizer must read the sibling files, or refuse by name."""

    def _resolve(self, path):
        from sglang.srt.utils.hf_transformers.tokenizer import _resolve_tokenizer_name

        kwargs = {}
        resolved = _resolve_tokenizer_name(str(path), kwargs)
        return str(resolved), kwargs

    def test_sibling_tokenizer_is_loaded_without_gguf_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            path = _write_gguf(
                directory, tokenizer_files=("tokenizer.json", "tokenizer_config.json")
            )
            resolved, kwargs = self._resolve(path)

        self.assertEqual(resolved, str(directory))
        # Withholding gguf_file IS the fix: with it, AutoTokenizer re-enters
        # the reader that cannot read this arch.
        self.assertNotIn("gguf_file", kwargs)

    def test_tokenizer_model_alone_is_enough(self):
        """Any one of the sibling tokenizer files takes the route; a
        sentencepiece-only export must not fall back into transformers."""
        for name in ("tokenizer_config.json", "tokenizer.model", "vocab.json"):
            with tempfile.TemporaryDirectory() as tmp:
                directory = pathlib.Path(tmp)
                path = _write_gguf(directory, tokenizer_files=(name,))
                resolved, kwargs = self._resolve(path)
            self.assertEqual(resolved, str(directory), msg=name)
            self.assertNotIn("gguf_file", kwargs, msg=name)

    def test_missing_sibling_tokenizer_refuses_by_name(self):
        """The wall itself. Falling through to gguf_file reaches
        'architecture deepseek4 is not supported yet' one frame inside
        transformers, naming neither the cause nor the fix."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            path = _write_gguf(directory)
            with self.assertRaises(ValueError) as cm:
                self._resolve(path)
        message = str(cm.exception)
        self.assertIn("dflash-draft", message)
        self.assertIn(str(directory), message)
        # Names what is missing, not just that something is.
        self.assertIn("tokenizer.json", message)
        self.assertIn("tokenizer_config.json", message)
        self.assertIn("--tokenizer-path", message)

    def test_non_bespoke_gguf_keeps_the_transformers_route(self):
        """Negative control against an over-broad fix: an arch transformers
        CAN read still goes through gguf_file, sibling files or not."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            path = _write_gguf(directory, arch="llama")
            resolved, kwargs = self._resolve(path)

        self.assertEqual(resolved, str(directory))
        self.assertEqual(kwargs.get("gguf_file"), str(path))

    def test_a_plain_directory_is_untouched(self):
        """--tokenizer-path <dir> never entered this branch and still does
        not: the workaround keeps working, it is merely unnecessary."""
        with tempfile.TemporaryDirectory() as tmp:
            resolved, kwargs = self._resolve(pathlib.Path(tmp))
        self.assertEqual(resolved, tmp)
        self.assertEqual(kwargs, {})


@unittest.skipUnless(
    _V4_SHARD.is_file(), f"V4-Flash export not on this machine: {_V4_SHARD}"
)
class TestOnTheRealV4FlashExport(CustomTestCase):
    """The pristine upstream files, as shipped -- no hand edits.

    Header-level only: the config peek and the tokenizer. Nothing is loaded
    from the 128 GiB of shards.
    """

    def test_pristine_config_json_still_declares_fp8(self):
        """The premise. If upstream ever stops shipping the block, the two
        tests below stop proving anything and this one says so."""
        declared = json.loads((_V4_TREE / "config.json").read_text())
        self.assertEqual(
            declared["quantization_config"]["quant_method"],
            "fp8",
            msg="the model dir's config.json must be the pristine upstream one",
        )

    def test_model_config_builds_from_the_pristine_config(self):
        """model_config.py:1791, the frame boot attempt 1 died in."""
        from sglang.srt.configs.model_config import ModelConfig

        config = ModelConfig(
            model_path=str(_V4_SHARD), trust_remote_code=True, quantization="gguf"
        )
        self.assertEqual(config.quantization, "gguf")
        self.assertEqual(config.hf_config.architectures, ["DeepseekV4ForCausalLM"])
        self.assertEqual(config.num_hidden_layers, 43)
        self.assertFalse(hasattr(config.hf_config, "quantization_config"))

    def test_tokenizer_comes_from_the_shard_directory(self):
        """No --tokenizer-path: the tokenizer files now sit next to the
        shards, and the tokenizer built from them is the one the workaround
        directory produced."""
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        tokenizer = get_tokenizer(str(_V4_SHARD), trust_remote_code=True)
        self.assertEqual(len(tokenizer), 129280)
        text = "Hello uneven TP, DeepSeek V4 Flash."
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
