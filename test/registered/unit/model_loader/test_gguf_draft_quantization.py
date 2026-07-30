# SPDX-License-Identifier: Apache-2.0
"""#290: a GGUF drafter must be BUILT quantized, and an unloaded one must be loud.

The bug this file pins down cost two GPU windows. A DFLASH drafter shipped as
``Qwen3.6-27B-DFlash-Q8_0.gguf`` booted, reported ``Load weight end`` and then
accepted 1.005 tokens per round on every prompt -- the signature of a drafter
proposing noise, not of a weak drafter.

Three separate things had to line up for that to be silent:

1. ``ModelConfig.from_server_args`` reads ``quantization`` for a draft model
   from ``--speculative-draft-model-quantization`` only. The GGUF coupling
   (``arg_groups/overrides._gguf_quantization``) keys on the TARGET's
   ``model_path``, so a ``.gguf`` drafter was built DENSE while the shared
   ``load_format=gguf`` still routed it through the GGUF loader.
2. ``ReplicatedLinear.weight_loader`` had no GGUF branch, unlike every other
   parallel linear -- so even a correctly quantized skeleton could not take the
   drafter's packed ``fc``.
3. ``DFlashDraftModel.load_weights`` skipped names that resolve to nothing.
   Against a dense skeleton the GGUF stream's ``*.qweight`` names resolve to
   nothing -- ALL of them -- so 36 of 58 tensors were dropped and only the 22
   F32 norms landed. Nothing raised.

The desk gate that preceded the window (``test_gguf_dflash_name_map.py``:
``TestTheQ8CheckpointLoadsCompletely``) reported 94/94 names resolved. It built
the model with ``GGUFConfig()`` -- a skeleton the runtime never built. That is
why a green gate and a dead drafter coexisted, and why the checks here are
anchored on the config resolution rather than on a hand-picked quant config.
"""

from __future__ import annotations

import os
import types
import unittest

import torch

from sglang.srt.configs.model_config import ModelConfig

MODEL_ROOT = "/spinning/llm_stuff/club-3090/models-cache"
DFLASH_GGUF = f"{MODEL_ROOT}/qwen3.6-27b-dflash-gguf/Qwen3.6-27B-DFlash-Q8_0.gguf"
DFLASH_GGUF_DIR = f"{MODEL_ROOT}/qwen3.6-27b-dflash-gguf"
DFLASH_HF = f"{MODEL_ROOT}/qwen3.6-27b-dflash"
TARGET_GGUF = f"{MODEL_ROOT}/Qwen3.6-27B-MTP-Q3_K_M-GGUF/Qwen3.6-27B-Q3_K_M.gguf"


def _server_args(**kwargs):
    """The attributes ``from_server_args`` reads, and nothing else.

    A real ``ServerArgs`` cannot be constructed without an accelerator, and
    this test is about one branch of attribute plumbing.
    """
    defaults = dict(
        model_path=TARGET_GGUF,
        speculative_draft_model_path=DFLASH_GGUF,
        speculative_draft_model_quantization=None,
        quantization="gguf",
        decrypted_draft_config_file=None,
        decrypted_config_file=None,
        trust_remote_code=True,
        revision=None,
        context_length=None,
        json_model_override_args="{}",
        is_embedding=False,
        enable_multimodal=None,
        dtype="auto",
        model_impl="auto",
        sampling_defaults="model",
        quantize_and_serve=False,
        enable_multi_layer_eagle=False,
        language_only=False,
        encoder_only=False,
        disable_hybrid_swa_memory=False,
        model_config_parser="auto",
        speculative_algorithm="DFLASH",
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


class _Captured(Exception):
    """Carries the kwargs ``from_server_args`` would hand to ``ModelConfig``."""

    def __init__(self, kwargs):
        super().__init__("captured")
        self.kwargs = kwargs


class TestDraftQuantizationResolution(unittest.TestCase):
    """Which ``quantization`` a draft ModelConfig is built with."""

    def _resolved_quantization(self, server_args, model_path):
        captured = {}

        def _capture(self, **kwargs):
            captured.update(kwargs)
            raise _Captured(kwargs)

        original = ModelConfig.__init__
        ModelConfig.__init__ = _capture
        try:
            with self.assertRaises(_Captured):
                ModelConfig.from_server_args(
                    server_args, model_path=model_path, is_draft_model=True
                )
        finally:
            ModelConfig.__init__ = original
        return captured["quantization"]

    def test_a_gguf_drafter_is_built_quantized(self):
        """THE regression: a `.gguf` drafter must not come out dense."""
        self.assertEqual(
            self._resolved_quantization(_server_args(), DFLASH_GGUF), "gguf"
        )

    def test_the_draft_path_decides_not_the_target_path(self):
        """A dense drafter beside a GGUF target keeps the validated behaviour.

        This is the configuration af197b9d31 measured at accept 2.6-8.0; it
        must stay byte-identical, so nothing may be inherited from the target's
        `quantization="gguf"`.
        """
        self.assertIsNone(
            self._resolved_quantization(_server_args(), DFLASH_HF),
        )

    def test_the_drafter_directory_form_is_not_a_gguf_file(self):
        """`check_gguf_file` is a FILE test; a directory stays unquantized."""
        self.assertIsNone(
            self._resolved_quantization(_server_args(), DFLASH_GGUF_DIR),
        )

    def test_an_explicit_flag_still_wins(self):
        sa = _server_args(speculative_draft_model_quantization="awq")
        self.assertEqual(self._resolved_quantization(sa, DFLASH_GGUF), "awq")

    def test_a_non_draft_model_is_untouched(self):
        """The target keeps reading `server_args.quantization`."""
        captured = {}

        def _capture(self, **kwargs):
            captured.update(kwargs)
            raise _Captured(kwargs)

        original = ModelConfig.__init__
        ModelConfig.__init__ = _capture
        try:
            with self.assertRaises(_Captured):
                ModelConfig.from_server_args(
                    _server_args(quantization="fp8"), model_path=TARGET_GGUF
                )
        finally:
            ModelConfig.__init__ = original
        self.assertEqual(captured["quantization"], "fp8")


class TestReplicatedLinearTakesAPackedWeight(unittest.TestCase):
    """`fc` is a ReplicatedLinear, and the drafter ships it packed."""

    def _module(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.gguf import GGUFConfig

        return ReplicatedLinear(
            256, 64, bias=False, quant_config=GGUFConfig(), prefix="fc"
        )

    def test_the_packed_weight_materializes_and_the_type_is_recorded(self):
        from torch.nn.parameter import UninitializedParameter

        lin = self._module()
        self.assertIsInstance(lin.qweight, UninitializedParameter)

        # Q8_0 == ggml type 8. The loader's first pass carries the type, the
        # second the bytes; both go through the same weight_loader.
        lin.qweight_type.weight_loader(lin.qweight_type, torch.tensor(8))
        rows = torch.randint(0, 255, (64, 256 // 32 * 34), dtype=torch.uint8)
        lin.qweight.weight_loader(lin.qweight, rows)

        self.assertNotIsInstance(lin.qweight, UninitializedParameter)
        self.assertEqual(tuple(lin.qweight.shape), (64, 272))
        self.assertEqual(lin.qweight.dtype, torch.uint8)
        self.assertTrue(torch.equal(lin.qweight.data, rows))
        # `.weight_type` is the ATTRIBUTE the kernels read; copying the value
        # into `.data` alone left it at 0 (= F32) and dequantized Q8_0 bytes
        # as floats.
        self.assertEqual(lin.qweight_type.weight_type, 8)

    def test_a_dense_replicated_linear_is_unchanged(self):
        from sglang.srt.layers.linear import ReplicatedLinear

        lin = ReplicatedLinear(256, 64, bias=False, prefix="fc")
        w = torch.randn(64, 256, dtype=lin.weight.dtype)
        lin.weight.weight_loader(lin.weight, w)
        self.assertTrue(torch.equal(lin.weight.data, w))


def _draft_hf_config():
    import json

    from transformers import PretrainedConfig

    with open(os.path.join(DFLASH_GGUF_DIR, "config.json")) as f:
        return PretrainedConfig(**json.load(f))


@unittest.skipUnless(
    os.path.exists(DFLASH_GGUF), f"drafter checkpoint absent: {DFLASH_GGUF}"
)
class TestTheDrafterLoadsOrSaysSo(unittest.TestCase):
    """Against the real Q8_0 file: it loads completely, or it raises."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.runtime_context import _CONTEXT

        # RotaryEmbedding reads one server-args field during construction.
        if getattr(_CONTEXT, "_server_args", None) is None:
            _CONTEXT._server_args = types.SimpleNamespace(rl_on_policy_target=None)
        cls.cfg = _draft_hf_config()

    def _build(self, quant_config):
        from sglang.srt.models.dflash import DFlashDraftModel
        from sglang.srt.runtime_context import get_parallel

        with get_parallel().override(
            tp_size=1, tp_rank=0, world_size=1, world_rank=0, pp_size=1, pp_rank=0
        ):
            prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                return DFlashDraftModel(self.cfg, quant_config=quant_config, prefix="")
            finally:
                torch.set_default_dtype(prev)

    def _stream(self):
        from sglang.srt.model_loader.gguf_dflash import build_dflash_name_map
        from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator

        return gguf_quant_weights_iterator(DFLASH_GGUF, build_dflash_name_map(self.cfg))

    def _gguf_quant_config(self):
        from sglang.srt.layers.quantization.gguf import GGUFConfig
        from sglang.srt.model_loader.gguf_dflash import (
            dflash_unquantized_module_prefixes,
        )

        qc = GGUFConfig()
        for prefix in dflash_unquantized_module_prefixes(self.cfg):
            if prefix not in qc.modules_to_not_convert:
                qc.modules_to_not_convert.append(prefix)
        return qc

    def test_a_dense_skeleton_refuses_the_packed_stream(self):
        """The exact shape of the #290 boot -- now an error, not an accept rate."""
        model = self._build(quant_config=None)
        with self.assertRaises(ValueError) as ctx:
            model.load_weights(self._stream())
        message = str(ctx.exception)
        self.assertIn("unloaded", message)
        # The 22 F32 norms DO land; the 21 packed parameters (36 file tensors,
        # fused into qkv_proj/gate_up_proj) are the ones named.
        self.assertIn("fc.weight", message)

    def test_the_quantized_skeleton_loads_every_parameter(self):
        model = self._build(quant_config=self._gguf_quant_config())
        model.load_weights(self._stream())  # the guard raises if anything is left

        self.assertEqual(model.fc.qweight_type.weight_type, 8)  # Q8_0
        self.assertEqual(tuple(model.fc.qweight.shape), (5120, 25600 // 32 * 34))
        self.assertEqual(
            model.layers[0].self_attn.qkv_proj.qweight_type.shard_weight_type,
            {"q": 8, "k": 8, "v": 8},
        )
        self.assertEqual(
            model.layers[0].mlp.gate_up_proj.qweight_type.shard_weight_type,
            {0: 8, 1: 8},
        )

    @unittest.skipUnless(
        os.path.exists(os.path.join(DFLASH_HF, "model.safetensors")),
        "BF16 reference drafter absent",
    )
    def test_the_loaded_bytes_dequantize_to_the_reference_weights(self):
        """Loaded, not merely present: the packed parameter must BE the weight.

        A Q8_0 round trip lands ~0.6% relative error; a mis-oriented or
        mis-routed tensor lands at 100%+. Two unfused modules are enough to
        pin orientation and routing: `fc` (the projection that consumes the
        target's context features, and the tensor a bare `nn.Linear` could not
        hold at all) and one MLP `down_proj`.
        """
        import gguf
        import numpy as np
        from safetensors import safe_open

        model = self._build(quant_config=self._gguf_quant_config())
        model.load_weights(self._stream())

        with safe_open(os.path.join(DFLASH_HF, "model.safetensors"), "pt") as st:
            for param_name in ("fc", "layers.0.mlp.down_proj"):
                packed = model.get_parameter(f"{param_name}.qweight")
                deq = gguf.quants.dequantize(
                    packed.detach().cpu().numpy(), gguf.GGMLQuantizationType.Q8_0
                )
                ref = st.get_tensor(f"{param_name}.weight").float().numpy()
                self.assertEqual(deq.shape, ref.shape)
                rel = float(np.abs(deq - ref).mean()) / float(np.abs(ref).mean())
                self.assertLess(rel, 0.02, f"{param_name}: rel_mean_err={rel}")


if __name__ == "__main__":
    unittest.main()
