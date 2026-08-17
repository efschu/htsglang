# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#727: quantized-resident vocab for compressed-tensors, not just GGUF.

WHAT WAS MISSING. #724 established that our dequant-on-gather path is wired but
scoped to GGUF: ``qwen3_5.py`` only passes a ``quant_config`` to
``VocabParallelEmbedding`` when ``quant_config.get_name() == "gguf"``, and the
whole tree carried exactly two embedding methods --
``UnquantizedEmbeddingMethod`` and ``GGUFEmbeddingMethod``. Compressed-tensors'
``get_quant_method`` handles ``LinearBase`` and ``FusedMoE`` and nothing else,
so an int8 vocab tensor had no method to load it even if the checkpoint carried
one. Closing the gap therefore meant BUILDING a method, not flipping a flag.

THE GATHER IS THE POINT. An embedding is a row lookup, and the checkpoint's
scheme is per-output-channel -- i.e. per VOCAB ROW. So dequantizing on gather
is exact and costs one multiply on the handful of rows a batch actually
touches, never the 248320-row matrix. That is why this is the cheap half of
#727; ``lm_head`` is a GEMM over the whole vocab and is a separate decision.

DEFAULT UNCHANGED. A checkpoint whose ``ignore`` list still excludes
``embed_tokens`` -- which every checkpoint we serve today does -- must keep
taking the dense BF16 path byte-identically.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.quantization.base_config import (
    method_has_implemented_embedding,
)
from sglang.srt.layers.quantization.compressed_tensors.ct_embedding import (
    CompressedTensorsEmbeddingMethod,
    vocab_is_quantized,
)


class _Layer(torch.nn.Module):
    """Stand-in for VocabParallelEmbedding's parameter holder."""

    def register(self, name, tensor):
        self.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))


def _make(rows=8, dim=4):
    layer = _Layer()
    method = CompressedTensorsEmbeddingMethod(params_dtype=torch.float32)
    method.create_weights(
        layer,
        dim,
        [rows],
        dim,
        rows,
        params_dtype=torch.float32,
        weight_loader=lambda *a, **k: None,
    )
    return layer, method


class TestItSatisfiesTheEmbeddingContract(unittest.TestCase):
    def test_the_guard_accepts_it(self):
        # VocabParallelEmbedding refuses any method that has not overridden
        # `embedding`; this is the check it performs.
        self.assertTrue(
            method_has_implemented_embedding(CompressedTensorsEmbeddingMethod)
        )

    def test_create_weights_shapes_match_the_checkpoint_convention(self):
        layer, _ = _make(rows=8, dim=4)
        self.assertEqual(tuple(layer.weight.shape), (8, 4))
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertEqual(tuple(layer.weight_scale.shape), (8, 1))


class TestDequantOnGather(unittest.TestCase):
    def test_it_dequantizes_only_the_gathered_rows(self):
        layer, method = _make(rows=8, dim=4)
        with torch.no_grad():
            layer.weight.copy_(
                torch.arange(-16, 16, dtype=torch.int8).reshape(8, 4)
            )
            layer.weight_scale.copy_(
                torch.arange(1, 9, dtype=torch.float32).reshape(8, 1)
            )
        out = method.embedding(layer, torch.tensor([0, 3, 7]))
        expected = torch.stack(
            [
                layer.weight[0].float() * 1.0,
                layer.weight[3].float() * 4.0,
                layer.weight[7].float() * 8.0,
            ]
        )
        self.assertTrue(torch.equal(out, expected))

    def test_the_output_dtype_is_the_activation_dtype(self):
        layer, method = _make()
        out = method.embedding(layer, torch.tensor([1]))
        self.assertEqual(out.dtype, torch.float32)

    def test_an_empty_gather_is_not_an_error(self):
        layer, method = _make()
        out = method.embedding(layer, torch.tensor([], dtype=torch.long))
        self.assertEqual(tuple(out.shape), (0, 4))

    def test_a_round_trip_against_the_requant_tool_is_exact_per_row(self):
        # The tool and the runtime must agree, or the checkpoint is unreadable
        # in the only way that matters. Same scheme, both directions.
        import os
        import sys

        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "..", "tools",
            ),
        )
        try:
            from requant_vocab_int8 import quantize_per_channel_symmetric
        finally:
            sys.path.pop(0)

        torch.manual_seed(7)
        w = torch.randn(16, 8, dtype=torch.bfloat16) * 2.0
        q, s = quantize_per_channel_symmetric(w)

        layer, method = _make(rows=16, dim=8)
        with torch.no_grad():
            layer.weight.copy_(q)
            layer.weight_scale.copy_(s.float())
        got = method.embedding(layer, torch.arange(16))
        rowmax = w.float().abs().amax(dim=1, keepdim=True)
        err = (got - w.float()).abs()
        self.assertTrue(bool((err <= (rowmax / 127.0) * 1.01 + 1e-6).all()))


class TestTheDefaultPathIsUnchanged(unittest.TestCase):
    """A checkpoint that still excludes the vocab must not take this path."""

    def test_an_ignored_embed_is_not_quantized(self):
        cfg = {"ignore": ["lm_head", "re:.*embed_tokens.*", "re:.*norm.*"]}
        self.assertFalse(vocab_is_quantized(cfg, "model.language_model.embed_tokens"))

    def test_a_requantized_checkpoint_is_quantized(self):
        cfg = {"ignore": ["lm_head", "re:.*norm.*"]}
        self.assertTrue(vocab_is_quantized(cfg, "model.language_model.embed_tokens"))

    def test_a_plain_name_entry_also_excludes(self):
        cfg = {"ignore": ["lm_head"]}
        self.assertFalse(vocab_is_quantized(cfg, "lm_head"))

    def test_a_missing_ignore_list_means_quantized(self):
        self.assertTrue(vocab_is_quantized({}, "model.embed_tokens"))

    def test_a_malformed_ignore_entry_does_not_raise(self):
        # A bad regex in someone else's checkpoint must not abort the boot;
        # it is treated as a literal, which can only ever be MORE conservative.
        cfg = {"ignore": ["re:[unclosed"]}
        self.assertTrue(vocab_is_quantized(cfg, "model.embed_tokens"))


if __name__ == "__main__":
    unittest.main()
