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
    is_compressed_tensors_config,
    vocab_is_quantized,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    VocabParallelEmbeddingShardIndices,
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


class TestTheFamilyGateActuallyMatches(unittest.TestCase):
    """#763 ROOT: the gate that selects this method must match the real config.

    The method, the dispatch and the ignore-list scan were all correct; the
    model-side gate compared ``get_name()`` against ``"compressed-tensors"``
    while the config answers ``"compressed_tensors"``. A gate that never fires
    is invisible to every test that exercises the method directly, so the
    requantized checkpoint kept the DENSE path: int8 rows copied into a BF16
    embedding with no scale applied, the scale itself homeless -- serving
    logged ``Parameter model.embed_tokens.weight_scale not found in
    params_dict`` on every rank and generated token soup.

    These tests bind the predicate to the real config object, so a rename on
    either side reds here instead of silently disabling the feature.
    """

    def _real_config(self):
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        return CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "int-quantized",
                "ignore": ["lm_head"],
                "config_groups": {},
            }
        )

    def test_the_real_config_is_recognised_as_compressed_tensors(self):
        self.assertTrue(is_compressed_tensors_config(self._real_config()))

    def test_the_hyphen_literal_alone_would_have_missed_it(self):
        # The exact #763 defect, pinned: had the gate stayed a bare equality
        # against the hyphen spelling, it could never have fired.
        self.assertNotEqual(self._real_config().get_name(), "compressed-tensors")

    def test_both_spellings_match(self):
        class _Cfg:
            def __init__(self, name):
                self._name = name

            def get_name(self):
                return self._name

        for name in ("compressed_tensors", "compressed-tensors", "Compressed-Tensors"):
            self.assertTrue(is_compressed_tensors_config(_Cfg(name)), name)

    def test_a_foreign_or_absent_config_does_not_match(self):
        class _Cfg:
            def get_name(self):
                return "gguf"

        class _Raises:
            def get_name(self):
                raise RuntimeError("no name")

        self.assertFalse(is_compressed_tensors_config(None))
        self.assertFalse(is_compressed_tensors_config(_Cfg()))
        self.assertFalse(is_compressed_tensors_config(_Raises()))
        self.assertFalse(is_compressed_tensors_config(object()))


class TestItSurvivesTpVocabSharding(unittest.TestCase):
    """#763: the same checkpoint must read correctly when the vocab is SHARDED.

    Every test above drives a single unsharded partition, and ``_make`` hands
    ``create_weights`` a dummy loader -- so the real
    ``VocabParallelEmbedding.weight_loader`` never ran against these
    parameters. That is the hole this class closes, and it is exactly the
    axis on which #763 was observed: the requantized checkpoint generates
    coherent text under ``--pp-size 3`` (where the embedding sits whole on
    one stage, tp_size == 1) and pure token soup under ``--tp-size 3``, where
    the vocab is row-sharded across ranks.

    The loader decides how to slice from ``param.output_dim``. A method that
    registers its parameters without that attribute gets the
    "copy onto all gpus" branch intended for shard-invariant tensors like
    gptq's ``g_idx`` -- which for a vocab matrix is simply the wrong contract.
    """

    VOCAB, DIM, TP = 12, 4, 3

    @staticmethod
    def _shard_layer(rows, start, end, org_vocab_size, tp_size):
        """A stand-in carrying the fields the REAL loader reads."""
        layer = _Layer()
        layer.shard_indices = VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=start,
            padded_org_vocab_end_index=end,
            padded_added_vocab_start_index=org_vocab_size,
            padded_added_vocab_end_index=org_vocab_size,
            org_vocab_start_index=start,
            org_vocab_end_index=end,
            added_vocab_start_index=org_vocab_size,
            added_vocab_end_index=org_vocab_size,
        )
        layer.num_embeddings_per_partition = rows
        layer.org_vocab_size = org_vocab_size
        layer.tp_size = tp_size
        layer.use_presharded_weights = False
        layer.vocab_partition_sizes = None
        return layer

    def _global_weights(self):
        torch.manual_seed(763)
        w = torch.randint(-127, 127, (self.VOCAB, self.DIM), dtype=torch.int8)
        s = (torch.arange(1, self.VOCAB + 1, dtype=torch.float32) / 8.0).reshape(-1, 1)
        return w, s

    def _reference(self, w, s, ids):
        """TP=1: what PP=3 serving computes, and what the shards must match."""
        layer, method = _make(rows=self.VOCAB, dim=self.DIM)
        with torch.no_grad():
            layer.weight.copy_(w)
            layer.weight_scale.copy_(s)
        return method.embedding(layer, ids)

    def _sharded(self, w, s, ids):
        """TP=3, loaded through the real loader and reduced like forward()."""
        per_rank = self.VOCAB // self.TP
        total = None
        for rank in range(self.TP):
            start, end = rank * per_rank, (rank + 1) * per_rank
            layer = self._shard_layer(per_rank, start, end, self.VOCAB, self.TP)
            method = CompressedTensorsEmbeddingMethod(params_dtype=torch.float32)
            method.create_weights(
                layer,
                self.DIM,
                [per_rank],
                self.DIM,
                self.VOCAB,
                params_dtype=torch.float32,
                weight_loader=lambda p, lw: VocabParallelEmbedding.weight_loader(
                    layer, p, lw
                ),
            )
            # The stock load path: each parameter goes through its loader with
            # the FULL checkpoint tensor; slicing to this rank is the loader's
            # job, and doing it correctly for BOTH weight and scale is the
            # whole contract under test.
            layer.weight.weight_loader(layer.weight, w)
            layer.weight_scale.weight_loader(layer.weight_scale, s)

            # VocabParallelEmbedding.forward: mask ids outside this shard,
            # rebase to local rows, zero the masked rows, all-reduce by sum.
            mask = (ids < start) | (ids >= end)
            local = (ids - start).masked_fill(mask, 0)
            out = method.embedding(layer, local)
            out = out.masked_fill(mask.unsqueeze(-1), 0.0)
            total = out if total is None else total + out
        return total

    def test_a_sharded_vocab_gathers_the_same_rows_as_tp1(self):
        w, s = self._global_weights()
        ids = torch.tensor([0, 1, 5, 6, 11, 4])
        expected = self._reference(w, s, ids)
        got = self._sharded(w, s, ids)
        self.assertTrue(
            torch.equal(got, expected),
            f"TP={self.TP} sharded embedding diverged from the TP=1 reference:\n"
            f"expected\n{expected}\ngot\n{got}",
        )

    def test_tp1_through_the_real_loader_is_unchanged(self):
        """The PP=3 layout (tp_size 1) must keep loading byte-identically.

        The whole vocab IS the local shard there, so this is the case that
        already worked; it is pinned so the #763 sharding fix cannot regress
        the layout serving runs on today.
        """
        w, s = self._global_weights()
        layer = self._shard_layer(self.VOCAB, 0, self.VOCAB, self.VOCAB, 1)
        method = CompressedTensorsEmbeddingMethod(params_dtype=torch.float32)
        method.create_weights(
            layer,
            self.DIM,
            [self.VOCAB],
            self.DIM,
            self.VOCAB,
            params_dtype=torch.float32,
            weight_loader=lambda p, lw: VocabParallelEmbedding.weight_loader(
                layer, p, lw
            ),
        )
        layer.weight.weight_loader(layer.weight, w)
        layer.weight_scale.weight_loader(layer.weight_scale, s)
        self.assertTrue(torch.equal(layer.weight, w))
        self.assertTrue(torch.equal(layer.weight_scale, s))
        ids = torch.tensor([0, 4, 11])
        self.assertTrue(
            torch.equal(
                method.embedding(layer, ids), self._reference(w, s, ids)
            )
        )

    def test_the_scale_is_sharded_with_its_rows(self):
        """The narrow failure mode, named: row r must keep ITS OWN scale.

        A weight sliced per rank while the scale is copied whole (or vice
        versa) still produces a full-shaped, finite, entirely wrong tensor --
        which is what token soup looks like from the outside.
        """
        w, s = self._global_weights()
        per_rank = self.VOCAB // self.TP
        rank = 2
        start = rank * per_rank
        layer = self._shard_layer(per_rank, start, start + per_rank, self.VOCAB, self.TP)
        method = CompressedTensorsEmbeddingMethod(params_dtype=torch.float32)
        method.create_weights(
            layer,
            self.DIM,
            [per_rank],
            self.DIM,
            self.VOCAB,
            params_dtype=torch.float32,
            weight_loader=lambda p, lw: VocabParallelEmbedding.weight_loader(
                layer, p, lw
            ),
        )
        layer.weight.weight_loader(layer.weight, w)
        layer.weight_scale.weight_loader(layer.weight_scale, s)
        self.assertEqual(tuple(layer.weight.shape), (per_rank, self.DIM))
        self.assertEqual(tuple(layer.weight_scale.shape), (per_rank, 1))
        self.assertTrue(torch.equal(layer.weight, w[start : start + per_rank]))
        self.assertTrue(
            torch.equal(layer.weight_scale, s[start : start + per_rank]),
            "the per-row scales did not follow their rows into the shard",
        )


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
