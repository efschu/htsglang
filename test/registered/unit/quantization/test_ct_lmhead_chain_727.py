"""#727 arm C: the lm_head int8 chain, pinned end to end.

The ct_embedding module was shipped with the honest note that "nothing here
selects it" for the head. That is no longer true -- the chain EXISTS on this
lineage -- but nothing pinned it, and two of its links are silent-failure
shaped:

  1. selection: qwen3_vl passes quant_config to ParallelLMHead
     unconditionally for compressed-tensors, and get_quant_method matches
     the head through ``isinstance(layer, VocabParallelEmbedding)`` gated on
     the checkpoint's own ignore list -- the arm-B/C discriminator (the
     ``-embed`` artifact keeps ``lm_head`` ignored -> dense; ``-both`` drops
     it -> int8);
  2. routing: ``should_apply_lm_head_quant_method``'s DEFAULT arm returns
     True, which is what sends the head matmul through
     ``CompressedTensorsEmbeddingMethod.apply``. A future refactor of that
     default into an allowlist would break arm C with plausible-looking
     garbage logits and no exception -- exactly the silent-wrongness class.

Every link is pinned here in both directions, plus numeric identity of the
dequant-linear against the dense reference (per-row scales make it exact for
values that round-trip int8).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _HeadStub(torch.nn.Module):
    pass


class TestSelectionDiscriminatesTheArms(CustomTestCase):
    def _select(self, ignore):
        from types import SimpleNamespace

        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        carrier = SimpleNamespace(config={"ignore": ignore})
        head = object.__new__(ParallelLMHead)
        return CompressedTensorsConfig.get_quant_method(carrier, head, "lm_head")

    def test_both_artifact_selects_the_int8_method_for_the_head(self):
        from sglang.srt.layers.quantization.compressed_tensors.ct_embedding import (
            CompressedTensorsEmbeddingMethod,
        )

        method = self._select(ignore=[])
        self.assertIsInstance(method, CompressedTensorsEmbeddingMethod)

    def test_embed_only_artifact_keeps_the_head_dense(self):
        """Arm B's whole point: lm_head stays BF16 there, so the ignore
        entry stays and the head must take the unquantized path."""
        from sglang.srt.layers.quantization.base_config import QuantizeMethodBase

        method = self._select(ignore=["lm_head"])
        self.assertNotEqual(
            type(method).__name__, "CompressedTensorsEmbeddingMethod"
        )
        self.assertIsInstance(method, QuantizeMethodBase)


class TestTheHeadMatmulRoutesThroughApply(CustomTestCase):
    def test_gate_default_sends_the_int8_head_through_apply(self):
        from sglang.srt.layers.logits_processor import (
            should_apply_lm_head_quant_method,
        )
        from sglang.srt.layers.quantization.compressed_tensors.ct_embedding import (
            CompressedTensorsEmbeddingMethod,
        )

        head = _HeadStub()
        head.weight = torch.zeros(4, 2, dtype=torch.int8)
        self.assertTrue(
            should_apply_lm_head_quant_method(
                head, CompressedTensorsEmbeddingMethod()
            ),
            "the head matmul no longer routes through apply(): arm C would "
            "matmul raw int8 rows and produce garbage logits silently",
        )

    def test_the_method_is_not_listed_unquantized(self):
        from sglang.srt.layers.logits_processor import _UNQUANTIZED_LM_HEAD_METHODS

        self.assertNotIn(
            "CompressedTensorsEmbeddingMethod", _UNQUANTIZED_LM_HEAD_METHODS
        )

    def test_a_dense_head_still_skips_apply(self):
        """The other direction: arms A and B must keep the stock matmul."""
        from sglang.srt.layers.logits_processor import (
            should_apply_lm_head_quant_method,
        )
        from sglang.srt.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        head = _HeadStub()
        head.weight = torch.zeros(4, 2)
        self.assertFalse(
            should_apply_lm_head_quant_method(head, UnquantizedEmbeddingMethod())
        )


class TestDequantLinearMatchesDense(CustomTestCase):
    def test_apply_is_numerically_the_dense_head(self):
        """Per-row symmetric int8: values that are exact int8 multiples of
        their row scale must produce BIT-identical logits to the dense
        head -- the checkpoint's own contract."""
        from sglang.srt.layers.quantization.compressed_tensors.ct_embedding import (
            CompressedTensorsEmbeddingMethod,
        )

        torch.manual_seed(7)
        rows, dim, toks = 16, 8, 3
        q = torch.randint(-127, 128, (rows, dim), dtype=torch.int8)
        scale = (torch.rand(rows, 1) + 0.5) / 127.0
        dense = q.to(torch.float32) * scale

        head = _HeadStub()
        head.weight = torch.nn.Parameter(q, requires_grad=False)
        head.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        x = torch.randn(toks, dim)

        method = CompressedTensorsEmbeddingMethod(params_dtype=torch.float32)
        out = method.apply(head, x)
        ref = torch.nn.functional.linear(x, dense)
        self.assertTrue(torch.equal(out, ref))

    def test_engaged_line_names_the_head_class(self):
        """GATE 0 counts ENGAGED lines (0/1/2 for A/B/C); the line carries
        the layer class name so the two engagements are tellable apart in
        the boot log."""
        import logging

        from sglang.srt.layers.quantization.compressed_tensors import ct_embedding

        head = _HeadStub()
        method = ct_embedding.CompressedTensorsEmbeddingMethod()
        with self.assertLogs(ct_embedding.logger, level=logging.INFO) as cm:
            method.create_weights(
                head,
                input_size_per_partition=8,
                output_partition_sizes=[16],
                input_size=8,
                output_size=16,
            )
        joined = "\n".join(cm.output)
        self.assertIn("INT8-VOCAB ENGAGED", joined)
        self.assertIn("_HeadStub", joined)


if __name__ == "__main__":
    unittest.main()
