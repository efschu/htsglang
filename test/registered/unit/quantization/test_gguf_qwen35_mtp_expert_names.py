"""Falsification + inertness proof for the Qwen3.5/3.6 GGUF NEXTN/MTP-draft
routed-expert name rewrite (task #113).

Root cause: the shared GGUF weight iterator
(weight_utils.gguf_quant_weights_iterator) splits every ``blk.N.ffn_*_exps``
tensor per expert and emits the MAIN-model HARDCODED name
``model.layers.{N}.mlp.experts.{e}.{proj}.qweight(_type)`` — where N is the
tensor's block index. For the A3B MTP-preserved GGUF the draft block is
``blk.<num_hidden_layers>`` (e.g. blk.40). Qwen3_5ForCausalLMMTP keeps its one
decoder layer under ``mtp.layers.0`` and its ``load_weights`` DROPS every
tensor whose name lacks the substring ``"mtp"``. So the draft's routed experts
were silently never loaded: their fused-MoE weights stayed
``GGUFUninitializedParameter`` and full-perf (graphs ON) crashed in
draft-decode CUDA-graph capture at ``fused_moe_gguf`` (``expert_up = w1[ii]``).
Eager only deferred, it did not fix it.

Fix (gguf_qwen35.py ``transform_stream``, is_draft branch): rewrite the block-N
expert names to ``mtp.layers.0.mlp.experts.{e}.{proj}.*``. After the draft
loader's ``mtp.`` -> ``model.`` rewrite these resolve to
``model.layers.0.mlp.experts.*`` — exactly the main model's routed-expert
names — so the existing per-expert GGUF weight loader materializes them.

These tests drive the REAL ``Qwen3_5GGUFAdapter.transform_stream`` on CPU
tensors (no GPU, no GGUF file, no server): the adapter is built via
``__new__`` with only the attributes the expert branch reads.
"""

import unittest

import torch

from sglang.srt.model_loader.gguf_qwen35 import Qwen35GGUFAdapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Cfg:
    torch_dtype = torch.bfloat16


def _make_adapter(is_draft: bool, num_layers: int = 40) -> Qwen35GGUFAdapter:
    """Build an adapter without touching a GGUF file: transform_stream's expert
    branch only reads self.is_draft and self.num_layers."""
    a = Qwen35GGUFAdapter.__new__(Qwen35GGUFAdapter)
    a.is_draft = is_draft
    a.num_layers = num_layers
    a.config = _Cfg()
    return a


def _run(adapter, names):
    stream = [(n, torch.zeros(1, dtype=torch.int32)) for n in names]
    return [n for n, _ in adapter.transform_stream(stream)]


class TestQwen35MTPExpertNames(CustomTestCase):
    N = 40  # MTP block index == num_hidden_layers

    def test_draft_block_experts_are_remapped_to_mtp(self):
        # The generic iterator hands the draft the block-N experts under the
        # main-model hardcoded name; every one must come out under mtp.layers.0
        # so the draft loader's "mtp" filter keeps it.
        inputs = [
            f"model.layers.{self.N}.mlp.experts.0.gate_proj.qweight",
            f"model.layers.{self.N}.mlp.experts.0.gate_proj.qweight_type",
            f"model.layers.{self.N}.mlp.experts.7.up_proj.qweight",
            f"model.layers.{self.N}.mlp.experts.127.down_proj.qweight",
        ]
        out = _run(_make_adapter(is_draft=True, num_layers=self.N), inputs)
        self.assertEqual(
            out,
            [
                "mtp.layers.0.mlp.experts.0.gate_proj.qweight",
                "mtp.layers.0.mlp.experts.0.gate_proj.qweight_type",
                "mtp.layers.0.mlp.experts.7.up_proj.qweight",
                "mtp.layers.0.mlp.experts.127.down_proj.qweight",
            ],
        )
        # Falsification of the bug: every rewritten name now survives the draft
        # loader's `if "mtp" not in name: continue` gate.
        for n in out:
            self.assertIn("mtp", n)
            # After the draft's mtp.->model. rewrite it lands on the real param.
            self.assertEqual(
                n.replace("mtp.", "model.", 1),
                n.replace("mtp.layers.0", "model.layers.0", 1),
            )

    def test_bug_repro_without_rewrite_name_has_no_mtp(self):
        # Documents exactly what broke: the raw iterator name the draft
        # received carries no "mtp", so the draft loader dropped it.
        raw = f"model.layers.{self.N}.mlp.experts.0.gate_proj.qweight"
        self.assertNotIn("mtp", raw)

    def test_draft_other_block_experts_untouched(self):
        # Only the MTP block (index == num_layers) is the draft's; experts from
        # the base decoder blocks 0..N-1 keep their main-model names and are
        # dropped by the draft loader as before (no accidental capture).
        inputs = [
            "model.layers.0.mlp.experts.3.gate_proj.qweight",
            "model.layers.39.mlp.experts.3.down_proj.qweight",
        ]
        out = _run(_make_adapter(is_draft=True, num_layers=self.N), inputs)
        self.assertEqual(out, inputs)
        for n in out:
            self.assertNotIn("mtp", n)

    def test_non_draft_is_byte_identical_noop(self):
        # The main (target) model must be completely unaffected: block-N expert
        # names pass through untouched (the target simply has no layer N).
        inputs = [
            f"model.layers.{self.N}.mlp.experts.0.gate_proj.qweight",
            f"model.layers.{self.N}.mlp.experts.0.gate_proj.qweight_type",
            "model.layers.5.mlp.experts.2.up_proj.qweight",
        ]
        out = _run(_make_adapter(is_draft=False, num_layers=self.N), inputs)
        self.assertEqual(out, inputs)

    def test_rewrite_keys_off_num_layers(self):
        # The rewritten block index tracks num_hidden_layers, not a hardcoded
        # 40: a 62-layer model's draft block is blk.62.
        inputs = ["model.layers.62.mlp.experts.1.gate_proj.qweight"]
        out = _run(_make_adapter(is_draft=True, num_layers=62), inputs)
        self.assertEqual(out, ["mtp.layers.0.mlp.experts.1.gate_proj.qweight"])


if __name__ == "__main__":
    unittest.main()
