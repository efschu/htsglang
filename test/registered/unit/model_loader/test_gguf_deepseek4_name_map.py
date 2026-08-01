# SPDX-License-Identifier: Apache-2.0
"""#391: the GGUF name map for DeepSeek V4 Flash (arch ``deepseek4``).

Two levels, deliberately, following the DFLASH map test:

* The GENERATED map is checked on its own against a synthetic tensor set that
  reproduces the published export's per-layer structure. This runs anywhere.
* The map is checked AGAINST THE FILE when the checkpoint is on this machine.
  That is the check that catches a re-export with renamed tensors, and it is
  skipped rather than faked when the file is absent.

Both load-time gates (split file, unexecutable GGML type) are exercised with
inputs that must trip them -- a gate that has never failed is not known to be
a gate.
"""

from __future__ import annotations

import os
import unittest

import torch

from sglang.srt.model_loader.gguf_deepseek4 import (
    DEEPSEEK4_GGUF_ARCH,
    Deepseek4GGUFAdapter,
)
from sglang.srt.model_loader.gguf_registry import (
    get_gguf_adapter_class,
    sibling_config_gguf_archs,
)

DSV4_GGUF_DIR = (
    "/spinning/llm_stuff/club-3090/models-cache/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL"
)
DSV4_GGUF_FIRST_SHARD = os.path.join(
    DSV4_GGUF_DIR, "DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf"
)

NUM_LAYERS = 43
#: Layers carrying a KV compressor / a sparse indexer in the published export.
COMPRESSOR_LAYERS = list(range(2, 43))
INDEXER_LAYERS = list(range(2, 43, 2))
HASH_LAYERS = [0, 1, 2]
#: Layers carrying the router bias (every layer that is not hash-routed, minus
#: the one the export leaves without a bias).
BIAS_LAYERS = list(range(3, 43))


class _Cfg:
    """Only the fields the adapter reads."""

    model_type = "deepseek_v4"

    def __init__(self, num_hidden_layers: int = NUM_LAYERS):
        self.num_hidden_layers = num_hidden_layers


def _synthetic_tensor_names() -> set:
    """The tensor set of the published UD-Q3_K_XL export, by structure."""
    names = {
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "output_hc_base.weight",
        "output_hc_fn.weight",
        "output_hc_scale.weight",
    }
    per_layer = [
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn_q_a.weight",
        "attn_q_b.weight",
        "attn_kv.weight",
        "attn_output_a.weight",
        "attn_output_b.weight",
        "attn_q_a_norm.weight",
        "attn_kv_a_norm.weight",
        "attn_sinks.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_shexp.weight",
        "ffn_down_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_gate_exps.weight",
        "ffn_down_exps.weight",
        "ffn_up_exps.weight",
        "hc_attn_base.weight",
        "hc_attn_fn.weight",
        "hc_attn_scale.weight",
        "hc_ffn_base.weight",
        "hc_ffn_fn.weight",
        "hc_ffn_scale.weight",
    ]
    for i in range(NUM_LAYERS):
        for suffix in per_layer:
            names.add(f"blk.{i}.{suffix}")
        if i in COMPRESSOR_LAYERS:
            for suffix in (
                "attn_compressor_ape.weight",
                "attn_compressor_gate.weight",
                "attn_compressor_kv.weight",
                "attn_compressor_norm.weight",
            ):
                names.add(f"blk.{i}.{suffix}")
        if i in INDEXER_LAYERS:
            for suffix in (
                "indexer.attn_q_b.weight",
                "indexer.proj.weight",
                "indexer_compressor_ape.weight",
                "indexer_compressor_gate.weight",
                "indexer_compressor_kv.weight",
                "indexer_compressor_norm.weight",
            ):
                names.add(f"blk.{i}.{suffix}")
        if i in HASH_LAYERS:
            names.add(f"blk.{i}.ffn_gate_tid2eid.weight")
        if i in BIAS_LAYERS:
            names.add(f"blk.{i}.exp_probs_b.bias")
    return names


def _adapter_with(names: set) -> Deepseek4GGUFAdapter:
    adapter = Deepseek4GGUFAdapter(_Cfg(), "/nonexistent/deepseek4.gguf")
    adapter._file_tensor_names = names
    return adapter


class TestDeepseek4NameMapGenerated(unittest.TestCase):
    """No file needed."""

    def setUp(self):
        self.names = _synthetic_tensor_names()
        self.map = _adapter_with(self.names)._build_name_map_unchecked()

    def test_covers_every_tensor(self):
        self.assertEqual(set(self.map), self.names)
        self.assertEqual(len(self.map), 1328)

    def test_targets_are_unique(self):
        # A collision would silently drop a tensor at load time.
        self.assertEqual(len(set(self.map.values())), len(self.map))

    def test_attention_names_are_deepseek_native(self):
        self.assertEqual(self.map["blk.7.attn_q_a.weight"], "layers.7.attn.wq_a.weight")
        self.assertEqual(self.map["blk.7.attn_kv.weight"], "layers.7.attn.wkv.weight")
        self.assertEqual(
            self.map["blk.7.attn_output_b.weight"], "layers.7.attn.wo_b.weight"
        )
        self.assertEqual(
            self.map["blk.7.attn_q_a_norm.weight"], "layers.7.attn.q_norm.weight"
        )

    def test_bare_parameters_have_no_weight_suffix(self):
        for gguf_name in (
            "blk.7.attn_sinks.weight",
            "blk.7.hc_attn_fn.weight",
            "blk.4.attn_compressor_ape.weight",
            "output_hc_scale.weight",
        ):
            self.assertFalse(
                self.map[gguf_name].endswith(".weight"),
                f"{gguf_name} -> {self.map[gguf_name]} must be a bare parameter",
            )
        self.assertEqual(self.map["blk.7.attn_sinks.weight"], "layers.7.attn.attn_sink")
        self.assertEqual(self.map["output_hc_scale.weight"], "hc_head_scale")

    def test_shared_expert_projection_order(self):
        # llama.cpp gate/up/down == DeepSeek w1/w3/w2. Getting this pair
        # crossed produces fluent nonsense rather than an error.
        self.assertEqual(
            self.map["blk.3.ffn_gate_shexp.weight"],
            "layers.3.ffn.shared_experts.w1.weight",
        )
        self.assertEqual(
            self.map["blk.3.ffn_up_shexp.weight"],
            "layers.3.ffn.shared_experts.w3.weight",
        )
        self.assertEqual(
            self.map["blk.3.ffn_down_shexp.weight"],
            "layers.3.ffn.shared_experts.w2.weight",
        )

    def test_router_and_hash_layers(self):
        self.assertEqual(
            self.map["blk.9.ffn_gate_inp.weight"], "layers.9.ffn.gate.weight"
        )
        self.assertEqual(self.map["blk.9.exp_probs_b.bias"], "layers.9.ffn.gate.bias")
        self.assertEqual(
            self.map["blk.0.ffn_gate_tid2eid.weight"], "layers.0.ffn.gate.tid2eid"
        )
        # Hash-routed layers carry no learned bias.
        self.assertNotIn("blk.0.exp_probs_b.bias", self.map)

    def test_unmapped_tensor_is_an_error(self):
        adapter = _adapter_with(self.names | {"blk.7.attn_brand_new.weight"})
        with self.assertRaises(RuntimeError) as ctx:
            adapter._build_name_map_unchecked()
        self.assertIn("not mapped", str(ctx.exception))

    def test_registry_dispatch(self):
        self.assertIs(get_gguf_adapter_class("deepseek_v4"), Deepseek4GGUFAdapter)
        self.assertIn(DEEPSEEK4_GGUF_ARCH, sibling_config_gguf_archs())


class TestDeepseek4TransformStream(unittest.TestCase):
    """The two repairs of the generic iterator's assumptions."""

    def setUp(self):
        self.adapter = _adapter_with(_synthetic_tensor_names())

    def test_integer_table_marker_is_dropped(self):
        name = "layers.0.ffn.gate.tid2eid"
        table = torch.zeros(129280, 6, dtype=torch.int32)
        stream = [
            (name, torch.tensor(24)),  # the 0-dim qweight_type marker
            (name, table),
        ]
        out = list(self.adapter.transform_stream(stream))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], name)
        self.assertEqual(out[0][1].shape, table.shape)

    def test_bf16_router_gate_is_renamed_and_reinterpreted(self):
        original = torch.randn(256, 4096, dtype=torch.bfloat16)
        raw = original.contiguous().view(torch.uint8)
        stream = [
            ("model.layers.3.mlp.gate.qweight_type", torch.tensor(30)),
            ("model.layers.3.mlp.gate.qweight", raw),
        ]
        out = list(self.adapter.transform_stream(stream))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "model.layers.3.mlp.gate.weight")
        self.assertEqual(out[0][1].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out[0][1], original))

    def test_everything_else_passes_through(self):
        tensor = torch.zeros(4)
        out = list(
            self.adapter.transform_stream([("layers.1.attn.wq_a.qweight", tensor)])
        )
        self.assertEqual(out, [("layers.1.attn.wq_a.qweight", tensor)])


@unittest.skipUnless(
    os.path.isfile(DSV4_GGUF_FIRST_SHARD),
    f"DeepSeek V4 Flash GGUF not on this machine ({DSV4_GGUF_FIRST_SHARD})",
)
class TestDeepseek4NameMapAgainstFile(unittest.TestCase):
    """Needs the real 119 GiB export; header reads only, no tensor payload."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = Deepseek4GGUFAdapter(_Cfg(), DSV4_GGUF_FIRST_SHARD)

    def test_all_four_shards_are_resolved(self):
        self.assertEqual(len(self.adapter.shard_paths()), 4)

    def test_map_covers_the_file(self):
        name_map = self.adapter._build_name_map_unchecked()
        self.assertEqual(len(name_map), 1328)
        self.assertEqual(len(set(name_map.values())), 1328)

    def test_split_file_is_refused_with_the_merge_instruction(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.adapter.assert_not_split()
        self.assertIn("llama-gguf-split --merge", str(ctx.exception))

    def test_unexecutable_quant_type_is_refused(self):
        # UD-Q3_K_XL stores the routed down projection as MXFP4, for which
        # this build has no dequantize or matmul kernel.
        with self.assertRaises(RuntimeError) as ctx:
            self.adapter.assert_quant_types_executable()
        self.assertIn("MXFP4", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
