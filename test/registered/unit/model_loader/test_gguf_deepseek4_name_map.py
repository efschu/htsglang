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
from sglang.test.gguf_mxfp4_state import native_path, repack_path

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

    def test_the_shard_set_resolves_from_any_part(self):
        """#391 blocker 2: split sets are loaded, not refused. ``split.count``
        is in every part's KV block, so pointing at part 3 must resolve the same
        four files as pointing at part 1."""
        from sglang.srt.model_loader.gguf_shards import (
            gguf_metadata_path,
            resolve_gguf_shard_paths,
        )

        expected = self.adapter.shard_paths()
        for part in expected:
            self.assertEqual(resolve_gguf_shard_paths(part), expected)
            self.assertEqual(gguf_metadata_path(part), expected[0])

    def test_map_covers_tensors_that_are_not_on_the_metadata_shard(self):
        """Part 1 holds ZERO tensors; a map built from it alone would be empty
        and the unmapped-tensor audit would pass vacuously."""
        import gguf

        first_shard_tensors = {
            str(t.name)
            for t in gguf.GGUFReader(self.adapter.shard_paths()[0], "r").tensors
        }
        self.assertEqual(first_shard_tensors, set())
        self.assertIn("token_embd.weight", self.adapter._file_tensors())

    def test_a_tensor_on_a_later_part_is_readable(self):
        """Header reads prove resolution; this proves the payload is reachable
        through the same stream. ``attn_sinks`` is 64 floats, so it costs a
        page, not a shard."""
        from sglang.srt.model_loader.gguf_shards import iter_gguf_tensors

        wanted = "blk.0.attn_sinks.weight"
        for tensor in iter_gguf_tensors(self.adapter.shard_paths()):
            if str(tensor.name) == wanted:
                self.assertEqual(tuple(tensor.shape), (64,))
                self.assertEqual(len(tensor.data.reshape(-1)), 64)
                return
        self.fail(f"{wanted} not found across the shard set")

    def test_sibling_config_reconciles_against_the_shard_set(self):
        """The vocab cross-check reads ``token_embd.weight``, which lives on
        part 2 while every KV field it compares lives on part 1."""
        import json

        from sglang.srt.model_loader.gguf_registry import reconcile_sibling_config

        config_path = os.path.join(DSV4_GGUF_DIR, "config.json")
        if not os.path.isfile(config_path):
            self.skipTest(f"sibling config.json not placed at {config_path}")
        with open(config_path) as handle:
            raw = json.load(handle)

        class _Sibling:
            pass

        sibling = _Sibling()
        for attr in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "num_hidden_layers",
        ):
            if attr in raw:
                setattr(sibling, attr, raw[attr])

        reconcile_sibling_config(sibling, DSV4_GGUF_FIRST_SHARD, DEEPSEEK4_GGUF_ARCH)
        self.assertEqual(sibling.num_hidden_layers, NUM_LAYERS)

        sibling.vocab_size = raw["vocab_size"] + 1
        with self.assertRaises(ValueError) as ctx:
            reconcile_sibling_config(
                sibling, DSV4_GGUF_FIRST_SHARD, DEEPSEEK4_GGUF_ARCH
            )
        self.assertIn("vocab_size", str(ctx.exception))

    def test_mxfp4_passes_the_gate_only_while_the_repack_is_on(self):
        """UD-Q3_K_XL stores 45 expert tensors as MXFP4, for which a pre-#398
        build has no dequantize or matmul kernel. The load-time repack to Q5_0
        (#391 blocker 1) makes them executable, so the gate must let the file
        through -- and must go back to refusing it, by name, the moment the
        repack is switched off. Both directions on the real shard set.

        Scoped to the repack path (#529): with the native kernels present the
        repack is not what carries MXFP4, so the second direction does not hold
        and this assertion described nothing. The native half is the test
        below.
        """
        from sglang.srt.environ import envs

        with repack_path():
            self.adapter.assert_quant_types_executable()

            with envs.SGLANG_GGUF_MXFP4_REPACK.override(False):
                with self.assertRaises(RuntimeError) as ctx:
                    self.adapter.assert_quant_types_executable()
            message = str(ctx.exception)
            self.assertIn("MXFP4", message)
            self.assertIn("SGLANG_GGUF_MXFP4_REPACK", message)

    def test_mxfp4_passes_the_gate_without_the_repack_when_kernels_are_native(self):
        """The path this rig actually serves (#529).

        With the #398 kernels the gate must accept the same file for a reason
        that has nothing to do with the repack: MXFP4 is in ``DEQUANT_TYPES``
        on its own. Switching the repack OFF must therefore change nothing --
        the property the repack-scoped test above cannot express, and the one
        that decides whether today's boot gets past the gate.
        """
        from sglang.srt.environ import envs

        with native_path():
            self.adapter.assert_quant_types_executable()
            with envs.SGLANG_GGUF_MXFP4_REPACK.override(False):
                self.adapter.assert_quant_types_executable()


if __name__ == "__main__":
    unittest.main()
