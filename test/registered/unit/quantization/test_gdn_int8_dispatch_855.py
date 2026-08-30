"""#855 — the GDN dense projections must dispatch to the SAME W8A8 int8 scheme
as the rest of the model once the `re:.*linear_attn.*` exclusion is lifted.

Hermetic: no CUDA, no checkpoint bytes, no server. The artifact's
`quantization_config` is reproduced literally here, so the test states what the
built artifact claims and fails if the tree stops honouring it.

Two things are proven, and they are different things:

1. ROUTING — `should_ignore_layer` no longer skips in_proj_qkv / in_proj_z /
   out_proj, and still skips in_proj_a / in_proj_b / conv1d / norm / embed /
   lm_head / vision.
2. DISPATCH — the scheme those layers land on is `CompressedTensorsW8A8Int8`
   with `is_static_input_scheme=False` (dynamic per-token activations) and
   `strategy="channel"`, i.e. byte-for-byte the lane the MLP projections
   already run on.

`get_scheme()` itself is deliberately NOT called: its tail applies a device
capability gate, which cannot be evaluated without a GPU. The test calls the
two functions that gate carries — `get_scheme_dict` (routing) and
`_get_scheme_from_parts` (dispatch) — which together are the whole decision.

The packed-module check is the one that would have caught a wrong cut line:
`qwen3_5.py` packs in_proj_qkv+in_proj_z into `in_proj_qkvz` and
in_proj_b+in_proj_a into `in_proj_ba`, and
`compressed_tensors/utils.py:74-79` RAISES if the shards of one packed module
disagree about being ignored. Quantizing qkv without z is not a quality choice,
it is a load-time crash.
"""

import unittest

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

# Verbatim from Qwen3.8-27B-INT8-gdncov/config.json. The only delta against the
# incumbent Qwen3.8-27B-INT8 is the last two ignore entries replacing the single
# `re:.*linear_attn.*`.
GDNCOV_QUANT_CONFIG = {
    "config_groups": {
        "INT8": {
            "format": "int-quantized",
            "input_activations": {
                "dynamic": True,
                "num_bits": 8,
                "observer": None,
                "strategy": "token",
                "symmetric": True,
                "type": "int",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "dynamic": False,
                "num_bits": 8,
                "observer": "memoryless_minmax",
                "strategy": "channel",
                "symmetric": True,
                "type": "int",
            },
        }
    },
    "format": "int-quantized",
    "ignore": [
        "re:.*(vision|visual).*",
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*norm.*",
        "re:.*conv1d.*",
        r"re:.*linear_attn\.in_proj_a.*",
        r"re:.*linear_attn\.in_proj_b.*",
    ],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}

INCUMBENT_IGNORE = [
    "re:.*(vision|visual).*",
    "lm_head",
    "re:.*embed_tokens.*",
    "re:.*norm.*",
    "re:.*conv1d.*",
    "re:.*linear_attn.*",
]

L = "model.language_model.layers.3."
GDN = L + "linear_attn."

NEWLY_COVERED = [GDN + "in_proj_qkv", GDN + "in_proj_z", GDN + "out_proj"]
STILL_IGNORED = [
    GDN + "in_proj_a",
    GDN + "in_proj_b",
    GDN + "conv1d",
    GDN + "norm",
    "model.language_model.embed_tokens",
    "lm_head",
    "model.visual.blocks.0.attn.qkv",
]
ALREADY_COVERED = [L + "mlp.down_proj", L + "mlp.gate_proj", L + "self_attn.o_proj"]


def _config():
    cfg = CompressedTensorsConfig.from_config(dict(GDNCOV_QUANT_CONFIG))
    # The loader injects the model class's table; without it the packed GDN
    # names would not resolve at all (the #332 lesson).
    cfg.packed_modules_mapping = dict(Qwen3_5ForCausalLM.packed_modules_mapping)
    return cfg


class TestGdnInt8Routing(unittest.TestCase):
    """Which layers the new ignore list selects."""

    def test_incumbent_ignores_all_gdn(self):
        """The baseline this artifact changes -- proves the test can tell the
        two configs apart instead of passing vacuously."""
        fused = Qwen3_5ForCausalLM.packed_modules_mapping
        for name in NEWLY_COVERED:
            self.assertTrue(
                should_ignore_layer(name, ignore=INCUMBENT_IGNORE, fused_mapping=fused),
                f"{name} should be ignored by the INCUMBENT config",
            )

    def test_gdn_dense_projections_no_longer_ignored(self):
        fused = Qwen3_5ForCausalLM.packed_modules_mapping
        for name in NEWLY_COVERED:
            self.assertFalse(
                should_ignore_layer(name, ignore=GDNCOV_QUANT_CONFIG["ignore"], fused_mapping=fused),
                f"{name} must be quantized by the gdncov config",
            )

    def test_gates_and_non_gdn_exclusions_survive(self):
        fused = Qwen3_5ForCausalLM.packed_modules_mapping
        for name in STILL_IGNORED:
            self.assertTrue(
                should_ignore_layer(name, ignore=GDNCOV_QUANT_CONFIG["ignore"], fused_mapping=fused),
                f"{name} must STAY ignored -- gdncov does not touch this axis",
            )

    def test_already_quantized_layers_unchanged(self):
        fused = Qwen3_5ForCausalLM.packed_modules_mapping
        for name in ALREADY_COVERED:
            self.assertFalse(
                should_ignore_layer(name, ignore=GDNCOV_QUANT_CONFIG["ignore"], fused_mapping=fused),
                f"{name} was quantized before and must remain so",
            )

    def test_packed_module_shards_agree(self):
        """`utils.py:74-79` raises when one packed module's shards disagree.
        Resolving the PACKED names is what the loader actually does."""
        fused = Qwen3_5ForCausalLM.packed_modules_mapping
        ignore = GDNCOV_QUANT_CONFIG["ignore"]
        # in_proj_qkvz -> [in_proj_qkv, in_proj_z]: both quantized
        self.assertFalse(should_ignore_layer(GDN + "in_proj_qkvz", ignore=ignore, fused_mapping=fused))
        # in_proj_ba -> [in_proj_b, in_proj_a]: both ignored
        self.assertTrue(should_ignore_layer(GDN + "in_proj_ba", ignore=ignore, fused_mapping=fused))

    def test_gate_patterns_do_not_leak_onto_siblings(self):
        """`in_proj_a`/`in_proj_b` patterns must not accidentally match
        `in_proj_qkv`, `in_proj_z`, or the packed `in_proj_ba` siblings' peers."""
        gate_only = [
            r"re:.*linear_attn\.in_proj_a.*",
            r"re:.*linear_attn\.in_proj_b.*",
        ]
        for name in NEWLY_COVERED:
            self.assertFalse(
                should_ignore_layer(name, ignore=gate_only, fused_mapping={}),
                f"gate-only patterns must not match {name}",
            )


class TestGdnInt8Dispatch(unittest.TestCase):
    """Which kernel the selected layers land on."""

    def test_gdn_dispatches_to_w8a8_int8_dynamic_token(self):
        cfg = _config()
        linear = torch.nn.Linear(8, 8, bias=False)
        for name in NEWLY_COVERED:
            parts = cfg.get_scheme_dict(linear, name)
            self.assertIsNotNone(parts, f"{name} produced no scheme dict")
            scheme = cfg._get_scheme_from_parts(
                weight_quant=parts["weights"],
                input_quant=parts["input_activations"],
            )
            self.assertIsInstance(
                scheme,
                CompressedTensorsW8A8Int8,
                f"{name} dispatched to {type(scheme).__name__}, not CompressedTensorsW8A8Int8",
            )
            self.assertFalse(
                scheme.is_static_input_scheme,
                f"{name} must use DYNAMIC per-token activation quant",
            )
            self.assertEqual(scheme.strategy, "channel")

    def test_gdn_scheme_identical_to_mlp_scheme(self):
        """The point of the artifact: GDN joins the existing lane rather than
        acquiring one of its own."""
        cfg = _config()
        linear = torch.nn.Linear(8, 8, bias=False)

        def scheme_of(name):
            parts = cfg.get_scheme_dict(linear, name)
            s = cfg._get_scheme_from_parts(
                weight_quant=parts["weights"], input_quant=parts["input_activations"]
            )
            return (type(s).__name__, s.strategy, s.is_static_input_scheme, s.input_symmetric)

        reference = scheme_of(L + "mlp.down_proj")
        for name in NEWLY_COVERED:
            self.assertEqual(scheme_of(name), reference, f"{name} differs from the MLP lane")

    def test_ignored_layers_get_no_scheme(self):
        cfg = _config()
        linear = torch.nn.Linear(8, 8, bias=False)
        for name in STILL_IGNORED:
            self.assertIsNone(
                cfg.get_scheme_dict(linear, name),
                f"{name} must resolve to no scheme at all",
            )


class TestGdnOutProjScaleNotSharded(unittest.TestCase):
    """`out_proj` is the one newly covered projection that is RowParallel, so
    its INPUT dim (6144) is split across TP ranks while its per-channel scales
    are indexed by the OUTPUT dim (5120) and must therefore be held WHOLE on
    every rank.

    If `ChannelQuantScaleParameter` ever acquired a row/input-dim loader, the
    scale vector would be narrowed alongside the K shard and every rank would
    dequantize with the wrong scales -- numerically wrong output, no error
    raised. That is precisely the #763 failure shape (a newly quantized
    component silently wrong under plain TP=3 uneven-DCP), which is why it gets
    a guard here rather than a comment.

    Static, type-level assertions: constructing a real RowParallelLinear needs
    an initialized process group, and the property being protected is a typing
    property, not a runtime value.
    """

    def test_channel_scale_is_output_dim_only(self):
        from sglang.srt.layers.parameter import (
            ChannelQuantScaleParameter,
            RowvLLMParameter,
            _ColumnvLLMParameter,
        )

        self.assertTrue(issubclass(ChannelQuantScaleParameter, _ColumnvLLMParameter))
        self.assertFalse(
            issubclass(ChannelQuantScaleParameter, RowvLLMParameter),
            "ChannelQuantScaleParameter must NOT be row-shardable: a per-channel "
            "scale is indexed by the output dim and must survive a K-dim shard whole",
        )

    def test_row_parallel_load_is_the_whole_copy_path(self):
        """`RowParallelLinear.weight_loader_v2` narrows only for RowvLLMParameter;
        everything else falls through to the base full-copy loader. Pin which
        implementation the channel scale inherits."""
        from sglang.srt.layers.parameter import (
            BasevLLMParameter,
            ChannelQuantScaleParameter,
        )

        self.assertEqual(
            ChannelQuantScaleParameter.load_row_parallel_weight,
            BasevLLMParameter.load_row_parallel_weight,
            "the channel scale must use the base full-copy row loader, not a "
            "narrowing one",
        )

    def test_w8a8_int8_allocates_channel_scale_per_output(self):
        """The scheme allocates `(sum(output_partition_sizes), 1)`. For a
        RowParallelLinear that list is the FULL output size on every rank, so
        the allocation is TP-invariant."""
        import inspect

        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w8a8_int8 as mod,
        )

        src = inspect.getsource(mod.CompressedTensorsW8A8Int8.create_weights)
        self.assertIn("ChannelQuantScaleParameter", src)
        self.assertIn("sum(output_partition_sizes), 1", src)
        self.assertIn("output_dim=0", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
