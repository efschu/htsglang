"""Task #332, posten 2: an exclusion list must hit fused and unfused alike.

A quantiser writes down what it did NOT quantise, and it writes it in the
checkpoint's own names -- which are the UNFUSED ones, because a safetensors
file has no ``qkv_proj``. The runtime builds the fused module. If the match is
not fusion-aware, the entry silently misses and the layer is built quantised
against tensors that are dense (or, for a draft block, absent entirely).

Measured 2026-07-31: with ``--speculative-algorithm NEXTN`` on
``ocicek/Qwen3.6-27B-NVFP4`` the draft died in the #318 guard with 8 unloaded
parameters, because the checkpoint names ``mtp.layers.0.self_attn.q_proj`` and
the drafter builds ``mtp.layers.0.self_attn.qkv_proj``. The report filed this
as a prefix (``mtp.`` vs ``model.``) plus fusion problem; the prefix half does
not hold and is corrected here -- see ``TestThePrefixHalfOfTheDiagnosis``.

Root cause: ``_get_quantization_config`` reads ``packed_modules_mapping`` off
the MODEL CLASS, and for a draft the model class is the MTP wrapper -- which
declared none, so the fused table handed to the quant config was empty. Two
fixes, both pinned here: the wrapper now declares the table, and the
compressed-tensors matcher falls back to the same shared table
``is_layer_skipped`` (AWQ / FP8 / ModelOpt) already used.

The three exclusion-list dialects that exist on this box are exercised with
excerpts of the real lists.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

from sglang.srt.layers.quantization.awq.awq import is_layer_skipped_awq
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from sglang.srt.layers.quantization.utils import FALLBACK_FUSED_SHARDS, is_layer_skipped
from sglang.test.test_utils import CustomTestCase

TARGET = "model.language_model.layers.0"
DRAFT = "mtp.layers.0"

#: ocicek/Qwen3.6-27B-NVFP4, `quantization_config.ignore`, the 9 non-vision
#: entries verbatim (the other 110 are `model.visual.*`).
V4_IGNORE = [
    "lm_head",
    "mtp.fc",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj",
]

#: Huihui-Qwen3.6-27B-abliterated-AWQ-MTP, `modules_to_not_convert`: the whole
#: draft under one coarse entry, and the GDN gates named UNFUSED.
AWQ_MODULES_TO_NOT_CONVERT = [
    "mtp",
    f"{TARGET}.linear_attn.in_proj_a",
    f"{TARGET}.linear_attn.in_proj_b",
]

#: Qwen3.6-27B-FP8, `modules_to_not_convert`: the GDN gates BOTH ways round,
#: fused and unfused, plus the non-linear parameters of the same block.
FP8_MODULES_TO_NOT_CONVERT = [
    "lm_head",
    "model.embed_tokens",
    f"{TARGET}.input_layernorm",
    f"{TARGET}.post_attention_layernorm",
    f"{TARGET}.mlp.gate",
    f"{TARGET}.linear_attn.A_log",
    f"{TARGET}.linear_attn.conv1d",
    f"{TARGET}.linear_attn.dt_bias",
    f"{TARGET}.linear_attn.in_proj_ba",
    f"{TARGET}.linear_attn.in_proj_b",
    f"{TARGET}.linear_attn.in_proj_a",
    f"{TARGET}.linear_attn.norm",
]


class TestTheSharedFusionTable(CustomTestCase):
    def test_it_covers_both_universal_fusions(self):
        self.assertEqual(
            FALLBACK_FUSED_SHARDS["qkv_proj"], ["q_proj", "k_proj", "v_proj"]
        )
        self.assertEqual(
            FALLBACK_FUSED_SHARDS["gate_up_proj"], ["gate_proj", "up_proj"]
        )

    def test_it_covers_the_gated_delta_net_fusions(self):
        """Qwen3.5/3.6 fuses two more pairs, and AWQ never passes a table."""
        self.assertEqual(
            FALLBACK_FUSED_SHARDS["in_proj_ba"], ["in_proj_b", "in_proj_a"]
        )
        self.assertEqual(
            FALLBACK_FUSED_SHARDS["in_proj_qkvz"], ["in_proj_qkv", "in_proj_z"]
        )

    def test_it_agrees_with_the_model_class(self):
        from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

        self.assertEqual(
            dict(FALLBACK_FUSED_SHARDS), Qwen3_5ForCausalLM.packed_modules_mapping
        )


class TestTheDraftArchitectureDeclaresItsFusions(CustomTestCase):
    def test_the_mtp_wrapper_carries_the_target_table(self):
        from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
        from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP

        self.assertEqual(
            Qwen3_5ForCausalLMMTP.packed_modules_mapping,
            Qwen3_5ForCausalLM.packed_modules_mapping,
        )

    def test_the_table_is_copied_not_aliased(self):
        """``_get_quantization_config`` mutates it in place for quark / NPU."""
        from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
        from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP

        self.assertIsNot(
            Qwen3_5ForCausalLMMTP.packed_modules_mapping,
            Qwen3_5ForCausalLM.packed_modules_mapping,
        )


class TestTheV4Ignore(CustomTestCase):
    """compressed-tensors ``ignore``, matched with an EMPTY fused table.

    Empty is the state a draft architecture without ``packed_modules_mapping``
    produced, and the state any not-yet-updated architecture still produces;
    the fallback has to carry it on its own.
    """

    def ignored(self, name, fused_mapping=None):
        return should_ignore_layer(name, V4_IGNORE, fused_mapping or {})

    def test_the_fused_draft_attention_is_ignored(self):
        self.assertTrue(self.ignored(f"{DRAFT}.self_attn.qkv_proj"))

    def test_the_fused_draft_mlp_is_ignored(self):
        self.assertTrue(self.ignored(f"{DRAFT}.mlp.gate_up_proj"))

    def test_the_unfused_draft_layers_were_already_ignored(self):
        for name in ("self_attn.o_proj", "mlp.down_proj"):
            self.assertTrue(self.ignored(f"{DRAFT}.{name}"), name)
        self.assertTrue(self.ignored("mtp.fc"))
        self.assertTrue(self.ignored("lm_head"))

    def test_the_whole_draft_block_is_now_exempt(self):
        """The 8 parameters the #318 guard listed, resolved by their modules."""
        for name in (
            "self_attn.qkv_proj",
            "self_attn.o_proj",
            "mlp.gate_up_proj",
            "mlp.down_proj",
        ):
            self.assertTrue(self.ignored(f"{DRAFT}.{name}"), name)

    def test_the_target_stays_quantised(self):
        """V4 is all-Linear: nothing in the language model may become dense."""
        for name in (
            "mlp.gate_up_proj",
            "mlp.down_proj",
            "linear_attn.in_proj_ba",
            "linear_attn.in_proj_qkvz",
            "linear_attn.out_proj",
        ):
            self.assertFalse(self.ignored(f"{TARGET}.{name}"), name)

    def test_an_explicit_table_gives_the_same_answers(self):
        table = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        for name in (
            f"{DRAFT}.self_attn.qkv_proj",
            f"{DRAFT}.mlp.gate_up_proj",
            f"{TARGET}.mlp.gate_up_proj",
        ):
            self.assertEqual(self.ignored(name), self.ignored(name, table), name)

    def test_a_half_excluded_fused_layer_is_loud(self):
        """Silently picking one shard's scheme for both is the worse outcome."""
        with self.assertRaises(ValueError):
            should_ignore_layer(
                f"{DRAFT}.mlp.gate_up_proj", ["mtp.layers.0.mlp.gate_proj"], {}
            )


class TestTheAwqModulesToNotConvert(CustomTestCase):
    def ignored(self, name):
        return is_layer_skipped_awq(name, AWQ_MODULES_TO_NOT_CONVERT)

    def test_the_fused_gdn_gate_is_matched_from_unfused_entries(self):
        """The producer named in_proj_a / in_proj_b; the runtime builds ba.

        Before the shared table grew the GDN pairs this returned False, and
        the 96-row gate was built quantised against dense checkpoint tensors.
        """
        self.assertTrue(self.ignored(f"{TARGET}.linear_attn.in_proj_ba"))

    def test_the_unfused_gdn_gates_still_match(self):
        for name in ("in_proj_a", "in_proj_b"):
            self.assertTrue(self.ignored(f"{TARGET}.linear_attn.{name}"), name)

    def test_the_coarse_draft_entry_still_covers_the_fused_draft(self):
        self.assertTrue(self.ignored(f"{DRAFT}.self_attn.qkv_proj"))
        self.assertTrue(self.ignored(f"{DRAFT}.mlp.gate_up_proj"))

    def test_the_quantised_layers_stay_quantised(self):
        for name in (
            f"{TARGET}.mlp.gate_up_proj",
            f"{TARGET}.mlp.down_proj",
            f"{TARGET}.linear_attn.out_proj",
            f"{TARGET}.linear_attn.in_proj_qkvz",
        ):
            self.assertFalse(self.ignored(name), name)


class TestTheFp8ModulesToNotConvert(CustomTestCase):
    def ignored(self, name):
        return is_layer_skipped(name, FP8_MODULES_TO_NOT_CONVERT)

    def test_the_fused_gdn_gate_matches_both_ways_round(self):
        """This list names in_proj_ba AND its two shards; both must agree."""
        self.assertTrue(self.ignored(f"{TARGET}.linear_attn.in_proj_ba"))
        self.assertTrue(self.ignored(f"{TARGET}.linear_attn.in_proj_a"))
        self.assertTrue(self.ignored(f"{TARGET}.linear_attn.in_proj_b"))

    def test_the_moe_gate_does_not_capture_the_dense_mlp(self):
        """``mlp.gate`` must not match ``mlp.gate_up_proj`` (dotted boundary)."""
        self.assertFalse(self.ignored(f"{TARGET}.mlp.gate_up_proj"))
        self.assertTrue(self.ignored(f"{TARGET}.mlp.gate"))

    def test_the_quantised_layers_stay_quantised(self):
        for name in (
            f"{TARGET}.mlp.down_proj",
            f"{TARGET}.linear_attn.out_proj",
            f"{TARGET}.linear_attn.in_proj_qkvz",
            f"{TARGET}.self_attn.qkv_proj",
        ):
            self.assertFalse(self.ignored(name), name)


class TestThePrefixHalfOfTheDiagnosis(CustomTestCase):
    """The report blamed prefix AND fusion. Only fusion was real.

    ``Qwen3_5ForCausalLMMTP`` builds its inner model with ``prefix="mtp"``, so
    the string ``get_quant_method`` receives IS ``mtp.layers.0....`` -- the
    checkpoint's own namespace, no translation needed. The
    ``model.layers.0.*`` names in the #318 guard message come from
    ``named_parameters()``, the Python attribute tree, which is a different
    naming system that ``load_weights`` bridges separately.

    Pinning it so the next reader does not build a namespace translator that
    nothing needs.
    """

    def test_the_unfused_draft_names_match_without_any_translation(self):
        for name in ("self_attn.o_proj", "mlp.down_proj"):
            self.assertTrue(should_ignore_layer(f"{DRAFT}.{name}", V4_IGNORE, {}), name)

    def test_the_attribute_path_namespace_is_not_what_is_matched(self):
        """`model.layers.0.*` is the attribute tree, and must NOT match."""
        self.assertFalse(
            should_ignore_layer("model.layers.0.self_attn.o_proj", V4_IGNORE, {})
        )

    def test_the_draft_prefix_is_the_checkpoint_namespace(self):
        from sglang.srt.utils import add_prefix

        self.assertEqual(add_prefix("mtp", ""), "mtp")


if __name__ == "__main__":
    unittest.main()
