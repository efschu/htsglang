"""A quantised draft namespace must not read as dense (Qwen3.8-27B-INT8).

Companion to ``test_draft_quantization_namespace.py``, which pins the #318
direction: a draft block stored DENSE inside a quantised checkpoint must be
built dense, recovered from the index when the producer forgot to say so. This
file pins the OPPOSITE direction, which Qwen3.8 is the first checkpoint on
this rig to exercise: a draft block stored QUANTISED must not be mistaken for
a dense one.

Measured on the real bring-up, 2026-08-14. Qwen3.8-27B-INT8 quantises its MTP
head -- ``mtp.fc.weight_scale`` and a ``weight_scale`` for every projection of
``mtp.layers.0`` sit in the index next to the weights. The probe nevertheless
returned "dense", so the boot logged

    Draft checkpoint ... declares quant_method 'compressed-tensors' but stores
    the draft namespace unquantized; building the draft model dense.

and the drafter was built in bf16 while int8 payload was copied into it. The
server came up healthy and answered coherently -- the TARGET model is fine --
but speculation collapsed: 1 accepted draft out of 354 proposed,
``spec_accept_length`` 1.017 against a 4-token draft. That is #318's signature
with the two sides swapped, and nothing in the log calls it an error.

The cause is a marker list whose comment does not match its code:

    "scales",  # gptq / awq / marlin (covers weight_scale* too)

``"scales"`` is not a substring of ``"weight_scale"``. The plural covers
``weight_scales``; compressed-tensors writes the SINGULAR. So no marker
matched, and per the probe's own contract "no marker found" means "build this
namespace unquantized" -- the direction its docstring already identifies as
the dangerous one.

The comment is pinned here as well as the behaviour, because a claim in a
comment that the code does not implement is exactly what let this through.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import json
import os
import tempfile
import unittest

from sglang.srt.configs.model_config import (
    _IN_CHECKPOINT_DRAFT_PREFIXES,
    _draft_checkpoint_is_dense,
)
from sglang.srt.utils.common import (
    _PACKED_WEIGHT_MARKERS,
    checkpoint_namespace_is_dense,
)
from sglang.test.test_utils import CustomTestCase

#: The 15 tensors Qwen3.6-27B-INT8-W8A8 stores under `mtp.` -- all dense, no
#: scale of any kind. Its `ignore` list carries `re:.*mtp.*`.
MTP_DENSE = [
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
]

#: The 8 additional tensors Qwen3.8-27B-INT8 stores, verbatim from its index.
MTP_INT8_SCALES = [
    "mtp.fc.weight_scale",
    "mtp.layers.0.self_attn.q_proj.weight_scale",
    "mtp.layers.0.self_attn.k_proj.weight_scale",
    "mtp.layers.0.self_attn.v_proj.weight_scale",
    "mtp.layers.0.self_attn.o_proj.weight_scale",
    "mtp.layers.0.mlp.gate_proj.weight_scale",
    "mtp.layers.0.mlp.up_proj.weight_scale",
    "mtp.layers.0.mlp.down_proj.weight_scale",
]

INT8_QUANT_CFG = {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "config_groups": {
        "INT8": {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "int", "strategy": "channel"},
        }
    },
    "ignore": [
        "re:.*(vision|visual).*",
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*norm.*",
        "re:.*conv1d.*",
        "re:.*linear_attn.*",
    ],
}


def _write_checkpoint(directory, names, quant_cfg=None):
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"weight_map": {n: "model-00001-of-00001.safetensors" for n in names}}, f
        )
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"]}
    if quant_cfg is not None:
        config["quantization_config"] = quant_cfg
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(config, f)
    return directory


class TestAQuantisedDraftNamespaceIsNotDense(CustomTestCase):
    def test_int8_scaled_mtp_block_is_not_dense(self):
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(d, MTP_DENSE + MTP_INT8_SCALES, INT8_QUANT_CFG)
            self.assertIs(
                checkpoint_namespace_is_dense(
                    d, prefixes=_IN_CHECKPOINT_DRAFT_PREFIXES
                ),
                False,
                "a draft namespace carrying weight_scale is quantised; reading "
                "it as dense builds a bf16 drafter and copies int8 payload into "
                "it, which costs every accepted draft and raises no error",
            )

    def test_the_end_to_end_verdict_keeps_quantisation_on(self):
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(d, MTP_DENSE + MTP_INT8_SCALES, INT8_QUANT_CFG)
            self.assertFalse(_draft_checkpoint_is_dense(d))

    def test_the_dense_block_still_reads_dense(self):
        """#318 must keep working: this is the case that DOES need dense."""
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(d, MTP_DENSE, INT8_QUANT_CFG)
            self.assertIs(
                checkpoint_namespace_is_dense(
                    d, prefixes=_IN_CHECKPOINT_DRAFT_PREFIXES
                ),
                True,
            )
            self.assertTrue(_draft_checkpoint_is_dense(d))


class TestTheMarkerListMatchesItsOwnComment(CustomTestCase):
    """A marker list is only as good as the names it actually matches."""

    def test_a_singular_weight_scale_is_matched(self):
        name = "mtp.fc.weight_scale"
        self.assertTrue(
            any(m in name for m in _PACKED_WEIGHT_MARKERS),
            f"no entry of _PACKED_WEIGHT_MARKERS matches {name!r}; the list "
            "claims to cover weight_scale* but 'scales' is plural and this "
            "name is singular",
        )

    def test_the_plural_form_is_still_matched(self):
        self.assertTrue(
            any(m in "model.layers.0.mlp.down_proj.weight_scales" for m in _PACKED_WEIGHT_MARKERS)
        )

    def test_every_compressed_tensors_scale_name_is_matched(self):
        for name in MTP_INT8_SCALES:
            with self.subTest(name=name):
                self.assertTrue(any(m in name for m in _PACKED_WEIGHT_MARKERS))


if __name__ == "__main__":
    unittest.main()
