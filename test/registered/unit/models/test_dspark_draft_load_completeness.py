"""Hermetic unit test for the DSpark draft's post-load completeness check.

``load_weights`` drops a checkpoint tensor it cannot match to a parameter
with only a warning. That direction is deliberate (forward compatibility:
a checkpoint may carry tensors this build has no module for), but it is
also how the C1 class of defects stays silent: a remap that produces a
name matching NO parameter leaves that parameter at its uninitialised
construction value, the draft still "loads", and the only symptom is a
speculative accept rate pinned at zero. The unanchored ``.scale`` rename
and the #113 GGUF draft-MTP namespace bug are the same family.

``_assert_required_params_loaded`` closes the other direction: every
parameter the draft DECLARES must have been written by the load.

Pure Python, no GPU, no weights: the check reads only ``params_dict``,
``loaded_params`` and ``self.confidence_head``, so the shipped methods are
driven directly on a stand-in rather than on a constructed model.
"""

import unittest

from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_assert_required = DeepseekV4ForCausalLMDSpark._assert_required_params_loaded
_remap_rest = DeepseekV4ForCausalLMDSpark._remap_mtp_rest
_remap_name = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name


class _Stand:
    """Enough of the draft module for the two loader methods under test."""

    # The real rename rule, so the C1 replay below runs the shipped code.
    _remap_mtp_rest = staticmethod(_remap_rest)

    def __init__(self, confidence_head=None):
        self.confidence_head = confidence_head


_HEAD = object()  # stands in for a built confidence head (only identity matters)


def _params(*names):
    # The check only ever reads the KEYS of params_dict.
    return {name: None for name in names}


class TestDSparkDraftLoadCompleteness(CustomTestCase):
    def test_fully_loaded_checkpoint_passes(self):
        params = _params(
            "stages.0.self_attn.wo_b.weight",
            "stages.0.mlp.gate_proj.weight",
            "markov_head.w1.weight",
        )
        _assert_required(
            _Stand(), params_dict=params, loaded_params=set(params)
        )  # must not raise

    def test_missing_required_tensor_raises_and_names_it(self):
        params = _params(
            "stages.0.self_attn.wo_b.weight",
            "stages.0.mlp.gate_proj.weight",
        )
        with self.assertRaises(ValueError) as ctx:
            _assert_required(
                _Stand(),
                params_dict=params,
                loaded_params={"stages.0.self_attn.wo_b.weight"},
            )
        msg = str(ctx.exception)
        self.assertIn("stages.0.mlp.gate_proj.weight", msg)
        self.assertNotIn("stages.0.self_attn.wo_b.weight", msg)

    def test_shared_target_modules_are_not_required(self):
        # embed_tokens / lm_head are attached from the TARGET after the load
        # (attach_shared_modules); the draft checkpoint never carries them and
        # _remap_dspark_weight_name skips those names for the same reason.
        params = _params(
            "stages.0.mlp.gate_proj.weight",
            "embed_tokens.weight",
            "lm_head.weight",
        )
        _assert_required(
            _Stand(),
            params_dict=params,
            loaded_params={"stages.0.mlp.gate_proj.weight"},
        )  # must not raise

    def test_confidence_head_keeps_its_actionable_message(self):
        params = _params("confidence_head.proj.weight")
        with self.assertRaises(ValueError) as ctx:
            _assert_required(
                _Stand(confidence_head=_HEAD),
                params_dict=params,
                loaded_params=set(),
            )
        msg = str(ctx.exception)
        self.assertIn("confidence_head.proj.weight", msg)
        self.assertIn("enable_confidence_head=False", msg)

    def test_wholly_unloaded_checkpoint_reports_count_and_caps_the_list(self):
        # A wrong checkpoint leaves every parameter unwritten; the message
        # must stay readable while still stating the full extent.
        params = _params(*(f"stages.0.layer{i}.weight" for i in range(50)))
        with self.assertRaises(ValueError) as ctx:
            _assert_required(_Stand(), params_dict=params, loaded_params=set())
        msg = str(ctx.exception)
        self.assertIn("50 declared parameter(s)", msg)
        self.assertIn("and 30 more", msg)

    def test_c1_class_mangled_scale_name_is_caught(self):
        """The end-to-end C1 chain, with the real remap driving the check.

        A packed checkpoint's ``.scales`` must survive the remap. If it does
        not (the pre-#491 unanchored rename), the mapped name matches no
        parameter, the loader warns and moves on, and the scale parameter
        stays unloaded -- which is exactly what this check must catch.
        """
        stand = _Stand()
        params = _params(
            "stages.0.mlp.gate_proj.qweight",
            "stages.0.mlp.gate_proj.scales",
        )
        ckpt = ["mtp.0.ffn.w1.qweight", "mtp.0.ffn.w1.scales"]

        # Real remap: with the suffix anchor in place every name matches.
        loaded = {
            mapped
            for mapped in (_remap_name(stand, name) for name in ckpt)
            if mapped in params
        }
        _assert_required(stand, params_dict=params, loaded_params=loaded)

        # The defect the anchor prevents, replayed on the same check.
        mangled = {
            name.replace(".scales", ".weight_scale_invs") for name in params
        } & set(params)
        with self.assertRaises(ValueError) as ctx:
            _assert_required(
                stand,
                params_dict=params,
                loaded_params={"stages.0.mlp.gate_proj.qweight"} | mangled,
            )
        self.assertIn("stages.0.mlp.gate_proj.scales", str(ctx.exception))
        # Guard the premise: the shipped remap keeps the packed suffix.
        self.assertEqual(_remap_rest("ffn.w1.scales"), "mlp.gate_proj.scales")


if __name__ == "__main__":
    unittest.main(verbosity=3)
