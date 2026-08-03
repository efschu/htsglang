# SPDX-License-Identifier: Apache-2.0
"""DSpark draft head: fused shared experts must reach the fused slot.

Port of upstream sglang #33312. ``DeepseekV4ForCausalLMDSpark`` never resolved
``num_fused_shared_experts``, so its expert mapping covered
``n_routed_experts`` alone. On a checkpoint that ships the shared expert FUSED
into the routed tensors -- which the official DeepSeek-V4-Flash-0731 DSpark
stages do -- every ``ffn.shared_experts.*`` tensor matched no parameter and
fell into the loader's "unexpected weight -> continue" drop: the same silent
class #491 fixed for the ``.scale`` rename, with a different name family.

Hermetic. The real ``load_weights`` runs against a stub that declares exactly
the parameters under test; nothing is constructed on a device and no kernel is
launched. What is pinned:

* with fusion ON, ``stages.N.mlp.shared_experts.*`` is rewritten into
  ``stages.N.mlp.experts.{n_routed}.*`` and lands in the fused slot;
* the expert mapping is widened to ``n_routed + fused`` -- without that the
  rewritten name would still match nothing, so both halves of the port are
  needed and both are checked;
* with fusion OFF the loader is byte-identical to before: shared-expert names
  keep their own module path and the mapping stays at ``n_routed``;
* the draft head and the target model resolve the SAME number, which is the
  actual defect (two resolvers that could disagree were the bug).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sglang.srt.models import deepseek_v4 as v4mod
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark

N_ROUTED = 3


class _Config:
    n_routed_experts = N_ROUTED
    n_shared_experts = 1
    num_hidden_layers = 2


class _Param(torch.nn.Parameter):
    """A parameter that records what the loader wrote into it."""

    def __new__(cls, loader_name: str):
        obj = super().__new__(cls, torch.zeros(1), requires_grad=False)
        obj.calls = []
        obj.loader_name = loader_name
        return obj

    @property
    def weight_loader(self):
        def _loader(param, loaded_weight, *args, **kwargs):
            param.calls.append((args, kwargs))

        return _loader


class _StubDraft:
    """Just enough of the draft head to run the real ``load_weights``."""

    config = _Config()
    confidence_head = None

    # The methods under test, taken from the production class rather than
    # re-implemented -- that is the whole point of the stub.
    load_weights = DeepseekV4ForCausalLMDSpark.load_weights
    _remap_dspark_weight_name = DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name
    # `_remap_mtp_rest` is a staticmethod on the production class; reading it
    # off the class yields the plain function, so re-wrap it to keep the
    # descriptor behaviour the caller relies on.
    _remap_mtp_rest = staticmethod(DeepseekV4ForCausalLMDSpark._remap_mtp_rest)

    def __init__(self, num_fused_shared_experts: int):
        self.num_fused_shared_experts = num_fused_shared_experts
        self.params = {
            "stages.0.mlp.experts.w13_weight": _Param("experts.w13_"),
            "stages.0.mlp.experts.w2_weight": _Param("experts.w2_"),
            "stages.0.mlp.shared_experts.gate_proj.weight": _Param("shared"),
        }

    def named_parameters(self):
        return list(self.params.items())

    def _assert_required_params_loaded(self, *, params_dict, loaded_params):
        # The production check demands that EVERY declared parameter be
        # written; this stub declares parameters for both layouts at once, so
        # the check is not the subject here and would always fire. What the
        # tests assert instead is which parameter the weight reached.
        return None


def _load(num_fused: int, name: str):
    draft = _StubDraft(num_fused)
    weight = torch.ones(1)
    with patch(
        "sglang.srt.models.deepseek_v4_dspark.logger"
    ) as log:
        draft.load_weights([(name, weight)])
    warnings = [call for call in log.warning.call_args_list]
    return draft, warnings


SHARED_NAME = "mtp.0.ffn.shared_experts.gate_proj.weight"


class TestFusedSharedExpertReachesTheSlot(unittest.TestCase):
    def test_with_fusion_on_the_weight_lands_in_the_fused_expert_slot(self):
        draft, warnings = _load(1, SHARED_NAME)
        fused = draft.params["stages.0.mlp.experts.w13_weight"]
        self.assertEqual(
            len(fused.calls), 1, f"weight was not loaded; warnings={warnings}"
        )
        _, kwargs = fused.calls[0]
        self.assertEqual(kwargs["expert_id"], N_ROUTED)
        self.assertEqual(kwargs["shard_id"], "w1")
        self.assertEqual(warnings, [], "the loader still dropped a weight")
        self.assertEqual(
            draft.params["stages.0.mlp.shared_experts.gate_proj.weight"].calls,
            [],
            "the standalone shared-expert parameter must not also be written",
        )

    def test_with_fusion_off_the_weight_keeps_its_own_module_path(self):
        """Neutrality: the pre-port behaviour, unchanged."""
        draft, warnings = _load(0, SHARED_NAME)
        standalone = draft.params["stages.0.mlp.shared_experts.gate_proj.weight"]
        self.assertEqual(len(standalone.calls), 1)
        self.assertEqual(draft.params["stages.0.mlp.experts.w13_weight"].calls, [])
        self.assertEqual(warnings, [])

    def test_a_routed_expert_is_unaffected_either_way(self):
        """The rewrite must be anchored, not a loose substring replace."""
        routed = "mtp.0.ffn.experts.1.gate_proj.weight"
        for num_fused in (0, 1):
            draft, warnings = _load(num_fused, routed)
            fused = draft.params["stages.0.mlp.experts.w13_weight"]
            self.assertEqual(len(fused.calls), 1, num_fused)
            _, kwargs = fused.calls[0]
            self.assertEqual(kwargs["expert_id"], 1, num_fused)
            self.assertEqual(warnings, [], num_fused)

    def test_the_widened_mapping_is_what_makes_the_slot_addressable(self):
        """Both halves of the port are load-bearing, stated as arithmetic."""
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        narrow = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=N_ROUTED,
        )
        wide = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=N_ROUTED + 1,
        )
        self.assertNotIn(
            N_ROUTED, {expert_id for _, _, expert_id, _ in narrow}
        )
        self.assertIn(N_ROUTED, {expert_id for _, _, expert_id, _ in wide})


class TestResolverIsShared(unittest.TestCase):
    """The draft head and the target model must not be able to disagree."""

    def _server_args(self, **kwargs):
        defaults = {
            "disable_shared_experts_fusion": False,
            "enforce_shared_experts_fusion": True,
        }
        defaults.update(kwargs)
        return type("_SA", (), defaults)()

    def test_the_target_method_delegates_to_the_shared_resolver(self):
        seen = []

        def spy(config, quant_config=None):
            seen.append((config, quant_config))
            return 7

        target = v4mod.DeepseekV4ForCausalLM.__new__(v4mod.DeepseekV4ForCausalLM)
        target.config = _Config()
        target.quant_config = None
        with patch.object(v4mod, "_resolve_num_fused_shared_experts", spy):
            target.determine_num_fused_shared_experts()
        self.assertEqual(target.num_fused_shared_experts, 7)
        self.assertEqual(len(seen), 1)

    def test_the_draft_head_imports_the_same_resolver(self):
        from sglang.srt.models import deepseek_v4_dspark as dmod

        self.assertIs(
            dmod._resolve_num_fused_shared_experts,
            v4mod._resolve_num_fused_shared_experts,
        )

    def test_the_draft_head_constructor_actually_calls_it(self):
        """The wiring, executed -- not asserted about.

        A resolver the draft head imports but never calls would leave the
        defect exactly where it was, and every loader test above would still
        pass because they set the count directly. The construction is driven
        only as far as the resolve: everything after it needs a real DSpark
        config, so the expected failure downstream is caught and the
        observation is that the resolver ran BEFORE it.
        """
        from sglang.srt.models import deepseek_v4_dspark as dmod

        seen = []

        def spy(config, quant_config=None):
            seen.append(config)
            return 1

        draft = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        with patch.object(dmod, "_resolve_num_fused_shared_experts", spy):
            try:
                draft.__init__(_Config(), None)
            except Exception:
                pass
        self.assertEqual(len(seen), 1, "the draft head never resolved the count")
        self.assertEqual(draft.num_fused_shared_experts, 1)

    def test_enforced_fusion_yields_the_config_count(self):
        with patch.object(
            v4mod, "get_server_args", lambda: self._server_args()
        ), patch.object(v4mod, "is_gguf_quant_config", lambda _: False):
            self.assertEqual(
                v4mod._resolve_num_fused_shared_experts(_Config(), None), 1
            )

    def test_gguf_refuses_enforced_fusion_by_name(self):
        """Fork-specific branch: it must survive the extraction."""
        with patch.object(
            v4mod, "get_server_args", lambda: self._server_args()
        ), patch.object(v4mod, "is_gguf_quant_config", lambda _: True):
            with self.assertRaises(NotImplementedError) as ctx:
                v4mod._resolve_num_fused_shared_experts(_Config(), object())
        self.assertIn("GGUF", str(ctx.exception))

    def test_the_disable_switch_short_circuits(self):
        with patch.object(
            v4mod,
            "get_server_args",
            lambda: self._server_args(disable_shared_experts_fusion=True),
        ):
            self.assertEqual(
                v4mod._resolve_num_fused_shared_experts(_Config(), None), 0
            )

    def test_more_than_one_shared_expert_is_refused_under_enforcement(self):
        class _Two(_Config):
            n_shared_experts = 2

        with patch.object(
            v4mod, "get_server_args", lambda: self._server_args()
        ), patch.object(v4mod, "is_gguf_quant_config", lambda _: False):
            with self.assertRaises(ValueError):
                v4mod._resolve_num_fused_shared_experts(_Two(), None)


if __name__ == "__main__":
    unittest.main()
