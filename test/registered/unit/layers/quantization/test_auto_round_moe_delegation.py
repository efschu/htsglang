# SPDX-License-Identifier: Apache-2.0
"""auto-round GPTQ MoE: the MoeWNA16 delegation must be able to reach Marlin.

Port of upstream sglang commit ``00cdd4b85f`` (PR #33271). Two defects in
``AutoRoundConfig.apply_gptq_quant_layer``'s FusedMoE branch:

1. the config dict handed to ``MoeWNA16Config.from_config`` omitted
   ``desc_act``, and ``GPTQMarlinConfig.is_gptq_marlin_compatible`` treats a
   MISSING key as ineligible -- ``if num_bits is None or group_size is None or
   sym is None or desc_act is None: return False``
   (``gptq/gptq.py:575``). The delegation therefore always resolved
   ``use_marlin=False`` inside MoeWNA16 (``moe_wna16.py:98``) and landed on the
   Triton runner, while the same checkpoint's DENSE layers ran on Marlin;
2. the ``use_marlin=False`` arm returned ``GPTQMarlinMoEMethod(
   quant_args_marlin)``, and ``quant_args_marlin`` is only bound in the
   ``use_marlin`` branch -- a latent ``NameError``.

Both are pinned here without a checkpoint or a device: the eligibility
predicate is a pure dict test, and the delegated config is observable by
recording what ``MoeWNA16Config.from_config`` receives. Only auto-round
checkpoints reach this code; plain GPTQ has its own path.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import sglang.srt.layers.quantization.gptq.gptq as gptq_mod
import sglang.srt.layers.quantization.marlin_utils as marlin_utils
from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig


def _config(**overrides):
    base = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "lm_head": False,
    }
    base.update(overrides)
    return base


class TestEligibilityPredicate(unittest.TestCase):
    """The gate the missing key was tripping, read at its source.

    The predicate's LAST term calls ``check_marlin_supported``, which reads the
    device capability and is always False on a machine with no visible GPU.
    That term is stubbed True here so the KEY-PRESENCE term - the one the port
    is about - is what the assertions below discriminate on. Everything before
    it, including the ``desc_act is None`` test at ``gptq/gptq.py:575``, is the
    real code.
    """

    def setUp(self):
        patcher = patch.object(gptq_mod, "check_marlin_supported", lambda **kw: True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_missing_desc_act_is_ineligible(self):
        without = _config()
        del without["desc_act"]
        self.assertFalse(GPTQMarlinConfig.is_gptq_marlin_compatible(without))

    def test_the_same_config_with_desc_act_false_is_eligible(self):
        """Spread precondition: the predicate CAN say yes here.

        Without this arm the test above would also pass on an instrument that
        answers False to everything, and it would prove nothing.
        """
        self.assertTrue(GPTQMarlinConfig.is_gptq_marlin_compatible(_config()))

    def test_desc_act_true_stays_eligible_too(self):
        """The key's PRESENCE is the gate, not its value."""
        self.assertTrue(
            GPTQMarlinConfig.is_gptq_marlin_compatible(_config(desc_act=True))
        )


class TestDelegatedConfig(unittest.TestCase):
    """What apply_gptq_quant_layer actually hands to MoeWNA16."""

    def _delegate(self, use_marlin: bool):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.quantization import auto_round as ar

        seen = {}

        class _FakeMoeWNA16Config:
            @staticmethod
            def from_config(config):
                seen["config"] = dict(config)

                class _Resolved:
                    @staticmethod
                    def get_quant_method(layer, prefix):
                        return "delegated"

                return _Resolved()

        cfg = ar.AutoRoundConfig.__new__(ar.AutoRoundConfig)
        cfg.weight_bits = 4
        cfg.group_size = 128
        cfg.sym = True
        cfg.packing_format = "auto_round:auto_gptq"
        cfg.backend = "auto"
        cfg.extra_config = None
        # Per-layer bit/group resolution is a different mechanism with its own
        # tests; pin it so this file observes only the delegation.
        cfg.get_layer_config = lambda layer, prefix: (4, 128, True)

        layer = FusedMoE.__new__(FusedMoE)
        import sglang.srt.layers.quantization.moe_wna16 as wna

        # Both helpers are imported INSIDE apply_gptq_quant_layer
        # (auto_round.py:328-331), so they are patched at their source module.
        with patch.object(wna, "MoeWNA16Config", _FakeMoeWNA16Config), patch.object(
            marlin_utils, "check_marlin_supported", lambda *a, **k: use_marlin
        ), patch.object(
            marlin_utils, "check_moe_marlin_supports_layer", lambda *a, **k: use_marlin
        ):
            result = ar.AutoRoundConfig.apply_gptq_quant_layer(cfg, layer, "prefix")
        return result, seen.get("config")

    def test_the_delegated_config_carries_desc_act(self):
        result, config = self._delegate(use_marlin=True)
        self.assertEqual(result, "delegated")
        self.assertIsNotNone(config, "MoeWNA16Config.from_config was never called")
        self.assertIn("desc_act", config)
        self.assertIs(config["desc_act"], False)

    def test_the_delegated_config_is_marlin_eligible(self):
        """The end the port is about: what MoeWNA16 will decide with it."""
        _, config = self._delegate(use_marlin=True)
        with patch.object(gptq_mod, "check_marlin_supported", lambda **kw: True):
            self.assertTrue(GPTQMarlinConfig.is_gptq_marlin_compatible(config))
            without = dict(config)
            del without["desc_act"]
            self.assertFalse(
                GPTQMarlinConfig.is_gptq_marlin_compatible(without),
                "the pre-port config shape must still be the ineligible one",
            )

    def test_the_non_marlin_arm_delegates_instead_of_raising(self):
        """The latent NameError: this call used to reach an unbound name."""
        result, config = self._delegate(use_marlin=False)
        self.assertEqual(result, "delegated")
        self.assertIn("desc_act", config)

    def test_both_arms_delegate_the_same_config(self):
        """One delegation, not two: MoeWNA16 owns the Marlin decision now."""
        _, marlin = self._delegate(use_marlin=True)
        _, triton = self._delegate(use_marlin=False)
        self.assertEqual(marlin, triton)


if __name__ == "__main__":
    unittest.main()
