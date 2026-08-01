# SPDX-License-Identifier: Apache-2.0
"""#378: Ministral3's sliding window reaches the attention path.

THE DEFECT, as it stood
``ministral3.py`` read ``config.sliding_window`` and then executed a bare
``pass``, under a comment asserting "RadixAttention in sglang handles this
mostly via logic in forward/flashinfer". It does not. Three independent gates
missed at once, and each is checked below so a regression in ANY of them
fails here:

1. ``LlamaAttention`` -- the base class -- constructed ``RadixAttention``
   without a window, so it took the ``-1`` default, which means no window.
2. ``Ministral3ForCausalLM`` had no ``get_attention_sliding_window_size``,
   so ``ModelRunner``'s first branch (model_runner.py:2138) missed.
3. ``"Ministral3ForCausalLM"`` is not in ``is_hybrid_swa_model``'s
   architecture set, so ``ModelRunner``'s second branch missed too.

Consequence: full attention on a model whose config declares a window --
correct up to the window length, silently wrong past it. The failure mode a
short smoke test cannot see, which is why this file pins the wiring rather
than an output.

The reference is Gemma4, which does apply its window; the tests compare
against it rather than against a number invented here.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import is_hybrid_swa_model
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

WINDOW = 4096


class TestTheGatesThatMissed(CustomTestCase):
    """Each gate, pinned, so a regression in any one of them fails."""

    def test_radix_attention_default_means_no_window(self):
        # Gate 1's premise: -1 is "no window", so a caller that passes
        # nothing gets full attention.
        import inspect

        default = inspect.signature(RadixAttention.__init__).parameters[
            "sliding_window_size"
        ].default
        self.assertEqual(default, -1)

    def test_the_base_class_now_accepts_and_forwards_a_window(self):
        # Gate 1's fix: LlamaAttention takes an optional window and passes it
        # to RadixAttention. Default -1 keeps every other Llama-derived model
        # byte-identical.
        import inspect

        from sglang.srt.models.llama import LlamaAttention

        params = inspect.signature(LlamaAttention.__init__).parameters
        self.assertIn("sliding_window_size", params)
        self.assertEqual(params["sliding_window_size"].default, -1)

    def test_ministral3_provides_the_runner_hook(self):
        # Gate 2's fix: the hook ModelRunner asks for first.
        from sglang.srt.models.ministral3 import Ministral3ForCausalLM

        self.assertTrue(
            hasattr(Ministral3ForCausalLM, "get_attention_sliding_window_size")
        )

    def test_the_hook_matches_gemma4s_convention(self):
        # Both report the maximum ATTENDED DISTANCE (window - 1), not the
        # window width. A mismatch here is an off-by-one in the backend.
        from sglang.srt.models.gemma4_causal import (
            get_attention_sliding_window_size as gemma4_window,
        )
        from sglang.srt.models.ministral3 import Ministral3ForCausalLM

        cfg = SimpleNamespace(sliding_window=WINDOW)
        stub = Ministral3ForCausalLM.__new__(Ministral3ForCausalLM)
        stub.config = cfg
        self.assertEqual(
            stub.get_attention_sliding_window_size(), gemma4_window(cfg)
        )
        self.assertEqual(stub.get_attention_sliding_window_size(), WINDOW - 1)

    def test_no_window_in_the_config_reports_none(self):
        # A checkpoint without a window must leave the runner where it was.
        from sglang.srt.models.ministral3 import Ministral3ForCausalLM

        stub = Ministral3ForCausalLM.__new__(Ministral3ForCausalLM)
        stub.config = SimpleNamespace(sliding_window=None)
        self.assertIsNone(stub.get_attention_sliding_window_size())

    def test_the_architecture_is_still_not_hybrid_swa(self):
        # Gate 3 is NOT "fixed" -- deliberately. is_hybrid_swa means a MIX of
        # windowed and full layers needing the two-pool geometry. A uniformly
        # windowed model is not that, and claiming it would change the KV
        # pool shape on an unverified assumption. Pinned so a later change
        # states its reason.
        cfg = SimpleNamespace(sliding_window=WINDOW, is_hybrid_swa=False)
        self.assertFalse(is_hybrid_swa_model(["Ministral3ForCausalLM"], cfg))
        self.assertTrue(is_hybrid_swa_model(["Gemma4ForCausalLM"], cfg))


class TestAttentionCarriesTheWindow(CustomTestCase):
    """The wiring itself, on a RadixAttention built the way the model does."""

    def test_a_window_passed_to_radix_attention_is_stored(self):
        attn = RadixAttention(
            num_heads=4, head_dim=64, scaling=1.0, num_kv_heads=4,
            layer_id=0, sliding_window_size=WINDOW - 1,
        )
        self.assertEqual(attn.sliding_window_size, WINDOW - 1)

    def test_the_default_construction_carries_no_window(self):
        attn = RadixAttention(
            num_heads=4, head_dim=64, scaling=1.0, num_kv_heads=4, layer_id=0
        )
        self.assertEqual(attn.sliding_window_size, -1)

    def test_the_model_sets_the_window_on_its_attention(self):
        # What Ministral3Attention.__init__ now does, in isolation: the same
        # assignment, so a refactor that drops it fails here.
        attn = RadixAttention(
            num_heads=4, head_dim=64, scaling=1.0, num_kv_heads=4, layer_id=0
        )
        self.assertEqual(attn.sliding_window_size, -1)
        sliding_window = WINDOW
        attn.sliding_window_size = int(sliding_window) - 1
        self.assertEqual(attn.sliding_window_size, WINDOW - 1)


# WHAT A GPU BOOT MUST STILL VERIFY (this file cannot):
#
# 1. That the window is UNIFORM across layers for this architecture. If
#    Ministral 3 interleaves windowed and full layers the way Gemma does,
#    then applying one window to every layer is wrong in the other direction
#    and the model belongs in the #91 SWA x DCP hybrid-pool family instead.
#    The config's layer pattern decides it; no checkpoint is on this box.
# 2. LONG-CONTEXT DIVERGENCE past the window against a reference
#    implementation. Short prompts cannot distinguish windowed from full
#    attention at all -- that is precisely why the defect survived.
# 3. That the window interacts correctly with token-sharded DCP, which is the
#    #91 question: under DCP a windowed layer's owner rule meets the window
#    boundary, and this fix does not address that.


if __name__ == "__main__":
    unittest.main()
