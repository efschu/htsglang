# SPDX-License-Identifier: Apache-2.0
"""#138: adaptive draft length on MultiLayerEagleWorkerV2 -- DELIVERED, pinned.

The determination found #138 already implemented, and the code says so itself
(``adaptive_spec_params.py:225-231``):

    enable_multi_layer_eagle used to be rejected here ("MultiLayerEagleWorkerV2
    does not implement adaptive"). It does now (#138): the worker implements the
    AdaptiveSpecWorker protocol, the ladder ceiling is bounded by the loaded MTP
    layer count, and MultiLayerEagleWorkerV2._assert_adaptive_supported enforces
    the multi-layer-specific constraints ... with a hard error instead of a
    silent fallback.

These pins exist because the failure mode of a delivered-then-regressed feature
here is SILENT: if the refusal were reinstated, or the protocol methods dropped,
adaptive would simply stop applying on this worker with no error -- the
#505a silent-ignore class. So the delivery is pinned from three directions:
the gate must NOT name the multi-layer worker, the worker must keep satisfying
the protocol, and its own constraint check must keep RAISING rather than
falling back.
"""

import inspect
import unittest


class TestTheGateNoLongerRefusesMultiLayer(unittest.TestCase):
    def test_multi_layer_is_not_an_unsupported_reason(self):
        from sglang.srt.speculative import adaptive_spec_params as asp

        src = inspect.getsource(asp.adaptive_unsupported_reason)
        # The rejection must not be reachable: any mention is historical, in a
        # comment, never a returned reason.
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertNotIn(
                "enable_multi_layer_eagle",
                stripped,
                "the multi-layer refusal is back in live code -- #138 regressed",
            )

    def test_the_gate_still_refuses_what_it_should(self):
        """The delivery did not open the gate generally."""
        from sglang.srt.speculative import adaptive_spec_params as asp

        src = inspect.getsource(asp.adaptive_unsupported_reason)
        for still_refused in (
            "enable_two_batch_overlap",
            "enable_pdmux",
            "enable_dp_attention",
            "speculative_eagle_topk",
        ):
            self.assertIn(still_refused, src)


class TestTheWorkerSatisfiesTheProtocol(unittest.TestCase):
    """If these methods vanish, adaptive silently stops applying."""

    def test_it_implements_both_protocol_methods(self):
        from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
            MultiLayerEagleWorkerV2,
        )

        for method in ("build_adaptive_runtime_state", "apply_runtime_state"):
            self.assertTrue(
                callable(getattr(MultiLayerEagleWorkerV2, method, None)),
                f"MultiLayerEagleWorkerV2 lost {method} -- adaptive would stop "
                "applying with no error (#505a silent-ignore class)",
            )

    def test_the_protocol_itself_still_declares_them(self):
        from sglang.srt.speculative.adaptive_runtime_state import AdaptiveSpecWorker

        src = inspect.getsource(AdaptiveSpecWorker)
        self.assertIn("build_adaptive_runtime_state", src)
        self.assertIn("apply_runtime_state", src)


class TestItRaisesRatherThanFallingBack(unittest.TestCase):
    """A silent fallback here would be the #505a defect the delivery avoided."""

    def test_the_constraint_check_exists_and_raises(self):
        from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
            MultiLayerEagleWorkerV2,
        )

        src = inspect.getsource(MultiLayerEagleWorkerV2._assert_adaptive_supported)
        self.assertIn("raise ValueError", src)
        self.assertNotIn("return None  # fall back", src)

    def test_it_refuses_candidate_steps_below_one(self):
        from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
            MultiLayerEagleWorkerV2,
        )

        src = inspect.getsource(MultiLayerEagleWorkerV2._assert_adaptive_supported)
        self.assertIn("candidate steps >= 1", src)

    def test_the_refusal_names_the_remedy(self):
        """A refusal that does not say what to set sends the operator hunting."""
        from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
            MultiLayerEagleWorkerV2,
        )

        src = inspect.getsource(MultiLayerEagleWorkerV2._assert_adaptive_supported)
        self.assertIn("--speculative-adaptive-config", src)


if __name__ == "__main__":
    unittest.main()
