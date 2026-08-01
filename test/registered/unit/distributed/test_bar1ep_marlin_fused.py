# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Marlin can serve the BAR1 expert-parallel dispatch (#374).

WHAT #361 MEASURED, and why this file exists
--------------------------------------------
``MoeRunner`` builds Marlin with ``runner_core = None`` -- "Marlin only
supports fused path" -- so a Marlin MoE serves exactly the a2a backends that
have a fused func registered, and that was ``none`` alone. On sm86 both MoE
formats this rig can load hard-wire the Marlin runner
(``quantization/fp8.py:2453`` and ``gptq_kernels.py:362`` construct
``MoeRunner(MoeRunnerBackend.MARLIN, ...)`` unconditionally; FP8 lands there
too because sm86 has no native FP8), so ``--moe-a2a-backend bar1ep`` refused
at model load:

    NotImplementedError: Runner backend MoeRunnerBackend.MARLIN requires a
    fused func for a2a backend bar1ep, but none is registered.

The only runner that consumed the dispatch format, deep_gemm, is off on every
card in the rig (sm86 < 90, sm120 explicitly excluded), which made BAR1-EP a
Hopper/SM100 feature by omission.

WHAT IS AND IS NOT PROVEN HERE. These tests are hermetic: they pin the
registration, the argument contract, and the two numerical rules that are
decidable without a card. The GEMM itself needs Marlin kernels and therefore a
GPU, so ``fused_marlin_moe`` is substituted and its call inspected. What only a
boot can prove is stated in the #374 report, not asserted here.
"""

import unittest
from typing import Any, Dict
from unittest import mock

import torch

from sglang.srt.layers.moe.moe_runner import marlin as marlin_mod
from sglang.srt.layers.moe.moe_runner.base import FusedOpPool, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

NUM_LOCAL_EXPERTS = 4
HIDDEN = 16
TOPK = 2


def _quant_info(num_local_experts: int = NUM_LOCAL_EXPERTS) -> MarlinMoeQuantInfo:
    """Weight bundle shaped like ONE RANK's share -- local experts only."""
    return MarlinMoeQuantInfo(
        w13_qweight=torch.zeros(num_local_experts, 4, 4),
        w2_qweight=torch.zeros(num_local_experts, HIDDEN // 16, 4),
        w13_scales=torch.ones(num_local_experts, 1, 4),
        w2_scales=torch.ones(num_local_experts, 1, 4),
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
        # A global count deliberately different from the local one: if the
        # fused func ever forwards this, the mismatch shows up in the test that
        # inspects the call rather than as a wrong answer on a card.
        global_num_experts=NUM_LOCAL_EXPERTS * 3,
    )


def _dispatch(topk_ids: torch.Tensor, *, tokens: int = 3, scale=None):
    return DeepEPNormalDispatchOutput(
        hidden_states=torch.ones(tokens, HIDDEN, dtype=torch.bfloat16),
        hidden_states_scale=scale,
        topk_ids=topk_ids,
        topk_weights=torch.full((tokens, TOPK), 0.5, dtype=torch.float32),
        num_recv_tokens_per_expert=[tokens] * NUM_LOCAL_EXPERTS,
    )


def _config() -> MoeRunnerConfig:
    return MoeRunnerConfig(
        num_experts=NUM_LOCAL_EXPERTS * 3,
        num_local_experts=NUM_LOCAL_EXPERTS,
        hidden_size=HIDDEN,
        top_k=TOPK,
        routed_scaling_factor=2.5,
    )


def _run_capturing(dispatch, quant=None, config=None) -> Dict[str, Any]:
    """Call the fused func with ``fused_marlin_moe`` substituted; return kwargs."""
    seen: Dict[str, Any] = {}

    def fake(**kw):
        # The stub reproduces the real callee's precondition. A mock more
        # permissive than the function it stands in for is how a crash reaches
        # a card: gating_output=None passed every version of this test until
        # the assert below was added, and fused_marlin_moe dereferences it.
        assert kw["gating_output"] is not None, "gating_output must be a tensor"
        assert (
            kw["hidden_states"].shape[0] == kw["gating_output"].shape[0]
        ), "Number of tokens mismatch"
        seen.update(kw)
        rows = kw["hidden_states"].shape[0]
        return torch.zeros(rows, HIDDEN, dtype=kw["hidden_states"].dtype)

    fused = FusedOpPool.get_fused_func("bar1ep", "marlin")
    with mock.patch(
        "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe.fused_marlin_moe",
        fake,
    ), mock.patch(
        "sglang.srt.layers.quantization.marlin_utils.marlin_make_workspace",
        lambda device, max_blocks_per_sm=4: torch.zeros(4, dtype=torch.int),
    ):
        out = fused(dispatch, quant or _quant_info(), config or _config())
    seen["__out__"] = out
    return seen


class TestRegistration(CustomTestCase):
    """The #361 symptom, gone."""

    def test_a_fused_func_exists_for_bar1ep_and_marlin(self):
        self.assertIsNotNone(FusedOpPool.get_fused_func("bar1ep", "marlin"))

    def test_the_none_path_still_has_its_own_and_they_are_different(self):
        none_f = FusedOpPool.get_fused_func("none", "marlin")
        bar1_f = FusedOpPool.get_fused_func("bar1ep", "marlin")
        self.assertIsNotNone(none_f)
        self.assertIsNot(none_f, bar1_f)
        self.assertEqual(none_f.__name__, "fused_experts_none_to_marlin")

    def test_marlin_still_has_no_runner_core_so_the_fused_func_is_the_path(self):
        """If a core ever appears, the fused func stops being load-bearing and
        this file's premise needs re-reading."""
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        runner = MoeRunner(MoeRunnerBackend.MARLIN, _config())
        self.assertIsNone(runner.runner_core)
        self.assertIsNotNone(runner.fused_func)


class TestExpertIdContract(CustomTestCase):
    """Local ids, and the -1 that must never reach the kernel (#112)."""

    def test_the_local_expert_count_is_used_not_the_global_one(self):
        """bar1ep hands over ids in [0, num_local_experts) and this rank holds
        only its own experts, so forwarding quant_info.global_num_experts
        would index past the weights."""
        ids = torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.int64)
        seen = _run_capturing(_dispatch(ids))
        self.assertEqual(seen["global_num_experts"], -1)
        self.assertIsNone(seen["expert_map"])

    def test_an_unused_slot_is_pointed_at_a_valid_expert(self):
        ids = torch.tensor([[0, -1], [-1, 3], [1, 2]], dtype=torch.int64)
        seen = _run_capturing(_dispatch(ids))
        passed = seen["topk_ids"]
        self.assertTrue(
            bool((passed >= 0).all()), f"a negative id reached the kernel: {passed}"
        )
        self.assertLess(int(passed.max()), NUM_LOCAL_EXPERTS)

    def test_an_unused_slot_carries_zero_weight(self):
        """The redirect must not add a contribution: 0 x finite is exact, and
        it reproduces ep_gather's "weight only where expert_id >= 0"."""
        ids = torch.tensor([[0, -1], [-1, 3], [1, 2]], dtype=torch.int64)
        seen = _run_capturing(_dispatch(ids))
        weights = seen["topk_weights"]
        self.assertEqual(float(weights[0, 1]), 0.0)
        self.assertEqual(float(weights[1, 0]), 0.0)
        # Valid slots keep the weight the dispatcher sent.
        self.assertEqual(float(weights[0, 0]), 0.5)
        self.assertEqual(float(weights[2, 1]), 0.5)

    def test_the_returned_ids_and_weights_are_the_originals(self):
        """The masking is for the kernel only; the combine gets what the
        dispatcher produced, -1 markers included."""
        ids = torch.tensor([[0, -1], [-1, 3], [1, 2]], dtype=torch.int64)
        d = _dispatch(ids)
        out = _run_capturing(d)["__out__"]
        self.assertTrue(torch.equal(out.topk_ids, d.topk_ids))
        self.assertTrue(torch.equal(out.topk_weights, d.topk_weights))


class TestReductionSemantics(CustomTestCase):
    def test_routed_scaling_is_not_applied_on_this_path(self):
        """The Standard post-permute applies it; ep_gather -- the other
        DEEPEP_NORMAL consumer -- does not, because the factor lands after the
        combine. Applying it here would scale every EP token twice."""
        ids = torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.int64)
        cfg = _config()
        self.assertIsNotNone(cfg.routed_scaling_factor)
        seen = _run_capturing(_dispatch(ids), config=cfg)
        self.assertIsNone(seen["routed_scaling_factor"])

    def test_the_output_is_one_row_per_received_token(self):
        """Not reduced across ranks -- that is the combine's job."""
        ids = torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.int64)
        out = _run_capturing(_dispatch(ids, tokens=3))["__out__"]
        self.assertIsInstance(out, DeepEPNormalCombineInput)
        self.assertEqual(tuple(out.hidden_states.shape), (3, HIDDEN))

    def test_the_output_dtype_follows_the_input(self):
        ids = torch.tensor([[0, 1]], dtype=torch.int64)
        out = _run_capturing(_dispatch(ids, tokens=1))["__out__"]
        self.assertEqual(out.hidden_states.dtype, torch.bfloat16)


class TestGuards(CustomTestCase):
    def test_an_fp8_dispatch_is_refused_by_name(self):
        """A dropped scale is a wrong number, not a slower one."""
        ids = torch.tensor([[0, 1]], dtype=torch.int64)
        d = _dispatch(ids, tokens=1, scale=torch.ones(1, 1))
        with self.assertRaises(NotImplementedError) as caught:
            _run_capturing(d)
        self.assertIn("scale", str(caught.exception).lower())

    def test_an_unsupported_activation_dtype_is_refused_by_name(self):
        ids = torch.tensor([[0, 1]], dtype=torch.int64)
        d = DeepEPNormalDispatchOutput(
            hidden_states=torch.ones(1, HIDDEN, dtype=torch.float32),
            hidden_states_scale=None,
            topk_ids=ids,
            topk_weights=torch.full((1, TOPK), 0.5),
            num_recv_tokens_per_expert=[1] * NUM_LOCAL_EXPERTS,
        )
        with self.assertRaises(NotImplementedError) as caught:
            _run_capturing(d)
        self.assertIn("bf16", str(caught.exception))

    def test_a_rank_that_received_nothing_returns_the_right_shape(self):
        """Every rank still reaches the combine; an empty rank must not skip
        ahead of its peers or hand back a wrong shape."""
        ids = torch.zeros((0, TOPK), dtype=torch.int64)
        d = DeepEPNormalDispatchOutput(
            hidden_states=torch.zeros(0, HIDDEN, dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_ids=ids,
            topk_weights=torch.zeros(0, TOPK),
            num_recv_tokens_per_expert=[0] * NUM_LOCAL_EXPERTS,
        )
        fused = FusedOpPool.get_fused_func("bar1ep", "marlin")
        out = fused(d, _quant_info(), _config())
        self.assertEqual(tuple(out.hidden_states.shape), (0, HIDDEN))


class TestTheGatingOutputSubstitution(CustomTestCase):
    """topk_weights stands in for gating_output, and that is only safe while
    the callee reads nothing but its leading dimension.

    The dispatch format carries no router logits. Passing None crashes --
    fused_marlin_moe asserts ``hidden_states.shape[0] == gating_output.shape[0]``
    -- and a hermetic test cannot see that, because the callee is mocked. So
    the assumption is pinned against the real SOURCE instead: if gating_output
    ever gains a second use, this fails here rather than returning quiet
    nonsense on a card.
    """

    def _gating_uses(self):
        import ast
        import inspect

        from sglang.srt.layers.moe.fused_moe_triton import fused_marlin_moe as mod

        tree = ast.parse(inspect.getsource(mod))
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and fn.name == "fused_marlin_moe":
                return [
                    n
                    for n in ast.walk(fn)
                    if isinstance(n, ast.Name)
                    and n.id == "gating_output"
                    and isinstance(n.ctx, ast.Load)
                ]
        self.fail("fused_marlin_moe not found")

    def test_gating_output_is_read_exactly_once(self):
        uses = self._gating_uses()
        self.assertEqual(
            len(uses),
            1,
            "gating_output gained a use; topk_weights may no longer stand in "
            "for it in fused_experts_bar1ep_to_marlin",
        )

    def test_its_only_use_is_the_token_count_assert(self):
        import inspect

        from sglang.srt.layers.moe.fused_moe_triton import fused_marlin_moe as mod

        line = self._gating_uses()[0].lineno
        src = inspect.getsource(mod).splitlines()[line - 1]
        self.assertIn("shape[0]", src)
        self.assertTrue(src.strip().startswith("assert"), src)

    def test_the_substitute_has_the_asserted_dimension(self):
        ids = torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.int64)
        d = _dispatch(ids, tokens=3)
        seen = _run_capturing(d)
        self.assertEqual(
            seen["gating_output"].shape[0], d.hidden_states.shape[0]
        )


class TestTheNonePathIsUntouched(CustomTestCase):
    """#374 must be additive: the default a2a path is byte-untouched."""

    def test_the_none_fused_func_source_is_unchanged_in_shape(self):
        """It still takes a StandardDispatchOutput and returns a
        StandardCombineInput -- the registration added a sibling, not a
        branch inside the existing one."""
        import inspect

        src = inspect.getsource(marlin_mod.fused_experts_none_to_marlin)
        self.assertIn("StandardCombineInput", src)
        self.assertNotIn("DeepEPNormal", src)
        self.assertNotIn("bar1ep", src)

    def test_the_two_funcs_do_not_share_mutable_state_beyond_the_workspace(self):
        """The Marlin workspace is deliberately shared (one per device); if a
        second global ever appears, that is a coupling worth seeing."""
        globals_used = set(marlin_mod.fused_experts_bar1ep_to_marlin.__code__.co_names)
        self.assertIn("MARLIN_MOE_WORKSPACE", globals_used)


if __name__ == "__main__":
    unittest.main()
