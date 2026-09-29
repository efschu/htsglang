# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 stage 1b (F26): the P stage hull's spec and gate (CPU).

DANGER DIRECTIONS guarded here:
* the gate is P-only and env-only: D, an unset env, or a draft worker never
  take the hull path (the model runner hook also excludes the draft);
* D's vectors are NEVER derived: a missing base vector is refused by name
  (a vector one unit off would make the "shared" part a different shard);
* the plan is one segment per D rank, so the FAST ratio equals D's ratio
  exactly (nesting by construction) for every family;
* the stage->D-rank map must be one-to-one onto D's ranks.
"""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.model_executor import dual_stage_hull as H
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ENV = {H.DUAL_SHARE_ENV: "1", "SGLANG_WEG2_GROUP": "P", H.D_TP_RATIO_ENV: "58,25,25",
       H.D_FAMILIES_ENV: "mlp=98,19,19; vocab=1,1,1"}


class DualStageHullSpec(CustomTestCase):
    def test_gate_is_p_only(self):
        self.assertTrue(H.dual_share_armed(ENV))
        self.assertFalse(H.dual_share_armed({**ENV, "SGLANG_WEG2_GROUP": "D"}))
        self.assertFalse(H.dual_share_armed({**ENV, H.DUAL_SHARE_ENV: ""}))
        self.assertFalse(H.dual_share_armed({}))

    def test_spec_and_plan_nest_exactly(self):
        spec = H.DualShareSpec.from_env(ENV, pp_size=3)
        self.assertEqual(spec.tp_ratio, (58, 25, 25))
        self.assertEqual(dict(spec.families), {"mlp": (98, 19, 19), "vocab": (1, 1, 1)})
        self.assertEqual(spec.d_rank_of_stage, (0, 1, 2))
        plan = spec.nested_plan()
        self.assertEqual(plan.fast_size, 3)
        self.assertEqual(plan.fast_ratio, (58, 25, 25))
        self.assertEqual(dict(plan.fast_family_ratios), {"mlp": (98, 19, 19), "vocab": (1, 1, 1)})
        self.assertEqual(plan.shared_fast_ranks, (0, 1, 2))

    def test_stage_map(self):
        spec = H.DualShareSpec.from_env({**ENV, H.D_RANK_OF_STAGE_ENV: "1,0,2"}, pp_size=3)
        self.assertEqual([spec.d_rank_on_stage(s) for s in range(3)], [1, 0, 2])
        with self.assertRaises(H.DualShareError):
            spec.d_rank_on_stage(3)

    def test_refusals(self):
        bad = (
            {**ENV, H.D_TP_RATIO_ENV: ""},
            {**ENV, H.D_TP_RATIO_ENV: "58,x,25"},
            {**ENV, H.D_FAMILIES_ENV: "mlp=98,19"},
            {**ENV, H.D_FAMILIES_ENV: "attn=1,1,1"},
            {**ENV, H.D_RANK_OF_STAGE_ENV: "0,0,2"},
            {**ENV, H.D_RANK_OF_STAGE_ENV: "0,1"},
        )
        for env in bad:
            with self.assertRaises(H.DualShareError, msg=str(env)):
                H.DualShareSpec.from_env(env, pp_size=3)

    def test_model_runner_hook_is_gated_and_skips_the_draft(self):
        from sglang.srt.model_executor import model_runner as MR

        src = inspect.getsource(MR.ModelRunner.load_model)
        self.assertIn("elif not self.is_draft_worker and _dual_share_on():", src)
        self.assertIn("self.model = build_dual_stage_model(self)", src)
        env0 = os.environ.pop(H.DUAL_SHARE_ENV, None)
        try:
            self.assertFalse(MR._dual_share_on())
        finally:
            if env0 is not None:
                os.environ[H.DUAL_SHARE_ENV] = env0
