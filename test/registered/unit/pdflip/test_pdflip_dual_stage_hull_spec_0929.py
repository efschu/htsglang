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

from flliper.srt.model_executor import dual_stage_hull as H
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ENV = {H.DUAL_SHARE_ENV: "1", "FLLIPER_PDFLIP_GROUP": "P", H.D_TP_RATIO_ENV: "58,25,25",
       H.D_FAMILIES_ENV: "mlp=98,19,19; vocab=1,1,1"}


class DualStageHullSpec(CustomTestCase):
    def test_gate_is_p_only(self):
        self.assertTrue(H.dual_share_armed(ENV))
        self.assertFalse(H.dual_share_armed({**ENV, "FLLIPER_PDFLIP_GROUP": "D"}))
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
        from flliper.srt.model_executor import model_runner as MR

        src = inspect.getsource(MR.ModelRunner.load_model)
        self.assertIn("elif not self.is_draft_worker and _dual_share_on():", src)
        self.assertIn("self.model = build_dual_stage_model(self)", src)
        init_src = inspect.getsource(MR.ModelRunner.init_torch_distributed)
        self.assertLess(init_src.index("wait_for_d_before_load(self)"),
                        init_src.index("pre_model_load_memory = get_available_gpu_memory("))
        env0 = os.environ.pop(H.DUAL_SHARE_ENV, None)
        try:
            self.assertFalse(MR._dual_share_on())
        finally:
            if env0 is not None:
                os.environ[H.DUAL_SHARE_ENV] = env0


class DRatiosTravelWithTheImage(CustomTestCase):
    """D resolves its vectors at runtime (auto / d-reshard); it publishes the
    INSTALLED ones next to its union socket and P reads exactly those."""

    def test_roundtrip(self):
        import tempfile

        from flliper.srt.distributed.utils import scoped_tp_partition_ratios
        from flliper.srt.pdflip.union_arena_bind import write_d_ratios

        with tempfile.TemporaryDirectory() as d:
            with scoped_tp_partition_ratios([58, 25, 25], {"mlp": [98, 19, 19]}):
                write_d_ratios(d)
            old = {k: os.environ.pop(k, None) for k in (H.D_TP_RATIO_ENV, H.D_FAMILIES_ENV)}
            try:
                env = H._env_with_d_ratios(d)
            finally:
                for k, v in old.items():
                    if v is not None:
                        os.environ[k] = v
            spec = H.DualShareSpec.from_env(env, pp_size=3)
            self.assertEqual(spec.tp_ratio, (58, 25, 25))
            self.assertEqual(dict(spec.families), {"mlp": (98, 19, 19)})

    def test_an_explicit_env_wins_and_a_missing_file_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            os.environ[H.D_TP_RATIO_ENV] = "2,1,1"
            try:
                self.assertEqual(H._env_with_d_ratios(d)[H.D_TP_RATIO_ENV], "2,1,1")
            finally:
                os.environ.pop(H.D_TP_RATIO_ENV, None)
            with self.assertRaises(H.DualShareError):
                H._env_with_d_ratios(d)


class ThreeShardShellsAreTheMonolith(CustomTestCase):
    """The P-stage algebra: shells over D's THREE shards (one per D rank, uneven
    [2,1,1] over 8 units) reproduce the full-width linear -- column parts by
    concat (per sub-output for merged gate|up), row parts by the local sum.
    The middle rank is the 'local' one on the 5090's neighbour card; the shell
    does not care which part is shared."""

    def test_merged_column_and_row(self):
        import torch

        from flliper.srt.distributed.utils import partition_sizes, scoped_tp_partition_ratios
        from flliper.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
        from flliper.srt.model_executor.dual_group_lane import (
            LaneColumnParallelShell,
            LaneRowParallelShell,
        )

        torch.manual_seed(3)
        H, SUB, UNITS, BIG = 32, 128, 8, [2, 1, 1]
        gate, up, down = torch.randn(SUB, H), torch.randn(SUB, H), torch.randn(H, SUB)
        cols, rows = [], []
        sizes = partition_sizes(SUB, BIG, UNITS)
        for r in range(3):
            off = sum(sizes[:r])
            with scoped_tp_partition_ratios(BIG):
                c = MergedColumnParallelLinear(H, [SUB, SUB], bias=False, params_dtype=torch.float32,
                                               tp_rank=r, tp_size=3, tp_units=UNITS)
                c.weight.data.copy_(torch.cat([gate[off:off + sizes[r]], up[off:off + sizes[r]]]))
                w = RowParallelLinear(SUB, H, bias=False, input_is_parallel=True, reduce_results=False,
                                      params_dtype=torch.float32, tp_rank=r, tp_size=3, tp_units=UNITS)
                w.weight_loader(w.weight, down)
            cols.append(c)
            rows.append(w)
        x = torch.randn(5, H)
        gu, _ = LaneColumnParallelShell(cols)(x)
        torch.testing.assert_close(gu, x @ torch.cat([gate, up]).t(), rtol=1e-5, atol=1e-5)
        h = torch.randn(5, SUB)
        out, _ = LaneRowParallelShell(rows)(h)
        torch.testing.assert_close(out, h @ down.t(), rtol=1e-4, atol=1e-4)
        self.assertEqual([p.weight.shape[0] for p in cols], [2 * s for s in sizes])
