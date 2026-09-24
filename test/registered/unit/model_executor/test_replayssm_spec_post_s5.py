"""The spec ring's budget post: runtime, allocation and planner agree (S5).

27B ReplaySSM package, slice S5 (REPLAYSSM_PLAN.md). Under
--enable-linear-replayssm-spec the mamba budget post "speculative intermediate
state" must price what the pool allocates per request row -- the conv verify
windows plus the ring -- instead of D full per-draft states, and the D-group
planner (uneven_perf.PerfCostModel, which --rank-tp-ratio auto and the weg2
launcher's D pricing read) must say the same, or the dry run prices group D
with ~0.4 GiB per rank that the boot then hands to KV. Pinned:

* ``spec_ring_workspace_bytes_per_req`` == the pool's real per-row spec
  allocation (dedup conv windows + ring), at the 27B D ranks' own GDN heads
  (18/6 and 15/5) and head dims;
* the mixin's fractional-D unit is ``D`` itself (the same object) with the
  flag off, and workspace / per_req with it on; a draft-KV producer keeps D;
* ``handle_max_mamba_cache`` on the demand branch (the 27B D's, uneven DCP):
  the post shrinks to capped_reqs x workspace, the pool size is unchanged, and
  every freed byte reaches the KV budget;
* the planner's per-UNIT mirror gives the runtime's unit for one GDN unit,
  byte for byte, is the identity with the flag off, and PlanInputs carries the
  ring length only when the flag is on.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
    _spec_workspace_draft_units,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.uneven_perf import PerfCostModel, PlanInputs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GIB = 1 << 30
KD = VD = 128  # 27B GDN head dims
CONV_K = 4
DRAFT = 8  # DFLASH block on D
RING = 16


def _params(hv, hk, layers):
    conv_dim = (2 * hk * KD) + hv * VD
    shape = Mamba2StateShape(
        conv=[(conv_dim, CONV_K - 1)],
        temporal=(hv, VD, KD),
        intermediate_size=hv * VD,
        conv_dim=conv_dim,
        ssm_state_size=KD,
        num_heads=hv,
        head_dim=VD,
        state_size=KD,
        conv_kernel=CONV_K,
        num_k_heads_per_tp=hk,
    )
    return Mamba2CacheParams(
        shape=shape,
        layers=list(range(layers)),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.bfloat16),
    )


def _runner(spec=True, producer=False, sa=None):
    sa = sa or ServerArgs(model_path="dummy")
    object.__setattr__(sa, "enable_linear_replayssm_spec", spec)
    object.__setattr__(sa, "linear_replayssm_cache_len", RING)
    return SimpleNamespace(
        server_args=sa,
        is_draft_kv_only_producer=producer,
        hybrid_gdn_config=object(),
    )


class _Base(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


class TestWorkspaceIsTheAllocation(_Base):
    def test_per_row_bytes_match_the_pool(self):
        from sglang.srt.mem_cache import memory_pool as mp

        for hv, hk in ((18, 6), (15, 5)):
            with self.subTest(hv=hv, hk=hk):
                params = _params(hv, hk, layers=2)
                # CUDA + linear chain -> the deduplicated conv windows (the
                # CPU default is the dense fallback layout)
                with mock.patch.object(
                    mp, "conv_window_dedup_enabled", lambda *a: True
                ):
                    pool = mp.MambaPool(
                        size=4,
                        spec_state_size=2,
                        cache_params=params,
                        mamba_layer_ids=[0, 1],
                        device="cpu",
                        speculative_num_draft_tokens=DRAFT,
                        speculative_eagle_topk=1,
                        linear_replayssm_cache_len=RING,
                        enable_linear_replayssm_spec=True,
                    )
                c = pool.mamba_cache
                rows = 3
                spec_bytes = sum(
                    t.numel() * t.element_size()
                    for t in pool._intermediate_conv_window_phys
                )
                for name in (
                    "replayssm_d",
                    "replayssm_k",
                    "replayssm_g",
                    "replayssm_rawv",
                    "replayssm_rawk",
                ):
                    t = getattr(c, name)
                    spec_bytes += t.numel() * t.element_size()
                self.assertIsNone(c.intermediate_ssm)
                self.assertEqual(
                    spec_bytes,
                    rows * params.spec_ring_workspace_bytes_per_req(DRAFT, RING),
                )


class TestDraftUnits(_Base):
    def test_flag_off_is_the_same_object(self):
        cfg = SimpleNamespace(mamba2_cache_params=_params(18, 6, 48))
        d = DRAFT
        self.assertIs(_spec_workspace_draft_units(_runner(spec=False), cfg, d), d)
        self.assertEqual(_spec_workspace_draft_units(_runner(), cfg, 0), 0)

    def test_producer_keeps_d(self):
        cfg = SimpleNamespace(mamba2_cache_params=_params(18, 6, 48))
        self.assertIs(
            _spec_workspace_draft_units(_runner(producer=True), cfg, DRAFT), DRAFT
        )

    def test_flag_on_is_workspace_over_state(self):
        p = _params(18, 6, 48)
        cfg = SimpleNamespace(mamba2_cache_params=p)
        got = _spec_workspace_draft_units(_runner(), cfg, DRAFT)
        want = p.spec_ring_workspace_bytes_per_req(DRAFT, RING) / p.mamba_cache_per_req
        self.assertEqual(got, want)
        # 27B D rank 18/6: 13.1 MiB instead of 8 x 28.9 MiB per request row
        self.assertLess(got, 0.5)
        self.assertGreater(got, 0.4)


def _demand_stub(sa, params, spec):
    runner = _runner(spec=spec, sa=sa)
    stub = SimpleNamespace(
        server_args=runner.server_args,
        is_draft_kv_only_producer=False,
        hybrid_gdn_config=object(),
        dp_size=1,
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
        mambaish_config=SimpleNamespace(mamba2_cache_params=params),
    )
    stub._calculate_mamba_ratio = lambda: 5
    stub._auto_mamba_demand_active = lambda: True
    stub._sync_uneven_mamba_cache_size = lambda: None
    stub._auto_mamba_target_concurrency = (
        lambda: ModelRunnerKVCacheMixin._auto_mamba_target_concurrency(stub)
    )
    stub._auto_mamba_demand_size = lambda ratio: (
        ModelRunnerKVCacheMixin._auto_mamba_demand_size(stub, ratio)
    )
    stub._mamba_pool_budget_cost_gb = lambda *a: (
        ModelRunnerKVCacheMixin._mamba_pool_budget_cost_gb(stub, *a)
    )
    stub._fit_mamba_pool_to_budget = lambda *a: (
        ModelRunnerKVCacheMixin._fit_mamba_pool_to_budget(stub, *a)
    )
    return stub


class TestDemandBranchPost(_Base):
    def _run(self, spec):
        params = _params(18, 6, 48)
        sa = ServerArgs(
            model_path="dummy",
            max_running_requests=2,
            speculative_num_draft_tokens=DRAFT,
        )
        object.__setattr__(sa, "max_running_requests_user_set", True)
        stub = _demand_stub(sa, params, spec)
        rest = ModelRunnerKVCacheMixin.handle_max_mamba_cache(stub, 12.0)
        post = stub._mamba_budget_components["speculative intermediate state"]
        return params, sa, rest, post

    def test_ring_post_and_the_bytes_it_frees(self):
        params, sa_off, rest_off, post_off = self._run(False)
        _, sa_on, rest_on, post_on = self._run(True)
        size = sa_on.max_mamba_cache_size
        self.assertEqual(size, sa_off.max_mamba_cache_size)
        capped = min(sa_on.max_running_requests, size // 5)
        self.assertAlmostEqual(
            post_off, capped * params.mamba_cache_per_req * DRAFT / GIB, places=9
        )
        self.assertAlmostEqual(
            post_on,
            capped * params.spec_ring_workspace_bytes_per_req(DRAFT, RING) / GIB,
            places=9,
        )
        # every byte the post gives back reaches the KV budget
        self.assertAlmostEqual(rest_on - rest_off, post_off - post_on, places=9)
        self.assertGreater(post_off - post_on, 0.4)


def _perf_stub(ring):
    return SimpleNamespace(
        replayssm_spec_ring_len=ring,
        gdn_k_dim=KD,
        gdn_v_dim=VD,
        conv_kernel=CONV_K,
        gdn_layers=48,
    )


class TestPlannerMirror(_Base):
    def test_one_unit_is_the_runtime_unit(self):
        hpu = 3  # 27B: 48 v heads over 16 k heads
        state = hpu * VD * KD * 2
        conv = (2 * KD + hpu * VD) * (CONV_K - 1) * 2
        per_req_per_unit = 48 * (state + conv)
        got = PerfCostModel._spec_pad_draft_units(
            _perf_stub(RING), DRAFT, hpu, per_req_per_unit
        )
        one_unit = _params(hpu, 1, 48)
        self.assertEqual(one_unit.mamba_cache_per_req, per_req_per_unit)
        want = _spec_workspace_draft_units(
            _runner(), SimpleNamespace(mamba2_cache_params=one_unit), DRAFT
        )
        self.assertEqual(got, want)

    def test_flag_off_identity(self):
        self.assertIs(
            PerfCostModel._spec_pad_draft_units(_perf_stub(None), DRAFT, 3, 1), DRAFT
        )

    def test_plan_inputs_carry_the_ring_only_when_on(self):
        sa = ServerArgs(model_path="dummy")
        self.assertIsNone(PlanInputs.from_server_args(sa).linear_replayssm_spec_ring_len)
        object.__setattr__(sa, "enable_linear_replayssm_spec", True)
        self.assertEqual(
            PlanInputs.from_server_args(sa).linear_replayssm_spec_ring_len,
            sa.linear_replayssm_cache_len,
        )


if __name__ == "__main__":
    unittest.main()
