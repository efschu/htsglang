"""CPU unit tests for the #201 slice 3 world sizing rules.

A pipeline's KV/token ceiling is an ADMISSION currency: a request's tokens
occupy KV on every stage, so the ceiling must be the world minimum over the
per-stage capacities -- and nothing rank-local may lower a stage below the
agreed value afterwards. Three rule families are covered, each with its
falsifier (a planted violation must fail):

* cap-before-reduce -- the #79/#90 hybrid ceilings fold in BEFORE the world
  MIN under PP, and the world-agreement check fires on divergence;
* stage-local mamba budget arithmetic -- handle_max_mamba_cache charges only
  this stage's linear layers (Teil 2 par. 6.2 defect 1), with the sentinel
  for a stage that has none;
* the #188 measured-budget registry is stage-keyed under PP.

No GPU, no distributed init: collectives are patched, the runner is a stub
(same pattern as test_uneven_tp_memory.py).
"""

import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _install_sgl_kernel_stub():
    if importlib.util.find_spec("sgl_kernel") is not None:
        return

    def _make(name, pkg=False):
        mod = types.ModuleType(name)
        if pkg:
            mod.__path__ = []

        def _getattr(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return lambda *a, **k: None

        mod.__getattr__ = _getattr
        sys.modules.setdefault(name, mod)

    _make("sgl_kernel", pkg=True)
    _make("sgl_kernel.quantization")
    _make("sgl_kernel.kvcacheio")


_install_sgl_kernel_stub()

import torch  # noqa: E402

import sglang.srt.model_executor.model_runner_kv_cache_mixin as mixin  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

MiB = 1 << 20
GiB = 1 << 30


class FakeWorld:
    """Patches the world group and the two collectives the sizing chain
    uses. all_reduce(MIN) folds in a scripted `other_capacity`; the gather
    returns this rank's payload plus scripted peer payloads."""

    def __init__(self, world_size, other_capacity=None, peer_payloads=None):
        self.world_size = world_size
        self.other_capacity = other_capacity
        self.peer_payloads = peer_payloads or []
        self.calls = []

    def __enter__(self):
        fake_group = SimpleNamespace(world_size=self.world_size, cpu_group=object())
        self._patches = [
            patch.object(mixin, "get_world_group", return_value=fake_group),
            patch("torch.distributed.all_reduce", side_effect=self._all_reduce),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=self._all_gather_object,
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False

    def _all_reduce(self, tensor, op=None, group=None):
        self.calls.append("reduce")
        if self.other_capacity is not None:
            tensor.copy_(torch.minimum(tensor, torch.tensor(self.other_capacity)))

    def _all_gather_object(self, out, payload, group=None):
        self.calls.append("gather")
        filled = [payload] + list(self.peer_payloads)
        for i in range(len(out)):
            out[i] = filled[i % len(filled)]


def make_constraint_stub(*, pp_size, hybrid_cap=None, user_limit=None, cap_log=None):
    """Stub runner for _apply_token_constraints. The cap hook records its
    application order relative to the collectives."""
    cap_log = cap_log if cap_log is not None else []

    def apply_cap(tc, cap, kind="mamba"):
        cap_log.append("cap")
        if cap is None or tc <= cap:
            return tc
        return cap

    stub = SimpleNamespace(
        server_args=SimpleNamespace(
            max_total_tokens=user_limit,
            uneven_memory_budgets_active=lambda: False,
        ),
        pp_size=pp_size,
        pp_rank=0,
        dcp_size=1,
        tp_rank=0,
        _hybrid_kv_token_cap=lambda: hybrid_cap,
        _swa_hybrid_kv_token_cap=lambda: None,
        _apply_hybrid_kv_token_cap=apply_cap,
    )
    stub._assert_pp_world_kv_capacity_agreement = (
        lambda tc: mixin.ModelRunnerKVCacheMixin._assert_pp_world_kv_capacity_agreement(
            stub, tc
        )
    )
    return stub, cap_log


class TestCapBeforeWorldMin(CustomTestCase):
    """#79/#90 caps fold in BEFORE the world MIN under PP."""

    def test_stage_divergent_cap_binds_world(self):
        # This stage: huge profiled capacity, hybrid cap 50k. Peer stage
        # already reduced its capped value 100k into the MIN. The agreed
        # world value must be 50k -- the cap participates in the MIN.
        stub, _ = make_constraint_stub(pp_size=2, hybrid_cap=50_000)
        with FakeWorld(
            2, other_capacity=100_000, peer_payloads=[(1, 0, 50_000)]
        ) as world:
            result = mixin.ModelRunnerKVCacheMixin._apply_token_constraints(
                stub, 200_000
            )
        self.assertEqual(result, 50_000)
        self.assertIn("reduce", world.calls)

    def test_cap_applied_before_reduce_under_pp(self):
        cap_log = []
        stub, _ = make_constraint_stub(pp_size=2, hybrid_cap=50_000, cap_log=cap_log)
        order = []

        def apply_cap(tc, cap, kind="mamba"):
            order.append("cap")
            return tc if cap is None or tc <= cap else cap

        stub._apply_hybrid_kv_token_cap = apply_cap
        with FakeWorld(2, other_capacity=60_000, peer_payloads=[(1, 0, 50_000)]) as w:
            # Interleave the collective calls into the same order log.
            base_reduce = w._all_reduce

            def traced_reduce(*a, **k):
                order.append("reduce")
                base_reduce(*a, **k)

            with patch("torch.distributed.all_reduce", side_effect=traced_reduce):
                mixin.ModelRunnerKVCacheMixin._apply_token_constraints(stub, 200_000)
        self.assertGreater(len(order), 1)
        self.assertEqual(order[0], "cap", f"cap must precede the reduce: {order}")
        self.assertIn("reduce", order)

    def test_pp1_keeps_stock_order(self):
        # pp_size == 1 with uneven budgets: the cap stays AFTER the reduce
        # (stock order, byte-identical) and no agreement gather runs.
        order = []

        def apply_cap(tc, cap, kind="mamba"):
            order.append("cap")
            return tc if cap is None or tc <= cap else cap

        stub, _ = make_constraint_stub(pp_size=1, hybrid_cap=800)
        stub._apply_hybrid_kv_token_cap = apply_cap
        stub.server_args.uneven_memory_budgets_active = lambda: True
        with FakeWorld(2, other_capacity=900) as w:
            base_reduce = w._all_reduce

            def traced_reduce(*a, **k):
                order.append("reduce")
                base_reduce(*a, **k)

            with patch("torch.distributed.all_reduce", side_effect=traced_reduce):
                result = mixin.ModelRunnerKVCacheMixin._apply_token_constraints(
                    stub, 1000
                )
            self.assertEqual(result, 800)
            self.assertEqual(order, ["reduce", "cap"])
            self.assertNotIn("gather", w.calls)


class TestWorldAgreementCheck(CustomTestCase):
    """The planted wrong world-min MUST fail (falsifier duty)."""

    def test_divergence_raises(self):
        stub, _ = make_constraint_stub(pp_size=2)
        with FakeWorld(2, peer_payloads=[(1, 0, 40_000)]):
            with self.assertRaisesRegex(RuntimeError, "world agreement violated"):
                mixin.ModelRunnerKVCacheMixin._assert_pp_world_kv_capacity_agreement(
                    stub, 50_000
                )

    def test_agreement_passes(self):
        stub, _ = make_constraint_stub(pp_size=2)
        with FakeWorld(2, peer_payloads=[(1, 0, 50_000)]):
            mixin.ModelRunnerKVCacheMixin._assert_pp_world_kv_capacity_agreement(
                stub, 50_000
            )

    def test_check_runs_at_constraint_exit_under_pp(self):
        # End-to-end through _apply_token_constraints: a peer that (by the
        # planted fault) ends on a different value fails the boot.
        stub, _ = make_constraint_stub(pp_size=2)
        with FakeWorld(2, other_capacity=100_000, peer_payloads=[(1, 0, 90_000)]):
            with self.assertRaisesRegex(RuntimeError, "world agreement violated"):
                mixin.ModelRunnerKVCacheMixin._apply_token_constraints(stub, 100_000)


def make_mamba_config(n_layers, per_layer_bytes):
    layers = list(range(0, n_layers * 4, 4))  # global GDN layer ids, spaced
    return SimpleNamespace(
        mamba2_cache_params=SimpleNamespace(
            layers=layers,
            mamba_cache_per_req=per_layer_bytes * n_layers,
        )
    )


def make_mamba_stub(
    *,
    pp_size,
    config,
    start_layer=0,
    end_layer=10**9,
    mamba_full_memory_ratio=0.9,
    max_mamba_cache_size=None,
):
    overrides = {}

    sa = SimpleNamespace(
        max_mamba_cache_size=max_mamba_cache_size,
        mamba_full_memory_ratio=mamba_full_memory_ratio,
        disable_radix_cache=False,
        max_running_requests=None,
        max_speculative_num_draft_tokens=None,
        speculative_num_draft_tokens=None,
        enable_dp_attention=False,
        dp_size=1,
        gdn_resident_state_slots=None,
        uneven_memory_budgets_active=lambda: False,
    )

    def override(tag, **kw):
        overrides[tag] = kw
        for k, v in kw.items():
            setattr(sa, k, v)

    sa.override = override

    stub = SimpleNamespace(
        mambaish_config=config,
        server_args=sa,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        pp_size=pp_size,
        pp_rank=0,
        tp_rank=0,
        dp_size=1,
        start_layer=start_layer,
        end_layer=end_layer,
        overrides=overrides,
    )
    stub._stage_mamba_layer_counts = (
        lambda cfg: mixin.ModelRunnerKVCacheMixin._stage_mamba_layer_counts(stub, cfg)
    )
    stub._stage_local_mamba_cache_per_req = (
        lambda cfg: mixin.ModelRunnerKVCacheMixin._stage_local_mamba_cache_per_req(
            stub, cfg
        )
    )
    stub._calculate_mamba_ratio = lambda: 1
    stub._auto_mamba_demand_active = lambda: False
    stub._sync_uneven_mamba_cache_size = (
        lambda: mixin.ModelRunnerKVCacheMixin._sync_uneven_mamba_cache_size(stub)
    )
    return stub


class TestStageLocalMambaPerReq(CustomTestCase):
    def test_pp1_returns_global(self):
        config = make_mamba_config(48, 1 * MiB)
        stub = make_mamba_stub(pp_size=1, config=config)
        self.assertEqual(
            mixin.ModelRunnerKVCacheMixin._stage_local_mamba_cache_per_req(
                stub, config
            ),
            48 * MiB,
        )

    def test_stage_window_scales_exactly(self):
        config = make_mamba_config(48, 1 * MiB)
        # Window covers the first 12 GDN layers (ids 0,4,...,44).
        stub = make_mamba_stub(pp_size=2, config=config, start_layer=0, end_layer=48)
        self.assertEqual(
            mixin.ModelRunnerKVCacheMixin._stage_local_mamba_cache_per_req(
                stub, config
            ),
            12 * MiB,
        )

    def test_zero_gdn_window_is_zero(self):
        config = make_mamba_config(48, 1 * MiB)
        stub = make_mamba_stub(
            pp_size=2, config=config, start_layer=1, end_layer=4
        )  # ids are multiples of 4; window [1,4) holds none
        self.assertEqual(
            mixin.ModelRunnerKVCacheMixin._stage_local_mamba_cache_per_req(
                stub, config
            ),
            0,
        )


class TestHandleMaxMambaCacheStageLocal(CustomTestCase):
    TOTAL_REST_GB = 10.0

    def _run(self, stub):
        with patch.object(
            mixin,
            "get_world_group",
            return_value=SimpleNamespace(world_size=1, cpu_group=object()),
        ):
            return mixin.ModelRunnerKVCacheMixin.handle_max_mamba_cache(
                stub, self.TOTAL_REST_GB
            )

    def test_ratio_branch_charges_stage_local_bytes(self):
        # 48 GDN layers globally, 12 in this stage's window. The budget
        # formula must divide by the STAGE bytes (12 MiB/req), not the
        # global 48 MiB/req -- the falsifier asserts the old (global)
        # answer is NOT produced.
        config = make_mamba_config(48, 1 * MiB)
        stub = make_mamba_stub(pp_size=2, config=config, start_layer=0, end_layer=48)
        rest = self._run(stub)
        budget_gb = self.TOTAL_REST_GB * 0.9 / 1.9
        expected = int(budget_gb * GiB // (12 * MiB))
        wrong_old = int(budget_gb * GiB // (48 * MiB))
        size = stub.server_args.max_mamba_cache_size
        self.assertEqual(size, expected)
        self.assertNotEqual(size, wrong_old)
        # And the KV budget is charged the STAGE bytes only.
        self.assertAlmostEqual(
            rest, self.TOTAL_REST_GB - size * 12 * MiB / GiB, places=6
        )

    def test_pp1_byte_identical(self):
        config = make_mamba_config(48, 1 * MiB)
        stub = make_mamba_stub(pp_size=1, config=config)
        rest = self._run(stub)
        budget_gb = self.TOTAL_REST_GB * 0.9 / 1.9
        expected = int(budget_gb * GiB // (48 * MiB))
        self.assertEqual(stub.server_args.max_mamba_cache_size, expected)
        self.assertAlmostEqual(
            rest, self.TOTAL_REST_GB - expected * 48 * MiB / GiB, places=6
        )

    def test_zero_gdn_stage_contributes_sentinel_and_charges_nothing(self):
        config = make_mamba_config(48, 1 * MiB)
        stub = make_mamba_stub(pp_size=2, config=config, start_layer=1, end_layer=4)
        rest = self._run(stub)
        self.assertEqual(
            stub.server_args.max_mamba_cache_size,
            mixin.PP_STAGE_NO_MAMBA_STATE_SLOTS,
        )
        self.assertEqual(rest, self.TOTAL_REST_GB)

    def test_sentinel_never_binds_the_world_min(self):
        # World of two: the peer holds real GDN layers and reduces its real
        # count into the MIN; the sentinel stage must adopt it.
        config = make_mamba_config(48, 1 * MiB)
        stub = make_mamba_stub(pp_size=2, config=config, start_layer=1, end_layer=4)
        with FakeWorld(2, other_capacity=1234):
            rest = mixin.ModelRunnerKVCacheMixin.handle_max_mamba_cache(
                stub, self.TOTAL_REST_GB
            )
        self.assertEqual(stub.server_args.max_mamba_cache_size, 1234)
        self.assertEqual(rest, self.TOTAL_REST_GB)  # still no local bytes


class TestMambaSyncGate(CustomTestCase):
    def test_pp_syncs_even_with_uniform_budgets(self):
        config = make_mamba_config(48, 1 * MiB)
        stub = make_mamba_stub(pp_size=2, config=config)
        stub.server_args.max_mamba_cache_size = 500
        with FakeWorld(2, other_capacity=400) as world:
            mixin.ModelRunnerKVCacheMixin._sync_uneven_mamba_cache_size(stub)
        self.assertEqual(stub.server_args.max_mamba_cache_size, 400)
        self.assertIn("reduce", world.calls)

    def test_default_path_still_skips(self):
        config = make_mamba_config(48, 1 * MiB)
        stub = make_mamba_stub(pp_size=1, config=config)
        stub.server_args.max_mamba_cache_size = 500
        with FakeWorld(3, other_capacity=1) as world:
            mixin.ModelRunnerKVCacheMixin._sync_uneven_mamba_cache_size(stub)
        self.assertEqual(stub.server_args.max_mamba_cache_size, 500)
        self.assertEqual(world.calls, [])


class TestMeasuredBudgetStageKeying(CustomTestCase):
    def _stub(self, pp_size, pp_rank):
        sa = SimpleNamespace(
            _measured_kv_budget_registry_path="/tmp/kv_budget-abcdef012345.json"
        )
        return SimpleNamespace(server_args=sa, pp_size=pp_size, pp_rank=pp_rank)

    def test_pp_paths_are_stage_disjoint(self):
        p0 = mixin.ModelRunnerKVCacheMixin._measured_kv_budget_cache_path(
            self._stub(2, 0)
        )
        p1 = mixin.ModelRunnerKVCacheMixin._measured_kv_budget_cache_path(
            self._stub(2, 1)
        )
        self.assertNotEqual(p0, p1)
        self.assertIn("-stage0", p0)
        self.assertIn("-stage1", p1)

    def test_pp1_path_unchanged(self):
        p = mixin.ModelRunnerKVCacheMixin._measured_kv_budget_cache_path(
            self._stub(1, 0)
        )
        self.assertEqual(p, "/tmp/kv_budget-abcdef012345.json")

    def test_fingerprint_gains_pp_fields_only_under_pp(self):
        from sglang.srt.uneven_perf import measured_kv_budget_fingerprint_fields

        def args(pp_size, ratio=None):
            return SimpleNamespace(
                model_path="m",
                tp_size=1,
                mem_fraction_static=0.9,
                kv_cache_dtype="auto",
                context_length=None,
                page_size=1,
                quantization=None,
                max_running_requests=None,
                chunked_prefill_size=None,
                speculative_algorithm=None,
                speculative_draft_model_path=None,
                speculative_adaptive=False,
                speculative_adaptive_config=None,
                speculative_num_draft_tokens=None,
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(max_bs=None)
                ),
                pp_size=pp_size,
                pp_layer_ratio=ratio,
            )

        base = measured_kv_budget_fingerprint_fields(args(1))
        self.assertNotIn("pp_size", base)
        pp = measured_kv_budget_fingerprint_fields(args(2, [44, 20]))
        self.assertEqual(pp["pp_size"], 2)
        self.assertEqual(pp["pp_layer_ratio"], [44, 20])


if __name__ == "__main__":
    unittest.main()
