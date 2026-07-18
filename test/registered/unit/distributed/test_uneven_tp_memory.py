"""CPU unit tests for uneven-TP memory handling (phase 3).

Covers the --rank-gpu-memory-mib MiB -> mem_fraction_static conversion
(derived once per rank at resolution time, applied unmodified in the
scheduler child), the rank-local byte profiling + token-capacity
min-sync gating in the KV pipeline, the rank-aware Mamba2 (GDN) state
shapes, and the auto rank resolution of model_config.get_num_kv_heads.

No GPU, no distributed init: NVML is mocked like in the phase-1 args
tests, torch.distributed collectives are patched, and `sgl_kernel` is
stubbed before the sglang imports.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _install_sgl_kernel_stub():
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

import sglang.srt.server_args as server_args_module  # noqa: E402
from sglang.srt.configs.mamba_utils import Mamba2StateShape  # noqa: E402
from sglang.srt.distributed.utils import (  # noqa: E402
    set_tp_partition_ratios,
)
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# NVML totals/free (MiB): GPU 0 is a 32 GiB card, GPUs 1/2 are 20 GiB
# cards (mirrors the phase-1 args tests and the target hardware layout).
FAKE_GPU_MEMORY = {
    0: (32768, 30000),
    1: (20480, 19000),
    2: (20480, 19000),
}


def _fake_query(gpu_ids):
    result = {}
    for gpu_id in sorted(set(gpu_ids)):
        result[gpu_id] = FAKE_GPU_MEMORY[gpu_id]
    return result


def make_args(**kwargs):
    """model_path='dummy' short-circuits __post_init__ so the uneven-TP
    handler can be exercised in isolation."""
    return ServerArgs(model_path="dummy", **kwargs)


def run_handler(args):
    with patch.object(server_args_module, "_query_rank_gpu_memory_mib", _fake_query):
        args._handle_uneven_tp()
    return args


class UnevenTPTestCase(CustomTestCase):
    def setUp(self):
        set_tp_partition_ratios(None)

    def tearDown(self):
        set_tp_partition_ratios(None)


class TestRankMemFraction(UnevenTPTestCase):
    def test_scalar_budget_heterogeneous_fractions(self):
        # Same MiB budget on different cards -> DIFFERENT fractions.
        args = run_handler(
            make_args(tp_size=2, rank_gpu_id=[0, 1], rank_gpu_memory_mib=16400)
        )
        fractions = args._rank_mem_fraction_static
        self.assertAlmostEqual(fractions[0], 16400 / 32768)
        self.assertAlmostEqual(fractions[1], 16400 / 20480)

    def test_colocation_scalar_budget(self):
        # TP=4, two ranks on the 32 GiB card: each gets budget/total of
        # THAT card; no additional ceiling on top of the MiB value.
        args = run_handler(
            make_args(
                tp_size=4,
                rank_gpu_id=[0, 0, 1, 2],
                rank_gpu_memory_mib=15000,
            )
        )
        fractions = args._rank_mem_fraction_static
        self.assertAlmostEqual(fractions[0], 15000 / 32768)
        self.assertAlmostEqual(fractions[1], 15000 / 32768)
        self.assertAlmostEqual(fractions[2], 15000 / 20480)
        self.assertAlmostEqual(fractions[3], 15000 / 20480)

    def test_budget_list_with_ratio(self):
        args = run_handler(
            make_args(
                tp_size=3,
                rank_gpu_id=[0, 1, 2],
                rank_gpu_memory_mib=[26000, 13000, 13000],
                rank_tp_ratio=[2, 1, 1],
            )
        )
        fractions = args._rank_mem_fraction_static
        self.assertAlmostEqual(fractions[0], 26000 / 32768)
        self.assertAlmostEqual(fractions[1], 13000 / 20480)
        self.assertAlmostEqual(fractions[2], 13000 / 20480)

    def test_apply_rank_memory_budget_overrides_fraction(self):
        args = run_handler(
            make_args(tp_size=2, rank_gpu_id=[0, 1], rank_gpu_memory_mib=16400)
        )
        fraction = args.apply_rank_memory_budget(1)
        self.assertAlmostEqual(fraction, 16400 / 20480)
        self.assertAlmostEqual(args.mem_fraction_static, 16400 / 20480)

    def test_apply_is_noop_on_default_path(self):
        args = make_args(tp_size=2)
        self.assertIsNone(args.apply_rank_memory_budget(0))
        self.assertFalse(args.uneven_memory_budgets_active())

    def test_uneven_memory_budgets_active(self):
        args = run_handler(
            make_args(tp_size=2, rank_gpu_id=[0, 1], rank_gpu_memory_mib=16400)
        )
        self.assertTrue(args.uneven_memory_budgets_active())

    def test_fractions_survive_pickling(self):
        # The scheduler children get the ServerArgs via pickling (spawn).
        import pickle

        args = run_handler(
            make_args(tp_size=2, rank_gpu_id=[0, 1], rank_gpu_memory_mib=16400)
        )
        clone = pickle.loads(pickle.dumps(args))
        self.assertEqual(
            clone._rank_mem_fraction_static, args._rank_mem_fraction_static
        )
        self.assertAlmostEqual(
            clone.apply_rank_memory_budget(0), 16400 / 32768
        )


class TestTokenCapacitySync(UnevenTPTestCase):
    """_apply_token_constraints: min-sync gating for uneven budgets."""

    def _run(self, *, uneven, world_size, capacity, other_capacity, user_limit=None):
        import sglang.srt.model_executor.model_runner_kv_cache_mixin as mixin

        fake_self = SimpleNamespace(
            server_args=SimpleNamespace(
                max_total_tokens=user_limit,
                uneven_memory_budgets_active=lambda: uneven,
            ),
            pp_size=1,
            # Weighted-DCP C-min-reduce gate: no token vector installed in
            # these tests, so dcp_size=1 keeps uneven_dcp_active() False and
            # exercises the classic min-sync branch below it.
            dcp_size=1,
            # #79/#90 hybrid physical ceilings (mamba / SWA): inactive for
            # this plain-MHA stub. _apply_hybrid_kv_token_cap with cap=None
            # is the real no-op contract.
            _hybrid_kv_token_cap=lambda: None,
            _swa_hybrid_kv_token_cap=lambda: None,
            _apply_hybrid_kv_token_cap=lambda tc, cap, kind="mamba": tc,
        )
        calls = []

        def fake_all_reduce(tensor, op=None, group=None):
            calls.append(op)
            tensor.copy_(torch.minimum(tensor, torch.tensor(other_capacity)))

        fake_group = SimpleNamespace(world_size=world_size, cpu_group=object())
        with patch.object(mixin, "get_world_group", return_value=fake_group), patch(
            "torch.distributed.all_reduce", side_effect=fake_all_reduce
        ):
            result = mixin.ModelRunnerKVCacheMixin._apply_token_constraints(
                fake_self, capacity
            )
        return result, calls

    def test_default_path_no_sync(self):
        result, calls = self._run(
            uneven=False, world_size=3, capacity=1000, other_capacity=1
        )
        self.assertEqual(result, 1000)
        self.assertEqual(calls, [])

    def test_uneven_min_syncs_token_capacity(self):
        result, calls = self._run(
            uneven=True, world_size=3, capacity=1000, other_capacity=950
        )
        self.assertEqual(result, 950)
        self.assertEqual(len(calls), 1)

    def test_uneven_single_process_no_sync(self):
        result, calls = self._run(
            uneven=True, world_size=1, capacity=1000, other_capacity=1
        )
        self.assertEqual(result, 1000)
        self.assertEqual(calls, [])

    def test_user_cap_applies_before_sync(self):
        result, _ = self._run(
            uneven=True,
            world_size=2,
            capacity=1000,
            other_capacity=900,
            user_limit=800,
        )
        self.assertEqual(result, 800)


class TestMamba2StateShapeUneven(UnevenTPTestCase):
    # Qwen3-Next-like GDN geometry: 16 k heads, 32 v heads.
    KW = dict(
        intermediate_size=32 * 128,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )

    def test_default_path_unchanged(self):
        shape = Mamba2StateShape.create(tp_world_size=4, **self.KW)
        self.assertEqual(shape.num_k_heads_per_tp, 4)
        conv_dim = 32 * 128 + 2 * 16 * 128
        self.assertEqual(shape.conv, [(conv_dim // 4, 3)])
        self.assertEqual(shape.temporal, (8, 128, 128))
        # tp_rank is accepted but ignored without a plan.
        shape2 = Mamba2StateShape.create(tp_world_size=4, tp_rank=2, **self.KW)
        self.assertEqual(shape2.conv, shape.conv)
        self.assertEqual(shape2.temporal, shape.temporal)

    def test_uneven_per_rank_shapes(self):
        set_tp_partition_ratios([2, 1, 1])
        conv_dim = 32 * 128 + 2 * 16 * 128  # 8192, 512 per k-head unit
        expect = {0: (8, 4096, 16), 1: (4, 2048, 8), 2: (4, 2048, 8)}
        for rank, (k_heads, conv_rows, v_heads) in expect.items():
            shape = Mamba2StateShape.create(
                tp_world_size=3, tp_rank=rank, **self.KW
            )
            self.assertEqual(shape.num_k_heads_per_tp, k_heads)
            self.assertEqual(shape.conv, [(conv_rows, 3)])
            self.assertEqual(shape.temporal, (v_heads, 128, 128))
            self.assertEqual(shape.conv_dim, conv_dim)

    def test_uneven_shapes_partition_the_whole_state(self):
        set_tp_partition_ratios([3, 2, 2])
        conv_total = 0
        v_total = 0
        for rank in range(3):
            shape = Mamba2StateShape.create(
                tp_world_size=3, tp_rank=rank, **self.KW
            )
            conv_total += shape.conv[0][0]
            v_total += shape.temporal[0]
        self.assertEqual(conv_total, 32 * 128 + 2 * 16 * 128)
        self.assertEqual(v_total, 32)

    def test_uneven_requires_rank(self):
        set_tp_partition_ratios([2, 1, 1])
        with self.assertRaisesRegex(ValueError, "tp_rank"):
            Mamba2StateShape.create(tp_world_size=3, **self.KW)


class TestCustomAllReduceFallback(UnevenTPTestCase):
    """--rank-gpu-id disables custom all-reduce whenever its symmetric-
    peer assumptions cannot hold (heterogeneous cards, co-location,
    uneven shard plan); NCCL remains the collective backend."""

    def test_heterogeneous_cards_disable(self):
        args = run_handler(
            make_args(tp_size=2, rank_gpu_id=[0, 1], rank_gpu_memory_mib=16400)
        )
        self.assertTrue(args.disable_custom_all_reduce)

    def test_homogeneous_even_placement_keeps_it_enabled(self):
        # Two identical 20-GiB cards, one rank each, no ratio: the
        # symmetric assumptions hold — do not touch the default.
        args = run_handler(
            make_args(tp_size=2, rank_gpu_id=[1, 2], rank_gpu_memory_mib=15000)
        )
        self.assertFalse(args.disable_custom_all_reduce)

    def test_colocation_disables(self):
        args = run_handler(
            make_args(tp_size=2, rank_gpu_id=[0, 0], rank_gpu_memory_mib=15000)
        )
        self.assertTrue(args.disable_custom_all_reduce)

    def test_uneven_plan_disables_even_on_equal_cards(self):
        args = run_handler(
            make_args(
                tp_size=2,
                rank_gpu_id=[1, 2],
                rank_gpu_memory_mib=[13000, 6500],
                rank_tp_ratio=[2, 1],
            )
        )
        self.assertTrue(args.disable_custom_all_reduce)

    def test_explicit_disable_stays_disabled(self):
        args = run_handler(
            make_args(
                tp_size=2,
                rank_gpu_id=[0, 1],
                rank_gpu_memory_mib=16400,
                disable_custom_all_reduce=True,
            )
        )
        self.assertTrue(args.disable_custom_all_reduce)

    def test_default_path_untouched(self):
        args = make_args(tp_size=2)
        args._handle_uneven_tp()
        self.assertFalse(args.disable_custom_all_reduce)


class TestNumKvHeadsAutoRank(UnevenTPTestCase):
    def _mc(self, total_kv=8):
        from sglang.srt.configs.model_config import ModelConfig

        mc = ModelConfig.__new__(ModelConfig)
        mc.hf_config = SimpleNamespace(model_type="llama")
        mc.hf_text_config = SimpleNamespace(num_key_value_heads=total_kv)
        return mc

    def test_auto_resolves_rank_from_parallel_context(self):
        # Worker process: get_parallel().attn_tp_rank is available, so
        # pool/backends callers WITHOUT an explicit rank get THEIR share.
        import sglang.srt.runtime_context as rc

        set_tp_partition_ratios([2, 1, 1])
        mc = self._mc()
        for rank, expected in [(0, 4), (1, 2), (2, 2)]:
            fake_parallel = SimpleNamespace(attn_tp_rank=rank)
            with patch.object(rc, "get_parallel", return_value=fake_parallel):
                self.assertEqual(mc.get_num_kv_heads(3), expected)

    def test_falls_back_to_min_without_parallel_context(self):
        # Engine level: no initialized group -> conservative smallest share.
        set_tp_partition_ratios([2, 1, 1])
        mc = self._mc()
        self.assertEqual(mc.get_num_kv_heads(3), 2)

    def test_explicit_rank_still_wins(self):
        import sglang.srt.runtime_context as rc

        set_tp_partition_ratios([2, 1, 1])
        mc = self._mc()
        fake_parallel = SimpleNamespace(attn_tp_rank=1)
        with patch.object(rc, "get_parallel", return_value=fake_parallel):
            self.assertEqual(mc.get_num_kv_heads(3, rank=0), 4)

    def test_default_path_ignores_parallel_context(self):
        import sglang.srt.runtime_context as rc

        mc = self._mc()
        fake_parallel = SimpleNamespace(attn_tp_rank=0)
        with patch.object(rc, "get_parallel", return_value=fake_parallel):
            self.assertEqual(mc.get_num_kv_heads(4), 2)


class _FakeMlpLayer(torch.nn.Module):
    def __init__(self, rows, cols, tp_family="mlp", tp_units=None):
        super().__init__()
        self.tp_family = tp_family
        self.tp_units = tp_units
        self.weight = torch.nn.Parameter(
            torch.zeros(rows, cols), requires_grad=False
        )


class _FakeMoeLayer(torch.nn.Module):
    """FusedMoE-shaped module: family/units via moe_tp_* attributes."""

    def __init__(self, rows, cols, moe_tp_units):
        super().__init__()
        self.moe_tp_family = "moe"
        self.moe_tp_units = moe_tp_units
        self.w13_weight = torch.nn.Parameter(
            torch.zeros(rows, cols), requires_grad=False
        )


class TestMlpRebalanceHint(UnevenTPTestCase):
    """_maybe_suggest_mlp_rebalance: gating + the one-line restart hint."""

    PLAN = [5, 3, 3]
    UNITS_TOTAL = 28

    def _fake_model(self, with_mlp=True, with_moe=False):
        model = torch.nn.Module()
        if with_mlp:
            # partition_units(28, [5,3,3]) = [13, 8, 7]; rank 0 sees 13.
            model.mlp = _FakeMlpLayer(13, 8, tp_units=self.UNITS_TOTAL)
        if with_moe:
            model.moe = _FakeMoeLayer(64, 8, moe_tp_units=self.UNITS_TOTAL)
        model.attn = _FakeMlpLayer(14, 8, tp_family=None)
        return model

    def _run(
        self,
        *,
        uneven=True,
        world_size=3,
        plan=True,
        with_mlp=True,
        with_moe=False,
        gathered=None,
        user_limit=None,
        budget_bytes=10 * (1 << 30),
        local_tokens=500_000,
    ):
        import sglang.srt.model_executor.model_runner_kv_cache_mixin as mixin
        import sglang.srt.model_executor.pool_configurator as pool_configurator

        if plan:
            set_tp_partition_ratios(self.PLAN)
        fake_self = SimpleNamespace(
            server_args=SimpleNamespace(
                max_total_tokens=user_limit,
                uneven_memory_budgets_active=lambda: uneven,
            ),
            tp_size=3,
            tp_rank=0,
            page_size=1,
            model=self._fake_model(with_mlp=with_mlp, with_moe=with_moe),
        )
        # Bind the sibling mixin members onto the fake runner.
        fake_self._family_local_stats = (
            lambda family: mixin.ModelRunnerKVCacheMixin._family_local_stats(
                fake_self, family
            )
        )
        fake_self._CALIBRATION_FAMILY_ENV = (
            mixin.ModelRunnerKVCacheMixin._CALIBRATION_FAMILY_ENV
        )
        gather_calls = []

        def fake_all_gather_object(out, payload, group=None):
            gather_calls.append(payload)
            if gathered is not None:
                out[:] = list(gathered)
            else:
                out[:] = [payload] * len(out)

        fake_configurator = SimpleNamespace(
            calculate_pool_sizes=lambda budget, page: SimpleNamespace(
                max_total_num_tokens=local_tokens
            )
        )
        fake_group = SimpleNamespace(world_size=world_size, cpu_group=object())
        warnings_logged = []
        with patch.object(
            mixin, "get_world_group", return_value=fake_group
        ), patch.object(
            pool_configurator,
            "create_memory_pool_configurator",
            return_value=fake_configurator,
        ), patch(
            "torch.distributed.all_gather_object",
            side_effect=fake_all_gather_object,
        ), patch.object(
            mixin.logger,
            "warning",
            side_effect=lambda msg, *a: warnings_logged.append(msg % a),
        ):
            mixin.ModelRunnerKVCacheMixin._maybe_suggest_mlp_rebalance(
                fake_self, budget_bytes
            )
        return gather_calls, warnings_logged

    def test_default_path_is_silent_noop(self):
        gather_calls, warnings_logged = self._run(uneven=False)
        self.assertEqual(gather_calls, [])
        self.assertEqual(warnings_logged, [])

    def test_single_process_noop(self):
        gather_calls, warnings_logged = self._run(world_size=1)
        self.assertEqual(gather_calls, [])
        self.assertEqual(warnings_logged, [])

    def test_no_plan_noop(self):
        gather_calls, warnings_logged = self._run(plan=False)
        self.assertEqual(gather_calls, [])
        self.assertEqual(warnings_logged, [])

    def test_no_mlp_family_layers_noop(self):
        gather_calls, warnings_logged = self._run(with_mlp=False)
        self.assertEqual(gather_calls, [])
        self.assertEqual(warnings_logged, [])

    def test_balanced_ranks_stay_silent(self):
        # Equal capacities on every rank (gathered = own payload): no hint
        # — in particular an active vector that balanced the ranks.
        gather_calls, warnings_logged = self._run()
        self.assertEqual(len(gather_calls), 1)
        self.assertEqual(warnings_logged, [])

    def test_imbalanced_ranks_log_restart_hint(self):
        GB = 1 << 30
        bpu = 48 * 3 * 5120.0
        gathered = [
            (594_999 * 26_000.0 + 1.23 * GB, 642_303, {"mlp": (4608, 4608 * bpu)}),
            (594_999 * 13_000.0, 594_999, {"mlp": (2765, 2765 * bpu)}),
            (594_999 * 13_000.0 + 1.14 * GB, 689_157, {"mlp": (2765, 2765 * bpu)}),
        ]
        gather_calls, warnings_logged = self._run(gathered=gathered)
        self.assertEqual(len(gather_calls), 1)
        self.assertEqual(len(warnings_logged), 1)
        msg = warnings_logged[0]
        self.assertIn("SGLANG_UNEVEN_MLP_VECTOR=", msg)
        self.assertIn("from 594999 to ~", msg)
        # The suggested vector conserves the units and sheds from TP1.
        vector = [
            int(v)
            for v in msg.split("SGLANG_UNEVEN_MLP_VECTOR=")[1]
            .split(" ")[0]
            .split(",")
        ]
        self.assertEqual(sum(vector), 4608 + 2765 + 2765)
        self.assertLess(vector[1], 2765)

    def test_moe_family_supplies_the_shiftable_mass(self):
        # MoE model: the dense-MLP family is tiny (its shiftable bytes
        # cannot unpin TP1) while the expert family carries the weight
        # mass — the hint must rebalance via SGLANG_UNEVEN_MOE_VECTOR.
        GB = 1 << 30
        mlp_bpu = 3 * 512.0  # tiny shared expert
        moe_bpu = 48 * 3 * 5120.0 * 8  # expert weights dominate
        def fams(units_mlp, units_moe):
            return {
                "mlp": (units_mlp, units_mlp * mlp_bpu),
                "moe": (units_moe, units_moe * moe_bpu),
            }
        gathered = [
            (594_999 * 26_000.0 + 1.23 * GB, 642_303, fams(4608, 4608)),
            (594_999 * 13_000.0, 594_999, fams(2765, 2765)),
            (594_999 * 13_000.0 + 1.14 * GB, 689_157, fams(2765, 2765)),
        ]
        _, warnings_logged = self._run(gathered=gathered, with_moe=True)
        self.assertEqual(len(warnings_logged), 1)
        msg = warnings_logged[0]
        self.assertIn("SGLANG_UNEVEN_MOE_VECTOR=", msg)
        moe_vector = [
            int(v)
            for v in msg.split("SGLANG_UNEVEN_MOE_VECTOR=")[1]
            .split(" ")[0]
            .split(",")
        ]
        self.assertEqual(sum(moe_vector), 4608 + 2765 + 2765)
        self.assertLess(moe_vector[1], 2765)  # shed from the pinned rank

    def test_user_cap_below_capacity_stays_silent(self):
        GB = 1 << 30
        bpu = 48 * 3 * 5120.0
        gathered = [
            (594_999 * 26_000.0 + 1.23 * GB, 642_303, {"mlp": (4608, 4608 * bpu)}),
            (594_999 * 13_000.0, 594_999, {"mlp": (2765, 2765 * bpu)}),
            (594_999 * 13_000.0 + 1.14 * GB, 689_157, {"mlp": (2765, 2765 * bpu)}),
        ]
        _, warnings_logged = self._run(gathered=gathered, user_limit=500_000)
        self.assertEqual(warnings_logged, [])


if __name__ == "__main__":
    unittest.main()
