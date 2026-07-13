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


if __name__ == "__main__":
    unittest.main()
