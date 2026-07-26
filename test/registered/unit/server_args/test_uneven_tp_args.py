"""Unit tests for the heterogeneous rank placement / uneven TP server args
(--rank-gpu-id, --rank-gpu-memory-mib, --rank-tp-ratio,
--rank-auto-reserve-mib) — CPU only, NVML mocked."""

import argparse
import os
import unittest

import pytest
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# NVML totals/free (MiB) used by all tests: GPU 0 is a 32 GiB card, GPUs 1/2
# are 20 GiB cards. Keyed by CUDA device index (the mapping to NVML physical
# indices is inside the mocked function).
FAKE_GPU_MEMORY = {
    0: (32768, 30000),
    1: (20480, 19000),
    2: (20480, 19000),
}


def _fake_query(gpu_ids):
    result = {}
    for gpu_id in sorted(set(gpu_ids)):
        if gpu_id not in FAKE_GPU_MEMORY:
            raise ValueError(
                f"--rank-gpu-id names GPU {gpu_id}, but NVML only reports "
                f"{len(FAKE_GPU_MEMORY)} device(s) "
                f"(indices 0-{len(FAKE_GPU_MEMORY) - 1})."
            )
        result[gpu_id] = FAKE_GPU_MEMORY[gpu_id]
    return result


def make_args(**kwargs):
    """ServerArgs with model_path='dummy' short-circuits __post_init__, so
    the uneven-TP handler can be exercised in isolation."""
    return ServerArgs(model_path="dummy", **kwargs)


def run_handler(args):
    with patch.object(
        server_args_module, "_query_rank_gpu_memory_mib", _fake_query
    ):
        args._handle_uneven_tp()
    return args


class TestCliParsing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def parse(self, *extra):
        return self.parser.parse_args(["--model-path", "m", *extra])

    def test_defaults(self):
        parsed = self.parse()
        self.assertIsNone(parsed.rank_gpu_id)
        self.assertIsNone(parsed.rank_gpu_memory_mib)
        self.assertIsNone(parsed.rank_tp_ratio)
        # #68: the default reserve is the demand-derived 'auto' sentinel,
        # not the former flat 2048 MiB.
        self.assertEqual(
            parsed.rank_auto_reserve_mib,
            ServerArgs.AUTO_RANK_MEMORY_RESERVE_MIB,
        )

    def test_rank_gpu_id_list(self):
        parsed = self.parse("--rank-gpu-id", "0,0,1,2")
        self.assertEqual(parsed.rank_gpu_id, [0, 0, 1, 2])

    def test_memory_mib_scalar_and_list(self):
        self.assertEqual(
            self.parse("--rank-gpu-memory-mib", "15000").rank_gpu_memory_mib,
            15000,
        )
        self.assertEqual(
            self.parse(
                "--rank-gpu-memory-mib", "26000,17000,17000"
            ).rank_gpu_memory_mib,
            [26000, 17000, 17000],
        )

    def test_ratio_list_and_auto(self):
        self.assertEqual(self.parse("--rank-tp-ratio", "2,1,1").rank_tp_ratio, [2, 1, 1])
        self.assertEqual(self.parse("--rank-tp-ratio", "auto").rank_tp_ratio, "auto")

    def test_reserve_stays_string(self):
        parsed = self.parse("--rank-auto-reserve-mib", "2048,2048,10240")
        self.assertEqual(parsed.rank_auto_reserve_mib, "2048,2048,10240")


class TestValidationDefaults(CustomTestCase):
    def test_no_flags_is_a_noop(self):
        args = make_args(tp_size=4, mem_fraction_static=0.8, base_gpu_id=1)
        run_handler(args)  # must not raise, must not mutate
        self.assertEqual(args.mem_fraction_static, 0.8)
        self.assertIsNone(args.rank_tp_ratio)

    def test_reserve_without_auto_rejected(self):
        args = make_args(rank_auto_reserve_mib="4096")
        with self.assertRaisesRegex(ValueError, "rank-auto-reserve-mib"):
            run_handler(args)


class TestMutualRequirements(CustomTestCase):
    def test_gpu_id_requires_memory(self):
        args = make_args(tp_size=2, rank_gpu_id=[0, 1])
        with self.assertRaisesRegex(ValueError, "rank-gpu-memory-mib"):
            run_handler(args)

    def test_memory_requires_gpu_id(self):
        args = make_args(tp_size=2, rank_gpu_memory_mib=15000)
        with self.assertRaisesRegex(ValueError, "requires --rank-gpu-id"):
            run_handler(args)

    def test_ratio_does_not_require_gpu_id(self):
        """--rank-tp-ratio is a pure PARTITION description and is independent
        of device placement, so it must be usable without --rank-gpu-id.

        This replaces the former test_ratio_requires_gpu_id, which asserted the
        opposite. The coupling was removed deliberately: it blocked the
        cross-vendor bring-up, where two launchers (--nnodes 2 --node-rank 0/1,
        one CUDA venv + one ROCm venv on ONE host) each place their own rank,
        and where --rank-gpu-id could not describe the AMD rank at all because
        it resolves devices through NVML.
        """
        args = make_args(tp_size=2, rank_tp_ratio=[2, 1])
        run_handler(args)
        self.assertEqual(args.rank_tp_ratio, [2, 1])
        self.assertIsNone(args.rank_gpu_id)

    def test_ratio_validated_without_gpu_id(self):
        """Decoupling must not lose validation: the ratio checks have to run on
        the placement-free path too, not just after --rank-gpu-id is seen."""
        with self.assertRaisesRegex(ValueError, "length"):
            run_handler(make_args(tp_size=2, rank_tp_ratio=[2, 1, 1]))
        with self.assertRaisesRegex(ValueError, "identical entries"):
            run_handler(make_args(tp_size=2, rank_tp_ratio=[1, 1]))
        with self.assertRaisesRegex(ValueError, "positive integers"):
            run_handler(make_args(tp_size=2, rank_tp_ratio=[2, 0]))


class TestLengthAndValueChecks(CustomTestCase):
    def test_gpu_id_length_mismatch(self):
        args = make_args(tp_size=4, rank_gpu_id=[0, 1], rank_gpu_memory_mib=15000)
        with self.assertRaisesRegex(ValueError, "length"):
            run_handler(args)

    def test_negative_gpu_id(self):
        args = make_args(tp_size=2, rank_gpu_id=[0, -1], rank_gpu_memory_mib=15000)
        with self.assertRaisesRegex(ValueError, ">= 0"):
            run_handler(args)

    def test_unknown_gpu_id(self):
        args = make_args(tp_size=2, rank_gpu_id=[0, 7], rank_gpu_memory_mib=15000)
        with self.assertRaisesRegex(ValueError, "GPU 7"):
            run_handler(args)

    def test_ratio_length_mismatch(self):
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=15000,
            rank_tp_ratio=[2, 1],
        )
        with self.assertRaisesRegex(ValueError, "rank-tp-ratio length"):
            run_handler(args)

    def test_ratio_nonpositive(self):
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=15000,
            rank_tp_ratio=[2, 0, 1],
        )
        with self.assertRaisesRegex(ValueError, "positive integers"):
            run_handler(args)

    def test_ratio_identical_entries_rejected(self):
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=15000,
            rank_tp_ratio=[2, 2, 2],
        )
        with self.assertRaisesRegex(ValueError, "even.*split"):
            run_handler(args)

    def test_invalid_ratio_string(self):
        args = make_args(
            tp_size=2,
            rank_gpu_id=[0, 1],
            rank_gpu_memory_mib=15000,
            rank_tp_ratio="fastest",
        )
        with self.assertRaisesRegex(ValueError, "auto"):
            run_handler(args)

    def test_memory_list_requires_ratio(self):
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 17000, 17000],
        )
        with self.assertRaisesRegex(ValueError, "requires\\s+--rank-tp-ratio"):
            run_handler(args)

    def test_memory_list_length_mismatch(self):
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 17000],
            rank_tp_ratio=[2, 1, 1],
        )
        with self.assertRaisesRegex(ValueError, "list length"):
            run_handler(args)

    def test_memory_nonpositive(self):
        args = make_args(tp_size=2, rank_gpu_id=[0, 1], rank_gpu_memory_mib=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            run_handler(args)


class TestPureTpScope(CustomTestCase):
    def _base(self, **kwargs):
        return make_args(
            tp_size=2, rank_gpu_id=[0, 1], rank_gpu_memory_mib=15000, **kwargs
        )

    def test_pp_rejected(self):
        with self.assertRaisesRegex(ValueError, "pp-size"):
            run_handler(self._base(pp_size=2))

    def test_dp_rejected(self):
        with self.assertRaisesRegex(ValueError, "dp-size"):
            run_handler(self._base(dp_size=2))

    def test_ep_rejected(self):
        with self.assertRaisesRegex(ValueError, "ep-size"):
            run_handler(self._base(ep_size=2))

    def test_multi_node_rejected(self):
        with self.assertRaisesRegex(ValueError, "single-node"):
            run_handler(self._base(nnodes=2))

    def test_mem_fraction_static_conflict(self):
        with self.assertRaisesRegex(ValueError, "mem-fraction-static"):
            run_handler(self._base(mem_fraction_static=0.8))

    def test_base_gpu_id_conflict(self):
        with self.assertRaisesRegex(ValueError, "base-gpu-id"):
            run_handler(self._base(base_gpu_id=1))

    def test_gpu_id_step_conflict(self):
        with self.assertRaisesRegex(ValueError, "gpu-id-step"):
            run_handler(self._base(gpu_id_step=2))


class TestPhysicalImpossibility(CustomTestCase):
    def test_scalar_budget_fits(self):
        # 2 x 15000 on the 32 GiB card, 1 x 15000 on each 20 GiB card.
        args = make_args(
            tp_size=4, rank_gpu_id=[0, 0, 1, 2], rank_gpu_memory_mib=15000
        )
        run_handler(args)  # must not raise
        self.assertIsNone(args.rank_tp_ratio)

    def test_scalar_budget_colocated_overflow(self):
        # 2 x 17000 = 34000 > 32768 on GPU 0.
        args = make_args(
            tp_size=4, rank_gpu_id=[0, 0, 1, 2], rank_gpu_memory_mib=17000
        )
        with self.assertRaisesRegex(
            ValueError, r"Physical impossibility.*GPU 0.*32768.*34000"
        ):
            run_handler(args)

    def test_list_budget_overflow(self):
        # 25000 > 20480 on GPU 1.
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 25000, 17000],
            rank_tp_ratio=[2, 1, 1],
        )
        with self.assertRaisesRegex(ValueError, "GPU 1"):
            run_handler(args)

    def test_list_budget_fits(self):
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 17000, 17000],
            rank_tp_ratio=[2, 1, 1],
        )
        run_handler(args)
        self.assertEqual(args.rank_tp_ratio, [2, 1, 1])


class TestAutoRatio(CustomTestCase):
    def test_auto_derives_budgets_and_weights(self):
        # An explicit flat reserve keeps the documented arithmetic exact
        # and device-independent (the 'auto' default derives the reserve
        # from the capacity tier; covered separately below).
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="2048",
        )
        run_handler(args)
        # budget = min(total - 2048, free - 1024) per GPU, one rank each:
        # GPU 0: min(30720, 28976) = 28976; GPU 1/2: min(18432, 17976) = 17976.
        self.assertEqual(args.rank_gpu_memory_mib, [28976, 17976, 17976])
        # gcd(28976, 17976, 17976) = 8.
        self.assertEqual(args.rank_tp_ratio, [3622, 2247, 2247])

    def test_auto_colocated_budget_split(self):
        args = make_args(
            tp_size=4,
            rank_gpu_id=[0, 0, 1, 2],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="2048",
        )
        run_handler(args)
        # GPU 0 budget 28976 shared by two ranks -> 14488 each.
        self.assertEqual(
            args.rank_gpu_memory_mib, [14488, 14488, 17976, 17976]
        )
        self.assertEqual(args.rank_tp_ratio, [1811, 1811, 2247, 2247])

    def test_auto_uniform_budgets_collapse_to_even_split(self):
        # Same card type for both ranks -> uniform budgets: the ratio must
        # collapse to None and the budget list to a single scalar
        # (vLLM bugfix 74cfec1a9).
        args = make_args(
            tp_size=2,
            rank_gpu_id=[1, 2],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="2048",
        )
        run_handler(args)
        self.assertIsNone(args.rank_tp_ratio)
        self.assertEqual(args.rank_gpu_memory_mib, 17976)

    def test_auto_reserve_auto_derives_from_demand(self):
        # #68 default path ('auto' reserve): the short-circuited ServerArgs
        # stub needs the pieces the derivation touches -- a real
        # CudaGraphConfig and a mocked device capacity (the reserve is a
        # function of the capacity tier, so the raw NVML totals must not
        # leak in). Wiring-level assertion: the budgets follow
        # min(total - derived_reserve, free - 1024) with the reserve from
        # derived_rank_auto_reserve_mib.
        from sglang.srt.model_executor.cuda_graph_config import (
            default_cuda_graph_config,
        )

        args = make_args(
            tp_size=2, rank_gpu_id=[1, 2], rank_tp_ratio="auto"
        )
        args.cuda_graph_config = default_cuda_graph_config()
        args.disable_cuda_graph = True  # capture term off: runtime reserve only
        gpu_mem = 20480.0
        with patch.object(
            server_args_module,
            "get_device_memory_capacity",
            return_value=gpu_mem,
            create=True,
        ):
            expected_reserve = args.derived_rank_auto_reserve_mib(gpu_mem, 1)
            run_handler(args)
        total, free = FAKE_GPU_MEMORY[1]
        expected_budget = min(total - expected_reserve, free - 1024)
        self.assertGreater(expected_reserve, 0)
        # Uniform cards -> collapse to the even split with a scalar budget.
        self.assertIsNone(args.rank_tp_ratio)
        self.assertEqual(args.rank_gpu_memory_mib, expected_budget)

    def test_auto_with_explicit_uniform_budget_list_collapses(self):
        args = make_args(
            tp_size=2,
            rank_gpu_id=[0, 1],
            rank_gpu_memory_mib=[15000, 15000],
            rank_tp_ratio="auto",
        )
        run_handler(args)
        self.assertIsNone(args.rank_tp_ratio)
        self.assertEqual(args.rank_gpu_memory_mib, 15000)

    def test_auto_with_explicit_budget_list_derives_weights(self):
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 17000, 17000],
            rank_tp_ratio="auto",
        )
        run_handler(args)
        self.assertEqual(args.rank_tp_ratio, [26, 17, 17])

    def test_auto_requires_gpu_id(self):
        args = make_args(tp_size=2, rank_tp_ratio="auto")
        with self.assertRaisesRegex(ValueError, "requires --rank-gpu-id"):
            run_handler(args)

    def test_auto_reserve_scalar(self):
        args = make_args(
            tp_size=2,
            rank_gpu_id=[1, 2],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="4096",
        )
        run_handler(args)
        # min(20480 - 4096, 19000 - 1024) = 16384 per rank.
        self.assertEqual(args.rank_gpu_memory_mib, 16384)

    def test_auto_reserve_per_rank_max_wins_per_gpu(self):
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 0, 1],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="2048,8192,2048",
        )
        run_handler(args)
        # GPU 0: reserve max(2048, 8192) = 8192 -> min(24576, 28976) // 2
        # = 12288 per rank; GPU 1: min(18432, 17976) = 17976.
        self.assertEqual(args.rank_gpu_memory_mib, [12288, 12288, 17976])

    def test_auto_reserve_list_length_mismatch(self):
        args = make_args(
            tp_size=2,
            rank_gpu_id=[0, 1],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="2048,2048,2048",
        )
        with self.assertRaisesRegex(ValueError, "entries"):
            run_handler(args)

    def test_auto_reserve_negative(self):
        args = make_args(
            tp_size=2,
            rank_gpu_id=[0, 1],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="-1",
        )
        with self.assertRaisesRegex(ValueError, ">= 0"):
            run_handler(args)

    def test_auto_reserve_not_an_int(self):
        args = make_args(
            tp_size=2,
            rank_gpu_id=[0, 1],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="lots",
        )
        with self.assertRaisesRegex(ValueError, "integer"):
            run_handler(args)

    def test_auto_reserve_leaves_no_budget(self):
        args = make_args(
            tp_size=2,
            rank_gpu_id=[1, 2],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="20480",
        )
        with self.assertRaisesRegex(ValueError, "leaves no budget"):
            run_handler(args)


class TestGpuIdLookup(CustomTestCase):
    def test_default_formula_unchanged(self):
        args = make_args(tp_size=4, base_gpu_id=2, gpu_id_step=2)
        for tp_rank in range(4):
            self.assertEqual(
                args.gpu_id_for_rank(0, tp_rank, 1, 4), 2 + tp_rank * 2
            )

    def test_default_formula_with_pp(self):
        args = make_args(tp_size=2, pp_size=2)
        self.assertEqual(args.gpu_id_for_rank(0, 0, 2, 2), 0)
        self.assertEqual(args.gpu_id_for_rank(0, 1, 2, 2), 1)
        self.assertEqual(args.gpu_id_for_rank(1, 0, 2, 2), 2)
        self.assertEqual(args.gpu_id_for_rank(1, 1, 2, 2), 3)

    def test_explicit_mapping_wins(self):
        args = make_args(tp_size=4, rank_gpu_id=[0, 0, 1, 2])
        self.assertEqual(
            [args.gpu_id_for_rank(0, r, 1, 4) for r in range(4)], [0, 0, 1, 2]
        )


class TestMlpRatio(CustomTestCase):
    """--rank-mlp-ratio / SGLANG_UNEVEN_MLP_VECTOR: the MLP-family weight
    vector of the uneven-TP self-calibration (env wins over CLI; only
    valid on top of an active base plan)."""

    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def _valid_base(self, **kwargs):
        return make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 17000, 17000],
            rank_tp_ratio=[2, 1, 1],
            **kwargs,
        )

    def test_cli_parsing(self):
        parsed = self.parser.parse_args(
            ["--model-path", "m", "--rank-mlp-ratio", "5,3,3"]
        )
        self.assertEqual(parsed.rank_mlp_ratio, [5, 3, 3])
        self.assertIsNone(
            self.parser.parse_args(["--model-path", "m"]).rank_mlp_ratio
        )

    def test_valid_with_base_plan(self):
        args = run_handler(self._valid_base(rank_mlp_ratio=[5, 3, 3]))
        self.assertEqual(args.rank_mlp_ratio, [5, 3, 3])

    def test_requires_base_plan(self):
        # With --rank-gpu-id but WITHOUT a ratio plan the vector is
        # meaningless (every family already splits evenly).
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=15000,
            rank_mlp_ratio=[5, 3, 3],
        )
        with self.assertRaisesRegex(ValueError, "base plan"):
            run_handler(args)

    def test_requires_base_plan_on_default_path(self):
        # Fully default path (no rank flags at all): a vector must fail
        # fast rather than being silently ignored.
        args = make_args(tp_size=3, rank_mlp_ratio=[5, 3, 3])
        with self.assertRaisesRegex(ValueError, "base plan"):
            run_handler(args)

    def test_length_must_match_tp_size(self):
        args = self._valid_base(rank_mlp_ratio=[5, 3])
        with self.assertRaisesRegex(ValueError, "length"):
            run_handler(args)

    def test_entries_must_be_positive(self):
        for bad in ([5, 0, 3], [5, -1, 3]):
            args = self._valid_base(rank_mlp_ratio=bad)
            with self.assertRaisesRegex(ValueError, "positive"):
                run_handler(args)

    def test_env_wins_over_cli(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_UNEVEN_MLP_VECTOR.override("7,4,4"):
            args = run_handler(self._valid_base(rank_mlp_ratio=[5, 3, 3]))
        self.assertEqual(args.rank_mlp_ratio, [7, 4, 4])

    def test_env_alone(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_UNEVEN_MLP_VECTOR.override("7,4,4"):
            args = run_handler(self._valid_base())
        self.assertEqual(args.rank_mlp_ratio, [7, 4, 4])

    def test_env_validated_like_cli(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_UNEVEN_MLP_VECTOR.override("7,4"):
            with self.assertRaisesRegex(ValueError, "length"):
                run_handler(self._valid_base())
        with envs.SGLANG_UNEVEN_MLP_VECTOR.override("banana"):
            with self.assertRaisesRegex(ValueError, "integer"):
                run_handler(self._valid_base())

    def test_env_without_base_plan_fails_fast(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_UNEVEN_MLP_VECTOR.override("7,4,4"):
            with self.assertRaisesRegex(ValueError, "base plan"):
                run_handler(make_args(tp_size=3))


class TestMoeRatio(CustomTestCase):
    """--rank-moe-ratio / SGLANG_UNEVEN_MOE_VECTOR: the expert-weight
    family vector (same rules as the mlp family: env wins, base plan
    required)."""

    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def _valid_base(self, **kwargs):
        return make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 17000, 17000],
            rank_tp_ratio=[2, 1, 1],
            **kwargs,
        )

    def test_cli_parsing(self):
        parsed = self.parser.parse_args(
            ["--model-path", "m", "--rank-moe-ratio", "5,3,3"]
        )
        self.assertEqual(parsed.rank_moe_ratio, [5, 3, 3])
        self.assertIsNone(
            self.parser.parse_args(["--model-path", "m"]).rank_moe_ratio
        )

    def test_valid_with_base_plan(self):
        args = run_handler(self._valid_base(rank_moe_ratio=[5, 3, 3]))
        self.assertEqual(args.rank_moe_ratio, [5, 3, 3])

    def test_requires_base_plan(self):
        args = make_args(tp_size=3, rank_moe_ratio=[5, 3, 3])
        with self.assertRaisesRegex(ValueError, "base plan"):
            run_handler(args)

    def test_length_and_positivity(self):
        with self.assertRaisesRegex(ValueError, "length"):
            run_handler(self._valid_base(rank_moe_ratio=[5, 3]))
        with self.assertRaisesRegex(ValueError, "positive"):
            run_handler(self._valid_base(rank_moe_ratio=[5, 0, 3]))

    def test_env_wins_over_cli(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_UNEVEN_MOE_VECTOR.override("9,5,5"):
            args = run_handler(self._valid_base(rank_moe_ratio=[5, 3, 3]))
        self.assertEqual(args.rank_moe_ratio, [9, 5, 5])

    def test_both_families_together(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_UNEVEN_MLP_VECTOR.override("7,4,4"):
            with envs.SGLANG_UNEVEN_MOE_VECTOR.override("9,5,5"):
                args = run_handler(self._valid_base())
        self.assertEqual(args.rank_mlp_ratio, [7, 4, 4])
        self.assertEqual(args.rank_moe_ratio, [9, 5, 5])


class TestTreeSpecDcpGuard(CustomTestCase):
    """#76 guard: tree/branching speculation (--speculative-eagle-topk > 1)
    must HARD-ERROR under uneven-weighted DCP, because the tree-masked
    draft->draft verify attention on that path produces tree-topology-dependent
    verify logits -> non-deterministic, non-greedy output (proven on GPU vs a
    topk=1 oracle at temp 0). topk == 1 (linear chain) stays allowed and is
    bitwise-deterministic. CPU only; is_cuda()/is_hip() and the SGLANG_UNEVEN_DCP
    env are mocked so the CUDA weighted-DCP branch is exercised off-GPU."""

    def _run_guard(self, eagle_topk):
        # Weighted uneven-DCP + spec: dcp_size==tp_size, non-uniform ratio, both
        # env flags set -> uneven_weighted_dcp is True inside the handler.
        args = make_args(
            tp_size=3,
            dcp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 17000, 17000],
            rank_tp_ratio=[2, 1, 1],
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=eagle_topk,
        )
        env = {
            **os.environ,
            "SGLANG_UNEVEN_DCP": "1",
            "SGLANG_UNEVEN_DCP_WEIGHTED": "1",
        }
        with patch.object(server_args_module, "is_hip", return_value=False), patch.object(
            server_args_module, "is_cuda", return_value=True
        ), patch.dict(os.environ, env, clear=True):
            args._handle_dcp_validation()
        return args

    def test_topk_gt1_rejected(self):
        # topk 2, 4, 8 must all fail fast at arg validation.
        for topk in (2, 4, 8):
            with self.subTest(topk=topk):
                with self.assertRaisesRegex(ValueError, "eagle-topk"):
                    self._run_guard(topk)

    def test_topk1_chain_allowed(self):
        # The correct, deterministic chain path must NOT be disturbed.
        self._run_guard(1)  # must not raise


class TestTreeSpecDcpGuardBroadened(CustomTestCase):
    """#139: the #76 topk>1 guard must cover the whole set of conditions that
    activate the flashinfer dcp_tree_mask (uneven_dcp = kv-replicated uneven
    DCP OR weightless_kv), not only the uneven-WEIGHTED subset. Otherwise a
    future loosening of the blanket DCP+spec gates would let topk>1 flow onto
    the weightless-KV lane (or even-modulo uneven DCP) unguarded."""

    def _dcp_validate(self, args, env_extra=None):
        env = {**os.environ, **(env_extra or {})}
        with patch.object(
            server_args_module, "is_hip", return_value=False
        ), patch.object(
            server_args_module, "is_cuda", return_value=True
        ), patch.dict(os.environ, env, clear=True):
            args._handle_dcp_validation()
        return args

    def test_weightless_lane_topk_gt1_rejected_in_dcp_validation(self):
        # Fastlane + dcp>1 + topk>1 must hard-error even WITHOUT a
        # --speculative-algorithm and without any --rank-tp-ratio (this is the
        # exact hole: the old guard keyed off uneven_weighted_dcp, which can
        # never be True on the lane).
        for topk in (2, 8):
            with self.subTest(topk=topk):
                args = make_args(
                    tp_size=2,
                    dcp_size=2,
                    weightless_kv_fastlane=True,
                    speculative_eagle_topk=topk,
                )
                with self.assertRaisesRegex(ValueError, "eagle-topk"):
                    self._dcp_validate(args)

    def test_even_modulo_uneven_dcp_topk_gt1_rejected(self):
        # A --rank-tp-ratio plan with dcp>1 but NO weighted env pair and NO
        # spec algorithm = even-modulo uneven_dcp (kv-replicated). The
        # dcp_tree_mask keys off exactly this superset -> must hard-error.
        args = make_args(
            tp_size=3,
            dcp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 17000, 17000],
            rank_tp_ratio=[2, 1, 1],
            speculative_eagle_topk=2,
        )
        with self.assertRaisesRegex(ValueError, "eagle-topk"):
            self._dcp_validate(args)

    def test_weightless_lane_topk1_not_rejected_by_dcp_validation(self):
        # topk=1 (chain) on the lane must NOT trip the tree-mask guard.
        args = make_args(
            tp_size=2,
            dcp_size=2,
            weightless_kv_fastlane=True,
            speculative_eagle_topk=1,
        )
        self._dcp_validate(args)  # must not raise

    def test_fastlane_handler_topk_gt1_rejected_independently(self):
        # The lane's own handler must reject topk>1 as a distinct guard --
        # even with NO --speculative-algorithm set -- so it survives a future
        # relaxation of the lane's blanket all-spec reject.
        args = make_args(
            tp_size=2,
            dcp_size=2,
            weightless_kv_fastlane=True,
            speculative_eagle_topk=2,
        )
        with self.assertRaisesRegex(ValueError, "eagle-topk"):
            args._handle_weightless_kv_fastlane()

    def test_fastlane_handler_topk1_passes(self):
        # topk=1 without a spec algorithm must pass the lane handler (the
        # blanket reject only fires on an actual --speculative-algorithm).
        args = make_args(
            tp_size=2,
            dcp_size=2,
            weightless_kv_fastlane=True,
            speculative_eagle_topk=1,
        )
        args._handle_weightless_kv_fastlane()  # must not raise


if __name__ == "__main__":
    unittest.main()



def test_uneven_tp_is_rejected_on_models_whose_attention_is_not_aware():
    """A non-uniform ratio must be REFUSED, not half-applied.

    qwen2 / qwen3 / qwen3_moe derive `num_heads = total // tp_size` (an even
    split) while their Linear layers follow the installed ratio plan. Measured
    on TP=2 with `--rank-tp-ratio 11,21`: both ranks kept 7 heads
    (448 = 7x64, clamped to whole KV units) while o_proj's input was cut by the
    RAW ratio into 308/588 -> "mat1 and mat2 shapes cannot be multiplied".

    Clamping o_proj would be the WRONG repair: the shapes would agree while
    self.num_heads still reports the even count, turning a loud failure into a
    silent one. So the ratio is rejected until an architecture opts in.
    """
    import sglang.srt.distributed.utils as du
    from sglang.srt.models.qwen3 import _reject_uneven_tp_unaware_attention

    saved = du.get_tp_partition_ratios
    saved_active = du.tp_plan_active

    try:
        # non-uniform plan -> reject, naming the numbers and a way out
        du.get_tp_partition_ratios = lambda family=None: [11, 21]
        du.tp_plan_active = lambda tp_size, family=None: True
        with pytest.raises(ValueError) as ei:
            _reject_uneven_tp_unaware_attention("Qwen3", 2)
        msg = str(ei.value)
        assert "11, 21" in msg or "[11, 21]" in msg, msg
        assert "not uneven-TP aware" in msg, msg

        # uniform vector IS the even split -> must NOT be rejected
        du.get_tp_partition_ratios = lambda family=None: [1, 1]
        _reject_uneven_tp_unaware_attention("Qwen3", 2)

        # no plan installed -> must NOT be rejected
        du.tp_plan_active = lambda tp_size, family=None: False
        du.get_tp_partition_ratios = lambda family=None: None
        _reject_uneven_tp_unaware_attention("Qwen3", 2)
    finally:
        # exact restore, NOT importlib.reload: reloading swaps the module
        # object underneath every other module that imported from it and
        # poisons later tests in the same process (observed as a
        # test_uneven_tp_memory failure that vanishes in isolation)
        du.get_tp_partition_ratios = saved
        du.tp_plan_active = saved_active


def test_kv_eq_tp_stays_in_normal_mode_by_measurement():
    """kv == tp is deliberately NOT replicated -- a `<` -> `<=` flip was tried
    and REVERTED on measurement. Pin both the behavior and the reason.

    At kv == tp the kv-boundary alignment has groups == ranks, so the only
    non-straddling q split is the even one; `_partition_units_kv_aligned`
    falls back to the raw split at `groups >= n`, and the #105 uniform-GQA
    ragged kernel rejects a straddling split at the FIRST FORWARD. Measured
    on Qwen3.5-2B (q=8/kv=2, TP=2, ratio 11,21): the flip produced a green
    boot, duplicated KV, q split [2, 6] -- and a ValueError on the first
    request. The `<` semantics on the same config ran coherent,
    token-identical to TP=1, with the plan applied to every other dimension.
    """
    import sglang.srt.distributed.utils as du

    saved_active = du.tp_plan_active
    try:
        du.tp_plan_active = lambda tp_size, family=None: True

        # kv == tp: NORMAL mode (not replicated), even attention by geometry
        assert not du.attn_kv_replicated(2, 2)
        # kv < tp: replicated, unchanged
        assert du.attn_kv_replicated(4, 2)
        # kv > tp: normal, unchanged
        assert not du.attn_kv_replicated(2, 8)

        # WHY the flip cannot work at kv == tp: groups == ranks makes the
        # aligned partitioner fall back to the raw (straddling) split --
        # the exact split the #105 kernel then rejects at runtime.
        assert du._partition_units_kv_aligned(4, [11, 21], 2) == [1, 3], (
            "at groups == ranks the aligned partitioner returns the raw "
            "split; if this changed, re-evaluate the kv == tp decision"
        )

        # ...while at ranks > groups the repair engages and the o_proj input
        # follows the head-true units (the second-host numbers, kv < tp):
        sizes = du.partition_sizes(896, [11, 21], units=7)
        assert sizes == [256, 640], sizes  # head-true, NOT the raw 308/588

        # default path: no plan -> never replicated
        du.tp_plan_active = lambda tp_size, family=None: False
        assert not du.attn_kv_replicated(2, 2)
        assert not du.attn_kv_replicated(4, 2)
    finally:
        du.tp_plan_active = saved_active


def test_token_vector_without_a_plan_is_rejected_not_ignored():
    """SGLANG_UNEVEN_TOKEN_VECTOR without a non-uniform --rank-tp-ratio used
    to be SILENTLY IGNORED: resolve_cp_token_ratios bails on `not weights`
    BEFORE reading the env, the server boots green, flashinfer's even-DCP
    no-op serves plain TP output -- and the requested token ownership never
    exists. Measured on Qwen3.5-2B TP=2/DCP=2 TOKVEC=2,1: coherent,
    token-identical to TP=1, zero uneven-machinery log lines.

    A configured-looking server that does nothing that was asked is worse
    than a startup error. Pin the rejection, and pin that the plainest
    default (no vector, no plan) still resolves to None silently.
    """
    import os
    import types

    import sglang.srt.distributed.utils as du

    sa = types.SimpleNamespace(rank_tp_ratio=None, dcp_size=2)
    saved = os.environ.get("SGLANG_UNEVEN_TOKEN_VECTOR")
    try:
        os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = "2,1"
        with pytest.raises(ValueError) as ei:
            du.resolve_cp_token_ratios(sa)
        assert "silently ignored" in str(ei.value)

        # dcp_size == 1: the vector is inert by definition, no rejection
        sa1 = types.SimpleNamespace(rank_tp_ratio=None, dcp_size=1)
        assert du.resolve_cp_token_ratios(sa1) is None

        # default path: no vector, no plan -> silent None, unchanged
        del os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"]
        assert du.resolve_cp_token_ratios(sa) is None
    finally:
        if saved is None:
            os.environ.pop("SGLANG_UNEVEN_TOKEN_VECTOR", None)
        else:
            os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = saved
        import importlib
        importlib.reload(du)


class TestTreeSpecDcpGuardHardenedForNewFlagPaths(CustomTestCase):
    """#98 sighting: the #76 guard must reject the NEW tree-verify door too.

    Upstream carries three unconsolidated PRs (#31069 / #29587 / #29907) adding
    `--speculative-dflash-tree-verify` -- a second flag that reaches the same
    tree-masked draft->draft verify this guard exists to keep off uneven-DCP.
    The flag does not exist in this tree yet, so the guard reads it via getattr
    and is ARMED FOR ITS ARRIVAL: a guard that names one flag is a guard a
    merge walks past.

    The evidence behind the refusal is not theoretical. Byte gate, window #2,
    RTX 5090, native sgl_kernel vs Triton spec kernels on identical inputs:
    topk=1 chains were bit-identical 16/16, while topk>1 trees DIFFERED
    (tree_mask 6-13 elements) and one case raised a CUDA error. Trees on the
    fallback path are measurably not the same computation.
    """

    def _run_guard(self, **overrides):
        kwargs = dict(
            tp_size=3,
            dcp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 17000, 17000],
            rank_tp_ratio=[2, 1, 1],
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
        )
        kwargs.update(overrides)
        dflash_tree = kwargs.pop("dflash_tree_verify", None)
        args = make_args(**kwargs)
        if dflash_tree is not None:
            # Simulate the upstream flag landing on ServerArgs.
            args.speculative_dflash_tree_verify = dflash_tree
        env = {
            **os.environ,
            "SGLANG_UNEVEN_DCP": "1",
            "SGLANG_UNEVEN_DCP_WEIGHTED": "1",
        }
        with patch.object(
            server_args_module, "is_hip", return_value=False
        ), patch.object(
            server_args_module, "is_cuda", return_value=True
        ), patch.dict(
            os.environ, env, clear=True
        ):
            args._handle_dcp_validation()
        return args

    def test_new_dflash_tree_verify_flag_rejected_under_uneven_dcp(self):
        """The point of the hardening: topk is 1, so the OLD guard would have
        let this through."""
        with self.assertRaisesRegex(ValueError, "dflash-tree-verify"):
            self._run_guard(dflash_tree_verify=True)

    def test_healthy_path_still_open(self):
        """Both doors shut: chain draft, no tree verify -> must NOT raise."""
        self._run_guard(dflash_tree_verify=False)
        self._run_guard()  # flag absent entirely, as in this tree today

    def test_old_topk_door_still_rejected(self):
        """Hardening must not have replaced the original trigger."""
        with self.assertRaisesRegex(ValueError, "eagle-topk"):
            self._run_guard(speculative_eagle_topk=4)

    def test_reason_is_named_in_the_message(self):
        """Same explanation quality for the new door as for the old one: the
        message must say WHICH flag tripped it, not just 'tree'."""
        with self.assertRaises(ValueError) as cm:
            self._run_guard(dflash_tree_verify=True)
        msg = str(cm.exception)
        self.assertIn("#31069", msg)
        self.assertIn("non-deterministic", msg)
        self.assertIn("#76", msg)

    def test_reason_helper_is_the_single_definition(self):
        """Both doors must go through one predicate, so a third door has one
        place to be added rather than two to be forgotten."""
        args = make_args(tp_size=3, speculative_eagle_topk=4)
        self.assertIn("eagle-topk", args.tree_verify_activation_reason())
        args2 = make_args(tp_size=3, speculative_eagle_topk=1)
        self.assertIsNone(args2.tree_verify_activation_reason())
        args2.speculative_dflash_tree_verify = True
        self.assertIn("dflash-tree-verify", args2.tree_verify_activation_reason())

    def test_guard_does_not_fire_without_a_dcp_variant(self):
        """A plain single-GPU tree run must stay allowed -- the guard is about
        uneven DCP, not about trees in general."""
        args = make_args(
            tp_size=1,
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=4,
        )
        args.speculative_dflash_tree_verify = True
        with patch.object(
            server_args_module, "is_hip", return_value=False
        ), patch.object(server_args_module, "is_cuda", return_value=True):
            args._handle_dcp_validation()  # must not raise
