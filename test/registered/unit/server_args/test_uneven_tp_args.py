"""Unit tests for the heterogeneous rank placement / uneven TP server args
(--rank-gpu-id, --rank-gpu-memory-mib, --rank-tp-ratio,
--rank-auto-reserve-mib) — CPU only, NVML mocked."""

import argparse
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pytest

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
    the uneven-TP handler can be exercised in isolation.

    ``enable_vram_ledger`` defaults to False HERE, not because the product
    default is False -- it is now True, the ledger is the VRAM authority --
    but because this file tests the LEGACY --rank-auto-reserve-mib reserve
    path, and the two are mutually exclusive by design
    (``_check_vram_ledger_flags``). Saying so once here is better than every
    case in the file rediscovering the conflict; a case that wants the ledger
    passes ``enable_vram_ledger=True`` explicitly.
    """
    kwargs.setdefault("enable_vram_ledger", False)
    return ServerArgs(model_path="dummy", **kwargs)


def run_handler(args):
    with patch.object(server_args_module, "_query_rank_gpu_memory_mib", _fake_query):
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
        self.assertEqual(
            self.parse("--rank-tp-ratio", "2,1,1").rank_tp_ratio, [2, 1, 1]
        )
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

    def test_ratio_with_pipeline_parallel_rejected(self):
        """#202: the placement-free path had no parallelism reject at all.

        The --pp-size/--dp-size/--ep-size/--nnodes rejects all live after the
        `rank_gpu_id is None` early return, so --rank-tp-ratio alone reached a
        pipeline unchecked — and it is NOT inert there:
        configure_scheduler_process installs the plan in every scheduler
        process, so each PP stage would shard its TP dimension by a vector
        that no part of the uneven-TP machinery was ever validated with.
        """
        with self.assertRaisesRegex(ValueError, r"--rank-tp-ratio.*--pp-size"):
            run_handler(make_args(tp_size=2, pp_size=2, rank_tp_ratio=[2, 1]))

    def test_ratio_without_pipeline_parallel_unchanged(self):
        """The default pp_size=1 lane must be untouched by the reject."""
        args = make_args(tp_size=2, pp_size=1, rank_tp_ratio=[2, 1])
        run_handler(args)
        self.assertEqual(args.rank_tp_ratio, [2, 1])

    def test_even_split_with_pipeline_parallel_still_allowed(self):
        """No ratio, no reject: plain PP keeps working."""
        args = make_args(tp_size=2, pp_size=2)
        run_handler(args)
        self.assertIsNone(args.rank_tp_ratio)


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
        args = make_args(tp_size=4, rank_gpu_id=[0, 0, 1, 2], rank_gpu_memory_mib=15000)
        run_handler(args)  # must not raise
        self.assertIsNone(args.rank_tp_ratio)

    def test_scalar_budget_colocated_overflow(self):
        # 2 x 17000 = 34000 > 32768 on GPU 0.
        args = make_args(tp_size=4, rank_gpu_id=[0, 0, 1, 2], rank_gpu_memory_mib=17000)
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
        # budget = NVML TOTAL - reserve per GPU, one rank each (#260: the
        # live free reading is NOT part of this -- see
        # TestBudgetIsTotalMinusReserve below).
        self.assertEqual(args.rank_gpu_memory_mib, [30720, 18432, 18432])
        # gcd(30720, 18432, 18432) = 6144.
        self.assertEqual(args.rank_tp_ratio, [5, 3, 3])

    def test_auto_colocated_budget_split(self):
        args = make_args(
            tp_size=4,
            rank_gpu_id=[0, 0, 1, 2],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="2048",
        )
        run_handler(args)
        # GPU 0 budget 30720 shared by two ranks -> 15360 each.
        self.assertEqual(args.rank_gpu_memory_mib, [15360, 15360, 18432, 18432])
        self.assertEqual(args.rank_tp_ratio, [5, 5, 6, 6])

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
        self.assertEqual(args.rank_gpu_memory_mib, 18432)

    def test_auto_reserve_auto_derives_from_demand(self):
        # #68 default path ('auto' reserve): the short-circuited ServerArgs
        # stub needs the pieces the derivation touches -- a real
        # CudaGraphConfig and a mocked device capacity (the reserve is a
        # function of the capacity tier, so the raw NVML totals must not
        # leak in). Wiring-level assertion: the budgets follow
        # total - derived_reserve with the reserve from
        # derived_rank_auto_reserve_mib.
        from sglang.srt.model_executor.cuda_graph_config import (
            default_cuda_graph_config,
        )

        args = make_args(tp_size=2, rank_gpu_id=[1, 2], rank_tp_ratio="auto")
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
        total, _free = FAKE_GPU_MEMORY[1]
        expected_budget = total - expected_reserve
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
        # 20480 - 4096 = 16384 per rank.
        self.assertEqual(args.rank_gpu_memory_mib, 16384)

    def test_auto_reserve_per_rank_max_wins_per_gpu(self):
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 0, 1],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="2048,8192,2048",
        )
        run_handler(args)
        # GPU 0: reserve max(2048, 8192) = 8192 -> (32768 - 8192) // 2
        # = 12288 per rank; GPU 1: 20480 - 2048 = 18432.
        self.assertEqual(args.rank_gpu_memory_mib, [12288, 12288, 18432])

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
            self.assertEqual(args.gpu_id_for_rank(0, tp_rank, 1, 4), 2 + tp_rank * 2)

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
        self.assertIsNone(self.parser.parse_args(["--model-path", "m"]).rank_mlp_ratio)

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
        self.assertIsNone(self.parser.parse_args(["--model-path", "m"]).rank_moe_ratio)

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
        with patch.object(
            server_args_module, "is_hip", return_value=False
        ), patch.object(server_args_module, "is_cuda", return_value=True), patch.dict(
            os.environ, env, clear=True
        ):
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
        ), patch.object(server_args_module, "is_cuda", return_value=True), patch.dict(
            os.environ, env, clear=True
        ):
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

    def test_tree_guard_fires_on_hip_too(self):
        # #139 audit: the guard used to sit inside the `elif is_cuda()` branch,
        # i.e. behind `if is_hip(): return`. On ROCm an uneven-DCP tree config
        # therefore passed arg validation and only died later in the backend's
        # defensive check, after the full weight load. The condition mirrors a
        # backend flag, not a platform capability, so it must fire on HIP.
        args = make_args(
            tp_size=3,
            dcp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_gpu_memory_mib=[26000, 17000, 17000],
            rank_tp_ratio=[2, 1, 1],
            speculative_eagle_topk=2,
        )
        with patch.object(
            server_args_module, "is_hip", return_value=True
        ), patch.object(server_args_module, "is_cuda", return_value=False):
            with self.assertRaisesRegex(ValueError, "eagle-topk"):
                args._handle_dcp_validation()

    def test_stock_even_dcp_on_hip_still_allows_trees(self):
        # The hoist must not start rejecting stock (head-sharded, even) DCP on
        # HIP: without --rank-tp-ratio and without the weightless lane there is
        # no dcp_tree_mask to protect.
        args = make_args(tp_size=2, dcp_size=2, speculative_eagle_topk=4)
        with patch.object(
            server_args_module, "is_hip", return_value=True
        ), patch.object(server_args_module, "is_cuda", return_value=False):
            args._handle_dcp_validation()  # must not raise

    def test_tree_guard_absent_without_dcp(self):
        # #139 verdict: topk > 1 on a non-DCP path (TP=1, and plain TP without
        # --dcp-size) reaches the stock single-wrapper EAGLE verify -- full
        # (prefix + tree ancestors) custom mask, one attention call, no LSE
        # merge -- which is NOT the #76 terrain. It must stay allowed.
        for tp in (1, 2):
            with self.subTest(tp_size=tp):
                args = make_args(
                    tp_size=tp,
                    dcp_size=1,
                    speculative_algorithm="EAGLE",
                    speculative_eagle_topk=4,
                )
                self._dcp_validate(args)  # must not raise


class TestWeightlessChainSpecAdmission(CustomTestCase):
    """#143: the lane admits CHAIN speculation, in exactly one shape.

    The Stage-1 blanket "no speculative decoding" reject is replaced by a set of
    NAMED rejects, each protecting one asymmetric piece the weightless workers
    would otherwise have to mirror. These tests pin both halves: the admitted
    config boots, and every neighbouring config still refuses with its reason.

    Deliberately NOT relaxed here (covered by TestTreeSpecGuardBreadth above):
    topk > 1 and the tree-verify door.
    """

    def _chain_args(self, **overrides):
        # NOTE the algorithm is spelled EAGLE, not NEXTN: the admission check
        # runs AFTER handle_speculative_decoding resolves the alias, so it only
        # ever sees resolved names. Constructing raw args here mirrors that.
        kwargs = dict(
            tp_size=3,
            dcp_size=3,
            weightless_kv_fastlane=True,
            weightless_kv_head_rank=0,
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_draft_placement="solo",
        )
        kwargs.update(overrides)
        return make_args(**kwargs)

    def test_chain_spec_on_the_lane_is_admitted(self):
        args = self._chain_args()
        args._handle_weightless_kv_fastlane()  # must not raise
        args._reject_unsupported_weightless_spec()  # must not raise

    def test_dcp_validation_no_longer_shuts_the_blanket_door(self):
        # The lane forces dcp_size == tp_size and is never uneven_WEIGHTED, so
        # before #143 the blanket CUDA "DCP + spec" reject fired here and
        # relaxing the lane handler alone would have been useless.
        args = self._chain_args()
        with patch.object(
            server_args_module, "is_hip", return_value=False
        ), patch.object(server_args_module, "is_cuda", return_value=True), patch.dict(
            os.environ, dict(os.environ), clear=True
        ):
            args._handle_dcp_validation()  # must not raise

    def test_split_placement_rejected(self):
        # THE load-bearing condition: a weightless rank has no draft weights,
        # no draft KV pool and no draft backends.
        args = self._chain_args(speculative_draft_placement="split")
        with self.assertRaisesRegex(ValueError, "draft-placement solo"):
            args._reject_unsupported_weightless_spec()

    def test_solo_rank_must_equal_head_rank(self):
        args = self._chain_args(
            tp_size=3,
            weightless_kv_head_rank=1,
            speculative_draft_gpu=None,
        )
        # solo rank resolves to 0 (no --speculative-draft-gpu), head rank is 1.
        with self.assertRaisesRegex(ValueError, "weightless-kv-head-rank"):
            args._reject_unsupported_weightless_spec()

    def test_non_eagle_family_rejected(self):
        for algo in ("NGRAM", "STANDALONE"):
            with self.subTest(algo=algo):
                args = self._chain_args(speculative_algorithm=algo)
                with self.assertRaisesRegex(ValueError, "EAGLE-family"):
                    args._reject_unsupported_weightless_spec()

    def test_adaptive_k_rejected(self):
        # One symmetric capture shape per boot; a runtime-varying draft length
        # would need a bucketed head+worker ladder.
        args = self._chain_args(speculative_adaptive=True)
        with self.assertRaisesRegex(ValueError, "speculative-adaptive"):
            args._reject_unsupported_weightless_spec()

    def test_block_decode_lane_rejected_with_spec(self):
        # The streaming block-decode graph ladder is captured over block COUNT
        # for a decode-shaped step; a verify's bs*(k+1) rows are a second axis.
        args = self._chain_args(weightless_kv_chunked_block_size=2048)
        with self.assertRaisesRegex(ValueError, "chunked-block-size"):
            args._reject_unsupported_weightless_spec()

    def test_block_decode_lane_still_allowed_without_spec(self):
        args = self._chain_args(
            speculative_algorithm=None,
            speculative_draft_placement="split",
            weightless_kv_chunked_block_size=2048,
        )
        args._handle_weightless_kv_fastlane()  # must not raise

    def test_topk_gt_1_still_rejected_with_a_chain_capable_config(self):
        # The #139 guard must survive this relaxation: same admitted shape,
        # only topk flipped.
        args = self._chain_args(speculative_eagle_topk=2)
        with self.assertRaisesRegex(ValueError, "eagle-topk"):
            args._handle_weightless_kv_fastlane()


class TestPinnedReserveShortfall(CustomTestCase):
    """#250: a PINNED --rank-auto-reserve-mib replaces the derived demand
    model wholesale (runtime/activation reserve AND graph-capture term), and
    on the uneven-DCP path nothing else charges those -- so the shortfall
    surfaces as an OOM in the first real prefill, not at startup.

    Numbers are the reference rig's: 1x 32 GiB + 2x 20 GiB (FAKE_GPU_MEMORY),
    TP=3, MTP with 4 draft tokens, and the Qwen3.6-27B GDN geometry.
    """

    # linear_num_key_heads, linear_num_value_heads, linear_key_head_dim,
    # linear_value_head_dim, activation itemsize (bf16).
    QWEN36_27B_GDN = (16, 48, 128, 128, 2)
    # get_device_memory_capacity() reports the FIRST visible device: the
    # 32 GiB card -> chunked_prefill_size 2048, decode max_bs 24 (tp < 4).
    TIER_GPU_MEM = 32768.0

    def _args(self, reserve):
        from sglang.srt.model_executor.cuda_graph_config import (
            default_cuda_graph_config,
        )

        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib=reserve,
            speculative_algorithm="NEXTN",
            speculative_num_draft_tokens=4,
        )
        args.cuda_graph_config = default_cuda_graph_config()
        return args

    def test_derived_demand_for_the_reference_rig(self):
        args = self._args("auto")
        args._apply_gpu_mem_capacity_defaults(self.TIER_GPU_MEM)
        # 512 + 2048 * 1.5 + 3 * 1 / 8 * 1024 = 3968 MiB runtime reserve,
        # plus 24 * 4 captured tokens * 2 MiB = 192 MiB graph capture.
        self.assertEqual(args.chunked_prefill_size, 2048)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 24)
        self.assertEqual(args.derived_rank_auto_reserve_mib(self.TIER_GPU_MEM, 1), 4160)

    def test_gdn_prefill_scratch_formula(self):
        args = self._args("auto")
        args._apply_gpu_mem_capacity_defaults(self.TIER_GPU_MEM)
        with patch.object(
            ServerArgs,
            "_gdn_linear_attention_dims",
            return_value=self.QWEN36_27B_GDN,
        ):
            # A 20 GiB rank of this rig owns ~27.7% of the summed budgets ->
            # 4 of 16 key-head units -> Hk=4, Hv=12, q_dim=k_dim=512,
            # v_dim=1536; T=2048, NT=32, s=2.
            scratch = args.gdn_prefill_scratch_mib(17976 / 64928)
        expected = (
            # in_proj_qkvz's packed output (+ in_proj_ba), added 2026-08-21
            # after an OOM landed on exactly that allocation; see
            # gdn_prefill_scratch_mib's docstring.
            2 * 2048 * (2 * 512 + 2 * 1536 + 2 * 12)
            + 2 * 2048 * (3 * 512 + 3 * 512 + 5 * 1536 + 12 * 128 + 64 * 12)
            + 2 * 32 * 12 * 128 * 128
            + 12 * 2048 * 12
        ) / (1 << 20)
        self.assertAlmostEqual(scratch, expected, places=6)
        # The scratch is real but SMALL next to the reserve: it explains the
        # OOM site, not the 500 MiB gap between two pinned reserve values.
        self.assertLess(scratch, 100.0)

    def test_no_gdn_layers_means_no_scratch_item(self):
        args = self._args("auto")
        with patch.object(ServerArgs, "_gdn_linear_attention_dims", return_value=None):
            self.assertIsNone(args.gdn_prefill_scratch_mib(1 / 3))

    def test_note_fires_below_the_derived_value_and_names_the_items(self):
        args = self._args("3000,2200,2200")
        args._apply_gpu_mem_capacity_defaults(self.TIER_GPU_MEM)
        with patch.object(
            ServerArgs,
            "_gdn_linear_attention_dims",
            return_value=self.QWEN36_27B_GDN,
        ):
            note = args.pinned_reserve_shortfall_note(
                1, 2200, self.TIER_GPU_MEM, 1, 17976 / 64928
            )
        self.assertIsNotNone(note)
        self.assertIn("2200 MiB on GPU 1", note)
        self.assertIn("4160 MiB", note)
        self.assertIn("short by 1960 MiB", note)
        self.assertIn("3968 MiB", note)  # runtime/activation reserve
        self.assertIn("192 MiB", note)  # graph capture
        self.assertIn("GDN prefill scratch", note)
        # 63 -> 79 MiB per layer on 2026-08-21: the estimator now counts
        # in_proj_qkvz's packed output, the allocation an OOM landed on.
        self.assertIn("79 MiB per layer", note)
        self.assertIn("--rank-auto-reserve-mib auto", note)

    def test_note_fires_for_the_value_that_boots_today_too(self):
        """2700 MiB carries on the reference rig while 2200 tips over, but
        both are below the derived demand. The advisory says so and stays a
        warning -- the tipping point is the fragmentation-sensitive sum of
        several unbudgeted terms, not a single sizing item."""
        args = self._args("2700")
        args._apply_gpu_mem_capacity_defaults(self.TIER_GPU_MEM)
        for pinned in (2200, 2700, 3000):
            self.assertIsNotNone(
                args.pinned_reserve_shortfall_note(
                    1, pinned, self.TIER_GPU_MEM, 1, 17976 / 64928
                ),
                f"{pinned} MiB is below the derived demand and must be noted",
            )

    def test_note_silent_when_the_pin_covers_the_demand(self):
        args = self._args("4160")
        args._apply_gpu_mem_capacity_defaults(self.TIER_GPU_MEM)
        self.assertIsNone(
            args.pinned_reserve_shortfall_note(
                1, 4160, self.TIER_GPU_MEM, 1, 17976 / 64928
            )
        )

    def _run_and_capture_warnings(self, reserve):
        args = self._args(reserve)
        with patch.object(
            server_args_module,
            "get_device_memory_capacity",
            return_value=self.TIER_GPU_MEM,
            create=True,
        ), patch.object(
            ServerArgs,
            "_gdn_linear_attention_dims",
            return_value=self.QWEN36_27B_GDN,
        ), patch.object(
            server_args_module, "logger"
        ) as mock_logger:
            run_handler(args)
        return args, [
            call.args[1] if len(call.args) > 1 else call.args[0]
            for call in mock_logger.warning.call_args_list
        ]

    def test_pinned_reserve_warns_at_boot(self):
        args, warnings = self._run_and_capture_warnings("3000,2200,2200")
        shortfall = [w for w in warnings if "rank-auto-reserve-mib pins" in str(w)]
        # One note per physical GPU, all three below the derived 4160 MiB.
        self.assertEqual(len(shortfall), 3)
        self.assertTrue(any("3000 MiB on GPU 0" in str(w) for w in shortfall))
        self.assertTrue(any("2200 MiB on GPU 1" in str(w) for w in shortfall))
        self.assertTrue(any("2200 MiB on GPU 2" in str(w) for w in shortfall))
        # Advisory only: the budgets are exactly what they were before.
        self.assertEqual(args.rank_gpu_memory_mib, [29768, 18280, 18280])

    def test_auto_reserve_does_not_warn(self):
        args, warnings = self._run_and_capture_warnings("auto")
        self.assertEqual(
            [w for w in warnings if "rank-auto-reserve-mib pins" in str(w)], []
        )


class TestLadderReserveDerivation(CustomTestCase):
    """#313: the adaptive step ladder funds its own rungs.

    The constellation is the #707 window of 2026-07-30 on the reference rig
    (1x 32 GiB + 2x 20 GiB, TP=3, NEXTN, high-accept ladder [1..5],
    chunked_prefill_size 2048, solo draft on rank 0). At the standard
    bar1_hi reserve (4500,4200,4200) the boot check missed the largest
    state by 54 MiB: 1376 MiB free with every rung paused against 918 MiB
    (adaptive_state_k5) + 512 MiB serving margin. 5200 on rank 0 carried
    it. Before this fix the derived demand was 4160 MiB on every GPU --
    the ladder itself was charged nothing at all.
    """

    TIER_GPU_MEM = 32768.0
    # The 2026-07-30 measurements this class reasons against.
    MEASURED_K5_FOOTPRINT_MIB = 918
    MEASURED_MARGIN_MIB = 512
    OBSERVED_SHORTFALL_MIB = 54
    STANDARD_RANK0_RESERVE_MIB = 4500
    # Derived demand WITHOUT any ladder term (the pre-#313 number).
    BASE_DEMAND_MIB = 4160

    def _args(self, **overrides):
        from sglang.srt.model_executor.cuda_graph_config import (
            default_cuda_graph_config,
        )

        kwargs = dict(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="auto",
            speculative_algorithm="NEXTN",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_adaptive=True,
            speculative_adaptive_config="high-accept",
            speculative_draft_placement="solo",
        )
        kwargs.update(overrides)
        args = make_args(**kwargs)
        args.cuda_graph_config = default_cuda_graph_config()
        args._apply_gpu_mem_capacity_defaults(self.TIER_GPU_MEM)
        return args

    def _mocked_measured_posts(self, args):
        """Patch the ladder's own posts to the MEASURED k5 footprint, so the
        derivation is exercised against the numbers the rig produced rather
        than against its own estimate."""
        from sglang.srt.speculative.adaptive_graph_memory import (
            LadderReserveDemand,
            LadderRungPost,
        )

        demand = LadderReserveDemand(
            posts=(
                LadderRungPost(rung=4, workspace_mib=384, capture_mib=240),
                # 918 MiB = the measured adaptive_state_k5 footprint
                # (scratch 424 + int-ws 184 + capture pool 352, rounded as
                # the boot check reported it).
                LadderRungPost(rung=5, workspace_mib=566, capture_mib=352),
            ),
            boot_rung=3,
            margin_mib=self.MEASURED_MARGIN_MIB,
            resident=False,
        )
        return patch.object(type(args), "ladder_reserve_demand", return_value=demand)

    # -- the derivation --------------------------------------------------

    def test_ladder_is_charged_only_to_the_solo_draft_gpu(self):
        args = self._args()
        self.assertEqual(args.ladder_reserve_gpu_id(), 0)
        demand = args.reserve_demand_per_gpu(self.TIER_GPU_MEM, {0: 1, 1: 1, 2: 1})
        self.assertGreater(demand[0], self.BASE_DEMAND_MIB)
        self.assertEqual(demand[1], self.BASE_DEMAND_MIB)
        self.assertEqual(demand[2], self.BASE_DEMAND_MIB)

    def test_estimated_posts_fund_the_707_constellation(self):
        """The ladder's own estimate (no mock): peak rung k5 = flashinfer
        workspace 384 + graph capture 24 bs x 6 draft tokens x 2 MiB = 672,
        plus the 512 MiB serving margin -> 4160 + 1184 = 5344 MiB, which
        clears the 4554 MiB the rig actually wanted."""
        args = self._args()
        derived = args.derived_rank_auto_reserve_mib(
            self.TIER_GPU_MEM, 1, hosts_solo_draft=True
        )
        self.assertEqual(derived, 5344)
        self.assertGreaterEqual(
            derived,
            self.STANDARD_RANK0_RESERVE_MIB + self.OBSERVED_SHORTFALL_MIB,
        )

    def test_measured_posts_propose_at_least_972_mib_on_rank_0(self):
        """With the ladder's posts mocked to what the rig measured, the
        derivation adds the whole boot-check requirement (918 + 512) -- more
        than the 972 MiB that covers the observed 54 MiB miss with the
        headroom the working 5200 MiB reserve had."""
        args = self._args()
        with self._mocked_measured_posts(args):
            derived = args.derived_rank_auto_reserve_mib(
                self.TIER_GPU_MEM, 1, hosts_solo_draft=True
            )
        extra = derived - self.BASE_DEMAND_MIB
        self.assertEqual(
            extra, self.MEASURED_K5_FOOTPRINT_MIB + self.MEASURED_MARGIN_MIB
        )
        self.assertGreaterEqual(extra, 972)
        self.assertGreaterEqual(
            derived,
            self.STANDARD_RANK0_RESERVE_MIB + self.OBSERVED_SHORTFALL_MIB,
        )

    def test_colocated_ranks_scale_the_capture_posts(self):
        args = self._args()
        one = args.derived_rank_auto_reserve_mib(
            self.TIER_GPU_MEM, 1, hosts_solo_draft=True
        )
        two = args.derived_rank_auto_reserve_mib(
            self.TIER_GPU_MEM, 2, hosts_solo_draft=True
        )
        # Each co-located rank process captures its own graphs; the
        # workspace/margin posts are per rank as well, but only the capture
        # term is scaled here -- so the ladder part grows by exactly the
        # peak rung's capture post.
        self.assertEqual(two - one, (24 * 6 * 2) + (24 * 4 * 2))

    # -- regression protection -------------------------------------------

    def test_split_placement_keeps_the_pre_313_derivation(self):
        args = self._args(speculative_draft_placement="split")
        self.assertIsNone(args.ladder_reserve_gpu_id())
        self.assertEqual(
            args.reserve_demand_per_gpu(self.TIER_GPU_MEM, {0: 1, 1: 1, 2: 1}),
            {0: self.BASE_DEMAND_MIB, 1: self.BASE_DEMAND_MIB, 2: self.BASE_DEMAND_MIB},
        )

    def test_no_adaptive_ladder_keeps_the_pre_313_derivation(self):
        args = self._args(speculative_adaptive=False, speculative_adaptive_config=None)
        self.assertIsNone(args.ladder_reserve_gpu_id())
        self.assertEqual(
            args.reserve_demand_per_gpu(self.TIER_GPU_MEM, {0: 1, 1: 1, 2: 1}),
            {0: self.BASE_DEMAND_MIB, 1: self.BASE_DEMAND_MIB, 2: self.BASE_DEMAND_MIB},
        )

    def test_no_speculation_at_all_keeps_the_pre_313_derivation(self):
        args = self._args(
            speculative_algorithm=None,
            speculative_num_steps=None,
            speculative_num_draft_tokens=None,
            speculative_adaptive=False,
            speculative_adaptive_config=None,
            speculative_draft_placement=None,
        )
        self.assertIsNone(args.ladder_reserve_gpu_id())
        for reserve in args.reserve_demand_per_gpu(
            self.TIER_GPU_MEM, {0: 1, 1: 1, 2: 1}
        ).values():
            # 3968 runtime reserve + 24 captured tokens x 2 MiB.
            self.assertEqual(reserve, 4016)

    def test_budgets_follow_the_ladder_aware_reserve(self):
        args = self._args()
        with patch.object(
            server_args_module,
            "get_device_memory_capacity",
            return_value=self.TIER_GPU_MEM,
            create=True,
        ):
            run_handler(args)
        # GPU 0 (32 GiB) now holds back 5344 instead of 4160 MiB; the two
        # 20 GiB cards are untouched.
        self.assertEqual(
            args.rank_gpu_memory_mib, [32768 - 5344, 20480 - 4160, 20480 - 4160]
        )

    # -- an explicit reserve keeps standing, and is told what it misses ---

    def test_pinned_reserve_is_not_inflated_but_names_the_derived_demand(self):
        # 4000 on the two 20 GiB cards is below their (unchanged) 4160 MiB
        # demand, so their note fires too and can be compared against GPU 0's.
        args = self._args(rank_auto_reserve_mib="4500,4000,4000")
        with patch.object(
            server_args_module,
            "get_device_memory_capacity",
            return_value=self.TIER_GPU_MEM,
            create=True,
        ), patch.object(server_args_module, "logger") as mock_logger:
            run_handler(args)
        # The pinned vector stands, exactly as passed.
        self.assertEqual(
            args.rank_gpu_memory_mib, [32768 - 4500, 20480 - 4000, 20480 - 4000]
        )
        notes = [
            str(call.args[1] if len(call.args) > 1 else call.args[0])
            for call in mock_logger.warning.call_args_list
        ]
        gpu0 = [n for n in notes if "4500 MiB on GPU 0" in n]
        self.assertEqual(len(gpu0), 1)
        self.assertIn("5344 MiB", gpu0[0])  # the ladder-aware derived demand
        self.assertIn("short by 844 MiB", gpu0[0])
        self.assertIn("adaptive ladder: +1184 MiB", gpu0[0])
        self.assertIn("peak built rung k5", gpu0[0])
        self.assertIn("serving margin 512 MiB", gpu0[0])
        self.assertIn("--rank-auto-reserve-mib auto", gpu0[0])
        # The 20 GiB cards host no ladder: their note keeps the old items
        # and the old (4160 MiB) bar.
        gpu1 = [n for n in notes if "4000 MiB on GPU 1" in n]
        self.assertEqual(len(gpu1), 1)
        self.assertIn("4160 MiB", gpu1[0])
        self.assertIn("short by 160 MiB", gpu1[0])
        self.assertNotIn("adaptive ladder", gpu1[0])

    def test_boot_suggestion_for_a_pinned_reserve(self):
        args = self._args(rank_auto_reserve_mib="4500,4200,4200")
        with patch.object(
            server_args_module,
            "get_device_memory_capacity",
            return_value=self.TIER_GPU_MEM,
            create=True,
        ):
            note = args.ladder_reserve_boot_suggestion(
                self.OBSERVED_SHORTFALL_MIB, tp_rank=0
            )
        self.assertIn("PINNED", note)
        self.assertIn("4500 MiB", note)
        self.assertIn("that value stands as passed", note)
        self.assertIn("5344 MiB", note)
        self.assertIn("at least 4554 MiB", note)
        self.assertIn("peak built rung k5", note)

    def test_boot_suggestion_under_auto(self):
        args = self._args()
        with patch.object(
            server_args_module,
            "get_device_memory_capacity",
            return_value=self.TIER_GPU_MEM,
            create=True,
        ):
            note = args.ladder_reserve_boot_suggestion(
                self.OBSERVED_SHORTFALL_MIB, tp_rank=0
            )
        self.assertIn("DERIVED reserve of 5344 MiB", note)
        self.assertIn("at least 54 MiB", note)
        self.assertIn(">= 5398 MiB", note)

    def test_boot_suggestion_degrades_without_a_rank(self):
        args = self._args()
        self.assertIsNone(args.ladder_reserve_boot_suggestion(54, tp_rank=None))
        self.assertIsNone(args.ladder_reserve_boot_suggestion(54, tp_rank=9))


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
        ), patch.object(server_args_module, "is_cuda", return_value=True), patch.dict(
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


class TestBudgetIsTotalMinusReserve(CustomTestCase):
    """#260: a co-resident process must not shrink the derived budgets.

    The budget is documented as NVML TOTAL minus the reserve. The derivation
    used to cap it at ``free_mib - 1024``, which charges a neighbour twice --
    once because it really holds those bytes, and once through the reserve
    the operator sized to cover it -- and re-derives the SHARD RATIO from a
    transient occupancy reading on top.

    Numbers below are the measured 2026-07-27 case: a 5090 (NVML total 32607
    MiB) with a PD instance holding 22227 MiB, plus two idle 3080s.
    """

    GPU_MEMORY = {
        0: (32607, 32607 - 22227),  # 5090, co-resident PD instance
        1: (20480, 20100),  # 3080, idle
        2: (20480, 20100),  # 3080, idle
    }
    CLEAN_MEMORY = {
        0: (32607, 32300),  # same rig, nothing else on the cards
        1: (20480, 20100),
        2: (20480, 20100),
    }

    def _run(self, reserve, memory, tp_size=3, rank_gpu_id=(0, 1, 2)):
        args = make_args(
            tp_size=tp_size,
            rank_gpu_id=list(rank_gpu_id),
            rank_tp_ratio="auto",
            rank_auto_reserve_mib=reserve,
        )
        with patch.object(
            server_args_module,
            "_query_rank_gpu_memory_mib",
            lambda ids: {g: memory[g] for g in sorted(set(ids))},
        ), patch.object(server_args_module, "logger") as mock_logger:
            args._handle_uneven_tp()
        warnings = [
            call.args[1] if len(call.args) > 1 else call.args[0]
            for call in mock_logger.warning.call_args_list
        ]
        return args, warnings

    def test_measured_case_keeps_total_minus_reserve(self):
        args, _ = self._run("25800,7000,7000", self.GPU_MEMORY)
        # Exactly the budgets the failing boot logged.
        self.assertEqual(args.rank_gpu_memory_mib, [6807, 13480, 13480])

    def test_neighbour_larger_than_the_reserve_does_not_rewrite_the_plan(self):
        """The case where the old cap silently took over.

        Reserve 7000 on a card whose neighbour holds 22227 MiB: the old
        formula returned min(25607, 10380 - 1024) = 9356 -- a budget, and
        therefore a shard ratio, derived from who else was on the card.
        """
        args, warnings = self._run("7000", self.GPU_MEMORY)
        self.assertEqual(args.rank_gpu_memory_mib, [25607, 13480, 13480])
        self.assertNotIn(9356, args.rank_gpu_memory_mib)
        # ... and the impossible part is SAID, not silently absorbed.
        shortfall = [w for w in warnings if "GPU 0" in str(w)]
        self.assertEqual(len(shortfall), 1)
        self.assertIn("22227 MiB already held by other processes", str(shortfall[0]))
        self.assertIn("10380 MiB free right now", str(shortfall[0]))
        self.assertIn("ABSOLUTE allowance", str(shortfall[0]))

    def test_clean_cards_are_unchanged_by_the_fix(self):
        """No co-residence -> the old cap never bound, and neither does the
        new arithmetic: byte-for-byte the same budgets and ratio."""
        args, warnings = self._run("3000,2700,2700", self.CLEAN_MEMORY)
        old_formula = [
            min(total - reserve, free - 1024)
            for (total, free), reserve in zip(
                [self.CLEAN_MEMORY[g] for g in (0, 1, 2)], [3000, 2700, 2700]
            )
        ]
        self.assertEqual(args.rank_gpu_memory_mib, old_formula)
        self.assertEqual(args.rank_gpu_memory_mib, [29607, 17780, 17780])
        self.assertEqual([w for w in warnings if "free right now" in str(w)], [])

    def test_colocated_ranks_split_the_absolute_budget(self):
        args, warnings = self._run(
            "7000", self.CLEAN_MEMORY, tp_size=4, rank_gpu_id=(0, 0, 1, 2)
        )
        # (32607 - 7000) // 2 = 12803 per co-located rank.
        self.assertEqual(args.rank_gpu_memory_mib, [12803, 12803, 13480, 13480])
        # 2 x 12803 = 25606 <= 32300 free -> nothing to warn about.
        self.assertEqual([w for w in warnings if "free right now" in str(w)], [])

    def test_note_is_silent_when_the_plan_fits(self):
        self.assertIsNone(
            ServerArgs.budget_free_shortfall_note(1, [0], 13480, 20480, 20100, 1024)
        )

    def test_note_flags_a_tight_context_margin(self):
        note = ServerArgs.budget_free_shortfall_note(1, [0], 19500, 20480, 20100, 1024)
        self.assertIn("less than 1024 MiB per rank", note)


class TestDerivedPlanVectorLog(CustomTestCase):
    """#391 harness item C: plain ``--rank-tp-ratio auto`` must name the
    per-rank vectors its weights partition into.

    Boots 8 and 9 resolved ``[28639,16512,16512]`` and the runbook claims
    that lands on ``[32,16,16]`` of V4's 64 heads, but nothing in the log
    said so -- and both boots died mid-load, before any rank could be asked
    for its own tensor shapes. The weight vector is a ratio; the plan is
    what the ratio partitions into, and only the second one is checkable.
    """

    # DeepSeek-V4-Flash geometry, trimmed to the keys the unit grid reads.
    CONFIG = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "head_dim": 512,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "num_hidden_layers": 43,
        "o_groups": 8,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048,
        "vocab_size": 129280,
    }

    def _model_dir(self, **overrides):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        config = dict(self.CONFIG)
        config.update(overrides)
        with open(os.path.join(directory, "config.json"), "w") as handle:
            json.dump(config, handle)
        return directory

    def _resolve(self, model_path):
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="2048",
        )
        args.model_path = model_path
        with self.assertLogs(server_args_module.logger, level="INFO") as captured:
            run_handler(args)
        return args, captured.output

    def _plan_lines(self, output):
        return [line for line in output if "partition into per-rank" in line]

    def test_auto_names_the_head_and_expert_vectors(self):
        args, output = self._resolve(self._model_dir())
        # (32768/20480/20480) - 2048 reserve, gcd-reduced.
        self.assertEqual(args.rank_tp_ratio, [5, 3, 3])
        lines = self._plan_lines(output)
        self.assertEqual(len(lines), 1, output)
        # 8 o_groups, largest-remainder: [4,2,2] groups x 8 heads each.
        self.assertIn("[32, 16, 16] of 64 q-heads", lines[0])
        self.assertIn("[4, 2, 2] of 8", lines[0])
        # 256 routed experts under the same weights.
        self.assertIn("[116, 70, 70] of 256 routed experts", lines[0])

    def test_the_head_vector_follows_the_unit_grid_and_can_disagree(self):
        """Falsifier: drop ``o_groups`` and the attention unit becomes the
        kv-head count, which for V4 is 1 -- unsplittable across 3 ranks, so
        the attention term must DISAPPEAR rather than report a made-up
        vector. A test that cannot distinguish the two says nothing."""
        _args, output = self._resolve(self._model_dir(o_groups=None))
        lines = self._plan_lines(output)
        self.assertEqual(len(lines), 1, output)
        self.assertNotIn("q-heads", lines[0])
        self.assertIn("[116, 70, 70] of 256 routed experts", lines[0])

    def test_an_unreadable_config_costs_one_debug_line_and_no_boot(self):
        """The line is diagnostic: a checkpoint the cost model cannot read
        must not turn a working boot into a failing one."""
        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib="2048",
        )
        args.model_path = os.path.join(tempfile.mkdtemp(), "absent")
        with self.assertLogs(server_args_module.logger, level="INFO") as captured:
            run_handler(args)
        self.assertEqual(self._plan_lines(captured.output), [])
        self.assertEqual(args.rank_tp_ratio, [5, 3, 3])


class TestDerivedReserveInfeasibility(CustomTestCase):
    """#458: what ``--rank-auto-reserve-mib auto`` cannot see, said out loud.

    The 2026-08-03 #439 confirmation window spent a boot on this. Its spec
    mandated ``RESERVE_MIB=auto``; ``auto`` derived 3968 MiB uniformly per GPU
    and the boot died after loading the weights with

        The per-rank budget leaves no GPU memory for the KV cache under
        --rank-gpu-memory-mib on rank 1: the 16512 MiB (16.12 GiB) budget is
        spent on weights + runtime state 17.59 GiB -- 17.59 GiB together,
        1498 MiB more than the budget, before a single KV token.

    The derivation is not wrong, it is BLIND: ``auto`` sizes the reserve from
    the stock activation heuristic (512 + max(chunked_prefill, 2048) * 1.5 +
    tp * pp / 8 * 1024, graph term 0 under --disable-cuda-graph) and nothing in
    it looks at the checkpoint. Making it look is not available at that point
    either -- the weight bytes depend on the shard ratio, and the shard ratio is
    derived FROM these budgets.

    So the fix is the message, not the model: the remedy the error used to give
    ("lower --rank-auto-reserve-mib for this GPU by the same amount") is not
    followable under ``auto``, where there is no value to lower and the next
    boot derives the identical number. These tests reproduce the infeasibility
    arithmetically and pin the note that now names it.
    """

    # The reference rig, NVML totals as the window read them.
    RIG_MEMORY = {
        0: (32607, 32300),  # RTX 5090
        1: (20480, 20100),  # RTX 3080
        2: (20480, 20100),  # RTX 3080
    }

    def _window_args(self, reserve):
        from sglang.srt.model_executor.cuda_graph_config import (
            default_cuda_graph_config,
        )

        args = make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_tp_ratio="auto",
            rank_auto_reserve_mib=reserve,
        )
        args.cuda_graph_config = default_cuda_graph_config()
        # The recipe: --disable-cuda-graph and --chunked-prefill-size 512, so
        # the capture term is 0 and the 2048-token floor sets the activation
        # part.
        args.disable_cuda_graph = True
        args.chunked_prefill_size = 512
        # A local path that does not exist: the pinned-reserve advisory reads
        # the HF config for its GDN scratch line, and 'dummy' sends it to the
        # hub with a retry loop.
        args.model_path = os.path.join(tempfile.mkdtemp(), "absent")
        return args

    def _resolve(self, args):
        with patch.object(
            server_args_module,
            "_query_rank_gpu_memory_mib",
            lambda ids: {g: self.RIG_MEMORY[g] for g in sorted(set(ids))},
        ), patch.object(
            server_args_module,
            "get_device_memory_capacity",
            return_value=20480.0,
            create=True,
        ):
            args._handle_uneven_tp()
        return args

    def test_auto_derives_the_3968_mib_the_window_measured(self):
        args = self._window_args("auto")
        with patch.object(
            server_args_module,
            "get_device_memory_capacity",
            return_value=20480.0,
            create=True,
        ):
            derived = args.derived_rank_auto_reserve_mib(20480.0, 1)
        # 512 + max(512, 2048) * 1.5 + 3 * 1 / 8 * 1024
        self.assertEqual(derived, 512 + 3072 + 384)
        self.assertEqual(derived, 3968)

    def test_the_derived_reserve_produces_the_infeasible_budget(self):
        """The falsifier for the SPEC inconsistency, arithmetic only.

        Gate 4 of ARM3_COMPUTE.md demands the base plan 30407,19080,19080, and
        that plan is 32607-2200 / 20480-1400 / 20480-1400 -- it is produced by
        the pinned reserve 2200,1400,1400 and by nothing else. ``auto`` cannot
        satisfy it even if the boot survived, and the boot does not: 16512 MiB
        is below the 17.59 GiB of weights + runtime state the window measured.
        """
        auto = self._resolve(self._window_args("auto"))
        self.assertEqual(auto.rank_gpu_memory_mib, [28639, 16512, 16512])
        # The window's error reported the shortfall as 1498 MiB and the demand
        # as 17.59 GiB; 16512 + 1498 = 18010 MiB is that 17.59 GiB, so the
        # budget derived here is the one that boot was handed.
        self.assertLess(auto.rank_gpu_memory_mib[1], 17.59 * 1024)
        self.assertEqual(round((auto.rank_gpu_memory_mib[1] + 1498) / 1024, 2), 17.59)
        # And the plan Gate 4 asks for is not the one 'auto' resolves.
        self.assertNotEqual(auto.rank_tp_ratio, [30407, 19080, 19080])

        pinned = self._resolve(self._window_args("2200,1400,1400"))
        self.assertEqual(pinned.rank_gpu_memory_mib, [30407, 19080, 19080])
        self.assertEqual(pinned.rank_tp_ratio, [30407, 19080, 19080])

    def test_the_corridor_repaired_reserve_resolves_its_own_base_plan(self):
        """The repaired recipe (#458): +400 MiB on each 3080.

        Measured serving minima were 211-251 MiB free on the 3080s against the
        400 MiB corridor floor; +400 MiB of reserve returns the same 400 MiB to
        the card, so the minimum lands at ~611-651 MiB. It moves the budgets,
        hence the base plan, which is why it is a new window rather than a
        tweak -- pinned here so the window's Gate 4 has a number to check.
        """
        repaired = self._resolve(self._window_args("2200,1800,1800"))
        self.assertEqual(repaired.rank_gpu_memory_mib, [30407, 18680, 18680])
        self.assertEqual(repaired.rank_tp_ratio, [30407, 18680, 18680])

    def test_the_note_names_the_derivation_and_a_reserve_that_fits(self):
        args = self._resolve(self._window_args("auto"))
        with patch.object(
            server_args_module,
            "get_device_memory_capacity",
            return_value=20480.0,
            create=True,
        ):
            note = args.derived_reserve_infeasible_note(1, 1498)
        self.assertIsNotNone(note)
        self.assertIn("3968 MiB of headroom for GPU 1", note)
        self.assertIn("never sees the checkpoint", note)
        # One rank on this GPU, so every MiB off the reserve is a MiB of
        # budget: 3968 - 1498 = 2470.
        self.assertIn("<= 2470 MiB", note)
        self.assertIn("--rank-auto-reserve-mib", note)

    def test_a_pinned_reserve_gets_no_note(self):
        """The standing advice is followable there, and repeating it would
        make the message longer without making it more useful."""
        args = self._resolve(self._window_args("2200,1400,1400"))
        self.assertIsNone(args.derived_reserve_infeasible_note(1, 1498))

    def test_the_note_halves_the_credit_when_two_ranks_share_the_card(self):
        """budget = (total - reserve) // colocated, so a MiB taken off the
        reserve is worth 1/colocated MiB to THIS rank. Getting that backwards
        would name a reserve that fails again for the same reason."""
        args = self._window_args("auto")
        args.rank_gpu_id = [0, 0, 1]
        with patch.object(
            server_args_module,
            "get_device_memory_capacity",
            return_value=20480.0,
            create=True,
        ):
            solo = args.derived_reserve_infeasible_note(2, 100)
            shared = args.derived_reserve_infeasible_note(0, 100)
        self.assertIn("<= 3868 MiB", solo)  # 3968 - 100 * 1
        self.assertIn("<= 3768 MiB", shared)  # 3968 - 100 * 2
        self.assertIn("2 co-located rank(s)", shared)

    def test_the_exhausted_budget_message_carries_the_note(self):
        """The two halves meet: the ValueError the window saw now ends with
        the reason its own remedy line could not be followed."""
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        args = self._resolve(self._window_args("auto"))
        with patch.object(
            server_args_module,
            "get_device_memory_capacity",
            return_value=20480.0,
            create=True,
        ):
            note = args.derived_reserve_infeasible_note(1, 1498)
        message = ModelRunnerKVCacheMixin.budget_exhausted_message(
            tp_rank=1,
            budget_mib=16512,
            budget_gb=16.12,
            posts=[("weights + runtime state", 17.59)],
            rest_memory_gb=-1498 / 1024,
            device_free_gb=2.0,
            occupancy=(20.0, 1.0),
            reserve_note=note,
        )
        self.assertIn("before a single KV token", message)
        self.assertIn("That budget was DERIVED, not chosen", message)
        # Without the note the message ends on advice that cannot be taken.
        bare = ModelRunnerKVCacheMixin.budget_exhausted_message(
            tp_rank=1,
            budget_mib=16512,
            budget_gb=16.12,
            posts=[("weights + runtime state", 17.59)],
            rest_memory_gb=-1498 / 1024,
            device_free_gb=2.0,
            occupancy=(20.0, 1.0),
        )
        self.assertNotIn("DERIVED", bare)
