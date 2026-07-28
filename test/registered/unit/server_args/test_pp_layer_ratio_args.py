"""Unit tests for the uneven pipeline layer split (--pp-layer-ratio) and the
pipeline admission matrix of the heterogeneous-placement flags (#201).

CPU only, NVML mocked. Two halves:

* LAYER ALLOCATION -- what get_pp_indices actually hands each stage: sums,
  boundaries, contiguity, and which hybrid layer TYPES land where. The last
  one matters because a GDN/full-attention model does not have uniform layers:
  a stage's KV cost follows its full-attention count, not its layer count.
* REJECT MATRIX -- every (pp_size, rank_tp_ratio, rank_gpu_id) combination
  that the pipeline admission rules must accept or refuse by name.
"""

import argparse
import os
import unittest
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

PARTITION_ENV = "SGLANG_PP_LAYER_PARTITION"

# Qwen3.6-27B geometry: 64 backbone layers, full_attention_interval 4, so
# every 4th layer (1-based) is full attention -> 16 full, 48 linear (GDN).
QWEN36_LAYERS = 64
QWEN36_FULL_ATTENTION_INTERVAL = 4

# NVML totals/free (MiB): GPU 0 is a 32 GiB card, GPUs 1/2 are 20 GiB cards.
FAKE_GPU_MEMORY = {
    0: (32768, 30000),
    1: (20480, 19000),
    2: (20480, 19000),
    3: (20480, 19000),
}


def _fake_query(gpu_ids):
    return {gpu_id: FAKE_GPU_MEMORY[gpu_id] for gpu_id in sorted(set(gpu_ids))}


def layer_types(num_layers=QWEN36_LAYERS, interval=QWEN36_FULL_ATTENTION_INTERVAL):
    """The layer-type list a Qwen3.5/3.6 config derives, mirroring
    configs/qwen3_next.py: layers_block_type."""
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(num_layers)
    ]


def make_args(**kwargs):
    """ServerArgs with model_path='dummy' short-circuits __post_init__, so a
    single handler can be exercised in isolation."""
    return ServerArgs(model_path="dummy", **kwargs)


def run_uneven_tp(args):
    with patch.object(server_args_module, "_query_rank_gpu_memory_mib", _fake_query):
        args._handle_uneven_tp()
    return args


class PartitionEnvTestCase(CustomTestCase):
    """Base class that keeps SGLANG_PP_LAYER_PARTITION out of the ambient
    environment -- both handlers and get_pp_indices read it globally."""

    def setUp(self):
        self._saved = os.environ.pop(PARTITION_ENV, None)

    def tearDown(self):
        os.environ.pop(PARTITION_ENV, None)
        if self._saved is not None:
            os.environ[PARTITION_ENV] = self._saved


class TestDefaultSplitUnchanged(PartitionEnvTestCase):
    """Without the flag nothing is touched: the stock even split runs."""

    def test_handler_does_not_touch_env_when_unset(self):
        args = make_args(pp_size=2)
        args._handle_pp_layer_ratio()
        self.assertNotIn(PARTITION_ENV, os.environ)

    def test_even_split_is_the_stock_partition(self):
        # 64 layers over 4 stages: exactly 16 each, contiguous, no env var.
        windows = [get_pp_indices(QWEN36_LAYERS, r, 4) for r in range(4)]
        self.assertEqual(windows, [(0, 16), (16, 32), (32, 48), (48, 64)])

    def test_remainder_goes_to_the_last_stages(self):
        # The documented upstream rule, restated as a test so a change to it
        # cannot pass silently: 64 layers over 5 stages -> 12,12,13,13,14?  No:
        # base 12, remainder 4, the LAST 4 stages get the extra one.
        windows = [get_pp_indices(QWEN36_LAYERS, r, 5) for r in range(5)]
        counts = [end - start for start, end in windows]
        self.assertEqual(counts, [12, 13, 13, 13, 13])
        self.assertEqual(sum(counts), QWEN36_LAYERS)

    def test_explicit_even_vector_reproduces_the_default(self):
        os.environ[PARTITION_ENV] = "16,16,16,16"
        windows = [get_pp_indices(QWEN36_LAYERS, r, 4) for r in range(4)]
        self.assertEqual(windows, [(0, 16), (16, 32), (32, 48), (48, 64)])


class TestLayerAllocation(PartitionEnvTestCase):
    """What each stage actually receives under an uneven split."""

    def windows(self, partition):
        os.environ[PARTITION_ENV] = ",".join(str(n) for n in partition)
        return [
            get_pp_indices(QWEN36_LAYERS, r, len(partition))
            for r in range(len(partition))
        ]

    def test_counts_match_the_requested_vector(self):
        partition = [52, 12]
        windows = self.windows(partition)
        self.assertEqual([end - start for start, end in windows], partition)

    def test_window_boundaries_are_contiguous_and_cover_every_layer(self):
        windows = self.windows([52, 12])
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], QWEN36_LAYERS)
        for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
            self.assertEqual(prev_end, next_start)
        covered = [layer for start, end in windows for layer in range(start, end)]
        self.assertEqual(covered, list(range(QWEN36_LAYERS)))

    def test_extreme_split_one_layer_on_the_last_stage(self):
        windows = self.windows([63, 1])
        self.assertEqual(windows, [(0, 63), (63, 64)])

    def test_three_stages_uneven(self):
        windows = self.windows([40, 16, 8])
        self.assertEqual(windows, [(0, 40), (40, 56), (56, 64)])

    def test_hybrid_layer_types_per_stage(self):
        """A stage's KV cost follows its FULL-ATTENTION count, and with
        interval 4 that is not proportional to its layer count at the
        boundaries. 52/12 puts 13 of the 16 full-attention layers on stage 0
        and only 3 on stage 1 -- so the small stage is even cheaper in KV
        terms than its layer share suggests."""
        types = layer_types()
        self.assertEqual(types.count("full_attention"), 16)
        self.assertEqual(types.count("linear_attention"), 48)

        windows = self.windows([52, 12])
        per_stage = [types[start:end] for start, end in windows]
        self.assertEqual(
            [stage.count("full_attention") for stage in per_stage], [13, 3]
        )
        self.assertEqual(
            [stage.count("linear_attention") for stage in per_stage], [39, 9]
        )

    def test_hybrid_layer_types_boundary_offset(self):
        """Shifting the cut by one layer moves a full-attention layer across
        the boundary: 51/13 gives 12/4, not 13/3."""
        types = layer_types()
        windows = self.windows([51, 13])
        per_stage = [types[start:end] for start, end in windows]
        self.assertEqual(
            [stage.count("full_attention") for stage in per_stage], [12, 4]
        )

    def test_length_mismatch_is_rejected_by_get_pp_indices(self):
        os.environ[PARTITION_ENV] = "32,32"
        with self.assertRaises(ValueError):
            get_pp_indices(QWEN36_LAYERS, 0, 3)

    def test_sum_mismatch_is_rejected_by_get_pp_indices(self):
        """#154: an MTP-carrying Qwen3.6-27B GGUF reports block_count 65, but
        the backbone is 64 (the draft block is not a layer). A partition that
        sums to the block count is refused -- the sum must be the backbone
        depth."""
        os.environ[PARTITION_ENV] = "52,13"  # 65 = block_count, not layers
        with self.assertRaises(ValueError):
            get_pp_indices(QWEN36_LAYERS, 0, 2)


class TestPpLayerRatioValidation(PartitionEnvTestCase):
    def handler(self, **kwargs):
        args = make_args(**kwargs)
        args._handle_pp_layer_ratio()
        return args

    def test_accepted_vector_is_exported(self):
        self.handler(pp_size=2, pp_layer_ratio=[52, 12])
        self.assertEqual(os.environ[PARTITION_ENV], "52,12")

    def test_requires_a_pipeline(self):
        with self.assertRaisesRegex(ValueError, "--pp-size is 1"):
            self.handler(pp_size=1, pp_layer_ratio=[52, 12])

    def test_length_must_equal_pp_size(self):
        with self.assertRaisesRegex(ValueError, "length"):
            self.handler(pp_size=3, pp_layer_ratio=[52, 12])

    def test_every_stage_needs_at_least_one_layer(self):
        with self.assertRaisesRegex(ValueError, "positive integers"):
            self.handler(pp_size=2, pp_layer_ratio=[64, 0])

    def test_negative_entry_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integers"):
            self.handler(pp_size=2, pp_layer_ratio=[70, -6])

    def test_conflicts_with_the_pd_prefill_layer_split(self):
        with self.assertRaisesRegex(ValueError, "SGLANG_PP_LAYER_PARTITION"):
            self.handler(
                pp_size=2,
                pp_layer_ratio=[52, 12],
                disaggregation_prefill_layer_split=[52, 12],
            )

    def test_conflicting_preset_env_is_rejected(self):
        os.environ[PARTITION_ENV] = "32,32"
        with self.assertRaisesRegex(ValueError, "already set in the environment"):
            self.handler(pp_size=2, pp_layer_ratio=[52, 12])

    def test_identical_preset_env_is_accepted(self):
        os.environ[PARTITION_ENV] = "52,12"
        self.handler(pp_size=2, pp_layer_ratio=[52, 12])
        self.assertEqual(os.environ[PARTITION_ENV], "52,12")

    def test_cli_parses_the_flag(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            ["--model-path", "dummy", "--pp-size", "2", "--pp-layer-ratio", "52,12"]
        )
        self.assertEqual(parsed.pp_layer_ratio, [52, 12])


class TestPipelineRejectMatrix(PartitionEnvTestCase):
    """The (pp_size, rank_tp_ratio, rank_gpu_id) admission matrix.

    Baseline (pp_size == 1) behavior must be unchanged; a pipeline is admitted
    only with an explicit world-length placement whose per-stage GPU groups are
    disjoint.
    """

    # --- baseline: no pipeline, unchanged --------------------------------
    def test_tp_only_ratio_without_placement_still_accepted(self):
        args = run_uneven_tp(make_args(tp_size=2, rank_tp_ratio=[3, 2]))
        self.assertEqual(args.rank_tp_ratio, [3, 2])

    def test_tp_only_placement_length_must_equal_tp_size(self):
        with self.assertRaisesRegex(ValueError, r"must equal --tp-size"):
            run_uneven_tp(
                make_args(tp_size=2, rank_gpu_id=[0, 1, 2], rank_gpu_memory_mib=15000)
            )

    # --- pipeline x uneven TP --------------------------------------------
    def test_ratio_under_pipeline_without_placement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"requires --rank-gpu-id"):
            run_uneven_tp(make_args(tp_size=2, pp_size=2, rank_tp_ratio=[3, 2]))

    def test_ratio_under_pipeline_with_disjoint_groups_is_admitted(self):
        args = run_uneven_tp(
            make_args(
                tp_size=2,
                pp_size=2,
                rank_tp_ratio=[3, 2],
                rank_gpu_id=[0, 1, 2, 3],
                rank_gpu_memory_mib=15000,
            )
        )
        self.assertEqual(args.rank_gpu_id, [0, 1, 2, 3])
        # One fraction per WORLD rank, each against its own card's NVML total.
        self.assertEqual(len(args._rank_mem_fraction_static), 4)
        self.assertAlmostEqual(args._rank_mem_fraction_static[0], 15000 / 32768)
        self.assertAlmostEqual(args._rank_mem_fraction_static[3], 15000 / 20480)

    def test_stages_sharing_a_card_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"stages 0 and 1 on the same"):
            run_uneven_tp(
                make_args(
                    tp_size=2,
                    pp_size=2,
                    rank_tp_ratio=[3, 2],
                    rank_gpu_id=[0, 1, 1, 2],
                    rank_gpu_memory_mib=8000,
                )
            )

    def test_placement_length_must_be_world_sized_under_a_pipeline(self):
        with self.assertRaisesRegex(ValueError, r"--pp-size x --tp-size"):
            run_uneven_tp(
                make_args(
                    tp_size=2,
                    pp_size=2,
                    rank_gpu_id=[0, 1],
                    rank_gpu_memory_mib=15000,
                )
            )

    def test_budget_list_length_must_be_world_sized_under_a_pipeline(self):
        with self.assertRaisesRegex(ValueError, r"number of placed ranks"):
            run_uneven_tp(
                make_args(
                    tp_size=2,
                    pp_size=2,
                    rank_tp_ratio=[3, 2],
                    rank_gpu_id=[0, 1, 2, 3],
                    rank_gpu_memory_mib=[15000, 15000],
                )
            )

    def test_auto_ratio_is_refused_under_a_pipeline(self):
        with self.assertRaisesRegex(ValueError, r"auto/auto-performance"):
            run_uneven_tp(
                make_args(
                    tp_size=2,
                    pp_size=2,
                    rank_tp_ratio="auto",
                    rank_gpu_id=[0, 1, 2, 3],
                )
            )

    def test_co_location_within_one_stage_is_still_allowed(self):
        """Two ranks of the SAME stage may share a card (the pre-existing
        multi-rank-per-GPU mode); only cross-stage sharing is refused."""
        args = run_uneven_tp(
            make_args(
                tp_size=2,
                pp_size=2,
                rank_tp_ratio=[3, 2],
                rank_gpu_id=[0, 0, 1, 2],
                rank_gpu_memory_mib=15000,
            )
        )
        self.assertEqual(args.rank_gpu_id, [0, 0, 1, 2])


class TestWorldRankMapping(PartitionEnvTestCase):
    def test_world_rank_formula(self):
        args = make_args(tp_size=3, pp_size=2)
        self.assertEqual(
            [args.world_rank(pp, tp) for pp in range(2) for tp in range(3)],
            [0, 1, 2, 3, 4, 5],
        )

    def test_gpu_id_for_rank_uses_the_world_index(self):
        args = make_args(tp_size=2, pp_size=2)
        args.rank_gpu_id = [0, 1, 2, 3]
        placed = [
            args.gpu_id_for_rank(pp, tp, pp_size_per_node=2, tp_size_per_node=2)
            for pp in range(2)
            for tp in range(2)
        ]
        self.assertEqual(placed, [0, 1, 2, 3])

    def test_gpu_id_for_rank_without_pipeline_is_unchanged(self):
        args = make_args(tp_size=3)
        args.rank_gpu_id = [2, 0, 1]
        placed = [
            args.gpu_id_for_rank(0, tp, pp_size_per_node=1, tp_size_per_node=3)
            for tp in range(3)
        ]
        self.assertEqual(placed, [2, 0, 1])


if __name__ == "__main__":
    unittest.main()
