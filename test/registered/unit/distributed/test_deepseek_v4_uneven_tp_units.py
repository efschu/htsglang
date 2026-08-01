# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 attention declares its uneven-TP unit count (#402, wall 3).

THE DEFECT, as it stood
``MqaAttentionBase`` built ``wq_b`` as a plain ColumnParallelLinear over
``n_heads * head_dim`` (64 x 512 = 32768 on V4-Flash) with no unit count, and
derived its per-rank geometry with ``//``::

    self.n_local_heads  = self.n_heads  // self.attn_tp_size
    self.n_local_groups = self.n_groups // self.attn_tp_size
    self.wq_b = ColumnParallelLinear(q_lora_rank, n_heads * head_dim, ...)

``--rank-tp-ratio auto`` derives its weights from the NVML budgets, so they
are BYTE counts -- on this rig ``[29607, 17780, 17780]``, summing to 65167.
Without a unit count ``partition_sizes`` requires the dimension to be
divisible by that sum and refuses (boot attempt 3, 2026-08-01)::

    ValueError: Cannot partition dimension of size 32768 with weight vector
    [29607, 17780, 17780]: 32768 is not divisible by sum(weights)=65167.
    Choose weights whose sum divides every sharded dimension, or pass the
    dimension's unit count.

THE UNIT IS THE o_group, NOT THE HEAD
Heads and o_groups are coupled. After attention the output is reshaped to
``(tokens, n_local_groups, -1)`` and multiplied by ``wo_a``, whose per-group
input width is the GLOBAL ``n_heads * head_dim // n_groups`` and is NOT
sharded. A rank therefore has to hold whole groups' worth of heads --
``n_heads // n_groups`` (8 on V4-Flash) at a time. Declaring the head count
would have satisfied the partitioner and then produced a wo_a einsum against
the wrong per-group width; declaring ``o_groups`` keeps every dimension in
step. That also fixes an explicit ratio: ``--rank-tp-ratio 30,17,17`` used to
give heads [30,17,17] against groups [2,2,2].

CPU only, real modules on the meta device: no GPU, no distributed group.
"""

import os

# Captured at import time by the DSV4 attention module; pinned so the test is
# deterministic wherever it runs (the fp8 wo_a path is a separate vehicle).
os.environ.setdefault("SGLANG_OPT_FP8_WO_A_GEMM", "0")

import unittest

import torch

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    partition_sizes,
    set_tp_partition_ratios,
)
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.models.deepseek_v4 import MqaAttentionBase
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# DeepSeek-V4-Flash-0731, verbatim from its config.json.
N_HEADS = 64
O_GROUPS = 8
HEAD_DIM = 512
Q_LORA_RANK = 1024
O_LORA_RANK = 1024
HIDDEN = 4096
WQ_B_OUT = N_HEADS * HEAD_DIM  # 32768
WO_OUT = O_GROUPS * O_LORA_RANK  # 8192

#: The vector `--rank-tp-ratio auto` emitted on this rig (NVML totals minus
#: the per-card reserve, gcd-reduced to 1). Sums to 65167; divides nothing.
AUTO_WEIGHTS = [29607, 17780, 17780]
#: The hand-picked vector the same rig is normally booted with.
EXPLICIT_WEIGHTS = [30, 17, 17]


def _config(**overrides):
    cfg = dict(
        quantization_config=None,
        num_hidden_layers=1,
        num_attention_heads=N_HEADS,
        o_groups=O_GROUPS,
        head_dim=HEAD_DIM,
        qk_rope_head_dim=64,
        q_lora_rank=Q_LORA_RANK,
        o_lora_rank=O_LORA_RANK,
        hidden_size=HIDDEN,
        num_key_value_heads=1,
        vocab_size=512,
        max_position_embeddings=1024,
        compress_ratios=[0],
        index_n_heads=64,
        index_head_dim=128,
        index_topk=32,
        num_hash_layers=1,
        hc_mult=2,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 2,
            "original_max_position_embeddings": 512,
            "type": "yarn",
        },
    )
    cfg.update(overrides)
    return DeepSeekV4Config(**cfg)


class _V4AttentionCase(CustomTestCase):
    """Builds the real MqaAttentionBase for one rank under an injected plan."""

    def setUp(self):
        self._saved = get_tp_partition_ratios()
        set_tp_partition_ratios(None)
        self._server_args = get_context().override_server_args()
        self._server_args.install()

    def tearDown(self):
        self._server_args.restore()
        set_tp_partition_ratios(self._saved)

    def _attn(self, tp_size, tp_rank, config=None):
        config = config if config is not None else _config()
        with torch.device("meta"), get_parallel().override(
            tp_size=tp_size,
            tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_tp_rank=tp_rank,
        ):
            return MqaAttentionBase(config, 0, None, "")

    def _ranks(self, tp_size, config=None):
        return [self._attn(tp_size, r, config) for r in range(tp_size)]

    def assert_block_is_coherent(self, mods):
        """Every relation the forward pass depends on, on every rank."""
        heads_per_group = N_HEADS // O_GROUPS
        for rank, m in enumerate(mods):
            with self.subTest(rank=rank):
                # wq_b emits exactly this rank's heads.
                self.assertEqual(
                    m.wq_b.output_size_per_partition, m.n_local_heads * m.head_dim
                )
                # wo_a emits this rank's groups; wo_b consumes them back.
                self.assertEqual(
                    m.wo_a.output_size_per_partition, m.n_local_groups * m.o_lora_rank
                )
                self.assertEqual(
                    m.wo_b.input_size_per_partition, m.n_local_groups * m.o_lora_rank
                )
                # The coupling: `o.view(T, n_local_groups, -1)` must present
                # wo_a with the GLOBAL per-group width, which is unsharded.
                self.assertEqual(
                    m.n_local_heads * m.head_dim // m.n_local_groups,
                    m.wo_a.input_size,
                )
                self.assertEqual(m.n_local_heads // m.n_local_groups, heads_per_group)


class TestUnevenAutoVector(_V4AttentionCase):
    """The byte-valued vector that produced the reported abort."""

    def test_auto_vector_snaps_to_whole_groups(self):
        set_tp_partition_ratios(AUTO_WEIGHTS)
        mods = self._ranks(3)

        # Exhaustive and disjoint: the shards reconstruct the whole model.
        self.assertEqual([m.n_local_heads for m in mods], [32, 16, 16])
        self.assertEqual(sum(m.n_local_heads for m in mods), N_HEADS)
        self.assertEqual([m.n_local_groups for m in mods], [4, 2, 2])
        self.assertEqual(sum(m.n_local_groups for m in mods), O_GROUPS)
        self.assertEqual(sum(m.wq_b.output_size_per_partition for m in mods), WQ_B_OUT)
        self.assertEqual(sum(m.wo_a.output_size_per_partition for m in mods), WO_OUT)
        self.assert_block_is_coherent(mods)

    def test_declaration_removed_reproduces_the_boot_error(self):
        """Can-fail. The same layer without the unit count is the pre-fix
        code, and it raises the boot's exact message."""
        set_tp_partition_ratios(AUTO_WEIGHTS)
        with self.assertRaises(ValueError) as cm:
            with torch.device("meta"), get_parallel().override(
                tp_size=3, tp_rank=0, attn_tp_size=3, attn_tp_rank=0
            ):
                ColumnParallelLinear(
                    Q_LORA_RANK, WQ_B_OUT, bias=False, tp_rank=0, tp_size=3
                )
        message = str(cm.exception)
        self.assertIn(f"Cannot partition dimension of size {WQ_B_OUT}", message)
        self.assertIn("sum(weights)=65167", message)

    def test_attn_sink_slice_follows_the_same_partition(self):
        """The per-rank sink slice is a prefix sum, not rank * width: under
        an uneven plan ranks 1 and 2 would otherwise read the wrong heads."""
        from sglang.srt.distributed.utils import tp_partition_offset

        set_tp_partition_ratios(AUTO_WEIGHTS)
        mods = self._ranks(3)
        starts = [
            tp_partition_offset(N_HEADS, 3, r, m.attn_tp_units)
            for r, m in enumerate(mods)
        ]
        self.assertEqual(starts, [0, 32, 48])
        # Contiguous, non-overlapping, and exactly covering the head axis.
        for start, m in zip(starts, mods):
            self.assertLessEqual(start + m.n_local_heads, N_HEADS)
        self.assertEqual(starts[-1] + mods[-1].n_local_heads, N_HEADS)


class TestExplicitVector(_V4AttentionCase):
    """An explicit --rank-tp-ratio also has to respect the coupling."""

    def test_explicit_ratio_snaps_to_whole_groups(self):
        """30,17,17 sums to 64, so the raw head split [30,17,17] was legal
        arithmetic -- and wrong: 30 heads do not fill whole o_groups, and the
        group split was `8 // 3 == 2` on every rank."""
        set_tp_partition_ratios(EXPLICIT_WEIGHTS)
        mods = self._ranks(3)
        self.assertEqual([m.n_local_heads for m in mods], [32, 16, 16])
        self.assertEqual([m.n_local_groups for m in mods], [4, 2, 2])
        self.assert_block_is_coherent(mods)

        # What the old code would have produced, for the record.
        self.assertEqual(
            [s // HEAD_DIM for s in partition_sizes(WQ_B_OUT, EXPLICIT_WEIGHTS)],
            [30, 17, 17],
        )

    def test_two_ranks(self):
        set_tp_partition_ratios([3, 1])
        mods = self._ranks(2)
        self.assertEqual([m.n_local_heads for m in mods], [48, 16])
        self.assertEqual([m.n_local_groups for m in mods], [6, 2])
        self.assert_block_is_coherent(mods)


class TestEvenPathUnchanged(_V4AttentionCase):
    """No plan installed: byte-identical to the pre-fix `// tp_size` split."""

    def test_even_split_matches_integer_division(self):
        for tp in (1, 2, 4, 8):
            for rank in range(tp):
                m = self._attn(tp, rank)
                with self.subTest(tp=tp, rank=rank):
                    self.assertEqual(m.n_local_heads, N_HEADS // tp)
                    self.assertEqual(m.n_local_groups, O_GROUPS // tp)
                    self.assertEqual(m.wq_b.output_size_per_partition, WQ_B_OUT // tp)
                    self.assertEqual(m.wo_a.output_size_per_partition, WO_OUT // tp)
                    self.assertEqual(m.wo_b.input_size_per_partition, WO_OUT // tp)

    def test_uniform_vector_is_the_even_split(self):
        set_tp_partition_ratios([1, 1, 1, 1])
        mods = self._ranks(4)
        self.assertEqual([m.n_local_heads for m in mods], [16] * 4)
        self.assertEqual([m.n_local_groups for m in mods], [2] * 4)
        self.assert_block_is_coherent(mods)


class TestGeometryRefusals(_V4AttentionCase):
    """Two configurations the o_group unit cannot express, refused by name."""

    def test_more_ranks_than_groups_is_refused(self):
        """Previously silent: `8 // 16 == 0` groups per rank, and the failure
        surfaced as a view() on a zero-sized dimension much later."""
        with self.assertRaises(ValueError) as cm:
            self._attn(16, 0)
        self.assertIn("o_groups=8", str(cm.exception))
        self.assertIn("16", str(cm.exception))

    def test_heads_not_a_multiple_of_groups_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            self._attn(1, 0, config=_config(num_attention_heads=63))
        self.assertIn("not a multiple of o_groups", str(cm.exception))


class TestAutoPerformanceUnitGrid(unittest.TestCase):
    """The planner's unit grid must be the model's unit count.

    ``PerfCostModel`` sizes candidate vectors against ``attn_units`` -- the
    comment at that assignment says so ("must match the model's tp_units so
    candidate vectors materialize identically to the real partition"). V4
    pins ``num_key_value_heads`` to 1, so the kv-head grid described nothing
    there and the planner fell into its replicated-KV branch and gridded on
    the 64 q heads, which is not what the layers partition.
    """

    _V4_TEXT = {
        "hidden_size": HIDDEN,
        "num_hidden_layers": 43,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "o_groups": O_GROUPS,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048,
        "vocab_size": 129280,
    }

    def _model(self, text, tp_size=3, base_plan=None):
        from unittest import mock

        from sglang.srt.uneven_perf import PerfCostModel, PlanInputs

        base_plan = base_plan or AUTO_WEIGHTS
        inputs = PlanInputs(tp_size=tp_size, model_path="<fixture>")
        with mock.patch.object(PerfCostModel, "_load_config", return_value=text):
            return PerfCostModel(
                inputs, base_plan=base_plan, budgets_mib=list(base_plan)
            )

    def test_v4_grids_on_o_groups(self):
        model = self._model(self._V4_TEXT)
        self.assertEqual(model.attn_units, O_GROUPS)

    def test_the_attn_shard_fractions_are_the_models_own_split(self):
        """The property the grid exists for: the planner's per-rank attention
        share equals what MqaAttentionBase actually materializes."""
        for weights in (AUTO_WEIGHTS, EXPLICIT_WEIGHTS, [5, 3, 3]):
            model = self._model(self._V4_TEXT, base_plan=weights)
            planned = model._shard_fractions("attn", weights)
            real = [g / O_GROUPS for g in partition_sizes(O_GROUPS, weights, O_GROUPS)]
            self.assertEqual(planned, real, msg=str(weights))

    def test_a_config_without_o_groups_keeps_the_kv_head_grid(self):
        text = dict(self._V4_TEXT)
        text.pop("o_groups")
        text["num_key_value_heads"] = 8
        self.assertEqual(self._model(text).attn_units, 8)


if __name__ == "__main__":
    unittest.main()
