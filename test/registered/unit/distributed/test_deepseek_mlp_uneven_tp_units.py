# SPDX-License-Identifier: Apache-2.0
"""DeepseekV2MLP declares its uneven-TP unit family (#405).

THE DEFECT, as it stood
``DeepseekV2MLP`` built both projections with no unit count::

    self.gate_up_proj = MergedColumnParallelLinear(hidden, [intermediate]*2, ...)
    self.down_proj    = RowParallelLinear(intermediate, hidden, ...)

``--rank-tp-ratio auto`` derives its weights from the NVML budgets, so they are
BYTE counts -- ``[29607, 17780, 17780]`` on this rig, summing to 65167. Without
a unit count ``partition_sizes`` requires the dimension to be divisible by that
sum and refuses. #402 fixed the DeepSeek-V4-Flash attention block (o_groups on
wq_b / wo_a / wo_b) and recorded this module as the NEXT stopper on the same
boot: V4's shared expert is a ``DeepseekV2MLP`` with intermediate
``moe_intermediate_size * n_shared_experts`` = 2048, and it aborted with

    ValueError: Cannot partition dimension of size 2048 with weight vector
    [29607, 17780, 17780]: 2048 is not divisible by sum(weights)=65167.

THE UNIT IS THE ACTIVATION-VECTOR GROUP, COARSENED TO THE QUANT BLOCK
The family is the 16-element MLP grain of the a6f7192c2 lineage (LlamaMLP #100,
Qwen2MoeMLP #82): the jit activation kernel vector-loads up to 32 bytes and
rejects per-rank intermediate sizes that are not a multiple of its width.

On top of that comes the #385 coupled-dimension rule: gate_up's OUTPUT and
down_proj's INPUT are the SAME intermediate dimension and must coarsen
IDENTICALLY. They do not if the coarsening is left to the layers -- the
layer-level pass skips the output dim for GGUF (ggml blocks lie along the input
dim), so gate_up would keep 128 16-element units while down_proj drops to 8
256-element ones, and the two ends of the residual would be built for different
per-rank intermediate sizes (measured below: [928,560,560] vs [1024,512,512]).
The units are therefore derived ONCE in ``DeepseekV2MLP.__init__`` and handed to
both layers.

CPU only, real modules on the meta device: no GPU, no distributed group.
"""

import math
import unittest

import torch

from sglang.srt.distributed.utils import (
    ACTIVATION_VEC_ELEMS,
    get_tp_partition_ratios,
    partition_sizes,
    set_tp_partition_ratios,
)
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# DeepSeek-V4-Flash-0731, verbatim from its config.json.
HIDDEN = 4096
MOE_INTERMEDIATE = 2048
N_SHARED_EXPERTS = 1
#: What DeepseekV2MoE hands the shared expert.
SHARED_INTERMEDIATE = MOE_INTERMEDIATE * N_SHARED_EXPERTS  # 2048
#: A dense DeepSeek V2/V3 layer's MLP, for the non-V4 half of the sweep.
DENSE_INTERMEDIATE = 18432

#: The vector `--rank-tp-ratio auto` emitted on this rig (NVML totals minus
#: the per-card reserve, gcd-reduced to 1). Sums to 65167; divides nothing.
AUTO_WEIGHTS = [29607, 17780, 17780]
#: The hand-picked vector the same rig is normally booted with. Sums to 64.
EXPLICIT_WEIGHTS = [30, 17, 17]

#: GGUF's superblock, both axes (GGUFConfig.weight_block_size).
GGUF_BLOCK = 256


def _gguf_config():
    from sglang.srt.layers.quantization.gguf import GGUFConfig

    return GGUFConfig()


class _MLPCase(CustomTestCase):
    """Builds the real DeepseekV2MLP for one rank under an injected plan."""

    def setUp(self):
        self._saved = get_tp_partition_ratios()
        set_tp_partition_ratios(None)
        self._server_args = get_context().override_server_args()
        self._server_args.install()

    def tearDown(self):
        self._server_args.restore()
        set_tp_partition_ratios(self._saved)

    def _mlp(self, group_size, group_rank, intermediate, quant_config=None, **kwargs):
        with torch.device("meta"), get_parallel().override(
            tp_size=group_size, tp_rank=group_rank
        ):
            return DeepseekV2MLP(
                hidden_size=HIDDEN,
                intermediate_size=intermediate,
                hidden_act="silu",
                quant_config=quant_config,
                prefix="mlp",
                **kwargs,
            )

    def _ranks(self, group_size, intermediate, quant_config=None, **kwargs):
        return [
            self._mlp(group_size, r, intermediate, quant_config, **kwargs)
            for r in range(group_size)
        ]

    def assert_mlp_is_coherent(self, mods, intermediate):
        """THE CONTRACT: the two projections partition the shared intermediate
        dimension identically, and the shards tile it exactly once."""
        gate_up_shards = []
        for rank, m in enumerate(mods):
            with self.subTest(rank=rank):
                # A merged 2x layer: both parts get the SAME per-rank width,
                # because tp_units is defined against the per-part size.
                parts = m.gate_up_proj.output_partition_sizes
                self.assertEqual(len(parts), 2)
                self.assertEqual(parts[0], parts[1])
                # gate_up OUT-shard == down IN-shard: the coupled dimension.
                self.assertEqual(parts[0], m.down_proj.input_size_per_partition)
                # The silu_and_mul input is the merged 2x block.
                self.assertEqual(m.gate_up_proj.output_size_per_partition, 2 * parts[0])
                # Every shard is activation-kernel loadable.
                self.assertEqual(parts[0] % ACTIVATION_VEC_ELEMS, 0)
                # down_proj is row-parallel: its OUTPUT stays global.
                self.assertEqual(m.down_proj.output_size, HIDDEN)
                gate_up_shards.append(parts[0])
        # Exhaustive and disjoint.
        self.assertEqual(sum(gate_up_shards), intermediate)
        self.assertEqual(
            sum(m.down_proj.input_size_per_partition for m in mods), intermediate
        )
        return gate_up_shards


class TestV4SharedExpertAutoVector(_MLPCase):
    """The byte-valued vector on the V4-Flash shared-expert geometry."""

    def test_auto_vector_partitions_the_shared_expert(self):
        set_tp_partition_ratios(AUTO_WEIGHTS)
        mods = self._ranks(3, SHARED_INTERMEDIATE, _gguf_config())

        # 2048 in 256-element GGUF superblocks = 8 units -> [4, 2, 2].
        self.assertEqual([m.gate_up_proj.tp_units for m in mods], [8] * 3)
        self.assertEqual([m.down_proj.tp_units for m in mods], [8] * 3)
        shards = self.assert_mlp_is_coherent(mods, SHARED_INTERMEDIATE)
        self.assertEqual(shards, [1024, 512, 512])
        # Whole quant blocks on both axes.
        for s in shards:
            self.assertEqual(s % GGUF_BLOCK, 0)

    def test_family_tag_is_declared(self):
        set_tp_partition_ratios(AUTO_WEIGHTS)
        m = self._mlp(3, 0, SHARED_INTERMEDIATE, _gguf_config())
        self.assertEqual(m.gate_up_proj.tp_family, "mlp")
        self.assertEqual(m.down_proj.tp_family, "mlp")

    def test_declaration_removed_reproduces_the_boot_error(self):
        """Can-fail. The same two layers WITHOUT the unit count are the
        pre-fix code, and each raises the boot's exact message."""
        set_tp_partition_ratios(AUTO_WEIGHTS)
        quant_config = _gguf_config()
        for build in (
            lambda: MergedColumnParallelLinear(
                HIDDEN,
                [SHARED_INTERMEDIATE] * 2,
                bias=False,
                quant_config=quant_config,
                prefix="gate_up_proj",
            ),
            lambda: RowParallelLinear(
                SHARED_INTERMEDIATE,
                HIDDEN,
                bias=False,
                quant_config=quant_config,
                prefix="down_proj",
            ),
        ):
            with self.subTest(build=build):
                with self.assertRaises(ValueError) as cm:
                    with torch.device("meta"), get_parallel().override(
                        tp_size=3, tp_rank=0
                    ):
                        build()
                message = str(cm.exception)
                self.assertIn(
                    f"Cannot partition dimension of size {SHARED_INTERMEDIATE}",
                    message,
                )
                self.assertIn("sum(weights)=65167", message)

    def test_coupled_dimension_needs_the_caller_side_coarsening(self):
        """Can-fail for the #385 rule. Declaring the RAW 16-element family and
        letting each layer coarsen on its own is not enough: the layer-level
        pass skips the output dim for GGUF, so the two ends of the residual
        are built for different per-rank intermediate sizes."""
        set_tp_partition_ratios(AUTO_WEIGHTS)
        quant_config = _gguf_config()
        raw_units = SHARED_INTERMEDIATE // math.gcd(
            SHARED_INTERMEDIATE, ACTIVATION_VEC_ELEMS
        )
        self.assertEqual(raw_units, 128)
        with torch.device("meta"), get_parallel().override(tp_size=3, tp_rank=0):
            gate_up = MergedColumnParallelLinear(
                HIDDEN,
                [SHARED_INTERMEDIATE] * 2,
                bias=False,
                quant_config=quant_config,
                prefix="gate_up_proj",
                tp_units=raw_units,
                tp_family="mlp",
            )
            down = RowParallelLinear(
                SHARED_INTERMEDIATE,
                HIDDEN,
                bias=False,
                quant_config=quant_config,
                prefix="down_proj",
                tp_units=raw_units,
                tp_family="mlp",
            )
        self.assertNotEqual(gate_up.tp_units, down.tp_units)
        self.assertEqual(
            partition_sizes(SHARED_INTERMEDIATE, AUTO_WEIGHTS, gate_up.tp_units),
            [928, 560, 560],
        )
        self.assertEqual(
            partition_sizes(SHARED_INTERMEDIATE, AUTO_WEIGHTS, down.tp_units),
            [1024, 512, 512],
        )
        # What the module builds instead: one count, both layers.
        m = self._mlp(3, 0, SHARED_INTERMEDIATE, quant_config)
        self.assertEqual(m.gate_up_proj.tp_units, m.down_proj.tp_units)
        self.assertEqual(m.gate_up_proj.tp_units, SHARED_INTERMEDIATE // GGUF_BLOCK)

    def test_two_and_four_ranks(self):
        for weights, expected in (
            ([3, 1], [1536, 512]),
            ([29607, 17780, 17780, 17780], [768, 512, 512, 256]),
        ):
            with self.subTest(weights=weights):
                set_tp_partition_ratios(weights)
                mods = self._ranks(len(weights), SHARED_INTERMEDIATE, _gguf_config())
                self.assertEqual(
                    self.assert_mlp_is_coherent(mods, SHARED_INTERMEDIATE), expected
                )


class TestExplicitVector(_MLPCase):
    """Explicit --rank-tp-ratio: whole-unit vectors are unchanged."""

    def test_dense_mlp_explicit_ratio_is_byte_identical(self):
        """18432 over [30,17,17] divides exactly (sum 64), so the unit family
        must reproduce the pre-fix split element for element."""
        set_tp_partition_ratios(EXPLICIT_WEIGHTS)
        mods = self._ranks(3, DENSE_INTERMEDIATE)
        shards = self.assert_mlp_is_coherent(mods, DENSE_INTERMEDIATE)
        self.assertEqual(shards, partition_sizes(DENSE_INTERMEDIATE, EXPLICIT_WEIGHTS))
        self.assertEqual(shards, [8640, 4896, 4896])

    def test_v4_shared_expert_explicit_ratio_snaps_to_quant_blocks(self):
        """2048 over [30,17,17] was legal arithmetic -- and wrong for a GGUF
        checkpoint: [960,544,544] cuts inside the 256-element superblock. It
        now lands on the same [1024,512,512] the auto vector produces."""
        set_tp_partition_ratios(EXPLICIT_WEIGHTS)
        mods = self._ranks(3, SHARED_INTERMEDIATE, _gguf_config())
        self.assertEqual(
            self.assert_mlp_is_coherent(mods, SHARED_INTERMEDIATE),
            [1024, 512, 512],
        )
        # What the old code produced, for the record.
        self.assertEqual(
            partition_sizes(SHARED_INTERMEDIATE, EXPLICIT_WEIGHTS), [960, 544, 544]
        )


class TestEvenPathUnchanged(_MLPCase):
    """No plan installed: byte-identical to the pre-fix divide() split."""

    def test_even_split_matches_integer_division(self):
        for intermediate, quant_config in (
            (DENSE_INTERMEDIATE, None),
            (SHARED_INTERMEDIATE, _gguf_config()),
        ):
            for tp in (1, 2, 4, 8):
                for rank in range(tp):
                    m = self._mlp(tp, rank, intermediate, quant_config)
                    with self.subTest(inter=intermediate, tp=tp, rank=rank):
                        self.assertEqual(
                            m.gate_up_proj.output_partition_sizes,
                            [intermediate // tp] * 2,
                        )
                        self.assertEqual(
                            m.down_proj.input_size_per_partition, intermediate // tp
                        )

    def test_uniform_vector_is_the_even_split(self):
        set_tp_partition_ratios([1, 1, 1, 1])
        mods = self._ranks(4, SHARED_INTERMEDIATE, _gguf_config())
        self.assertEqual(
            self.assert_mlp_is_coherent(mods, SHARED_INTERMEDIATE), [512] * 4
        )


class TestReplicatedSharedExpert(_MLPCase):
    """tp_size=1 (SGLANG_SHARED_EXPERT_TP1, DeepEP/FP4 allgather, moe-dense-dp):
    the module is replicated and must not join the "mlp" family."""

    def test_replicated_declares_no_units_and_keeps_the_full_dimension(self):
        set_tp_partition_ratios(AUTO_WEIGHTS)
        m = self._mlp(3, 0, SHARED_INTERMEDIATE, _gguf_config(), tp_rank=0, tp_size=1)
        self.assertIsNone(m.gate_up_proj.tp_units)
        self.assertIsNone(m.gate_up_proj.tp_family)
        self.assertIsNone(m.down_proj.tp_units)
        self.assertIsNone(m.down_proj.tp_family)
        self.assertEqual(
            m.gate_up_proj.output_partition_sizes, [SHARED_INTERMEDIATE] * 2
        )
        self.assertEqual(m.down_proj.input_size_per_partition, SHARED_INTERMEDIATE)


class TestActivationAlignmentGuard(_MLPCase):
    """A geometry no unit grain can rescue is refused at construction, not at
    the first forward (the kernel would only complain after weight load)."""

    def test_misaligned_intermediate_is_refused_by_name(self):
        set_tp_partition_ratios(AUTO_WEIGHTS)
        # gcd(2056, 16) == 8, so the finest legal grain is 8 elements and the
        # plan lands on [936, 560, 560]; 936 is not a multiple of 16.
        with self.assertRaises(ValueError) as cm:
            self._mlp(3, 0, 2056)
        message = str(cm.exception)
        self.assertIn("MLP intermediate", message)
        self.assertIn("[936, 560, 560]", message)
        self.assertIn(str(ACTIVATION_VEC_ELEMS), message)

    def test_the_same_geometry_is_fine_without_a_plan(self):
        m = self._mlp(1, 0, 2056)
        self.assertEqual(m.gate_up_proj.output_partition_sizes, [2056] * 2)


class TestNonDeepseekPathUntouched(_MLPCase):
    """LlamaMLP already declares the same family; the two must agree, and
    nothing outside deepseek_v2 changed to make that true."""

    def test_llama_and_deepseek_partition_identically(self):
        from sglang.srt.models.llama import LlamaMLP

        for weights in (AUTO_WEIGHTS, EXPLICIT_WEIGHTS, [5, 3, 3]):
            for intermediate in (DENSE_INTERMEDIATE, 11008, SHARED_INTERMEDIATE):
                with self.subTest(weights=weights, inter=intermediate):
                    set_tp_partition_ratios(weights)
                    ds = self._ranks(3, intermediate)
                    with torch.device("meta"):
                        llama = [
                            LlamaMLP(
                                hidden_size=HIDDEN,
                                intermediate_size=intermediate,
                                hidden_act="silu",
                                prefix="mlp",
                                tp_rank=r,
                                tp_size=3,
                            )
                            for r in range(3)
                        ]
                    self.assertEqual(
                        [m.gate_up_proj.output_partition_sizes[0] for m in ds],
                        [m.gate_up_proj.output_partition_sizes[0] for m in llama],
                    )
                    self.assertEqual(
                        [m.down_proj.input_size_per_partition for m in ds],
                        [m.down_proj.input_size_per_partition for m in llama],
                    )


if __name__ == "__main__":
    unittest.main()
