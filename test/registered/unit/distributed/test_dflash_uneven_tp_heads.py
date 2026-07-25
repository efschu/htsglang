"""Uneven-TP head sharding for the DFLASH draft model (task #161).

Background: ``DFlashAttention.__init__`` inherited the stock even-TP
assumption from the upstream DFLASH feature commit (f08726fd56)::

    assert self.total_num_heads % tp_size == 0
    self.num_heads = self.total_num_heads // tp_size

On this fork that is a GUARD, not a kernel constraint. DFLASH's draft
attention is built from the ordinary TP primitives -- QKVParallelLinear /
RowParallelLinear / RadixAttention -- and every one of them already splits
UNEVENLY when a ``--rank-tp-ratio`` plan is installed (kv heads as the
indivisible GQA units). The assert therefore did not protect anything; it
merely refused the configuration that the rest of the stack supports, which
is why DFLASH could only run with ``--speculative-draft-placement solo`` on a
TP=3 rig. There is no DFLASH-specific attention kernel: the draft's
RadixAttention goes through the same backend as the target, and flashinfer
already derives per-rank head counts itself (``_local_attn_head_counts``,
commit f7ff51435).

These tests construct the REAL modules on CPU (no GPU, no server, no
distributed group) under an injected parallel context plus an installed
shard plan, and pin four things:

1. UNEVEN -- 32 q / 8 kv heads over TP=3 shard into whole GQA groups, and
   the qkv/o_proj/MLP weight shapes agree with the per-rank head counts.
   (Before the fix this raised the reported AssertionError.)
2. INERTNESS -- with no plan installed the even path is exactly the old
   behaviour, including the preserved divisibility assert.
3. RANK-UNIFORM COLLECTIVES -- head COUNTS may differ per rank, collective
   SEQUENCES may not: every rank runs the same all-reduces in the same
   order with the same payload shape (only the sharded input dim differs).
   Rank-divergent sequences show up as NCCL hangs, not wrong numbers.
4. FAIL-FAST -- kv < tp (REPLICATED-KV geometry) is rejected with a clear
   error instead of silently building shapes qkv_proj does not allocate.
"""

import math
import unittest
from types import SimpleNamespace

from sglang.srt.distributed.utils import set_tp_partition_ratios
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.models.dflash import (
    DFlashAttention,
    DFlashDecoderLayer,
    DFlashLagunaAttention,
    DFlashMLP,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# The shipped Qwen3.6-27B DFLASH draft head (qwen3.6-27b-dflash/config.json):
# 32 q heads, 8 kv heads, head_dim 128, intermediate 17408. 32 % 3 != 0 --
# exactly the configuration the assert rejected.
HIDDEN = 5120
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 17408

# A plain 2:1:1 plan (5090 + 2x 3080) and the memory-proportional vector an
# actual `--rank-tp-ratio auto-performance` boot produced on this rig.
RATIO_211 = [2, 1, 1]
AUTO_WEIGHTS = [16280, 29207, 17080]


def _config(**overrides):
    cfg = dict(
        hidden_size=HIDDEN,
        num_attention_heads=Q_HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=INTERMEDIATE,
        hidden_act="silu",
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_theta=10000000,
        rope_scaling=None,
        layer_types=["full_attention"],
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class _DFlashShardCase(CustomTestCase):
    """Builds DFLASH modules for one rank under an injected TP context."""

    def setUp(self):
        set_tp_partition_ratios(None)
        self._server_args = get_context().override_server_args()
        self._server_args.install()

    def tearDown(self):
        self._server_args.restore()
        set_tp_partition_ratios(None)

    def _build(self, factory, tp_size, tp_rank, config=None):
        config = config if config is not None else _config()
        with get_parallel().override(
            tp_size=tp_size,
            tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_tp_rank=tp_rank,
        ):
            return factory(config)

    def _attention(self, tp_size, tp_rank, config=None):
        return self._build(
            lambda c: DFlashAttention(c, layer_id=0), tp_size, tp_rank, config
        )

    def _mlp(self, tp_size, tp_rank, config=None):
        return self._build(DFlashMLP, tp_size, tp_rank, config)


class TestDFlashUnevenHeads(_DFlashShardCase):
    """32 q heads over TP=3: the case the guard rejected."""

    def test_heads_split_into_whole_gqa_groups(self):
        set_tp_partition_ratios(RATIO_211)
        attns = [self._attention(3, r) for r in range(3)]

        self.assertEqual([a.num_heads for a in attns], [16, 8, 8])
        self.assertEqual([a.num_kv_heads for a in attns], [4, 2, 2])
        # Exhaustive + disjoint: the shards reconstruct the full head set.
        self.assertEqual(sum(a.num_heads for a in attns), Q_HEADS)
        self.assertEqual(sum(a.num_kv_heads for a in attns), KV_HEADS)
        # Whole GQA groups per rank -- the attention kernels require
        # num_qo % num_kv == 0 on every rank.
        for a in attns:
            self.assertEqual(a.num_heads % a.num_kv_heads, 0)
            self.assertEqual(a.num_heads // a.num_kv_heads, Q_HEADS // KV_HEADS)

    def test_weight_shapes_match_the_per_rank_head_counts(self):
        """The real repro of the underlying defect: had the guard simply been
        deleted, ``total // tp`` would have disagreed with the shapes
        QKVParallelLinear actually allocates, and ``qkv.split([q_size,
        kv_size, kv_size])`` would have sliced the wrong columns."""
        set_tp_partition_ratios(RATIO_211)
        for rank, (q, kv) in enumerate([(16, 4), (8, 2), (8, 2)]):
            attn = self._attention(3, rank)
            self.assertEqual(attn.q_size, q * HEAD_DIM)
            self.assertEqual(attn.kv_size, kv * HEAD_DIM)
            # qkv_proj rows == q_size + 2 * kv_size for THIS rank
            self.assertEqual(
                attn.qkv_proj.weight.shape[0],
                attn.q_size + 2 * attn.kv_size,
            )
            self.assertEqual(attn.qkv_proj.weight.shape[1], HIDDEN)
            # o_proj consumes exactly this rank's q shard (tp_units).
            self.assertEqual(attn.o_proj.input_size_per_partition, attn.q_size)

    def test_mlp_shards_are_activation_kernel_aligned(self):
        """DFlashMLP needed the same 16-element unit family as LlamaMLP: the
        jit activation kernel rejects per-rank intermediate sizes that are
        not divisible by its vector width."""
        set_tp_partition_ratios(AUTO_WEIGHTS)
        mlps = [self._mlp(3, r) for r in range(3)]
        shards = [m.down_proj.input_size_per_partition for m in mlps]

        self.assertEqual(sum(shards), INTERMEDIATE)
        for s in shards:
            self.assertEqual(s % 16, 0, f"shard {s} is not 16-aligned")
        # gate_up packs [gate, up] -> two shards of the same size.
        for m, s in zip(mlps, shards):
            self.assertEqual(m.gate_up_proj.weight.shape[0], 2 * s)

    def test_realistic_auto_performance_plan(self):
        """The vector a real `--rank-tp-ratio auto-performance` boot emits;
        before the fix the attention assert fired first, and even without it
        the o_proj / MLP dims raised 'not divisible by sum(weights)'."""
        set_tp_partition_ratios(AUTO_WEIGHTS)
        attns = [self._attention(3, r) for r in range(3)]
        self.assertEqual([a.num_heads for a in attns], [8, 16, 8])
        self.assertEqual([a.num_kv_heads for a in attns], [2, 4, 2])
        self.assertEqual(sum(a.num_heads for a in attns), Q_HEADS)
        for a in attns:
            self.assertEqual(a.o_proj.input_size_per_partition, a.q_size)

    def test_laguna_gate_follows_the_q_split(self):
        """Per-head gating: the gate shard must line up with this rank's
        attention output, so it takes the same kv-head unit split."""
        set_tp_partition_ratios(RATIO_211)
        cfg = _config(gating="per-head")
        for rank, q in enumerate([16, 8, 8]):
            attn = self._build(
                lambda c: DFlashLagunaAttention(c, layer_id=0), 3, rank, cfg
            )
            self.assertEqual(attn.num_heads, q)
            self.assertEqual(attn.g_proj.weight.shape[0], q)

        cfg_full = _config(gating="per-element")
        for rank, q in enumerate([16, 8, 8]):
            attn = self._build(
                lambda c: DFlashLagunaAttention(c, layer_id=0), 3, rank, cfg_full
            )
            self.assertEqual(attn.g_proj.weight.shape[0], q * HEAD_DIM)


class TestDFlashEvenTPUnchanged(_DFlashShardCase):
    """Flag OFF / evenly divisible: exactly today's behaviour."""

    def test_even_split_unchanged(self):
        for tp in (1, 2, 4, 8):
            for rank in range(tp):
                attn = self._attention(tp, rank)
                self.assertEqual(attn.num_heads, Q_HEADS // tp)
                self.assertEqual(attn.num_kv_heads, max(1, KV_HEADS // tp))
                self.assertEqual(attn.q_size, (Q_HEADS // tp) * HEAD_DIM)
                self.assertEqual(
                    attn.qkv_proj.weight.shape[0],
                    attn.q_size + 2 * attn.kv_size,
                )
                self.assertEqual(attn.o_proj.input_size_per_partition, attn.q_size)

                mlp = self._mlp(tp, rank)
                self.assertEqual(
                    mlp.down_proj.input_size_per_partition, INTERMEDIATE // tp
                )

    def test_even_split_unchanged_with_a_uniform_plan_installed(self):
        """A uniform ratio vector IS the even split -- must stay identical to
        the no-plan path (the uneven branch is entered, and must agree)."""
        for tp in (2, 4):
            for rank in range(tp):
                set_tp_partition_ratios(None)
                base = self._attention(tp, rank)
                set_tp_partition_ratios([1] * tp)
                planned = self._attention(tp, rank)
                self.assertEqual(planned.num_heads, base.num_heads)
                self.assertEqual(planned.num_kv_heads, base.num_kv_heads)
                self.assertEqual(planned.q_size, base.q_size)
                self.assertEqual(
                    planned.o_proj.input_size_per_partition,
                    base.o_proj.input_size_per_partition,
                )

    def test_divisibility_assert_preserved_without_a_plan(self):
        # No plan installed and 32 % 3 != 0: the classic assert must still
        # fire -- the fix must not turn an unsupported even-TP config into a
        # silently wrong one.
        with self.assertRaises(AssertionError):
            self._attention(3, 0)

    def test_mlp_units_are_inert_without_a_plan(self):
        units = INTERMEDIATE // math.gcd(INTERMEDIATE, 16)
        self.assertEqual(INTERMEDIATE % units, 0)
        for tp in (1, 2, 4):
            mlp = self._mlp(tp, 0)
            self.assertEqual(
                mlp.down_proj.input_size_per_partition, INTERMEDIATE // tp
            )


class TestDFlashRankUniformCollectives(_DFlashShardCase):
    """Uneven head COUNTS are fine; divergent collective SEQUENCES are not.

    Every rank must issue the same all-reduces, in the same order, with the
    same payload shape -- only the sharded input dim may differ. In this fork
    a rank-divergent sequence has repeatedly surfaced as an NCCL hang rather
    than a wrong result, so it is pinned explicitly.
    """

    def _reduce_signature(self, tp_size, tp_rank):
        """Ordered signature of the collective-carrying layers in one decoder
        layer: (qualified name, all-reduce payload width, reduce enabled)."""
        with get_parallel().override(
            tp_size=tp_size,
            tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_tp_rank=tp_rank,
        ):
            layer = DFlashDecoderLayer(_config(), layer_id=0)
        signature = []
        for name, module in layer.named_modules():
            if isinstance(module, RowParallelLinear):
                signature.append(
                    (
                        name,
                        int(module.output_size),
                        bool(getattr(module, "reduce_results", True)),
                    )
                )
        return signature

    def test_collective_sequence_identical_across_uneven_ranks(self):
        set_tp_partition_ratios(RATIO_211)
        signatures = [self._reduce_signature(3, r) for r in range(3)]
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])
        # Sanity: the layer really does carry the two expected reduces
        # (attention o_proj and MLP down_proj), both over hidden_size.
        self.assertEqual(len(signatures[0]), 2)
        for _, payload, _ in signatures[0]:
            self.assertEqual(payload, HIDDEN)

    def test_collective_sequence_identical_under_auto_weights(self):
        set_tp_partition_ratios(AUTO_WEIGHTS)
        signatures = [self._reduce_signature(3, r) for r in range(3)]
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])

    def test_only_the_sharded_dim_differs(self):
        set_tp_partition_ratios(RATIO_211)
        attns = [self._attention(3, r) for r in range(3)]
        # Reduce payload identical on every rank ...
        self.assertEqual({int(a.o_proj.output_size) for a in attns}, {HIDDEN})
        # ... while the contracted (sharded) dim is what differs.
        self.assertEqual(
            [a.o_proj.input_size_per_partition for a in attns],
            [16 * HEAD_DIM, 8 * HEAD_DIM, 8 * HEAD_DIM],
        )


class TestDFlashUnevenFailFast(_DFlashShardCase):
    """kv < tp under a plan: reject clearly instead of mis-shaping."""

    def test_replicated_kv_geometry_rejected(self):
        cfg = _config(num_key_value_heads=2)
        set_tp_partition_ratios(RATIO_211)
        with self.assertRaisesRegex(ValueError, "total_num_kv_heads >= tp_size"):
            self._attention(3, 0, cfg)


if __name__ == "__main__":
    unittest.main()
