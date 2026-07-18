"""Inertness + repro proof for the Qwen2MoeMLP uneven-TP fix (task #82).

Qwen2MoeMLP (the dense MLP of the qwen3.5/3.6 hybrid models and the MoE
shared expert) now derives its tp_units as

    units = intermediate_size // gcd(intermediate_size, 16)      # act vec
    units = _quant_block_aligned_units(intermediate, units, qc, 1)  # in-dim
                                                                    # block

and passes the SAME units to gate_up_proj and down_proj, mirroring
LlamaMLP (task #100 / 1244d1b2e) plus the caller-side input-dim
quant-block coarsening (mirroring qwen3_5's model-level gdn_tp_units
derivation). These tests prove, at the partition-helper level the layers
call into:

1. INERTNESS — with NO installed shard plan (the default / even-TP path)
   the split equals the pre-change classic even split exactly, and the
   divisibility assert is preserved.
2. REPRO — the pre-change element-granular units yield the exact
   kernel-fatal shards observed live at the TP=3 GGUF oracle boot of
   Qwen3.6-27B (task #82): [7790, 4809, 4809], rejected by the jit
   activation kernel ("hidden size must be divisible by vector size").
   Additionally, the down_proj input dim was quant-block-coarsened to
   256-element ggml units by linear.py while gate_up's output dim was not
   (GGUF skips output-dim coarsening), so the two layers PARTITIONED THE
   SAME DIMENSION DIFFERENTLY: [7680, 4864, 4864] vs [7790, 4809, 4809].
   Both failure modes are pinned here so the fix stays falsifiable.
3. FIX — the post-change units keep every rank's shard divisible by the
   widest activation vector (16) for bf16, and identical & 256-aligned on
   both layers for GGUF, at TP=3 and TP=5 (the task #82 emulation size).
4. GUARD — assert_activation_aligned_shards rejects an incompatible
   geometry at plan time with a clear error, and is a no-op on the
   default path.

Pure functions, no GPU, no server.
"""

import math
import unittest

from sglang.srt.distributed.utils import (
    ACTIVATION_VEC_ELEMS,
    assert_activation_aligned_shards,
    partition_units,
    set_tp_partition_ratios,
    tp_partition_sizes,
)
from sglang.srt.layers.linear import _quant_block_aligned_units
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# Qwen3.6-27B's intermediate_size and the auto ratio vector from the
# failing task-#82 TP=3 GGUF oracle boot (5090 + 2x3080).
INTERMEDIATE = 17408
T82_AUTO_WEIGHTS = [29607, 18280, 18280]
# A 3+1+1 co-location plan for the TP=5 emulation (three ranks share the
# 5090, whose per-rank budgets are smaller than a whole 3080's).
T82_TP5_WEIGHTS = [9869, 9869, 9869, 18280, 18280]

GGML_BLOCK = 256  # Q4_K superblock along the input dim


class _FakeGGUFConfig:
    """Just enough of GGUFConfig for _quant_block_aligned_units."""

    weight_block_size = [GGML_BLOCK, GGML_BLOCK]

    def get_name(self):
        return "gguf"


def _qwen2_moe_mlp_units(intermediate: int, quant_config=None) -> int:
    """Mirrors the derivation in sglang.srt.models.qwen2_moe.Qwen2MoeMLP."""
    units = intermediate // math.gcd(intermediate, 16)
    return _quant_block_aligned_units(intermediate, units, quant_config, 1)


class TestQwen2MoeMLPUnitsInertOnDefaultPath(CustomTestCase):
    """No installed plan -> units/family must be a no-op vs the old call."""

    def setUp(self):
        set_tp_partition_ratios(None)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_even_split_identical_to_pre_change(self):
        for qc in (None, _FakeGGUFConfig()):
            units = _qwen2_moe_mlp_units(INTERMEDIATE, qc)
            for tp in (1, 2, 4):
                pre = tp_partition_sizes(INTERMEDIATE, tp)  # old call
                post = tp_partition_sizes(
                    INTERMEDIATE, tp, units=units, family="mlp"
                )
                self.assertEqual(post, pre)
                self.assertEqual(post, [INTERMEDIATE // tp] * tp)

    def test_divisibility_assert_preserved(self):
        # 17408 % 3 != 0: the even path asserts identically pre/post
        # (ratios are empty -> same branch).
        units = _qwen2_moe_mlp_units(INTERMEDIATE)
        with self.assertRaises(AssertionError):
            tp_partition_sizes(INTERMEDIATE, 3)
        with self.assertRaises(AssertionError):
            tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")

    def test_guard_noop_without_plan(self):
        # Must not raise, and must not even compute a partition (17408/3
        # would assert in the even path if it did).
        assert_activation_aligned_shards(
            INTERMEDIATE, 3, _qwen2_moe_mlp_units(INTERMEDIATE)
        )


class TestQwen2MoeMLPUnitsFixUnderUnevenPlan(CustomTestCase):
    """Installed uneven plan -> old units are kernel-fatal AND internally
    inconsistent between gate_up and down_proj; new units are aligned and
    identical on both layers."""

    def setUp(self):
        set_tp_partition_ratios(T82_AUTO_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_pre_change_units_reproduce_boot_crash(self):
        # Element-granular units (old gate_up path under GGUF, where
        # output-dim quant coarsening is skipped): the exact shards from
        # the crashed task-#82 oracle boot; rejected by silu_and_mul
        # (vec 8 on Ampere, 16 on Blackwell).
        sizes = tp_partition_sizes(
            INTERMEDIATE, 3, units=INTERMEDIATE, family="mlp"
        )
        self.assertEqual(sizes, [7790, 4809, 4809])
        self.assertTrue(any(s % 8 for s in sizes))

    def test_pre_change_gate_down_partitions_disagreed(self):
        # down_proj's input dim WAS 256-coarsened by linear.py (GGUF
        # blocks lie along the input dim), giving a partition that
        # disagrees with gate_up's element-granular one — a latent weight/
        # activation shape mismatch hidden behind the activation crash.
        gate = tp_partition_sizes(
            INTERMEDIATE, 3, units=INTERMEDIATE, family="mlp"
        )
        down_units = _quant_block_aligned_units(
            INTERMEDIATE, INTERMEDIATE, _FakeGGUFConfig(), 1
        )
        down = tp_partition_sizes(INTERMEDIATE, 3, units=down_units, family="mlp")
        self.assertEqual(down, [7680, 4864, 4864])
        self.assertNotEqual(gate, down)

    def test_post_change_bf16_aligned(self):
        units = _qwen2_moe_mlp_units(INTERMEDIATE)  # unquantized
        self.assertEqual(units, INTERMEDIATE // 16)
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        self.assertEqual(sum(sizes), INTERMEDIATE)
        self.assertTrue(all(s % ACTIVATION_VEC_ELEMS == 0 for s in sizes))
        # units-mode == largest-remainder unit split scaled back up.
        self.assertEqual(
            sizes, [u * 16 for u in partition_units(units, T82_AUTO_WEIGHTS)]
        )
        assert_activation_aligned_shards(INTERMEDIATE, 3, units)  # no raise

    def test_post_change_gguf_aligned_and_consistent(self):
        qc = _FakeGGUFConfig()
        units = _qwen2_moe_mlp_units(INTERMEDIATE, qc)
        self.assertEqual(units, INTERMEDIATE // GGML_BLOCK)  # 68
        # Both layers receive these units; layer-level coarsening is an
        # idempotent pass-through (output dim: GGUF skip; input dim:
        # already whole ggml blocks).
        self.assertEqual(_quant_block_aligned_units(INTERMEDIATE, units, qc, 0), units)
        self.assertEqual(_quant_block_aligned_units(INTERMEDIATE, units, qc, 1), units)
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        self.assertEqual(sizes, [7680, 4864, 4864])
        self.assertTrue(all(s % GGML_BLOCK == 0 for s in sizes))
        self.assertTrue(all(s % ACTIVATION_VEC_ELEMS == 0 for s in sizes))
        assert_activation_aligned_shards(INTERMEDIATE, 3, units)  # no raise


class TestQwen2MoeMLPUnitsAtTP5(CustomTestCase):
    """The task-#82 emulation size: TP=5, kv(4) < ranks(5), 3+1+1
    co-location — the MLP split must stay aligned for both checkpoints."""

    def setUp(self):
        set_tp_partition_ratios(T82_TP5_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_tp5_gguf(self):
        units = _qwen2_moe_mlp_units(INTERMEDIATE, _FakeGGUFConfig())
        sizes = tp_partition_sizes(INTERMEDIATE, 5, units=units, family="mlp")
        self.assertEqual(sum(sizes), INTERMEDIATE)
        self.assertTrue(all(s >= 1 for s in sizes))
        self.assertTrue(all(s % GGML_BLOCK == 0 for s in sizes))
        assert_activation_aligned_shards(INTERMEDIATE, 5, units)

    def test_tp5_bf16(self):
        units = _qwen2_moe_mlp_units(INTERMEDIATE)
        sizes = tp_partition_sizes(INTERMEDIATE, 5, units=units, family="mlp")
        self.assertEqual(sum(sizes), INTERMEDIATE)
        self.assertTrue(all(s % ACTIVATION_VEC_ELEMS == 0 for s in sizes))
        assert_activation_aligned_shards(INTERMEDIATE, 5, units)

    def test_sweep_intermediates_and_tp(self):
        # Condition-3 sweep: alignment must hold for common qwen/gemma
        # intermediates across TP counts and both quant flavors.
        for total in (17408, 9728, 25600, 4304):
            for qc in (None, _FakeGGUFConfig()):
                units = _qwen2_moe_mlp_units(total, qc)
                for tp in (3, 5):
                    weights = (
                        T82_AUTO_WEIGHTS if tp == 3 else T82_TP5_WEIGHTS
                    )
                    set_tp_partition_ratios(weights)
                    if units < tp:
                        continue  # cannot give every rank a unit
                    sizes = tp_partition_sizes(total, tp, units=units, family="mlp")
                    self.assertEqual(sum(sizes), total)
                    self.assertTrue(
                        all(s % ACTIVATION_VEC_ELEMS == 0 for s in sizes),
                        f"total={total} tp={tp} qc={qc} sizes={sizes}",
                    )


class TestActivationAlignmentGuard(CustomTestCase):
    """assert_activation_aligned_shards fails fast, clearly, at plan time."""

    def setUp(self):
        set_tp_partition_ratios([1, 1, 1])

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_rejects_misaligned_geometry(self):
        # 17412 has gcd(17412, 16) == 4 -> 4-element units; every rank's
        # shard is 5804 (% 16 == 12): the next incompatible geometry must
        # error at boot, not at the first forward.
        total = 17412
        units = total // math.gcd(total, 16)
        with self.assertRaisesRegex(ValueError, "activation kernel"):
            assert_activation_aligned_shards(total, 3, units)

    def test_accepts_aligned_geometry(self):
        units = _qwen2_moe_mlp_units(INTERMEDIATE)
        assert_activation_aligned_shards(INTERMEDIATE, 3, units)

    def test_tp_size_mismatch_is_noop(self):
        # Plan of 3 does not apply to a tp_size-2 group.
        assert_activation_aligned_shards(17412, 2, 17412 // 4)


if __name__ == "__main__":
    unittest.main()
