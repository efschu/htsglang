"""Inertness + repro proof for the LlamaMLP uneven-TP fix (task #100).

LlamaMLP now passes tp_units=intermediate_size // gcd(intermediate_size, 16),
tp_family="mlp" to its gate_up_proj / down_proj. These tests prove, at the
partition-helper level the layers call into:

1. INERTNESS — with NO installed shard plan (the default / even-TP path),
   adding units+family changes NOTHING: the split equals the pre-change
   classic even split exactly, and the divisibility assert is preserved.
   (tp_partition_sizes short-circuits to the even split before consulting
   units when no ratio vector is installed.)
2. REPRO — under an installed uneven plan (--rank-tp-ratio), the
   pre-change call (no units) raises the exact ValueError observed at
   EAGLE3 draft init (16384 intermediate vs auto weights), while the
   post-change call partitions whole units.
3. ACT-KERNEL ALIGNMENT — element-granular units yield odd shards under
   the T92 auto plan ([4263, 7648, 4473]), which the jit activation
   kernel rejects ("hidden size must be divisible by vector size",
   kMaxVecBytes=32 -> 16 bf16 elements on Blackwell). The 16-element
   units LlamaMLP now uses keep every rank's shard divisible by 16.

Pure functions, no GPU, no server.
"""

import math
import unittest

from sglang.srt.distributed.utils import (
    partition_units,
    set_tp_partition_ratios,
    tp_partition_sizes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The TW Gemma-4-31B-Eagle3 draft head's intermediate_size, and the auto
# ratio vector from the failing T92 boot (5090 + 2x3080).
INTERMEDIATE = 16384
T92_AUTO_WEIGHTS = [16280, 29207, 17080]


def _llama_mlp_units(intermediate: int) -> int:
    """Mirrors the derivation in sglang.srt.models.llama.LlamaMLP."""
    return intermediate // math.gcd(intermediate, 16)


class TestLlamaMLPUnitsInertOnDefaultPath(CustomTestCase):
    """No installed plan -> units/family must be a no-op vs the old call."""

    def setUp(self):
        set_tp_partition_ratios(None)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_even_split_identical_to_pre_change(self):
        units = _llama_mlp_units(INTERMEDIATE)
        for tp in (1, 2, 4, 8, 16):
            pre = tp_partition_sizes(INTERMEDIATE, tp)  # old LlamaMLP call
            post = tp_partition_sizes(  # new LlamaMLP call
                INTERMEDIATE, tp, units=units, family="mlp"
            )
            self.assertEqual(post, pre)
            self.assertEqual(post, [INTERMEDIATE // tp] * tp)

    def test_divisibility_assert_preserved(self):
        # 16384 % 3 != 0: the pre-change path asserts; the post-change
        # call must fail identically (same branch — ratios are empty).
        units = _llama_mlp_units(INTERMEDIATE)
        with self.assertRaises(AssertionError):
            tp_partition_sizes(INTERMEDIATE, 3)
        with self.assertRaises(AssertionError):
            tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")

    def test_other_sizes_sweep(self):
        # Common llama-family intermediates; includes 11008 (gcd 16 -> 16)
        # and a deliberately 16-indivisible size exercising the gcd
        # fallback (22016 % 16 == 0, 5504 % 16 == 0, 1376 % 16 == 0,
        # 344 has gcd(344, 16) == 8).
        for total in (11008, 14336, 21504, 16384, 344):
            units = _llama_mlp_units(total)
            self.assertEqual(total % units, 0)
            for tp in (1, 2, 4):
                if total % tp:
                    continue
                self.assertEqual(
                    tp_partition_sizes(total, tp, units=units, family="mlp"),
                    tp_partition_sizes(total, tp),
                )


class TestLlamaMLPUnitsFixUnderUnevenPlan(CustomTestCase):
    """Installed uneven plan -> old call raises (the T92 boot crash), new
    call partitions aligned whole units."""

    def setUp(self):
        set_tp_partition_ratios(T92_AUTO_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_pre_change_call_reproduces_boot_crash(self):
        with self.assertRaisesRegex(ValueError, "not divisible by"):
            tp_partition_sizes(INTERMEDIATE, 3)

    def test_element_granular_units_are_kernel_incompatible(self):
        # The first-cut fix (units == intermediate, gemma-style) produced
        # odd per-rank shards that the jit activation kernel rejects at
        # CUDA graph capture — pin the repro so the alignment rationale
        # stays falsifiable.
        sizes = tp_partition_sizes(
            INTERMEDIATE, 3, units=INTERMEDIATE, family="mlp"
        )
        self.assertEqual(sizes, [4263, 7648, 4473])
        self.assertTrue(any(s % 16 for s in sizes))

    def test_post_change_call_partitions_aligned(self):
        units = _llama_mlp_units(INTERMEDIATE)
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        self.assertEqual(sum(sizes), INTERMEDIATE)
        self.assertTrue(all(s >= 1 for s in sizes))
        # Every rank's shard is compatible with the activation kernel's
        # widest vector (16 bf16 elements).
        self.assertTrue(all(s % 16 == 0 for s in sizes))
        # units-mode == largest-remainder unit split scaled by 16.
        self.assertEqual(
            sizes, [u * 16 for u in partition_units(units, T92_AUTO_WEIGHTS)]
        )

    def test_family_vector_respected_when_installed(self):
        units = _llama_mlp_units(INTERMEDIATE)
        set_tp_partition_ratios(T92_AUTO_WEIGHTS, families={"mlp": [1, 2, 1]})
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        self.assertEqual(sizes, [4096, 8192, 4096])


if __name__ == "__main__":
    unittest.main()
