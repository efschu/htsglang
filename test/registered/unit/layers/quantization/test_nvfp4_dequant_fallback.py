"""Task #332, posten 1: a quantised projection with no Marlin form is loaded
packed and materialised dense instead of killing the boot.

The blocker, falsified on the card at TP=1 (2026-07-31, see the "NVFP4-Beleg"
section of ``docs/dev/INTEGRATION_R3_VALIDATION.md``): the all-Linear NVFP4
checkpoint ``ocicek/Qwen3.6-27B-NVFP4`` targets ``Linear`` wholesale, so the
gated-delta-net's merged b/a gate is quantised too. Its unsharded width is
2 x 48 = 96 rows against a 64-wide Marlin tile, so NO shard plan and NO
pre-Blackwell card can serve it -- the same abort appears at TP=1 with no plan
in the picture.

What is pinned here:

* the dequantisation is arithmetically exact against an independent
  element-by-element reference, including the compressed-tensors scale
  DIRECTION (divide by ``weight_global_scale``);
* the guard ORDER: the geometry verdict is taken on the UNSHARDED width
  (#316's rule), it decides WHICH layers take the lane, and only then does the
  consequence become "dequantise" instead of "refuse";
* a tile-legal layer is untouched -- it still takes the Marlin lane, and a
  shard that misses the tile under an uneven plan still dies loudly;
* the lane change is announced, per layer, by name.

Task #336 extends the same lane to the NATIVE FP4 rank, which #332 left out.
Measured on the card 2026-07-31 over three uneven ratios, TP0 dying in the
first forward at ``nvfp4_scaled_mm_kernels.cuh:81`` ("Expected n to be
divisible by 32"): ``auto`` -> n 42, ``2,1,1`` -> n 48, ``4,1,1`` -> n 60. The
three points falsified the naive proportional model (which predicted 64 for
the third) and produced the real one: the gated-delta-net splits in groups of
three value heads, so ``n = 6 * floor(16 * r0/sum r)``, and ``n % 32 == 0``
then forces every group onto one rank. NO tensor-parallel split of that layer
satisfies the native lane, the even one included -- so the fix is the dequant
lane, not a better shard plan. Those three points are fixtures below.

CPU-only: pure tensor arithmetic and mocked backend resolution.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import contextlib
import logging
import unittest

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
    CompressedTensorsW4A4Fp4Dequant,
    CompressedTensorsW8A16Fp8,
    nvfp4_marlin_unpackable_reason,
    nvfp4_native_unpackable_reason,
    nvfp4_unpackable_reason,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (
    NVFP4_BLOCK_SIZE,
    NVFP4_NATIVE_MIN_N,
    dequantize_nvfp4,
)
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.marlin_utils import GPTQ_MARLIN_MIN_THREAD_N
from sglang.test.test_utils import CustomTestCase

HIDDEN = 5120
#: Qwen3.5/3.6 gated-delta-net: in_proj_b and in_proj_a, 48 rows each.
GDN_BA_SHARD = 48
GDN_BA_WIDTH = 2 * GDN_BA_SHARD
#: The per-rank shards 96 splits into under the fork's 3-rank uneven plan.
UNEVEN_BA_SHARDS = (42, 30, 24)
#: Qwen3.6-27B-NVFP4 GDN: 48 value heads against 16 key heads -> the split
#: moves whole groups of three value heads, and the merged b/a gate carries
#: two rows per value head.
GDN_VALUE_HEADS = 48
GDN_KEY_HEADS = 16
GDN_ROWS_PER_GROUP = 2 * (GDN_VALUE_HEADS // GDN_KEY_HEADS)
#: The three rank-0 widths measured on the card, 2026-07-31:
#: ``ratio -> (groups on rank 0, n reported by the kernel assert)``.
#: ``4,1,1`` is the point that falsified the naive proportional model (it
#: predicted 32 heads / n 64; the card said 30 heads / n 60). ``auto`` derives
#: its weights from NVML, so its group count is read back from the
#: measurement rather than from a ratio.
MEASURED_TP0 = {"auto": (7, 42), "2,1,1": (8, 48), "4,1,1": (10, 60)}
#: The rank-0 weight and weight sum of the two EXPLICIT ratios above.
MEASURED_RATIO_WEIGHTS = {"2,1,1": (2, 4), "4,1,1": (4, 6)}

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@contextlib.contextmanager
def _fp4_backend(backend: Fp4GemmRunnerBackend):
    previous = fp4_utils.FP4_GEMM_RUNNER_BACKEND
    fp4_utils.FP4_GEMM_RUNNER_BACKEND = backend
    try:
        yield
    finally:
        fp4_utils.FP4_GEMM_RUNNER_BACKEND = previous


def _fake_linear(output_size: int, input_size: int = HIDDEN) -> LinearBase:
    """A ``LinearBase`` carrying only the geometry the verdict reads.

    ``LinearBase.__init__`` sets ``input_size`` / ``output_size`` (the
    UNSHARDED sizes) and then immediately calls
    ``quant_config.get_quant_method(self, ...)``; the per-partition sizes are
    computed by the subclass afterwards. So this is exactly the state the
    verdict sees on a real boot.
    """
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.input_size = input_size
    layer.output_size = output_size
    return layer


def _reference_dequantize(packed, scales, global_scale):
    """Element-by-element NVFP4 dequantisation, written independently.

    Deliberately a slow explicit loop over nibbles: it shares no code with the
    implementation, so agreement is evidence and not a tautology.
    """
    n, k_half = packed.shape
    k = k_half * 2
    out = torch.empty(n, k, dtype=torch.float32)
    for row in range(n):
        for col in range(k):
            byte = int(packed[row, col // 2])
            nibble = (byte & 0x0F) if col % 2 == 0 else ((byte >> 4) & 0x0F)
            magnitude = _E2M1[nibble & 0x07]
            value = -magnitude if nibble & 0x08 else magnitude
            block_scale = float(scales[row, col // NVFP4_BLOCK_SIZE].float())
            out[row, col] = value * block_scale / float(global_scale)
    return out


def _random_nvfp4(n: int, k: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=generator)
    scales = (torch.rand(n, k // NVFP4_BLOCK_SIZE, generator=generator) * 3 + 0.25).to(
        torch.float8_e4m3fn
    )
    return packed, scales


# ===========================================================================
# The arithmetic
# ===========================================================================


class TestDequantizeIsExact(CustomTestCase):
    def test_matches_an_independent_reference(self):
        packed, scales = _random_nvfp4(GDN_BA_WIDTH, 128)
        global_scale = torch.tensor(3.5)
        produced = dequantize_nvfp4(packed, scales, global_scale, torch.float32)
        expected = _reference_dequantize(packed, scales, global_scale)
        self.assertTrue(torch.equal(produced, expected))

    def test_bf16_output_matches_the_reference_cast(self):
        packed, scales = _random_nvfp4(64, 64, seed=7)
        global_scale = torch.tensor(11.25)
        produced = dequantize_nvfp4(packed, scales, global_scale, torch.bfloat16)
        expected = _reference_dequantize(packed, scales, global_scale).to(
            torch.bfloat16
        )
        self.assertEqual(produced.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(produced, expected))

    def test_the_global_scale_direction_is_divide(self):
        """compressed-tensors stores the QUANTISE direction, so recovery divides.

        Pinned on a single hand-computable element rather than a round trip:
        getting this backwards does not crash, it scales the layer by
        ``global_scale ** 2``, which is exactly the failure mode the Marlin
        lane's reciprocal comment warns about.
        """
        packed = torch.zeros(1, NVFP4_BLOCK_SIZE // 2, dtype=torch.uint8)
        packed[0, 0] = 0x02  # nibbles 2, 0
        scales = torch.tensor([[2.0]]).to(torch.float8_e4m3fn)
        dense = dequantize_nvfp4(packed, scales, torch.tensor(4.0), torch.float32)
        # low nibble 2 -> E2M1 1.0; 1.0 * 2.0 / 4.0
        self.assertEqual(dense[0, 0].item(), 0.5)
        self.assertEqual(dense[0, 1].item(), 0.0)

    def test_the_low_nibble_is_the_first_element(self):
        packed = torch.zeros(1, NVFP4_BLOCK_SIZE // 2, dtype=torch.uint8)
        packed[0, 0] = 0x70  # low nibble 0, high nibble 7
        scales = torch.tensor([[1.0]]).to(torch.float8_e4m3fn)
        dense = dequantize_nvfp4(packed, scales, torch.tensor(1.0), torch.float32)
        self.assertEqual(dense[0, 0].item(), 0.0)
        self.assertEqual(dense[0, 1].item(), 6.0)

    def test_the_sign_bit_is_the_high_nibble_bit(self):
        packed = torch.zeros(1, NVFP4_BLOCK_SIZE // 2, dtype=torch.uint8)
        packed[0, 0] = 0x0F
        scales = torch.tensor([[1.0]]).to(torch.float8_e4m3fn)
        dense = dequantize_nvfp4(packed, scales, torch.tensor(1.0), torch.float32)
        self.assertEqual(dense[0, 0].item(), -6.0)

    def test_a_mismatched_scale_shape_is_named(self):
        packed, _ = _random_nvfp4(32, 64)
        with self.assertRaisesRegex(ValueError, "block scales of shape"):
            dequantize_nvfp4(
                packed,
                torch.zeros(32, 3, dtype=torch.float8_e4m3fn),
                torch.tensor(1.0),
                torch.bfloat16,
            )

    def test_a_non_packed_weight_is_named(self):
        with self.assertRaisesRegex(ValueError, "uint8"):
            dequantize_nvfp4(
                torch.zeros(8, 8),
                torch.zeros(8, 1, dtype=torch.float8_e4m3fn),
                torch.tensor(1.0),
                torch.bfloat16,
            )


# ===========================================================================
# The verdict: unsharded geometry decides WHICH layers
# ===========================================================================


class TestTheGeometryVerdict(CustomTestCase):
    def test_the_gdn_gate_has_no_marlin_form(self):
        reason = nvfp4_marlin_unpackable_reason(_fake_linear(GDN_BA_WIDTH))
        self.assertIsNotNone(reason)
        self.assertIn(str(GDN_BA_WIDTH), reason)
        self.assertIn(str(GPTQ_MARLIN_MIN_THREAD_N), reason)

    def test_a_tile_multiple_has_one(self):
        for width in (64, 128, 5120, 17408):
            self.assertIsNone(nvfp4_marlin_unpackable_reason(_fake_linear(width)))

    def test_the_verdict_reads_the_unsharded_width_not_the_shard(self):
        """#316's rule, restated here: a bad SHARD is not a bad MODULE.

        Every per-rank shard of the 96-wide gate misses the tile, and so does
        every per-rank shard of a 128-wide module under the same 3-rank plan --
        but only the first is a property of the checkpoint. The verdict must
        therefore never consult ``output_size_per_partition``.
        """
        layer = _fake_linear(128)
        layer.output_size_per_partition = 42  # a tile-illegal shard of a legal module
        self.assertIsNone(nvfp4_marlin_unpackable_reason(layer))
        for shard in UNEVEN_BA_SHARDS:
            self.assertNotEqual(shard % GPTQ_MARLIN_MIN_THREAD_N, 0)

    def test_no_shard_plan_can_rescue_the_gate(self):
        """Exhaustive: 96 has no tile-legal split at any TP, including TP=1."""
        for tp in range(1, 9):
            for shard in range(1, GDN_BA_WIDTH + 1):
                if shard * tp == GDN_BA_WIDTH:
                    self.assertNotEqual(shard % GPTQ_MARLIN_MIN_THREAD_N, 0)

    def test_a_non_linear_module_has_no_verdict(self):
        self.assertIsNone(nvfp4_marlin_unpackable_reason(torch.nn.Module()))


# ===========================================================================
# The routing: verdict first, dequant instead of refusal
# ===========================================================================


class TestTheRouting(CustomTestCase):
    def route(self, scheme, layer, name="linear_attn.in_proj_ba"):
        return CompressedTensorsConfig._maybe_dequantize_unpackable(scheme, layer, name)

    def test_the_unpackable_layer_takes_the_dequant_lane(self):
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            routed = self.route(CompressedTensorsW4A4Fp4(), _fake_linear(GDN_BA_WIDTH))
        self.assertIsInstance(routed, CompressedTensorsW4A4Fp4Dequant)

    def test_a_packable_layer_keeps_the_marlin_lane(self):
        scheme = CompressedTensorsW4A4Fp4()
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            routed = self.route(scheme, _fake_linear(17408), "mlp.gate_up_proj")
        self.assertIs(routed, scheme)
        self.assertNotIsInstance(routed, CompressedTensorsW4A4Fp4Dequant)

    def test_the_native_lane_never_dequantises(self):
        """Blackwell's FP4 GEMM has no thread tile -- 96 rows are fine there."""
        for backend in (
            Fp4GemmRunnerBackend.CUTLASS,
            Fp4GemmRunnerBackend.FLASHINFER_TRTLLM,
        ):
            scheme = CompressedTensorsW4A4Fp4()
            with _fp4_backend(backend):
                routed = self.route(scheme, _fake_linear(GDN_BA_WIDTH))
            self.assertIs(routed, scheme)

    def test_other_schemes_are_returned_untouched(self):
        scheme = CompressedTensorsW8A16Fp8(
            strategy="tensor", is_static_input_scheme=False
        )
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            routed = self.route(scheme, _fake_linear(GDN_BA_WIDTH))
        self.assertIs(routed, scheme)

    def test_the_lane_change_is_announced_by_name(self):
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            with self.assertLogs(
                "sglang.srt.layers.quantization.compressed_tensors."
                "compressed_tensors",
                level=logging.WARNING,
            ) as captured:
                self.route(
                    CompressedTensorsW4A4Fp4(),
                    _fake_linear(GDN_BA_WIDTH),
                    "model.language_model.layers.7.linear_attn.in_proj_ba",
                )
        line = "\n".join(captured.output)
        self.assertIn("model.language_model.layers.7.linear_attn.in_proj_ba", line)
        self.assertIn("DEQUANTISED", line)
        self.assertIn(str(GDN_BA_WIDTH), line)

    def test_every_layer_gets_its_own_line(self):
        """Per layer, not warning_once: 48 gates cost 48 lines, by design."""
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            with self.assertLogs(
                "sglang.srt.layers.quantization.compressed_tensors."
                "compressed_tensors",
                level=logging.WARNING,
            ) as captured:
                for layer_id in range(3):
                    self.route(
                        CompressedTensorsW4A4Fp4(),
                        _fake_linear(GDN_BA_WIDTH),
                        f"layers.{layer_id}.linear_attn.in_proj_ba",
                    )
        self.assertEqual(len(captured.output), 3)


# ===========================================================================
# The lane itself
# ===========================================================================


def _build_and_load(scheme, output_partition_sizes, input_size, global_scales):
    """Run ``create_weights`` and fill it as the weight loader would."""
    layer = torch.nn.Module()
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=list(output_partition_sizes),
        input_size_per_partition=input_size,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
        output_size=sum(output_partition_sizes),
    )
    n = sum(output_partition_sizes)
    packed, scales = _random_nvfp4(n, input_size, seed=3)
    layer.weight_packed.data.copy_(packed)
    layer.weight_scale.data.copy_(scales)
    layer.weight_global_scale.data.copy_(torch.tensor(global_scales))
    layer.input_global_scale.data.copy_(
        torch.tensor([1.0] * len(output_partition_sizes))
    )
    return layer, packed, scales


class TestTheDequantLane(CustomTestCase):
    def test_create_weights_accepts_the_unpackable_width(self):
        """The parent raises here; this lane must not, or nothing is fixed."""
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            with self.assertRaisesRegex(ValueError, str(GDN_BA_WIDTH)):
                CompressedTensorsW4A4Fp4().create_weights(
                    layer=torch.nn.Module(),
                    output_partition_sizes=[GDN_BA_SHARD, GDN_BA_SHARD],
                    input_size_per_partition=64,
                    params_dtype=torch.bfloat16,
                    weight_loader=lambda *a, **k: None,
                    output_size=GDN_BA_WIDTH,
                )
            layer, _, _ = _build_and_load(
                CompressedTensorsW4A4Fp4Dequant(),
                [GDN_BA_SHARD, GDN_BA_SHARD],
                64,
                [2.0, 4.0],
            )
        # The packed parameters still exist: unlike #316's GPTQ case the
        # tensors ARE on disk and something has to receive them.
        for name in (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        ):
            self.assertTrue(hasattr(layer, name), name)

    def test_the_shard_plan_error_still_fires_for_a_packable_module(self):
        """A tile-illegal SHARD of a tile-legal module stays a loud error."""
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            with self.assertRaisesRegex(ValueError, "shard plan was not coarsened"):
                CompressedTensorsW4A4Fp4().create_weights(
                    layer=torch.nn.Module(),
                    output_partition_sizes=[42],
                    input_size_per_partition=64,
                    params_dtype=torch.bfloat16,
                    weight_loader=lambda *a, **k: None,
                    output_size=128,
                )

    def test_process_weights_materialises_the_dense_weight(self):
        scheme = CompressedTensorsW4A4Fp4Dequant()
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            layer, packed, scales = _build_and_load(
                scheme, [GDN_BA_SHARD, GDN_BA_SHARD], 64, [2.0, 4.0]
            )
            scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.shape, (GDN_BA_WIDTH, 64))
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        for name in (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        ):
            self.assertFalse(hasattr(layer, name), name)

        # Per LOGICAL shard, each with its OWN global scale -- not the max()
        # collapse the kernel lanes are forced into.
        top = _reference_dequantize(
            packed[:GDN_BA_SHARD], scales[:GDN_BA_SHARD], 2.0
        ).to(torch.bfloat16)
        bottom = _reference_dequantize(
            packed[GDN_BA_SHARD:], scales[GDN_BA_SHARD:], 4.0
        ).to(torch.bfloat16)
        self.assertTrue(torch.equal(layer.weight[:GDN_BA_SHARD], top))
        self.assertTrue(torch.equal(layer.weight[GDN_BA_SHARD:], bottom))

    def test_a_shared_global_scale_is_broadcast(self):
        scheme = CompressedTensorsW4A4Fp4Dequant()
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            layer, packed, scales = _build_and_load(scheme, [GDN_BA_WIDTH], 64, [3.0])
            scheme.process_weights_after_loading(layer)
        expected = _reference_dequantize(packed, scales, 3.0).to(torch.bfloat16)
        self.assertTrue(torch.equal(layer.weight, expected))

    def test_apply_is_a_plain_linear(self):
        scheme = CompressedTensorsW4A4Fp4Dequant()
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            layer, _, _ = _build_and_load(scheme, [GDN_BA_WIDTH], 64, [3.0])
            scheme.process_weights_after_loading(layer)
        x = torch.randn(5, 64, dtype=torch.bfloat16)
        bias = torch.randn(GDN_BA_WIDTH, dtype=torch.bfloat16)
        self.assertTrue(
            torch.equal(
                scheme.apply_weights(layer, x, bias),
                torch.nn.functional.linear(x, layer.weight, bias),
            )
        )

    def test_the_lane_declares_no_capability_floor(self):
        """Dequant + F.linear needs no kernel, so no floor may be checked."""
        self.assertTrue(CompressedTensorsW4A4Fp4().needs_device_kernel())
        self.assertFalse(CompressedTensorsW4A4Fp4Dequant().needs_device_kernel())

    def test_a_scale_count_that_fits_no_shard_layout_is_named(self):
        scheme = CompressedTensorsW4A4Fp4Dequant()
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            layer, _, _ = _build_and_load(
                scheme, [GDN_BA_SHARD, GDN_BA_SHARD], 64, [2.0, 4.0]
            )
        layer.weight_global_scale.data = torch.tensor([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "one weight_global_scale per"):
            scheme.process_weights_after_loading(layer)

    def test_a_shard_of_the_gate_dequantises_too(self):
        """Under an uneven plan each rank holds 42 / 30 / 24 of the 96 rows.

        Dequantisation is row-wise, so every shard is served -- which is what
        makes the lane work at TP=3 and not only at TP=1.
        """
        scheme = CompressedTensorsW4A4Fp4Dequant()
        for shard in UNEVEN_BA_SHARDS:
            with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
                layer, packed, scales = _build_and_load(scheme, [shard], 64, [2.5])
                scheme.process_weights_after_loading(layer)
            self.assertEqual(layer.weight.shape, (shard, 64))
            self.assertTrue(
                torch.equal(
                    layer.weight,
                    _reference_dequantize(packed, scales, 2.5).to(torch.bfloat16),
                )
            )


# ===========================================================================
# #336: the native lane's own geometry, judged on the SHARDED width
# ===========================================================================


def _model_n(groups: int) -> int:
    """The card-derived model: rows on a rank holding ``groups`` head groups."""
    return GDN_ROWS_PER_GROUP * groups


class TestTheCardModel(CustomTestCase):
    """The three measured points, and what they force."""

    def test_the_group_model_hits_all_three_measured_points(self):
        for ratio, (groups, measured_n) in MEASURED_TP0.items():
            self.assertEqual(_model_n(groups), measured_n, ratio)

    def test_the_explicit_ratios_reproduce_their_group_counts(self):
        """``groups = floor(key_heads * r0 / sum r)`` -- 16 groups, not 48 heads."""
        for ratio, (weight, total) in MEASURED_RATIO_WEIGHTS.items():
            expected_groups = (GDN_KEY_HEADS * weight) // total
            self.assertEqual(expected_groups, MEASURED_TP0[ratio][0], ratio)

    def test_the_falsified_prediction_is_recorded(self):
        """The naive proportion said 4,1,1 would pass; the card said 60.

        Kept as a test because the falsification is what produced the model:
        2 * (48 * r0/sum) gives 64 for 4,1,1, which would have been legal.
        """
        weight, total = MEASURED_RATIO_WEIGHTS["4,1,1"]
        naive_n = 2 * (GDN_VALUE_HEADS * weight // total)
        self.assertEqual(naive_n, 64)
        self.assertEqual(naive_n % NVFP4_NATIVE_MIN_N, 0)
        self.assertNotEqual(naive_n, MEASURED_TP0["4,1,1"][1])

    def test_no_tp_split_of_the_gate_satisfies_the_native_lane(self):
        """6g % 32 == 0 needs g % 16 == 0: every group on one rank.

        Exhaustive over every group count a TP > 1 rank can hold.
        """
        legal = [
            groups
            for groups in range(1, GDN_KEY_HEADS + 1)
            if _model_n(groups) % NVFP4_NATIVE_MIN_N == 0
        ]
        self.assertEqual(legal, [GDN_KEY_HEADS])
        self.assertEqual(_model_n(GDN_KEY_HEADS), GDN_BA_WIDTH)

    def test_the_even_split_fails_too(self):
        """The control arm the balance guard prevented would also have died.

        16 groups over 3 ranks is 6/5/5 by the plan's own rounding, and none
        of those widths is a multiple of 32.
        """
        for groups in (6, 5, 5):
            self.assertNotEqual(_model_n(groups) % NVFP4_NATIVE_MIN_N, 0)
        # And for every even TP that divides 16 groups exactly.
        for tp in (2, 4, 8, 16):
            self.assertNotEqual(
                _model_n(GDN_KEY_HEADS // tp) % NVFP4_NATIVE_MIN_N, 0, tp
            )

    def test_tp1_is_the_only_legal_native_configuration(self):
        """Which is exactly why the solo arm was the only one that booted."""
        self.assertEqual(GDN_BA_WIDTH % NVFP4_NATIVE_MIN_N, 0)


class TestTheNativeVerdict(CustomTestCase):
    def test_every_measured_width_is_refused(self):
        for ratio, (_, width) in MEASURED_TP0.items():
            reason = nvfp4_native_unpackable_reason(width, sharded=True)
            self.assertIsNotNone(reason, ratio)
            self.assertIn(str(width), reason)
            self.assertIn(str(NVFP4_NATIVE_MIN_N), reason)
            self.assertIn("sharded", reason)

    def test_a_legal_width_has_no_verdict(self):
        for width in (32, 64, 96, 5120):
            self.assertIsNone(nvfp4_native_unpackable_reason(width, sharded=True))

    def test_the_unsharded_form_names_itself(self):
        reason = nvfp4_native_unpackable_reason(48, sharded=False)
        self.assertIn("unsharded", reason)


class TestTheDispatchOnTheResolvedBackend(CustomTestCase):
    """One judgement, two lane forms, picked by THIS rank's backend."""

    NATIVE = (Fp4GemmRunnerBackend.CUTLASS, Fp4GemmRunnerBackend.FLASHINFER_TRTLLM)

    def test_a_marlin_rank_keeps_the_unsharded_tile_verdict(self):
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            reason = nvfp4_unpackable_reason(_fake_linear(GDN_BA_WIDTH))
        self.assertIn(str(GPTQ_MARLIN_MIN_THREAD_N), reason)
        self.assertIn("unsharded", reason)

    def test_a_marlin_rank_ignores_the_shard_width(self):
        """#332's rule survives: a bad SHARD is a plan bug there, not a lane
        change, so passing the shard must not alter the Marlin verdict."""
        with _fp4_backend(Fp4GemmRunnerBackend.MARLIN):
            for shard in UNEVEN_BA_SHARDS:
                self.assertIsNone(nvfp4_unpackable_reason(_fake_linear(128), shard))

    def test_a_native_rank_judges_the_shard(self):
        for backend in self.NATIVE:
            with _fp4_backend(backend):
                for _, width in MEASURED_TP0.values():
                    reason = nvfp4_unpackable_reason(_fake_linear(GDN_BA_WIDTH), width)
                    self.assertIsNotNone(reason, (backend, width))
                    self.assertIn(str(NVFP4_NATIVE_MIN_N), reason)

    def test_a_native_rank_without_a_shard_width_answers_unsharded(self):
        """At scheme-selection time only the module width exists. 96 passes
        the native granularity, so TP=1 keeps the kernel lane."""
        for backend in self.NATIVE:
            with _fp4_backend(backend):
                self.assertIsNone(nvfp4_unpackable_reason(_fake_linear(GDN_BA_WIDTH)))
                # ...but a module no split can rescue is caught right there.
                self.assertIsNotNone(nvfp4_unpackable_reason(_fake_linear(48)))

    def test_a_non_linear_module_has_no_verdict_on_either_lane(self):
        for backend in (Fp4GemmRunnerBackend.MARLIN, *self.NATIVE):
            with _fp4_backend(backend):
                self.assertIsNone(nvfp4_unpackable_reason(torch.nn.Module()))


class TestTheLateRouting(CustomTestCase):
    """``create_weights`` is the first moment the shard width exists."""

    def _method(self):
        import types

        config = types.SimpleNamespace(
            _maybe_dequantize_unpackable=(
                CompressedTensorsConfig._maybe_dequantize_unpackable
            )
        )
        return CompressedTensorsLinearMethod(config)

    def _run(
        self,
        backend,
        output_partition_sizes,
        scheme=None,
        output_size: int = GDN_BA_WIDTH,
    ):
        layer = _fake_linear(output_size)
        layer.scheme = scheme if scheme is not None else CompressedTensorsW4A4Fp4()
        layer.prefix = "model.layers.3.linear_attn.in_proj_ba"
        with _fp4_backend(backend):
            self._method().create_weights(
                layer=layer,
                input_size_per_partition=64,
                output_partition_sizes=list(output_partition_sizes),
                input_size=HIDDEN,
                output_size=output_size,
                params_dtype=torch.bfloat16,
                weight_loader=lambda *a, **k: None,
            )
        return layer

    def test_the_native_rank_swaps_to_the_dequant_lane(self):
        """The third stopper, closed: TP0 no longer reaches the FP4 GEMM."""
        for _, width in MEASURED_TP0.values():
            layer = self._run(Fp4GemmRunnerBackend.CUTLASS, [width])
            self.assertIsInstance(layer.scheme, CompressedTensorsW4A4Fp4Dequant)
            self.assertEqual(layer.weight_packed.shape[0], width)

    def test_a_native_shard_on_the_granularity_keeps_the_kernel_lane(self):
        scheme = CompressedTensorsW4A4Fp4()
        layer = self._run(
            Fp4GemmRunnerBackend.CUTLASS, [64], scheme=scheme, output_size=128
        )
        self.assertIs(layer.scheme, scheme)

    def test_the_marlin_rank_is_unchanged(self):
        """The sm86 side keeps #332's behaviour exactly.

        Three parts: a layer already routed at scheme-selection time is left
        alone, the 96-wide gate is still judged against the 64-tile (not the
        native 32, which it would pass), and a tile-illegal shard of a
        tile-LEGAL module still dies loudly instead of quietly dequantising.
        """
        already = CompressedTensorsW4A4Fp4Dequant()
        layer = self._run(
            Fp4GemmRunnerBackend.MARLIN, UNEVEN_BA_SHARDS[:1], scheme=already
        )
        self.assertIs(layer.scheme, already)

        # The 96er pin: 96 % 32 == 0 but 96 % 64 != 0, so the Marlin rank
        # dequantises where a native rank at TP=1 would not.
        self.assertEqual(GDN_BA_WIDTH % NVFP4_NATIVE_MIN_N, 0)
        routed = self._run(Fp4GemmRunnerBackend.MARLIN, [GDN_BA_WIDTH])
        self.assertIsInstance(routed.scheme, CompressedTensorsW4A4Fp4Dequant)

        with self.assertRaisesRegex(ValueError, "shard plan was not coarsened"):
            self._run(Fp4GemmRunnerBackend.MARLIN, [42], output_size=128)

    def test_the_swap_is_announced_with_the_layer_name(self):
        with self.assertLogs(
            "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors",
            level=logging.WARNING,
        ) as captured:
            self._run(Fp4GemmRunnerBackend.CUTLASS, [42])
        line = "\n".join(captured.output)
        self.assertIn("model.layers.3.linear_attn.in_proj_ba", line)
        self.assertIn("DEQUANTISED", line)
        self.assertIn("42", line)

    def test_the_bypass_guard_names_itself(self):
        """If the routing is ever skipped, the scheme must say so instead of
        letting the width reach the kernel assert."""
        with _fp4_backend(Fp4GemmRunnerBackend.CUTLASS):
            with self.assertRaisesRegex(ValueError, "routing was bypassed"):
                CompressedTensorsW4A4Fp4().create_weights(
                    layer=torch.nn.Module(),
                    output_partition_sizes=[42],
                    input_size_per_partition=64,
                    params_dtype=torch.bfloat16,
                    weight_loader=lambda *a, **k: None,
                    output_size=GDN_BA_WIDTH,
                )

    def test_the_dequantised_shard_serves_exactly(self):
        """A native rank's odd shard materialises byte-exactly as well.

        With this in place every one of the 48 GDN layers logs a DEQUANTISED
        line on EVERY rank of a TP=3 boot, native rank included -- the beleg's
        96 (two Marlin ranks) becomes 144.
        """
        scheme = CompressedTensorsW4A4Fp4Dequant()
        with _fp4_backend(Fp4GemmRunnerBackend.CUTLASS):
            layer, packed, scales = _build_and_load(scheme, [42], 64, [2.5])
            scheme.process_weights_after_loading(layer)
        self.assertTrue(
            torch.equal(
                layer.weight,
                _reference_dequantize(packed, scales, 2.5).to(torch.bfloat16),
            )
        )


if __name__ == "__main__":
    unittest.main()
