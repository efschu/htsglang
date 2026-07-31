"""Arithmetic and chain-graph guards for the Class-3 video-enhance tenant.

The numbers asserted here are the ones DESIGN #333 §8.3 publishes. They are
pinned because the whole M2 feasibility gate is "compute the reservation
before booking a GPU window": if the arithmetic drifts, the gate silently
stops gating.
"""

import unittest

from sglang.srt.video_enhance.chain import (
    CANONICAL_ORDER,
    ChainError,
    ChainRequest,
    StageKind,
    build_chain,
    validate_chain,
)
from sglang.srt.video_enhance.frame_math import (
    GIB,
    MIB,
    R4K,
    R8K,
    R540P,
    R720P,
    R1080P,
    PixelFormat,
    Resolution,
    UnprobedFootprintError,
    chain_reservation,
    frame_bytes,
    max_in_flight_for_budget,
    rife_footprint,
    sr_footprint,
    sr_reserved_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFrameBytes(CustomTestCase):
    def test_matches_published_table(self):
        # DESIGN #333 §8.3, "Frame byte sizes, exact".
        table = {
            (R540P, PixelFormat.NV12): 0.74,
            (R540P, PixelFormat.RGB_FP16): 2.97,
            (R540P, PixelFormat.RGB_FP32): 5.93,
            (R720P, PixelFormat.NV12): 1.32,
            (R720P, PixelFormat.RGB_FP16): 5.27,
            (R720P, PixelFormat.RGB_FP32): 10.55,
            (R1080P, PixelFormat.NV12): 2.97,
            (R1080P, PixelFormat.RGB_FP16): 11.87,
            (R1080P, PixelFormat.RGB_FP32): 23.73,
            (R4K, PixelFormat.NV12): 11.87,
            (R4K, PixelFormat.RGB_FP16): 47.46,
            (R4K, PixelFormat.RGB_FP32): 94.92,
            (R8K, PixelFormat.NV12): 47.46,
            (R8K, PixelFormat.RGB_FP16): 189.84,
            (R8K, PixelFormat.RGB_FP32): 379.69,
        }
        for (res, fmt), expected_mib in table.items():
            with self.subTest(res=str(res), fmt=fmt.value):
                self.assertAlmostEqual(
                    frame_bytes(res, fmt) / MIB, expected_mib, places=2
                )

    def test_nv12_rejects_odd_dimensions(self):
        with self.assertRaises(ValueError):
            frame_bytes(Resolution(1921, 1080), PixelFormat.NV12)


class TestSrFootprint(CustomTestCase):
    def test_activation_table(self):
        # §8.3 "one activation, fp16" / "two live" / "x4 output, fp16".
        expected = {
            R540P: (63.28, 126.56, 47.46),
            R720P: (112.50, 225.00, 84.38),
            R1080P: (253.12, 506.25, 189.84),
        }
        for res, (one, two, out) in expected.items():
            fp = sr_footprint(res, "fp16")
            with self.subTest(res=str(res)):
                self.assertAlmostEqual(fp.posts["activations"] / 2 / MIB, one, places=1)
                self.assertAlmostEqual(fp.posts["activations"] / MIB, two, places=1)
                self.assertAlmostEqual(fp.posts["output_frame"] / MIB, out, places=1)

    def test_per_stream_subtotals(self):
        # §8.3 "per-stream subtotal": ~178 MiB at 540p, ~708 MiB at 1080p.
        self.assertAlmostEqual(sr_footprint(R540P).tensor_bytes / MIB, 177.0, delta=2.0)
        self.assertAlmostEqual(
            sr_footprint(R1080P).tensor_bytes / MIB, 708.0, delta=2.0
        )

    def test_reserved_reproduces_published_budgets(self):
        # §8.3: "budget 1.0 GiB per in-flight 1080p frame and 0.25 GiB per
        # in-flight 540p frame". The overhead fraction exists to make these
        # come out of the formula rather than be written next to it.
        self.assertAlmostEqual(sr_reserved_bytes(R1080P) / GIB, 1.0, delta=0.01)
        self.assertAlmostEqual(sr_reserved_bytes(R540P) / GIB, 0.25, delta=0.01)

    def test_fp32_is_twice_fp16(self):
        self.assertEqual(
            sr_footprint(R1080P, "fp32").tensor_bytes,
            2 * sr_footprint(R1080P, "fp16").tensor_bytes,
        )


class TestRifeFootprintIsUnprobed(CustomTestCase):
    def test_refuses_to_invent_a_number(self):
        # §8.3 registers RIFE's footprint as measurement post P4 and asserts no
        # value. The estimator must not quietly supply one.
        with self.assertRaises(UnprobedFootprintError):
            rife_footprint(R4K, "fp16")

    def test_accepts_a_measured_value(self):
        fp = rife_footprint(R4K, "fp16", measured_bytes_per_pair=512 * MIB)
        self.assertEqual(fp.posts["flow_pyramids_and_output"], 512 * MIB)


class TestReservation(CustomTestCase):
    def test_formula_is_itemised(self):
        res = chain_reservation(
            source=R1080P, target=R4K, streams_in_flight=2, with_rife=False
        )
        self.assertIn("tenant_ctx_overhead", res.posts)
        self.assertIn("stage_sr", res.posts)
        self.assertIn("stage_resize", res.posts)
        self.assertIn("nvdec_surface_pool", res.posts)
        self.assertEqual(res.total_bytes, sum(res.posts.values()))

    def test_sr_scales_linearly_with_in_flight_depth(self):
        one = chain_reservation(
            source=R1080P, target=R4K, streams_in_flight=1, with_rife=False
        )
        two = chain_reservation(
            source=R1080P, target=R4K, streams_in_flight=2, with_rife=False
        )
        self.assertEqual(two.posts["stage_sr"], 2 * one.posts["stage_sr"])

    def test_1080p_source_at_depth_two_settles_the_design_claim(self):
        # §8.3: "a 1080p-source chain at two in-flight frames is affordable and
        # a 4K-source chain at the same depth is not" on a 5090 shared with a
        # HOT LLM. Concretely on this rig: 32607 MiB total, a 22 GiB LLM rank
        # budget and the 400 MiB card corridor leave the tenant 9679 MiB.
        budget = (32607 - 22 * 1024 - 400) * MIB
        affordable = chain_reservation(
            source=R1080P, target=R4K, streams_in_flight=2, with_rife=False
        )
        unaffordable = chain_reservation(
            source=R4K, target=R4K, streams_in_flight=2, with_rife=False
        )
        self.assertLess(affordable.total_bytes, budget)
        self.assertGreater(unaffordable.total_bytes, budget)

    def test_max_in_flight_is_derived_not_configured(self):
        budget = 6 * GIB
        depth = max_in_flight_for_budget(
            source=R1080P, target=R4K, budget_bytes=budget, with_rife=False
        )
        self.assertGreaterEqual(depth, 1)
        fits = chain_reservation(
            source=R1080P, target=R4K, streams_in_flight=depth, with_rife=False
        )
        overflows = chain_reservation(
            source=R1080P, target=R4K, streams_in_flight=depth + 1, with_rife=False
        )
        self.assertLessEqual(fits.total_bytes, budget)
        self.assertGreater(overflows.total_bytes, budget)

    def test_tiny_budget_yields_zero_depth(self):
        self.assertEqual(
            max_in_flight_for_budget(
                source=R1080P, target=R4K, budget_bytes=64 * MIB, with_rife=False
            ),
            0,
        )


class TestChainGraph(CustomTestCase):
    def test_default_chain_order(self):
        chain = build_chain(ChainRequest(source=R1080P, target=R4K, fps_multiplier=2))
        self.assertEqual(
            chain.kinds,
            (
                StageKind.DECODE,
                StageKind.COLOR_TO_RGB,
                StageKind.SR,
                StageKind.RESIZE,
                StageKind.RIFE,
                StageKind.COLOR_TO_YUV,
                StageKind.ENCODE,
            ),
        )

    def test_resize_precedes_rife_and_sizes_the_rife_engine(self):
        chain = build_chain(ChainRequest(source=R1080P, target=R4K, fps_multiplier=2))
        self.assertLess(
            chain.kinds.index(StageKind.RESIZE), chain.kinds.index(StageKind.RIFE)
        )
        rife = chain.stage(StageKind.RIFE)
        # The engine is built at the target size, not at the 8K SR output.
        self.assertEqual(rife.options["trt_max_shape"], (3840, 2160))
        self.assertEqual(chain.stage(StageKind.SR).out_res, R8K)

    def test_resize_is_skipped_when_sr_already_lands_on_target(self):
        chain = build_chain(ChainRequest(source=R540P, target=R4K))
        self.assertNotIn(StageKind.RESIZE, chain.kinds)
        self.assertEqual(chain.stage(StageKind.SR).out_res, R4K)

    def test_rife_absent_at_multiplier_one(self):
        chain = build_chain(ChainRequest(source=R1080P, target=R4K, fps_multiplier=1))
        self.assertNotIn(StageKind.RIFE, chain.kinds)

    def test_rife_arity(self):
        chain = build_chain(ChainRequest(source=R1080P, target=R4K, fps_multiplier=3))
        rife = chain.stage(StageKind.RIFE)
        self.assertEqual(rife.arity_in, 2)
        self.assertEqual(rife.arity_out, 2)

    def test_geometry_and_format_are_continuous(self):
        chain = build_chain(ChainRequest(source=R1080P, target=R4K, fps_multiplier=2))
        for left, right in zip(chain.stages, chain.stages[1:]):
            self.assertEqual(left.out_res, right.in_res)
            self.assertEqual(left.out_format, right.in_format)

    def test_upscale_without_sr_is_refused(self):
        with self.assertRaises(ChainError):
            build_chain(ChainRequest(source=R1080P, target=R4K, enable_sr=False))

    def test_downscale_without_sr_is_allowed(self):
        chain = build_chain(ChainRequest(source=R1080P, target=R720P, enable_sr=False))
        self.assertEqual(chain.kinds.index(StageKind.RESIZE), 2)

    def test_invalid_rife_scale_is_refused(self):
        with self.assertRaises(ChainError):
            ChainRequest(source=R1080P, target=R4K, fps_multiplier=2, rife_scale=0.75)

    def test_out_of_order_chain_is_refused(self):
        chain = build_chain(ChainRequest(source=R1080P, target=R4K, fps_multiplier=2))
        stages = list(chain.stages)
        i, j = chain.kinds.index(StageKind.RESIZE), chain.kinds.index(StageKind.RIFE)
        stages[i], stages[j] = stages[j], stages[i]
        broken = type(chain)(request=chain.request, stages=tuple(stages))
        with self.assertRaises(ChainError):
            validate_chain(broken)

    def test_canonical_order_is_the_design_order(self):
        self.assertEqual(
            [k.value for k in CANONICAL_ORDER],
            [
                "decode",
                "color_to_rgb",
                "sr",
                "resize",
                "rife",
                "color_to_yuv",
                "encode",
            ],
        )

    def test_reservation_from_chain_matches_direct_call(self):
        request = ChainRequest(source=R1080P, target=R4K, streams_in_flight=2)
        chain = build_chain(request)
        direct = chain_reservation(
            source=R1080P, target=R4K, streams_in_flight=2, with_rife=False
        )
        self.assertEqual(chain.reservation().total_bytes, direct.total_bytes)


if __name__ == "__main__":
    unittest.main()
