"""The fused SR tail resize (#457).

Falsifier-first. The claim the whole route rests on is a filter *identity*:
that a stride-2 depthwise convolution with one fixed tap vector computes
exactly what ``resize.lanczos3_resize`` computes at a 2:1 ratio. Everything
else -- the ONNX surgery, the derived artifact, the re-priced chain -- is
worthless if that is false, so it is asserted three ways and each way can
fail:

*   the tap vector against the general tap table, term by term;
*   the convolution against ``lanczos3_resize`` on real tensors;
*   the parity gate against a tail built to be wrong (``nearest``), which
    must be REJECTED. A gate that has only ever passed is an assertion.

The graph surgery itself needs ``onnx``, which is a build-time dependency and
deliberately absent from the serving requirements, so those cases skip when it
is not importable. They are not the load-bearing ones: the torch twin
``apply_tail_torch`` is the same arithmetic and it always runs.

Everything here is CPU. No device, no engine.
"""

import importlib.util
import unittest

import torch

from sglang.srt.video_enhance.frame_math import R4K, R8K, PixelFormat, Resolution
from sglang.srt.video_enhance.fused_tail import (
    FusedTailError,
    apply_tail_torch,
    fused_tail_reference,
    grade_fused_tail,
    plan_fused_tail,
    refuse_unless_halving,
)
from sglang.srt.video_enhance.resize import (
    _taps,
    halving_pad,
    halving_taps,
    is_exact_halving,
    lanczos3_resize,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_HAS_ONNX = importlib.util.find_spec("onnx") is not None


def _sample(width: int, height: int, seed: int = 0x5EED):
    """CPU-sampled input. Never a device RNG -- two arches disagree on randn."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, height, width), generator=generator, dtype=torch.float32)


class HalvingTapsTest(CustomTestCase):
    """The collapse of the tap table at ratio 2:1, which is what makes fusion possible."""

    def test_every_row_of_the_general_table_is_the_same_vector(self):
        closed_form = halving_taps()
        for size in (64, 512, 7680):
            indices, weights = _taps(size, size // 2)
            self.assertEqual(len(closed_form), len(weights[0]))
            for row in weights:
                for got, want in zip(row, closed_form):
                    self.assertAlmostEqual(got, want, places=12)
            # ... and the source offsets advance by exactly the stride, which
            # is the other half of "this is a convolution". Only away from the
            # borders: there the table clamps, and edge replication is what
            # the graph's Pad node reproduces.
            pad_begin, _ = halving_pad()
            for out_index in (pad_begin, size // 4, size // 2 - pad_begin):
                first = indices[out_index][0]
                self.assertEqual(first, 2 * out_index - pad_begin)

    def test_the_collapse_is_a_property_of_the_ratio_not_of_the_builder(self):
        # It holds for any exact integer decimation and for nothing else. A
        # test that only ever looked at 2:1 could not tell the difference
        # between "this ratio collapses" and "the builder emits one row".
        def row_spread(source: int, target: int) -> float:
            _indices, weights = _taps(source, target)
            middle = weights[target // 2]
            return max(
                abs(got - want) for row in weights for got, want in zip(row, middle)
            )

        for source, target in ((600, 200), (600, 150)):  # 1/3 and 1/4
            self.assertLess(row_spread(source, target), 1e-12)
        for source, target in ((600, 400), (600, 250)):  # 2/3 and 5/12
            self.assertGreater(row_spread(source, target), 0.1)

    def test_the_taps_are_normalised(self):
        self.assertAlmostEqual(sum(halving_taps()), 1.0, places=12)

    def test_the_pad_matches_the_window(self):
        pad_begin, pad_end = halving_pad()
        self.assertEqual(pad_begin + pad_end + 1, len(halving_taps()))


class GeometryRefusalTest(CustomTestCase):
    def test_the_chain_geometry_is_an_exact_halving(self):
        self.assertTrue(is_exact_halving(R8K, R4K))

    def test_a_non_halving_target_is_refused_by_name(self):
        with self.assertRaises(FusedTailError) as caught:
            refuse_unless_halving(R8K, Resolution(2560, 1440))
        self.assertIn("exactly 2:1", str(caught.exception))
        self.assertIn("period p", str(caught.exception))

    def test_an_odd_scale_model_has_no_halving_tail(self):
        with self.assertRaises(FusedTailError) as caught:
            plan_fused_tail(model_scale=3)
        self.assertIn("net scale", str(caught.exception))

    def test_an_unknown_tail_kind_is_refused(self):
        with self.assertRaises(FusedTailError):
            plan_fused_tail(kind="lanczos4")

    def test_the_plan_reports_the_net_scale(self):
        self.assertEqual(plan_fused_tail(model_scale=4).net_scale, 2)


class ConvolutionIdentityTest(CustomTestCase):
    """The load-bearing claim, on real tensors, with no ONNX in sight."""

    def test_the_conv_tail_reproduces_lanczos3_resize(self):
        plan = plan_fused_tail()
        for width, height in ((64, 48), (128, 96), (256, 160)):
            with self.subTest(size=(width, height)):
                x = _sample(width, height)
                target = Resolution(width // 2, height // 2)
                fused = apply_tail_torch(x, plan)
                reference = lanczos3_resize(x, target)
                self.assertEqual(tuple(fused.shape), tuple(reference.shape))
                self.assertLess(torch.max(torch.abs(fused - reference)).item(), 1e-5)

    def test_the_output_is_exactly_half_on_both_axes(self):
        plan = plan_fused_tail()
        out = apply_tail_torch(_sample(96, 64), plan)
        self.assertEqual(tuple(out.shape), (1, 3, 32, 48))

    def test_the_nearest_tail_is_a_different_picture(self):
        # Same machinery, wrong taps. If this passed, the identity test above
        # would be measuring the harness rather than the filter.
        x = _sample(64, 48)
        lanczos = apply_tail_torch(x, plan_fused_tail())
        nearest = apply_tail_torch(x, plan_fused_tail(kind="nearest"))
        self.assertGreater(torch.max(torch.abs(lanczos - nearest)).item(), 0.01)

    def test_the_torch_twin_refuses_the_arm_it_does_not_implement(self):
        with self.assertRaises(FusedTailError):
            apply_tail_torch(_sample(32, 32), plan_fused_tail(kind="bicubic_antialias"))


class ParityGateTest(CustomTestCase):
    """The gate, both directions."""

    def test_the_reference_is_the_existing_two_stage_path(self):
        sr_output = _sample(64, 48)
        target = Resolution(32, 24)
        self.assertTrue(
            torch.equal(
                fused_tail_reference(sr_output, target),
                lanczos3_resize(sr_output, target),
            )
        )

    def test_the_gate_passes_the_conv_tail_by_a_wide_margin(self):
        sr_output = _sample(64, 48)
        target = Resolution(32, 24)
        result = grade_fused_tail(
            apply_tail_torch(sr_output, plan_fused_tail()),
            sr_output,
            target=target,
            note="conv tail",
        )
        self.assertTrue(result.passed)
        # The fp16 engine clears the same 40 dB bar at 48.1 dB. This is a
        # filter identity, so it should clear it by far more than that; a
        # regression that made it merely "good" would show up here.
        self.assertGreater(result.psnr_db, 90.0)

    def test_the_gate_rejects_the_nearest_tail(self):
        sr_output = _sample(64, 48)
        target = Resolution(32, 24)
        result = grade_fused_tail(
            apply_tail_torch(sr_output, plan_fused_tail(kind="nearest")),
            sr_output,
            target=target,
            note="nearest tail",
        )
        self.assertFalse(result.passed)
        self.assertLess(result.psnr_db, 40.0)

    def test_the_reference_refuses_a_geometry_it_cannot_halve(self):
        with self.assertRaises(FusedTailError):
            fused_tail_reference(_sample(64, 48), Resolution(20, 15))


@unittest.skipUnless(_HAS_ONNX, "onnx is a build-time dependency, not a serving one")
class GraphSurgeryTest(CustomTestCase):
    """The ONNX form, when the build-time dependency is present.

    Executed at a desk against the real pinned artifact on 2026-08-03; the
    numbers are in ``TASK_333_M2_VIDEO_ENHANCE.md`` §17.7. These cases rebuild
    the same surgery on a tiny synthetic graph so they need no model file.
    """

    def _identity_graph(self):
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper

        weight = numpy_helper.from_array(
            np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1), name="w"
        )
        node = helper.make_node("Conv", ["input", "w"], ["output"], kernel_shape=[1, 1])
        graph = helper.make_graph(
            [node],
            "identity",
            [
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, [1, 3, "h", "w_"]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [1, 3, "h", "w_"]
                )
            ],
            [weight],
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=8
        )

    def test_the_tail_is_appended_without_renaming_the_output(self):
        import onnx

        from sglang.srt.video_enhance.fused_tail import append_halving_tail

        model, added = append_halving_tail(self._identity_graph(), plan_fused_tail())
        onnx.checker.check_model(model)
        self.assertEqual(added, 4)
        self.assertEqual(model.graph.output[0].name, "output")
        self.assertEqual([imp.version for imp in model.opset_import], [16])

    def test_the_graph_computes_what_the_torch_twin_computes(self):
        import onnxruntime as ort

        from sglang.srt.video_enhance.fused_tail import append_halving_tail

        plan = plan_fused_tail()
        model, _added = append_halving_tail(self._identity_graph(), plan)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        x = _sample(64, 48)
        got = torch.from_numpy(session.run(None, {"input": x.numpy()})[0])
        self.assertEqual(tuple(got.shape), (1, 3, 24, 32))
        self.assertLess(
            torch.max(torch.abs(got - apply_tail_torch(x, plan))).item(), 1e-5
        )

    def test_the_antialias_arm_raises_the_opset_and_the_conv_arm_does_not(self):
        from sglang.srt.video_enhance.fused_tail import append_halving_tail

        conv, _ = append_halving_tail(self._identity_graph(), plan_fused_tail())
        bicubic, _ = append_halving_tail(
            self._identity_graph(), plan_fused_tail(kind="bicubic_antialias")
        )
        self.assertEqual([imp.version for imp in conv.opset_import], [16])
        self.assertEqual([imp.version for imp in bicubic.opset_import], [18])

    def test_a_multi_output_graph_is_refused(self):
        from sglang.srt.video_enhance.fused_tail import append_halving_tail

        model = self._identity_graph()
        model.graph.output.append(model.graph.output[0])
        with self.assertRaises(FusedTailError):
            append_halving_tail(model, plan_fused_tail())


class PayloadTest(CustomTestCase):
    """What the fusion is actually for, in bytes."""

    def test_the_engine_stops_emitting_the_8k_frame(self):
        from sglang.srt.video_enhance.frame_math import MIB, frame_bytes

        unfused = frame_bytes(R8K, PixelFormat.RGB_FP16) / MIB
        fused = frame_bytes(R4K, PixelFormat.RGB_FP16) / MIB
        self.assertAlmostEqual(unfused, 189.84, places=2)
        self.assertAlmostEqual(fused, 47.46, places=2)
        self.assertAlmostEqual(unfused / fused, 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
