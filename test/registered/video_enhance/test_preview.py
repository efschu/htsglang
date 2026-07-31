"""Preview taps: the rule is that they cannot cost the pipeline anything.

§8.1 and the user directive. Everything here runs on CPU with fake frames,
because the property under test is structural rather than numerical: a
preview viewer who stops reading must lose preview frames and must not slow
the chain by one frame. That is decidable without a card, and it is the thing
that would be easiest to get wrong and hardest to notice -- a tap that costs
throughput still produces a correct video.

The throughput *cost* of a working tap, which is a different question, is not
decidable here and is measured on cards by
`scripts/video_enhance/preview_tap_bench.py`.
"""

import asyncio
import unittest

from sglang.srt.video_enhance.chain import ChainRequest, StageKind, build_chain
from sglang.srt.video_enhance.frame_math import PixelFormat, Resolution
from sglang.srt.video_enhance.frames import Frame
from sglang.srt.video_enhance.preview import (
    PreviewConfig,
    PreviewError,
    PreviewLanes,
    PreviewStats,
    PreviewTap,
    build_preview_lanes,
    output_tap_stage,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def frame(index: int) -> Frame:
    """A frame whose payload is a plain object.

    The tap must not touch the payload -- it only buffers a reference until
    its own task picks it up -- so a non-tensor stands in fine and keeps this
    file torch-free.
    """
    return Frame(
        data=object(),
        resolution=Resolution(1920, 1080),
        format=PixelFormat.RGB_FP16,
        index=index,
    )


class TapNeverBlocksTest(CustomTestCase):
    """``offer`` is synchronous by contract, and the contract is the design."""

    def test_offer_is_not_a_coroutine(self):
        """If this ever becomes async, the main chain can be suspended by a
        preview viewer, which is the one thing §8.1 forbids."""
        tap = PreviewTap("t")
        self.assertFalse(asyncio.iscoroutinefunction(tap.offer))

    def test_offer_returns_without_an_event_loop_at_all(self):
        """The strongest available statement that it cannot await: it works
        with no running loop, where any await would raise."""
        tap = PreviewTap("t")
        for i in range(50):
            tap.offer(frame(i))
        self.assertEqual(tap.stats.offered, 50)

    def test_a_full_buffer_drops_the_oldest_and_keeps_the_newest(self):
        tap = PreviewTap("t", PreviewConfig(ingress_depth=2))
        for i in range(5):
            tap.offer(frame(i))
        self.assertEqual(tap.stats.offered, 5)
        self.assertEqual(tap.stats.dropped_full, 3)
        # What survives is the tail, because a stale preview frame is worth
        # less than a current one.
        self.assertEqual([f.index for f in tap._items], [3, 4])

    def test_offer_never_raises_even_when_closed(self):
        tap = PreviewTap("t")
        tap.close()
        self.assertFalse(tap.offer(frame(0)))

    def test_an_end_of_stream_sentinel_is_not_a_preview_frame(self):
        tap = PreviewTap("t")
        self.assertFalse(tap.offer(Frame.eos(7)))
        self.assertEqual(tap.stats.offered, 0)

    def test_decimation_happens_before_the_buffer(self):
        """A decimated frame must cost one modulo, not a buffer slot."""
        tap = PreviewTap("t", PreviewConfig(fps_divisor=3, ingress_depth=8))
        for i in range(9):
            tap.offer(frame(i))
        self.assertEqual(tap.stats.decimated, 6)
        self.assertEqual([f.index for f in tap._items], [0, 3, 6])


class StalledViewerCostsOnlyPreviewFramesTest(CustomTestCase):
    """The headline property, as an end-to-end statement about the executor.

    A consumer that never drains the preview must not stop the pipeline. The
    falsifier is a pipeline that is *only* allowed to run as many frames as a
    stalled tap permits -- if the tap could apply back-pressure, the frame
    count here would collapse to the ingress depth.
    """

    def _run_chain(self, tap):
        from sglang.srt.video_enhance.pipeline import PipelineExecutor
        from sglang.srt.video_enhance.ring import OverloadPolicy

        # Resize-only: one enhancement so the chain is legal, arity 1
        # everywhere so the encoded count equals the source count and the
        # assertion below is about the tap rather than about interpolation.
        request = ChainRequest(
            source=Resolution(128, 128),
            target=Resolution(64, 64),
            fps_multiplier=1,
            enable_sr=False,
            enable_resize=True,
            streams_in_flight=1,
        )
        chain = build_chain(request)

        class Passthrough:
            name = "passthrough"

            def process(self, frames):
                return tuple(frames)

        class Sink:
            name = "encode"

            def process(self, frames):
                return (b"x" * 8,)

        stages = {}
        for spec in chain.stages:
            if spec.kind is StageKind.ENCODE:
                stages[spec.kind] = Sink()
            else:
                stages[spec.kind] = Passthrough()

        total = 200

        async def source():
            for i in range(total):
                yield Frame(
                    data=_FakeCuda(),
                    resolution=Resolution(128, 128),
                    format=PixelFormat.NV12,
                    index=i,
                )
                await asyncio.sleep(0)

        async def sink(payload: bytes) -> None:
            return None

        executor = PipelineExecutor(
            job_id="preview",
            chain=chain,
            stages=stages,
            source=source(),
            sink=sink,
            ring_depth=2,
            policy=OverloadPolicy.STALL,
            use_cuda_events=False,
            taps={StageKind.COLOR_TO_RGB: [tap]} if tap is not None else None,
        )
        stats = asyncio.run(executor.run())
        return stats, total

    def test_a_tap_nobody_drains_does_not_stop_the_chain(self):
        tap = PreviewTap("input", PreviewConfig(ingress_depth=2))
        stats, total = self._run_chain(tap)
        # Every frame reached the encoder.
        self.assertEqual(stats.frames_encoded, total)
        self.assertEqual(stats.state, "done")
        # And the tap paid for the stall in dropped preview frames.
        self.assertEqual(tap.stats.offered, total)
        self.assertEqual(tap.stats.dropped_full, total - 2)

    def test_the_chain_encodes_the_same_count_with_and_without_a_tap(self):
        """No tap and a stalled tap must be indistinguishable downstream."""
        without, total = self._run_chain(None)
        with_tap, _ = self._run_chain(PreviewTap("input"))
        self.assertEqual(without.frames_encoded, total)
        self.assertEqual(with_tap.frames_encoded, without.frames_encoded)
        self.assertEqual(with_tap.frames_decoded, without.frames_decoded)

    def test_the_lanes_the_builder_produces_actually_receive_frames(self):
        """The integration the unit tests missed, and a real bug lived in it.

        ``build_preview_lanes`` returns ``by_stage`` mapping a stage to
        *lanes*, and the executor calls ``offer`` on whatever it is handed.
        ``PreviewLane`` had no ``offer``, so every frame raised
        ``AttributeError`` -- which the executor swallows on purpose, because
        a broken preview must not fail a job. The job therefore looked
        healthy while the preview delivered nothing, and 861 swallowed
        exceptions went unnoticed until a throughput measurement reported
        ``offered: 0``.

        Every earlier test in this file drove ``PreviewTap`` directly and so
        could not see it. This one drives what the server actually builds.
        """
        from sglang.srt.video_enhance.pipeline import PipelineExecutor
        from sglang.srt.video_enhance.ring import OverloadPolicy

        request = ChainRequest(
            source=Resolution(128, 128),
            target=Resolution(64, 64),
            fps_multiplier=1,
            enable_sr=False,
            enable_resize=True,
            streams_in_flight=1,
        )
        chain = build_chain(request)
        lanes = build_preview_lanes(chain, fps=24)
        self.assertTrue(lanes.by_stage, "the builder produced no attachment points")

        class Passthrough:
            def process(self, frames):
                return tuple(frames)

        class Sink:
            def process(self, frames):
                return (b"x",)

        stages = {
            spec.kind: (Sink() if spec.kind is StageKind.ENCODE else Passthrough())
            for spec in chain.stages
        }
        total = 20

        async def source():
            for i in range(total):
                yield Frame(
                    data=_FakeCuda(),
                    resolution=Resolution(128, 128),
                    format=PixelFormat.NV12,
                    index=i,
                )
                await asyncio.sleep(0)

        async def sink(payload: bytes) -> None:
            return None

        executor = PipelineExecutor(
            job_id="wiring",
            chain=chain,
            stages=stages,
            source=source(),
            sink=sink,
            ring_depth=2,
            policy=OverloadPolicy.STALL,
            use_cuda_events=False,
            taps=dict(lanes.by_stage),
        )
        asyncio.run(executor.run())
        # The lanes were never started, so nothing drains them -- but the
        # frames must still have been *offered*, which is the thing that was
        # broken. Zero offered is the exact signature of the bug.
        self.assertEqual(lanes.input_lane.stats.offered, total)
        self.assertGreater(lanes.input_lane.stats.dropped_full, 0)

    def test_a_tap_that_raises_does_not_fail_the_job(self):
        """A preview is a convenience; the job is the product."""

        class Exploding:
            def offer(self, frame):
                raise RuntimeError("preview encoder is on fire")

        stats, total = self._run_chain(Exploding())
        self.assertEqual(stats.state, "done")
        self.assertEqual(stats.frames_encoded, total)


class _FakeCuda:
    """A payload that satisfies ``require_device`` without importing torch."""

    class _Device:
        type = "cuda"

    device = _Device()


class PreviewGeometryTest(CustomTestCase):
    def test_preview_preserves_aspect_and_stays_even(self):
        config = PreviewConfig(width=480)
        out = config.preview_resolution(Resolution(1920, 1080))
        self.assertEqual(out, Resolution(480, 270))

    def test_odd_dimensions_are_rounded_down_because_nv12_subsamples(self):
        config = PreviewConfig(width=481)
        out = config.preview_resolution(Resolution(1919, 1079))
        self.assertEqual(out.width % 2, 0)
        self.assertEqual(out.height % 2, 0)

    def test_a_source_smaller_than_the_preview_is_not_upscaled(self):
        config = PreviewConfig(width=480)
        out = config.preview_resolution(Resolution(320, 240))
        self.assertEqual(out, Resolution(320, 240))

    def test_a_nonsense_config_is_refused_at_construction(self):
        with self.assertRaises(PreviewError):
            PreviewConfig(width=4)
        with self.assertRaises(PreviewError):
            PreviewConfig(fps_divisor=0)
        with self.assertRaises(PreviewError):
            PreviewConfig(ingress_depth=0)


class TapAttachmentTest(CustomTestCase):
    """Where the taps attach, for chains that differ by request."""

    def _chain(self, **kwargs):
        base = dict(
            source=Resolution(960, 540),
            target=Resolution(1920, 1080),
            fps_multiplier=2,
            streams_in_flight=1,
        )
        base.update(kwargs)
        return build_chain(ChainRequest(**base))

    def test_the_output_tap_is_the_last_rgb_stage(self):
        chain = self._chain(enable_sr=True, sr_scale=4, enable_resize=True)
        self.assertEqual(output_tap_stage(chain), StageKind.RIFE)

    def test_a_rife_only_chain_still_has_both_taps(self):
        chain = self._chain(
            source=Resolution(1920, 1080),
            target=Resolution(1920, 1080),
            enable_sr=False,
            enable_resize=False,
        )
        lanes = build_preview_lanes(chain, fps=24)
        self.assertIsNotNone(lanes.input_lane)
        self.assertIsNotNone(lanes.output_lane)
        self.assertIn(StageKind.COLOR_TO_RGB, lanes.by_stage)
        self.assertIn(StageKind.RIFE, lanes.by_stage)

    def test_the_input_tap_sees_source_geometry_and_the_output_tap_target(self):
        chain = self._chain(enable_sr=True, sr_scale=4, enable_resize=True)
        lanes = build_preview_lanes(chain, fps=24)
        self.assertEqual(lanes.input_lane.source, Resolution(960, 540))
        self.assertEqual(lanes.output_lane.source, Resolution(1920, 1080))

    def test_a_caller_can_ask_for_one_side_only(self):
        chain = self._chain(
            source=Resolution(1920, 1080),
            target=Resolution(1920, 1080),
            enable_sr=False,
            enable_resize=False,
        )
        lanes = build_preview_lanes(chain, fps=24, want_input=False)
        self.assertIsNone(lanes.input_lane)
        self.assertIsNotNone(lanes.output_lane)

    def test_the_preview_rate_accounts_for_decimation(self):
        """A preview decoder told the wrong rate plays at the wrong speed."""
        chain = self._chain(
            source=Resolution(1920, 1080),
            target=Resolution(1920, 1080),
            enable_sr=False,
            enable_resize=False,
        )
        lanes = build_preview_lanes(chain, fps=48, config=PreviewConfig(fps_divisor=4))
        self.assertEqual(lanes.input_lane.fps, 12)


class PreviewStatsTest(CustomTestCase):
    def test_delivered_fraction_is_the_number_a_viewer_experiences(self):
        stats = PreviewStats(name="input", offered=100, encoded=25)
        self.assertAlmostEqual(stats.delivered_fraction, 0.25)

    def test_an_empty_lane_does_not_divide_by_zero(self):
        self.assertEqual(PreviewStats(name="input").delivered_fraction, 0.0)

    def test_the_snapshot_names_drops_separately_from_decimation(self):
        """They are different costs: one is the viewer being slow, the other
        is the operator choosing to spend less on the preview."""
        stats = PreviewStats(
            name="input", offered=90, dropped_full=30, decimated=45, encoded=15
        )
        snap = stats.snapshot()
        self.assertEqual(snap["dropped_full"], 30)
        self.assertEqual(snap["decimated"], 45)


class PreviewLanesLifecycleTest(CustomTestCase):
    def test_closing_lanes_with_no_task_started_is_safe(self):
        chain = build_chain(
            ChainRequest(
                source=Resolution(1920, 1080),
                target=Resolution(1920, 1080),
                fps_multiplier=2,
                enable_sr=False,
                enable_resize=False,
                streams_in_flight=1,
            )
        )
        lanes = build_preview_lanes(chain, fps=24)
        asyncio.run(lanes.close())
        self.assertTrue(lanes.input_lane.tap.closed)

    def test_snapshot_reports_both_lanes(self):
        chain = build_chain(
            ChainRequest(
                source=Resolution(1920, 1080),
                target=Resolution(1920, 1080),
                fps_multiplier=2,
                enable_sr=False,
                enable_resize=False,
                streams_in_flight=1,
            )
        )
        lanes = build_preview_lanes(chain, fps=24)
        snap = lanes.snapshot()
        self.assertEqual(sorted(snap), ["input", "output"])

    def test_empty_lanes_snapshot_to_nothing(self):
        self.assertEqual(PreviewLanes().snapshot(), {})


if __name__ == "__main__":
    unittest.main()
