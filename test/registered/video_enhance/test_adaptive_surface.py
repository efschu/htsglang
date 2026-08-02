"""The #451 planner and the #448 admission where they meet the HTTP surface.

Two claims are checked here that neither unit suite can make on its own:

*   **The default path is unchanged.** A request that names neither
    ``chain_policy`` nor ``source_kind`` plans through ``plan_job`` exactly as
    before, keeps the depth-1 response bridge, and gets no adaptive verdict in
    its status. That is asserted rather than assumed, because both features
    add code to ``stream_response`` and the cheap mistake is to make the new
    branch the common one.

*   **An adaptive plan reaches the client.** The mode and the one-line reason
    appear in the job status, and the request body the rest of the pipeline
    runs on is the *resolved* one -- if the planner changes the multiplier and
    the muxer keeps retiming against the caller's, the container declares a
    rate the frames do not have.

Hermetic: fake stages, a fake frame source, ``remux=False`` so no ffmpeg, and
a probe report written into a temporary directory. No card.
"""

import asyncio
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from sglang.srt.video_enhance.chain import StageKind
from sglang.srt.video_enhance.chain_policy import ChainPolicyError
from sglang.srt.video_enhance.frame_math import MIB, PixelFormat, Resolution
from sglang.srt.video_enhance.frames import Frame
from sglang.srt.video_enhance.mux import MediaInfo, TrackInfo
from sglang.srt.video_enhance.ring import OverloadPolicy
from sglang.srt.video_enhance.server import (
    RESPONSE_BRIDGE_DEPTH,
    EnhanceRequestBody,
    VideoEnhanceService,
)
from sglang.srt.video_enhance.streaming import StreamingAdmissionError
from sglang.srt.video_enhance.tenant import TenantConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

RIFE_PAIR_BYTES = int(1185.4 * MIB)

def probe_report(scale=1.0):
    """One card, fast enough that the 4K rife-only chain clears 24 fps.

    Written to a temporary directory so the service reads it the way a
    deployment does, through ``TenantConfig.measurement_dir``.
    """
    return {
        "host": {"card_name": "TEST-CARD", "total_mib": 32607},
        "noise_floor_pct": 2.77,
        "finished_at": 1.0,
        "samples": [
            {
                "post": "P1",
                "stage": stage,
                "card": "TEST-CARD",
                "resolution": resolution,
                "dtype": "fp16",
                "options": {},
                "ms_per_frame": ms * scale,
                "ms_stdev": 0.0,
                "iterations": 10,
            }
            for stage, resolution, ms in (
                ("decode", "3840x2160", 5.0),
                ("rife", "3840x2160", 20.68),
                ("encode", "3840x2160", 3.10),
            )
        ],
    }


def media_info(width=3840, height=2160, rate="24/1"):
    return MediaInfo(
        tracks=(
            TrackInfo(
                index=0,
                codec_type="video",
                codec_name="h264",
                width=width,
                height=height,
                avg_frame_rate=rate,
            ),
            TrackInfo(index=1, codec_type="audio", codec_name="aac"),
        ),
        duration_s=600.0,
        format_name="mov,mp4",
    )


class _FakeDevice:
    type = "cuda"


class _FakeTensor:
    device = _FakeDevice()

    def __init__(self, index):
        self.index = index


class _PassthroughStage:
    def __init__(self, name, resolution):
        self.name = name
        self.resolution = resolution
        self.closed = False

    def process(self, frames):
        return [
            Frame(
                data=frame.data,
                resolution=self.resolution,
                format=frame.format,
                index=frame.index,
            )
            for frame in frames
        ]

    def close(self):
        self.closed = True


class _EncodeStage:
    name = "encode"

    def __init__(self):
        self.closed = False

    def process(self, frames):
        return [b"x" * 32 for _ in frames]

    def flush(self):
        return []

    def close(self):
        self.closed = True


def stages_for(chain):
    out = {}
    for spec in chain.stages:
        out[spec.kind] = (
            _EncodeStage()
            if spec.kind is StageKind.ENCODE
            else _PassthroughStage(spec.kind.value, spec.out_res)
        )
    return out


def source_for(_chain, count=8):
    async def gen():
        for index in range(count):
            yield Frame(
                data=_FakeTensor(index),
                resolution=Resolution(3840, 2160),
                format=PixelFormat.NV12,
                index=index,
            )
            await asyncio.sleep(0)

    return gen()


def body(**kwargs):
    fields = {
        "source_url": "/tmp/none.mp4",
        "source_width": 3840,
        "source_height": 2160,
        "target": "3840x2160",
        "fps_multiplier": 1,
        "streams_in_flight": 2,
    }
    fields.update(kwargs)
    return EnhanceRequestBody(**fields)


def service(measurement_dir=None):
    return VideoEnhanceService(
        TenantConfig(
            budget_mib=30000,
            rife_measured_bytes_per_pair=RIFE_PAIR_BYTES,
            measurement_dir=measurement_dir,
        )
    )


class TheDefaultPathIsUnchangedTest(CustomTestCase):
    def test_a_plain_request_plans_through_plan_job_with_no_verdict(self):
        request = body()
        planned, resolved, decision = service().plan_with_policy(request)
        self.assertIsNone(decision)
        # The very same object, not a rebuilt copy: nothing on the default
        # path may rewrite the caller's request.
        self.assertIs(resolved, request)
        self.assertEqual(planned.chain.request.source, Resolution(3840, 2160))
        self.assertEqual(planned.chain.request.fps_multiplier, 1)

    def test_a_finished_source_keeps_the_depth_one_bridge(self):
        admission = service().admit(body(), Fraction(48))
        self.assertTrue(admission.admitted)
        self.assertFalse(admission.is_streaming)
        # The depth the service will actually use for a non-streaming job.
        self.assertEqual(RESPONSE_BRIDGE_DEPTH, 1)

    def test_the_job_status_carries_the_rate_window_and_nothing_adaptive(self):
        async def scenario():
            svc = service()
            stream = svc.stream_response(
                body(fps_multiplier=2),
                source_factory=source_for,
                stage_factory=stages_for,
                remux=False,
                job_id="plain",
            )
            await stream.__anext__()
            snapshot = svc.progress("plain")
            await stream.aclose()
            return snapshot

        snapshot = asyncio.run(scenario())
        self.assertNotIn("chain_policy", snapshot)
        self.assertNotIn("streaming", snapshot)
        # The rate window is unconditional: it is what #344's live watch reads
        # and it costs one append per chunk on a path already off the chain.
        self.assertIn("sustained_rate", snapshot)
        self.assertIn("fps_out", snapshot["sustained_rate"])


class AdaptivePlanningReachesTheClientTest(CustomTestCase):
    def test_the_chosen_mode_and_reason_are_in_the_job_status(self):
        async def scenario(directory):
            (Path(directory) / "p1.json").write_text(json.dumps(probe_report()))
            svc = service(Path(directory))
            request = body(
                chain_policy="adaptive",
                target_fps="48",
                source_frame_rate="24/1",
                fps_multiplier=2,
            )
            stream = svc.stream_response(
                request,
                source_factory=source_for,
                stage_factory=stages_for,
                media_info=media_info(),
                remux=False,
                job_id="adaptive",
            )
            await stream.__anext__()
            snapshot = svc.progress("adaptive")
            await stream.aclose()
            return snapshot

        with tempfile.TemporaryDirectory() as directory:
            snapshot = asyncio.run(scenario(directory))
        verdict = snapshot["chain_policy"]
        self.assertTrue(verdict["feasible"])
        self.assertTrue(verdict["runnable"])
        self.assertEqual(verdict["mode"], "rife_only")
        self.assertEqual(verdict["provenance"], "measured")
        self.assertIn("chain fps", verdict["reason"])
        self.assertTrue(verdict["considered"])

    def test_the_resolved_body_is_what_the_rest_of_the_pipeline_runs_on(self):
        """The multiplier the muxer retimes against must be the planner's.

        The request asks for x2 and a 48 fps target from a 24 fps source,
        which happens to agree. Asking for a target that implies x3 while the
        body still says x2 is the case that would ship a container declaring a
        rate its frames do not have.
        """
        with tempfile.TemporaryDirectory() as directory:
            # A quarter of the measured cost, so the x3 chain clears 24 fps
            # and the multiplier under test is not confounded by the gate.
            (Path(directory) / "p1.json").write_text(json.dumps(probe_report(0.25)))
            svc = service(Path(directory))
            request = body(
                chain_policy="adaptive",
                target_fps="72",
                source_frame_rate="24/1",
                fps_multiplier=2,
            )
            planned, resolved, decision = svc.plan_with_policy(request, media_info())
            self.assertIsNotNone(decision)
            self.assertEqual(resolved.fps_multiplier, 3)
            self.assertEqual(planned.chain.request.fps_multiplier, 3)
            self.assertNotEqual(resolved.fps_multiplier, request.fps_multiplier)

    def test_an_infeasible_target_is_refused_with_the_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "p1.json").write_text(json.dumps(probe_report()))
            svc = service(Path(directory))
            request = body(
                chain_policy="adaptive",
                target_fps="480",
                source_frame_rate="240/1",
                fps_multiplier=2,
            )
            with self.assertRaises(ChainPolicyError) as caught:
                svc.plan_with_policy(request, media_info(rate="240/1"))
        message = str(caught.exception)
        self.assertIn("no chain reaches", message)
        self.assertIn("chain fps", message)

    def test_a_tenant_with_no_measurement_directory_refuses_rather_than_guesses(self):
        svc = service(None)
        request = body(
            chain_policy="adaptive", target_fps="48", source_frame_rate="24/1"
        )
        with self.assertRaises(ChainPolicyError):
            svc.plan_with_policy(request, media_info())

    def test_adaptive_planning_without_a_rate_says_what_is_missing(self):
        svc = service(None)
        request = body(chain_policy="adaptive")
        with self.assertRaises(ChainPolicyError) as caught:
            svc.plan_with_policy(request)
        self.assertIn("source frame rate", str(caught.exception))


class StreamingAdmissionAtTheSurfaceTest(CustomTestCase):
    def test_a_growing_source_widens_the_bridge_to_the_watermark(self):
        async def scenario():
            svc = service()
            stream = svc.stream_response(
                body(
                    fps_multiplier=2,
                    source_kind="growing",
                    output_watermark_s=0.5,
                    source_frame_rate="24/1",
                ),
                source_factory=source_for,
                stage_factory=stages_for,
                media_info=media_info(),
                remux=False,
                job_id="growing",
            )
            await stream.__anext__()
            snapshot = svc.progress("growing")
            await stream.aclose()
            return snapshot

        snapshot = asyncio.run(scenario())
        streaming = snapshot["streaming"]
        self.assertEqual(streaming["kind"], "growing")
        # remux=False leaves the source rate unknown, so the output rate falls
        # back to the request's own multiplier -- the watermark is still a
        # declared duration converted through whatever rate is known, and the
        # job status reports both halves so the conversion is checkable.
        self.assertGreater(streaming["buffer_depth_frames"], RESPONSE_BRIDGE_DEPTH)
        self.assertAlmostEqual(
            streaming["watermark_s"],
            streaming["buffer_depth_frames"]
            / float(Fraction(streaming["output_frame_rate"])),
            places=3,
        )
        self.assertTrue(streaming["notes"])

    def test_a_live_source_under_stall_is_refused_at_the_surface(self):
        svc = service()
        with self.assertRaises(StreamingAdmissionError) as caught:
            svc.admit(
                body(source_kind="live", overload_policy=OverloadPolicy.STALL.value),
                Fraction(50),
            )
        self.assertIn("drop_frames", str(caught.exception))

    def test_the_same_live_source_with_drop_frames_is_admitted(self):
        """Can-fail proof for the refusal above: one field differs."""
        admission = service().admit(
            body(
                source_kind="live",
                overload_policy=OverloadPolicy.DROP_FRAMES.value,
            ),
            Fraction(50),
        )
        self.assertTrue(admission.admitted)
        self.assertTrue(admission.is_streaming)

    def test_the_sustained_rate_is_sampled_where_the_transport_accepted(self):
        async def scenario():
            svc = service()
            stream = svc.stream_response(
                body(fps_multiplier=2, rate_window_s=60.0),
                source_factory=source_for,
                stage_factory=stages_for,
                remux=False,
                job_id="rates",
            )
            chunks = 0
            async for _chunk in stream:
                chunks += 1
                if chunks >= 4:
                    break
            snapshot = svc.progress("rates")
            await stream.aclose()
            return chunks, snapshot

        chunks, snapshot = asyncio.run(scenario())
        rate = snapshot["sustained_rate"]
        self.assertGreaterEqual(chunks, 4)
        # One sample per accepted chunk plus one for this status call.
        self.assertGreaterEqual(rate["samples"], chunks)
        self.assertGreater(rate["frames_out"], 0)


if __name__ == "__main__":
    unittest.main()
