"""The #338 additions to the enhance HTTP surface.

Three things the browser extension needs from the server and one it inherits:

*   a time range, so a client that wants forty seconds of a two-hour film does
    not pay for the prefix,
*   a job id it can choose, because a URL handed to a ``<video>`` element
    never surfaces a response header and DELETE is unreachable without one,
*   a capability probe that keeps measured rates and budget arithmetic apart,
*   and the #344b consumer watchdog, which already covers this endpoint --
    asserted here rather than assumed, because "the plugin relies on it" is
    only true if it is actually wired.

Everything is hermetic: no card, no ffmpeg, no socket. The range arithmetic is
pure, the capability endpoint reads probe JSON from a temporary directory, and
the watchdog assertions read the job registry the service keeps.
"""

import asyncio
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from sglang.srt.liveness import EndpointClass, LivenessConfig
from sglang.srt.video_enhance.chain import StageKind
from sglang.srt.video_enhance.frame_math import PixelFormat, Resolution
from sglang.srt.video_enhance.frames import Frame, StageBase
from sglang.srt.video_enhance.mux import MediaInfo, TrackInfo, build_remux_command
from sglang.srt.video_enhance.probes import load_frontier, load_probe_reports
from sglang.srt.video_enhance.server import (
    CHAIN_PRESETS,
    WHOLE_SOURCE,
    EnhanceRequestBody,
    JobIdError,
    RangeError,
    TimeRange,
    VideoEnhanceService,
    normalize_job_id,
    resolve_time_range,
)
from sglang.srt.video_enhance.tenant import TenantConfig, build_stages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NTSC = Fraction(24000, 1001)


class TimeRangeArithmeticTest(CustomTestCase):
    """Seconds in, frame indices out, with the refusals spelled out."""

    def test_no_range_is_the_whole_source(self):
        got = resolve_time_range(start_s=0.0, duration_s=None, rate=None)
        self.assertIs(got, WHOLE_SOURCE)
        self.assertTrue(got.is_whole_source)

    def test_start_and_duration_become_frame_indices(self):
        got = resolve_time_range(start_s=10.0, duration_s=5.0, rate=Fraction(24, 1))
        self.assertEqual(got.start_frame, 240)
        self.assertEqual(got.frame_limit, 120)
        self.assertFalse(got.is_whole_source)

    def test_a_fractional_rate_is_not_rounded_through_float(self):
        """23.976 fps is 24000/1001, and a float product lands off by one.

        ``10 * (24000/1001)`` is 239.760..., so frame 239 is the first frame at
        or after ten seconds. The Fraction path says 239 exactly; a float
        product that happened to land on 240.0 would start the range a frame
        late, which is the one error a viewer sees at a seam.
        """
        got = resolve_time_range(start_s=10.0, duration_s=1.0, rate=NTSC)
        self.assertEqual(got.start_frame, 239)
        self.assertEqual(got.frame_limit, 23)
        self.assertEqual(got.frame_rate, "24000/1001")

    def test_negative_start_is_refused(self):
        with self.assertRaises(RangeError) as caught:
            resolve_time_range(start_s=-1.0, duration_s=None, rate=NTSC)
        self.assertIn("negative", str(caught.exception))

    def test_non_positive_duration_is_refused(self):
        for duration in (0.0, -3.0):
            with self.subTest(duration=duration):
                with self.assertRaises(RangeError):
                    resolve_time_range(start_s=0.0, duration_s=duration, rate=NTSC)

    def test_a_range_without_a_rate_is_refused_and_says_why(self):
        with self.assertRaises(RangeError) as caught:
            resolve_time_range(start_s=5.0, duration_s=None, rate=None)
        self.assertIn("frame rate", str(caught.exception))

    def test_starting_past_the_end_is_refused(self):
        with self.assertRaises(RangeError) as caught:
            resolve_time_range(
                start_s=90.0, duration_s=None, rate=NTSC, source_duration_s=60.0
            )
        self.assertIn("past the source duration", str(caught.exception))

    def test_a_duration_shorter_than_one_frame_is_refused(self):
        """An empty body is indistinguishable from a broken server."""
        with self.assertRaises(RangeError) as caught:
            resolve_time_range(start_s=0.0, duration_s=0.01, rate=Fraction(24, 1))
        self.assertIn("shorter than one frame", str(caught.exception))

    def test_a_range_past_the_end_is_allowed(self):
        """The source simply ends, exactly as it does without a range."""
        got = resolve_time_range(
            start_s=50.0,
            duration_s=3600.0,
            rate=Fraction(24, 1),
            source_duration_s=60.0,
        )
        self.assertEqual(got.start_frame, 1200)
        self.assertEqual(got.frame_limit, 86400)

    def test_describe_carries_every_resolved_field(self):
        described = resolve_time_range(
            start_s=1.0, duration_s=2.0, rate=Fraction(30, 1)
        ).describe()
        self.assertEqual(
            described,
            {
                "start_s": 1.0,
                "duration_s": 2.0,
                "start_frame": 30,
                "frame_limit": 60,
                "frame_rate": "30",
            },
        )


class RangeReachesTheDecodeStageTest(CustomTestCase):
    """A resolved range is only useful if it lands on the two decode knobs."""

    def _chain(self):
        from sglang.srt.video_enhance.chain import ChainRequest, build_chain

        # A downscaling resize-only chain: it has the decode stage the range
        # lands on and needs neither a TensorRT engine nor RIFE weights to
        # instantiate. Resize alone cannot upscale -- the chain builder
        # refuses that -- so the target is below the source.
        return build_chain(
            ChainRequest(
                source=Resolution(1920, 1080),
                target=Resolution(1280, 720),
                fps_multiplier=1,
                enable_sr=False,
                enable_resize=True,
            )
        )

    def test_build_stages_forwards_the_range_to_the_decoder(self):
        config = TenantConfig(budget_mib=8192, rife_measured_bytes_per_pair=1)
        stages = build_stages(config, self._chain(), start_frame=239, frame_limit=48)
        decode = stages[StageKind.DECODE]
        self.assertEqual(decode.start_frame, 239)
        self.assertEqual(decode.frame_limit, 48)

    def test_the_default_call_still_decodes_from_zero(self):
        config = TenantConfig(budget_mib=8192, rife_measured_bytes_per_pair=1)
        decode = build_stages(config, self._chain())[StageKind.DECODE]
        self.assertEqual(decode.start_frame, 0)
        self.assertIsNone(decode.frame_limit)


class RemuxRangeTest(CustomTestCase):
    """The passthrough tracks have to start where the enhanced video does."""

    def _info(self):
        return MediaInfo(
            tracks=(
                TrackInfo(
                    index=0,
                    codec_type="video",
                    codec_name="h264",
                    width=1920,
                    height=1080,
                    avg_frame_rate="24/1",
                ),
                TrackInfo(index=1, codec_type="audio", codec_name="aac"),
            ),
            duration_s=600.0,
            format_name="mov,mp4",
        )

    def _command(self, **kwargs):
        from sglang.srt.video_enhance.mux import TrackSelection

        return build_remux_command(
            source_url="/tmp/a.mp4",
            info=self._info(),
            selection=TrackSelection(),
            enhanced_codec="h264",
            output_rate=Fraction(48, 1),
            **kwargs,
        )

    def test_a_range_seeks_the_source_input_only(self):
        cmd = self._command(source_seek_s=10.0, source_duration_s=5.0)
        # The seek must sit between the elementary stream's -i and the
        # source's -i, or it would apply to the wrong input.
        elementary = cmd.index("pipe:0")
        source = cmd.index("/tmp/a.mp4")
        self.assertLess(elementary, cmd.index("-ss"))
        self.assertLess(cmd.index("-ss"), source)
        self.assertEqual(cmd[cmd.index("-ss") + 1], "10.000000")
        self.assertEqual(cmd[cmd.index("-t") + 1], "5.000000")

    def test_no_range_adds_no_seek_flags(self):
        """The unranged command must be byte-identical to what it always was."""
        cmd = self._command()
        self.assertNotIn("-ss", cmd)
        self.assertNotIn("-t", cmd)

    def test_the_fragmented_mp4_flags_survive_a_range(self):
        """A <video> element cannot play an MP4 whose moov is at the end."""
        cmd = self._command(source_seek_s=3.0)
        flags = cmd[cmd.index("-movflags") + 1]
        for required in ("frag_keyframe", "empty_moov", "delay_moov"):
            self.assertIn(required, flags)


class JobIdTest(CustomTestCase):
    def test_an_absent_id_is_minted(self):
        first, second = normalize_job_id(None), normalize_job_id(None)
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 16)

    def test_a_clean_client_id_is_kept_verbatim(self):
        self.assertEqual(normalize_job_id("ext-abc_123"), "ext-abc_123")

    def test_a_path_traversing_id_is_refused(self):
        for bad in ("../../etc", "a/b", "a b", "a?b", ""):
            with self.subTest(job_id=bad):
                with self.assertRaises(JobIdError):
                    normalize_job_id(bad)

    def test_an_over_long_id_is_refused(self):
        with self.assertRaises(JobIdError):
            normalize_job_id("a" * 65)

    def test_a_live_id_cannot_be_claimed_twice(self):
        service = _service()
        service.jobs["taken"] = _FakeJob(done=False)
        with self.assertRaises(JobIdError) as caught:
            service.claim_job_id("taken")
        self.assertIn("already running", str(caught.exception))

    def test_a_finished_id_is_free_again(self):
        service = _service()
        service.jobs["done"] = _FakeJob(done=True)
        self.assertEqual(service.claim_job_id("done"), "done")


class _FakeTask:
    def __init__(self, done: bool) -> None:
        self._done = done

    def done(self) -> bool:
        return self._done


class _FakeJob:
    def __init__(self, done: bool) -> None:
        self.task = _FakeTask(done)


def _service(**config_kwargs) -> VideoEnhanceService:
    kwargs = {"budget_mib": 16384, "rife_measured_bytes_per_pair": 1242930688}
    kwargs.update(config_kwargs)
    return VideoEnhanceService(TenantConfig(**kwargs))


class CapabilityProbeTest(CustomTestCase):
    """Measured rates and budget arithmetic must not be confused."""

    def _report(self, directory: Path) -> None:
        (directory / "p1.json").write_text(
            json.dumps(
                {
                    "host": {"card_name": "TEST-CARD", "total_mib": 32607},
                    "noise_floor_pct": 1.4,
                    "finished_at": 1.0,
                    "samples": [
                        {
                            "post": "P1",
                            "stage": "rife",
                            "card": "TEST-CARD",
                            "resolution": "3840x2160",
                            "dtype": "fp16",
                            "options": {"scale": 0.5},
                            "ms_per_frame": 10.0,
                            "ms_stdev": 0.1,
                            "iterations": 10,
                        }
                    ],
                }
            )
        )

    def test_without_measurements_the_frontier_says_so(self):
        payload = _service().capabilities()
        self.assertFalse(payload["frontier"]["measured"])
        self.assertIn("no measurement directory", payload["frontier"]["reason"])
        self.assertEqual(payload["frontier"]["rows"], [])

    def test_an_empty_directory_is_not_a_measured_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            frontier = load_frontier(tmp)
            self.assertFalse(frontier["measured"])
            self.assertEqual(frontier["rows"], [])

    def test_a_non_report_json_is_skipped_not_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "e2e.json").write_text(json.dumps({"frames": 480}))
            samples, sources = load_probe_reports(tmp)
            self.assertEqual(samples, [])
            self.assertEqual(sources, [])

    def test_a_measured_frontier_carries_its_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._report(Path(tmp))
            payload = _service(measurement_dir=Path(tmp)).capabilities()
            frontier = payload["frontier"]
            self.assertTrue(frontier["measured"])
            self.assertEqual(len(frontier["rows"]), 1)
            self.assertEqual(frontier["rows"][0]["configuration"], "rife_only")
            self.assertEqual(frontier["rows"][0]["aggregate_max_fps"], 100.0)
            self.assertEqual(frontier["sources"][0]["noise_floor_pct"], 1.4)
            self.assertEqual(frontier["sources"][0]["host"]["card_name"], "TEST-CARD")

    def test_a_rate_question_is_answered_from_the_frontier(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._report(Path(tmp))
            payload = _service(measurement_dir=Path(tmp)).capabilities(
                source=Resolution(3840, 2160),
                target_fps=48.0,
                configuration="rife_only",
            )
            self.assertTrue(payload["answer"]["achievable"])
            self.assertEqual(payload["answer"]["max_fps_at_resolution"], 100.0)

    def test_an_unmeasured_rate_question_is_refused_not_guessed(self):
        payload = _service().capabilities(
            source=Resolution(1920, 1080), target_fps=60.0, configuration="full_chain"
        )
        self.assertFalse(payload["answer"]["achievable"])
        self.assertIn("no measurement exists", payload["answer"]["reason"])

    def test_the_budget_verdict_needs_no_measurement_at_all(self):
        payload = _service().capabilities(
            source=Resolution(1920, 1080), target=Resolution(3840, 2160)
        )
        by_name = {row["preset"]: row for row in payload["budget"]}
        self.assertEqual(sorted(by_name), sorted(CHAIN_PRESETS))
        self.assertTrue(by_name["rife_only"]["fits"])
        # rife_only changes no geometry, so its target is the source size --
        # sending the caller's 4K target with it would be a refusal about the
        # target, reported as if the preset did not fit.
        self.assertEqual(by_name["rife_only"]["target"], "1920x1080")
        self.assertEqual(by_name["full_chain"]["target"], "3840x2160")
        self.assertNotIn("sr", by_name["rife_only"]["stages"])
        self.assertIn("sr", by_name["full_chain"]["stages"])

    def test_a_budget_that_cannot_hold_a_frame_says_what_it_would_need(self):
        payload = _service(budget_mib=64).capabilities(source=Resolution(3840, 2160))
        for row in payload["budget"]:
            self.assertFalse(row["fits"])
            self.assertIn("MiB", row["reason"])

    def test_the_static_description_names_the_presets_and_the_range_support(self):
        payload = _service().capabilities()
        self.assertEqual(sorted(payload["chain_presets"]), sorted(CHAIN_PRESETS))
        self.assertTrue(payload["supports_time_range"])
        self.assertIn("video/mp4", payload["containers"])
        self.assertIn("h264", payload["video_codecs"])

    def test_a_rate_question_without_a_resolution_is_refused(self):
        with self.assertRaises(ValueError):
            _service().capabilities(target_fps=48.0)


class _PassthroughStage(StageBase):
    def __init__(self, name, out_res):
        self.name = name
        self.out_res = out_res
        self.closed = False

    def process(self, frames):
        return list(frames)

    def close(self):
        self.closed = True


class _EncodeStage(StageBase):
    name = "encode"

    def __init__(self):
        self.closed = False

    def process(self, frames):
        return [b"x" * 32 for _ in frames]

    def flush(self):
        return []

    def close(self):
        self.closed = True


class _FakeDevice:
    """``Frame.require_device`` reads ``device.type`` and nothing else."""

    type = "cuda"


class _FakeTensor:
    device = _FakeDevice()

    def __init__(self, index):
        self.index = index


class WatchdogWiringTest(CustomTestCase):
    """#344b covers this endpoint. Asserted, because the plugin relies on it.

    The K3 server is FastAPI on Starlette -- an asyncio ASGI application, not
    a ThreadingHTTPServer -- so the asyncio watchdog applies to it directly
    and needs no thread-side equivalent. What has to be true is that
    ``stream_response`` actually starts one, files it under the video_stream
    class, and declares what the stream holds.
    """

    def _body(self):
        return EnhanceRequestBody(
            source_url="/tmp/none.mp4",
            source_width=1920,
            source_height=1080,
            target="3840x2160",
            fps_multiplier=1,
            streams_in_flight=2,
        )

    def _stages(self, chain):
        stages = {}
        for spec in chain.stages:
            if spec.kind is StageKind.ENCODE:
                stages[spec.kind] = _EncodeStage()
            else:
                stages[spec.kind] = _PassthroughStage(spec.kind.value, spec.out_res)
        return stages

    def _source(self, chain):
        async def gen():
            for index in range(8):
                yield Frame(
                    data=_FakeTensor(index),
                    resolution=Resolution(1920, 1080),
                    format=PixelFormat.NV12,
                    index=index,
                )
                await asyncio.sleep(0)

        return gen()

    def _run(self, coro):
        return asyncio.run(coro)

    def test_the_stream_registers_a_video_stream_watchdog(self):
        async def scenario():
            service = VideoEnhanceService(
                TenantConfig(budget_mib=8192, rife_measured_bytes_per_pair=1),
                liveness=LivenessConfig(poll_interval_s=0.01),
            )
            stream = service.stream_response(
                self._body(),
                source_factory=self._source,
                stage_factory=self._stages,
                remux=False,
                job_id="client-named",
            )
            await stream.__anext__()
            self.assertIn("client-named", service.jobs)
            job = service.jobs["client-named"]
            self.assertIsNotNone(job.watchdog)
            self.assertEqual(
                job.watchdog.policy.timeout_s,
                service.liveness.policy_for(EndpointClass.VIDEO_STREAM).timeout_s,
            )
            # The claims are what put the job's bytes on the reclamation
            # ladder while its consumer is a dead suspect.
            kinds = {claim.kind.value for claim in job.watchdog._claims}
            self.assertIn("pipeline", kinds)
            await stream.aclose()

        self._run(scenario())

    def test_a_client_named_job_is_the_one_delete_reaches(self):
        async def scenario():
            service = VideoEnhanceService(
                TenantConfig(budget_mib=8192, rife_measured_bytes_per_pair=1),
                liveness=LivenessConfig(poll_interval_s=0.01),
            )
            stream = service.stream_response(
                self._body(),
                source_factory=self._source,
                stage_factory=self._stages,
                remux=False,
                job_id="ext-deadbeef",
            )
            await stream.__anext__()
            snapshot = await service.cancel("ext-deadbeef")
            self.assertIn(snapshot["state"], {"cancelled", "done", "running"})
            self.assertTrue(service.jobs["ext-deadbeef"].executor.cancelled)
            await stream.aclose()

        self._run(scenario())

    def test_a_disconnect_tears_down_without_waiting_out_the_timeout(self):
        """The third way out of a stream, and the one the extension uses.

        A tab closing makes Starlette throw into the response generator. Its
        ``finally`` must close the chain's own rings, not only the response
        bridge: the decode and middle stage tasks are suspended in
        ``ring.put`` on rings nobody drains any more, and the cancel flag is
        only read after that await returns. Without that close the teardown
        falls through to the 30 s guard timeout, holding the decoder, the
        encoder and the reservation for half a minute after the socket is
        gone. Asserted as a wall-clock bound because that is the symptom.
        """

        async def scenario():
            import time as clock

            service = VideoEnhanceService(
                TenantConfig(budget_mib=8192, rife_measured_bytes_per_pair=1),
                liveness=LivenessConfig(poll_interval_s=0.01),
            )
            stream = service.stream_response(
                self._body(),
                source_factory=self._source,
                stage_factory=self._stages,
                remux=False,
                job_id="disconnect",
            )
            await stream.__anext__()
            started = clock.monotonic()
            await stream.aclose()
            elapsed = clock.monotonic() - started
            self.assertLess(elapsed, 5.0, f"teardown took {elapsed:.1f} s")
            self.assertTrue(service.jobs["disconnect"].executor.cancelled)

        self._run(scenario())

    def test_cancelling_an_unknown_job_is_a_key_error(self):
        async def scenario():
            with self.assertRaises(KeyError):
                await _service().cancel("never-existed")

        self._run(scenario())

    def test_progress_reports_the_range_only_when_one_was_asked_for(self):
        async def scenario():
            service = VideoEnhanceService(
                TenantConfig(budget_mib=8192, rife_measured_bytes_per_pair=1),
                liveness=LivenessConfig(poll_interval_s=0.01),
            )
            stream = service.stream_response(
                self._body(),
                source_factory=self._source,
                stage_factory=self._stages,
                remux=False,
                job_id="plain",
            )
            await stream.__anext__()
            self.assertNotIn("time_range", service.progress("plain"))
            await stream.aclose()

            ranged = self._body()
            ranged.start_s = 10.0
            ranged.duration_s = 2.0
            ranged.source_frame_rate = "24/1"
            stream = service.stream_response(
                ranged,
                source_factory=self._source,
                stage_factory=self._stages,
                remux=False,
                job_id="ranged",
            )
            await stream.__anext__()
            described = service.progress("ranged")["time_range"]
            self.assertEqual(described["start_frame"], 240)
            self.assertEqual(described["frame_limit"], 48)
            await stream.aclose()

        self._run(scenario())


class RangeResolutionOnTheRequestTest(CustomTestCase):
    def test_a_declared_rate_avoids_the_probe_entirely(self):
        body = EnhanceRequestBody(
            source_url="/does/not/exist.mp4",
            source_width=1920,
            source_height=1080,
            start_s=4.0,
            duration_s=1.0,
            source_frame_rate="25/1",
        )
        resolved = _service().resolve_range(body)
        self.assertEqual(resolved.start_frame, 100)
        self.assertEqual(resolved.frame_limit, 25)

    def test_an_unprobeable_source_refuses_the_range_with_the_path_in_it(self):
        body = EnhanceRequestBody(
            source_url="/does/not/exist.mp4",
            source_width=1920,
            source_height=1080,
            start_s=4.0,
        )
        with self.assertRaises(RangeError) as caught:
            _service().resolve_range(body)
        self.assertIn("/does/not/exist.mp4", str(caught.exception))

    def test_no_range_never_probes_at_all(self):
        """The default path must not gain an ffprobe it did not have."""
        body = EnhanceRequestBody(
            source_url="/does/not/exist.mp4", source_width=1920, source_height=1080
        )
        self.assertIs(_service().resolve_range(body), WHOLE_SOURCE)

    def test_the_range_is_read_off_the_probed_media_info(self):
        info = MediaInfo(
            tracks=(
                TrackInfo(
                    index=0,
                    codec_type="video",
                    codec_name="h264",
                    width=1920,
                    height=1080,
                    avg_frame_rate="30000/1001",
                ),
            ),
            duration_s=120.0,
            format_name="mov,mp4",
        )
        body = EnhanceRequestBody(
            source_url="/tmp/a.mp4",
            source_width=1920,
            source_height=1080,
            start_s=2.0,
            duration_s=1.0,
        )
        resolved = _service().resolve_range(body, info)
        self.assertEqual(resolved.start_frame, 59)
        self.assertEqual(resolved.frame_limit, 29)

    def test_a_start_past_the_probed_duration_is_refused(self):
        info = MediaInfo(
            tracks=(
                TrackInfo(
                    index=0,
                    codec_type="video",
                    codec_name="h264",
                    width=1920,
                    height=1080,
                    avg_frame_rate="24/1",
                ),
            ),
            duration_s=30.0,
            format_name="mov,mp4",
        )
        body = EnhanceRequestBody(
            source_url="/tmp/a.mp4",
            source_width=1920,
            source_height=1080,
            start_s=45.0,
        )
        with self.assertRaises(RangeError):
            _service().resolve_range(body, info)


class RouteTableTest(CustomTestCase):
    """The endpoints the extension's README documents must exist."""

    def test_every_documented_route_is_mounted(self):
        from sglang.srt.video_enhance.server import create_app

        app = create_app(TenantConfig(budget_mib=8192))
        mounted = {
            (route.path, method)
            for route in app.routes
            for method in getattr(route, "methods", ()) or ()
        }
        for path, method in (
            ("/v1/video/enhance", "GET"),
            ("/v1/video/enhance", "POST"),
            ("/v1/video/enhance/{job_id}", "GET"),
            ("/v1/video/enhance/{job_id}", "DELETE"),
            ("/v1/video/capabilities", "GET"),
            ("/v1/video/tracks", "GET"),
            ("/v1/video/engines", "GET"),
            ("/v1/video/liveness", "GET"),
        ):
            with self.subTest(path=path, method=method):
                self.assertIn((path, method), mounted)

    def test_the_enhance_get_accepts_the_range_and_job_id_parameters(self):
        from sglang.srt.video_enhance.server import create_app

        app = create_app(TenantConfig(budget_mib=8192))
        route = next(
            r
            for r in app.routes
            if r.path == "/v1/video/enhance" and "GET" in (r.methods or ())
        )
        names = {param.name for param in route.dependant.query_params}
        for required in ("source_url", "start_s", "duration_s", "job_id"):
            with self.subTest(param=required):
                self.assertIn(required, names)


class DefaultPathUnchangedTest(CustomTestCase):
    """What #338 must not have moved."""

    def test_the_request_body_defaults_to_the_whole_source(self):
        body = EnhanceRequestBody(source_url="x", source_width=1920, source_height=1080)
        self.assertEqual(body.start_s, 0.0)
        self.assertIsNone(body.duration_s)
        self.assertFalse(body.has_time_range())

    def test_the_whole_source_singleton_is_the_neutral_range(self):
        self.assertEqual(WHOLE_SOURCE, TimeRange())
        self.assertTrue(WHOLE_SOURCE.is_whole_source)
        self.assertEqual(WHOLE_SOURCE.start_frame, 0)
        self.assertIsNone(WHOLE_SOURCE.frame_limit)


if __name__ == "__main__":
    unittest.main()
