"""Detecting a consumer that stopped reading, and releasing what it held.

The case under test is not the client that disconnects -- that one already
works, because Starlette throws into the response generator and its
``finally`` runs. It is the client that neither closes nor reads: the socket
stays open, the TCP window stays full, the sink coroutine never returns, and
back-pressure working exactly as designed holds a decoder, an encoder and a
VRAM reservation for a viewer who is gone.

From the server's side that is indistinguishable from a very slow viewer.
The only thing separating them is how long, which is why every assertion
here is about a configured duration rather than about a state.
"""

import asyncio
import unittest

from sglang.srt.video_enhance.frame_math import PixelFormat, Resolution
from sglang.srt.video_enhance.chain import StageKind
from sglang.srt.video_enhance.frames import Frame, StageBase
from sglang.srt.video_enhance.liveness import (
    DEFAULT_TIMEOUTS_S,
    ConsumerWatchdog,
    EndpointClass,
    LivenessConfig,
    LivenessPolicy,
)
from sglang.srt.video_enhance.server import EnhanceRequestBody, VideoEnhanceService
from sglang.srt.video_enhance.tenant import TenantConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeClock:
    """A clock the test drives, so a 300 s timeout costs no wall time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class PolicyTest(CustomTestCase):
    def test_each_endpoint_class_has_a_default(self):
        for endpoint_class in EndpointClass:
            self.assertIn(endpoint_class, DEFAULT_TIMEOUTS_S)

    def test_a_preview_tap_is_less_patient_than_a_video_stream(self):
        """A viewer may pause a film; a preview has no reason to go quiet."""
        self.assertLess(
            DEFAULT_TIMEOUTS_S[EndpointClass.PREVIEW_TAP],
            DEFAULT_TIMEOUTS_S[EndpointClass.VIDEO_STREAM],
        )

    def test_an_explicit_timeout_overrides_the_class_default(self):
        policy = LivenessPolicy(
            endpoint_class=EndpointClass.VIDEO_STREAM, timeout_s=12.0
        )
        self.assertEqual(policy.resolved_timeout(), 12.0)

    def test_zero_disables_detection_rather_than_meaning_immediately(self):
        """A batch export nobody watches by design must be expressible."""
        policy = LivenessPolicy(
            endpoint_class=EndpointClass.VIDEO_STREAM, timeout_s=0.0
        )
        self.assertIsNone(policy.resolved_timeout())

    def test_a_nonsense_poll_interval_is_refused(self):
        with self.assertRaises(ValueError):
            LivenessPolicy(poll_interval_s=0)


class ConfigParsingTest(CustomTestCase):
    def test_parses_a_per_class_spec(self):
        config = LivenessConfig.parse("video_stream=42,preview_tap=7")
        self.assertEqual(
            config.policy_for(EndpointClass.VIDEO_STREAM).resolved_timeout(), 42.0
        )
        self.assertEqual(
            config.policy_for(EndpointClass.PREVIEW_TAP).resolved_timeout(), 7.0
        )

    def test_unmentioned_classes_keep_their_default(self):
        config = LivenessConfig.parse("preview_tap=7")
        self.assertEqual(
            config.policy_for(EndpointClass.VIDEO_STREAM).resolved_timeout(),
            DEFAULT_TIMEOUTS_S[EndpointClass.VIDEO_STREAM],
        )

    def test_an_unknown_class_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            LivenessConfig.parse("llm_stream=5")
        self.assertIn("llm_stream", str(ctx.exception))

    def test_a_non_numeric_timeout_is_refused(self):
        with self.assertRaises(ValueError):
            LivenessConfig.parse("video_stream=soon")

    def test_empty_spec_is_all_defaults(self):
        config = LivenessConfig.parse(None)
        self.assertEqual(config.describe()["video_stream"], 300.0)


class WatchdogTest(CustomTestCase):
    def _watchdog(self, clock, timeout=10.0, released=None):
        released = released if released is not None else []

        async def release() -> None:
            released.append(True)

        watchdog = ConsumerWatchdog(
            job_id="j",
            policy=LivenessPolicy(
                endpoint_class=EndpointClass.VIDEO_STREAM,
                timeout_s=timeout,
                poll_interval_s=0.01,
            ),
            release=release,
            clock=clock,
        )
        return watchdog, released

    def test_a_silent_consumer_is_released_after_the_timeout(self):
        async def scenario():
            clock = FakeClock()
            watchdog, released = self._watchdog(clock)
            watchdog.start()
            clock.advance(11.0)
            for _ in range(200):
                if watchdog.released:
                    break
                await asyncio.sleep(0.01)
            await watchdog.stop()
            return released, watchdog.state

        released, state = asyncio.run(scenario())
        self.assertEqual(released, [True])
        self.assertTrue(state.declared_dead)

    def test_a_slow_but_alive_consumer_is_not_released(self):
        """Progress just inside the limit, repeatedly, must keep the job."""

        async def scenario():
            clock = FakeClock()
            watchdog, released = self._watchdog(clock)
            watchdog.start()
            for _ in range(5):
                clock.advance(9.0)
                await asyncio.sleep(0.02)
                watchdog.note_progress(1024)
            await asyncio.sleep(0.02)
            await watchdog.stop()
            return released, watchdog.state

        released, state = asyncio.run(scenario())
        self.assertEqual(released, [])
        self.assertFalse(state.declared_dead)
        self.assertEqual(state.writes_accepted, 5)
        self.assertEqual(state.bytes_accepted, 5 * 1024)

    def test_progress_resets_the_clock(self):
        clock = FakeClock()
        watchdog, _ = self._watchdog(clock)
        clock.advance(8.0)
        self.assertAlmostEqual(watchdog.state.silent_for(clock()), 8.0)
        watchdog.note_progress(10)
        self.assertAlmostEqual(watchdog.state.silent_for(clock()), 0.0)

    def test_a_disabled_timeout_never_releases(self):
        async def scenario():
            clock = FakeClock()
            watchdog, released = self._watchdog(clock, timeout=0.0)
            watchdog.start()
            clock.advance(100000.0)
            await asyncio.sleep(0.05)
            await watchdog.stop()
            return released

        self.assertEqual(asyncio.run(scenario()), [])

    def test_a_release_that_hangs_does_not_hang_the_watchdog(self):
        """A teardown that will not finish must not hold the reservation."""

        async def scenario():
            clock = FakeClock()

            async def release() -> None:
                await asyncio.sleep(30)

            watchdog = ConsumerWatchdog(
                job_id="j",
                policy=LivenessPolicy(
                    timeout_s=1.0, poll_interval_s=0.01, teardown_timeout_s=0.05
                ),
                release=release,
                clock=clock,
            )
            watchdog.start()
            clock.advance(2.0)
            for _ in range(200):
                if watchdog.released:
                    break
                await asyncio.sleep(0.01)
            await watchdog.stop()
            return watchdog.released

        self.assertTrue(asyncio.run(scenario()))

    def test_a_release_that_raises_is_reported_not_swallowed_into_a_hang(self):
        async def scenario():
            clock = FakeClock()

            async def release() -> None:
                raise RuntimeError("encoder context already gone")

            watchdog = ConsumerWatchdog(
                job_id="j",
                policy=LivenessPolicy(timeout_s=1.0, poll_interval_s=0.01),
                release=release,
                clock=clock,
            )
            watchdog.start()
            clock.advance(2.0)
            for _ in range(200):
                if watchdog.released:
                    break
                await asyncio.sleep(0.01)
            await watchdog.stop()
            return watchdog.released

        self.assertTrue(asyncio.run(scenario()))

    def test_stopping_a_live_watchdog_leaves_the_job_alone(self):
        """The normal path: the stream ended, nobody died."""

        async def scenario():
            clock = FakeClock()
            watchdog, released = self._watchdog(clock)
            watchdog.start()
            watchdog.note_progress(1)
            await watchdog.stop()
            clock.advance(1000.0)
            await asyncio.sleep(0.05)
            return released

        self.assertEqual(asyncio.run(scenario()), [])

    def test_state_is_reportable_by_the_progress_endpoint(self):
        clock = FakeClock()
        watchdog, _ = self._watchdog(clock)
        watchdog.note_progress(2048)
        clock.advance(3.0)
        snapshot = watchdog.state.snapshot(clock())
        self.assertEqual(snapshot["bytes_accepted"], 2048)
        self.assertEqual(snapshot["endpoint_class"], "video_stream")
        self.assertAlmostEqual(snapshot["silent_for_s"], 3.0)
        self.assertFalse(snapshot["declared_dead"])


class ReleaseSemanticsTest(CustomTestCase):
    """What release has to do, expressed against the two things the server
    actually calls: cancel the executor and close the bridge."""

    def test_release_cancels_and_unblocks(self):
        async def scenario():
            cancelled = []
            closed = []

            async def release() -> None:
                cancelled.append(True)
                closed.append(True)

            clock = FakeClock()
            watchdog = ConsumerWatchdog(
                job_id="j",
                policy=LivenessPolicy(timeout_s=1.0, poll_interval_s=0.01),
                release=release,
                clock=clock,
            )
            watchdog.start()
            clock.advance(2.0)
            for _ in range(200):
                if watchdog.released:
                    break
                await asyncio.sleep(0.01)
            await watchdog.stop()
            return cancelled, closed

        cancelled, closed = asyncio.run(scenario())
        self.assertEqual((cancelled, closed), ([True], [True]))


class _FakeDevice:
    type = "cuda"
    index = 0


class _FakeTensor:
    device = _FakeDevice()

    def __init__(self, tag=0):
        self.tag = tag


class _PassthroughStage(StageBase):
    def __init__(self, name, out_res=None):
        self.name = name
        self.out_res = out_res
        self.closed = False

    def process(self, frames):
        return [
            f.with_data(_FakeTensor(f.index), resolution=self.out_res or f.resolution)
            for f in frames
        ]

    def close(self):
        self.closed = True


class _EncodeStage(StageBase):
    name = "encode"

    def __init__(self):
        self.closed = False

    def process(self, frames):
        return [b"x" * 64 for _ in frames]

    def flush(self):
        return []

    def close(self):
        self.closed = True


class DeadConsumerReleasesTheJobTest(CustomTestCase):
    """The end-to-end shape, through the real service and the real executor.

    A consumer that takes one chunk and then stops reading forever must have
    its decoder stopped and its stages closed within the configured timeout,
    without the test closing the generator -- because a real dead client does
    not close anything either.
    """

    def _service(self, timeout_s):
        config = TenantConfig(budget_mib=8192, rife_measured_bytes_per_pair=1)
        return VideoEnhanceService(
            config,
            liveness=LivenessConfig(
                timeouts_s={"video_stream": timeout_s},
                poll_interval_s=0.01,
                teardown_timeout_s=2.0,
            ),
        )

    def _body(self):
        return EnhanceRequestBody(
            source_url="/tmp/none.mp4",
            source_width=1920,
            source_height=1080,
            target="3840x2160",
            fps_multiplier=1,
            streams_in_flight=2,
        )

    def _stage_factory(self, record):
        def factory(chain):
            stages = {}
            for spec in chain.stages:
                if spec.kind is StageKind.ENCODE:
                    stages[spec.kind] = _EncodeStage()
                else:
                    stages[spec.kind] = _PassthroughStage(
                        spec.kind.value, out_res=spec.out_res
                    )
            record.extend(stages.values())
            return stages

        return factory

    def test_a_consumer_that_stops_reading_is_released(self):
        """The whole point: the consumer does not disconnect, it just stops.

        The generator is therefore left suspended at its ``yield`` and never
        reaches its own ``finally``, which is exactly why the watchdog's
        release has to close the stages itself. Asserting on the stages
        rather than on the stream ending is the difference between "the
        client was noticed" and "the decoder and encoder were given back".
        """

        async def scenario():
            service = self._service(timeout_s=0.15)
            stages: list = []
            pulled = {"count": 0}

            def source_factory(chain):
                async def gen():
                    index = 0
                    while index < 100000:
                        pulled["count"] += 1
                        yield Frame(
                            data=_FakeTensor(index),
                            resolution=Resolution(1920, 1080),
                            format=PixelFormat.NV12,
                            index=index,
                        )
                        index += 1
                        await asyncio.sleep(0)

                return gen()

            stream = service.stream_response(
                self._body(),
                source_factory=source_factory,
                stage_factory=self._stage_factory(stages),
                remux=False,
            )
            # One chunk taken, proving the peer was alive. Then nothing: no
            # further __anext__, no aclose, exactly as a dead client behaves.
            await stream.__anext__()
            after_first = pulled["count"]

            job = next(iter(service.jobs.values()))
            deadline = asyncio.get_running_loop().time() + 5.0
            while asyncio.get_running_loop().time() < deadline:
                if job.watchdog is not None and job.watchdog.released:
                    break
                await asyncio.sleep(0.01)
            released = job.watchdog is not None and job.watchdog.released
            # Let the decoder run if it were still going.
            await asyncio.sleep(0.05)
            total = pulled["count"]
            await stream.aclose()
            return released, job, stages, after_first, total

        released, job, stages, after_first, total = asyncio.run(scenario())
        self.assertTrue(released, "the consumer was never declared dead")
        self.assertTrue(job.watchdog.state.declared_dead)
        self.assertTrue(job.executor.cancelled, "the executor was not cancelled")
        self.assertTrue(stages, "no stages were built")
        self.assertTrue(
            all(getattr(s, "closed", False) for s in stages),
            "stages were not closed, so decoder and encoder contexts leaked",
        )
        # The decoder stopped rather than running on: the source is willing
        # to produce 100000 frames and the rings bound how far it got.
        self.assertLess(total, after_first + 64, "the decoder kept pulling")

    def test_a_consumer_that_keeps_reading_is_not_released(self):
        """The control. The same job, consumed normally, runs to completion."""

        async def scenario():
            service = self._service(timeout_s=0.15)
            stages: list = []

            def source_factory(chain):
                async def gen():
                    for index in range(12):
                        yield Frame(
                            data=_FakeTensor(index),
                            resolution=Resolution(1920, 1080),
                            format=PixelFormat.NV12,
                            index=index,
                        )
                        await asyncio.sleep(0)

                return gen()

            chunks = 0
            async for _chunk in service.stream_response(
                self._body(),
                source_factory=source_factory,
                stage_factory=self._stage_factory(stages),
                remux=False,
            ):
                chunks += 1
            job = next(iter(service.jobs.values()))
            return chunks, job

        chunks, job = asyncio.run(scenario())
        self.assertEqual(chunks, 12)
        self.assertIsNotNone(job.watchdog)
        self.assertFalse(job.watchdog.state.declared_dead)
        self.assertEqual(job.watchdog.state.writes_accepted, 12)

    def test_the_progress_endpoint_reports_the_consumer(self):
        async def scenario():
            service = self._service(timeout_s=30.0)
            stages: list = []

            def source_factory(chain):
                async def gen():
                    for index in range(4):
                        yield Frame(
                            data=_FakeTensor(index),
                            resolution=Resolution(1920, 1080),
                            format=PixelFormat.NV12,
                            index=index,
                        )
                        await asyncio.sleep(0)

                return gen()

            async for _ in service.stream_response(
                self._body(),
                source_factory=source_factory,
                stage_factory=self._stage_factory(stages),
                remux=False,
            ):
                pass
            job_id = next(iter(service.jobs))
            return service.progress(job_id)

        snapshot = asyncio.run(scenario())
        self.assertIn("consumer", snapshot)
        self.assertEqual(snapshot["consumer"]["endpoint_class"], "video_stream")
        self.assertGreater(snapshot["consumer"]["bytes_accepted"], 0)


if __name__ == "__main__":
    unittest.main()
