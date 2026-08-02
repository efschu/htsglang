"""Streaming-input admission, the seconds-deep buffer, and sustained rate (#448).

All hermetic and all torch-free: the producers below are lists of integers and
the clock is a counter, because the properties under test are about waiting,
bounding and counting rather than about frames.

The admission tests are the important half. Each refusal exists because the
alternative is a job that *looks* like it worked -- a growing source cut into
chunks against the frame count it happened to have, or a live feed silently
losing the frames that arrive while the reader is stalled -- so every refusal
is paired with the admission of the nearest configuration that is sound,
which is what makes the gate a gate rather than a blanket no.
"""

import asyncio
import unittest
from fractions import Fraction

from sglang.srt.video_enhance.ring import OverloadPolicy, RingClosed
from sglang.srt.video_enhance.streaming import (
    DEFAULT_WATERMARK_S,
    NO_MORE_FRAMES,
    NOT_YET,
    RateWindow,
    SecondsDeepBuffer,
    SourceKind,
    StreamingAdmissionError,
    StreamingPolicy,
    admit_streaming_source,
    drain_to_sink,
    growing_frames,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


OUT_50 = Fraction(50)


def policy(**kwargs):
    kwargs.setdefault("output_frame_rate", OUT_50)
    return StreamingPolicy(**kwargs)


class FakeClock:
    """A clock the test moves by hand, so no test waits on real time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class AdmissionTest(CustomTestCase):
    def test_a_finished_source_is_the_unchanged_default_path(self):
        admission = admit_streaming_source(policy())
        self.assertTrue(admission.admitted)
        self.assertIs(admission.kind, SourceKind.FINISHED)
        self.assertFalse(admission.is_streaming)
        self.assertIn("unchanged", " ".join(admission.notes))

    def test_a_finished_source_may_be_chunked_when_its_length_is_known(self):
        admission = admit_streaming_source(policy(), chunked=True, total_frames=480)
        self.assertTrue(admission.admitted)

    def test_a_chunked_run_without_a_frame_count_is_refused(self):
        admission = admit_streaming_source(policy(), chunked=True)
        self.assertFalse(admission.admitted)
        self.assertIn("frame count", admission.reason)

    def test_a_growing_source_is_refused_on_the_chunk_executor(self):
        """The refusal that matters: chunking a source with no final length.

        ``verify_chunk_arithmetic`` would happily verify a split of the frames
        that exist right now, and the job would end successfully over a prefix
        of the source.
        """
        admission = admit_streaming_source(
            policy(kind=SourceKind.GROWING), chunked=True, total_frames=480
        )
        self.assertFalse(admission.admitted)
        self.assertIn("chunk executor", admission.reason)
        self.assertIn("final count", admission.reason)

    def test_the_same_growing_source_is_admitted_on_the_single_card_path(self):
        """Can-fail proof for the refusal above: one argument differs."""
        admission = admit_streaming_source(policy(kind=SourceKind.GROWING))
        self.assertTrue(admission.admitted)
        self.assertTrue(admission.is_streaming)
        self.assertIn("treated as finished", " ".join(admission.notes))

    def test_a_live_source_cannot_be_stalled(self):
        admission = admit_streaming_source(
            policy(kind=SourceKind.LIVE, overload=OverloadPolicy.STALL)
        )
        self.assertFalse(admission.admitted)
        self.assertIn("drop_frames", admission.reason)

    def test_a_live_source_with_drop_frames_is_admitted_and_says_drops_are_counted(self):
        admission = admit_streaming_source(
            policy(kind=SourceKind.LIVE, overload=OverloadPolicy.DROP_FRAMES)
        )
        self.assertTrue(admission.admitted)
        self.assertIn("never silent", " ".join(admission.notes))

    def test_the_watermark_is_seconds_converted_through_the_output_rate(self):
        for seconds, rate, frames in (
            (2.0, Fraction(50), 100),
            (2.0, Fraction(25), 50),
            (0.5, Fraction(24000, 1001), 12),
            (0.0, Fraction(50), 1),
        ):
            with self.subTest(seconds=seconds, rate=rate):
                admission = admit_streaming_source(
                    policy(
                        kind=SourceKind.GROWING,
                        output_frame_rate=rate,
                        watermark_s=seconds,
                    )
                )
                self.assertEqual(admission.buffer_depth_frames, frames)

    def test_the_default_watermark_is_declared_not_incidental(self):
        admission = admit_streaming_source(
            policy(kind=SourceKind.GROWING, watermark_s=DEFAULT_WATERMARK_S)
        )
        self.assertAlmostEqual(admission.watermark_s, DEFAULT_WATERMARK_S, places=6)

    def test_require_raises_the_refusal_text(self):
        admission = admit_streaming_source(policy(kind=SourceKind.GROWING), chunked=True)
        with self.assertRaises(StreamingAdmissionError) as caught:
            admission.require()
        self.assertIn("chunk executor", str(caught.exception))

    def test_the_verdict_serialises_for_the_job_status(self):
        payload = admit_streaming_source(
            policy(kind=SourceKind.LIVE, overload=OverloadPolicy.DROP_FRAMES)
        ).as_dict()
        self.assertEqual(payload["kind"], "live")
        self.assertEqual(payload["overload"], "drop_frames")
        self.assertEqual(payload["buffer_depth_frames"], 100)
        self.assertTrue(payload["notes"])


class SecondsDeepBufferTest(CustomTestCase):
    def test_the_depth_is_the_watermark_in_frames(self):
        buffer = SecondsDeepBuffer(
            "out", policy(kind=SourceKind.GROWING, watermark_s=0.2)
        )
        self.assertEqual(buffer.depth_frames, 10)
        self.assertAlmostEqual(buffer.watermark_s, 0.2, places=6)

    def test_it_never_grows_past_the_watermark_and_stalls_the_producer(self):
        """Back-pressure across the buffer, as a measured stall rather than a claim."""

        async def scenario():
            buffer = SecondsDeepBuffer(
                "out", policy(kind=SourceKind.GROWING, watermark_s=0.06)
            )
            self.assertEqual(buffer.depth_frames, 3)
            occupancies = []

            async def produce():
                for i in range(20):
                    await buffer.put(i)
                    occupancies.append(len(buffer.ring))

            async def consume():
                await asyncio.sleep(0.01)
                for _ in range(20):
                    await buffer.get()
                    await asyncio.sleep(0)

            await asyncio.gather(produce(), consume())
            return buffer, occupancies

        buffer, occupancies = asyncio.run(scenario())
        self.assertLessEqual(max(occupancies), buffer.depth_frames)
        self.assertGreater(buffer.ring.stats.producer_stall_seconds, 0.0)
        self.assertEqual(buffer.frames_in, 20)
        self.assertEqual(buffer.frames_out, 20)

    def test_a_live_buffer_drops_the_oldest_and_counts_it(self):
        async def scenario():
            buffer = SecondsDeepBuffer(
                "out",
                policy(
                    kind=SourceKind.LIVE,
                    watermark_s=0.04,
                    overload=OverloadPolicy.DROP_FRAMES,
                ),
            )
            for i in range(6):
                await buffer.put(i)
            return buffer

        buffer = asyncio.run(scenario())
        self.assertEqual(buffer.depth_frames, 2)
        self.assertEqual(len(buffer.ring), 2)
        self.assertEqual(buffer.ring.stats.dropped, 4)
        self.assertEqual(buffer.snapshot()["dropped"], 4)

    def test_seconds_buffered_reads_back_in_seconds(self):
        async def scenario():
            buffer = SecondsDeepBuffer(
                "out", policy(kind=SourceKind.GROWING, watermark_s=1.0)
            )
            for i in range(25):
                await buffer.put(i)
            return buffer

        buffer = asyncio.run(scenario())
        self.assertAlmostEqual(buffer.seconds_buffered(), 0.5, places=6)

    def test_draining_to_a_sink_stops_when_the_buffer_closes(self):
        async def scenario():
            buffer = SecondsDeepBuffer(
                "out", policy(kind=SourceKind.GROWING, watermark_s=1.0)
            )
            got = []

            async def sink(item):
                got.append(item)

            task = asyncio.create_task(drain_to_sink(buffer, sink))
            for i in range(5):
                await buffer.put(i)
            await asyncio.sleep(0)
            await buffer.close()
            await asyncio.wait_for(task, timeout=2)
            return got

        self.assertEqual(asyncio.run(scenario()), [0, 1, 2, 3, 4])

    def test_a_closed_buffer_refuses_a_get(self):
        async def scenario():
            buffer = SecondsDeepBuffer("out", policy(kind=SourceKind.GROWING))
            await buffer.close()
            with self.assertRaises(RingClosed):
                await buffer.get()

        asyncio.run(scenario())


class RateWindowTest(CustomTestCase):
    def test_one_sample_is_not_a_rate(self):
        clock = FakeClock()
        window = RateWindow(window_s=10.0, clock=clock)
        window.observe(frames_in=0, frames_out=0)
        self.assertIsNone(window.fps_in)
        self.assertIsNone(window.fps_out)
        # Zero and "not measured" are different states and the snapshot keeps
        # them different, because a live watch showing 0 fps for "no data yet"
        # is reporting a stall that is not there.
        self.assertIsNone(window.snapshot()["fps_out"])

    def test_a_steady_chain_reports_its_rate_in_and_out(self):
        clock = FakeClock()
        window = RateWindow(window_s=10.0, clock=clock)
        for step in range(11):
            window.observe(frames_in=25 * step, frames_out=50 * step)
            clock.advance(1.0)
        self.assertAlmostEqual(window.fps_in, 25.0, places=6)
        self.assertAlmostEqual(window.fps_out, 50.0, places=6)

    def test_the_window_slides_so_an_old_burst_stops_counting(self):
        clock = FakeClock()
        window = RateWindow(window_s=5.0, clock=clock)
        # Ten seconds at 100 fps out, then ten at 10 fps out.
        frames_out = 0
        for _ in range(10):
            window.observe(frames_in=0, frames_out=frames_out)
            frames_out += 100
            clock.advance(1.0)
        fast = window.fps_out
        for _ in range(10):
            window.observe(frames_in=0, frames_out=frames_out)
            frames_out += 10
            clock.advance(1.0)
        self.assertAlmostEqual(fast, 100.0, places=6)
        self.assertAlmostEqual(window.fps_out, 10.0, places=6)

    def test_sparse_samples_still_span_the_window(self):
        """A sample at the window edge is kept, not pruned into a shorter span."""
        clock = FakeClock()
        window = RateWindow(window_s=10.0, clock=clock)
        window.observe(frames_in=0, frames_out=0)
        clock.advance(30.0)
        window.observe(frames_in=300, frames_out=600)
        self.assertAlmostEqual(window.fps_out, 20.0, places=6)
        self.assertEqual(window.snapshot()["measured_span_s"], 30.0)

    def test_sustaining_needs_both_a_target_and_a_measurement(self):
        clock = FakeClock()
        window = RateWindow(window_s=10.0, clock=clock, target_output_fps=50.0)
        self.assertIsNone(window.sustaining)
        window.observe(frames_in=0, frames_out=0)
        clock.advance(1.0)
        window.observe(frames_in=25, frames_out=60)
        self.assertTrue(window.sustaining)
        clock.advance(1.0)
        window.observe(frames_in=50, frames_out=70)
        self.assertFalse(window.sustaining)

    def test_a_window_with_no_target_reports_the_rate_and_no_verdict(self):
        clock = FakeClock()
        window = RateWindow(window_s=10.0, clock=clock)
        window.observe(frames_in=0, frames_out=0)
        clock.advance(2.0)
        window.observe(frames_in=50, frames_out=100)
        snapshot = window.snapshot()
        self.assertEqual(snapshot["fps_out"], 50.0)
        self.assertIsNone(snapshot["sustaining"])
        self.assertEqual(snapshot["frames_out"], 100)


class GrowingSourceTest(CustomTestCase):
    """A producer that has nothing *yet* must not end the stream."""

    def test_not_yet_is_not_the_end_of_the_stream(self):
        script = [1, NOT_YET, NOT_YET, 2, NOT_YET, 3, NO_MORE_FRAMES]
        clock = FakeClock()

        async def scenario():
            admission = admit_streaming_source(policy(kind=SourceKind.GROWING))

            async def sleep(_seconds):
                clock.advance(_seconds)

            got = []
            async for frame in growing_frames(
                lambda: script.pop(0), admission, sleep=sleep, clock=clock
            ):
                got.append(frame)
            return got

        self.assertEqual(asyncio.run(scenario()), [1, 2, 3])

    def test_a_plain_iterator_would_have_stopped_at_the_first_gap(self):
        """The falsifier for the whole adapter, spelled out.

        This is what the executor's existing source contract does with the
        same script: the first ``NOT_YET`` is just an object, and a producer
        that instead returned "no more" at the tail would end the job over a
        prefix. The adapter exists to make those two answers different.
        """
        script = [1, NO_MORE_FRAMES, 2, 3]
        clock = FakeClock()

        async def scenario():
            admission = admit_streaming_source(policy(kind=SourceKind.GROWING))
            got = []
            async for frame in growing_frames(
                lambda: script.pop(0), admission, clock=clock
            ):
                got.append(frame)
            return got

        self.assertEqual(asyncio.run(scenario()), [1])

    def test_the_idle_timeout_ends_a_writer_that_stopped(self):
        clock = FakeClock()

        async def scenario():
            admission = admit_streaming_source(
                policy(kind=SourceKind.GROWING, idle_timeout_s=5.0)
            )

            async def sleep(seconds):
                clock.advance(seconds)

            got = []
            async for frame in growing_frames(
                lambda: got and NOT_YET or 7, admission, sleep=sleep, clock=clock
            ):
                got.append(frame)
            return got, clock.now

        got, elapsed = asyncio.run(scenario())
        self.assertEqual(got, [7])
        # The timeout is measured from the last *frame*, not the last call, so
        # a producer answering NOT_YET a hundred times still takes the full
        # five seconds and no longer.
        self.assertGreaterEqual(elapsed, 5.0)
        self.assertLess(elapsed, 5.0 + 0.06)

    def test_an_async_producer_is_awaited(self):
        clock = FakeClock()
        script = [1, 2, NO_MORE_FRAMES]

        async def produce():
            return script.pop(0)

        async def scenario():
            admission = admit_streaming_source(policy(kind=SourceKind.GROWING))
            return [
                frame
                async for frame in growing_frames(produce, admission, clock=clock)
            ]

        self.assertEqual(asyncio.run(scenario()), [1, 2])

    def test_it_refuses_to_run_on_an_unadmitted_source(self):
        async def scenario():
            admission = admit_streaming_source(
                policy(kind=SourceKind.LIVE, overload=OverloadPolicy.STALL)
            )
            async for _ in growing_frames(lambda: NO_MORE_FRAMES, admission):
                pass

        with self.assertRaises(StreamingAdmissionError):
            asyncio.run(scenario())

    def test_it_is_a_pull_source_and_runs_no_further_than_it_is_drained(self):
        """Back-pressure is unmediated: nothing here buffers a frame.

        The producer is asked exactly once per frame the consumer takes, so a
        consumer that stops taking stops the producer -- which is what makes
        the existing ring the only thing bounding memory.
        """
        calls = []
        clock = FakeClock()

        def produce():
            calls.append(len(calls))
            return len(calls) if len(calls) <= 50 else NO_MORE_FRAMES

        async def scenario():
            admission = admit_streaming_source(policy(kind=SourceKind.GROWING))
            source = growing_frames(produce, admission, clock=clock)
            taken = [await anext(source) for _ in range(3)]
            await source.aclose()
            return taken

        taken = asyncio.run(scenario())
        self.assertEqual(taken, [1, 2, 3])
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
