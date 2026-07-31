"""Back-pressure and executor guards.

DESIGN #333 §10 M2 acceptance gate 3 calls this the gate that matters most:
an artificially slowed client must stall the decoder within one ring depth and
device memory must stay flat. The failure it guards against -- a slow network
client causing a VRAM OOM -- is untraceable from the GPU side, so it has to be
proven structurally rather than observed in production.

These tests are hermetic: no GPU, no sockets. Frames carry a stub payload that
reports itself as CUDA-resident, so the executor's device-residency check is
exercised without a device.
"""

import asyncio
import unittest

from sglang.srt.video_enhance.chain import ChainRequest, StageKind, build_chain
from sglang.srt.video_enhance.frame_math import R4K, R1080P, PixelFormat, Resolution
from sglang.srt.video_enhance.frames import Frame, HostResidencyError, StageBase
from sglang.srt.video_enhance.pipeline import PipelineExecutor
from sglang.srt.video_enhance.ring import (
    BoundedRing,
    OverloadPolicy,
    RingClosed,
    ring_depths_for,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _FakeDevice:
    type = "cuda"
    index = 0


class _FakeTensor:
    """Minimal stand-in for a CUDA tensor: it only has to answer ``.device``."""

    device = _FakeDevice()

    def __init__(self, tag=0):
        self.tag = tag


def _frame(index, res=R1080P, fmt=PixelFormat.RGB_FP16):
    return Frame(data=_FakeTensor(index), resolution=res, format=fmt, index=index)


class _CountingStage(StageBase):
    """Identity stage that records how many frames it saw."""

    def __init__(self, name, out_res=None, delay=0.0):
        self.name = name
        self.seen = 0
        self.out_res = out_res
        self.delay = delay

    def process(self, frames):
        self.seen += len(frames)
        out = []
        for f in frames:
            res = self.out_res or f.resolution
            out.append(f.with_data(_FakeTensor(f.index), resolution=res))
        return out


class _EncodeStage(StageBase):
    name = "encode"

    def __init__(self):
        self.seen = 0

    def process(self, frames):
        self.seen += len(frames)
        return [bytes([f.index % 251]) * 16 for f in frames]

    def flush(self):
        return [b"TRAILER"]


class _RifeStage(StageBase):
    """Pair-consuming stage: returns only the frames it invents.

    Matches the real stage's contract -- source frames are interleaved by the
    executor, not re-emitted by the stage.
    """

    name = "rife"

    def __init__(self):
        self.pairs = 0

    def process(self, frames):
        assert len(frames) == 2, "RIFE must be fed a pair"
        self.pairs += 1
        left, _right = frames
        return [
            Frame(
                data=_FakeTensor(-1),
                resolution=left.resolution,
                format=left.format,
                index=left.index,
                sub_index=1,
                sub_count=2,
            )
        ]


class TestBoundedRing(CustomTestCase):
    def test_never_exceeds_depth(self):
        async def scenario():
            ring = BoundedRing("r", 3)
            for i in range(3):
                await ring.put(i)
            self.assertEqual(len(ring), 3)
            self.assertTrue(ring.is_full)
            producer = asyncio.create_task(ring.put(99))
            await asyncio.sleep(0.01)
            self.assertFalse(producer.done())
            self.assertEqual(len(ring), 3)
            await ring.get()
            await producer
            self.assertEqual(len(ring), 3)
            await ring.close()

        asyncio.run(scenario())

    def test_producer_stall_is_accounted(self):
        async def scenario():
            ring = BoundedRing("r", 1)
            await ring.put(1)
            producer = asyncio.create_task(ring.put(2))
            await asyncio.sleep(0.05)
            await ring.get()
            await producer
            self.assertGreater(ring.stats.producer_stall_seconds, 0.01)
            await ring.close()

        asyncio.run(scenario())

    def test_drop_policy_counts_and_never_grows(self):
        async def scenario():
            ring = BoundedRing("r", 2, OverloadPolicy.DROP_FRAMES)
            for i in range(6):
                await ring.put(i)
            self.assertEqual(len(ring), 2)
            self.assertEqual(ring.stats.dropped, 4)
            # The oldest are dropped: the newest two survive.
            self.assertEqual([await ring.get(), await ring.get()], [4, 5])
            await ring.close()

        asyncio.run(scenario())

    def test_get_on_closed_drained_ring_raises(self):
        async def scenario():
            ring = BoundedRing("r", 1)
            await ring.close()
            with self.assertRaises(RingClosed):
                await ring.get()

        asyncio.run(scenario())

    def test_depth_zero_is_refused(self):
        with self.assertRaises(ValueError):
            BoundedRing("r", 0)

    def test_ring_depths_sum_to_the_budget(self):
        depths = ring_depths_for(["a", "b", "c", "d"], max_in_flight=6)
        self.assertEqual(len(depths), 3)
        self.assertTrue(all(v >= 1 for v in depths.values()))
        self.assertLessEqual(sum(depths.values()), 6)


class TestExecutorBackPressure(CustomTestCase):
    def _chain_and_stages(self, fps_multiplier=1):
        request = ChainRequest(
            source=R1080P,
            target=R4K,
            fps_multiplier=fps_multiplier,
            streams_in_flight=2,
        )
        chain = build_chain(request)
        stages = {}
        for spec in chain.stages:
            if spec.kind is StageKind.ENCODE:
                stages[spec.kind] = _EncodeStage()
            elif spec.kind is StageKind.RIFE:
                stages[spec.kind] = _RifeStage()
            else:
                stages[spec.kind] = _CountingStage(
                    spec.kind.value, out_res=spec.out_res
                )
        return chain, stages

    def test_slow_client_stalls_the_decoder_within_one_ring_depth(self):
        """The acceptance-gate shape: a blocked sink must reach the source."""
        chain, stages = self._chain_and_stages()
        pulled = {"count": 0}
        release = asyncio.Event()
        ring_depth = 2

        async def source():
            index = 0
            while index < 200:
                pulled["count"] += 1
                yield _frame(index)
                index += 1

        async def slow_sink(payload):
            # Never accepts anything until released: the worst-case slow client.
            await release.wait()

        async def scenario():
            executor = PipelineExecutor(
                job_id="t",
                chain=chain,
                stages=stages,
                source=source(),
                sink=slow_sink,
                ring_depth=ring_depth,
                use_cuda_events=False,
            )
            task = asyncio.create_task(executor.run())
            await asyncio.sleep(0.2)

            # The decoder must have stopped. The bound is what the rings can
            # hold plus one frame resident in each stage task, not the 200
            # frames the source is willing to produce.
            boundaries = len(executor.rings.rings)
            ceiling = boundaries * (ring_depth + 1) + 4
            self.assertFalse(task.done())
            self.assertLess(
                pulled["count"],
                ceiling,
                f"decoder pulled {pulled['count']} frames while the sink was "
                f"blocked; back-pressure did not reach the source",
            )
            self.assertGreater(pulled["count"], 0)

            # Occupancy is flat: nothing grew past its declared depth. This is
            # the "device memory stays flat" half of the gate, expressed in the
            # unit the reservation counts.
            for name, occupancy in executor.rings.occupancies().items():
                self.assertLessEqual(
                    occupancy, ring_depth, f"ring {name} exceeded its depth"
                )

            executor.cancel()
            release.set()
            await executor.rings.close()
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.TimeoutError, RingClosed, asyncio.CancelledError):
                pass

        asyncio.run(scenario())

    def test_full_stream_reaches_the_sink_in_order(self):
        chain, stages = self._chain_and_stages()
        received = []

        async def source():
            for index in range(12):
                yield _frame(index)

        async def sink(payload):
            received.append(payload)

        async def scenario():
            executor = PipelineExecutor(
                job_id="t",
                chain=chain,
                stages=stages,
                source=source(),
                sink=sink,
                ring_depth=2,
                use_cuda_events=False,
            )
            stats = await executor.run()
            return stats

        stats = asyncio.run(scenario())
        self.assertEqual(stats.frames_decoded, 12)
        self.assertEqual(stats.frames_encoded, 12)
        self.assertEqual(stats.state, "done")
        self.assertEqual(received[-1], b"TRAILER")
        self.assertEqual(stages[StageKind.SR].seen, 12)

    def test_rife_receives_pairs_and_emits_interpolated_frames(self):
        chain, stages = self._chain_and_stages(fps_multiplier=2)
        received = []

        async def source():
            for index in range(6):
                yield _frame(index)

        async def sink(payload):
            received.append(payload)

        async def scenario():
            executor = PipelineExecutor(
                job_id="t",
                chain=chain,
                stages=stages,
                source=source(),
                sink=sink,
                ring_depth=2,
                use_cuda_events=False,
            )
            return await executor.run()

        stats = asyncio.run(scenario())
        # Six inputs form five consecutive pairs. Each pair yields one
        # interpolated frame; the executor interleaves the source frames, so
        # the output is f0 mid f1 mid f2 ... f5 -- 11 frames, which is
        # expected_frame_count(6, 2) and not the naive 6*2.
        self.assertEqual(stages[StageKind.RIFE].pairs, 5)
        self.assertEqual(stats.frames_encoded, 11)

    def test_host_resident_frame_is_rejected_at_the_boundary(self):
        chain, stages = self._chain_and_stages()

        class _HostStage(StageBase):
            name = "sr"

            def process(self, frames):
                return [
                    Frame(
                        data=object(),
                        resolution=R4K,
                        format=PixelFormat.RGB_FP16,
                        index=f.index,
                    )
                    for f in frames
                ]

        stages[StageKind.SR] = _HostStage()

        async def source():
            yield _frame(0)

        async def sink(payload):
            return None

        async def scenario():
            executor = PipelineExecutor(
                job_id="t",
                chain=chain,
                stages=stages,
                source=source(),
                sink=sink,
                ring_depth=2,
                use_cuda_events=False,
            )
            await executor.run()

        with self.assertRaises(HostResidencyError):
            asyncio.run(scenario())

    def test_missing_stage_implementation_is_refused_before_running(self):
        chain, stages = self._chain_and_stages()
        del stages[StageKind.SR]

        async def source():
            if False:
                yield None

        with self.assertRaises(ValueError):
            PipelineExecutor(
                job_id="t",
                chain=chain,
                stages=stages,
                source=source(),
                sink=lambda p: asyncio.sleep(0),
                ring_depth=2,
                use_cuda_events=False,
            )

    def test_cancellation_stops_the_run(self):
        chain, stages = self._chain_and_stages()

        async def source():
            index = 0
            while True:
                yield _frame(index)
                index += 1
                await asyncio.sleep(0)

        async def sink(payload):
            await asyncio.sleep(0)

        async def scenario():
            executor = PipelineExecutor(
                job_id="t",
                chain=chain,
                stages=stages,
                source=source(),
                sink=sink,
                ring_depth=2,
                use_cuda_events=False,
            )
            task = asyncio.create_task(executor.run())
            await asyncio.sleep(0.05)
            executor.cancel()
            await executor.rings.close()
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.TimeoutError, RingClosed, asyncio.CancelledError):
                pass
            return executor

        executor = asyncio.run(scenario())
        self.assertTrue(executor.cancelled)


class TestFrameResidency(CustomTestCase):
    def test_eos_frame_bypasses_the_device_check(self):
        Frame.eos(7).require_device("x")

    def test_non_tensor_payload_is_rejected(self):
        frame = Frame(
            data=object(),
            resolution=Resolution(4, 4),
            format=PixelFormat.RGB_FP16,
            index=0,
        )
        with self.assertRaises(HostResidencyError):
            frame.require_device("x")


if __name__ == "__main__":
    unittest.main()
