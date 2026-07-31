# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The chain executor: stage tasks joined by bounded rings.

One asyncio task per stage, one :class:`BoundedRing` per boundary. Stages are
synchronous with respect to the task that drives them and asynchronous with
respect to the device -- they enqueue CUDA work and return -- so the pipeline
overlaps naturally and the rings, not the device queue, are what bounds
memory.

Two properties are enforced here rather than documented:

*   **The socket write is the throttle.** The encode task ``await``s the sink
    coroutine. When the client's TCP window is full the sink blocks, the
    encode task stops draining its input ring, the ring fills, and within one
    ring depth the block reaches the decode task, which is a pull source and
    stops pulling. That is §8.4 rules 2 and 3, and the executor is where they
    are true or not true.
*   **No host round-trip between decode and encode.** Every frame handed
    between stages goes through :meth:`Frame.require_device`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Mapping, Sequence

from sglang.srt.video_enhance.chain import Chain, StageKind
from sglang.srt.video_enhance.frames import Frame
from sglang.srt.video_enhance.ring import (
    BoundedRing,
    OverloadPolicy,
    RingClosed,
    RingSet,
)
from sglang.srt.video_enhance.timing import ChainTimers

logger = logging.getLogger(__name__)

FrameSource = AsyncIterator[Frame]
ByteSink = Callable[[bytes], Awaitable[None]]


@dataclass
class EnhanceStats:
    """What ``GET /v1/video/enhance/{id}`` reports."""

    job_id: str
    state: str = "pending"
    frames_decoded: int = 0
    frames_enhanced: int = 0
    frames_encoded: int = 0
    #: Frames that reached the encode task and were withheld by an
    #: ``encode_filter``. Only a multi-card shard sets one; the count is
    #: reported so a seam frame withheld here can be matched against the frame
    #: the neighbouring chunk emitted in its place.
    frames_skipped: int = 0
    bytes_out: int = 0
    dropped: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    ring_occupancy: dict[str, int] = field(default_factory=dict)
    stage_ms_per_frame: dict[str, float] = field(default_factory=dict)

    def snapshot(self) -> dict:
        elapsed = (self.finished_at or time.time()) - self.started_at
        return {
            "id": self.job_id,
            "state": self.state,
            "frames_decoded": self.frames_decoded,
            "frames_enhanced": self.frames_enhanced,
            "frames_encoded": self.frames_encoded,
            "frames_skipped": self.frames_skipped,
            "bytes_out": self.bytes_out,
            "dropped": self.dropped,
            "elapsed_s": round(elapsed, 3),
            "ring_occupancy": dict(self.ring_occupancy),
            "stage_ms_per_frame": {
                k: round(v, 4) for k, v in self.stage_ms_per_frame.items()
            },
            "error": self.error,
        }


class _ArityWindow:
    """Feeds a stage that consumes more than one frame per invocation.

    RIFE reads a pair and emits ``multiplier - 1`` frames between them. The
    window holds the previous frame, so a pair is formed from consecutive
    inputs and the trailing input is emitted unchanged after the interpolated
    ones -- which is what keeps output ordering monotonic in
    ``Frame.order_key``.
    """

    def __init__(self, arity_in: int) -> None:
        if arity_in < 1:
            raise ValueError("arity_in must be at least 1")
        self.arity_in = arity_in
        self._held: list[Frame] = []
        self._emitted_any = False

    def feed(self, frame: Frame) -> list[Frame] | None:
        self._held.append(frame)
        if len(self._held) < self.arity_in:
            return None
        window = list(self._held)
        self._emitted_any = True
        # Consecutive windows overlap by ``arity_in - 1`` frames, so a pair
        # consumer sees (f0,f1), (f1,f2), ... A single-frame stage retains
        # nothing; retaining there would re-submit every frame twice.
        self._held = self._held[1:] if self.arity_in > 1 else []
        return window

    def flush(self) -> list[Frame]:
        """Frames still held at end of stream that were never processed.

        Once a window has been emitted, the frames still held are the overlap
        with that window and are already downstream; returning them would
        duplicate them. Only a stream too short to fill a single window has
        genuinely unprocessed frames left here, and those are passed through
        rather than dropped.
        """
        held, self._held = self._held, []
        return [] if self._emitted_any else held


class PipelineExecutor:
    """Runs one chain for one request."""

    def __init__(
        self,
        *,
        job_id: str,
        chain: Chain,
        stages: dict[StageKind, object],
        source: FrameSource,
        sink: ByteSink,
        ring_depth: int,
        policy: OverloadPolicy = OverloadPolicy.STALL,
        use_cuda_events: bool = True,
        encode_filter: Callable[[Frame], bool] | None = None,
        taps: Mapping[StageKind, Sequence] | None = None,
    ) -> None:
        self.job_id = job_id
        self.chain = chain
        self.stages = stages
        self.source = source
        self.sink = sink
        self.policy = policy
        # Preview taps (§8.1), keyed by the stage whose output feeds them.
        # Every object in the sequence needs one method, ``offer(frame)``,
        # which must be synchronous and must not raise -- see preview.py, where
        # that contract is the whole design. The executor calls it and ignores
        # the result, so a tap cannot influence the chain even by failing.
        self.taps = dict(taps or {})
        # Multi-card shards only (``multicard.py``). A chunk pulls one frame
        # past its own range so RIFE has the second input for the seam pair,
        # and that trailing frame is encoded by the *next* chunk -- encoding it
        # here too would duplicate one frame per seam. None, the default, is
        # the single-card path and encodes every frame it is handed.
        self.encode_filter = encode_filter
        self.stats = EnhanceStats(job_id=job_id)
        self._cancelled = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

        missing = [k.value for k in chain.kinds if k not in stages]
        if missing:
            raise ValueError(f"no implementation supplied for stages {missing}")

        # Stage order excluding decode, which is the source, and encode, which
        # is the sink; those two bracket the ring chain.
        self._middle = [
            k for k in chain.kinds if k not in (StageKind.DECODE, StageKind.ENCODE)
        ]
        self.rings = RingSet()
        names = (
            [StageKind.DECODE.value]
            + [k.value for k in self._middle]
            + [StageKind.ENCODE.value]
        )
        for left, right in zip(names, names[1:]):
            self.rings.add(BoundedRing(f"{left}->{right}", ring_depth, policy))
        self.timers = ChainTimers(
            [k.value for k in chain.kinds], use_cuda_events=use_cuda_events
        )

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def _ring(self, left: str, right: str) -> BoundedRing:
        return self.rings.rings[f"{left}->{right}"]

    def _offer_to_taps(self, taps: Sequence, frame: Frame) -> None:
        """Hand one frame to every tap on a stage. Cannot stall, cannot fail.

        No ``await``: a tap that could suspend this coroutine would be
        back-pressure from a preview viewer into the pipeline, which is the
        one thing §8.1 forbids. No exception either: a preview is a
        convenience and the job is the product, so a broken tap is logged
        once and the frame moves on.
        """
        for tap in taps:
            try:
                tap.offer(frame)
            except Exception as exc:  # noqa: BLE001 - a tap must not fail a job
                logger.warning(
                    "preview tap on job %s raised and is being ignored for this "
                    "frame; the job continues: %s",
                    self.job_id,
                    exc,
                )

    async def run(self) -> EnhanceStats:
        self.stats.state = "running"
        # Warm up before any frame moves. Engine builds, pool allocation and
        # graph capture all happen here, in one window, rather than being
        # triggered by the first frame -- capture during steady state is a bug
        # under the arbiter's capture-lock rule.
        for stage in self.stages.values():
            warmup = getattr(stage, "warmup", None)
            if callable(warmup):
                warmup()
        names = (
            [StageKind.DECODE.value]
            + [k.value for k in self._middle]
            + [StageKind.ENCODE.value]
        )
        self._tasks = [asyncio.create_task(self._decode_task(names[1]))]
        for i, kind in enumerate(self._middle):
            self._tasks.append(
                asyncio.create_task(self._stage_task(kind, names[i], names[i + 2]))
            )
        self._tasks.append(asyncio.create_task(self._encode_task(names[-2])))

        try:
            await asyncio.gather(*self._tasks)
            self.stats.state = "cancelled" if self.cancelled else "done"
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as state
            self.stats.state = "failed"
            self.stats.error = f"{type(exc).__name__}: {exc}"
            for task in self._tasks:
                task.cancel()
            raise
        finally:
            self.stats.finished_at = time.time()
            self.stats.dropped = self.rings.dropped
            self.stats.ring_occupancy = self.rings.occupancies()
            self.stats.stage_ms_per_frame = self.timers.ms_per_frame()
            await self.rings.close()
        return self.stats

    async def _decode_task(self, downstream: str) -> None:
        ring = self._ring(StageKind.DECODE.value, downstream)
        timer = self.timers[StageKind.DECODE.value]
        index = 0
        try:
            async for frame in self.source:
                if self.cancelled:
                    break
                if frame.end_of_stream:
                    # The sentinel is control, not content. Counting it made
                    # frames_decoded one larger than the number of frames the
                    # source actually produced, which is the number every
                    # shard's arithmetic is checked against.
                    break
                timer.start()
                frame.require_device(StageKind.DECODE.value)
                timer.stop()
                self.stats.frames_decoded += 1
                # This put is the back-pressure point: a full ring stops the
                # pull source, which is the decoder.
                await ring.put(frame)
                index = frame.index + 1
        except RingClosed:
            pass
        finally:
            try:
                await ring.put(Frame.eos(index))
            except RingClosed:
                pass

    async def _stage_task(
        self, kind: StageKind, upstream: str, downstream: str
    ) -> None:
        stage = self.stages[kind]
        spec = self.chain.stage(kind)
        assert spec is not None
        in_ring = self._ring(upstream, kind.value)
        out_ring = self._ring(kind.value, downstream)
        timer = self.timers[kind.value]
        window = _ArityWindow(spec.arity_in)
        seen_first_window = False
        try:
            while True:
                frame = await in_ring.get()
                if frame.end_of_stream:
                    for leftover in window.flush():
                        if not leftover.end_of_stream:
                            await out_ring.put(leftover)
                    await out_ring.put(frame)
                    return
                if self.cancelled:
                    return
                batch = window.feed(frame)
                if batch is None:
                    continue
                timer.start(frames=len(batch))
                produced: Sequence[Frame] = stage.process(batch)
                timer.stop()
                # An interpolating stage returns only the frames it invented.
                # The executor is what interleaves them with the source frames,
                # so the stage never has to know about stream ordering: for the
                # window (f_i, f_i+1) the output is f_i (first window only),
                # then the produced frames, then f_i+1.
                emitted: list[Frame] = []
                if spec.arity_in > 1:
                    if not seen_first_window:
                        emitted.append(batch[0])
                        seen_first_window = True
                    emitted.extend(produced)
                    emitted.append(batch[-1])
                else:
                    emitted.extend(produced)
                taps = self.taps.get(kind)
                for out_frame in emitted:
                    out_frame.require_device(kind.value)
                    if taps:
                        self._offer_to_taps(taps, out_frame)
                    await out_ring.put(out_frame)
                if kind is StageKind.RIFE:
                    self.stats.frames_enhanced += len(produced)
        except RingClosed:
            return

    async def _encode_task(self, upstream: str) -> None:
        stage = self.stages[StageKind.ENCODE]
        in_ring = self._ring(upstream, StageKind.ENCODE.value)
        timer = self.timers[StageKind.ENCODE.value]
        try:
            while True:
                frame = await in_ring.get()
                if frame.end_of_stream:
                    break
                if self.cancelled:
                    break
                if self.encode_filter is not None and not self.encode_filter(frame):
                    self.stats.frames_skipped += 1
                    continue
                timer.start()
                chunks = stage.process([frame])
                timer.stop()
                self.stats.frames_encoded += 1
                for chunk in chunks:
                    payload = chunk if isinstance(chunk, (bytes, bytearray)) else None
                    if payload is None:
                        continue
                    self.stats.bytes_out += len(payload)
                    # Awaiting the sink is what makes a slow client throttle
                    # the whole chain rather than fill VRAM.
                    await self.sink(bytes(payload))
                self.timers.drain()
                self.stats.ring_occupancy = self.rings.occupancies()
        except RingClosed:
            pass
        finally:
            tail = getattr(stage, "flush", None)
            if callable(tail):
                for chunk in tail() or []:
                    self.stats.bytes_out += len(chunk)
                    await self.sink(bytes(chunk))
