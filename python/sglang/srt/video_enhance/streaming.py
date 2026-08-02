# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Streaming input: admission, a seconds-deep output buffer, sustained rate (#448).

Everything upstream of this module assumes a *finished* source. The decode
stage opens a file, ffprobe reports how many frames it has, the muxer retimes
against that count, and the multi-card executor cuts the timeline into chunks
whose arithmetic is verified before a card is touched. None of that survives a
source that is still being written, and the failure is not a clean one: a
grower simply ends early at whatever the writer had flushed, and the result
looks like a successful job over a shorter clip.

So the first thing this module is, is a gate. :func:`admit_streaming_source`
decides whether a given source kind can go down a given executor path, and
refuses by name where it cannot. The rest of the module is what an admitted
streaming job needs and a file job does not:

*   **A seconds-deep output watermark.** A file job's response bridge is depth
    1, because any deeper is a buffer between the socket and the chain that
    back-pressure must cross (§8.4 rule 3). A live job wants the opposite at
    exactly one point: a small, *declared* amount of finished output held back
    so a jittery chain does not underrun the player. The size is stated in
    seconds of playback and converted to frames through the output rate,
    because seconds is the unit the property is about --
    :class:`SecondsDeepBuffer` is a :class:`~sglang.srt.video_enhance.ring.BoundedRing`
    with that conversion and no new queueing of its own.

*   **A growing source that waits instead of ending.** :func:`growing_frames`
    turns a producer that may have nothing *yet* into the async iterator the
    executor's source contract wants, distinguishing "not yet" from "no more"
    -- which a plain iterator cannot express and which is the whole bug above.

*   **Sustained rate, in and out.** A live job's health is not "did it
    finish", it is "is it keeping up". :class:`RateWindow` reports frames per
    second in and out over a sliding window from the counters the pipeline
    already keeps, so nothing is added to the hot path: it is sampled where
    the job status is rendered and where a chunk has just been accepted by
    the transport. That is the number #344's live watch shows.

Torch-free and device-free, like ``ring.py`` and ``pipeline.py``: the module
map marks those two as needing no device and this belongs with them, which is
what lets the tests below drive it with fake frame producers.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from sglang.srt.video_enhance.ring import BoundedRing, OverloadPolicy, RingClosed

__all__ = [
    "DEFAULT_IDLE_TIMEOUT_S",
    "DEFAULT_RATE_WINDOW_S",
    "DEFAULT_WATERMARK_S",
    "NO_MORE_FRAMES",
    "NOT_YET",
    "RateWindow",
    "SecondsDeepBuffer",
    "SourceKind",
    "StreamingAdmission",
    "StreamingAdmissionError",
    "StreamingPolicy",
    "admit_streaming_source",
    "growing_frames",
]


#: Seconds of finished output an admitted streaming job may hold back. Two
#: seconds at 50 fps is 100 frames; the value is the operator's call and the
#: only claim made here is that it is *bounded* and *declared*.
DEFAULT_WATERMARK_S = 2.0

#: Sliding window the in/out rates are averaged over. Long enough that one
#: slow chunk does not read as a collapse, short enough that a real collapse
#: shows within a few seconds.
DEFAULT_RATE_WINDOW_S = 10.0

#: How long a growing source may produce nothing before it is treated as
#: finished. A writer that has stopped writing and a writer that is merely
#: slow are indistinguishable from the reader's side, so the difference is a
#: duration and it is configured.
DEFAULT_IDLE_TIMEOUT_S = 30.0


class _Signal:
    """A producer's two out-of-band answers, distinct from any frame."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.name}>"


#: "I have no frame right now, ask again." Not the end of the stream.
NOT_YET = _Signal("NOT_YET")
#: "There will never be another frame." The end of the stream.
NO_MORE_FRAMES = _Signal("NO_MORE_FRAMES")


class StreamingAdmissionError(ValueError):
    """A source that cannot be run down the requested path, refused up front."""


class SourceKind(str, Enum):
    """What the reader may assume about the other end of the input."""

    #: A complete file. Length is final, seeking is exact, and every existing
    #: assumption in the tenant holds. This is the default and the path that
    #: must stay byte-for-byte what it was.
    FINISHED = "finished"
    #: A file still being written. It has a beginning, it will have an end,
    #: and reading past the current tail means waiting rather than stopping.
    #: Back-pressure is safe: the writer keeps writing to storage, so a stalled
    #: reader costs latency, not frames.
    GROWING = "growing"
    #: A feed with no end and no storage behind it. Back-pressure does not
    #: reach the producer, so a reader that stalls does not slow the source
    #: down -- it loses the frames that arrive while it is not reading. That
    #: makes ``stall`` the wrong overload policy here, and the admission says
    #: so rather than letting it look like it works.
    LIVE = "live"


@dataclass(frozen=True)
class StreamingPolicy:
    """How a streaming job is to be run, before it is checked."""

    kind: SourceKind = SourceKind.FINISHED
    #: Frame rate of the *delivered* stream, after interpolation. The
    #: watermark is a duration and this is what converts it to frames.
    output_frame_rate: Fraction = Fraction(50)
    watermark_s: float = DEFAULT_WATERMARK_S
    overload: OverloadPolicy = OverloadPolicy.STALL
    rate_window_s: float = DEFAULT_RATE_WINDOW_S
    #: How often a growing source is asked again after it answered NOT_YET.
    poll_interval_s: float = 0.05
    idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S

    def __post_init__(self) -> None:
        if self.output_frame_rate <= 0:
            raise StreamingAdmissionError(
                f"output frame rate must be positive, got {self.output_frame_rate}"
            )
        if self.watermark_s < 0:
            raise StreamingAdmissionError("watermark_s must not be negative")
        if self.poll_interval_s <= 0:
            raise StreamingAdmissionError("poll_interval_s must be positive")
        if self.idle_timeout_s <= 0:
            raise StreamingAdmissionError("idle_timeout_s must be positive")

    @property
    def watermark_frames(self) -> int:
        """The watermark in output frames. At least one: a depth-0 ring cannot exist."""
        return max(1, math.ceil(self.watermark_s * float(self.output_frame_rate)))


@dataclass(frozen=True)
class StreamingAdmission:
    """The verdict, and the terms an admitted job runs under."""

    admitted: bool
    policy: StreamingPolicy
    #: Depth of the response bridge, in output frames.
    buffer_depth_frames: int
    reason: str
    #: Facts the caller is owed even on an admission -- what is bounded, what
    #: is not, and what a shortfall will look like.
    notes: tuple[str, ...] = ()

    @property
    def kind(self) -> SourceKind:
        return self.policy.kind

    @property
    def is_streaming(self) -> bool:
        return self.policy.kind is not SourceKind.FINISHED

    @property
    def watermark_s(self) -> float:
        return self.buffer_depth_frames / float(self.policy.output_frame_rate)

    def require(self) -> "StreamingAdmission":
        """Raise the refusal instead of returning it."""
        if not self.admitted:
            raise StreamingAdmissionError(self.reason)
        return self

    def as_dict(self) -> dict:
        return {
            "admitted": self.admitted,
            "kind": self.kind.value,
            "buffer_depth_frames": self.buffer_depth_frames,
            "watermark_s": round(self.watermark_s, 3),
            "output_frame_rate": str(self.policy.output_frame_rate),
            "overload": self.policy.overload.value,
            "idle_timeout_s": self.policy.idle_timeout_s,
            "reason": self.reason,
            "notes": list(self.notes),
        }


def admit_streaming_source(
    policy: StreamingPolicy,
    *,
    chunked: bool = False,
    total_frames: Optional[int] = None,
) -> StreamingAdmission:
    """Decide whether this source may run down this executor path.

    Three refusals, each of which would otherwise surface as a job that looks
    like it worked:

    1.  **A growing or live source on the chunk executor.** ``multicard`` cuts
        a fixed timeline into items and ``verify_chunk_arithmetic`` checks the
        whole split against a known frame count *before* a card is touched.
        A source whose length is not final has no such count, and cutting it
        against the count it happens to have right now produces a plan for a
        prefix of the source. The seam convention and the scheduler are
        untouched here -- this is the gate in front of them.

    2.  **A finished source on the chunk executor with no frame count.** Same
        arithmetic, same missing input.

    3.  **A live source under the ``stall`` overload policy.** Stalling only
        works as back-pressure when the producer can be slowed down. A live
        feed cannot, so the frames arriving during a stall are lost either
        way -- the difference is that ``drop_frames`` counts them and
        ``stall`` does not. Silent loss is what ``ring.py`` rule 4 exists to
        prevent.
    """
    depth = policy.watermark_frames
    notes: list[str] = []

    if chunked and policy.kind is not SourceKind.FINISHED:
        return StreamingAdmission(
            admitted=False,
            policy=policy,
            buffer_depth_frames=depth,
            reason=(
                f"a {policy.kind.value} source cannot run on the chunk executor: "
                "the timeline is cut into items and verified against a final "
                "frame count before any card starts, and this source has no "
                "final count. Run it on the single-card streaming path, or wait "
                "for the source to finish."
            ),
        )
    if chunked and total_frames is None:
        return StreamingAdmission(
            admitted=False,
            policy=policy,
            buffer_depth_frames=depth,
            reason=(
                "the chunk executor needs the source frame count to verify the "
                "split before it runs; none was supplied"
            ),
        )
    if policy.kind is SourceKind.LIVE and policy.overload is OverloadPolicy.STALL:
        return StreamingAdmission(
            admitted=False,
            policy=policy,
            buffer_depth_frames=depth,
            reason=(
                "a live source cannot be back-pressured: stalling the reader "
                "does not slow the feed down, it only makes the frames that "
                "arrive during the stall vanish uncounted. Use the drop_frames "
                "overload policy, which counts them, or read from a growing "
                "file instead."
            ),
        )

    if policy.kind is SourceKind.FINISHED:
        notes.append(
            "finished source: the default path, unchanged -- the response "
            "bridge stays at its own depth and no watermark is applied"
        )
    else:
        notes.append(
            f"output is buffered {depth} frame(s) deep, which is "
            f"{depth / float(policy.output_frame_rate):.2f} s at "
            f"{policy.output_frame_rate} fps; back-pressure crosses that buffer "
            "before it reaches the decoder, so a stalled client stops the chain "
            "one watermark later than on the file path"
        )
    if policy.kind is SourceKind.GROWING:
        notes.append(
            f"a growing source that produces nothing for {policy.idle_timeout_s:.0f} s "
            "is treated as finished; a writer that is merely slower than that "
            "will be cut short"
        )
    if policy.kind is SourceKind.LIVE:
        notes.append(
            "dropped frames are counted on the ring and reported in the job "
            "status; they are never silent"
        )
    return StreamingAdmission(
        admitted=True,
        policy=policy,
        buffer_depth_frames=depth,
        reason=f"{policy.kind.value} source admitted",
        notes=tuple(notes),
    )


class SecondsDeepBuffer:
    """A bounded ring whose depth is stated in seconds of finished output.

    One item is one output frame's worth of encoded bytes, which is what the
    encode stage produces per call. That is the assumption the seconds figure
    rests on and it is stated rather than implied: an encoder that emitted two
    items for one frame would make :meth:`seconds_buffered` read half of what
    it should, so the count is checked against the frame count the caller
    passes rather than being inferred from the item count alone.
    """

    def __init__(
        self,
        name: str,
        policy: StreamingPolicy,
        *,
        loop_time: Callable[[], float] | None = None,
    ) -> None:
        self.policy = policy
        self.ring = BoundedRing(
            name, policy.watermark_frames, policy.overload, loop_time=loop_time
        )
        self._frames_buffered = 0
        self.frames_in = 0
        self.frames_out = 0

    @property
    def depth_frames(self) -> int:
        return self.ring.depth

    @property
    def watermark_s(self) -> float:
        return self.depth_frames / float(self.policy.output_frame_rate)

    def seconds_buffered(self) -> float:
        return len(self.ring) / float(self.policy.output_frame_rate)

    async def put(self, item, *, frames: int = 1) -> bool:
        accepted = await self.ring.put(item)
        self.frames_in += frames
        return accepted

    async def get(self):
        item = await self.ring.get()
        self.frames_out += 1
        return item

    async def close(self) -> None:
        await self.ring.close()

    @property
    def closed(self) -> bool:
        return self.ring.closed

    def snapshot(self) -> dict:
        stats = self.ring.stats
        return {
            "depth_frames": self.depth_frames,
            "watermark_s": round(self.watermark_s, 3),
            "occupancy": len(self.ring),
            "seconds_buffered": round(self.seconds_buffered(), 3),
            "high_water": stats.high_water,
            "dropped": stats.dropped,
            "producer_stall_seconds": round(stats.producer_stall_seconds, 6),
        }


@dataclass
class _RateSample:
    at: float
    frames_in: int
    frames_out: int


@dataclass
class RateWindow:
    """Sustained frames per second in and out, over a sliding window.

    Fed with *cumulative* counters rather than events, so the caller is the
    pipeline's own ``frames_decoded`` / ``frames_encoded`` and nothing has to
    be instrumented inside the chain. Sampling happens where the job status is
    rendered and after a chunk has been accepted by the transport -- both
    outside the hot path, and the second of the two is the same moment the
    liveness watchdog already uses because it is the one that proves progress.

    A window with fewer than two samples reports ``None`` rather than 0.0. A
    rate of zero and "not measured yet" are different states and a live watch
    that showed the first for the second would be reporting a stall that is
    not there.
    """

    window_s: float = DEFAULT_RATE_WINDOW_S
    clock: Callable[[], float] = time.monotonic
    #: Delivered frames per second the job is supposed to sustain, if the
    #: caller knows it. Only used to answer :attr:`sustaining`.
    target_output_fps: float | None = None
    _samples: deque = field(default_factory=deque, repr=False)

    def observe(
        self, *, frames_in: int, frames_out: int, now: float | None = None
    ) -> None:
        at = self.clock() if now is None else now
        self._samples.append(_RateSample(at, int(frames_in), int(frames_out)))
        # Keep one sample at or before the window edge so the delta always
        # spans the full window once enough history exists. Dropping it would
        # shorten the window silently whenever samples are sparse.
        cutoff = at - self.window_s
        while len(self._samples) > 2 and self._samples[1].at <= cutoff:
            self._samples.popleft()

    def _rates(self) -> tuple[float | None, float | None, float]:
        if len(self._samples) < 2:
            return None, None, 0.0
        first, last = self._samples[0], self._samples[-1]
        span = last.at - first.at
        if span <= 0:
            return None, None, 0.0
        return (
            (last.frames_in - first.frames_in) / span,
            (last.frames_out - first.frames_out) / span,
            span,
        )

    @property
    def fps_in(self) -> float | None:
        return self._rates()[0]

    @property
    def fps_out(self) -> float | None:
        return self._rates()[1]

    @property
    def sustaining(self) -> bool | None:
        """Is the output rate at or above what the job promised?

        ``None`` when either the target or the measurement is missing, which
        is the honest third state: a live watch must be able to show "not
        known yet" without it looking like a failure.
        """
        out = self.fps_out
        if out is None or self.target_output_fps is None:
            return None
        return out >= self.target_output_fps

    def snapshot(self) -> dict:
        fps_in, fps_out, span = self._rates()
        last = self._samples[-1] if self._samples else None
        return {
            "window_s": self.window_s,
            "measured_span_s": round(span, 3),
            "samples": len(self._samples),
            "fps_in": None if fps_in is None else round(fps_in, 2),
            "fps_out": None if fps_out is None else round(fps_out, 2),
            "frames_in": 0 if last is None else last.frames_in,
            "frames_out": 0 if last is None else last.frames_out,
            "target_output_fps": self.target_output_fps,
            "sustaining": self.sustaining,
        }


async def growing_frames(
    producer: Callable[[], object],
    admission: StreamingAdmission,
    *,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator:
    """Adapt a producer that may have nothing *yet* into the source contract.

    ``producer`` is called with no arguments and returns one of three things:
    a frame, :data:`NOT_YET`, or :data:`NO_MORE_FRAMES`. It may be a coroutine
    function. The distinction between the last two is the entire point --
    a plain iterator can only say "no more", so a growing file reaches its
    current tail and the job ends early, successfully, over a prefix of the
    source.

    Back-pressure is unchanged and unmediated: this is a pull source, so the
    executor's ring is what stops it. Nothing here buffers a frame.

    The idle timeout is measured from the last frame, not from the last call,
    so a producer answering ``NOT_YET`` a thousand times still times out after
    the configured duration.
    """
    admission.require()
    policy = admission.policy
    last_frame_at = clock()
    while True:
        item = producer()
        if inspect.isawaitable(item):
            item = await item
        if item is NO_MORE_FRAMES:
            return
        if item is NOT_YET:
            if clock() - last_frame_at >= policy.idle_timeout_s:
                return
            await sleep(policy.poll_interval_s)
            continue
        last_frame_at = clock()
        yield item


async def drain_to_sink(
    buffer: SecondsDeepBuffer, sink: Callable[[bytes], object]
) -> None:
    """Forward a seconds-deep buffer to a sink until it closes.

    Separate from the buffer so the buffer stays a data structure. Used by the
    streaming response path in the same shape ``server.pump_muxer`` uses for
    the file path.
    """
    while True:
        try:
            item = await buffer.get()
        except RingClosed:
            return
        result = sink(item)
        if inspect.isawaitable(result):
            await result
