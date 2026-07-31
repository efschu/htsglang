# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Live preview taps: watching a job's input and output while it runs (§8.1).

DESIGN #333 §8.1, and the user directive recorded in TASK_333 §8.1: while a
video is being enhanced, the frontend must be able to watch **both** the
incoming stream and the outgoing one. Two low-bitrate side encodes at preview
resolution, served as their own endpoints.

The hard rule, and the whole reason this module is shaped the way it is:

    **A tap must never stall the main chain.** A slow preview viewer costs
    preview frames, never pipeline throughput.

That is a structural property here, not a tuning goal:

1.  :meth:`PreviewTap.offer` is an ordinary synchronous method with no
    ``await`` anywhere in it. The main chain therefore cannot be suspended by
    a tap even for one event-loop turn, whatever the tap is doing. This is
    why the ingress buffer is a plain list with a hand-written drop rule
    rather than :class:`~sglang.srt.video_enhance.ring.BoundedRing`, whose
    ``put`` is a coroutine by design -- even its drop path takes an
    ``asyncio.Condition``, which is an await, which is a place a stall can
    live.
2.  The ingress buffer **drops the oldest frame** when it is full. A preview
    frame whose moment has passed is worthless; the newest one is the one a
    viewer wants.
3.  Back-pressure exists, but it is confined to the preview lane. A viewer
    who stops reading fills the encoded-byte ring, which stalls the preview
    encoder task, which stops draining the ingress buffer, which then drops.
    The chain of consequences terminates at the drop counter and never
    reaches the pipeline.

**Dropping frames, not bytes.** The byte ring stalls rather than drops
because an H.264 elementary stream cannot survive having arbitrary byte
ranges removed from the middle. Decimation has to happen before the encoder
or not at all, so the only thing that is ever discarded is a whole frame that
has not been encoded yet, and the stream a viewer receives is always
well-formed.

**What a tap costs the main chain is not zero, and is measured rather than
asserted.** The downscale and the preview encode run on the same device as
the chain. They do not block it -- they compete with it. The A/B measurement
against a noise floor is `scripts/video_enhance/preview_tap_bench.py`; the
number it produces belongs in TASK_333, whatever it says. ``fps_divisor`` is
the lever that trades preview smoothness for that cost.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from fractions import Fraction
from typing import AsyncIterator

from sglang.srt.video_enhance.chain import StageKind
from sglang.srt.video_enhance.frame_math import Resolution
from sglang.srt.video_enhance.frames import Frame
from sglang.srt.video_enhance.ring import BoundedRing, OverloadPolicy, RingClosed

logger = logging.getLogger(__name__)

#: Default preview geometry. Small enough that the downscale and encode are a
#: rounding error next to a 4K chain, large enough to see what is happening.
DEFAULT_PREVIEW_WIDTH = 480

#: Default preview bitrate. A preview is a monitoring surface, not a
#: deliverable, and at 480x270 this is visually adequate.
DEFAULT_PREVIEW_BITRATE = 1_500_000

#: Ingress depth. Two, not one: one slot in the encoder's hand and one being
#: filled, so a tap that is keeping up never drops, and a tap that is not
#: drops immediately rather than accumulating a backlog of stale frames.
DEFAULT_INGRESS_DEPTH = 2

#: Encoded-byte ring depth, in segments. The only buffer between the preview
#: encoder and an HTTP client.
DEFAULT_BYTE_DEPTH = 4


class PreviewError(RuntimeError):
    """A preview lane that cannot be built. Never raised into the main chain."""


@dataclass
class PreviewStats:
    """What a tap did, in the only terms that matter for the hard rule."""

    name: str
    #: Frames the main chain handed to the tap.
    offered: int = 0
    #: Frames dropped because the ingress buffer was full -- the cost a slow
    #: viewer pays, and the number that must be non-zero rather than the
    #: pipeline slowing down.
    dropped_full: int = 0
    #: Frames skipped by ``fps_divisor`` before they ever reached the buffer.
    decimated: int = 0
    #: Frames the preview encoder actually encoded.
    encoded: int = 0
    bytes_out: int = 0
    #: Set if the preview lane failed. A dead preview must not fail the job,
    #: so the error is recorded here and the tap goes quiet.
    error: str | None = None

    @property
    def delivered_fraction(self) -> float:
        return self.encoded / self.offered if self.offered else 0.0

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "offered": self.offered,
            "dropped_full": self.dropped_full,
            "decimated": self.decimated,
            "encoded": self.encoded,
            "bytes_out": self.bytes_out,
            "delivered_fraction": round(self.delivered_fraction, 4),
            "error": self.error,
        }


@dataclass(frozen=True)
class PreviewConfig:
    """Preview geometry and cost knobs.

    ``fps_divisor`` is the honest lever on what a tap costs the chain: at 2
    the tap does half the downscales and half the encodes. It decimates
    *before* the ingress buffer, so a decimated frame costs one modulo and
    nothing else.
    """

    width: int = DEFAULT_PREVIEW_WIDTH
    bitrate: int = DEFAULT_PREVIEW_BITRATE
    fps_divisor: int = 1
    ingress_depth: int = DEFAULT_INGRESS_DEPTH
    byte_depth: int = DEFAULT_BYTE_DEPTH
    codec: str = "h264"
    #: ffmpeg, not "auto". "auto" selects PyNvVideoCodec, which on this rig
    #: rejects a device tensor with "incorrect usage of CPU input buffer" --
    #: the defect TASK_333 §9.5 records as open and undiagnosed. The main
    #: chain has always pinned ffmpeg for that reason; a preview built on
    #: "auto" inherited the bug, and because a failing lane is deliberately
    #: swallowed it presented as a preview that delivered zero bytes while
    #: the job ran perfectly. Pin the backend that works until §9.5 closes.
    encode_backend: str = "ffmpeg"

    def __post_init__(self) -> None:
        if self.width < 16:
            raise PreviewError(f"preview width {self.width} is below 16")
        if self.fps_divisor < 1:
            raise PreviewError("fps_divisor must be at least 1")
        if self.ingress_depth < 1:
            raise PreviewError("ingress_depth must be at least 1")
        if self.byte_depth < 1:
            raise PreviewError("byte_depth must be at least 1")

    def preview_resolution(self, source: Resolution) -> Resolution:
        """Preview geometry for a source, preserving aspect, even dimensions.

        Odd dimensions are not a style preference: NV12 subsamples chroma 2x2,
        so an odd width or height has no representation and the encoder either
        refuses or silently pads.
        """
        width = min(self.width, source.width)
        height = max(2, round(source.height * width / source.width))
        return Resolution(width - (width % 2), height - (height % 2))


class PreviewTap:
    """The ingress side of one preview lane. Synchronous, lossy, unblockable.

    :meth:`offer` is what the pipeline calls. It contains no ``await``, takes
    no lock, and returns in constant time, so no state of the preview lane --
    a stalled viewer, a wedged encoder, a full ring -- can suspend the stage
    that called it.
    """

    def __init__(self, name: str, config: PreviewConfig | None = None) -> None:
        self.name = name
        self.config = config or PreviewConfig()
        self.stats = PreviewStats(name=name)
        self._items: list[Frame] = []
        self._seen = 0
        self._closed = False
        # Created lazily: a tap may be constructed off the event loop (the
        # tenant builds one before the job starts) and asyncio.Event binds to
        # the running loop on first wait in older semantics.
        self._wake: asyncio.Event | None = None

    def _ensure_wake(self) -> asyncio.Event:
        if self._wake is None:
            self._wake = asyncio.Event()
        return self._wake

    @property
    def closed(self) -> bool:
        return self._closed

    def offer(self, frame: Frame) -> bool:
        """Hand a frame to the preview lane. Never blocks, never raises.

        Returns True if the frame was buffered, False if it was decimated,
        dropped, or the lane is closed. The main chain ignores the return
        value; it is here for the tests, which have to be able to tell the
        three apart.
        """
        if self._closed or frame.end_of_stream:
            return False
        self.stats.offered += 1
        self._seen += 1
        if self.config.fps_divisor > 1 and (self._seen - 1) % self.config.fps_divisor:
            self.stats.decimated += 1
            return False
        if len(self._items) >= self.config.ingress_depth:
            # Drop the oldest. A preview frame whose moment has passed is
            # worth less than the one arriving now, and the alternative --
            # refusing the new frame -- would show the viewer a frozen image
            # that is getting staler rather than a jerky one that is current.
            self._items.pop(0)
            self.stats.dropped_full += 1
        self._items.append(frame)
        if self._wake is not None:
            self._wake.set()
        return True

    async def take(self) -> Frame | None:
        """The encoder side. ``None`` once the tap is closed and drained."""
        wake = self._ensure_wake()
        while True:
            if self._items:
                return self._items.pop(0)
            if self._closed:
                return None
            wake.clear()
            await wake.wait()

    def close(self) -> None:
        self._closed = True
        self._items.clear()
        if self._wake is not None:
            self._wake.set()


class PreviewLane:
    """One tap, its encoder task, and the bytes a viewer reads.

    The encoder is a task rather than inline work precisely so that the
    downscale and the encode happen on the preview lane's own schedule. It
    competes with the main chain for the device -- which is measurable and is
    measured -- but it cannot suspend it.
    """

    def __init__(
        self,
        *,
        name: str,
        source: Resolution,
        fps: Fraction | int,
        dtype: str = "fp16",
        device_id: int = 0,
        config: PreviewConfig | None = None,
    ) -> None:
        self.config = config or PreviewConfig()
        self.name = name
        self.source = source
        self.target = self.config.preview_resolution(source)
        self.dtype = dtype
        self.device_id = device_id
        # The preview stream's own frame rate. Decimation changes it, and a
        # decoder told the wrong rate plays the preview at the wrong speed.
        self.fps = Fraction(fps) / self.config.fps_divisor
        self.tap = PreviewTap(name, self.config)
        self.bytes_ring = BoundedRing(
            f"preview-{name}", self.config.byte_depth, OverloadPolicy.STALL
        )
        self._task: asyncio.Task | None = None
        self._stages: dict = {}

    @property
    def stats(self) -> PreviewStats:
        return self.tap.stats

    def offer(self, frame: Frame) -> bool:
        """Delegate to the tap, so a lane *is* a tap as far as the executor sees.

        ``PreviewLanes.by_stage`` holds lanes, and the executor calls
        ``offer`` on whatever it is given. Without this method the executor
        raised ``AttributeError`` on every single frame -- and because a tap
        that raises is deliberately swallowed so it cannot fail the job, the
        result was a preview that silently delivered nothing while the job
        looked perfectly healthy. It was found by a throughput measurement
        that reported ``offered: 0``, not by the tests, which had exercised
        ``PreviewTap`` directly and never the wiring around it.
        """
        return self.tap.offer(frame)

    def _build_stages(self) -> None:
        """Colour conversion and encoder for the preview, built on first use."""
        from sglang.srt.video_enhance import codec

        self._stages["to_yuv"] = codec.ColorToYuvStage(dtype=self.dtype)
        self._stages["encode"] = codec.EncodeStage(
            self.target,
            fps=self.fps,
            codec=self.config.codec,
            device_id=self.device_id,
            backend=self.config.encode_backend,
            bitrate=self.config.bitrate,
        )

    def _downscale(self, frame: Frame) -> Frame:
        """Bilinear, not Lanczos-3.

        The main chain's resize is Lanczos because its output is the
        deliverable. A preview is a monitoring surface at a fifth of the
        width, where the difference is invisible and the cost is not: bilinear
        is one fused kernel against Lanczos's two separable passes with six
        taps each.
        """
        import torch.nn.functional as F  # noqa: N812 - torch's own convention

        data = frame.data
        if data.dim() != 4:
            raise PreviewError(f"preview expected NCHW, got {tuple(data.shape)}")
        small = F.interpolate(
            data.float(),
            size=(self.target.height, self.target.width),
            mode="bilinear",
            align_corners=False,
        ).to(data.dtype)
        return frame.with_data(small, resolution=self.target)

    async def _encoder_task(self) -> None:
        try:
            self._build_stages()
            while True:
                frame = await self.tap.take()
                if frame is None:
                    break
                small = self._downscale(frame)
                (yuv,) = self._stages["to_yuv"].process([small])
                chunks = self._stages["encode"].process([yuv])
                self.stats.encoded += 1
                for chunk in chunks:
                    if not isinstance(chunk, (bytes, bytearray)):
                        continue
                    self.stats.bytes_out += len(chunk)
                    # STALL, deliberately. A viewer who stops reading fills
                    # this ring and stalls *this* task, which stops draining
                    # the tap, which then drops on ingress. The stall is
                    # confined to the preview lane by construction: nothing
                    # upstream of here is waiting on this task.
                    await self.bytes_ring.put(bytes(chunk))
        except RingClosed:
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a dead preview must not fail the job
            # The job is the product; the preview is a convenience. An
            # exception here is recorded and the lane goes quiet, because
            # taking a running enhance job down because nobody could watch it
            # would be the wrong trade in every case.
            self.stats.error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "preview lane %s failed and was stopped; the job continues: %s",
                self.name,
                exc,
            )
        finally:
            await self.bytes_ring.close()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._encoder_task())

    async def close(self) -> None:
        self.tap.close()
        await self.bytes_ring.close()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        for stage in self._stages.values():
            closer = getattr(stage, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - one bad stage must not mask
                    continue
        self._stages.clear()

    async def stream(self) -> AsyncIterator[bytes]:
        """The HTTP body: encoded preview segments as they are produced."""
        try:
            while True:
                yield await self.bytes_ring.get()
        except RingClosed:
            return


@dataclass
class PreviewLanes:
    """The two lanes of one job, and where each attaches to the chain.

    Attachment is by stage kind rather than by position because the chain is
    request-level configuration: a 4K RIFE-only request has no SR and no
    resize, and a tap pinned to "the third stage" would tap a different thing
    per request.

    *   **input** is the output of ``color_to_rgb`` -- the source frame, in the
        chain's own working format, before anything has enhanced it.
    *   **output** is the input of ``color_to_yuv`` -- the last RGB in the
        chain, which is the enhanced frame in its final geometry.

    In a degenerate chain those are the same stage, and then one frame feeds
    both taps, which is correct: with nothing between them the input and the
    output *are* the same picture.
    """

    input_lane: PreviewLane | None = None
    output_lane: PreviewLane | None = None
    #: ``StageKind`` -> the lanes fed by that stage's output.
    by_stage: dict = field(default_factory=dict)

    def lanes(self) -> list[PreviewLane]:
        return [
            lane for lane in (self.input_lane, self.output_lane) if lane is not None
        ]

    def start(self) -> None:
        for lane in self.lanes():
            lane.start()

    async def close(self) -> None:
        for lane in self.lanes():
            await lane.close()

    def snapshot(self) -> dict:
        return {
            which: lane.stats.snapshot()
            for which, lane in (
                ("input", self.input_lane),
                ("output", self.output_lane),
            )
            if lane is not None
        }


def output_tap_stage(chain) -> StageKind | None:
    """The stage whose output is the final enhanced RGB frame.

    The last stage before ``color_to_yuv``. Returns None for a chain that has
    no RGB working section at all, which is a chain with nothing to preview on
    the output side.
    """
    kinds = list(chain.kinds)
    if StageKind.COLOR_TO_YUV not in kinds:
        return None
    index = kinds.index(StageKind.COLOR_TO_YUV)
    if index == 0:
        return None
    return kinds[index - 1]


def build_preview_lanes(
    chain,
    *,
    fps: Fraction | int,
    dtype: str = "fp16",
    device_id: int = 0,
    config: PreviewConfig | None = None,
    want_input: bool = True,
    want_output: bool = True,
) -> PreviewLanes:
    """Wire the two lanes onto a resolved chain.

    Pure with respect to the device: it builds tap objects and decides where
    they attach. No CUDA context, no encoder session -- those are the encoder
    task's, created when the lane starts.
    """
    lanes = PreviewLanes()
    kinds = list(chain.kinds)
    if StageKind.COLOR_TO_RGB not in kinds:
        # An NV12-only chain has no RGB frame to downscale with the code here.
        # Refusing to build the lane is better than building one that would
        # fail on its first frame.
        return lanes

    if want_input:
        spec = chain.stage(StageKind.COLOR_TO_RGB)
        lanes.input_lane = PreviewLane(
            name="input",
            source=spec.out_res,
            fps=fps,
            dtype=dtype,
            device_id=device_id,
            config=config,
        )
        lanes.by_stage.setdefault(StageKind.COLOR_TO_RGB, []).append(lanes.input_lane)

    out_kind = output_tap_stage(chain)
    if want_output and out_kind is not None:
        spec = chain.stage(out_kind)
        lanes.output_lane = PreviewLane(
            name="output",
            source=spec.out_res,
            # The output side carries the interpolated frames too, so its rate
            # is the retimed one. Declaring the source rate here would make a
            # 2x-interpolated preview play at half speed.
            fps=fps,
            dtype=dtype,
            device_id=device_id,
            config=config,
        )
        lanes.by_stage.setdefault(out_kind, []).append(lanes.output_lane)
    return lanes


__all__ = [
    "DEFAULT_BYTE_DEPTH",
    "DEFAULT_INGRESS_DEPTH",
    "DEFAULT_PREVIEW_BITRATE",
    "DEFAULT_PREVIEW_WIDTH",
    "PreviewConfig",
    "PreviewError",
    "PreviewLane",
    "PreviewLanes",
    "PreviewStats",
    "PreviewTap",
    "build_preview_lanes",
    "output_tap_stage",
]
