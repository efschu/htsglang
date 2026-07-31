# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The chunked-HTTP surface of the Class-3 video-enhance tenant.

Four endpoints, per DESIGN #333 §8.5:

======================================  ====================================
``POST   /v1/video/enhance``            enhance a stream, chunked response
``GET    /v1/video/enhance/{id}``       progress, per-stage ms/frame, rings
``DELETE /v1/video/enhance/{id}``       cancel, releasing rings and contexts
``GET    /v1/video/engines``            what is cached, what needs a build
======================================  ====================================

The last one is operational, not cosmetic: an engine build is minutes long
and exclusive on the card, so a request implying one has to say so before it
starts rather than appearing to hang.

Back-pressure is structural rather than configured. The response body is an
async generator; Starlette awaits each yield until the transport has accepted
it, so a client whose TCP window is full blocks the generator, which stops
draining the bridge ring, which stops the encode stage, which fills the
upstream rings and stalls the decoder within one ring depth. Nothing in this
file buffers a frame outside a bounded ring.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from fractions import Fraction
from typing import AsyncIterator

from sglang.srt.video_enhance.chain import ChainError, ChainRequest
from sglang.srt.video_enhance.engine_cache import EngineCache
from sglang.srt.video_enhance.frame_math import Resolution
from sglang.srt.video_enhance.mux import (
    MediaInfo,
    MuxError,
    StreamRemuxer,
    TrackSelection,
    build_remux_command,
    describe_selection,
    expected_frame_count,
    probe,
    retimed_rate,
)
from sglang.srt.video_enhance.pipeline import PipelineExecutor
from sglang.srt.video_enhance.ring import BoundedRing, OverloadPolicy, RingClosed
from sglang.srt.video_enhance.tenant import (
    PlannedJob,
    TenantConfig,
    TenantConfigError,
    build_stages,
    plan_job,
)

#: Depth of the bridge between the encode stage and the ASGI send coroutine.
#: One, deliberately: any larger value is a buffer between the socket and the
#: chain that back-pressure would have to traverse before reaching the
#: decoder, which is exactly what §8.4 rule 3 forbids.
RESPONSE_BRIDGE_DEPTH = 1

#: HTTP media type to ffmpeg muxer name. The response advertises the media
#: type; ffmpeg is told the muxer.
_CONTAINER_BY_MEDIA_TYPE: dict[str, str] = {
    "video/mp4": "mp4",
    "video/x-matroska": "matroska",
    "video/webm": "webm",
    "video/mp2t": "mpegts",
}

#: Elementary-stream format the encode stage emits, per requested codec. The
#: muxer needs it because a raw elementary stream carries no container.
_ELEMENTARY_FORMAT: dict[str, str] = {
    "h264": "h264",
    "hevc": "hevc",
    "av1": "obu",
}


@dataclass
class EnhanceRequestBody:
    """Parsed request body. Kept as a plain dataclass so the planning path is
    importable and testable without FastAPI."""

    source_url: str
    source_width: int
    source_height: int
    target: str = "3840x2160"
    fps_multiplier: int = 1
    dtype: str = "fp16"
    enable_sr: bool = True
    sr_scale: int = 4
    enable_resize: bool = True
    rife_scale: float = 1.0
    rife_version: str = "4.6"
    streams_in_flight: int = 2
    overload_policy: str = OverloadPolicy.STALL.value
    container: str = "video/mp4"
    video_codec: str = "h264"
    #: Which video track to enhance. None selects the default (or first) video
    #: track. Every other track is copied through bit-identically unless
    #: explicitly excluded below.
    enhance_video_index: int | None = None
    passthrough_audio: bool = True
    passthrough_subtitles: bool = True
    passthrough_other_video: bool = True
    passthrough_data: bool = True
    #: Source frame rate as "num/den". Probed when absent.
    source_frame_rate: str | None = None

    def track_selection(self) -> TrackSelection:
        return TrackSelection(
            enhance_video_index=self.enhance_video_index,
            passthrough_audio=self.passthrough_audio,
            passthrough_subtitles=self.passthrough_subtitles,
            passthrough_other_video=self.passthrough_other_video,
            passthrough_data=self.passthrough_data,
        )

    def container_format(self) -> str:
        """ffmpeg muxer name for the negotiated media type."""
        return _CONTAINER_BY_MEDIA_TYPE.get(self.container, "mp4")

    def to_chain_request(self) -> ChainRequest:
        return ChainRequest(
            source=Resolution(self.source_width, self.source_height),
            target=Resolution.parse(self.target),
            fps_multiplier=self.fps_multiplier,
            dtype=self.dtype,
            enable_sr=self.enable_sr,
            sr_scale=self.sr_scale,
            enable_resize=self.enable_resize,
            rife_scale=self.rife_scale,
            rife_version=self.rife_version,
            streams_in_flight=self.streams_in_flight,
        )

    def policy(self) -> OverloadPolicy:
        return OverloadPolicy(self.overload_policy)


@dataclass
class Job:
    job_id: str
    executor: PipelineExecutor
    planned: PlannedJob
    task: asyncio.Task | None = None
    created_at: float = 0.0
    source_rate: Fraction | None = None


class VideoEnhanceService:
    """Job registry and the request lifecycle, independent of the web framework.

    Keeping the framework out of this class is what makes the back-pressure
    test hermetic: the test drives ``stream_response`` with a slow consumer
    and asserts on ring occupancy, with no socket involved.
    """

    def __init__(self, config: TenantConfig, *, device_id: int = 0) -> None:
        self.config = config
        self.device_id = device_id
        self.jobs: dict[str, Job] = {}
        self.cache = EngineCache(config.engine_cache_dir)

    # -- planning ---------------------------------------------------------
    def plan(self, body: EnhanceRequestBody) -> PlannedJob:
        return plan_job(self.config, body.to_chain_request())

    def track_plan(
        self, body: EnhanceRequestBody, info: MediaInfo | None = None
    ) -> dict:
        """What happens to every track of the source, before anything runs."""
        info = info if info is not None else probe(body.source_url)
        return describe_selection(info, body.track_selection())

    # -- execution --------------------------------------------------------
    async def stream_response(
        self,
        body: EnhanceRequestBody,
        *,
        source_factory=None,
        stage_factory=None,
        media_info: MediaInfo | None = None,
        remux: bool = True,
    ) -> AsyncIterator[bytes]:
        """Run one job and yield muxed chunks as they are produced.

        The encoded elementary stream is fed to a remuxer that copies every
        kept non-video track through bit-identically and retimes the enhanced
        video to the interpolated rate. Back-pressure survives the extra
        process: the remuxer's ``feed`` awaits a drain, and its stdout is only
        read when the bounded response bridge has room.
        """
        planned = self.plan(body)
        job_id = uuid.uuid4().hex[:16]
        bridge = BoundedRing(
            f"muxer->socket:{job_id}", RESPONSE_BRIDGE_DEPTH, body.policy()
        )

        remuxer, source_rate = (
            self._make_remuxer(body, media_info) if remux else (None, None)
        )

        if remuxer is None:

            async def sink(payload: bytes) -> None:
                await bridge.put(payload)

        else:

            async def sink(payload: bytes) -> None:
                await remuxer.feed(payload)

        stages = (
            stage_factory(planned.chain)
            if stage_factory is not None
            else build_stages(self.config, planned.chain, device_id=self.device_id)
        )
        if source_factory is not None:
            source = source_factory(planned.chain)
        else:
            source = stages[planned.chain.stages[0].kind].frames(body.source_url)  # type: ignore[index]

        executor = PipelineExecutor(
            job_id=job_id,
            chain=planned.chain,
            stages=stages,
            source=source,
            sink=sink,
            ring_depth=planned.ring_depth,
            policy=body.policy(),
        )
        job = Job(
            job_id=job_id, executor=executor, planned=planned, created_at=time.time()
        )
        self.jobs[job_id] = job

        async def pump_muxer() -> None:
            assert remuxer is not None
            async for chunk in remuxer.read_chunks():
                await bridge.put(chunk)

        async def drive() -> None:
            reader = None
            try:
                if remuxer is not None:
                    await remuxer.start()
                    reader = asyncio.create_task(pump_muxer())
                await executor.run()
                if remuxer is not None:
                    await remuxer.close_input()
                    if reader is not None:
                        await reader
                    await remuxer.wait()
            finally:
                if reader is not None and not reader.done():
                    reader.cancel()
                if remuxer is not None:
                    await remuxer.terminate()
                await bridge.close()

        job.task = asyncio.create_task(drive())
        job.source_rate = source_rate
        try:
            while True:
                try:
                    chunk = await bridge.get()
                except RingClosed:
                    break
                yield chunk
        finally:
            if not job.task.done():
                executor.cancel()
                await bridge.close()
                try:
                    await asyncio.wait_for(job.task, timeout=30)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    job.task.cancel()
            self._close_stages(stages)
        # The trailer carries the drop count. A dropped frame is never silent.
        job.executor.stats.dropped = executor.rings.dropped + bridge.stats.dropped

    def _make_remuxer(
        self, body: EnhanceRequestBody, media_info: MediaInfo | None
    ) -> tuple[StreamRemuxer | None, Fraction | None]:
        """Build the remuxer for this request, or None when muxing is skipped.

        Muxing is skipped only when the source cannot be probed. A source with
        no side tracks still goes through the muxer, because the container it
        needs on the wire is the muxer's job either way.
        """
        try:
            info = media_info if media_info is not None else probe(body.source_url)
        except MuxError:
            return None, None

        selection = body.track_selection()
        enhanced_index = selection.resolve_video_index(info)
        rate = info.track(enhanced_index).frame_rate()
        if body.source_frame_rate:
            num, _, den = body.source_frame_rate.partition("/")
            rate = Fraction(int(num), int(den or 1))
        if rate is None:
            raise MuxError(
                "source frame rate is unknown and no source_frame_rate was given; "
                "retiming after interpolation needs it"
            )
        output_rate = retimed_rate(rate, body.fps_multiplier)
        command = build_remux_command(
            source_url=body.source_url,
            info=info,
            selection=selection,
            enhanced_codec=_ELEMENTARY_FORMAT.get(body.video_codec, "h264"),
            output_rate=output_rate,
            container=body.container_format(),
        )
        return StreamRemuxer(command), rate

    def _close_stages(self, stages: dict) -> None:
        for stage in stages.values():
            close = getattr(stage, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - one bad stage must not block the rest
                    continue

    # -- introspection ----------------------------------------------------
    def progress(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        stats = job.executor.stats
        stats.ring_occupancy = job.executor.rings.occupancies()
        snapshot = stats.snapshot()
        snapshot["rings"] = job.executor.rings.snapshot()
        snapshot["reserved_mib"] = job.planned.reservation.total_mib
        snapshot["max_in_flight"] = job.planned.max_in_flight
        if job.source_rate is not None:
            multiplier = job.planned.chain.request.fps_multiplier
            out_rate = retimed_rate(job.source_rate, multiplier)
            snapshot["source_frame_rate"] = str(job.source_rate)
            snapshot["output_frame_rate"] = str(out_rate)
            snapshot["expected_frames_out"] = expected_frame_count(
                stats.frames_decoded, multiplier
            )
        return snapshot

    async def cancel(self, job_id: str) -> dict:
        """Cancel synchronously enough that the reservation stays honest.

        The endpoint must not return before rings and execution contexts are
        released, or the ledger would show bytes the tenant no longer holds
        while a new job is admitted against them.
        """
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.executor.cancel()
        await job.executor.rings.close()
        if job.task is not None and not job.task.done():
            try:
                await asyncio.wait_for(asyncio.shield(job.task), timeout=30)
            except asyncio.TimeoutError:
                job.task.cancel()
        return job.executor.stats.snapshot()

    def engines(self) -> dict:
        """What is cached for this card, and what a request would have to build."""
        entries = []
        for path, manifest in self.cache.entries():
            key = manifest.get("key", {})
            entries.append(
                {
                    "engine": path.name,
                    "model_id": key.get("model_id"),
                    "precision": key.get("precision"),
                    "shapes": key.get("shapes", {}).get("token"),
                    "nvml_uuid": key.get("nvml_uuid"),
                    "runtime": f"{key.get('runtime')} {key.get('runtime_version')}",
                    "parity": manifest.get("parity"),
                    "bytes": path.stat().st_size,
                }
            )
        return {
            "cache_dir": str(self.cache.root),
            "cached": entries,
            # An engine build is exclusive on the card and minutes long. The
            # estimate is deliberately coarse and labelled as such: it exists so
            # a caller can decide to wait, not to be accurate.
            "build_estimate_seconds": 300,
            "build_is_exclusive_on_card": True,
        }


def create_app(config: TenantConfig, *, device_id: int = 0):
    """Build the FastAPI application. Imported lazily by the entry point."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel

    service = VideoEnhanceService(config, device_id=device_id)
    app = FastAPI(title="htsglang video-enhance (Class 3)")

    class Body(BaseModel):
        source_url: str
        source_width: int
        source_height: int
        target: str = "3840x2160"
        fps_multiplier: int = 1
        dtype: str = "fp16"
        enable_sr: bool = True
        sr_scale: int = 4
        enable_resize: bool = True
        rife_scale: float = 1.0
        rife_version: str = "4.6"
        streams_in_flight: int = 2
        overload_policy: str = "stall"
        container: str = "video/mp4"
        video_codec: str = "h264"
        enhance_video_index: int | None = None
        passthrough_audio: bool = True
        passthrough_subtitles: bool = True
        passthrough_other_video: bool = True
        passthrough_data: bool = True
        source_frame_rate: str | None = None

    @app.post("/v1/video/enhance")
    async def enhance(body: Body):
        parsed = EnhanceRequestBody(**body.model_dump())
        try:
            service.plan(parsed)
        except (ChainError, TenantConfigError) as exc:
            # Refusal carries the arithmetic, so the caller can fix the request
            # rather than guess.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return StreamingResponse(
            service.stream_response(parsed),
            media_type=parsed.container,
            headers={"X-Enhance-Tenant": config.tenant_id},
        )

    @app.get("/v1/video/enhance")
    async def enhance_by_url(
        source_url: str,
        source_width: int,
        source_height: int,
        target: str = "3840x2160",
        fps_multiplier: int = 1,
        enable_sr: bool = True,
        sr_scale: int = 4,
        rife_scale: float = 1.0,
        rife_version: str = "4.6",
        container: str = "video/mp4",
        overload_policy: str = "stall",
        enhance_video_index: int | None = None,
    ):
        """Enhance-by-URL: a GET that returns the enhanced stream.

        This is the primary consumption form. A player that can open an HTTP
        URL is already a client -- VLC and mpv open this endpoint directly, no
        plugin, no filter graph, no local install. The player integrations on
        the roadmap are convenience wrappers around this URL, not a separate
        transport.
        """
        parsed = EnhanceRequestBody(
            source_url=source_url,
            source_width=source_width,
            source_height=source_height,
            target=target,
            fps_multiplier=fps_multiplier,
            enable_sr=enable_sr,
            sr_scale=sr_scale,
            rife_scale=rife_scale,
            rife_version=rife_version,
            container=container,
            overload_policy=overload_policy,
            enhance_video_index=enhance_video_index,
        )
        try:
            service.plan(parsed)
        except (ChainError, TenantConfigError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return StreamingResponse(
            service.stream_response(parsed),
            media_type=parsed.container,
            headers={"X-Enhance-Tenant": config.tenant_id},
        )

    @app.get("/v1/video/tracks")
    async def tracks(source_url: str, enhance_video_index: int | None = None):
        """What the muxer would do with every track of this source."""
        parsed = EnhanceRequestBody(
            source_url=source_url,
            source_width=0,
            source_height=0,
            enhance_video_index=enhance_video_index,
        )
        try:
            return JSONResponse(service.track_plan(parsed))
        except MuxError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/video/enhance/{job_id}")
    async def progress(job_id: str):
        try:
            return JSONResponse(service.progress(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no job {job_id}") from exc

    @app.delete("/v1/video/enhance/{job_id}")
    async def cancel(job_id: str):
        try:
            return JSONResponse(await service.cancel(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no job {job_id}") from exc

    @app.get("/v1/video/engines")
    async def engines():
        return JSONResponse(service.engines())

    @app.get("/v1/video/plan")
    async def plan(body: Body):
        parsed = EnhanceRequestBody(**body.model_dump())
        try:
            planned = service.plan(parsed)
        except (ChainError, TenantConfigError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(
            {
                "chain": [s.describe() for s in planned.chain.stages],
                "reserved_mib": planned.reservation.total_mib,
                "posts_mib": {
                    k: round(v / (1024 * 1024), 1)
                    for k, v in planned.reservation.posts.items()
                },
                "max_in_flight": planned.max_in_flight,
            }
        )

    return app
