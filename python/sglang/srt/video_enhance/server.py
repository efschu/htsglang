# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The chunked-HTTP surface of the Class-3 video-enhance tenant.

Per DESIGN #333 §8.5, extended by #338 for the browser-extension client:

==========================================  ================================
``POST   /v1/video/enhance``                enhance a stream, chunked body
``GET    /v1/video/enhance``                enhance-by-URL, same stream
``GET    /v1/video/enhance/{id}``           progress, ms/frame, rings
``DELETE /v1/video/enhance/{id}``           cancel, releasing rings/contexts
``GET    /v1/video/capabilities``           what this deployment sustains
``GET    /v1/video/tracks``                 what the muxer does per track
``GET    /v1/video/engines``                what is cached, what needs a build
``GET    /v1/video/liveness``               the dead-consumer timeouts
==========================================  ================================

``/v1/video/engines`` is operational, not cosmetic: an engine build is
minutes long and exclusive on the card, so a request implying one has to say
so before it starts rather than appearing to hang.

Both enhance forms accept ``start_s`` and ``duration_s`` (#338): a client
that only wants a stretch of a long source should not pay for the prefix.
The range is resolved to frame indices once, at the HTTP surface, and reaches
the decode stage as ``start_frame``/``frame_limit`` -- the same two knobs a
multi-card shard already uses -- and the remuxer as an input seek on the
source, so passthrough audio and subtitles start at the same point.

Back-pressure is structural rather than configured. The response body is an
async generator; Starlette awaits each yield until the transport has accepted
it, so a client whose TCP window is full blocks the generator, which stops
draining the bridge ring, which stops the encode stage, which fills the
upstream rings and stalls the decoder within one ring depth. Nothing in this
file buffers a frame outside a bounded ring.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import AsyncIterator

from sglang.srt.liveness import (
    ClaimKind,
    ConsumerWatchdog,
    EndpointClass,
    LivenessConfig,
    ResourceClaim,
)
from sglang.srt.video_enhance.chain import ChainError, ChainRequest
from sglang.srt.video_enhance.chain_policy import (
    ChainDecision,
    ChainPolicyError,
    PolicyInputs,
    PolicyRequest,
    SourceProbe,
    require_chain,
)
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
from sglang.srt.video_enhance.preview import (
    DEFAULT_PREVIEW_BITRATE,
    DEFAULT_PREVIEW_WIDTH,
    PreviewConfig,
    PreviewLanes,
    build_preview_lanes,
)
from sglang.srt.video_enhance.probes import answer_capability, load_frontier
from sglang.srt.video_enhance.ring import BoundedRing, OverloadPolicy, RingClosed
from sglang.srt.video_enhance.streaming import (
    DEFAULT_RATE_WINDOW_S,
    DEFAULT_WATERMARK_S,
    RateWindow,
    SourceKind,
    StreamingAdmission,
    StreamingAdmissionError,
    StreamingPolicy,
    admit_streaming_source,
)
from sglang.srt.video_enhance.tenant import (
    PlannedJob,
    TenantConfig,
    TenantConfigError,
    build_stages,
    plan_job,
)

logger = logging.getLogger(__name__)

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

#: Media type for a bare preview elementary stream. RFC 6184 for H.264;
#: there is no registered type for a raw AV1 OBU stream, so it is served as
#: the generic binary type rather than as a container it is not.
_PREVIEW_MEDIA_TYPE: dict[str, str] = {
    "h264": "video/H264",
    "hevc": "video/H265",
    "av1": "application/octet-stream",
}


#: The chain configurations a client picks between by name. The browser
#: extension (``clients/browser-extension``) offers exactly these three, and
#: the capability endpoint reports all of them, so the name a user selects in
#: a settings page and the name a measurement row is filed under are the same
#: string.
#:
#: The three are the three things a viewer can ask for, and each one names
#: exactly the stages it runs: ``sr_only`` is super-resolution and resize with
#: the source frame rate kept; ``rife_only`` is interpolation alone, which is
#: the cheap arm and the one that sustains 4K (see
#: docs/dev/TASK_333_M2_MEASUREMENTS.md); ``full_chain`` is both.
#:
#: ``sr_only`` carries ``fps_multiplier: 1``, which the chain builder reads as
#: "no RIFE stage" (``chain.py``), so the preset costs no interpolation memory
#: and no RIFE engine -- the reason it is worth having as its own name rather
#: than as ``full_chain`` with the multiplier turned down.
CHAIN_PRESETS: dict[str, dict] = {
    "sr_only": {
        "enable_sr": True,
        "sr_scale": 4,
        "enable_resize": True,
        "fps_multiplier": 1,
        "description": "x4 super-resolution and resize to target; source frame rate preserved",
    },
    "rife_only": {
        "enable_sr": False,
        "enable_resize": False,
        "fps_multiplier": 2,
        "description": "interpolation only; source resolution preserved",
    },
    "full_chain": {
        "enable_sr": True,
        "sr_scale": 4,
        "enable_resize": True,
        "fps_multiplier": 2,
        "description": "x4 super-resolution, resize to target, then interpolation",
    },
}


class RangeError(ValueError):
    """A time range that cannot be turned into a frame range."""


class PreviewUnavailable(LookupError):
    """A preview was asked for on a job that has no lane to serve it."""


class JobIdError(ValueError):
    """A client-supplied job id that cannot be used."""


#: Characters a client-supplied job id may contain. Deliberately narrow: the
#: id appears in a URL path on ``DELETE`` and as a ring name in logs, and a
#: value that has to be escaped in either place is not worth accepting.
_JOB_ID_ALPHABET = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)
JOB_ID_MAX_LENGTH = 64


def normalize_job_id(job_id: str | None) -> str:
    """Validate a client-supplied job id, or mint one.

    A client that hands the enhance URL to something else -- a ``<video>``
    element, a player, a download -- never sees the response headers, so it
    cannot learn a server-minted id. Letting it name the job is what makes
    ``DELETE /v1/video/enhance/{id}`` reachable from such a client at all;
    without it the liveness watchdog is the *only* way a job ever ends, and
    its ``video_stream`` timeout is 300 s.
    """
    if job_id is None:
        return uuid.uuid4().hex[:16]
    if not job_id or len(job_id) > JOB_ID_MAX_LENGTH:
        raise JobIdError(
            f"job_id must be 1 to {JOB_ID_MAX_LENGTH} characters (got {len(job_id)})"
        )
    bad = sorted(set(job_id) - _JOB_ID_ALPHABET)
    if bad:
        raise JobIdError(
            f"job_id may contain only letters, digits, '-' and '_'; "
            f"rejected {''.join(bad)!r}"
        )
    return job_id


@dataclass(frozen=True)
class TimeRange:
    """A requested stretch of a source, resolved to decode-stage frame indices.

    ``frame_limit`` is None for "to the end of the source", which is also what
    an absent ``duration_s`` means. ``start_frame`` is an index into the
    source timeline, not into the range, because every frame index in the
    chain is absolute -- the same invariant the multi-card shards rely on.
    """

    start_s: float = 0.0
    duration_s: float | None = None
    start_frame: int = 0
    frame_limit: int | None = None
    frame_rate: str | None = None

    @property
    def is_whole_source(self) -> bool:
        return self.start_frame == 0 and self.frame_limit is None

    def describe(self) -> dict:
        return {
            "start_s": self.start_s,
            "duration_s": self.duration_s,
            "start_frame": self.start_frame,
            "frame_limit": self.frame_limit,
            "frame_rate": self.frame_rate,
        }


#: Whole-source range: what every request that names no range resolves to, and
#: therefore the value that keeps the default path byte-for-byte what it was.
WHOLE_SOURCE = TimeRange()


def resolve_time_range(
    *,
    start_s: float,
    duration_s: float | None,
    rate: Fraction | None,
    source_duration_s: float | None = None,
) -> TimeRange:
    """Turn a requested (start, duration) in seconds into a frame range.

    Pure, so the validation is testable without a source, a card or a socket.

    The conversion goes through :class:`~fractions.Fraction` rather than
    float multiplication. At 24000/1001 fps a float product lands a hair
    under or over an integer depending on the value, and the difference is a
    whole frame at the seam -- which is the one place a viewer would see it.

    A range past the end of the source is not an error: the source simply
    ends, exactly as it does without a range. A range that *starts* past the
    end is an error, because it can only produce an empty stream, and an
    empty video body is indistinguishable from a broken server at the client.
    """
    if start_s < 0:
        raise RangeError(f"start_s must not be negative (got {start_s})")
    if duration_s is not None and duration_s <= 0:
        raise RangeError(f"duration_s must be positive (got {duration_s})")
    if not start_s and duration_s is None:
        return WHOLE_SOURCE
    if rate is None or rate <= 0:
        raise RangeError(
            "a time range needs the source frame rate: seconds cannot be "
            "converted to frame indices without it. Pass source_frame_rate, "
            "or use a source ffprobe can read a rate from."
        )
    if source_duration_s is not None and start_s >= source_duration_s:
        raise RangeError(
            f"start_s {start_s} is at or past the source duration "
            f"{source_duration_s:.3f} s; the range would be empty"
        )
    start_frame = int(Fraction(start_s).limit_denominator(1000000) * rate)
    frame_limit: int | None = None
    if duration_s is not None:
        frame_limit = int(Fraction(duration_s).limit_denominator(1000000) * rate)
        if frame_limit < 1:
            raise RangeError(
                f"duration_s {duration_s} is shorter than one frame at "
                f"{rate} fps; the range would be empty"
            )
    return TimeRange(
        start_s=float(start_s),
        duration_s=None if duration_s is None else float(duration_s),
        start_frame=start_frame,
        frame_limit=frame_limit,
        frame_rate=str(rate),
    )


async def decode_frames(stage, source_url: str):
    """Bind a source to the decode stage and drive it as the executor's source.

    The executor wants an async iterator of frames; the decode stage is a
    synchronous pull source. One frame per await and never running ahead, so
    back-pressure still stops at the decoder -- the same bridge
    ``chunk_worker._as_async`` uses inside a multi-card shard, kept here as
    well because importing it would drag the multi-card module into the
    single-card path.
    """
    stage.set_source(source_url)
    for frame in stage:
        yield frame
        await asyncio.sleep(0)


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
    #: #460. Let the RIFE ladder pick the highest-quality variant whose
    #: measured cost the chain can carry, instead of running ``rife_version``.
    #: Off by default so an existing caller's stream does not change model
    #: underneath it; the chosen version and the whole ladder come back in the
    #: job's ``chain_decision``.
    auto_rife_version: bool = False
    #: #460. Force one variant. Beats both the ladder and ``rife_version``,
    #: and is the only way to run a variant whose frontier is unmeasured.
    pin_rife_version: str | None = None
    #: #460. Milliseconds per frame pair the ladder may spend, for a caller
    #: that has done its own budget arithmetic. Absent, the ladder walks the
    #: rungs in quality order against the chain's own aggregate gate.
    rife_budget_ms: float | None = None
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
    #: #338 time range. ``start_s`` 0.0 with ``duration_s`` None is the whole
    #: source and is the default, so a request that names no range takes
    #: exactly the path it took before the range existed.
    start_s: float = 0.0
    duration_s: float | None = None
    #: #344a live preview taps. Off by default, and that is the backward
    #: compatibility statement: a request that does not ask for a preview
    #: builds no tap, so the chain runs exactly the code it ran before.
    preview: bool = False
    preview_width: int = DEFAULT_PREVIEW_WIDTH
    preview_bitrate: int = DEFAULT_PREVIEW_BITRATE
    #: Encode every Nth frame into the preview. The honest lever on what a
    #: tap costs the chain, exposed because the right value depends on how
    #: much throughput the operator is willing to spend on being able to look.
    preview_fps_divisor: int = 1
    #: #451 adaptive chain planning. ``"off"`` -- the default -- runs exactly
    #: the chain the request names, which is what every existing client does
    #: and what keeps this path what it was. ``"adaptive"`` hands the source
    #: probe and the target to ``chain_policy`` and runs the shape it picks,
    #: reporting the mode and the reason in the job status.
    chain_policy: str = "off"
    #: Delivered frame rate the adaptive planner should reach, as ``"50"`` or
    #: ``"60000/1001"``. Absent means source rate times ``fps_multiplier``,
    #: which is what the non-adaptive path would have produced.
    target_fps: str | None = None
    #: Let the planner consider dropping input frames. Off by default: the
    #: mode discards source content and must be asked for by name.
    allow_decimation: bool = False
    max_decimation: int = 2
    #: Let the planner price a stage nobody measured by extrapolation. Off by
    #: default; when on, the reported provenance says ``estimate``.
    allow_estimates: bool = False
    #: Seconds of lead the client is willing to buffer before playback. 0.0
    #: makes the aggregate throughput gate strict.
    watch_ahead_s: float = 0.0
    #: #448 streaming input. ``"finished"`` is the default and the unchanged
    #: path; ``"growing"`` is a file still being written and ``"live"`` a feed
    #: with no end. See ``streaming.admit_streaming_source`` for what each
    #: kind is and is not allowed to do.
    source_kind: str = SourceKind.FINISHED.value
    #: Seconds of finished output an admitted streaming job may hold back.
    #: Ignored on a finished source, whose bridge stays at depth 1.
    output_watermark_s: float = DEFAULT_WATERMARK_S
    #: Sliding window the reported in/out rates are averaged over.
    rate_window_s: float = DEFAULT_RATE_WINDOW_S
    #: How long a growing source may produce nothing before it counts as done.
    stream_idle_timeout_s: float = 30.0

    def has_time_range(self) -> bool:
        return bool(self.start_s) or self.duration_s is not None

    def preview_config(self) -> PreviewConfig:
        return PreviewConfig(
            width=self.preview_width,
            bitrate=self.preview_bitrate,
            fps_divisor=self.preview_fps_divisor,
            codec=self.video_codec,
        )

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

    def is_adaptive(self) -> bool:
        return self.chain_policy == "adaptive"

    def policy_request(self, source_rate: Fraction | None) -> PolicyRequest:
        """The #451 target, resolved against whatever rate is known.

        ``require_runnable`` is set unconditionally here. The policy is
        allowed to answer "a pre-downscale would fit" as a *question*, but the
        two shapes that need a decode stage this executor does not have must
        never be started -- the decoder would ignore the plan and the muxer
        would retime against a frame count that never arrives.
        """
        if self.target_fps:
            num, _, den = self.target_fps.partition("/")
            target_rate = Fraction(int(num), int(den or 1))
        elif source_rate is not None:
            target_rate = Fraction(source_rate) * self.fps_multiplier
        else:
            raise ChainPolicyError(
                "adaptive chain planning needs a target frame rate: pass "
                "target_fps, or use a source whose rate ffprobe can read"
            )
        return PolicyRequest(
            target=Resolution.parse(self.target),
            target_frame_rate=target_rate,
            dtype=self.dtype,
            sr_scale=self.sr_scale,
            rife_scale=self.rife_scale,
            rife_version=self.rife_version,
            auto_rife_version=self.auto_rife_version,
            pin_rife_version=self.pin_rife_version,
            rife_budget_ms=self.rife_budget_ms,
            streams_in_flight=self.streams_in_flight,
            allow_decimation=self.allow_decimation,
            max_decimation=self.max_decimation,
            allow_estimates=self.allow_estimates,
            require_runnable=True,
            max_watch_ahead_s=self.watch_ahead_s,
        )

    def streaming_policy(self, output_rate: Fraction | None) -> StreamingPolicy:
        return StreamingPolicy(
            kind=SourceKind(self.source_kind),
            output_frame_rate=output_rate or Fraction(self.fps_multiplier * 25),
            watermark_s=self.output_watermark_s,
            overload=self.policy(),
            rate_window_s=self.rate_window_s,
            idle_timeout_s=self.stream_idle_timeout_s,
        )


@dataclass
class Job:
    job_id: str
    executor: PipelineExecutor
    planned: PlannedJob
    task: asyncio.Task | None = None
    created_at: float = 0.0
    source_rate: Fraction | None = None
    watchdog: ConsumerWatchdog | None = None
    time_range: TimeRange = WHOLE_SOURCE
    #: Live preview taps (§8.1), or None when the request did not ask for
    #: them. Built with the job so a viewer can attach at any point while it
    #: runs; they cost nothing until they are started.
    previews: PreviewLanes | None = None
    #: #451 adaptive planning verdict, or None when the request named its own
    #: chain. Client-visible in the job status: a client handed a chain it did
    #: not ask for is owed the mode and the reason.
    decision: ChainDecision | None = None
    #: #448 streaming verdict, always present. A finished source carries the
    #: unchanged-path admission.
    admission: StreamingAdmission | None = None
    #: #448 sustained in/out rate over a sliding window, sampled where the
    #: status is rendered and after each chunk the transport accepted.
    rates: RateWindow | None = None


class VideoEnhanceService:
    """Job registry and the request lifecycle, independent of the web framework.

    Keeping the framework out of this class is what makes the back-pressure
    test hermetic: the test drives ``stream_response`` with a slow consumer
    and asserts on ring occupancy, with no socket involved.
    """

    def __init__(
        self,
        config: TenantConfig,
        *,
        device_id: int = 0,
        liveness: LivenessConfig | None = None,
    ) -> None:
        self.config = config
        self.device_id = device_id
        self.jobs: dict[str, Job] = {}
        self.cache = EngineCache(config.engine_cache_dir)
        self.liveness = liveness or LivenessConfig()

    # -- planning ---------------------------------------------------------
    def plan(self, body: EnhanceRequestBody) -> PlannedJob:
        return self.plan_with_policy(body)[0]

    def plan_with_policy(
        self, body: EnhanceRequestBody, info: MediaInfo | None = None
    ) -> tuple[PlannedJob, EnhanceRequestBody, ChainDecision | None]:
        """Resolve the chain, adaptively when the request asks for it (#451).

        Returns the plan, the request body the rest of the pipeline should
        use, and the decision. The body comes back because an adaptive plan
        may change the frame-rate multiplier and the SR flag, and the remuxer
        retimes against the multiplier -- leaving the caller's original value
        in place would produce a container whose declared rate does not match
        the frames in it.
        """
        if not body.is_adaptive():
            return plan_job(self.config, body.to_chain_request()), body, None
        probe_result = self.source_probe(body, info)
        decision = require_chain(
            probe_result,
            body.policy_request(probe_result.frame_rate),
            PolicyInputs.from_probe_dir(
                self.config.measurement_dir, rife_ladder=self._rife_ladder()
            ),
            self.config,
        )
        chosen = decision.request
        assert chosen is not None
        resolved = replace(
            body,
            source_width=chosen.source.width,
            source_height=chosen.source.height,
            target=str(chosen.target),
            fps_multiplier=chosen.fps_multiplier,
            enable_sr=chosen.enable_sr,
            sr_scale=chosen.sr_scale,
            enable_resize=chosen.enable_resize,
            # The ladder may have moved the version; the executor has to build
            # the stage the planner priced, not the one the caller named.
            rife_version=chosen.rife_version,
        )
        return plan_job(self.config, chosen), resolved, decision

    def _rife_ladder(self):
        """The #460 ladder, or ``None`` when it cannot be built.

        Built lazily and never fatally: a deployment whose weight directory is
        unreadable should still be able to plan a chain with the version the
        caller named. A ``None`` ladder makes ``auto_rife_version`` report that
        no ladder was supplied rather than silently choosing.
        """
        cached = getattr(self, "_rife_ladder_cache", "unset")
        if cached != "unset":
            return cached
        try:
            from sglang.srt.video_enhance.rife_ladder import default_ladder

            ladder = default_ladder()
        except Exception as exc:  # noqa: BLE001 - planning must not die of this
            logger.warning("RIFE ladder unavailable, version stays as requested: %s", exc)
            ladder = None
        self._rife_ladder_cache = ladder
        return ladder

    def source_probe(
        self, body: EnhanceRequestBody, info: MediaInfo | None = None
    ) -> SourceProbe:
        """The #451 planner's view of the input.

        ``MediaInfo`` when there is one, because it carries the duration a
        watch-ahead calculation needs; the request's own fields otherwise, so
        a caller that already knows its source can plan without an ffprobe.
        """
        if info is not None:
            return SourceProbe.from_media_info(info, body.enhance_video_index)
        if not body.source_frame_rate:
            raise ChainPolicyError(
                "adaptive chain planning needs the source frame rate: pass "
                "source_frame_rate, or a source that can be probed"
            )
        num, _, den = body.source_frame_rate.partition("/")
        return SourceProbe(
            resolution=Resolution(body.source_width, body.source_height),
            frame_rate=Fraction(int(num), int(den or 1)),
        )

    def admit(
        self, body: EnhanceRequestBody, output_rate: Fraction | None
    ) -> StreamingAdmission:
        """The #448 gate. Refuses rather than returns for an inadmissible source."""
        return admit_streaming_source(
            body.streaming_policy(output_rate), chunked=False
        ).require()

    def track_plan(
        self, body: EnhanceRequestBody, info: MediaInfo | None = None
    ) -> dict:
        """What happens to every track of the source, before anything runs."""
        info = info if info is not None else probe(body.source_url)
        return describe_selection(info, body.track_selection())

    def claim_job_id(self, job_id: str | None) -> str:
        """Validate and reserve an id, refusing one that is already running.

        Silently reusing a live id would make ``DELETE`` ambiguous -- two
        streams, one name, and the cancel reaching whichever the registry
        happens to hold. A finished job's id is free again; only a live one is
        taken.
        """
        candidate = normalize_job_id(job_id)
        existing = self.jobs.get(candidate)
        if existing is not None and (existing.task is None or not existing.task.done()):
            raise JobIdError(f"job {candidate} is already running")
        return candidate

    def resolve_range(
        self, body: EnhanceRequestBody, info: MediaInfo | None = None
    ) -> TimeRange:
        """Resolve the requested time range against the source's real rate.

        Called twice per ranged request on purpose: once by the endpoint, so a
        bad range is a 422 with the arithmetic in it, and once by
        :meth:`stream_response`. A validation failure discovered after the
        response has begun cannot be a status code any more -- it can only be
        a truncated video body, which at the client is indistinguishable from
        a crashed server. The second probe is the price of that distinction.
        """
        if not body.has_time_range():
            return WHOLE_SOURCE
        rate: Fraction | None = None
        duration_s: float | None = None
        if body.source_frame_rate:
            num, _, den = body.source_frame_rate.partition("/")
            rate = Fraction(int(num), int(den or 1))
        if info is None and rate is None:
            try:
                info = probe(body.source_url)
            except MuxError as exc:
                raise RangeError(
                    f"a time range was requested but {body.source_url!r} could "
                    f"not be probed for its frame rate: {exc}"
                ) from exc
        if info is not None:
            duration_s = info.duration_s
            if rate is None:
                selection = body.track_selection()
                rate = info.track(selection.resolve_video_index(info)).frame_rate()
        return resolve_time_range(
            start_s=body.start_s,
            duration_s=body.duration_s,
            rate=rate,
            source_duration_s=duration_s,
        )

    # -- execution --------------------------------------------------------
    async def stream_response(
        self,
        body: EnhanceRequestBody,
        *,
        source_factory=None,
        stage_factory=None,
        media_info: MediaInfo | None = None,
        remux: bool = True,
        job_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Run one job and yield muxed chunks as they are produced.

        The encoded elementary stream is fed to a remuxer that copies every
        kept non-video track through bit-identically and retimes the enhanced
        video to the interpolated rate. Back-pressure survives the extra
        process: the remuxer's ``feed`` awaits a drain, and its stdout is only
        read when the bounded response bridge has room.
        """
        planned, body, decision = self.plan_with_policy(body, media_info)
        time_range = self.resolve_range(body, media_info)
        job_id = self.claim_job_id(job_id)

        # The remuxer is built before the bridge because it is what resolves
        # the source rate, and the #448 watermark is a duration that only
        # becomes a depth once the output rate is known.
        remuxer, source_rate = (
            self._make_remuxer(body, media_info, time_range) if remux else (None, None)
        )
        output_rate = (
            retimed_rate(source_rate, body.fps_multiplier)
            if source_rate is not None
            else None
        )
        admission = self.admit(body, output_rate)
        # A finished source keeps the depth-1 bridge exactly as it was: any
        # deeper is a buffer between the socket and the chain that
        # back-pressure has to cross (§8.4 rule 3). A streaming job accepts
        # that crossing deliberately, in exchange for not underrunning a
        # player, and the depth it accepts is the declared watermark.
        bridge_depth = (
            admission.buffer_depth_frames
            if admission.is_streaming
            else RESPONSE_BRIDGE_DEPTH
        )
        bridge = BoundedRing(f"muxer->socket:{job_id}", bridge_depth, body.policy())
        rates = RateWindow(
            window_s=body.rate_window_s,
            target_output_fps=float(output_rate) if output_rate else None,
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
            else build_stages(
                self.config,
                planned.chain,
                device_id=self.device_id,
                start_frame=time_range.start_frame,
                frame_limit=time_range.frame_limit,
            )
        )
        if source_factory is not None:
            source = source_factory(planned.chain)
        else:
            source = decode_frames(
                stages[planned.chain.stages[0].kind], body.source_url
            )

        # Preview lanes are built before the executor because the executor
        # needs the tap map. They are inert until started: a lane with no
        # viewer holds an empty list and a closed encoder task.
        previews: PreviewLanes | None = None
        taps: dict = {}
        if body.preview:
            previews = build_preview_lanes(
                planned.chain,
                fps=retimed_rate(
                    media_info.track(body.enhance_video_index).frame_rate()
                    or Fraction(30),
                    body.fps_multiplier,
                ),
                dtype=body.dtype,
                device_id=self.device_id,
                config=body.preview_config(),
            )
            taps = dict(previews.by_stage)
            previews.start()

        executor = PipelineExecutor(
            job_id=job_id,
            chain=planned.chain,
            stages=stages,
            source=source,
            sink=sink,
            ring_depth=planned.ring_depth,
            policy=body.policy(),
            taps=taps or None,
        )
        job = Job(
            job_id=job_id,
            executor=executor,
            planned=planned,
            created_at=time.time(),
            time_range=time_range,
            previews=previews,
            decision=decision,
            admission=admission,
            rates=rates,
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

        async def release() -> None:
            """What a dead consumer's job costs to give back.

            All three steps are needed, and the third is the one that is easy
            to miss. A client that stopped reading leaves this generator
            suspended at its ``yield``, and a suspended generator never
            reaches its own ``finally`` -- so the stage teardown that runs
            there on a normal or a cancelled stream does not run here.
            Cancelling the executor stops the decoder and closing the bridge
            lets the driver finish, but the decoder session and the encoder
            session are only released by closing the stages, which this has
            to do itself.
            """
            executor.cancel()
            # Setting the cancel flag is not enough on its own. A stage
            # blocked in ``ring.put`` on a full ring only sees the flag after
            # that await returns, and on a stalled pipeline it never does --
            # nothing is draining. Closing the rings is what wakes every
            # blocked producer and consumer, which is the same thing the
            # DELETE endpoint does and for the same reason.
            await executor.rings.close()
            await bridge.close()
            if job.task is not None and not job.task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(job.task),
                        timeout=self.liveness.teardown_timeout_s,
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    job.task.cancel()
            self._close_stages(stages)
            # The preview lanes hold an encoder session each. A job released
            # for a dead consumer must not leave two NVENC sessions and a
            # CUDA context behind for a viewer who is also gone.
            if job.previews is not None:
                await job.previews.close()

        watchdog = ConsumerWatchdog(
            job_id=job_id,
            policy=self.liveness.policy_for(EndpointClass.VIDEO_STREAM),
            release=release,
            # #344: what this stream holds, in the terms the reclamation
            # ladder thinks in. Declaring it is what puts the job's bytes on
            # the ladder while the consumer is a dead suspect instead of
            # leaving them pinned for the full 300 s window. Without a card
            # UUID there is no ledger entry to point at, so the claim carries
            # only the local pipeline and the ledger bridge skips it.
            claims=self._stream_claims(job),
        )
        job.watchdog = watchdog
        watchdog.start()
        try:
            while True:
                try:
                    chunk = await bridge.get()
                except RingClosed:
                    break
                yield chunk
                # After the yield, not before. Control returns here only once
                # the transport has taken the chunk, so this is the one place
                # in the process that knows the peer is still there.
                watchdog.note_progress(len(chunk))
                # Same moment, same reason (#448): a rate sampled here is a
                # rate the transport accepted, and the sample costs one
                # append on a path that has already left the chain.
                rates.observe(
                    frames_in=executor.stats.frames_decoded,
                    frames_out=executor.stats.frames_encoded,
                )
        finally:
            await watchdog.stop()
            if not job.task.done():
                executor.cancel()
                # Closing the chain's own rings, not just the response bridge.
                # A client that disconnects mid-stream leaves the decode and
                # middle stage tasks suspended in ``ring.put`` on rings nobody
                # is draining any more, and the cancel flag is only read after
                # that await returns -- which it never does. Without this the
                # teardown falls through to the timeout below and holds the
                # decoder, the encoder and the reservation for a further 30 s
                # after the socket is already gone. The watchdog's release path
                # and the DELETE endpoint both close the rings for exactly this
                # reason; the disconnect path is the third way out and needs it
                # just as much.
                await executor.rings.close()
                await bridge.close()
                try:
                    await asyncio.wait_for(job.task, timeout=30)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    job.task.cancel()
            self._close_stages(stages)
            if job.previews is not None:
                await job.previews.close()
        # The trailer carries the drop count. A dropped frame is never silent.
        job.executor.stats.dropped = executor.rings.dropped + bridge.stats.dropped

    def _stream_claims(self, job) -> tuple[ResourceClaim, ...]:
        """What one enhance stream holds, for the #344 grace registry.

        The pipeline claim is always there. The VRAM claim only when the
        tenant is configured against a specific card, because the ledger
        addresses an entry by (card UUID, tenant id) and a claim missing
        either coordinate cannot be published.
        """
        claims = [
            ResourceClaim(kind=ClaimKind.PIPELINE, key=job.job_id),
            ResourceClaim(kind=ClaimKind.JOB_SLOT, key=job.job_id),
        ]
        reserved_mib = getattr(
            getattr(job.planned, "reservation", None), "total_mib", 0
        )
        if self.config.card_uuid and reserved_mib:
            claims.append(
                ResourceClaim(
                    kind=ClaimKind.VRAM_LEASE,
                    key=str(self.config.card_uuid),
                    nbytes=int(reserved_mib) * 1024 * 1024,
                    tenant_id=str(self.config.tenant_id),
                )
            )
        return tuple(claims)

    def _make_remuxer(
        self,
        body: EnhanceRequestBody,
        media_info: MediaInfo | None,
        time_range: TimeRange = WHOLE_SOURCE,
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
            source_seek_s=time_range.start_s,
            source_duration_s=time_range.duration_s,
        )
        return StreamRemuxer(command), rate

    # -- previews ---------------------------------------------------------
    def preview_lane(self, job_id: str, which: str):
        """The lane a viewer wants, or a named refusal.

        ``which`` is "input" or "output". A job that was not started with
        ``preview: true`` has no lane, and saying so is better than handing
        back an empty stream a client would read as a stalled encoder.
        """
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.previews is None:
            raise PreviewUnavailable(
                f"job {job_id} was started without preview taps; pass "
                "preview=true on the enhance request to enable them. Taps "
                "cannot be attached to a running job because the chain's tap "
                "map is fixed when the executor is built."
            )
        lane = {
            "input": job.previews.input_lane,
            "output": job.previews.output_lane,
        }.get(which)
        if lane is None:
            raise PreviewUnavailable(
                f"job {job_id} has no {which!r} preview lane; this chain has "
                f"no stage to tap for it. Available: "
                f"{sorted(job.previews.snapshot())}"
            )
        return lane

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
        if not job.time_range.is_whole_source:
            snapshot["time_range"] = job.time_range.describe()
        if job.watchdog is not None:
            snapshot["consumer"] = job.watchdog.state.snapshot()
        if job.previews is not None:
            # Reported next to the pipeline's own numbers on purpose: the
            # claim that a tap costs the chain nothing is checkable from one
            # response, by reading preview drops against frames_encoded.
            snapshot["previews"] = job.previews.snapshot()
        if job.rates is not None:
            # Sampled here as well as in the response loop, so a status poll
            # keeps the window fresh on a job whose client is reading slowly.
            job.rates.observe(
                frames_in=stats.frames_decoded, frames_out=stats.frames_encoded
            )
            snapshot["sustained_rate"] = job.rates.snapshot()
        if job.admission is not None and job.admission.is_streaming:
            snapshot["streaming"] = job.admission.as_dict()
        if job.decision is not None:
            # The client-visible half of #451: which chain shape it is getting
            # and the one line saying why. A client handed a decimated stream
            # without being told is being lied to.
            snapshot["chain_policy"] = job.decision.as_dict()
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

    # -- capability -------------------------------------------------------
    def capabilities(
        self,
        *,
        source: Resolution | None = None,
        target: Resolution | None = None,
        target_fps: float | None = None,
        configuration: str = "full_chain",
    ) -> dict:
        """What this deployment can actually do, separated by how it is known.

        A client picking a chain preset needs two different kinds of answer
        and they must not be confused with one another:

        ``frontier``
            measured. Rows come from probe reports on disk (P1, §8.6), so
            "3840x2160 rife_only sustains 87.8 fps" is a number somebody
            measured on this hardware. With no reports the frontier is empty
            and says so; nothing is extrapolated into it.
        ``budget``
            arithmetic. :func:`~sglang.srt.video_enhance.tenant.plan_job` is
            pure and needs no card, so whether a preset *fits* in the
            configured MiB is always answerable even when no rate has ever
            been measured. Fitting is not the same as being fast enough, and
            the two live under different keys for exactly that reason.
        """
        frontier = load_frontier(self.config.measurement_dir)
        payload: dict = {
            "tenant_id": self.config.tenant_id,
            "budget_mib": self.config.budget_mib,
            "card_uuid": self.config.card_uuid,
            "chain_presets": {
                name: dict(preset) for name, preset in CHAIN_PRESETS.items()
            },
            "containers": sorted(_CONTAINER_BY_MEDIA_TYPE),
            "video_codecs": sorted(_ELEMENTARY_FORMAT),
            "supports_time_range": True,
            "frontier": frontier,
        }
        if source is not None:
            payload["budget"] = self._budget_answers(source, target)
        if target_fps is not None:
            resolution = target or source
            if resolution is None:
                raise ValueError(
                    "target_fps needs a resolution to be answered against; "
                    "pass source= or target="
                )
            payload["answer"] = answer_capability(
                frontier=frontier["rows"],
                resolution=resolution,
                target_fps=target_fps,
                configuration=configuration,
            ).as_dict()
        return payload

    def _budget_answers(
        self, source: Resolution, target: Resolution | None
    ) -> list[dict]:
        """Does each preset fit in the budget for this source? One row each."""
        rows: list[dict] = []
        for name, preset in CHAIN_PRESETS.items():
            # A preset that neither upscales nor resizes keeps the source
            # geometry by definition, so a requested target is not applicable
            # to it. Feeding the target in anyway would make ``rife_only``
            # refuse every request that also named a target -- a refusal about
            # the caller's target, reported as if the preset did not fit.
            upscales = preset.get("enable_sr") or preset.get("enable_resize")
            request = ChainRequest(
                source=source,
                target=(target or source) if upscales else source,
                fps_multiplier=int(preset.get("fps_multiplier", 1)),
                enable_sr=bool(preset.get("enable_sr", False)),
                sr_scale=int(preset.get("sr_scale", 4)),
                enable_resize=bool(preset.get("enable_resize", False)),
            )
            row: dict = {"preset": name, "source": str(source)}
            try:
                planned = plan_job(self.config, request)
            except (ChainError, TenantConfigError) as exc:
                # The refusal carries the arithmetic that produced it, which is
                # what a client needs to pick a preset that does fit.
                rows.append({**row, "fits": False, "reason": str(exc)})
                continue
            rows.append(
                {
                    **row,
                    "fits": True,
                    "target": str(planned.chain.request.target),
                    "reserved_mib": planned.reservation.total_mib,
                    "max_in_flight": planned.max_in_flight,
                    "stages": [s.kind.value for s in planned.chain.stages],
                }
            )
        return rows

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


def create_app(
    config: TenantConfig,
    *,
    device_id: int = 0,
    liveness: LivenessConfig | None = None,
    api_key: str | None = None,
    admin_api_key: str | None = None,
):
    """Build the FastAPI application. Imported lazily by the entry point.

    #510: the two state-changing routes (submit a job, delete a running job)
    are marked ADMIN_OPTIONAL, and the api-key middleware is installed when
    either key is configured. Without a key the behaviour is unchanged. Job ids
    are client-chosen and short, so ``DELETE /v1/video/enhance/{job_id}`` was
    a guessable way to kill someone else's running job (audit #506, A2-F13).
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel

    from sglang.srt.utils.auth import (
        AuthLevel,
        add_api_key_middleware,
        auth_level,
    )

    service = VideoEnhanceService(config, device_id=device_id, liveness=liveness)
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
        start_s: float = 0.0
        duration_s: float | None = None
        job_id: str | None = None

    def _headers(job_id: str) -> dict[str, str]:
        """Response headers every enhance form carries.

        ``X-Enhance-Job`` is the id ``DELETE`` takes. It is echoed even when
        the client supplied it, so a client that reads headers and one that
        cannot both end up holding the same string.
        """
        return {
            "X-Enhance-Tenant": config.tenant_id,
            "X-Enhance-Job": job_id,
        }

    @app.post("/v1/video/enhance")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def enhance(body: Body):
        payload = body.model_dump()
        requested_id = payload.pop("job_id", None)
        parsed = EnhanceRequestBody(**payload)
        try:
            service.plan(parsed)
            service.resolve_range(parsed)
            job_id = service.claim_job_id(requested_id)
        except (
            ChainError,
            TenantConfigError,
            RangeError,
            ChainPolicyError,
            StreamingAdmissionError,
        ) as exc:
            # Refusal carries the arithmetic, so the caller can fix the request
            # rather than guess.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except JobIdError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return StreamingResponse(
            service.stream_response(parsed, job_id=job_id),
            media_type=parsed.container,
            headers=_headers(job_id),
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
        start_s: float = 0.0,
        duration_s: float | None = None,
        source_frame_rate: str | None = None,
        job_id: str | None = None,
    ):
        """Enhance-by-URL: a GET that returns the enhanced stream.

        This is the primary consumption form. A player that can open an HTTP
        URL is already a client -- VLC and mpv open this endpoint directly, no
        plugin, no filter graph, no local install; the browser extension in
        ``clients/browser-extension`` puts this URL into a ``<video>`` element's
        ``src``. The player integrations on the roadmap are convenience
        wrappers around this URL, not a separate transport.

        ``start_s`` and ``duration_s`` are the #338 time range. Both default to
        the whole source, so an unranged request is the request this endpoint
        has always served.

        ``job_id`` lets the caller name the job it is about to start. A client
        that puts this URL into a ``<video>`` element never sees the response
        headers, so naming the job in the URL is the only way it can later
        ``DELETE`` it.
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
            start_s=start_s,
            duration_s=duration_s,
            source_frame_rate=source_frame_rate,
        )
        try:
            service.plan(parsed)
            service.resolve_range(parsed)
            claimed = service.claim_job_id(job_id)
        except (
            ChainError,
            TenantConfigError,
            RangeError,
            ChainPolicyError,
            StreamingAdmissionError,
        ) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except JobIdError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return StreamingResponse(
            service.stream_response(parsed, job_id=claimed),
            media_type=parsed.container,
            headers=_headers(claimed),
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

    @app.get("/v1/video/preview/{job_id}/{which}")
    async def preview(job_id: str, which: str):
        """Watch a running job's input or output live (§8.1).

        ``which`` is "input" or "output". The body is a bare elementary
        stream, which is what a preview wants: no container means no moov
        atom to wait for and no duration to declare, so a player can start on
        the first IDR and a stream with no defined end is not a special case.

        A viewer who reads slowly loses preview frames -- the tap drops on
        ingress -- and costs the enhance job nothing. That is the §8.1 rule
        and it is enforced in ``preview.PreviewTap.offer``, not here.
        """
        if which not in ("input", "output"):
            raise HTTPException(
                status_code=422,
                detail=f"preview side must be 'input' or 'output', got {which!r}",
            )
        try:
            lane = service.preview_lane(job_id, which)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no job {job_id}") from exc
        except PreviewUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return StreamingResponse(
            lane.stream(),
            media_type=_PREVIEW_MEDIA_TYPE.get(lane.config.codec, "video/H264"),
            headers={
                "X-Preview-Resolution": str(lane.target),
                "X-Preview-Frame-Rate": str(lane.fps),
                "Cache-Control": "no-store",
            },
        )

    @app.delete("/v1/video/enhance/{job_id}")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def cancel(job_id: str):
        """Explicit abort. Returns only once the job has actually let go.

        A client that can say "stop" should say it: the liveness watchdog
        (#344b) is the backstop for clients that cannot, and its
        ``video_stream`` timeout is a deliberately generous 300 s because a
        paused player is a normal thing. A browser extension closing a tab
        knows immediately, so it sends this and the card is free in
        milliseconds rather than in five minutes.

        The await before the return is the contract: rings closed, execution
        contexts released, reservation given back. Returning earlier would let
        the ledger admit a new job against bytes this one still holds.
        """
        try:
            return JSONResponse(await service.cancel(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no job {job_id}") from exc

    @app.get("/v1/video/capabilities")
    async def capabilities(
        source: str | None = None,
        target: str | None = None,
        target_fps: float | None = None,
        configuration: str = "full_chain",
    ):
        """What this deployment sustains, measured and arithmetic kept apart.

        Called by a client before it offers the user a chain preset. With no
        query parameters it is a static description: presets, containers,
        codecs, and the measured frontier if one has been imported. With
        ``source`` it adds the per-preset budget verdict for that source size,
        and with ``target_fps`` the frontier answer for the requested rate.
        """
        try:
            return JSONResponse(
                service.capabilities(
                    source=Resolution.parse(source) if source else None,
                    target=Resolution.parse(target) if target else None,
                    target_fps=target_fps,
                    configuration=configuration,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/video/engines")
    async def engines():
        return JSONResponse(service.engines())

    @app.get("/v1/video/liveness")
    async def liveness_policy():
        """The configured dead-consumer timeout for every endpoint class.

        Operational rather than cosmetic: a job that vanished and a job that
        is being held for a paused viewer look the same from outside, and the
        difference is this number.
        """
        return JSONResponse(
            {
                "timeouts_s": service.liveness.describe(),
                "poll_interval_s": service.liveness.poll_interval_s,
                "teardown_timeout_s": service.liveness.teardown_timeout_s,
            }
        )

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

    # Same rule as the runtime and the registry: a key has to be configured
    # before anything is closed, so an existing deployment is unchanged.
    if api_key or admin_api_key:
        add_api_key_middleware(app, api_key=api_key, admin_api_key=admin_api_key)

    return app
