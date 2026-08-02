# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Adaptive chain planning: which chain shape carries this source (#451).

``tenant.plan_job`` answers "does the chain the caller asked for fit in the
budget". It does not answer "which chain should the caller have asked for",
and on a rig with three unequal cards that is the question a client cannot
answer for itself: it would need the measured per-stage rates, the reservation
formula and the geometry rules, all of which live here.

This module is the policy layer above ``plan_job``. It takes an input probe
(resolution and frame rate, off ``mux.probe``), a target, and a measured
capability frontier, generates the chain shapes that could reach that target,
prices every one of them against the frontier *and* the §6.2 reservation, and
picks the best one that is feasible. If none is, it refuses with the numbers
that made every candidate fail rather than with a boolean.

The four shapes, in the order a viewer would rank them:

``FULL``
    Every input frame goes through super-resolution and interpolation. This
    is what the target scenario wants and it wins automatically whenever the
    frontier says the cards carry it -- there is no configuration that
    prefers a cheaper shape when the full one fits.

``RIFE_ONLY``
    The input is already at or above the target resolution, so the SR stage
    would upscale detail only to throw it away in a resize. Skipping it is
    not a downgrade, it is the correct chain, and it is therefore ranked
    alongside ``FULL`` rather than below it. ``sr_scale`` on a 4K source
    produces 8K, which §3.4 of the implementation record already refuses to
    do unasked.

``PRE_DOWNSCALE``
    The input sits above the largest SR input resolution the frontier can
    carry. A Lanczos-3 pre-resize down to that entry point makes the full
    chain affordable at the cost of the detail thrown away before SR. The
    entry point is *solved* from the measured SR rows, not guessed: the
    candidate ladder is exactly the SR input resolutions somebody measured,
    filtered to those that still reach the target after the x``sr_scale``
    upscale.

``DECIMATE_RESYNTH``
    Drop every ``d``-th input frame, run the survivors through the chain, and
    let RIFE re-synthesise the dropped ones back. This is the only shape that
    discards real input, so it is **opt-in by name** (``allow_decimation``)
    and never a silent fallback, and the job response carries both the
    fraction of input discarded and the fraction of output that is synthetic.

Three properties this module holds itself to:

**A chain nobody measured is not priced.** Every stage cost is a
:class:`~sglang.srt.planner.cost_model.Rate` carrying ``measured`` /
``estimate`` / ``absent``. A candidate with an absent stage is UNPRICEABLE and
is refused; with ``allow_estimates`` it may instead be priced from a
pixel-count extrapolation off a measured row on the same card, and then the
whole decision is labelled ``estimate`` so a caller can tell the two apart.
There is no third path where a number is invented quietly.

**The reservation is the same arithmetic the executor will use.** Budget
feasibility goes through :func:`~sglang.srt.video_enhance.tenant.plan_job`,
so the policy and the tenant cannot drift apart; a card whose budget cannot
hold one in-flight frame is dropped from the aggregate with its own MiB
figures attached.

**The throughput gate is the aggregate, not a single frame.** Regime A
replicated: each card runs a whole chain over its own stretch of the
timeline, so the rates add. :func:`~sglang.srt.video_enhance.probes.playback_feasibility`
does that sum and also reports the watch-ahead lead a shortfall would need,
which is the number that decides whether a few seconds of latency absorbs it.

**BOOT-PENDING.** The frontier this planner reads today is the 4.6 /
fp32-parity-era P1 table (``docs/dev/TASK_333_M2_MEASUREMENTS.md``). The
operating point named by the user is the fp16/bf16 TensorRT engine built from
the pinned ONNX and RIFE 4.26, and neither has been measured. The tipping
points between the modes are therefore correct *for the table they are given*
and will move when that measurement lands. Nothing here hard-codes a
threshold, which is why the move is a re-read of the table rather than a code
change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from sglang.srt.planner.cost_model import (
    Provenance,
    Rate,
    StageRateTable,
    stage_rates_from_samples,
)
from sglang.srt.video_enhance.chain import (
    Chain,
    ChainError,
    ChainRequest,
    StageKind,
    build_chain,
)
from sglang.srt.video_enhance.frame_math import Resolution
from sglang.srt.video_enhance.probes import playback_feasibility
from sglang.srt.video_enhance.shard_plan import stage_stream_factors
from sglang.srt.video_enhance.tenant import PlannedJob, TenantConfig, plan_job

__all__ = [
    "REQUIRES_FRAME_DECIMATION",
    "REQUIRES_SCALED_DECODE",
    "ChainDecision",
    "ChainMode",
    "ChainPolicyError",
    "Candidate",
    "CandidatePrice",
    "CardPrice",
    "PolicyInputs",
    "PolicyRequest",
    "SourceProbe",
    "StagePrice",
    "candidate_chains",
    "choose_chain",
    "price_candidate",
    "require_chain",
    "sr_entry_points",
]


class ChainPolicyError(ValueError):
    """A policy question that cannot be asked, let alone answered."""


class ChainMode(str, Enum):
    """The user-facing name of a chain shape. Reported in the job response."""

    FULL = "full"
    RIFE_ONLY = "rife_only"
    PRE_DOWNSCALE = "pre_downscale"
    DECIMATE_RESYNTH = "decimate_resynth"


#: How much of the source a mode gives up, as a rank. Lower is better, and the
#: ranking is over *what the viewer loses*, not over cost: ``FULL`` and
#: ``RIFE_ONLY`` both deliver every input frame at full detail (``RIFE_ONLY``
#: only skips a stage that had nothing to contribute), a pre-downscale throws
#: away detail before SR, and a decimation throws away whole input frames.
#: Ties are broken by throughput headroom, so cost only ever decides between
#: shapes that cost the viewer the same.
QUALITY_TIER: dict[ChainMode, int] = {
    ChainMode.FULL: 0,
    ChainMode.RIFE_ONLY: 0,
    ChainMode.PRE_DOWNSCALE: 1,
    ChainMode.DECIMATE_RESYNTH: 2,
}

#: Stages the P1 probe grid measures, keyed by the chain's stage kind. The two
#: colour conversions are deliberately absent: nothing in ``probes.py`` ever
#: measured them, so they are reported as *unpriced* on every candidate rather
#: than being given a number nobody took. They are the same two stages in
#: every candidate, so omitting them cannot reorder the ranking -- but it does
#: mean an absolute fps figure from this module is a chain-stage figure and
#: slightly optimistic, and the decision says so.
PRICED_STAGES: frozenset[StageKind] = frozenset(
    {
        StageKind.DECODE,
        StageKind.SR,
        StageKind.RESIZE,
        StageKind.RIFE,
        StageKind.ENCODE,
    }
)

#: Aspect-ratio tolerance when a measured SR input resolution is offered as a
#: pre-downscale entry point. A 1 percent mismatch is a rounding artefact of
#: an even-dimension constraint; anything larger would change the framing.
ASPECT_TOLERANCE = 0.01

#: Executor capability a ``PRE_DOWNSCALE`` plan needs and the M2 executor does
#: not have. ``chain.build_chain`` places resize strictly *after* SR (§8.1),
#: so a resize *before* SR is not expressible as a chain at all. The nearest
#: thing that exists is the ffmpeg decode backend, whose command already
#: carries ``scale=W:H`` (``codec.build_ffmpeg_decode_command``) -- but
#: ``codec.DecodeStage._open`` refuses outright when the source size differs
#: from the planned size, and the PyNvVideoCodec backend has no scaler at all.
#: Two further facts a caller needs before treating this as a small change:
#: ffmpeg's ``scale`` is not the chain's Lanczos-3, and the §6.2 reservation
#: would have to keep charging the decoder pool at the *source* size while
#: every later stage is charged at the entry size.
#:
#: The policy is allowed to recommend the shape -- that is what it is for --
#: but it names the requirement, and ``PolicyRequest.require_runnable``
#: excludes it for the callers that have to start a job rather than answer a
#: question.
REQUIRES_SCALED_DECODE = "scaled_decode_or_pre_sr_resize_stage"

#: Executor capability a ``DECIMATE_RESYNTH`` plan needs and the M2 executor
#: does not have. ``codec.DecodeStage`` takes ``start_frame`` and
#: ``frame_limit`` -- a contiguous range -- and has no stride: there is no way
#: to ask it for every second frame. The chain would also have to be told that
#: its input rate is the decimated one, because the muxer retimes against
#: ``expected_frame_count`` over the frames that actually entered.
#:
#: Named for the same reason as :data:`REQUIRES_SCALED_DECODE`: the shape is a
#: real answer to "what would fit", and a caller that has to start a job filters
#: it out with ``PolicyRequest.require_runnable`` rather than discovering at
#: runtime that the decoder ignored the plan.
REQUIRES_FRAME_DECIMATION = "strided_decode_stage"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceProbe:
    """What ffprobe says about the input, reduced to what the policy needs."""

    resolution: Resolution
    frame_rate: Fraction
    duration_s: float | None = None

    def __post_init__(self) -> None:
        if self.frame_rate <= 0:
            raise ChainPolicyError(
                f"source frame rate must be positive, got {self.frame_rate}"
            )

    @classmethod
    def from_media_info(cls, info, video_index: int | None = None) -> "SourceProbe":
        """Reduce a :class:`~sglang.srt.video_enhance.mux.MediaInfo`.

        The track is resolved the same way the enhance request resolves it, so
        the policy prices the track the chain will actually read.
        """
        from sglang.srt.video_enhance.mux import TrackSelection

        selection = TrackSelection(enhance_video_index=video_index)
        track = info.track(selection.resolve_video_index(info))
        rate = track.frame_rate()
        if rate is None:
            raise ChainPolicyError(
                f"video track {track.index} of the source declares no frame "
                "rate; the policy cannot decide an interpolation factor "
                "without it"
            )
        if not track.width or not track.height:
            raise ChainPolicyError(
                f"video track {track.index} declares no resolution "
                f"({track.width}x{track.height})"
            )
        return cls(
            resolution=Resolution(int(track.width), int(track.height)),
            frame_rate=rate,
            duration_s=info.duration_s,
        )


@dataclass(frozen=True)
class PolicyRequest:
    """The target, and the freedoms the caller grants the planner."""

    target: Resolution
    target_frame_rate: Fraction
    dtype: str = "fp16"
    sr_scale: int = 4
    rife_scale: float = 1.0
    rife_version: str = "4.6"
    streams_in_flight: int = 2
    #: Opt-in for :attr:`ChainMode.DECIMATE_RESYNTH`. Off by default and that
    #: is the point: a mode that discards real input frames must be asked for
    #: by name, never reached by a planner looking for something that fits.
    allow_decimation: bool = False
    #: Largest decimation factor to consider when the mode is allowed. 2 drops
    #: every second frame, 3 every second and third of each triple.
    max_decimation: int = 2
    #: Opt-in for pricing a stage from a pixel-count extrapolation instead of
    #: a measurement. Off by default: an unmeasured stage makes a candidate
    #: UNPRICEABLE, and a refusal that names the missing row is more useful
    #: than a plausible number.
    allow_estimates: bool = False
    #: Exclude shapes the executor cannot express today (see
    #: :data:`REQUIRES_SCALED_DECODE`). Off for the pure question "what would
    #: work on this rig"; on for a caller that has to start a job, so a plan
    #: is never admitted that would run at a geometry nobody asked for.
    require_runnable: bool = False
    #: Seconds of output the client may buffer before playback starts. 0.0 is
    #: the strict gate the target scenario asks for -- aggregate throughput at
    #: or above the input rate. A positive value lets a bounded shortfall be
    #: absorbed by lead time, and the required lead is reported either way.
    max_watch_ahead_s: float = 0.0

    def __post_init__(self) -> None:
        if self.target_frame_rate <= 0:
            raise ChainPolicyError(
                f"target frame rate must be positive, got {self.target_frame_rate}"
            )
        if self.sr_scale < 1:
            raise ChainPolicyError("sr_scale must be >= 1")
        if self.max_decimation < 1:
            raise ChainPolicyError("max_decimation must be >= 1")
        if self.max_watch_ahead_s < 0:
            raise ChainPolicyError("max_watch_ahead_s must not be negative")


@dataclass(frozen=True)
class PolicyInputs:
    """Everything measured that the policy prices against.

    ``rates`` is the pricing instrument: per (stage, card, resolution), a
    provenance-carrying cell from the shared cost library, which is the same
    table ``shard_plan`` weights a multi-card split with. ``frontier`` is the
    client-facing capability view built by
    :func:`~sglang.srt.video_enhance.probes.aggregate_frontier`; the policy
    reads the ``sr_only`` rows out of it to solve the pre-downscale entry
    ladder, and falls back to the rate table's own SR resolutions when no
    frontier was supplied.
    """

    rates: StageRateTable
    frontier: tuple[Mapping, ...] = ()
    #: Cards in play. Defaults to every card the rate table knows.
    cards: tuple[str, ...] = ()
    #: Per-card device budget in MiB. A card with no entry uses the tenant
    #: config's own ``budget_mib``, which is the single-card default.
    budgets_mib: Mapping[str, int] = field(default_factory=dict)

    def card_keys(self) -> tuple[str, ...]:
        return self.cards or tuple(self.rates.cards)

    @classmethod
    def from_samples(
        cls,
        samples: Iterable[Mapping],
        *,
        cards: Sequence[str] | None = None,
        budgets_mib: Mapping[str, int] | None = None,
        noise_floor_pct: float | None = None,
        source: str = "probe_samples",
    ) -> "PolicyInputs":
        """Build both views from one list of probe samples.

        The samples are the ``ProbeReport.samples`` shape, so a directory of
        P1 reports, a synthetic table in a test and a live measurement all
        enter through the same door.
        """
        from sglang.srt.video_enhance.probes import (
            Sample,
            aggregate_frontier,
            frontier_from_samples,
        )

        rows = list(samples)
        rates = stage_rates_from_samples(
            rows, source=source, noise_floor_pct=noise_floor_pct
        )
        typed = [
            Sample(
                post=str(row.get("post", "")),
                stage=str(row.get("stage", "")),
                card=str(row.get("card", "")),
                resolution=str(row.get("resolution", "")),
                dtype=str(row.get("dtype", "")),
                options=dict(row.get("options") or {}),
                ms_per_frame=float(row.get("ms_per_frame") or float("nan")),
                ms_stdev=float(row.get("ms_stdev") or 0.0),
                iterations=int(row.get("iterations") or 0),
            )
            for row in rows
            if row.get("stage") and row.get("resolution")
        ]
        frontier = tuple(aggregate_frontier(frontier_from_samples(typed)))
        return cls(
            rates=rates,
            frontier=frontier,
            cards=tuple(cards) if cards is not None else (),
            budgets_mib=dict(budgets_mib or {}),
        )

    @classmethod
    def from_probe_dir(
        cls,
        directory: Path | str | None,
        *,
        cards: Sequence[str] | None = None,
        budgets_mib: Mapping[str, int] | None = None,
    ) -> "PolicyInputs":
        """Read a tenant's configured measurement directory.

        A missing or empty directory is not an error here. It produces inputs
        with no rows, and every candidate then fails to price with a named
        absence -- which is the honest answer to "how fast is this rig" when
        nobody has measured it.
        """
        from sglang.srt.video_enhance.probes import load_probe_reports

        if directory is None:
            return cls(
                rates=StageRateTable(cells={}, source="no measurement directory"),
                cards=tuple(cards) if cards is not None else (),
                budgets_mib=dict(budgets_mib or {}),
            )
        samples, sources = load_probe_reports(directory)
        floors = [
            float(s["noise_floor_pct"])
            for s in sources
            if s.get("noise_floor_pct") is not None
        ]
        return cls.from_samples(
            [vars(s) for s in samples],
            cards=cards,
            budgets_mib=budgets_mib,
            noise_floor_pct=max(floors) if floors else None,
            source=f"probe reports under {directory}",
        )


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One chain shape that could reach the target, before it is priced."""

    mode: ChainMode
    request: ChainRequest
    #: Input frames kept: 1 keeps every frame, 2 keeps every second one.
    decimation: int = 1
    #: Resolution the input is Lanczos-resized down to before the chain.
    #: None for every mode but :attr:`ChainMode.PRE_DOWNSCALE`.
    pre_downscale_from: Resolution | None = None
    #: Executor capabilities this shape needs and M2 may not have.
    requires: tuple[str, ...] = ()

    @property
    def tier(self) -> int:
        return QUALITY_TIER[self.mode]

    @property
    def detail_loss(self) -> float:
        """Fraction of the source's pixels thrown away before SR sees them."""
        if self.pre_downscale_from is None:
            return 0.0
        return 1.0 - self.request.source.pixels / self.pre_downscale_from.pixels

    @property
    def quality_cost(self) -> tuple[int, float, float]:
        """What this shape costs the viewer, as a sort key. Lower is better.

        Three terms rather than one, because they are not interchangeable: the
        tier separates the kinds of loss, and within a tier the ladder is
        ordered by how much is lost. Without the middle term the pre-downscale
        ladder would be ranked by throughput, which picks the *smallest* entry
        point -- the one that discards the most detail -- whenever several fit.
        """
        return (self.tier, self.detail_loss, self.discarded_input_fraction)

    @property
    def discarded_input_fraction(self) -> float:
        """Fraction of the source's own frames that never reach the chain."""
        return 0.0 if self.decimation <= 1 else (self.decimation - 1) / self.decimation

    @property
    def synthetic_output_fraction(self) -> float:
        """Fraction of delivered frames RIFE invented rather than enhanced."""
        m = self.request.fps_multiplier
        return 0.0 if m <= 1 else (m - 1) / m

    def quality_note(self) -> str:
        """The one sentence a client is owed about what this shape costs."""
        if self.mode is ChainMode.DECIMATE_RESYNTH:
            kept = f"every {self.decimation}{_ordinal_suffix(self.decimation)} frame"
            return (
                f"{self.discarded_input_fraction:.0%} of the source frames are "
                f"discarded (only {kept} is enhanced) and "
                f"{self.synthetic_output_fraction:.0%} of the delivered frames are "
                "interpolated rather than enhanced source content"
            )
        if self.mode is ChainMode.PRE_DOWNSCALE:
            return (
                f"the input is Lanczos-resized from {self.pre_downscale_from} down "
                f"to {self.request.source} before super-resolution, so detail "
                "above that entry point is discarded before SR can use it"
            )
        if self.mode is ChainMode.RIFE_ONLY:
            return (
                "the source is already at or above the target resolution, so no "
                "super-resolution is applied and no source detail is lost"
            )
        return (
            f"every source frame is super-resolved; "
            f"{self.synthetic_output_fraction:.0%} of the delivered frames are "
            "interpolated, which is what the requested frame-rate multiplier means"
        )

    def as_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "source": str(self.request.source),
            "target": str(self.request.target),
            "fps_multiplier": self.request.fps_multiplier,
            "enable_sr": self.request.enable_sr,
            "sr_scale": self.request.sr_scale,
            "decimation": self.decimation,
            "pre_downscale_from": (
                None if self.pre_downscale_from is None else str(self.pre_downscale_from)
            ),
            "discarded_input_fraction": round(self.discarded_input_fraction, 4),
            "synthetic_output_fraction": round(self.synthetic_output_fraction, 4),
            "detail_loss": round(self.detail_loss, 4),
            "requires": list(self.requires),
            "quality_note": self.quality_note(),
        }


def _ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def sr_entry_points(inputs: PolicyInputs) -> tuple[Resolution, ...]:
    """SR input resolutions somebody measured, largest first.

    This is the pre-downscale ladder, and it is the reason that mode's factor
    is *solved* rather than picked: the only entry points on offer are the
    ones the frontier can price. The ``sr_only`` rows of
    :func:`~sglang.srt.video_enhance.probes.aggregate_frontier` are preferred
    because they are the published capability view; the rate table's own SR
    resolutions are folded in as well, so a deployment that has a rate table
    but no frontier export still gets a ladder.
    """
    found: set[Resolution] = set()
    for row in inputs.frontier:
        if row.get("configuration") != "sr_only":
            continue
        try:
            found.add(Resolution.parse(str(row.get("resolution", ""))))
        except (ValueError, TypeError):
            continue
    for text in inputs.rates.resolutions(StageKind.SR.value):
        try:
            found.add(Resolution.parse(text))
        except (ValueError, TypeError):
            continue
    return tuple(sorted(found, key=lambda r: -r.pixels))


def _multiplier(probe: SourceProbe, request: PolicyRequest) -> int:
    ratio = Fraction(request.target_frame_rate) / Fraction(probe.frame_rate)
    if ratio.denominator != 1 or ratio < 1:
        raise ChainPolicyError(
            f"target rate {request.target_frame_rate} is not a whole multiple of "
            f"the source rate {probe.frame_rate} (ratio {ratio}); interpolation "
            "produces an integer number of frames per source pair, so the policy "
            "will not round it. Ask for a target rate that is a whole multiple, "
            "or retime the source."
        )
    return int(ratio)


def _aspect_matches(a: Resolution, b: Resolution) -> bool:
    return (
        abs(a.width / a.height - b.width / b.height)
        <= ASPECT_TOLERANCE * a.width / a.height
    )


def candidate_chains(
    probe: SourceProbe, request: PolicyRequest, inputs: PolicyInputs
) -> tuple[Candidate, ...]:
    """Every chain shape that could reach the target, unpriced.

    Geometry decides which shapes exist at all -- a ``RIFE_ONLY`` chain on a
    source below the target is not a cheaper option, it is a chain
    ``build_chain`` refuses to build, because a Lanczos upscale cannot stand
    in for SR. Pricing then decides between the shapes that do exist.
    """
    multiplier = _multiplier(probe, request)
    source = probe.resolution
    target = request.target
    base: list[Candidate] = []
    rejected: list[str] = []

    def chain_request(src: Resolution, mult: int, *, sr: bool) -> ChainRequest | None:
        try:
            return ChainRequest(
                source=src,
                target=target,
                fps_multiplier=mult,
                dtype=request.dtype,
                enable_sr=sr,
                sr_scale=request.sr_scale if sr else 1,
                enable_resize=True,
                rife_scale=request.rife_scale,
                rife_version=request.rife_version,
                streams_in_flight=request.streams_in_flight,
            )
        except ChainError as exc:
            rejected.append(f"{src} -> {target} at x{mult}: {exc}")
            return None

    if source.width >= target.width and source.height >= target.height:
        # SR would upscale detail only for the resize to throw it away again.
        req = chain_request(source, multiplier, sr=False)
        if req is not None:
            base.append(Candidate(ChainMode.RIFE_ONLY, req))
    else:
        req = chain_request(source, multiplier, sr=True)
        if req is not None:
            base.append(Candidate(ChainMode.FULL, req))
        for entry in sr_entry_points(inputs):
            if entry.pixels >= source.pixels:
                continue
            if entry.width % 2 or entry.height % 2:
                continue
            if not _aspect_matches(source, entry):
                continue
            upscaled = entry.scaled(request.sr_scale)
            if upscaled.width < target.width or upscaled.height < target.height:
                # Downscaling this far would make SR unable to reach the
                # target at all, and the remaining gap would have to be closed
                # by a Lanczos *upscale*, which is what SR exists to avoid.
                continue
            entry_req = chain_request(entry, multiplier, sr=True)
            if entry_req is None:
                continue
            base.append(
                Candidate(
                    ChainMode.PRE_DOWNSCALE,
                    entry_req,
                    pre_downscale_from=source,
                    requires=(REQUIRES_SCALED_DECODE,),
                )
            )

    if not base:
        raise ChainPolicyError(
            "no chain shape can reach this target from this source: "
            + "; ".join(rejected or ["no candidate geometry was generated"])
        )

    out = list(base)
    if request.allow_decimation:
        for d in range(2, request.max_decimation + 1):
            for candidate in base:
                out.append(
                    Candidate(
                        ChainMode.DECIMATE_RESYNTH,
                        replace(
                            candidate.request, fps_multiplier=multiplier * d
                        ),
                        decimation=d,
                        pre_downscale_from=candidate.pre_downscale_from,
                        requires=candidate.requires + (REQUIRES_FRAME_DECIMATION,),
                    )
                )
    return tuple(out)


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StagePrice:
    """What one stage of one candidate costs on one card, and where that came from."""

    stage: str
    resolution: str
    #: Milliseconds contributed per frame *entering the chain*, already
    #: multiplied by how often this stage runs for such a frame.
    ms: float | None
    #: Times this stage runs per chain-input frame. Above 1 downstream of an
    #: interpolator, and above 1 for decode under decimation, because a
    #: decoder cannot skip the frames the policy is discarding.
    invocations: float
    provenance: Provenance
    source: str

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "resolution": self.resolution,
            "ms": None if self.ms is None else round(self.ms, 3),
            "invocations": round(self.invocations, 4),
            "provenance": self.provenance.value,
            "source": self.source,
        }


@dataclass(frozen=True)
class CardPrice:
    """One card's contribution to a candidate, or the reason it has none."""

    card: str
    #: Milliseconds of this card's time per frame entering the chain. None
    #: when a stage could not be priced or the card was not admitted.
    ms_per_chain_frame: float | None
    stages: tuple[StagePrice, ...]
    admitted: bool
    reserved_mib: int | None = None
    max_in_flight: int | None = None
    blocker: str | None = None

    @property
    def provenance(self) -> Provenance:
        return _worst(price.provenance for price in self.stages)

    @property
    def chain_fps(self) -> float:
        if not self.admitted or not self.ms_per_chain_frame:
            return 0.0
        return 1000.0 / self.ms_per_chain_frame

    def as_dict(self) -> dict:
        return {
            "card": self.card,
            "admitted": self.admitted,
            "ms_per_chain_frame": (
                None
                if self.ms_per_chain_frame is None
                else round(self.ms_per_chain_frame, 3)
            ),
            "chain_fps": round(self.chain_fps, 2),
            "reserved_mib": self.reserved_mib,
            "max_in_flight": self.max_in_flight,
            "provenance": self.provenance.value,
            "blocker": self.blocker,
            "stages": [s.as_dict() for s in self.stages],
        }


@dataclass(frozen=True)
class CandidatePrice:
    """A candidate scored against the frontier and the budget."""

    candidate: Candidate
    per_card: tuple[CardPrice, ...]
    #: Chain-input frames per second the cards deliver together.
    aggregate_chain_fps: float
    #: Chain-input frames per second the source demands. Under decimation this
    #: is the source rate divided by the decimation factor, which is the whole
    #: reason that mode helps.
    required_chain_fps: float
    #: Seconds of output that must be buffered before playback if the
    #: aggregate falls short. 0.0 when it does not.
    watch_ahead_s: float
    feasible: bool
    reason: str
    #: Chain stages nobody ever measured, so nobody priced. The same two
    #: colour conversions on every candidate; named so an absolute figure is
    #: read as the chain-stage figure it is.
    unpriced_stages: tuple[str, ...] = ()

    @property
    def provenance(self) -> Provenance:
        return _worst(card.provenance for card in self.per_card)

    @property
    def headroom(self) -> float:
        """Aggregate over required. 1.0 is exactly at the gate."""
        if self.required_chain_fps <= 0:
            return 0.0
        return self.aggregate_chain_fps / self.required_chain_fps

    def as_dict(self) -> dict:
        return {
            **self.candidate.as_dict(),
            "feasible": self.feasible,
            "provenance": self.provenance.value,
            "aggregate_chain_fps": round(self.aggregate_chain_fps, 2),
            "required_chain_fps": round(self.required_chain_fps, 2),
            "headroom": round(self.headroom, 3),
            "watch_ahead_s": round(self.watch_ahead_s, 2),
            "unpriced_stages": list(self.unpriced_stages),
            "reason": self.reason,
            "cards": [c.as_dict() for c in self.per_card],
        }


_PROVENANCE_ORDER = {Provenance.MEASURED: 0, Provenance.ESTIMATE: 1, Provenance.ABSENT: 2}


def _worst(values: Iterable[Provenance]) -> Provenance:
    worst = Provenance.MEASURED
    for value in values:
        if _PROVENANCE_ORDER[value] > _PROVENANCE_ORDER[worst]:
            worst = value
    return worst


def _stage_rate(
    inputs: PolicyInputs,
    stage: str,
    card: str,
    resolution: Resolution,
    *,
    allow_estimates: bool,
) -> Rate:
    """One measured cell, one extrapolation off a measured cell, or an absence.

    The extrapolation is linear in pixel count and that is a stated model, not
    a fudge: SRVGGNetCompact runs every convolution at input resolution, a
    separable Lanczos-3 pass is one tap loop per output pixel, RIFE's flow
    pyramids scale with the frame, and NVDEC/NVENC are per-macroblock. The P1
    table agrees to within a few percent where it can be checked (SR 960x540
    at 35.19 ms predicts 140.8 ms at 1920x1080 against a measured 146.29). It
    is still labelled ``estimate`` and still needs ``allow_estimates``,
    because a model that is right on this chain is not a measurement.
    """
    exact = inputs.rates.rate(stage, card, str(resolution))
    if not exact.is_absent:
        return exact
    if not allow_estimates:
        return exact
    measured: list[tuple[Resolution, float, str]] = []
    for text in inputs.rates.resolutions(stage):
        cell = inputs.rates.rate(stage, card, text)
        if cell.is_absent or cell.value is None:
            continue
        try:
            measured.append((Resolution.parse(text), float(cell.value), cell.source))
        except (ValueError, TypeError):
            continue
    if not measured:
        return exact
    basis, ms, source = min(
        measured, key=lambda row: abs(math.log(resolution.pixels / row[0].pixels))
    )
    scaled = ms * resolution.pixels / basis.pixels
    return Rate.estimate(
        scaled,
        f"linear-in-pixels extrapolation from {stage} on {card} at {basis} "
        f"({ms:.2f} ms, {source})",
        unit="ms",
        label=f"{stage}@{card}",
    )


def _price_card(
    chain: Chain,
    candidate: Candidate,
    card: str,
    inputs: PolicyInputs,
    config: TenantConfig,
    *,
    allow_estimates: bool,
) -> tuple[CardPrice, tuple[str, ...]]:
    """Price one candidate on one card: budget first, then the stopwatch."""
    budget_mib = int(inputs.budgets_mib.get(card, config.budget_mib))
    planned: PlannedJob | None = None
    blocker: str | None = None
    try:
        planned = plan_job(replace(config, budget_mib=budget_mib), candidate.request)
    except Exception as exc:  # noqa: BLE001 - every refusal is reported, not raised
        blocker = f"{type(exc).__name__}: {exc}".replace("\n", " ")

    prices: list[StagePrice] = []
    unpriced: list[str] = []
    total_ms = 0.0
    unpriceable = False

    # A pre-downscale pays one Lanczos pass at the *source* size before the
    # chain starts, and the chain itself is priced at the entry size. Charging
    # it to the resize stage at the source resolution is exactly what the
    # executor would run.
    if candidate.pre_downscale_from is not None:
        rate = _stage_rate(
            inputs,
            StageKind.RESIZE.value,
            card,
            candidate.pre_downscale_from,
            allow_estimates=allow_estimates,
        )
        pre_ms = None if rate.value is None else float(rate.value)
        prices.append(
            StagePrice(
                stage="pre_downscale_resize",
                resolution=str(candidate.pre_downscale_from),
                ms=pre_ms,
                invocations=1.0,
                provenance=rate.provenance,
                source=rate.source,
            )
        )
        if pre_ms is None:
            unpriceable = True
        else:
            total_ms += pre_ms

    stream_rate = 1.0  # frames entering the current stage per chain-input frame
    for spec in chain.stages:
        invocations_per_input, out_per_in = stage_stream_factors(spec)
        if spec.kind not in PRICED_STAGES:
            unpriced.append(spec.kind.value)
            stream_rate *= out_per_in
            continue
        # A decoder cannot skip the frames decimation discards: every source
        # frame is decoded and only the survivors go on. That is the honest
        # price of the mode and it is charged here rather than assumed away.
        decode_factor = candidate.decimation if spec.kind is StageKind.DECODE else 1
        if spec.kind is StageKind.RIFE:
            # A P1 RIFE cell is milliseconds per *interpolated frame*: it was
            # measured at multiplier 2, where a pair yields exactly one. At a
            # higher multiplier a pair yields ``arity_out`` of them and costs
            # accordingly. ``stage_stream_factors`` returns one invocation per
            # pair regardless, which is right for the multiplier-2 chains
            # ``shard_plan`` was written for and would under-price every
            # decimated candidate here -- the mode that raises the multiplier
            # is exactly the mode that must not be flattered.
            invocations = stream_rate * max(1, spec.arity_out)
        else:
            invocations = stream_rate * invocations_per_input * decode_factor
        # A pre-downscale moves every stage but the decoder to the entry size.
        # The decoder still reads the source, so it is priced at the source
        # size; pricing it at the chain's own ``source`` would credit the mode
        # with a decode saving it does not make.
        priced_res = (
            candidate.pre_downscale_from
            if spec.kind is StageKind.DECODE and candidate.pre_downscale_from
            else spec.in_res
        )
        rate = _stage_rate(
            inputs,
            spec.kind.value,
            card,
            priced_res,
            allow_estimates=allow_estimates,
        )
        stage_ms = None if rate.value is None else float(rate.value) * invocations
        prices.append(
            StagePrice(
                stage=spec.kind.value,
                resolution=str(priced_res),
                ms=stage_ms,
                invocations=invocations,
                provenance=rate.provenance,
                source=rate.source,
            )
        )
        if stage_ms is None:
            unpriceable = True
        else:
            total_ms += stage_ms
        stream_rate *= out_per_in

    if unpriceable and blocker is None:
        missing = [p.stage for p in prices if p.ms is None]
        blocker = (
            f"unpriceable on {card}: no measured rate for {', '.join(missing)}"
            + ("" if allow_estimates else " (allow_estimates is off)")
        )

    admitted = planned is not None and not unpriceable
    return (
        CardPrice(
            card=card,
            ms_per_chain_frame=None if unpriceable else total_ms,
            stages=tuple(prices),
            admitted=admitted,
            reserved_mib=None if planned is None else planned.reservation.total_mib,
            max_in_flight=None if planned is None else planned.max_in_flight,
            blocker=blocker,
        ),
        tuple(unpriced),
    )


def price_candidate(
    candidate: Candidate,
    probe: SourceProbe,
    request: PolicyRequest,
    inputs: PolicyInputs,
    config: TenantConfig,
) -> CandidatePrice:
    """Score one candidate: does it fit, and do the cards together carry it?

    Two independent gates, both reported with their numbers whichever way they
    come out. The budget gate is per card and comes from ``plan_job``, so it is
    the §6.2 reservation and not a second opinion about it. The throughput gate
    is the Regime-A aggregate over the cards that passed the budget gate.
    """
    chain = build_chain(candidate.request)
    per_card: list[CardPrice] = []
    unpriced: tuple[str, ...] = ()
    for card in inputs.card_keys():
        price, card_unpriced = _price_card(
            chain,
            candidate,
            card,
            inputs,
            config,
            allow_estimates=request.allow_estimates,
        )
        per_card.append(price)
        unpriced = unpriced or card_unpriced

    required = float(probe.frame_rate) / candidate.decimation
    admitted = {
        c.card: c.ms_per_chain_frame
        for c in per_card
        if c.admitted and c.ms_per_chain_frame is not None
    }
    if not admitted:
        blockers = "; ".join(
            f"{c.card}: {c.blocker or 'no rate'}" for c in per_card
        ) or "no cards were offered to the planner"
        return CandidatePrice(
            candidate=candidate,
            per_card=tuple(per_card),
            aggregate_chain_fps=0.0,
            required_chain_fps=required,
            watch_ahead_s=float("inf") if probe.duration_s else 0.0,
            feasible=False,
            reason=(
                f"{candidate.mode.value}: no card can run this chain. {blockers}"
            ),
            unpriced_stages=unpriced,
        )

    # ``playback_feasibility`` sums per-card rates and reports the lead a
    # shortfall needs. Its unit here is a frame *entering the chain* rather
    # than a delivered frame: under decimation the two differ by the
    # decimation factor, and the required rate below is divided by the same
    # factor, so the comparison stays in one unit throughout.
    verdict = playback_feasibility(
        per_card_ms_per_frame=admitted,
        target_fps=required,
        source_duration_s=probe.duration_s or 0.0,
        fps_multiplier=candidate.request.fps_multiplier,
        max_watch_ahead_s=request.max_watch_ahead_s,
    )
    # A shortfall is only absorbable when there is a lead budget to absorb it
    # into AND the source has a known length: ``playback_feasibility`` scales
    # the required lead by the duration, so an unknown duration reports 0 s of
    # lead for a real deficit. Treating that 0 as "fits" would turn a missing
    # ffprobe field into a green light, which is exactly the silent assumption
    # this module refuses to make.
    if verdict.realtime:
        feasible = True
        shortfall_note = ""
    elif request.max_watch_ahead_s <= 0:
        feasible = False
        shortfall_note = " and no watch-ahead budget was granted"
    elif not probe.duration_s:
        feasible = False
        shortfall_note = (
            " and the source declares no duration, so the required lead cannot "
            "be computed"
        )
    else:
        feasible = verdict.watch_ahead_s <= request.max_watch_ahead_s
        shortfall_note = ""
    dropped = [c for c in per_card if not c.admitted]
    note = (
        ""
        if not dropped
        else " (" + ", ".join(f"{c.card} excluded: {c.blocker}" for c in dropped) + ")"
    )
    if feasible:
        reason = (
            f"{candidate.mode.value}: {verdict.aggregate_fps:.1f} chain fps across "
            f"{len(admitted)} card(s) against {required:.1f} required"
            + (
                ""
                if verdict.realtime
                else f", absorbed by {verdict.watch_ahead_s:.1f} s of lead"
            )
            + note
        )
    else:
        reason = (
            f"{candidate.mode.value}: {verdict.aggregate_fps:.1f} chain fps across "
            f"{len(admitted)} card(s) is short of the {required:.1f} required; "
            f"closing the gap needs {verdict.watch_ahead_s:.0f} s of lead against "
            f"a {request.max_watch_ahead_s:.0f} s budget" + shortfall_note + note
        )
    return CandidatePrice(
        candidate=candidate,
        per_card=tuple(per_card),
        aggregate_chain_fps=verdict.aggregate_fps,
        required_chain_fps=required,
        watch_ahead_s=verdict.watch_ahead_s,
        feasible=feasible,
        reason=reason,
        unpriced_stages=unpriced,
    )


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainDecision:
    """What the policy chose, why, and what it looked at.

    ``as_dict`` is the client-visible form. It carries the mode, the one-line
    reason and the quality note unconditionally, because a client that is
    handed a decimated stream without being told is being lied to.
    """

    feasible: bool
    reason: str
    price: CandidatePrice | None
    considered: tuple[CandidatePrice, ...]
    planned: PlannedJob | None = None
    #: Cards the chosen candidate would run on, in the planner's order.
    cards: tuple[str, ...] = ()

    @property
    def mode(self) -> ChainMode | None:
        return None if self.price is None else self.price.candidate.mode

    @property
    def request(self) -> ChainRequest | None:
        return None if self.price is None else self.price.candidate.request

    @property
    def provenance(self) -> Provenance:
        return Provenance.ABSENT if self.price is None else self.price.provenance

    @property
    def requires(self) -> tuple[str, ...]:
        return () if self.price is None else self.price.candidate.requires

    @property
    def runnable(self) -> bool:
        """Feasible *and* expressible by the executor that exists today."""
        return self.feasible and not self.requires

    def as_dict(self) -> dict:
        chosen = None if self.price is None else self.price.as_dict()
        return {
            "feasible": self.feasible,
            "runnable": self.runnable,
            "mode": None if self.mode is None else self.mode.value,
            "reason": self.reason,
            "provenance": self.provenance.value,
            "requires": list(self.requires),
            "cards": list(self.cards),
            "chosen": chosen,
            "considered": [p.as_dict() for p in self.considered],
        }


def choose_chain(
    probe: SourceProbe,
    request: PolicyRequest,
    inputs: PolicyInputs,
    config: TenantConfig,
) -> ChainDecision:
    """Generate, price and rank the chain shapes; pick one or refuse.

    The ranking is :attr:`Candidate.quality_cost` first and ``-headroom``
    second: the shape that costs the viewer least wins, and throughput only
    decides between shapes that cost the same. That is what makes ``FULL``
    automatic rather than configured -- it is tier 0 with no losses, so as
    soon as it is feasible nothing below it can outrank it, however much
    faster the cheaper shape would be.
    """
    try:
        candidates = candidate_chains(probe, request, inputs)
    except ChainPolicyError as exc:
        # The question itself is unanswerable -- a rate ratio that is not a
        # whole number, or a source and target that leave nothing to do. That
        # is a refusal with a reason, not a traceback at the client.
        return ChainDecision(
            feasible=False,
            reason=str(exc),
            price=None,
            considered=(),
            cards=inputs.card_keys(),
        )
    priced: list[CandidatePrice] = []
    for candidate in candidates:
        if request.require_runnable and candidate.requires:
            priced.append(
                CandidatePrice(
                    candidate=candidate,
                    per_card=(),
                    aggregate_chain_fps=0.0,
                    required_chain_fps=float(probe.frame_rate) / candidate.decimation,
                    watch_ahead_s=0.0,
                    feasible=False,
                    reason=(
                        f"{candidate.mode.value}: excluded because require_runnable "
                        f"is set and this shape needs "
                        f"{', '.join(candidate.requires)}"
                    ),
                )
            )
            continue
        try:
            priced.append(price_candidate(candidate, probe, request, inputs, config))
        except ChainError as exc:
            # A shape geometry forbids. Recorded rather than dropped, so a
            # refusal can say why a mode the caller expected never competed.
            priced.append(
                CandidatePrice(
                    candidate=candidate,
                    per_card=(),
                    aggregate_chain_fps=0.0,
                    required_chain_fps=float(probe.frame_rate) / candidate.decimation,
                    watch_ahead_s=0.0,
                    feasible=False,
                    reason=f"{candidate.mode.value}: not a buildable chain: {exc}",
                )
            )
    ordered = tuple(
        sorted(
            priced,
            key=lambda p: (
                *p.candidate.quality_cost,
                -p.headroom,
                str(p.candidate.request.source),
            ),
        )
    )
    winners = [p for p in ordered if p.feasible]
    if not winners:
        return ChainDecision(
            feasible=False,
            reason=_refusal(probe, request, ordered),
            price=None,
            considered=ordered,
            cards=inputs.card_keys(),
        )
    best = winners[0]
    admitted = tuple(c.card for c in best.per_card if c.admitted)
    planned = None
    if admitted:
        budget = int(inputs.budgets_mib.get(admitted[0], config.budget_mib))
        planned = plan_job(replace(config, budget_mib=budget), best.candidate.request)
    reason = best.reason + ". " + best.candidate.quality_note()
    if best.provenance is Provenance.ESTIMATE:
        reason += (
            ". Priced from a linear-in-pixels extrapolation, not a measurement "
            "(allow_estimates was set)"
        )
    if best.candidate.requires:
        reason += (
            ". Not runnable on the M2 executor: needs "
            + ", ".join(best.candidate.requires)
        )
    return ChainDecision(
        feasible=True,
        reason=reason,
        price=best,
        considered=ordered,
        planned=planned,
        cards=admitted,
    )


def _refusal(
    probe: SourceProbe, request: PolicyRequest, priced: Sequence[CandidatePrice]
) -> str:
    """A refusal that names every number that made it one."""
    head = (
        f"no chain reaches {request.target} at {request.target_frame_rate} fps from "
        f"a {probe.resolution} at {probe.frame_rate} fps source"
    )
    if not priced:
        return (
            head
            + ": no candidate chain could even be generated. Check that the target "
            "rate is a whole multiple of the source rate."
        )
    lines = [f"{head}. Candidates considered:"]
    for entry in priced:
        lines.append(f"  - {entry.reason}")
    if not request.allow_decimation:
        lines.append(
            "  - decimate_resynth was not considered: it discards source frames "
            "and must be requested by name (allow_decimation)"
        )
    if not request.allow_estimates:
        lines.append(
            "  - a candidate refused as unpriceable would be priced by "
            "extrapolation with allow_estimates, and labelled estimate"
        )
    return "\n".join(lines)


def require_chain(
    probe: SourceProbe,
    request: PolicyRequest,
    inputs: PolicyInputs,
    config: TenantConfig,
) -> ChainDecision:
    """:func:`choose_chain`, but a refusal is raised instead of returned.

    For the callers that cannot act on an infeasible decision -- the HTTP
    surface turns this into a 422 whose body is the refusal text, numbers and
    all.
    """
    decision = choose_chain(probe, request, inputs, config)
    if not decision.feasible:
        raise ChainPolicyError(decision.reason)
    return decision
