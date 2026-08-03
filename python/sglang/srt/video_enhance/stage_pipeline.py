# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Stage-pipeline pricing for the enhance chain (#457, desk half).

``chain_policy`` prices Regime A: every card runs the whole chain over its own
stretch of the timeline, so the per-card rates *add* and the cost of one frame
is the serial sum of its stages. That is the right model for the chunk
executor and it is what §16.2 of the ledger describes.

This module prices Regime B: the stages are *placed* on cards and every frame
walks the whole rig. Three things change, and each of them is why the regime
can win where Regime A cannot:

*   **Throughput is bound by the busiest card, not by the serial sum.** A card
    that holds only the encode stage is idle while another card super-resolves,
    so the pipeline delivers one frame per ``max`` over the cards rather than
    one per ``sum`` over the stages.
*   **A stage boundary that crosses a card is a transfer, and it is priced.**
    This rig has no NVLink and no GPUDirect P2P (all PHB), so a raw-frame move
    is a host bounce: device-to-host over the sending card's link, then
    host-to-device over the receiving card's. Both halves are charged to the
    card whose link carries them. ``barlink`` BAR1 peer access is the named
    alternative and is the house default transport wherever a combination
    supports it -- but nobody has measured a BAR1 raw-frame move on this rig,
    so :func:`host_bounce_links` leaves it ABSENT rather than guessing, and a
    plan that asks for it is refused by name.
*   **A transfer that fits inside the receiving card's current step costs
    nothing.** User directive 2026-08-03, translated: the next raw data a card
    is to work on can be spilled onto that card shortly before the current
    step finishes, so there is no waiting time in between. That
    is the #125 double-buffer pattern applied to frames: a dedicated copy
    stream moves frame *n+1* while the compute stream works on frame *n*, so
    only the part of the transfer that does not fit inside that window is
    real. :func:`price_placement` charges ``max(0, transfer - window)``.

Two hard constraints the pricer enforces rather than reports:

*   **Co-residency.** SR and the tail resize must sit on one card. The SR
    output is a 7680x4320 fp16 frame -- 189.8 MiB, ~13.5 ms one way over x8 at
    the measured 13.70 GiB/s -- which exceeds the entire SR budget on its own.
*   **The x4 taboo.** A card on a x4 link is disqualified as an endpoint for a
    transfer above :attr:`CardProfile.max_transfer_mib`. On this rig that is
    NVML index 0. The constraint is expressed per card rather than as a
    hard-coded index, because NVML enumeration order is not stable across
    boots and the IdentityMap, not this module, resolves cards.

Everything here is arithmetic over a table. It imports no torch and runs on a
CPU-only host, like ``frame_math`` and ``chain``.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from sglang.srt.planner.cost_model import Provenance, Rate
from sglang.srt.video_enhance.frame_math import (
    MIB,
    R8K,
    PixelFormat,
    Resolution,
    frame_bytes,
)

__all__ = [
    "EIGHT_K_FP16_MIB",
    "UNPRICED_CHAIN_STAGES",
    "CardProfile",
    "PipelineError",
    "PipelinePrice",
    "PipelineStage",
    "Placement",
    "StageBoundary",
    "TransferKind",
    "TransferPrice",
    "absorbed_tail_rates",
    "best_placement",
    "enumerate_placements",
    "barlink_link",
    "compare_regimes",
    "fuse_stages",
    "host_bounce_links",
    "price_placement",
    "replicated_throughput",
    "stage_table",
]

#: 1 GiB/s expressed in MiB per millisecond, so a rate in GiB/s divides bytes
#: in MiB straight into milliseconds.
MIB_PER_MS_PER_GIBS = 1024.0 / 1000.0


class PipelineError(ValueError):
    """A placement that cannot be priced, or a constraint it violates."""


class TransferKind(str, enum.Enum):
    """How a raw frame crosses a card boundary."""

    #: Device-to-host on the sender, host-to-device on the receiver. The only
    #: path measured on this rig.
    HOST_BOUNCE = "host_bounce"
    #: barlink BAR1 peer window. The house default transport wherever the
    #: combination supports it; unmeasured for raw video frames, so a plan
    #: that names it is refused until somebody times it.
    BARLINK_BAR1 = "barlink_bar1"


# --------------------------------------------------------------------------
# Cards and links
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CardProfile:
    """One card as the pipeline sees it: a link rate and a transfer ceiling.

    ``key`` is the free-form card string the rate tables use. Device identity
    -- UUID, PCI BDF, negotiated link width -- is the IdentityMap's job; this
    module never resolves a device and only compares keys for equality.
    """

    key: str
    #: One-way host transfer rate over this card's own link.
    link_gib_s: Rate
    #: Payload size, in MiB, at or above which this card may not be a transfer
    #: endpoint. ``None`` means no ceiling. On this rig the x4 card gets
    #: :data:`EIGHT_K_FP16_MIB`, which is the payload class ticket V
    #: disqualified it for by name.
    max_transfer_mib: float | None = None

    def transfer_ms(self, payload_mib: float) -> Rate:
        """One-way cost of moving ``payload_mib`` over this card's link."""
        if self.link_gib_s.is_absent:
            return Rate.absent(
                f"card {self.key!r} has no measured link rate, so a transfer "
                f"across it cannot be priced: {self.link_gib_s.source}",
                unit="ms",
                label=f"transfer@{self.key}",
            )
        ms = payload_mib / (float(self.link_gib_s.value) * MIB_PER_MS_PER_GIBS)
        return Rate(
            ms,
            self.link_gib_s.provenance,
            f"{payload_mib:.2f} MiB over {self.key} at "
            f"{self.link_gib_s.value:.2f} GiB/s ({self.link_gib_s.source})",
            unit="ms",
            label=f"transfer@{self.key}",
        )


#: The 13.70 GiB/s figure is the round-trip rate recorded in ticket V's
#: ``p1_5090.json`` and used there as a one-way price for the 189.8 MiB 8K
#: intermediate ("~13.5 ms one-way over x8"). It is reused with the same
#: meaning here so the two documents cannot disagree.
MEASURED_X8_GIB_S = 13.70
_X8_SOURCE = (
    "ticket V 2026-08-03, p1_5090.json H2D round-trip rate, used one-way as in "
    "RESULTS.md 'Link-width caveat'"
)


#: The 8K fp16 SR intermediate, 189.84 MiB. Ticket V measured its crossing at
#: ~13.5 ms one way over x8 and ~27.1 ms over x4, and disqualified the x4 card
#: (NVML index 0 on this rig) as an endpoint for it on the grounds that the
#: crossing alone exceeds the whole 25 ms SR budget. That is the taboo, and
#: this is the number it is stated in -- derived from the geometry rather than
#: typed in, so a future format change moves it automatically.
EIGHT_K_FP16_MIB = frame_bytes(R8K, PixelFormat.RGB_FP16) / MIB


def host_bounce_links(
    *,
    x8_cards: Sequence[str],
    x4_cards: Sequence[str] = (),
    x4_max_transfer_mib: float = EIGHT_K_FP16_MIB,
) -> tuple[CardProfile, ...]:
    """Card profiles for this rig's host-bounce topology.

    The x8 rate is measured. The x4 rate is *derived* by halving it, which is a
    stated model (PCIe lanes scale linearly in the regime that matters here)
    and is labelled ``estimate`` accordingly -- nobody ran a transfer benchmark
    on the x4 card. ``x4_max_transfer_mib`` defaults to
    :data:`EIGHT_K_FP16_MIB`, so the taboo is on by default and has to be
    lifted deliberately rather than remembered.
    """
    x8 = Rate.measured(MEASURED_X8_GIB_S, _X8_SOURCE, unit="GiB/s")
    x4 = Rate.estimate(
        MEASURED_X8_GIB_S / 2.0,
        f"half the measured x8 rate ({MEASURED_X8_GIB_S} GiB/s, {_X8_SOURCE}); "
        "no transfer benchmark was run on the x4 card",
        unit="GiB/s",
    )
    profiles = [CardProfile(key=key, link_gib_s=x8) for key in x8_cards]
    profiles += [
        CardProfile(key=key, link_gib_s=x4, max_transfer_mib=x4_max_transfer_mib)
        for key in x4_cards
    ]
    return tuple(profiles)


def barlink_link(key: str) -> CardProfile:
    """A card whose peer transport is barlink BAR1. Unpriced on this rig.

    barlink is the house default transport wherever the combination supports
    it, and a BAR1 peer window is exactly the mechanism that would remove the
    host bounce from a stage split. It has not been measured for raw video
    frames, so the profile carries an absence rather than a number, and any
    placement that needs a crossing on this card is refused with that text.
    """
    return CardProfile(
        key=key,
        link_gib_s=Rate.absent(
            "barlink BAR1 peer bandwidth for raw video frames is unmeasured; "
            "TICKET_460 registers it as a transport validation point",
            unit="GiB/s",
        ),
    )


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineStage:
    """One stage, its per-card cost, and what it hands on.

    ``ms`` is keyed by card and is already multiplied by ``invocations`` --
    that is, it is milliseconds of that card's time per frame *entering the
    chain*, the same unit ``chain_policy.CardPrice.ms_per_chain_frame`` sums.
    Keeping the unit identical is what lets the two regimes be compared
    directly at the end of a report.
    """

    name: str
    ms: Mapping[str, Rate]
    #: Geometry of the frame this stage emits, and how many of them per
    #: chain-input frame. ``None`` for the last stage, which hands nothing on.
    out_resolution: Resolution | None = None
    out_format: PixelFormat = PixelFormat.RGB_FP16
    out_frames: float = 1.0
    #: Stage names that must sit on the same card as this one.
    co_resident_with: tuple[str, ...] = ()

    def payload_mib(self) -> float:
        if self.out_resolution is None:
            return 0.0
        return frame_bytes(self.out_resolution, self.out_format) * self.out_frames / MIB

    def rate_on(self, card: str) -> Rate:
        cell = self.ms.get(card)
        if cell is not None:
            return cell
        return Rate.absent(
            f"stage {self.name!r} has no rate on card {card!r}; priced cards: "
            f"{sorted(self.ms) or 'none'}",
            unit="ms",
            label=f"{self.name}@{card}",
        )


@dataclass(frozen=True)
class StageBoundary:
    """A crossing between two consecutive stages placed on different cards."""

    from_stage: str
    to_stage: str
    from_card: str
    to_card: str
    payload_mib: float


@dataclass(frozen=True)
class TransferPrice:
    """One half of a crossing, charged to the card whose link carries it."""

    boundary: StageBoundary
    card: str
    direction: str  # "d2h" on the sender, "h2d" on the receiver
    raw_ms: float | None
    hidden_ms: float
    unhidden_ms: float | None
    provenance: Provenance
    source: str

    def as_dict(self) -> dict:
        return {
            "from_stage": self.boundary.from_stage,
            "to_stage": self.boundary.to_stage,
            "card": self.card,
            "direction": self.direction,
            "payload_mib": round(self.boundary.payload_mib, 2),
            "raw_ms": None if self.raw_ms is None else round(self.raw_ms, 3),
            "hidden_ms": round(self.hidden_ms, 3),
            "unhidden_ms": (
                None if self.unhidden_ms is None else round(self.unhidden_ms, 3)
            ),
            "provenance": self.provenance.value,
            "source": self.source,
        }


Placement = Mapping[str, str]  # stage name -> card key


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelinePrice:
    """A placement scored: throughput, the card that binds, and why."""

    placement: dict[str, str]
    #: Per card: compute ms + unhidden transfer ms, per chain-input frame.
    card_load_ms: dict[str, float]
    #: Per card, compute only.
    card_compute_ms: dict[str, float]
    transfers: tuple[TransferPrice, ...]
    #: The card whose load sets the pipeline's period.
    binding_card: str | None
    #: The single most expensive stage on the binding card.
    binding_stage: str | None
    #: Chain-input frames per second the pipeline delivers.
    throughput_fps: float
    feasible: bool
    reason: str
    provenance: Provenance = Provenance.MEASURED
    unpriced_stages: tuple[str, ...] = ()
    frames_in_flight: int = 0
    latency_s: float = 0.0

    @property
    def period_ms(self) -> float:
        return 0.0 if not self.card_load_ms else max(self.card_load_ms.values())

    def as_dict(self) -> dict:
        return {
            "placement": dict(self.placement),
            "feasible": self.feasible,
            "reason": self.reason,
            "provenance": self.provenance.value,
            "period_ms": round(self.period_ms, 3),
            "throughput_fps": round(self.throughput_fps, 3),
            "binding_card": self.binding_card,
            "binding_stage": self.binding_stage,
            "card_load_ms": {k: round(v, 3) for k, v in self.card_load_ms.items()},
            "card_compute_ms": {
                k: round(v, 3) for k, v in self.card_compute_ms.items()
            },
            "transfers": [t.as_dict() for t in self.transfers],
            "unpriced_stages": list(self.unpriced_stages),
            "frames_in_flight": self.frames_in_flight,
            "latency_s": round(self.latency_s, 3),
        }


def _refusal(placement: Placement, reason: str) -> PipelinePrice:
    return PipelinePrice(
        placement=dict(placement),
        card_load_ms={},
        card_compute_ms={},
        transfers=(),
        binding_card=None,
        binding_stage=None,
        throughput_fps=0.0,
        feasible=False,
        reason=reason,
        provenance=Provenance.ABSENT,
    )


def price_placement(
    stages: Sequence[PipelineStage],
    placement: Placement,
    cards: Sequence[CardProfile],
    *,
    prefetch_depth: int = 1,
    frames_in_flight: int = 0,
    max_latency_s: float | None = None,
    unpriced_stages: Sequence[str] = (),
) -> PipelinePrice:
    """Price one stage->card assignment.

    The period of the pipeline is the busiest card's load, and a card's load is
    its stages' compute plus the transfer halves its own link has to carry that
    prefetch could not hide. ``prefetch_depth`` is how many frames ahead the
    copy stream may run; ``0`` disables hiding entirely and prices every byte,
    which is the pessimistic bound worth having next to the optimistic one.

    ``frames_in_flight`` is the deep-buffering knob. It buys nothing in
    throughput -- the period is set by the binding card either way -- and costs
    latency, which is exactly the trade the user allowed (a few seconds of
    latency is acceptable, 2026-08-03). The bound is reported and, if
    ``max_latency_s`` is given, enforced.
    """
    by_key = {c.key: c for c in cards}
    by_name = {s.name: s for s in stages}
    for stage in stages:
        card = placement.get(stage.name)
        if card is None:
            return _refusal(placement, f"stage {stage.name!r} was not placed")
        if card not in by_key:
            return _refusal(
                placement,
                f"stage {stage.name!r} is placed on unknown card {card!r}; known "
                f"cards: {sorted(by_key)}",
            )

    # -- hard constraint 1: co-residency ---------------------------------
    for stage in stages:
        for partner in stage.co_resident_with:
            if partner not in by_name:
                return _refusal(
                    placement,
                    f"stage {stage.name!r} declares co-residency with unknown "
                    f"stage {partner!r}",
                )
            if placement[partner] != placement[stage.name]:
                return _refusal(
                    placement,
                    f"co-residency violated: {stage.name!r} is on "
                    f"{placement[stage.name]!r} and {partner!r} is on "
                    f"{placement[partner]!r}, but the intermediate between them "
                    "must not cross a card",
                )

    # -- compute ----------------------------------------------------------
    compute: dict[str, float] = {c.key: 0.0 for c in cards}
    per_stage_ms: dict[str, float] = {}
    provenance = Provenance.MEASURED
    for stage in stages:
        card = placement[stage.name]
        rate = stage.rate_on(card)
        if rate.is_absent:
            return _refusal(
                placement,
                f"unpriceable: {rate.source}",
            )
        if rate.provenance is Provenance.ESTIMATE:
            provenance = Provenance.ESTIMATE
        compute[card] += float(rate.value)
        per_stage_ms[stage.name] = float(rate.value)

    # -- transfers --------------------------------------------------------
    transfers: list[TransferPrice] = []
    transfer_load: dict[str, float] = {c.key: 0.0 for c in cards}
    for left, right in zip(stages, stages[1:]):
        from_card = placement[left.name]
        to_card = placement[right.name]
        if from_card == to_card:
            continue
        payload = left.payload_mib()
        boundary = StageBoundary(
            from_stage=left.name,
            to_stage=right.name,
            from_card=from_card,
            to_card=to_card,
            payload_mib=payload,
        )
        # -- hard constraint 2: the x4 taboo ------------------------------
        for endpoint in (from_card, to_card):
            ceiling = by_key[endpoint].max_transfer_mib
            if ceiling is not None and payload >= ceiling:
                return _refusal(
                    placement,
                    f"transport taboo: the {left.name}->{right.name} boundary "
                    f"moves {payload:.1f} MiB and card {endpoint!r} is "
                    f"disqualified as an endpoint above {ceiling:.1f} MiB "
                    "(narrow link). Keep the two stages co-resident or place "
                    "the boundary on a wider-linked card",
                )
        for card_key, direction in ((from_card, "d2h"), (to_card, "h2d")):
            profile = by_key[card_key]
            rate = profile.transfer_ms(payload)
            if rate.is_absent:
                return _refusal(placement, f"unpriceable transfer: {rate.source}")
            raw = float(rate.value)
            # The window a copy stream can hide this transfer behind is the
            # compute the same card is already doing for the frames it holds.
            # Depth 0 hides nothing; depth n overlaps n steps of compute.
            window = compute[card_key] * max(0, prefetch_depth)
            hidden = min(raw, window)
            unhidden = raw - hidden
            # A transfer priced from an estimated link rate only degrades the
            # verdict's provenance if some of it is actually paid. A crossing
            # that prefetch hides completely contributes exactly zero
            # milliseconds to the period, so an estimate in its rate cannot
            # move the answer and must not be reported as if it had.
            if unhidden > 0.0 and rate.provenance is Provenance.ESTIMATE:
                provenance = Provenance.ESTIMATE
            transfers.append(
                TransferPrice(
                    boundary=boundary,
                    card=card_key,
                    direction=direction,
                    raw_ms=raw,
                    hidden_ms=hidden,
                    unhidden_ms=unhidden,
                    provenance=rate.provenance,
                    source=rate.source,
                )
            )
            transfer_load[card_key] += unhidden

    load = {key: compute[key] + transfer_load[key] for key in compute}
    active = {k: v for k, v in load.items() if v > 0.0}
    if not active:
        return _refusal(placement, "no card carries any work under this placement")
    binding_card = max(active, key=lambda k: active[k])
    on_binding = [s.name for s in stages if placement[s.name] == binding_card]
    binding_stage = (
        max(on_binding, key=lambda name: per_stage_ms[name]) if on_binding else None
    )
    period = active[binding_card]
    throughput = 1000.0 / period

    latency_s = 0.0 if throughput <= 0 else frames_in_flight / throughput
    feasible = True
    reason = (
        f"pipeline period {period:.2f} ms set by card {binding_card!r} "
        f"({throughput:.2f} chain fps); the most expensive stage on it is "
        f"{binding_stage!r} at {per_stage_ms.get(binding_stage, 0.0):.2f} ms"
    )
    if max_latency_s is not None and latency_s > max_latency_s:
        feasible = False
        reason = (
            f"{frames_in_flight} frames in flight at {throughput:.2f} fps is "
            f"{latency_s:.2f} s of latency, above the {max_latency_s:.2f} s bound. "
            + reason
        )
    return PipelinePrice(
        placement=dict(placement),
        card_load_ms=load,
        card_compute_ms=dict(compute),
        transfers=tuple(transfers),
        binding_card=binding_card,
        binding_stage=binding_stage,
        throughput_fps=throughput,
        feasible=feasible,
        reason=reason,
        provenance=provenance,
        unpriced_stages=tuple(unpriced_stages),
        frames_in_flight=frames_in_flight,
        latency_s=latency_s,
    )


def enumerate_placements(
    stages: Sequence[PipelineStage], cards: Sequence[CardProfile]
) -> Iterable[dict[str, str]]:
    """Every stage->card assignment. Small by construction: five stages over
    three cards is 243 combinations, so an exhaustive sweep is cheaper than any
    heuristic and cannot miss the optimum of the model it is sweeping."""
    keys = [c.key for c in cards]
    names = [s.name for s in stages]
    for combo in itertools.product(keys, repeat=len(names)):
        yield dict(zip(names, combo))


def best_placement(
    stages: Sequence[PipelineStage],
    cards: Sequence[CardProfile],
    **kwargs,
) -> tuple[PipelinePrice | None, tuple[PipelinePrice, ...]]:
    """Highest-throughput feasible placement, and every one that was rejected.

    Returns ``(best, refusals)``. ``refusals`` carries the reason each rejected
    placement failed, deduplicated by reason so a caller gets the *kinds* of
    refusal rather than 200 copies of the same sentence.
    """
    best: PipelinePrice | None = None
    seen_reasons: dict[str, PipelinePrice] = {}
    for placement in enumerate_placements(stages, cards):
        price = price_placement(stages, placement, cards, **kwargs)
        if not price.feasible:
            seen_reasons.setdefault(price.reason, price)
            continue
        if best is None or price.throughput_fps > best.throughput_fps:
            best = price
    return best, tuple(seen_reasons.values())


# --------------------------------------------------------------------------
# Regime comparison
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeComparison:
    """Pipeline against whole-chain replication, in one unit."""

    pipeline_fps: float
    replicated_fps: float
    pipeline_provenance: Provenance
    replicated_provenance: Provenance
    #: Cards whose replicated column had an absent cell, so their contribution
    #: is missing from ``replicated_fps`` and that figure is an upper bound.
    replicated_gaps: tuple[str, ...] = ()
    note: str = ""

    @property
    def winner(self) -> str:
        return "pipeline" if self.pipeline_fps >= self.replicated_fps else "replicated"

    def as_dict(self) -> dict:
        return {
            "pipeline_fps": round(self.pipeline_fps, 3),
            "replicated_fps": round(self.replicated_fps, 3),
            "pipeline_provenance": self.pipeline_provenance.value,
            "replicated_provenance": self.replicated_provenance.value,
            "replicated_gaps": list(self.replicated_gaps),
            "winner": self.winner,
            "note": self.note,
        }


def replicated_throughput(
    stages: Sequence[PipelineStage],
    cards: Sequence[CardProfile],
    *,
    omit_absent_stages: bool = False,
) -> tuple[float, Provenance, tuple[str, ...]]:
    """Regime A for the same table: reciprocal sum over whole-chain cards.

    Two readings of a card with an absent stage rate, and the caller has to say
    which one it wants because they answer different questions:

    ``omit_absent_stages=False`` (default)
        The card cannot be shown to run the whole chain, so it contributes
        nothing and its name goes into the gap list. This is the *lower* bound
        on the regime.
    ``omit_absent_stages=True``
        The absent term is dropped from that card's serial sum and the rest is
        kept, which makes the card look faster than it is and the aggregate an
        explicit **upper** bound. This is the reading ticket V's RESULTS.md
        used to reach 18.30 src-fps, and it is reproducible here only because
        it is named rather than assumed.
    """
    total_fps = 0.0
    gaps: list[str] = []
    provenance = Provenance.MEASURED
    for card in cards:
        serial = 0.0
        missing = False
        for stage in stages:
            rate = stage.rate_on(card.key)
            if rate.is_absent:
                missing = True
                continue
            if rate.provenance is Provenance.ESTIMATE:
                provenance = Provenance.ESTIMATE
            serial += float(rate.value)
        if missing:
            gaps.append(card.key)
            if not omit_absent_stages:
                continue
        if serial <= 0.0:
            continue
        total_fps += 1000.0 / serial
    if gaps:
        provenance = (
            Provenance.ESTIMATE if provenance is Provenance.MEASURED else provenance
        )
    return total_fps, provenance, tuple(gaps)


def compare_regimes(
    stages: Sequence[PipelineStage],
    cards: Sequence[CardProfile],
    *,
    omit_absent_stages: bool = False,
    **kwargs,
) -> tuple[PipelinePrice | None, RegimeComparison]:
    """Price both regimes off one stage table and say which one wins."""
    best, _refusals = best_placement(stages, cards, **kwargs)
    replicated, rep_prov, gaps = replicated_throughput(
        stages, cards, omit_absent_stages=omit_absent_stages
    )
    note = ""
    if gaps and omit_absent_stages:
        note = (
            "the replicated figure is an UPPER BOUND: "
            + ", ".join(gaps)
            + " have an absent stage rate whose term was dropped from their "
            "serial sum, so their real cost is higher and their real "
            "contribution lower"
        )
    elif gaps:
        note = (
            "the replicated figure is a LOWER BOUND: "
            + ", ".join(gaps)
            + " could not be shown to run the whole chain (a stage rate is "
            "absent) and contribute nothing to the sum"
        )
    return best, RegimeComparison(
        pipeline_fps=0.0 if best is None else best.throughput_fps,
        replicated_fps=replicated,
        pipeline_provenance=Provenance.ABSENT if best is None else best.provenance,
        replicated_provenance=rep_prov,
        replicated_gaps=gaps,
        note=note,
    )


def stage_table(
    rows: Iterable[tuple[str, Mapping[str, float | None]]],
    *,
    source: str,
    geometry: Mapping[str, tuple[Resolution | None, PixelFormat, float]],
    co_residency: Mapping[str, tuple[str, ...]] = (),  # type: ignore[assignment]
) -> tuple[PipelineStage, ...]:
    """Build stages from a plain ``{stage: {card: ms}}`` table.

    ``None`` for a cell means the measurement does not exist, and it becomes a
    named absence rather than being skipped -- the difference between "this
    card is slow at resize" and "nobody timed resize on this card" is the whole
    reason the 1080p@25 verdict has an upper bound on one regime and not on the
    other.
    """
    co = dict(co_residency or {})
    out: list[PipelineStage] = []
    for name, per_card in rows:
        cells: dict[str, Rate] = {}
        for card, ms in per_card.items():
            if ms is None:
                cells[card] = Rate.absent(
                    f"stage {name!r} on card {card!r} was never measured", unit="ms"
                )
            else:
                cells[card] = Rate.measured(float(ms), source, unit="ms")
        res, fmt, frames = geometry.get(name, (None, PixelFormat.RGB_FP16, 1.0))
        out.append(
            PipelineStage(
                name=name,
                ms=cells,
                out_resolution=res,
                out_format=fmt,
                out_frames=frames,
                co_resident_with=tuple(co.get(name, ())),
            )
        )
    return tuple(out)


#: Stages whose rate nobody ever took, carried through every report so an
#: absolute fps figure is read as the chain-stage figure it is. Same two colour
#: conversions ``chain_policy.PRICED_STAGES`` omits, for the same reason.
UNPRICED_CHAIN_STAGES: tuple[str, ...] = ("color_to_rgb", "color_to_yuv")


# --------------------------------------------------------------------------
# Fusion (#457)
# --------------------------------------------------------------------------


def _consecutive_slice(stages: Sequence[PipelineStage], members: Sequence[str]) -> int:
    names = [s.name for s in stages]
    if len(members) < 2:
        raise PipelineError("fusing needs at least two stages")
    for member in members:
        if member not in names:
            raise PipelineError(
                f"cannot fuse unknown stage {member!r}; table has {names}"
            )
    first = names.index(members[0])
    if names[first : first + len(members)] != list(members):
        raise PipelineError(
            f"stages {list(members)} are not consecutive in {names}; fusing "
            "non-adjacent stages would change the order frames pass through"
        )
    return first


def absorbed_tail_rates(
    stages: Sequence[PipelineStage],
    members: Sequence[str],
    *,
    tail_ms: float = 0.0,
    source: str,
) -> dict[str, Rate]:
    """Per-card cost of a fusion in which the LAST member is absorbed.

    That is what fusing the tail resize into the SR engine does: the resize
    stops being a separate pass over device memory and becomes part of the
    producer's last layer, so its own cost is replaced by ``tail_ms`` rather
    than added to the producer's. Everything ahead of it in the fused run is
    kept at its measured value.

    Every cell is an **estimate**, and deliberately so: the fused stage has
    never been timed. ``tail_ms=0.0`` is the optimistic end of the band and
    the fused engine cannot be cheaper than that, so a verdict computed from
    it is an upper bound on the fusion's benefit.

    A card whose *absorbed* stage cell was absent becomes priceable, which is
    not a trick: after the fusion there is no separate resize pass on that
    card to have measured. A card whose *kept* cell is absent stays absent.
    """
    _consecutive_slice(stages, members)
    kept = list(members[:-1])
    by_name = {s.name: s for s in stages}
    cards = sorted({card for name in kept for card in by_name[name].ms})
    out: dict[str, Rate] = {}
    for card in cards:
        total = 0.0
        missing: str | None = None
        for name in kept:
            rate = by_name[name].rate_on(card)
            if rate.is_absent:
                missing = rate.source
                break
            total += float(rate.value)
        if missing is not None:
            out[card] = Rate.absent(missing, unit="ms", label=f"fused@{card}")
            continue
        out[card] = Rate.estimate(
            total + tail_ms,
            f"{' + '.join(kept)} at {source}, with {members[-1]} absorbed into "
            f"the producer at {tail_ms:.3f} ms; the fused stage has not been "
            "timed",
            unit="ms",
            label=f"fused@{card}",
        )
    return out


def fuse_stages(
    stages: Sequence[PipelineStage],
    members: Sequence[str],
    *,
    name: str,
    ms: Mapping[str, Rate],
) -> tuple[PipelineStage, ...]:
    """Collapse consecutive stages into one, and repair the table around them.

    Fusion is not "set one stage's cost to zero". Three things move together,
    and a re-priced verdict that changes only the first is wrong:

    1.  **Cost.** One cell per card instead of several.
    2.  **Geometry.** The fused stage hands on what its *last* member handed
        on. For the SR pair that is the difference between emitting a
        189.84 MiB 8K frame and a 47.46 MiB 4K one -- which is the whole point
        of the fusion and also what decides whether the next boundary is
        legal.
    3.  **Constraints.** A co-residency requirement between two fused members
        is discharged: the intermediate they were kept together for no longer
        exists. A requirement between a member and an outside stage survives
        and is re-pointed at the fused name.
    """
    first = _consecutive_slice(stages, members)
    absorbed = set(members)
    tail = stages[first + len(members) - 1]
    fused = PipelineStage(
        name=name,
        ms=dict(ms),
        out_resolution=tail.out_resolution,
        out_format=tail.out_format,
        out_frames=tail.out_frames,
        co_resident_with=tuple(
            sorted(
                {
                    partner
                    for member in members
                    for partner in next(
                        s for s in stages if s.name == member
                    ).co_resident_with
                    if partner not in absorbed
                }
            )
        ),
    )
    out: list[PipelineStage] = []
    for index, stage in enumerate(stages):
        if index == first:
            out.append(fused)
            continue
        if stage.name in absorbed:
            continue
        partners = tuple(
            name if partner in absorbed else partner
            for partner in stage.co_resident_with
        )
        out.append(
            stage
            if partners == stage.co_resident_with
            else PipelineStage(
                name=stage.name,
                ms=stage.ms,
                out_resolution=stage.out_resolution,
                out_format=stage.out_format,
                out_frames=stage.out_frames,
                co_resident_with=tuple(dict.fromkeys(partners)),
            )
        )
    return tuple(out)
