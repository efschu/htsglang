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
from dataclasses import dataclass, field
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
    "split_shares",
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


#: stage name -> the card it runs on, or the cards it is REPLICATED across.
#:
#: A tuple value is stage-level replication (#484): that one stage's frames are
#: split over several cards while the rest of the chain stays where it is. A
#: bare string is the degenerate width-1 case and is what every pre-#484 caller
#: passes, so the two forms cost the same to write and price identically where
#: they mean the same thing.
Placement = Mapping[str, "str | Sequence[str]"]


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelinePrice:
    """A placement scored: throughput, the card that binds, and why."""

    placement: dict[str, "str | tuple[str, ...]"]
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
    #: For each replicated stage, the fraction of its frames each card takes
    #: (#484). Empty when nothing is replicated, so a report that never asked
    #: for replication reads exactly as it did before.
    stage_shares: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def period_ms(self) -> float:
        return 0.0 if not self.card_load_ms else max(self.card_load_ms.values())

    def as_dict(self) -> dict:
        return {
            "placement": {
                name: (value if isinstance(value, str) else list(value))
                for name, value in self.placement.items()
            },
            "stage_shares": {
                stage: {card: round(share, 4) for card, share in shares.items()}
                for stage, shares in self.stage_shares.items()
            },
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


def _normalise_placement(placement: Placement) -> dict[str, tuple[str, ...]]:
    """Every stage's cards as a tuple, whichever form the caller wrote."""
    out: dict[str, tuple[str, ...]] = {}
    for stage_name, value in placement.items():
        if isinstance(value, str):
            out[stage_name] = (value,)
        else:
            out[stage_name] = tuple(value)
    return out


def split_shares(
    fixed_ms: Mapping[str, float], stage_ms: Mapping[str, float]
) -> tuple[dict[str, float], float]:
    """Split one stage's frames over cards that already carry other work.

    ``fixed_ms`` is what each participating card owes per chain-input frame
    BEFORE this stage; ``stage_ms`` is what one frame of this stage costs on
    that card. Returns the per-card share of the stage's frames and the period
    those shares produce.

    This is a water-fill, and it is the honest reading of the per-family x
    per-phase law for a replicated stage: the cards differ on the resource the
    stage is bound by, so an equal split would hand the slow card the same
    number of frames as the fast one and let it set the period. The shares
    that minimise the period are the ones that bring every participating card
    to the SAME finishing time, except for cards whose fixed load already
    exceeds it -- those take nothing rather than a negative share.

    Exactly: with cards active above their own fixed load, the period P solves
    ``sum_i (P - fixed_i) / stage_i = 1``. ``g(P)`` is piecewise linear and
    increasing, its breakpoints are the ``fixed_i``, so the solution is found
    by walking the cards in fixed-load order and taking the first segment
    whose closed-form root lies inside it. No search, no tolerance.
    """
    if not stage_ms:
        raise PipelineError("a replicated stage must name at least one card")
    for card, ms in stage_ms.items():
        if ms <= 0.0:
            raise PipelineError(
                f"card {card!r} prices the replicated stage at {ms} ms; a "
                "non-positive per-frame cost has no share that means anything"
            )
    order = sorted(stage_ms, key=lambda c: fixed_ms.get(c, 0.0))
    active: list[str] = []
    period = 0.0
    for index, card in enumerate(order):
        active.append(card)
        inv = sum(1.0 / stage_ms[c] for c in active)
        weighted = sum(fixed_ms.get(c, 0.0) / stage_ms[c] for c in active)
        candidate = (1.0 + weighted) / inv
        # The segment ends where the next card's fixed load would join in.
        upper = fixed_ms.get(order[index + 1], 0.0) if index + 1 < len(order) else None
        if upper is None or candidate <= upper:
            period = candidate
            break
    shares = {
        card: max(0.0, (period - fixed_ms.get(card, 0.0)) / stage_ms[card])
        for card in stage_ms
    }
    total = sum(shares.values())
    if total <= 0.0:  # pragma: no cover - the walk above cannot produce this
        raise PipelineError("water-fill produced no share for any card")
    return {card: value / total for card, value in shares.items()}, period


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
    cards_for = _normalise_placement(placement)
    for stage in stages:
        assigned = cards_for.get(stage.name)
        if not assigned:
            return _refusal(placement, f"stage {stage.name!r} was not placed")
        for card in assigned:
            if card not in by_key:
                return _refusal(
                    placement,
                    f"stage {stage.name!r} is placed on unknown card {card!r}; known "
                    f"cards: {sorted(by_key)}",
                )
        if len(set(assigned)) != len(assigned):
            return _refusal(
                placement,
                f"stage {stage.name!r} names card {assigned!r} twice; a "
                "replicated stage takes each card at most once",
            )

    replicated = tuple(name for name, cs in cards_for.items() if len(cs) > 1)
    if len(replicated) > 1:
        # V1 limit, and it is a real one rather than a shortcut not taken:
        # with one replicated stage the split has a closed form (see
        # split_shares). With two, the shares interact through the cards they
        # share and the minimum-period problem becomes a linear program. A
        # water-fill applied twice would return AN answer, not the optimum,
        # and there is no way to tell the two apart from the output.
        return _refusal(
            placement,
            f"stage-level replication is priced for one stage at a time; "
            f"{sorted(replicated)} were all replicated. The joint split is a "
            "linear program rather than a water-fill and is not built",
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
            if stage.name in replicated or partner in replicated:
                return _refusal(
                    placement,
                    f"co-residency violated: {stage.name!r} and {partner!r} must "
                    "share a card, so neither can be replicated across cards",
                )
            if cards_for[partner] != cards_for[stage.name]:
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
    replicated_name = replicated[0] if replicated else None
    replicated_ms: dict[str, float] = {}
    for stage in stages:
        for card in cards_for[stage.name]:
            rate = stage.rate_on(card)
            if rate.is_absent:
                return _refusal(
                    placement,
                    f"unpriceable: {rate.source}",
                )
            if rate.provenance is Provenance.ESTIMATE:
                provenance = Provenance.ESTIMATE
            if stage.name == replicated_name:
                replicated_ms[card] = float(rate.value)
            else:
                compute[card] += float(rate.value)
                per_stage_ms[stage.name] = float(rate.value)

    # -- the replicated stage's split -------------------------------------
    #
    # Solved on COMPUTE, before transfers are priced. That ordering is exact
    # whenever prefetch hides the crossings -- which is the case the split is
    # worth taking at all -- and is an approximation when it does not, because
    # a share that shifts also shifts the bytes that cross behind it. The
    # alternative is a fixed point with no closed form; the reason is recorded
    # here rather than left to be rediscovered from a surprising number.
    stage_shares: dict[str, dict[str, float]] = {}
    if replicated_name is not None:
        shares, _fill_period = split_shares(compute, replicated_ms)
        stage_shares[replicated_name] = shares
        carried = 0.0
        for card, share in shares.items():
            compute[card] += share * replicated_ms[card]
            carried = max(carried, share * replicated_ms[card])
        per_stage_ms[replicated_name] = carried

    def _share_of(stage_name: str, card: str) -> float:
        shares = stage_shares.get(stage_name)
        if shares is None:
            return 1.0
        return shares.get(card, 0.0)

    # -- transfers --------------------------------------------------------
    transfers: list[TransferPrice] = []
    transfer_load: dict[str, float] = {c.key: 0.0 for c in cards}
    for left, right in zip(stages, stages[1:]):
        # At most one side of a boundary can be replicated (the V1 limit
        # above), so the frames of the replicated side each have exactly one
        # destination and the crossing is that side's share of the payload.
        crossings: list[tuple[str, str, float]] = []
        for from_card in cards_for[left.name]:
            for to_card in cards_for[right.name]:
                if from_card == to_card:
                    continue
                fraction = _share_of(left.name, from_card) * _share_of(
                    right.name, to_card
                )
                if fraction > 0.0:
                    crossings.append((from_card, to_card, fraction))
        for from_card, to_card, fraction in crossings:
            payload = left.payload_mib() * fraction
            boundary = StageBoundary(
                from_stage=left.name,
                to_stage=right.name,
                from_card=from_card,
                to_card=to_card,
                payload_mib=payload,
            )
            # -- hard constraint 2: the x4 taboo --------------------------
            for endpoint in (from_card, to_card):
                ceiling = by_key[endpoint].max_transfer_mib
                # The taboo is a per-MOVE ceiling, so it is judged on the
                # frame that actually crosses, not on the stage's whole
                # output: replication does not make an 8K frame smaller and a
                # share of the frames is still whole frames.
                if ceiling is not None and left.payload_mib() >= ceiling:
                    return _refusal(
                        placement,
                        f"transport taboo: the {left.name}->{right.name} boundary "
                        f"moves {left.payload_mib():.1f} MiB and card "
                        f"{endpoint!r} is "
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
    on_binding = [s.name for s in stages if binding_card in cards_for[s.name]]
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
        stage_shares=stage_shares,
    )


def enumerate_placements(
    stages: Sequence[PipelineStage],
    cards: Sequence[CardProfile],
    *,
    replicable: Sequence[str] = (),
) -> Iterable[dict[str, "str | tuple[str, ...]"]]:
    """Every stage->card assignment. Small by construction: five stages over
    three cards is 243 combinations, so an exhaustive sweep is cheaper than any
    heuristic and cannot miss the optimum of the model it is sweeping.

    ``replicable`` names stages that may additionally be spread over several
    cards (#484). Each named stage contributes its multi-card subsets on top of
    the single-card assignments, and since :func:`price_placement` prices one
    replicated stage at a time, the candidates are generated the same way: the
    base sweep, then for each replicable stage the sweep again with that stage
    widened. The sweep therefore grows additively rather than multiplicatively,
    and an empty ``replicable`` yields exactly what it always did.
    """
    keys = [c.key for c in cards]
    names = [s.name for s in stages]
    known = set(names)
    for name in replicable:
        if name not in known:
            raise PipelineError(
                f"cannot replicate unknown stage {name!r}; stages: {names}"
            )
    for combo in itertools.product(keys, repeat=len(names)):
        yield dict(zip(names, combo))
    for target in replicable:
        others = [n for n in names if n != target]
        widths = range(2, len(keys) + 1)
        subsets = [
            subset for width in widths for subset in itertools.combinations(keys, width)
        ]
        for combo in itertools.product(keys, repeat=len(others)):
            base = dict(zip(others, combo))
            for subset in subsets:
                yield {**base, target: subset}


def best_placement(
    stages: Sequence[PipelineStage],
    cards: Sequence[CardProfile],
    *,
    replicable: Sequence[str] = (),
    **kwargs,
) -> tuple[PipelinePrice | None, tuple[PipelinePrice, ...]]:
    """Highest-throughput feasible placement, and every one that was rejected.

    Returns ``(best, refusals)``. ``refusals`` carries the reason each rejected
    placement failed, deduplicated by reason so a caller gets the *kinds* of
    refusal rather than 200 copies of the same sentence.

    ``replicable`` opens stage-level replication for the named stages (#484).
    It is opt-in per call rather than always-on because replicating a stage has
    consequences outside this arithmetic -- an encode split across two cards
    produces two elementary streams that the executor has to interleave back
    into output order -- and a pricer must not quietly recommend a shape the
    executor cannot run.
    """
    best: PipelinePrice | None = None
    seen_reasons: dict[str, PipelinePrice] = {}
    for placement in enumerate_placements(stages, cards, replicable=replicable):
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
