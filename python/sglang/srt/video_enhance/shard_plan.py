# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Capacity-weighted chunk-shard planning across cards (DESIGN #333 §8.2).

Regime A holds the whole chain on one card. With several streams and several
cards the useful shape is Regime A *replicated*: one whole chain per card,
the source timeline distributed across cards. VSGAN reaches the same
granularity and stops there -- its multi-GPU mechanism is a hand-written
``core.std.SelectEvery(..., cycle=N, offsets=i)`` per device, a static,
equal-share, modulo-N frame round robin with no capacity weighting and no
way to derive the split ratio (ANALYSE_333_prior_art_vsgan.md §2). This
module distributes by measured stage rates instead, and it distributes
*contiguous chunks* rather than an interleave.

**Why chunks and not ``cycle=N``.** RIFE interpolates between adjacent
frames. Under a stride-N interleave the successor of every frame a card owns
lives on a different card, so every single interpolation crosses a card
boundary: the card must either re-run the whole pre-RIFE prefix (decode,
colour, SR, resize) on a frame it does not own, or drop the frame at the
seam. Under contiguous chunks only the two chunk edges have that problem,
and one overlapping frame per seam fixes it. The cost model below prices
both, which is what makes the comparison between the two an arithmetic
result rather than an assertion.

Three plan shapes share one cost model so the before/after measurement is two
calls in this module rather than a hand-rolled comparison:

*   :func:`capacity_weighted_plan` -- the proposal.
*   :func:`vsgan_style_modulo_plan` -- the prior-art baseline.
*   :func:`static_single_card_plan` -- the status quo, everything on one card.

:func:`predict_makespan` scores any of them from a measurement post P1 table
on CPU, with no device involved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from sglang.srt.video_enhance.chain import Chain, StageKind, StageSpec
from sglang.srt.video_enhance.frame_math import (
    MIB,
    Resolution,
    chain_reservation,
    max_in_flight_for_budget,
)

logger = logging.getLogger(__name__)


class ShardPlanError(ValueError):
    """A plan that cannot be built. Raised at plan time, never at run time."""


class MissingRateError(ShardPlanError):
    """The P1 table has no entry for a (stage, card, resolution) the chain needs."""


# --------------------------------------------------------------------------
# The P1 rate table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageRate:
    """One measurement post P1 cell: ms per invocation of one stage.

    The resolution is part of the key and not an annotation. Each stage in the
    chain runs at a different size -- SR at source resolution, resize at 4x
    source, RIFE at target -- so a table keyed only by card would silently
    price the SR stage at whatever resolution happened to be measured. The
    lookup therefore demands all three coordinates and fails loudly on a miss.

    ``ms`` is per invocation, not per source frame. Converting between the two
    is what :func:`stage_stream_factors` does, and it is not the identity for
    RIFE or for anything downstream of it.
    """

    stage: StageKind
    card: str
    resolution: Resolution
    ms: float

    def __post_init__(self) -> None:
        if not (self.ms > 0.0):
            raise ShardPlanError(
                f"stage rate must be positive: {self.stage.value} on {self.card} "
                f"at {self.resolution} was measured as {self.ms} ms. A zero or "
                "negative cell means the P1 run did not produce a number for "
                "this combination; planning against it would hand that card an "
                "unbounded share."
            )


class RateTable:
    """Measurement post P1, indexed by (stage, card, resolution)."""

    def __init__(self, rates: Iterable[StageRate]) -> None:
        self._cells: dict[tuple[StageKind, str, Resolution], float] = {}
        for rate in rates:
            self._cells[(rate.stage, rate.card, rate.resolution)] = rate.ms

    @classmethod
    def from_rows(
        cls, rows: Iterable[tuple[StageKind, str, Resolution, float]]
    ) -> "RateTable":
        return cls(StageRate(*row) for row in rows)

    @classmethod
    def from_stage_rates(
        cls, table, *, cards: Sequence[str] | None = None
    ) -> "RateTable":
        """Build from the shared cost library's :class:`StageRateTable`.

        This is the seam #348b left open. Until now this table was only ever
        constructed by hand in tests -- there was no path from a measured
        probe report into the planner at all, so every plan this module made
        on a real rig was made from numbers a human had typed. Going through
        ``cost_model`` means the video planner prices cards from the same
        provenance-tagged library the LLM and diffusion planners do, and an
        absent cell arrives as a named absence instead of as a hole that a
        ``KeyError`` reports three frames later.

        Absent cells are dropped rather than defaulted. The planner's own
        :class:`MissingRateError` then names the miss at the point a chain
        actually needs it, which is more useful than refusing to build a
        table over a cell no chain in this request touches.
        """
        rows: list[StageRate] = []
        by_value = {kind.value: kind for kind in StageKind}
        for (stage, card, resolution), cell in table.cells.items():
            if cell.is_absent:
                continue
            kind = by_value.get(stage)
            if kind is None:
                # A probe measured something this chain has no stage for.
                # Not an error: the grid is allowed to be wider than a chain.
                continue
            if cards is not None and card not in cards:
                continue
            rows.append(StageRate(kind, card, Resolution.parse(resolution), cell.value))
        return cls(rows)

    @property
    def cards(self) -> tuple[str, ...]:
        return tuple(sorted({card for _, card, _ in self._cells}))

    def ms(self, stage: StageKind, card: str, resolution: Resolution) -> float:
        try:
            return self._cells[(stage, card, resolution)]
        except KeyError:
            known = sorted(
                str(res) for st, cd, res in self._cells if st is stage and cd == card
            )
            raise MissingRateError(
                f"no P1 measurement for stage {stage.value} on card {card} at "
                f"{resolution}. Measured resolutions for that stage on that "
                f"card: {known or 'none'}. The planner will not extrapolate "
                "across resolutions."
            ) from None

    def __len__(self) -> int:
        return len(self._cells)


# --------------------------------------------------------------------------
# Arity: how many times a stage runs, and how much stream it emits
# --------------------------------------------------------------------------


def stage_stream_factors(spec: StageSpec) -> tuple[float, float]:
    """``(invocations per input frame, output frames per input frame)``.

    A per-frame sum over the stages is wrong as soon as RIFE is in the chain,
    in both directions:

    *   RIFE reads a *sliding pair*. ``F`` input frames yield ``F-1`` pairs,
        so asymptotically one invocation per input frame, not one per two.
    *   RIFE *amplifies the stream*. It passes the originals through and adds
        ``arity_out`` interpolated frames per pair, so everything downstream
        of it -- the RGB-to-YUV conversion and the encoder -- runs
        ``1 + arity_out`` times per source frame. At ``fps_multiplier=2``
        the encoder does twice the work of the decoder.
    """
    if spec.arity_in == 1:
        return 1.0, float(spec.arity_out)
    if spec.arity_in == 2:
        return 1.0, 1.0 + float(spec.arity_out)
    raise ShardPlanError(
        f"stage {spec.kind.value} declares arity_in={spec.arity_in}; the cost "
        "model handles per-frame stages and sliding-pair stages only."
    )


# --------------------------------------------------------------------------
# Per-card cost of the whole chain
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainCost:
    """What one source frame costs on one card, with the chain placed whole."""

    card: str
    per_stage_ms: tuple[tuple[StageKind, float], ...]
    #: Whole chain, per source frame, invocation-weighted.
    full_ms: float
    #: The stages an *overlap* frame has to go through: everything strictly
    #: before RIFE, because an overlap frame exists only to be RIFE's second
    #: input. Zero when the chain has no RIFE stage, which is also when a
    #: chunk boundary costs nothing.
    prefix_ms: float
    rate_scale: float = 1.0

    @property
    def frames_per_second(self) -> float:
        return 1000.0 / self.full_ms


def chain_cost(
    chain: Chain, rates: RateTable, card: str, *, rate_scale: float = 1.0
) -> ChainCost:
    """Price the whole chain on one card from the P1 table."""
    if rate_scale <= 0.0:
        raise ShardPlanError(f"rate_scale for card {card} must be positive")

    per_stage: list[tuple[StageKind, float]] = []
    stream_rate = 1.0  # frames entering the current stage, per source frame
    full_ms = 0.0
    prefix_ms = 0.0
    has_rife = StageKind.RIFE in chain.kinds
    seen_rife = False

    for spec in chain.stages:
        invocations_per_input, out_per_in = stage_stream_factors(spec)
        cell = rates.ms(spec.kind, card, spec.in_res) * rate_scale
        stage_ms = stream_rate * invocations_per_input * cell
        per_stage.append((spec.kind, stage_ms))
        full_ms += stage_ms
        if has_rife and not seen_rife and spec.kind is not StageKind.RIFE:
            prefix_ms += stage_ms
        if spec.kind is StageKind.RIFE:
            seen_rife = True
        stream_rate *= out_per_in

    return ChainCost(
        card=card,
        per_stage_ms=tuple(per_stage),
        full_ms=full_ms,
        prefix_ms=prefix_ms if has_rife else 0.0,
        rate_scale=rate_scale,
    )


# --------------------------------------------------------------------------
# Card availability and the plan-time headroom check
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CardAvailability:
    """A card offered to the planner, with its budget and its co-tenancy."""

    card: str
    #: This tenant's whole budget on that card, in bytes -- the ledger's
    #: ``reserved_bytes``. Not a fraction and not subject to a further ceiling.
    reserved_bytes: int
    has_llm_cotenant: bool = False
    #: Rate derate from measurement post P6 (AR decode next to the chain).
    #: 1.0 means no measured slowdown; it does not mean no co-tenant.
    rate_scale: float = 1.0


@dataclass(frozen=True)
class ReservationInputs:
    """The terms of ``reserved(C)`` that the chain itself does not fix."""

    engine_device_bytes: int = 0
    decoder_pool_depth: int = 4
    encoder_pool_depth: int = 4
    rife_measured_bytes_per_pair: int | None = None


def _reservation_kwargs(chain: Chain, inputs: ReservationInputs) -> dict[str, Any]:
    """Reservation terms read off the resolved chain, not off the request.

    The chain is what the executor will run, so the stage set and the SR scale
    are taken from its resolved geometry. A planner that priced a stage the
    chain does not contain, or at a scale the chain did not resolve, would
    admit a configuration that then does not fit.
    """
    sr = chain.stage(StageKind.SR)
    return {
        "source": chain.request.source,
        "target": chain.request.target,
        "dtype": chain.request.dtype,
        "engine_device_bytes": inputs.engine_device_bytes,
        "decoder_pool_depth": inputs.decoder_pool_depth,
        "encoder_pool_depth": inputs.encoder_pool_depth,
        "rife_measured_bytes_per_pair": inputs.rife_measured_bytes_per_pair,
        "with_rife": StageKind.RIFE in chain.kinds,
        "with_sr": sr is not None,
        "sr_scale": sr.out_res.width // sr.in_res.width if sr is not None else 1,
        "with_resize": StageKind.RESIZE in chain.kinds,
    }


def _required_bytes(chain: Chain, inputs: ReservationInputs, depth: int) -> int:
    return chain_reservation(
        streams_in_flight=depth, **_reservation_kwargs(chain, inputs)
    ).total_bytes


def check_headroom(
    chain: Chain, availability: CardAvailability, inputs: ReservationInputs
) -> int:
    """Largest in-flight depth this card's budget affords; raise if it is short.

    A plan that does not fit must fail here, not as an OOM once frames are
    already moving. The affordable depth comes from ``max_in_flight_for_budget``
    so the planner and the reservation formula cannot drift apart.
    """
    affordable = max_in_flight_for_budget(
        budget_bytes=availability.reserved_bytes,
        **_reservation_kwargs(chain, inputs),
    )
    wanted = chain.request.streams_in_flight
    if affordable < wanted:
        needed = _required_bytes(chain, inputs, wanted)
        raise ShardPlanError(
            f"card {availability.card} has no reservation headroom for this "
            f"chain: the request asks for {wanted} in-flight stream(s), which "
            f"needs {needed / MIB:.0f} MiB, but the card's reserved budget is "
            f"{availability.reserved_bytes / MIB:.0f} MiB and affords "
            f"{affordable}. Raise the budget for that card, lower "
            f"streams_in_flight, or drop the card from the plan."
        )
    return affordable


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


class PlanStrategy(str, Enum):
    CAPACITY_WEIGHTED = "capacity_weighted"
    VSGAN_MODULO = "vsgan_modulo"
    STATIC_SINGLE_CARD = "static_single_card"


@dataclass(frozen=True)
class ShardAssignment:
    """One card's slice of the source timeline."""

    card: str
    start: int
    stop: int
    #: 1 for a contiguous chunk. The modulo baseline uses ``stride = N cards``.
    stride: int = 1
    #: Frames before ``start`` this card must additionally pull through the
    #: pre-RIFE prefix so the seam interpolation has both of its inputs.
    lead_overlap: int = 0
    #: The same at the trailing edge. For a strided assignment the successor
    #: of *every* owned frame is off-card, so this equals the owned count --
    #: that is the price of the interleave, made explicit rather than hidden.
    tail_overlap: int = 0

    @property
    def owned_frames(self) -> int:
        return len(range(self.start, self.stop, self.stride))

    @property
    def overlap_frames(self) -> int:
        return self.lead_overlap + self.tail_overlap

    def frame_indices(self) -> range:
        return range(self.start, self.stop, self.stride)

    def describe(self) -> str:
        span = (
            f"[{self.start}:{self.stop}]"
            if self.stride == 1
            else f"[{self.start}::{self.stride}]"
        )
        return (
            f"{self.card:<12} {span:<16} {self.owned_frames:>7} frames "
            f"(+{self.overlap_frames} overlap)"
        )


@dataclass(frozen=True)
class CardCost:
    card: str
    owned_frames: int
    overlap_frames: int
    owned_ms: float
    overlap_ms: float

    @property
    def total_ms(self) -> float:
        return self.owned_ms + self.overlap_ms


@dataclass(frozen=True)
class MakespanPrediction:
    strategy: PlanStrategy
    total_frames: int
    per_card: tuple[CardCost, ...]

    @property
    def makespan_ms(self) -> float:
        return max((c.total_ms for c in self.per_card), default=0.0)

    @property
    def busiest_card(self) -> str:
        return max(self.per_card, key=lambda c: c.total_ms).card

    @property
    def overlap_frames(self) -> int:
        return sum(c.overlap_frames for c in self.per_card)

    @property
    def overlap_ms(self) -> float:
        return sum(c.overlap_ms for c in self.per_card)

    @property
    def idle_ms(self) -> dict[str, float]:
        """Per card, how long it waits for the busiest card once it is done."""
        span = self.makespan_ms
        return {c.card: span - c.total_ms for c in self.per_card}

    def render(self) -> str:
        lines = [f"{self.strategy.value}: makespan {self.makespan_ms:.1f} ms"]
        for cost in self.per_card:
            lines.append(
                f"  {cost.card:<12} {cost.owned_frames:>7} frames "
                f"{cost.owned_ms:>10.1f} ms  +{cost.overlap_ms:>7.1f} ms overlap "
                f"= {cost.total_ms:>10.1f} ms"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class ShardPlan:
    strategy: PlanStrategy
    chain: Chain
    total_frames: int
    assignments: tuple[ShardAssignment, ...]
    rate_scales: Mapping[str, float] = field(default_factory=dict)
    predicted_makespan_ms: float = 0.0
    rationale: str = ""

    @property
    def cards(self) -> tuple[str, ...]:
        return tuple(a.card for a in self.assignments)

    @property
    def owned_frames(self) -> int:
        return sum(a.owned_frames for a in self.assignments)

    @property
    def overlap_frames(self) -> int:
        return sum(a.overlap_frames for a in self.assignments)

    def describe(self) -> str:
        lines = [
            f"{self.strategy.value}: {self.total_frames} frames over "
            f"{len(self.assignments)} card(s), predicted makespan "
            f"{self.predicted_makespan_ms:.1f} ms"
        ]
        lines += [f"  {a.describe()}" for a in self.assignments]
        if self.rationale:
            lines.append(f"  {self.rationale}")
        return "\n".join(lines)


def predict_makespan(plan: ShardPlan, rates: RateTable) -> MakespanPrediction:
    """Score a plan of any shape against a P1 table. No device involved.

    Owned frames pay the whole chain; overlap frames pay only the pre-RIFE
    prefix, because an overlap frame is pulled in as RIFE's second input and
    is not encoded by this card.
    """
    costs: list[CardCost] = []
    for assignment in plan.assignments:
        cost = chain_cost(
            plan.chain,
            rates,
            assignment.card,
            rate_scale=plan.rate_scales.get(assignment.card, 1.0),
        )
        costs.append(
            CardCost(
                card=assignment.card,
                owned_frames=assignment.owned_frames,
                overlap_frames=assignment.overlap_frames,
                owned_ms=assignment.owned_frames * cost.full_ms,
                overlap_ms=assignment.overlap_frames * cost.prefix_ms,
            )
        )
    return MakespanPrediction(
        strategy=plan.strategy, total_frames=plan.total_frames, per_card=tuple(costs)
    )


# --------------------------------------------------------------------------
# Planners
# --------------------------------------------------------------------------


def _prepare(
    chain: Chain,
    rates: RateTable,
    cards: Sequence[CardAvailability],
    total_frames: int,
    inputs: ReservationInputs,
) -> tuple[list[CardAvailability], dict[str, ChainCost]]:
    if total_frames <= 0:
        raise ShardPlanError(f"total_frames must be positive, got {total_frames}")
    if not cards:
        raise ShardPlanError("no cards offered to the planner")
    seen = [c.card for c in cards]
    if len(set(seen)) != len(seen):
        raise ShardPlanError(f"a card is offered twice: {seen}")

    costs: dict[str, ChainCost] = {}
    for availability in cards:
        if availability.has_llm_cotenant and availability.rate_scale == 1.0:
            logger.warning(
                "card %s declares an LLM co-tenant but no P6 derate; planning "
                "it at full rate, which will over-promise if the co-tenant is "
                "active.",
                availability.card,
            )
        check_headroom(chain, availability, inputs)
        costs[availability.card] = chain_cost(
            chain, rates, availability.card, rate_scale=availability.rate_scale
        )
    return list(cards), costs


def _seam_overlaps(
    chunks: Sequence[tuple[str, int, int]], total_frames: int, per_seam: int
) -> list[tuple[int, int]]:
    """Lead/tail overlap per contiguous chunk. Interior seams only."""
    out: list[tuple[int, int]] = []
    for _card, start, stop in chunks:
        lead = per_seam if start > 0 else 0
        tail = per_seam if stop < total_frames else 0
        out.append((lead, tail))
    return out


def _weighted_boundaries(weights: Sequence[float], total_frames: int) -> list[int]:
    """Cumulative rounding, so the chunks tile the timeline exactly.

    The rule itself lives in the shared cost library (#348b) next to the
    largest-remainder rule the diffusion SP split uses, because the two do not
    always agree and a reader comparing a video shard against an SP shard has
    to be able to see which rounding produced which. Imported lazily: this
    module is plan-time code in a runtime that should not pay the planner
    package's import for a chunk boundary.
    """
    from sglang.srt.planner.cost_model import cumulative_boundaries

    return cumulative_boundaries(total_frames, weights)


def _overlap_per_seam(chain: Chain, requested: int) -> int:
    # A chunk boundary only costs something when a stage reads across frames.
    # RIFE is the only such stage in this chain.
    return requested if StageKind.RIFE in chain.kinds else 0


def capacity_weighted_plan(
    *,
    chain: Chain,
    rates: RateTable,
    cards: Sequence[CardAvailability],
    total_frames: int,
    reservation: ReservationInputs = ReservationInputs(),
    overlap_frames_per_seam: int = 1,
) -> ShardPlan:
    """Contiguous chunks sized by measured per-card chain throughput.

    Each card's weight is its chain frame rate, ``1000 / sum(ms per source
    frame over the stages placed on it)``, normalised over the offered cards.
    A card twice as fast gets twice the timeline, so all cards finish at
    roughly the same wall-clock instant and the makespan collapses to the
    weighted mean instead of being set by the slowest card.
    """
    offered, costs = _prepare(chain, rates, cards, total_frames, reservation)
    per_seam = _overlap_per_seam(chain, overlap_frames_per_seam)

    weights = [costs[c.card].frames_per_second for c in offered]
    boundaries = _weighted_boundaries(weights, total_frames)

    chunks: list[tuple[str, int, int]] = []
    start = 0
    for availability, stop in zip(offered, boundaries):
        if stop > start:
            chunks.append((availability.card, start, stop))
        start = stop

    dropped = [c.card for c in offered if c.card not in {ch[0] for ch in chunks}]
    if dropped:
        logger.info(
            "cards %s received no frames: %d frames do not divide across %d cards "
            "at these rates",
            dropped,
            total_frames,
            len(offered),
        )

    overlaps = _seam_overlaps(chunks, total_frames, per_seam)
    assignments = tuple(
        ShardAssignment(
            card=card, start=lo, stop=hi, lead_overlap=lead, tail_overlap=tail
        )
        for (card, lo, hi), (lead, tail) in zip(chunks, overlaps)
    )
    plan = ShardPlan(
        strategy=PlanStrategy.CAPACITY_WEIGHTED,
        chain=chain,
        total_frames=total_frames,
        assignments=assignments,
        rate_scales={c.card: c.rate_scale for c in offered},
        rationale=(
            "contiguous chunks weighted by measured chain rate; contiguous "
            "because RIFE reads adjacent pairs and an interleave would split "
            "every pair across cards"
        ),
    )
    return replace(
        plan, predicted_makespan_ms=predict_makespan(plan, rates).makespan_ms
    )


def vsgan_style_modulo_plan(
    *,
    chain: Chain,
    rates: RateTable,
    cards: Sequence[CardAvailability],
    total_frames: int,
    reservation: ReservationInputs = ReservationInputs(),
) -> ShardPlan:
    """Baseline: the prior-art static equal-share modulo-N round robin.

    This is ``SelectEvery(cycle=N, offsets=i)`` per device, reproduced with
    the same cost model as the proposal so the comparison is arithmetic. Two
    properties are priced honestly rather than assumed away:

    *   equal shares regardless of card speed, so the makespan is set by the
        slowest card;
    *   with RIFE in the chain, every owned frame's successor lives on another
        card, so the whole pre-RIFE prefix runs a second time per owned frame.
    """
    offered, _costs = _prepare(chain, rates, cards, total_frames, reservation)
    cycle = len(offered)
    has_rife = StageKind.RIFE in chain.kinds

    assignments: list[ShardAssignment] = []
    for offset, availability in enumerate(offered):
        owned = len(range(offset, total_frames, cycle))
        if owned == 0:
            continue
        assignments.append(
            ShardAssignment(
                card=availability.card,
                start=offset,
                stop=total_frames,
                stride=cycle,
                tail_overlap=owned if has_rife else 0,
            )
        )

    plan = ShardPlan(
        strategy=PlanStrategy.VSGAN_MODULO,
        chain=chain,
        total_frames=total_frames,
        assignments=tuple(assignments),
        rate_scales={c.card: c.rate_scale for c in offered},
        rationale=(
            f"static equal-share modulo-{cycle} interleave, the VSGAN "
            "mechanism; kept as a measurement baseline only"
        ),
    )
    return replace(
        plan, predicted_makespan_ms=predict_makespan(plan, rates).makespan_ms
    )


def static_single_card_plan(
    *,
    chain: Chain,
    rates: RateTable,
    cards: Sequence[CardAvailability],
    total_frames: int,
    card: str | None = None,
    reservation: ReservationInputs = ReservationInputs(),
) -> ShardPlan:
    """Baseline: the status quo, the whole timeline on one card.

    Without ``card`` the fastest offered card is chosen, which is the most
    favourable reading of the status quo and therefore the honest one to
    measure against.
    """
    offered, costs = _prepare(chain, rates, cards, total_frames, reservation)
    if card is None:
        chosen = min(offered, key=lambda c: costs[c.card].full_ms).card
    else:
        if card not in {c.card for c in offered}:
            raise ShardPlanError(
                f"card {card!r} was requested but not offered; offered: "
                f"{[c.card for c in offered]}"
            )
        chosen = card

    plan = ShardPlan(
        strategy=PlanStrategy.STATIC_SINGLE_CARD,
        chain=chain,
        total_frames=total_frames,
        assignments=(ShardAssignment(card=chosen, start=0, stop=total_frames),),
        rate_scales={c.card: c.rate_scale for c in offered},
        rationale="whole timeline on one card; no seams, so no overlap cost",
    )
    return replace(
        plan, predicted_makespan_ms=predict_makespan(plan, rates).makespan_ms
    )


def compare_plans(
    *,
    chain: Chain,
    rates: RateTable,
    cards: Sequence[CardAvailability],
    total_frames: int,
    reservation: ReservationInputs = ReservationInputs(),
) -> dict[PlanStrategy, MakespanPrediction]:
    """All three plan shapes, one cost model, for the before/after report."""
    plans = (
        capacity_weighted_plan(
            chain=chain,
            rates=rates,
            cards=cards,
            total_frames=total_frames,
            reservation=reservation,
        ),
        vsgan_style_modulo_plan(
            chain=chain,
            rates=rates,
            cards=cards,
            total_frames=total_frames,
            reservation=reservation,
        ),
        static_single_card_plan(
            chain=chain,
            rates=rates,
            cards=cards,
            total_frames=total_frames,
            reservation=reservation,
        ),
    )
    return {plan.strategy: predict_makespan(plan, rates) for plan in plans}
