# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The RIFE version ladder and its auto-selection policy (#460).

``rife.py`` knows which RIFE versions exist and which architectures are
vendored. It does not know which of them is *better*, how fast any of them is
on a given card, or whether its checkpoint is on disk -- and those three
questions together are what decides which version a chain should run.

The user directive this module implements (2026-08-03) is: make the version
selectable, and by default pick automatically the highest-quality variant that
can actually be computed inside the frame budget. Explicitly *not* "pick the
newest": the ticket-V measurements show 4.26 costing 45-123 % more than 4.6 at
every measured point for a quality gain nobody on this rig has graded, so a
ladder that always climbed to the top would be a regression dressed as an
upgrade.

Three rules hold the ladder honest.

**A variant nobody timed is never auto-picked.** Every frontier cell is a
:class:`~sglang.srt.planner.cost_model.Rate` carrying ``measured`` /
``estimate`` / ``absent``. Auto-selection considers ``measured`` cells only;
an absent cell puts the variant on the report as *measure me first* rather
than into the plan. An explicit pin is the one way to run an unmeasured
variant, and the selection says so in its reason.

**The quality order is an ASSUMPTION and is labelled as one.**
:data:`DEFAULT_QUALITY_RANK` encodes the user's stated belief -- the 4.1x and
lite families sit above 4.6, and the newest heavy variant is not required to
win -- and nothing in this tree has graded RIFE output quality. Until a
quality gate exists (a PSNR/SSIM-against-ground-truth harness on held-out
frame triples, which is a GPU ticket, not a desk one), the order is a
configurable input, not a measurement. :attr:`RifeVariant.quality_basis`
carries that word to every report that prints a rank.

**A variant whose weights are neither on disk nor pinned is rejected at
registry-construction time.** Not at selection time and not at warmup: a
ladder that can offer an entry it cannot fetch reproducibly is a ladder with a
rung missing, and finding that out when the job starts is too late.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from sglang.srt.planner.cost_model import Provenance, Rate
from sglang.srt.video_enhance.frame_math import Resolution
from sglang.srt.video_enhance.rife import (
    KNOWN_WEIGHT_SHA256,
    SUPPORTED_VERSIONS,
    VALID_SCALES,
    require_known,
    require_valid_scale,
    weight_filename,
    weights_are_cached,
)

__all__ = [
    "DEFAULT_QUALITY_RANK",
    "LadderError",
    "LadderReport",
    "LadderRow",
    "RifeFrontier",
    "RifeLadder",
    "RifeVariant",
    "Selection",
    "VramClass",
    "WeightState",
    "default_ladder",
    "seeded_frontier",
]


class LadderError(ValueError):
    """The ladder cannot be built, or cannot answer the question asked."""


# --------------------------------------------------------------------------
# Quality order -- an assumption, and it says so
# --------------------------------------------------------------------------

#: Lower is better. Read this as "the order the ladder climbs", not as a
#: measurement: no quality gate exists in this tree, so every rank here is the
#: user's stated preference (2026-08-03, translated: the 4.1x and lite
#: variants are believed to look better than 4.6, and the newest heavy variant
#: is not required to win) turned into a total order,
#: with two mechanical tie-breaks inside it -- newer beats older within a
#: family, and a full variant beats its own ``.lite`` sibling because the lite
#: IFNet is the same graph with roughly half the channel width.
#:
#: What is *not* asserted: that 4.26 actually looks better than 4.18, or that
#: 4.15.lite actually looks better than 4.6. Both are plausible and neither is
#: graded. Override with :meth:`RifeLadder.with_quality_ranks` -- the whole
#: point of making it data is that a future quality gate replaces it without
#: touching the selection code.
DEFAULT_QUALITY_RANK: Mapping[str, int] = {
    "4.26": 0,
    "4.18": 1,
    "4.17": 2,
    "4.15": 3,
    "4.17.lite": 4,
    "4.16.lite": 5,
    "4.15.lite": 6,
    "4.6": 7,
}

#: The one sentence every report that prints a rank must carry with it.
QUALITY_BASIS_ASSUMPTION = (
    "ASSUMPTION: no quality gate has graded RIFE output in this tree. The rank "
    "is the user's stated ordering (2026-08-03), not a measurement; a "
    "PSNR/SSIM-against-ground-truth harness would replace it"
)


class VramClass(str, enum.Enum):
    """Coarse per-frame-pair device-memory class of a variant.

    Coarse on purpose. The exact figure is resolution-dependent and lives in
    :attr:`RifeFrontier.vram_peak_mib`, which is measured or absent like every
    other cell; this class is the shape of the answer -- does the variant carry
    an encode head, and how wide is it -- and is derivable from the vendored
    architecture without a measurement.
    """

    #: No encode head, four pyramid levels (4.6 only).
    HEADLESS = "headless"
    #: Encode head at half channel width, four pyramid levels (the lite family).
    LITE = "lite"
    #: Encode head at full width, four pyramid levels (4.15/4.17/4.18).
    STANDARD = "standard"
    #: Encode head plus a fifth pyramid level (4.26). Measured at +66 % device
    #: bytes against 4.6 at 4K on the 5090 (ticket V, P4).
    DEEP = "deep"


class WeightState(str, enum.Enum):
    """Whether this rung's checkpoint can be obtained, and how."""

    #: The file is in the cache directory and its sidecar validates.
    PRESENT = "present"
    #: Not on disk, but ``rife.KNOWN_WEIGHT_SHA256`` pins it, so a fetch is
    #: reproducible.
    PINNED = "pinned"
    #: Neither on disk nor pinned. The registry refuses to hold such an entry.
    UNAVAILABLE = "unavailable"


# --------------------------------------------------------------------------
# Frontier
# --------------------------------------------------------------------------


def _frontier_key(version: str, card: str, resolution, scale: float):
    return (version, card, str(resolution), float(scale))


@dataclass(frozen=True)
class RifeFrontier:
    """Measured RIFE cost, keyed by ``(version, card, resolution, scale)``.

    Separate from :class:`~sglang.srt.planner.cost_model.StageRateTable` for
    one reason: that table's key is ``(stage, card, resolution)``, and RIFE's
    cost depends on two more axes the chain actually varies -- the version and
    the flow ``scale``. Folding them into the resolution string would make the
    key unparsable; adding them to the shared table would change a key every
    other stage uses. So the ladder carries its own table and reuses ``Rate``
    for the cell, which is what keeps the provenance discipline identical.

    ``ms`` is milliseconds per frame *pair* at multiplier 2, the unit the
    ticket-V A/B was taken in.
    """

    cells: Mapping[tuple[str, str, str, float], Rate] = field(default_factory=dict)
    #: Peak device bytes per frame pair, same key. Same provenance rules.
    vram_peak_mib: Mapping[tuple[str, str, str, float], Rate] = field(
        default_factory=dict
    )
    source: str = ""

    def rate(self, version: str, card: str, resolution, scale: float) -> Rate:
        key = _frontier_key(version, card, resolution, scale)
        cell = self.cells.get(key)
        if cell is not None:
            return cell
        known = sorted(
            f"{r} s{s}" for v, c, r, s in self.cells if v == version and c == card
        )
        return Rate.absent(
            f"no measured RIFE {version} frontier on card {card!r} at "
            f"{resolution} scale {scale}; measured points for that pair: "
            f"{', '.join(known) if known else 'none'}",
            unit="ms",
            label=f"rife{version}@{card}",
        )

    def vram(self, version: str, card: str, resolution, scale: float) -> Rate:
        key = _frontier_key(version, card, resolution, scale)
        cell = self.vram_peak_mib.get(key)
        if cell is not None:
            return cell
        return Rate.absent(
            f"no measured peak device bytes for RIFE {version} on {card!r} at "
            f"{resolution} scale {scale}",
            unit="MiB",
            label=f"rife{version}@{card}",
        )

    def versions(self) -> tuple[str, ...]:
        return tuple(sorted({v for v, _c, _r, _s in self.cells}))

    def points(self, version: str, card: str) -> tuple[tuple[str, float], ...]:
        return tuple(
            sorted((r, s) for v, c, r, s in self.cells if v == version and c == card)
        )

    def with_cells(
        self,
        rows: Iterable[tuple[str, str, Resolution, float, Rate]],
        *,
        source: str | None = None,
    ) -> "RifeFrontier":
        cells = dict(self.cells)
        for version, card, resolution, scale, rate in rows:
            require_known(version)
            require_valid_scale(scale)
            cells[_frontier_key(version, card, resolution, scale)] = rate
        return replace(
            self,
            cells=cells,
            source=self.source if source is None else source,
        )


# --------------------------------------------------------------------------
# The seed: ticket V, 2026-08-03
# --------------------------------------------------------------------------

#: Card keys used by the seeded frontier. They are the same free-form strings
#: the rate table and the probe reports use on this rig; the ladder never
#: resolves a device from them (device identity is the IdentityMap's job) and
#: only ever compares them for equality.
CARD_5090 = "5090"
CARD_3080 = "3080"

_TICKET_V = (
    "ticket V, 2026-08-03, /spinning/gpu-battery-results/2026-08-03_ticketV/"
    "RESULTS.md §3 (torch eager fp16, ms per frame pair; the 4.26 column is the "
    "encode-cache *amortised* figure, which is what a sequential stream pays)"
)

_R1080 = Resolution(1920, 1080)
_R2160 = Resolution(3840, 2160)

#: ``(version, card, resolution, scale) -> ms per frame pair``. Every number
#: here was measured in the ticket-V window against a same-boot A-vs-A floor
#: (0.682 % on the 5090, 0.108 % on the 3080). Nothing else is seeded: the
#: other six rungs of the ladder are ABSENT until TICKET_460 runs them.
_SEED_MS: Mapping[tuple[str, str, Resolution, float], float] = {
    ("4.6", CARD_5090, _R1080, 1.0): 5.648,
    ("4.6", CARD_5090, _R1080, 0.5): 5.057,
    ("4.6", CARD_5090, _R2160, 1.0): 20.539,
    ("4.6", CARD_5090, _R2160, 0.5): 11.359,
    ("4.26", CARD_5090, _R1080, 1.0): 8.185,
    ("4.26", CARD_5090, _R1080, 0.5): 7.421,
    ("4.26", CARD_5090, _R2160, 1.0): 32.974,
    ("4.26", CARD_5090, _R2160, 0.5): 25.373,
    ("4.6", CARD_3080, _R1080, 1.0): 16.088,
    ("4.6", CARD_3080, _R1080, 0.5): 9.142,
    ("4.6", CARD_3080, _R2160, 1.0): 63.108,
    ("4.6", CARD_3080, _R2160, 0.5): 31.999,
    ("4.26", CARD_3080, _R1080, 1.0): 24.252,
    ("4.26", CARD_3080, _R1080, 0.5): 19.145,
    ("4.26", CARD_3080, _R2160, 1.0): 91.630,
    ("4.26", CARD_3080, _R2160, 0.5): 67.520,
}

#: Peak per-frame-pair device MiB, ticket V P4. Only the two 4K scale-1.0
#: points were taken.
_SEED_VRAM_MIB: Mapping[tuple[str, str, Resolution, float], float] = {
    ("4.6", CARD_5090, _R2160, 1.0): 4742.0,
    ("4.26", CARD_5090, _R2160, 1.0): 7855.0,
}


def seeded_frontier() -> RifeFrontier:
    """The ticket-V frontier, and nothing invented around it."""
    cells = {
        _frontier_key(v, c, r, s): Rate.measured(ms, _TICKET_V, unit="ms")
        for (v, c, r, s), ms in _SEED_MS.items()
    }
    vram = {
        _frontier_key(v, c, r, s): Rate.measured(mib, _TICKET_V, unit="MiB")
        for (v, c, r, s), mib in _SEED_VRAM_MIB.items()
    }
    return RifeFrontier(cells=cells, vram_peak_mib=vram, source=_TICKET_V)


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------

_VRAM_CLASS: Mapping[str, VramClass] = {
    "4.6": VramClass.HEADLESS,
    "4.15": VramClass.STANDARD,
    "4.17": VramClass.STANDARD,
    "4.18": VramClass.STANDARD,
    "4.15.lite": VramClass.LITE,
    "4.16.lite": VramClass.LITE,
    "4.17.lite": VramClass.LITE,
    "4.26": VramClass.DEEP,
}


@dataclass(frozen=True)
class RifeVariant:
    """One rung: a version, where it sits, and whether it can be run."""

    version: str
    quality_rank: int
    vram_class: VramClass
    weight_state: WeightState
    weight_path: Path | None
    weight_sha256: str | None
    quality_basis: str = QUALITY_BASIS_ASSUMPTION

    @property
    def runnable(self) -> bool:
        """Architecture vendored *and* weights obtainable."""
        return (
            self.version in SUPPORTED_VERSIONS
            and self.weight_state is not WeightState.UNAVAILABLE
        )

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "quality_rank": self.quality_rank,
            "quality_basis": self.quality_basis,
            "vram_class": self.vram_class.value,
            "weight_state": self.weight_state.value,
            "weight_path": None if self.weight_path is None else str(self.weight_path),
            "weight_sha256": self.weight_sha256,
            "runnable": self.runnable,
        }


def _weight_state(
    version: str, weight_dir: Path | None
) -> tuple[WeightState, Path | None, str | None]:
    pin = KNOWN_WEIGHT_SHA256.get(version)
    path = None if weight_dir is None else weight_dir / weight_filename(version)
    if weight_dir is not None and weights_are_cached(version, weight_dir):
        return WeightState.PRESENT, path, pin
    if pin is not None:
        return WeightState.PINNED, path, pin
    return WeightState.UNAVAILABLE, path, None


# --------------------------------------------------------------------------
# Selection report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LadderRow:
    """One variant scored against one budget, with the verdict spelled out."""

    variant: RifeVariant
    ms: float | None
    provenance: Provenance
    source: str
    fits: bool
    verdict: str

    def as_dict(self) -> dict:
        return {
            **self.variant.as_dict(),
            "ms_per_pair": None if self.ms is None else round(self.ms, 3),
            "provenance": self.provenance.value,
            "source": self.source,
            "fits": self.fits,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class Selection:
    """What the ladder chose, and every rung it looked at."""

    version: str | None
    ms: float | None
    provenance: Provenance
    reason: str
    pinned: bool
    rows: tuple[LadderRow, ...]
    #: Rungs whose frontier cell at this point is absent. Named so the GPU
    #: ticket has its work list straight out of the policy report.
    measure_first: tuple[str, ...] = ()

    @property
    def chosen(self) -> bool:
        return self.version is not None

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "ms_per_pair": None if self.ms is None else round(self.ms, 3),
            "provenance": self.provenance.value,
            "pinned": self.pinned,
            "reason": self.reason,
            "measure_first": list(self.measure_first),
            "quality_basis": QUALITY_BASIS_ASSUMPTION,
            "ladder": [row.as_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class LadderReport:
    """The ladder itself, for the ``/capabilities``-style surfaces."""

    variants: tuple[RifeVariant, ...]
    frontier_source: str
    quality_basis: str = QUALITY_BASIS_ASSUMPTION

    def as_dict(self) -> dict:
        return {
            "quality_basis": self.quality_basis,
            "frontier_source": self.frontier_source,
            "variants": [v.as_dict() for v in self.variants],
        }


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def _default_weight_dir() -> Path | None:
    from sglang.srt.video_enhance.rife import default_weight_dir

    directory = default_weight_dir()
    return directory if directory.is_dir() else None


@dataclass(frozen=True)
class RifeLadder:
    """Every runnable RIFE variant, ordered by assumed quality, plus a frontier."""

    variants: tuple[RifeVariant, ...]
    frontier: RifeFrontier

    def __post_init__(self) -> None:
        if not self.variants:
            raise LadderError(
                "the RIFE ladder is empty: no vendored version has weights that "
                "are either present on disk or pinned in "
                "rife.KNOWN_WEIGHT_SHA256"
            )
        seen: set[str] = set()
        for variant in self.variants:
            if variant.version in seen:
                raise LadderError(f"duplicate ladder entry for {variant.version!r}")
            seen.add(variant.version)
            if variant.weight_state is WeightState.UNAVAILABLE:
                raise LadderError(
                    f"ladder entry {variant.version!r} has neither a checkpoint on "
                    "disk nor a sha256 in rife.KNOWN_WEIGHT_SHA256, so it cannot be "
                    "obtained reproducibly. Fetch it with "
                    "scripts/video_enhance/fetch_rife_weights.py --record-new-pin "
                    "and add the printed pin, or drop the entry"
                )
            if variant.version not in SUPPORTED_VERSIONS:
                raise LadderError(
                    f"ladder entry {variant.version!r} has no vendored IFNet; "
                    "see video_enhance/_vendor/rife/README.md"
                )

    def version(self, version: str) -> RifeVariant:
        for variant in self.variants:
            if variant.version == version:
                return variant
        raise LadderError(
            f"{version!r} is not on the ladder. Rungs: "
            f"{', '.join(v.version for v in self.variants)}"
        )

    def ordered(self) -> tuple[RifeVariant, ...]:
        """Best assumed quality first, then version string for determinism."""
        return tuple(sorted(self.variants, key=lambda v: (v.quality_rank, v.version)))

    def report(self) -> LadderReport:
        return LadderReport(
            variants=self.ordered(), frontier_source=self.frontier.source
        )

    def with_quality_ranks(self, ranks: Mapping[str, int]) -> "RifeLadder":
        """Re-rank the ladder, e.g. from a quality gate that graded output."""
        updated = tuple(
            replace(v, quality_rank=int(ranks.get(v.version, v.quality_rank)))
            for v in self.variants
        )
        return replace(self, variants=updated)

    def with_frontier(self, frontier: RifeFrontier) -> "RifeLadder":
        return replace(self, frontier=frontier)

    # -- the policy -------------------------------------------------------

    def worst_case_rate(
        self, version: str, cards: Sequence[str], resolution: Resolution, scale: float
    ) -> Rate:
        """This variant's cost on the *slowest* of ``cards``, or an absence.

        Regime A runs the same chain on every card, so a version that is too
        slow on the weakest card drags the aggregate down even though it fits
        comfortably on the strongest. The admissible cost of a variant across a
        set of cards is therefore the worst of them, and a variant that is
        unmeasured on any one of them is unmeasured for the set -- the absence
        names which card is missing, so the gap is actionable rather than
        merely fatal.
        """
        if not cards:
            raise LadderError("no cards were offered to the ladder")
        worst: Rate | None = None
        for card in cards:
            rate = self.frontier.rate(version, card, resolution, scale)
            if rate.is_absent:
                return rate
            if worst is None or float(rate.value) > float(worst.value):
                worst = rate
        assert worst is not None
        if len(cards) == 1:
            return worst
        return Rate(
            worst.value,
            worst.provenance,
            f"worst of {len(cards)} cards ({', '.join(cards)}): {worst.source}",
            unit=worst.unit,
            label=worst.label,
        )

    def select(
        self,
        *,
        card: str | Sequence[str],
        resolution: Resolution,
        scale: float,
        budget_ms: float | None,
        pin: str | None = None,
        allowed: Sequence[str] | None = None,
    ) -> Selection:
        """Highest-ranked variant whose MEASURED cost fits ``budget_ms``.

        ``budget_ms`` is milliseconds per frame pair -- the same unit the
        frontier is in -- and is the slice of the frame budget the caller has
        left for interpolation after every other stage is paid for. ``None``
        means "no budget constraint", which is the honest way to ask "what is
        the best variant you can price at all" and is *not* the same as a very
        large budget: with no measurement there is still nothing to pick.

        ``card`` may be one card or several. With several, a variant is judged
        on the slowest of them (see :meth:`worst_case_rate`) and is unmeasured
        for the set if it is unmeasured on any one of them.

        ``pin`` overrides everything, including an absent frontier. That is
        deliberate: pinning is how a GPU window runs the variant it is about to
        measure. The selection reports ``provenance=absent`` in that case, so a
        caller cannot mistake a pinned unmeasured run for a priced one.
        """
        require_valid_scale(scale)
        cards = (card,) if isinstance(card, str) else tuple(card)
        card_label = cards[0] if len(cards) == 1 else "/".join(cards)
        candidates = self.ordered()
        if allowed is not None:
            allow = set(allowed)
            candidates = tuple(v for v in candidates if v.version in allow)
            if not candidates:
                raise LadderError(
                    f"none of the allowed versions {sorted(allow)} is on the "
                    f"ladder ({', '.join(v.version for v in self.ordered())})"
                )

        rows: list[LadderRow] = []
        measure_first: list[str] = []
        for variant in candidates:
            rate = self.worst_case_rate(variant.version, cards, resolution, scale)
            ms = rate.or_none()
            if rate.provenance is Provenance.ABSENT:
                measure_first.append(variant.version)
                verdict = "no measured frontier at this point; never auto-picked"
                fits = False
            elif rate.provenance is not Provenance.MEASURED:
                verdict = (
                    f"{rate.provenance.value} rather than measured; auto-selection "
                    "takes measured cells only"
                )
                fits = False
            elif budget_ms is None:
                verdict = f"measured at {ms:.3f} ms/pair; no budget was given"
                fits = True
            elif ms <= budget_ms:
                verdict = f"measured {ms:.3f} ms/pair fits the {budget_ms:.3f} ms budget"
                fits = True
            else:
                verdict = (
                    f"measured {ms:.3f} ms/pair exceeds the {budget_ms:.3f} ms budget"
                )
                fits = False
            rows.append(
                LadderRow(
                    variant=variant,
                    ms=ms,
                    provenance=rate.provenance,
                    source=rate.source,
                    fits=fits,
                    verdict=verdict,
                )
            )

        if pin is not None:
            require_known(pin)
            variant = self.version(pin)
            row = next((r for r in rows if r.variant.version == pin), None)
            if row is None:
                raise LadderError(
                    f"pinned version {pin!r} is excluded by allowed={list(allowed or ())}"
                )
            note = ""
            if row.provenance is Provenance.ABSENT:
                note = (
                    ". Its cost at this point is unmeasured, so this plan carries "
                    "no throughput guarantee"
                )
            elif budget_ms is not None and not row.fits:
                note = (
                    f". It does NOT fit the {budget_ms:.3f} ms budget "
                    f"({row.ms:.3f} ms measured); the pin overrides the budget gate"
                )
            return Selection(
                version=pin,
                ms=row.ms,
                provenance=row.provenance,
                reason=(
                    f"RIFE {pin} was pinned explicitly, so the ladder did not "
                    f"choose (rank {variant.quality_rank}, {variant.vram_class.value} "
                    f"VRAM class)" + note
                ),
                pinned=True,
                rows=tuple(rows),
                measure_first=tuple(measure_first),
            )

        winners = [r for r in rows if r.fits]
        if not winners:
            priced = [r for r in rows if r.provenance is Provenance.MEASURED]
            if not priced:
                reason = (
                    f"no RIFE variant has a measured frontier on card {card_label!r} at "
                    f"{resolution} scale {scale}. Unmeasured rungs: "
                    f"{', '.join(measure_first)}. Run TICKET_460's frontier sweep, "
                    "or pin a version explicitly to run it unpriced"
                )
            else:
                cheapest = min(priced, key=lambda r: r.ms)
                reason = (
                    f"no RIFE variant fits a {budget_ms:.3f} ms/pair budget on card "
                    f"{card_label!r} at {resolution} scale {scale}; the cheapest measured "
                    f"rung is {cheapest.variant.version} at {cheapest.ms:.3f} ms. "
                    f"Unmeasured rungs that were not considered: "
                    f"{', '.join(measure_first) or 'none'}"
                )
            return Selection(
                version=None,
                ms=None,
                provenance=Provenance.ABSENT,
                reason=reason,
                pinned=False,
                rows=tuple(rows),
                measure_first=tuple(measure_first),
            )

        # ``rows`` is already in quality order, so the first fitting row is the
        # highest-ranked one that fits. Cost never outranks quality here -- it
        # only ever decides whether a rung is admissible at all.
        best = winners[0]
        skipped = [r.variant.version for r in rows if r is not best and not r.fits]
        reason = (
            f"RIFE {best.variant.version} is the highest-ranked variant whose "
            f"measured cost fits: {best.verdict} on card {card_label!r} at {resolution} "
            f"scale {scale} (rank {best.variant.quality_rank}, "
            f"{best.variant.vram_class.value} VRAM class). "
            + (
                f"Passed over: {', '.join(skipped)}. "
                if skipped
                else ""
            )
            + QUALITY_BASIS_ASSUMPTION
        )
        return Selection(
            version=best.variant.version,
            ms=best.ms,
            provenance=best.provenance,
            reason=reason,
            pinned=False,
            rows=tuple(rows),
            measure_first=tuple(measure_first),
        )


def default_ladder(
    *,
    weight_dir: str | os.PathLike[str] | None = None,
    frontier: RifeFrontier | None = None,
    quality_ranks: Mapping[str, int] | None = None,
    versions: Sequence[str] | None = None,
) -> RifeLadder:
    """Build the ladder from what is vendored, pinned and on disk.

    ``weight_dir`` defaults to ``rife.default_weight_dir()`` when that
    directory exists and to "nothing on disk" when it does not; in the latter
    case every rung is :attr:`WeightState.PINNED`, which is still runnable
    because the pin makes the fetch reproducible.
    """
    directory = (
        Path(weight_dir)
        if weight_dir is not None
        else _default_weight_dir()
    )
    ranks = dict(DEFAULT_QUALITY_RANK)
    if quality_ranks:
        ranks.update(quality_ranks)
    wanted = tuple(versions) if versions is not None else tuple(sorted(SUPPORTED_VERSIONS))

    entries: list[RifeVariant] = []
    for version in wanted:
        require_known(version)
        state, path, pin = _weight_state(version, directory)
        if state is WeightState.UNAVAILABLE:
            # Not an error here -- the ladder simply does not offer a rung it
            # cannot obtain. ``RifeLadder.__post_init__`` is what refuses an
            # entry somebody constructed by hand.
            continue
        if version not in ranks:
            raise LadderError(
                f"no quality rank for {version!r}; add it to DEFAULT_QUALITY_RANK "
                "or pass quality_ranks. A rung with no rank cannot be ordered, and "
                "guessing its place is exactly what this ladder must not do"
            )
        entries.append(
            RifeVariant(
                version=version,
                quality_rank=ranks[version],
                vram_class=_VRAM_CLASS.get(version, VramClass.STANDARD),
                weight_state=state,
                weight_path=path,
                weight_sha256=pin,
            )
        )
    return RifeLadder(
        variants=tuple(entries),
        frontier=frontier if frontier is not None else seeded_frontier(),
    )


def valid_scales() -> tuple[float, ...]:
    """Re-exported so a caller building a sweep does not import ``rife`` too."""
    return VALID_SCALES
