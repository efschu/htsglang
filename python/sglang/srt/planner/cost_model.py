# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""One cost library, many placement planners (#348b).

Every placement decision in the fork prices the same two physical facts:

  1. how fast is this card at this kind of work   (per-card compute rate)
  2. what does moving a byte between two cards cost (pair-matrix hop cost)

Before this module those two facts were re-derived independently by three
planners -- the K1 key solver (LLM tensor/DCP splits), the K2/K3 video
``shard_plan``, and the #333-M3 diffusion sequence-parallel split -- each with
its own reader, its own fallback, and its own idea of what an absent
measurement means. A measurement improvement therefore reached exactly one
consumer, and two planners could price the same card pair differently without
anything noticing. This module is the single source; the planners are
consumers.

Three rules the library enforces, all of them load-bearing:

**Measured or named.** Every number is a :class:`Rate` carrying its
provenance. ``MEASURED`` came off a probe. ``ESTIMATE`` came out of a formula
over measured inputs. ``ABSENT`` has no value at all and says why. There is no
fourth case: a rate that could not be measured is never silently filled with a
roofline, a 1.0, or a 0.0. :meth:`Rate.require` raises :class:`AbsentRate`
with the absence's own text rather than handing a caller a plausible float.

**The roofline never ranks a split** (the #216/#264 guard). An analytic peak
may describe a card; it may not order two candidate placements against each
other. Where an absent link rate would otherwise stop a ranking dead, the
library offers exactly one documented, split-invariant placeholder
(:data:`ABSENT_LINK_RANKING_PLACEHOLDER_GBS`) whose contract is that it adds
the same constant to every candidate -- and it is a placeholder for the
*ranking* only; the absolute prediction stays absent.

**A hop is between two different cards.** Same-card ("loopback") entries are
rejected at parse time, in both on-disk pair-matrix shapes, and counted in
:attr:`PairMatrix.rejected` so the rejection is visible rather than a silent
row-drop -- the intra-rig form of the #214/#271 rule that a loopback number
must never wear a wire's label.

Public surface
--------------

Provenance:      :class:`Provenance`, :class:`Rate`, :class:`AbsentRate`
Per-card rate:   :class:`ComputeRates`, :func:`compute_rates_from_entries`,
                 :func:`compute_rates_for_cards`, :func:`memory_rates_from_entries`
Hop cost:        :class:`Hop`, :class:`PairMatrix`,
                 :func:`pair_matrix_from_card_probe`,
                 :func:`pair_matrix_from_hardware_profile`,
                 :func:`reconcile_pair_matrices`
Composition:     :func:`ring_factor`, :func:`allreduce_seconds`,
                 :func:`apportion_largest_remainder`,
                 :func:`cumulative_boundaries`, :func:`apportion_cumulative`
Bundle:          :class:`CostSources`, :func:`load_cost_sources`

Import weight: this module is stdlib-only at module scope. ``uneven_perf`` --
which pulls torch -- is imported lazily inside the functions that actually
need a measured profile, so a consumer that only wants the composition
primitives (the diffusion SP split calls :func:`apportion_largest_remainder`
per request) pays nothing for them.

Entry point for #302 (expert placement)
---------------------------------------

Expert placement is a consumer of this library, not a re-implementation. It
needs both axes and nothing else:

* per-expert compute rate: :meth:`ComputeRates.for_family` with
  ``uneven_perf.GEMM_FAMILY_MOE`` -- the #324 per-family widening already
  resolves a MoE family onto its own lane per card, so a rig where one card
  runs the experts on native fp8 and another falls back to Marlin scores them
  apart without #302 deriving anything;
* the all-to-all hop: :meth:`PairMatrix.hop` for a directed expert->rank
  route, or :meth:`PairMatrix.narrowest_bandwidth_gbs` for the group bound,
  composed with :func:`allreduce_seconds` (the dispatch/combine pair of an
  MoE layer is two collectives of the token payload).

:func:`load_cost_sources` returns both, already resolved and provenance-
tagged, from whichever artifacts exist on disk. #302 should call it and then
add only its own objective -- expert-to-card assignment -- on top. If it finds
itself reading ``profile["gpus"]`` or ``probe["pairs"]`` directly, that is the
signal the library is missing a primitive and should grow one.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ABSENT_LINK_ASSUMED_GBS",
    "ABSENT_LINK_RANKING_PLACEHOLDER_GBS",
    "AbsentRate",
    "ComputeRates",
    "CostSources",
    "Hop",
    "PairMatrix",
    "Provenance",
    "Rate",
    "StageRateTable",
    "allreduce_seconds",
    "apportion_cumulative",
    "apportion_largest_remainder",
    "compute_rates_for_cards",
    "compute_rates_from_entries",
    "cumulative_boundaries",
    "load_cost_sources",
    "memory_rates_from_entries",
    "pair_matrix_from_card_probe",
    "pair_matrix_from_hardware_profile",
    "reconcile_pair_matrices",
    "ring_factor",
    "stage_rates_from_reports",
    "stage_rates_from_samples",
]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class Provenance(str, enum.Enum):
    """Where a number came from. There is no 'probably' tier on purpose."""

    #: Read off a probe measurement on this rig.
    MEASURED = "measured"
    #: Computed by a formula whose inputs are measured.
    ESTIMATE = "estimate"
    #: No value exists. The reason is carried in ``Rate.source``.
    ABSENT = "absent"


class AbsentRate(LookupError):
    """A caller asked for the value of a rate that was never measured.

    Carries the absence's own text so the message names what is missing and
    what would produce it, instead of surfacing as a ``None`` three frames
    later or, worse, as a plausible default nobody measured.
    """

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"{name}: {reason}")
        self.name = name
        self.reason = reason


@dataclasses.dataclass(frozen=True)
class Rate:
    """One number with its provenance, or a named absence.

    ``value`` is ``None`` exactly when ``provenance`` is ``ABSENT``; the
    constructor enforces it, so a truthiness test on ``value`` and a check of
    ``provenance`` can never disagree.
    """

    #: The number, in ``unit``. ``None`` iff absent.
    value: Optional[float]
    #: Measured / estimate / absent.
    provenance: Provenance
    #: Where it came from, or -- when absent -- why it is not there.
    source: str
    #: What the number counts, e.g. ``"TFLOP/s"`` or ``"GB/s"``.
    unit: str = ""
    #: Optional detail, e.g. the GEMM lane the score was taken on.
    label: str = ""

    def __post_init__(self) -> None:
        absent = self.provenance is Provenance.ABSENT
        if absent and self.value is not None:
            raise ValueError("an absent rate cannot carry a value")
        if not absent and self.value is None:
            raise ValueError(
                f"a {self.provenance.value} rate must carry a value; use "
                "Rate.absent() to record that it does not exist"
            )

    @classmethod
    def measured(cls, value: float, source: str, *, unit: str = "", label: str = ""):
        return cls(float(value), Provenance.MEASURED, source, unit, label)

    @classmethod
    def estimate(cls, value: float, source: str, *, unit: str = "", label: str = ""):
        return cls(float(value), Provenance.ESTIMATE, source, unit, label)

    @classmethod
    def absent(cls, reason: str, *, unit: str = "", label: str = ""):
        return cls(None, Provenance.ABSENT, reason, unit, label)

    @property
    def is_absent(self) -> bool:
        return self.provenance is Provenance.ABSENT

    def require(self, name: str = "rate") -> float:
        """The value, or :class:`AbsentRate` naming what is missing."""
        if self.value is None:
            raise AbsentRate(name, self.source)
        return float(self.value)

    def or_none(self) -> Optional[float]:
        """The value, or ``None`` -- for callers that report absence
        themselves rather than raising."""
        return None if self.value is None else float(self.value)


# ---------------------------------------------------------------------------
# Per-card compute rate
# ---------------------------------------------------------------------------

#: The checkpoint format key for an unquantized (dense bf16) model. It is a
#: real key of ``uneven_perf._FORMAT_LANES``, so it resolves to the dense GEMM
#: probe with a proper lane label and NO fallback warning -- which is exactly
#: right for the diffusion DiT and for any bf16 LLM.
FORMAT_DENSE_BF16 = "bf16"


@dataclasses.dataclass(frozen=True)
class ComputeRates:
    """Per-position compute rates in the checkpoint's own format.

    "Position" is whatever order the caller passed its cards in: rank order
    for the K1 solver, card order for a shard planner. The library does not
    reorder, because the device-order trap (torch ordinal vs NVML index vs
    ``--rank-gpu-id``) is the caller's to resolve and silently re-sorting here
    would hide it.

    ``families`` holds a vector only for a #324 GEMM family whose format
    diverges from the checkpoint-wide one, mirroring
    ``uneven_perf.GemmScores``: on a uniform checkpoint it is empty and every
    family lookup returns the scalar.
    """

    #: One rate per position, in the format the checkpoint declares.
    rates: Tuple[Rate, ...]
    #: The caller's key per position (uuid, card name, index -- its choice).
    keys: Tuple[str, ...]
    #: The checkpoint format these were resolved for.
    fmt: str = FORMAT_DENSE_BF16
    #: family -> one rate per position, only for families that diverge.
    families: Mapping[str, Tuple[Rate, ...]] = dataclasses.field(default_factory=dict)
    #: Loud fallbacks from the lane resolution (wrong-lane scoring warnings).
    warnings: Tuple[str, ...] = ()

    @property
    def mixed(self) -> bool:
        """Whether any family genuinely scores apart from the scalar."""
        return bool(self.families)

    def for_family(self, family: Optional[str]) -> Tuple[Rate, ...]:
        """This family's rates, or the scalar when the family has none.

        The #302 expert-placement entry point: pass
        ``uneven_perf.GEMM_FAMILY_MOE``.
        """
        if family and family in self.families:
            return tuple(self.families[family])
        return self.rates

    def absences(self) -> List[str]:
        """One line per position whose rate does not exist. Empty when every
        position is measured."""
        return [
            f"{key}: {rate.source}"
            for key, rate in zip(self.keys, self.rates)
            if rate.is_absent
        ]

    def values(self, family: Optional[str] = None) -> List[float]:
        """The rates as plain floats.

        Raises :class:`AbsentRate` naming the first position that has none --
        a planner that divides by a compute rate must not be handed a zero
        standing in for "not measured".
        """
        out: List[float] = []
        for key, rate in zip(self.keys, self.for_family(family)):
            out.append(rate.require(f"compute rate for {key}"))
        return out

    def weights(self, family: Optional[str] = None) -> List[float]:
        """The rates normalised to sum 1.0 -- the shape both shard planners
        want. Same absence contract as :meth:`values`."""
        vals = self.values(family)
        total = math.fsum(vals)
        if total <= 0:
            raise ValueError(
                "every measured compute rate is zero or negative; a split "
                "cannot be weighted by it"
            )
        return [v / total for v in vals]


def compute_rates_from_entries(
    entries: Sequence[Mapping],
    keys: Sequence[str],
    *,
    fmt: str = FORMAT_DENSE_BF16,
    family_formats: Optional[Mapping[str, str]] = None,
) -> ComputeRates:
    """Per-card compute rates from already-loaded profile entries.

    ``entries`` are the per-GPU dicts of the hardware profile
    (``profile["gpus"][uuid]``) or of the card probe (``probe["cards"][i]``);
    both carry ``gemm_tflops`` plus the optional ``gemm_lanes`` map. The lane
    resolution is NOT re-derived here -- it delegates to
    ``uneven_perf.rank_gemm_family_scores``, the #324 per-family widening, so
    a lane the probe learns tomorrow reaches every consumer at once.

    An entry with no ``gemm_tflops`` becomes a NAMED absence rather than a
    zero: the whole point of the wrapper.
    """
    if len(entries) != len(keys):
        raise ValueError(
            f"{len(entries)} profile entries against {len(keys)} keys; one "
            "entry per card is required"
        )

    scored: List[int] = []
    holes: Dict[int, str] = {}
    for i, entry in enumerate(entries):
        if entry is None or entry.get("gemm_tflops") is None:
            holes[i] = (
                f"no measured GEMM rate for {keys[i]} in the profile. Run the "
                "rig probe (or POST /api/card_probe) so this card has a score."
            )
        else:
            scored.append(i)

    if not scored:
        return ComputeRates(
            rates=tuple(
                Rate.absent(holes[i], unit="TFLOP/s") for i in range(len(keys))
            ),
            keys=tuple(keys),
            fmt=fmt,
        )

    from sglang.srt import uneven_perf  # noqa: PLC0415  (heavy: pulls torch)

    dense = [dict(entries[i]) for i in scored]
    gemm = uneven_perf.rank_gemm_family_scores(
        dense, fmt, dict(family_formats) if family_formats else None
    )

    def _spread(vals: Sequence[float], labels: Sequence[str]) -> Tuple[Rate, ...]:
        out: List[Rate] = []
        pos = 0
        for i in range(len(keys)):
            if i in holes:
                out.append(Rate.absent(holes[i], unit="TFLOP/s"))
            else:
                out.append(
                    Rate.measured(
                        vals[pos],
                        f"rig probe, {fmt} GEMM lane resolution",
                        unit="TFLOP/s",
                        label=labels[pos],
                    )
                )
                pos += 1
        return tuple(out)

    families = {
        family: _spread(vals, gemm.family_labels.get(family, [""] * len(vals)))
        for family, vals in gemm.families.items()
    }
    return ComputeRates(
        rates=_spread(gemm.scalar, gemm.scalar_labels),
        keys=tuple(keys),
        fmt=fmt,
        families=families,
        warnings=tuple(gemm.warnings),
    )


def compute_rates_for_cards(
    card_uuids: Sequence[str],
    *,
    fmt: str = FORMAT_DENSE_BF16,
    family_formats: Optional[Mapping[str, str]] = None,
    profile: Optional[Mapping] = None,
) -> ComputeRates:
    """Per-card compute rates straight from the cached hardware profile.

    ``card_uuids`` are NVML UUIDs, the fork's canonical card identity (#331).
    ``profile`` overrides the on-disk read, which is what tests use.

    The profile is read CACHE-ONLY: a placement question must never trigger a
    multi-second GPU probe as a side effect. When no profile exists at all,
    every card comes back as a named absence, so the caller can say what is
    missing and what would produce it.
    """
    if profile is None:
        from sglang.srt import uneven_perf  # noqa: PLC0415  (heavy: pulls torch)

        profile, _inventory = uneven_perf.get_cached_hardware_profile()
    gpus = (profile or {}).get("gpus") or {}
    entries = [gpus.get(str(u)) for u in card_uuids]
    if not gpus:
        reason = (
            "no cached hardware profile for this rig, so no card has a "
            "measured GEMM rate. Run the rig probe first."
        )
        return ComputeRates(
            rates=tuple(Rate.absent(reason, unit="TFLOP/s") for _ in card_uuids),
            keys=tuple(str(u) for u in card_uuids),
            fmt=fmt,
        )
    return compute_rates_from_entries(
        entries,
        [str(u) for u in card_uuids],
        fmt=fmt,
        family_formats=family_formats,
    )


#: Memory-side rates the profile and the card probe both carry, with the
#: field aliases each artifact uses. Order matters: the first field present
#: wins, matching the pre-existing readers.
_MEMORY_RATE_FIELDS: Dict[str, Tuple[Tuple[str, ...], str, str]] = {
    "membw": (("membw_read_gbs", "membw_gbs"), "GB/s", "streaming read rate"),
    "gemv": (("membw_gemv_gbs",), "GB/s", "decode-shaped GEMV rate"),
    "h2d": (("h2d_gbs",), "GB/s", "pinned host-to-device rate"),
    "d2h": (("d2h_gbs",), "GB/s", "pinned device-to-host rate"),
}


def memory_rates_from_entries(
    entries: Sequence[Mapping],
    keys: Sequence[str],
    kind: str,
) -> Tuple[Rate, ...]:
    """Per-card memory-side rate (``membw`` / ``gemv`` / ``h2d`` / ``d2h``).

    Same contract as the compute side: a card the probe did not measure gets
    a named absence, never a 0.0. This is the primitive that closes the gap
    where the K1 solver defaulted a missing streaming bandwidth to ``0.0``
    without recording it -- a zero that then survived as an
    almost-but-not-quite-valid divisor.
    """
    if kind not in _MEMORY_RATE_FIELDS:
        raise ValueError(
            f"unknown memory rate {kind!r}; known: {sorted(_MEMORY_RATE_FIELDS)}"
        )
    fields, unit, what = _MEMORY_RATE_FIELDS[kind]
    out: List[Rate] = []
    for entry, key in zip(entries, keys):
        value = None
        for field in fields:
            candidate = (entry or {}).get(field)
            if candidate:
                value = float(candidate)
                break
        if value is None:
            out.append(
                Rate.absent(
                    f"no measured {what} for {key} (profile predates it, or "
                    "the probe group did not run)",
                    unit=unit,
                )
            )
        else:
            out.append(Rate.measured(value, f"rig probe, {what}", unit=unit))
    return tuple(out)


# ---------------------------------------------------------------------------
# Pair-matrix hop cost
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Hop:
    """The measured cost of moving bytes from one card to another.

    Directed: ``src -> dst`` and ``dst -> src`` are two hops, because on a rig
    without peer access in one direction they genuinely differ (measured on
    the reference rig: 4.52 vs 6.88 GB/s to the same card over x4 against x8).
    A shape that only stores unordered pairs is widened on read, and the
    widening is recorded in :attr:`PairMatrix.notes` rather than passed off as
    two measurements.
    """

    src: str
    dst: str
    bandwidth_gbs: Rate
    latency_us: Rate
    transport: str = ""


@dataclasses.dataclass(frozen=True)
class PairMatrix:
    """Measured card-to-card costs over one set of participating cards.

    Measured-only by construction: there is no synthesised hop for a pair the
    probe did not reach, and no topology rule that invents one from a bus
    layout. An absent pair is absent, and :meth:`narrowest_bandwidth_gbs`
    returns an absent :class:`Rate` rather than a number.
    """

    #: (src, dst) -> hop, directed, same-card entries excluded.
    hops: Mapping[Tuple[str, str], Hop]
    #: The cards this matrix was built over, in the caller's order.
    keys: Tuple[str, ...]
    #: Where the rows came from (``"card_probe"`` / ``"hardware_profile"``).
    source: str = ""
    #: Rows dropped at parse time, with the reason -- loopback above all.
    rejected: Tuple[str, ...] = ()
    #: Non-fatal remarks, e.g. an unordered shape widened to two directions.
    notes: Tuple[str, ...] = ()

    def hop(self, src: str, dst: str) -> Optional[Hop]:
        """The directed hop, or ``None`` when the probe has no such row."""
        return self.hops.get((str(src), str(dst)))

    def transports(self) -> List[str]:
        return sorted({h.transport for h in self.hops.values() if h.transport})

    def narrowest_bandwidth_gbs(self) -> Rate:
        """The slowest ordered pair among the participating cards.

        The bound a group collective inherits: a ring is only as fast as its
        narrowest wire. Absent when no pair was measured.
        """
        vals = [
            h.bandwidth_gbs.value
            for h in self.hops.values()
            if h.bandwidth_gbs.value is not None
        ]
        if not vals:
            return Rate.absent(self._no_pairs_reason(), unit="GB/s")
        transports = ", ".join(self.transports())
        return Rate.measured(
            min(vals),
            f"pair matrix, narrowest of {len(vals)} ordered pairs"
            + (f" ({transports})" if transports else ""),
            unit="GB/s",
        )

    def worst_latency_us(self) -> Rate:
        """The worst ordered-pair latency. Absent when no pair was measured."""
        vals = [
            h.latency_us.value
            for h in self.hops.values()
            if h.latency_us.value is not None
        ]
        if not vals:
            return Rate.absent(self._no_pairs_reason(), unit="us")
        return Rate.measured(
            max(vals),
            f"pair matrix, worst of {len(vals)} ordered pairs",
            unit="us",
        )

    def absences(self) -> List[str]:
        """Named absences: the missing matrix, or the pairs it does not cover.

        Every ordered pair among the participating cards is expected; the ones
        that are not there are listed by name so the caller can print which
        wire was never measured instead of "some link data is missing".
        """
        if len(self.keys) < 2:
            return []
        if not self.hops:
            return [self._no_pairs_reason()]
        missing = [
            f"{a} -> {b}"
            for a in self.keys
            for b in self.keys
            if a != b and (a, b) not in self.hops
        ]
        if not missing:
            return []
        return [
            "pair matrix is incomplete: no measured hop for "
            + ", ".join(missing[:6])
            + (f" (+{len(missing) - 6} more)" if len(missing) > 6 else "")
        ]

    def _no_pairs_reason(self) -> str:
        return (
            "card-to-card pair matrix — no ordered pair was measured among "
            f"{', '.join(self.keys) or 'the participating cards'}, so the "
            "collective term cannot be predicted and is reported absent"
        )


def _reject_loopback(src: str, dst: str, rejected: List[str]) -> bool:
    """True when this row is not a hop between two different cards.

    A same-card entry is not a wire. Left in, it would win ``min(bandwidth)``
    with a device-local copy rate and make every collective look free -- the
    intra-rig form of the #214/#271 rule against a loopback number wearing a
    wire's label. Rejections are recorded, not silently dropped.
    """
    if src == dst:
        rejected.append(
            f"loopback row {src} -> {src}: a same-card copy is not a hop and "
            "would understate every collective it entered"
        )
        return True
    return False


def pair_matrix_from_card_probe(
    probe: Optional[Mapping],
    keys: Sequence[str],
    *,
    uuid_of_key: Optional[Mapping[str, str]] = None,
) -> PairMatrix:
    """Read the ``card_probe`` artifact's ORDERED pair list.

    Shape: ``probe["pairs"] = [{src_uuid, dst_uuid, bandwidth_gbs,
    latency_us, transport}, ...]``, one row per direction. ``keys`` are the
    caller's card names in its own order; ``uuid_of_key`` maps them to the
    probe's UUIDs when the caller does not key by UUID itself.
    """
    order = [str(k) for k in keys]
    to_uuid = {k: str((uuid_of_key or {}).get(k, k)) for k in order}
    key_of_uuid = {u: k for k, u in to_uuid.items()}
    hops: Dict[Tuple[str, str], Hop] = {}
    rejected: List[str] = []
    for row in (probe or {}).get("pairs") or []:
        src = key_of_uuid.get(str(row.get("src_uuid")))
        dst = key_of_uuid.get(str(row.get("dst_uuid")))
        if src is None or dst is None:
            continue
        if _reject_loopback(src, dst, rejected):
            continue
        bw = row.get("bandwidth_gbs")
        lat = row.get("latency_us")
        transport = str(row.get("transport") or "")
        hops[(src, dst)] = Hop(
            src=src,
            dst=dst,
            bandwidth_gbs=(
                Rate.measured(bw, "card probe, ordered pair", unit="GB/s")
                if bw
                else Rate.absent(
                    f"card probe measured no bandwidth for {src} -> {dst}",
                    unit="GB/s",
                )
            ),
            latency_us=(
                Rate.measured(lat, "card probe, ordered pair", unit="us")
                if lat
                else Rate.absent(
                    f"card probe measured no latency for {src} -> {dst}",
                    unit="us",
                )
            ),
            transport=transport,
        )
    return PairMatrix(
        hops=hops,
        keys=tuple(order),
        source="card_probe",
        rejected=tuple(rejected),
    )


def pair_matrix_from_hardware_profile(
    profile: Optional[Mapping],
    keys: Sequence[str],
    *,
    uuid_of_key: Optional[Mapping[str, str]] = None,
) -> PairMatrix:
    """Read the hardware profile's UNORDERED link map.

    Shape: ``profile["links"]["<uuidA>|<uuidB>"] = {"p2p_gbs": x}`` with the
    UUIDs sorted, plus a ``"__group__"`` row holding all-reduce latencies for
    the whole group rather than for a pair.

    One row therefore describes both directions. It is widened to two hops so
    consumers see one shape, and the widening is stated in
    :attr:`PairMatrix.notes` -- the asymmetry the card probe measures directly
    is not available here, and pretending otherwise is how a plan ends up
    scored on the fast direction of a lopsided link.
    """
    order = [str(k) for k in keys]
    to_uuid = {k: str((uuid_of_key or {}).get(k, k)) for k in order}
    key_of_uuid = {u: k for k, u in to_uuid.items()}
    links = (profile or {}).get("links") or {}
    group = links.get("__group__") or {}
    group_lat = group.get("ar_1mb_us")
    hops: Dict[Tuple[str, str], Hop] = {}
    rejected: List[str] = []
    widened = False
    for raw_key, row in links.items():
        if raw_key == "__group__" or not isinstance(row, Mapping):
            continue
        parts = str(raw_key).split("|")
        if len(parts) != 2:
            continue
        a, b = key_of_uuid.get(parts[0]), key_of_uuid.get(parts[1])
        if a is None or b is None:
            continue
        if _reject_loopback(a, b, rejected):
            continue
        gbs = row.get("p2p_gbs")
        bw = (
            Rate.measured(gbs, "hardware profile, unordered p2p link", unit="GB/s")
            if gbs
            else Rate.absent(
                f"hardware profile has no p2p bandwidth for {a} <-> {b}",
                unit="GB/s",
            )
        )
        # The profile times all-reduce for the GROUP, not per pair. Charging
        # that group number to a single pair would be an invented per-pair
        # latency, so it stays absent and the group figure is carried in the
        # notes for a caller that wants it.
        lat = Rate.absent(
            "the hardware profile measures all-reduce latency for the whole "
            "group, not per pair; no per-pair latency exists in this artifact",
            unit="us",
        )
        for src, dst in ((a, b), (b, a)):
            hops[(src, dst)] = Hop(src, dst, bw, lat, transport="")
        widened = True
    notes: List[str] = []
    if widened:
        notes.append(
            "the hardware profile stores one unordered row per pair; both "
            "directions carry the same number and a real asymmetry between "
            "them is invisible in this artifact (the card probe measures it)"
        )
    if group_lat:
        notes.append(f"group all-reduce latency at 1 MiB: {group_lat} us")
    return PairMatrix(
        hops=hops,
        keys=tuple(order),
        source="hardware_profile",
        rejected=tuple(rejected),
        notes=tuple(notes),
    )


def reconcile_pair_matrices(
    *matrices: PairMatrix,
    tolerance: float = 0.10,
) -> Tuple[PairMatrix, List[str]]:
    """One matrix out of several artifacts, with the disagreements named.

    The rig writes its pair costs into two artifacts with different shapes and
    different probe methods (the card probe's ordered pinned-transfer rate and
    the hardware profile's NCCL p2p rate). Two planners reading one artifact
    each would price the SAME card pair differently and neither would know.

    The earlier argument wins a contested pair -- callers pass their preferred
    artifact first -- and every disagreement beyond ``tolerance`` (relative)
    comes back as a line the caller is expected to surface. A disagreement is
    a measurement question, not something to average away: the two numbers
    describe different transports and the honest answer is to say so.
    """
    if not matrices:
        raise ValueError("reconcile_pair_matrices needs at least one matrix")
    merged: Dict[Tuple[str, str], Hop] = {}
    divergences: List[str] = []
    for matrix in matrices:
        for pair, hop in matrix.hops.items():
            kept = merged.get(pair)
            if kept is None:
                merged[pair] = hop
                continue
            a, b = kept.bandwidth_gbs.value, hop.bandwidth_gbs.value
            if a and b and abs(a - b) > tolerance * max(a, b):
                divergences.append(
                    f"{pair[0]} -> {pair[1]}: {kept.bandwidth_gbs.source} says "
                    f"{a:.2f} GB/s, {hop.bandwidth_gbs.source} says {b:.2f} "
                    f"GB/s ({abs(a - b) / max(a, b):.0%} apart). Keeping the "
                    "first; the two probes measure different transports and "
                    "the gap is a measurement question, not a rounding one."
                )
            if kept.bandwidth_gbs.is_absent and not hop.bandwidth_gbs.is_absent:
                merged[pair] = dataclasses.replace(
                    kept, bandwidth_gbs=hop.bandwidth_gbs
                )
    first = matrices[0]
    return (
        PairMatrix(
            hops=merged,
            keys=first.keys,
            source=" + ".join(m.source for m in matrices if m.source),
            rejected=tuple(r for m in matrices for r in m.rejected),
            notes=tuple(n for m in matrices for n in m.notes),
        ),
        divergences,
    )


# ---------------------------------------------------------------------------
# Composition primitives
# ---------------------------------------------------------------------------

#: Stand-in link bandwidth for a RANKING when the pair matrix is absent.
#:
#: The #216/#264 guard says the roofline never ranks a split. This constant is
#: what keeps that true while still letting a ranking run: the collective term
#: it produces is IDENTICAL for every candidate in one solve -- it depends on
#: the layer count, the hidden size and the rank count, none of which a
#: candidate split varies -- so it shifts every candidate's predicted time by
#: the same additive constant and cannot reorder them.
#:
#: What it must never do is reach an ABSOLUTE number a reader could take for a
#: prediction. Callers that report a time (rather than a ratio) check the
#: matrix for absence first and report absent; this value exists only so a
#: ratio between two candidates stays computable.
ABSENT_LINK_RANKING_PLACEHOLDER_GBS: float = 1e-3

#: The OTHER value the fork substitutes for an absent link, and the reason
#: this module names both instead of quietly picking one.
#:
#: Two code paths answer the same question -- "no pair matrix was measured,
#: what is the link worth?" -- with numbers 80x apart:
#:
#:   * ``0.1`` GB/s, the floor inside ``PerfCostModel._prefill_sharded_time``,
#:     reached from the key solver's ``1e-3`` above;
#:   * ``8.0`` GB/s, ``lever_profiles._FALLBACK_LINK_GBS`` and
#:     ``uneven_perf.apply_auto_performance``, labelled "assumed".
#:
#: Both feed a candidate comparison, and they are NOT equally safe under the
#: #216/#264 guard. The collective term they scale does not depend on the
#: candidate split -- it is ``n_layers * hidden`` and the rank count -- so it
#: is an additive constant and cannot reorder an argmax. But it is NOT
#: invariant for a RATIO: ``lever_profiles._speed_ratios`` divides two
#: predicted times and compares the result against a move threshold, and a
#: collective term 80x larger pulls every ratio toward 1.0. At ``8.0`` GB/s a
#: lever can clear that threshold and at ``0.1`` GB/s the same lever cannot,
#: on identical measured inputs.
#:
#: Unifying the value would change outputs on an unprobed rig, which is a
#: re-tune and not this refactor. The library therefore names the divergence
#: rather than averaging it away; the fix is for the ratio consumer to report
#: the absence instead of ranking through it. See DESIGN_348b §4.
ABSENT_LINK_ASSUMED_GBS: float = 8.0


def ring_factor(ranks: int) -> float:
    """Payload multiplier of a ring all-reduce: ``2(R-1)/R``.

    Each rank sends and receives ``(R-1)/R`` of the payload in the
    reduce-scatter half and again in the all-gather half. Zero below two
    ranks, where there is no collective at all.
    """
    return 0.0 if ranks < 2 else 2.0 * (ranks - 1) / ranks


def allreduce_seconds(
    payload_bytes: float,
    ranks: int,
    bandwidth_gbs: float,
    latency_us: float,
    *,
    efficiency: float = 1.0,
) -> float:
    """One ring all-reduce, priced against a measured pair.

    ``(ranks - 1)`` latencies for the hops plus the ring payload over the
    narrowest wire. ``efficiency`` is the achieved fraction of what the pair
    matrix measured; it is deliberately shipped at 1.0 by the K1 solver (see
    ``key_solver.COLLECTIVE_EFFICIENCY`` for the measured one-sided bias that
    choice carries) and is a parameter here rather than a constant so a
    consumer that fits its own does not have to re-derive the formula.
    """
    if ranks < 2:
        return 0.0
    latency_s = (ranks - 1) * latency_us * 1e-6
    wire_s = ring_factor(ranks) * payload_bytes / (bandwidth_gbs * 1e9 * efficiency)
    return latency_s + wire_s


def apportion_largest_remainder(
    total: int,
    weights: Sequence[float],
    *,
    min_one: bool = True,
) -> List[int]:
    """Split ``total`` into integer counts proportional to ``weights``.

    Largest-remainder (Hamilton) apportionment: floor every ideal share, then
    hand the leftover units to the positions with the largest fractional
    parts. The result sums to ``total`` exactly -- the coverage invariant every
    uneven split rests on, since a lost unit is a dropped token and a created
    one is an out-of-bounds read.

    ``min_one`` guarantees every position at least one unit when
    ``total >= len(weights)``, moving units off the largest holders. A position
    with nothing to do is a degenerate configuration, not a split.

    This is one of TWO rounding rules the fork uses (see
    :func:`cumulative_boundaries`) and they do not always agree; the
    difference is pinned by test rather than papered over.
    """
    n = len(weights)
    if any(w <= 0 for w in weights):
        raise ValueError(f"capacity weights must be positive; got {list(weights)}")
    total_w = math.fsum(weights)
    ideal = [total * w / total_w for w in weights]
    counts = [int(math.floor(x)) for x in ideal]
    remainder = total - sum(counts)
    order = sorted(range(n), key=lambda i: ideal[i] - counts[i], reverse=True)
    for k in range(remainder):
        counts[order[k]] += 1
    if min_one and total >= n:
        donors = sorted(range(n), key=lambda i: counts[i], reverse=True)
        d = 0
        for i in range(n):
            if counts[i] == 0:
                while counts[donors[d]] <= 1:
                    d += 1
                counts[donors[d]] -= 1
                counts[i] = 1
    return counts


def cumulative_boundaries(total: int, weights: Sequence[float]) -> List[int]:
    """Cut points that tile ``[0, total)`` proportionally to ``weights``.

    Cumulative rounding: the i-th boundary is the rounded cumulative share, so
    the chunks tile the range exactly and stay contiguous. Contiguity is the
    reason this rule exists next to :func:`apportion_largest_remainder`: a
    video shard needs a start and a stop on a shared timeline, not just a
    count, and rounding each boundary independently against the cumulative
    total keeps the drift bounded to one unit rather than accumulating.

    Cumulative rounding cannot invert, but it can repeat when a share rounds
    to nothing; the boundaries are clamped so they stay ordered and the empty
    chunk is visible to the caller as ``stop == start``.
    """
    total_weight = math.fsum(weights)
    boundaries: List[int] = []
    cumulative = 0.0
    for weight in weights[:-1]:
        cumulative += weight
        boundaries.append(round(total * cumulative / total_weight))
    boundaries.append(total)
    for i in range(1, len(boundaries)):
        boundaries[i] = max(boundaries[i], boundaries[i - 1])
    return boundaries


def apportion_cumulative(total: int, weights: Sequence[float]) -> List[int]:
    """:func:`cumulative_boundaries` expressed as counts, for comparison with
    :func:`apportion_largest_remainder`.

    The two rules agree on most inputs and disagree on some -- with weights
    ``(1, 3)`` over 10 units, Hamilton gives ``[3, 7]`` and cumulative
    rounding gives ``[2, 8]``, because Python's round-half-to-even sends the
    2.5 boundary down while Hamilton's tie-break sends the spare unit to the
    first position. Which rule is right is a policy question for the consumer;
    what the library removes is the possibility of not knowing which one ran.
    """
    bounds = cumulative_boundaries(total, weights)
    out: List[int] = []
    start = 0
    for stop in bounds:
        out.append(stop - start)
        start = stop
    return out


# ---------------------------------------------------------------------------
# Per-stage rates: the axis a heterogeneous *pipeline* is priced on
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StageRateTable:
    """Measured cost of one pipeline stage on one card, at one resolution.

    :class:`ComputeRates` prices a card with a single number because an LLM
    rank runs the same kind of work throughout. A media pipeline does not: a
    frame passes through decode, super-resolution, resize, interpolation and
    encode, and the ratio between two cards is different for each of them.
    That difference is the entire economic case for splitting a *stage* across
    cards rather than splitting the frame stream -- comparative advantage
    needs a per-stage table or it has nothing to compare.

    Resolution is part of the key rather than an annotation because the same
    stage runs at different sizes within one chain: super-resolution at the
    source size, resize at four times it, interpolation at the target. A
    table keyed only by (stage, card) would silently price the wrong work.

    Every cell is a :class:`Rate`, so a combination nobody measured is a named
    absence rather than a ``KeyError`` at plan time or, worse, an
    extrapolation. The planner is allowed to know that it does not know.
    """

    #: ``(stage, card, resolution)`` -> ms per invocation.
    cells: Mapping[Tuple[str, str, str], Rate]
    #: Card keys, in the caller's order.
    keys: Tuple[str, ...] = ()
    source: str = ""
    #: Run-to-run spread of the measurement session, if it established one.
    #: A planner comparing two assignments whose predicted makespans differ by
    #: less than this is choosing noise.
    noise_floor_pct: Optional[float] = None

    def rate(self, stage: str, card: str, resolution: str) -> Rate:
        cell = self.cells.get((stage, card, str(resolution)))
        if cell is not None:
            return cell
        measured = sorted(
            res for st, cd, res in self.cells if st == stage and cd == card
        )
        return Rate.absent(
            f"no measurement for stage {stage!r} on card {card!r} at "
            f"{resolution}; measured resolutions for that pair: "
            f"{measured or 'none'}",
            unit="ms",
            label=f"{stage}@{card}",
        )

    def ms(self, stage: str, card: str, resolution: str) -> float:
        """The measured value, or :class:`AbsentRate` naming what is missing."""
        return self.rate(stage, card, resolution).require(
            f"{stage} on {card} at {resolution}"
        )

    @property
    def stages(self) -> Tuple[str, ...]:
        return tuple(sorted({stage for stage, _c, _r in self.cells}))

    @property
    def cards(self) -> Tuple[str, ...]:
        return tuple(sorted({card for _s, card, _r in self.cells}))

    def resolutions(self, stage: str) -> Tuple[str, ...]:
        return tuple(sorted({r for s, _c, r in self.cells if s == stage}))

    def absences(self) -> List[str]:
        return [c.source for c in self.cells.values() if c.is_absent]

    def coverage(self, stages: Sequence[str], cards: Sequence[str]) -> List[str]:
        """Which (stage, card) pairs a plan would need and nobody measured.

        The question a Regime-B optimiser has to ask before it starts, because
        a stage assignment is only comparable against another if both are
        priced from measurements. Answering it up front turns "the plan is
        surprising" into "the plan could not have been made".
        """
        gaps: List[str] = []
        for stage in stages:
            for card in cards:
                if not any(s == stage and c == card for s, c, _r in self.cells):
                    gaps.append(f"{stage} on {card}")
        return gaps

    def advantage(self, stage: str, resolution: str) -> Dict[str, float]:
        """Each card's rate for one stage, normalised to the fastest as 1.0.

        Comparative advantage in the form the assignment question is actually
        asked in: not "which card is fastest" -- that is usually the same card
        for everything -- but "on which stage is a given card *least*
        disadvantaged", which is what decides whether specialising beats
        replicating.
        """
        measured = {
            card: cell.value
            for (s, card, r), cell in self.cells.items()
            if s == stage and r == str(resolution) and not cell.is_absent
        }
        if not measured:
            return {}
        best = min(measured.values())
        return {card: round(ms / best, 4) for card, ms in measured.items()}


def stage_rates_from_samples(
    samples: Iterable[Mapping],
    *,
    source: str = "probe_report",
    noise_floor_pct: Optional[float] = None,
    key_of: Optional[Mapping[str, str]] = None,
) -> StageRateTable:
    """Build a :class:`StageRateTable` from probe samples.

    A sample is any mapping carrying ``stage``, ``card``, ``resolution`` and
    ``ms_per_frame`` -- the shape
    :class:`sglang.srt.video_enhance.probes.Sample` serialises to, so a
    directory of P1 reports loads with no adapter in between.

    ``key_of`` renames the card as recorded (a device name, which is not
    unique on a rig with two identical cards) to the key the planner indexes
    by (an NVML index or a UUID, which is). Without it the recorded string is
    used as-is.

    Repeated cells keep the **fastest** observation. A slow repeat is
    contention or a cold cache; the floor is the closest thing a short probe
    gets to the card's actual capability, and a planner that took the mean
    would price every card by how busy the rig was when it was measured.
    """
    cells: Dict[Tuple[str, str, str], Rate] = {}
    keys: List[str] = []
    for sample in samples:
        stage = sample.get("stage")
        card = sample.get("card")
        resolution = sample.get("resolution")
        ms = sample.get("ms_per_frame")
        if not stage or not card or not resolution:
            continue
        card = (key_of or {}).get(card, card)
        if card not in keys:
            keys.append(card)
        key = (stage, card, str(resolution))
        if ms is None or not (ms > 0.0):
            # A non-positive or NaN cell is not a measurement. Recording it as
            # an absence keeps it visible; recording it as a rate would hand
            # that card an unbounded share of the work.
            #
            # It must not clobber a good value already recorded for this cell.
            # A probe grid that measures a point twice and fails once has
            # measured it -- letting the failure win would delete a real
            # number because of a transient, and the deletion would show up
            # later as a plan that refused to price a card it could price.
            if key not in cells:
                cells[key] = Rate.absent(
                    f"probe recorded no usable time for {stage} on {card} at "
                    f"{resolution} ({ms!r})",
                    unit="ms",
                    label=f"{stage}@{card}",
                )
            continue
        existing = cells.get(key)
        if existing is not None and not existing.is_absent and existing.value <= ms:
            continue
        cells[key] = Rate.measured(
            float(ms), source, unit="ms", label=f"{stage}@{card}"
        )
    return StageRateTable(
        cells=cells,
        keys=tuple(keys),
        source=source,
        noise_floor_pct=noise_floor_pct,
    )


def stage_rates_from_reports(reports: Iterable[Mapping]) -> StageRateTable:
    """Merge several probe reports -- typically one per card -- into one table.

    The noise floor kept is the **largest** any contributing report declared.
    Two cards measured in different sessions have different floors, and a
    comparison across them is only as trustworthy as the worse one.
    """
    samples: List[Mapping] = []
    floors: List[float] = []
    sources: List[str] = []
    for report in reports:
        samples.extend(report.get("samples", ()))
        floor = report.get("noise_floor_pct")
        if floor is not None:
            floors.append(float(floor))
        host = report.get("host") or {}
        name = host.get("card_name") or host.get("nvml_uuid") or "unknown"
        sources.append(str(name))
    table = stage_rates_from_samples(
        samples,
        source="probe_reports: " + ", ".join(sources) if sources else "probe_reports",
        noise_floor_pct=max(floors) if floors else None,
    )
    return table


# ---------------------------------------------------------------------------
# Bundle: both axes, resolved once (the #302 entry point)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CostSources:
    """Both cost axes over one set of cards, resolved from disk.

    What a new class planner needs and all it needs: per-card compute rate and
    the measured hop matrix, each item provenance-tagged. #302 expert
    placement is expected to start here (see the module docstring).
    """

    #: Per-card compute rates in the checkpoint's format.
    compute: ComputeRates
    #: Measured card-to-card costs.
    links: PairMatrix
    #: Card keys, the order both axes are indexed in.
    keys: Tuple[str, ...]
    #: Pair-cost disagreements between artifacts, if several were read.
    divergences: Tuple[str, ...] = ()

    def absences(self) -> List[str]:
        """Every named absence across both axes, ready to print."""
        return list(self.compute.absences()) + list(self.links.absences())


def load_cost_sources(
    card_keys: Sequence[str],
    *,
    fmt: str = FORMAT_DENSE_BF16,
    family_formats: Optional[Mapping[str, str]] = None,
    card_probe: Optional[Mapping] = None,
    hardware_profile: Optional[Mapping] = None,
    uuid_of_key: Optional[Mapping[str, str]] = None,
) -> CostSources:
    """Resolve both cost axes for ``card_keys`` from the artifacts on hand.

    Pass the artifacts a caller already has; whatever is not passed is read
    cache-only from disk. Both pair-matrix shapes are read when both exist,
    and the card probe wins a contested pair because it measures the ORDERED
    direction the collective actually takes -- with the disagreement reported
    rather than averaged.
    """
    keys = [str(k) for k in card_keys]
    if hardware_profile is None:
        from sglang.srt import uneven_perf  # noqa: PLC0415  (heavy: pulls torch)

        hardware_profile, _inventory = uneven_perf.get_cached_hardware_profile()

    compute = compute_rates_for_cards(
        keys, fmt=fmt, family_formats=family_formats, profile=hardware_profile
    )
    matrices = []
    if card_probe:
        matrices.append(
            pair_matrix_from_card_probe(card_probe, keys, uuid_of_key=uuid_of_key)
        )
    matrices.append(
        pair_matrix_from_hardware_profile(
            hardware_profile, keys, uuid_of_key=uuid_of_key
        )
    )
    links, divergences = reconcile_pair_matrices(*matrices)
    return CostSources(
        compute=compute,
        links=links,
        keys=tuple(keys),
        divergences=tuple(divergences),
    )
