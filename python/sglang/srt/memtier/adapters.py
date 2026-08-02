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
"""Reading the measurements this tree already takes (#407 slice 1).

#407 adds no probe. Three harnesses already write numbers to disk, each with a
schema, a status vocabulary and an absent-value rule of its own, and slice 1's
job is to *read* them into tier records rather than to measure anything a
second time. A parallel artifact format would be the #348b defect one layer
down -- two sources of the same physical fact, disagreeing quietly.

Sources, and the exact key each one is written under:

``card_probe`` (#213 / E4) -- :func:`from_card_probe`
    ``{"version": 1, "uuids": [...], "cards": [{"uuid", "name", "membw_read_gbs",
    "membw_gemv_gbs", "h2d_gbs", "d2h_gbs", ...}], "pairs": [{"src_uuid",
    "dst_uuid", "bandwidth_gbs", "latency_us", "transport"}]}``, cached at
    ``~/.cache/sglang/card_probe-<sha1>.json``. UUID-keyed throughout.
    Contributes ``caps.bandwidth_gbs`` to DEVICE tiers from ``membw_read_gbs``.

``rig artifact`` (#271) -- :func:`from_rig_artifact`
    ``{"schema": "htsglang-rig-artifact/v1", "measurements": [{"id", "label",
    "source", "unit", "value", "status", "n", "spread_pct", ...}]}``. Rows are
    routed to a tier field by :data:`ARTIFACT_ROUTES`, keyed on the measurement
    id, so a new comm-suite arm becomes one table row and no code.

``capability_matrix`` (#278 / p2p_readiness) -- :func:`from_capability_matrix`
    ``{"schema_version": 3, "kind": "capability_matrix", "devices": [{"uuid",
    "pci_bus_id", ...}], "directed_pairs": [{"src_pci", "dst_pci",
    "effective_max_single_copy_bytes", ...}]}``. BDF-keyed rows plus a
    ``devices`` table that carries both names, so the BDF -> UUID resolution is
    inside the artifact and this adapter needs no live IdentityMap.
    Contributes ``caps.aperture_bytes``.

Three rules hold for all three, and they are why these are adapters rather
than parsers:

1. **Nothing writes a tier field directly.** Every adapter returns
   :class:`~sglang.srt.memtier.probe.ProbeOutcome` values, and the only way one
   reaches a record is :func:`~sglang.srt.memtier.probe.apply_outcome`, which
   refuses an ``ok`` outcome that does not carry a ``MEASURED`` rate and
   refuses to overwrite an existing measurement with a different one.
2. **A row that did not succeed becomes an outcome that did not succeed.** The
   ``ok | warn | error | absent`` vocabularies of comm-suite and of this
   package are the same four words on purpose; a mapping table would have been
   a place for them to drift.
3. **A number is never re-labelled.** An ``h2d`` figure is a property of the
   PCIe edge between a card and the host; it does not become the host tier's
   DRAM bandwidth by being the only host-adjacent number available. Rows with
   no honest node-level target are reported in
   :attr:`IngestReport.unrouted`, not coerced.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import msgspec

from sglang.srt.memtier.probe import ProbeOutcome, ProbeTarget, apply_outcome
from sglang.srt.memtier.tiers import TierDescriptor, TierId, TierKind
from sglang.srt.planner.cost_model import Rate

__all__ = [
    "ARTIFACT_ROUTES",
    "ArtifactRoute",
    "IngestReport",
    "apply_outcomes",
    "from_capability_matrix",
    "from_card_probe",
    "from_rig_artifact",
]

#: comm-suite / rig-artifact statuses that carry a usable number. ``warn``
#: does: the reservation is in the row's note, and dropping the value would
#: lose a measurement over a caveat.
_USABLE_STATUSES = ("ok", "warn")

_CARD_PROBE_VERSION = 1
_CAPABILITY_MATRIX_KIND = "capability_matrix"
#: ``capability_matrix.py`` writes its rows under ``directed_pairs``; the
#: legacy name is still read, exactly as ``barlink_path_rates`` reads it, so
#: an older artifact stays loadable and neither reader invents a third key.
_CAPABILITY_ROW_KEYS = ("directed_pairs", "pairs")


class IngestReport(msgspec.Struct, frozen=True, kw_only=True):
    """What an adapter found, what it could not use, and why.

    ``unrouted`` and ``skipped`` are separate because they are different
    findings: a row nobody declared a destination for is a gap in
    :data:`ARTIFACT_ROUTES`, while a row the producer marked absent is a gap in
    the measurement. Folding them together is how the first one stops being
    noticed.
    """

    outcomes: Tuple[ProbeOutcome, ...] = ()
    #: Rows whose id no route claims: ``(row id, reason)``.
    unrouted: Tuple[Tuple[str, str], ...] = ()
    #: Rows a route claimed but which carried no usable value.
    skipped: Tuple[Tuple[str, str], ...] = ()
    #: Structural defects in the artifact itself.
    errors: Tuple[str, ...] = ()
    source: str = ""

    def render(self) -> str:
        lines = [
            f"{self.source or 'artifact'}: {len(self.outcomes)} outcome(s), "
            f"{len(self.unrouted)} unrouted, {len(self.skipped)} skipped, "
            f"{len(self.errors)} error(s)"
        ]
        lines.extend(f"  error: {e}" for e in self.errors)
        lines.extend(f"  unrouted {i}: {r}" for i, r in self.unrouted)
        lines.extend(f"  skipped {i}: {r}" for i, r in self.skipped)
        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "outcomes": [o.to_json() for o in self.outcomes],
            "unrouted": [{"id": i, "reason": r} for i, r in self.unrouted],
            "skipped": [{"id": i, "reason": r} for i, r in self.skipped],
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# card_probe (#213)
# ---------------------------------------------------------------------------


def from_card_probe(
    document: Mapping[str, Any], tiers: Sequence[TierDescriptor]
) -> IngestReport:
    """Per-card membw rows -> DEVICE tier bandwidth.

    ``membw_read_gbs`` is a NODE property: it is what the card's own memory
    delivers to its own SMs, which is exactly what ``caps.bandwidth_gbs`` means
    for a ``vram:`` tier. The other three numbers in the same row are NOT, and
    are deliberately left alone:

    * ``membw_gemv_gbs`` measures the same memory through a different access
      pattern. It lands in ``properties`` at profile-writing time, never in the
      cap, because two numbers in one field is one number too many;
    * ``h2d_gbs`` / ``d2h_gbs`` are the PCIe EDGE between this card and the
      host. Writing either into the host tier's bandwidth would put a wire's
      number under a memory's name -- the #214/#271 rule this tree already
      enforces for loopback rows (``cost_model._reject_loopback``).

    The ``pairs`` list is an edge fact and belongs to ``cost_model.PairMatrix``,
    which already parses this exact shape (``pair_matrix_from_card_probe``).
    This adapter does not re-parse it.
    """
    outcomes: List[ProbeOutcome] = []
    unrouted: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, str]] = []
    errors: List[str] = []
    version = document.get("version")
    if version != _CARD_PROBE_VERSION:
        errors.append(
            f"card_probe artifact carries version {version!r}, this reader "
            f"understands {_CARD_PROBE_VERSION}. Reinterpreting fields across "
            "a version change is how a renamed column becomes a wrong number; "
            "nothing was read."
        )
        return IngestReport(errors=tuple(errors), source="card_probe")
    by_uuid = {
        t.parsed.card_key: t
        for t in tiers
        if t.kind is TierKind.DEVICE and t.parsed.card_key
    }
    for index, row in enumerate(document.get("cards") or ()):
        if not isinstance(row, Mapping):
            errors.append(f"cards[{index}] is not an object")
            continue
        uuid = str(row.get("uuid") or "")
        tier = by_uuid.get(uuid)
        if tier is None:
            unrouted.append(
                (
                    f"cards[{index}]",
                    f"card {uuid or '<no uuid>'} is not a tier in this registry",
                )
            )
            continue
        value = row.get("membw_read_gbs")
        if not value:
            skipped.append(
                (
                    tier.id,
                    "card_probe row carries no membw_read_gbs; the membw arm "
                    "did not produce a read figure for this card",
                )
            )
            continue
        outcomes.append(
            ProbeOutcome(
                probe_id="M9",
                tier_id=tier.id,
                status="ok",
                rate=Rate.measured(
                    float(value),
                    f"rigmon card_probe membw read arm, card {uuid} "
                    f"({row.get('name') or tier.card_model})",
                    unit="GB/s",
                ),
            )
        )
    return IngestReport(
        outcomes=tuple(outcomes),
        unrouted=tuple(unrouted),
        skipped=tuple(skipped),
        errors=tuple(errors),
        source="card_probe",
    )


# ---------------------------------------------------------------------------
# rig artifact (#271)
# ---------------------------------------------------------------------------


class ArtifactRoute(msgspec.Struct, frozen=True, kw_only=True):
    """One rig-artifact measurement id, and the tier field it fills."""

    #: Matched against ``Measurement.id`` as an exact string or, when it ends
    #: in ``/``, as a prefix. Exact by default: a prefix that swallowed a
    #: sibling arm would silently route the wrong number.
    id: str
    probe_id: str
    target: ProbeTarget
    kinds: Tuple[TierKind, ...]
    unit: str
    #: The unit the artifact writes. A row in any other unit is skipped rather
    #: than converted: the two comm-suite unit strings in this tree are
    #: ``"us (median)"`` and ``"GB/s"``, and guessing a conversion from a free
    #: text field is how a microsecond becomes a millisecond.
    artifact_unit: str
    #: When set, only tiers carrying this ``properties`` entry are filled.
    requires_property: Tuple[str, str] = ("", "")

    def matches(self, measurement_id: str) -> bool:
        if self.id.endswith("/"):
            return measurement_id.startswith(self.id)
        return measurement_id == self.id

    def applies_to(self, tier: TierDescriptor) -> bool:
        if tier.kind not in self.kinds:
            return False
        key, wanted = self.requires_property
        return not key or tier.properties.get(key) == wanted


#: The declared routing table. Deliberately small and deliberately explicit:
#: every row here is a claim that a specific harness arm measures a specific
#: node property, and a claim of that kind should cost a review, not a regex.
#:
#: The ids follow ``comm_suite.to_sections``'s own spelling,
#: ``comm/<arm_id>/<cell_name>``. **None of the three arms exists yet** -- M1,
#: M4 and M5 are the three measurements DESIGN_407 §4 says the tree does not
#: have. Declaring the route before the arm is the point: it fixes the arm id
#: and the cell name the arm must emit, so adding the measurement is a
#: comm-suite change and not also a memtier change. Until the arm exists these
#: rows simply never match, and the tier fields stay ABSENT naming the probe.
ARTIFACT_ROUTES: Tuple[ArtifactRoute, ...] = (
    ArtifactRoute(
        id="comm/memtier_host_dram/read",
        probe_id="M4",
        target=ProbeTarget.BANDWIDTH,
        kinds=(TierKind.HOST,),
        unit="GB/s",
        artifact_unit="GB/s",
    ),
    ArtifactRoute(
        id="comm/memtier_nvme/read_latency",
        probe_id="M5",
        target=ProbeTarget.LATENCY,
        kinds=(TierKind.FILESYSTEM,),
        unit="us",
        artifact_unit="us (median)",
        requires_property=("medium", "nvme"),
    ),
    ArtifactRoute(
        id="comm/memtier_bar1_ladder/point_latency",
        probe_id="M1",
        target=ProbeTarget.LATENCY,
        kinds=(TierKind.DEVICE,),
        unit="us",
        artifact_unit="us (median)",
    ),
)


def from_rig_artifact(
    document: Mapping[str, Any],
    tiers: Sequence[TierDescriptor],
    *,
    routes: Sequence[ArtifactRoute] = ARTIFACT_ROUTES,
) -> IngestReport:
    """Rig-artifact measurement rows -> tier caps, via :data:`ARTIFACT_ROUTES`.

    A shared artifact names no card -- it is anonymised by construction
    (``rig_artifact.assert_anonymized`` strips UUIDs, hostnames and PCI
    addresses before a digest leaves the machine). The tier therefore comes
    from the row's own ``context["tier_id"]`` when the producer recorded one,
    and otherwise from the route, but only when the route matches exactly ONE
    tier. A route matching several tiers SKIPS them all rather than picking:
    writing a measured host-DRAM figure onto the wrong one of two host tiers is
    worse than leaving both absent.
    """
    outcomes: List[ProbeOutcome] = []
    unrouted: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, str]] = []
    errors: List[str] = []
    by_id = {t.id: t for t in tiers}
    for index, row in enumerate(document.get("measurements") or ()):
        if not isinstance(row, Mapping):
            errors.append(f"measurements[{index}] is not an object")
            continue
        row_id = str(row.get("id") or "")
        route = next((r for r in routes if r.matches(row_id)), None)
        if route is None:
            unrouted.append(
                (
                    row_id or f"measurements[{index}]",
                    "no ARTIFACT_ROUTES entry claims this id; the row names a "
                    "measurement no tier field is declared to receive",
                )
            )
            continue
        status = str(row.get("status") or "ok")
        if status not in _USABLE_STATUSES:
            skipped.append(
                (
                    row_id,
                    f"the producer marked this row {status!r}: "
                    f"{row.get('note') or 'no reason recorded'}",
                )
            )
            continue
        value = row.get("value")
        if value is None:
            skipped.append((row_id, f"row is {status!r} but carries no value"))
            continue
        artifact_unit = str(row.get("unit") or "")
        if artifact_unit != route.artifact_unit:
            skipped.append(
                (
                    row_id,
                    f"row is in {artifact_unit!r}, the route expects "
                    f"{route.artifact_unit!r}; a unit is not converted on a guess",
                )
            )
            continue
        target_tier = _resolve_tier(row, route, tiers, by_id)
        if isinstance(target_tier, str):
            skipped.append((row_id, target_tier))
            continue
        outcomes.append(
            ProbeOutcome(
                probe_id=route.probe_id,
                tier_id=target_tier.id,
                status="ok",
                rate=Rate.measured(
                    float(value),
                    _artifact_source(row, route),
                    unit=route.unit,
                ),
            )
        )
    return IngestReport(
        outcomes=tuple(outcomes),
        unrouted=tuple(unrouted),
        skipped=tuple(skipped),
        errors=tuple(errors),
        source=str(document.get("schema") or "rig_artifact"),
    )


def _resolve_tier(
    row: Mapping[str, Any],
    route: ArtifactRoute,
    tiers: Sequence[TierDescriptor],
    by_id: Mapping[TierId, TierDescriptor],
):
    """The tier this row is about, or the sentence explaining why nobody knows."""
    raw_context = row.get("context")
    context: Mapping[str, Any] = raw_context if isinstance(raw_context, Mapping) else {}
    named = context.get("tier_id") or context.get("tier")
    if named:
        tier = by_id.get(str(named))
        if tier is None:
            return f"row names tier {named!r}, which this registry does not declare"
        if not route.applies_to(tier):
            return (
                f"row names tier {tier.id}, which the route does not apply to "
                f"(kinds {[k.value for k in route.kinds]})"
            )
        return tier
    candidates = [t for t in tiers if route.applies_to(t)]
    if not candidates:
        return (
            f"no tier of kind {[k.value for k in route.kinds]} in this registry "
            "can receive this row"
        )
    if len(candidates) > 1:
        return (
            "the row names no tier and "
            f"{len(candidates)} tiers could receive it "
            f"({', '.join(t.id for t in candidates)}); a measurement is not "
            "assigned by guessing which one it was taken on"
        )
    return candidates[0]


def _artifact_source(row: Mapping[str, Any], route: ArtifactRoute) -> str:
    parts = [f"rig artifact row {row.get('id')}"]
    if row.get("source"):
        parts.append(f"source {row['source']}")
    if row.get("n"):
        parts.append(f"n={row['n']}")
    if row.get("spread_pct") is not None:
        parts.append(f"spread {row['spread_pct']}%")
    if row.get("taken_at"):
        parts.append(str(row["taken_at"]))
    if str(row.get("status")) == "warn":
        parts.append(f"WARN: {row.get('note') or 'reservation not recorded'}")
    parts.append(f"probe {route.probe_id}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# capability matrix (#278 / p2p_readiness)
# ---------------------------------------------------------------------------


def from_capability_matrix(
    document: Mapping[str, Any], tiers: Sequence[TierDescriptor]
) -> IngestReport:
    """Directed BAR1 rows -> DEVICE tier ``aperture_bytes``.

    The artifact is directed and BDF-keyed; a tier's aperture is neither. The
    reduction is the **minimum effective window any source has into this
    card**, and it is a minimum rather than a maximum or a mean for the reason
    #286's window policy exists: the aperture decides whether a transfer is
    rejected or chunked, and a tier that advertised the widest window some
    peer happens to enjoy would let a mover attempt a copy through the narrow
    one. The sources it was taken over are named in the rate's ``source``, so
    the reduction is auditable rather than folded away.

    The BDF -> UUID resolution comes out of the artifact's own ``devices``
    table, which carries both names for every card. No live IdentityMap is
    needed and none is consulted -- an artifact read on a different boot must
    not be re-keyed through the current enumeration.
    """
    outcomes: List[ProbeOutcome] = []
    unrouted: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, str]] = []
    errors: List[str] = []
    kind = document.get("kind")
    if kind != _CAPABILITY_MATRIX_KIND:
        errors.append(
            f"payload kind is {kind!r}, expected {_CAPABILITY_MATRIX_KIND!r} "
            f"(schema_version {document.get('schema_version')!r})"
        )
        return IngestReport(errors=tuple(errors), source="capability_matrix")
    uuid_by_bdf: Dict[str, str] = {}
    for device in document.get("devices") or ():
        if (
            isinstance(device, Mapping)
            and device.get("pci_bus_id")
            and device.get("uuid")
        ):
            uuid_by_bdf[_norm_bdf(str(device["pci_bus_id"]))] = str(device["uuid"])
    if not uuid_by_bdf:
        errors.append(
            "the artifact's devices table carries no pci_bus_id -> uuid pairs, "
            "so its BDF-keyed rows cannot be resolved to a tier identity. "
            "Re-keying them through the live enumeration is refused: NVML "
            "order is not stable across boots."
        )
        return IngestReport(errors=tuple(errors), source="capability_matrix")
    rows = _capability_rows(document)
    if rows is None:
        errors.append(
            f"no rows under any of {_CAPABILITY_ROW_KEYS} (keys present: "
            f"{sorted(document)}); an empty aperture table is indistinguishable "
            "from a rig with no P2P and is not read as one"
        )
        return IngestReport(errors=tuple(errors), source="capability_matrix")
    by_uuid = {
        t.parsed.card_key: t
        for t in tiers
        if t.kind is TierKind.DEVICE and t.parsed.card_key
    }
    inbound: Dict[str, List[Tuple[str, int]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"row[{index}] is not an object")
            continue
        src, dst = row.get("src_pci"), row.get("dst_pci")
        if not src or not dst:
            errors.append(f"row[{index}]: missing src_pci/dst_pci")
            continue
        effective = row.get("effective_max_single_copy_bytes")
        if effective is None:
            skipped.append(
                (
                    f"{src}->{dst}",
                    "no effective aperture measured; the nominal BAR1 fields "
                    "in this row are not read -- a nominal window is not an "
                    "effective one",
                )
            )
            continue
        dst_uuid = uuid_by_bdf.get(_norm_bdf(str(dst)))
        if dst_uuid is None:
            unrouted.append(
                (f"{src}->{dst}", f"destination {dst} is not in the devices table")
            )
            continue
        inbound.setdefault(dst_uuid, []).append((str(src), int(effective)))
    for uuid, entries in sorted(inbound.items()):
        tier = by_uuid.get(uuid)
        if tier is None:
            unrouted.append((uuid, "card is not a tier in this registry"))
            continue
        narrowest = min(entries, key=lambda e: e[1])
        outcomes.append(
            ProbeOutcome(
                probe_id="M2",
                tier_id=tier.id,
                status="ok",
                rate=Rate.measured(
                    float(narrowest[1]),
                    f"p2p_readiness capability_matrix, narrowest inbound "
                    f"effective window into {uuid} over "
                    f"{len(entries)} source(s) "
                    f"({', '.join(sorted(s for s, _ in entries))}); "
                    f"narrowest is from {narrowest[0]}",
                    unit="bytes",
                ),
            )
        )
    return IngestReport(
        outcomes=tuple(outcomes),
        unrouted=tuple(unrouted),
        skipped=tuple(skipped),
        errors=tuple(errors),
        source="capability_matrix",
    )


def _capability_rows(document: Mapping[str, Any]) -> Optional[Sequence[Any]]:
    for key in _CAPABILITY_ROW_KEYS:
        rows = document.get(key)
        if isinstance(rows, list):
            return rows
    return None


def _norm_bdf(bus_id: str) -> str:
    """Lower-cased, domain-qualified BDF. ``01:00.0`` and ``0000:01:00.0`` are
    the same slot, and the two producers in this tree spell it both ways."""
    text = bus_id.strip().lower()
    parts = text.split(":")
    if len(parts) == 2:
        text = f"0000:{text}"
    return text


# ---------------------------------------------------------------------------
# folding outcomes back into records
# ---------------------------------------------------------------------------


def apply_outcomes(
    tiers: Iterable[TierDescriptor], outcomes: Iterable[ProbeOutcome]
) -> Tuple[Tuple[TierDescriptor, ...], Tuple[str, ...]]:
    """Fold every outcome into its tier. Returns the new records and refusals.

    A :class:`~sglang.srt.memtier.probe.ProvenanceUpgradeRefused` is collected
    rather than raised, because one artifact disagreeing with an existing
    measurement must not abandon the other twenty rows -- but it is returned,
    not logged and dropped, so the caller cannot fail to see it.
    """
    from sglang.srt.memtier.probe import ProvenanceUpgradeRefused

    records = {t.id: t for t in tiers}
    order = list(records)
    refusals: List[str] = []
    for outcome in outcomes:
        tier = records.get(outcome.tier_id)
        if tier is None:
            refusals.append(
                f"outcome for {outcome.tier_id} has no such tier in this registry"
            )
            continue
        try:
            records[outcome.tier_id] = apply_outcome(tier, outcome)
        except ProvenanceUpgradeRefused as exc:
            refusals.append(str(exc))
    return tuple(records[i] for i in order), tuple(refusals)
