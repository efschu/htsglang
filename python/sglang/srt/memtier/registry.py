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
"""The tier registry: what exists, what it costs, what it refuses (#407 cut 1).

The registry answers two questions and declines a third. It answers *what
memories exist* and *what does each cost*; it does not answer *what should we
do*. :meth:`TierRegistry.select` returns an ordered candidate list **with its
ordering key exposed**, plus a refusal for every tier it rejected and the
reason -- so a policy layer (#305's arbiter, #286's policy half, #363's
controller) can disagree with the order without re-deriving the facts. The
shape is copied from ``offload_movement._select_target``'s ``MovementError``,
which already names every rung's refusal individually.

Scope of cut 1, stated so the gaps read as decisions rather than omissions:

*   **No staging graph.** DESIGN_407 §3.5 resolves contradiction C1 -- #224's
    "``local`` must be first" law versus #286's "``peer_vram`` above
    ``host_ram``" ladder -- by making the order a shortest path over a graph
    of the moves that physically exist. That belongs to cut 3, together with
    the consumer it fixes. Until then the ordering key is measured bandwidth
    with provenance ahead of it, and :attr:`TierCandidate.order_key` is public
    precisely so the substitution is visible when it happens.
*   **No writes.** The registry reads the cross-process ledger; reserving
    against a tier goes through :mod:`sglang.srt.memtier.reservations`, and no
    consumer is wired to it yet.
*   **No picking.** ``select`` never returns one tier.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import msgspec

from sglang.srt.memtier.bootstrap import bootstrap_profile, default_host_name
from sglang.srt.memtier.fingerprint import fingerprint_from_facts
from sglang.srt.memtier.profile import (
    LocalFacts,
    RigProfile,
    apply_local_facts,
    bind_device_tiers,
)
from sglang.srt.memtier.profile_store import ProfileSelection, select_profile
from sglang.srt.memtier.tiers import (
    PayloadClass,
    TierDescriptor,
    TierId,
    TierKind,
    admission_refusal,
)
from sglang.srt.planner.cost_model import Provenance, Rate
from sglang.srt.planner.rig_coupling import GateRow

__all__ = [
    "Refusal",
    "RefusalRule",
    "TierCandidate",
    "TierQuery",
    "TierRegistry",
    "TierSelection",
    "UnknownTier",
]


class UnknownTier(KeyError):
    """A tier id nobody declared. Never resolved to a near match."""


class RefusalRule(str, enum.Enum):
    """Why a tier was rejected. One rule per reason, so a refusal is countable."""

    HEALTH = "health"
    VOLATILITY = "volatility"
    OBJECT_CLASS = "object_class"
    CAPACITY = "capacity"
    BANDWIDTH_ABSENT = "bandwidth_absent"
    BANDWIDTH_BELOW = "bandwidth_below"
    BANDWIDTH_UNMEASURED = "bandwidth_unmeasured"
    KIND = "kind"
    HOST = "host"
    PROPERTY = "property"


class Refusal(msgspec.Struct, frozen=True, kw_only=True):
    """One tier that was considered and rejected, with the reason."""

    tier_id: TierId
    rule: RefusalRule
    reason: str

    def to_json(self) -> Dict[str, Any]:
        return {"tier_id": self.tier_id, "rule": self.rule.value, "reason": self.reason}


#: Provenance rank used by the ordering key. An estimate never outranks a
#: measurement of similar cost, so the registry cannot prefer a guess.
_PROVENANCE_RANK: Mapping[Provenance, int] = {
    Provenance.MEASURED: 0,
    Provenance.ESTIMATE: 1,
    Provenance.ABSENT: 2,
}


class TierCandidate(msgspec.Struct, frozen=True, kw_only=True):
    """A tier that passed every filter, and the key it was ordered by."""

    tier: TierDescriptor
    #: ``(provenance_rank, -bandwidth_gbs, tier_id)``. Public so a policy
    #: layer can re-sort by its own key rather than reverse-engineer this one.
    order_key: Tuple[int, float, str]
    bandwidth_gbs: Rate
    headroom_bytes: Rate
    #: Non-fatal observations, e.g. a ``warn`` health verdict.
    notes: Tuple[str, ...] = ()

    @property
    def tier_id(self) -> TierId:
        return self.tier.id

    def to_json(self) -> Dict[str, Any]:
        return {
            "tier_id": self.tier.id,
            "order_key": list(self.order_key),
            "bandwidth_gbs": self.bandwidth_gbs.or_none(),
            "bandwidth_provenance": self.bandwidth_gbs.provenance.value,
            "headroom_bytes": self.headroom_bytes.or_none(),
            "notes": list(self.notes),
        }


class TierSelection(msgspec.Struct, frozen=True, kw_only=True):
    """Candidates in order, plus every refusal that was made getting there."""

    candidates: Tuple[TierCandidate, ...] = ()
    refusals: Tuple[Refusal, ...] = ()

    @property
    def tier_ids(self) -> Tuple[TierId, ...]:
        return tuple(c.tier.id for c in self.candidates)

    def refusal_for(self, tier_id: TierId) -> Optional[Refusal]:
        for refusal in self.refusals:
            if refusal.tier_id == tier_id:
                return refusal
        return None

    def render(self) -> str:
        """The itemisation a refusal (or a boot log) prints."""
        lines = [f"{len(self.candidates)} tier(s) admitted:"]
        lines.extend(
            f"  {c.tier.id:<28} {c.bandwidth_gbs.provenance.value:<9} "
            f"{'-' if c.bandwidth_gbs.is_absent else c.bandwidth_gbs.value} GB/s"
            for c in self.candidates
        )
        if self.refusals:
            lines.append(f"{len(self.refusals)} tier(s) refused:")
            lines.extend(
                f"  {r.tier_id:<28} {r.rule.value:<20} {r.reason}"
                for r in self.refusals
            )
        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        return {
            "candidates": [c.to_json() for c in self.candidates],
            "refusals": [r.to_json() for r in self.refusals],
        }


class TierQuery(msgspec.Struct, frozen=True, kw_only=True):
    """What a caller needs. Every field narrows; none of them picks."""

    #: What the content needs from wherever it comes to rest. Required: there
    #: is no "unspecified" volatility, because the default would silently be
    #: the most permissive one.
    payload: PayloadClass
    #: Bytes the caller intends to place. ``0`` asks only about admissibility.
    bytes_needed: int = 0
    #: An ``OFFLOAD_CLASSES`` member, when the caller has one. Hibernate
    #: images and HiCache pages do not, and are gated by volatility alone.
    object_class: Optional[str] = None
    #: GB/s floor. A tier whose bandwidth is ABSENT is REFUSED against a
    #: floor, never assumed to clear it (#286: "an unmeasured path is NEVER
    #: assumed usable for parking, however tempting").
    min_bandwidth_gbs: Optional[float] = None
    #: Refuse an ESTIMATE as well, for callers that will act on the number.
    require_measured_bandwidth: bool = False
    #: Admit tiers whose bandwidth is unknown when no floor was asked for.
    #: Off by default: an inventory read wants :meth:`TierRegistry.tiers`.
    allow_unmeasured_bandwidth: bool = False
    #: The tier the bytes are coming from -- the relation that makes a card's
    #: VRAM "local" to one rank and "peer" to the next.
    origin: Optional[TierId] = None
    kinds: Tuple[TierKind, ...] = ()
    hosts: Tuple[str, ...] = ()
    #: Required :attr:`TierDescriptor.properties` entries, e.g.
    #: ``{"flock": "yes"}`` for #89. A capability filter, never a name check.
    require: Mapping[str, str] = {}


class TierRegistry:
    """The enumeration. Read-only in cut 1.

    Construction is explicit about where the numbers came from: a
    :class:`~sglang.srt.memtier.profile.RigProfile` supplies the measured
    record, live :class:`~sglang.srt.memtier.profile.LocalFacts` supply
    current capacity, and both are injectable so a test builds a rig without a
    driver, a mount or a wire.
    """

    def __init__(
        self,
        tiers: Sequence[TierDescriptor],
        *,
        profile_id: str,
        local_host: str,
        caveat: str = "",
    ) -> None:
        self.profile_id = profile_id
        self.local_host = local_host
        self.caveat = caveat
        self._tiers: Tuple[TierDescriptor, ...] = tuple(tiers)
        self._by_id: Dict[TierId, TierDescriptor] = {}
        for tier in self._tiers:
            if tier.id in self._by_id:
                raise ValueError(f"duplicate tier id {tier.id!r}")
            self._by_id[tier.id] = tier

    # -- construction -----------------------------------------------------

    @classmethod
    def from_profile(
        cls,
        profile: RigProfile,
        facts: Optional[LocalFacts] = None,
    ) -> TierRegistry:
        """Declared tiers plus whatever the live facts bind and update.

        ``profile`` is **required**. It used to default to the bundled rig-1
        record, which meant every machine that built a registry without
        arguments received one particular development box's host RAM, ZFS pool
        and 40G peer -- the exact failure ``profile.py``'s docstring says the
        file format exists to prevent. :meth:`for_machine` is the entry point
        that decides which profile, if any, applies here.
        """
        facts = facts if facts is not None else LocalFacts()
        tiers: List[TierDescriptor] = list(profile.tiers)
        tiers.extend(bind_device_tiers(profile, facts.cards))
        return cls(
            apply_local_facts(tiers, facts),
            profile_id=profile.profile_id,
            local_host=profile.host,
            caveat=profile.caveat,
        )

    @classmethod
    def for_machine(
        cls,
        facts: LocalFacts,
        *,
        selection: Optional[ProfileSelection] = None,
        fs_types: Optional[Mapping[str, str]] = None,
        host: Optional[str] = None,
    ) -> Tuple[TierRegistry, ProfileSelection]:
        """The registry for THIS machine, and the record of how it was chosen.

        The whole generality contract in one call:

        1. fingerprint the machine from ``facts`` alone (no ``/proc/cpuinfo``,
           no hostname, nothing this function did not receive);
        2. match every stored profile against it; a profile that does not match
           contributes nothing, and a profile that matches only by card MODEL
           contributes device templates and no host/filesystem/remote tier;
        3. when nothing matches, bootstrap from the live facts -- measured
           sizes, absent costs, every absence naming its probe.

        The selection is returned rather than logged, because "which profile
        did this rig use and what did it pass over" is the first question
        anybody asks about a wrong number, and a caller that has to grep a log
        for it will not.
        """
        resolved_host = host or default_host_name()
        fingerprint = fingerprint_from_facts(facts, host=resolved_host)
        chosen = selection if selection is not None else select_profile(fingerprint)
        if chosen.profile is None:
            profile = bootstrap_profile(facts, host=resolved_host, fs_types=fs_types)
            return cls.from_profile(profile, LocalFacts(source=facts.source)), chosen
        return cls.from_profile(chosen.profile, facts), chosen

    # -- enumeration ------------------------------------------------------

    def tiers(self) -> Tuple[TierDescriptor, ...]:
        """Every declared tier, blocked and unreachable ones included.

        Omission is how a spill target silently becomes a different spill
        target, so nothing is filtered here -- a tier that cannot be used
        carries a ``block`` verdict and its reason instead of vanishing.
        """
        return self._tiers

    def ids(self) -> Tuple[TierId, ...]:
        return tuple(self._by_id)

    def get(self, tier_id: TierId) -> TierDescriptor:
        try:
            return self._by_id[tier_id]
        except KeyError:
            raise UnknownTier(
                f"no tier {tier_id!r} in profile {self.profile_id!r}; declared: "
                + ", ".join(self._by_id)
            ) from None

    def of_kind(self, kind: TierKind) -> Tuple[TierDescriptor, ...]:
        return tuple(t for t in self._tiers if t.kind is kind)

    # -- filtering --------------------------------------------------------

    def select(self, query: TierQuery) -> TierSelection:
        """Admissible tiers in order, and a named refusal for every other one."""
        candidates: List[TierCandidate] = []
        refusals: List[Refusal] = []
        for tier in self._tiers:
            refusal = self._refuse(tier, query)
            if refusal is not None:
                refusals.append(refusal)
                continue
            candidates.append(self._candidate(tier, query))
        candidates.sort(key=lambda c: c.order_key)
        return TierSelection(candidates=tuple(candidates), refusals=tuple(refusals))

    def _refuse(self, tier: TierDescriptor, query: TierQuery) -> Optional[Refusal]:
        """The first rule this tier fails, or ``None``.

        Order matters for the message, not for the outcome: scope filters
        first (a caller that asked for filesystems is not interested in a
        card's health), then reachability, then the laws.
        """
        scope = self._refuse_scope(tier, query)
        if scope is not None:
            return scope
        if tier.health.verdict == "block" or not tier.health.reachable:
            return Refusal(
                tier_id=tier.id,
                rule=RefusalRule.HEALTH,
                reason=tier.health.reason or "tier is not reachable",
            )
        volatility = admission_refusal(tier, query.payload, origin=query.origin)
        if volatility is not None:
            return Refusal(
                tier_id=tier.id, rule=RefusalRule.VOLATILITY, reason=volatility
            )
        if query.object_class is not None and query.object_class not in tier.admits:
            return Refusal(
                tier_id=tier.id,
                rule=RefusalRule.OBJECT_CLASS,
                reason=(
                    f"tier {tier.id} admits {sorted(tier.admits)}; it does not "
                    f"hold {query.object_class!r} content"
                ),
            )
        return self._refuse_cost(tier, query) or self._refuse_capacity(tier, query)

    def _refuse_scope(
        self, tier: TierDescriptor, query: TierQuery
    ) -> Optional[Refusal]:
        if query.kinds and tier.kind not in query.kinds:
            return Refusal(
                tier_id=tier.id,
                rule=RefusalRule.KIND,
                reason=f"kind {tier.kind.value} is not one of "
                f"{[k.value for k in query.kinds]}",
            )
        if query.hosts and tier.host not in query.hosts:
            return Refusal(
                tier_id=tier.id,
                rule=RefusalRule.HOST,
                reason=f"host {tier.host!r} is not one of {list(query.hosts)}",
            )
        for key, wanted in query.require.items():
            actual = tier.properties.get(key)
            if actual != wanted:
                return Refusal(
                    tier_id=tier.id,
                    rule=RefusalRule.PROPERTY,
                    reason=(
                        f"property {key!r} is "
                        f"{'absent' if actual is None else repr(actual)}, "
                        f"not {wanted!r}"
                    ),
                )
        return None

    def _refuse_cost(self, tier: TierDescriptor, query: TierQuery) -> Optional[Refusal]:
        bandwidth = tier.caps.bandwidth_gbs
        if bandwidth.is_absent:
            if query.min_bandwidth_gbs is not None:
                return Refusal(
                    tier_id=tier.id,
                    rule=RefusalRule.BANDWIDTH_ABSENT,
                    reason=(
                        f"a floor of {query.min_bandwidth_gbs} GB/s was asked for "
                        f"and this tier's bandwidth is absent ({bandwidth.source}). "
                        "An unmeasured path is never assumed usable."
                    ),
                )
            if not query.allow_unmeasured_bandwidth:
                return Refusal(
                    tier_id=tier.id,
                    rule=RefusalRule.BANDWIDTH_ABSENT,
                    reason=(
                        f"bandwidth is absent ({bandwidth.source}); pass "
                        "allow_unmeasured_bandwidth to consider it anyway"
                    ),
                )
            return None
        if (
            query.require_measured_bandwidth
            and bandwidth.provenance is not Provenance.MEASURED
        ):
            return Refusal(
                tier_id=tier.id,
                rule=RefusalRule.BANDWIDTH_UNMEASURED,
                reason=(
                    f"bandwidth is an {bandwidth.provenance.value} "
                    f"({bandwidth.source}) and the caller requires a measurement"
                ),
            )
        floor = query.min_bandwidth_gbs
        if floor is not None and bandwidth.require("bandwidth_gbs") < floor:
            return Refusal(
                tier_id=tier.id,
                rule=RefusalRule.BANDWIDTH_BELOW,
                reason=(
                    f"{bandwidth.value} GB/s is below the {floor} GB/s floor "
                    f"({bandwidth.provenance.value}: {bandwidth.source})"
                ),
            )
        return None

    def _refuse_capacity(
        self, tier: TierDescriptor, query: TierQuery
    ) -> Optional[Refusal]:
        if query.bytes_needed <= 0:
            return None
        headroom = tier.capacity.headroom()
        if headroom.is_absent:
            # #400's rule: the caller refuses rather than guesses. A headroom
            # nobody could compute is not a large headroom.
            return Refusal(
                tier_id=tier.id,
                rule=RefusalRule.CAPACITY,
                reason=(
                    f"{query.bytes_needed} bytes were asked for and the headroom "
                    f"cannot be computed ({headroom.source})"
                ),
            )
        available = headroom.require("headroom")
        if available < query.bytes_needed:
            return Refusal(
                tier_id=tier.id,
                rule=RefusalRule.CAPACITY,
                reason=(
                    f"{query.bytes_needed} bytes were asked for and "
                    f"{int(available)} are free "
                    f"(total {_bytes_text(tier.capacity.total)}, floor "
                    f"{_bytes_text(tier.capacity.floor)}, reserved "
                    f"{tier.capacity.reserved}, corridor {tier.capacity.corridor})"
                ),
            )
        return None

    def _candidate(self, tier: TierDescriptor, query: TierQuery) -> TierCandidate:
        bandwidth = tier.caps.bandwidth_gbs
        notes: List[str] = []
        if tier.health.verdict == "warn":
            notes.append(f"health warn: {tier.health.reason}")
        if bandwidth.provenance is not Provenance.MEASURED:
            notes.append(
                f"bandwidth is {bandwidth.provenance.value}: {bandwidth.source}"
            )
        if query.origin is not None:
            notes.append(
                f"role from {query.origin}: "
                f"{tier.role(origin=query.origin, local_host=self.local_host)}"
            )
        return TierCandidate(
            tier=tier,
            order_key=(
                _PROVENANCE_RANK[bandwidth.provenance],
                -float(bandwidth.or_none() or 0.0),
                tier.id,
            ),
            bandwidth_gbs=bandwidth,
            headroom_bytes=tier.capacity.headroom(),
            notes=tuple(notes),
        )

    # -- reporting --------------------------------------------------------

    def gate_rows(self) -> List[GateRow]:
        """One dashboard row per tier, in ``planner.rig_coupling``'s shape.

        The registry emits rows rather than defining a health model, because
        the tree already has the surface and adding a row there costs "one
        ``_row_*`` builder plus one append -- no schema change".
        """
        rows = []
        for tier in self._tiers:
            bandwidth = tier.caps.bandwidth_gbs
            remote = tier.host != self.local_host
            rows.append(
                GateRow(
                    key=f"memtier:{tier.id}",
                    label=f"{tier.role(local_host=self.local_host)} {tier.id}",
                    verdict=tier.health.verdict,
                    reason=tier.health.reason,
                    remedy=_remedy(tier),
                    local="" if remote else _bandwidth_text(bandwidth),
                    remote=_bandwidth_text(bandwidth) if remote else "",
                    provenance=bandwidth.provenance.value,
                    evidence=bandwidth.source,
                    register_key=None,
                )
            )
        return rows

    def to_json(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "local_host": self.local_host,
            "caveat": self.caveat,
            "tiers": [t.to_json(local_host=self.local_host) for t in self._tiers],
        }


def _bytes_text(rate: Rate) -> str:
    return "absent" if rate.is_absent else str(int(rate.value or 0))


def _bandwidth_text(rate: Rate) -> str:
    return "not measured" if rate.is_absent else f"{rate.value} GB/s"


def _remedy(tier: TierDescriptor) -> str:
    """What would turn this row green. Empty when it is already green."""
    if tier.health.verdict == "ok":
        return ""
    if not tier.is_bound:
        return "enumerate the card from this host (NVML) so the tier binds to a UUID"
    return "probe the tier; see the memtier probe catalogue for which arm produces it"


def gate_rows_for(tiers: Iterable[TierDescriptor], *, local_host: str) -> List[GateRow]:
    """Rows for a tier subset, for a caller that already has descriptors."""
    return TierRegistry(list(tiers), profile_id="", local_host=local_host).gate_rows()
