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
"""Tier identity and tier records (#407 cut 1, DESIGN_407 §1 and §3.1-§3.4).

This module is the NODE half of the memory hierarchy: what memories exist,
what each may hold, what each costs, and whether it is reachable right now.
The EDGE half -- what does it cost to move a byte from here to there -- is not
re-implemented here; it already exists in ``planner.cost_model`` (``PairMatrix``
/ ``Hop``), ``barlink_path_rates`` and ``rigmon.card_probe``, and cut 3 wires
the staging graph to them.

Three laws are enforced by the types rather than by convention:

**Identity is never positional.** A :data:`TierId` carries an NVML UUID (or a
named absence of one), never a CUDA ordinal and never an NVML index. This is
DESIGN_407's contradiction C2: ``offload_movement.CapacityLedger`` keys
peer-VRAM budgets by CUDA ordinal, which is legitimate in-process and a
silent card swap the moment it crosses a process boundary -- on this rig the
5090 is CUDA ordinal 0 and NVML index 1. :func:`parse_tier_id` rejects a bare
integer with that sentence.

**Every number carries its provenance.** Capacity, bandwidth, latency and
aperture are :class:`~sglang.srt.planner.cost_model.Rate` values: measured, a
formula over measured inputs, or a named absence -- and no fourth case. The
defect this inherits from is #348b's D4, where a missing bandwidth read
downstream as an extremely slow but *valid* card. A missing tier bandwidth
must not read as an extremely slow but usable tier, so an absent number is a
refusal, not a low score.

**Volatility is a refusal, not a ranking.** A tier declares what it can
guarantee; a payload declares what it needs; :func:`admission_refusal` names
the mismatch. A registry that merely ordered tiers by cost would happily offer
the GDN pool a remote tier -- and recurrent state that made a lossy round trip
is a correctness failure, not a slow one.
"""

from __future__ import annotations

import enum
import re
from typing import Any, Dict, Mapping, Optional, Tuple

import msgspec

from sglang.srt.model_executor.offload_register import OFFLOAD_CLASSES
from sglang.srt.planner.cost_model import Provenance, Rate

__all__ = [
    "ADMITTED_PAYLOADS",
    "HEALTH_VERDICTS",
    "PayloadClass",
    "TierCapacity",
    "TierCaps",
    "TierDescriptor",
    "TierHealth",
    "TierId",
    "TierIdError",
    "TierKind",
    "TierTransport",
    "Volatility",
    "admission_refusal",
    "blob_tier_id",
    "device_tier_id",
    "filesystem_tier_id",
    "host_tier_id",
    "parse_tier_id",
    "unenumerated_card_key",
]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

#: A structured, never positional tier name. See :func:`parse_tier_id` for the
#: grammar. It is a plain ``str`` so it survives JSON, a log line and a
#: dictionary key without a codec.
TierId = str


class TierIdError(ValueError):
    """A tier name that does not parse, with the reason it does not."""


class TierKind(str, enum.Enum):
    """What sort of memory a tier is. Locality is NOT part of the kind.

    A remote host's RAM is ``HOST`` on another ``host``, not a fourth kind:
    NORDSTERN (TP=5 across two rigs) needs the same record shape on both
    sides, and a kind that encoded locality would need a parallel vocabulary
    the day a rig grew a third machine. The human-readable label that does
    fold locality and transport back in is :meth:`TierDescriptor.role`.
    """

    DEVICE = "device"
    HOST = "host"
    FILESYSTEM = "filesystem"
    BLOB = "blob"


#: ``TierId`` prefix per kind. The prefix is the parse key.
_PREFIX_BY_KIND: Dict[TierKind, str] = {
    TierKind.DEVICE: "vram",
    TierKind.HOST: "host",
    TierKind.FILESYSTEM: "fs",
    TierKind.BLOB: "blob",
}
_KIND_BY_PREFIX: Dict[str, TierKind] = {v: k for k, v in _PREFIX_BY_KIND.items()}

#: An NVML card UUID as NVML itself spells it.
_CARD_UUID_RE = re.compile(r"^(?:GPU|MIG)-[0-9A-Za-z][0-9A-Za-z\-]*$")

#: A card that is DECLARED but has never been enumerated from this host --
#: rig-2's cards, for example, which no local NVML has ever seen. Declaring
#: them keeps NORDSTERN a dashboard row instead of an invisible gap (§6), and
#: keeps the day they work a data change rather than a code change.
_UNENUMERATED_PREFIX = "unenumerated@"
_UNENUMERATED_RE = re.compile(r"^unenumerated@[0-9A-Za-z][0-9A-Za-z._\-]*$")

_HOST_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._\-]*$")
_BACKEND_RE = re.compile(r"^[0-9a-z][0-9a-z_\-]*$")


def unenumerated_card_key(host: str) -> str:
    """The card key for a declared card on ``host`` that local NVML cannot see."""
    if not _HOST_RE.match(host):
        raise TierIdError(f"not a host name: {host!r}")
    return f"{_UNENUMERATED_PREFIX}{host}"


class ParsedTierId(msgspec.Struct, frozen=True, kw_only=True):
    """The fields a :data:`TierId` is made of, after parsing."""

    kind: TierKind
    #: DEVICE only: the NVML UUID, or ``unenumerated@<host>``.
    card_key: Optional[str] = None
    #: HOST / FILESYSTEM: the machine. DEVICE: only when unenumerated.
    host: Optional[str] = None
    #: FILESYSTEM only: the mount point.
    mount: Optional[str] = None
    #: BLOB only.
    backend: Optional[str] = None
    scope: Optional[str] = None

    @property
    def is_bound(self) -> bool:
        """False for a declared card no local NVML has ever resolved."""
        return not (self.card_key or "").startswith(_UNENUMERATED_PREFIX)


def device_tier_id(card_key: str) -> TierId:
    """``vram:GPU-<uuid>`` -- or ``vram:unenumerated@<host>`` for a declared card."""
    tier_id = f"vram:{card_key}"
    parse_tier_id(tier_id)
    return tier_id


def host_tier_id(host: str) -> TierId:
    """``host:<hostname>`` -- one machine's RAM."""
    tier_id = f"host:{host}"
    parse_tier_id(tier_id)
    return tier_id


def filesystem_tier_id(host: str, mount: str) -> TierId:
    """``fs:<hostname>:<mount>`` -- one filesystem on one machine."""
    tier_id = f"fs:{host}:{mount}"
    parse_tier_id(tier_id)
    return tier_id


def blob_tier_id(backend: str, scope: str) -> TierId:
    """``blob:<backend>:<scope>`` -- a HiCache-class blob store."""
    tier_id = f"blob:{backend}:{scope}"
    parse_tier_id(tier_id)
    return tier_id


def parse_tier_id(tier_id: TierId) -> ParsedTierId:
    """Parse a tier name, or raise :class:`TierIdError` naming the defect.

    Grammar (DESIGN_407 §3.1)::

        vram:GPU-<uuid>            local or peer device memory
        vram:unenumerated@<host>   a declared card local NVML cannot see
        host:<hostname>            host RAM of one machine
        fs:<hostname>:<mount>      a filesystem (NVMe, tmpfs, ...)
        blob:<backend>:<scope>     a HiCache-class blob store
    """
    prefix, _, rest = tier_id.partition(":")
    kind = _KIND_BY_PREFIX.get(prefix)
    if kind is None or not rest:
        raise TierIdError(
            f"{tier_id!r} is not a tier id; expected one of "
            + ", ".join(f"{p}:..." for p in _PREFIX_BY_KIND.values())
        )
    if kind is TierKind.DEVICE:
        return _parse_device_tier_id(tier_id, rest)
    if kind is TierKind.HOST:
        if not _HOST_RE.match(rest):
            raise TierIdError(f"{tier_id!r}: {rest!r} is not a host name")
        return ParsedTierId(kind=kind, host=rest)
    if kind is TierKind.FILESYSTEM:
        host, _, mount = rest.partition(":")
        if not _HOST_RE.match(host) or not mount.startswith("/"):
            raise TierIdError(
                f"{tier_id!r}: expected fs:<hostname>:<absolute mount path>"
            )
        return ParsedTierId(kind=kind, host=host, mount=mount)
    backend, _, scope = rest.partition(":")
    if not _BACKEND_RE.match(backend) or not scope:
        raise TierIdError(f"{tier_id!r}: expected blob:<backend>:<scope>")
    return ParsedTierId(kind=kind, backend=backend, scope=scope)


def _parse_device_tier_id(tier_id: TierId, card_key: str) -> ParsedTierId:
    if _UNENUMERATED_RE.match(card_key):
        host = card_key[len(_UNENUMERATED_PREFIX) :]
        return ParsedTierId(kind=TierKind.DEVICE, card_key=card_key, host=host)
    if _CARD_UUID_RE.match(card_key):
        return ParsedTierId(kind=TierKind.DEVICE, card_key=card_key)
    if card_key.lstrip("-").isdigit():
        # DESIGN_407 contradiction C2, refused at the door: a device index is
        # positional. On this rig the 5090 is CUDA ordinal 0 and NVML index 1,
        # so an index that crosses a process boundary silently names a
        # different card (#331 / #392).
        raise TierIdError(
            f"{tier_id!r}: {card_key!r} is a device INDEX, not a card identity. "
            "A CUDA ordinal and an NVML index disagree on this rig (5090 = CUDA "
            "0 / NVML 1); resolve it through registry.nvml.IdentityMap and name "
            "the card by UUID."
        )
    raise TierIdError(
        f"{tier_id!r}: {card_key!r} is not an NVML card UUID (GPU-... / MIG-...) "
        f"and not {_UNENUMERATED_PREFIX}<host>"
    )


# ---------------------------------------------------------------------------
# Volatility: what a payload needs, what a tier guarantees
# ---------------------------------------------------------------------------


class PayloadClass(str, enum.Enum):
    """What the CONTENT needs from wherever it comes to rest.

    The taxonomy is #274's, promoted from prose to a checked property. The
    members are fixed by DESIGN_407 §1.2, which also names the site that
    assigns each existing offload class to one of them.
    """

    #: Droppable and regenerable at a known, bounded cost. Needs no evacuation
    #: path at all -- ``kv_shadow``'s park/discard is free because the old
    #: layout stays the source of truth.
    RECONSTRUCTABLE = "reconstructable"
    #: Reconstructable only by redoing user-visible work (a re-prefill). Must
    #: be evacuated, never dropped -- #224's whole point.
    EXPENSIVE_RECONSTRUCTABLE = "expensive_reconstructable"
    #: Correctness depends on surviving process exit, and for #89 a reboot.
    PERSISTENCE_REQUIRED = "persistence_required"
    #: May not leave the owning card's VRAM. Not a preference: recurrent error
    #: accumulates, so a lossy or reordered round trip of GDN/Mamba state is a
    #: correctness failure (DESIGN_407 X2).
    DEVICE_BOUND = "device_bound"


class Volatility(str, enum.Enum):
    """What the TIER can guarantee. DESIGN_407 §3.1's field, verbatim."""

    #: Survives nothing but the moment; only droppable content may rest here.
    RECONSTRUCTABLE_OK = "reconstructable_ok"
    #: Holds content that must not be dropped, but dies with the process.
    EXPENSIVE_OK = "expensive_ok"
    #: Survives process exit (and, when declared so, a reboot).
    PERSISTENT = "persistent"
    #: The owning card's own VRAM: the one place device-bound content may live.
    DEVICE_BOUND_ONLY = "device_bound_only"


#: Which payload classes each tier volatility admits. The table is the whole
#: law -- there is no per-consumer override -- and it is deliberately NOT a
#: total order: ``DEVICE_BOUND`` is admitted only by ``DEVICE_BOUND_ONLY`` and
#: ``PERSISTENCE_REQUIRED`` only by ``PERSISTENT``, so neither can be reached
#: by "close enough".
ADMITTED_PAYLOADS: Mapping[Volatility, frozenset] = {
    Volatility.RECONSTRUCTABLE_OK: frozenset({PayloadClass.RECONSTRUCTABLE}),
    Volatility.EXPENSIVE_OK: frozenset(
        {PayloadClass.RECONSTRUCTABLE, PayloadClass.EXPENSIVE_RECONSTRUCTABLE}
    ),
    Volatility.PERSISTENT: frozenset(
        {
            PayloadClass.RECONSTRUCTABLE,
            PayloadClass.EXPENSIVE_RECONSTRUCTABLE,
            PayloadClass.PERSISTENCE_REQUIRED,
        }
    ),
    Volatility.DEVICE_BOUND_ONLY: frozenset(
        {
            PayloadClass.RECONSTRUCTABLE,
            PayloadClass.EXPENSIVE_RECONSTRUCTABLE,
            PayloadClass.DEVICE_BOUND,
        }
    ),
}


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def _rate_json(rate: Rate) -> Dict[str, Any]:
    return {
        "value": rate.value,
        "provenance": rate.provenance.value,
        "source": rate.source,
        "unit": rate.unit,
        "label": rate.label,
    }


class TierCapacity(msgspec.Struct, frozen=True, kw_only=True):
    """How much of a tier there is, split the way #330 splits a card.

    ``total`` minus a **pinned floor** (weights, graphs, GDN pool, activation
    workspaces, allocator metadata -- measured once per boot as
    ``nvml_process_used_bytes - vmm_backed_bytes``) is the dialable span. A
    single "free bytes" field would erase that distinction and DESIGN_330
    would have to re-derive it.

    ``reserved`` and ``corridor`` come from ``registry.ledger`` -- the one
    implementation of the cross-process invariant on this rig -- and are never
    recomputed here.
    """

    #: Bytes. Measured, or a named absence.
    total: Rate
    #: Bytes that are not reclaimable at this tier. Absent before a boot has
    #: measured it, which makes :meth:`headroom` absent too, on purpose.
    floor: Rate
    #: Bytes declared by other holders, read from the cross-process ledger.
    reserved: int = 0
    #: #330's 400 MiB absolutely-free rule. Belongs to the card, so no tenant
    #: accounts for it. Zero on non-device kinds.
    corridor: int = 0

    def headroom(self) -> Rate:
        """What a new holder may declare here, or a named absence.

        Subtraction of measured quantities is bookkeeping, not a model, so the
        result keeps the WEAKEST provenance of its inputs rather than being
        demoted to ``ESTIMATE`` wholesale. An absent input makes the result
        absent: #400's rule is that a caller must refuse rather than guess,
        and a headroom derived from an unknown floor is a guess.
        """
        for name, rate in (("total", self.total), ("floor", self.floor)):
            if rate.is_absent:
                return Rate.absent(
                    f"capacity {name} is absent ({rate.source}); a headroom "
                    "cannot be derived from it",
                    unit="bytes",
                    label="headroom",
                )
        value = (
            float(self.total.require("total"))
            - float(self.floor.require("floor"))
            - float(self.reserved)
            - float(self.corridor)
        )
        weakest = (
            Provenance.ESTIMATE
            if Provenance.ESTIMATE in (self.total.provenance, self.floor.provenance)
            else Provenance.MEASURED
        )
        return Rate(
            value,
            weakest,
            "total - floor - reserved - corridor over "
            f"{self.total.source}; {self.floor.source}",
            "bytes",
            "headroom",
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "total": _rate_json(self.total),
            "floor": _rate_json(self.floor),
            "reserved": self.reserved,
            "corridor": self.corridor,
            "headroom": _rate_json(self.headroom()),
        }


class TierCaps(msgspec.Struct, frozen=True, kw_only=True):
    """What the tier costs, and which accounting bucket it posts to."""

    #: Microseconds for a point access. Absent on every tier of this rig today
    #: (DESIGN_407 §4 M1/M4/M5) -- and absent is the honest word: the
    #: "1-3 us posted write" peer-VRAM class is an ASSUMPTION, not a probe.
    latency_us: Rate
    #: GB/s sustained.
    bandwidth_gbs: Rate
    #: Bytes of the EFFECTIVE window, never the nominal one. This rig's
    #: nominal 256 MiB BAR1 maps 96 MiB contiguously, and #278 W1 showed BAR
    #: size moves nothing at any measured step, so a nominal number would be
    #: both wrong and misleading. A ``Rate`` rather than DESIGN_407's
    #: ``Optional[int]`` for one reason: an int cannot say WHY it is missing.
    aperture_bytes: Rate
    #: The named bucket a reservation on this tier posts to, in #400 terms.
    ledger_key: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "latency_us": _rate_json(self.latency_us),
            "bandwidth_gbs": _rate_json(self.bandwidth_gbs),
            "aperture_bytes": _rate_json(self.aperture_bytes),
            "ledger_key": self.ledger_key,
        }


#: ``GateRow``'s vocabulary (``planner.rig_coupling``), reused rather than
#: redefined so a tier row and a coupling row read the same on one dashboard.
HEALTH_VERDICTS: Tuple[str, ...] = ("ok", "warn", "block")


class TierHealth(msgspec.Struct, frozen=True, kw_only=True):
    """Whether the tier is usable right now, and why not when it is not.

    An unreachable tier is enumerated with ``verdict="block"``, never omitted:
    omission is how a spill target silently becomes a different spill target.
    """

    reachable: bool
    verdict: str = "ok"
    reason: str = ""
    last_seen_s: float = 0.0

    def __post_init__(self) -> None:
        if self.verdict not in HEALTH_VERDICTS:
            raise ValueError(
                f"verdict {self.verdict!r} is not one of {HEALTH_VERDICTS}"
            )
        if self.verdict != "ok" and not self.reason:
            raise ValueError(
                f"a {self.verdict!r} verdict must carry a reason; an unexplained "
                "degradation is indistinguishable from a bug"
            )

    def to_json(self) -> Dict[str, Any]:
        return {
            "reachable": self.reachable,
            "verdict": self.verdict,
            "reason": self.reason,
            "last_seen_s": self.last_seen_s,
        }


class TierTransport(msgspec.Struct, frozen=True, kw_only=True):
    """How bytes reach the tier, and through what.

    ``stages_through`` is the reachability fact DESIGN_407 §3.5 turns into a
    staging graph in cut 3. It is recorded from cut 1 because #224's
    "``local`` must be first" law is exactly this edge -- the device D2H copy
    lands in locally pinned host RAM, so every blob tier stages through the
    local host tier -- and recording it now is what lets that law stop being
    a hardcoded total order later.
    """

    #: Short transport name, e.g. ``pcie``, ``barlink-bar1``, ``posix``, ``roce-40g``.
    name: str
    #: The module or route that actually moves the bytes. Free text on
    #: purpose: it names an owner for a question, it is not dispatched on.
    handle: str = ""
    #: The tier a mover must pass through to get here, or ``None`` for a
    #: direct edge.
    stages_through: Optional[TierId] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "handle": self.handle,
            "stages_through": self.stages_through,
        }


class TierDescriptor(msgspec.Struct, frozen=True, kw_only=True):
    """One memory, everything the registry knows about it.

    Immutable: an update is a replacement (:meth:`evolve`), so a descriptor
    handed to a consumer cannot change underneath it, and a provenance can
    never be edited in place.
    """

    id: TierId
    kind: TierKind
    #: The machine the tier is attached to. Carried from cut 1 so NORDSTERN
    #: needs no schema change (§6).
    host: str
    capacity: TierCapacity
    volatility: Volatility
    #: ``OFFLOAD_CLASSES`` this tier may hold. A class absent from the set is
    #: refused BY NAME rather than being unranked.
    admits: frozenset = frozenset()
    caps: TierCaps
    health: TierHealth
    transport: TierTransport
    #: Name-derived capability checks land here rather than in an ``if name ==``:
    #: ``flock``, ``pointer_io``, ``rewritable``, ``medium``,
    #: ``producer_fingerprint``, ``rig_is_lower_bound``, ...
    properties: Mapping[str, str] = {}
    #: Which rig profile these numbers were measured on. A number without this
    #: is a number somebody will read as universal.
    profile_id: str = ""
    #: Card model for a DEVICE tier, empty otherwise.
    card_model: str = ""

    def __post_init__(self) -> None:
        parsed = parse_tier_id(self.id)
        if parsed.kind is not self.kind:
            raise ValueError(
                f"tier {self.id!r} parses as {parsed.kind.value}, but the record "
                f"declares {self.kind.value}"
            )
        unknown = set(self.admits) - set(OFFLOAD_CLASSES)
        if unknown:
            raise ValueError(
                f"tier {self.id!r} admits unknown offload classes "
                f"{sorted(unknown)}; the vocabulary is offload_register."
                f"OFFLOAD_CLASSES = {list(OFFLOAD_CLASSES)}"
            )

    @property
    def parsed(self) -> ParsedTierId:
        return parse_tier_id(self.id)

    @property
    def is_bound(self) -> bool:
        """False for a DEVICE tier no local NVML has ever resolved."""
        return self.parsed.is_bound

    def admitted_payloads(self) -> frozenset:
        return ADMITTED_PAYLOADS[self.volatility]

    def evolve(self, **changes: Any) -> TierDescriptor:
        """A copy with fields replaced. The only way a record changes."""
        return msgspec.structs.replace(self, **changes)

    def role(
        self,
        *,
        origin: Optional[TierId] = None,
        local_host: Optional[str] = None,
    ) -> str:
        """The human label that folds locality and transport back in.

        ``vram-local``, ``vram-peer-bar1``, ``ram-local``, ``nvme-local``,
        ``ram-remote``, ``vram-remote``, ``disk-remote`` and friends are a
        VIEW, derived per query -- "peer" is a relation between an origin and
        a tier, not a property of the tier, and the same card's VRAM is local
        to the rank that owns it and peer to the rank beside it.
        """
        remote = local_host is not None and self.host != local_host
        where = "remote" if remote else "local"
        if self.kind is TierKind.DEVICE:
            if origin is not None and origin == self.id:
                return "vram-local"
            if remote:
                return "vram-remote"
            if origin is not None:
                suffix = self.transport.name or "peer"
                return f"vram-peer-{suffix.split('-')[-1]}"
            return "vram-local"
        if self.kind is TierKind.HOST:
            return f"ram-{where}"
        if self.kind is TierKind.FILESYSTEM:
            return f"{self.properties.get('medium', 'disk')}-{where}"
        return f"blob-{where}"

    def to_json(self, *, local_host: Optional[str] = None) -> Dict[str, Any]:
        """The dashboard/API shape. ``local_host`` is what makes ``role``
        read ``ram-remote`` rather than ``ram-local``; the registry supplies
        it because a descriptor alone cannot know which machine is reading."""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "host": self.host,
            "role": self.role(local_host=local_host),
            "card_model": self.card_model,
            "profile_id": self.profile_id,
            "volatility": self.volatility.value,
            "admits": sorted(self.admits),
            "admitted_payloads": sorted(p.value for p in self.admitted_payloads()),
            "capacity": self.capacity.to_json(),
            "caps": self.caps.to_json(),
            "health": self.health.to_json(),
            "transport": self.transport.to_json(),
            "properties": dict(self.properties),
            "bound": self.is_bound,
        }


def admission_refusal(
    tier: TierDescriptor,
    payload: PayloadClass,
    *,
    origin: Optional[TierId] = None,
) -> Optional[str]:
    """``None`` when ``tier`` may hold ``payload``, else the reason it may not.

    Two rules beyond the :data:`ADMITTED_PAYLOADS` table, both of them
    relations rather than properties:

    * ``DEVICE_BOUND`` content may rest only on the card that owns it, so a
      peer card's VRAM refuses it even though both tiers are
      ``DEVICE_BOUND_ONLY``.
    * a tier no local NVML has ever resolved admits nothing, whatever its
      declared volatility says.
    """
    if payload not in tier.admitted_payloads():
        return (
            f"tier {tier.id} is {tier.volatility.value} and admits "
            f"{sorted(p.value for p in tier.admitted_payloads())}; "
            f"{payload.value} content may not rest there"
        )
    if (
        payload is PayloadClass.DEVICE_BOUND
        and origin is not None
        and origin != tier.id
    ):
        return (
            f"tier {tier.id} is not the origin card {origin}; device-bound "
            "content (GDN/Mamba recurrent state) never travels, not even one "
            "hop over P2P"
        )
    if not tier.is_bound:
        return (
            f"tier {tier.id} is declared but has never been enumerated from "
            "this host; an unresolved card admits nothing"
        )
    return None
