# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Registry entities: ``EngineSpec``, ``EngineInstance``, ``Slot`` (#333 §7.1).

The whole point of these three types is that none of them mentions a model
architecture, a KV cache, a latent, or a TensorRT engine. A spec says which
class it is, which adapter runs it, where it may live and how big its shape
ceiling is; the adapter turns that into bytes. Adding a fourth class later is
writing a fourth adapter, not touching this file (§7.6).

The residency states are the ledger's :class:`~sglang.srt.registry.ledger.
TenantState`, imported rather than redeclared: a registry that disagreed with
its own ledger about what ``WARM_GPU`` means would be the first bug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from sglang.srt.registry.ledger import (
    GPU_RESIDENT_STATES,
    MIB,
    CardDemand,
    TenantState,
)

#: The residency ladder of §3.4. Same enum as the ledger's tenant state.
ResidencyState = TenantState

#: Placement value meaning "the registry picks one card, most free first".
#: Anything richer is the planner's job (§7.7) and is expressed as an explicit
#: UUID list on the spec.
AUTO_PLACEMENT = "auto"

_ENGINE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class SpecError(ValueError):
    """A spec that cannot be accepted, detected without booting anything."""


class EngineClass(int, Enum):
    AUTOREGRESSIVE = 1
    DIFFUSION = 2
    UTILITY = 3


@dataclass(frozen=True)
class ResourceProfile:
    """What one adapter says a spec costs, per card, per post (§3.5).

    ``posts`` is the per-card breakdown the memory-post sections (§4.2, §5.2,
    §6.2) define; the registry never interprets a post name, it only sums and
    reports them. ``steady_bytes`` may be lower than the reservation: the
    reservation is the *peak*, because a card cannot be handed a peak it does
    not have when the peak arrives.
    """

    posts: Mapping[str, Mapping[str, int]]
    peak_bytes: Mapping[str, int]
    steady_bytes: Mapping[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        missing = set(self.posts) - set(self.peak_bytes)
        if missing:
            raise SpecError(
                f"resource profile declares posts for cards {sorted(missing)} "
                "but no peak for them"
            )

    @property
    def cards(self) -> tuple[str, ...]:
        return tuple(sorted(self.peak_bytes))

    @property
    def total_peak_bytes(self) -> int:
        return sum(self.peak_bytes.values())

    def demands(self) -> list[CardDemand]:
        """The reservation this profile implies: peak, per card.

        Peak rather than steady is the #287 rule (§3.8) generalised. A rung, a
        resolution or an in-flight depth may move freely inside a reservation;
        the reservation itself is the maximum over everything the tenant is
        allowed to do. The difference is real waste and #330 makes it a
        registered item -- correctly, because the operator should see it.
        """
        return [
            CardDemand(
                card_uuid=card,
                reserved_bytes=int(self.peak_bytes[card]),
                posts=dict(self.posts.get(card, {})),
            )
            for card in self.cards
        ]

    def slack_bytes(self) -> dict[str, int]:
        """Per card: reserved peak minus declared steady. A #330 register item."""
        return {
            card: int(self.peak_bytes[card]) - int(self.steady_bytes.get(card, 0))
            for card in self.cards
            if card in self.steady_bytes
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "posts": {k: dict(v) for k, v in self.posts.items()},
            "peak_bytes": dict(self.peak_bytes),
            "steady_bytes": dict(self.steady_bytes),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EngineSpec:
    """A registered engine, boot-independent (§7.1)."""

    engine_id: str
    klass: EngineClass
    adapter: str
    launch: Mapping[str, Any] = field(default_factory=dict)
    placement: Sequence[str] = (AUTO_PLACEMENT,)
    declared_max: Mapping[str, Any] = field(default_factory=dict)
    pinned: bool = False
    priority: int = 0

    def __post_init__(self) -> None:
        if not _ENGINE_ID_RE.match(self.engine_id or ""):
            raise SpecError(
                f"engine_id {self.engine_id!r} must be 1-64 characters of "
                "[A-Za-z0-9._-] and start alphanumeric; it names a file and a "
                "ledger tenant, so it may not contain a path separator"
            )
        object.__setattr__(self, "klass", EngineClass(int(self.klass)))
        if not self.adapter:
            raise SpecError(f"engine {self.engine_id!r} names no adapter")
        placement = tuple(self.placement or ())
        if not placement:
            raise SpecError(
                f"engine {self.engine_id!r} has an empty placement; use "
                f"{AUTO_PLACEMENT!r} or list card UUIDs"
            )
        if AUTO_PLACEMENT in placement and len(placement) != 1:
            raise SpecError(
                f"engine {self.engine_id!r} mixes {AUTO_PLACEMENT!r} with explicit "
                "card UUIDs; placement is either automatic or fully explicit"
            )
        if len(set(placement)) != len(placement):
            raise SpecError(
                f"engine {self.engine_id!r} lists the same card twice in "
                "placement; a tenant holds one reservation per card"
            )
        object.__setattr__(self, "placement", placement)

    @property
    def is_auto_placed(self) -> bool:
        return tuple(self.placement) == (AUTO_PLACEMENT,)

    @property
    def tenant_id(self) -> str:
        """Ledger key. Same string as the engine id: one engine, one tenant."""
        return self.engine_id

    def to_json(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "klass": int(self.klass),
            "adapter": self.adapter,
            "launch": dict(self.launch),
            "placement": list(self.placement),
            "declared_max": dict(self.declared_max),
            "pinned": self.pinned,
            "priority": self.priority,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "EngineSpec":
        unknown = set(data) - {
            "engine_id",
            "klass",
            "adapter",
            "launch",
            "placement",
            "declared_max",
            "pinned",
            "priority",
        }
        if unknown:
            # Silently dropping a field the operator wrote is how a spec that
            # "was accepted" turns out not to mean what they typed.
            raise SpecError(f"unknown spec field(s): {sorted(unknown)}")
        try:
            return cls(
                engine_id=str(data["engine_id"]),
                klass=EngineClass(int(data["klass"])),
                adapter=str(data["adapter"]),
                launch=dict(data.get("launch") or {}),
                placement=tuple(data.get("placement") or (AUTO_PLACEMENT,)),
                declared_max=dict(data.get("declared_max") or {}),
                pinned=bool(data.get("pinned", False)),
                priority=int(data.get("priority", 0)),
            )
        except KeyError as exc:
            raise SpecError(f"spec is missing required field {exc.args[0]!r}") from None
        except ValueError as exc:
            if isinstance(exc, SpecError):
                raise
            raise SpecError(str(exc)) from None


@dataclass(frozen=True)
class Slot:
    """A reservation of bytes on one card, held by one instance in one state.

    The registry's view of a ledger entry. The ledger owns the bytes; this owns
    the fact that a particular engine is the reason they are held.
    """

    engine_id: str
    card_uuid: str
    reserved_bytes: int
    state: ResidencyState
    posts: Mapping[str, int] = field(default_factory=dict)
    measured_bytes: int = 0

    @property
    def is_gpu_resident(self) -> bool:
        return self.state in GPU_RESIDENT_STATES

    def to_json(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "card_uuid": self.card_uuid,
            "reserved_bytes": self.reserved_bytes,
            "reserved_mib": self.reserved_bytes // MIB,
            "state": self.state.value,
            "posts": dict(self.posts),
            "measured_bytes": self.measured_bytes,
        }


@dataclass
class EngineInstance:
    """The mutable half: what this engine is doing right now (§7.1)."""

    spec: EngineSpec
    state: ResidencyState = ResidencyState.COLD
    cards: tuple[str, ...] = ()
    profile: ResourceProfile | None = None
    pids: tuple[int, ...] = ()
    last_used_ts: float = 0.0
    health: str = "unknown"
    #: Exponential moving averages, in milliseconds, measured not guessed. None
    #: until the corresponding transition has been observed once.
    promotion_cost_ms: float | None = None
    demotion_cost_ms: float | None = None
    transitions: int = 0
    last_error: str | None = None

    _EMA_ALPHA = 0.5

    @property
    def engine_id(self) -> str:
        return self.spec.engine_id

    @property
    def is_gpu_resident(self) -> bool:
        return self.state in GPU_RESIDENT_STATES

    def observe_promotion(self, elapsed_ms: float) -> None:
        self.promotion_cost_ms = self._blend(self.promotion_cost_ms, elapsed_ms)

    def observe_demotion(self, elapsed_ms: float) -> None:
        self.demotion_cost_ms = self._blend(self.demotion_cost_ms, elapsed_ms)

    @classmethod
    def _blend(cls, previous: float | None, sample: float) -> float:
        if previous is None:
            return float(sample)
        return cls._EMA_ALPHA * float(sample) + (1.0 - cls._EMA_ALPHA) * previous

    def reserved_bytes(self) -> int:
        return self.profile.total_peak_bytes if self.profile else 0

    def to_json(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "klass": int(self.spec.klass),
            "state": self.state.value,
            "cards": list(self.cards),
            "pids": list(self.pids),
            "pinned": self.spec.pinned,
            "priority": self.spec.priority,
            "last_used_ts": self.last_used_ts,
            "health": self.health,
            "promotion_cost_ms": self.promotion_cost_ms,
            "demotion_cost_ms": self.demotion_cost_ms,
            "transitions": self.transitions,
            "reserved_bytes": self.reserved_bytes(),
            "last_error": self.last_error,
            "profile": self.profile.to_json() if self.profile else None,
        }
