# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The lane contract (#333 §3.5) and the adapter table.

Deliberately narrow: the arbiter must not need to understand a request. What
it needs is bytes per card (``estimate``), state changes (``promote`` /
``demote``), and what the tenant actually used (``measured``). Everything
class-specific lives behind those three.

M1 implements the residency and estimation half of §3.5 in full. The
execution half -- ``can_admit`` / ``submit`` / ``tick_request`` / ``grant`` --
is declared here and left to M3, because the tick broker (§3.7) has no second
class to arbitrate against until the diffusion lane launches. Declaring it now
and not implementing it is the honest shape: the protocol is what the adapters
will be judged against, and a stub that pretended to schedule would be worse
than an absent one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from sglang.srt.registry.spec import EngineSpec, ResidencyState, ResourceProfile


class AdapterError(RuntimeError):
    """An adapter could not do what the arbiter asked."""


class EstimateError(AdapterError):
    """A spec cannot be costed. Registration fails; nothing boots."""


@dataclass(frozen=True)
class Health:
    ok: bool
    detail: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


@dataclass(frozen=True)
class Admission:
    """accept | defer(t) | reject(reason) -- §3.5."""

    verdict: str  # "accept" | "defer" | "reject"
    wait_ms: float = 0.0
    reason: str = ""

    @classmethod
    def accept(cls) -> "Admission":
        return cls("accept")

    @classmethod
    def defer(cls, wait_ms: float, reason: str = "") -> "Admission":
        return cls("defer", wait_ms=wait_ms, reason=reason)

    @classmethod
    def reject(cls, reason: str) -> "Admission":
        return cls("reject", reason=reason)

    @property
    def accepted(self) -> bool:
        return self.verdict == "accept"

    def to_json(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "wait_ms": self.wait_ms, "reason": self.reason}


@dataclass
class AdapterContext:
    """What the arbiter hands an adapter so the adapter needs no globals.

    ``card_totals`` is NVML truth, resolved once by the arbiter, so an adapter
    never re-enumerates devices and never has to decide whether an index is a
    CUDA index or an NVML index.
    """

    card_totals: Mapping[str, int] = field(default_factory=dict)
    #: Per-card exclusive window. Profiling and engine builds take it; two
    #: co-located tenants must not measure free memory at the same time.
    card_exclusive_lock: Callable[..., Any] | None = None
    work_dir: str | None = None


@runtime_checkable
class LaneAdapter(Protocol):
    """§3.5. Implemented by every class; understood by the arbiter as a whole."""

    klass: int

    # -- planning, before anything boots ------------------------------------

    def estimate(self, spec: EngineSpec, cards: tuple[str, ...]) -> ResourceProfile:
        """Bytes per card per post for this spec on these cards.

        Must not touch a GPU and must not require the engine to have booted.
        This is what makes "does this fit on this rig" a millisecond question
        rather than a GPU window (§7.4).
        """

    # -- residency -----------------------------------------------------------

    def promote(self, target: ResidencyState) -> None: ...

    def demote(self, target: ResidencyState) -> None: ...

    def state(self) -> ResidencyState: ...

    # -- observability --------------------------------------------------------

    def measured(self) -> Mapping[str, int]:
        """Per-card device bytes actually held. Feeds the ledger."""

    def health(self) -> Health: ...

    def pids(self) -> tuple[int, ...]: ...


#: adapter name -> factory(spec, context) -> LaneAdapter
_ADAPTERS: dict[str, Callable[[EngineSpec, AdapterContext], LaneAdapter]] = {}


def register_adapter(
    name: str, factory: Callable[[EngineSpec, AdapterContext], LaneAdapter]
) -> None:
    if name in _ADAPTERS and _ADAPTERS[name] is not factory:
        raise AdapterError(f"adapter {name!r} is already registered")
    _ADAPTERS[name] = factory


def build_adapter(spec: EngineSpec, context: AdapterContext) -> LaneAdapter:
    load_builtin_adapters()
    try:
        factory = _ADAPTERS[spec.adapter]
    except KeyError:
        raise AdapterError(
            f"engine {spec.engine_id!r} names adapter {spec.adapter!r}, which is "
            f"not registered; known adapters: {sorted(_ADAPTERS)}"
        ) from None
    return factory(spec, context)


def known_adapters() -> tuple[str, ...]:
    load_builtin_adapters()
    return tuple(sorted(_ADAPTERS))


_BUILTINS_LOADED = False


def load_builtin_adapters() -> None:
    """Import the three shipped adapters once, lazily.

    Lazily because the Class-1 adapter reaches into ``srt`` server arguments
    and the registry has to stay importable from a planner process that has no
    intention of booting anything.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from sglang.srt.registry.adapters import (  # noqa: F401,PLC0415
        class1_srt,
        class2_diffusion,
        class3_utility,
    )
