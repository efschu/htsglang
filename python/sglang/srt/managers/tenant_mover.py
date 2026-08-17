# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#553 cut 2 (second half): park and restore a named TENANT.

THE GAP THIS FILLS, in the analysis's own words (§2, tenant COLD row): parking
"works for the translator (``ledger.park_all``) and for GDN slots (#364). **No
generic per-tenant mover; no single caller that can address 'tenant X'**." The
first half of this cut wired the two byte probes so the bridge stops answering
all-unavailable; this is the actuator a cold event calls once that bridge has
ranked a plan.

NOT A SECOND ASSET LEDGER. ``AudioAssetLedger`` already parks and restores, and
its ``ParkRoute`` protocol -- ``device`` / ``park`` / ``restore`` /
``size_bytes`` -- is already generic. The plan says "what makes it
translator-specific is only its wake-rank vocabulary". So this reimplements no
parking: it registers TENANTS over that same protocol and takes the ordering as
CONFIGURATION. Building a parallel ledger would be exactly the two-authorities
defect the #553 bridge exists to reconcile.

REPORTS, NOT PLANS (#694). Released bytes come from what a route RETURNED. A
route that was asked and reported nothing is carried as STRANDED -- never
counted as zero, never dropped. That is the same distinction ``cold_event``
makes for sources and it is made here for the same reason: bytes that left one
ledger and entered none are the shape that goes unnoticed for weeks. A route
that reports 0 is an accounting ("nothing to give"); silence is the absence of
one, and the two are distinguishable in the result.

REFUSALS. An unknown tenant raises rather than returning zero: "no such tenant"
and "that tenant had nothing to give" are different answers, and a caller that
cannot tell them apart keeps asking the wrong one.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Rank given to a route the tenant's vocabulary does not mention. Sorts after
#: every named rank, so an unlisted route restores last rather than first --
#: an unranked asset is unknown, not urgent.
UNRANKED = 1 << 30


class UnknownTenant(KeyError):
    """A tenant id that was never registered."""

    def __init__(self, tenant: str, known: Sequence[str]):
        self.tenant = tenant
        self.known = tuple(known)
        super().__init__(
            f"no tenant {tenant!r} is registered with this mover. Registered: "
            f"{sorted(self.known)!r}. Refused rather than returning zero: "
            "'no such tenant' and 'that tenant had nothing to give' are "
            "different answers."
        )


@dataclasses.dataclass(frozen=True)
class StrandedRoute:
    """A route that was asked to park and reported nothing back."""

    name: str
    device: Optional[str]


@dataclasses.dataclass(frozen=True)
class ParkResult:
    tenant: str
    released_bytes: int
    stranded: Tuple[StrandedRoute, ...] = ()

    @property
    def ok(self) -> bool:
        """False when anything was stranded.

        A park with a stranded route is not a smaller success; it is a park
        whose accounting has a hole in it.
        """
        return not self.stranded


@dataclasses.dataclass
class _Tenant:
    routes: List[Any]
    ranks: Dict[str, int]
    parked: Dict[str, Tuple[Optional[str], int]] = dataclasses.field(
        default_factory=dict
    )


def _route_name(route: Any, index: int) -> str:
    return str(getattr(route, "name", f"route{index}"))


def _route_device(route: Any) -> Optional[str]:
    device = getattr(route, "device", None)
    if callable(device):
        try:
            return device()
        except Exception:  # pragma: no cover - a route that cannot say
            return None
    return device


class TenantMover:
    """Registry of movable tenants, each a set of ``ParkRoute``-shaped units."""

    def __init__(self) -> None:
        self._tenants: Dict[str, _Tenant] = {}

    def register(
        self,
        tenant: str,
        routes: Sequence[Any],
        ranks: Optional[Mapping[str, int]] = None,
    ) -> None:
        """Register a tenant's routes and its OWN restore vocabulary.

        ``ranks`` maps a route name to its position in that tenant's need
        order. It is per tenant on purpose: the translator's ASR -> talker ->
        codec order is one tenant's physics, not a property of moving tenants,
        and hard-coding it here would make a second tenant wear it.
        """
        self._tenants[str(tenant)] = _Tenant(
            routes=list(routes), ranks={str(k): int(v) for k, v in (ranks or {}).items()}
        )

    def tenants(self) -> Tuple[str, ...]:
        return tuple(self._tenants)

    def _get(self, tenant: str) -> _Tenant:
        found = self._tenants.get(str(tenant))
        if found is None:
            raise UnknownTenant(str(tenant), self.tenants())
        return found

    def park_tenant(self, tenant: str) -> ParkResult:
        """Park every route of one tenant, reporting only what came back."""
        entry = self._get(tenant)
        released = 0
        stranded: List[StrandedRoute] = []
        for index, route in enumerate(entry.routes):
            name = _route_name(route, index)
            device = _route_device(route)
            try:
                got = route.park()
            except Exception as exc:
                logger.warning(
                    "#553 tenant %s: route %s raised while parking (%s); "
                    "carried as stranded",
                    tenant,
                    name,
                    exc,
                )
                stranded.append(StrandedRoute(name, device))
                continue
            if got is None:
                stranded.append(StrandedRoute(name, device))
                continue
            got = int(got)
            released += got
            entry.parked[name] = (device, got)
        return ParkResult(str(tenant), released, tuple(stranded))

    def restore_tenant(self, tenant: str) -> Tuple[str, ...]:
        """Restore a tenant's routes in ITS need order. Returns that order."""
        entry = self._get(tenant)
        ordered = sorted(
            enumerate(entry.routes),
            key=lambda pair: (
                entry.ranks.get(_route_name(pair[1], pair[0]), UNRANKED),
                pair[0],
            ),
        )
        done: List[str] = []
        for index, route in ordered:
            name = _route_name(route, index)
            route.restore()
            entry.parked.pop(name, None)
            done.append(name)
        return tuple(done)

    def parked_bytes_by_device(self) -> Dict[str, int]:
        """Parked bytes per device, from route REPORTS.

        Absent devices are absent rather than zero: a device with nothing
        parked and a device that never reported are not the same statement
        (#606), and an all-zeros map would read as the former.
        """
        totals: Dict[str, int] = {}
        for entry in self._tenants.values():
            for device, got in entry.parked.values():
                if device is None:
                    continue
                totals[device] = totals.get(device, 0) + got
        return totals

    def release_fn(self) -> Callable[[Any], Tuple[bool, Optional[int], str]]:
        """A ``cold_event``-shaped release for this mover.

        Returns ``(ok, delivered_bytes, detail)`` with ``delivered_bytes=None``
        when anything stranded -- which is precisely how ``cold_event`` reads
        "did not report", so a stranded tenant surfaces there rather than being
        counted as a delivered zero.
        """

        def _release(source: Any) -> Tuple[bool, Optional[int], str]:
            tenant = getattr(source, "name", source)
            result = self.park_tenant(str(tenant))
            if result.stranded:
                names = [s.name for s in result.stranded]
                return (
                    False,
                    None,
                    f"tenant {tenant!r} parked {result.released_bytes} bytes but "
                    f"{len(names)} route(s) reported nothing: {names!r}",
                )
            return (
                True,
                result.released_bytes,
                f"tenant {tenant!r} released {result.released_bytes} bytes",
            )

        return _release
