# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The engine registry / arbiter (#333 §7).

N engines are registered; M are hot. **M is not configured.** The binding
constraint is the ledger invariant of §3.3 and M is whatever fits, which is
the only formulation that survives unlike cards: a count-based cap either
wastes capacity or fails at runtime on a rig where "one engine" means 2 GiB on
one card and 18 GiB on another. ``max_hot`` exists here as a blunt safety
valve for operators who want to bound process count, and it is applied *after*
the byte derivation, never instead of it.

Everything in this file is policy over the ledger's mechanism: which tenant
may hold which bytes, who is evicted, what the caller is told when the answer
is no. It contains no reference to a model architecture, a KV cache, a latent
or a TensorRT engine -- those live behind ``adapter.estimate()`` (§7.6).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from sglang.srt.registry.adapter import (
    AdapterContext,
    AdapterError,
    build_adapter,
)
from sglang.srt.registry.ledger import (
    MIB,
    CardDemand,
    FeasibilityReport,
    LedgerError,
    MultiCardReservation,
    ReservationStore,
    plan_reservation,
)
from sglang.srt.registry.spec import (
    EngineInstance,
    EngineSpec,
    ResidencyState,
    ResourceProfile,
    Slot,
    SpecError,
)

logger = logging.getLogger(__name__)

#: Used only until a real transition has been observed, so that the first
#: rejection can still quote a projected wait instead of "unknown". Class 1 is
#: the slow one: a cold promotion is a model load.
DEFAULT_PROMOTION_COST_MS = {1: 90_000.0, 2: 30_000.0, 3: 20_000.0}
DEFAULT_DEMOTION_COST_MS = {1: 15_000.0, 2: 5_000.0, 3: 3_000.0}

#: A promote/demote pair that flips more often than this inside the window is
#: a named thrash event. Named, not suppressed: the registry does not silently
#: degrade, it says which two engines are fighting.
THRASH_WINDOW_S = 300.0
THRASH_FLIPS = 4


class RegistryError(RuntimeError):
    """The registry refused. Always with the numbers that produced the refusal."""


class UnknownEngineError(RegistryError):
    pass


class RegistrationRejected(RegistryError):
    def __init__(self, message: str, *, report: FeasibilityReport | None = None):
        super().__init__(message)
        self.report = report


class PromotionRejected(RegistryError):
    """§7.5: the caller is told the projected wait and the eviction."""

    def __init__(
        self,
        message: str,
        *,
        engine_id: str,
        projected_wait_ms: float,
        would_evict: Sequence[str],
        report: FeasibilityReport | None = None,
        cost_is_estimated: bool = True,
    ) -> None:
        super().__init__(message)
        self.engine_id = engine_id
        self.projected_wait_ms = projected_wait_ms
        self.would_evict = tuple(would_evict)
        self.report = report
        self.cost_is_estimated = cost_is_estimated

    def to_json(self) -> dict[str, Any]:
        return {
            "error": "promotion_rejected",
            "engine_id": self.engine_id,
            "message": str(self),
            "projected_wait_ms": self.projected_wait_ms,
            "projected_wait_is_estimated": self.cost_is_estimated,
            "would_evict": list(self.would_evict),
            "shortfalls": (
                [
                    {
                        "card_uuid": s.card_uuid,
                        "requested_bytes": s.requested_bytes,
                        "held_bytes": s.held_bytes,
                        "corridor_bytes": s.corridor_bytes,
                        "total_bytes": s.total_bytes,
                        "shortfall_bytes": s.shortfall_bytes,
                        "holders": list(s.holders),
                    }
                    for s in self.report.shortfalls
                ]
                if self.report
                else []
            ),
        }


@dataclass(frozen=True)
class CardView:
    """What ``GET /registry/cards`` answers for one physical card."""

    card_uuid: str
    total_bytes: int
    corridor_bytes: int
    reserved_bytes: int
    measured_bytes: int
    free_bytes_nvml: int | None
    tenants: tuple[str, ...]

    @property
    def waste_bytes(self) -> int:
        return self.reserved_bytes - self.measured_bytes

    @property
    def available_bytes(self) -> int:
        return max(0, self.total_bytes - self.reserved_bytes - self.corridor_bytes)

    @property
    def corridor_ok(self) -> bool:
        """#330's rule, checked against the driver when the driver can be read.

        The ledger cannot answer this on its own: reservations describe
        promises, and an undeclared CUDA context still occupies bytes.
        """
        if self.free_bytes_nvml is None:
            return self.available_bytes >= 0
        return self.free_bytes_nvml >= self.corridor_bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "card_uuid": self.card_uuid,
            "total_bytes": self.total_bytes,
            "total_mib": self.total_bytes // MIB,
            "corridor_bytes": self.corridor_bytes,
            "reserved_bytes": self.reserved_bytes,
            "reserved_mib": self.reserved_bytes // MIB,
            "measured_bytes": self.measured_bytes,
            "waste_bytes": self.waste_bytes,
            "waste_mib": self.waste_bytes // MIB,
            "available_bytes": self.available_bytes,
            "available_mib": self.available_bytes // MIB,
            "free_bytes_nvml": self.free_bytes_nvml,
            "corridor_ok": self.corridor_ok,
            "tenants": list(self.tenants),
        }


@dataclass(frozen=True)
class PlanResult:
    """The dry-run answer of ``GET /registry/plan`` (§7.4)."""

    engine_id: str
    fits: bool
    cards: tuple[str, ...]
    profile: ResourceProfile | None
    report: FeasibilityReport
    would_evict: tuple[str, ...] = ()
    projected_wait_ms: float = 0.0
    projected_wait_is_estimated: bool = True
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "fits": self.fits,
            "cards": list(self.cards),
            "profile": self.profile.to_json() if self.profile else None,
            "feasible_without_eviction": self.report.fits,
            "shortfall_bytes": self.report.shortfall_bytes,
            "shortfall_detail": self.report.render(),
            "would_evict": list(self.would_evict),
            "projected_wait_ms": self.projected_wait_ms,
            "projected_wait_is_estimated": self.projected_wait_is_estimated,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HotCapacity:
    """M, derived (§7.2). Not a configured constant."""

    count: int
    engine_ids: tuple[str, ...]
    excluded: tuple[tuple[str, str], ...]
    capped_by_max_hot: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "derived_max_hot": self.count,
            "would_be_hot": list(self.engine_ids),
            "excluded": [{"engine_id": e, "why": w} for e, w in self.excluded],
            "capped_by_max_hot": self.capped_by_max_hot,
        }


#: Descending residency, most resident first. Only used to tell a promotion
#: from a demotion; the ladder's own ordering lives in ``ladder.RUNG_ORDER``
#: and this is pinned against it by test.
_RESIDENCY_RANK: dict[ResidencyState, int] = {
    ResidencyState.HOT: 0,
    ResidencyState.WARM_GPU: 1,
    ResidencyState.WARM_HOST: 2,
    ResidencyState.COLD: 3,
}


@dataclass
class _Transition:
    engine_id: str
    to_state: ResidencyState
    ts: float


class EngineRegistry:
    """N registered, M hot, one ledger, one lock discipline.

    Thread-safe by a single re-entrant lock around every mutating path. A
    finer-grained scheme would buy nothing: transitions are seconds-to-minutes
    long and are already serialised by the physical card.
    """

    def __init__(
        self,
        *,
        store: ReservationStore,
        card_totals: Mapping[str, int],
        default_hot: Sequence[str] = (),
        max_hot: int | None = None,
        idle_after_s: float = 900.0,
        work_dir: str | None = None,
        clock: Callable[[], float] = time.time,
        free_bytes_resolver: Callable[[str], int | None] | None = None,
    ) -> None:
        self.store = store
        self.card_totals = dict(card_totals)
        if not self.card_totals:
            raise RegistryError(
                "the registry needs at least one card; pass card_totals from NVML"
            )
        self.max_hot = None if max_hot is None else int(max_hot)
        if self.max_hot is not None and self.max_hot < 0:
            raise RegistryError("--registry-max-hot must not be negative")
        self.idle_after_s = float(idle_after_s)
        self.work_dir = work_dir
        self._clock = clock
        self._free_bytes_resolver = free_bytes_resolver
        self._lock = threading.RLock()
        self._instances: dict[str, EngineInstance] = {}
        self._adapters: dict[str, Any] = {}
        self._holds: dict[str, MultiCardReservation] = {}
        self._default_hot: tuple[str, ...] = tuple(default_hot)
        self._history: list[_Transition] = []
        self.thrash_events: list[dict[str, Any]] = []
        #: #305 request-path binding: how many requests are being served by
        #: each engine right now. ``acquire_for_request`` raises it,
        #: ``release_after_request`` lowers it. The control tick reads it, and
        #: that is the whole reason it exists: ``last_used_ts`` says when a
        #: request STARTED, which is not enough to keep a long generation from
        #: being demoted out from under itself.
        self._inflight: dict[str, int] = {}

    # -- introspection -----------------------------------------------------

    @property
    def default_hot(self) -> tuple[str, ...]:
        return self._default_hot

    def engines(self) -> tuple[EngineInstance, ...]:
        with self._lock:
            return tuple(self._instances.values())

    def instance(self, engine_id: str) -> EngineInstance:
        try:
            return self._instances[engine_id]
        except KeyError:
            raise UnknownEngineError(
                f"no engine {engine_id!r} is registered; known: "
                f"{sorted(self._instances)}"
            ) from None

    def adapter(self, engine_id: str) -> Any:
        self.instance(engine_id)
        return self._adapters[engine_id]

    def context(self) -> AdapterContext:
        return AdapterContext(
            card_totals=dict(self.card_totals),
            card_exclusive_lock=self.store.card_exclusive_lock,
            work_dir=self.work_dir,
        )

    # -- cards -------------------------------------------------------------

    def reap(self, cards: Iterable[str] | None = None) -> list[str]:
        """Drop reservations whose lease lapsed *and* whose process is gone.

        Called before every admission and every card report. A plan computed
        against a crashed tenant's bytes rejects an engine that would fit,
        which is the same user-visible failure as the ledger not existing.
        Nothing live is ever reclaimed: both conditions are required.
        """
        reclaimed: list[str] = []
        for card_uuid in cards if cards is not None else self.card_totals:
            for entry in self.store.reap(card_uuid):
                reclaimed.append(entry.tenant_id)
        return reclaimed

    def cards(self) -> tuple[CardView, ...]:
        self.reap()
        views = []
        for card_uuid, total in sorted(self.card_totals.items()):
            ledger = self.store.read(card_uuid)
            free = None
            if self._free_bytes_resolver is not None:
                free = self._free_bytes_resolver(card_uuid)
            views.append(
                CardView(
                    card_uuid=card_uuid,
                    total_bytes=int(total),
                    corridor_bytes=self.store.corridor_bytes,
                    reserved_bytes=ledger.reserved_bytes,
                    measured_bytes=ledger.measured_bytes,
                    free_bytes_nvml=free,
                    tenants=tuple(e.tenant_id for e in ledger.entries),
                )
            )
        return tuple(views)

    def slots(self) -> tuple[Slot, ...]:
        out: list[Slot] = []
        for card_uuid in sorted(self.card_totals):
            for entry in self.store.read(card_uuid).entries:
                out.append(
                    Slot(
                        engine_id=entry.tenant_id,
                        card_uuid=card_uuid,
                        reserved_bytes=entry.reserved_bytes,
                        state=entry.state,
                        posts=dict(entry.posts),
                        measured_bytes=entry.measured_bytes,
                    )
                )
        return tuple(out)

    def capture_lock(self, card_uuid: str, **kwargs: Any):
        """§3.6: one capture lock per physical GPU.

        Graph capture takes over the capturing stream and other tenants'
        kernels on that device can stall or fail, so the window is exclusive.
        The same lock serialises KV-cache profiling, which measures free memory
        and is meaningless while a neighbour allocates.
        """
        return self.store.card_exclusive_lock(card_uuid, **kwargs)

    # -- registration ------------------------------------------------------

    def register(self, spec: EngineSpec, *, replace: bool = False) -> PlanResult:
        """Validate a spec and record it. Boots nothing (§7.4).

        Rejection happens here, in milliseconds, rather than at the first
        request or -- worse -- three minutes into a load. The plan is returned
        either way so the caller sees the numbers behind the verdict.
        """
        with self._lock:
            if spec.engine_id in self._instances and not replace:
                raise RegistrationRejected(
                    f"engine {spec.engine_id!r} is already registered; deregister it "
                    "or pass replace=true"
                )
            adapter = build_adapter(spec, self.context())
            cards, profile = self._resolve_placement(spec, adapter)
            report = plan_reservation(
                self.store,
                profile.demands(),
                self.card_totals,
                excluding_tenant=spec.tenant_id,
            )
            # A spec is registrable even when it does not fit *right now*:
            # fitting is a function of what else is hot, and the answer changes
            # when a neighbour demotes. What is NOT registrable is a spec that
            # cannot fit on an empty rig, which is a configuration error the
            # operator must see immediately.
            empty = plan_reservation(
                self.store,
                profile.demands(),
                self.card_totals,
                on_empty_rig=True,
            )
            if not empty.fits:
                raise RegistrationRejected(
                    f"engine {spec.engine_id!r} cannot fit on this rig even with every "
                    f"other tenant evicted: {empty.render()}",
                    report=empty,
                )
            if spec.engine_id in self._instances and replace:
                self._teardown(spec.engine_id)
            instance = EngineInstance(spec=spec, cards=cards, profile=profile)
            self._instances[spec.engine_id] = instance
            self._adapters[spec.engine_id] = adapter
            if hasattr(adapter, "bind"):
                adapter.bind(cards)
            self._validate_default_hot(self._default_hot)
            logger.info(
                "registry: registered %s (class %d, %s, %d MiB over %d card(s))",
                spec.engine_id,
                int(spec.klass),
                spec.adapter,
                profile.total_peak_bytes // MIB,
                len(cards),
            )
            return PlanResult(
                engine_id=spec.engine_id,
                fits=report.fits,
                cards=cards,
                profile=profile,
                report=report,
                reason="registered" if report.fits else "registered; does not fit now",
            )

    def deregister(self, engine_id: str) -> None:
        """§7.4: demote to COLD, release the slots, forget the spec."""
        with self._lock:
            self.instance(engine_id)
            self._teardown(engine_id)
            self._instances.pop(engine_id, None)
            self._adapters.pop(engine_id, None)
            self._default_hot = tuple(e for e in self._default_hot if e != engine_id)
            logger.info("registry: deregistered %s", engine_id)

    def _teardown(self, engine_id: str) -> None:
        instance = self._instances.get(engine_id)
        adapter = self._adapters.get(engine_id)
        if instance is not None and adapter is not None:
            if instance.state != ResidencyState.COLD:
                try:
                    adapter.demote(ResidencyState.COLD)
                except AdapterError as exc:
                    logger.warning(
                        "registry: %s did not demote cleanly (%s); releasing its "
                        "slots anyway, because a slot held by a dead tenant is "
                        "worse than a leaked process the operator can see",
                        engine_id,
                        exc,
                    )
                instance.state = ResidencyState.COLD
        hold = self._holds.pop(engine_id, None)
        if hold is not None:
            hold.release()

    def set_default_hot(self, engine_ids: Sequence[str]) -> None:
        """§7.3. Validated now, so an unsatisfiable set is a startup error."""
        with self._lock:
            wanted = tuple(engine_ids)
            self._validate_default_hot(wanted)
            self._default_hot = wanted

    def _validate_default_hot(self, engine_ids: Sequence[str]) -> None:
        unknown = [e for e in engine_ids if e not in self._instances]
        if unknown:
            raise RegistrationRejected(
                f"default_hot names unregistered engine(s) {unknown}"
            )
        demands: list[CardDemand] = []
        merged: dict[str, int] = {}
        posts: dict[str, dict[str, int]] = {}
        for engine_id in engine_ids:
            profile = self._instances[engine_id].profile
            if profile is None:  # pragma: no cover - registration always sets one
                continue
            for demand in profile.demands():
                merged[demand.card_uuid] = (
                    merged.get(demand.card_uuid, 0) + demand.reserved_bytes
                )
                posts.setdefault(demand.card_uuid, {})[
                    engine_id
                ] = demand.reserved_bytes
        demands = [
            CardDemand(card_uuid=c, reserved_bytes=b, posts=posts.get(c, {}))
            for c, b in merged.items()
        ]
        report = plan_reservation(
            self.store,
            demands,
            self.card_totals,
            on_empty_rig=True,
        )
        if not report.fits:
            raise RegistrationRejected(
                "the idle default_hot set "
                f"{list(engine_ids)} does not satisfy the ledger invariant, so the "
                "arbiter could never return to it: " + report.render(),
                report=report,
            )

    # -- placement ---------------------------------------------------------

    def _resolve_placement(
        self, spec: EngineSpec, adapter: Any
    ) -> tuple[tuple[str, ...], ResourceProfile]:
        if not spec.is_auto_placed:
            unknown = [c for c in spec.placement if c not in self.card_totals]
            if unknown:
                raise RegistrationRejected(
                    f"engine {spec.engine_id!r} is placed on card(s) {unknown}, which "
                    f"NVML does not report. Known cards: {sorted(self.card_totals)}. "
                    "Placement is by UUID because enumeration order shifts between "
                    "boots."
                )
            cards = tuple(spec.placement)
            return cards, adapter.estimate(spec, cards)
        # AUTO: one card, most available first, ties by UUID so the choice is
        # reproducible. Anything richer is the planner's job (§7.7).
        views = sorted(self.cards(), key=lambda v: (-v.available_bytes, v.card_uuid))
        first_error: Exception | None = None
        for view in views:
            cards = (view.card_uuid,)
            try:
                profile = adapter.estimate(spec, cards)
            except (SpecError, AdapterError) as exc:
                first_error = first_error or exc
                continue
            report = plan_reservation(
                self.store,
                profile.demands(),
                self.card_totals,
                excluding_tenant=spec.tenant_id,
            )
            if report.fits:
                return cards, profile
        # Nothing fits right now: fall back to the most-free card so the spec is
        # still costed and the rejection quotes real numbers.
        cards = (views[0].card_uuid,)
        try:
            return cards, adapter.estimate(spec, cards)
        except (SpecError, AdapterError) as exc:
            raise RegistrationRejected(
                f"engine {spec.engine_id!r} could not be costed on any card: "
                f"{first_error or exc}"
            ) from None

    # -- derived M ---------------------------------------------------------

    def hot_capacity(self) -> HotCapacity:
        """§7.2: how many registered engines fit hot, derived from bytes.

        Greedy in the order the arbiter would actually promote -- pinned
        first, then the idle default set, then priority, then registration
        order -- so the answer is the one the rig would really reach, not an
        optimistic bin-packing.
        """
        with self._lock:
            held: dict[str, int] = {
                c: self.store.read(c).reserved_bytes for c in self.card_totals
            }
            # Start from an empty rig: this is a capacity question, not a
            # snapshot of who happens to hold what.
            for engine in self._instances.values():
                if engine.profile is None:
                    continue
                for demand in engine.profile.demands():
                    if engine.is_gpu_resident:
                        held[demand.card_uuid] = max(
                            0, held[demand.card_uuid] - demand.reserved_bytes
                        )
            chosen: list[str] = []
            excluded: list[tuple[str, str]] = []
            capped = False
            for engine in self._promotion_order():
                if engine.profile is None:  # pragma: no cover
                    continue
                if self.max_hot is not None and len(chosen) >= self.max_hot:
                    capped = True
                    excluded.append(
                        (engine.engine_id, f"--registry-max-hot={self.max_hot} reached")
                    )
                    continue
                blocked = None
                for demand in engine.profile.demands():
                    total = self.card_totals[demand.card_uuid]
                    if (
                        held[demand.card_uuid]
                        + demand.reserved_bytes
                        + self.store.corridor_bytes
                        > total
                    ):
                        blocked = (
                            f"card {demand.card_uuid} would hold "
                            f"{(held[demand.card_uuid] + demand.reserved_bytes) // MIB}"
                            f" MiB + {self.store.corridor_bytes // MIB} MiB corridor "
                            f"against {total // MIB} MiB total"
                        )
                        break
                if blocked:
                    excluded.append((engine.engine_id, blocked))
                    continue
                for demand in engine.profile.demands():
                    held[demand.card_uuid] += demand.reserved_bytes
                chosen.append(engine.engine_id)
            return HotCapacity(
                count=len(chosen),
                engine_ids=tuple(chosen),
                excluded=tuple(excluded),
                capped_by_max_hot=capped,
            )

    def _promotion_order(self) -> list[EngineInstance]:
        order = list(self._instances.values())
        default_hot = set(self._default_hot)
        order.sort(
            key=lambda e: (
                not e.spec.pinned,
                e.engine_id not in default_hot,
                -e.spec.priority,
                e.engine_id,
            )
        )
        return order

    # -- planning ----------------------------------------------------------

    def plan(self, spec_or_id: EngineSpec | str) -> PlanResult:
        """Dry-run: would this fit, and what would be evicted (§7.4)."""
        with self._lock:
            if isinstance(spec_or_id, str):
                instance = self.instance(spec_or_id)
                spec = instance.spec
                cards, profile = instance.cards, instance.profile
                if profile is None:  # pragma: no cover
                    raise RegistryError(f"engine {spec_or_id!r} has no profile")
            else:
                spec = spec_or_id
                adapter = build_adapter(spec, self.context())
                cards, profile = self._resolve_placement(spec, adapter)
            self.reap({d.card_uuid for d in profile.demands()})
            report = plan_reservation(
                self.store,
                profile.demands(),
                self.card_totals,
                excluding_tenant=spec.tenant_id,
            )
            if report.fits:
                wait, estimated = self._promotion_cost(spec.engine_id, spec)
                return PlanResult(
                    engine_id=spec.engine_id,
                    fits=True,
                    cards=cards,
                    profile=profile,
                    report=report,
                    projected_wait_ms=wait,
                    projected_wait_is_estimated=estimated,
                    reason="fits without eviction",
                )
            victims, after = self._eviction_plan(spec, profile)
            wait, estimated = self._projected_wait(spec, victims)
            return PlanResult(
                engine_id=spec.engine_id,
                fits=after.fits,
                cards=cards,
                profile=profile,
                report=report,
                would_evict=tuple(v.engine_id for v in victims),
                projected_wait_ms=wait,
                projected_wait_is_estimated=estimated,
                reason=(
                    "fits after eviction"
                    if after.fits
                    else "does not fit even after evicting every candidate: "
                    + after.render()
                ),
            )

    def _eviction_plan(
        self, spec: EngineSpec, profile: ResourceProfile
    ) -> tuple[list[EngineInstance], FeasibilityReport]:
        """§7.5 candidate order: not pinned, not default_hot, priority, LRU."""
        needed = {d.card_uuid for d in profile.demands()}
        default_hot = set(self._default_hot)

        def candidates(allow_default_hot: bool) -> list[EngineInstance]:
            out = [
                e
                for e in self._instances.values()
                if e.engine_id != spec.engine_id
                and e.is_gpu_resident
                and not e.spec.pinned
                and (allow_default_hot or e.engine_id not in default_hot)
                and needed.intersection(e.cards)
            ]
            out.sort(key=lambda e: (e.spec.priority, e.last_used_ts, e.engine_id))
            return out

        for allow_default_hot in (False, True):
            victims: list[EngineInstance] = []
            report = plan_reservation(
                self.store,
                profile.demands(),
                self.card_totals,
                excluding_tenant=spec.tenant_id,
                ignoring_tenants=[],
            )
            for candidate in candidates(allow_default_hot):
                if report.fits:
                    break
                victims.append(candidate)
                report = plan_reservation(
                    self.store,
                    profile.demands(),
                    self.card_totals,
                    excluding_tenant=spec.tenant_id,
                    ignoring_tenants=[v.engine_id for v in victims],
                )
            if report.fits:
                return victims, report
        return victims, report

    def _promotion_cost(
        self, engine_id: str, spec: EngineSpec | None = None
    ) -> tuple[float, bool]:
        instance = self._instances.get(engine_id)
        if instance is not None and instance.promotion_cost_ms is not None:
            return instance.promotion_cost_ms, False
        klass = int((spec or instance.spec).klass) if (spec or instance) else 1
        return DEFAULT_PROMOTION_COST_MS.get(klass, 60_000.0), True

    def _demotion_cost(self, instance: EngineInstance) -> tuple[float, bool]:
        if instance.demotion_cost_ms is not None:
            return instance.demotion_cost_ms, False
        return DEFAULT_DEMOTION_COST_MS.get(int(instance.spec.klass), 10_000.0), True

    def _projected_wait(
        self, spec: EngineSpec, victims: Sequence[EngineInstance]
    ) -> tuple[float, bool]:
        wait, estimated = self._promotion_cost(spec.engine_id, spec)
        for victim in victims:
            cost, victim_estimated = self._demotion_cost(victim)
            wait += cost
            estimated = estimated or victim_estimated
        return wait, estimated

    # -- transitions -------------------------------------------------------

    def ensure_state(
        self,
        engine_id: str,
        target: ResidencyState,
        *,
        max_promotion_wait_ms: float | None = None,
        allow_eviction: bool = True,
    ) -> EngineInstance:
        """Move one engine along the ladder, evicting if §7.5 permits it."""
        with self._lock:
            instance = self.instance(engine_id)
            adapter = self._adapters[engine_id]
            if instance.state == target:
                instance.last_used_ts = self._clock()
                return instance
            # Direction decides, not the target's name. WARM_GPU is a
            # promotion when reached from COLD and a demotion when reached
            # from HOT; routing on the target alone sent HOT -> WARM_GPU into
            # ``_promote``, where Class 1's adapter refuses it outright ("no
            # promotion path HOT -> WARM_GPU", ``class1_srt.py:244-247``). The
            # TEIL-HOT rung was therefore unreachable downward for the class
            # that is the only one implementing it.
            if _RESIDENCY_RANK[target] < _RESIDENCY_RANK[instance.state]:
                return self._promote(
                    instance, adapter, target, max_promotion_wait_ms, allow_eviction
                )
            return self._demote(instance, adapter, target)

    def _promote(
        self,
        instance: EngineInstance,
        adapter: Any,
        target: ResidencyState,
        max_promotion_wait_ms: float | None,
        allow_eviction: bool,
    ) -> EngineInstance:
        spec = instance.spec
        profile = instance.profile
        assert profile is not None
        self.reap({d.card_uuid for d in profile.demands()})
        report = plan_reservation(
            self.store,
            profile.demands(),
            self.card_totals,
            excluding_tenant=spec.tenant_id,
        )
        victims: list[EngineInstance] = []
        if not report.fits:
            if not allow_eviction:
                raise PromotionRejected(
                    f"engine {spec.engine_id!r} does not fit and eviction was not "
                    f"permitted: {report.render()}",
                    engine_id=spec.engine_id,
                    projected_wait_ms=0.0,
                    would_evict=(),
                    report=report,
                )
            victims, after = self._eviction_plan(spec, profile)
            if not after.fits:
                raise PromotionRejected(
                    f"engine {spec.engine_id!r} cannot be promoted: "
                    f"{after.render()}. Evictable tenants "
                    f"{[v.engine_id for v in victims] or 'none'} would not free "
                    "enough; the remaining holders are pinned or are themselves the "
                    "idle default set.",
                    engine_id=spec.engine_id,
                    projected_wait_ms=0.0,
                    would_evict=[v.engine_id for v in victims],
                    report=after,
                )
            wait, estimated = self._projected_wait(spec, victims)
            if max_promotion_wait_ms is not None and wait > max_promotion_wait_ms:
                raise PromotionRejected(
                    f"engine {spec.engine_id!r} would take about {wait:.0f} ms to "
                    f"promote ({'estimated' if estimated else 'measured'}), which "
                    f"exceeds the caller's budget of {max_promotion_wait_ms:.0f} ms. "
                    f"Getting there would evict {[v.engine_id for v in victims]}. "
                    "Wait, raise the budget, or use an engine that is already hot.",
                    engine_id=spec.engine_id,
                    projected_wait_ms=wait,
                    would_evict=[v.engine_id for v in victims],
                    report=report,
                    cost_is_estimated=estimated,
                )
            for victim in victims:
                logger.info(
                    "registry: evicting %s to make room for %s",
                    victim.engine_id,
                    spec.engine_id,
                )
                self._demote(
                    victim, self._adapters[victim.engine_id], ResidencyState.COLD
                )
        started = time.monotonic()
        hold = self._holds.get(spec.engine_id)
        if hold is None:
            hold = MultiCardReservation(
                self.store,
                tenant_id=spec.tenant_id,
                klass=int(spec.klass),
                totals=self.card_totals,
            )
            self._holds[spec.engine_id] = hold
        try:
            hold.acquire(profile.demands(), state=target)
        except LedgerError as exc:
            raise PromotionRejected(
                str(exc),
                engine_id=spec.engine_id,
                projected_wait_ms=0.0,
                would_evict=[v.engine_id for v in victims],
                report=report,
            ) from None
        try:
            adapter.promote(target)
        except Exception:
            hold.release()
            self._holds.pop(spec.engine_id, None)
            instance.state = ResidencyState.COLD
            instance.last_error = "promotion failed"
            raise
        elapsed_ms = (time.monotonic() - started) * 1000.0
        instance.observe_promotion(elapsed_ms)
        instance.state = target
        instance.pids = tuple(adapter.pids())
        instance.last_used_ts = self._clock()
        instance.transitions += 1
        instance.last_error = None
        hold.update_measured(adapter.measured())
        self._record(spec.engine_id, target)
        logger.info(
            "registry: %s -> %s in %.0f ms",
            spec.engine_id,
            target.value,
            elapsed_ms,
        )
        return instance

    def _demote(
        self, instance: EngineInstance, adapter: Any, target: ResidencyState
    ) -> EngineInstance:
        started = time.monotonic()
        adapter.demote(target)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        instance.observe_demotion(elapsed_ms)
        previous, instance.state = instance.state, target
        instance.transitions += 1
        hold = self._holds.get(instance.engine_id)
        if hold is not None:
            if target == ResidencyState.COLD:
                hold.release()
                self._holds.pop(instance.engine_id, None)
                instance.pids = ()
            else:
                hold.set_state(target)
                instance.pids = tuple(adapter.pids())
        self._record(instance.engine_id, target)
        logger.info(
            "registry: %s %s -> %s in %.0f ms",
            instance.engine_id,
            previous.value,
            target.value,
            elapsed_ms,
        )
        return instance

    def acquire_for_request(
        self, engine_id: str, *, max_promotion_wait_ms: float | None = None
    ) -> EngineInstance:
        """§7.5's admission path: make this engine serve, or say why not.

        Three refusals, all before any actuator runs, all naming themselves:

        * ``UnknownEngineError`` -- the model is not registered here.
        * ``LadderRefusal`` -- the engine IS registered but the edge from where
          it currently sits up to HOT is not implemented for its class. This is
          the ``ladder.py`` gate, and it is checked here rather than left to the
          adapter so the caller is refused instead of discovering an
          ``AdapterError`` mid-promotion.
        * ``PromotionRejected`` -- the edge exists but the rig cannot fund it
          now, or funding it would cost more than the caller's budget. Carries
          the projected wait and the eviction list.

        On success the in-flight count for this engine rises by one and stays
        up until :meth:`release_after_request`. A caller that acquires and
        never releases pins the engine hot, which is why the request path
        releases in a ``finally``.
        """
        from sglang.srt.registry import ladder  # noqa: PLC0415 - avoid a cycle

        with self._lock:
            instance = self.instance(engine_id)
            current = ladder.rung_of_state(instance.state)
            klass = instance.spec.adapter
            if current != ladder.HOT:
                ladder.check_transition(klass, current, ladder.HOT)
        instance = self.ensure_state(
            engine_id,
            ResidencyState.HOT,
            max_promotion_wait_ms=max_promotion_wait_ms,
        )
        with self._lock:
            instance.last_used_ts = self._clock()
            self._inflight[engine_id] = self._inflight.get(engine_id, 0) + 1
        return instance

    def release_after_request(self, engine_id: str) -> int:
        """Give back one in-flight slot. Returns what is still in flight.

        Never raises for an unknown or over-released engine: this runs in a
        ``finally`` on the request path, and turning a release into an error
        would replace a completed response with a 500. It floors at zero and
        says so in the log instead.
        """
        with self._lock:
            current = self._inflight.get(engine_id, 0)
            if current <= 0:
                logger.warning(
                    "registry: release_after_request(%s) with nothing in flight; "
                    "an acquire was lost or a release ran twice",
                    engine_id,
                )
                self._inflight.pop(engine_id, None)
                return 0
            remaining = current - 1
            if remaining:
                self._inflight[engine_id] = remaining
            else:
                self._inflight.pop(engine_id, None)
            instance = self._instances.get(engine_id)
            if instance is not None:
                # Stamp on RELEASE as well as on acquire: last_used_ts must
                # mean "last moment this engine was busy", or a generation
                # longer than idle_after_s looks idle while it is still
                # producing tokens.
                instance.last_used_ts = self._clock()
            return remaining

    def inflight(self, engine_id: str) -> int:
        with self._lock:
            return self._inflight.get(engine_id, 0)

    # -- idle --------------------------------------------------------------

    def idle_deadline(self) -> float | None:
        used = [i.last_used_ts for i in self._instances.values() if i.last_used_ts]
        return (max(used) + self.idle_after_s) if used else None

    def return_to_idle(self, *, force: bool = False) -> list[str]:
        """§7.3: demote everything outside ``default_hot``, promote everything in.

        Returns the engine ids that changed state. Idle handling is explicit
        rather than a background thread: the control plane calls it on its own
        tick, so a test can drive it deterministically and an operator can see
        it in a log line rather than infer it from a memory graph.
        """
        with self._lock:
            deadline = self.idle_deadline()
            if not force and deadline is not None and self._clock() < deadline:
                return []
            changed: list[str] = []
            wanted = set(self._default_hot)
            for instance in list(self._instances.values()):
                if instance.engine_id in wanted or instance.spec.pinned:
                    continue
                if instance.state != ResidencyState.COLD:
                    self._demote(
                        instance,
                        self._adapters[instance.engine_id],
                        ResidencyState.COLD,
                    )
                    changed.append(instance.engine_id)
            for engine_id in self._default_hot:
                instance = self._instances[engine_id]
                if instance.state != ResidencyState.HOT:
                    self.ensure_state(engine_id, ResidencyState.HOT)
                    changed.append(engine_id)
            return changed

    # -- thrash ------------------------------------------------------------

    def _record(self, engine_id: str, state: ResidencyState) -> None:
        now = self._clock()
        self._history.append(_Transition(engine_id, state, now))
        cutoff = now - THRASH_WINDOW_S
        self._history = [t for t in self._history if t.ts >= cutoff]
        counts: dict[str, int] = {}
        for transition in self._history:
            counts[transition.engine_id] = counts.get(transition.engine_id, 0) + 1
        hot = sorted(
            (eid for eid, n in counts.items() if n >= THRASH_FLIPS),
            key=lambda e: -counts[e],
        )
        if len(hot) >= 2:
            event = {
                "ts": now,
                "engines": hot[:2],
                "flips": {e: counts[e] for e in hot[:2]},
                "window_s": THRASH_WINDOW_S,
            }
            if not self.thrash_events or self.thrash_events[-1]["engines"] != hot[:2]:
                logger.warning(
                    "registry: THRASH between %s and %s (%d and %d transitions in "
                    "%.0fs). Neither is pinned and both are wanted; pin one, raise a "
                    "priority, or accept that they do not both fit.",
                    hot[0],
                    hot[1],
                    counts[hot[0]],
                    counts[hot[1]],
                    THRASH_WINDOW_S,
                )
                self.thrash_events.append(event)

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Everything ``GET /registry`` answers."""
        with self._lock:
            return {
                "engines": [i.to_json() for i in self._instances.values()],
                "slots": [s.to_json() for s in self.slots()],
                "cards": [c.to_json() for c in self.cards()],
                "default_hot": list(self._default_hot),
                "max_hot": self.max_hot,
                "hot_capacity": self.hot_capacity().to_json(),
                "idle_after_s": self.idle_after_s,
                "thrash_events": list(self.thrash_events),
                "inflight": dict(self._inflight),
            }

    def refresh_measured(self) -> None:
        """Pull ``adapter.measured()`` into the ledger for every live tenant."""
        with self._lock:
            for engine_id, adapter in self._adapters.items():
                hold = self._holds.get(engine_id)
                if hold is None:
                    continue
                try:
                    hold.update_measured(adapter.measured())
                except Exception as exc:  # noqa: BLE001 - reporting is best effort
                    logger.debug(
                        "registry: measured() for %s failed: %s", engine_id, exc
                    )

    def heartbeat(self) -> None:
        """gpu-arb convention: touch every held lease so nothing is reaped."""
        with self._lock:
            for hold in self._holds.values():
                hold.heartbeat()

    def shutdown(self) -> None:
        with self._lock:
            for engine_id in list(self._instances):
                self._teardown(engine_id)


def card_totals_from_nvml() -> dict[str, int]:
    """``{uuid: total_bytes}`` for every physical card, NVML order irrelevant."""
    from sglang.srt.registry.nvml import list_devices  # noqa: PLC0415

    return {d.uuid: d.total_bytes for d in list_devices()}


def free_bytes_from_nvml(card_uuid: str) -> int | None:
    from sglang.srt.registry.nvml import memory_info_for_uuid  # noqa: PLC0415

    try:
        return memory_info_for_uuid(card_uuid).free_bytes
    except Exception:  # noqa: BLE001 - the corridor check degrades, not fails
        return None


def iter_specs(payload: Iterable[Mapping[str, Any]]) -> list[EngineSpec]:
    return [EngineSpec.from_json(item) for item in payload]
