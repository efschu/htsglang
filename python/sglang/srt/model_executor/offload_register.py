# SPDX-License-Identifier: Apache-2.0
"""Short-term VRAM item offload register (#286, DESIGN_201 Nachtrag-13
Ergaenzungen 4/6/7/7b/7c).

ONE generic VRAM item tiering for the dual-lane runtime: everything that is
short-term dispensable during dual-lane operation is RAM-parkable, granularly
per item class. The register does NOT build a new movement mechanism -- per
Ergaenzung 7b it treats its items like the MoE expert-spill offload does
(#77/#123/#125/#236/#242): the expert offload owns the movement machinery
(pinned host pools, H2D wave-in behind compute, spill budgets); this register
introduces the NEW item classes into that tiering and owns the policy side
(class knobs, presets, hysteresis, priority protection, latency terms).

Item classes (each with its own knob ``resident | ram | auto`` plus a
depth FRACTION 0..1, Ergaenzung 7c):

* ``graph_rungs``     -- cold capture rungs of the K-/algo ladder (Erg. 4).
* ``drafter_heads``   -- the inactive speculative algo head (Erg. 6).
* ``lane_workspaces`` -- (lane, name)-keyed workspaces / input-buffer pools of
                         a lane idle longer than the hysteresis window (D2/D3).
* ``cold_lane``       -- everything a work-less lane owns alone (#89 path).
* ``experts``         -- the EXISTING MoE expert-spill items. Referenced here
                         so accounting and telemetry are unified; movement of
                         experts stays with the expert-offload machinery in
                         this phase (wiring both through one backend is a
                         GPU-phase item).

Three TIME-CONSTANT TIERS of the same mechanic (Ergaenzung 7c):

* ``wave``  (ms)        -- per-layer expert streaming behind compute
                           (#125 double-buffer, #254 wave order): the existing
                           machinery, referenced, not rebuilt.
* ``phase`` (10-100 ms) -- NEW: items carry a PHASE USAGE MASK
                           (``draft | verify | prefill | ...``); the stage-2
                           scheduler hook names what the next phase does not
                           need (``on_phase_boundary``), to be waved out and
                           back behind the running phase's compute -- the #125
                           mechanic at phase granularity (subsumes the #140
                           phase-multiplexed VRAM slots). In the CPU phase the
                           hook is a NO-OP interface: it plans, nothing moves.
* ``turn``  (s)         -- turn/hysteresis boundaries (Ergaenzung 7): cold
                           rungs, inactive drafter, idle-lane workspaces,
                           cold lane.

Presets (``--lane-offload-profile``) are ONLY bundles of the per-class knobs
(see ``resolve_class_policies``); every class knob remains individually
overridable via ``--lane-offload-class-policy`` (``class=mode`` or
``class=mode:fraction``).

Invariants -- enforced in code here, not by caller discipline:

* nothing is parked that was touched within its tier's hysteresis window;
* HOT items (active head, active rung, active-lane workspaces) are refused by
  ``park()`` outright, whatever the policy says;
* priority-protected items are never parked to make room for another item;
* per class no deeper than the configured FRACTION of the class's bytes;
* stage-2 (phase) candidates pass an OVERLAP-BUDGET check (bytes moved /
  PCIe rate vs hideable compute time of the neighbour phase -- honest
  physics, Erg. 7c: a 1.8-GB drafter per ~25-30 ms round is NOT hideable,
  small items or long phases are). The check is an injectable cost function;
  real rates come from the GPU measurement phase, tests inject fakes.
* a parked item is a LATENCY TERM for the #279 dispatcher
  (``retrieval_latency_ms``), never a non-availability.

The movement backend is an interface (``park`` / ``wave_in``); the CPU phase
ships only ``CpuFakeMovementBackend`` for hermetic tests. The real backend
(VMM remap #93 for VA-stable items + the #89 suspend path, riding the expert
offload's prefetch/wave machinery) is the GPU phase.

Everything here is pure Python and CPU-testable
(test/registered/unit/model_executor/test_offload_register.py).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Union,
)

OFFLOAD_CLASSES = (
    "graph_rungs",
    "drafter_heads",
    "lane_workspaces",
    "cold_lane",
    "experts",
)

OFFLOAD_POLICIES = ("resident", "ram", "auto")

OFFLOAD_PROFILES = ("latency", "capacity", "auto")

# Ergaenzung 7c: the three time-constant tiers of the same mechanic.
TIME_CONSTANT_TIERS = ("wave", "phase", "turn")

# Default turn-tier hysteresis: matches the lane-lending threshold's order of
# magnitude (--dual-group-lane-lend-threshold-s); turn-tier items move at
# turn/hysteresis boundaries (1e2-1e3 ms and up), never per round.
DEFAULT_HYSTERESIS_WINDOW_S = 5.0

# Default phase-tier hysteresis: phase-tier items move at phase boundaries
# (10-100 ms grain); their hysteresis clock is a separate, much shorter one
# than the turn window (three time constants, one mechanic). 0.0 until the
# measurement phase supplies a figure.
DEFAULT_PHASE_HYSTERESIS_WINDOW_S = 0.0


class OffloadRefused(RuntimeError):
    """A park() the invariants forbid. Callers treat this as 'stay resident';
    it is a normal control-flow answer, not a configuration error."""


@dataclass(frozen=True)
class ClassPolicy:
    """Per-class knob: mode plus how DEEP the class may be parked.

    ``fraction`` is the share of the class's registered bytes that may be
    parked at most (fully, or only as deep as desired -- Erg. 7c): 0.0 =
    nothing, 1.0 = everything. ``None`` is the auto/Messpflicht placeholder:
    the depth will come from the measured per-(class, phase) overlap
    accounting in the GPU phase; until then ``None`` imposes no static cap
    (the mode gate -- sensor for ``auto`` -- is what admits a park at all).
    """

    mode: str
    fraction: Optional[float]

    def __post_init__(self):
        if self.mode not in OFFLOAD_POLICIES:
            raise ValueError(
                f"unknown offload policy {self.mode!r}; known policies: "
                f"{', '.join(OFFLOAD_POLICIES)}."
            )
        if self.fraction is not None and not (0.0 <= self.fraction <= 1.0):
            raise ValueError(
                f"offload fraction must be within [0, 1], got {self.fraction}"
            )


def _default_fraction(mode: str) -> Optional[float]:
    # resident = park nothing; ram = full depth; auto = Messpflicht
    # placeholder (measured depth in the GPU phase).
    return {"resident": 0.0, "ram": 1.0, "auto": None}[mode]


def _parse_one_policy(klass: str, spec: str, flag: str) -> ClassPolicy:
    mode, sep, frac_str = spec.partition(":")
    mode = mode.strip()
    if mode not in OFFLOAD_POLICIES:
        raise ValueError(
            f"{flag} sets class {klass!r} to unknown policy {mode!r}; "
            f"known policies: {', '.join(OFFLOAD_POLICIES)}."
        )
    if not sep:
        return ClassPolicy(mode, _default_fraction(mode))
    frac_str = frac_str.strip()
    try:
        fraction = float(frac_str)
    except ValueError:
        raise ValueError(
            f"{flag} entry {klass}={spec!r}: fraction {frac_str!r} is not a number."
        ) from None
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(
            f"{flag} entry {klass}={spec!r}: fraction must be within "
            f"[0, 1], got {fraction}."
        )
    return ClassPolicy(mode, fraction)


def parse_class_policy_overrides(spec: Optional[str]) -> Dict[str, ClassPolicy]:
    """Parse ``--lane-offload-class-policy`` (e.g. ``"drafter_heads=ram:0.5,
    graph_rungs=auto"``) into ``{class: ClassPolicy}``. Hard errors on unknown
    classes/policies and malformed entries -- this runs at argument time so a
    typo fails the boot, not the first park."""
    flag = "--lane-offload-class-policy"
    if not spec or not spec.strip():
        return {}
    overrides: Dict[str, ClassPolicy] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"{flag} entry {entry!r} is not of the form "
                f"<class>=<policy>[:<fraction>] (classes: "
                f"{', '.join(OFFLOAD_CLASSES)}; policies: "
                f"{', '.join(OFFLOAD_POLICIES)})."
            )
        klass, _, policy_spec = entry.partition("=")
        klass, policy_spec = klass.strip(), policy_spec.strip()
        if klass not in OFFLOAD_CLASSES:
            raise ValueError(
                f"{flag} names unknown item class {klass!r}; known classes: "
                f"{', '.join(OFFLOAD_CLASSES)}."
            )
        policy = _parse_one_policy(klass, policy_spec, flag)
        if klass in overrides and overrides[klass] != policy:
            raise ValueError(
                f"{flag} sets class {klass!r} twice "
                f"({overrides[klass]} and {policy}); pick one."
            )
        overrides[klass] = policy
    return overrides


def resolve_class_policies(
    profile: str,
    overrides: Optional[Union[str, Mapping[str, Union[str, ClassPolicy]]]] = None,
) -> Dict[str, ClassPolicy]:
    """Expand a profile preset into the per-class policy map, then apply the
    explicit per-class overrides ON TOP (an explicit knob always beats the
    preset -- presets are only bundles of the individual knobs).

    * ``latency``  -- every class resident, fraction 0 (flips stay us-fast,
                      costs KV).
    * ``capacity`` -- every class ram, fraction 1 (park aggressively, max
                      KV/max-token gain), still subject to the runtime
                      invariants (hysteresis, hot refusal, priority
                      protection, overlap budget on the phase tier).
    * ``auto``     -- every class auto, fraction = Messpflicht placeholder:
                      park only what the saturation sensor (the group-wide
                      signal shared with 13e/#279) and the measured overlap
                      accounting admit.
    """
    if profile not in OFFLOAD_PROFILES:
        raise ValueError(
            f"--lane-offload-profile must be one of "
            f"{', '.join(OFFLOAD_PROFILES)}; got {profile!r}."
        )
    base_mode = {
        "latency": "resident",
        "capacity": "ram",
        "auto": "auto",
    }[profile]
    base = ClassPolicy(base_mode, _default_fraction(base_mode))
    policies = {klass: base for klass in OFFLOAD_CLASSES}
    if overrides is not None:
        if isinstance(overrides, str):
            overrides = parse_class_policy_overrides(overrides)
        for klass, policy in overrides.items():
            if klass not in OFFLOAD_CLASSES:
                raise ValueError(
                    f"unknown offload item class {klass!r}; known classes: "
                    f"{', '.join(OFFLOAD_CLASSES)}."
                )
            if isinstance(policy, str):
                policy = _parse_one_policy(klass, policy, "--lane-offload-class-policy")
            policies[klass] = policy
    return policies


@dataclass
class OffloadItem:
    """One registered VRAM item.

    ``restore_cost_ms`` is a callable so the estimate can start as a static
    class-level figure and later be replaced by the measured wave-in latency
    from the offload telemetry (Messpflicht, Ergaenzung 4/7) without changing
    this interface.

    ``hot`` is the item's hot criterion: True means the item is in active use
    (active head, active rung, active-lane workspace) and must stay resident.
    ``park()`` refuses hot items unconditionally.

    ``phase_mask`` (Erg. 7c) is the set of round phases (``draft``,
    ``verify``, ``prefill``, ...) in which the item is needed. An EMPTY mask
    means "needed in every phase" (always hot for the stage-2 scheduler); the
    phase vocabulary is open, so "full mask" is expressed as empty.

    ``time_constant_tier`` names which clock governs the item: ``wave``
    (existing per-layer expert streaming), ``phase`` (stage-2 phase
    multiplexing), ``turn`` (Ergaenzung-7 hysteresis boundaries).
    """

    item_id: str
    offload_class: str
    size_bytes: int
    restore_cost_ms: Callable[[], float]
    hot: Callable[[], bool]
    va_stable_required: bool = False
    prio_protected: bool = False
    phase_mask: FrozenSet[str] = frozenset()
    time_constant_tier: str = "turn"
    last_access_s: float = 0.0
    parked: bool = False


class MovementBackend:
    """Interface of the movement half. The register decides WHETHER an item
    may move (policy + invariants); the backend does the moving. The GPU-phase
    backend is the expert-offload machinery (pinned pools, H2D wave behind
    compute, VMM remap #93 for ``va_stable_required`` items, #89 suspend for
    ``cold_lane``); the CPU phase ships only the fake below."""

    def park(self, item: OffloadItem) -> None:
        raise NotImplementedError

    def wave_in(self, item: OffloadItem) -> None:
        raise NotImplementedError


class CpuFakeMovementBackend(MovementBackend):
    """Records movements for hermetic CPU tests. No tensors, no CUDA."""

    def __init__(self):
        self.parked_ids: List[str] = []
        self.waved_in_ids: List[str] = []

    def park(self, item: OffloadItem) -> None:
        self.parked_ids.append(item.item_id)

    def wave_in(self, item: OffloadItem) -> None:
        self.waved_in_ids.append(item.item_id)


@dataclass
class PhaseBoundaryPlan:
    """What the stage-2 scheduler WOULD move at one phase boundary.

    CPU phase: planning only -- ``on_phase_boundary`` never moves anything.
    ``park_candidates`` is deterministic (sorted item ids) so the plan is
    testable and reproducible; ``skipped`` maps every considered-but-rejected
    item to the reason (invariant or budget) that excluded it.
    """

    current_phase: str
    next_phase: str
    park_candidates: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)


@dataclass
class OffloadRegisterStats:
    """Accounting the CPU-phase adapters feed (size/access bookkeeping only)."""

    registered: int = 0
    parks: int = 0
    park_refusals: int = 0
    wave_ins: int = 0
    phase_plans: int = 0


class OffloadRegister:
    """The class-registration + policy half of the generic VRAM item tiering.

    Thread-safe (one lock; adapters may register from lane threads). The
    register never moves bytes itself -- ``park()``/``wave_in()`` delegate to
    the movement backend once the policy and the invariants allow the move.

    ``overlap_budget_fn(item, current_phase, next_phase) -> bool`` is the
    injectable stage-2 cost function (Erg. 7c honest physics: bytes moved /
    PCIe rate vs hideable compute time of the neighbour phase). Absent, the
    phase planner names NO candidates: without measured rates nothing is
    provably hideable, so the hook stays a true no-op. Real rates come from
    the GPU measurement phase; tests inject fakes.
    """

    def __init__(
        self,
        policies: Optional[Mapping[str, Union[str, ClassPolicy]]] = None,
        backend: Optional[MovementBackend] = None,
        hysteresis_window_s: float = DEFAULT_HYSTERESIS_WINDOW_S,
        phase_hysteresis_window_s: float = DEFAULT_PHASE_HYSTERESIS_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
        overlap_budget_fn: Optional[Callable[[OffloadItem, str, str], bool]] = None,
    ):
        if policies is None:
            # Default preset is latency (all resident) until the measurement
            # phase supplies per-class figures (Messpflicht before any
            # auto/ram default, Ergaenzung 4/7).
            resolved: Dict[str, ClassPolicy] = resolve_class_policies("latency")
        else:
            resolved = {}
            for klass, policy in policies.items():
                if klass not in OFFLOAD_CLASSES:
                    raise ValueError(f"invalid policy map class {klass!r}")
                if isinstance(policy, str):
                    policy = ClassPolicy(policy, _default_fraction(policy))
                resolved[klass] = policy
        missing = set(OFFLOAD_CLASSES) - set(resolved)
        if missing:
            raise ValueError(
                f"policy map is missing classes {sorted(missing)}; use "
                f"resolve_class_policies() to build a complete map."
            )
        if hysteresis_window_s < 0 or phase_hysteresis_window_s < 0:
            raise ValueError("hysteresis windows must be >= 0")
        self._policies = resolved
        self._backend = backend or CpuFakeMovementBackend()
        self._hysteresis_window_s = float(hysteresis_window_s)
        self._phase_hysteresis_window_s = float(phase_hysteresis_window_s)
        self._clock = clock
        self._items: Dict[str, OffloadItem] = {}
        self._lock = threading.Lock()
        # Saturation-sensor hook for the ``auto`` policy: the group-wide
        # saturation/occupancy signal from 13e/#279 (which will also know the
        # PCIe side -- the bus is itself a budgeted item, Erg. 7c). Interface
        # now, sharpness later (same pattern as the drafter policy): until a
        # sensor is attached, ``auto`` behaves like ``resident`` -- the safe
        # direction.
        self._saturation_sensor: Optional[Callable[[], bool]] = None
        self._overlap_budget_fn = overlap_budget_fn
        self.stats = OffloadRegisterStats()

    # --- configuration ------------------------------------------------------
    @property
    def policies(self) -> Dict[str, ClassPolicy]:
        return dict(self._policies)

    @property
    def hysteresis_window_s(self) -> float:
        return self._hysteresis_window_s

    @property
    def phase_hysteresis_window_s(self) -> float:
        return self._phase_hysteresis_window_s

    def set_saturation_sensor(self, sensor: Optional[Callable[[], bool]]):
        """Attach the 13e/#279 group-wide saturation signal. ``sensor()``
        returning True means VRAM pressure justifies parking auto-class
        items. Placeholder hook in the CPU phase."""
        with self._lock:
            self._saturation_sensor = sensor

    def set_overlap_budget_fn(
        self, fn: Optional[Callable[[OffloadItem, str, str], bool]]
    ):
        """Attach the stage-2 overlap cost function (bytes/PCIe-rate vs
        hideable compute; measured in the GPU phase, faked in tests)."""
        with self._lock:
            self._overlap_budget_fn = fn

    # --- registration -------------------------------------------------------
    def register(
        self,
        item_id: str,
        offload_class: str,
        size_bytes: int,
        restore_cost_ms: Union[float, Callable[[], float]],
        hot: Optional[Callable[[], bool]] = None,
        va_stable_required: bool = False,
        prio_protected: bool = False,
        phase_mask: Optional[Iterable[str]] = None,
        time_constant_tier: str = "turn",
    ) -> OffloadItem:
        """Register one VRAM item. Duplicate ids and unknown classes/tiers
        are hard errors (an adapter registering twice is a lifecycle bug, not
        a situation to paper over)."""
        if offload_class not in OFFLOAD_CLASSES:
            raise ValueError(
                f"unknown offload item class {offload_class!r}; known: "
                f"{', '.join(OFFLOAD_CLASSES)}."
            )
        if time_constant_tier not in TIME_CONSTANT_TIERS:
            raise ValueError(
                f"unknown time-constant tier {time_constant_tier!r}; known: "
                f"{', '.join(TIME_CONSTANT_TIERS)}."
            )
        if size_bytes < 0:
            raise ValueError(f"size_bytes must be >= 0, got {size_bytes}")
        if callable(restore_cost_ms):
            cost_fn = restore_cost_ms
        else:
            cost_value = float(restore_cost_ms)
            if cost_value < 0:
                raise ValueError(f"restore_cost_ms must be >= 0, got {cost_value}")

            def cost_fn(_v=cost_value):
                return _v

        item = OffloadItem(
            item_id=item_id,
            offload_class=offload_class,
            size_bytes=int(size_bytes),
            restore_cost_ms=cost_fn,
            hot=hot if hot is not None else (lambda: False),
            va_stable_required=bool(va_stable_required),
            prio_protected=bool(prio_protected),
            phase_mask=frozenset(phase_mask) if phase_mask else frozenset(),
            time_constant_tier=time_constant_tier,
            last_access_s=self._clock(),
        )
        with self._lock:
            if item_id in self._items:
                raise ValueError(
                    f"offload item {item_id!r} is already registered "
                    f"(class {self._items[item_id].offload_class!r}); "
                    f"unregister first or fix the adapter lifecycle."
                )
            self._items[item_id] = item
            self.stats.registered += 1
        return item

    def unregister(self, item_id: str) -> None:
        with self._lock:
            self._items.pop(item_id, None)

    def touch(self, item_id: str) -> None:
        """Record an access (resets the item's hysteresis clock)."""
        with self._lock:
            item = self._items.get(item_id)
            if item is not None:
                item.last_access_s = self._clock()

    # --- queries ------------------------------------------------------------
    def get(self, item_id: str) -> Optional[OffloadItem]:
        with self._lock:
            return self._items.get(item_id)

    def is_parked(self, item_id: str) -> bool:
        item = self._require(item_id)
        return item.parked

    def items_of_class(self, offload_class: str) -> List[OffloadItem]:
        with self._lock:
            return [i for i in self._items.values() if i.offload_class == offload_class]

    def parked_bytes(self, offload_class: Optional[str] = None) -> int:
        with self._lock:
            return self._parked_bytes_locked(offload_class)

    def resident_bytes(self) -> int:
        with self._lock:
            return sum(i.size_bytes for i in self._items.values() if not i.parked)

    def retrieval_latency_ms(self, item_id: str) -> float:
        """The #279-dispatcher view of an item: a parked item contributes its
        estimated wave-in latency as a LATENCY TERM; a resident item
        contributes 0.0. There is deliberately no 'unavailable' answer --
        parked never means gone."""
        item = self._require(item_id)
        return float(item.restore_cost_ms()) if item.parked else 0.0

    def latency_term_ms(self, offload_class: Optional[str] = None) -> float:
        """Summed retrieval latency of the parked items (optionally of one
        class) -- what a dispatch choice that needs them all would pay."""
        with self._lock:
            return sum(
                float(i.restore_cost_ms())
                for i in self._items.values()
                if i.parked
                and (offload_class is None or i.offload_class == offload_class)
            )

    # --- stage-2 scheduler hook (Erg. 7c; plans only, moves nothing) --------
    def on_phase_boundary(
        self, current_phase: str, next_phase: str
    ) -> PhaseBoundaryPlan:
        """Name the phase-tier items the stage-2 scheduler COULD wave out at
        this boundary (their phase mask does not contain the next phase),
        after every invariant and the overlap budget had their say. No-op in
        the CPU phase: the plan is returned, nothing is moved.

        Deterministic: items are considered in sorted-id order, and the
        per-class fraction cap is applied greedily in that order.
        """
        plan = PhaseBoundaryPlan(current_phase, next_phase)
        with self._lock:
            self.stats.phase_plans += 1
            now = self._clock()
            planned_class_bytes: Dict[str, int] = {}
            for item_id in sorted(self._items):
                item = self._items[item_id]
                if item.time_constant_tier != "phase":
                    continue
                if item.parked:
                    continue
                if not item.phase_mask:
                    # Empty mask = needed in every phase (always hot here).
                    plan.skipped[item_id] = "needed in every phase"
                    continue
                if next_phase in item.phase_mask:
                    plan.skipped[item_id] = f"needed in phase {next_phase!r}"
                    continue
                if item.prio_protected:
                    plan.skipped[item_id] = "priority-protected"
                    continue
                if item.hot():
                    plan.skipped[item_id] = "hot (in active use)"
                    continue
                age_s = now - item.last_access_s
                if age_s < self._phase_hysteresis_window_s:
                    plan.skipped[item_id] = (
                        f"inside the {self._phase_hysteresis_window_s:.3f}s "
                        f"phase hysteresis window"
                    )
                    continue
                policy = self._policies[item.offload_class]
                if policy.mode == "resident":
                    plan.skipped[item_id] = "class policy is 'resident'"
                    continue
                if policy.mode == "auto":
                    sensor = self._saturation_sensor
                    if sensor is None or not sensor():
                        plan.skipped[item_id] = (
                            "auto policy without saturation pressure"
                        )
                        continue
                planned = planned_class_bytes.get(item.offload_class, 0)
                cap = self._fraction_cap_locked(item.offload_class)
                if cap is not None:
                    already = self._parked_bytes_locked(item.offload_class)
                    if already + planned + item.size_bytes > cap:
                        plan.skipped[item_id] = (
                            f"class fraction cap ({policy.fraction}) exhausted"
                        )
                        continue
                if self._overlap_budget_fn is None:
                    # Honest physics: without measured PCIe/compute rates
                    # nothing is provably hideable -- the hook stays no-op.
                    plan.skipped[item_id] = "no overlap-budget function"
                    continue
                if not self._overlap_budget_fn(item, current_phase, next_phase):
                    plan.skipped[item_id] = (
                        "movement not hideable behind neighbour-phase compute"
                    )
                    continue
                planned_class_bytes[item.offload_class] = planned + item.size_bytes
                plan.park_candidates.append(item_id)
        return plan

    # --- movement -----------------------------------------------------------
    def park(self, item_id: str, make_room_for: Optional[str] = None) -> None:
        """Park one item into RAM, if the policy and every invariant allow it.

        Raises ``OffloadRefused`` (a normal 'stays resident' answer) when:
        * the item's class policy is ``resident``;
        * the class policy is ``auto`` and the saturation sensor is absent or
          does not report pressure;
        * the item is HOT (its hot criterion says active use);
        * the item was touched within its tier's hysteresis window;
        * parking it would exceed the class's fraction cap;
        * the item is priority-protected and the park would make room for
          another item (``make_room_for`` given): the protected class is
          never parked to give someone else its space.
        """
        with self._lock:
            item = self._require_locked(item_id)
            if item.parked:
                return  # idempotent
            policy = self._policies[item.offload_class]
            if policy.mode == "resident":
                self._refuse(f"class {item.offload_class!r} policy is 'resident'")
            if policy.mode == "auto":
                sensor = self._saturation_sensor
                if sensor is None or not sensor():
                    # Interface now, sharpness later: without the 13e/#279
                    # signal (or without pressure) auto stays resident.
                    self._refuse(
                        f"class {item.offload_class!r} policy is 'auto' and "
                        f"the saturation sensor reports no pressure"
                    )
            if item.prio_protected and make_room_for is not None:
                self._refuse(
                    f"item {item_id!r} is priority-protected and would be "
                    f"parked to make room for {make_room_for!r}"
                )
            if item.hot():
                self._refuse(f"item {item_id!r} is hot (in active use)")
            window = (
                self._phase_hysteresis_window_s
                if item.time_constant_tier == "phase"
                else self._hysteresis_window_s
            )
            age_s = self._clock() - item.last_access_s
            if age_s < window:
                self._refuse(
                    f"item {item_id!r} was accessed {age_s:.3f}s ago, inside "
                    f"the {window:.3f}s hysteresis window of tier "
                    f"{item.time_constant_tier!r}"
                )
            cap = self._fraction_cap_locked(item.offload_class)
            if cap is not None:
                already = self._parked_bytes_locked(item.offload_class)
                if already + item.size_bytes > cap:
                    self._refuse(
                        f"parking item {item_id!r} ({item.size_bytes} B) "
                        f"would exceed class {item.offload_class!r} fraction "
                        f"cap {policy.fraction} ({already} B already parked, "
                        f"cap {cap} B)"
                    )
            self._backend.park(item)
            item.parked = True
            self.stats.parks += 1

    def wave_in(self, item_id: str) -> None:
        """Bring a parked item back. Always allowed -- retrieval is never
        refused; the cost is the latency term the dispatcher already saw."""
        with self._lock:
            item = self._require_locked(item_id)
            if not item.parked:
                return  # idempotent
            self._backend.wave_in(item)
            item.parked = False
            item.last_access_s = self._clock()
            self.stats.wave_ins += 1

    # --- internals ----------------------------------------------------------
    def _parked_bytes_locked(self, offload_class: Optional[str] = None) -> int:
        return sum(
            i.size_bytes
            for i in self._items.values()
            if i.parked and (offload_class is None or i.offload_class == offload_class)
        )

    def _fraction_cap_locked(self, offload_class: str) -> Optional[int]:
        """Class fraction cap in bytes, or None for 'no static cap'
        (fraction None = auto/Messpflicht placeholder)."""
        fraction = self._policies[offload_class].fraction
        if fraction is None:
            return None
        total = sum(
            i.size_bytes
            for i in self._items.values()
            if i.offload_class == offload_class
        )
        return int(fraction * total)

    def _require(self, item_id: str) -> OffloadItem:
        with self._lock:
            return self._require_locked(item_id)

    def _require_locked(self, item_id: str) -> OffloadItem:
        item = self._items.get(item_id)
        if item is None:
            raise ValueError(f"unknown offload item {item_id!r}")
        return item

    def _refuse(self, why: str) -> None:
        self.stats.park_refusals += 1
        raise OffloadRefused(f"park refused: {why}")


# --- process-global accessor (adapters) -------------------------------------
# The CPU-phase adapters only do registration + size/access bookkeeping, gated
# behind SGLANG_OFFLOAD_REGISTER (default off => the default path stays
# byte-identical: every maybe_* helper is a no-op returning None). State lives
# on this holder (no ``global`` statements; reset_global_register() gives
# tests scoped teardown).
class _GlobalRegisterHolder:
    def __init__(self):
        self.register: Optional[OffloadRegister] = None
        self.lock = threading.Lock()


_HOLDER = _GlobalRegisterHolder()


def offload_register_enabled() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_OFFLOAD_REGISTER.get())


def configure_global_register(
    profile: str,
    class_policy_overrides: Optional[str] = None,
    backend: Optional[MovementBackend] = None,
    hysteresis_window_s: float = DEFAULT_HYSTERESIS_WINDOW_S,
    phase_hysteresis_window_s: float = DEFAULT_PHASE_HYSTERESIS_WINDOW_S,
) -> OffloadRegister:
    """Build (or rebuild) the process-global register from the server-args
    knobs. Called once at runner init when the register is enabled."""
    policies = resolve_class_policies(profile, class_policy_overrides)
    reg = OffloadRegister(
        policies=policies,
        backend=backend,
        hysteresis_window_s=hysteresis_window_s,
        phase_hysteresis_window_s=phase_hysteresis_window_s,
    )
    with _HOLDER.lock:
        _HOLDER.register = reg
    return reg


def get_global_register() -> Optional[OffloadRegister]:
    """The process-global register, or None when the feature flag is off.
    When enabled but not yet configured, a default (latency-profile) register
    is created so early adapters can already book their items."""
    if not offload_register_enabled():
        return None
    with _HOLDER.lock:
        if _HOLDER.register is None:
            _HOLDER.register = OffloadRegister()
        return _HOLDER.register


def reset_global_register() -> None:
    """Tests (and a second in-process model load): drop the global register."""
    with _HOLDER.lock:
        _HOLDER.register = None


def maybe_register_item(
    item_id: str,
    offload_class: str,
    size_bytes: int,
    restore_cost_ms: Union[float, Callable[[], float]] = 0.0,
    hot: Optional[Callable[[], bool]] = None,
    va_stable_required: bool = False,
    prio_protected: bool = False,
    phase_mask: Optional[Iterable[str]] = None,
    time_constant_tier: str = "turn",
) -> Optional[OffloadItem]:
    """Adapter entry point: register an item iff the feature flag is on.
    Re-registration of a known id refreshes the bookkeeping (adapters run at
    creation sites that may rebuild their object) instead of raising."""
    reg = get_global_register()
    if reg is None:
        return None
    if reg.get(item_id) is not None:
        reg.unregister(item_id)
    return reg.register(
        item_id,
        offload_class,
        size_bytes,
        restore_cost_ms,
        hot=hot,
        va_stable_required=va_stable_required,
        prio_protected=prio_protected,
        phase_mask=phase_mask,
        time_constant_tier=time_constant_tier,
    )


def maybe_touch_item(item_id: str) -> None:
    """Adapter entry point: record an access iff the feature flag is on."""
    reg = get_global_register()
    if reg is not None:
        reg.touch(item_id)
