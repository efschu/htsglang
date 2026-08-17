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
* ``gdn_state_sets``  -- GDN-/Mamba session state SETS (Erg. 8): one item per
                         session slot across all GDN layers. The pool is
                         dimensioned for MAX sessions; with fewer running
                         sessions the surplus sets are dead weight. Parking
                         one is a VACATE-then-move, never a move of the live
                         set (#461): live GDN state is device-bound (DESIGN_407
                         X2) and is in any case a stride slice of the pool's
                         ``[num_layers, num_slots, ...]`` tensors rather than a
                         page range, so there is nothing to unmap. What travels
                         is the EXPORTED blob -- ``MambaPool.export_state_blob``
                         copies the slot's persistent fields to host tensors and
                         ``import_state_blob`` restores them into any free slot
                         -- and that blob may rest in host RAM or peer VRAM.
                         Turn-class time constant with its OWN boundary type:
                         moves happen at ADMISSION boundaries, planned by the
                         session-set ladder (``on_admission_boundary``), with the
                         invariant that an arriving session NEVER meets a
                         parked set (raise before admission, lower with
                         hysteresis).

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

import logging
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
    Sequence,
    Union,
)

logger = logging.getLogger(__name__)

OFFLOAD_CLASSES = (
    "graph_rungs",
    "drafter_heads",
    "lane_workspaces",
    "cold_lane",
    "experts",
    # Erg. 8: GDN-/Mamba state SETS. One item = ONE session slot across all
    # GDN layers (not the whole pool). Turn-class time constant, but its
    # moves happen at ADMISSION boundaries, planned by the session-set
    # ladder (offload_gdn_states.SessionSetLadder) via
    # ``on_admission_boundary`` -- never mid-decode.
    "gdn_state_sets",
    # Erg. 9b: the speculative KV SHADOW COPY of the KV pressure ladder
    # (kv_pressure_ladder.py). One item per pre-staged ladder rung, living on
    # the card that is probably about to join. Its distinguishing property:
    # the OLD layout stays the source of truth while a shadow exists, so this
    # is the only class whose park/discard is FREE -- there is nothing to
    # copy back, and the target card may drop it on demand (lowest priority
    # there). Transport is a bus-arbiter consumer like expert streaming and
    # KV spill; the target space comes from the peer-budget grant. Planned by
    # ``KvPressureLadder.on_pressure_boundary``, never by the phase or
    # admission planners.
    "kv_shadow",
    # #466/#488: in-process audio modules of the live-translator tenant -- the
    # TTS talker trunk, its residual code predictor, the 12 Hz codec decoder
    # and the speaker encoder. They are ordinary resident weight blocks with
    # no captured addresses, so the plain tensor route applies. They are
    # registered here rather than tracked privately because the user order of
    # 2026-08-03 is that EVERY asset lives under one runtime and one ledger:
    # a component whose VRAM the register cannot see makes every cross-asset
    # arbitration decision silently wrong.
    "audio_modules",
)

OFFLOAD_POLICIES = ("resident", "ram", "auto")

# Erg. 7c park-target tier ladder: own_vram (tier 0 = resident, NOT a park
# target) -> peer_vram (P2P one-hop) -> host_ram -> remote (#224 stub). The
# constants live HERE (not in offload_movement) so the policy parser can name
# them without importing the movement half.
OWN_VRAM_TIER = "own_vram"
PARK_TARGETS = ("peer_vram", "host_ram", "remote")

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

    ``targets`` (Erg. 7c tier ladder) overrides the global park-target
    preference order (``--lane-offload-park-targets``) for this class; None
    = use the global order. Syntax on the flag:
    ``class=mode[:fraction][@target1>target2]``.
    """

    mode: str
    fraction: Optional[float]
    targets: Optional[tuple] = None

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
        if self.targets is not None:
            for target in self.targets:
                if target not in PARK_TARGETS:
                    raise ValueError(
                        f"unknown park target {target!r}; known targets: "
                        f"{', '.join(PARK_TARGETS)} ({OWN_VRAM_TIER!r} is "
                        f"tier 0 = stay resident, not a park target)."
                    )


def _default_fraction(mode: str) -> Optional[float]:
    # resident = park nothing; ram = full depth; auto = Messpflicht
    # placeholder (measured depth in the GPU phase).
    return {"resident": 0.0, "ram": 1.0, "auto": None}[mode]


def parse_park_target_order(
    spec: Optional[str], flag: str, default: tuple = ("host_ram",)
) -> tuple:
    """Parse a ``target1>target2`` park-target preference chain (Erg. 7c
    tier ladder), e.g. ``peer_vram>host_ram``. Hard errors on
    unknown/duplicate targets and on ``own_vram`` (tier 0 = 'stay resident'
    is not a park destination)."""
    if spec is None or not spec.strip():
        return default
    order = []
    for part in spec.split(">"):
        part = part.strip()
        if not part:
            continue
        if part == OWN_VRAM_TIER:
            raise ValueError(
                f"{flag}: {OWN_VRAM_TIER!r} is tier 0 of the ladder (= stay "
                f"resident) and cannot be a park target; parkable targets: "
                f"{', '.join(PARK_TARGETS)}."
            )
        if part not in PARK_TARGETS:
            raise ValueError(
                f"{flag} names unknown park target {part!r}; known targets: "
                f"{', '.join(PARK_TARGETS)}."
            )
        if part in order:
            raise ValueError(f"{flag} lists park target {part!r} twice.")
        order.append(part)
    if not order:
        return default
    return tuple(order)


def _parse_one_policy(klass: str, spec: str, flag: str) -> ClassPolicy:
    # Optional per-class park-target suffix: mode[:fraction][@t1>t2].
    spec, at_sep, target_spec = spec.partition("@")
    targets = None
    if at_sep:
        if not target_spec.strip():
            raise ValueError(
                f"{flag} entry {klass}={spec!r}@: empty park-target list; "
                f"expected @<target>[><target>...] with targets from "
                f"{', '.join(PARK_TARGETS)}."
            )
        targets = parse_park_target_order(target_spec, f"{flag} entry {klass!r}")
    mode, sep, frac_str = spec.partition(":")
    mode = mode.strip()
    if mode not in OFFLOAD_POLICIES:
        raise ValueError(
            f"{flag} sets class {klass!r} to unknown policy {mode!r}; "
            f"known policies: {', '.join(OFFLOAD_POLICIES)}."
        )
    if not sep:
        return ClassPolicy(mode, _default_fraction(mode), targets)
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
    return ClassPolicy(mode, fraction, targets)


def parse_class_policy_overrides(spec: Optional[str]) -> Dict[str, ClassPolicy]:
    """Parse ``--lane-offload-class-policy`` (e.g. ``"drafter_heads=
    ram:0.5@peer_vram>host_ram,graph_rungs=auto"``; entry syntax
    ``<class>=<policy>[:<fraction>][@<target>[><target>...]]``, the
    @-suffix being the per-class park-target override of the Erg.-7c tier
    ladder) into ``{class: ClassPolicy}``. Hard errors on unknown
    classes/policies/targets and malformed entries -- this runs at argument
    time so a typo fails the boot, not the first park."""
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
                f"<class>=<policy>[:<fraction>][@<target>[><target>...]] "
                f"(classes: {', '.join(OFFLOAD_CLASSES)}; policies: "
                f"{', '.join(OFFLOAD_POLICIES)}; targets: "
                f"{', '.join(PARK_TARGETS)})."
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

    ``size_source`` is the item's LIVE size source (see offload_sizes):
    the object -- or zero-arg callable -- the registered bytes were resolved
    from. Kept so ``refresh_sizes()`` can re-resolve once the real
    allocations exist (e.g. after graph capture); on GPU the true figure
    then appears without any adapter change.
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
    size_source: Optional[object] = None


class MovementBackend:
    """Interface of the movement half. The register decides WHETHER an item
    may move (policy + invariants); the backend does the moving. The real
    backend is ``offload_movement.RealMovementBackend`` (pinned pools + H2D
    behind compute, VMM tag pools #93 for ``va_stable_required`` items, #89
    suspend for ``cold_lane``, park-target ladder Erg. 7c); the CPU fake
    below keeps the register hermetically testable.

    ``target`` names a park destination from ``PARK_TARGETS``; None lets the
    backend's target policy walk the tier ladder.
    """

    def park(self, item: OffloadItem, target: Optional[str] = None) -> None:
        raise NotImplementedError

    def wave_in(self, item: OffloadItem) -> None:
        raise NotImplementedError

    def bind(self, item_id: str, payload: object, source_device: int = 0) -> None:
        """Attach an item's movement payload (which route moves it). The
        base interface accepts and ignores it so adapters may bind
        unconditionally; the real backend stores it."""


class CpuFakeMovementBackend(MovementBackend):
    """Records movements for hermetic CPU tests. No tensors, no CUDA."""

    def __init__(self):
        self.parked_ids: List[str] = []
        self.parked_targets: List[Optional[str]] = []
        self.waved_in_ids: List[str] = []
        self.bound: Dict[str, object] = {}

    def park(self, item: OffloadItem, target: Optional[str] = None) -> None:
        self.parked_ids.append(item.item_id)
        self.parked_targets.append(target)

    def wave_in(self, item: OffloadItem) -> None:
        self.waved_in_ids.append(item.item_id)

    def bind(self, item_id: str, payload: object, source_device: int = 0) -> None:
        self.bound[item_id] = payload


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
class AdmissionBoundaryPlan:
    """What the session-set ladder WOULD move at one admission boundary
    (Erg. 8).

    Same pattern as ``PhaseBoundaryPlan``: in the CPU phase
    ``on_admission_boundary`` plans and moves NOTHING; the GPU phase executes
    ``park_candidates`` / ``wave_in_candidates`` through the movement
    backend. Deterministic: parks pick the highest set ids first, wave-ins
    the lowest parked ids first; ``skipped`` maps every considered-but-
    rejected item to its reason.

    ``needed_sets`` is ``running_sessions + incoming`` (what correctness
    requires resident); ``target_sets`` is the ladder's answer (>=
    needed_sets, capped at the registered set count). The invariant "an
    arriving session never meets a parked set" is enforced here in code:
    wave-ins are unconditional and sized so that after the plan at least
    ``min(needed_sets, total sets)`` sets are resident.
    """

    running_sessions: int
    incoming: int
    needed_sets: int = 0
    target_sets: int = 0
    park_candidates: List[str] = field(default_factory=list)
    wave_in_candidates: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)


@dataclass
class OffloadRegisterStats:
    """Accounting the CPU-phase adapters feed (size/access bookkeeping only)."""

    registered: int = 0
    parks: int = 0
    park_refusals: int = 0
    wave_ins: int = 0
    phase_plans: int = 0
    admission_plans: int = 0


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
        park_target_order: Optional[Sequence[str]] = None,
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
        # PCIe bus as a budgeted item (Erg. 7c): when attached, the stage-2
        # planner asks the arbiter IN ADDITION to the overlap cost function.
        self._bus_arbiter = None
        self._bus_consumer: str = "stage2_phase"
        # Erg. 8: the session-set ladder governing the gdn_state_sets class.
        # None = ladder off (today's behavior: every set stays resident and
        # on_admission_boundary plans nothing).
        self._session_ladder = None
        # Erg. 7c park-target ladder (--lane-offload-park-targets). Held here
        # because the register is the one object that exists at runner init
        # and is reachable from every adapter; the MovementEngine that
        # consumes it is built later, in the GPU phase, and reads it from
        # ``park_target_order``. None = the movement layer's own default.
        if park_target_order is not None:
            for target in park_target_order:
                if target == OWN_VRAM_TIER:
                    raise ValueError(
                        f"{OWN_VRAM_TIER!r} is tier 0 of the ladder (= stay "
                        f"resident) and cannot be a park target"
                    )
                if target not in PARK_TARGETS:
                    raise ValueError(
                        f"unknown park target {target!r}; known targets: "
                        f"{', '.join(PARK_TARGETS)}."
                    )
            park_target_order = tuple(park_target_order)
        self._park_target_order: Optional[tuple] = park_target_order
        self.stats = OffloadRegisterStats()

    # --- configuration ------------------------------------------------------
    @property
    def policies(self) -> Dict[str, ClassPolicy]:
        return dict(self._policies)

    @property
    def backend(self) -> MovementBackend:
        """The movement backend (adapters use this to bind payloads)."""
        return self._backend

    @property
    def park_target_order(self) -> Optional[tuple]:
        """The operator's ``--lane-offload-park-targets`` chain, or None when
        none was given (the movement layer then uses its own default)."""
        return self._park_target_order

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

    def set_session_ladder(self, ladder) -> None:
        """Attach the Erg.-8 session-set ladder
        (``offload_gdn_states.SessionSetLadder``) that plans the
        ``gdn_state_sets`` class at admission boundaries. ``None`` detaches
        (= ladder off, every set stays resident)."""
        with self._lock:
            self._session_ladder = ladder

    @property
    def session_ladder(self):
        return self._session_ladder

    def set_bus_arbiter(self, arbiter, consumer: str = "stage2_phase"):
        """Attach the PCIe bus-budget arbiter (offload_bus_budget). The
        stage-2 planner then announces each candidate's bytes to the arbiter
        as consumer ``consumer`` and drops candidates the bus budget does
        not admit -- the bus is itself a budgeted item (Erg. 7c)."""
        with self._lock:
            self._bus_arbiter = arbiter
            self._bus_consumer = consumer

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
        size_source: Optional[object] = None,
    ) -> OffloadItem:
        """Register one VRAM item. Duplicate ids and unknown classes/tiers
        are hard errors (an adapter registering twice is a lifecycle bug, not
        a situation to paper over).

        ``size_source`` (offload_sizes vocabulary: tensor / module / state
        record / zero-arg callable / ...) supersedes a zero ``size_bytes``
        and is kept on the item so ``refresh_sizes()`` can re-resolve once
        the real allocation exists -- on GPU the true figure then appears
        without any adapter change."""
        if size_source is not None and not size_bytes:
            from sglang.srt.model_executor.offload_sizes import (
                resolve_size_bytes,
            )

            size_bytes = resolve_size_bytes(size_source)
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
            size_source=size_source,
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

    def refresh_sizes(self) -> int:
        """Re-resolve every item registered with a ``size_source`` (adapters
        register before the real allocations exist, e.g. before graph
        capture; a later refresh -- lane worker start, post-capture hook --
        turns the 0 into the true figure). PARKED items keep their park-time
        size: their bytes ARE the parked bytes. Returns the number of items
        whose size changed. Sources are evaluated OUTSIDE the lock (they are
        adapter callables and may take locks of their own)."""
        from sglang.srt.model_executor.offload_sizes import resolve_size_bytes

        with self._lock:
            pending = [
                (item.item_id, item.size_source)
                for item in self._items.values()
                if item.size_source is not None and not item.parked
            ]
        resolved = [(iid, resolve_size_bytes(src)) for iid, src in pending]
        changed = 0
        with self._lock:
            for item_id, size in resolved:
                item = self._items.get(item_id)
                if item is None or item.parked:
                    continue
                if item.size_bytes != size:
                    item.size_bytes = int(size)
                    changed += 1
        return changed

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

    def reclaimable_bytes(self, offload_class: Optional[str] = None) -> int:
        """#553 Cut 2: bytes this class could give back RIGHT NOW.

        Resident AND not hot. Both conditions are load-bearing:

        * ``parked`` bytes are already gone -- counting them would promise the
          same bytes twice;
        * ``hot()`` items are refused by ``park()`` unconditionally, so their
          bytes are resident but NOT reclaimable. Including them would hand a
          caller a figure it cannot spend, which is the silent-partial shape
          the #553 bridge exists to avoid.

        Reads under the register's own lock and from its own items, so this is
        the register answering about itself rather than a second authority
        re-deriving its accounting from outside.
        """
        with self._lock:
            total = 0
            for item in self._items.values():
                if item.parked:
                    continue
                if offload_class is not None and item.offload_class != offload_class:
                    continue
                try:
                    if item.hot():
                        continue
                except Exception:
                    # A hotness predicate that cannot answer is treated as HOT.
                    # The safe direction is refusing to reclaim, not assuming
                    # the item is free to move.
                    continue
                total += int(item.size_bytes)
            return total

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
                if self._bus_arbiter is not None:
                    # Erg. 7c: the bus is a budgeted item shared with expert
                    # streaming and KV spill -- the planner asks the arbiter,
                    # not only the per-item overlap physics.
                    grant = self._bus_arbiter.request(
                        self._bus_consumer, item.size_bytes
                    )
                    if not grant.granted:
                        plan.skipped[item_id] = f"bus budget: {grant.reason}"
                        continue
                planned_class_bytes[item.offload_class] = planned + item.size_bytes
                plan.park_candidates.append(item_id)
        return plan

    # --- admission-boundary hook (Erg. 8; plans only, moves nothing) --------
    def on_admission_boundary(
        self, running_sessions: int, incoming: int = 0
    ) -> AdmissionBoundaryPlan:
        """Plan the ``gdn_state_sets`` moves for one admission boundary.

        Called BEFORE the ``incoming`` sessions are admitted (same no-op
        pattern as ``on_phase_boundary``: the plan is returned, nothing is
        moved in the CPU phase). The attached session ladder answers with the
        target resident-set count; this method turns it into a deterministic
        park/wave-in plan over the registered set items.

        Invariant, enforced HERE in code (and again by the ladder's target
        arithmetic): an arriving session never meets a parked set. Wave-ins
        are unconditional (retrieval is never refused) and cover at least
        ``min(running + incoming, total sets)``; parking never plans below
        that line. Raising is immediate (correctness before savings); the
        ladder applies its lowering hysteresis internally.

        Deterministic: parks pick the highest set ids first (skipping hot /
        policy-refused sets with a recorded reason), wave-ins the lowest
        parked ids first.
        """
        if running_sessions < 0 or incoming < 0:
            raise ValueError("running_sessions and incoming must be >= 0")
        plan = AdmissionBoundaryPlan(running_sessions, incoming)
        with self._lock:
            self.stats.admission_plans += 1
            items = sorted(
                (
                    i
                    for i in self._items.values()
                    if i.offload_class == "gdn_state_sets"
                ),
                key=lambda i: i.item_id,
            )
            total = len(items)
            needed = running_sessions + incoming
            needed_resident = min(needed, total)
            plan.needed_sets = needed
            ladder = self._session_ladder
            if ladder is None:
                # Ladder off = today's behavior: every set stays resident.
                # Correctness half still applies: if sets are parked (e.g. a
                # manual park), wave back whatever an admission would hit.
                target = total
            else:
                target = min(int(ladder.on_admission_cycle(needed)), total)
            # The ladder must never answer below the correctness line; do not
            # trust it blindly (enforced invariant, not caller discipline).
            target = max(target, needed_resident)
            plan.target_sets = target

            resident = [i for i in items if not i.parked]
            parked = [i for i in items if i.parked]

            # Raise: wave the lowest parked ids back in, up to the target.
            shortfall = target - len(resident)
            for item in parked[: max(0, shortfall)]:
                plan.wave_in_candidates.append(item.item_id)

            # Lower: park the highest resident ids, but never below target
            # (>= needed_resident), and only sets every invariant admits.
            excess = len(resident) - target
            if excess > 0 and ladder is not None:
                now = self._clock()
                planned = 0
                planned_bytes = 0
                policy = self._policies["gdn_state_sets"]
                for item in reversed(resident):
                    if planned >= excess:
                        break
                    if policy.mode == "resident":
                        plan.skipped[item.item_id] = "class policy is 'resident'"
                        continue
                    if policy.mode == "auto":
                        sensor = self._saturation_sensor
                        if sensor is None or not sensor():
                            plan.skipped[item.item_id] = (
                                "auto policy without saturation pressure"
                            )
                            continue
                    if item.hot():
                        # HOT = the set belongs to an active session; it is
                        # never parked, whatever the ladder says.
                        plan.skipped[item.item_id] = "hot (active session)"
                        continue
                    age_s = now - item.last_access_s
                    if age_s < self._hysteresis_window_s:
                        plan.skipped[item.item_id] = (
                            f"inside the {self._hysteresis_window_s:.3f}s "
                            f"turn hysteresis window"
                        )
                        continue
                    cap = self._fraction_cap_locked("gdn_state_sets")
                    if cap is not None:
                        already = self._parked_bytes_locked("gdn_state_sets")
                        if already + planned_bytes + item.size_bytes > cap:
                            plan.skipped[item.item_id] = (
                                f"class fraction cap ({policy.fraction}) " f"exhausted"
                            )
                            continue
                    plan.park_candidates.append(item.item_id)
                    planned += 1
                    planned_bytes += item.size_bytes

            resident_after = (
                len(resident) - len(plan.park_candidates) + len(plan.wave_in_candidates)
            )
            if resident_after < needed_resident:
                raise RuntimeError(
                    f"admission-boundary plan violates the resident-set "
                    f"invariant: {resident_after} resident after the plan, "
                    f"{needed_resident} needed for {needed} sessions "
                    f"({total} sets registered) -- this is a planner bug."
                )
        return plan

    # --- movement -----------------------------------------------------------
    def park(
        self,
        item_id: str,
        make_room_for: Optional[str] = None,
        target: Optional[str] = None,
    ) -> None:
        """Park one item, if the policy and every invariant allow it.

        ``target`` names a park destination from ``PARK_TARGETS`` (Erg. 7c
        tier ladder: peer_vram -> host_ram -> remote); None lets the
        backend's target policy choose (global/per-class preference order,
        peer_vram degrading to host_ram without a usable P2P path).

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
        if target is not None and target not in PARK_TARGETS:
            raise ValueError(
                f"unknown park target {target!r}; known targets: "
                f"{', '.join(PARK_TARGETS)}."
            )
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
            self._backend.park(item, target=target)
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

    def mark_parked(self, item_id: str, parked: bool) -> None:
        """Record that a CLASS-OWNED mover has already moved an item's bytes.

        Not a second park path: this moves no byte and applies no policy. It
        exists for the ``wired=True`` classes whose route lives outside this
        module (the #466 translator's audio ledger is the first), where the
        decision and the movement both belong to the owner and only the
        ACCOUNTING belongs here.

        Why the accounting cannot be skipped: ``_parked_bytes_locked`` is what
        the class fraction cap and every planner that prices this class read.
        Without this call an idle tenant whose weights are in host RAM still
        reads as fully resident, so the one ladder #286 exists to unify would
        be arbitrating on a number that is wrong in exactly the state the park
        was performed to reach.

        Unknown ids are ignored rather than raised: an adapter that registers
        conditionally (no register configured at construction time, one
        configured later) must not turn a bookkeeping call into an outage.
        """
        with self._lock:
            item = self._items.get(item_id)
            if item is None or item.parked == bool(parked):
                return
            item.parked = bool(parked)
            item.last_access_s = self._clock()
            if parked:
                self.stats.parks += 1
            else:
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
        # True once the register was built FROM the server args, as opposed to
        # by get_global_register()'s bare-default fallback. Two runners can
        # exist in one process (#274 dual-group lanes, draft runners), and a
        # rebuild would drop every item the first runner's adapters booked --
        # so the configure step is once-per-process and says so.
        self.configured_from_server_args = False


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
    park_target_order: Optional[Sequence[str]] = None,
) -> OffloadRegister:
    """Build (or rebuild) the process-global register from the server-args
    knobs. Called once at runner init when the register is enabled."""
    policies = resolve_class_policies(profile, class_policy_overrides)
    reg = OffloadRegister(
        policies=policies,
        backend=backend,
        hysteresis_window_s=hysteresis_window_s,
        phase_hysteresis_window_s=phase_hysteresis_window_s,
        park_target_order=park_target_order,
    )
    with _HOLDER.lock:
        _HOLDER.register = reg
    return reg


def configure_global_register_from_server_args(
    server_args,
) -> Optional[OffloadRegister]:
    """Runner-init entry point: build the register the operator asked for.

    #421 F2: ``server_args._handle_lane_offload_register`` validated
    ``--lane-offload-profile`` / ``--lane-offload-class-policy`` /
    ``--lane-offload-park-targets`` and then discarded the resolved values
    ("recomputed at configure time"), but nothing in production ever reached
    configure time. ``get_global_register()``'s fallback then built a bare
    latency-profile register, so the three flags were accepted, validated,
    and silently ignored. This is the missing half.

    Contract:

    * ``None`` and no side effect when ``SGLANG_OFFLOAD_REGISTER`` is off --
      the default path stays byte-identical.
    * Once per process. A second runner in the same process (draft runner,
      #274 dual-group lane) must NOT rebuild: a rebuild drops every item the
      first runner's adapters already booked.
    * The values are re-resolved here rather than passed down from argument
      time, so a typo refuses LOUDLY at runner init too -- including on the
      direct-``ServerArgs``-construction path that never runs
      ``__post_init__``'s validator.
    """
    if not offload_register_enabled():
        return None
    with _HOLDER.lock:
        already = _HOLDER.configured_from_server_args
    if already:
        return get_global_register()

    profile = getattr(server_args, "lane_offload_profile", "latency")
    overrides = getattr(server_args, "lane_offload_class_policy", None)
    park_spec = getattr(server_args, "lane_offload_park_targets", None)
    # Raises on an unknown profile / class / policy / fraction / target. A
    # planted typo must fail the boot here, not degrade to the default preset.
    order = parse_park_target_order(park_spec, "--lane-offload-park-targets")
    reg = configure_global_register(
        profile,
        overrides,
        park_target_order=order,
    )
    with _HOLDER.lock:
        _HOLDER.configured_from_server_args = True
    logger.info(
        "[offload-register] configured from server args: profile=%s, "
        "class-policy=%s, park-targets=%s",
        profile,
        overrides or "(none)",
        ">".join(order),
    )
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
        _HOLDER.configured_from_server_args = False


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
    size_source: Optional[object] = None,
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
        size_source=size_source,
    )


def maybe_touch_item(item_id: str) -> None:
    """Adapter entry point: record an access iff the feature flag is on."""
    reg = get_global_register()
    if reg is not None:
        reg.touch(item_id)


def maybe_mark_parked(item_id: str, parked: bool) -> None:
    """Adapter entry point: record a class-owned mover's park/restore iff the
    feature flag is on. See :meth:`OffloadRegister.mark_parked`."""
    reg = get_global_register()
    if reg is not None:
        reg.mark_parked(item_id, parked)


def maybe_refresh_item_sizes() -> int:
    """Adapter entry point: re-resolve live size sources iff the feature
    flag is on (call at points where real allocations exist, e.g. lane
    worker start / after graph capture). Returns the changed-item count."""
    reg = get_global_register()
    if reg is None:
        return 0
    return reg.refresh_sizes()


def maybe_bind_movement_payload(
    item_id: str, payload: object, source_device: int = 0
) -> None:
    """Adapter entry point: bind an item's movement payload (offload_movement
    TensorPayload / TagPayload / SuspendPayload) iff the feature flag is on.
    On the CPU fake backend this is recording only; the real backend uses it
    to route the actual movement."""
    reg = get_global_register()
    if reg is not None:
        reg.backend.bind(item_id, payload, source_device)
