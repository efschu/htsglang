# SPDX-License-Identifier: Apache-2.0
"""Size- and load-aware path dispatcher for barlink (#279) -- CPU skeleton.

Per transfer, pick the best path (named transport/route) for its (message
class, size, lane, priority), and when that path is saturated, overflow to
the next-best one. This module is the COMPLETE decision skeleton with
INJECTABLE rate tables; the real tables arrive later from three sources
(see ``barlink_path_rates``): the #278 GDR matrix, the pending NCCL/system-RAM
reference measurement, and the ``scripts/p2p_readiness/`` probe output.

Hard rules, each enforced at exactly one place in this file:

1. PLACEHOLDER NEUTRALITY (``_compute``): as long as any candidate path for
   a message class carries placeholder rates -- or a class has no candidates
   at all -- the dispatcher returns the STATUS QUO, i.e. the caller keeps
   the choice the existing #240 class wiring (--collective-net-small/-bulk)
   already made. Cost models, saturation and latency terms are consulted
   ONLY after every candidate is measured, so placeholder numbers can never
   flip a decision away from today's behaviour.
2. GRAPH SAFETY (``decide`` / ``round_boundary`` / ``begin_capture``):
   decisions are latched per (lane, class, priority, size band) and change
   only at a round boundary; ``round_boundary()`` during an active capture
   is refused (logged, latches kept), so a captured region can never replay
   a different path than it recorded. ``end_capture()`` is itself a
   boundary. The same flip-only-at-boundaries contract as the adaptive-K
   ladder.
3. PRIORITY PROTECTION (``_compute``): a protected request always receives
   the best path; only unprotected requests overflow on saturation. A
   protected class is therefore never displaced to a slower path to make
   room for another class.
4. PARKED OFFLOAD ITEMS ARE A LATENCY TERM (``_cost``): a path whose
   ``offload_class`` has parked items pays their wave-in latency in the
   cost comparison. The named interface is
   ``OffloadRegister.latency_term_ms`` -- attach it via
   ``set_offload_latency_term(register.latency_term_ms)``.

Saturation is an INJECTABLE, group-wide signal (``set_saturation_sensor``,
the same 13e signal and the same placeholder-hook pattern as
``OffloadRegister.set_saturation_sensor``): the sensor MUST return the same
answer on every rank of the group -- it is consulted only inside boundary
recomputation, so with a group-uniform sensor and group-uniform tables all
ranks flip together. No sensor attached = nothing saturated (the safe
direction, mirroring the offload register's ``auto``-without-sensor rule).

The PCIe bus is a budgeted item (#286, ``offload_bus_budget``). Dispatcher
and bus arbiter share the BUS NOTION through the named adapter
``bus_saturation_sensor(arbiter, path_to_consumer)`` -- an arbiter-backed
sensor, not a merge of the two mechanisms.

Multi-group runtime: every latch key carries the ``lane`` string, so the
dual-group lanes dispatch independently without a shared mutable choice.

Wiring: ``SGLANG_BARLINK_PATH_DISPATCHER=1`` (default off) makes
``BarlinkCommunicator`` build a dispatcher and consult it in ``_select`` via
a thin hook; with the empty registry every decision is status quo, so the
default path -- and today even the enabled path -- stays byte-identical.

Pure Python, CPU-hermetic. Tests:
test/registered/unit/distributed/test_barlink_path_dispatcher.py.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROVENANCE_MEASURED = "measured"
PROVENANCE_PLACEHOLDER = "placeholder"

# Sentinel path name meaning "keep the caller's existing (#240) choice".
STATUS_QUO = "status_quo"

# transport_hint values the barlink _select hook can act on.
HINT_TRANSPORT = "transport"  # use the configured transport object
HINT_GLOO = "gloo"  # use the inline gloo data plane


@dataclass
class RatePoint:
    """One measured (size, duration) sample of a path."""

    size_bytes: int
    ms: float


@dataclass
class PathProfile:
    """One named transport/route with an affine cost model.

    ``ms(bytes) = base_ms + per_byte_ms * bytes``, fitted from measured
    points. ``provenance`` labels every entry: only ``measured`` profiles
    ever take part in a cost comparison (hard rule 1); ``placeholder`` is
    the honest default until one of the three sources fills the entry.
    """

    name: str
    provenance: str = PROVENANCE_PLACEHOLDER
    points: List[RatePoint] = field(default_factory=list)
    base_ms: float = 0.0
    per_byte_ms: float = 0.0
    # Saturation limit of the path (bytes/s); informational for the skeleton,
    # the saturation VERDICT comes from the injected group-wide sensor.
    capacity_bytes_per_s: Optional[float] = None
    # EFFECTIVE aperture (measured, e.g. the usable slice of a 256-MiB BAR1
    # window). None = no known limit. Transfers larger than the aperture are
    # ineligible for the path -- chunked traversal is a with-cards follow-up
    # and deliberately NOT presumed here.
    aperture_bytes: Optional[int] = None
    # Utilization (0..1) at and above which the path counts as saturated.
    saturation_threshold: float = 1.0
    # Actuation handle for the barlink _select hook (HINT_* above). Paths
    # without a hint can win a comparison but not (yet) be acted on there.
    transport_hint: Optional[str] = None
    # Offload class whose PARKED items this path depends on; their wave-in
    # latency (OffloadRegister.latency_term_ms) is added to the cost.
    offload_class: Optional[str] = None
    # Which loader/file filled the entry (diagnostics only).
    source: str = ""

    def fit(self) -> None:
        """Least-squares affine fit over ``points`` (clamped to >= 0)."""
        pts = self.points
        if not pts:
            return
        if len(pts) == 1:
            self.base_ms = max(0.0, pts[0].ms)
            self.per_byte_ms = 0.0
            return
        n = float(len(pts))
        sx = sum(p.size_bytes for p in pts)
        sy = sum(p.ms for p in pts)
        sxx = sum(float(p.size_bytes) ** 2 for p in pts)
        sxy = sum(float(p.size_bytes) * p.ms for p in pts)
        denom = n * sxx - sx * sx
        slope = 0.0 if denom == 0 else (n * sxy - sx * sy) / denom
        self.per_byte_ms = max(0.0, slope)
        self.base_ms = max(0.0, (sy - self.per_byte_ms * sx) / n)

    def ms(self, nbytes: int) -> float:
        return self.base_ms + self.per_byte_ms * max(0, int(nbytes))


@dataclass(frozen=True)
class DispatchRequest:
    """One transfer to place: class, size, lane, priority.

    ``priority``: lower value = more important, the ``offload_bus_budget``
    convention. ``protected`` marks the class that hard rule 3 shields.
    """

    message_class: str
    nbytes: int
    lane: str = "default"
    priority: int = 1
    protected: bool = False


@dataclass
class DispatchDecision:
    path: str  # a registered path name, or STATUS_QUO
    status_quo: bool
    overflowed: bool = False
    reason: str = ""


def _size_band(nbytes: int) -> int:
    """Log2 bucket, so one latched choice covers one size decade rather
    than one exact byte count."""
    return max(0, int(nbytes)).bit_length()


class PathDispatcher:
    """Path registry + latched, boundary-flipped dispatch decisions."""

    def __init__(self) -> None:
        self._paths: Dict[str, PathProfile] = {}
        # message class -> candidate path names, in registration order.
        self._class_paths: Dict[str, List[str]] = {}
        self._saturation_sensor: Optional[Callable[[str], float]] = None
        self._latency_term_fn: Optional[Callable[[Optional[str]], float]] = None
        self._latched: Dict[Tuple[str, str, int, int], DispatchDecision] = {}
        self._capture_active = False
        self._lock = threading.Lock()

    # -- registry ------------------------------------------------------------
    def register_path(
        self, profile: PathProfile, message_classes: Tuple[str, ...] = ("collective",)
    ) -> None:
        if profile.provenance not in (PROVENANCE_MEASURED, PROVENANCE_PLACEHOLDER):
            raise ValueError(
                f"path {profile.name!r}: unknown provenance "
                f"{profile.provenance!r} (must be "
                f"{PROVENANCE_MEASURED!r} or {PROVENANCE_PLACEHOLDER!r})"
            )
        with self._lock:
            self._paths[profile.name] = profile
            for klass in message_classes:
                names = self._class_paths.setdefault(klass, [])
                if profile.name not in names:
                    names.append(profile.name)
        if profile.provenance == PROVENANCE_PLACEHOLDER:
            logger.warning(
                "barlink path dispatcher: path %r registered with PLACEHOLDER "
                "rates (source: %s) -- its classes stay on the status-quo "
                "choice until every candidate is measured.",
                profile.name,
                profile.source or "none",
            )

    def paths(self) -> Dict[str, PathProfile]:
        with self._lock:
            return dict(self._paths)

    def transport_hint(self, path_name: str) -> Optional[str]:
        with self._lock:
            p = self._paths.get(path_name)
            return p.transport_hint if p is not None else None

    # -- injectable hooks (placeholder pattern, cf. OffloadRegister) ---------
    def set_saturation_sensor(
        self, sensor: Optional[Callable[[str], float]]
    ) -> None:
        """Attach the 13e group-wide saturation signal:
        ``sensor(path_name) -> utilization in [0, 1]``. MUST be
        group-uniform (see module docstring). None = nothing saturated."""
        with self._lock:
            self._saturation_sensor = sensor

    def set_offload_latency_term(
        self, fn: Optional[Callable[[Optional[str]], float]]
    ) -> None:
        """Attach ``OffloadRegister.latency_term_ms`` (hard rule 4): parked
        items of a path's ``offload_class`` show up as added cost."""
        with self._lock:
            self._latency_term_fn = fn

    # -- boundaries (hard rule 2) -------------------------------------------
    def round_boundary(self) -> None:
        """The ONLY point where latched decisions may change. During an
        active capture the call is refused: flipping there would let a
        captured region replay a different path than it recorded."""
        with self._lock:
            if self._capture_active:
                logger.warning(
                    "barlink path dispatcher: round_boundary() during active "
                    "capture refused -- path choices stay latched "
                    "(graph-safety contract)."
                )
                return
            self._latched.clear()

    def begin_capture(self) -> None:
        with self._lock:
            self._capture_active = True

    def end_capture(self) -> None:
        """Capture end is itself a boundary: latches drop, the next round
        recomputes outside any captured region."""
        with self._lock:
            self._capture_active = False
            self._latched.clear()

    # -- dispatch ------------------------------------------------------------
    def decide(self, request: DispatchRequest) -> DispatchDecision:
        key = (
            request.lane,
            request.message_class,
            request.priority,
            _size_band(request.nbytes),
        )
        with self._lock:
            hit = self._latched.get(key)
            if hit is not None:
                return hit
            if self._capture_active:
                # First sight of a key inside a capture: no dynamic choice
                # may be born here -- latch the status quo so every replay
                # sees the identical decision.
                decision = DispatchDecision(
                    STATUS_QUO,
                    status_quo=True,
                    reason="capture active: no new path choice inside a "
                    "captured region",
                )
            else:
                decision = self._compute_locked(request)
            self._latched[key] = decision
            return decision

    def _compute_locked(self, request: DispatchRequest) -> DispatchDecision:
        names = self._class_paths.get(request.message_class, [])
        candidates = [self._paths[n] for n in names]
        # Hard rule 1: placeholder neutrality. No candidates, or any
        # placeholder among them -> status quo, before any cost model,
        # sensor or latency term is consulted.
        if not candidates:
            return DispatchDecision(
                STATUS_QUO,
                status_quo=True,
                reason=f"no paths registered for class "
                f"{request.message_class!r} (status quo = #240 class choice)",
            )
        unmeasured = [
            p.name for p in candidates if p.provenance != PROVENANCE_MEASURED
        ]
        if unmeasured:
            return DispatchDecision(
                STATUS_QUO,
                status_quo=True,
                reason="placeholder rates present "
                f"({sorted(unmeasured)}): status quo until every candidate "
                "is measured",
            )
        eligible = [
            p
            for p in candidates
            if p.aperture_bytes is None or request.nbytes <= p.aperture_bytes
        ]
        if not eligible:
            return DispatchDecision(
                STATUS_QUO,
                status_quo=True,
                reason=f"{request.nbytes} B exceeds every candidate's "
                "effective aperture",
            )
        ranked = sorted(eligible, key=lambda p: self._cost_locked(p, request))
        best = ranked[0]
        # Hard rule 3: a protected request always gets the best path; it is
        # never displaced to a slower one to make room for another class.
        if request.protected:
            return DispatchDecision(
                best.name,
                status_quo=False,
                reason="protected class: best path, never displaced",
            )
        if self._utilization_locked(best.name) >= best.saturation_threshold:
            for alt in ranked[1:]:
                if (
                    self._utilization_locked(alt.name)
                    < alt.saturation_threshold
                ):
                    return DispatchDecision(
                        alt.name,
                        status_quo=False,
                        overflowed=True,
                        reason=f"best path {best.name!r} saturated: overflow "
                        "to next-best",
                    )
            # Everything saturated: stay on the best path; throttling is the
            # arbiter's job, not the dispatcher's.
            return DispatchDecision(
                best.name,
                status_quo=False,
                reason="all candidates saturated: best path, throttling is "
                "the bus arbiter's concern",
            )
        return DispatchDecision(best.name, status_quo=False, reason="best path")

    def _cost_locked(self, profile: PathProfile, request: DispatchRequest) -> float:
        cost = profile.ms(request.nbytes)
        # Hard rule 4: parked offload items are a latency term
        # (OffloadRegister.latency_term_ms), never an "unavailable".
        if self._latency_term_fn is not None and profile.offload_class is not None:
            cost += float(self._latency_term_fn(profile.offload_class))
        return cost

    def _utilization_locked(self, path_name: str) -> float:
        if self._saturation_sensor is None:
            return 0.0  # no sensor = unsaturated (safe direction)
        return float(self._saturation_sensor(path_name))


# ---------------------------------------------------------------------------
# named bus interface (#286): arbiter-backed saturation sensor
# ---------------------------------------------------------------------------


def bus_saturation_sensor(
    arbiter, path_to_consumer: Dict[str, str]
) -> Callable[[str], float]:
    """Shared BUS NOTION with ``offload_bus_budget.BusBudgetArbiter``.

    A path that rides the budgeted PCIe bus maps to its arbiter consumer
    name; the sensor reads the arbiter's telemetry (pending unmet demand =
    saturated) instead of inventing a second bus model. Paths without a
    mapping report 0.0 -- their saturation comes from the group-wide 13e
    signal, not from the bus. Interface named, mechanisms not merged.
    """

    def sensor(path_name: str) -> float:
        consumer = path_to_consumer.get(path_name)
        if consumer is None:
            return 0.0
        stats = arbiter.as_dict().get(consumer)
        if stats is None:
            return 0.0
        return 1.0 if stats.get("pending_demand") else 0.0

    return sensor


# ---------------------------------------------------------------------------
# flag-gated construction + the thin barlink hook
# ---------------------------------------------------------------------------


def dispatcher_enabled() -> bool:
    """Read at call time, not import time: the flag is exported during
    server-args resolution (same reasoning as _collective_net_device)."""
    return bool(int(os.environ.get("SGLANG_BARLINK_PATH_DISPATCHER", "0")))


def maybe_build_dispatcher() -> Optional[PathDispatcher]:
    """The BarlinkCommunicator entry point: None unless
    SGLANG_BARLINK_PATH_DISPATCHER=1. The freshly built dispatcher has an
    EMPTY registry, so every decision is status quo until rate tables are
    loaded (barlink_path_rates) -- flag on is byte-identical today."""
    if not dispatcher_enabled():
        return None
    logger.info(
        "barlink path dispatcher enabled (skeleton): empty registry, every "
        "decision is the status-quo #240 class choice until measured rate "
        "tables are loaded."
    )
    return PathDispatcher()


def refine_transport_choice(
    dispatcher: Optional[PathDispatcher],
    op: str,
    nbytes: int,
    chosen,
    lane: str = "default",
):
    """The thin #240 hook used by ``BarlinkCommunicator._select``.

    ``chosen`` is the transport the existing selection produced (None = the
    inline gloo plane). Status-quo decisions -- today: all of them -- return
    it unchanged. A measured decision is acted on only through the two
    transport hints; a hint that cannot be honored here (unknown, or
    "transport" while no transport handles the op) falls back to the status
    quo rather than guessing.
    """
    if dispatcher is None:
        return chosen
    decision = dispatcher.decide(
        DispatchRequest("collective", nbytes, lane=lane)
    )
    if decision.status_quo:
        return chosen
    hint = dispatcher.transport_hint(decision.path)
    if hint == HINT_GLOO:
        return None
    if hint == HINT_TRANSPORT and chosen is not None:
        return chosen
    logger.warning(
        "barlink path dispatcher: decision %r for op %r has no actionable "
        "transport hint here; keeping the status-quo choice.",
        decision.path,
        op,
    )
    return chosen
