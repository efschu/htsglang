# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Runtime attach/detach of a speculative drafter -- the state machine (#309).

The user program is "alles auswaehlbar": a drafter should be something an
operator adds to and removes from a RUNNING server, not something a boot flag
freezes. This module is the part of that whose failure modes are silent or
corrupting, and it is therefore pure: no torch, no scheduler, no device. What
it decides is WHEN a transition is legal and WHY a refusal happened; the actual
weight load and VRAM return are executed by the scheduler at the boundary this
machine hands them (see ``docs/dev/TASK_309_RUNTIME_DRAFT.md``).

WHY A STATE MACHINE AND NOT A BOOLEAN. Three of the hard cases the task names
are the same shape -- a transition requested while the engine is in a state
that makes it unsafe:

* attach while requests are in flight,
* detach while a spec batch is mid-verify,
* attach on a model whose spec support is refused.

A boolean ``drafter_attached`` cannot express "requested but not yet safe", so
every caller would invent its own deferral, and the one that forgets rewrites a
KV pool under a replaying CUDA graph. The states below make the pending window
explicit, and :meth:`DrafterLifecycle.step` is the ONLY place a transition
happens.

THE BOUNDARY IS THE #364 ONE, reused rather than re-derived. The GDN
resident-slot executor already established where a state rewrite is safe: the
between-tick window in ``Scheduler`` where the previous batch is retired and
the next is not yet selected, so no captured graph can replay while the
executor rewrites a slot (#52/#53). Its executor REFUSES to run outside that
window, which is what makes the placement enforced rather than documented. A
drafter attach/detach is the same class of rewrite and takes the same window;
:meth:`DrafterLifecycle.step` asserts it was called with a quiesce report, so
a caller outside the boundary cannot get a transition by accident.

REFUSALS ARE NAMED, NEVER SILENT. An unsupported selection, a model that
refuses speculation, a detach during verify -- each returns a distinct error
naming the condition and what would make it legal. A runtime path that quietly
does nothing is the failure this fork keeps paying for (#340's silent env,
#212's store route, #345's silent-no-op).
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Callable, List, Optional


class DrafterState(str, enum.Enum):
    """Where the drafter is in its lifecycle.

    The two PENDING states are the point of the machine: a request that cannot
    be honoured this instant is held, visibly, until the boundary -- not
    dropped, not executed early.
    """

    #: No drafter. The server runs exactly as a spec-less boot.
    DETACHED = "detached"
    #: Attach requested and accepted; waiting for the quiesce boundary.
    ATTACH_PENDING = "attach_pending"
    #: Drafter loaded and serving.
    ATTACHED = "attached"
    #: Detach requested and accepted; waiting for the quiesce boundary.
    DETACH_PENDING = "detach_pending"

    @property
    def is_pending(self) -> bool:
        return self in (DrafterState.ATTACH_PENDING, DrafterState.DETACH_PENDING)

    @property
    def serves_drafts(self) -> bool:
        """Whether a forward may use the drafter RIGHT NOW.

        ATTACH_PENDING is False: the weights are not in yet. DETACH_PENDING is
        TRUE -- the drafter is still loaded and the in-flight verify that is
        keeping us in this state is entitled to finish against it. Getting this
        backwards is the mid-verify detach the task asks to handle loudly.
        """
        return self in (DrafterState.ATTACHED, DrafterState.DETACH_PENDING)


class DrafterTransitionError(RuntimeError):
    """Base for every refusal. Always names the condition."""


class DrafterUnsupported(DrafterTransitionError):
    """This server can never host the requested drafter.

    Distinct from a busy refusal because the answer will not change by
    retrying: it is the parse-time refusal, arriving at runtime.
    """


class DrafterBusy(DrafterTransitionError):
    """A transition is already pending. Retry after the boundary."""


class DrafterStateError(DrafterTransitionError):
    """The request does not apply in the current state (attach when attached)."""


@dataclasses.dataclass(frozen=True)
class QuiesceReport:
    """What the scheduler observed at the between-tick boundary.

    Every field is a fact the scheduler already has when it calls the executor;
    nothing here is measured by this module. Passing a report -- rather than
    letting the machine query the scheduler -- is what keeps the machine pure
    and the transition rule testable against synthetic load.
    """

    #: Requests in the running batch. Zero means the engine is idle.
    running_requests: int = 0
    #: Requests waiting to be scheduled. They have not touched the drafter yet,
    #: so they do NOT block a transition -- but they are reported so an
    #: operator can see why a boundary keeps not arriving under sustained load.
    waiting_requests: int = 0
    #: True while a speculative verify is between its draft and its target
    #: step. A detach here would free the draft KV pool under a verify that
    #: still has to read it.
    spec_verify_in_flight: bool = False
    #: True while any CUDA graph is being captured or replayed. The #364 window
    #: exists precisely because this is False there.
    graph_active: bool = False

    @property
    def is_quiesced(self) -> bool:
        """Safe to rewrite drafter-owned state.

        Running requests block: a request admitted under one drafter
        configuration must not have the drafter changed underneath it
        mid-generation. Waiting requests do not block -- they have not been
        scheduled, so they will pick up whatever configuration is live when
        they are.
        """
        return (
            self.running_requests == 0
            and not self.spec_verify_in_flight
            and not self.graph_active
        )

    def why_not_quiesced(self) -> str:
        reasons: List[str] = []
        if self.running_requests:
            reasons.append(f"{self.running_requests} request(s) still running")
        if self.spec_verify_in_flight:
            reasons.append("a speculative verify is mid-flight")
        if self.graph_active:
            reasons.append("a CUDA graph is active")
        if not reasons:
            return "quiesced"
        extra = ""
        if self.waiting_requests:
            extra = f" ({self.waiting_requests} more waiting)"
        return "; ".join(reasons) + extra


@dataclasses.dataclass(frozen=True)
class DrafterRequest:
    """An accepted attach/detach, held until the boundary."""

    action: str  # "attach" | "detach"
    #: Only for attach: what to load. Opaque here on purpose -- validating it
    #: is the selection module's job, and this machine must not grow a second
    #: opinion about what a legal drafter is.
    spec: Optional[dict] = None
    #: Correlates the eventual completion with the API call that asked.
    request_id: str = ""


@dataclasses.dataclass
class TransitionOutcome:
    """What :meth:`DrafterLifecycle.step` did at one boundary."""

    state: DrafterState
    #: The action the caller must now EXECUTE (load weights / free them), or
    #: None when the boundary passed without a transition.
    executed: Optional[str] = None
    #: Human-readable, for the log line and the status endpoint.
    detail: str = ""


class DrafterLifecycle:
    """The attach/detach state machine. Pure; one transition point.

    Usage from the scheduler's between-tick window::

        outcome = lifecycle.step(report)
        if outcome.executed == "attach":
            ...load the drafter...
        elif outcome.executed == "detach":
            ...release it...

    ``step`` never loads anything itself: a pure machine that owned a torch
    call could not be tested against synthetic load, which is exactly the
    surface where the interesting bugs are.
    """

    def __init__(
        self,
        *,
        state: DrafterState = DrafterState.DETACHED,
        supports_spec: Callable[[dict], Optional[str]] = lambda spec: None,
    ) -> None:
        """``supports_spec`` returns None when the spec is hostable, or a
        reason string when it is not -- the parse-time refusal reused as a
        runtime one, injected so this module does not import ServerArgs."""
        self._state = state
        self._supports_spec = supports_spec
        self._pending: Optional[DrafterRequest] = None
        #: Every completed transition, for the status endpoint and for tests
        #: that assert an attach did not happen twice.
        self.history: List[str] = []

    @property
    def state(self) -> DrafterState:
        return self._state

    @property
    def pending(self) -> Optional[DrafterRequest]:
        return self._pending

    @property
    def serves_drafts(self) -> bool:
        return self._state.serves_drafts

    # -- requests -----------------------------------------------------------

    def request_attach(self, spec: dict, *, request_id: str = "") -> DrafterRequest:
        """Accept an attach, or refuse by name. Does not load anything."""
        if self._pending is not None:
            raise DrafterBusy(
                f"a {self._pending.action} is already pending; the drafter "
                "changes at a between-tick boundary and only one transition "
                "is in flight at a time"
            )
        if self._state is DrafterState.ATTACHED:
            raise DrafterStateError(
                "a drafter is already attached; detach it first, or use the "
                "selection endpoint to change algorithm/k without a reload"
            )
        reason = self._supports_spec(spec)
        if reason is not None:
            # The parse-time refusal, arriving at runtime. Distinct type
            # because retrying will not help.
            raise DrafterUnsupported(
                f"this server cannot host the requested drafter: {reason}"
            )
        self._pending = DrafterRequest("attach", spec=spec, request_id=request_id)
        self._state = DrafterState.ATTACH_PENDING
        return self._pending

    def request_detach(self, *, request_id: str = "") -> DrafterRequest:
        if self._pending is not None:
            raise DrafterBusy(
                f"a {self._pending.action} is already pending; only one "
                "transition is in flight at a time"
            )
        if self._state is DrafterState.DETACHED:
            raise DrafterStateError("no drafter is attached")
        self._pending = DrafterRequest("detach", request_id=request_id)
        self._state = DrafterState.DETACH_PENDING
        return self._pending

    def cancel_pending(self) -> Optional[DrafterRequest]:
        """Withdraw a pending transition; the state reverts to where it was.

        A boundary that never arrives (sustained load) must be escapable
        without restarting the server, or the operator's only lever is a
        reboot -- which is the thing this task exists to remove.
        """
        if self._pending is None:
            return None
        req = self._pending
        self._pending = None
        self._state = (
            DrafterState.DETACHED if req.action == "attach" else DrafterState.ATTACHED
        )
        return req

    # -- the one transition point -------------------------------------------

    def step(self, report: QuiesceReport) -> TransitionOutcome:
        """Advance at the between-tick boundary. The ONLY transition point.

        Called unconditionally by the scheduler in the #364 window; a boundary
        with nothing pending is one predictable branch and no work.
        """
        if not isinstance(report, QuiesceReport):
            raise TypeError(
                "DrafterLifecycle.step needs the scheduler's QuiesceReport: "
                "the transition rule is a function of the boundary's observed "
                "state, and a caller that cannot produce one is not at the "
                "boundary (#364: the executor refuses to run outside its "
                "window)"
            )
        if self._pending is None:
            return TransitionOutcome(self._state, None, "nothing pending")
        if not report.is_quiesced:
            return TransitionOutcome(
                self._state,
                None,
                f"{self._pending.action} deferred: {report.why_not_quiesced()}",
            )
        req = self._pending
        self._pending = None
        if req.action == "attach":
            self._state = DrafterState.ATTACHED
            self.history.append("attach")
            return TransitionOutcome(self._state, "attach", "drafter attached")
        self._state = DrafterState.DETACHED
        self.history.append("detach")
        return TransitionOutcome(self._state, "detach", "drafter detached")

    def snapshot(self) -> dict:
        """What the status endpoint reports."""
        return {
            "state": self._state.value,
            "serves_drafts": self.serves_drafts,
            "pending": None if self._pending is None else self._pending.action,
            "pending_request_id": (
                None if self._pending is None else self._pending.request_id
            ),
            "transitions": list(self.history),
        }
