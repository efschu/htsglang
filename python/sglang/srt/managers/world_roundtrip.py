# SPDX-License-Identifier: Apache-2.0
"""#329 cut 2 -- the in-process round trip, with membership held FIXED.

DESIGN_329 §8 cut 2: "Quiesce + snapshot + restore with NO membership change.
Same member set, full round trip through the state machine." Its falsifier is
round-trip byte-identity of KV and GDN state, and it is worth building even if
#329 never ships -- it is the missing half of #89's disk park, which today can
park a process and restore a NEW one but cannot round-trip a LIVE one.

WHAT THIS MODULE IS AND IS NOT
------------------------------
It is the phase machine, the asset ledger and the gates -- the pure pieces.
Everything that touches a card arrives as an injected callable, exactly as
``session_handover.py`` does at SESSION scope (this is the same five-phase
vocabulary one tier up, and that module is the prior art this one follows
rather than re-invents). The split is what lets the falsifiers run hermetically:
a planted asset omission MUST fail the completeness gate, and the gate must
also be able to pass.

It is NOT wired to the scheduler, and nothing here triggers itself.

RE-FORM IS ABSENT BY CONSTRUCTION, NOT SKIPPED BY FLAG
-----------------------------------------------------
DESIGN_329 §1: NCCL communicators cannot shrink or grow in place. Cut 2's whole
point is the round trip WITHOUT that step, so this module has no RE-FORM phase
to disable -- and :meth:`WorldRoundTrip.quiesce` REFUSES when the requested
target membership differs from the live one, rather than silently doing four
fifths of a membership change. Cut 1 (measuring live communicator
teardown/rebuild) is the unmeasured step and is explicitly not this cut.

ROLLBACK, WHICH IS FREE HERE AND MUST STAY THAT WAY
---------------------------------------------------
§2: rollback exists only between QUIESCE and RE-FORM, because RE-FORM is the
only irreversible step. With no RE-FORM, EVERY phase in this cut is reversible,
so :meth:`abort` is legal at any point up to RESUME. That is a property to
assert, not a convenience to enjoy quietly: if a future edit makes some step
here destructive, the rollback guarantee silently stops holding, and the pin
below is what makes that edit fail instead.

NEVER A TRANSIENT-FAILURE REFLEX
--------------------------------
§9: "Not as a response to transient failure. This is a planned-maintenance and
capacity mechanism." A round trip costs a maintenance window; firing one at a
link hiccup is a self-inflicted outage. So every entry point here takes an
explicit :class:`Trigger` naming a human or a planner, and there is no code
path that constructs one from an error. The pin lives in the test file.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "ASSET_CLASSES",
    "AssetClass",
    "Phase",
    "Trigger",
    "WorldRoundTrip",
    "WorldRoundTripError",
    "validate_roundtrip_completeness",
]


class WorldRoundTripError(Exception):
    """Loud failure of the #329 cut-2 round trip. Never a silent degrade."""


class Phase(str, enum.Enum):
    """The phases this cut owns.

    RE-FORM is deliberately absent: see the module docstring. A reader who
    expects five members and finds four should find the reason there rather
    than conclude something was forgotten.
    """

    STABLE = "stable"
    QUIESCE = "quiesce"
    SNAPSHOT = "snapshot"
    RESTORE = "restore"
    RESUME = "resume"


#: The phases in the order they run.
PHASE_ORDER: Tuple[Phase, ...] = (
    Phase.QUIESCE,
    Phase.SNAPSHOT,
    Phase.RESTORE,
    Phase.RESUME,
)


class Trigger(str, enum.Enum):
    """Who asked for this round trip.

    There is no ``TRANSIENT_FAILURE`` member and there must never be one
    (DESIGN_329 §9). Both members name a deliberate decision: an operator, or
    the planner's cost-library verdict. An availability signal may PROMPT a
    planner verdict; it may not be the verdict.
    """

    OPERATOR = "operator"
    PLANNER = "planner"


@dataclasses.dataclass(frozen=True)
class AssetClass:
    """One kind of state that must survive the round trip.

    ``required`` marks a class whose ABSENCE from a manifest is a refusal
    rather than a note. ``re_derivable`` marks one that could be rebuilt from
    disk instead of carried -- weights are the case, and the distinction is
    DESIGN_329 §3's central asymmetry: weights are re-derivable, KV and GDN
    are not, and the payload that must actually survive is the smaller one.
    """

    name: str
    required: bool
    re_derivable: bool
    why: str


#: What constitutes a world snapshot. Each entry carries the reason it is in
#: the list, because a list of names is exactly how a class goes missing.
ASSET_CLASSES: Tuple[AssetClass, ...] = (
    AssetClass(
        name="kv_pages",
        required=True,
        re_derivable=False,
        why=(
            "the sessions being protected; re-prefilling them is the cost the "
            "round trip exists to avoid"
        ),
    ),
    AssetClass(
        name="gdn_state",
        required=True,
        re_derivable=False,
        why=(
            "#212: recurrent state is POSITIONAL and not prefix-shareable. It "
            "must travel explicitly as an opaque blob -- the radix route would "
            "silently re-prefill and the output would be wrong, not slow"
        ),
    ),
    AssetClass(
        name="session_table",
        required=True,
        re_derivable=False,
        why=(
            "the scheduler's own bookkeeping; no card is involved, and losing "
            "it orphans every KV page the other classes preserved"
        ),
    ),
    AssetClass(
        name="weights",
        required=True,
        re_derivable=True,
        why=(
            "re-derivable from local disk (#89), but still accounted: an "
            "unaccounted class is one nobody notices is missing until restore"
        ),
    ),
    AssetClass(
        name="non_persistent_buffers",
        required=True,
        re_derivable=True,
        why=(
            "buffers registered persistent=False (rotary inv_freq and its "
            "family) are absent from state_dict BY CONSTRUCTION, so any "
            "round trip built on state_dict alone silently drops them -- "
            "#568: park captured state_dict, restore did to_empty() + "
            "load_state_dict, and to_empty re-materialised every "
            "non-persistent buffer with whatever the allocator handed back. "
            "NaN cos/sin, NaN logits. The rule that fixes it is general and "
            "already exists TWICE, and this class is listed so a third "
            "implementation cannot quietly go back to state_dict: iterate "
            "named_buffers() and carry every buffer state_dict omits "
            "(translator/ledger.py, post-#568), or capture named_buffers() "
            "directly and restore in place into the live model "
            "(weight_updater._export_static_state / _import_static_state, "
            "which never had the bug). Required, and cheap: the payload is "
            "small and the failure mode is wrong output rather than a crash"
        ),
    ),
)

_REQUIRED = tuple(a.name for a in ASSET_CLASSES if a.required)


def validate_roundtrip_completeness(manifest: Dict[str, object]) -> None:
    """Refuse a manifest that does not account for every required class.

    Mirrors ``session_handover.validate_manifest_completeness`` one tier up,
    and for the same reason: the failure this catches is not a crash, it is a
    round trip that SUCCEEDS while having quietly dropped a tier, which then
    shows up as wrong output rather than as an error.

    An explicit empty entry is ACCEPTED -- "this world has no GDN state" is a
    fact, and forcing a fake blob to satisfy a gate teaches everyone to fake
    blobs. What is refused is SILENCE: the key missing altogether.
    """
    missing = [name for name in _REQUIRED if name not in manifest]
    if missing:
        raise WorldRoundTripError(
            f"round-trip manifest is incomplete: {', '.join(sorted(missing))} "
            f"unaccounted for. Every required asset class must appear, even "
            f"when empty -- an absent key is indistinguishable from a class "
            f"that was forgotten, which is the failure this gate exists for. "
            f"Classes: {', '.join(_REQUIRED)}"
        )


@dataclasses.dataclass
class _Seams:
    """Everything that touches a card or the scheduler, injected.

    Defaults are deliberately absent: a seam that silently no-ops would let
    the whole round trip "succeed" against nothing, which is the shape this
    module's tests exist to make impossible.
    """

    live_membership: Callable[[], Sequence[int]]
    in_flight_count: Callable[[], int]
    pause_admission: Callable[[str], None]
    resume_admission: Callable[[], None]
    drain_to_boundary: Callable[[float], bool]
    write_snapshot: Callable[[], Dict[str, object]]
    read_snapshot: Callable[[Dict[str, object]], None]
    #: Bring the CUDA graphs back. The seam is deliberately named for the
    #: OUTCOME and not for one mechanism, because this cut can use the cheaper
    #: of the two that exist: membership does not change, so the captured
    #: shapes are provably identical, and a VMM remap of the graph pages
    #: (``resume_memory_occupation`` / ``torch_memory_saver_adapter``) restores
    #: them without capturing anything. True recapture
    #: (``ModelRunner.init_decode_cuda_graph``) is the fallback for a caller
    #: whose shapes DID move -- which in this cut would itself be a bug.
    restore_graphs: Callable[[], None]
    park_drafter: Optional[Callable[[], None]] = None
    unpark_drafter: Optional[Callable[[], None]] = None


class WorldRoundTrip:
    """QUIESCE -> SNAPSHOT -> RESTORE -> RESUME, same process, same members.

    One instance drives one round trip. It refuses to be reused, because a
    phase machine that can silently restart mid-flight is one whose state a
    reader cannot trust from the outside.
    """

    def __init__(self, seams: _Seams, *, trigger: Trigger):
        if not isinstance(trigger, Trigger):
            raise WorldRoundTripError(
                f"trigger must be a Trigger naming who decided, got "
                f"{trigger!r}. DESIGN_329 section 9: a round trip is a "
                f"planned-maintenance mechanism and never a reflex to a "
                f"transient failure"
            )
        self._seams = seams
        self.trigger = trigger
        self.phase = Phase.STABLE
        self.manifest: Dict[str, object] = {}
        self.aborted = False
        self._membership_at_quiesce: Optional[Tuple[int, ...]] = None
        self._history: List[Phase] = []

    # -- guards ----------------------------------------------------------

    def _require_phase(self, expected: Phase, doing: str) -> None:
        if self.aborted:
            raise WorldRoundTripError(
                f"cannot {doing}: this round trip was aborted and instances "
                f"are single-use"
            )
        if self.phase is not expected:
            raise WorldRoundTripError(
                f"cannot {doing} from phase {self.phase.value!r}; "
                f"{expected.value!r} is required. Phases run "
                f"{' -> '.join(p.value for p in PHASE_ORDER)}"
            )

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self._history.append(phase)

    @property
    def history(self) -> Tuple[Phase, ...]:
        return tuple(self._history)

    # -- phases ----------------------------------------------------------

    def quiesce(
        self, *, target_membership: Optional[Sequence[int]] = None, deadline_s: float
    ) -> None:
        """Stop admitting, drain to a boundary, park the drafter.

        ``target_membership`` is accepted only to be REFUSED when it differs
        from the live set. That is not defensive clutter: this cut deliberately
        has no RE-FORM, so a caller who wants a membership change must be told
        so at the first phase, not discover after SNAPSHOT that the machine was
        never going to change anything.
        """
        self._require_phase(Phase.STABLE, "quiesce")
        live = tuple(int(x) for x in self._seams.live_membership())
        if target_membership is not None:
            want = tuple(int(x) for x in target_membership)
            if want != live:
                raise WorldRoundTripError(
                    f"membership change requested ({list(live)} -> "
                    f"{list(want)}) but this is #329 cut 2, which round-trips "
                    f"a world WITHOUT re-forming it. A different member set is "
                    f"a different communicator (DESIGN_329 section 1) and live "
                    f"teardown/rebuild is cut 1, still unmeasured. Refusing "
                    f"rather than performing four fifths of a membership change"
                )
        self._membership_at_quiesce = live

        in_flight = int(self._seams.in_flight_count())
        # Admission closes BEFORE the drain, or the queue refills behind it.
        self._seams.pause_admission(
            f"#329 round trip in progress (trigger: {self.trigger.value})"
        )
        drained = bool(self._seams.drain_to_boundary(deadline_s))
        if not drained:
            # Rollback is free here and must actually be taken, or a failed
            # quiesce leaves a world that refuses traffic forever.
            self._seams.resume_admission()
            raise WorldRoundTripError(
                f"drain did not reach a tick boundary within {deadline_s}s "
                f"({in_flight} request(s) in flight at entry); admission "
                f"re-opened and the world left STABLE. No collective may be "
                f"in flight when a round trip proceeds"
            )
        if self._seams.park_drafter is not None:
            self._seams.park_drafter()
        self._enter(Phase.QUIESCE)

    def snapshot(self) -> Dict[str, object]:
        """Serialize every asset class, then gate the result for completeness."""
        self._require_phase(Phase.QUIESCE, "snapshot")
        manifest = dict(self._seams.write_snapshot())
        validate_roundtrip_completeness(manifest)
        self.manifest = manifest
        self._enter(Phase.SNAPSHOT)
        return manifest

    def restore(self) -> None:
        """Rehydrate from the manifest into the SAME geometry.

        No reshard: the member set did not change, so the KV/GDN geometry the
        snapshot describes is the geometry it is going back into. A restore
        that needed #297 would mean membership moved, which quiesce refused.
        """
        self._require_phase(Phase.SNAPSHOT, "restore")
        live = tuple(int(x) for x in self._seams.live_membership())
        if live != self._membership_at_quiesce:
            raise WorldRoundTripError(
                f"membership changed underneath the round trip "
                f"({list(self._membership_at_quiesce or ())} -> {list(live)}). "
                f"This cut restores into the SAME geometry; a changed member "
                f"set needs RE-FORM, which does not exist here"
            )
        self._seams.read_snapshot(self.manifest)
        self._enter(Phase.RESTORE)

    def resume(self) -> None:
        """Bring graphs back, unpark the drafter, re-open admission."""
        self._require_phase(Phase.RESTORE, "resume")
        self._seams.restore_graphs()
        if self._seams.unpark_drafter is not None:
            self._seams.unpark_drafter()
        self._seams.resume_admission()
        self._enter(Phase.RESUME)

    # -- rollback --------------------------------------------------------

    def abort(self, reason: str) -> None:
        """Roll back to STABLE. Legal at every phase before RESUME.

        In the full #329 machine rollback dies at RE-FORM, because destroying
        the old communicators leaves nothing to return to. This cut never
        destroys anything, so the window is the whole round trip -- and the
        test file asserts that, so a future edit that makes a phase
        destructive has to confront the guarantee rather than quietly void it.
        """
        if self.phase is Phase.RESUME:
            raise WorldRoundTripError(
                "cannot abort after RESUME: the world is already serving "
                "again; there is nothing left to roll back"
            )
        if self._seams.unpark_drafter is not None:
            self._seams.unpark_drafter()
        self._seams.resume_admission()
        self.aborted = True
        self.phase = Phase.STABLE
        logger.warning(
            "#329 round trip ABORTED in phase %s: %s. Admission re-opened, "
            "membership unchanged, nothing was destroyed.",
            self._history[-1].value if self._history else Phase.STABLE.value,
            reason,
        )
