# SPDX-License-Identifier: Apache-2.0
"""Short-term offload REGISTER -- the asset-class layer on top of #286.

This module is the SECOND half of task #286. The first half already exists and
is NOT rebuilt here:

* ``offload_register.py``  -- item registration, per-class knobs
  (``resident|ram|auto`` + depth fraction), hysteresis, hot refusal, priority
  protection, phase/admission planners, the process-global holder.
* ``offload_movement.py``  -- the movement state machine, ``TensorPayload`` /
  ``TagPayload`` (#93 pause/resume) / ``SuspendPayload`` routes, the park-target
  ladder, the peer-path capability model.
* ``offload_sizes.py``     -- live size resolution, including the
  ``footprint_bytes`` attribute of a #102 tagged capture state.
* ``offload_bus_budget.py``-- the PCIe arbiter every byte mover consults.

What was missing, and what this module adds:

1. **Asset-class descriptors.** The existing register knows class NAMES
   (``OFFLOAD_CLASSES``) but nothing about what a class IS: where it sits on
   the one global importance ladder, what its content needs from a resting
   place, whether its addresses must survive a park, and at what granularity
   it can be cut. :data:`ASSET_CLASSES` supplies that, one
   :class:`AssetClassDescriptor` per class.

2. **The global importance ladder as code.** ``DESIGN_407_memtier_registry.md``
   §8 fixes ONE eviction order for the whole fork; ``FEATURE_CATALOG.md`` §17
   makes consuming it mandatory for anything that evicts. Until now it existed
   only as prose, so every consumer was free to invent a local victim list --
   which is the failure mode §8 names explicitly. :class:`LadderRank` and
   :func:`plan_spill` are that order, executable.

3. **Partial spill.** §8's rule is that an asset which does not fit moves only
   its OVERFLOWING part, least-important-first. :func:`plan_spill` stops at the
   first item that covers the shortfall instead of moving a whole class.

4. **Graph-state FAMILIES, fully wired.** ``graph_rungs`` items exist today for
   the speculative K-ladder (``dual_group_lane.py``) and the KV pressure
   ladder's steps (``kv_pressure_ladder.py``). Neither is the asset
   ``DESIGN_363_regime_controller.md`` §20.3 needs: a whole LAYOUT/ALGO family's
   capture state, pre-captured at boot and parked while its family is inactive.
   :class:`GraphFamilyRegister` is that asset class, moved over the existing
   #93 machinery in ``speculative/adaptive_graph_memory.py``.

5. **A refusal under active capture.** #452 refuted fetch-INSIDE-graph
   (``layers/moe/offload_capture_gate.CapturableOffloadRefuted``). The surviving
   pattern is eager work into slots BETWEEN replays. This register moves bytes
   strictly between replays and says so by refusing, by name, when asked while
   a capture is active -- the precedent
   ``expert_heat_migration.refuse_heat_migration_under_graph_capture`` sets for
   its own layout changes.

6. **Tier targets priced from the #407 registry.** ``PARK_TARGETS`` in
   ``offload_register.py`` is a hand-written list of three strings.
   ``DESIGN_407_memtier_registry.md`` §5 cut 4 names that list as the donor to
   be replaced by ``TierRegistry``. :func:`price_park_target` is that
   replacement for the classes this module owns: it asks the registry, and it
   REFUSES a tier whose bandwidth provenance is ``ABSENT``. An unmeasured path
   is never assumed usable.

7. **Classification by content STATE (#461).** A payload class describes what
   content needs from a resting place, and for GDN/Mamba state the answer is
   different for the live set (``DEVICE_BOUND``: kernels update it in place,
   and one set is a stride slice of the pool's tensors rather than a page
   range) and for the exported blob (an ordinary byte payload #364 already
   carries to a #224 tier). :class:`ContentState` is that axis; a park of such
   a class is always a VACATE-then-move. See
   ``docs/dev/DESIGN_286_short_term_register.md`` §8.

8. **VA stability acquired from the ROUTE (#468).** ``experts`` may move under
   the eager offload and may not under #462's breakable route, where a
   captured graph holds the slot arena's addresses. A graph family declares
   which classes its captures address; :func:`refuse_if_move_illegal` is the
   one gate, with two grounds. See §8b of the same note.

Scope of this slice, stated so the next reader does not over-read it:

* Graph-state families get the full wiring (descriptor, sizing, ladder rank,
  tier pricing, offload/onload, capture refusal, spill planning).
* ``drafter_heads`` / ``lane_workspaces`` / ``cold_lane`` get descriptors and
  ladder ranks -- they participate in :func:`plan_spill` -- but this module
  adds no new mover for them; they keep the routes ``offload_movement.py``
  already gives them.
* Nothing here has run on a GPU. Every claim that needs a card is listed in
  ``docs/dev/DESIGN_286_short_term_register.md`` §7 under BOOT-PENDING. In
  particular this module states NO reload-latency figure of its own: the only
  measured number in the tree for a comparable move is #102's own
  40-51 ms average / 85 ms max per state swap, which is already ~3x the
  ~25 ms projection ``DESIGN_363`` §20.3 carries as provisional.

Hermetic: no torch import at module scope, no CUDA. Tests run under
``CUDA_VISIBLE_DEVICES=99``
(``test/registered/unit/model_executor/test_short_term_offload_register.py``).
"""

from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from sglang.srt.memtier.registry import TierQuery, TierRegistry
from sglang.srt.memtier.tiers import PayloadClass, TierId
from sglang.srt.model_executor.offload_register import (
    OFFLOAD_CLASSES,
    OffloadItem,
    OffloadRefused,
    OffloadRegister,
    get_global_register,
)
from sglang.srt.planner.cost_model import Provenance, Rate

logger = logging.getLogger(__name__)

__all__ = [
    "ASSET_CLASSES",
    "GROUND_CAPTURE_ACTIVE",
    "GROUND_GRAPH_ADDRESSED",
    "AssetClassDescriptor",
    "ContentState",
    "GraphFamilyRegister",
    "GraphStateMover",
    "AdaptiveGraphStateMover",
    "FakeGraphStateMover",
    "LadderRank",
    "OffloadUnderCaptureRefused",
    "PricedTarget",
    "SpillPlan",
    "SpillStep",
    "UnpricedTierRefused",
    "capture_is_active",
    "describe_class",
    "graph_addressed_classes",
    "graph_family_item_id",
    "plan_spill",
    "price_park_target",
    "refuse_if_capture_active",
    "refuse_if_move_illegal",
    "set_capture_probe",
    "set_graph_reference_probe",
]


# ---------------------------------------------------------------------------
# 1. The one global importance ladder (DESIGN_407 §8)
# ---------------------------------------------------------------------------


class LadderRank(enum.IntEnum):
    """Position on ``DESIGN_407_memtier_registry.md`` §8's one global
    importance ladder. LOWER goes first -- rank 1 is the most disposable.

    The five rungs are §8's own, verbatim in order. They are an ``IntEnum``
    rather than strings so "least important first" is a plain sort and cannot
    drift from the doctrine's order by a re-spelling.

    Rank is an attribute of the ASSET, maintained here once for the whole
    fork. A feature that wants its assets to survive contention longer raises
    their registered rank; it does not write a second eviction policy. That
    second policy is exactly the "private victim list" failure mode §8 exists
    to close, and the reason ``FEATURE_CATALOG.md`` §17 makes this ladder a
    lookup before it is a design question.
    """

    #: §8.1 -- a cold-provisioned second model (#305 COLD/WARM rungs).
    COLD_SECOND_MODEL = 1
    #: §8.2 -- inactive layout/graph families. THIS module's class.
    INACTIVE_GRAPH_FAMILY = 2
    #: §8.3 -- cold experts, quantised per #126's spill-quant tier.
    COLD_EXPERTS = 3
    #: §8.4 -- idle sessions, #242 idle-first, #273 FCFS inviolable.
    IDLE_SESSIONS = 4
    #: §8.5 -- active work, last, and never out of FCFS order. Nothing in
    #: this module ever plans a rank-5 item; :func:`plan_spill` refuses it by
    #: name so a caller cannot reach active work by widening a budget.
    ACTIVE_WORK = 5


#: Ranks :func:`plan_spill` will never plan. §8.5 is not a soft preference:
#: active work is evicted by the scheduler's own admission path under #273
#: FCFS, never by a memory-pressure planner reaching past four other rungs.
_UNPLANNABLE_RANKS = frozenset({LadderRank.ACTIVE_WORK})


# ---------------------------------------------------------------------------
# 2. Asset-class descriptors
# ---------------------------------------------------------------------------


class ContentState(str, enum.Enum):
    """The FORM the content is in when a move is planned (#461).

    A payload class describes what the content needs from a resting place.
    For most classes that is a property of the class alone. For one it is
    not: GDN/Mamba state is device-bound while it is LIVE and a transportable
    byte payload once it has been SUSPENDED, and the two answers are not a
    contradiction but a missing axis. Every consumer that prices or plans a
    move names the state it is pricing; the default everywhere is
    :attr:`LIVE`, which is the conservative one.
    """

    #: Kernels may still write it, and a slot-addressed layout applies.
    LIVE = "live"
    #: Evacuated into a self-describing blob; nothing points at it any more.
    SUSPENDED = "suspended"


@dataclass(frozen=True)
class AssetClassDescriptor:
    """What one ``OFFLOAD_CLASSES`` member IS, as opposed to what it is called.

    ``payload`` is the ``memtier`` :class:`PayloadClass`: what the CONTENT
    needs from wherever it comes to rest. It is what makes a tier admissible
    or not (``memtier.tiers.ADMITTED_PAYLOADS`` is the whole law), so getting
    it wrong is a correctness question, not a tuning one -- each choice below
    carries its reason. It describes the LIVE form.

    ``suspended_payload`` is the payload class of the same content once it has
    been evacuated into a blob, when that differs (#461). ``None`` means the
    class has no distinct suspended form and :meth:`payload_for` answers the
    same in both states. A class that declares one can only be parked in the
    suspended form -- see :attr:`park_requires_suspend`.

    ``va_stable_required`` marks a class whose VIRTUAL addresses must survive
    the park. For a captured CUDA graph this is not an optimisation: a graph
    pins addresses, so a park that frees and re-allocates invalidates the
    capture, while a #93 unmap/remap at the same VA does not.

    ``va_stable_when_graph_addressed`` is the same requirement acquired from
    the ROUTE rather than owned by the class (#468). ``experts`` is the case:
    under the eager offload the VRAM copy is a droppable cache that may move,
    but under #462's breakable route a captured graph holds the slot arena's
    device addresses, so the same class must not move while such a graph
    exists. :meth:`va_stability_required` is the one place the two are
    combined; nothing reads the raw fields to decide whether a move is legal.

    ``dimension_presets`` is the granularity at which the class can be cut --
    the unit :func:`plan_spill` moves. Partial spill (§8) is only meaningful
    at a declared grain; a class with one indivisible item is all-or-nothing
    however the doctrine is worded, and saying so here is more honest than
    discovering it in a planner.

    ``wired`` distinguishes a class this module MOVES from one it only
    DESCRIBES. Descriptor-only classes still participate in :func:`plan_spill`
    (they have a rank and a size), but their execution goes through the routes
    ``offload_movement.py`` already owns.
    """

    offload_class: str
    ladder_rank: LadderRank
    payload: PayloadClass
    va_stable_required: bool
    time_constant_tier: str
    dimension_presets: Tuple[str, ...]
    wired: bool
    rationale: str
    suspended_payload: Optional[PayloadClass] = None
    va_stable_when_graph_addressed: bool = False

    def __post_init__(self) -> None:
        if self.offload_class not in OFFLOAD_CLASSES:
            raise ValueError(
                f"asset-class descriptor names unknown offload class "
                f"{self.offload_class!r}; known: {', '.join(OFFLOAD_CLASSES)}."
            )

    def payload_for(self, state: ContentState = ContentState.LIVE) -> PayloadClass:
        """The payload class of this content in ``state``.

        LIVE is the default because it is the stricter answer for every class
        that has two: a caller that has not evacuated the content yet must not
        be offered a tier that only the evacuated form may rest on.
        """
        if state is ContentState.SUSPENDED and self.suspended_payload is not None:
            return self.suspended_payload
        return self.payload

    @property
    def park_requires_suspend(self) -> bool:
        """True when a park of this class is a VACATE-then-move.

        The live form of such a class has no legal resting place at all, so
        "park it" is not an operation the mover can perform: the owner has to
        evacuate the content first and move the resulting blob.
        """
        return self.suspended_payload is not None

    def va_stability_required(self, *, graph_addressed: bool = False) -> bool:
        """Whether this class's VIRTUAL addresses must survive a move now.

        ONE predicate, two sources: the class's own permanent requirement, and
        the requirement a captured graph imposes on a class that would
        otherwise be free to move (#468). Callers ask this rather than reading
        ``va_stable_required``, so a route-acquired requirement cannot be
        missed by a consumer that only knows the static field.
        """
        return self.va_stable_required or (
            graph_addressed and self.va_stable_when_graph_addressed
        )


def _descriptors() -> Dict[str, AssetClassDescriptor]:
    d = (
        AssetClassDescriptor(
            offload_class="graph_rungs",
            ladder_rank=LadderRank.INACTIVE_GRAPH_FAMILY,
            payload=PayloadClass.EXPENSIVE_RECONSTRUCTABLE,
            va_stable_required=True,
            time_constant_tier="turn",
            dimension_presets=("family", "rung", "tag"),
            wired=True,
            rationale=(
                "EXPENSIVE_RECONSTRUCTABLE and not RECONSTRUCTABLE, "
                "deliberately: regenerating a parked capture state means a "
                "cold recapture, which DESIGN_363 §20.3 RUNG 2 prices at 3-6 s "
                "of visible stall. That is not the 'known, bounded cost' a "
                "droppable payload means, and declaring it droppable would let "
                "the tier table admit a RECONSTRUCTABLE_OK resting place that "
                "may lose the bytes. The whole point of the #93 route is that "
                "the content is evacuated (TagPayload cpu_backup) rather than "
                "discarded, so the graph stays valid without recapture."
            ),
        ),
        AssetClassDescriptor(
            offload_class="drafter_heads",
            ladder_rank=LadderRank.INACTIVE_GRAPH_FAMILY,
            payload=PayloadClass.EXPENSIVE_RECONSTRUCTABLE,
            va_stable_required=False,
            time_constant_tier="turn",
            dimension_presets=("head", "layer"),
            wired=False,
            rationale=(
                "The inactive speculative algo head (#309 runtime add/remove). "
                "Same rung as a graph family -- it is the weight half of the "
                "same 'this family is not running' decision -- but its "
                "addresses need not survive: a head is re-loaded into a fresh "
                "allocation, no capture points at it."
            ),
        ),
        AssetClassDescriptor(
            offload_class="lane_workspaces",
            ladder_rank=LadderRank.INACTIVE_GRAPH_FAMILY,
            payload=PayloadClass.RECONSTRUCTABLE,
            va_stable_required=False,
            time_constant_tier="phase",
            dimension_presets=("lane", "workspace"),
            wired=False,
            rationale=(
                "(lane, name)-keyed workspaces and input-buffer pools. Genuine "
                "scratch: the content after a resume is undefined anyway (see "
                "adaptive_graph_memory's Stage-1 note), so this is the one "
                "class here that is truly droppable and may rest on a "
                "RECONSTRUCTABLE_OK tier."
            ),
        ),
        AssetClassDescriptor(
            offload_class="cold_lane",
            ladder_rank=LadderRank.INACTIVE_GRAPH_FAMILY,
            payload=PayloadClass.EXPENSIVE_RECONSTRUCTABLE,
            va_stable_required=False,
            time_constant_tier="turn",
            dimension_presets=("lane",),
            wired=False,
            rationale=(
                "Everything a work-less lane owns alone, moved by the #89 "
                "suspend route (SuspendPayload). Indivisible by construction: "
                "the suspend path takes a lane's whole tag set, so its only "
                "dimension preset is the lane itself and partial spill of this "
                "class is a no-op rather than a fraction."
            ),
        ),
        AssetClassDescriptor(
            offload_class="experts",
            ladder_rank=LadderRank.COLD_EXPERTS,
            payload=PayloadClass.RECONSTRUCTABLE,
            va_stable_required=False,
            va_stable_when_graph_addressed=True,
            time_constant_tier="wave",
            dimension_presets=("expert", "layer", "half"),
            wired=False,
            rationale=(
                "The existing MoE expert spill (#77/#123/#125/#236/#242). "
                "Referenced so accounting and the ladder are unified; movement "
                "stays with the expert-offload machinery. Droppable: the host "
                "pool is the source of truth, the VRAM copy is a cache. VA "
                "stability is route-acquired rather than permanent (#468): the "
                "eager offload may move the VRAM copy freely, but #462's "
                "breakable route captures decode graphs against the slot "
                "arena's device addresses (DESIGN_462 §6), and moving the "
                "arena invalidates every graph that baked them in. The "
                "condition is 'a live captured graph family declares it "
                "addresses this class', not 'the breakable flag is set' -- the "
                "flag is not what makes the addresses baked in."
            ),
        ),
        AssetClassDescriptor(
            offload_class="gdn_state_sets",
            ladder_rank=LadderRank.IDLE_SESSIONS,
            payload=PayloadClass.DEVICE_BOUND,
            suspended_payload=PayloadClass.EXPENSIVE_RECONSTRUCTABLE,
            va_stable_required=True,
            time_constant_tier="turn",
            dimension_presets=("session_set",),
            wired=False,
            rationale=(
                "GDN/Mamba session state sets, classified per STATE (#461). "
                "LIVE it is DEVICE_BOUND per DESIGN_407 X2 -- recurrent error "
                "accumulates, so a lossy or reordered round trip is a "
                "correctness failure -- and it is not even physically "
                "parkable in place: one set is a stride slice of the pool's "
                "[num_layers, num_slots, ...] tensors, not a page range, and "
                "GDN kernels write it where it lies. Hence "
                "va_stable_required: the pool tensors' addresses never move, "
                "which is also what makes offload_movement refuse the plain "
                "tensor route for this class. SUSPENDED it is an ordinary "
                "byte payload: MambaPool.export_state_blob copies every "
                "persistent per-slot field to CPU tensors keyed by NAME and "
                "import_state_blob restores them into ANY free slot, so #364's "
                "executor already carries the blob to a #224 destination tier "
                "as one flat uint8 buffer plus a manifest. Losing a parked "
                "blob costs the session a re-prefill, so the suspended form is "
                "EXPENSIVE_RECONSTRUCTABLE and not droppable. A park of this "
                "class is therefore always vacate-then-move, never move-live. "
                "The planner that owns it is the Erg.-8 admission ladder "
                "(offload_gdn_states.SessionSetLadder), not this module."
            ),
        ),
        AssetClassDescriptor(
            offload_class="kv_shadow",
            ladder_rank=LadderRank.COLD_SECOND_MODEL,
            payload=PayloadClass.RECONSTRUCTABLE,
            va_stable_required=False,
            time_constant_tier="turn",
            dimension_presets=("rung",),
            wired=False,
            rationale=(
                "The speculative KV shadow copy of the pressure ladder "
                "(Erg. 9b). Ranked most disposable of all: the OLD layout "
                "stays the source of truth while a shadow exists, so its "
                "park/discard is free -- there is nothing to copy back. Being "
                "cheaper than a cold second model to give up, it goes first."
            ),
        ),
        AssetClassDescriptor(
            offload_class="audio_modules",
            ladder_rank=LadderRank.COLD_SECOND_MODEL,
            payload=PayloadClass.RECONSTRUCTABLE,
            va_stable_required=False,
            time_constant_tier="turn",
            dimension_presets=("module",),
            wired=True,
            rationale=(
                "#466 rung B: the live translator's in-process audio modules "
                "(TTS talker trunk, residual code predictor, 12 Hz codec "
                "decoder, speaker encoder). RECONSTRUCTABLE and not "
                "EXPENSIVE_: these are read-only weight blocks re-readable "
                "from a local checkpoint, so a discarded park costs a file "
                "read rather than a recomputation, unlike a captured graph. "
                "Ranked with the cold second model and deliberately NOT at "
                "IDLE_SESSIONS: an idle translator holding ~2 GB IS the "
                "cold-provisioned model of DESIGN_407 8.1, and ranking it "
                "above idle sessions would let a silent phone outrank a "
                "user's live conversation. va_stable_required is False "
                "because nothing captures these addresses under rung B (our "
                "own generation loop, no CUDA graph) -- the native-lane rung "
                "of #488 must revisit exactly that. Granularity is one "
                "module: the four park independently, which matters because "
                "the codec decoder alone is 229 MB and is the one a turn "
                "needs last."
            ),
        ),
    )
    return {desc.offload_class: desc for desc in d}


#: One descriptor per ``OFFLOAD_CLASSES`` member. Completeness is checked at
#: import: a class added to ``offload_register.OFFLOAD_CLASSES`` without a
#: rank and a payload class would otherwise be silently unplannable, which is
#: the same silence #421 was written to catch.
ASSET_CLASSES: Mapping[str, AssetClassDescriptor] = _descriptors()

_MISSING_DESCRIPTORS = tuple(k for k in OFFLOAD_CLASSES if k not in ASSET_CLASSES)
if _MISSING_DESCRIPTORS:  # pragma: no cover - import-time completeness guard
    raise RuntimeError(
        f"offload classes without an asset-class descriptor: "
        f"{_MISSING_DESCRIPTORS}. Every class needs a ladder rank and a "
        f"payload class, or it cannot take part in the DESIGN_407 §8 "
        f"eviction order and would be silently unplannable."
    )


def describe_class(offload_class: str) -> AssetClassDescriptor:
    """The descriptor of one class. Unknown classes are a hard error."""
    try:
        return ASSET_CLASSES[offload_class]
    except KeyError:
        raise ValueError(
            f"unknown offload class {offload_class!r}; known: "
            f"{', '.join(sorted(ASSET_CLASSES))}."
        ) from None


# ---------------------------------------------------------------------------
# 3. The capture gate (#452's surviving pattern)
# ---------------------------------------------------------------------------


#: Ground 1: a capture is RECORDING right now on this context. Class-agnostic
#: -- unmapping pages a recording capture writes into corrupts the recording.
GROUND_CAPTURE_ACTIVE = "capture_active"
#: Ground 2 (#468): a capture has ALREADY RECORDED this class's addresses.
#: Class-specific, and it bites between replays too, which is exactly where
#: ground 1 says the move would otherwise be legal.
GROUND_GRAPH_ADDRESSED = "captured_graph_addresses_class"


class OffloadUnderCaptureRefused(RuntimeError):
    """A register move a captured CUDA graph forbids.

    Named, with structured fields, for the same reason
    ``offload_capture_gate.OffloadCaptureBreach`` is: a caller that wants to
    retry between replays must be able to catch THIS and not every
    ``RuntimeError`` the move path can raise.

    ONE error with a ``ground``, not two error types, because it is one rule
    -- "a captured graph forbids this move" -- reached two ways:

    * :data:`GROUND_CAPTURE_ACTIVE` -- #452 settled the underlying question at
      boot: work inside a capture is refuted (``CapturableOffloadRefuted``),
      and the surviving pattern is eager work into slots BETWEEN replays with
      the compute captured. A park is eager work by definition (it unmaps
      physical pages), so it is legal only between replays. Retrying later
      helps.
    * :data:`GROUND_GRAPH_ADDRESSED` -- the capture is over, but it BAKED IN
      this class's device addresses, and moving them invalidates it. Retrying
      later does not help; the graphs have to be dropped first.

    A caller that catches this and retries at the next replay boundary must
    look at ``ground``: only the first one is a timing problem.
    """

    def __init__(
        self,
        operation: str,
        subject: str,
        where: str = "",
        *,
        ground: str = GROUND_CAPTURE_ACTIVE,
        reason: str = "",
    ) -> None:
        self.operation = operation
        self.subject = subject
        self.where = where
        self.ground = ground
        if not reason:
            reason = (
                "a CUDA graph capture is active on this context. The register "
                "moves physical pages, which is eager work; per #452 the only "
                "legal placement for it is BETWEEN replays, with the compute "
                "captured. Retry at the next replay boundary."
            )
        super().__init__(
            f"short-term offload register refuses {operation} of {subject!r}"
            f"{f' ({where})' if where else ''}: {reason}"
        )


#: Injectable capture probe. Production leaves it None and the real
#: ``get_is_capture_mode`` is imported lazily -- keeping torch out of this
#: module's import graph so the hermetic tests need no CUDA.
_CAPTURE_PROBE: Optional[Callable[[], bool]] = None
#: Injectable source of "which offload classes do live captured graphs
#: address". Production installs a :class:`GraphFamilyRegister`'s own
#: ``addressed_classes``; None means nothing is known to be addressed, which
#: is the pre-#462 world and leaves every class's static VA requirement in
#: force and nothing else.
_GRAPH_REFERENCE_PROBE: Optional[Callable[[], object]] = None
_PROBE_LOCK = threading.Lock()


def set_capture_probe(probe: Optional[Callable[[], bool]]) -> None:
    """Override the capture probe (tests). ``None`` restores the real one."""
    with _PROBE_LOCK:
        global _CAPTURE_PROBE  # noqa: PLW0603 - one module-level test seam
        _CAPTURE_PROBE = probe


def set_graph_reference_probe(probe: Optional[Callable[[], object]]) -> None:
    """Install the source of "which classes do captured graphs address" (#468).

    The natural production argument is
    :meth:`GraphFamilyRegister.addressed_classes` of the register that owns
    the pre-captured families, so the answer comes from the object that knows
    when a family's graphs are built and dropped rather than from a flag a
    second module has to remember to clear. ``None`` detaches.
    """
    with _PROBE_LOCK:
        global _GRAPH_REFERENCE_PROBE  # noqa: PLW0603 - one module-level seam
        _GRAPH_REFERENCE_PROBE = probe


def graph_addressed_classes() -> frozenset:
    """Offload classes whose device addresses a live captured graph holds.

    Empty when no probe is installed. Empty is the right default: it leaves
    every class's PERMANENT VA requirement untouched and adds no refusal, so a
    tree without #462 behaves exactly as before.
    """
    return frozenset().union(*_addressing_families().values() or [frozenset()])


def capture_is_active() -> bool:
    """True while a CUDA graph capture is active on this context.

    The real probe is ``runner_utils.capture_mode.get_is_capture_mode``, which
    ORs the model-capture contextvar with the breakable-graph one. Both are
    CONTEXT-local rather than process-global on purpose (#274 slice C: two
    lanes capture independently), so this answer is about the calling
    context's capture, which is exactly the one a park on this thread would
    corrupt.

    A missing probe answers False, not True: with no CUDA in the process there
    is no capture to be inside of, and refusing every move in a hermetic test
    run would make the gate untestable in the direction that matters.
    """
    with _PROBE_LOCK:
        probe = _CAPTURE_PROBE
    if probe is not None:
        return bool(probe())
    try:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )
    except Exception:  # pragma: no cover - no-torch environments
        return False
    return bool(get_is_capture_mode())


def _graph_addressed_refusal(offload_class: str) -> Optional[str]:
    """Ground-2 reason for ``offload_class``, or None (#468).

    The grain is the CLASS, because the descriptor is what carries the VA
    requirement and a family declares which CLASSES its captures address. A
    finer grain (this arena vs that arena) would need the graphs to enumerate
    the allocations they baked in, which no torch API exposes.

    The refusal is for the ACQUIRED requirement only. A class that declares VA
    stability permanently (``graph_rungs``, ``gdn_state_sets``) is not refused
    here: it has a VA-PRESERVING route precisely because it always needed one
    -- the #93 tag park -- and ``offload_movement`` already refuses every
    other route for it (``item.va_stable_required`` there). A class whose
    requirement only arrives with the route has no such mover, so for it the
    only correct answer is "do not move this at all".
    """
    desc = ASSET_CLASSES.get(offload_class)
    if desc is None or desc.va_stable_required:
        return None
    if not desc.va_stable_when_graph_addressed:
        return None
    holders = sorted(k for k, v in _addressing_families().items() if offload_class in v)
    if not holders:
        return None
    return (
        f"captured graph famil{'y' if len(holders) == 1 else 'ies'} "
        f"{', '.join(holders)} hold this class's device addresses, so its VA "
        f"stability is required on THIS route even though the class does not "
        f"require it in general (#468 / DESIGN_462 §6). Moving it would "
        f"invalidate every graph that baked the addresses in, which no replay "
        f"boundary makes safe -- drop those graph families (unregister them; a "
        f"#93 family PARK deliberately preserves their VAs and therefore keeps "
        f"the addresses live) before moving this class."
    )


def _addressing_families() -> Dict[str, frozenset]:
    """``{family_key: classes it addresses}`` from the installed probe.

    A probe that answers a flat set of class names (the documented shape) is
    reported under one synthetic key, so the refusal message can always name a
    holder instead of degrading to "something".
    """
    with _PROBE_LOCK:
        probe = _GRAPH_REFERENCE_PROBE
    if probe is None:
        return {}
    answer = probe() or ()
    if isinstance(answer, Mapping):
        return {str(k): frozenset(v) for k, v in answer.items()}
    if not isinstance(answer, Iterable):
        raise TypeError(
            f"the graph-reference probe must answer a mapping "
            f"{{family: classes}} or an iterable of class names, got "
            f"{type(answer).__name__}"
        )
    return {"<captured graphs>": frozenset(answer)}


def refuse_if_move_illegal(
    operation: str,
    subject: str,
    *,
    offload_class: Optional[str] = None,
    where: str = "",
) -> None:
    """THE park/onload gate: raise if a captured graph forbids this move.

    One predicate with the two grounds :class:`OffloadUnderCaptureRefused`
    documents, evaluated in that order because ground 1 is the unrecoverable
    one (it corrupts a recording rather than failing it) and because it holds
    for every class, named or not.

    ``offload_class`` is optional so the pre-#468 callers keep their exact
    behaviour; naming it is what buys ground 2. A caller that owns bytes of a
    known class should always name it -- the ground it unlocks is the one that
    still bites between replays, where ground 1 says the move is fine.
    """
    if capture_is_active():
        raise OffloadUnderCaptureRefused(
            operation, subject, where, ground=GROUND_CAPTURE_ACTIVE
        )
    if offload_class is None:
        return
    reason = _graph_addressed_refusal(offload_class)
    if reason is not None:
        raise OffloadUnderCaptureRefused(
            operation,
            subject,
            where,
            ground=GROUND_GRAPH_ADDRESSED,
            reason=reason,
        )


def refuse_if_capture_active(operation: str, subject: str, where: str = "") -> None:
    """Raise :class:`OffloadUnderCaptureRefused` if a capture is running.

    The class-agnostic spelling of :func:`refuse_if_move_illegal`, kept
    because it is the gate ``layers/moe/breakable_offload`` and this module's
    own family moves already call, and because a move whose class is genuinely
    unknown can still be checked against ground 1.
    """
    refuse_if_move_illegal(operation, subject, where=where)


# ---------------------------------------------------------------------------
# 4. Tier targets, priced from the #407 registry
# ---------------------------------------------------------------------------


class UnpricedTierRefused(RuntimeError):
    """No tier could be priced for this move, so no move is planned.

    Separate from :class:`OffloadRefused` on purpose. ``OffloadRefused`` means
    "the policy says stay resident" -- a normal control-flow answer. This one
    means "the registry has no measured path to anywhere that admits this
    payload", which is a configuration or measurement gap: the remedy is to run
    the probe the absence names, not to change a knob.
    """

    def __init__(self, offload_class: str, detail: str) -> None:
        self.offload_class = offload_class
        super().__init__(
            f"no priced park target for offload class {offload_class!r}: "
            f"{detail}. An unmeasured path is never assumed usable for "
            f"parking, however tempting -- run the probe the absence names."
        )


@dataclass(frozen=True)
class PricedTarget:
    """One admissible park target with the numbers behind it.

    ``move_ms`` is DERIVED, never measured: bytes over the tier's bandwidth.
    Its provenance is therefore at best ``ESTIMATE`` even when the bandwidth
    it divides is ``MEASURED``, and this module refuses to label it otherwise.
    The reason is concrete and in-tree: #102's own GPU validation timed a
    comparable state swap at 40-51 ms average and 85 ms max for ~1 GB, which
    is the remap-and-zero cost, not the link cost. A bandwidth model
    under-predicts this move, so ``move_ms`` is a FLOOR on the real figure and
    is documented as one wherever it is shown.
    """

    tier_id: TierId
    bandwidth_gbs: Rate
    headroom_bytes: Rate
    bytes_planned: int
    move_ms: Rate

    def render(self) -> str:
        return (
            f"{self.tier_id}: {self.bandwidth_gbs.value} GB/s "
            f"({self.bandwidth_gbs.provenance.value}), "
            f"{self.bytes_planned} B -> >= {self.move_ms.value:.1f} ms "
            f"({self.move_ms.provenance.value}, floor)"
        )


def price_park_target(
    registry: TierRegistry,
    *,
    offload_class: str,
    bytes_needed: int,
    origin: TierId,
    require_measured: bool = True,
    content_state: ContentState = ContentState.LIVE,
) -> PricedTarget:
    """Pick and price a park target for one class through the #407 registry.

    This is the fork's FIRST production consumer of ``memtier``. Before it, the
    catalog's "all new spill/offload consumers must pick targets from it" was
    aspirational -- #421 finding F6 pinned the absence -- and #286's own
    ``PARK_TARGETS`` triple was the hand-written list ``DESIGN_407`` §5 cut 4
    names as the donor to replace.

    ``origin`` is REQUIRED and has no default, because the registry orders
    candidates by bandwidth and the fastest tier admitting a class is almost
    always the card the bytes are already on. Parking there is not parking:
    ``offload_register`` already spells the rule out for its own ladder --
    ``own_vram`` is tier 0, "stay resident", and not a park destination. So
    the origin is dropped from the candidate list here rather than left to
    every caller to notice. Note this is a rule about the ORIGIN, not about
    device tiers: a PEER card stays a legitimate target, which is what makes
    a ``DEVICE_BOUND`` class parkable at all.

    Refusals, all of them the registry's own:

    * a tier whose volatility does not admit the class's payload never appears
      (``ADMITTED_PAYLOADS`` is the whole law -- ``DEVICE_BOUND`` reaches only
      ``DEVICE_BOUND_ONLY``, ``PERSISTENCE_REQUIRED`` only ``PERSISTENT``);
    * a tier without the headroom for ``bytes_needed`` never appears;
    * with ``require_measured`` (the default) a tier whose bandwidth is an
      ESTIMATE never appears either, and an ABSENT one never appears under any
      setting.

    ``content_state`` names the FORM the bytes are in (#461). It defaults to
    ``LIVE``, the conservative answer: a class whose live form is
    ``DEVICE_BOUND`` gets no target at all until the caller states that it has
    evacuated the content, because the evacuation is precisely what makes the
    payload transportable. Only ``gdn_state_sets`` declares two forms today;
    for every other class the argument changes nothing.

    When nothing survives, this raises :class:`UnpricedTierRefused` carrying
    the registry's full itemisation -- every refused tier with its rule and
    reason -- rather than a bare "no target".
    """
    desc = describe_class(offload_class)
    if bytes_needed < 0:
        raise ValueError(f"bytes_needed must be >= 0, got {bytes_needed}")
    selection = registry.select(
        TierQuery(
            payload=desc.payload_for(content_state),
            bytes_needed=bytes_needed,
            object_class=offload_class,
            require_measured_bandwidth=require_measured,
            origin=origin,
        )
    )
    candidates = [c for c in selection.candidates if c.tier_id != origin]
    if not candidates:
        remedy = ""
        suspended = desc.suspended_payload
        if suspended is not None and content_state is ContentState.LIVE:
            remedy = (
                f"; the LIVE form of {offload_class!r} is "
                f"{desc.payload.value} and has no resting place by design -- a "
                f"park of this class is a VACATE-then-move: evacuate the "
                f"content first and price the "
                f"{suspended.value} blob with "
                f"content_state=SUSPENDED"
            )
        raise UnpricedTierRefused(
            offload_class,
            f"origin {origin} excluded (tier 0 of the ladder = stay resident, "
            f"not a park destination); {selection.render()}{remedy}",
        )
    best = candidates[0]
    bw = best.bandwidth_gbs
    if bw.is_absent or not bw.value:
        # Defensive: the registry already refuses this. Keeping the check
        # means a future query flag that admits unmeasured bandwidth cannot
        # silently produce a division by an absence here.
        raise UnpricedTierRefused(
            offload_class,
            f"tier {best.tier_id} was admitted with bandwidth "
            f"{bw.provenance.value} ({bw.source}); this register never "
            f"divides by an unmeasured rate",
        )
    move_ms = Rate(
        value=bytes_needed / (float(bw.value) * 1e9) * 1e3,
        # ESTIMATE even off a MEASURED bandwidth: this is the LINK time, and
        # the observed move (#102: 40-51 ms avg, 85 ms max for ~1 GB) is
        # dominated by remap+zero, not by the link. Treat it as a floor.
        provenance=Provenance.ESTIMATE,
        source=(
            f"bytes / {best.tier_id} bandwidth ({bw.provenance.value}: "
            f"{bw.source}); link time only, excludes VMM remap and zeroing -- "
            f"a FLOOR, not a prediction"
        ),
        unit="ms",
        label=f"park:{offload_class}",
    )
    return PricedTarget(
        tier_id=best.tier_id,
        bandwidth_gbs=bw,
        headroom_bytes=best.headroom_bytes,
        bytes_planned=int(bytes_needed),
        move_ms=move_ms,
    )


# ---------------------------------------------------------------------------
# 5. Partial spill, least-important-first (DESIGN_407 §8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpillStep:
    """One item the plan would park, with the rank that put it there.

    ``requires_suspend`` marks a step whose park is a VACATE-then-move (#461):
    the live content has no legal resting place, so an executor must evacuate
    it into a blob and move that. It is on the STEP rather than looked up by
    the executor so a plan can be handed to a caller that does not consult the
    descriptors, and still cannot mistake it for a plain page park.
    """

    item_id: str
    offload_class: str
    ladder_rank: LadderRank
    size_bytes: int
    last_access_s: float
    requires_suspend: bool = False


@dataclass
class SpillPlan:
    """A least-important-first partial spill.

    ``steps`` is in EXECUTION order, which is also ladder order: rank
    ascending, then coldest-first inside a rank, then item id. Executing a
    prefix of it is always valid -- that is what makes the spill partial. The
    plan stops at the first step whose cumulative bytes cover the shortfall
    (§8: "only the overflowing part spills"), so a class is never emptied to
    satisfy a request a single item already covers.

    ``satisfied`` is False when the whole plannable inventory is smaller than
    the request. That is reported, never rounded up by reaching into a rank
    :data:`_UNPLANNABLE_RANKS` forbids.
    """

    bytes_needed: int
    steps: List[SpillStep] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)
    satisfied: bool = False

    @property
    def bytes_planned(self) -> int:
        return sum(s.size_bytes for s in self.steps)

    def render(self) -> str:
        head = (
            f"spill plan: {self.bytes_planned} of {self.bytes_needed} B in "
            f"{len(self.steps)} step(s), "
            f"{'satisfied' if self.satisfied else 'SHORT'}"
        )
        body = [
            f"  {i + 1}. rank {s.ladder_rank.value} "
            f"({s.ladder_rank.name}) {s.item_id} [{s.offload_class}] "
            f"{s.size_bytes} B"
            f"{' (vacate-then-move)' if s.requires_suspend else ''}"
            for i, s in enumerate(self.steps)
        ]
        return "\n".join([head, *body])


def _ladder_sort_key(item: OffloadItem) -> Tuple[int, float, str]:
    """(rank, last access, id): least important first, coldest first within a
    rank, then a stable tiebreak so a plan is reproducible."""
    rank = ASSET_CLASSES[item.offload_class].ladder_rank
    return (int(rank), float(item.last_access_s), item.item_id)


def plan_spill(
    register: OffloadRegister,
    bytes_needed: int,
    *,
    classes: Optional[Sequence[str]] = None,
    now: Optional[float] = None,
) -> SpillPlan:
    """Plan the least-important-first partial spill of ``bytes_needed``.

    Pure: it reads the register and returns a plan. Nothing moves, so a caller
    can price a plan before committing to it, and a test can assert the ORDER
    without a movement backend at all.

    The order is ``DESIGN_407`` §8's, and only §8's:

    1. ascending :class:`LadderRank` -- kv shadow, then inactive graph/layout
       families and their heads, then cold experts, then idle sessions;
    2. coldest-first inside a rank (``last_access_s`` ascending), which is the
       "coldest-first within a class" §8 spells out;
    3. item id, purely so two items of identical age plan reproducibly.

    Rank-5 (:attr:`LadderRank.ACTIVE_WORK`) is never planned: §8.5's "active
    work, last -- and never out of FCFS order" is a #273 guarantee owned by
    the scheduler's admission path, not something a pressure planner may
    override by widening a budget.

    Two class-level rules run before the per-item ones:

    * a class whose device addresses a live captured graph family holds is
      skipped entirely (#468) -- the planner may not plan what
      :func:`refuse_if_move_illegal` would refuse;
    * a class whose park is a vacate-then-move (#461) is still planned, but
      its steps carry ``requires_suspend`` so an executor cannot mistake the
      step for a plain page park.

    Already-parked and priority-protected items are skipped with a reason, as
    are items whose class policy or hysteresis would make
    :meth:`OffloadRegister.park` refuse -- but this planner does NOT
    re-implement those invariants. It records the cheap, side-effect-free ones
    (parked / protected / hot / unplannable rank) and leaves the rest to
    ``park()``, which is the single place they are enforced.
    """
    if bytes_needed < 0:
        raise ValueError(f"bytes_needed must be >= 0, got {bytes_needed}")
    plan = SpillPlan(bytes_needed=bytes_needed)
    if bytes_needed == 0:
        plan.satisfied = True
        return plan

    wanted = set(classes) if classes is not None else None
    if wanted is not None:
        unknown = sorted(wanted - set(ASSET_CLASSES))
        if unknown:
            raise ValueError(
                f"plan_spill was asked for unknown classes {unknown}; known: "
                f"{', '.join(sorted(ASSET_CLASSES))}."
            )

    candidates: List[OffloadItem] = []
    for klass in ASSET_CLASSES:
        if wanted is not None and klass not in wanted:
            continue
        candidates.extend(register.items_of_class(klass))

    running = 0
    for item in sorted(candidates, key=_ladder_sort_key):
        desc = ASSET_CLASSES[item.offload_class]
        graph_refusal = _graph_addressed_refusal(item.offload_class)
        if graph_refusal is not None:
            # #468: the planner may not plan what the gate would refuse.
            # Otherwise the rule holds only for callers who remember to ask.
            plan.skipped[item.item_id] = graph_refusal
            continue
        if desc.ladder_rank in _UNPLANNABLE_RANKS:
            plan.skipped[item.item_id] = (
                f"ladder rank {desc.ladder_rank.value} "
                f"({desc.ladder_rank.name}) is never planned: DESIGN_407 §8.5 "
                f"leaves active work to the scheduler's FCFS admission path"
            )
            continue
        if item.parked:
            plan.skipped[item.item_id] = "already parked"
            continue
        if item.prio_protected:
            plan.skipped[item.item_id] = "priority-protected"
            continue
        if item.hot():
            plan.skipped[item.item_id] = "hot (in active use)"
            continue
        if item.size_bytes <= 0:
            plan.skipped[item.item_id] = (
                "size is 0 -- an unrefreshed live source, not a free move; "
                "call OffloadRegister.refresh_sizes() once the allocation exists"
            )
            continue
        plan.steps.append(
            SpillStep(
                item_id=item.item_id,
                offload_class=item.offload_class,
                ladder_rank=desc.ladder_rank,
                size_bytes=item.size_bytes,
                last_access_s=item.last_access_s,
                requires_suspend=desc.park_requires_suspend,
            )
        )
        running += item.size_bytes
        if running >= bytes_needed:
            plan.satisfied = True
            break
    return plan


# ---------------------------------------------------------------------------
# 6. Graph-state families -- the fully wired asset class
# ---------------------------------------------------------------------------


def graph_family_item_id(family_key: str) -> str:
    """Register item id (class ``graph_rungs``) of one layout/algo family.

    Namespaced away from the two existing ``graph_rungs`` producers -- the
    per-lane spec ladder (``lane{n}/graph_rung/k{k}``) and the KV pressure
    ladder (``kv_ladder/graph_rung/{step}``) -- because all three share one
    class and one id space, and a collision would silently re-bind an item.
    """
    return f"graph_family/{family_key}"


class GraphStateMover:
    """The #93 half: unmap a family's physical pages, keep its VAs.

    Deliberately three methods wide. The register decides WHETHER a family may
    move (policy, ladder, capture gate); the mover only moves. The production
    implementation delegates to ``speculative/adaptive_graph_memory``, which
    already owns per-tag ``torch.cuda.MemPool``s, per-tag
    ``graph_pool_handle()``s and the ``pause``/``resume`` tag interception --
    this module adds no second VMM layer, for the same reason #330 did not.
    """

    def park(self, family_key: str, tag: str) -> None:
        raise NotImplementedError

    def wave_in(self, family_key: str, tag: str) -> None:
        raise NotImplementedError

    def footprint_bytes(self, family_key: str, tag: str) -> int:
        """Bytes attributable to this family's tag, or 0 for 'not known yet'.

        0 means unknown, never free: there is no driver query-by-tag API, so
        the real figure is the measured free-memory delta around ``pause``
        (``_StateRecord.paused_bytes``) and only exists after a first park.
        """
        return 0


class FakeGraphStateMover(GraphStateMover):
    """Records moves for hermetic tests. No torch, no CUDA, no tags."""

    def __init__(self, footprints: Optional[Mapping[str, int]] = None) -> None:
        self.parked: List[str] = []
        self.waved_in: List[str] = []
        self._footprints = dict(footprints or {})

    def park(self, family_key: str, tag: str) -> None:
        self.parked.append(family_key)

    def wave_in(self, family_key: str, tag: str) -> None:
        self.waved_in.append(family_key)

    def footprint_bytes(self, family_key: str, tag: str) -> int:
        return int(self._footprints.get(family_key, 0))


class AdaptiveGraphStateMover(GraphStateMover):
    """Production mover over ``AdaptiveGraphMemoryManager`` (#93 + #102).

    NEVER EXECUTED ON HARDWARE as of this slice -- desk-written, and listed as
    such in ``DESIGN_286_short_term_register.md`` §7. What is verified is the
    call surface: ``pause_after_build(state_key)``, ``ensure_active(state_key)``
    and the ``_StateRecord.footprint_bytes`` accessor all exist at the names
    used here (``adaptive_graph_memory.py``). What is NOT verified is that a
    pause/resume pair driven from THIS caller, at a replay boundary rather than
    at the manager's own ladder boundary, leaves a captured graph replayable --
    that is exactly what the GPU ticket in §7 is for.

    ``state_key`` is the manager's own rung key: an ``int`` for the classic
    k-ladder or an ``(algorithm, k)`` tuple for a cross-algorithm rung. The
    register's ``family_key`` is a string and stays a string; the mapping
    between the two is the caller's, supplied at registration.
    """

    def __init__(self, manager=None) -> None:
        self._manager = manager
        self._state_keys: Dict[str, object] = {}

    def bind_state_key(self, family_key: str, state_key) -> None:
        """Associate a register family with the manager's rung key."""
        self._state_keys[family_key] = state_key

    def _resolve_manager(self):
        if self._manager is not None:
            return self._manager
        from sglang.srt.speculative.adaptive_graph_memory import get_active_manager

        manager = get_active_manager()
        if manager is None:
            raise RuntimeError(
                "no active AdaptiveGraphMemoryManager: graph-family parking "
                "needs the #93 tagged-state machinery, which exists only when "
                "--adaptive-graph-memory selects an offload mode. Without it "
                "there are no per-tag pools to unmap and a park would free "
                "memory a captured graph still points at."
            )
        return manager

    def _state_key(self, family_key: str):
        try:
            return self._state_keys[family_key]
        except KeyError:
            raise ValueError(
                f"graph family {family_key!r} has no bound state key; call "
                f"bind_state_key() with the AdaptiveGraphMemoryManager rung "
                f"key (int, or (algorithm, k)) before moving it."
            ) from None

    def park(self, family_key: str, tag: str) -> None:  # pragma: no cover - GPU
        # State key FIRST: an unbound family is the caller's lifecycle bug and
        # must be named as one, not surfaced as whatever the manager lookup
        # happens to raise.
        state_key = self._state_key(family_key)
        self._resolve_manager().pause_after_build(state_key)

    def wave_in(self, family_key: str, tag: str) -> None:  # pragma: no cover - GPU
        state_key = self._state_key(family_key)
        self._resolve_manager().ensure_active(state_key)

    def footprint_bytes(
        self, family_key: str, tag: str
    ) -> int:  # pragma: no cover - GPU
        manager = self._resolve_manager()
        record = getattr(manager, "_states", {}).get(tag)
        if record is None:
            return 0
        return int(getattr(record, "footprint_bytes", 0) or 0)


@dataclass
class GraphFamily:
    """One pre-captured layout/algo family known to the register.

    ``addresses_classes`` are the offload classes whose DEVICE ADDRESSES this
    family's captured graphs baked in (#468). A family declares them at
    registration because that is where the capture's contents are known;
    nothing infers them from a flag. The declaration outlives a park by
    design: a #93 park preserves the family's VAs, so the graphs -- and the
    addresses they hold -- survive it. Only unregistering the family (its
    graphs are gone) drops the declaration.
    """

    family_key: str
    tag: str
    item_id: str
    active: bool = False
    addresses_classes: Tuple[str, ...] = ()


@dataclass
class GraphFamilyStats:
    registered: int = 0
    offloads: int = 0
    onloads: int = 0
    capture_refusals: int = 0
    policy_refusals: int = 0


class GraphFamilyRegister:
    """The ``graph_rungs`` asset class at LAYOUT/ALGO FAMILY granularity.

    This is what ``DESIGN_363_regime_controller.md`` §20.3 consumes. Its RUNG 1
    -- "the inactive layout's non-shared slabs are evicted through the #286
    graph-rungs/offload class BEFORE any KV admission is refused" -- calls
    :meth:`rung1_evict`, and its RUNG 0 <-> RUNG 1 transition calls
    :meth:`onload_family` on the family being flipped to.

    Composition, stated because duplicating any of it would be the bug:

    * policy, hysteresis, hot refusal, fraction caps and the parked/resident
      bookkeeping stay in :class:`OffloadRegister`. This class calls
      ``park()`` / ``wave_in()`` and lets ``OffloadRefused`` through.
    * physical movement stays in the #93 machinery behind
      :class:`GraphStateMover`. No page is unmapped here.
    * page release for the KV pool stays with #330's dial, which drives the
      same VMM arena. A family park and a dial shrink both hand pages back to
      the driver; they are sequenced by the caller, never nested, and this
      class never calls the dial.
    * victim order stays in :func:`plan_spill`, i.e. in DESIGN_407 §8.
    """

    def __init__(
        self,
        mover: Optional[GraphStateMover] = None,
        register: Optional[OffloadRegister] = None,
    ) -> None:
        self._mover = mover or FakeGraphStateMover()
        self._register = register
        self._families: Dict[str, GraphFamily] = {}
        self._lock = threading.Lock()
        self.stats = GraphFamilyStats()

    # --- wiring ------------------------------------------------------------
    @property
    def mover(self) -> GraphStateMover:
        return self._mover

    def _reg(self) -> OffloadRegister:
        reg = self._register if self._register is not None else get_global_register()
        if reg is None:
            raise RuntimeError(
                "the short-term offload register needs the #286 register, "
                "which is off: set SGLANG_OFFLOAD_REGISTER=1 or pass an "
                "explicit OffloadRegister. Graph families are not booked "
                "silently -- an unbooked family is invisible to the "
                "DESIGN_407 §8 eviction order."
            )
        return reg

    # --- registration ------------------------------------------------------
    def register_family(
        self,
        family_key: str,
        tag: str,
        *,
        size_bytes: int = 0,
        size_source: Optional[object] = None,
        restore_cost_ms: float = 0.0,
        active: bool = False,
        addresses_classes: Sequence[str] = (),
    ) -> GraphFamily:
        """Book one pre-captured family as a ``graph_rungs`` register item.

        ``size_source`` is the live source ``offload_sizes`` resolves; for a
        #102 state the natural one is the ``_StateRecord`` itself, whose
        ``footprint_bytes`` the resolver already understands. Registering
        before capture with ``size_bytes=0`` is normal -- the true figure
        appears at the first ``refresh_sizes()`` after the pause measured it.

        ``restore_cost_ms`` defaults to 0.0 and SHOULD be left there until a
        card measures it. 0.0 is honest here in the one way that matters: it
        makes the item look free to retrieve, so a planner that acts on it is
        biased AGAINST parking, which is the safe direction. Do not substitute
        DESIGN_363 §20.3's ~25 ms -- that number is a projection the tree's own
        #102 measurement (40-51 ms avg, 85 ms max) already contradicts.

        ``addresses_classes`` declares the offload classes whose device
        addresses this family's captures baked in (#468). #462's breakable
        decode route is the case that needs it: its graphs address the MoE
        expert slot arena, which makes ``experts`` unmovable for as long as
        those graphs exist. Unknown class names are a hard error at
        registration -- a typo here would silently under-protect a class.
        """
        desc = describe_class("graph_rungs")
        item_id = graph_family_item_id(family_key)
        addressed = tuple(dict.fromkeys(addresses_classes))
        for klass in addressed:
            describe_class(klass)
        with self._lock:
            if family_key in self._families:
                raise ValueError(
                    f"graph family {family_key!r} is already registered "
                    f"(tag {self._families[family_key].tag!r}); unregister "
                    f"first or fix the pre-capture lifecycle."
                )
            family = GraphFamily(
                family_key=family_key,
                tag=tag,
                item_id=item_id,
                active=active,
                addresses_classes=addressed,
            )
            self._families[family_key] = family
        self._reg().register(
            item_id,
            "graph_rungs",
            size_bytes,
            restore_cost_ms,
            hot=lambda k=family_key: self.is_active(k),
            va_stable_required=desc.va_stable_required,
            time_constant_tier=desc.time_constant_tier,
            size_source=size_source,
        )
        self.stats.registered += 1
        return family

    def unregister_family(self, family_key: str) -> None:
        with self._lock:
            family = self._families.pop(family_key, None)
        if family is not None:
            self._reg().unregister(family.item_id)

    def families(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._families))

    def addressed_classes(self) -> Dict[str, frozenset]:
        """``{family_key: classes its captured graphs address}`` (#468).

        Install this as the module's graph-reference probe
        (:func:`set_graph_reference_probe`) and the register refuses a park of
        a class a live capture holds addresses of, by name, at the same gate
        that refuses a park during an active capture -- one rule, reached from
        the object that knows when the graphs exist.
        """
        with self._lock:
            return {
                key: frozenset(family.addresses_classes)
                for key, family in self._families.items()
                if family.addresses_classes
            }

    def is_active(self, family_key: str) -> bool:
        """True for the family currently serving. The ACTIVE family is HOT and
        the base register refuses to park it whatever the policy says."""
        with self._lock:
            family = self._families.get(family_key)
            return bool(family is not None and family.active)

    def set_active(self, family_key: str) -> None:
        """Flip which family is active. Exactly one is, so this clears the
        others -- the hot criterion is a property of the flip, not something
        two callers can leave inconsistent."""
        with self._lock:
            if family_key not in self._families:
                raise ValueError(f"unknown graph family {family_key!r}")
            for key, family in self._families.items():
                family.active = key == family_key

    def _family(self, family_key: str) -> GraphFamily:
        with self._lock:
            family = self._families.get(family_key)
        if family is None:
            raise ValueError(
                f"unknown graph family {family_key!r}; registered: "
                f"{', '.join(self.families()) or '(none)'}"
            )
        return family

    # --- movement ----------------------------------------------------------
    def offload_family(self, family_key: str, target: Optional[str] = None) -> None:
        """Park one inactive family's capture state.

        Order of checks, and why:

        1. the family exists (a typo must not be a silent no-op);
        2. no capture is active -- :class:`OffloadUnderCaptureRefused`. This is
           FIRST among the runtime gates because it is the one whose violation
           is not recoverable: unmapping pages a recording capture is writing
           into corrupts the capture rather than failing it;
        3. the base register's invariants (``park()``), which refuse the active
           family as hot and honour the class policy and hysteresis;
        4. only then the physical move.

        The register's parked flag is set by ``park()`` BEFORE the mover runs,
        so a mover failure leaves the item marked parked while its pages are
        still mapped. That asymmetry is deliberate for now and named in the
        design note's open risks: making it transactional needs a real mover to
        fail against, and there is none until the GPU window.
        """
        family = self._family(family_key)
        refuse_if_capture_active("offload", family_key, where="graph family park")
        try:
            self._reg().park(family.item_id, target=target)
        except OffloadRefused:
            self.stats.policy_refusals += 1
            raise
        self._mover.park(family_key, family.tag)
        self.stats.offloads += 1
        logger.info(
            "[offload-register] parked graph family %s (tag %s)",
            family_key,
            family.tag,
        )

    def onload_family(self, family_key: str) -> None:
        """Bring a parked family's pages back at its preserved VAs.

        Never refused on policy grounds -- retrieval is not a decision, it is
        the cost the ladder already accounted for. It IS refused under an
        active capture, for the same physical reason a park is.
        """
        family = self._family(family_key)
        refuse_if_capture_active("onload", family_key, where="graph family wave-in")
        self._mover.wave_in(family_key, family.tag)
        self._reg().wave_in(family.item_id)
        self.stats.onloads += 1

    # --- the DESIGN_363 §20.3 consumption contract -------------------------
    def rung1_evict(self, bytes_needed: int, *, execute: bool = False) -> SpillPlan:
        """RUNG 1 of ``DESIGN_363`` §20.3's residency ladder.

        The contract, as the controller sees it: KV pressure crossed the #287
        mark, so the inactive layout's non-shared slabs must go BEFORE any KV
        admission is refused. The controller supplies the shortfall in bytes
        and gets back the plan; it does not pick victims, because §20.3 says
        outright that victim selection "is not a rung-1-local policy" and
        routes it to DESIGN_407 §8 -- which is :func:`plan_spill`.

        With ``execute=False`` (the default) this is pure: plan out, nothing
        moved, so the controller can price the rung before committing and the
        hermetic tests can assert the ORDER without a mover. With
        ``execute=True`` each step is parked in plan order; a step the base
        register refuses is recorded in ``skipped`` and the walk continues,
        because one item's hysteresis window is not a reason to abandon the
        eviction the rung was entered for.

        Restricted to the graph/layout classes on purpose. RUNG 1 is about the
        inactive family; reaching cold experts or idle sessions from here would
        make this a second global planner competing with §8's one, which is the
        failure mode §8 exists to close. A caller that wants the whole ladder
        calls :func:`plan_spill` directly.
        """
        plan = plan_spill(
            self._reg(),
            bytes_needed,
            classes=("kv_shadow", "graph_rungs", "drafter_heads", "lane_workspaces"),
        )
        if not execute:
            return plan
        refuse_if_capture_active(
            "rung-1 eviction", f"{bytes_needed} B", where="DESIGN_363 §20.3 RUNG 1"
        )
        executed: List[SpillStep] = []
        for step in plan.steps:
            if step.offload_class == "graph_rungs" and step.item_id.startswith(
                "graph_family/"
            ):
                key = step.item_id[len("graph_family/") :]
                try:
                    self.offload_family(key)
                except OffloadRefused as exc:
                    plan.skipped[step.item_id] = str(exc)
                    continue
                executed.append(step)
                continue
            try:
                self._reg().park(step.item_id)
            except OffloadRefused as exc:
                plan.skipped[step.item_id] = str(exc)
                continue
            executed.append(step)
        plan.steps = executed
        plan.satisfied = plan.bytes_planned >= bytes_needed
        return plan
