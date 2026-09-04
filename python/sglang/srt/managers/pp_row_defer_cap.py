# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#1180: a BOUND on the #631 row-probe's defer, counted in RING LAPS.

WHAT THE DEFER IS AND WHY IT MUST BE BOUNDED
--------------------------------------------
``SchedulerPPMixin._pp_proxy_frame_pending`` peeks the head proxy frame's
admission row without consuming it. If the row names a rid this rank cannot
locate, the probe answers False and LEAVES THE FRAME IN THE INBOX -- the
boot 631row14 protection, which exists because a frame consumed before the
request-chain hop that carries its rids produces a plan that finds nothing
and a ring that dies upstream-waiting.

That protection is correct and this module does not weaken it. What it
lacked is a bound, and a deferred frame is not free: PP0 has ALREADY
launched the pass the frame belongs to and is ALREADY parked in the receive
for its output. A downstream rank that defers the frame for ever therefore
closes a ring by construction. That is boot weg1b7 (2026-09-04 03:50:24Z)
exactly.

THE PREMISE THE DEFER RESTS ON, AND WHERE IT IS FALSE
-----------------------------------------------------
The defer site states its premise in its own comment: "a frame whose row
names an unlocatable rid PROVES a chain send is in flight ... bounded by
rendezvous+transfer". That holds for a rid whose delivery is the request
chain's job. It does NOT hold for either of the two populations measured in
weg1b7:

  * a RE-ADMITTED resident (``#968 READMIT CACHED``, ``last_queued_as=
    cutover-requeue``) is requeued rank-locally by
    ``Scheduler._add_request_to_queue``; no chain hop for it exists or ever
    will, so "wait for the chain" can never be satisfied;
  * a rid PP0 RETRACTED after sealing the row -- in the specimen PP0 shipped
    ``#631 PROXY-SEND t40`` naming rids ``772368da`` and ``c25108f3``, then
    the discretionary ``#888b`` relief actuator retracted ``c25108f3`` in the
    next scheduling iteration. The sealed row cannot be amended, so the row
    is permanently false and the peek's ``_missing`` list can never empty.

In both shapes the defer is not a latency hedge, it is a livelock. The
ranks disagree about a rid, and RAENGE-NIE-UNEINS says a detected
disagreement is a bounded, NAMED stop -- never a silent unbounded wait.

WHY LAPS AND NOT SECONDS (the #1180 review's finding, and the root of it)
------------------------------------------------------------------------
The first cut of this module counted WALL TIME between two probes. That was
wrong on two independent axes and both were measured, not argued:

  * UNREACHABLE. Every defer sets ``_pp_row_chain_owed`` (the hedge), and
    the top of the very next loop iteration turns that flag into a BLOCKING
    chain receive (``scheduler_pp_mixin`` ``_pp_row_chain_pending`` ->
    ``request_receiver.recv_requests``). A clock sampled only by the probe
    is therefore sampled only by a loop the defect halts: in weg1b7 PP1
    emitted its three defers inside ONE log second and then produced no
    further line for 60 s. A 20 s wall bound could never have been read.
  * WRONG UNIT. The legitimate gaps on this boot form are long and are not
    wedges: a cutover refill, a HiCache write-back, a cold JIT build window
    the tree deliberately stretches. Wall time cannot separate "this rank
    was busy elsewhere" from "the ring is dead"; the sibling bound at this
    seam (``pp_chain_receiver``'s ``SGLANG_PP_CHAIN_RECV_STALL_S``) says so
    in its own docstring and ships OFF by default for that reason.

THE IN-FORK EQUIVALENT, ADOPTED RATHER THAN DUPLICATED
------------------------------------------------------
``pp_admission_congruence.UNRESOLVED_DEFER_CAP`` already answers this exact
question -- "how many consecutive rounds may a rid be deferred for being
unresolvable" -- in ring laps, reasoned from ``pp_size - 1`` (a full round
trip plus one). This module IMPORTS that constant rather than inventing a
second number in a second unit: one authority, one value, one rationale.

What is NOT adopted is that mechanism's ESCAPE (PP0 pins the next offer to
``told=0`` and keeps serving) and its OWNER (PP0 counts, downstream ranks
only report on the wire). Neither is available here, and the reason is the
defect itself: the report would have to ride the request chain, and the
request chain is the closed ring. A wire-carried escalation is exactly what
the #824 ring cut already attempts and exactly what weg1b7 measured failing.
So the stop is rank-local and it is a STOP -- the same shape #1071's
occupant horizon ships for the sibling case.

WHY A REPEAT DEFER NO LONGER RE-ARMS THE HEDGE
-----------------------------------------------
The caller arms ``_pp_row_chain_owed`` only on the FIRST sighting of a
missing set (``verdict.occurrence == 1``). That keeps the boot 631row15
world exactly as it is -- there hops keep landing, so the missing set keeps
CHANGING and every defer is a first sighting -- while removing the re-arm
whose premise the previous receive already falsified: coming back to this
probe with the SAME missing set means a chain receive completed and did not
deliver the rid. Not re-arming leaves the loop cycling (the #1071 occupant
arm, 2 ms) instead of blocking, which is what makes this bound reachable
from the rank alone, with no peer required to move.

WHAT THIS MODULE IS NOT
-----------------------
It is not a recovery: it does not consume the frame, drain a wire, send a
void, or alter any rank's state. Consuming the frame at the bound would
trade this deadlock for the boot 631row14 deadlock and emitting a void would
reopen the #801 reverse corpse. Narrowing the row -- executing the frame
with only the rids this rank can locate -- is refused too: that is a
state-changing decision taken rank-locally on a group fact, and two ranks
could narrow it differently. The bound's only product is a verdict and a
message.

There is deliberately NO env knob. A second per-mechanism switch would be
the second accounting this module exists to avoid; the layer-level
discriminator ``SGLANG_PP_ROW_AUTHORITY=0`` already disarms the row
authority -- and with it this defer -- in one boot arm.

PURE BY CONSTRUCTION. No logger, no wire, no scheduler attribute, no clock
-- the caller owns the raise and the logging, exactly as
``parked_carrier_relief`` and ``min_free_slots_delayer`` are structured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

from sglang.srt.managers.pp_admission_congruence import UNRESOLVED_DEFER_CAP

__all__ = [
    "ROW_DEFER_LAP_CAP",
    "PpRowDeferCapExceeded",
    "RowDeferVerdict",
    "RowDeferCap",
]

#: How many CONSECUTIVE defers of the same (frame, missing-rid set) this rank
#: may take before the disagreement is named and raised. Deliberately THE SAME
#: object as the admission layer's cap -- see the module docstring: same
#: question, same unit, one authority. Do not fork it into a local literal.
ROW_DEFER_LAP_CAP = UNRESOLVED_DEFER_CAP


class PpRowDeferCapExceeded(RuntimeError):
    """#1180: a proxy frame's row named a rid this rank can never locate.

    Raised by the row probe when the SAME missing rid set has held the SAME
    frame in the inbox for more consecutive probes than the ring needs to
    deliver a hop. It is a STOP, not a recovery: the sender's published row
    and this rank's request population disagree, the sender is already parked
    on the output of the pass that frame belongs to, and no amount of further
    waiting can change either fact.
    """


@dataclass(frozen=True)
class RowDeferVerdict:
    """The bound's answer for one (slot, frame, missing-set) observation.

    ``defer`` True means "keep leaving the frame in the inbox" -- the shipped
    behaviour. ``defer`` False means the cap lapsed and the caller owes a
    named stop; ``message`` is then non-empty and names every term.

    ``occurrence`` counts CONSECUTIVE observations of THIS identity on THIS
    slot -- the denominator is the identity, not the scheduler. It is 1 on a
    first sighting, which is also the caller's signal that arming the chain
    hedge is still a fresh inference rather than a falsified one.
    """

    defer: bool
    occurrence: int
    first_missing: Optional[str]
    n_missing: int
    cap: int
    message: Optional[str] = None


class RowDeferCap:
    """Per-slot lap counter over the row probe's UNCHANGING missing-rid set."""

    def __init__(self) -> None:
        # slot -> (identity, consecutive observations), where the identity is
        # (frame token, missing-rid set). The TOKEN half carries the frame's
        # own stamp, which includes the pass counter, so a NEW frame for the
        # same slot -- and every frame after a cutover, which restamps -- can
        # never inherit a predecessor's count. That is why this class needs no
        # cutover hook and deliberately has none.
        self._laps: Dict[int, Tuple[Tuple[object, FrozenSet[str]], int]] = {}

    def clear(self, mb_id: int) -> None:
        """Forget a slot's count. Called when the slot has no disagreement.

        A slot whose head row is satisfied (or unreadable) has nothing
        outstanding, so its count must not survive into the next frame.

        NOT called before the raise: if any future caller ever swallows
        ``PpRowDeferCapExceeded``, the count must still be at the cap so the
        next probe of the same identity raises again immediately, rather than
        degrading the stop into a periodic raise-and-retry.
        """
        self._laps.pop(int(mb_id), None)

    def observe(
        self,
        mb_id: int,
        missing_rids: Iterable[str],
        *,
        token: object = None,
        cap: Optional[int] = None,
    ) -> RowDeferVerdict:
        """Record one defer of ``mb_id`` on ``missing_rids`` and rule on it.

        First sighting of a (slot, token, missing-set) triple always answers
        ``defer=True`` with ``occurrence == 1`` -- the ordinary hedge must
        stay free. A CHANGED missing set is progress (a hop landed) and
        restarts the count; so does a different ``token``, which the caller
        passes as the frame's own stamp. Only the same frame with the same
        missing set, seen more than ``cap`` times in a row, answers
        ``defer=False``.

        ``cap`` <= 0 disables the bound and restores the pre-#1180 behaviour
        exactly (defer for ever). It exists for the tests that must exercise
        the disabled direction; production passes nothing and gets
        ``ROW_DEFER_LAP_CAP``.
        """
        slot = int(mb_id)
        missing = frozenset(str(r) for r in missing_rids)
        identity = (_hashable(token), missing)
        bound = ROW_DEFER_LAP_CAP if cap is None else int(cap)

        prior = self._laps.get(slot)
        occurrence = prior[1] + 1 if (prior is not None and prior[0] == identity) else 1
        self._laps[slot] = (identity, occurrence)

        if bound <= 0 or occurrence <= bound:
            return RowDeferVerdict(
                defer=True,
                occurrence=occurrence,
                first_missing=_first(missing),
                n_missing=len(missing),
                cap=bound,
            )

        return RowDeferVerdict(
            defer=False,
            occurrence=occurrence,
            first_missing=_first(missing),
            n_missing=len(missing),
            cap=bound,
            message=self._message(slot, missing, occurrence, bound),
        )

    @staticmethod
    def _message(
        slot: int,
        missing: FrozenSet[str],
        occurrence: int,
        bound: int,
    ) -> str:
        named = ",".join(sorted(r[:8] for r in missing)[:4])
        return (
            f"#1180 PP ROW DEFER PAST ITS LAP CAP: slot {slot} has held the "
            f"SAME proxy frame in the inbox for {occurrence} consecutive "
            f"probes (cap {bound}) because its admission row names "
            f"{len(missing)} rid(s) this rank cannot locate, and that set has "
            f"not changed once (rids={named or 'NONE'}). A CHANGING set is a "
            f"chain hop landing; an unchanging one held past a full ring "
            f"round trip is not a slow hop, it is the sender and this rank "
            f"disagreeing about who holds the rid. THE TWO MEASURED SHAPES: "
            f"the sender RETRACTED a rid after sealing the row (weg1b7 "
            f"03:49:20, #888b relief on c25108f3 one iteration after #631 "
            f"PROXY-SEND t40 named it), or the row names a re-admitted "
            f"resident that is requeued rank-locally and traverses no chain "
            f"at all (#968 READMIT CACHED, last_queued_as=cutover-requeue). "
            f"Neither can ever be cured by waiting, and the sender is already "
            f"parked on the output of the pass this frame belongs to -- so "
            f"deferring further closes the ring instead of surviving it. "
            f"COUNTED IN LAPS, NOT SECONDS, on purpose: this rank stops "
            f"cycling the chain hedge after the first sighting, so the count "
            f"advances from this rank's own loop and cannot be starved by the "
            f"peer that is not sending. Deliberately NOT repaired here: "
            f"consuming the frame would revive boot 631row14 (plan finds "
            f"nothing, ring dies upstream-waiting), a void would revive the "
            f"#801 reverse corpse, and narrowing the row to the locatable "
            f"rids would be a rank-local state change on a group fact. The "
            f"cap is pp_admission_congruence.UNRESOLVED_DEFER_CAP -- one "
            f"authority for both defers; disarm the whole row-authority layer "
            f"with SGLANG_PP_ROW_AUTHORITY=0 if this must not stop a boot."
        )


def _first(missing: FrozenSet[str]) -> Optional[str]:
    """A stable representative of the set, for the log line and the tests."""
    return sorted(missing)[0] if missing else None


def _hashable(token: object) -> object:
    """A comparable stand-in for a frame stamp.

    Stamps are tuples today, but this module must not crash a boot over an
    unhashable one: an unusable token degrades to ``None``, which merely
    makes the count key on the missing set alone -- the pre-token behaviour,
    never an exception on the serving path.
    """
    try:
        hash(token)
    except TypeError:
        return None
    return token
