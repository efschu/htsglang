# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#1180: a BOUND on the #631 row-probe's defer, and a named stop at its end.

WHAT THE DEFER IS AND WHY IT MUST BE BOUNDED
--------------------------------------------
``SchedulerPPMixin._pp_proxy_frame_pending`` peeks the head proxy frame's
admission row without consuming it. If the row names a rid this rank cannot
locate, the probe answers False and LEAVES THE FRAME IN THE INBOX -- the
boot 631row14 protection, which exists because a frame consumed before the
request chain hop that carries its rids produces a plan that finds nothing
and a ring that dies upstream-waiting.

That protection is correct and this module does not weaken it. What it lacks
is a bound, and a deferred frame is not free: PP0 has ALREADY launched the
pass the frame belongs to and is ALREADY parked in the receive for its
output. A downstream rank that defers the frame for ever therefore closes a
ring by construction -- PP0 waits on the last rank's output, the last rank
waits on its upstream's chain, and the upstream is the rank that refuses to
execute the frame. That is boot weg1b7 (2026-09-04 03:50:24Z) exactly.

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
disagreement is a bounded, NAMED stop -- never a silent unbounded wait. The
#1071 occupant horizon says the same thing for the sibling case and is the
precedent this module follows deliberately, down to the "0 disables" shape.

WHY THE CLOCK RESTARTS ON A CHANGED MISSING SET
------------------------------------------------
The healthy case is not "no defers" -- boot 631row15 measured 43k of them,
hot, in a turning ring. In that world hops keep landing, so the set of rids
the rank cannot locate KEEPS CHANGING. Progress is therefore observable
without any wire access: any change to the missing set is a hop landing and
restarts the clock. Only an UNCHANGING missing set ages toward the bound,
which is exactly the state no chain hop can cure.

WHAT THIS MODULE IS NOT
-----------------------
It is not a recovery: it does not consume the frame, drain a wire, send a
void, or alter any rank's state. Consuming the frame at the bound would
trade this deadlock for the boot 631row14 deadlock (plan finds nothing,
ring dies upstream-waiting) and emitting a void would reopen the #801
reverse corpse. The bound's only product is a verdict and a message. That
is the whole design: an OBSERVATION that ends in a named stop, never a
compensation.

PURE BY CONSTRUCTION. No logger, no wire, no scheduler attribute -- the
caller owns the raise and the logging, exactly as ``parked_carrier_relief``
and ``min_free_slots_delayer`` are structured.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

__all__ = [
    "HORIZON_ENV",
    "DEFAULT_HORIZON_S",
    "PpRowDeferHorizonLapsed",
    "RowDeferVerdict",
    "RowDeferHorizon",
]

#: Operator override. ``0`` disables the bound and restores the pre-#1180
#: behaviour byte for byte (defer for ever), for a workload where a
#: genuinely long unsatisfiable row is somehow legitimate.
HORIZON_ENV = "SGLANG_PP_ROW_DEFER_HORIZON_S"

#: Default 20 s, chosen against three measured neighbours rather than picked:
#:
#:   * ABOVE the healthy case by orders of magnitude. In weg1b7 the three
#:     defers of one frame land inside the same log second, and the site's
#:     own premise bounds a legitimate wait by "rendezvous+transfer".
#:   * BELOW the request chain's stall bound (``SGLANG_PP_CHAIN_RECV_STALL_S``,
#:     60 s in the specimen), so THIS names the contradiction before the ring
#:     cut fires and reports it as "closed ring on the request chain" -- the
#:     misattribution that cost the weg1b7 postmortem its first reading.
#:   * BELOW the #1071 occupant horizon (90 s), for the same reason.
DEFAULT_HORIZON_S = 20.0


class PpRowDeferHorizonLapsed(RuntimeError):
    """#1180: a proxy frame's row named a rid this rank can never locate.

    Raised by the row probe when the SAME missing rid set has held a frame in
    the inbox past the horizon. It is a STOP, not a recovery: the sender's
    published row and this rank's request population disagree, the sender is
    already parked on the output of the pass that frame belongs to, and no
    amount of further waiting can change either fact.
    """


@dataclass(frozen=True)
class RowDeferVerdict:
    """The bound's answer for one (slot, missing-set) observation.

    ``defer`` True means "keep leaving the frame in the inbox" -- the shipped
    behaviour. ``defer`` False means the horizon lapsed and the caller owes a
    named stop; ``message`` is then non-empty and names every term.
    """

    defer: bool
    waited_s: float
    occurrence: int
    first_missing: Optional[str]
    n_missing: int
    message: Optional[str] = None


def _bound_from_env() -> float:
    """The configured bound, or 0.0 when it is unset/unreadable/disabled.

    Read per call rather than cached at import, mirroring
    ``_pp_occupant_horizon_lapsed``: the env is the operator's live knob on
    a boot that is already running under a deadman, and a cached read would
    make ``0`` (the documented escape hatch) unusable after import.
    """
    raw = os.environ.get(HORIZON_ENV)
    if raw is None:
        return DEFAULT_HORIZON_S
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return DEFAULT_HORIZON_S


class RowDeferHorizon:
    """Per-slot clock over the row probe's UNCHANGING missing-rid set."""

    def __init__(self) -> None:
        # slot -> (first_seen_monotonic, identity, occurrences), where the
        # identity is (frame token, missing-rid set). The TOKEN half is what
        # keeps a slot's clock honest across a ring that alternates heads: a
        # slot whose frame is not at the head is simply not observed, and the
        # next frame that arrives for it carries a different stamp, so it
        # cannot inherit the age of a long-resolved predecessor.
        self._since: Dict[int, Tuple[float, Tuple[object, FrozenSet[str]], int]] = {}

    def clear(self, mb_id: int) -> None:
        """Forget a slot's clock. Called on every non-deferring exit.

        A slot whose frame was delivered, judged, drained or found absent has
        no outstanding disagreement, so its clock must not survive into the
        next frame -- otherwise an old, long-resolved wait would age a fresh
        one straight into the bound.
        """
        self._since.pop(int(mb_id), None)

    def clear_all(self) -> None:
        """Forget every clock (the ring was rebuilt, e.g. by a cutover)."""
        self._since.clear()

    def observe(
        self,
        mb_id: int,
        missing_rids: Iterable[str],
        *,
        token: object = None,
        now: Optional[float] = None,
        bound_s: Optional[float] = None,
    ) -> RowDeferVerdict:
        """Record one defer of ``mb_id`` on ``missing_rids`` and rule on it.

        First sighting of a (slot, token, missing-set) triple starts its clock
        and always answers ``defer=True`` -- the ordinary hedge must stay free.
        A CHANGED missing set is progress (a hop landed) and restarts the
        clock; so does a different ``token``, which the caller passes as the
        frame's own stamp so that a NEW frame for the same slot can never
        inherit its predecessor's age. Only the same frame with the same
        missing set, held past the bound, answers ``defer=False``.
        """
        slot = int(mb_id)
        missing = frozenset(str(r) for r in missing_rids)
        identity = (_hashable(token), missing)
        clock = time.monotonic() if now is None else float(now)
        bound = _bound_from_env() if bound_s is None else float(bound_s)

        prior = self._since.get(slot)
        if prior is None or prior[1] != identity:
            self._since[slot] = (clock, identity, 1)
            return RowDeferVerdict(
                defer=True,
                waited_s=0.0,
                occurrence=1,
                first_missing=_first(missing),
                n_missing=len(missing),
            )

        first_seen, _held, occurrences = prior
        occurrences += 1
        self._since[slot] = (first_seen, identity, occurrences)
        waited = max(0.0, clock - first_seen)

        if bound <= 0.0 or waited <= bound:
            return RowDeferVerdict(
                defer=True,
                waited_s=waited,
                occurrence=occurrences,
                first_missing=_first(missing),
                n_missing=len(missing),
            )

        return RowDeferVerdict(
            defer=False,
            waited_s=waited,
            occurrence=occurrences,
            first_missing=_first(missing),
            n_missing=len(missing),
            message=self._message(slot, missing, waited, occurrences, bound),
        )

    @staticmethod
    def _message(
        slot: int,
        missing: FrozenSet[str],
        waited: float,
        occurrences: int,
        bound: float,
    ) -> str:
        named = ",".join(sorted(r[:8] for r in missing)[:4])
        return (
            f"#1180 PP ROW DEFER PAST ITS HORIZON: slot {slot} held a proxy "
            f"frame in the inbox for {waited:.1f}s over {occurrences} "
            f"probe(s) because its admission row names {len(missing)} rid(s) "
            f"this rank cannot locate, and the SAME set has not changed once "
            f"in that time (rids={named or 'NONE'}). A changing set is a "
            f"chain hop landing; an unchanging one is not a slow hop, it is "
            f"the sender and this rank disagreeing about who holds the rid. "
            f"THE TWO MEASURED SHAPES: the sender RETRACTED a rid after "
            f"sealing the row (weg1b7 03:49:20, #888b relief on c25108f3 one "
            f"iteration after #631 PROXY-SEND t40 named it), or the row names "
            f"a re-admitted resident that is requeued rank-locally and "
            f"traverses no chain at all (#968 READMIT CACHED, "
            f"last_queued_as=cutover-requeue). Neither can ever be cured by "
            f"waiting, and the sender is already parked on the output of the "
            f"pass this frame belongs to -- so deferring further closes the "
            f"ring instead of surviving it. Deliberately NOT repaired here: "
            f"consuming the frame would revive boot 631row14 (plan finds "
            f"nothing, ring dies upstream-waiting) and a void would revive "
            f"the #801 reverse corpse. Raise or disable the bound with "
            f"{HORIZON_ENV} (0 disables; {bound:g}s in force)."
        )


def _first(missing: FrozenSet[str]) -> Optional[str]:
    """A stable representative of the set, for the log line and the tests."""
    return sorted(missing)[0] if missing else None


def _hashable(token: object) -> object:
    """A comparable stand-in for a frame stamp.

    Stamps are tuples today, but this module must not crash a boot over an
    unhashable one: an unusable token degrades to ``None``, which merely
    makes the clock key on the missing set alone -- the pre-token behaviour,
    never an exception on the serving path.
    """
    try:
        hash(token)
    except TypeError:
        return None
    return token
