# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#850: which presence-withhold reasons the armed service turn can actually clear.

THE GENERALISATION #800 STOPPED ONE SHORT OF. #800 proved that a rank can
withhold presence forever on a message whose only consumer is the PP loop body,
because the presence gate is what prevents that body from running. It fixed the
instance it measured -- ``admission_decision`` parked in the tensor-dict inbox,
57922 withheld rounds -- by giving that ONE channel a disposition table and an
escape clock. The gate's other withhold reasons were left with no clock and no
classification at all.

They are not equivalent. ``pp_flip_service`` is the armed loop's one service
turn, and it holds exactly four actuators:

    ``pp_flip_consume_inbound``        the request-chain WIRE
    ``pp_flip_drain_tensor_dicts``     the tensor-dict WIRE
    ``pp_flip_flush_drained_sends``    this rank's outstanding sends
    ``pp_flip_retire_undeclared_stash`` the UNDECLARED stash, on #800's clock

A reason an actuator covers is SELF_CLEARING: waiting is how it gets fixed, and
the 609 ``send_output_work is not reaped`` withholds measured across 291 boot
logs are exactly that -- healthy, transient, reaped on a later turn. A reason no
actuator covers is CONSUMER_EXCLUDED: nothing in the armed window can change it,
so every round spent waiting is spent losing. The rank will withhold until the
presence deadline, the group will abandon on a 60 s timer, and the outcome after
those 60 s is identical to the outcome available in the first round.

WHY THIS IS A DEFECT AND NOT A TUNING QUESTION. The gate's withhold branch is a
WAIT, and a wait is only meaningful against a thing that can happen. Waiting on
a consumer the waiter itself excludes is not a slow path, it is a category
error, and it presents as the same silence #800 documented: no announcement, no
alarm, and 60 s later an abandonment naming the rank as one that "never reached
the entry" -- which is false, it reached the entry and declined. Naming the
futility at the FIRST round turns that silence into a fact.

MEASURED, on the 291 boot logs under ``/spinning/evidence-665-f1`` (tally by
reason, 2026-08-24):

    609  send_output_work is not reaped          SELF_CLEARING
    203  still owes a chain send                 SELF_CLEARING
     65  tensor-dict inbox holds N stashed        <- #800's killer, now classified
     11  request-chain inbox holds N unhandled    CONSUMER_EXCLUDED, unfixed until now

The 11 are the residue #800 did not reach. ``pp_flip_consume_inbound`` moves a
request-chain message from the WIRE into ``receiver.inbox`` -- which is correct
and load-bearing, since the upstream's blocking send is satisfied by the message
leaving the wire -- but the only thing that pops that inbox is
``_pull_raw_reqs`` at the top of a PP pass. Consuming while armed therefore
converts a clearable reason into an unclearable one, once per message.

WHAT THIS MODULE DOES NOT DO. It does not move, drop, retire or re-route a
single message. Requests must not be lost and must exist in exactly one place
across a cutover (#731); deciding where a parked request belongs in the new
layout is a routing question this module deliberately leaves alone. It only
answers "can waiting possibly help?", so the caller can stop waiting when the
answer is no. Abandoning is already the free and safe outcome at this point --
``_abandon_no_quorum``'s own line is "NOTHING was entered and no request was
touched" -- and ``pp_flip_channels_empty``'s ``pp_outputs`` branch already
states the intent verbatim: "an output that never drains costs a loud
abandonment, which is the behaviour this feature had under load anyway".
Abandoning in the first round instead of the sixtieth returns the group to
serving 58 s sooner and changes nothing else.

UNCLASSIFIED IS CONSERVATIVE, AND LOUD. A clause no marker matches is NOT
treated as futile -- this module will not abandon a flip on a sentence it does
not understand. It is reported by name from the first round, the same bargain
#800 struck with ``UNDECLARED``: a future fifth reason costs a named line
instead of an unbounded silence. ``test_pp_presence_disposition`` pins the
marker table against the reason literals ``pp_flip_channels_empty`` actually
produces, so adding a reason without classifying it fails a test rather than
reaching metal unclassified.

Pure and module-level, so every decision here is testable without a scheduler,
a process group or a boot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

__all__ = [
    "ALARM_PRESENCE_FUTILE",
    "CONSUMER_EXCLUDED",
    "SELF_CLEARING",
    "UNCLASSIFIED",
    "WithholdCensus",
    "census_withhold_reason",
    "classify_withhold_clause",
    "withhold_markers",
]

#: THE GREP KEY. A BOOT-ACCEPTANCE CRITERION, so it is a literal with no
#: interpolation in it and it never moves -- the same contract #838's
#: ``ALARM_CONFORMANCE`` states for the same reason. Expect 0 on a healthy
#: boot: a non-zero count means a rank waited on a consumer it was itself
#: excluding, which is #800's shape on another channel.
ALARM_PRESENCE_FUTILE = "DEFECT PRESENCE-WAIT-FUTILE (#850)"

#: An actuator in ``pp_flip_service`` covers this reason. Waiting is the fix.
SELF_CLEARING = "self_clearing"

#: No armed-window actuator covers this reason: only the resumed PP loop body,
#: or a consumer that runs after the cutover, can clear it. Waiting cannot.
CONSUMER_EXCLUDED = "consumer_excluded"

#: No marker matched. Held conservatively -- never futile -- and named.
UNCLASSIFIED = "unclassified"


#: ``(marker, class, who clears it)``, matched in order against a reason clause.
#:
#: ORDER IS LOAD-BEARING for the two ``tensor-dict inbox holds`` clauses: #800's
#: census emits both an owed-consumer variant and an UNDECLARED variant with the
#: same opening words, and only the second is on an escape clock. The UNDECLARED
#: marker is therefore tested first.
_MARKERS: Tuple[Tuple[str, str, str], ...] = (
    (
        "of UNDECLARED disposition",
        SELF_CLEARING,
        "pp_flip_retire_undeclared_stash, on #800's escape clock",
    ),
    (
        "request-chain inbox holds",
        CONSUMER_EXCLUDED,
        "_pull_raw_reqs at the top of a PP pass -- which the presence gate "
        "structurally excludes (#850)",
    ),
    (
        "request chain has",
        SELF_CLEARING,
        "pp_flip_consume_inbound",
    ),
    (
        "request chain is HALF-RECEIVED",
        SELF_CLEARING,
        "pp_flip_consume_inbound, which completes the posted payload receive",
    ),
    (
        "tensor-dict wire has",
        SELF_CLEARING,
        "pp_flip_drain_tensor_dicts",
    ),
    (
        "tensor-dict inbox holds",
        CONSUMER_EXCLUDED,
        "a consumer that looks for it AFTER the cutover (#800 BLOCKS_FLIP), so "
        "no armed turn can clear it",
    ),
    (
        "is not reaped",
        SELF_CLEARING,
        "pp_flip_flush_drained_sends",
    ),
    (
        "last_rank_comm_queue is not empty",
        CONSUMER_EXCLUDED,
        "the last rank's own send loop, which does not run while armed",
    ),
    (
        "pp_outputs holds",
        CONSUMER_EXCLUDED,
        "the next PP pass, which turns the received output into tokens -- "
        "draining it here is corpse S and stays refused",
    ),
)


def withhold_markers() -> Tuple[Tuple[str, str, str], ...]:
    """The marker table, for tests and diagnostics."""
    return _MARKERS


def classify_withhold_clause(clause: str) -> Tuple[str, Optional[str]]:
    """Classify ONE reason clause; ``(class, who clears it)``.

    Never raises and never returns ``None`` for the class. An unmatched clause
    is ``UNCLASSIFIED`` with no clearer, which the caller must treat as
    non-futile -- see this module's docstring.
    """
    text = str(clause)
    for marker, klass, clearer in _MARKERS:
        if marker in text:
            return klass, clearer
    return UNCLASSIFIED, None


@dataclass(frozen=True)
class WithholdCensus:
    """One withhold reason, split by whether waiting can possibly help.

    ``futile`` and ``self_clearing`` are DISJOINT, and a reason with any
    self-clearing clause is NOT futile overall: the gate must keep waiting for
    the clause that a service turn can still fix, even when another clause
    beside it never will. Abandoning early is only correct when EVERY clause is
    beyond the armed window's reach.
    """

    #: ``(clause, who would have to clear it)`` for consumer-excluded clauses.
    futile: Tuple[Tuple[str, str], ...] = ()
    #: Clauses an armed-window actuator covers.
    self_clearing: Tuple[str, ...] = ()
    #: Clauses no marker matched.
    unclassified: Tuple[str, ...] = ()

    @property
    def is_futile(self) -> bool:
        """True when no clause can be cleared by any armed service turn.

        Requires at least one futile clause AND no clause of any other class:
        an unclassified clause suppresses the verdict, because this module does
        not abandon a flip on a sentence it cannot read.
        """
        return bool(self.futile) and not self.self_clearing and not self.unclassified

    def futile_reason(self) -> Optional[str]:
        """Why waiting cannot help, naming each clause and its real consumer."""
        if not self.futile:
            return None
        return "; ".join(
            f"{clause!r} can only be cleared by {clearer}" for clause, clearer in self.futile
        )


def census_withhold_reason(reason: Optional[str]) -> WithholdCensus:
    """Split ``pp_flip_channels_empty``'s joined reason string by disposition.

    The producer joins its clauses with ``"; "`` and so does #800's stash
    census, so splitting on that separator yields clauses at both levels --
    which is what the marker table expects.
    """
    if not reason:
        return WithholdCensus()
    futile: List[Tuple[str, str]] = []
    self_clearing: List[str] = []
    unclassified: List[str] = []
    for raw in str(reason).split("; "):
        clause = raw.strip()
        if not clause:
            continue
        klass, clearer = classify_withhold_clause(clause)
        if klass == CONSUMER_EXCLUDED:
            futile.append((clause, clearer or "an unnamed consumer"))
        elif klass == SELF_CLEARING:
            self_clearing.append(clause)
        else:
            unclassified.append(clause)
    return WithholdCensus(
        futile=tuple(futile),
        self_clearing=tuple(self_clearing),
        unclassified=tuple(unclassified),
    )
