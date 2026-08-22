# SPDX-License-Identifier: Apache-2.0
"""#800: make the admission-wedge recovery attempt SAY WHAT IT DID.

THE SPECIMEN THIS MODULE WAS WRITTEN AGAINST
--------------------------------------------
Boot r8 @ ``4c20c5b0d3``, log ``evidence-665-f1/wedge_1122_112408/boot.log``.
The admission-wedge detector (#699/#739) worked perfectly: 46 alarm lines, no
false positive, ``2 queued, 0 running`` and no first token for 218.2 s (all
four counts read out of the log for this module, not inherited). #788 had added
ONE bounded recovery action on top of that report. It announced four times and
reported a result three times -- the missing fourth is defect 3 below -- and
each result read:

    11:22:06 PP2  RECOVERY: forced-admission attempt returned None (None means
                  the gate is off or inert on this boot -- ...)

Three separate defects are in those two lines, and this module closes all
three.

DEFECT 1 -- THE REASON WAS GUESSED, AND THE GUESS WAS WRONG
    ``None`` came from ``before_admission``'s ``free - want >= floor`` exit:
    the corridor was HEALTHY, so the relief ladder had nothing to relieve. The
    log line asserted the opposite ("off or inert") about a gate the same boot
    had logged ``ARMED on device 0`` for on all three ranks at 10:44:01Z, and
    which had cleared four admissions at 11:19:47Z reclaiming 232 / 1112 /
    1126 / 1216 MiB. Six exits, one return value, one guess.
    Closed in ``corridor_admission.py`` by naming every exit; this module
    consumes those names.

DEFECT 2 -- THE ACTUATOR CANNOT DO THE THING ITS CALLER NAMED IT FOR
    ``guard_prefill_admission`` is a spill-before-alloc VRAM gate. Its own
    docstring is unambiguous: *"IT SPILLS. IT NEVER REFUSES"*, and
    ``before_admission``'s is *"THE RETURN VALUE IS EVIDENCE, NOT A DECISION
    ... Callers must admit the batch either way."* It has no admission
    actuator, on any boot, under any flag. Calling it a "forced-admission
    attempt" was a category error, and arming it harder cannot fix a wedge
    whose cards are at 0% load with the corridor intact.

    So this module does not pretend to force an admission. It performs the one
    action that is both truthful and diagnostic: it asks the SCHEDULER THREAD
    to run the relief ladder, and reports whether the scheduler thread was
    still running its loop at all. On the specimen that single fact -- posted,
    never consumed -- is a strictly better diagnosis than anything the old
    path could produce, because it separates "the scheduler is looping and
    declining to admit" from "the scheduler thread is not looping".

DEFECT 3 -- THE ACTUATOR RAN ON THE WRONG THREAD, AND IT ATE THE DETECTOR
    #788's own docstring named this hazard and accepted it: the watchdog
    thread called into CUDA allocator state (the ladder's first provider is
    ``torch.cuda.empty_cache()``) concurrently with a stuck forward thread.
    The specimen shows the hazard materialising. Four recovery announcements
    were logged and only three results: the missing one is PP0's. PP0
    announced at 11:23:09 and then emitted ZERO log lines of any kind over the
    remaining 88 s of the log, while PP1 and PP2 emitted 18 in that same
    window. Its own alarms had been arriving on the 10 s poll cadence up to
    that point (11:19:29 -> 11:19:39, 11:20:39 -> 11:20:49). The recovery
    attempt silenced the only mechanism that was still reporting. Both
    ``_attempt_recovery`` and its caller catch ``Exception``, so a clean raise
    is excluded; the thread blocked or died below Python.

    Hence the split below: the watchdog thread only ever writes an int and
    reads two ints. Every CUDA-touching byte runs on the scheduler thread, at
    a point that thread already reaches once per loop iteration.

A SECOND SPECIMEN, 2026-08-22 12:10-12:19Z (boot_r8_0822_1210.log), CONFIRMS
THE SHAPE AND ADDS ONE FINDING THIS MODULE CANNOT FIX ALONE
------------------------------------------------------------------------
Same code, new boot, same line. What it adds is WHICH RANK got the attempt:

    rank   last alarm   progress-clock age   recovery attempted
    PP1    12:16:50     56.0 s               NO
    PP2    12:16:50     55.0 s               NO
    PP0    12:16:55     61.4 s               YES

PP1 is the stuck rank. Its own log says so from the inside -- "WITHHOLDING
presence (5853 rounds so far) ... This rank is AT the entry and declining to
announce; it is not blocked upstream of it" -- while PP0's quorum waiter says
about the same rank, from the outside, "rank(s) [1] never reached the flip
entry". Those are ONE state described twice, and the outside description is
wrong: PP1 reached the entry. What PP0 can observe is that PP1 never
ANNOUNCED, which is a different fact.

And the only rank that crossed the 60 s recovery threshold is PP0, the rank
with nothing wrong with it. The threshold is compared against
``last_first_token_progress_time``, an ABSOLUTE per-rank clock, so on a
36 s episode the stuck rank cannot reach it by construction while a rank
whose clock happened to be older can.

THE PER-RANK CHANNEL BELOW IS THE HALF OF THAT THIS MODULE OWNS: each rank
posts to its OWN scheduler and classifies its OWN answer, so PP1 would report
UNCONSUMED (its loop is spinning inside the flip entry and never reaches
``process_input_requests``) while PP0 reports NOT_APPLICABLE. Two ranks, two
different states, which is the discrimination that did not exist before.
THE HALF IT DOES NOT OWN is the threshold: no per-rank classification helps a
rank that never reaches the attempt. Deliberately NOT retuned here -- 60 s is
a documented policy number with its own rationale in
``ADMISSION_WEDGE_RECOVERY_SECONDS`` -- and recorded instead so the next
reader has the measurement rather than the guess.


THE CHANNEL
-----------
One request, one acknowledgement, both plain attributes on the scheduler.

    watchdog thread            scheduler thread
    ---------------            ----------------
    post(seq)         ------>  drain(): runs the ladder, writes ack(seq, reason)
    settle(seq)       <------

``drain`` is called from ``Scheduler.process_input_requests``, which is the one
function every loop family reaches once per iteration: ``event_loop_normal``
(scheduler.py:2534), ``event_loop_overlap`` (:2597) and the three PP loops via
``_pp_forward_and_process_input_requests`` (scheduler_pp_mixin.py:2196). This
boot re-dispatches between ``event_loop_pp`` and ``event_loop_normal`` per
phase (``run_phase_flip_event_loops``), so a drain point in only one of them
would be inert in the other phase -- which is the failure mode this whole
module exists to stop shipping.

LOCK-FREE ON THE HOT PATH. ``drain`` is on the scheduler's per-iteration path,
so its fast path is one attribute read and one int comparison, with no lock
and no allocation. Under the GIL an int store is atomic and a torn read is not
possible; the worst a race can do is defer a drain by one iteration, and the
watchdog's grace window is three orders of magnitude larger than an iteration.
"""

from __future__ import annotations

import logging
import time
from typing import Any, NamedTuple, Optional

from sglang.srt.managers.corridor_admission import (
    ACTUATING_REASONS,
    REASON_NO_SCHEDULER,
    guard_prefill_admission_explained,
    reason_is_defect,
)

logger = logging.getLogger(__name__)

#: Attribute the per-scheduler channel is memoised under.
RECOVERY_CHANNEL_ATTR = "admission_wedge_recovery_channel"

# -- outcome states ---------------------------------------------------------
#: The scheduler thread consumed the request and the relief ladder RAN.
STATE_ACTUATED = "ACTUATED"
#: The scheduler thread consumed the request, the gate was PRESENT and
#: WORKING, and it correctly had nothing to do (the corridor was already
#: clear, or a relief had just run). Not a defect. NOT LOUD.
#:
#: This is the state the 2026-08-22 specimen was actually in, and reporting it
#: as "the gate is off or inert" is the whole reason this module exists. It is
#: still NON-ACTUATING -- the recovery did not help the wedge -- so it feeds
#: the escalation below like any other non-actuating outcome. What changes is
#: that a single one of them no longer shouts.
STATE_NOT_APPLICABLE = "NOT_APPLICABLE"
#: The scheduler thread consumed the request and the MECHANISM was missing or
#: broke: no corridor guard reachable, the card unreadable, the ladder raised.
#: A defect, and LOUD -- this is the state that looks identical to health from
#: the outside, which is exactly why it has to announce itself.
STATE_INERT = "INERT"
#: The scheduler thread did not consume the request within the grace window.
#: It is therefore not running its loop, which is a finding about the WEDGE
#: and not about the gate. The old path could not produce this state at all.
STATE_UNCONSUMED = "UNCONSUMED"
#: Posted, grace window not yet elapsed, no acknowledgement yet. Not a result.
STATE_PENDING = "PENDING"
#: Nothing has been posted since the last reset.
STATE_IDLE = "IDLE"

#: States in which the recovery demonstrably failed to change anything. A run
#: of these is what escalation counts.
#:
#: NOT_APPLICABLE IS IN HERE ON PURPOSE. "The gate correctly had nothing to
#: do" is not a defect of the gate and is not logged as one -- but as an
#: answer to "is this wedge recovering", it is a no. A single one is quiet; a
#: RUN of them is the finding, and the run is what gets the loud line.
NON_ACTUATING_STATES = frozenset({STATE_NOT_APPLICABLE, STATE_INERT, STATE_UNCONSUMED})

#: How long the scheduler thread may take to pick a request up before its
#: silence is reported as a finding, in seconds.
#:
#: PROVENANCE, and why it is not smaller. A healthy scheduler reaches
#: ``process_input_requests`` many times a second, so any value above a
#: millisecond would do -- except during a phase flip, where the loop is inside
#: the flip machinery and legitimately does not reach that function at all. The
#: number to beat is therefore the LONGEST completed flip leg in the specimen,
#: not the last one: 26918.8 ms (boot.log; the last leg, ``tp_to_pp (epoch 4)``,
#: was 23756.5 ms). 45 s is 1.67x the longest observed leg, so a flip in
#: progress cannot be reported as a stalled scheduler, and it fits comfortably
#: inside the 218.2 s the wedge episode ran.
DEFAULT_ACK_GRACE_SECONDS: float = 45.0

#: Consecutive non-actuating recoveries after which the state becomes loud and
#: distinguishable. Two, not one: a single INERT can be an honest "the corridor
#: was fine", and a single UNCONSUMED can be one long flip leg. Two in a row,
#: spaced by the retry interval below, is a mechanism that is not working.
DEFAULT_ESCALATE_AFTER: int = 2

#: Minimum seconds between two recovery attempts inside one wedge episode.
#:
#: #788 made exactly ONE attempt per episode, which is why the specimen's
#: 218.2 s episode produced one data point per rank and no way to see a trend.
#: Retrying lets the escalation above have something to count, and 30 s keeps
#: the attempt rate two orders of magnitude below the scheduler's own loop.
DEFAULT_RETRY_SECONDS: float = 30.0

#: Log token an external watchdog can grep for. Deliberately NOT the
#: ADMISSION_WEDGE token: the whole point is that this state must not look
#: like the 46 identical alarm lines it is buried in.
ESCALATION_TOKEN = "ADMISSION-WEDGE-UNRECOVERED"


def _classify(reason: str) -> str:
    """One gate exit -> one of the three outcome states.

    THREE, not two. ``verdict is not None`` was one bit for at least three
    distinguishable situations, and the caller that had to pick a meaning
    picked the wrong one. The three are: the ladder ran; it was present and
    correctly idle; it was missing or broke.
    """
    if reason in ACTUATING_REASONS:
        return STATE_ACTUATED
    if reason_is_defect(reason):
        return STATE_INERT
    return STATE_NOT_APPLICABLE


class RecoveryOutcome(NamedTuple):
    """The result of one recovery attempt, or the absence of one."""

    state: str
    #: The ``corridor_admission`` REASON_* constant, when the ladder ran or
    #: declined to. Empty for UNCONSUMED -- nobody got far enough to have one.
    reason: str
    seq: int
    #: Seconds between posting the request and this classification.
    waited_s: float

    @property
    def actuated(self) -> bool:
        return self.state == STATE_ACTUATED


class WedgeRecoveryChannel:
    """The watchdog<->scheduler request channel. One per scheduler."""

    def __init__(self) -> None:
        # NO CLOCK OF ITS OWN. ``post`` and ``settle`` are handed the caller's
        # ``now``, so the driver's clock is the only clock in the decision and
        # a test cannot accidentally measure the driver against wall time --
        # which is exactly how the first draft of this file reported PENDING
        # for a request it had aged 50 simulated seconds.
        # Written by the watchdog thread, read by the scheduler thread.
        self.requested_seq: int = 0
        self.requested_tokens: int = 0
        # Written by the scheduler thread, read by the watchdog thread.
        self.acked_seq: int = 0
        self.acked_reason: str = ""
        self.acked_at: float = 0.0
        # Watchdog-thread-only bookkeeping.
        self._posted_at: float = 0.0
        self._consecutive_non_actuating: int = 0
        self._escalated: bool = False
        #: The last SETTLED outcome, kept so the recovery's own result is a
        #: readable fact and not only a log line. #788 had no such field, and
        #: that is the whole reason its ineffectiveness could be reported
        #: three times without anything acting on it: a log line is not a
        #: consumer. Anything that publishes a wedge verdict -- the #799
        #: ``wedge_status`` file publisher when the two branches meet, an HTTP
        #: probe, an external supervisor -- reads this rather than re-deriving
        #: it. PENDING is never stored: it is not a result.
        self.last_outcome: Optional[RecoveryOutcome] = None

    # -- watchdog thread. NO CUDA, NO IMPORTS, NO ALLOCATION BEYOND AN INT --

    def post(self, now: float, tokens: int = 0) -> int:
        """Ask the scheduler thread to run the ladder. Returns the sequence."""
        self.requested_tokens = int(tokens)
        self._posted_at = now
        # LAST. The scheduler thread gates its drain on this field, so the
        # tokens it will read must already be in place when it becomes visible.
        self.requested_seq += 1
        return self.requested_seq

    def settle(
        self, now: float, grace_s: float = DEFAULT_ACK_GRACE_SECONDS
    ) -> RecoveryOutcome:
        """Classify the outstanding request. Watchdog thread only."""
        seq = self.requested_seq
        if seq == 0:
            return RecoveryOutcome(STATE_IDLE, "", 0, 0.0)
        waited = now - self._posted_at
        if self.acked_seq >= seq:
            reason = self.acked_reason
            return RecoveryOutcome(_classify(reason), reason, seq, waited)
        if waited < grace_s:
            return RecoveryOutcome(STATE_PENDING, "", seq, waited)
        return RecoveryOutcome(STATE_UNCONSUMED, "", seq, waited)

    def record(self, outcome: RecoveryOutcome, escalate_after: int) -> bool:
        """Fold one settled outcome into the escalation run.

        Returns True on the transition INTO the escalated state, so the caller
        logs the loud line exactly once per run rather than on every poll.
        PENDING and IDLE are not results and must not move the counter --
        counting them would let a slow scheduler escalate itself by being slow
        once, which is the false-positive shape #699 was careful to avoid.
        """
        if outcome.state == STATE_ACTUATED:
            self.last_outcome = outcome
            self._consecutive_non_actuating = 0
            self._escalated = False
            return False
        if outcome.state not in NON_ACTUATING_STATES:
            return False
        self.last_outcome = outcome
        self._consecutive_non_actuating += 1
        if self._escalated or self._consecutive_non_actuating < escalate_after:
            return False
        self._escalated = True
        return True

    def reset_episode(self) -> None:
        """The wedge cleared. Forget the run, keep the sequence monotone."""
        self._consecutive_non_actuating = 0
        self._escalated = False

    @property
    def consecutive_non_actuating(self) -> int:
        return self._consecutive_non_actuating

    @property
    def escalated(self) -> bool:
        return self._escalated

    # -- scheduler thread ------------------------------------------------

    def drain(self, scheduler: Any) -> Optional[RecoveryOutcome]:
        """Run any outstanding request HERE, on the calling thread.

        Fast path: one attribute read, one int compare, return. This runs once
        per scheduler loop iteration and must stay free of locks, imports and
        allocation.
        """
        seq = self.requested_seq
        if seq == self.acked_seq:
            return None
        try:
            actuation = guard_prefill_admission_explained(
                scheduler, self.requested_tokens
            )
            reason = actuation.reason
        except Exception as e:  # noqa: BLE001 - must never take the loop down
            logger.error(
                "[#800 WEDGE-RECOVERY] the relief ladder raised on the "
                "scheduler thread: %s. Acknowledging the request anyway so "
                "the watchdog reports INERT with this reason rather than "
                "UNCONSUMED, which would name the wrong thread.",
                e,
            )
            reason = "drain-raised"
        self.acked_reason = reason
        self.acked_at = time.monotonic()
        # LAST, and deliberately so: the watchdog reads ``acked_seq`` as the
        # gate on the other two fields, so it must become visible after them.
        self.acked_seq = seq
        return RecoveryOutcome(_classify(reason), reason, seq, 0.0)


def get_recovery_channel(scheduler: Any) -> Optional[WedgeRecoveryChannel]:
    """The scheduler's memoised channel, built on first use."""
    if scheduler is None:
        return None
    channel = getattr(scheduler, RECOVERY_CHANNEL_ATTR, None)
    if channel is None:
        channel = WedgeRecoveryChannel()
        setattr(scheduler, RECOVERY_CHANNEL_ATTR, channel)
    return channel


def drain_recovery_request(scheduler: Any) -> Optional[RecoveryOutcome]:
    """Scheduler-thread entry point. Called once per loop iteration.

    NOTHING IS BUILT UNLESS SOMETHING WAS POSTED. The channel is only created
    by ``get_recovery_channel``, and the only caller that creates it is the
    watchdog's post path; this function reads the attribute and returns on a
    plain ``None`` for every boot on which no wedge was ever detected. That is
    the whole cost of this feature on a healthy run.
    """
    channel = getattr(scheduler, RECOVERY_CHANNEL_ATTR, None)
    if channel is None:
        return None
    return channel.drain(scheduler)


__all__ = [
    "DEFAULT_ACK_GRACE_SECONDS",
    "DEFAULT_ESCALATE_AFTER",
    "DEFAULT_RETRY_SECONDS",
    "ESCALATION_TOKEN",
    "NON_ACTUATING_STATES",
    "RECOVERY_CHANNEL_ATTR",
    "REASON_NO_SCHEDULER",
    "RecoveryOutcome",
    "STATE_ACTUATED",
    "STATE_IDLE",
    "STATE_INERT",
    "STATE_NOT_APPLICABLE",
    "STATE_PENDING",
    "STATE_UNCONSUMED",
    "WedgeRecoveryChannel",
    "drain_recovery_request",
    "get_recovery_channel",
]
