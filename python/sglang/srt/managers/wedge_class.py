# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#739: tell the two wedge populations apart, or say UNCLEAR.

WHY. The admission-wedge detector fires on one predicate -- "work is queued and
no FIRST token has appeared for N seconds" -- and that predicate is true of two
structurally different machines:

  A  THE PIPELINE IS DEAD. A rank sits in a blocking receive, the scheduler
     thread never reaches ``process_input_requests``, forward progress has
     stopped, and the recovery post is never consumed. Specimen
     wedge_802f_1712 (2026-08-22 17:10): all three ranks idle in blocking
     receives, relief posts UNCONSUMED five times, batch lines stopped.

  B  THE PIPELINE IS RUNNING AND THE POOL IS FULL. Resident requests decode
     normally, every slot is held, and nothing NEW can be admitted until one
     finishes. Specimen wedge_arm1_1845 (2026-08-22 18:45): 28-40 batch lines
     per minute THROUGHOUT the alarm, two of three ranks caught mid-forward by
     py-spy, relief posts consumed and answered NOT APPLICABLE / headroom
     sufficient, resolved by ordinary completion when slots came free.

Reported as one thing, these two mix two populations into every root hunt. The
2026-08-22 investigation spent a window on B believing it was A, and A's real
root (#788) is still open partly because its evidence was diluted.

THE THIRD STATE IS NOT OPTIONAL. Anything that does not clearly match A or B is
``UNCLEAR`` and says so. Silently folding an ambiguous case into the nearer
class is how a detector stops being evidence -- an indicator is only a finding
once it has been shown to measure what it claims, and this module's whole
purpose is that distinction. In particular the two CROSS combinations are
deliberately UNCLEAR, not "close enough":

  * no forward progress AND the post was consumed -- the pipeline is not
    running, yet the scheduler thread evidently is. That contradicts A's own
    mechanism, so it is the falsifier of this very split and must be visible,
    not absorbed.
  * forward progress AND the post was never consumed -- the mirror image, and
    equally a reason to doubt the split rather than to pick a side.

Pure and module-level: every decision here is testable without a scheduler, a
process group or a boot.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

__all__ = [
    "CLASS_PIPELINE_DEAD",
    "CLASS_POOL_SATURATED",
    "CLASS_UNCLEAR",
    "WedgeClass",
    "classify_wedge",
]

#: A rank is stuck; nothing is computing. The #788 comms family.
CLASS_PIPELINE_DEAD = "A-PIPELINE-DEAD"

#: Everything computes; the pool is full so nothing new is admitted.
CLASS_POOL_SATURATED = "B-POOL-SATURATED"

#: Not decidable from the evidence in hand. Never a synonym for "probably A".
CLASS_UNCLEAR = "UNCLEAR"


class WedgeClass(NamedTuple):
    """The verdict and the numbers it was reached on.

    ``detail`` always names the figures, because a class without its evidence
    is exactly the unfalsifiable label this module exists to replace.
    """

    label: str
    detail: str

    def render(self) -> str:
        return f"CLASS={self.label} ({self.detail})"


def classify_wedge(
    forward_delta: Optional[int],
    post_state: Optional[str],
    window_s: Optional[float] = None,
    usage_at_ceiling: Optional[bool] = None,
) -> WedgeClass:
    """Classify one wedge alarm from three cheap, already-available facts.

    ``forward_delta``    forward passes completed during the alarm window; 0
                         means the pipeline produced nothing, >0 means it did.
                         ``None`` when no sample exists yet.
    ``post_state``       the recovery channel's last outcome state
                         (``wedge_recovery.STATE_*``). ``None`` when the
                         channel has not answered.
    ``window_s``         the window the delta was measured over, for the log.
    ``usage_at_ceiling`` corroboration only: pool/slot occupancy at its limit.
                         It can strengthen B's detail line but never decides a
                         class on its own -- saturation is the steady state of
                         a busy server and was already true minutes BEFORE the
                         1845 alarm began, so treating it as the discriminator
                         would have misclassified that specimen.
    """
    from sglang.srt.managers.wedge_recovery import (
        STATE_NOT_APPLICABLE,
        STATE_UNCONSUMED,
    )

    span = "" if window_s is None else f" over {float(window_s):.0f}s"
    if forward_delta is None:
        return WedgeClass(
            CLASS_UNCLEAR,
            "no forward-progress sample yet, so neither class is available",
        )
    if post_state is None:
        return WedgeClass(
            CLASS_UNCLEAR,
            f"forward_delta={int(forward_delta)}{span} but the recovery "
            "channel has not answered yet, so the consumed/unconsumed half "
            "of the discriminator is missing",
        )

    delta = int(forward_delta)
    consumed = post_state == STATE_NOT_APPLICABLE
    unconsumed = post_state == STATE_UNCONSUMED

    if delta == 0 and unconsumed:
        return WedgeClass(
            CLASS_PIPELINE_DEAD,
            f"forward_delta=0{span} and the recovery post was {post_state}: "
            "nothing computed and the scheduler thread never took the post. "
            "This is the #788 comms family -- look for a rank in a blocking "
            "receive",
        )
    if delta > 0 and consumed:
        ceiling = ""
        if usage_at_ceiling:
            ceiling = ", pool/slot occupancy at its ceiling"
        return WedgeClass(
            CLASS_POOL_SATURATED,
            f"forward_delta={delta}{span} and the recovery post was "
            f"{post_state}{ceiling}: the pipeline is running and the actuator "
            "found nothing to relieve. Nothing is stuck -- new work waits for "
            "a slot. Not a comms defect",
        )
    if delta == 0 and consumed:
        return WedgeClass(
            CLASS_UNCLEAR,
            f"forward_delta=0{span} but the recovery post was {post_state}, "
            "i.e. the scheduler thread IS alive while nothing computes. That "
            "contradicts class A's own mechanism and is the named falsifier "
            "of this A/B split -- reported, deliberately not folded into A",
        )
    if delta > 0 and unconsumed:
        return WedgeClass(
            CLASS_UNCLEAR,
            f"forward_delta={delta}{span} but the recovery post was "
            f"{post_state}, i.e. work is completing while the scheduler "
            "thread never takes the post. The mirror falsifier -- reported, "
            "deliberately not folded into B",
        )
    return WedgeClass(
        CLASS_UNCLEAR,
        f"forward_delta={delta}{span} and recovery state {post_state!r} is "
        "neither of the two states this split is defined on",
    )
