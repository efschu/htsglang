# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#656: the contract that lets kv-session-offload and the phase flip coexist.

THE CONFLICT THIS REPLACES
--------------------------
``flip_blocking_guards`` refused flip arming outright whenever
``scheduler.kv_session_offload`` was not None::

    if getattr(scheduler, "kv_session_offload", None) is not None:
        guards.append("kv-session-offload")

A FEATURE was the guard. So enabling kvso turned the flip off, the host half
of spec items 6/12/15c could not be reached at all, and the two halves of the
order -- "spill the cold half to system RAM" and "flip PP<->TP automatically"
-- were mutually exclusive in this tree. That is a state, not a feature, and
the guard should test the state.

WHY THE BLANKET REFUSAL WAS NOT SIMPLY WRONG
--------------------------------------------
It was too broad, but it was protecting something real, and a successor who
lifts it without replacing it re-opens a correctness hole rather than a
capability.

**A kvso host image is LAYOUT-SPECIFIC.** In the PP phase a rank holds its
stage's layers for every token; in the TP phase it holds a token shard of
every layer. The same session's host image is therefore a different object in
the two phases, and there is no reading of the bytes that is correct in both.
Restoring a PP-captured image into a TP layout would not raise -- it would
return the wrong K/V and generate quietly wrong text, which is the worst
failure class this chain ships against.

**And a spilled kvso session is not passive.** It keeps generating THROUGH
the host tier: one spill tick per scheduler iteration recomputes over host
resident K/V. A session mid-tick during a cutover is a device-to-host copy
racing a layout change.

THE CONTRACT
------------
Three states, and only one of them refuses:

``absent``  kvso is off. Nothing to say.
``idle``    kvso is on and holds no spilled session. A flip is as safe as it
            is with kvso off entirely -- the manager is a destination that
            nothing has used yet.
``parked``  kvso holds spilled sessions, every one of them stamped with the
            phase it was captured in, no copy is in flight, and no stamp
            names the phase the flip is about to ENTER. The images belong to
            the phase being left; that phase returns (under strict purity
            both phases recur every few seconds), and until it does, those
            slots do not tick. A flip is safe.
``busy``    a copy is in flight, or a slot carries no stamp, or a stamp is
            unreadable. REFUSE, and retry next round -- this is a transient
            state, not a configuration error, so the refusal costs one round.

``incoherent`` is folded into ``busy`` deliberately: an unstamped slot means
some spill path did not go through :func:`stamp_spill`, and the honest
response to "I cannot prove which layout these bytes belong to" is the same
as to "a copy is running" -- do not flip.

WHAT MAKES THE PARKED STATE SAFE IS THE TICK GATE, NOT THE STAMP
-----------------------------------------------------------------
The stamp is only evidence. The enforcement is :func:`pin_spills_to_phase`,
called at the post-cutover hook: every slot whose stamp is not the phase now
live has ``suppress_tick`` set, so the scheduler's tick picker will not run
it. A stamp with no gate would document the hazard instead of preventing it.

The gate reuses ``SpillSlot.suppress_tick``, which already exists for the
restore-readiness handshake and is already honoured by the tick picker. It is
one-shot there (the picker clears it), so the pin is RE-APPLIED every round
by :func:`pin_spills_to_phase` rather than set once -- a one-shot flag used as
a latch would release itself on the first tick it prevented, which is exactly
the tick it must keep preventing.

RANK UNIFORMITY
---------------
Every input here is replicated scheduler state: the spill registry is driven
by rank-uniform decisions, the phase is agreed by the flip's own consensus,
and the CUDA event query is rank-local but only ever makes a rank say "wait".
This function is therefore safe to call on the flip's arming path, which is
rank-local by construction. A rank that says ``busy`` alone simply does not
arm this round, which is the pre-existing behaviour of every other guard in
that list.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "KVSO-FLIP"

STATE_ABSENT = "absent"
STATE_IDLE = "idle"
STATE_PARKED = "parked"
STATE_BUSY = "busy"

#: The states in which a flip may arm.
FLIP_SAFE_STATES = (STATE_ABSENT, STATE_IDLE, STATE_PARKED)


def slot_layout_stamp(slot: Any) -> Optional[str]:
    """The phase a spill slot's host image was captured in, or None.

    None means "not provable", never "fine". See the module docstring.
    """
    stamp = getattr(slot, "flip_layout", None)
    if stamp is None:
        return None
    stamp = str(stamp)
    return stamp or None


def stamp_spill(slot: Any, phase: Optional[str]) -> None:
    """Record the layout a host image is being captured in.

    Called at every site that registers a slot into ``manager.spills``. A
    phase of None leaves the slot unstamped, which the state machine reads as
    ``busy`` -- the safe direction, because an unstamped image is one nobody
    can place.
    """
    try:
        slot.flip_layout = str(phase) if phase else None
    except AttributeError:
        # A slot class with __slots__ that does not carry the field: the
        # stamp is then simply unavailable, and the state machine refuses.
        # Better a refused flip than a silent mis-restore.
        logger.warning(
            "%s spill slot %r cannot carry a layout stamp; flips will be "
            "refused while it is parked",
            LOG_PREFIX,
            type(slot).__name__,
        )


def copy_in_flight(manager: Any) -> bool:
    """True when kvso's host copy stream has work outstanding.

    Reads the same event the wave controller reads
    (``backend._sess_wave_done``). Unreadable is treated as IN FLIGHT: the
    question being asked is "is it safe to change the layout under this
    copy", and an unanswerable question is not a yes.
    """
    backend = getattr(manager, "backend", None)
    if backend is None:
        return False
    event = getattr(backend, "_sess_wave_done", None)
    if event is None:
        return False
    try:
        return not bool(event.query())
    except Exception as e:  # noqa: BLE001 - an unreadable event is "busy"
        logger.warning("%s wave event unreadable (%s); reporting in-flight", LOG_PREFIX, e)
        return True


def flip_safety_state(
    manager: Any,
    *,
    current_phase: Optional[str],
    incoming_phase: Optional[str] = None,
) -> Tuple[str, str]:
    """``(state, detail)`` for the guard. See the module docstring.

    ``current_phase`` is the phase now live; ``incoming_phase`` the one the
    flip would enter. The incoming phase matters because a slot stamped with
    it would be restored into a layout its bytes were not captured in -- the
    one case where parked images are NOT safe to carry across.
    """
    if manager is None:
        return STATE_ABSENT, ""
    if copy_in_flight(manager):
        return STATE_BUSY, "a host copy is in flight"
    spills = getattr(manager, "spills", None) or {}
    if not spills:
        return STATE_IDLE, "no spilled session"
    unstamped = []
    incoming_stamped = []
    stamps = {}
    for key, slot in spills.items():
        stamp = slot_layout_stamp(slot)
        if stamp is None:
            unstamped.append(key)
            continue
        stamps[key] = stamp
        if incoming_phase is not None and stamp == str(incoming_phase):
            incoming_stamped.append(key)
    if unstamped:
        return (
            STATE_BUSY,
            f"{len(unstamped)} spilled session(s) carry no layout stamp "
            f"(req_pool_idx {sorted(unstamped)[:4]}), so their host images "
            f"cannot be placed in either layout",
        )
    if incoming_stamped:
        return (
            STATE_BUSY,
            f"{len(incoming_stamped)} spilled session(s) are stamped with the "
            f"INCOMING phase {incoming_phase!r} (req_pool_idx "
            f"{sorted(incoming_stamped)[:4]}): entering it would make them "
            f"restore-eligible against a layout change they did not survive",
        )
    return (
        STATE_PARKED,
        f"{len(spills)} spilled session(s) parked, all stamped "
        f"{sorted(set(stamps.values()))} against live phase "
        f"{current_phase!r}; their ticks are pinned until that phase returns",
    )


def pin_spills_to_phase(manager: Any, phase: Optional[str]) -> int:
    """Suppress the tick of every spilled session not belonging to ``phase``.

    THE ENFORCEMENT half of the contract. Returns the number of slots pinned.
    Must be called EVERY round while a foreign-layout image is parked, not
    once at the cutover: ``suppress_tick`` is a one-shot the tick picker
    clears, so a single set would release on the first tick it suppressed.
    """
    if manager is None or not phase:
        return 0
    pinned = 0
    for slot in (getattr(manager, "spills", None) or {}).values():
        stamp = slot_layout_stamp(slot)
        if stamp is not None and stamp == str(phase):
            continue
        try:
            slot.suppress_tick = True
        except AttributeError:
            continue
        pinned += 1
    return pinned


def restore_permitted(slot: Any, current_phase: Optional[str]) -> bool:
    """May this slot's host image be read back into device memory NOW?

    Only into the layout it was captured in. This is the last line of defence
    and it is deliberately independent of the tick gate: the gate stops the
    session from RUNNING in the wrong phase, this stops its bytes from being
    COPIED BACK in the wrong phase, and a bug in either one alone is then
    still caught by the other.

    NO PHASE MEANS NO PHASE FLIP, WHICH MEANS ALWAYS PERMITTED. A process
    without the flip has exactly one KV layout for its whole life, so there
    is no wrong layout to restore into. Reading a missing phase as "refuse"
    would switch kvso's restore path off for every user who never enabled
    the flip -- a live feature disabled as a side effect of a guard for a
    hazard that cannot occur there. The refuse-by-default direction applies
    to a missing STAMP inside a flip-enabled process, which is the case
    where the answer is genuinely unknown.
    """
    if not current_phase:
        return True
    stamp = slot_layout_stamp(slot)
    if stamp is None:
        return False
    return stamp == str(current_phase)
