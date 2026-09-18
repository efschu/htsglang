"""Weg 2 group P intake stall (weg2xsn272, 18.09.2026).

P prefills the whole backlog and KEEPS every prefilled request's KV in its
tree for D (write-through to the host arena, device rows evictable only
after the flip's readmission). The smallest PP stage bounds one request's
extend by its pool (xsn272: max_total_num_tokens=121190 on PP0). When the
next queued request does not fit what the pool can still give, the adder
answers ``NO_TOKEN`` silently, the running batch is EMPTY (nothing runs, so
nothing will ever free a token), the scheduler loop spins (#969M ARM every
millisecond for 150 s) and the front waits for a drain that cannot end:
IDLE-WEDGE, boot killed.

The watch below turns that silence into a NAMED refusal. It is a pure
object: the scheduler feeds it the refused request's terms on every pass;
it answers with the refusal message once the SAME request has been refused
with an empty running batch for ``hold_s`` (a transient shortage while a
batch finishes must not be mistaken for a stall -- with the batch empty
there is no such transient, the hold only rides out one loop's jitter).
The front recognises the message (:func:`is_intake_stall`), puts the request
back at the HEAD of its queue, stops dispatching and flips to D; the request
is prefilled in the next P phase, when the flip has emptied P's pool.
"""

from __future__ import annotations

from typing import Optional

#: The refusal's name; the front matches on it in the leg-1 error text.
STALL_MARK = "WEG2-INTAKE-STALL"

#: Refused-with-empty-batch this long -> a stall, not a transient.
HOLD_S_DEFAULT = 1.0


class IntakeStallWatch:
    """Observe the adder's NO_TOKEN refusals; name a stall once."""

    def __init__(self, hold_s: float = HOLD_S_DEFAULT) -> None:
        self.hold_s = float(hold_s)
        self._rid: Optional[str] = None
        self._since: float = 0.0
        self._reported: set = set()
        self.stalls = 0

    def progress(self) -> None:
        """Something was admitted or ran: no request is stalling right now."""
        self._rid = None
        self._since = 0.0

    def observe(self, *, rid: str, need_tokens: int, rem_total_tokens: int,
                cur_rem_tokens: int, running_empty: bool, now: float) -> Optional[str]:
        """One NO_TOKEN refusal of ``rid``. Returns the refusal message the
        moment the stall is established (once per rid), else None."""
        if not running_empty:
            # a running batch will free tokens when it finishes: wait for it
            self._rid = None
            return None
        if rid in self._reported:
            return None
        if self._rid != rid:
            self._rid = rid
            self._since = float(now)
            return None
        held = float(now) - self._since
        if held < self.hold_s:
            return None
        self._reported.add(rid)
        self.stalls += 1
        return (
            f"{STALL_MARK} rid={rid} need_tokens={int(need_tokens)} "
            f"rem_total_tokens={int(rem_total_tokens)} cur_rem_tokens={int(cur_rem_tokens)} "
            f"held_s={held:.1f} running=empty -- this group's pool cannot admit the "
            f"request while it keeps the prefilled backlog for the other phase; "
            f"the front requeues it for the next phase of this group and flips"
        )


def is_intake_stall(text: object) -> bool:
    """Does a leg-1 error (status line + body) carry the stall refusal?"""
    return STALL_MARK in str(text or "")
