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

    def forget(self, rid: Optional[str]) -> None:
        """weg2xsn288: the request was aborted (the front's /abort_request
        after the refusal, or any abort). Its hold and its ``reported`` mark
        go with it: the SAME rid comes back in the next P phase (the front
        requeues it at the head) and must be refusable again, and a hold
        that began before an abort must never ride into the next phase --
        measured xsn288: PP2 named a stall held for 270.9 s across a sleep,
        a wake and the abort, and dropped the request alone. ``None`` (an
        abort_all) forgets everything. Prefix semantics as the abort's."""
        if rid is None:
            self.reset()
            return
        rid = str(rid)
        self._reported = {r for r in self._reported if not str(r).startswith(rid)}
        if self._rid is not None and str(self._rid).startswith(rid):
            self._rid = None
            self._since = 0.0

    def reset(self) -> None:
        """A new phase (the group woke): no hold, no reported rid survives."""
        self._rid = None
        self._since = 0.0
        self._reported = set()

    def observe(self, *, rid: str, need_tokens: int, rem_total_tokens: int,
                cur_rem_tokens: int, running_empty: bool, now: float,
                extra: str = "", immediate: bool = False) -> Optional[str]:
        """One refusal of ``rid`` with nothing admitted this pass -- the
        adder's NO_TOKEN, or (xsn273) the seat gate in front of the adder:
        ``get_num_allocatable_reqs(0) <= 0`` while the parked, prefilled
        backlog holds every request slot. Returns the refusal message the
        moment the stall is established (once per rid), else None.
        ``extra`` names the gate's own terms in the message."""
        if not running_empty:
            # a running batch will free tokens when it finishes: wait for it
            self._rid = None
            return None
        if rid in self._reported:
            return None
        if self._rid != rid:
            self._rid = rid
            self._since = float(now)
            if not immediate:
                return None
        held = float(now) - self._since
        if held < self.hold_s and not immediate:
            return None
        self._reported.add(rid)
        self.stalls += 1
        return (
            f"{STALL_MARK} rid={rid} need_tokens={int(need_tokens)} "
            f"rem_total_tokens={int(rem_total_tokens)} cur_rem_tokens={int(cur_rem_tokens)} "
            f"held_s={held:.1f} running=empty{(' ' + extra) if extra else ''} -- this "
            f"group's pool cannot admit the request while it keeps the prefilled "
            f"backlog for the other phase; the front requeues it for the next "
            f"phase of this group and flips"
        )


def abort_must_reach_every_rank(*, rid_known: bool, abort_all: bool) -> bool:
    """weg2xsn276: should the tokenizer dispatch an AbortReq it would
    otherwise drop (rid unknown to it)? On a Weg 2 group (env
    SGLANG_WEG2_GROUP set) yes: the intake-stall refusal finalises the
    stream before the front's abort arrives, and the PP followers still hold
    the request. Outside Weg 2 the upstream short-cut stands."""
    import os

    if rid_known or abort_all:
        return True
    return bool(str(os.environ.get("SGLANG_WEG2_GROUP", "")).strip())


def is_intake_stall(text: object) -> bool:
    """Does a leg-1 error (status line + body) carry the stall refusal?"""
    return STALL_MARK in str(text or "")
