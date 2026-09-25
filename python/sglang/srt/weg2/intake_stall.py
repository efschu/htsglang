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

H91a (25.09.2026): the premise "P keeps the backlog's device rows" no longer
holds -- a finished prefill's rows are evictable (published to the host
arena / L3 before a demotion). The watch is now consulted only for a request
:func:`intake_verdict` calls impossible; a request that fits free + evictable
rows stays queued and is admitted in order (weg2/p_intake.py).
"""

from __future__ import annotations

from typing import Optional

#: The refusal's name; the front matches on it in the leg-1 error text.
STALL_MARK = "WEG2-INTAKE-STALL"
#: weg2xsn291: the request would not fit this group's pool even EMPTY --
#: requeueing it is a flip loop (133,837 tokens against PP0's 121,190:
#: stalled twice, the fairness switch flipped D->P 200 ms after D's wake,
#: W35/W68 on the legs, both groups dead). The front refuses it by name.
TOO_LARGE_MARK = "WEG2-INTAKE-TOO-LARGE"

#: Refused-with-empty-batch this long -> a stall, not a transient.
HOLD_S_DEFAULT = 1.0

#: H91a (25.09.2026) -- the three answers of :func:`intake_verdict`.
#: The free + evictable rows cover the request: it is admitted in order.
INTAKE_FITS = "fits"
#: Short right now, but a write-through / load-back in flight frees rows.
INTAKE_WAITS = "waits"
#: No row this phase can give covers it: the stall (503, requeue, flip).
INTAKE_IMPOSSIBLE = "impossible"


def intake_verdict(*, need_tokens: int, pool_tokens: int, free_tokens: int,
                   evictable_tokens: int, inflight: bool) -> str:
    """H91a: is a refused request the intake stall, or merely queued?

    Since P's finished prefills stay EVICTABLE (released into the tree at
    their finish, written back to the shared host arena on eviction and read
    by D from there or from the L3 file store), the prefilled backlog of a
    phase no longer holds P's pool. The stall is therefore only what no row
    of this phase can fund: a request larger than the whole pool, or one
    larger than free + evictable with nothing in flight that would free
    more (rows locked by something only the flip releases). Everything else
    waits in the queue and is admitted in order -- no 503, no extra flip
    (cu130 53572be2: need 42,442 against free 220,480 + evictable 41,664 was
    answered 503 after 61 s)."""
    need = int(need_tokens)
    if int(pool_tokens) > 0 and need > int(pool_tokens):
        return INTAKE_IMPOSSIBLE
    if need <= max(0, int(free_tokens)) + max(0, int(evictable_tokens)):
        return INTAKE_FITS
    if inflight:
        return INTAKE_WAITS
    return INTAKE_IMPOSSIBLE


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


def pending_prefetch_is_a_stall(*, need_tokens: int, pool_free_tokens: int,
                                running_empty: bool, waiting: int) -> bool:
    """weg2xsn297 (Task #13): a request whose store read cannot finish because
    its rows would not fit the FREE pool, with nothing running that could
    free any, is the intake stall -- not a slow read. xsn293: the seventh
    prompt sat in the prefetch-pending skip for 66 s (only the wedge path
    named it) while the loop spun at 45 passes/s over barlink. A slow but
    fundable read (room for it) is never flagged here."""
    if not running_empty or int(waiting) <= 0:
        return False
    return int(need_tokens) > int(pool_free_tokens)


def is_too_large(text: object) -> bool:
    """Does a leg-1 error say the request exceeds the group's pool outright?"""
    return TOO_LARGE_MARK in str(text or "")


def is_intake_stall(text: object) -> bool:
    """Does a leg-1 error (status line + body) carry the stall refusal?"""
    return STALL_MARK in str(text or "")
