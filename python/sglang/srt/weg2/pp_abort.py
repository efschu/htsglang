"""xsn324 (18.09.2026): an abort of the IN-FLIGHT chunked request on a PP
group must stop the chunk pipeline at the SAME chunk on every stage.

PP0 receives the AbortReq from the tokenizer at the start of its pass and
forwards it on the request wire; stage r reads it r passes later. A stage
that clears ``chunked_req`` at the pass it receives the abort stops
launching chunks while the stage behind it -- which has not read the abort
yet and continues the chunk on its own (the chunk continuation is
rank-local, only the admission rides the wire) -- has already scheduled the
next chunk and waits for proxy tensors that never come. xsn324: PP0 stopped
at receipt, PP1 waited in ``_pp_recv_proxy_tensors``, PP2 behind it, PP0 in
the request-wire send: the ring stood, the flip's flush_cache never
returned.

The rule: stage r applies the chunked abort ``pp_size - 1 - r`` passes after
it received the AbortReq. Then every stage launches the same chunks
(k .. k + pp_size - 2) and stops before the same one. Pure module, desk-testable.

#791C ON THE FORM WITHOUT THE ROW AUTHORITY (NF z30w-park, 29.09. 09:11Z,
weg2-112-205, 111k prefill in 16k chunks, PP3): the premise above holds in
WALL time only. Without the row authority every follower receives PP0's
request list once per pass, in the pass of the SAME index (blocking recv at
the top of the pass; PP0 forwards before it plans), and continues the chunk
rank-locally in lockstep -- so every stage reads the AbortReq right before it
plans the SAME chunk. Measured: PP0 09:11:10, PP1 :13, PP2 :16, each in slot 2
of the lap, each right before 69888. The static delays 2/1/0 then gave three
launch sets -- PP0 69888 and 86272, PP1 69888, PP2 none -- and the ring stood
(PP0 in recv_object[src=2] awaiting the output of chunks PP2 never ran, PP1 in
recv_object[src=0]). On that form the wire position of the AbortReq IS PP0's
decision, and the rule is delay 0 on every stage: PP0 launches no chunk after
it, the followers drain only what PP0 already sent. Under the row authority
(27B Fix B) the xsn324 delays stay byte for byte: PP0 keeps its countdown and
the followers follow PP0's forwarded schedule (``follower_row_verdict``).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def chunked_abort_delay(pp_size: int, pp_rank: int, row_authority=None) -> int:
    """Passes stage ``pp_rank`` keeps launching chunks after it received an
    abort for its in-flight chunked request. 0 on a non-PP engine and on the
    last stage; ``pp_size - 1`` on PP0.

    ``row_authority=False`` (a PP group WITHOUT the #631 row authority: the
    request wire is pass-aligned, see the module docstring): 0 on EVERY stage
    -- all stages read the abort before planning the same chunk, so applying
    it at receipt stops them at the same chunk. ``True`` or ``None`` (not
    known): the xsn324 delays, unchanged."""
    n = int(pp_size or 1)
    if n <= 1 or row_authority is False:
        return 0
    return max(0, n - 1 - int(pp_rank or 0))


def row_authority_of(scheduler):
    """#791C-NF: does the #631 row authority apply on this rank's PP group?
    ``True``/``False``; ``None`` on a non-PP engine or when it cannot be read
    -- then the xsn324 rule stands (``chunked_abort_delay``), never a guess."""
    try:
        if int(getattr(getattr(scheduler, "ps", None), "pp_size", 1) or 1) <= 1:
            return None
        from sglang.srt.weg2 import p_row_authority

        return bool(p_row_authority.applies(scheduler))
    except Exception:  # noqa: BLE001 - an unreadable form never blocks the abort
        return None


def countdown_step(delay: int) -> tuple:
    """One scheduling pass: returns (apply_now, remaining)."""
    d = int(delay or 0)
    if d <= 0:
        return True, 0
    return False, d - 1


def follower_row_verdict(pp_rank: int, row_authority: bool, scheduled_extents, rid: str,
                         pp0_drained: bool = False):
    """#791C (27B rc12z24 bb84760576, 15:59:23Z, weg2-0-6): under the #631 row
    authority (Fix B) the xsn324 premise "stage r reads the abort r passes
    later" is false -- the AbortReq rides the request chain and reached PP0,
    PP1 and PP2 in the same second, while the stages were 0/1/2 FRAMES behind
    PP0. The static delays (2/1/0) let PP0 launch two more chunks (193, 194)
    that PP1 (dropped before 194) and PP2 (dropped before 192) could no longer
    execute: #791 FORWARDED SCHEDULE UNEXECUTABLE on both followers.

    A follower under row authority therefore takes the abort off PP0's
    DECISION, the forwarded schedule it received for THIS pass (no new
    collective, no local count):

      ``None``  -- not this form (PP0, or no row authority): the xsn324
                   countdown stays the rule.
      ``False`` -- keep the chunk: the schedule still names ``rid`` (PP0
                   launched this chunk; its hidden states are on the wire), or
                   there is no frame this pass (None / {} -- the plan bypass
                   admits nothing, so keeping it cannot add a request).
      ``True``  -- apply now: PP0's schedule for this pass names other rids
                   but not ``rid`` -- PP0 has applied its abort; keeping the
                   chunk would add a request the decision does not name (the
                   other #791 refusal)."""
    if int(pp_rank or 0) <= 0 or not row_authority:
        return None
    if pp0_drained:
        # #791C liveness: PP0 voted idle in a #1268 lap -- it applied its own
        # abort and every pass it launched completed the ring, so every frame
        # that named ``rid`` has been executed here. The group-uniform release
        # when PP0 sends no further frame (idle queue, quiesce before a flip).
        return True
    if not scheduled_extents:
        return False
    return str(rid) not in scheduled_extents


def pp0_idle_in_vote(slots) -> bool:
    """#791C liveness: did PP0 attach an IDLE slot to this #1268 idle vote?
    ``slots`` are ``(rank, idle 0/1, blockers)`` as ``attach_slot`` writes them;
    PP0 attaches its own slot before the vote leaves it (``_weg2_vote_maybe_stamp``)."""
    try:
        return any(int(r) == 0 and int(i) == 1 for r, i, *_ in (slots or ()))
    except Exception:  # noqa: BLE001 - an unreadable slot is not an idle PP0
        return False


# ---------------------------------------------------------------------------
# K14 (NF h16p 15.10.2026 21:52:38Z, weg2-164-2265): a WAITING-QUEUE abort on a
# follower is PP0's group decision, and it rides the request chain with PP0's
# verdict on it.
#
# THE DEATH. PP0 admitted weg2-164-2265 (61001 tokens, 265 to prefill) into
# slot 2 (#969N ADMIT fwd_ct=346) and launched it. One pass later the front's
# H102 client-gone abort reached PP0, which forwarded it down the chain
# (#1460 CTRL-FWD) and set to_finish on its in-flight request. PP1 and PP2 still
# held the rid in their WAITING queues -- the slot-2 frame had not been consumed
# yet -- and ``_abort_request_now`` popped it at receipt. The slot-2 frame then
# named a rid PP1 could no longer locate: #631 ROW-PROBE DEFER x3, then
# PpRowDeferCapExceeded (#1180), RANK_EXCEPTION, the group died. The abort was
# applied rank-locally at whatever pass it landed on (the #801 marker said so
# in so many words) while PP0 kept the rid in the group's plan: RAENGE NIE
# UNEINS, broken by the abort path itself.
#
# THE FORM. Only PP0 knows whether a frame it has sent (or launched) names the
# rid: it does iff the rid is in PP0's in-flight set at the moment PP0 reads the
# abort -- admission takes a request out of PP0's waiting queue and into a
# launched microbatch, and a request PP0 has already finished was finished on
# the ring's return, i.e. after every follower executed its frame. So PP0 stamps
# the verdict on the AbortReq it forwards (``pp0_stamp_aborts``, the CTRL-FWD
# site, before the send; the same "PP0 decides, the chain distributes" form as
# z30j's flush verdict). A follower under the row authority then:
#
#   * stamp False / no stamp  -> applies the abort at receipt (today's path, and
#     correct: no frame names the rid, none ever will);
#   * stamp True and the rid still in this follower's waiting queue -> HOLDS the
#     whole abort (``hold_waiting_abort``): no pop, no told/ledger forget, no
#     side effect at all, so the frame finds the request exactly as PP0 planned
#     it. When PP0's frame admits it, the held abort is applied right after that
#     plan (``resolve_waiting_aborts``), through the ordinary path: an in-flight
#     request gets to_finish (as on PP0), a chunked one goes to the #791C
#     follower verdict (PP0's schedule decides its last chunk). The request runs
#     exactly the passes PP0 runs and is finished where PP0 finishes it -- no
#     extra pass (no KV leak), no missing one (no #1180).
#
# No timer, no lap count, no switch: the hold ends on PP0's frame, or -- the
# #791C liveness form -- when PP0 votes idle in a #1268 lap while the rid is
# still queued here (then no frame can name it any more; pop it, by name).
# ---------------------------------------------------------------------------

#: scheduler attribute: rid -> the held AbortReq (follower only).
HELD_ATTR = "_weg2_pp_held_waiting_aborts"


def _matches(rid, target: str, abort_all: bool) -> bool:
    return bool(abort_all) or str(rid or "").startswith(str(target or ""))


def pp0_inflight_reqs(scheduler):
    """Every request a frame PP0 sent names and whose pass has not come back:
    the batches of ``_pp_launched_batches`` (set at the launch on every rank,
    popped when the slot's result is consumed -- #1225) plus the chunked
    request (its next chunk's frame is still PP0's to send). NOT ``mbs`` /
    ``last_batch``: both keep a COMPLETED batch until the slot is planned
    again, and a rid that ran everywhere and was requeued everywhere (#968b
    requeue, void park, retreat) would read admitted -- the followers would
    hold an abort no frame ever ends (nf-review-k14-1016 A1). ``getattr``
    throughout -- a stand-in answers what it has."""
    seen, out = set(), []

    def _add(r):
        if r is not None and id(r) not in seen:
            seen.add(id(r))
            out.append(r)

    _add(getattr(scheduler, "chunked_req", None))
    for b in list((getattr(scheduler, "_pp_launched_batches", None) or {}).values()):
        for r in (getattr(b, "reqs", None) or ()) if b is not None else ():
            _add(r)
    return out


def pp0_admitted(scheduler, rid: str, abort_all: bool = False) -> bool:
    """K14: has PP0 admitted (a frame named, or will name) a request this abort
    targets? True iff one is in PP0's in-flight set."""
    return any(_matches(getattr(r, "rid", None), rid, abort_all)
               for r in pp0_inflight_reqs(scheduler))


def pp0_stamp_aborts(scheduler, wire_reqs):
    """K14, PP0 only, at the CTRL-FWD site BEFORE the send: stamp every
    ``AbortReq`` it forwards with ``weg2_pp0_admitted``. Off the row authority
    (pass-aligned wire, non-PP, unreadable) nothing is stamped -- the follower
    then keeps today's at-receipt path. Returns ``wire_reqs`` (stamped in place)."""
    try:
        ps = getattr(scheduler, "ps", None)
        if int(getattr(ps, "pp_rank", 0) or 0) != 0 or row_authority_of(scheduler) is not True:
            return wire_reqs
        for r in wire_reqs or ():
            if type(r).__name__ != "AbortReq" or not hasattr(r, "weg2_pp0_admitted"):
                continue
            r.weg2_pp0_admitted = bool(pp0_admitted(
                scheduler, getattr(r, "rid", "") or "", bool(getattr(r, "abort_all", False))))
            logger.info(
                "WEG2-PP-ABORT-VERDICT rid=%s abort_all=%s pp0_admitted=%s: PP0's group "
                "decision rides the chain -- %s (K14)",
                getattr(r, "rid", None), getattr(r, "abort_all", None), r.weg2_pp0_admitted,
                "a frame names it; followers that still queue it hold the abort until "
                "that frame admits it" if r.weg2_pp0_admitted
                else "no frame names it; every stage applies it at receipt")
    except Exception:  # noqa: BLE001 - an unstampable abort keeps today's path
        logger.warning("WEG2-PP-ABORT-VERDICT stamp failed (abort forwarded unstamped)",
                       exc_info=True)
    return wire_reqs


def hold_waiting_abort(scheduler, recv_req) -> bool:
    """K14, follower side, at ``abort_request`` BEFORE any side effect: True =
    the abort is held (PP0 stamped admitted, the row authority applies here, and
    this follower still queues the request). Never on PP0, non-PP, abort_all or
    the pass-aligned form; an unstamped abort is never held."""
    try:
        ps = getattr(scheduler, "ps", None)
        if int(getattr(ps, "pp_size", 1) or 1) <= 1 or int(getattr(ps, "pp_rank", 0) or 0) <= 0:
            return False
        if getattr(recv_req, "abort_all", False):
            return False
        rid = str(getattr(recv_req, "rid", "") or "")
        if not rid:
            return False
        open_holds = [k for k in (getattr(scheduler, HELD_ATTR, None) or {})
                      if _matches(k, rid, False)]
        if open_holds and unadmitted_here(scheduler, rid):
            # A second abort of a rid whose hold is open JOINS the hold
            # (nf-review-k14-1016 defence): the first abort already carried
            # PP0's "a frame names it"; popping now on a later stamp would
            # reopen #1180 for the frame still on its way.
            logger.info(
                "WEG2-PP-WAITING-ABORT joined rid=%s pp_rank=%s stamp=%s: a hold for it is "
                "open -- this abort is settled with it (K14)",
                rid, getattr(ps, "pp_rank", "?"), getattr(recv_req, "weg2_pp0_admitted", None))
            return True
        if getattr(recv_req, "weg2_pp0_admitted", None) is not True:
            return False
        if row_authority_of(scheduler) is False:
            return False
        queued = unadmitted_here(scheduler, rid)
        if not queued:
            return False
    except Exception:  # noqa: BLE001 - an unreadable state keeps today's path
        return False
    held = scheduler.__dict__.setdefault(HELD_ATTR, {})
    held[rid] = recv_req
    logger.info(
        "WEG2-PP-WAITING-ABORT held rid=%s pp_rank=%s queued=%d: PP0 admitted it (its "
        "frame names it), this follower has not executed that frame yet -- the abort is "
        "applied right after the plan that admits it, never at receipt (K14)",
        rid, getattr(ps, "pp_rank", "?"), len(queued))
    return True


def unadmitted_here(scheduler, rid: str):
    """K14: the request objects this rank holds for ``rid`` that no plan has
    admitted yet -- the waiting queue and the two queues beside it that an
    abort also reaches (#1445 dormant hold, H91b D park). A request that only
    moved between them is still PP0's to admit; only admission (or a group
    reset) ends the hold."""
    out = []
    for name in ("waiting_queue", "weg2_dormant_hold", "weg2_d_parked"):
        for r in getattr(scheduler, name, None) or ():
            if _matches(getattr(r, "rid", None), rid, False):
                out.append(r)
    return out


def waiting_abort_verdict(still_queued: bool, pp0_drained: bool) -> str:
    """K14, per held rid and plan: ``keep`` (PP0's frame has not admitted it
    yet), ``apply`` (it left the waiting queue -- admitted by PP0's frame, or
    removed by a group reset -- apply the abort through the ordinary path),
    ``pop`` (still queued but PP0 voted idle: no frame can name it any more)."""
    if not still_queued:
        return "apply"
    return "pop" if pp0_drained else "keep"


def resolve_waiting_aborts(scheduler, apply, pp0_drained: bool = False) -> int:
    """K14: settle the held waiting-queue aborts. Called right after every plan
    of a PP pass (before the launch), at the top of ``get_next_batch_to_run``
    (``process_pending_chunked_abort``) and on PP0's idle vote. ``apply(recv_req)``
    is the ordinary abort (``Scheduler._abort_request_apply``). Returns how many
    were applied."""
    held = getattr(scheduler, HELD_ATTR, None)
    if not held:
        return 0
    n = 0
    for rid, recv_req in list(held.items()):
        still = bool(unadmitted_here(scheduler, rid))
        verdict = waiting_abort_verdict(still, pp0_drained)
        if verdict == "keep":
            continue
        del held[rid]
        logger.info(
            "WEG2-PP-WAITING-ABORT %s rid=%s pp_rank=%s: %s (K14)", verdict, rid,
            getattr(getattr(scheduler, "ps", None), "pp_rank", "?"),
            "PP0's frame admitted it -- applied through the ordinary path (in flight: "
            "to_finish as on PP0; chunked: the #791C follower verdict)" if verdict == "apply"
            else "PP0 voted idle with the rid still queued here -- no frame can name it any "
            "more, popped (liveness)")
        apply(recv_req)
        n += 1
    return n
