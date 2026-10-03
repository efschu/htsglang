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
