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
"""
from __future__ import annotations


def chunked_abort_delay(pp_size: int, pp_rank: int) -> int:
    """Passes stage ``pp_rank`` keeps launching chunks after it received an
    abort for its in-flight chunked request. 0 on a non-PP engine and on the
    last stage; ``pp_size - 1`` on PP0."""
    n = int(pp_size or 1)
    if n <= 1:
        return 0
    return max(0, n - 1 - int(pp_rank or 0))


def countdown_step(delay: int) -> tuple:
    """One scheduling pass: returns (apply_now, remaining)."""
    d = int(delay or 0)
    if d <= 0:
        return True, 0
    return False, d - 1


def follower_row_verdict(pp_rank: int, row_authority: bool, scheduled_extents, rid: str):
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
    if not scheduled_extents:
        return False
    return str(rid) not in scheduled_extents
