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
