"""DEGEN-STOP: stage 2 of the decode-tail repetition watch (degen_detect.py).

Metal 03.10. (27B dual y8w, D): rid weg2-0-58 looped one 210-token reasoning
pattern for 64000 tokens at ~110 tok/s; the seat and its growing KV pressed D
for minutes and P paused in a cascade (~3 min stall). Stage 1 only named it.

PATH. The detokenizer (where the detector runs, off the decode round) sends
``degen_stop_abort_req`` to the tokenizer manager over the socket every output
takes; the tokenizer manager forwards it to the scheduler with the same
dispatch as any abort (every TP rank gets it); the scheduler finishes the
running request with ``running_abort_finish`` = FINISH_LENGTH, so the client
sees a normal end of stream (finish_reason "length"), not an error.

Armed by SGLANG_WEG2_DEGEN_STOP (default OFF: the user's call).
"""

from __future__ import annotations


def degen_stop_abort_req(rid: str, part: str, period: int, reps: int, out_len: int):
    """The AbortReq that ends a looping request (detokenizer side)."""
    from sglang.srt.managers.io_struct import AbortReq

    return AbortReq(
        rid=rid,
        finished_reason={"type": "length", "length": int(out_len)},
        abort_message="DEGEN-STOP part=%s period=%d reps=%d out_len=%d"
        % (part, int(period), int(reps), int(out_len)),
        degen_stop=True,
    )


def running_abort_finish(recv_req):
    """The finish reason a running request gets from ``recv_req`` (scheduler
    side): a DEGEN-STOP abort finishes it with ``length``, every other abort
    stays FINISH_ABORT."""
    from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH

    fr = getattr(recv_req, "finished_reason", None)
    if getattr(recv_req, "degen_stop", False) and isinstance(fr, dict) and fr.get("type") == "length":
        return FINISH_LENGTH(length=int(fr.get("length") or 0))
    return FINISH_ABORT()
