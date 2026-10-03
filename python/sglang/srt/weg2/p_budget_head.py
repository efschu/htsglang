"""Fix A (27.09.2026): the P pass budget is sized from RANK-IDENTICAL data only.

THE DEATH. Boot dkr27breleasedraftbar1w109270737 (2d680cbe66), P, 08:14:59: PP1 refused
``#1233 W27 PP WIDTH DIVERGENCE: 512 row(s) for a batch of 1024 token(s)``. PP0 and PP1 had
admitted the SAME request (weg2-50-185, extend from 0) in the same pass (fwd_ct 563) -- PP0 with
512 tokens, PP1 with 1024. The pass budget (``Scheduler._p_chunk_policy_width`` /
``_p_layer_split_width``) was planned for the HEAD of the waiting queue, weg2-50-184, a released TW
fork twin whose store read was still running (not admissible on any rank this pass). Its position
``len(prefix_indices)`` differed by rank: PP0 had already written the twin's device match into it at
``#TW TWIN-REGISTER ... head=21246`` (pos off the plan -> replan without ramp -> 512), the followers
hold the twin unregistered until PP0's told arrives (pos 0 -> the plan's 1024). The next admissible
request, -185, got that rank-local budget.

THE RULE (#791: every rank derives its admission locally, so every INPUT must be rank-identical):
  * the in-flight chunked request: ``len(prefix_indices)`` -- it is running on every rank with the
    same extents (the congruence guard checks them each pass);
  * a request still in the waiting queue: its PUBLISHED told (``scheduler._weg2_store_told``, which
    every rank holds after the same absorb, #1400/#1416e), else 0 -- never its local registration
    match, which only PP0 has written (TW) or which a rank-local prefetch may have moved.

The cost: the first pass of a new head may run a width from the plan at 0 instead of its real
position; the next pass replans at the executed position (identical on every rank). A quality
question, never a disagreement.

INSTRUMENT: one ``P-CHUNK-BUDGET`` line per new queued head per rank, with the position the old code
would have used (``local_prefix``) beside the one used now -- a PP0/follower difference reads off
the lines directly.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple, Optional

logger = logging.getLogger(__name__)

MARKER = "P-CHUNK-BUDGET"


class BudgetHead(NamedTuple):
    req: Any
    pos: int
    end: int
    src: str          # "chunked" | "told" | "kept" (SF) | "zero"
    local_prefix: int  # len(prefix_indices) on THIS rank (what the pre-fix code used)


def _end_of(req) -> int:
    fill = getattr(req, "full_untruncated_fill_ids", None)
    if fill is not None and len(fill):
        return len(fill)
    return len(req.origin_input_ids) + len(getattr(req, "output_ids", ()) or ())


def _local_prefix(req) -> int:
    prefix = getattr(req, "prefix_indices", None)
    return 0 if prefix is None else len(prefix)


def budget_head(scheduler) -> Optional[BudgetHead]:
    """The request the pass budget is planned for, and its rank-identical
    position; None when there is nothing to serve."""
    req = getattr(scheduler, "chunked_req", None)
    if req is not None:
        local = _local_prefix(req)
        return BudgetHead(req, local, _end_of(req), "chunked", local)
    queue = getattr(scheduler, "waiting_queue", None) or []
    if not queue:
        return None
    req = queue[0]
    end = _end_of(req)
    local = _local_prefix(req)
    told_map = getattr(scheduler, "_weg2_store_told", None) or {}
    rid = str(getattr(req, "rid", ""))
    if rid in told_map:
        try:
            told = int(told_map[rid])
        except (TypeError, ValueError):
            told = 0
        pos = max(0, min(told, max(end - 1, 0)))
        return BudgetHead(req, pos, end, "told", local)
    # SF (b23 #1004): #1400's admission pops the told on the first visit and H91
    # keeps the verdict (_weg2_told_kept). A head that visited without admitting
    # sizes the pass from its KEPT told -- kept the same way on every rank -- not
    # from 0 (PP0 planned start=0 -> 1024 on an extent #988 moved to 40958).
    from sglang.srt.weg2 import pp_slot_fidelity as _sf

    kept = _sf.kept_told(scheduler, req)
    if kept is not None:
        pos = max(0, min(int(kept), max(end - 1, 0)))
        return BudgetHead(req, pos, end, "kept", local)
    return BudgetHead(req, 0, end, "zero", local)


def note_new_head(scheduler, head: BudgetHead, width: int, where: str) -> None:
    """One ``P-CHUNK-BUDGET`` line per new queued head on this rank. Never raises."""
    try:
        if head.src == "chunked":
            return
        rid = str(getattr(head.req, "rid", ""))
        attr = "_p_budget_last_head_" + where
        if getattr(scheduler, attr, None) == rid:
            return
        setattr(scheduler, attr, rid)
        pp = getattr(scheduler, "pp_rank", "?")
        logger.info(
            "%s where=%s pp_rank=%s head=%s pos=%d pos_src=%s end=%d width=%d local_prefix=%d "
            "(Fix A: a queued head sizes the pass from rank-identical data -- its published told or 0; "
            "local_prefix is the rank-local position the pre-fix budget used)",
            MARKER, where, pp, rid[:12], head.pos, head.src, head.end, int(width), head.local_prefix,
        )
    except Exception:  # noqa: BLE001 -- an instrument never stops a pass
        pass
