"""SF SLOT FIDELITY (27B rc12k27 b23, 27.09. 10:27:56Z, rid weg2-68-235): on the
carrierless PP form every stage admits a request in the SAME pass and at the SAME
width, or none does.

THE DEATH. P log boot_weg2_dkr27bparkodirectdraftbar1w2309270957 (9417507cd2):
PP1 ``#1004 SLOT DISAGREEMENT: PP1 is launching slot 0 (fwd_ct=1582,
rids=[weg2-68-235], extend=512) but the upstream's proxy names slot 1 (rows=1024,
extent=('weg2-68-235', 40958, 41982))``. The told was 40958 on every rank, every
rank planned the head at 40958 (``P-CHUNK-POLICY ... start=40958 widths=[512x39,469]``),
the host read was complete on all three (``#905 ... completed=40958``). Then:

  PP0   WEG2-LOADBACK-EVICT kv_tokens=27095 floor=13035: the uniform floor is short by
        14060 ... this pass refuses (n=4)
  PP0   WEG2-LOADBACK-WAIT extent=40958 applied=0 ... the request waits
  PP1/2 #988 LOADBACK prefix moved to 40958 -> #969N ADMIT slot=0 extend=512
  PP0   (next pass) P-CHUNK-POLICY start=0 widths=[1024x59,979] pick=mid=1024
  PP0   H91 STORE-TOLD KEPT told=40958 -> #988 LOADBACK prefix moved to 40958
        -> #969N ADMIT slot=1 extend=1024

TERM (a) -- THE ROOM VERDICT WAS RANK-LOCAL. The "uniform floor" the load-back
decides from is the MIN over ``tp_cpu_group`` -- and on TP=1/PP=3 that group has one
member (boot line 1810: ``#788 UNIFORM-FLOOR SCOPE: tp_cpu_group world=1 -> floors
OFF ... the ranks that must agree are NOT in this reduce group``). The single-rank
path publishes THIS rank's ``available_size()`` (#1045), so the refuse-this-pass /
retry-next-pass rule built for a TP group's published MIN (xsn285) turns a local
shortfall into a PASS SKEW: PP0 was short, its peers were not, PP0 admitted one
pass later.

THE RULE (a). On a floor that is this rank's own value under pp > 1, the load-back
does what upstream's ``load_back`` does: evict what is short and load in the SAME
pass. The retry-next-pass exists so that a TP group re-reads a new MIN; a group of
one has nothing to re-read, so waiting a pass only buys the skew. Every rank then
admits in the same pass whenever free + evictable covers the host hit -- 218004
evictable against a 14060 shortfall on the specimen. Only when this rank cannot
hold the load-back even with every evictable row freed does it still refuse (the
residual, named ``SF LOADBACK-ROOM PP-RESIDUAL``; on this form a peer with room
admits in this pass, so that line is the cause of the ``#1004`` that follows).

TERM (b) -- THE CHUNK PLAN SURVIVED THE PREFIX MOVE. #1400's ``admission`` pops
``_weg2_store_told[rid]`` on the first visit; H91 keeps the verdict in
``_weg2_told_kept``. Fix A's ``budget_head`` read only the told map, so PP0's second
visit sized the pass from position 0 (``pos_src=zero``) -- a plan for start=0 whose
width (1024) then ran on an extent that #988 moved to 40958, where the plan says 512.

THE RULE (b). (b1) ``budget_head`` reads the KEPT told when the told map no longer
has the rid (same request object): every rank keeps the same verdict the same way,
so the position stays rank-identical. (b2) Whenever #988 moves the prefix of the
request the pass budget was planned for to a position OTHER than the planned one,
the plan is replanned at the moved prefix and the pass budget NARROWED to it (never
widened: the corridor granted the old width, not a wider one). Deterministic in the
planner's call sequence, which is the same on every rank when the move is.

NOT BUILT: followers taking PP0's width from PP0's row. On this form
(``pp_row_carrier_present`` False: no ``pp_flip_counters`` side channel, followers
log ``#1460 FOLLOWER-GATE gate=None``) PP0's row reaches a follower only with the
proxy of the pass it already admitted (Fix B, unbuilt). The rule here is the other
half of the same law: every input of the width and of the pass is rank-identical,
and ``#1004`` / ``#1233 W27`` stay the named stops if they ever are not.

Switch ``SGLANG_WEG2_PP_SLOT_FIDELITY``, default on; ``0`` = the old behaviour byte
for byte (no flag is written, no hook installed, no extra lookup).
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_PP_SLOT_FIDELITY"
#: tree attribute: True while this iteration's evict floor is THIS rank's own
#: ``available_size()`` on a multi-stage PP form (tp group of one).
FLOOR_LOCAL_PP_ATTR = "weg2_sf_floor_local_pp"
#: scheduler attribute: the plan this pass's budget was sized from (cleared at
#: the top of every budget sizing, consumed by the adder's move hook).
PLAN_ATTR = "_weg2_sf_budget_plan"
#: adder attribute: the chunk budget the adder was built with.
ADDER_INITIAL_ATTR = "sf_rem_chunk_initial"

_LOG_FIRST = 8


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _sampled(holder, attr: str) -> Optional[int]:
    """Counter on ``holder``; returns n when this occurrence prints (first 8,
    then powers of two), else None. Never raises."""
    try:
        n = int(getattr(holder, attr, 0) or 0) + 1
        setattr(holder, attr, n)
    except Exception:  # noqa: BLE001 -- a slotted double
        return None
    return n if (n <= _LOG_FIRST or (n & (n - 1)) == 0) else None


# ---------------------------------------------------------------------------
# (a) the floor's scope, published by the scheduler
# ---------------------------------------------------------------------------


def mark_floor_scope(tree, local_pp: bool) -> None:
    """Record on the tree whether this iteration's evict floor is this rank's own
    value on a multi-stage PP form. Switch off: nothing is written."""
    if tree is None or not enabled():
        return
    try:
        setattr(tree, FLOOR_LOCAL_PP_ATTR, bool(local_pp))
    except Exception:  # noqa: BLE001 -- a slotted double
        pass


def pp_size_of(holder) -> int:
    args = getattr(holder, "server_args", None)
    try:
        return int(getattr(args, "pp_size", 1) or 1) if args is not None else 1
    except (TypeError, ValueError):
        return 1


def local_pp_room(tree, kv_tokens: int, floor: int, rid=None) -> Optional[bool]:
    """The load-back room verdict on a local floor under pp > 1.

    None  -- not this form (switch off, a group floor, pp == 1): the caller keeps
             its path unchanged.
    True  -- room is there now (after evicting the shortfall): load in THIS pass.
    False -- the residual: even every evictable row does not make room; the
             caller refuses as before."""
    if not enabled() or not getattr(tree, FLOOR_LOCAL_PP_ATTR, False):
        return None
    alloc = getattr(tree, "token_to_kv_pool_allocator", None)
    if alloc is None:
        return None
    kv_tokens = int(kv_tokens)
    avail0 = int(alloc.available_size())
    evictable = int(tree.evictable_size())
    evicted = 0
    short = kv_tokens - avail0
    if short > 0 and evictable > 0:
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        res = tree.evict(EvictParams(num_tokens=min(short, evictable)))
        evicted = int(getattr(res, "num_tokens_evicted", 0) or 0)
    avail1 = int(alloc.available_size())
    ok = avail1 >= kv_tokens
    if ok:
        n = _sampled(tree, "_weg2_sf_room_same_pass")
        if n is not None:
            logger.info(
                "SF LOADBACK-ROOM SAME-PASS rid=%s kv_tokens=%d floor=%d avail=%d "
                "evictable=%d evicted=%d avail_after=%d (n=%d): the floor is this "
                "rank's own value (tp group of one, pp>1), so the shortfall is "
                "evicted and the host hit loads in THIS pass -- a next-pass retry "
                "would only skew this stage one pass behind its peers (#1004, b23)",
                rid, kv_tokens, int(floor), avail0, evictable, evicted, avail1, n,
            )
    else:
        n = _sampled(tree, "_weg2_sf_room_residual")
        if n is not None:
            logger.warning(
                "SF LOADBACK-ROOM PP-RESIDUAL rid=%s kv_tokens=%d floor=%d avail=%d "
                "evictable=%d evicted=%d avail_after=%d (n=%d): this rank cannot hold "
                "the load-back even with every evictable row freed and refuses this "
                "pass. The verdict is RANK-LOCAL on a carrierless PP form: a peer "
                "with room admits in this pass, and if one does the #1004 SLOT "
                "DISAGREEMENT that follows has THIS line as its cause",
                rid, kv_tokens, int(floor), avail0, evictable, evicted, avail1, n,
            )
    return ok


# ---------------------------------------------------------------------------
# (b1) the budget head's position from the KEPT told
# ---------------------------------------------------------------------------


def kept_told(scheduler, req) -> Optional[int]:
    """The H91 kept told of ``req`` (same request object), or None. Switch off:
    None, the lookup is not made."""
    if not enabled():
        return None
    kept = getattr(scheduler, "_weg2_told_kept", None)
    if not kept:
        return None
    entry = kept.get(str(getattr(req, "rid", "")))
    if entry is None or getattr(entry, "req", None) is not req:
        return None
    try:
        return int(entry.told)
    except (TypeError, ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# (b2) replan at the prefix move
# ---------------------------------------------------------------------------


class BudgetPlan(NamedTuple):
    rid: str
    pos: int
    end: int
    src: str
    width: int


def note_budget(scheduler, head, width: int) -> None:
    """Remember the position/width this pass's budget was sized from."""
    if not enabled():
        return
    try:
        setattr(scheduler, PLAN_ATTR, BudgetPlan(
            str(getattr(head.req, "rid", "")), int(head.pos), int(head.end),
            str(head.src), int(width)))
    except Exception:  # noqa: BLE001 -- an instrument never stops a pass
        pass


def clear_budget(scheduler) -> None:
    """Start of a pass's budget sizing: no plan is carried over from an earlier
    pass (a static / layer-split pass notes none). Switch off: nothing."""
    if not enabled():
        return
    try:
        setattr(scheduler, PLAN_ATTR, None)
    except Exception:  # noqa: BLE001
        pass


def replan_hook(scheduler, adder):
    """The #988 move hook for this pass's adder, or None (switch off, no plan
    policy, no plan noted this pass): ``hook(req, new_prefix_len)``. The plan is
    CONSUMED here -- bound into this adder's hook, gone from the scheduler -- so
    it never serves a second adder."""
    if not enabled() or getattr(scheduler, "_p_chunk_planner", None) is None:
        return None
    plan = getattr(scheduler, PLAN_ATTR, None)
    if plan is None:
        return None
    try:
        setattr(scheduler, PLAN_ATTR, None)
        setattr(adder, ADDER_INITIAL_ATTR, adder.rem_chunk_tokens)
    except Exception:  # noqa: BLE001
        return None
    box = [plan]

    def _hook(req, new_pos):
        return replan_after_move(scheduler, adder, req, new_pos, box)

    return _hook


def replan_after_move(scheduler, adder, req, new_pos: int, box) -> Optional[int]:
    """#988 moved ``req``'s prefix to ``new_pos``. If the pass budget was planned
    for this request at a different position, replan at ``new_pos`` and narrow
    the adder's chunk budget to the new plan's width. ``box`` holds this pass's
    plan. Returns the new budget, or None when nothing changed. Never raises."""
    try:
        plan = box[0]
        rid = str(getattr(req, "rid", ""))
        if plan is None or plan.rid != rid or plan.src == "chunked":
            return None
        new_pos = int(new_pos)
        if new_pos == plan.pos:
            return None
        rem = getattr(adder, "rem_chunk_tokens", None)
        initial = getattr(adder, ADDER_INITIAL_ATTR, None)
        if rem is None or initial is None:
            return None
        if getattr(adder, "can_run_list", None) or int(rem) != int(initial):
            n = _sampled(scheduler, "_weg2_sf_replan_skipped")
            if n is not None:
                logger.warning(
                    "SF P-CHUNK REPLAN-AT-MOVE SKIPPED rid=%s planned_start=%d moved_to=%d "
                    "rem_chunk=%s initial=%s can_run=%d (n=%d): the pass budget was "
                    "already spent on another request; the next pass replans at the "
                    "executed position",
                    rid[:16], plan.pos, new_pos, rem, initial,
                    len(getattr(adder, "can_run_list", None) or ()), n,
                )
            return None
        from sglang.srt.weg2.p_chunk_policy import forward_budget

        planner = getattr(scheduler, "_p_chunk_planner", None)
        if planner is None:
            return None
        width = int(forward_budget(planner, req.rid, new_pos, plan.end, queued=True))
        cap = int(getattr(scheduler, "chunked_prefill_size", 0) or 0)
        if cap > 0:
            width = min(width, cap)
        new_budget = min(width, int(initial)) if width > 0 else int(initial)
        adder.rem_chunk_tokens = new_budget
        box[0] = plan._replace(pos=new_pos, width=new_budget)
        n = _sampled(scheduler, "_weg2_sf_replan_n")
        if n is not None:
            logger.info(
                "SF P-CHUNK REPLAN-AT-MOVE rid=%s planned_start=%d (src=%s) moved_to=%d "
                "width %d -> %d (plan %d, never widened) (n=%d): the pass budget was "
                "planned for another start; #988 moved the prefix, so the plan is "
                "redone at the moved prefix on every rank alike",
                rid[:16], plan.pos, plan.src, new_pos, int(initial), new_budget, width, n,
            )
        return new_budget
    except Exception as exc:  # noqa: BLE001 -- a plan must never stop a pass
        n = _sampled(scheduler, "_weg2_sf_replan_err")
        if n is not None:
            logger.warning("SF P-CHUNK REPLAN-AT-MOVE failed (n=%d): %r", n, exc)
        return None
