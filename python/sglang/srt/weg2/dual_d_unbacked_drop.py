"""#1390f UNBACKED-DROP (y9d4d/B9, desk analysis deskq/done/1390-p-kv-wait-stall-y9d4d.md Fix 2).

B9 (PKVWAIT-INSTR marker=cache_yield): dev_unbacked=2 (34905 tokens), ev_after unchanged -- with the shared
host arena FULL the backup of a write_back leaf is refused (BACKUP-REFUSED arena_claim), the leaf stays on the
card, ``pp_slot_fidelity.unbacked_drop_allowed`` is False on a TP group, D hands no VRAM to the waiting P.

THE ORDER: while P waits for its card past ``SGLANG_WEG2_DUAL_D_UNBACKED_DROP_WAIT_S`` AND the group has no
demand AND no hold AND the arena refused a claim recently, D's cache yield may DROP an unbacked childless
unlocked leaf (cache content, recomputable) instead of leaving it. Dual D only (``dual_d_kv_stage.armed``),
default OFF (``SGLANG_WEG2_DUAL_D_UNBACKED_DROP_ON_WAIT``): off = the old code path byte for byte.

RANK AGREEMENT (a rank-local divergence is crash-stop class): the proposal of one tick rides the NEXT tick's
group collective as one more element (MAX over the D ranks); every rank applies the same group value, all
inputs of ``propose`` are group values of the collective (p_wait_s, demand, holds, arena need). Only when the
switch is on does the collective carry the extra element (the switch is one profile ENV on every rank).

NEVER dropped: ``holds`` (a parked / W50-midstream / D-HOLD-FOR-GROW request: ``d_holds`` = group MAX) blocks the
whole order; a node with children, with a lock, or with a write-through in flight is not dropped either."""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

MARK = "#1390f UNBACKED-DROP"
ENV_NAME = "SGLANG_WEG2_DUAL_D_UNBACKED_DROP_ON_WAIT"
WAIT_ENV_NAME = "SGLANG_WEG2_DUAL_D_UNBACKED_DROP_WAIT_S"
#: the tree attribute the D tick sets around ONE ``tree.evict`` of the order
ORDER_ATTR = "_weg2_dual_d_drop_order"
STATS_ATTR = "_weg2_dual_d_drop_stats"
#: an arena refusal (group-uniform ``arena_need`` > 0 on the tick collective) counts as "arena full" this long
ARENA_RECENT_S = 30.0
LOG_MIN_GAP_S = 5.0


def switch_on(env=None) -> bool:
    """Dual D layout (armed) AND the switch (default OFF)."""
    e = os.environ if env is None else env
    if str(e.get(ENV_NAME, "0")).strip().lower() not in ("1", "true", "yes", "y", "on"):
        return False
    from sglang.srt.weg2 import dual_d_kv_stage as _dk

    return _dk.armed(e)


def wait_s(env=None) -> float:
    e = os.environ if env is None else env
    try:
        return float(e.get(WAIT_ENV_NAME, "8.0"))
    except ValueError:
        return 8.0


def propose(p_waiting: bool, p_wait_s: float, demand_group: int, holds: bool, arena_recent: bool,
            threshold_s: float) -> bool:
    """One rank's proposal from GROUP values only. holds / demand / no wait / arena not full = no."""
    return bool(p_waiting and float(p_wait_s) >= float(threshold_s) and int(demand_group) <= 0
                and not holds and arena_recent)


def drop_order_allows(tree, node) -> bool:
    """The tree-side gate: the order is set, the node is a childless, unlocked leaf without a write-through
    in flight (the #841 precondition of ``_ud_drop_unbacked_leaf`` and a holder's lock both exclude it)."""
    if not getattr(tree, ORDER_ATTR, False):
        return False
    if getattr(node, "children", None):
        return False
    ongoing = getattr(tree, "ongoing_write_through", None) or {}
    if getattr(node, "id", None) in ongoing:
        return False
    for attr in ("lock_ref", "host_ref_counter"):
        try:
            if int(getattr(node, attr, 0) or 0) > 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def note_drop(tree, tokens: int) -> None:
    st = getattr(tree, STATS_ATTR, None)
    if st is None:
        st = [0, 0]
        setattr(tree, STATS_ATTR, st)
    st[0] += 1
    st[1] += max(0, int(tokens))


def evict_with_order(tree, params) -> tuple:
    """``tree.evict(params)`` with the order set; returns (dropped_leaves, dropped_tokens). The flag is
    cleared in ``finally``."""
    setattr(tree, STATS_ATTR, [0, 0])
    setattr(tree, ORDER_ATTR, True)
    try:
        tree.evict(params)
    finally:
        setattr(tree, ORDER_ATTR, False)
    st = getattr(tree, STATS_ATTR, None) or [0, 0]
    return int(st[0]), int(st[1])


def log_drop(actor, leaves: int, tokens: int, p_wait_s: float, arena_fill: float, now: float) -> None:
    """``#1390f UNBACKED-DROP dropped_leaves=n dropped_tok=n p_wait_s=x arena_fill=f`` (rate-limited)."""
    if now < float(getattr(actor, "_ud1390f_log_next", 0.0) or 0.0):
        return
    actor._ud1390f_log_next = now + LOG_MIN_GAP_S
    logger.info("%s dropped_leaves=%d dropped_tok=%d p_wait_s=%.1f arena_fill=%.3f", MARK, int(leaves),
                int(tokens), float(p_wait_s), float(arena_fill))
