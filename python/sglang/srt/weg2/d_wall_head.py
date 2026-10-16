"""PW-R: a D queue head the backup wall makes unservable goes back to the front.

NF int18, 08.10. (D log boot_weg2_dkrnfint4h6ablxcbar1dauer10081149_e69f28a7f6_
1008_115011.D.log): weg2-28-97 (182428 tokens, 924 uncached, its prefix host-
backed: ``HEAD-VOTE ANCHOR matched=178112 (device 0 + host 178112)``) was routed
SHORT to D -- X-conform, D prefills 924 (front ``ROUTE-VERDICT verdict=short
uncached=1052 presence_src=d_leg2_cached``). Admitting it needs 180928 device rows
(load-back + first chunk) against ``available=144832..150080``; the rest of the
room was "evictable" only on paper: write_back leaves whose backup the full host
arena refused (``EVICT-FRONTIER-CENSUS request=31552 delivered=0
reported_evictable=182528``, ``ARENA-DROP freed=0``). Under that wall nothing on
D frees a device row -- a displaced seat's retained span, a finished request's
cache go back to the tree as unpayable as the rest (D drops no un-backed leaf on a
TP group: ``pp_slot_fidelity.unbacked_drop_allowed``). The head waited to the
flip (12:12:54-12:17:30) and its follow-up weg2-38-135 with weg2-38-153 stood
D still at running=0 (12:27:36-12:29:52, ``available=1408
reported_evictable=507136``).

THE RULE: when the pass's first NO_TOKEN refusal on group D (the HOL head, the
same rid on every rank: the Form A gather) cannot be held by this rank's device
even with every evictable token the peel can PAY while the wall stands, and every
rank says so (group MIN), the head is answered with the named W50 refusal, its
extent being the device rows it needs: the front re-routes it through P and the
flip clears D's tree (12:19:11Z ``W50-REROUTE`` -> 12:19:26Z ``ARRIVAL-SEAT
verdict=flip_now`` -> 12:19:54Z ``D-ADMIT``, served). Without a measured wall
nothing changes: the head waits for the room the running seats give back
(SEAT-AGE decides displacement as before)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)


def _n(x) -> int:
    """len() of a list OR a tensor; never ``x or ()`` (a tensor has no truth value)."""
    return 0 if x is None else len(x)


def device_need_rows(req: Any) -> int:
    """The device rows admitting ``req`` allocates: everything not already
    resident on the device -- the host load-back plus the uncached extend."""
    return max(0, _n(req.full_untruncated_fill_ids) - _n(req.prefix_indices))


def _payable(tree: Any) -> int:
    from sglang.srt.mem_cache.common import payable_evictable_or

    return int(payable_evictable_or(tree, tree.evictable_size))


def wall_unservable(*, tree: Any, available: int, need: int) -> bool:
    """This rank's half of the verdict: the backup wall stands and ``need``
    exceeds the free rows plus every evictable token the peel can pay."""
    from sglang.srt.weg2.d_park_runtime import backup_wall_up

    if tree is None or not backup_wall_up(tree):
        return False
    return int(need) > int(available) + _payable(tree)


def pick_unservable_head(
    *,
    head_rid: Optional[str],
    waiting: Sequence[Any],
    tree: Any,
    available: int,
    group_min: Callable[[list], list],
) -> Optional[Any]:
    """The HOL head to hand back, or None. ``group_min`` is entered exactly when
    the head is in the queue -- replicated inputs, so on every rank alike."""
    if head_rid is None:
        return None
    head = next((q for q in waiting if str(q.rid) == str(head_rid)), None)
    if head is None:
        return None
    need = device_need_rows(head)
    local = wall_unservable(tree=tree, available=available, need=need)
    if not bool(group_min([local])[0]):
        return None
    logger.warning(
        "WEG2 D-WALL-HEAD rid=%s need_rows=%d available=%d reported_evictable=%d "
        "payable_evictable=%d: the backup wall stands (the host arena takes no "
        "backup, nothing D holds can leave the device) and the head cannot be held "
        "even with every payable row -- answered W50, the front re-routes it through "
        "P and the flip clears D's tree",
        head.rid, need, int(available), int(tree.evictable_size()), _payable(tree),
    )
    return head


# ---------------------------------------------------------------------------
# PW-W (NF h14 ba1d5becd7, D 14:50:53-14:56:52Z, weg2-2-21): the STRUCTURAL
# wedge -- a second way into the same W50 exit.
#
# weg2-2-21 (lane 1, 163616 tokens, prefix host-backed after D evicted the
# device copy one pass earlier) was refused by the attention host's H105 gate
# 1151 times in 317 s: ``host=NO_TOKEN host_price=167776 host_budget=33536``,
# nothing running, the same two numbers on every refusal. PW-R above never
# fired: it needs the BACKUP WALL (the host arena refusing backups) and asks
# each rank's own device view, while the Form A workers price the same head
# at 4704 (their bookkeeping tree holds the prefix as device rows) -- so the
# group MIN of the local verdict was False every time. H106's watch never
# fired either: it is built only for the host-broadcast form, and this group
# runs the token cut (gather form). The lane floor 1 held every lane-0
# request behind the head: the whole server stood.
#
# THE RULE: the group-DECIDED gate tuple ``(rid, code, price, budget)`` of the
# pass (``Scheduler._form_a_admission_follow_fn`` ``last_verdict``: the
# broadcast / gather result, identical on every rank) refuses the pass's HOL
# head with ``price > budget`` while nothing runs, and the SAME
# (rid, price, budget) has done so for ``SGLANG_WEG2_D_WEDGE_REROUTE_N``
# consecutive evaluations: the head is answered with the named W50 refusal
# (extent = the decided price), the front re-routes it through P and the
# floor's lane runs on. Every input is replicated (the decided tuple, the
# queue, the running set), the counter is advanced on every rank in the same
# evaluation, so the verdict needs no collective. An admit, another head, a
# running request or a moved number resets it. 0 = off.
# ---------------------------------------------------------------------------

STRUCT_MARK = "WEG2 D-WEDGE-HEAD"
_STATE_ATTR = "_weg2_wedge_head_state"


def wedge_reroute_n() -> int:
    import os

    raw = os.environ.get("SGLANG_WEG2_D_WEDGE_REROUTE_N", "64") or "64"
    try:
        return max(0, int(raw))
    except ValueError:
        return 64


def pick_structural_wedge_head(
    *,
    sched: Any,
    head_rid: Optional[str],
    waiting: Sequence[Any],
    verdict: Optional[tuple],
    running_empty: bool,
    limit: Optional[int] = None,
) -> Optional[Any]:
    """The HOL head a structurally refused gate has held for ``limit``
    consecutive evaluations, or None. ``verdict``: the group-decided
    ``(rid, code, price, budget)`` of this pass's last gate call."""
    n_lim = wedge_reroute_n() if limit is None else int(limit)
    if n_lim <= 0 or head_rid is None:
        return None
    st = getattr(sched, _STATE_ATTR, None)
    ok = (
        verdict is not None
        and len(verdict) >= 4
        and str(verdict[0]) == str(head_rid)
        and str(verdict[1]) != "ADMIT"
        and running_empty
    )
    try:
        price, budget = int(verdict[2]), int(verdict[3])
    except (TypeError, ValueError, IndexError):
        ok = False
        price = budget = 0
    if not ok or price <= budget:
        setattr(sched, _STATE_ATTR, None)
        return None
    key = (str(head_rid), price, budget)
    if st is None or st[0] != key:
        st = [key, 0]
    st[1] += 1
    setattr(sched, _STATE_ATTR, st)
    if st[1] < n_lim:
        return None
    head = next((q for q in waiting if str(q.rid) == str(head_rid)), None)
    setattr(sched, _STATE_ATTR, None)
    if head is None:
        return None
    logger.warning(
        "%s rid=%s code=%s price=%d budget=%d refusals=%d running=0: the group's "
        "admission gate refused the queue head with a price above its budget and "
        "nothing running for %d consecutive evaluations -- answered W50, the front "
        "re-routes it through P so the lanes behind it run on",
        STRUCT_MARK, head.rid, verdict[1], price, budget, n_lim, n_lim,
    )
    return head


# ---------------------------------------------------------------------------
# PW-F (NF h16p K10 15.10. 12:21:52-12:27:33Z, D TP0-2): the NO-FUND wedge --
# a third way into the same W50 exit.
#
# 7 queued, 0 running, every pass the HOL head refused NO_TOKEN by the H105d
# cut (``FORM-A-CUT LOAD-BACK ROOM ... evicted=0``), the peel delivering 0
# (``EVICT-FRONTIER-CENSUS delivered=0 reported_evictable=516288``,
# ``ARENA-DROP freed=0`` x3366) -- for six minutes. PW-R did not fire after
# 12:12:57: its local verdict reads ``payable_evictable``, a rank-local memo
# with a 2 s TTL and a rank-local key (evict_frontier_census.UNPAYABLE_TTL_S),
# so the group MIN was never True on all ranks at once (H105d 12:24:02 TP0
# payable=516288, 12:23:44 TP1 payable=0). PW-W did not fire: the H105 gate's
# budget (rem_total_tokens 521280) holds the unpayable evictable, price <=
# budget. H106 did not fire: the host said ADMIT, the NO_TOKEN came from the cut.
#
# THE RULE: a pass whose group-D HOL head was refused (``_hol.head``, the same
# rid on every rank), with nothing running and nothing admitted (the built set
# is the host's after the H105 riegel), advances one counter; any other pass
# resets it. After ``SGLANG_WEG2_D_NOFUND_REROUTE_N`` consecutive such passes
# the CURRENT head is answered with the named W50 (extent = its device rows)
# and the count starts again. Every input is replicated -- no tree reading, no
# clock -- so every rank answers the same head in the same pass without a
# collective. 0 = off.
# ---------------------------------------------------------------------------

NOFUND_MARK = "WEG2 D-NOFUND-HEAD"
_NOFUND_ATTR = "_weg2_nofund_head_state"


def nofund_reroute_n() -> int:
    import os

    raw = os.environ.get("SGLANG_WEG2_D_NOFUND_REROUTE_N", "64") or "64"
    try:
        return max(0, int(raw))
    except ValueError:
        return 64


def reset_nofund(sched: Any) -> None:
    """A pass that was not a no-fund pass (an admit, no refused head, a run)."""
    if getattr(sched, _NOFUND_ATTR, None) is not None:
        setattr(sched, _NOFUND_ATTR, None)


def pick_nofund_head(
    *,
    sched: Any,
    head_rid: Optional[str],
    waiting: Sequence[Any],
    running_empty: bool,
    admitted_n: int,
    limit: Optional[int] = None,
) -> Optional[Any]:
    """The HOL head to answer W50 after ``limit`` consecutive no-fund passes, or
    None. Inputs replicated only (see the rule above)."""
    n_lim = nofund_reroute_n() if limit is None else int(limit)
    if n_lim <= 0 or head_rid is None or not running_empty or int(admitted_n or 0) > 0:
        reset_nofund(sched)
        return None
    n = int(getattr(sched, _NOFUND_ATTR, None) or 0) + 1
    setattr(sched, _NOFUND_ATTR, n)
    if n < n_lim:
        return None
    setattr(sched, _NOFUND_ATTR, None)
    head = next((q for q in waiting if str(q.rid) == str(head_rid)), None)
    if head is None:
        return None
    logger.warning(
        "%s rid=%s need_rows=%d passes=%d running=0 admitted=0 queued=%d: the group-D head was refused "
        "with nothing running and nothing admitted for %d consecutive passes (the peel pays nothing: "
        "backup wall) -- answered W50, the front re-routes it through P and the flip clears D's tree",
        NOFUND_MARK, head.rid, device_need_rows(head), n, len(waiting), n_lim,
    )
    return head
