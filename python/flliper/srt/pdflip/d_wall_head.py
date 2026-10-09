"""PW-R: a D queue head the backup wall makes unservable goes back to the front.

NF int18, 08.10. (D log boot_weg2_dkrnfint4h6ablxcbar1dauer10081149_e69f28a7f6_
1008_115011.D.log): pdflip-28-97 (182428 tokens, 924 uncached, its prefix host-
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
flip (12:12:54-12:17:30) and its follow-up pdflip-38-135 with pdflip-38-153 stood
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
    from flliper.srt.mem_cache.common import payable_evictable_or

    return int(payable_evictable_or(tree, tree.evictable_size))


def wall_unservable(*, tree: Any, available: int, need: int) -> bool:
    """This rank's half of the verdict: the backup wall stands and ``need``
    exceeds the free rows plus every evictable token the peel can pay."""
    from flliper.srt.pdflip.d_park_runtime import backup_wall_up

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
        "PDFLIP D-WALL-HEAD rid=%s need_rows=%d available=%d reported_evictable=%d "
        "payable_evictable=%d: the backup wall stands (the host arena takes no "
        "backup, nothing D holds can leave the device) and the head cannot be held "
        "even with every payable row -- answered W50, the front re-routes it through "
        "P and the flip clears D's tree",
        head.rid, need, int(available), int(tree.evictable_size()), _payable(tree),
    )
    return head
