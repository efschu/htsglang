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

THE RULE: when the pass's first NO_TOKEN refusal on group D (the HOL head)
cannot be held by this rank's device
even with every evictable token the peel can PAY while the wall stands, and every
rank says so (group MIN), the head is answered with the named W50 refusal, its
extent being the device rows it needs: the front re-routes it through P and the
flip clears D's tree (12:19:11Z ``W50-REROUTE`` -> 12:19:26Z ``ARRIVAL-SEAT
verdict=flip_now`` -> 12:19:54Z ``D-ADMIT``, served). Without a measured wall
nothing changes: the head waits for the room the running seats give back
(SEAT-AGE decides displacement as before).

NF D-HANG 1011 (RAENGE-NIE-UNEINS, "the group collective is entered by every rank,
the rank-local predicate travels as payload"): the head is NOT the same rid on every
rank. The Form A gather makes a NO_TOKEN verdict the host's on every rank, but the
load-back of ``schedule_policy.add_one_req`` has an exit AFTER the group ADMIT that is
rank-local -- a worker whose load-back finds no device room returns NO_TOKEN alone
(WEG2-LOADBACK-WAIT) while the host admits. ``HolPass.head`` is then set on that worker
only; with the group MIN entered "iff the head is queued" only the workers entered it and
the host went on into the forward -- the 3-rank picture of the 11.10. 10:25:45Z D hang
(TP0 in the extend forward, TP1/TP2 inside the admission pass). So now EVERY rank enters
the MIN once per pass (``pick_unservable_head`` is called whenever D-HOL is on, a
replicated condition) and the local verdict -- "I have a head, it is queued, my wall says
unservable" plus a 24-bit tag of the head's rid -- is the payload. The answer is taken
after the collective and is the same on every rank: a head is handed back only when every
rank has the same head and every rank finds it unservable."""

from __future__ import annotations

import logging
import zlib
from typing import Any, Callable, List, Optional, Sequence

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


#: bits of the head-rid tag that travel in the MIN (two flags per bit: the bit and its
#: complement, so the MIN of both tells whether every rank holds the same value)
_TAG_BITS = 24


def head_tag_bits(head_rid: Optional[str]) -> List[bool]:
    """A fixed-width tag of the head rid; all zero for no head. Deterministic across
    processes (crc32, not ``hash``)."""
    if head_rid is None:
        return [False] * _TAG_BITS
    h = zlib.crc32(str(head_rid).encode("utf-8")) & ((1 << _TAG_BITS) - 1)
    return [bool((h >> i) & 1) for i in range(_TAG_BITS)]


def pick_unservable_head(
    *,
    head_rid: Optional[str],
    waiting: Sequence[Any],
    tree: Any,
    available: int,
    group_min: Callable[[list], list],
) -> Optional[Any]:
    """The HOL head to hand back, or None -- the same answer on every rank.

    ``group_min`` is entered UNCONDITIONALLY, exactly once, by every rank that calls this
    (the caller's condition is replicated: D-HOL on). ``head_rid`` is a rank-local fact and
    may be None here while a peer has a head; it travels as payload, never as the entry
    condition. Payload: [local verdict] + [tag bits] + [complement bits]; the head is
    answered only when the MIN says every rank has a queued head, every rank finds it
    unservable under its wall, and every rank holds the SAME head."""
    head = None
    local = False
    if head_rid is not None:
        head = next((q for q in waiting if str(q.rid) == str(head_rid)), None)
    need = 0
    if head is not None:
        need = device_need_rows(head)
        local = wall_unservable(tree=tree, available=available, need=need)
    bits = head_tag_bits(None if head is None else str(head.rid))
    got = group_min([local] + bits + [not b for b in bits])
    if not bool(got[0]):
        return None
    same = all(bool(got[1 + i]) or bool(got[1 + _TAG_BITS + i]) for i in range(_TAG_BITS))
    if not same or head is None:
        logger.warning(
            "WEG2 D-WALL-HEAD-SPLIT head=%s: every rank finds a wall-unservable head but "
            "not the same one (this rank's tag bits differ from a peer's) -- nobody is "
            "answered W50 this pass; the heads wait as before (RAENGE-NIE-UNEINS)",
            head_rid,
        )
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
