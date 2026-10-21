"""UD-LAST (NF rc27 reboot 1021_023225, P PP0 02:37:15): an un-backed write_back leaf is dropped only when the peel
cannot be paid otherwise -- after every backed leaf (L1 -> L2 demotion, nothing lost) had its turn.

THE LOSS (P.log of boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwflbar1dauer10210232_893d09fa3f_1021_023225, front log same
prefix). pdflip-0-22 (184544 tokens, cached 170240) computed [170240, 184512) on P; PP0 wrote its END anchor
(17742 ``PDFLIP END-ANCHOR ... anchor=184512 ok=True``) but the KV arena (6485 slots, ``ARENA-REF-CENSUS ...
complete=6485 pinned=6485 refs=18172``: EVERY slot referenced by a tree node, a prefetch span or a park) refused the
backup: ``RETAIN-PUBLISH rid=pdflip-0-22 {'unbacked': 34, 'issued': 0, 'refused': 34 ...}``, ``#1427 ARENA-DROP n=1
need=1 freed=0 stages=i:0,ii:0,iii:0`` (the claim's room maker takes UNREFERENCED slots only -- there were none),
``#1421 BACKUP-REFUSED why=arena_claim node=15 tokens=64 depth=170304``. Three seconds later PP0 made room for the
load-back of pdflip-0-19 (``SF LOADBACK-ROOM SAME-PASS rid=pdflip-0-19 kv_tokens=113408 floor=85760 ... evictable=
243648 evicted=165632``, need 27648) and the LRU peel hit 0-22's fresh tail FIRST: ``EVICT-UNBACKED-DROP node=16
tokens=1920``, ``node=19 tokens=4096``, ``node=18 tokens=4096``, ``node=17 tokens=4096``, ``node=15 tokens=64`` --
14272 tokens, exactly the ``PREFIX-DIVERGE verdict=store_short best_prev=pdflip-0-22 lost=14272`` of the follow-up
pdflip-1-23, which recomputed [170240, 184701) (``#969 EXTENT ... 170240, 184701, 170240, 14461``). PP1 (02:37:21)
and PP2 (02:37:24) dropped the same five nodes in the same order on their own row-room peels; PP0 02:37:28 dropped
node 21 (44288 tokens, UD-H) the same way while 337408 tokens stayed evictable. The peel left 78016 evictable
tokens behind (``evictable=78016`` on the next line): the drop was not needed to pay the request -- the LRU order
simply reached the un-backed leaf before backed ones.

THE RULE. The full component's peel (``FullComponent._peel``) runs in two passes. Pass 1: a write_back leaf whose
backup was refused and that UD would drop (local-PP floor) is HELD -- it stays on the device, in the evictable set,
and the peel goes on with the next leaf; backed leaves are demoted to the host as before. Pass 2, only if pass 1
ended short of the request: the held leaves are peeled in the old order (backup tried once more, then the UD /
UD-H drop, their parents next) -- exactly the old behaviour, so the rc12q liveness (``249856 tokens behind one
un-backable leaf``) holds: a peel that can only be paid by dropping still drops, never more than it needs. A held
leaf keeps its fresh KV/anchor on the device for the next publish sweep, which backs it up as soon as the arena has
room (the W3 room set / write-behind make that room in the background).

RANK AGREEMENT. No new verdict: the eviction on the local-PP floor already was each stage's own (SF: the floor is
this rank's ``available_size()``), every stage runs the same deterministic order over its own tree (same tree shape,
same backup states -> same held set), PP0's rows and told are untouched. No timer, no wait, no I/O: pass 1 only
skips; pass 2 is the old code path.

Switch ``FLLIPER_PDFLIP_UD_LAST`` (default on; 0 = the one-pass peel byte for byte).
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_UD_LAST"
#: tree attribute: the list of leaves held by the running pass 1 (None = no pass 1 running: drop as before)
HOLD_ATTR = "_ud_last_hold"
#: tree attribute: ids of the leaves pass 1 held, while pass 2 runs (None otherwise)
REFUSED_ATTR = "_ud_last_refused"
_LOG_FIRST = 16
_N = {"held": 0, "dropped_after": 0}


def on(env=None) -> bool:
    e = os.environ if env is None else env
    if str(e.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "") or "").strip() == "1":
        return False  # 27B NACHT 21.10.: the dual layout stays as it was (user 20.10.) -- off under FLLIPER_PDFLIP_DUAL_LAYOUT=1 (set on every rank of both groups: rank-uniform)
    raw = (e.get(ENV, "1") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def begin(tree: Any):
    """Open pass 1 on ``tree``; returns the outer peel's state (re-entrant peels nest) or ``False`` when the switch
    is off (nothing written)."""
    if not on():
        return False
    d = getattr(tree, "__dict__", None)
    if d is None:
        return False                   # a slotted double: no pass 1
    prev = (d.get(HOLD_ATTR), d.get(REFUSED_ATTR))
    setattr(tree, HOLD_ATTR, [])
    setattr(tree, REFUSED_ATTR, None)
    return prev


def take(tree: Any) -> List[Any]:
    """Close pass 1: returns the leaves it held (in hold order) and leaves NO hold list open, so pass 2 (and any
    peel nested in it) drops as before. The held leaves are remembered as 'backup refused in this peel': pass 2
    goes straight to the UD branch for them (demotions free no arena slot -- a second claim is futile work on the
    scheduler thread, and doubled #1421/#1427 lines)."""
    held = list(getattr(tree, HOLD_ATTR, None) or [])
    setattr(tree, HOLD_ATTR, None)
    setattr(tree, REFUSED_ATTR, {id(n) for n in held} if held else None)
    return held


def end(tree: Any, prev) -> None:
    """Restore the outer peel's state."""
    if prev is False:
        return
    setattr(tree, HOLD_ATTR, prev[0])
    setattr(tree, REFUSED_ATTR, prev[1])


def refused_in_peel(tree: Any, node: Any) -> bool:
    """True for a leaf pass 1 of the running peel held (its backup was refused moments ago in this peel)."""
    s = getattr(tree, REFUSED_ATTR, None)
    return bool(s) and id(node) in s


def hold(tree: Any, node: Any) -> bool:
    """Called by ``_evict_device_leaf`` where UD would drop ``node`` (backup refused, local-PP floor): True = pass 1
    holds it (the caller returns without freeing anything), False = no pass 1 running (drop as before)."""
    lst = getattr(tree, HOLD_ATTR, None)
    if lst is None or lst is False:
        return False
    lst.append(node)
    return True


def note(tree: Any, held: List[Any], got_pass1: int, request: int, got_after: int, dropped_after: int) -> None:
    """One line per peel that held a leaf: how many, how many tokens, whether pass 2 had to drop any."""
    if not held:
        return
    tokens = sum(len(getattr(n, "key", None) or []) for n in held)
    if dropped_after:
        _N["dropped_after"] += 1
        k = _N["dropped_after"]
        if k <= _LOG_FIRST or (k & (k - 1)) == 0:
            logger.warning(
                "UD-LAST DROP-AFTER n=%d held=%d held_tokens=%d request=%d pass1=%d after=%d dropped_nodes=%d: the "
                "backed leaves could not pay the peel -- the held un-backed leaves were dropped (old order, only as "
                "many as needed)", k, len(held), tokens, int(request), int(got_pass1), int(got_after),
                int(dropped_after))
        return
    _N["held"] += 1
    k = _N["held"]
    if k <= _LOG_FIRST or (k & (k - 1)) == 0:
        logger.info(
            "UD-LAST HELD n=%d nodes=%s held_tokens=%d request=%d paid=%d: un-backed write_back leaves (backup "
            "refused, arena full) stay on the device -- the backed leaves paid the peel, no fresh KV was dropped",
            k, [getattr(n, "id", "?") for n in held[:8]], tokens, int(request), int(got_pass1))
