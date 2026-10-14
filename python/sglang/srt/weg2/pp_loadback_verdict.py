# SPDX-License-Identifier: Apache-2.0
"""LB-VERDICT-1014: on the carrierless PP form PP0 alone decides WHEN a host
load-back is admitted; the verdict rides the request chain and every stage
admits the load-back in the SAME pass (the pass that absorbs the verdict).

THE DEATH. NF h16 a2cfc77f76, P, 14.10. 17:30:29Z (P.log ``..._1014_164206``
158839-158959): ``#1004 SLOT DISAGREEMENT: PP2 is launching slot 1 (fwd_ct=87,
rids=[weg2-40-310], extend=275) but the upstream's proxy names slot 2``.
weg2-40-310 resumed with a 150720-token host hit (``kv_tokens=150720``). Every
stage decided the load-back's room on ITS OWN pool (the local-PP floor: TP=1,
the floor is this rank's ``available_size()``):

  PP2  #1421 BACKUP-REFUSED arena_claim -> UD-H HOST-CHILDREN-CLEARED ->
       EVICT-UNBACKED-DROP node=487 freed=118720 -> #988 LOADBACK -> admitted (pass N)
  PP0  UD-REFUSED why=host_children:child_host_locked node=479 -> SF LOADBACK-ROOM
       PP-RESIDUAL avail_after=140416 < 150720 -> WEG2-LOADBACK-WAIT -> admitted pass N+1
  PP1  SF LOADBACK-ROOM PP-RESIDUAL (same numbers) -> pass N+1

``pp_slot_fidelity.local_pp_room`` names it in its own docstring: "The verdict is
RANK-LOCAL on a carrierless PP form: a peer with room admits in this pass, and
if one does the #1004 SLOT DISAGREEMENT that follows has THIS line as its
cause". The PR room vote (``weg2_pp_room_cap``) is published on PP0's tree only,
so a follower's ``agreed_room`` is None and its verdict stays its own; the row
authority that would carry PP0's admission is DISABLED on this form ('#631 ROW
AUTHORITY DISABLED: no pp_flip_counters side channel').

THE RULE (the H125f vision verdict's law, applied to the load-back): no stage
loads a host hit back before PP0's verdict about that rid has reached it over
the chain. PP0 decides from its own room (and the agreed R_m when it has one):
when the load-back fits it puts ``Weg2PpLoadbackVerdict(rid)`` on the wire at
its next intake and holds the rid itself; every stage dispatches the identical
intake before it plans, so all of them release the rid in the same pass and
load it there. A stage that cannot pay a RELEASED load-back stops by name
(``AgreedRoomShort``, raenge-nie-uneins-crash-stop) -- it never refuses alone.
The price is one pass of latency per host load-back on this form.

Switch ``SGLANG_WEG2_PP_LOADBACK_VERDICT`` (default on); ``0`` = the rank-local
room verdict as before. Armed only where the floor is the local-PP floor
(``pp_slot_fidelity.FLOOR_LOCAL_PP_ATTR``) and the stage has a pp rank > 1 group.

THE ROW IS PP0'S VERDICT (NF-RANG-AUDIT 1014, S5 x Fix B). With the #631 row
authority on (``weg2/p_row_authority.py``, nextflash since 14.10.) a follower
executes PP0's admission row: ``execute_scheduled_prefix`` materialises exactly
the scheduled prefix from this stage's host tier (``init_load_back`` ->
``load_back``). On the layer-split form the host hit is NOT rank-uniform
(schedule_policy: PP0 may hold the prefix on the device while a follower holds
it only on the host), so PP0 never raised ``hold`` for that rid and never sent
a verdict -- the gate below would hold the follower's materialisation until
its bound and stop it (``#968 PREFIX MATERIALISATION SHORTFALL``) while PP0
proceeds: a split made by the gate itself. A load-back that EXECUTES a
forwarded row therefore passes the gate (``gated`` is False while
``ROW_ATTR`` is set on the request); the row already is the one decision, and
a stage that cannot execute it stops by name in the materialisation station.
Switch ``SGLANG_WEG2_LBV_ROW_YIELD`` (default on); ``0`` = the gate holds a
row-executed load-back as before (the split).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_PP_LOADBACK_VERDICT"
MARK = "LB-VERDICT"
_STATE_ATTR = "_weg2_lbv_state"
_RELEASED_CAP = 4096
_LOG_FIRST = 16
#: NF-RANG-AUDIT 1014: the gate yields to a load-back that executes PP0's row.
ROW_YIELD_ENV = "SGLANG_WEG2_LBV_ROW_YIELD"
#: request attribute, True only while ``execute_scheduled_prefix`` materialises
#: a forwarded row (set and cleared by that station, one straight-line stretch).
ROW_ATTR = "_weg2_lbv_row_executed"


@dataclass
class Weg2PpLoadbackVerdict:
    """PP0's verdict for one host load-back: every stage loads it in the pass
    that absorbs this object."""

    rid: str


class _State:
    __slots__ = ("pp0", "released", "sent", "out", "said")

    def __init__(self, pp0: bool) -> None:
        self.pp0 = bool(pp0)
        self.released: set = set()
        self.sent: set = set()
        self.out: List[str] = []
        self.said = 0


def enabled() -> bool:
    return (os.environ.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def arm(scheduler) -> bool:
    """Arm the gate on this stage (PP0 and followers alike): pp_size > 1 and the
    switch on. The tree carries the state (the load-back runs inside it)."""
    ps = getattr(scheduler, "ps", None)
    pp_size = int(getattr(ps, "pp_size", 1) or 1)
    tree = getattr(scheduler, "tree_cache", None)
    if tree is None or pp_size <= 1 or not enabled():
        return False
    try:
        setattr(tree, _STATE_ATTR, _State(int(getattr(ps, "pp_rank", 0) or 0) == 0))
    except Exception:  # noqa: BLE001 -- a slotted double
        return False
    return True


def state(tree) -> Optional[_State]:
    return getattr(tree, _STATE_ATTR, None)


def armed(tree) -> bool:
    """The gate applies to this load-back: armed, switch on, and the floor is this
    rank's own value on a multi-stage PP form (the carrierless P)."""
    from sglang.srt.weg2.pp_slot_fidelity import FLOOR_LOCAL_PP_ATTR

    return (enabled() and state(tree) is not None
            and bool(getattr(tree, FLOOR_LOCAL_PP_ATTR, False)))


def row_yield_enabled() -> bool:
    return (os.environ.get(ROW_YIELD_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def row_executed(req) -> bool:
    """True while ``req``'s load-back executes a forwarded row (PP0's decision)."""
    return bool(getattr(req, ROW_ATTR, False))


def gated(tree, req) -> bool:
    """Does the LB-VERDICT gate apply to THIS load-back? Armed on this tree, the
    request named, and -- NF-RANG-AUDIT 1014 -- not a load-back that executes
    PP0's row: the row is the verdict, the station stops by name on a shortfall."""
    if req is None or getattr(req, "rid", None) is None or not armed(tree):
        return False
    if row_yield_enabled() and row_executed(req):
        return False
    return True


def absorb(tree, verdict: Weg2PpLoadbackVerdict) -> None:
    """Dispatcher handler on every stage: the rid is released for this pass."""
    st = state(tree)
    if st is None:
        return
    st.released.add(str(verdict.rid))
    st.sent.discard(str(verdict.rid))
    if len(st.released) > _RELEASED_CAP:
        st.released.pop()
    if st.said < _LOG_FIRST:
        st.said += 1
        logger.info("%s RELEASE rid=%s pp0=%s (PP0's load-back verdict absorbed: every stage loads it in "
                    "this pass)", MARK, verdict.rid, st.pp0)


def take(tree) -> List[Weg2PpLoadbackVerdict]:
    """PP0's origin hook: the verdicts decided since the last intake."""
    st = state(tree)
    if st is None or not st.out:
        return []
    items = [Weg2PpLoadbackVerdict(rid=r) for r in st.out]
    st.out.clear()
    return items


def released(tree, rid) -> bool:
    st = state(tree)
    return st is not None and str(rid) in st.released


def consume(tree, rid) -> None:
    """The released load-back ran: the verdict is spent."""
    st = state(tree)
    if st is not None:
        st.released.discard(str(rid))


def hold(tree, rid, kv_tokens: int, fits_on_pp0) -> None:
    """A load-back without a released verdict waits this pass on every stage.
    PP0 alone decides: ``fits_on_pp0()`` (called on PP0 only) True -> the verdict
    goes on the wire at the next intake."""
    st = state(tree)
    if st is None or rid is None:
        return
    rid = str(rid)
    if st.pp0:
        if rid in st.sent:
            return
        if fits_on_pp0():
            st.sent.add(rid)
            st.out.append(rid)
            logger.info("%s VERDICT rid=%s kv_tokens=%d ok (on the wire at the next intake; PP0 holds it until "
                        "then like every stage -- one decision, one pass, every stage)", MARK, rid, int(kv_tokens))
        return
    if st.said < _LOG_FIRST:
        st.said += 1
        logger.info("%s HOLD rid=%s kv_tokens=%d (a follower loads a host hit only in the pass that absorbs "
                    "PP0's verdict; its own room decides nothing)", MARK, rid, int(kv_tokens))
