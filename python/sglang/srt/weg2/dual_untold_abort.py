"""Q-693 DUAL UNTOLD WAITING-ABORT AT RECEIPT (27B NVFP4 dual y8y, boot
dkr27bnvfp4dual1mpsleepbar1fs10031814, image 36b5a4d3e9, P PP1 death 18:25:45Z).

METAL (P log, rid weg2-0-152; the log lines truncate rids to 8 characters, so
the #791T text reads "weg2-0-1" -- the debug-hold locals name weg2-0-152):

  18:25:43.27  front DUAL P-PAUSE (instance 1, three chunks in flight)
  18:25:43.89  front RESUME-UNSTARVE reason=short wait_s=0.0 -> instance 2 to P
  18:25:44     PP0 GRANT + DISPATCH instance 2, then its pause abort:
               'P-KV GRANT-RETURN why=abort: the told carrying PP0's group grant
               never left' + 'Q-580 TOLD-FORGET ... dropped=held' -- PP0 popped
               instance 2 at receipt, unadmitted, no told on the wire;
  18:25:44     PP1 'WEG2-PP-WAITING-ABORT held rid=weg2-0-152' (#1180-W): the
               follower keeps instance 2 in its waiting queue until PP0's
               forwarded schedule decides it;
  18:25:44.29  front RESUME-UNSTARVE again -> instance 3 under the SAME rid;
  18:25:45     PP0's frames name weg2-0-152 again (instance 3): the #1180-W
               verdict is 'keep' for ever (it reads the rid, not the object);
               PP1 admits instance 3 at slot 0, and at slot 1 the #791T probe
               maps the rid to the zombie instance 2 still queued, whose told
               never comes -> PpRowDeferCapExceeded, #1223 DEBUG-HOLD, W17.

THE RULE. On the dual P layout every admission of PP0 is preceded on the
request wire by its Weg2StoreTold / Weg2StoreAdmit (the told carries PP0's
group grant; a follower admits only on a told), and the follower absorbs the
told objects of a list BEFORE it dispatches that list's AbortReq. A follower
that reads an abort for a waiting-queue request whose told has NOT reached it
therefore knows PP0 had not admitted that request when PP0 read the abort --
PP0 pops it at receipt (Q-580 TOLD-FORGET dropped=held). The follower does the
same, at receipt, instead of holding it: no zombie stays behind for a later
instance of the rid to collide with. A request with a told here keeps the
#1180-W hold byte for byte (PP0 may have admitted it), and so does a retracted
or parked request (its admission is not a fresh told's).

Dual P only (``dual_p_kv_stage.armed``): the flip/INT8/NF forms never enter it.

Q-697 SAME-LIST TOLD (27B NVFP4 dual, boot dkr27bnvfp4dual1mpsleepsharebar1fs10032157,
image fda96a5338, P PP1 death 22:07:08Z; the log lines cut rids to 8 characters, the
'Q-695 #791T OVERDUE FULL' line names weg2-0-63):

  22:05:45     front leg 1 of weg2-0-63 (instance 1) to P; PP0 GRANT 22:05:53,
               paced read-ahead (PF TOLD-OPEN), PP1 starts its read 22:06:01
  22:06:03.59  front DUAL P-PAUSE weg2-0-63 -> /abort_request on P
  22:06:08     PP0 (its pass blocked 5 s by a load harvest) reads r101 = [Tok 64,
               Abort 63, Abort 54, Abort 64]: pp0_publish FIRST decides the PF
               Frist -- 'PF TOLD-FALLBACK rid=weg2-0-6 told=94720 -> 0' appends
               Weg2StoreAdmit(63, 0, fallback) to THIS list -- THEN dispatches the
               abort: instance 1 is popped at receipt, never admitted (the admission
               of this pass runs after the dispatch); the echo reaches the front
  22:06:08.41  front RESUME-UNSTARVE -> instance 2 of weg2-0-63 to P
  22:06:14     PP1 r152 = the same list + PP0's Admit (n=5): the Admit is absorbed
               first ('PF TOLD-FALLBACK ABSORBED (n=2)'), so the abort finds a told
               here -- Q-693 does not apply -- 'WEG2-PP-WAITING-ABORT held rid=weg2-0-63'
               (#1180-W). PP2 the same at 22:06:20.
  22:06:24..29 PP0 admits instance 2 at told=0 (fwd 820..822); PP1 absorbs its told,
               admits [0,1024) -- and the zombie instance 1 is still queued under the
               rid: at slot 1 the #791T probe maps weg2-0-63 to the queued object whose
               told was consumed ('queued(told=-)') -> 4 laps -> PpRowDeferCapExceeded,
               #1223 DEBUG-HOLD, W17 (the #1180-W verdict reads the rid, PP0's frames
               name it for instance 2: 'keep' for ever).

THE RULE. PP0 builds its forwarded list as ``recv_reqs + told objects of this pass``
(``pp0_publish``) and dispatches ``recv_reqs`` -- the aborts -- AFTER it, before its
admission step; a follower takes ONE PP0 list per pass (``PpChainReceiver.recv``).
A told verdict (``Weg2StoreAdmit`` or a single-phase ``Weg2StoreTold``) that rides
the SAME list as the abort of its rid was therefore issued in the pass in which PP0
read that abort, before PP0 could admit the request: PP0 popped it at receipt. The
follower applies such an abort at receipt too. A told from an EARLIER list keeps the
#1180-W hold byte for byte (PP0 may have admitted it in that earlier pass).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

MARK = "Q-693 WAITING-ABORT AT-RECEIPT"
#: Q-697: the abort's told rode the same PP0 list -- PP0 popped it before admitting
MARK_SAME_LIST = "Q-697 SAME-LIST TOLD ABORT AT-RECEIPT"
#: the follower's record of the list it is dispatching: (AbortReq objects, told rids)
LIST_ATTR = "_q697_same_list"


def _fresh(req: Any) -> bool:
    """Never admitted on this rank: not retracted, not parked, no pool row."""
    if getattr(req, "is_retracted", False):
        return False
    if int(getattr(req, "weg2_parked_span", 0) or 0) > 0:
        return False
    return getattr(req, "req_pool_idx", None) is None


def _told_verdict_rid(item: Any) -> Optional[str]:
    """The rid of a told VERDICT on the wire (what sets ``_weg2_store_told``): a
    ``Weg2StoreAdmit`` or a single-phase ``Weg2StoreTold``; a paced read-ahead is
    no verdict (nobody admits on it)."""
    from sglang.srt.managers import weg2_store_told as _st

    if isinstance(item, _st.Weg2StoreAdmit):
        return str(item.rid)
    if isinstance(item, _st.Weg2StoreTold) and not getattr(item, "paced", False):
        return str(item.rid)
    return None


def note_list(scheduler: Any, recv_reqs: Sequence[Any], env=None) -> None:
    """Q-697: a dual P follower notes, before it absorbs PP0's list, which told
    verdicts and which aborts ride it (the AbortReq objects themselves, so a later
    pass can never match them). Off the dual P layout / on PP0: nothing is written."""
    try:
        e = os.environ if env is None else env
        if str(e.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() != "1":
            return  # the flip / NF / 27B INT8 forms: nothing imported, nothing written
        from sglang.srt.weg2 import dual_p_kv_stage as _dpk

        if not _dpk.armed(env):
            return
        if int(getattr(getattr(scheduler, "ps", None), "pp_rank", 0) or 0) <= 0:
            return
        from sglang.srt.managers.io_struct import AbortReq

        aborts, rids = [], set()
        for item in recv_reqs or ():
            if isinstance(item, AbortReq):
                aborts.append(item)
                continue
            rid = _told_verdict_rid(item)
            if rid is not None:
                rids.add(rid)
        setattr(scheduler, LIST_ATTR, (tuple(aborts), frozenset(rids)) if (aborts and rids) else None)
    except Exception:  # noqa: BLE001 - advisory: without the record the #1180-W hold stands
        try:
            setattr(scheduler, LIST_ATTR, None)
        except Exception:  # noqa: BLE001
            pass


def told_in_same_list(scheduler: Any, recv_req: Any, rid: Any) -> bool:
    """Q-697: did a told verdict for ``rid`` ride the same PP0 list as ``recv_req``?"""
    rec = getattr(scheduler, LIST_ATTR, None)
    if not rec or recv_req is None:
        return False
    aborts, rids = rec
    return str(rid) in rids and any(a is recv_req for a in aborts)


def applies_at_receipt(scheduler: Any, held: Sequence[Any], env=None, recv_req: Any = None) -> bool:
    """True = the follower applies this waiting-queue abort NOW (PP0 never
    admitted any of ``held``): no told of it reached this rank (Q-693), or its told
    rode the SAME PP0 list as ``recv_req`` (Q-697). False = the #1180-W hold,
    unchanged."""
    from sglang.srt.weg2 import dual_p_kv_stage as _dpk

    if not _dpk.armed(env):
        return False
    if not held:
        return False
    from sglang.srt.weg2 import p_intake as _p_intake

    same_list = []
    for req in held:
        if not _fresh(req):
            return False
        if _p_intake.told_pending(scheduler, req):
            continue
        if told_in_same_list(scheduler, recv_req, getattr(req, "rid", None)):
            same_list.append(req)  # PP0 issued this told in the pass it read the abort
            continue
        return False  # a told from an earlier list: PP0 may have admitted it
    if same_list:
        n = int(getattr(scheduler, "_q697_same_list_n", 0) or 0) + 1
        scheduler._q697_same_list_n = n
        logger.warning(
            "%s rid=%s pp_rank=%s n=%d: the told verdict of this waiting request rode the SAME PP0 "
            "list as its abort -- PP0 issued it in the pass it read the abort, before its admission "
            "step, and popped the request at receipt; applied here at receipt too, no #1180-W zombie "
            "for the next instance of this rid (y9 22:06:14 weg2-0-63)",
            MARK_SAME_LIST, ",".join(str(getattr(r, "rid", "?")) for r in same_list),
            getattr(getattr(scheduler, "ps", None), "pp_rank", "?"), n)
        if len(same_list) == len(held):
            return True
    n = int(getattr(scheduler, "_q693_at_receipt_n", 0) or 0) + 1
    scheduler._q693_at_receipt_n = n
    logger.warning(
        "%s rid=%s pp_rank=%s n=%d: no Weg2StoreTold of this request ever reached this rank, so "
        "PP0 had not admitted it when it read the abort (PP0 pops it at receipt) -- applied here "
        "at receipt too; no #1180-W hold, no zombie for a later instance of this rid (y8y)",
        MARK, ",".join(str(getattr(r, "rid", "?")) for r in held),
        getattr(getattr(scheduler, "ps", None), "pp_rank", "?"), n)
    return True
