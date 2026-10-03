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
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

MARK = "Q-693 WAITING-ABORT AT-RECEIPT"


def _fresh(req: Any) -> bool:
    """Never admitted on this rank: not retracted, not parked, no pool row."""
    if getattr(req, "is_retracted", False):
        return False
    if int(getattr(req, "weg2_parked_span", 0) or 0) > 0:
        return False
    return getattr(req, "req_pool_idx", None) is None


def applies_at_receipt(scheduler: Any, held: Sequence[Any], env=None) -> bool:
    """True = the follower applies this waiting-queue abort NOW (PP0 never
    admitted any of ``held``). False = the #1180-W hold, unchanged."""
    from sglang.srt.weg2 import dual_p_kv_stage as _dpk

    if not _dpk.armed(env):
        return False
    if not held:
        return False
    from sglang.srt.weg2 import p_intake as _p_intake

    for req in held:
        if not _fresh(req):
            return False
        if not _p_intake.told_pending(scheduler, req):
            return False  # a told reached this rank: PP0 may have admitted it
    n = int(getattr(scheduler, "_q693_at_receipt_n", 0) or 0) + 1
    scheduler._q693_at_receipt_n = n
    logger.warning(
        "%s rid=%s pp_rank=%s n=%d: no Weg2StoreTold of this request ever reached this rank, so "
        "PP0 had not admitted it when it read the abort (PP0 pops it at receipt) -- applied here "
        "at receipt too; no #1180-W hold, no zombie for a later instance of this rid (y8y)",
        MARK, ",".join(str(getattr(r, "rid", "?")) for r in held),
        getattr(getattr(scheduler, "ps", None), "pp_rank", "?"), n)
    return True
