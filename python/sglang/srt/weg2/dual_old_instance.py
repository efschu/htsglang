"""Q-695 DUAL OLD-INSTANCE CHUNK (27B NVFP4 dual y8z, boot
dkr27bnvfp4dual1mpsleepbar1fs10031909, image ceff4aae7b, P PP1 death 19:26:06Z).

METAL (P log; the log lines cut rids to 8 characters, "weg2-0-2" -- the debug-hold
locals name weg2-0-235: ``_told_missing=['weg2-0-235']``, stamp
``(1, 1560, 1024, -1, 1560, ('weg2-0-235', 5120, 6144))``):

  19:26:04.905  front DUAL P-PAUSE weg2-0-235 (instance 2, admitted 19:26:04 at
                told=0, chunk 4096 of 13449 in flight) -> /abort_request on P
  19:26:05.24   PP0 '#801 ABORT RECEIVED' -> 'applied in 2 pass(es)' (xsn324): PP0
                still launches 1559 [4096,5120) and 1560 [5120,6144), then applies
                and echoes the abort;
  19:26:05.25   PP1/PP2 'WEG2-PP-CHUNKED-ABORT recorded ... applied when PP0's
                forwarded schedule stops naming it (#791C)': instance 2 stays the
                followers' chunked request until those two frames ran there;
  19:26:05.897  front WEG2-SERVED (the echo) + P-PAUSED, 19:26:05.898
                RESUME-UNSTARVE (weg2-0-236 was in flight, so Q-693's gate let it
                pass) -> instance 3 under the SAME rid to P;
  19:26:05.9    PP1 ran 1559, then '#1037 REQ RE-CONSTRUCTED ... instance=3': the
                request chain delivered instance 3 BEFORE frame 1560 (two wires, no
                cross-wire order);
  19:26:05.98   PP1 probes frame 1560 -- the row names weg2-0-235 at 5120, the
                continuation of ITS OWN chunked instance 2 (extend_range.end 5120)
                -- but the #791T told check maps the rid to the QUEUED instance 3,
                whose told cannot exist yet (PP0 had not even read it) -> '#791T
                ROW-PROBE DEFER' x4 -> PpRowDeferCapExceeded, #1223 DEBUG-HOLD, W17.

Q-693 covered a WAITING request aborted untold (the zombie sat in the waiting
queue); here the aborted request is the RUNNING chunked one, legitimately kept by
#791C until PP0's last frames for it ran -- and a new instance of its rid landed in
between. The frame is not waiting for any told: it continues a request this rank is
executing. Three rules, dual P followers only (``dual_p_kv_stage.armed``), none of
them reached by the flip / INT8 / NF forms:

  1. ``old_chunk_continued`` -- the #791T probe does not count a row entry as
     told-missing when the entry continues this rank's own chunked request of that
     rid (a DIFFERENT object than the queued one, its abort recorded and pending,
     the entry's prefix_len == its extend_range.end). The frame is then planned as
     before #791T: #791C keeps the chunk while PP0's schedule names it.
  2. ``newer_instance_queued`` -- when the old instance's chunked abort is applied
     at last, the rid-keyed store release (``release_aborted_request(rid)``) is
     skipped while a newer instance of the rid waits here: every rid-keyed record
     left (prefetch op, completed tokens, span pin) is the NEW instance's read
     (PP1 'WEG2-READ-STAGES req=weg2-0-235 pages=8192' at 19:26:06), and cutting
     it would leave PP1's ack/admission of instance 3 short of PP0's told. The old
     instance's own read ended at its admission (told=0 fallback / consumed).
  3. ``schedule_names_new_instance`` -- #791C keeps the old chunk while the
     schedule NAMES the rid. A row that names the rid at a start other than the old
     instance's continuation, while a newer instance with its told sits queued here,
     is PP0 admitting the NEW instance (PP0 applied its own abort before it could
     ever read instance 3): the old chunk is released in that pass and the new one
     is admitted off the row, instead of continuing the dead instance against PP0's
     geometry.

Residual (named, not covered): a new instance whose told equals EXACTLY where the
old instance stopped is indistinguishable on the row (same rid, same start); the
follower then runs it as the old instance's continuation over the same prompt
tokens -- the base behaviour.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

MARK_CONTINUED = "Q-695 OLD-INSTANCE CHUNK"
MARK_KEEP_READ = "Q-695 OLD-INSTANCE ABORT KEEPS THE NEW READ"
MARK_NEW_NAMED = "Q-695 NEW-INSTANCE NAMED"

_LOG_FIRST = 8
_LOG_EVERY = 256


def _armed(env=None) -> bool:
    from sglang.srt.weg2 import dual_p_kv_stage as _dpk

    return bool(_dpk.armed(env))


def _follower(sched: Any) -> bool:
    try:
        return int(getattr(getattr(sched, "ps", None), "pp_rank", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _count(sched: Any, attr: str) -> int:
    n = int(getattr(sched, attr, 0) or 0) + 1
    setattr(sched, attr, n)
    return n


def _loud(n: int) -> bool:
    return n <= _LOG_FIRST or n % _LOG_EVERY == 0


def _chunk_end(req: Any) -> Optional[int]:
    end = getattr(getattr(req, "extend_range", None), "end", None)
    if end is None:
        return None
    try:
        return max(0, int(end))
    except (TypeError, ValueError):
        return None


def _newer_queued(sched: Any, old: Any) -> list:
    rid = getattr(old, "rid", None)
    if rid is None:
        return []
    return [r for r in (getattr(sched, "waiting_queue", None) or ())
            if getattr(r, "rid", None) == rid and r is not old]


def old_chunk_continued(sched: Any, rid: Any, prefix_len: Any, slot_chunked: Any = None,
                        env=None) -> bool:
    """Rule 1: does the row entry ``(rid, prefix_len)`` continue this follower's own
    chunked request of ``rid`` -- an older instance whose abort is recorded and
    pending, while a newer instance of the rid waits in the queue? Then the entry
    is NOT waiting for a told (#791T must not defer it). False otherwise, and
    always False off the dual P layout or on PP0."""
    try:
        if not _armed(env) or not _follower(sched):
            return False
        pending = getattr(sched, "_pending_chunked_abort_req", None)
        if pending is None or getattr(pending, "rid", None) != rid:
            return False
        if pending is not getattr(sched, "chunked_req", None) and pending is not slot_chunked:
            return False
        end = _chunk_end(pending)
        if end is None or end != int(prefix_len):
            return False
        newer = _newer_queued(sched, pending)
        if not newer:
            return False
        n = _count(sched, "_q695_continued_n")
        if _loud(n):
            logger.warning(
                "%s rid=%s pp_rank=%s at=%d n=%d: the frame's row continues THIS rank's chunked "
                "instance of the rid (abort recorded, #791C keeps it while PP0 names it) -- the "
                "queued instance(s) %d of the same rid are newer and wait for their OWN told; "
                "no #791T defer for this entry (y8z 19:26:06 weg2-0-235)",
                MARK_CONTINUED, rid, getattr(sched.ps, "pp_rank", "?"), end, n, len(newer))
        return True
    except Exception:  # noqa: BLE001 - advisory: the base defer stands
        return False


def newer_instance_queued(sched: Any, old: Any, env=None) -> bool:
    """Rule 2: the old chunked instance's abort is being applied while a newer
    instance of its rid waits here -- skip the rid-keyed store release."""
    try:
        if not _armed(env) or not _follower(sched):
            return False
        newer = _newer_queued(sched, old)
        if not newer:
            return False
        n = _count(sched, "_q695_keep_read_n")
        if _loud(n):
            logger.warning(
                "%s rid=%s pp_rank=%s n=%d: the aborted chunked instance leaves, a newer instance "
                "of its rid waits here with its own store read -- the rid-keyed "
                "release_aborted_request is skipped (it would cut the NEW read); the old "
                "instance's KV goes back by its own object",
                MARK_KEEP_READ, getattr(old, "rid", "?"), getattr(sched.ps, "pp_rank", "?"), n)
        return True
    except Exception:  # noqa: BLE001 - advisory: the base release stands
        return False


def schedule_names_new_instance(sched: Any, old: Any, extents: Optional[Mapping], env=None) -> bool:
    """Rule 3: PP0's forwarded schedule names the old instance's rid at a start
    other than the old chunk's continuation, and a newer instance of the rid with
    its told on this rank is queued: the row admits the NEW instance -- release the
    old chunk now (the #791C 'keep' would continue a dead instance)."""
    try:
        if not _armed(env) or not _follower(sched) or not extents:
            return False
        rid = getattr(old, "rid", None)
        ext = extents.get(rid) if rid is not None else None
        if ext is None:
            return False
        start = int(ext[0])
        end = _chunk_end(old)
        if end is None or start == end:
            return False
        newer = _newer_queued(sched, old)
        if not newer:
            return False
        from sglang.srt.weg2 import p_intake as _p_intake

        if any(_p_intake.told_pending(sched, r) for r in newer):
            return False  # the #791T probe holds such a frame until the told lands
        n = _count(sched, "_q695_new_named_n")
        if _loud(n):
            logger.warning(
                "%s rid=%s pp_rank=%s start=%d old_end=%d n=%d: PP0's schedule names the rid at a "
                "start the old chunked instance does not continue, and a newer instance with its "
                "told waits here -- PP0 admits the NEW instance; the old chunk is released in this "
                "pass (#791C)", MARK_NEW_NAMED, rid, getattr(sched.ps, "pp_rank", "?"), start, end, n)
        return True
    except Exception:  # noqa: BLE001 - advisory: the base #791C verdict stands
        return False


MARK_OVERDUE = "Q-695 #791T OVERDUE FULL"


def _instances(sched: Any, rid: Any) -> str:
    out = []
    ch = getattr(sched, "chunked_req", None)
    if ch is not None and getattr(ch, "rid", None) == rid:
        out.append("chunked(end=%s,abort_pending=%s)" % (
            _chunk_end(ch), getattr(sched, "_pending_chunked_abort_req", None) is ch))
    for r in getattr(sched, "waiting_queue", None) or ():
        if getattr(r, "rid", None) == rid:
            out.append("queued(told=%s)" % (
                (getattr(sched, "_weg2_store_told", None) or {}).get(str(rid), "-")))
    for r in getattr(getattr(sched, "running_batch", None), "reqs", None) or ():
        if getattr(r, "rid", None) == rid:
            out.append("running")
    return "+".join(out) or "none"


def log_told_overdue(sched: Any, rids, decision: Any, mb_id: Any, env=None) -> None:
    """Dual P only: one ERROR line before the #791T stop with the FULL rids, the
    row's prefix for each and every instance this rank holds under that rid --
    y8y and y8z needed the debug-hold locals to learn which rid 'weg2-0-2' was."""
    try:
        if not _armed(env):
            return
        rows = {getattr(e, "rid", None): getattr(e, "prefix_len", None)
                for e in getattr(decision, "entries", ()) or ()}
        parts = ["%s@%s[%s]" % (r, rows.get(r, "?"), _instances(sched, r)) for r in list(rids)[:4]]
        logger.error("%s pp_rank=%s slot=%s rids=%s", MARK_OVERDUE,
                     getattr(getattr(sched, "ps", None), "pp_rank", "?"), mb_id, " ".join(parts))
    except Exception:  # noqa: BLE001 - a diagnostic never blocks the named stop
        pass
