# SPDX-License-Identifier: Apache-2.0
"""27B-PP-SKIPOUT-1024: ONE chunked slot per stage, and the slot agrees across the PP stages (SCHEDULER-SOLL v2 R11).

SPECIMEN. 27B int4ar, P = PP2 on 2x3080, image rc35i4ar3 = 27B fe572f3224,
boot ``boot_weg2_dkr27bint4arbar1int4ar10240630_fe572f3224_1024_063046.P.log``:

    06:36:00 PP0  chunked prefill PARKED: the pool can fund 0 tokens          (pdflip-3-7 at 25157 of 40067, #679)
    06:36:02 PP0  PDFLIP-PARK n=1 rid=pdflip-3-7 span=25157 of 40067           (rows back to the tree, slot EMPTY)
    06:36:02 PP0  #969N ADMIT slot=1 fwd_ct=26 extend=4096 rids=[pdflip-3-8]    (PP0 mints pdflip-3-8 as ITS continuation)
    06:36:04 PP1  #992 CHUNKED CONTINUATION NOT NAMED rid=pdflip-3-7           (PP1 still holds pdflip-3-7 in its slot)
    06:36:04 PP1  [#967] SECOND CONTINUATION REFUSED rid=pdflip-3-8 site=_add_scheduled_req
    06:36:05 PP1  #969N ADMIT slot=1 fwd_ct=26 extend=4096 rids=[pdflip-3-8]    (the chunk runs anyway, #995 "do not announce")
    06:36:08 PP1  AssertionError: PP skip output comm invariant violated: req pdflip-3-8 has
                  inflight_middle_chunks=0 but output was skipped (contains_last_prefill_chunk=False)

The PDFLIP-PARK is PP0's: it happens at the head of a step, after the #679 in-place park of a pass whose decision
named nothing, so no follower ever reaches ``add_chunked_req`` for it (#992 skips it) and no follower parks. From then
on the stages disagree about ONE field, ``scheduler.chunked_req``, and every reader of that field disagrees with them:

  (1) the #995 branch of ``PrefillAdder._add_scheduled_req`` runs PP0's new continuation as an UNANNOUNCED middle chunk:
      no ``inflight_middle_chunks`` increment (the assert above), no stash of its KV, and the merge would have put a
      half-prefilled request into the follower's running batch -- the state ``PrefillAdder._mint_chunked`` describes as
      "tracked by nobody, gets treated as a finished prefill";
  (2) ``contains_last_prefill_chunk`` asked "is there a chunked request" instead of "is the batch's only member the
      chunked request": on the follower the stale continuation made a final chunk look like a middle one -> the follower
      skips the output send while PP0 (slot empty) receives -> hang;
  (3) ``inflight_middle_chunks += 1`` ran for the chunked request whether or not it was in the batch: a continuation
      that took no seat collected a count nothing ever decrements, and its eventual final chunk reads as a middle chunk.

THE FIX keeps the assert (the invariant is right) and corrects the three paths:

  (1) DISPLACE. A follower that is told to run a FRESH request's NOT-LAST chunk (no anchor body) while its own slot is
      occupied has been told, by construction, that PP0's slot was empty: PP0 can mint a continuation only with
      ``chunked_req is None`` (scheduler.py asserts it; #959 refuses every other mint). So the resident continuation is
      no longer anybody's continuation on PP0 -- PP0 parked it (PDFLIP-PARK) or otherwise let it go. The follower mints
      the named request (as PP0 did) and hands the old resident to ``displace_resident``: out of the slot NOW, rows
      given back (``release_kv_cache(is_insert=True)``, the PDFLIP-PARK give-back) at the head of a later step once no
      launched chunk of it still writes (``inflight_middle_chunks == 0``), back to the head of the waiting queue so the
      decision that re-admits it later finds it. Decided from forwarded facts only (the carried last-chunk verdict and
      the forwarded extent), never from a rank-local quantity: R11.
  (2) ``contains_last_prefill_chunk`` asks for the SEAT (anchor_tails.py).
  (3) the middle-chunk count is taken only for a SEATED continuation (``counts_middle_chunk``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)

#: scheduler attribute: continuations displaced from the slot whose rows are still to be given back
DISPLACED_ATTR = "_pp_displaced_continuations"
#: request attribute: an abort that reached this stage while the request sat on the displaced list, as ``(finished_reason,)``
ABORT_ATTR = "_pp_displaced_abort"


def seated(req: Any, can_run_list: Iterable[Any]) -> bool:
    """``req`` takes a seat in the batch this pass builds (identity, as everywhere the scheduler compares requests)."""
    return req is not None and any(r is req for r in (can_run_list or ()))


def counts_middle_chunk(chunked_req: Any, can_run_list: Iterable[Any]) -> bool:
    """(3): the batch builder's ``chunked_req.inflight_middle_chunks += 1`` is owed exactly when a chunk of it RUNS in
    this batch. Its only decrement is in ``process_batch_result_prefill``, which walks ``batch.reqs`` -- a continuation
    outside the batch (the #679 in-place park, the lane hold of FIX 4, the #992 not-named seat) would keep the count for
    ever: the head-of-step park waits for it to drain, and the final chunk of the continuation reads as a middle chunk
    (no first token on that stage while the others emit one). Supersedes the lane-only ``lanes_p.held_without_seat``,
    which is the lane case of the same membership question."""
    return chunked_req is not None and seated(chunked_req, can_run_list)


def displaced(sched: Any) -> List[Any]:
    """The continuations displaced on this stage whose rows have not gone back yet (a live list, possibly empty)."""
    return sched.__dict__.setdefault(DISPLACED_ATTR, [])


def is_displaced(sched: Any, req: Any) -> bool:
    return req is not None and any(r is req for r in sched.__dict__.get(DISPLACED_ATTR) or ())


def _held_rows(req: Any) -> int:
    pi = getattr(req, "prefix_indices", None)
    try:
        return 0 if pi is None else len(pi)
    except Exception:  # noqa: BLE001 - instrument only
        return -1


def _pool_free(sched: Any) -> Any:
    al = getattr(sched, "token_to_kv_pool_allocator", None)
    try:
        return al.available_size() if al is not None else None
    except Exception:  # noqa: BLE001 - instrument only
        return None


def displace_resident(sched: Any, new_req: Any, *, can_run_list: Iterable[Any], site: str = "_add_scheduled_req") -> Any:
    """(1), the scheduler half: the adder minted ``new_req`` while ``sched.chunked_req`` was occupied (the adder only
    does that on a follower, on PP0's word -- see the module doc). Empty the slot for the scheduler's
    ``assert self.chunked_req is None``, keep the old resident for the give-back. Returns the displaced request.

    Loud on the shapes that would mean the adder's flag and the scheduler's slot disagree -- never a state to continue
    from:
      * an empty slot, or the named request itself in it;
      * REVIEW A2: the resident is SEATED in this very batch. The premise "PP0 minted, so PP0 let my resident go" has
        two holes in which the resident still runs a chunk here: the #996 fallback (no carried verdict, the local fill
        keeps the resident "not last" while PP0 ran its LAST chunk) and an H42 END-ANCHOR body PP0 re-admitted as a tail
        while this stage carries it as ``chunked_req``. Displacing it would run its chunk uncounted, feed its output after
        a not-last chunk, merge it into the running batch and give its rows back while it decodes. Before the fix those
        corners died at the #959/#995 assert; they keep dying loudly, by name.

    S1 (review): the follower holds the resident's rows and GDN slot until the head of a later step (PP0 gave them back
    before it planned this pass). Not given back mid-pass: a tree insert after admission and a requeue that an own-void
    restore of this pass would have to undo are a worse trade than one pass of rows. Instrumented instead: ``held_rows``,
    ``mamba_slot`` and ``pool_free`` name the follower-only overlap if it ever binds."""
    old = getattr(sched, "chunked_req", None)
    if old is None or old is new_req:
        raise RuntimeError(
            "27B-PP-SKIPOUT DISPLACE WITHOUT A RESIDENT: the adder displaced the slot for rid=%s at %s but the "
            "scheduler's chunked_req is %s" % (getattr(new_req, "rid", "?"), site, "empty" if old is None else "the same request")
        )
    if seated(old, can_run_list):
        raise RuntimeError(
            "27B-PP-SKIPOUT DISPLACE OF A SEATED RESIDENT: rid=%s runs a chunk in this batch (extend=%s) and is this "
            "stage's chunked_req, while the forwarded decision mints rid=%s as a new continuation at %s. PP0 cannot have "
            "let a request go that it schedules in the same pass: the stages disagree about the continuation (#996 "
            "fallback without a carried verdict, or an END-ANCHOR body carried as chunked_req). Stopping by name instead "
            "of running it uncounted (SOLL R11)."
            % (getattr(old, "rid", "?"), getattr(old, "extend_range", None), getattr(new_req, "rid", "?"), site)
        )
    sched.chunked_req = None
    displaced(sched).append(old)
    n = int(getattr(sched, "_pp_displace_n", 0) or 0) + 1
    sched._pp_displace_n = n
    if n <= 8 or n % 256 == 0:
        logger.info(
            "PP-SLOT DISPLACE n=%d pp_rank=%s resident=%s local_end=%s inflight=%s held_rows=%s mamba_slot=%s pool_free=%s "
            "new=%s site=%s: the forwarded decision mints rid=%s as a continuation, which PP0 can only do with its own slot "
            "empty -- PP0 let the resident go (PDFLIP-PARK). This stage follows: the resident leaves the slot now and gives "
            "its rows back once no launched chunk of it still writes (SOLL R11, 27B-PP-SKIPOUT-1024)",
            n,
            getattr(getattr(sched, "ps", None), "pp_rank", "?"),
            getattr(old, "rid", "?"),
            getattr(getattr(old, "extend_range", None), "end", None),
            getattr(old, "inflight_middle_chunks", None),
            _held_rows(old),
            getattr(old, "mamba_pool_idx", None),
            _pool_free(sched),
            getattr(new_req, "rid", "?"),
            site,
            getattr(new_req, "rid", "?"),
        )
    return old


def mark_aborts(sched: Any, rid: Any, abort_all: bool, finished_reason: Any = None) -> List[Any]:
    """REVIEW A1: an AbortReq for a displaced continuation. It is in no queue, no slot and no batch on this stage, so
    every other abort route (``_abort_request_now``'s slot / waiting-queue / tail scans, the #1180-W waiting hold) misses
    it, and the drain would queue it again on the follower only -- a zombie PP0 never names (R11). Marked here, applied
    by the drain (``abort`` callback) once no launched chunk of it still writes. A request back in the slot (an own-void
    restore, A3) is the slot abort's, not this one's. Returns the marked requests."""
    out: List[Any] = []
    slot = getattr(sched, "chunked_req", None)
    want = str(rid or "")
    for req in sched.__dict__.get(DISPLACED_ATTR) or ():
        if req is slot:
            continue
        if abort_all or (want and str(getattr(req, "rid", "")).startswith(want)):
            setattr(req, ABORT_ATTR, (finished_reason,))
            out.append(req)
            logger.info(
                "PP-SLOT DISPLACED ABORT recorded rid=%s pp_rank=%s abort_all=%s: applied when no launched chunk of it "
                "still writes (27B-PP-SKIPOUT-1024 review A1)",
                getattr(req, "rid", "?"), getattr(getattr(sched, "ps", None), "pp_rank", "?"), bool(abort_all),
            )
    return out


def drain_displaced(
    sched: Any,
    give_back: Callable[[Any], Any],
    *,
    abort: Optional[Callable[[Any], Any]] = None,
    requeue: Optional[Callable[[Any], Any]] = None,
) -> int:
    """(1), the give-back half, at the head of a step (beside ``process_pending_pdflip_park``). Returns how many left.

    * REVIEW A3: it is ``sched.chunked_req`` again (an own-void restore -- ``pp_void_pass`` frame form,
      ``scheduler_pp_mixin`` restore of ``chunked_before`` -- undid the displacing pass): the displacement is void. It
      leaves the list untouched (no give-back, no reset: the stash at the head of this step reads its ``extend_range``,
      the instr19 death), loudly; an abort recorded while it was displaced moves to the slot's chunked abort;
    * its chunked abort is pending (``_pending_chunked_abort_req``): left to ``process_pending_chunked_abort``;
    * finished: dropped, nothing to give back;
    * a launched chunk of it still writes (``inflight_middle_chunks > 0``): kept, retried next step -- the same rule
      ``process_pending_pdflip_park`` applies before ``release_kv_cache``;
    * an abort was recorded for it (A1): ``abort(req)`` -- the chunked-abort teardown -- and dropped;
    * REVIEW S3: live but without rows (``req_pool_idx is None``): ``requeue(req)`` with a WARNING -- never silently out
      of every slot and queue;
    * otherwise ``give_back(req)`` -- the PDFLIP-PARK give-back -- and dropped.
    A callback that returns False keeps the request for the next step rather than losing it."""
    pend = sched.__dict__.get(DISPLACED_ATTR)
    if not pend:
        return 0
    keep: List[Any] = []
    gone = 0
    slot = getattr(sched, "chunked_req", None)
    abort_target = getattr(sched, "_pending_chunked_abort_req", None)
    for req in list(pend):
        rid = getattr(req, "rid", "?")
        if req is slot:
            marked = getattr(req, ABORT_ATTR, None)
            logger.warning(
                "PP-SLOT DISPLACE UNDONE rid=%s pp_rank=%s: the request is this stage's chunked_req again (an own-void "
                "restore undid the displacing pass); the displacement is void, nothing is given back%s "
                "(27B-PP-SKIPOUT-1024 review A3)",
                rid, getattr(getattr(sched, "ps", None), "pp_rank", "?"),
                "; its recorded abort moves to the slot's chunked abort" if marked else "",
            )
            if marked:
                try:
                    delattr(req, ABORT_ATTR)
                except Exception:  # noqa: BLE001
                    pass
                if getattr(sched, "_pending_chunked_abort_req", None) is None:
                    sched._pending_chunked_abort_req = req
                    sched._pending_chunked_abort_reason = marked[0]
                    sched._pending_chunked_abort_delay = 0
            continue
        if req is abort_target:
            keep.append(req)
            continue
        try:
            finished = bool(req.finished())
        except Exception:  # noqa: BLE001 - a stand-in without finished() is live
            finished = False
        if finished:
            continue
        marked = getattr(req, ABORT_ATTR, None)
        if getattr(req, "req_pool_idx", None) is not None and int(getattr(req, "inflight_middle_chunks", 0) or 0) > 0:
            keep.append(req)
            continue
        if marked is not None and abort is not None:
            if abort(req) is False:
                keep.append(req)
                continue
            gone += 1
            continue
        if getattr(req, "req_pool_idx", None) is None:
            logger.warning(
                "PP-SLOT DISPLACED WITHOUT ROWS rid=%s pp_rank=%s: a live displaced request holds no pool row; queued "
                "without a give-back so PP0's next naming finds it (27B-PP-SKIPOUT-1024 review S3)",
                rid, getattr(getattr(sched, "ps", None), "pp_rank", "?"),
            )
            if requeue is not None and requeue(req) is False:
                keep.append(req)
                continue
            gone += 1
            continue
        if give_back(req) is False:
            keep.append(req)
            continue
        gone += 1
    pend[:] = keep
    return gone


def forget_displaced(sched: Any, req: Any) -> None:
    """Drop ``req`` from the displaced list (its chunked abort was applied)."""
    pend = sched.__dict__.get(DISPLACED_ATTR)
    if pend:
        pend[:] = [r for r in pend if r is not req]


def assert_no_unannounced_middle_chunk(
    sched: Any, can_run_list: Iterable[Any], chunked_req: Any, tails: Iterable[Any], told_last_chunk: Callable[[Any], Any]
) -> None:
    """REVIEW S4: the invariant the old slot question in ``contains_last_prefill_chunk`` held by accident, stated. Every
    batch member that runs a NOT-LAST chunk is a continuation somebody tracks -- ``chunked_req`` or an anchor tail --
    because only those take the middle-chunk count, stay out of the merge and get their chunk stashed. "Last" is read
    exactly as the adder decides it: the carried verdict when the decision sent one (#996), else the local fill. PP only
    (pp_size > 1): the skip of the output exchange that turns a violation into a hang exists only there. A violation
    stops the group by name (the 27B 1024_063046 / NF 1016_163900 shape)."""
    ps = getattr(sched, "ps", None)
    if int(getattr(ps, "pp_size", 1) or 1) <= 1:
        return
    tails = tuple(tails or ())
    for r in can_run_list or ():
        if r is chunked_req or any(r is t for t in tails):
            continue
        er = getattr(r, "extend_range", None)
        end = getattr(er, "end", None)
        if end is None:
            continue
        told = told_last_chunk(r)
        if told is None:
            fill = getattr(r, "full_untruncated_fill_ids", None)
            if fill is None:
                continue
            last = int(end) >= len(fill)
        else:
            last = bool(told)
        if not last:
            raise RuntimeError(
                "27B-PP-SKIPOUT UNANNOUNCED MIDDLE CHUNK rid=%s pp_rank=%s extend=%s fill=%s told_last=%s: a NOT-LAST chunk "
                "runs for a request that is neither chunked_req (%s) nor an anchor tail -- it would take no middle-chunk "
                "count, be merged into the running batch and consume an output after a not-last chunk. Stopping by name "
                "(review S4, SOLL R11)."
                % (getattr(r, "rid", "?"), getattr(ps, "pp_rank", "?"), er,
                   len(getattr(r, "full_untruncated_fill_ids", None) or ()), told, getattr(chunked_req, "rid", None))
            )
