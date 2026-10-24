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
from typing import Any, Callable, Iterable, List

logger = logging.getLogger(__name__)

#: scheduler attribute: continuations displaced from the slot whose rows are still to be given back
DISPLACED_ATTR = "_pp_displaced_continuations"


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


def displace_resident(sched: Any, new_req: Any, *, site: str = "_add_scheduled_req") -> Any:
    """(1), the scheduler half: the adder minted ``new_req`` while ``sched.chunked_req`` was occupied (the adder only
    does that on a follower, on PP0's word -- see the module doc). Empty the slot for the scheduler's
    ``assert self.chunked_req is None``, keep the old resident for the give-back. Returns the displaced request.

    Loud on the two shapes that would mean the adder's flag and the scheduler's slot disagree (an empty slot, or the
    named request itself in it): both are a defect in this mechanism, never a state to continue from."""
    old = getattr(sched, "chunked_req", None)
    if old is None or old is new_req:
        raise RuntimeError(
            "27B-PP-SKIPOUT DISPLACE WITHOUT A RESIDENT: the adder displaced the slot for rid=%s at %s but the "
            "scheduler's chunked_req is %s" % (getattr(new_req, "rid", "?"), site, "empty" if old is None else "the same request")
        )
    sched.chunked_req = None
    displaced(sched).append(old)
    n = int(getattr(sched, "_pp_displace_n", 0) or 0) + 1
    sched._pp_displace_n = n
    if n <= 8 or n % 256 == 0:
        logger.info(
            "PP-SLOT DISPLACE n=%d pp_rank=%s resident=%s executed=%s inflight=%s new=%s site=%s: the forwarded decision "
            "mints rid=%s as a continuation, which PP0 can only do with its own slot empty -- PP0 let the resident go "
            "(PDFLIP-PARK). This stage follows: the resident leaves the slot now and gives its rows back once no launched "
            "chunk of it still writes (SOLL R11, 27B-PP-SKIPOUT-1024)",
            n,
            getattr(getattr(sched, "ps", None), "pp_rank", "?"),
            getattr(old, "rid", "?"),
            getattr(getattr(old, "extend_range", None), "end", None),
            getattr(old, "inflight_middle_chunks", None),
            getattr(new_req, "rid", "?"),
            site,
            getattr(new_req, "rid", "?"),
        )
    return old


def drain_displaced(sched: Any, give_back: Callable[[Any], None]) -> int:
    """(1), the give-back half, at the head of a step (beside ``process_pending_pdflip_park``). Returns how many went back.

    * a launched chunk of it still writes (``inflight_middle_chunks > 0``): kept, retried next step -- the same rule
      ``process_pending_pdflip_park`` applies before ``release_kv_cache``;
    * its chunked abort is pending (``_pending_chunked_abort_req``): left to ``process_pending_chunked_abort``, which
      knows a displaced request (it is not ``chunked_req`` any more);
    * finished (aborted) or already without rows: dropped from the list, nothing to give back;
    * otherwise ``give_back(req)`` -- the PDFLIP-PARK give-back -- and dropped from the list; a give-back that
      returns False (the release raised) keeps it for the next step rather than losing it out of every slot and queue."""
    pend = sched.__dict__.get(DISPLACED_ATTR)
    if not pend:
        return 0
    keep: List[Any] = []
    gone = 0
    abort_target = getattr(sched, "_pending_chunked_abort_req", None)
    for req in list(pend):
        if req is abort_target:
            keep.append(req)
            continue
        try:
            finished = bool(req.finished())
        except Exception:  # noqa: BLE001 - a stand-in without finished() is live
            finished = False
        if finished or getattr(req, "req_pool_idx", None) is None:
            continue
        if int(getattr(req, "inflight_middle_chunks", 0) or 0) > 0:
            keep.append(req)
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
