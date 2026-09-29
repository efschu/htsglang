"""H91 Teil B: the scheduler half of the D park (policy: ``weg2/d_seats.py``).

Collaborator of ``Scheduler`` (large-class-style: the frozen orchestrator
delegates, the domain logic lives here).  Every function takes the LIVE
scheduler because what it moves is the scheduler's own bookkeeping -- which
list a request sits in (``running_batch``, ``waiting_queue``,
``weg2_d_parked``, ``weg2_dormant_hold``) -- and nothing narrower describes
that.  The verdicts themselves (order, gate, due) are d_seats functions of
replicated state, so every rank moves the same requests at the same point.

Entry points (all no-ops off group D / with nothing parked):
  park_running   -- ``POST /weg2/park_running`` (front, before D's sleep)
  hold_late_arrival -- H91c3: a hand-off reaching D after the park is held
  hold_parked    -- the sleep leg's dormant point (weight_updater)
  park_tick      -- every pass: a park whose sleep never came re-queues
  note_retracted -- a decode-pressure retraction is a PRESSURE park
  admission      -- D's admission verdict for one pass
  park_abort     -- an abort reaches the parked list
  note_wake_seats -- H95: the wake of D fixes the phase's seat count n
  seat_vram_wake  -- H95c: every D resume request maps the posts of n seats
  seat_cap        -- H95c: D admits at most n running requests
  seat_guard      -- H95c: a forward batch wider than n is refused (W-SEAT)

H91d: the parked request's MTP draft rows ride along (``d_park_draft``):
saved before the flip park's retraction, dropped with an abort.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from sglang.srt.managers import weg2_resumable_depth
from sglang.srt.weg2 import d_park_draft, d_park_read, d_seats

logger = logging.getLogger(__name__)


def parked_list(sched) -> list:
    parked = getattr(sched, "weg2_d_parked", None)
    if parked is None:
        parked = sched.weg2_d_parked = []
    return parked


#: H91c3-2: set by park_running (the park's rank-local stamp) while every NEW
#: arrival is held behind the park; None = no park open. Closed by the sleep
#: (hold_parked) or by the awake re-queue (park_tick).
LATE_HOLD_ATTR = "_weg2_d_park_late_since"

#: PARK-SETTLE (28.09.): a #1471 post-wake settle request the park folded into
#: its list; an awake re-queue gives it back to the settle, not to the queue.
FROM_SETTLE_ATTR = "_weg2_park_from_settle"


def rearm_window_draft_cold(sched, reqs) -> int:
    """27B PARK (DFlash2): a flip-parked request on a SLOT-MAPPED draft pool
    (the DFlash window pool, ``dflash_solo_pool.DraftKVSlotMapper``) resumes
    exactly like a fresh hand-off -- its draft admission is armed again.

    Why only there, and why this is enough (code read at the park's tree):
    the window pool is not indexed by target slots, the sleep's tree flush
    frees the parked span's target slots and the mapper drops their draft
    rows with them; the resume loads the prefix back into NEW target slots,
    which the mapper does not know -> the window reads the HOLE slot (zeros,
    corrected out of the softmax under SGLANG_DFLASH_WINDOW_HOLE_MASK) until
    the resumed decode writes real rows. That is the state of EVERY 27B
    hand-off from P (P computes no draft, the draft tier is off), so nothing
    has to be carried (H91d's MTP carry skips this pool by name) and the
    target's verify keeps every emitted token exact. What differed was only
    the admission's bookkeeping: ``COLD_ARMED_ATTR`` from the first admission
    survives ``reset_for_retract``, so the resume skipped the draft-cold
    evaluation a fresh hand-off gets -- no ``WEG2 DRAFT-COLD rid=... of N
    prefix pages`` line (the one per-rid reading of how much prefix came
    back), a wrong ``draft_cold`` label in the accept profile. Cleared here,
    the resume is evaluated like the hand-off it now is. An MTP pool (NF,
    target-slot indexed, H91d carry) is untouched. No collective, nothing
    feeds a scheduling decision; rank-local attribute only."""
    if not reqs:
        return 0
    from sglang.srt.managers.phase_flip_draft_bootstrap import COLD_ARMED_ATTR, draft_kv_pool

    pool = draft_kv_pool(getattr(sched, "draft_worker", None))
    if pool is None or getattr(pool, "weg2_slot_mapper", None) is None:
        return 0
    n = 0
    for req in reqs:
        if getattr(req, COLD_ARMED_ATTR, False):
            setattr(req, COLD_ARMED_ATTR, False)
            n += 1
    return n


def park_running(sched, recv_req, *, late_hold_armed: bool = False):
    """Retract every running D request RETAINING its span (KV, the node's
    GDN/Mamba anchor, the draft rows) with a forced host write-through -- the
    sleep flushes the tree and the store is the only copy that survives it
    (#969D/#1068) -- and keep it, with whatever only queued on D, in
    ``weg2_d_parked`` (the sleep asserts an idle group). The in-flight batch
    lands first, in upstream ``pause_generation``'s retract shape. Nothing is
    aborted and nothing is told to the tokenizer: the streams stay open.

    H91c3-2, ``late_hold_armed`` (the scheduler passes #1443's dormant admit):
    a hand-off the front had already sent (``D.outstanding``) may reach this
    scheduler only AFTER the park (tokenizer / HTTP pipe). It was admitted and
    decoded to its end while the front's drain waited for it. From the park
    until the sleep (or the awake re-queue) every new arrival is held behind
    the park instead (:func:`hold_late_arrival`), and the answer says so
    (``late_hold``) -- the front then counts its in-flight hand-offs as
    parked. Only with the dormant hold armed: an arrival after the sleep is
    then held too (#1443) instead of refused (W25)."""
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqOutput
    from sglang.srt.mem_cache.base_prefix_cache import FORCE_HOST_WRITE_THROUGH_ATTR

    epoch = int(getattr(recv_req, "epoch", 0) or 0)
    reason = str(getattr(recv_req, "reason", "") or "")
    if not d_seats.d_flip_park_active():
        return Weg2ParkRunningReqOutput(
            success=False, parked=[], epoch=epoch,
            message="W-PARK refused: not group D (or SGLANG_WEG2_D_PARK=0, or neither the "
                    "standard form nor SGLANG_WEG2_D_PARK_IMMEDIATE) -- nothing parked",
        )
    parked = parked_list(sched)
    if getattr(sched, "weg2_dormant", False):
        return Weg2ParkRunningReqOutput(
            success=True, parked=[str(r.rid) for r in parked], epoch=epoch,
            message="group D is dormant: nothing runs, the listed requests were parked earlier",
        )
    if getattr(sched, "anchor_tails", None):
        return Weg2ParkRunningReqOutput(
            success=False, parked=[], epoch=epoch,
            message="W-PARK refused: anchor tails present (a P-group structure) -- nothing parked",
        )
    if sched.enable_overlap and sched.last_batch and sched.result_queue:
        tmp_batch, tmp_result = sched.result_queue.popleft()
        sched.process_batch_result(tmp_batch, tmp_result)
    last = sched.last_batch
    if last and last.forward_mode.is_extend():
        last.filter_batch(chunked_req_to_exclude=[])
        if not last.is_empty():
            if sched.running_batch.is_empty():
                sched.running_batch = last
            else:
                sched.running_batch.merge_batch(last)
    sched.last_batch = None
    running = []
    if not sched.running_batch.is_empty():
        sched.running_batch.filter_batch()
        running = list(sched.running_batch.reqs)
    for req in running:
        setattr(req, FORCE_HOST_WRITE_THROUGH_ATTR, True)
        # PARK-RETAIN READ: the retraction's insert stamps what it retains;
        # a stamp left from an earlier insert must not stand in for it.
        setattr(req, d_park_read.RETAINED_ATTR, None)
    # H91d: the draft rows have no host twin (tier off) -- copy them off
    # BEFORE the retraction hands the slots to the tree (d_park_draft).
    d_park_draft.save_parked(sched, running, site=d_seats.SITE_FLIP)
    # 27B PARK: a DFlash window pool carries nothing; the resume is a fresh
    # hand-off's admission (rearm_window_draft_cold). MTP pools: no-op.
    rearmed = rearm_window_draft_cold(sched, running)
    retracted = (
        sched.running_batch.retract_all(sched.server_args, offload_kv=False, retain=True)
        if running else []
    )
    sched.running_batch.batch_is_full = False
    sched.chunked_req = None
    now = time.monotonic()
    for req in retracted:
        d_seats.mark_parked(req, d_seats.SITE_FLIP, epoch=epoch, now=now)
        sched._969ad_note_retract(req, "weg2_park_running")
        # STALE-DELIVERED (b23 10:23:15, weg2-24-100): a flip park opens a NEW
        # read cycle. The #1324 stamp of the previous cycle's read (79103) made
        # the #1471 wake settle release a request whose THIS-cycle read had
        # answered zero ("SETTLE-TAIL delivered=79103 remainder=1405"); its
        # admission found nothing (state=cold host_hit=0) and the X gate
        # refused it mid-stream (W50, client stream dead). The stamp belongs
        # to one read -- cleared here, set again by this cycle's own read.
        # W88 CYCLE: and with it the progress witness and the store-short
        # bound of that read (d_park_read.READ_CYCLE_ATTRS).
        d_park_read.clear_read_cycle(req)
        # PARK-RETAIN READ: the store read of this request ends at what the
        # retraction just retained (the KV above the mamba track was freed).
        if d_park_read.stamp_parked(req) is not None:
            logger.info("WEG2-D-PARK RETAINED rid=%s %s", str(req.rid)[:12], d_park_read.describe(req))
    queued = list(sched.waiting_queue)
    sched.waiting_queue = []
    # PARK-SETTLE (28.09., 27B park boot 27.09. 10:24:34, rid weg2-58-201): a
    # #1471 post-wake settle request (its store read was still short at the
    # wake) is D work that joins the queue by itself the moment its re-read
    # completes -- it did 1 s after the park (#988 LOADBACK 10:24:35) and was
    # decoded to its end while the front's D->P drain waited for it. And
    # because a pending settle also withdrew the late hold, a hand-off still in
    # the pipe (weg2-68-220) was admitted and decoded too: the park was
    # PARTIAL, the flip stalled, the parked ones lapsed (30 s) and ran to their
    # end as well -- the over-X request that fired the park waited 171 s. The
    # park takes the settle into its list as held work (no park site): the
    # sleep holds it, the wake's #1471 verdict reads it again; an awake
    # re-queue gives it back to the settle (park_tick). Replicated: the settle
    # list is built and shrunk by group-MIN verdicts only.
    settle = list(getattr(sched, "weg2_post_wake_settle", None) or [])
    if settle:
        sched.weg2_post_wake_settle = []
        for req in settle:
            setattr(req, FROM_SETTLE_ATTR, True)
    sched.weg2_d_parked = d_seats.order_waiting(list(parked) + list(retracted) + settle + queued)
    # #248: every parked request is kept by ORDER over the flip -- the
    # sleep's reset gives its references back, the hold reads it at the wake
    try:
        from sglang.srt.weg2 import park_l3

        park_l3.mark_parked(sched, sched.weg2_d_parked)
    except Exception:  # noqa: BLE001 - the order is an improvement, never a wall
        logger.warning("#248 PARK-MARK failed", exc_info=True)
    # H91c2: park_tick's awake requeue is the net for a sleep that never comes
    # after THIS park, so its clock starts now for every request the park
    # holds. A request decode pressure parked earlier kept its older stamp and
    # awake_requeue_due reads the OLDEST: past 30 s the next pass re-queued the
    # whole park, D decoded the parked requests again while the front, told
    # they were parked, flipped (quiesce never idle -> W3). Rank-local
    # monotonic, like every stamp here; park_tick MIN-reduces the verdict.
    for req in sched.weg2_d_parked:
        setattr(req, d_seats.SINCE_ATTR, now)
    sched._weg2_d_park_slept = False
    # PARK-SETTLE: the settle is in the park list now (above), so nothing
    # releases into the queue behind the park's back -- the late hold holds.
    late_hold = bool(late_hold_armed)
    setattr(sched, LATE_HOLD_ATTR, now if late_hold else None)
    rids = [str(r.rid) for r in sched.weg2_d_parked if d_seats.park_site(r) is not None]
    held = [str(r.rid) for r in sched.weg2_d_parked if d_seats.park_site(r) is None]
    # #59b: the depth each parked request resumes from, after the retraction
    # above retained its span (every rank parks the same list).
    resumable = weg2_resumable_depth.park_depths(
        getattr(sched, "tree_cache", None),
        [r for r in sched.weg2_d_parked if d_seats.park_site(r) is not None],
        getattr(sched, "ps", None),
    )
    # PARK-READ = RESUMABLE: the read of a request this park retracted ends at
    # the depth it resumes from, not at the tombstoned KV above the anchor
    # (d_park_read.clamp_to_resumable). Group-uniform: #59b's depths are.
    if resumable:
        bigram = bool(getattr(getattr(sched, "tree_cache", None), "is_eagle", False))
        for req in retracted:
            before = d_park_read.read_cap(req)
            if d_park_read.clamp_to_resumable(
                req, resumable.get(str(req.rid)), is_bigram=bigram
            ) is not None:
                logger.info("WEG2-D-PARK READ=RESUMABLE rid=%s cap %s -> %s",
                            str(req.rid)[:12], before, d_park_read.describe(req))
    logger.info(
        "WEG2-D-PARK park_running epoch=%d reason=%s: %d running retracted (span retained, "
        "forced host write-through), parked=%s queued-behind=%s settle-folded=%s late_hold=%s -- "
        "the sleep holds them first, the wake resumes oldest first%s",
        epoch, reason, len(retracted), [r[:12] for r in rids], [r[:12] for r in held],
        [str(r.rid)[:12] for r in settle], late_hold,
        (" (DFlash window draft: %d resume(s) re-armed like a fresh hand-off)" % rearmed
         if rearmed else ""),
    )
    return Weg2ParkRunningReqOutput(
        success=True, parked=rids, held=held, epoch=epoch, late_hold=late_hold,
        message="parked %d, queued behind them %d" % (len(rids), len(held)),
        weg2_resumable_depth=resumable,
    )


def hold_late_arrival(sched, req) -> bool:
    """H91c3-2: a NEW request (never a re-queue) that reaches D between
    ``park_running`` and the sleep joins the park's list as held (no park
    site, behind the parked ones) instead of the waiting queue -- the admission
    loop never sees it, the sleep moves it into the #1443 hold with the rest,
    an awake re-queue brings it back with the rest. Its requeue clock is the
    park's (``awake_requeue_due`` reads the oldest). True = held; the caller
    returns. REPLICATED: the park is a broadcast control request and the
    intake order is the group's, so every rank holds the same arrivals."""
    since = getattr(sched, LATE_HOLD_ATTR, None)
    if since is None or getattr(sched, "weg2_dormant", False) or not d_seats.d_flip_park_active():
        return False
    setattr(req, d_seats.SINCE_ATTR, since)
    parked = parked_list(sched)
    parked.append(req)
    logger.info("WEG2-D-PARK late-hold rid=%s: arrived after park_running (a hand-off in flight "
                "at the park), held behind the park (%d in the park list) -- the front counts it "
                "parked", str(req.rid)[:12], len(parked))
    return True


def hold_parked(sched, *, hold_armed: bool) -> int:
    """The sleep leg's dormant point (``weg2_dormant`` just set, HiCache
    drained, tree flushed): the parked requests enter the #1443 dormant hold
    FIRST, oldest first, their storage prefetch issued by the ordinary intake
    so it runs during the flip. Hold not armed: they stay parked and the first
    awake pass re-queues them."""
    setattr(sched, LATE_HOLD_ATTR, None)  # H91c3-2: the sleep closes the late hold
    parked = list(getattr(sched, "weg2_d_parked", None) or [])
    if not parked:
        return 0
    sched._weg2_d_park_slept = True
    if not hold_armed:
        logger.info("WEG2-D-PARK hold: SGLANG_WEG2_DORMANT_ADMIT off -- %d parked request(s) "
                    "wait for the wake", len(parked))
        return 0
    sched.weg2_d_parked = []
    for req in parked:
        # PARK-SETTLE: held over the sleep, a folded settle request is hold work
        # like the rest -- the wake's #1471 verdict reads it again.
        setattr(req, FROM_SETTLE_ATTR, False)
        sched._add_request_to_queue(req, is_retracted=True)
    hold = getattr(sched, "weg2_dormant_hold", None)
    if hold is None:
        return 0
    moved = [r for r in parked if any(r is h for h in hold)]
    ids = {id(r) for r in moved}
    hold[:] = d_seats.order_waiting(moved) + [h for h in hold if id(h) not in ids]
    logger.info("WEG2-D-PARK hold: %d parked request(s) at the head of the dormant hold %s "
                "(prefetch issued during the flip; the wake releases them first)",
                len(moved), [str(r.rid)[:12] for r in moved])
    return len(moved)


def _to_queue_head(sched, reqs) -> list:
    ids = {id(r) for r in reqs}
    mine = [q for q in sched.waiting_queue if id(q) in ids]
    sched.waiting_queue = d_seats.order_waiting(mine) + [
        q for q in sched.waiting_queue if id(q) not in ids
    ]
    return mine


def park_tick(sched) -> int:
    """Parked requests on an AWAKE D re-join the queue head: at once after a
    sleep whose hold was not armed, or when the park's sleep never came
    (``SGLANG_WEG2_D_PARK_AWAKE_REQUEUE_S``; group-MIN verdict because the
    clock is rank-local -- the weg2xsn296 rule)."""
    parked = getattr(sched, "weg2_d_parked", None)
    if not parked or getattr(sched, "weg2_dormant", False):
        return 0
    due = bool(getattr(sched, "_weg2_d_park_slept", False))
    if not due:
        local = d_seats.awake_requeue_due(
            parked, now=time.monotonic(), bound_s=d_seats.awake_requeue_s()
        )
        due = bool(sched._weg2_group_min_flags([local])[0])
    if not due:
        return 0
    moved = list(parked)
    sched.weg2_d_parked = []
    sched._weg2_d_park_slept = False
    setattr(sched, LATE_HOLD_ATTR, None)  # H91c3-2: the re-queue closes the late hold
    # PARK-SETTLE: a folded settle request goes back to the settle (its read is
    # still short; the queue would hand it to the X gate unsettled).
    back = [r for r in moved if getattr(r, FROM_SETTLE_ATTR, False)]
    moved = [r for r in moved if not getattr(r, FROM_SETTLE_ATTR, False)]
    if back:
        now = time.monotonic()
        settle = getattr(sched, "weg2_post_wake_settle", None)
        if settle is None:
            settle = sched.weg2_post_wake_settle = []
        for req in back:
            setattr(req, FROM_SETTLE_ATTR, False)
            req._1471_since = now
            settle.append(req)
    for req in moved:
        sched._add_request_to_queue(req, is_retracted=True)
    mine = _to_queue_head(sched, moved)
    logger.info("WEG2-D-PARK requeue (awake): %d parked request(s) at the queue head %s%s",
                len(mine), [str(r.rid)[:12] for r in mine],
                (", %d back to the #1471 settle %s" % (len(back), [str(r.rid)[:12] for r in back]))
                if back else "")
    return len(mine) + len(back)


def note_retracted(sched, retracted_reqs) -> int:
    """A decode-pressure retraction on D is a PRESSURE park: the victims were
    retained (``retract_retain``), they move to the queue head and resume when
    no older request is live (``d_seats.admission_gate``)."""
    if not retracted_reqs or not d_seats.d_park_active():
        return 0
    now = time.monotonic()
    for req in retracted_reqs:
        # H91c2: ALWAYS a pressure park. A park site is never cleared, so a
        # request resumed from a flip park still carries SITE_FLIP; keeping it
        # let the admission gate resume it "as soon as it fits" -- the next
        # pass, while the older request still decodes -- and the pressure
        # retracted it again (thrash) instead of parking it until the older
        # one is done. retract_decode only takes RUNNING requests, so no
        # waiting flip park is ever re-marked here.
        d_seats.mark_parked(req, d_seats.SITE_PRESSURE, now=now)
    mine = _to_queue_head(sched, retracted_reqs)
    logger.info("WEG2-D-PARK pressure: %d youngest request(s) parked %s (span retained; "
                "resume when no older request is live)",
                len(mine), [str(r.rid)[:12] for r in mine])
    return len(mine)


def _apply_park_defer(sched) -> int:
    """#244 SEAT-ROTATE: a parked request the front deferred this phase is an
    ordinary waiting request -- its park site is cleared (no barrier, not
    resumed first) and it goes behind the rest of the queue. Its span stays
    retained / held (#243). Applied where it is found (queue or dormant hold);
    each rid once. Replicated input (the wake object), so every rank moves the
    same requests."""
    defer = getattr(sched, "weg2_park_defer", None)
    if not defer:
        return 0
    moved = []
    for r in list(sched.waiting_queue) + list(getattr(sched, "weg2_dormant_hold", None) or []):
        rid = str(getattr(r, "rid", ""))
        if rid in defer and d_seats.park_site(r) is not None:
            d_seats.clear_park(r)
            defer.discard(rid)
            moved.append(r)
    if moved:
        ids = {id(r) for r in moved}
        from sglang.srt.weg2 import seat_age as _sa

        rest = [q for q in sched.waiting_queue if id(q) not in ids]
        mine = [q for q in sched.waiting_queue if id(q) in ids]
        if _sa.enabled():
            # SA: each deferred request goes before the first waiting request
            # YOUNGER than it (the rest keeps the group's order) -- it moves in
            # as soon as the older ones leave, never behind younger arrivals.
            for r in _sa.by_age(mine):
                a = _sa.rid_age(r.rid)
                k = next((i for i, q in enumerate(rest) if d_seats.park_site(q) is None
                          and _sa.rid_age(getattr(q, "rid", "")) > a), len(rest))
                rest.insert(k, r)
            sched.waiting_queue = rest
        else:
            sched.waiting_queue = rest + mine
        logger.info("WEG2-D-PARK seat-rotate: %d parked request(s) deferred this phase %s "
                    "(not resumed first; ordinary waiting work behind the hand-offs)",
                    len(moved), [str(r.rid)[:12] for r in moved])
    return len(moved)


def displace_for_age(sched, running_batch) -> Optional[str]:
    """SA (3) (#244, the user's design): an older request waits on D while
    every seat is held and a YOUNGER one runs -> the youngest running one is
    parked WHOLE (retract, span retained, pressure park: it resumes when it is
    the oldest live request). One per pass. Under speculative decoding only
    the back of the batch may leave; when the youngest is not the back the
    displacement waits (named). Partial parking (only part of a request's KV)
    does not exist -- named follow-up. Replicated inputs (running set, queue,
    rids, seat cap), so every rank decides alike. Returns the displaced rid."""
    from sglang.srt.weg2 import seat_age as _sa

    if not _sa.enabled() or not d_seats.d_flip_park_active():
        return None
    reqs = list(getattr(running_batch, "reqs", None) or [])
    if not reqs:
        return None
    cap = seat_cap(sched)
    if cap is None:
        cap = int(getattr(getattr(sched, "server_args", None), "max_running_requests", 0) or 0)
    # SP (partial park, KV trigger): the rid the adder refused with NO_TOKEN in
    # the LAST pass (Scheduler sets it; consumed here, one pass old).
    no_token_rid = getattr(sched, "_weg2_sa_no_token", None)
    try:
        sched._weg2_sa_no_token = None
    except Exception:  # noqa: BLE001
        pass
    waiting = [q for q in sched.waiting_queue if d_seats.park_site(q) != d_seats.SITE_PRESSURE]
    trigger = "seat"
    pair = None
    if cap and len(reqs) >= int(cap):
        pair = _sa.displace_victim([str(q.rid) for q in waiting], [str(r.rid) for r in reqs], True)
    elif _kv_displace_enabled():
        # KV trigger: a seat is free, but an OLDER waiting request did not fit
        # D's KV while a younger one runs. The PRECONDITION is replicated (the
        # queue, the running set and the rids are); the NO_TOKEN verdict is
        # read through the group MIN (_weg2_group_min_flags), entered by every
        # rank because the precondition is the same on every rank -- so the
        # group displaces only when EVERY rank refused (RAENGE-NIE-UNEINS).
        cand = _sa.displace_victim([str(q.rid) for q in waiting], [str(r.rid) for r in reqs], True)
        if cand is not None:
            local = no_token_rid is not None and _sa.rid_age(no_token_rid) <= _sa.rid_age(cand[0])
            gm = getattr(sched, "_weg2_group_min_flags", None)
            agreed = bool(gm([local])[0]) if callable(gm) else bool(local)
            if agreed:
                pair, trigger = cand, "kv"
    if pair is None:
        return None
    older, victim_rid = pair
    idx = next(i for i, r in enumerate(reqs) if str(r.rid) == victim_rid)
    spec = not (getattr(running_batch, "spec_algorithm", None) is None
                or running_batch.spec_algorithm.is_none())
    if spec and idx != len(reqs) - 1:
        n = getattr(sched, "_sa_displace_waits", 0) + 1
        sched._sa_displace_waits = n
        if n <= 8 or (n & (n - 1)) == 0:
            logger.info("SEAT-AGE DISPLACE-WAIT older=%s youngest=%s (n=%d): under speculative "
                        "decoding only the back of the batch may leave; the youngest is not the back",
                        older[:16], victim_rid[:16], n)
        return None
    snap = sched._weg2_d_park_draft_snapshot(running_batch) if hasattr(
        sched, "_weg2_d_park_draft_snapshot") else None
    victim = reqs[idx]
    running_batch.release_req(idx, len(reqs) - 1, sched.server_args, retain=True)
    running_batch.filter_batch(keep_indices=[i for i in range(len(reqs)) if i != idx])
    if snap is not None and hasattr(sched, "_weg2_d_park_draft_save"):
        sched._weg2_d_park_draft_save([victim], snap)
    sched._add_request_to_queue(victim, is_retracted=True)
    d_seats.mark_parked(victim, d_seats.SITE_PRESSURE, now=time.monotonic())
    sched.waiting_queue = [q for q in sched.waiting_queue if q is not victim] + [victim]
    sched._weg2_sa_displaced = getattr(sched, "_weg2_sa_displaced", 0) + 1
    # H106b (rc12z22-dwell30 D 15:51:41, weg2-6-33): this runs AFTER the pass's
    # #580 prefetch drain, so the victim joins a queue the drain never saw.
    # The scheduler excludes it from THIS pass's admission by name
    # (Scheduler._weg2_sa_exclude_displaced); it resumes from the next pass.
    sched._weg2_sa_displaced_now = str(victim_rid)
    older_req = next((q for q in waiting if str(q.rid) == older), None)
    pages_out = _partial_keep(sched, victim, older_req if trigger == "kv" else None)
    logger.warning("SEAT-AGE DISPLACE rid_out=%s older_waiting=%s trigger=%s running=%d cap=%s "
                   "pages_out=%s: the youngest running request pauses (span retained: its KV leaves "
                   "the device only as far as the older one needs it -- LRU eviction, tail last)",
                   victim_rid[:16], older[:16], trigger, len(reqs), cap, pages_out)
    return victim_rid


def exclude_displaced(sched, prefetch_verdicts) -> Optional[str]:
    """H106b (rc12z22-dwell30 D 15:51:41, all three ranks): the SA displacement
    of this pass requeues its victim AFTER the #580 prefetch drain, so the
    admission loop met weg2-6-33 without a drained verdict and every rank
    raised ('the queue was mutated in between'). The displacement is
    replicated, so its victim is excluded from THIS pass by name on every
    rank alike -- a not-done verdict, the ordinary skip -- and resumes from
    the next pass's drain. A victim the drain did cover keeps its verdict.
    Returns the excluded rid."""
    rid = getattr(sched, "_weg2_sa_displaced_now", None)
    sched._weg2_sa_displaced_now = None
    if rid is None or not isinstance(prefetch_verdicts, dict) or rid in prefetch_verdicts:
        return None
    prefetch_verdicts[rid] = False
    logger.info("SEAT-AGE DISPLACED-THIS-PASS rid=%s: requeued after the pass's prefetch "
                "drain; excluded from this pass's admission (not-done verdict), it "
                "resumes from the next pass", str(rid)[:16])
    return rid


def _kv_displace_enabled(env=None) -> bool:
    """SP: ``SGLANG_WEG2_SEAT_AGE_KV_DISPLACE`` (default on; 0 = seat trigger only)."""
    e = os.environ if env is None else env
    raw = (e.get("SGLANG_WEG2_SEAT_AGE_KV_DISPLACE", "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _partial_keep(sched, victim, older_req) -> str:
    """SP: the ONE call site towards NF's #248 keep role
    (``keep_role(rid, "park", page_range=(a, b), page_size=page)``, guarded
    import). The window [a, b) is the victim's hindmost pages the older request lacks: a = span end
    - ceil(shortfall / page_size), shortfall = the older's uncached need minus
    the free rows (0 on a seat trigger: a pure pause, empty window). Module
    missing / raising: today's retain (LRU eviction) -- the displacement stands.
    Returns what is logged as pages_out."""
    def _n(x) -> int:
        # never ``x or ()``: on D ``prefix_indices`` is a torch tensor and its
        # truth value raises (review 28.09., the 7e227e5f76 class) -- which the
        # except below turned into a silent "?" and no keep_role call
        return 0 if x is None else len(x)

    try:
        span = _n(getattr(victim, "origin_input_ids", None)) + _n(
            getattr(victim, "output_ids", None))
        tree = getattr(sched, "tree_cache", None)
        page = int(getattr(tree, "page_size", 1) or 1)
        shortfall = 0
        if older_req is not None:
            need = _n(getattr(older_req, "origin_input_ids", None)) + _n(
                getattr(older_req, "output_ids", None)) - _n(
                getattr(older_req, "prefix_indices", None))
            alloc = getattr(sched, "token_to_kv_pool_allocator", None)
            free = int(alloc.available_size()) if alloc is not None else 0
            shortfall = max(0, int(need) - free)
        n_pages = -(-shortfall // page) if shortfall else 0
        b = -(-span // page)
        a = max(0, b - n_pages)
    except Exception:  # noqa: BLE001
        return "?"
    if n_pages == 0:
        return "0"
    try:
        from sglang.srt.weg2 import handoff_pending as _hp  # NF #248 (name pending)

        keep_role = getattr(_hp, "keep_role", None)
        if not callable(keep_role):
            return f"window={a}-{b}(retain)"
        # page_range is in pages of the tree's page_size; the unit goes along
        # (NF #248 records it, 0 would read as "unknown").
        got = keep_role(str(victim.rid), "park", page_range=(a, b), page_size=page)
        return str(got) if isinstance(got, int) else f"window={a}-{b}"
    except Exception:  # noqa: BLE001 -- no module / raising: today's retain
        return f"window={a}-{b}(retain)"


def admission(sched, running_batch):
    """One pass's D admission verdict: parked first, the rest in the group's
    order, the barrier/blocked set of ``d_seats.admission_gate``. None when
    the park is off or nothing is parked -- the stock loop, untouched.
    27B immediate park (d_flip_park_active without d_park_active): only flip
    parks exist there, so with none waiting this is the stock loop too."""
    if not d_seats.d_flip_park_active():
        return None
    # SA (3) (#244): runs on both forms (27B immediate park / NF), before the
    # nothing-parked early return -- the older waiting one may be a deferred
    # parked request whose park site is already cleared.
    _apply_park_defer(sched)
    displace_for_age(sched, running_batch)
    if not d_seats.d_park_active() and not any(
        d_seats.park_site(r) is not None
        for r in list(sched.waiting_queue)
        + list(getattr(sched, "weg2_dormant_hold", None) or [])
    ):
        return None  # immediate park, nothing flip-parked waits (settle excluded, PK2): stock loop
    _apply_park_defer(sched)
    sched.waiting_queue = d_seats.order_waiting(sched.waiting_queue)
    book = getattr(sched, "_weg2_d_resume_book", None)
    if book is None:
        book = sched._weg2_d_resume_book = d_seats.ResumeBook.from_env()
    # PK2 (metal dkr27bparkdraftbar1w209270645, park probe, 27.09.): under the
    # 27B immediate park alone a flip-parked request still READING after the
    # wake (#1471 post-wake settle) does not hold newcomers back. B (20477
    # tokens, fully loaded at the wake) sat 20.1 s behind A in the settle
    # (A's read was short, the settle bound lapsed) -> B-TTFT 26.8-27.1 s. The
    # barrier exists so no newcomer takes a seat a parked request is coming
    # back to; D has --d-bs seats and the settle-held one is not coming back
    # this pass. NF (d_park_active, standard form) keeps the settle in the
    # barrier unchanged.
    settle = list(getattr(sched, "weg2_post_wake_settle", None) or [])
    if not d_seats.d_park_active():
        settle = []
    pending = settle + list(getattr(sched, "weg2_dormant_hold", None) or [])
    avail = sched.uniform_min_avail() if book.margin_tokens >= 0 else None
    gate = d_seats.admission_gate(
        sched.waiting_queue,
        running=list(running_batch.reqs),
        pending_outside=pending,
        avail_tokens=avail,
        resume_book=book,
    )
    return gate if gate.barrier else None


def park_abort(sched, recv_req) -> int:
    """An abort reaches the parked list exactly as it reaches the dormant hold
    (#1445): a parked request owns no device rows (retracted), so dropping it
    and telling the tokenizer is the whole release."""
    from sglang.srt.managers.io_struct import AbortReq

    parked = getattr(sched, "weg2_d_parked", None)
    if not parked:
        return 0
    rid = str(getattr(recv_req, "rid", "") or "")
    abort_all = bool(getattr(recv_req, "abort_all", False))
    gone = [r for r in parked if abort_all or str(r.rid).startswith(rid)]
    if not gone:
        return 0
    ids = {id(r) for r in gone}
    sched.weg2_d_parked = [r for r in parked if id(r) not in ids]
    d_park_draft.drop_all(sched, gone, "abort")  # H91d
    for req in gone:
        if getattr(sched, "enable_hicache_storage", False):
            sched.tree_cache.release_aborted_request(req.rid)
        sched.ipc_channels.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
    logger.info("WEG2-D-PARK abort: %d parked request(s) dropped (rid=%s abort_all=%s)",
                len(gone), rid[:12], abort_all)
    return len(gone)


def note_wake_seats(sched, recv_req):
    """H95 (H91 Teil B, Stufe 2): the kv_cache resume that wakes D carries the
    front's ``handoff_n``/``parked_n``; the phase's seat count n is
    ``d_seats.phase_seats`` of those two integers and of the boot's
    --max-running-requests (the --d-bs cap) -- the same request object on
    every rank, so every rank holds the same n without a collective.

    Kept on ``sched.weg2_d_phase_seats`` for the phase (the per-seat posts
    that a later per-flip re-partition would size by it) and named once per
    wake (``WEG2 D-PHASE-SEATS (H95)``). None and untouched state off group
    D or on a wake without the count."""
    if str(os.environ.get(d_seats.GROUP_ENV, "")).strip().upper() != "D":
        return None
    handoff_n = getattr(recv_req, "handoff_n", None)
    parked_n = getattr(recv_req, "parked_n", None)
    # #244 SEAT-ROTATE: the parked rids the front defers this phase (replicated:
    # the same wake object on every rank).
    _defer = getattr(recv_req, "park_defer_rids", None)
    sched.weg2_park_defer = set(str(r) for r in _defer) if _defer else set()
    cap = int(getattr(getattr(sched, "server_args", None), "max_running_requests", 0) or 1)
    seats = d_seats.phase_seats(handoff_n, parked_n, cap=cap,
                                epoch=getattr(recv_req, "epoch", None))
    if seats is None:
        return None
    sched.weg2_d_phase_seats = seats
    logger.info("%s", seats.line())
    return seats


def seat_vram_wake(sched, recv_req, seats):
    """H95c: every D resume request, BEFORE the saver resumes its tags --
    the posts of the phase's ``seats`` (``d_seats.PhaseSeats``, None when the
    request carries no count) become pages: the slot limit (replicated), the
    Mamba span plan and the expert seat rows (TP0). A no-op unless
    SGLANG_OPT_WEG2_D_SEAT_VRAM on group D (weg2/d_seat_vram.py)."""
    from sglang.srt.weg2 import d_seat_vram

    return d_seat_vram.on_wake(sched, recv_req, seats)


def seat_cap(sched):
    """H95c: the phase's n as a running-request cap, None = no cap."""
    from sglang.srt.weg2 import d_seat_vram

    return d_seat_vram.admission_cap(sched)


def seat_guard(sched, batch) -> None:
    """H95c W-SEAT: a batch wider than the phase's n never runs."""
    from sglang.srt.weg2 import d_seat_vram

    d_seat_vram.guard(sched, batch)
