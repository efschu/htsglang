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
from sglang.srt.weg2 import d_park_draft, d_park_read, d_seats, park_hold_yield, park_retract_split
from sglang.srt.weg2 import handback_claim as _hb

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
    if str(getattr(recv_req, "youngest", "") or ""):
        return park_youngest(sched, recv_req)
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
    # FLIPCYCLE H3 (02.10.): the park's own phases on the scheduler thread
    # (y6z dispatch_ms 142-591, ~110 ms per parked request; floor ~30 ms: the
    # END-state D2H of ~65 MB per request). One line per park, TP0 reads it.
    _pt = [time.perf_counter()]
    _pk: list = []

    def _ph(name):
        now = time.perf_counter()
        _pk.append((name, (now - _pt[0]) * 1000.0))
        _pt[0] = now

    _t_park0 = _pt[0]
    if sched.enable_overlap and sched.last_batch and sched.result_queue:
        tmp_batch, tmp_result = sched.result_queue.popleft()
        sched.process_batch_result(tmp_batch, tmp_result)
    _ph("result")
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
    _ph("filter")
    d_park_draft.save_parked(sched, running, site=d_seats.SITE_FLIP)
    _ph("draft")
    # 27B PARK: a DFlash window pool carries nothing; the resume is a fresh
    # hand-off's admission (rearm_window_draft_cold). MTP pools: no-op.
    rearmed = rearm_window_draft_cold(sched, running)
    # F4 (#259 4c): the END state of each running request as a park tail part,
    # gathered before the retraction hands its slots to the tree.
    _park_end(sched, running)
    _ph("end")
    park_hold_yield.begin(getattr(sched, "tree_cache", None))
    # PARK-RETRACT-SPLIT: the retract phase split per request (release /
    # write-through backup / controller write / rest), one line per park
    retracted = (
        park_retract_split.run_split(
            getattr(sched, "tree_cache", None),
            lambda: sched.running_batch.retract_all(sched.server_args, offload_kv=False, retain=True),
            epoch,
        )
        if running else []
    )
    _ph("retract")
    sched.running_batch.batch_is_full = False
    sched.chunked_req = None
    now = time.monotonic()
    for req in retracted:
        d_seats.mark_parked(req, d_seats.SITE_FLIP, epoch=epoch, now=now)
        sched._969ad_note_retract(req, "weg2_park_running")
        _hb.note_origin(req.rid, _hb.ORIGIN_PARK)  # ZR: the resume computes 0 tokens again
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
            logger.info("WEG2-D-PARK RETAINED rid=%s %s", str(req.rid), d_park_read.describe(req))
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
    _ph("mark")
    # HY: a retained span whose backup the full arena refused takes the
    # L3-copied pages of a held (not running) request -- one group vote, then
    # the give-back and the backup again on every rank, or nothing moves
    park_hold_yield.settle(sched, retracted=retracted, parked=sched.weg2_d_parked)
    _ph("yield")
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
    _ph("l3mark")
    resumable = weg2_resumable_depth.park_depths(
        getattr(sched, "tree_cache", None),
        [r for r in sched.weg2_d_parked if d_seats.park_site(r) is not None],
        getattr(sched, "ps", None),
    )
    _ph("depth")
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
                            str(req.rid), before, d_park_read.describe(req))
    _ph("clamp")
    logger.info("WEG2-FLIPCYCLE stage=park epoch=%d n=%d ms=%.0f floor_ms=%d sub=%s (H3: the park's "
                "phases on the scheduler thread; floor = the END-state D2H)", epoch, len(retracted),
                (time.perf_counter() - _t_park0) * 1000.0, 30 * max(1, len(retracted)) // 3 or 10,
                ",".join("%s:%.0f" % kv for kv in _pk))
    logger.info(
        "WEG2-D-PARK park_running epoch=%d reason=%s: %d running retracted (span retained, "
        "forced host write-through), parked=%s queued-behind=%s settle-folded=%s late_hold=%s -- "
        "the sleep holds them first, the wake resumes oldest first%s",
        epoch, reason, len(retracted), list(rids), list(held),
        [str(r.rid) for r in settle], late_hold,
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
                "parked", str(req.rid), len(parked))
    return True


def _park_end(sched, running, *, reduce_min=None) -> int:
    """F4 (#259 4c, SGLANG_WEG2_ENABLE_D_PARK_END): every rank writes its
    part of each running request's END state (weg2/tail_handoff
    ``publish_park_end``) -- the resume adopts it as E2's skip instead of
    extending the tail from the last mamba anchor. The gathers are waited
    for here, MEASURED (``sync_ms`` in the line): the retraction below frees
    the rows. Returns the number of parts written on this rank."""
    from sglang.srt.weg2 import tail_handoff as th

    if not th.park_end_enabled():
        return 0
    ps = getattr(sched, "ps", None)
    tp_rank = int(getattr(ps, "tp_rank", getattr(sched, "tp_rank", 0)) or 0)
    tp_size = int(getattr(ps, "tp_size", getattr(sched, "tp_size", 1)) or 1)
    # leak (29.09.): the parts of rids D no longer holds leave before new ones come
    n_reaped, reaped = th.reap_orphan_parks(_live_rids(sched, running), f"{th.PARK_PART}{tp_rank}")
    if n_reaped:
        logger.info("F4 PARK-REAP files=%d rids=%s (parts of rids D no longer holds)",
                    n_reaped, list(reaped))
    if not running:
        return 0
    part = f"{th.PARK_PART}{tp_rank}-{os.getpid()}"
    # the resume re-enters at the last mamba track point (#59b resumable
    # depth), at most one track interval below the end: two cover the lag of
    # a spec-decode track
    interval = int(getattr(sched.server_args, "mamba_track_interval", 0) or 0) or int(sched.page_size)
    # PARK-ANCHOR (0929): a D-direct prefill's track point sits at its
    # extend's START, a whole extend below the end -- the window reaches down
    # to the anchor the retraction below leaves (group-uniform), capped at X
    t_anchor = time.perf_counter()
    anchors, mode = _park_anchors(sched, running, reduce_min=reduce_min)
    anchor_ms = (time.perf_counter() - t_anchor) * 1000.0
    max_rows = int(getattr(sched.server_args, "tp_prefill_max_tokens", 0) or 0)
    events, why = [], {}
    for req, anchor in zip(running, anchors):
        refusal, ev = th.publish_park_end(
            req, sched.req_to_token_pool, sched.token_to_kv_pool_allocator, sched.page_size,
            part, tp_size, 2 * interval, anchor=anchor, max_rows=max_rows,
        )
        if refusal:
            why[str(req.rid)] = refusal
        else:
            events.append(ev)
    sync_ms = th.park_end_barrier(events) if events else 0.0
    logger.info("F4 PARK-END park: parts=%d of %d running sync_ms=%.1f refused=%s part=%s "
                "anchor_mode=%s anchor_ms=%.1f",
                len(events), len(running), sync_ms, why or "-", part, mode, anchor_ms)
    return len(events)


#: PARK-ANCHOR: a rank that names no anchor (a Form A worker, a request
#: without a tracked position or prefix) votes this -- the MIN takes the
#: deciding rank's value, and a reduce that stays here means "no anchor".
_NO_ANCHOR = 1 << 62


def _local_anchor(req, tree_cache=None) -> int:
    """This rank's view of the depth ``req``'s retaining retraction leaves
    as its resume anchor: the tracked position the mamba retention inserts
    at (``mamba_last_track_seqlen``, the #1469 RETAIN ``cache_len`` -- y3p
    weg2-4-8 2368, weg2-8-12 16704).

    H' (30.09.): without a pending track point the anchor is the one the TREE
    already holds on the request's path -- the depth #59b names a few lines
    later and the wake resumes from (``weg2_resumable_depth.local_depth``, the
    side-effect-free admission probe). NOT ``len(prefix_indices)``: after the
    D-direct extend's own retain (``cache_unfinished_req``) the track is
    cleared and ``prefix_indices`` covers the whole extended KV, tombstoned
    above the anchor. y3y weg2-14-34: RETAIN cache_len=2368, F4 anchor=4446
    (window=default [3904, 4444)), #59b 2368, 'adopt=skipped:prefix:2368' and
    D computed 2078 tokens again behind the wake; y3w weg2-2-7 (4448 vs
    2368) and weg2-6-11 (18782 vs 16704) the same. Only without a tree (desk
    stubs) the matched prefix stays the fallback."""
    t = req.mamba_last_track_seqlen
    if t is not None and int(t) >= 0:
        return int(t)
    if tree_cache is not None:
        # an unpriceable probe votes 0: no anchor, today's window
        depth = int(weg2_resumable_depth.local_depth(tree_cache, req))
        return depth if depth > 0 else _NO_ANCHOR
    if req.prefix_indices is not None:
        return len(req.prefix_indices)
    return _NO_ANCHOR


def _park_anchors(sched, running, *, reduce_min=None):
    """PARK-ANCHOR (0929): per running request the resume anchor the F4
    window must reach, the SAME list on every rank (the park parts of one rid
    are OR-ed together and must name one geometry, ``end_differs``).

    Only the rank that decides the recurrent anchor knows it: a Form A expert
    worker tracks no mamba state (y3p TP1/TP2 ``#1469 RETAIN cache_len=4416``
    where the host retained 2368) and votes ``_NO_ANCHOR``; the group takes
    the MIN over the TP cpu group -- the collective #59b runs a few lines
    later in the same park, entered by every rank with the same list (the
    running batch is replicated). A group that cannot make a depth uniform
    (``weg2_resumable_depth.MODE_NONE``: PP, DP attention) names none and
    keeps today's window. Returns (anchors, mode)."""
    if not running:
        return [], "-"
    mode = weg2_resumable_depth.group_mode(getattr(sched, "ps", None))
    if mode == weg2_resumable_depth.MODE_NONE:
        return [None] * len(running), mode
    reduce = mode != weg2_resumable_depth.MODE_SOLO
    if reduce and reduce_min is None:
        import torch.distributed as dist

        if not dist.is_initialized():
            # process-global, so group-uniform: no group to agree with
            return [None] * len(running), "no_dist"
    from sglang.srt.managers import tp_match_floor

    follows = tp_match_floor.this_rank_follows()
    tree = getattr(sched, "tree_cache", None)
    local = [_NO_ANCHOR if follows else _local_anchor(r, tree) for r in running]
    if reduce:
        local = (reduce_min or weg2_resumable_depth._tp_min)(local)
    return [None if int(v) >= _NO_ANCHOR else int(v) for v in local], mode


def _live_rids(sched, running) -> set:
    """Every rid D still holds: running (about to park), parked, queued, in
    the #1471 settle or the #1443 dormant hold, the chunked one."""
    live = set()
    for group in (running, getattr(sched, "weg2_d_parked", None),
                  getattr(sched, "waiting_queue", None),
                  getattr(sched, "weg2_post_wake_settle", None),
                  getattr(sched, "weg2_dormant_hold", None),
                  [getattr(sched, "chunked_req", None)]):
        for r in group or ():
            if r is not None:
                live.add(str(getattr(r, "rid", "")))
    return live


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
    if not hold_armed:
        # the list stays parked over the sleep: the first awake pass re-queues it
        sched._weg2_d_park_slept = True
        logger.info("WEG2-D-PARK hold: SGLANG_WEG2_DORMANT_ADMIT off -- %d parked request(s) "
                    "wait for the wake", len(parked))
        return 0
    # DRAIN-W50 (z30y12 epoch 51, 19.6 s drain): the hold takes the WHOLE list,
    # nothing stays parked for park_tick to re-queue -- so no "slept" mark. Set
    # here before, it outlived the list (park_tick returns early on an empty
    # one and never cleared it): the phase's FIRST W50 midstream hold
    # (resume_via_p.keep_on_d, rid weg2-35-112 23:35:46) was re-queued in the
    # same pass, re-admitted on D and decoded to its end (19.2 s) while the
    # front, told it was held, sat in quiesce. Metal: 6 of 6 first-after-wake
    # W50 holds of that boot were re-queued at once, the second one of a phase
    # (weg2-62-182) held.
    sched._weg2_d_park_slept = False
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
                len(moved), [str(r.rid) for r in moved])
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
        return _capacity_requeue(sched, parked)
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
                len(mine), [str(r.rid) for r in mine],
                (", %d back to the #1471 settle %s" % (len(back), [str(r.rid) for r in back]))
                if back else "")
    return len(mine) + len(back)


def _capacity_requeue(sched, parked) -> int:
    """#248h: the capacity-parked requests (``resume_via_p.park_for_capacity``)
    re-join the queue head as soon as the arena holds their re-read -- the
    other parked requests keep waiting for their P leg / the awake bound.
    One group verdict per parked request (MIN, rank-local clock)."""
    from sglang.srt.weg2 import resume_via_p as _rvp

    if not any(getattr(r, _rvp.CAPPARK_AT_ATTR, None) is not None for r in parked):
        return 0
    due_ids = {id(r) for r in _rvp.capacity_requeue_due(sched, parked)}
    flags = sched._weg2_group_min_flags([id(r) in due_ids for r in parked])
    moved = [r for r, f in zip(list(parked), flags) if f]
    if not moved:
        return 0
    ids = {id(r) for r in moved}
    sched.weg2_d_parked = [r for r in parked if id(r) not in ids]
    for req in moved:
        setattr(req, _rvp.CAPPARK_AT_ATTR, None)  # a re-park stamps it again
        sched._add_request_to_queue(req, is_retracted=True)
    mine = _to_queue_head(sched, moved)
    logger.info("#248h WEG2-D-PARK requeue (capacity): %d parked request(s) at the queue head %s -- "
                "the arena holds their re-read now", len(mine), [str(r.rid) for r in mine])
    return len(mine)


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
                len(mine), [str(r.rid) for r in mine])
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
                    len(moved), [str(r.rid) for r in moved])
    return len(moved)


#: Q-698: at most this many SEAT-AGE displacements for ONE older waiting
#: request. A request needs at most one victim per younger seat (--d-bs 6 on
#: NF); the bound only ends a rotation the hold below cannot see (the
#: verdict's KV fit is not the adder's NO_TOKEN budget: decode reservations,
#: the group floor, the commitment ledger, mamba slots).
DISPLACE_MAX_PER_OLDER = 8


def _displace_counts(sched) -> dict:
    counts = getattr(sched, "_weg2_sa_displace_counts", None)
    if counts is None:
        counts = sched._weg2_sa_displace_counts = {}
    return counts


def _waiting_rids(sched) -> set:
    return {str(getattr(q, "rid", "")) for q in list(getattr(sched, "waiting_queue", None) or [])
            + list(getattr(sched, "weg2_post_wake_settle", None) or [])
            + list(getattr(sched, "weg2_dormant_hold", None) or [])}


def _prune_displace_counts(sched) -> None:
    """An older request that is admitted or gone starts a new count when it
    waits again (replicated: the queue is)."""
    counts = getattr(sched, "_weg2_sa_displace_counts", None)
    if not counts:
        return
    live = _waiting_rids(sched)
    for rid in [r for r in counts if r not in live]:
        del counts[rid]
    told = getattr(sched, "_weg2_sa_exhausted_told", None)
    if told:
        sched._weg2_sa_exhausted_told = {r for r in told if r in live}


def _release_holds(sched, older: Optional[str] = None) -> list:
    """Q-698: end the hold of every victim parked for ``older`` (None = for
    any older request). Returns the released victims' rids."""
    out = []
    for q in list(getattr(sched, "waiting_queue", None) or []):
        held_for = getattr(q, d_seats.DISPLACED_FOR_ATTR, None)
        if held_for is not None and (older is None or str(held_for) == str(older)):
            setattr(q, d_seats.DISPLACED_FOR_ATTR, None)
            out.append(str(q.rid))
    return out


def _lift_holds_when_idle(sched) -> None:
    """Q-698: nothing runs on D, victims are held for an older request, and
    that older one was refused NO_TOKEN in the last pass -- on every rank
    (group MIN; the precondition is replicated, so every rank enters it). No
    running request will release KV for it, so holding its victims would only
    idle D: the holds end, named. The older one keeps waiting as any request
    does (the victims' own end releases their KV)."""
    held = {str(getattr(q, d_seats.DISPLACED_FOR_ATTR, None))
            for q in list(getattr(sched, "waiting_queue", None) or [])
            if getattr(q, d_seats.DISPLACED_FOR_ATTR, None) is not None}
    if not held:
        return
    no_token_rid = getattr(sched, "_weg2_sa_no_token", None)
    sched._weg2_sa_no_token = None
    if not getattr(sched, "_weg2_sa_idle_armed", False):
        # the refusal read here may stem from a pass in which something still
        # ran: only a refusal from an idle pass counts (replicated: the
        # running set is)
        sched._weg2_sa_idle_armed = True
        return
    local = no_token_rid is not None and str(no_token_rid) in held
    gm = getattr(sched, "_weg2_group_min_flags", None)
    agreed = bool(gm([local])[0]) if callable(gm) else bool(local)
    if not agreed:
        return
    released = _release_holds(sched)
    logger.warning("Q-698 SEAT-AGE HOLD-LIFT older=%s released=%s: refused NO_TOKEN with nothing "
                   "running on D -- no running request releases KV for it; its victims resume, "
                   "it waits for their KV", sorted(held),
                   [r for r in released])


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
    _prune_displace_counts(sched)
    if not reqs:
        _lift_holds_when_idle(sched)
        return None
    sched._weg2_sa_idle_armed = False
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
        cand = _sa.displace_victim([str(q.rid) for q in waiting], [str(r.rid) for r in reqs], True)
        if cand is not None:
            # NF review of 2d49cd45bf: the seat trigger freed a seat without asking
            # whether the older one then FITS the KV -- a victim for nothing, against the
            # user's "nur wenn es reicht". Seat AND KV after the k youngest, else nobody.
            # The KV half is rank-local, so it goes through the group MIN like the KV
            # trigger (the precondition -- full seats, a candidate -- is replicated,
            # so every rank enters the collective).
            local = kv_displace_would_fit(sched, cand[0], reqs, seat=True)
            gm = getattr(sched, "_weg2_group_min_flags", None)
            agreed = bool(gm([local])[0]) if callable(gm) else bool(local)
            if agreed:
                pair = cand
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
            if local:
                # user 30.09. (verbatim): "natürlich muss ein jüngerer nur verdrängt werden,
                # wenn der ältere nicht draufpasst. nicht pauschal den jüngeren verdrängen" --
                # only when displacing the youngest seats is ENOUGH for the older one (and it
                # does not already fit); else nobody leaves and the backfill stays.
                local = kv_displace_would_fit(sched, cand[0], reqs)
            gm = getattr(sched, "_weg2_group_min_flags", None)
            agreed = bool(gm([local])[0]) if callable(gm) else bool(local)
            if agreed:
                pair, trigger = cand, "kv"
    if pair is None:
        return None
    older, victim_rid = pair
    counts = _displace_counts(sched)
    if counts.get(older, 0) >= DISPLACE_MAX_PER_OLDER:
        # Q-698 end state: the displacements did not get the older one in (the
        # adder's NO_TOKEN budget is not the verdict's fit) -- no further victim
        # for it, the held ones resume; it waits for the KV the running
        # requests release, as any request does. Named once per older rid.
        released = _release_holds(sched, older)
        if older not in getattr(sched, "_weg2_sa_exhausted_told", set()):
            sched._weg2_sa_exhausted_told = getattr(sched, "_weg2_sa_exhausted_told", set()) | {older}
            logger.warning("Q-698 SEAT-AGE DISPLACE-EXHAUSTED older=%s displaced=%d max=%d "
                           "released=%s: displacing younger seats did not get the older request "
                           "admitted -- no further victim for it; the held victims resume and it "
                           "waits for the KV the running requests release",
                           older, counts.get(older, 0), DISPLACE_MAX_PER_OLDER,
                           [str(r) for r in released])
        return None
    idx = next(i for i, r in enumerate(reqs) if str(r.rid) == victim_rid)
    spec = not (getattr(running_batch, "spec_algorithm", None) is None
                or running_batch.spec_algorithm.is_none())
    if spec and idx != len(reqs) - 1:
        n = getattr(sched, "_sa_displace_waits", 0) + 1
        sched._sa_displace_waits = n
        if n <= 8 or (n & (n - 1)) == 0:
            logger.info("SEAT-AGE DISPLACE-WAIT older=%s youngest=%s (n=%d): under speculative "
                        "decoding only the back of the batch may leave; the youngest is not the back",
                        older, victim_rid, n)
        return None
    victim = _displace_at(sched, running_batch, reqs, idx)
    # Q-698: the victim waits for the older one it was parked for
    # (d_seats.admission_gate holds it while that one waits and anything runs).
    setattr(victim, d_seats.DISPLACED_FOR_ATTR, older)
    counts[older] = counts.get(older, 0) + 1
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
                   victim_rid, older, trigger, len(reqs), cap, pages_out)
    return victim_rid



def _displace_at(sched, running_batch, reqs, idx):
    """Park ``reqs[idx]`` WHOLE at this round boundary: retract it retaining its
    span (KV, the node's GDN anchor, the draft rows), re-queue it as a pressure
    park at the back (it resumes when it is the oldest live request and a seat
    is free). The one retraction shape of SA (3) and ARRIVAL-SEAT (c)."""
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
    return victim


def park_youngest(sched, recv_req):
    """ARRIVAL-SEAT (c) (weg2/arrival_seat_rule.py; user rule #246 "Ältester rückt
    nach und verdrängt Jüngere"): the front's oldest waiter passed the wait bound,
    so ONE running request -- the front's youngest, named in ``youngest`` -- is
    parked at this round boundary in SA (3)'s pressure shape; the seat it frees
    is the arrival rule's to fill (D prefill or the flip). Replicated inputs
    (the running set, the named rid), so every rank parks alike. Under
    speculative decoding only the back of the batch may leave: otherwise it is
    refused by name and the front asks again on its next tick. Nothing else is
    touched -- no sleep follows, the batch keeps decoding."""
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqOutput

    epoch = int(getattr(recv_req, "epoch", 0) or 0)
    rid = str(getattr(recv_req, "youngest", "") or "")

    def _out(ok, parked, msg):
        return Weg2ParkRunningReqOutput(success=ok, parked=list(parked), epoch=epoch, message=msg)

    if not d_seats.d_flip_park_active():
        return _out(False, [], "W-PARK refused: not group D -- nothing parked")
    if getattr(sched, "weg2_dormant", False):
        return _out(False, [], "group D is dormant: nothing runs, nothing parked")
    # the in-flight result lands first (park_running's shape)
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
    running_batch = sched.running_batch
    reqs = list(getattr(running_batch, "reqs", None) or [])
    idx = next((i for i, r in enumerate(reqs) if str(r.rid) == rid), None)
    if idx is None:
        return _out(True, [], f"{rid} is not running here (finished or not admitted): nothing parked")
    spec = not (getattr(running_batch, "spec_algorithm", None) is None
                or running_batch.spec_algorithm.is_none())
    if spec and idx != len(reqs) - 1:
        return _out(False, [], f"{rid} is not the back of the batch under speculative decoding: "
                               "refused, the front asks again")
    _displace_at(sched, running_batch, reqs, idx)
    n = getattr(sched, "_weg2_asr_parked", 0) + 1
    sched._weg2_asr_parked = n
    logger.warning("WEG2 ARRIVAL-SEAT YOUNGEST-PARK rid=%s running_before=%d n=%d: the youngest running "
                   "decode pauses at this round boundary (span retained, pressure park); its seat goes "
                   "to the oldest waiter (user rule #246)", rid, len(reqs), n)
    return _out(True, [rid], "parked (arrival-seat youngest)")


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
                "resumes from the next pass", str(rid))
    return rid


def victims_needed(need: int, available: int, young_first_sizes) -> Optional[int]:
    """How many of the youngest running seats (sizes given youngest first) must
    leave so that ``need`` tokens fit: 0 = it fits the free KV already; k = the
    k youngest are enough; None = not even all of them are (older seats hold the
    room -- nobody is displaced, the backfill stays)."""
    need, have = int(need), int(available)
    if need <= have:
        return 0
    for k, size in enumerate(young_first_sizes, start=1):
        have += max(0, int(size))
        if need <= have:
            return k
    return None


def _n(x) -> int:
    """len() of a list OR a torch tensor; never ``x or ()`` -- a tensor has no
    truth value (metal z30y6 16:47:04: an empty ``prefix_indices`` tensor ->
    'Boolean value of Tensor with no values is ambiguous' on every D rank, W17)."""
    return 0 if x is None else len(x)


def _req_kv_tokens(r) -> int:
    return _n(getattr(r, "origin_input_ids", None)) + _n(getattr(r, "output_ids", None))


def kv_displace_would_fit(sched, older_rid: str, running, seat: bool = False) -> bool:
    """This rank's half of the displacement verdict (the group takes the MIN):
    KV trigger (seat=False): the older waiting request does not fit the free KV,
    and parking the youngest running seats younger than it frees enough for it.
    Seat trigger (seat=True): every seat is held, so one victim is needed for the
    seat anyway; it is taken only when the older one then also fits the KV
    (k youngest, k >= 1). Logs the outcome (throttled)."""
    from sglang.srt.weg2 import seat_age as _sa

    older = next((q for q in getattr(sched, "waiting_queue", ()) or () if str(q.rid) == str(older_rid)), None)
    if older is None:
        return False
    need = max(0, _req_kv_tokens(older) - _n(getattr(older, "prefix_indices", None)))
    try:
        avail = int(sched.token_to_kv_pool_allocator.available_size())
    except Exception:  # noqa: BLE001 -- no reading: no displacement
        return False
    try:
        avail += int(sched.tree_cache.evictable_size() or 0)
    except Exception:  # noqa: BLE001
        pass
    young = sorted((r for r in running if _sa.rid_age(str(r.rid)) > _sa.rid_age(str(older_rid))),
                   key=lambda r: _sa.rid_age(str(r.rid)), reverse=True)
    k = victims_needed(need, avail, [_req_kv_tokens(r) for r in young])
    if seat and k is not None:
        if not young:
            k = None                                   # no younger seat to give
        else:
            k = max(1, k)                              # the seat itself needs one victim
    n = getattr(sched, "_sa_kv_fit_n", 0) + 1
    sched._sa_kv_fit_n = n
    if k != 1 and (n <= 8 or (n & (n - 1)) == 0):
        logger.info("SEAT-AGE %s-DISPLACE-VERDICT older=%s need=%d free=%d younger_running=%d -> %s (n=%d)",
                    "SEAT" if seat else "KV", str(older_rid), need, avail, len(young),
                    "fits free, nobody leaves" if k == 0 else
                    "not even with all younger seats: nobody leaves, backfill stays" if k is None else
                    "%d youngest must leave (one per pass)" % k, n)
    return bool(k)


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
        decode_first=decode_first_facts(sched, running_batch),
    )
    if gate.deferred:
        _note_decode_first(sched, gate)
    return gate if gate.barrier else None


#: F3: (wake_seq, decode rounds run since that wake) -- the event loop counts
#: every decode batch it launches (:func:`note_decode_round`).
ROUNDS_ATTR = "_weg2_decode_first_rounds"


def note_decode_round(sched, batch) -> None:
    """F3: one decode batch launched on this group (called by the event loop
    right after ``run_batch``; every rank launches the same batches)."""
    mode = getattr(batch, "forward_mode", None)
    if mode is None or not mode.is_decode():
        return
    seq = getattr(sched, "_weg2_wake_seq", None)
    have = getattr(sched, ROUNDS_ATTR, None)
    n = have[1] if have is not None and have[0] == seq else 0
    setattr(sched, ROUNDS_ATTR, (seq, n + 1))


def rounds_since_wake(sched) -> int:
    have = getattr(sched, ROUNDS_ATTR, None)
    seq = getattr(sched, "_weg2_wake_seq", None)
    return int(have[1]) if have is not None and have[0] == seq else 0


def resume_tail(req) -> int:
    """The tokens a parked request's resume extends: above the park's read
    cap (the resumable depth, #59b / PARK-READ = RESUMABLE). No cap = unknown
    = the whole request (it is not a cheap resume)."""
    ntok = d_park_read._request_tokens(req)
    if ntok is None:
        return 1 << 30
    cap = d_park_read.read_cap(req)
    return int(ntok) if cap is None else max(0, int(ntok) - int(cap))


def realised_resume_tail(sched, req, head_inputs) -> int:
    """F3b (01.10., y6h 10011531): the tail the resume will REALLY extend.

    The park's read cap is a promise, not a measurement. At the wake the read
    can come back short and the prefix demoted (``#1036 PREFIX DEMOTED``,
    ``#1028B FETCH CAP lost``): weg2-26-124 had a park tail of 1 token
    (``SETTLE-TAIL tail=1``) and extended 4138 (``X-GATE uncached=4138``,
    5.5 s eager pass); F3 kept it in the first pass and the hand-off
    weg2-28-127 (7 tokens) got its first token 11.9 s after P's end.

    So the tail is the larger of the promise and the GROUP's priced extent --
    the same ``weg2_uncached_extent`` term X-GATE prices on this pass, from
    #823's MIN-reduced match (replicated). Without a group match on a
    multi-rank group the extent is rank-local, so the promise alone decides
    (the verdict stays group-uniform)."""
    tail = resume_tail(req)
    extent = getattr(sched, "weg2_uncached_extent", None)
    if extent is None:
        return tail
    tp_size = int(getattr(getattr(sched, "ps", None), "tp_size", 1) or 1)
    if tp_size > 1:
        from sglang.srt.managers import tp_head_congruence

        if tp_head_congruence.group_match_for(head_inputs, str(req.rid)) is None:
            return tail
    return max(tail, int(extent(req, head_inputs)))


def decode_first_facts(sched, running_batch) -> Optional["d_seats.DecodeFirst"]:
    """F3's replicated inputs for this pass, or None (switch off, no wake yet)."""
    if not d_seats.decode_first_enabled():
        return None
    seq = getattr(sched, "_weg2_wake_seq", None)
    if seq is None:
        return None
    from sglang.srt.environ import envs

    settle = list(getattr(sched, "weg2_post_wake_settle", None) or [])
    head_inputs = getattr(sched, "_pp_head_inputs_this_pass", None)
    tails = {
        str(r.rid): realised_resume_tail(sched, r, head_inputs)
        for r in sched.waiting_queue if d_seats.park_site(r) == d_seats.SITE_FLIP
    }
    return d_seats.DecodeFirst(
        wake_seq=seq,
        rounds=rounds_since_wake(sched),
        running_n=len(getattr(running_batch, "reqs", None) or ()),
        settle_cohort_n=len(settle),  # the #1471 settle holds only reads of the wake
        tails=tails,
        tail_over=int(envs.SGLANG_WEG2_D_DECODE_FIRST_TAIL.get()),
        max_rounds=int(envs.SGLANG_WEG2_D_DECODE_FIRST_ROUNDS.get()),
    )


def _note_decode_first(sched, gate) -> None:
    """One line per wake and deferred set (the census counts every pass)."""
    key = (getattr(sched, "_weg2_wake_seq", None), gate.deferred)
    if getattr(sched, "_weg2_decode_first_said", None) == key:
        return
    sched._weg2_decode_first_said = key
    logger.info(
        "F3 DECODE-FIRST wake=%s deferred=%s rounds=%d -- the wake's hand-offs extend and "
        "decode first; the flip-parked resume(s) keep their seat and extend after",
        key[0], sorted(gate.deferred), rounds_since_wake(sched))


def park_abort(sched, recv_req) -> int:
    """An abort reaches the parked list exactly as it reaches the dormant hold
    (#1445): a parked request owns no device rows (retracted), so dropping it
    and telling the tokenizer is the whole release."""
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.weg2 import tail_handoff as th

    rid = str(getattr(recv_req, "rid", "") or "")
    abort_all = bool(getattr(recv_req, "abort_all", False))
    if th.park_end_enabled():
        # F4 leak (29.09.): an aborted rid is never resumed -- no adopt verdict
        # takes its park parts; wherever it was queued, they go with the abort
        n = th.remove_parks_aborted(rid, abort_all)
        if n:
            logger.info("F4 PARK-END abort: %d park part file(s) removed (rid=%s abort_all=%s)",
                        n, rid, abort_all)
    # Q-699: the aborted rid's tail staging ends with it (its parts may be gone)
    try:
        from sglang.srt.weg2 import tail_adopt as _ta

        _ta.drop_aborted(rid, abort_all)
    except Exception:  # noqa: BLE001 -- bookkeeping, never the abort itself
        pass
    parked = getattr(sched, "weg2_d_parked", None)
    if not parked:
        return 0
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
                len(gone), rid, abort_all)
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
