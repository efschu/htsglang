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

from sglang.srt.weg2 import d_park_draft, d_seats

logger = logging.getLogger(__name__)


def parked_list(sched) -> list:
    parked = getattr(sched, "weg2_d_parked", None)
    if parked is None:
        parked = sched.weg2_d_parked = []
    return parked


def park_running(sched, recv_req):
    """Retract every running D request RETAINING its span (KV, the node's
    GDN/Mamba anchor, the draft rows) with a forced host write-through -- the
    sleep flushes the tree and the store is the only copy that survives it
    (#969D/#1068) -- and keep it, with whatever only queued on D, in
    ``weg2_d_parked`` (the sleep asserts an idle group). The in-flight batch
    lands first, in upstream ``pause_generation``'s retract shape. Nothing is
    aborted and nothing is told to the tokenizer: the streams stay open."""
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqOutput
    from sglang.srt.mem_cache.base_prefix_cache import FORCE_HOST_WRITE_THROUGH_ATTR

    epoch = int(getattr(recv_req, "epoch", 0) or 0)
    reason = str(getattr(recv_req, "reason", "") or "")
    if not d_seats.d_park_active():
        return Weg2ParkRunningReqOutput(
            success=False, parked=[], epoch=epoch,
            message="W-PARK refused: not group D (or SGLANG_WEG2_D_PARK=0) -- nothing parked",
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
    # H91d: the draft rows have no host twin (tier off) -- copy them off
    # BEFORE the retraction hands the slots to the tree (d_park_draft).
    d_park_draft.save_parked(sched, running, site=d_seats.SITE_FLIP)
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
    queued = list(sched.waiting_queue)
    sched.waiting_queue = []
    sched.weg2_d_parked = d_seats.order_waiting(list(parked) + list(retracted) + queued)
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
    rids = [str(r.rid) for r in sched.weg2_d_parked if d_seats.park_site(r) is not None]
    held = [str(r.rid) for r in sched.weg2_d_parked if d_seats.park_site(r) is None]
    logger.info(
        "WEG2-D-PARK park_running epoch=%d reason=%s: %d running retracted (span retained, "
        "forced host write-through), parked=%s queued-behind=%s -- the sleep holds them "
        "first, the wake resumes oldest first",
        epoch, reason, len(retracted), [r[:12] for r in rids], [r[:12] for r in held],
    )
    return Weg2ParkRunningReqOutput(
        success=True, parked=rids, held=held, epoch=epoch,
        message="parked %d, queued behind them %d" % (len(rids), len(held)),
    )


def hold_parked(sched, *, hold_armed: bool) -> int:
    """The sleep leg's dormant point (``weg2_dormant`` just set, HiCache
    drained, tree flushed): the parked requests enter the #1443 dormant hold
    FIRST, oldest first, their storage prefetch issued by the ordinary intake
    so it runs during the flip. Hold not armed: they stay parked and the first
    awake pass re-queues them."""
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
    for req in moved:
        sched._add_request_to_queue(req, is_retracted=True)
    mine = _to_queue_head(sched, moved)
    logger.info("WEG2-D-PARK requeue (awake): %d parked request(s) at the queue head %s",
                len(mine), [str(r.rid)[:12] for r in mine])
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
                len(mine), [str(r.rid)[:12] for r in mine])
    return len(mine)


def admission(sched, running_batch):
    """One pass's D admission verdict: parked first, the rest in the group's
    order, the barrier/blocked set of ``d_seats.admission_gate``. None when
    the park is off or nothing is parked -- the stock loop, untouched."""
    if not d_seats.d_park_active():
        return None
    sched.waiting_queue = d_seats.order_waiting(sched.waiting_queue)
    book = getattr(sched, "_weg2_d_resume_book", None)
    if book is None:
        book = sched._weg2_d_resume_book = d_seats.ResumeBook.from_env()
    pending = list(getattr(sched, "weg2_post_wake_settle", None) or []) + list(
        getattr(sched, "weg2_dormant_hold", None) or []
    )
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
