"""PP-VOID-PASS (NF 13.10.2026, FIXES-NACH-FREEZE Nr. 45): the whole PP group discards the same pass, the server lives.

THE GAP THIS CLOSES. ``prefill_oom_retreat`` (Nr. 43) takes a pass back when ``PrefillOutOfPages`` / ``AgreedRoomShort``
hits, but only where the followers execute PP0's row (``followers_execute_row``: ``FLLIPER_PDFLIP_P_ROW_AUTHORITY``). NF group P
does not (``#631 ROW AUTHORITY DISABLED``, ``pp_row_carrier_present`` False): PP1/PP2 plan every pass from their OWN queue
and pool and then block in the proxy receive for the frame PP0 sends iff it has a batch (``_event_loop_pp_body``: send iff
``cur_batch``, receive iff ``cur_batch``). A PP0 that retreated alone sends no frame, the followers wait for one (RING COMMIT
TIMEOUT / #1233 width divergence) -- so on NF-P the typed exception propagated and the rank died, four times on 13.10.
(11:59:14 prefill, 18:30:21 / 19:36:02 / 20:11:43 load-back). PLANNER DECISION: ``FLLIPER_PDFLIP_P_ROW_AUTHORITY`` stays OFF on
NF (never measured, changes the PP execution); instead the group discards the pass TOGETHER.

THE CARRIER AND THE COMMON ABORT POINT. The one place every stage already meets for every pass is the PROXY FRAME of that
pass: one frame per (sender batch, receiver batch) pair, received by exactly the stages that hold a batch in the slot, in
FIFO order, before the stage launches (``_pp_recv_proxy_tensors`` -> ``_pp_launch_batch``), and nothing of the pass leaves
a stage before its launch (no output, no forward-side send). So the common abort point is the receive of that frame:

* the stage that cannot pay the pass (PP0: ``prepare_for_extend`` / ``add_one_req`` -> ``load_back``) takes its own plan back
  BEFORE its launch and, instead of staying silent, sends a VOID FRAME in the slot of the frame it owes (``{KEY: statement}``,
  no tensors, the ordinary stamp (slot, seq, ...) so ``pp_proxy_stamp_names_pass`` / the #1057 stale filter judge it like any
  frame). Nothing else of the pass has left PP0: no hidden states, no output expectation (a middle chunk skips the output
  ring anyway; a slot without a batch is expected to have none, ``_pp_send_output_to_next_stage``), no chain-side decision;
* a follower holding a batch for that slot receives the void frame INSTEAD of hidden states, voids its own batch with the
  established own-void (``_pp_void_own_batch``: members parked unreset, the prepared chunk's rows given back
  (``_park_chunked_prefill_chunk(pass_allocated=True)``), the admission lock refs returned, ``chunked_req`` restored) and forwards
  the same statement to the next stage in the slot of the frame IT owes. The last stage forwards nothing and sends no output
  (no batch in the slot). Every stage thus performs the SAME act at the SAME pass, in the order of the frame wire; nobody
  decides alone: the statement is the origin's, the followers execute it, the stamp names the pass;
* the verdict "N voids for the same request -> 503" is PP0's alone and rides the request wire (``intake_origin_abort`` ORIGIN-ABORT:
  every stage drops the rid at the same list position); the void frame says ``abort=1`` so the followers hold the request in place
  (``_pdflip_oom_abort_pending``) until the abort lands;
* a LOAD-BACK refusal (``AgreedRoomShort``, add_one_req) on the frame form voids the WHOLE pass, not the one request: a follower
  plans from its own queue and would admit and load back the request PP0 skipped (the 18:30/19:36/20:11 pass skew). The request is
  skipped in the loop as before (``NO_TOKEN``), the pass is planned to its end, and the event loop discards it as a whole right
  after the plan (``after_plan``: own-void + void frame) -- the unwinding is the proven own-void, not a second implementation.

FOLLOWER FIRST. Followers plan rank-locally, so the exception can come up on PP1/PP2 first. The follower unwinds its own plan
(same helpers), does NOT decide anything for the group, and takes the frame it owes: a VOID frame means PP0 failed the same pass
(the usual order: PP0 allocates every chunk first) -- consistent, forwarded; a REAL frame means PP0 has already launched the pass
(its hidden states are on the wire, a middle chunk has no output to wait for): a stage cannot take a launched pass back from the
other stages, so the group stops by name (:class:`VoidPassLate`, RAENGE-NIE-UNEINS), exactly as before this change -- but with
the terms. That residue is open and stated, not hidden (FIXES Nr. 45).

NOT HERE: tp groups (a rank-local void in a tp group is a split), the row form (no frame = nothing planned, Nr. 43 as is), D, a
gapped wire, ``--p-layer-split dynamic``, anchor tails. Switch ``FLLIPER_PDFLIP_PP_VOID_PASS_ENABLE`` (default on; ``0`` = Nr. 43 as it was:
carrierless -> ``NOT TAKEN (no_row_carrier)``, the rank dies).

MARKER (every rank, one line per void pass): ``PP-VOID-PASS rank= cause= rid= need= free= pass= tries=`` (+ ``role=origin|relay|
follower-first abort= wall[...]``)."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_PP_VOID_PASS_ENABLE"
KEY = "__void_pass__"
MARK = "PP-VOID-PASS"
GROUP_ENV = "FLLIPER_PDFLIP_GROUP"

CAUSE_PREFILL = "prefill_oom"
CAUSE_LOADBACK = "agreed_room_short"

#: scheduler attributes (rank-local, one pass wide)
OUT_ATTR = "_pp_void_pass_out"  # statement to put on the frame this rank owes downstream in this slot
IN_ATTR = "_pp_void_pass_in"  # statement that arrived on the frame just received
AFTER_PLAN_ATTR = "_pp_void_pass_after_plan"  # load-back refusal: discard the whole planned pass after the plan
OWES_ATTR = "_pp_void_pass_owes_frame"  # this rank unwound its own plan: it must still take the frame of the slot
SEEN_ATTR = "_pp_void_pass_seen"  # census
QUEUE0_ATTR = "_pp_void_pass_queue0"  # the waiting queue's rid order right before this pass's plan


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    return (e.get(ENV, "") or "").strip().lower() not in ("0", "false", "no", "off")


class VoidPassLate(RuntimeError):
    """A follower unwound its own plan after PP0 had launched the pass for real. The group cannot discard a launched pass from
    one stage (its hidden states are on the wire); it stops by name with the terms (RAENGE-NIE-UNEINS)."""


class Statement(NamedTuple):
    cause: str
    rid: str
    need: int
    free: int
    pass_no: int  # the origin's forward_ct when it voided (the stamp carries the same counter)
    tries: int  # the origin's consecutive-void count for the rid
    abort: bool  # the origin answered this rid 503 (ORIGIN-ABORT on the request wire is on its way)
    origin: int  # pp_rank that raised first

    def to_wire(self) -> Tuple:
        return (str(self.cause), str(self.rid), int(self.need), int(self.free), int(self.pass_no), int(self.tries),
                1 if self.abort else 0, int(self.origin))

    @staticmethod
    def from_wire(w) -> Optional["Statement"]:
        try:
            c, rid, need, free, pn, tries, ab, org = w[:8]
            return Statement(str(c), str(rid), int(need), int(free), int(pn), int(tries), bool(int(ab)), int(org))
        except Exception:  # noqa: BLE001 -- a statement this rank cannot read is no statement (the frame then stops by name)
            return None


def make_statement(sched, *, cause: str, rid: Any, need: int, free: int, tries: int, abort: bool) -> Statement:
    return Statement(cause, str(rid), int(need), int(free), int(getattr(sched, "forward_ct", -1) or -1), int(tries),
                     bool(abort), int(getattr(getattr(sched, "ps", None), "pp_rank", 0) or 0))


# ---------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------


def frame_carrier(sched) -> Tuple[bool, str]:
    """(True, '') when the void FRAME is the carrier on this boot form: group P, a multi-stage PP chain whose followers plan
    for themselves (no row carrier), a tp group of one, an ungapped wire. The row form needs no frame (no frame = nothing
    planned) and the single-stage group has nobody to tell."""
    if not enabled():
        return False, "switch_off"
    if str(os.environ.get(GROUP_ENV, "")).strip().upper() != "P":
        return False, "not_group_p"
    ps = getattr(sched, "ps", None)
    if ps is None or int(getattr(ps, "pp_size", 1) or 1) <= 1:
        return False, "single_stage"
    if int(getattr(ps, "tp_size", 1) or 1) != 1:
        return False, "tp_group"
    if getattr(sched, "_pp_gapped_wire", False):
        return False, "gapped_wire"
    from flliper.srt.pdflip.prefill_oom_retreat import followers_execute_row

    if followers_execute_row(sched):
        return False, "row_form"
    return True, ""


def rank_of(sched) -> int:
    return int(getattr(getattr(sched, "ps", None), "pp_rank", 0) or 0)


# ---------------------------------------------------------------------------
# per-pass state
# ---------------------------------------------------------------------------


def set_out(sched, stmt: Optional[Statement]) -> None:
    setattr(sched, OUT_ATTR, stmt)


def take_out(sched) -> Optional[Statement]:
    stmt = getattr(sched, OUT_ATTR, None)
    setattr(sched, OUT_ATTR, None)
    return stmt


def owe_frame(sched, stmt: Statement) -> None:
    setattr(sched, OWES_ATTR, stmt)


def owes_frame(sched) -> bool:
    return getattr(sched, OWES_ATTR, None) is not None


def note_incoming(sched, stmt: Statement) -> None:
    setattr(sched, IN_ATTR, stmt)


def take_incoming(sched) -> Optional[Statement]:
    stmt = getattr(sched, IN_ATTR, None)
    setattr(sched, IN_ATTR, None)
    return stmt


def frame_dict(stmt: Statement) -> Dict[str, Any]:
    """The void frame: no tensors, one non-tensor entry (precedent: ``__msg_type__`` / ``__stamp__`` travel the same way and
    are popped by the receiver before anything maps over the dict)."""
    return {KEY: stmt.to_wire()}


def pop_from_frame(raw) -> Optional[Statement]:
    """Receiver side, before any other pop: the statement of a void frame, else None. A present but unreadable statement is
    an error (a frame without tensors that is not a void frame must never be computed on)."""
    if not isinstance(raw, dict) or KEY not in raw:
        return None
    w = raw.pop(KEY)
    stmt = Statement.from_wire(w)
    if stmt is None:
        raise RuntimeError(f"{MARK}: a void frame carried an unreadable statement {w!r}")
    return stmt


# ---------------------------------------------------------------------------
# marker
# ---------------------------------------------------------------------------


def wall_census(sched) -> str:
    """What the next case needs to show WHY the pool could not pay (FIXES Nr. 45 part 2): write-throughs / load-backs in flight,
    parked requests, tree evictable, host pool room. Best effort, never raises."""
    out = []
    try:
        tree = getattr(sched, "tree_cache", None)
        out.append("wt_inflight=%d" % len(getattr(tree, "ongoing_write_through", None) or ()))
        out.append("lb_inflight=%d" % len(getattr(tree, "ongoing_load_back", None) or ()))
        ev = getattr(tree, "evictable_size", None)
        if callable(ev):
            out.append("evictable=%s" % (ev(),))
        # the host arena per component (the 13.10. walls all stood at ARENA-DROP freed=0 with stages i/ii/iii empty)
        for comp in getattr(tree, "_components_tuple", None) or ():
            pool = (getattr(comp, "_full_kv_pool_host", None) or getattr(comp, "_mamba_pool_host", None)
                    or getattr(comp, "_swa_kv_pool_host", None))
            av = getattr(pool, "available_size", None)
            if callable(av):
                out.append("host_free[%s]=%s" % (getattr(getattr(comp, "component_type", None), "name", "?"), av()))
        q = list(getattr(sched, "waiting_queue", None) or ())
        out.append("queue=%d parked=%d" % (len(q), sum(1 for r in q if getattr(r, "pdflip_pool_parked", False))))
    except Exception as exc:  # noqa: BLE001
        out.append("census_error=%s" % type(exc).__name__)
    return " ".join(out)


def log_marker(sched, stmt: Statement, role: str) -> None:
    n = int(getattr(sched, SEEN_ATTR, 0) or 0) + 1
    setattr(sched, SEEN_ATTR, n)
    logger.warning(
        "%s rank=%d cause=%s rid=%s need=%d free=%d pass=%d tries=%d role=%s origin_rank=%d abort=%s n=%d wall[%s]",
        MARK, rank_of(sched), stmt.cause, stmt.rid, stmt.need, stmt.free, stmt.pass_no, stmt.tries, role, stmt.origin,
        stmt.abort, n, wall_census(sched),
    )


# ---------------------------------------------------------------------------
# event-loop hooks (called from scheduler_pp_mixin._event_loop_pp_body)
# ---------------------------------------------------------------------------


def snapshot_queue(sched) -> None:
    """Right before the plan: remember the order of the waiting queue. A void is "the pass never happened" and must leave the
    queue as it was on EVERY rank -- the followers' own-void appends its members at the tail, PP0's retreat gives fresh
    requests back at the tail, and a refused load-back stays where it stands on PP0 only; with one waiting request more than
    the pass touched, ranks would admit in different orders next pass (W27). Cheap: one list of rids per pass, only on a
    carrierless multi-stage group P (otherwise nothing is stored)."""
    if not frame_carrier(sched)[0]:
        if getattr(sched, QUEUE0_ATTR, None) is not None:
            setattr(sched, QUEUE0_ATTR, None)
        return
    setattr(sched, QUEUE0_ATTR, [str(getattr(r, "rid", "")) for r in (getattr(sched, "waiting_queue", None) or ())])


def restore_queue_order(sched) -> None:
    """After a void: the waiting queue in the order of :func:`snapshot_queue` (stable; whatever the snapshot does not know
    keeps its relative place at the end). Never drops or duplicates a request."""
    order = getattr(sched, QUEUE0_ATTR, None)
    q = getattr(sched, "waiting_queue", None)
    if not order or not q:
        return
    pos = {rid: i for i, rid in enumerate(order)}
    big = len(order)
    try:
        sched.waiting_queue = sorted(q, key=lambda r: pos.get(str(getattr(r, "rid", "")), big))
    except Exception:  # noqa: BLE001 -- the order is a courtesy of the void, never a reason to stop
        logger.warning("%s: could not restore the waiting-queue order", MARK, exc_info=True)


def _batch_of(sched, mb_id: int):
    mbs = getattr(sched, "mbs", None) or ()
    return mbs[mb_id] if 0 <= mb_id < len(mbs) else None


def _forget_offers(sched, batch) -> None:
    from flliper.srt.pdflip.prefill_oom_retreat import _void_decision

    _void_decision(sched, list(getattr(batch, "reqs", None) or ()))


def after_plan(sched, mb_id: int) -> bool:
    """A load-back refusal (``AgreedRoomShort``) marked this pass for discard: planned to its end with the refused request
    skipped, it is discarded AS A WHOLE right here -- before anything of it is sent or launched -- by the own-void every stage
    already uses, and the slot's frame becomes a void frame (PP0) / is owed (follower). True iff the pass was discarded."""
    stmt = getattr(sched, AFTER_PLAN_ATTR, None)
    if stmt is None:
        return False
    setattr(sched, AFTER_PLAN_ATTR, None)
    batch = _batch_of(sched, mb_id)
    _forget_offers(sched, batch)  # also voids the pass's decision / load-back row when there is no batch
    if batch is not None:
        if not sched._pp_void_own_batch(mb_id) and _batch_of(sched, mb_id) is not None:
            # the slot holds a launched, unconsumed pass of this rank: it cannot be vacated (#1020) -- stop by name
            raise VoidPassLate(
                f"{MARK} rank={rank_of(sched)}: slot {mb_id} holds a launched pass whose result is outstanding; the "
                f"load-back void ({stmt.rid}) cannot vacate it"
            )
    restore_queue_order(sched)
    if rank_of(sched) == 0:
        set_out(sched, stmt)
        log_marker(sched, stmt, "origin")
    else:
        owe_frame(sched, stmt)
        log_marker(sched, stmt, "follower-first")
    return True


def _mark_abort_pending(sched, rid: str) -> None:
    """The origin answered ``rid`` 503: until the ORIGIN-ABORT lands on this rank, nothing may requeue or advance it."""
    for r in [getattr(sched, "chunked_req", None), *list(getattr(sched, "waiting_queue", None) or ())]:
        if r is not None and str(getattr(r, "rid", "")) == rid:
            try:
                r._pdflip_oom_abort_pending = True
                r.pdflip_pool_parked = False
            except Exception:  # noqa: BLE001
                pass


def at_receive(sched, mb_id: int, cur_batch, recv: Callable[[], Any]):
    """The slot's frame. ``recv`` posts the stage's proxy receive (it returns the proxy; a void frame surfaces through
    :func:`take_incoming`). Returns ``(cur_batch, proxy)``; ``(None, None)`` = the pass is discarded here, the launch is
    skipped. Not called for the first stage."""
    owes = owes_frame(sched)
    if cur_batch is None and not owes:
        return None, None
    proxy = recv()
    stmt = take_incoming(sched)
    if stmt is None:
        if owes:
            mine = getattr(sched, OWES_ATTR)
            setattr(sched, OWES_ATTR, None)
            raise VoidPassLate(
                f"{MARK} rank={rank_of(sched)} cause={mine.cause} rid={mine.rid} need={mine.need} free={mine.free}: this "
                f"stage could not pay its part of pass {getattr(sched, 'forward_ct', '?')} and unwound its plan, but the "
                f"upstream frame for slot {mb_id} is a REAL one -- the pass is launched upstream and one stage cannot take "
                f"a launched pass back from the others. Stopping by name (RAENGE-NIE-UNEINS); the origin voids first in "
                f"the usual order, so this means the walls of the stages differ (UD-REFUSED / EVICT-FRONTIER-CENSUS name them)."
            )
        return cur_batch, proxy
    setattr(sched, OWES_ATTR, None)
    if cur_batch is not None:
        _forget_offers(sched, cur_batch)
        if not sched._pp_void_own_batch(mb_id) and _batch_of(sched, mb_id) is not None:
            raise VoidPassLate(
                f"{MARK} rank={rank_of(sched)}: a void frame arrived for slot {mb_id} but the slot holds a launched, "
                f"unconsumed pass of this rank; it cannot be vacated (#1020)"
            )
    restore_queue_order(sched)
    if stmt.abort:
        _mark_abort_pending(sched, stmt.rid)
    if not owes:  # a stage that raised first has already printed its line (one line per rank and void pass)
        log_marker(sched, stmt, "relay")
    set_out(sched, stmt)  # the next stage is owed the same statement in this slot
    return None, None


def send_owed_frame(sched, mb_id: int, send: Callable[[Dict[str, Any], int], Any]) -> bool:
    """After the plan/launch junction on a non-last stage without a batch: put the void statement on the frame the next
    stage waits for. ``send(frame_dict, mb_id)`` posts it (the real wire adds ``__msg_type__`` and the stamp)."""
    stmt = take_out(sched)
    if stmt is None:
        return False
    send(frame_dict(stmt), mb_id)
    return True
