"""PREFILL-OOM-RETREAT (NF 13.10.2026, fix 43): a prefill batch the pool cannot pay is taken back, the server lives.

THE DEATH (NF int24f4, boot ``..._b60520f01d_1013_073910``, P.log 1214208-1214253, 11:59:14Z). PP0 planned chunk 7 of
weg2-587-1256 (16384 rows); ``alloc_paged_token_slots_extend`` found 13568 free and a peel that paid 320 of the 3200 it
asked for (150336 reported evictable, 150016 of them behind the un-backable leaf [6193]) and raised
``RuntimeError('Prefill out of memory')`` -> RANK-DEATH -> SIGQUIT -> W17 -> Server STOP after 4 h of load. The same wall
had been survived twice that boot (10:19:08, 11:55:42/46) only because the allocation happened to fit the free room.

USER LAW (13.10.): the server never dies of this -- the twin of "the leak check warns and carries on".

WHAT THIS DOES. ``mem_cache.common.alloc_paged_token_slots_extend`` raises the typed ``PrefillOutOfPages`` (a
``RuntimeError`` subclass, same message) with the terms need/free/evictable_paid. ``Scheduler._get_new_batch_prefill_raw``
catches it around ``prepare_for_extend`` -- which runs BEFORE the batch is launched and BEFORE anything of this pass is
sent to a follower -- and calls :func:`retreat`. The retreat leaves the pass exactly as an admission pass that admitted
NOTHING would leave it (the ``len(can_run_list) == 0`` return, the shape the #679 park already produces every day), so a
follower sees no half batch: PP0 never named the pass. Concretely:

* the pass's decision is voided (``_pp_admission_last_built_decision``, ``_pp_load_back_wire``, the guard's offer streak);
* the carried continuation is parked IN PLACE (#679 form: ``set_extend_range(prefix, prefix)``, ``weg2_pool_parked``) --
  its rows stay locked, they do not become more un-backed evictable mass; its in-flight count is given back;
* a request the pass admitted FRESH is given back by the established retract flow (``release_kv_cache`` -> ``reset_for_retract``
  -> ``_add_request_to_queue(is_retracted=True)``, the sequence ``process_pending_weg2_park`` runs), and the batch's
  ``alloc_req_slots`` rows/mamba states go with it;
* line ``PREFILL-OOM-RETREAT rid= need= free= evictable_paid= retries=`` names every retreat.

THE WALL STANDS. A request retreated ``SGLANG_WEG2_PREFILL_OOM_RETRIES`` times in a row (default 3) is answered
503 with the ``WEG2-INTAKE-STALL`` mark (fix round 2, review finding 7: the H91a ``intake_verdict`` IMPOSSIBLE shortcut --
503 at the first retreat when the REMAINING need exceeds free + paid with nothing in flight -- is OPT-IN now,
``SGLANG_WEG2_PREFILL_OOM_FAST_ABORT=1``: it reads "paid" from one peel, the number the incidents showed to be unreliable, and
the order says "after N attempts") -- the front requeues exactly that request at the head and flips (its ``is_intake_stall``
test is the mark) -- and the abort rides the request wire (``intake_origin_abort``) so every PP stage drops the rid in the
same pass. The server lives; one request is re-routed.

SCOPE (everything else dies exactly as before, byte for byte): group P, the decision authority (pp_rank 0), a tp group of
one, the local-PP floor (``pp_slot_fidelity.FLOOR_LOCAL_PP_ATTR``), no dynamic layer split, no anchor tails in the pass, and
-- fix round 1, review blocker 2 -- ONLY WHERE THE FOLLOWERS EXECUTE PP0's ROW (:func:`followers_execute_row`: the #631 row
authority with its frame probe, ``pp_row_carrier_present``'s law: "PP0 may withhold only when the followers will execute its
row"). There "no frame" is PP0's statement "I planned nothing this pass" (#969J) and a PP0 that takes the pass back is the
same thing as a PP0 that admitted nothing. On the CARRIERLESS form (NF group P today: ``#631 ROW AUTHORITY DISABLED``,
``SGLANG_WEG2_P_ROW_AUTHORITY`` off) PP1/PP2 plan the same pass from their own pool and then block in the proxy receive for a
frame a retreating PP0 never sends (#969Z/#973: ``RING COMMIT TIMEOUT`` after 120 s, or a #1233 width divergence) -- the skew
the retreat exists to avoid would be built into it, so there the retreat is NOT taken, the typed exception propagates and the
rank dies as before (``PREFILL-OOM-RETREAT NOT TAKEN (no_row_carrier)``). A FOLLOWER (PP1/PP2) that fails the same allocation
after PP0 launched cannot take its stage of the pass back either -- the upstream's hidden states are on the wire; it still
raises (the typed exception names the terms). Prevention for those forms is the PR room vote and the walk fix
(``pp_room_vote.estimate_payable``).

THE LOAD-BACK ENTRANCE (NF 13.10. 18:30:21Z, int24f6 died after 4 h 38: ``PR AGREED-ROOM SHORT rid=weg2-302-1956
kv_tokens=92288 agreed=191744 avail=92032 evictable=99712 evicted=64 avail_after=92096``, the same wall one door earlier:
99712 rows reported evictable behind ONE un-payable frontier leaf, the peel delivered 64 of the 256 missing). The
``pp_slot_fidelity.AgreedRoomShort`` of ``local_pp_room`` is deliberately loud (#1004: a stage that refuses ALONE starts the
request on its peers only). :func:`retreat_loadback` is the one exception, under the same scope as the prefill retreat
(PP0, group P, followers execute PP0's row): PP0 skips the request for this pass (``NO_TOKEN``, the request stays in its
queue exactly as at ``WEG2-LOADBACK-WAIT``), so the row it sends names nothing of it and every stage leaves the load-back of
this pass undone together. ``load_back`` has unwound its two lock refs before the exception leaves it. Same ledger, same
N-then-503 escalation, same ``PREFILL-OOM-RETREAT`` marker (``where=loadback``). A follower that raises it, or the carrierless
form, stops as before.

A write-through in flight is the candidate for the transient form of this wall (the publish sweep issued 6 in that same second):
it frees the leaf within a few hundred ms, which is shorter than N back-to-back passes. So while a write-through is in flight
the N retreats do not end in a 503 before ``SGLANG_WEG2_PREFILL_OOM_HOLD_S`` (default 5 s) have passed since the first one.

THE CARRIERLESS FORM (NF group P today) -- FIXES Nr. 45, :mod:`sglang.srt.weg2.pp_void_pass`. There the retreat is taken too,
but as a GROUP-WIDE VOID PASS: the rank that cannot pay unwinds its plan and the slot's proxy frame becomes a VOID FRAME
(``pp_void_pass``) that every stage executes at its receive, before its launch; a load-back refusal discards the whole planned
pass (the followers would admit the skipped request themselves). A follower that raises first unwinds its plan the same way and
takes the frame it owes (void frame = consistent; real frame = ``VoidPassLate``, stop by name). The N-then-503 verdict stays
PP0's alone (ORIGIN-ABORT on the request wire). Switch ``SGLANG_WEG2_PP_VOID_PASS_ENABLE`` (default on); off, the carrierless form is
refused as it was (``NOT TAKEN (no_row_carrier)``).

Switch ``SGLANG_WEG2_PREFILL_OOM_RETREAT`` (default on; ``0`` = the raise of before)."""

from __future__ import annotations

import logging
import os
import time
import types
from http import HTTPStatus
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_PREFILL_OOM_RETREAT"
ENV_RETRIES = "SGLANG_WEG2_PREFILL_OOM_RETRIES"
ENV_HOLD = "SGLANG_WEG2_PREFILL_OOM_HOLD_S"
ENV_FAST_ABORT = "SGLANG_WEG2_PREFILL_OOM_FAST_ABORT"
HOLD_DEFAULT_S = 5.0
CONSECUTIVE_S = 60.0  # retreats further apart than this are not "in a row"
GROUP_ENV = "SGLANG_WEG2_GROUP"
MARK = "PREFILL-OOM-RETREAT"
RETRIES_DEFAULT = 3

RETRY = "retry"
ABORT = "abort"


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    return (e.get(ENV, "") or "").strip().lower() not in ("0", "false", "no", "off")


def max_retries(env=None) -> int:
    e = os.environ if env is None else env
    try:
        return max(1, int((e.get(ENV_RETRIES, "") or "").strip() or RETRIES_DEFAULT))
    except ValueError:
        return RETRIES_DEFAULT


def fast_abort_enabled(env=None) -> bool:
    """Opt-in (default OFF): answer 503 at the FIRST retreat when the request's remaining need exceeds free + paid with
    nothing in flight (the H91a ``intake_verdict`` IMPOSSIBLE). Off, the wall stands only after N consecutive retreats, as
    ordered -- the shortcut reads "paid" from ONE peel, which is exactly the number the incidents showed to be unreliable
    (the load-back entrance would 503 a request 192 rows short at its first visit)."""
    e = os.environ if env is None else env
    return (e.get(ENV_FAST_ABORT, "") or "").strip().lower() in ("1", "true", "yes", "on")


def hold_s(env=None) -> float:
    e = os.environ if env is None else env
    try:
        return max(0.0, float((e.get(ENV_HOLD, "") or "").strip() or HOLD_DEFAULT_S))
    except ValueError:
        return HOLD_DEFAULT_S


class RetreatLedger:
    """Consecutive retreats per rid. A rid that gets through ``prepare_for_extend`` leaves it."""

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self.first_t: Dict[str, float] = {}
        self.last_t: Dict[str, float] = {}
        #: rids whose 503 abort is on its way through the request wire (the carried continuation: it is dropped by
        #: the chunked-abort machinery of every stage a few passes from now). Never answered twice, never parked.
        self.abort_pending: set = set()
        self.retreats = 0
        self.aborts = 0

    def note(self, rid: Any) -> int:
        k = str(rid)
        now = time.monotonic()
        if now - self.last_t.get(k, now) > CONSECUTIVE_S:
            self.counts.pop(k, None)  # not consecutive any more: an old retreat is no history of this wall
            self.first_t.pop(k, None)
        self.last_t[k] = now
        self.counts[k] = self.counts.get(k, 0) + 1
        self.first_t.setdefault(k, now)
        self.retreats += 1
        return self.counts[k]

    def elapsed(self, rid: Any) -> float:
        t = self.first_t.get(str(rid))
        return 0.0 if t is None else max(0.0, time.monotonic() - t)

    def clear(self, rid: Any) -> None:
        self.counts.pop(str(rid), None)
        self.first_t.pop(str(rid), None)
        self.last_t.pop(str(rid), None)

    def clear_batch(self, reqs: Iterable[Any]) -> None:
        if not self.counts:
            return
        for r in reqs:
            self.clear(getattr(r, "rid", ""))

    def forget(self, rid: Optional[str]) -> None:
        """An abort ends the rid's history (``None`` = abort_all, prefix semantics as the abort's)."""
        if rid is None:
            self.counts.clear()
            self.first_t.clear()
            self.last_t.clear()
            self.abort_pending.clear()
            return
        rid = str(rid)
        for k in [k for k in self.counts if k.startswith(rid)]:
            del self.counts[k]
            self.first_t.pop(k, None)
            self.last_t.pop(k, None)
        self.abort_pending = {k for k in self.abort_pending if not k.startswith(rid)}


def verdict(*, count: int, retries: int, remaining_need: int, free: int, paid: int,
            pool_tokens: int, inflight: bool, elapsed: float = 0.0, hold: float = 0.0,
            wt_inflight: bool = False, fast_abort: bool = False) -> Tuple[str, str]:
    """(RETRY|ABORT, why). ABORT when the retry budget is spent -- and, while a write-through is in flight (it frees the
    wall by itself within ms, faster than N back-to-back passes), only once ``hold`` seconds have passed since the first
    retreat. With ``fast_abort`` (opt-in, ``SGLANG_WEG2_PREFILL_OOM_FAST_ABORT``) also at once when what the pool can still
    give (free + paid) cannot cover the request's REMAINING need and nothing in flight would free more (the H91a intake
    verdict)."""
    from sglang.srt.weg2.intake_stall import INTAKE_IMPOSSIBLE, intake_verdict

    if int(count) >= int(retries) and not (bool(wt_inflight) and float(elapsed) < float(hold)):
        return ABORT, f"retries={int(count)}>={int(retries)}"
    if fast_abort and int(remaining_need) > 0 and intake_verdict(
        need_tokens=int(remaining_need), pool_tokens=int(pool_tokens),
        free_tokens=max(0, int(free)) + max(0, int(paid)), evictable_tokens=0,
        inflight=bool(inflight),
    ) == INTAKE_IMPOSSIBLE:
        return ABORT, f"impossible need={int(remaining_need)} free+paid={max(0, int(free)) + max(0, int(paid))}"
    return RETRY, ""


def abort_message(rid: Any, exc, count: int, why: str, *, too_large_pool: int = 0) -> str:
    from sglang.srt.weg2.intake_stall import STALL_MARK, TOO_LARGE_MARK

    # a request larger than this group's WHOLE pool can never be admitted by a flip: name it, so the front refuses it
    # instead of requeueing it (weg2xsn291, the same rule ``_weg2_intake_stall_observe`` applies)
    big = f" {TOO_LARGE_MARK} pool_tokens={int(too_large_pool)}" if int(too_large_pool) > 0 else ""
    return (
        f"{STALL_MARK} rid={rid} {MARK}{big} need_tokens={int(exc.need)} free={int(exc.free)} "
        f"evictable_reported={int(exc.evictable_reported)} evictable_paid={int(exc.evictable_paid)} "
        f"retries={int(count)} why={why} -- this group's pool could not pay the request's next chunk "
        f"(the evictable rows behind an un-backable leaf are not payable); the front requeues exactly "
        f"this request for the next phase of this group and flips"
    )


# ---------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------


def followers_execute_row(sched) -> bool:
    """Does the pipeline behind PP0 execute PP0's row, so that a pass PP0 sizes or withholds ALONE is the whole group's pass?

    ``pp_row_carrier_present``'s law, without the per-term split (a retreat is no admission term of its own: it is
    "PP0 planned nothing", which the row form's frame probe reads as no frame): the #631 row authority is enabled
    (``pp_row_authority_enabled``) AND either the re-armed P row form runs (``p_row_authority.ROW_ONLY_ATTR``: frame probe
    and chain receiver are built, ``start_check`` refuses to start without them) or the legacy ``pp_flip_counters`` side
    channel exists. A single-stage group has nobody behind PP0 -- trivially true. Unreadable form: False (never a guess
    in the direction of "PP0 may decide alone")."""
    try:
        ps = getattr(sched, "ps", None)
        if int(getattr(ps, "pp_size", 1) or 1) <= 1:
            return True
        from sglang.srt.managers.pp_admission_congruence import pp_row_authority_enabled
        from sglang.srt.weg2 import p_row_authority as _prow

        if not pp_row_authority_enabled(sched):
            return False
        if getattr(sched, _prow.ROW_ONLY_ATTR, False):
            return True
        return getattr(sched, "pp_flip_counters", None) is not None
    except Exception:  # noqa: BLE001
        return False


def applies(sched, adder=None, can_run_list=()) -> Tuple[bool, str]:
    """(True, '') when this seat may retreat; else (False, why) and the caller re-raises.

    Two forms: the ROW form (PP0 only; no frame = nothing planned, Nr. 43) and the FRAME form (carrierless group P: the void
    frame of ``pp_void_pass`` carries the retreat to the followers; PP0 AND the followers may unwind their own plan)."""
    if not enabled():
        return False, "switch_off"
    if str(os.environ.get(GROUP_ENV, "")).strip().upper() != "P":
        return False, "not_group_p"
    ps = getattr(sched, "ps", None)
    if ps is None:
        return False, "not_pp0"
    if int(getattr(ps, "tp_size", 1) or 1) != 1:
        return False, "tp_group"
    from sglang.srt.weg2 import pp_void_pass as _vp

    frame_ok = _vp.frame_carrier(sched)[0]
    if int(getattr(ps, "pp_rank", 0) or 0) != 0:
        if not frame_ok:
            return False, "not_pp0"
    elif not followers_execute_row(sched) and not frame_ok:
        return False, "no_row_carrier"
    from sglang.srt.weg2.pp_slot_fidelity import FLOOR_LOCAL_PP_ATTR

    if not getattr(getattr(sched, "tree_cache", None), FLOOR_LOCAL_PP_ATTR, False):
        return False, "not_local_pp_floor"
    from sglang.srt.weg2 import p_layer_split_runtime as _pls_rt

    if _pls_rt.active() is not None:
        return False, "layer_split_dynamic"
    if adder is not None and getattr(adder, "new_anchor_tails", None):
        return False, "anchor_tails"
    if getattr(sched, "anchor_tails", None):
        return False, "anchor_tails"
    return True, ""


def frame_form(sched) -> bool:
    """True when the retreat of this seat rides the void frame (carrierless P). Only meaningful after ``applies``."""
    from sglang.srt.weg2 import pp_void_pass as _vp

    return _vp.frame_carrier(sched)[0]


def _rank(sched) -> int:
    return int(getattr(getattr(sched, "ps", None), "pp_rank", 0) or 0)


def _inflight(sched) -> bool:
    tree = getattr(sched, "tree_cache", None)
    running = getattr(sched, "running_batch", None)
    try:
        running_n = 0 if running is None else len(running.reqs)
    except Exception:  # noqa: BLE001
        running_n = 0
    return bool(
        running_n
        or getattr(tree, "ongoing_write_through", None)
        or getattr(tree, "ongoing_load_back", None)
    )


def _wt_inflight(sched) -> bool:
    return bool(getattr(getattr(sched, "tree_cache", None), "ongoing_write_through", None))


def _remaining_need(req) -> int:
    try:
        return max(0, len(req.full_untruncated_fill_ids) - len(req.prefix_indices))
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# the retreat
# ---------------------------------------------------------------------------


def _void_decision(sched, can_run_list) -> None:
    """The pass's PP0 decision was built before ``prepare_for_extend`` ran; nothing was sent. Void what stands."""
    decision = getattr(sched, "_pp_admission_last_built_decision", None)
    guard = getattr(sched, "_pp_admission_guard", None)
    rids = [getattr(r, "rid", None) for r in can_run_list]
    if guard is not None and hasattr(guard, "forget_offer"):
        for rid in rids:
            guard.forget_offer(rid)
    sched._pp_admission_last_built_decision = None
    sched._pp_load_back_wire = None
    if decision is not None:
        logger.info("%s voided the pass's admission decision (%d entr%s): it was never sent", MARK,
                    len(getattr(decision, "entries", ()) or ()),
                    "y" if len(getattr(decision, "entries", ()) or ()) == 1 else "ies")


def _park_in_place(req, flag: bool = True) -> None:
    """The #679 park form (schedule_policy.add_chunked_req): no chunk this pass, the request keeps its rows and place."""
    n = len(req.prefix_indices)
    req.set_extend_range(n, n)
    if not flag:
        req.weg2_pool_parked = False
        return
    try:
        from sglang.srt.weg2.park import park_active

        if park_active():
            req.weg2_pool_parked = True
    except Exception:  # noqa: BLE001 -- without the flag the in-place park simply retries next pass
        pass


def _requeue_front(sched, reqs: List[Any]) -> None:
    """The requests given back through the retract flow go to the HEAD of PP0's queue in the order the pass had admitted them
    (review finding 6): ``_add_request_to_queue`` appends, so a given-back request would otherwise change its place from one
    retreat to the next and the N-in-a-row ledger would never see the same rid twice. The followers do not use PP0's queue
    order on the row form (they execute PP0's row), so the move is PP0-local by design, like the park it mirrors."""
    if not reqs:
        return
    ids = {id(r) for r in reqs}
    present = [r for r in reqs if any(q is r for q in sched.waiting_queue)]
    if not present:
        return
    sched.waiting_queue = present + [q for q in sched.waiting_queue if id(q) not in ids]


def _give_back_fresh(sched, req) -> None:
    """A request this pass admitted fresh goes back through the retract flow ``process_pending_weg2_park`` uses:
    release (the batch's ``alloc_req_slots`` already gave it a row, so the ordinary finish release is the matched one),
    reset, queue."""
    from sglang.srt.mem_cache.common import release_kv_cache

    release_kv_cache(req, sched.tree_cache, is_insert=False)
    req.reset_for_retract()
    sched._add_request_to_queue(req, is_retracted=True)


def _answer_abort(sched, req, message: str, *, queued_elsewhere: bool) -> None:
    """503 for exactly this request, relayed to every stage (``intake_origin_abort``), the way the intake stall does it."""
    from sglang.srt.managers.io_struct import AbortReq

    ioa = getattr(sched, "_weg2_intake_origin_aborts", None)
    relayed = bool(ioa.note(req.rid, message)) if ioa is not None else False
    if queued_elsewhere:
        # the rid is held by the chunked-abort machinery of every stage (PP0 included): it answers the tokenizer
        # (``Scheduler.process_pending_chunked_abort`` carries the 503 reason); nothing is sent from here
        if not relayed:
            sched._abort_request_now(AbortReq(
                rid=req.rid,
                finished_reason={"type": "abort", "status_code": HTTPStatus.SERVICE_UNAVAILABLE, "message": message},
            ))
        return
    sched.ipc_channels.send_to_tokenizer.send_output(
        AbortReq(
            rid=req.rid,
            finished_reason={"type": "abort", "status_code": HTTPStatus.SERVICE_UNAVAILABLE, "message": message},
        ),
        req,
    )
    try:
        tree = sched.tree_cache
        if getattr(sched, "enable_hicache_storage", False):
            tree.release_aborted_request(req.rid)
        elif getattr(sched, "enable_hierarchical_cache", False):
            tree.terminate_prefetch(req.rid)
    except Exception:  # noqa: BLE001 -- the answer has gone out regardless
        logger.warning("%s rid=%s: releasing the store read raised", MARK, req.rid, exc_info=True)


def _frame_void(sched, *, cause: str, rid, need, free, count: int, abort: bool) -> None:
    """FRAME form: the retreat is a statement for the whole group (``pp_void_pass``). PP0 owes it on the slot's frame; a
    follower that unwound its own plan owes itself the frame of the slot (void = consistent, real = stop by name)."""
    from sglang.srt.weg2 import pp_void_pass as _vp

    stmt = _vp.make_statement(sched, cause=cause, rid=rid, need=need, free=free, tries=count, abort=abort)
    if _rank(sched) == 0:
        _vp.set_out(sched, stmt)
        _vp.log_marker(sched, stmt, "origin")
    else:
        _vp.owe_frame(sched, stmt)
        _vp.log_marker(sched, stmt, "follower-first")


def retreat(sched, exc, new_batch, can_run_list: List[Any], adder, carried_before=None) -> bool:
    """Take the pass back. True = retreated (the caller returns ``None, running_batch``); False = out of scope or the
    retreat itself failed -- the caller re-raises ``exc`` and the rank dies as before.

    ``carried_before`` is the continuation the pass STARTED with (``Scheduler.chunked_req`` before ``add_chunked_req``).
    The attribute after the adder is not enough: for the LAST chunk of a continuation the adder returns None, the
    attribute is None, and the request -- still in ``can_run_list``, still holding its rows and its place -- would be
    taken for a fresh admission and torn down through the retract flow while the followers keep it as their chunked
    request (review finding 3)."""
    ok, why = applies(sched, adder, can_run_list)
    if not ok:
        if why != "not_group_p":
            logger.error("%s NOT TAKEN (%s): the typed Prefill-out-of-memory propagates -- need=%d free=%d "
                         "evictable_paid=%d", MARK, why, exc.need, exc.free, exc.evictable_paid)
        return False
    frame = frame_form(sched)
    ledger = getattr(sched, "_weg2_prefill_oom_ledger", None)
    if ledger is None:
        ledger = sched._weg2_prefill_oom_ledger = RetreatLedger()
    chunked = getattr(sched, "chunked_req", None)
    new_chunked = getattr(adder, "new_chunked_req", None)
    cont = carried_before if carried_before is not None else (chunked if new_chunked is None else None)
    carried = cont if (cont is not None and new_chunked is None and any(r is cont for r in can_run_list)) else None
    # the middle-chunk count was added for this pass only when the continuation is still the chunked request after the
    # adder (a truncated chunk); the last chunk of a continuation was never counted
    counted = carried is not None and chunked is carried
    restore_chunked = carried is not None and chunked is None
    # the blame: the carried continuation, else the pass's first request (the batch is all-or-nothing)
    blamed = carried if carried is not None else (can_run_list[0] if can_run_list else None)
    if carried is not None and str(carried.rid) in ledger.abort_pending:
        # its 503 is already on the request wire; the chunked abort of every stage drops it a few passes from now.
        # This pass still must not launch a chunk it cannot pay: take it back, answer nothing a second time.
        try:
            _void_decision(sched, can_run_list)
            if counted and int(getattr(carried, "inflight_middle_chunks", 0) or 0) > 0:
                carried.inflight_middle_chunks -= 1
            if restore_chunked:
                sched.chunked_req = carried  # last chunk: the adder had cleared it, the request is still the continuation
            _park_in_place(carried, flag=False)
            logger.warning("%s rid=%s need=%d free=%d evictable_paid=%d: the pass is taken back, the abort is pending",
                           MARK, carried.rid, exc.need, exc.free, exc.evictable_paid)
            if frame:
                _frame_void(sched, cause="prefill_oom", rid=carried.rid, need=exc.need, free=exc.free, count=0, abort=True)
            return True
        except Exception:  # noqa: BLE001
            logger.exception("%s FAILED for rid=%s: the original Prefill-out-of-memory is re-raised", MARK, carried.rid)
            return False
    retries = max_retries()
    count = ledger.note(blamed.rid) if blamed is not None else 0
    pool_tokens = int(getattr(sched, "max_total_num_tokens", 0) or 0)
    v, vwhy = verdict(
        count=count, retries=retries, remaining_need=_remaining_need(blamed) if blamed is not None else 0,
        free=exc.free, paid=exc.evictable_paid, pool_tokens=pool_tokens, inflight=_inflight(sched),
        elapsed=ledger.elapsed(blamed.rid) if blamed is not None else 0.0, hold=hold_s(),
        wt_inflight=_wt_inflight(sched), fast_abort=fast_abort_enabled(),
    )
    if _rank(sched) != 0:
        v, vwhy = RETRY, ""  # the 503 is the origin's verdict alone (ORIGIN-ABORT on the request wire); a follower only unwinds
    try:
        _void_decision(sched, can_run_list)
        if carried is not None:
            if restore_chunked:
                # the LAST chunk of a continuation: ``add_chunked_req`` returned None and cleared the attribute, but the
                # request still holds its rows and its place and the followers still hold it as their chunked request.
                # It stays the continuation here too (its park / its pending 503 need it to be).
                sched.chunked_req = carried
            if counted and int(getattr(carried, "inflight_middle_chunks", 0) or 0) > 0:
                carried.inflight_middle_chunks -= 1  # counted for a chunk that never launched
            # an aborted continuation is NOT offered to the head-of-step rows-back park: that park would requeue it
            # while the abort of the other stages still has to land (a W27 split of PP0 against the followers)
            # FRAME form: no rows-back flag. ``process_pending_weg2_park`` would give PP0's rows back and requeue the
            # continuation on PP0 ALONE, while the followers keep it as their chunked request (their own-void parks without
            # the flag): a rank split. The followers' state after the void is the in-place park, and so is PP0's.
            _park_in_place(carried, flag=not (v == ABORT and blamed is carried) and not frame)
            if v == ABORT and blamed is carried:
                carried._weg2_oom_abort_pending = True
        if new_chunked is not None and chunked is new_chunked:
            sched.chunked_req = None  # the fresh request that became the chunked request goes back below
        given_back = []
        for req in can_run_list:
            if req is carried:
                continue
            if v == ABORT and req is blamed:
                _release_for_abort(sched, req)
                continue
            _give_back_fresh(sched, req)
            given_back.append(req)
        if frame:
            # FRAME form: the followers' own-void parks at the tail and every rank restores the pre-plan order (the void is
            # "the pass never happened"); a front-move here would be a PP0-only order -- a rank split next pass
            from sglang.srt.weg2 import pp_void_pass as _vp

            _vp.restore_queue_order(sched)
        else:
            _requeue_front(sched, given_back)
        if v == ABORT and blamed is not None:
            ledger.aborts += 1
            _rem = _remaining_need(blamed)
            msg = abort_message(blamed.rid, exc, count, vwhy,
                                too_large_pool=pool_tokens if (pool_tokens > 0 and _rem > pool_tokens) else 0)
            logger.error("%s rid=%s need=%d free=%d evictable_paid=%d retries=%d -> ABORT 503 (%s): %s",
                         MARK, blamed.rid, exc.need, exc.free, exc.evictable_paid, count, vwhy, msg)
            _answer_abort(sched, blamed, msg, queued_elsewhere=blamed is carried)
            ledger.clear(blamed.rid)
            if blamed is carried:
                ledger.abort_pending.add(str(blamed.rid))
        else:
            logger.warning(
                "%s rid=%s need=%d free=%d evictable_paid=%d retries=%d (of %d): the pass is taken back before "
                "its launch -- %s; the pool's wall is read by the next pass (PW memo, PR room vote)",
                MARK, getattr(blamed, "rid", "?"), exc.need, exc.free, exc.evictable_paid, count, retries,
                "continuation parked in place" if carried is not None else
                "fresh request(s) given back to the queue",
            )
        if frame:
            _frame_void(sched, cause="prefill_oom", rid=getattr(blamed, "rid", "?"), need=exc.need, free=exc.free,
                        count=count, abort=(v == ABORT))
        return True
    except Exception:  # noqa: BLE001 -- a retreat that cannot complete must not leave a half state alive
        logger.exception("%s FAILED for rid=%s: the original Prefill-out-of-memory is re-raised",
                         MARK, getattr(blamed, "rid", "?"))
        return False


ENV_LBV_SHORT_STOP = "SGLANG_WEG2_LBV_SHORT_STOP"


def lbv_short_stop_enabled(env=None) -> bool:
    """NF-RANG-AUDIT 1014 (S4): an ``LB-VERDICT SHORT`` is a named group stop, never a retreat. Default ON; ``0`` = the
    retreat of before (PP0 voids the pass while the followers have already spent the verdict and loaded H->D).

    h16r (ARENA-CHAIN on h16h 853534a9ff): this line does NOT carry f99da565a2, so here the switch governs only the chain
    void's released-verdict stop (:func:`retreat_chain_mismatch`); the AgreedRoomShort LB-VERDICT path is unchanged."""
    e = os.environ if env is None else env
    return (e.get(ENV_LBV_SHORT_STOP, "") or "").strip().lower() not in ("0", "false", "no", "off")


ENV_CHAIN = "SGLANG_WEG2_ARENA_CHAIN_VOID"
CHAIN_MARK = "ARENA-CHAIN-VOID"


def chain_void_enabled(env=None) -> bool:
    """ARENA-CHAIN (NF 14.10. 07:29:33Z): a refused load-back chain (``ArenaChainMismatch``) on group P is a void pass, not a
    rank death. Default ON; ``0`` = the raise of before (the rank stops)."""
    e = os.environ if env is None else env
    return (e.get(ENV_CHAIN, "") or "").strip().lower() not in ("0", "false", "no", "off")


def chain_abort_message(rid: Any, exc, count: int, why: str) -> str:
    """The 503 of a request whose load-back chain stayed unprovable after N voids. Carries the intake-stall mark, so the
    front requeues exactly this request for the next phase of this group (as it does for a pool that could not pay)."""
    from sglang.srt.weg2.intake_stall import STALL_MARK

    return (
        f"{STALL_MARK} rid={rid} {CHAIN_MARK} page={getattr(exc, 'page', None)} retries={int(count)} why={why} -- a host "
        f"page of this request's match is not proven against its tokens (its arena slot belongs to another page); the "
        f"pass was voided {int(count)} time(s) and the page stays unprovable; the front requeues exactly this request"
    )


def retreat_chain_mismatch(sched, exc, req, adder) -> Optional[str]:
    """ARENA-CHAIN (NF 14.10. 07:29:33Z, ``..._065207.P.log`` PP0 ``#1424 CHAIN MISMATCH node=460 page=174 of 848 bad=256: key slot
    -1 (state 0)``, rank death): the proof of a load-back chain (``verify_load_chain``, "the LAST latch") refused it. On group P
    nobody cuts the match before it (the ``proof_cut`` vote runs in the TP group D only), so the exception used to leave the
    scheduler uncaught and the rank stopped. It is a load-back refusal of the same shape as ``AgreedRoomShort``: raised before
    any device allocation, the host lock released by ``load_back`` -- so the same retreat takes it: the request is skipped for
    the pass and the pass is voided group-wide (frame form), every stage decides alike (RAENGE-NIE-UNEINS), the N-then-503
    verdict stays PP0's alone. ``None`` = not taken (switch off / out of scope): the caller re-raises, the rank stops as before.

    Port onto the h16 line (h18port 1014): the h18 ``weg2_cut_unproven_chain`` is NOT here -- it cut the node out of THIS
    rank's host tier alone, a rank-local repair that moves this stage's host reach against its peers (NF-RANG-AUDIT S11). The
    page is not cut: the next read re-proves it (#1424 ARENA-REF RECHECK) and the N-then-503 verdict ends a page that stays
    unprovable. And NOT TAKEN while PP0's distributed load-back verdict for this rid is released on this stage (the S4 shape of
    ``lb_verdict_short``): the stages that hold it load back in THIS pass, a void here would leave them split -- named stop."""
    if not chain_void_enabled():
        logger.error("%s NOT TAKEN (switch_off %s=0): ArenaChainMismatch propagates -- rid=%s page=%s", CHAIN_MARK, ENV_CHAIN,
                     getattr(req, "rid", "?"), getattr(exc, "page", None))
        return None
    if lbv_short_stop_enabled() and _lb_verdict_released(sched, req):
        logger.error("%s NOT TAKEN where=chain (lb_verdict_released): PP0's load-back verdict for rid=%s is released on this "
                     "stage (page=%s); the stages that hold it load back in this pass, a void would leave them split -- "
                     "ArenaChainMismatch propagates, the group stops by name (raenge-nie-uneins)", CHAIN_MARK,
                     getattr(req, "rid", "?"), getattr(exc, "page", None))
        return None
    if chain_hold_enabled() and _lb_verdict_held(sched, req):
        return _chain_hold(sched, exc, req, adder)
    return retreat_loadback(sched, exc, req, adder, chain=True)


ENV_CHAIN_HOLD = "SGLANG_WEG2_ARENA_CHAIN_HOLD"
CHAIN_HOLD_MARK = "ARENA-CHAIN-HOLD"


def chain_hold_enabled(env=None) -> bool:
    """h16r (ARENA-CHAIN on h16h 853534a9ff): a refused chain whose rid is gated by an ARMED, NOT yet released LB-VERDICT is
    held like the verdict holds it -- no void frame. Default ON; ``0`` = the void pass of the port (frame form)."""
    e = os.environ if env is None else env
    return (e.get(ENV_CHAIN_HOLD, "") or "").strip().lower() not in ("0", "false", "no", "off")


def _lb_verdict_held(sched, req) -> bool:
    """h16r: is the LB-VERDICT gate armed on this stage while PP0's verdict for ``req`` is NOT released? Then no stage loads
    this host load-back in this pass: the followers' own ``load_back`` holds it at the gate (``_lbv.hold``), and PP0 holds it
    as well until its verdict comes back over its intake (853534a9ff). ``verify_load_chain`` runs BEFORE that gate, so a
    refused chain in that pass meets stages that planned nothing for the rid. Unknown = False (the port's path decides)."""
    try:
        from sglang.srt.weg2 import pp_loadback_verdict as _lbv

        tree = getattr(sched, "tree_cache", None)
        rid = getattr(req, "rid", None)
        return bool(tree is not None and rid is not None and _lbv.armed(tree) and not _lbv.released(tree, rid))
    except Exception:  # noqa: BLE001
        return False


def _chain_hold(sched, exc, req, adder) -> Optional[str]:
    """h16r: ARENA-CHAIN under the h16h LB-VERDICT form. The port's group void puts a VOID frame on the wire -- on the
    carrierless h16h form the followers that HOLD the rid at the verdict gate planned nothing for the slot and never take that
    frame; it stays in their typed inbox and the next row probe turns it into ``#631 ROW AUTHORITY: ... no admission row``
    (the h16h death 14.10. 19:58:35Z; on h16h every void pass so far was such an orphan: role=origin 3, role=relay 0). While the
    verdict is not released nobody loads the rid in this pass anyway, so the refusal is held exactly like the gate holds it:
    the request is skipped (``RETRY`` -> ``NO_TOKEN``), no void statement, no frame, no verdict from PP0 (``_lbv.hold`` is not
    reached). The next pass reads the chain again (#1424 ARENA-REF RECHECK); a chain that stays unprovable ends in PP0's
    N-then-503 over the request wire (ORIGIN-ABORT: every stage drops the rid in the same pass) -- never a void frame."""
    ok, why = applies(sched, adder, getattr(adder, "can_run_list", ()) or ())
    if not ok:
        if why != "not_group_p":
            logger.error("%s NOT TAKEN where=chain_hold (%s): ArenaChainMismatch propagates -- rid=%s page=%s", CHAIN_HOLD_MARK,
                         why, getattr(req, "rid", "?"), getattr(exc, "page", None))
        return None
    try:
        ledger = getattr(sched, "_weg2_prefill_oom_ledger", None)
        if ledger is None:
            ledger = sched._weg2_prefill_oom_ledger = RetreatLedger()
        retries = max_retries()
        count = ledger.note(req.rid)
        v, vwhy = verdict(
            count=count, retries=retries, remaining_need=int(exc.kv_tokens), free=int(exc.avail_after), paid=0,
            pool_tokens=int(getattr(sched, "max_total_num_tokens", 0) or 0), inflight=_inflight(sched),
            elapsed=ledger.elapsed(req.rid), hold=hold_s(), wt_inflight=_wt_inflight(sched), fast_abort=fast_abort_enabled(),
        )
        if _rank(sched) != 0:
            v, vwhy = RETRY, ""  # a follower only holds; the 503 is the origin's verdict alone
        if v != ABORT:
            logger.warning(
                "%s rid=%s page=%s tries=%d (of %d) rank=%d: PP0's load-back verdict for this rid is not released, no stage "
                "loads it in this pass -- held like the LB-VERDICT gate holds it, no void frame; the next pass re-proves "
                "the chain", CHAIN_HOLD_MARK, req.rid, getattr(exc, "page", None), count, retries, _rank(sched))
            return RETRY
        msg = chain_abort_message(req.rid, exc, count, vwhy)
        logger.error("%s rid=%s page=%s tries=%d -> ABORT 503 (%s): %s", CHAIN_HOLD_MARK, req.rid, getattr(exc, "page", None),
                     count, vwhy, msg)
        ledger.aborts += 1
        ledger.clear(req.rid)
        sched.waiting_queue = [q for q in sched.waiting_queue if q is not req]
        _answer_abort(sched, req, msg, queued_elsewhere=False)
        return ABORT
    except Exception:  # noqa: BLE001 -- a hold that cannot complete must not leave a half state alive
        logger.exception("%s FAILED rid=%s: ArenaChainMismatch is re-raised", CHAIN_HOLD_MARK, getattr(req, "rid", "?"))
        return None


def _lb_verdict_released(sched, req) -> bool:
    """h18port: is PP0's distributed LB-VERDICT for ``req`` live on this stage (gated and released, not yet consumed)?
    ``verify_load_chain`` runs before the verdict gate in ``load_back``, so a refused chain can meet a released verdict."""
    try:
        from sglang.srt.weg2 import pp_loadback_verdict as _lbv

        tree = getattr(sched, "tree_cache", None)
        rid = getattr(req, "rid", None)
        # h16r: this line has no LBV-ROW-YIELD (``gated``, 31aa655fff); the gate of h16h's load_back is ``armed(tree)``
        return bool(tree is not None and rid is not None and _lbv.armed(tree) and _lbv.released(tree, rid))
    except Exception:  # noqa: BLE001 -- unknown = assume released: the stop is the safe direction
        return True


def retreat_loadback(sched, exc, req, adder, chain: bool = False) -> Optional[str]:
    """The load-back entrance of the same wall. ``exc`` is the ``AgreedRoomShort`` that ``load_back`` raised for ``req``
    (its two lock refs are already unwound); with ``chain`` the ``ArenaChainMismatch`` of ``load_back`` (the host lock is
    unwound, the device lock was never taken: :func:`retreat_chain_mismatch`). Returns ``None`` = not taken (out of scope / failed: the caller re-raises and
    the rank stops as before), ``RETRY`` = the request is skipped for this pass and stays queued, ``ABORT`` = the request
    is answered 503 (``WEG2-INTAKE-STALL``) and dropped from PP0's queue, its drop relayed to every stage.

    PP0 only, on the form where the followers execute PP0's row (``applies``): the row of this pass names nothing of the
    request, so no stage loads it back -- the stages discard the load-back TOGETHER, none refuses alone (#1004)."""
    ok, why = applies(sched, adder, getattr(adder, "can_run_list", ()) or ())
    if not ok:
        if why != "not_group_p":
            logger.error("%s NOT TAKEN where=loadback (%s): %s propagates -- rid=%s need=%d free=%d "
                         "evictable=%d evicted=%d", MARK, why, type(exc).__name__, getattr(req, "rid", "?"), exc.kv_tokens,
                         exc.avail_after, exc.evictable, exc.evicted)
        return None
    try:
        ledger = getattr(sched, "_weg2_prefill_oom_ledger", None)
        if ledger is None:
            ledger = sched._weg2_prefill_oom_ledger = RetreatLedger()
        retries = max_retries()
        count = ledger.note(req.rid)
        pool_tokens = int(getattr(sched, "max_total_num_tokens", 0) or 0)
        v, vwhy = verdict(
            count=count, retries=retries, remaining_need=int(exc.kv_tokens), free=int(exc.avail_after), paid=0,
            pool_tokens=pool_tokens, inflight=_inflight(sched), elapsed=ledger.elapsed(req.rid), hold=hold_s(),
            wt_inflight=_wt_inflight(sched), fast_abort=fast_abort_enabled(),
        )
        terms = types.SimpleNamespace(need=exc.kv_tokens, free=exc.avail_after, evictable_reported=exc.evictable,
                                      evictable_paid=exc.evicted)
        if _rank(sched) != 0:
            v, vwhy = RETRY, ""  # a follower only unwinds; the 503 is the origin's verdict alone
        if frame_form(sched):
            # FRAME form: the followers plan from their own queue and would admit and load back what PP0 skips -- the WHOLE
            # pass is discarded after the plan (``pp_void_pass.after_plan``), not only this request
            from sglang.srt.weg2 import pp_void_pass as _vp

            setattr(sched, _vp.AFTER_PLAN_ATTR, _vp.make_statement(
                sched, cause=_vp.CAUSE_CHAIN if chain else _vp.CAUSE_LOADBACK, rid=req.rid, need=exc.kv_tokens,
                free=exc.avail_after, tries=count, abort=(v == ABORT)))
        if v != ABORT:
            logger.warning(
                "%s rid=%s need=%d free=%d evictable_paid=%d retries=%d (of %d) where=loadback agreed=%d "
                "evictable_reported=%d write_through_in_flight=%s: the load-back of this pass is left undone by every stage "
                "(the row names nothing of it), the request stays queued; the wall is read by the next pass",
                MARK, req.rid, exc.kv_tokens, exc.avail_after, exc.evicted, count, retries, exc.agreed, exc.evictable,
                _wt_inflight(sched))
            return RETRY
        if chain:
            msg = chain_abort_message(req.rid, exc, count, vwhy)
        else:
            msg = abort_message(req.rid, terms, count, vwhy,
                                too_large_pool=pool_tokens if (pool_tokens > 0 and exc.kv_tokens > pool_tokens) else 0)
        logger.error("%s rid=%s need=%d free=%d evictable_paid=%d retries=%d where=loadback -> ABORT 503 (%s): %s",
                     MARK, req.rid, exc.kv_tokens, exc.avail_after, exc.evicted, count, vwhy, msg)
        ledger.aborts += 1
        ledger.clear(req.rid)
        sched.waiting_queue = [q for q in sched.waiting_queue if q is not req]
        # answers the tokenizer once and puts the drop on the request wire (every stage drops the rid in the same pass)
        _answer_abort(sched, req, msg, queued_elsewhere=False)
        return ABORT
    except Exception:  # noqa: BLE001 -- a retreat that cannot complete must not leave a half state alive
        logger.exception("%s FAILED where=loadback rid=%s: %s is re-raised", MARK, getattr(req, "rid", "?"), type(exc).__name__)
        try:
            from sglang.srt.weg2 import pp_void_pass as _vp

            setattr(sched, _vp.AFTER_PLAN_ATTR, None)  # the pass is not discarded: the rank stops as before
        except Exception:  # noqa: BLE001
            pass
        return None


def _release_for_abort(sched, req) -> None:
    """The blamed FRESH request that is aborted, not requeued: same release, no queue."""
    from sglang.srt.mem_cache.common import release_kv_cache

    release_kv_cache(req, sched.tree_cache, is_insert=False)
    req.reset_for_retract()
