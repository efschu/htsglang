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

THE WALL STANDS. A request retreated ``SGLANG_WEG2_PREFILL_OOM_RETRIES`` times in a row (default 3), or one whose REMAINING
need exceeds free + paid with nothing in flight that could free more (the H91a ``intake_verdict`` IMPOSSIBLE), is answered
503 with the ``WEG2-INTAKE-STALL`` mark -- the front requeues exactly that request at the head and flips (its ``is_intake_stall``
test is the mark) -- and the abort rides the request wire (``intake_origin_abort``) so every PP stage drops the rid in the
same pass. The server lives; one request is re-routed.

SCOPE (everything else dies exactly as before, byte for byte): group P, the decision authority (pp_rank 0), a tp group of
one, the local-PP floor (``pp_slot_fidelity.FLOOR_LOCAL_PP_ATTR``), no dynamic layer split, no anchor tails in the pass.
A FOLLOWER (PP1/PP2) that fails the same allocation after PP0 launched cannot take its stage of the pass back -- the
upstream's hidden states are on the wire; it still raises (the typed exception names the terms). Prevention for that case is
the PR room vote and the walk fix (``pp_room_vote.estimate_payable``).

Switch ``SGLANG_WEG2_PREFILL_OOM_RETREAT`` (default on; ``0`` = the raise of before)."""

from __future__ import annotations

import logging
import os
from http import HTTPStatus
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_PREFILL_OOM_RETREAT"
ENV_RETRIES = "SGLANG_WEG2_PREFILL_OOM_RETRIES"
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


class RetreatLedger:
    """Consecutive retreats per rid. A rid that gets through ``prepare_for_extend`` leaves it."""

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        #: rids whose 503 abort is on its way through the request wire (the carried continuation: it is dropped by
        #: the chunked-abort machinery of every stage a few passes from now). Never answered twice, never parked.
        self.abort_pending: set = set()
        self.retreats = 0
        self.aborts = 0

    def note(self, rid: Any) -> int:
        k = str(rid)
        self.counts[k] = self.counts.get(k, 0) + 1
        self.retreats += 1
        return self.counts[k]

    def clear(self, rid: Any) -> None:
        self.counts.pop(str(rid), None)

    def clear_batch(self, reqs: Iterable[Any]) -> None:
        if not self.counts:
            return
        for r in reqs:
            self.counts.pop(str(getattr(r, "rid", "")), None)

    def forget(self, rid: Optional[str]) -> None:
        """An abort ends the rid's history (``None`` = abort_all, prefix semantics as the abort's)."""
        if rid is None:
            self.counts.clear()
            self.abort_pending.clear()
            return
        rid = str(rid)
        for k in [k for k in self.counts if k.startswith(rid)]:
            del self.counts[k]
        self.abort_pending = {k for k in self.abort_pending if not k.startswith(rid)}


def verdict(*, count: int, retries: int, remaining_need: int, free: int, paid: int,
            pool_tokens: int, inflight: bool) -> Tuple[str, str]:
    """(RETRY|ABORT, why). ABORT when the retry budget is spent, or when what the pool can still give (free + paid)
    cannot cover the request's REMAINING need and nothing in flight would free more (the H91a intake verdict)."""
    from sglang.srt.weg2.intake_stall import INTAKE_IMPOSSIBLE, intake_verdict

    if int(count) >= int(retries):
        return ABORT, f"retries={int(count)}>={int(retries)}"
    if int(remaining_need) > 0 and intake_verdict(
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


def applies(sched, adder=None, can_run_list=()) -> Tuple[bool, str]:
    """(True, '') when this seat may retreat; else (False, why) and the caller re-raises."""
    if not enabled():
        return False, "switch_off"
    if str(os.environ.get(GROUP_ENV, "")).strip().upper() != "P":
        return False, "not_group_p"
    ps = getattr(sched, "ps", None)
    if ps is None or int(getattr(ps, "pp_rank", 0) or 0) != 0:
        return False, "not_pp0"
    if int(getattr(ps, "tp_size", 1) or 1) != 1:
        return False, "tp_group"
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


def retreat(sched, exc, new_batch, can_run_list: List[Any], adder) -> bool:
    """Take the pass back. True = retreated (the caller returns ``None, running_batch``); False = out of scope or the
    retreat itself failed -- the caller re-raises ``exc`` and the rank dies as before."""
    ok, why = applies(sched, adder, can_run_list)
    if not ok:
        if why != "not_group_p":
            logger.error("%s NOT TAKEN (%s): the typed Prefill-out-of-memory propagates -- need=%d free=%d "
                         "evictable_paid=%d", MARK, why, exc.need, exc.free, exc.evictable_paid)
        return False
    ledger = getattr(sched, "_weg2_prefill_oom_ledger", None)
    if ledger is None:
        ledger = sched._weg2_prefill_oom_ledger = RetreatLedger()
    chunked = getattr(sched, "chunked_req", None)
    new_chunked = getattr(adder, "new_chunked_req", None)
    carried = chunked if (chunked is not None and new_chunked is None
                          and any(r is chunked for r in can_run_list)) else None
    # the blame: the carried continuation, else the pass's first request (the batch is all-or-nothing)
    blamed = carried if carried is not None else (can_run_list[0] if can_run_list else None)
    if carried is not None and str(carried.rid) in ledger.abort_pending:
        # its 503 is already on the request wire; the chunked abort of every stage drops it a few passes from now.
        # This pass still must not launch a chunk it cannot pay: take it back, answer nothing a second time.
        try:
            _void_decision(sched, can_run_list)
            if int(getattr(carried, "inflight_middle_chunks", 0) or 0) > 0:
                carried.inflight_middle_chunks -= 1
            _park_in_place(carried, flag=False)
            logger.warning("%s rid=%s need=%d free=%d evictable_paid=%d: the pass is taken back, the abort is pending",
                           MARK, carried.rid, exc.need, exc.free, exc.evictable_paid)
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
    )
    try:
        _void_decision(sched, can_run_list)
        if carried is not None:
            if int(getattr(carried, "inflight_middle_chunks", 0) or 0) > 0:
                carried.inflight_middle_chunks -= 1  # counted for a chunk that never launched
            # an aborted continuation is NOT offered to the head-of-step rows-back park: that park would requeue it
            # while the abort of the other stages still has to land (a W27 split of PP0 against the followers)
            _park_in_place(carried, flag=not (v == ABORT and blamed is carried))
            if v == ABORT and blamed is carried:
                carried._weg2_oom_abort_pending = True
        if new_chunked is not None and chunked is new_chunked:
            sched.chunked_req = None  # the fresh request that became the chunked request goes back below
        for req in can_run_list:
            if req is carried:
                continue
            if v == ABORT and req is blamed:
                _release_for_abort(sched, req)
                continue
            _give_back_fresh(sched, req)
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
        return True
    except Exception:  # noqa: BLE001 -- a retreat that cannot complete must not leave a half state alive
        logger.exception("%s FAILED for rid=%s: the original Prefill-out-of-memory is re-raised",
                         MARK, getattr(blamed, "rid", "?"))
        return False


def _release_for_abort(sched, req) -> None:
    """The blamed FRESH request that is aborted, not requeued: same release, no queue."""
    from sglang.srt.mem_cache.common import release_kv_cache

    release_kv_cache(req, sched.tree_cache, is_insert=False)
    req.reset_for_retract()
