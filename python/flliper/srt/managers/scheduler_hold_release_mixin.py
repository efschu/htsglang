"""``Scheduler`` methods moved out of ``scheduler.py`` (refactor, behaviour-neutral): bodies are byte-identical,
the class stays reachable as ``Scheduler.<name>`` through the mixin. Logger name unchanged."""

from flliper.srt.managers import discard_named as _discard_named
from flliper.srt.managers import pdflip_store_told
from flliper.srt.managers.io_struct import AbortReq
from flliper.srt.pdflip import short_read as _pdflip_short_read
from flliper.srt.pdflip.abort_match import rid_hit as _abort_rid_hit
import logging
import os
import time

from flliper.srt.managers.scheduler_pdflip_helpers import (  # noqa: F401
    _pdflip_pp_producer_awake,
    _pdflip_producer_wake_verdict,
)

logger = logging.getLogger("flliper.srt.managers.scheduler")


class SchedulerHoldReleaseMixin:
    """Methods moved out of ``Scheduler`` (see module docstring)."""

    def _pdflip_abort_dormant_hold(self, recv_req) -> int:
        """#1445: an abort reaches the #1443 dormant hold exactly as it reaches
        ``waiting_queue`` -- prefix match or abort_all, the hicache reservation
        released, the tokenizer told.  Before this, an abort landing while the
        request was HELD found nothing, and the wake released a request whose
        sender had already given up (boot weg2xsn205, rid HEALTH_CHECK).
        Held requests own no device rows and no mamba slot (the hold sits
        before alloc), so nothing else is released.  Returns the count."""
        hold = getattr(self, "pdflip_dormant_hold", None)
        if not hold:
            return 0
        rid = str(getattr(recv_req, "rid", "") or "")
        abort_all = bool(getattr(recv_req, "abort_all", False))
        keep, gone = [], []
        for req in hold:
            if abort_all or _abort_rid_hit(req.rid, rid):
                gone.append(req)
            else:
                keep.append(req)
        if not gone:
            return 0
        hold[:] = keep
        from flliper.srt.pdflip import d_park_draft as _dpd  # H91d: the park's draft buffer

        _dpd.drop_all(self, gone, "abort")
        try:
            _cc = self.tree_cache.cache_controller
            for _r in gone:
                getattr(_cc, "pdflip_hold_rids", set()).discard(_r.rid)  # #1461
        except Exception:  # noqa: BLE001
            pass
        for req in gone:
            if getattr(self, "enable_hicache_storage", False):
                self.tree_cache.release_aborted_request(req.rid)
            self.ipc_channels.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
        logger.info("#1445 DORMANT-HOLD abort: %d held request(s) dropped (rid=%s abort_all=%s), %d still held",
                    len(gone), rid, abort_all, len(keep))
        return len(gone)

    def _pdflip_refetch_one(self, req, now: float, allow_reissue: bool = True) -> str:
        """#1456/#1471: one held request's read -- "reading" (still in
        flight), "complete" (whole prefix on the host), "wait" (short, but
        re-issued less than 2 s ago), "reissued" (short, re-read now)."""
        if _pdflip_pp_producer_awake(self):
            # TK: group P after the wake is the PRODUCER -- never parked.
            return _pdflip_producer_wake_verdict(self, req)
        if getattr(req, "_pdflip_248_read_at_wake", False):
            # #248: looked up only -- no read in the sleep, nothing to top up;
            # the wake issues it (pdflip.park_l3.issue_deferred_reads)
            return "complete"
        from flliper.srt.pdflip import settle_writer as _sw

        if _sw.budget_pending(req):
            # NW (30.09., y3u pdflip-30-52/31-53): the last read was REFUSED by the
            # host budget -- nothing registered, nothing read. Not short: unread.
            return self._pdflip_refetch_budget(req, now, allow_reissue)
        if not self.tree_cache.check_prefetch_progress(req.rid):
            return "reading"
        reason = self._pdflip_note_store_shortfall(req)
        if reason is None:
            # #1471c (weg2xsn238): the re-read issued at the hold finishes its
            # DEVICE half only after the wake (#1455: host read in the sleep,
            # device load at the wake; xsn238: HOLD-REFETCH 13:28:48, wake
            # 13:29:44, PREFETCH-COMPLETE 97870 at 13:29:46), and at the wake
            # the shortfall verdict answers None while the record still holds
            # the short read (4095).  "complete" therefore requires the record
            # to cover the span the hold registered; a request once seen short
            # stays "reading" until it does (bounded by the settle tick).
            _records = getattr(getattr(self, "tree_cache", None), "prefetch_loaded_tokens_by_reqid", None) or {}
            _span = int(getattr(req, "_prefetch_span_tokens", 0) or 0)
            _have = _records.get(str(req.rid))
            # #1478 (weg2xsn241, rid pdflip-6-2, 4316 tokens): the FIRST hold read
            # answered NOTHING (store held 64 leading KV pages, no Mamba anchor
            # in range yet -> the component pool cut the claim to 0, #1035c),
            # and a zero answer is recorded with deliverable=0, so it is not
            # "incomplete" and raised no #1324 shortfall: the request was
            # "complete" with nothing on the host, and at the wake the anchor
            # refusal (#928) priced the whole prompt -> W31.  A record that
            # materialized nothing for a span > 0 is short by construction.
            _zero = (_have is not None and _span > 0
                     and int(getattr(_have, "materialized", _have) or 0) == 0)
            if _zero:
                req._1471_short = True
            if getattr(req, "_1471_short", False) and (_have is None or (_span > 0 and int(_have) < _span)):
                # #1479 (weg2xsn241, rid pdflip-6-3): the re-read issued at the hold
                # (14:08:14) probed the store BEFORE P's write-through of that
                # request had landed (WT-ACK 14:08:24), found 0 of 93,775 pages
                # and was REVOKED below the prefetch threshold -- no operation
                # in flight, record still 4095, and this branch answered
                # "reading" for 43 s (until the settle bound lapsed -> W31 ->
                # a second prefill).  Nothing in flight + a short record is a
                # shortfall to re-issue, not a read to wait for.
                _ongoing = getattr(getattr(self, "tree_cache", None), "ongoing_prefetch", None)
                if isinstance(_ongoing, dict) and (req.rid in _ongoing or str(req.rid) in _ongoing):
                    return "reading"
                reason = "zero-answer" if _zero else "record-short"
            else:
                req._1471_short = False
                return "complete"
        # fnFL2x38 (23.09.): A TAIL SHORTER THAN ONE STORE READ IS RECOMPUTED,
        # NOT RE-READ. pdflip-0-1 (8815 tokens): the store held 8704, P never
        # published the last full page (deliverable 8768), every 2-s re-read
        # answered ZERO, and the request sat the whole 20-s settle bound out
        # before the lapse queued it "as it is" -- whereupon the extend
        # computed the 111 tokens in one pass. After ONE re-read has answered
        # short, a tail within the store's own prefetch threshold goes to the
        # extend now; the flip time is the reader's, not the store's.
        _rec2 = getattr(getattr(self, "tree_cache", None), "prefetch_loaded_tokens_by_reqid", None) or {}
        _span2 = int(getattr(req, "_prefetch_span_tokens", 0) or 0)
        _have2 = _rec2.get(str(req.rid))
        _have2 = int(getattr(_have2, "materialized", _have2) or 0) if _have2 is not None else 0
        _tail = _span2 - _have2
        if 0 < _tail <= self.PDFLIP_TAIL_RECOMPUTE_TOKENS and int(getattr(req, "_1456_n", 0) or 0) >= 1:
            req._1471_short = False
            logger.info("#x38 SETTLE-TAIL rid=%s tail=%d tokens <= %d after %d re-read(s) (%s): "
                        "released -- D-NORECOMPUTE sends it through RESUME-VIA-P (no D compute "
                        "after a flip)",
                        str(req.rid), _tail, self.PDFLIP_TAIL_RECOMPUTE_TOKENS,
                        int(getattr(req, "_1456_n", 0) or 0), reason)
            return "complete"
        # TK (#1400 told form, tp_size 1 = no group MIN): a follower re-reads
        # only PP0's told span, never before the told, and PP0 never after
        # its told is on the wire (pdflip_store_told.refetch_plan).
        _told_plan = pdflip_store_told.refetch_plan(self, req)
        if _told_plan == pdflip_store_told.REFETCH_SKIP:
            req._1471_short = False
            return "complete"
        req._1471_short = True
        if (now - float(getattr(req, "_1456_last", 0.0) or 0.0) < 2.0
                and not _pdflip_short_read.reread_now(req, _have2)):
            # a read reaped short with its span in the store is re-read at the
            # next tick while it moves (pdflip/short_read.py); else the 2 s timer
            return "wait"
        if not allow_reissue:
            # weg2xsn296 (Task #9, xsn287 class): the re-read is a TP
            # collective (prefetch_from_storage -> _all_reduce_attn_groups)
            # and this 2 s timer is rank-local -- TP0 re-issued alone while
            # TP1/TP2 sat in the next pass's request broadcast: D stood for
            # the rest of the boot (stacks stall_weg2xsn296_ep7_*). The
            # caller now asks every rank "due?" first, takes the group MIN,
            # and only then re-issues on every rank in the same pass.
            return "due"
        req._1456_last = now
        req._1456_n = int(getattr(req, "_1456_n", 0) or 0) + 1
        _pdflip_short_read.note_reissue(req, _have2)
        # #221 Fix A (site hold_refetch_clear): the shortfall mark is ours to re-issue, not the
        # drain's -- cleared BY NAME with its census exit (defer_cleared_refetch), not silently
        _discard_named.discard_mark(
            req, _discard_named.SITE_HOLD_REFETCH_CLEAR, logger,
            getattr(self, "_clear_prefetch_deferral_fields", None), via="refetch")
        if _told_plan is None:
            verdict = self._prefetch_kvcache(req)
        else:
            verdict = self._prefetch_kvcache(req, limit_tokens=int(_told_plan))
        if req._1456_n <= 4 or req._1456_n % 16 == 0:
            logger.info("#1456 HOLD-REFETCH rid=%s n=%d reason=%s verdict=%s (the store was short; "
                        "re-read from the registered extent)", str(req.rid), req._1456_n, reason, verdict)
        if _sw.note_read_verdict(req, verdict, now, tree=self.tree_cache):
            return "budget"  # NW: refused, not issued -- never a read in flight
        return "reissued"

    def _pdflip_refetch_budget(self, req, now: float, allow_reissue: bool) -> str:
        """NW (30.09.): one settle/hold step of a request whose last store read
        the host budget REFUSED (pdflip/settle_writer.py): "wait" while this
        rank's budget is still full or the retry interval runs, "due" when it
        may be re-read (the caller takes the group MIN -- the read is a
        collective), then the re-read: "reissued", or "budget" when refused
        again. Never "complete": nothing of this read reached the host."""
        from flliper.srt.pdflip import settle_writer as _sw

        req._1471_short = True
        try:
            limited = bool(self.tree_cache.cache_controller.prefetch_rate_limited())
        except Exception:  # noqa: BLE001 - no budget readable: retry on the interval
            limited = False
        if not _sw.budget_retry_due(req, now, limited):
            return "wait"
        if not allow_reissue:
            return "due"
        req._1456_last = now
        req._1456_n = int(getattr(req, "_1456_n", 0) or 0) + 1
        _discard_named.discard_mark(  # #221 Fix A (site hold_refetch_clear), the budget re-read
            req, _discard_named.SITE_HOLD_REFETCH_CLEAR, logger,
            getattr(self, "_clear_prefetch_deferral_fields", None), via="budget")
        plan = pdflip_store_told.refetch_plan(self, req)
        if plan is None or plan == pdflip_store_told.REFETCH_SKIP:
            verdict = self._prefetch_kvcache(req)
        else:
            verdict = self._prefetch_kvcache(req, limit_tokens=int(plan))
        refused = _sw.note_read_verdict(req, verdict, now, tree=self.tree_cache)
        n = int(getattr(self, "_1471b_n", 0) or 0) + 1
        self._1471b_n = n
        if n <= 16 or n % 64 == 0:
            logger.info("#1471b BUDGET-REREAD rid=%s n=%d verdict=%s refused=%s (n_all=%d): the read the "
                        "host budget refused is re-issued now that it has room", str(req.rid),
                        req._1456_n, verdict, refused, n)
        return "budget" if refused else "reissued"

    def _pdflip_run_deferred_hold_release(self) -> int:
        """HOLD-RELEASE-AFTER-REPLY (30.09., z30y8 epoch 26): the #1443 release the kv resume marked
        due (weight_updater, FLLIPER_PDFLIP_HOLD_RELEASE_AFTER_REPLY). Measured: inside the RPC it held
        the P->D flip 15 s (wake=16638 ms) -- TP0 issued the hold reads' store work while TP1/TP2
        waited in its hicache collective. Here the flip is already done. Runs on every rank at the
        same point (see the call site); watched by the RPC-STALL-WATCHDOG as kind ``hold_release``.
        Exceptions propagate as they did out of the resume (a partial release on one rank is the
        disagreement the group stop exists for)."""
        self._pdflip_hold_release_due = False
        from flliper.srt.pdflip import rpc_stall_watchdog as _rsw

        armed = _rsw.arm("hold_release", rank=_rsw.rank_of(self),
                         group=os.environ.get("FLLIPER_PDFLIP_GROUP", "") or "?")
        t0 = time.perf_counter()
        n_hold = len(getattr(self, "pdflip_dormant_hold", None) or [])
        try:
            n = self._pdflip_release_dormant_hold()
        finally:
            _rsw.disarm(armed)
        logger.info("PDFLIP-HOLD-RELEASE after-reply held=%d released=%s ms=%.0f wake_seq=%s (the #1443 release "
                    "outside the kv resume RPC: the flip was done before it began)", n_hold, n,
                    (time.perf_counter() - t0) * 1000.0, getattr(self, "_pdflip_wake_seq", None))
        return n

    def _pdflip_join_store_writes_before_reset(self) -> None:
        """#1470b: delegate the bounded join to the tree (same switch as #1470)."""
        if not self.enable_hierarchical_cache or os.environ.get("FLLIPER_HICACHE_FLUSH_PUBLISH_SWEEP", "1") == "0":
            return
        join = getattr(self.tree_cache, "join_storage_backups", None)
        if join is None:
            return
        from flliper.srt.managers.cache_controller import STORAGE_THREAD_JOIN_BOUND_S

        drained, left, waited_ms = join(STORAGE_THREAD_JOIN_BOUND_S)
        if drained or left:
            (logger.warning if left else logger.info)(
                "#1470b FLUSH-STORE-JOIN drained=%d left=%d waited_ms=%.0f (store writes the "
                "write-through acks had queued behind the backup thread's current one, handed to "
                "it BEFORE the reset rebuilds the queue; left>0 = the bound or a dead backup "
                "thread, those pages miss in the store)",
                drained, left, waited_ms)
