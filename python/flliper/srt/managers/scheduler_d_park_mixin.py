"""``Scheduler`` methods moved out of ``scheduler.py`` (refactor, behaviour-neutral): bodies are byte-identical,
the class stays reachable as ``Scheduler.<name>`` through the mixin. Logger name unchanged."""

from flliper.srt.managers.io_struct import AbortReq
from flliper.srt.managers.schedule_batch import ScheduleBatch
from flliper.srt.pdflip import d_head_hold as _pdflip_d_head_hold
from flliper.srt.pdflip.abort_match import rid_hit as _abort_rid_hit
from typing import Optional
import logging

logger = logging.getLogger("flliper.srt.managers.scheduler")


class SchedulerDParkMixin:
    """Methods moved out of ``Scheduler`` (see module docstring)."""

    def _pdflip_d_head_fingerprint(self, running_batch, prefetch_verdicts) -> Optional[tuple]:
        """DQH: what the held head's verdict reads (replicated values).
        ``prefetch_verdicts`` is this iteration's drain memo as the pass found
        it (the pass consumes it, so the caller reads it before the pass)."""
        try:
            from flliper.srt.mem_cache.hicache_phase_binding import current_generation

            gen = int(current_generation())
        except Exception:  # noqa: BLE001 - no binding: a constant term, the rest decides
            gen = None
        return _pdflip_d_head_hold.fingerprint(
            hol_state=self.__dict__.get("_pdflip_hol"),
            waiting=self.waiting_queue,
            running=running_batch.reqs if running_batch is not None else (),
            prefetch_verdicts=prefetch_verdicts,
            continuation=self.chunked_req is not None or bool(getattr(self, "anchor_tails", None)),
            generation=gen,
        )

    def _pdflip_d_park_tick(self) -> int:
        from flliper.srt.pdflip import d_park_runtime

        return d_park_runtime.park_tick(self)

    def _pdflip_d_lane_hold_queued(self) -> int:
        """PRIORITY LANES fix 2: the queued requests below D's lane floor enter the lane hold
        (d_park_runtime.hold_queued); every rank moves the same ones (queue and floor are replicated)."""
        from flliper.srt.pdflip import d_park_runtime

        return len(d_park_runtime.hold_queued(self))

    def _pdflip_d_park_note_retracted(self, retracted_reqs) -> int:
        from flliper.srt.pdflip import d_park_runtime

        return d_park_runtime.note_retracted(self, retracted_reqs)

    def _pdflip_d_park_abort(self, recv_req) -> int:
        from flliper.srt.pdflip import d_park_runtime

        return d_park_runtime.park_abort(self, recv_req)

    def _pdflip_d_seat_guard(self, batch) -> None:
        from flliper.srt.pdflip import d_park_runtime

        d_park_runtime.seat_guard(self, batch)

    # H91d: a parked request's MTP draft rows (pdflip/d_park_draft.py).
    def _pdflip_d_park_draft_snapshot(self, batch):
        from flliper.srt.pdflip import d_park_draft

        return d_park_draft.snapshot(self, batch)

    def _pdflip_d_park_draft_save(self, retracted_reqs, snap) -> int:
        from flliper.srt.pdflip import d_park_draft

        return d_park_draft.save_retracted(self, retracted_reqs, snap)

    def _pdflip_d_park_draft_restore(self, batch) -> int:
        from flliper.srt.pdflip import d_park_draft

        return d_park_draft.restore_admitted(self, batch)

    def _pdflip_defer_waiting_abort(self, recv_req) -> bool:
        """#1180-W: True = this follower holds the abort of a waiting-queue
        request until PP0's forwarded schedule decides it. Only a PP follower
        whose group is not pass-aligned (row authority True / unknown); never
        for abort_all; never twice for a rid this method itself releases."""
        ps = getattr(self, "ps", None)
        if int(getattr(ps, "pp_size", 1) or 1) <= 1 or int(getattr(ps, "pp_rank", 0) or 0) <= 0:
            return False
        if getattr(recv_req, "abort_all", False):
            return False
        rid = str(getattr(recv_req, "rid", "") or "")
        force = self.__dict__.setdefault("_pdflip_force_waiting_abort", set())
        if not rid or rid in force:
            force.discard(rid)
            return False
        from flliper.srt.pdflip.pp_abort import row_authority_of

        if row_authority_of(self) is False:
            return False  # pass-aligned request wire: applied at receipt (#791C-NF)
        held = [r for r in self.waiting_queue if _abort_rid_hit(r.rid, rid)]
        if not held:
            return False
        # #DPJ-W (24.10., 27B NVFP4 dual rc39a b9cddabda1, P log ..._1024_085659):
        # PP0 stamped this abort pp0_admitted=False (K14: no frame of PP0 names
        # it, PP0 popped it at receipt -- 'PDFLIP-PP-ABORT-VERDICT rid=pdflip-0-28
        # pp0_admitted=False ... every stage applies it at receipt'), yet a told of
        # an EARLIER list stood here (Q-693/Q-697 did not apply), so this #1180-W
        # hold kept the instance: PP1 popped it on misses=3, PP2 (idle, no frames)
        # did not before the front's RESUME sent the next instance under the SAME
        # rid -- PP0's frames then named the rid for the new instance ('keep' for
        # ever), the zombie stayed queued, and the adder minted pdflip-0-28 twice
        # in one pass ('#996 SECOND CHUNKED MINT', PP2 09:09:46, group death).
        # PP0's stamped verdict is the group's decision (SCHEDULER-SOLL v2 R11):
        # an explicit False is applied at receipt here as well. Unstamped (None)
        # keeps the #1180-W hold byte for byte.
        if getattr(recv_req, "pdflip_pp0_admitted", None) is False:
            logger.info(
                "PDFLIP-PP-WAITING-ABORT at-receipt rid=%s pp_rank=%s queued=%d: PP0 stamped "
                "pp0_admitted=False (no frame names it, PP0 popped it at receipt) -- applied "
                "here at receipt too, no #1180-W hold (#DPJ-W, K14)",
                rid, getattr(ps, "pp_rank", 0), len(held))
            return False
        # Q-693 (dual P only): no told of it ever reached this follower, so PP0
        # had not admitted it -- applied at receipt like PP0 does (y8y PP1 zombie);
        # Q-697: nor did a told that rode the SAME PP0 list as this abort (y9 PP1)
        from flliper.srt.pdflip import dual_untold_abort as _q693

        if _q693.applies_at_receipt(self, held, recv_req=recv_req):
            return False
        pend = self.__dict__.setdefault("_pdflip_pending_waiting_aborts", {})
        for r in held:
            pend[str(r.rid)] = [recv_req, 0]
            logger.info(
                "PDFLIP-PP-WAITING-ABORT held rid=%s pp_rank=%s: still in this follower's "
                "waiting queue -- applied when PP0's forwarded schedule decides it (#1180-W)",
                r.rid, getattr(ps, "pp_rank", 0))
        _q693.note_hold(self, recv_req, held)  # #1430q: records the held objects; a no-op unless the env is on
        return True

    def _pdflip_process_waiting_aborts(self) -> None:
        """#1180-W: once per scheduling pass, before the plan."""
        pend = getattr(self, "_pdflip_pending_waiting_aborts", None)
        if not pend:
            return
        from flliper.srt.pdflip.pp_abort import follower_waiting_abort_verdict
        from flliper.srt.pdflip import dual_untold_abort as _q693_1470

        try:
            sched = self._pp_scheduled_extents()
        except Exception:  # noqa: BLE001 -- no frame readable = no frame this pass
            sched = None
        drained = bool(getattr(self, "_791c_pp0_drained", False))
        laps = int(getattr(self.ps, "pp_size", 1) or 1)
        for rid, ent in list(pend.items()):
            recv_req, misses = ent
            ch = self.chunked_req
            admitted = ch is not None and str(ch.rid) == rid
            in_wait = any(str(r.rid) == rid for r in self.waiting_queue)
            verdict, misses = follower_waiting_abort_verdict(
                rid, sched, misses, laps=laps, pp0_drained=drained,
                admitted=admitted, in_waiting=in_wait)
            if verdict == "keep":
                ent[1] = misses
                continue
            del pend[rid]
            logger.info("PDFLIP-PP-WAITING-ABORT %s rid=%s pp_rank=%s misses=%d (#1180-W)",
                        verdict, rid, getattr(self.ps, "pp_rank", 0), misses)
            # #1470 (dual P follower, env default OFF -> False at once): a 'pop' with a newer instance of the
            # rid queued removes only the held objects; the hold record is discarded for every verdict
            if _q693_1470.settle_hold(self, rid, verdict):
                continue
            if verdict in ("chunked", "pop"):
                self._pdflip_force_waiting_abort.add(rid)
                self._abort_request_now(AbortReq(rid=rid, finished_reason=getattr(
                    recv_req, "finished_reason", None)))

    def _pdflip_d_ext_arm_done(self, batch: ScheduleBatch) -> None:
        """NF-R6-REST-EXAKT (SOLL R6 + R11, user order 19.10. "NICHTS SCHÄTZEN, EXAKTE ZAHLEN"): a device event behind this
        extend forward; ``_pdflip_d_ext_poll`` writes the batch's after-rest (D-EXTEND-BEACON v3) as soon as it has
        completed -- not only from ``process_batch_result``, which under overlap runs after the NEXT batch was built and
        launched (rc28 pdflip-12-275: the front read the chunk-start rest 1394 at 18:36:32.98, 50 ms after chunk 1's
        forward had ended; D's exact rest was 626). Never a sync: the event is only queried."""
        try:
            ev = self.device_module.Event()
            if self.enable_overlap:
                ev.record(self.forward_stream)
            else:
                ev.record()
            self.pdflip_d_ext_beacon.arm_done(
                key=int(batch.forward_iter), rids=[req.rid for req in batch.reqs], ready=ev.query
            )
        except Exception:  # noqa: BLE001 -- no event: the result path writes the after-rest as before
            pass

    def _pdflip_d_ext_commit(self) -> None:
        """NF-RC30-R6STALE review A1 (SOLL R6/R7, R11; user order 19.10. "NICHTS SCHÄTZEN, EXAKTE ZAHLEN"): called at the END of
        the recv step (aborts applied). A running chunked prefill (``chunked_req``) is continued by this iteration's batch:
        its next chunk is fixed now, before a probe answer sent in this recv step can lead the front to a verdict. The beacon
        marks it COMMITTED until that chunk's start record. Rank-local, no collective, no sync; the front decides."""
        b = self.pdflip_d_ext_beacon
        if b is None:
            return
        try:
            cr = getattr(self, "chunked_req", None)
            rid = getattr(cr, "rid", None) if cr is not None else None
            if rid:
                b.commit_next(rids=[rid])
        except Exception:  # noqa: BLE001 -- an instrument never stops the scheduler
            pass

    def _pdflip_d_ext_resolve(self, batch) -> None:
        """NF-RC30-R6STALE review A1: the batch is built -- a committed rid it does not launch (held, parked, aborted) gets
        its real free back, so the marker never outlives its iteration."""
        b = self.pdflip_d_ext_beacon
        if b is None:
            return
        try:
            launched = ([req.rid for req in batch.reqs]
                        if batch is not None and batch.forward_mode.is_extend() else [])
            b.resolve_commit(launched=launched)
        except Exception:  # noqa: BLE001
            pass

    def _pdflip_d_ext_poll(self) -> None:
        """NF-R6-REST-EXAKT: write the after-rest of every extend batch whose forward has ended (non-blocking)."""
        b = self.pdflip_d_ext_beacon
        if b is not None:
            try:
                b.poll_done()
            except Exception:  # noqa: BLE001 -- an instrument never stops the scheduler
                pass

    def _pdflip_rearm_defer_tick(self, batch) -> None:
        """H31b: start / promote the deferred Platztausch extra rows
        (expert_offload.DeferredRowsFill). Called on the stream the forward
        runs on, before it; one attribute read while nothing is pending."""
        from flliper.srt.layers.moe.expert_offload import deferred_rows_tick

        deferred_rows_tick(batch)

    def _pdflip_arm_wake_round_census(self) -> None:
        """fnFL2 H23: the first pass after the wake arms DECODE-ROUND-COST
        (decode_round_log / wake_round_census) for the next decode rounds."""
        drl = self.metrics_reporter.decode_round_log
        if drl is not None:
            drl.arm_wake_census(wake_mono=self._pdflip_last_wake_t)
