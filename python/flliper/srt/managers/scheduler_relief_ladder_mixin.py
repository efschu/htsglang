"""``Scheduler`` methods moved out of ``scheduler.py`` (refactor, behaviour-neutral): bodies are byte-identical,
the class stays reachable as ``Scheduler.<name>`` through the mixin. Logger name unchanged."""

from flliper.srt.managers.admission_limiter import throttle_before_retract
from flliper.srt.managers.io_struct import AbortReq
from flliper.srt.managers.schedule_batch import FINISH_ABORT
from flliper.srt.managers.schedule_batch import ScheduleBatch
from typing import Optional
import logging

logger = logging.getLogger("flliper.srt.managers.scheduler")


class SchedulerReliefLadderMixin:
    """Methods moved out of ``Scheduler`` (see module docstring)."""

    def _maybe_spend_admission_relief(self, running_batch: ScheduleBatch) -> int:
        """Decide, group-uniformly, whether the chunked prefill needs relief.

        #679. Split from the ladder so the TRIGGER and the ACTUATORS can be
        tested apart, and so the trigger's one job is visible: read the agreed
        number, compare it against what the next chunk would ask for, and spend
        nothing at all when the pool is comfortable. The comfortable case is
        every iteration on a healthy instance, so it must cost one comparison.

        THE SHORTFALL IS SIZED FROM THE REDUCED VALUE, which makes every rank
        ask its rungs for the same number of tokens. Sizing it from this rank's
        own availability would have the binding rank retract more victims than
        its peers -- #583, one layer up from where #583 was found.
        """
        from flliper.srt.mem_cache.common import admission_relief_ladder_enabled

        if not admission_relief_ladder_enabled():
            return 0
        req = self.chunked_req
        if req is None:
            return 0
        try:
            want = int(self.server_args.chunked_prefill_size or 0)
            if want <= 0:
                return 0
            avail = int(self.uniform_min_avail())
            if avail >= want:
                # The common case, and it must stay cheap: one reduced read,
                # one comparison, no rung entered.
                return 0
            return self._admission_relief_ladder(running_batch, want - avail)
        except Exception as e:  # noqa: BLE001 - relief must never fail a boot
            logger.warning(
                "%s trigger failed (%s); admission proceeds and the park guard "
                "decides unaided",
                self._LADDER_PREFIX,
                e,
            )
            return 0

    def _admission_relief_ladder(
        self, running_batch: ScheduleBatch, need_tokens: int
    ) -> int:
        """Spend relief so an admission need not PARK. Returns tokens freed.

        #679 rung 1-3, built to DESIGN_679_admission_relief_ladder.md. The park
        guard remains the floor of this ladder and the final authority: this
        function only changes how much the pool can fund BEFORE that guard
        decides. It never admits anything and never refuses anything.

        THE ORDER, and why it is this order (see the design note for the
        costing of each rung):

          rung 0  radix eviction -- already spent by the caller, and by
                  alloc_token_slots after it. Not repeated here.
          rung 1  kvso.try_spill -- a BOUNDED, CHOSEN amount (the victim's
                  block-aligned tail overhang) at the cost of host bandwidth
                  and no request's progress. Best rung available.
          rung 2  throttle_before_retract -- frees NOTHING now; lowers inflow
                  so rung 3 does not repeat next round. Placed between the two
                  for exactly the reason the decode-OOM branch places it there.
          rung 3  retract_decode -- the most tokens per call and the loudest:
                  the victim loses all decode progress and re-prefills.

        EXHAUSTION IS A RUNG OUTCOME, NEVER AN ERROR. ``try_spill`` returns
        False when no host region is free -- a reachable state whose bound has
        never been measured under the 5-lane load that produced the crash -- and
        the ladder simply falls through to the next rung. Same for a rung that
        frees less than asked: the ladder continues, the caller re-reads the
        pool, and the park guard has the last word. Nothing here raises.

        GROUP UNIFORMITY. Every decision reads ``uniform_min_avail()``, the
        value the pre-branch reduce published at the top of this iteration
        (scheduler.py's _update_uniform_pool_budget, unconditional and once per
        rank), so no rung takes a collective of its own and no rung can split
        the group. ``need`` is sized from that same reduced value, so every
        rank spills and retracts for the same shortfall. This is the property
        #603 and #583 were paid for; it is not re-derived here, it is reused.

        OFF BY DEFAULT. Without the env flag this returns 0 immediately and the
        caller behaves exactly as c4b88e1923 did.
        """
        from flliper.srt.mem_cache.common import (
            admission_relief_ladder_enabled,
            admission_retraction_enabled,
        )

        if not admission_relief_ladder_enabled():
            return 0
        if running_batch is None or running_batch.is_empty():
            # Nothing to spill and nothing to retract: every rung below acts on
            # the RUNNING batch. An empty one means the pressure is not coming
            # from resident work, so there is nothing this ladder can take.
            return 0

        need = max(0, int(need_tokens))
        if need <= 0:
            return 0

        before = int(self.uniform_min_avail())
        freed_by = []

        # -- rung 1: spill a session's tail to host ---------------------------
        if self.kv_session_offload is not None:
            try:
                if self.kv_session_offload.try_spill(running_batch, need=need):
                    freed_by.append("kvso_spill")
            except Exception as e:  # noqa: BLE001 - a rung must not kill a boot
                logger.warning(
                    "%s rung 1 (kvso spill) failed: %s", self._LADDER_PREFIX, e
                )
            if int(self.uniform_min_avail()) - before >= need:
                self._log_ladder(need, before, freed_by)
                return int(self.uniform_min_avail()) - before

        # -- rung 2: lower inflow so rung 3 does not repeat -------------------
        # Frees nothing. Deliberately spent even when rung 3 is skipped below:
        # the pressure that got us here is inflow, and the throttle is the only
        # rung that addresses that rather than its symptom.
        try:
            if throttle_before_retract(
                self.admission_limiter, running_batch.batch_size()
            ):
                freed_by.append("throttle")
        except Exception as e:  # noqa: BLE001
            logger.warning("%s rung 2 (throttle) failed: %s", self._LADDER_PREFIX, e)

        # -- rung 3: retract decode victims -----------------------------------
        # THE PRECONDITION IS NOT OPTIONAL. retract_decode's loop bound and its
        # last-survivor test both read uniform_avail_floor; handing it a
        # rank-local value is #583 exactly -- ranks enter together and pop
        # DIFFERENT numbers of victims.
        if admission_retraction_enabled():
            try:
                running_batch.uniform_avail_floor = self.uniform_min_avail()
                gained = self._retract_decode_and_requeue(
                    running_batch, kv_full_retract_flag=True
                )
                if gained:
                    freed_by.append(f"retract({gained})")
            except Exception as e:  # noqa: BLE001
                from flliper.srt.pdflip.dual_d_priority import PdFlipDualDRetract

                if isinstance(e, PdFlipDualDRetract):
                    raise  # a named stop is never a logged warning
                logger.warning("%s rung 3 (retract) failed: %s", self._LADDER_PREFIX, e)

        freed = max(0, int(self.uniform_min_avail()) - before)
        self._log_ladder(need, before, freed_by)
        return freed

    def _log_ladder(self, need: int, before: int, freed_by: list) -> None:
        after = int(self.uniform_min_avail())
        logger.warning(
            "%s asked for %d tokens: group-uniform available %d -> %d (%+d) "
            "via %s. The park guard still decides; this only changed what "
            "there is to decide from.",
            self._LADDER_PREFIX,
            need,
            before,
            after,
            after - before,
            ", ".join(freed_by) if freed_by else "nothing (every rung was spent)",
        )

    def _retract_decode_and_requeue(
        self,
        batch: ScheduleBatch,
        *,
        kv_full_retract_flag: bool,
        reason: Optional[str] = None,
    ) -> int:
        """Retract decode victims and put every one of them back. Returns the
        tokens the pool gained.

        #679 rung 3: EXTRACTED VERBATIM from update_running_batch so the
        admission ladder can reach this actuator without owning a second copy
        of it. That matters more than it looks. The retraction itself is one
        call; what surrounds it is the part that must not drift -- the metrics,
        the new_token_ratio handover, the abort dispatch for requests that
        could not be kept, and above all

            for req in retracted_reqs:
                self._969ad_note_retract(req, "_retract_decode_and_requeue")
                self._add_request_to_queue(req, is_retracted=True)

        A second implementation that forgot that line would LEAK every victim
        it retracted, which is a worse failure than the crash this ladder
        exists to prevent. One implementation, two call sites, no drift.

        THE CALLER OWNS THE PRECONDITIONS, and they are not optional:
          * ``batch.uniform_avail_floor`` must already be the reduced value --
            it bounds the retraction loop and the last-survivor test, and #583
            is exactly the case where the entry decision was uniform and the
            loop bound was not, so ranks popped DIFFERENT numbers of victims;
          * the decision to call at all must be group-uniform for the same
            reason.
        """
        # W-DUAL-D-RETRACT (user decision 01.10.): group D of the dual layout
        # never retracts a running decode -- a named stop, before anything is
        # mutated (pdflip/dual_d_priority.py). Off the dual layout a no-op.
        from flliper.srt.pdflip.dual_d_priority import refuse_d_retract

        refuse_d_retract(batch, kv_full=kv_full_retract_flag, reason=reason)
        old_available_tokens = self.token_to_kv_pool_allocator.available_size()
        old_ratio = self.new_token_ratio_tracker.current
        mamba_allocator = getattr(
            self.tree_cache.req_to_token_pool, "mamba_allocator", None
        )
        old_mamba_available = (
            mamba_allocator.available_size() if mamba_allocator is not None else None
        )
        _draft_snap = self._pdflip_d_park_draft_snapshot(batch)  # H91d
        retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
            self.server_args
        )
        self._pdflip_d_park_draft_save(retracted_reqs, _draft_snap)  # H91d
        new_available_tokens = self.token_to_kv_pool_allocator.available_size()
        new_token_gained = new_available_tokens - old_available_tokens
        mamba_num_gained = (
            mamba_allocator.available_size() - old_mamba_available
            if mamba_allocator is not None
            else None
        )

        self.metrics_reporter.num_retracted_reqs = len(retracted_reqs)
        if self.metrics_reporter.enable_metrics and len(retracted_reqs) > 0:
            self.metrics_reporter.metrics_collector.increment_retracted_reqs(
                num_retracted_reqs=len(retracted_reqs),
                num_retracted_input_tokens=sum(
                    len(r.origin_input_ids) for r in retracted_reqs
                ),
                num_retracted_output_tokens=sum(
                    len(r.output_ids) for r in retracted_reqs
                ),
            )
        self.new_token_ratio_tracker.current = new_token_ratio
        for req in reqs_to_abort:
            abort_reason: FINISH_ABORT = req.to_finish
            self.ipc_channels.send_to_tokenizer.send_output(
                AbortReq(
                    finished_reason=abort_reason.to_json(),
                    rid=req.rid,
                ),
                req,
            )

        # #888b: a THIRD caller needs a third sentence. The two below are the
        # only reasons this actuator used to run, and neither is true when a
        # carrier yields its request seat: "KV cache pool is full" is exactly
        # the mis-attribution that ticket corrects (12876 tokens were free at
        # the measured stall) and "Testing retraction" is simply false. A
        # caller that supplies its own reason gets it; `reason=None` keeps the
        # two existing call sites byte-identical.
        msg_prefix = (
            reason
            if reason
            else (
                "KV cache pool is full. Retract requests. "
                if kv_full_retract_flag
                else "Testing retraction. "
            )
        )
        msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
        if mamba_num_gained is not None:
            msg_details += f", #mamba_num_gained: {mamba_num_gained}"
        if kv_full_retract_flag:
            msg_details += (
                f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
            )
        logger.warning(msg_prefix + msg_details)

        for req in retracted_reqs:
            self._969ad_note_retract(req, "_retract_decode_and_requeue")
            self._add_request_to_queue(req, is_retracted=True)
        self._pdflip_d_park_note_retracted(retracted_reqs)  # H91b: D parks, never discards
        return new_token_gained
