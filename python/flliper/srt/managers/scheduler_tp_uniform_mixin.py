"""``Scheduler`` methods moved out of ``scheduler.py`` (refactor, behaviour-neutral): bodies are byte-identical,
the class stays reachable as ``Scheduler.<name>`` through the mixin. Logger name unchanged."""

from flliper.srt.distributed.device_communicators.barlink_capture_census import capture_census
from flliper.srt.distributed.device_communicators.barlink_capture_census import capture_census_enabled
from flliper.srt.managers import prefetch_ballot
from flliper.srt.managers import tp_head_congruence
from flliper.srt.managers import tp_match_floor
from flliper.srt.managers import uniform_floor_scope
from flliper.srt.managers.phase_purity import UNIFORM_SEAM_PREMISE_ATTR
from flliper.srt.managers.schedule_policy import AddReqResult
from flliper.srt.managers.schedule_policy import CacheAwarePolicy
from flliper.srt.mem_cache.common import cap_above_evictable
from flliper.srt.mem_cache.common import publish_cap_blind_gap
from flliper.srt.pdflip import extend_trim as _pdflip_extend_trim
from flliper.srt.pdflip import head_vote_hold as _pdflip_head_vote_hold
from flliper.srt.pdflip import store_reprobe_epoch as _store_reprobe_epoch
from flliper.srt.utils.nvtx_utils import scheduler_nvtx_method
from typing import Dict
from typing import Optional
import logging
import os
import time
import torch.distributed

logger = logging.getLogger("flliper.srt.managers.scheduler")


class SchedulerTpUniformMixin:
    """Methods moved out of ``Scheduler`` (see module docstring)."""

    def _form_a_tp_exchange(self, site: str, payload):
        """H105: one small broadcast from the attention host (TP rank 0) under
        the #1028d blocked-recv stamp, so a peer that never arrives is named by
        the watchdog (``chain-recv/<site>``) instead of read as a silent wedge.
        The host passes its payload, a worker ``None``; every rank returns the
        host's."""
        from flliper.srt.utils import broadcast_pyobj

        self._note_pp_chain_blocked(site, time.monotonic())
        try:
            out = broadcast_pyobj(
                [payload] if payload is not None else [],
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        finally:
            self._note_pp_chain_blocked(None, None)
        return out[0] if out else None

    def _form_a_tp_gather(self, site: str, payload):
        """#239 S3d: :meth:`_form_a_tp_exchange` for a verdict EVERY rank
        contributes to (the token cut): an all-gather over the TP cpu group,
        in TP order, under the same blocked-recv stamp."""
        self._note_pp_chain_blocked(site, time.monotonic())
        try:
            out = [None] * len(self.tp_group.ranks)
            torch.distributed.all_gather_object(out, payload, group=self.tp_cpu_group)
        finally:
            self._note_pp_chain_blocked(None, None)
        return out

    def _form_a_is_host(self) -> Optional[bool]:
        """H105: None off a Form A follow group (tp 1, pp > 1, switch off);
        else whether this rank is the attention host. The host must be the
        broadcast source (TP rank 0) -- anything else is a named stop."""
        if self.ps.tp_size <= 1 or self.ps.pp_size > 1:
            return None
        from flliper.srt.managers import tp_match_floor as _tmf

        if not _tmf.form_a_follow_active():
            return None
        is_host = not _tmf.this_rank_follows()
        if is_host != (self.tp_group.rank == self.tp_group.ranks[0]):
            raise _tmf.FormAAdmissionSplit(
                f"H105 RU FORM-A ADMISSION HOST-NOT-SOURCE tp_rank={self.tp_group.rank} "
                f"src={self.tp_group.ranks[0]} is_host={is_host}: the attention "
                "host must be TP rank 0, the source of the admission broadcast."
            )
        return is_host

    def _form_a_admission_follow_fn(self):
        """H105: the callable `PrefillAdder.add_one_req` hands its gate verdict
        to on a Form A group (None elsewhere -- every gate stays rank-local)."""
        is_host = self._form_a_is_host()
        if is_host is None:
            return None
        from flliper.srt.managers import tp_match_floor as _tmf

        from flliper.srt.rank_role import form_a_token_cut_active

        def _exchange(payload):
            return self._form_a_tp_exchange("form-a-admission/tp<-verdict", payload)

        # #239 S3d: under the token cut every rank's gate counts (MIN).
        _gather = None
        if form_a_token_cut_active():

            def _gather(payload):
                return self._form_a_tp_gather("form-a-admission/tp<->verdict", payload)

        # H106 (rc12z23 D 15:28:40-15:36:12): the host's load-back drain and
        # its deadlock verdict ride this verdict (tp_match_floor docstring).
        _tree = getattr(self, "tree_cache", None)
        _drain_rides = _gather is None and _tree is not None
        _watch = None
        if is_host and _gather is None:
            _watch = getattr(self, "_h106_wedge_watch", None)
            if _watch is None:
                from flliper.srt.environ import envs

                _watch = self._h106_wedge_watch = _tmf.FormAAdmissionWedgeWatch(
                    envs.FLLIPER_PDFLIP_FORM_A_DEADLOCK_STOP_S.get()
                )

        def _worker_drain(host_drained):
            # The worker's own leaf set, whole -- the host drained its whole set
            # (the replicas hold the same leaves); the gate's lock on
            # req.last_node keeps the matched prefix here as it does on the host.
            from flliper.srt.mem_cache.base_prefix_cache import EvictParams

            ev = int(_tree.evictable_size())
            got = _tree.evict(EvictParams(num_tokens=ev)) if ev > 0 else None
            logger.info(
                "H106 FORM-A LOADBACK-DRAIN FOLLOW host_drained=%d evictable=%d "
                "evicted=%d: the attention host drained its evictable leaves for "
                "a refused load-back; this worker drains alike so the group floor "
                "sees the room",
                int(host_drained), ev,
                int(getattr(got, "num_tokens_evicted", 0) or 0),
            )

        def _follow(req, gate, price=None, budget=None):
            local = _tmf.ADMISSION_ADMIT if gate is None else gate.name
            drained, stop = 0, ""
            if is_host and _drain_rides:
                # set by add_one_req around ITS host-first load-back only
                drained = int(getattr(req, "_h106_host_drained", 0) or 0)
                req._h106_host_drained = 0
            if _watch is not None:
                rb = getattr(self, "running_batch", None)
                running_empty = (
                    rb is not None
                    and rb.is_empty()
                    and getattr(self, "chunked_req", None) is None
                )
                stop = _watch.observe(
                    req.rid, local, price, budget,
                    running_empty=running_empty, now=time.monotonic(),
                )
            _numbers = {}
            code = _tmf.form_a_admission_verdict(
                req.rid,
                local,
                is_host=is_host,
                exchange=_exchange,
                price=price,
                budget=budget,
                gather=_gather,
                drained=drained,
                stop=stop,
                on_host_drain=_worker_drain if _drain_rides else None,
                sink=_numbers,
            )
            # Q-702 (Auftrag 1522): the GROUP's gate numbers of this call -- the
            # host's (or, under the token cut, the decider's), identical on every
            # rank. A worker's own gate said ADMIT (its pool is not the host's),
            # so its adder never wrote a lifetime refusal; the SEAT-AGE verdict
            # (`pdflip/d_park_runtime.note_adder_refusal`) reads these instead.
            _follow.last_verdict = (
                str(req.rid), code, _numbers.get("price"), _numbers.get("budget")
            )
            return None if code == _tmf.ADMISSION_ADMIT else AddReqResult[code]

        # H105b (rc12z20 D 12:55:34, pdflip-36-145): the host runs its host
        # load-back BEFORE it sends the verdict, so a load-back WAIT on the
        # host (no device room, anchor given back) rides the same broadcast.
        # Under the token cut the verdict is a gather of real gates on every
        # rank -- unchanged there.
        _follow.host_decides_load_back = bool(is_host) and _gather is None
        _follow.last_verdict = None  # Q-702: (rid, code, price, budget) of the last call
        return _follow

    def _form_a_extend_set_riegel(self, can_run_list) -> None:
        """H105 riegel after the admission loop: the host's built extend set
        against this rank's; a split is a named stop before the forward."""
        is_host = self._form_a_is_host()
        if is_host is None:
            return
        from flliper.srt.managers import tp_match_floor as _tmf

        follow = _tmf.form_a_extend_set_check(
            _tmf.form_a_extend_set(can_run_list),
            is_host=is_host,
            exchange=lambda p: self._form_a_tp_exchange(
                "form-a-admission/tp<-extend-set", p
            ),
            # H105f: this rank's prompt length per entry -- a difference in the
            # end of a chunk that stays chunked is the host's to decide
            fill_lens=[
                len(getattr(r, "full_untruncated_fill_ids", ()) or ())
                for r in can_run_list
            ],
        )
        if follow:
            for r in can_run_list:
                end = follow.get(str(getattr(r, "rid", "?")))
                if end is not None:
                    r.set_extend_range(int(r.extend_range.start), int(end))

    def _uniform_kv_available(self):
        """Rank-uniform KV rows available, or None (#708).

        The BOTH-BLOCKED decline names its binding resource from this. It must
        be the GROUP MIN, not the local pool: every PhasePolicyInputs field is
        replicated by contract, and under uneven DCP the local availability
        differs per rank, so a local value would make the decline text -- and
        anything later keyed on it -- rank-dependent. That is the #616g
        divergence class this codebase already pays to avoid.
        ``uniform_avail_for_evict`` is the existing accessor: it returns the
        published group-min floor when the pools are uneven and the live local
        value when they agree.

        Returns None rather than a guess when it cannot be read, so the policy
        can say "not measured" instead of asserting.
        """
        try:
            tree = getattr(self, "tree_cache", None)
            if tree is None:
                return None
            allocator = getattr(tree, "token_to_kv_pool_allocator", None)
            if allocator is None:
                return None
            from flliper.srt.mem_cache.common import uniform_avail_for_evict

            return int(uniform_avail_for_evict(tree, allocator))
        except Exception:  # noqa: BLE001 - a diagnosis must never break a round
            return None

    def _pdflip_group_min_flags(self, flags):
        """#1471e: ONE verdict for the group.  weg2xsn240: rank 2 released a
        parked request from its rank-local verdict while ranks 0/1 still held
        it -> '#791b PREFETCH-BALLOT DIGEST MISMATCH STOP ... queue_len=1
        head=[pdflip-6-3]' on all three ranks (RAENGE-NIE-UNEINS).  The parked
        lists are replicated in arrival order, so an element-wise MIN over the
        TP cpu group makes every rank release the same requests on the same
        pass.  Falls back to the local flags on a single rank / no group."""
        vals = [1 if f else 0 for f in flags]
        if not vals:
            return vals
        try:
            tp_size = int(getattr(getattr(self, "ps", None), "tp_size", 1) or 1)
            group = getattr(self, "tp_cpu_group", None)
            if tp_size > 1 and group is not None and torch.distributed.is_initialized():
                t = torch.tensor(vals, dtype=torch.int64)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group)
                return [int(x) for x in t.tolist()]
        except Exception as exc:  # noqa: BLE001
            # NF-RANG-AUDIT 1014 Fix 3 (B2): a failed group verdict is a named stop, never the local flags
            from flliper.srt.pdflip import group_min_strict as _gms

            if _gms.enabled():
                logger.error("%s flags n=%d (%s: %s)", _gms.MARK, len(vals), type(exc).__name__, exc)
                raise _gms.failed("flags", len(vals), exc) from exc
            logger.info("#1471 group verdict n/a (%s: %s) -- local flags used", type(exc).__name__, exc)
        return vals

    def _pdflip_group_min_ints(self, vals):
        """#1479b: element-wise MIN of small ints over the TP cpu group;
        local values when there is no group.  Fail-soft."""
        vals = [int(v) for v in vals]
        if not vals:
            return vals
        try:
            tp_size = int(getattr(getattr(self, "ps", None), "tp_size", 1) or 1)
            group = getattr(self, "tp_cpu_group", None)
            if tp_size > 1 and group is not None and torch.distributed.is_initialized():
                t = torch.tensor(vals, dtype=torch.int64)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group)
                return [int(x) for x in t.tolist()]
        except Exception as exc:  # noqa: BLE001
            # NF-RANG-AUDIT 1014 Fix 3 (B2): a failed group verdict is a named stop, never the local values
            from flliper.srt.pdflip import group_min_strict as _gms

            if _gms.enabled():
                logger.error("%s ints n=%d (%s: %s)", _gms.MARK, len(vals), type(exc).__name__, exc)
                raise _gms.failed("ints", len(vals), exc) from exc
            logger.info("#1479 group min n/a (%s: %s) -- local values used", type(exc).__name__, exc)
        return vals

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def _update_uniform_pool_budget(self, running_batch=None) -> None:
        """MIN-reduce this iteration's pool headroom across the TP group.

        ``running_batch``: the pass's running batch after the prefill merge
        (W-SEAT, see ``_local_admit_limit``); None reads ``self.running_batch``.

        The rank-uniform counterpart to ``token_to_kv_pool_allocator.
        available_size()``, which under uneven DCP/TP differs per rank
        because the pools do (weighted ownership). Anything that decides a
        BRANCH from the local value can split the group across branches that
        carry different collectives -- the rank-local-test-before-a-group-
        collective family.

        COST, named exactly: ONE MIN ``all_reduce`` of a ONE-element int64
        CPU tensor -- 8 bytes -- on ``tp_cpu_group``, i.e. the gloo CPU group,
        NOT the device group and not BAR1. It therefore neither touches the
        BAR1 aperture nor serialises with the forward stream; it is a host
        synchronisation point once per scheduler iteration. World size 1 takes
        no collective at all and is byte-identical to the previous path.

        (The offload variant packs two elements, because it also reduces
        available+evictable for the prefill admission budget. This path needs
        only the decode-mem quantity, so it reduces one.)

        When the offload manager is active it has ALREADY done exactly this
        reduce for this iteration (``update_dcp_admission_state``), so the
        value is read from there rather than reduced a second time -- two
        reduces would be two chances for the counts to diverge, which is the
        failure this is closing.

        Valid for the whole iteration, on the same argument the offload
        version documents: between here and the decode-mem check the
        scheduler only builds batches from offsets and defers real allocation
        to the forward, so the snapshot still holds.
        """
        kvso = self.kv_session_offload
        if kvso is not None:
            self._uniform_min_avail = int(kvso.dcp_min_avail())
            # The offload manager only exists under uneven DCP, so the pools
            # ARE uneven here; publish unconditionally (no max to compare).
            self._publish_uniform_evict_floor(self._uniform_min_avail)
            # #639, A NAMED GAP rather than a silent one. The host floor does
            # NOT ride the offload manager's reduce, and this branch's whole
            # contract is that it takes NO reduce of its own -- "a second
            # reduce here would be a second chance for the collective counts
            # to diverge", pinned by
            # test_uniform_decode_mem_603.test_the_offload_manager_is_not_reduced_twice.
            # Adding one to carry the host term would trade a divergence this
            # branch has closed for one it has not, on a path that cannot be
            # exercised without --enable-kv-session-offload.
            #
            # So under kv-session-offload the host floor stays OFF and
            # `write_backup` keeps its pre-#639 rank-local gate. The right
            # close is to widen `update_dcp_admission_state`'s existing packed
            # reduce by one pair, in kv_session_offload.py, where the reduce
            # already lives -- not here.
            self._publish_uniform_host_floor(None)
            # #639b: the mamba floor is off here for the same reason and with
            # the same named gap -- this branch's contract is that it takes NO
            # reduce of its own, and the offload manager's existing reduce
            # does not carry a mamba term. Under --enable-kv-session-offload
            # the mamba eviction keeps its pre-#639b rank-local gate. The right
            # close is to widen `update_dcp_admission_state`'s packed reduce in
            # kv_session_offload.py, where the reduce already lives, not to add
            # a second one here.
            self._publish_uniform_mamba_floor(None)
            # The offload manager reduced the admission quantity itself in
            # `update_dcp_admission_state`; `get_new_batch_prefill` reads it
            # from there, so nothing is stored here.
            self._uniform_budget_deficit = 0
            # #791b: NAMED GAP, same shape as the #639 host floor above --
            # this branch's contract is that it takes no reduce of its own,
            # so the prefetch ballot stays OFF under kv-session-offload and
            # admission keeps its rank-local prefetch verdict there. The
            # right close is to widen `update_dcp_admission_state`'s packed
            # reduce, where the reduce already lives.
            self._uniform_prefetch_ballot = None
            # #823 W9b: and the head verdict with it. CLEARED ON EVERY
            # EARLY RETURN, not left standing: this branch takes no reduce,
            # so anything still published here is the LAST pass's group
            # verdict, and a stale verdict is worse than none -- it would
            # reorder and truncate this pass's queue on numbers computed for
            # a different one, while every log line said "group".
            self._uniform_head_inputs = None
            # #794: NAMED GAP, same shape as the #639 host floor above. This
            # branch takes no reduce of its own, so the corridor width stays
            # unreduced and the consumer declines to narrow rather than narrow
            # on a rank-local reading. The right close is one more term in
            # `update_dcp_admission_state`'s packed reduce, where it already
            # lives.
            self._uniform_corridor_width = None
            # KVOFFSET-1024: NAMED GAP, same shape -- no reduce of its own here, so no group ratio; the publisher then
            # prices with TP0's own ratio and says so (``decode_reserve_scope=tp0``)
            self._kvoffset_group_ratio = None
            # #1203: NAMED GAP, same shape as the #639 host floor above. This
            # branch takes no reduce of its own, so the seam-transport premise
            # stays unreduced and its read site decides on this rank's own
            # answer -- byte-identical to the pre-#1203 path under
            # --enable-kv-session-offload. The right close is one more term in
            # `update_dcp_admission_state`'s packed reduce, where it lives.
            setattr(self, UNIFORM_SEAM_PREMISE_ATTR, None)
            return

        alloc = self.token_to_kv_pool_allocator
        local_avail = int(alloc.available_size())
        grp = getattr(self, "tp_cpu_group", None)
        if grp is None or torch.distributed.get_world_size(grp) <= 1:
            # #788: SAY IT ONCE, LOUDLY, THAT THE FLOORS ARE OFF.
            #
            # The comment below ("nothing to diverge from") is true for TP and
            # FALSE for PP. This reduce group is tp_cpu_group, so on a
            # TP=1/PP=3 boot it has one member on every rank and all three
            # uniformity floors switch off -- while three PP ranks go on
            # deriving admission from purely local availability and prefix
            # matches. That is the measured cause of a pipeline deadlock, so
            # the condition should be readable from the boot log rather than
            # inferred from source months later.
            # #824: REPORT THE TRANSITION, NOT THE FIRST SIGHTING. Under
            # --enable-phase-flip this scope is not a boot constant:
            # phase_flip_runtime rebuilds the TP group per phase
            # (`want_tp_size = n if tp_phase else 1`), so the floors are ON
            # through the TP decode phase and OFF through the PP prefill
            # phase, switching at every cutover. The once-per-process latch
            # this replaces meant a boot that flipped four times in 55 s
            # reported its coverage once, at the first PP iteration, and never
            # again -- so "off for the whole run" and "off for the prefill
            # half of every cutover" read identically in the log, and coverage
            # coming BACK was never reported at all.
            uniform_floor_scope.report_scope(
                self,
                None if grp is None else int(torch.distributed.get_world_size(grp)),
                logger,
            )
            self._uniform_min_avail = local_avail
            self._uniform_budget_deficit = 0
            # KVOFFSET-1024: one rank -- its own ratio IS the group's
            from flliper.srt.pdflip import d_lifetime_reading as _dlr1

            self._kvoffset_group_ratio = _dlr1.local_ratio(self)
            # One rank: nothing to diverge from, so the floor stays OFF and
            # every cache-mutation trigger keeps reading its live local value
            # -- byte-identical to the pre-#616g path.
            # #1045: PUBLISH THE LOCAL VALUE, NOT None, ON THE SINGLE-RANK
            # PATH. This is what lets `None` become a CONSTRUCTION VIOLATION
            # everywhere downstream instead of an ambiguous "single rank or
            # nobody published". `local_avail` IS the group min when the group
            # is one rank, so the value is correct rather than a placeholder,
            # and every floor reader behaves exactly as it did on the local
            # path. The host and mamba floors are a different axis and keep
            # their None; only the evict floor is being made total here.
            self._publish_uniform_evict_floor(local_avail)
            # F1 (nf-next-1006-01): nothing is pinned on one rank; clear what a
            # previous (TP) layout published so a flip cannot read it
            publish_cap_blind_gap(
                getattr(self, "tree_cache", None),
                allocator=getattr(self, "token_to_kv_pool_allocator", None),
                pin_admission=False, min_avail=0, min_admission=0,
                local_evict_full=0, local_cap_above=0,
            )
            # SF (b23 #1004): on pp > 1 this "group min" is THIS rank's own
            # value and the ranks that must agree are not in the group; the
            # load-back reads the mark and loads in the same pass instead of
            # retrying next pass (pdflip/pp_slot_fidelity.py). Switch off: no mark.
            from flliper.srt.pdflip import pp_slot_fidelity as _sf

            _sf.mark_floor_scope(getattr(self, "tree_cache", None), _sf.pp_size_of(self) > 1)
            if _sf.pp_size_of(self) > 1:
                _sf.ensure_relief_provider(self)  # SF-X: the rank-local extend net
            self._publish_uniform_host_floor(None)
            self._publish_uniform_mamba_floor(None)
            # #791b: one rank -- nothing to diverge from, ballot off, the
            # local prefetch verdict is already the group verdict.
            self._uniform_prefetch_ballot = None
            # #823 W9b: one rank in this group -- there is no group verdict
            # to hand down, and the enforcer's own gate is off for the same
            # reason (`enforcer_gate`: a MIN over a singleton is this rank's
            # own vote). Cleared rather than left standing, so a phase flip
            # back into PP cannot form on the TP phase's last verdict.
            self._uniform_head_inputs = None
            # #794: one rank in this group -- the local corridor decision IS
            # the group decision, taken at the call site.
            self._uniform_corridor_width = None
            # #1203: one rank -- nothing to diverge from, and the read site's
            # local answer IS the group answer. CLEARED rather than left
            # standing, so a flip back into PP cannot decide the seam on the
            # TP phase's last verdict (#823 W9b's rule for `_uniform_head_
            # inputs`, and it is why this is None and not the local vote).
            setattr(self, UNIFORM_SEAM_PREMISE_ATTR, None)
            return
        # #824: the ON side had no report at all, so coverage COMING BACK --
        # every pp_to_tp cutover -- was invisible. Same call, same predicate.
        uniform_floor_scope.report_scope(
            self, int(torch.distributed.get_world_size(grp)), logger
        )
        # #610: PREFILL ADMISSION rides on this same reduce. `PrefillAdder`'s
        # `rem_total_tokens` / `cur_rem_tokens` (schedule_policy.py:681/719) read
        # `available_size() + evictable_size()` -- rank-LOCAL under uneven DCP,
        # where the ranks own weighted shares of the token axis. `budget_state`
        # (:808) turns that straight into NO_TOKEN, so the binding rank stops
        # admitting while the slack ranks continue: `can_run_list` differs, the
        # prefill batch is composed differently per rank, and the per-layer TP
        # collectives in the forward are entered with mismatched shapes. The
        # kv-session-offload path has been pinned against exactly this since
        # `update_dcp_admission_state` (kv_session_offload.py:2741), but that
        # pin only exists when the offload manager is constructed -- with
        # `--enable-kv-session-offload` off, which is the default, prefill
        # admission was still deciding rank-locally. Same defect, same fix, one
        # branch further out.
        #
        # MIN-ballot, not gate removal: the budget every rank admits against is
        # the BINDING rank's, so no rank is asked to hold a request it cannot.
        # The per-rank surplus is exported as a non-negative deficit the adder
        # subtracts, which is the representation the offload variant already
        # uses and the adder already consumes (`dcp_avail_deficit`).
        #
        # PACK SIZE is decided by `uneven_dcp_active(dcp_size)`, a replicated
        # config-derived predicate (the token-ratio vector is installed
        # identically on every rank), so the payload width is rank-uniform. Even
        # TP keeps the one-element payload and a structurally zero deficit --
        # byte-identical, no behaviour change on the default path.
        from flliper.srt.distributed.utils import uneven_dcp_active

        pin_admission = uneven_dcp_active(self.server_args.dcp_size)
        local_admission = 0
        local_evict_full = 0
        local_cap_above = 0
        if pin_admission:
            tree = self.tree_cache
            fe = getattr(tree, "full_evictable_size", None)
            # Mirrors `update_dcp_admission_state`'s quantity exactly, including
            # its known simplification: the all-SWA adder reads
            # `swa_evictable_size` instead. SWA models route to
            # UnifiedRadixCache and are out of this path's reach.
            local_evict = int(fe() if fe is not None else tree.evictable_size())
            # F1 (nf-next-1006-01): the evictable tokens above the engaged
            # residency cap are not fundable (the pool takes every id they free
            # straight back). They go into THIS rank's vote, through the MIN this
            # reduce already takes -- no new site, no new payload element, no new
            # guard -- so every rank admits against the same group number.
            local_evict_full = local_evict
            local_cap_above = cap_above_evictable(
                tree,
                allocator=getattr(self, "token_to_kv_pool_allocator", None),
                page_size=int(getattr(self, "page_size", 1) or 1),
            )
            local_evict = local_evict - local_cap_above
            local_admission = local_avail + local_evict
        # #616g: `-local_avail` rides the SAME reduce so one MIN yields both
        # the group minimum and (negated) the group MAXIMUM. The pair is what
        # decides whether the pools are uneven AT ALL this iteration, which is
        # the activation predicate for the eviction floor below -- derived from
        # a collective, so it is rank-uniform by construction and needs no
        # server-arg coupling. Width stays rank-uniform: the element is added
        # unconditionally, and `pin_admission` (the only other width term) is a
        # replicated config predicate.
        vals = (
            [local_avail, local_admission, -local_avail]
            if pin_admission
            else [local_avail, -local_avail]
        )
        # #639: the HOST tier's availability rides the SAME reduce, as a
        # (x, -x) pair so one MIN yields its group min and max -- the same
        # trick and the same activation predicate the device floor above
        # uses. Appended UNCONDITIONALLY: a boot with no host tier still
        # contributes the pair, carrying `_HOST_AVAIL_ABSENT`, so the payload
        # width never depends on a per-rank capability. (Whether a host tier
        # exists is `--enable-hierarchical-cache`, a replicated server arg, so
        # the sentinel is all-or-nothing in practice; contributing it
        # unconditionally is what makes that a property of the code rather
        # than of the flagset.)
        local_host_avail = self._local_host_avail()
        vals = vals + [local_host_avail, -local_host_avail]
        # #639b: the MAMBA slot pool's availability rides the SAME reduce, as
        # a third (x, -x) pair, on the identical argument the host pair above
        # documents -- one MIN yields its group min and max, and the
        # min-vs-max comparison is the activation predicate. Appended
        # UNCONDITIONALLY, carrying `_MAMBA_AVAIL_ABSENT` on a boot with no
        # mamba pool, so the payload width never depends on a per-rank
        # capability. Whether a mamba pool exists is a property of the model
        # (hybrid SSM or not), replicated across ranks, so the sentinel is
        # all-or-nothing in practice; contributing it unconditionally is what
        # makes the uniform width a property of the code rather than of the
        # checkpoint.
        local_mamba_avail = self._local_mamba_avail()
        vals = vals + [local_mamba_avail, -local_mamba_avail]
        # #794: THE CORRIDOR WIDTH CEILING rides this reduce too, and it has to.
        #
        # The flip changes the topology under this scheduler AT RUNTIME. In the
        # PP phase this group has world 1 and PP0 decides the pass geometry for
        # the ring; in the TP phase the SAME process reads pp_size=1 and
        # tp_cpu_group world=3, every rank builds its own prefill batch, and a
        # rank-local width cut would split the group. The first metal run of the
        # actuator printed exactly that, from all three ranks:
        #     #794 CORRIDOR WIDTH ACTUATOR OFF on this topology:
        #     tp_cpu_group world=3 with pp_size=1
        # -- i.e. the actuator switched itself off in the phase where this fork
        # runs prefill (--phase-flip-purity prefill_in_tp) and where the
        # 2026-08-21 17:33 OOM actually happened, inside the first flip. A gap
        # that swallows the whole use case is not a gap, it is an outage.
        #
        # ONE MORE INT ON A REDUCE THAT ALREADY RUNS, which is what the gap note
        # itself named as the right close. Contributed UNCONDITIONALLY so the
        # payload width never depends on a per-rank capability: a rank that
        # cannot price, or has no guard, contributes the configured width and is
        # therefore never the binding vote. MIN, because the group may take on
        # only what its tightest card can fund.
        #
        # Placed BETWEEN the mamba pair and the ballot on purpose: everything
        # above is indexed from the head and the ballot is indexed from the
        # TAIL, so inserting here leaves both readings intact.
        _corridor_at = len(vals)
        vals = vals + [self._local_corridor_width_ceiling()]
        # VOTE-FIXED-LIVE (24.10., pdflip/vote_fixed_live.py rule 4, SOLL R11): the D ranks' measured fixed part
        # of an extend rides this reduce as ONE MIN slot carrying -ceil(F) -- the group MAX, so every rank's
        # width-vote line holds the worst rank's reading (TP-symmetric passes, but trimmed on different ranks).
        # Contributed UNCONDITIONALLY (0 without a reading: neutral, F >= 0), its own head index captured
        # before the append, read back before the tail-indexed ballot. No new collective.
        _vfix_at = len(vals)
        vals = vals + [_pdflip_extend_trim.live_fixed_payload()]
        # #791b: THE PREFETCH BALLOT rides the SAME reduce (instr22/instr23:
        # the admission loop's rank-local prefetch verdict split the TP
        # replicas -- one rank admitted a batch its peers declined, three
        # lockstep families crossed, barlink aborted the group 150 s later).
        # The drain that feeds the local verdicts is pulled forward to HERE
        # -- rank-local, no collective, and this site runs exactly once per
        # TP-loop iteration -- and memoised for `_get_new_batch_prefill_raw`
        # to consume once, so the pass still drains exactly once. Appended
        # AFTER the mamba pair, so every existing index below keeps its
        # meaning. See prefetch_ballot.py for the layout and the MIN==AND
        # argument.
        # #823 W9: THE HEAD-CONGRUENCE BLOCK rides this reduce too, and it is
        # placed HERE for the same reason the corridor width is -- between the
        # mamba pair and the ballot. Everything above is indexed from the
        # HEAD, the ballot below is indexed from the TAIL
        # (`len(vals) - (PREFETCH_BALLOT_SLOTS + 2)`), so an insertion at this
        # seam leaves both readings intact. Both blocks below capture their
        # own explicit index BEFORE appending; nothing here is read back by a
        # negative offset, which is the mistake #639b's note records ("the
        # `t[-2]`/`t[-1]` the host floor used to read would have silently
        # started reading MAMBA availability").
        #
        # Payload order, in full, after this block:
        #   [avail, (admission,) -avail, host, -host, mamba, -mamba,
        #    corridor, -vote_fixed, admit_limit, seam_premise, phase_domain[..],
        #    head_match[0..TP_HEAD_SLOTS-1],
        #    ballot_digest, -ballot_digest, ballot_v0..v{K-1}]
        #
        # FIX 2 (round 2): THE HEAD BLOCK MOVED DOWN, below the prefetch
        # drain, and the block that used to sit here now sits under it.  It
        # is still read by its own captured index and the ballot is still
        # read from the TAIL, so the two readings this seam protects are
        # unchanged; only the ORDER in which the local inputs are measured
        # moved.  Why, in full, at the head block's new site.
        # The COUNT arm's vote. Its own slot rather than a derivation from
        # `avail` above, because `get_num_allocatable_reqs` is bounded by
        # `admission_limiter.current` (:6526-6529) -- rank-local floating
        # state that the availability floor does not capture, so a count
        # derived from the uniform avail would still diverge.
        _limit_at = len(vals)
        vals = vals + tp_head_congruence.build_admit_limit_payload(
            self._local_admit_limit(running_batch)
        )
        # #1203 (family A1): THE SEAM-TRANSPORT PREMISE rides this reduce too,
        # as ONE AND-slot, and it is placed HERE for the reason the corridor
        # width and the head block above are -- everything before this point is
        # indexed from the HEAD and the ballot below is indexed from the TAIL
        # (`len(vals) - (PREFETCH_BALLOT_SLOTS + 2)`), so an insertion at this
        # seam leaves both readings intact. Its own explicit index is captured
        # BEFORE the append, like every block above it.
        #
        # WHAT IT CLOSES. `phase_purity.seam_transport_premise_holds` decides
        # whether the TP layout may build a seam-transport prefill batch, and
        # its whole read set was rank-local: the store witness reads the tree
        # cache's prefetch records, whose reduced half exists only under
        # `UnifiedRadixCache.tp_world_size > 1`, and THAT field is written once
        # at cache construction and never rebound at the cutover -- while THIS
        # group is rebound to world N when the TP phase is entered. On the
        # shipping form (--tp-size 1 --pp-size 3) the cache's world is 1 in
        # both phases, so three ranks weighed three different readings of the
        # same store and branched into different collectives: True builds a
        # transport extend batch, False yields no batch at all under strict:3
        # with drain mode. Measured divergence at this commit, one second
        # before the cutover of boot_855_weg1b9_1116175f6d_0904_164023
        # (log:1900 vs :1977/:2067): PP0 `absent=67`, PP1 and PP2
        # `assembling=67`, same 67 stems, same second.
        #
        # MIN OVER {0, 1} IS AND -- the same argument prefetch_ballot.py makes
        # for its own slots -- so the group transports only what every rank
        # agrees is a restore. No new collective: one more int on a reduce
        # that already runs, which is the close #794 and #823 W9 both used.
        _seam_premise_at = len(vals)
        vals = vals + [self._local_seam_premise_vote()]
        # #1068 weg1 S0: THE PHASE-DOMAIN VERDICT BUS rides this same reduce,
        # as ONE block, and it is placed HERE for the reason the #1203 seam
        # above is: everything before this point is indexed from the HEAD and
        # the ballot below is indexed from the TAIL, so an insertion at this
        # seam leaves both readings intact. Its head index is captured BEFORE
        # the append, like every block above it.
        #
        # ONE CARRIER, NOT N: four separate verdicts at this seam would be four
        # hand-rolled head indices and four chances to get the layout wrong,
        # which on a MIN reduce reads as agreement. The whole slot table --
        # including the slots whose producers land in later batches, which read
        # their MIN-neutral values until then -- is declared in one place.
        #
        # THE PER-PASS COUNTERS ARE READ AND CLEARED INSIDE THIS BUILD, not once
        # per scheduler pass: this site sits after the world-size guard's return
        # at :6978, so it runs exactly as often as the reduce. A per-pass reset
        # would erase every PP-phase detection before the next TP reduce could
        # vote it -- a group STOP deleted by a reset rather than by a missing
        # vote.
        #
        # IMPORTED LOCALLY, the way this same method already imports
        # `uneven_dcp_active` and the way `:5515` imports `bound_phase`: the
        # module is referenced only from this seam, and the two statements the
        # bus adds live inside the one range this slice owns in this file.
        # Resolved through the module object at call time, so a test can still
        # neuter exactly one of its functions.
        from flliper.srt.managers import phase_domain_verdict

        _phase_domain_at = len(vals)
        _phase_domain_local = phase_domain_verdict.build_phase_domain_payload(self)
        vals = vals + _phase_domain_local
        _phase_domain_phase = None  # #1233 (WEG 2, S0): one layout per process.
        # `_phase_domain_layout_announced` IS THIS EMITTER'S OWN BOOKMARK of the
        # phase it last logged the layout for, and it cannot be recomputed where
        # it is read: nothing else in the process records whether THIS process
        # already emitted the line, so the alternative to the field is a line on
        # every reduce -- the once-per-process latch #824 replaced, read from the
        # other end. It is not a second record of a fact another object owns:
        # the phase itself is read live from `phase_flip_active_stack` on the
        # line above and is never copied here. The tree's own precedent for this
        # shape is `uniform_floor_scope.py:135-137`, the say-it-once-per-change
        # bookmark the world-size guard above already keeps on this same object.
        if getattr(self, "_phase_domain_layout_announced", None) != _phase_domain_phase:
            self._phase_domain_layout_announced = _phase_domain_phase
            logger.info(
                "%s",
                phase_domain_verdict.layout_instrument_line(
                    _phase_domain_at, _phase_domain_phase
                ),
            )
        _ballot_verdicts = self._drain_prefetch_progress()
        self._pass_prefetch_verdicts = _ballot_verdicts
        # WHY THE LOCAL INPUTS ARE COMPUTED HERE. The sort key
        # (`num_matched_prefix_tokens`) is normally populated inside
        # `calc_priority`, which runs LATER in the pass -- so at reduce time
        # it is either zero or last pass's value, and reducing that would
        # agree on a stale number. The fix is the one #791b already used in
        # this function for the prefetch verdicts: pull the RANK-LOCAL
        # computation forward to here (no collective, and this site runs
        # exactly once per TP-loop iteration) and memoise it for the batch
        # formation to consume. Same move, same reason.
        #
        # FIX 2 (round 2): AND IT IS MEASURED BELOW THE DRAIN, not above it.
        # `_drain_prefetch_progress` is the call that publishes a completed
        # store read into the host tier (`check_prefetch_progress` ->
        # `_insert_helper_host`), and the pass on which a request first
        # becomes eligible is exactly the pass its prefetch completes. Taken
        # above the drain, this vote was a PRE-LOAD snapshot of a tree the
        # gate then read POST-load: the group's match for the flip's own
        # request was 0 while every rank's own match was the whole loaded
        # prefix, and the X gate priced the difference as uncached work --
        # the whole prompt, W31, re-route, P re-prefills a store-resident
        # rid, flip, W31 again. #823's MIN is safe for the ORDER arm because
        # a smaller match there means MORE WORK (tp_head_congruence.py:60-64);
        # for a REFUSING consumer a smaller match means a REFUSAL, so the
        # borrowed reduce's safety direction is inverted and the vote has to
        # be a snapshot the gate can actually see. Nothing between the old
        # site and this one takes a collective or reads the head vote, so the
        # move is the whole of the fix on this side.
        # S42 HEAD-VOTE HOLD (pdflip/head_vote_hold.py): the vote's only reader
        # is this iteration's admission pass. When DQH holds that pass (decided
        # HERE, once, after the drain whose verdicts its fingerprint reads; the
        # elif chain consumes the same decision), every head arm abstains with
        # its MIN-neutral value -- no match walk, no probe, no store or
        # completion vote. rc11 14:13 (6 waiting): sched_ms 56.9 per decode
        # round, all of it this vote on held passes. Group-uniform: DQH's
        # inputs are replicated; a lone abstention MIN-reduces to ABSENT, i.e.
        # the head is delayed, the collective count is unchanged.
        _s42_held = _pdflip_head_vote_hold.decide(self, running_batch, _ballot_verdicts)
        if _s42_held:
            _head_canonical, _head_local_matches = [], {}
        else:
            _head_canonical, _head_local_matches = self._local_head_prefix_matches()
        # #823 W9b: the canonical head is no longer published as its own
        # attribute -- it goes into the single verdict value below, together
        # with the two numbers this reduce is about to produce, so the four
        # halves of one decision cannot get out of step with each other.
        _head_at = len(vals)
        vals = vals + tp_head_congruence.build_head_order_payload(
            _head_canonical, _head_local_matches
        )
        # FIX 7 (round 7): THE COMPLETION ARM, on the SAME canonical head and
        # the SAME reduce. `_admission_held_for_deferred_prefetch`'s own
        # docstring names this as the honest close of #1203 family A3 ("a
        # group-uniform defer budget or a second per-rid ballot arm"); this is
        # the second arm, and it is here rather than beside the ballot because
        # the ballot is indexed by QUEUE ORDER while the X gate reads the
        # canonical head. No new collective (MUST NOT 6).
        _xpend_at = len(vals)
        # S42: a held pass does not touch the pending stamps (an empty head
        # would DROP them and restart the wait); the age keeps running.
        vals = vals + tp_head_congruence.build_x_pending_payload(
            _head_canonical,
            {} if _s42_held else self._pdflip_local_store_read_pending_ages(_head_canonical),
        )
        # #1317 DESIGN A: THE STORE-PRICED MATCH ARM, third consumer of this
        # same reduce (after ORDER and COMPLETION), on the SAME canonical head.
        # The order arm votes the match against the TREE; a token group P
        # prefilled and wrote through to L3 is in neither of D's tiers, so the
        # X gate priced it as uncached work and refused the request -- W50,
        # re-route, P prefills a store-resident prompt AGAIN. This arm votes
        # what the STORE holds beyond the tree, so "uncached" can mean "not in
        # the tree AND not in the store". No new collective (MUST NOT 6).
        _xstore_at = len(vals)
        vals = vals + tp_head_congruence.build_x_store_match_payload(
            _head_canonical,
            {} if _s42_held else self._pdflip_local_store_matches(_head_canonical),
        )
        # REPROBE-1023 (pdflip/store_reprobe_epoch.py, SOLL R11): this rank's
        # STORE EPOCH (storage writes seen land), ONE slot on every rank; the
        # MIN is the group epoch -- moved on ALL ranks -- that re-schedules the
        # store arm's re-ask. Same reduce, no new collective (MUST NOT 6).
        _xstore_epoch_at = len(vals)
        vals = vals + _store_reprobe_epoch.build_epoch_payload(self)
        # RU (nf_rank_divergence): THE USABLE-MATCH ARM, fourth consumer of this
        # same reduce, on the SAME canonical head. The order arm votes the
        # match LENGTH; this arm votes it as 0 when the match ends on a node
        # whose recurrent anchor this rank cannot use (#928 a/b, evaluated
        # without COW side effects on the walk `_local_head_prefix_matches`
        # just took). Admission zeroes a rank's match when the group MIN is 0
        # (tp_match_floor.group_floor_zeroes, inside the mamba finalize before
        # the COW). Placed BEFORE the ballot, which is indexed from the tail.
        # No new collective (MUST NOT 6).
        _usable_at = len(vals)
        _usable_by_rid = {req.rid: req for req in self.waiting_queue}
        _usable_local = {} if _s42_held else tp_match_floor.local_usable_matches(
            getattr(self, "tree_cache", None), _usable_by_rid, _head_local_matches
        )
        vals = vals + tp_match_floor.build_usable_match_payload(
            _head_canonical, _usable_local, tp_head_congruence.TP_HEAD_SLOTS
        )
        # H97: the same vote negated, so this reduce also yields the group MAX.
        _usable_max_at = len(vals)
        vals = vals + tp_match_floor.build_usable_max_payload(
            _head_canonical, _usable_local, tp_head_congruence.TP_HEAD_SLOTS
        )
        # S42r A3 (review nf-review-s42-1018): THE HOLD DECISION RIDES THIS
        # REDUCE as an (x, -x) pair, so every rank learns the group's MIN and
        # MAX of "this rank holds". Placed before the tail-indexed ballot, its
        # own head index captured before the append. No new collective.
        _s42_hold_at = len(vals)
        vals = vals + _pdflip_head_vote_hold.build_hold_payload(_s42_held)
        # KVOFFSET-1024 (SOLL R11, Rang-Einigkeit): THE DECODE-RESERVE RATIO rides this reduce as ONE negated slot, so
        # the MIN yields the group MAX of ``new_token_ratio``. The ratio is RANK-LOCAL (NF y6h 10011531: TP0 priced a
        # reservation at 4964, TP1/TP2 at 6592 -- idle resets and decay steps differ per rank), and each rank's
        # lifetime gate subtracts ITS OWN reserve; the group admits only when every rank admits (tp_admission_agree),
        # so the binding reserve is the largest one. ``pdflip_kv.decode_reserve`` (get_internal_state, TP0) prices the
        # running decodes with this agreed ratio, never TP0's own. Its head index is captured before the append, it is
        # read back by that index, and it sits before the tail-indexed ballot. No new collective.
        _ratio_at = len(vals)
        from flliper.srt.pdflip import d_lifetime_reading as _dlr

        vals = vals + [-_dlr.ratio_ppm(_dlr.local_ratio(self))]
        _ballot_rids = [
            req.rid
            for req in self.waiting_queue[: prefetch_ballot.PREFETCH_BALLOT_SLOTS]
        ]
        vals = vals + prefetch_ballot.build_prefetch_ballot_payload(
            _ballot_rids, _ballot_verdicts
        )
        t = torch.tensor(vals, dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=grp)
        # S42r A3: the chain consumes the GROUP's hold, not this rank's: held
        # when ANY rank held (it abstained from the head vote, so no rank may
        # run a pass on it -- a lone pass would reach the Form A per-candidate
        # gather in add_one_req alone: X gate abstains on an ABSENT match and
        # does not refuse). On a split every rank also disarms DQH, so the next
        # iteration evaluates jointly.
        _pdflip_head_vote_hold.resolve_group(
            self, t[_s42_hold_at : _s42_hold_at + 2].tolist()
        )
        # KVOFFSET-1024: the group MAX of the decode-reserve ratio, read back by its captured index
        self._kvoffset_group_ratio = _dlr.ratio_from_ppm(-int(t[_ratio_at]))
        # #794: the group's tightest corridor width, read back before the
        # ballot so a later change to the ballot layout cannot silently move it.
        self._uniform_corridor_width = int(t[_corridor_at])
        # VOTE-FIXED-LIVE: the group max of F (ratchet up) for the next vote's line
        _pdflip_extend_trim.live_fixed_group(int(t[_vfix_at]))
        # #823 W9: read back by the indices captured above, before the ballot,
        # so a later change to the ballot layout cannot silently move them --
        # the same discipline the corridor width is read under.
        # THE PAYLOAD IS UNCHANGED BY W9b: same slots, same order, same
        # head-captured indices. Only the DESTINATION changes, from four
        # separate attributes to one value.
        _head_match_lens = t[
            _head_at : _head_at + tp_head_congruence.TP_HEAD_SLOTS
        ].tolist()
        # FIX 7: read back by its own captured head index, before the ballot,
        # under exactly the discipline the corridor width and the head block
        # are read under -- so a later change to the ballot layout cannot
        # silently move it.
        _xpend_lens = t[
            _xpend_at : _xpend_at + tp_head_congruence.TP_HEAD_SLOTS
        ].tolist()
        if len(_xpend_lens) != tp_head_congruence.TP_HEAD_SLOTS:
            raise RuntimeError(
                "#1234 W37 X-COMPLETION LAYOUT STOP: the reduced payload carries "
                f"no completion slice (head={_xpend_at}, "
                f"expected={tp_head_congruence.TP_HEAD_SLOTS}, "
                f"available={len(vals) - _xpend_at}). Pricing law 4 on a slice "
                "of the wrong width would read another arm's numbers as store-read "
                "ages, so the group stops here by name instead."
            )
        # #1317 DESIGN A: read back by its own captured head index, before the
        # ballot, under the same discipline as every other arm of this reduce.
        _xstore_lens = t[
            _xstore_at : _xstore_at + tp_head_congruence.TP_HEAD_SLOTS
        ].tolist()
        if len(_xstore_lens) != tp_head_congruence.TP_HEAD_SLOTS:
            raise RuntimeError(
                "#1317 W86 PdFlipXStoreLayoutStop: the reduced payload carries no "
                f"store-match slice (head={_xstore_at}, "
                f"expected={tp_head_congruence.TP_HEAD_SLOTS}, "
                f"available={len(vals) - _xstore_at}). Pricing the uncached "
                "extent on a slice of the wrong width would read another arm's "
                "numbers as store depths -- and this arm ADMITS work, so a "
                "foreign number here would admit a prefill the group cannot "
                "serve. The group stops here by name instead."
            )
        # REPROBE-1023: the group epoch, read back by its own captured index;
        # every rank then re-schedules the store arm from the GROUP's values
        # (tick, epoch, store vote) alike.
        if _xstore_epoch_at >= len(vals):
            raise RuntimeError(
                "REPROBE-1023 STORE-EPOCH LAYOUT STOP: the reduced payload carries "
                f"no store-epoch slot (at={_xstore_epoch_at}, available={len(vals)})."
            )
        _store_reprobe_epoch.after_reduce(
            self, _head_canonical, _xstore_lens, int(t[_xstore_epoch_at].item()),
        )
        # RU: read back by its own captured head index, before the ballot.
        _usable_lens = t[
            _usable_at : _usable_at + tp_head_congruence.TP_HEAD_SLOTS
        ].tolist()
        if len(_usable_lens) != tp_head_congruence.TP_HEAD_SLOTS:
            raise RuntimeError(
                "RU USABLE-MATCH LAYOUT STOP: the reduced payload carries no "
                f"usable-match slice (head={_usable_at}, "
                f"expected={tp_head_congruence.TP_HEAD_SLOTS}, "
                f"available={len(vals) - _usable_at})"
            )
        _usable_group = tp_match_floor.decode_group_usable(
            _head_canonical, _usable_lens
        )
        # H97 (rc9m pdflip-18-15): MIN < MAX means the deeper ranks must cut to
        # the group depth -- and may hold no anchor there. Every rank sees the
        # same skew (same reduce), so every rank takes the same branch: one
        # more small MIN over "I can realize the group depth", 0 -> group 0.
        _usable_skew = tp_match_floor.skewed_rids(
            _usable_group,
            tp_match_floor.decode_group_max(
                _head_canonical,
                t[
                    _usable_max_at : _usable_max_at + tp_head_congruence.TP_HEAD_SLOTS
                ].tolist(),
            ),
        )
        if _usable_skew:
            _realize = torch.tensor(
                tp_match_floor.build_realize_payload(
                    _head_canonical,
                    _usable_skew,
                    _usable_local,
                    getattr(self, "tree_cache", None),
                    _usable_by_rid,
                    tp_head_congruence.TP_HEAD_SLOTS,
                ),
                dtype=torch.int64,
            )
            torch.distributed.all_reduce(
                _realize, op=torch.distributed.ReduceOp.MIN, group=grp
            )
            _usable_group = tp_match_floor.apply_realize_verdict(
                _usable_group, _head_canonical, _usable_skew, _realize.tolist()
            )
        self._uniform_usable_floor = _usable_group
        _admit_limit = int(t[_limit_at])
        # #1203: read back by its captured head index, before the ballot, under
        # the same discipline as the corridor width and the head block.
        setattr(self, UNIFORM_SEAM_PREMISE_ATTR, int(t[_seam_premise_at]))
        # #1068 weg1 S0: read back by its captured head index, BEFORE the ballot
        # slice, under the same discipline as the corridor width, the head block
        # and the seam premise. `unpack_phase_domain` RAISES on a disagreement
        # -- the (x, -x) pairs make the group's min and max visible to every
        # rank of this reduce, so all of them raise on the same pass -- and
        # returns None only for a slice of the WRONG WIDTH, which is a layout
        # defect and stops here too.
        _phase_domain_verdict = phase_domain_verdict.unpack_phase_domain(
            t[
                _phase_domain_at : _phase_domain_at
                + phase_domain_verdict.PHASE_DOMAIN_SLOTS
            ].tolist(),
            rank=int(self.ps.tp_rank),
            local=phase_domain_verdict.local_terms_from_payload(_phase_domain_local),
            world_size=int(torch.distributed.get_world_size(grp)),
            phase=_phase_domain_phase,
        )
        if _phase_domain_verdict is None:
            raise RuntimeError(
                "#1068 PHASE-DOMAIN LAYOUT STOP: the reduced payload carries no "
                f"phase-domain slice (head={_phase_domain_at}, "
                f"expected={phase_domain_verdict.PHASE_DOMAIN_SLOTS}, "
                f"available={len(vals) - _phase_domain_at})"
            )
        _ballot_at = len(vals) - (prefetch_ballot.PREFETCH_BALLOT_SLOTS + 2)
        # #1158 / raenge-nie-uneins: A DIGEST MISMATCH IS A GROUP STOP. The
        # (x, -x) pair makes the group's min and max digest visible to EVERY
        # rank of this reduce, so `unpack_prefetch_ballot` raises on all of
        # them on the FIRST diverged pass, and the process dies through
        # run_scheduler_process's except -> SIGQUIT -> kill_process_tree. No
        # void ballot, no fallback to the rank-local verdict, no streak: on
        # boot weg1b3 that fallback ran 32 consecutive void passes
        # (23:56:17 -> 23:59:52) until the rank-local verdicts split and
        # PP0/PP2 formed a batch PP1 never joined. `None` is reserved for the
        # callers that take no ballot (single rank, PP loop); in THIS loop it
        # is a layout defect and stops too.
        self._uniform_prefetch_ballot = prefetch_ballot.unpack_prefetch_ballot(
            t[_ballot_at:].tolist(),
            _ballot_rids,
            rank=int(self.ps.tp_rank),
            queue_len=len(self.waiting_queue),
        )
        if self._uniform_prefetch_ballot is None:
            raise RuntimeError(
                "#791b PREFETCH-BALLOT LAYOUT STOP: the reduced payload carries "
                f"no ballot slice (slice_len={len(vals) - _ballot_at}, "
                f"expected={prefetch_ballot.PREFETCH_BALLOT_SLOTS + 2})"
            )
        # #823 W9b: ONE VALUE, NOT FOUR ATTRIBUTES. Four attributes published
        # here and read separately over there is what let one of them --
        # #791b's consume-once ballot -- be destroyed between publication and
        # use while the other three stayed live, and the batch formed
        # rank-locally on all 105 passes of boot_window2_0823_1554 without
        # anything noticing that three quarters of the verdict was intact.
        # Bundled, the verdict is taken once by `_take_uniform_head_inputs`
        # and handed to both consumers; there is no window in which it is
        # partly present.
        self._uniform_head_inputs = tp_head_congruence.build_uniform_head_inputs(
            _head_canonical,
            _head_match_lens,
            _admit_limit,
            self._uniform_prefetch_ballot is not None,
            _xpend_lens,
            _xstore_lens,
        )
        self._uniform_min_avail = int(t[0].item())
        # >= 0: the local budget can only exceed the group minimum.
        self._uniform_budget_deficit = (
            local_admission - int(t[1].item()) if pin_admission else 0
        )
        # Indices are explicit rather than negative, and now ALL of them are.
        # #639b appended the mamba pair after the host pair, so the `t[-2]` /
        # `t[-1]` the host floor used to read would have silently started
        # reading MAMBA availability -- the exact hazard the previous revision
        # of this comment named. Layout, in payload order:
        #   [avail, (admission,) -avail, host, -host, mamba, -mamba]
        max_avail_at = 2 if pin_admission else 1
        host_at = 3 if pin_admission else 2
        mamba_at = 5 if pin_admission else 4
        self._publish_uniform_evict_floor(
            int(t[0].item()), max_avail=-int(t[max_avail_at].item())
        )
        # F1 (nf-next-1006-01): the cap-blind gap, from the REDUCED values only.
        publish_cap_blind_gap(
            getattr(self, "tree_cache", None),
            allocator=getattr(self, "token_to_kv_pool_allocator", None),
            pin_admission=pin_admission,
            min_avail=int(t[0].item()),
            min_admission=int(t[1].item()) if pin_admission else 0,
            local_evict_full=local_evict_full,
            local_cap_above=local_cap_above,
        )
        self._publish_uniform_host_floor(
            int(t[host_at].item()), max_host_avail=-int(t[host_at + 1].item())
        )
        self._publish_uniform_mamba_floor(
            int(t[mamba_at].item()), max_mamba_avail=-int(t[mamba_at + 1].item())
        )

    def _local_seam_premise_vote(self) -> int:
        """#1203 (family A1): this rank's vote on the seam-transport premise.

        THE PULL-FORWARD, the same one #791b made for the prefetch verdicts
        and #823 W9 for the head matches: the quantity is rank-local and takes
        no collective of its own, this site runs exactly once per TP-loop
        iteration, and the reduce it rides already runs.

        NOT MEMOISED FOR THE READ SITE. `seam_transport_premise_holds` runs its
        own local check again and ANDs it with the reduced value, so this vote
        can only ever tighten the group's answer, never license one. That costs
        a second walk of the seam candidates per pass and buys immunity to the
        #823 W9b failure -- a published verdict consumed, destroyed or outliving
        its pass while every log line still said "group".
        """
        from flliper.srt.managers.phase_purity import local_seam_premise_vote

        return int(local_seam_premise_vote(self))

    def _tp_head_enforcer_gate(self) -> tp_head_congruence.GateVerdict:
        """#823 W9: is the group's batch-formation decision in force, and WHY.

        #823 W9b: this used to return a bare ``bool`` and say nothing. That
        is how the enforcer spent a whole GPU window unable to answer its own
        criterion: on ``--tp-size 1 --pp-size 3`` it was off for 45 of the
        boot's 51 prefill batches, and the log could not distinguish "gated
        off, correctly, because the PP phase is #791's territory" from
        "wired and inert". Those two need different fixes and the boot
        offered no way to tell them apart.

        The decision itself is pure and lives in ``tp_head_congruence``,
        including the three code-anchored reasons the TP world of one is NOT
        widened to cover the PP phase. ``self.ps`` is read with getattr, not
        just its members: this predicate sits on the admission path and is
        reached by harnesses that model a Scheduler without a ParallelState.
        """
        ps = getattr(self, "ps", None)
        verdict = tp_head_congruence.enforcer_gate(
            os.environ.get("FLLIPER_TP_HEAD_CONGRUENCE", "1") == "1",
            None if ps is None else int(getattr(ps, "tp_size", 1) or 1),
            None if ps is None else int(getattr(ps, "pp_size", 1) or 1),
        )
        # Transition-reported, not latched: under --enable-phase-flip this
        # gate switches at every cutover, so a once-per-process line would
        # describe the first phase and never the run. Same rule as #824's
        # uniform_floor_scope.report_scope, one line per change of state.
        self._tp_head_gate_reason, _message = tp_head_congruence.advance_gate_report(
            getattr(self, "_tp_head_gate_reason", None), verdict
        )
        if _message is not None:
            logger.info("#823 head-congruence enforcer %s", _message)
        return verdict

    def _tp_head_enforcer_enabled(self) -> bool:
        """The gate as a bare predicate, for call sites that only branch."""
        return self._tp_head_enforcer_gate().enabled

    def _note_tp_head_degradation(self, arm: str, degraded: bool) -> None:
        """#823 W9b: a rank-local formation under an ARMED enforcer is a DEFECT.

        THE COUNT ARM HAD NO LINE AT ALL. ``_uniform_allocatable_reqs`` read
        its memo through ``getattr(..., None)`` and fell back to the local
        number in silence, so boot_window2_0823_1554 could report the ORDER
        arm's inertness 105 times and say nothing whatsoever about the COUNT
        arm -- and a future boot showing ZERO #823 lines would still say
        nothing about it. Absence of a warning was not evidence of
        enforcement; it was the absence of an instrument. So each arm gets a
        named counter, and a counter is what a boot-log grep can actually
        assert on.

        NOT UNBOUNDED LOGGING, and the same specimen is the reason the
        cadence is borrowed rather than invented: that boot emitted 7710
        #797d/#798 void lines in seven seconds. ``advance_mismatch_streak``
        is geometric-then-capped, so a persistent degradation costs a handful
        of lines per minute -- and it reports the RECOVERY EDGE, which is the
        half a latch can never report ("it healed after N passes" and "it
        never healed" are the same silence to a latched logger).
        """
        _streak_attr = f"_tp_head_degrade_streak_{arm}"
        _total_attr = f"_tp_head_degrade_total_{arm}"
        streak, total, event = prefetch_ballot.advance_mismatch_streak(
            getattr(self, _streak_attr, 0),
            getattr(self, _total_attr, 0),
            bool(degraded),
        )
        setattr(self, _streak_attr, streak)
        setattr(self, _total_attr, total)
        if event is None:
            return
        kind, run = event
        if kind == "diverged":
            logger.warning(
                "#823 head-congruence DEGRADED: the %s arm formed rank-locally "
                "while the enforcer is ARMED. This is the pre-#823 behaviour "
                "the ticket exists to remove, not a fallback. Consecutive "
                "degraded passes: %d (total this process: %d).",
                arm,
                run,
                total,
            )
        else:
            logger.info(
                "#823 head-congruence RESTORED: the %s arm is group-uniform "
                "again after %d degraded pass(es) (total this process: %d).",
                arm,
                run,
                total,
            )

    def _uniform_allocatable_reqs(
        self,
        running_bs: int,
        head_inputs: Optional[tp_head_congruence.UniformHeadInputs],
        slot_held: int = 0,
    ) -> int:
        """#823 W9 COUNT arm: how many of the head this rank may admit.

        The group's MIN, so every rank stops at the same candidate count.
        Falls back to the local number whenever the group has no opinion,
        which is the pre-#823 expression unchanged.

        #823 W9b: ``head_inputs`` IS A REQUIRED PARAMETER, and that is the
        fix rather than a style preference. It used to read
        ``getattr(self, "_uniform_admit_limit", None)`` -- and a
        getattr-default cannot tell "the group priced this pass at zero"
        apart from "nobody published anything and this arm is inert", so the
        arm degraded to rank-local without a single line in a 74-second boot
        that logged 105 lines about its sibling. A parameter has no default
        to hide behind: the caller must have the decision in hand, and if it
        does not, the absence is a value this method can COUNT (see
        ``_note_tp_head_degradation``) instead of a hole it cannot see.

        Called once per candidate in the admission loop, so the degradation
        is recorded on the pass's FIRST degraded evaluation only -- one
        accounting event per pass, matching the order arm, rather than one
        per queued request.
        """
        local = (
            self.get_num_allocatable_reqs(running_bs, slot_held)
            if slot_held
            else self.get_num_allocatable_reqs(running_bs)
        )
        gate = self._tp_head_enforcer_gate()
        limit, source = tp_head_congruence.admit_limit_decision(
            local,
            None if head_inputs is None else head_inputs.admit_limit,
            gate.enabled,
        )
        if tp_head_congruence.degradation_is_a_defect(gate.enabled, source):
            if not self._tp_head_count_degraded_this_pass:
                self._tp_head_count_degraded_this_pass = True
                self._note_tp_head_degradation(tp_head_congruence.ARM_COUNT, True)
        return local if limit is None else int(limit)

    def _take_uniform_head_inputs(
        self,
    ) -> Optional[tp_head_congruence.UniformHeadInputs]:
        """#823 W9b: take this pass's group verdict ONCE, at one point.

        THE LIFECYCLE IS THE DEFECT THIS REPLACES, so it is worth stating
        exactly. The verdict used to live as four separate attributes
        published by ``_update_uniform_pool_budget`` and read, later and
        separately, by the two consumers. One of the four --
        ``_uniform_prefetch_ballot`` -- is ALSO #791b's consume-once memo,
        and ``_get_new_batch_prefill_raw`` pops it off ``__dict__`` fifty-odd
        lines before the order arm read it directly. Pop, then read: the read
        raised ``AttributeError`` on every pass, the handler caught it, and
        the batch formed rank-locally. 105 out of 105 passes, all three
        ranks, boot_window2_0823_1554.

        THE OBVIOUS REPAIR IS THE ONE NOT TAKEN. Giving that read a
        ``getattr(..., None)`` like its three neighbours stops the exception
        and produces the count arm's failure instead -- inert, and silent
        about it. That is the #606 getattr-default family: a default that
        converts a structural defect into a plausible number. So the four
        attributes become ONE value, taken here, handed to both consumers as
        a parameter. A parameter cannot be missing at the point of use, which
        is a stronger guarantee than any default can offer.

        CONSUME-ONCE IS KEPT, with a defined reset point: the pop is what
        stops a PP-loop pass -- which never runs the budget site -- from
        forming on a stale TP-loop verdict, and the reset point is the next
        publication by ``_update_uniform_pool_budget``. What changes is that
        the take happens BEFORE any consumer instead of between two of them,
        and that an absent verdict under an ARMED enforcer is counted and
        logged rather than silently absorbed.
        """
        # Reset the per-pass accounting flag for the count arm, which is
        # evaluated once per candidate and must still report once per pass.
        # The ORDER arm accounts for itself: it runs exactly once per pass
        # and can degrade for reasons this point cannot see (an empty
        # canonical head, a decision that came back rank-local), so
        # accounting here as well would double-count the streak and make the
        # cadence report a divergence twice as long as it was.
        self._tp_head_count_degraded_this_pass = False
        return self.__dict__.pop("_uniform_head_inputs", None)

    def _head_order_arrival_seqs(self) -> Optional[Dict[str, int]]:
        """FIX 3 (round 3): the ORDER arm's key, or None to keep the old one.

        LAW 2 OF WEG2_SCHEDULING_SPEC_0907 ("D admits OLDEST-first ... refills
        a freed seat from the arrival order") IS BROKEN AT D BY #823's ORDER
        ARM, and R-14's justification ("append order IS admission order,
        because calc_priority does not sort under FCFS") missed this method:
        it runs immediately AFTER `calc_priority` and rewrites the head
        whatever the policy did.  Measured at the parent commit for arrival
        order 1,2,3: `['pdflip-1-2', 'pdflip-1-3', 'pdflip-1-1']`.

        THE RULE, and it is general rather than Weg-2 special-casing: the
        ORDER arm exists to replace the divergence born in
        `_sort_by_longest_prefix` (schedule_policy.py:225-232, reachable only
        under a CacheAwarePolicy).  Under a cache-AGNOSTIC policy that sort
        never runs, so there is nothing to replace and a prefix-length
        re-sort here is a reordering the policy never asked for.  The key is
        then `kv_arrival_seq` -- assigned once per request in
        `_add_request_to_queue` (:5247) off the same broadcast stream on every
        rank, hence rank-uniform (phase_flip_runtime.py:9689), so #823's
        uniformity requirement is met exactly as well as by the MIN-reduced
        match.  What is given up is a cache-locality heuristic that this arm,
        under this policy, was never asked to provide.

        THE X FLAG FORCES IT TOO, whatever the policy: a group serving Weg 2's
        law 2 may not have its admission order rewritten by a cache
        heuristic, and `--tp-prefill-max-tokens` is that group's own marker
        (it reaches argv_d and never argv_p).

        None -- the pre-FIX-3 match-length key -- under a CacheAwarePolicy on
        a non-Weg-2 boot, where the sort DID run and this arm is replacing it.
        """
        # DEFENSIVE, and for the same reason `_local_admit_limit` is: a KEY
        # may never break the arm it feeds. Every read is a `getattr` and the
        # whole derivation is caught, because `_apply_uniform_head_order`
        # turns any exception below it into the #823 HEAD-ORDER APPLY STOP --
        # a group STOP that would then name the wrong cause. No key means the
        # pre-FIX-3 behaviour, which is a valid order, never a divergence.
        try:
            server_args = getattr(self, "server_args", None)
            pdflip = int(getattr(server_args, "tp_prefill_max_tokens", 0) or 0) > 0
            if not pdflip:
                policy = getattr(getattr(self, "policy", None), "policy", None)
                if isinstance(policy, CacheAwarePolicy):
                    return None
            seqs: Dict[str, int] = {}
            for req in getattr(self, "waiting_queue", ()) or ():
                seq = getattr(req, "kv_arrival_seq", None)
                if seq is not None:
                    seqs[str(req.rid)] = int(seq)
            # A VACUOUS KEY IS NOT A KEY. An empty mapping would order the
            # whole head by rid string (every rid lands in the "no arrival
            # rank" bucket), which is neither arrival order nor the match
            # order -- a third order nobody asked for. `kv_arrival_seq` is
            # assigned unconditionally in `_add_request_to_queue` (:5320) and
            # survives a retraction, so an empty mapping means a queue of
            # requests that never entered through it, i.e. a stand-in.
            return seqs or None
        except Exception:  # noqa: BLE001 - a key may never break the order arm
            return None

    def _apply_uniform_head_order(
        self, head_inputs: Optional[tp_head_congruence.UniformHeadInputs]
    ) -> None:
        """#823 W9 ORDER arm: put ``waiting_queue``'s head in the GROUP's order.

        Runs immediately after ``calc_priority``, so it overrides exactly the
        rank-local `_sort_by_longest_prefix` result that diverges
        (schedule_policy.py:225-232) and nothing else. Requests the group did
        not name -- deeper than the head, or not held everywhere -- keep
        their relative order behind it, so this REORDERS and never drops:
        a request the group omitted waits for a later pass rather than
        disappearing.
        """
        gate = self._tp_head_enforcer_gate()
        source = tp_head_congruence.SOURCE_RANK_LOCAL
        try:
            if gate.enabled and head_inputs is not None:
                # #823 W9b: NOTHING IS READ OFF `self` HERE ANY MORE. The
                # group's verdict arrives as a value taken once at the top of
                # the pass, so there is no attribute left for the pass's own
                # consume-once pop to have removed -- which is precisely what
                # `digest_agreed=self._uniform_prefetch_ballot is not None`
                # did on every one of boot_window2_0823_1554's 105 passes.
                #
                # `digest_agreed` is passed for the record, not for the
                # decision: `head_decision` deliberately does not branch on
                # it, because the group order depends only on the canonical
                # rid SET and the MIN, and is therefore still computable in
                # exactly the pass where the digests disagree. That is change
                # 2 of this ticket, and it is why an unreadable digest must
                # never have been able to void the order in the first place.
                order, source = tp_head_congruence.head_decision(
                    head_inputs.canonical,
                    head_inputs.group_match_lens,
                    [req.rid for req in self.waiting_queue],
                    {},
                    digest_agreed=head_inputs.digest_agreed,
                    enforcer_enabled=True,
                    arrival_seqs=self._head_order_arrival_seqs(),
                )
                by_rid = {req.rid: req for req in self.waiting_queue}
                head = [by_rid[rid] for rid in order if rid in by_rid]
                named = {id(req) for req in head}
                self.waiting_queue[:] = head + [
                    req for req in self.waiting_queue if id(req) not in named
                ]
        except Exception as exc:  # noqa: BLE001 - re-raised as the named STOP
            # #1158 sibling of the ballot fallback: a group order this rank
            # cannot apply used to degrade to a RANK-LOCAL formation under an
            # ARMED enforcer -- the divergence class the enforcer exists to
            # end, delivered silently. A detected inability to follow the
            # group is a STOP, never a compensation (raenge-nie-uneins).
            raise RuntimeError(
                "#823 HEAD-ORDER APPLY STOP: the group head order could not be "
                f"applied on this rank ({exc!r}); a rank-local formation under "
                "an ARMED enforcer is the divergence class #1158 ends"
            ) from exc
        # Counted on EVERY pass, degraded or not: the recovery edge is the
        # half a latch can never report, and "it healed" is the answer the
        # window criterion actually needs.
        self._note_tp_head_degradation(
            tp_head_congruence.ARM_ORDER,
            tp_head_congruence.degradation_is_a_defect(gate.enabled, source),
        )

    def _local_admit_limit(self, running=None) -> Optional[int]:
        """#823 W9: this rank's vote for HOW MANY of the head may be admitted.

        Its own slot rather than a derivation from the availability floor,
        because ``get_num_allocatable_reqs`` is bounded by
        ``admission_limiter.current`` -- rank-local floating state the
        availability reduce does not capture. ``None`` means "no opinion" and
        rides as the sentinel, leaving every rank's local limit untouched.

        W-SEAT (29.09., z30x2 base 11:48:10Z): ``running`` is THE PASS'S
        running batch, handed in by ``get_next_batch_to_run``. Its merge
        rebinds the LOCAL ``running_batch`` to the just-finished extend batch
        when the running batch was empty (``running_batch = last_batch``);
        ``self.running_batch`` still names the old empty batch until the
        event loop stores the plan. Read from ``self`` the vote counted 0
        running while the admission loop counted 1, and the COUNT arm takes
        the group vote as the limit (``admit_limit_decision``): a D phase of
        n=2 seats admitted two more beside the merged one and died on
        ``PdFlipDSeatOverrun`` (a batch of 3).
        """
        try:
            if running is None:
                running = getattr(self, "running_batch", None)
            running_bs = len(running.reqs) if running is not None else 0
            held = self._chunk_rest_slot_held(running)
            if held:
                return int(self.get_num_allocatable_reqs(running_bs, held))
            return int(self.get_num_allocatable_reqs(running_bs))
        except Exception:  # noqa: BLE001 - a vote may never break the reduce
            return None

    def _publish_uniform_evict_floor(
        self, min_avail: Optional[int], max_avail: Optional[int] = None
    ) -> None:
        """Publish this iteration's rank-uniform availability floor on the tree
        cache, for the CACHE-MUTATION triggers to decide from (#616g).

        WHY, measured: #603 and #610 made the decode-mem and prefill-admission
        DECISIONS rank-uniform, but both left eviction as an explicitly local
        side effect ("Still evict locally for the space"). Under uneven pools
        that side effect is itself a divergence source, one level below the
        decisions:

          `evict_from_tree_cache` fires on `available_size() < num_tokens`,
          where the DEMAND is replicated (`batch.extend_num_tokens`) but the
          AVAILABILITY is this rank's own shard -- 179825 / 143860 / 136667
          tokens on the 2026-08-06 boot. The rank with the roomiest pool
          declines to evict while the tight ranks evict, so the radix trees
          stop being replicas. `match_prefix` then returns a rank-dependent
          prefix, `prepare_for_extend` computes
          `extend_num_tokens = sum(seq_len - len(prefix_indices))` from it,
          and every per-layer TP all_reduce of that forward is entered with a
          rank-dependent token count. That is the 21:52:25 wedge exactly:
          all three ranks parked in the SAME collective at the SAME layer_idx
          with rank 0 reducing 1690 tokens against 1818 on its peers.

        The floor is the MIN over ranks of the same quantity, taken from the
        reduce that already runs once per iteration, pre-branch -- NO new
        collective and no new synchronisation point.

        DIRECTION IS SAFE BY CONSTRUCTION: min <= local, so
        `floor < num_tokens` is true whenever the local test was, and
        sometimes when it was not. Every rank therefore evicts at least as
        often as before the fix; under-eviction (which would turn into an
        allocator OOM) is arithmetically impossible. The price is that slack
        ranks drop cache they did not personally need, which is what keeping
        the replicas identical costs.

        ACTIVATION is itself collective-derived: the floor is published only
        when the group's min and max availability DIFFER, i.e. when the pools
        are actually uneven this iteration. On even TP the two are equal, the
        floor stays None, and every trigger reads its live local value exactly
        as before -- byte-identical, no behaviour change on the default path.
        """
        tree = getattr(self, "tree_cache", None)
        if tree is None:
            return
        # SF (b23 #1004): a floor published here is a GROUP value (or None);
        # the single-rank path re-marks it as local right after this call
        # (pdflip/pp_slot_fidelity.py). Switch off: nothing is written.
        from flliper.srt.pdflip import pp_slot_fidelity as _sf

        _sf.mark_floor_scope(tree, False)
        # #1045 OPTION A: THE FLOOR IS PUBLISHED UNCONDITIONALLY ON A GROUP.
        #
        # It used to stay None whenever the pools happened to AGREE this
        # iteration, and that conditionality was load-bearing in the wrong
        # direction: `unified_radix_cache`'s load-back memory check falls into a
        # RANK-LOCAL branch when the floor is None, and its only remaining guard
        # there was the delivery signal `pp_load_back_told`. Tying a
        # correctness watchman on the MEMORY axis to a tie in availability meant
        # the watchman stood down exactly when nothing looked wrong.
        #
        # PRICE: ZERO NEW COLLECTIVE. Both `min_avail` and `max_avail` are
        # harvested from the SAME already-performed all-reduce at :6396
        # (`t[0]` and `t[max_avail_at]`), so dropping the comparison removes
        # work rather than adding it. Verified before building, per the
        # standing "no new per-pass collective without a price report" rule.
        #
        # BEHAVIOURALLY IDENTICAL ON EVEN POOLS: when the pools agree,
        # `min_avail` IS this rank's live availability, so every trigger reads
        # the same number it read from the local path before -- the docstring's
        # own "byte-identical" claim, now reached by publishing the equal value
        # instead of by publishing nothing. `uniform_admitted_since_floor` is
        # the #694 ledger that keeps a published floor tracking allocations, so
        # the equivalence holds through the iteration and not only at the
        # instant of publication.
        #
        # `min_avail is None` remains the ONE None case: the single-rank path
        # (:6082, guarded by `grp is None or world_size <= 1`), where there is
        # no peer to diverge from.
        if min_avail is None:
            tree.uniform_avail_floor = None
            # #694: the ledger corrects a floor; with no floor there is nothing
            # to correct, and a value left over from a previous generation
            # would be charged against the NEXT one.
            tree.uniform_admitted_since_floor = 0
            return
        tree.uniform_avail_floor = int(min_avail)
        # #1045 LIVENESS: a floor that is published on every iteration must be
        # SEEN to be published on every iteration, or "the watchman is armed" is
        # an assumption again. Unconditional first, then sampled WITH the
        # suppression count printed -- a zero from a rate-limited emitter is not
        # a zero (this campaign's fourth denominator instance, now a rule).
        _n = getattr(self, "_1045_floor_pub", 0) + 1
        self._1045_floor_pub = _n
        _eq = max_avail is not None and int(min_avail) >= int(max_avail)
        if _n <= 3 or _n % 512 == 0:
            logger.info(
                "#1045 FLOOR PUBLISHED n=%d floor=%d max=%s pools_equal=%s -- "
                "published unconditionally on a group; the memory-axis watchman "
                "no longer stands down on a tie. (sampled every 512 after the "
                "third; suppressed since last print: %d)",
                _n,
                int(min_avail),
                max_avail,
                _eq,
                0 if _n <= 3 else 511,
            )
        # #694: RESET IN THE SAME CALL THAT PUBLISHES, so the ledger never
        # outlives the number it corrects. The floor is a snapshot of this
        # instant; allocations charged against the PREVIOUS snapshot have
        # already been reflected in this new MIN, and charging them twice would
        # drive the predicate to zero and evict on every allocation.
        tree.uniform_admitted_since_floor = 0

    def _local_host_avail(self) -> int:
        """This rank's free HOST-tier slots, or ``_HOST_AVAIL_ABSENT`` (#639).

        Read through the tree cache's controller because that is the object
        ``write_backup`` reads it from -- the pin and the gate must sample the
        same pool or the floor would describe a different tier than the one it
        governs.
        """
        tree = getattr(self, "tree_cache", None)
        controller = getattr(tree, "cache_controller", None) if tree else None
        pool = getattr(controller, "mem_pool_host", None) if controller else None
        if pool is None:
            return self._HOST_AVAIL_ABSENT
        try:
            return int(pool.available_size())
        except Exception:  # noqa: BLE001
            # A diagnostic must never be what turns a scheduler iteration into
            # a crash; an unreadable pool means "no floor", i.e. the pre-#639
            # local behaviour.
            return self._HOST_AVAIL_ABSENT

    def _publish_uniform_host_floor(
        self,
        min_host_avail: Optional[int],
        max_host_avail: Optional[int] = None,
    ) -> None:
        """Publish this iteration's rank-uniform HOST availability floor on
        the tree cache, for ``write_backup`` to decide from (#639).

        WHY, measured. #616g pinned the DEVICE-side mutation triggers and
        explicitly left the host tier alone, on the reading that under
        ``write_through`` "the eviction path gates on write_policy ==
        'write_back', not on backup state, so the chain to the device tree is
        NOT established". ``UnifiedRadixCache._evict_device_leaf`` gates on
        ``node.backuped`` FIRST and consults ``write_policy`` only inside that
        branch -- and its write_through arm is the one that removes the node
        from the tree entirely. So the chain IS established, and it exists
        only on the policy this deployment runs:

          ``write_backup`` refuses when this rank's own host pool cannot hold
          the node (359652 / 287722 / 273336 slots on the crashing boot,
          against a REPLICATED node length). The roomy rank backs the node up
          and demotes it on eviction -- it stays in the tree, matchable, and
          ``load_back`` can restore it. The tight ranks refuse and DELETE it.
          ``match_prefix`` then returns a rank-dependent ``prefix_indices``,
          ``prepare_for_extend`` turns that into a rank-dependent
          ``extend_num_tokens``, and every per-layer TP all_reduce of that
          forward is entered with a shape that cannot pair. Four specimens,
          all with the roomiest rank LOW: 1690 vs 1818, 828 vs 2048, 912 vs
          2048, 914 vs 2048.

        The floor is the group MIN of the host ``available_size()``, taken
        from the reduce that already runs once per iteration, pre-branch --
        NO new collective on the default path.

        DIRECTION IS SAFE BY CONSTRUCTION: min <= local, so the pinned gate
        refuses whenever the local gate refused and sometimes when it did not.
        Over-admission into the host pool -- the only way this pin could
        itself become a fault -- is arithmetically impossible. The price is
        that the rank with the roomiest host tier caches less than it could,
        which is what keeping the replicas identical costs.

        ACTIVATION is collective-derived, exactly as the device floor's is:
        published only when the group's min and max host availability differ.
        When they agree, the local test compares the same left side against a
        replicated right side on every rank and is already uniform, so there
        is nothing to fix and the path stays byte-identical.
        """
        tree = getattr(self, "tree_cache", None)
        if tree is None:
            return
        absent = self._HOST_AVAIL_ABSENT
        if (
            min_host_avail is None
            or min_host_avail >= absent
            or (max_host_avail is not None and min_host_avail >= max_host_avail)
        ):
            tree.uniform_host_avail_floor = None
            tree.uniform_host_admitted_since_floor = 0
            tree.uniform_host_refusals_since_floor = 0
            return
        tree.uniform_host_avail_floor = int(min_host_avail)
        # #645: the floor above is a snapshot taken at the TOP of the
        # iteration, but the backups that read it run at the END of it
        # (`process_batch_result_prefill` -> `cache_unfinished_req` ->
        # `insert` -> `write_backup`). Without a ledger it keeps counting as
        # free every slot this iteration's own backups have already taken, so
        # the rank whose pool IS the floor runs out for real while its peers,
        # reading the same optimistic number, sail on -- and the tight rank
        # then takes a rank-LOCAL host-eviction branch its peers do not.
        # Reset here, in the same call that publishes the number it corrects,
        # so the ledger can never outlive its floor. Every path through
        # `_update_uniform_pool_budget` reaches this publisher exactly once
        # per iteration, so the reset inherits that cadence for free.
        tree.uniform_host_admitted_since_floor = 0
        # Same cadence for the refusal counter, which only exists to keep the
        # #645 warning to one line per published floor instead of one per
        # insert.
        tree.uniform_host_refusals_since_floor = 0

    def _local_mamba_avail(self) -> int:
        """This rank's free MAMBA slots, or ``_MAMBA_AVAIL_ABSENT`` (#639b).

        Read off ``req_to_token_pool.mamba_allocator`` because that is the
        allocator every gated call site allocates from -- the pin and the
        gates must sample the same pool or the floor would describe a
        different pool than the one it governs.

        ``available_size()`` and not ``schedulable_available_size()``: the
        raw free-slot count is defined identically on both allocator classes,
        while the schedulable view is the LARGER of the two on the shared
        ``UnifiedMambaSlotAllocator``. Reducing the smaller, universally
        defined quantity is what lets ``uniform_mamba_avail_for_evict`` stay
        direction-safe for callers reading either one.
        """
        pool = getattr(self, "req_to_token_pool", None)
        alloc = getattr(pool, "mamba_allocator", None) if pool else None
        if alloc is None:
            return self._MAMBA_AVAIL_ABSENT
        try:
            return int(alloc.available_size())
        except Exception:  # noqa: BLE001
            # A diagnostic must never be what turns a scheduler iteration into
            # a crash; an unreadable pool means "no floor", i.e. the pre-#639b
            # local behaviour.
            return self._MAMBA_AVAIL_ABSENT

    def _publish_uniform_mamba_floor(
        self,
        min_mamba_avail: Optional[int],
        max_mamba_avail: Optional[int] = None,
    ) -> None:
        """Publish this iteration's rank-uniform MAMBA availability floor on
        the tree cache, for the mamba eviction triggers to decide from (#639b).

        WHY, measured. The two floors above pin the KV token axis on the
        device and on the host. Neither reaches the mamba slot pool, and the
        2026-08-07 07:45 and 10:04 crashes are that gap:

            rank 0: n=1 sum=19711 has_prefix=True [19711]
            rank 1: n=1 sum=16957 has_prefix=True [16957]

        ``alloc_req_slots`` evicts on
        ``mamba_available_size < num_reqs * factor`` -- a REPLICATED demand
        against this rank's OWN occupancy -- and then evicts
        ``mamba_state_needed - mamba_available_size`` slots, a magnitude that
        is itself rank-local. ``MambaComponent.evict_component`` tombstones
        the mamba component of a node and LEAVES ITS KV, so the node stays in
        the tree on both ranks but stops satisfying the mamba validator on
        the rank that evicted. ``_match_prefix_helper`` advances only while
        every component is resident, so that rank's match stops short, its
        extend covers more tokens, it takes more slots, and it evicts more
        next round -- which is how the two ranks got 2754 tokens apart.

        A pre-existing comment in ``MambaRadixCache._alloc_mamba_slot``
        concluded this path was "rank-uniform without a collective" because
        ``max_mamba_cache_size`` is min-reduced at startup. That reasoning
        does not hold and is corrected in this change: a uniform pool SIZE is
        not a uniform eviction OUTCOME. Which node is tombstoned comes from
        ``drive_eviction``'s rank-local LRU walk, and occupancy diverges from
        lock_ref history and from the rank-local "skip this cache insert"
        degradation that an exhausted pool triggers.

        The floor is the group MIN of the mamba allocator's
        ``available_size()``, taken from the reduce that already runs once per
        iteration, pre-branch -- NO new collective.

        DIRECTION IS SAFE BY CONSTRUCTION: min <= local, so every rank evicts
        at least as often, and at least as much, as before. Under-eviction --
        which would turn into mamba slot starvation -- is arithmetically
        impossible. The price is that the rank with the roomiest mamba pool
        drops checkpoints it did not personally need.

        ACTIVATION is collective-derived, exactly as both siblings are:
        published only when the group's min and max mamba availability
        DIFFER. When they agree, every rank's local test already compares the
        same two numbers and the path stays byte-identical.
        """
        tree = getattr(self, "tree_cache", None)
        if tree is None:
            return
        absent = self._MAMBA_AVAIL_ABSENT
        if (
            min_mamba_avail is None
            or min_mamba_avail >= absent
            or (max_mamba_avail is not None and min_mamba_avail >= max_mamba_avail)
        ):
            tree.uniform_mamba_avail_floor = None
            return
        tree.uniform_mamba_avail_floor = int(min_mamba_avail)

    def uniform_budget_deficit(self) -> int:
        """This iteration's non-negative prefill-admission surplus over the
        group minimum, for `PrefillAdder` to subtract (#610).

        0 when the pin is not active (even TP, world size 1) and 0 when the
        per-iteration value was never computed -- the same fallback shape
        `KVSessionOffload.dcp_budget_deficit` documents, so an entry point that
        reaches prefill without passing `_update_uniform_pool_budget` (the PP
        mixin's own loop) keeps today's behaviour rather than reading a stale
        reduce from a differently-shaped path.
        """
        return int(getattr(self, "_uniform_budget_deficit", 0))

    def _capture_census_once(self, group, rank: int) -> None:
        """Diff the CAPTURED collective sequences across ranks. Exactly once.

        #603b. The #583 census above compares host-side collective COUNTS,
        and a captured collective is a host-side call exactly once in a boot
        -- at capture. Everything after that is replay, which makes no host
        call at all, so the count census is structurally blind to what a
        replayed graph does. That is where this crash family lives: the ranks
        are wedged inside a replayed decode graph, waiting on peer flags.

        A graph's behaviour is fixed when it is recorded. So if the ranks
        recorded different sequences, the mismatch is already present and
        readable at boot, minutes before the first wedge -- no reproduction
        needed. This asks that question once and logs the answer either way.

        Instrument discipline, as everywhere in this family: warn-never-raise.
        """
        if not capture_census_enabled():
            return
        cc = capture_census()
        if cc.compared:
            return
        try:
            path = cc.dump_to_file(rank)
            cc.compare_across_ranks(
                group, torch.distributed.get_world_size(group), rank
            )
            if path:
                logger.info("barlink capture census: per-rank record at %s", path)
        except Exception as exc:  # noqa: BLE001 - instrument must not raise
            # Mark it done regardless: a comparison that cannot run will not
            # start working on the next tick, and retrying it every interval
            # would turn one warning into a per-tick stream.
            cc.compared = True
            logger.warning(
                "barlink capture census: one-shot comparison unavailable (%s: %s)",
                type(exc).__name__,
                exc,
            )

    def uniform_min_avail(self) -> int:
        """This iteration's rank-uniform available_size.

        Falls back to the live local value ONLY on a single rank, where
        there is nothing to diverge from. On a group a missing reduce is
        refused loudly instead -- a decision must never read a value that
        silently means something else, and "this rank's own pool" is exactly
        that when the caller asked for the group's.
        """
        v = getattr(self, "_uniform_min_avail", None)
        if v is not None:
            return int(v)
        # #583 follow-up: refuse rather than silently reinstate the defect.
        #
        # The old fallback returned this rank's local available_size()
        # whenever the reduce had not run. On a single rank that is exactly
        # right. On a GROUP it is the rank-local predicate #603 removed,
        # restored silently by any path that reaches a decode-mem decision
        # before `_update_uniform_pool_budget` -- and a getattr default is
        # precisely the shape that hides such a path (#606): the guard reads
        # as present in the source while being absent at runtime.
        if self.ps.tp_size > 1:
            raise RuntimeError(
                "uniform_min_avail() was reached before "
                "_update_uniform_pool_budget() ran on a multi-rank boot. "
                "Returning this rank's local available_size() here would "
                "restore the rank-local decode-mem predicate #603 removed "
                "and split the ranks across branches carrying different "
                "collectives. The reduce is unconditional and pre-branch in "
                "get_next_batch_to_run; a caller reaching this line means a "
                "decode-mem decision escaped that ordering."
            )
        return int(self.token_to_kv_pool_allocator.available_size())

    def uniform_corridor_width(self):
        """The group's tightest corridor width this iteration, or None.

        None means the reduce did not run (world size 1, or the offload branch),
        in which case the local decision IS the group decision.
        """
        return getattr(self, "_uniform_corridor_width", None)
