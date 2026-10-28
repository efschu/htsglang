"""``Scheduler`` methods moved out of ``scheduler.py`` (refactor, behaviour-neutral): bodies are byte-identical,
the class stays reachable as ``Scheduler.<name>`` through the mixin. Logger name unchanged."""

from array import array
from flliper.srt.managers.io_struct import PdFlipMatchProbeReqOutput
from flliper.srt.managers.schedule_batch import Req
from flliper.srt.pdflip import store_reprobe_epoch as _store_reprobe_epoch
from flliper.srt.rank_role import this_rank_is_form_a_worker
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import logging
import time
import torch.distributed

from flliper.srt.managers.scheduler_pdflip_helpers import (  # noqa: F401
    _dmp_probe_locked_spans,
    _s64_note_store_arm,
    _s64_store_arm_fresh_on,
)

logger = logging.getLogger("flliper.srt.managers.scheduler")

_VERDICT_STORE_SHORT = "terminated:store_short"
_CAPACITY_CUT_CAUSES = (None, "arena_full", "l2_evicted_before_ref", "fill_evicted_before_ref")


class SchedulerStoreProbeMixin:
    """Methods moved out of ``Scheduler`` (see module docstring)."""

    def _pdflip_match_probe_one(self, rid, input_ids, mm_inputs, extra_key, tokenize_ms) -> PdFlipMatchProbeReqOutput:
        """S56/S56r: one probe of ``handle_pdflip_match_probe`` -- padding like the intake, then the READ-ONLY peek.

        27B NACHT 21.10. (D-PROBE GROUP-MIN, SOLL R11 + user order 19.10. "NICHTS SCHÄTZEN, EXAKTE ZAHLEN"): on an
        ASYMMETRIC host tier without Form A (27B D: TP3 over 3080/5090/3080, uneven DCP) the admission takes the RU MIN
        of every rank's usable match, which no single rank can read -- the probe used to refuse by name
        (``group_min_needed``), 22 of 22 times on the 27B INT8 metal 21.10. (rc29 ..._041a871f4b_1021_001910), and every
        arrival went through the measuring barrier (MEASURE_ON_D): the halb requests pdflip-4-40 / 4-42 were refused
        there, and on rc29b pdflip-0-1 / 0-4 (u 29484 / 29330, unpriced) sat unstarted on D through a D->P flip and came
        back as ``store_short_after_flip`` -> a second flip pair. Now every rank peeks its own tree (the same read-only
        walk) and the group reduces (ok, dev+host, dev, key depth) by MIN on the TP CPU group -- the group's number,
        exactly the admission's MIN. Rank-uniform by construction: the probe is a broadcast control request every rank
        handles at the same point of its intake; the mode is decided from launch facts (host tier asymmetric, Form A
        off) BEFORE any per-rank work, and every rank reaches the reduce even when its own padding or walk failed
        (ok=0 -> the group answers success=False by name, never one rank's number)."""
        from flliper.srt.managers.tp_match_floor import form_a_follow_active

        tc = getattr(self, "tree_cache", None)
        group_min = False
        if tc is not None:
            # ``_hicache_prefetch_symmetric`` is True when the host tiers are ASYMMETRIC (the name is historical)
            asym = bool(getattr(tc, "_hicache_prefetch_symmetric", lambda: False)())
            group_min = asym and not form_a_follow_active()
        if not group_min:
            return self._pdflip_match_probe_local(rid, input_ids, mm_inputs, extra_key, tokenize_ms)
        return self._pdflip_match_probe_group_min(rid, input_ids, mm_inputs, extra_key, tokenize_ms, tc)

    def _pdflip_match_probe_pad(self, rid, input_ids, mm_inputs, extra_key):
        """The probe's ids and image spans, padded like the intake (every rank: a broadcast mm process needs all)."""
        import types

        from flliper.srt.managers.io_struct import unwrap_from_pickle
        from flliper.srt.pdflip import d_match_probe as _dmp
        from flliper.srt.sampling.sampling_params import SamplingParams

        ids = array("q", input_ids or ())
        spans: List[List[int]] = []
        mm = unwrap_from_pickle(mm_inputs) if mm_inputs is not None else None
        if mm is not None:
            image_inputs = self._get_multimodal_inputs(mm)
            req = Req(
                str(rid),
                None,
                ids,
                SamplingParams(max_new_tokens=1),
                vocab_size=self.model_config.vocab_size,
                extra_key=extra_key,
            )
            _recv = types.SimpleNamespace(input_ids=input_ids)
            if (
                not self._try_apply_padded_mm_input_ids(_recv, req, image_inputs)
                and self.pad_input_ids_func
            ):
                req.origin_input_ids = array(
                    "q", self.pad_input_ids_func(req.origin_input_ids, image_inputs)
                )
            ids = array("q", req.origin_input_ids)
            spans = _dmp.mm_spans(image_inputs)
        return ids, spans

    def _pdflip_match_probe_group_min(self, rid, input_ids, mm_inputs, extra_key, tokenize_ms, tc
                                      ) -> PdFlipMatchProbeReqOutput:
        """D-PROBE GROUP-MIN (see ``_pdflip_match_probe_one``): this rank's peek, then the group MIN. Every rank reaches
        the reduce exactly once per probe, whatever its own work did."""
        from flliper.srt.pdflip import d_match_probe as _dmp

        t0 = time.perf_counter()
        ok, total, dev_l, key_l, err = 1, 0, 0, 0, ""
        ids: Any = array("q")
        spans: List[List[int]] = []
        locked, lk_ok, lk_tot, lk_fp = [], 0, 0, 0
        try:
            ids, spans = self._pdflip_match_probe_pad(rid, input_ids, mm_inputs, extra_key)
            n = len(ids)
            if n > 1:
                dev, host, key_l = _dmp.peek_admission_match(tc, ids, extra_key, n - 1)
                dev_l, total = int(dev), int(dev) + int(host)
                locked, lk_ok = _dmp_probe_locked_spans(tc, ids, extra_key, n - 1)
                lk_tot = sum(int(b) - int(a) for a, b in locked)
                lk_fp = sum(int(b) for _a, b in locked)   # review SOLL 3: the span ends, a fingerprint of WHERE
        except Exception as e:  # noqa: BLE001 -- named below; the reduce is still entered
            ok, err = 0, f"{type(e).__name__}: {str(e)[:160]}"
        # #215b (SOLL R11): the locked device rows ride the SAME MIN reduce -- read ok on every rank, and the group's
        # MIN equal to its MAX (one negated slot): the ranks agree, TP0's spans are the group's; else no number
        vals = [int(ok), int(total), int(dev_l), int(key_l), int(lk_ok), int(lk_tot), -int(lk_tot), int(lk_fp),
                -int(lk_fp)]
        grp = getattr(self, "tp_cpu_group", None)
        if grp is not None and torch.distributed.get_world_size(grp) > 1:
            t = torch.tensor(vals, dtype=torch.int64)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=grp)
            vals = [int(v) for v in t.tolist()]
        g_ok, g_total, g_dev, g_key, g_lk_ok, g_lk_min, g_lk_negmax, g_fp_min, g_fp_negmax = vals
        g_locked = _dmp.locked_agreed(locked, g_lk_ok, g_lk_min, -g_lk_negmax, rid, g_fp_min, -g_fp_negmax)
        if not g_ok:
            if err:
                logger.warning("PDFLIP D-MATCH-PROBE FAILED rid=%s %s", rid, err)
            return PdFlipMatchProbeReqOutput(
                rid=rid, success=False,
                message="group_probe_failed: a rank's padding or read-only walk failed -- no group number "
                        "(this rank: %s)" % (err or "ok"),
                handle_ms=(time.perf_counter() - t0) * 1000.0,
            )
        g_dev = min(g_dev, g_total)
        return PdFlipMatchProbeReqOutput(
            rid=rid,
            success=True,
            input_ids=ids,
            device_tokens=int(g_dev),
            host_tokens=int(g_total - g_dev),
            key_match_depth=int(g_key),
            mm_spans=spans,
            device_locked_spans=g_locked,
            handle_ms=(time.perf_counter() - t0) * 1000.0,
            tokenize_ms=float(tokenize_ms or 0.0),
        )

    def _pdflip_match_probe_local(self, rid, input_ids, mm_inputs, extra_key, tokenize_ms) -> PdFlipMatchProbeReqOutput:
        """S56/S56r: the probe on a Form A group or a symmetric host tier -- TP0's number is the group's."""
        from flliper.srt.managers.tp_match_floor import this_rank_follows
        from flliper.srt.pdflip import d_match_probe as _dmp

        t0 = time.perf_counter()
        try:
            ids, spans = self._pdflip_match_probe_pad(rid, input_ids, mm_inputs, extra_key)
            n = len(ids)
            dev = host = key_depth = 0
            locked = None
            tc = getattr(self, "tree_cache", None)
            if n > 1 and tc is not None:
                if not this_rank_follows():  # a Form A worker's anchors are bookkeeping: it adopts the host's depth
                    # the admission's key: the whole ids, limit = _compute_max_prefix_len (n - 1)
                    dev, host, key_depth = _dmp.peek_admission_match(tc, ids, extra_key, n - 1)
                    # #215b: TP0's locked spans ARE the group's here, by the rule that makes its dev/host the group's
                    # (Form A: the host's tree decides; symmetric tier: replicated trees and lock walks)
                    _lk, _lk_ok = _dmp_probe_locked_spans(tc, ids, extra_key, n - 1)
                    locked = [list(x) for x in _lk] if _lk_ok else None
            return PdFlipMatchProbeReqOutput(
                rid=rid,
                success=True,
                input_ids=ids,
                device_tokens=int(dev),
                host_tokens=int(host),
                key_match_depth=int(key_depth),
                mm_spans=spans,
                device_locked_spans=locked,
                handle_ms=(time.perf_counter() - t0) * 1000.0,
                tokenize_ms=float(tokenize_ms or 0.0),
            )
        except Exception as e:  # noqa: BLE001 -- the front names a failed probe; the scheduler never dies of one
            logger.warning("PDFLIP D-MATCH-PROBE FAILED rid=%s %s: %s", rid, type(e).__name__, e)
            return PdFlipMatchProbeReqOutput(
                rid=rid, success=False, message=f"{type(e).__name__}: {str(e)[:200]}",
                handle_ms=(time.perf_counter() - t0) * 1000.0,
            )

    def _pdflip_note_lost_anchors(self) -> None:
        """ANCHOR-LOST: the Mamba anchors this flush's reset drops (device
        only, no host copy -- a write-through the pin budget or the arena
        claim refused, '#1421 BACKUP-REFUSED why=mamba_pin|mamba_claim').
        Kept in ``_pdflip_anchors_lost`` until the sleep leg's answer carries
        them to the front, which then stops crediting presence at those
        depths (pdflip/front.py ``SpanLRU.retract_lost_anchors``)."""
        probe = getattr(self.tree_cache, "pdflip_unbacked_anchors", None)
        if probe is None:
            return
        # FORM-A-WORKER (1304, NF y9nf5 04:54:46): a Form A worker holds no
        # GDN/mamba state (``Mamba Cache ... ssm_state size: 0.00GB`` on
        # TP1/TP2, 2.06GB on TP0) and no arena: its tree anchors are byteless
        # bookkeeping whose write-through is refused by construction
        # (``anchor_only_no_arena`` / ``mamba_pin``). Counted here they were 84
        # ANCHOR-LOST lines (42 sleeps x TP1+TP2) while TP0, the only rank that
        # can resume at an anchor, had 0, and each one made the front retract
        # presence credit (PRESENCE-ANCHOR-LOST, union over ranks). TP0 alone
        # speaks for the anchors; the rank role is the installed plan, the same
        # answer on every rank, and the group fence still gathers (an empty
        # list) from every rank.
        if this_rank_is_form_a_worker():
            _n = self.__dict__.get("_pdflip_anchor_lost_worker_skips", 0) + 1
            self._pdflip_anchor_lost_worker_skips = _n
            if _n <= 2:
                logger.info(
                    "PDFLIP-ANCHOR-LOST at=flush skipped on a Form A worker (no mamba state, no arena: "
                    "TP0's own probe is the group's authority) n=%d", _n,
                )
            return
        try:
            lost = probe()
        except Exception as exc:  # noqa: BLE001 - an instrument never blocks a flush
            logger.warning("PDFLIP-ANCHOR-LOST probe raised: %s: %s", type(exc).__name__, exc)
            return
        if not lost:
            return
        ledger = getattr(self, "_pdflip_anchors_lost", None)
        if ledger is None:
            ledger = self._pdflip_anchors_lost = []
        ledger.extend(int(d) for d, _rid in lost)
        logger.warning(
            "PDFLIP-ANCHOR-LOST at=flush n=%d depths=%s rids=%s (device Mamba anchors without "
            "a host copy; the reset drops them and the front retracts presence credit at "
            "these depths)",
            len(lost), [d for d, _ in lost][:16], [r for _, r in lost][:16],
        )

    def _pdflip_local_store_matches(self, canonical) -> Dict[str, int]:
        """#1317 DESIGN A: this rank's STORE-PRICED match vote, per canonical rid.

        ``tree match + the store depth beyond it``, in tokens, for every rid
        of the canonical head that this rank holds in its waiting queue. The
        value goes into the packed MIN reduce and comes back as the group's
        store match; nothing here is ever a verdict on its own.

        WHY THE PROBE IS LEGAL AS AN INPUT TO A GROUP VERDICT.
        ``store_presence_pages`` answers BY CONTENT KEY, and under #706 a page
        carries all attention layers for its tokens with the cut taken at READ
        time -- so the key is a function of the CONTENT, not of any rank's
        layout or residency (its own docstring says exactly that). The ranks
        therefore normally return the same depth, and the MIN only ever costs
        work when they do not.

        AND IT IS THE DEPTH A FETCH CAN USE, not a KV-only promise: since
        #869b the probe asks ``batch_exists_v2`` with the tree's own component
        transfers, which min-clamps the answer to the last page that also
        carries a mamba anchor. That anchor cap is what bounds the surviving
        remainder to one chunk (#939) and therefore keeps design A inside the
        phase law.

        COST: one backend round-trip per rid per BINDING GENERATION, not per
        pass. The verdict is cached on the request under the same
        generation-keyed shape ``_prefetch_kvcache`` uses
        (``_pp_store_presence_cache``), because a presence verdict is a
        statement about the pool the readers were bound to when it was taken
        and the cutover rebinds them (#1060).

        A VOTE MAY NEVER BREAK THE REDUCE. Same guard and same reason as
        ``_pdflip_local_store_read_pending_ages`` and
        ``_local_head_prefix_matches``: this runs on the collective path on
        every rank, so an exception here would leave one rank out of an
        ``all_reduce`` its peers are already in -- the #580 split, arriving
        through the door marked "safety". An empty vote rides
        ``_ABSENT_MATCH``, which MIN-reduces the arm away and falls the X gate
        back to the TREE match, i.e. to the pre-design-A behaviour. The
        failure mode is "no store credit", never "the ranks disagree".
        """
        out: Dict[str, int] = {}
        _tree_only: Dict[str, int] = {}
        if not canonical:
            return out
        try:
            ctrl = getattr(self.tree_cache, "cache_controller", None)
            probe = getattr(ctrl, "store_presence_pages", None)
            if not callable(probe):
                # #1317c NAMED, not silent: no probe means no store arm at all,
                # and the X gate then prices exactly as it did before design A.
                # A boot that reads zero credits must be able to tell this from
                # "the store held nothing".
                self._pdflip_store_vote_no_probe = (
                    getattr(self, "_pdflip_store_vote_no_probe", 0) + 1
                )
                n = self._pdflip_store_vote_no_probe
                if n <= 3 or n % 256 == 0:
                    logger.warning(
                        "#1317 X-STORE-VOTE NO PROBE n=%d: the tree cache has no "
                        "callable store_presence_pages, so this group takes NO "
                        "store vote and the X gate prices against the tree only "
                        "(pre-design-A behaviour). (denominator: passes in which "
                        "the vote was attempted)", n,
                    )
                return out
            page_size = int(getattr(self.tree_cache, "page_size", 1) or 1)
            from flliper.srt.mem_cache.hicache_phase_binding import (
                current_generation as _current_generation,
            )

            gen = int(_current_generation())
            # S64 S-ARM: one id per vote (= per pass); a probe is reused only inside it
            _vote_pass = getattr(self, "_s64_store_vote_pass", 0) + 1
            self._s64_store_vote_pass = _vote_pass
            wanted = set(canonical)
            for req in self.waiting_queue:
                rid = str(getattr(req, "rid", "") or "")
                if rid not in wanted or rid in out:
                    continue
                fill_ids = getattr(req, "full_untruncated_fill_ids", None)
                if not fill_ids:
                    continue
                total = len(fill_ids)
                local_match = len(req.prefix_indices) + int(
                    getattr(req, "host_hit_length", 0) or 0
                )
                # The span the store would have to answer for, ending where an
                # upstream match must end: `_compute_max_prefix_len` is
                # input_len - 1, because a request must always keep one token
                # to forward.
                match_end = req._compute_max_prefix_len(total)
                if match_end <= local_match:
                    out[rid] = min(local_match, total)
                    continue
                span = fill_ids[local_match:match_end]
                node = getattr(req, "last_host_node", None)
                last_hash = (
                    node.get_last_hash_value() if node is not None else None
                )
                prefix_keys = None
                if node is not None and getattr(
                    self.tree_cache, "hicache_storage_pass_prefix_keys", False
                ):
                    prefix_keys = node.get_prefix_hash_values(node.parent)
                key = (gen, local_match, len(span))
                cached = getattr(req, "_pdflip_store_match_cache", None)
                # S64 S-ARM (rc15 D 08:31:34 pdflip-72-525 store=68608 while the
                # store held 98560; rc14 16-211 20032/27200 vs 37888): the key
                # (generation, local_match, span) does not move while a waiting
                # request's tree match stands still -- and the store fills under
                # it (P's write-through, D's own park demote, the anchor demote).
                # The verdict is reused only inside the pass that took it; a new
                # pass asks the store again, so the arm is the store at decision
                # time (FLLIPER_PDFLIP_S64_STORE_ARM_FRESH, default ON; 0 = the
                # generation-keyed cache as before).
                # COST: an answer that already covers the whole span cannot grow
                # and stays cached; only a SHORT answer -- the one a W31 is priced
                # on -- is asked again (the hash chain is ~20 ms per 100k tokens).
                _fresh = _s64_store_arm_fresh_on()
                # REPROBE-1023 (pdflip/store_reprobe_epoch.py, SOLL R11): a SHORT
                # cached answer under the same key is re-asked on the GROUP's
                # schedule (store-epoch move / changed group vote / back-off
                # bound), read from replicated values only -- not in every pass
                # (27B 1023_144422: 2 x ~210 ms per round for two lifetime-refused
                # hand-backs, 1984 probes, STALE 0).
                _reprobe_held = False
                if cached is not None and cached[0] == key and _fresh and not (
                        (len(cached) > 2 and cached[2] == _vote_pass)
                        or int(cached[1]) * page_size >= len(span)):
                    _reprobe_held = _store_reprobe_epoch.held(self, req)
                    _store_reprobe_epoch.note_vote(not _reprobe_held)
                if cached is not None and cached[0] == key and (
                        not _fresh
                        or (len(cached) > 2 and cached[2] == _vote_pass)
                        or int(cached[1]) * page_size >= len(span)
                        or _reprobe_held):
                    pages = int(cached[1])
                else:
                    # Q-460 SALT-ISOLATION: the request's namespace
                    _ek = getattr(req, "extra_key", None)
                    _ns = {"extra_key": _ek} if _ek is not None else {}
                    # MAMBA-ANCHOR 1015: the fetch's key form (bigram RadixKey)
                    from flliper.srt.pdflip import presence_fetch_key as _pfk

                    _t0 = time.perf_counter()
                    pages = int(probe(_pfk.span_key(self.tree_cache, req, span),
                                      last_hash, prefix_keys, **_ns) or 0)
                    _s64_note_store_arm(self, rid, key, cached, pages, page_size,
                                        (time.perf_counter() - _t0) * 1000.0)
                    req._pdflip_store_match_cache = (key, pages, _vote_pass)
                # The probe counts PAGES; the arm votes TOKENS, because the
                # extent it prices is a token count. Clamped to the span so a
                # backend that over-reports can never credit tokens the
                # request does not have.
                store_tokens = min(max(0, pages) * page_size, len(span))
                _tree_only[rid] = local_match
                out[rid] = min(local_match + store_tokens, total)
        except Exception as e:  # noqa: BLE001 - a vote may never break the reduce
            # #1317c THE SWALLOW WAS SILENT, AND THAT IS WHY BOOT weg2sn6c
            # COULD NOT BE DIAGNOSED. The vote must not raise (an exception
            # here leaves one rank out of an all_reduce its peers are already
            # in), but a bare `return out` made FOUR different outcomes look
            # identical in every log: the probe raised, the probe was absent,
            # the rid was not in this rank's queue, and the store genuinely
            # held nothing. weg2sn6c read `X-STORE-CREDIT` = 0 lines with no
            # way to say which -- the #505a warn-then-continue swallow, in my
            # own code, on the one path the build exists for.
            self._pdflip_store_vote_raised = (
                getattr(self, "_pdflip_store_vote_raised", 0) + 1
            )
            n = self._pdflip_store_vote_raised
            if n <= 5 or n % 64 == 0:
                logger.warning(
                    "#1317 X-STORE-VOTE RAISED n=%d %s: %s -- this rank casts an "
                    "EMPTY store vote for this pass, so the arm MIN-reduces away "
                    "and the X gate falls back to the tree match (the pre-design-A "
                    "behaviour). Never a raise: an exception here would leave this "
                    "rank out of a collective its peers are already in. "
                    "(denominator: every pass in which the store vote was attempted)",
                    n, type(e).__name__, e,
                )
            return out
        # #1317c NAME THE OUTCOME EVERY PASS, so an absence of credit is
        # readable as WHICH absence. A census with no denominator is what made
        # weg2sn6c's zero unreadable.
        self._pdflip_store_vote_passes = getattr(self, "_pdflip_store_vote_passes", 0) + 1
        n = self._pdflip_store_vote_passes
        if n <= 5 or n % 256 == 0:
            logger.info(
                "#1317 X-STORE-VOTE n=%d head=%d voted=%d credited=%d zero=%d "
                "probe=%s (voted: rids of the canonical head this rank holds and "
                "priced; credited: those whose store depth exceeded their tree "
                "match; zero: priced but the store held nothing for them -- which "
                "is the CORRECT answer for a cold prompt whose write-through has "
                "not landed yet, and must not be read as the arm failing. "
                "denominator: passes in which the vote was built)",
                n, len(canonical), len(out),
                sum(1 for rid, v in out.items() if v > _tree_only.get(rid, 0)),
                sum(1 for rid, v in out.items() if v <= _tree_only.get(rid, 0)),
                "callable",
            )
        return out

    def _pdflip_local_store_read_pending_ages(self, canonical) -> Dict[str, int]:
        """This rank's pending-age vote for the canonical head, in ms.

        The stamp is taken the first pass a rid is seen pending and dropped
        the pass it stops being pending, so the age measures THE WAIT and not
        the request's life.  LIFECYCLE TABLE for ``_pdflip_x_defer_since``
        (the rule for every new state field):

          WRITER   this method, one entry per canonical rid observed pending.
          READER   this method only, on the next pass (the age is emitted
                   into the payload; nothing else reads the dict).
          DELETER  this method, in the same call: a rid that is no longer
                   pending, or no longer in the canonical head, is dropped.
                   The canonical head is derived from the waiting queue, so a
                   request that leaves the queue takes its stamp with it and
                   the dict cannot outgrow the head.

        Separating event: none -- writer, reader and deleter are one call per
        pass, which is why no cutover can delete a stamp its consumer still
        needs.

        A VOTE MAY NEVER BREAK THE REDUCE, and this one sits on the
        collective path on every rank -- the same guard and the same reason
        as ``_local_head_prefix_matches``: an exception here would leave one
        rank out of an ``all_reduce`` its peers are already in, which is the
        #580 split this arm exists to prevent, arriving through the door
        marked "safety".  An empty vote is neutral (:data:`NOT_PENDING_MS`
        everywhere), so the failure mode is "nobody defers", never "the
        ranks disagree".
        """
        stamps = getattr(self, "_pdflip_x_defer_since", None)
        if stamps is None:
            stamps = {}
            self._pdflip_x_defer_since = stamps
        ages: Dict[str, int] = {}
        try:
            now = time.monotonic()
            by_rid = {req.rid: req for req in self.waiting_queue}
            alive = set()
            for rid in canonical:
                req = by_rid.get(rid)
                if req is None or not self._pdflip_store_read_is_pending(req):
                    continue
                alive.add(rid)
                if rid not in stamps:
                    stamps[rid] = now
                ages[rid] = max(0, int((now - stamps[rid]) * 1000.0))
            for rid in [r for r in stamps if r not in alive]:
                stamps.pop(rid, None)
        except Exception as exc:  # noqa: BLE001 - a vote may never break the reduce
            n = getattr(self, "_pdflip_x_pending_unvoted", 0) + 1
            self._pdflip_x_pending_unvoted = n
            if n <= 5 or n % 64 == 0:
                logger.warning(
                    "FIX 7 completion arm: could not price this rank's pending "
                    "head (%s); contributing an empty vote, which can only make "
                    "the gate price as it did before this arm (occurrence=%d)",
                    exc, n,
                )
            return {}
        return ages

    def _pdflip_note_store_shortfall(self, req) -> Optional[str]:
        """#1324: defer a read that TERMINATED short of the prefix it asked for.

        THE WALL THIS CLOSES (boot weg2sn6s, 2026-09-10, rid 00f5bc40 =
        front rid pdflip-2-2). A 109,132-token repeat was prefilled whole by P
        at 15:21:49. D read the store 46 s later and got 53,247 tokens,
        because the write-through is asynchronous and had not landed. Nothing
        was reaped, rate-limited, truncated or dropped -- the read simply
        terminated holding half its prefix, and reported
        ``HiCache prefetch success completed_local=53247 completed_synced=53247
        synced=yes`` beside ``#1040 EXTENT STATE-ALIGN loss=0 class=aligned``.
        With no incompleteness anywhere on the record, ``X-DEFER`` had nothing
        to wait on (0 hits on that boot) and the X gate priced the remaining
        55,885 tokens as work D must do: ``X-GATE uncached=55885 X=11101`` ->
        W31 -> W50 after the first stream byte -> 413, re-route impossible.

        WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT.  It raises the
        existing A12.2 shortfall verdict for the rid, so the mark stands, the
        X gate's completion predicate stays True and
        ``_retry_deferred_prefetches`` re-issues the read on a coming pass --
        by which time the write-through has delivered more. It introduces no
        bound, no clock and no new exit: the progress witness
        (``_pdflip_note_prefetch_progress``, which carries the delivered prefix
        as a term since #1324) waits without limit while the store keeps
        delivering and takes the named W88 exit when it stops. That is the
        user's own ruling on this chain -- *"either it loads (wait) or it does
        not load (error -> abort)"* -- and it is why widening a wall clock to
        cover a measured 46 s lag was the wrong repair: the next producer is
        slower or faster, and the number would have to move again.

        NEVER a prefill over X. The terminal exit answers the client 503 by
        name; it does not admit the request "priced as it stands", which is
        the path that recomputed 80,459 tokens against X=11,101 on
        weg2sn6k/sn6l and stands under the user's standing veto.

        Returns the deferral's named outcome, or None when there is nothing to
        do -- which is the overwhelmingly common case and costs one dict
        lookup per queued request per pass.
        """
        rid = str(getattr(req, "rid", "") or "")
        if not rid:
            return None
        if getattr(req, "_pdflip_store_short_fallback", False):
            # rc12y: released by the bound (PREFETCH-DEFER-FALLBACK) -- the
            # short record stays, and must not raise a fresh mark every pass.
            return None
        # #1400: a PP (prefill) group is the PRODUCER of what the store lacks.
        # A read that terminated short of the span it asked for is a PARTIAL
        # HIT to admit -- the remainder is this group's own prefill -- never
        # a producer to wait for. Boot xsn121 (rid 06089842): PP0 deferred
        # here on delivered=4095 deliverable=11126, re-issued the 7031-token
        # remainder every pass into an exhausted host anchor pool, sat
        # '1 queued, 0 running' for 120 s and died on the ring commit while
        # PP1 (told 4095) had admitted -- the #969Z skew. The arm keeps its
        # whole meaning on the single-stage consumer form (group D, leg 2).
        if int(getattr(getattr(self, "ps", None), "pp_size", 1) or 1) > 1:
            return None
        records = getattr(getattr(self, "tree_cache", None),
                          "prefetch_loaded_tokens_by_reqid", None)
        if not records:
            return None
        outcome = records.get(rid)
        # A bare `int` record (the admission sites store one) has no
        # `is_incomplete`, and a `PrefetchOutcome` with `deliverable == 0` is
        # not a terminated store read. Both answer "nothing to do" here, so
        # every pre-#1324 path stays byte-identical.
        if not getattr(outcome, "is_incomplete", False):
            return None
        delivered = int(outcome.materialized)
        deliverable = int(outcome.deliverable)
        # HFB (rc12z21 D 13:35:14, pdflip-8-34, PrefetchBallotDigestMismatch):
        # on a Form A END-vote read `materialized` counts from each rank's
        # OWN span start -- TP0 13376 (base 12544/15104), the workers 9536 /
        # 12096 (base 16384). The store-short cycle bound and the remainder
        # fed on it saw growth on the workers and none on TP0: only TP0 took
        # 'cycles=5 > bound=4 ... over_x' -> W88 503, its queue emptied, the
        # workers kept the rid, and the next ballot's digest split. The
        # reduced END is the group's one answer (25920 / 28480 on all three
        # ranks) and it is the absolute prompt depth the remainder needs.
        _synced_end = getattr(outcome, "synced_end", None)
        if _synced_end is not None:
            # both ends absolute: the span END is the group's too (#580 END vote)
            _dend = getattr(outcome, "deliverable_end", None)
            deliverable = (int(_dend) if _dend is not None
                           else deliverable + int(_synced_end) - int(outcome.synced))
            delivered = int(_synced_end)
        # SHORT-FLIP-REPRICE-1024 (SOLL R2/R7/R11; 27B rc41 D 11:40:21-22, pdflip-21-215): off the END vote the
        # record is SPAN-RELATIVE (deliverable = len(span), materialized from the span start), while every reader
        # of the witness takes it as the ABSOLUTE depth. 215 read [53750, 70994) behind its L1.5-retained device
        # prefix and got 17091: the witness said 17091, D-NORECOMPUTE priced 70999 - 17091 = 53908 > front X and
        # sent a kurz request (real remainder 158) through P. The span start is added here (0 = the old reading:
        # END vote, no stamp, a span from 0, or a stamp that cannot be this read's start).
        from flliper.srt.pdflip import d_norecompute as _dnr_w

        _base = _dnr_w.span_base(req, outcome)
        if _base:
            delivered += _base
            deliverable += _base
        # The witness term, written BEFORE the arm reads it: a re-read that
        # gains nothing leaves this value unchanged and the standstill count
        # advances, which is what ends the wait honestly.
        # S64 S-CAP: the previous pair and this read's cut cause, for the name
        from flliper.srt.pdflip import short_read_cause as _src

        _src.note_record(req, delivered, deliverable, getattr(outcome, "cut_cause", None))
        req._pdflip_store_delivered = delivered
        req._pdflip_store_deliverable = deliverable  # #248h: what the store holds
        self._pdflip_store_short_seen = getattr(self, "_pdflip_store_short_seen", 0) + 1
        # DASHBOARD-AUS-IPC (30.09., FEHLT 5): the tokens these incomplete reads
        # delivered and could have delivered, summed (rankstats cache block)
        # (store tokens: the span start a SHORT-FLIP-REPRICE base added is device-resident, not read)
        self._pdflip_store_short_delivered = getattr(self, "_pdflip_store_short_delivered", 0) + delivered - _base
        self._pdflip_store_short_deliverable = getattr(self, "_pdflip_store_short_deliverable", 0) + deliverable - _base
        n = self._pdflip_store_short_seen
        # S18 (h16v3 1017_224250, pdflip-1-14/-15/-16): the store HELD the whole span (the probe's hit covers the
        # deliverable prefix; TAIL-CONSUME hit=118400) and D's read was cut by its own full arena (PDFLIP-READ-DEPTH
        # cause=arena_full, READ-ROOM SHORT class=not_posted). That is a capacity cut, not P's write-through
        # lagging, yet the line said "the store did not hold the rest yet", so the acceptance "STORE READ
        # INCOMPLETE == 0" could not tell the two apart. Only the NAME and the counter split: the deferral
        # below is identical for both, and the STORE-HELD-DECIDE-1014 predicate it already feeds is untouched.
        # S23 (h16v4b 1018_013102, D TP0 01:36:24, pdflip-0-2 delivered=0): the label said "D's own arena room", yet the
        # READ-DEPTH cause of that very cut was late_complete (a JOINED claim reaped under the read; arena_full on D
        # = 0 in the whole boot, D POSTed one room request, at 01:49:56). "The probe covers the span" proves the STORE
        # holds it, not that the ARENA cut it: CAPACITY-CUT needs the recorded cause arena_full (or none recorded).
        _cut_cause = getattr(outcome, "cut_cause", None)   # CAPACITY causes: _CAPACITY_CUT_CAUSES
        if (int(getattr(outcome, "hit_tokens", 0) or 0) >= int(getattr(outcome, "deliverable", 0) or 0) > 0
                and getattr(outcome, "probed", False) and _cut_cause not in _CAPACITY_CUT_CAUSES):
            self._pdflip_store_short_held_other = getattr(self, "_pdflip_store_short_held_other", 0) + 1
            nh = self._pdflip_store_short_held_other
            if nh <= 8 or nh % 64 == 0:
                logger.warning(
                    "#1324 STORE READ HELD-CUT rid=%s delivered=%d deliverable=%d shortfall=%d hit_tokens=%d "
                    "cut_cause=%s site=drain occurrence=%d held_cut_occurrence=%d -- the store HOLDS the whole span "
                    "(probe hit covers it) and the read was cut by %s, a cause that is not "
                    "arena room pressure (room pressure records arena_full / l2_evicted_before_ref / "
                    "fill_evicted_before_ref). Deferring through the A12.2 shortfall arm. Denominator: every "
                    "terminated store read of this rank whose record reports a page-floored shortfall AND whose "
                    "probe hit covers the deliverable prefix AND whose recorded cut cause is not a room-pressure cause",
                    rid, delivered, deliverable, deliverable - delivered, int(outcome.hit_tokens), _cut_cause, n,
                    nh, _cut_cause,
                )
            return self._apply_prefetch_deferral(req, _VERDICT_STORE_SHORT, site="drain")
        if (int(getattr(outcome, "hit_tokens", 0) or 0) >= int(getattr(outcome, "deliverable", 0) or 0) > 0
                and getattr(outcome, "probed", False)):
            self._pdflip_store_short_capacity = getattr(self, "_pdflip_store_short_capacity", 0) + 1
            nc = self._pdflip_store_short_capacity
            if nc <= 8 or nc % 64 == 0:
                logger.warning(
                    "#1324 STORE READ CAPACITY-CUT rid=%s delivered=%d deliverable=%d shortfall=%d hit_tokens=%d "
                    "site=drain occurrence=%d capacity_occurrence=%d -- the store HOLDS the whole span (probe hit "
                    "covers it) and the read ended short on D's own arena room; this is not P's write-through "
                    "lagging. Deferring through the A12.2 shortfall arm; the room is the defect (READ-ROOM serves "
                    "group P and, via TP0 at its broadcast, group D). Denominator: every terminated store read of this rank whose record reports a "
                    "page-floored shortfall AND whose probe hit covers the deliverable prefix",
                    rid, delivered, deliverable, deliverable - delivered, int(outcome.hit_tokens), n, nc,
                )
            return self._apply_prefetch_deferral(req, _VERDICT_STORE_SHORT, site="drain")
        if n <= 8 or n % 64 == 0:
            logger.warning(
                "#1324 STORE READ INCOMPLETE rid=%s delivered=%d deliverable=%d "
                "shortfall=%d site=drain occurrence=%d -- the read TERMINATED "
                "holding less than the prefix it asked for (the store did not "
                "hold the rest yet; P's write-through is asynchronous). "
                "Deferring through the A12.2 shortfall arm instead of letting "
                "the X gate price the shortfall as tokens D must prefill. "
                "Denominator: every terminated store read of this rank whose "
                "record reports a page-floored shortfall",
                rid, delivered, deliverable, deliverable - delivered, n,
            )
        return self._apply_prefetch_deferral(req, _VERDICT_STORE_SHORT, site="drain")
