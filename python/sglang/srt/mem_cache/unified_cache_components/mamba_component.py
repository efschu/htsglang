from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
    zero_match_result,
)
from sglang.srt.mem_cache.common import peer_needs_mamba_evict
from sglang.srt.mem_cache.mamba_state_pool import (
    active_mamba_state_pool,
    anchor_bytes_pool,
    anchor_bytes_reachable,
    anchor_provenance_verdict,
    note_anchor_bytes,
)
from sglang.srt.mem_cache.memory_pool import sync_free_tensor_repr
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.mem_cache.mamba_ckpt_utils import (
    floor_to_interval,
    is_on_interval,
    is_resume_candidate,
    resume_refusal_reason,
    retention_shrinks_protected,
)
from sglang.srt.runtime_context import get_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )

logger = logging.getLogger(__name__)


class MambaLoadBackUnservable(Exception):
    """#968 FIX-4: the GDN half of a host load-back cannot be served.

    HAZARD THIS NAMES: a KV prefix booked as WARM over a recurrent state that
    was never loaded. `load_back` grows the request's device prefix to the
    told extent; the mamba restore is a separate transfer in the same call.
    If the slot acquisition for that transfer fails, skipping only the mamba
    half leaves the scan resuming at the told position from a foreign or
    zeroed GDN slot -- no assert can fire, because the KV geometry is
    self-consistent at that length.

    The sibling site on the ordinary prefix-resume COW already answers this
    exact question correctly (`finalize_match_result`: "Reusing the KV prefix
    without the matching mamba state would be silently wrong, so the whole
    match is zeroed" -> `zero_match_result`). This exception is that same
    answer for the host load-back path: KV extent and GDN state stand or fall
    together, so the whole load-back falls. Caught in
    `UnifiedRadixCache.load_back`, which unwinds its locks and returns False.
    """


def _decline_retention(is_finished: bool) -> Optional[int]:
    """Answer for "mamba has no on-grid state to file at this position".

    #783: the answer differs by CALLER, because the two callers hand KV
    ownership over differently.

    FINISHED (``None``, no constraint): the request is over and the tree takes
    its KV outright. Retaining it under a mamba tombstone is safe and is the
    whole point -- the full-attention KV does not depend on the mamba grid,
    and a 0 here would collapse the shared ``effective_cache_len`` and cache
    nothing at all.

    UNFINISHED (``0``, cache nothing this step): ``cache_unfinished_req``
    inserts and then IMMEDIATELY re-matches, handing ownership over on the
    strength of what the match returns
    (``req.cache_protected_len = len(new_indices)``). An anchorless node is
    deliberately unmatchable, so that round trip comes back EMPTY: the tree
    would hold the KV while the request kept using -- and later freeing --
    the same slots, with ``cache_protected_len`` falling back to 0. Measured
    as exactly that on the first boot of this branch: a 122-token off-grid
    step left 122 slots unaccounted ("pool memory leak detected! [full]
    total=161378, available=40952, evictable=130"). Mid-flight steps
    therefore keep the pre-existing "cache nothing" behaviour; the KV stays
    with the request until it finishes, where the FINISHED branch retains it.
    """
    return None if is_finished else 0


def _same_mamba_slot(a, b) -> bool:
    """#929: do these two handles name the SAME mamba slot?

    Compared by VALUE, never by object identity: the donation on the
    no-int8-checkpoint path is `req.mamba_pool_idx.unsqueeze(-1).clone()`
    (mamba_component.py:956/:965) -- a distinct tensor carrying the same id, so
    `is` and even `==` on the objects answer the wrong question. Shapes differ
    between the two handles (0-d/1-d against 1-d/2-d), so both are flattened to
    a set of ints before comparing.

    Answers False on anything unreadable. That is the conservative direction
    HERE: a False means "different slots", so the donation is released -- and
    the release itself is guarded by the allocator's own #924 double-free
    check, which raises rather than corrupting. A True on unreadable input
    would instead SKIP a release and leak, which is the defect this whole
    ticket is about.
    """
    if a is None or b is None:
        return False
    try:
        ia = {int(x) for x in a.reshape(-1).tolist()}
        ib = {int(x) for x in b.reshape(-1).tolist()}
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(ia) and ia == ib


class MambaComponent(TreeComponent):
    component_type = ComponentType.MAMBA

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

        assert isinstance(cache.req_to_token_pool, HybridReqToTokenPool), (
            f"MambaComponent requires HybridReqToTokenPool, got {type(cache.req_to_token_pool)}"
        )
        if not params.enable_mamba_extra_buffer:
            assert cache.page_size == 1, (
                f"MambaComponent requires page_size=1 when mamba_extra_buffer is disabled, got {cache.page_size}"
            )
        super().__init__(cache, params)
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer
        self.enable_mamba_extra_buffer_lazy = params.enable_mamba_extra_buffer_lazy
        # HiCache state
        self._mamba_pool_host = None  # set to host mamba pool when HiCache enabled
        # #747: --mamba-checkpoint-interval grid, cached once exactly like
        # MambaRadixCache (:503/:511). None = grid off, every decision below
        # degenerates to the pre-#747 behaviour.
        self.mamba_checkpoint_interval = get_server_args().mamba_checkpoint_interval
        self.mamba_ckpt_strict_resume = envs.SGLANG_MAMBA_CKPT_STRICT_RESUME.get()
        self._off_grid_insert_refusals = 0
        # #783: request ends that carried no on-grid state to file. Counted
        # because a grid coarser than the traffic makes EVERY end decline,
        # which leaves the tree anchor-free -- a state that is otherwise
        # indistinguishable from idle traffic in the logs.
        self._off_grid_retention_declines = 0
        #: #824 declines, counted separately from the off-grid ones above: a
        #: structurally non-monotone tracked position must announce itself
        #: rather than look like ordinary grid misses.
        self._protected_retention_declines = 0
        #: #928 resume refusals, split by which term said no. A foreign-pool
        #: anchor and a stateless node are different defects with different
        #: remedies (carry the anchor across the cutover / stop planting
        #: tombstones under a full key), so one counter for each.
        self._foreign_pool_resume_refusals = 0
        self._stateless_resume_refusals = 0

    def _raw_token_pos(self, key_units: int) -> int:
        """Absolute RAW-token position of a node measured in KEY units.

        #783: the checkpoint grid is a statement about how many tokens the
        recurrent state has consumed, so every grid test must be evaluated in
        raw tokens. Two of the three test sites were not: retention measures
        `token_ids_len` (raw), while the insert backstop measures
        `len(params.key)` and the match walk accumulates `cum_tokens`, both of
        which are KEY units. Under EAGLE the radix key is a bigram view, where
        k bigrams span k+1 raw tokens, so the two unit systems disagree by one
        and `interval | n` and `interval | (n-1)` can never hold together --
        making an anchor that retention certified unacceptable on match, for
        EVERY request length. EAGLE plus --mamba-checkpoint-interval therefore
        could not produce a single prefix hit. Identity outside EAGLE, where
        key units already are raw tokens.
        """
        if key_units <= 0:
            return 0
        return key_units + 1 if self.cache.is_eagle else key_units

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode, int], bool]:
        ct = self.component_type
        interval = self.mamba_checkpoint_interval
        # #747 match seam: the anchor decision (state present AND on the
        # checkpoint grid) is the shared `is_resume_candidate` rule -- the
        # same call MambaRadixCache._match_prefix_helper makes, so the two
        # lineages cannot drift. With interval=None this is byte-identical
        # to the old pure presence test.
        raw_pos = self._raw_token_pos
        if match_device_only:
            return lambda node, depth: is_resume_candidate(
                raw_pos(depth),
                interval,
                has_device_value=node.component_data[ct].value is not None,
            )

        # HiCache: evicted + backuped (host_value present) is also a valid match
        #
        # #758 emitter (2 of 3): MAMBA HOST-BACKED RESUME.
        #
        # The comp4 ladder could not evidence "at least one host-backed resume"
        # because nothing said when one happened -- 0 lines matching
        # "host-backed" across the window, which again cannot distinguish a
        # working host tier from an idle one. THIS predicate is the moment of
        # truth: it is the only place an anchor is accepted on the strength of
        # a host copy rather than a resident one.
        #
        # ONLY the host-accepted branch logs, and that is what keeps it cheap:
        # this lambda runs per node in every match walk, but a match that has a
        # device value is the overwhelmingly common case and returns before the
        # emitter is reached. A host-only acceptance is exactly the rare event
        # being counted, and it is the one that triggers load_back.
        def _resume_with_host(node, depth):
            data = node.component_data[ct]
            has_dev = data.value is not None
            ok = is_resume_candidate(
                raw_pos(depth),
                interval,
                has_device_value=has_dev,
                has_host_value=data.host_value is not None,
                device_only=False,
            )
            if ok and not has_dev:
                try:
                    n = getattr(MambaComponent, "_host_resume_count", 0) + 1
                    MambaComponent._host_resume_count = n
                    if n == 1 or n % 8 == 0:
                        logger.info(
                            "MAMBA-HOST-RESUME n=%d: anchor accepted at depth=%d on a "
                            "HOST-backed state (device copy evicted); this match "
                            "triggers load_back. interval=%s",
                            n,
                            depth,
                            interval,
                        )
                except Exception:  # noqa: BLE001 - never break a match walk
                    pass
            return ok

        return _resume_with_host

    def explain_match_refusal(
        self, node, depth: int, match_device_only: bool = False
    ) -> Optional[str]:
        """Which term of the resume rule declined this node (#913/W42).

        THE MEASUREMENT THIS UNBLOCKS. The 0826 acceptance window's census read
        ``verdict=refused reached=45 accepted=0 refusers=MambaComponent:45`` on
        361 walks and the same at 49 on 301 more -- 671 of 675 -- and the line
        could not say whether those nodes held no recurrent state (a write-side
        tombstone from ``commit_insert_component_data``) or held one at a
        position off the ``--mamba-checkpoint-interval`` grid (a read-side
        determinism policy that is refusing a usable anchor on purpose). The
        remedies are in different files and one of them is the #767 corruption
        direction, so guessing between them is not an option.

        Routed through ``resume_refusal_reason``, which is the same expression
        ``is_resume_candidate`` is now defined in terms of -- so the predicate
        and its explanation cannot disagree about a node, which is exactly how
        #747 records the two match lineages drifting apart before.

        Returns None if the node is in fact admissible: the caller only asks
        about nodes some validator refused, and on a multi-component walk that
        refuser may have been a DIFFERENT component. Answering "no reason" then
        is the truthful answer, not a missing one.
        """
        ct = self.component_type
        data = node.component_data[ct]
        return resume_refusal_reason(
            self._raw_token_pos(depth),
            self.mamba_checkpoint_interval,
            has_device_value=data.value is not None,
            has_host_value=data.host_value is not None,
            device_only=match_device_only or self.cache.cache_controller is None,
        )

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        cow_mamba = params.cow_mamba
        req = params.req
        last_node = result.best_match_node
        interval = self.mamba_checkpoint_interval

        # #747 strict resume, mirroring the SGLANG_MAMBA_CKPT_STRICT_RESUME
        # block in MambaRadixCache._match_post_processor (:1591-1607):
        # identical requests must resume at the DEEPEST interval boundary of
        # their match or not at all -- a shallower surviving anchor depends on
        # which entries the LRU churn spared and would vary run-to-run.
        # Mirrored only where the chunk sums are exact token depths
        # (cache_controller is None: no evicted-but-backuped nodes, so
        # `value_chunks` is gap-free). Under a host tier the premise is weaker
        # to begin with: an evicted anchor stays matchable via its host backup
        # (see `create_match_validator`), so device-LRU churn does not move
        # the resume point there.
        zeroed_by_strict_resume = False
        effective_best_len = best_value_len
        if (
            interval is not None
            and self.mamba_ckpt_strict_resume
            and self.cache.cache_controller is None
            and best_value_len > 0
        ):
            total_match_tokens = sum(len(v) for v in value_chunks)
            best_depth = sum(len(v) for v in value_chunks[:best_value_len])
            if best_depth != floor_to_interval(total_match_tokens, interval):
                result = zero_match_result(self.cache, result)
                effective_best_len = 0
                zeroed_by_strict_resume = True

        # #767: THE FILL RUNS UNDER A HOST TIER TOO. This was gated on
        # `cache_controller is None` with the note that branching-state fill was
        # "temporarily" skipped in that mode. The gate split the checkpoint
        # interval's contract in half: inserts stay grid-gated (an off-grid
        # finish caches nothing) while the fill that ESTABLISHES the grid anchor
        # never runs, so a match can reach a depth its recurrent state never
        # did. Measured as a clean 2x2 on the short-prompt gate -- HiCache off /
        # interval off: 0 REP; interval alone: 0 REP and 1 distinct (a full
        # pass); HiCache alone: 0 REP; HiCache AND interval: 10/16 REP. Neither
        # flag alone does it, which is the signature of a contract split rather
        # than of either feature.
        #
        # branch_grid keeps the same semantics in both modes (interval when one
        # is configured, else the FLA chunk size), so the two lineages cannot
        # drift apart again the way #747 records them doing before.
        if len(value_chunks) > effective_best_len:
            # #747: the branching position must sit on the CHECKPOINT grid when
            # one is configured, not merely on the FLA chunk grid. Mirrors
            # mamba_radix_cache.py:1614 (`branch_grid = interval or chunk_size`)
            # -- the same rule, read from the same place, so the two lineages
            # cannot drift apart the way they did before #747.
            branch_grid = interval or get_server_args().mamba_cache_chunk_size
            aligned_seqlen = floor_to_interval(
                sum(len(v) for v in value_chunks), branch_grid
            )
            branching_seqlen = aligned_seqlen if aligned_seqlen > 0 else None
        else:
            branching_seqlen = None

        if zeroed_by_strict_resume:
            # Full re-prefill; the branching seqlen still points at the grid
            # position whose checkpoint the re-prefill will re-establish
            # (device lineage: the branch-grid block runs after the strict
            # zeroing too).
            return result._replace(mamba_branching_seqlen=branching_seqlen)

        mamba_value = last_node.component_data[self.component_type].value

        # #928: TWO WAYS THIS NODE IS NOT A RESUME POINT, AND BOTH USED TO BE
        # SILENT. The rule is not new -- it is stated verbatim thirty lines
        # below for the slot-starvation case ("Reusing the KV prefix without
        # the matching mamba state would be silently wrong, so the whole match
        # is zeroed") -- it simply was not applied to the two cases that
        # actually occur on a phase-flip boot.
        #
        # Both refusals keep `branching_seqlen`, as the strict-resume zeroing
        # above does: the re-prefill re-establishes the anchor at that grid
        # position, and dropping it would make the next request pay again.
        if cow_mamba:
            host_value = last_node.component_data[self.component_type].host_value
            if mamba_value is None and host_value is None:
                # (a) NO STATE AT ALL. `prepare_for_caching_req` plants such a
                # tombstone by design (:753/:799/:830) and
                # `commit_insert_component_data` accepts it, so the tree
                # legitimately holds full-length keys with no anchor. Falling
                # through left `req.mamba_cow_src_index` unset and
                # `req.mamba_needs_clear` True (memory_pool.py:2090), so the
                # request ZEROED its recurrent state and reused the whole KV
                # prefix anyway: the full-attention layers saw every token,
                # the recurrent layers saw a sequence that had not started.
                self._stateless_resume_refusals += 1
                count = self._stateless_resume_refusals
                if count <= 3 or count % 1000 == 0:
                    # #1040: NAME BOTH NUMBERS. "Refusing" alone cannot say
                    # whether the KV extent overshot a state point that exists
                    # further back, or whether the path carries no state at all
                    # -- and those two have different fixes (round the extent
                    # down vs. make the anchor survive its node, #1039). The
                    # match depth reached and the extent the tree WOULD have
                    # served are printed beside the verdict so the next reader
                    # measures instead of infers.
                    logger.warning(
                        "[#928 anchor] REFUSING resume: node carries no "
                        "recurrent state on device or host, so its KV prefix "
                        "cannot be reused; re-prefilling. occurrence=%d rid=%s "
                        "#1040 match_tokens=%d best_value_len=%d "
                        "kv_host_hit=%s next_state_point=NONE-ON-THIS-PATH",
                        count,
                        getattr(req, "rid", None),
                        sum(len(v) for v in value_chunks),
                        best_value_len,
                        result.host_hit_length,
                    )
                return zero_match_result(self.cache, result)._replace(
                    mamba_branching_seqlen=branching_seqlen
                )
            if mamba_value is not None and not anchor_bytes_reachable(
                self.cache, mamba_value
            ):
                # (b) STATE THIS PHASE CANNOT READ. The anchor's bytes are in
                # the other stack's tensors at the same slot id; the deferred
                # COW (model_runner.py:4368-4373) copies from the EXECUTING
                # runner's pool and would hand this request whatever the other
                # phase last left at that index. See mamba_state_pool.py for
                # why the bytes cannot simply be fetched here.
                self._foreign_pool_resume_refusals += 1
                count = self._foreign_pool_resume_refusals
                # #928 LIVELOCK WATCH, and it is the honest cost of this
                # refusal. Refusing sends the request back to a full prefill.
                # Under strict purity that prefill runs in PP, donates a fresh
                # PP-pool anchor, and the pp_to_tp re-admission meets the same
                # foreign-pool verdict again -- so a request that is RETRACTED
                # rather than carried across the cutover can refuse forever,
                # paying two cutovers a lap. The loop is not created here: it
                # is the pre-existing hole that the wrong answer was hiding,
                # and the way out is the carry (gdn_flip_mover already moves
                # RESIDENT slots; the 2g cutover moved none -- "0 live slots,
                # sent 0 cells / 0.00 MiB"), not a softer refusal. A second
                # refusal of the SAME request is therefore the signal that the
                # carry is the blocking posten, and it says so by name rather
                # than looping in silence.
                seen = getattr(req, "_mamba_resume_refusals", 0) + 1
                if req is not None:
                    req._mamba_resume_refusals = seen
                if seen >= 2:
                    logger.error(
                        "[#928 anchor] REPEAT refusal #%d for rid=%s: this "
                        "request has now re-prefilled and been refused again, "
                        "which means it is being RETRACTED at the cutover "
                        "instead of carried. The anchor can never be local "
                        "while that holds, and the request cannot make "
                        "progress. The blocking posten is the resident carry "
                        "of the mamba state across pp_to_tp, not this seam",
                        seen,
                        getattr(req, "rid", None),
                    )
                if count <= 3 or count % 1000 == 0:
                    logger.warning(
                        "[#928 anchor] REFUSING resume: verdict=%s "
                        "anchor_pool=0x%x active_pool=0x%x -- the anchor's "
                        "state bytes belong to the other phase's pool at this "
                        "slot id and are unreachable from the computing "
                        "layout; re-prefilling. occurrence=%d rid=%s",
                        anchor_provenance_verdict(self.cache, mamba_value),
                        id(anchor_bytes_pool(self.cache, mamba_value)),
                        id(active_mamba_state_pool(self.cache)),
                        count,
                        getattr(req, "rid", None),
                    )
                return zero_match_result(self.cache, result)._replace(
                    mamba_branching_seqlen=branching_seqlen
                )

        if cow_mamba and mamba_value is not None:
            assert req is not None
            if req.mamba_pool_idx is None:
                dst_index = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
                if dst_index is None:
                    # Capture the inc result and thread swa_uuid_for_lock back
                    # into dec. Without it, SWA's release walks past this
                    # request's window boundary all the way to root and
                    # over-decrements SWA locks held by other resident requests
                    # on ancestor nodes.
                    lock_result = self.cache.inc_lock_ref(last_node)
                    self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                    dst_index = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
                    self.cache.dec_lock_ref(last_node, lock_result.to_dec_params())
                elif peer_needs_mamba_evict(self.cache):
                    # #639b: see `_alloc_mamba_slot`. This rank's COW slot came
                    # straight out of the pool while a peer had to tombstone a
                    # node for its own; match that tombstone or the two radix
                    # replicas stop agreeing on which nodes carry mamba state.
                    # Locked exactly as the sibling branch above, so eviction
                    # cannot reclaim the node this request is resuming from.
                    lock_result = self.cache.inc_lock_ref(last_node)
                    self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                    self.cache.dec_lock_ref(last_node, lock_result.to_dec_params())
                if dst_index is None:
                    # REQUIRED-allocation path: this slot would hold the
                    # request's OWN resumed state. Degrade to a full cache MISS
                    # instead of killing the scheduler -- the request re-prefills
                    # and takes its slot through normal admission (which defers
                    # when `rem_mamba_slots` is exhausted). Reusing the KV prefix
                    # without the matching mamba state would be silently wrong,
                    # so the whole match is zeroed.
                    self._log_mamba_slot_starvation("mamba (prefix-resume COW)")
                    return zero_match_result(self.cache, result)
                req.mamba_pool_idx = dst_index[0]
                # #991: acquired speculatively by THIS admission round's match.
                # If `add_one_req` then rejects the request, scheduler.py's
                # revert site owes this slot back -- and ONLY this one.
                req.mamba_slot_acquired_this_admission = True
            req.mamba_cow_src_index = mamba_value
            req.mamba_needs_clear = False

        # HiCache: if mamba was evicted from device but has host backup,
        # ensure mamba_host_hit_length >= 1 so load_back is triggered.
        cd = last_node.component_data[self.component_type]
        if cd.value is None and cd.host_value is not None:
            result = result._replace(
                mamba_host_hit_length=max(result.mamba_host_hit_length, 1)
            )

        if os.getenv("SGLANG_767_TRACE", "") not in ("", "0"):
            # #790 sweep: this trace is opt-in (off by default), but it is
            # reached on every prefix match on the admission path -- exactly
            # the hot path that wedged for 25+ min in the #790 incident, and
            # an operator turning this flag on during a live #767 debugging
            # session is the least convenient moment for that wedge to
            # recur. `mamba_value.tolist()` and the two bare tensor args
            # below used to force a D2H sync + stream sync inside
            # `logging.emit` (`.tolist()`, and `%s` on a raw CUDA tensor
            # calling `Tensor.__repr__`); see `sync_free_tensor_repr` for
            # why the values themselves are not read here.
            logger.warning(
                "#767-TRACE match: rid=%s match_tokens=%s best_len=%s "
                "effective=%s cow=%s mamba_value=%s cow_src=%s pool_idx=%s "
                "host_hit=%s zeroed=%s",
                getattr(req, "rid", None),
                sum(len(v) for v in value_chunks),
                best_value_len,
                effective_best_len,
                cow_mamba,
                sync_free_tensor_repr(mamba_value),
                sync_free_tensor_repr(getattr(req, "mamba_cow_src_index", None)),
                sync_free_tensor_repr(getattr(req, "mamba_pool_idx", None)),
                result.mamba_host_hit_length,
                zeroed_by_strict_resume,
            )

        return result._replace(mamba_branching_seqlen=branching_seqlen)

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        # #783: NO STATE TO DONATE IS A TOMBSTONE, NOT AN ERROR. Two producers
        # legitimately reach this with `mamba_value=None`: the int8 checkpoint
        # pool exhausted with nothing evictable (see `prepare_for_caching_req`,
        # which already documents "the node is then inserted without a mamba
        # value ... so the KV stays cached and only the mamba resume point is
        # lost"), and an off-grid request end, which has no on-grid state to
        # file. Both keep the full-attention KV and drop only the anchor. The
        # `assert` that stood here contradicted that documented contract and
        # made the tombstone path a crash, which is why the off-grid case had
        # to be bought off earlier with `cache_len = 0` -- the veto that
        # emptied the whole tree. `mamba_exist=True` routes the (absent)
        # donation into the caller's cleanup, which frees the request's own
        # mamba slot exactly as an uncached step would.
        if params.mamba_value is None:
            result.mamba_exist = True
            return
        # #747 retention backstop: mamba is leaf-only data, so the target
        # node's absolute position is the full inserted key length. An
        # off-grid commit is refused (the node keeps a tombstone) instead of
        # planting an anchor that would move resume points off the grid.
        # Unreachable through `prepare_for_caching_req`, which already gates
        # `cache_len`; this covers every OTHER InsertParams producer (session
        # restore paths) and the case where another component shrank the
        # effective cache_len below mamba's on-grid choice -- in both cases
        # the donated state would sit DEEPER than the key it is filed under.
        # `mamba_exist=True` routes the donated value into the caller's
        # existing duplicate-cleanup, which frees it.
        # #783: in RAW tokens, like every other grid test -- `len(params.key)`
        # is a bigram count under EAGLE and would refuse the very anchor the
        # retention gate just certified. See `_raw_token_pos`.
        key_raw_pos = self._raw_token_pos(len(params.key))
        if not is_on_interval(key_raw_pos, self.mamba_checkpoint_interval):
            self._off_grid_insert_refusals += 1
            count = self._off_grid_insert_refusals
            if count <= 3 or count % 1000 == 0:
                logger.warning(
                    "mamba checkpoint interval: refusing off-grid insert at "
                    "raw token position %d (interval %d), occurrence=%d",
                    key_raw_pos,
                    self.mamba_checkpoint_interval,
                    count,
                )
            result.mamba_exist = True
            return
        if is_new_leaf:
            # #928: the anchor enters the tree here, so THIS is the moment its
            # bytes' owner is known -- the stack that was computing when the
            # state was produced. The id alone cannot say it later.
            note_anchor_bytes(self.cache, params.mamba_value)
            node.component_data[self.component_type].value = params.mamba_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(
                params.mamba_value
            )
            return
        if node.component_data[self.component_type].value is None:
            # #928: same moment, the other arm -- a node that held only a host
            # copy (or a tombstone) is being given device state now.
            note_anchor_bytes(self.cache, params.mamba_value)
            node.component_data[self.component_type].value = params.mamba_value
            # move from host LRU to device LRU
            host_lru = self.cache.host_lru_lists[self.component_type]
            if host_lru.in_list(node):
                host_lru.remove_node(node)
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(
                params.mamba_value
            )
            node.last_access_time = get_and_increase_time_counter()
            return
        self.cache.lru_lists[self.component_type].reset_node_mru(node)
        node.last_access_time = get_and_increase_time_counter()
        result.mamba_exist = True

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        ct = self.component_type
        new_parent.component_data[ct].value = None
        new_parent.component_data[ct].lock_ref = 0
        # HiCache: mamba host_value stays on child (mamba = leaf-only data)
        new_parent.component_data[ct].host_value = None
        new_parent.component_data[ct].host_lock_ref = 0

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            self._free_mamba_value(cd.value)
            freed = len(cd.value)
            self.cache.component_evictable_size_[self.component_type] -= freed
            cd.value = None

        # Host layer
        host_lru = self.cache.host_lru_lists[self.component_type]
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            if self._mamba_pool_host is not None:
                self._mamba_pool_host.free(cd.host_value)
            cd.host_value = None
            if host_lru.in_list(node):
                host_lru.remove_node(node)

        # After device tombstone: if only host_value remains, insert into host LRU
        if (
            target is EvictLayer.DEVICE
            and cd.value is None
            and cd.host_value is not None
        ):
            if not host_lru.in_list(node):
                host_lru.insert_mru(node)

        return freed, host_freed

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        request = params.mamba_num
        ct = self.component_type
        lru = self.cache.lru_lists[ct]
        x = lru.get_lru_no_lock()
        while tracker[ct] < request and x is not None and lru.in_list(x):
            assert x.component_data[ct].value is not None
            if x in self.cache.evictable_device_leaves:
                # D-leaf: atomic eviction of all components
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_device_leaf(x, tracker)
                if not lru.in_list(x_next):
                    x_next = lru.get_lru_no_lock()
                x = x_next
            else:
                # Internal: tombstone Mamba + cascade
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.DEVICE, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = x_next

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type
        if node is self.cache.root_node:
            return result
        cd = node.component_data[ct]
        value = cd.host_value if lock_host else cd.value
        # A node in skip_lock_node_ids was a tombstone when this lock was acquired.
        if value is None:
            result.skip_lock_node_ids.setdefault(ct, set()).add(node.id)
            return result

        if lock_host:
            if cd.host_lock_ref == 0:
                host_lru = self.cache.host_lru_lists[ct]
                if host_lru.in_list(node):
                    host_lru.remove_node(node)
            cd.host_lock_ref += 1
        else:
            if cd.lock_ref == 0:
                vlen = len(value)
                self.cache.component_evictable_size_[ct] -= vlen
                self.cache.component_protected_size_[ct] += vlen
            cd.lock_ref += 1
        if self.cache._pin_trace_every:
            self.cache.record_pin_trace_mamba("inc", host=lock_host)
        return result

    def anchor_release_admissible(self, node: Optional[UnifiedTreeNode]) -> bool:
        """#773/#755: may THIS node's mamba pin be released before the alloc?

        The #755 reorder turns `alloc -> insert -> dec(old) -> inc(new)` into
        `dec(old) -> alloc -> insert -> inc(new)`, so the old and new anchors
        never coexist and a running request holds `active + donated` instead
        of `active + donated + old pin`. That is the whole slot it saves.

        Between the release and the new pin the old node's state slot is
        evictable, so the release is only safe when losing it costs a
        `load_back` rather than the anchor itself. Two facts, both required,
        and neither of them is a config question:

        * the node carries a HOST copy -- `is_resume_candidate(...,
          device_only=False)` is what makes an evicted-but-backed anchor a
          valid match, and that is exactly the degradation being relied on;
        * that copy has LANDED. #767: write-through publishes `host_value`
          the moment the transfer is handed to the controller, and the same
          block records the node in `ongoing_write_through`. Between those
          two facts the anchor exists as an intention only, and releasing
          there is precisely the dead anchor this guard exists to prevent.

        A node whose mamba value is already gone has no pin to release, so it
        answers False rather than pretending it did something.
        """
        if node is None or node is self.cache.root_node:
            return False
        ct = self.component_type
        if len(node.component_data) <= int(ct):
            return False
        cd = node.component_data[ct]
        if cd.value is None:
            return False
        if cd.host_value is None:
            return False
        return node.id not in self.cache.ongoing_write_through

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        if node is self.cache.root_node:
            return
        cd = node.component_data[ct]
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        if node.id in skip_lock_node_ids:
            return

        value = cd.host_value if lock_host else cd.value
        if lock_host:
            cd.host_lock_ref -= 1
            if cd.host_lock_ref == 0 and cd.value is None and cd.host_value is not None:
                host_lru = self.cache.host_lru_lists[ct]
                if not host_lru.in_list(node):
                    host_lru.insert_mru(node)
            if self.cache._pin_trace_every:
                self.cache.record_pin_trace_mamba("dec", host=True)
            return

        if cd.lock_ref > 0:
            if cd.lock_ref == 1:
                vlen = len(value)
                self.cache.component_evictable_size_[ct] += vlen
                self.cache.component_protected_size_[ct] -= vlen
            cd.lock_ref -= 1
            if self.cache._pin_trace_every:
                self.cache.record_pin_trace_mamba("dec", host=False)

    def _alloc_mamba_slot(self) -> Optional[torch.Tensor]:
        """Allocate one mamba pool slot, evicting if necessary.

        Returns ``None`` when the pool is exhausted and eviction frees nothing
        because every cached state is locked by a running request. Mirrors
        `MambaRadixCache._alloc_mamba_slot`; see the rationale there. Callers
        MUST handle ``None`` -- cache-insert paths skip the insert.
        """
        slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
        if slot is None:
            self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
            if slot is None:
                self._log_mamba_slot_starvation("mamba")
        elif peer_needs_mamba_evict(self.cache):
            # #639b: this rank had a slot, a peer did not. The peer is
            # tombstoning a mamba node right now; skipping the eviction here
            # would leave that node resident on this rank only, and
            # `_match_prefix_helper` would then walk past it here and stop at
            # it there. Evict to keep the tombstone sets equal. Unreachable
            # unless the scheduler published a floor, i.e. never on a single
            # rank or on ranks whose occupancy agrees.
            self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        return slot

    def _log_mamba_slot_starvation(self, pool_name: str) -> None:
        """Rate-limited warning for an unservable mamba slot request."""
        self._mamba_starvation_count = getattr(self, "_mamba_starvation_count", 0) + 1
        count = self._mamba_starvation_count
        if count <= 3 or count % 1000 == 0:
            logger.warning(
                "%s slot pool exhausted and nothing evictable (all cached states "
                "are locked by running requests): skipping this cache insert. "
                "occurrence=%d mamba_evictable=%d mamba_protected=%d",
                pool_name,
                count,
                self.cache.mamba_evictable_size(),
                self.cache.mamba_protected_size(),
            )

    @property
    def int8_ckpt_pool(self):
        return getattr(self.cache.req_to_token_pool, "mamba_ckpt_pool", None)

    def _alloc_int8_ckpt_slot(self) -> Optional[torch.Tensor]:
        """Returns ``None`` when exhausted with nothing evictable; see
        `_alloc_mamba_slot`."""
        slot = self.int8_ckpt_pool.alloc(1)
        if slot is None:
            self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.int8_ckpt_pool.alloc(1)
            if slot is None:
                self._log_mamba_slot_starvation("int8 mamba checkpoint")
        return slot

    def _commit_int8_checkpoint(
        self, active_slots: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Returns ``None`` when no checkpoint slot is available; the caller
        must then skip the cache insert."""
        ckpt_slot = self._alloc_int8_ckpt_slot()
        if ckpt_slot is None:
            return None
        # #767: read the active slots from the computing stack's pool (see
        # the donate-copy site below; same wrong-pool class).
        self.int8_ckpt_pool.store_from_active(
            active_mamba_state_pool(self.cache),
            active_slots.view(-1),
            ckpt_slot,
        )
        return ckpt_slot

    def _free_mamba_value(self, mamba_value: torch.Tensor) -> None:
        if self.int8_ckpt_pool is not None:
            self.int8_ckpt_pool.free(mamba_value)
        else:
            self.cache.req_to_token_pool.mamba_allocator.free(mamba_value)

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        # #991 BACKSTOP, NAMED AS A DEFECT AND NOT AS A DESIGN.
        #
        # Every dereference of `req.mamba_pool_idx` below (:907-908, :983,
        # :1027, :1056) is unguarded, and that is the correct shape: a request
        # whose result is being cached went through `HybridReqToTokenPool.alloc`
        # and owns an active state slot by construction. So "the slot is never
        # None here" is a LIFECYCLE obligation of every path that can reach
        # re-admission, not a local fact -- and boot 10 (b27d7c2ff4) broke it
        # at `scheduler.py`'s revert site, killing rank 0 with an
        # AttributeError on :1056.
        #
        # THE FIX FOR THAT IS AT THE JUNCTION, in `scheduler.py` (#991
        # provenance stamp), not here. This is only the net that turns the
        # NEXT unknown producer from a dead scheduler into a named, counted
        # line -- because a crash here takes the whole instance and a decline
        # takes one step's retention.
        #
        # IT IS NOT FREE, AND THE COST IS STATED: `_decline_retention(False)`
        # returns 0, so the shared `effective_cache_len` collapses and this
        # step retains nothing. That is a re-prefill of the chunk on the next
        # match -- i.e. it spends exactly what the standing no-double-prefill
        # order forbids as a routine. Reaching this line is therefore a BUG
        # REPORT with a rid on it, and the counter exists so it cannot become
        # quiet routine.
        if req.mamba_pool_idx is None:
            self._991_absent_active_slot = (
                getattr(self, "_991_absent_active_slot", 0) + 1
            )
            count = self._991_absent_active_slot
            if count <= 8 or count % 256 == 0:
                logger.warning(
                    "#991 ACTIVE MAMBA SLOT ABSENT AT CACHE-INSERT rid=%s "
                    "is_finished=%s occurrence=%d: a request whose batch ran "
                    "holds no active state slot, so some path released it "
                    "between `alloc` and this result. Declining retention "
                    "(0) instead of dereferencing None -- this costs the "
                    "step's retention and re-prefills the chunk, so it is a "
                    "DEFECT REPORT, not a supported degradation. The known "
                    "producer is the admission revert site fixed by #991; a "
                    "hit here names a second one.",
                    getattr(req, "rid", None),
                    is_finished,
                    count,
                )
            insert_params.mamba_value = None
            return _decline_retention(is_finished)

        if self.enable_mamba_extra_buffer:
            cache_len = req.mamba_last_track_seqlen
            # #747 cache_len seam (mirrors mamba_radix_cache.py:626-640 and
            # :795-809): the tracked position is on the checkpoint grid by
            # construction (prefill targets and decode tracking both use the
            # interval); enforce it so an off-grid state can never enter the
            # tree. Off-grid -> NO MAMBA ANCHOR. Never floor: rounding the
            # retained key down while donating a deeper state would pair state
            # and key at different positions (silent corruption).
            #
            # #783: declining the anchor is a MAMBA decision, so it returns
            # `None` ("no constraint from me"), not 0. The shared
            # `effective_cache_len` in `UnifiedRadixCache.cache_finished_req`
            # is a `min` across components; a 0 here collapsed it and threw
            # away the full-attention KV too, which does not depend on the
            # mamba grid at all. The node is inserted at full length with a
            # mamba tombstone, and the match walk still refuses it as a resume
            # anchor because the MAMBA validator sees no state.
            if cache_len is not None and not is_on_interval(
                cache_len, self.mamba_checkpoint_interval
            ):
                self._off_grid_retention_declines += 1
                count = self._off_grid_retention_declines
                if count <= 3 or count % 1000 == 0:
                    logger.warning(
                        "mamba checkpoint interval: off-grid tracked position "
                        "%d (interval %d), caching KV without a mamba anchor, "
                        "occurrence=%d, rid=%s",
                        cache_len,
                        self.mamba_checkpoint_interval,
                        count,
                        req.rid,
                    )
                insert_params.mamba_value = None
                return _decline_retention(is_finished)
        else:
            cache_len = token_ids_len
            # ReplaySSM (no_buffer): `temporal[slot]` lags the live state by the
            # slot's unflushed ring depth (`write_pos`), so on request finish cap
            # the donate to the last flush boundary (where temporal is current)
            # and reset the cursor, keeping the donated checkpoint consistent with
            # its key length. page_size is asserted == 1, so no realign. Mirrors
            # MambaRadixCache.cache_finished_req.
            if is_finished:
                write_pos_buf = (
                    self.cache.req_to_token_pool.mamba_pool.replayssm_write_pos
                )
                if write_pos_buf is not None:
                    cache_len -= int(write_pos_buf[req.mamba_pool_idx].item())
                    write_pos_buf[req.mamba_pool_idx] = 0
            # #747 retention seam (mirrors mamba_radix_cache.py:652-659 and
            # :795-809): the donated state sits exactly at `cache_len`; there
            # is no mechanism to snapshot an earlier position, so an off-grid
            # end gets NO ANCHOR. The cursor reset above must still happen.
            #
            # #783: this is the branch the rig actually ran (no
            # --enable-mamba-extra-buffer), and it was BOTH a total veto and
            # silent -- the two properties that made an instance-wide dark
            # cache survive 2892 prefill batches without one log line. An
            # interval coarser than the traffic (8192 against ~6k-token
            # prompts) puts EVERY request end off-grid, so `return 0` meant
            # the tree never retained anything and every prefill read
            # `#cached-token: 0`. Retention now survives the missing anchor
            # (see the extra_buffer branch above for the full rationale), and
            # the decline is counted so a structurally dark grid announces
            # itself instead of looking like idle traffic.
            if not is_on_interval(cache_len, self.mamba_checkpoint_interval):
                self._off_grid_retention_declines += 1
                count = self._off_grid_retention_declines
                if count <= 3 or count % 1000 == 0:
                    logger.warning(
                        "mamba checkpoint interval: off-grid end %d (interval "
                        "%d), caching KV without a mamba anchor, "
                        "occurrence=%d, rid=%s",
                        cache_len,
                        self.mamba_checkpoint_interval,
                        count,
                        getattr(req, "rid", None),
                    )
                insert_params.mamba_value = None
                return _decline_retention(is_finished)

        # #824: BOTH branches above converge here, and either can hand back a
        # position BELOW what this request has already published as
        # `cache_protected_len` -- the tracked position is not monotone
        # (`mamba_branching_seqlen`, schedule_batch.py:2660, can sit under an
        # earlier track). Capping the shared `effective_cache_len` there would
        # retain FEWER indices than the tree already owns, and
        # `cache_unfinished_req`'s assert catches it by killing the rank:
        # measured 2026-08-23 06:07:06, protected 16384 against 8192 retained.
        #
        # Declining, not flooring, for the reason stated two branches up: the
        # donated state sits at the tracked position, so correcting the LENGTH
        # would pair state and key at different positions. `_decline_retention`
        # gives the per-caller answer (#783) -- 0 for an unfinished step, None
        # for a finished one -- which is why the decision is routed through it
        # rather than returned from here.
        if retention_shrinks_protected(cache_len, req.cache_protected_len):
            self._protected_retention_declines += 1
            count = self._protected_retention_declines
            if count <= 3 or count % 1000 == 0:
                logger.warning(
                    "mamba retention would truncate a protected prefix: "
                    "tracked position %s under cache_protected_len %d, "
                    "caching without a mamba anchor, occurrence=%d, rid=%s",
                    cache_len,
                    int(req.cache_protected_len or 0),
                    count,
                    getattr(req, "rid", None),
                )
            insert_params.mamba_value = None
            return _decline_retention(is_finished)

        if is_finished:
            if cache_len is None:
                # #1012: NO TRACKED POSITION IS THE SAME ANSWER AS AN OFF-GRID
                # ONE, AND IT WAS THE ONLY ONE STILL RETURNING 0.
                #
                # `cache_len = req.mamba_last_track_seqlen` is None whenever the
                # request never reached a checkpoint boundary -- with
                # `--mamba-checkpoint-interval 4096` that is every request
                # shorter than 4096 tokens, i.e. all of them on a probe or a
                # chat turn. The off-grid branch above cannot catch it (`cache_len
                # is not None and not is_on_interval(...)`), so control fell here
                # and `cache_len = 0` collapsed the shared `effective_cache_len`
                # in `UnifiedRadixCache.cache_finished_req` to 0: the key is
                # truncated to nothing, no node is inserted, and every row is
                # freed.
                #
                # That is precisely the veto #783 removed from the two branches
                # above, still standing on the third. It is the same "dark cache"
                # signature that ticket names -- the tree retains nothing and
                # every prefill reads `#cached-token: 0` -- and it was measured
                # here: after three finished requests the TREE CENSUS read
                # `nodes=1 ... recomputed_evictable=0`, an empty tree.
                #
                # The answer is the one `_decline_retention` already gives for a
                # FINISHED caller: decline the ANCHOR (`mamba_value=None`, the
                # tombstone `commit_insert_component_data` documents and
                # accepts), impose NO constraint on the length. The
                # full-attention KV does not depend on the mamba grid, so it
                # stays cached at full length while the match walk keeps
                # refusing the node as a resume point -- exactly the split #783
                # established.
                self._off_grid_retention_declines += 1
                count = self._off_grid_retention_declines
                if count <= 3 or count % 1000 == 0:
                    logger.warning(
                        "mamba checkpoint interval: no tracked position at "
                        "finish (interval %s), caching KV without a mamba "
                        "anchor, occurrence=%d, rid=%s",
                        self.mamba_checkpoint_interval,
                        count,
                        getattr(req, "rid", None),
                    )
                insert_params.mamba_value = None
                return _decline_retention(is_finished)
            if self.enable_mamba_extra_buffer:
                keep_idx = self.cache.req_to_token_pool.get_mamba_ping_pong_keep_idx(
                    req
                )
                active_value = (
                    req.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1).clone()
                )
            else:
                active_value = req.mamba_pool_idx.unsqueeze(-1).clone()
            if self.int8_ckpt_pool is not None:
                # `None` when the checkpoint pool is exhausted with nothing
                # evictable: the node is then inserted without a mamba value
                # (a tombstone the tree already supports), so the KV stays
                # cached and only the mamba resume point is lost.
                # `cleanup_after_caching_req` handles a `None` mamba_value.
                insert_params.mamba_value = self._commit_int8_checkpoint(active_value)
            else:
                insert_params.mamba_value = active_value
            return cache_len
        else:
            if cache_len is None:
                return 0
            # Donate the mamba index to the radix cache instead of copying.
            # CACHE-INSERT path: an unservable slot degrades to "cache nothing
            # this step" (returning 0 makes UnifiedRadixCache take its
            # effective_cache_len <= 0 skip branch), never to a crash. Every
            # allocation that can fail happens BEFORE the ping-pong swap so a
            # skip cannot leave the request's buffer half-donated.
            if self.int8_ckpt_pool is not None:
                if self.enable_mamba_extra_buffer:
                    new_slot = self._alloc_mamba_slot()
                    ckpt_slot = (
                        None if new_slot is None else self._alloc_int8_ckpt_slot()
                    )
                    if ckpt_slot is None:
                        if new_slot is not None:
                            self.cache.req_to_token_pool.mamba_allocator.free(new_slot)
                        return 0
                    src_active = (
                        self.cache.req_to_token_pool.donate_mamba_ping_pong_slot(
                            req, new_slot
                        )
                    )
                    self.int8_ckpt_pool.store_from_active(
                        active_mamba_state_pool(self.cache),
                        src_active.view(-1),
                        ckpt_slot,
                    )
                    mamba_value_donated = ckpt_slot
                    self.cache.req_to_token_pool.mamba_allocator.free(src_active)
                else:
                    mamba_value_donated = self._commit_int8_checkpoint(
                        req.mamba_pool_idx.view(-1)
                    )
                    if mamba_value_donated is None:
                        return 0
            elif self.enable_mamba_extra_buffer:
                new_slot = self._alloc_mamba_slot()
                if new_slot is None:
                    return 0
                mamba_value_donated = (
                    self.cache.req_to_token_pool.donate_mamba_ping_pong_slot(
                        req, new_slot
                    )
                )
            else:
                mamba_value_donated = self._alloc_mamba_slot()
                if mamba_value_donated is None:
                    return 0
                # mamba_pool is a pure PHYSICAL store; translate both slot ids
                # virtual->physical (identity for the non-unified memory pool) first.
                translate = self.cache.req_to_token_pool.translate_mamba_indices
                # #767: THE STATE BYTES LIVE IN THE ACTIVE PHASE'S POOL.
                # Under a phase-flip build the bound pool is the primary PP
                # stack's; copying there while the TP stack computes
                # duplicated whatever stale bytes the PP tensors still held
                # at the request's slot into the checkpoint (measured as a
                # foreign request's essay resumed into a fresh prompt).
                # Bookkeeping (allocator, translate) stays on the bound
                # pool; only the byte copy follows the computing stack.
                active_mamba_state_pool(self.cache).copy_from(
                    translate(req.mamba_pool_idx.unsqueeze(0)),
                    translate(mamba_value_donated),
                )
            insert_params.mamba_value = mamba_value_donated
            return cache_len

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: Optional[InsertResult] = None,
        insert_params: Optional[InsertParams] = None,
    ) -> None:
        if is_finished:
            mamba_value_inserted = (
                insert_result is not None and not insert_result.mamba_exist
            )
            pool = self.cache.req_to_token_pool

            if self.int8_ckpt_pool is not None:
                insert_value_unused = (
                    not mamba_value_inserted
                    and insert_params is not None
                    and insert_params.mamba_value is not None
                )
                if insert_value_unused:
                    self._free_mamba_value(insert_params.mamba_value)
                pool.free_mamba_cache(req)
                return

            if self.enable_mamba_extra_buffer:
                keep_idx = (
                    pool.get_mamba_ping_pong_keep_idx(req)
                    if mamba_value_inserted
                    else None
                )
                pool.free_mamba_cache(
                    req, mamba_ping_pong_track_buffer_to_keep=keep_idx
                )
                return

            if not mamba_value_inserted:
                # #929 THE DONATION IS A SEPARATE SLOT AND IT NEEDED ITS OWN
                # RELEASE. On this path `_donate_mamba_value` ALLOCATES a fresh
                # slot (:902-903) and copies the state into it, so
                # `insert_params.mamba_value` is NOT `req.mamba_pool_idx`.
                # `free_mamba_cache(req)` releases only the request's slot, so
                # when the tree did not take the donation nothing released it:
                # it is on neither the free list nor in any node, which is
                # exactly what the on-idle ledger reported as
                # `leaked_mamba_pages={11}` (window 2g, one slot of twenty,
                # available=19 against peers at 20/20).
                #
                # This mirrors, rather than invents, what the two sibling paths
                # already do: the int8 branch above (`insert_value_unused`) and
                # the not-finished branch below both release an unused
                # donation. Only this one lacked it.
                #
                # GUARDED ON `not mamba_value_inserted` DELIBERATELY. If the
                # tree DID take the donation it owns live state, and returning
                # it here would let `alloc()` hand one Mamba state to two
                # requests -- a wrong answer that never crashes, strictly worse
                # than the leak this replaces. `insert_params` is Optional in
                # the signature (see the `is_finished=True` callsite at
                # `unified_radix_cache.py:1081`, which passes neither result
                # nor params), so it is checked rather than assumed.
                # AND THE DONATION IS NOT ALWAYS A SECOND SLOT. `_donate_mamba_
                # value` allocates a fresh one only on the branch at :1030; the
                # branch taken when there is no int8 checkpoint pool builds
                # `active_value = req.mamba_pool_idx.unsqueeze(-1).clone()`
                # (:956) and assigns THAT (:965). There the donation IS the
                # request's own slot, and `free_mamba_cache(req)` below already
                # releases it -- freeing both handles is one slot returned
                # twice. Measured: boot 2i died on it, MambaSlotDoubleFree on
                # slot 6, and the first version of this very fix is what
                # produced it. So the release is conditioned on the donation
                # being a DIFFERENT slot, compared by id and not by identity:
                # the clone is a distinct tensor object carrying the same value.
                donated = insert_params.mamba_value if insert_params is not None else None
                if donated is not None and not _same_mamba_slot(
                    donated, getattr(req, "mamba_pool_idx", None)
                ):
                    self._free_mamba_value(donated)
                pool.free_mamba_cache(req)
                return

            # #1051 THE OTHER HALF OF #929, AND IT IS THE ONE THAT KILLED SIX
            # BOOTS. The block above reasons about the donation the tree did
            # NOT take. When the tree DID take it, this path still ran
            # `free_mamba_cache(req)` unconditionally -- and on THIS branch
            # (`no_buffer`, no int8 checkpoint pool) the donation IS the
            # request's own active slot: `prepare_for_caching_req` builds
            # `active_value = req.mamba_pool_idx.unsqueeze(-1).clone()` (:1109)
            # and hands THAT to the tree (:1118). So the node holds slot X and
            # the allocator is told X is free, in the same call.
            #
            # The consequence is not a crash here; it is a crash three passes
            # later, in a stranger's stack. Measured, boot 23
            # (boot_855_1033c_0840f82601_0831_134138.log, 13:45:39Z, all three
            # ranks, slot 1): `alloc_group_begin` legitimately re-drew the slot
            # from the free list, nothing consumed it, `alloc_group_end`
            # returned the remainder -- and the #1033b provenance named THAT as
            # the FIRST RELEASER, with the node's eviction as the second. Both
            # were innocent: the free list and the tree had held the same slot
            # since this line ran.
            #
            # THE GUARD IS NOT NEW CODE, IT IS A PORT OMISSION. The sibling
            # implementation this component replaced states the same rule
            # explicitly (`mamba_radix_cache.py:781-792`):
            #     free_mamba_cache = True if (extra_buffer or int8) else mamba_exist
            # i.e. on the plain path the active slot goes back ONLY when the
            # tree refused the donation. That term was lost in the move to the
            # unified component; this restores it rather than inventing a
            # second bookkeeping for ownership.
            #
            # Compared BY VALUE (`_same_mamba_slot`), because the donation is a
            # clone: a distinct tensor carrying the same id. An unreadable
            # handle answers False and takes the free path, which is the old
            # behaviour -- narrower than the defect, never wider.
            donated = insert_params.mamba_value if insert_params is not None else None
            if _same_mamba_slot(donated, getattr(req, "mamba_pool_idx", None)):
                pool.relinquish_mamba_cache(req)
            else:
                pool.free_mamba_cache(req)
        else:
            if insert_params.mamba_value is not None and (
                insert_result is None or insert_result.mamba_exist
            ):
                self._free_mamba_value(insert_params.mamba_value)
            req.mamba_last_track_seqlen = None

    # ---- HiCache Hooks ----

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        req: Optional[Req] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            cd = node.component_data[ct]
            # #969H BACKUP DECISION PROBE (temporary). §N1 established that this
            # writer is PRESENT AND WIRED and declines only on an empty node.
            # The open question is which of two things the retention path does:
            #   "inserts a node WITHOUT mamba state"  -> we are called, value None
            #   "inserts no node at all"              -> we are never called
            # Reaching this line at all answers half of it; the flag answers the
            # rest. Grep: "#969H BACKUP".
            try:
                _n = getattr(type(self), "_969h_n", 0) + 1
                type(self)._969h_n = _n
                _has = cd.value is not None
                _k = "has_value" if _has else "EMPTY"
                _c = getattr(type(self), "_969h_counts", None)
                if _c is None:
                    _c = type(self)._969h_counts = {}
                _c[_k] = _c.get(_k, 0) + 1
                if _n <= 40 or _n % 256 == 0:
                    logger.warning(
                        "#969H BACKUP n=%d mamba_value=%s counts=%s",
                        _n,
                        _k,
                        _c,
                    )
            except Exception:  # noqa: BLE001
                logger.warning("#969H BACKUP PROBE RAISED", exc_info=True)
            if cd.value is None:
                return None
            return [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    device_indices=cd.value,
                )
            ]

        if phase == CacheTransferPhase.LOAD_BACK:
            transfers: list[PoolTransfer] = []

            cd = node.component_data[ct]
            if cd.value is not None:
                return None

            # restore single node if host_value exists
            if cd.host_value is not None and cd.value is None:
                transfers.append(
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=cd.host_value,
                        nodes_to_load=[node],
                    )
                )

            # Per-request mamba CoW (H→D copy into request's device slot)
            cd = node.component_data[ct]
            if req is not None and cd.host_value is not None:
                if req.mamba_pool_idx is None:
                    dst = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
                    if dst is None:
                        self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                        dst = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
                    elif peer_needs_mamba_evict(self.cache):
                        # #639b: see `_alloc_mamba_slot`. Host load-back sibling.
                        self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                    if dst is not None:
                        req.mamba_pool_idx = dst[0]
                        # #991: host load-back sibling of the COW acquire.
                        req.mamba_slot_acquired_this_admission = True
                    else:
                        # #968 FIX-4: FAIL THE WHOLE LOAD-BACK, NOT HALF OF IT.
                        #
                        # This branch used to log and fall through, on the
                        # justification "the request re-prefills the segment,
                        # which is a slowdown, not a dead scheduler". That
                        # justification is the #581 sibling's, and it does not
                        # hold HERE: on this path the KV prefix does NOT get
                        # re-prefilled -- `add_one_req` grows
                        # `req.prefix_indices` to the told extent immediately
                        # after this call returns. The request would then carry
                        # a prefix booked as warm over a GDN state that was
                        # never loaded, and the scan would resume at the told
                        # position from a foreign or zeroed slot. Silently
                        # wrong tokens, no assert reachable.
                        #
                        # Same answer as the ordinary prefix-resume COW gives
                        # (`finalize_match_result` -> `zero_match_result`):
                        # KV extent and GDN state stand or fall together.
                        self._log_mamba_slot_starvation("mamba (host load-back)")
                        raise MambaLoadBackUnservable(
                            "mamba slot pool exhausted with nothing evictable "
                            "during a host load-back for rid="
                            f"{getattr(req, 'rid', '?')}: refusing the whole "
                            "load-back rather than growing the KV prefix over "
                            "an unloaded GDN state."
                        )
            if (
                req is not None
                and cd.host_value is not None
                and req.mamba_pool_idx is not None
            ):
                transfers.append(
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=cd.host_value,
                        device_indices=req.mamba_pool_idx.unsqueeze(0),
                    )
                )
                # #968 FIX-3 (writer): this load-back plants the node-END
                # anchor -- the state AFTER `node`'s LAST token -- into the
                # request's own slot. The caller needs to know that, because
                # its S1 clamp may cut the KV indices SHORT of that position
                # and cannot cut this transfer (it is already built, and by
                # the time the caller sees the length it has already run).
                # A state ahead of its prefix is the hazard spelled out at
                # the BACKUP_STORAGE comment below: the reader continues from
                # a state that does not belong to that prefix.
                # Reset by the caller immediately before `init_load_back`, so
                # writer, reader and deleter share one straight-line pass.
                req.mamba_loadback_anchor_adopted = True

            return transfers if transfers else None

        if phase == CacheTransferPhase.BACKUP_STORAGE:
            cd = node.component_data[ct]
            if cd.host_value is None or not node.hash_value:
                return None
            # #1039: THE ANCHOR IS PUBLISHED AT THE NODE END, and a reader can
            # never request past input_len-1 (upstream cap, consumed at
            # scheduler.py -> Req._compute_max_prefix_len). A node whose end
            # coincides with a prompt's own end is therefore UNREACHABLE for an
            # identical repeat of that prompt -- at ANY anchor density, because
            # density cannot rescue a span that stops one page short. Measured
            # 2026-08-30 by recomputing the hash chain offline against the anchor
            # blobs on disk: a 350-token prompt's only anchor sits at prefix
            # length 350, the read span ends at 349.
            #
            # This is accepted, not repaired: the loss is bounded by one chunk
            # (worked arithmetic at the _match_end comment in scheduler.py), which
            # is the #939 bound. Earlier chunk-end anchors of a multi-chunk prompt
            # stay reachable -- for a 4618-token prompt the anchor at 4096 lies
            # inside the read span -- and that is where a warm hit comes from.
            #
            # HAZARD, if someone "fixes" the off-by-one by keying this at
            # hash_value[-2]: the state in cd.host_value is the mamba state AFTER
            # the node's LAST token. Filing it under the prefix-length-minus-one
            # key would advertise a state that does not belong to that prefix, and
            # the reader would continue from it -- a silently wrong continuation
            # instead of a miss. Shifting the KEY requires capturing the STATE at
            # that position too; the two are not separable.
            return [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=cd.host_value,
                    keys=[node.hash_value[-1]],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        if phase == CacheTransferPhase.PREFETCH:
            host_indices = self._mamba_pool_host.alloc(1)
            if host_indices is None:
                self.cache.evict_host(1, ComponentType.MAMBA)
                host_indices = self._mamba_pool_host.alloc(1)
            if host_indices is None:
                return []
            return [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=host_indices,
                    keys=["__placeholder__"],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                cd = node.component_data[ct]
                if cd.host_value is None:
                    cd.host_value = transfers[0].host_indices.clone()

        elif phase == CacheTransferPhase.LOAD_BACK:
            if not transfers:
                return
            transfer = transfers[0]
            if transfer.device_indices is not None:
                cd = node.component_data[ct]
                cd.value = transfer.device_indices.clone()
                count = len(cd.value)
                # Move from host LRU to device LRU
                host_lru = self.cache.host_lru_lists[ct]
                if host_lru.in_list(node):
                    host_lru.remove_node(node)
                self.cache.lru_lists[ct].insert_mru(node)
                self.cache.component_evictable_size_[ct] += count

        elif phase == CacheTransferPhase.PREFETCH:
            if not transfers:
                return
            transfer = transfers[0]
            host_indices = transfer.host_indices
            loaded = (
                pool_storage_result is not None
                and pool_storage_result.extra_pool_hit_pages.get(PoolName.MAMBA, 0) >= 1
            )
            target_node = (
                insert_result.inserted_host_node if insert_result is not None else None
            )
            if (
                host_indices is None
                or target_node is None
                or not loaded
                or target_node.component_data[ct].host_value is not None
            ):
                self.cache.cache_controller.append_host_mem_release(
                    extra_pools=[transfer]
                )
                if insert_result is not None:
                    insert_result.mamba_exist = True
                return

            target_node.component_data[ct].host_value = host_indices.clone()
            if target_node.component_data[ct].value is None:
                host_lru = self.cache.host_lru_lists[ct]
                if not host_lru.in_list(target_node):
                    host_lru.insert_mru(target_node)
            if insert_result is not None:
                insert_result.mamba_exist = False

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict mamba host resources.
        Internal nodes: private tombstone (free host mamba only).
        Host leaves: atomic eviction via _evict_host_leaf."""
        ct = self.component_type
        host_lru = self.cache.host_lru_lists[ct]
        x = host_lru.get_lru_no_host_lock()
        while tracker[ct] < num_tokens and x is not None and host_lru.in_list(x):
            x_next = host_lru.get_prev_no_host_lock(x)
            cd = x.component_data[ct]
            if x in self.cache.evictable_host_leaves:
                # Host leaf: atomic eviction (all components host + delete)
                self.cache._evict_host_leaf(x, tracker)
            else:
                # Internal: tombstone Mamba + cascade
                assert cd.host_value is not None
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.HOST, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker, target=EvictLayer.HOST)
                self.cache._update_evictable_leaf_sets(x)
            x = x_next
