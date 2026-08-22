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
from sglang.srt.mem_cache.mamba_state_pool import active_mamba_state_pool
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
            node.component_data[self.component_type].value = params.mamba_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(
                params.mamba_value
            )
            return
        if node.component_data[self.component_type].value is None:
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

        if is_finished:
            if cache_len is None:
                cache_len = 0
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
                    else:
                        # #581 sibling site: exhausted with nothing evictable.
                        # Skip the host->device restore instead of asserting;
                        # the request re-prefills the segment, which is a
                        # slowdown, not a dead scheduler.
                        self._log_mamba_slot_starvation("mamba (host load-back)")
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

            return transfers if transfers else None

        if phase == CacheTransferPhase.BACKUP_STORAGE:
            cd = node.component_data[ct]
            if cd.host_value is None or not node.hash_value:
                return None
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
