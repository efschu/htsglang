from __future__ import annotations

import logging
import time
from array import array

from sglang.srt.environ import envs
from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor
from sglang.srt.utils import get_bool_env_var

_ROUTING_KEY_POLICY_DEBUG_LOG = get_bool_env_var("SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG")
logger = logging.getLogger(__name__)

#: #988: rate-limit state for the load-back instrument, module-level.
_988_LOADBACK_SEEN = {"n": 0}


def _note_988_loadback(req, new_prefix_len: int) -> None:
    """#988 instrument (standing order): the prefix moved; say so.

    Module-level, not a method (the holder lesson, three measured
    instances). Prints the first occurrence and every 64th; the boot-8
    falsifier question -- was the request already in a batch when its
    prefix moved -- is answered by the caller-side guard's skip census
    ('already_in_batch'), so this line carries the geometry facts only.
    """
    _988_LOADBACK_SEEN["n"] += 1
    n = _988_LOADBACK_SEEN["n"]
    if n == 1 or n % 64 == 0:
        logger.info(
            "#988 LOADBACK rid=%s prefix moved to %d, extend_range re-derived "
            "to the parked shape at the mutation (seen=%d)",
            getattr(req, "rid", None),
            new_prefix_len,
            n,
        )


#: #1035: rank-local host load-backs refused by the PP congruence rule.
_1035_LOADBACK_REFUSED = {"n": 0}


def _pp_forbids_rank_local_load_back(req) -> bool:
    """#1035: under PP a host load-back is a rank-local prefix mutation.

    THE BOOT THIS CLOSES. window-flip-0828 boot `1815081d46` died at 23:40:52
    with `Bar1CollectiveAborted` on all three ranks. Ninety-four consecutive
    `#969 EXTENT` events were rank-uniform; the ninety-fifth was not, and it
    was the boot's ONLY host load-back:

        PP0  rid=5373b3a7  prefix 8541  extend   1   (host_hit=349, mamba=1)
        PP1  rid=5373b3a7  prefix 8192  extend 350   (host_hit=0)
        PP2  rid=5373b3a7  prefix 8192  extend 350   (host_hit=0)

    PP0 then ran a 1-row forward while its peers ran 350-row ones, the
    attention-TP `all_reduce` never matched, and every rank's spin kernel sat
    on its cycle deadline until the abort check raised. The message says "a
    peer did not arrive"; the peer did arrive, at a different shape.

    WHY THIS IS CONSTRUCTION AND NOT DETECTION. Under PP each stage holds a
    different LAYER slice, so the host tier is layer-partitioned and
    `host_hit_length` is non-uniform BY CONSTRUCTION -- there is no value of
    it that all ranks can be relied on to share, and no amount of checking
    afterwards makes one. A mechanism that can put two ranks into different
    admission shapes may not exist; it is not something to detect and
    compensate on the next pass.

    THE RULE ALREADY EXISTS ONE PATH OVER, AND THIS IS ITS SIBLING. The
    forwarded-schedule path refuses the identical load-back for the identical
    reason -- "a load-back is a rank-local improvement to a quantity this rank
    no longer owns, so on this path it does not run" (the instr20 line,
    `_add_one_req_from_schedule`). Instr20 was the MIRROR of this death (PP1
    and PP2 grew, PP0 did not), and the rule was fixed only on the path that
    death happened to be found on. The self-building path kept the defect and
    spent a boot proving it.

    THE COST, STATED RATHER THAN ASSUMED. The revived tokens are recomputed
    instead. Measured on the dead boot: 349 tokens, once across 57 cutovers,
    against a HiCache chunk of 8192 -- an order of magnitude inside the
    standing "at most one chunk of recompute" bound, and paid per re-admission
    rather than per token. The counter below is what re-opens this trade if
    the frequency ever stops being negligible.

    NOT A DELETION OF HOST LOAD-BACK. `pp_size == 1` -- plain TP, and every
    upstream configuration -- is untouched: there the host tier is sharded by
    HEAD, holds the same tokens on every rank, and the hit is uniform.
    """
    if int(getattr(get_server_args(), "pp_size", 1) or 1) <= 1:
        return False
    _1035_LOADBACK_REFUSED["n"] += 1
    n = _1035_LOADBACK_REFUSED["n"]
    if n <= 5 or n % 64 == 0:
        logger.warning(
            "#1035 RANK-LOCAL LOAD-BACK REFUSED rid=%s: this rank has a host "
            "hit (kv=%s swa=%s mamba=%s) that its peers need not have, and "
            "applying it would grow this rank's prefix alone -- the shape "
            "divergence that killed boot 1815081d46 in the attention-TP "
            "all_reduce. The prefix stays at the rank-uniform match and these "
            "tokens are recomputed. occurrence=%d",
            getattr(req, "rid", None),
            getattr(req, "host_hit_length", None),
            getattr(req, "swa_host_hit_length", None),
            getattr(req, "mamba_host_hit_length", None),
            n,
        )
    return True


#: #967: how many times the #959 "one continuation at a time" guard has
#: refused a FRESH request in this process, per mint site.
#:
#: THE GUARD WAS UNOBSERVABLE, and that is the whole posten. #959 is closed by
#: two bare `return AddReqResult.OTHER` statements with no trace of any kind,
#: so whether they ever fire is not readable from any boot log -- a refusal
#: that leaves no trace is indistinguishable from a scheduler that simply
#: built nothing, which is precisely the state the next window has to tell
#: apart. Its neighbour `Scheduler._note_seam_chunk_refused` shows the
#: counter-pattern and this follows it: unconditional count, first three
#: occurrences logged, then every thousandth, so a guard that fires every
#: round costs a handful of lines rather than one per iteration.
#:
#: MODULE-LEVEL, not on the adder: `PrefillAdder` is rebuilt every pass, so an
#: instance counter would reset before anyone could read it. The question this
#: answers -- "was this guard reached at all in this boot" -- is a
#: process-lifetime question.
_SECOND_CONTINUATION_REFUSALS = {}  # site -> count (str -> int)


def note_second_continuation_refused(req, site: str) -> int:
    """Count and (rate-limited) name one #959 refusal. Returns the new count."""
    n = _SECOND_CONTINUATION_REFUSALS.get(site, 0) + 1
    _SECOND_CONTINUATION_REFUSALS[site] = n
    if n <= 3 or n % 1000 == 0:
        logger.info(
            "[#967] SECOND CONTINUATION REFUSED rid=%s site=%s: a resident "
            "chunked request is still outstanding, so this FRESH request is "
            "left for a later pass rather than minted as a second "
            "continuation (#959). Nothing of it has run, so no progress is "
            "lost and no double prefill is incurred; it is admitted as soon "
            "as the resident continuation finishes. occurrence=%d",
            getattr(req, "rid", None),
            site,
            n,
        )
    return n


# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Request scheduler policy"""

import os
import random
from collections import Counter, defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_in_seq_split
from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled
from sglang.srt.managers.kv_session_offload import prefill_spill_deep_ok
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.planner.chunked_admission import (
    ChunkedCommitmentLedger,
    effective_rem_total_tokens,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.runtime_context import get_server_args
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (against existing cache) less than this value,
# the scheduler runs the in-batch prefix caching check for this request.
# If we set it to -1, it means we disable in-batch prefix caching.
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (within the waiting queue) larger than this value,
# the scheduler deprioritizes this request
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")
)


IGNORE_EOS_RESERVE_TOKENS = 1


def match_prefix_for_req(
    tree_cache: BasePrefixCache,
    req: Req,
    token_ids: Optional[array[int]] = None,
    *,
    cow_mamba: bool = False,
    include_req: bool = False,
):
    if token_ids is None:
        token_ids = req.origin_input_ids + req.output_ids

    # unified_kv SWA lives in a per-request ring that's not content-stable and is
    # never stored in the radix tree, so a reused prefix carries stale SWA. Cap
    # the match by the trailing sliding window so it gets re-prefilled, rewriting
    # this request's SWA ring. No-op for other layouts.
    reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
    key_limit = max(0, len(token_ids) - reprefill_tail) if reprefill_tail else None

    match_result = tree_cache.match_prefix(
        MatchPrefixParams(
            key=RadixKey(token_ids=token_ids, extra_key=req.extra_key, limit=key_limit),
            cow_mamba=cow_mamba,
            req=req if include_req else None,
        )
    )
    if envs.SGLANG_RADIX_FORCE_MISS.get():
        match_result = zero_match_result(tree_cache, match_result)
    (
        req.prefix_indices,
        req.last_node,
        req.last_host_node,
        req.best_match_node,
        req.host_hit_length,
        req.swa_host_hit_length,
        req.mamba_host_hit_length,
    ) = (
        match_result.device_indices,
        match_result.last_device_node,
        match_result.last_host_node,
        match_result.best_match_node,
        match_result.host_hit_length,
        match_result.swa_host_hit_length,
        match_result.mamba_host_hit_length,
    )
    max_len = req._compute_max_prefix_len(len(token_ids))
    req.num_matched_prefix_tokens = min(
        len(req.prefix_indices) + req.host_hit_length, max_len
    )
    if match_result.mamba_branching_seqlen is not None:
        req.mamba_branching_seqlen = match_result.mamba_branching_seqlen
    # #927: THE TWO SETTERS MUST NOT GUESS DIFFERENTLY. This assigns
    # `req.prefix_indices` unconditionally, and on a hit those ARE the tree's
    # row ids -- the request reuses them, it does not copy them. The only thing
    # that later stops `_insert_helper`'s duplicate free from running over them
    # is `req.cache_protected_len`, which arrives there as
    # `InsertParams.prev_prefix_len`.
    #
    # `MatchResult.cache_protected_len` defaults to None and `UnifiedRadixCache`
    # never populates it, so on this rig the branch below never fired and the
    # field kept whatever it held -- 0 for a fresh Req
    # (`schedule_batch.py:1677`). Its sibling `Req.init_next_round_input`
    # already falls back to `len(self.prefix_indices)` for exactly this case
    # (`schedule_batch.py:1351-1354`); this one did not, so which value a
    # request carried depended on which of the two touched it last.
    #
    # Measured consequence, with `prev_prefix_len=0` and a full prefix hit:
    # every row of the hit prefix ends up in the free list AND in the tree at
    # once -- the `double_owned=N src=live` population, pinned in
    # test_insert_dup_free_927.py.
    if match_result.cache_protected_len is not None:
        req.cache_protected_len = match_result.cache_protected_len
    else:
        req.cache_protected_len = len(req.prefix_indices)
    return match_result


class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""

    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting


class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""

    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first
    RANDOM = "random"
    ROUTING_KEY = "routing-key"  # prioritize by routing key frequency in running batch


class SchedulePolicy:
    Policy = Union[CacheAwarePolicy, CacheAgnosticPolicy]

    def __init__(
        self,
        policy: str,
        tree_cache: BasePrefixCache,
        enable_hierarchical_cache: bool,
        enable_priority_scheduling: bool,
        schedule_low_priority_values_first: bool,
        enable_fast_lane: bool = False,
        fast_lane_priority: int = 0,
        fast_lane_heavy_aging_ms: float = 0.0,
    ):
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)
        self.tree_cache = tree_cache
        self.enable_hierarchical_cache = enable_hierarchical_cache
        self.enable_priority_scheduling = enable_priority_scheduling
        self.schedule_low_priority_values_first = schedule_low_priority_values_first
        self.priority_sign = 1 if schedule_low_priority_values_first else -1
        # Fast lane (Variant C Stage 0) anti-starvation aging config.
        self.enable_fast_lane = enable_fast_lane
        self.fast_lane_priority = fast_lane_priority
        self.fast_lane_heavy_aging_ms = fast_lane_heavy_aging_ms

        # It is used to find the matching prefix for in-batch prefix caching.
        self.waiting_queue_radix_tree = RadixCache.create_simulated()

    def calc_priority(
        self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = None
    ) -> None:
        policy = self._determine_active_policy(waiting_queue)

        # Populate req.num_matched_prefix_tokens at schedule time. Cache-aware policies
        # set it in _compute_prefix_matches; do the same full match for
        # cache-agnostic policies when the radix supports it, so the load
        # snapshot has it. Skip on decode (never prefills).
        if (
            not isinstance(policy, CacheAwarePolicy)
            and self.tree_cache.supports_fast_match_prefix()
            and get_server_args().disaggregation_mode != "decode"
        ):
            for r in waiting_queue:
                match_prefix_for_req(self.tree_cache, r, include_req=True)

        if self.policy == CacheAgnosticPolicy.FCFS:
            if self.enable_priority_scheduling:
                SchedulePolicy._sort_by_priority_and_fcfs(
                    waiting_queue,
                    self.priority_sign,
                    enable_fast_lane=self.enable_fast_lane,
                    fast_lane_priority=self.fast_lane_priority,
                    heavy_aging_ms=self.fast_lane_heavy_aging_ms,
                )
            return

        if isinstance(policy, CacheAwarePolicy):
            temporary_deprioritized = self._compute_prefix_matches(
                waiting_queue, policy
            )
            if policy == CacheAwarePolicy.LPM:
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
            elif policy == CacheAwarePolicy.DFS_WEIGHT:
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
            else:
                raise ValueError(f"Unknown CacheAware Policy: {policy=}")
        else:
            if policy == CacheAgnosticPolicy.FCFS:
                pass
            elif policy == CacheAgnosticPolicy.LOF:
                SchedulePolicy._sort_by_longest_output(
                    waiting_queue,
                    self.enable_priority_scheduling,
                    self.priority_sign,
                )
            elif policy == CacheAgnosticPolicy.RANDOM:
                SchedulePolicy._sort_randomly(waiting_queue)
            elif policy == CacheAgnosticPolicy.ROUTING_KEY:
                if running_batch is not None:
                    SchedulePolicy._sort_by_routing_key(waiting_queue, running_batch)
            else:
                raise ValueError(f"Unknown CacheAgnostic Policy: {policy=}")

    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS
        return self.policy

    def _validate_and_adjust_policy(
        self, policy: str, tree_cache: BasePrefixCache
    ) -> Policy:
        """
        Validates the policy and adjusts it if necessary based on tree cache settings.
        """
        try:
            policy_enum = CacheAwarePolicy(policy)
            if getattr(tree_cache, "disable", True):
                # If tree_cache is disabled, using CacheAgnosticPolicy policy
                return CacheAgnosticPolicy.FCFS
            return policy_enum
        except ValueError:
            try:
                return CacheAgnosticPolicy(policy)
            except ValueError:
                raise ValueError(f"Unknown schedule_policy: {policy=}")

    def _compute_prefix_matches(
        self, waiting_queue: List[Req], policy: CacheAwarePolicy
    ) -> Set[int]:
        """
        Computes and caches the matching prefixes for requests in the waiting queue,
            and handles in-batch prefix caching logic.
        """
        temporary_deprioritized: Set[int] = set()
        self.waiting_queue_radix_tree.reset()

        for r in waiting_queue:
            prefix_ids = r.origin_input_ids + r.output_ids
            extra_key = r.extra_key
            match_result = match_prefix_for_req(
                self.tree_cache, r, prefix_ids, include_req=True
            )

            # NOTE(sang): This logic is for in-batch prefix caching;
            # If there are more than 1 request that have small matching prefix from
            # existing cache, but all those requests share the same prefix, we prefer
            # to schedule only one of them so that we can increase the cache hit rate.
            # We prefer to set IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD > 0 because too small
            # threshold means we cannot use in-batch prefix caching for short prefixes.
            # It is kind of common when the engine is long running (e.g., imagine the prefix "the").
            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                match_result = self.waiting_queue_radix_tree.match_prefix(
                    MatchPrefixParams(
                        key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
                    )
                )
                if envs.SGLANG_RADIX_FORCE_MISS.get():
                    match_result = zero_match_result(
                        self.waiting_queue_radix_tree, match_result
                    )
                in_batch_matching_prefixes = match_result.device_indices
                if (
                    len(in_batch_matching_prefixes)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    temporary_deprioritized.add(r.rid)
                else:
                    # Insert with a dummy key
                    self.waiting_queue_radix_tree.insert(
                        InsertParams(
                            key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),
                            value=torch.empty(len(prefix_ids), dtype=torch.bool),
                        )
                    )
        return temporary_deprioritized

    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        waiting_queue.sort(
            key=lambda r: (
                -r.num_matched_prefix_tokens
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )

    @staticmethod
    def _sort_by_dfs_weight(
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sorts the waiting queue based on a depth-first search weighting."""
        last_node_to_reqs = defaultdict(list)
        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)

        node_to_weight = defaultdict(int)
        for node in last_node_to_reqs:
            node_to_weight[node] = len(last_node_to_reqs[node])
        SchedulePolicy._calc_weight(tree_cache.root_node, node_to_weight)

        waiting_queue.clear()
        SchedulePolicy._get_dfs_priority(
            tree_cache.root_node,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
        )

    @staticmethod
    def _sort_by_longest_output(
        waiting_queue: List[Req],
        enable_priority_scheduling: bool,
        priority_sign: int,
    ) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens). If using priority scheduling, sort by priority first."""
        if enable_priority_scheduling:
            waiting_queue.sort(
                key=lambda x: (
                    x.priority * priority_sign,
                    -x.sampling_params.max_new_tokens,
                )
            )
        else:
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)

    @staticmethod
    def _sort_randomly(waiting_queue: List[Req]) -> None:
        """Shuffles the waiting queue randomly."""
        random.shuffle(waiting_queue)

    @staticmethod
    def _sort_by_priority_and_fcfs(
        waiting_queue: List[Req],
        priority_sign: int,
        enable_fast_lane: bool = False,
        fast_lane_priority: int = 0,
        heavy_aging_ms: float = 0.0,
    ) -> None:
        """Sorts the waiting queue based on the request priority then received titmestamp.

        Fast lane (Variant C Stage 0) anti-starvation: when enabled with a
        positive aging window, a heavy request that has waited longer than
        ``heavy_aging_ms`` is promoted just BELOW the fast tier (effective
        priority ``fast_lane_priority - 1``) — ahead of un-aged heavy requests
        but still behind fast requests. This lets an aged heavy request win the
        next freed slot ahead of fresh heavy work, without ever outranking a
        fast request (which must stay first so its preemption path fires; a
        heavy req cannot preempt, so promoting it ABOVE fast would only wedge
        the admission loop and starve the fast lane). Ordering within a promoted
        set stays FCFS. When aging is off, behavior is byte-identical to before.
        """
        aging_active = (
            enable_fast_lane and heavy_aging_ms > 0 and priority_sign == -1
        )
        if not aging_active:
            waiting_queue.sort(
                key=lambda x: (
                    x.priority * priority_sign,
                    x.time_stats.wait_queue_entry_time,
                )
            )
            return

        now = time.time()
        promote_before = now - (heavy_aging_ms / 1000.0)

        def _effective_priority(x: Req) -> int:
            # Only heavy requests age; a heavy request whose queue-entry time is
            # older than the aging window jumps ahead of the fast tier.
            if not getattr(x, "is_fast_lane", False) and (
                x.time_stats.wait_queue_entry_time <= promote_before
            ):
                return fast_lane_priority - 1
            return x.priority

        # priority_sign == -1 here (higher value = higher priority).
        waiting_queue.sort(
            key=lambda x: (
                _effective_priority(x) * priority_sign,
                x.time_stats.wait_queue_entry_time,
            )
        )

    @staticmethod
    def _sort_by_routing_key(
        waiting_queue: List[Req], running_batch: ScheduleBatch
    ) -> None:
        """Sorts waiting queue by routing key frequency in running batch."""
        routing_key_counts = Counter(
            r.routing_key for r in running_batch.reqs if r.routing_key
        )

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_before = [r.routing_key for r in waiting_queue]
            logger.info(
                f"routing_key_counts={dict(routing_key_counts)}, "
                f"waiting_keys_before={waiting_keys_before}"
            )

        if not routing_key_counts:
            return

        def sort_key(req: Req):
            key = req.routing_key
            if key and key in routing_key_counts:
                count = routing_key_counts[key]
                return (0, -count, key)
            else:
                return (1, 0, key or "")

        waiting_queue.sort(key=sort_key)

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_after = [r.routing_key for r in waiting_queue]
            logger.info(f"waiting_keys_after={waiting_keys_after}")

    @staticmethod
    def _calc_weight(cur_node: TreeNode, node_to_weight: Dict[TreeNode, int]) -> None:
        for child in cur_node.children.values():
            SchedulePolicy._calc_weight(child, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    @staticmethod
    def _get_dfs_priority(
        cur_node: TreeNode,
        node_to_priority: Dict[TreeNode, int],
        last_node_to_reqs: Dict[TreeNode, List[Req]],
        q: List,
    ) -> None:
        children = [child for child in cur_node.children.values()]
        children.sort(key=lambda x: -node_to_priority[x])
        for child in children:
            SchedulePolicy._get_dfs_priority(
                child, node_to_priority, last_node_to_reqs, q
            )
        q.extend(last_node_to_reqs[cur_node])


class AddReqResult(Enum):
    CONTINUE = auto()  # Continue to add requests
    NO_TOKEN = auto()  # No token left
    OTHER = auto()  # Other reasons to stop adding requests


def truncation_align_admission_error(
    chunked_prefill_size: Optional[int],
    page_size: int,
    truncation_align_size: Optional[int],
    sources: Sequence[str] = (),
    dynamic_chunking: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Reject a chunk budget that can never satisfy the truncation alignment.

    ``PrefillAdder.add_one_req``'s chunked branch aligns the chunk it is about
    to take and REFUSES outright when the whole chunk budget is smaller than
    one alignment unit::

        trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
        if truncation_align_size is not None:
            if trunc_len < truncation_align_size:
                return AddReqResult.OTHER

    ``rem_chunk_tokens`` is bounded above by ``chunked_prefill_size``, so when
    the aligned chunk budget is below the alignment size that branch returns
    ``OTHER`` for **every** request longer than the budget, on every
    iteration, forever. The scheduler's admission loop ``break``s on any
    non-CONTINUE verdict, so one such request at the head of the FCFS waiting
    queue blocks the queue behind it, ``can_run_list`` stays empty, and no
    batch is ever built.

    The resulting instance is the worst failure shape this tree knows: it
    boots, prints "fired up and ready to roll", serves its warmup prefills
    (which are short enough to take the non-chunked branch) and then admits
    NOTHING. Measured (booked as C30): zero ``Decode batch`` lines across an
    entire boot, an 8-token ``/generate`` hung for 55 s, ``/health`` timing
    out while ``/get_model_info`` answered instantly, and the collective
    census FROZEN on both ranks at an identical count -- no crash, no
    collective hang, no rank divergence and no log line. It looks exactly
    like a deadlock and it is a refused predicate.

    ``truncation_align_size`` has TWO independent sources and this guard
    covers both, because either alone is sufficient to arm the trap:

    * ``--enable-deterministic-inference`` on the flashinfer or triton
      backend (align = ``SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE`` /
      ``SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE``, both default 4096);
    * ``--mamba-checkpoint-interval``, which sets the align size on its own
      when deterministic inference is OFF and is lcm-ed into it when on.

    THE CHECK IS AGAINST THE STATIC FLAG, DELIBERATELY. With
    ``--enable-dynamic-chunking`` the per-batch width comes from
    ``dynamic_chunked_prefill_size()``, which can deviate in BOTH directions.
    The static value is still the right thing to refuse on, because the
    predictor returns None until it has profiled, so the static size is what
    is in force for the first requests of every boot -- exactly the ones that
    would wedge. But the predictor's floor is ``base_chunk_size // 4``
    (``scheduler_pp_mixin.predict_next_chunk_size``), so a config that passes
    the static check can still dip below the alignment once the predictor
    engages. That case is WARNED about rather than refused: it is conditional
    on runtime behaviour, and refusing it would reject configurations that
    mostly work.

    Returns ``(error, warning)``. ``error`` non-None means refuse.
    """
    if truncation_align_size is None or int(truncation_align_size) <= 0:
        return None, None
    if chunked_prefill_size is None or int(chunked_prefill_size) <= 0:
        # Chunked prefill is off -> rem_chunk_tokens is None -> the aligned
        # branch is unreachable and nothing can be refused by it.
        return None, None
    chunked_prefill_size = int(chunked_prefill_size)
    page_size = max(1, int(page_size))
    truncation_align_size = int(truncation_align_size)
    budget = chunked_prefill_size // page_size * page_size
    why = f" ({', '.join(sources)})" if sources else ""
    if budget >= truncation_align_size:
        dyn_floor = budget // 4
        if dynamic_chunking and dyn_floor < truncation_align_size:
            return None, (
                f"--enable-dynamic-chunking can shrink the prefill chunk "
                f"width to a quarter of --chunked-prefill-size "
                f"({chunked_prefill_size} -> as low as {dyn_floor}), which is "
                f"below the truncation alignment of {truncation_align_size}"
                f"{why}. The static budget satisfies the alignment, so this "
                "boots and serves; but if the predictor ever settles below "
                f"{truncation_align_size} the scheduler will refuse every "
                "request longer than the chunk and stop admitting entirely. "
                f"Raise --chunked-prefill-size to at least "
                f"{truncation_align_size * 4}, or disable dynamic chunking, "
                "to remove the possibility."
            )
        return None, None
    return (
        f"--chunked-prefill-size={chunked_prefill_size} cannot satisfy a "
        f"prefill truncation alignment of {truncation_align_size}"
        f"{why}. The chunk budget aligns down to {budget} tokens "
        f"(page_size={page_size}), which is below one alignment unit, so the "
        "scheduler's chunked-prefill branch refuses EVERY request longer than "
        f"{budget} tokens and breaks the waiting-queue loop on it. The server "
        "would boot, report ready, serve its warmups and then admit nothing "
        "at all: no batch, no crash, no hang in any collective and no log "
        f"line. Raise --chunked-prefill-size to at least "
        f"{truncation_align_size}, or lower the alignment."
    ), None


class PrefillAdder:
    def __init__(
        self,
        page_size: int,
        tree_cache: BasePrefixCache,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        rem_chunk_tokens: Optional[int],
        num_mixed_decode_tokens: int = 0,
        priority_scheduling_preemption_threshold: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: Optional[int] = None,
        prefill_max_requests: Optional[int] = None,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor] = None,
        dllm_config: Optional[DllmConfig] = None,
        waiting_queue_len: int = 0,
        dcp_avail_deficit: int = 0,
        prefill_spill_regions: int = 0,
        prefill_spill_region_tokens: int = 0,
        prefill_spill_deep: bool = False,
        fundable_extend_floor: Optional[int] = None,
        commitment_ledger: Optional[ChunkedCommitmentLedger] = None,
        chunked_admission_enabled: bool = True,
        scheduled_extents: Optional[Dict[str, Tuple[int, int]]] = None,
        scheduled_fill_carry: Optional[Dict[str, Tuple[int, Tuple[int, ...]]]] = None,
        scheduled_last_chunk: Optional[Dict[str, bool]] = None,
    ):
        # #987: `rid -> (upstream_fill_len, upstream_fill_tail)` off the SAME
        # forwarded decision as `scheduled_extents` below, or None on every
        # rank that owns its own admission truth. The geometry and the fill it
        # is measured against travel together; see `_add_scheduled_req`, the
        # one place either is read.
        self.scheduled_fill_carry = scheduled_fill_carry
        # #996: `rid -> last_chunk`, the DECIDING rank's own verdict, off that
        # same decision. Absent rid = say nothing = fall back to the local
        # derivation, which is the pre-#996 behaviour exactly.
        self.scheduled_last_chunk = scheduled_last_chunk
        # #791 CORE: `rid -> (prefix_len, extend_len)` decided ONCE by the
        # first PP rank and forwarded on the admission decision, or None.
        #
        # None on every rank that owns its own admission truth -- PP0, and
        # every non-PP boot -- so the default path below is not merely
        # equivalent to the pre-#791 arithmetic, it is that arithmetic
        # unentered. A downstream PP rank gets a mapping, and for the rids in
        # it this adder stops DECIDING a chunk length and starts EXECUTING
        # one. See `pp_admission_congruence.forwarded_schedule` for the boot
        # instr20 specimen (one 845-token prompt, split 0+512 on PP0 and
        # 512+333 on PP1, in the same second, off the same decision).
        self.scheduled_extents = scheduled_extents
        # #701 defect (b). The ledger is owned by the SCHEDULER, not by this
        # adder: a PrefillAdder is rebuilt every pass, so anything held here
        # would forget a resident chunked request's outstanding prefill exactly
        # when the next pass needs to see it. Passed in, never constructed here.
        self.commitment_ledger = commitment_ledger
        # DEFAULT ON: a wedge is strictly worse than a refusal. Off restores the
        # pre-#701 arithmetic byte-for-byte, for A/B on a reviewed window.
        self.chunked_admission_enabled = bool(chunked_admission_enabled)
        self.page_size = page_size
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        # Prefill-Spill (kv-session-offload, PS1-V1a). Number of born-spilled
        # admissions still allowed THIS iteration (= free host regions; 0 when
        # the feature is off -> the relaxation below is inert, byte-identical).
        # Decremented as born-spilled prompts are admitted so we never admit
        # more born-spilled prompts than there are free host regions to hold
        # them. Replicated across DCP ranks (see prefill_spill_free_regions) and
        # decremented identically per rank -> rank-uniform, no collective.
        self.prefill_spill_regions = int(prefill_spill_regions)
        # PS2 (deep prefill-spill). `prefill_spill_deep` is the master gate
        # (--kv-session-offload-prefill); `region_tokens` bounds a session's
        # host tail. `_deep_taken` enforces the ALL-OR-NOTHING batch
        # separation: a born-spilled-deep prompt is admitted only into an EMPTY
        # can_run_list and closes the batch behind it, so its host-sentinel
        # out_cache_loc is never mixed with real device slots. All three are
        # replicated -> the partition is identical on every DCP rank.
        self.prefill_spill_deep = bool(prefill_spill_deep)
        self.prefill_spill_region_tokens = int(prefill_spill_region_tokens)
        self.prefill_spill_deep_taken = False
        # RANK-UNIFORM admission under uneven DCP (kv-session-offload): a
        # non-negative correction (local_avail - min_reduce(local_avail))
        # subtracted from every `available_size()`-based admission budget so all
        # DCP ranks admit against the binding (least-slack) rank's pool and take
        # the identical decision (no divergent prefill batch -> no forward
        # desync). 0 on the default path -> byte-identical.
        self.dcp_avail_deficit = dcp_avail_deficit
        # #681: what the GROUP can actually fund this iteration, or None when
        # nothing was published (single rank, or pools that agree -> the
        # default path, where this is inert by construction).
        #
        # WHY THE NEW-REQUEST PATH NEEDS ITS OWN CEILING. #679 taught the
        # CHUNKED gate to park on `fundable_extend_tokens`, which reads the
        # group MIN via `uniform_avail_for_evict`. `rem_total_tokens` below
        # never learned that lesson: it reads THIS RANK's availability, so a
        # rank roomier than the binding one admits work the group cannot pay
        # and the batch dies in `alloc_for_extend` (2026-08-16 01:46:10, all
        # three ranks). It is also a rank-local BRANCH upstream of a
        # collective, which splits the group into different batch shapes --
        # a hang rather than a stall, the family #583/#603/#616g/#639 paid for.
        #
        # A CEILING, NOT A SUBSTITUTE. `rem_total_tokens` also subtracts
        # reservations this floor knows nothing about (the running batch's
        # hold, mamba gap, page overhead). Replacing the local term with the
        # floor would talk a rank that is ITSELF short back UP. The budget is
        # therefore the MIN of the two, and both spend down the same
        # `rem_total_token_offset` -- a floor consulted per request without
        # that shared accounting is not a bound at all, since every request in
        # the round would compare itself against the same untouched pool.
        self.fundable_extend_floor = (
            None if fundable_extend_floor is None else int(fundable_extend_floor)
        )
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        self.rem_input_tokens = rem_input_tokens - num_mixed_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        self.dllm_config = dllm_config

        if self.dllm_config is not None:
            self._init_dllm_meta(dllm_config)

        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= num_mixed_decode_tokens
        self.rem_total_token_offset = num_mixed_decode_tokens
        self.cur_rem_token_offset = num_mixed_decode_tokens

        self.req_states = None
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        #: #959 IS A CONTINUATION ALREADY RESIDENT THIS PASS?
        #:
        #: `scheduler.chunked_req` is the authority and it is not this object's
        #: to read, so the scheduler STAMPS it here right after it has settled
        #: (scheduler.py, immediately below `add_chunked_req`). Default False
        #: keeps every caller that does not stamp it exactly as it was.
        #:
        #: WHY IT HAS TO BE STAMPED RATHER THAN INFERRED: the continuation can
        #: stay resident on a pass where `add_chunked_req` is never called at
        #: all -- the #906 seam refusal keeps the request as
        #: `scheduler.chunked_req` and skips the adder entirely. Inferring
        #: residency from this object's own calls would read that pass as
        #: "nothing resident", which is precisely the pass that then mints a
        #: second one.
        self.chunked_req_outstanding = False
        self.log_hit_tokens = 0
        self.reprocessed_log_hit_tokens = 0
        # TODO(lsyin): report the real input tokens excluding page alignment
        self.log_input_tokens = 0
        self.reprocessed_log_input_tokens = 0

        if running_batch is not None:
            # Estimate the offset in the remaining token space
            self.rem_total_token_offset += sum(
                [
                    self._get_running_request_total_token_offset(r)
                    for r in running_batch.reqs
                ]
            )

        # DeepSeek V4 HiSparse wraps an SWATokenToKVPoolAllocator internally and
        # exposes the full SWA allocator interface.
        self.is_hybrid_swa = isinstance(
            self.token_to_kv_pool_allocator,
            (SWATokenToKVPoolAllocator, DeepSeekV4HiSparseTokenToKVPoolAllocator),
        )
        self.is_all_swa = isinstance(
            self.token_to_kv_pool_allocator, PureSWATokenToKVPoolAllocator
        )
        self.is_hybrid_ssm_cache = self.tree_cache.supports_mamba()

        self.rem_swa_token_offset = 0

        # Unified-pool joint budget: a new mamba state consumes shared-gap bytes
        # that `rem_total_tokens` (full KV) otherwise counts as free, so reserve
        # the gap per new mamba slot or admission over-commits. Gate on the
        # ALLOCATOR being the unified Mamba composite, NOT on `is_hybrid_ssm_cache`
        # (False for `ChunkCache`, which would skip the reservation on the
        # chunk-cache path): the gap coupling is a property of the byte buffer.
        self._mamba_slot_cost = 0
        if isinstance(
            self.token_to_kv_pool_allocator, UnifiedMambaTokenToKVPoolAllocator
        ):
            self._mamba_slot_cost = (
                self.token_to_kv_pool_allocator.mamba_slot_full_token_cost()
            )

        # `mamba_gap_reserve` is charged to `rem_total_tokens`, which INCLUDES
        # `full_evictable_size()` — but `alloc_req_slots` can only recover
        # MAMBA-recoverable bytes for a mamba slot (shared gap + peer holes +
        # mamba-evictable radix), NOT full-evictable. Gate new mamba slots on
        # that mamba-recoverable budget separately or an over-admit hits the
        # fail-loud `RuntimeError`. `None` outside the unified Mamba pool.
        self.rem_mamba_slots = None
        # Slots one newly admitted request will hold at once (active state +
        # ping-pong track buffers + donation slot + pinned checkpoint). The
        # old code charged a flat 1, which is what the FIXME in
        # `_mamba_gap_budget_for_req` warned about: admission believed a
        # request cost one slot while it actually holds `floor_per_req`, so
        # the pool was over-committed and the shortfall surfaced as a bare
        # assert deep inside `HybridReqToTokenPool.alloc` (#581).
        self._mamba_slots_per_req = 1
        if self._mamba_slot_cost:
            self.rem_mamba_slots = (
                self.token_to_kv_pool_allocator.mamba_allocator.schedulable_available_size()
            )
            if self.is_hybrid_ssm_cache:
                self.rem_mamba_slots += self.tree_cache.mamba_evictable_size()
        elif self.is_hybrid_ssm_cache:
            # Non-unified hybrid pool (`HybridReqToTokenPool`): there was NO
            # mamba admission gate here at all -- `rem_mamba_slots` stayed
            # None, so `PrefillAdder` admitted requests the state pool could
            # not serve and the over-admission died on an assert instead of
            # deferring. Gate it on the same budget the unified path uses.
            from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

            # Gate on the concrete pool type, not on duck-typing: the budget
            # is only meaningful for a real state pool, and a stubbed cache
            # (unit tests) must leave the baseline path untouched.
            pool = getattr(self.tree_cache, "req_to_token_pool", None)
            if isinstance(pool, HybridReqToTokenPool):
                from sglang.srt.mem_cache.mamba_pool_floor import (
                    mamba_slots_per_running_req,
                )

                self._mamba_slots_per_req = mamba_slots_per_running_req(
                    get_server_args()
                )
                self.rem_mamba_slots = (
                    pool.mamba_allocator.available_size()
                    + self.tree_cache.mamba_evictable_size()
                )

        self.priority_scheduling_preemption_threshold = (
            priority_scheduling_preemption_threshold
        )
        self.dsa_prefill_cp_in_seq_split = is_dsa_prefill_cp_in_seq_split()
        self.max_running_requests = max_running_requests
        self.prefill_context_parallel_enabled = is_prefill_context_parallel_enabled()
        self.prefill_max_requests = prefill_max_requests
        self.prefill_delayer_single_pass = prefill_delayer_single_pass
        self.max_prefill_bs = max_prefill_bs
        # Snapshot of scheduler waiting_queue length at the start of this
        # prefill pass. Used by PrefillDelayer's queue-based trigger.
        self.waiting_queue_len = waiting_queue_len

    def _init_dllm_meta(self, dllm_config: DllmConfig):
        self.dllm_block_size = dllm_config.block_size
        max_running_reqs = dllm_config.max_running_requests

        self.rem_dllm_tokens = max_running_reqs * self.dllm_block_size

    def _get_running_request_total_token_offset(self, req: Req) -> int:
        return (
            min(
                (req.sampling_params.max_new_tokens - len(req.output_ids)),
                CLIP_MAX_NEW_TOKENS,
            )
            * self.new_token_ratio
        )

    @property
    def rem_total_tokens(self):
        if self.is_all_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.swa_available_size()
                + self.tree_cache.swa_evictable_size()
            )
        elif self.is_hybrid_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size()
            )
        elif self.is_hybrid_ssm_cache:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.full_evictable_size()
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )
        # dcp_avail_deficit (0 off the uneven-DCP kv-session-offload path) pins
        # the budget to the binding rank's available_size -> rank-uniform.
        local_budget = (
            available_and_evictable
            - self.rem_total_token_offset
            - self.dcp_avail_deficit
        )
        # #681: cap by what the GROUP can fund. The floor is already the
        # group MIN (`uniform_avail_for_evict`), so `dcp_avail_deficit` -- the
        # other mechanism for pinning a local number to the binding rank -- is
        # deliberately NOT subtracted a second time here. `rem_total_token_offset`
        # IS, so the floor is spent down by this round's admissions exactly as
        # the local term is. None -> no reduce published -> untouched.
        if self.fundable_extend_floor is None:
            budget = local_budget
        else:
            budget = min(
                local_budget, self.fundable_extend_floor - self.rem_total_token_offset
            )
        # #701 defect (b): CROSS-PASS RESERVATION. PrefillAdder is rebuilt every
        # pass and reserves only remaining DECODE, so a resident chunked
        # request's remaining PREFILL is invisible here and later admissions
        # spend its committed future -- the two-actor deadlock. The ledger is
        # scheduler-owned and survives the rebuild, so subtracting it is what
        # makes the commitment visible across passes. Single chokepoint: every
        # sibling admission site reads this property.
        if not self.chunked_admission_enabled:
            return budget
        return effective_rem_total_tokens(budget, self.commitment_ledger)

    @property
    def rem_swa_tokens(self):
        return (
            self.token_to_kv_pool_allocator.swa_available_size()
            + self.tree_cache.swa_evictable_size()
            - self.rem_swa_token_offset
        )

    @property
    def cur_rem_tokens(self):
        if self.is_all_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.swa_available_size()
                + self.tree_cache.swa_evictable_size()
            )
        elif self.is_hybrid_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size()
            )
        elif self.is_hybrid_ssm_cache:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.full_evictable_size()
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )

        # dcp_avail_deficit (0 off the uneven-DCP kv-session-offload path) pins
        # the budget to the binding rank's available_size -> rank-uniform.
        return (
            available_and_evictable
            - self.cur_rem_token_offset
            - self.dcp_avail_deficit
        )

    def _swa_budget_for_req(
        self, extend_input_len: int, swa_host_hit_length: int = 0
    ) -> int:
        """SWA pool budget per request. Only valid when is_hybrid_swa is True.

        With chunked prefill + overlap scheduler, the peak SWA occupancy is:
          chunk N (running, not yet in tree) + sliding window (locked in tree)
          + chunk N+1 (new allocation)
        Since chunk N and locked tokens are already excluded from
        swa_available + swa_evictable, the budget only needs to cover the
        chunk N+1 allocation. We floor at sliding_window_size to reserve
        room for the decode phase.
        """
        if self.rem_chunk_tokens is not None:
            alloc = min(extend_input_len, self.rem_chunk_tokens)
        else:
            alloc = extend_input_len
        budget = max(alloc, self.tree_cache.sliding_window_size) + self.page_size
        if swa_host_hit_length > 0:
            budget += self.ceil_paged_tokens(swa_host_hit_length)
        return budget

    def _mamba_gap_budget_for_req(self, req: Req) -> int:
        """Shared-gap reservation (full-token-equivalents) for a request's new
        mamba state. Charged only on the SHARED Mamba pool (`_mamba_slot_cost > 0`)
        and only when the req has no state yet (`mamba_pool_idx is None`, mirroring
        `HybridReqToTokenPool.alloc`); 0 keeps baseline / SWA / non-Mamba unchanged.

        Conservative by design (`_mamba_slot_cost` rounds UP). Does NOT reserve
        radix COW headroom or locked-but-evictable bytes — that residual is
        backstopped by the fail-loud RuntimeError in `alloc_req_slots`. FIXME: if
        over-admission crashes under pressure, make this more conservative (e.g.
        multiply by `MAMBA_STATE_PER_REQ_PREFIX_CACHE`)."""
        if self._mamba_slot_cost and req.mamba_pool_idx is None:
            return self._mamba_slot_cost
        return 0

    def _mamba_slots_for_req(self, req: Req) -> int:
        """Mamba state slots a request will hold once admitted, or 0 if it
        already owns its state (chunked continuation / radix CoW resume).

        `_mamba_slots_per_req` is 1 on the unified pool, where the shared-gap
        reservation carries the rest of the cost, and the full per-request
        demand floor on the non-unified `HybridReqToTokenPool` -- the case
        that previously had no admission gate at all (#581).
        """
        if self.rem_mamba_slots is None or req.mamba_pool_idx is not None:
            return 0
        return self._mamba_slots_per_req

    def ceil_paged_tokens(self, tokens: int) -> int:
        return -(-tokens // self.page_size) * self.page_size

    def budget_state(self):
        # PS2 batch separation: once a born-spilled-deep prompt is in the list
        # the extend batch is CLOSED -- its out_cache_loc is a row of host
        # sentinels and must not be concatenated with real device slots.
        if self.prefill_spill_deep_taken:
            return AddReqResult.OTHER
        no_token = self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0
        if not no_token and self.is_hybrid_swa:
            no_token = self.rem_swa_tokens <= 0
        # Gate new mamba slots separately: rem_total_tokens' full_evictable can't
        # cover a mamba slot, which needs mamba-recoverable bytes (see __init__).
        if not no_token and self.rem_mamba_slots is not None:
            no_token = self.rem_mamba_slots <= 0
        if no_token:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0:
            return AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER
        else:
            if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _admit_born_spilled(self, req, born_input_tokens: int) -> bool:
        """Prefill-Spill (kv-session-offload, PS1-V1a): decide whether a prompt
        that FAILED the lifetime device-budget gate (its input + max_new won't
        fit VRAM) can instead be ADMITTED born-spilled -- prefilled on device
        (its input transiently fits) and then rode into the host pool by the
        EXISTING decode-OOM spill (try_spill), rather than wedged / retracted.

        Relaxes ONLY the lifetime: the current-step guard stays strict -- the
        prefill INPUT must still transiently fit the (rank-uniform, deficit-
        pinned) device budget, so the prefill itself cannot OOM. The exclusive
        suffix then spills at decode time. (Never-materialize-on-device born
        writes are PS2, only for the deep case where even one chunk won't fit.)

        RANK-UNIFORM (R2-safe): every input is replicated (born_input_tokens
        from request metadata) or already min-reduced (rem_total_tokens via
        dcp_avail_deficit; prefill_spill_regions a replicated free-region
        count), so the verdict is identical on every DCP rank WITHOUT a
        collective. Over-admission beyond the free host regions is crash-safe:
        the surplus born-spilled prompt's later try_spill finds no free region,
        returns False, and the request takes the pre-existing graceful retract
        path -- no crash. Off (prefill_spill_regions == 0) -> byte-identical."""
        if self.prefill_spill_regions <= 0:
            return False
        # Current-step guard (STRICT): the prefill input must transiently fit
        # the device. If not, this is the DEEP case (PS2) -> reject, today's
        # wedge/wait behaviour, no born-spill.
        if born_input_tokens >= self.rem_total_tokens:
            return False
        req.born_spilled = True
        logger.info(
            "kv-session-offload prefill-spill: admit rid=%s BORN-SPILLED "
            "(input=%d fits device budget=%d; full lifetime would wedge) -> "
            "rides the decode-OOM spill into host.",
            getattr(req, "rid", "?"),
            int(born_input_tokens),
            int(self.rem_total_tokens),
        )
        return True

    def _admit_born_spilled_deep(self, req, born_input_tokens: int) -> bool:
        """PS2 (deep prefill-spill): admit a prompt whose INPUT does not even
        transiently fit the device budget, by never giving it device KV slots.

        The strict COMPLEMENT of ``_admit_born_spilled``'s window: PS1 is tried
        first and PS2 only sees what PS1 rejected, so the validated PS1 path
        keeps its exact behaviour and PS2 adds a new admission, never a
        different one.

        Guards (all hard, all replicated -> RANK-UNIFORM without a collective;
        see ``prefill_spill_deep_ok`` for the verdict itself):
          * feature on and a free host region available;
          * ONE CHUNK: the prompt must fit ``rem_chunk_tokens`` whole. Without
            PS3 (host-prefix extend read) a second chunk would attend the first
            chunk's SENTINEL rows -- garbage. A single non-chunked extend
            attends only its own ragged keys plus the device-resident radix
            prefix, so it needs no host read;
          * the tail fits one region;
          * BATCH SEPARATION: only into an empty ``can_run_list``, and the
            batch is closed behind it (``prefill_spill_deep_taken``), because
            ``out_cache_loc`` cannot hold device slots and host sentinels at
            once.
        Off (``prefill_spill_deep`` False) -> byte-identical."""
        if not self.prefill_spill_deep:
            return False
        if self.can_run_list or self.prefill_spill_deep_taken:
            return False
        input_tokens = self.ceil_paged_tokens(
            len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
        )
        if not prefill_spill_deep_ok(
            free_regions=self.prefill_spill_regions,
            born_input_tokens=born_input_tokens,
            rem_total_tokens=self.rem_total_tokens,
            input_tokens=input_tokens,
            rem_chunk_tokens=self.rem_chunk_tokens,
            region_tokens=self.prefill_spill_region_tokens,
        ):
            return False
        req.born_spilled_deep = True
        self.prefill_spill_deep_taken = True
        self.prefill_spill_regions -= 1
        logger.info(
            "kv-session-offload prefill-spill (PS2): admit rid=%s BORN-SPILLED "
            "DEEP (input=%d does NOT fit device budget=%d) -- prefilled "
            "straight into a host region, no device KV slots.",
            getattr(req, "rid", "?"),
            int(input_tokens),
            int(self.rem_total_tokens),
        )
        return True

    def _update_prefill_budget(
        self,
        prefix_len: int,
        extend_input_len: int,
        max_new_tokens: int,
        retracted_stain: bool,
        mamba_gap_reserve: int = 0,
        mamba_slot_charge: int = 0,
    ):
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        extend_input_len = self.ceil_paged_tokens(extend_input_len)

        # alloc_extend reserves an extra page_size per request to make sure the budget doesn't over-commit
        page_overhead = self.page_size
        # `mamba_gap_reserve` (shared Mamba pool only; 0 otherwise) charges the new
        # mamba state's shared-gap cost to BOTH full budgets: the slot is allocated
        # immediately (counts against `cur_rem`) and held for the request lifetime
        # (counts against `rem_total`). See `_mamba_gap_budget_for_req`.
        self.rem_total_token_offset += (
            extend_input_len + max_new_tokens + page_overhead + mamba_gap_reserve
        )
        self.cur_rem_token_offset += (
            extend_input_len + page_overhead + mamba_gap_reserve
        )
        # The new mamba state also consumes mamba-recoverable slots (gated
        # separately so full_evictable can't cover them — see __init__).
        # `_mamba_slots_per_req` is 1 on the unified path (where the gap
        # reservation already carries the rest) and the full per-request floor
        # on the non-unified hybrid pool.
        if self.rem_mamba_slots is not None and mamba_slot_charge:
            self.rem_mamba_slots -= mamba_slot_charge
        self.rem_input_tokens -= extend_input_len

        if self.is_hybrid_swa:
            self.rem_swa_token_offset += self._swa_budget_for_req(extend_input_len)

        if self.dllm_config is not None:
            self.rem_dllm_tokens -= extend_input_len
        elif self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= extend_input_len

        # reprocessed_log_* is a subset of log_*; metrics_reporter subtracts it
        # when computing the first-attempt prefix cache hit rate.
        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len
        if retracted_stain:
            self.reprocessed_log_hit_tokens += prefix_len
            self.reprocessed_log_input_tokens += extend_input_len

    def _get_dllm_remain_tokens(self) -> int:
        _rem_tokens = min(
            self.rem_dllm_tokens,
            self.dllm_block_size,
            int(self.rem_total_tokens),
        )
        if _rem_tokens <= 0:
            _rem_tokens = self.rem_dllm_tokens

        return _rem_tokens

    def _add_dllm_req(self, req: Req, prefix_len: int):
        # FIXME: consider the case when rem_dllm_tokens < dllm_block_size,
        # the diffusion unmask process may have some problems
        # Make sure at least one page is available
        trunc_len = (
            min(self.rem_dllm_tokens, self.dllm_block_size)
            // self.page_size
            * self.page_size
        )

        req.set_extend_range(prefix_len, prefix_len + trunc_len)

        self.can_run_list.append(req)

        self._update_prefill_budget(
            prefix_len,
            trunc_len,
            0,
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
            mamba_slot_charge=self._mamba_slots_for_req(req),
        )

    def _req_inc_lock_ref(self, req: Req):
        result = self.tree_cache.inc_lock_ref(req.last_node)
        if self.is_hybrid_swa:
            req.swa_uuid_for_lock = result.swa_uuid_for_lock

    def add_dllm_staging_req(self, req: Req):
        assert self.dllm_config is not None
        _rem_tokens = self._get_dllm_remain_tokens()

        if _rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        # Truncate input length to available tokens and update request metadata
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)
        self.can_run_list.append(req)

        # Update budget: reserve max_new_tokens only if not truncated
        max_new_tokens = (
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            if not truncated
            else 0
        )
        self._update_prefill_budget(
            0,
            req.extend_range.length,
            max_new_tokens,
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
            mamba_slot_charge=self._mamba_slots_for_req(req),
        )

        # Return based on remaining token availability
        return (
            AddReqResult.NO_TOKEN
            if self._get_dllm_remain_tokens() <= 0
            else AddReqResult.CONTINUE
        )

    def scheduled_extent_for(self, req: Req) -> Optional[Tuple[int, int]]:
        """The forwarded `(prefix_len, extend_len)` for `req`, or None.

        None is the whole default path: no mapping (PP0, every non-PP boot),
        or a rid the schedule does not name. A rid a forwarded schedule does
        not name must never reach the local chunking arithmetic either -- the
        admission loop refuses to admit it at all (scheduler.py), which is a
        membership decision and not this method's subject.
        """
        if not self.scheduled_extents:
            return None
        return self.scheduled_extents.get(req.rid)

    def _mint_chunked(self, req: Req, site: str) -> None:
        """#996 RATCHET: announce `req` as THIS pass's new chunked request.

        `scheduler.chunked_req` is a single field, and the scheduler asserts
        `self.chunked_req is None` before adopting `new_chunked_req`
        (scheduler.py). That assert is about the RESIDENT continuation from an
        earlier pass -- which is also all `chunked_req_outstanding` knows, since
        it is set exactly once per pass (scheduler.py, after `add_chunked_req`)
        and never at a mint. So neither watcher can see the OTHER way to end up
        with two: two mints inside ONE pass.

        Nothing has been observed doing that. This exists because the three
        mint sites below wrote a bare assignment, so a second one would have
        SILENTLY overwritten the first: the overwritten request stays in
        `can_run_list` with a partial extend range and is tracked by nobody,
        gets treated as a finished prefill, and is re-prefilled next pass --
        the double prefill the standing law forbids, with no assert and no log
        line anywhere on the path. The whole family (#951, #959, #995, #996) is
        one invariant held "by ARITHMETIC, not by a check"; this is the check,
        at the mint, naming both rids and the site.

        It is deliberately an assert and not a refusal: a refusal here would be
        the #858 wedge shape, and there is no evidence this is reachable. If it
        ever fires, it fires with everything needed to root it in one line
        instead of 300 lines later at a watcher that is asking about something
        else.
        """
        assert self.new_chunked_req is None, (
            f"#996 SECOND CHUNKED MINT IN ONE PASS at {site}: this adder "
            f"already minted rid={getattr(self.new_chunked_req, 'rid', '?')} "
            f"and is now being asked to mint rid={getattr(req, 'rid', '?')}. "
            f"`scheduler.chunked_req` is a single field, so the first would be "
            f"silently dropped, left in can_run_list with a partial extend "
            f"range, and re-prefilled next pass."
        )
        self.new_chunked_req = req

    def _told_last_chunk(self, req: Req) -> Optional[bool]:
        """#996: the deciding rank's last-chunk verdict for `req`, or None.

        ONE DEFINITION, TWO READERS. `_add_scheduled_req` executes this verdict
        and `add_chunked_req` asks the same question again immediately
        afterwards ("is this still the chunked request"). Before #996 both
        derived it locally and so could not disagree; now that one of them
        executes a CARRIED verdict, the other must read the same source or the
        two halves of one fact drift -- which is the defect class this ticket
        exists to close, one level up.

        `None` = the decision said nothing for this rid (legacy sender,
        unreadable fill, or no forwarded schedule at all), and both readers
        fall back to their pre-#996 local derivation.
        """
        if not self.scheduled_last_chunk:
            return None
        return self.scheduled_last_chunk.get(req.rid)

    def _add_scheduled_req(
        self,
        req: Req,
        extent: Tuple[int, int],
        *,
        carried_chunk: bool = False,
    ):
        """EXECUTE a forwarded pass geometry. Derive nothing.

        This is the whole of #791's core on the adder side: the two numbers
        that decide the cross-stage tensor's row count -- where the reused
        prefix ends and how many tokens are freshly computed -- are READ off
        the upstream's decision rather than recomputed from this rank's
        `rem_chunk_tokens`, `rem_total_tokens`, radix match or host-hit
        length. Every one of those is rank-local and every one of them moved
        under this rank's feet on boot instr20 between the pass being decided
        and the pass being built.

        NO LOCAL BUDGET GATE, DELIBERATELY. `add_one_req`'s `NO_TOKEN` /
        `OTHER` returns all mean "leave this request for a later pass", which
        a rank executing a forwarded schedule is not entitled to decide: the
        upstream has already built and launched a batch containing this
        request. The budget is still CHARGED below, so the rest of the pass
        accounts for what this request costs -- what is gone is the local
        veto, not the bookkeeping. If the geometry is genuinely impossible the
        refusal is loud and the whole pass is voided (`PPScheduleRefused`),
        which is the one direction in which uniform membership can still be
        restored.

        NO HOST LOAD-BACK EITHER, and this is the instr20 line specifically.
        `add_one_req` grows `req.prefix_indices` by `init_load_back`'s freshly
        revived indices, which on instr20 put 512 host-resident prefix tokens
        back onto a `prefix_indices` the admission loop had just clamped to
        the schedule's 0 -- turning a 512-token chunk into a 333-token
        remainder on PP1 and PP2 while PP0, whose `needs_host_load_back()` was
        false, kept the 512. A load-back is a rank-local improvement to a
        quantity this rank no longer owns, so on this path it does not run.
        """
        # Imported here rather than at module scope: this file's import block
        # is already an E402 region, and a local import keeps the new
        # dependency out of it without adding a finding -- the same call this
        # repo's `prefix_lens_check` import makes in schedule_batch.py.
        from sglang.srt.managers.pp_admission_congruence import (
            PPScheduleRefused,
            adopt_carried_fill,
            schedule_refusal_reason,
        )

        prefix_len, extend_len = int(extent[0]), int(extent[1])
        local_prefix_len = len(req.prefix_indices)
        # #987 THE ADOPT, IMMEDIATELY BEFORE THE COMPARISON IT SERVES. The
        # third clause of `schedule_refusal_reason` measures the forwarded
        # geometry against `len(full_untruncated_fill_ids)`, and R9's census
        # (boots 6-7) found 506 of 513 void-causing refusals to be that
        # comparison failing by ONE TOKEN: rank 0 holds a sampled token that
        # never crossed the `tp_to_pp` seam, so it reads 8447 where its
        # followers read 8446.
        #
        # The clause is not weakened -- it is asked the question after the
        # disagreement has been ended rather than before. `adopt_carried_fill`
        # materialises the upstream's trailing OUTPUT tokens into this rank's
        # fill (a shadow pair honoured by `Req._refresh_fill_ids`; NOT
        # `output_ids`, NOT `prefix_indices`, so nothing is recomputed and
        # nothing is emitted twice), or declines loudly and leaves the fill
        # exactly as it found it, in which case the clause below refuses this
        # pass on its own unchanged terms.
        #
        # ONE JUNCTION. This is the only site that reads the carried fill, and
        # it is the same statement that reads the carried geometry, so a rank
        # can never adopt one half of a decision and not the other.
        if self.scheduled_fill_carry:
            adopt_carried_fill(req, self.scheduled_fill_carry.get(req.rid))
        local_fill_len = len(req.full_untruncated_fill_ids)
        reason = schedule_refusal_reason(
            rid=req.rid,
            scheduled_prefix_len=prefix_len,
            scheduled_extend_len=extend_len,
            local_prefix_len=local_prefix_len,
            local_fill_len=local_fill_len,
        )
        if reason is not None:
            raise PPScheduleRefused(reason)

        req.set_extend_range(prefix_len, prefix_len + extend_len)
        self.can_run_list.append(req)
        # LAST CHUNK OR NOT IS THE SCHEDULE'S TO SAY -- AND UNTIL #996 IT DID
        # NOT SAY IT. The old line here read
        #
        #     last_chunk = prefix_len + extend_len >= local_fill_len
        #
        # under a comment claiming that was "arithmetic on forwarded integers
        # rather than re-taken against a local budget". Two of the three
        # integers are forwarded. `local_fill_len` is not: it is rebuilt from
        # THIS rank's `origin_input_ids + output_ids (+ carried tail)` on every
        # pass (`Req._refresh_fill_ids`, schedule_batch.py:1326, unconditional
        # and ahead of the `tree_cache` gate at :1355). So the verdict was a
        # rank-local quantity wearing a forwarded one's clothes, and the fill
        # is precisely the quantity this seam is known to disagree about.
        #
        # `schedule_refusal_reason` above does not cover it: its third clause
        # refuses `prefix + extend > local_fill_len`, the decision wanting more
        # than this rank holds. The opposite skew -- this rank holding MORE
        # than the rank that decided -- passes every clause and silently turns
        # a final chunk into a new continuation. `adopt_carried_fill` cannot
        # close it either; it only ever APPENDS, so it lifts a short follower
        # up to the decider and never brings a long one down.
        #
        # MEASURED, boot 16 (996fbf4aca, 2026-08-28 22:21:48, 68 s after first
        # load): PP1 and PP2 both logged `#987 FILL-ADOPT rid=da614e20...
        # local=8446 -> upstream=8447`, and PP1 died at
        # scheduler_pp_mixin.py:2147 with `#631 PROXY LEFTOVER REFUSED ...
        # mb_id=1 seq=17 rows=4096 epoch=2 arrived while this rank is on
        # mb_id=2` -- same-epoch, a proxy for a pass this rank had already
        # left. The continuation nobody decided on is what put it on the wire.
        #
        # NOW CARRIED, NOT RE-DERIVED. The fallback stays exactly the old
        # expression for an absent rid (legacy sender, unreadable fill), so a
        # mixed-version group and every non-PP boot behave as before.
        told_last_chunk = self._told_last_chunk(req)
        local_last_chunk = prefix_len + extend_len >= local_fill_len
        last_chunk = local_last_chunk if told_last_chunk is None else told_last_chunk
        if told_last_chunk is not None and told_last_chunk != local_last_chunk:
            # THE EXECUTION PROOF, and the one line that makes the next boot
            # able to read this seam directly. It fires only when the carry
            # actually CHANGED the verdict -- i.e. exactly on the passes that
            # used to mint a continuation nobody decided on -- so a quiet log
            # means the skew did not occur, not that the fix did not run.
            logger.info(
                "[#996] LAST-CHUNK CARRIED OVER LOCAL rid=%s prefix=%d extend=%d "
                "local_fill=%d told_last_chunk=%s local_last_chunk=%s: the "
                "deciding rank's verdict is executed. Deriving it here would "
                "have %s -- the boot 16 shape (local=8446 upstream=8447).",
                req.rid,
                prefix_len,
                extend_len,
                local_fill_len,
                told_last_chunk,
                local_last_chunk,
                (
                    "minted a continuation the decision never named"
                    if local_last_chunk is False
                    else "dropped a continuation the decision did name"
                ),
            )
        # `carried_chunk`: this request is ALREADY `scheduler.chunked_req`, so
        # announcing it as a NEW one would trip the `assert self.chunked_req
        # is None` the scheduler takes before adopting `new_chunked_req`, and
        # its prefix is carried allocation rather than a fresh cache hit --
        # the same two distinctions the local `add_chunked_req` makes.
        if not last_chunk and not carried_chunk:
            # #995 THE THIRD MINT SITE, AND THE ONE #959's SWEEP MISSED.
            #
            # #959 gave `add_one_req` and `add_one_req_ignore_eos` an explicit
            # `chunked_req_outstanding` check because the invariant behind
            # `scheduler.py`'s `assert self.chunked_req is None` is held "by
            # ARITHMETIC, not by a check". It skipped THIS site, reasoning (see
            # the note at the sibling) that it "already has its own
            # (`carried_chunk`)". That is the guard-comment-names-the-hazard
            # trap: `carried_chunk` answers "is THIS request the resident
            # continuation", which is a different question from "is there a
            # resident continuation at all". It covers the request being
            # re-announced; it does not cover a DIFFERENT named request
            # becoming a second one while the first is resident.
            #
            # MEASURED, boot 14 (cf16281b3f, 2026-08-28 22:01:29, PP1, 39 s):
            # the resident continuation survived `add_chunked_req`, so
            # `chunked_req_outstanding` was True and both sibling sites
            # correctly refused -- and this one minted anyway, on another rid
            # the same forwarded schedule named. `assert self.chunked_req is
            # None` then killed rank 1. Boot 13 never reached this line: the
            # #791 geometry refusal killed every pass before a batch was
            # built. #994 removed that refusal and made this site reachable in
            # the ordinary flow -- so #994 exposed this, it did not create it.
            # Third recorded fundstelle of the family, after :9286 (#951) and
            # :9367 (#959).
            #
            # DO NOT ANNOUNCE -- AND DO NOT REFUSE. The first version of this
            # guard raised `PPScheduleRefused` here, and boot 15 (473f3ad7b0,
            # 2026-08-28 22:12-22:13) measured that as a LIVELOCK: 175
            # refusals on ONE rid (ddb6f38b…) in ~40 s, 4 batches, rank 2 idle
            # at 0% while ranks 0-1 burned on corridor-reclaim. A voided pass
            # computes nothing, so the resident continuation never advanced
            # and every following pass rebuilt the identical refusal -- the
            # same 512-refusal shape this branch already has on record. The
            # refusal ALSO armed a leak that is only safe while rare: the
            # `except PPScheduleRefused` handler in scheduler.py documents
            # that requests admitted earlier in the same loop keep their
            # `inc_lock_ref` when the batch never completes, and justifies
            # leaving it open with "Bounded: it takes a genuinely unexecutable
            # geometry to reach this line at all". This condition is ordinary,
            # not exotic, so the refusal turned a bounded leak into a per-pass
            # ratchet. Both reasons point the same way and the refusal is
            # withdrawn.
            #
            # WHY NOT ANNOUNCING LOSES NOTHING HERE, which is what the raise
            # got wrong. `new_chunked_req` / `scheduler.chunked_req` is LOCAL
            # bookkeeping for a chunk this rank decided itself: it exists so
            # the next pass can find the continuation and resume it. On a
            # FORWARDED schedule that job belongs to the upstream -- this
            # method is handed `prefix_len` and `extend_len` fresh every pass
            # (`_add_scheduled_req`'s own "LAST CHUNK OR NOT IS ALSO THE
            # SCHEDULE'S TO SAY"), so the continuation is re-established from
            # the decision whether or not this rank remembered it. The chunk
            # therefore runs, as decided, and nothing is re-prefilled: no
            # double prefill, no dropped named request, and the assert stays
            # intact because the single field keeps its one occupant.
            if self.chunked_req_outstanding:
                note_second_continuation_refused(req, "_add_scheduled_req")
            else:
                self._mint_chunked(req, "_add_scheduled_req")
        if not carried_chunk:
            self._req_inc_lock_ref(req)
        self._update_prefill_budget(
            0 if carried_chunk else prefix_len,
            extend_len,
            (
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
                if last_chunk
                else 0
            ),
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
            mamba_slot_charge=self._mamba_slots_for_req(req),
        )
        return AddReqResult.CONTINUE

    def add_chunked_req(self, req: Req):
        # #791 CORE: the carried chunked request is the consumer that had NO
        # forwarded-decision gate at all. `add_one_req` at least read the
        # schedule's prefix through the admission loop's clamp; this path is
        # entered from scheduler.py before that loop runs and decided its
        # chunk length purely from `rem_chunk_tokens`/`rem_total_tokens` and
        # the #679 park, all rank-local. PP0 carried this rid as `chunked=1`
        # on instr20 while PP1 and PP2 carried it as `chunked=0`.
        scheduled = self.scheduled_extent_for(req)
        if scheduled is not None:
            self._add_scheduled_req(req, scheduled, carried_chunk=True)
            # `add_chunked_req` answers "is this still the chunked request",
            # and under a forwarded schedule that is the same last-chunk fact
            # `_add_scheduled_req` just applied -- so it must come from the
            # same source. #996: the read-back off the request was equivalent
            # only while BOTH sides derived the verdict locally. Now that
            # `_add_scheduled_req` executes a carried one, re-deriving here
            # would put this rank's own `full_untruncated_fill_ids` back into
            # the answer through the side door -- the very quantity the carry
            # exists to take out of it, and the boot 16 skew (8446 vs 8447)
            # would reappear one method along.
            told_last_chunk = self._told_last_chunk(req)
            if told_last_chunk is not None:
                return None if told_last_chunk else req
            return (
                req
                if req.extend_range.end < len(req.full_untruncated_fill_ids)
                else None
            )
        if self.dllm_config is not None:
            _rem_tokens = self._get_dllm_remain_tokens()
        else:
            _rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))
            if self.is_hybrid_swa:
                # alloc_extend needs extend_num_tokens + page_size per request,
                # so reserve one page here to avoid OOM
                _rem_tokens = min(
                    _rem_tokens, int(self.rem_swa_tokens) - self.page_size
                )
            # The chunked_req must be added to the list; otherwise, it will cause a memory leak.
            # Therefore, in certain cases where _rem_tokens <= 0, it should be replaced with rem_chunk_tokens.
            if _rem_tokens <= 0:
                if self.is_hybrid_swa:
                    # #961 LEFT DELIBERATELY UNCHANGED, and the reason is
                    # recorded because the opposite edit was written and
                    # withdrawn. This branch returns the request WITHOUT
                    # `set_extend_range`, while the #679 park below writes
                    # `Range(prefix, prefix)` -- so the two park branches
                    # describe the same state differently, and the #679
                    # comment even cites this one as its precedent while
                    # defining the park by its GEOMETRY.
                    #
                    # It is not a #961 producer any more: the only way this
                    # return could hand on an unreadable geometry was a
                    # truncation having nulled it first, and
                    # `Req.truncate_prefix_to` now re-derives instead of
                    # nulling. Closing it here as well would be consistency,
                    # not a fix -- and it is not free. `test_prefill_adder.py::
                    # test_add_chunked_req_hybrid_swa_defers_when_swa_below_page`
                    # pins "returned unchanged" by asserting
                    # `set_extend_range.assert_not_called()`, and overwriting
                    # the geometry here would also overwrite the PREVIOUS
                    # chunk's range on any path that reaches this branch before
                    # that chunk is stashed. In production the stash at
                    # scheduler.py:7010 runs earlier in the same pass, so the
                    # write would be value-neutral -- but that is an argument,
                    # not a measurement, and hybrid-SWA is not a configuration
                    # this fork boots. Named as open rather than changed blind.
                    return req
                # #679: DO NOT SCHEDULE A CHUNK THE POOL CANNOT FUND.
                #
                # Overriding a zero budget with ``rem_chunk_tokens`` is what
                # killed the instance at 23:41:01. The budget said nothing was
                # available, this line admitted a 512-token chunk anyway,
                # ``alloc_token_slots`` found available 0 / evictable 0, its
                # only relief (evict_from_tree_cache) was a no-op with nothing
                # evictable, and the hard RuntimeError took all three ranks
                # down together. The override exists for a real reason -- a
                # chunked request that leaves this function unhandled leaks --
                # but "admit it anyway" is not the only way to keep it.
                #
                # PARKING IS ALREADY A FIRST-CLASS STATE. The hybrid-SWA branch
                # above returns the request unmodified, and the scheduler
                # documents the result at the chunked-request stash: a parked
                # chunk "leaves extend_range.end == len(prefix_indices), so
                # there is nothing new to cache and stashing would be a no-op".
                # So the request stays the chunked request, is retried next
                # round, and nothing leaks -- the same contract, without the
                # allocation that cannot succeed.
                #
                # THE PREDICATE IS GROUP-UNIFORM. ``fundable_extend_tokens``
                # reads the published availability floor rather than this
                # rank's own, so every rank parks on the same iteration.
                # Deciding this from a rank-local size would split the group
                # across different batches, which is a hang rather than a
                # stall.
                from sglang.srt.mem_cache.common import (
                    chunk_tokens_the_pool_can_fund,
                    fundable_extend_tokens,
                )

                fundable = fundable_extend_tokens(self.tree_cache)
                grant = chunk_tokens_the_pool_can_fund(
                    fundable, self.page_size, self.rem_chunk_tokens
                )
                if grant <= 0:
                    req.set_extend_range(
                        len(req.prefix_indices), len(req.prefix_indices)
                    )
                    logger.warning(
                        "chunked prefill PARKED: the pool can fund %d tokens, "
                        "below one page (%d). The request keeps its place and "
                        "is retried when memory frees; admitting it here is "
                        "what took the instance down on 2026-08-15.",
                        fundable,
                        self.page_size,
                    )
                    return req
                # Fundable, but only just: take what the pool can actually
                # give rather than the nominal chunk size.
                _rem_tokens = grant

        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)
        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            req.extend_range.length,
            (
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
                if not truncated
                else 0
            ),
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
            mamba_slot_charge=self._mamba_slots_for_req(req),
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        dec_lock_params = None
        try:
            result = self.tree_cache.inc_lock_ref(last_node)
            if self.tree_cache.is_tree_cache():
                # init_load_back may revive SWA/Mamba tombstones while this
                # temporary admission lock is held. Release must mirror the
                # exact nodes skipped at acquire time.
                dec_lock_params = result.to_dec_params()
            yield None
        finally:
            if dec_lock_params is not None:
                self.tree_cache.dec_lock_ref(last_node, dec_lock_params)
            else:
                self.tree_cache.dec_lock_ref(last_node)

    def add_one_req_ignore_eos(self, req: Req):
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        paged_input = self.ceil_paged_tokens(cand_extend_input_len)
        # Shared Mamba pool: fold the new mamba state's shared-gap cost into the
        # budget gate so admission can't over-commit (0 for baseline / non-Mamba).
        paged_input += self._mamba_gap_budget_for_req(req)
        if paged_input > min(self.cur_rem_tokens, self.rem_total_tokens):
            return AddReqResult.NO_TOKEN
        if self.is_hybrid_swa:
            if self._swa_budget_for_req(cand_extend_input_len) > self.rem_swa_tokens:
                return AddReqResult.NO_TOKEN

        def add_req_state(r, insert_sort=False):
            new_token_ratio = (
                1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
            )
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(
                r.output_ids
            )
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)

            if tokens_left <= 0:
                return

            if not insert_sort:
                self.req_states.append((tokens_left, tokens_occupied))
            else:
                i = 0
                for i in range(len(self.req_states)):
                    if tokens_left <= self.req_states[i][0]:
                        break
                self.req_states.insert(i, (tokens_left, tokens_occupied))

        if self.req_states is None:
            self.req_states = []
            add_req_state(req)
            if self.running_batch is not None:
                for r in self.running_batch.reqs:
                    add_req_state(r)
            for r in self.can_run_list:
                add_req_state(r)
            self.req_states.sort(key=lambda x: x[0])
        else:
            add_req_state(req, insert_sort=True)

        if not self.is_hybrid_swa:
            # Skip this logic for swa. The SWA has different memory management, and
            # this mechanism is underestimating the memory usage.
            cur_rem_tokens = self.cur_rem_tokens - self.ceil_paged_tokens(
                cand_extend_input_len
            )
            tokens_freed = 0
            for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
                # tokens_left gives a reservative calculation as the last token is not stored
                bs = len(self.req_states) - i
                min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
                # reserve tokens for corner cases
                if min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs:
                    return AddReqResult.NO_TOKEN
                tokens_freed += tokens_occupied

        if (self.prefill_delayer_single_pass is not None) and (
            not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True
            )
        ):
            return AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER

            self._add_dllm_req(req, 0)
        elif (
            self.rem_chunk_tokens is None  # chunked prefill is disabled
            or cand_extend_input_len <= self.rem_chunk_tokens  # it is the last chunk
        ):
            # Non-chunked prefill — the whole sequence is committed this iter.
            req.set_extend_range(
                len(req.prefix_indices), len(req.full_untruncated_fill_ids)
            )
            self.can_run_list.append(req)
            self._update_prefill_budget(
                0,
                req.extend_range.length,
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),
                req.retracted_stain,
                mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
                mamba_slot_charge=self._mamba_slots_for_req(req),
            )
        else:
            if self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

            # #959 ONE CONTINUATION AT A TIME, STATED RATHER THAN EMERGENT.
            #
            # `scheduler.py`'s `assert self.chunked_req is None` is upheld
            # today by ARITHMETIC: a surviving continuation has normally spent
            # all of `rem_chunk_tokens`, so this branch is not reached. #951
            # recorded that as emergent and BREAKABLE, with witnesses under
            # /spinning/evidence-665-f1/witness_951/ driving three states where
            # `add_chunked_req` leaves `rem_chunk_tokens` positive (a
            # `rem_total_tokens`-bound truncation, the hybrid-SWA early exit,
            # the #679 park) and a mid-pass replenishment then mints a SECOND
            # continuation. #951 closed only the PP instance (a #798-voided
            # pass no longer runs this function); the general case was left as
            # its own posten.
            #
            # THIS IS THAT POSTEN, and it is reachable: window-955-boot's
            # second boot died on that assert on ALL THREE ranks three seconds
            # after the first clean `pp_to_tp` cutover, with a `tp_to_pp` flip
            # already armed and reporting "NOT QUIESCENT: a chunked prefill is
            # incomplete". A cutover resizes the pool, which is exactly the
            # mid-pass replenishment the witnesses need. window-951's 0/0 on
            # that line was VACUOUS -- every batch it saw was `phase=pp`.
            #
            # REFUSING IS THE SAFE DIRECTION HERE, and that is the whole
            # danger-direction analysis. This request is FRESH: nothing of it
            # has been committed, no KV is held for it, no chunk of it has run.
            # Leaving it for a later pass is the requeue-for-free the admission
            # loop already relies on -- no progress is lost, so no double
            # prefill (the standing law). It cannot starve either: the resident
            # continuation is consuming chunks, and when it finishes
            # `chunked_req` is None and this request is admitted.
            #
            # WHAT WOULD NOT BE SAFE is the mirror-image fix -- clearing
            # `scheduler.chunked_req` at the cutover. That drops a request
            # MID-PREFILL and re-prefills it, which is the double prefill the
            # law forbids outright, and the #858 wedge shape besides. The
            # resident continuation is never the one to give way; the fresh
            # admission is.
            #
            # The precedent is `_add_scheduled_req`'s `carried_chunk` flag,
            # which already refuses to announce a NEW chunked req for exactly
            # this reason and names this assert while doing it.
            if self.chunked_req_outstanding:
                # #967: count and name it -- see note_second_continuation_refused.
                note_second_continuation_refused(req, "add_one_req_ignore_eos")
                return AddReqResult.OTHER

            # Chunked prefill
            trunc_len = self.rem_chunk_tokens

            assert len(req.prefix_indices) == 0
            req.set_extend_range(
                len(req.prefix_indices), len(req.prefix_indices) + trunc_len
            )
            self.can_run_list.append(req)
            self._mint_chunked(req, "add_one_req_ignore_eos")
            self._update_prefill_budget(
                0,
                trunc_len,
                0,
                req.retracted_stain,
                mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
                mamba_slot_charge=self._mamba_slots_for_req(req),
            )

        return self.budget_state()

    def add_one_req(
        self, req: Req, truncation_align_size: Optional[int]
    ):
        # PS2 batch separation (see budget_state): a born-spilled-deep prompt
        # owns its extend batch exclusively.
        if self.prefill_spill_deep_taken:
            return AddReqResult.OTHER
        if (self.prefill_delayer_single_pass is not None) and (
            not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
                waiting_queue_len=self.waiting_queue_len,
            )
        ):
            return AddReqResult.OTHER
        # TODO support cp with multiple requests
        # Enabling context parallelism currently presents precision issues;
        # therefore, the prefill-batch setting is temporarily set to 1.
        if (self.dsa_prefill_cp_in_seq_split) and len(self.can_run_list) >= 1:
            return AddReqResult.OTHER

        if (x := self.prefill_max_requests) is not None and len(self.can_run_list) >= x:
            return AddReqResult.OTHER

        if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
            return self.add_one_req_ignore_eos(req)

        # #791 CORE: EXECUTE, DO NOT DERIVE.
        #
        # Placed above every rank-local gate in this method on purpose. The
        # gates below (`rem_total_tokens`, `rem_input_tokens`,
        # `rem_chunk_tokens`, the host load-back, the page/alignment
        # truncation) each answer a question the first rank has already
        # answered for this pass and forwarded. Re-answering them is not a
        # second opinion, it is a SECOND SCHEDULE -- and the upstream's hidden
        # states are already on the wire for the first one.
        #
        # NOT a new communication and not a new branch on peer state: the
        # mapping was received earlier in this same pass by the wire #791
        # already built, and it is None on every rank that owns its own
        # admission truth. See `scheduled_extent_for`.
        scheduled = self.scheduled_extent_for(req)
        if scheduled is not None:
            if self.dllm_config is not None:
                from sglang.srt.managers.pp_admission_congruence import (
                    PPScheduleRefused,
                )

                raise PPScheduleRefused(
                    f"#791 FORWARDED SCHEDULE UNEXECUTABLE for rid={req.rid}: "
                    f"DLLM prefill re-derives its own block geometry "
                    f"(`_add_dllm_req`) and cannot execute a forwarded one. "
                    f"Refusing rather than running two schedules at once."
                )
            with self._lock_node(req.last_node):
                return self._add_scheduled_req(req, scheduled)

        # Reserve page_size for page-alignment overhead: the paged allocator may
        # consume one extra page per request (see alloc_extend), which
        # _update_prefill_budget also deducts.
        max_new = min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        total_tokens = cand_extend_input_len + max_new + self.page_size
        # Shared Mamba pool: fold the new mamba state's shared-gap cost into
        # `total_tokens` so both `rem_total_tokens` gates reflect the joint budget.
        total_tokens += self._mamba_gap_budget_for_req(req)
        # Prefill-Spill (PS1-V1a): the born-spilled current-step demand is the
        # lifetime demand MINUS the future decode (max_new) that will spill to
        # host -- i.e. just the prefill input (+ page + mamba gap). Used only if
        # the lifetime gate below fails and the feature is on.
        born_input_tokens = total_tokens - max_new

        # adjusting the input_tokens based on host_hit_length and page_size
        real_input_tokens = cand_extend_input_len - req.host_hit_length
        real_input_tokens = self.ceil_paged_tokens(real_input_tokens)
        prefix_len = len(req.prefix_indices)

        if total_tokens >= self.rem_total_tokens:
            # Lifetime doesn't fit VRAM: wedge -- UNLESS Prefill-Spill can admit
            # it born-spilled (input transiently fits, a host region is free),
            # or -- PS2, the strict complement -- born-spilled DEEP (not even
            # the input fits; the prefill never materializes device KV slots).
            if not self._admit_born_spilled(
                req, born_input_tokens
            ) and not self._admit_born_spilled_deep(req, born_input_tokens):
                return AddReqResult.NO_TOKEN

        if self.is_hybrid_swa:
            swa_needed = self._swa_budget_for_req(
                cand_extend_input_len, swa_host_hit_length=req.swa_host_hit_length
            )
            if swa_needed >= self.rem_swa_tokens:
                return AddReqResult.NO_TOKEN

        if (
            self.rem_chunk_tokens is None
            and len(self.can_run_list) != 0
            and real_input_tokens >= self.rem_input_tokens
        ):
            # If without chunked prefill:
            # - if the can_run_list is not empty, we satisfy the constraint of (max_prefill_tokens)
            # - if the can_run_list is empty, always accept the first prefill request
            return AddReqResult.OTHER

        with self._lock_node(req.last_node):
            # self.rem_total_tokens may decrease after the lock acquisition
            if total_tokens >= self.rem_total_tokens:
                # Prefill-Spill: a prompt already admitted born-spilled at the
                # pre-lock gate stays admitted as long as its input still fits
                # the (possibly shrunk) device budget; otherwise wedge as usual.
                # PS2: a born-spilled-DEEP prompt allocates NO device KV slots
                # at all, so a shrunk device budget cannot invalidate it -- it
                # stays admitted (its host region was reserved at the pre-lock
                # verdict, replicated on every rank).
                if not req.born_spilled_deep and not (
                    req.born_spilled and born_input_tokens < self.rem_total_tokens
                ):
                    return AddReqResult.NO_TOKEN

            if self.is_hybrid_swa:
                swa_needed = self._swa_budget_for_req(
                    cand_extend_input_len, swa_host_hit_length=req.swa_host_hit_length
                )
                if swa_needed >= self.rem_swa_tokens:
                    return AddReqResult.NO_TOKEN

            # #1035: the load-back runs only where its result can be
            # rank-uniform. Under PP the host tier is layer-partitioned, so
            # this rank's hit is its own and growing the prefix on it alone
            # is the shape divergence boot 1815081d46 died of. The refusal
            # and its instrument live behind the same predicate, so the line
            # can never report a state the branch did not take.
            if req.needs_host_load_back() and not _pp_forbids_rank_local_load_back(
                req
            ):
                new_indices, req.last_node = self.tree_cache.init_load_back(
                    InitLoadBackParams(
                        best_match_node=req.best_match_node,
                        host_hit_length=req.host_hit_length,
                        req=req,
                    )
                )
                req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
                prefix_len = len(req.prefix_indices)
                req.cache_protected_len = prefix_len
                # #988 JOIN THE ENSEMBLE AT THE MUTATION, NOT AT THE EXITS.
                # This host load-back is the ONE prefix mover in the tree that
                # did not re-derive its co-derived geometry (#965's invariant:
                # extend_range.start == len(prefix_indices) -- asserted at the
                # batch boundary, schedule_batch.py prepare_for_extend). Every
                # early return below this line used to leave the FIRST visit's
                # extend_range behind a moved prefix; boot 8 of
                # window-flip-0828 died on exactly that at 25s (a waiting_queue
                # member already resident in can_run_list was visited twice,
                # grew its prefix at this line, bailed at a budget return, and
                # prepare_for_extend read the halves apart). Re-deriving HERE,
                # immediately, makes every current and future early return
                # inherit a consistent parked shape automatically -- the same
                # Range(prefix, prefix) the void park writes -- and the success
                # paths below overwrite it with the real range as before.
                # Per-branch bail patches are how #965 was paid for twice.
                req.set_extend_range(prefix_len, prefix_len)
                _note_988_loadback(req, prefix_len)

            input_tokens = self.ceil_paged_tokens(
                len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
            )

            if (
                self.rem_chunk_tokens is None
                and len(self.can_run_list) != 0
                and input_tokens >= self.rem_input_tokens
            ):
                # If without chunked prefill:
                # - if the can_run_list is not empty, we satisfy the constraint of (max_prefill_tokens)
                # - if the can_run_list is empty, always accept the first prefill request
                return AddReqResult.OTHER

            if self.dllm_config is not None:
                if self.rem_dllm_tokens <= 0:
                    return AddReqResult.OTHER

                assert (
                    truncation_align_size is None
                ), "truncation_align_size is not supported for dllm prefill"

                self._add_dllm_req(req, prefix_len)
                self._req_inc_lock_ref(req)
            elif self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
                # Non-chunked prefill — the whole sequence is committed this iter.
                req.set_extend_range(
                    len(req.prefix_indices), len(req.full_untruncated_fill_ids)
                )
                self.can_run_list.append(req)

                self._req_inc_lock_ref(req)
                self._update_prefill_budget(
                    prefix_len,
                    input_tokens,
                    min(
                        req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKENS,
                    ),
                    req.retracted_stain,
                    mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
                    mamba_slot_charge=self._mamba_slots_for_req(req),
                )
            else:
                # Make sure at least one page is available
                trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size

                if trunc_len <= 0:
                    return AddReqResult.OTHER

                # When truncation align size is set, we want to assert that the prefill prefix length is multiple of truncation align size
                # A typical use case is when deterministic inference is enabled with flashinfer attention backend,
                # we need the prefill prefix length to be multiple of attention split size
                if truncation_align_size is not None:
                    if trunc_len < truncation_align_size:
                        return AddReqResult.OTHER
                    else:
                        trunc_len = truncation_align_size * (
                            trunc_len // truncation_align_size
                        )

                now_input_len = trunc_len + len(req.prefix_indices)
                now_input_len = now_input_len // self.page_size * self.page_size
                trunc_len = now_input_len - len(req.prefix_indices)

                if trunc_len <= 0:
                    return AddReqResult.OTHER

                # #959 ONE CONTINUATION AT A TIME -- the sibling of the guard
                # in the no-prefix branch above, which carries the reasoning.
                # Both fresh-request mint sites need it; the forwarded-schedule
                # site already has its own (`carried_chunk`). Missing it here
                # would leave the same assert reachable by the longer path.
                if self.chunked_req_outstanding:
                    # #967: same guard, second mint site, same instrument.
                    note_second_continuation_refused(req, "add_one_req")
                    return AddReqResult.OTHER

                # Chunked prefill
                req.set_extend_range(
                    len(req.prefix_indices), len(req.prefix_indices) + trunc_len
                )

                self.can_run_list.append(req)
                self._mint_chunked(req, "add_one_req")

                self._req_inc_lock_ref(req)
                self._update_prefill_budget(
                    prefix_len,
                    trunc_len,
                    0,
                    req.retracted_stain,
                    mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
                    mamba_slot_charge=self._mamba_slots_for_req(req),
                )

        return self.budget_state()

    def preempt_to_schedule(self, req: Req, server_args: ServerArgs) -> bool:
        """
        Preempt running requests to serve the new request if the priority threshold is met and token count sum is verified.
        Returns True if preemption was committed, and the new request can be scheduled.
        """
        # Iterate running requests to find preemptible requests
        priority_sign = 1 if server_args.schedule_low_priority_values_first else -1

        # NOTE: A request finishes in two phases:
        #   1) update_finish_state + release_kv_cache  (in process_batch_result)
        #   2) filter out of batch                (in get_next_batch_to_run / update_running_batch)
        # Preemption runs between these two phases (inside get_new_batch_prefill),
        # so running_batch may still contain requests whose KV cache is already freed.
        # We must skip them here to avoid a double-free on release_req.
        valid_running_reqs = (
            r
            for r in self.running_batch.reqs
            if r not in self.preempt_list and not r.finished()
        )

        sorted_valid_running_reqs = sorted(
            valid_running_reqs,
            key=lambda x: (
                x.priority * (-priority_sign),
                -x.time_stats.wait_queue_entry_time,
            ),
        )

        preemptible_reqs = []
        min_tokens_to_remove = (
            len(req.full_untruncated_fill_ids)
            - len(req.prefix_indices)
            + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            - self.rem_total_tokens
        )

        # Fast lane (Variant C Stage 0) anti-starvation: never preempt heavy
        # ('is_fast_lane' False) requests below the reserved floor, so at least
        # --fast-lane-reserved-heavy-slots heavy requests keep running and make
        # forward progress even under sustained fast-lane load. Disabled (no
        # cap) when the fast lane is off, preserving the default behavior.
        max_heavy_preemptible = None
        if getattr(server_args, "enable_fast_lane", False):
            num_heavy_running = sum(
                1 for r in sorted_valid_running_reqs if not getattr(r, "is_fast_lane", False)
            )
            max_heavy_preemptible = max(
                0, num_heavy_running - server_args.fast_lane_reserved_heavy_slots
            )
        heavy_preempted = 0

        for running_req in sorted_valid_running_reqs:
            # Priority difference needs to meet the threshold to be preemptible.
            priority_diff = (req.priority - running_req.priority) * (-priority_sign)

            if priority_diff > self.priority_scheduling_preemption_threshold:
                is_heavy = not getattr(running_req, "is_fast_lane", False)
                if (
                    max_heavy_preemptible is not None
                    and is_heavy
                    and heavy_preempted >= max_heavy_preemptible
                ):
                    # Reserved-heavy-slots floor reached: stop preempting.
                    break
                preemptible_reqs.append(running_req)
                if is_heavy:
                    heavy_preempted += 1
                min_tokens_to_remove -= self._get_running_request_total_token_offset(
                    running_req
                )
                if min_tokens_to_remove <= 0:
                    break
            else:
                break

        # Check max token count limit can be met
        if len(preemptible_reqs) == 0 or min_tokens_to_remove > 0:
            return False

        # Preempt running requests. Release allocated resources for immediate usage.
        preemptible_reqs = set(preemptible_reqs)
        keep_indices = []
        release_counter = 0
        for i, running_req in enumerate(self.running_batch.reqs):
            if running_req in preemptible_reqs:
                self.rem_total_token_offset -= (
                    self._get_running_request_total_token_offset(running_req)
                )
                release_counter += 1
                self.running_batch.release_req(
                    i, len(self.running_batch.reqs) - release_counter, server_args
                )
            else:
                keep_indices.append(i)
        self.running_batch.filter_batch(keep_indices=keep_indices)
        self.preempt_list.extend(preemptible_reqs)
        return True
