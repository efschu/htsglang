from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Deque,
    List,
    Optional,
    Tuple,
)

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    PoolStats,
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import (
    ceil_align,
    raise_error_or_warn,
)
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


logger = logging.getLogger(__name__)

# Number of recent busy-check messages buffered for the level-1 dump-on-leak path.
BUSY_MEM_CHECK_LOG_RING_SIZE = 1000


def _dcp_global_slot_total(
    server_args: ServerArgs,
    allocator: BaseTokenToKVPoolAllocator,
    max_total_num_tokens: int,
) -> int:
    """``total`` for the full-pool leak check, in the allocator's slot space.

    Thin adapter: it only resolves whether this process is on the uneven-DCP
    token-sharded lane and hands the two scalars to
    ``dcp_accounting_total_slots``, which states the rule and the reason once.
    """
    from sglang.srt.distributed.utils import uneven_dcp_kv_replicated
    from sglang.srt.layers.dcp.owner import dcp_accounting_total_slots

    dcp_size = int(getattr(server_args, "dcp_size", 1) or 1)
    return dcp_accounting_total_slots(
        max_total_num_tokens,
        getattr(allocator, "size", None),
        token_sharded_dcp=uneven_dcp_kv_replicated(dcp_size),
    )


@dataclass(kw_only=True, slots=True)
class SchedulerInvariantChecker:
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    disaggregation_mode: DisaggregationMode
    page_size: int
    full_tokens_per_layer: Optional[int]
    swa_tokens_per_layer: Optional[int]
    max_total_num_tokens: int
    server_args: ServerArgs
    tree_cache: BasePrefixCache
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    req_to_token_pool: ReqToTokenPool
    pool_stats_observer: SchedulerPoolStatsObserver
    get_last_batch: Callable
    get_running_batch: Callable
    count_req_pool_leak_warnings: int = 0
    count_memory_leak_warnings: int = 0
    recent_busy_msgs: Deque[str] = field(
        default_factory=lambda: deque(maxlen=BUSY_MEM_CHECK_LOG_RING_SIZE)
    )

    @staticmethod
    def _check_pool_invariant(
        pool_name: str,
        available: int,
        evictable: int,
        protected: int,
        session_held: int,
        total: int,
        uncached: int = 0,
        withheld: int = 0,
    ) -> Tuple[bool, str]:
        """Check: available + evictable + protected + session_held + uncached
        + withheld == total.

        ``withheld`` is capacity the #656 residency controller has DELIBERATELY
        taken out of circulation: slot ids above the KV pool's backed watermark,
        held out of the free list so that nothing is handed out over unmapped
        memory. It is not available, not cached and not held by a session, so
        without a term of its own it reads as a leak -- and it killed the boot
        that first exercised it ("total=500000, available=419745").

        It is a NAMED POSTEN for the #486 reason: anything that durably occupies
        or removes pool slots must be named in this ledger, or the next
        unexplained delta gets attributed to the wrong holder.
        """
        total_accounted = (
            available + evictable + protected + session_held + uncached + withheld
        )
        leak = total_accounted != total
        msg = (
            f"[{pool_name}] {total=}, {available=}, {evictable=}, "
            f"{protected=}, {session_held=}, {uncached=}, {withheld=}"
        )
        return leak, msg

    def _check_full_pool(self, ps: PoolStats, uncached: int = 0) -> Tuple[bool, str]:
        if self.is_hybrid_swa and not self.full_tokens_per_layer:
            return False, ""
        if self.is_hybrid_swa:
            protected = self.tree_cache.full_protected_size()
            session_held = self.pool_stats_observer.session_held_full_tokens()
            total = self.full_tokens_per_layer
        elif self.is_hybrid_ssm:
            # Branch on cache type for the protected accessor (MambaRadixCache
            # splits full/mamba; ChunkCache only has the single protected_size).
            # Use the allocator's `.size` for `total`: static max_total_num_tokens for
            # non-unified pools, the dynamic byte-coordinated cap (matching
            # `available_size`) for the unified pool.
            if self.tree_cache.supports_mamba():
                protected = self.tree_cache.full_protected_size()
            else:
                protected = self.tree_cache.protected_size()
            session_held = self.pool_stats_observer.session_held_tokens()
            total = self.token_to_kv_pool_allocator.size
        else:
            protected = self.tree_cache.protected_size()
            session_held = self.pool_stats_observer.session_held_tokens()
            # `available_size` counts the ALLOCATOR's index space, so `total`
            # has to be the same quantity. They coincide everywhere except on
            # the uneven-DCP token-sharded lane under the EVEN-MODULO owner
            # rule (SGLANG_UNEVEN_DCP=1, SGLANG_UNEVEN_DCP_WEIGHTED=0): there
            # `max_total_num_tokens` is this rank's PHYSICAL pool while the
            # allocator hands out GLOBAL slot ids over
            # `max_total * cp_token_split_factor(dcp_size)` of them, so the
            # very first idle check reported `available` at exactly dcp_size x
            # `total` and the server died at boot with "pool memory leak
            # detected! [full] total=28640, available=57280" (#345). Under the
            # weighted rule `max_total_num_tokens` already IS the global
            # context C == the allocator size, so nothing moves there, and off
            # the lane the two are the same number by construction.
            total = _dcp_global_slot_total(
                self.server_args,
                self.token_to_kv_pool_allocator,
                self.max_total_num_tokens,
            )
        full_evictable_size = ps.full_evictable_size
        allocator = self.token_to_kv_pool_allocator
        if getattr(self.server_args, "dcp_size", 1) > 1 and allocator.page_size > 1:
            # DCP stores logical tokens in widened physical pages.  Prefix cache
            # counters are logical-token based, while the allocator frees whole
            # physical pages, so round cached tokens up to physical page units.
            full_evictable_size = (
                (full_evictable_size + allocator.page_size - 1)
                // allocator.page_size
                * allocator.page_size
            )
        leak, msg = self._check_pool_invariant(
            "full",
            ps.full_available_size,
            full_evictable_size,
            protected,
            session_held,
            total,
            uncached,
            # Slots the residency controller holds out of the free list because
            # the pages under them are unmapped (#656 item 12). Published by
            # KvRowCap in the same unit available_size() reports.
            int(getattr(allocator, "residency_withheld_slots", 0) or 0),
        )
        if (
            leak
            and getattr(self.server_args, "dcp_size", 1) > 1
            and allocator.page_size > 1
        ):
            # Radix/Mamba cache accounting is logical-token based while DCP full
            # KV allocation is physical-page based. Partial physical pages can
            # leave a small page-level slack even when all pages are owned by
            # either the allocator or the prefix cache.
            return False, f"{msg}, dcp_physical_page_slack_allowed=True"
        return leak, msg

    def _check_swa_pool(self, ps: PoolStats, uncached: int = 0) -> Tuple[bool, str]:
        return self._check_pool_invariant(
            "swa",
            ps.swa_available_size,
            ps.swa_evictable_size,
            self.tree_cache.swa_protected_size(),
            self.pool_stats_observer.session_held_swa_tokens(),
            self.swa_tokens_per_layer,
            uncached,
        )

    def _check_mamba_pool(self, ps: PoolStats) -> Tuple[bool, str]:
        ckpt_pool = getattr(self.req_to_token_pool, "mamba_ckpt_pool", None)
        if ckpt_pool is not None:
            return self._check_mamba_pool_with_int8(ps, ckpt_pool)
        leak, msg = self._check_pool_invariant(
            "mamba",
            ps.mamba_available_size,
            ps.mamba_evictable_size,
            self.tree_cache.mamba_protected_size(),
            self.pool_stats_observer.session_held_mamba_slots(),
            self.req_to_token_pool.mamba_pool.size,
        )
        if leak:
            # Page-level leak diagnosis for mamba
            free_full_pages = set(
                self.token_to_kv_pool_allocator.free_pages.tolist()
                + self.token_to_kv_pool_allocator.release_pages.tolist()
            )
            cached_full_pages = set(self.tree_cache.all_values_flatten().tolist())
            expected_full_pages = set(
                range(1, self.token_to_kv_pool_allocator.size + 1)
            )
            leaked_full_pages = (
                expected_full_pages - free_full_pages - cached_full_pages
            )
            mamba_allocator = self.req_to_token_pool.mamba_allocator
            free_mamba_pages = set(mamba_allocator.free_slots.tolist())
            cached_mamba_pages = set(
                self.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(1, mamba_allocator.size + 1))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            msg += (
                f", leaked_full_pages={leaked_full_pages or None}"
                f", leaked_mamba_pages={leaked_mamba_pages or None}"
            )
        return leak, msg

    def _check_mamba_pool_with_int8(self, ps: PoolStats, ckpt_pool) -> Tuple[bool, str]:
        """Two-pool invariant for int8 mamba checkpoints.

        The radix-cached states live in the int8 checkpoint pool, NOT the active
        bf16 pool. So the single-pool equation (active.available + radix_cached ==
        active.size) is wrong -- it double-counts the radix states against a pool
        that does not hold them. Instead check the two pools independently:

          * active bf16 pool: backs running requests only; the radix owns ZERO
            active slots. Checked at idle (in-flight == 0) -> available == total.
          * int8 checkpoint pool: backs the radix-cached states; its occupancy is
            exactly the radix evictable + protected counts.
        """
        active_leak, active_msg = self._check_pool_invariant(
            "mamba-active",
            ps.mamba_available_size,
            ps.mamba_evictable_size,  # 0 in int8 mode (radix owns no active slots)
            0,
            self.pool_stats_observer.session_held_mamba_slots(),
            self.req_to_token_pool.mamba_pool.size,
        )
        int8_leak, int8_msg = self._check_pool_invariant(
            "mamba-int8",
            ckpt_pool.available_size(),
            self.tree_cache.mamba_evictable_size(),
            self.tree_cache.mamba_protected_size(),
            0,
            ckpt_pool.num_slots,
        )
        return active_leak or int8_leak, active_msg + "\n" + int8_msg

    def _get_total_uncached_sizes(
        self,
    ) -> Tuple[int, int]:
        """Sum uncached tokens for full and SWA pools across all active batches.

        Returns (full_uncached, swa_uncached). For non-SWA models, swa_uncached is 0.

        For full pool: uncached = allocated - cache_protected_len
        For SWA pool:  uncached = allocated - max(cache_protected_len, swa_evicted_seqlen)
        """
        # After decode: running_batch IS last_batch (same object), count once.
        # After prefill: they differ, both hold uncached tokens.
        # Use identity (is / is not), not membership or ==: ScheduleBatch's
        # dataclass __eq__ compares tensor fields and raises on ambiguous bools.
        last_batch = self.get_last_batch()
        running_batch = self.get_running_batch()
        batches = [last_batch]
        if (
            running_batch is not None
            and running_batch is not last_batch
            and not running_batch.is_empty()
        ):
            batches.append(running_batch)

        full_uncached = 0
        swa_uncached = 0
        for batch in batches:
            for req in batch.reqs:
                assert req.kv_committed_freed == req.kv_overallocated_freed
                if req.kv_committed_freed or req.req_pool_idx is None:
                    continue

                allocated_len = req.kv_allocated_len
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0

                full_uncached += allocated_len - req.cache_protected_len
                if self.is_hybrid_swa:
                    swa_uncached += allocated_len - max(
                        req.cache_protected_len, req.swa_evicted_seqlen
                    )

        return full_uncached, swa_uncached

    def self_check_during_busy(self):
        if self.get_last_batch() is None:
            return

        ps = self.pool_stats_observer.get_pool_stats()
        full_uncached, swa_uncached = self._get_total_uncached_sizes()

        full_leak, full_msg = self._check_full_pool(ps, uncached=full_uncached)

        swa_leak, swa_msg = False, ""
        if self.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(ps, uncached=swa_uncached)

        level = envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get()
        full_line = f"[Mem Check (BUSY)] {full_msg}"
        swa_line = f"[Mem Check (BUSY)] {swa_msg}" if swa_msg else None

        if level > 1:
            # Verbose: log every iteration.
            logger.info(full_line)
            if swa_line:
                logger.info(swa_line)
        elif level == 1:
            # Quiet: buffer and stay silent; flush the recent ones only on a leak.
            self.recent_busy_msgs.append(full_line)
            if swa_line:
                self.recent_busy_msgs.append(swa_line)
            if full_leak or swa_leak:
                for msg in self.recent_busy_msgs:
                    logger.info(msg)

        assert not full_leak, f"Full Pool Mem Leak Detected! {full_msg}"
        assert not swa_leak, f"SWA Pool Mem Leak Detected! {swa_msg}"

        if envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.get():
            self._check_kv_page_invariants()

    def _check_kv_page_invariants(self):
        """committed<=allocated for every req/slot, and no double free:
          A. no owner references a page that is in the free pool (use-after-free).
          B. the free pool has no duplicate pages (two owners freed the same page).
        All heavy work runs on GPU to avoid per-token device->host sync."""
        rtt = self.req_to_token_pool.req_to_token
        row_width = rtt.shape[1]

        def _add_owner(req_or_slot, label, rpi, committed, allocated):
            assert 0 <= committed <= allocated <= row_width
            owners.append((label, rpi, allocated))

        owners: list[tuple[str, Optional[int], int]] = []
        batch = self.get_last_batch()
        if batch is not None:
            for req in batch.reqs:
                _add_owner(
                    req,
                    f"req {req.rid}",
                    req.req_pool_idx,
                    req.kv_committed_len,
                    req.kv_allocated_len,
                )
        sess = getattr(self.tree_cache, "slots", None)
        if sess:
            for sid, slot in sess.items():
                if getattr(slot, "is_holding_kv", False):
                    _add_owner(
                        slot,
                        f"slot {sid[:8]}",
                        slot.req_pool_idx,
                        slot.kv_committed_len,
                        slot.kv_allocated_len,
                    )

        active = [
            (label, rpi, al) for label, rpi, al in owners if rpi is not None and al > 0
        ]
        if not active:
            return

        idx = torch.as_tensor([rpi for _, rpi, _ in active], device=rtt.device)
        allocs = torch.as_tensor([al for _, _, al in active], device=rtt.device)
        mask = torch.arange(row_width, device=rtt.device)[None, :] < allocs[:, None]
        owner_pages = rtt[idx][mask] // self.page_size

        # Sub-allocators to check: a flat allocator is its own single sub; a
        # hybrid-SWA wrapper exposes full_attn_allocator + swa_attn_allocator.
        alloc = self.token_to_kv_pool_allocator
        sub_allocs = (
            [alloc]
            if getattr(alloc, "free_pages", None) is not None
            else [
                sub
                for n in ("full_attn_allocator", "swa_attn_allocator")
                if (sub := getattr(alloc, n, None)) is not None
                and getattr(sub, "free_pages", None) is not None
            ]
        )
        if not sub_allocs:
            return

        def _free_pages(a):
            free = a.free_pages
            release = getattr(a, "release_pages", None)
            return (
                torch.cat((free, release))
                if release is not None and len(release) > 0
                else free
            )

        # Check B: every sub-pool's free set has no duplicate pages.
        for i, sub in enumerate(sub_allocs):
            free = _free_pages(sub)
            uniq = torch.unique(free)
            if uniq.numel() != free.numel():
                raise_error_or_warn(
                    self,
                    envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                    "count_memory_leak_warnings",
                    f"KV double free: sub-pool {i} has {free.numel() - uniq.numel()} duplicate pages.",
                )

        # Check A: owner pages (full-pool indices) must not be in the full free
        # set (sub_allocs[0] is the full pool, even on hybrid-SWA).
        full_unique = torch.unique(_free_pages(sub_allocs[0]))
        stale = owner_pages[torch.isin(owner_pages, full_unique)]
        if stale.numel() > 0:
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_memory_leak_warnings",
                f"KV page use-after-free: {stale.numel()} owner page refs are in "
                f"the free pool, sample pages={torch.unique(stale)[:8].tolist()}.",
            )

    def _check_req_pool(self):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        session_req_count = self.pool_stats_observer.session_held_req_count()
        if len(self.req_to_token_pool.free_slots) + session_req_count != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"session_held={session_req_count}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def _report_leak(self, pool_name: str, token_msg: str):
        msg = f"{pool_name} memory leak detected! {token_msg}"
        raise_error_or_warn(
            self,
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
            "count_memory_leak_warnings",
            msg,
        )

    def _check_all_pools(
        self, ps: PoolStats, uncached: int = 0
    ) -> Tuple[bool, List[str]]:
        """Check memory invariant across all pools. Returns (has_leak, messages)."""
        has_leak = False
        messages = []

        full_leak, full_msg = self._check_full_pool(ps, uncached=uncached)
        has_leak |= full_leak
        messages.append(full_msg)

        if self.is_hybrid_swa:
            swa_leak, swa_msg = self._check_swa_pool(ps)
            has_leak |= swa_leak
            messages.append(swa_msg)

        if self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            mamba_leak, mamba_msg = self._check_mamba_pool(ps)
            has_leak |= mamba_leak
            messages.append(mamba_msg)

        return has_leak, messages

    def _check_tree_cache(self):
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()


#: #699: seconds a request may sit admissible-but-unserved before the
#: admission wedge is called. 20 s is chosen from the measured specimen, not
#: round: the #713 idle refusal held a TEN-token prompt for 31.64 s with 0
#: running and 1 queued, and an 8-arm run put every TTFT between 11.87 s and
#: 62.65 s. A threshold above 11.87 s would miss the fastest real wedge; 20 s
#: sits above ordinary scheduling latency on this rig (healthy TTFT for a short
#: prompt is sub-second) and below the fastest observed wedge's own duration.
ADMISSION_WEDGE_SECONDS: float = 20.0

ADMISSION_WEDGE = "ADMISSION-WEDGE"


def admission_wedge_verdict(
    queued: int,
    running: int,
    seconds_since_progress: float,
    threshold: float = ADMISSION_WEDGE_SECONDS,
    idle_locked_seen: bool = False,
    seconds_since_prefill_progress: Optional[float] = None,
):
    """``(alarm, detail)`` for the admission-wedge class (#699, from #713).

    WHY forward_ct CANNOT SEE THIS, which is the whole reason this exists.
    ``create_scheduler_watchdog`` fires when ``forward_ct`` stops advancing
    while a batch exists. In the measured wedge BOTH signals read healthy:
    chunked prefill kept running, so ``forward_ct`` advanced, and
    ``cur_batch_for_debug`` stayed non-None -- for 31.64 s during which a
    ten-token prompt produced no first token, with 0 running and 1 queued.
    Health-200 is blind to it for the same reason. The signal is therefore
    QUEUE AGE VERSUS PROGRESS, never batch existence or forward count.

    PROGRESS means a request reached its FIRST TOKEN. Completions are the wrong
    clock -- a long generation legitimately produces none for minutes -- and
    forward passes are the wrong clock for the reason above.

    PREFILL IS ALSO PROGRESS (#739). The first-token clock alone cannot tell a
    wedge from a mega-prefill: a ~500k-token backlog chunking at 512 produces
    NO first token for minutes while prefill batches run the whole time, and it
    presents with the identical three numbers (queued > 0, running 0, large
    age) because a chunked request stays in the waiting queue and nothing is
    decoding. ``seconds_since_prefill_progress`` is the age of the last
    COMPLETED prefill chunk; when one landed inside the window this is not a
    wedge, whatever the first-token clock says.

    That clock is an EVENT TIMESTAMP, deliberately not a pending-token delta:
    the pending counter is the one #731 shows double-billed, and a detector
    keyed to a counter under repair would inherit its noise and its rebases.

    ``None`` means the signal is unavailable, and reproduces the pre-#739
    verdict exactly.

    ``idle_locked_seen`` is the PHASE-POLICY IDLE-LOCKED TERMS line, which
    corroborates when present. It is deliberately NOT required: the wedge class
    is broader than the phase-policy path, and a detector that only fired
    alongside a policy log would miss every wedge arising anywhere else.
    """
    q = int(queued)
    r = int(running)
    age = float(seconds_since_progress)
    if q <= 0:
        return False, "no queue: nothing is waiting, so nothing is wedged"
    if r > 0:
        return False, (
            f"{r} request(s) running: the box is serving, not wedged " f"(queued {q})"
        )
    if age < float(threshold):
        return False, (
            f"queued {q}, running 0, but only {age:.1f}s since the last first "
            f"token (< {float(threshold):.1f}s) -- ordinary scheduling latency"
        )
    prefill_age = (
        None
        if seconds_since_prefill_progress is None
        else float(seconds_since_prefill_progress)
    )
    if prefill_age is not None and prefill_age < float(threshold):
        return False, (
            f"queued {q}, running 0, and no first token for {age:.1f}s -- but a "
            f"prefill chunk completed {prefill_age:.1f}s ago "
            f"(< {float(threshold):.1f}s): the box is PREFILLING, not wedged. A "
            f"mega-prefill produces no first token for minutes by construction"
        )
    return True, (
        f"{ADMISSION_WEDGE}: {q} queued, 0 running, and NO first token for "
        f"{age:.1f}s (>= {float(threshold):.1f}s)"
        + (
            " and no prefill chunk either"
            if prefill_age is not None
            else " (no prefill-progress signal available)"
        )
        + ". Work is admissible and "
        f"nothing is serving it. forward_ct and health both read healthy in "
        f"this state, which is why neither catches it"
        + (
            "; PHASE-POLICY IDLE-LOCKED TERMS corroborates on this round"
            if idle_locked_seen
            else "; no phase-policy corroboration seen -- the wedge class is "
            "broader than that path"
        )
    )


#: #699 wiring: how often the admission-wedge watchdog polls scheduler state.
#: Half the alarm threshold, matching the existing forward_ct watchdog's own
#: poll-vs-timeout ratio (see WatchdogRaw._watchdog_once), so the alarm is
#: never more than one poll interval late.
ADMISSION_WEDGE_POLL_SECONDS: float = ADMISSION_WEDGE_SECONDS / 2

#: #788: seconds a CONFIRMED admission-wedge (i.e. ``admission_wedge_verdict``
#: has already been alarming for this long) must persist before the watchdog
#: attempts recovery, on top of already having crossed ``ADMISSION_WEDGE_
#: SECONDS`` to alarm at all.
#:
#: HAND PIN #770: unlike ``ADMISSION_WEDGE_SECONDS`` (anchored to the #713
#: specimen's own 11.87s-62.65s spread), this number has no independent
#: measurement behind it -- no specimen recorded how long recovery should
#: wait once the report has already fired. It is set to 3x the report
#: threshold (60s) so a forced action never fires on the FIRST poll that
#: alarms, or on a borderline age that a single missed poll interval could
#: have produced: by the time this fires, the wedge has been confirmed,
#: independently, on at least ADMISSION_WEDGE_RECOVERY_SECONDS /
#: ADMISSION_WEDGE_POLL_SECONDS - 1 = 5 separate polls. "Default
#: conservative" here means biased toward NOT acting rather than toward
#: acting fast; a wrong choice in this direction costs a slower recovery,
#: never a spurious one. Override with SGLANG_ADMISSION_WEDGE_RECOVERY_
#: SECONDS if a future specimen shows this is wrong in either direction.
ADMISSION_WEDGE_RECOVERY_SECONDS: float = 3.0 * ADMISSION_WEDGE_SECONDS


def _admission_wedge_recovery_threshold() -> float:
    """The configured recovery threshold, env-overridable, default conservative.

    A non-positive override (including the env var's own unset sentinel, -1)
    falls back to the documented default rather than being taken literally --
    a recovery action that fires on EVERY poll the moment the report itself
    does is not "conservative" under any reading of that word, and a 0 or
    negative override is far more likely to be a misconfiguration than a
    deliberate request for that behaviour.
    """
    override = envs.SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS.get()
    if override is not None and override > 0:
        return float(override)
    return ADMISSION_WEDGE_RECOVERY_SECONDS


def check_admission_wedge_once(
    scheduler: Scheduler,
    now: Optional[float] = None,
    log_on_alarm: bool = False,
) -> Tuple[bool, str]:
    """One admission-wedge check against live scheduler state (#699 wiring).

    Reads exactly the three numbers ``admission_wedge_verdict`` needs off the
    scheduler: ``len(waiting_queue)``, ``len(running_batch.reqs)``, and the
    age of ``last_first_token_progress_time`` -- the clock a request's first
    committed output token stamps (see
    ``SchedulerBatchResultProcessor.process_batch_result_prefill``), never a
    forward-pass counter. That distinction is the entire point of #699:
    forward_ct and cur_batch_for_debug both read healthy during the measured
    wedge, so a clock keyed to either would stay blind to it too.

    During scheduler startup (``is_initializing``) the progress clock has not
    been seeded by any real request yet, so polling would read a false age;
    this returns silent instead.
    """
    if scheduler.is_initializing:
        return False, "scheduler is initializing: admission-wedge check skipped"

    now = now if now is not None else time.perf_counter()
    queued = len(scheduler.waiting_queue)
    running = len(scheduler.running_batch.reqs)
    age = now - scheduler.last_first_token_progress_time
    # #739: absent on an older scheduler -> None -> the pre-#739 verdict.
    prefill_stamp = getattr(scheduler, "last_prefill_progress_time", None)
    seconds_since_prefill_progress = (
        None if prefill_stamp is None else now - prefill_stamp
    )

    alarm, verdict_detail = admission_wedge_verdict(
        queued,
        running,
        age,
        seconds_since_prefill_progress=seconds_since_prefill_progress,
    )
    detail = (
        f"queue age {age:.1f}s since last first-token progress "
        f"(perf_counter={scheduler.last_first_token_progress_time:.1f}): "
        f"{verdict_detail}"
    )
    if alarm and log_on_alarm:
        logger.error(detail)
    return alarm, detail


def create_admission_wedge_watchdog(
    scheduler: Scheduler,
    poll_interval: float = ADMISSION_WEDGE_POLL_SECONDS,
) -> threading.Thread:
    """#699: log-only admission-wedge watchdog, wired to the real progress clock.

    Deliberately NOT a ``WatchdogRaw``: that class's signal is forward_ct
    staleness, which #699 (commit 9c686ca936) proved blind to this exact
    wedge shape -- chunked prefill keeps forward_ct advancing for tens of
    seconds while zero requests reach a first token. This thread instead
    polls ``check_admission_wedge_once``, which reads queue/running counts
    and the first-token-progress clock.

    Log-only for the REPORT, still, by design: there is no SIGQUIT path here
    and this thread does not decide to restart anything. #788 adds exactly
    one bounded action on top of the report: once a wedge has stayed
    continuously alarming past ``_admission_wedge_recovery_threshold()`` (see
    that function; env-overridable, default conservative, always well above
    the report threshold itself), this loop makes ONE forced-admission
    recovery attempt for that episode and logs the attempt loudly whichever
    way it goes. "Episode" means a maximal run of consecutive alarming
    polls; the attempt flag resets the moment a poll reports no alarm, so a
    wedge that recurs later gets its own attempt.

    The action reuses ``corridor_admission.guard_prefill_admission`` --
    the SAME spill-before-alloc actuator normal prefill admission already
    calls before building a chunk -- rather than inventing a new subsystem,
    per #699/#788's own instruction to reuse what exists. It is best-effort
    and admittedly narrow: ``guard_prefill_admission`` no-ops when phase-flip
    is off (see its own docstring), and even when it runs, it only relieves
    VRAM pressure at the admission site. A wedge whose cause is elsewhere --
    the #788 PP comms deadlock this file's own ADMISSION_WEDGE docstring
    calls "broader than that path" is exactly such a case -- will not be
    moved by it, and the report keeps firing on every subsequent poll
    regardless of whether the attempt ran.

    CROSS-THREAD CAVEAT, named rather than assumed away: this watchdog
    thread already reads scheduler.waiting_queue and scheduler.running_batch
    unsynchronized (see check_admission_wedge_once), which this codebase
    already treats as an accepted read-only risk. Calling
    guard_prefill_admission from here goes further -- it can touch CUDA
    allocator state (torch.cuda.memory_reserved/allocated, an empty_cache
    provider on the relief ladder) concurrently with the scheduler's own
    forward thread. That is a real, new concurrency shape, not a hazard this
    change removes; it is accepted here because the alternative -- wiring a
    new cross-thread signal/flag into the scheduler's own loop so the
    scheduler thread runs the actuator itself -- is the "new subsystem"
    #788 was explicit about not building for this slice.
    """

    recovery_attempted_this_episode = False

    def _attempt_recovery(age: float, threshold: float) -> None:
        logger.error(
            "%s RECOVERY: wedge has been continuously alarming for %.1fs "
            "(>= %.1fs recovery threshold). Making ONE forced-admission "
            "attempt for this episode via corridor_admission."
            "guard_prefill_admission before reporting again.",
            ADMISSION_WEDGE,
            age,
            threshold,
        )
        try:
            from sglang.srt.managers.corridor_admission import (
                guard_prefill_admission,
            )

            verdict = guard_prefill_admission(scheduler, tokens=0)
        except Exception as e:  # noqa: BLE001 - recovery must not kill the watchdog
            logger.error("%s RECOVERY: forced attempt raised: %s", ADMISSION_WEDGE, e)
            return
        logger.error(
            "%s RECOVERY: forced-admission attempt returned %s (None means "
            "the gate is off or inert on this boot -- see guard_prefill_"
            "admission's own docstring; this does not mean the wedge is "
            "resolved either way, only that the attempt ran)",
            ADMISSION_WEDGE,
            verdict,
        )

    def _loop() -> None:
        nonlocal recovery_attempted_this_episode
        while True:
            time.sleep(poll_interval)
            try:
                alarm, _detail = check_admission_wedge_once(
                    scheduler, log_on_alarm=True
                )
            except Exception as e:  # noqa: BLE001 - a watchdog must not die
                logger.error(f"admission-wedge watchdog check failed: {e}")
                continue
            if not alarm:
                recovery_attempted_this_episode = False
                continue
            if recovery_attempted_this_episode:
                continue
            threshold = _admission_wedge_recovery_threshold()
            age = time.perf_counter() - scheduler.last_first_token_progress_time
            if age < threshold:
                continue
            recovery_attempted_this_episode = True
            try:
                _attempt_recovery(age, threshold)
            except Exception as e:  # noqa: BLE001 - a watchdog must not die
                logger.error(
                    "%s RECOVERY: attempt wrapper failed: %s", ADMISSION_WEDGE, e
                )

    t = threading.Thread(target=_loop, daemon=True, name="admission-wedge-watchdog")
    t.start()
    return t


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    def dump_info() -> str:
        if scheduler.is_initializing:
            return ""
        _, messages = scheduler.invariant_checker._check_all_pools(
            scheduler.pool_stats_observer.get_pool_stats(),
        )
        return (
            f"{scheduler.cur_batch_for_debug.batch_size()=}\n"
            f"{scheduler.cur_batch_for_debug.reqs=}\n" + "\n".join(messages)
        )

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: scheduler.forward_ct,
        is_active=lambda: (
            scheduler.is_initializing or scheduler.cur_batch_for_debug is not None
        ),
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
