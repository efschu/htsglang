from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter, deque
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
from sglang.srt.managers.wedge_recovery import (
    DEFAULT_ACK_GRACE_SECONDS,
    DEFAULT_ESCALATE_AFTER,
    DEFAULT_RETRY_SECONDS,
    ESCALATION_TOKEN,
    RECOVERY_CHANNEL_ATTR,
    STATE_NOT_APPLICABLE,
    STATE_PENDING,
    STATE_UNCONSUMED,
    RecoveryOutcome,
    get_recovery_channel,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.kv_row_ownership import read_free_rows
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
    def _live_double_claimed_rows(free_reading, tree_cache) -> Optional[int]:
        """Rows claimed RIGHT NOW by both the free enumeration and the tree.

        #912 REST, and the reason this exists is that the term it re-derives is
        a SNAPSHOT while everything it is subtracted from is LIVE.

        ``double_owned`` reaches ``_check_full_pool`` as
        ``allocator.double_owned_slots``, published by the phase-flip census's
        ownership audit and, at every cutover, cleared to ``None`` by
        ``_retire_row_id_space`` (phase_flip_runtime.py:6255-6264, whose own
        comment says the cleared state makes the on-idle check "fall back to
        the pre-#912 raw comparison for the short window before the
        post-cutover census runs"). ``available`` and ``evictable``, meanwhile,
        are read at the instant of the check. So a row that becomes doubly
        claimed AFTER the last census is invisible to the correction and the
        ledger raises with ``double_owned=0`` -- a surplus with no named
        holder, which sends the reader to the wrong place.

        MEASURED, boot 2 of the 2c acceptance
        (/spinning/evidence-665-f1/boot_accept2c0827_0827_0049.log:21372, all
        three ranks, 00:57:49Z, three minutes into a deliberate idle)::

            pool memory leak detected! [full] total=432089, available=133120,
              evictable=1, protected=0, session_held=0, uncached=0,
              withheld=298969, double_owned=0

        ``available + evictable + withheld = 432090``, one over ``total``. The
        SAME boot's post-cutover census one second earlier (:21367) partitions
        the id space with nothing left over::

            size=432089 free=133120 cached=0 withheld=298969 unaccounted=0

        ``free + withheld == size`` exactly, so every id already has an owner;
        the ``evictable=1`` the checker then reads is a row the tree took on in
        the round between the two readings (the boot's own OUTTRACE names it:
        ``HEALTH_C n=1 (new) off=1 tail=[49276]``, :21340, and 49276 is below
        the 133120 cap, i.e. inside the free range). One row, two owners, and
        the census that could have named it had already been taken.

        NOT A TOLERANCE AND NOT AN EPSILON. This returns the SIZE OF AN
        ENUMERATED INTERSECTION -- the ids are known, not estimated -- and the
        caller subtracts exactly that and still raises on any residue. It is
        the same population ``double_owned`` already names, read from live
        state instead of from a snapshot a cutover invalidated.

        ``None`` when the intersection cannot be taken (a non-enumerable
        allocator, a tree that cannot flatten its values). ``None`` is not
        zero: zero would assert that nothing is doubly claimed, which is
        exactly the false statement the stale snapshot was already making.
        """
        if free_reading is None or not getattr(free_reading, "is_enumerable", False):
            return None
        flatten = getattr(tree_cache, "all_values_flatten", None)
        if not callable(flatten):
            return None
        try:
            cached = flatten()
            cached_rows = frozenset(int(v) for v in cached.tolist())
            free_rows = frozenset(int(v) for v in free_reading.rows)
        except Exception:  # noqa: BLE001 -- an unreadable set is not an empty one
            return None
        return len(free_rows & cached_rows)

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
        double_owned: int = 0,
    ) -> Tuple[bool, str]:
        """Check: available + evictable + protected + session_held + uncached
        + withheld - double_owned == total.

        ``withheld`` is capacity the #656 residency controller has DELIBERATELY
        taken out of circulation: slot ids above the KV pool's backed watermark,
        held out of the free list so that nothing is handed out over unmapped
        memory. It is not available, not cached and not held by a session, so
        without a term of its own it reads as a leak -- and it killed the boot
        that first exercised it ("total=500000, available=419745").

        It is a NAMED POSTEN for the #486 reason: anything that durably occupies
        or removes pool slots must be named in this ledger, or the next
        unexplained delta gets attributed to the wrong holder.

        ``double_owned`` is #912's posten, and it is SUBTRACTED rather than
        added because it does not occupy anything -- it is the count of rows
        the #822 ``RowOwnershipAuthority`` found claimed by MORE THAN ONE of
        the other terms at once (e.g. a row the free list still lists in
        ``available`` while the prefix tree also counts it in ``evictable``,
        or a resident request also counts it in ``session_held``). Each such
        row is a genuine ownership defect ("silent corruption, not a crash",
        kv_row_ownership.py's own EXCLUSIVITY wording) and is reported
        separately by the authority; here it is only subtracted so that a
        SURPLUS against ``total`` is not misread as an unexplained leak. Five
        specimens across two boots (2026-08-26,
        /spinning/evidence-665-f1/boot_accept0826_0826_1754.log:1861,1886,1911
        and boot_accept0826r2_0826_1748.log:2360,2385,2411) all showed
        ``available + evictable + withheld`` exceeding ``total`` by exactly
        22 -- never a deficit -- while the SAME boot's own "PHASE-FLIP POOL
        CENSUS" line balanced ``free + withheld == total`` exactly, because
        that census's ``free`` figure is the authority's deduplicated
        enumeration and the checker's ``available`` was the raw, non-
        deduplicating ``available_size()``. A deficit (rows with NO owner,
        e.g. #832/#856) is the opposite sign and stays fatal -- this term
        never turns a real leak into a pass, because a real leak's
        ``double_owned`` reading is zero.
        """
        total_accounted = (
            available
            + evictable
            + protected
            + session_held
            + uncached
            + withheld
            - double_owned
        )
        leak = total_accounted != total
        msg = (
            f"[{pool_name}] {total=}, {available=}, {evictable=}, "
            f"{protected=}, {session_held=}, {uncached=}, {withheld=}, "
            f"{double_owned=}"
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
        # #912: read free capacity the SAME way the phase-flip census and the
        # #822 authority already do, instead of re-deriving it from the raw
        # allocator method. `TokenToKVPoolAllocator.available_size()`
        # (allocator/token.py:52-54) is `len(free_pages) + len(release_pages)`,
        # a plain SUM that double-counts any row id present in both lists.
        # `read_free_rows()` (kv_row_ownership.py:743-843) answers the same
        # question with `frozenset(free_pages) | frozenset(release_pages)`
        # (kv_row_ownership.py:814), a UNION that counts such a row once. Five
        # measured specimens (2026-08-26,
        # /spinning/evidence-665-f1/boot_accept0826_0826_1754.log:1861,1886,1911
        # and boot_accept0826r2_0826_1748.log:2360,2385,2411) showed the raw
        # sum reading exactly 21 higher than the SAME boot's own "PHASE-FLIP
        # POOL CENSUS" free= field at the same withheld= value (e.g.
        # boot_accept0826_0826_1730.log:1826: free=124928 vs the crash's
        # available=124949, withheld=344053 identical on both sides) -- the
        # union-vs-sum divergence this reader now closes. `available_size()`
        # itself is left untouched: it also feeds admission/capacity math
        # (pool_stats_observer.py:245,265) where over-reporting free capacity
        # by a transient list overlap is a separate, pre-existing concern, not
        # this ticket's; only the LEAK CHECK's own comparison is re-pointed at
        # the one authority the census and #822 already share, per that
        # function's own "ONE authority, used by both consumers" rationale.
        free_reading = read_free_rows(allocator)
        full_available_size = (
            free_reading.count
            if free_reading.is_enumerable
            else ps.full_available_size
        )
        if getattr(self.server_args, "dcp_size", 1) > 1 and allocator.page_size > 1:
            # DCP stores logical tokens in widened physical pages.  Prefix cache
            # counters are logical-token based, while the allocator frees whole
            # physical pages, so round cached tokens up to physical page units.
            full_evictable_size = (
                (full_evictable_size + allocator.page_size - 1)
                // allocator.page_size
                * allocator.page_size
            )
        # #912 REST: the correction term, from LIVE state when the published
        # snapshot has nothing to say. `double_owned_slots` is `None` for the
        # whole window between a cutover's id-space retirement and the next
        # census (phase_flip_runtime.py:6255-6264) and 0 whenever the last
        # census found nothing -- and in both readings a row that became doubly
        # claimed since then is missing from the ledger. `_live_double_claimed_
        # rows` enumerates the intersection at the instant of the check.
        #
        # ONLY WHEN THE SNAPSHOT IS ABSENT OR EMPTY. A census reading that DID
        # find rows is a measurement over the whole authority (free list, tree,
        # residents, cap) and is strictly wider than the two sets available
        # here, so it keeps precedence and is never overwritten by this
        # narrower reading.
        published_double_owned = getattr(allocator, "double_owned_slots", None)
        double_owned = int(published_double_owned or 0)
        double_owned_src = "census"
        if not double_owned:
            live_double_owned = self._live_double_claimed_rows(
                free_reading, self.tree_cache
            )
            if live_double_owned:
                double_owned = int(live_double_owned)
                double_owned_src = "live"
                # #927: SAY WHICH OF THE TWO FREE READINGS IS SPEAKING.
                #
                # This term is subtracted from a ledger built on
                # `available_size()` (a COUNT) while it is itself derived from
                # `free_reading.rows` (an ENUMERATION). The type's own docstring
                # warns that "HOW MANY" and "WHICH ONES" are different questions
                # on this allocator family, and nothing compared them here.
                #
                # Measured, boot 2f repro (0649Z): the ledger balanced EXACTLY
                # -- available 125865 + protected 8192 + evictable 1111 = 135168
                # = the post-cutover free pool, + withheld 305265 = total
                # 440433, surplus ZERO -- and the run still died, because
                # subtracting a live reading of 8192 turned that exact balance
                # into an 8192-row deficit. A correction that large against a
                # ledger with nothing to correct means the two readings disagree
                # by exactly the protected count, and the log could not say so.
                #
                # Printed, not acted on. Whether the enumeration over-reports or
                # `available_size()` under-reports decides which side is the
                # defect, and that is a different fix on each side -- so this
                # names the disagreement and leaves the verdict to the reader,
                # rather than guessing and calling one of them wrong.
                try:
                    enumerated = len(free_reading.rows)
                    counted = free_reading.count
                    if counted is not None and enumerated != counted:
                        logger.warning(
                            "#927 FREE READING DISAGREES WITH ITSELF: the "
                            "enumeration holds %d row(s) while the count says "
                            "%d (delta %d); double_owned=%d is derived from the "
                            "ENUMERATION and subtracted from a ledger built on "
                            "the COUNT. available=%s evictable=%s protected=%s",
                            enumerated,
                            counted,
                            enumerated - counted,
                            double_owned,
                            full_available_size,
                            full_evictable_size,
                            protected,
                        )
                except Exception:  # noqa: BLE001 - a diagnostic may never raise
                    pass
        leak, msg = self._check_pool_invariant(
            "full",
            full_available_size,
            full_evictable_size,
            protected,
            session_held,
            total,
            uncached,
            # Slots the residency controller holds out of the free list because
            # the pages under them are unmapped (#656 item 12). Published by
            # KvRowCap in the same unit available_size() reports.
            int(getattr(allocator, "residency_withheld_slots", 0) or 0),
            # #912: rows the #822 authority's EXCLUSIVITY law found claimed by
            # more than one owner (free list + tree, free list + a resident
            # request, ...) at the last phase-flip census. Published by
            # phase_flip_runtime.py's `_census_ownership_audit`
            # (kv_row_ownership.py's own detection, not re-derived here), or --
            # when that snapshot is absent or empty -- enumerated live just
            # above. Zero whenever neither reading finds a doubly claimed row,
            # so this is a no-op on every path that does not exercise #822.
            double_owned,
        )
        # WHICH READING PAID, in the line that raises. A surplus explained by a
        # snapshot and one explained by a live intersection are different
        # findings, and the boot log has to be able to tell them apart without
        # re-deriving anything (#912 rest).
        msg = f"{msg}, double_owned_src={double_owned_src}"
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

    @staticmethod
    def _mamba_double_claimed(mamba_allocator, tree_cache) -> Optional[tuple]:
        """``(duplicate_count, duplicate_ids, free_and_cached)``, or ``None``.

        #924, and it is the MAMBA half of the term ``_check_full_pool`` has
        carried since #912. The full pool's reading comes from
        ``read_free_rows`` (a UNION over free + release) and from the #822
        authority; the mamba allocator has neither, and its own free list is a
        bare ``torch.cat`` (``allocator/mamba.py:88-91``) with no membership
        ledger, so a slot returned twice is REPRESENTABLE and inflates
        ``available_size()`` -- which is ``len(free_slots)`` -- by exactly the
        duplicate count.

        MEASURED, 2026-08-27 boot 1 of the 2d acceptance, all three ranks
        (/spinning/evidence-665-f1/boot_accept2d0827_0827_0253.log:32825)::

            [mamba] total=20, available=23, evictable=0, protected=0,
                    session_held=0, uncached=0, withheld=0, double_owned=0,
                    leaked_mamba_pages=None

        +3 on a 20-slot pool, and ``leaked_mamba_pages=None`` in the SAME line
        because the existing diagnosis builds ``set(free_slots.tolist())``
        (below) and a set collapses exactly the duplicates that caused the
        surplus. The ledger said "leak" and the diagnosis said "nothing is
        missing"; both were right and neither named the defect.

        TWO POPULATIONS, deliberately reported apart:

        * ``duplicate_count`` -- a slot listed MORE THAN ONCE in the free list.
          This is a DOUBLE FREE, not a shared claim: the allocator will hand
          the same state slot to two requests, which is a wrong answer with no
          crash. It is named, its ids are printed, and it stays FATAL below.
        * ``free_and_cached`` -- a slot in the free list that the radix tree
          also holds. That is the #912 population, one pool over, and it is
          subtracted for the #912 reason: it does not occupy anything, and
          leaving it unnamed makes a surplus read as an unexplained leak.

        ``None`` when the free list cannot be read. ``None`` is not zero.
        """
        free_slots = getattr(mamba_allocator, "free_slots", None)
        try:
            free_list = [int(v) for v in free_slots.tolist()]
        except (AttributeError, TypeError, ValueError):
            return None
        free_set = frozenset(free_list)
        duplicate_ids = sorted(
            slot for slot, seen in Counter(free_list).items() if seen > 1
        )
        duplicate_count = len(free_list) - len(free_set)
        shared = 0
        flatten = getattr(tree_cache, "all_mamba_values_flatten", None)
        if callable(flatten):
            try:
                cached = frozenset(int(v) for v in flatten().tolist())
            except Exception:  # noqa: BLE001 -- unreadable is not empty
                cached = None
            if cached is not None:
                shared = len(free_set & cached)
        return duplicate_count, duplicate_ids, shared

    def _mamba_double_owned_terms(self) -> tuple:
        """``(double_owned, source, duplicate_count, suffix)`` for the mamba
        ledger -- the #912 census-then-live precedence, one pool over."""
        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        published = getattr(mamba_allocator, "double_owned_slots", None)
        double_owned = int(published or 0)
        source = "census"
        duplicate_count = 0
        suffix = ""
        if not double_owned:
            reading = self._mamba_double_claimed(mamba_allocator, self.tree_cache)
            if reading is not None:
                duplicate_count, duplicate_ids, shared = reading
                if duplicate_count or shared:
                    double_owned = duplicate_count + shared
                    source = "live"
                    suffix = (
                        f", free_list_duplicates={duplicate_count}"
                        f", duplicate_slot_ids={duplicate_ids or None}"
                        f", free_and_cached={shared}"
                    )
        return double_owned, source, duplicate_count, suffix

    def _check_mamba_pool(self, ps: PoolStats) -> Tuple[bool, str]:
        ckpt_pool = getattr(self.req_to_token_pool, "mamba_ckpt_pool", None)
        if ckpt_pool is not None:
            return self._check_mamba_pool_with_int8(ps, ckpt_pool)
        double_owned, double_owned_src, duplicates, detail = (
            self._mamba_double_owned_terms()
        )
        leak, msg = self._check_pool_invariant(
            "mamba",
            ps.mamba_available_size,
            ps.mamba_evictable_size,
            self.tree_cache.mamba_protected_size(),
            self.pool_stats_observer.session_held_mamba_slots(),
            self.req_to_token_pool.mamba_pool.size,
            0,
            0,
            double_owned,
        )
        msg = f"{msg}, double_owned_src={double_owned_src}{detail}"
        if duplicates:
            # A DUPLICATE IN THE FREE LIST IS NOT ABSOLVED BY BALANCING THE
            # EQUATION. Subtracting it stops the reader hunting a leak that is
            # not there; it does not make the allocator sound. The next
            # ``alloc`` hands one state slot to two requests, and two requests
            # sharing a Mamba state is a wrong answer that never crashes -- the
            # #767 direction. So the check stays fatal here even when the sums
            # now close, and it says WHICH ids.
            leak = True
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
        # #924 sibling: the int8 lane's ACTIVE pool is the same
        # ``mamba_allocator`` free list, so it inherits the same double-free
        # blindness. Same term, same fatality rule.
        double_owned, double_owned_src, duplicates, detail = (
            self._mamba_double_owned_terms()
        )
        active_leak, active_msg = self._check_pool_invariant(
            "mamba-active",
            ps.mamba_available_size,
            ps.mamba_evictable_size,  # 0 in int8 mode (radix owns no active slots)
            0,
            self.pool_stats_observer.session_held_mamba_slots(),
            self.req_to_token_pool.mamba_pool.size,
            0,
            0,
            double_owned,
        )
        active_msg = f"{active_msg}, double_owned_src={double_owned_src}{detail}"
        if duplicates:
            active_leak = True
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
            f"{r} request(s) running: the box is serving, not wedged (queued {q})"
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
    # #739: SAY WHICH OF THE TWO WEDGES THIS IS.
    #
    # The predicate above is true of a dead pipeline AND of a busy one whose
    # pool is full, and reporting them as one thing mixed two populations into
    # every root hunt on 2026-08-22 (see wedge_class for both specimens). The
    # class is appended to the SAME alarm line so no future reader has to
    # correlate two logs, and the numbers it was reached on travel with it.
    #
    # The window is stamped when the alarm STARTS and cleared when it ends, so
    # `forward_delta` is "forward passes completed since this wedge began" --
    # not a rate over an arbitrary interval.
    if alarm:
        stamp = getattr(scheduler, "_wedge_class_sample", None)
        fwd_now = int(getattr(scheduler, "forward_ct", 0) or 0)
        if stamp is None:
            stamp = (now, fwd_now)
            scheduler._wedge_class_sample = stamp
        detail = f"{detail} | {_wedge_class_for(scheduler, stamp, fwd_now, now)}"
    else:
        # Cleared: the next wedge is a new window, never this one's tail.
        if getattr(scheduler, "_wedge_class_sample", None) is not None:
            scheduler._wedge_class_sample = None
    if alarm and log_on_alarm:
        logger.error(detail)
    return alarm, detail


def _wedge_class_for(scheduler, stamp, fwd_now: int, now: float) -> str:
    """#739: render this alarm's A/B class, or UNCLEAR, never raising.

    An instrument may not kill the thing it measures, so every probe here is
    defensive and any failure degrades to UNCLEAR with the reason named --
    which is itself a legitimate verdict of this classifier rather than a
    silent fallback into one of the two classes.
    """
    from sglang.srt.managers.wedge_class import CLASS_UNCLEAR, classify_wedge

    try:
        started_at, fwd_then = stamp
        delta = max(0, fwd_now - int(fwd_then))
        window_s = max(0.0, float(now) - float(started_at))
        post_state = None
        try:
            from sglang.srt.managers.wedge_recovery import get_recovery_channel

            channel = get_recovery_channel(scheduler)
            outcome = getattr(channel, "last_outcome", None) if channel else None
            post_state = getattr(outcome, "state", None)
        except Exception:  # noqa: BLE001 - no channel is "no verdict yet"
            post_state = None
        return classify_wedge(
            delta,
            post_state,
            window_s=window_s,
            usage_at_ceiling=_pool_at_ceiling(scheduler),
        ).render()
    except Exception as exc:  # noqa: BLE001
        return f"CLASS={CLASS_UNCLEAR} (classifier failed: {exc!r})"


def _pool_at_ceiling(scheduler) -> Optional[bool]:
    """Corroboration only: is a request-admitting pool at its limit?

    Deliberately NOT a discriminator. Slot saturation was already true minutes
    before the 2026-08-22 18:45 alarm began, while the server was decoding
    normally, so a classifier that keyed on it would have called that specimen
    a wedge before it was one.
    """
    try:
        pool = getattr(scheduler, "req_to_token_pool", None)
        size = getattr(pool, "size", None)
        free = getattr(pool, "free_slots", None)
        if size and free is not None:
            return len(free) == 0
    except Exception:  # noqa: BLE001
        return None
    return None


class AdmissionWedgeRecovery:
    """#800: drive the recovery attempt and CONSUME its outcome.

    #788 shipped the attempt. What it did not ship was a consumer: the result
    went to ``logger.error`` and nowhere else, so on 2026-08-22 the code
    reported its own ineffectiveness three times, correctly, and
    nothing in the system was any the wiser. A log line is not a consumer.

    This class is that consumer. It owns three things the old closure did not:

    1. **A reason, not a guess.** It reads
       ``guard_prefill_admission_explained``'s named exit instead of inferring
       a cause from a ``None`` that six different exits produce.
    2. **A thread split.** It POSTS the request and lets the scheduler thread
       run the actuator. The old path called into the CUDA allocator from the
       watchdog thread, and the specimen shows PP0's watchdog going silent
       from the moment it made that call: four recovery announcements were
       logged and only three results, the missing one PP0's, and PP0 emitted
       ZERO log lines of any kind over the remaining 88 s while PP1 and PP2
       emitted 18.
    3. **An escalation.** After ``escalate_after`` consecutive non-actuating
       outcomes it emits ``ESCALATION_TOKEN``, once, on its own line. That
       token is deliberately NOT ``ADMISSION_WEDGE``: the state it marks was
       previously invisible precisely because it was buried in 46 identically
       worded alarm lines, and an external watchdog needs something it can
       grep for that the ordinary alarm does not also match.

    The state that only exists here is ``UNCONSUMED``: the request was posted
    and the scheduler thread never reached ``process_input_requests`` to take
    it. That is a statement about the WEDGE -- the scheduler thread is not
    looping -- and it is the finding the specimen needed and could not make.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        *,
        grace_s: float = DEFAULT_ACK_GRACE_SECONDS,
        retry_s: float = DEFAULT_RETRY_SECONDS,
        escalate_after: int = DEFAULT_ESCALATE_AFTER,
        clock=time.monotonic,
    ) -> None:
        self._scheduler = scheduler
        self._grace_s = float(grace_s)
        self._retry_s = float(retry_s)
        self._escalate_after = int(escalate_after)
        self._clock = clock
        self._last_post_at = float("-inf")
        self._outstanding = False

    def step(self, alarm: bool) -> Optional[RecoveryOutcome]:
        """One watchdog poll's worth of recovery work. Returns what it learnt.

        Called on the watchdog thread. Touches no CUDA, imports nothing at
        call time, and allocates nothing on the common (no-alarm) path.
        """
        if self._outstanding:
            # An outstanding request must be settled whatever the alarm says:
            # a recovery that DID actuate has to be credited before the wedge
            # clearing resets the counters, or the one outcome worth knowing
            # about is the one thrown away.
            channel = get_recovery_channel(self._scheduler)
            outcome = self._settle(channel)
            if outcome.state != STATE_PENDING:
                self._outstanding = False
            if not alarm:
                self._outstanding = False
                channel.reset_episode()
            return outcome
        if not alarm:
            # NOTHING IS BUILT ON A HEALTHY BOX. The channel is created by the
            # first post and by nothing else, so a boot that never wedges
            # never allocates one -- and the test asserting its absence is how
            # that stays true.
            channel = getattr(self._scheduler, RECOVERY_CHANNEL_ATTR, None)
            if channel is not None:
                channel.reset_episode()
            return None
        threshold = _admission_wedge_recovery_threshold()
        age = time.perf_counter() - self._scheduler.last_first_token_progress_time
        if age < threshold:
            return None
        now = self._clock()
        if now - self._last_post_at < self._retry_s:
            return None
        channel = get_recovery_channel(self._scheduler)
        if channel is None:
            return None
        self._last_post_at = now
        seq = channel.post(now, tokens=0)
        self._outstanding = True
        logger.error(
            "%s RECOVERY: wedge has been continuously alarming for %.1fs "
            "(>= %.1fs recovery threshold). Posted corridor-relief request "
            "#%d to the SCHEDULER thread (not run here: the actuator touches "
            "CUDA allocator state and running it on this thread silenced "
            "PP0's own detector on 2026-08-22). The next poll reports whether "
            "the scheduler thread consumed it, and what the relief ladder did.",
            ADMISSION_WEDGE,
            age,
            threshold,
            seq,
        )
        return None

    def recovery_status(self) -> Optional[dict]:
        """This rank's recovery state, as a plain dict, or None if never tried.

        #800 SEAM. The outcome has to leave the process or it is a log line
        again, and a log line has no consumer -- that is the defect this whole
        class exists to close, and publishing the wedge verdict WITHOUT it
        just moves the blindness one layer out. ``None`` means no attempt has
        been settled on this rank, which a reader must not read as "fine": a
        missing measurement is not a measurement, the same rule
        ``WedgeSignal.verdict`` already follows.
        """
        channel = getattr(self._scheduler, RECOVERY_CHANNEL_ATTR, None)
        if channel is None or channel.last_outcome is None:
            return None
        outcome = channel.last_outcome
        return {
            "state": outcome.state,
            "reason": outcome.reason,
            "seq": outcome.seq,
            "waited_s": round(float(outcome.waited_s), 3),
            "consecutive_non_actuating": channel.consecutive_non_actuating,
            "escalated": channel.escalated,
        }

    def _settle(self, channel) -> RecoveryOutcome:
        outcome = channel.settle(self._clock(), self._grace_s)
        if outcome.state == STATE_PENDING:
            return outcome
        if outcome.state == STATE_UNCONSUMED:
            logger.error(
                "%s RECOVERY: request #%d was NOT CONSUMED after %.1fs "
                "(grace %.1fs). The scheduler thread has not reached "
                "process_input_requests in that window, so it is not running "
                "its loop -- this is a finding about the wedge, not about the "
                "corridor gate. A phase-flip leg cannot explain it: the grace "
                "window is set above one.",
                ADMISSION_WEDGE,
                outcome.seq,
                outcome.waited_s,
                self._grace_s,
            )
        elif outcome.actuated:
            logger.error(
                "%s RECOVERY: request #%d ACTUATED on the scheduler thread "
                "(relief ladder exit '%s' after %.1fs). The scheduler thread "
                "IS looping, and the ladder did run.",
                ADMISSION_WEDGE,
                outcome.seq,
                outcome.reason,
                outcome.waited_s,
            )
        elif outcome.state == STATE_NOT_APPLICABLE:
            # DELIBERATELY NOT logger.error, and that is the point of the
            # three-state split. The gate was there, it worked, and it
            # correctly had nothing to do -- reporting that at the same volume
            # as a broken mechanism is what produced "the gate is off or
            # inert" about a gate that had cleared four admissions an hour
            # earlier. The escalation below is what gets loud if this keeps
            # being the answer.
            logger.info(
                "%s RECOVERY: request #%d was NOT APPLICABLE -- the scheduler "
                "thread consumed it after %.1fs and the gate correctly had "
                "nothing to do (exit '%s'). The gate is present and working; "
                "this wedge is not a corridor problem.",
                ADMISSION_WEDGE,
                outcome.seq,
                outcome.waited_s,
                outcome.reason,
            )
        else:
            logger.error(
                "%s RECOVERY: request #%d was INERT -- the scheduler thread "
                "consumed it after %.1fs and the MECHANISM was missing or "
                "broke, exit '%s'. This is a defect in the recovery path "
                "itself, not a report about the wedge, and it is distinct "
                "from NOT_APPLICABLE by construction. It is not the old "
                "'returned None', which six exits produced and which was read "
                "as 'the gate is off' on a boot where the gate was armed.",
                ADMISSION_WEDGE,
                outcome.seq,
                outcome.waited_s,
                outcome.reason,
            )
        if channel.record(outcome, self._escalate_after):
            logger.error(
                "%s: %d consecutive recovery attempts changed nothing on this "
                "rank (latest: %s / '%s'). The recovery path is not working "
                "and the wedge is not clearing; this line is the distinct, "
                "greppable state an external supervisor should act on, and it "
                "is emitted ONCE per run so it cannot be mistaken for the "
                "repeating %s alarm it sits among.",
                ESCALATION_TOKEN,
                channel.consecutive_non_actuating,
                outcome.state,
                outcome.reason,
                ADMISSION_WEDGE,
            )
        return outcome


def _rank_label(scheduler: Scheduler) -> str:
    """A stable per-process name for the published verdict file (#799).

    The detector runs in EVERY scheduler process, so the published files must
    not collide -- and the label is also the only thing that tells an operator
    which rank went quiet. Ranks are read through ``scheduler.ps`` where the
    parallel state keeps them, with a pid fallback: a label that raises during
    telemetry would take down the scheduler it is describing, and a wrong-but-
    unique label is strictly better than no signal at all.
    """
    ps = getattr(scheduler, "ps", None)
    pp = getattr(ps, "pp_rank", None)
    tp = getattr(ps, "tp_rank", None)
    if pp is None and tp is None:
        return f"pid{os.getpid()}"
    return f"pp{pp if pp is not None else 'x'}-tp{tp if tp is not None else 'x'}"


def make_admission_wedge_poller(scheduler: Scheduler):
    """One watchdog poll, as a callable. The thread calls this; so do tests.

    Extracted so a test exercises the SAME callable the production thread
    runs, rather than a helper the thread happens to resemble. A test that
    calls the driver directly proves the driver works and says nothing about
    whether anything calls it -- which is the exact gap that let #788's
    recovery ship inert.
    """
    driver = AdmissionWedgeRecovery(scheduler)

    def _publish(alarm: bool, detail: str, recovery: Optional[dict]) -> None:
        """#799: carry this poll's verdict OUT of the scheduler process.

        The report above is a log line, and a log line has no consumer that
        can act. On boot 0822_0829 this detector was RIGHT 146 times across
        thirteen minutes with zero decode batches, and nothing restarted the
        lane, because the only reader was a human who was not reading.

        #800 ADDS ``recovery`` AND MOVES THE CALL. Without the field the
        published verdict says a rank is wedged and cannot say whether
        anything was tried or what came back, so a supervisor reading it is
        blind in exactly the way that produced the wrong diagnosis on
        2026-08-22. Published on EVERY poll, alarm or not, so the healthy
        verdict gives the reader's staleness check something to measure
        against.

        Best-effort by construction: ``publish_verdict`` swallows its own I/O
        errors, and this wrapper catches anything else. A telemetry sink must
        never be able to kill the scheduler it observes.
        """
        try:
            from sglang.srt.managers import wedge_status

            wedge_status.publish_verdict(
                _rank_label(scheduler), alarm, detail, recovery=recovery
            )
        except Exception as e:  # noqa: BLE001 - telemetry must not kill serving
            logger.warning("%s status publish failed: %s", ADMISSION_WEDGE, e)

    def _poll_once() -> Optional[bool]:
        """One poll: check, RECOVER, then publish. Returns the alarm verdict.

        THE ORDER IS THE #800 FIX AND IT IS THE WHOLE POINT OF THIS FUNCTION.
        #799 published BEFORE the recovery attempt, so the file it wrote could
        never contain that attempt's outcome -- every published recovery
        statement was structurally one poll stale at best and absent at worst.
        A status that cannot report whether recovery did anything is what let
        an operator read "the gate is off" off a boot where the gate was
        armed. Stepping first costs nothing: the driver's own rate limits
        decide whether an attempt is even made, and on the common path
        ``step`` is a couple of comparisons.

        Returns the alarm verdict, or ``None`` when the check itself failed.
        """
        try:
            alarm, detail = check_admission_wedge_once(scheduler, log_on_alarm=True)
        except Exception as e:  # noqa: BLE001 - a watchdog must not die
            logger.error(f"admission-wedge watchdog check failed: {e}")
            return None
        try:
            driver.step(alarm)
        except Exception as e:  # noqa: BLE001 - a watchdog must not die
            logger.error("%s RECOVERY: attempt wrapper failed: %s", ADMISSION_WEDGE, e)
        _publish(alarm, detail, driver.recovery_status())
        return bool(alarm)

    return _poll_once


def create_admission_wedge_watchdog(
    scheduler: Scheduler,
    poll_interval: float = ADMISSION_WEDGE_POLL_SECONDS,
    stop: Optional[threading.Event] = None,
) -> threading.Thread:
    """#699: log-only admission-wedge watchdog, wired to the real progress clock.

    Deliberately NOT a ``WatchdogRaw``: that class's signal is forward_ct
    staleness, which #699 (commit 9c686ca936) proved blind to this exact
    wedge shape -- chunked prefill keeps forward_ct advancing for tens of
    seconds while zero requests reach a first token. This thread instead
    polls ``check_admission_wedge_once``, which reads queue/running counts
    and the first-token-progress clock.

    Log-only for the REPORT, still, by design: there is no SIGQUIT path here
    and this thread does not decide to restart anything. On top of the report
    it drives ``AdmissionWedgeRecovery`` (see that class), which posts a
    corridor-relief request to the scheduler thread once a wedge has stayed
    continuously alarming past ``_admission_wedge_recovery_threshold()``, and
    then CONSUMES the outcome: actuated, inert-with-a-named-exit, or never
    consumed at all.

    #800 CHANGED TWO THINGS #788 GOT WRONG, both measured on the 2026-08-22
    11:22Z specimen and both recorded here so the reasoning is not lost:

    * The actuator no longer runs on THIS thread. #788's own docstring named
      the hazard -- ``guard_prefill_admission``'s relief ladder touches CUDA
      allocator state, including an ``empty_cache`` provider, concurrently
      with the scheduler's forward thread -- and accepted it rather than
      build the cross-thread signal. The specimen shows the hazard firing:
      four recovery announcements were logged and only three results, and the
      missing one is PP0's. PP0 announced at 11:23:09 and then emitted ZERO
      log lines of any kind over the remaining 88 s, while PP1 and PP2 emitted
      18 in that window. The recovery attempt silenced the only mechanism
      still reporting. Both the attempt and its caller catch ``Exception``, so
      a clean raise is excluded. The cross-thread signal is now built; it is
      ``managers/wedge_recovery.py`` and it is 3 ints wide.
    * The outcome is no longer a ``None`` with a guessed meaning. It carries
      the gate's own named exit, and after ``DEFAULT_ESCALATE_AFTER``
      consecutive non-actuating outcomes it emits ``ESCALATION_TOKEN`` on its
      own line, once, so an external supervisor has something to grep that the
      46 identically worded alarm lines do not also match.

    What has NOT changed, and must not be misread: the relief ladder still
    cannot un-wedge anything. It is a spill-before-alloc VRAM gate whose own
    docstring says *"IT SPILLS. IT NEVER REFUSES"*, and there is no
    forced-admission actuator in this codebase for it to be. What the request
    now buys is a DIAGNOSIS the old path could not make -- whether the
    scheduler thread is still running its loop at all -- and an honest report
    of what the ladder did when it is.
    """

    poll = make_admission_wedge_poller(scheduler)

    # #799: an optional stop, for callers that must be able to END this
    # thread. In serving the thread is daemon and lives for the process, so
    # nothing passes one. A TEST must: an unstoppable daemon thread outlives
    # the test that created it and keeps polling -- #799 observed it fall back
    # to the DEFAULT status directory once its test popped the env override
    # and leave a stale verdict in the real /run path, and #800 observed the
    # same thread flood every later test in the suite with ERROR lines at a
    # 10 ms cadence. Two tickets, two symptoms, one unstoppable loop.
    def _loop() -> None:
        while True:
            time.sleep(poll_interval)
            # Checked AFTER the sleep and BEFORE the poll: a stop requested
            # while this thread was sleeping must not buy one more publish
            # into a directory the caller is already tearing down.
            if stop is not None and stop.is_set():
                return
            poll()

    t = threading.Thread(target=_loop, daemon=True, name="admission-wedge-watchdog")
    t.start()
    return t


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    def pp_receive_is_overdue() -> bool:
        """#821: is this rank parked in a blocking PP receive far too long?

        THE BLINDNESS THIS CLOSES. `is_active` below used to read exactly two
        things, and a PP rank wedged in a mandatory dict receive shows neither:
        `forward_ct` cannot advance (no forward is running) and
        `cur_batch_for_debug` carries the PREVIOUS pass's value, which is None
        whenever that pass was idle -- and an idle pass is precisely what
        precedes a rank that never gets its next admission decision. Every
        assignment to `cur_batch_for_debug` on the PP path
        (scheduler_pp_mixin.py:1567 / :1799 / :1950) happens AFTER that
        receive, so there is no ordering in which the old predicate could have
        seen this state. The watchdog therefore stayed inactive through the
        one failure it was best placed to diagnose, which is why the #816
        specimen ends without a watchdog line: no SIGQUIT, no py-spy dump, no
        evidence -- just the launcher, 119.7s later.

        WHY A DURATION AND NOT A FLAG, and this is the whole care in this
        function. A healthy PP rank blocks in that receive on nearly every
        pass, and an IDLE server blocks in it almost continuously -- the ring
        is paced by those receives. A flag would therefore make `is_active`
        permanently true on a healthy idle boot, where `forward_ct` is also
        frozen by definition, and the watchdog would SIGQUIT a server whose
        only sin is having no work. The threshold is the watchdog's OWN
        timeout: this arm can only contribute once a single receive has
        outlasted the entire budget the watchdog already grants a frozen
        counter, which no legitimate pass approaches (a slow upstream forward
        is bounded by that forward, orders of magnitude below it).

        This adds no new deadline to the wire and touches no transport: it
        reads a timestamp the receive already records, and its only effect is
        to let an EXISTING backstop see an existing state.
        """
        since = getattr(scheduler, "_pp_blocked_recv_since", None)
        if since is None:
            return False
        return (time.monotonic() - since) > watchdog_timeout

    def dump_info() -> str:
        if scheduler.is_initializing:
            return ""
        _, messages = scheduler.invariant_checker._check_all_pools(
            scheduler.pool_stats_observer.get_pool_stats(),
        )
        batch = scheduler.cur_batch_for_debug
        if batch is None:
            # #821: NOT DEFENSIVE PADDING -- WITHOUT THIS THE NEW ARM IS WORSE
            # THAN NO ARM. This function used to dereference
            # `cur_batch_for_debug` unconditionally, which was safe only while
            # `is_active` could not be true with it set to None. The overdue-
            # receive arm above makes exactly that combination the common
            # firing case, and an AttributeError raised here does not merely
            # spoil the dump: `WatchdogRaw._watchdog_thread` catches it, logs
            # "watchdog thread crashed", and RETURNS -- so the thread dies, the
            # SIGQUIT is never sent, and the process loses its watchdog for
            # good. A silent wedge would have been traded for a silent wedge
            # with no watchdog left.
            since = getattr(scheduler, "_pp_blocked_recv_since", None)
            waited = "unknown" if since is None else f"{time.monotonic() - since:.1f}s"
            # #824 W5(b): NAME THE ARM. This line used to assert "a blocking
            # PP dict receive" for every firing, because _pp_recv_typed_dict
            # was the only site that set the marker. On boot_827 the two
            # ranks that actually wedged were in the request-relay chain
            # receive instead, so the one line an operator reads first named
            # the wrong channel. The marker now carries its own arm.
            arm = getattr(scheduler, "_pp_blocked_recv_arm", None) or "unnamed"
            return (
                "no cur_batch (the previous pass was idle); this rank is "
                f"parked in a blocking PP receive on arm={arm}, "
                f"waited={waited}. Arms are stamped by _pp_recv_typed_dict "
                "(#821) and by the pp_chain_receiver / request_receiver "
                "funnel (#824 W5).\n" + "\n".join(messages)
            )
        return f"{batch.batch_size()=}\n" f"{batch.reqs=}\n" + "\n".join(messages)

    def describe_arm() -> str:
        """#824 W5(a): which arm held this rank when the watchdog fired.

        boot_827 is the case this exists for. The watchdog did its job --
        fired at 09:17:58 on PP0, dumped py-spy, SIGQUIT at 09:18:03 -- and
        the trigger line still named no arm, so "PP0 was blocked in a SEND,
        not a typed-dict recv" had to be recovered from the py-spy output
        afterwards. The marker is already maintained; this only reads it.
        """
        parts = []
        if scheduler.is_initializing:
            parts.append("initializing")
        if scheduler.cur_batch_for_debug is not None:
            parts.append("forward-counter-frozen(cur_batch set)")
        since = getattr(scheduler, "_pp_blocked_recv_since", None)
        if since is not None:
            arm = getattr(scheduler, "_pp_blocked_recv_arm", None) or "unnamed"
            parts.append(f"blocked-recv[{arm}] waited={time.monotonic() - since:.1f}s")
        return ", ".join(parts)

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: scheduler.forward_ct,
        is_active=lambda: (
            scheduler.is_initializing
            or scheduler.cur_batch_for_debug is not None
            or pp_receive_is_overdue()
        ),
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
        describe_arm=describe_arm,
    )
