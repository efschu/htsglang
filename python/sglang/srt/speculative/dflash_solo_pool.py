"""Small DFLASH solo draft-KV pool (T156 task D).

PROBLEM: the solo-hosted DFLASH draft KV pool mirrors the GLOBAL KV slot
space -- every writer addresses it with the allocator's global cache
locations, so the pool must span max_total_num_tokens (~10 KiB/token for the
5-layer drafter; ~4.7 GiB at C=484k on rank 0) even though, under the
deterministic drafter policy (and under the bandit's context gate), DFLASH
only ever serves batches whose context is BELOW a known threshold.

DESIGN: keep the global cache locations as the EXTERNAL addressing scheme
(they encode radix sharing and per-request rows), but give the solo host a
small draft pool of S slots plus an explicit global->draft slot mapping:

* ``DraftKVSlotMapper`` owns ``map[global_slot] -> draft_slot`` (int32, -1 =
  unmapped; ~2 MB at C=500k), a free stack of draft slots and a reverse map.
  WRITERS translate through ``translate_write`` (allocating draft slots for
  unmapped globals); the per-round decode prep translates the request's
  prefix locations through ``translate_read`` and materializes them into the
  draft's PRIVATE req_to_token table, so the draft attention (incl. its CUDA
  graphs) runs entirely in draft-slot space. No global index ever reaches
  the small pool.
* Draft slot 0 is the reserved HOLE slot (mirroring the allocator's padded
  global slot 0): reads of unmapped globals -- possible only after an LRU
  reclaim dropped a radix-retained prefix, or when an over-threshold request
  shares a radix prefix with a sub-threshold one (its draft writes are
  skipped by design) -- resolve to slot 0 (zero K/V: attention-neutral,
  accept-rate-degrading only) and are counted + logged.
* LIFECYCLE: draft slots die when their global slots are freed. The global
  allocator notifies a registered free listener; because allocator.free runs
  on the scheduler thread while the mapper mutates on the worker thread
  (overlap scheduling), the listener only ENQUEUES the freed indices and the
  mapper drains the queue at the start of its next translate call -- all
  tensor mutation stays on one thread. ``clear`` (flush_cache, resizes)
  resets the whole mapping the same way.
* CAPACITY: S = (ctx_cap + block_size) * max_running_requests * factor
  (factor default 2.0, env SGLANG_DFLASH_SOLO_POOL_FACTOR): the first term
  bounds the LIVE sub-threshold requests, the factor is the safety margin
  for radix-retained sub-threshold prefixes, which keep their draft KV (and
  their radix reuse) as long as slots last. If retained prefixes ever
  exceed the margin, an LRU sweep (by last-use round) reclaims the oldest
  slots not touched in the current round -- logged, accept-degrading for
  later radix hits on the dropped prefixes, never wrong. A readable
  RuntimeError (never a silent OOB) fires only when even the sweep cannot
  satisfy an allocation.

ENABLEMENT (resolve_dflash_solo_pool_cap): force=policy -> the max upper
context bound over the table's DFLASH stages (next stage start, min'd with
the ctx-gate threshold; an unbounded DFLASH stage disables the feature);
force=auto -> the ctx-gate threshold (gate + eviction already guarantee
DFLASH never runs a batch at/above it; the decode-path assert enforces it
at the pool boundary). Static force=dflash / schedule / single-algorithm
servers keep the full mirror pool. Env SGLANG_DFLASH_SOLO_POOL_CAP: 'off'/
'0' disables, an integer overrides the cap.

Phase-2 option (NOT built, documented for the report): additionally
offloading the 3.3 GiB DFLASH draft WEIGHTS to host RAM with hysteresis
would cost 0.2-0.4 s per direction over PCIe and only pays off in long
phases without sub-threshold requests.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

SOLO_POOL_CAP_ENV = "SGLANG_DFLASH_SOLO_POOL_CAP"
SOLO_POOL_FACTOR_ENV = "SGLANG_DFLASH_SOLO_POOL_FACTOR"
DEFAULT_SOLO_POOL_FACTOR = 2.0
# Log holes at most every N occurrences after the first (accept-degrading,
# not fatal; a flood would drown the log).
_HOLE_LOG_EVERY = 1000


def resolve_dflash_solo_pool_cap(server_args) -> Tuple[Optional[int], str]:
    """(ctx cap in tokens or None=disabled, human-readable source)."""
    raw = os.environ.get(SOLO_POOL_CAP_ENV)
    if raw is not None:
        val = raw.strip().lower()
        if val in ("off", "0", "false", "disable"):
            return None, f"{SOLO_POOL_CAP_ENV}={raw} (disabled)"
        try:
            cap = int(val)
        except ValueError:
            raise ValueError(
                f"{SOLO_POOL_CAP_ENV} must be 'off' or a positive integer; "
                f"got {raw!r}."
            )
        if cap < 1:
            raise ValueError(f"{SOLO_POOL_CAP_ENV} must be >= 1; got {cap}.")
        return cap, f"{SOLO_POOL_CAP_ENV}={cap} (explicit)"

    from sglang.srt.speculative.cross_algo_utils import CROSS_SHAPES_ATTR

    shapes = getattr(server_args, CROSS_SHAPES_ATTR, None)
    if shapes is None:
        return None, "not cross-algorithm (full-context DFLASH server)"
    force = shapes.get("force")
    gate = (shapes.get("ctx_gate") or {}).get("threshold")
    if force == "auto":
        if gate is None:
            return None, "force=auto with ctx gate off"
        return int(gate), f"force=auto ctx-gate threshold {int(gate)}"
    if force == "policy":
        table = shapes.get("policy_table") or []
        bounds: List[Optional[int]] = []
        for i, (_start, (fam, _val)) in enumerate(table):
            if fam != "dflash":
                continue
            end = table[i + 1][0] if i + 1 < len(table) else None
            if end is None:
                end = gate  # the gate still fences an unbounded stage
            elif gate is not None:
                end = min(int(end), int(gate))
            bounds.append(None if end is None else int(end))
        if not bounds:
            # No DFLASH stage at all: the rung never runs; a minimal pool
            # (cap 0 -> block-size head room only) suffices.
            return 0, "force=policy without a DFLASH stage"
        if any(b is None for b in bounds):
            return None, (
                "force=policy with an unbounded DFLASH stage (no next stage, "
                "ctx gate off)"
            )
        cap = max(bounds)
        return cap, f"force=policy DFLASH stage bound {cap}"
    return None, f"force={force!r} (DFLASH may serve any context)"


class DraftKVSlotMapper:
    """Global-KV-slot -> small-draft-pool-slot mapping (module docstring).

    THREADING: all translate/drain mutation happens on the worker thread;
    ``on_global_free``/``on_global_clear`` (scheduler thread, allocator
    listener) only enqueue under a lock.

    SYNC-FREE MODE (``sync_free=True``; the worker sets it only for the
    window pool under ``SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE=1``, default
    off). The legacy mapper reads the device on every call -- ``.item()``
    for the allocation count, boolean-mask indexing (an implicit nonzero),
    ``torch.unique`` and a scalar index_put -- and each of those makes the
    host wait for the stream to drain. On the DFLASH decode round the last
    of them sits AFTER the verify launch (the draft-KV append), so the host
    cannot run ahead of the verify and the overlap scheduler has nothing to
    overlap (census xsn421/422: dflash_solo_pool.py 253/271/294/300/309 on
    every round). In this mode the allocation count lives on the device
    (``_counters[0]``) and every translate/free is fixed-shape tensor math:
    the j-th allocating entry pops ``_free[count - n + j]`` exactly as the
    legacy pop does, frees push in the legacy (sorted, deduplicated) order,
    so the map, the reverse map and the free stack stay IDENTICAL to the
    legacy mapper's. What changes is only what the host knows: a lower
    bound on the free count (``_host_free_lb``), kept from non-blocking
    pinned snapshots minus the allocations issued since; a write whose size
    exceeds that bound -- a large prefill append, or a pool close to full --
    takes the legacy path with ONE exact read (and the legacy LRU reclaim /
    exhaustion error unchanged). Hole counts are summed on the device and
    reported from the snapshots, i.e. up to one snapshot late.

    STREAMS. Only the stream bound by ``bind_owner_stream`` (the worker's
    forward stream, bound at the start of every forward) takes the
    sync-free path. A call from any other stream (the HiCache load/write
    paths translate draft rows on their own streams) first orders itself
    after the owner's queued work, then runs the legacy body on an exact
    count and synchronizes its own stream before returning -- strictly more
    ordered than the legacy mapper ever was, and off the decode hot path.
    """

    def __init__(
        self,
        num_global_slots: int,
        num_draft_slots: int,
        ctx_cap: int,
        device,
        sync_free: bool = False,
    ):
        assert num_draft_slots >= 2, "need at least the hole slot + one slot"
        self.num_global_slots = int(num_global_slots)
        self.num_draft_slots = int(num_draft_slots)
        self.ctx_cap = int(ctx_cap)
        self.device = device
        # global slot -> draft slot; -1 unmapped. Global slot 0 is the
        # allocator's padded dummy slot and permanently maps to the hole.
        self.map = torch.full(
            (self.num_global_slots + 1,), -1, dtype=torch.int32, device=device
        )
        self.map[0] = 0
        # Free stack of draft slots 1..S-1 (slot 0 = hole, never allocated).
        self._free = torch.arange(
            1, self.num_draft_slots, dtype=torch.int32, device=device
        )
        self._free_count = self.num_draft_slots - 1
        # Reverse map + last-use round per draft slot (LRU reclaim).
        self._slot_global = torch.full(
            (self.num_draft_slots,), -1, dtype=torch.int64, device=device
        )
        self._slot_epoch = torch.zeros(
            (self.num_draft_slots,), dtype=torch.int64, device=device
        )
        self._epoch = 1
        # Cross-thread free queue (allocator listener -> worker drain).
        self._pending_lock = threading.Lock()
        self._pending_free: List[torch.Tensor] = []
        self._pending_clear = False
        # Observability.
        self.holes_read_total = 0
        self.reclaim_events = 0
        self.reclaimed_slots_total = 0
        self._holes_logged_at = 0
        # Draft rows carried across radix dedup (on_global_alias); device.
        self._alias_carried = torch.zeros(1, dtype=torch.int64, device=device)
        # Sync-free mode (class docstring). Off: nothing below is allocated
        # and every method runs its legacy body unchanged.
        self.sync_free = bool(sync_free)
        self._in_exact = False
        if self.sync_free:
            self._init_sync_free_state()

    # -- sync-free state -------------------------------------------------
    def _init_sync_free_state(self) -> None:
        S = self.num_draft_slots
        dev = self.map.device
        # Free stack with ONE extra cell at index S-1: the dump target of
        # the masked scatters (the stack itself never holds more than S-1
        # entries, so indices 0..S-2 are the only ones ever popped).
        self._dump_pos = S - 1
        self._free = torch.zeros(S, dtype=torch.int32, device=dev)
        # [0] = free count (authoritative in sync-free mode), [1] = holes read.
        self._counters = torch.zeros(2, dtype=torch.int64, device=dev)
        self._alloc_ub_total = 0
        self._holes_dev_seen = 0
        self._owner_stream = None
        self._snap_inflight = False
        self._snap_alloc_ub = 0
        self._snap_host = None
        self._snap_event = None
        self._fill_sync_free_device_state(fold_holes=False)
        if dev.type == "cuda":
            self._stream_mode = "cuda"
            try:
                self._snap_host = torch.zeros(2, dtype=torch.int64, pin_memory=True)
                self._snap_event = torch.cuda.Event()
            except Exception as e:  # noqa: BLE001 -- degrade to exact reads
                logger.warning(
                    "DFLASH window pool (sync-free): no pinned snapshot (%s); "
                    "the free-count bound is refreshed by exact reads only.",
                    e,
                )
                self._snap_host = None
                self._snap_event = None
        elif dev.type == "meta":
            # Shape-only device (tests): no data, nothing to snapshot.
            self._stream_mode = "off"
        else:
            # CPU: every op is synchronous, so a read IS the snapshot.
            self._stream_mode = "immediate"

    def _fill_sync_free_device_state(self, fold_holes: bool) -> None:
        """(Re)build the device stack and counters from HOST constants only.

        Never from device bytes the mapper held before: a phase release of an
        un-backed memory-saver region can hand them back arbitrary, and the
        legacy ``_reset`` never read them either (it allocates a fresh
        arange). The device hole counter restarts at zero; on a CPU mapper its
        unharvested part is folded in first (a free read), on CUDA at most one
        snapshot's worth of hole COUNTS is dropped -- observability only."""
        S = self.num_draft_slots
        dev = self._free.device
        if fold_holes and dev.type == "cpu":
            self._note_holes(int(self._counters[1]))
        self._free[: S - 1].copy_(torch.arange(1, S, dtype=torch.int32, device=dev))
        self._free[S - 1 :].zero_()
        self._counters[0].fill_(S - 1)
        self._counters[1].zero_()
        self._holes_dev_seen = 0
        self._host_free_lb = S - 1
        # A snapshot in flight predates this; the next one is queued on the
        # same (owner) stream behind it, so dropping its bookkeeping is safe.
        self._snap_inflight = False

    def _reset_sync_free_state(self) -> None:
        self._fill_sync_free_device_state(fold_holes=True)

    def bind_owner_stream(self) -> None:
        """Make the CURRENT stream the one that takes the sync-free path.

        Called by the worker at the start of every forward. Rebinding to a
        different stream orders the new stream after the old one's queued
        work first -- including an in-flight snapshot copy, so the next copy
        into the same pinned page still lands after it -- and drops that
        snapshot's bookkeeping."""
        if not self.sync_free or self._stream_mode != "cuda":
            return
        cur = torch.cuda.current_stream(self.map.device)
        owner = self._owner_stream
        if owner is None:
            self._owner_stream = cur
            return
        if cur == owner:
            return
        cur.wait_stream(owner)
        self._snap_inflight = False
        self._owner_stream = cur

    def _on_owner_stream(self) -> bool:
        if self._stream_mode != "cuda":
            return True
        owner = self._owner_stream
        return owner is not None and torch.cuda.current_stream(
            self.map.device
        ) == owner

    def _note_holes(self, dev_total: int) -> None:
        delta = int(dev_total) - self._holes_dev_seen
        if delta <= 0:
            return
        self._holes_dev_seen = int(dev_total)
        self.holes_read_total += delta
        if (
            self.holes_read_total - self._holes_logged_at >= _HOLE_LOG_EVERY
            or self._holes_logged_at == 0
        ):
            self._holes_logged_at = self.holes_read_total
            logger.warning(
                "DFLASH small solo pool: %d unmapped prefix slots read "
                "(total %d) -- zero-KV holes; accept rate degrades for "
                "the affected radix prefixes (raise %s if frequent). "
                "[sync-free mapper: counted on the device, reported up to "
                "one snapshot late]",
                delta,
                self.holes_read_total,
                SOLO_POOL_FACTOR_ENV,
            )

    def _poll_snapshot(self) -> None:
        """Refresh the host's free-count lower bound WITHOUT waiting.

        CUDA: harvest a completed pinned copy (``Event.query``, never a
        synchronize) and queue the next one behind the work issued so far.
        The bound is the snapshot minus every allocation upper bound issued
        after the snapshot was queued; frees issued since only make the truth
        larger, so the bound can lag but never overshoot."""
        mode = self._stream_mode
        if mode == "off":
            return
        if mode == "immediate":
            vals = self._counters.tolist()
            self._host_free_lb = int(vals[0])
            self._note_holes(int(vals[1]))
            return
        if self._snap_host is None or not self._on_owner_stream():
            return
        if self._snap_inflight:
            if not self._snap_event.query():
                return
            self._host_free_lb = int(self._snap_host[0]) - (
                self._alloc_ub_total - self._snap_alloc_ub
            )
            self._note_holes(int(self._snap_host[1]))
            self._snap_inflight = False
        self._snap_host.copy_(self._counters, non_blocking=True)
        self._snap_event.record()
        self._snap_alloc_ub = self._alloc_ub_total
        self._snap_inflight = True

    def _exact_section(self, body, *args, **kwargs):
        """Run a LEGACY body on an exact host free count (sync-free mode).

        One device read for the count; the legacy body then mutates the
        host count as it always did, and the result is written back to the
        device counter. Off the owner stream, the read is first ordered
        after the owner's queued mapper work and the call ends with a
        synchronize of its own stream, so the owner's next op (issued later
        by the host) can never overtake it."""
        cuda = self._stream_mode == "cuda"
        foreign = cuda and not self._on_owner_stream()
        cur = torch.cuda.current_stream(self.map.device) if cuda else None
        if foreign and self._owner_stream is not None:
            cur.wait_stream(self._owner_stream)
        self._free_count = int(self._counters[0].item())
        self._in_exact = True
        try:
            return body(*args, **kwargs)
        finally:
            self._in_exact = False
            self._counters[0].fill_(self._free_count)
            if foreign:
                cur.synchronize()
            self._host_free_lb = self._free_count
            # A snapshot queued before this call predates its writes. Its copy
            # is behind the read above (owner stream, or ordered by the wait),
            # so the next copy into the pinned page lands after it.
            self._snap_inflight = False

    # -- allocator listener side (scheduler thread; enqueue only) --------
    def on_global_free(self, free_index: torch.Tensor) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        with self._pending_lock:
            # Clone: the allocator may go on to cat/mutate its tensors.
            self._pending_free.append(free_index.detach().clone())

    def on_global_alias(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Radix dedup (allocator alias listener, scheduler thread; enqueue
        only): the tree keeps its slots ``dst`` and is about to free the
        request's fresh duplicates ``src``. Queued IN ORDER with the frees,
        so the drain carries the draft rows ``src`` holds over to ``dst``
        before the free of ``src`` is applied (SGLANG_DFLASH_WINDOW_POOL_
        DEDUP_CARRY). Never registered without the switch."""
        if src is None or dst is None or src.numel() == 0:
            return
        if src.numel() != dst.numel():
            return  # not an element-wise pairing; the free stays as it was
        with self._pending_lock:
            self._pending_free.append(
                ("alias", src.detach().clone(), dst.detach().clone())
            )

    def on_global_clear(self) -> None:
        with self._pending_lock:
            self._pending_free.clear()
            self._pending_clear = True

    # -- worker-thread side ----------------------------------------------
    def begin_round(self) -> None:
        self._epoch += 1

    def _drain_pending(self) -> None:
        with self._pending_lock:
            pending = self._pending_free
            clear = self._pending_clear
            self._pending_free = []
            self._pending_clear = False
        if clear:
            self._reset()
            return
        for idx in pending:
            if isinstance(idx, tuple):
                self._apply_alias(idx[1], idx[2])
            else:
                self._apply_free(idx)

    def _apply_alias(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Move the draft slot of each fresh ``src`` to its kept ``dst`` when
        ``dst`` has none: map[dst] := map[src], map[src] := -1, reverse map
        follows. Where ``dst`` already has a row (or ``src`` has none) nothing
        moves and the following free of ``src`` recycles as before. Fixed-shape
        device math, no host read (safe on the sync-free hot path); the free
        count is untouched (a slot changes owner, none is allocated)."""
        dev = self.map.device
        s = src.to(device=dev, dtype=torch.int64).reshape(-1)
        d = dst.to(device=dev, dtype=torch.int64).reshape(-1)
        ok = (s > 0) & (s <= self.num_global_slots) & (d > 0) & (d <= self.num_global_slots)
        zero = torch.zeros_like(s)
        s_safe = torch.where(ok, s, zero)
        d_safe = torch.where(ok, d, zero)
        ms = self.map[s_safe].to(torch.int64)
        md = self.map[d_safe].to(torch.int64)
        take = ok & (ms > 0) & (md < 0)
        # Non-taking entries write harmless constants: map[0] := 0 (always 0)
        # and the hole slot's reverse entry := -1 (always -1).
        self.map.index_put_(
            (torch.where(take, d_safe, zero),),
            torch.where(take, ms, zero).to(self.map.dtype),
        )
        self.map.index_put_(
            (torch.where(take, s_safe, zero),),
            torch.where(take, torch.full_like(ms, -1), zero).to(self.map.dtype),
        )
        self._slot_global.index_put_(
            (torch.where(take, ms, zero),),
            torch.where(take, d_safe, torch.full_like(d_safe, -1)),
        )
        # Observability without a host read: summed on the device, read by
        # stats() (a diagnostic sync, never on the decode path).
        self._alias_carried.add_(take.sum())

    def _reset(self) -> None:
        self.map.fill_(-1)
        self.map[0] = 0
        if self.sync_free:
            # Same state, in place (the stack keeps its dump cell).
            self._reset_sync_free_state()
        else:
            self._free = torch.arange(
                1, self.num_draft_slots, dtype=torch.int32, device=self.device
            )
        self._free_count = self.num_draft_slots - 1
        self._slot_global.fill_(-1)
        self._slot_epoch.zero_()

    def _apply_free(self, free_index: torch.Tensor) -> None:
        if self.sync_free and not self._in_exact and self._on_owner_stream():
            self._apply_free_sync_free(free_index)
            return
        self._apply_free_legacy(free_index)

    def _apply_free_sync_free(self, free_index: torch.Tensor) -> None:
        """``_apply_free`` without a host read: the same pushes, in the same
        (sorted, first-occurrence) order, as fixed-shape masked scatters."""
        if free_index.device != self.map.device:
            free_index = free_index.to(self.map.device)
        idx = free_index.to(torch.int64).reshape(-1)
        if idx.numel() == 0:
            return
        s, _ = torch.sort(idx)
        first = torch.ones_like(s, dtype=torch.bool)
        if s.numel() > 1:
            first[1:] = s[1:] != s[:-1]
        in_range = (s >= 0) & (s <= self.num_global_slots)
        zero = torch.zeros_like(s)
        s_safe = torch.where(in_range, s, zero)
        d = self.map[s_safe]
        live = (d > 0) & first & in_range  # never the hole slot
        live64 = live.to(torch.int64)
        rank = torch.cumsum(live64, 0) - 1
        n = live64.sum()
        # Non-live entries write harmless constants: the stack's dump cell,
        # map[0] (always 0) and the hole slot's reverse entry (always -1).
        pos = torch.where(
            live, self._counters[0] + rank, torch.full_like(s, self._dump_pos)
        )
        self._free.index_put_((pos,), torch.where(live, d, torch.zeros_like(d)))
        self.map.index_put_(
            (torch.where(live, s_safe, zero),),
            torch.where(live, torch.full_like(d, -1), torch.zeros_like(d)),
        )
        self._slot_global.index_put_(
            (torch.where(live, d.to(torch.int64), zero),), torch.full_like(s, -1)
        )
        self._counters[0].add_(n)

    def _apply_free_legacy(self, free_index: torch.Tensor) -> None:
        if free_index.device != self.map.device:
            free_index = free_index.to(self.map.device)
        # Dedupe defensively: a duplicated index would push the same draft
        # slot onto the free stack twice (double-alloc corruption later).
        free_index = torch.unique(free_index.to(torch.int64))
        in_range = (free_index >= 0) & (free_index <= self.num_global_slots)
        if not bool(in_range.all()):
            free_index = free_index[in_range]
            if free_index.numel() == 0:
                return
        d = self.map[free_index]
        live = d > 0  # never the hole slot
        n = int(live.sum().item())
        if n == 0:
            return
        slots = d[live]
        self.map[free_index[live]] = -1
        self._slot_global[slots.to(torch.int64)] = -1
        self._free[self._free_count : self._free_count + n] = slots
        self._free_count += n

    def translate_read(self, global_locs: torch.Tensor) -> torch.Tensor:
        """Draft slots for *global_locs*; unmapped -> hole slot 0 (counted)."""
        if self.sync_free and not self._in_exact:
            if self._on_owner_stream():
                return self._translate_read_sync_free(global_locs)
            return self._exact_section(self._translate_read_legacy, global_locs)
        return self._translate_read_legacy(global_locs)

    def _translate_read_sync_free(self, global_locs: torch.Tensor) -> torch.Tensor:
        self._drain_pending()
        if global_locs.numel() == 0:
            return global_locs.to(torch.int64)
        g = global_locs.to(torch.int64)
        d = self.map[g].to(torch.int64)
        holes = d < 0
        self._counters[1].add_(holes.sum())
        d = torch.where(holes, torch.zeros_like(d), d)
        self._slot_epoch.index_fill_(0, d.reshape(-1), self._epoch)
        self._poll_snapshot()
        return d

    def translate_read_rows(
        self, global_2d: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """``translate_read`` over a PADDED block, without compacting it.

        ``mask`` (bool, same shape) marks the real entries. Masked entries get
        exactly what ``translate_read`` of the compacted entries would give
        (unmapped -> hole slot 0, counted); unmasked ones get 0 and are not
        counted. Sync-free mode, owner stream only -- the legacy mapper
        compacts with a boolean index instead (an implicit nonzero)."""
        if not self.sync_free or not self._on_owner_stream():
            raise RuntimeError(
                "translate_read_rows is the sync-free window-pool path: it "
                "needs sync_free=True and the owner stream "
                "(bind_owner_stream() at the start of the forward)."
            )
        self._drain_pending()
        if global_2d.numel() == 0:
            return global_2d.to(torch.int64)
        g = global_2d.to(torch.int64)
        d = self.map[g].to(torch.int64)
        m = mask.to(device=d.device, dtype=torch.bool)
        holes = (d < 0) & m
        self._counters[1].add_(holes.sum())
        d = torch.where(m & (d >= 0), d, torch.zeros_like(d))
        self._slot_epoch.index_fill_(0, d.reshape(-1), self._epoch)
        self._poll_snapshot()
        return d

    def _translate_read_legacy(self, global_locs: torch.Tensor) -> torch.Tensor:
        self._drain_pending()
        if global_locs.numel() == 0:
            return global_locs.to(torch.int64)
        g = global_locs.to(torch.int64)
        d = self.map[g].to(torch.int64)
        holes = d < 0
        num_holes = int(holes.sum().item())
        if num_holes:
            self.holes_read_total += num_holes
            if (
                self.holes_read_total - self._holes_logged_at
                >= _HOLE_LOG_EVERY
                or self._holes_logged_at == 0
            ):
                self._holes_logged_at = self.holes_read_total
                logger.warning(
                    "DFLASH small solo pool: %d unmapped prefix slots read "
                    "(total %d) -- zero-KV holes; accept rate degrades for "
                    "the affected radix prefixes (raise %s if frequent).",
                    num_holes,
                    self.holes_read_total,
                    SOLO_POOL_FACTOR_ENV,
                )
            d = torch.where(holes, torch.zeros_like(d), d)
        self._slot_epoch[d] = self._epoch
        return d

    def translate_write(
        self,
        global_locs: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Draft slots for a WRITE at *global_locs*, allocating unmapped
        entries. ``valid`` (bool, same shape flat): rows outside the mask
        (e.g. beyond a request's commit length, or gated requests) get the
        hole slot and allocate nothing -- prefix-valid writers never touch
        those rows."""
        if self.sync_free and not self._in_exact:
            if self._on_owner_stream():
                return self._translate_write_sync_free(global_locs, valid)
            return self._exact_section(
                self._translate_write_legacy, global_locs, valid
            )
        return self._translate_write_legacy(global_locs, valid)

    def _translate_write_sync_free(
        self,
        global_locs: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The allocation as fixed-shape device math (class docstring).

        Needs the host to KNOW the pop cannot underflow: the entries that can
        allocate are at most ``numel`` (a host fact), so the fast path runs
        only while that fits under the snapshot bound; otherwise the legacy
        body runs on one exact read, reclaim and exhaustion error included."""
        self._drain_pending()
        if global_locs.numel() == 0:
            return global_locs.to(torch.int64)
        n_max = int(global_locs.numel())
        if n_max > self._host_free_lb:
            self._poll_snapshot()
            if n_max > self._host_free_lb:
                return self._exact_section(
                    self._translate_write_legacy, global_locs, valid
                )
        g = global_locs.to(torch.int64).reshape(-1)
        if valid is None:
            valid_mask = torch.ones_like(g, dtype=torch.bool)
        else:
            valid_mask = valid.reshape(-1).to(device=g.device, dtype=torch.bool)
        cur = self.map[g]
        need = (cur < 0) & valid_mask
        need64 = need.to(torch.int64)
        rank = torch.cumsum(need64, 0) - 1
        n = need64.sum()
        # Legacy pop: the j-th allocating entry takes _free[count - n + j].
        pos = (self._counters[0] - n + rank).clamp_(min=0)
        new_slots = self._free[pos]
        zero = torch.zeros_like(g)
        # Non-allocating entries write map[0] := 0 and the hole slot's
        # reverse entry := -1, both of which already hold those values.
        self.map.index_put_(
            (torch.where(need, g, zero),),
            torch.where(need, new_slots, torch.zeros_like(new_slots)),
        )
        self._slot_global.index_put_(
            (torch.where(need, new_slots.to(torch.int64), zero),),
            torch.where(need, g, torch.full_like(g, -1)),
        )
        self._counters[0].sub_(n)
        cur = self.map[g]
        d = torch.where(valid_mask & (cur >= 0), cur.to(torch.int64), zero)
        self._slot_epoch.index_fill_(0, d, self._epoch)
        self._alloc_ub_total += n_max
        self._host_free_lb -= n_max
        self._poll_snapshot()
        return d.view(global_locs.shape)

    def _translate_write_legacy(
        self,
        global_locs: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._drain_pending()
        if global_locs.numel() == 0:
            return global_locs.to(torch.int64)
        g = global_locs.to(torch.int64).reshape(-1)
        if valid is None:
            valid_mask = torch.ones_like(g, dtype=torch.bool)
        else:
            valid_mask = valid.reshape(-1).to(device=g.device, dtype=torch.bool)
        cur = self.map[g]
        need = (cur < 0) & valid_mask
        n = int(need.sum().item())
        if n:
            if n > self._free_count:
                self._reclaim(n - self._free_count)
            new_slots = self._free[self._free_count - n : self._free_count]
            self._free_count -= n
            fresh_g = g[need]
            self.map[fresh_g] = new_slots
            self._slot_global[new_slots.to(torch.int64)] = fresh_g
            cur = self.map[g]
        d = torch.where(
            valid_mask & (cur >= 0),
            cur.to(torch.int64),
            torch.zeros_like(g),
        )
        self._slot_epoch[d] = self._epoch
        return d.view(global_locs.shape)

    def _reclaim(self, shortfall: int) -> None:
        """LRU sweep: free the oldest mapped slots not touched this round."""
        target = max(int(shortfall), self.num_draft_slots // 8)
        mapped = self._slot_global >= 0
        mapped[0] = False
        untouched = self._slot_epoch < self._epoch
        cand = (mapped & untouched).nonzero(as_tuple=False).flatten()
        if int(cand.numel()) < shortfall:
            raise RuntimeError(
                "DFLASH small solo pool exhausted: need "
                f"{shortfall} more draft slots but only {int(cand.numel())} "
                f"reclaimable of {self.num_draft_slots} total "
                f"(ctx cap {self.ctx_cap}). The live sub-threshold working "
                "set exceeds the sizing model -- raise "
                f"{SOLO_POOL_FACTOR_ENV} (safety factor) or override "
                f"{SOLO_POOL_CAP_ENV}."
            )
        take = min(int(cand.numel()), target)
        order = torch.argsort(self._slot_epoch[cand])
        victims = cand[order[:take]]
        self.reclaim_events += 1
        self.reclaimed_slots_total += take
        # #790 sweep: this call is unconditional (no debug/trace gate) and
        # sits on the decode-time draft-slot allocation path
        # (`translate_write` -> `_reclaim`), so it runs under real load, not
        # only when a human opted into extra tracing. The previous line read
        # `self._slot_epoch[victims].max().item()` -- an `.item()` inside a
        # log argument is exactly the #790 shape: a D2H copy + stream sync
        # whose only purpose is a diagnostic string. `victims` are drawn
        # from the ASCENDING argsort's low end (the oldest, untouched
        # entries), so their epoch is bounded above by `self._epoch` (the
        # current round, already a plain host int -- see `self._epoch += 1`
        # a few lines above the caller) by construction: nothing here can be
        # "younger" than the round being evicted at. Reporting `self._epoch`
        # instead of the exact per-victim max is a tight, zero-cost stand-in
        # -- "reclaimed at round X" rather than "victims' newest round was
        # X" -- without ever reading `_slot_epoch`'s device values.
        logger.warning(
            "DFLASH small solo pool: LRU reclaim of %d draft slots "
            "(event #%d, at round %d; radix prefixes untouched since before "
            "this round lose their draft KV -- accept rate degrades on "
            "their next hit).",
            take,
            self.reclaim_events,
            self._epoch,
        )
        self.map[self._slot_global[victims]] = -1
        self._slot_global[victims] = -1
        self._free[self._free_count : self._free_count + take] = victims.to(
            torch.int32
        )
        self._free_count += take

    def stats(self) -> dict:
        free = self._free_count
        if self.sync_free:
            # The device counter is authoritative in sync-free mode; this is a
            # diagnostic read (a sync), never on the decode hot path.
            if (
                self._stream_mode == "cuda"
                and self._owner_stream is not None
                and not self._on_owner_stream()
            ):
                torch.cuda.current_stream(self.map.device).wait_stream(
                    self._owner_stream
                )
            vals = self._counters.tolist()
            free = int(vals[0])
            self._note_holes(int(vals[1]))
        return {
            "draft_slots": self.num_draft_slots,
            "mapped": int((self._slot_global >= 0).sum().item()),
            "free": free,
            "holes_read_total": self.holes_read_total,
            "reclaim_events": self.reclaim_events,
            "reclaimed_slots_total": self.reclaimed_slots_total,
            "alias_carried_total": int(self._alias_carried.sum().item()),
        }


def rebuild_window_rows_sync_free(
    *,
    mapper: DraftKVSlotMapper,
    target_req_to_token: torch.Tensor,
    draft_req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    start: torch.Tensor,
    lengths: torch.Tensor,
    max_len: int,
) -> None:
    """Window pool, sync-free: rows ``[0, lengths[b])`` of the draft's private
    req_to_token, in draft-slot space, WITHOUT a host read.

    The legacy chain (``_gather_req_to_token_segments`` -> ``translate_read``
    -> ``assign_req_to_token_pool_func``) reads ``lengths.max().item()`` and
    compacts with a boolean mask -- two host syncs per round before the draft
    forward. Here the width is the HOST bound ``max_len`` (the compact
    seq-lens mirror the backends already plan with, >= every device length),
    the block stays padded, and the write back is masked: row ``b`` gets
    exactly the legacy values in ``[0, lengths[b])`` and every other column
    keeps what it held (the verify block is written right after by the
    caller, as before).
    """
    bs = int(req_pool_indices.shape[0])
    if bs == 0 or max_len <= 0:
        return
    if max_len > int(draft_req_to_token.shape[1]):
        raise RuntimeError(
            f"DFLASH window pool: host length bound {max_len} exceeds the "
            f"draft req_to_token width {int(draft_req_to_token.shape[1])}."
        )
    device = target_req_to_token.device
    rows = req_pool_indices.to(device=device, dtype=torch.int64).unsqueeze(1)
    offsets = torch.arange(max_len, device=device, dtype=torch.int64).unsqueeze(0)
    mask = offsets < lengths.to(device=device, dtype=torch.int64).unsqueeze(1)
    pos = (start.to(device=device, dtype=torch.int64).unsqueeze(1) + offsets)
    pos = pos.masked_fill(~mask, 0)
    draft_slots = mapper.translate_read_rows(target_req_to_token[rows, pos], mask)
    rows2d = rows.expand(-1, max_len)
    cols2d = offsets.expand(bs, -1)
    held = draft_req_to_token[rows2d, cols2d]
    draft_req_to_token[rows2d, cols2d] = torch.where(
        mask, draft_slots.to(draft_req_to_token.dtype), held
    )
