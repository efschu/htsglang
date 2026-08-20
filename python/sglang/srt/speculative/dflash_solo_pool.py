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
    """

    def __init__(
        self,
        num_global_slots: int,
        num_draft_slots: int,
        ctx_cap: int,
        device,
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

    # -- allocator listener side (scheduler thread; enqueue only) --------
    def on_global_free(self, free_index: torch.Tensor) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        with self._pending_lock:
            # Clone: the allocator may go on to cat/mutate its tensors.
            self._pending_free.append(free_index.detach().clone())

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
            self._apply_free(idx)

    def _reset(self) -> None:
        self.map.fill_(-1)
        self.map[0] = 0
        self._free = torch.arange(
            1, self.num_draft_slots, dtype=torch.int32, device=self.device
        )
        self._free_count = self.num_draft_slots - 1
        self._slot_global.fill_(-1)
        self._slot_epoch.zero_()

    def _apply_free(self, free_index: torch.Tensor) -> None:
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
        return {
            "draft_slots": self.num_draft_slots,
            "mapped": int((self._slot_global >= 0).sum().item()),
            "free": self._free_count,
            "holes_read_total": self.holes_read_total,
            "reclaim_events": self.reclaim_events,
            "reclaimed_slots_total": self.reclaimed_slots_total,
        }
