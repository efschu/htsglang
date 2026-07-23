"""Per-session KV offload to host RAM (S1) -- FCFS spill / FIFO restore.

Feature flag: --enable-kv-session-offload (default OFF -> byte-identical).

Policy (S1):
  * First come, first served. The OLDEST running session is never touched.
  * On decode OOM (after tree eviction), the YOUNGEST running session
    (max ``kv_arrival_seq``) is spilled: its full-attention KV shard is
    backed up per rank into a pinned host pool (DCP owner rule preserved,
    no remap), its device slots are freed, and it KEEPS DECODING from host
    via a separate, eager, bs=1 "spill tick" interleaved between the
    device-batch iterations. GDN/Mamba state stays device-resident.
  * Exactly ONE spilled session at a time (S1). If more memory is needed,
    the stock retract path runs as fallback.
  * FIFO restore with hysteresis: when enough device KV is free (stable
    over N scheduler iterations, margin on top), the spilled session's
    shard is copied back H2D into freshly allocated slots (owner-matched
    under the weighted DCP rule) and the session rejoins the device batch.

Slot-space encoding while spilled ("sentinels"): the request's
``req_to_token`` row is rewritten to
    sentinel(p) = HB + p * S + res(p)
where HB is a page-aligned logical base strictly above every allocator
slot id, S is the DCP token-split factor (``cp_token_split_factor``;
1 for plain TP) and res(p) is the token's OWNER RESIDUE:
  * pre-spill tokens keep their original ``loc % S`` (weighted rule:
    ownership is a function of the allocated slot, so the residue must be
    preserved for the per-rank host shards to stay rank-local), and
  * tokens generated while spilled get ``p % S``.
Because HB % S == 0, ``sentinel % S == res``: every rank can re-derive the
full ownership vector from its (replicated) req_to_token row alone. The
i-th owned token of a rank (in position order) lives at HOST ROW i of that
rank's session host pool -- no mapping table needed for a whole-session
spill.

Rank-uniformity (NCCL-critical): every spill / tick / restore decision is
derived exclusively from replicated scheduler state (allocator free list,
req_to_token rows, arrival counters, scheduler iteration counters). No new
collectives are added anywhere; the per-layer DCP collective sequence of a
spill-tick forward is identical to a device decode step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

# Per-process singleton (one scheduler per TP-rank process). Registered by
# KVSessionOffloadManager.__init__; consumed by ScheduleBatch's spill-tick
# decode allocation without a scheduler back-reference.
_MANAGER: Optional["KVSessionOffloadManager"] = None


def get_kv_session_offload_manager() -> "KVSessionOffloadManager":
    assert _MANAGER is not None, (
        "kv-session-offload manager not initialized in this process"
    )
    return _MANAGER


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without GPU / scheduler)
# ---------------------------------------------------------------------------


def sentinel_base(alloc_max_slot: int, split_factor: int) -> int:
    """First multiple of ``split_factor`` strictly greater than every
    allocator slot id. The paged allocator hands out slot ids in
    ``[1, size]`` (page 0 is the padded dummy), so ``alloc_max_slot`` must
    be the allocator's ``size``."""
    s = max(1, int(split_factor))
    return ((int(alloc_max_slot) // s) + 1) * s


def make_sentinels(
    host_base: int, split_factor: int, residues: torch.Tensor
) -> torch.Tensor:
    """Sentinel slot ids for positions ``0..len(residues)-1`` with the given
    owner residues. int64; caller casts for the req_to_token write."""
    n = residues.numel()
    p = torch.arange(n, dtype=torch.int64, device=residues.device)
    return host_base + p * int(split_factor) + residues.to(torch.int64)


def new_token_residue(position: int, split_factor: int) -> int:
    """Owner residue assigned to a token generated WHILE spilled."""
    return int(position) % max(1, int(split_factor))


def owned_counts_weighted(
    residues: torch.Tensor, prefix: List[int]
) -> List[int]:
    """Per-rank owned-token counts under the weighted owner rule.

    ``prefix`` is ``cp_token_prefix``: rank r owns residues in
    ``[prefix[r], prefix[r+1])``. One D2H sync (bincount)."""
    S = prefix[-1]
    hist = torch.bincount(residues.to(torch.int64), minlength=S).cpu()
    return [int(hist[prefix[r] : prefix[r + 1]].sum()) for r in range(len(prefix) - 1)]


def owned_counts_even(seq_len: int, dcp_size: int) -> List[int]:
    """Per-rank owned counts under the even-modulo (position) owner rule."""
    return [
        seq_len // dcp_size + (1 if (seq_len % dcp_size) > r else 0)
        for r in range(dcp_size)
    ]


def num_blocks_rank_uniform(counts: List[int], block_size: int) -> int:
    """Rank-uniform streamed-block count: every rank iterates the MAX over
    all ranks' ceil(owned/B); empty trailing blocks are skipped locally
    without any collective, so the loop count stays identical everywhere."""
    b = max(1, int(block_size))
    return max(1, max((c + b - 1) // b for c in counts))


def compact_weighted(loc: torch.Tensor, S: int, lo: int, hi: int) -> Tuple[
    torch.Tensor, torch.Tensor
]:
    """(owned_mask, compact_slots) for GLOBAL slot ids under the weighted
    owner rule -- the exact inverse of the ``_dcp_masked_write`` packing."""
    loc64 = loc.to(torch.int64)
    off = loc64 % S
    owned = (off >= lo) & (off < hi)
    compact = (loc64 // S) * (hi - lo) + (off - lo)
    return owned, compact


def owned_device_indices(
    row: torch.Tensor,
    *,
    mode: str,
    S: int,
    lo: int,
    hi: int,
    dcp_size: int,
    dcp_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This rank's owned tokens of a req_to_token row (position order).

    Returns (owned_positions_mask, compact_device_indices_of_owned).
    ``mode`` in {"weighted", "even", "plain"}:
      * weighted: ownership/compaction from the slot id (loc % S rule).
      * even:     ownership from POSITION (p % dcp_size == rank), compact
                  slot = loc // dcp_size (the classic modulo layout).
      * plain:    single owner, compact slot = loc (dcp_size == 1).
    """
    L = row.numel()
    if mode == "weighted":
        owned, compact = compact_weighted(row, S, lo, hi)
        return owned, compact[owned].contiguous()
    if mode == "even":
        pos = torch.arange(L, dtype=torch.int64, device=row.device)
        owned = pos % dcp_size == dcp_rank
        compact = row.to(torch.int64) // dcp_size
        return owned, compact[owned].contiguous()
    assert mode == "plain"
    owned = torch.ones(L, dtype=torch.bool, device=row.device)
    return owned, row.to(torch.int64).contiguous()


def assign_owner_matched_slots(
    residues: torch.Tensor,
    prefix: List[int],
    class_slots: List[torch.Tensor],
) -> torch.Tensor:
    """Restore assignment (weighted rule): token p (owner residue
    ``residues[p]``, owner rank r) gets the next unused slot from
    ``class_slots[r]`` in position order. Returns the per-position slot id
    tensor (int64). Pure + deterministic: identical on every rank."""
    n = residues.numel()
    out = torch.empty(n, dtype=torch.int64, device=residues.device)
    res64 = residues.to(torch.int64)
    for r in range(len(prefix) - 1):
        mask = (res64 >= prefix[r]) & (res64 < prefix[r + 1])
        cnt = int(mask.sum().item())
        slots = class_slots[r]
        assert slots.numel() == cnt, (
            f"owner-matched restore: rank {r} got {slots.numel()} slots for "
            f"{cnt} owned tokens"
        )
        if cnt:
            out[mask] = slots.to(out.device)
    return out


def session_priority_key(req) -> Tuple[int, int]:
    """Pure spill-protection ordering (HIGHER key = MORE protected):

        (is_fast_lane, -arrival_seq)

    * Fast-lane requests rank above EVERY normal request (they are never
      spill victims; user decision: fast beats FCFS).
    * Within a class, the OLDER request (smaller arrival_seq) is more
      protected -- so two fast-lane requests order FCFS among themselves,
      exactly like two normal ones.
    The spill victim is always the LEAST protected candidate."""
    fast = 1 if getattr(req, "is_fast_lane", False) else 0
    seq = getattr(req, "kv_arrival_seq", -1)
    if seq is None:
        seq = -1
    return (fast, -seq)


def select_spill_victim(
    reqs,
    sizes: Optional[List[int]] = None,
    need: int = 0,
    fast_pressure: bool = False,
) -> Optional[int]:
    """Spill victim index, or None.

    Candidates are the NORMAL (non-fast-lane) requests only, ordered by
    ``session_priority_key`` (youngest = least protected).

    fast_pressure=False (plain decode-OOM among normal sessions): the
    OLDEST running normal session is untouchable (removed from the
    candidate set entirely).
    fast_pressure=True (a fast-lane request needs device space): fast
    beats FCFS -- any normal session may be victimized, INCLUDING the
    oldest. Fast-lane requests are never victims either way.

    MINIMAL EVICTION (S1 granularity = whole sessions): when ``sizes``
    (spillable tokens per request) and ``need`` (shortfall) are given, the
    victim is the YOUNGEST candidate whose size covers the need -- if the
    youngest alone would NOT cover it but an older one alone would, the
    older one is picked instead of spilling several. If no single
    candidate suffices, fall back to the strict-FCFS youngest (documented
    trade-off: default stays young-first as specified; the remainder is
    handled by the stock retraction fallback / an S2 multi-eviction).
    """
    candidates = [
        i for i in range(len(reqs)) if not getattr(reqs[i], "is_fast_lane", False)
    ]
    if not candidates:
        return None
    if not fast_pressure:
        # Oldest normal is untouchable: drop the most-protected candidate.
        oldest = max(candidates, key=lambda i: session_priority_key(reqs[i]))
        candidates = [i for i in candidates if i != oldest]
        if not candidates:
            return None
    by_youth = sorted(candidates, key=lambda i: session_priority_key(reqs[i]))
    if sizes is not None and need > 0:
        for i in by_youth:
            if sizes[i] >= need:
                return i
    return by_youth[0]


class RestoreHysteresis:
    """Restore fires only after the memory condition has held for
    ``steps`` consecutive checks (anti-flutter)."""

    def __init__(self, steps: int):
        self.steps = max(1, int(steps))
        self._streak = 0

    def update(self, ok: bool) -> bool:
        self._streak = self._streak + 1 if ok else 0
        return self._streak >= self.steps

    def reset(self):
        self._streak = 0


# ---------------------------------------------------------------------------
# Scheduler-side manager
# ---------------------------------------------------------------------------


class KVSessionOffloadManager:
    """Owns the S1 spill/restore state machine inside one scheduler process.

    All decisions are functions of replicated scheduler state; the manager
    performs only rank-local GPU<->host copies (no collectives).
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        sa = scheduler.server_args
        self.block_size = int(sa.kv_session_offload_block_size)
        self.tick_interval = max(1, int(sa.kv_session_offload_tick_interval))
        self.restore_margin_tokens = int(sa.kv_session_offload_restore_margin_tokens)
        self.hysteresis = RestoreHysteresis(
            sa.kv_session_offload_restore_hysteresis_steps
        )

        mr = scheduler.tp_worker.model_runner
        self.model_runner = mr
        # Hybrid GDN models wrap the full-attention backend.
        backend = getattr(mr.attn_backend, "full_attn_backend", mr.attn_backend)
        assert getattr(backend, "_sess_enabled", False), (
            "kv-session-offload: the attention backend was not wired "
            "(flashinfer backend with --enable-kv-session-offload required)"
        )
        # Geometry mirrors the backend's (single source: the backend derives
        # it from the installed DCP vector at init).
        self.mode = backend._sess_mode  # "weighted" | "even" | "plain"
        self.S = backend._sess_S
        self.cp_prefix = list(backend._sess_prefix)
        self.dcp_size = backend.dcp_size if self.mode != "plain" else 1
        self.dcp_rank = backend.dcp_rank if self.mode != "plain" else 0
        self.lo = self.cp_prefix[self.dcp_rank]
        self.hi = self.cp_prefix[self.dcp_rank + 1]
        self.host_base = backend._sess_host_base
        self.host_pool = backend._sess_host_pool
        self.full_pool = backend._sess_full_pool

        self.req_to_token_pool = scheduler.req_to_token_pool
        self.allocator = scheduler.token_to_kv_pool_allocator
        self.tree_cache = scheduler.tree_cache

        # S1 state: at most one spilled session.
        self.spilled_req: Optional["Req"] = None
        self.spilled_batch: Optional["ScheduleBatch"] = None
        self._iter_ct = 0  # incremented once per pre_schedule call
        self._spill_iter = -1
        self._last_tick_iter = -(1 << 30)
        self._fast_queue_logged_rid = None
        self._fast_lane_enabled = bool(getattr(sa, "enable_fast_lane", False))

        global _MANAGER
        _MANAGER = self

        logger.info(
            "kv-session-offload (S1) armed: mode=%s S=%d prefix=%s rank=%d "
            "block=%d tick_interval=%d restore_margin=%d hysteresis=%d "
            "host_pool=%d tokens/rank",
            self.mode,
            self.S,
            self.cp_prefix,
            self.dcp_rank,
            self.block_size,
            self.tick_interval,
            self.restore_margin_tokens,
            self.hysteresis.steps,
            self.host_pool.size,
        )

    # -- helpers ----------------------------------------------------------

    def _log(self, msg, *args):
        # One line per event per TP-rank process would triple the log; keep
        # rank 0 at info, others at debug (the decisions are replicated).
        if getattr(self.scheduler, "tp_rank", 0) == 0:
            logger.info(msg, *args)
        else:
            logger.debug(msg, *args)

    def _wait_forward_stream(self):
        """Order our schedule-stream copies after any in-flight forward
        (overlap mode). Event-based; no host sync."""
        fs = getattr(self.scheduler, "forward_stream", None)
        if fs is not None:
            torch.cuda.current_stream().wait_stream(fs)

    def has_spilled(self) -> bool:
        return self.spilled_req is not None

    # -- spill ------------------------------------------------------------

    def _fast_lane_pressure(self, batch_reqs) -> bool:
        """True when a fast-lane request is competing for device space:
        one is decoding in the batch, or one waits for admission. All
        inputs are replicated scheduler state -> rank-uniform."""
        if not self._fast_lane_enabled:
            return False
        if any(getattr(r, "is_fast_lane", False) for r in batch_reqs):
            return True
        return any(
            getattr(r, "is_fast_lane", False)
            for r in getattr(self.scheduler, "waiting_queue", ())
        )

    def try_spill(self, batch: "ScheduleBatch", fast_pressure=None) -> bool:
        """Spill the least-protected running session out of ``batch``
        (youngest NORMAL; fast-lane requests are never victims; under
        fast-lane pressure even the oldest normal may lose its device
        residency). Called from update_running_batch when check_decode_mem
        failed (after tree eviction) and from the fast-lane admission
        trigger. Returns True when a session was spilled.

        S1 limit: ONE spill slot. If a fast-lane request needs MORE room
        than one eviction frees, it stays cleanly queued (multi-eviction is
        an S2 extension) -- never an inconsistent partial spill."""
        if self.spilled_req is not None:
            return False
        if fast_pressure is None:
            fast_pressure = self._fast_lane_pressure(batch.reqs)
        # Shortfall X (tokens) + per-session spillable sizes -> minimal
        # single-session eviction (youngest sufficient; see
        # select_spill_victim). Only req-exclusive slots count as freed
        # (the shared radix prefix stays tree-owned, merely evictable).
        need = max(
            0,
            batch.new_tokens_required_next_decode()
            - self.allocator.available_size(),
        )
        sizes = [
            max(
                0,
                int(batch.seq_lens_cpu[i].item())
                - int(batch.reqs[i].cache_protected_len or 0),
            )
            for i in range(len(batch.reqs))
        ]
        idx = select_spill_victim(
            batch.reqs, sizes=sizes, need=need, fast_pressure=fast_pressure
        )
        if idx is None:
            return False
        req = batch.reqs[idx]
        if req.finished() or getattr(req, "to_finish", None) is not None:
            return False
        spill_margin = sizes[idx] - need  # over-eviction metric (S1b -> ~0)

        L = int(batch.seq_lens_cpu[idx].item())
        if L != req.kv_committed_len or L != req.kv_allocated_len:
            # Never spill a request whose slot bookkeeping we do not fully
            # understand (e.g. overallocation) -- stock retraction handles it.
            logger.warning(
                "kv-session-offload: skip spill of rid=%s (seq_lens %d, "
                "committed %d, allocated %d); falling back to retraction.",
                req.rid,
                L,
                req.kv_committed_len,
                req.kv_allocated_len,
            )
            return False
        row = self.req_to_token_pool.req_to_token[req.req_pool_idx, :L]
        owned_mask, dev_idx = owned_device_indices(
            row,
            mode=self.mode,
            S=self.S,
            lo=self.lo,
            hi=self.hi,
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
        )
        n_own = int(dev_idx.numel())
        if n_own > self.host_pool.size:
            logger.warning(
                "kv-session-offload: session rid=%s needs %d host rows > "
                "pool %d; falling back to stock retraction.",
                req.rid,
                n_own,
                self.host_pool.size,
            )
            return False

        # Owner residues BEFORE the row is overwritten. Weighted rule:
        # ownership is a function of the slot id -> preserve loc % S.
        # Even/plain rules: ownership is positional -> p % S keeps the
        # sentinel encoding uniform (the residue field is not consulted).
        if self.mode == "weighted":
            residues = (row.to(torch.int64) % self.S).contiguous()
        else:
            residues = (
                torch.arange(L, dtype=torch.int64, device=row.device) % self.S
            )

        # 1. Order after any in-flight forward that still writes this
        #    session's last KV row (overlap mode), then D2H-backup the whole
        #    owned shard (all full-attention layers) into host rows [0, n).
        self._wait_forward_stream()
        if n_own > 0:
            host_ids = torch.arange(n_own, dtype=torch.int64, device=row.device)
            self.host_pool.backup_from_device_all_layer(
                self.full_pool, host_ids, dev_idx, io_backend="kernel"
            )

        # 2. Release device references exactly like a retract, EXCEPT that
        #    the request keeps its req_to_token row and its Mamba/GDN state:
        #    free the req-exclusive slots, unlock the shared radix prefix
        #    (the tree keeps those slots; eviction reclaims them lazily --
        #    our backup above already covers them).
        protected = int(req.cache_protected_len or 0)
        if protected < L:
            self.allocator.free(row[protected:L].to(torch.int64))
        if req.last_node is not None:
            self.tree_cache.dec_lock_ref(req.last_node)
        req.last_node = getattr(self.tree_cache, "root_node", None)
        req.prefix_indices = torch.empty(0, dtype=torch.int64, device=row.device)
        req.cache_protected_len = 0
        req.num_matched_prefix_tokens = 0

        # 3. Rewrite the row with sentinels (residues preserved).
        sent = make_sentinels(self.host_base, self.S, residues)
        assert int(sent[-1].item()) < (1 << 31) - self.S, (
            "kv-session-offload: sentinel overflow (int32 req_to_token)"
        )
        self.req_to_token_pool.req_to_token[req.req_pool_idx, :L] = sent.to(
            torch.int32
        )

        req.kv_spill_state = "host"
        self.spilled_req = req
        self.spilled_batch = None  # built lazily at the first tick
        self._spill_iter = self._iter_ct
        self.hysteresis.reset()

        keep = [i for i in range(len(batch.reqs)) if i != idx]
        batch.filter_batch(keep_indices=keep)
        batch.batch_is_full = False

        self._log(
            "kv-session-offload SPILL: rid=%s arrival_seq=%s L=%d owned=%d "
            "(rank %d) protected_prefix=%d -> host; need=%d freed=%d "
            "over-eviction margin=%d tokens; device batch bs=%d",
            req.rid,
            getattr(req, "kv_arrival_seq", None),
            L,
            n_own,
            self.dcp_rank,
            protected,
            need,
            L - protected,
            spill_margin,
            len(batch.reqs),
        )
        return True

    # -- spill tick -------------------------------------------------------

    def _build_spill_batch(self) -> "ScheduleBatch":
        """Decode ScheduleBatch for the spilled session (persistent across
        ticks, exactly like running_batch). Mirrors the hisparse
        staging->decode builder: the last sampled token is stashed into the
        future map so resolve_forward_inputs picks it up."""
        from sglang.srt.managers.overlap_utils import RelayPayload
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

        sch = self.scheduler
        req = self.spilled_req
        device = sch.device

        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=sch.req_to_token_pool,
            token_to_kv_pool_allocator=sch.token_to_kv_pool_allocator,
            tree_cache=sch.tree_cache,
            model_config=sch.model_config,
            enable_overlap=sch.enable_overlap,
            spec_algorithm=sch.spec_algorithm,
        )
        batch.kv_session_spill_tick = True
        # mrope models index batch.multimodal_inputs per request in
        # ForwardBatch.init_new (None entries take the plain-text path).
        batch.multimodal_inputs = [getattr(req, "multimodal_inputs", None)]
        batch.req_pool_indices = torch.tensor(
            [req.req_pool_idx], dtype=torch.int64, device=device
        )
        batch.req_pool_indices_cpu = torch.tensor(
            [req.req_pool_idx], dtype=torch.int64
        )
        seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        assert seq_len == req.kv_committed_len, (
            f"kv-session-offload tick build: seq_len {seq_len} != committed "
            f"{req.kv_committed_len} (rid={req.rid})"
        )
        batch.seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(
            [seq_len], dtype=torch.int32, device=device
        )
        batch.seq_lens_sum = seq_len
        last_tokens = torch.tensor(
            [req.output_ids[-1]], dtype=torch.int64, device=device
        )
        sch.future_map.stash(
            batch.req_pool_indices, RelayPayload(bonus_tokens=last_tokens)
        )
        batch.input_ids = None
        if batch.return_logprob:
            batch.top_logprobs_nums = [req.logprob.top_logprobs_num]
            batch.token_ids_logprobs = [list(req.origin_input_ids)]
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, sch.model_config.vocab_size
        )
        return batch

    def spill_decode_alloc(self, batch: "ScheduleBatch") -> torch.Tensor:
        """prepare_for_decode replacement for the spill tick: no device
        allocation -- assign the next sentinel slot and write the
        req_to_token row. Counter updates stay in prepare_for_decode."""
        assert len(batch.reqs) == 1 and batch.kv_session_spill_tick
        req = batch.reqs[0]
        p = int(batch.seq_lens_cpu[0].item())
        res = new_token_residue(p, self.S)
        sent = self.host_base + p * self.S + res
        self.req_to_token_pool.req_to_token[req.req_pool_idx, p] = sent
        return torch.tensor([sent], dtype=torch.int64, device=batch.seq_lens.device)

    # -- scheduling hooks ---------------------------------------------------

    def pre_schedule(self, running_batch, last_batch):
        """Early hook in get_next_batch_to_run (before prefill/decode batch
        selection): cleanup of a finished spilled session, fast-lane
        admission pressure, and FIFO restore with hysteresis. Returns the
        (possibly merged) running_batch."""
        self._iter_ct += 1

        if self.spilled_req is None:
            self._maybe_spill_for_fast_lane(running_batch)
            return running_batch

        # Finished while spilled? (Release ran in the result processor.)
        if self.spilled_req.finished():
            if self.spilled_batch is not None:
                self.spilled_batch.filter_batch()
            self._clear_spill_state("finished on host")
            return running_batch

        return self._maybe_restore_flow(running_batch, last_batch)

    def _maybe_spill_for_fast_lane(self, running_batch):
        """Fast-lane admission trigger: a WAITING fast-lane request that
        does not fit into (free + tree-evictable) KV evicts the youngest
        running normal session -- under fast pressure even the oldest
        (select_spill_victim fast_pressure=True). S1 limit: one spill slot;
        if a single eviction is not enough, the fast request stays cleanly
        queued (multi-eviction is S2)."""
        sch = self.scheduler
        # Zero-overhead invariant: without --enable-fast-lane no request can
        # be fast-lane -> skip the queue scan entirely.
        if not self._fast_lane_enabled:
            return
        if running_batch is None or running_batch.is_empty():
            return
        fast_waiting = [
            r
            for r in getattr(sch, "waiting_queue", ())
            if getattr(r, "is_fast_lane", False) and not r.finished()
        ]
        if not fast_waiting:
            return
        # FCFS among fast-lane requests.
        fr = min(
            fast_waiting, key=lambda r: getattr(r, "kv_arrival_seq", 0) or 0
        )
        ratio = getattr(
            getattr(sch, "new_token_ratio_tracker", None), "current", 1.0
        )
        max_new = getattr(fr.sampling_params, "max_new_tokens", 0) or 0
        need = len(fr.origin_input_ids) + int(max_new * ratio) + 1
        have = self.allocator.available_size() + self._tree_evictable_size()
        if have >= need:
            return  # normal admission will take it
        if self.try_spill(running_batch, fast_pressure=True):
            self._log(
                "kv-session-offload: spilled for FAST-LANE request rid=%s "
                "(need %d tokens, had %d)",
                fr.rid,
                need,
                have,
            )
        elif self._fast_queue_logged_rid != fr.rid:
            self._fast_queue_logged_rid = fr.rid
            self._log(
                "kv-session-offload (S1): fast-lane request rid=%s needs %d "
                "tokens, %d available and the single spill slot cannot free "
                "enough -- request stays queued (multi-eviction is an S2 "
                "extension).",
                fr.rid,
                need,
                have,
            )

    def _tree_evictable_size(self) -> int:
        tc = self.tree_cache
        for name in ("full_evictable_size", "evictable_size"):
            fn = getattr(tc, name, None)
            if fn is None:
                continue
            try:
                v = fn()
                if isinstance(v, int):
                    return v
            except NotImplementedError:
                continue
        return 0

    def _maybe_restore_flow(self, running_batch, last_batch):
        # While a fast-lane request is waiting for admission, the freed
        # device space belongs to IT (fast beats FCFS): restoring now would
        # only re-trigger the fast-pressure spill (spill<->restore thrash,
        # one full D2H+H2D per cycle). Normal-session restore resumes once
        # the fast request is admitted or gone.
        if self._fast_lane_enabled and any(
            getattr(r, "is_fast_lane", False)
            for r in getattr(self.scheduler, "waiting_queue", ())
        ):
            self.hysteresis.reset()
            return running_batch

        # Restore only when the spilled session is quiescent: it was not
        # the batch launched last iteration (its result must be processed so
        # seq_lens/output_ids are settled), and memory holds.
        if self.spilled_batch is not None:
            if last_batch is self.spilled_batch:
                return running_batch
            L = int(self.spilled_batch.seq_lens_cpu[0].item())
        else:
            # Spilled but never ticked yet: restorable once the victim's
            # last device result has been processed (>= one iteration).
            if self._iter_ct <= self._spill_iter:
                return running_batch
            L = len(self.spilled_req.origin_input_ids) + len(
                self.spilled_req.output_ids
            ) - 1
        ok = self._restore_memory_ok(L)
        if not self.hysteresis.update(ok):
            return running_batch
        return self._restore(running_batch, L)

    def maybe_take_tick(self, running_batch) -> Optional["ScheduleBatch"]:
        """Late hook: decide whether THIS iteration runs the spill tick
        instead of the device decode batch. Must be called BEFORE
        update_running_batch so the device batch is not prepared and then
        dropped."""
        if self.spilled_req is None:
            return None
        if self.spilled_req.finished():
            return None
        # First tick strictly after the spill iteration (the victim's last
        # device result must be processed so output_ids is current).
        if self._iter_ct <= self._spill_iter:
            return None
        # Cadence: with device work present, leave at least tick_interval
        # device iterations between two spill ticks (interval=1 ->
        # alternate). Without device work the session ticks every iteration.
        device_has_work = running_batch is not None and not running_batch.is_empty()
        if (
            device_has_work
            and (self._iter_ct - self._last_tick_iter) <= self.tick_interval
        ):
            return None

        if self.spilled_batch is None:
            self.spilled_batch = self._build_spill_batch()
            self._log(
                "kv-session-offload: first spill tick for rid=%s (L=%d)",
                self.spilled_req.rid,
                int(self.spilled_batch.seq_lens_cpu[0].item()),
            )
        batch = self.spilled_batch
        batch.filter_batch()
        if batch.is_empty():
            self._clear_spill_state("finished on host")
            return None
        batch.prepare_for_decode()
        self._last_tick_iter = self._iter_ct
        return batch

    # -- restore ------------------------------------------------------------

    def _restore_memory_ok(self, L: int) -> bool:
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        need = L + self.restore_margin_tokens
        if self.allocator.available_size() < need:
            evict_from_tree_cache(self.tree_cache, need)
        if self.allocator.available_size() < need:
            return False
        if self.mode == "weighted":
            # Per-owner-class availability (free slots whose residue class
            # matches each rank's owned-token count).
            req = self.spilled_req
            row = self.req_to_token_pool.req_to_token[req.req_pool_idx, :L]
            residues = row.to(torch.int64) % self.S
            counts = owned_counts_weighted(residues, self.cp_prefix)
            free = self.allocator.free_pages
            res_free = free % self.S
            for r in range(len(counts)):
                lo, hi = self.cp_prefix[r], self.cp_prefix[r + 1]
                have = int(((res_free >= lo) & (res_free < hi)).sum().item())
                if have < counts[r]:
                    return False
        return True

    def _restore(self, running_batch, L: int):
        req = self.spilled_req
        row = self.req_to_token_pool.req_to_token[req.req_pool_idx, :L]
        assert int(row.min().item()) >= self.host_base, (
            "kv-session-offload restore: non-sentinel slot in a spilled row"
        )
        residues = (row.to(torch.int64) % self.S).contiguous()

        if self.mode == "weighted":
            counts = owned_counts_weighted(residues, self.cp_prefix)
            bounds = [
                (self.cp_prefix[r], self.cp_prefix[r + 1])
                for r in range(len(counts))
            ]
            class_slots = self.allocator.alloc_owner_matched_classes(
                self.S, bounds, counts
            )
            if class_slots is None:
                self.hysteresis.reset()
                return running_batch
            new_locs = assign_owner_matched_slots(
                residues, self.cp_prefix, class_slots
            )
        else:
            new_locs = self.allocator.alloc(L)
            if new_locs is None:
                self.hysteresis.reset()
                return running_batch
            new_locs = new_locs.to(torch.int64)

        owned_mask, dev_idx = owned_device_indices(
            new_locs,
            mode=self.mode,
            S=self.S,
            lo=self.lo,
            hi=self.hi,
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
        )
        n_own = int(dev_idx.numel())
        if self.mode == "weighted":
            assert n_own == counts[self.dcp_rank], (
                "kv-session-offload restore: owner-matched allocation "
                f"produced {n_own} owned slots, expected {counts[self.dcp_rank]}"
            )

        self._wait_forward_stream()
        host_ids = torch.arange(n_own, dtype=torch.int64, device=row.device)
        layer_num = getattr(self.host_pool, "layer_num", 0)
        for fl in range(layer_num):
            self.host_pool.load_to_device_per_layer(
                self.full_pool, host_ids, dev_idx, fl, io_backend="kernel"
            )
        self.req_to_token_pool.req_to_token[req.req_pool_idx, :L] = new_locs.to(
            torch.int32
        )

        req.kv_spill_state = None
        if self.spilled_batch is None:
            # Never ticked while spilled: build the decode batch now so the
            # session can be merged back like any resumed decode request.
            self.spilled_batch = self._build_spill_batch()
        batch = self.spilled_batch
        # Back on device: this batch (or its reqs inside running_batch) must
        # take the normal decode path again.
        batch.kv_session_spill_tick = False
        self._clear_spill_state("restored to device")
        self._log(
            "kv-session-offload RESTORE: rid=%s L=%d owned=%d (rank %d) "
            "H2D complete; rejoining device batch",
            req.rid,
            L,
            n_own,
            self.dcp_rank,
        )

        if running_batch.is_empty():
            return batch
        running_batch.merge_batch(batch)
        return running_batch

    # -- finish / cleanup ---------------------------------------------------

    def release_finished_spilled_req(self, req: "Req"):
        """Finish path for a request that ends WHILE spilled: its
        req_to_token row holds sentinels (never allocator slots) and it owns
        no device KV -- only the Mamba state and the req slot are freed. No
        radix insert (there is no device KV to donate)."""
        assert getattr(req, "kv_spill_state", None) == "host"
        pool = self.req_to_token_pool
        if req.mamba_pool_idx is not None and hasattr(pool, "free_mamba_cache"):
            pool.free_mamba_cache(req)
        pool.free(req)
        req.kv_spill_state = None
        self._log(
            "kv-session-offload: spilled session rid=%s finished on host; "
            "released mamba + req slot (no device KV, no radix insert)",
            req.rid,
        )
        # State is cleared in pre_schedule (spilled_req.finished()).

    def _clear_spill_state(self, why: str):
        rid = self.spilled_req.rid if self.spilled_req is not None else "?"
        self.spilled_req = None
        self.spilled_batch = None
        self.hysteresis.reset()
        logger.debug("kv-session-offload: spill slot cleared (%s, rid=%s)", why, rid)

    def inflight_batches(self):
        """For abort scanning: the spilled session is running too (also in
        the window between spill and first tick, before a batch exists)."""
        from types import SimpleNamespace

        if self.spilled_batch is not None:
            return [self.spilled_batch]
        if self.spilled_req is not None:
            return [SimpleNamespace(reqs=[self.spilled_req])]
        return []
