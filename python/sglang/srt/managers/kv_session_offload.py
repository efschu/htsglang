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
import os
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
    host_base: int, split_factor: int, residues: torch.Tensor, start: int = 0
) -> torch.Tensor:
    """Sentinel slot ids for positions ``start..start+len(residues)-1`` with
    the given owner residues. int64; caller casts for the req_to_token
    write. ``start`` > 0 encodes a PARTIAL spill segment (S1b): only the
    positions inside the tier segment [a, b) carry sentinels."""
    n = residues.numel()
    p = torch.arange(start, start + n, dtype=torch.int64, device=residues.device)
    return host_base + p * int(split_factor) + residues.to(torch.int64)


def chunk_ceil(need: int, chunk: int) -> int:
    """Round a token shortfall up to the spill-growth chunk (amortizes the
    per-event D2H cost; the over-eviction margin is bounded by chunk-1)."""
    c = max(1, int(chunk))
    n = max(0, int(need))
    return ((n + c - 1) // c) * c


def bundle_spillable_sizes(req, seq_len: int) -> List[Tuple[str, int]]:
    """Per-resource spillable sizes of one session's RESIDENCY BUNDLE.

    Factoring seam for the future GDN-state host tier: a session's
    residency unit is the BUNDLE of (target KV shard, GDN/Mamba states,
    MTP-draft share). Victim ORDERING stays `session_priority_key` /
    `select_spill_victim` (resource-agnostic); the MECHANICS consume this
    list. In S2 the KV shard is the only element; a GDN tier adds
    ("gdn_state", ...) / ("draft_kv", ...) entries here without touching
    the ordering logic. NOTE: the S1 spec-decode rejection will later be
    lifted through exactly this bundle -- the draft share spills WITH the
    session instead of speculative decoding being forbidden.
    """
    kv = max(0, int(seq_len) - int(req.cache_protected_len or 0))
    return [("kv", kv)]


def partial_spill_plan(
    L: int, protected: int, need: int, chunk: int
) -> Tuple[int, int]:
    """Tier boundary for a PARTIAL (S1b) spill of one session.

    Only the TAIL overhang moves to host; the head ``[0, boundary)`` stays
    device-resident (fast). Returns ``(boundary, spill_count)`` where the
    spilled tail is exactly ``[boundary, L)`` (``spill_count = L - boundary``
    host tokens).

    * ``spill_count`` is the shortfall ``need`` rounded UP to ``chunk`` (the
      streamed-block granularity), so the freed tail is block-aligned and the
      over-eviction margin is bounded by ``chunk - 1`` (vs. a whole-session
      spill's margin of ``spillable - need``).
    * The tail never dips below the protected radix prefix: at most
      ``L - protected`` tokens (the req-exclusive suffix) are spillable. When
      the rounded shortfall meets or exceeds that cap the whole exclusive
      suffix spills (``boundary == protected``) -- the shared prefix ALWAYS
      stays device-resident and tree-locked, it is never backed up or freed.
    * ``need <= 0`` yields ``spill_count == 0`` (boundary == L): the caller
      asked to free nothing, so nothing moves.
    """
    max_spill = max(0, int(L) - int(protected))
    want = min(chunk_ceil(need, chunk), max_spill)
    boundary = int(L) - want
    return boundary, want


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


# ---------------------------------------------------------------------------
# S5 bs=1 spill-tick CUDA-graph planning (PURE; the capture/replay itself is
# GPU-only). Mirrors the proven weightless-KV #136a mechanics: a fixed-count
# streamed-block graph selected from a BUCKETED RUNG LADDER over the host
# block count, with all index/staging maps built OUT of the captured region
# and empty trailing blocks sanitized to the (o=0, lse=-inf) contract
# in-graph. A wasted trailing captured block still pays its fixed H2D copy +
# kernel launch (measured there: one wasted block ~halves the streamed rate),
# so the ladder stays DENSE in the common low range and only coarsens higher.
# ---------------------------------------------------------------------------


def spill_graph_enabled() -> bool:
    """Master gate for the S5 spill-tick graph. Default OFF: the flag being
    unset means NO S5 code runs -- the spill tick stays on the eager block
    loop, so the CODE PATH is byte-identical to S2/S3/S4 ('flag AUS
    byte-identisch'). The GPU validation opts in via SGLANG_KVSO_SPILL_GRAPH=1.

    NORMAL-PATH NUMERIC CAVEAT (GPU-measured, classified DECODE-CLASS): turning
    the flag ON also captures the spill-tick rung graphs at boot, which shifts
    the CUDA workspace / autotune layout enough to flip an occasional
    near-tie argmax on this heterogeneous non-batch-invariant rig (5090 sm120 +
    3080 sm86, uneven DCP). So flag ON is NOT bit-exact vs flag OFF on the
    pre-spill device path (measured: greedy solo diverges ~token 70). This is
    BENIGN decode-class, not a regression: on this rig there is NO bit-exact
    device-decode baseline to begin with -- flag OFF greedy solo is itself
    non-reproducible cross-boot (measured: three fresh boots diverge at tokens
    34 / 112, single-origin flips that compound autoregressively). flag ON was
    in fact cross-boot IDENTICAL in the same test. The 'flag OFF byte-identisch'
    claim is a CODE-PATH claim (no S5 code executes) and holds; numeric
    cross-boot determinism is a rig property independent of this feature.
    Making flag ON bit-exact-neutral is structurally impossible here (no
    bit-exact target exists), so it is documented rather than 'fixed'."""
    return os.environ.get("SGLANG_KVSO_SPILL_GRAPH", "0") == "1"


def spill_decouple_enabled() -> bool:
    """Master gate for the decoupled spill lane (design_decoupled_spill.md):
    run the spill forward CONCURRENTLY with the device batch on its own stream
    + flashinfer workspace + DCP communicator, so the device session stops
    waiting on the spill's PCIe H2D. Default OFF -> the spill tick stays serial
    (taken instead-of the device batch) and every extra resource collapses to
    the shared one, so flag OFF is byte-identical. Built incrementally (S2
    workspace, S3 comm, S4 stream/overlap), each step gated by this flag and
    verified byte-identical while still serial."""
    return os.environ.get("SGLANG_KVSO_DECOUPLE", "0") == "1"


def spill_graph_blocks_needed(owned_tokens: int, block_size: int) -> int:
    """Number of streamed host blocks for ``owned_tokens`` at ``block_size``
    (ceil, at least 1). For rank-uniform capture pass the MAX owned count
    across ranks (``num_blocks_rank_uniform`` already yields that block
    count)."""
    b = max(1, int(block_size))
    n = max(0, int(owned_tokens))
    return max(1, (n + b - 1) // b)


def spill_graph_rung_ladder(max_blocks: int) -> List[int]:
    """Bucketed block-count ladder covering 1..``max_blocks``: dense (step 1)
    up to 8 blocks, then ~x1.5 geometric to the max. Identical construction
    to the proven ``wl_build_graph_ladder``. Replay picks the smallest
    covering rung; over-ladder seq lens fall back to eager."""
    r_max = max(1, int(max_blocks))
    ladder = set()
    r = 1
    while r < r_max:
        ladder.add(r)
        r = r + 1 if r < 8 else max(r + 1, int(r * 1.5))
    ladder.add(r_max)
    return sorted(ladder)


def spill_graph_pick_rung(
    needed_blocks: int, ladder: List[int]
) -> Optional[int]:
    """Smallest captured rung >= ``needed_blocks`` (the RANK-UNIFORM host
    block count), or None when the seq len needs more blocks than the largest
    rung -> eager fallback. Rank-uniform: ``needed_blocks`` derives only from
    replicated scheduler state, so every rank picks the same rung (or all
    fall back together), preserving the per-layer collective lockstep."""
    if not ladder:
        return None
    n = max(1, int(needed_blocks))
    for r in ladder:
        if r >= n:
            return r
    return None


def spill_graph_block_stage_counts(
    owned_tokens: int, block_size: int, rung: int
) -> List[int]:
    """Per-captured-block staged row count for a ``rung``-block graph: block j
    stages ``clamp(owned - j*B, 0, B)`` rows. Blocks that reach 0 are captured
    NO-OPS -- their empty attention is sanitized to (o=0, lse=-inf) in-graph
    and folded into the online merge as identity, matching the eager loop's
    ``continue``. This is the out-of-graph plan input, built once per step."""
    b = max(1, int(block_size))
    n = max(0, int(owned_tokens))
    return [max(0, min(b, n - j * b)) for j in range(max(0, int(rung)))]


def spill_graph_out_plan(
    host_base_row: int,
    owned_tokens: int,
    block_size: int,
    rung: int,
    *,
    device=None,
) -> List[dict]:
    """Out-of-graph staging plan for one spill-tick decode step (built ONCE
    per step, NEVER inside the captured region -- this is the S5 analogue of
    hoisting .plan()/index maps out of the graph, the actual #136a speedup).

    The active host tail owns host-pool rows ``[base, base + owned)`` where
    ``base = region_base + wave_drain`` (S3/S4 region-scoped). For a fixed
    ``rung``-block graph returns, per block j, a dict with:
      * ``cnt``:       staged row count (0 => empty, sanitized in-graph),
      * ``host_rows``: int64 host-pool rows to gather
                       ``[base + j*B, base + j*B + cnt)`` (empty tensor when 0),
      * ``indptr``:    int32 ``[0, cnt]`` single-request page indptr.
    Every block is present (fixed count == rung) so the captured H2D/gather/
    run/merge sequence has a constant shape; only ``cnt`` varies the planned
    row count of the (persistent, out-of-graph-planned) wrapper."""
    counts = spill_graph_block_stage_counts(owned_tokens, block_size, rung)
    b = max(1, int(block_size))
    base = int(host_base_row)
    plan = []
    for j, cnt in enumerate(counts):
        s = base + j * b
        host_rows = torch.arange(
            s, s + cnt, dtype=torch.int64, device=device
        )
        indptr = torch.tensor([0, cnt], dtype=torch.int32, device=device)
        plan.append({"cnt": cnt, "host_rows": host_rows, "indptr": indptr})
    return plan


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
    pos_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This rank's owned tokens of a req_to_token row (position order).

    Returns (owned_positions_mask, compact_device_indices_of_owned).
    ``mode`` in {"weighted", "even", "plain"}:
      * weighted: ownership/compaction from the slot id (loc % S rule).
      * even:     ownership from POSITION (p % dcp_size == rank), compact
                  slot = loc // dcp_size (the classic modulo layout).
      * plain:    single owner, compact slot = loc (dcp_size == 1).

    ``pos_offset`` (S1b): when ``row`` is a TAIL SEGMENT starting at absolute
    position ``pos_offset``, the even-mode positional owner rule must key on
    the ABSOLUTE position (``pos_offset + i``), not the segment-relative one.
    Ignored by weighted/plain (their ownership is slot-derived, position-
    independent)."""
    L = row.numel()
    if mode == "weighted":
        owned, compact = compact_weighted(row, S, lo, hi)
        return owned, compact[owned].contiguous()
    if mode == "even":
        pos = torch.arange(
            pos_offset, pos_offset + L, dtype=torch.int64, device=row.device
        )
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

    BY-DESIGN CONSEQUENCE (a SINGLE session never self-spills): under plain
    decode-OOM with exactly one running session, that session IS the oldest
    -> untouchable -> no victim -> None -> the stock retract/length-truncate
    path caps it at the device budget instead of spilling it to host. Spill
    is a SCHEDULER-PRESSURE relief (free room for OTHER sessions / a fast
    request), not a single-session context extender. A lone request > the
    device KV budget therefore truncates; it does not spill. (Extending a
    single over-budget session to host would need a different trigger and is
    out of scope -- cf. the weightless-KV lane's single-request 262k path.)

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


def wave_back_advance(
    boundary: int, seq_len: int, wave_step: int, remaining_cap: Optional[int] = None
) -> int:
    """Token advance for ONE incremental wave-back step (S3).

    The device head is ``[0, boundary)``; the host tail is
    ``[boundary, seq_len)``. Wave-back migrates the FRONT of the host tail
    back to device, advancing the boundary by a generic block. Returns the
    number of tokens to move this step (0 when the tail is empty).

    * Bounded by ``wave_step`` (the generic owner-block unit -- amortizes the
      per-event H2D cost, mirrors the spill-side ``chunk``).
    * Bounded by what REMAINS on host (``seq_len - boundary``); the final,
      short block completes the restore even when smaller than a block.
    * Optionally bounded by ``remaining_cap`` (device slots actually free
      right now) so a step never asks for more room than exists -- the
      caller's owner-matched allocation is still the hard gate.

    Because ``remaining`` is recomputed from the LIVE ``seq_len`` every call,
    tokens appended to the tail while spilled (the boundary is fixed but
    ``seq_len`` grows) are handled without special-casing: a step still only
    ever peels ``wave_step`` off the FRONT."""
    remaining = max(0, int(seq_len) - int(boundary))
    if remaining == 0:
        return 0
    step = min(max(1, int(wave_step)), remaining)
    if remaining_cap is not None:
        step = min(step, max(0, int(remaining_cap)))
    return max(0, step)


class WaveBackController:
    """Opportunistic wave-back gate (S3): decides, per scheduler iteration,
    whether and how far to migrate the host tail's front block back to the
    device head. Pure/deterministic -- identical on every TP rank (all inputs
    are replicated scheduler state).

    Policy:
      * A warmup gate (``RestoreHysteresis``) requires the "device space is
        available" condition to hold for N consecutive checks before the
        FIRST wave of a spill (anti-flutter, same discipline as full restore).
      * ``copy_inflight`` (the previous wave's H2D still running on the copy
        stream) backs off WITHOUT breaking the warmup streak: under PCIe
        contention we simply wave slower, never queue contending copies onto
        the device path.
      * ``space_ok`` False (device filled up again) resets the streak: never
        wave back into a device that has no room."""

    def __init__(self, wave_step: int, warmup_steps: int):
        self.wave_step = max(1, int(wave_step))
        self.gate = RestoreHysteresis(warmup_steps)

    def plan(
        self,
        boundary: int,
        seq_len: int,
        *,
        space_ok: bool,
        copy_inflight: bool,
        remaining_cap: Optional[int] = None,
    ) -> int:
        if max(0, int(seq_len) - int(boundary)) == 0:
            return 0
        if copy_inflight:
            # Back off this window; keep the warmup streak (contention, not a
            # memory shortage).
            return 0
        if not self.gate.update(space_ok):
            return 0
        return wave_back_advance(boundary, seq_len, self.wave_step, remaining_cap)

    def reset(self):
        self.gate.reset()


# ---------------------------------------------------------------------------
# Scheduler-side manager
# ---------------------------------------------------------------------------


class SpillSlot:
    """One CONCURRENTLY spilled session's scheduling state (S4).

    Each spilled session owns a distinct host-pool REGION (index ``region``,
    host rows ``[region * region_tokens, ...)``) and its own wave-back /
    restore-hysteresis gates. The ticks of all slots are still serialized
    (one spill tick per scheduler iteration, round-robin by
    ``last_tick_iter``), so the transient per-tick backend state (staging,
    copy stream, wave event) stays singular; only this persistent per-session
    state is multiplied. The pure victim ORDERING is unchanged -- multi-spill
    just repeats the single-victim selection over free regions."""

    __slots__ = (
        "req",
        "batch",
        "region",
        "spill_iter",
        "last_tick_iter",
        "wave",
        "hysteresis",
    )

    def __init__(self, req, region, spill_iter, wave, hysteresis):
        self.req = req
        self.batch = None  # bs=1 decode batch, built lazily at the first tick
        self.region = region
        self.spill_iter = spill_iter
        self.last_tick_iter = -(1 << 30)
        self.wave = wave
        self.hysteresis = hysteresis


class KVSessionOffloadManager:
    """Owns the spill/restore state machine inside one scheduler process.

    All decisions are functions of replicated scheduler state; the manager
    performs only rank-local GPU<->host copies (no collectives). S4: up to
    ``max_spills`` sessions may be spilled at once, one per host-pool region.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        sa = scheduler.server_args
        self.block_size = int(sa.kv_session_offload_block_size)
        self.tick_interval = max(1, int(sa.kv_session_offload_tick_interval))
        # DECOUPLE S4b: master gate for the concurrent spill lane. When ON the
        # scheduler dispatches the device decode batch AND a due spill tick in
        # the SAME iteration on two streams (device -> forward_stream/comm A,
        # spill -> spill_stream/comm B) instead of taking the tick INSTEAD-OF
        # the device batch. OFF -> the tick stays serial and this attr is
        # unused (byte-identical). tick_interval keeps its meaning but its ROLE
        # shifts from "protect the device lane" (serial) to a PCIe-backpressure
        # knob (how often the spill lane advances) since the lanes no longer
        # steal each other's iterations.
        self.decouple = bool(spill_decouple_enabled())
        self.restore_margin_tokens = int(sa.kv_session_offload_restore_margin_tokens)
        self._hysteresis_steps = int(
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
        self.backend = backend
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

        # S4 multi-spill: the host pool is partitioned into `max_spills` equal
        # regions of `region_tokens` rows each; region r owns host rows
        # [r * region_tokens, (r+1) * region_tokens). At most one session per
        # region is spilled at a time.
        self.max_spills = max(1, int(sa.kv_session_offload_max_spills))
        self.region_tokens = int(
            getattr(mr, "kv_sess_region_tokens", self.host_pool.size)
        )
        assert self.region_tokens * self.max_spills <= self.host_pool.size, (
            "kv-session-offload: host pool too small for "
            f"{self.max_spills} x {self.region_tokens} regions"
        )
        self._free_regions = list(range(self.max_spills))
        # Active spilled sessions, keyed by req_pool_idx (rpi). Insertion
        # order is preserved (FCFS-ish tick fairness fallback).
        self.spills: dict[int, SpillSlot] = {}

        self._iter_ct = 0  # incremented once per pre_schedule call
        self._last_tick_iter = -(1 << 30)  # AGGREGATE cadence across slots
        self._fast_queue_logged_rid = None
        self._fast_lane_enabled = bool(getattr(sa, "enable_fast_lane", False))

        global _MANAGER
        _MANAGER = self

        logger.info(
            "kv-session-offload (S4) armed: mode=%s S=%d prefix=%s rank=%d "
            "block=%d tick_interval=%d restore_margin=%d hysteresis=%d "
            "host_pool=%d tokens/rank max_spills=%d region=%d tokens",
            self.mode,
            self.S,
            self.cp_prefix,
            self.dcp_rank,
            self.block_size,
            self.tick_interval,
            self.restore_margin_tokens,
            self._hysteresis_steps,
            self.host_pool.size,
            self.max_spills,
            self.region_tokens,
        )

    # -- slot bookkeeping -------------------------------------------------

    def _slot_of(self, req) -> Optional["SpillSlot"]:
        return self.spills.get(req.req_pool_idx)

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
        (overlap mode). Event-based; no host sync.

        DECOUPLE S4b: with the concurrent spill lane there are TWO in-flight
        forwards (device on forward_stream, spill on spill_stream). A wave-back
        / restore H2D into freshly allocated device slots must be ordered after
        BOTH, so also wait spill_stream when decoupling is on. Cheap (an event
        wait on the schedule stream); no host sync. The spill forward never
        writes the device KV pool (its output token lands in a host sentinel
        slot), so this is conservative, but keeping the barrier symmetric
        avoids any reader/writer edge being missed."""
        cur = torch.cuda.current_stream()
        fs = getattr(self.scheduler, "forward_stream", None)
        if fs is not None:
            cur.wait_stream(fs)
        if self.decouple:
            ss = getattr(self.scheduler, "spill_stream", None)
            if ss is not None:
                cur.wait_stream(ss)

    def has_spilled(self) -> bool:
        return len(self.spills) > 0

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

    def try_spill(
        self, batch: "ScheduleBatch", fast_pressure=None, need: Optional[int] = None
    ) -> bool:
        """Spill the least-protected running session out of ``batch``
        (youngest NORMAL; fast-lane requests are never victims; under
        fast-lane pressure even the oldest normal may lose its device
        residency). Called from update_running_batch when check_decode_mem
        failed (after tree eviction) and from the fast-lane admission
        trigger. Returns True when a session was spilled.

        S1b PARTIAL SPILL: only the victim's TAIL overhang moves to host --
        the head ``[0, boundary)`` stays device-resident and keeps its tree
        lock / protected prefix. The freed tail is ``chunk_ceil(need,
        block_size)`` tokens (block-aligned; over-eviction margin <=
        block-1). A whole-session spill is just the boundary==protected
        special case (need >= the exclusive suffix).

        ``need`` (token shortfall to free) is computed from the batch's next
        decode step when not given; the fast-lane trigger passes the waiting
        request's shortfall explicitly. If a single victim's exclusive
        suffix cannot cover ``need`` the whole suffix spills; the caller
        (decode-OOM re-check, or the fast-lane loop) spills further victims
        into further free regions.

        S4: spills ONE victim into ONE free host region per call. Returns
        False when no region is free (falls back to stock retraction) --
        never an inconsistent partial spill."""
        if not self._free_regions:
            return False
        if fast_pressure is None:
            fast_pressure = self._fast_lane_pressure(batch.reqs)
        # Shortfall X (tokens) + per-session spillable sizes -> minimal
        # single-session eviction (youngest sufficient; see
        # select_spill_victim). Only req-exclusive slots count as freed
        # (the shared radix prefix stays tree-owned, merely evictable).
        if need is None:
            need = max(
                0,
                batch.new_tokens_required_next_decode()
                - self.allocator.available_size(),
            )
        need = max(0, int(need))
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

        # S1b PARTIAL SPILL: only the block-aligned TAIL overhang migrates to
        # host. boundary splits the row into a device-resident head
        # [0, boundary) (kept, tree-locked) and a host tail [boundary, L).
        protected = int(req.cache_protected_len or 0)
        boundary, spill_count = partial_spill_plan(
            L, protected, need, self.block_size
        )
        if spill_count <= 0:
            # need <= 0 after the internal recompute: nothing to free. Leave
            # the batch untouched (the stock retract path decides).
            return False
        spill_margin = spill_count - need  # over-eviction metric (<= block-1)

        row = self.req_to_token_pool.req_to_token[req.req_pool_idx, :L]
        seg = row[boundary:L]  # migrating tail; wholly req-exclusive (>= protected)
        owned_mask, dev_idx = owned_device_indices(
            seg,
            mode=self.mode,
            S=self.S,
            lo=self.lo,
            hi=self.hi,
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
            pos_offset=boundary,  # even-mode ownership keys on absolute position
        )
        n_own = int(dev_idx.numel())
        if n_own > self.region_tokens:
            logger.warning(
                "kv-session-offload: session rid=%s tail needs %d host rows > "
                "region %d; falling back to stock retraction.",
                req.rid,
                n_own,
                self.region_tokens,
            )
            return False

        # Claim a free host region for this session (S4).
        region = self._free_regions.pop(0)
        region_base = region * self.region_tokens

        # Owner residues of the TAIL SEGMENT, before the row is overwritten.
        # Weighted rule: ownership is a function of the slot id -> preserve
        # seg % S. Even/plain rules: ownership is positional -> keyed on the
        # ABSOLUTE position (boundary..L) so the sentinel encoding (which
        # carries position) stays consistent with the retained head.
        if self.mode == "weighted":
            residues = (seg.to(torch.int64) % self.S).contiguous()
        else:
            residues = (
                torch.arange(boundary, L, dtype=torch.int64, device=row.device)
                % self.S
            )

        # Register the session's backend slot (region base, fresh head/count
        # caches, drain=0). The tick sources its per-session state from here.
        self.backend._sess_open_slot(req.req_pool_idx, region_base)

        # 1. Order after any in-flight forward that still writes this
        #    session's last KV row (overlap mode), then D2H-backup the tail's
        #    owned slots (all full-attention layers) into this region's host
        #    rows [region_base, region_base + n). Also quiesce the streamed-
        #    prefetch copy stream: a previous spill's queued H2D reads of these
        #    host rows must complete before we overwrite them (double buffer).
        self._wait_forward_stream()
        torch.cuda.current_stream().wait_stream(self.backend._sess_copy_stream)
        if n_own > 0:
            host_ids = torch.arange(
                region_base, region_base + n_own, dtype=torch.int64, device=row.device
            )
            self.host_pool.backup_from_device_all_layer(
                self.full_pool, host_ids, dev_idx, io_backend="kernel"
            )

        # 2. Free ONLY the tail device slots. The head [0, boundary) -- the
        #    shared protected radix prefix AND the retained exclusive head --
        #    stays device-resident and tree-locked (last_node / prefix_indices
        #    / cache_protected_len are NOT reset): the hybrid spill tick
        #    attends the head on device every layer, the tail from host.
        self.allocator.free(seg.to(torch.int64))

        # 3. Rewrite ONLY the tail row [boundary, L) with sentinels; the head
        #    keeps its real slot ids. start=boundary encodes the absolute
        #    position so every rank re-derives head/tail split + tail
        #    ownership from the (replicated) row alone.
        sent = make_sentinels(self.host_base, self.S, residues, start=boundary)
        assert int(sent[-1].item()) < (1 << 31) - self.S, (
            "kv-session-offload: sentinel overflow (int32 req_to_token)"
        )
        self.req_to_token_pool.req_to_token[req.req_pool_idx, boundary:L] = sent.to(
            torch.int32
        )

        req.kv_spill_state = "host"
        req.kv_spill_boundary = boundary
        slot = SpillSlot(
            req=req,
            region=region,
            spill_iter=self._iter_ct,
            wave=WaveBackController(self.block_size, self._hysteresis_steps),
            hysteresis=RestoreHysteresis(self._hysteresis_steps),
        )
        self.spills[req.req_pool_idx] = slot

        keep = [i for i in range(len(batch.reqs)) if i != idx]
        batch.filter_batch(keep_indices=keep)
        batch.batch_is_full = False

        self._log(
            "kv-session-offload SPILL(partial): rid=%s arrival_seq=%s L=%d "
            "boundary=%d device_head=%d host_tail=%d owned_tail=%d (rank %d) "
            "region=%d protected_prefix=%d; need=%d freed=%d over-eviction "
            "margin=%d tokens; device batch bs=%d spills=%d/%d",
            req.rid,
            getattr(req, "kv_arrival_seq", None),
            L,
            boundary,
            boundary,
            spill_count,
            n_own,
            self.dcp_rank,
            region,
            protected,
            need,
            spill_count,
            spill_margin,
            len(batch.reqs),
            len(self.spills),
            self.max_spills,
        )
        return True

    # -- spill tick -------------------------------------------------------

    def _build_spill_batch(self, req: "Req") -> "ScheduleBatch":
        """Decode ScheduleBatch for a spilled session (persistent across
        ticks, exactly like running_batch). Mirrors the hisparse
        staging->decode builder: the last sampled token is stashed into the
        future map so resolve_forward_inputs picks it up."""
        from sglang.srt.managers.overlap_utils import RelayPayload
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

        sch = self.scheduler
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
        selection): cleanup of finished spilled sessions, fast-lane admission
        pressure, and incremental wave-back / FIFO restore of each spilled
        session. Returns the (possibly merged) running_batch."""
        self._iter_ct += 1

        # 1. Reap sessions that finished on host (release ran in the result
        #    processor and already freed the region + backend slot).
        reaped = False
        for rpi, slot in list(self.spills.items()):
            if slot.req.finished():
                if slot.batch is not None:
                    slot.batch.filter_batch()
                self._close_slot(rpi, "finished on host")
                reaped = True
        if reaped and running_batch is not None:
            # A host-finished session freed device KV + its req slot but was
            # never in running_batch: un-stick the prefill admission gate so a
            # request waiting on that KV is re-evaluated (see the DEADLOCK FIX
            # note in release_finished_spilled_req). Safety net for finish
            # paths that bypass release (e.g. abort). Rank-uniform.
            running_batch.batch_is_full = False

        if not self.spills:
            self._maybe_spill_for_fast_lane(running_batch)
            return running_batch

        # 2. Fast-lane admission may still evict more sessions (into further
        #    free regions) even while some are already spilled.
        self._maybe_spill_for_fast_lane(running_batch)

        # 3. Restore / wave-back each spilled session independently. A
        #    completed restore merges that session back into running_batch.
        for rpi, slot in list(self.spills.items()):
            running_batch = self._maybe_restore_flow(slot, running_batch, last_batch)
        return running_batch

    def _maybe_spill_for_fast_lane(self, running_batch):
        """Fast-lane admission trigger: a WAITING fast-lane request that does
        not fit into (free + tree-evictable) KV evicts the youngest running
        normal session -- under fast pressure even the oldest
        (select_spill_victim fast_pressure=True). S4: MULTIPLE normal sessions
        may be evicted (one per free region) until the fast request fits or no
        region/victim remains -- lifting the S1 single-eviction limit."""
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

        spilled_any = 0
        while self._free_regions:
            have = self.allocator.available_size() + self._tree_evictable_size()
            if have >= need:
                break  # normal admission will take it now
            # Free the residual shortfall (block-rounded inside try_spill);
            # each victim is a partial tail spill into its own region.
            if not self.try_spill(
                running_batch, fast_pressure=True, need=need - have
            ):
                break  # no eligible victim in the batch
            spilled_any += 1

        if spilled_any:
            self._log(
                "kv-session-offload: spilled %d session(s) for FAST-LANE "
                "request rid=%s (need %d tokens)",
                spilled_any,
                fr.rid,
                need,
            )
        elif self._fast_queue_logged_rid != fr.rid:
            self._fast_queue_logged_rid = fr.rid
            self._log(
                "kv-session-offload (S4): fast-lane request rid=%s needs %d "
                "tokens but no free region / no eligible victim can free "
                "enough -- request stays queued.",
                fr.rid,
                need,
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

    def _maybe_restore_flow(self, slot, running_batch, last_batch):
        # While a fast-lane request is waiting for admission, the freed
        # device space belongs to IT (fast beats FCFS): restoring now would
        # only re-trigger the fast-pressure spill (spill<->restore thrash,
        # one full D2H+H2D per cycle). Normal-session restore resumes once
        # the fast request is admitted or gone.
        if self._fast_lane_enabled and any(
            getattr(r, "is_fast_lane", False)
            for r in getattr(self.scheduler, "waiting_queue", ())
        ):
            slot.hysteresis.reset()
            return running_batch

        req = slot.req
        # Restore only when this session is quiescent: it was not the batch
        # launched last iteration (its result must be processed so
        # seq_lens/output_ids are settled), and memory holds.
        if slot.batch is not None:
            if last_batch is slot.batch:
                return running_batch
            L = int(slot.batch.seq_lens_cpu[0].item())
        else:
            # Spilled but never ticked yet: restorable once the victim's
            # last device result has been processed (>= one iteration).
            if self._iter_ct <= slot.spill_iter:
                return running_batch
            L = len(req.origin_input_ids) + len(req.output_ids) - 1

        row = self.req_to_token_pool.req_to_token[req.req_pool_idx, :L]
        boundary = int((row < self.host_base).sum().item())
        if boundary >= L:
            # Tail fully drained by earlier wave-back steps -- nothing left on
            # host; just rejoin the device batch (no H2D remains).
            return self._finalize_restore(slot, running_batch, L)

        remaining = L - boundary
        avail = self.allocator.available_size()

        # FALLBACK fast path: the whole tail fits right now -> one-shot
        # stop-restore (S1/S2 path; faster than nibbling when space is
        # abundant, e.g. the pressure that caused the spill is fully gone).
        if avail >= remaining + self.restore_margin_tokens:
            if slot.hysteresis.update(self._restore_memory_ok(req, L)):
                return self._restore(slot, running_batch, L)
            return running_batch
        slot.hysteresis.reset()

        # PRIMARY path: opportunistic incremental wave-back of one owner-block
        # from the host-tail front. H2D on the copy stream -> other device
        # sessions are never delayed; under copy-stream contention we simply
        # wave slower (copy_inflight backs off). The owner-matched allocation
        # inside _wave_back is the hard gate; `remaining_cap` keeps the step
        # from ever asking for more slots than are free.
        copy_inflight = not self.backend._sess_wave_done.query()
        advance = slot.wave.plan(
            boundary,
            L,
            space_ok=avail > 0,
            copy_inflight=copy_inflight,
            remaining_cap=avail,
        )
        if advance > 0:
            self._wave_back(slot, L, boundary, advance)
        return running_batch

    def _pick_tick_slot(self, running_batch):
        """Round-robin the spill tick over the active slots: among the slots
        that are DUE (past their spill iteration, not finished), pick the one
        that ticked least recently (fairness). Rank-uniform (all inputs are
        replicated: iter counters + finished flags)."""
        due = [
            slot
            for slot in self.spills.values()
            if not slot.req.finished() and self._iter_ct > slot.spill_iter
        ]
        if not due:
            return None
        return min(due, key=lambda s: s.last_tick_iter)

    def maybe_take_tick(self, running_batch) -> Optional["ScheduleBatch"]:
        """Late hook: decide whether THIS iteration runs a spill tick instead
        of the device decode batch, and for WHICH spilled session (round-
        robin). Must be called BEFORE update_running_batch so the device batch
        is not prepared and then dropped."""
        if not self.spills:
            return None
        # Cadence: with device work present, leave at least tick_interval
        # device iterations between two spill ticks (AGGREGATE across all
        # spilled sessions -- they share one tick slot). Without device work
        # a spilled session ticks every iteration.
        device_has_work = running_batch is not None and not running_batch.is_empty()
        if (
            device_has_work
            and (self._iter_ct - self._last_tick_iter) <= self.tick_interval
        ):
            return None

        slot = self._pick_tick_slot(running_batch)
        if slot is None:
            return None

        if slot.batch is None:
            slot.batch = self._build_spill_batch(slot.req)
            self._log(
                "kv-session-offload: first spill tick for rid=%s (L=%d)",
                slot.req.rid,
                int(slot.batch.seq_lens_cpu[0].item()),
            )
        batch = slot.batch
        batch.filter_batch()
        if batch.is_empty():
            self._close_slot(slot.req.req_pool_idx, "finished on host")
            return None
        batch.prepare_for_decode()
        slot.last_tick_iter = self._iter_ct
        self._last_tick_iter = self._iter_ct
        return batch

    # -- restore ------------------------------------------------------------

    def _restore_memory_ok(self, req, L: int) -> bool:
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        # Only the HOST TAIL needs fresh device slots; the head [0, boundary)
        # kept its slots throughout the (partial) spill. boundary is derived
        # from the replicated row (leading non-sentinel run).
        row = self.req_to_token_pool.req_to_token[req.req_pool_idx, :L]
        boundary = int((row < self.host_base).sum().item())
        tail = L - boundary
        need = tail + self.restore_margin_tokens
        if self.allocator.available_size() < need:
            evict_from_tree_cache(self.tree_cache, need)
        if self.allocator.available_size() < need:
            return False
        if self.mode == "weighted":
            # Per-owner-class availability (free slots whose residue class
            # matches each rank's owned TAIL-token count).
            residues = row[boundary:L].to(torch.int64) % self.S
            counts = owned_counts_weighted(residues, self.cp_prefix)
            free = self.allocator.free_pages
            res_free = free % self.S
            for r in range(len(counts)):
                lo, hi = self.cp_prefix[r], self.cp_prefix[r + 1]
                have = int(((res_free >= lo) & (res_free < hi)).sum().item())
                if have < counts[r]:
                    return False
        return True

    def _restore(self, slot, running_batch, L: int):
        req = slot.req
        bslot = self.backend._sess_slots[req.req_pool_idx]
        row = self.req_to_token_pool.req_to_token[req.req_pool_idx, :L]
        # Wave back ONLY the host tail [boundary, L); the head kept its slots.
        boundary = int((row < self.host_base).sum().item())
        seg = row[boundary:L]
        tail = int(seg.numel())
        assert tail == 0 or int(seg.min().item()) >= self.host_base, (
            "kv-session-offload restore: non-sentinel slot in a spilled tail"
        )
        residues = (seg.to(torch.int64) % self.S).contiguous()

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
                slot.hysteresis.reset()
                return running_batch
            new_locs = assign_owner_matched_slots(
                residues, self.cp_prefix, class_slots
            )
        else:
            new_locs = self.allocator.alloc(tail)
            if new_locs is None:
                slot.hysteresis.reset()
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
            pos_offset=boundary,  # even-mode ownership keys on absolute position
        )
        n_own = int(dev_idx.numel())
        if self.mode == "weighted":
            assert n_own == counts[self.dcp_rank], (
                "kv-session-offload restore: owner-matched allocation "
                f"produced {n_own} owned slots, expected {counts[self.dcp_rank]}"
            )

        self._wait_forward_stream()
        # Within this session's region, the active tail owns host rows
        # [region_base + drain, ... + n_own): wave-back drained [.., drain).
        # n_own == 0 when this rank owns NONE of the tail (weighted uneven-DCP:
        # the tail's residues all fall in other ranks' classes). The H2D
        # transfer is rank-LOCAL (a pure host->device memcpy kernel, no
        # collective), and its grid dim is div_ceil(len(indices), ...) which
        # is 0 for an empty index tensor -> a 0-block launch throws CUDA
        # "invalid configuration argument". Skip the launch on 0-owned ranks;
        # the row rewrite below still runs everywhere (replicated slot ids).
        base = bslot.region_base + bslot.host_row_base
        if n_own > 0:
            host_ids = torch.arange(
                base, base + n_own, dtype=torch.int64, device=row.device
            )
            layer_num = getattr(self.host_pool, "layer_num", 0)
            for fl in range(layer_num):
                self.host_pool.load_to_device_per_layer(
                    self.full_pool, host_ids, dev_idx, fl, io_backend="kernel"
                )
        self.req_to_token_pool.req_to_token[req.req_pool_idx, boundary:L] = (
            new_locs.to(torch.int32)
        )
        self._log(
            "kv-session-offload RESTORE(stop): rid=%s L=%d boundary=%d "
            "tail=%d owned_tail=%d region=%d host_base=%d (rank %d) H2D complete",
            req.rid,
            L,
            boundary,
            tail,
            n_own,
            slot.region,
            base,
            self.dcp_rank,
        )
        return self._finalize_restore(slot, running_batch, L)

    def _wave_back(self, slot, L: int, boundary: int, advance: int) -> bool:
        """Migrate the host-tail FRONT block ``[boundary, boundary+advance)``
        back to the device head on the COPY stream (S3 incremental restore).

        The tier boundary advances by the block; the session stays a hybrid
        spill tick with a larger device head and a shorter host tail. Purely
        additive to the device path: the H2D runs on the dedicated copy
        stream (never the compute/default stream) and the freshly written
        head slots are consumed by the NEXT tick, which waits the copy via
        ``_sess_wave_done`` -- so OTHER device sessions are never delayed.

        Returns True when a block was moved, False when the owner-matched
        allocation could not be satisfied right now (wave stalls, retried
        next window; the stock full-restore stays available as fallback)."""
        req = slot.req
        bslot = self.backend._sess_slots[req.req_pool_idx]
        hi_pos = min(boundary + advance, L)
        row = self.req_to_token_pool.req_to_token[req.req_pool_idx, :L]
        block = row[boundary:hi_pos]
        assert int(block.min().item()) >= self.host_base, (
            "kv-session-offload wave-back: non-sentinel slot in the tail front"
        )
        residues = (block % self.S).to(torch.int64).contiguous()

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
                return False  # not enough owner-matched room this window
            new_locs = assign_owner_matched_slots(
                residues, self.cp_prefix, class_slots
            )
        else:
            new_locs = self.allocator.alloc(hi_pos - boundary)
            if new_locs is None:
                return False
            new_locs = new_locs.to(torch.int64)

        owned_mask, dev_idx = owned_device_indices(
            new_locs,
            mode=self.mode,
            S=self.S,
            lo=self.lo,
            hi=self.hi,
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
            pos_offset=boundary,  # even-mode ownership keys on absolute position
        )
        blk_own = int(dev_idx.numel())

        # H2D on the copy stream: read this block's owned host rows
        # [base, base+blk_own) into the new device slots. Order after any
        # in-flight forward, record the wave event so the next tick's device-
        # head read (and the next wave's copy_inflight check) can wait it.
        # blk_own == 0 when this rank owns NONE of the block (weighted
        # uneven-DCP, or a small final block whose residues miss this rank's
        # class): the transfer is rank-local and its grid dim is
        # div_ceil(len(indices), ...) == 0 for an empty index tensor, so a
        # 0-block launch throws CUDA "invalid configuration argument". Skip
        # the launches on 0-owned ranks; the boundary/drain advance + row
        # rewrite below still run everywhere. The wave event is still recorded
        # (idle when nothing was copied) so the next tick's wait stays valid.
        base = bslot.region_base + bslot.host_row_base
        layer_num = getattr(self.host_pool, "layer_num", 0)
        cs = self.backend._sess_copy_stream
        self._wait_forward_stream()
        with torch.cuda.stream(cs):
            cs.wait_stream(torch.cuda.current_stream())
            if blk_own > 0:
                host_ids = torch.arange(
                    base, base + blk_own, dtype=torch.int64, device=row.device
                )
                for fl in range(layer_num):
                    self.host_pool.load_to_device_per_layer(
                        self.full_pool, host_ids, dev_idx, fl, io_backend="kernel"
                    )
            self.backend._sess_wave_done.record(cs)

        # Rewrite the block's row with the real device slots; advance the
        # boundary (implicitly, via the now-non-sentinel entries) and this
        # session's host-row drain. Force re-derivation of the head split +
        # tail counts on the next tick.
        self.req_to_token_pool.req_to_token[req.req_pool_idx, boundary:hi_pos] = (
            new_locs.to(torch.int32)
        )
        bslot.host_row_base += blk_own
        self.backend._sess_slot_reset_head(req.req_pool_idx)
        # Head grew to [0, hi_pos): keep the finish-path free bound current.
        req.kv_spill_boundary = hi_pos
        self._log(
            "kv-session-offload WAVE-BACK: rid=%s boundary %d->%d (+%d) "
            "owned_block=%d region=%d drain->%d tail_left=%d (rank %d)",
            req.rid,
            boundary,
            hi_pos,
            hi_pos - boundary,
            blk_own,
            slot.region,
            bslot.host_row_base,
            L - hi_pos,
            self.dcp_rank,
        )
        return True

    def _finalize_restore(self, slot, running_batch, L: int):
        """Common rejoin tail shared by the one-shot stop-restore and a
        completed wave-back (host tail empty): flip the session back to the
        device decode path, free its region, and merge it into the running
        batch."""
        req = slot.req
        req.kv_spill_state = None
        req.kv_spill_boundary = 0
        if slot.batch is None:
            # Never ticked while spilled: build the decode batch now so the
            # session can be merged back like any resumed decode request.
            slot.batch = self._build_spill_batch(req)
        batch = slot.batch
        # Back on device: this batch (or its reqs inside running_batch) must
        # take the normal decode path again.
        batch.kv_session_spill_tick = False
        self._close_slot(req.req_pool_idx, "restored to device")
        self._log(
            "kv-session-offload RESTORE complete: rid=%s L=%d (rank %d) "
            "rejoining device batch",
            req.rid,
            L,
            self.dcp_rank,
        )
        if running_batch.is_empty():
            return batch
        running_batch.merge_batch(batch)
        return running_batch

    # -- finish / cleanup ---------------------------------------------------

    def release_finished_spilled_req(self, req: "Req"):
        """Finish path for a request that ends WHILE (partially) spilled: its
        row holds host sentinels in the tail [boundary, L) and REAL device
        slots in the retained head [0, boundary). Release the exclusive head
        slots [protected, boundary), drop the shared-prefix tree lock, and
        free the Mamba state + req slot. The host tail needs no free (the
        pinned host pool is a fixed ring reclaimed with the spill slot). No
        radix insert (the tail is on host -- there is no full device KV to
        donate)."""
        assert getattr(req, "kv_spill_state", None) == "host"
        pool = self.req_to_token_pool
        # Capture rpi BEFORE pool.free() nulls req.req_pool_idx (else the slot
        # lookup / region free below would key on None and leak the region).
        rpi = req.req_pool_idx
        slot = self.spills.get(rpi)
        region = slot.region if slot is not None else -1
        boundary = int(getattr(req, "kv_spill_boundary", 0) or 0)
        protected = int(req.cache_protected_len or 0)
        head_freed = 0
        if boundary > protected:
            # Retained exclusive device head [protected, boundary).
            head = pool.req_to_token[rpi, protected:boundary]
            self.allocator.free(head.to(torch.int64))
            head_freed = boundary - protected
        if req.last_node is not None:
            # The shared radix prefix stayed tree-locked across the spill.
            self.tree_cache.dec_lock_ref(req.last_node)
        if req.mamba_pool_idx is not None and hasattr(pool, "free_mamba_cache"):
            pool.free_mamba_cache(req)
        if slot is not None and slot.batch is not None:
            slot.batch.filter_batch()
        pool.free(req)
        req.kv_spill_state = None
        req.kv_spill_boundary = 0
        # Free this session's host region + backend slot (using the captured
        # rpi). The pre_schedule reap loop is then a no-op for this slot.
        self._close_slot(rpi, "finished on host")
        # DEADLOCK FIX: a session finishing ON HOST frees its device head +
        # req slot but was never in running_batch, so -- unlike the device-
        # finish / restore path (which resets batch_is_full via
        # update_running_batch) -- nothing here un-sticks the prefill
        # admission gate. Without this, a request waiting on exactly that
        # freed KV is never re-evaluated (batch_is_full stays True) and the
        # scheduler wedges at GPU 0%. Reset it so the next
        # get_new_batch_prefill retries admission. Rank-uniform: every rank
        # reaps the same finished session at the same iteration. Safe: it only
        # forces a re-check, which re-sets True if the batch is still full.
        rb = getattr(self.scheduler, "running_batch", None)
        if rb is not None:
            rb.batch_is_full = False
        self._log(
            "kv-session-offload: spilled session rid=%s finished on host; "
            "released device head=%d (boundary=%d protected=%d) + tree lock "
            "+ mamba + req slot + region %d (no radix insert); admission gate "
            "reset",
            req.rid,
            head_freed,
            boundary,
            protected,
            region,
        )

    def _close_slot(self, rpi: int, why: str):
        """Retire one spilled session: return its host region to the free
        pool, drop its backend per-session state, and remove it from the
        active set. Idempotent (safe if already closed)."""
        slot = self.spills.pop(rpi, None)
        if slot is None:
            return
        self._free_regions.append(slot.region)
        # The backend's per-session head/tail split + owned-count cache belong
        # to THIS session; a later spill re-derives from its own sentinel row.
        self.backend._sess_close_slot(rpi)
        logger.debug(
            "kv-session-offload: spill slot closed (%s, rpi=%d, region=%d)",
            why,
            rpi,
            slot.region,
        )

    def inflight_batches(self):
        """For abort scanning: every spilled session is running too (also in
        the window between spill and first tick, before a batch exists)."""
        from types import SimpleNamespace

        out = []
        for slot in self.spills.values():
            if slot.batch is not None:
                out.append(slot.batch)
            else:
                out.append(SimpleNamespace(reqs=[slot.req]))
        return out
