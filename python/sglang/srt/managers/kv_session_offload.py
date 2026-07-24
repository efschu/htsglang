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
import math
import os
import time
from collections import deque
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple

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


def mtp_resident_tail_fits(tail_tokens: int, resident_cap_slices: int) -> bool:
    """spec-in-spill-tick, Option (b'): whether a spilled session's DRAFT-KV
    tail can be kept DEVICE-resident within the
    ``--kv-session-offload-mtp-resident-slices`` cap.

    Under Option (b') the tiny (1-layer NEXTN / few-layer EAGLE) draft-KV tail
    is snapshotted into a dedicated device buffer at spill (the large,
    multi-layer, DCP-sharded TARGET KV still spills to host), so draft() runs
    on device while spilled. The draft pool is NOT DCP-token-sharded -- every
    rank holds the FULL token context (M4) -- so ``tail_tokens`` is the FULL
    tail ``L - boundary`` and this decision is RANK-UNIFORM by construction
    (every rank sees the same replicated L / boundary -> same verdict, no
    collective, no desync).

    ``resident_cap_slices == 0`` disables the cap (any tail is kept resident).
    A tail that EXCEEDS a positive cap does NOT OOM: the session falls back
    GRACEFULLY to the plain (spec-off) host tick for as long as it overflows
    (the caller keeps ``spec_algorithm=NONE`` for it and logs the fallback).
    The cap is a per-session, per-rank DEVICE-buffer ceiling the operator sets
    to bound the resident draft-KV growth of deep offloads; it is a QoS knob,
    not a correctness guard.
    """
    if int(resident_cap_slices) <= 0:
        return True
    return int(tail_tokens) <= int(resident_cap_slices)


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


def resume_under_spec_enabled() -> bool:
    """Bring-up gate for ON-DEVICE MTP RESUME (lift the host-finish guard so a
    spilled spec session waves back and rejoins the LIVE spec decode batch,
    instead of finishing on host). Default OFF keeps the validated host-finish
    path. Set KVSO_RESUME=1 to opt into the resume path (draft-KV bundle +
    spec-in-spill-tick). Mirrors KVSO_ALLOW_SPEC's staged-bring-up role."""
    return os.environ.get("KVSO_RESUME", "0") == "1"


def draft_kv_verify_enabled() -> bool:
    """Stage-1 self-check: after restoring the draft-KV bundle into the new
    slots, read it back and assert byte-exact equality with the pinned-CPU
    snapshot (validates the snapshot/restore plumbing -- slot indexing, dtype,
    shapes -- end to end). Off by default; KVSO_S1_VERIFY=1 for bring-up."""
    return os.environ.get("KVSO_S1_VERIFY", "0") == "1"


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


def spec_decline_non_back_spill(spec_active: bool, idx: int, n_reqs: int) -> bool:
    """Return True when a spill of request ``idx`` (out of ``n_reqs`` running)
    must be DECLINED because speculative decoding is active and ``idx`` is not
    the back-most request.

    SPEC BACK-ONLY REMOVAL INVARIANT: under EAGLE/MTP, ScheduleBatch.filter_batch
    may only drop requests FROM THE BACK. The spec_info (EagleDraftInput /
    EagleVerifyInput) carries cross-forward draft state -- topk_p, topk_index,
    hidden_states, and under overlap a deferred future_indices stub -- whose
    filter collapses to a PREFIX slice, so removing a middle request desyncs the
    remaining sessions' spec_info from the batch size (a [0,1] topk stub vs
    raw_bs>=1 makes the draft graph's grouped foreach_copy assert). Stock
    retract_decode obeys the identical rule: under spec it leaves the retraction
    order unsorted and pops only from the end (see
    _get_decode_retraction_order / "filter_batch API can only filter requests
    from the back"). Pure function of the replicated batch order -> the same
    decision on every rank of the communicator.
    """
    return spec_active and idx != n_reqs - 1


def spec_overlap_deferred_commit_hazard(
    spec_active: bool, enable_overlap: bool
) -> bool:
    """Return True when a spill under speculative decoding must take the
    POST-VERIFY SNAPSHOT path (spec + overlap) instead of the plain snapshot.

    THE HAZARD (MTP + overlap): a session's spec verify accepts accept_len(>=1)
    tokens and writes their KV into req_to_token DURING the forward, but
    kv_committed_len is bumped only later, in the deferred result processor
    (batch_result_processor: `kv_committed_len += num_accept_tokens`). Under
    overlap that deferred commit runs AFTER get_next_batch_to_run -> try_spill
    in the SAME loop iteration (scheduler.event_loop_overlap: pop_and_process
    follows the batch launch). So batch.seq_lens_cpu and kv_committed_len are
    both STALE at try_spill time -- they lag the physical row by the pending
    accept count. Snapshotting the stale length would leave the freshly accepted
    tokens' REAL slots sitting inside what becomes the sentinel tail, so the
    spill tick's hybrid attention (which requires the host tail to be all
    sentinels) aborts ("non-sentinel slot id in a spill-tick tail").

    ROBUST HANDLING (try_spill, this predicate as the GATE): when True, try_spill
    reads the TRUE post-verify length from the future map's published seq_lens
    (new_seq_lens_buf[req_pool_idx]) so the sentinel tail covers every real
    accepted slot, and frees the draft overhang from that true length (not the
    stale committed length, which would span the accepted slots). The deferred
    result processor remains the single writer of kv_committed_len. Plain decode
    / non-overlap (accept_len==1, committed bumped synchronously in
    prepare_for_decode, so seq_lens_cpu == committed) takes the unchanged plain
    snapshot -> False. Rank-uniform: spec_active and enable_overlap are both
    server-global, so every rank of the communicator decides identically.
    """
    return spec_active and enable_overlap


class SpillSnapshot(NamedTuple):
    length: int  # L: the snapshot length for boundary / sentinel / D2H backup
    free_from: int  # start of the draft-overhang free range [free_from, allocated)
    pre_valid: bool  # False -> caller must decline before touching the row


def spill_snapshot(
    spec_overlap: bool,
    stale_seq_lens: int,
    kv_committed_len: int,
    kv_allocated_len: int,
    true_L: int,
) -> SpillSnapshot:
    """Resolve the spill-snapshot length + draft-overhang free range, correcting
    for the spec+overlap deferred-commit lag (see
    spec_overlap_deferred_commit_hazard). Pure -> identical on every rank when
    fed replicated state.

    Non-spec / non-overlap: seq_lens_cpu == kv_committed_len (committed bumped
    synchronously in prepare_for_decode), so ``length`` is the plain
    ``stale_seq_lens`` and the overhang free is the classic
    ``[kv_committed_len, kv_allocated_len)``. Byte-identical to the pre-fix path.

    Spec + overlap: both seq_lens_cpu and kv_committed_len LAG the physical row
    by the pending (not-yet-committed) accept count; ``true_L`` (the future
    map's published post-verify seq_lens) is the real length. ``length`` is
    ``true_L`` so the sentinel tail covers every freshly accepted slot, and
    ``free_from`` is ``true_L`` so the overhang free ``[true_L, allocated)``
    reclaims ONLY the drafted-but-not-accepted slots -- never the accepted slots
    ``[kv_committed_len, true_L)``. ``pre_valid`` is False when ``true_L`` lags
    committed (an unseeded / stale published buffer) so the caller declines
    instead of corrupting the row.
    """
    if spec_overlap:
        return SpillSnapshot(
            length=true_L,
            free_from=true_L,
            pre_valid=true_L >= kv_committed_len,
        )
    return SpillSnapshot(
        length=stale_seq_lens,
        free_from=kv_committed_len,
        pre_valid=True,
    )


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
        "last_hidden",
        "draft_kv_k",
        "draft_kv_v",
        "draft_spill_boundary",
        "draft_spill_L",
        "tick_hiddens",
        "tick_hidden_start",
        "suppress_tick",
    )

    def __init__(self, req, region, spill_iter, wave, hysteresis):
        self.req = req
        self.batch = None  # bs=1 decode batch, built lazily at the first tick
        self.region = region
        self.spill_iter = spill_iter
        self.last_tick_iter = -(1 << 30)
        self.wave = wave
        self.hysteresis = hysteresis
        # ON-DEVICE MTP RESUME: the target hidden state of the LAST host-decoded
        # token, captured (cloned) from that spill tick's plain target forward
        # (capture_hidden_mode=LAST). Used at restore to re-seed the EAGLE/MTP
        # draft state so the resumed session rejoins the live spec decode batch
        # with a valid draft input instead of spec_info=None. GDN/Mamba state is
        # NOT recomputed: this hidden state comes from the single, correct target
        # forward the host-decode tick already ran (one legit SSM advance), so no
        # second advance / no corruption. None until the first tick captures one.
        self.last_hidden = None
        # DRAFT-KV BUNDLE (Stage 1): pinned-CPU snapshot of the spilled TAIL's
        # EAGLE/NEXTN draft KV shard, taken at spill and written back into the
        # restored slots on wave-back so the draft KV of [draft_spill_boundary,
        # draft_spill_L) survives the free+realloc round-trip (its slots would
        # otherwise be reused, and its target hidden states -- needed to rebuild
        # it -- are unrecoverable / GDN forbids a target re-forward). Covers ONLY
        # the positions that existed at spill; host-grown positions carry no
        # draft KV under a plain host tick (that gap is the spec-in-tick stage).
        # Shapes: [layer_num, tail_len, *k_row] / [..., *v_row] in store_dtype.
        self.draft_kv_k = None
        self.draft_kv_v = None
        self.draft_spill_boundary = 0
        self.draft_spill_L = 0
        # ON-DEVICE MTP RESUME (Option B backfill): per-tick target hidden
        # states of the HOST-GROWN positions [draft_spill_L, L), accumulated on
        # pinned CPU (one per plain host tick, in position order). At restore a
        # draft-ONLY extend over [draft_spill_L, L) consumes them to backfill the
        # draft KV that the plain host tick never wrote (the drafter did not run)
        # AND to produce the real resume seed. Forward-only / GDN-safe: these are
        # the hiddens of the single correct host-decode forward, never a target
        # re-forward. tick_hidden_start pins the first accumulated position.
        self.tick_hiddens = []
        self.tick_hidden_start = None
        # RESTORE-READINESS handshake (quiescence-trap fix): when a spilled
        # session is restore-ready but its last host-tick result is still
        # pending (last_batch is its own tick), one tick is suppressed so the
        # session goes quiescent next iteration and can restore with a settled
        # length. Without this, a SOLE-ACTIVE spilled session ticks every
        # iteration -> last_batch is always its tick -> restore defers forever
        # -> it finishes on host even though the device is free. Reset once the
        # tick picker honors it (one-shot suppression).
        self.suppress_tick = False


# Tick-cost estimate smoothing (EMA weight of a fresh sample) and the floor
# below which a measured tick cost is treated as "not yet measured".
_TICK_COST_EMA_ALPHA = 0.3
_MEAS_EPS_MS = 1e-3
# Finite transport sentinel for the "no measurement yet" +inf under a gloo
# all-reduce(MIN) (gloo dislikes inf). Any real headroom ratio is far below it.
_REDUCE_INF = 1e18


class SpillTickController:
    """SELF-CALIBRATING spill-tick cadence regulator (QoS: device fast, spill
    best effort). Replaces the fixed demand->interval setpoint with a cadence
    that EMERGES from runtime MEASUREMENT: it times the device SLACK (idle) per
    iteration and the marginal COST of one spill tick, and picks the frequency
    that maximises total (device + spilled) throughput.

    Measured signal (no guess, no interval table)
    ---------------------------------------------
    Each dwell window yields ONE ``headroom_ratio`` =
        (measured device idle time per iteration) / (measured cost of one tick)
      * ratio >= 1 : a whole tick fits in the idle the device would waste
                     anyway -> the tick is ~free (the slack evaporates
                     otherwise) -> drive the interval to 1 (tick every
                     iteration; the spilled session advances for free);
      * ratio -> 0 : the device is saturated, a tick fully steals a device
                     step -> back off to the FLOOR (tick only at the operator's
                     guaranteed minimum rate; protect main throughput);
      * between    : monotone interpolation floor..1.
    The endpoints are PHYSICAL (fits-in-idle vs fully-steals), not tuned
    constants; the interval is a consequence of the timing, not a lookup.

    Rank-uniformity of a MEASURED signal (the trap)
    -----------------------------------------------
    Device idle is NOT rank-uniform: a 5090 (sm120) and a 3080 (sm86) have
    different slack. If each rank chose its own interval from its own idle, the
    collective spill tick (a rank-uniform NCCL event) would DESYNC -> hang. So
    the control decision comes from a rank-uniform scalar: the per-rank
    ``headroom_ratio`` is combined with **all-reduce(MIN)**. A collective tick
    is only truly free when the LEAST-slack (bottleneck) rank has slack, and a
    tick costs THAT rank, so MIN is the binding-constraint semantics -- not
    mean (would over-tick when only the fast card is idle) nor max (would
    starve spill whenever any card is busy). The reduce (injected as
    ``reduce_fn``; identity for a single rank) runs ONLY at the dwell boundary,
    which is gated purely on the REPLICATED integer counters (window fill +
    dwell), so every rank enters the collective at the SAME iteration; MIN over
    one float is bit-exact, so the reduced ratio -- and every downstream
    pure-Python step -- is identical on every rank. The hot per-iteration path
    stays collective-free (the reduce fires once per dwell, ~1/64 iters).

    Anti-flap (kept -- matters MORE for a noisy measured signal)
    -----------------------------------------------------------
      * trailing WINDOW of rank-uniform binding ratios; decisions read the
        window mean, not one sample;
      * minimum DWELL (``min_dwell_iters`` scheduler iterations, rank-uniform)
        between two interval changes;
      * noise-scaled DEADZONE (``deadzone_sigma`` stderr of the ratio mean,
        widened into interval units) so a ratio straddling a rung does not
        oscillate;
      * one interval unit per decision (rate-limited, anti-overshoot);
      * bounds ``[1, floor_interval]``.

    Bootstrap: until the bottleneck rank has TIMED a tick (first tick still
    pending, or lazy-harvest lag) the ratio is undefined -> the controller
    HOLDS the conservative floor. A rank with no local measurement contributes
    +inf to the MIN (never lowers it), so a lagging rank cannot corrupt the
    binding ratio; the floor guarantees the first tick fires so the cost gets
    measured -> the loop self-bootstraps.
    """

    _NO_MEASUREMENT = float("inf")

    def __init__(
        self,
        floor_interval: int,
        window_size: int = 16,
        min_dwell_iters: int = 64,
        deadzone_sigma: float = 1.0,
        reduce_fn=None,
    ):
        self.min_interval = 1
        # Anti-starvation FLOOR = the operator QoS knob: the maximum iterations
        # a spilled session may go without a tick == its guaranteed minimum
        # progress rate. It is also the saturated-cadence and the absolute cap.
        self.floor_interval = max(1, int(floor_interval))
        self.window_size = max(1, int(window_size))
        self.min_dwell_iters = max(0, int(min_dwell_iters))
        self.deadzone_sigma = max(0.0, float(deadzone_sigma))
        # MIN-reduce of the per-rank headroom across the collective (identity
        # for a single rank / unit tests). Called ONLY at the dwell boundary.
        self._reduce = reduce_fn if reduce_fn is not None else (lambda r: r)

        # Trailing window of rank-uniform BINDING ratios (each a reduce output).
        self._ratio_window: deque[float] = deque(maxlen=self.window_size)
        # Per-rank measurement accumulators for the CURRENT dwell window. Wall
        # and busy are summed (not pre-divided): per-iteration busy is noisy
        # under the lazy-query harvest, but the window SUM recovers the true
        # utilisation and the phase error cancels.
        self._win_wall_ms = 0.0
        self._win_busy_ms = 0.0
        self._win_iters = 0
        self._tick_cost_ms: Optional[float] = None  # latest local estimate

        # Bootstrap: hold the conservative floor until a finite binding ratio.
        self._effective = self.floor_interval
        self._iters_since_change = self.min_dwell_iters  # first decision prompt
        self.n_changes = 0

    # -- per-iteration sampling ------------------------------------------
    def observe_sample(
        self, wall_ms: float, busy_ms: float, tick_cost_ms: Optional[float]
    ) -> None:
        """One MEASURED per-iteration sample: device wall time, device-busy
        time, and the current tick-cost estimate (None until a tick is timed).
        Accumulates wall/busy as window sums and advances the dwell clock.
        Called once per scheduler iteration on every rank (rank-uniform call
        site) so the dwell counters stay in lock-step."""
        self._win_wall_ms += max(0.0, float(wall_ms))
        self._win_busy_ms += max(0.0, float(busy_ms))
        self._win_iters += 1
        if tick_cost_ms is not None:
            self._tick_cost_ms = float(tick_cost_ms)
        self._iters_since_change += 1

    def _local_ratio(self) -> float:
        """Per-rank headroom = mean device idle per iter / tick cost. Returns
        the +inf sentinel when this rank has no valid tick cost yet (safe under
        MIN)."""
        if (
            self._win_iters <= 0
            or self._tick_cost_ms is None
            or self._tick_cost_ms <= _MEAS_EPS_MS
        ):
            return self._NO_MEASUREMENT
        idle_total = max(0.0, self._win_wall_ms - self._win_busy_ms)
        return (idle_total / self._win_iters) / self._tick_cost_ms

    def _reset_window(self) -> None:
        self._win_wall_ms = 0.0
        self._win_busy_ms = 0.0
        self._win_iters = 0

    def _desired(self, ratio: float) -> float:
        """Interval EMERGES from measured headroom: ratio>=1 (a whole tick fits
        in the wasted idle) -> 1 (tick freely); ratio->0 (saturated) -> floor
        (tick only at the guaranteed rate). Physical endpoints; monotone
        interpolation between."""
        r = 0.0 if ratio < 0.0 else (1.0 if ratio > 1.0 else ratio)
        return self.floor_interval - (self.floor_interval - self.min_interval) * r

    def _mean_and_margin(self) -> Tuple[float, float]:
        w = self._ratio_window
        n = len(w)
        mean = sum(w) / n
        if n < 2 or self.deadzone_sigma == 0.0:
            return mean, 0.0
        var = sum((x - mean) ** 2 for x in w) / (n - 1)
        stderr = math.sqrt(var / n)
        return mean, self.deadzone_sigma * stderr

    def maybe_update(self) -> bool:
        """At a rank-uniform dwell boundary: MIN-reduce the per-rank headroom,
        then step the effective interval one unit toward the measured target
        under the deadzone. Returns True on a change.

        RANK-UNIFORM COLLECTIVE ENTRY: the two gates below are purely replicated
        integer counters (window fill + dwell), NEVER a per-rank-variable
        condition, so all ranks reach the reduce at the SAME iteration."""
        if self._win_iters < max(1, self.window_size // 2):
            return False
        if self._iters_since_change < self.min_dwell_iters:
            return False
        binding = self._reduce(self._local_ratio())
        self._reset_window()
        if not math.isfinite(binding):
            # No rank has timed a tick yet -> hold the conservative floor.
            return False
        self._ratio_window.append(float(binding))
        mean, margin = self._mean_and_margin()
        desired = self._desired(mean)
        slope = float(self.floor_interval - self.min_interval)
        margin_iv = margin * slope  # ratio stderr -> interval units
        changed = False
        if (
            desired > self._effective + 0.5 + margin_iv
            and self._effective < self.floor_interval
        ):
            self._effective += 1
            changed = True
        elif (
            desired < self._effective - 0.5 - margin_iv
            and self._effective > self.min_interval
        ):
            self._effective -= 1
            changed = True
        if changed:
            self._iters_since_change = 0
            self.n_changes += 1
        return changed

    def effective_interval(self, fast_pressure: bool) -> int:
        """The cadence gate value for this iteration. Fast-lane pressure hard-
        pins to the floor (device maximally protected) without disturbing the
        regulator state -- fast > FCFS, always."""
        if fast_pressure:
            return self.floor_interval
        return self._effective


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

        # === DRAFT-KV BUNDLE (Stage 1): the spilled session's EAGLE/NEXTN draft
        # KV shard rides WITH the session (the ("draft_kv", ...) bundle element).
        # The draft pool shares the allocator + req_to_token virtual slot space
        # with the target (verified at bring-up), so restore's new_locs are valid
        # draft slot ids. Unlike the DCP-token-sharded TARGET pool, the draft pool
        # is NOT token-sharded: it holds the FULL token context on every rank
        # (M4, flashinfer_backend._sess_wire note; head-sharded [2,1,1] GQA
        # groups). So the draft-KV tail is snapshotted/restored in FULL (no owner
        # filter) -- a single tiny layer per rank, direct pinned-CPU stash rather
        # than a host-pool region. None when the server runs no spec algorithm.
        # DFLASH is excluded from spill sessions (short-ctx regime); only the
        # model-configured NEXTN/EAGLE-family drafter is bundled.
        self.draft_full_pool = None
        try:
            from sglang.srt.mem_cache.kv_cache_builder import get_draft_kv_pool

            spec_algo0 = getattr(scheduler, "spec_algorithm", None)
            dw = getattr(scheduler, "draft_worker", None)
            if (
                dw is not None
                and spec_algo0 is not None
                and not spec_algo0.is_none()
                and not spec_algo0.is_dflash_family()
            ):
                dpool = get_draft_kv_pool(
                    draft_worker=dw,
                    spec_algorithm=spec_algo0,
                    server_args=sa,
                )
                self.draft_full_pool = getattr(dpool, "full_kv_pool", dpool)
        except Exception as _e:  # noqa: BLE001
            logger.warning("kv-session-offload: draft-KV bundle disabled: %r", _e)
        if self.draft_full_pool is not None:
            logger.info(
                "kv-session-offload: draft-KV bundle armed (draft pool "
                "size=%d layer_num=%d head_num=%d).",
                getattr(self.draft_full_pool, "size", -1),
                getattr(self.draft_full_pool, "layer_num", -1),
                getattr(self.draft_full_pool, "head_num", -1),
            )

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

        # Self-calibrating spill-tick cadence regulator (default OFF -> the
        # cadence gate below reads the STATIC self.tick_interval, byte-
        # identical). When ON, maybe_take_tick reads
        # tick_controller.effective_interval() instead. See SpillTickController.
        self.tick_controller = None
        # Per-rank measurement state feeding the regulator (device-timer
        # reporter + host wall clock). Untouched when the regulator is off.
        self._busy_ms_accum = 0.0        # device-busy ms harvested since last iter
        self._tick_cost_ms = None        # EMA of a spill-tick forward's cost (ms)
        self._last_iter_wall = None      # host perf_counter of the last pre_schedule
        # A spill tick is a bs=1 PLAIN decode (spec_algo NONE) -> device-timer
        # category "decode". In a SPEC server the main forwards are
        # target_verify / extend / draft, so category=="decode" cleanly isolates
        # the tick's marginal cost. The regulator therefore targets a spec
        # server (the intended + validated config); in a non-spec server it can
        # not separate tick from main decode and conservatively holds the floor.
        _spec0 = getattr(scheduler, "spec_algorithm", None)
        self._regulator_spec_server = _spec0 is not None and not _spec0.is_none()
        if bool(getattr(sa, "kv_session_offload_tick_adaptive", False)):
            self.tick_controller = SpillTickController(
                floor_interval=int(sa.kv_session_offload_tick_floor),
                reduce_fn=self._min_reduce_headroom,
            )
            self._install_regulator_device_timer()

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
        if self.tick_controller is not None:
            logger.info(
                "kv-session-offload: SELF-CALIBRATING spill-tick cadence armed "
                "(floor=%d iters, bounds=[1,%d], window=%d, dwell=%d iters, "
                "sigma=%.1f, spec_server=%s) -- effective interval EMERGES from "
                "measured device idle vs tick cost; MIN-reduced per-rank "
                "headroom keeps every rank in lock-step",
                self.tick_controller.floor_interval,
                self.tick_controller.floor_interval,
                self.tick_controller.window_size,
                self.tick_controller.min_dwell_iters,
                self.tick_controller.deadzone_sigma,
                self._regulator_spec_server,
            )

    # -- self-calibrating regulator: measurement + rank-uniform reduce ----

    def _install_regulator_device_timer(self) -> None:
        """Wire the regulator's device-idle / tick-cost measurement onto the
        existing CUDA-event DeviceTimer (the head-split / decoupling fallback:
        Event(enable_timing=True) pairs harvested lazily via .query(), never a
        host sync -> the measurement never stalls the pipeline nor steals device
        time; samples are a few iters stale, which the trailing window absorbs).

        TARGET RUNNER ONLY -- deliberately NOT the draft runners. Instrumenting
        the EAGLE/NEXTN draft runners' forwards with timing events races with
        the draft-KV bundle's spill/restore stream choreography under sustained
        pool pressure (a co-located retraction + wave-back interleaving): the
        extra per-forward event enqueues on the draft stream surface a
        cross-stream free-before-read that a plain (uninstrumented) run does not
        hit -- reproduced as an async CUDA illegal-access at retract time,
        closed by CUDA_LAUNCH_BLOCKING, and gone entirely once the draft runners
        are left uninstrumented. The target timer already captures BOTH signals
        the regulator needs: the spill tick (a bs=1 plain "decode" forward on
        the target) and the dominant verify/extend busy. Draft-forward time is
        therefore counted as device IDLE -- a conservative, monotonic offset
        (headroom biased slightly up -> ticks a touch more eagerly): the
        RELATIVE response (idle falls as the verify batch grows) and the
        anti-starvation floor are both preserved, and under topk=1 the draft
        forward is a small fraction of the verify. Runs only when the regulator
        flag is on, so the default path is untouched.

        Reuses the metrics DeviceTimer on the target if one is already installed
        (adds a reporter); else creates and installs one on the target only."""
        from sglang.srt.utils.device_timer import DeviceTimer

        mr = self.model_runner
        timer = getattr(mr, "device_timer", None)
        if timer is None:
            mr.device_timer = DeviceTimer(reporter=self._device_timer_report)
        else:
            timer.add_reporter(self._device_timer_report)

    def _device_timer_report(self, t, category=None, **kw) -> None:
        """DeviceTimer reporter (t = elapsed SECONDS for one forward). Runs at
        lazy harvest time. Accumulates total device-busy ms since the last
        iteration and, in a spec server, EMAs the spill-tick cost from the
        "decode"-category forward (the bs=1 plain tick; see __init__ note)."""
        ms = t * 1000.0
        self._busy_ms_accum += ms
        if self._regulator_spec_server and category == "decode":
            self._tick_cost_ms = (
                ms
                if self._tick_cost_ms is None
                else (1.0 - _TICK_COST_EMA_ALPHA) * self._tick_cost_ms
                + _TICK_COST_EMA_ALPHA * ms
            )

    def _min_reduce_headroom(self, local_ratio: float) -> float:
        """all-reduce(MIN) of the per-rank headroom ratio over the TP collective
        (the group that runs the spill tick). MIN = binding-constraint: a
        collective tick is free only if the least-slack rank has slack, and a
        tick costs that rank. Called ONLY at the rank-uniform dwell boundary, so
        every rank enters this collective at the same iteration. MIN over one
        float is bit-exact -> the reduced value is identical on every rank. The
        +inf "no measurement" sentinel is transported as a large finite value
        (gloo dislikes inf) and restored."""
        grp = getattr(self.scheduler, "tp_cpu_group", None)
        if grp is None or torch.distributed.get_world_size(grp) <= 1:
            return local_ratio
        val = local_ratio if math.isfinite(local_ratio) else _REDUCE_INF
        t = torch.tensor([val], dtype=torch.float64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=grp)
        out = float(t.item())
        return float("inf") if out >= _REDUCE_INF else out

    # -- slot bookkeeping -------------------------------------------------

    def _slot_of(self, req) -> Optional["SpillSlot"]:
        return self.spills.get(req.req_pool_idx)

    def capture_tick_hidden(self, req_pool_idx: int, hidden: torch.Tensor) -> None:
        """ON-DEVICE MTP RESUME: record the LAST token's target hidden state
        from a spill tick's plain target forward (called from the worker, same
        process). Cloned so it survives the pooled forward buffer's reuse by the
        next forward. Overwritten every tick -> at restore the slot holds the
        hidden of the MOST RECENT host-decoded token, i.e. the correct draft
        seed position. Rank-uniform: every DCP rank runs the same tick forward
        and captures its own (replicated) hidden state. No-op if the session is
        no longer spilled (already reaped)."""
        slot = self.spills.get(req_pool_idx)
        if slot is None:
            return
        h = hidden.detach()
        slot.last_hidden = h.clone()
        # Option B: accumulate the host-grown positions' hiddens (pinned CPU) for
        # the restore-time draft-extend backfill. Only when the resume path is
        # armed and the draft-KV bundle is active; the host-finish path never
        # reads them. One token per plain tick -> appended in position order,
        # starting at draft_spill_L.
        if self.draft_full_pool is not None and resume_under_spec_enabled():
            if not slot.tick_hiddens:
                slot.tick_hidden_start = slot.draft_spill_L
            hc = h.to("cpu", copy=True)
            try:
                hc = hc.pin_memory()
            except RuntimeError:
                pass
            slot.tick_hiddens.append(hc)

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

        # SPEC (MTP/EAGLE) BACK-ONLY REMOVAL INVARIANT (see
        # spec_decline_non_back_spill): under speculative decoding
        # filter_batch may only drop the back-most request, so if the
        # policy-chosen victim is not last we decline and let stock retraction
        # (also back-only under spec) relieve the pressure.
        spec_active = not (
            batch.spec_algorithm is None or batch.spec_algorithm.is_none()
        )
        if spec_decline_non_back_spill(spec_active, idx, len(batch.reqs)):
            logger.debug(
                "kv-session-offload: spec active, victim rid=%s at idx=%d is not "
                "the back-most request (n=%d); declining spill (back-only under "
                "spec) -> stock retraction handles the pressure.",
                req.rid,
                idx,
                len(batch.reqs),
            )
            return False

        # POST-VERIFY SNAPSHOT (robust MTP spill under overlap). The deferred-
        # commit hazard: a spec verify writes the just-accepted tokens' KV into
        # req_to_token[committed:true_L] DURING the forward, but the deferred
        # result processor bumps kv_committed_len only LATER (next pop_and_process
        # in the same event-loop iteration, AFTER this try_spill). So at spill
        # time both batch.seq_lens_cpu and req.kv_committed_len are STALE -- they
        # lag the physical row by the pending accept count. Snapshotting the
        # stale length would sentinelise only [boundary, committed_stale) and
        # leave the freshly accepted real slots [committed_stale, true_L) sitting
        # inside what the spill tick treats as the sentinel tail -> the hybrid
        # attention's "all-sentinel tail" invariant breaks (flashinfer_backend
        # "non-sentinel slot id in a spill-tick tail").
        #
        # FIX: read the TRUE post-verify length straight from the future map's
        # published seq_lens (new_seq_lens_buf[req_pool_idx] == the seq_lens the
        # NEXT forward's resolve_seq_lens_cpu would gather -- see overlap_utils
        # publish/resolve_seq_lens_cpu). We only READ the persistent buffer here
        # (ordered after the in-flight forward's publish via _wait_forward_stream,
        # the same barrier the D2H backup below uses); we do NOT touch the
        # consume-once resolve path (publish_ready / _publish_fresh), so the
        # overlap seq-lens pipeline is untouched. Rank-uniform: new_seq_lens is a
        # replicated collective output, so every DCP rank reads the same true_L.
        #
        # We do NOT mirror the commit here: the deferred result processor stays
        # the single writer of kv_committed_len (it advances committed -> true_L
        # for this same session this same iteration, via the batch copy in the
        # result queue that still lists it). try_spill only corrects the geometry
        # snapshot and frees the true draft overhang. Gated on spec+overlap; the
        # plain-decode / non-overlap path is byte-identical (committed bumped
        # synchronously in prepare_for_decode, so seq_lens_cpu == committed).
        spec_overlap = spec_overlap_deferred_commit_hazard(
            spec_active, bool(getattr(self.scheduler, "enable_overlap", False))
        )
        stale_L = int(batch.seq_lens_cpu[idx].item())
        if spec_overlap:
            # Read the TRUE post-verify length; order after the in-flight
            # forward's publish (same barrier the D2H backup below uses) so the
            # published value is visible. Only READS the persistent buffer -- the
            # consume-once resolve path (publish_ready / _publish_fresh) is not
            # touched.
            self._wait_forward_stream()
            true_L = int(
                self.scheduler.future_map.new_seq_lens_buf[req.req_pool_idx].item()
            )
        else:
            true_L = stale_L

        snap = spill_snapshot(
            spec_overlap,
            stale_L,
            req.kv_committed_len,
            req.kv_allocated_len,
            true_L,
        )
        if not snap.pre_valid:
            # true_L lags committed: published buffer stale/unseeded (should not
            # happen once the session has decoded). Decline rather than corrupt.
            logger.warning(
                "kv-session-offload: spec+overlap true_L %d < committed %d for "
                "rid=%s; declining spill -> retraction.",
                true_L,
                req.kv_committed_len,
                req.rid,
            )
            return False
        L = snap.length
        if spec_overlap and getattr(req, "kv_arrival_seq", None) is not None:
            self._log(
                "kv-session-offload spec+overlap snapshot rid=%s: "
                "seq_lens_cpu=%d committed=%d allocated=%d true_L=%d "
                "pending_accept=%d (rank %d)",
                req.rid,
                stale_L,
                req.kv_committed_len,
                req.kv_allocated_len,
                true_L,
                true_L - req.kv_committed_len,
                self.dcp_rank,
            )

        # MTP / speculative decoding OVER-ALLOCATES the main-pool KV:
        # kv_allocated_len > L by the drafted-but-NOT-accepted slots at
        # row[L:allocated). Free that speculative overhang FIRST -- real device
        # slots, exactly what the stock retraction reclaims via
        # pop_overallocated_kv_cache -- so the session's bookkeeping is clean
        # (allocated == L) and the committed tail can be spilled normally. The
        # dropped draft is re-drafted on restore; the spill tick decodes plain
        # (no spec) while on host. Rank-uniform: the overhang count is replicated.
        #
        # snap.free_from is L (== true post-verify length) under spec+overlap so
        # the freshly accepted slots [committed_stale, true_L) are NOT freed;
        # outside spec+overlap it is kv_committed_len (== L), the original
        # [committed, allocated) free unchanged.
        if (
            req.kv_allocated_len > snap.free_from
            and (spec_overlap or L == req.kv_committed_len)
            and not getattr(req, "kv_overallocated_freed", False)
        ):
            over = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, snap.free_from : req.kv_allocated_len
            ]
            self.allocator.free(over.to(torch.int64))
            req.kv_allocated_len = snap.free_from
            req.kv_overallocated_freed = True

        # After reclaiming the overhang, allocated must equal L. Under
        # spec+overlap kv_committed_len legitimately still lags L by the pending
        # (not-yet-committed) accept count -- the deferred result processor
        # settles it this same iteration -- so it is validated only OFF the
        # spec+overlap path.
        committed_ok = spec_overlap or (L == req.kv_committed_len)
        if not committed_ok or L != req.kv_allocated_len:
            # Never spill a request whose slot bookkeeping we STILL do not
            # understand after reclaiming the speculative overhang -- stock
            # retraction handles it.
            logger.warning(
                "kv-session-offload: skip spill of rid=%s (L %d, "
                "committed %d, allocated %d, spec_overlap=%s); falling back to "
                "retraction.",
                req.rid,
                L,
                req.kv_committed_len,
                req.kv_allocated_len,
                spec_overlap,
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

        # 1b. DRAFT-KV BUNDLE: snapshot the tail's draft KV shard to pinned CPU
        #     BEFORE the free below reclaims these slots (they get reused, and
        #     the draft KV would be clobbered / unrecoverable). The draft pool is
        #     NOT DCP-token-sharded (full context on every rank, M4), so the
        #     snapshot is the FULL tail seg -- no owner filter, unlike the target
        #     backup above. Ordered after the in-flight forward via the
        #     _wait_forward_stream() already issued above.
        draft_k_snap = draft_v_snap = None
        if self.draft_full_pool is not None and resume_under_spec_enabled():
            # Only the resume path reads it back; the host-finish path (guard up)
            # never restores, so skip the D2H entirely there.
            draft_k_snap, draft_v_snap = self._draft_kv_snapshot(seg)

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
        if draft_k_snap is not None:
            slot.draft_kv_k = draft_k_snap
            slot.draft_kv_v = draft_v_snap
            slot.draft_spill_boundary = boundary
            slot.draft_spill_L = L
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

        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        sch = self.scheduler
        device = sch.device

        # The spill tick decodes PLAIN bs=1 (no speculative draft/verify) while
        # the session is host-resident: a host-streamed decode has no device
        # draft-KV, and spill_decode_alloc allocates exactly one sentinel slot
        # per tick. Force spec_algorithm=NONE on the spill batch (the device
        # session runs MTP normally; on restore the session rejoins the device
        # batch and MTP resumes from its resident GDN state). Correct, just
        # unaccelerated while spilled.
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=sch.req_to_token_pool,
            token_to_kv_pool_allocator=sch.token_to_kv_pool_allocator,
            tree_cache=sch.tree_cache,
            model_config=sch.model_config,
            enable_overlap=sch.enable_overlap,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.kv_session_spill_tick = True
        # ON-DEVICE MTP RESUME: under an active server spec algorithm, ask the
        # tick's plain target forward to also emit the LAST token's hidden state
        # (capture_hidden_mode=LAST). It is cloned into the slot after the
        # forward and used at restore to re-seed the draft state. Only affects
        # what the forward RETURNS, never the sampled token -> the host-decode
        # output stays byte-identical. Gated on spec active so the non-spec /
        # flag-OFF path emits nothing extra (byte-identical there too).
        spec_algo = getattr(sch, "spec_algorithm", None)
        if spec_algo is not None and not spec_algo.is_none():
            from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

            batch.capture_hidden_mode = CaptureHiddenMode.LAST
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

        # Self-calibrating cadence: every iteration, feed the regulator one
        # MEASURED sample -- host wall time, harvested device-busy time, and the
        # current tick-cost estimate -- then advance it. observe_sample /
        # maybe_update are called unconditionally on every rank (rank-uniform
        # call site) so the dwell counters and the boundary MIN-reduce stay in
        # lock-step. Done before any early return so the window stays continuous
        # across prefill / no-spill iterations and is warm when a spill starts.
        # No-op when the regulator is off (flag OFF -> byte-identical).
        if self.tick_controller is not None:
            now = time.perf_counter()
            wall_ms = (
                (now - self._last_iter_wall) * 1000.0
                if self._last_iter_wall is not None
                else 0.0
            )
            self._last_iter_wall = now
            busy_ms = self._busy_ms_accum
            self._busy_ms_accum = 0.0
            tick_cost = (
                self._tick_cost_ms
                if (self._tick_cost_ms is not None and self._tick_cost_ms > _MEAS_EPS_MS)
                else None
            )
            self.tick_controller.observe_sample(wall_ms, busy_ms, tick_cost)
            if self.tick_controller.maybe_update():
                tc = self.tick_controller
                headroom = tc._ratio_window[-1] if tc._ratio_window else float("nan")
                self._log(
                    "kv-session-offload: self-cal tick interval -> %d "
                    "(headroom=%.2f, tick_cost=%.2fms, spilled=%d, changes=%d)",
                    tc._effective,
                    headroom,
                    self._tick_cost_ms if self._tick_cost_ms is not None else float("nan"),
                    len(self.spills),
                    tc.n_changes,
                )

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
        # DEVICE-RESUME UNDER SPEC -- MILESTONE GUARD. A spilled session decodes
        # PLAIN on host: its EAGLE/MTP draft state (hidden_states / topk /
        # future_indices) is dropped at spill and never rebuilt while
        # host-resident. Merging it back into a LIVE spec decode batch therefore
        # contributes no valid EagleDraftInput, and ScheduleBatch.merge_batch ->
        # EagleDraftInput.merge_batch asserts on the missing spec_info. Rebuilding
        # the resumed session's draft state (a one-shot draft-extend from its
        # committed KV + last token) is the follow-up for true on-device MTP
        # resume; until then, under an active server spec algorithm we keep the
        # session on host through completion -- the validated, crash-free
        # spill + host-decode + finish path. Non-spec sessions restore to device
        # fully (unchanged / byte-identical). Rank-uniform: the server spec
        # algorithm is global, so every rank makes the same decision.
        spec_algo = getattr(self.scheduler, "spec_algorithm", None)
        if (
            spec_algo is not None
            and not spec_algo.is_none()
            and not resume_under_spec_enabled()
        ):
            return running_batch

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
        # Quiescence: a spilled session may only FINALIZE/one-shot-restore on an
        # iteration whose previous batch was NOT its own host tick, so the tick's
        # result (the last host-decoded token) is settled before it rejoins.
        # Incremental WAVE-BACK is exempt: it migrates the settled tail FRONT,
        # never the pending last position, so it runs every iteration to make
        # progress as space frees (FIX 2).
        if slot.batch is not None:
            quiescent = last_batch is not slot.batch
            L = int(slot.batch.seq_lens_cpu[0].item())
        else:
            # Spilled but never ticked yet: restorable once the victim's
            # last device result has been processed (>= one iteration).
            if self._iter_ct <= slot.spill_iter:
                return running_batch
            quiescent = True
            L = len(req.origin_input_ids) + len(req.output_ids) - 1

        row = self.req_to_token_pool.req_to_token[req.req_pool_idx, :L]
        boundary = int((row < self.host_base).sum().item())
        remaining = L - boundary
        avail = self.allocator.available_size()

        # RESTORE-READY: tail already fully drained, OR the whole tail fits now.
        drained = boundary >= L
        fits_now = avail >= remaining + self.restore_margin_tokens
        if drained or fits_now:
            if not quiescent:
                # Restore-ready but the last host-tick result is still pending.
                # Suppress this session's tick for one iteration so it goes
                # quiescent and can finalize with a settled length next iter.
                # THIS breaks the sole-active quiescence trap: without device
                # work the session would otherwise tick every iteration and
                # defer restore forever (measured: finishes on host with the
                # device idle). One-shot: the picker resets the flag.
                slot.suppress_tick = True
                return running_batch
            if drained:
                # Tail fully drained by earlier wave-back steps -> rejoin now.
                return self._finalize_restore(slot, running_batch, L)
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
        due = []
        for slot in self.spills.values():
            if slot.req.finished() or self._iter_ct <= slot.spill_iter:
                continue
            if slot.suppress_tick:
                # Restore-ready this iteration: skip its tick so it goes
                # quiescent (last_batch != tick) and finalizes next iteration.
                # One-shot -- clear the flag now (re-armed by _maybe_restore_flow
                # if still restore-ready). Rank-uniform: the flag is set from
                # replicated restore-readiness state on every rank.
                slot.suppress_tick = False
                continue
            due.append(slot)
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
        # Cadence gate value. STATIC self.tick_interval by default (byte-
        # identical). Adaptive regulator overrides it with a demand-driven,
        # rank-uniform effective interval; fast-lane pressure hard-pins to the
        # regulator maximum (device fully protected).
        interval = self.tick_interval
        if self.tick_controller is not None:
            fast_pressure = (
                self._fast_lane_pressure(running_batch.reqs)
                if device_has_work
                else False
            )
            interval = self.tick_controller.effective_interval(fast_pressure)
        if (
            device_has_work
            and (self._iter_ct - self._last_tick_iter) <= interval
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
        # Re-arm LAST-hidden capture EVERY tick: the persistent spill batch's
        # capture_hidden_mode is reset (to None) after each forward, so setting
        # it once at build time captures only the first tick. Without this the
        # per-tick target hidden that the resume seed / draft-KV backfill needs
        # is emitted on ~1 tick in N (measured), starving the backfill.
        if self.draft_full_pool is not None and resume_under_spec_enabled():
            from sglang.srt.model_executor.forward_batch_info import (
                CaptureHiddenMode,
            )

            batch.capture_hidden_mode = CaptureHiddenMode.LAST
        slot.last_tick_iter = self._iter_ct
        self._last_tick_iter = self._iter_ct
        return batch

    # -- restore ------------------------------------------------------------

    def _draft_kv_snapshot(self, seg):
        """DRAFT-KV BUNDLE (Stage 1): clone the draft KV shard at the tail's
        virtual slots ``seg`` into pinned CPU, stacked ``[layer_num, n, *row]``
        per K and V. Raw ``store_dtype`` (fp8 stored as uint8) keeps it byte
        exact. FULL tail -- the draft pool is not DCP-token-sharded (M4), every
        rank holds all token positions; no owner filter. Ordered after the
        in-flight forward by the caller's ``_wait_forward_stream()``."""
        dp = self.draft_full_pool
        assert getattr(dp, "page_size", 1) == 1, (
            "kv-session-offload draft-KV bundle: draft pool page_size must be 1 "
            f"(got {getattr(dp, 'page_size', None)})"
        )
        seg64 = seg.to(torch.int64)
        cap = int(dp.k_buffer[0].shape[0])
        if seg64.numel():
            hi = int(seg64.max().item())
            lo = int(seg64.min().item())
            assert 0 <= lo and hi < cap, (
                "kv-session-offload draft-KV snapshot: slot id out of draft pool "
                f"bounds [0,{cap}): min={lo} max={hi} (rank {self.dcp_rank})"
            )
        ln = int(dp.layer_num)
        k_layers, v_layers = [], []
        for l in range(ln):
            k_layers.append(dp.k_buffer[l][seg64].to("cpu", copy=True))
            v_layers.append(dp.v_buffer[l][seg64].to("cpu", copy=True))
        k_cpu = torch.stack(k_layers, dim=0)
        v_cpu = torch.stack(v_layers, dim=0)
        try:
            k_cpu = k_cpu.pin_memory()
            v_cpu = v_cpu.pin_memory()
        except RuntimeError:
            pass  # pinning is a perf nicety; correctness holds either way
        return k_cpu, v_cpu

    def _draft_kv_restore_block(self, slot, new_locs, boundary: int, hi: int):
        """DRAFT-KV BUNDLE (Stage 1): write the snapshot back into the restored
        slots for positions ``[boundary, hi)`` that overlap the snapshot range
        ``[draft_spill_boundary, draft_spill_L)``. ``new_locs`` is positional for
        ``[boundary, hi)``. Host-grown positions (>= draft_spill_L) carry NO
        draft KV under a plain host tick -- that residual gap is closed by the
        spec-in-spill-tick stage, not here."""
        if slot.draft_kv_k is None or self.draft_full_pool is None:
            return
        sb = slot.draft_spill_boundary
        sL = slot.draft_spill_L
        lo = max(boundary, sb)
        hio = min(hi, sL)
        if hio <= lo:
            return
        dp = self.draft_full_pool
        dst = new_locs[lo - boundary : hio - boundary].to(torch.int64)
        s0, s1 = lo - sb, hio - sb
        cap = int(dp.k_buffer[0].shape[0])
        if dst.numel():
            dhi = int(dst.max().item())
            dlo = int(dst.min().item())
            assert 0 <= dlo and dhi < cap, (
                "kv-session-offload draft-KV restore: slot id out of draft pool "
                f"bounds [0,{cap}): min={dlo} max={dhi} positions[{lo},{hio}) "
                f"(rank {self.dcp_rank})"
            )
        assert 0 <= s0 <= s1 <= int(slot.draft_kv_k.shape[1]), (
            "kv-session-offload draft-KV restore: snapshot slice out of range "
            f"[{s0},{s1}) vs len {int(slot.draft_kv_k.shape[1])}"
        )
        verify = draft_kv_verify_enabled()
        for l in range(int(dp.layer_num)):
            k_src = slot.draft_kv_k[l, s0:s1].to(dp.k_buffer[l].device)
            v_src = slot.draft_kv_v[l, s0:s1].to(dp.v_buffer[l].device)
            dp.k_buffer[l][dst] = k_src
            dp.v_buffer[l][dst] = v_src
            if verify:
                ok_k = torch.equal(dp.k_buffer[l][dst], k_src)
                ok_v = torch.equal(dp.v_buffer[l][dst], v_src)
                assert ok_k and ok_v, (
                    "kv-session-offload draft-KV bundle S1 self-check FAILED "
                    f"rid={slot.req.rid} layer={l} k_ok={ok_k} v_ok={ok_v}"
                )
        if verify:
            self._log(
                "kv-session-offload draft-KV bundle S1 self-check PASS: rid=%s "
                "positions[%d,%d) (snapshot[%d,%d)) layers=%d slots=%d (rank %d)",
                slot.req.rid,
                lo,
                hio,
                s0,
                s1,
                int(dp.layer_num),
                int(dst.numel()),
                self.dcp_rank,
            )

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
        # DRAFT-KV BUNDLE: write the snapshotted draft KV of the overlapping
        # pre-spill positions back into these same new_locs (draft shares the
        # virtual slot space with the target, so new_locs are valid draft slots).
        self._draft_kv_restore_block(slot, new_locs, boundary, L)
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
        # DRAFT-KV BUNDLE: restore this block's overlapping pre-spill draft KV
        # into the same new_locs (positional for [boundary, hi_pos)).
        self._draft_kv_restore_block(slot, new_locs, boundary, hi_pos)
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
        # try_spill freed the speculative draft overhang and set
        # kv_overallocated_freed=True. The resumed session decodes spec again and
        # re-allocates a fresh overhang; clear the flag so the finish path frees
        # that new overhang exactly once (else pop_overallocated_kv_cache asserts
        # "Overallocated KV cache already freed"). No-op for non-spec restore
        # (the flag was never set there) -> byte-identical.
        req.kv_overallocated_freed = False
        if slot.batch is None:
            # Never ticked while spilled: build the decode batch now so the
            # session can be merged back like any resumed decode request.
            slot.batch = self._build_spill_batch(req)
        batch = slot.batch
        # Back on device: this batch (or its reqs inside running_batch) must
        # take the normal decode path again.
        batch.kv_session_spill_tick = False
        # ON-DEVICE MTP RESUME: under an active server spec algorithm, re-seed
        # the session's EAGLE/MTP draft state so it rejoins the LIVE spec decode
        # batch with a valid EagleDraftInput (future_indices) instead of the
        # spec_algorithm=NONE spill batch's spec_info=None (which trips
        # EagleDraftInput.merge_batch's future_indices assert). Non-spec restore
        # is unchanged (byte-identical). Rank-uniform: every DCP rank captured
        # its own replicated tick hidden and runs the identical seed here.
        spec_algo = getattr(self.scheduler, "spec_algorithm", None)
        if spec_algo is not None and not spec_algo.is_none():
            # The spill batch was built spec_algorithm=NONE for the plain host
            # tick; flip it back to the server algorithm so the resumed session
            # runs a REAL spec decode (the post-forward `batch.spec_info =
            # next_draft_input` handoff is gated on `not spec_algorithm.is_none()`
            # -- leaving it NONE strands a stale/extend spec_info that later trips
            # resolve_seq_lens_cpu's future_indices read).
            batch.spec_algorithm = spec_algo
            self._seed_resumed_draft_state(slot, L)
        # Restore hygiene (defensive; restore is infrequent so the sync is
        # cheap): the rejoined row [0, L) must be fully real slots -- a leftover
        # sentinel would later be freed by finish/retract and corrupt the
        # allocator (the retract x spill-sentinel class). Fail loud here rather
        # than as a downstream CUDA illegal-access.
        row_chk = self.req_to_token_pool.req_to_token[req.req_pool_idx, :L]
        n_sent = int((row_chk >= self.host_base).sum().item())
        assert n_sent == 0, (
            "kv-session-offload RESTORE row not clean: rid=%s L=%d has %d "
            "sentinel(s) after restore (rank %d)"
            % (req.rid, L, n_sent, self.dcp_rank)
        )
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

    def _seed_resumed_draft_state(self, slot, L: int) -> None:
        """ON-DEVICE MTP RESUME (Option B -- draft-only backfill, no target
        re-forward, GDN-safe).

        Re-prime the resumed session's EAGLE/MTP draft state and relay it
        through the overlap FutureMap so the next iteration's
        ``_resolve_spec_extras`` gathers it by ``future_indices``. Two segments
        of the committed prefix are made gap-free before the session rejoins the
        LIVE spec batch:

          * ``[0, L_spill)`` -- restored device-resident via the draft-KV bundle
            (Stage 1): head kept its slots, the spilled tail's draft KV was
            written back byte-exact into the restored slots.
          * ``[L_spill, L)`` -- the HOST-GROWN positions, whose draft KV the
            plain host tick never wrote (the drafter did not run). A draft-ONLY
            EXTEND over this range (``backfill_draft_extend_for_resume``) fills
            it from the per-tick captured target hidden states and produces the
            REAL seed (topk_p/topk_index/recurrent hidden of the last position).
            The TARGET is NEVER re-forwarded -> the resident GDN/Mamba state is
            advanced exactly once (on the host tick), never twice.

        DFLASH is excluded from spill sessions (short-ctx regime): the bundle is
        only armed for the model-configured NEXTN/EAGLE-family drafter, so this
        path is generic, not NEXTN-hardcoded.

        If the backfill is unavailable (gap == 0, no draft pool, or a tick-count
        mismatch) it falls back to a seed-only re-prime from the last captured
        hidden -- correct rejoin, with the small host-grown draft-KV hole
        recovering over the next rounds.

        Consume-once / rank-uniformity: the backfill extend is a collective
        draft forward run identically on every DCP rank (replicated inputs); the
        stash + publish is an off-forward-stream FutureMap write (``publish``
        chains ``publish_ready`` so no in-flight fence is dropped).
        """
        seed = None
        prefix_len = int(slot.tick_hidden_start) if slot.tick_hiddens else L
        gap = L - prefix_len
        nh = len(slot.tick_hiddens)
        # tick_hiddens[i] is the target hidden of position (tick_hidden_start+i),
        # captured one-per-plain-tick. The backfill needs positions
        # [tick_hidden_start, L) == indices [0, gap). nh >= gap suffices (a
        # trailing extra from a pending-result lag is ignored); nh < gap means a
        # capture was missed -> fall back to the seed-only re-prime.
        if self.draft_full_pool is not None and 0 < gap <= nh:
            seed = self._backfill_resume_seed(slot, prefix_len, L, gap)
        elif self._iter_ct and (gap > 0):
            self._log(
                "kv-session-offload MTP resume: backfill unavailable rid=%s "
                "gap=%d n_hiddens=%d -> seed-only (rank %d)",
                slot.req.rid, gap, nh, self.dcp_rank,
            )
        if seed is None:
            seed = self._seed_only_resume(slot, L)
        self._publish_resume_seed(slot, L, seed)

    def _seed_only_resume(self, slot, L: int):
        """Fallback re-prime from the last captured tick hidden (no backfill):
        a valid one-row EAGLE seed. Correct rejoin; the residual host-grown
        draft-KV hole recovers over the next rounds."""
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        req = slot.req
        device = self.scheduler.device
        assert slot.last_hidden is not None, (
            "kv-session-offload MTP resume: no captured tick hidden for "
            f"rid={req.rid} (should have been gated in _maybe_restore_flow)"
        )
        last_tok = int(req.output_ids[-1])
        return EagleDraftInput(
            topk_p=torch.ones((1, 1), dtype=torch.float32, device=device),
            topk_index=torch.tensor([[last_tok]], dtype=torch.int64, device=device),
            hidden_states=slot.last_hidden,
            bonus_tokens=torch.tensor([last_tok], dtype=torch.int64, device=device),
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

    def _backfill_resume_seed(self, slot, prefix_len: int, L: int, gap: int):
        """Run the draft-only EXTEND over the host-grown tail [prefix_len, L)
        to backfill its draft KV and return the real seed. Returns None (caller
        falls back) on any structural mismatch."""
        req = slot.req
        device = self.scheduler.device
        dw = getattr(self.scheduler, "draft_worker", None)
        if dw is None or not hasattr(dw, "backfill_draft_extend_for_resume"):
            return None
        full = list(req.origin_input_ids) + list(req.output_ids)
        # Committed length invariant: L == len(full) - 1 (the last entry is the
        # next-decode input token, i.e. the extend tail).
        if len(full) != L + 1 or prefix_len < 0 or prefix_len >= L:
            self._log(
                "kv-session-offload MTP resume backfill SKIP: rid=%s L=%d "
                "prefix=%d len(full)=%d -> seed-only fallback",
                req.rid,
                L,
                prefix_len,
                len(full),
            )
            return None
        self._wait_forward_stream()
        # Target hiddens of positions [prefix_len, L) == the first `gap`
        # captures (one per tick, in position order). Any trailing extra is a
        # pending-result lag and is dropped.
        target_hiddens = torch.cat(
            [h.to(device, non_blocking=True) for h in slot.tick_hiddens[:gap]],
            dim=0,
        )
        next_token_ids = torch.tensor([full[L]], dtype=torch.int64, device=device)
        batch = self._build_backfill_extend_batch(req, prefix_len, L, full)
        seed = dw.backfill_draft_extend_for_resume(
            batch, target_hiddens, next_token_ids
        )
        self._log(
            "kv-session-offload MTP RESUME backfill: rid=%s L=%d prefix=%d "
            "gap=%d (draft-only extend, target NOT re-forwarded) (rank %d)",
            req.rid,
            L,
            prefix_len,
            gap,
            self.dcp_rank,
        )
        return seed

    def _build_backfill_extend_batch(self, req, prefix_len: int, L: int, full):
        """Build a bs=1 DRAFT extend ScheduleBatch over [prefix_len, L): prefix
        (cached, draft KV present) + gap new tokens. out_cache_loc points at the
        session's restored slots so the draft KV lands where the shared
        req_to_token already maps (1-to-1 with the target)."""
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

        sch = self.scheduler
        device = sch.device
        gap = L - prefix_len
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=sch.req_to_token_pool,
            token_to_kv_pool_allocator=sch.token_to_kv_pool_allocator,
            tree_cache=sch.tree_cache,
            model_config=sch.model_config,
            enable_overlap=sch.enable_overlap,
            spec_algorithm=sch.spec_algorithm,
        )
        batch.forward_mode = ForwardMode.EXTEND
        batch.req_pool_indices = torch.tensor(
            [req.req_pool_idx], dtype=torch.int64, device=device
        )
        batch.req_pool_indices_cpu = torch.tensor(
            [req.req_pool_idx], dtype=torch.int64
        )
        batch.input_ids = torch.tensor(
            full[prefix_len:L], dtype=torch.int64, device=device
        )
        batch.seq_lens = torch.tensor([L], dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor([L], dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor([L], dtype=torch.int32, device=device)
        batch.seq_lens_sum = L
        batch.extend_lens = [gap]
        batch.prefix_lens = [prefix_len]
        batch.extend_num_tokens = gap
        batch.extend_logprob_start_lens = [gap]
        batch.out_cache_loc = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, prefix_len:L
        ].to(torch.int64)
        batch.multimodal_inputs = [getattr(req, "multimodal_inputs", None)]
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, sch.model_config.vocab_size
        )
        return batch

    def _publish_resume_seed(self, slot, L: int, seed) -> None:
        """Relay the resume seed into the FutureMap at this rpi and publish the
        (unchanged) length so the next iter's resolve gathers a FRESH row, then
        arm the merge via the overlap future_indices branch."""
        from sglang.srt.managers.overlap_utils import RelayPayload

        req = slot.req
        batch = slot.batch
        device = self.scheduler.device
        future_indices = torch.tensor(
            [req.req_pool_idx], dtype=torch.int64, device=device
        )
        fmap = self.scheduler.future_map
        fmap.stash(future_indices, RelayPayload.from_draft_input(seed))
        fmap.publish(future_indices, batch.seq_lens)
        seed.future_indices = future_indices
        batch.spec_info = seed
        slot.last_hidden = None
        slot.tick_hiddens = []
        self._log(
            "kv-session-offload MTP RESUME seed published: rid=%s L=%d (rank %d)",
            req.rid,
            L,
            self.dcp_rank,
        )

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
        # RETRACT/ABORT-PATH RACE FIX: a session that ends WHILE spilled may
        # still have IN-FLIGHT device work touching the very slots freed below:
        # a streamed wave-back / prefetch H2D copy on ``_sess_copy_stream`` (it
        # restores the tail into freshly allocated device slots and reads the
        # region) and/or a spill-tick forward on ``forward_stream`` (it reads
        # the retained device head every layer). This method runs on the
        # schedule stream, and the retract path frees + INSTANTLY REUSES the
        # slots (evict_from_tree_cache). Without ordering the frees after those
        # streams, the reused slots collide with the still-running copy/forward
        # -> a free-before-copy-completes race -> an async CUDA illegal memory
        # access that surfaces at a LATER kernel (observed as the crash inside
        # retract_decode's torch.unique; CUDA_LAUNCH_BLOCKING=1 hid it). Order
        # the frees after BOTH streams (event waits; no host sync). Cheap and
        # only on this spilled-req cleanup path; the stock (non-spilled) free
        # path is untouched. Mirrors the try_spill quiesce (backup before free).
        # Guarded on a real copy Stream so the CPU unit path (mock backend)
        # skips it; on GPU both streams exist.
        copy_stream = getattr(self.backend, "_sess_copy_stream", None)
        if isinstance(copy_stream, torch.cuda.Stream):
            self._wait_forward_stream()
            torch.cuda.current_stream().wait_stream(copy_stream)
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
