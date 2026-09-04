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

from sglang.srt.environ import envs
from sglang.srt.managers.admission_limiter import (
    current_admission_limiter,
    spill_session_cap,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

# Per-process singleton (one scheduler per TP-rank process). Registered by
# KVSessionOffloadManager.__init__; consumed by ScheduleBatch's spill-tick
# decode allocation without a scheduler back-reference.
_MANAGER: Optional[KVSessionOffloadManager] = None


def get_kv_session_offload_manager() -> KVSessionOffloadManager:
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


def owned_counts_weighted(residues: torch.Tensor, prefix: List[int]) -> List[int]:
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


def spill_tail_rows_max_over_ranks(
    seg: torch.Tensor,
    *,
    mode: str,
    S: int,
    cp_prefix: List[int],
    dcp_size: int,
    boundary: int,
    L: int,
) -> int:
    """Host rows the WIDEST rank needs for the tail segment ``[boundary, L)``.

    The number of host rows a spill claims is rank-LOCAL: under the weighted
    owner rule rank r takes the tail slots whose residue falls in
    ``[cp_prefix[r], cp_prefix[r+1])``, and those windows have different widths
    under uneven DCP (a [1, 3] token vector puts three quarters of the tail on
    one rank). Comparing THIS rank's count against the replicated region size
    therefore yields a rank-DEPENDENT spill verdict, and the two outcomes --
    spill vs. stock retraction -- are different collective sequences. This
    fork's rule (module header: "RANK-UNIFORMITY ... divergence == NCCL hang,
    not a wrong number") makes that a hang, not a wrong number, so the region
    check has to be the ANY-rank check: it is the widest rank that decides
    whether one region can hold the tail.

    NO COLLECTIVE is needed for it. Every input here is REPLICATED: ``seg`` is
    a slice of the ``req_to_token`` row, which carries GLOBAL slot ids and is
    identical on every rank (the weighted rule derives ownership FROM the
    global id -- see ``compact_weighted``, the inverse of the ``_dcp_masked_write``
    packing), and ``cp_prefix`` / ``dcp_size`` / ``boundary`` / ``L`` are
    replicated geometry. So each rank can compute EVERY rank's count locally --
    the same trick ``_restore`` already uses when it sizes the owner-matched
    allocation for all ranks at once.

    Pure and deterministic: identical on every rank by construction.
    Single-rank ("plain", ``dcp_size == 1``) returns exactly the local count,
    so that path is byte-identical to the pre-fix predicate."""
    if mode == "weighted":
        residues = (seg.to(torch.int64) % S).contiguous()
        return max(owned_counts_weighted(residues, cp_prefix))
    if mode == "even":
        # Positional rule keyed on the ABSOLUTE position (same convention as
        # ``owned_device_indices(..., pos_offset=boundary)``): the counts over
        # [boundary, L) are the prefix counts of L minus those of boundary.
        head = owned_counts_even(int(boundary), dcp_size)
        whole = owned_counts_even(int(L), dcp_size)
        return max(w - h for w, h in zip(whole, head))
    assert mode == "plain"
    return int(seg.numel())


def prefill_spill_deep_reject_reason(
    spec_active: bool,
    spec_in_tick_ready: bool,
    resume_under_spec: bool,
    dflash_prefill_append: bool,
    backend_write_hook: bool = True,
) -> Optional[str]:
    """The condition that blocks PS2, or ``None``.

    THE FIRST CONDITION IS NOT ABOUT SPECULATION AND IT IS THE ONE THAT KILLED
    AN INSTANCE (register C26). PS2 hands the extend a ``out_cache_loc`` full
    of HOST SENTINELS -- see ``spill_extend_alloc``, which returns
    ``make_sentinels(...)`` -- and exactly one thing in the tree diverts that
    tensor away from the KV write: ``_dcp_write_scatter``'s
    ``_sess_prefill_owner_write`` branch. That branch is reachable ONLY from
    the token-sharded DCP lane (``forward_extend`` enters it under
    ``if self.uneven_dcp``). On plain TP the backend still BUILDS the
    prefill-spill state and nothing complains, but ``forward_extend`` falls
    through to the stock ``set_kv_buffer``, and the sentinels go straight into
    ``store_kvcache``:

        jit_kernel/csrc/elementwise/kvcache.cuh:112
          Assertion `index >= 0 && index < size_limit` failed

    Measured, not inferred: with ``host_base=4097`` and a 4096-row allocator,
    a request with ``boundary=2620 L=3012`` writes indices 6717..7108 against
    a ``size_limit`` of ~4097 -- every one of the 392 rows out of bounds, on
    layer 0, on both ranks. The admission gate had no way to know, because it
    was never told which backend it was admitting onto.

    So PS2 declines when the backend has no born-spilled EXTEND write hook.
    The decline is the pre-PS2 behaviour exactly -- the request stays queued
    and the fast lane retries it -- and it is RANK-UNIFORM by construction:
    ``_sess_mode`` is derived purely from ``dcp_size``, which is replicated
    boot configuration, fixed before the first forward and identical on every
    rank. No collective, and the refusal is a NON-ADMISSION rather than a
    rank-local skip around a collective, so register law 14 is satisfied.

    The default is ``True`` because the pure-function pins below exercise the
    speculation conditions and predate this one; the single production caller
    passes it explicitly and must keep doing so.

    THE REST OF THIS DOCSTRING IS ABOUT SPECULATION.

    The mechanical problem: the target prefill is followed by a DRAFT extend
    that reuses the SAME ScheduleBatch and hence the same ``out_cache_loc``
    (``eagle_worker_v2._draft_extend_for_prefill`` -> ``ForwardBatch.init_new(
    batch, self.draft_runner)``). For a born-spilled prompt that tensor holds
    HOST SENTINELS, which the draft pool -- a separate pool sharing the target's
    slot id space -- cannot address, so the draft write would land out of
    bounds.

    That is a PLACEMENT problem, and for the common configuration it dissolves:
    the draft extend of a born-spilled prefill produces nothing anyone reads, so
    it is skipped outright (``EagleDraftWorkerBase.born_spilled_stub_draft_
    input`` carries the argument). The three conditions below are the ones under
    which the prompt's draft KV IS read again, and only there does PS2 decline:

      * ``spec_in_tick_ready`` -- the spilled session drafts on device during
        the spill tick and attends the prompt's draft KV through the
        ``spec_in_tick_draft_pre`` req_to_token surgery;
      * ``resume_under_spec`` (KVSO_RESUME=1) -- the session waves back and
        rejoins the live spec decode batch, whose draft attends the prompt
        positions on device;
      * ``dflash_prefill_append`` -- DFLASH appends the prompt's context
        features to its own draft KV in ``dflash_worker_v2.prefill_after_
        target``, a second write path that the skip above does not cover. This
        is a BOOT-time predicate on purpose: under cross-algorithm switching
        that append runs on every prefill regardless of which rung is active
        (``cross_algo_worker.py:1737``), so keying it to the active rung would
        miss it.

    Lifting the first two is the same work: give the born-spilled prefill's
    draft KV a home outside the target's slot space (a draft-pool-only carve
    plus the existing device-resident ``draft_dev_k/v`` snapshot format), which
    is what ``--draft-kv-layout`` (#108) is about.

    Rank-uniform: every input is replicated config or a rank-0-broadcast rung
    id (``_effective_spec_algorithm``)."""
    if not backend_write_hook:
        return (
            "the attention backend runs in plain-TP mode and has no "
            "born-spilled EXTEND write hook (_sess_prefill_owner_write is "
            "reachable only from _dcp_write_scatter, i.e. only on the "
            "token-sharded DCP lane). Admitting PS2 here sends host sentinel "
            "rows into store_kvcache and asserts device-side, killing the "
            "instance (register C26). Run kv-session-offload on a DCP lane to "
            "use PS2, or write the plain-TP extend twin of "
            "_sess_forward_decode_plain."
        )
    if not spec_active:
        return None
    if spec_in_tick_ready:
        return (
            "spec-in-tick is armed (--kv-session-offload-spec-in-tick + "
            "KVSO_ALLOW_SPEC=1): the spilled session drafts on device and "
            "attends the prompt's draft KV, which a born-spilled prefill never "
            "wrote. Drop spec-in-tick to use PS2, or vice versa."
        )
    if resume_under_spec:
        return (
            "resume-under-spec is armed (KVSO_RESUME=1): a session that waves "
            "back rejoins the live spec batch and its draft attends the prompt "
            "positions, which a born-spilled prefill never wrote. Drop "
            "KVSO_RESUME to use PS2, or vice versa."
        )
    if dflash_prefill_append:
        return (
            "a DFLASH-family drafter is configured (primary or cross-algorithm "
            "secondary): its prefill append "
            "(dflash_worker_v2.prefill_after_target) writes the born-spilled "
            "prompt's draft KV through a second path that the skip does not "
            "cover."
        )
    return None


def prefill_spill_deep_gate(
    prefill_spill: bool,
    spec_active: bool,
    spec_in_tick_ready: bool = False,
    resume_under_spec: bool = False,
    dflash_prefill_append: bool = False,
    backend_write_hook: bool = True,
) -> bool:
    """PS2 MASTER GATE. ``--kv-session-offload-prefill`` AND no blocking spec
    condition (``prefill_spill_deep_reject_reason`` carries the reasoning and
    the exact wording of each block).

    PS1 born-spilled admission is unaffected by all of this -- it prefills on
    device and rides the decode-OOM spill, so its draft extend is a normal
    device write.

    Rank-uniform: every input is replicated config."""
    if not prefill_spill:
        return False
    return (
        prefill_spill_deep_reject_reason(
            spec_active,
            spec_in_tick_ready,
            resume_under_spec,
            dflash_prefill_append,
            backend_write_hook,
        )
        is None
    )


def prefill_spill_deep_ok(
    free_regions: int,
    born_input_tokens: int,
    rem_total_tokens: int,
    input_tokens: int,
    rem_chunk_tokens: Optional[int],
    region_tokens: int,
) -> bool:
    """PS2 (deep prefill-spill) ADMISSION VERDICT -- pure, rank-uniform.

    PS1-V1a admits a prompt born-spilled when its INPUT still fits the device
    transiently (``born_input_tokens < rem_total_tokens``): it prefills on
    device and rides the existing decode-OOM spill to host. PS2 is the strict
    COMPLEMENT of that window -- the input does not even transiently fit, so
    the prefill must never materialize device KV slots at all.

    Every input is replicated or already min-reduced (``rem_total_tokens`` /
    ``rem_chunk_tokens`` carry the ``dcp_avail_deficit`` pin, ``free_regions``
    is the replicated free-region count, the token counts are request
    metadata), so the verdict is IDENTICAL on every DCP rank without a
    collective (U8: no asynchronous region claim, no rank-local pool read).

    Conditions, all hard:
      * a free host region exists (the whole session lives in ONE region);
      * PS1's window does NOT apply (``born_input_tokens >= rem_total_tokens``)
        -- PS2 never takes a prompt PS1 can serve, so the validated PS1 path
        keeps its exact behaviour;
      * ONE CHUNK ONLY (``input_tokens <= rem_chunk_tokens``): without PS3
        (host-prefix extend read) chunk i+1 would attend chunk i's sentinel
        rows, i.e. garbage. A single non-chunked extend attends only its own
        RAGGED keys plus the device-resident radix prefix, so it needs no
        host read at all. ``rem_chunk_tokens is None`` means chunked prefill
        is off -> the whole prompt is one extend, which also qualifies;
      * the spilled tail fits one region.
    """
    if free_regions <= 0:
        return False
    if born_input_tokens < rem_total_tokens:
        return False  # PS1's window -- leave it to the validated PS1 path
    if rem_chunk_tokens is not None and input_tokens > rem_chunk_tokens:
        return False  # would be CHUNKED -> needs PS3
    if input_tokens <= 0 or input_tokens > region_tokens:
        return False
    return True


def prefill_stage_tokens(chunk_tokens: int, split_factor: int, max_ratio: int) -> int:
    """RANK-UNIFORM size (in tokens) of the device STAGING CARVE the
    born-spilled prefill write needs (PS2 stage B).

    The chunk's owned share differs per rank under uneven DCP (rank r owns the
    residue window ``[prefix[r], prefix[r+1])``), but the CARVE must be sized
    from replicated config only, so every rank reserves the same number of
    device rows. Sizes may differ per rank at RUN time (fill level); the
    RESERVATION may not (S3b.4 item 3).

    The bound is ``ceil(T / S) * max_ratio``: a T-token POSITION window covers
    ``ceil(T/S)`` residue periods at most, and a rank owns ``ratio_r`` residues
    of each period. Flooring the per-rank share (``T * ratio_r // S``) is BOTH
    rank-varying and too small whenever ``T % S != 0`` -- the window's partial
    period can land entirely inside one rank's residue range."""
    T = max(0, int(chunk_tokens))
    S = max(1, int(split_factor))
    r = max(1, int(max_ratio))
    if T == 0:
        return 0
    return ((T + S - 1) // S) * r


def prefill_spill_owner_split(
    positions: torch.Tensor, split_factor: int, lo: int, hi: int
) -> torch.Tensor:
    """Indices (into the extend chunk) of the tokens THIS rank owns, ascending.

    A born-spilled token at absolute position ``p`` carries the sentinel
    ``host_base + p * S + (p % S)``, so its owner residue is ``p % S`` and this
    rank owns it iff that residue falls in its window ``[lo, hi)`` -- the same
    positional rule ``spill_decode_alloc`` uses for tokens generated while
    spilled, and the same one ``_sess_prepare_step`` re-derives from the row.

    The j-th returned index is the j-th owned token of the tail in ASCENDING
    POSITION order, which is exactly how the tick addresses host rows
    (``region_base + host_row_base + j``). Keeping this ordering is the PS2
    half of the LOCKSTEP invariant: the host row of a token is never stored,
    it is recomputed every tick from (L, boundary, host_row_base, owner rule),
    so the write must place row j where that recomputation will look for it."""
    S = max(1, int(split_factor))
    res = positions.to(torch.int64) % S
    mask = (res >= int(lo)) & (res < int(hi))
    return mask.nonzero(as_tuple=False).flatten()


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
    """ON-DEVICE MTP RESUME gate: lift the host-finish guard so a spilled spec
    session waves back and rejoins the LIVE spec decode batch, instead of
    finishing on host. Default OFF keeps the validated host-finish path; the
    reasoning for that default lives in the flag's help text, which is the
    surface an operator can actually find.

    TWO SOURCES, OR-ed: ``--kv-session-offload-resume-under-spec`` and its env
    twin ``SGLANG_KVSO_RESUME`` (legacy alias ``KVSO_RESUME``). The OR is
    deliberate rather than a precedence rule -- there is no meaningful "off
    beats on" here, and a boot-matrix arm that exports the env must not be
    silently overruled by the flag's default. Neither set -> False -> every
    caller takes the byte-identical pre-flag path.

    Env FIRST so the predicate keeps working in a process that has no runtime
    context yet (unit tests, a worker before ``set_server_args``); the server
    arg is read defensively for the same reason."""
    if envs.SGLANG_KVSO_RESUME.get():
        return True
    try:
        from sglang.srt.runtime_context import get_server_args

        return bool(
            getattr(get_server_args(), "kv_session_offload_resume_under_spec", False)
        )
    except Exception:  # noqa: BLE001 -- no runtime context: env is the answer
        return False


def draft_kv_verify_enabled() -> bool:
    """Stage-1 self-check: after restoring the draft-KV bundle into the new
    slots, read it back and assert byte-exact equality with the pinned-CPU
    snapshot (validates the snapshot/restore plumbing -- slot indexing, dtype,
    shapes -- end to end). Off by default; KVSO_S1_VERIFY=1 for bring-up."""
    return os.environ.get("KVSO_S1_VERIFY", "0") == "1"


#: Graph-coverage states the spill-tick trace can report. The three are NOT
#: two-plus-noise: "unattributed" is the honest answer whenever no collective
#: slot reached the reporter, and collapsing it into "eager" is exactly the
#: wrong-zero the CollectiveClock's own docstring refuses to emit
#: (utils/collective_clock.py:41-44).
TICK_GRAPH_UNATTRIBUTED = "unattributed"  # no collective slot reached us
TICK_GRAPH_COVERED = "covered"  # a collective ran under graph capture
TICK_GRAPH_EAGER = "eager"  # slot present, nothing skipped for capture


def tick_graph_state_from_slot(slot: Optional[object]) -> str:
    """Classify one spill-tick forward's collective slot for the trace.

    WHY THIS READS ONE BOOLEAN AND NOTHING ELSE. The slot arrives as
    ``collective_slot`` metadata from :class:`SplitDeviceTimer`, and the same
    slot object is handed to EVERY reporter registered on that timer. Harvesting
    it is destructive -- ``CollectiveClock.harvest_detail`` returns the events
    to the pool and clears ``slot.pairs`` -- so whoever harvests first empties
    it for everyone after. kvso installs its reporter with ``add_reporter``
    (:meth:`_install_regulator_device_timer`), i.e. LAST, so any ms read here
    would be an order-dependent zero. ``graph_capture_skipped`` is the one
    field harvesting does not touch, so it is order-independent and safe; the
    ms axis stays with the existing ``tick_cost`` and with whoever owns the
    harvest.

    Arms H and I of the #550 window ran under full CUDA graphs
    (``boot_matrix/arms.py`` BASE_EXPECT ``graphs=True``) with no way to say
    which part of a tick was graph-covered and which was eager. This is that
    axis, at the one place a spill tick is already measured.
    """
    if slot is None:
        return TICK_GRAPH_UNATTRIBUTED
    return (
        TICK_GRAPH_COVERED
        if getattr(slot, "graph_capture_skipped", False)
        else TICK_GRAPH_EAGER
    )


def tick_trace_enabled() -> bool:
    """Diagnostic-only time-series trace of the self-cal spill-tick regulator:
    when SGLANG_KVSO_TICK_TRACE=1, emit one throttled log line per spilled
    session while any spill is in flight -- effective interval, measured
    tick_cost, binding headroom ratio, and the CURRENT host-tail size
    (offload length in tokens). Pure logging; reads only existing regulator /
    slot state and changes NO control decision, so default OFF is byte-
    identical (and this trace is a no-op unless the adaptive regulator is on)."""
    return os.environ.get("SGLANG_KVSO_TICK_TRACE", "0") == "1"


# ---------------------------------------------------------------------------
# P2 (deep-offload S1): host-pool sizing from an explicit RAM BUDGET
#
# Today the pinned host pool is sized PURELY from --context-length:
#   region_tokens = (ctx // S + 2) * max_ratio;  need = region_tokens * max_spills
# so reaching the ten-thousand-token tail depth the deep-offload story needs
# (a big --context-length) also multiplies the host allocation, with no knob
# to bound it -- the box has 108 GB and NO swap, so an over-large auto-size is
# an OOM-killer event, not a graceful failure.
#
# --kv-session-offload-host-ram-gib decouples the two: the per-session DEPTH
# (region_tokens) stays the full context (a session can never hold more than
# context_len anyway), and the budget bounds only HOW MANY regions are really
# dimensioned == the effective --kv-session-offload-max-spills. The budget is
# a PHYSICAL CEILING, never a cadence / regulator input (P3 guard: it must not
# appear anywhere in ScheduleBatchRegulator / observe_sample / _local_ratio /
# _desired -- there is a unit test asserting exactly that).
#
# RANK-UNIFORMITY (this fork: divergence == NCCL hang, not a wrong number):
# every function here is PURE and takes only REPLICATED inputs (server args,
# context_len, the DCP prefix geometry). The one rank-LOCAL quantity in the
# sizing, per_token_bytes (uneven TP -> different kv-head shares), is folded
# out by the host pool's OWN existing min-all-reduce over the token capacity
# (sync_fixed_hicache_size, pool_host/base.py) -- so the effective region
# count is derived from the post-sync host_pool.size and needs NO new
# collective. Never introduce a per-rank free-memory query into this path.
# ---------------------------------------------------------------------------

# Host RAM left free for the OS and everything else when a budget is given.
# Same magnitude as HICACHE_HOST_MEMORY_RESERVE_BYTES (pool_host/base.py); the
# pool is PINNED (non-swappable) and this box has no swap at all.
HOST_RAM_BUDGET_RESERVE_BYTES: int = 10 * (1024**3)


def host_ram_budget_error(
    budget_gib: float,
    total_bytes: int,
    available_bytes: int,
    reserve_bytes: int = HOST_RAM_BUDGET_RESERVE_BYTES,
) -> Optional[str]:
    """Plausibility check for --kv-session-offload-host-ram-gib against the
    REAL host RAM. Returns None when the budget is plausible, else a complete
    operator-facing message (fail fast and loud instead of letting the OOM
    killer pick a victim later -- on a swap-less box that victim is often an
    unrelated process).

    The budget is the NODE-WIDE total across all TP ranks (each rank allocates
    budget/tp_size), so it is compared against the machine's RAM as a whole.

    Deliberately checked ONCE at argument-parse time in the launcher process,
    NOT per rank at pool-alloc time: `available` shrinks as each rank pins its
    share, so a per-rank check would pass on rank 0 and raise on rank 2 --
    a rank-divergent boot decision, i.e. an NCCL hang."""
    b = float(budget_gib)
    if b <= 0:
        return None
    want = b * (1024**3)
    if want > int(total_bytes):
        return (
            "--kv-session-offload-host-ram-gib="
            f"{b:g} GiB exceeds the machine's TOTAL host RAM "
            f"({int(total_bytes) / (1024**3):.1f} GiB). The kv-session-offload "
            "host pool is PINNED memory and cannot be swapped out."
        )
    usable = int(available_bytes) - int(reserve_bytes)
    if want > usable:
        return (
            "--kv-session-offload-host-ram-gib="
            f"{b:g} GiB does not fit in the currently available host RAM: "
            f"{int(available_bytes) / (1024**3):.1f} GiB available minus a "
            f"{int(reserve_bytes) / (1024**3):.1f} GiB OS reserve = "
            f"{max(0, usable) / (1024**3):.1f} GiB usable. Lower the budget, "
            "or free host memory first (the pool is pinned; over-committing it "
            "invokes the OOM killer instead of swapping)."
        )
    return None


def host_pool_budget_bytes_per_rank(budget_gib: float, n_pool_ranks: int) -> int:
    """Per-rank share of the NODE-WIDE host-RAM budget. Every rank that
    attaches a host pool (== the non-draft TP ranks; PP/DP are rejected for
    this feature) gets the same share, so the value is replicated by
    construction."""
    return int((float(budget_gib) * (1024**3)) // max(1, int(n_pool_ranks)))


def host_pool_request_gb(
    need_tokens: int,
    per_token_bytes: int,
    budget_gib: float,
    n_pool_ranks: int,
) -> float:
    """Host-pool size to REQUEST on this rank, in GB (10**9 B -- the unit
    HostKVCache's ``host_size`` uses).

    ``budget_gib <= 0`` (flag OFF, the default) reproduces today's
    context-derived auto-size EXACTLY: ceil(need_bytes / 1e9), at least 1.

    With a budget the request is ``min(context_need, budget/rank)`` -- the
    budget is a CEILING, never an inflation: a budget larger than the context
    need yields the identical context-derived size, so a generous budget is
    behaviourally identical to flag OFF."""
    need_bytes = int(need_tokens) * int(per_token_bytes)
    ctx_gb = float(max(1, -(-need_bytes // 10**9)))
    if budget_gib is None or float(budget_gib) <= 0:
        return ctx_gb
    per_rank_bytes = host_pool_budget_bytes_per_rank(budget_gib, n_pool_ranks)
    return min(ctx_gb, per_rank_bytes / 1e9)


def host_pool_effective_max_spills(
    pool_size_tokens: int, region_tokens: int, max_spills: int
) -> int:
    """How many full-context regions the ALLOCATED host pool really holds,
    capped by the configured --kv-session-offload-max-spills.

    Called with the pool's POST-min-all-reduce ``size`` (rank-uniform) and the
    replicated ``region_tokens`` -> rank-uniform without a new collective.
    0 means not even ONE full-context session fits: the caller must fail fast
    (physical impossibility), never silently shrink the per-session depth."""
    region = max(1, int(region_tokens))
    fits = int(pool_size_tokens) // region
    return max(0, min(int(max_spills), fits))


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


def spill_graph_pick_rung(needed_blocks: int, ladder: List[int]) -> Optional[int]:
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
        host_rows = torch.arange(s, s + cnt, dtype=torch.int64, device=device)
        indptr = torch.tensor([0, cnt], dtype=torch.int32, device=device)
        plan.append({"cnt": cnt, "host_rows": host_rows, "indptr": indptr})
    return plan


def compact_weighted(
    loc: torch.Tensor, S: int, lo: int, hi: int
) -> Tuple[torch.Tensor, torch.Tensor]:
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


# -- per-session spill (latency) class -----------------------------------
#
# A user-supplied regler, never inferred: the caller declares how tolerant a
# session is of losing device residency. Nothing in the runtime promotes or
# demotes a session between classes.
#   "never"     -- latency-critical; tabu as a spill victim, exactly like the
#                  oldest-running / sole-session rule. Tabu also under
#                  fast-lane pressure (a fast request that only fits by
#                  evicting a 'never' session stays queued instead).
#   "normal"    -- today's FCFS order (DEFAULT; the whole feature is inert
#                  while every session carries it).
#   "preferred" -- latency-tolerant; offered as a victim BEFORE any normal
#                  session, ahead of FCFS.
SPILL_CLASS_NEVER = "never"
SPILL_CLASS_NORMAL = "normal"
SPILL_CLASS_PREFERRED = "preferred"
SPILL_CLASSES = (SPILL_CLASS_PREFERRED, SPILL_CLASS_NORMAL, SPILL_CLASS_NEVER)

# Protection rank inside session_priority_key: HIGHER = MORE protected. The
# absent / unknown / default case must yield the NORMAL rank so that a fleet
# which never sets the field keeps exactly today's ordering.
_SPILL_CLASS_RANK = {
    SPILL_CLASS_PREFERRED: 0,
    SPILL_CLASS_NORMAL: 1,
    SPILL_CLASS_NEVER: 2,
}
_SPILL_RANK_NORMAL = _SPILL_CLASS_RANK[SPILL_CLASS_NORMAL]


def spill_class_of(req) -> str:
    """The request's spill class, defaulting to NORMAL.

    Deliberately tolerant: an unset attribute, ``None`` and an unknown string
    all resolve to "normal". The value is validated once at the API boundary
    (tokenizer manager) and at arg-parse (server default), so an unknown value
    reaching here means an internal path that never carried the field -- which
    must behave exactly like today rather than raise inside the spill hot
    path."""
    cls = getattr(req, "spill_class", None)
    return cls if cls in _SPILL_CLASS_RANK else SPILL_CLASS_NORMAL


def spill_class_rank(req) -> int:
    return _SPILL_CLASS_RANK[spill_class_of(req)]


def session_priority_key(req) -> Tuple[int, int, int]:
    """Pure spill-protection ordering (HIGHER key = MORE protected):

        (spill_class_rank, is_fast_lane, -arrival_seq)

    * The per-session spill class dominates: a 'never' session outranks every
      other, a 'preferred' one ranks below every other. With no class set
      anywhere the leading element is the constant NORMAL rank, so the
      ordering degenerates to the pre-class ``(is_fast_lane, -arrival_seq)``
      -- byte-identical.
    * Fast-lane requests rank above EVERY normal request of the same spill
      class (they are never spill victims; user decision: fast beats FCFS).
    * Within a class, the OLDER request (smaller arrival_seq) is more
      protected -- so two fast-lane requests order FCFS among themselves,
      exactly like two normal ones.
    The spill victim is always the LEAST protected candidate."""
    fast = 1 if getattr(req, "is_fast_lane", False) else 0
    seq = getattr(req, "kv_arrival_seq", -1)
    if seq is None:
        seq = -1
    return (spill_class_rank(req), fast, -seq)


def spill_victim_candidates(
    reqs,
    fast_pressure: bool = False,
    blocked: Optional[set] = None,
) -> List[int]:
    """Indices a spill may legitimately victimize, in no particular order.

    Extracted from ``select_spill_victim`` so that the ORDER question (who is
    picked) and the ELIGIBILITY question (who may be picked at all) can be
    asked separately. Nothing else may re-derive eligibility: a second copy of
    these rules is how a user regler quietly stops holding on one path while
    still holding on another.

    The rules, unchanged: fast-lane requests are never victims; a session with
    spill class 'never' is never a victim; under plain decode-OOM pressure the
    most-protected (oldest) normal session is tabu, resolved over the FULL
    candidate set BEFORE the cooldown exclusion so a blocked oldest cannot
    shift the tabu onto the second-oldest; the #236 cooldown can only ever
    REMOVE candidates.
    """
    candidates = [
        i
        for i in range(len(reqs))
        if not getattr(reqs[i], "is_fast_lane", False)
        and spill_class_of(reqs[i]) != SPILL_CLASS_NEVER
    ]
    if not candidates:
        return []
    if not fast_pressure:
        oldest = max(candidates, key=lambda i: session_priority_key(reqs[i]))
        candidates = [i for i in candidates if i != oldest]
        if not candidates:
            return []
    if blocked:
        candidates = [i for i in candidates if i not in blocked]
    return candidates


def spec_back_only_victim(
    reqs,
    sizes: Optional[List[int]] = None,
    need: int = 0,
    fast_pressure: bool = False,
    blocked: Optional[set] = None,
) -> Optional[int]:
    """Under speculative decoding, the back-most request IF it may be spilled.

    WHY THIS EXISTS. Under EAGLE/MTP a request may only leave the batch from
    the BACK (``spec_decline_non_back_spill`` carries the reasoning). When the
    policy-chosen victim was not the back-most, ``try_spill`` used to decline
    outright and hand the pressure to stock ``retract_decode`` -- which, under
    spec, is back-only too (``_get_decode_retraction_order`` returns the
    indices UNSORTED and the loop pops from the tail). So the back-most request
    was evicted either way; the only thing the decline changed was HOW. Spill
    keeps the session's work on host and it decodes on through the spill tick;
    retraction throws the work away and the request re-prefills from scratch
    when it is re-admitted.

    That made speculative decoding silently cost the whole offload feature in
    exactly the case the feature is for. This function closes it: when the
    FCFS/minimal-eviction pick is unreachable under the back-only rule, offer
    the back-most request instead -- but ONLY if it is a legitimate victim by
    the SAME rules (``spill_victim_candidates``). Not a weakening: every
    protection still holds, and the alternative for a protected back-most
    request is unchanged (decline -> stock retraction, whose disregard for
    those protections is a documented bound of speculative decoding, not
    something this function introduces).

    ``sizes``/``need`` are accepted for symmetry with ``select_spill_victim``
    and are deliberately NOT used to reject: a partial spill that frees less
    than the shortfall still frees real device tokens and still preserves the
    work, which beats a retraction that frees the same slots and preserves
    nothing.

    Pure function of replicated batch state -> the same answer on every rank.
    """
    n = len(reqs)
    if n == 0:
        return None
    back = n - 1
    eligible = spill_victim_candidates(reqs, fast_pressure, blocked)
    return back if back in eligible else None


def select_spill_victim(
    reqs,
    sizes: Optional[List[int]] = None,
    need: int = 0,
    fast_pressure: bool = False,
    blocked: Optional[set] = None,
) -> Optional[int]:
    """Spill victim index, or None.

    Candidates are the NORMAL (non-fast-lane) requests only, ordered by
    ``session_priority_key`` (youngest = least protected).

    ``blocked`` (#236 cooldown): indices excluded as victims by the
    post-restore cooldown (progress lock / time cap). Exclusion happens AFTER
    the oldest-tabu is resolved over the FULL normal candidate set, so the
    protection semantics are untouched: the oldest normal session stays tabu
    (never shifted onto the second-oldest by a blocked entry), a sole running
    session still never self-spills, and ``blocked=None`` (the default) is
    byte-identical to the pre-#236 behaviour. The cooldown can only ever
    REMOVE victims, never add one.

    SPILL CLASS (user regler): a session declared ``spill_class="never"`` is
    removed from the candidate set outright -- the same kind of tabu as the
    sole-session rule, and it holds under fast-lane pressure too (a fast
    request that would only fit by evicting a 'never' session stays queued;
    the user asked for that session's latency, not the scheduler). A
    ``"preferred"`` session sorts below every normal one and is therefore
    offered as the victim first, ahead of FCFS. With no class set anywhere
    both effects vanish and the selection is byte-identical.

    fast_pressure=False (plain decode-OOM among normal sessions): the
    OLDEST running normal session is untouchable (removed from the
    candidate set entirely). "Oldest" is resolved by the same protection
    key, so when 'preferred' sessions are present the tabu falls on the
    oldest NON-preferred one and every 'preferred' session -- including the
    oldest -- stays victimizable. If ALL candidates are 'preferred' the
    most-protected of them keeps the tabu, so a lone session still never
    self-spills.
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
    candidates = spill_victim_candidates(reqs, fast_pressure, blocked)
    if not candidates:
        return None
    by_youth = sorted(candidates, key=lambda i: session_priority_key(reqs[i]))
    if sizes is not None and need > 0:
        for i in by_youth:
            if sizes[i] >= need:
                return i
    return by_youth[0]


def mtp_resident_reservation_error(
    pool_tokens: int, slices: int, chunked_prefill_size: int
) -> Optional[str]:
    """Reject a spec-in-tick scratch reservation that starves the KV pool.

    ``--kv-session-offload-mtp-resident-slices`` reserves device KV slots for
    the spill tick's device draft and takes them OUT of the allocator for the
    manager's whole lifetime (``allocator.size -= slices``). Nothing bounded
    that against the pool it is carved from.

    The failure it produces is the worst kind. If what remains cannot hold one
    maximum prefill chunk, the scheduler can never assemble a full prefill
    again: requests are still accepted and queued, they are simply never
    admitted. The server does not crash, does not hang in any collective (the
    ranks keep looping in lockstep and issue their scheduling collectives at
    full rate), and logs nothing -- it just stops answering. Measured on this
    rig: pool 3600 tokens, slices 2048 -> 1552 left against
    chunked_prefill_size 2048, and the server wedged after 6 requests with the
    7th queued forever. The same load with slices 256 (3344 left) ran all 9.

    Returns the error text, or None when the reservation is safe.
    """
    slices = int(slices)
    pool_tokens = int(pool_tokens)
    chunked_prefill_size = int(chunked_prefill_size)
    if slices <= 0 or pool_tokens <= 0:
        return None
    remaining = pool_tokens - slices
    if remaining <= 0:
        return (
            f"--kv-session-offload-mtp-resident-slices={slices} exceeds this "
            f"rank's entire KV pool of {pool_tokens} tokens. The reservation "
            "is permanent, so no request could ever be admitted."
        )
    if chunked_prefill_size > 0 and remaining < chunked_prefill_size:
        return (
            f"--kv-session-offload-mtp-resident-slices={slices} would "
            f"permanently remove {slices} of the {pool_tokens} KV tokens in "
            f"this rank's pool, leaving {remaining} -- less than one chunked "
            f"prefill (chunked_prefill_size={chunked_prefill_size}). The "
            "reservation is held for the manager's lifetime, so the scheduler "
            "could never admit a full prefill chunk again: new requests are "
            "accepted, queued, and never run, with no crash, no collective "
            "hang and no log line. Lower "
            "--kv-session-offload-mtp-resident-slices, raise "
            "--max-total-tokens, or lower --chunked-prefill-size."
        )
    return None


def draft_scratch_carveout_error(
    before: int, after: int, slices: int, allocator_name: str
) -> Optional[str]:
    """Reject a draft-scratch carve-out whose write did not take effect.

    ``--kv-session-offload-mtp-resident-slices`` reserves device KV slots for
    the spill tick's device draft FOREVER, and the reservation is only honest
    if the slots also leave ``allocator.size`` -- that is the ``total`` the
    scheduler's leak invariant balances against. Some allocators cannot honour
    the write: on the hybrid composites (``UnifiedMambaTokenToKVPoolAllocator``
    / ``UnifiedSWATokenToKVPoolAllocator``) ``size`` is COMPUTED from live
    sub-allocator state and the setter is a no-op absorber, so ``size -= n``
    is silently dropped. The permanent allocation then shows up as a standing
    leak of ``slices`` slots in every invariant check, while the advertised
    capacity still counts slots no request can ever get.

    Returns the error text, or None when the carve-out really happened.
    """
    before, after, slices = int(before), int(after), int(slices)
    if after == before - slices:
        return None
    return (
        f"the draft-read scratch carve-out did not take effect on "
        f"{allocator_name}: allocator.size stayed {after} where "
        f"{before - slices} was expected ({before} - {slices} reserved "
        "slots). On this allocator `size` is a computed property whose "
        "setter absorbs the write, so the reservation would stay counted in "
        "the advertised capacity while being permanently out of circulation "
        "-- the scheduler's pool-leak invariant would report exactly these "
        f"{slices} slots as leaked, every check, for the manager's life. "
        "Refusing instead of logging a carve-out that did not happen. Run "
        "spec-in-tick with an allocator whose size is settable, or unset "
        "--kv-session-offload-mtp-resident-slices."
    )


#: Set to 1 to honour ``--kv-session-offload-restore-margin-tokens`` verbatim
#: even when it is unsatisfiable against the pool. The operator keeps the last
#: word; the resolution still logs, at ERROR, what it was told to ignore.
RESTORE_MARGIN_FORCE_ENV = "SGLANG_KVSO_RESTORE_MARGIN_FORCE"


def restore_margin_force_enabled() -> bool:
    return os.environ.get(RESTORE_MARGIN_FORCE_ENV, "0") == "1"


def restore_margin_shipped_default() -> Optional[int]:
    """The margin's dataclass default, READ from ``ServerArgs``.

    Restating ``4096`` here would create a second copy of a constant whose
    whole role is to answer "did the operator choose this value, or did we?".
    A copy that drifted would silently move a boot from the clamp branch to
    the refusal branch. Imported lazily: ``server_args`` imports this module's
    package, so a module-level import would close a cycle.

    Returns None when the field cannot be found. ``resolve_restore_margin_
    tokens`` then treats every value as operator-chosen and REFUSES rather
    than clamping -- the loud side, which is the right side to fail to.
    """
    import dataclasses

    try:
        from sglang.srt.server_args import ServerArgs

        for f in dataclasses.fields(ServerArgs):
            if f.name == "kv_session_offload_restore_margin_tokens":
                return int(f.default)
    except Exception:  # pragma: no cover - import shape change
        return None
    return None


def resolve_restore_margin_tokens(
    pool_tokens: int,
    configured: int,
    shipped_default: Optional[int],
    forced: bool = False,
) -> Tuple[int, Optional[str], Optional[str]]:
    """Size the restore margin against the KV pool it is spent from.

    ``--kv-session-offload-restore-margin-tokens`` is an ABSOLUTE token count
    and was validated only against ``< 0``. The restore gate is

        restorable >= remaining + restore_margin_tokens

    where ``restorable`` is bounded above by the whole pool. A margin at or
    above the pool therefore demands more slots than exist, for every session,
    forever: **the gate cannot open even once**. It does not crash, does not
    warn, and does not log -- every spilled session simply finishes on the host
    floor while the device sits idle, which reads exactly like "restore is not
    implemented". Measured: successor 44 saw restores=0 across an entire boot
    on a 4096-token pool against the shipped default margin of 4096; successor
    45 got the first ``RESTORE complete`` of this whole line of shifts out of
    the same tree purely by passing ``--kv-session-offload-restore-margin-
    tokens 64``. Nothing in the code said which of those two boots was the
    broken one. That silence is the defect being fixed here.

    Returns ``(effective_margin, error, warning)``.

    * ``error`` is non-None only when the operator EXPLICITLY chose an
      unsatisfiable margin -- the caller raises. An explicit unsatisfiable
      request is an operator error and gets the same treatment as an
      unsatisfiable ``--kv-session-offload-mtp-resident-slices``.
    * A margin left at the SHIPPED DEFAULT is clamped instead, and the clamp
      is reported through ``warning`` for the caller to log at ERROR. The
      operator did not choose the default, so refusing to boot on it would
      turn a shipped constant into a hard failure on every small-pool
      instance -- but it must never go quietly inert, which is the whole
      point. An operator who explicitly passes exactly the default value on a
      too-small pool lands in the clamp branch rather than the refusal
      branch; the two are indistinguishable from the parsed args and the
      clamp is the safe side of that ambiguity.
    * ``forced`` (``SGLANG_KVSO_RESTORE_MARGIN_FORCE=1``) suppresses both the
      refusal and the clamp and honours ``configured`` verbatim, still
      reporting through ``warning``. The operator keeps the last word.

    The clamp target is HALF the pool. Half is where the margin stops being
    the binding term: the largest session that can coexist with its own
    margin occupies at most ``pool - margin`` slots, so at ``margin > pool/2``
    the margin excludes sessions SMALLER than itself from ever restoring,
    while at ``margin <= pool/2`` every session the pool can hold alongside
    the margin can still reach the gate. It is a bound, not a tuning: a margin
    that has to be clamped is already misconfigured and the log says so.
    """
    pool_tokens = int(pool_tokens)
    configured = int(configured)
    shipped_default = None if shipped_default is None else int(shipped_default)
    if pool_tokens <= 0 or configured <= 0:
        # No pool figure to judge against (or no margin at all -> the gate
        # reduces to `restorable >= remaining`, which is satisfiable).
        return configured, None, None

    if configured < pool_tokens:
        if configured * 2 > pool_tokens:
            return (
                configured,
                None,
                f"--kv-session-offload-restore-margin-tokens={configured} is "
                f"more than half of this rank's {pool_tokens}-token KV pool, "
                f"so any spilled session with a host tail longer than "
                f"{pool_tokens - configured} tokens can never satisfy the "
                "restore gate and will finish on the host floor. The margin "
                "is anti-flutter headroom, not a reserve; lower it or raise "
                "--max-total-tokens.",
            )
        return configured, None, None

    clamped = max(1, pool_tokens // 2)
    detail = (
        f"--kv-session-offload-restore-margin-tokens={configured} is at or "
        f"above this rank's ENTIRE KV pool of {pool_tokens} tokens. The "
        f"restore gate asks for (session tail + {configured}) free slots, so "
        "it cannot open for any session, ever: every spilled session finishes "
        "on the host floor with the device idle, silently, with no restore "
        "and no error."
    )
    if forced:
        return (
            configured,
            None,
            detail + f" Honoured verbatim because {RESTORE_MARGIN_FORCE_ENV}=1.",
        )
    if configured == shipped_default:
        return (
            clamped,
            None,
            detail
            + f" This is the SHIPPED DEFAULT ({shipped_default}), not an "
            f"operator choice, so it is clamped to {clamped} (half the pool) "
            "rather than refused. Set --kv-session-offload-restore-margin-"
            "tokens explicitly to choose your own value.",
        )
    return (
        configured,
        detail
        + f" Lower it below {pool_tokens} (half the pool, {clamped}, is the "
        "largest value that leaves every session the pool can hold able to "
        f"restore), raise --max-total-tokens, or set "
        f"{RESTORE_MARGIN_FORCE_ENV}=1 to run with the gate shut anyway.",
        None,
    )


def spill_tick_seq_len(n_origin_input_ids: int, n_output_ids: int) -> Optional[int]:
    """Committed sequence length for a spill-tick decode batch, or None when
    the session has no token to decode yet.

    ONE formula, two entry paths into the tick:

    * DECODE-SPILL (the path this was written for): the session spilled
      mid-decode, so it always holds at least one output token, and its LAST
      output token is the one whose KV this tick is about to write. Hence
      ``origin + output - 1`` == ``kv_committed_len``. Unchanged.

    * BORN-SPILLED (PS1/PS2): the session is handed to the tick straight out
      of prefill. ``prepare_for_extend`` already set
      ``kv_committed_len = seq_len`` for the WHOLE input, and under the
      overlap scheduler the prefill's sampled token is appended to
      ``output_ids`` only when its result is processed -- one iteration later.
      In that window ``output_ids`` is EMPTY, and the old expression silently
      undercounted by one: measured on this rig ``origin=1967 output=0
      committed=1967`` -> 1966, tripping the tick-build assert and taking
      SIGQUIT on all three ranks.

      The right answer there is not a different arithmetic. With no output
      token there is nothing to decode FROM -- a decode step needs a token to
      feed, and ``req.output_ids[-1]`` would raise IndexError two lines on.
      The session is simply not tickable yet, so the tick defers one
      iteration; by then ``output=1``, ``committed=1967`` and the SAME formula
      gives 1967.

    Returning None (rather than raising) keeps the caller's decision explicit
    and is rank-uniform: ``output_ids`` is replicated batch metadata, and the
    diagnostic above showed identical values on all three ranks.
    """
    if int(n_output_ids) <= 0:
        return None
    return int(n_origin_input_ids) + int(n_output_ids) - 1


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


#: #552: consecutive fast-lane deferrals a spilled session tolerates before its
#: restore is forced through once.
#:
#: The fast-lane deferral in ``_maybe_restore_flow`` is correct and has a good
#: reason -- restoring while a fast request waits only re-triggers the
#: fast-pressure spill, one full D2H+H2D per cycle -- but it was UNBOUNDED: it
#: resets the hysteresis streak and returns, so no progress accumulates, and
#: under continuous fast-lane traffic an older spilled session never restores.
#: "Fast beats FCFS" was meant as a tie-break, not as an indefinite hold.
#:
#: The scheduler already solved this exact shape for the other lane:
#: ``fast_lane_heavy_aging_ms`` (server_args) promotes a heavy request that has
#: waited too long AHEAD of the fast tier for one admission. This is the same
#: rule in the units this loop actually has (iterations, not milliseconds).
#:
#: Generous on purpose: it fires only in the pathological case, so the normal
#: "fast beats FCFS" behaviour is untouched. FAILURE DIRECTION, stated: when it
#: does fire, one restore happens while a fast request is waiting, which may
#: cost that fast request a re-spill. That is the price of not stranding a
#: session forever, and it is bounded to one restore per aged-out session.
DEFAULT_RESTORE_DEFER_LIMIT = 100


class RestoreHysteresis:
    """Restore fires only after the memory condition has held for
    ``steps`` consecutive checks (anti-flutter).

    Also carries the #552 anti-starvation counter, because the thing that
    zeroes the streak (a fast-lane deferral) is exactly the thing that has to
    be bounded -- keeping the two in one object means a deferral cannot reset
    progress without also recording that it did.
    """

    def __init__(self, steps: int, defer_limit: int = DEFAULT_RESTORE_DEFER_LIMIT):
        self.steps = max(1, int(steps))
        #: <= 0 disables the bound entirely (restores the pre-#552 behaviour).
        self.defer_limit = int(defer_limit)
        self._streak = 0
        self._deferrals = 0

    def update(self, ok: bool) -> bool:
        self._streak = self._streak + 1 if ok else 0
        return self._streak >= self.steps

    def defer(self) -> bool:
        """Record one fast-lane deferral. True when the bound is EXCEEDED.

        Returning True does not itself restore anything -- it tells the caller
        this session has been held off long enough that the next check must be
        allowed through, so the decision stays at the call site.
        """
        self._streak = 0
        self._deferrals += 1
        if self.defer_limit <= 0:
            return False
        return self._deferrals >= self.defer_limit

    @property
    def deferrals(self) -> int:
        return self._deferrals

    def reset(self):
        self._streak = 0

    def clear_deferrals(self) -> None:
        """Called once a restore actually happens: the session is no longer
        being starved, so the count starts again from zero. Kept separate from
        ``reset`` because a deferral must NOT clear it -- that would make the
        bound unreachable, which is the bug this exists to fix."""
        self._deferrals = 0


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


def wave_back_gate(
    local_avail: int, uniform_avail: int, min_free_tokens: int
) -> Tuple[bool, int]:
    """Resolve the wave-back SPACE gate (P1): is there enough free device room
    this iteration to wave a block of the host tail back, and how many free
    slots may the step assume? Returns ``(space_ok, remaining_cap)``.

    ``min_free_tokens <= 0`` -- the DEFAULT -- reproduces the historical gate
    verbatim: ``space_ok = local_avail > 0`` with the LIVE local pool as the
    cap. P1 is inert until an operator sets the knob; no new collective, no
    changed decision, no changed step size.

    ``min_free_tokens > 0`` waves back only once at least that many device
    slots are free, and reads availability from the RANK-UNIFORM min-reduced
    snapshot instead of the rank-local pool. Both halves are load-bearing:

    * PURPOSE. Today ANY single free slot passes the gate, so wave-back peels
      one block off the tail front every single iteration; in a regime that
      keeps freeing slots it drains the tail faster than the host tick
      (~2.7 tok/s) refills it, and the tail -- i.e. the reachable context
      DEPTH -- stays shallow. A threshold lets the tail accumulated under
      pressure STAY on host; the operator decides how much VRAM must be free
      before trimming it is worth doing. The measurable win is CAPACITY, not
      speed.
    * RANK-UNIFORMITY. ``available_size()`` is rank-LOCAL, and under uneven
      DCP the per-rank pools differ, so comparing a threshold against the
      local pool would let ranks disagree about whether to wave. A divergent
      wave decision in this fork does not produce a wrong number, it HANGS
      NCCL: the wave rewrites ``req_to_token`` and moves the tier boundary,
      so the next tick's collectives would be built from different geometries
      on different ranks. ``uniform_avail`` is the MIN-reduce that
      ``update_dcp_admission_state`` already computes once per iteration
      (``dcp_min_avail``) -- reused deliberately, NO new collective. Being a
      minimum it is also the conservative side of the comparison.

    The uniform value is an iteration-start snapshot, so it may overstate the
    room still free at wave time. That is harmless and already handled: the
    owner-matched allocation inside ``_wave_back`` is the hard gate and simply
    declines, retrying next window.

    NOTE for the RESTORE path: this gate gets the tail out of the drip-feed,
    it does not strand it. RESTORE-READY (``_restore_memory_ok`` / ``fits_now``)
    is untouched, so a session whose whole tail fits still comes back in one
    step regardless of this threshold.

    Pure -> identical on every rank for identical inputs.
    """
    if int(min_free_tokens) <= 0:
        return bool(int(local_avail) > 0), int(local_avail)
    ua = int(uniform_avail)
    return ua >= int(min_free_tokens), ua


# ---------------------------------------------------------------------------
# #236 SPILL BUDGET (pure layer)
#
# Bounds the KV-session spill along independent axes: session COUNT, VOLUME
# (total / per session / per phase), RATE (tokens/s across PCIe), and a per-
# episode TIME WINDOW. Every regler is OFF at its zero default ("open"), so an
# unarmed budget changes not a byte of today's behaviour. Several reglers may
# be armed at once; at every decision point they are evaluated in ONE fixed,
# documented order and the FIRST violated regler names the event (counter +
# log reason) -- "der erste greifende gewinnt".
#
# Exhaustion is a DEMOTION, not an abort: the session stops generating (loses
# its liveness), its already-spilled work drains back to device through the
# UNCHANGED wave-back/restore machinery and is donated to the radix tree by
# the stock finish path -- a continuation is then a prefix hit, not a full
# re-prefill. Under HiRadixCache the donated node migrates to the host tier
# via HiCache's own write-through/eviction policy; the budget layer never
# touches a cache pool directly.
#
# RANK-UNIFORMITY (this fork: divergence == NCCL hang): every decision here is
# a pure function of replicated scheduler state (token counts, iteration
# counters, arrival sequences) plus the manager's RANK-UNIFORM clock (a
# MAX-reduced monotonic timestamp, refreshed once per iteration at an
# unconditional call site). No function in this layer may read a rank-local
# quantity -- in particular NOT the allocator free list and NOT the tree
# cache: budget policy neither gates nor replaces RESTORE-READINESS, which
# keeps counting the radix-evictable memory (#217) in the unchanged
# _maybe_restore_flow.
# ---------------------------------------------------------------------------

# GDN/Mamba states are charged to the budget at their NATIVE dtype size (bf16,
# itemsize 2; ~75 MB per session on the 27B, length-independent). They are
# NEVER quantized -- the recurrent state accumulates error, so a compressed
# state corrupts every later token. The budget layer enforces the invariant
# instead of merely assuming it: an itemsize below bf16 is rejected loudly.
GDN_STATE_MIN_ITEMSIZE = 2


def gdn_token_equivalent(
    gdn_state_bytes: int, per_token_kv_bytes: int, state_itemsize: int = 2
) -> int:
    """Token-equivalent budget charge of one session's GDN/Mamba state
    (ceil(state bytes / KV bytes per token)); 0 when either size is unknown.

    ``state_itemsize`` documents-and-enforces the no-quantization invariant:
    the state is accounted at its native bf16 (or wider) size, and a caller
    holding a sub-bf16 state is a bug upstream of the budget, not something to
    account for."""
    if int(state_itemsize) < GDN_STATE_MIN_ITEMSIZE:
        raise ValueError(
            "GDN/Mamba state must stay at its native dtype (>= bf16, itemsize "
            f">= {GDN_STATE_MIN_ITEMSIZE}); got itemsize {int(state_itemsize)}. "
            "GDN states are never quantized (recurrent error accumulates)."
        )
    g = int(gdn_state_bytes)
    p = int(per_token_kv_bytes)
    if g <= 0 or p <= 0:
        return 0
    return -(-g // p)


class SpillBudgetConfig(NamedTuple):
    """All #236 reglers. 0 (or 0.0) disables the individual regler; the
    all-zero default is the OPEN budget (today's behaviour, byte-identical).

    Volumes are in TOKENS (replicated units; a session's GDN state is folded
    in as a token equivalent). ``demote_grace_iters`` is the coarse upper
    bound on how long a demoted session may wait for its drain-handover
    before it falls back to a host finish."""

    total_tokens: int = 0
    session_tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    rate_tokens_per_s: float = 0.0
    episode_seconds: float = 0.0
    max_sessions: int = 0
    progress_lock_tokens: int = 0
    spill_hysteresis_steps: int = 0
    cooldown_seconds: float = 0.0
    demote_grace_iters: int = 256

    @classmethod
    def from_server_args(cls, sa) -> SpillBudgetConfig:
        def g(name, default):
            return getattr(sa, "kv_session_offload_" + name, default)

        return cls(
            total_tokens=int(g("budget_total_tokens", 0) or 0),
            session_tokens=int(g("budget_session_tokens", 0) or 0),
            prefill_tokens=int(g("budget_prefill_tokens", 0) or 0),
            decode_tokens=int(g("budget_decode_tokens", 0) or 0),
            rate_tokens_per_s=float(g("budget_rate_tokens_per_s", 0.0) or 0.0),
            episode_seconds=float(g("budget_episode_seconds", 0.0) or 0.0),
            max_sessions=int(g("budget_max_sessions", 0) or 0),
            progress_lock_tokens=int(g("spill_progress_lock_tokens", 0) or 0),
            spill_hysteresis_steps=int(g("spill_hysteresis_steps", 0) or 0),
            cooldown_seconds=float(g("spill_cooldown_seconds", 0.0) or 0.0),
            demote_grace_iters=int(g("budget_demote_grace_iters", 256) or 0),
        )

    @property
    def armed(self) -> bool:
        """Any regler set -> the budget machinery runs. All-zero (the
        default) -> every hook is skipped, byte-identical."""
        return bool(
            self.total_tokens > 0
            or self.session_tokens > 0
            or self.prefill_tokens > 0
            or self.decode_tokens > 0
            or self.rate_tokens_per_s > 0
            or self.episode_seconds > 0
            or self.max_sessions > 0
            or self.progress_lock_tokens > 0
            or self.spill_hysteresis_steps > 0
            or self.cooldown_seconds > 0
        )

    @property
    def needs_clock(self) -> bool:
        """Whether any regler reads wall time -> the manager must refresh the
        rank-uniform clock every iteration."""
        return bool(
            self.rate_tokens_per_s > 0
            or self.episode_seconds > 0
            or self.cooldown_seconds > 0
        )

    @property
    def has_volume(self) -> bool:
        return bool(
            self.total_tokens > 0
            or self.session_tokens > 0
            or self.prefill_tokens > 0
            or self.decode_tokens > 0
        )


# Fixed admission evaluation order -- the first violated regler names the
# decline. Count before volume (cheapest, structural), per-session before
# per-phase before total (most specific first), rate last (the only regler
# whose verdict can recover on its own next iteration).
BUDGET_ADMISSION_ORDER = (
    "max-sessions",
    "session-tokens",
    "prefill-tokens",
    "decode-tokens",
    "total-tokens",
    "rate",
)

# Fixed in-episode evaluation order (demotion reasons). Per-session reglers
# first, then the per-episode window, then the aggregate phase/total volumes
# (which demote the youngest live session of the class, one per iteration).
BUDGET_EPISODE_ORDER = (
    "session-tokens",
    "episode-window",
    "prefill-tokens",
    "decode-tokens",
    "total-tokens",
)


def budget_admission_violation(
    cfg: SpillBudgetConfig,
    *,
    n_open_slots: int,
    spill_tokens: int,
    phase: str,
    session_tokens_after: int,
    prefill_tokens_after: int,
    decode_tokens_after: int,
    total_tokens_after: int,
    rate_ready: bool,
) -> Optional[str]:
    """First violated regler at SPILL ADMISSION, or None (admit).

    A violation here DECLINES the spill (try_spill returns False; the stock
    retraction fallback handles the pressure exactly as it does today when no
    host region is free) -- admission never demotes. All ``*_after`` volumes
    are the projected totals INCLUDING the candidate spill. Pure over
    replicated inputs -> identical verdict on every rank."""
    if cfg.max_sessions > 0 and int(n_open_slots) >= cfg.max_sessions:
        return "max-sessions"
    if cfg.session_tokens > 0 and int(session_tokens_after) > cfg.session_tokens:
        return "session-tokens"
    if (
        phase == "prefill"
        and cfg.prefill_tokens > 0
        and int(prefill_tokens_after) > cfg.prefill_tokens
    ):
        return "prefill-tokens"
    if (
        phase == "decode"
        and cfg.decode_tokens > 0
        and int(decode_tokens_after) > cfg.decode_tokens
    ):
        return "decode-tokens"
    if cfg.total_tokens > 0 and int(total_tokens_after) > cfg.total_tokens:
        return "total-tokens"
    if cfg.rate_tokens_per_s > 0 and int(spill_tokens) > 0 and not rate_ready:
        return "rate"
    return None


def budget_episode_violation(
    cfg: SpillBudgetConfig,
    *,
    session_tokens: int,
    episode_elapsed_s: float,
) -> Optional[str]:
    """First violated PER-SESSION regler of a RUNNING episode, or None.

    A violation DEMOTES the session (see the manager's ``_budget_demote``):
    liveness ends, the work drains and is handed over. The aggregate volume
    reglers (phase / total) are evaluated by the caller over all sessions --
    they demote the youngest live session of the class, not this one."""
    if cfg.session_tokens > 0 and int(session_tokens) > cfg.session_tokens:
        return "session-tokens"
    if cfg.episode_seconds > 0 and float(episode_elapsed_s) > cfg.episode_seconds:
        return "episode-window"
    return None


class SpillRateBucket:
    """Token bucket over the manager's RANK-UNIFORM clock, protecting the
    PCIe link (the spill path shares it with the prefill-offload ingest).

    DEBT MODEL, deliberately: a single consumption may exceed the burst
    capacity (a spill tick streams its WHOLE host tail every forward, which
    can be larger than one second of budget). Blocking such a consumer until
    the bucket covers it in full would starve it forever; instead ``ready()``
    gates on a NON-NEGATIVE level and ``consume`` may push the level into
    debt, which subsequent refill pays off -- the average rate converges to
    the configured budget and nothing stalls permanently. Exceeding a budget
    transiently is throttling (defer the tick / decline the spill), never a
    demotion: rate pressure recovers on its own.

    Pure arithmetic over (uniform timestamps, replicated token counts) ->
    bit-identical level on every rank."""

    def __init__(self, rate_tokens_per_s: float, burst_seconds: float = 1.0):
        self.rate = float(rate_tokens_per_s)
        self.cap = max(1.0, self.rate * float(burst_seconds))
        self.level = self.cap
        self._last: Optional[float] = None

    def advance(self, now: float) -> None:
        """Refill from the uniform clock. Idempotent for a repeated ``now``."""
        if self._last is None:
            self._last = float(now)
            return
        dt = max(0.0, float(now) - self._last)
        self._last = float(now)
        self.level = min(self.cap, self.level + dt * self.rate)

    def ready(self) -> bool:
        return self.level >= 0.0

    def consume(self, tokens: int) -> None:
        self.level -= float(max(0, int(tokens)))


class SpillCooldownRegistry:
    """Post-restore cooldown against spill<->restore pendulum, ranked as in
    the design:

    (a) PROGRESS LOCK (primary): a session restored from host is not a spill
        victim again until it has produced ``progress_lock_tokens`` output
        tokens since the restore -- no progress, no second transfer. This
        attacks the pendulum at its root; time-based locks only do so by
        accident.
    (c) TIME CAP (coarse, on top): additionally not a victim for
        ``cooldown_seconds`` after the restore (uniform clock).

    ((b), the pressure hysteresis, is stateful per manager and lives in
    ``_maybe_spill_for_fast_lane`` -- see the manager.)

    Entries expire lazily once BOTH caps have passed. All inputs are
    replicated (output lengths) or uniform (the clock) -> every rank blocks
    and expires identically."""

    def __init__(self, progress_lock_tokens: int, cooldown_seconds: float):
        self.progress_lock_tokens = max(0, int(progress_lock_tokens))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._entries: dict = {}

    def note_restore(self, rid, output_len: int, now: float) -> None:
        self._entries[rid] = (int(output_len), float(now))

    def in_window(self, rid, output_len_now: int, now: float) -> bool:
        """Non-mutating probe: is ``rid`` still inside its cooldown window?
        Used by the actual-pendulum detector (a spill of a within-window
        victim = a Spill->Restore->Spill round INSIDE the lock, which the
        blocked() exclusion exists to make impossible)."""
        e = self._entries.get(rid)
        if e is None:
            return False
        restored_len, restored_t = e
        if (
            self.progress_lock_tokens > 0
            and int(output_len_now) - restored_len < self.progress_lock_tokens
        ):
            return True
        return bool(
            self.cooldown_seconds > 0
            and float(now) - restored_t < self.cooldown_seconds
        )

    def blocked(self, rid, output_len_now: int, now: float) -> bool:
        e = self._entries.get(rid)
        if e is None:
            return False
        restored_len, restored_t = e
        if (
            self.progress_lock_tokens > 0
            and int(output_len_now) - restored_len < self.progress_lock_tokens
        ):
            return True
        if (
            self.cooldown_seconds > 0
            and float(now) - restored_t < self.cooldown_seconds
        ):
            return True
        # Both caps passed: the entry is spent, drop it (bounds the registry).
        del self._entries[rid]
        return False


class SpillBudgetCounters:
    """#236 visibility: without these the policy is not assessable. Plain
    replicated integers; ``as_dict`` feeds the (later) dashboard."""

    def __init__(self):
        self.spilled_tokens_prefill = 0  # cumulative tokens host-spilled at prefill
        self.spilled_tokens_decode = 0  # cumulative: decode spills + host growth
        self.episodes_started = 0
        self.episodes_restored = 0
        self.episodes_finished_on_host = 0
        self.episodes_demoted = 0
        self.demotions_drained = 0  # handover complete: full prefix donated
        self.demotions_host_finished = 0  # grace fallback: host tail dropped
        # ACTUAL pendulum rounds: a session spilled again while still inside
        # its post-restore cooldown window. The blocked() exclusion makes this
        # structurally impossible while the lock is armed, so this counter
        # MUST stay 0 there -- it is the guarantee, kept as a counter so a
        # regression is a number, not an assumption.
        self.pendulum_events = 0
        # PREVENTED pendulum rounds: re-spill attempts the cooldown excluded
        # (the lock visibly working).
        self.pendulum_blocked = 0
        self.rate_throttled_ticks = 0
        self.admission_declines = 0  # budget declines -> stock retraction
        self.prefill_gate_closures = 0  # born-spill admissions gated off
        self.exhaustions: dict = {}  # reason -> count

    def note_exhaustion(self, reason: str) -> None:
        self.exhaustions[reason] = self.exhaustions.get(reason, 0) + 1

    def as_dict(self) -> dict:
        return {
            "spilled_tokens_prefill": self.spilled_tokens_prefill,
            "spilled_tokens_decode": self.spilled_tokens_decode,
            "episodes_started": self.episodes_started,
            "episodes_restored": self.episodes_restored,
            "episodes_finished_on_host": self.episodes_finished_on_host,
            "episodes_demoted": self.episodes_demoted,
            "demotions_drained": self.demotions_drained,
            "demotions_host_finished": self.demotions_host_finished,
            "pendulum_events": self.pendulum_events,
            "pendulum_blocked": self.pendulum_blocked,
            "rate_throttled_ticks": self.rate_throttled_ticks,
            "admission_declines": self.admission_declines,
            "prefill_gate_closures": self.prefill_gate_closures,
            "exhaustions": dict(self.exhaustions),
        }


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


from sglang.srt.managers.kvso_flip_contract import (
    restore_permitted,
    stamp_spill,
)


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
        # #656: the phase this image was captured in (kvso_flip_contract).
        "flip_layout",
        # C2 (spec-in-spill-tick, Option b'): the spilled session's DRAFT KV
        # kept DEVICE-resident so draft() runs as a normal device decode while
        # the (large, DCP-sharded) TARGET KV lives on host. draft_dev_k/v are
        # stacked [layer_num, tail_cap, *row] device tensors holding the draft
        # tail [draft_spill_boundary, draft_dev_len); draft_dev_len grows with
        # the host tail as accepted tokens append their draft KV. Only set when
        # spec-in-tick routes this session (else None -> plain host tick).
        "draft_dev_k",
        "draft_dev_v",
        "draft_dev_len",
        "spec_in_tick",
        # PS2 (deep prefill-spill): this session was BORN spilled -- its KV was
        # never device-resident, the prefill wrote it straight into the region.
        # ``adopted`` flips at the handover (the iteration after the prefill),
        # which is also where the copy-stream D2H is joined (U9).
        "born_spilled",
        "adopted",
        # #236 SPILL BUDGET episode state. ``budget_phase`` classifies the
        # EPISODE ("decode" spill vs "prefill"/born-spilled -- different cost
        # models, different budgets); ``budget_initial_tail`` is the volume the
        # episode STARTED with (the born prompt's write-once share; growth on
        # top is decode-phase). ``budget_demoted`` flips once when a budget
        # exhausts: the session loses its liveness (generation capped at the
        # current output), keeps its region, and drains through the unchanged
        # wave-back/restore machinery toward the radix-tree handover.
        # ``budget_tick_release`` re-allows ONE finishing host tick after the
        # drain grace expires (the coarse fallback: host finish, tail dropped).
        # All fields are set from replicated decisions -> rank-uniform.
        "budget_phase",
        "budget_initial_tail",
        "budget_episode_start",
        "budget_demoted",
        "budget_demote_iter",
        "budget_tick_release",
        # #224 destinations: chosen for parking (tick + restore stop so the
        # region content settles before the transfer snapshots it). Always
        # False unless the destination chain is armed.
        "park_pending",
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
        # C2 (spec-in-spill-tick): device-resident draft-KV buffer for this
        # spilled session (Option b'). None until try_spill routes the session
        # under spec-in-tick; draft_dev_len tracks the resident tail length.
        self.draft_dev_k = None
        self.draft_dev_v = None
        self.draft_dev_len = 0
        self.spec_in_tick = False
        # PS2: decode-spilled sessions are never "born" spilled and need no
        # handover, so the defaults keep every existing path unchanged.
        self.born_spilled = False
        self.adopted = True
        # #236 SPILL BUDGET episode state (inert defaults; see __slots__ note).
        self.budget_phase = "decode"
        self.budget_initial_tail = 0
        self.budget_episode_start = 0.0
        self.budget_demoted = False
        self.budget_demote_iter = 0
        self.budget_tick_release = False
        # #224: never set on the default path (destination chain unarmed).
        self.park_pending = False
        # #656 kvso_flip_contract: the PHASE this host image was captured in.
        # A host image is layout-specific (PP: this stage's layers, all
        # tokens; TP: a token shard of every layer), so the stamp is what
        # lets a flip decide whether carrying the image across is safe, and
        # what stops a restore into a layout the bytes never lived in. None
        # means "not provable", which the contract reads as refuse.
        self.flip_layout = None
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
        # Rank-uniform admission/decode budget for the current iteration
        # (recomputed once per iteration in update_dcp_admission_state). None ->
        # not yet computed; dcp_min_avail then falls back to the live local pool.
        self._dcp_min_avail = None
        self._dcp_budget_deficit = 0
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
        self._hysteresis_steps = int(sa.kv_session_offload_restore_hysteresis_steps)
        # P1 (S3): minimum free device slots before wave-back peels another
        # block off the host tail. 0 = today's "any free slot waves" behaviour
        # (byte-identical). Server-global -> replicated on every rank, so the
        # threshold itself can never be a source of rank divergence.
        self.wave_back_min_free_tokens = max(
            0, int(sa.kv_session_offload_wave_back_min_free_tokens)
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
        #: Whether this backend has a born-spilled EXTEND write hook, i.e.
        #: whether PS2's sentinel ``out_cache_loc`` gets diverted before it
        #: reaches ``store_kvcache``. Only the DCP lane does
        #: (``_dcp_write_scatter`` -> ``_sess_prefill_owner_write``); the plain
        #: lane has the DECODE twin (``_sess_forward_decode_plain``) but the
        #: EXTEND twin was never written. C26: admitting PS2 without it is a
        #: device-side assert. Read from replicated boot config, so this is
        #: identical on every rank and fixed before the first forward.
        self.prefill_spill_deep_backend_ok = self.mode != "plain"
        self.S = backend._sess_S
        self.cp_prefix = list(backend._sess_prefix)
        self.dcp_size = backend.dcp_size if self.mode != "plain" else 1
        self.dcp_rank = backend.dcp_rank if self.mode != "plain" else 0
        self.lo = self.cp_prefix[self.dcp_rank]
        self.hi = self.cp_prefix[self.dcp_rank + 1]
        self.host_base = backend._sess_host_base
        self.host_pool = backend._sess_host_pool
        self.full_pool = backend._sess_full_pool

        # NOT cached: see the `req_to_token_pool` property below (#1040).
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

        # === C2/C3 (spec-in-spill-tick, Phase 1 EAGER): run the model-configured
        # NEXTN/EAGLE drafter DURING the spill tick (Option b': draft KV kept
        # DEVICE-resident, only the TARGET KV host-streams via the C4 verify
        # twin). DFLASH is EXCLUDED for spilled (long-context) sessions -- only
        # the ONE configured NEXTN/EAGLE-family drafter is used (short-ctx DFLASH
        # is off by the ctx-gate at spill lengths anyway). Gated so OFF is
        # byte-identical: spec_in_tick_ready False -> _build_spill_batch keeps
        # spec_algorithm=NONE (plain host tick, unchanged).
        self.spec_in_tick = bool(getattr(sa, "kv_session_offload_spec_in_tick", False))
        self.mtp_resident_slices = int(
            getattr(sa, "kv_session_offload_mtp_resident_slices", 0) or 0
        )
        # Prefill-Spill (born-spilled) master gate (PS0). Read once here; the
        # prefill-time admission (PS1), the born-spilled write hook (PS2), the
        # host-prefix read (PS3) and the handover (PS4) are all gated on this.
        # Default False -> the prefill path is byte-identical to today.
        self.prefill_spill = bool(getattr(sa, "kv_session_offload_prefill", False))
        if self.prefill_spill:
            logger.info(
                "kv-session-offload prefill-spill (born-spilled) ENABLED: a "
                "prompt whose KV exceeds VRAM is admitted and its KV is written "
                "to host during the prefill, then handed to the off-batch decode "
                "tick. (default path unchanged when off)."
            )
        _spec = getattr(scheduler, "spec_algorithm", None)
        self.server_spec_algorithm = _spec
        # Ready only when the flag is on, the env allows spec on the spill lane
        # (KVSO_ALLOW_SPEC -- the same bring-up gate server_args validates for
        # --kv-session-offload-spec-in-tick), the server's CONFIGURED spec
        # algorithm is a non-DFLASH family, and the draft pool exists (device-
        # resident draft KV needs it). NOTE: this is INDEPENDENT of KVSO_RESUME
        # (the wave-back/host-finish gate) -- spec-in-tick runs draft/verify
        # DURING the spill and does not require the resume path to engage.
        #
        # This is a BOOT-TIME gate over the boot configuration and is NOT on its
        # own sufficient to exclude DFLASH: under cross-algorithm switching the
        # configured family is the PRIMARY (NEXTN/EAGLE) while the family that
        # actually runs moves with the active rung. The runtime half of the
        # exclusion lives in _effective_spec_algorithm() and is applied at both
        # the admission and the per-tick site below.
        self.spec_in_tick_ready = bool(
            self.spec_in_tick
            and os.environ.get("KVSO_ALLOW_SPEC", "0") == "1"
            and _spec is not None
            and not _spec.is_none()
            and not _spec.is_dflash_family()
            and self.draft_full_pool is not None
        )
        if self.spec_in_tick:
            logger.info(
                "kv-session-offload spec-in-spill-tick: flag=%s ready=%s "
                "(server_spec=%s dflash_excluded=%s resident_slices=%d). When "
                "ready, spilled sessions run draft()/verify() during the tick; "
                "DFLASH is excluded (only the configured NEXTN/EAGLE drafter).",
                self.spec_in_tick,
                self.spec_in_tick_ready,
                None if _spec is None else _spec,
                _spec is not None and _spec.is_dflash_family(),
                self.mtp_resident_slices,
            )

        # C3/d4 draft-read SCRATCH: the spill-tick draft attends its resident
        # draft-KV tail through draft-POOL slots (req_to_token surgery). Under
        # spill PRESSURE the shared allocator has NO free slots (that is why the
        # session spilled), so per-tick alloc fails -> the scratch is reserved
        # ONCE here, sized to the resident cap, and held for the manager's life.
        # A positive --kv-session-offload-mtp-resident-slices is REQUIRED to arm
        # the device draft (the reservation must be bounded); with 0 (uncapped)
        # no scratch is reserved and every spilled session falls back to the
        # plain host tick (spec-in-tick effectively off but crash-free). The
        # reserved slot ids index BOTH pools (shared slot space); only the draft
        # pool is written at them, and the tail must fit (mtp_resident_tail_fits
        # uses the same cap so an over-cap tail plain-falls-back before surgery).
        self._draft_read_scratch = None
        if self.spec_in_tick_ready and self.mtp_resident_slices > 0:
            # Fail fast BEFORE the carve: a reservation that leaves less than
            # one prefill chunk wedges the scheduler silently (see
            # mtp_resident_reservation_error). Validating here -- at arm time,
            # on every rank, with the numbers in hand -- turns a server that
            # simply stops answering into a startup error naming the cause.
            _err = mtp_resident_reservation_error(
                getattr(self.allocator, "size", 0),
                self.mtp_resident_slices,
                getattr(self.scheduler, "chunked_prefill_size", 0) or 0,
            )
            if _err is not None:
                raise ValueError("kv-session-offload spec-in-tick: " + _err)
            _sc = self.allocator.alloc(self.mtp_resident_slices)
            if _sc is not None:
                self._draft_read_scratch = _sc.to(torch.int64)
                self._carve_out_draft_scratch()
            else:
                logger.warning(
                    "kv-session-offload spec-in-tick: could NOT reserve %d "
                    "draft-read scratch slots (avail %d) -> spilled sessions "
                    "fall back to the plain host tick (no device draft).",
                    self.mtp_resident_slices,
                    self.allocator.available_size(),
                )
        elif self.spec_in_tick_ready:
            logger.warning(
                "kv-session-offload spec-in-tick: --kv-session-offload-mtp-"
                "resident-slices is 0 (uncapped); the device draft-read scratch "
                "needs a POSITIVE bound, so spilled sessions fall back to the "
                "plain host tick. Set a positive cap to arm the device draft."
            )
        # The device draft can only arm when the read scratch is actually
        # reserved -- otherwise every spilled session would plain-fall-back. Tie
        # readiness to the reservation so try_spill never routes a session it
        # cannot draft (spec_in_tick stays False -> the spill batch stays plain,
        # byte-identical to the pre-feature path).
        if self.spec_in_tick_ready and self._draft_read_scratch is None:
            self.spec_in_tick_ready = False

        # C29: size the restore margin against the pool it is spent from.
        # Deliberately placed HERE, after the draft-scratch carve-out, because
        # that carve permanently shrinks `allocator.size` -- the margin has to
        # be judged against the pool the gate will actually see, not the one
        # the boot started with. The margin is an ABSOLUTE token count, so the
        # shipped default is simultaneously fine on a 512552-token pool and
        # unsatisfiable on a 4096-token one; nothing checked which.
        _pool_tokens = int(getattr(self.allocator, "size", 0) or 0)
        _margin, _margin_err, _margin_warn = resolve_restore_margin_tokens(
            _pool_tokens,
            self.restore_margin_tokens,
            restore_margin_shipped_default(),
            forced=restore_margin_force_enabled(),
        )
        if _margin_err is not None:
            raise ValueError("kv-session-offload: " + _margin_err)
        if _margin_warn is not None:
            # ERROR, not warning: the failure this describes is invisible at
            # every other level (no crash, no hang, no restore) so the log line
            # is the ONLY place it can be seen. Never let it go quiet.
            logger.error(
                "kv-session-offload restore margin: %s (rank %d)",
                _margin_warn,
                self.dcp_rank,
            )
        if _margin != self.restore_margin_tokens:
            logger.error(
                "kv-session-offload restore margin: %d -> %d against a "
                "%d-token pool (rank %d).",
                self.restore_margin_tokens,
                _margin,
                _pool_tokens,
                self.dcp_rank,
            )
            self.restore_margin_tokens = _margin

        # S4 multi-spill: the host pool is partitioned into `max_spills` equal
        # regions of `region_tokens` rows each; region r owns host rows
        # [r * region_tokens, (r+1) * region_tokens). At most one session per
        # region is spilled at a time.
        # P2 (deep-offload S1): with --kv-session-offload-host-ram-gib the model
        # runner may have dimensioned FEWER regions than configured (the RAM
        # budget is a physical ceiling). It publishes the effective count; the
        # fallback is the configured value, so the flag-OFF path is unchanged
        # (the runner then publishes exactly that value anyway). The effective
        # count is derived from the post-min-reduce host_pool.size -> the same
        # integer on every rank, which _free_regions / prefill_spill_free_regions
        # rely on (rank-divergent region counts desync, they do not just differ).
        self.max_spills = max(
            1,
            int(getattr(mr, "kv_sess_max_spills", sa.kv_session_offload_max_spills)),
        )
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
        self._busy_ms_accum = 0.0  # device-busy ms harvested since last iter
        self._tick_cost_ms = None  # EMA of a spill-tick forward's cost (ms)
        self._last_iter_wall = None  # host perf_counter of the last pre_schedule
        # A spill tick is a bs=1 PLAIN decode (spec_algo NONE) -> device-timer
        # category "decode". In a SPEC server the main forwards are
        # target_verify / extend / draft, so category=="decode" cleanly isolates
        # the tick's marginal cost. The regulator therefore targets a spec
        # server (the intended + validated config); in a non-spec server it can
        # not separate tick from main decode and conservatively holds the floor.
        _spec0 = getattr(scheduler, "spec_algorithm", None)
        self._regulator_spec_server = _spec0 is not None and not _spec0.is_none()
        # Diagnostic-only regulator time-series trace (default OFF -> byte-
        # identical); cached once so the per-iteration path stays branch-cheap.
        self._tick_trace = tick_trace_enabled()
        self._tick_trace_iter = 0
        # Graph-coverage attribution for the spill tick (see
        # _device_timer_report). One of TICK_GRAPH_* below; starts
        # "unattributed" and only ever moves when a tick forward is actually
        # reported, so an absent value never reads as "eager".
        self._tick_graph_state = TICK_GRAPH_UNATTRIBUTED
        if bool(getattr(sa, "kv_session_offload_tick_adaptive", False)):
            self.tick_controller = SpillTickController(
                floor_interval=int(sa.kv_session_offload_tick_floor),
                reduce_fn=self._min_reduce_headroom,
            )
            self._install_regulator_device_timer()

        # === #236 SPILL BUDGET. All-zero defaults -> _budget_armed False ->
        # every budget hook below is a skipped boolean, byte-identical.
        self._budget = SpillBudgetConfig.from_server_args(sa)
        self._budget_armed = self._budget.armed
        self._budget_counters = SpillBudgetCounters()
        self._budget_bucket = (
            SpillRateBucket(self._budget.rate_tokens_per_s)
            if self._budget.rate_tokens_per_s > 0
            else None
        )
        self._budget_cooldown = (
            SpillCooldownRegistry(
                self._budget.progress_lock_tokens,
                self._budget.cooldown_seconds,
            )
            if (
                self._budget.progress_lock_tokens > 0
                or self._budget.cooldown_seconds > 0
            )
            else None
        )
        # (b) Pressure hysteresis, the spill-side MIRROR of
        # --kv-session-offload-restore-hysteresis-steps: the fast-lane
        # shortfall must HOLD for N consecutive iterations before sessions are
        # evicted for it. It applies to the fast-lane path only, deliberately:
        # a decode-OOM spill relieves pressure that must yield THIS iteration
        # (declining it would only route the same pressure into the harsher
        # stock retraction), while a fast-lane eviction is elective -- the
        # waiting request simply waits one more iteration, which is exactly
        # the flutter the mirror is meant to damp.
        self._fast_spill_pressure_gate = (
            RestoreHysteresis(self._budget.spill_hysteresis_steps)
            if self._budget.spill_hysteresis_steps > 0
            else None
        )
        # RANK-UNIFORM CLOCK: refreshed once per iteration (MAX all-reduce of
        # time.monotonic over the TP cpu group) when any time-based regler is
        # armed, so every time comparison below reads the SAME value on every
        # rank -- a per-rank clock read would flip decisions at window
        # boundaries and desync the collective sequence.
        self._budget_now = time.monotonic()
        # GDN/Mamba per-session token equivalent (charged to every episode's
        # volume; ~75 MB bf16 on the 27B, length-independent). MAX-reduced
        # once here (a rank-uniform init site) because both inputs are
        # rank-local under uneven TP; the reduce runs only when a volume
        # regler is armed (replicated config -> uniform collective count).
        self._budget_gdn_eq = (
            self._budget_gdn_token_equivalent() if self._budget.has_volume else 0
        )
        if self._budget_armed:
            logger.info(
                "kv-session-offload SPILL BUDGET (#236) armed: %s; "
                "gdn_eq=%d tokens/session, demote_grace=%d iters. First "
                "violated regler wins; exhaustion demotes (drain + radix "
                "handover), never discards.",
                {
                    k: v
                    for k, v in self._budget._asdict().items()
                    if v not in (0, 0.0) or k == "demote_grace_iters"
                },
                self._budget_gdn_eq,
                self._budget.demote_grace_iters,
            )

        # #224 spill destinations: ordered target chain (local host RAM +
        # park tiers). None when --kv-session-offload-destinations is unset
        # -> every _dest-gated site below is inert (byte-identical default).
        from sglang.srt.managers.kv_session_spill_destination import (
            attach_destinations,
        )

        self._dest = attach_destinations(self)

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
        if self.wave_back_min_free_tokens > 0:
            # Only logged when the knob is actually engaged: the default path's
            # log output stays exactly as before.
            logger.info(
                "kv-session-offload (P1): wave-back THRESHOLD armed -- a host "
                "tail is only pulled back once >= %d device slots are free "
                "(rank-uniform min-reduced availability). Deep tails now stay "
                "put under pressure instead of draining one block per "
                "iteration; RESTORE-READY (margin=%d) is unaffected.",
                self.wave_back_min_free_tokens,
                self.restore_margin_tokens,
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


    @property
    def req_to_token_pool(self):
        """The scheduler's CURRENT request pool, read at use (#1040).

        This was a cached object reference. Since the phase flip gives each
        phase its own ``ReqToTokenPool`` and rebinds the scheduler at every
        cutover, a cached reference goes on naming the OUTGOING phase's tensor
        -- and because both pools hold the same number of rows, the writes in
        this file (``req_to_token[req.req_pool_idx, ...]``, ~20 sites) would
        land in range on the wrong pool instead of failing. Reading the
        scheduler here costs one attribute lookup and cannot go stale.
        """
        return self.scheduler.req_to_token_pool
    def _carve_out_draft_scratch(self) -> None:
        """Take the reserved draft-read scratch slots OUT of the allocator's
        accounted size -- and PROVE the write took effect.

        The slots are allocated once and held for the manager's life, so they
        must leave the advertised capacity: the SchedulerInvariantChecker
        balances ``available + evictable + protected + session_held + uncached
        == total`` with ``total = allocator.size``
        (``scheduler_components/invariant_checker.py:124``), and an allocation
        nobody ever frees shows up there as a permanent leak of exactly this
        many slots.

        The write is VERIFIED rather than assumed. On the hybrid composites
        this fork serves (``UnifiedMambaTokenToKVPoolAllocator``,
        ``UnifiedSWATokenToKVPoolAllocator``) ``size`` is a COMPUTED property
        whose setter is a deliberate no-op absorber, so the assignment vanishes
        WITHOUT raising -- the pre-#514 ``try: ... except Exception: pass``
        caught nothing, the carve-out silently did not happen, and the success
        log was printed anyway. A framework's success message about state has
        to be backed by an independent state probe (CLAUDE.md), so the size is
        read back and a carve-out that did not take is refused BY NAME instead
        of running the manager on a capacity figure that is a lie.

        The refusal leaves no partial state (#501 house rule): the reserved
        slots go back to the allocator and the handle is cleared before the
        raise, so the caller's ``spec_in_tick_ready`` teardown sees exactly the
        "could not reserve" state."""
        slices = int(self.mtp_resident_slices)
        name = type(self.allocator).__name__
        before = int(self.allocator.size)
        failure: Optional[str] = None
        try:
            self.allocator.size = before - slices
        except Exception as exc:  # a property with no setter at all
            failure = f"the write raised {type(exc).__name__}: {exc}"
        after = int(self.allocator.size)
        if failure is None:
            failure = draft_scratch_carveout_error(before, after, slices, name)
        if failure is not None:
            self.allocator.free(self._draft_read_scratch)
            self._draft_read_scratch = None
            raise ValueError("kv-session-offload spec-in-tick: " + failure)
        logger.info(
            "kv-session-offload spec-in-tick: reserved %d draft-read "
            "scratch slots (held for the manager's life; index the "
            "shared draft/target slot space, written on the draft pool "
            "only; allocator.size shrunk %d -> %d to keep the leak "
            "invariant balanced). (rank %d)",
            slices,
            before,
            after,
            self.dcp_rank,
        )

    def _effective_spec_algorithm(self):
        """The spec family that would ACTUALLY run on a spill tick right now.

        ``scheduler.spec_algorithm`` is fixed at boot and names the PRIMARY
        family. Under cross-algorithm switching (--speculative-cross-algorithm-
        force auto|policy) the family a forward really takes moves with the
        active rung, so the boot value answers "nextn" while DFLASH is running
        -- and spec-in-tick is only valid for the NEXTN/EAGLE-family drafter
        (the device-resident draft KV, the seed primitive and the C4 verify twin
        are all built for it; DFLASH relays through a different path entirely).

        Thin one-way coupling by design: this READS a public property off the
        spec worker and keeps no state of its own, and the spec worker knows
        nothing about spilling. Any worker that does not publish
        ``active_spec_algorithm`` (i.e. every non-cross-algo worker) falls
        through to the boot value, which is exactly today's behaviour -- so the
        flag-OFF and no-cross-algo paths are unchanged.

        RANK-UNIFORM: the cross-algo worker derives ``active_spec_algorithm``
        from ``_active_name``, which is only written by ``_apply_rung`` from the
        rank-0 rung-id broadcast. Every rank therefore sees the same family at
        the same round boundary, and this adds NO collective.
        """
        dw = getattr(self.scheduler, "draft_worker", None)
        active = getattr(dw, "active_spec_algorithm", None)
        return self.server_spec_algorithm if active is None else active

    def _spec_in_tick_allowed_now(self) -> bool:
        """Runtime half of the DFLASH exclusion (see _effective_spec_algorithm).

        False while the ACTIVE rung is a DFLASH-family drafter, even though the
        boot gate ``spec_in_tick_ready`` passed on the primary family.
        """
        eff = self._effective_spec_algorithm()
        return eff is not None and not eff.is_none() and not eff.is_dflash_family()

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
            # Graph-coverage attribution, diagnostic only: latch what this tick
            # forward was, so the tick trace can say whether the number next to
            # it came from a graph-covered or an eager segment. Reads one
            # boolean off the slot and never harvests it -- see
            # tick_graph_state_from_slot for why that distinction is load-
            # bearing rather than fastidious.
            self._tick_graph_state = tick_graph_state_from_slot(
                kw.get("collective_slot")
            )
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

    def _min_reduce_avail(self, local_avail: int) -> int:
        """all-reduce(MIN) of the per-rank allocator available_size over the TP
        collective. Under uneven DCP the per-rank pool sizes differ, so a spec
        spill tick's device candidate allocation must be gated on the BINDING
        (least-slack) rank -- else ranks diverge on whether to tick and the
        collective spill forward DESYNCs (NCCL hang). Called only when a spill
        tick is due (rank-uniform), so every rank enters this collective at the
        same iteration; MIN over one int is bit-exact -> identical on every
        rank."""
        grp = getattr(self.scheduler, "tp_cpu_group", None)
        if grp is None or torch.distributed.get_world_size(grp) <= 1:
            return int(local_avail)
        t = torch.tensor([int(local_avail)], dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=grp)
        return int(t.item())

    def update_dcp_admission_state(self) -> None:
        """Compute the RANK-UNIFORM admission/decode budget for THIS iteration
        with a SINGLE collective, at a guaranteed-uniform pre-branch call site.

        Root problem (uneven DCP): the per-rank KV allocator available_size AND
        the tree-cache evictable_size both differ per rank (weighted ownership
        [18,23,23]). Every scheduling decision that reads them -- prefill
        admission (rem_total_tokens / cur_rem_tokens / the ignore-eos sort check)
        and the decode-spill trigger (check_decode_mem) -- can therefore flip on
        the binding (least-slack) rank while the slack ranks still fit. A
        divergent decision makes rank 0 take the prefill branch while ranks 1/2
        take the decode branch of get_next_batch_to_run; those branches carry
        DIFFERENT collectives, so the ranks desync (NCCL/gloo hang, observed in
        recv_requests one iteration later). Normally this stays latent because
        the global cap keeps every rank in slack; a spec-in-tick flat scratch
        shrink (disproportionate to ownership) pushes ranks 1/2 to saturation
        while rank 0 has slack -> deterministic divergence.

        Fix: MIN-reduce the pool-derived budget ONCE per iteration here (called
        unconditionally by every rank at the top of get_next_batch_to_run when
        the offload manager is active, so the collective count is ALWAYS
        rank-uniform regardless of the later prefill/decode branch), and have
        every downstream decision read the reduced value instead of its local
        one. Two quantities are packed into one MIN all-reduce:
          * min available_size  -> the decode-spill trigger + try_spill `need`
            (check_decode_mem compares available vs the replicated token demand).
          * min (available + evictable) -> the prefill admission budget
            (rem_total_tokens / cur_rem_tokens add the evictable radix).
        The per-rank surplus over the reduced value is stored as a non-negative
        deficit the PrefillAdder subtracts. Conservative (binding-rank-safe): the
        spill an admission forgoes still fires later at DECODE time (KV growth)
        via the uniform trigger, so this is NOT a "never spills" regression.

        No collective / byte-identical when TP world size is 1. Valid for the
        whole iteration: between here and the admission / decode-mem checks the
        scheduler only builds batches from offsets (ignore-eos) and defers real
        allocation to the forward, so the reduced snapshot still holds."""
        alloc = self.scheduler.token_to_kv_pool_allocator
        tree = self.scheduler.tree_cache
        local_avail = int(alloc.available_size())
        fe = getattr(tree, "full_evictable_size", None)
        local_evict = int(fe() if fe is not None else tree.evictable_size())
        grp = getattr(self.scheduler, "tp_cpu_group", None)
        # #224 destinations: the in-flight park/unpark transfer's rank-local
        # (done, ok) flags ride on THIS reduce -- per-rank network I/O must
        # never feed a decision directly, so state transitions consume only
        # the MIN of these flags. None (flag unset) -> today's 2-element
        # payload, byte-identical.
        _dest = getattr(self, "_dest", None)
        dest_extra = _dest.reduce_extra() if _dest is not None else None
        if grp is None or torch.distributed.get_world_size(grp) <= 1:
            self._dcp_min_avail = local_avail
            self._dcp_budget_deficit = 0
            if _dest is not None and dest_extra is not None:
                _dest.consume_reduced(dest_extra[0], dest_extra[1])
            return
        vals = [local_avail, local_avail + local_evict]
        if dest_extra is not None:
            vals.extend(dest_extra)
        t = torch.tensor(vals, dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=grp)
        self._dcp_min_avail = int(t[0].item())
        # >= 0: the local budget can only exceed the group minimum.
        self._dcp_budget_deficit = (local_avail + local_evict) - int(t[1].item())
        if _dest is not None and dest_extra is not None:
            _dest.consume_reduced(int(t[2].item()), int(t[3].item()))

    def dcp_min_avail(self) -> int:
        """Rank-uniform available_size for THIS iteration (min-reduced in
        update_dcp_admission_state). Used by the decode-spill trigger and
        try_spill so every DCP rank agrees. Falls back to the live local value
        if the per-iteration state was never computed (single-rank / flag-off)."""
        v = getattr(self, "_dcp_min_avail", None)
        if v is None:
            return int(self.allocator.available_size())
        return int(v)

    def dcp_budget_deficit(self) -> int:
        """Non-negative per-rank surplus (available + evictable) over the group
        minimum, subtracted from the prefill admission budget so all DCP ranks
        admit against the binding rank. 0 until first computed / single-rank."""
        return int(getattr(self, "_dcp_budget_deficit", 0))

    # -- #236 spill budget ------------------------------------------------

    def _budget_gdn_token_equivalent(self) -> int:
        """Per-session GDN/Mamba token equivalent, MAX-reduced over the TP cpu
        group. Both inputs (mamba pool bytes, per-token KV bytes) are
        rank-LOCAL under uneven TP, so the raw quotient would differ per rank
        and a volume comparison against it would desync; the init-time MAX
        (conservative: charges the largest rank's share everywhere) makes it a
        replicated constant. Unknown pools charge 0 (dense models)."""
        local = 0
        try:
            mp = getattr(self.req_to_token_pool, "mamba_pool", None)
            fp = self.full_pool
            if mp is not None and fp is not None:
                n_slots = int(getattr(mp, "size", 0) or 0)
                gib = float(getattr(mp, "mem_usage", 0.0) or 0.0)
                per_token = (
                    int(getattr(fp, "head_num", 0))
                    * int(getattr(fp, "head_dim", 0))
                    * int(getattr(fp, "layer_num", 0))
                    * int(getattr(fp, "store_dtype", torch.bfloat16).itemsize)
                    * 2
                )
                state_itemsize = GDN_STATE_MIN_ITEMSIZE
                cache = getattr(mp, "mamba_cache", None)
                ts = getattr(cache, "temporal_state", None)
                if ts is not None and hasattr(ts, "dtype"):
                    state_itemsize = ts.dtype.itemsize
                if n_slots > 0 and gib > 0:
                    local = gdn_token_equivalent(
                        int(gib * (1 << 30) / (n_slots + 1)),
                        per_token,
                        state_itemsize,
                    )
        except ValueError:
            raise  # a quantized GDN state is a hard error, never accounted
        except Exception as e:  # noqa: BLE001 -- sizing only, never fatal
            logger.warning(
                "kv-session-offload budget: GDN token equivalent unavailable "
                "(%r); GDN states are NOT charged to the volume budgets.",
                e,
            )
            local = 0
        grp = getattr(self.scheduler, "tp_cpu_group", None)
        if grp is None or torch.distributed.get_world_size(grp) <= 1:
            return local
        t = torch.tensor([local], dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=grp)
        return int(t.item())

    def _budget_begin_iteration(self) -> None:
        """Once per pre_schedule (unconditional when armed): refresh the
        rank-uniform clock and refill the rate bucket from it. The clock
        reduce fires every iteration while a time-based regler is armed --
        gated purely on replicated config, so the collective count is
        identical on every rank."""
        if self._budget.needs_clock:
            now = time.monotonic()
            grp = getattr(self.scheduler, "tp_cpu_group", None)
            if grp is not None and torch.distributed.get_world_size(grp) > 1:
                t = torch.tensor([now], dtype=torch.float64)
                torch.distributed.all_reduce(
                    t, op=torch.distributed.ReduceOp.MAX, group=grp
                )
                # MAX = the furthest-along rank's monotonic clock. Single-node
                # scope: CLOCK_MONOTONIC shares one base across all local
                # processes, so the reduced value is a coherent point in time.
                now = float(t.item())
            self._budget_now = now
            if self._budget_bucket is not None:
                self._budget_bucket.advance(now)

    def _budget_resident_volumes(self):
        """(total, prefill, decode, per_slot) resident host-token volumes,
        recomputed from replicated slot state every call (no incremental
        drift). Each session is charged its host tail plus the GDN token
        equivalent; a born-spilled session's write-once prompt share counts
        as prefill, growth on top as decode."""
        per_slot = {}
        tot_prefill = 0
        tot_decode = 0
        for rpi, slot in self.spills.items():
            req = slot.req
            L = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
            b = int(getattr(req, "kv_spill_boundary", 0) or 0)
            tail = max(0, L - b)
            sess = tail + self._budget_gdn_eq
            per_slot[rpi] = sess
            if slot.budget_phase == "prefill":
                pre = min(tail, int(slot.budget_initial_tail))
                tot_prefill += pre + self._budget_gdn_eq
                tot_decode += tail - pre
            else:
                tot_decode += sess
        return tot_prefill + tot_decode, tot_prefill, tot_decode, per_slot

    def _budget_note_spill(self, slot, phase: str, spilled_tokens: int) -> None:
        """Register a freshly opened episode (decode spill or born-spilled
        prefill) with the budget: phase classification, episode clock start,
        cumulative counters, and the rate-bucket charge for the D2H volume."""
        slot.budget_phase = phase
        slot.budget_initial_tail = int(spilled_tokens)
        slot.budget_episode_start = self._budget_now
        self._budget_counters.episodes_started += 1
        if phase == "prefill":
            self._budget_counters.spilled_tokens_prefill += int(spilled_tokens)
        else:
            self._budget_counters.spilled_tokens_decode += int(spilled_tokens)
        if self._budget_bucket is not None:
            self._budget_bucket.consume(int(spilled_tokens))

    def budget_session_cap(self) -> int:
        """Concurrent-spilled-session cap in force, 0 = uncapped.

        #287: the #236 ``budget_max_sessions`` regler and the floating
        admission limit bound the same quantity, so the spill path reads the
        limiter instead of keeping a second number -- a server that has
        throttled itself down to N sessions must not keep more than N parked
        on the host. Rank-uniform: the limiter only ever moves on replicated
        inputs, so every rank answers the same cap here (a rank-divergent
        spill verdict is the desync this file guards against everywhere).
        """
        return spill_session_cap(self._budget.max_sessions, current_admission_limiter())

    def _budget_admission_check(self, spill_tokens: int) -> Optional[str]:
        """Budget verdict for a candidate DECODE spill of ``spill_tokens``
        host tokens; returns the violated regler (decline -> stock
        retraction) or None (admit). Pure over replicated volumes."""
        total, pre, dec, _ = self._budget_resident_volumes()
        sess_after = int(spill_tokens) + self._budget_gdn_eq
        reason = budget_admission_violation(
            self._budget._replace(max_sessions=self.budget_session_cap()),
            n_open_slots=len(self.spills),
            spill_tokens=int(spill_tokens),
            phase="decode",
            session_tokens_after=sess_after,
            prefill_tokens_after=pre,
            decode_tokens_after=dec + sess_after,
            total_tokens_after=total + sess_after,
            rate_ready=(
                self._budget_bucket.ready() if self._budget_bucket is not None else True
            ),
        )
        if reason is not None:
            self._budget_counters.admission_declines += 1
            self._budget_counters.note_exhaustion(reason)
        return reason

    def _budget_blocked_victims(self, reqs) -> Optional[set]:
        """Indices in ``reqs`` currently under the post-restore cooldown
        (progress lock / time cap), or None when the cooldown is off/idle."""
        if self._budget_cooldown is None:
            return None
        blocked = {
            i
            for i, r in enumerate(reqs)
            if self._budget_cooldown.blocked(r.rid, len(r.output_ids), self._budget_now)
        }
        return blocked or None

    def _budget_evaluate_episodes(self) -> None:
        """Once per iteration while sessions are spilled: demote episodes
        whose budget is exhausted, and release the finishing tick of demoted
        sessions whose drain grace expired. Deterministic slot order (by
        region index -- replicated) and replicated inputs -> every rank
        demotes the same sessions at the same iteration."""
        cfg = self._budget
        total, pre, dec, per_slot = self._budget_resident_volumes()
        now = self._budget_now
        ordered_items = sorted(self.spills.items(), key=lambda kv: kv[1].region)
        ordered = [slot for _, slot in ordered_items]
        # Per-session reglers first (most specific).
        for rpi, slot in ordered_items:
            if slot.budget_demoted:
                continue
            elapsed = (
                now - float(slot.budget_episode_start)
                if cfg.episode_seconds > 0
                else 0.0
            )
            reason = budget_episode_violation(
                cfg,
                session_tokens=per_slot[rpi],
                episode_elapsed_s=elapsed,
            )
            if reason is not None:
                self._budget_demote(slot, reason)

        # Aggregate reglers: one demotion per class per iteration, youngest
        # live session of the class (FCFS-consistent; re-evaluated next iter).
        def _youngest(phase=None):
            live = [
                s
                for s in ordered
                if not s.budget_demoted and (phase is None or s.budget_phase == phase)
            ]
            if not live:
                return None
            return max(
                live,
                key=lambda s: getattr(s.req, "kv_arrival_seq", -1) or -1,
            )

        if cfg.prefill_tokens > 0 and pre > cfg.prefill_tokens:
            v = _youngest("prefill")
            if v is not None:
                self._budget_demote(v, "prefill-tokens")
        if cfg.decode_tokens > 0 and dec > cfg.decode_tokens:
            v = _youngest("decode")
            if v is not None:
                self._budget_demote(v, "decode-tokens")
        if cfg.total_tokens > 0 and total > cfg.total_tokens:
            v = _youngest()
            if v is not None:
                self._budget_demote(v, "total-tokens")
        # Drain grace: a demoted session waits (not ticking) for its tail to
        # drain to device, where the stock finish donates the full prefix.
        # Past the grace it gets ONE budget-exempt host tick to finish and
        # deliver -- the coarse fallback that keeps the client from hanging
        # when device memory never frees (the tail is then dropped, counted).
        for slot in ordered:
            if not slot.budget_demoted or slot.budget_tick_release:
                continue
            if self._iter_ct - slot.budget_demote_iter > cfg.demote_grace_iters:
                slot.budget_tick_release = True
                # Quiescent since the demote (no ticks ran) -> the cap is
                # race-free here; the released tick appends the finishing
                # token and update_finish_state ends the session on host.
                self._budget_finish_cap(slot.req, extra=1)
                self._log(
                    "kv-session-offload BUDGET: demoted rid=%s drain grace "
                    "(%d iters) expired -> finishing on host (tail dropped).",
                    slot.req.rid,
                    cfg.demote_grace_iters,
                )

    def _budget_demote(self, slot, reason: str) -> None:
        """HERABSTUFUNG, not an abort (#236): end the episode's liveness and
        keep its work. Generation is capped at the current output (the client
        receives everything produced so far, finished as length); the session
        stops ticking and its host tail drains through the UNCHANGED
        wave-back/restore machinery -- restore-readiness keeps counting the
        radix-evictable memory (#217), the budget adds no readiness predicate
        of its own. Once drained, the stock device finish donates the full
        row to the radix tree: a continuation is a prefix hit, not a full
        re-prefill. Under HiRadixCache the donated node then migrates to the
        host tier by HiCache's own write-through/eviction policy.

        PREFIX-KEY PRODUCER IDENTITY (checked, documented): the in-process
        radix key is (token ids, extra_key) -- producer identity (model,
        quantization, geometry) is implicit and safe because one scheduler
        process serves exactly one model configuration for its lifetime. The
        persistent HiCache STORAGE tier keys additionally carry model_name
        and the TP/PP/CP geometry (hicache_storage.py config_suffix) but NOT
        the KV dtype/quantization -- a shared storage backend can therefore
        collide across server runs of the same model with different
        --kv-cache-dtype. Finding reported upstream, not worked around here.

        Under the spec host-finish guard (spec active, KVSO_RESUME unset) no
        restore path exists, so the drain grace is skipped: the session
        finishes on its next host tick (tail dropped -- the handover cannot
        preserve it on that configuration).

        Rank-uniform: reason and cap derive from replicated state."""
        # #224 <-> #236 SEAM (wired at the merge): when the destination chain
        # is armed and the slot is park-eligible, the exhaustion EXTENDS into
        # the next tier instead of demoting -- work AND liveness kept, merely
        # suspended. Guarding here covers every demotion site uniformly; the
        # committed park pops the slot from self.spills, so the recomputed
        # _budget_resident_volumes discounts it structurally. False (chain
        # unarmed, busy, or slot ineligible) -> demote exactly as before.
        if self.park_instead_of_demote(slot):
            self._log(
                "kv-session-offload BUDGET: rid=%s budget exhausted (%s) -> "
                "parked to next tier instead of demoted (#224 chain armed).",
                slot.req.rid,
                reason,
            )
            return
        req = slot.req
        slot.budget_demoted = True
        slot.budget_demote_iter = self._iter_ct
        # LOSSLESS HAND-OVER (#242): the donation at the end of the drain is
        # the session's last chance to reach the host tier -- the device slots
        # are freed by the very finish that donates them. HiCache's write-
        # through is hit-rate driven (write_through_threshold; write_back defers
        # to eviction), so left to itself it hands over the shared prefix and
        # silently drops the leaves under the threshold -- precisely the tokens
        # this session just produced. Mark the request so its finishing insert
        # writes the whole chain through (see
        # ``requests_forced_host_write_through``). Rank-uniform: the demotion
        # decision is replicated, so every rank marks the same request. Nothing
        # else about the finish changes, and an unmarked request keeps the stock
        # heuristic byte for byte.
        req.force_host_write_through = True
        self._budget_counters.episodes_demoted += 1
        self._budget_counters.note_exhaustion(reason)
        # LIVENESS ends here (the tick exclusion below stops all host decode),
        # but the max_new_tokens CAP is deliberately NOT set yet. Capping at
        # the demote instant races the in-flight tick pipeline: the pending
        # tick result runs update_finish_state with len(output) already at the
        # cap, so the session finishes ON HOST in the same iteration and the
        # release path (head freed, no radix insert) swallows the handover --
        # demotions_drained would be structurally unreachable (GPU-measured:
        # DEMOTING and 'finished on host' in the same second). The cap is
        # applied at the two QUIESCENT completion points instead:
        # _finalize_restore (drain done -> device finish + donation) and the
        # tick_release grant (grace expiry / spec host-finish guard).
        spec_algo = getattr(self.scheduler, "spec_algorithm", None)
        if (
            spec_algo is not None
            and not spec_algo.is_none()
            and not resume_under_spec_enabled()
        ):
            slot.budget_tick_release = True
            self._budget_finish_cap(req, extra=1)
        self._log(
            "kv-session-offload BUDGET: DEMOTING rid=%s (budget '%s' "
            "exhausted) at %d output tokens: liveness ends (no further host "
            "ticks); work is kept -- the spilled tail drains and the finish "
            "donates the prefix (continuation = prefix hit). "
            "host_finish_fallback=%s",
            req.rid,
            reason,
            len(req.output_ids),
            slot.budget_tick_release,
        )

    @staticmethod
    def _budget_finish_cap(req, extra: int) -> None:
        """Cap a demoted session's generation at its settled output length
        (+``extra`` for a path whose finishing step must still append a
        token). Called only at quiescent completion points -- never at the
        demote instant (see _budget_demote). update_finish_state then ends
        the session with FINISH_LENGTH and finished_len == the cap, so the
        client receives exactly what was produced."""
        sp = getattr(req, "sampling_params", None)
        if sp is None:
            return
        cap = max(1, len(req.output_ids) + max(0, int(extra)))
        if getattr(sp, "max_new_tokens", None) is None or sp.max_new_tokens > cap:
            sp.max_new_tokens = cap

    def budget_stats(self) -> dict:
        """Counters + the four residency states, for the dashboard. The
        'retracted' state is not directly observable from the manager (stock
        retraction runs in the scheduler); its budget-caused share is the
        admission_declines counter."""
        live = sum(1 for s in self.spills.values() if not s.budget_demoted)
        demoted_pending = len(self.spills) - live
        rb = getattr(self.scheduler, "running_batch", None)
        device_resident = len(getattr(rb, "reqs", ()) or ()) if rb is not None else 0
        out = self._budget_counters.as_dict()
        out.update(
            {
                "state_device_resident": device_resident,
                "state_spilled_live": live,
                "state_demoted_pending": demoted_pending,
                "state_retracted_by_budget_decline": (
                    self._budget_counters.admission_declines
                ),
                "gdn_token_equivalent": self._budget_gdn_eq,
                "rate_bucket_level": (
                    self._budget_bucket.level
                    if self._budget_bucket is not None
                    else None
                ),
            }
        )
        return out

    def prefill_spill_free_regions(self) -> int:
        """Prefill-Spill (PS1-V1a): number of free host regions available to
        born-spill a would-be-wedged prompt this iteration. 0 when the feature
        is off -> the PrefillAdder relaxation is inert (byte-identical).

        RANK-UNIFORM without a collective (R2-safe): host regions are claimed /
        freed ONLY in rank-uniform spill / restore / finish events, and every
        region is max_ratio-sized (holds any rank's per-rank shard of a full-
        context session), so `len(self._free_regions)` is identical on every DCP
        rank. `region_tokens` and `max_spills` are replicated config. The adder
        reads this replicated count and decrements it identically per rank as it
        admits born-spilled prompts, so the born-spilled verdict never diverges
        -> no branch-local collective, no desync."""
        if not self.prefill_spill:
            return 0
        # #236: the budget gates born-spill ADMISSION coarsely through the
        # region count the PrefillAdder already consults -- a closed gate
        # reads as "no free region" and the prompt takes today's chunk/wedge
        # path. Coarse on purpose: the adder knows the prompt length, this
        # gate does not, so the LAST admitted prompt may overshoot a volume
        # regler by one prompt; the overshoot then demotes through the
        # episode evaluation. All inputs replicated -> rank-uniform.
        if getattr(self, "_budget_armed", False):
            cfg = self._budget
            closed = False
            session_cap = self.budget_session_cap()
            if session_cap > 0 and len(self.spills) >= session_cap:
                closed = True
            elif cfg.has_volume:
                total, pre, _dec, _ = self._budget_resident_volumes()
                if cfg.prefill_tokens > 0 and pre >= cfg.prefill_tokens:
                    closed = True
                if cfg.total_tokens > 0 and total >= cfg.total_tokens:
                    closed = True
            if not closed and self._budget_bucket is not None:
                closed = not self._budget_bucket.ready()
            if closed:
                self._budget_counters.prefill_gate_closures += 1
                return 0
        return len(self._free_regions)

    # -- slot bookkeeping -------------------------------------------------

    def _slot_of(self, req) -> Optional[SpillSlot]:
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
        self, batch: ScheduleBatch, fast_pressure=None, need: Optional[int] = None
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
            if getattr(self, "_dest", None) is not None:
                # #224: a spill declined for lack of a region is FRESH
                # pressure -- the signal that lets the destination chain
                # park the oldest spilled session to free a region for the
                # NEXT spill (this one still falls back to stock retraction,
                # honestly: a network tier cannot free a region in-line).
                # `manager=self` lets the #659 ladder report name WHICH tier
                # was full and by how much; without it the shortfall is only a
                # timestamp. Read-only -- the destination chain still decides.
                self._dest.note_region_shortfall(self._iter_ct, manager=self)
            return False
        if fast_pressure is None:
            fast_pressure = self._fast_lane_pressure(batch.reqs)
        budget_armed = getattr(self, "_budget_armed", False)
        session_cap = self.budget_session_cap()
        if session_cap > 0 and len(self.spills) >= session_cap:
            # #236 breadth regler: at the session-count budget, decline before
            # any victim work -- stock retraction handles the pressure, today's
            # no-free-region behaviour.
            self._budget_counters.admission_declines += 1
            self._budget_counters.note_exhaustion("max-sessions")
            return False
        # Shortfall X (tokens) + per-session spillable sizes -> minimal
        # single-session eviction (youngest sufficient; see
        # select_spill_victim). Only req-exclusive slots count as freed
        # (the shared radix prefix stays tree-owned, merely evictable).
        if need is None:
            # RANK-UNIFORM shortfall under uneven DCP: use the iteration's
            # min-reduced available (dcp_min_avail, computed once at the
            # unconditional pre-branch site) so the freed-token target `need`
            # (-> victim selection + block-aligned spill boundary) is IDENTICAL
            # on every rank -- no extra collective here (the reduce already
            # happened). `new_tokens_required_next_decode` is replicated batch
            # metadata -> uniform. Single-rank: dcp_min_avail == local available
            # -> byte-identical to the old local computation.
            need = max(
                0,
                batch.new_tokens_required_next_decode() - self.dcp_min_avail(),
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
        # #236 cooldown: recently restored sessions are excluded as victims
        # (progress lock / time cap). Pendulum accounting: when the cooldown
        # changes the outcome, a Spill->Restore->Spill round inside the lock
        # was just PREVENTED (pendulum_blocked); an ACTUAL round inside the
        # lock (pendulum_events) is what the exclusion makes impossible and
        # is counted defensively below, at the spill commit.
        blocked = self._budget_blocked_victims(batch.reqs) if budget_armed else None
        idx = select_spill_victim(
            batch.reqs,
            sizes=sizes,
            need=need,
            fast_pressure=fast_pressure,
            blocked=blocked,
        )
        if blocked and idx != select_spill_victim(
            batch.reqs, sizes=sizes, need=need, fast_pressure=fast_pressure
        ):
            self._budget_counters.pendulum_blocked += 1
            if self._budget_counters.pendulum_blocked in (1, 8) or (
                self._budget_counters.pendulum_blocked % 64 == 0
            ):
                self._log(
                    "kv-session-offload BUDGET: cooldown excluded the natural "
                    "spill victim (pendulum prevented; blocked=%d, actual=%d).",
                    self._budget_counters.pendulum_blocked,
                    self._budget_counters.pendulum_events,
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
            # The FCFS/minimal-eviction pick is unreachable under the back-only
            # rule. Stock retraction is back-only too, so the BACK-MOST request
            # is the one that gets evicted either way -- the only open question
            # is whether its work survives. Offer it to the spill (work kept on
            # host, session decodes on through the tick) instead of declining
            # into a retraction that frees the same slots and throws the work
            # away. Eligibility is re-checked by the SAME rules, so a fast-lane
            # or 'never' back-most request still declines.
            alt = spec_back_only_victim(
                batch.reqs,
                sizes=sizes,
                need=need,
                fast_pressure=fast_pressure,
                blocked=blocked,
            )
            if alt is None:
                logger.debug(
                    "kv-session-offload: spec active, victim rid=%s at idx=%d is "
                    "not the back-most request (n=%d) and the back-most one may "
                    "not be spilled; declining spill (back-only under spec) -> "
                    "stock retraction handles the pressure.",
                    req.rid,
                    idx,
                    len(batch.reqs),
                )
                return False
            self._log(
                "kv-session-offload: spec active, policy victim rid=%s at "
                "idx=%d is not the back-most request (n=%d); spilling the "
                "back-most rid=%s instead -- stock retraction would evict "
                "exactly that request and lose its work.",
                req.rid,
                idx,
                len(batch.reqs),
                batch.reqs[alt].rid,
            )
            idx = alt
            req = batch.reqs[idx]
            if req.finished() or getattr(req, "to_finish", None) is not None:
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
        #
        # #501: the reclaim is PLANNED here and COMMITTED below, past the LAST
        # decline point. It is not a reversible bookkeeping tweak -- it hands
        # device slots back to the allocator and sets req.kv_overallocated_freed,
        # which pop_overallocated_kv_cache asserts is still False
        # (schedule_batch.py:1141). EVERY decline in this function returns False
        # and leaves the request RUNNING in the batch, so committing the reclaim
        # before a decline armed that assert on a live request: the request's
        # eventual stock retraction / finish killed the scheduler instead of
        # falling back cleanly. Planning is pure, and the predicates between here
        # and the commit are unaffected by the reclaim: they read the snapshot,
        # the row HEAD [0, L) and replicated budget state, while the reclaim only
        # ever frees [snap.free_from, kv_allocated_len) -- and snap.free_from
        # equals L whenever it fires (true_L under spec+overlap, the
        # L == kv_committed_len conjunct otherwise), so the two ranges are
        # disjoint by construction.
        reclaim_overhang = (
            req.kv_allocated_len > snap.free_from
            and (spec_overlap or L == req.kv_committed_len)
            and not getattr(req, "kv_overallocated_freed", False)
        )
        allocated_after_reclaim = (
            snap.free_from if reclaim_overhang else req.kv_allocated_len
        )

        # After reclaiming the overhang, allocated must equal L; the planned
        # value is validated here so this check keeps its original position in
        # the sequence without depending on a mutation. Under spec+overlap
        # kv_committed_len legitimately still lags L by the pending
        # (not-yet-committed) accept count -- the deferred result processor
        # settles it this same iteration -- so it is validated only OFF the
        # spec+overlap path.
        committed_ok = spec_overlap or (L == req.kv_committed_len)
        if not committed_ok or L != allocated_after_reclaim:
            # Never spill a request whose slot bookkeeping we STILL do not
            # understand after reclaiming the speculative overhang -- stock
            # retraction handles it.
            logger.warning(
                "kv-session-offload: skip spill of rid=%s (L %d, "
                "committed %d, allocated %d -> %d after the planned overhang "
                "reclaim, spec_overlap=%s); falling back to retraction.",
                req.rid,
                L,
                req.kv_committed_len,
                req.kv_allocated_len,
                allocated_after_reclaim,
                spec_overlap,
            )
            return False

        # S1b PARTIAL SPILL: only the block-aligned TAIL overhang migrates to
        # host. boundary splits the row into a device-resident head
        # [0, boundary) (kept, tree-locked) and a host tail [boundary, L).
        protected = int(req.cache_protected_len or 0)
        boundary, spill_count = partial_spill_plan(L, protected, need, self.block_size)
        if spill_count <= 0:
            # need <= 0 after the internal recompute: nothing to free. Leave
            # the batch untouched (the stock retract path decides).
            return False
        spill_margin = spill_count - need  # over-eviction metric (<= block-1)

        # #236 volume/rate reglers, checked BEFORE any region claim, D2H or the
        # draft-overhang reclaim (#501), so a decline leaves no partial state --
        # neither on the manager nor on the request. First violated regler wins;
        # decline -> stock retraction (today's fallback), never a corrupt
        # spill. Replicated inputs -> rank-uniform verdict.
        if budget_armed:
            _viol = self._budget_admission_check(spill_count)
            if _viol is not None:
                self._log(
                    "kv-session-offload BUDGET: spill of rid=%s (%d tokens) "
                    "declined by regler '%s' -> stock retraction.",
                    req.rid,
                    spill_count,
                    _viol,
                )
                return False

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
        # RANK-UNIFORM region check. `n_own` is this rank's owned row count and
        # is rank-LOCAL under uneven DCP (different owner-window widths), while
        # `region_tokens` is replicated -- comparing the two decided SPILL vs.
        # STOCK RETRACTION per rank, i.e. two different collective sequences in
        # the same iteration. The verdict is taken on the WIDEST rank instead
        # (`spill_tail_rows_max_over_ranks`, computed locally from replicated
        # inputs -- no collective, see its docstring); `n_own` stays in the log
        # line because the local count is what a rank-side trace shows.
        n_own_max = spill_tail_rows_max_over_ranks(
            seg,
            mode=self.mode,
            S=self.S,
            cp_prefix=self.cp_prefix,
            dcp_size=self.dcp_size,
            boundary=boundary,
            L=L,
        )
        if n_own_max > self.region_tokens:
            logger.warning(
                "kv-session-offload: session rid=%s tail needs %d host rows on "
                "the widest rank (%d on rank %d) > region %d; falling back to "
                "stock retraction (rank-uniform verdict).",
                req.rid,
                n_own_max,
                n_own,
                self.dcp_rank,
                self.region_tokens,
            )
            return False

        # COMMIT the planned draft-overhang reclaim (#501). Every decline point
        # of this function is behind us -- from here on try_spill runs to
        # completion and the victim leaves the batch -- so the free and the
        # kv_overallocated_freed flag can no longer strand on a live request.
        # Ordering is unchanged from the pre-#501 code on both paths: under
        # spec+overlap the _wait_forward_stream() that precedes the snapshot
        # still orders this free after the in-flight forward, and the plain path
        # still frees without one (committed is settled synchronously there).
        if reclaim_overhang:
            over = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, snap.free_from : req.kv_allocated_len
            ]
            self.allocator.free(over.to(torch.int64))
            req.kv_allocated_len = snap.free_from
            req.kv_overallocated_freed = True

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
                torch.arange(boundary, L, dtype=torch.int64, device=row.device) % self.S
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

        # C2 (spec-in-spill-tick, Option b'): if this session is routed for
        # spec-in-tick and its draft tail fits the resident cap, snapshot the
        # draft KV into a DEVICE buffer (kept resident so draft() runs on device
        # while spilled). Overflow -> graceful plain host tick for this session
        # (spec_in_tick stays False; NOT an OOM). Captured BEFORE the free below
        # reclaims the slots, ordered after the in-flight forward by the
        # _wait_forward_stream() already issued above.
        draft_dev_k = draft_dev_v = None
        session_spec_in_tick = False
        # R1: the boot gate is not enough -- under cross-algo the ACTIVE rung
        # decides which drafter runs. Do not route a session into spec-in-tick
        # while a DFLASH-family rung is active; it would take the plain tick on
        # its very first tick anyway, and the device draft-KV snapshot below
        # would be wasted VRAM.
        if self.spec_in_tick_ready and not self._spec_in_tick_allowed_now():
            self._log(
                "kv-session-offload spec-in-tick: rid=%s admitted while the "
                "ACTIVE cross-algo rung is %s (DFLASH family) -> plain host "
                "tick (no spec) for this session.",
                req.rid,
                self._effective_spec_algorithm(),
            )
        elif self.spec_in_tick_ready:
            tail_tokens = L - boundary  # full tail (draft pool not DCP-sharded)
            if mtp_resident_tail_fits(tail_tokens, self.mtp_resident_slices):
                cap = (
                    self.mtp_resident_slices
                    if self.mtp_resident_slices > 0
                    else self.region_tokens
                )
                draft_dev_k, draft_dev_v, _ndev = self._draft_kv_snapshot_device(
                    seg, cap
                )
                session_spec_in_tick = True
            else:
                self._log(
                    "kv-session-offload spec-in-tick: rid=%s draft tail %d "
                    "exceeds resident cap %d -> graceful plain host tick (no "
                    "spec) for this session.",
                    req.rid,
                    tail_tokens,
                    self.mtp_resident_slices,
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
        if draft_k_snap is not None:
            slot.draft_kv_k = draft_k_snap
            slot.draft_kv_v = draft_v_snap
            slot.draft_spill_boundary = boundary
            slot.draft_spill_L = L
        if draft_dev_k is not None:
            # C2 (spec-in-spill-tick): the device-resident draft-KV tail. The
            # draft() surgery (d4) copies [0, draft_dev_len) into draft-pool
            # scratch slots around the tick and appends accepted-token draft KV.
            slot.draft_dev_k = draft_dev_k
            slot.draft_dev_v = draft_dev_v
            slot.draft_dev_len = L - boundary
            slot.draft_spill_boundary = boundary
            slot.draft_spill_L = L
            slot.spec_in_tick = True
        stamp_spill(slot, getattr(self.scheduler, "phase_flip_active_stack", None))
        self.spills[req.req_pool_idx] = slot
        if budget_armed:
            # Episode opened: decode phase, clock started, D2H volume charged.
            self._budget_note_spill(slot, "decode", spill_count)
            if self._budget_cooldown is not None and self._budget_cooldown.in_window(
                req.rid, len(req.output_ids), self._budget_now
            ):
                # ACTUAL Spill->Restore->Spill inside the cooldown window --
                # structurally excluded by the blocked() filter above, counted
                # so a regression is a number (must stay 0 while armed).
                self._budget_counters.pendulum_events += 1
                self._log(
                    "kv-session-offload BUDGET: PENDULUM round for rid=%s "
                    "inside its cooldown window (events=%d) -- the progress "
                    "lock failed to exclude it.",
                    req.rid,
                    self._budget_counters.pendulum_events,
                )

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

    def _build_spill_batch(self, req: Req) -> ScheduleBatch:
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
        #
        # C3/d1 (spec-in-spill-tick, Option b'): when this session is routed
        # for spec-in-tick (try_spill snapshotted its draft KV DEVICE-resident,
        # slot.spec_in_tick=True) the tick runs the model-configured NEXTN/EAGLE
        # drafter DURING the spill: keep the SERVER spec algorithm on the batch
        # so prepare_for_decode -> eagle_prepare_for_decode allocates candidate
        # slots and forward_batch_generation takes the real draft->verify path.
        # DFLASH is EXCLUDED for spilled (long-context) sessions. That exclusion
        # has TWO halves: spec_in_tick_ready gates out a DFLASH-family BOOT
        # configuration, and _spec_in_tick_allowed_now() gates out a DFLASH
        # ACTIVE RUNG under cross-algorithm switching -- the boot value names the
        # primary family and stays "nextn" while DFLASH is running, so the boot
        # gate alone would let a DFLASH rung drive the spill tick (R1). Do NOT
        # rely on the DFLASH ctx-gate to cover this: it is a coincidence between
        # two features that do not know each other, and retire-ctx defaults off.
        # OFF path is byte-identical: slot None / spec_in_tick False ->
        # spec_algorithm stays NONE.
        # spec-in-tick armed from the FIRST tick (no plain bootstrap tick, so the
        # committed L never shifts out of lockstep with the frozen host region /
        # draft_dev). The first tick has no draft seed yet -> the worker runs a
        # TRIVIAL 1-node verify (device-suffix token, captures last_hidden); every
        # later tick re-seeds the draft from that captured hidden. Both go through
        # the C4 verify twin, so the committed prefix stays head + frozen host
        # sentinels + growing device suffix (Option A).
        _slot = self.spills.get(req.req_pool_idx)
        # R1 runtime gate: an ALREADY-ARMED session whose active rung switched to
        # DFLASH mid-flight must stop running spec on the tick. Disarm PERMANENTLY
        # via the established idiom (see spec_in_tick_append_accepted's resident-
        # buffer-full path): spec_in_tick=False + batch=None forces a plain
        # rebuild. Permanent and not re-armed on a switch back to NEXTN on
        # purpose -- re-arming mid-session would shift the committed L out of
        # lockstep with the frozen host region / draft_dev, which is exactly the
        # invariant the "armed from the FIRST tick" note below protects. Losing
        # the spec speedup on an already-spilled session is a graceful
        # degradation; running DFLASH over the device-resident NEXTN draft KV is
        # not.
        if (
            self.spec_in_tick_ready
            and _slot is not None
            and getattr(_slot, "spec_in_tick", False)
            and not self._spec_in_tick_allowed_now()
        ):
            self._log(
                "kv-session-offload spec-in-tick: rid=%s active cross-algo rung "
                "switched to %s (DFLASH family) -> disarming spec-in-tick "
                "permanently for this session, plain host tick from now.",
                req.rid,
                self._effective_spec_algorithm(),
            )
            _slot.spec_in_tick = False
            _slot.batch = None  # force rebuild as plain (spec_algorithm=NONE)
        _spec_in_tick = (
            self.spec_in_tick_ready
            and _slot is not None
            and getattr(_slot, "spec_in_tick", False)
        )
        _server_spec = getattr(sch, "spec_algorithm", None)
        _batch_spec = (
            _server_spec
            if (_spec_in_tick and _server_spec is not None)
            else SpeculativeAlgorithm.NONE
        )
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=sch.req_to_token_pool,
            token_to_kv_pool_allocator=sch.token_to_kv_pool_allocator,
            tree_cache=sch.tree_cache,
            model_config=sch.model_config,
            enable_overlap=sch.enable_overlap,
            spec_algorithm=_batch_spec,
        )
        batch.kv_session_spill_tick = True
        if _spec_in_tick:
            self._log(
                "kv-session-offload spec-in-tick: rid=%s spill batch armed with "
                "server spec algo %s (draft KV device-resident, len=%d) (rank %d)",
                req.rid,
                _server_spec,
                getattr(_slot, "draft_dev_len", 0),
                self.dcp_rank,
            )
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
        batch.req_pool_indices_cpu = torch.tensor([req.req_pool_idx], dtype=torch.int64)
        seq_len = spill_tick_seq_len(len(req.origin_input_ids), len(req.output_ids))
        assert seq_len is not None, (
            "kv-session-offload tick build: rid=%s has no output token yet "
            "(born-spilled, prefill result not processed); _pick_tick_slot "
            "must defer such a session" % req.rid
        )
        assert seq_len == req.kv_committed_len, (
            f"kv-session-offload tick build: seq_len {seq_len} != committed "
            f"{req.kv_committed_len} (rid={req.rid})"
        )
        batch.seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
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

    def spill_decode_alloc(self, batch: ScheduleBatch) -> torch.Tensor:
        """prepare_for_decode replacement for the spill tick: no device
        allocation -- assign the next sentinel slot and write the
        req_to_token row. Counter updates stay in prepare_for_decode."""
        assert len(batch.reqs) == 1 and batch.kv_session_spill_tick
        req = batch.reqs[0]
        p = int(batch.seq_lens_cpu[0].item())
        res = new_token_residue(p, self.S)
        sent = self.host_base + p * self.S + res
        self.req_to_token_pool.req_to_token[req.req_pool_idx, p] = sent
        if getattr(self, "_budget_armed", False):
            # #236: one token grown on host per tick (decode-phase volume;
            # resident volumes are recomputed from state, this is the
            # cumulative counter only).
            self._budget_counters.spilled_tokens_decode += 1
        return torch.tensor([sent], dtype=torch.int64, device=batch.seq_lens.device)

    def prefill_spill_extend_ready(self, batch: ScheduleBatch) -> bool:
        """Whether ``prepare_for_extend`` must take the PS2 born-spilled-deep
        allocation instead of ``alloc_for_extend``.

        ALL-OR-NOTHING per batch (S3b.3 stage A): a mixed extend batch would
        make ``out_cache_loc`` span two disjoint address spaces (real device
        slots + host sentinels) in one tensor, which the write scatter cannot
        split. The PrefillAdder guarantees the separation
        (``_admit_born_spilled_deep`` refuses to join a batch that already
        holds normal requests and vice versa); this is the enforcement, and a
        violation raises rather than silently corrupting the slot space.

        Rank-uniform: ``born_spilled_deep`` is set by the replicated admission
        verdict, so every rank sees the same partition and takes the same
        branch -> identical collective sequence in the forward."""
        if not self.prefill_spill or not batch.reqs:
            return False
        deep = [bool(getattr(r, "born_spilled_deep", False)) for r in batch.reqs]
        if not any(deep):
            return False
        if not all(deep):
            raise RuntimeError(
                "kv-session-offload prefill-spill (PS2): MIXED extend batch "
                f"({sum(deep)} born-spilled-deep of {len(deep)} requests). "
                "out_cache_loc would span device slots and host sentinels at "
                "once; refusing the batch instead of corrupting the slot space."
            )
        if len(batch.reqs) != 1:
            raise RuntimeError(
                "kv-session-offload prefill-spill (PS2): born-spilled-deep "
                f"extend batch carries {len(batch.reqs)} requests; V1 admits "
                "exactly one per batch (one session, one host region)."
            )
        return True

    def spill_extend_alloc(self, batch: ScheduleBatch) -> torch.Tensor:
        """PS2 stage A -- ``alloc_for_extend`` replacement for a born-spilled
        DEEP prompt: allocate NO device KV slots at all.

        The stock path (``mem_cache/common.alloc_for_extend``) allocates real
        device slots for every new token BEFORE any write happens, so a prompt
        that does not transiently fit wedges in the allocator and the write
        retarget never gets a chance. Here the new positions
        ``[prefix_len, seq_len)`` get HOST SENTINELS straight away -- the same
        encoding ``spill_decode_alloc`` hands to tokens generated while
        spilled, just vectorised over a whole chunk.

        The request slot itself (``alloc_req_slots``) is still allocated by the
        caller path: it is cheap and the row is needed.

        LOCKSTEP (I1-I3): ``L``, ``boundary``, ``host_row_base`` and the region
        content advance as ONE step here. The row is written, the region is
        claimed, the backend slot is opened and the SpillSlot is registered in
        the same synchronous scheduler section, BEFORE the forward that fills
        the region runs; nothing is provisional and nothing is committed later
        (I3 -- there is no speculative start whose commit could be deferred).
        The region content for ``[boundary, L)`` is written exactly once by
        this prefill and is FROZEN afterwards, exactly like a decode spill's
        frozen region.

        RANK-UNIFORM (U8): the region is claimed HERE, synchronously, inside
        the replicated scheduler section -- never from a copy-stream callback
        and never gated on rank-local progress. Every rank pops the same region
        index in the same iteration."""
        assert self.prefill_spill_extend_ready(batch)
        req = batch.reqs[0]
        rpi = int(req.req_pool_idx)
        boundary = int(batch.prefix_lens[0])
        L = int(batch.seq_lens_cpu[0].item())
        n_new = L - boundary
        assert n_new == int(batch.extend_num_tokens), (
            "kv-session-offload prefill-spill (PS2): extend_num_tokens "
            f"{batch.extend_num_tokens} != seq_len-prefix {n_new}; the prompt "
            "is chunked, which needs PS3 (host-prefix extend read)."
        )
        # C26 BELT AND BRACES. Everything below builds a sentinel
        # ``out_cache_loc``, and only the DCP lane's ``_dcp_write_scatter``
        # diverts it before ``store_kvcache``. Reaching here on a plain-TP
        # backend means the admission gate was bypassed, and the failure mode
        # is a device-side assert that kills the instance -- so fail HERE,
        # legibly, in Python, where the message names the cause.
        #
        # Raising is correct rather than returning a soft refusal: the caller
        # has already committed to the PS2 path, and the alternative
        # (falling through to ``alloc_for_extend``) would ask for device slots
        # this request was admitted precisely because it cannot have. Every
        # rank raises on the same replicated boot config, so this is not a
        # rank-local branch around a collective (law 14).
        if self.mode == "plain":
            raise RuntimeError(
                "kv-session-offload prefill-spill (PS2): born-spilled prompt "
                f"rid={req.rid} reached spill_extend_alloc on a plain-TP "
                "backend, which has no EXTEND write hook to divert the host "
                "sentinels away from store_kvcache. The admission gate "
                "(prefill_spill_deep_gate backend_write_hook) should have "
                "declined this request. Register C26."
            )
        device = self.req_to_token_pool.req_to_token.device
        positions = torch.arange(boundary, L, dtype=torch.int64, device=device)
        own_idx = prefill_spill_owner_split(positions, self.S, self.lo, self.hi)
        n_own = int(own_idx.numel())
        if n_own > self.region_tokens:
            raise RuntimeError(
                "kv-session-offload prefill-spill (PS2): born-spilled prompt "
                f"rid={req.rid} needs {n_own} host rows > region "
                f"{self.region_tokens}; the admission gate should have "
                "rejected it."
            )
        if not self._free_regions:
            raise RuntimeError(
                "kv-session-offload prefill-spill (PS2): no free host region "
                f"for born-spilled prompt rid={req.rid}; the admission gate "
                "and the region book-keeping disagree."
            )
        region = self._free_regions.pop(0)
        region_base = region * self.region_tokens

        # Sentinel row for the new positions. The device-resident radix prefix
        # [0, boundary) keeps its REAL slot ids -- it is tree-locked and is
        # read normally (paged) by this very extend forward.
        residues = positions % self.S
        sent = make_sentinels(self.host_base, self.S, residues, start=boundary)
        assert int(sent[-1].item()) < (1 << 31) - self.S, (
            "kv-session-offload: sentinel overflow (int32 req_to_token)"
        )
        self.req_to_token_pool.req_to_token[rpi, boundary:L] = sent.to(torch.int32)

        # Open the backend slot BEFORE the forward: the write path reads
        # region_base from it, and the tick re-derives every host row from
        # (row, region_base, host_row_base) afterwards.
        self.backend._sess_open_slot(rpi, region_base)
        self.backend._sess_prefill_open(rpi, boundary, own_idx, region_base)

        req.kv_spill_state = "host"
        req.kv_spill_boundary = boundary
        slot = SpillSlot(
            req=req,
            region=region,
            spill_iter=self._iter_ct,
            wave=WaveBackController(self.block_size, self._hysteresis_steps),
            hysteresis=RestoreHysteresis(self._hysteresis_steps),
        )
        slot.born_spilled = True
        slot.adopted = False
        stamp_spill(slot, getattr(self.scheduler, "phase_flip_active_stack", None))
        self.spills[rpi] = slot
        if getattr(self, "_budget_armed", False):
            # #236: born-spilled episode -- write-once prefill phase; the
            # admission gate ran through prefill_spill_free_regions.
            self._budget_note_spill(slot, "prefill", n_new)
        # One verdict, one prefill: clear the admission flag so a request that
        # is later RETRACTED and re-prefilled goes through a fresh admission
        # (and does not silently claim a second host region here).
        req.born_spilled_deep = False

        self._log(
            "kv-session-offload PREFILL-SPILL (PS2, born-spilled deep): rid=%s "
            "L=%d boundary=%d host_tail=%d owned_tail=%d (rank %d) region=%d "
            "spills=%d/%d -- NO device KV slots allocated",
            req.rid,
            L,
            boundary,
            n_new,
            n_own,
            self.dcp_rank,
            region,
            len(self.spills),
            self.max_spills,
        )
        return sent.to(torch.int64)

    def adopt_born_spilled_prefills(self, running_batch):
        """PS2 handover: hand a just-prefilled born-spilled session over to the
        spill tick.

        Runs at the TOP of ``pre_schedule``, i.e. after the prefill batch has
        been merged into ``running_batch`` and before any restore / wave-back /
        tick decision looks at the region.

        U9 -- THE JOIN EDGE SITS HERE, NOT AT THE ADMISSION VERDICT. The
        per-chunk D2H of the freshly computed KV runs on the copy stream, so
        the host region is not readable until that copy retires. Waiting the
        event here means every rank waits at the SAME iteration and at the
        SAME point in its collective sequence (this is a stream wait, not a
        collective and not a condition): ranks may wait for different amounts
        of time -- their owned shares differ under uneven DCP -- but none of
        them skips or repeats a step because of it.

        I2: the session is removed from ``running_batch`` before the tick /
        wave-back machinery can touch it, so no device decode forward and no
        boundary-advancing wave-back is ever in flight against the prefill
        write."""
        if not getattr(self, "prefill_spill", False):
            return running_batch
        pending = [
            slot
            for slot in self.spills.values()
            if getattr(slot, "born_spilled", False)
            and not getattr(slot, "adopted", True)
        ]
        if not pending:
            return running_batch
        self.backend._sess_prefill_join()
        for slot in pending:
            slot.adopted = True
            self.backend._sess_prefill_close(int(slot.req.req_pool_idx))
            self._log(
                "kv-session-offload PREFILL-SPILL (PS2): rid=%s handed over to "
                "the spill tick (region %d, boundary %d) (rank %d)",
                slot.req.rid,
                slot.region,
                int(slot.req.kv_spill_boundary or 0),
                self.dcp_rank,
            )
        if running_batch is not None and not running_batch.is_empty():
            adopted_rpi = {int(s.req.req_pool_idx) for s in pending}
            keep = [
                i
                for i, r in enumerate(running_batch.reqs)
                if int(r.req_pool_idx) not in adopted_rpi
            ]
            if len(keep) != len(running_batch.reqs):
                running_batch.filter_batch(keep_indices=keep)
                running_batch.batch_is_full = False
        return running_batch

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
                if (
                    self._tick_cost_ms is not None and self._tick_cost_ms > _MEAS_EPS_MS
                )
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
                    self._tick_cost_ms
                    if self._tick_cost_ms is not None
                    else float("nan"),
                    len(self.spills),
                    tc.n_changes,
                )
            # Diagnostic-only time series (SGLANG_KVSO_TICK_TRACE=1): correlate
            # the emergent interval / measured tick_cost / binding headroom with
            # the CURRENT host-tail (offload) size, so "offload grows -> tick
            # cost grows -> interval widens" (or not) is directly observable.
            # Throttled; reads only existing state -> no control effect.
            if self._tick_trace and self.spills:
                self._tick_trace_iter += 1
                if self._tick_trace_iter % 8 == 0:
                    tc = self.tick_controller
                    headroom = (
                        tc._ratio_window[-1] if tc._ratio_window else float("nan")
                    )
                    max_tail = 0
                    tot_tail = 0
                    for _s in self.spills.values():
                        _r = _s.req
                        _L = len(_r.origin_input_ids) + len(_r.output_ids) - 1
                        _b = int(getattr(_r, "kv_spill_boundary", 0) or 0)
                        _t = max(0, _L - _b)
                        tot_tail += _t
                        if _t > max_tail:
                            max_tail = _t
                    self._log(
                        "kv-session-offload: tick-trace t=%.3f interval=%d "
                        "tick_cost=%.3fms headroom=%.4f host_tail=%d "
                        "host_tail_sum=%d spilled=%d tick_graph=%s",
                        time.time(),
                        tc._effective,
                        self._tick_cost_ms
                        if self._tick_cost_ms is not None
                        else float("nan"),
                        headroom,
                        max_tail,
                        tot_tail,
                        len(self.spills),
                        # Which segment class the tick_cost next to it came
                        # from. "unattributed" means no collective slot reached
                        # the reporter (no SplitDeviceTimer installed on this
                        # runner) -- it is not a synonym for "eager".
                        self._tick_graph_state,
                    )

        # #236: refresh the rank-uniform budget clock + rate bucket BEFORE any
        # spill/adopt event of this iteration reads them. Unconditional call
        # site when armed (replicated config) -> uniform collective count.
        if getattr(self, "_budget_armed", False):
            self._budget_begin_iteration()

        # 0. PS2 handover: a born-spilled prompt that prefilled straight into
        #    its host region last iteration leaves the running batch here and
        #    becomes a normal spilled session. Done FIRST so the reap / restore
        #    / wave-back steps below already see a consistent slot, and so the
        #    copy-stream D2H is joined before anything reads the region (U9).
        running_batch = self.adopt_born_spilled_prefills(running_batch)

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

        # 1b. #236: evaluate running episodes against the armed budgets --
        #     demote exhausted ones, release grace-expired demotions. After
        #     the reap (fresh slot set), before restore/tick decisions.
        #     Runs BEFORE the park flow so a seam-initiated park
        #     (park_instead_of_demote) is advanced in the same iteration.
        if getattr(self, "_budget_armed", False) and self.spills:
            self._budget_evaluate_episodes()

        # 1c. #224 destinations: advance the park/unpark state machine.
        #     BEFORE the empty-spills early return -- unparking must run
        #     even when no session is currently host-resident. Inert
        #     (single gated call, no other state touched) when the
        #     destination chain is unarmed.
        if getattr(self, "_dest", None) is not None:
            from sglang.srt.managers.kv_session_spill_destination import (
                maybe_park_flow,
            )

            running_batch = maybe_park_flow(self, running_batch, last_batch)

        if not self.spills:
            self._maybe_spill_for_fast_lane(running_batch)
            return running_batch

        # 2. Fast-lane admission may still evict more sessions (into further
        #    free regions) even while some are already spilled.
        self._maybe_spill_for_fast_lane(running_batch)

        # 3. Restore / wave-back each spilled session independently. A
        #    completed restore merges that session back into running_batch.
        for rpi, slot in list(self.spills.items()):
            if getattr(slot, "park_pending", False):
                # #224: on its way OUT to a park tier -- restoring it now
                # would race the snapshot. False unless the chain armed it.
                continue
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
        fr = min(fast_waiting, key=lambda r: getattr(r, "kv_arrival_seq", 0) or 0)
        ratio = getattr(getattr(sch, "new_token_ratio_tracker", None), "current", 1.0)
        max_new = getattr(fr.sampling_params, "max_new_tokens", 0) or 0
        need = len(fr.origin_input_ids) + int(max_new * ratio) + 1

        # #236 (b) pressure hysteresis -- the spill-side MIRROR of the restore
        # hysteresis: the fast-lane shortfall must HOLD for N consecutive
        # iterations before sessions are evicted for it, so freed-then-
        # reconsumed memory around the restore threshold does not flutter
        # into an immediate re-eviction. Applies to this elective path only
        # (decode-OOM spills must yield the same iteration; declining them
        # would just route the pressure into the harsher stock retraction).
        # The shortfall is compared against the rank-uniform admission budget
        # (min-reduced availability minus this rank's surplus) so the streak
        # counter advances identically on every rank. Gate off (default) ->
        # byte-identical.
        gate = getattr(self, "_fast_spill_pressure_gate", None)
        if gate is not None:
            have_u = (
                self.allocator.available_size()
                + self._tree_evictable_size()
                - self.dcp_budget_deficit()
            )
            if not gate.update(have_u < need):
                return

        spilled_any = 0
        while self._free_regions:
            have = self.allocator.available_size() + self._tree_evictable_size()
            if have >= need:
                break  # normal admission will take it now
            # Free the residual shortfall (block-rounded inside try_spill);
            # each victim is a partial tail spill into its own region.
            if not self.try_spill(running_batch, fast_pressure=True, need=need - have):
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
        # #656 LAYOUT GATE, before every other consideration. A host image is
        # specific to the phase it was captured in (PP: this stage's layers
        # for all tokens; TP: a token shard of every layer), so reading it
        # back in the other phase returns the wrong K/V -- silently, since
        # the shapes still line up. This gate covers BOTH ways back: the
        # incremental wave-back below and the committing restore, because
        # both are H2D copies into a device layout.
        #
        # It is deliberately separate from the tick pin
        # (kvso_flip_contract.pin_spills_to_phase), which stops the session
        # from RUNNING in the wrong phase. Two independent gates on the same
        # hazard means a bug in either one alone is still caught.
        if not restore_permitted(
            slot, getattr(self.scheduler, "phase_flip_active_stack", None)
        ):
            return running_batch

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
            # #552: BOUNDED. `defer()` resets the streak exactly as the bare
            # `reset()` did, and additionally counts. Only when the count
            # exceeds the limit does this fall through and let the restore be
            # considered -- otherwise an older spilled session is stranded for
            # as long as fast-lane traffic continues, which is an indefinite
            # hold rather than the tie-break "fast beats FCFS" describes.
            if not slot.hysteresis.defer():
                return running_batch
            logger.warning(
                "kv-session-offload: session %s restore forced through after "
                "%d consecutive fast-lane deferrals: 'fast beats FCFS' is a "
                "tie-break, not an indefinite hold. One fast request may pay a "
                "re-spill for this.",
                getattr(slot.req, "rid", "?"),
                slot.hysteresis.deferrals,
            )

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

        # Slots a RESTORE could actually obtain. Radix-tree-evictable tokens
        # count here, because _restore_memory_ok() evicts exactly that way
        # before it commits and re-checks the free list afterwards -- so this
        # test may be optimistic, the commit below is the authority.
        #
        # Gating on the free list alone deadlocks: the eviction that would
        # grow `avail` sits BEHIND this test. A finished co-resident session
        # does not return its KV to the allocator, it inserts it into the
        # radix tree (radix_cache.cache_finished_req), so `avail` stays ~0
        # while the tree holds thousands of evictable tokens. `fits_now` then
        # never turns true and the victim finishes on the host floor even
        # though the GPU is otherwise idle. The spill side already accounts
        # this way (_maybe_spill_for_fast_lane).
        #
        # Wave-back below deliberately keeps the un-augmented `avail`: it
        # allocates on the spot without evicting first.
        restorable = avail + self._tree_evictable_size()

        # RESTORE-READY: tail already fully drained, OR the whole tail fits now.
        drained = boundary >= L
        fits_now = restorable >= remaining + self.restore_margin_tokens
        if tick_trace_enabled() and self._iter_ct % 16 == 0:
            self._log(
                "kv-session-offload restore-gate: iter=%d L=%d boundary=%d "
                "remaining=%d avail=%d evictable=%d margin=%d drained=%s "
                "fits_now=%s quiescent=%s suppress_tick=%s",
                self._iter_ct,
                L,
                boundary,
                remaining,
                avail,
                restorable - avail,
                self.restore_margin_tokens,
                drained,
                fits_now,
                quiescent,
                slot.suppress_tick,
            )
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
        # P1 (S3): the wave-back space gate is a knob, not a constant. With the
        # default threshold 0 the call below resolves to the historical
        # `space_ok=avail > 0, remaining_cap=avail` on the LIVE LOCAL pool --
        # byte-identical. With a threshold set it compares the rank-uniform
        # min-reduced availability (already reduced once this iteration in
        # update_dcp_admission_state -- no new collective) so every DCP rank
        # waves in lock-step. dcp_min_avail() is a side-effect-free read of
        # that snapshot; see wave_back_gate for the full rationale.
        space_ok, wave_cap = wave_back_gate(
            avail, self.dcp_min_avail(), self.wave_back_min_free_tokens
        )
        advance = slot.wave.plan(
            boundary,
            L,
            space_ok=space_ok,
            copy_inflight=copy_inflight,
            remaining_cap=wave_cap,
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
            if getattr(slot, "park_pending", False):
                # #224: chosen for parking -- its tick stops so the region
                # content settles before the transfer snapshots it. False on
                # every slot unless the destination chain armed it.
                continue
            if (
                spill_tick_seq_len(
                    len(slot.req.origin_input_ids), len(slot.req.output_ids)
                )
                is None
            ):
                # BORN-SPILLED, prefill result not processed yet: no output
                # token exists, so there is nothing for a decode tick to feed
                # (see spill_tick_seq_len). Defer one iteration instead of
                # building a batch whose seq_len contradicts kv_committed_len.
                # A decode-spilled session always holds an output token, so
                # this never fires on that path.
                continue
            if slot.suppress_tick:
                # Restore-ready this iteration: skip its tick so it goes
                # quiescent (last_batch != tick) and finalizes next iteration.
                # One-shot -- clear the flag now (re-armed by _maybe_restore_flow
                # if still restore-ready). Rank-uniform: the flag is set from
                # replicated restore-readiness state on every rank.
                slot.suppress_tick = False
                continue
            if getattr(slot, "budget_demoted", False) and not getattr(
                slot, "budget_tick_release", False
            ):
                # #236 demoted, drain grace running: liveness is over -- no
                # further host decode; the session waits for its wave-back
                # drain and the radix handover. tick_release (grace expiry /
                # spec host-finish guard) re-allows the ONE finishing tick.
                continue
            due.append(slot)
        if not due:
            return None
        return min(due, key=lambda s: s.last_tick_iter)

    def maybe_take_tick(self, running_batch) -> Optional[ScheduleBatch]:
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
        if device_has_work and (self._iter_ct - self._last_tick_iter) <= interval:
            return None

        slot = self._pick_tick_slot(running_batch)
        if slot is None:
            return None

        # #236 rate regler: a tick streams the session's WHOLE host tail H2D,
        # the dominant PCIe consumer of the spill path. In debt -> defer the
        # tick (throttle, transient by construction -- the bucket refills),
        # never demote. Demoted sessions on their release tick are exempt
        # (they must finish and deliver). Coarse per-iteration gate: deferring
        # the picked slot defers the tick slot as a whole this iteration.
        # Replicated tail size + uniform bucket -> rank-uniform.
        if (
            getattr(self, "_budget_armed", False)
            and self._budget_bucket is not None
            and not getattr(slot, "budget_tick_release", False)
        ):
            if not self._budget_bucket.ready():
                self._budget_counters.rate_throttled_ticks += 1
                return None
            _r = slot.req
            _L = len(_r.origin_input_ids) + max(0, len(_r.output_ids) - 1)
            _tail = max(0, _L - int(getattr(_r, "kv_spill_boundary", 0) or 0))
            self._budget_bucket.consume(_tail)

        if slot.batch is None:
            slot.batch = self._build_spill_batch(slot.req)
            self._log(
                "kv-session-offload: first spill tick for rid=%s (L=%d spec=%s)",
                slot.req.rid,
                int(slot.batch.seq_lens_cpu[0].item()),
                not slot.batch.spec_algorithm.is_none(),
            )
        batch = slot.batch
        batch.filter_batch()
        if batch.is_empty():
            self._close_slot(slot.req.req_pool_idx, "finished on host")
            return None
        # C4 (spec-in-spill-tick, Option A): a spec spill tick allocates DEVICE
        # candidate slots (eagle_prepare_for_decode) and accepted tokens stay
        # device-resident (growing suffix), so under device pressure the pool
        # can be exhausted -> alloc_token_slots would OOM-crash. Gate the spec
        # tick on a RANK-UNIFORM MIN-reduced headroom: if the binding rank
        # cannot fit the per-decode candidate reserve, ALL ranks skip THIS tick
        # (the session waits for headroom -- correct spill behavior; it resumes
        # once co-resident sessions free device KV, or restores via wave-back).
        # Plain (non-spec) spill ticks use sentinel slots (no device alloc) and
        # are never gated. The MIN-reduce keeps the collective forward in
        # lock-step (a per-rank available_size gate would desync -> NCCL hang).
        if not batch.spec_algorithm.is_none():
            from sglang.srt.mem_cache.common import get_alloc_reserve_per_decode

            need = get_alloc_reserve_per_decode() + 8
            min_avail = self._min_reduce_avail(self.allocator.available_size())
            if min_avail < need:
                self._log(
                    "kv-session-offload spec-in-tick: device headroom %d < %d "
                    "(binding rank) rid=%s -> skip spec tick this iter",
                    min_avail,
                    need,
                    slot.req.rid,
                )
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

    def _draft_kv_snapshot_device(self, seg, cap):
        """C2 (spec-in-spill-tick, Option b'): clone the draft KV tail ``seg``
        into a DEVICE-resident stacked buffer ``[layer_num, cap, *row]`` per K
        and V, filled ``[0, len(seg))``. Kept on device (unlike the pinned-CPU
        ``_draft_kv_snapshot`` used by the resume path) so the spill-tick
        ``draft()`` attends it as a normal device decode while the TARGET KV
        host-streams. FULL tail (draft pool is not DCP-token-sharded, M4). Raw
        ``store_dtype``. Ordered after the in-flight forward by the caller's
        ``_wait_forward_stream()``. ``cap`` bounds the resident growth (the
        session appends accepted-token draft KV up to ``cap``)."""
        dp = self.draft_full_pool
        assert getattr(dp, "page_size", 1) == 1, (
            "kv-session-offload spec-in-tick: draft pool page_size must be 1 "
            f"(got {getattr(dp, 'page_size', None)})"
        )
        seg64 = seg.to(torch.int64)
        n = int(seg64.numel())
        pcap = int(dp.k_buffer[0].shape[0])
        if n:
            hi = int(seg64.max().item())
            lo = int(seg64.min().item())
            assert 0 <= lo and hi < pcap, (
                "kv-session-offload spec-in-tick device snapshot: slot id out of "
                f"draft pool bounds [0,{pcap}): min={lo} max={hi}"
            )
        assert n <= cap, (
            f"kv-session-offload spec-in-tick: draft tail {n} exceeds resident "
            f"buffer cap {cap} (should have been caught by mtp_resident_tail_fits)"
        )
        ln = int(dp.layer_num)
        kbuf = dp.k_buffer[0].new_empty((ln, cap) + tuple(dp.k_buffer[0].shape[1:]))
        vbuf = dp.v_buffer[0].new_empty((ln, cap) + tuple(dp.v_buffer[0].shape[1:]))
        for l in range(ln):
            if n:
                kbuf[l, :n] = dp.k_buffer[l][seg64]
                vbuf[l, :n] = dp.v_buffer[l][seg64]
        return kbuf, vbuf, n

    # === C3/d4 (spec-in-spill-tick): DRAFT req_to_token surgery ============
    # The spilled session's committed-prefix TAIL [boundary, L) carries HOST
    # SENTINELS in the shared req_to_token (the target KV lives on host). The
    # draft READ (draft attention over req_to_token[rpi, :L]) would attend
    # garbage at those sentinel slots. Around draft() we temporarily point that
    # tail at DEVICE draft-scratch slots loaded from the resident draft_dev_k/v
    # snapshot, run draft(), then restore the sentinels. The device HEAD
    # [0, boundary) keeps its real (never-freed, tree-locked) draft slots. The
    # draft WRITE target [L, L+num_steps] is the candidate region eagle_prepare_
    # for_decode allocated -- untouched by this surgery, so writes hit real
    # device draft slots (no pool corruption). Rank-uniform: every DCP rank
    # holds the full replicated draft context and runs the identical surgery.

    # Signal returned by spec_in_tick_draft_pre when a device draft CANNOT run
    # for this tick (no reserved scratch, or the tail overflowed the reserved
    # cap): the worker MUST fall back to the plain host tick -- running the
    # spec draft over the host SENTINELS would be a CUDA illegal access.
    DRAFT_SURGERY_FALLBACK = "kvso_spec_in_tick_fallback"

    def spec_in_tick_draft_pre(self, batch):
        """d4 PRE (call right before draft()): redirect the spilled session's
        committed-prefix tail req_to_token slots to the RESERVED device draft-
        scratch, loaded from draft_dev_k/v. Returns:
          * an opaque handle           -> surgery applied, run the device draft;
          * ``DRAFT_SURGERY_FALLBACK`` -> device draft CANNOT run (no scratch /
                                          tail over cap) -> caller plain-falls-
                                          back (drafting over the host sentinels
                                          would be an illegal memory access);
          * ``None``                   -> no surgery needed (empty tail)."""
        rpi = int(batch.req_pool_indices[0].item())
        slot = self.spills.get(rpi)
        if slot is None or not getattr(slot, "spec_in_tick", False):
            return None
        if slot.draft_dev_k is None:
            return None
        boundary = int(slot.draft_spill_boundary)
        L = int(slot.draft_spill_L)
        n = int(slot.draft_dev_len)
        if n <= 0:
            return None
        assert n == L - boundary, (
            f"kv-session-offload spec-in-tick draft surgery: draft_dev_len {n} "
            f"!= L-boundary {L - boundary} (rid={slot.req.rid})"
        )
        scratch = self._draft_read_scratch
        if scratch is None or n > int(scratch.numel()):
            # No reserved scratch, or the resident tail overflowed the reserved
            # cap -> the device draft cannot run this tick: plain-fallback.
            self._log(
                "kv-session-offload spec-in-tick: draft tail %d has no "
                "reserved scratch (cap=%s) rid=%s -> plain host tick this tick",
                n,
                None if scratch is None else int(scratch.numel()),
                slot.req.rid,
            )
            return self.DRAFT_SURGERY_FALLBACK
        scratch = scratch[:n]
        dp = self.draft_full_pool
        self._wait_forward_stream()
        ln = int(dp.layer_num)
        for l in range(ln):
            dp.k_buffer[l][scratch] = slot.draft_dev_k[l, :n]
            dp.v_buffer[l][scratch] = slot.draft_dev_v[l, :n]
        r2t = self.req_to_token_pool.req_to_token
        saved = r2t[rpi, boundary:L].clone()
        r2t[rpi, boundary:L] = scratch.to(r2t.dtype)
        return (rpi, boundary, L, saved)

    def spec_in_tick_seed(self, batch):
        """d2: build the per-tick EAGLE draft seed for a spec-in-tick spilled
        session from the LAST committed token's target hidden (the proven
        seed-only resume primitive -- topk_p=1, topk_index=[last_tok],
        hidden_states=last_hidden). Valid one-row seed; the draft's num_steps
        forwards then attend the surgery-provided device draft prefix KV.
        Returns None when the session is not spec-in-tick / has no seed yet."""
        rpi = int(batch.req_pool_indices[0].item())
        slot = self.spills.get(rpi)
        if slot is None or not getattr(slot, "spec_in_tick", False):
            return None
        if getattr(slot, "last_hidden", None) is None:
            return None
        L = int(batch.seq_lens_cpu[0].item())
        return self._seed_only_resume(slot, L)

    def spec_in_tick_bootstrap_seed(self, batch):
        """d2 BOOTSTRAP: the first spec-in-tick after a spill has no captured
        hidden yet, so it runs a TRIVIAL 1-node verify (no draft). Provide the
        minimal EagleDraftInput the trivial-verify builder reads -- just
        bonus_tokens = the last committed token (the 1-node root). Its verify
        forward captures the hidden that seeds the NEXT tick's real draft."""
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        device = self.scheduler.device
        req = batch.reqs[0]
        last_tok = int(req.output_ids[-1])
        return EagleDraftInput(
            bonus_tokens=torch.tensor([last_tok], dtype=torch.int64, device=device),
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

    def spec_in_tick_note_committed(self, batch, last_token_id, last_hidden):
        """d2/d6: after a spec-in-tick verify commits, refresh the seed source
        (last committed token id + its target hidden) so the NEXT tick re-seeds
        the draft chain. Cloned to survive pooled-buffer reuse."""
        rpi = int(batch.req_pool_indices[0].item())
        slot = self.spills.get(rpi)
        if slot is None:
            return
        slot.last_hidden = last_hidden.detach().clone()

    def spec_in_tick_draft_post(self, handle):
        """d4 POST (call right after draft(), before the accepted-KV append):
        restore the committed-prefix sentinels. The scratch is reserved (held
        for the manager's life), so nothing is freed here."""
        if handle is None or handle == self.DRAFT_SURGERY_FALLBACK:
            return
        rpi, boundary, L, saved = handle
        r2t = self.req_to_token_pool.req_to_token
        r2t[rpi, boundary:L] = saved

    def spec_in_tick_append_accepted(self, batch, accepted_slots):
        """d6 (call after verify commits): append the accepted candidate tokens'
        DRAFT KV (at draft-pool slots ``accepted_slots``) to the session's
        device-resident draft_dev_k/v so the NEXT tick's draft attends them, and
        advance the resident tail bounds. ``accepted_slots`` is the int64 draft-
        pool slot tensor for the accepted candidates (in commit order). Grows
        draft_dev_len / draft_spill_L by len(accepted_slots) (capped)."""
        rpi = int(batch.req_pool_indices[0].item())
        slot = self.spills.get(rpi)
        if slot is None or not getattr(slot, "spec_in_tick", False):
            return
        if slot.draft_dev_k is None or accepted_slots is None:
            return
        a = int(accepted_slots.numel())
        if a <= 0:
            return
        dp = self.draft_full_pool
        cap = int(slot.draft_dev_k.shape[1])
        n = int(slot.draft_dev_len)
        if n + a > cap:
            # Resident buffer full: fall back to plain host tick from now on
            # (graceful, no OOM). The overflow is logged once.
            self._log(
                "kv-session-offload spec-in-tick: draft resident buffer full "
                "(%d+%d>%d) rid=%s -> plain host tick from now",
                n,
                a,
                cap,
                slot.req.rid,
            )
            slot.spec_in_tick = False
            slot.batch = None  # force rebuild as plain (spec_algorithm=NONE)
            return
        sl = accepted_slots.to(torch.int64)
        ln = int(dp.layer_num)
        for l in range(ln):
            slot.draft_dev_k[l, n : n + a] = dp.k_buffer[l][sl]
            slot.draft_dev_v[l, n : n + a] = dp.v_buffer[l][sl]
        slot.draft_dev_len = n + a
        slot.draft_spill_L = int(slot.draft_spill_L) + a

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
                (self.cp_prefix[r], self.cp_prefix[r + 1]) for r in range(len(counts))
            ]
            class_slots = self.allocator.alloc_owner_matched_classes(
                self.S, bounds, counts
            )
            if class_slots is None:
                slot.hysteresis.reset()
                return running_batch
            new_locs = assign_owner_matched_slots(residues, self.cp_prefix, class_slots)
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
        self.req_to_token_pool.req_to_token[req.req_pool_idx, boundary:L] = new_locs.to(
            torch.int32
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
                (self.cp_prefix[r], self.cp_prefix[r + 1]) for r in range(len(counts))
            ]
            class_slots = self.allocator.alloc_owner_matched_classes(
                self.S, bounds, counts
            )
            if class_slots is None:
                return False  # not enough owner-matched room this window
            new_locs = assign_owner_matched_slots(residues, self.cp_prefix, class_slots)
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
        # #552: the session is no longer being held off, so the anti-starvation
        # count starts again from zero. Deliberately NOT done in
        # RestoreHysteresis.reset(): a fast-lane deferral calls that, and
        # clearing the count there would make the bound unreachable -- which is
        # exactly the bug the bound exists to fix.
        slot.hysteresis.clear_deferrals()
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
            "sentinel(s) after restore (rank %d)" % (req.rid, L, n_sent, self.dcp_rank)
        )
        if getattr(self, "_budget_armed", False):
            self._budget_counters.episodes_restored += 1
            if slot.budget_demoted:
                # #236 handover complete: the full row is device-resident;
                # the capped session finishes on its next device step and the
                # STOCK finish donates the whole prefix to the radix tree
                # (HiRadixCache then write-backs per its own policy). The
                # continuation is a prefix hit -- work kept, liveness gone.
                # The cap lands HERE (quiescent -- finalize requires the tick
                # result to be settled), never at the demote instant.
                self._budget_finish_cap(req, extra=0)
                self._budget_counters.demotions_drained += 1
            elif self._budget_cooldown is not None:
                # (a)/(c) cooldown arms for a LIVE restored session: not a
                # victim again until it produced progress_lock_tokens outputs
                # and the time cap passed.
                self._budget_cooldown.note_restore(
                    req.rid, len(req.output_ids), self._budget_now
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
                slot.req.rid,
                gap,
                nh,
                self.dcp_rank,
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
        batch.req_pool_indices_cpu = torch.tensor([req.req_pool_idx], dtype=torch.int64)
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

    def release_finished_spilled_req(self, req: Req):
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
        if getattr(self, "_budget_armed", False):
            self._budget_counters.episodes_finished_on_host += 1
            if slot is not None and getattr(slot, "budget_demoted", False):
                # #236 coarse fallback fired: the demoted session finished on
                # host before its drain completed. This path donates nothing
                # (head freed, host tail reclaimed with the region), so the
                # continuation pays a full re-prefill -- the pre-#236 failure
                # floor. Counted; demotions_drained is the counterpart.
                self._budget_counters.demotions_host_finished += 1
        boundary = int(getattr(req, "kv_spill_boundary", 0) or 0)
        protected = int(req.cache_protected_len or 0)
        head_freed = 0
        if boundary > protected:
            # Retained exclusive device head [protected, boundary).
            head = pool.req_to_token[rpi, protected:boundary]
            self.allocator.free(head.to(torch.int64))
            head_freed = boundary - protected
        # C4 (spec-in-spill-tick, Option A): a spec-in-tick session accumulated a
        # DEVICE SUFFIX [draft_spill_L, L) -- the accepted tokens whose target+
        # draft KV stayed device-resident at real committed slots (the FROZEN
        # host region is only [boundary, draft_spill_L)). The finish path above
        # only frees the head; free the suffix real slots here too or they leak
        # (the "pool memory leak" the invariant checker flags at cleanup).
        suffix_freed = 0
        if slot is not None and getattr(slot, "spec_in_tick", False):
            spill_L = int(getattr(slot, "draft_spill_L", 0) or 0)
            # Free the device SUFFIX [draft_spill_L, committed) AND the last
            # tick's candidate OVER-ALLOCATION [committed, allocated) -- both are
            # real device slots past the frozen host region; the normal spec
            # overhang-free (pop_overallocated_kv_cache) never ran for the
            # persistent spill batch, so free them here (masked to real slots so
            # a stray sentinel is never double-freed).
            hi = max(
                int(getattr(req, "kv_committed_len", 0) or 0),
                int(getattr(req, "kv_allocated_len", 0) or 0),
            )
            if hi > spill_L > 0:
                suffix = pool.req_to_token[rpi, spill_L:hi]
                real = suffix[suffix < self.host_base]
                if int(real.numel()) > 0:
                    self.allocator.free(real.to(torch.int64))
                    suffix_freed = int(real.numel())
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
        if getattr(self, "_budget_armed", False):
            # One stats line per episode end (rare): the counters the policy
            # is judged by, greppable without a dashboard.
            self._log(
                "kv-session-offload BUDGET stats (%s): %s", why, self.budget_stats()
            )
        # The backend's per-session head/tail split + owned-count cache belong
        # to THIS session; a later spill re-derives from its own sentinel row.
        self.backend._sess_close_slot(rpi)
        logger.debug(
            "kv-session-offload: spill slot closed (%s, rpi=%d, region=%d)",
            why,
            rpi,
            slot.region,
        )

    def live_offload_reqs(self):
        """Every LIVE session this manager owns outside the running batch:
        host-resident spills AND #224-parked ones.

        ``self.spills`` alone is not that set -- ``_commit_park`` pops a parked
        session out of it while the session stays alive and keeps its req-pool
        slot, its radix tree lock and its GDN/Mamba state slot. A consumer that
        enumerates through ``spills`` therefore sees a parked session as
        ABSENT, and "absent" is indistinguishable from "gone" unless the
        consumer knows to ask here. That mistake is not hypothetical: the #364
        GDN slot ladder treats an id it cannot find as a dead session and drops
        its exported state blob.

        Rank-uniform: park and spill decisions are replicated, so every rank
        enumerates the same sessions in the same iteration."""
        reqs = [slot.req for slot in self.spills.values() if slot.req is not None]
        if getattr(self, "_dest", None) is not None:
            from sglang.srt.managers.kv_session_spill_destination import parked_reqs

            reqs.extend(parked_reqs(self))
        return reqs

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
        if getattr(self, "_dest", None) is not None:
            # #224: parked sessions are in NO real batch while parked; the
            # abort scan must still reach them.
            from sglang.srt.managers.kv_session_spill_destination import (
                parked_inflight_entries,
            )

            out.extend(parked_inflight_entries(self))
        return out

    def park_instead_of_demote(self, slot) -> bool:
        """#224 <-> #236 SEAM. The spill-budget branch calls this at its
        demotion sites (immediately before ``_budget_demote``): when the
        destination chain is armed and the slot is park-eligible, the
        session is routed to the next tier instead of being demoted -- the
        budget exhaustion EXTENDS into remote capacity (work kept AND
        liveness kept, merely suspended) rather than capping generation.
        Returns False (demote as before) when the chain is unarmed, busy,
        or the slot is ineligible. A committed park moves the tail out of
        the local host volume; the budget branch discounts it at its
        _budget_note_* sites when this returned True."""
        if getattr(self, "_dest", None) is None:
            return False
        from sglang.srt.managers.kv_session_spill_destination import (
            park_instead_of_demote,
        )

        return park_instead_of_demote(self, slot)
