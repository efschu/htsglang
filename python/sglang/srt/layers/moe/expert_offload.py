# SPDX-License-Identifier: Apache-2.0
"""MoE expert-offload cache (feat/moe-expert-offload, M-B).

Keeps only a hot subset of a FusedMoE layer's local routed experts resident on
GPU; the full set lives in a pinned host-RAM pool. Before each MoE apply(), the
needed experts (from topk_ids) are resolved against the resident slots; misses
are async H2D-copied from the pinned pool into LRU-evicted slots, and topk_ids
are remapped to slot indices so the unmodified grouped-GEMM runs over the small
resident buffer.

Wave processing (fix for the prefill overflow crash)
----------------------------------------------------
A single forward can route to MORE unique experts than there are resident
slots (n_slots). This is the norm on prefill of a 256-expert / top-8 model:
even a short prompt touches nearly every expert. Rather than crash (the old
`_acquire_slot` evicted a still-needed expert -> KeyError) or silently serve
only the first n_slots, the forward is split into WAVES.

The split is over TOKENS, not over experts. Each token routes to at most
`top_k` (<= n_slots) unique experts, so every token's complete top-k set fits
in the resident buffer at once. We greedily pack consecutive token rows into a
wave until the union of their unique experts would exceed n_slots, then close
the wave. For each wave we (a) fetch its experts into resident slots, (b) remap
that wave's topk_ids -> slot ids, (c) run the unmodified grouped-GEMM over the
wave's token rows, and (d) scatter the per-row outputs back into the full
output buffer.

Byte-identity: a token's MoE output depends only on its own hidden state and
its own routed experts' weights -- it is independent of which other tokens
share the batch. Because every token is computed EXACTLY ONCE, with ALL of its
experts resident, and its top-k reduction runs in the original slot order, the
per-token result is bit-identical to the no-offload (fraction == 1.0) path.
There is no cross-wave accumulation of a single token's partial sums, so no
floating-point re-association is introduced.

Expert-major waves (SGLANG_MOE_OFFLOAD_WAVE_ORDER=expert, #254)
---------------------------------------------------------------
The token-major split above re-fetches a spill expert in EVERY wave whose
tokens route to it -- with C=16 scratch slots a 2048-token chunk runs ~62
waves, so each spill expert is streamed ~62 times (hundreds of GiB per chunk
and rank). The opt-in expert-major split inverts the axis: waves are disjoint
groups of at most C SPILL EXPERTS, and each group is fetched exactly once per
forward.

That breaks the "a wave holds a token's complete top-k" property the identity
argument above rests on, so the reduction is taken out of the wave: each routed
(token, k-slot) pair is submitted as its own pseudo-token with top_k == 1 (the
fused kernel then writes the weighted contribution straight out, with no
internal reduction) and stored at its own k-slot in a [T, top_k, H] buffer. The
k-slot is fixed by the routing, so the buffer holds the same values in the same
places for ANY wave split; the top-k reduction runs once at the end over the
full buffer, in k order, with the same reduction the unsplit kernel applies to
its own intermediate_cache3. The result is bit-identical to both the
token-major and the no-offload path (measured on bf16 and fp8-blockwise,
tests/moe_offload/test_wave_order_gpu.py). Cost: one transient [T, top_k, H]
buffer per layer. Decode (single wave) is unaffected; default stays token.

Design notes
------------
* Cold experts are FETCHED and computed on GPU (this rig's AMD CPU has no AMX,
  so ktransformers-style CPU compute via kt_ep_wrapper is not viable here).
* Default path is untouched: with SGLANG_MOE_RESIDENT_EXPERT_FRACTION == 1.0
  the layer never installs a cache and behaves byte-identically.
* The resolve/LRU/wave bookkeeping (`ExpertResidencyPlanner` + `plan_token_waves`)
  is pure Python and is unit-tested on CPU without CUDA
  (tests/moe_offload/test_planner.py); only `MoEExpertOffloadCache` touches
  tensors.
* CUDA-graph incompatible by nature: `prepare()`/`run_waves()` do a device->host
  sync (`topk_ids.tolist()`) plus data-dependent Python planning, which is
  illegal during graph capture. Offload therefore REQUIRES --disable-cuda-graph;
  the layer fails fast at construction otherwise (see layer.py).
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --- M-C routing trace ------------------------------------------------------
# Append-only JSONL sink consumed by moe_offload/sim.py. One handle per output
# file, shared by every FusedMoE layer in a process and serialized by a lock so
# concurrent layers never interleave a line. Distinct TP/EP ranks write to
# distinct files (rank tag in the name), so no cross-process contention exists.
_TRACE_HANDLES: Dict[str, "object"] = {}
_TRACE_LOCK = threading.Lock()

# The one layer whose per-chunk H2D volume is logged at INFO (see
# MoEExpertOffloadCache._log_wave_h2d). Latched to the first layer that reports.
_H2D_LOG_LAYER = None


def write_routing_trace(
    path: str,
    rank_tag: str,
    layer_id: int,
    step: int,
    experts_per_token: List[List[int]],
) -> None:
    """Append one JSONL record ``{"layer","step","experts"}`` for the offline
    hit-rate simulator (sim.py). ``experts_per_token`` is the per-token list of
    routed expert ids (``-1`` padding preserved; sim filters it). Measurement
    tooling only — reached exclusively when SGLANG_MOE_OFFLOAD_TRACE is set."""
    fname = f"{path}.{rank_tag}.jsonl"
    rec = json.dumps(
        {"layer": int(layer_id), "step": int(step), "experts": experts_per_token}
    )
    with _TRACE_LOCK:
        fh = _TRACE_HANDLES.get(fname)
        if fh is None:
            fh = open(fname, "a", buffering=1)
            _TRACE_HANDLES[fname] = fh
        fh.write(rec + "\n")


@dataclass
class ExpertOffloadRelease:
    """Per-rank tally of expert weight memory the offload took OFF the GPU.

    #119: the KV pool is sized from a live free-memory reading taken after the
    weights are resident, so the VRAM the offload releases flows into the KV
    budget on its own -- PROVIDED the install happens before the profiling and
    the freed blocks are actually back with the driver. That reclaim used to be
    an invisible side effect: nothing said how much was released, and a
    regression that moved the install back behind the sizing step (the #77
    "known limitation") would silently cost the whole win with no log line to
    notice it by. These counters make the reclaim an accounted, assertable
    quantity.

    ``device_bytes`` is the expert weight VRAM no longer held on the GPU;
    ``host_bytes`` is what the pinned spill pool took over in its place.
    """

    device_bytes: int = 0
    host_bytes: int = 0
    layers: int = 0
    tensors: int = 0


_RELEASE_TALLY = ExpertOffloadRelease()


def expert_offload_released_device_bytes(
    num_local_experts: int, buffer_slots: int, row_bytes: int
) -> int:
    """Pure: VRAM (bytes) one expert-major tensor stops holding under offload.

    The layer used to hold ``num_local_experts`` expert rows on GPU; after the
    split it holds ``buffer_slots`` (= R resident + C scratch). The difference
    is what the KV pool may claim. Returns 0 whenever the split keeps at least
    as many slots as there are experts (fully-resident = no offload), so the
    no-offload path tallies exactly nothing.
    """
    experts = int(num_local_experts)
    slots = max(0, int(buffer_slots))
    width = max(0, int(row_bytes))
    if experts <= 0 or slots >= experts:
        return 0
    return (experts - slots) * width


def record_expert_offload_release(
    device_bytes: int, host_bytes: int, tensors: int = 1, count_layer: bool = True
) -> None:
    """Tally one layer's release. Called once per layer that actually split.

    ``count_layer=False`` is for callers that tally a layer one TENSOR at a
    time (the #123-GGUF materialization-time staging stages w13 and w2 in
    separate calls) and must not report the layer twice.
    """
    _RELEASE_TALLY.device_bytes += max(0, int(device_bytes))
    _RELEASE_TALLY.host_bytes += max(0, int(host_bytes))
    _RELEASE_TALLY.tensors += max(0, int(tensors))
    if count_layer:
        _RELEASE_TALLY.layers += 1


def expert_offload_release_totals() -> ExpertOffloadRelease:
    """Snapshot of this rank's release tally (a copy; callers must not mutate)."""
    return ExpertOffloadRelease(
        device_bytes=_RELEASE_TALLY.device_bytes,
        host_bytes=_RELEASE_TALLY.host_bytes,
        layers=_RELEASE_TALLY.layers,
        tensors=_RELEASE_TALLY.tensors,
    )


def reset_expert_offload_release() -> None:
    """Clear the tally (tests; and a second model load in one process)."""
    _RELEASE_TALLY.device_bytes = 0
    _RELEASE_TALLY.host_bytes = 0
    _RELEASE_TALLY.layers = 0
    _RELEASE_TALLY.tensors = 0


@dataclass
class ResidencyStats:
    fetches: int = 0  # experts H2D-copied (misses that fit)
    hits: int = 0  # needed experts already resident
    misses: int = 0  # needed experts not resident
    evictions: int = 0  # resident experts kicked out
    forwards: int = 0  # resolve() calls (== number of waves run)
    overflow_forwards: int = 0  # forwards that needed >n_slots unique experts
    waves: int = 0  # total waves run across all forwards
    h2d_bytes: int = 0  # bytes streamed host->device by _fetch()
    # #394 slice 2: the subset of the above that came out of a PEER rank's
    # shared cold-tier segment rather than this rank's own pinned pool. Kept
    # separate because it is the direct measure of whether the shared tier is
    # being used at all -- a proportional arm whose remote counters stay zero
    # is an arm that silently ran the baseline.
    remote_fetches: int = 0
    remote_h2d_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 1.0


def plan_token_waves(
    experts_per_token: Sequence[Sequence[int]],
    resident_count: int,
    scratch: int,
    resident_ids: Optional[frozenset] = None,
) -> List[List[int]]:
    """Greedily partition token indices into waves whose union of unique SPILL
    experts is <= ``scratch``.

    Fixed-resident + scratch model: the resident experts are always resident on
    GPU (fixed slots, never fetched), so they impose NO wave budget. Only the
    SPILL experts consume the ``scratch`` slots and must be fetched, so a wave
    may include any number of resident experts plus at most ``scratch`` unique
    spill experts.

    Residency set: when ``resident_ids`` is None (default) the resident set is
    the static ``[0, resident_count)`` (spill == global id >= resident_count).
    When ``resident_ids`` is given (Stage-1 hot residency), the resident set is
    exactly that frozen id set (spill == id not in resident_ids); its size still
    equals ``resident_count`` so the scratch budget is unchanged. The wave split
    is over TOKENS either way, so every token is still computed exactly once with
    all its experts resident -> byte-identical regardless of which set is chosen.

    Pure-python, CPU-testable. ``experts_per_token[t]`` is the list of routed
    expert ids for token ``t`` (``-1`` padding allowed and ignored). Returns a
    list of waves; each wave is a list of token indices in original order.

    Raises ``ValueError`` if a single token needs more than ``scratch`` unique
    spill experts -- offload cannot serve even one token; fail fast.
    """
    if scratch < 1:
        raise ValueError("scratch must be >= 1")

    def _is_spill(e: int) -> bool:
        return (
            e not in resident_ids if resident_ids is not None else e >= resident_count
        )

    waves: List[List[int]] = []
    cur_rows: List[int] = []
    cur_spill: set = set()
    for t, experts in enumerate(experts_per_token):
        spill = {int(e) for e in experts if e is not None and _is_spill(int(e))}
        if len(spill) > scratch:
            raise ValueError(
                f"token {t} routes to {len(spill)} spill experts but only "
                f"scratch={scratch} scratch slots are available (a single "
                f"token's spilled top-k must fit in the scratch region; raise "
                f"the scratch size or the resident fraction)."
            )
        if cur_rows and len(cur_spill | spill) > scratch:
            waves.append(cur_rows)
            cur_rows = []
            cur_spill = set()
        cur_spill |= spill
        cur_rows.append(t)
    if cur_rows:
        waves.append(cur_rows)
    return waves


def plan_expert_waves(
    experts_per_token: Sequence[Sequence[int]],
    resident_count: int,
    scratch: int,
    resident_ids: Optional[frozenset] = None,
) -> Tuple[List[int], List[List[int]]]:
    """Partition the forward's routed SPILL experts into waves of <= ``scratch``.

    The expert-major counterpart of ``plan_token_waves``. Returns
    ``(resident_used, spill_waves)``:

    * ``resident_used`` -- the resident experts this forward routes to, sorted.
      They are already on GPU, need no scratch slot and no fetch, so they are
      computed in ONE wave of their own however many there are.
    * ``spill_waves`` -- the routed spill experts, sorted and chunked into
      groups of at most ``scratch``. Each group is fetched ONCE; a spill expert
      therefore crosses PCIe exactly once per forward instead of once per
      token-major wave.

    Deterministic (sorted ids, fixed chunking), pure-python, CPU-testable.
    Unlike the token-major split this cannot fail: a token's top-k may be
    spread across waves, so no per-token scratch bound exists.
    """
    if scratch < 1:
        raise ValueError("scratch must be >= 1")

    def _is_spill(e: int) -> bool:
        return (
            e not in resident_ids if resident_ids is not None else e >= resident_count
        )

    resident_used: set = set()
    spill_used: set = set()
    for experts in experts_per_token:
        for e in experts:
            if e is None:
                continue
            e = int(e)
            if e < 0:
                continue
            (spill_used if _is_spill(e) else resident_used).add(e)

    spill_sorted = sorted(spill_used)
    spill_waves = [
        spill_sorted[i : i + scratch] for i in range(0, len(spill_sorted), scratch)
    ]
    return sorted(resident_used), spill_waves


def resolve_wave_order(value: Optional[str]) -> str:
    """Normalize SGLANG_MOE_OFFLOAD_WAVE_ORDER; reject anything else loudly."""
    order = (value or "token").strip().lower()
    if order not in ("token", "expert"):
        raise RuntimeError(
            f"SGLANG_MOE_OFFLOAD_WAVE_ORDER must be 'token' or 'expert', got {value!r}"
        )
    return order


def combine_topk_partials(partials, out, routed_scaling_factor):  # pragma: no cover
    """Reduce a [T, top_k, H] per-(token, k-slot) contribution stack to [T, H].

    This is the SAME reduction ``_fused_moe_kernel_sequence`` applies to its own
    ``intermediate_cache3`` (see moe_runner/triton_utils/fused_moe.py, the
    combine block after the second kernel): the branch selection depends only on
    ``top_k``, the token count and ``routed_scaling_factor`` -- all of which the
    expert-major path keeps at their unsplit, full-batch values. Feeding it the
    same values in the same k order therefore reproduces the unsplit output bit
    for bit.
    """
    import torch

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _use_moe_sum_reduce_torch_compile,
        moe_sum_reduce,
        moe_sum_reduce_torch_compile,
    )

    topk = partials.shape[1]
    rsf = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    if topk == 2 and rsf == 1.0:
        torch.add(partials[:, 0], partials[:, 1], out=out)
        return out
    if _use_moe_sum_reduce_torch_compile(partials.shape[0]):
        moe_sum_reduce_torch_compile(partials, out, rsf)
    else:
        moe_sum_reduce(partials, out, rsf)
    return out


@dataclass
class ExpertResidencyPlanner:
    """Pure-python FIXED-RESIDENT + SCRATCH residency for one MoE layer.

    This is the host-capping, deterministic residency (Variant-C B2b):
      * experts ``[0, resident_count)`` are ALWAYS resident on GPU at slot==id
        (never fetched, never evicted). The host pinned pool therefore only
        needs the SPILL experts ``[resident_count, num_local_experts)`` -> the
        host footprint is ~spill, not the full expert set.
      * a wave's SPILL experts (id >= resident_count), taken in SORTED order,
        are fetched into the scratch region
        ``[resident_count, resident_count + scratch)``.
      * the GPU buffer size (``resident_count + scratch``) is FIXED, and the
        per-wave layout is a pure function of the wave's needed set (fixed
        resident slots + sorted scratch), so the marlin moe_align tiling is
        deterministic -> greedy output is self-deterministic at temp=0 (no
        cross-request drift). Resident experts are reused across waves without
        re-fetching (throughput win vs the earlier refetch-all scheme).

    A single ``resolve()`` must contain <= ``scratch`` unique spill experts
    (guaranteed by ``plan_token_waves``).
    """

    num_local_experts: int
    resident_count: int
    scratch: int
    stats: ResidencyStats = field(default_factory=ResidencyStats)
    # Stage-1 hot residency: when set, the resident set is exactly ``resident_ids``
    # (a frozen set of size resident_count) and ``resident_slot`` maps each
    # resident expert id -> its GPU slot in [0, resident_count). When None
    # (default) the resident set is the static [0, resident_count) at slot==id.
    resident_ids: Optional[frozenset] = None
    resident_slot: Optional[Dict[int, int]] = None
    # #394 link-proportional cold shard: cold experts a PEER rank's host tier
    # owns. This rank has no spill-pool row for them, so routing one here is a
    # caller bug (a layer that delegated without remapping foreign expert ids
    # away). None on every path without a host-shard ratio -> one `is not None`
    # per resolve on the default path.
    delegated_ids: Optional[frozenset] = None
    # #394 slice 2: True once a shared cold tier makes those ids REACHABLE --
    # the bytes live in a peer's segment and this rank can DMA the row. The
    # planner then treats a delegated expert exactly like any other spill
    # expert (scratch slot, fetch entry) and only the fetch SOURCE differs;
    # ``MoEExpertOffloadCache._fetch`` is the one place that knows which. False
    # keeps the slice-1 behaviour, which is a named refusal, because without a
    # shared tier a delegated expert really is absent.
    delegated_reachable: bool = False

    def __post_init__(self):
        if self.scratch < 1:
            raise ValueError("scratch must be >= 1")
        if self.resident_count < 0:
            raise ValueError("resident_count must be >= 0")
        if self.resident_count > self.num_local_experts:
            self.resident_count = self.num_local_experts

    @property
    def buffer_size(self) -> int:
        """GPU buffer slot count = fixed resident + scratch (capped at E)."""
        return min(self.resident_count + self.scratch, self.num_local_experts)

    @property
    def fully_resident(self) -> bool:
        return self.resident_count >= self.num_local_experts

    def resolve(
        self, needed: Sequence[int]
    ) -> Tuple[Dict[int, int], List[Tuple[int, int]]]:
        """Return (slot_of_needed, fetch_plan) for one wave.

        slot_of_needed: expert_id -> slot for every needed expert.
        fetch_plan: list of (spill_expert_id, scratch_slot) to H2D-copy.
        Resident experts (id < resident_count) map to slot==id and are NOT
        fetched (already resident). Spill experts (id >= resident_count), sorted,
        map to scratch slots [resident_count + i] and are fetched. The layout is
        a pure function of ``needed`` (history-independent) -> deterministic.
        """
        self.stats.forwards += 1
        self.stats.waves += 1
        needed_unique = sorted(set(int(e) for e in needed if e >= 0))
        if self.fully_resident:
            self.stats.hits += len(needed_unique)
            return {e: e for e in needed_unique}, []

        if self.resident_ids is None:
            # Static residency: resident == [0, R) at slot==id.
            resident = [e for e in needed_unique if e < self.resident_count]
            spill = [e for e in needed_unique if e >= self.resident_count]  # sorted
            resident_slot_of = {e: e for e in resident}
        else:
            # Hot residency: resident == frozen id set at its assigned slot.
            resident = [e for e in needed_unique if e in self.resident_ids]
            spill = [e for e in needed_unique if e not in self.resident_ids]  # sorted
            resident_slot_of = {e: self.resident_slot[e] for e in resident}
        if self.delegated_ids is not None:
            foreign = [e for e in spill if e in self.delegated_ids]
            if foreign and not self.delegated_reachable:
                raise RuntimeError(
                    f"experts {foreign} were delegated to a peer rank's host "
                    f"tier by the #394 link-proportional cold shard, but this "
                    f"rank's router asked for them and no shared cold tier is "
                    f"attached, so they are absent rather than relocated. "
                    f"Either enable the shared tier "
                    f"(SGLANG_MOE_COLD_TIER_SHM=1, #394 slice 2) or remap the "
                    f"foreign expert id away (the #82 dim-0 shard's padding "
                    f"expert) before delegating any cold expert."
                )
            if foreign:
                # Reachable: the row comes from a peer's segment instead of
                # this rank's pool. Counted here rather than in _fetch so the
                # tally survives the desk path, which has no CUDA stream.
                self.stats.remote_fetches += len(foreign)
        if len(spill) > self.scratch:
            raise RuntimeError(
                f"resolve() got {len(spill)} spill experts but only "
                f"{self.scratch} scratch slots exist; caller must wave-split "
                f"with plan_token_waves() first."
            )
        self.stats.hits += len(resident)
        self.stats.misses += len(spill)
        self.stats.fetches += len(spill)

        slot_of_needed: Dict[int, int] = dict(resident_slot_of)
        fetch_plan: List[Tuple[int, int]] = []
        for i, e in enumerate(spill):
            slot = self.resident_count + i
            slot_of_needed[e] = slot
            fetch_plan.append((e, slot))
        return slot_of_needed, fetch_plan


def resident_slot_count(num_local_experts: int, fraction: float) -> int:
    """Resident-expert count to keep on GPU for a given fraction (<1)."""
    n = int(math.ceil(fraction * num_local_experts))
    return max(1, min(num_local_experts, n))


def scratch_slot_count(resident_count: int) -> int:
    """Scratch slots C for the fixed-resident buffer (env-overridable).

    The GPU buffer is (resident_count + C) slots; C bounds the unique SPILL
    experts a single wave may fetch. Default C = max(8, resident_count // 4):
    big enough to hold a decode step's spilled top-k, small enough to keep the
    GPU buffer modest (buffer/E fraction determines resident-VRAM). Override via
    SGLANG_MOE_SCRATCH_SLOTS.
    """
    import os

    env = os.environ.get("SGLANG_MOE_SCRATCH_SLOTS", "")
    if env.strip():
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(8, resident_count // 4)


# ===========================================================================
# #394: LINK-PROPORTIONAL COLD-EXPERT SHARDING
#
# A cold expert is paid for in PCIe seconds, and a fetch wave is over only when
# the LAST rank's share has landed. Splitting the cold pool EQUALLY across TP
# ranks whose links are not equal therefore lets the narrowest link set the
# clock for all of them. On this rig the links are gen4 x4 / x8 / x8 -- 6.4 /
# 13 / 13 GB/s measured H2D out of pinned host memory -- so the x4 rank moves
# the same bytes over half the link and every other rank waits for it.
#
# ANALYSE_393 §7.3/§7.4 puts numbers on it under its parameterised 2.79
# GB/token model: 0.93 GB over 6.4 GB/s = 145 ms/token with equal shards,
# against 2.79 GB over the 32.4 GB/s the three links absorb together = 86
# ms/token with proportional shares. That is a 1.69x ceiling on the cold tier,
# and 84% of the total headroom a full host-side compute lane could reach.
#
# DIRECTION -- the deliberate inverse of the standing "slowest rank is the
# metronome" rule. That rule governs CAPACITY splits: give a card work in
# proportion to its capacity and the weakest card carries the largest RELATIVE
# load, so it still sets the clock. What is being split here is not work a card
# must have room for, it is BYTES THAT MUST CROSS A LINK, and the weak
# participant is the LINK, not the card. So the weak link is handed FEWER cold
# experts, not more, and the shares are sized so all links finish their share
# at the same instant -- which is the only instant a wave cares about. The two
# orderings genuinely disagree on this box: the 20 GB 3080 sits in the x4 slot,
# so a share sized by VRAM is the worst possible share to send down that link.
#
# WHAT DOES NOT MOVE -- device residency. The resident tier is sized by the
# per-card VRAM budget and is untouched: R, the [R+C] buffer and every #400
# ledger figure derived from them are the same numbers with and without a
# ratio. Only HOST-side ownership of the cold pool moves, and only on dim 0 in
# whole experts, because a GGUF expert row is a run of opaque quantization
# blocks (#82/#109) and the expert axis is the one axis with no block structure
# on it.
#
# PRECONDITION for a caller -- delegating a cold expert to a peer is only legal
# on a layer that shards experts on dim 0 and remaps a foreign expert id away
# from this rank (the #82 GGUF expert-dim shard with its zero padding expert).
# On an intermediate-dim TP MoE every rank holds an essential slice of EVERY
# expert and nothing can be delegated; such a layer must not construct a
# ColdShardContext. A delegated id that reaches this rank's router anyway is
# caught by name in ExpertResidencyPlanner.resolve rather than surfacing as a
# missing spill-pool row.
# ===========================================================================

#: Env override for the per-rank host-shard ratio. Comma-separated positive
#: floats, one per TP rank, e.g. ``6.4,13,13`` for this rig's measured H2D
#: bandwidths. Read through ``os.environ`` (same as SGLANG_MOE_SCRATCH_SLOTS)
#: rather than ``environ.py`` so the policy stays inside this module.
HOST_SHARD_RATIO_ENV = "SGLANG_MOE_HOST_SHARD_RATIO"

#: Lowest provenance this rank will WEIGHT a split on: ``measured`` or
#: ``estimate``. ``absent`` is not accepted in either setting -- a split has to
#: come from a number, and "nobody measured this link" is not a number. The
#: default admits the nameplate derivation below, whose ratios land within 2 %
#: of the measured ones on this rig; a run that must not be weighted by a
#: datasheet at all sets ``measured`` and gets an equal split until the probe
#: has run.
HOST_SHARD_MIN_PROVENANCE_ENV = "SGLANG_MOE_HOST_SHARD_MIN_PROVENANCE"

#: Ratio sources, strongest first. The label is not decoration: it is what
#: decides whether the number may weight a split at all, and it is written into
#: the log line and the #390 dump so an A/B arm names its own policy.
HOST_SHARD_SOURCE_ENV = "env"
HOST_SHARD_SOURCE_PROBE = "card-probe-h2d"
HOST_SHARD_SOURCE_NVML = "nvml-pcie"
HOST_SHARD_SOURCE_EQUAL = "equal"

#: source -> provenance, in the #348b/#407 vocabulary
#: (:class:`sglang.srt.planner.cost_model.Provenance`). ``env`` counts as
#: MEASURED because the vector an operator types is the measurement they took;
#: the nameplate derivation is an ESTIMATE by construction (a formula over a
#: measured link width, not a transfer anybody timed); ``equal`` is ABSENT --
#: it is the shape a refusal takes, not a ratio.
_HOST_SHARD_PROVENANCE = {
    HOST_SHARD_SOURCE_ENV: "measured",
    HOST_SHARD_SOURCE_PROBE: "measured",
    HOST_SHARD_SOURCE_NVML: "estimate",
    HOST_SHARD_SOURCE_EQUAL: "absent",
}

_PROVENANCE_RANK = {"measured": 0, "estimate": 1, "absent": 2}

#: Encoding-adjusted per-lane throughput of one PCIe generation, GB/s, one
#: direction. Only the RATIOS between ranks are used, so these nominal figures
#: are enough; the measured numbers (6.4 vs 13 GB/s, i.e. 1.00 : 2.03 against
#: this table's 1 : 2) come from the card probe, which outranks this derivation
#: precisely because a measurement beats a nameplate.
_PCIE_LANE_GBPS = {1: 0.250, 2: 0.500, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}

#: Latch so the chosen ratio is logged once per process, not once per layer.
#: Read and set under _HOST_SHARD_LOG_LOCK: the plan that reaches it is built
#: inside the loader's thread pool (#391), and an unguarded check-then-set latch
#: is exactly the pattern that let two threads through at once.
_HOST_SHARD_LOGGED = False
_HOST_SHARD_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class HostShardRatio:
    """Per-rank host->device bandwidth weights, plus where they came from.

    ``weights`` is normalized to sum 1.0 and is the same tuple on every rank
    (the partition is a pure function of it, so the ranks agree without
    talking). ``source`` is one of ``"env"``, ``"nvml-pcie"`` or ``"equal"``,
    and ``detail`` carries the provenance in a form fit for a log line.
    """

    weights: Tuple[float, ...]
    source: str
    detail: str = ""

    @property
    def provenance(self) -> str:
        """``measured`` / ``estimate`` / ``absent``, derived from the source.

        Derived rather than stored so the two can never be set to disagree:
        the source IS the provenance claim, and a second field would let a
        caller construct a nameplate ratio labelled as a measurement.
        """
        return _HOST_SHARD_PROVENANCE.get(self.source, "absent")

    @property
    def is_equal(self) -> bool:
        """True when the ratio carries no information a split could use.

        This is the default-unchanged predicate: an equal ratio must produce
        exactly today's assignment, so callers test it rather than comparing
        floats themselves.
        """
        if not self.weights:
            return True
        hi, lo = max(self.weights), min(self.weights)
        return (hi - lo) <= 1e-9 * hi

    def describe(self) -> str:
        shares = ", ".join(f"rank{i}={w:.4f}" for i, w in enumerate(self.weights))
        tail = f" ({self.detail})" if self.detail else ""
        return f"source={self.source} provenance={self.provenance} [{shares}]{tail}"


def _normalize_weights(values: Sequence[float]) -> Tuple[float, ...]:
    """Positive floats -> weights summing to 1.0. Raises on anything else."""
    out = [float(v) for v in values]
    for i, v in enumerate(out):
        if not (v > 0.0) or v != v or v == float("inf"):
            raise ValueError(
                f"host-shard weight for rank {i} is {v!r}; every weight must be "
                "a finite positive number"
            )
    total = sum(out)
    return tuple(v / total for v in out)


def equal_host_shard_ratio(world_size: int, detail: str = "") -> HostShardRatio:
    """The safe default: nothing is known about the links, so split equally."""
    n = max(1, int(world_size))
    return HostShardRatio(tuple(1.0 / n for _ in range(n)), "equal", detail)


def _host_shard_ratio_from_env(world_size: int) -> Optional[HostShardRatio]:
    """``SGLANG_MOE_HOST_SHARD_RATIO`` or ``None`` when it is unset.

    A malformed or wrong-length vector is a hard error, never a silent fall
    back to the derivation below: an operator who typed a ratio meant it, and
    quietly running a different split than the one they asked for is how a
    measurement arm turns into a lie. (Same contract as
    ``SGLANG_SP_CAPACITY_WEIGHTS`` on the diffusion lane.)
    """
    import os

    raw = os.environ.get(HOST_SHARD_RATIO_ENV, "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"{HOST_SHARD_RATIO_ENV}={raw!r} is not a comma-separated list of "
            f"positive floats ({exc})"
        ) from exc
    if len(values) != int(world_size):
        raise ValueError(
            f"{HOST_SHARD_RATIO_ENV}={raw!r} has {len(values)} entries but the "
            f"MoE group has {int(world_size)} ranks; give exactly one weight "
            "per rank"
        )
    return HostShardRatio(
        _normalize_weights(values),
        HOST_SHARD_SOURCE_ENV,
        f"{HOST_SHARD_RATIO_ENV}={raw}",
    )


def _min_provenance() -> str:
    """The weakest provenance this process will weight a split on."""
    import os

    raw = os.environ.get(HOST_SHARD_MIN_PROVENANCE_ENV, "").strip().lower()
    if not raw:
        return "estimate"
    if raw not in ("measured", "estimate"):
        raise ValueError(
            f"{HOST_SHARD_MIN_PROVENANCE_ENV}={raw!r} must be 'measured' or "
            "'estimate'. 'absent' is not selectable: a split weighted by a "
            "number nobody has is not a split, it is a guess."
        )
    return raw


def _provenance_admitted(provenance: str, minimum: str) -> bool:
    """True when a ratio of this provenance may weight the split."""
    return _PROVENANCE_RANK.get(provenance, 2) <= _PROVENANCE_RANK[minimum]


#: Process-wide memo of the card probe, so 40+ MoE layers do not each read and
#: parse the same JSON. ``False`` distinguishes "looked, found nothing" from
#: "have not looked yet"; both are reached from the loader's thread pool
#: (#391), hence the lock.
_CARD_PROBE_MEMO = None
_CARD_PROBE_LOCK = threading.Lock()


def _card_probe_h2d_table():
    """``{uuid: measured pinned H2D GB/s}`` from the rigmon card probe, or ``{}``.

    The probe (``rigmon/card_probe.py``, #271) times a 64 MiB pinned host->device
    copy per card, best-of wall clock, and caches the result under a path keyed
    on the sorted card UUIDs AND the driver version. Reading it here is a pure
    lookup: ``load_card_probe`` never triggers a measurement, so a weight load
    can never turn into a multi-second GPU probe as a side effect.

    THE PATH IS BUILT FROM NVML'S CARD SET, NOT THE PROCESS'S CUDA VIEW, and
    that is the whole reason this helper exists instead of a bare
    ``load_card_probe()``. The cache key is a digest over the SORTED UUIDS of
    the cards the caller can see, and a worker's ``CUDA_VISIBLE_DEVICES`` is
    narrowed to one GPU (``SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS``, forced by
    ``--rank-gpu-id``). A worker therefore computes a one-card digest, misses
    the three-card profile the probe actually wrote, and silently falls through
    to the nameplate ESTIMATE. Measured on the reference rig 2026-08-02: the
    same call returns the profile with all cards visible and ``None`` under
    ``CUDA_VISIBLE_DEVICES=1``. NVML is not masked by that variable, so the full
    card set is still available here and the digest can be reconstructed.

    This is the SAME artifact the planner and the dashboard price cards from
    (``planner.cost_model.memory_rates_from_entries``, kind ``h2d``), which is
    the point -- a second H2D opinion measured by a second kernel would be
    indistinguishable from this one after the fact.
    """
    global _CARD_PROBE_MEMO

    with _CARD_PROBE_LOCK:
        if _CARD_PROBE_MEMO is not None:
            return _CARD_PROBE_MEMO or {}
        table = {}
        try:
            from sglang.srt.rigmon.card_probe import load_card_probe

            profile = load_card_probe(_nvml_card_probe_path())
            if profile is None:
                # Last resort: the process's own view. Correct whenever the
                # caller can see every card (the launcher, the dashboard, a
                # desk test), and no worse than nothing when it cannot.
                profile = load_card_probe()
            if profile is not None:
                for card in profile.cards:
                    if card.h2d_gbs and float(card.h2d_gbs) > 0.0:
                        table[card.uuid] = float(card.h2d_gbs)
        except Exception:  # noqa: BLE001 - an unreadable probe is an absence
            table = {}
        _CARD_PROBE_MEMO = table or False
        return table


def _nvml_card_probe_path():
    """The probe cache path keyed on EVERY physical card, or ``None``.

    NVML enumerates the whole rig regardless of ``CUDA_VISIBLE_DEVICES``, so
    this reconstructs the key the probe was written under even inside a worker
    that can only see its own GPU. The driver version must come from the same
    place the probe took it, or the digest differs for that reason instead.
    """
    try:
        from sglang.srt.registry.nvml import list_devices, nvml_session
        from sglang.srt.rigmon.card_probe import card_probe_cache_path

        uuids = [d.uuid for d in list_devices()]
        if not uuids:
            return None
        with nvml_session() as pynvml:
            driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        return card_probe_cache_path(uuids, driver)
    except Exception:  # noqa: BLE001 - no NVML here is simply no reconstruction
        return None


def reset_card_probe_memo() -> None:
    """Test hook: re-read the card probe on the next resolution."""
    global _CARD_PROBE_MEMO

    with _CARD_PROBE_LOCK:
        _CARD_PROBE_MEMO = None


def _measured_h2d_gbps_by_uuid(uuid: str):
    """Measured pinned H2D GB/s for this card, or ``None`` if unmeasured."""
    return _card_probe_h2d_table().get(uuid)


def _pcie_link_gbps_by_uuid(uuid: str) -> Optional[float]:
    """PCIe bandwidth of the SLOT this card sits in, GB/s, or ``None``.

    The link itself comes from ``registry.nvml.pcie_link_for_uuid``, which is
    the ONE authority for the question (#736) and carries the canon in full:
    WIDTH FROM THE CURRENT LINK, GENERATION FROM THE MAXIMUM, resolved through
    the #331 IdentityMap by UUID and never positionally (#392). That rule used
    to be spelled out here and again in the #732 per-peer transport binding;
    two copies of a subtle rule is one too many, so it moved DOWN to the
    registry and both consumers import it. Change it there, not here.

    What stays here is the only part specific to this consumer: turning lanes
    and generation into GB/s through ``_PCIE_LANE_GBPS``. That remains an
    ESTIMATE -- lanes x an encoding constant, not a transfer anybody timed. The
    measured card probe outranks it.
    """
    from sglang.srt.registry.nvml import pcie_link_for_uuid

    link = pcie_link_for_uuid(uuid)
    if link is None:
        return None
    lane = _PCIE_LANE_GBPS.get(link.generation)
    if lane is None:
        return None
    return lane * link.width


def derive_link_weights(
    card_uuids: Sequence[str], link_gbps=None
) -> Optional[Tuple[float, ...]]:
    """Per-rank weights from PCIe link width x generation, or ``None``.

    ``card_uuids[r]`` is the NVML UUID of the card serving rank ``r``; the
    caller gathers it (a rank knows its own via
    ``registry.nvml.current_device_uuid``). ``link_gbps`` is the injectable
    ``uuid -> Optional[float]`` lookup the hermetic tests use in place of a
    driver.

    Two ranks CO-LOCATED on one card share one link, so that card's bandwidth
    is divided between them: the quantity being apportioned is link seconds,
    and two ranks behind one x8 slot have an x4's worth each. ``None`` when any
    card cannot be resolved -- a partial derivation is worse than no
    derivation, because the ranks would disagree about the split.
    """
    from collections import Counter

    if not card_uuids:
        return None
    lookup = _pcie_link_gbps_by_uuid if link_gbps is None else link_gbps
    ranks_per_card = Counter(card_uuids)
    weights = []
    for uuid in card_uuids:
        gbps = lookup(uuid)
        if gbps is None or not (float(gbps) > 0.0):
            return None
        weights.append(float(gbps) / ranks_per_card[uuid])
    return _normalize_weights(weights)


def resolve_host_shard_ratio(
    world_size: int,
    card_uuids: Optional[Sequence[str]] = None,
    link_gbps=None,
    probe_gbps=None,
) -> HostShardRatio:
    """The ratio and its provenance, in strict preference order.

    1. ``SGLANG_MOE_HOST_SHARD_RATIO`` -- an explicit vector always wins;
       malformed input raises rather than falling through.
    2. MEASURED pinned H2D bandwidth per rank's card, read from the rigmon
       card probe by UUID. This is a timed 64 MiB transfer over the link the
       cold experts will actually cross, which is the quantity being
       apportioned -- nothing else in the chain measures it.
    3. ESTIMATE: NVML PCIe link width x generation for each rank's card, again
       resolved by UUID through the #331 IdentityMap. A formula over a measured
       link width, not a transfer anybody timed, and admitted only while
       ``SGLANG_MOE_HOST_SHARD_MIN_PROVENANCE`` allows an estimate.
    4. Equal -- the shape a REFUSAL takes. An unknown link is not an excuse to
       guess, and an equal ratio reproduces today's assignment exactly, so
       refusing costs nothing beyond the speedup nobody could justify.

    ``link_gbps`` / ``probe_gbps`` are the injectable ``uuid -> Optional[float]``
    lookups the hermetic tests use in place of a driver and a probe cache.
    """
    n = max(1, int(world_size))
    minimum = _min_provenance()

    from_env = _host_shard_ratio_from_env(n)
    if from_env is not None:
        return from_env

    if card_uuids is None:
        return equal_host_shard_ratio(n, "no per-rank card identity supplied")
    if len(card_uuids) != n:
        return equal_host_shard_ratio(
            n,
            f"the card vector names {len(card_uuids)} ranks but the group has "
            f"{n}; that vector describes a different group",
        )

    measured = derive_link_weights(
        card_uuids, link_gbps=probe_gbps or _measured_h2d_gbps_by_uuid
    )
    if measured is not None:
        return HostShardRatio(
            measured,
            HOST_SHARD_SOURCE_PROBE,
            "measured pinned H2D per rank's card (rigmon card probe, by UUID)",
        )

    if not _provenance_admitted("estimate", minimum):
        return equal_host_shard_ratio(
            n,
            "no measured H2D bandwidth for every rank's card and "
            f"{HOST_SHARD_MIN_PROVENANCE_ENV}=measured forbids weighting on "
            "the nameplate derivation; run the card probe "
            "(python -m sglang.srt.rigmon.card_probe) to fill it in",
        )

    derived = derive_link_weights(card_uuids, link_gbps=link_gbps)
    if derived is not None:
        return HostShardRatio(
            derived,
            HOST_SHARD_SOURCE_NVML,
            "max PCIe link width x generation per rank's card (NVML, by UUID)",
        )
    return equal_host_shard_ratio(
        n, "neither a measured H2D rate nor NVML PCIe link data for these cards"
    )


def plan_proportional_shares(total: int, weights: Sequence[float]) -> Tuple[int, ...]:
    """Apportion ``total`` WHOLE units over ``weights`` (largest remainder).

    Whole units because a cold expert cannot be cut: dim 0 is the one axis of a
    quantized expert stack with no block structure on it (#82/#109). Largest
    remainder (Hamilton) rather than repeated rounding, so the shares sum to
    ``total`` exactly and the result is a pure function of the inputs -- every
    rank computes the same partition without a collective. Ties go to the lower
    rank index, which is arbitrary but fixed.
    """
    total = int(total)
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    norm = _normalize_weights(weights)
    exact = [total * w for w in norm]
    floors = [int(math.floor(x)) for x in exact]
    remaining = total - sum(floors)
    order = sorted(range(len(norm)), key=lambda i: (-(exact[i] - floors[i]), i))
    for i in order[:remaining]:
        floors[i] += 1
    return tuple(floors)


def partition_cold_experts(
    cold_ids: Sequence[int], weights: Sequence[float]
) -> Tuple[Tuple[int, ...], ...]:
    """Split the cold pool into one ascending, contiguous block per rank.

    Contiguous and ascending on purpose: the owning rank's spill pool then has
    the same "row j holds the j-th smallest owned id" shape the static layout
    has, so ``_spill_pool_index``, the frozen-layout adoption in
    ``MoEExpertOffloadCache`` and the capturable LUT builder all keep working
    unchanged. Nothing here depends on WHICH experts a rank gets, only on how
    many -- routing is uniform enough over the cold pool that the cheapest
    correct partition is the right one.
    """
    ids = [int(e) for e in cold_ids]
    shares = plan_proportional_shares(len(ids), weights)
    out = []
    cursor = 0
    for count in shares:
        out.append(tuple(ids[cursor : cursor + count]))
        cursor += count
    return tuple(out)


@dataclass(frozen=True)
class ColdShardContext:
    """This rank's view of the link-proportional cold-expert partition.

    Constructed ONLY by a caller whose layer shards experts on dim 0 and remaps
    foreign expert ids away from this rank (see the PRECONDITION note above).
    ``None`` in place of a context is the default path, byte for byte.
    """

    rank: int
    world_size: int
    ratio: HostShardRatio

    def __post_init__(self):
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"rank {self.rank} is outside [0,{self.world_size})")
        if len(self.ratio.weights) != self.world_size:
            raise ValueError(
                f"host-shard ratio has {len(self.ratio.weights)} weights but "
                f"the group has {self.world_size} ranks"
            )

    @property
    def active(self) -> bool:
        """False when the ratio says nothing -- then the plan is today's plan."""
        return self.world_size > 1 and not self.ratio.is_equal


def cold_shard_context(
    rank: int,
    world_size: int,
    card_uuids: Optional[Sequence[str]] = None,
    link_gbps=None,
    probe_gbps=None,
) -> Optional[ColdShardContext]:
    """Resolve the ratio and wrap it, or ``None`` when there is nothing to do.

    ``None`` for a single-rank group or an equal ratio, so the caller's
    ``cold_shard=`` argument is literally absent on the default path instead of
    being a context that happens to be a no-op. That is what makes "no ratio
    known" and "no #394" the same code path.
    """
    if int(world_size) < 2:
        return None
    ratio = resolve_host_shard_ratio(
        world_size, card_uuids, link_gbps=link_gbps, probe_gbps=probe_gbps
    )
    if ratio.is_equal:
        return None
    return ColdShardContext(int(rank), int(world_size), ratio)


def _log_host_shard_choice(context: "ColdShardContext", owned: int, pool: int) -> None:
    """One INFO line per process naming the ratio, its source and this share."""
    global _HOST_SHARD_LOGGED

    import logging

    with _HOST_SHARD_LOG_LOCK:
        if _HOST_SHARD_LOGGED:
            return
        _HOST_SHARD_LOGGED = True
    share = (owned / pool) if pool else 0.0
    logging.getLogger(__name__).info(
        "MoE cold-expert host shard (#394): rank %d/%d owns %d of %d cold "
        "experts (%.1f%% of the pool) -- %s",
        context.rank,
        context.world_size,
        owned,
        pool,
        100.0 * share,
        context.ratio.describe(),
    )


def reset_host_shard_log_latch() -> None:
    """Test hook: re-arm the once-per-process log line."""
    global _HOST_SHARD_LOGGED

    _HOST_SHARD_LOGGED = False


def host_shard_row(plan: "ExpertStagingPlan") -> dict:
    """The #394 policy this layer was staged under, as a dump row.

    Written into the #390 expert-stats file so a measurement arm identifies its
    own placement policy. Two runs of an A/B differ in exactly this row, and
    reading which arm produced a JSON file out of the file itself is what stops
    a pair of dumps from being un-attributable a week later.

    ``owned`` and ``delegated`` are counts of THIS rank's cold pool, so
    ``owned / (owned + delegated)`` is the realized share against the ratio the
    policy asked for -- the whole-expert rounding error is visible rather than
    assumed away.
    """
    owned = len(plan.spill_ids)
    delegated = len(plan.delegated_ids)
    pool = owned + delegated
    return {
        "policy": "link-proportional" if delegated else "equal",
        "ratio": plan.host_shard or "",
        "cold_pool": pool,
        "owned_cold_experts": owned,
        "delegated_cold_experts": delegated,
        "owned_share": (owned / pool) if pool else 0.0,
        "resident_count": plan.resident_count,
        "num_experts": plan.num_experts,
    }


def publish_host_shard_on_layer(layer, plan: "ExpertStagingPlan") -> None:
    """Attach the #394 row to the layer, for the #390 instrument to pick up.

    On the layer rather than passed down a call chain because the two staging
    doors reach the cache by different routes and only the layer is common to
    both. Always written, including on the equal/default path -- a row saying
    ``policy=equal`` is what makes the baseline arm of an A/B self-describing
    instead of merely silent.
    """
    try:
        layer._moe_offload_host_shard = host_shard_row(plan)
    except Exception:  # noqa: BLE001 - instrumentation must never fail a load
        pass


# ===========================================================================
# #123-GGUF: MATERIALIZATION-TIME staging (the third load-time entry point).
#
# fp8 / GPTQ / AWQ reach the offload through
# ``presplit_expert_offload_after_repack``: by the time it runs, a real
# ``[E, ...]`` expert stack already exists and is merely split. GGUF cannot use
# that door. Its expert parameter is a ``GGUFUninitializedParameter`` with no
# storage at all until ``materialize_gguf_weights`` stacks the per-expert
# tensors the loader collected -- and that stack is exactly the allocation the
# offload exists to avoid (it is built on the host and copied to the card in
# full, so both peaks are paid before any presplit could run).
#
# So the GGUF half intercepts one step EARLIER: instead of splitting a stack
# that exists, it decides residency FIRST and then materializes only the
# resident slots on the device, streaming every other expert straight into the
# pinned host tier. The full stack is never formed on either side.
#
# The three functions below are the reusable half of that: plan (pure), stage
# (per-expert copy into the two tiers), register (hand the tiers to the cache
# in the same ``_moe_offload_presplit`` shape the repack door uses). They take
# a per-expert ``source(expert_id) -> Tensor`` callable rather than a stacked
# tensor, which is what makes them usable before materialization.
# ===========================================================================


@dataclass(frozen=True)
class ExpertStagingPlan:
    """Which expert lands where, decided BEFORE any tensor is allocated.

    ``resident_ids[i]`` is the expert that occupies GPU slot ``i`` (``i < R``);
    ``spill_ids[j]`` is the expert at pinned-pool row ``j``. Tuples, so the
    plan is hashable, comparable and printable in a test.

    ``pinned_experts`` (see ``plan_load_time_staging``) is why the layout is
    carried explicitly instead of being the implicit static ``[0, R)``.

    ``delegated_ids`` (#394) are cold experts a PEER rank's host tier owns.
    They are neither staged nor fetched here, and the three tuples together
    always cover ``range(num_experts)`` exactly once. Empty on every path that
    does not pass a ``ColdShardContext``, which is every path today.
    """

    num_experts: int
    resident_count: int
    buffer_slots: int
    resident_ids: Tuple[int, ...]
    spill_ids: Tuple[int, ...]
    delegated_ids: Tuple[int, ...] = ()
    host_shard: Optional[str] = None

    @property
    def is_static_layout(self) -> bool:
        """True when the plan is exactly the default ``[0,R)`` residency."""
        R = self.resident_count
        return (
            not self.delegated_ids
            and self.resident_ids == tuple(range(R))
            and self.spill_ids == tuple(range(R, self.num_experts))
        )


def plan_load_time_staging(
    num_experts: int,
    fraction: Optional[float] = None,
    pinned_experts: Sequence[int] = (),
    cold_shard: Optional[ColdShardContext] = None,
) -> Optional[ExpertStagingPlan]:
    """Residency plan for a load-time split, or ``None`` when there is none.

    ``None`` means "do not offload this layer": either the resident fraction is
    >= 1.0, or ceil(fraction * E) already covers every expert. Callers treat
    ``None`` as "materialize the full stack the way you always did", which is
    what keeps the default path byte-identical.

    ``pinned_experts`` are expert ids that MUST be resident regardless of the
    ordering. The GGUF uneven-TP expert-dim shard (#82) needs exactly this: its
    trailing all-zero padding expert sits at id ``E-1`` -- the last id, so the
    static ``[0,R)`` layout would put the one expert that EVERY foreign token
    routes to in the spill pool and re-fetch it on every single forward. Pinned
    ids take the lowest slots; the remaining slots are filled in ascending id
    order, so the layout stays a pure function of (E, R, pinned) and therefore
    deterministic across ranks and runs.

    ``cold_shard`` (#394) narrows the SPILL set to this rank's link-proportional
    share of the cold pool; the rest is recorded as ``delegated_ids`` and is a
    peer's host tier to hold. Residency is decided FIRST and is not a function
    of the ratio, so ``resident_ids``, ``resident_count`` and ``buffer_slots``
    -- the three numbers every VRAM figure and #400 ledger entry comes from --
    are identical with and without a ratio. ``None`` (the default, and the only
    thing any caller passes today) reproduces the previous plan exactly.
    """

    E = int(num_experts)
    if E <= 0:
        return None
    from sglang.srt.layers.moe.resident_fraction import resident_fraction_for_rank

    # SIZING: this number decides how many experts stay on THIS rank's GPU, so
    # it must be this rank's own fraction, not a group-wide one.
    frac = resident_fraction_for_rank() if fraction is None else float(fraction)
    if frac >= 1.0:
        return None
    R = resident_slot_count(E, frac)
    if R >= E:
        return None
    pinned = sorted({int(e) for e in pinned_experts})
    for e in pinned:
        if e < 0 or e >= E:
            raise ValueError(f"pinned expert id {e} out of range [0,{E})")
    if len(pinned) > R:
        raise ValueError(
            f"{len(pinned)} experts must stay resident but only {R} resident "
            f"slots exist at fraction {frac} over {E} experts; raise "
            f"SGLANG_MOE_RESIDENT_EXPERT_FRACTION."
        )
    pinned_set = set(pinned)
    rest = [e for e in range(E) if e not in pinned_set]
    resident_ids = pinned + rest[: R - len(pinned)]
    resident_set = set(resident_ids)
    spill_ids = [e for e in range(E) if e not in resident_set]
    C = scratch_slot_count(R)

    # #394: residency above is already fixed; only the cold pool is re-owned.
    # A pinned expert (the #82 pad expert at id E-1) is resident, so it is not
    # in the pool and cannot be delegated -- the two features compose without
    # either knowing about the other.
    delegated_ids: Tuple[int, ...] = ()
    host_shard = None
    if cold_shard is not None and cold_shard.active:
        shares = partition_cold_experts(spill_ids, cold_shard.ratio.weights)
        owned = set(shares[cold_shard.rank])
        delegated_ids = tuple(e for e in spill_ids if e not in owned)
        spill_ids = [e for e in spill_ids if e in owned]
        host_shard = cold_shard.ratio.describe()
        _log_host_shard_choice(
            cold_shard, len(spill_ids), len(spill_ids) + len(delegated_ids)
        )

    return ExpertStagingPlan(
        num_experts=E,
        resident_count=R,
        buffer_slots=min(R + C, E),
        resident_ids=tuple(resident_ids),
        spill_ids=tuple(spill_ids),
        delegated_ids=delegated_ids,
        host_shard=host_shard,
    )


def allocate_spill_pool(spill_ids, row_shape, dtype, cold_tier=None, param_attr=""):
    """The pinned host cold pool for one expert-major tensor.

    One function for the two #123-GGUF doors (the pull loop and the streaming
    stager) so they cannot drift on WHERE the cold bytes live. The marlin
    repack door keeps its own allocation because it refuses a cold shard
    outright (:func:`refuse_cold_shard_at_repack_door`) and therefore never has
    a tier to share. Without a ``cold_tier`` this is the allocation it was: a
    private ``torch.empty(...).pin_memory()``, page-locked only when a CUDA
    context exists (desk tests have none).

    With one (#394 slice 2), the storage IS the shared segment -- not a copy
    into it. That distinction is the difference between a feature that shares
    the cold tier and one that doubles it, and the reference rig's host RAM was
    already the binding constraint.
    """
    import torch

    shape = tuple(int(d) for d in row_shape)
    ids = tuple(int(e) for e in spill_ids)
    if cold_tier is not None:
        return cold_tier.allocate_spill_pool(param_attr, ids, shape, dtype)
    if not ids:
        return None
    pool = torch.empty((len(ids),) + shape, dtype=dtype, device="cpu")
    if torch.cuda.is_available():
        pool = pool.pin_memory()
    return pool


def stage_experts_into_tiers(
    plan: ExpertStagingPlan, source, out, release=None, cold_tier=None, param_attr=""
):
    """Fill the two tiers from a per-expert ``source(expert_id) -> Tensor``.

    ``out`` is the caller-allocated ``[buffer_slots, ...]`` device buffer; rows
    ``[0, R)`` are written from ``plan.resident_ids`` and the scratch region
    ``[R, buffer_slots)`` is deliberately left uninitialized (the fetch path
    overwrites it before any read). Returns the ``[E-R, ...]`` pinned host
    spill pool, row ``j`` holding ``plan.spill_ids[j]``.

    ``source`` is called EXACTLY ONCE per expert, in staging order, and the
    result is copied immediately -- so a source that frees its per-expert
    tensor after handing it over (``release``) keeps host peak at one expert
    above the two tiers, not at the full stack. Pinning is skipped when there
    is no CUDA context (desk tests); production always has one, and the pinned
    pool is what makes the H2D fetch async.

    No reshaping, padding or re-blocking happens here: an expert's bytes are
    copied whole. That is the property GGUF depends on -- its rows are opaque
    quantization blocks (Q4_K 144 B / 256 values, Q6_K 210 B / 256 values), so
    ANY split other than "whole experts on the expert axis" would cut a block
    in half. The expert axis is the one axis with no block structure on it.
    """
    import torch

    R = plan.resident_count
    for slot, expert_id in enumerate(plan.resident_ids):
        out[slot].copy_(source(expert_id))
        if release is not None:
            release(expert_id)
    if R != len(plan.resident_ids):  # defensive: plan invariant
        raise RuntimeError("staging plan resident_ids length != resident_count")

    spill = None
    for row, expert_id in enumerate(plan.spill_ids):
        src = source(expert_id)
        if spill is None:
            spill = allocate_spill_pool(
                plan.spill_ids,
                tuple(src.shape),
                src.dtype,
                cold_tier=cold_tier,
                param_attr=param_attr,
            )
        spill[row].copy_(src)
        if release is not None:
            release(expert_id)

    # #394: a delegated cold expert belongs to a peer's host tier. Its bytes are
    # never read here -- but the loader is still holding them, so they are
    # released without a copy. Skipping the release instead would trade the
    # VRAM this feature saves for host RAM it never used.
    for expert_id in plan.delegated_ids:
        if release is not None:
            release(expert_id)
    return spill


def register_load_time_presplit(layer, attr, resident_buf, spill, plan):
    """Publish one staged tensor in the shape ``MoEExpertOffloadCache.install``
    already understands, and tally the VRAM it means the layer never took.

    Same contract as ``presplit_expert_offload_after_repack``'s stash
    (``layer._moe_offload_presplit[attr] = (resident_buf, spill)`` +
    ``_moe_offload_full_experts``), plus ``_moe_offload_frozen_layout`` for the
    non-static case, which the cache adopts as its frozen residency map.
    """
    presplit = getattr(layer, "_moe_offload_presplit", None)
    first_tensor_of_layer = presplit is None
    if presplit is None:
        presplit = {}
        layer._moe_offload_presplit = presplit
    presplit[attr] = (resident_buf, spill)
    layer._moe_offload_full_experts = plan.num_experts
    if not plan.is_static_layout:
        layer._moe_offload_frozen_layout = (
            list(plan.resident_ids),
            list(plan.spill_ids),
        )
    if plan.delegated_ids:
        # #394: the cache turns this into a named refusal if a delegated expert
        # ever reaches this rank's router, instead of a missing pool row.
        layer._moe_offload_delegated_experts = list(plan.delegated_ids)
    publish_host_shard_on_layer(layer, plan)
    row_bytes = (
        (resident_buf.numel() // resident_buf.shape[0]) * resident_buf.element_size()
        if resident_buf.shape[0]
        else 0
    )
    record_expert_offload_release(
        expert_offload_released_device_bytes(
            plan.num_experts, plan.buffer_slots, row_bytes
        ),
        (spill.numel() * spill.element_size()) if spill is not None else 0,
        1,
        count_layer=first_tensor_of_layer,
    )


# ===========================================================================
# #391c: STREAMING staging -- the same two tiers, filled from the weight stream
#
# ``stage_experts_into_tiers`` above is a PULL loop: it asks a ``source`` for
# expert after expert, which presumes every expert is already sitting somewhere
# the source can hand it over from. For GGUF that "somewhere" is the loader's
# ``param.expert_data_map``, and filling it is the whole load pass -- so the
# residency plan only ever got to act on a set that had already been paid for
# in host RAM. On DeepSeek-V4-Flash UD-Q3_K_XL that set is 126.19 GiB of
# post-repack experts against 98.5 GiB of swapless host RAM, and boot attempt 5
# of #391 was OOM-killed at 90.7 GiB of anon mid-load, before the plan existed.
#
# ``StreamingExpertStager`` is the same placement, PUSHED: the plan is computed
# from config-level facts alone (expert count, resident fraction, the #82 pad
# expert, the #394 ratio), so it exists before the first tensor arrives, and
# each expert is copied into its resident slot or its pinned row AS IT LEAVES
# THE STREAM and then dropped. Nothing is retained but the tiers themselves and
# the shards of experts whose set is still incomplete -- for GGUF's w13 that is
# at most one layer's gate shards, because the iterator emits one whole
# ``ffn_gate_exps`` tensor before the matching ``ffn_up_exps``.
#
# The tiers this produces are byte-for-byte the tiers the pull loop produces
# from the same plan and the same inputs; only the ORDER of the copies differs
# (stream order rather than plan order), and a copy's destination is a pure
# function of the plan.
# ===========================================================================


@dataclass
class StreamingStagingLedger:
    """The stager's own byte accounting, cumulative over a process.

    Kept next to the code that does the copying rather than derived from an
    external RAM monitor: a monitor sees the whole process, this sees only what
    the staging is responsible for, and the interesting number is whether the
    two move together. ``peak_host_bytes`` is the claim boot 6 has to beat --
    pinned tier plus whatever was in flight at the worst moment.
    """

    streamed_bytes: int = 0
    resident_bytes: int = 0
    pinned_bytes: int = 0
    delegated_bytes: int = 0
    inflight_bytes: int = 0
    peak_inflight_bytes: int = 0
    peak_host_bytes: int = 0
    layers: int = 0
    tensors: int = 0
    #: #391: the weight loaders run on a ThreadPoolExecutor, so several stagers
    #: (and several experts of one stager) reach this ledger at once. ``x += n``
    #: on a field is a read-modify-write and drops updates under threads, which
    #: would make the very number the host-RAM model is judged against
    #: silently low. All mutation goes through :meth:`record`.
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def record(
        self,
        *,
        streamed: int = 0,
        resident: int = 0,
        pinned: int = 0,
        delegated: int = 0,
        inflight: int = 0,
        tensors: int = 0,
        layers: int = 0,
    ) -> None:
        """Apply one atomic set of deltas and re-touch the peaks."""
        with self._lock:
            self.streamed_bytes += streamed
            self.resident_bytes += resident
            self.pinned_bytes += pinned
            self.delegated_bytes += delegated
            self.inflight_bytes += inflight
            self.tensors += tensors
            self.layers += layers
            self._touch_peak()

    def _touch_peak(self) -> None:
        self.peak_inflight_bytes = max(self.peak_inflight_bytes, self.inflight_bytes)
        self.peak_host_bytes = max(
            self.peak_host_bytes, self.pinned_bytes + self.inflight_bytes
        )


_STAGING_LEDGER = StreamingStagingLedger()


def streaming_staging_ledger() -> StreamingStagingLedger:
    """The process-wide staging ledger (read-only for callers; tests reset)."""
    return _STAGING_LEDGER


def reset_streaming_staging_ledger() -> None:
    """Clear the ledger (tests; and a second model load in one process)."""
    global _STAGING_LEDGER
    _STAGING_LEDGER = StreamingStagingLedger()


def trim_host_allocator() -> None:
    """Give the per-expert buffers back to the OS, not just to glibc's arena.

    The #256 lesson: dropping the last reference to an expert returns its bytes
    to torch's CPU allocator and glibc's arena, and RSS does not move. Across
    40+ MoE layers that retention IS the expert set, on a box with no swap.
    Called at each layer boundary -- often enough that the arena never grows to
    layer count x layer size, rarely enough that the arena walk is noise next to
    a layer's copies.
    """
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 - non-glibc platforms simply have no trim
        pass


def _human_bytes(nbytes: int) -> str:
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if abs(nbytes) >= scale:
            return f"{nbytes / scale:.2f} {unit}"
    return f"{nbytes} B"


def log_streaming_staging_layer(label: str, plan: "ExpertStagingPlan") -> None:
    """One trace line per finished layer, gated on SGLANG_MOE_STAGING_TRACE.

    Emitted at the LAYER boundary, which is the granularity an external
    ram-monitor can actually be lined up against: the cumulative pinned figure
    here should track the monitor's anon curve, and ``in-flight peak`` is the
    transient the curve is allowed to bulge by.
    """
    import logging

    from sglang.srt.environ import envs

    ledger = _STAGING_LEDGER
    ledger.record(layers=1)
    if not envs.SGLANG_MOE_STAGING_TRACE.get():
        return
    logging.getLogger(__name__).info(
        "[moe-staging-trace] %s staged (#%d): %d/%d experts resident, "
        "%d pinned, %d delegated | cumulative streamed=%s resident=%s "
        "pinned(host)=%s delegated=%s | in-flight now=%s peak=%s | "
        "peak host held (pinned+in-flight)=%s",
        label,
        ledger.layers,
        plan.resident_count,
        plan.num_experts,
        len(plan.spill_ids),
        len(plan.delegated_ids),
        _human_bytes(ledger.streamed_bytes),
        _human_bytes(ledger.resident_bytes),
        _human_bytes(ledger.pinned_bytes),
        _human_bytes(ledger.delegated_bytes),
        _human_bytes(ledger.inflight_bytes),
        _human_bytes(ledger.peak_inflight_bytes),
        _human_bytes(ledger.peak_host_bytes),
    )


def _nbytes(tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


class StreamingExpertStager:
    """Place each expert into its tier as the weight stream delivers it.

    One instance per (layer, expert-major parameter). The caller feeds shards
    with :meth:`submit` in whatever order the checkpoint happens to use and
    calls :meth:`finalize` once the stream is over; the result is the same
    ``(resident_buffer, pinned_spill_pool)`` pair
    ``stage_experts_into_tiers`` returns.

    ``shard_keys`` is the ordered tuple of per-expert shards that make up one
    row -- ``("w1", "w3")`` for a GGUF ``w13_qweight`` (gate above up, the
    concatenation the stacked path builds) and ``("w2",)`` for the down
    projection. An expert is placed only once ALL of its shards have arrived,
    which is what bounds the retained set: shards of incomplete experts.

    ``allocate(row_shape, dtype) -> Tensor`` materializes the caller's
    ``[buffer_slots, *row_shape]`` resident buffer. It is called lazily, on the
    first complete expert, because the row shape is a property of the
    checkpoint's quantized bytes and is not known before one has been seen.

    ``zero_experts`` are ids with no bytes in the stream that must still occupy
    their planned slot -- exactly the #82 uneven-TP shard's trailing all-zero
    padding expert, the target of every foreign topk id. They are written in
    :meth:`finalize`, once the row shape is known.
    """

    def __init__(
        self,
        plan: "ExpertStagingPlan",
        shard_keys: Sequence[str],
        allocate,
        zero_experts: Sequence[int] = (),
        label: str = "",
        cold_tier=None,
        param_attr: str = "",
    ):
        self.plan = plan
        self.shard_keys = tuple(shard_keys)
        self.label = label
        # #394 slice 2: when set, this stager's cold rows are written straight
        # into the shared segment a peer will read them from. ``None`` is the
        # default and the previous private pinned pool.
        self._cold_tier = cold_tier
        self._param_attr = param_attr
        self._allocate = allocate
        self._zero_experts = tuple(sorted({int(e) for e in zero_experts}))
        # expert id -> ("resident", slot) | ("spill", row) | None (delegated).
        self._dest: Dict[int, Optional[Tuple[str, int]]] = {}
        for slot, expert_id in enumerate(plan.resident_ids):
            self._dest[int(expert_id)] = ("resident", slot)
        for row, expert_id in enumerate(plan.spill_ids):
            self._dest[int(expert_id)] = ("spill", row)
        for expert_id in plan.delegated_ids:
            self._dest[int(expert_id)] = None
        self._pending: Dict[int, Dict[str, Optional[object]]] = {}
        #: Host bytes this stager is currently holding per incomplete expert.
        #: Kept per expert rather than recomputed at placement time so the
        #: all-zero pad expert -- which is built in finalize() and was never in
        #: flight -- cannot subtract bytes it never added.
        self._inflight: Dict[int, int] = {}
        self._placed: set = set()
        self._row_shape: Optional[Tuple[int, ...]] = None
        self._dtype = None
        self.resident_buf = None
        self.spill = None
        self.finalized = False
        #: #391: one stager is fed by MANY loader threads. The weight loaders
        #: run on a ThreadPoolExecutor and every GGUF tensor is a CPU tensor, so
        #: gate and up of one layer -- both routed into the same ``w13_qweight``
        #: stager -- are submitted concurrently. This lock covers every piece of
        #: shared state below: the two tiers and the row-shape guard that says
        #: whether they exist, the pending-shard map, and the placed set. Held
        #: only around bookkeeping; the per-expert ``copy_`` runs outside it, so
        #: experts still land in parallel (they write disjoint rows).
        self._lock = threading.RLock()

    @property
    def is_complete(self) -> bool:
        """Every expert the stream owes this tensor has been placed.

        The all-zero experts are excluded: they carry no stream bytes and are
        written in :meth:`finalize`, so waiting for them would mean the layer
        boundary never fires during the load.
        """
        with self._lock:
            return not self._pending and len(self._placed) == len(self._dest) - len(
                self._zero_experts
            )

    # -- stream side --------------------------------------------------------

    def submit(self, expert_id: int, shard_id: str, tensor) -> None:
        """Take one shard of one expert out of the stream.

        The tensor is either copied into its tier (when this completes the
        expert) or held until the rest of the expert arrives. Either way the
        caller must not keep a reference of its own -- holding one turns the
        "one incomplete expert set" bound back into "the whole loaded set",
        which is the #256 lesson and the whole point of this class.

        Concurrent-safe: the shard set of an expert is completed and CLAIMED
        under the lock, so two threads carrying the two halves of one expert
        cannot both decide they were the last one. Only the claiming thread
        goes on to copy, and it copies outside the lock.
        """
        expert_id = int(expert_id)
        with self._lock:
            if self.finalized:
                raise RuntimeError(
                    f"streaming stager for {self.label!r} got expert "
                    f"{expert_id}/{shard_id} after finalize()"
                )
            if shard_id not in self.shard_keys:
                raise ValueError(
                    f"streaming stager for {self.label!r} takes shards "
                    f"{list(self.shard_keys)}, got {shard_id!r}"
                )
            if expert_id not in self._dest:
                raise KeyError(
                    f"streaming stager for {self.label!r} has no plan slot for "
                    f"expert {expert_id}; the plan covers "
                    f"[0,{self.plan.num_experts})"
                )
            if expert_id in self._placed:
                raise RuntimeError(
                    f"streaming stager for {self.label!r}: expert {expert_id} was "
                    "already placed; the stream delivered it twice"
                )
            parts = self._pending.setdefault(expert_id, {})
            if shard_id in parts:
                raise RuntimeError(
                    f"streaming stager for {self.label!r}: expert {expert_id} got "
                    f"shard {shard_id!r} twice"
                )
            nbytes = _nbytes(tensor)
            if self._dest[expert_id] is None:
                # #394: a peer rank's host tier owns this cold expert. Released
                # here without ever being copied -- keeping it would trade the
                # VRAM the feature saves for host RAM nobody reads.
                parts[shard_id] = None
                _STAGING_LEDGER.record(streamed=nbytes, delegated=nbytes)
            else:
                parts[shard_id] = tensor
                self._inflight[expert_id] = self._inflight.get(expert_id, 0) + nbytes
                _STAGING_LEDGER.record(streamed=nbytes, inflight=nbytes)
            complete = len(parts) == len(self.shard_keys)
            if complete:
                del self._pending[expert_id]
                self._placed.add(expert_id)
        if complete:
            self._place(expert_id, parts)

    def _place(self, expert_id: int, parts) -> None:
        import torch

        with self._lock:
            self._placed.add(expert_id)
            held = self._inflight.pop(expert_id, 0)
        dest = self._dest[expert_id]
        if dest is None:
            parts.clear()
            return
        ordered = [parts[key] for key in self.shard_keys]
        parts.clear()
        row = ordered[0] if len(ordered) == 1 else torch.cat(ordered, dim=0)
        del ordered
        resident_buf, spill = self._ensure_tiers(row)
        kind, index = dest
        if kind == "resident":
            resident_buf[index].copy_(row)
            _STAGING_LEDGER.record(resident=_nbytes(row), inflight=-held)
        else:
            spill[index].copy_(row)
            _STAGING_LEDGER.record(inflight=-held)

    def _ensure_tiers(self, row):
        """Build the two tiers on the first complete expert; return them.

        #391 boot 10: this used to publish ``self._row_shape`` -- the guard that
        says "the tiers exist" -- BEFORE allocating them, and ran unlocked while
        the loader's thread pool pushed experts of the same tensor in parallel.
        A second thread arriving inside that window saw the guard, skipped the
        build and got a pair whose halves were still ``None``, and ``_place``
        subscripted it: ``TypeError: 'NoneType' object is not subscriptable``,
        three boots out of three. ``spill.pin_memory()`` is what makes the
        window wide enough to hit reliably -- it is a page-locking allocation of
        the whole cold tier.

        The build is therefore serialized and the guard published LAST, after
        both tiers exist. Both halves matter: the lock alone would still let the
        ``elif`` shape check read a half-published ``_row_shape``/``_dtype``
        pair if the build ever raised, and the ordering alone would still race
        two threads into two allocations of the same tier.
        """
        import torch

        row_shape = tuple(int(d) for d in row.shape)
        with self._lock:
            if self._row_shape is None:
                buf = self._allocate(row_shape, row.dtype)
                if tuple(buf.shape) != (self.plan.buffer_slots,) + row_shape:
                    raise RuntimeError(
                        f"streaming stager for {self.label!r}: allocate() returned "
                        f"{tuple(buf.shape)}, expected "
                        f"{(self.plan.buffer_slots,) + row_shape}"
                    )
                # Pinning is what makes the later H2D fetch async; skipped when
                # there is no CUDA context (desk tests), as in the pull loop.
                # With a #394 cold tier the pool IS the shared segment and the
                # page-locking is a cudaHostRegister on the mapping instead.
                spill = allocate_spill_pool(
                    self.plan.spill_ids,
                    row_shape,
                    row.dtype,
                    cold_tier=self._cold_tier,
                    param_attr=self._param_attr,
                )
                # Publish only now: every reader of _row_shape takes it to mean
                # that BOTH tiers below are already built.
                self.resident_buf = buf
                self.spill = spill
                self._dtype = row.dtype
                self._row_shape = row_shape
                _STAGING_LEDGER.record(
                    pinned=_nbytes(spill) if spill is not None else 0, tensors=1
                )
            elif row_shape != self._row_shape or row.dtype != self._dtype:
                raise RuntimeError(
                    f"streaming stager for {self.label!r}: expert row is "
                    f"{row_shape}/{row.dtype} but the tiers were built for "
                    f"{self._row_shape}/{self._dtype}; experts of one tensor must "
                    "be uniform"
                )
            return self.resident_buf, self.spill

    # -- end of stream ------------------------------------------------------

    def finalize(self):
        """Write the zero experts, check the plan is covered, hand over tiers.

        Returns ``(resident_buffer, spill_pool_or_None)`` -- the same pair
        ``stage_experts_into_tiers`` returns, for the same
        ``register_load_time_presplit`` call.

        Under the same lock as the stream side: the drain happens after the
        loader's pool has joined, but a stager that closes while a straggler is
        still in ``submit`` has to see either the whole submission or none of
        it, not a half-filled shard set that reads as "incomplete".
        """
        import torch

        with self._lock:
            if self.finalized:
                raise RuntimeError(
                    f"streaming stager for {self.label!r} finalized twice"
                )
            if self._pending:
                incomplete = {
                    e: sorted(k for k in parts)
                    for e, parts in sorted(self._pending.items())
                }
                raise RuntimeError(
                    f"streaming stager for {self.label!r}: the stream ended with "
                    f"incomplete experts {incomplete}; every expert needs all of "
                    f"{list(self.shard_keys)}"
                )
            if self._row_shape is None:
                raise RuntimeError(
                    f"streaming stager for {self.label!r}: the stream delivered no "
                    "expert at all, so there is no row shape to build the tiers "
                    "from"
                )
            zero_shard_shape = self._zero_shard_shape(self._row_shape)
            for expert_id in self._zero_experts:
                if expert_id in self._placed:
                    raise RuntimeError(
                        f"streaming stager for {self.label!r}: expert {expert_id} "
                        "is declared all-zero but the stream delivered bytes for it"
                    )
                self._place(
                    expert_id,
                    {
                        key: torch.zeros(zero_shard_shape, dtype=self._dtype)
                        for key in self.shard_keys
                    },
                )
            missing = sorted(set(self._dest) - self._placed)
            if missing:
                raise RuntimeError(
                    f"streaming stager for {self.label!r}: the stream never "
                    f"delivered experts {missing}; the plan reserved a tier slot "
                    "for each of them"
                )
            self.finalized = True
            return self.resident_buf, self.spill

    def _zero_shard_shape(self, row_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Shape of ONE shard of the all-zero pad expert.

        The shards of one expert are concatenated on the row axis, and the pad
        expert's row must match every other expert's, so each of the ``k``
        shards contributes ``rows // k``. GGUF's w13 is gate+up of identical
        width, which is the only multi-shard case; a non-integral split means
        the assumption no longer holds and is an error rather than a guess.
        """
        rows = row_shape[0]
        k = len(self.shard_keys)
        if rows % k:
            raise RuntimeError(
                f"streaming stager for {self.label!r}: {rows} rows do not "
                f"divide over {k} shards, so the all-zero pad expert cannot be "
                "built shard-wise"
            )
        return (rows // k,) + tuple(row_shape[1:])


# --- #268: quant-path fail-fast for the expert-offload installer -----------
# Ascend GGUF-MoE (GGUFMoEAscendMethod) and MoeWNA16 (MoeWNA16Method) have no
# load-time offload half: unlike fp8 / GPTQ-Marlin / AWQ-Marlin, their
# per-expert tensors either aren't materialized yet at install time
# (GGUFUninitializedParameter only takes real shape in the loader postprocess
# step, #123) or use a quant layout the offload cache's tensor-slicing/LRU
# fetch was never validated against. Installing the cache on one of these
# quant methods anyway would run EXPERT_TENSOR_ATTRS slicing over a parameter
# that is either not a real tensor yet (crash) or real but semantically
# unsupported (silently wrong per-expert weights, not a crash) -- undefined
# behavior either way, so this must hard-abort before install(), not fall
# back to the try/except's silent per-layer degrade.
#
# #323b: NVFP4 MoE was the named residual risk of #268 and it materialized.
# The guard is an EXCLUSION list, so every quant method not named here passes
# by default -- which is right for a family whose members share one tensor
# layout, and wrong for a genuinely new one. NVFP4 MoE is a genuinely new one:
# EXPERT_TENSOR_ATTRS below lists none of its per-expert tensors
# (w13_weight_packed / w13_weight_scale_2 / w13_blockscale_swizzled /
# w13_alphas / w13_input_scale_quant ...), and no NVFP4 MoE method calls
# presplit_expert_offload_after_repack (fp8.py, gptq_moe.py and awq_moe.py do).
# So the installer would stage a strict subset of the tensors the kernel reads
# and run with per-expert weights paired against another expert's scales:
# silently wrong output, not a crash. Named here until an NVFP4 offload half
# actually exists.
#
# #123-GGUF: ``GGUFMoEMethod`` moved out of the unconditional set into
# _OFFLOAD_CONDITIONAL_QUANT_METHOD_NAMES below. It is admitted ONLY on a layer
# that actually carries the materialization-time staging marker
# (``_moe_offload_gguf_staged``), which the GGUF half sets after it has staged
# both expert tensors into the two tiers. Every GGUF path the half does not
# cover -- a ggml type with no MoE kernel (MXFP4 type 39 among them), the
# dense-linear-only Ascend method, a layer whose expert set is too small to
# split -- leaves the marker unset and is refused exactly as before. The guard
# therefore still fails fast rather than downgrading: what changed is that
# there is now a covered case, not that the refusal got softer.
_OFFLOAD_UNSUPPORTED_QUANT_METHOD_NAMES = (
    "GGUFMoEAscendMethod",
    "MoeWNA16Method",
    # NVFP4 MoE (#323b) -- ModelOpt serialized, ModelOpt online-converted, and
    # the compressed-tensors scheme.
    "ModelOptNvFp4FusedMoEMethod",
    "ModelOptNvFp4OnlineFusedMoEMethod",
    "CompressedTensorsW4A4Nvfp4MoE",
)

#: Quant methods with a load-time offload half that only covers PART of the
#: paths the method can take. Value = name of the layer attribute the half sets
#: once it has actually staged this layer; absent/False => refuse.
_OFFLOAD_CONDITIONAL_QUANT_METHOD_NAMES = {
    "GGUFMoEMethod": "_moe_offload_gguf_staged",
}


def assert_expert_offload_quant_supported(
    quant_method, layer_id=None, scheme=None, layer=None
) -> None:
    """Fail-fast guard (#268/#323b) for the MoE expert-offload installer.

    Call this BEFORE constructing a ``MoEExpertOffloadCache`` for a layer.
    Raises ``RuntimeError`` if the layer's quant path has no load-time offload
    half (GGUF-MoE, MoeWNA16, NVFP4 MoE). No-op for every supported quant
    method (fp8, GPTQ-Marlin, AWQ-Marlin, unquantized) -- those are matched by
    NOT being in the unsupported set, so a new supported quant method never
    needs to be added here.

    ``scheme`` is the compressed-tensors MoE scheme when there is one
    (``layer.scheme``). It has to be checked separately because a
    compressed-tensors layer's ``quant_method`` is always the same delegating
    ``CompressedTensorsFusedMoEMethod`` wrapper -- the class that decides the
    tensor layout is the scheme behind it, so checking only the wrapper would
    either miss NVFP4 or deny every compressed-tensors checkpoint.

    ``layer`` is the FusedMoE layer about to be wrapped. It is what makes the
    CONDITIONAL verdict possible (#123-GGUF): ``GGUFMoEMethod`` passes only on
    a layer its half has actually staged, so an uncovered GGUF path is refused
    with the same hard error as before instead of installing a cache over an
    unstaged (or MXFP4-typed) expert parameter.

    Matched by class name (not isinstance) to avoid importing
    ``sglang.srt.layers.quantization.gguf`` / ``.moe_wna16`` /
    ``.modelopt_quant`` from this module at call sites that must stay
    import-light (this file is imported from the hot FusedMoE construction
    path).
    """
    for candidate in (quant_method, scheme):
        if candidate is None:
            continue
        name = type(candidate).__name__
        marker = _OFFLOAD_CONDITIONAL_QUANT_METHOD_NAMES.get(name)
        if marker is not None:
            if getattr(layer, marker, False):
                continue  # the half staged this layer -> covered
            reason = (
                f"{name!r} has a load-time offload half (#123-GGUF), but it "
                f"did not stage this layer: the materialization-time staging "
                f"marker {marker!r} is absent. That happens when the ggml "
                f"quantization type has no GGUF MoE kernel (MXFP4 / type 39 "
                f"and every other type outside MMVQ_QUANT_TYPES | "
                f"MMQ_QUANT_TYPES), when the layer's expert parameters were "
                f"already materialized by another path, or when the expert "
                f"count is too small to split at this fraction. Installing "
                f"the cache anyway would slice a parameter the half never "
                f"tiered."
            )
        elif name in _OFFLOAD_UNSUPPORTED_QUANT_METHOD_NAMES:
            reason = (
                "GGUF-MoE (Ascend), MoeWNA16 and NVFP4 MoE have no load-time "
                "offload half: the Ascend GGUF MoE method materializes and "
                "pre-dequantizes on its own path (#123 covers the CUDA method "
                "only), MoeWNA16's per-expert tensor layout was never "
                "validated against the offload cache's slice/fetch path, and "
                "the NVFP4 MoE layouts (#323b) are absent from "
                "EXPERT_TENSOR_ATTRS and from "
                "presplit_expert_offload_after_repack, so only part of each "
                "expert would be staged -- installing the cache here would "
                "either crash on an uninitialized parameter or silently run "
                "with undefined per-expert weight contents."
            )
        else:
            continue
        layer_tag = f" (layer_id={layer_id})" if layer_id is not None else ""
        raise RuntimeError(
            f"MoE expert-offload (SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0) "
            f"is not supported for quant method {name!r}{layer_tag}. "
            f"{reason} Supported quant paths for "
            "--moe-resident-expert-fraction < 1.0: fp8, GPTQ (Marlin), AWQ "
            "(Marlin), GGUF-MoE (CUDA, ggml types with a MoE kernel). Leave "
            "--moe-resident-expert-fraction at 1.0 (or unset) for this "
            "checkpoint's quant type, or use a supported quant path."
        )


# ===========================================================================
# Stage-3: CUDA-graph-capturable routing math (host-sync-free).
#
# The eager offload path (run_waves) does per layer per step: topk_ids.tolist()
# (D->H sync), Python set/sort/dict planning, and a per-(attr,expert) Python
# copy loop -- none capturable. The functions below replace the single-wave
# fast path with pure, fixed-shape tensor algebra over the 256-expert axis that
# reproduces ExpertResidencyPlanner.resolve() + _build_lut + _remap BIT-FOR-BIT
# (proven in tests/moe_offload/test_capturable_planner.py), plus the fetch
# source indices for a single captured gather. All ops are scatter / cumsum /
# gather over fixed axes (E, C) => no host sync, CUDA-graph capturable. They are
# written to run identically on CPU tensors (for the unit test) and CUDA.
# ===========================================================================


def build_capturable_luts(
    num_local_experts: int,
    resident_count: int,
    resident_slot: Optional[Dict[int, int]],
    spill_pool_index: Optional[Dict[int, int]],
    device="cpu",
):
    """Build the three frozen device-constant LUTs (int32[E]) the capturable
    path gathers from. Pure derivation from the (already-frozen) residency maps.

    resident_slot / spill_pool_index None => the static [0,R) layout
    (resident e<R at slot==e; spill e>=R at pool row e-R). Otherwise the frozen
    hot-set maps (from _freeze_hotset). Returns:
      resident_slot_lut : int32[E]  slot in [0,R) for resident e, else -1
      is_spill          : bool[E]   (resident_slot_lut < 0)
      spill_pool_row_lut: int32[E]  pool row in [0,E-R) for spill e, else -1
    """
    import torch

    E, R = int(num_local_experts), int(resident_count)
    resident_slot_lut = torch.full((E,), -1, dtype=torch.int32)
    spill_pool_row_lut = torch.full((E,), -1, dtype=torch.int32)
    if resident_slot is None:
        # Static [0,R): resident ids are exactly [0,R) at slot==id.
        idx = torch.arange(R, dtype=torch.int32)
        resident_slot_lut[:R] = idx
        spill_pool_row_lut[R:E] = torch.arange(E - R, dtype=torch.int32)
    else:
        for e, s in resident_slot.items():
            resident_slot_lut[int(e)] = int(s)
        for e, r in spill_pool_index.items():
            spill_pool_row_lut[int(e)] = int(r)
    is_spill = resident_slot_lut < 0
    return (
        resident_slot_lut.to(device),
        is_spill.to(device),
        spill_pool_row_lut.to(device),
    )


def refuse_capturable_cold_tier(num_experts: int) -> None:
    """#394 slice 2 graph seam: BOOT-PENDING, and say exactly what is missing.

    Nothing here is blocked in principle, and the corrected canon says so: a
    CUDA graph pins ADDRESSES, not CONTENTS, so a captured ``index_select``
    over a stable device-addressable source replays correctly after the host
    bytes change -- that is already how the local capturable path works
    (:func:`device_view_of_pinned`, verified on this rig).

    What is unverified is one link in the chain. The local pool is a torch
    ``pin_memory()`` allocation, so its UVA device pointer equals its host
    pointer and ``torch.as_tensor`` aliases it. A PEER's cold row lives in a
    ``mmap`` that this process page-locked with ``cudaHostRegister`` instead;
    the device address for such a range is obtained from
    ``cudaHostGetDevicePointer``, and whether ``is_pinned()``/``as_tensor``
    reproduce the aliasing for it has NOT been exercised on hardware. Capturing
    a graph over an address that has not been verified to alias is precisely
    the failure mode that cannot be detected after the fact: the graph replays
    happily and reads whatever is at that address.

    So the eager path is complete and the capturable installer refuses, until a
    card window proves the pointer. ``SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE=1`` is
    that window's switch, not a performance option.
    """
    import logging

    from sglang.srt.environ import envs

    if envs.SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE.get():
        logging.getLogger(__name__).warning(
            "MoE cold tier (#394): capturing a decode graph over PEER-owned "
            "cold rows (%d experts). The UVA device pointer for a "
            "cudaHostRegister'd mapping is BOOT-PENDING -- verify the gathered "
            "rows against the eager path before trusting any output.",
            num_experts,
        )
        return
    raise RuntimeError(
        "SGLANG_MOE_OFFLOAD_CUDA_GRAPH cannot yet be combined with the shared "
        "cold tier (SGLANG_MOE_COLD_TIER_SHM, #394 slice 2): the capturable "
        "scratch gather needs a UVA device pointer for the PEER segment's "
        "cudaHostRegister'd mapping, and that pointer has not been verified on "
        "hardware. This is an implementation gap, not a limit -- graphs pin "
        "addresses, not contents. Run eager (--disable-cuda-graph), or set "
        "SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE=1 in a card window to develop it."
    )


class _PinnedDeviceViewHolder:
    """Minimal ``__cuda_array_interface__`` producer used to alias PINNED host
    memory as a CUDA tensor (UVA zero-copy). Under CUDA UVA every page-locked
    allocation (torch ``pin_memory``) is device-addressable at the SAME virtual
    address, so a CUDA gather kernel can read it directly across PCIe. Verified
    empirically on this rig (torch 2.11/cu130): ``torch.as_tensor`` aliases the
    pointer without copying, and a graph-captured ``index_select`` sourcing the
    view honors post-capture host-content changes bit-identically."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


def device_view_of_pinned(pinned):  # pragma: no cover - requires CUDA
    """Return a CUDA tensor aliasing ``pinned`` (no copy; UVA zero-copy view).

    The view has the same shape/dtype and a stable device pointer (== the host
    pointer), so it can be baked into a CUDA graph as the SOURCE of the scratch
    gather (design §4). Caller must keep ``pinned`` alive for the view's life.
    """
    import torch

    if not pinned.is_pinned():
        raise RuntimeError("device_view_of_pinned: tensor is not page-locked")
    if not pinned.is_contiguous():
        raise RuntimeError("device_view_of_pinned: tensor must be contiguous")
    nbytes = pinned.numel() * pinned.element_size()
    holder = _PinnedDeviceViewHolder(pinned.data_ptr(), nbytes)
    dev_bytes = torch.as_tensor(holder, device="cuda")
    if dev_bytes.data_ptr() != pinned.data_ptr():
        raise RuntimeError(
            "device_view_of_pinned: torch.as_tensor copied instead of aliasing "
            "(UVA zero-copy unavailable?); cannot build a capturable spill pool"
        )
    return dev_bytes.view(pinned.dtype).view(pinned.shape)


def prepare_capturable_remap(
    topk_ids,
    resident_slot_lut,
    is_spill,
    spill_pool_row_lut,
    resident_count: int,
    scratch: int,
):
    """Fixed-shape, host-sync-free reproduction of resolve()+_build_lut+_remap
    for the single-wave case. Returns (remapped_topk_ids, src_row, num_spill):

      remapped_topk_ids : like topk_ids; global expert id -> resident/scratch
                          slot (-1 padding preserved). Bit-identical to the
                          eager _remap output.
      src_row           : int32[C]; src_row[j] = pinned-pool row to gather into
                          scratch slot j (global slot R+j). Unused slots => 0
                          (harmless: no topk_id maps there).
      num_spill         : int32 scalar (device); # unique spill experts routed.

    Correctness (see test_capturable_planner.py): the cumsum-rank over the
    ascending expert-id axis reproduces resolve()'s sorted(spill)->R+i loop
    index i exactly, so scratch-slot assignment and remap match the eager path.
    """
    import torch

    E = int(resident_slot_lut.shape[0])
    R, C = int(resident_count), int(scratch)
    device = topk_ids.device
    idsf = topk_ids.reshape(-1)
    valid = idsf >= 0
    clamped = idsf.clamp(min=0).to(torch.long)

    # (1) presence over experts via accumulate-scatter of 1s (padding adds 0).
    presence = torch.zeros(E, dtype=torch.int32, device=device)
    presence.index_put_((clamped,), valid.to(torch.int32), accumulate=True)
    present = presence > 0
    spill_present = present & is_spill  # bool[E]

    # (2) sorted scratch-slot assignment via cumulative rank over ascending id.
    rank = torch.cumsum(spill_present.to(torch.int32), dim=0) - 1  # int32[E]
    num_spill = spill_present.to(torch.int32).sum()

    # (3) fetch source rows: src_row[rank[e]] = pool_row(e) for present spill e.
    #     Non-spill entries are routed to a trash slot C and discarded.
    dst = torch.where(
        spill_present,
        rank.clamp(0, C - 1),
        torch.full_like(rank, C),
    ).to(torch.long)
    src_row_ext = torch.zeros(C + 1, dtype=torch.int32, device=device)
    src_row_ext.index_put_((dst,), spill_pool_row_lut, accumulate=False)
    src_row = src_row_ext[:C].contiguous()

    # (4) global-id -> slot LUT + remap (replaces _build_lut/_remap, no loop).
    slot_of = torch.where(is_spill, R + rank, resident_slot_lut)  # int32[E]
    remapped = torch.where(
        topk_ids >= 0,
        slot_of[topk_ids.clamp(min=0).to(torch.long)].to(topk_ids.dtype),
        topk_ids,
    )
    return remapped, src_row, num_spill


class MoEExpertOffloadCache:
    """Tensor-level wrapper around ExpertResidencyPlanner for a FusedMoE layer.

    Wiring lives here (built during the GPU window). It expects the layer's
    stacked expert tensors (w13_weight/w2_weight [+scales]) and moves the full
    set to a pinned host pool, allocating a resident buffer of n_slots experts.

    NOTE: the tensor path requires CUDA and is exercised in the GPU window; the
    planner/wave bookkeeping carries all correctness-critical logic and is
    tested on CPU now (tests/moe_offload/test_planner.py).
    """

    #: names of the stacked per-expert tensors to pool/fetch (dim 0 == expert).
    EXPERT_TENSOR_ATTRS = (
        # FP8 / triton fused path (M-B original).
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
        # Optional fp8 expert biases (GPT-OSS-style). Expert-major like the
        # weights, and read by the triton runner at the SLOT index that the
        # offload remap produces -- so a full [E] bias next to a [R+C] weight
        # buffer would pair every expert with the wrong bias. Stage them.
        "w13_weight_bias",
        "w2_weight_bias",
        # GPTQ-Int4 Marlin path (Variant-C B2b): the POST-repack marlin tensors.
        # The apply kernel reads these; for GPTQ qzeros is unused (sym) and g_idx
        # is empty (desc_act=False). All are expert-major (dim 0 == num_experts)
        # and per-expert sliceable in the marlin layout.
        "w13_qweight",
        "w2_qweight",
        "w13_scales",
        "w2_scales",
        # AWQ-Int4 Marlin path (same qwen3_5_moe fused_marlin_moe path, used for
        # the small-model cross-fraction proof): AWQ is asymmetric, so the marlin
        # apply ALSO reads the per-expert zero-points -> stage them too. Tensors
        # absent for a given quant method are skipped by the shape check below.
        "w13_qzeros",
        "w2_qzeros",
    )

    def __init__(self, layer, fraction: float):
        self.layer = layer
        self.fraction = fraction
        # E: captured BEFORE install shrinks layer.num_local_experts. A prior
        # load-time presplit stashes the real E on the layer; else read it now.
        presplit = getattr(layer, "_moe_offload_presplit", None)
        self.num_local_experts = int(
            getattr(layer, "_moe_offload_full_experts", None)
            or getattr(layer, "num_local_experts")
        )
        self.resident_count = resident_slot_count(self.num_local_experts, fraction)
        self.scratch = scratch_slot_count(self.resident_count)
        self.planner = ExpertResidencyPlanner(
            num_local_experts=self.num_local_experts,
            resident_count=self.resident_count,
            scratch=self.scratch,
        )
        self._pinned: Dict[str, "object"] = {}  # attr -> pinned spill [E-R,...]
        self._resident: Dict[str, "object"] = {}  # attr -> GPU buffer [R+C,...]
        self._stream = None
        self._installed = False

        # --- Stage-1 hot-expert residency ----------------------------------
        # When enabled, per-expert routing counts are accumulated over the first
        # `_hot_calib_steps` forwards; then the R hottest experts are frozen as
        # the resident set and the buffers are physically rearranged so those
        # experts sit in [0,R) and the rest form the spill pool. `_spill_pool_index`
        # maps a (cold) global expert id -> its row in the pinned spill pool
        # (identity `id-R` in the static/default layout). See _freeze_hotset.
        from collections import Counter as _Counter

        from sglang.srt.environ import envs

        self._hot_enabled = bool(envs.SGLANG_MOE_HOT_RESIDENCY.get())
        self._hot_calib_steps = max(1, int(envs.SGLANG_MOE_HOT_CALIB_STEPS.get()))
        self._hot_counts = _Counter()
        self._hot_seen = 0
        self._hot_frozen = False
        self._spill_pool_index: Optional[Dict[int, int]] = None  # None => id-R

        # --- #123-GGUF: residency decided at LOAD time -----------------------
        # A load-time stager that could not use the plain [0,R) layout (the GGUF
        # uneven-TP shard must keep its zero-padding expert resident) publishes
        # the layout it actually built. Adopt it verbatim: the buffers on the
        # layer are ALREADY arranged that way, so this is a map install, not a
        # rearrange. Marked frozen so live hot calibration never permutes
        # buffers whose physical layout the stager chose.
        layout = getattr(layer, "_moe_offload_frozen_layout", None)
        if layout is not None:
            resident_ids, spill_ids = layout
            if len(resident_ids) != self.resident_count:
                raise RuntimeError(
                    f"load-time residency layout has {len(resident_ids)} "
                    f"resident experts but this cache computed "
                    f"{self.resident_count} at fraction {fraction} over "
                    f"{self.num_local_experts} experts -- the stager and the "
                    f"installer disagree (did the fraction change between "
                    f"load and install?)"
                )
            self.planner.resident_ids = frozenset(int(e) for e in resident_ids)
            self.planner.resident_slot = {int(e): i for i, e in enumerate(resident_ids)}
            self._spill_pool_index = {int(e): j for j, e in enumerate(spill_ids)}
            self._hot_frozen = True

        # #394: cold experts a peer's host tier owns. Adopted as a planner
        # guard, not as state: without a shared tier this rank simply has no
        # row for them, and the named error is worth far more than the KeyError
        # it replaces.
        delegated = getattr(layer, "_moe_offload_delegated_experts", None)
        if delegated:
            self.planner.delegated_ids = frozenset(int(e) for e in delegated)

        # #394 slice 2: turn that guard into a fetch route when the shared cold
        # tier is on. ``resolver_for_layer`` returns None on every launch that
        # did not ask for the tier, so the default path takes one attribute
        # read and keeps the refusal above.
        self._cold_tier = None
        self._remote_ids: frozenset = frozenset()
        if delegated:
            from sglang.srt.layers.moe.cold_tier_fetch import resolver_for_layer

            resolver = resolver_for_layer(layer)
            if resolver is not None:
                self._cold_tier = resolver
                self._remote_ids = resolver.remote_ids
                self.planner.delegated_reachable = True
                missing = self.planner.delegated_ids - self._remote_ids
                if missing:
                    raise RuntimeError(
                        f"the staging plan delegated experts {sorted(missing)} "
                        f"but the cold-tier assignment gives them no remote "
                        f"owner. The plan and the assignment were built from "
                        f"different cold pools, and fetching under that "
                        f"disagreement is how a rank reads a plausible wrong "
                        f"row (#394)."
                    )

        # --- #254 prefill wave order ---------------------------------------
        # "token" (default) = disjoint token subsets, every wave re-fetches its
        # tokens' spill experts. "expert" = disjoint spill-expert groups, each
        # spill expert fetched once per forward; byte-identical via the fixed
        # k-order combine in _run_waves_expert_major.
        self._wave_order = resolve_wave_order(envs.SGLANG_MOE_OFFLOAD_WAVE_ORDER.get())

        # --- Stage-3 CUDA-graph-capturable path ----------------------------
        # Built by install_capturable_buffers() (after install(), and after any
        # freeze_from_source() rearrange): frozen device LUTs + UVA device
        # views of the pinned spill pool + stable scratch dest views.
        self._graph_mode = bool(envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH.get())
        if self._graph_mode and self._hot_enabled:
            # §5: live calibration cannot be frozen before graph capture.
            raise RuntimeError(
                "SGLANG_MOE_HOT_RESIDENCY (live hot calibration) cannot be "
                "combined with SGLANG_MOE_OFFLOAD_CUDA_GRAPH: the residency "
                "layout must be frozen BEFORE graph capture. Supply "
                "SGLANG_MOE_HOTSET_FILE or use static residency."
            )

        # --- #390 router / residency instrument -----------------------------
        # Opt-in (SGLANG_EXPERT_STATS=1). Resolved ONCE here: on the default
        # path this stays None and run_waves pays a single `is not None` test.
        from sglang.srt.layers.moe.expert_stats import get_collector, maybe_layer_stats

        self._router_stats = maybe_layer_stats(
            layer_id=getattr(layer, "layer_id", None),
            num_experts=self.num_local_experts,
            resident_count=self.resident_count,
            rank_tag=(
                f"tp{getattr(layer, 'moe_tp_rank', 0)}"
                f"ep{getattr(layer, 'moe_ep_rank', 0)}"
            ),
            graph_mode=self._graph_mode,
        )
        self._stats_collector = None
        if self._router_stats is not None:
            # Hand the planner's own fetch/H2D tally to the dump so the routing
            # histogram and what the offload actually paid for it land in one
            # file instead of two places.
            self._router_stats.residency = self.planner.stats
            # #394: the placement policy this layer was staged under, so the
            # dump names its own A/B arm (see publish_host_shard_on_layer),
            # plus how a delegated expert is REACHED. Without the second field
            # a proportional arm and a proportional arm whose shared tier never
            # attached look identical in the dump, and only one of them is the
            # thing being measured.
            row = getattr(layer, "_moe_offload_host_shard", None)
            if row is not None:
                row = dict(row)
                row["reachability"] = (
                    "shared-cold-tier"
                    if self._cold_tier is not None
                    else ("refused" if delegated else "local-only")
                )
            self._router_stats.host_shard = row
            self._stats_collector = get_collector()

        self._capturable_ready = False
        self._cap_resident_slot_lut = None  # int32[E] device
        self._cap_is_spill = None  # bool[E] device
        self._cap_spill_pool_row_lut = None  # int32[E] device
        self._cap_pool_dev: Dict[str, "object"] = {}  # attr -> UVA view [E-R,...]
        self._cap_scratch_dst: Dict[str, "object"] = {}  # attr -> resident[R:R+C]
        self._cap_view_holders: List["object"] = []  # keep pinned bases alive

    # --- lifecycle (GPU window) --------------------------------------------
    def install(self):
        """Build the [R+C]-slot GPU buffer (fixed resident [0,R) + scratch) and
        the [E-R]-slot pinned host spill pool. Idempotent.

        Source is either (a) a load-time presplit stashed on the layer
        (``_moe_offload_presplit``: attr -> (resident_buf[R+C], spill_pinned)),
        which never let the full [E] stack sit on host -- the RAM-safe path; or
        (b) the layer's full [E] tensor still present (GPU or CPU-pinned), which
        we split here (used by the small-model proof where the full stack fits).
        """
        import torch

        if self._installed or self.planner.fully_resident:
            return
        # The copy stream and the ambient device are the only CUDA-only pieces
        # of install/_fetch. Making them optional lets the whole
        # install -> resolve -> fetch -> remap -> apply chain run on CPU
        # tensors, which is what turns the #123-GGUF round-trip proof into a
        # hermetic desk test instead of a GPU-window claim. Production always
        # has a context, so the CUDA branch is unchanged.
        cuda = torch.cuda.is_available()
        self._stream = torch.cuda.Stream() if cuda else None
        dev = torch.cuda.current_device() if cuda else torch.device("cpu")
        R = self.resident_count
        buf_slots = self.planner.buffer_size  # R + C
        presplit = getattr(self.layer, "_moe_offload_presplit", None)
        # #119 tally. Only the split-here branch releases VRAM *here*; the
        # presplit branch already released it at load time (and tallied there),
        # so counting it again would double-report the reclaim.
        freed_device = 0
        freed_host = 0

        for attr in self.EXPERT_TENSOR_ATTRS:
            if presplit is not None:
                if attr not in presplit:
                    continue
                resident_buf, spill = presplit[attr]  # buf[R+C] GPU, spill host
                self._resident[attr] = resident_buf
                self._pinned[attr] = spill
                setattr(
                    self.layer,
                    attr,
                    torch.nn.Parameter(resident_buf, requires_grad=False),
                )
                continue
            # Split-here path (full [E] tensor present).
            full = getattr(self.layer, attr, None)
            if full is None:
                continue
            full = full.data if hasattr(full, "data") else full
            if full.dim() == 0 or full.shape[0] != self.num_local_experts:
                continue  # not an expert-major tensor
            # GPU buffer [R+C]: [0:R] = fixed resident experts, scratch left as-is.
            buf = torch.empty(
                (buf_slots,) + tuple(full.shape[1:]), dtype=full.dtype, device=dev
            )
            buf[:R].copy_(full[:R])
            self._resident[attr] = buf
            # Pinned host spill pool = experts [R:E].
            spill_src = full[R:].contiguous()
            if spill_src.is_cpu:
                spill = spill_src if spill_src.is_pinned() else spill_src.pin_memory()
            else:
                spill = torch.empty_like(spill_src, device="cpu").pin_memory()
                spill.copy_(spill_src)
            self._pinned[attr] = spill
            setattr(
                self.layer,
                attr,
                (
                    torch.nn.Parameter(buf, requires_grad=False)
                    if isinstance(getattr(self.layer, attr), torch.nn.Parameter)
                    else buf
                ),
            )
            if not full.is_cpu:
                row_bytes = (full.numel() // full.shape[0]) * full.element_size()
                freed_device += expert_offload_released_device_bytes(
                    self.num_local_experts, buf_slots, row_bytes
                )
            freed_host += spill.numel() * spill.element_size()

        # The marlin apply reads E = w1.shape[0] = buffer size (R+C). Advertise
        # it so moe_align / the runner size to the buffer; topk_ids arriving at
        # apply() are already slot ids in [0, R+C).
        self._orig_num_local_experts = self.layer.num_local_experts
        self.layer.num_local_experts = buf_slots
        runner_cfg = getattr(self.layer, "moe_runner_config", None)
        if runner_cfg is not None and hasattr(runner_cfg, "num_local_experts"):
            try:
                runner_cfg.num_local_experts = buf_slots
            except Exception:
                pass  # frozen/dataclass runner configs: kernel reads the layer attr
        if presplit is not None:
            # Release the layer's ref to the presplit dict (tensors now owned by
            # self._resident / self._pinned).
            try:
                delattr(self.layer, "_moe_offload_presplit")
            except Exception:
                self.layer._moe_offload_presplit = None
        elif self._resident:
            record_expert_offload_release(freed_device, freed_host, len(self._resident))
        self._installed = True

    # --- fetch / remap helpers (GPU window) --------------------------------
    def _fetch(self, fetch_plan):
        """Async H2D-copy each wave's SPILL experts into their scratch slots,
        then join the copy stream before compute reads them. ``fetch_plan`` is
        (spill_expert_id, scratch_slot); the spill pool is indexed by
        (expert_id - resident_count).

        #394 slice 2: an expert in ``self._remote_ids`` is not in this rank's
        pool at all -- its row is a zero-copy view of a PEER's shared segment,
        resolved through ``self._cold_tier``. The copy itself is the same
        ``copy_`` over the same link; only the source address differs, which is
        the whole design (the storage moved, the transport did not)."""
        import torch

        if not fetch_plan:
            return
        R = self.resident_count
        pool_index = self._spill_pool_index  # None => static layout (id - R)
        remote = self._cold_tier
        moved = 0
        remote_moved = 0

        # Write-after-read: the scratch slots this fetch overwrites are still
        # being READ by the previous wave's grouped-GEMM, which was enqueued on
        # the compute stream. Without this the copy stream can overtake that
        # GEMM and swap an expert's weights out from under it -- silently wrong
        # output, and timing-dependent, so it only shows up once waves get long
        # enough for the copies to win the race (expert-major waves do; the
        # short token-major waves happened not to). The join below covers the
        # other direction (compute must not read before the copy lands).
        def _copies():
            nonlocal moved, remote_moved
            # With no shared tier the iteration set is exactly what it always
            # was. With one, a rank can own ZERO local cold rows for a tensor
            # (a lopsided ratio is legal), so the attr set has to come from the
            # resident buffers -- the pinned pool may not be there at all.
            attrs = self._pinned if remote is None else self._resident
            for attr in attrs:
                spill = self._pinned.get(attr)
                dst = self._resident[attr]
                per_expert = dst[0].numel() * dst.element_size()
                for expert_id, slot in fetch_plan:
                    if remote is not None and expert_id in self._remote_ids:
                        dst[slot].copy_(remote.row(attr, expert_id), non_blocking=True)
                        moved += per_expert
                        remote_moved += per_expert
                        continue
                    row = (
                        pool_index[expert_id]
                        if pool_index is not None
                        else expert_id - R
                    )
                    dst[slot].copy_(spill[row], non_blocking=True)
                    moved += per_expert

        if self._stream is None:
            # No CUDA context (desk test): same copies, same order, no streams.
            _copies()
        else:
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                _copies()
            torch.cuda.current_stream().wait_stream(self._stream)
        self.planner.stats.h2d_bytes += moved
        self.planner.stats.remote_h2d_bytes += remote_moved

    def _build_lut(self, slot_of_needed, dtype, device):
        """Global expert id -> slot LUT for one wave; -1 for every id the wave
        does not need.

        Written as ONE ``index_copy_`` over host-built index/value vectors. The
        previous per-entry ``lut[e] = s`` issued one device scalar store per
        needed expert, and a scalar store from a Python int blocks the host
        until the stream drains -- so every wave stalled the host behind its own
        queued scratch fetches. A 2048-token prefill chunk runs ~886 waves per
        layer x 40 layers, ~18 needed experts each: ~640k blocking stores per
        chunk. Measured on one RTX 3080 with the real per-expert shapes
        (Qwen3.6-35B-A3B-FP8, TP=3): 5.5 s/chunk of pure LUT time, and 5.8
        s/chunk even in the fetch-bound regime because the stalls serialize the
        host against the H2D copies.

        Bit-identical to the per-entry build (tests/moe_offload/test_build_lut.py),
        and CPU-runnable, so the equivalence is proven without CUDA.
        """
        import numpy as np
        import torch

        lut = torch.full((self.num_local_experts,), -1, dtype=dtype, device=device)
        n = len(slot_of_needed)
        if n == 0:
            return lut
        idx = torch.from_numpy(np.fromiter(slot_of_needed.keys(), np.int64, n))
        val = torch.from_numpy(np.fromiter(slot_of_needed.values(), np.int64, n)).to(
            dtype
        )
        lut.index_copy_(
            0,
            idx.to(device, non_blocking=True),
            val.to(device, non_blocking=True),
        )
        return lut

    @staticmethod
    def _remap(topk_ids, lut):
        import torch

        # -1 padding stays -1; every real id maps to its resident slot.
        return torch.where(topk_ids >= 0, lut[topk_ids.clamp(min=0)], topk_ids)

    # --- per-forward (GPU window) ------------------------------------------
    def prepare(self, topk_ids):  # pragma: no cover - requires CUDA
        """Single-wave remap for a forward whose unique experts fit in n_slots
        (e.g. decode). Resolves residency, async-fetches misses, returns a
        remapped topk_ids (global expert id -> resident slot; -1 stays -1).

        Raises if the forward needs more than n_slots unique experts -- callers
        that can hit prefill overflow must use ``run_waves`` instead.
        """
        import torch

        if self.planner.fully_resident:
            return topk_ids
        needed = torch.unique(topk_ids[topk_ids >= 0]).tolist()
        slot_of_needed, fetch_plan = self.planner.resolve(needed)
        self._fetch(fetch_plan)
        lut = self._build_lut(slot_of_needed, topk_ids.dtype, topk_ids.device)
        return self._remap(topk_ids, lut)

    # --- Stage-3 capturable path (GPU window) ------------------------------
    def install_capturable_buffers(self):  # pragma: no cover - requires CUDA
        """Build the frozen device LUTs (§3.1), the UVA device views of the
        pinned spill pool, and the stable scratch destination views (§4).

        MUST run after install() and after any freeze_from_source() rearrange
        (the LUTs and pool views snapshot the FROZEN layout), and before graph
        capture. In practice it runs on the first (warmup) forward, which is
        eager and precedes DecodeCudaGraphRunner's stream capture (§5).
        Idempotent.
        """
        import torch

        if self._capturable_ready:
            return
        if not self._installed:
            raise RuntimeError("install_capturable_buffers() requires install() first")
        R, C = self.resident_count, self.scratch
        if self.planner.buffer_size != R + C:
            raise RuntimeError(
                f"capturable offload requires buffer_size == R+C "
                f"({self.planner.buffer_size} != {R}+{C}); the scratch region "
                f"was capped by num_local_experts -- lower "
                f"SGLANG_MOE_SCRATCH_SLOTS or the resident fraction."
            )
        if (self.planner.resident_slot is None) != (self._spill_pool_index is None):
            raise RuntimeError(
                "inconsistent frozen residency maps (resident_slot vs "
                "spill_pool_index); freeze must install both or neither"
            )
        if self._cold_tier is not None:
            refuse_capturable_cold_tier(self.num_local_experts)
        device = torch.device("cuda", torch.cuda.current_device())
        (
            self._cap_resident_slot_lut,
            self._cap_is_spill,
            self._cap_spill_pool_row_lut,
        ) = build_capturable_luts(
            self.num_local_experts,
            R,
            self.planner.resident_slot,
            self._spill_pool_index,
            device=device,
        )
        for attr, pinned in self._pinned.items():
            self._cap_pool_dev[attr] = device_view_of_pinned(pinned)
            self._cap_view_holders.append(pinned)
            # Stable scratch sub-view of the resident buffer: fixed address,
            # contiguous (slice of dim 0 of a contiguous [R+C,...] tensor).
            self._cap_scratch_dst[attr] = self._resident[attr][R : R + C]
        self._capturable_ready = True

    def _issue_fetch_capturable(self, src_row):  # pragma: no cover - CUDA
        """Captured scratch fetch (§4): one gather KERNEL per expert-tensor
        attr, sourcing the UVA device view of the pinned pool and writing the
        stable scratch region [R:R+C] in place. Stable in/out pointers; the
        data-dependent part is only the CONTENT of ``src_row`` -> capturable.
        Runs on the current stream, so program order guarantees the routed
        apply (issued later on the same stream) reads a complete scratch (§7 R1).
        """
        import torch

        for attr, pool_dev in self._cap_pool_dev.items():
            torch.index_select(pool_dev, 0, src_row, out=self._cap_scratch_dst[attr])

    def prepare_capturable(self, topk_ids):  # pragma: no cover - requires CUDA
        """Single-wave, host-sync-free prepare for the captured decode path:
        on-device remap (bit-identical to resolve()+_build_lut+_remap, proven
        on CPU) + the captured scratch gather. Returns remapped topk_ids.

        Caller must guarantee the §2 invariant (worst-case unique spill <= C,
        i.e. topk_ids.numel() <= C); enforced loudly in layer.py.
        """
        if not self._capturable_ready:
            raise RuntimeError(
                "prepare_capturable() before install_capturable_buffers()"
            )
        remapped, src_row, _num_spill = prepare_capturable_remap(
            topk_ids,
            self._cap_resident_slot_lut,
            self._cap_is_spill,
            self._cap_spill_pool_row_lut,
            self.resident_count,
            self.scratch,
        )
        self._issue_fetch_capturable(src_row)
        return remapped

    def freeze_from_source(self):  # pragma: no cover - requires CUDA
        """§5 freeze-before-capture: load this layer's frozen hot set from
        SGLANG_MOE_HOTSET_FILE and drive the existing _freeze_hotset physical
        rearrange from it (instead of live calibration counts). No file =>
        static [0,R) fallback (no-op). Runs before install_capturable_buffers.

        File format (JSON): ``{"<layer_id>": [expert_id, ...], ...}`` with the
        per-layer list ordered hottest-first (>= R entries; the first R are
        taken). Produced offline from the M-C routing trace.
        """
        import logging

        from sglang.srt.environ import envs

        path = envs.SGLANG_MOE_HOTSET_FILE.get()
        if not path:
            return  # static [0,R) residency (F3)
        if self._hot_frozen:
            return
        data = _load_hotset_file(path)
        layer_id = getattr(self.layer, "layer_id", None)
        key = str(layer_id)
        if key not in data:
            raise RuntimeError(
                f"SGLANG_MOE_HOTSET_FILE {path!r} has no entry for layer "
                f"{key!r} (keys: {sorted(data)[:8]}...)"
            )
        R, E = self.resident_count, self.num_local_experts
        ids = [int(e) for e in data[key]]
        if any(e < 0 or e >= E for e in ids):
            raise RuntimeError(
                f"SGLANG_MOE_HOTSET_FILE layer {key}: expert id out of [0,{E})"
            )
        hot = sorted(set(ids[:R]))
        if len(hot) != R:
            raise RuntimeError(
                f"SGLANG_MOE_HOTSET_FILE layer {key}: need {R} unique hot "
                f"experts, got {len(hot)} from the first {R} listed"
            )
        self._apply_hotset_freeze(hot)
        logging.getLogger(__name__).info(
            "MoE hot-residency FROZEN FROM FILE on layer %s: R=%d (source %s)",
            key,
            R,
            path,
        )

    def run_waves(self, dispatch_output, apply_fn):
        """Run the grouped-GEMM for one forward, wave-splitting when the forward
        needs more unique experts than there are resident slots.

        ``apply_fn(sub_dispatch_output) -> CombineInput`` runs the unmodified
        MoE math (``quant_method.apply``) over the resident buffer. We call it
        once per wave over that wave's token rows and scatter the results back.

        Returns a CombineInput whose hidden_states is the full [T, H] output,
        byte-identical to the no-offload path (see module docstring).
        """
        import torch

        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids

        if self.planner.fully_resident:
            return apply_fn(dispatch_output)

        ids_list = topk_ids.tolist()  # [T][k]  (device->host sync; eager only)

        # #390: fold this forward's routing decision into the per-layer expert
        # histogram and the hit/miss tally against the resident set. This is the
        # fetch-decision point -- residency is already known here and the ids
        # are already on the host, so the instrument adds no device sync. Taken
        # BEFORE any hot-set freeze below, so an activation is attributed to the
        # residency that was actually in force when it was routed.
        if self._router_stats is not None:
            self._router_stats.record(
                ids_list, self.planner.resident_ids, self.resident_count
            )
            self._stats_collector.maybe_dump_periodic()

        # Stage-1 hot residency: accumulate routing counts, then freeze the R
        # hottest experts (physical rearrange) once calibration is complete. Done
        # BEFORE this forward's resolve/fetch/apply so the triggering forward's
        # own output already uses the frozen set (no intra-run drift).
        if self._hot_enabled and not self._hot_frozen:
            for row in ids_list:
                for e in row:
                    if e >= 0:
                        self._hot_counts[e] += 1
            self._hot_seen += 1
            if self._hot_seen >= self._hot_calib_steps:
                self._freeze_hotset()

        if self._wave_order == "expert":
            # #254: split over SPILL EXPERTS instead of tokens. The single-wave
            # case is bit-for-bit the token-major fast path below.
            resident_used, spill_waves = plan_expert_waves(
                ids_list, self.resident_count, self.scratch, self.planner.resident_ids
            )
            if len(spill_waves) > 1:
                self.planner.stats.overflow_forwards += 1
                return self._run_waves_expert_major(
                    dispatch_output, apply_fn, ids_list, resident_used, spill_waves
                )
            return self._run_single_wave(dispatch_output, apply_fn, ids_list)

        waves = plan_token_waves(
            ids_list, self.resident_count, self.scratch, self.planner.resident_ids
        )

        # Fast path: the whole forward fits in one wave (typical decode). Remap
        # the full batch and run a single apply -- no token slicing overhead.
        if len(waves) == 1:
            return self._run_single_wave(dispatch_output, apply_fn, ids_list)

        # Multi-wave (prefill overflow): process disjoint token subsets.
        self.planner.stats.overflow_forwards += 1
        h2d_before = self.planner.stats.h2d_bytes
        hidden = dispatch_output.hidden_states
        scale = dispatch_output.hidden_states_scale
        topk_weights = topk_output.topk_weights
        router_logits = getattr(topk_output, "router_logits", None)
        T = hidden.shape[0]
        out_full = torch.empty_like(hidden)
        combine_out = None

        for rows in waves:
            rows_t = torch.tensor(rows, device=topk_ids.device, dtype=torch.long)
            needed = sorted({e for r in rows for e in ids_list[r] if e >= 0})
            slot_of_needed, fetch_plan = self.planner.resolve(needed)
            self._fetch(fetch_plan)
            lut = self._build_lut(slot_of_needed, topk_ids.dtype, topk_ids.device)

            tid_w = self._remap(topk_ids.index_select(0, rows_t), lut)
            tw_w = topk_weights.index_select(0, rows_t)
            hs_w = hidden.index_select(0, rows_t)
            sc_w = (
                scale.index_select(0, rows_t)
                if isinstance(scale, torch.Tensor) and scale.shape[0] == T
                else scale
            )
            rl_w = (
                router_logits.index_select(0, rows_t)
                if isinstance(router_logits, torch.Tensor)
                and router_logits.dim() >= 1
                and router_logits.shape[0] == T
                else router_logits
            )

            sub_topk = topk_output._replace(
                topk_weights=tw_w, topk_ids=tid_w, router_logits=rl_w
            )
            sub = dispatch_output._replace(
                hidden_states=hs_w,
                hidden_states_scale=sc_w,
                topk_output=sub_topk,
            )
            combine_out = apply_fn(sub)
            out_full.index_copy_(
                0, rows_t, combine_out.hidden_states.to(out_full.dtype)
            )

        # Reuse the last wave's CombineInput type/fields, swapping in full output.
        self._log_wave_h2d("token", len(waves), h2d_before)
        return combine_out._replace(hidden_states=out_full)

    def _log_wave_h2d(self, order, waves, before):  # pragma: no cover - CUDA
        """One line per multi-wave forward with the PCIe volume it cost, so the
        token- vs expert-major difference is readable off the log instead of
        being inferred from the wall clock.

        INFO for ONE representative layer (the first that reports; layer 0 is
        dense in most MoE models, so keying on id 0 logs nothing), DEBUG for the
        rest -- one line per chunk at the default log level, the full per-layer
        picture at DEBUG.
        """
        import logging

        global _H2D_LOG_LAYER

        gib = (self.planner.stats.h2d_bytes - before) / float(1 << 30)
        layer_id = getattr(self.layer, "layer_id", "?")
        if _H2D_LOG_LAYER is None:
            _H2D_LOG_LAYER = layer_id
        logging.getLogger(__name__).log(
            logging.INFO if layer_id == _H2D_LOG_LAYER else logging.DEBUG,
            "MoE offload layer %s: %s-major prefill, %d waves, %.2f GiB H2D",
            layer_id,
            order,
            waves,
            gib,
        )

    def _run_single_wave(self, dispatch_output, apply_fn, ids_list):
        """One apply over the full batch: every routed expert fits the buffer at
        once (typical decode, and any prefill whose spill set fits the scratch).
        Shared by both wave orders -- with a single wave they are the same path."""
        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids
        needed = sorted({e for row in ids_list for e in row if e >= 0})
        slot_of_needed, fetch_plan = self.planner.resolve(needed)
        self._fetch(fetch_plan)
        lut = self._build_lut(slot_of_needed, topk_ids.dtype, topk_ids.device)
        remapped = self._remap(topk_ids, lut)
        sub = dispatch_output._replace(
            topk_output=topk_output._replace(topk_ids=remapped)
        )
        return apply_fn(sub)

    def _run_waves_expert_major(
        self, dispatch_output, apply_fn, ids_list, resident_used, spill_waves
    ):  # pragma: no cover - requires CUDA
        """#254 expert-major prefill: waves are disjoint SPILL-EXPERT groups, so
        every spill expert crosses PCIe exactly ONCE per forward instead of once
        per token-major wave (~62x at C=16 on a 2048-token chunk).

        Byte-identity to the token-major path
        -------------------------------------
        A wave no longer holds a token's complete top-k, so the per-token
        reduction may not happen inside the wave -- accumulating per-wave partial
        sums would re-associate it and lose bit-identity. Instead every wave
        computes its (token, k-slot) contributions as INDEPENDENT rows: each
        routed pair becomes its own pseudo-token with top_k == 1, which makes the
        fused kernel write the weighted contribution straight out (no internal
        reduction), and it is stored at its own k-slot in a [T, top_k, H] buffer.
        The k-slot comes from the routing and is independent of the wave split,
        so the buffer's contents are the same values in the same places for ANY
        split. The top-k reduction then runs ONCE at the end over the full buffer
        via ``combine_topk_partials`` -- the same reduction the unsplit kernel
        applies to its own intermediate_cache3.  Padded (-1) slots are never
        assigned to a wave and stay zero, which is what the kernel writes for
        them. Verified bit-exact against the unsplit apply on bf16 and
        fp8-blockwise up to T=2048/E=64/top_k=8 (tests/moe_offload).

        ``routed_scaling_factor`` is applied by the FINAL reduction, so the
        per-wave applies run with it neutralized to 1.0 (the kernel's top_k == 1
        path does not carry a scaling step of its own).

        Cost: one transient [T, top_k, H] buffer per layer, freed when the
        forward returns.
        """
        import numpy as np
        import torch

        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        hidden = dispatch_output.hidden_states
        scale = dispatch_output.hidden_states_scale
        router_logits = getattr(topk_output, "router_logits", None)
        device = topk_ids.device
        T, K = int(topk_ids.shape[0]), int(topk_ids.shape[1])

        # pair index (t*K + k) -> wave: 0 = the fetch-free resident wave,
        # 1..n = spill groups, -1 = padded slot (contributes an exact zero).
        flat_np = np.asarray(ids_list, dtype=np.int64).reshape(-1)
        wave_lut = np.full(self.num_local_experts, -1, dtype=np.int64)
        for e in resident_used:
            wave_lut[e] = 0
        for w, group in enumerate(spill_waves):
            for e in group:
                wave_lut[e] = w + 1
        wave_of_pair = np.where(flat_np >= 0, wave_lut[np.maximum(flat_np, 0)], -1)

        flat_ids = topk_ids.reshape(-1)
        flat_weights = topk_weights.reshape(-1, 1).contiguous()
        partials = None
        combine_out = None

        h2d_before = self.planner.stats.h2d_bytes
        cfg = self.layer.moe_runner_config
        saved_rsf = cfg.routed_scaling_factor
        try:
            cfg.routed_scaling_factor = 1.0
            for w, needed in enumerate([resident_used] + spill_waves):
                idx_np = np.flatnonzero(wave_of_pair == w)
                if idx_np.size == 0:
                    continue
                slot_of_needed, fetch_plan = self.planner.resolve(needed)
                self._fetch(fetch_plan)
                lut = self._build_lut(slot_of_needed, topk_ids.dtype, device)

                idx = torch.from_numpy(idx_np).to(device, non_blocking=True)
                rows = torch.div(idx, K, rounding_mode="floor")
                tid_w = lut[flat_ids.index_select(0, idx)].unsqueeze(1)
                tw_w = flat_weights.index_select(0, idx)
                hs_w = hidden.index_select(0, rows).contiguous()
                sc_w = (
                    scale.index_select(0, rows)
                    if isinstance(scale, torch.Tensor) and scale.shape[0] == T
                    else scale
                )
                rl_w = (
                    router_logits.index_select(0, rows)
                    if isinstance(router_logits, torch.Tensor)
                    and router_logits.dim() >= 1
                    and router_logits.shape[0] == T
                    else router_logits
                )

                sub_topk = topk_output._replace(
                    topk_weights=tw_w, topk_ids=tid_w, router_logits=rl_w
                )
                sub = dispatch_output._replace(
                    hidden_states=hs_w,
                    hidden_states_scale=sc_w,
                    topk_output=sub_topk,
                )
                combine_out = apply_fn(sub)
                part = combine_out.hidden_states
                if partials is None:
                    partials = torch.zeros(
                        (T * K, part.shape[-1]), dtype=part.dtype, device=device
                    )
                partials.index_copy_(0, idx, part.to(partials.dtype))
        finally:
            cfg.routed_scaling_factor = saved_rsf

        out_full = torch.empty(
            (T, partials.shape[-1]), dtype=partials.dtype, device=device
        )
        combine_topk_partials(partials.view(T, K, -1), out_full, saved_rsf)
        self._log_wave_h2d("expert", len(spill_waves) + 1, h2d_before)
        return combine_out._replace(hidden_states=out_full)

    # --- Stage-1 hot-set freeze (GPU window) -------------------------------
    def _freeze_hotset(self):  # pragma: no cover - requires CUDA
        """Compute the R most-frequently-routed experts (from accumulated
        calibration counts), physically rearrange every expert tensor so those R
        occupy the resident GPU slots [0,R) and the rest form the pinned spill
        pool, install the id->slot / id->pool-row maps on the planner+cache, and
        FREEZE. One-time, deterministic (tie-break by ascending expert id).

        Byte-identity: this only permutes WHICH physical expert lives in WHICH
        slot/pool-row. A token's MoE output depends only on its own routed
        experts' weights and its top-k reduction order (unchanged); each expert's
        per-block GEMM is independent of its slot. Buffer size (R+C) and every
        expert's token set are unchanged, so the marlin moe_align tiling is
        unchanged -> output is bit-identical to the static-[0,R) layout at the
        same fraction. The win is purely a higher resident hit-rate => fewer H2D
        fetches. Frozen after this call => residency never drifts => self-det.
        """
        import logging

        R = self.resident_count
        E = self.num_local_experts
        counts = self._hot_counts

        # Deterministic hot set: highest count first, ties broken by ascending id.
        ranked = sorted(range(E), key=lambda e: (-counts.get(e, 0), e))
        hot = sorted(ranked[:R])
        self._apply_hotset_freeze(hot)
        self._hot_counts = None  # release; never consulted again

        total = sum(counts.values()) or 1
        hot_mass = sum(counts.get(e, 0) for e in hot)
        logging.getLogger(__name__).info(
            "MoE hot-residency FROZEN on layer %s: R=%d hottest experts hold "
            "%.1f%% of routed mass over %d calib forwards (static [0,R) held "
            "%.1f%%); spill pool = %d cold experts.",
            getattr(self.layer, "layer_id", "?"),
            R,
            100.0 * hot_mass / total,
            self._hot_seen,
            100.0 * sum(counts.get(e, 0) for e in range(R)) / total,
            E - R,
        )

    def _apply_hotset_freeze(self, hot):  # pragma: no cover - requires CUDA
        """Physically install ``hot`` (sorted list of R expert ids) as the
        frozen resident set: in-place buffer rearrange + planner/cache map
        install + freeze. Shared by live calibration (_freeze_hotset) and the
        §5 file-driven freeze_from_source."""
        import torch

        R = self.resident_count
        E = self.num_local_experts
        if self._capturable_ready:
            # The LUTs / pool device views snapshot the frozen layout; a
            # rearrange after they were built would silently desync them.
            raise RuntimeError(
                "hot-set freeze after install_capturable_buffers(); the freeze "
                "must happen before the capturable buffers are built"
            )
        hot_set = set(hot)
        cold = [e for e in range(E) if e not in hot_set]  # ascending
        resident_slot = {e: i for i, e in enumerate(hot)}
        spill_pool_index = {e: j for j, e in enumerate(cold)}

        # Rearrange with ZERO extra GPU memory: the co-located 3080 ranks sit at
        # their full mem budget after load, so a transient second GPU buffer would
        # OOM. Instead we snapshot the resident region to host, rebuild the spill
        # pool on host, and overwrite the EXISTING resident buffer in place (H2D
        # into buf[0:R]); no new GPU tensor is allocated. Per-attr host temp is
        # one layer's expert set (~O(100 MB)), freed each iteration.
        for attr in list(self._resident.keys()):
            buf = self._resident[attr]  # [R+C,...]; slot i (i<R) == expert i
            old_spill = self._pinned[attr]  # [E-R,...]; row (e-R) == expert e>=R
            tail = tuple(buf.shape[1:])

            # Host snapshot of the current resident experts [0,R) so overwriting
            # buf[0:R] in place can never corrupt a not-yet-moved source.
            resident_host = buf[:R].to("cpu")

            def _src(e):  # current physical tensor for expert e (STATIC layout)
                return resident_host[e] if e < R else old_spill[e - R]

            # New spill pool (cold experts), pinned for async H2D fetches later.
            new_spill = torch.empty(
                (E - R,) + tail, dtype=old_spill.dtype, device="cpu"
            ).pin_memory()
            for j, e in enumerate(cold):
                new_spill[j].copy_(_src(e))  # host<-host (snapshot or old_spill)

            # Push hot experts into the existing resident slots, in place.
            for i, e in enumerate(hot):
                buf[i].copy_(_src(e))  # GPU<-host (H2D); no new GPU alloc

            self._pinned[attr] = new_spill  # self._resident[attr] stays `buf`
            del resident_host, old_spill

        torch.cuda.synchronize()
        # Install the frozen maps; from here resolve()/_fetch() use the hot set.
        self.planner.resident_ids = frozenset(hot_set)
        self.planner.resident_slot = resident_slot
        self._spill_pool_index = spill_pool_index
        self._hot_frozen = True

    @property
    def stats(self) -> ResidencyStats:
        return self.planner.stats


# Per-process cache for SGLANG_MOE_HOTSET_FILE: every MoE layer freezes from
# the same small JSON, so parse it once.
_HOTSET_FILE_CACHE: Dict[str, dict] = {}


def _load_hotset_file(path: str) -> dict:
    data = _HOTSET_FILE_CACHE.get(path)
    if data is None:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise RuntimeError(
                f"SGLANG_MOE_HOTSET_FILE {path!r}: expected a JSON object "
                f'{{"<layer_id>": [expert ids hottest-first]}}'
            )
        _HOTSET_FILE_CACHE[path] = data
    return data


def repack_door_shards_experts_on_dim0(layer) -> bool:
    """Does this marlin-repack layer hold a DISJOINT slice of the expert set?

    The two ways a FusedMoE layer ends up expert-major-sharded: expert
    parallelism (``moe_ep_size > 1``, with ``expert_map`` remapping foreign
    ids away) and the #82 GGUF expert-dim shard. Anything else is an
    intermediate-dim TP MoE, which holds an essential slice of EVERY expert
    and can therefore delegate none of them.
    """
    if getattr(layer, "_gguf_expert_shard", False):
        return True
    return int(getattr(layer, "moe_ep_size", 1) or 1) > 1


def refuse_cold_shard_at_repack_door(layer) -> None:
    """#421 F8 / #394: the marlin-repack door does not take a cold shard.

    The #394 link-proportional cold-expert policy has two load-time doors.
    ``a2b21c2880`` wired the GGUF one; this one stayed at ``cold_shard=None``
    at every production call site, and the merge message nonetheless claimed
    both halves "take their layout from ONE plan object". The audit recorded
    that as partial wiring. This function is the resolution, and it is a
    refusal rather than a wiring for a reason that was measured, not assumed:

    ``partition_cold_experts`` keeps this rank's share of ITS OWN cold experts
    and drops the rest -- delegating the remainder to a peer's host tier. That
    is sound only if a delegated expert stays REACHABLE from the rank whose
    router asks for it. It does not: booted on the reference rig 2026-08-02
    (V4-Flash UD-IQ3_XXS, TP=3) the GGUF door died on the first forward inside
    ``ExpertResidencyPlanner.resolve`` -- "experts [80, 83, 94] were delegated
    to a peer rank's host tier ... but this rank's router asked for them".
    A delegated expert under a disjoint expert shard is not relocated, it is
    absent.

    The precondition documented for THIS door -- "only legal on a layer that
    shards experts on dim 0 and remaps foreign ids away" -- is exactly the
    disjoint-shard case, i.e. exactly the case in which delegation is unsound.
    So the door is shut in both directions:

    * an intermediate-dim TP MoE cannot delegate at all (nothing is
      expert-major here);
    * an EP / GGUF-expert-shard layer could delegate structurally, but the
      delegated experts would be unreachable.

    Until a reachability mechanism exists (shared pinned host pools a rank can
    DMA out of, or EP-style dispatch to the owner), the honest answer is that
    no production caller passes a cold shard, and one that does gets this
    error instead of a load that dies on the first token.
    ``SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE`` -- the same escape hatch the
    GGUF door carries -- exists so the mechanism can be developed against a
    real boot. It is not a performance option.
    """
    from sglang.srt.environ import envs

    eligible = repack_door_shards_experts_on_dim0(layer)
    if eligible and envs.SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE.get():
        return
    if not eligible:
        why = (
            "this layer is an intermediate-dim TP MoE: it holds an essential "
            "slice of EVERY expert (moe_ep_size=1, no GGUF expert-dim shard), "
            "so there is no whole expert it could delegate"
        )
    else:
        why = (
            "this layer shards experts disjointly, which is exactly when a "
            "delegated expert becomes UNREACHABLE rather than relocated -- "
            "measured on the reference rig 2026-08-02, the router asks for "
            "experts no rank holds and the first forward dies. The #394 slice-2 "
            "shared cold tier (SGLANG_MOE_COLD_TIER_SHM=1) is that missing "
            "reachability mechanism, and it is wired at the GGUF streaming "
            "door -- not here, because this door's own callers are "
            "intermediate-dim TP MoEs. Set "
            "SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE=1 only to develop it "
            "against a real boot"
        )
    raise ValueError(
        "presplit_expert_offload_after_repack(cold_shard=...) is refused "
        f"(#394 / #421 F8): {why}. Every production caller (fp8.py, "
        "gptq_moe.py, awq_moe.py) passes no cold shard, and that is the "
        "intended state, not an oversight."
    )


def presplit_expert_offload_after_repack(
    layer, cold_shard: Optional[ColdShardContext] = None
) -> None:  # pragma: no cover - CUDA
    """Variant-C B2b LOAD-TIME RAM cap: called right after a FusedMoE layer's
    marlin repack (the repacked expert tensors are on GPU, inside
    device_loading_context). Splits each expert-major tensor into a [R+C]-slot
    GPU buffer (fixed resident [0,R) + scratch) plus an [E-R]-slot pinned host
    SPILL pool, stashes them on the layer (``_moe_offload_presplit``), and
    replaces the registered param with a 0-row GPU placeholder so
    device_loading_context's exit copies ~nothing back to host. The full [E,...]
    stack therefore NEVER sits in host RAM -> host peak ~= spill, not the full
    expert set. The eager installer later wires the stash into a
    MoEExpertOffloadCache. No-op unless SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.

    The layout now comes from ``plan_load_time_staging`` -- the same plan the
    #123-GGUF half stages against -- so the two halves cannot drift apart and
    both honour a non-static layout. With no ``cold_shard`` (every production
    caller today: fp8.py, gptq_moe.py, awq_moe.py) the plan is the static
    ``[0,R)`` one and the copies below are the same two slices as before.

    ``cold_shard`` (#394) is REFUSED here by default, and the refusal is a
    measurement rather than caution -- see
    :func:`refuse_cold_shard_at_repack_door` for the reason and the escape
    hatch. #421 F8 recorded this door as the second, unwired half of the #394
    policy; the honest resolution is the named refusal, not a wiring, because
    the door's own precondition (experts sharded on dim 0) is exactly the case
    in which delegation was measured to be unsound.
    """
    import torch

    from sglang.srt.layers.moe.resident_fraction import resident_fraction_for_rank

    if cold_shard is not None:
        refuse_cold_shard_at_repack_door(layer)

    # SIZING: drives plan_load_time_staging below, i.e. the resident set and
    # every buffer booked against this rank's VRAM.
    frac = resident_fraction_for_rank()
    if frac >= 1.0:
        return
    E = getattr(layer, "num_local_experts", None)
    if not E:
        return
    plan = plan_load_time_staging(int(E), fraction=frac, cold_shard=cold_shard)
    if plan is None:
        return
    R = plan.resident_count
    buf_slots = plan.buffer_slots
    static = plan.is_static_layout

    presplit = {}
    freed_device = 0
    freed_host = 0
    for attr in MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS:
        p = getattr(layer, attr, None)
        if p is None:
            continue
        t = p.data if hasattr(p, "data") else p
        if t.dim() == 0 or t.shape[0] != int(E):
            continue  # not an expert-major tensor
        # [R+C] GPU buffer: [0:R] fixed resident; scratch [R:R+C] left uninit.
        buf = torch.empty(
            (buf_slots,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device
        )
        # Spill -> pinned host; the GPU [E] stack is then freed. The static
        # plan is two contiguous slices, exactly as before; a #394 plan gathers
        # the rows the plan names (whole experts on dim 0 either way).
        spill = torch.empty(
            (len(plan.spill_ids),) + tuple(t.shape[1:]), dtype=t.dtype, device="cpu"
        ).pin_memory()
        if static:
            buf[:R].copy_(t[:R])
            spill.copy_(t[R:])
        else:
            buf[:R].copy_(
                t.index_select(
                    0,
                    torch.as_tensor(
                        plan.resident_ids, dtype=torch.long, device=t.device
                    ),
                )
            )
            if plan.spill_ids:
                spill.copy_(
                    t.index_select(
                        0,
                        torch.as_tensor(
                            plan.spill_ids, dtype=torch.long, device=t.device
                        ),
                    )
                )
        presplit[attr] = (buf, spill)
        # #119: tally the VRAM this tensor stops holding, so the KV-pool sizing
        # step can report (and assert on) the reclaim it is about to inherit.
        row_bytes = (t.numel() // t.shape[0]) * t.element_size()
        freed_device += expert_offload_released_device_bytes(
            int(E), buf_slots, row_bytes
        )
        freed_host += spill.numel() * spill.element_size()
        # Replace the param with a 0-row placeholder so device_loading_context
        # copies nothing back to host (the full [E] GPU tensor is dropped here).
        empty = torch.empty((0,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
        if isinstance(p, torch.nn.Parameter):
            setattr(layer, attr, torch.nn.Parameter(empty, requires_grad=False))
        else:
            setattr(layer, attr, empty)

    if presplit:
        layer._moe_offload_presplit = presplit
        layer._moe_offload_full_experts = int(E)
        if not static:
            # Same publication contract as the #123-GGUF half: the buffers are
            # already arranged this way, so the cache adopts the map verbatim.
            layer._moe_offload_frozen_layout = (
                list(plan.resident_ids),
                list(plan.spill_ids),
            )
            if plan.delegated_ids:
                layer._moe_offload_delegated_experts = list(plan.delegated_ids)
        publish_host_shard_on_layer(layer, plan)
        record_expert_offload_release(freed_device, freed_host, len(presplit))
        # Return freed host memory to the OS NOW. create_weights loaded the full
        # [E] expert set to host CPU; the loader frees each layer's loaded tensor
        # (via device_loading_context) as it is repacked, but glibc/torch retain
        # the freed CPU buffers in the allocator pool -- so across the 48-layer
        # repack the retained-but-unused buffers accumulate (~ the whole [E]
        # set) and squeeze MemAvailable toward the no-swap floor. A per-layer
        # gc + malloc_trim returns them so the host peak stays ~= spill, not the
        # full loaded set. Cheap (runs once per MoE layer at load).
        del t, buf, spill
        import gc as _gc

        _gc.collect()
        try:
            import ctypes as _ct

            _ct.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
