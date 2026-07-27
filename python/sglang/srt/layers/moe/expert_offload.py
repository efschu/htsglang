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
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --- M-C routing trace ------------------------------------------------------
# Append-only JSONL sink consumed by moe_offload/sim.py. One handle per output
# file, shared by every FusedMoE layer in a process and serialized by a lock so
# concurrent layers never interleave a line. Distinct TP/EP ranks write to
# distinct files (rank tag in the name), so no cross-process contention exists.
_TRACE_HANDLES: Dict[str, "object"] = {}
_TRACE_LOCK = threading.Lock()


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
    device_bytes: int, host_bytes: int, tensors: int = 1
) -> None:
    """Tally one layer's release. Called once per layer that actually split."""
    _RELEASE_TALLY.device_bytes += max(0, int(device_bytes))
    _RELEASE_TALLY.host_bytes += max(0, int(host_bytes))
    _RELEASE_TALLY.tensors += max(0, int(tensors))
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
        self._capturable_ready = False
        self._cap_resident_slot_lut = None  # int32[E] device
        self._cap_is_spill = None  # bool[E] device
        self._cap_spill_pool_row_lut = None  # int32[E] device
        self._cap_pool_dev: Dict[str, "object"] = {}  # attr -> UVA view [E-R,...]
        self._cap_scratch_dst: Dict[str, "object"] = {}  # attr -> resident[R:R+C]
        self._cap_view_holders: List["object"] = []  # keep pinned bases alive

    # --- lifecycle (GPU window) --------------------------------------------
    def install(self):  # pragma: no cover - requires CUDA
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
        self._stream = torch.cuda.Stream()
        dev = torch.cuda.current_device()
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
    def _fetch(self, fetch_plan):  # pragma: no cover - requires CUDA
        """Async H2D-copy each wave's SPILL experts into their scratch slots,
        then join the copy stream before compute reads them. ``fetch_plan`` is
        (spill_expert_id, scratch_slot); the spill pool is indexed by
        (expert_id - resident_count)."""
        import torch

        if not fetch_plan:
            return
        R = self.resident_count
        pool_index = self._spill_pool_index  # None => static layout (id - R)
        moved = 0
        with torch.cuda.stream(self._stream):
            for attr, spill in self._pinned.items():
                dst = self._resident[attr]
                per_expert = dst[0].numel() * dst.element_size()
                for expert_id, slot in fetch_plan:
                    row = (
                        pool_index[expert_id]
                        if pool_index is not None
                        else expert_id - R
                    )
                    dst[slot].copy_(spill[row], non_blocking=True)
                    moved += per_expert
        torch.cuda.current_stream().wait_stream(self._stream)
        self.planner.stats.h2d_bytes += moved

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
    def _remap(topk_ids, lut):  # pragma: no cover - requires CUDA
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

    def run_waves(self, dispatch_output, apply_fn):  # pragma: no cover - CUDA
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
        being inferred from the wall clock. INFO on layer 0 (one line per chunk),
        DEBUG on the rest."""
        import logging

        gib = (self.planner.stats.h2d_bytes - before) / float(1 << 30)
        layer_id = getattr(self.layer, "layer_id", "?")
        logging.getLogger(__name__).log(
            logging.INFO if layer_id == 0 else logging.DEBUG,
            "MoE offload layer %s: %s-major prefill, %d waves, %.2f GiB H2D",
            layer_id,
            order,
            waves,
            gib,
        )

    def _run_single_wave(self, dispatch_output, apply_fn, ids_list):  # pragma: no cover
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


def presplit_expert_offload_after_repack(layer) -> None:  # pragma: no cover - CUDA
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
    """
    import torch

    from sglang.srt.environ import envs

    frac = envs.SGLANG_MOE_RESIDENT_EXPERT_FRACTION.get()
    if frac >= 1.0:
        return
    E = getattr(layer, "num_local_experts", None)
    if not E:
        return
    R = resident_slot_count(int(E), frac)
    if R >= int(E):
        return
    C = scratch_slot_count(R)
    buf_slots = min(R + C, int(E))

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
        buf[:R].copy_(t[:R])
        # Spill [R:E] -> pinned host (contiguous copy; the GPU [E] is then freed).
        spill = torch.empty(
            (int(E) - R,) + tuple(t.shape[1:]), dtype=t.dtype, device="cpu"
        ).pin_memory()
        spill.copy_(t[R:])
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
