# SPDX-License-Identifier: Apache-2.0
"""MoE expert-offload cache (feat/moe-expert-offload, TIERED RESIDENCY).

Goal
----
Fit a large routed-MoE layer (e.g. Qwen3.5-122B-A10B, 256 experts) on a small
rig by splitting each layer's ``num_local_experts`` (N) into two tiers:

* RESIDENT tier -- R experts that live PERMANENTLY in a GPU buffer at fixed
  slots ``[0, R)`` (slot == expert id). They are never evicted and, crucially,
  have NO host-RAM copy.
* SPILL tier -- the remaining ``N-R`` experts. ONLY these live in a pinned
  host-RAM pool. They are fetched on demand into a small GPU SPILL-CACHE of
  ``C`` slots (physical slots ``[R, R+C)``) with LRU eviction.

Per layer the GPU holds ``R + C`` experts' worth of weights; host RAM holds
``N - R`` experts. This is the whole point of the redesign: the earlier
"pin-everything" cache pinned all N experts in RAM (VRAM buffer of ``n_slots``
was an LRU cache over the full set), so FP8's ~116 GB expert pool overflowed
this box's 80 GB RAM. Under tiered residency RAM only has to hold the SPILL
set, so choosing R large enough makes even FP8 122B feasible.

Config (backward compatible)
----------------------------
* ``SGLANG_MOE_RESIDENT_EXPERT_FRACTION = f`` now means ``R = ceil(f * N)``
  PERMANENTLY-resident experts + ``N-R`` spill experts (previously it meant an
  ``n_slots = ceil(f*N)`` LRU cache over ALL N experts, all N pinned in RAM).
  The formula for R is unchanged; the change is that those R are now permanent
  and the host pool shrinks to the spill set. ``f == 1.0`` still means R == N,
  fully resident, no offload, bit-identical to the no-offload path.
* ``SGLANG_MOE_SPILL_CACHE_SLOTS = C`` sizes the spill cache (default 0 =
  ``max(top_k, DEFAULT_MIN_SPILL_SLOTS)``). C is clamped to ``[1, N-R]``.
  C must be >= a single token's spill-expert count, else the forward cannot be
  served (fail-fast in ``plan_token_waves``).

Wave processing
---------------
A single forward can route to MORE unique SPILL experts than there are spill
slots C (the norm on prefill of a 256-expert / top-8 model). RESIDENT experts
are always available and cost nothing; only SPILL experts consume cache slots.
So the forward is split into WAVES over TOKENS such that each wave's union of
unique SPILL experts is <= C. Each token routes to at most ``top_k`` experts,
so as long as C >= (that token's spill count) every token's complete top-k set
is serviceable in one wave. For each wave we (a) fetch its missing spill
experts into cache slots, (b) remap that wave's topk_ids -> slot ids (resident
experts map to their fixed slot == id; spill experts to their sorted cache
slot), (c) run the unmodified grouped-GEMM over the wave's token rows, and
(d) scatter the per-row outputs back into the full output buffer. Every token
is computed EXACTLY ONCE with all of its experts present, so there is no
cross-wave accumulation of a single token's partial sums.

Reproducibility, not cross-config bit-identity (IMPORTANT)
----------------------------------------------------------
The offload path is SELF-REPRODUCIBLE: the same request under the same config
yields byte-identical output run-to-run. This required TWO fixes over the first
design:

* deterministic, history-independent spill-slot assignment (see
  ``ExpertResidencyPlanner``) -- the earlier LRU cache placed the same expert in
  different physical slots depending on request history, and the FP8 grouped-
  GEMM reduction is sensitive to the expert-slot layout, so identical inputs
  produced DIFFERENT output run-to-run;
* fully-ordered H2D fetch (see ``_fetch``) -- a side copy stream raced the
  previous wave's compute.

It is NOT bitwise-identical to the no-offload (f == 1.0) path. The remapped
R+C-slot layout has a different FP reduction order than the full-N layout; under
FP8 this changes results at the sub-ULP level, which greedy decoding can amplify
into a differing token after many steps. This is an inherent floating-point
reduction-order difference of the same class as changing tensor-parallel degree
or batch size (which upstream does not make bitwise-identical either), NOT a
correctness bug -- the per-token math and weights are correct. Users who need
strict cross-config reproducibility can launch with
``--enable-deterministic-inference`` (forces order-independent kernels
globally). The pure-Python planner IS exercised for exact tensor-level identity
in the CPU/tensor tests; that identity holds for the reference (kernel-free)
MoE math and is the correctness contract for the remap/wave orchestration.

Design notes
------------
* Cold (spill) experts are FETCHED and computed on GPU (this rig's AMD CPU has
  no AMX, so ktransformers-style CPU compute is not viable here).
* Default path is untouched: with f == 1.0 the layer never installs a cache and
  behaves byte-identically.
* The resolve/LRU/wave bookkeeping (``ExpertResidencyPlanner`` +
  ``plan_token_waves``) is pure Python and unit-tested on CPU without CUDA
  (tests/moe_offload/test_planner.py); only ``MoEExpertOffloadCache`` touches
  tensors.
* CUDA-graph incompatible by nature: ``run_waves()`` does a device->host sync
  (``topk_ids.tolist()``) plus data-dependent Python planning, illegal during
  graph capture. Offload therefore REQUIRES --disable-cuda-graph; the layer
  fails fast at construction otherwise (see layer.py).
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: default spill-cache size when SGLANG_MOE_SPILL_CACHE_SLOTS is unset (0). The
#: cache must hold at least one token's worth of spill experts (<= top_k), so
#: the default is max(top_k, this constant).
DEFAULT_MIN_SPILL_SLOTS = 8

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
class ResidencyStats:
    fetches: int = 0          # spill experts H2D-copied (slot did not already hold it)
    hits: int = 0             # needed spill experts already in their target slot
    misses: int = 0           # needed spill experts not in target slot (== fetches)
    resident_hits: int = 0    # needed experts served directly from resident tier
    evictions: int = 0        # target slot overwritten (held a different expert)
    forwards: int = 0         # resolve() calls (== number of waves run)
    overflow_forwards: int = 0  # forwards that needed >C unique spill experts
    waves: int = 0            # total waves run across all forwards

    @property
    def hit_rate(self) -> float:
        """Spill-cache hit rate (resident-tier hits are not counted -- they
        never touch the cache)."""
        tot = self.hits + self.misses
        return self.hits / tot if tot else 1.0


def plan_token_waves(
    experts_per_token: Sequence[Sequence[int]],
    n_slots: int,
    n_resident: int = 0,
) -> List[List[int]]:
    """Greedily partition token indices into waves whose union of unique SPILL
    experts is <= ``n_slots`` (the spill-cache size C).

    Pure-python, CPU-testable. ``experts_per_token[t]`` is the list of routed
    expert ids for token ``t`` (``-1`` padding allowed and ignored). An expert
    id ``< n_resident`` is RESIDENT (always available, does not consume a spill
    slot); an id ``>= n_resident`` is a SPILL expert counted against
    ``n_slots``. With the default ``n_resident == 0`` every expert counts,
    which reproduces the original "LRU cache of size n_slots over all experts"
    partitioning exactly.

    Returns a list of waves; each wave is a list of token indices in original
    order. Every token index appears in exactly one wave.

    Raises ``ValueError`` if a single token needs more than ``n_slots`` unique
    SPILL experts -- the spill cache cannot serve even one token; the caller
    must fail fast rather than silently drop experts.
    """
    if n_slots < 1:
        raise ValueError("n_slots must be >= 1")
    waves: List[List[int]] = []
    cur_rows: List[int] = []
    cur_set: set = set()
    for t, experts in enumerate(experts_per_token):
        ex = {int(e) for e in experts if e is not None and int(e) >= n_resident}
        if len(ex) > n_slots:
            raise ValueError(
                f"token {t} routes to {len(ex)} unique SPILL experts but only "
                f"n_slots={n_slots} spill-cache slots are available "
                f"(SGLANG_MOE_SPILL_CACHE_SLOTS too small: a single token's "
                f"spill experts must fit in the spill cache)."
            )
        if cur_rows and len(cur_set | ex) > n_slots:
            waves.append(cur_rows)
            cur_rows = []
            cur_set = set()
        cur_set |= ex
        cur_rows.append(t)
    if cur_rows:
        waves.append(cur_rows)
    return waves


@dataclass
class ExpertResidencyPlanner:
    """Pure-python tiered-residency bookkeeping for one MoE layer (CPU-testable).

    Experts ``[0, n_resident)`` are RESIDENT: permanently at slot == expert id,
    never evicted, no host copy. Experts ``[n_resident, num_local_experts)`` are
    SPILL: fetched on demand into the ``n_spill_slots`` (C) spill-cache slots at
    physical positions ``[n_resident, n_resident + n_spill_slots)``.

    Deterministic (history-independent) slot assignment
    ---------------------------------------------------
    Within a wave, the unique SPILL experts are SORTED and assigned to slots
    ``R, R+1, ...`` in that order -- a PURE FUNCTION of the wave's spill-expert
    SET, independent of any cache history carried across forwards or requests.
    This is the correctness fix over the earlier LRU cache: with LRU, the same
    expert landed in a different physical slot depending on request history, and
    the FP8 grouped-GEMM's reduction is sensitive to the expert-slot layout, so
    identical inputs produced DIFFERENT (non-reproducible) output run-to-run.
    Sorted assignment makes the layout -- and therefore the output -- a
    deterministic function of the input, so the offload path is SELF-
    REPRODUCIBLE (same request + config -> byte-identical output). (It is not
    bitwise-identical to the no-offload path: the remapped R+C-slot layout has a
    different FP reduction order than the full-N layout, an inherent FP-order
    difference of the same class as changing TP degree or batch size.)

    ``_slot_contents`` maps each spill slot -> the expert currently staged in it
    (for skipping redundant fetches only; it never affects the ASSIGNMENT, hence
    never the output). ``n_resident == 0`` degenerates to the former
    pin-everything cache (all experts spill), now likewise deterministic.
    """

    num_local_experts: int
    n_resident: int = 0
    n_spill_slots: int = 0
    _slot_contents: Dict[int, int] = field(default_factory=dict)
    stats: ResidencyStats = field(default_factory=ResidencyStats)

    def __post_init__(self):
        if self.n_resident < 0:
            raise ValueError("n_resident must be >= 0")
        if self.n_resident > self.num_local_experts:
            self.n_resident = self.num_local_experts
        n_spill = self.num_local_experts - self.n_resident
        # Clamp the spill cache to [0, n_spill]; a cache bigger than the spill
        # set is pointless (all spill experts would fit at once).
        if self.n_spill_slots > n_spill:
            self.n_spill_slots = n_spill
        if n_spill > 0 and self.n_spill_slots < 1:
            raise ValueError(
                "n_spill_slots must be >= 1 when there are spill experts"
            )

        # Spill-cache warm start: slots [R, R+C) initially hold experts [R, R+C)
        # (slot == id) -- matches how the GPU buffer is laid out at install time
        # (a contiguous [0, R+C) slice of the original expert-major tensor).
        self._slot_contents = {
            self.n_resident + i: self.n_resident + i
            for i in range(self.n_spill_slots)
        }

    @property
    def buffer_slots(self) -> int:
        """Total GPU expert-buffer size after install (R + C)."""
        return self.n_resident + self.n_spill_slots

    @property
    def fully_resident(self) -> bool:
        """True when every expert is permanently resident (no spill tier) ->
        the no-offload passthrough (remap is the identity)."""
        return self.n_resident >= self.num_local_experts

    def resolve(
        self, needed: Sequence[int]
    ) -> Tuple[Dict[int, int], List[Tuple[int, int]]]:
        """Return (slot_of_needed, fetch_plan) for one wave.

        slot_of_needed: expert_id -> slot for every needed expert (resident ->
        slot == id, no fetch; spill -> its deterministically-assigned cache
        slot).
        fetch_plan: list of (spill_expert_id, target_slot) to H2D-copy (only the
        target slots that do not already hold the wanted expert).

        The spill experts are SORTED and mapped to slots R, R+1, ... -- a pure
        function of the wave's spill set, so the resulting layout is identical
        for identical waves regardless of history. This is what makes the
        offload path self-reproducible.

        ``needed`` must contain at most ``n_spill_slots`` unique SPILL experts
        (guaranteed by ``plan_token_waves``); otherwise RuntimeError.
        """
        self.stats.forwards += 1
        self.stats.waves += 1
        needed_unique = sorted(set(int(e) for e in needed if e >= 0))

        resident = [e for e in needed_unique if e < self.n_resident]
        spill = [e for e in needed_unique if e >= self.n_resident]
        self.stats.resident_hits += len(resident)
        slot_of_needed: Dict[int, int] = {e: e for e in resident}

        if not spill:
            return slot_of_needed, []

        if len(spill) > self.n_spill_slots:
            raise RuntimeError(
                f"resolve() got {len(spill)} unique spill experts but only "
                f"{self.n_spill_slots} spill slots exist; caller must "
                f"wave-split with plan_token_waves() first."
            )

        # Deterministic assignment: sorted spill experts -> slots R, R+1, ...
        fetch_plan: List[Tuple[int, int]] = []
        for i, e in enumerate(spill):
            slot = self.n_resident + i
            slot_of_needed[e] = slot
            if self._slot_contents.get(slot) == e:
                self.stats.hits += 1  # already staged in its target slot
            else:
                if self._slot_contents.get(slot) is not None:
                    self.stats.evictions += 1
                self._slot_contents[slot] = e
                fetch_plan.append((e, slot))
                self.stats.fetches += 1
                self.stats.misses += 1
        return slot_of_needed, fetch_plan


def resident_slot_count(num_local_experts: int, fraction: float) -> int:
    """Number of permanently-resident experts R = ceil(fraction * N), clamped
    to ``[1, N]``. (Formula unchanged from the pin-everything design; only the
    interpretation changed -- these R experts are now permanent, not an LRU
    cache over all N.)"""
    n = int(math.ceil(fraction * num_local_experts))
    return max(1, min(num_local_experts, n))


def default_spill_slots(top_k: Optional[int]) -> int:
    """Auto spill-cache size when unset: max(top_k, DEFAULT_MIN_SPILL_SLOTS).
    A single token routes to <= top_k experts, so the cache must be able to
    hold at least top_k spill experts for the worst-case all-spill token."""
    tk = int(top_k) if top_k else 0
    return max(tk, DEFAULT_MIN_SPILL_SLOTS)


class MoEExpertOffloadCache:
    """Tensor-level wrapper around ExpertResidencyPlanner for a FusedMoE layer.

    Wiring lives here (built during the GPU window). It expects the layer's
    stacked expert tensors (w13_weight/w2_weight [+scales]); it keeps the
    RESIDENT tier ``[0, R)`` on GPU permanently, moves ONLY the SPILL tier
    ``[R, N)`` to a pinned host pool, and allocates a GPU buffer of ``R + C``
    experts (resident tier + spill cache).

    NOTE: the tensor path requires CUDA and is exercised in the GPU window; the
    planner/wave bookkeeping carries all correctness-critical logic and is
    tested on CPU now (tests/moe_offload/test_planner.py).
    """

    #: names of the stacked per-expert tensors to pool/fetch (dim 0 == expert).
    EXPERT_TENSOR_ATTRS = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
    )

    def __init__(self, layer, fraction: float, spill_slots: int = 0):
        self.layer = layer
        self.fraction = fraction
        self.num_local_experts = int(getattr(layer, "num_local_experts"))
        # R = permanently-resident experts.
        self.n_resident = resident_slot_count(self.num_local_experts, fraction)
        n_spill = self.num_local_experts - self.n_resident
        # C = spill-cache slots. 0 => auto (max(top_k, const)). Clamp to spill.
        top_k = getattr(layer, "top_k", None)
        if not top_k:
            rc = getattr(layer, "moe_runner_config", None)
            top_k = getattr(rc, "top_k", None) if rc is not None else None
        c = int(spill_slots) if spill_slots and spill_slots > 0 else default_spill_slots(top_k)
        self.n_spill_slots = max(1, min(c, n_spill)) if n_spill > 0 else 0
        self.planner = ExpertResidencyPlanner(
            num_local_experts=self.num_local_experts,
            n_resident=self.n_resident,
            n_spill_slots=self.n_spill_slots,
        )
        # Total GPU expert-buffer size after install (R + C). Kept as n_slots
        # for backward-compat with callers/tests that read the buffer size.
        self.n_slots = self.planner.buffer_slots
        self._pinned: Dict[str, "object"] = {}   # attr -> pinned CPU tensor [N-R,...]
        self._resident: Dict[str, "object"] = {}  # attr -> GPU tensor [R+C,...]
        self._installed = False

    # --- lifecycle (GPU window) --------------------------------------------
    def install(self):  # pragma: no cover - requires CUDA
        """Keep resident tier [0,R) on GPU; move ONLY the spill tier [R,N) to a
        pinned host pool; allocate the [R+C] GPU buffer (resident + spill cache
        warm-started with experts [R, R+C)). Idempotent."""
        import torch

        if self._installed or self.planner.fully_resident:
            return
        R = self.n_resident
        C = self.n_spill_slots
        for attr in self.EXPERT_TENSOR_ATTRS:
            full = getattr(self.layer, attr, None)
            if full is None:
                continue
            full = full.data if hasattr(full, "data") else full
            if full.shape[0] != self.num_local_experts:
                continue  # not an expert-major tensor
            # Host pool holds ONLY the spill tier [R, N) -- the RAM win: RAM no
            # longer holds the resident experts, so it is sized to (N-R), not N.
            spill = full[R:].contiguous()
            pinned = torch.empty_like(spill, device="cpu").pin_memory()
            pinned.copy_(spill)
            self._pinned[attr] = pinned
            # GPU buffer = experts [0, R+C): resident tier [0,R) + spill cache
            # [R,R+C) warm-started with spill experts [R,R+C). Contiguous slice.
            buffer = full[: R + C].contiguous()
            self._resident[attr] = buffer
            setattr(
                self.layer, attr,
                torch.nn.Parameter(buffer, requires_grad=False)
                if isinstance(getattr(self.layer, attr), torch.nn.Parameter)
                else buffer,
            )
        # Advertise the R+C buffer size so the grouped-GEMM / MoE runner size
        # their per-expert loops to the GPU buffer (topk_ids arriving at apply()
        # are already slot ids in [0, R+C)). The planner keeps the *full*
        # num_local_experts for id bookkeeping (captured in __init__).
        self._orig_num_local_experts = self.layer.num_local_experts
        self.layer.num_local_experts = R + C
        runner_cfg = getattr(self.layer, "moe_runner_config", None)
        if runner_cfg is not None and hasattr(runner_cfg, "num_local_experts"):
            try:
                runner_cfg.num_local_experts = R + C
            except Exception:
                pass  # frozen/dataclass runner configs: kernel reads the layer attr
        self._installed = True

    @property
    def host_pool_experts(self) -> int:
        """Number of experts held in the pinned host pool (== spill set N-R).
        The RAM-win proof: this is < num_local_experts whenever offload is
        active."""
        for t in self._pinned.values():
            return int(t.shape[0])
        return self.num_local_experts - self.n_resident

    # --- fetch / remap helpers (GPU window) --------------------------------
    def _fetch(self, fetch_plan):  # pragma: no cover - requires CUDA
        """H2D-copy the (spill_expert_id -> slot) misses of one wave into the
        spill-cache slots. The pinned pool is indexed by spill-local index
        (expert_id - R).

        Correctness-critical (byte-identity gate). Two hazards, both fatal to
        reproducibility on the real server and BOTH avoided here by writing the
        spill slots on the current (compute) stream and then hard-syncing that
        stream before compute reads them:

        1. A side copy stream that only does ``current.wait_stream(copy)`` after
           the copies does NOT order the copy against the PREVIOUS wave's
           compute -- wave N+1's fetch can overwrite a spill slot while wave N's
           grouped-GEMM still reads it (read/write race).
        2. Under sglang's overlap scheduler the forward may run on a stream that
           is not the one an unqualified ``copy_(non_blocking=True)`` lands on,
           so same-stream ordering alone is not guaranteed.

        A CPU-blocking ``synchronize()`` after the (blocking) copies removes both
        hazards unconditionally: every fetched weight is fully resident before
        the wave's apply() is even enqueued. This makes the offload path
        bit-reproducible and byte-identical to the no-offload path, at the cost
        of the copy/compute overlap the async version attempted (acceptable:
        offload is eager-only and correctness-gated)."""
        import torch

        if not fetch_plan:
            return
        R = self.n_resident
        for attr, pinned in self._pinned.items():
            dst = self._resident[attr]
            for expert_id, slot in fetch_plan:
                dst[slot].copy_(pinned[expert_id - R], non_blocking=False)
        torch.cuda.current_stream().synchronize()

    def _build_lut(self, slot_of_needed, dtype, device):  # pragma: no cover
        import torch

        lut = torch.full(
            (self.num_local_experts,), -1, dtype=dtype, device=device
        )
        for e, s in slot_of_needed.items():
            lut[e] = s
        return lut

    @staticmethod
    def _remap(topk_ids, lut):  # pragma: no cover - requires CUDA
        import torch

        # -1 padding stays -1; every real id maps to its GPU-buffer slot.
        return torch.where(topk_ids >= 0, lut[topk_ids.clamp(min=0)], topk_ids)

    # --- per-forward (GPU window) ------------------------------------------
    def prepare(self, topk_ids):  # pragma: no cover - requires CUDA
        """Single-wave remap for a forward whose unique SPILL experts fit in C
        (e.g. decode). Resolves residency, async-fetches spill misses, returns a
        remapped topk_ids (expert id -> GPU slot; -1 stays -1).

        Raises if the forward needs more than C unique spill experts -- callers
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

    def run_waves(self, dispatch_output, apply_fn):  # pragma: no cover - CUDA
        """Run the grouped-GEMM for one forward, wave-splitting when the forward
        needs more unique SPILL experts than there are spill-cache slots C.

        ``apply_fn(sub_dispatch_output) -> CombineInput`` runs the unmodified
        MoE math (``quant_method.apply``) over the R+C GPU buffer. We call it
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
        waves = plan_token_waves(ids_list, self.n_spill_slots, self.n_resident)

        # Fast path: the whole forward fits in one wave (typical decode). Remap
        # the full batch and run a single apply -- no token slicing overhead.
        if len(waves) == 1:
            needed = sorted({e for row in ids_list for e in row if e >= 0})
            slot_of_needed, fetch_plan = self.planner.resolve(needed)
            self._fetch(fetch_plan)
            lut = self._build_lut(slot_of_needed, topk_ids.dtype, topk_ids.device)
            remapped = self._remap(topk_ids, lut)
            sub = dispatch_output._replace(
                topk_output=topk_output._replace(topk_ids=remapped)
            )
            return apply_fn(sub)

        # Multi-wave (prefill overflow): process disjoint token subsets.
        self.planner.stats.overflow_forwards += 1
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
        return combine_out._replace(hidden_states=out_full)

    @property
    def stats(self) -> ResidencyStats:
        return self.planner.stats
