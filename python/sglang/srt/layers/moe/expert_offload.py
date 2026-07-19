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
class ResidencyStats:
    fetches: int = 0          # experts H2D-copied (misses that fit)
    hits: int = 0             # needed experts already resident
    misses: int = 0           # needed experts not resident
    evictions: int = 0        # resident experts kicked out
    forwards: int = 0         # resolve() calls (== number of waves run)
    overflow_forwards: int = 0  # forwards that needed >n_slots unique experts
    waves: int = 0            # total waves run across all forwards

    @property
    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 1.0


def plan_token_waves(
    experts_per_token: Sequence[Sequence[int]], n_slots: int
) -> List[List[int]]:
    """Greedily partition token indices into waves whose union of unique routed
    experts is <= ``n_slots``.

    Pure-python, CPU-testable. ``experts_per_token[t]`` is the list of routed
    expert ids for token ``t`` (``-1`` padding allowed and ignored). Returns a
    list of waves; each wave is a list of token indices in original order.
    Every token index appears in exactly one wave.

    Raises ``ValueError`` if a single token needs more than ``n_slots`` unique
    experts -- that means ``n_slots < top_k`` and offload cannot serve even one
    token; the caller must fail fast rather than silently drop experts.
    """
    if n_slots < 1:
        raise ValueError("n_slots must be >= 1")
    waves: List[List[int]] = []
    cur_rows: List[int] = []
    cur_set: set = set()
    for t, experts in enumerate(experts_per_token):
        ex = {int(e) for e in experts if e is not None and e >= 0}
        if len(ex) > n_slots:
            raise ValueError(
                f"token {t} routes to {len(ex)} unique experts but only "
                f"n_slots={n_slots} resident slots are available "
                f"(SGLANG_MOE_RESIDENT_EXPERT_FRACTION too small: a single "
                f"token's top-k must fit in the resident buffer)."
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
    """Pure-python LRU residency bookkeeping for one MoE layer (CPU-testable).

    slot_of_expert maps a resident global expert id -> its slot index in the
    resident buffer. `_lru` orders resident expert ids from least- to
    most-recently used.

    A single ``resolve()`` call must be given at most ``n_slots`` unique
    experts (guaranteed by ``plan_token_waves``); it never evicts an expert
    that is needed within the same call.
    """

    num_local_experts: int
    n_slots: int
    slot_of_expert: Dict[int, int] = field(default_factory=dict)
    _free_slots: List[int] = field(default_factory=list)
    _lru: "OrderedDict[int, None]" = field(default_factory=OrderedDict)
    stats: ResidencyStats = field(default_factory=ResidencyStats)

    def __post_init__(self):
        if self.n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        if self.n_slots > self.num_local_experts:
            self.n_slots = self.num_local_experts
        # Pre-populate residency with experts [0, n_slots) as a deterministic
        # warm start (matches how the pinned pool is laid out at load time).
        self._free_slots = []
        for e in range(self.n_slots):
            self.slot_of_expert[e] = e
            self._lru[e] = None

    @property
    def fully_resident(self) -> bool:
        return self.n_slots >= self.num_local_experts

    def resolve(
        self, needed: Sequence[int]
    ) -> Tuple[Dict[int, int], List[Tuple[int, int]]]:
        """Return (slot_of_needed, fetch_plan) for one wave.

        slot_of_needed: expert_id -> slot for every needed expert.
        fetch_plan: list of (expert_id, target_slot) to H2D-copy this wave.
        Experts already resident are LRU-touched. Misses evict the LRU
        residents that are not themselves needed this wave.

        ``needed`` must contain at most ``n_slots`` unique experts; otherwise a
        needed expert would have to be evicted (RuntimeError). Callers partition
        forwards with ``plan_token_waves`` to guarantee this.
        """
        self.stats.forwards += 1
        self.stats.waves += 1
        needed_unique = sorted(set(int(e) for e in needed if e >= 0))
        if self.fully_resident:
            self.stats.hits += len(needed_unique)
            return {e: e for e in needed_unique}, []

        if len(needed_unique) > self.n_slots:
            raise RuntimeError(
                f"resolve() got {len(needed_unique)} unique experts but only "
                f"{self.n_slots} slots exist; caller must wave-split with "
                f"plan_token_waves() first."
            )

        fetch_plan: List[Tuple[int, int]] = []
        misses = [e for e in needed_unique if e not in self.slot_of_expert]
        hits = [e for e in needed_unique if e in self.slot_of_expert]
        self.stats.hits += len(hits)
        self.stats.misses += len(misses)

        # Touch hits (most-recently-used) so they are not evicted below.
        for e in hits:
            self._lru.move_to_end(e)

        needed_set = set(needed_unique)
        for e in misses:
            slot = self._acquire_slot(protect=needed_set)
            self.slot_of_expert[e] = slot
            self._lru[e] = None
            self._lru.move_to_end(e)
            fetch_plan.append((e, slot))
            self.stats.fetches += 1

        slot_of_needed = {e: self.slot_of_expert[e] for e in needed_unique}
        return slot_of_needed, fetch_plan

    def _acquire_slot(self, protect: set) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        # Evict least-recently-used expert that is not needed this wave. Because
        # |needed| <= n_slots, a non-protected victim always exists.
        for victim in list(self._lru.keys()):
            if victim in protect:
                continue
            slot = self.slot_of_expert.pop(victim)
            del self._lru[victim]
            self.stats.evictions += 1
            return slot
        raise RuntimeError(
            "no evictable slot: all resident experts are needed this wave "
            "(should be unreachable when |needed| <= n_slots)."
        )


def resident_slot_count(num_local_experts: int, fraction: float) -> int:
    """Slots to keep resident for a given fraction (>=1)."""
    n = int(math.ceil(fraction * num_local_experts))
    return max(1, min(num_local_experts, n))


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
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
    )

    def __init__(self, layer, fraction: float):
        self.layer = layer
        self.fraction = fraction
        self.num_local_experts = int(getattr(layer, "num_local_experts"))
        self.n_slots = resident_slot_count(self.num_local_experts, fraction)
        self.planner = ExpertResidencyPlanner(
            num_local_experts=self.num_local_experts, n_slots=self.n_slots
        )
        self._pinned: Dict[str, "object"] = {}   # attr -> pinned CPU tensor [E,...]
        self._resident: Dict[str, "object"] = {}  # attr -> GPU tensor [n_slots,...]
        self._stream = None
        self._installed = False

    # --- lifecycle (GPU window) --------------------------------------------
    def install(self):  # pragma: no cover - requires CUDA
        """Move full expert tensors to a pinned host pool; allocate resident
        GPU slots pre-filled with experts [0, n_slots). Idempotent."""
        import torch

        if self._installed or self.planner.fully_resident:
            return
        self._stream = torch.cuda.Stream()
        for attr in self.EXPERT_TENSOR_ATTRS:
            full = getattr(self.layer, attr, None)
            if full is None:
                continue
            full = full.data if hasattr(full, "data") else full
            if full.shape[0] != self.num_local_experts:
                continue  # not an expert-major tensor
            pinned = torch.empty_like(full, device="cpu").pin_memory()
            pinned.copy_(full)
            self._pinned[attr] = pinned
            resident = full[: self.n_slots].contiguous()
            self._resident[attr] = resident
            setattr(self.layer, attr, torch.nn.Parameter(resident, requires_grad=False)
                    if isinstance(getattr(self.layer, attr), torch.nn.Parameter)
                    else resident)
        # Shrink the layer's advertised local-expert count to the resident slot
        # count so the grouped-GEMM / MoE runner size their per-expert loops to
        # the resident buffer (topk_ids arriving at apply() are already slot
        # ids). The planner keeps the *full* num_local_experts for id bookkeeping
        # (captured in __init__ before this runs). Runner config mirrors it in
        # case the kernel reads the config rather than the layer attribute.
        self._orig_num_local_experts = self.layer.num_local_experts
        self.layer.num_local_experts = self.n_slots
        runner_cfg = getattr(self.layer, "moe_runner_config", None)
        if runner_cfg is not None and hasattr(runner_cfg, "num_local_experts"):
            try:
                runner_cfg.num_local_experts = self.n_slots
            except Exception:
                pass  # frozen/dataclass runner configs: kernel reads the layer attr
        self._installed = True

    # --- fetch / remap helpers (GPU window) --------------------------------
    def _fetch(self, fetch_plan):  # pragma: no cover - requires CUDA
        """Async H2D-copy the (expert_id -> slot) misses of one wave into the
        resident buffers, then join the copy stream before compute reads them."""
        import torch

        if not fetch_plan:
            return
        with torch.cuda.stream(self._stream):
            for attr, pinned in self._pinned.items():
                dst = self._resident[attr]
                for expert_id, slot in fetch_plan:
                    dst[slot].copy_(pinned[expert_id], non_blocking=True)
        torch.cuda.current_stream().wait_stream(self._stream)

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
        waves = plan_token_waves(ids_list, self.n_slots)

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
