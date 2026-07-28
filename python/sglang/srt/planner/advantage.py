# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The honest advantage vs stock even-TP (design §3).

Three fields, three epistemic statuses — and STRUCTURALLY nothing else:

  * ``stock`` (feasibility)  — derived from the same geometry, EXACT:
    "stock even-TP cannot run because <machine-checkable reason>".
  * ``capacity_pct_range``   — % more KV/context than even-TP capped to the
    smallest card. A ratio of two same-cost-model estimates (robust), shown
    as a rounded range, never a precise point.
  * ``measured``             — per-card MEASURED scores, present ONLY when a
    cached hardware profile already exists (``get_cached_hardware_profile``
    is cache-only and never triggers a probe). Absent otherwise — absent,
    not estimated.

There is deliberately NO estimated-throughput/tok-s field in these types
(design §3.4): the planner is capacity/feasibility only, and emitting an
invented perf number is made structurally impossible.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

from sglang.srt.planner.capacity import CapacityReport
from sglang.srt.planner.hardware import HardwareSpec
from sglang.srt.uneven_perf import PlanInputs

__all__ = [
    "Advantage",
    "StockFeasibility",
    "MeasuredCardScore",
    "compute_advantage",
]


@dataclasses.dataclass(frozen=True)
class StockFeasibility:
    """Whether STOCK even-TP (upstream sglang, no fork flags) runs this
    model on this hardware, with machine-checkable reasons when not."""

    runs: bool
    reasons: Tuple[str, ...] = ()
    #: Stock's predicted max context when it runs (estimate; every card is
    #: throttled to the smallest card's budget under even TP).
    max_context_tokens: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class MeasuredCardScore:
    """A MEASURED micro-probe score from the cached hardware profile —
    never estimated, and only echoed when the cache already exists."""

    gpu_index: int
    name: str
    gemm_tflops: float
    membw_gbs: float
    #: The card's fp8 GEMM rate, when the #213 card probe measured one. None
    #: on a card with no fp8 tensor path and None when only the stage-0
    #: profile is cached -- absent rather than back-filled from the bf16
    #: figure, which would price an fp8 plan on the wrong number.
    gemm_fp8_tflops: Optional[float] = None
    #: Which probe these rates came from: "card-probe" or "stage-0".
    source: str = "stage-0"


@dataclasses.dataclass(frozen=True)
class Advantage:
    stock: StockFeasibility
    #: (lo, hi) rounded % capacity gain vs stock even-TP; None when stock
    #: does not run (then the advantage IS feasibility) or the fork plan
    #: itself does not fit.
    capacity_pct_range: Optional[Tuple[int, int]]
    #: Cache-only measured card scores (design §3.3); None when no cached
    #: profile exists — absent, not guessed.
    measured: Optional[Tuple[MeasuredCardScore, ...]] = None
    #: "no decode regression expected" boolean from the fork's measured
    #: knee guard — only computable when measured membw scores exist.
    decode_knee_ok: Optional[bool] = None


def _card_probe_entries() -> dict:
    """{uuid: entry} from the cached #213 card probe, in the same shape the
    stage-0 profile uses so the two are interchangeable at the call site.

    Empty when no probe is cached. Never triggers one: a multi-second GPU
    measurement as a side effect of computing an advantage would be a
    surprise, and the probe has its own explicit entry points.
    """
    try:
        from sglang.srt.rigmon.card_probe import load_card_probe

        cprobe = load_card_probe()
    except Exception:
        return {}
    if cprobe is None:
        return {}
    return {
        c.uuid: {
            "name": c.name,
            "gemm_tflops": c.gemm_bf16_tflops,
            "membw_gbs": c.membw_gbs,
            "gemm_fp8_tflops": c.gemm_fp8_tflops,
            "source": "card-probe",
        }
        for c in cprobe.cards
        if c.gemm_bf16_tflops and c.membw_gbs
    }


def _round_range(pct: float, step: int = 5) -> Tuple[int, int]:
    """Report the capacity gain at the cost model's honesty granularity:
    a band of ±5% RELATIVE width (the estimate's confidence margin, design
    §5) snapped outward to ``step``-point bounds — never a point estimate."""
    slack = max(abs(pct) * 0.05, 1.0)
    lo = int(math.floor((pct - slack) / step) * step)
    hi = int(math.ceil((pct + slack) / step) * step)
    return lo, hi


def compute_advantage(
    inputs: PlanInputs,
    hardware: HardwareSpec,
    fork_capacity: Optional[CapacityReport],
    user_free_reserve_mib: Optional[Sequence[int]] = None,
) -> Advantage:
    """Contrast the (already-computed) fork plan against stock even-TP on
    the same cards, using the SAME cost model for both sides — the ratio is
    trustworthy even where the absolutes are estimates."""
    from sglang.srt.planner.capacity import predict_capacity
    from sglang.srt.planner.plan import derive_auto_plan
    from sglang.srt.uneven_perf import PerfCostModel

    tp = inputs.tp_size
    reasons: List[str] = []

    # ---- geometry-derived stock feasibility (exact, design §3.1) --------
    used_cards = sorted(set(inputs.rank_gpu_id or []))
    if inputs.rank_gpu_id and len(used_cards) < tp:
        reasons.append(
            f"stock even-TP needs {tp} GPUs for TP={tp}; this plan runs "
            f"{tp} ranks on {len(used_cards)} card(s) via co-location "
            "(--rank-gpu-id duplicates), which stock cannot express."
        )
    if tp > len(hardware.gpus):
        reasons.append(
            f"stock even-TP needs {tp} GPUs; only {len(hardware.gpus)} "
            "present."
        )

    probe = PerfCostModel(inputs, [1] * tp, [1024] * tp)  # geometry read only
    if probe.kv_heads < tp:
        reasons.append(
            f"stock head-sharding fails: {probe.kv_heads} KV heads < "
            f"TP={tp} (stock caps TP at the KV-head count; the fork "
            "replicates KV + token-shards)."
        )
    else:
        if probe.q_heads % tp != 0:
            reasons.append(
                f"even TP cannot shard {probe.q_heads} query heads into "
                f"{tp} equal parts."
            )
        if probe.kv_heads % tp != 0:
            reasons.append(
                f"even TP cannot shard {probe.kv_heads} KV heads into "
                f"{tp} equal parts."
            )
    if probe.mlp_units % tp != 0:
        reasons.append(
            f"even TP cannot shard the FFN intermediate "
            f"({probe.intermediate}, {probe.mlp_units} quant-group units) "
            f"into {tp} group-aligned parts."
        )

    # ---- stock capacity: even split capped to the smallest card ---------
    stock_ctx: Optional[float] = None
    if not reasons:
        stock_inputs = dataclasses.replace(
            inputs,
            rank_gpu_id=used_cards or [g.index for g in hardware.gpus[:tp]],
            effective_vram_mib=None,
            rank_tp_ratio=None,
            rank_mlp_ratio=None,
            rank_moe_ratio=None,
            rank_vocab_ratio=None,
            kv_token_vector=None,
        )
        try:
            stock_auto = derive_auto_plan(
                stock_inputs, hardware, user_free_reserve_mib
            )
            # Even TP throttles every rank to the smallest card's budget.
            min_budget = min(stock_auto.budgets_mib)
            stock_inputs.effective_vram_mib = [min_budget] * tp
            stock_report = predict_capacity(
                stock_inputs, [1] * tp, [min_budget] * tp
            )
            if stock_report.feasible:
                stock_ctx = stock_report.max_context_tokens
            else:
                reasons.append(
                    "stock even-TP OOMs: capped to the smallest card's "
                    f"budget ({min_budget} MiB/rank), the weights + reserves "
                    "leave no viable KV cache (same cost model as the fork "
                    "estimate)."
                )
        except ValueError as e:
            reasons.append(f"stock even-TP cannot size this setup: {e}")

    stock = StockFeasibility(
        runs=not reasons,
        reasons=tuple(reasons),
        max_context_tokens=stock_ctx,
    )

    # ---- capacity-% (ratio of two same-model estimates, design §3.2) ----
    capacity_pct_range = None
    if (
        stock.runs
        and stock_ctx
        and fork_capacity is not None
        and fork_capacity.feasible
        and fork_capacity.max_context_tokens > 0
    ):
        pct = 100.0 * (fork_capacity.max_context_tokens / stock_ctx - 1.0)
        capacity_pct_range = _round_range(pct)

    # ---- measured scores: CACHE-ONLY, never probe (design §3.3) ---------
    # Only consulted when the hardware spec IS the live rig (NVML-derived):
    # for a manual/json spec the physical indices need not correspond to
    # the local machine's cards, so echoing local cached scores would be a
    # false attribution — measured stays absent instead.
    measured = None
    knee_ok = None
    if hardware.source not in ("pynvml", "nvidia-smi", "nvml"):
        return Advantage(
            stock=stock,
            capacity_pct_range=capacity_pct_range,
            measured=None,
            decode_knee_ok=None,
        )
    try:
        from sglang.srt.uneven_perf import get_cached_hardware_profile

        profile, gpus = get_cached_hardware_profile()
        # The #213 card probe measures the same two rates plus fp8, H2D/D2H
        # and the ordered pair matrix, so where it is cached it is the richer
        # answer about the same cards and wins per UUID. Where it is not, the
        # stage-0 profile still answers, unchanged.
        card_rates = _card_probe_entries()
        if (profile is not None or card_rates) and inputs.rank_gpu_id:
            uuid_by_idx = {g["cuda_index"]: g["uuid"] for g in gpus}
            gpu_entries = (profile or {}).get("gpus", {})
            scores = []
            membw = []
            for gid in inputs.rank_gpu_id:
                uuid = uuid_by_idx.get(gid, "")
                entry = card_rates.get(uuid) or gpu_entries.get(uuid)
                if entry is None:
                    scores = None
                    break
                scores.append(
                    MeasuredCardScore(
                        gpu_index=gid,
                        name=entry["name"],
                        gemm_tflops=float(entry["gemm_tflops"]),
                        membw_gbs=float(entry["membw_gbs"]),
                        gemm_fp8_tflops=entry.get("gemm_fp8_tflops"),
                        source=entry.get("source", "stage-0"),
                    )
                )
                membw.append(float(entry["membw_gbs"]))
            if scores:
                measured = tuple(scores)
                base_plan = inputs.rank_tp_ratio or [1] * tp
                model = PerfCostModel(
                    inputs, base_plan, inputs.effective_vram_mib or [1024] * tp
                )
                knee_ok, _ = model.decode_knee_detail(
                    inputs.rank_mlp_ratio or base_plan, membw
                )
    except Exception:
        # No NVML / no torch / no cache: measured stays ABSENT (never a
        # fallback estimate).
        measured = None
        knee_ok = None

    return Advantage(
        stock=stock,
        capacity_pct_range=capacity_pct_range,
        measured=measured,
        decode_knee_ok=knee_ok,
    )
