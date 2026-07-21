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
"""The fit check: AUTO plan and manual-edit validation (design §2.6).

``plan()`` is the single entry point for both modes — "auto" just means "no
override supplied for a knob → derive it"; a manual edit fills the
corresponding ``PlanInputs`` field and re-runs the SAME validation +
capacity code, so an invalid manual layout is rejected with a reason, never
silently accepted.

The structural validation rules mirror ``ServerArgs._handle_uneven_tp``
(lengths, positivity, list-budgets-require-ratio, the physical-impossibility
check "sum of co-located budgets <= card total") so a plan this module
accepts is a plan the server accepts.

Everything here is capacity/feasibility — infeasibility is reported
FAIL-LOUD, naming the specific card and the specific limit.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

from sglang.srt.planner.capacity import CapacityReport, predict_capacity
from sglang.srt.planner.hardware import HardwareSpec
from sglang.srt.planner.plan import AutoPlan, derive_auto_plan
from sglang.srt.uneven_perf import PlanInputs

__all__ = [
    "PlanResult",
    "PlanRejected",
    "OffloadAssessment",
    "plan",
    "validate_plan_inputs",
]


class PlanRejected(ValueError):
    """A structurally invalid plan (manual edit or CLI input), with the
    reasons joined into the message. ``reasons`` keeps them separate for
    UIs (design §2.6: every rejection carries its reason)."""

    def __init__(self, reasons: Sequence[str]):
        self.reasons = list(reasons)
        super().__init__("\n".join(self.reasons))


@dataclasses.dataclass(frozen=True)
class OffloadAssessment:
    """Whether a plan that does NOT fit purely in VRAM can still run by tiering
    weights to host RAM (design §PART 4). Three honest states:

      * ``"vram"``        — fits entirely in VRAM (fast; no host tier needed).
      * ``"ram_offload"`` — does not fit in VRAM, but moving ``offloaded_gib``
        of MoE routed experts to a pinned host pool makes the resident set fit
        AND the host has the RAM. SLOWER: those experts are fetched over PCIe
        on demand (bandwidth-bound), so this is a capacity win, not a speed
        one — stated plainly, never sugar-coated.
      * ``"cannot_fit"``  — even offloading every offloadable byte (or the host
        lacks the RAM) leaves the plan infeasible. The non-offloadable floor
        (attention + dense/shared MLP + embeddings + GDN state + min KV)
        already exceeds VRAM, or there is no host RAM headroom.
    """

    status: str  # "vram" | "ram_offload" | "cannot_fit"
    #: Weight GiB that must move to host RAM to fit (0.0 for "vram").
    offloaded_gib: float
    #: Host RAM the offloaded experts need, and what is available.
    host_ram_needed_mib: int
    host_ram_total_mib: Optional[int]
    #: Per-rank offloaded GiB (which rank sheds how much).
    per_rank_offloaded_gib: List[float]
    #: Human note (the honest tradeoff / why it cannot fit).
    note: str


@dataclasses.dataclass(frozen=True)
class PlanResult:
    """The planner's answer. Capacity numbers are estimates (labelled);
    there is NO throughput field anywhere in this type tree (design §3.4)."""

    fits: bool
    #: Fail-loud reasons when ``fits`` is False (specific card + limit).
    infeasible_reasons: List[str]
    inputs: PlanInputs
    hardware: HardwareSpec
    auto_plan: Optional[AutoPlan]
    capacity: Optional[CapacityReport]
    #: The launch flags this plan maps to (the §2.5 "honored identically at
    #: boot" contract) — one string per flag/env token.
    launch_flags: List[str]
    #: Honest advantage vs stock even-TP (feasibility + capacity-%; see
    #: sglang.srt.planner.advantage). Optional import cycle avoidance: typed
    #: loosely.
    advantage: Optional[object] = None
    #: Host-RAM-offload assessment (§PART 4): distinguishes fits-in-VRAM from
    #: fits-with-RAM-offload from genuinely-cannot-fit. None only when capacity
    #: could not be sized at all.
    offload: Optional[OffloadAssessment] = None
    #: Roofline throughput ESTIMATE (design #145) — a rough, clearly-labelled
    #: prefill+decode tok/s ballpark carrying ``provenance="planner-estimate"``,
    #: ADDITIVE and separate from the capacity/feasibility answer (never a
    #: measured number). None when it cannot be sized (unknown card peaks). Typed
    #: loosely to avoid an import cycle (sglang.srt.planner.roofline).
    roofline: Optional[object] = None


# ---------------------------------------------------------------------------
# Structural validation (mirror of _handle_uneven_tp, pure).
# ---------------------------------------------------------------------------


def validate_plan_inputs(
    inputs: PlanInputs, hardware: HardwareSpec
) -> List[str]:
    """All structural violations (empty list = valid). Mirrors the server's
    fail-fast rules so auto plans and manual edits share one code path."""
    errors: List[str] = []
    tp = inputs.tp_size
    if tp < 1:
        return [f"tp_size must be >= 1, got {tp}."]

    rank_gpu_id = inputs.rank_gpu_id
    if rank_gpu_id is not None:
        if len(rank_gpu_id) != tp:
            errors.append(
                f"--rank-gpu-id length ({len(rank_gpu_id)}) must equal "
                f"--tp-size ({tp})."
            )
        if any(g < 0 for g in rank_gpu_id):
            errors.append(
                f"--rank-gpu-id entries must be >= 0, got {rank_gpu_id}."
            )
        known = {g.index for g in hardware.gpus}
        for gid in sorted(set(rank_gpu_id)):
            if gid not in known:
                errors.append(
                    f"--rank-gpu-id names GPU {gid}, but the hardware spec "
                    f"({hardware.source}) only declares "
                    f"{sorted(known)}: "
                    + ", ".join(
                        f"[{g.index}] {g.name} {g.total_mib} MiB"
                        for g in hardware.gpus
                    )
                    + "."
                )

    ratio = inputs.rank_tp_ratio
    if ratio is not None:
        if len(ratio) != tp:
            errors.append(
                f"--rank-tp-ratio length ({len(ratio)}) must equal "
                f"--tp-size ({tp})."
            )
        elif any((not isinstance(r, int)) or r <= 0 for r in ratio):
            errors.append(
                f"--rank-tp-ratio entries must be positive integers, got {ratio}."
            )

    budgets = inputs.effective_vram_mib
    if budgets is not None:
        if len(budgets) != tp:
            errors.append(
                f"per-rank memory budgets have {len(budgets)} entries but "
                f"--tp-size is {tp}."
            )
        elif any(b <= 0 for b in budgets):
            errors.append(f"per-rank memory budgets must be positive: {budgets}.")
        elif (
            len(set(budgets)) > 1
            and ratio is None
        ):
            errors.append(
                "--rank-gpu-memory-mib as a per-rank list requires "
                "--rank-tp-ratio (with even TP all ranks are structurally "
                "equal — use a single scalar)."
            )

    for name, vec in (
        ("--rank-mlp-ratio", inputs.rank_mlp_ratio),
        ("--rank-moe-ratio", inputs.rank_moe_ratio),
        ("--rank-vocab-ratio", inputs.rank_vocab_ratio),
        ("KV token vector", inputs.kv_token_vector),
    ):
        if vec is None:
            continue
        if len(vec) != tp:
            errors.append(
                f"{name} length ({len(vec)}) must equal --tp-size ({tp})."
            )
        elif any((not isinstance(v, int)) or v <= 0 for v in vec):
            errors.append(
                f"{name} entries must be positive integers, got {vec}."
            )

    if inputs.dcp_size is not None and not (1 <= inputs.dcp_size <= tp):
        errors.append(
            f"--dcp-size must be in [1, tp_size={tp}], got {inputs.dcp_size}."
        )

    # Physical impossibility (mirror of _handle_uneven_tp step 6): the
    # summed budgets of the ranks co-located on one physical card must fit
    # its total VRAM. Checked against the PHYSICAL total, not the budgeted
    # one, so an over-carved manual budget fails fast with the real limit.
    if rank_gpu_id is not None and budgets is not None and not errors:
        for gpu_id in sorted(set(rank_gpu_id)):
            ranks = [r for r, g in enumerate(rank_gpu_id) if g == gpu_id]
            required = sum(budgets[r] for r in ranks)
            card = hardware.gpu(gpu_id)
            if required > card.total_mib:
                errors.append(
                    f"Physical impossibility: GPU {gpu_id} ({card.name}, "
                    f"total {card.total_mib} MiB) cannot fit ranks {ranks} "
                    f"whose budgets sum to {required} MiB. Reduce the "
                    "budgets or place fewer ranks on this GPU."
                )
    return errors


# ---------------------------------------------------------------------------
# The single planning entry (auto AND manual-edit — same code path).
# ---------------------------------------------------------------------------


def plan(
    model_path: str,
    hardware: HardwareSpec,
    *,
    tp_size: Optional[int] = None,
    rank_gpu_id: Optional[Sequence[int]] = None,
    rank_gpu_memory_mib: Optional[Sequence[int]] = None,
    rank_tp_ratio: Optional[Sequence[int]] = None,
    rank_mlp_ratio: Optional[Sequence[int]] = None,
    rank_moe_ratio: Optional[Sequence[int]] = None,
    rank_vocab_ratio: Optional[Sequence[int]] = None,
    dcp_size: Optional[int] = None,
    kv_token_vector: Optional[Sequence[int]] = None,
    user_free_reserve_mib: Optional[Sequence[int]] = None,
    kv_cache_dtype: str = "auto",
    speculative_algorithm: Optional[str] = None,
    speculative_num_draft_tokens: Optional[int] = None,
    speculative_draft_model_path: Optional[str] = None,
    max_running_requests: Optional[int] = None,
    disable_cuda_graph: bool = False,
    host_ram_mib: Optional[int] = None,
    include_vision: bool = True,
    with_advantage: bool = True,
    roofline_context_tokens: int = 4096,
) -> PlanResult:
    """Plan ``model_path`` on ``hardware``.

    Every keyword that stays ``None`` is auto-derived; passing it is the
    manual edit of design §2.6 and goes through the same validation.
    Raises :class:`PlanRejected` for structurally invalid inputs; returns
    ``fits=False`` with fail-loud reasons for valid-but-infeasible plans.
    """
    if not hardware.gpus:
        raise PlanRejected(["The hardware spec declares zero GPUs."])

    # Placement defaults: one rank per declared card.
    if rank_gpu_id is None:
        if tp_size is None:
            tp_size = len(hardware.gpus)
        if tp_size > len(hardware.gpus):
            raise PlanRejected(
                [
                    f"--tp-size {tp_size} exceeds the {len(hardware.gpus)} "
                    "declared card(s); co-location needs an explicit "
                    "--rank-gpu-id map (duplicates = ranks sharing a card)."
                ]
            )
        rank_gpu_id = [g.index for g in hardware.gpus[:tp_size]]
    else:
        rank_gpu_id = [int(g) for g in rank_gpu_id]
        if tp_size is None:
            tp_size = len(rank_gpu_id)

    inputs = PlanInputs(
        tp_size=tp_size,
        model_path=model_path,
        kv_cache_dtype=kv_cache_dtype,
        speculative_algorithm=speculative_algorithm,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        speculative_draft_model_path=speculative_draft_model_path,
        max_running_requests=max_running_requests,
        disable_cuda_graph=disable_cuda_graph,
        include_vision=include_vision,
        rank_gpu_id=list(rank_gpu_id),
        effective_vram_mib=(
            [int(b) for b in rank_gpu_memory_mib]
            if rank_gpu_memory_mib is not None
            else None
        ),
        rank_tp_ratio=(list(rank_tp_ratio) if rank_tp_ratio else None),
        rank_mlp_ratio=(list(rank_mlp_ratio) if rank_mlp_ratio else None),
        rank_moe_ratio=(list(rank_moe_ratio) if rank_moe_ratio else None),
        rank_vocab_ratio=(list(rank_vocab_ratio) if rank_vocab_ratio else None),
        dcp_size=dcp_size,
        kv_token_vector=(list(kv_token_vector) if kv_token_vector else None),
    )

    # ---- budgets: manual (validated) or auto-derived --------------------
    auto = None
    if inputs.effective_vram_mib is None:
        auto = derive_auto_plan(inputs, hardware, user_free_reserve_mib)
        inputs.effective_vram_mib = list(auto.budgets_mib)
        if inputs.rank_tp_ratio is None:
            inputs.rank_tp_ratio = (
                list(auto.rank_tp_ratio) if auto.rank_tp_ratio else None
            )

    errors = validate_plan_inputs(inputs, hardware)
    if errors:
        raise PlanRejected(errors)

    # ---- capacity via the server's own cost model -----------------------
    base_plan = inputs.rank_tp_ratio or [1] * inputs.tp_size
    try:
        capacity = predict_capacity(
            inputs, base_plan, inputs.effective_vram_mib
        )
    except ValueError as e:
        # partition_units & friends raise when the geometry cannot shard
        # (e.g. fewer KV heads than ranks) — report, don't crash.
        return PlanResult(
            fits=False,
            infeasible_reasons=[
                f"The cost model cannot size this plan "
                f"({inputs.tp_size} ranks): {e}"
            ],
            inputs=inputs,
            hardware=hardware,
            auto_plan=auto,
            capacity=None,
            launch_flags=[],
        )

    reasons: List[str] = []
    if not capacity.feasible:
        from sglang.srt.uneven_perf import _PREDICT_MIN_RANK_TOKENS

        for rc in capacity.per_rank:
            if rc.kv_tokens >= _PREDICT_MIN_RANK_TOKENS:
                continue
            card = hardware.gpu(rc.gpu_index)
            need_gib = (
                rc.weight_gib
                + rc.mamba_gib
                + capacity.overhead_mib / 1024.0
                + _PREDICT_MIN_RANK_TOKENS
                * _kv_cell_bytes(inputs)
                / 2**30
            )
            reasons.append(
                f"rank {rc.rank} on GPU {rc.gpu_index} ({card.name}, budget "
                f"{rc.budget_mib} MiB = {rc.budget_mib / 1024:.1f} GiB): "
                f"weights {rc.weight_gib:.1f} GiB + mamba "
                f"{rc.mamba_gib:.1f} GiB + overhead "
                f"{capacity.overhead_mib / 1024:.1f} GiB leave "
                f"{max(rc.kv_tokens, 0):.0f} KV tokens "
                f"(< the minimum viable {_PREDICT_MIN_RANK_TOKENS}); this "
                f"rank needs >= {need_gib:.1f} GiB. Estimate — the runtime "
                "measures real free bytes."
            )
        if not reasons:
            reasons.append(
                "The predicted capacity is below the viable minimum "
                "(cost-model estimate)."
            )

    offload = _assess_offload(
        inputs, hardware, capacity, host_ram_mib=host_ram_mib
    )
    result = PlanResult(
        fits=capacity.feasible,
        infeasible_reasons=reasons,
        inputs=inputs,
        hardware=hardware,
        auto_plan=auto,
        capacity=capacity,
        launch_flags=_launch_flags(inputs),
        offload=offload,
    )
    if with_advantage:
        from sglang.srt.planner.advantage import compute_advantage

        result = dataclasses.replace(
            result,
            advantage=compute_advantage(
                inputs, hardware, capacity, user_free_reserve_mib
            ),
        )

    # Roofline throughput ESTIMATE (design #145) — additive, explicitly separate
    # from the capacity/feasibility answer. Guarded: a failure here (unknown card
    # peaks, an un-shardable geometry) must never break the plan, so it degrades
    # to an absent estimate. Sized whenever capacity exists (fits in VRAM OR
    # fits with RAM-offload — the offloaded MoE case is exactly where the PCIe
    # -fetch term matters). Measured micro-probe scores (advantage) win over the
    # nameplate peaks inside the estimator.
    if capacity is not None:
        try:
            from sglang.srt.planner.roofline import estimate_roofline

            measured_scores = None
            if result.advantage is not None:
                measured_scores = getattr(result.advantage, "measured", None)
            roofline = estimate_roofline(
                inputs,
                hardware,
                capacity,
                offload,
                measured_scores=measured_scores,
                context_tokens=roofline_context_tokens,
            )
            result = dataclasses.replace(result, roofline=roofline)
        except Exception:
            result = dataclasses.replace(result, roofline=None)
    return result


def _assess_offload(
    inputs: PlanInputs,
    hardware: HardwareSpec,
    capacity: Optional[CapacityReport],
    host_ram_mib: Optional[int] = None,
) -> Optional[OffloadAssessment]:
    """Decide fits-in-VRAM vs fits-with-RAM-offload vs cannot-fit (§PART 4).

    Only the MoE routed-expert bytes are treated as host-offloadable (the
    fork's #77 expert-offload); everything else — attention, dense/shared MLP,
    embeddings, GDN state, and a minimum viable KV slice — must stay resident.
    We move as few experts to host as needed for the resident set to fit, and
    only call it feasible when the host actually has that much RAM. No safety
    margin is invented on top; the honest PCIe-bandwidth caveat is in the note.
    """
    if capacity is None:
        return None
    host_total = host_ram_mib if host_ram_mib is not None else hardware.host_ram_mib

    if capacity.feasible:
        return OffloadAssessment(
            status="vram",
            offloaded_gib=0.0,
            host_ram_needed_mib=0,
            host_ram_total_mib=host_total,
            per_rank_offloaded_gib=[0.0] * len(capacity.per_rank),
            note="Fits entirely in VRAM — no host-RAM tier needed (fastest).",
        )

    from sglang.srt.uneven_perf import _PREDICT_MIN_RANK_TOKENS

    kv_cell = _kv_cell_bytes(inputs)
    min_kv_bytes = _PREDICT_MIN_RANK_TOKENS * kv_cell
    overhead_bytes = capacity.overhead_mib * 2**20

    per_rank_off_gib: List[float] = []
    total_off_bytes = 0.0
    total_offloadable = 0.0
    hard_floor_exceeds = False
    for rc in capacity.per_rank:
        budget = rc.budget_mib * 2**20
        offloadable = rc.offloadable_weight_gib * 2**30
        total_offloadable += offloadable
        full_resident = (
            rc.weight_gib * 2**30
            + rc.mamba_gib * 2**30
            + overhead_bytes
            + min_kv_bytes
        )
        need = full_resident - budget  # bytes that must leave this rank
        if need <= 0:
            per_rank_off_gib.append(0.0)
            continue
        if need > offloadable + 1:  # even shedding all experts is not enough
            hard_floor_exceeds = True
            per_rank_off_gib.append(offloadable / 2**30)
            total_off_bytes += offloadable
        else:
            per_rank_off_gib.append(need / 2**30)
            total_off_bytes += need

    host_needed_mib = int(total_off_bytes / 2**20)

    if hard_floor_exceeds or total_offloadable <= 0:
        why = (
            "no host-offloadable weight class here (a dense checkpoint — only "
            "MoE routed experts can tier to host RAM)"
            if total_offloadable <= 0
            else "the non-offloadable floor (attention + dense/shared MLP + "
            "embeddings + GDN state + minimum KV) already exceeds VRAM even "
            "with every expert on host"
        )
        return OffloadAssessment(
            status="cannot_fit",
            offloaded_gib=total_off_bytes / 2**30,
            host_ram_needed_mib=host_needed_mib,
            host_ram_total_mib=host_total,
            per_rank_offloaded_gib=per_rank_off_gib,
            note=f"Cannot fit: {why}.",
        )

    if host_total is not None and host_needed_mib > host_total:
        return OffloadAssessment(
            status="cannot_fit",
            offloaded_gib=total_off_bytes / 2**30,
            host_ram_needed_mib=host_needed_mib,
            host_ram_total_mib=host_total,
            per_rank_offloaded_gib=per_rank_off_gib,
            note=(
                f"Would fit with ~{total_off_bytes / 2**30:.1f} GiB of experts "
                f"on host RAM, but the host has only "
                f"{host_total / 1024:.1f} GiB total — not enough headroom."
            ),
        )

    host_note = (
        f" (host has {host_total / 1024:.1f} GiB)"
        if host_total is not None
        else " (host RAM total unknown — pass one to confirm)"
    )
    return OffloadAssessment(
        status="ram_offload",
        offloaded_gib=total_off_bytes / 2**30,
        host_ram_needed_mib=host_needed_mib,
        host_ram_total_mib=host_total,
        per_rank_offloaded_gib=per_rank_off_gib,
        note=(
            f"Fits with ~{total_off_bytes / 2**30:.1f} GiB of MoE routed "
            f"experts offloaded to host RAM{host_note}. SLOWER than a pure-VRAM "
            "fit: offloaded experts are fetched over PCIe on demand "
            "(bandwidth-bound) — requires SGLANG_MOE_RESIDENT_EXPERT_FRACTION "
            "< 1 and --disable-cuda-graph."
        ),
    )


def _kv_cell_bytes(inputs: PlanInputs) -> int:
    """Full-kv-head KV cell bytes/token, read from the same cost model (for
    the fail-loud message's "needs >= X GiB" arithmetic only)."""
    from sglang.srt.uneven_perf import PerfCostModel

    return PerfCostModel(
        inputs, [1] * inputs.tp_size, [1024] * inputs.tp_size
    ).kv_cell_bytes


def _launch_flags(inputs: PlanInputs) -> List[str]:
    """The real server-args this plan maps to (design §2.5 table) — what a
    user copies into the launch command so the boot path honors the exact
    budgets the planner promised."""
    flags = [f"--tp-size {inputs.tp_size}"]
    if inputs.rank_gpu_id:
        flags.append(
            "--rank-gpu-id " + ",".join(str(g) for g in inputs.rank_gpu_id)
        )
    budgets = inputs.effective_vram_mib
    if budgets:
        if len(set(budgets)) == 1:
            flags.append(f"--rank-gpu-memory-mib {budgets[0]}")
        else:
            flags.append(
                "--rank-gpu-memory-mib " + ",".join(str(b) for b in budgets)
            )
    if inputs.rank_tp_ratio:
        flags.append(
            "--rank-tp-ratio " + ",".join(str(r) for r in inputs.rank_tp_ratio)
        )
        # Uneven ratios shard the KV cache on the token axis (uneven DCP).
        flags.append("SGLANG_UNEVEN_DCP=1")
    if inputs.rank_mlp_ratio:
        flags.append(
            "--rank-mlp-ratio " + ",".join(str(r) for r in inputs.rank_mlp_ratio)
        )
    if inputs.rank_moe_ratio:
        flags.append(
            "--rank-moe-ratio " + ",".join(str(r) for r in inputs.rank_moe_ratio)
        )
    if inputs.rank_vocab_ratio:
        flags.append(
            "--rank-vocab-ratio "
            + ",".join(str(r) for r in inputs.rank_vocab_ratio)
        )
    if inputs.dcp_size is not None:
        flags.append(f"--dcp-size {inputs.dcp_size}")
    if inputs.kv_token_vector:
        flags.append(
            "SGLANG_UNEVEN_TOKEN_VECTOR="
            + ",".join(str(v) for v in inputs.kv_token_vector)
        )
    return flags
