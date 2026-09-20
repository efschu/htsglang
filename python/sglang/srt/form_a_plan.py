# SPDX-License-Identifier: Apache-2.0
"""Form A -- the attention-host decode layout -- as one pure, testable solve.

FORM A, in one sentence: ONE card (the attention HOST) runs every dense part
of the model -- attention and the QSA indexer, the GDN/linear-attention path,
o_proj, the hyper-connection mixer, the norms, embeddings, PLE, lm_head, the
whole KV cache for the full context, the GDN recurrent states, the MTP draft
and the CUDA graphs -- while the other cards are pure expert WORKERS: they
hold nothing but their own share of the MoE experts and compute them for the
rows the host sends them. Per MoE layer the host broadcasts the MoE input
(a few rows), every rank computes its own experts, and the partial sums are
reduced back into the host.

Why this module exists, and why it is arithmetic and not policy: the layout
lives or dies on ONE question that no boot can answer after the fact -- how
many experts per layer can still be RESIDENT once one card has absorbed the
dense weights, the undivided KV and the unsharded draft. Today (three ranks
sharing everything) 301 of 515 experts per layer are resident. Form A frees
two whole cards for experts but pays for it on the host. Whether that trade
is positive is a subtraction, and this module is that subtraction -- with
every post named, so a wrong answer names the post that was wrong.

What it deliberately does NOT do:
  * it does not read NVML, torch or any device -- every figure is injected,
    so it runs at the desk and in a test without a driver;
  * it does not guess the posts. `MeasuredPosts.from_fn8ah()` carries the
    ONE measured boot they came from, by name, so a stale number is visible
    as a stale boot tag rather than as a plausible float;
  * it never rounds a refusal into a smaller plan. Every impossibility is a
    named exception carrying the numbers, because "it silently fit" is the
    failure mode that costs a GPU window.

Provenance of the default posts: boot fn8ah (2026-09-20 09:12Z,
``/spinning/evidence-665-f1/boot_fn_fn8ah_20260920T091216Z.server.log``),
the ``[vram-census] ... after pools`` lines, the ``KV pool sizing`` line
(``cell_size=14143``) and the ``[ct-stream-presplit]`` resident-buffer lines
(147/75/79 experts per layer at 0.35/0.18/0.19 GiB -> 2.45 MiB per expert).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

__all__ = [
    "FormAInfeasible",
    "FormAHostOverBudget",
    "FormAWorkerOverBudget",
    "FormAGeometryInvalid",
    "FormAOwnershipMismatch",
    "CardBudget",
    "MeasuredPosts",
    "ExpertGeometry",
    "FormAPlan",
    "solve_form_a",
    "RIG_CARDS",
]

_GIB = 1024.0**3
_MIB = 1024.0**2


# --------------------------------------------------------------------------
# Refusals. One class per seam, so a caller can tell "the host does not fit"
# from "the ownership vector does not add up" without parsing a message.
# --------------------------------------------------------------------------
class FormAInfeasible(ValueError):
    """Base: this Form A configuration cannot be booted as described."""


class FormAHostOverBudget(FormAInfeasible):
    """The attention host's fixed posts alone exceed its VRAM budget."""


class FormAWorkerOverBudget(FormAInfeasible):
    """An expert worker's fixed posts alone exceed its VRAM budget."""


class FormAGeometryInvalid(FormAInfeasible):
    """The model/rig geometry handed in is not a Form A geometry."""


class FormAOwnershipMismatch(FormAInfeasible):
    """The ownership vector does not cover every expert exactly once."""


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CardBudget:
    """One card as Form A sees it.

    `budget_mib` is the ``--rank-gpu-memory-mib`` entry for this rank: the
    ENTIRE budget the rank may use, already net of the user reserve
    (``--rank-user-reserve-mib``), which is kept separately only so a plan
    can print where the nameplate went.

    `link_gib_s` is the MEASURED host-to-device rate of that card's PCIe
    lane, not its nominal width -- on this rig rank 1 sits on x4 and moves
    6.5 GiB/s while rank 2 sits on x8 and moves 13.3. It is the weight of
    the spill split: the slow lane is handed FEWER non-resident experts,
    because a spilled expert costs a fetch over exactly that lane.
    """

    rank: int
    name: str
    nameplate_mib: int
    budget_mib: int
    reserve_mib: int
    link_gib_s: float
    role: str = "worker"  # "host" | "worker"

    @property
    def budget_gib(self) -> float:
        return self.budget_mib / 1024.0


@dataclass(frozen=True)
class MeasuredPosts:
    """The per-rank VRAM posts of Form A, in GiB, with their provenance.

    HOST posts are what one card must carry once nothing is sharded any
    more; WORKER posts are what a card carries when it holds experts and
    nothing else. `corridor_gib` is the free-VRAM floor every rank must
    leave untouched (the ``unaccounted`` term of the KV-budget line); it is
    NOT a reserve we choose, it is one we measured.
    """

    # host-only
    dense_gib: float
    kv_bytes_per_token: int
    context_tokens: int
    draft_gib: float
    gdn_state_gib: float
    spec_state_gib: float
    host_runtime_gib: float
    # every rank
    worker_runtime_gib: float
    corridor_gib: float
    # worker-only: the double-buffered MoE row exchange
    dispatch_buffer_gib: float
    source: str = "unset"

    @property
    def kv_gib(self) -> float:
        return self.kv_bytes_per_token * self.context_tokens / _GIB

    @classmethod
    def from_fn8ah(cls, context_tokens: int = 262151) -> "MeasuredPosts":
        """The posts as boot fn8ah measured them (mixer in INT8).

        dense_gib is the SUM of the three ranks' non-expert model tensors
        minus the double-counted replicas: replicated per rank are
        hyper_connection 0.63 + embed_tokens 0.20 + lm_head 0.20 +
        moe_gate 0.12 + ple 0.03 = 1.18 GiB, sharded across the three are
        linear_attn 1.26+0.38+0.38, other 0.38+0.25+0.25 and shared_expert
        0.14+0.05+0.05 = 3.14 GiB. Host total 1.18 + 3.14 = 4.32 GiB.

        draft_gib likewise: 1.39 + 0.83 + 0.81 GiB across the ranks, less
        the twice-counted replicated embed_tokens and lm_head (2 x 0.40).
        """
        return cls(
            dense_gib=4.32,
            kv_bytes_per_token=14143,
            context_tokens=context_tokens,
            draft_gib=2.23,
            gdn_state_gib=0.10,
            spec_state_gib=0.15,
            host_runtime_gib=3.20,
            worker_runtime_gib=1.45,
            corridor_gib=1.90,
            dispatch_buffer_gib=0.20,
            source="boot fn8ah 2026-09-20T09:12Z (vram-census after pools, "
            "KV pool sizing cell_size=14143, ct-stream-presplit)",
        )


@dataclass(frozen=True)
class ExpertGeometry:
    """The MoE dimension, and what one expert costs on a device."""

    num_experts: int
    num_layers: int
    expert_bytes: int
    pad_experts_per_rank: int = 1
    source: str = "unset"

    @classmethod
    def qwen4_exp(cls) -> "ExpertGeometry":
        """Qwen3.8-Flash-Next (Qwen4Exp), INT4-mixed, as fn8ah loaded it.

        2.45 MiB per expert is measured, not derived: rank 0 held 147
        experts per layer in 0.35 GiB of resident buffers, rank 1 75 in
        0.18 and rank 2 79 in 0.19 (ct-stream-presplit lines).
        """
        return cls(
            num_experts=512,
            num_layers=48,
            expert_bytes=int(2.45 * _MIB),
            pad_experts_per_rank=1,
            source="boot fn8ah ct-stream-presplit resident buffers",
        )

    @property
    def slot_gib(self) -> float:
        """VRAM cost of holding ONE expert on EVERY layer (one 'slot')."""
        return self.num_layers * self.expert_bytes / _GIB


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FormAPlan:
    cards: Tuple[CardBudget, ...]
    posts: MeasuredPosts
    geometry: ExpertGeometry
    fixed_gib: Tuple[float, ...]
    expert_gib: Tuple[float, ...]
    capacity: Tuple[int, ...]
    owned: Tuple[int, ...]
    spill: Tuple[int, ...]
    host_breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def residency(self) -> Tuple[int, ...]:
        """Resident experts PER LAYER per rank (capacity, capped by owned)."""
        return tuple(min(c, o) for c, o in zip(self.capacity, self.owned))

    @property
    def residency_total(self) -> int:
        """The one number Form A is bought for: resident experts per layer
        across the rig, against `geometry.num_experts`."""
        return sum(self.residency)

    @property
    def resident_fraction(self) -> Tuple[float, ...]:
        """``--rank-moe-resident-fraction`` -- resident share of OWNED."""
        return tuple(
            (min(c, o) / o if o else 0.0) for c, o in zip(self.capacity, self.owned)
        )

    def moe_ratio(self) -> List[int]:
        """``--rank-moe-ratio``: the ownership vector, pad experts removed
        (the pad expert is local to every rank and owned by none)."""
        return list(self.owned)

    def flags(self) -> Dict[str, str]:
        """The Form A fragment of a launch line, as flag -> value."""
        return {
            "--rank-moe-ratio": ",".join(str(o) for o in self.owned),
            "--rank-moe-resident-fraction": ",".join(
                f"{f:.3f}" for f in self.resident_fraction
            ),
            "--rank-gpu-memory-mib": ",".join(str(c.budget_mib) for c in self.cards),
            "--rank-user-reserve-mib": ",".join(str(c.reserve_mib) for c in self.cards),
        }

    def report(self) -> str:
        lines = [
            f"Form A plan -- posts: {self.posts.source}",
            f"  geometry: {self.geometry.num_experts} experts x "
            f"{self.geometry.num_layers} layers, "
            f"{self.geometry.expert_bytes / _MIB:.2f} MiB each, "
            f"{self.geometry.slot_gib:.4f} GiB per resident slot",
        ]
        for i, c in enumerate(self.cards):
            lines.append(
                f"  rank {c.rank} {c.name} [{c.role}] budget "
                f"{c.budget_gib:.2f} GiB = fixed {self.fixed_gib[i]:.2f} + "
                f"experts {self.expert_gib[i]:.2f} -> capacity "
                f"{self.capacity[i]} / owned {self.owned[i]} "
                f"({self.resident_fraction[i]* 100:.0f} % resident, "
                f"spill {self.spill[i]} over {c.link_gib_s:.1f} GiB/s)"
            )
        lines.append(
            f"  RESIDENCY CEILING {self.residency_total} of "
            f"{self.geometry.num_experts} experts per layer "
            f"({self.residency_total / self.geometry.num_experts * 100:.0f} %)"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The solve
# --------------------------------------------------------------------------
def _largest_remainder(total: int, weights: Sequence[float]) -> List[int]:
    """Split `total` over `weights` by largest remainder, ties to the lower
    index. A zero weight gets zero -- that is the point here."""
    wsum = float(sum(weights))
    if wsum <= 0:
        raise FormAGeometryInvalid(
            f"cannot split {total} over a weight vector summing to {wsum}."
        )
    quotas = [total * w / wsum for w in weights]
    sizes = [int(q) for q in quotas]
    order = sorted(
        range(len(weights)),
        key=lambda r: (quotas[r] - int(quotas[r]), -r),
        reverse=True,
    )
    for k in range(total - sum(sizes)):
        sizes[order[k % len(order)]] += 1
    return sizes


def solve_form_a(
    cards: Sequence[CardBudget],
    posts: MeasuredPosts,
    geometry: ExpertGeometry,
) -> FormAPlan:
    """Resolve Form A on a concrete rig: per-rank expert capacity, the
    ownership vector and the residency ceiling.

    The order is deliberate and is the argument of the whole layout:

    1. subtract every FIXED post from each rank's budget. The host pays for
       the full dense side, the undivided KV, the unsharded draft, the GDN
       states, the speculative state and its runtime; a worker pays only
       its runtime, its corridor floor and the MoE row buffer.
    2. what is left is expert VRAM. Divide by the cost of one resident slot
       (one expert on every layer) -> per-rank CAPACITY in experts/layer.
    3. the sum of the capacities is the RESIDENCY CEILING. Everything above
       it must be fetched, so the experts above it are SPILL.
    4. split the spill by LINK RATE, not by capacity: a spilled expert is
       paid for on the lane that fetches it, so the x4 card gets fewer.
    5. owned = capacity + spill share. That is ``--rank-moe-ratio``.
    """
    if len(cards) < 2:
        raise FormAGeometryInvalid(
            f"Form A needs at least one host and one worker, got {len(cards)} "
            "card(s)."
        )
    hosts = [c for c in cards if c.role == "host"]
    if len(hosts) != 1:
        raise FormAGeometryInvalid(
            "Form A has exactly ONE attention host; got "
            f"{[c.rank for c in hosts]!r} of {[c.rank for c in cards]!r}. "
            "Two hosts would need the dense side sharded again, which is "
            "the layout Form A replaces."
        )
    if geometry.num_experts <= 0 or geometry.num_layers <= 0:
        raise FormAGeometryInvalid(
            f"empty MoE geometry: {geometry.num_experts} experts, "
            f"{geometry.num_layers} layers."
        )

    host_breakdown = {
        "dense": posts.dense_gib,
        "kv": posts.kv_gib,
        "draft": posts.draft_gib,
        "gdn_states": posts.gdn_state_gib,
        "spec_state": posts.spec_state_gib,
        "runtime": posts.host_runtime_gib,
        "corridor": posts.corridor_gib,
    }
    host_fixed = sum(host_breakdown.values())
    worker_fixed = (
        posts.worker_runtime_gib + posts.corridor_gib + posts.dispatch_buffer_gib
    )

    fixed: List[float] = []
    expert_gib: List[float] = []
    capacity: List[int] = []
    for c in cards:
        f = host_fixed if c.role == "host" else worker_fixed
        left = c.budget_gib - f
        if left <= 0:
            detail = (
                ", ".join(f"{k} {v:.2f}" for k, v in host_breakdown.items())
                if c.role == "host"
                else (
                    f"runtime {posts.worker_runtime_gib:.2f}, "
                    f"corridor {posts.corridor_gib:.2f}, "
                    f"dispatch buffer {posts.dispatch_buffer_gib:.2f}"
                )
            )
            exc = FormAHostOverBudget if c.role == "host" else FormAWorkerOverBudget
            raise exc(
                f"rank {c.rank} ({c.name}, {c.role}): fixed posts "
                f"{f:.2f} GiB exceed its budget {c.budget_gib:.2f} GiB "
                f"by {f - c.budget_gib:.2f} GiB -- no VRAM is left for a "
                f"single expert. Posts: {detail}."
            )
        fixed.append(f)
        expert_gib.append(left)
        capacity.append(int(math.floor(left / geometry.slot_gib)))

    ceiling = sum(capacity)
    total_owned = geometry.num_experts
    spill_total = max(0, total_owned - ceiling)
    if spill_total:
        spill = _largest_remainder(spill_total, [c.link_gib_s for c in cards])
    else:
        spill = [0] * len(cards)

    owned = [cap + s for cap, s in zip(capacity, spill)]
    # Everything is resident: hand the surplus capacity back proportionally
    # so the vector still covers every expert exactly once.
    surplus = sum(owned) - total_owned
    if surplus > 0:
        give_back = _largest_remainder(surplus, [float(o) for o in owned])
        owned = [o - g for o, g in zip(owned, give_back)]
    if sum(owned) != total_owned:
        raise FormAOwnershipMismatch(
            f"ownership vector {owned} sums to {sum(owned)}, not "
            f"{total_owned}: every expert must be owned exactly once, or "
            "the MoE reduce sums a hole."
        )
    if any(o <= 0 for o in owned):
        raise FormAOwnershipMismatch(
            f"ownership vector {owned} leaves a rank with no expert at all; "
            "a rank that owns nothing is not a Form A worker, it is a rank "
            "that should not be in the group."
        )

    return FormAPlan(
        cards=tuple(cards),
        posts=posts,
        geometry=geometry,
        fixed_gib=tuple(fixed),
        expert_gib=tuple(expert_gib),
        capacity=tuple(capacity),
        owned=tuple(owned),
        spill=tuple(max(0, o - c) for o, c in zip(owned, capacity)),
        host_breakdown=host_breakdown,
    )


#: This rig, by NVML index, with the MEASURED lane rates (memory
#: rang-link-zuordnung-rig): rank 0 = RTX 5090 on x8 at 14.4 GiB/s, rank 1 =
#: RTX 3080 on x4 at 6.5, rank 2 = RTX 3080 on x8 at 13.3. The budgets are
#: the ones fn8ah booted with; the 3080 budgets are what a worker WITHOUT
#: KV and dense weights can be raised to, which is why they are a parameter
#: of the call and not a constant here.
def RIG_CARDS(
    host_budget_mib: int = 29500,
    worker_budget_mib: int = 17800,
) -> List[CardBudget]:
    return [
        CardBudget(0, "RTX 5090", 32607, host_budget_mib, 1800, 14.4, role="host"),
        CardBudget(1, "RTX 3080 x4", 20480, worker_budget_mib, 1400, 6.5),
        CardBudget(2, "RTX 3080 x8", 20480, worker_budget_mib, 1400, 13.3),
    ]
