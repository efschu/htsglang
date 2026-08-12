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
"""derive_auto_plan — the pure lift of ``ServerArgs._resolve_auto_rank_tp_ratio``.

The budget/ratio ALGORITHM is reused (design §1b):

    budget[rank] = (total − user_free_reserve − auto_reserve
                    [, capped at free − 1024 when live free is known])
                   // colocated_ranks
    weights      = gcd-reduced budgets;  uniform weights => even split

with the reserve formulas taken from the REAL ServerArgs methods
(``derived_rank_auto_reserve_mib`` / ``_apply_gpu_mem_capacity_defaults`` /
``mamba_pre_capture_reserve_mb`` / ``speculative_capture_token_multiplier``),
called unbound on a small attribute view — never re-implemented — so a change
to the server's reserve formula changes the planner with it (design §5
mitigation 1).

Design §2.5 (per-card free-VRAM budget): the user's per-card free-reserve is
an EXTERNAL headroom term added on top of the internal auto reserve; every
capacity number downstream is "max possible under your set budget".
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

from sglang.srt.planner.hardware import HardwareSpec

__all__ = ["AutoPlan", "derive_auto_plan", "auto_reserve_mib"]


@dataclasses.dataclass(frozen=True)
class AutoPlan:
    """The derived placement/budget plan (before capacity prediction)."""

    #: Per-rank byte budget (MiB) — the effective_vram_mib of PlanInputs.
    budgets_mib: List[int]
    #: gcd-reduced integer weights, or None when the budgets are uniform
    #: (collapse to the classic even split, exactly like the boot path).
    rank_tp_ratio: Optional[List[int]]
    #: Per physical GPU: the internal auto reserve (MiB) that was subtracted.
    auto_reserve_per_gpu: Dict[int, int]
    #: Per physical GPU: the user's free-reserve (MiB) that was subtracted.
    user_reserve_per_gpu: Dict[int, int]


class _ServerArgsView:
    """A minimal attribute view for calling the REAL ServerArgs reserve
    methods unbound (single source of truth — no formula duplication).

    Only the attributes those methods actually read exist here; adding a new
    dependency to the server-side formula makes this view fail loudly (an
    AttributeError) instead of silently diverging.
    """

    def __init__(self, plan_inputs, cuda_graph_config):
        self.tp_size = plan_inputs.tp_size
        self.pp_size = 1  # planner scope: pure single-node TP
        self.chunked_prefill_size = None  # filled by the tier defaults
        self.max_prefill_tokens = 16384  # ServerArgs default
        self.max_running_requests = plan_inputs.max_running_requests
        self.disaggregation_mode = "null"
        self.disable_cuda_graph = plan_inputs.disable_cuda_graph
        self.speculative_algorithm = plan_inputs.speculative_algorithm
        self.speculative_num_draft_tokens = (
            plan_inputs.speculative_num_draft_tokens
        )
        self.speculative_draft_model_path = (
            plan_inputs.speculative_draft_model_path
        )
        self.cuda_graph_config = cuda_graph_config
        self.device = "cuda"


def _make_view(plan_inputs) -> _ServerArgsView:
    # Heavy import chain (torch via sglang.srt.server_args) — deliberately
    # function-local so importing the planner package stays light.
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
    from sglang.srt.server_args import ServerArgs

    view = _ServerArgsView(plan_inputs, CudaGraphConfig())
    # Bind the real methods (unbound reuse, not a re-implementation).
    view._apply_gpu_mem_capacity_defaults = (
        ServerArgs._apply_gpu_mem_capacity_defaults.__get__(view)
    )
    view.mamba_pre_capture_reserve_mb = (
        ServerArgs.mamba_pre_capture_reserve_mb.__get__(view)
    )
    view.speculative_capture_token_multiplier = (
        ServerArgs.speculative_capture_token_multiplier.__get__(view)
    )
    view.speculative_capture_tokens = ServerArgs.speculative_capture_tokens.__get__(
        view
    )
    # #590 added a ``runtime_reserve_mib`` hop inside
    # ``derived_rank_auto_reserve_mib``; without this binding the view raises
    # AttributeError, which is the loud failure this class promises but which
    # nothing had yet answered. The planner calls it with no card UUID, and
    # that branch returns ``mamba_pre_capture_reserve_mb`` (bound above), so
    # binding it keeps the planner's numbers byte-identical to the formula's
    # documented "heuristic:not-asked" path rather than re-deriving anything.
    view.runtime_reserve_mib = ServerArgs.runtime_reserve_mib.__get__(view)
    view.derived_rank_auto_reserve_mib = (
        ServerArgs.derived_rank_auto_reserve_mib.__get__(view)
    )
    return view


def auto_reserve_mib(plan_inputs, gpu_mem_mib: int, colocated_ranks: int) -> int:
    """The server's ``--rank-auto-reserve-mib auto`` value (#68) for one
    physical GPU, computed by the REAL ServerArgs formula.

    Without the adaptive-ladder term (#313): a plan carries no adaptive
    config and no draft placement, so the ladder's rungs are not knowable
    here. A plan for a ladder boot therefore reads LOW by that GPU's ladder
    demand, which the boot itself derives and logs.
    """
    view = _make_view(plan_inputs)
    return int(view.derived_rank_auto_reserve_mib(gpu_mem_mib, colocated_ranks))


def derive_auto_plan(
    plan_inputs,
    hardware: HardwareSpec,
    user_free_reserve_mib: Optional[Sequence[int]] = None,
) -> AutoPlan:
    """Derive per-rank budgets + tp ratio for ``plan_inputs.rank_gpu_id`` on
    ``hardware`` — the offline mirror of ``--rank-tp-ratio auto``.

    ``user_free_reserve_mib``: optional per-CARD external headroom (aligned
    with ``hardware.gpus`` order; scalar lists are the CLI's job), design
    §2.5. The live ``free − 1024`` cap of the boot path applies only when
    the hardware source knows ``free_mib`` (NVML); manual specs plan against
    the budgeted total alone.
    """
    rank_gpu_id = plan_inputs.rank_gpu_id
    if not rank_gpu_id:
        raise ValueError("derive_auto_plan requires PlanInputs.rank_gpu_id.")
    if len(rank_gpu_id) != plan_inputs.tp_size:
        raise ValueError(
            f"--rank-gpu-id length ({len(rank_gpu_id)}) must equal "
            f"--tp-size ({plan_inputs.tp_size})."
        )

    user_reserve_per_gpu: Dict[int, int] = {}
    if user_free_reserve_mib is not None:
        if len(user_free_reserve_mib) != len(hardware.gpus):
            raise ValueError(
                f"--plan-free-reserve-gb has {len(user_free_reserve_mib)} "
                f"entries but the hardware spec declares "
                f"{len(hardware.gpus)} card(s); pass one value per card "
                "(or a single value for all)."
            )
        for g, res in zip(hardware.gpus, user_free_reserve_mib):
            if res < 0:
                raise ValueError(
                    f"--plan-free-reserve-gb for GPU {g.index} ({g.name}) "
                    "must be >= 0."
                )
            # Keyed in the --rank-gpu-id (CUDA) space, because that is what
            # the lookup below uses; keying on the NVML index would charge
            # another card's reserve on a rig where the orders differ (#392).
            user_reserve_per_gpu[hardware.rank_gpu_id_of(g)] = int(res)

    counts = Counter(rank_gpu_id)
    # Boot mirror: _resolve_auto_rank_tp_ratio computes the tier defaults
    # from get_device_memory_capacity(self.device) — the FIRST visible
    # device's total, i.e. CUDA ordinal 0. hardware.gpu(0) resolves that
    # ordinal; hardware.gpus[0] is the first card in NVML order, a different
    # card on a rig where the enumerations diverge (#392).
    try:
        tier_gpu_mem = hardware.gpu(0).total_mib
    except ValueError:
        tier_gpu_mem = hardware.gpus[0].total_mib
    view = _make_view(plan_inputs)
    auto_reserve_per_gpu: Dict[int, int] = {
        gpu_id: int(view.derived_rank_auto_reserve_mib(tier_gpu_mem, cnt))
        for gpu_id, cnt in counts.items()
    }

    budgets: List[int] = []
    for gpu_id in rank_gpu_id:
        card = hardware.gpu(gpu_id)  # raises with the card list if unknown
        reserve = auto_reserve_per_gpu[gpu_id] + user_reserve_per_gpu.get(
            gpu_id, 0
        )
        budget = card.total_mib - reserve
        if card.free_mib is not None:
            # Same live-free cap as the boot path (worker checks free AFTER
            # creating its own CUDA context).
            budget = min(budget, card.free_mib - 1024)
        budget //= counts[gpu_id]
        if budget <= 0:
            raise ValueError(
                f"No budget left on GPU {gpu_id} ({card.name}, NVML total "
                f"{card.total_mib} MiB, {counts[gpu_id]} rank(s)): auto "
                f"reserve {auto_reserve_per_gpu[gpu_id]} MiB + your "
                f"free-reserve {user_reserve_per_gpu.get(gpu_id, 0)} MiB "
                "consume the card. Lower --plan-free-reserve-gb or move "
                "ranks off this GPU."
            )
        budgets.append(int(budget))

    g = math.gcd(*budgets)
    weights = [b // g for b in budgets]
    if len(set(weights)) == 1:
        # Uniform budgets: collapse to the classic even split, and (exactly
        # like the boot path) a scalar budget — min() is lossless here.
        return AutoPlan(
            budgets_mib=[min(budgets)] * len(budgets),
            rank_tp_ratio=None,
            auto_reserve_per_gpu=auto_reserve_per_gpu,
            user_reserve_per_gpu=user_reserve_per_gpu,
        )
    return AutoPlan(
        budgets_mib=budgets,
        rank_tp_ratio=weights,
        auto_reserve_per_gpu=auto_reserve_per_gpu,
        user_reserve_per_gpu=user_reserve_per_gpu,
    )
