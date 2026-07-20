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
"""Thin wrapper over the fork's ``PerfCostModel`` capacity prediction.

The math is NOT re-implemented: this module constructs the exact
``PerfCostModel`` the server constructs at parse time (same ``PlanInputs``
shape, design §2.1/§5) and reads back ``predict_capacity`` /
``per_rank_weight_bytes``. All absolute token numbers are estimates (the
cost model's own docstring says so); only candidate-over-candidate ratios
are exact — the CLI labels them accordingly.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

__all__ = ["CapacityReport", "RankCapacity", "predict_capacity"]


@dataclasses.dataclass(frozen=True)
class RankCapacity:
    rank: int
    gpu_index: Optional[int]
    budget_mib: int
    weight_gib: float
    mamba_gib: float
    #: Predicted KV token capacity P_r (estimate; may be negative when the
    #: rank does not fit — kept raw so fail-loud messages can show by how
    #: much it misses).
    kv_tokens: float
    #: (weights + mamba + overhead + KV) as a % of this rank's BUDGET when
    #: feasible — the "capacity-%" the rank actually uses of its budget.
    budget_used_pct: float


@dataclasses.dataclass(frozen=True)
class CapacityReport:
    feasible: bool
    #: Predicted max context (converged weighted-DCP optimum; ESTIMATE).
    max_context_tokens: float
    per_rank: List[RankCapacity]
    #: gcd-reduced DCP token-split hint (None when even / not applicable).
    token_vector: Optional[List[int]]
    #: The MLP unit partition the plan materializes.
    mlp_units: List[int]
    #: Overhead constants (MiB) the prediction subtracted per rank — named
    #: so the CLI can show WHERE the bytes went.
    overhead_mib: int


def predict_capacity(plan_inputs, base_plan, budgets_mib) -> CapacityReport:
    """Run the server's parse-time capacity predictor for one plan."""
    # Heavy-ish import chain kept function-local.
    from sglang.srt import uneven_perf
    from sglang.srt.uneven_perf import PerfCostModel

    model = PerfCostModel(plan_inputs, list(base_plan), list(budgets_mib))
    mlp_vector = plan_inputs.rank_mlp_ratio or list(base_plan)
    pred = model.predict_capacity(mlp_vector)

    overhead_mib = int(
        uneven_perf._PREDICT_OVERHEAD_MIB
        + uneven_perf._PREDICT_MAMBA_ACT_RESERVE_MIB
    )
    per_rank = []
    for r in range(plan_inputs.tp_size):
        budget_bytes = budgets_mib[r] * 2**20
        weight_b = pred["weights_gib"][r] * 2**30
        mamba_b = model.mamba_pool_bytes[r]
        kv_tokens = pred["p"][r]
        used = weight_b + mamba_b + overhead_mib * 2**20
        if kv_tokens > 0:
            used += kv_tokens * model.kv_cell_bytes
        per_rank.append(
            RankCapacity(
                rank=r,
                gpu_index=(
                    plan_inputs.rank_gpu_id[r]
                    if plan_inputs.rank_gpu_id
                    else None
                ),
                budget_mib=int(budgets_mib[r]),
                weight_gib=pred["weights_gib"][r],
                mamba_gib=mamba_b / 2**30,
                kv_tokens=kv_tokens,
                budget_used_pct=100.0 * used / budget_bytes,
            )
        )

    # DCP token-split hint via the SAME function the runtime uses.
    token_vector = pred["token_vector"]
    if plan_inputs.kv_token_vector is not None:
        token_vector = list(plan_inputs.kv_token_vector)
    return CapacityReport(
        feasible=bool(pred["feasible"]),
        max_context_tokens=float(pred["ctx"]),
        per_rank=per_rank,
        token_vector=token_vector,
        mlp_units=model.mlp_unit_partition(mlp_vector),
        overhead_mib=overhead_mib,
    )
