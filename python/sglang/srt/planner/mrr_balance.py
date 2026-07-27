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
"""The GDN-state / KV balance point for ``--max-running-requests`` (#253).

On a hybrid GDN model two pools compete for the same VRAM:

* the **mamba/GDN state pool**, which scales with ``max_running_requests``
  (mrr) and is *independent of context length* -- a session's recurrent state
  is the same size whether it holds 200 or 200 000 tokens;
* the **KV pool**, which scales with tokens and therefore with
  ``sessions x context per session``.

Raising mrr admits more concurrent sessions but takes the VRAM for their KV
away. Lowering it hands VRAM back to KV but caps concurrency. Today the
operator picks mrr by hand. This module computes where the two curves cross,
which is where the number of sessions actually carried at a given target
context is maximal.

THE FORMULA
-----------
Per rank ``r`` let

    A_r  = state-pool bytes per SLOT on that rank
    F_r  = budget_r - weights_r - fixed overhead    (the bytes left to split
                                                     between state pool and KV)
    c    = KV bytes per token (the full-KV-head cell, all ranks equal)

The pool is sized like ``ModelRunner._auto_mamba_demand_size``:

    slots(m)    = ceil(m * ratio * safety)                    ratio=5, safety=1.25
    eff(m)      = slots(m) + min(m, slots(m) // ratio) * D    D = draft tokens
    sessions_state(m) = slots(m) // ratio                     = the admitted
                                                                concurrency,
                                                                ~ m * safety

so the marginal cost of ONE more admitted session is ``ratio * safety = 6.25``
slots, not one slot. Free bytes and KV capacity at a candidate m are

    free_r(m) = F_r - A_r * eff(m)
    P_r(m)    = free_r(m) / c
    C(m)      = min( sum_r P_r , 64 * min_r P_r )      (the converged
                                                        weighted-DCP optimum,
                                                        same rule as
                                                        ``PerfCostModel.predict_capacity``)

Sessions carried at a target context L is the smaller of the two bounds:

    S(m, L) = min( sessions_state(m) , floor(C(m) / L) )

The first term rises in m, the second falls, so the maximum sits at the
crossing. Ignoring the integer rounding and taking the ``sum_r`` branch of
``C`` gives the closed form used as the search seed:

    m* = sum_r F_r / ( safety * L * c + (ratio * safety + D) * sum_r A_r )

and the *context-length* break-even -- the length at which one more session's
state weighs exactly as much as that session's KV, independent of any budget --
is

    L_breakeven = (ratio * safety) * A_total / c

The reported point is then refined by an exact integer scan around the seed
(the closed form drops the ceil/floor and the ``64 * min`` branch).

ASSUMPTIONS (all stated, none of them hidden)
---------------------------------------------
1. A *session* is one admitted request holding ``L`` tokens of KV plus one
   state slot. The pool's spare slots for *cached inactive* states are not
   counted as extra sessions -- ``slots // ratio`` is the same admitted
   concurrency the runtime derives (``max_num_reqs``), so the recommendation
   is directly comparable with what the server prints.
2. All ranks see the same target context; KV is split across ranks by the
   plan's own token vector, which is exactly what ``C(m)`` models.
3. Weights, fixed overhead and the KV cell do not depend on mrr. The solo
   draft host's graph reserve does (``512 MiB * mrr``); it is charged.
4. ``A_r`` is read back from ``PerfCostModel.mamba_pool_bytes`` at a probe
   target, so the *byte* model stays the single source of truth; only the slot
   arithmetic above is mirrored here.
5. ``PerfCostModel._mamba_pool_bytes`` clamps its sizing target to 48
   (``min(target, 48)``) because an mrr the server auto-defaulted does not
   reflect real demand. A *user-set* mrr is not clamped by the runtime, so a
   recommendation above 48 would be under-charged by that predictor. This
   module charges the state pool at the unclamped target and reports the
   correction in ``predictor_clamp_note`` rather than inheriting a free lunch.

THIS IS A SUGGESTION
--------------------
Nothing here changes a default or a flag. The report is an additional field on
the plan; the operator keeps whatever ``--max-running-requests`` they passed.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "BalancePoint",
    "MrrBalanceReport",
    "DEFAULT_TARGET_CONTEXTS",
    "MAMBA_RATIO",
    "MAMBA_SAFETY_MARGIN",
    "state_slot_count",
    "admitted_sessions",
    "balance_report",
]

#: Mirrors ``PerfCostModel._mamba_pool_bytes`` (``ratio = 5``), which in turn
#: mirrors ``MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO (3) +
#: MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP (2)`` in
#: ``model_runner_kv_cache_mixin``. Keep the three in sync.
MAMBA_RATIO = 5
#: ``MAMBA_AUTO_SAFETY_MARGIN`` -- headroom on the concurrency target, NOT a
#: VRAM fraction.
MAMBA_SAFETY_MARGIN = 1.25
#: The clamp ``PerfCostModel._mamba_pool_bytes`` applies to its sizing target.
PREDICTOR_TARGET_CLAMP = 48

#: Target contexts the report is computed for, when the caller names none.
#: Short chat turn / long chat turn / document-sized session.
DEFAULT_TARGET_CONTEXTS: Tuple[int, ...] = (4096, 10240, 32768)

#: Upper bound of the integer refinement scan. Well past any point where the
#: state pool has eaten the whole budget on real hardware.
_SCAN_MAX = 4096


def state_slot_count(
    target: int,
    draft_tokens: int = 0,
    ratio: int = MAMBA_RATIO,
    safety: float = MAMBA_SAFETY_MARGIN,
    clamp: Optional[int] = None,
) -> int:
    """State slots the mamba pool is sized to for a concurrency ``target``.

    ``slots = ceil(target * ratio * safety)``, plus the spec-decode
    intermediate pad ``min(target, slots // ratio) * draft_tokens``. With
    ``clamp`` set the target is capped first -- pass ``PREDICTOR_TARGET_CLAMP``
    to reproduce ``PerfCostModel._mamba_pool_bytes`` byte-exactly.
    """
    t = max(int(target), 1)
    if clamp is not None:
        t = min(t, int(clamp))
    slots = math.ceil(t * ratio * safety)
    return slots + min(t, slots // ratio) * int(draft_tokens or 0)


def admitted_sessions(
    target: int, ratio: int = MAMBA_RATIO, safety: float = MAMBA_SAFETY_MARGIN
) -> int:
    """Concurrent requests the pool admits at concurrency ``target``.

    ``max_num_reqs = max_mamba_cache_size // mamba_ratio`` -- the runtime's own
    definition (``handle_max_mamba_cache``). Approximately ``target * safety``.
    """
    t = max(int(target), 1)
    return max(math.ceil(t * ratio * safety) // ratio, 1)


@dataclasses.dataclass(frozen=True)
class BalancePoint:
    """The recommended mrr for one target context, and what it buys."""

    #: Tokens per session this point was solved for.
    target_context_tokens: int
    #: The SUGGESTED ``--max-running-requests``. Nothing is changed by it.
    recommended_max_running_requests: int
    #: Sessions carried at ``target_context_tokens`` -- the min of the two
    #: bounds below, i.e. the quantity being maximised.
    sessions: int
    #: Sessions the state pool admits at that mrr (``slots // ratio``).
    state_bound_sessions: int
    #: Sessions the KV pool funds at that mrr (``C(m) / L``, not floored).
    kv_bound_sessions: float
    #: Predicted KV context tokens at that mrr (ESTIMATE, same cost model).
    kv_tokens: float
    #: State pool GiB across all ranks at that mrr.
    mamba_gib: float
    #: State slots the pool is sized to at that mrr.
    state_slots: int
    #: Which side binds at the recommendation: "state", "kv", or "balanced".
    binding: str
    #: Sessions the CURRENT mrr carries at this context (None when unset), so
    #: the suggestion can be read as a delta rather than an absolute.
    sessions_at_current: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class MrrBalanceReport:
    """Per-target-context balance points plus the constants behind them."""

    points: List[BalancePoint]
    #: ``--max-running-requests`` as planned (None = server default).
    current_max_running_requests: Optional[int]
    mamba_ratio: int
    safety_margin: float
    #: State bytes per SLOT, whole model (sum over ranks), in MiB.
    state_mib_per_slot: float
    #: State bytes per admitted SESSION (``ratio * safety`` slots), in MiB.
    state_mib_per_session: float
    #: Per-rank state MiB per slot -- shows which rank carries the GDN mass.
    state_mib_per_slot_per_rank: List[float]
    #: KV bytes per token (the full-KV-head cell).
    kv_cell_bytes: float
    #: Context length at which ONE session's state weighs as much as its KV.
    #: Budget-independent; the pure marginal break-even.
    break_even_context_tokens: float
    #: Human note: what the suggestion means and what it does not.
    note: str
    #: Set when a recommendation exceeds ``PREDICTOR_TARGET_CLAMP`` and the
    #: capacity report on the same plan therefore under-charges the pool.
    predictor_clamp_note: str = ""


def _free_bytes_at(model, weights, overhead_bytes, budgets_mib, per_slot, m):
    """Per-rank bytes left for KV at concurrency ``m`` (see the module
    docstring). Mirrors the ``free_bytes`` block of
    ``PerfCostModel.predict_capacity``, with the state pool re-charged at the
    unclamped target."""
    from sglang.srt import uneven_perf

    eff = state_slot_count(m, model.spec_draft_tokens if model.spec_active else 0)
    solo_extra = 0.0
    if model.solo_active:
        solo_extra = (
            uneven_perf._SOLO_HOST_WORKSPACE_MIB
            + uneven_perf._SOLO_HOST_GRAPH_MIB_PER_REQ * max(int(m), 1)
        ) * 2**20
    out = []
    for r in range(model.tp_size):
        f = (
            budgets_mib[r] * 2**20
            - weights[r]
            - per_slot[r] * eff
            - overhead_bytes
        )
        if model.solo_active and r == model.solo_rank:
            f -= solo_extra
        out.append(f)
    return out, eff


def _context_at(model, free_bytes):
    """Predicted context tokens for a free-bytes vector -- the SAME rule
    ``predict_capacity`` applies (``min(sum P, 64 * min P)``), including the
    solo-draft closed form when a solo host is in play."""
    from sglang.srt import uneven_perf

    if model.solo_active:
        p = model._solo_rank_token_capacity(free_bytes)
    else:
        p = [f / model.kv_cell_bytes for f in free_bytes]
    if any(x < uneven_perf._PREDICT_MIN_RANK_TOKENS for x in p):
        return 0.0
    return min(sum(p), uneven_perf._PREDICT_TOKEN_UNITS * min(p))


def balance_report(
    plan_inputs,
    base_plan,
    budgets_mib,
    target_contexts: Sequence[int] = DEFAULT_TARGET_CONTEXTS,
) -> Optional[MrrBalanceReport]:
    """Solve the state/KV balance for each target context.

    Returns ``None`` when there is nothing to balance: a non-hybrid model (no
    GDN layers -> no state pool), or a plan sized from MEASURED residencies
    (there the state pool is a measured constant, not an mrr-driven term, so
    the trade cannot be re-derived from the report).
    """
    from sglang.srt import uneven_perf
    from sglang.srt.uneven_perf import PerfCostModel

    model = PerfCostModel(plan_inputs, list(base_plan), list(budgets_mib))
    if not model.gdn_layers or not model.gdn_units:
        return None
    if model.measured is not None:
        return None

    mlp_vector = plan_inputs.rank_mlp_ratio or list(base_plan)
    weights = model.per_rank_weight_bytes(mlp_vector)
    overhead_bytes = (
        uneven_perf._PREDICT_OVERHEAD_MIB + uneven_perf._PREDICT_MAMBA_ACT_RESERVE_MIB
    ) * 2**20

    # Per-SLOT state bytes per rank, read back from the byte model at the very
    # probe target it was built with (<= the clamp, so the read-back is exact).
    draft = model.spec_draft_tokens if model.spec_active else 0
    probe = min(int(plan_inputs.max_running_requests or 16), PREDICTOR_TARGET_CLAMP)
    probe_eff = state_slot_count(probe, draft)
    if probe_eff <= 0:
        return None
    per_slot = [b / probe_eff for b in model.mamba_pool_bytes]
    slot_total = sum(per_slot)
    if slot_total <= 0:
        return None

    per_session_bytes = MAMBA_RATIO * MAMBA_SAFETY_MARGIN * slot_total
    break_even = per_session_bytes / model.kv_cell_bytes

    # F_r: the bytes that get split between state pool and KV.
    f_total = sum(
        budgets_mib[r] * 2**20 - weights[r] - overhead_bytes
        for r in range(model.tp_size)
    )

    def evaluate(m: int, ctx_len: int):
        free, eff = _free_bytes_at(
            model, weights, overhead_bytes, budgets_mib, per_slot, m
        )
        ctx = _context_at(model, free)
        state_side = admitted_sessions(m)
        kv_side = ctx / ctx_len
        return min(state_side, int(kv_side)), state_side, kv_side, ctx, eff

    current = plan_inputs.max_running_requests
    points: List[BalancePoint] = []
    clamped = False
    for ctx_len in target_contexts:
        ctx_len = int(ctx_len)
        if ctx_len <= 0:
            continue
        # Closed-form seed (module docstring), then an exact integer scan
        # around it -- the seed drops the ceil/floor and the 64*min branch.
        denom = (
            MAMBA_SAFETY_MARGIN * ctx_len * model.kv_cell_bytes
            + (MAMBA_RATIO * MAMBA_SAFETY_MARGIN + draft) * slot_total
        )
        seed = int(f_total / denom) if denom > 0 else 1
        lo = max(1, seed - 16)
        hi = min(_SCAN_MAX, max(seed + 16, lo + 32))
        best = None
        for m in range(lo, hi + 1):
            sess, state_side, kv_side, ctx, eff = evaluate(m, ctx_len)
            if sess <= 0:
                if state_side > kv_side:
                    break  # KV exhausted; larger m only gets worse
                continue
            # Tie-break to the SMALLER mrr: same sessions, more KV headroom.
            if best is None or sess > best[0]:
                best = (sess, m, state_side, kv_side, ctx, eff)
        if best is None:
            continue
        sess, m, state_side, kv_side, ctx, eff = best
        if m > PREDICTOR_TARGET_CLAMP:
            clamped = True
        if state_side < kv_side - 0.5:
            binding = "state"
        elif kv_side < state_side - 0.5:
            binding = "kv"
        else:
            binding = "balanced"
        at_current = None
        if current:
            at_current = evaluate(int(current), ctx_len)[0]
        points.append(
            BalancePoint(
                target_context_tokens=ctx_len,
                recommended_max_running_requests=m,
                sessions=sess,
                state_bound_sessions=state_side,
                kv_bound_sessions=kv_side,
                kv_tokens=ctx,
                mamba_gib=slot_total * eff / 2**30,
                state_slots=eff,
                binding=binding,
                sessions_at_current=at_current,
            )
        )
    if not points:
        return None

    note = (
        "SUGGESTION only — nothing is applied. Each point maximises "
        "min(sessions the state pool admits, sessions the KV pool funds at "
        "that context). One more admitted session costs "
        f"{MAMBA_RATIO * MAMBA_SAFETY_MARGIN:g} state slots "
        f"({per_session_bytes / 2**20:.0f} MiB, context-independent) and frees "
        f"{break_even:,.0f} tokens of KV — so above that context length KV "
        "buys more sessions than concurrency does. Capacity numbers are the "
        "cost model's estimates."
    )
    clamp_note = ""
    if clamped:
        clamp_note = (
            f"A recommendation above {PREDICTOR_TARGET_CLAMP} is charged here "
            "at the unclamped state-pool size. PerfCostModel._mamba_pool_bytes "
            f"clamps its sizing target to {PREDICTOR_TARGET_CLAMP}, so the "
            "plan's max-context figure for such an mrr under-states the pool "
            "and over-states KV. The runtime does not clamp a user-set "
            "--max-running-requests."
        )
    return MrrBalanceReport(
        points=points,
        current_max_running_requests=current,
        mamba_ratio=MAMBA_RATIO,
        safety_margin=MAMBA_SAFETY_MARGIN,
        state_mib_per_slot=slot_total / 2**20,
        state_mib_per_session=per_session_bytes / 2**20,
        state_mib_per_slot_per_rank=[b / 2**20 for b in per_slot],
        kv_cell_bytes=float(model.kv_cell_bytes),
        break_even_context_tokens=break_even,
        note=note,
        predictor_clamp_note=clamp_note,
    )
