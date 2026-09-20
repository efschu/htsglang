"""Task #49 after fn8c8b: the router half of the NaN hunt.

fn8c8b settled where the NaN is BORN. All three originating calls on TP0 read
``ORIGIN ENTERS-AT-ROUTER-W`` with ``bad_gemm2 == bad_routerw`` on the SAME
rows and ``transport={GEMM1:CLEAN, ACT:CLEAN, GEMM2:TRANSPORT}``. The Marlin
kernel is exonerated: it multiplies in a weight that is already non-finite
(``mul_topk_weights`` is the down-projection's flag and nowhere else).

That moves the question one layer up, and the wave-slice gather is NOT it:

    flat_weights = topk_weights.reshape(-1, 1).contiguous()
    tw_w         = flat_weights.index_select(0, idx)       # expert_offload.py

``idx`` is ``np.flatnonzero(wave_of_pair == w)``, so every index is a valid
flat (token, expert) pair and no padded slot is ever gathered. A pure
``index_select`` cannot invent a NaN -- it can only carry one that
``topk_weights`` already had.

Which leaves the router itself, and there the shape of the observed damage is
decisive. ``first_bad_routerw=[28..35]`` is EIGHT CONSECUTIVE PAIRS. Pairs are
``t*K + k``, so eight consecutive ones are eight experts OF ONE TOKEN. A
per-token, all-K failure is exactly what an unguarded renormalisation does:

    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

(topk.py, ``fused_topk_torch_native`` and ``biased_grouped_topk_impl``). When a
token's selected scores all underflow to zero the sum is zero and every one of
its K weights becomes 0/0 = NaN at once. It is data-dependent, hence "only
sometimes", and it is card-dependent to the extent that the gate GEMM emits
slightly different logits on sm_120 than on sm_86.

This module carries the two things that claim needs: a probe that RESOLVES
ROUTER-W into gate / renorm / gather, and a guard that can be switched on to
test the repair. Both are pure enough to prove at the desk.
"""

from __future__ import annotations

import os
from typing import Optional

#: Ordered, like STAGE_ORDER in fused_marlin_moe: the first level that is dirty
#: is the one that made the NaN.
ROUTER_LEVELS = ("GATE-LOGITS", "TOPK-WEIGHTS", "GATHERED")

ENV_RENORM_GUARD = "SGLANG_MOE_ROUTER_RENORM_GUARD"


def classify_router_origin(level_bad_counts) -> str:
    """Name the first router level that carries a non-finite value.

    ``GATE-LOGITS`` -- the gate GEMM or the model already produced inf/NaN, and
        softmax/sigmoid merely propagated it.
    ``TOPK-WEIGHTS`` -- the logits were finite and the selection/renormalisation
        made the NaN. This is the 0/0 case.
    ``GATHERED`` -- both were finite and only the wave-slice gather is dirty,
        which would mean an index defect after all.

    A level that was not measured is reported, never assumed clean."""
    seen = False
    for level in ROUTER_LEVELS:
        if level not in level_bad_counts:
            return f"UNMEASURED-AT-{level}"
        seen = True
        if int(level_bad_counts[level]) > 0:
            return f"ENTERS-AT-{level}"
    return "CLEAN" if seen else "UNMEASURED-AT-GATE-LOGITS"


def renorm_guard_on(env=None) -> bool:
    """``SGLANG_MOE_ROUTER_RENORM_GUARD=1`` -- clamp the renorm denominator.

    OPT-IN, and deliberately so. Switched on by default it would MASK the
    defect: every 0/0 token would silently become a uniform-zero routing row
    and the boot would look healthy while a token was being routed nowhere.
    One boot has to show the guard changing the outcome before anyone may
    decide that zeroing is the right semantics."""
    env = os.environ if env is None else env
    raw = str(env.get(ENV_RENORM_GUARD, "0")).strip().lower()
    return raw in ("1", "true", "on", "yes")


#: Smallest positive normal in float32. Chosen rather than an arbitrary 1e-9 so
#: the clamp can only ever act on a sum that is genuinely denormal-or-zero: any
#: real routing sum is orders of magnitude larger, so a guarded run and an
#: unguarded run differ ONLY on the rows that would have produced NaN.
RENORM_EPS = 1.1754943508222875e-38


def guarded_denominator(total, eps: float = RENORM_EPS):
    """``total`` clamped away from zero, preserving sign-free magnitude.

    Works on a torch tensor or a plain float, so the arithmetic is testable
    without CUDA and without touching the real routing path."""
    try:
        return total.clamp_min(eps)
    except AttributeError:
        return max(float(total), eps)


def renorm_rows_that_would_nan(sums, eps: float = RENORM_EPS) -> int:
    """How many rows the guard would actually change. Instrument, not a fix.

    Reported next to the guard so a boot can say 'the guard fired N times'
    rather than 'the guard was on'."""
    try:
        return int((sums <= eps).sum().item())
    except AttributeError:
        return int(float(sums) <= eps)
