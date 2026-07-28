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
"""Named working points, each priced against the others (#215).

:mod:`sglang.srt.planner.levers` names the DIRECTIONS this fork can be tuned
in and what each one costs in prose. This module turns four of those
directions into working points a reader can actually pick between: it resolves
each one to a concrete plan and then reports, in numbers taken from that plan,
what choosing it buys and what it gives up.

WHY THE NUMBERS COME FROM A RE-PLAN
-----------------------------------
The only honest way to say "this working point costs you 18 000 tokens of
context" is to plan the configuration and read the figure back. So each
profile is resolved to an MLP unit vector, the planner is run again with that
vector, and every number in the counter-reckoning is read off the resulting
:class:`~sglang.srt.planner.feasibility.PlanResult`:

    context tokens   ``capacity.max_context_tokens``  (the cost model's VRAM
                                                       arithmetic)
    sessions @ L     ``mrr_balance``                  (the state/KV balance
                                                       point, #253)
    decode tok/s     the balanced split's ``roofline.decode_tok_s_low``,
                     moved by ``PerfCostModel.decode_cost_percent``
    prefill tok/s    the balanced split's ``roofline.prefill_tok_s``, moved by
                     the ``prefill_time_model`` ratio

Nothing is a literal written here. A figure that cannot be computed for a
configuration is reported ABSENT with its reason -- an unsized roofline and a
throughput of zero are different statements, and a reader who cannot tell them
apart is worse off than one who is told nothing.

TWO INSTRUMENTS, EACH WHERE IT IS THE RIGHT ONE
-----------------------------------------------
The throughput rows are a product, and deliberately so. The roofline supplies
the LEVEL -- it is the only thing here that produces an absolute tok/s at all
-- and the cost model supplies the MOVE from one split to another, because
that is the term this fork fitted against measured concentration vectors.

Ranking splits by the roofline's peak-bandwidth arithmetic instead is the
error task #216 recorded in the boot path: the same expression over PROBED
PEAK bandwidth reported -22 % for a vector that measured +16.5 %. The
effective-bandwidth divisor with its residual exponent exists because of that.
So the roofline never ranks a split here; it only says where the scale starts.

THE SELECTION, AND WHY THE INVARIANTS HOLD
------------------------------------------
All four profiles choose from ONE candidate set: the base split plus the same
ladder ``uneven_perf._mlp_candidates`` hands the boot-time optimiser, filtered
to the vectors the capacity model calls feasible. Each profile then takes the
best candidate under its own objective over that shared set:

    max context   most  predict_capacity(v)["ctx"]
    balanced      the VRAM-auto base split, unchanged
    max decode    least decode_round_time(v)     (needs per-rank bandwidth)
    max prefill   least prefill_time_model(v)    (needs per-rank compute)

Each profile is therefore the best profile at the quantity it is named after,
measured by the very number the table prints -- the reported context IS
``predict_capacity(v)["ctx"]``, and the reported throughput move IS the ranked
one. ``test_lever_profiles`` pins all three.

A speed move smaller than :data:`_SPEED_TIE_TOL_PCT` keeps the base split. The
tolerance is this rig's measured boot-to-boot noise floor, so a difference no
boot could demonstrate never becomes a different configuration.

WHAT THIS MODULE REFUSES TO DO
------------------------------
It does not report a NET of a prefill gain against a decode cost. Both terms
are modelled, each fitted on one rig, and :data:`crossover.MODELLED_NET_REFUSED`
records why combining them off that rig is not defensible. The two sides are
shown separately, each carrying its own caveat, and the reader does the
weighing -- which is the entire point of a counter-reckoning.

It also does not invent a direction where the hardware supplies none. On
uniform card budgets the VRAM-auto split already collapses to the even split
and the MLP vector has nothing to redistribute; the boot path says so
(``apply_auto_performance``) and so does this module, by resolving every
profile to the same configuration and stating why.

THE SINGLE CONTROL
------------------
The four profiles are ordered along one axis: how much predicted context the
working point is willing to spend for speed. That is not an editorial
ordering, it is the axis ``--rank-perf-loose-ctx-percent`` already measures
along -- 0 % for the context lever, 10 % for decode, 15 % for prefill. A
dashboard can therefore expose them as one slider with four stops without
inventing a scale, and the expert view keeps every individual knob.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ProfileSpec",
    "MetricSpec",
    "PROFILES",
    "PROFILE_KEYS",
    "METRICS",
    "SLIDER_AXIS",
    "DEFAULT_SESSION_CONTEXT",
    "build_profiles",
]


# ---------------------------------------------------------------------------
# The profiles
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ProfileSpec:
    """One named working point.

    ``lever_key`` points at the :mod:`sglang.srt.planner.levers` entry that
    supplies the prose, the flag set and the evidence labelling. The prose is
    never duplicated here: a direction's stated cost is written once.
    """

    key: str
    label: str
    #: ``levers.LEVERS`` key, or None for the base split (which is not a
    #: direction -- it is the absence of one).
    lever_key: Optional[str]
    #: The objective this profile optimises over the shared candidate set.
    objective: str  # "context" | "base" | "decode" | "prefill"
    #: The ``--rank-perf-tune`` value the expert view's template buttons
    #: already use, so picking a profile and pressing a template are the same
    #: state rather than two competing ones.
    tune: str
    #: Where this profile sits on the context-for-speed axis: the
    #: ``--rank-perf-loose-ctx-percent`` the matching lever sets, or None when
    #: the profile sets no budget (the base split spends nothing by default).
    loose_ctx_percent: Optional[float]
    #: One line for the reader, about the AXIS position -- not a restatement
    #: of the lever's gain/cost, which comes from the registry.
    axis_note: str


#: Ordered left to right along the context-for-speed axis. The order is fixed
#: rather than data-derived: a control whose stops reshuffle when the numbers
#: move is a control nobody can build a habit with.
PROFILES: Tuple[ProfileSpec, ...] = (
    ProfileSpec(
        key="max_context",
        label="max context",
        lever_key="context",
        objective="context",
        tune="maxkv",
        loose_ctx_percent=0.0,
        axis_note=(
            "Spends no context for speed: the MLP split is the one that "
            "predicts the largest KV pool these cards can fund."
        ),
    ),
    ProfileSpec(
        key="balanced",
        label="balanced",
        lever_key=None,
        objective="base",
        tune="both",
        loose_ctx_percent=None,
        axis_note=(
            "The VRAM-auto split, unchanged. Every other profile is priced "
            "against this one."
        ),
    ),
    ProfileSpec(
        key="max_decode",
        label="max decode",
        lever_key="decode_speed",
        objective="decode",
        tune="dec",
        loose_ctx_percent=10.0,
        axis_note=(
            "Balances the per-step weight read across unequal cards. The "
            "measured decode optimum is flat across representable splits, so "
            "this profile often resolves BACK to the base split -- when it "
            "does, that is the answer, not a missing one."
        ),
    ),
    ProfileSpec(
        key="max_prefill",
        label="max prefill",
        lever_key="prefill_speed",
        objective="prefill",
        tune="enc",
        loose_ctx_percent=15.0,
        axis_note=(
            "Concentrates dense-MLP mass on the compute-strong rank. It is "
            "the far end of the axis because it spends the most context, and "
            "it pays back only on prompt-dominated workloads."
        ),
    ),
)

PROFILE_KEYS: Tuple[str, ...] = tuple(p.key for p in PROFILES)

#: The profile every counter-reckoning is measured against.
BASELINE_KEY = "balanced"

SLIDER_AXIS = (
    "One axis: how much predicted context this working point is willing to "
    "spend for speed. Left keeps all of it, right spends the most. The stops "
    "are the four profiles; the expert view keeps every individual knob."
)


# ---------------------------------------------------------------------------
# The metrics each profile is priced in
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str
    higher_is_better: bool
    #: Where the figure comes from, in the reader's terms. Rendered next to
    #: the number so an estimate is never mistaken for a measurement.
    basis: str


#: Target context the session count is solved for, when the caller names
#: none. The middle rung of ``mrr_balance.DEFAULT_TARGET_CONTEXTS`` -- a long
#: chat turn rather than a document or a one-liner.
DEFAULT_SESSION_CONTEXT = 10240

METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec(
        key="kv_tokens",
        label="context that fits",
        unit="tokens",
        higher_is_better=True,
        basis=(
            "planner arithmetic: the cost model's predicted max context for "
            "this exact configuration (an estimate, same one the verdict "
            "shows)"
        ),
    ),
    MetricSpec(
        key="decode_tok_s",
        label="decode",
        unit="tok/s",
        higher_is_better=True,
        basis=(
            "two instruments, each doing what it is good for: the LEVEL is "
            "the roofline estimate of the balanced split at its reference "
            "context (nameplate peaks, assumed efficiency factors -- a "
            "ballpark, not a measurement), and the MOVE from one split to "
            "another is the cost model's predicted step-time change, which is "
            "the term fitted against measured concentration vectors"
        ),
    ),
    MetricSpec(
        key="prefill_tok_s",
        label="prefill",
        unit="tok/s",
        higher_is_better=True,
        basis=(
            "same composition as decode: roofline level for the balanced "
            "split, cost-model ratio for the move. The prefill term is the "
            "weaker-calibrated of the two and is fitted on one rig"
        ),
    ),
    MetricSpec(
        key="sessions",
        label="sessions at the target context",
        unit="sessions",
        higher_is_better=True,
        basis=(
            "the state/KV balance point: sessions carried at the target "
            "context, at the concurrency that balance recommends (a "
            "suggestion the plan never applies)"
        ),
    ),
)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

#: Link bandwidth assumed when no probe measured one. Mirrors the fallback
#: ``uneven_perf.apply_auto_performance`` uses for the same term, so the
#: offline ranking and the boot-time ranking do not diverge on a rig that was
#: never probed.
_FALLBACK_LINK_GBS = 8.0

#: A predicted SPEED move smaller than this does not move the vector: the
#: profile keeps the base split. 1 % is the boot-to-boot noise floor measured
#: on this rig for the decode step (#210: 1.07 %), so below it the model is
#: predicting a difference no boot could demonstrate, and changing the
#: configuration for it would be changing it for nothing.
_SPEED_TIE_TOL_PCT = 1.0

#: Capacity is deterministic arithmetic rather than a timing, so there is no
#: noise floor to hide behind: only an exact tie falls back to the base split.
#: The total weight bytes are invariant under an MLP redistribution, so exact
#: ties are the NORMAL case whenever the ``64 * min P`` branch is not binding.
_CAPACITY_TIE_TOL = 1e-9


def _rank_scores(roofline) -> Optional[Dict[str, Any]]:
    """Per-rank bandwidth and compute peaks, with their provenance.

    Read back from the roofline estimate the base plan already produced, so
    the profiles rank candidates on exactly the peaks the throughput numbers
    are computed from. ``None`` when the roofline could not be sized -- then
    no speed objective is answerable and the profiles say so.
    """
    per_rank = list(getattr(roofline, "per_rank", None) or [])
    if not per_rank:
        return None
    bw = [float(r.peak_membw_gbs) for r in per_rank]
    gemm = [float(r.peak_flops_tflops) for r in per_rank]
    if not all(x > 0 for x in bw) or not all(x > 0 for x in gemm):
        return None
    sources = sorted(
        {r.membw_source for r in per_rank} | {r.flops_source for r in per_rank}
    )
    return {
        "membw_gbs": bw,
        "gemm_tflops": gemm,
        "sources": sources,
        "measured": all(r.membw_source == "measured" for r in per_rank)
        and all(r.flops_source == "measured" for r in per_rank),
    }


def _min_link_gbs() -> Tuple[float, str]:
    """The narrowest measured pair link, or the documented fallback."""
    try:
        from sglang.srt.uneven_perf import get_cached_hardware_profile

        profile, _ = get_cached_hardware_profile()
        links = (profile or {}).get("links") or {}
        bws = [v["p2p_gbs"] for k, v in links.items() if k != "__group__"]
        if bws:
            return float(min(bws)), "measured pair matrix"
    except Exception:
        pass
    return _FALLBACK_LINK_GBS, "assumed (no probed link matrix)"


def _candidates(model, base_plan: List[int], scores) -> List[List[int]]:
    """The shared candidate set: base plus the boot-time optimiser's ladder,
    filtered to what the capacity model calls feasible."""
    from sglang.srt.uneven_perf import _mlp_candidates

    # The base split enters VERBATIM, not gcd-reduced: every comparison in
    # this module ("is this the base split?", the tie-break, the plan cache)
    # is against the vector the plan actually carries, and a reduced copy of
    # it would compare unequal to itself. ``_mlp_candidates`` already reduces
    # its own output and already drops the base.
    base = list(base_plan)
    out: List[List[int]] = [base]
    if scores is not None and len(set(base_plan)) > 1:
        for cand in _mlp_candidates(model, list(scores["gemm_tflops"]), base_plan):
            out.append(list(cand))
    feasible: List[List[int]] = []
    for vec in out:
        try:
            if model.predict_capacity(vec)["feasible"]:
                feasible.append(vec)
        except Exception:
            continue
    return feasible or [base]


def _pick_best(cands, score, base: List[int], tol_pct: float):
    """The candidate with the highest ``score`` (a positive quantity: tokens,
    or the reciprocal of a modelled time), unless the base split is within
    ``tol_pct`` of it -- then the base split, because a move the instrument
    cannot resolve is not a reason to change the configuration.

    Returns ``(vector, margin_pct)``: how much the winner beats the base split
    by, in percent. ``margin`` is None only when the base split has no usable
    score, which is also the only case where the tolerance cannot be applied.
    """
    scored = [(c, score(c)) for c in cands]
    best, best_s = max(scored, key=lambda x: x[1])
    base_s = next((v for c, v in scored if list(c) == list(base)), None)
    if base_s is None or base_s <= 0:
        return list(best), None
    margin = (best_s - base_s) / base_s * 100.0
    if margin <= tol_pct:
        return list(base), margin
    return list(best), margin


def _margin_txt(margin: Optional[float], fmt: str = "%+.1f %%") -> str:
    return "not comparable" if margin is None else (fmt % margin)


def _choose(
    spec: ProfileSpec, model, base: List[int], cands, scores, min_link, uniform=False
):
    """(vector, reason) for one profile over the shared candidate set.

    Each objective is evaluated with the instrument the fork fitted for it:
    capacity from ``predict_capacity``, the decode move from the EFFECTIVE
    -bandwidth round time, the prefill move from the invariant-charged prefill
    time model. Ranking decode by probed PEAK bandwidth instead is exactly the
    error #216 recorded -- it reported -22 % for a vector that measured
    +16.5 % -- so the roofline's peak-based decode number is never the ranker
    here, only the level the ranked move is applied to.
    """
    if spec.objective == "base":
        return list(base), "the VRAM-auto split, by definition of this profile"
    if len(cands) <= 1:
        if uniform:
            return list(base), (
                "uniform card budgets: the VRAM-auto split is the even split "
                "and there is no asymmetry for an MLP vector to exploit, so "
                "every profile resolves to it (the boot path stops here for "
                "the same reason)"
            )
        if scores is None:
            return list(base), (
                "no per-rank speed figures for these cards -- neither a "
                "cached probe nor a nameplate spec -- so no candidate split "
                "can be ranked against another and the base split stands. "
                "Run the hardware probe."
            )
        return list(base), (
            "no candidate other than the base split is feasible here, so "
            "every profile resolves to it"
        )
    if spec.objective == "context":
        vec, margin = _pick_best(
            cands,
            lambda c: model.predict_capacity(c)["ctx"],
            base,
            _CAPACITY_TIE_TOL,
        )
        if list(vec) == list(base):
            return vec, (
                "no candidate predicts more context than the base split "
                "(total weight bytes do not change when MLP units move, so "
                "the KV pool only moves when the per-rank minimum binds)"
            )
        return vec, (
            f"largest predicted max context over {len(cands)} candidates "
            f"({_margin_txt(margin)} against the base split)"
        )
    if scores is None:
        return None, (
            "no per-rank speed figures: this rig has neither a cached probe "
            "nor nameplate peaks for every card, so a speed objective cannot "
            "be ranked. Run the hardware probe."
        )
    if spec.objective == "decode":
        vec, margin = _pick_best(
            cands,
            lambda c: 1.0 / max(model.decode_round_time(c, scores["membw_gbs"]), 1e-12),
            base,
            _SPEED_TIE_TOL_PCT,
        )
        if list(vec) == list(base):
            return vec, (
                "the best candidate stays inside this rig's measured "
                f"boot-to-boot noise floor ({_margin_txt(margin)} modelled), "
                "so this profile keeps the base split -- which is the M22 "
                "decode-flatness result, reached from this rig's own numbers"
            )
        return vec, (
            f"shortest modelled bs=1 decode round time over {len(cands)} "
            f"candidates ({_margin_txt(margin)} against the base split)"
        )
    if spec.objective == "prefill":
        gemm = list(scores["gemm_tflops"])
        vec, margin = _pick_best(
            cands,
            lambda c: 1.0 / max(model.prefill_time_model(c, gemm, min_link), 1e-12),
            base,
            _SPEED_TIE_TOL_PCT,
        )
        if list(vec) == list(base):
            return vec, (
                "no candidate beats the base split by more than the noise "
                f"floor ({_margin_txt(margin)} modelled); nothing to "
                "concentrate"
            )
        return vec, (
            f"shortest modelled prefill step time over {len(cands)} "
            f"candidates ({_margin_txt(margin)} against the base split)"
        )
    return list(base), "no objective for this profile"


# ---------------------------------------------------------------------------
# Metrics off a plan
# ---------------------------------------------------------------------------


def _session_point(result, target_context: int):
    """The balance point closest to ``target_context``, or None."""
    report = getattr(result, "mrr_balance", None)
    points = list(getattr(report, "points", None) or [])
    if not points:
        return None
    return min(points, key=lambda p: abs(p.target_context_tokens - target_context))


def _speed_ratios(model, vec: List[int], base: List[int], scores, min_link):
    """(decode, prefill) tok/s multipliers of ``vec`` against the base split.

    Both come from the cost model that was fitted for split-to-split
    comparison, never from the roofline's peak-bandwidth arithmetic. Returns
    ``(1.0, 1.0)`` for the base split itself and when no per-rank scores
    exist -- in the latter case no profile leaves the base split anyway.
    """
    if scores is None or list(vec) == list(base):
        return 1.0, 1.0
    try:
        dec_pct = model.decode_cost_percent(list(vec), scores["membw_gbs"])
        dec = 1.0 / (1.0 + dec_pct / 100.0) if dec_pct > -100.0 else 1.0
    except Exception:
        dec = 1.0
    try:
        gemm = list(scores["gemm_tflops"])
        t_base = model.prefill_time_model(list(base), gemm, min_link)
        t_cand = model.prefill_time_model(list(vec), gemm, min_link)
        pre = (t_base / t_cand) if t_cand > 0 else 1.0
    except Exception:
        pre = 1.0
    return dec, pre


def _metric_values(
    result, target_context: int, anchor, ratios
) -> Dict[str, Dict[str, Any]]:
    """Every metric for one planned profile: value, or absence with a reason.

    An absent figure names what is missing. Silence here would read as a zero
    and a zero is a different claim.

    ``anchor`` is the BALANCED split's roofline estimate; ``ratios`` are this
    profile's cost-model tok/s multipliers against it. The absolute throughput
    figures are the product of the two, which is why the metric's ``basis``
    names both instruments -- see the module docstring.
    """
    out: Dict[str, Dict[str, Any]] = {}

    cap = getattr(result, "capacity", None)
    if cap is not None and cap.feasible:
        out["kv_tokens"] = {"available": True, "value": float(cap.max_context_tokens)}
    else:
        out["kv_tokens"] = {
            "available": False,
            "reason": (
                "the capacity model finds no feasible KV pool for this " "configuration"
            ),
        }

    dec_ratio, pre_ratio = ratios
    if anchor is not None:
        out["decode_tok_s"] = {
            "available": True,
            "value": float(anchor.decode_tok_s_low) * dec_ratio,
            "detail": {
                "anchor_tok_s": float(anchor.decode_tok_s_low),
                "at_short_context": float(anchor.decode_tok_s_high) * dec_ratio,
                "reference_context_tokens": int(anchor.reference_context_tokens),
                "cost_model_ratio": dec_ratio,
                "provenance": anchor.provenance,
            },
        }
        out["prefill_tok_s"] = {
            "available": True,
            "value": float(anchor.prefill_tok_s) * pre_ratio,
            "detail": {
                "anchor_tok_s": float(anchor.prefill_tok_s),
                "cost_model_ratio": pre_ratio,
                "provenance": anchor.provenance,
            },
        }
    else:
        miss = {
            "available": False,
            "reason": (
                "no roofline estimate for the balanced split, so there is no "
                "level to apply the modelled move to: a card without a "
                "measured probe score AND without a nameplate spec, or a "
                "geometry the cost model cannot shard"
            ),
        }
        out["decode_tok_s"] = dict(miss)
        out["prefill_tok_s"] = dict(miss)

    point = _session_point(result, target_context)
    if point is not None:
        out["sessions"] = {
            "available": True,
            "value": float(point.sessions),
            "detail": {
                "target_context_tokens": int(point.target_context_tokens),
                "recommended_max_running_requests": int(
                    point.recommended_max_running_requests
                ),
                "binding": point.binding,
                "sessions_at_current": point.sessions_at_current,
            },
        }
    else:
        out["sessions"] = {
            "available": False,
            "reason": (
                "no state/KV balance for this plan: the model has no GDN "
                "state pool to trade against KV, or the plan is sized from "
                "measured residencies where the trade cannot be re-derived"
            ),
        }
    return out


def _delta(value: float, base: float, higher_is_better: bool) -> Dict[str, Any]:
    """One line of the counter-reckoning: the signed difference and whether it
    is a gain or a price. Direction, not colour -- the caller decides how to
    draw it."""
    diff = value - base
    pct = (diff / base * 100.0) if base else None
    if abs(diff) < 1e-9 or (pct is not None and abs(pct) < 0.05):
        direction = "same"
    elif (diff > 0) == higher_is_better:
        direction = "gain"
    else:
        direction = "cost"
    return {
        "delta": diff,
        "pct": pct,
        "direction": direction,
    }


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _lever_suggestions(
    heterogeneous: bool,
    probe,
    node_count: int,
    facility_keys,
    crossover,
    prompt_to_output_ratio,
) -> Dict[str, Any]:
    """The prose half, from the lever registry -- gains, costs, evidence and
    the per-opposing-lever price, all already written and already gated on
    what this build and this rig can support."""
    from sglang.srt.planner.levers import suggest_levers

    try:
        out = suggest_levers(
            heterogeneous=heterogeneous,
            probe=probe,
            node_count=node_count,
            facility_keys_available=facility_keys,
            crossover=crossover,
            prompt_to_output_ratio=prompt_to_output_ratio,
        )
    except Exception:
        return {}
    return {s.lever.key: s for s in out}


def build_profiles(
    model_path: str,
    hardware,
    plan_kwargs: Optional[Dict[str, Any]] = None,
    *,
    session_target_context: int = DEFAULT_SESSION_CONTEXT,
    probe: Optional[dict] = None,
    crossover=None,
    prompt_to_output_ratio: Optional[float] = None,
    node_count: int = 1,
    facility_keys: Optional[Sequence[str]] = None,
    plan_fn=None,
) -> Dict[str, Any]:
    """Resolve every profile against this model and this rig, and price them.

    ``plan_kwargs`` is the keyword set the dashboard would plan with anyway;
    ``rank_mlp_ratio`` inside it is REPLACED by each profile's own vector,
    which is stated as a caveat rather than done quietly.

    Never raises for a rejected or infeasible configuration: the report comes
    back with ``ok=False`` and the planner's own reasons, because a control
    surface that goes blank on a bad number is worse than one that explains.
    """
    from sglang.srt.planner import mrr_balance
    from sglang.srt.planner.feasibility import PlanRejected

    if plan_fn is None:
        from sglang.srt.planner.feasibility import plan as plan_fn

    kwargs = dict(plan_kwargs or {})
    pinned_mlp = kwargs.pop("rank_mlp_ratio", None)
    caveats: List[str] = []

    try:
        base_result = plan_fn(model_path, hardware, **kwargs)
    except PlanRejected as e:
        return {"ok": False, "reasons": list(e.reasons), "profiles": []}
    except ValueError as e:
        return {"ok": False, "reasons": [str(e)], "profiles": []}

    if base_result.capacity is None:
        return {
            "ok": False,
            "reasons": base_result.infeasible_reasons
            or ["the configuration could not be sized at all"],
            "profiles": [],
        }

    inputs = base_result.inputs
    budgets = list(inputs.effective_vram_mib or [])
    base_plan = list(inputs.rank_tp_ratio or [1] * inputs.tp_size)

    from sglang.srt.uneven_perf import PerfCostModel

    model = PerfCostModel(inputs, base_plan, budgets)
    # The balanced split's roofline is the LEVEL every profile's throughput is
    # expressed at; the cost model supplies the move off it. One level for all
    # four keeps the four numbers on one scale.
    anchor = getattr(base_result, "roofline", None)
    scores = _rank_scores(anchor)
    min_link, link_source = _min_link_gbs()
    uniform = len(set(base_plan)) <= 1
    if uniform:
        caveats.append(
            "Uniform card budgets: the VRAM-auto split is the even split and "
            "the MLP vector has nothing to redistribute, so every profile "
            "resolves to the same configuration. This mirrors the boot path "
            "(apply_auto_performance keeps the even split for the same "
            "reason), and it is a statement about the rig, not a gap."
        )
    if scores is None:
        caveats.append(
            "No per-rank speed figures for this rig: the speed profiles fall "
            "back to the base split and say so per profile."
        )
    elif not scores["measured"]:
        caveats.append(
            "The speed objectives are ranked on nameplate card specs ("
            + ", ".join(scores["sources"])
            + "), not on a probe of these cards. Run the hardware probe to "
            "rank them on measured rates."
        )

    cands = _candidates(model, base_plan, scores)
    levers = _lever_suggestions(
        heterogeneous=not uniform,
        probe=probe,
        node_count=node_count,
        facility_keys=facility_keys,
        crossover=crossover,
        prompt_to_output_ratio=prompt_to_output_ratio,
    )

    # One plan per distinct vector. The base plan is already computed, so a
    # profile that resolves back to it costs nothing.
    planned: Dict[Tuple[int, ...], Any] = {tuple(base_plan): base_result}

    def plan_for(vec: List[int]):
        key = tuple(vec)
        if key not in planned:
            try:
                planned[key] = plan_fn(
                    model_path, hardware, rank_mlp_ratio=list(vec), **kwargs
                )
            except (PlanRejected, ValueError):
                planned[key] = None
        return planned[key]

    rows: List[Dict[str, Any]] = []
    for idx, spec in enumerate(PROFILES):
        vec, why = _choose(
            spec, model, base_plan, cands, scores, min_link, uniform=uniform
        )
        if vec is None:
            vec, unresolved = list(base_plan), why
        else:
            unresolved = ""
        result = plan_for(vec)
        if result is None:
            rows.append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "slider_index": idx,
                    "resolved": False,
                    "reason": "the planner rejected this profile's vector",
                }
            )
            continue
        sug = levers.get(spec.lever_key) if spec.lever_key else None
        is_base = list(vec) == list(base_plan)
        flag_delta = [] if is_base else ["--rank-mlp-ratio " + ",".join(map(str, vec))]
        rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "slider_index": idx,
                "resolved": True,
                "lever_key": spec.lever_key,
                "tune": spec.tune,
                "loose_ctx_percent": spec.loose_ctx_percent,
                "axis_note": spec.axis_note,
                "mlp_vector": list(vec),
                "is_base_split": is_base,
                # A stop that resolves to the baseline says so, rather than
                # letting a reader conclude from four identical rows that the
                # control is broken.
                "same_as_baseline": is_base and spec.key != BASELINE_KEY,
                "selection_reason": unresolved or why,
                "objective_unresolved": bool(unresolved),
                "fits": bool(result.fits),
                "infeasible_reasons": list(result.infeasible_reasons),
                "flag_delta": flag_delta,
                "launch_flags": list(result.launch_flags),
                "_metrics": _metric_values(
                    result,
                    int(session_target_context),
                    anchor,
                    _speed_ratios(model, vec, base_plan, scores, min_link),
                ),
                # The prose half stays in the lever registry.
                "gains": list(sug.lever.gains) if sug else [],
                "costs": list(sug.lever.costs) if sug else [],
                "counter_reckoning": dict(sug.counter_reckoning) if sug else {},
                "evidence": [e.to_json() for e in sug.evidence] if sug else [],
                "notes": list(sug.lever.notes) if sug else [],
                "unavailable_flags": list(sug.unavailable_flags) if sug else [],
                "lever_flags": list(sug.command_flags) if sug else [],
                "confidence": sug.confidence if sug else "detailed",
            }
        )

    # Counter-reckoning: every metric of every profile, against the baseline.
    # The baseline's own figures have to exist before any delta can be taken,
    # so the values are collected first and the deltas are a second pass.
    values = {row["key"]: row.pop("_metrics") for row in rows if row.get("resolved")}
    base_vals = {
        m.key: values[BASELINE_KEY][m.key]["value"]
        for m in METRICS
        if BASELINE_KEY in values and values[BASELINE_KEY][m.key].get("available")
    }
    for row in rows:
        if not row.get("resolved"):
            continue
        vals = values[row["key"]]
        metrics = []
        for m in METRICS:
            cell = dict(vals[m.key])
            cell.update(
                {
                    "key": m.key,
                    "label": m.label,
                    "unit": m.unit,
                    "higher_is_better": m.higher_is_better,
                    "basis": m.basis,
                }
            )
            bv = base_vals.get(m.key)
            if row["key"] != BASELINE_KEY and cell.get("available") and bv is not None:
                cell["vs_baseline"] = _delta(cell["value"], bv, m.higher_is_better)
            metrics.append(cell)
        row["metrics"] = metrics

    distinct = {tuple(r["mlp_vector"]) for r in rows if r.get("resolved")}
    # Only worth saying when the pin is something no profile would have
    # chosen: a pin that came from the profile control itself is not news.
    if pinned_mlp and tuple(int(v) for v in pinned_mlp) not in distinct:
        caveats.append(
            "The MLP vector pinned on the form ("
            + ",".join(str(int(v)) for v in pinned_mlp)
            + ") is not one of these working points; picking one replaces it."
        )
    if len(distinct) < len([r for r in rows if r.get("resolved")]):
        caveats.append(
            "Some stops resolve to the same configuration on this rig. That "
            "is the answer, not a gap: the capacity-optimal split IS the "
            "VRAM-auto split, and a modelled speed move inside the measured "
            "boot-to-boot noise floor is not worth a different configuration. "
            "What separates them further is a knob the offline model cannot "
            "predict (KV-token ownership, --rank-kv-ratio) and a probe of "
            "these cards' real rates."
        )

    return {
        "ok": True,
        "model": model_path,
        "baseline": BASELINE_KEY,
        "distinct_working_points": len(distinct),
        "slider": {
            "axis": SLIDER_AXIS,
            "steps": [
                {"key": p.key, "label": p.label, "index": i}
                for i, p in enumerate(PROFILES)
            ],
            "default_index": PROFILE_KEYS.index(BASELINE_KEY),
        },
        "session_target_context_tokens": int(session_target_context),
        # The rungs the balance report itself is solved for, so the control
        # offering them holds no list of its own.
        "session_target_options": [int(x) for x in mrr_balance.DEFAULT_TARGET_CONTEXTS],
        "basis": {
            "base_split": base_plan,
            "uniform_budgets": uniform,
            "candidates_considered": len(cands),
            "speed_scores": (
                None
                if scores is None
                else {
                    "sources": scores["sources"],
                    "measured": scores["measured"],
                    "membw_gbs": scores["membw_gbs"],
                    "gemm_tflops": scores["gemm_tflops"],
                }
            ),
            "min_link_gbs": min_link,
            "min_link_source": link_source,
        },
        "caveats": caveats,
        "profiles": rows,
    }
