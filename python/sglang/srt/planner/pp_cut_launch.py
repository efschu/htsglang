# SPDX-License-Identifier: Apache-2.0
"""Launcher-side wiring of the PP cut solver: makespan under a pool floor.

WHAT THIS ADDS, AND WHAT IT DELIBERATELY DOES NOT.

``planner/pp_cut.py`` already carries every piece: ``PrefillTiming`` +
``solve_pp_cut_for_prefill_speed`` price the TIME axis from measured per-layer
ms, ``PhasePoolModel`` + ``pp_phase_pool`` price the CAPACITY axis, and
``attention_counts`` states which full-attention layers a contiguous cut
actually lands on -- the second axis, and NOT a free allocation: see
``pp_cut.attention_split_is_realizable`` for why a solver that allocated
attention layers independently would rank cuts this runtime cannot run.
NO SECOND SOLVER IS WRITTEN HERE. This module only (a) supplies those
functions their inputs from things the launcher already knows, (b) applies the
Weg-2 constraint that group P's KV pool must hold one whole prompt, and (c)
prints one provenance line so the trade is on the record rather than in a
commit message.

WHY NOT ``--pp-solve-cut`` (server_args.py ``_handle_pp_solve_cut``, #1018).
That path is the right one and stays the right one -- but it consumes a
CENSUS DIRECTORY: measured residual MiB per rank, per-load-state transients,
seam staging, and a measured card-rate library, produced by a prior
instrumented boot on this exact layout. Weg-2 group P has never taken one.
Building ``PPCutInputs`` without it would mean inventing ``RankResources``
terms, and #1009's own lesson is that an unpriced term does not read as
"unknown", it reads as "free". So this module uses the LIGHTER pp_cut
entry points, whose every input the launcher can source or refuse, and names
the difference instead of hiding it. When a census exists, ``--pp-solve-cut``
is still the better instrument and this wiring should defer to it.

THE CONSTRAINT (user decision 1, 2026-09-07). Group P does not decode: it
frees a request's rows once its prefill completes. But DURING that prefill the
whole prefix of the running request must be device-resident for the stage's
attention layers, so P's pool must hold at least ONE full-context prompt. That
is a FLOOR on capacity, and makespan is minimised subject to it -- not traded
against it.

THE TWO OBJECTIVES ARE BOTH PRINTED (#1018). ``makespan`` is what is chosen;
the pool-maximal cut ("kv-floor") is priced on the same axes and printed
beside it, so a boot cannot pay the trade by accident.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.planner.pp_cut import (
    PhasePoolModel,
    PrefillTiming,
    attention_counts,
    pipelined_prefill_ms,
    pp_phase_pool,
    solve_pp_cut_for_prefill_speed,
)


class PPCutRefused(RuntimeError):
    """W40: no cut satisfies the pool floor. Never a silent fallback."""


@dataclasses.dataclass(frozen=True)
class CutCandidate:
    layers: Tuple[int, ...]
    attn: Tuple[int, ...]
    makespan_ms: float
    pool_tokens: float

    def fmt(self) -> str:
        return (
            f"{','.join(str(n) for n in self.layers)}"
            f" attn {','.join(str(a) for a in self.attn)}"
        )


@dataclasses.dataclass(frozen=True)
class CutDecision:
    chosen: CutCandidate
    kv_floor: CutCandidate
    cap_tokens: int
    pinned: bool
    cost_provenance: str

    def provenance_line(self) -> str:
        """THE one line. Format is load-bearing: a reader greps ``PP-CUT solver:``."""
        pin = " PINNED (user override)" if self.pinned else ""
        return (
            "PP-CUT solver:%s layers=%s attn=%s makespan_ms=%.1f "
            "pool_tokens=%d (constraint pool >= %d; alternatives: kv-floor cut "
            "%s pool %d makespan %.1f) [%s]"
            % (
                pin,
                ",".join(str(n) for n in self.chosen.layers),
                ",".join(str(a) for a in self.chosen.attn),
                self.chosen.makespan_ms,
                int(self.chosen.pool_tokens),
                int(self.cap_tokens),
                self.kv_floor.fmt(),
                int(self.kv_floor.pool_tokens),
                self.kv_floor.makespan_ms,
                self.cost_provenance,
            )
        )


def ms_per_layer_from_card_library(
    card_names: Sequence[str],
    measured_ms_per_layer: Sequence[float],
    incumbent: Sequence[int],
) -> Optional[Tuple[Tuple[float, ...], str]]:
    """Per-stage per-layer ms from the MEASURED card-rate library, or None.

    The library (``planner/card_rate_pass.load_measured_library``) is the
    solver's existing per-card score source, so it is PREFERRED over the
    boot-derived per-layer figures -- it is per CARD NAME and therefore
    robust to a change of NVML/CUDA ordering, which a per-stage list is not.

    It scores cards in TFLOP/s, not milliseconds, so it fixes the RATIOS and
    not the scale. One scalar anchor closes that: ``C`` is set so the
    library's stage times sum to the MEASURED stage times of the incumbent
    cut. The anchor is a single number over three stages, and the per-stage
    residual against the measurement is printed so a disagreement between the
    two sources is visible rather than absorbed.

    Returns ``None`` -- never a guess -- when no measured rate exists for one
    of the cards; the caller then falls back to the measured list and says so.
    """
    try:
        from sglang.srt.planner.card_rate_pass import load_measured_library
    except Exception:
        return None
    library = load_measured_library()
    if library is None:
        return None
    rates: List[float] = []
    for name in card_names:
        variant = None
        try:
            variants = library.variants(name)
            for cand in variants or ():
                if getattr(cand, "gemm_tflops", None):
                    variant = cand
                    break
        except Exception:
            variant = None
        if variant is None:
            return None
        rates.append(float(variant.gemm_tflops))
    inv = [float(n) / r for n, r in zip(incumbent, rates)]
    measured_total = sum(
        float(n) * float(m) for n, m in zip(incumbent, measured_ms_per_layer)
    )
    if sum(inv) <= 0.0 or measured_total <= 0.0:
        return None
    anchor = measured_total / sum(inv)
    derived = tuple(anchor / r for r in rates)
    residual = ", ".join(
        "%s %.2f vs measured %.2f ms/layer (%+.1f%%)"
        % (n, d, m, 100.0 * (d - m) / m)
        for n, d, m in zip(card_names, derived, measured_ms_per_layer)
    )
    return derived, (
        "cost=MEASURED card-rate library (gemm_tflops %s), anchored on the "
        "measured incumbent total %.1f ms -> %s"
        % (
            ", ".join("%s %.2f" % (n, r) for n, r in zip(card_names, rates)),
            measured_total,
            residual,
        )
    )


def solve_launch_cut(
    *,
    layer_families: Sequence[str],
    incumbent_layers: Sequence[int],
    measured_ms_per_layer: Sequence[float],
    measured_provenance: str,
    card_names: Sequence[str],
    pool_model: PhasePoolModel,
    cap_tokens: int,
    pinned_layers: Optional[Sequence[int]] = None,
    pinned_attn: Optional[Sequence[int]] = None,
    min_layers_per_stage: int = 1,
) -> CutDecision:
    """Choose the layer + attention cut. Makespan-optimal among the feasible.

    ``pinned_layers``/``pinned_attn``: when the operator passed the existing
    flags explicitly, they WIN -- but they win by being announced with the
    word PINNED and priced on the same two axes as the solved cut, never by
    silently replacing it. Same discipline as ``_handle_pp_solve_cut``'s
    "a VALIDATED OVERRIDE still wins when you want it".
    """
    n_stages = len(incumbent_layers)
    total_layers = len(layer_families)
    library = ms_per_layer_from_card_library(
        card_names, measured_ms_per_layer, incumbent_layers
    )
    if library is not None:
        ms_per_layer, cost_provenance = library
        cost_provenance += "; " + measured_provenance
    else:
        ms_per_layer = tuple(float(m) for m in measured_ms_per_layer)
        cost_provenance = (
            "cost=" + measured_provenance + " (no measured card-rate library "
            "on this rig -- `python -m sglang.srt.planner.card_rate_pass "
            "--run` would supply the preferred per-card source)"
        )
    timing = PrefillTiming(
        ms_per_layer=tuple(ms_per_layer), fixed_ms=tuple(0.0 for _ in ms_per_layer)
    )

    # rank0's layer cap is DERIVED, not chosen: the largest count whose
    # weights plus the arming floor still fit rank0's own budget. A cut above
    # it is arithmetic that ignores the card.
    spendable0 = float(pool_model.free_mib[0]) - float(pool_model.arming_floor_mib[0])
    max_rank0 = int(spendable0 // float(pool_model.weight_mib_per_layer))
    max_rank0 = max(1, min(int(total_layers) - (n_stages - 1), max_rank0))

    # BOTH AXES, JOINTLY AND ONLY IN THEIR REALIZABLE COMBINATIONS. The
    # enumeration is over contiguous cuts, and each cut's attention split is
    # the one its boundaries produce -- which is exactly the set
    # ``derive_pp_layer_split`` can be asked for. Cuts that share an
    # attention split but differ in layer counts (up to period-1 linear
    # layers sliding across a boundary at zero KV cost) are separate
    # candidates here, so the decoupling the attention flag buys is priced;
    # what is NOT enumerated is a free allocation, which is not realizable.
    attn_cache: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], Optional[float]]] = {}

    def resolve(counts: Sequence[int]) -> Tuple[Tuple[int, ...], Optional[float]]:
        key = tuple(int(c) for c in counts)
        if key not in attn_cache:
            attn = attention_counts(layer_families, key)
            try:
                pool = pp_phase_pool(key, attn, pool_model)
            except ValueError:
                pool = None
            attn_cache[key] = (attn, pool)
        return attn_cache[key]

    def pool_fn(counts: Sequence[int]) -> Optional[float]:
        return resolve(counts)[1]

    ranked = solve_pp_cut_for_prefill_speed(
        total_layers=int(total_layers),
        timing=timing,
        incumbent=tuple(int(n) for n in incumbent_layers),
        max_rank0_layers=max_rank0,
        min_layers_per_stage=int(min_layers_per_stage),
        pool_fn=pool_fn,
    )
    candidates: List[CutCandidate] = []
    for cand in ranked:
        attn, pool = resolve(cand.counts)
        if pool is None:
            continue
        candidates.append(
            CutCandidate(
                layers=tuple(cand.counts),
                attn=attn,
                makespan_ms=cand.pipelined_ms,
                pool_tokens=pool,
            )
        )
    if not candidates:
        raise PPCutRefused(
            "W40 Weg2PPCutRefused: not one cut of %d layers over %d stages is "
            "priceable at all (rank0 cap %d layers) -- the pool model refused "
            "every candidate, so there is nothing to choose between."
            % (int(total_layers), n_stages, max_rank0)
        )

    kv_floor = max(candidates, key=lambda c: c.pool_tokens)
    feasible = [c for c in candidates if c.pool_tokens >= float(cap_tokens)]
    if not feasible:
        raise PPCutRefused(
            "W40 Weg2PPCutRefused: no cut holds one full-context prompt. The "
            "pool floor is %d tokens (--max-kv-per-request); the best-pool cut "
            "is layers=%s attn=%s at %d tokens (makespan %.1f ms), short by %d. "
            "Lower --max-kv-per-request, raise the per-rank budgets, or fund "
            "the unpriced terms named in the cost line: %s"
            % (
                int(cap_tokens),
                ",".join(str(n) for n in kv_floor.layers),
                ",".join(str(a) for a in kv_floor.attn),
                int(kv_floor.pool_tokens),
                kv_floor.makespan_ms,
                int(cap_tokens) - int(kv_floor.pool_tokens),
                cost_provenance,
            )
        )
    chosen = min(feasible, key=lambda c: (c.makespan_ms, -c.pool_tokens))

    pinned = pinned_layers is not None
    if pinned:
        layers = tuple(int(n) for n in pinned_layers)
        derived_attn, derived_pool = resolve(layers)
        if pinned_attn is not None and tuple(int(a) for a in pinned_attn) != derived_attn:
            raise PPCutRefused(
                "W40 Weg2PPCutRefused: the pinned pair layers=%s attn=%s is "
                "not realizable -- a contiguous cut of those layer counts "
                "lands on attn=%s, and asking for anything else is SNAPPED "
                "rather than refused by derive_pp_layer_split (the #505(a) "
                "silent-substitution class). Pin the realizable pair, or pin "
                "only --pp-stage-ratio."
                % (
                    ",".join(str(n) for n in layers),
                    ",".join(str(a) for a in pinned_attn),
                    ",".join(str(a) for a in derived_attn),
                )
            )
        if derived_pool is None:
            raise PPCutRefused(
                "W40 Weg2PPCutRefused: the pinned layer cut %s cannot be "
                "priced by the pool model (a stage does not fit its weights, "
                "mamba state and arming floor, or holds no attention layer)."
                % (",".join(str(n) for n in layers),)
            )
        chosen = CutCandidate(
            layers=layers,
            attn=derived_attn,
            makespan_ms=pipelined_prefill_ms(layers, timing),
            pool_tokens=derived_pool,
        )

    return CutDecision(
        chosen=chosen,
        kv_floor=kv_floor,
        cap_tokens=int(cap_tokens),
        pinned=pinned,
        cost_provenance=cost_provenance,
    )
