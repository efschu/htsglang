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

THE TWO OBJECTIVES ARE BOTH PRINTED (#1018), AND THE DEFAULT IS ``maxkv``
(#1254). Both cuts are priced on the same axes and both appear on the
provenance line with their pool AND their ms, so a boot cannot pay the trade by
accident in EITHER direction.

Which one is CHOSEN is the ``objective`` argument, and its default is the
pool-maximal cut. That is the standing law -- trades live behind one objective
knob whose default is maximum KV -- and it is not a preference about this rig:
on the perf branch the makespan default measured 44,10,10 attn 11,2,3 with a
499,967-token pool, i.e. it was paying -47.7 % of the pool for +33.5 % of
prefill without anyone selecting that trade. ``makespan`` remains one flag
away, and its row is printed on every boot that does not take it.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.planner.pp_cut import (
    LAYER_FAMILY_ATTENTION,
    FamilyDepthCost,
    LayerSets,
    PhasePoolModel,
    PrefillTiming,
    UnpricedCrossing,
    attention_counts,
    attn_counts_of,
    contiguous_layer_sets,
    counts_of,
    crossing_price,
    enumerate_gapped_splits,
    gapped_layer_sets,
    gapped_phase_pool,
    is_gapped,
    layer_set_flag,
    pipelined_prefill_ms,
    pp_phase_pool,
    solve_pp_cut_for_prefill_speed,
)


class PPCutRefused(RuntimeError):
    """W40: no cut satisfies the pool floor. Never a silent fallback."""


def _gapped_forward_gate() -> Tuple[bool, str]:
    """Would the RUNTIME serve a gapped forward if this solver chose one?

    Returns the answer and the NAME of the escape hatch that changes it, both
    fetched from the runtime's own module rather than restated here: a second
    copy of the condition -- or even of the variable's name -- is a second set
    of books, and the day the #753 forward is fixed the gate moves in one place
    and this solver follows it without an edit. Imported lazily because this
    module is imported by the launcher long before any distributed import is
    wanted.
    """
    from sglang.srt.distributed.utils import (
        PP_GAPPED_KNOWN_WRONG_ENV,
        pp_gapped_forward_known_wrong_allowed,
    )

    return pp_gapped_forward_known_wrong_allowed(), PP_GAPPED_KNOWN_WRONG_ENV


def refuse_unfunded_posts(pool_model: PhasePoolModel) -> None:
    """W40 when a BOOT is about to be priced with posts nobody funded (#1286).

    The library keeps pricing an under-funded model, because a desk explorer
    asking "how does the pool move with the cut" gets a useful upper bound from
    it. A LAUNCH does not: the number it publishes is compared against
    ``--max-kv-per-request`` and printed as the boot's pool, and every missing
    post pushes it the same way -- UP. Measured on weg2sb5f: 499,967 published
    against 304,655 sized, +64.1 %, with the three posts below all at zero.

    So the refusal lives at the launch seam rather than in the model: one
    caller, one sentence, and the desk paths are untouched.
    """
    missing = pool_model.unfunded_posts
    if not missing:
        return
    raise PPCutRefused(
        "W40 Weg2PPCutRefused: the pool model would price this boot's cuts "
        "with %d of the boot's own budget posts UNFUNDED (%s). An unpriced "
        "term does not read as 'unknown', it reads as 'free' (#1009), and "
        "every one of these is the OVER-pricing direction: on boot weg2sb5f "
        "the same omissions published 499,967 tokens for the cut group P then "
        "sized at 304,655 (+64.1 %%), which is a pool floor cleared on paper "
        "by a layout that could not hold the prompt. Fund them from the boot's "
        "'KV budget posts (GiB):' line -- weights + runtime state minus the "
        "per-layer half is --pp-cut-stage-fixed-mib, 'prefill activation "
        "reserve' is --pp-cut-activation-reserve-mib, 'gapped corridor "
        "holdback' is --pp-cut-corridor-holdback-mib, 'mamba state pool' "
        "divided by (linear layers x slots) is "
        "--pp-cut-mamba-mib-per-linear-layer-per-slot. A post the boot does "
        "NOT charge on this form is funded by NAMING it in "
        "PhasePoolModel.zero_posts_acknowledged, which states the claim "
        "instead of leaving the field silently at zero."
        % (len(missing), ", ".join(missing))
    )


def refuse_pool_model_geometry(pool_model: PhasePoolModel, stages: int) -> None:
    """W40 when the pool model's PER-STAGE vectors are not this launch's shape.

    #1286 F5, and it is a SWALLOW, not a typo. ``_stage_free_after_residency``
    raises a bare ``ValueError`` for a ``stage_fixed_mib`` whose length is not
    the stage count -- and ``solve_launch_cut``'s ``resolve()`` catches bare
    ``ValueError`` and turns it into ``pool = None``, which is its way of
    saying "this CUT does not fit". So a configuration error came out the far
    side as ``not one cut of %d layers over %d stages is priceable``: a
    sentence about geometry, for a fault that has nothing to do with geometry,
    with the actual cause nowhere in it. EVERY candidate fails identically,
    which is exactly what makes the wrong sentence convincing.

    ``budgets_p`` is ``len(cards)``, so any rig or card selection that is not
    exactly three meets a three-entry default and lands here.

    Checked HERE rather than by narrowing the catch: the catch is RIGHT --
    ``stage_pp_capacities`` legitimately raises ``ValueError`` for an
    infeasible cut, and that genuinely is "unpriceable". What was wrong was
    asking a per-candidate loop to diagnose a per-LAUNCH input. One check,
    before the loop, with its own sentence.
    """
    for name, vec in (
        ("--pp-cut-stage-fixed-mib", pool_model.stage_fixed_mib),
        ("the per-rank budgets (free_mib)", pool_model.free_mib),
        ("--pp-cut-arming-floor-mib", pool_model.arming_floor_mib),
    ):
        if not vec:
            continue
        if len(tuple(vec)) != int(stages):
            raise PPCutRefused(
                "W40 Weg2PPCutRefused: %s has %d entries for a %d-stage "
                "launch. These are PER-STAGE posts -- the embedding sits on "
                "stage 0, lm_head and the draft head on the last -- so they "
                "cannot be broadcast, and a length mismatch is a launch INPUT "
                "error, not an unpriceable layout. Fix the flag: no cut of any "
                "shape would have helped, which is why the solver's own "
                "geometry refusal must not be the sentence you get here "
                "(#1286 F5). Values: %s"
                % (
                    name,
                    len(tuple(vec)),
                    int(stages),
                    ", ".join("%.1f" % float(x) for x in vec),
                )
            )


def _refuse_below_pool_floor(
    candidate: "CutCandidate", what: str, cap_tokens: int, cost_provenance: str
) -> None:
    """W40 when a chosen layout cannot hold one full-context prompt.

    The SOLVED path has refused this since #1236 (``feasible`` is the set that
    clears the floor). The PINNED paths returned without ever asking, so boot
    weg2pp2's dry run printed ``pool_tokens=153611 (constraint pool >=
    262144)`` beside a pinned map and went on to build the argv -- a boot that
    cannot admit one full-context prompt is a spent window, and the sentence
    that names it already existed. One writer of that sentence, three callers.
    """
    if float(candidate.pool_tokens) >= float(cap_tokens):
        return
    raise PPCutRefused(
        "W40 Weg2PPCutRefused: %s does not hold one full-context prompt. The "
        "pool floor is %d tokens (--max-kv-per-request); this layout is "
        "layers=%s attn=%s at %d tokens (makespan %.1f ms, crossings %.1f ms), "
        "short by %d. Lower --max-kv-per-request, raise the per-rank budgets, "
        "or pin a layout that holds it. Cost model: %s"
        % (
            what,
            int(cap_tokens),
            ",".join(str(n) for n in candidate.layers),
            ",".join(str(a) for a in candidate.attn),
            int(candidate.pool_tokens),
            candidate.makespan_ms,
            candidate.crossing_ms,
            int(cap_tokens) - int(candidate.pool_tokens),
            cost_provenance,
        )
    )


@dataclasses.dataclass(frozen=True)
class CutCandidate:
    """One layout, priced on every axis the ranking uses.

    ``makespan_ms`` and ``crossing_ms`` are SEPARATE columns because they are
    separate physics -- compute on the cards, bytes on the links -- and the
    whole point of printing a table is that a reader can see which of the two
    a candidate lost on. The objective is their sum (:attr:`total_ms`).

    ``makespan_ms`` is the compute bound at the number of passes the layout
    ADMITS, which is not the same rule twice: a contiguous cut may run several
    passes in flight (``--pp-async-batch-depth``), so its bound is the slowest
    stage; a gapped map may not (the launcher's own W43, from
    ``scheduler_pp_mixin.init_pp_loop_state``), so its stages do not overlap
    and the bound is their sum.
    """

    layers: Tuple[int, ...]
    attn: Tuple[int, ...]
    makespan_ms: float
    pool_tokens: float
    kind: str = "contiguous"
    crossing_ms: float = 0.0
    crossings: int = 0
    depth_tokens: int = 0
    owned: Optional[LayerSets] = None
    layer_set: str = ""

    @property
    def total_ms(self) -> float:
        return float(self.makespan_ms) + float(self.crossing_ms)

    def fmt(self) -> str:
        return (
            f"{','.join(str(n) for n in self.layers)}"
            f" attn {','.join(str(a) for a in self.attn)}"
        )

    def line(self, marker: str) -> str:
        """THE table row. Field names are load-bearing: a reader greps them."""
        return (
            "PP-CUT solver: %s layers=%s attn=%s makespan_ms=%.1f "
            "crossing_ms=%.1f pool_tokens=%d depth_tokens=%d %s"
            % (
                self.kind,
                ",".join(str(n) for n in self.layers),
                ",".join(str(a) for a in self.attn),
                self.makespan_ms,
                self.crossing_ms,
                int(self.pool_tokens),
                int(self.depth_tokens),
                marker,
            )
        )


@dataclasses.dataclass(frozen=True)
class CutDecision:
    chosen: CutCandidate
    kv_floor: CutCandidate
    cap_tokens: int
    pinned: bool
    cost_provenance: str
    ranked: Tuple[CutCandidate, ...] = ()
    design_prefix_tokens: int = 0
    unpriced: Tuple[str, ...] = ()
    #: WHICH OBJECTIVE PICKED ``chosen`` (#1254). Default ``maxkv``.
    objective: str = "maxkv"
    #: The makespan-optimal feasible cut. Always populated, whether or not it
    #: was chosen, so the trade is visible in BOTH directions -- the kv-floor
    #: row alone only exposed the trade a makespan default was making.
    makespan: Optional[CutCandidate] = None

    def table_lines(self, top: int = 8) -> List[str]:
        """The chosen candidate and its alternatives, best first.

        Bounded rather than exhaustive: the contiguous enumeration alone is in
        the hundreds and a boot log is read by people. The bound is stated in
        the trailing line so a truncated table cannot be read as the whole
        field -- the same denominator discipline the rest of this strand runs
        on.
        """

        def same_as_chosen(c: CutCandidate) -> bool:
            return (
                c.kind == self.chosen.kind
                and c.layers == self.chosen.layers
                and c.attn == self.chosen.attn
            )

        out: List[str] = []
        shown = set()
        for i, cand in enumerate(self.ranked[: int(top)]):
            shown.add(i)
            out.append(cand.line("CHOSEN" if same_as_chosen(cand) else "alt#%d" % i))
        # THE CHOSEN ROW IS NEVER MISSING. A PINNED map wins the ranking
        # without necessarily ranking well, so it can sit far outside the
        # top-N -- and a table whose every row says "alt" while the
        # provenance line names something else is worse than no table.
        if not any("CHOSEN" in line for line in out):
            at = next((i for i, c in enumerate(self.ranked) if same_as_chosen(c)), None)
            shown.add(at)
            out.append(
                self.chosen.line("CHOSEN" if at is None else "CHOSEN (ranked #%d)" % at)
            )
        # EVERY KIND GETS A ROW, wherever it ranks. A top-N cut through a
        # ranking that happens to be swept by one family would print no gapped
        # candidate at all, and an absent row reads as an absent candidate --
        # the denominator trap, applied to a whole family. The best of each
        # kind is therefore always shown, with its true rank index, so the
        # trade the table exists to expose cannot be hidden by truncation.
        seen_kinds = {c.kind for c in self.ranked[: int(top)]}
        for kind in sorted({c.kind for c in self.ranked} - seen_kinds):
            best = next(
                (
                    i
                    for i, c in enumerate(self.ranked)
                    if c.kind == kind and i not in shown
                ),
                None,
            )
            if best is None:
                continue
            shown.add(best)
            out.append(self.ranked[best].line("best-%s#%d" % (kind, best)))
        # And the pool-maximal candidate, on the same row format as the rest:
        # it is the other objective, and #1018's rule is that a boot cannot pay
        # that trade by accident.
        if not same_as_chosen(self.kv_floor):
            out.append(self.kv_floor.line("kv-floor"))
        # ... and the makespan-optimal one, for exactly the same reason with
        # the sign flipped. With the default at maxkv the row a reader needs in
        # order to see the trade is the FAST one; printing only kv-floor would
        # be the old asymmetry pointing the other way.
        if self.makespan is not None and not same_as_chosen(self.makespan):
            out.append(self.makespan.line("makespan"))
        if len(self.ranked) > int(top):
            out.append(
                "PP-CUT solver: %d further candidates ranked and not printed "
                "(%d contiguous, %d gapped in all)"
                % (
                    len(self.ranked) - int(top),
                    sum(1 for c in self.ranked if c.kind == "contiguous"),
                    sum(1 for c in self.ranked if c.kind == "gapped"),
                )
            )
        for note in self.unpriced:
            out.append("PP-CUT solver: UNPRICED %s" % note)
        return out

    def provenance_line(self) -> str:
        """THE one line. Format is load-bearing: a reader greps ``PP-CUT solver:``.

        BOTH objectives' cuts appear with pool AND ms (#1254), always, whichever
        was chosen. A line that priced only the alternative it did not take
        could not be read as a trade at all -- only as a defence of the choice.
        """
        pin = " PINNED (user override)" if self.pinned else ""
        return (
            "PP-CUT solver:%s objective=%s layers=%s attn=%s makespan_ms=%.1f "
            "crossing_ms=%.1f pool_tokens=%d (constraint pool >= %d; BOTH "
            "objectives priced: maxkv cut %s pool %d total %.1f ms/chunk | "
            "makespan cut %s pool %d total %.1f ms/chunk) [%s]"
            % (
                pin,
                self.objective,
                ",".join(str(n) for n in self.chosen.layers),
                ",".join(str(a) for a in self.chosen.attn),
                self.chosen.makespan_ms,
                self.chosen.crossing_ms,
                int(self.chosen.pool_tokens),
                int(self.cap_tokens),
                self.kv_floor.fmt(),
                int(self.kv_floor.pool_tokens),
                self.kv_floor.total_ms,
                (self.makespan or self.chosen).fmt(),
                int((self.makespan or self.chosen).pool_tokens),
                (self.makespan or self.chosen).total_ms,
                self.cost_provenance,
            )
        )

    def trade_line(self) -> str:
        """What taking the OTHER objective would have cost, in both currencies.

        Separate from the provenance line on purpose: that line is greppable
        and its shape is depended on, while this one is arithmetic over it and
        exists so nobody has to do the division in their head to see whether
        the trade is small (a preference) or large (a defect candidate -- the
        law's own words).
        """
        other = self.makespan if self.objective == "maxkv" else self.kv_floor
        if other is None or (
            other.layers == self.chosen.layers and other.attn == self.chosen.attn
        ):
            return (
                "PP-CUT trade: none -- the maxkv and makespan objectives choose "
                "the SAME cut %s at %d tokens / %.1f ms per chunk, so this boot "
                "pays nothing for its default." % (
                    self.chosen.fmt(), int(self.chosen.pool_tokens),
                    self.chosen.total_ms,
                )
            )
        d_pool = (other.pool_tokens - self.chosen.pool_tokens) / max(
            1.0, self.chosen.pool_tokens
        )
        d_ms = (other.total_ms - self.chosen.total_ms) / max(1.0, self.chosen.total_ms)
        return (
            "PP-CUT trade: objective=%s chose %s (pool %d, %.1f ms/chunk); the "
            "%s cut %s would be pool %+.1f %% and time %+.1f %% (%d tokens, "
            "%.1f ms/chunk). Selectable with --pp-solve-objective."
            % (
                self.objective,
                self.chosen.fmt(),
                int(self.chosen.pool_tokens),
                self.chosen.total_ms,
                "makespan" if self.objective == "maxkv" else "maxkv",
                other.fmt(),
                100.0 * d_pool,
                100.0 * d_ms,
                int(other.pool_tokens),
                other.total_ms,
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
        "%s %.2f vs measured %.2f ms/layer (%+.1f%%)" % (n, d, m, 100.0 * (d - m) / m)
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
    family_cost: Optional[FamilyDepthCost] = None,
    design_prefix_tokens: Optional[int] = None,
    per_pair_crossing_ms: Optional[Mapping[Tuple[int, int], float]] = None,
    enumerate_gapped: bool = True,
    pinned_layer_set: Optional[str] = None,
    objective: str = "maxkv",
) -> CutDecision:
    """Choose the layer + attention cut, for ``objective``, among the feasible.

    ``objective`` is ``"maxkv"`` (default) or ``"makespan"``. maxkv takes the
    pool-maximal feasible cut; makespan takes the one with the smallest
    compute+crossing total. BOTH are priced and BOTH appear on the provenance
    line either way, so the trade is a decision and not a side effect.

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
    # ONE PRICING FUNCTION FOR BOTH FAMILIES (#1240). Without a
    # ``family_cost`` this is exactly what it always was: ``PrefillTiming``'s
    # per-stage total, no depth, no crossings, contiguous only. With one, the
    # SAME function prices every candidate of both kinds, which is the only
    # way a gapped map and a contiguous cut can be compared rather than merely
    # listed next to each other.
    depth = int(
        design_prefix_tokens
        if design_prefix_tokens is not None
        else (family_cost.ref_prefix_tokens if family_cost is not None else 0)
    )
    pair_ms: Mapping[Tuple[int, int], float] = per_pair_crossing_ms or {}
    unpriced: List[str] = []

    def price(
        owned: LayerSets, kind: str, fallback_ms: Optional[float] = None
    ) -> Optional[CutCandidate]:
        counts = counts_of(owned)
        attn = attn_counts_of(layer_families, owned)
        try:
            pool = (
                gapped_phase_pool(counts, attn, pool_model)
                if kind == "gapped"
                else pp_phase_pool(counts, attn, pool_model)
            )
        except ValueError:
            return None
        if family_cost is None:
            if fallback_ms is None:
                return None
            makespan, cross, n_cross = float(fallback_ms), 0.0, 0
        else:
            stage_ms = family_cost.stage_ms(counts, attn, depth)
            # The in-flight-pass rule, applied per candidate -- see
            # ``CutCandidate``. Gapped forbids depth, so its stages serialise.
            makespan = float(sum(stage_ms) if kind == "gapped" else max(stage_ms))
            try:
                cp = crossing_price(owned, int(total_layers), pair_ms)
            except UnpricedCrossing as exc:
                unpriced.append(
                    "%s layers=%s attn=%s: %s"
                    % (
                        kind,
                        ",".join(str(n) for n in counts),
                        ",".join(str(a) for a in attn),
                        exc,
                    )
                )
                return None
            cross, n_cross = cp.ms, cp.crossings
        return CutCandidate(
            layers=counts,
            attn=attn,
            makespan_ms=makespan,
            pool_tokens=pool,
            kind=kind,
            crossing_ms=cross,
            crossings=n_cross,
            depth_tokens=depth,
            owned=owned,
            layer_set=layer_set_flag(owned),
        )

    candidates: List[CutCandidate] = []
    for cand in ranked:
        _, pool = resolve(cand.counts)
        if pool is None:
            continue
        priced = price(
            contiguous_layer_sets(cand.counts), "contiguous", cand.pipelined_ms
        )
        if priced is not None:
            candidates.append(priced)

    # THE GAPPED FIELD. All linear layers on ONE stage -- which stage is
    # enumerated, not assumed, because "the 5090" is a fact about this rig and
    # not about the solver -- and the attention layers dealt out in ascending
    # id by every admissible split. That is the family the contiguous
    # enumeration cannot reach at all, and the user's 0/8/8 and 4/6/6 both
    # live in it.
    if enumerate_gapped and family_cost is not None:
        n_attn_total = sum(1 for f in layer_families if f == LAYER_FAMILY_ATTENTION)
        for gdn_stage in range(n_stages):
            for split in enumerate_gapped_splits(n_attn_total, n_stages, gdn_stage):
                owned = gapped_layer_sets(layer_families, gdn_stage, split)
                if not is_gapped(owned):
                    # A "gapped" map whose every stage is an unbroken run IS a
                    # contiguous cut, already enumerated above. Ranking it
                    # twice would put the same layout in the table under two
                    # names and, worse, charge it the crossing schedule of a
                    # protocol it would not run.
                    continue
                priced = price(owned, "gapped")
                if priced is not None:
                    candidates.append(priced)

    if not candidates:
        raise PPCutRefused(
            "W40 Weg2PPCutRefused: not one cut of %d layers over %d stages is "
            "priceable at all (rank0 cap %d layers) -- the pool model refused "
            "every candidate, so there is nothing to choose between."
            % (int(total_layers), n_stages, max_rank0)
        )

    # THE RUNTIME GATE BOUNDS THE CHOICE (#1240 FOLLOW FIX 1, boot weg2pp2).
    # A gapped map is priced and printed -- its prices are the answer to the
    # question this slice exists to ask -- but it may not be CHOSEN while
    # scheduler_pp_mixin._refuse_known_wrong_gapped_forward stands, because
    # that refusal fires on all three ranks after the weights are loaded and
    # turns a solved layout into a dead window. The condition is not restated
    # here: the same predicate both readers use is imported. Only the ranking's
    # accident (#498 and #654 of 1,531 at the design prefix that boot measured)
    # kept the launcher from publishing an unservable layout, and a deeper
    # design prefix -- the direction the user is heading -- makes a gapped
    # candidate win. An accident is not a gate.
    gapped_servable, gate_env = _gapped_forward_gate()
    n_gapped = sum(1 for c in candidates if c.kind == "gapped")
    if n_gapped and not gapped_servable:
        unpriced.append(
            "REFUSED (unservable, not outranked): %d gapped candidate(s) are "
            "priced and printed but excluded from the choice by "
            "scheduler_pp_mixin._refuse_known_wrong_gapped_forward, which "
            "refuses a gapped forward as NUMERICALLY WRONG (measured "
            "2026-08-18: 'Paris' becomes '\\n\\n'). Fixing that forward is "
            "what unblocks this family; %s=1 prices them as choosable for "
            "debugging it." % (n_gapped, gate_env)
        )
    choosable = [c for c in candidates if gapped_servable or c.kind != "gapped"]
    if not choosable:
        raise PPCutRefused(
            "W40 Weg2PPCutRefused: every priceable candidate is a GAPPED map "
            "and the runtime refuses a gapped forward "
            "(scheduler_pp_mixin._refuse_known_wrong_gapped_forward), so there "
            "is nothing servable to choose between. %s" % (unpriced[-1],)
        )

    # The kv-floor row is the OTHER objective, so it has to be a layout that
    # could actually be run: a pool on an unservable map is not available.
    kv_floor = max(choosable, key=lambda c: c.pool_tokens)
    # THE MAKESPAN ROW IS COMPUTED HERE TOO, over the same set, so it exists on
    # every return path -- including the PINNED ones, which return before the
    # feasible set is formed. Restricted to the cuts that clear the pool floor
    # when any do, because an alternative a boot could not run is not an
    # alternative; when none do, the whole field is the honest set to name and
    # the refusal below is what the operator actually sees.
    #
    # WHY #1286 DID NOT MOVE THIS ROW, ARGUED RATHER THAN OBSERVED. Funding the
    # four missing posts changes every candidate's price by the SAME per-rank
    # amount -- `-stage_fixed[r] - activation_reserve - mamba + (arming_floor -
    # corridor_holdback)` -- which is negative on every stage of this rig
    # (stage_fixed alone is >= 1105 MiB against a +205 MiB holdback
    # correction), and none of it depends on the cut. So every price falls,
    # `_floor_ok` can only SHRINK, and `total_ms` is untouched: a winner that
    # still clears the floor is still the winner. The kv-floor row has no such
    # protection -- it ranks BY the number that moved, and there the correction
    # inverted the order (#1286: 31,17,16 fell from pool-maximal to below the
    # incumbent). Two objectives, two different exposures to the same repair.
    _floor_ok = [c for c in choosable if c.pool_tokens >= float(cap_tokens)]
    makespan_row = min(
        _floor_ok or choosable, key=lambda c: (c.total_ms, -c.pool_tokens)
    )
    # THE OBJECTIVE IS THE SUM of the two time columns. Ranking on makespan
    # alone would hand a gapped map the crossings for free -- 31 per chunk at
    # a 40 MiB frame is not a rounding term -- and ranking on crossings alone
    # would pick the layout that moves the fewest bytes and computes slowest.
    candidates.sort(key=lambda c: (c.total_ms, -c.pool_tokens))

    # THE PINNED PATHS ARE PRICED BEFORE THE SOLVED FIELD IS JUDGED (FOLLOW
    # FIX 2 / finding 3). A pin decides the layout, so the SOLVED field's
    # feasibility is not the operator's question and must not preempt the
    # pin's own refusal: with the floor above every servable candidate, a
    # pinned map that HOLDS the prompt used to be answered with "no cut holds
    # one full-context prompt", which is false about the layout that was
    # actually asked for -- and it hid both the gate that refuses it and the
    # pinned map's own pool. Each pinned branch still refuses on the same
    # floor, through ``_refuse_below_pool_floor``; only the ORDER changed.
    if pinned_layer_set:
        # A PINNED MAP wins the same way a pinned cut does: announced, and
        # priced on every axis the solved one was, never by skipping the
        # pricing. Parsed by the RUNTIME's own parser so the string that
        # reaches the boot is the string this decision was priced on.
        from sglang.srt.distributed.utils import parse_pp_layer_sets

        owned = tuple(
            tuple(sorted(int(i) for i in s))
            for s in parse_pp_layer_sets(
                str(pinned_layer_set), int(total_layers), n_stages, allow_gapped=True
            )
        )
        kind = "gapped" if is_gapped(owned) else "contiguous"
        priced = price(owned, kind)
        if priced is None:
            raise PPCutRefused(
                "W40 Weg2PPCutRefused: the pinned --pp-layer-set %r cannot be "
                "priced (a stage does not fit its weights, mamba state and "
                "arming floor, no stage holds a full-attention layer, or a "
                "crossing pair has no measured link price). The refusals above "
                "name which: %s"
                % (str(pinned_layer_set), "; ".join(unpriced) or "pool model")
            )
        if kind == "gapped" and not gapped_servable:
            raise PPCutRefused(
                "W40 Weg2PPCutRefused: the pinned --pp-layer-set %r is a "
                "GAPPED map, and the runtime refuses a gapped forward as "
                "NUMERICALLY WRONG "
                "(scheduler_pp_mixin._refuse_known_wrong_gapped_forward; "
                "measured 2026-08-18, 'Paris' becomes '\\n\\n'). Boot weg2pp2 "
                "reached that refusal on all three ranks AFTER the weights "
                "were loaded, which is a spent window rather than a decision. "
                "Refused here instead. %s=1 reaches it anyway while debugging "
                "the forward." % (str(pinned_layer_set), gate_env)
            )
        _refuse_below_pool_floor(
            priced,
            "the pinned --pp-layer-set %r" % (str(pinned_layer_set),),
            int(cap_tokens),
            cost_provenance,
        )
        return CutDecision(
            chosen=priced,
            kv_floor=kv_floor,
            cap_tokens=int(cap_tokens),
            pinned=True,
            cost_provenance=cost_provenance,
            ranked=tuple(candidates),
            design_prefix_tokens=depth,
            unpriced=tuple(unpriced),
            objective=str(objective),
            makespan=makespan_row,
        )

    pinned = pinned_layers is not None
    if pinned:
        layers = tuple(int(n) for n in pinned_layers)
        derived_attn, derived_pool = resolve(layers)
        if (
            pinned_attn is not None
            and tuple(int(a) for a in pinned_attn) != derived_attn
        ):
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
        priced = price(
            contiguous_layer_sets(layers),
            "contiguous",
            pipelined_prefill_ms(layers, timing),
        )
        if priced is None:
            raise PPCutRefused(
                "W40 Weg2PPCutRefused: the pinned layer cut %s cannot be "
                "priced on every axis (pool, family compute, crossings)."
                % (",".join(str(n) for n in layers),)
            )
        _refuse_below_pool_floor(
            priced,
            "the pinned layer cut %s" % (",".join(str(n) for n in layers),),
            int(cap_tokens),
            cost_provenance,
        )
        chosen = priced
    else:
        feasible = [c for c in choosable if c.pool_tokens >= float(cap_tokens)]
        if not feasible:
            # THE POOL QUESTION IS ANSWERED ON THE REFUSAL, NOT ONLY ON
            # SUCCESS. ``table_lines()`` is reached only when a cut is chosen,
            # so this raise is the operator's whole view of the field -- and
            # the case it fires in is exactly the one the gapped family exists
            # for: the servable (contiguous) maps are too small for the
            # context, and a priced gapped map holds it. Dropping ``unpriced``
            # here discarded the gate's own sentence on the one path where it
            # is the answer.
            held = [
                c
                for c in candidates
                if c not in choosable and c.pool_tokens >= float(cap_tokens)
            ]
            if held:
                best_held = max(held, key=lambda c: c.pool_tokens)
                trade = (
                    " A %s map DOES hold it -- layers=%s attn=%s at %d tokens "
                    "(makespan %.1f ms, %d crossings) -- and it is EXCLUDED "
                    "from the choice, not outranked."
                    % (
                        best_held.kind,
                        ",".join(str(n) for n in best_held.layers),
                        ",".join(str(a) for a in best_held.attn),
                        int(best_held.pool_tokens),
                        best_held.makespan_ms,
                        best_held.crossings,
                    )
                )
            else:
                trade = ""
            raise PPCutRefused(
                "W40 Weg2PPCutRefused: no SERVABLE cut holds one full-context "
                "prompt. The pool floor is %d tokens (--max-kv-per-request); "
                "the best servable cut is layers=%s attn=%s at %d tokens "
                "(makespan %.1f ms), short by %d.%s Lower "
                "--max-kv-per-request, raise the per-rank budgets, or fund the "
                "unpriced terms named in the cost line: %s%s"
                % (
                    int(cap_tokens),
                    ",".join(str(n) for n in kv_floor.layers),
                    ",".join(str(a) for a in kv_floor.attn),
                    int(kv_floor.pool_tokens),
                    kv_floor.makespan_ms,
                    int(cap_tokens) - int(kv_floor.pool_tokens),
                    trade,
                    cost_provenance,
                    (" -- " + "; ".join(unpriced)) if unpriced else "",
                )
            )
        # THE OBJECTIVE PICKS, AND BOTH ROWS ARE KEPT (#1254). maxkv is the
        # default because the standing law puts the default at maximum KV;
        # makespan is one flag away and is priced on the same line either way.
        # Ties break on the OTHER axis in both arms, so an objective never
        # spends capacity or time it did not have to.
        chosen = (
            max(feasible, key=lambda c: (c.pool_tokens, -c.total_ms))
            if objective == "maxkv"
            else min(feasible, key=lambda c: (c.total_ms, -c.pool_tokens))
        )

    return CutDecision(
        chosen=chosen,
        kv_floor=kv_floor,
        cap_tokens=int(cap_tokens),
        pinned=pinned,
        cost_provenance=cost_provenance,
        ranked=tuple(candidates),
        design_prefix_tokens=depth,
        unpriced=tuple(unpriced),
        objective=str(objective),
        makespan=makespan_row,
    )
