"""Spreading decider for the multi-group runtime (#274 slice D, v1).

The solver optimizes a lane's KEY under a given placement and says so in its
own scope line: it never invents lanes, and enumerating lane STRUCTURES is
"the dispatcher's search space (Slice D)".  This module is that one step: given
N lanes with load-shape labels and a rig, it chooses WHICH CARDS each lane
runs on.  It does not choose keys, does not size budgets and does not route
requests -- those stay with ``key_solver.solve_lanes`` and with #279.

THE OBJECTIVE IS MEASURED, NOT ASSUMED.  Slice C measured the same pair of
classes on one card with two different lane load shapes (DESIGN_121 §11.5):

    lane 2048-token prefill (SM-saturating) x serving decode -> E = 1.130
    lane decode-shaped      (latency-bound) x serving decode -> E = 1.440

Both arms share the serving side, so the difference is the lane's load shape
alone: two SM-saturating loads cannot both run at full speed on one card, and
concurrency there only collects the gaps -- exactly +16 % of them.  A
latency-bound load overlaps for real.  The decider therefore has one rule, and
it is the one the numbers support: DO NOT PUT AN SM-SATURATING LOAD ON A CARD
THAT ALREADY CARRIES ANOTHER LANE.

WHERE THE LABEL COMES FROM.  Not from a profiler -- DCGM's profiling metrics
are not reliable on GeForce, and DESIGN_201 addendum 12 (3) settles the
question by labelling the load shape ANALYTICALLY instead: the cost model
already knows a step's parameter count and its streamed bytes, so it knows the
step's arithmetic intensity, and ``RigRates`` already knows each card's machine
balance.  ``intensity >= balance`` is compute-bound, below it is
bandwidth-bound.  Both operands exist today; nothing new has to be probed.

WHAT IS NOT CLAIMED.  Two regimes carry no measurement and the decider says so
rather than interpolating: two SM-saturating lanes on ONE card (bounded above
by the 1.130 point, never given a number of its own), and three or more lanes
on one card.  In both cases the RANKING still works -- it only needs the
ordering, which is measured -- but ``expected_e`` comes back ``None`` with a
reason, in the same measured/estimate/absent spirit as the solver's cells.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MEASURED_PAIR_E",
    "CardPlacement",
    "LaneLoad",
    "SpreadAnswer",
    "label_from_intensity",
    "lane_load_from_model",
    "spread_plan",
    "step_intensity",
]

SM_BOUND: str = "sm"
BW_BOUND: str = "bw"
UNKNOWN: str = "unknown"

# Card equivalents measured for TWO lanes on one card, keyed by how many of
# them are SM-saturating (DESIGN_121 §11.5, slice C, 27B-Q3 GGUF TP=3 uneven,
# lane on the 5090).  These are rig points, not a fitted model: what the
# decider uses is their ORDER, and the order is what was measured.
MEASURED_PAIR_E: Dict[int, float] = {
    0: 1.440,  # decode-shaped lane x serving decode
    1: 1.130,  # 2048-token-prefill lane x serving decode
}
# Two saturating lanes on one card was never measured. It cannot be BETTER
# than the one-saturating case (the gaps a second saturating load can collect
# are a subset), so the decider treats 1.130 as an upper bound for ranking and
# refuses to report it as an expected E.
SM_PAIR_UPPER_BOUND: float = MEASURED_PAIR_E[1]

# Enumeration cap. The decider is a chooser over a small rig, not a search
# engine; past this it stops and says it stopped.
MAX_PLACEMENTS: int = 20000


# ---------------------------------------------------------------------------
# 1. The label
# ---------------------------------------------------------------------------


def step_intensity(
    *,
    params: float,
    weight_bytes: float,
    tokens: int,
) -> Optional[float]:
    """FLOP per byte of one step ``tokens`` wide.

    ``2 * params`` is the usual FLOP proxy for a transformer step and is the
    same one ``KeyCostModel.prefill_compute_time`` uses; ``weight_bytes`` is
    what the step streams once.  The token width is therefore the whole
    discriminant for a fixed dtype -- which is exactly why batch and chunk
    size are the arithmetic-intensity knob (DESIGN_201 addendum 12 (2)).
    """
    if weight_bytes <= 0.0 or params <= 0.0:
        return None
    return 2.0 * params * max(int(tokens), 1) / weight_bytes


def machine_balance(gemm_tflops: float, membw_gbs: float) -> Optional[float]:
    """The card's own FLOP/byte break-even, from its measured rates."""
    if not gemm_tflops or not membw_gbs:
        return None
    return (gemm_tflops * 1e12) / (membw_gbs * 1e9)


def label_from_intensity(intensity: Optional[float], balance: Optional[float]) -> str:
    if intensity is None or balance is None or balance <= 0.0:
        return UNKNOWN
    return SM_BOUND if intensity >= balance else BW_BOUND


@dataclasses.dataclass(frozen=True)
class LaneLoad:
    """One lane as the decider sees it: a load shape and a placement freedom.

    ``cards`` pins the lane to a card set; leave it empty and give
    ``card_count`` plus optionally ``allowed_cards`` to let the decider place
    it.  ``label`` is the analytic load-shape label -- build it with
    :func:`lane_load_from_model` rather than by hand where a cost model
    exists.
    """

    key: str
    label: str = UNKNOWN
    cards: Tuple[int, ...] = ()
    card_count: int = 1
    allowed_cards: Tuple[int, ...] = ()
    intensity: Optional[float] = None
    balance: Optional[float] = None
    priority_class: int = 0
    basis: str = ""

    @property
    def fixed(self) -> bool:
        return bool(self.cards)

    def to_json(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "cards": list(self.cards),
            "card_count": self.card_count,
            "allowed_cards": list(self.allowed_cards),
            "intensity_flop_per_byte": (
                None if self.intensity is None else round(self.intensity, 3)
            ),
            "balance_flop_per_byte": (
                None if self.balance is None else round(self.balance, 3)
            ),
            "priority_class": self.priority_class,
            "basis": self.basis,
        }


def lane_load_from_model(
    key: str,
    model,
    units: Sequence[int],
    rates,
    *,
    tokens: int,
    rank: int = 0,
    **kw,
) -> LaneLoad:
    """Label a lane from a :class:`key_solver.KeyCostModel`.

    ``tokens`` is the width of the step that DOMINATES this lane's duty: the
    chunked-prefill chunk size for a prefill lane, ``batch x (draft + 1)`` for
    a decode lane.  It is a parameter and not a guess because it is the whole
    discriminant -- the same weights are compute-bound at 2048 tokens and
    bandwidth-bound at 4.
    """
    params = float(model.fixed_params[rank]) + float(model.unit_params) * int(
        units[rank]
    )
    wbytes = float(model.weight_bytes(list(units))[rank])
    intensity = step_intensity(params=params, weight_bytes=wbytes, tokens=tokens)
    balance = machine_balance(
        float(rates.gemm_tflops[rank]), float(rates.membw_gbs[rank])
    )
    return LaneLoad(
        key=key,
        label=label_from_intensity(intensity, balance),
        intensity=intensity,
        balance=balance,
        basis=f"analytic: 2*params*{int(tokens)} tokens / weight bytes, rank {rank}",
        **kw,
    )


# ---------------------------------------------------------------------------
# 2. The score
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CardPlacement:
    """What one card ends up carrying, and what that is worth."""

    gpu: int
    lanes: Tuple[str, ...]
    labels: Tuple[str, ...]
    equivalents: Optional[float]
    provenance: str
    note: str = ""

    def to_json(self) -> Dict[str, object]:
        return {
            "gpu": self.gpu,
            "lanes": list(self.lanes),
            "labels": list(self.labels),
            "equivalents": (
                None if self.equivalents is None else round(self.equivalents, 4)
            ),
            "provenance": self.provenance,
            "note": self.note,
        }


def _card_value(labels: Sequence[str]) -> Tuple[float, Optional[float], str, str]:
    """(ranking value, reportable E, provenance, note) for one card's load."""
    n = len(labels)
    if n == 0:
        return 0.0, 0.0, "trivial", "idle"
    if n == 1:
        return 1.0, 1.0, "trivial", "one lane on this card"
    sm = sum(1 for x in labels if x == SM_BOUND)
    unknown = any(x == UNKNOWN for x in labels)
    if n == 2:
        if unknown:
            return (
                MEASURED_PAIR_E[1],
                None,
                "absent",
                "a lane's load shape could not be labelled analytically",
            )
        if sm >= 2:
            return (
                SM_PAIR_UPPER_BOUND,
                None,
                "bounded",
                "two SM-saturating lanes on one card was never measured; "
                "bounded above by the one-saturating point (1.130)",
            )
        return (
            MEASURED_PAIR_E[sm],
            MEASURED_PAIR_E[sm],
            "measured",
            f"slice C, {sm} of 2 lanes SM-saturating",
        )
    # Three or more lanes on one card: the ranking still only needs the
    # ordering, so the card takes the value of its WORST pair -- but no
    # aggregate is reported for a regime nobody has measured.
    worst = min(_card_value([a, b])[0] for a, b in itertools.combinations(labels, 2))
    return (
        worst,
        None,
        "absent",
        f"{n} lanes on one card is outside the measured regime (max 2)",
    )


# ---------------------------------------------------------------------------
# 3. The chooser
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SpreadAnswer:
    ok: bool
    assignment: Dict[str, Tuple[int, ...]]
    per_card: Tuple[CardPlacement, ...]
    score: float
    sm_colocations: int
    expected_e: Optional[float]
    considered: int
    truncated: bool
    reasons: Tuple[str, ...]
    runners_up: Tuple[Dict[str, object], ...] = ()

    def to_json(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "assignment": {k: list(v) for k, v in sorted(self.assignment.items())},
            "per_card": [c.to_json() for c in self.per_card],
            "score": round(self.score, 4),
            "sm_colocations": self.sm_colocations,
            "expected_e": (
                None if self.expected_e is None else round(self.expected_e, 4)
            ),
            "considered": self.considered,
            "truncated": self.truncated,
            "reasons": list(self.reasons),
            "runners_up": list(self.runners_up),
        }


def _candidates(load: LaneLoad, cards: Sequence[int]) -> List[Tuple[int, ...]]:
    if load.fixed:
        return [tuple(sorted(load.cards))]
    pool = list(load.allowed_cards) if load.allowed_cards else list(cards)
    n = max(1, int(load.card_count))
    if n > len(pool):
        return []
    return [tuple(sorted(c)) for c in itertools.combinations(pool, n)]


def _evaluate(
    assignment: Mapping[str, Tuple[int, ...]],
    labels: Mapping[str, str],
    cards: Sequence[int],
) -> Tuple[float, int, Optional[float], List[CardPlacement]]:
    per_card: List[CardPlacement] = []
    score = 0.0
    sm_colocations = 0
    e_total: Optional[float] = 0.0
    for gpu in cards:
        here = tuple(sorted(k for k, gs in assignment.items() if gpu in gs))
        if not here:
            continue
        here_labels = tuple(labels[k] for k in here)
        value, reportable, provenance, note = _card_value(here_labels)
        score += value
        if len(here) > 1:
            sm_colocations += sum(1 for x in here_labels if x == SM_BOUND)
        if reportable is None or e_total is None:
            e_total = None
        else:
            e_total += reportable
        per_card.append(
            CardPlacement(gpu, here, here_labels, reportable, provenance, note)
        )
    return score, sm_colocations, e_total, per_card


def spread_plan(
    loads: Sequence[LaneLoad],
    cards: Sequence[int],
    *,
    feasible: Optional[Callable[[Mapping[str, Tuple[int, ...]]], bool]] = None,
    max_lanes_per_card: Optional[int] = None,
    pair_bandwidth_gbs: Optional[Mapping[Tuple[int, int], float]] = None,
    keep_runners_up: int = 3,
) -> SpreadAnswer:
    """Choose which cards each lane runs on.

    ``feasible`` is the seam to the sizing side: the decider prunes with it but
    never re-implements it.  The real bracket is
    ``key_solver.coresident_budget_plan`` / ``coexistence`` -- pass a closure
    over those rather than a VRAM heuristic, so there is one fit rule in the
    tree and not two that can disagree.

    ``pair_bandwidth_gbs`` is the ordered pair matrix from the card probe
    (#213).  It never overrides the load-shape objective; it breaks TIES
    between placements that are equal under it, by preferring the multi-card
    lane whose cards talk fastest.  That is the same input #279 routes on --
    two decision levels, one set of numbers.
    """
    reasons: List[str] = []
    cards = list(dict.fromkeys(int(c) for c in cards))
    labels = {load.key: load.label for load in loads}
    if any(load.label == UNKNOWN for load in loads):
        reasons.append(
            "at least one lane could not be labelled analytically; its pairings "
            "are ranked pessimistically and report no expected E"
        )

    per_lane = [(load.key, _candidates(load, cards)) for load in loads]
    empty = [k for k, c in per_lane if not c]
    if empty:
        return SpreadAnswer(
            ok=False,
            assignment={},
            per_card=(),
            score=0.0,
            sm_colocations=0,
            expected_e=None,
            considered=0,
            truncated=False,
            reasons=tuple(
                reasons
                + [f"no placement exists for lane(s) {sorted(empty)} on cards {cards}"]
            ),
        )

    total = 1
    for _k, c in per_lane:
        total *= len(c)
    truncated = total > MAX_PLACEMENTS
    if truncated:
        reasons.append(
            f"{total} placements exceed the cap of {MAX_PLACEMENTS}; the search "
            "stopped early and the answer is the best of what it saw"
        )

    best: Optional[Tuple[Tuple[float, int, float], Dict[str, Tuple[int, ...]]]] = None
    ranked: List[Tuple[Tuple[float, int, float], Dict[str, Tuple[int, ...]]]] = []
    considered = 0
    for combo in itertools.product(*[c for _k, c in per_lane]):
        if considered >= MAX_PLACEMENTS:
            break
        assignment = {k: combo[i] for i, (k, _c) in enumerate(per_lane)}
        if max_lanes_per_card is not None:
            counts: Dict[int, int] = {}
            for gs in assignment.values():
                for g in gs:
                    counts[g] = counts.get(g, 0) + 1
            if any(v > max_lanes_per_card for v in counts.values()):
                continue
        if feasible is not None and not feasible(assignment):
            continue
        considered += 1
        score, sm_col, _e, _rows = _evaluate(assignment, labels, cards)
        tie = _link_quality(assignment, pair_bandwidth_gbs)
        rank = (score, -sm_col, tie)
        ranked.append((rank, assignment))
        if best is None or rank > best[0]:
            best = (rank, assignment)

    if best is None:
        return SpreadAnswer(
            ok=False,
            assignment={},
            per_card=(),
            score=0.0,
            sm_colocations=0,
            expected_e=None,
            considered=considered,
            truncated=truncated,
            reasons=tuple(reasons + ["every placement was rejected as infeasible"]),
        )

    rank, assignment = best
    score, sm_col, e_total, per_card = _evaluate(assignment, labels, cards)
    if sm_col == 0:
        reasons.append(
            "objective met: no SM-saturating lane shares a card with another lane"
        )
    else:
        reasons.append(
            f"objective not fully met: {sm_col} SM-saturating lane(s) share a "
            "card; no placement on this rig avoids it"
        )
    ranked.sort(key=lambda r: r[0], reverse=True)
    runners = tuple(
        {
            "assignment": {k: list(v) for k, v in sorted(a.items())},
            "score": round(r[0], 4),
            "sm_colocations": -r[1],
        }
        for r, a in ranked[1 : 1 + max(0, keep_runners_up)]
    )
    return SpreadAnswer(
        ok=True,
        assignment=assignment,
        per_card=tuple(per_card),
        score=score,
        sm_colocations=sm_col,
        expected_e=e_total,
        considered=considered,
        truncated=truncated,
        reasons=tuple(reasons),
        runners_up=runners,
    )


def _link_quality(
    assignment: Mapping[str, Tuple[int, ...]],
    pair_bandwidth_gbs: Optional[Mapping[Tuple[int, int], float]],
) -> float:
    """Tie-break only: the narrowest link any multi-card lane has to live with.

    Deliberately the MINIMUM and not the mean -- a lane's collectives run at
    the speed of its worst ordered pair, which is the same reduction
    ``rates_from_probe`` already applies.
    """
    if not pair_bandwidth_gbs:
        return 0.0
    worst = None
    for gs in assignment.values():
        for a, b in itertools.permutations(gs, 2):
            bw = pair_bandwidth_gbs.get((a, b))
            if bw is None:
                continue
            worst = bw if worst is None else min(worst, bw)
    return 0.0 if worst is None else float(worst)


def pair_bandwidth_from_probe(
    probe: Mapping[str, object], rank_gpu_id: Sequence[int]
) -> Dict[Tuple[int, int], float]:
    """Ordered pair bandwidths keyed by PHYSICAL gpu index.

    The card probe keys its pairs by UUID and its cards by ``cuda_index``;
    this is the one translation, kept here so the decider never has to know
    the probe's schema.
    """
    cards = probe.get("cards") or []
    by_uuid = {}
    for c in cards:
        idx = c.get("cuda_index")
        if idx is not None:
            by_uuid[c.get("uuid")] = int(idx)
    want = set(int(g) for g in rank_gpu_id) if rank_gpu_id else None
    out: Dict[Tuple[int, int], float] = {}
    for p in probe.get("pairs") or []:
        src = by_uuid.get(p.get("src_uuid"))
        dst = by_uuid.get(p.get("dst_uuid"))
        bw = p.get("bandwidth_gbs")
        if src is None or dst is None or bw is None:
            continue
        if want is not None and (src not in want or dst not in want):
            continue
        out[(src, dst)] = float(bw)
    return out


def describe_objective() -> str:
    """One paragraph, for the API payload and for a log line."""
    return (
        "Place lanes so that no SM-saturating load shares a card with another "
        "lane. Measured basis (DESIGN_121 §11.5): one card carrying a serving "
        "decode plus a 2048-token-prefill lane yields E = 1.130 card "
        "equivalents, the same card carrying a decode-shaped lane yields "
        "E = 1.440 -- the lane's load shape is the whole difference. Load "
        "shape is labelled analytically (step FLOP/byte against the card's "
        "measured machine balance), never profiled."
    )
