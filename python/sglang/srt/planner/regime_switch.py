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
"""#363 slice 1 — the regime controller's DECISION layer.

`DESIGN_363_regime_controller.md` §20 decides three things on paper. This
module builds the part of them that is a computation:

* **§20.3 planner consequence** — :func:`layout_overlap` computes the
  per-rank shard-range overlap of a layout PAIR and the dual-residency extra
  bytes; :func:`solve_layout_pair` adds the secondary objective (maximal
  overlap among near-optimal candidates, bounded by a stated tolerance).
* **§20.3 residency ladder** — :func:`residency_rung` does the ledger
  arithmetic that says whether RUNG 0 (both layouts + both graph families
  resident) is affordable next to the current KV pool, or whether the
  configuration lands on RUNG 1 (evict the non-shared slabs) or RUNG 2
  (never pre-captured -> lazy recapture).

**What this module does NOT do, deliberately.** Nothing here moves a byte,
flips a pointer, spills a diff or captures a graph. Those are #363 slices 2+
(`ROADMAP_456_matrix_execution.md` WAVE 4). The controller stays off; this is
the layer that decides and plans, and its output is a report object.
The remaining §20 pieces of this slice land in the commits that follow
this one.

**Provenance discipline.** Every phase-table cell is a
:class:`~sglang.srt.planner.cost_model.Rate` and therefore carries
``measured`` / ``estimate`` / ``absent``. A cell the autocheck needs and does
not have makes the whole verdict ``UNPRICEABLE`` — the check never fills in a
plausible number, because the entire point of §20.1 is that the decision is
made from the TABLE rather than from a canon sentence. (§20.1 flags exactly
this tension for INT8: the "one layout" canon rests on #424's `10,1,1`
prefill arm, and `NOTE_433_int8_prefill_vector.md` / `/root/addendum_435.md`
record that the arm was a re-pinned vector whose re-solve is "not confirmed".
This machine takes the table it is given and states what that table implies;
it does not know the canon.)

**Switch-cost constants are estimates, and say so.** §20.2/§20.3's numbers —
~1-2 s diff spill, ~25 ms graph-state reload, 3-6 s cold recapture — are the
physics estimate and the #102 analogy, not card measurements. The only
measured component is the KV vector move (#297's ``kv_reshard_vectors``,
"< 1 s for the delta"). They are declared once, below, each with its source.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.planner.cost_model import Provenance, Rate

__all__ = [
    "LayoutVector",
    "PhaseTable",
    "PhaseCandidate",
    "WorkloadShape",
    "SwitchCost",
    "SwitchCostModel",
    "RankOverlap",
    "OverlapReport",
    "RankResidency",
    "ResidencyReport",
    "PairSolution",
    "unit_ranges",
    "layout_overlap",
    "residency_rung",
    "solve_layout_pair",
    "DEFAULT_PAIR_TOLERANCE_PCT",
    "DEFAULT_SWITCH_COST_MODEL",
]

GIB = 1024.0**3
MIB = 1024.0**2


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


class Verdict(str, enum.Enum):
    """The four answers §20.1 authorises. There is no "maybe" tier."""

    #: One layout dominates every phase, or the divergence does not clear the
    #: switch cost. Carries the comparison as its reason.
    NO_SWITCH = "NO_SWITCH"
    #: The two phase optima share their WEIGHT vector and differ only in the
    #: coupled KV token vector (#435). Nothing but the #297 delta moves.
    SWITCH_KV_ONLY = "SWITCH_KV_ONLY"
    #: The weight layouts genuinely differ and the divergence pays for the
    #: §20.2/§20.3 decomposition at the rung this configuration lands on.
    SWITCH_FULL = "SWITCH_FULL"
    #: A cell the decision needs is ``absent``. Not a guess, not a default.
    UNPRICEABLE = "UNPRICEABLE"


# ---------------------------------------------------------------------------
# Inputs: layouts and the phase table
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LayoutVector:
    """One solved layout: the weight vector, and the KV vector coupled to it.

    ``weights`` is the ``--rank-mlp-ratio``-space vector the phase arms solve
    (``10,1,1`` and friends). ``kv_tokens`` is the matched KV token vector
    #435 seeds into the boot alongside it; ``None`` means "this table does
    not carry one", which only matters for telling SWITCH_KV_ONLY apart from
    NO_SWITCH.
    """

    name: str
    weights: Tuple[int, ...]
    kv_tokens: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError(f"layout {self.name!r} has an empty weight vector")
        if any(w <= 0 for w in self.weights):
            raise ValueError(
                f"layout {self.name!r}: weight vector {list(self.weights)} has a "
                "non-positive entry; the partitioner gives every rank >= 1 unit"
            )
        if self.kv_tokens is not None and len(self.kv_tokens) != len(self.weights):
            raise ValueError(
                f"layout {self.name!r}: kv_tokens has {len(self.kv_tokens)} "
                f"entries against {len(self.weights)} ranks"
            )

    @property
    def tp_size(self) -> int:
        return len(self.weights)

    def same_weights_as(self, other: "LayoutVector") -> bool:
        """True when the two vectors partition identically.

        Compared on the REDUCED vector, because ``2,2,2`` and ``1,1,1`` are
        the same split and the partitioner cannot tell them apart either.
        """
        return _reduce(self.weights) == _reduce(other.weights)

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weights": list(self.weights),
            "kv_tokens": (list(self.kv_tokens) if self.kv_tokens else None),
        }


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _reduce(vec: Sequence[int]) -> Tuple[int, ...]:
    g = 0
    for v in vec:
        g = _gcd(g, int(v))
    if g <= 1:
        return tuple(int(v) for v in vec)
    return tuple(int(v) // g for v in vec)


@dataclasses.dataclass(frozen=True)
class PhaseTable:
    """The measured/estimated cross table the autocheck decides from.

    One row per layout, one column per phase, each cell a throughput
    :class:`Rate` in tokens/s (higher is better) with its own provenance. The
    canon example is #424's ``comparison_table.md``: the INT8 arm has both
    layouts scored on both phases, which is what makes "the decode layout
    beats the prefill layout even ON PREFILL" a statement the table supports.

    A cell that was never run is simply absent from ``cells`` (or present as
    a ``Rate.absent``); both read the same way to :func:`autocheck`.
    """

    #: Free-text identity of the (format, model, rig) point, e.g.
    #: ``"INT8-W8A8 / Qwen3.6-27B / 5090+2x3080"``. Carried into the verdict.
    triple: str
    layouts: Tuple[LayoutVector, ...]
    #: ``(layout name, phase) -> Rate`` in tok/s.
    cells: Mapping[Tuple[str, str], Rate]
    #: The A-vs-A floor this table was taken against, in percent (#360). A
    #: divergence below it is not a divergence. Defaults to the key solver's
    #: own reference-rig floor, whose provenance travels with it.
    noise_floor_pct: Optional[float] = None
    noise_floor_source: str = ""

    def __post_init__(self) -> None:
        names = [ly.name for ly in self.layouts]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate layout names in phase table: {names}")
        sizes = {ly.tp_size for ly in self.layouts}
        if len(sizes) > 1:
            raise ValueError(
                f"phase table mixes tp sizes {sorted(sizes)}; a layout pair is "
                "solved under one fixed rank set (§20.2)"
            )

    def layout(self, name: str) -> LayoutVector:
        for ly in self.layouts:
            if ly.name == name:
                return ly
        raise KeyError(f"no layout named {name!r} in this phase table")

    def cell(self, layout_name: str, phase: str) -> Rate:
        """The cell, or a named absence — never a KeyError into the caller."""
        got = self.cells.get((layout_name, phase))
        if got is None:
            return Rate.absent(
                f"({layout_name}, {phase}) was never run in this table",
                unit="tok/s",
            )
        return got

    def floor_pct(self) -> Tuple[float, str]:
        if self.noise_floor_pct is not None:
            return float(self.noise_floor_pct), (
                self.noise_floor_source or "supplied with the table"
            )
        from sglang.srt.planner import key_solver

        return float(key_solver.NOISE_FLOOR_PCT), key_solver.NOISE_FLOOR_SOURCE

    def to_json(self) -> Dict[str, Any]:
        floor, floor_src = self.floor_pct()
        return {
            "triple": self.triple,
            "layouts": [ly.to_json() for ly in self.layouts],
            "cells": [
                {
                    "layout": name,
                    "phase": phase,
                    "tok_s": rate.value,
                    "provenance": rate.provenance.value,
                    "source": rate.source,
                }
                for (name, phase), rate in sorted(self.cells.items())
            ],
            "noise_floor_pct": floor,
            "noise_floor_source": floor_src,
        }


@dataclasses.dataclass(frozen=True)
class WorkloadShape:
    """How much of each phase one "round" of serving contains.

    The autocheck cannot compare a tok/s divergence against a seconds-valued
    switch cost without one: +24 % prefill is worth a lot on a 20 k-token
    prompt and nothing on a 200-token one. The default is the reference
    recipe's own shape (a long prompt, a short completion) and is a PARAMETER,
    not a constant — a caller with its own traffic shape passes it.
    """

    prefill_tokens: int = 20000
    decode_tokens: int = 512
    #: Switches per round, as a rate rather than a count, because #363 is a
    #: REGIME controller: it flips when the traffic regime changes, not on
    #: every request. ``2.0`` is the pessimistic reading (flip into the decode
    #: layout and back on every single round) and is the default precisely
    #: because it is the one that makes switching hardest to justify. A
    #: controller that flips once per 100 rounds passes ``0.02``.
    switches_per_round: float = 2.0

    def __post_init__(self) -> None:
        if self.prefill_tokens < 0 or self.decode_tokens < 0:
            raise ValueError("workload token counts must be >= 0")
        if self.switches_per_round < 0:
            raise ValueError("switches_per_round must be >= 0")

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Switch-cost model (§20.2 / §20.3). Every entry names its own provenance.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SwitchCostModel:
    """Seconds per switch, decomposed the way §20.3's measurement duty asks.

    None of these is a card measurement of THIS mechanism except the KV move,
    which reuses #297's actuator unchanged and inherits its measured target.
    The rest are the physics estimate (§20.2) and the #102 analogy (§20.3),
    which is why they are ``Rate`` objects rather than floats: a report built
    on them can say so.
    """

    kv_delta_move_s: Rate
    weight_diff_spill_s: Rate
    graph_state_reload_s: Rate
    lazy_recapture_s: Rate
    #: Rung 0's own cost beyond the KV delta: a pointer flip.
    pointer_flip_s: Rate

    def to_json(self) -> Dict[str, Any]:
        return {
            field.name: _rate_json(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }


def _rate_json(rate: Rate) -> Dict[str, Any]:
    return {
        "value": rate.value,
        "provenance": rate.provenance.value,
        "source": rate.source,
        "unit": rate.unit,
    }


#: The §20.2/§20.3 decomposition as this fork can currently source it.
DEFAULT_SWITCH_COST_MODEL = SwitchCostModel(
    kv_delta_move_s=Rate.measured(
        1.0,
        "DESIGN_297_kv_resharding.md: move duration measured, target < 1 s "
        "for the delta; #363 §20.2 reuses that actuator as-is",
        unit="s",
    ),
    weight_diff_spill_s=Rate.estimate(
        2.0,
        "DESIGN_363 §20.2: ~1-2 s via host staging for the 27B-INT8 class, "
        "physics estimate; upper end taken",
        unit="s",
    ),
    graph_state_reload_s=Rate.estimate(
        0.025,
        "DESIGN_363 §20.3: ~25 ms projected from #102's 1.5 -> 0.3 GB per "
        "state over the host link; PROJECTION, never timed on this mechanism",
        unit="s",
    ),
    lazy_recapture_s=Rate.estimate(
        6.0,
        "ANALYSE_363 §Sequencing: 3-6 s including quantized repack and "
        "re-capture; upper end taken for the un-pre-captured fallback",
        unit="s",
    ),
    pointer_flip_s=Rate.estimate(
        0.0,
        "DESIGN_363 §20.3 RUNG 0: a pointer flip, no copying — charged as "
        "zero rather than as an unmeasured small number",
        unit="s",
    ),
)


@dataclasses.dataclass(frozen=True)
class SwitchCost:
    """One switch, priced at a named rung, with its components kept apart."""

    rung: int
    seconds: float
    components: Dict[str, float]
    provenance: Provenance
    notes: List[str]

    def to_json(self) -> Dict[str, Any]:
        return {
            "rung": self.rung,
            "seconds": self.seconds,
            "components_s": dict(self.components),
            "provenance": self.provenance.value,
            "notes": list(self.notes),
        }


def price_switch(
    rung: int,
    *,
    kv_only: bool,
    model: SwitchCostModel = DEFAULT_SWITCH_COST_MODEL,
) -> SwitchCost:
    """Seconds for one switch at ``rung``, per §20.3's three rungs.

    RUNG 0: pointer flip + the #297 KV delta.
    RUNG 1: the diff reload + the graph-state reload + the KV delta.
    RUNG 2: lazy recapture, the only rung where the ANALYSE estimate applies
    unamortised, + the KV delta.
    """
    comps: Dict[str, float] = {"kv_delta_move": float(model.kv_delta_move_s.value or 0.0)}
    notes: List[str] = []
    provenances = [model.kv_delta_move_s.provenance]
    if kv_only:
        notes.append(
            "weight layouts are identical: only the #435-coupled KV token "
            "vector moves, through the #297 actuator"
        )
    elif rung <= 0:
        comps["pointer_flip"] = float(model.pointer_flip_s.value or 0.0)
        provenances.append(model.pointer_flip_s.provenance)
        notes.append("RUNG 0: both layouts resident, no weight bytes move")
    elif rung == 1:
        comps["weight_diff_spill"] = float(model.weight_diff_spill_s.value or 0.0)
        comps["graph_state_reload"] = float(model.graph_state_reload_s.value or 0.0)
        provenances += [
            model.weight_diff_spill_s.provenance,
            model.graph_state_reload_s.provenance,
        ]
        notes.append(
            "RUNG 1: the inactive layout's non-shared slabs were evicted, so "
            "the switch reloads the diff"
        )
    else:
        comps["lazy_recapture"] = float(model.lazy_recapture_s.value or 0.0)
        comps["weight_diff_spill"] = float(model.weight_diff_spill_s.value or 0.0)
        provenances += [
            model.lazy_recapture_s.provenance,
            model.weight_diff_spill_s.provenance,
        ]
        notes.append(
            "RUNG 2: this family was never pre-captured, so the switch pays "
            "a cold recapture"
        )
    prov = (
        Provenance.MEASURED
        if all(p is Provenance.MEASURED for p in provenances)
        else Provenance.ESTIMATE
    )
    return SwitchCost(
        rung=int(rung),
        seconds=float(sum(comps.values())),
        components=comps,
        provenance=prov,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# §20.3 — shard-range overlap and the dual-residency bytes
# ---------------------------------------------------------------------------


def unit_ranges(units: int, weights: Sequence[int]) -> List[Tuple[int, int]]:
    """Per-rank ``[start, end)`` unit range under ``weights``.

    The fork's uneven-TP shard of a dimension is CONTIGUOUS and in rank
    order: ``distributed.utils.partition_units`` gives the sizes and
    ``tp_loaded_shard_start`` takes the prefix sum of them
    (``sum(sizes[:rank])``). Overlap between two layouts is therefore an
    interval intersection, not a set operation — and only because the unit
    ORDERING is the same for both layouts, which §20.3 names as the
    precondition for the property it relies on.
    """
    from sglang.srt.distributed.utils import partition_units

    sizes = partition_units(int(units), list(weights))
    out: List[Tuple[int, int]] = []
    start = 0
    for s in sizes:
        out.append((start, start + int(s)))
        start += int(s)
    return out


@dataclasses.dataclass(frozen=True)
class RankOverlap:
    """One rank's slice of the pair, in units and in bytes."""

    rank: int
    a_range: Tuple[int, int]
    b_range: Tuple[int, int]
    intersect_units: int
    union_units: int
    #: ``|union| - |active layout|`` — what dual residency ADDS on this rank
    #: over serving the active layout alone. This is the ledger item.
    extra_units_vs_active: int
    #: ``|union| - max(|a|, |b|)`` — zero exactly when one range nests inside
    #: the other. §20.3's "the big card carries zero extra bytes" is this
    #: quantity: nothing has to be allocated that the larger layout did not
    #: already need.
    extra_units_vs_larger: int
    nested: bool
    bytes_per_unit: float

    @property
    def extra_bytes_vs_active(self) -> float:
        return self.extra_units_vs_active * self.bytes_per_unit

    @property
    def extra_bytes_vs_larger(self) -> float:
        return self.extra_units_vs_larger * self.bytes_per_unit

    def to_json(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "a_range": list(self.a_range),
            "b_range": list(self.b_range),
            "intersect_units": self.intersect_units,
            "union_units": self.union_units,
            "extra_units_vs_active": self.extra_units_vs_active,
            "extra_units_vs_larger": self.extra_units_vs_larger,
            "nested": self.nested,
            "extra_bytes_vs_active": self.extra_bytes_vs_active,
            "extra_bytes_vs_larger": self.extra_bytes_vs_larger,
        }


@dataclasses.dataclass(frozen=True)
class OverlapReport:
    """The pair's overlap, per rank and in total."""

    units: int
    bytes_per_unit: float
    active: str
    a_name: str
    b_name: str
    per_rank: Tuple[RankOverlap, ...]

    @property
    def total_union_units(self) -> int:
        return sum(r.union_units for r in self.per_rank)

    @property
    def total_intersect_units(self) -> int:
        return sum(r.intersect_units for r in self.per_rank)

    @property
    def overlap_fraction(self) -> float:
        """``sum|intersect| / sum|union|`` — 1.0 for an identical pair, 0.0
        for a fully disjoint one. The pair objective maximises this."""
        union = self.total_union_units
        return (self.total_intersect_units / union) if union else 1.0

    @property
    def extra_bytes_vs_active(self) -> float:
        return sum(r.extra_bytes_vs_active for r in self.per_rank)

    @property
    def extra_bytes_vs_larger(self) -> float:
        return sum(r.extra_bytes_vs_larger for r in self.per_rank)

    @property
    def diff_units(self) -> int:
        """Units a RUNG 1 switch has to move: exactly the non-overlapping
        remainder (§20.3's closing note — the same overlap that makes RUNG 0
        cheap makes a RUNG 1 diff smaller)."""
        return sum(r.union_units - r.intersect_units for r in self.per_rank)

    def to_json(self) -> Dict[str, Any]:
        return {
            "units": self.units,
            "bytes_per_unit": self.bytes_per_unit,
            "active": self.active,
            "a_name": self.a_name,
            "b_name": self.b_name,
            "per_rank": [r.to_json() for r in self.per_rank],
            "total_union_units": self.total_union_units,
            "total_intersect_units": self.total_intersect_units,
            "overlap_fraction": self.overlap_fraction,
            "extra_bytes_vs_active": self.extra_bytes_vs_active,
            "extra_bytes_vs_larger": self.extra_bytes_vs_larger,
            "diff_units": self.diff_units,
        }


def layout_overlap(
    a: LayoutVector,
    b: LayoutVector,
    *,
    units: int,
    bytes_per_unit: float,
    active: Optional[str] = None,
) -> OverlapReport:
    """Per-rank shard-range overlap of a layout pair, and its byte cost.

    ``units`` is the shardable dimension's indivisible unit count (the MLP
    grid for the phase pair — ``uneven_perf`` exposes it as ``mlp_units``),
    ``bytes_per_unit`` the model bytes one unit of it carries. ``active``
    names the layout that is resident when nothing has switched yet; it
    defaults to ``b``'s name (the decode layout in the canon pair, which is
    the standing default per §20.1).
    """
    if a.tp_size != b.tp_size:
        raise ValueError(
            f"layout {a.name!r} has {a.tp_size} ranks and {b.name!r} has "
            f"{b.tp_size}; a layout pair lives under one fixed rank set"
        )
    if units < a.tp_size:
        raise ValueError(
            f"{units} shardable units cannot be split over {a.tp_size} ranks "
            "(the partitioner gives every rank at least one)"
        )
    active_name = active if active is not None else b.name
    if active_name not in (a.name, b.name):
        raise ValueError(
            f"active layout {active_name!r} is neither {a.name!r} nor {b.name!r}"
        )

    ra = unit_ranges(units, a.weights)
    rb = unit_ranges(units, b.weights)
    rows: List[RankOverlap] = []
    for rank, ((a0, a1), (b0, b1)) in enumerate(zip(ra, rb)):
        inter = max(0, min(a1, b1) - max(a0, b0))
        size_a, size_b = a1 - a0, b1 - b0
        union = size_a + size_b - inter
        active_size = size_a if active_name == a.name else size_b
        nested = inter == min(size_a, size_b)
        rows.append(
            RankOverlap(
                rank=rank,
                a_range=(a0, a1),
                b_range=(b0, b1),
                intersect_units=inter,
                union_units=union,
                extra_units_vs_active=union - active_size,
                extra_units_vs_larger=union - max(size_a, size_b),
                nested=nested,
                bytes_per_unit=float(bytes_per_unit),
            )
        )
    return OverlapReport(
        units=int(units),
        bytes_per_unit=float(bytes_per_unit),
        active=active_name,
        a_name=a.name,
        b_name=b.name,
        per_rank=tuple(rows),
    )


# ---------------------------------------------------------------------------
# §20.3 — the residency ladder, as ledger arithmetic
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RankResidency:
    """One rank's rung, with the byte math that produced it."""

    rank: int
    card_total_bytes: float
    #: Everything already committed on this rank that is not the layout pair:
    #: the active layout's own weights, the KV pool, the corridor, reserves.
    committed_bytes: float
    #: Dual-residency extra for the inactive layout's non-shared slabs.
    dual_extra_bytes: float
    #: Both graph families' capture state (ESTIMATE until #286/#102 measures
    #: the layout-family case; the constant is #102's spec-ladder figure
    #: carried over by analogy, as §20.3 states).
    graph_state_bytes: float
    graph_state_provenance: Provenance
    rung: int
    reason: str
    free_at_rung0_bytes: float
    free_at_rung1_bytes: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "card_total_mib": self.card_total_bytes / MIB,
            "committed_mib": self.committed_bytes / MIB,
            "dual_extra_mib": self.dual_extra_bytes / MIB,
            "graph_state_mib": self.graph_state_bytes / MIB,
            "graph_state_provenance": self.graph_state_provenance.value,
            "rung": self.rung,
            "reason": self.reason,
            "free_at_rung0_mib": self.free_at_rung0_bytes / MIB,
            "free_at_rung1_mib": self.free_at_rung1_bytes / MIB,
        }


@dataclasses.dataclass(frozen=True)
class ResidencyReport:
    """The pair's rung on this rig: the WORST rank governs.

    §20.3's ladder is per-configuration, not per-rank — a switch that RUNG 0
    can serve on two cards and not on the third is a RUNG 1 switch, because
    the third card still has to reload its diff and everybody waits for it.
    """

    rung: int
    reason: str
    per_rank: Tuple[RankResidency, ...]
    provenance: Provenance
    pre_captured: bool

    @property
    def rung0_feasible(self) -> bool:
        return self.rung == 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "rung": self.rung,
            "rung0_feasible": self.rung0_feasible,
            "reason": self.reason,
            "provenance": self.provenance.value,
            "pre_captured": self.pre_captured,
            "per_rank": [r.to_json() for r in self.per_rank],
        }


#: #102's measured figure for one full independent capture state, carried to
#: the layout-family case BY ANALOGY (DESIGN_363 §20.3, citing
#: INTEGRATION_R3_VALIDATION.md:7601 "1,5 -> 0,3 GB je State"). Upper end
#: taken. This is the number #286/#102 is expected to replace with a real
#: per-family measurement; until then every report built on it is tagged
#: ESTIMATE and says which constant it used.
GRAPH_FAMILY_STATE_BYTES_ESTIMATE = 1.5 * GIB
GRAPH_FAMILY_STATE_SOURCE = (
    "DESIGN_363 §20.3 / #102 analogy (INTEGRATION_R3_VALIDATION.md:7601, "
    "1.5 -> 0.3 GB per state); ESTIMATE until #286 measures the layout-family "
    "case"
)


def residency_rung(
    overlap: OverlapReport,
    *,
    card_total_bytes: Sequence[float],
    committed_bytes: Sequence[float],
    graph_state_bytes: Optional[Sequence[float]] = None,
    graph_state_provenance: Provenance = Provenance.ESTIMATE,
    graph_state_source: str = GRAPH_FAMILY_STATE_SOURCE,
    pre_captured: bool = True,
    corridor_bytes: float = 0.0,
) -> ResidencyReport:
    """Which rung of §20.3's residency ladder this configuration lands on.

    The question, per rank: do BOTH weight layouts and BOTH graph families
    fit next to what is already committed (the active layout, the current KV
    pool, reserves) plus the #330 corridor?

    * fits -> **RUNG 0**: pointer flip, nothing copies.
    * only the active layout + the graph families fit -> **RUNG 1**: the
      inactive layout's non-shared slabs are evicted through the #286 class
      before any KV admission is refused, and a switch reloads the diff.
    * not even that, or the family was never pre-captured -> **RUNG 2**:
      lazy recapture, the un-amortised ANALYSE estimate.

    ``committed_bytes`` must already contain the ACTIVE layout's own weights;
    ``overlap.extra_bytes_vs_active`` is what dual residency adds on top, and
    counting the active layout twice is the easiest way to get this wrong.
    ``graph_state_bytes`` defaults to two families at
    :data:`GRAPH_FAMILY_STATE_BYTES_ESTIMATE` each; passing measured numbers
    (with ``graph_state_provenance=MEASURED``) is how #286 will upgrade it.
    """
    n = len(overlap.per_rank)
    if len(card_total_bytes) != n or len(committed_bytes) != n:
        raise ValueError(
            f"residency_rung: overlap covers {n} ranks but got "
            f"{len(card_total_bytes)} card totals and {len(committed_bytes)} "
            "committed figures"
        )
    if graph_state_bytes is None:
        graph_state_bytes = [2.0 * GRAPH_FAMILY_STATE_BYTES_ESTIMATE] * n
    elif len(graph_state_bytes) != n:
        raise ValueError(
            f"graph_state_bytes has {len(graph_state_bytes)} entries for "
            f"{n} ranks"
        )

    rows: List[RankResidency] = []
    for i, row in enumerate(overlap.per_rank):
        total = float(card_total_bytes[i])
        committed = float(committed_bytes[i]) + float(corridor_bytes)
        graphs = float(graph_state_bytes[i])
        extra = row.extra_bytes_vs_active
        free0 = total - committed - graphs - extra
        free1 = total - committed - graphs
        if not pre_captured:
            rung, why = 2, (
                "this layout family is outside the boot's declared pre-capture "
                "set, so a switch falls back to lazy recapture (§20.3 RUNG 2)"
            )
        elif free0 >= 0.0:
            rung, why = 0, (
                f"both layouts and both graph families fit: "
                f"{free0 / MIB:.0f} MiB free after the "
                f"{extra / MIB:.0f} MiB dual-residency extra"
            )
        elif free1 >= 0.0:
            rung, why = 1, (
                f"the {extra / MIB:.0f} MiB dual-residency extra does not fit "
                f"({-free0 / MIB:.0f} MiB short), but the active layout and "
                f"the graph families do ({free1 / MIB:.0f} MiB free): the "
                "inactive layout's non-shared slabs are evictable"
            )
        else:
            rung, why = 2, (
                f"not even the active layout plus both graph families fit "
                f"({-free1 / MIB:.0f} MiB short): the second family cannot be "
                "pre-captured here at all"
            )
        rows.append(
            RankResidency(
                rank=row.rank,
                card_total_bytes=total,
                committed_bytes=committed,
                dual_extra_bytes=extra,
                graph_state_bytes=graphs,
                graph_state_provenance=graph_state_provenance,
                rung=rung,
                reason=why,
                free_at_rung0_bytes=free0,
                free_at_rung1_bytes=free1,
            )
        )

    worst = max(rows, key=lambda r: r.rung)
    prov = (
        Provenance.MEASURED
        if graph_state_provenance is Provenance.MEASURED
        else Provenance.ESTIMATE
    )
    return ResidencyReport(
        rung=worst.rung,
        reason=(
            f"rank {worst.rank} governs: {worst.reason}"
            + (
                ""
                if graph_state_provenance is Provenance.MEASURED
                else f" [graph-state size is an ESTIMATE: {graph_state_source}]"
            )
        ),
        per_rank=tuple(rows),
        provenance=prov,
        pre_captured=bool(pre_captured),
    )

# ---------------------------------------------------------------------------
# §20.3 planner consequence — the pair objective
# ---------------------------------------------------------------------------

#: How much of a phase's own optimum the pair objective may trade for
#: overlap, in percent. 2.0 % is deliberately BELOW the reference rig's
#: measured A-vs-A floor (4.2 %, `key_solver.NOISE_FLOOR_PCT`): a secondary
#: objective that is allowed to give away more than the noise band could
#: silently pick a genuinely slower layout and call the difference
#: unmeasurable. Staying under the floor means the primary objective is
#: never knowingly traded — only ties are broken. Callers with a measured
#: floor of their own pass their own tolerance.
DEFAULT_PAIR_TOLERANCE_PCT = 2.0


@dataclasses.dataclass(frozen=True)
class PhaseCandidate:
    """One near-optimal layout for one phase, with its score."""

    phase: str
    layout: LayoutVector
    score: Rate  # tok/s, higher better

    def to_json(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "layout": self.layout.to_json(),
            "tok_s": self.score.value,
            "provenance": self.score.provenance.value,
            "source": self.score.source,
        }


@dataclasses.dataclass(frozen=True)
class PairSolution:
    """The chosen pair, what it cost in primary performance, and why."""

    a: LayoutVector
    b: LayoutVector
    overlap: OverlapReport
    tolerance_pct: float
    #: Per phase: the optimum's tok/s, the chosen candidate's tok/s, and the
    #: shortfall in percent (>= 0, never above ``tolerance_pct``).
    concessions: Dict[str, Dict[str, float]]
    #: The overlap fraction of the pair the pure-optimum choice would give,
    #: so the trade is visible rather than asserted.
    baseline_overlap_fraction: float
    considered_pairs: int
    reason: str

    @property
    def max_concession_pct(self) -> float:
        return max((c["shortfall_pct"] for c in self.concessions.values()), default=0.0)

    def to_json(self) -> Dict[str, Any]:
        return {
            "a": self.a.to_json(),
            "b": self.b.to_json(),
            "overlap": self.overlap.to_json(),
            "tolerance_pct": self.tolerance_pct,
            "concessions": {k: dict(v) for k, v in self.concessions.items()},
            "max_concession_pct": self.max_concession_pct,
            "baseline_overlap_fraction": self.baseline_overlap_fraction,
            "considered_pairs": self.considered_pairs,
            "reason": self.reason,
        }


def solve_layout_pair(
    candidates_a: Sequence[PhaseCandidate],
    candidates_b: Sequence[PhaseCandidate],
    *,
    units: int,
    bytes_per_unit: float,
    tolerance_pct: float = DEFAULT_PAIR_TOLERANCE_PCT,
    active: Optional[str] = None,
) -> PairSolution:
    """Pick the layout PAIR with maximal shard overlap among near-optimal ones.

    §20.3's planner consequence, stated as an objective: RUNG 0's cost and a
    RUNG 1 diff are both set by the non-overlapping remainder, so among
    candidates that are performance-equivalent for their own phase, the pair
    that overlaps most is strictly better. It is a SECONDARY objective — a
    candidate more than ``tolerance_pct`` below its phase's own optimum is
    never admitted, whatever its overlap. The default tolerance is documented
    at :data:`DEFAULT_PAIR_TOLERANCE_PCT`.

    Ties in overlap are broken by the summed relative shortfall, so an
    equally-overlapping pair that gives away less performance wins.
    """
    if not candidates_a or not candidates_b:
        raise ValueError("solve_layout_pair needs at least one candidate per phase")
    if tolerance_pct < 0.0:
        raise ValueError("tolerance_pct must be >= 0")

    def _admissible(cands: Sequence[PhaseCandidate]):
        priced = [c for c in cands if c.score.value is not None]
        if not priced:
            raise ValueError(
                f"phase {cands[0].phase!r}: every candidate's score is absent; "
                "the pair objective cannot rank unpriced layouts"
            )
        best = max(float(c.score.value) for c in priced)
        if best <= 0.0:
            raise ValueError(
                f"phase {cands[0].phase!r}: the best score is {best}; scores "
                "are tokens/s and must be positive"
            )
        keep = [
            c
            for c in priced
            if (best - float(c.score.value)) / best * 100.0 <= tolerance_pct + 1e-9
        ]
        return best, keep

    best_a, adm_a = _admissible(candidates_a)
    best_b, adm_b = _admissible(candidates_b)

    def _shortfall(c: PhaseCandidate, best: float) -> float:
        return max(0.0, (best - float(c.score.value)) / best * 100.0)

    def _overlap_of(ca: PhaseCandidate, cb: PhaseCandidate) -> OverlapReport:
        return layout_overlap(
            ca.layout,
            cb.layout,
            units=units,
            bytes_per_unit=bytes_per_unit,
            active=active,
        )

    top_a = max(adm_a, key=lambda c: float(c.score.value))
    top_b = max(adm_b, key=lambda c: float(c.score.value))
    baseline = _overlap_of(top_a, top_b)

    chosen: Optional[Tuple[PhaseCandidate, PhaseCandidate, OverlapReport]] = None
    chosen_key: Optional[Tuple[float, float]] = None
    considered = 0
    for ca in adm_a:
        for cb in adm_b:
            considered += 1
            rep = _overlap_of(ca, cb)
            key = (
                -rep.overlap_fraction,
                _shortfall(ca, best_a) + _shortfall(cb, best_b),
            )
            if chosen_key is None or key < chosen_key:
                chosen_key, chosen = key, (ca, cb, rep)
    assert chosen is not None  # adm_a/adm_b are non-empty by construction

    ca, cb, rep = chosen
    concessions = {
        ca.phase: {
            "optimum_tok_s": best_a,
            "chosen_tok_s": float(ca.score.value),
            "shortfall_pct": _shortfall(ca, best_a),
        },
        cb.phase: {
            "optimum_tok_s": best_b,
            "chosen_tok_s": float(cb.score.value),
            "shortfall_pct": _shortfall(cb, best_b),
        },
    }
    gained = rep.overlap_fraction - baseline.overlap_fraction
    if gained > 1e-12:
        reason = (
            f"overlap {baseline.overlap_fraction:.3f} -> "
            f"{rep.overlap_fraction:.3f} by trading at most "
            f"{max(c['shortfall_pct'] for c in concessions.values()):.2f} % of "
            f"a phase optimum (tolerance {tolerance_pct:.2f} %); "
            f"dual-residency extra "
            f"{baseline.extra_bytes_vs_active / MIB:.0f} -> "
            f"{rep.extra_bytes_vs_active / MIB:.0f} MiB"
        )
    else:
        reason = (
            f"no admissible pair overlaps better than the pure optima "
            f"({baseline.overlap_fraction:.3f}); the primary choice stands"
        )
    return PairSolution(
        a=ca.layout,
        b=cb.layout,
        overlap=rep,
        tolerance_pct=float(tolerance_pct),
        concessions=concessions,
        baseline_overlap_fraction=baseline.overlap_fraction,
        considered_pairs=considered,
        reason=reason,
    )
