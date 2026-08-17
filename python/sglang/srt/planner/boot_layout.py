# SPDX-License-Identifier: Apache-2.0
"""#485 rescoped: the per-family x per-phase BOOT-TIME layout chooser.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT.

The original #485 promise was "every family its own optimum PER PHASE", which
only pays if a family's placement can CHANGE at a phase boundary. It cannot,
affordably: a layout change is a whole-arena host->device refill whose cost is
CONSTANT in the distance travelled (#704a, ~1575 ms), and the family split's
measured benefit beyond uneven-TP is 0.090 ms/round (#705). Repaying ONE
per-family flip therefore needs ~17,500 rounds -- about 8.8 minutes of
continuous decode at that operating point, longer than a phase regime lasts.
The runtime per-family flip is closed (VERDICT_485_REMAINDER.md).

What survives is this: solve the layout ONCE per phase, and say where each
family lands INSIDE it. That rides for free in a flip that is already
happening, so it pays the refill zero extra times.

**NO SECOND SOLVER.** This module does not search, price, or re-derive
anything. It consumes a ``PPCutSolution`` that ``solve_pp_cut`` already
produced and re-presents it per family. If the cut changes, it is because the
solver changed -- never because of anything here. That is the condition the
rescope was accepted under.

ONE AUTHORITY FOR THE PRINTED REMEDY. ``server_args`` already computes the
paste-ready ``--pp-attn-stage-ratio`` when a snapped cut needs announcing
(#713): *"the full-attention counts of the operator's own requested ranges ARE
the --pp-attn-stage-ratio that realizes them"*. This module emits the SAME
quantity from the SAME place in the solution -- ``attention_counts`` -- so the
printed remedy and the solved layout cannot drift apart.

ABSENT IS NOT ZERO (#606). A family the checkpoint does not have is absent from
the output, never present with a zero vector. The family list is derived from
``inputs.layer_families``, which tags real layers, so a model with no linear
family simply yields no linear placement.

NO SPECULATIVE COST TERMS (#253). ``PPCutInputs`` funds exactly two families
with measured terms -- attention (``attn_layer_weight_bytes``,
``attn_layer_flops_per_token``, ``attn_core_flops_per_token_pair``) and linear
(``linear_layer_weight_bytes``, ``linear_layer_flops_per_token``). Every other
family named in the #485 law has NO input in this cost model, so this module
reports them as named gaps with the measurement that would fund them, and
invents no term for any of them.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Sequence, Tuple

from sglang.srt.planner.pp_cut import (
    LAYER_FAMILY_ATTENTION,
    LAYER_FAMILY_LINEAR,
    PPCutInputs,
    PPCutSolution,
)

#: The phase this solver actually solves. Named rather than assumed: #485's
#: matrix has a decode column too, and nothing here solves it -- see
#: VERDICT_485_phase_matrix.md O2.
PHASE_PP_PREFILL = "pp_prefill"

#: Families the cost model carries MEASURED inputs for. Anything not in here
#: gets no placement priced by this solver, however visible it is elsewhere.
PRICED_FAMILIES: Tuple[str, ...] = (LAYER_FAMILY_ATTENTION, LAYER_FAMILY_LINEAR)

#: The enumeration gap, as DATA rather than as invented cost terms (#253).
#: Each entry names what measurement would fund it. Keeping this in the module
#: the chooser lives in is deliberate: the next reader asking "why is `experts`
#: missing" finds the answer next to the code that omits it, not in a doc.
UNFUNDED_FAMILIES: Dict[str, str] = {
    "kv_heads": (
        "not a PP-layer family at all: KV heads are a TP-axis quantity "
        "(--rank-tp-ratio / the #492 replication+token-shard axis), so a "
        "per-STAGE placement is not the right shape. Would need the head-axis "
        "solver, not a new cost term here."
    ),
    "vocab": (
        "embedding and lm_head are not per-layer, so layer_families never tags "
        "them and a per-stage count is meaningless. Would need measured "
        "embed/head resident bytes plus their stage affinity (they sit at the "
        "ends of the pipeline, not on an arbitrary stage)."
    ),
    "experts": (
        "no MoE layer family exists in this checkpoint's layer_types, so there "
        "is nothing to place. Would need a checkpoint with expert layers AND "
        "measured per-expert resident bytes."
    ),
    "nonlinear_kernels": (
        "no separate weight or FLOPs term: norm/activation cost is folded into "
        "the per-family layer FLOPs already. Would need a decomposed per-layer "
        "FLOPs measurement that separates them out."
    ),
    "linear_per_quant_lane": (
        "linear_layer_weight_bytes is ONE scalar for the whole linear family, "
        "so lanes are indistinguishable here. Would need per-lane resident "
        "bytes measured from the checkpoint (the same discipline the attention "
        "term already uses: measured, not formula-derived)."
    ),
}


class BootLayoutError(ValueError):
    """A layout that must not be emitted as flags."""


@dataclasses.dataclass(frozen=True)
class FamilyPlacement:
    """Where one family's layers land across the stages."""

    family: str
    per_stage: Tuple[int, ...]
    #: True when the cost model priced this family's placement; False would
    #: mean the placement is observed but not costed. Only priced families are
    #: emitted today, so this is a forward-compat marker, not decoration.
    priced: bool

    @property
    def total(self) -> int:
        return sum(self.per_stage)


@dataclasses.dataclass(frozen=True)
class BootLayout:
    """One phase's layout, with the family placement inside it."""

    phase: str
    counts: Tuple[int, ...]
    families: Tuple[FamilyPlacement, ...]
    flags: Tuple[str, ...]
    unfunded: Tuple[str, ...]

    def family(self, name: str) -> FamilyPlacement:
        for f in self.families:
            if f.family == name:
                return f
        raise BootLayoutError(
            f"no placement for family {name!r}: this checkpoint's layer_families "
            f"carries {[f.family for f in self.families]}. A family the model "
            "does not have is ABSENT, not zero (#606) -- check the model before "
            "reading this as a solver gap."
        )

    def flag_line(self) -> str:
        return " ".join(self.flags)


def family_placement(
    layer_families: Sequence[str], counts: Sequence[int]
) -> Tuple[FamilyPlacement, ...]:
    """Per-stage counts for every family the checkpoint ACTUALLY has.

    Derived from the tag vector, so absence is structural: a family with no
    tagged layer produces no entry. Order follows first appearance in the
    model, which keeps the output stable for a given checkpoint.
    """
    seen: List[str] = []
    for fam in layer_families:
        if fam not in seen:
            seen.append(fam)
    out: List[FamilyPlacement] = []
    for fam in seen:
        per_stage: List[int] = []
        start = 0
        for c in counts:
            per_stage.append(sum(1 for f in layer_families[start : start + c] if f == fam))
            start += int(c)
        out.append(
            FamilyPlacement(fam, tuple(per_stage), priced=fam in PRICED_FAMILIES)
        )
    return tuple(out)


def choose_boot_layout(
    solution: PPCutSolution,
    inputs: PPCutInputs,
    *,
    phase: str = PHASE_PP_PREFILL,
    emit_attn_ratio: bool = True,
) -> BootLayout:
    """Re-present a SOLVED cut per family, and emit the boot flags for it.

    Does not solve. Refuses an infeasible solution rather than printing flags
    for a layout the solver already rejected -- flags for an infeasible cut are
    how an operator pastes a boot that cannot come up.
    """
    if not solution.feasible:
        raise BootLayoutError(
            "refusing to emit boot flags for an INFEASIBLE cut. The solver's "
            f"own refusals: {list(solution.refusals)}. Fix the inputs, do not "
            "paste this layout."
        )
    if len(inputs.layer_families) != sum(solution.counts):
        raise BootLayoutError(
            f"layer_families has {len(inputs.layer_families)} tags but the cut "
            f"places {sum(solution.counts)} layers; they must describe the same "
            "model."
        )

    families = family_placement(inputs.layer_families, solution.counts)

    # --pp-layer-ratio comes from the solution VERBATIM. The chooser adds
    # information; it never changes the default solve.
    flags = [f"--pp-layer-ratio {','.join(str(c) for c in solution.as_layer_ratio())}"]

    # --pp-attn-stage-ratio is the full-attention counts, which is exactly the
    # quantity server_args prints as the paste-ready remedy for a snapped cut
    # (#713). Same source, so the printed remedy and the solved layout are one
    # authority rather than two that can drift.
    if emit_attn_ratio and any(f.family == LAYER_FAMILY_ATTENTION for f in families):
        attn = ",".join(str(c) for c in solution.attention_counts)
        flags.append(f"--pp-attn-stage-ratio {attn}")

    return BootLayout(
        phase=phase,
        counts=tuple(solution.counts),
        families=families,
        flags=tuple(flags),
        unfunded=tuple(sorted(UNFUNDED_FAMILIES)),
    )


def describe(layout: BootLayout) -> str:
    """The boot record line. Names the unfunded families rather than implying
    the enumeration is complete."""
    lines = [f"boot layout [{layout.phase}]: layers={list(layout.counts)}"]
    for f in layout.families:
        mark = "" if f.priced else "  (observed, NOT priced)"
        lines.append(f"  {f.family:<16} {list(f.per_stage)}{mark}")
    lines.append(f"  flags: {layout.flag_line()}")
    lines.append(
        "  families with no cost term in this solver "
        f"({len(layout.unfunded)}): {', '.join(layout.unfunded)} "
        "-- see UNFUNDED_FAMILIES for what would fund each"
    )
    return "\n".join(lines)
