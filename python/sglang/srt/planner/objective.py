# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Energy as a SELECTABLE planner objective (#350, ANALYSE_347 §6).

The planner already scores a placement by THROUGHPUT (tok/s for the LLM
classes, frame/s for the video classes). This module adds ENERGY -- tok/J,
or frame/J for the video classes -- as a sibling objective the caller can
select, so the same candidate set can be ranked for max throughput OR max
efficiency. On a heterogeneous rig the two genuinely diverge: a placement that
concentrates work on the efficient card can win tok/J while losing tok/s (the
5090-vs-3080 power/efficiency asymmetry on this rig), which is the whole point
of offering the dial.

WHY AN ORTHOGONAL --objective AXIS, NOT A NEW --rank-perf-tune ARM
-----------------------------------------------------------------
The ``--rank-perf-tune`` arms (enc / dec / maxkv / phase-prefill /
phase-decode) select WHICH throughput lever to pull and, for the phase arms,
WHICH search space to plan over -- they change the goal geometry and the
vector that gets installed. Energy is a different KIND of question: given
candidate placements, what do we VALUE when we compare them? The two compose
-- you can want the energy-optimal decode or the energy-optimal prefill -- so
folding energy into the arm list would multiply it combinatorially (enc-energy,
dec-energy, ...) and conflate "which lever" with "what we optimise". Energy is
therefore an orthogonal axis: ``--objective {throughput,energy}`` on top of
whatever ``--rank-perf-tune`` arm is active.

PROVENANCE (the #359 house rule)
--------------------------------
Every objective value is a :class:`~sglang.srt.planner.cost_model.Rate`, so it
carries its provenance: MEASURED (from the #146 energy harness / results
store), ESTIMATE (from the #148 roofline energy model), or ABSENT (no value --
never a silent substitute). A tok/J built from a measured tok/s and an
ESTIMATE J/token is an ESTIMATE, not a measurement: :func:`work_per_joule`
combines the two provenances by the weakest-wins rule so the label can never
over-claim. This module NEVER invents an energy number -- it divides a
throughput rate by a J/work rate the existing machinery produced.

Pure: no torch, no CUDA, no I/O. The energy source is injected
(``j_per_work_fn``), exactly as ``roofline_energy`` takes an injectable power
profile, so the ranking is hermetically testable and the production wiring
(measured-then-roofline-then-absent) lives at the call site.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Callable, List, Optional, Sequence, Tuple

from sglang.srt.planner.cost_model import Provenance, Rate

__all__ = [
    "Objective",
    "ScoredCandidate",
    "combine_provenance",
    "objective_value",
    "rank",
    "resolve_objective",
    "work_per_joule",
]


class Objective(str, enum.Enum):
    """What the planner maximises when it ranks candidate placements.

    Both are "higher is better": THROUGHPUT maximises work/s, ENERGY maximises
    work/J (equivalently minimises J/work). The work unit is the class's own
    -- tokens for the LLM classes, frames for the video classes -- so ENERGY
    reads as tok/J or frame/J with no separate mode.
    """

    THROUGHPUT = "throughput"
    ENERGY = "energy"


#: The strength order used to combine provenances. ABSENT is the strongest
#: contaminant (an absent input makes the result absent); between two present
#: values the ESTIMATE dominates a MEASURED one (a measurement divided by an
#: estimate is an estimate). Lower index = stronger claim.
_PROVENANCE_RANK = {
    Provenance.MEASURED: 0,
    Provenance.ESTIMATE: 1,
    Provenance.ABSENT: 2,
}


def combine_provenance(a: Provenance, b: Provenance) -> Provenance:
    """The provenance of a value derived from two inputs: the WEAKER of the
    two claims. MEASURED only survives if BOTH inputs are measured; a single
    ABSENT input makes the result ABSENT (there is nothing to divide)."""
    return a if _PROVENANCE_RANK[a] >= _PROVENANCE_RANK[b] else b


def work_per_joule(throughput: Rate, j_per_work: Rate, *, work_unit: str = "tok") -> Rate:
    """``throughput`` (work/s) divided by ``j_per_work`` (J/work) -> work/J.

    Dimensionally: ``(work/s) * (s / (J/work) ... )`` -- more directly, one
    unit of work costs ``j_per_work`` joules, so the efficiency is simply
    ``1 / j_per_work`` work per joule, and it does NOT depend on the
    throughput magnitude. Throughput is still passed (and its provenance
    folded in) because a candidate whose THROUGHPUT is only an estimate cannot
    claim a measured efficiency even if J/work was measured on a different
    run, and because callers that want a blended tok/s-and-tok/J score read
    both off one object.

    Returns a work/J :class:`Rate`. ABSENT when either input is absent (the
    named reason is carried from whichever input was absent -- j_per_work
    first, since energy is the axis being added). Provenance otherwise is the
    weaker of the two (see :func:`combine_provenance`).
    """
    unit = f"{work_unit}/J"
    if j_per_work.is_absent:
        return Rate.absent(j_per_work.source, unit=unit, label=j_per_work.label)
    if throughput.is_absent:
        return Rate.absent(throughput.source, unit=unit, label=throughput.label)
    jpw = j_per_work.require("j_per_work")
    if jpw <= 0:
        return Rate.absent(
            f"j_per_{work_unit} is {jpw} (must be > 0 to invert to {unit})",
            unit=unit,
            label=j_per_work.label,
        )
    value = 1.0 / jpw
    prov = combine_provenance(throughput.provenance, j_per_work.provenance)
    source = (
        f"work/J = 1 / (J/{work_unit}); throughput {throughput.provenance.value} "
        f"[{throughput.source}], energy {j_per_work.provenance.value} "
        f"[{j_per_work.source}]"
    )
    if prov is Provenance.MEASURED:
        return Rate.measured(value, source, unit=unit, label=j_per_work.label)
    return Rate.estimate(value, source, unit=unit, label=j_per_work.label)


def objective_value(
    objective: Objective,
    throughput: Rate,
    j_per_work: Optional[Rate] = None,
    *,
    work_unit: str = "tok",
) -> Rate:
    """The scalar a candidate is ranked by, higher-is-better, as a Rate.

    THROUGHPUT -> ``throughput`` unchanged (work/s).
    ENERGY     -> ``work_per_joule(throughput, j_per_work)`` (work/J).

    ENERGY with no ``j_per_work`` supplied is a programming error, not an
    absence: the caller must pass the measured-or-roofline rate (which may
    itself be ABSENT -- that is a data absence and propagates as one).
    """
    if objective is Objective.THROUGHPUT:
        return throughput
    if objective is Objective.ENERGY:
        if j_per_work is None:
            raise ValueError(
                "the ENERGY objective needs a j_per_work rate; pass the "
                "measured-or-roofline energy term (it may be Rate.absent, "
                "which propagates -- None is a wiring bug, not an absence)"
            )
        return work_per_joule(throughput, j_per_work, work_unit=work_unit)
    raise ValueError(f"unknown objective {objective!r}")


@dataclasses.dataclass(frozen=True)
class ScoredCandidate:
    """A candidate paired with its objective score. ``key`` is whatever the
    caller uses to identify the candidate (an index, a label, the object
    itself); this module only sorts by ``score``."""

    key: object
    score: Rate


def rank(
    candidates: Sequence[object],
    objective: Objective,
    *,
    throughput_fn: Callable[[object], Rate],
    j_per_work_fn: Optional[Callable[[object], Rate]] = None,
    work_unit: str = "tok",
) -> Tuple[List[ScoredCandidate], List[ScoredCandidate], Provenance]:
    """Rank ``candidates`` for ``objective``, best first.

    Returns ``(ranked, unscorable, floor)``:

    * ``ranked`` -- the candidates with a present score, best first. Ties keep
      input order (a stable sort), so the ranking is deterministic.
    * ``unscorable`` -- candidates whose score is ABSENT (missing energy data,
      a non-positive rate). They are NOT dropped silently: the caller decides
      whether to fall back or to refuse, and the absence reason rides in each
      ``score.source``.
    * ``floor`` -- the WEAKEST provenance among the ranked scores. A ranking
      that mixes a measured candidate with an estimated one is only as trust-
      worthy as the estimate, and the caller must be told rather than shown a
      measured-looking verdict. THROUGHPUT-only rankings return the throughput
      provenance floor; an all-absent ranking returns ABSENT.

    The energy source is injected: ``j_per_work_fn(candidate) -> Rate`` returns
    the measured (or roofline-estimate, or absent) J/work for that candidate.
    Required for ENERGY, ignored for THROUGHPUT.
    """
    if objective is Objective.ENERGY and j_per_work_fn is None:
        raise ValueError("ENERGY ranking requires j_per_work_fn")

    scored: List[ScoredCandidate] = []
    for cand in candidates:
        tput = throughput_fn(cand)
        jpw = j_per_work_fn(cand) if j_per_work_fn is not None else None
        score = objective_value(objective, tput, jpw, work_unit=work_unit)
        scored.append(ScoredCandidate(key=cand, score=score))

    present = [s for s in scored if not s.score.is_absent]
    unscorable = [s for s in scored if s.score.is_absent]
    # Stable sort, highest score first (both objectives are higher-is-better).
    ranked = sorted(present, key=lambda s: s.score.require("score"), reverse=True)

    if not present:
        floor = Provenance.ABSENT
    else:
        floor = Provenance.MEASURED
        for s in present:
            floor = combine_provenance(floor, s.score.provenance)
    return ranked, unscorable, floor


def resolve_objective(server_args) -> Objective:
    """The objective selected on the command line, defaulting to THROUGHPUT.

    Tolerant of a server_args stand-in without the field (unit tests, older
    snapshots): a missing attribute is the default, byte-identical to today.
    """
    value = getattr(server_args, "objective", None)
    if value is None:
        return Objective.THROUGHPUT
    if isinstance(value, Objective):
        return value
    try:
        return Objective(str(value))
    except ValueError:
        raise ValueError(
            f"unknown --objective {value!r}; known: "
            f"{', '.join(o.value for o in Objective)}"
        ) from None
