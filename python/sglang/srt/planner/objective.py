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
    "boot_energy_anchors",
    "EnergyModel",
    "RankPower",
    "energy_per_work",
    "energy_rate",
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


# ---------------------------------------------------------------------------
# Solver-side energy model (#350 phase 2)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RankPower:
    """The two power anchors of ONE rank's card, and where they came from.

    Same two-anchor shape ``roofline.CardEnergy`` uses, for the same reason:
    a card's board power rides between an idle floor and an active ceiling
    with its utilization. ``source`` mirrors that module's ``power_source``
    ("measured" when the anchors are this physical card's NVML-calibrated
    draw, "estimate-tdp" for the TDP heuristic), and it decides the
    provenance of every number derived from it.
    """

    idle_w: float
    active_w: float
    source: str = "estimate-tdp"

    def __post_init__(self) -> None:
        if self.active_w < self.idle_w:
            raise ValueError(
                f"active anchor {self.active_w} W is below the idle floor "
                f"{self.idle_w} W; the anchors are swapped"
            )

    @property
    def provenance(self) -> Provenance:
        return (
            Provenance.MEASURED
            if self.source == "measured"
            else Provenance.ESTIMATE
        )

    def watts(self, util: float) -> float:
        """Board power at a busy fraction in [0, 1]."""
        u = 0.0 if util < 0.0 else (1.0 if util > 1.0 else float(util))
        return self.idle_w + (self.active_w - self.idle_w) * u


@dataclasses.dataclass(frozen=True)
class EnergyModel:
    """Per-rank power anchors, i.e. everything the SOLVER needs to price a
    candidate in joules instead of seconds.

    Deliberately NOT a new energy model: the physics is the one #148 already
    documents in ``roofline_energy`` --

        J_per_work = t_lockstep * sum_r P_r(util_r)
        util_r     = (this rank's own busy time) / t_lockstep

    -- with the busy times supplied by the solver's existing per-rank cost
    terms. Ranks step together, so the slowest sets ``t_lockstep`` and a fast
    rank that finishes early idles at its floor: that is exactly how a
    concentrating plan buys efficiency, and it falls out of the model rather
    than being asserted.

    ``per_rank`` is indexed like every other per-rank vector in the solver.
    """

    per_rank: Tuple[RankPower, ...]

    def __post_init__(self) -> None:
        if not self.per_rank:
            raise ValueError("EnergyModel needs at least one rank")

    @property
    def provenance(self) -> Provenance:
        """Weakest-wins across the ranks: one estimated card makes the whole
        rig figure an estimate."""
        prov = Provenance.MEASURED
        for rp in self.per_rank:
            prov = combine_provenance(prov, rp.provenance)
        return prov

    @property
    def source(self) -> str:
        kinds = sorted({rp.source for rp in self.per_rank})
        return "power anchors: " + ", ".join(kinds)


def energy_per_work(
    busy_seconds: Sequence[float],
    energy_model: EnergyModel,
    *,
    work_per_lockstep: float = 1.0,
) -> float:
    """Joules per unit of work for one candidate. Lower is better.

    ``busy_seconds[r]`` is rank ``r``'s OWN time for one lockstep round (the
    quantity the solver's ``a_r + b_r * u_r`` terms already produce). The
    round takes ``max_r busy_seconds`` -- every other rank waits, drawing
    idle-to-active in proportion to how much of the round it was busy.

    ``work_per_lockstep`` converts a round into work units (tokens, frames)
    when one round is not one unit of work.
    """
    if len(busy_seconds) != len(energy_model.per_rank):
        raise ValueError(
            f"busy_seconds has {len(busy_seconds)} ranks, the energy model "
            f"has {len(energy_model.per_rank)}"
        )
    t_lockstep = max(float(t) for t in busy_seconds)
    if t_lockstep <= 0:
        raise ValueError("lockstep time must be > 0 to price energy")
    watts = 0.0
    for busy, rp in zip(busy_seconds, energy_model.per_rank):
        watts += rp.watts(float(busy) / t_lockstep)
    work = float(work_per_lockstep)
    if work <= 0:
        raise ValueError("work_per_lockstep must be > 0")
    return t_lockstep * watts / work


def energy_rate(
    busy_seconds: Sequence[float],
    energy_model: EnergyModel,
    *,
    work_unit: str = "tok",
    work_per_lockstep: float = 1.0,
) -> Rate:
    """:func:`energy_per_work` as a provenance-carrying J/work Rate."""
    value = energy_per_work(
        busy_seconds, energy_model, work_per_lockstep=work_per_lockstep
    )
    unit = f"J/{work_unit}"
    source = (
        f"solver energy model (lockstep t x sum_r P_r(util_r)); "
        f"{energy_model.source}"
    )
    if energy_model.provenance is Provenance.MEASURED:
        return Rate.measured(value, source, unit=unit)
    return Rate.estimate(value, source, unit=unit)


def boot_energy_anchors(gpu_names: Sequence[str], uuids: Optional[Sequence[str]] = None):
    """Per-rank power anchors for the BOOT planner (#350 phase 4).

    Sources them exactly where the rest of the tree already keeps them, so
    nothing new is measured and nothing is invented:

    * MEASURED -- ``power_calibration.load_power_profile()`` (#149), the NVML
      board-power table keyed by card UUID. ``p_idle_w`` is the floor and
      ``p_gemm_w`` the active ceiling (the prefill objective is compute-bound;
      it is the same anchor ``roofline_energy`` uses for its FLOPS-utilization
      phase).
    * ESTIMATE -- the card library's ``tdp_w`` with #148's documented
      ``IDLE_FRACTION_OF_TDP`` floor, i.e. the identical heuristic
      ``roofline_energy`` falls back to.
    * ABSENT -- a card with neither a measurement nor a known TDP. Returns
      ``None`` for the whole rig: a partially-priced rig would silently rank
      by whichever cards happened to be known.

    Returns ``(EnergyModel, notes)`` or ``(None, notes)``. ``notes`` names the
    tier each card resolved on, for the boot log.
    """
    from sglang.srt.planner.card_library import CardLibrary
    from sglang.srt.planner.roofline import IDLE_FRACTION_OF_TDP

    try:
        from sglang.srt.planner.power_calibration import load_power_profile

        measured = load_power_profile()
    except Exception:
        measured = {}

    library = CardLibrary()
    anchors: List[RankPower] = []
    notes: List[str] = []
    for i, name in enumerate(gpu_names):
        uuid = uuids[i] if uuids and i < len(uuids) else None
        row = measured.get(uuid) if uuid else None
        if row is not None:
            anchors.append(
                RankPower(
                    idle_w=float(row.p_idle_w),
                    active_w=float(row.p_gemm_w),
                    source="measured",
                )
            )
            notes.append(f"rank {i} ({name}): measured NVML anchors")
            continue
        spec = library.get(name) if library.has(name) else None
        tdp = float(getattr(spec, "tdp_w", 0) or 0) if spec is not None else 0.0
        if tdp <= 0:
            notes.append(
                f"rank {i} ({name}): no measured power row and no TDP in the "
                f"card library -- the rig cannot be priced in joules"
            )
            return None, notes
        anchors.append(
            RankPower(
                idle_w=tdp * IDLE_FRACTION_OF_TDP,
                active_w=tdp,
                source="estimate-tdp",
            )
        )
        notes.append(f"rank {i} ({name}): estimate from TDP {tdp:.0f} W")
    if not anchors:
        notes.append("no cards to price")
        return None, notes
    return EnergyModel(per_rank=tuple(anchors)), notes
