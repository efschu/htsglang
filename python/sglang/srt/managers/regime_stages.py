# SPDX-License-Identifier: Apache-2.0
"""Boot-time stage table and the act-mode entry gate (#363 phase 3, desk half).

DESIGN_363 section 1: the runtime never invents a plan, it only selects one.
This module is where the selectable set is built -- at boot, from the planner,
validated once -- and where the evidence that authorizes SELECTING at all is
read.

THE #287 LADDER, GENERALIZED
----------------------------
The KV pressure ladder already holds N discrete operating points, solved ahead
of time, and flips between them at a consensus boundary when the pool is about
to burst. #363 keeps the machinery and generalizes the TRIGGER from "KV about
to burst" to "the regime changed". The selectable set is therefore built the
same way the ladder's operating grid is: from the vectors the operator
DECLARED at boot (``--kv-reshard-vectors``), each validated against the pool
that was actually reserved for it.

WHAT A RUNTIME-REACHABLE STAGE IS, AND WHY MOST ARE NOT
-------------------------------------------------------
DESIGN_363 section 4.4 and the #354 measurement: the MLP/GEMM weight cut has
NO runtime actuator. ``uneven_perf`` says so in the plan log itself --
"switching arms needs a RESTART". So a stage that differs from the booted one
in its weight vector is a real, planner-solved configuration that this server
cannot move to, and calling it selectable would be a lie the first flip would
discover.

Reachability is therefore a property of the PAIR (booted stage, target stage),
computed here rather than asserted:

* same weight vector, and the target KV vector is in the declared reshard
  ceiling set -> reachable via #297;
* same weight vector, KV vector unchanged, budgets differ -> reachable via
  #330, and only in the GROW direction (a shrink flushes the radix cache and
  ``vram_dial`` reserves that for an explicit dial);
* weight vector differs -> NOT reachable. Listed with the reason, never
  selectable, so ``--regime-controller act`` on a table of such stages refuses
  instead of running as an expensive observe.

THE ENTRY GATE
--------------
DESIGN_363 section 11.7 named four things phase 3 must clear before an
actuator is wired. They are card measurements; this module reads the evidence
they produce and refuses ``act`` by name until all four are present. The gate
is openable -- a gate that cannot open is as wrong as one that cannot close --
and the hermetic falsifier opens it with a written evidence file and proves
the act path fires.

Pure: no torch, no CUDA, no scheduler. The planner is reached through an
injected callable, so the table logic is testable without a probe.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.managers.regime_classifier import (
    KV_ASCEND_MARK,
    REGIME_DECODE_HEAVY,
    REGIME_KV_PRESSURE,
    REGIME_PREFILL_HEAVY,
    REGIMES,
    RegimeError,
    Stage,
    StageTable,
)

__all__ = [
    "GATE_ITEMS",
    "REACH_NO_WEIGHT_MOVER",
    "REACH_RESHARD",
    "REACH_UNDECLARED_VECTOR",
    "REACH_VRAM_DIAL",
    "REACH_BOOTED",
    "EntryGate",
    "GateItem",
    "StagePlan",
    "StageTablePlan",
    "build_stage_table",
    "load_gate_evidence",
    "reachability",
]


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

#: The stage the server booted on. Always "reachable" -- it is where we are.
REACH_BOOTED = "booted"
#: Differs only in the KV token vector, and that vector was declared. #297.
REACH_RESHARD = "reshard"
#: Differs only in the per-rank VRAM budget, upward. #330 grow.
REACH_VRAM_DIAL = "vram_dial"
#: Differs in the weight vector. No runtime actuator exists (#354/#357).
REACH_NO_WEIGHT_MOVER = "no_weight_mover"
#: Differs in the KV vector, but that vector is not in the declared ceiling
#: set, so no pool rows were reserved for it and #297 would refuse the arm.
REACH_UNDECLARED_VECTOR = "undeclared_vector"

_REACH_REASONS = {
    REACH_BOOTED: "the booted configuration",
    REACH_RESHARD: "reachable: #297 phase-boundary KV reshard to a declared vector",
    REACH_VRAM_DIAL: "reachable: #330 VRAM dial, grow direction only",
    REACH_NO_WEIGHT_MOVER: (
        "NOT reachable: the weight (MLP/GEMM) vector differs and no runtime "
        "actuator moves weights -- switching arms needs a restart (#354/#357). "
        "The stage is real and planner-solved; this server cannot move to it."
    ),
    REACH_UNDECLARED_VECTOR: (
        "NOT reachable: the KV token vector is not in --kv-reshard-vectors, so "
        "no pool rows were reserved for it and #297 would refuse the arm at "
        "flip time. Add it to the flag and reboot."
    ),
}


def reachability(
    booted: Stage,
    target: Stage,
    declared_vectors: Sequence[Sequence[int]] = (),
) -> Tuple[str, str]:
    """``(code, reason)`` for moving from ``booted`` to ``target`` at runtime.

    Order matters: the weight check comes FIRST, because a stage that differs
    in both the weight vector and the KV vector is unreachable for the reason
    that cannot be fixed by a flag.
    """
    if target.name == booted.name:
        return REACH_BOOTED, _REACH_REASONS[REACH_BOOTED]
    if tuple(target.weight_vector or ()) != tuple(booted.weight_vector or ()):
        return REACH_NO_WEIGHT_MOVER, _REACH_REASONS[REACH_NO_WEIGHT_MOVER]
    declared = {tuple(int(x) for x in v) for v in declared_vectors}
    want_kv = tuple(int(x) for x in target.kv_token_vector)
    if want_kv != tuple(int(x) for x in booted.kv_token_vector):
        if want_kv not in declared:
            return REACH_UNDECLARED_VECTOR, _REACH_REASONS[REACH_UNDECLARED_VECTOR]
        return REACH_RESHARD, _REACH_REASONS[REACH_RESHARD]
    if tuple(target.vram_budget_mib) != tuple(booted.vram_budget_mib):
        return REACH_VRAM_DIAL, _REACH_REASONS[REACH_VRAM_DIAL]
    return REACH_BOOTED, "identical to the booted configuration in every axis"


@dataclasses.dataclass(frozen=True)
class StagePlan:
    """One stage plus what this server can actually do about it."""

    stage: Stage
    reach: str
    reason: str

    @property
    def reachable(self) -> bool:
        """Whether the runtime can move to this stage.

        The BOOTED stage counts: returning to it is the normal end of a
        regime episode (the prefill burst drains, the server goes back to the
        decode configuration), its KV vector is by definition backed by the
        pool that was reserved for it, and its budgets are the ones the server
        started with. Excluding it -- which the first version of this model
        did -- makes every flip a one-way door and the flip-back read as an
        unreachable stage.
        """
        return self.reach in (REACH_BOOTED, REACH_RESHARD, REACH_VRAM_DIAL)

    @property
    def is_flip_target(self) -> bool:
        """Reachable AND somewhere other than where we booted.

        This is the count that answers "does acting have anywhere to go": a
        table whose only reachable stage is the booted one authorizes nothing.
        """
        return self.reachable and self.reach != REACH_BOOTED

    def describe(self) -> str:
        s = self.stage
        return (
            f"{s.name} [{s.regime}] weights={list(s.weight_vector or []) or 'auto'} "
            f"kv={list(s.kv_token_vector)} pool={s.max_total_num_tokens} "
            f"gain={s.measured_gain_pct:+.1f}%/band={s.measured_band_pct:.1f}% "
            f"flip={s.flip_cost_s:.2f}s -- {self.reason}"
        )


class StageTablePlan:
    """The boot-time selectable set, with the unreachable stages kept visible.

    Refuses at construction, in the #287 style, when the declared table is
    incoherent -- a table whose stages the server cannot reach is a
    configuration error at boot, not a surprise at the first flip.
    """

    def __init__(
        self,
        plans: Sequence[StagePlan],
        *,
        booted: str,
        table: StageTable,
        notes: Sequence[str] = (),
    ):
        if not plans:
            raise RegimeError("a stage table plan needs at least one stage")
        names = [p.stage.name for p in plans]
        if len(set(names)) != len(names):
            raise RegimeError(f"duplicate stage names in the plan: {names}")
        if booted not in names:
            raise RegimeError(
                f"the booted stage {booted!r} is not in the plan ({names}); the "
                f"table has to contain where the server actually is, or "
                f"reachability is computed against a fiction."
            )
        self._plans = tuple(plans)
        self._by_name = {p.stage.name: p for p in plans}
        self.booted = booted
        self.table = table
        self.notes = tuple(notes)

    def __len__(self) -> int:
        return len(self._plans)

    @property
    def plans(self) -> Tuple[StagePlan, ...]:
        return self._plans

    @property
    def reachable(self) -> Tuple[StagePlan, ...]:
        return tuple(p for p in self._plans if p.reachable)

    @property
    def flip_targets(self) -> Tuple[StagePlan, ...]:
        """Where acting could actually go. Empty = acting authorizes nothing."""
        return tuple(p for p in self._plans if p.is_flip_target)

    def plan_for(self, name: str) -> Optional[StagePlan]:
        return self._by_name.get(name)

    def is_selectable(self, name: str) -> Tuple[bool, str]:
        plan = self._by_name.get(name)
        if plan is None:
            return False, f"stage {name!r} is not in the boot table"
        return plan.reachable, plan.reason

    def describe(self) -> List[str]:
        return [p.describe() for p in self._plans]

    def summary(self) -> Dict:
        return {
            "stages": len(self._plans),
            "reachable": len(self.reachable),
            "flip_targets": len(self.flip_targets),
            "booted": self.booted,
            "unreachable": {
                p.stage.name: p.reach for p in self._plans if not p.reachable
            },
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Building the table
# ---------------------------------------------------------------------------

#: Which planner goal answers each regime. KV_PRESSURE's stage is
#: INFORMATIONAL: the pressure regime outranks stage selection (DESIGN_363
#: section 3.2), so its entry exists to be logged, not selected.
REGIME_GOALS = {
    REGIME_PREFILL_HEAVY: "enc",
    REGIME_DECODE_HEAVY: "dec",
    REGIME_KV_PRESSURE: "maxkv",
}


def build_stage_table(
    *,
    booted: Stage,
    candidates: Sequence[Stage] = (),
    declared_vectors: Sequence[Sequence[int]] = (),
    ascend_mark: float = KV_ASCEND_MARK,
) -> StageTablePlan:
    """Assemble the boot table from the booted stage plus planner candidates.

    ``candidates`` are what the planner solved for the other regimes (see
    :func:`planner_candidates` for the production feed). Everything this
    function does is validation and reachability -- it never derives a plan,
    which is DESIGN_363 section 1's whole rule.

    Raises :class:`RegimeError` on a table that cannot be coherent: a
    duplicate regime, a stage whose measured gain does not clear its own band
    (the #360 rule, enforced by :class:`StageTable`), or a booted stage that
    is not in the set.
    """
    seen_regimes: Dict[str, str] = {}
    stages: List[Stage] = [booted]
    plans: List[StagePlan] = [
        StagePlan(booted, REACH_BOOTED, _REACH_REASONS[REACH_BOOTED])
    ]
    seen_regimes[booted.regime] = booted.name
    notes: List[str] = []

    for cand in candidates:
        if cand.name == booted.name:
            raise RegimeError(
                f"candidate stage {cand.name!r} has the same name as the "
                f"booted stage; two different configurations under one name "
                f"is how a flip log becomes unreadable."
            )
        if cand.regime in seen_regimes:
            raise RegimeError(
                f"two stages claim regime {cand.regime!r}: {seen_regimes[cand.regime]!r} "
                f"and {cand.name!r}. One regime selects one stage; a tie would "
                f"be broken by list order, which is not a decision."
            )
        seen_regimes[cand.regime] = cand.name
        reach, reason = reachability(booted, cand, declared_vectors)
        stages.append(cand)
        plans.append(StagePlan(cand, reach, reason))
        if reach == REACH_NO_WEIGHT_MOVER:
            notes.append(
                f"{cand.name}: solved and NOT selectable -- weight vector "
                f"{list(cand.weight_vector or [])} differs from the booted "
                f"{list(booted.weight_vector or [])}. Boot the other arm to use it."
            )
        elif reach == REACH_UNDECLARED_VECTOR:
            notes.append(
                f"{cand.name}: solved and NOT selectable -- KV vector "
                f"{list(cand.kv_token_vector)} is not in --kv-reshard-vectors."
            )

    # StageTable applies the #360 band rule and the reference check. Building
    # it here means a table that would refuse at the first select refuses at
    # boot instead.
    table = StageTable(stages, reference=booted.name)
    # An admissibility sanity note, not a refusal: a stage whose pool cannot
    # hold even a modest working set is legal but nearly useless, and saying
    # so at boot is cheaper than discovering it from a flip log.
    for plan in plans:
        if not plan.is_flip_target:
            continue
        ceiling = int(ascend_mark * plan.stage.max_total_num_tokens)
        notes.append(
            f"{plan.stage.name}: selectable up to {ceiling} held tokens "
            f"({ascend_mark:.2f} x {plan.stage.max_total_num_tokens})"
        )
    return StageTablePlan(plans, booted=booted.name, table=table, notes=notes)


def planner_candidates(
    server_args,
    *,
    solve_fn: Optional[Callable[[str], Optional[Stage]]] = None,
) -> Tuple[List[Stage], List[str]]:
    """The production feed: ask the planner for one stage per regime.

    ``solve_fn`` is the seam. Production binds it to the key solver
    (``planner.key_solver.solve`` through ``planner.solver_api``), which is
    #348b/#350-aware by construction -- it already ranks on the shared cost
    library and honours ``--objective``. Tests inject a stub, so nothing here
    needs a card probe.

    A goal the planner cannot answer yields NO stage and a note. That is the
    honest outcome on a rig with no probe, and it is why the observer's
    "table absent" report has to survive as a real state rather than being
    replaced by an empty table pretending to be a full one.
    """
    if solve_fn is None:
        return [], [
            "no planner feed bound: the stage table holds the booted stage "
            "only. Run the card probe and boot with --rank-tp-ratio "
            "auto-performance to have the planner solve the others."
        ]
    out: List[Stage] = []
    notes: List[str] = []
    for regime in REGIMES:
        goal = REGIME_GOALS.get(regime)
        if goal is None:
            continue
        try:
            stage = solve_fn(goal)
        except Exception as exc:  # noqa: BLE001 -- a planner failure is a note
            notes.append(f"{regime}: the planner could not solve {goal!r} ({exc!r})")
            continue
        if stage is None:
            notes.append(f"{regime}: the planner returned no stage for {goal!r}")
            continue
        out.append(stage)
    return out, notes


# ---------------------------------------------------------------------------
# The act-mode entry gate (DESIGN_363 section 11.7)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GateItem:
    """One piece of evidence the act mode requires, and what produces it."""

    key: str
    what: str
    produced_by: str


GATE_ITEMS: Tuple[GateItem, ...] = (
    GateItem(
        key="desyncs_zero",
        what=(
            "the observe run recorded zero classifier desyncs across the TP "
            "group on a real workload"
        ),
        produced_by=(
            "boot with --regime-controller observe and check "
            "summary()['desyncs'] == 0 at the end of the run"
        ),
    ),
    GateItem(
        key="f2_live_replay",
        what=(
            "F2 self-conditioning replay re-run against a LIVE observe trace, "
            "not only the synthetic one: every closed-loop transition also "
            "appears in the open-loop replay"
        ),
        produced_by="replay the observe run's record stream through the classifier",
    ),
    GateItem(
        key="f3_bands_measured",
        what=(
            "F3 per-signal A-vs-A bands measured on this rig, with every "
            "provisional constant in DESIGN_363 section 3.4 replaced or "
            "confirmed against its own band"
        ),
        produced_by="two identical boots, band = max |A - A_repeat| per signal",
    ),
    GateItem(
        key="f4_card_comparison",
        what=(
            "F4 do-nothing baseline passed: the act arm beats the off arm on "
            "ms/verify AND ms/prefill by more than the band, at equal or "
            "better max_total_num_tokens"
        ),
        produced_by="three arms (off / observe / act) over one workload",
    ),
)

_GATE_KEYS = tuple(item.key for item in GATE_ITEMS)


class EntryGate:
    """Whether the measured evidence for ``--regime-controller act`` exists.

    Deliberately openable. A gate that can only ever refuse is not a gate, it
    is a disabled feature with extra words, and it hides the case where the
    ACTING path is broken -- which is why the hermetic falsifier writes a
    complete evidence file and drives a flip through it.

    An item counts as passed only when it is explicitly ``passed: true`` AND
    names a non-empty ``source``. A bare boolean would let ``{"desyncs_zero":
    true}`` -- which anybody can type -- stand in for a card run.
    """

    def __init__(self, evidence: Optional[Mapping] = None, *, origin: str = ""):
        self.origin = origin
        self._raw = dict(evidence or {})
        self._passed: Dict[str, str] = {}
        self._missing: List[str] = []
        self._malformed: List[str] = []
        for item in GATE_ITEMS:
            entry = self._raw.get(item.key)
            if entry is None:
                self._missing.append(item.key)
                continue
            if not isinstance(entry, Mapping):
                self._malformed.append(
                    f"{item.key}: expected a mapping with 'passed' and "
                    f"'source', got {type(entry).__name__}"
                )
                continue
            source = str(entry.get("source") or "").strip()
            if entry.get("passed") is not True:
                self._missing.append(item.key)
                continue
            if not source:
                self._malformed.append(
                    f"{item.key}: passed=true with no 'source'. An unattributed "
                    f"pass is a claim, not evidence."
                )
                continue
            self._passed[item.key] = source

    @property
    def open(self) -> bool:
        return not self._missing and not self._malformed

    @property
    def passed(self) -> Dict[str, str]:
        return dict(self._passed)

    @property
    def missing(self) -> List[str]:
        return list(self._missing)

    def refusal(self) -> str:
        """The message ``--regime-controller act`` is refused with.

        Names every outstanding item and how to produce it -- the #350 refusal
        style: a parse-time refusal that tells the operator what to run, not
        that something is unsupported.
        """
        lines = [
            "--regime-controller act is refused: the entry gate of "
            "DESIGN_363 section 11.7 is not clear. act moves a live server's "
            "KV placement, so it is authorized by measurement, not by a flag."
        ]
        if self.origin:
            lines.append(f"Evidence read from: {self.origin}")
        else:
            lines.append(
                "No evidence file declared. Point --regime-gate-evidence at "
                "the JSON your observe/card runs wrote."
            )
        for item in GATE_ITEMS:
            if item.key in self._passed:
                lines.append(f"  [ok]      {item.key}: {self._passed[item.key]}")
            else:
                lines.append(f"  [MISSING] {item.key} -- {item.what}")
                lines.append(f"            produce it: {item.produced_by}")
        for bad in self._malformed:
            lines.append(f"  [BAD]     {bad}")
        lines.append(
            "Until then use --regime-controller observe, which classifies and "
            "logs the flips it WOULD make and costs nothing but a log line."
        )
        return "\n".join(lines)

    def summary(self) -> Dict:
        return {
            "open": self.open,
            "origin": self.origin,
            "passed": dict(self._passed),
            "missing": list(self._missing),
            "malformed": list(self._malformed),
            "items": list(_GATE_KEYS),
        }


def load_gate_evidence(path: Optional[str]) -> EntryGate:
    """Read the evidence JSON, or return an empty (closed) gate.

    A missing file is a CLOSED gate, not an error: the common case is that the
    card runs have not happened yet, and the refusal message says exactly
    that. A file that exists but cannot be parsed IS an error -- an operator
    who declared evidence and wrote it wrong must not silently get the
    refusal path, because they would read the refusal as "not measured yet".
    """
    if not path:
        return EntryGate(None, origin="")
    expanded = os.path.expanduser(str(path))
    if not os.path.exists(expanded):
        return EntryGate(None, origin=f"{expanded} (not found)")
    try:
        with open(expanded) as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise RegimeError(
            f"--regime-gate-evidence {expanded!r} could not be read as JSON "
            f"({exc}). A declared evidence file that cannot be parsed is an "
            f"error, not a closed gate: the refusal would read as 'not "
            f"measured yet' when the truth is 'measured and unreadable'."
        ) from exc
    if not isinstance(data, dict):
        raise RegimeError(
            f"--regime-gate-evidence {expanded!r} must hold a JSON object "
            f"keyed by gate item ({', '.join(_GATE_KEYS)}), got "
            f"{type(data).__name__}."
        )
    return EntryGate(data, origin=expanded)
