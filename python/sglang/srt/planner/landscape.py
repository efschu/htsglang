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
"""Crowdsourced landscape / benchmark-comparison matrix (design #97 S5, §5B.3).

Renders, per ``(model, quant_descriptor)``, the measured cross-rig spread —
"what runs this model, how fast, how efficiently, how big" — with the fork's
heterogeneous-rig wins visible against the SAME model at a MATCHED operating
point. Two modes:

  * MODE A — same-model cross-rig leaderboard (§5B.3 Mode A). Pivot =
    (model, quant); rows = rigs. Each rig ran its own best feasible config
    (shown, never hidden). Compared on efficiency (J/token at a USER-SELECTED
    batch bucket — the per-bucket curves are what make this apples-to-apples)
    and max values (peak tok/s, max context that fit), each labelled with its
    operating point.
  * MODE B — strict same-key reproducibility (§5B.3 Mode B). Key =
    (model, quant, tp/dcp config, batch, concurrency, kv_dtype); only
    identical-key entries share an axis (fork-vs-stock / did-my-rerun-repro).

HONESTY (design §5A.5/§5B.3, structural):
  * measured-preferred, measured-vs-estimate labelled — a cell is MEASURED
    only from a store entry with measured provenance; otherwise it is a
    planner/boot-log FEASIBILITY estimate, and the measured perf/energy
    columns are ABSENT (S2.5 fills them; until then they never carry a
    number).
  * efficiency is compared ONLY at a matched batch bucket; a rig with no data
    in the chosen bucket is greyed, not extrapolated.
  * every max carries its operating point; saved quantities are BANDS.
  * composed rigs stay estimates (S4 provenance) — never conflated with runs.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.planner.results_store import (
    QuantDescriptor,
    ResultEntry,
    ResultsStore,
)

__all__ = [
    "LandscapeCell",
    "Landscape",
    "build_mode_a",
    "build_mode_b",
    "render_mode_a_text",
]


@dataclasses.dataclass(frozen=True)
class MaxValue:
    """A peak with its operating point (design §5B.3: never a bare number)."""

    value: float
    at: str  # e.g. "batch 16" / "the auto uneven split"

    def render(self) -> str:
        return f"{self.value:g} @ {self.at}"


@dataclasses.dataclass(frozen=True)
class LandscapeCell:
    """One rig's row under a (model, quant) pivot."""

    rig: str
    #: "measured" | "boot-log-measured" | "planner-estimate" | "composed-estimate"
    provenance: str
    is_measured: bool
    fits: bool
    #: rig's own config (the answer, not a contaminant — §5B.3 Mode A).
    config: List[str]
    #: MEASURED efficiency at the selected bucket (J/token); None => no data in
    #: that bucket (greyed, never extrapolated) or S2.5 not yet available.
    j_per_prefill_token: Optional[float] = None
    j_per_decode_token: Optional[float] = None
    efficiency_bucket: Optional[int] = None
    #: MEASURED peaks (absent until S2.5).
    peak_prefill: Optional[MaxValue] = None
    peak_decode: Optional[MaxValue] = None
    #: FEASIBILITY max (available now: planner/boot log), labelled estimate
    #: when it came from the planner.
    max_context: Optional[MaxValue] = None
    reason: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Landscape:
    model: str
    quant: str  # canonical, or "~Nb (similar)" in similar mode
    approximate_quant: bool
    mode: str  # "A" | "B"
    bucket: Optional[int]
    rows: List[LandscapeCell]
    note: str


# ---------------------------------------------------------------------------
# Store selection helpers.
# ---------------------------------------------------------------------------


def _quant_matches(entry: ResultEntry, quant: QuantDescriptor, similar: bool) -> bool:
    if similar:
        return entry.quant.similar_key() == quant.similar_key()
    return entry.quant.exact_key() == quant.exact_key()


def _measured_entries_for(
    store: ResultsStore, model: str, quant: QuantDescriptor, similar: bool
) -> List[ResultEntry]:
    return [
        e
        for e in store.entries()
        if e.model == model and _quant_matches(e, quant, similar)
    ]


def _cell_from_measured(entry: ResultEntry, bucket: Optional[int]) -> LandscapeCell:
    jp = jd = eff_bucket = None
    if bucket is not None:
        if entry.j_per_prefill_token_by_bucket:
            jp = entry.j_per_prefill_token_by_bucket.get(bucket)
        if entry.j_per_decode_token_by_bucket:
            jd = entry.j_per_decode_token_by_bucket.get(bucket)
        if jp is not None or jd is not None:
            eff_bucket = bucket
    peak_p = (
        MaxValue(entry.peak_prefill_tok_s, f"batch {entry.batch}")
        if entry.peak_prefill_tok_s is not None
        else None
    )
    peak_d = (
        MaxValue(entry.peak_decode_tok_s, f"batch {entry.batch}")
        if entry.peak_decode_tok_s is not None
        else None
    )
    max_ctx = (
        MaxValue(entry.max_context_tokens, "measured run")
        if entry.max_context_tokens is not None
        else None
    )
    return LandscapeCell(
        rig=entry.hardware_class(),
        provenance=entry.provenance,
        is_measured=True,
        fits=True,
        config=list(entry.reproduce_flags),
        j_per_prefill_token=jp,
        j_per_decode_token=jd,
        efficiency_bucket=eff_bucket,
        peak_prefill=peak_p,
        peak_decode=peak_d,
        max_context=max_ctx,
    )


def _cell_from_planner(model_path: str, rig, quant_hint=None) -> LandscapeCell:
    """A FEASIBILITY cell computed live from the planner (measured perf columns
    stay absent). Composed rigs carry the S4 estimate provenance."""
    from sglang.srt.planner.explorer import provenance_of
    from sglang.srt.planner.feasibility import PlanRejected, plan

    prov, estimate, _ = provenance_of(rig)
    prov_label = "composed-estimate" if estimate else "planner-estimate"
    from collections import Counter

    items = [(g.name, g.total_mib) for g in rig.gpus]
    counter = Counter(items)
    seen = []
    for k in items:
        if k not in seen:
            seen.append(k)
    rig_label = ", ".join(f"{counter[k]}x {k[0]}" for k in seen)
    try:
        result = plan(model_path, rig)
    except (PlanRejected, ValueError) as e:
        reasons = getattr(e, "reasons", [str(e)])
        return LandscapeCell(
            rig=rig_label,
            provenance=prov_label,
            is_measured=False,
            fits=False,
            config=[],
            reason="; ".join(reasons),
        )
    max_ctx = (
        MaxValue(result.capacity.max_context_tokens, "the auto split (estimate)")
        if result.capacity is not None and result.fits
        else None
    )
    return LandscapeCell(
        rig=rig_label,
        provenance=prov_label,
        is_measured=False,
        fits=result.fits,
        config=list(result.launch_flags),
        max_context=max_ctx,
        reason=(
            "; ".join(result.infeasible_reasons) if not result.fits else None
        ),
    )


# ---------------------------------------------------------------------------
# Mode A — same-model cross-rig leaderboard (§5B.3 Mode A).
# ---------------------------------------------------------------------------


def build_mode_a(
    model: str,
    quant: QuantDescriptor,
    *,
    store: Optional[ResultsStore] = None,
    planner_rigs: Sequence[Tuple[str, object]] = (),
    bucket: Optional[int] = None,
    similar: bool = False,
) -> Landscape:
    """Build the (model, quant) leaderboard. Rows are rigs: measured store
    entries (preferred) PLUS planner feasibility cells for
    ``planner_rigs`` = [(model_path, HardwareSpec), ...] that have no measured
    entry. ``bucket`` selects the efficiency operating point; ``similar``
    groups by nominal bits (approximate)."""
    store = store or ResultsStore()
    rows: List[LandscapeCell] = []
    measured = _measured_entries_for(store, model, quant, similar)
    measured_rigs = set()
    for e in measured:
        rows.append(_cell_from_measured(e, bucket))
        measured_rigs.add(e.hardware_class())

    for model_path, rig in planner_rigs:
        cell = _cell_from_planner(model_path, rig)
        if cell.rig in measured_rigs:
            continue  # a measured entry already covers this rig class
        rows.append(cell)

    quant_label = (
        f"~{quant.nominal_bits:g}b (similar)" if similar else quant.canonical()
    )
    note = (
        "Mode A: same-model cross-rig. Efficiency is compared ONLY at the "
        "selected batch bucket (per-bucket curves); a rig with no data there "
        "is blank, never extrapolated. Measured cells come from measured runs; "
        "estimate cells are planner/composed feasibility (perf/energy columns "
        "stay empty until the energy module lands). Every max carries its "
        "operating point."
    )
    if similar:
        note += (
            " SIMILAR-QUANT view: grouped by nominal bits only — APPROXIMATE "
            "(scheme/group-size shift efficiency and quality)."
        )
    # measured first, then estimates; within, fitting first.
    rows.sort(key=lambda c: (not c.is_measured, not c.fits, c.rig))
    return Landscape(
        model=model,
        quant=quant_label,
        approximate_quant=similar,
        mode="A",
        bucket=bucket,
        rows=rows,
        note=note,
    )


# ---------------------------------------------------------------------------
# Mode B — strict same-key reproducibility (§5B.3 Mode B).
# ---------------------------------------------------------------------------


def build_mode_b(
    model: str,
    quant: QuantDescriptor,
    tp_config: str,
    batch: Optional[int],
    concurrency: Optional[int],
    kv_cache_dtype: str,
    *,
    store: Optional[ResultsStore] = None,
) -> Landscape:
    """Strict same-key comparison: only entries with the IDENTICAL
    (model, quant, tp/dcp config, batch, concurrency, kv_dtype) share the
    axis. Cross-key entries are excluded (never one chart)."""
    store = store or ResultsStore()
    want = (
        model,
        quant.exact_key(),
        None,  # hardware class differs per row -> not part of the shared key
        tp_config,
        batch,
        concurrency,
        kv_cache_dtype,
    )

    def key_without_hw(e: ResultEntry):
        k = list(e.mode_b_key())
        k[2] = None
        return tuple(k)

    rows = [
        _cell_from_measured(e, batch)
        for e in store.entries()
        if e.model == model and key_without_hw(e) == want
    ]
    rows.sort(key=lambda c: c.rig)
    note = (
        "Mode B: strict same-key reproducibility. Only runs with the identical "
        f"(model, quant, config={tp_config}, batch={batch}, "
        f"concurrency={concurrency}, kv_dtype={kv_cache_dtype}) appear — the "
        "fork-vs-stock / did-my-rerun-reproduce comparison. Cross-key runs are "
        "excluded, never merged into one chart."
    )
    return Landscape(
        model=model,
        quant=quant.canonical(),
        approximate_quant=False,
        mode="B",
        bucket=batch,
        rows=rows,
        note=note,
    )


# ---------------------------------------------------------------------------
# Text render (CLI).
# ---------------------------------------------------------------------------


def render_mode_a_text(ls: Landscape) -> str:
    lines: List[str] = []
    lines.append(
        f"Landscape — {ls.model} @ {ls.quant}  (Mode {ls.mode}"
        + (f", efficiency bucket = batch {ls.bucket}" if ls.bucket else "")
        + ")"
    )
    lines.append("")
    hdr = (
        f"{'rig':30s} {'prov':10s} {'fit':4s} "
        f"{'max ctx':>14s} {'J/dec-tok':>11s} {'peak dec tok/s':>15s}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for c in ls.rows:
        maxctx = (
            f"~{int(c.max_context.value):,}"
            if c.max_context is not None
            else "-"
        )
        jd = (
            f"{c.j_per_decode_token:g}"
            if c.j_per_decode_token is not None
            else ("(no data)" if c.is_measured else "-")
        )
        pd = c.peak_decode.render() if c.peak_decode is not None else "-"
        tag = "" if c.is_measured else "*"
        lines.append(
            f"{(c.rig[:29]):30s} {c.provenance[:10]:10s} "
            f"{('yes' if c.fits else 'NO'):4s} {maxctx:>14s} {jd:>11s} {pd:>15s}{tag}"
        )
    lines.append("")
    lines.append("  '*' = estimate (planner/composed feasibility), not a run.")
    lines.append(
        "  J/dec-tok + peak tok/s are MEASURED-only — empty until the energy "
        "module (S2.5) supplies them; no number is invented."
    )
    lines.append("  " + ls.note)
    return "\n".join(lines)
