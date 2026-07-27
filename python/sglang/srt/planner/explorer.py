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
"""Combination explorer: model x hardware-combo matrix (design #97 S4, §2.7).

Because the planner-core is a pure function of ``(model, HardwareSpec)``, the
matrix is just ``plan()`` run over a set of models x a set of rigs (real OR
composed). Each cell reports fit / capacity-% / the recommended split — the
same honest capacity/feasibility numbers the CLI and UI report, never an
estimated throughput.

STRUCTURAL composed != measured (design §8): every cell carries a
``provenance`` derived from ``HardwareSpec.source``:

  * ``"live"``      — rig read from live NVML (real hardware, real totals).
  * ``"declared"``  — a manually declared real rig (user typed real cards).
  * ``"composed"``  — a rig assembled from the card library
                      (``source == "library-composition"``): NO measured
                      free-VRAM, NO measured interconnect -> an ESTIMATE. The
                      cell is stamped ``estimate=True`` with the mandatory
                      "assumes pcie/nvlink topology, not measured" note.

A composed cell can NEVER be presented as measured: the flag is computed from
the spec's source, not passed in, so the two categories cannot be conflated.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence, Tuple

from sglang.srt.planner.hardware import HardwareSpec

__all__ = [
    "MatrixCell",
    "MatrixResult",
    "provenance_of",
    "plan_matrix",
]

#: The mandatory label a composed-rig cell must carry (design §8).
COMPOSED_ESTIMATE_NOTE = (
    "estimate — composed rig: assumes PCIe/NVLink topology, not measured. "
    "Capacity/feasibility is VRAM math (interconnect-independent); no "
    "throughput is implied."
)


def provenance_of(hardware: HardwareSpec) -> Tuple[str, bool, Optional[str]]:
    """(provenance, is_estimate, note) from a HardwareSpec's source alone —
    so a composed rig cannot be mislabelled as measured."""
    if hardware.source == "library-composition":
        return "composed", True, COMPOSED_ESTIMATE_NOTE
    if hardware.source in ("nvml", "pynvml", "nvidia-smi"):
        return "live", False, None
    if hardware.source == "manual":
        return "declared", False, None
    return hardware.source, False, None


@dataclasses.dataclass(frozen=True)
class MatrixCell:
    model: str
    rig: str
    fits: bool
    #: capacity gain range vs stock even-TP (None when stock also runs
    #: infeasibly, or the fork plan itself does not fit).
    capacity_pct_range: Optional[Tuple[int, int]]
    max_context_tokens: Optional[float]
    launch_flags: List[str]
    #: stock-even-TP feasibility (the honest "what does the fork unlock" axis).
    stock_runs: Optional[bool]
    reason: Optional[str]
    #: composed != measured, structural (design §8).
    provenance: str
    estimate: bool
    estimate_note: Optional[str]


@dataclasses.dataclass(frozen=True)
class MatrixResult:
    models: List[str]
    rigs: List[str]
    cells: List[MatrixCell]

    def cell(self, model: str, rig: str) -> MatrixCell:
        for c in self.cells:
            if c.model == model and c.rig == rig:
                return c
        raise KeyError((model, rig))


def _plan_one(model_path: str, hardware: HardwareSpec, **plan_kwargs) -> MatrixCell:
    from sglang.srt.planner.feasibility import PlanRejected, plan

    provenance, estimate, note = provenance_of(hardware)
    model_label = model_path.rstrip("/").rsplit("/", 1)[-1]
    rig_label = _rig_label(hardware)
    try:
        result = plan(model_path, hardware, **plan_kwargs)
    except (PlanRejected, ValueError) as e:
        reasons = getattr(e, "reasons", [str(e)])
        return MatrixCell(
            model=model_label,
            rig=rig_label,
            fits=False,
            capacity_pct_range=None,
            max_context_tokens=None,
            launch_flags=[],
            stock_runs=None,
            reason="; ".join(reasons),
            provenance=provenance,
            estimate=estimate,
            estimate_note=note,
        )
    adv = result.advantage
    return MatrixCell(
        model=model_label,
        rig=rig_label,
        fits=result.fits,
        capacity_pct_range=(adv.capacity_pct_range if adv else None),
        max_context_tokens=(
            result.capacity.max_context_tokens if result.capacity else None
        ),
        launch_flags=result.launch_flags,
        stock_runs=(adv.stock.runs if adv else None),
        reason=(
            "; ".join(result.infeasible_reasons)
            if not result.fits
            else None
        ),
        provenance=provenance,
        estimate=estimate,
        estimate_note=note,
    )


def _rig_label(hardware: HardwareSpec) -> str:
    from collections import Counter

    items = [(g.name, g.total_mib) for g in hardware.gpus]
    counter = Counter(items)
    seen = []
    for k in items:
        if k not in seen:
            seen.append(k)
    parts = [
        f"{counter[k]}x {k[0]}" for k in seen
    ]
    return ", ".join(parts)


def plan_matrix(
    models: Sequence[Tuple[str, str]],
    rigs: Sequence[HardwareSpec],
    **plan_kwargs,
) -> MatrixResult:
    """Build the model x rig matrix. ``models`` is a list of
    ``(label, model_path)``; ``rigs`` is a list of ``HardwareSpec`` (real or
    composed). Every cell runs the SAME ``plan()``; composed rigs are stamped
    as estimates via ``provenance_of`` (design §8)."""
    model_labels = [m[0] for m in models]
    rig_labels = [_rig_label(r) for r in rigs]
    cells: List[MatrixCell] = []
    for label, path in models:
        for rig in rigs:
            cell = _plan_one(path, rig, **plan_kwargs)
            # Force the user's model label (not the path basename) for the
            # matrix axis.
            cells.append(dataclasses.replace(cell, model=label))
    return MatrixResult(models=model_labels, rigs=rig_labels, cells=cells)


def render_matrix_text(matrix: MatrixResult) -> str:
    """A compact ASCII matrix for the CLI, with a measured-vs-estimate legend
    (design §8: composed cells are visibly distinct)."""
    lines: List[str] = []
    lines.append("Config-Planner combination matrix (capacity/feasibility — never tok/s)")
    lines.append("")
    # Column headers.
    col_w = max((len(r) for r in matrix.rigs), default=8)
    col_w = max(col_w, 18)
    row_w = max((len(m) for m in matrix.models), default=8)
    header = " " * row_w + "  " + "  ".join(r.ljust(col_w) for r in matrix.rigs)
    lines.append(header)
    for model in matrix.models:
        cells = []
        for rig in matrix.rigs:
            c = matrix.cell(model, rig)
            if not c.fits:
                token = "OOM/no-fit"
            else:
                pct = ""
                if c.capacity_pct_range:
                    lo, hi = c.capacity_pct_range
                    pct = f" +{lo}..{hi}%" if lo >= 0 else f" {lo}..{hi}%"
                ctx = (
                    f"~{int(c.max_context_tokens/1000)}k"
                    if c.max_context_tokens
                    else ""
                )
                mark = "*" if c.estimate else " "
                token = f"fit{mark}{ctx}{pct}"
            cells.append(token.ljust(col_w))
        lines.append(model.ljust(row_w) + "  " + "  ".join(cells))
    lines.append("")
    lines.append("  '*' = composed rig -> ESTIMATE (assumes pcie/nvlink; not measured).")
    lines.append("  no mark = real rig (live NVML / declared).")
    lines.append("  capacity % is vs stock even-TP (ratio of two same-model estimates).")
    return "\n".join(lines)
