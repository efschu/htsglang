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
"""Modeled ledger terms against measured boot posts, term by term.

This is where #605 pays out. The ledger MODELS an internal demand; the flight
recorder MEASURES what the boot actually allocated between named phases. Until
the two are put in one table nobody can say WHICH term is wrong -- only that the
total overpredicts by 4664 / 993 / 2701 MiB, which is a fact that funds no fix.

THE RULE THIS MODULE ENFORCES: every measured MiB is claimed by a term or lands
in a named residuum, and every modeled term is either matched to a measurement
or listed as UNMEASURED. Nothing is absorbed silently. A reconciliation whose
rows do not add up to its totals is a refusal, not a rounding note -- the
alternative is the failure this whole task exists to end, where a number that
nobody can trace gets carried forward because the table looked tidy.

WHY THE MAPPING IS DATA AND NOT CODE. A term does not become measurable by
being divided by a phase boundary: the boundary has to be the one that brackets
the allocation. :data:`TERM_TO_POST` states each correspondence and its BASIS,
so a reader can reject a mapping he disagrees with instead of reverse-
engineering it from arithmetic. A term with no entry here is not quietly
dropped -- it is reported as UNMEASURED, which is a statement about this
instrument's reach and a to-do, not a verdict on the term.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.mem_ledger.engine import (
    TERM_ACTIVATION,
    TERM_ATTN_WORKSPACE,
    TERM_GRAPH_CAPTURE,
    TERM_HARDWARE_RESIDUAL,
    TERM_MAMBA_POOL,
    TERM_NVML_CARVE_OUT,
    TERM_PARENT_CONTEXT,
    TERM_WEIGHTS,
)

MIB = 1 << 20

__all__ = [
    "TERM_TO_POST",
    "TermComparison",
    "CardReconciliation",
    "reconcile_card",
    "reconcile",
]


#: ``term name -> (measurement key, basis)``.
#:
#: A measurement key is either ``("delta", from_phase, to_phase)`` -- the torch
#: reserved growth across that gap -- or ``("field", mark_field, phase)``, a
#: quantity read directly off one mark.
TERM_TO_POST: Dict[str, Tuple[Tuple, str]] = {
    TERM_WEIGHTS: (
        ("delta", "pre_weight_load", "weights_loaded"),
        "the shard is allocated by load_model and nothing else runs in that gap",
    ),
    TERM_MAMBA_POOL: (
        ("delta", "weights_loaded", "kv_pool_sized"),
        "init_memory_pool allocates the KV pool AND the mamba/GDN state pool in "
        "this gap; the KV pool is the residual and is subtracted, so what is "
        "left is the state pool. Contaminated if any other pool moves here",
    ),
    TERM_ATTN_WORKSPACE: (
        ("delta", "kv_pool_sized", "capture_begin"),
        "the attention backends are constructed in this gap and their "
        "workspaces are their first allocation",
    ),
    TERM_GRAPH_CAPTURE: (
        ("delta", "capture_begin", "capture_end"),
        "the graph pool is what capture leaves behind; summed over runners, "
        "because a speculative process captures twice",
    ),
    TERM_HARDWARE_RESIDUAL: (
        ("field", "non_torch_bytes", "weights_loaded"),
        "CUDA context + allocator granularity + lazy workspaces are exactly "
        "the bytes NVML charges this pid that torch does not account for",
    ),
    TERM_ACTIVATION: (
        ("delta", "boot_complete", "first_forward"),
        "the prefill transient on top of the resident set. A LOWER BOUND: the "
        "first forward is not necessarily the deepest one the rank will see",
    ),
    TERM_NVML_CARVE_OUT: (
        ("field", "nvml_carve_out_bytes", "boot_complete"),
        "REPORTED by the driver, so the measurement should equal the model "
        "exactly; a difference here means the ledger read a different card",
    ),
    TERM_PARENT_CONTEXT: (
        ("processes", "parent_on_card", "boot_complete"),
        "whether a process other than this rank's siblings holds memory on "
        "this card, read from NVML's per-process list. A DIRECT check: the "
        "parent is either in the list or it is not",
    ),
}


@dataclasses.dataclass(frozen=True)
class TermComparison:
    """One modeled term beside its measurement."""

    term: str
    modeled_mib: int
    measured_mib: Optional[int]
    basis: str
    provenance: str = ""
    note: str = ""

    @property
    def measured(self) -> bool:
        return self.measured_mib is not None

    @property
    def error_mib(self) -> Optional[int]:
        """Model minus measurement. Positive = the term overpredicts."""
        if self.measured_mib is None:
            return None
        return self.modeled_mib - self.measured_mib

    def row(self) -> str:
        if self.measured_mib is None:
            return (
                f"  {self.term:<38} {self.modeled_mib:>7} MiB  "
                f"{'UNMEASURED':>12}  {'':>9}  {self.note or self.basis}"
            )
        return (
            f"  {self.term:<38} {self.modeled_mib:>7} MiB  "
            f"{self.measured_mib:>8} MiB  {self.error_mib:>+8} MiB  {self.basis}"
        )


@dataclasses.dataclass(frozen=True)
class CardReconciliation:
    """One card's model against one boot's measurement."""

    gpu_id: int
    card: str
    rank: int
    comparisons: Tuple[TermComparison, ...]
    #: Measured internal demand this boot, from the marks.
    measured_demand_mib: int
    #: Modeled internal demand, from the ledger.
    modeled_demand_mib: int
    #: Measured bytes no term claimed. NAMED, never folded into a term.
    residuum_mib: int
    residuum_note: str = ""
    missing_phases: Tuple[str, ...] = ()

    @property
    def overprediction_mib(self) -> int:
        return self.modeled_demand_mib - self.measured_demand_mib

    def render(self) -> str:
        lines = [
            f"VRAM reconciliation for {self.card} (rank {self.rank}): modeled "
            f"{self.modeled_demand_mib} MiB vs measured "
            f"{self.measured_demand_mib} MiB -- overpredicts by "
            f"{self.overprediction_mib} MiB",
            f"  {'term':<38} {'modeled':>11}  {'measured':>12}  {'error':>13}  basis",
        ]
        for comparison in self.comparisons:
            lines.append(comparison.row())
        lines.append(
            f"  {'RESIDUUM (measured, unclaimed)':<38} {'':>7}      "
            f"{self.residuum_mib:>8} MiB"
            + (f"  -- {self.residuum_note}" if self.residuum_note else "")
        )
        if self.missing_phases:
            lines.append(
                "  INCOMPLETE: the mark log lacks "
                + ", ".join(self.missing_phases)
                + "; the terms depending on those gaps read UNMEASURED rather "
                "than 0"
            )
        unmeasured = [c.term for c in self.comparisons if not c.measured]
        if unmeasured:
            lines.append(
                "  UNMEASURED terms are not evidence the term is wrong OR "
                "right: " + ", ".join(unmeasured)
            )
        return "\n".join(lines)


def _delta_bytes(
    marks: Sequence[Mapping[str, Any]], frm: str, to: str, field: str
) -> Optional[int]:
    """Growth of *field* between two named phases, summing repeats of *to*.

    ``capture_end`` occurs once per runner in a speculative process, and their
    costs ADD; taking the last one would silently drop the draft runner's
    graphs.
    """
    start = next((m for m in marks if m.get("phase") == frm), None)
    if start is None:
        return None
    ends = [m for m in marks if m.get("phase") == to]
    if not ends:
        return None
    return int(ends[-1].get(field, 0)) - int(start.get(field, 0))


def _field_bytes(
    marks: Sequence[Mapping[str, Any]], field: str, phase: str
) -> Optional[int]:
    mark = next((m for m in marks if m.get("phase") == phase), None)
    if mark is None or field not in mark:
        return None
    return int(mark[field])


def _parent_context_bytes(
    marks: Sequence[Mapping[str, Any]], phase: str, rank_pids: Sequence[int]
) -> Optional[int]:
    """Bytes held on this card by processes that are not a rank of this boot.

    The direct settlement of TERM_PARENT_CONTEXT. Returns 0 when the card is
    rank-only -- a MEASURED zero, which is a different statement from "we did
    not look", and the reason this returns None only when there is no list.
    """
    mark = next((m for m in marks if m.get("phase") == phase), None)
    if mark is None:
        return None
    procs = mark.get("nvml_processes")
    if not procs:
        return None
    known = {int(p) for p in rank_pids}
    return sum(int(v) for k, v in procs.items() if int(k) not in known)


def reconcile_card(
    ledger: Mapping[str, Any],
    marks: Sequence[Mapping[str, Any]],
    *,
    rank: int = 0,
    rank_pids: Sequence[int] = (),
) -> CardReconciliation:
    """One card's ledger against one rank's marks."""
    phases = {m.get("phase") for m in marks}
    missing = tuple(
        p
        for p in (
            "pre_weight_load",
            "weights_loaded",
            "kv_pool_sized",
            "capture_begin",
            "capture_end",
            "boot_complete",
            "first_forward",
        )
        if p not in phases
    )

    kv_pool_bytes = int(ledger.get("kv_pool_mib", 0)) * MIB
    comparisons: List[TermComparison] = []
    claimed_bytes = 0
    for term in ledger.get("terms", []):
        name = str(term.get("name", ""))
        modeled_mib = int(term.get("mib", 0))
        mapping = TERM_TO_POST.get(name)
        if mapping is None:
            comparisons.append(
                TermComparison(
                    term=name,
                    modeled_mib=modeled_mib,
                    measured_mib=None,
                    basis="",
                    provenance=str(term.get("provenance", "")),
                    note="no phase boundary brackets this allocation",
                )
            )
            continue
        (kind, *args), basis = mapping
        measured: Optional[int]
        if kind == "delta":
            measured = _delta_bytes(marks, args[0], args[1], "reserved_bytes")
            if measured is not None and name == TERM_MAMBA_POOL:
                # The KV pool is allocated in the same gap and is the ledger's
                # residual, not a term. Subtract it so what remains is the
                # state pool; clamp at 0 and SAY SO rather than reporting a
                # negative, which would be the sizing being wrong, not the pool.
                measured = measured - kv_pool_bytes
                if measured < 0:
                    comparisons.append(
                        TermComparison(
                            term=name,
                            modeled_mib=modeled_mib,
                            measured_mib=None,
                            basis=basis,
                            provenance=str(term.get("provenance", "")),
                            note=(
                                f"the gap grew {measured // MIB + kv_pool_bytes // MIB}"
                                f" MiB, less than the {kv_pool_bytes // MIB} MiB KV "
                                "pool the ledger placed in it; the pool did not get "
                                "what the ledger budgeted, so this gap cannot be "
                                "split"
                            ),
                        )
                    )
                    continue
        elif kind == "field":
            measured = _field_bytes(marks, args[0], args[1])
        elif kind == "processes":
            measured = _parent_context_bytes(marks, args[1], rank_pids)
        else:  # pragma: no cover - guarded by the table above
            measured = None

        if measured is not None:
            claimed_bytes += max(0, measured)
        comparisons.append(
            TermComparison(
                term=name,
                modeled_mib=modeled_mib,
                measured_mib=None if measured is None else measured // MIB,
                basis=basis,
                provenance=str(term.get("provenance", "")),
            )
        )

    # Measured internal demand: what this rank's process actually holds on the
    # card at boot_complete, plus the carve-out, which is the card's and not
    # the process's. NVML's per-pid figure is used rather than torch's reserved
    # so the CUDA context and the driver windows are inside the number.
    self_bytes = _field_bytes(marks, "nvml_self_bytes", "boot_complete") or 0
    carve_out = _field_bytes(marks, "nvml_carve_out_bytes", "boot_complete") or 0
    measured_demand = self_bytes + carve_out - kv_pool_bytes

    residuum = measured_demand - claimed_bytes
    return CardReconciliation(
        gpu_id=int(ledger.get("gpu_id", 0)),
        card=str(ledger.get("card", "")),
        rank=rank,
        comparisons=tuple(comparisons),
        measured_demand_mib=measured_demand // MIB,
        modeled_demand_mib=int(ledger.get("demand_mib", 0)),
        residuum_mib=residuum // MIB,
        residuum_note=(
            "measured bytes on this card that no term claims. Not an error "
            "bar: it is either a post nobody modeled or a mapping above that "
            "does not bracket what it says it does"
            if abs(residuum) // MIB
            else "every measured MiB is claimed by a term above"
        ),
        missing_phases=missing,
    )


def reconcile(
    ledger_payload: Mapping[str, Any],
    marks_by_rank: Mapping[int, Sequence[Mapping[str, Any]]],
) -> List[CardReconciliation]:
    """Every card of one boot. Cards are matched to ranks through the ledger's
    own ``ranks`` field, never by index."""
    out: List[CardReconciliation] = []
    pids_by_rank = {
        rank: int(marks[0].get("pid", 0))
        for rank, marks in marks_by_rank.items()
        if marks
    }
    for card in ledger_payload.get("cards", []):
        ranks = list(card.get("ranks") or ())
        for rank in ranks:
            marks = marks_by_rank.get(int(rank))
            if not marks:
                continue
            out.append(
                reconcile_card(
                    card,
                    marks,
                    rank=int(rank),
                    rank_pids=list(pids_by_rank.values()),
                )
            )
    return out
