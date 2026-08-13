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
    TERM_GDN_SCRATCH,
    TERM_GRAPH_CAPTURE,
    TERM_HARDWARE_RESIDUAL,
    TERM_LOAD_TRANSIENT,
    TERM_MAMBA_POOL,
    TERM_NCCL_BUFFERS,
    TERM_NVML_CARVE_OUT,
    TERM_PARENT_CONTEXT,
    TERM_WEIGHTS,
)

MIB = 1 << 20

__all__ = [
    "TERM_TO_POST",
    "TermComparison",
    "CardReconciliation",
    "LedgerIncomplete",
    "ReconcileRefusal",
    "completeness_failures",
    "require_complete",
    "marks_by_rank_from_pids",
    "reconcile_card",
    "reconcile",
]


class LedgerIncomplete(RuntimeError):
    """A ledger prices a structural post at zero."""


class ReconcileRefusal(RuntimeError):
    """The marks and the ledger cannot be matched up, and guessing is worse."""


def _is_rank_key(key: Any, wanted: Mapping[int, Any]) -> bool:
    """True when *key* is already one of the ranks the ledger names.

    A PID cannot collide with a rank in practice -- ranks are 0..n on a
    single node and the kernel does not hand out pids that low to a model
    worker -- so this distinguishes the two keyings without the caller having
    to declare which one it used.
    """
    try:
        return int(key) in wanted
    except (TypeError, ValueError):
        return False


def marks_by_rank_from_pids(
    marks_by_pid: Mapping[Any, Sequence[Mapping[str, Any]]],
    ledger_payload: Mapping[str, Any],
) -> Dict[int, Sequence[Mapping[str, Any]]]:
    """Re-key ``{pid: marks}`` onto the ranks the ledger names.

    WHY THIS FUNCTION EXISTS. ``flight_recorder.read_marks`` returns marks
    keyed by PID -- correctly, because the ship config runs ``--tp-size 1
    --pp-size 3`` and all three processes file under TP rank 0, so keying on
    the rank field merged three cards into one timeline. But
    :func:`reconcile` looks its marks up by the RANK the ledger names, and
    ``attribute_flight.py`` handed it the pid dict unchanged. Every
    ``marks_by_rank.get(int(rank))`` returned None, every card was skipped,
    and the tool printed "The ledger names no card whose rank left marks" --
    a sentence about the DATA that was actually about the CODE. It exited 1
    and looked like a finding.

    HOW A PROCESS IS MATCHED TO A CARD, in order of preference:

    1. ``card_uuid`` on the marks against a ``uuid`` on the ledger card. Exact,
       and the reason :meth:`CardVramLedger.to_json` now emits the uuid.
    2. The MAXIMUM ``rank`` the process's own marks carry. The process-level
       phases (``process_start``, ``boot_complete``) are written before the
       runner knows its pipeline rank and all report 0, so the rank is
       recoverable only from the runner-tagged marks -- and taking the max
       over the group is what recovers it. Verified against two boots:
       pids 1918126/7/8 yield 0/1/2, as do 1464746/7/8.

    Both routes are cross-checked against the card's ``total_mib``: a process
    whose ``nvml_total_bytes`` disagrees with the card it was matched to is a
    mismatch, not a rounding note.

    REFUSES rather than returning a partial map. A card with no process, or
    two processes claiming one rank, is exactly the condition that produced
    the silent skip; returning "what could be matched" would reproduce it one
    level down.
    """
    wanted: Dict[int, Mapping[str, Any]] = {}
    for card in ledger_payload.get("cards", []) or ():
        for rank in card.get("ranks") or ():
            wanted[int(rank)] = card

    uuid_to_rank: Dict[str, int] = {}
    for rank, card in wanted.items():
        card_uuid = str(card.get("uuid") or "")
        if card_uuid:
            uuid_to_rank[card_uuid] = rank

    out: Dict[int, Sequence[Mapping[str, Any]]] = {}
    claims: Dict[int, Any] = {}
    for key, marks in marks_by_pid.items():
        if not marks:
            continue
        rank: Optional[int] = None
        group_uuid = str(
            next((m.get("card_uuid") for m in marks if m.get("card_uuid")), "")
        )
        if group_uuid and group_uuid in uuid_to_rank:
            # 1. The card's own identity. Exact, and unaffected by the rank
            #    field's staleness on process-level marks.
            rank = uuid_to_rank[group_uuid]
        elif _is_rank_key(key, wanted):
            # 2. The caller already keyed by rank. Honour it rather than
            #    re-deriving a second answer from marks whose rank field is 0
            #    on every process-level phase -- re-deriving would collapse
            #    every rank-keyed caller onto rank 0.
            rank = int(key)
        else:
            # 3. The MAXIMUM rank the process's own marks carry. See the
            #    docstring: only the runner-tagged marks know the real one.
            ranks = [int(m["rank"]) for m in marks if m.get("rank") is not None]
            if ranks:
                rank = max(ranks)
        if rank is None or rank not in wanted:
            continue
        card = wanted[rank]
        total_mib = int(card.get("total_mib", 0))
        seen_total = next(
            (
                int(m["nvml_total_bytes"]) // MIB
                for m in marks
                if m.get("nvml_total_bytes")
            ),
            None,
        )
        if total_mib and seen_total and seen_total != total_mib:
            raise ReconcileRefusal(
                f"process {key} reports an {seen_total} MiB card but was "
                f"matched to rank {rank} on a {total_mib} MiB card "
                f"({card.get('card', '')}). The marks and the ledger disagree "
                "about which physical card this rank ran on, and a "
                "reconciliation across that mismatch would attribute one "
                "card's posts to another"
            )
        if rank in claims:
            raise ReconcileRefusal(
                f"processes {claims[rank]} and {key} both resolve to rank "
                f"{rank}. One of them is on a card the ledger does not name, "
                "and guessing which would silently mis-attribute a whole card"
            )
        claims[rank] = key
        out[rank] = marks

    missing = sorted(set(wanted) - set(out))
    if missing:
        named = ", ".join(f"rank {r} ({wanted[r].get('card', '')})" for r in missing)
        raise ReconcileRefusal(
            "the ledger names cards whose process left no marks: "
            + named
            + ". This is the condition that used to return an empty result "
            "and print 'The ledger names no card whose rank left marks', "
            "which reads as a fact about the boot and was a fact about the "
            "caller's keying"
        )
    return out


#: Terms whose zero is never a price. A card that holds a model shard holds a
#: nonzero number of bytes of it, so a 0 in this term is always the model
#: failing to compute rather than the boot failing to allocate. Kept as data so
#: the check reads as a claim a reader can dispute.
_NEVER_LEGITIMATELY_ZERO: Tuple[str, ...] = (TERM_WEIGHTS,)


def completeness_failures(ledger_payload: Mapping[str, Any]) -> List[str]:
    """Every structural post this ledger left unpriced, named per card.

    WHAT THIS CATCHES, AND WHY IT IS WORTH A CHECK OF ITS OWN. The ledger
    dumped on ship boot 1464299 carried ``model weights (shards) = 0 MiB`` on
    all three cards while the boot loaded 13674 / 8325 / 9293 MiB of them. The
    cause is structural: the shipped configuration pins
    ``--rank-gpu-memory-mib``, which is the PIN PATH, and the pin path skips
    the planner that computes the shard vector. So the term was constructed,
    formatted, dumped and read as a priced zero -- indistinguishable in the
    JSON from a model that genuinely needs no weights.

    A REFUSAL IS NOT A FAILURE HERE. A term listed in the ledger's
    ``unbounded`` entries has said out loud that it could not be priced, which
    is the honest outcome and the one this whole module prefers. Only a
    silent zero, or a term missing altogether, is a failure.
    """
    failures: List[str] = []
    for card in ledger_payload.get("cards", []) or ():
        label = str(card.get("card", f"GPU {card.get('gpu_id', '?')}"))
        priced = {str(t.get("name", "")): t for t in card.get("terms", []) or ()}
        refused_text = " ".join(str(x) for x in (card.get("unbounded") or ()))
        for term in _NEVER_LEGITIMATELY_ZERO:
            if term in refused_text:
                continue
            entry = priced.get(term)
            if entry is None:
                failures.append(
                    f"{term} on {label}: the ledger carries no such term at "
                    "all, and no refusal naming it. A card that holds a shard "
                    "holds a nonzero number of its bytes"
                )
            elif int(entry.get("mib", 0)) <= 0:
                failures.append(
                    f"{term} on {label}: priced at {int(entry.get('mib', 0))} "
                    "MiB. This is the PIN PATH signature -- pinning "
                    "--rank-gpu-memory-mib skips the planner that computes the "
                    "shard vector, so the term is built and dumped as a zero "
                    "rather than refused. It is the ledger's dominant post and "
                    "a zero here silently removes it from every total"
                )
    return failures


def require_complete(ledger_payload: Mapping[str, Any]) -> None:
    """Raise :class:`LedgerIncomplete` if any structural post is unpriced."""
    failures = completeness_failures(ledger_payload)
    if failures:
        raise LedgerIncomplete("; ".join(failures))


#: ``term name -> (measurement key, basis)``.
#:
#: A measurement key is either ``("delta", from_phase, to_phase)`` -- the torch
#: reserved growth across that gap -- or ``("field", mark_field, phase)``, a
#: quantity read directly off one mark.
TERM_TO_POST: Dict[str, Tuple[Tuple, str]] = {
    TERM_WEIGHTS: (
        ("episodes",),
        "each (pre_weight_load -> weights_loaded) pair is ONE weight-load "
        "episode, measured on allocated bytes OUTSIDE the KV arena; the term "
        "is the LARGEST episode, never their sum, because every episode is "
        "freed before the next one loads",
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
    TERM_LOAD_TRANSIENT: (
        ("peak", "allocator_transient_bytes"),
        "the allocator's reserved PEAK above what is still reserved, taken as "
        "the MAXIMUM over every mark of the boot. Reading it at one phase is "
        "what the first reconciliation did and it measured 0 on a boot whose "
        "true peak was 13392 MiB, three marks further on: the term is a "
        "property of the whole boot, so its measurement has to be too",
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
    TERM_GDN_SCRATCH: (
        ("delta", "gdn_scratch_begin", "gdn_scratch_end"),
        "the GDN/Mamba prefill scratch is allocated between these two marks "
        "and released after; #595 left the term with no boundary at all, so "
        "it read UNMEASURED on every boot ever recorded",
    ),
    TERM_NCCL_BUFFERS: (
        ("delta", "nccl_init_begin", "nccl_init_end"),
        "NCCL allocates its communicator buffers at init and does not grow "
        "them with message size, so the gap around init is the whole term. "
        "This was #595's gap: the taxonomy carried the term and the recorder "
        "had nowhere to see it",
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
    #: ``None`` when the MODEL refused to price the term (an UNBOUNDED entry
    #: on the ledger). Distinct from 0, which is a priced zero, and distinct
    #: from the term being absent, which is how a refusal used to look.
    modeled_mib: Optional[int]
    measured_mib: Optional[int]
    basis: str
    provenance: str = ""
    note: str = ""
    #: True when the ledger REFUSED this term rather than modelling it.
    refused: bool = False
    #: True when the CONFIGURATION removes the term. A priced zero, and a
    #: different verdict from UNMEASURED in both directions.
    not_applicable: bool = False

    @property
    def measured(self) -> bool:
        return self.measured_mib is not None

    @property
    def error_mib(self) -> Optional[int]:
        """Model minus measurement. Positive = the term overpredicts."""
        if self.measured_mib is None or self.modeled_mib is None:
            return None
        return self.modeled_mib - self.measured_mib

    def row(self) -> str:
        modeled = "REFUSED" if self.modeled_mib is None else f"{self.modeled_mib} MiB"
        if self.not_applicable:
            return (
                f"  {self.term:<38} {modeled:>11}  "
                f"{'N/A':>12}  {'':>9}  {self.note}"
            )
        if self.measured_mib is None:
            return (
                f"  {self.term:<38} {modeled:>11}  "
                f"{'UNMEASURED':>12}  {'':>9}  {self.note or self.basis}"
            )
        if self.modeled_mib is None:
            return (
                f"  {self.term:<38} {modeled:>11}  "
                f"{self.measured_mib:>8} MiB  {'':>9}  {self.note or self.basis}"
            )
        return (
            f"  {self.term:<38} {modeled:>11}  "
            f"{self.measured_mib:>8} MiB  {self.error_mib:>+8} MiB  {self.basis}"
        )


@dataclasses.dataclass(frozen=True)
class CardReconciliation:
    """One card's model against one boot's measurement."""

    gpu_id: int
    card: str
    rank: int
    comparisons: Tuple[TermComparison, ...]
    #: Measured internal demand this boot, from the marks. ``None`` when the
    #: boot cannot supply a MEASURED KV pool -- the modelled budget is never
    #: substituted, see :func:`_measured_kv_pool_bytes`.
    measured_demand_mib: Optional[int]
    #: Modeled internal demand, from the ledger.
    modeled_demand_mib: int
    #: Measured bytes no term claimed. NAMED, never folded into a term.
    residuum_mib: Optional[int]
    residuum_note: str = ""
    #: Why the measured demand could not be formed, when it could not.
    demand_refusal: str = ""
    missing_phases: Tuple[str, ...] = ()
    #: Terms computed with a global delta because marks lacked draft_worker tags.
    #: Empty when all marks carry runner tags; non-empty signals that the
    #: measurement is still valid but runner attribution is unavailable.
    ambiguous_runner_terms: Tuple[str, ...] = ()

    @property
    def overprediction_mib(self) -> Optional[int]:
        if self.measured_demand_mib is None:
            return None
        return self.modeled_demand_mib - self.measured_demand_mib

    def render(self) -> str:
        if self.measured_demand_mib is None:
            headline = (
                f"VRAM reconciliation for {self.card} (rank {self.rank}): "
                f"modeled {self.modeled_demand_mib} MiB vs measured "
                f"UNAVAILABLE -- {self.demand_refusal}"
            )
        else:
            headline = (
                f"VRAM reconciliation for {self.card} (rank {self.rank}): "
                f"modeled {self.modeled_demand_mib} MiB vs measured "
                f"{self.measured_demand_mib} MiB -- overpredicts by "
                f"{self.overprediction_mib} MiB"
            )
        lines = [
            headline,
            f"  {'term':<38} {'modeled':>11}  {'measured':>12}  {'error':>13}  basis",
        ]
        for comparison in self.comparisons:
            lines.append(comparison.row())
        if self.residuum_mib is None:
            lines.append(
                f"  {'RESIDUUM (measured, unclaimed)':<38} {'':>7}      "
                f"{'UNAVAILABLE':>12}  -- no measured demand to subtract from"
            )
        else:
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
        unmeasured = [
            c.term
            for c in self.comparisons
            if not c.measured and not c.not_applicable and not c.refused
        ]
        if unmeasured:
            lines.append(
                "  UNMEASURED terms are not evidence the term is wrong OR "
                "right: " + ", ".join(unmeasured)
            )
        if self.ambiguous_runner_terms:
            lines.append(
                "  AMBIGUOUS RUNNER: the mark log lacks draft_worker tags on "
                "per-runner phases; these terms were computed with a global "
                "delta (first -> last) rather than per-runner sum -- the "
                "measurement is still valid but runner attribution is "
                "unavailable: " + ", ".join(self.ambiguous_runner_terms)
            )
        return "\n".join(lines)


def _delta_bytes(
    marks: Sequence[Mapping[str, Any]], frm: str, to: str, field: str
) -> Tuple[Optional[int], bool]:
    """Growth of *field* between two named phases, per-runner then summed.

    Under speculative decoding a rank runs TWO model runners (target + NEXTN
    draft), so ``initialize()`` runs twice and there are 14 marks per rank,
    not 8. The target runner loads its weights, then the draft runner loads
    its. A global ``first(frm) -> last(to)`` span would cross the draft's
    weight load and charge draft weights to ``TERM_MAMBA_POOL``.

    FIX: partition marks by ``extra.draft_worker``, compute a delta per runner,
    sum across runners. ``capture_end`` already added correctly by accident
    (taking the last one includes both); weights and the state pool do not.

    When ``extra.draft_worker`` is MISSING on the marks (old boots predating
    the tag), a single global delta is computed and the caller is informed via
    the returned boolean so the result is not silently mis-attributed.

    Returns ``(delta_bytes, had_ambiguous_runner)`` where the boolean is True
    when the marks lacked runner tags and a global delta was used.
    """
    # Only partition marks that carry the requested phases. Process-level
    # phases (boot_complete, first_forward) have no runner tag and would
    # pollute the "other" bucket if we partitioned every mark.
    relevant_phases = {frm, to}
    relevant_marks = [m for m in marks if m.get("phase") in relevant_phases]

    tagged_false: List[Mapping[str, Any]] = []
    tagged_true: List[Mapping[str, Any]] = []
    other: List[Mapping[str, Any]] = []
    for m in relevant_marks:
        extra = m.get("extra") or {}
        tag = extra.get("draft_worker")
        if tag is True:
            tagged_true.append(m)
        elif tag is False:
            tagged_false.append(m)
        else:
            other.append(m)

    # If no marks carry a runner tag, use the global delta.
    if not tagged_false and not tagged_true:
        # Deduplicate: within the untagged set, keep the first occurrence of
        # each phase so process-level phases that fire once per runner do not
        # double-count their delta.
        deduped = _dedup_first(other, [frm, to])
        start = next((m for m in deduped if m.get("phase") == frm), None)
        if start is None:
            return None, True
        ends = [m for m in deduped if m.get("phase") == to]
        if not ends:
            return None, True
        return int(ends[-1].get(field, 0)) - int(start.get(field, 0)), True

    # Mixed tags: cannot compute per-runner because the untagged marks would
    # be silently attributed. Return None and report the ambiguity.
    if other:
        return None, True

    # Pure per-runner: compute delta within each tagged partition, sum.
    total: int = 0
    for partition in (tagged_false, tagged_true):
        if not partition:
            continue
        # Deduplicate within the partition so process-level phases that fire
        # once per runner with identical values do not double-count.
        deduped = _dedup_first(partition, [frm, to])
        start = next((m for m in deduped if m.get("phase") == frm), None)
        if start is None:
            continue
        ends = [m for m in deduped if m.get("phase") == to]
        if not ends:
            continue
        total += int(ends[-1].get(field, 0)) - int(start.get(field, 0))

    return total if total else None, False


def _dedup_first(
    marks: Sequence[Mapping[str, Any]], phases: Sequence[str]
) -> List[Mapping[str, Any]]:
    """Keep only the FIRST occurrence of each named phase.

    Process-level phases like ``first_forward`` fire once per runner with
    identical values. Taking both would double-count the activation delta.
    The spec says to dedupe by taking the first, never sum.
    """
    seen: Dict[str, bool] = {}
    result: List[Mapping[str, Any]] = []
    for m in marks:
        phase = m.get("phase")
        if phase in phases and seen.get(phase):
            continue
        if phase in phases:
            seen[phase] = True
        result.append(m)
    return result


def _measured_kv_pool_bytes(
    marks: Sequence[Mapping[str, Any]],
) -> Optional[int]:
    """The KV pool this boot ACTUALLY got, or None.

    ONE source: the arena census, ``kv_arena_backed_bytes`` at
    ``boot_complete``. That is what the pool actually committed.

    THE OBVIOUS SECOND SOURCE IS REJECTED, and the rejection is measured
    rather than argued. The target runner's ``weights_loaded ->
    kv_pool_sized`` reserved growth looks like the pool, but that gap
    allocates the KV pool AND the mamba/GDN state pool together: on boot
    1464299 it grew 7720 MiB while the arena backed 6916, the ~800 MiB
    difference being the state pool. Using it would over-subtract by exactly
    the term the ledger books separately, which is the same
    conflate-two-posts-in-one-gap error the mamba mapping already carries a
    warning about.

    None means "this boot cannot say", and the caller REFUSES on it. The
    ledger's budget is not a fallback: it is the modelled number, and the
    whole purpose of this figure is to be independent of the model it checks.
    """
    backed = _field_bytes(marks, "kv_arena_backed_bytes", "boot_complete")
    return backed or None


def _peak_field(marks: Sequence[Mapping[str, Any]], field: str) -> Optional[int]:
    """The LARGEST value *field* takes anywhere in this rank's record.

    For a term the ledger charges over the whole boot, the honest measurement
    is the boot's maximum and not the value at one chosen phase. The first
    reconciliation read ``allocator_transient_bytes`` at the target runner's
    ``weights_loaded`` -- a mark at which the allocator has, by definition,
    just finished handing its peak back -- and reported 0 for a term whose
    real peak on that same card, three marks later, was 13392 MiB.
    """
    values = [int(m[field]) for m in marks if field in m]
    return max(values) if values else None


def _weight_episodes(
    marks: Sequence[Mapping[str, Any]],
) -> List[Tuple[str, int]]:
    """Every weight-load episode, in order, as ``(label, bytes)``.

    An EPISODE is one ``pre_weight_load -> weights_loaded`` pair. A boot can
    run several: this rig's phase-flip configuration loads the PP-prefill
    layout, frees it, loads the TP-decode layout, frees that, and then loads
    the NEXTN draft -- three episodes in one process.

    TWO CORRECTIONS OVER THE FIRST RUN, both of which inflated the answer:

    1. Episodes are not summed. Each is released before the next loads, so the
       card's weight demand is the LARGEST episode. Summing the first two gave
       27800 MiB on a card whose entire process footprint was 28436 MiB.
    2. The measurement runs on allocated bytes OUTSIDE the KV arena
       (``allocated_bytes - kv_arena_backed_bytes``) rather than on
       ``reserved_bytes``. Between episodes the arena takes over the pages the
       weights gave back -- that is what ``--phase-flip-spill-depth arena``
       does -- so a reserved-bytes delta charges the arena's growth to the
       weights.

    DEGRADED BASIS FOR OLD BOOTS. A mark written before ``allocated_bytes``
    existed carries only ``reserved_bytes``. Such a boot is still reconciled,
    on the reserved basis, rather than reading UNMEASURED -- it has no KV
    arena fields either, so on those boots the two bases agree. The degradation
    is recorded in the episode label so a reader can see which basis produced
    the number instead of having to date the boot.
    """
    episodes: List[Tuple[str, int]] = []
    pending: Optional[int] = None
    index = 0
    for m in marks:
        phase = m.get("phase")
        if phase not in ("pre_weight_load", "weights_loaded"):
            continue
        if "allocated_bytes" in m:
            outside = int(m.get("allocated_bytes", 0)) - int(
                m.get("kv_arena_backed_bytes", 0) or 0
            )
        elif "reserved_bytes" in m:
            outside = int(m["reserved_bytes"])
        else:
            continue
        if phase == "pre_weight_load":
            pending = outside
        elif pending is not None:
            index += 1
            tag = (m.get("extra") or {}).get("draft_worker")
            basis = "" if "allocated_bytes" in m else " [reserved basis]"
            label = (
                f"episode {index}"
                + (
                    " (target runner)"
                    if tag is False
                    else " (draft/second runner)" if tag is True else ""
                )
                + basis
            )
            episodes.append((label, outside - pending))
            pending = None
    return episodes


def _field_bytes(
    marks: Sequence[Mapping[str, Any]], field: str, phase: str
) -> Optional[int]:
    """A field at one phase, taken as the HIGHEST value any runner reported.

    WHY NOT THE FIRST MATCH. A speculative process reaches every per-runner
    phase twice or more, and ``next(...)`` therefore always returned the
    TARGET runner's mark. The draft runner routinely carries the higher
    level: on live boot 1917721 the hardware residual reads 886 MiB on the
    target and 896 MiB on the draft, so first-match under-read the term on
    every card of every speculative boot. The same shape, one term over, is
    what made the load transient read 0 while the boot's real peak was 13392
    MiB -- a target-runner zero hiding a draft-runner peak.

    A field term is a LEVEL, not a delta, so summing runners would be wrong;
    the card must fund the highest level any runner reached, which is the
    maximum. Process-level phases (``boot_complete``) appear once with one
    value, so their measurement is unchanged by this and stays exact.
    """
    values = [
        int(m[field])
        for m in marks
        if m.get("phase") == phase and field in m and m[field] is not None
    ]
    if not values:
        return None
    return max(values)


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
    #: Terms computed with a global delta because marks lacked runner tags.
    ambiguous_runner_terms: List[str] = []
    for term in ledger.get("terms", []):
        name = str(term.get("name", ""))
        modeled_mib = int(term.get("mib", 0))
        if term.get("not_applicable"):
            # A term the CONFIGURATION removes is not a term the instrument
            # failed to see. NCCL communicator buffers are correctly 0 when
            # barlink carries the collectives and PyNccl is never built;
            # reporting that as UNMEASURED invites a successor to go looking
            # for a boundary, find the one that now exists, measure 0, and
            # conclude the recorder is broken.
            comparisons.append(
                TermComparison(
                    term=name,
                    modeled_mib=modeled_mib,
                    measured_mib=None,
                    basis="",
                    provenance=str(term.get("provenance", "")),
                    note=(
                        "not applicable to this launch"
                        + (
                            " (barlink carries the collectives, so no PyNccl "
                            "communicator is built)"
                            if name == TERM_NCCL_BUFFERS
                            else ""
                        )
                        + " -- a priced zero, not an absent measurement"
                    ),
                    not_applicable=True,
                )
            )
            continue
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
        note = ""
        if kind == "peak":
            measured = _peak_field(marks, args[0])
            if measured is None:
                note = (
                    f"no mark in this boot's record carries {args[0]!r}; the "
                    "recorder predates the field"
                )
        elif kind == "episodes":
            episodes = _weight_episodes(marks)
            if not episodes:
                measured = None
                note = (
                    "no (pre_weight_load -> weights_loaded) pair carrying "
                    "allocated_bytes in this boot's record"
                )
            else:
                measured = max(b for _label, b in episodes)
                note = (
                    "largest of "
                    + ", ".join(f"{label} {b // MIB} MiB" for label, b in episodes)
                    + "; NOT their sum -- each is freed before the next loads"
                )
        elif kind == "delta":
            absent = [p for p in (args[0], args[1]) if p not in phases]
            if absent:
                measured = None
                note = (
                    "the boot recorded no "
                    + " and no ".join(repr(p) for p in absent)
                    + f" mark, so the gap {args[0]!r} -> {args[1]!r} that "
                    "brackets this term does not exist in the record. The "
                    "boundary is named so the gap is a to-do with an address "
                    "rather than a permanent blank"
                )
            else:
                measured, ambiguous = _delta_bytes(
                    marks, args[0], args[1], "reserved_bytes"
                )
                if ambiguous and measured is not None:
                    ambiguous_runner_terms.append(name)
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
                note=note,
            )
        )

    # Terms the MODEL refused to price. Without these rows a refusal and a
    # term that does not exist in the taxonomy look identical in the table,
    # which is precisely the ambiguity the ledger's UNBOUNDED list exists to
    # remove. They carry no modeled number by construction, so they are shown
    # against their measurement where one exists -- a refused term with a
    # measured post is the strongest possible argument for calibrating it.
    for refusal in ledger.get("unbounded", ()) or ():
        text = str(refusal)
        refused_term = next(
            (t for t in TERM_TO_POST if text.startswith(t)),
            text.split(" on ")[0],
        )
        mapping = TERM_TO_POST.get(refused_term)
        measured_refused: Optional[int] = None
        if mapping is not None:
            (kind, *args), _basis = mapping
            if kind == "peak":
                measured_refused = _peak_field(marks, args[0])
            elif kind == "delta" and not [
                p for p in (args[0], args[1]) if p not in phases
            ]:
                measured_refused, _amb = _delta_bytes(
                    marks, args[0], args[1], "reserved_bytes"
                )
            elif kind == "field":
                measured_refused = _field_bytes(marks, args[0], args[1])
        comparisons.append(
            TermComparison(
                term=refused_term,
                modeled_mib=None,
                measured_mib=(
                    None if measured_refused is None else measured_refused // MIB
                ),
                basis="",
                provenance="unbounded",
                note="REFUSED BY THE MODEL: " + text,
                refused=True,
            )
        )

    # Measured internal demand: what this rank's process actually holds on the
    # card at boot_complete, plus the carve-out, which is the card's and not
    # the process's. NVML's per-pid figure is used rather than torch's reserved
    # so the CUDA context and the driver windows are inside the number.
    self_bytes = _field_bytes(marks, "nvml_self_bytes", "boot_complete") or 0
    carve_out = _field_bytes(marks, "nvml_carve_out_bytes", "boot_complete") or 0
    # The KV pool subtracted here is the one the boot ACTUALLY got, read from
    # the arena census, and only falls back to the ledger's budget when the
    # marks cannot say. The budget is the wrong number to subtract from a
    # measurement: on boot 1464299 the ledger budgeted 29927 MiB on the 5090
    # and the arena ended up backing 21130, so subtracting the budget removed
    # 8797 MiB that the process never held and drove the measured demand
    # NEGATIVE. A demand of -3045 MiB is not a small error; it is a totals row
    # that cannot be read at all, and the first reconciliation had to patch
    # around it by hand in prose.
    measured_kv_pool = _measured_kv_pool_bytes(marks)
    demand_refusal = ""
    measured_demand: Optional[int]
    if measured_kv_pool is None:
        # NEVER the modelled budget. Subtracting a BUDGET from a MEASUREMENT
        # is what drove this number negative on live boots -- -2023 MiB on the
        # 5090 and -180 MiB on a 3080 of boot 1917721, where the ledger
        # budgeted 29927 / 18245 MiB and the profiler clamped the real pool
        # well below it. A negative demand is not a small error: it is a
        # totals row that cannot be read, and the temptation is then to patch
        # it in prose, which the first reconciliation had to do.
        measured_demand = None
        demand_refusal = (
            "REFUSED: this boot's marks carry no measured KV pool "
            "(kv_arena_backed_bytes at boot_complete, nor a target-runner "
            "weights_loaded -> kv_pool_sized growth), so the measured demand "
            "cannot be formed. The ledger's budgeted kv_pool_mib is "
            "deliberately NOT used as a fallback: subtracting a budget from a "
            "measurement is what made this figure negative on live boots"
        )
    else:
        measured_demand = self_bytes + carve_out - measured_kv_pool

    residuum = None if measured_demand is None else measured_demand - claimed_bytes
    return CardReconciliation(
        gpu_id=int(ledger.get("gpu_id", 0)),
        card=str(ledger.get("card", "")),
        rank=rank,
        comparisons=tuple(comparisons),
        measured_demand_mib=(
            None if measured_demand is None else measured_demand // MIB
        ),
        modeled_demand_mib=int(ledger.get("demand_mib", 0)),
        demand_refusal=demand_refusal,
        residuum_mib=None if residuum is None else residuum // MIB,
        residuum_note=(
            ""
            if residuum is None
            else (
                "measured bytes on this card that no term claims. Not an error "
                "bar: it is either a post nobody modeled or a mapping above that "
                "does not bracket what it says it does"
                if abs(residuum) // MIB
                else "every measured MiB is claimed by a term above"
            )
        ),
        missing_phases=missing,
        ambiguous_runner_terms=tuple(ambiguous_runner_terms),
    )


def reconcile(
    ledger_payload: Mapping[str, Any],
    marks_by_rank: Mapping[int, Sequence[Mapping[str, Any]]],
) -> List[CardReconciliation]:
    """Every card of one boot. Cards are matched to ranks through the ledger's
    own ``ranks`` field, never by index.

    ACCEPTS EITHER KEYING. ``flight_recorder.read_marks`` returns marks keyed
    by PID and every caller in this tree hands its result straight in;
    :func:`marks_by_rank_from_pids` re-keys it and REFUSES on anything it
    cannot match exactly. A pid-keyed dict used to sail through this function
    and produce an empty list, which the CLI printed as a fact about the boot.
    """
    marks_by_rank = marks_by_rank_from_pids(marks_by_rank, ledger_payload)
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
