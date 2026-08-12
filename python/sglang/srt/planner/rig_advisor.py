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
"""Rig buying advisor (#413): what would card X actually buy me?

The question a rig owner asks before spending money is "if I add this card --
or swap that one for it -- what do I get?". The honest answer is rarely the
one a product page implies, so this module is built to produce the
UNCOMFORTABLE version:

  * A card that is not in this machine cannot be measured, so no "after"
    number may carry the ``measured`` label. That is enforced structurally by
    :func:`rig_with_candidate` (the candidate rig's ``source`` makes
    ``explorer.provenance_of`` report composed/estimate), not by remembering
    to pass the right string at each call site.
  * A bought card lands in the slot that is physically FREE, not in the slot
    its datasheet assumes. An x16 card in an x4 slot negotiates x4, and the
    slot -- not the card -- sets the width.
  * The group runs at the pace of its slowest rank. Adding a fast card beside
    an unchanged bottleneck buys VRAM and almost no decode rate, and this
    module says so out loud instead of reporting the average.
  * Adding a card makes the TP collective WORSE for every existing rank. On a
    no-P2P rig the cross-card knock-down grows with the number of cards
    crossed, so the fourth card is charged to the three you already own.
  * Where the model has no basis for a number, the answer is ``absent``. An
    absent cell is a real, informative result, never a plausible filler.

WHAT THIS MODULE IS NOT: new physics. Every number comes from machinery that
already exists and is already calibrated:

  ``feasibility.plan()``      -- fit, refusal reasons, capacity and the
                                 recommended split; it already attaches
                                 ``.roofline`` (#145) and ``.roofline_energy``
                                 (#148) to its result.
  ``card_library.CardSpec``   -- the candidate's datasheet record (#397: a
                                 spec record with nowhere to put a device
                                 index, so it cannot fake one).
  ``explorer.provenance_of``  -- the composed-vs-live label, derived from
                                 ``HardwareSpec.source`` alone.
  ``wizard.cell``             -- the wire format for one labelled number, so
                                 the advisor's cells render through the same
                                 provenance pills as the guide's family
                                 matrix (#270).

The advisor is the DIFF of two ``plan()`` runs -- the rig as it is, and the
rig as it would be. Adding nothing must therefore change nothing, and that
identity is pinned in ``test_rig_advisor_413.py``.

PROVENANCE VOCABULARY: the three words of ``bench_factors``
(``measured`` / ``estimate`` / ``absent``), imported rather than restated.
The roofline's own more specific ``planner-estimate`` tag is not a fourth
label -- it travels in each cell's ``basis`` text, where it stays visible
without inventing a pill the stylesheet has no colour for.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.planner.bench_factors import ABSENT, ESTIMATE, MEASURED
from sglang.srt.planner.card_library import CardLibrary, CardSpec
from sglang.srt.planner.explorer import provenance_of
from sglang.srt.planner.hardware import GpuDescriptor, HardwareSpec
from sglang.srt.planner.roofline import ROOFLINE_PROVENANCE
from sglang.srt.planner.wizard import cell

__all__ = [
    "ADVISOR_RIG_SOURCE",
    "FreeSlot",
    "UNKNOWN_SLOT",
    "Metric",
    "ModelVerdict",
    "AdvisorResult",
    "rig_with_candidate",
    "advise",
]

#: The ``HardwareSpec.source`` every candidate rig carries. Deliberately the
#: SAME string ``card_library.compose_rig`` uses: ``explorer.provenance_of``
#: already maps it to ("composed", estimate=True, note), so a rig holding a
#: candidate card cannot be rendered as measured by any caller -- including
#: callers written later that never read this module.
ADVISOR_RIG_SOURCE = "library-composition"

#: Bases, spelled once. Each names the instrument AND its honest weakness.
_BASIS_CAPACITY_LIVE = "VRAM ledger over live NVML totals"
_BASIS_CAPACITY_COMPOSED = (
    "VRAM ledger over datasheet totals (composed rig; capacity is "
    "interconnect-independent VRAM math, so it is the most trustworthy row "
    "here)"
)
_BASIS_ROOFLINE = (
    f"roofline #145, {ROOFLINE_PROVENANCE} provenance -- a ballpark from peak "
    f"rates times documented efficiency factors, never a measurement"
)
_BASIS_ENERGY = (
    f"roofline energy #148, {ROOFLINE_PROVENANCE} provenance -- board power "
    f"modelled from TDP between an idle floor and the nameplate ceiling"
)
_BASIS_TTFT = (
    f"prompt tokens / prefill tok-s from roofline #145 "
    f"({ROOFLINE_PROVENANCE}) -- prefill-bound FLOOR: it ignores queueing, "
    f"scheduling and the first decode step, so the real TTFT is higher"
)

#: Why a throughput cell is absent when the geometry was refused. The
#: roofline would happily price a rig that cannot boot -- it is a hardware
#: ballpark, not a feasibility check -- so the refusal, not a plausible tok/s,
#: is the answer that belongs in the cell.
_BASIS_REFUSED = (
    "the planner refused this configuration, so there is no throughput to "
    "report -- see the refusal reason on this row"
)

#: The study that would turn an absent throughput row into a measured one.
_STUDY_SPLIT_PROBE = "split-probe boot of this model on this rig"


@dataclasses.dataclass(frozen=True)
class FreeSlot:
    """The physical slot a bought card would actually occupy.

    A property of the MACHINE, not of the card, and the term most often left
    out of a buying decision. NVML cannot report it -- an empty slot has no
    device to query -- so it is either declared by the operator or it stays
    ``absent``, and absent propagates into the note rather than quietly
    becoming x16.

    Measured precedent on the reference rig (``nvidia-smi topo -m``
    2026-07-20 plus the cached card probe): all pairs PHB, no P2P, no NVLink,
    widths x4 / x8 / x8, and the x4-slotted RTX 3080 stages host traffic at
    6.45 GB/s against 13.41 GB/s for its identical x8 twin. Same silicon, a
    bit under half the bandwidth, purely from the slot.
    """

    pcie_gen: Optional[int] = None
    pcie_width: Optional[int] = None
    #: "declared" (the operator stated it) | ``ABSENT`` (nobody knows).
    provenance: str = ABSENT
    note: str = ""

    @property
    def known(self) -> bool:
        return self.pcie_gen is not None and self.pcie_width is not None

    def describe(self) -> str:
        if not self.known:
            return (
                "free slot: UNDECLARED -- an empty slot cannot be read from "
                "NVML, so the width a bought card would negotiate is unknown "
                "and the host-staging term falls back to the roofline's "
                "gen4 x8 default"
            )
        return f"free slot: gen{self.pcie_gen} x{self.pcie_width}"


#: The default: nobody has told us what the free slot is.
UNKNOWN_SLOT = FreeSlot()


@dataclasses.dataclass(frozen=True)
class Metric:
    """One target axis, before and after, each side with its OWN provenance.

    The two sides genuinely differ in kind -- the current rig's VRAM total is
    measured, the candidate rig's is a datasheet figure -- and collapsing them
    into one label would have to round in somebody's favour. Both are carried.
    """

    key: str
    label: str
    unit: str
    before: Optional[float]
    after: Optional[float]
    before_provenance: str
    after_provenance: str
    before_basis: str
    after_basis: str
    #: True when more is better; False for J/token and TTFT, where down wins.
    higher_is_better: bool = True
    study: Optional[str] = None

    @property
    def delta(self) -> Optional[float]:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    @property
    def delta_pct(self) -> Optional[float]:
        if not self.before or self.after is None:
            return None
        return (self.after - self.before) / self.before * 100.0

    @property
    def verdict(self) -> str:
        """``better`` / ``worse`` / ``unchanged`` / ``absent``, direction-aware.

        Both sides are estimates from the same model, so a hair's-breadth
        difference is arithmetic noise and must not be sold as an
        improvement: under a tenth of a percent reads as unchanged.
        """
        d = self.delta
        if d is None:
            return ABSENT
        if not self.before or abs(d / self.before) < 1e-3:
            return "unchanged"
        improved = d > 0 if self.higher_is_better else d < 0
        return "better" if improved else "worse"

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "label": self.label,
            "unit": self.unit,
            # wizard.cell() shape, so the guide's provenance pills render
            # these without a second renderer.
            "before": cell(
                self.before,
                self.before_provenance,
                self.before_basis,
                unit=self.unit,
                study=self.study,
            ),
            "after": cell(
                self.after,
                self.after_provenance,
                self.after_basis,
                unit=self.unit,
                study=self.study,
            ),
            "higher_is_better": self.higher_is_better,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
            "verdict": self.verdict,
        }


@dataclasses.dataclass(frozen=True)
class ModelVerdict:
    """The before/after answer for ONE model family."""

    label: str
    model_path: str
    before_fits: bool
    after_fits: bool
    #: Refusal reasons verbatim from the planner (#270). Empty when it fits.
    before_refusal: List[str]
    after_refusal: List[str]
    metrics: List[Metric]
    #: Who clocks the group, before and after -- the slowest-rank truth.
    before_bottleneck: Optional[str]
    after_bottleneck: Optional[str]
    #: What the planner would solve this geometry to (the launch flags).
    after_split: List[str]
    notes: List[str]

    def metric(self, key: str) -> Metric:
        for m in self.metrics:
            if m.key == key:
                return m
        raise KeyError(key)

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "model_path": self.model_path,
            "before_fits": self.before_fits,
            "after_fits": self.after_fits,
            "before_refusal": self.before_refusal,
            "after_refusal": self.after_refusal,
            "metrics": [m.to_dict() for m in self.metrics],
            "before_bottleneck": self.before_bottleneck,
            "after_bottleneck": self.after_bottleneck,
            "after_split": self.after_split,
            "notes": self.notes,
        }


@dataclasses.dataclass(frozen=True)
class AdvisorResult:
    candidate: str
    mode: str  # "add" | "replace"
    replaced: Optional[str]
    before_rig: str
    after_rig: str
    before_provenance: str
    after_provenance: str
    free_slot: str
    rows: List[ModelVerdict]
    #: The things a product page does not say. Always rendered, never folded
    #: away behind a disclosure control.
    truths: List[str]

    def to_dict(self) -> Dict:
        return {
            "ok": True,
            "candidate": self.candidate,
            "mode": self.mode,
            "replaced": self.replaced,
            "before_rig": self.before_rig,
            "after_rig": self.after_rig,
            "before_provenance": self.before_provenance,
            "after_provenance": self.after_provenance,
            "free_slot": self.free_slot,
            "rows": [r.to_dict() for r in self.rows],
            "truths": self.truths,
        }


# ---------------------------------------------------------------------------
# Rig composition: the rig as it would be.
# ---------------------------------------------------------------------------


def rig_with_candidate(
    hardware: HardwareSpec,
    candidate: CardSpec,
    *,
    mode: str = "add",
    replace_index: Optional[int] = None,
    free_slot: FreeSlot = UNKNOWN_SLOT,
) -> HardwareSpec:
    """The rig with ``candidate`` added, or swapped in for ``replace_index``.

    Three honesty properties are STRUCTURAL here -- they hold whatever the
    caller then does with the result:

    1. The spec's source is :data:`ADVISOR_RIG_SOURCE`, so
       ``explorer.provenance_of`` reports composed/estimate. No code path
       yields a candidate rig labelled live or measured.
    2. Every card comes back with ``free_mib=None`` and ``cuda_index=None``.
       A hypothetical rig has no live free-VRAM reading, and #397 forbids
       inventing a CUDA ordinal for a card that is not in the machine: an
       unknown identity stays None rather than becoming a plausible integer
       that a later lookup would trust.
    3. The candidate's link width comes from the SLOT, not the datasheet.
    """
    if mode not in ("add", "replace"):
        raise ValueError(f"mode must be 'add' or 'replace', got {mode!r}")
    if not hardware.gpus:
        raise ValueError("cannot advise on a rig that declares zero GPUs.")

    # The candidate enters as a spec record (#397). Its width is the slot's,
    # and stays None when the slot is undeclared -- the roofline then applies
    # its own documented default and the truth list says so, which is honest;
    # silently substituting the card's nameplate x16 would not be.
    new_card = GpuDescriptor(
        index=0,  # positional, fixed up once the final order is known
        name=candidate.name,
        total_mib=candidate.total_mib,
        free_mib=None,
        uuid=None,
        pcie_gen=free_slot.pcie_gen,
        pcie_width=free_slot.pcie_width,
        cuda_index=None,
    )

    cards = list(hardware.gpus)
    if mode == "replace":
        if replace_index is None:
            raise ValueError("mode='replace' needs replace_index.")
        if not 0 <= replace_index < len(cards):
            raise ValueError(
                f"replace_index {replace_index} is out of range for a rig "
                f"with {len(cards)} card(s)."
            )
        # A swap reuses the outgoing card's slot, and THAT width was read
        # from NVML -- a measured property of the machine, which beats both a
        # declaration and the incoming card's datasheet.
        outgoing = cards[replace_index]
        if outgoing.pcie_width is not None and not free_slot.known:
            new_card = dataclasses.replace(
                new_card,
                pcie_gen=outgoing.pcie_gen,
                pcie_width=outgoing.pcie_width,
            )
        cards[replace_index] = new_card
    else:
        cards.append(new_card)

    # Renumber positionally and strip every live-only field. A leftover
    # free_mib or cuda_index from the live spec would be the one field still
    # claiming to be measured on a rig that no longer exists.
    composed = tuple(
        dataclasses.replace(c, index=i, free_mib=None, cuda_index=None)
        for i, c in enumerate(cards)
    )
    return HardwareSpec(
        gpus=composed,
        source=ADVISOR_RIG_SOURCE,
        host_ram_mib=hardware.host_ram_mib,
        cuda_index_source=None,
    )


def _rig_label(hardware: HardwareSpec) -> str:
    from collections import Counter

    names = [g.name for g in hardware.gpus]
    counter = Counter(names)
    seen: List[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return ", ".join(f"{counter[n]}x {n}" for n in seen)


# ---------------------------------------------------------------------------
# Metric extraction from a PlanResult.
# ---------------------------------------------------------------------------


def _capacity_basis(hardware: HardwareSpec) -> Tuple[str, str]:
    """(provenance, basis) for the VRAM-math row."""
    prov, estimate, _ = provenance_of(hardware)
    if estimate or prov != "live":
        return ESTIMATE, _BASIS_CAPACITY_COMPOSED
    return MEASURED, _BASIS_CAPACITY_LIVE


def _bottleneck_label(result) -> Optional[str]:
    """ "rank 2 (RTX 3080)" -- who clocks the group, per the roofline."""
    roof = getattr(result, "roofline", None)
    if roof is None:
        return None
    rank = getattr(roof, "bottleneck_rank", None)
    if rank is None:
        return None
    for entry in getattr(roof, "per_rank", None) or ():
        if getattr(entry, "rank", None) == rank:
            return f"rank {rank} ({getattr(entry, 'gpu_name', '?')})"
    return f"rank {rank}"


def _runnable(result) -> bool:
    """Whether this plan describes something that could actually boot.

    ``fits`` alone is too strict: a plan that overflows VRAM but tiers the
    MoE experts to host RAM does run, slower, and its throughput figures are
    meaningful. A plan whose offload assessment is ``cannot_fit`` does not run
    at all.
    """
    if getattr(result, "fits", False):
        return True
    offload = getattr(result, "offload", None)
    return getattr(offload, "status", None) == "ram_offload"


def _plan_metrics(result, prompt_tokens: int) -> Dict[str, Optional[float]]:
    """The five target axes out of one PlanResult, each possibly absent.

    Absent is a real outcome here, and one case deserves naming: the roofline
    is happy to price a geometry that CANNOT BOOT. It is a hardware ballpark,
    not a feasibility check, so on a plan the planner refused it still returns
    a perfectly plausible tok/s -- for the reference rig, 960 prefill tok/s
    for a swap that leaves every rank with 0 KV tokens. Printing that beside a
    refusal is precisely the marketing arithmetic this feature exists to
    avoid: it reads as "somewhat slower" when the truth is "does not run". So
    throughput, TTFT and energy are ABSENT unless the plan is runnable, and
    the refusal reason carries the answer instead.

    ``max_context`` is kept in every case, because on a refused plan it is
    0 tokens -- which is not a filler value but the refusal restated in the
    row's own unit.
    """
    out: Dict[str, Optional[float]] = {
        "max_context": None,
        "prefill_tok_s": None,
        "decode_tok_s": None,
        "ttft_s": None,
        "j_per_token": None,
    }
    if result is None:
        return out

    capacity = getattr(result, "capacity", None)
    if capacity is not None:
        out["max_context"] = getattr(capacity, "max_context_tokens", None)

    if not _runnable(result):
        return out

    roof = getattr(result, "roofline", None)
    if roof is not None:
        out["prefill_tok_s"] = getattr(roof, "prefill_tok_s", None)
        # The LOW end of the decode range: the rate AT the reference context,
        # with KV reads paid for. The high end is a short-context ceiling that
        # would flatter every candidate equally and decide nothing.
        out["decode_tok_s"] = getattr(roof, "decode_tok_s_low", None)
        if out["prefill_tok_s"]:
            out["ttft_s"] = prompt_tokens / float(out["prefill_tok_s"])

    energy = getattr(result, "roofline_energy", None)
    if energy is not None:
        out["j_per_token"] = getattr(energy, "j_per_decode_token_total", None)
    return out


#: (key, label, unit, higher_is_better, basis, study)
_AXES: Tuple[Tuple[str, str, str, bool, str, Optional[str]], ...] = (
    ("max_context", "Max context", "tokens", True, "", None),
    (
        "prefill_tok_s",
        "Prefill throughput",
        "tok/s",
        True,
        _BASIS_ROOFLINE,
        _STUDY_SPLIT_PROBE,
    ),
    (
        "decode_tok_s",
        "Decode throughput",
        "tok/s",
        True,
        _BASIS_ROOFLINE,
        _STUDY_SPLIT_PROBE,
    ),
    (
        "ttft_s",
        "TTFT (prefill-bound floor)",
        "s",
        False,
        _BASIS_TTFT,
        _STUDY_SPLIT_PROBE,
    ),
    (
        "j_per_token",
        "Energy per decode token",
        "J/token",
        False,
        _BASIS_ENERGY,
        "power-calibration sweep with the candidate card installed",
    ),
)


def _refusal_reasons(result, error) -> List[str]:
    if error is not None:
        return list(getattr(error, "reasons", None) or [str(error)])
    if result is None:
        return ["the planner returned no result"]
    if not getattr(result, "fits", False):
        return list(getattr(result, "infeasible_reasons", None) or ["does not fit"])
    return []


def _run_plan(model_path: str, hardware: HardwareSpec, plan_kwargs: Dict):
    """``plan()`` with its refusals captured rather than raised.

    A refusal is an ANSWER in this tab -- "that card would not let this model
    run, and here is the planner's own reason" -- so it must reach the UI
    intact instead of collapsing into an empty row.
    """
    from sglang.srt.planner.feasibility import PlanRejected, plan

    try:
        return plan(model_path, hardware, **plan_kwargs), None
    except (PlanRejected, ValueError) as e:
        return None, e


# ---------------------------------------------------------------------------
# The uncomfortable truths.
# ---------------------------------------------------------------------------


def _measured_truth(candidate: CardSpec) -> str:
    if candidate.gemm_tflops is not None or candidate.membw_gbs is not None:
        return (
            f"Provenance: {candidate.name} carries probe-derived rates in the "
            f"card library, but it is not in THIS machine and its rates were "
            f"not measured beside these cards -- every 'after' number stays "
            f"an estimate."
        )
    return (
        f"Provenance: {candidate.name} has never been measured here. Its "
        f"rates are datasheet peaks knocked down by the roofline's documented "
        f"efficiency factors, so every 'after' number is an estimate and no "
        f"'after' cell may be read as a measurement."
    )


def _slot_truth(candidate: CardSpec, free_slot: FreeSlot, mode: str) -> Optional[str]:
    """What the free slot costs -- and, precisely, which term it moves.

    Being exact matters here: the width integer reaches the roofline's
    host-staging bandwidth (``_pcie_fetch_gbs``, the MIN across the cards in
    the plan) and nothing else. It does NOT enter the TP collective discount,
    which is tiered on the NUMBER of cards crossed. Claiming otherwise would
    be inventing a mechanism the code does not have.
    """
    if mode != "add":
        return None
    if not free_slot.known:
        return (
            "Slot width: UNDECLARED. An empty slot cannot be read from NVML, "
            "so the width this card would negotiate is unknown and the "
            "host-staging term uses the roofline's gen4 x8 default. If the "
            "free slot is narrower, the offload and staging figures below are "
            "optimistic."
        )
    nameplate = candidate.pcie_width
    tail = (
        " That width sets the host-staging bandwidth (the roofline takes the "
        "MIN across the cards in the plan, so the narrowest card throttles "
        "staging for all of them). It does not change the TP collective "
        "discount, which is tiered on the number of cards crossed."
    )
    if nameplate and free_slot.pcie_width and free_slot.pcie_width < nameplate:
        return (
            f"Slot width: the card is an x{nameplate} part but the free slot "
            f"is gen{free_slot.pcie_gen} x{free_slot.pcie_width}, so it "
            f"negotiates x{free_slot.pcie_width}. Measured precedent on this "
            f"rig: the x4-slotted RTX 3080 stages host traffic at 6.45 GB/s "
            f"against 13.41 GB/s for its identical x8 twin." + tail
        )
    return f"Slot width: gen{free_slot.pcie_gen} x{free_slot.pcie_width}." + tail


def _collective_truth(before: HardwareSpec, after: HardwareSpec) -> Optional[str]:
    """Adding a card is charged to every rank you already own."""
    from sglang.srt.planner import roofline as _rf

    n_before, n_after = len(before.gpus), len(after.gpus)
    if n_after <= n_before:
        return None
    tiers = _rf._PCIE_NOP2P_BY_CROSS_CARDS
    d_before = tiers.get(n_before, _rf._PCIE_NOP2P_MANY)
    d_after = tiers.get(n_after, _rf._PCIE_NOP2P_MANY)
    if d_after >= d_before:
        return None
    loss = (1.0 - d_after / d_before) * 100.0
    return (
        f"Collective cost: with no P2P and no NVLink every TP collective is "
        f"host-staged. Going from {n_before} to {n_after} cards moves the "
        f"roofline's cross-card discount from {d_before:.2f} to {d_after:.2f} "
        f"-- about {loss:.0f}% off the shared throughput term, charged to the "
        f"ranks you already own and not only to the new card."
    )


def _bottleneck_truth(rows: Sequence[ModelVerdict]) -> Optional[str]:
    """The slowest rank clocks the group -- name it when it does not move."""
    stuck = [
        r
        for r in rows
        if r.after_fits
        and r.before_bottleneck
        and r.after_bottleneck == r.before_bottleneck
    ]
    if not stuck:
        return None
    return (
        f"Slowest rank: {stuck[0].after_bottleneck} still clocks the group "
        f"after the change ({', '.join(r.label for r in stuck)}). Ranks step "
        f"together, so a faster card added beside an unchanged bottleneck "
        f"buys capacity, not decode rate."
    )


def _unlocked_truth(rows: Sequence[ModelVerdict]) -> Optional[str]:
    """The genuinely good news, stated only where it is true."""
    unlocked = [r.label for r in rows if r.after_fits and not r.before_fits]
    lost = [r.label for r in rows if r.before_fits and not r.after_fits]
    parts = []
    if unlocked:
        parts.append(
            f"Newly possible: {', '.join(unlocked)} does not fit today and "
            f"would fit after this change."
        )
    if lost:
        parts.append(
            f"REGRESSION: {', '.join(lost)} fits today and would STOP fitting "
            f"after this change -- read the refusal reason in the row."
        )
    return " ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# The entry point.
# ---------------------------------------------------------------------------


def advise(
    hardware: HardwareSpec,
    candidate,
    models: Sequence[Tuple[str, str]],
    *,
    mode: str = "add",
    replace_index: Optional[int] = None,
    free_slot: FreeSlot = UNKNOWN_SLOT,
    library: Optional[CardLibrary] = None,
    prompt_tokens: int = 4096,
    plan_kwargs: Optional[Dict] = None,
) -> AdvisorResult:
    """Answer "what would ``candidate`` buy me?" for each model in ``models``.

    ``candidate`` is a :class:`CardSpec` or a name in ``library``. ``models``
    is a list of ``(label, model_path)`` -- the wizard's families, or any
    subset the operator picks.

    The result is the diff of two ``plan()`` runs. Running it against a
    candidate rig identical to the current one must reproduce the current
    numbers exactly; that identity is what makes the diff trustworthy, and it
    is pinned by ``test_rig_advisor_413.py``.
    """
    library = library or CardLibrary()
    spec = candidate if isinstance(candidate, CardSpec) else library.get(candidate)

    # The candidate may be a hand-typed card the seed set has never heard of.
    # plan() looks nameplate peaks up in the library it is given, so the
    # library carrying this spec has to travel with the plan -- otherwise the
    # roofline finds no peaks for it and the whole estimate goes absent.
    plan_library = CardLibrary({n: library.get(n) for n in library.names()})
    plan_library.add(spec, overwrite=False)
    plan_kwargs = dict(plan_kwargs or {})
    plan_kwargs.setdefault("card_library", plan_library)

    after_hw = rig_with_candidate(
        hardware,
        spec,
        mode=mode,
        replace_index=replace_index,
        free_slot=free_slot,
    )
    before_cap_prov, before_cap_basis = _capacity_basis(hardware)
    after_cap_prov, after_cap_basis = _capacity_basis(after_hw)

    rows: List[ModelVerdict] = []
    for label, model_path in models:
        before_res, before_err = _run_plan(model_path, hardware, plan_kwargs)
        after_res, after_err = _run_plan(model_path, after_hw, plan_kwargs)
        before_vals = _plan_metrics(before_res, prompt_tokens)
        after_vals = _plan_metrics(after_res, prompt_tokens)
        before_runs, after_runs = _runnable(before_res), _runnable(after_res)

        metrics: List[Metric] = []
        for key, axis_label, unit, higher_better, basis, study in _AXES:
            b, a = before_vals[key], after_vals[key]
            if key == "max_context":
                b_prov, b_basis = before_cap_prov, before_cap_basis
                a_prov, a_basis = after_cap_prov, after_cap_basis
            else:
                # Throughput and energy are roofline derivations on BOTH
                # sides. The current rig's are estimates too -- measured
                # serving figures live in the split-probe store, not here --
                # and saying so is the difference between an honest
                # comparison and one that flatters whichever side happens to
                # be measured.
                b_prov = a_prov = ESTIMATE
                b_basis = a_basis = basis
                # Name the absence rather than leaving a bare dash: "no
                # number" and "the planner refused this configuration" are
                # different answers, and only the second one is advice.
                if not before_runs:
                    b_basis = _BASIS_REFUSED
                if not after_runs:
                    a_basis = _BASIS_REFUSED
            metrics.append(
                Metric(
                    key=key,
                    label=axis_label,
                    unit=unit,
                    before=b,
                    after=a,
                    before_provenance=b_prov if b is not None else ABSENT,
                    after_provenance=a_prov if a is not None else ABSENT,
                    before_basis=b_basis,
                    after_basis=a_basis,
                    higher_is_better=higher_better,
                    study=study,
                )
            )

        notes: List[str] = []
        for res, side in ((before_res, "before"), (after_res, "after")):
            roof = getattr(res, "roofline", None)
            note = getattr(roof, "interconnect_note", None) if roof else None
            if note:
                notes.append(f"{side}: {note}")

        rows.append(
            ModelVerdict(
                label=label,
                model_path=model_path,
                before_fits=bool(getattr(before_res, "fits", False)),
                after_fits=bool(getattr(after_res, "fits", False)),
                before_refusal=_refusal_reasons(before_res, before_err),
                after_refusal=_refusal_reasons(after_res, after_err),
                metrics=metrics,
                before_bottleneck=_bottleneck_label(before_res),
                after_bottleneck=_bottleneck_label(after_res),
                after_split=list(getattr(after_res, "launch_flags", None) or []),
                notes=notes,
            )
        )

    truths: List[str] = [_measured_truth(spec)]
    for maybe in (
        _slot_truth(spec, free_slot, mode),
        _collective_truth(hardware, after_hw),
        _bottleneck_truth(rows),
        _unlocked_truth(rows),
    ):
        if maybe:
            truths.append(maybe)

    replaced = (
        hardware.gpus[replace_index].name
        if mode == "replace" and replace_index is not None
        else None
    )
    return AdvisorResult(
        candidate=spec.name,
        mode=mode,
        replaced=replaced,
        before_rig=_rig_label(hardware),
        after_rig=_rig_label(after_hw),
        before_provenance=provenance_of(hardware)[0],
        after_provenance=provenance_of(after_hw)[0],
        free_slot=free_slot.describe(),
        rows=rows,
        truths=truths,
    )
