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
"""Offload DEPTH as a wizard dimension, with the counter-reckoning attached.

WHY DEPTH IS A DIMENSION AND NOT A SWITCH
-----------------------------------------
Expert offload was a modifier in the v1 matrix: on or off, composes with
every family. That is true and it is not the decision anybody actually makes.
The decision is HOW MUCH -- which fraction of the routed experts stays
resident on the card and which fraction lives in the pinned host pool. The
fraction is a continuous knob (``--moe-resident-expert-fraction`` /
``SGLANG_MOE_RESIDENT_EXPERT_FRACTION``) whose two ends are two entirely
different configurations: at 1.0 nothing is offloaded and the model may not
fit at all; at 0.0 everything routed crosses PCIe on demand and the model
fits on hardware that could not otherwise hold it.

So it is a dimension with STEPS, and every step owes the reader the same
three-part line:

    what it BUYS (VRAM released, and what that VRAM then buys),
    what it COSTS (the routed bytes move from the card's ~TB/s memory to a
    single-digit-GB/s PCIe path),
    and WHO it is worth it for (the condition under which the trade is the
    right one).

THE HONEST HALF
---------------
The buy side is arithmetic over the plan's own offloadable pool, so it is an
``estimate`` and it is reported per step. The cost side for THIS model on
THIS rig is ``absent``: no boot has measured it. What exists is three boots
of other models, and they are quoted as what they are -- evidence that the
mechanism works and roughly what it costs, never as this model's number.
That is the whole discipline: the number that flatters the feature is
computed, the number that prices it is not, and pretending otherwise by
interpolating somebody else's decode rate is exactly the failure the three
provenance words exist to prevent.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

from sglang.srt.planner.bench_factors import ABSENT, ESTIMATE

__all__ = [
    "OFFLOAD_STEPS",
    "EVIDENCE",
    "OffloadSources",
    "offload_dimension",
]

_MIB = 1024.0 * 1024.0

#: The resident fractions the wizard offers. Steps rather than a free slider
#: because each one is a different ANSWER -- "fits without offload", "fits
#: with the tail parked", "fits only because most of the stack is in host
#: RAM" -- and a slider invites tuning a quantity nobody has measured here.
OFFLOAD_STEPS: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)


@dataclasses.dataclass(frozen=True)
class OffloadEvidence:
    """One boot that measured something about offload, and its limits."""

    key: str
    model: str
    hardware: str
    fraction: Optional[float]
    showed: str
    #: Why this figure is not this configuration's figure.
    does_not_transfer: str

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


#: The boots the project has on record. Quoted for what they establish -- the
#: mechanism works, and roughly what it releases -- and never interpolated to
#: another checkpoint.
EVIDENCE: Tuple[OffloadEvidence, ...] = (
    OffloadEvidence(
        key="123-awq",
        model="Qwen3.6-35B-A3B-AWQ-4bit, 23.25 GiB of weights",
        hardware="one RTX 3080, 20.00 GiB",
        fraction=0.25,
        showed=(
            "the boot itself is the proof: 23.25 GiB of weights served from a "
            "20.00 GiB card. 40 layers at 64/256 residents plus 16 scratch "
            "released 11.92 GiB of weight VRAM (13.01 GiB into the pinned "
            "host pool); steady state 14540 of 20480 MiB against a "
            "counterfactual 26.12 GiB. Load took 2 min 36 s"
        ),
        does_not_transfer=(
            "an AWQ-marlin path on one sm86 card. Both the released bytes and "
            "the load time follow the checkpoint's expert geometry and the "
            "quantisation path, neither of which carries to another model"
        ),
    ),
    OffloadEvidence(
        key="256-fp8",
        model="Qwen3.6-35B-A3B-FP8, 31 GiB",
        hardware="one RTX 5090, TP=1",
        fraction=0.25,
        showed=(
            "20.63 GiB of weight VRAM released to the pinned host pool with "
            "coherent output. It also closed the wave-order question: "
            "token-major and expert-major are byte-identical (386/386 "
            "characters), and expert-major cuts the per-layer per-chunk H2D "
            "traffic about 3.5x (1.11-1.21 -> 0.34 GiB, 27-31 waves -> 9)"
        ),
        does_not_transfer=(
            "a single-card fp8 boot. The wave-order ratio is a property of "
            "the chunk size and the expert count, and the released bytes are "
            "a property of this checkpoint"
        ),
    ),
    OffloadEvidence(
        key="77-122b",
        model="a 122B-A10B MoE",
        hardware="three mixed cards, offload x uneven TP x uneven DCP",
        fraction=None,
        showed=(
            "the end state of the offload line: a model far past the rig's "
            "VRAM served at about 5-7 tok/s aggregate, which beat running it "
            "on one card. That is the price, stated plainly -- orders of "
            "magnitude below the VRAM-resident roofline, because the "
            "offloaded active experts are fetched over PCIe instead of read "
            "from the card's own memory"
        ),
        does_not_transfer=(
            "a much larger model with a much smaller active fraction. It "
            "establishes the ORDER of the cost, not its value for a 27B or a "
            "35B checkpoint"
        ),
    ),
)


@dataclasses.dataclass
class OffloadSources:
    """What the caller already worked out from the plan.

    ``offloadable_mib`` is the plan's own routed-expert pool -- the only
    host-offloadable weight class -- so the arithmetic here never re-derives
    a byte count the cost model already owns.
    """

    is_moe: bool = False
    is_gguf: bool = False
    tp_size: int = 1
    offloadable_mib: Optional[float] = None
    per_rank_offloadable_mib: List[float] = dataclasses.field(default_factory=list)
    current_fraction: Optional[float] = None
    fits_without_offload: Optional[bool] = None
    #: How far over the budget the plan is when it does not fit, in MiB.
    shortfall_mib: Optional[float] = None
    kv_bytes_per_token: Optional[float] = None
    host_ram_free_mib: Optional[float] = None
    cuda_graph_allowed: bool = False
    #: The register row that blocks offload for this checkpoint kind, if any.
    blocked_verdict: str = ""
    blocked_evidence: str = ""


def _cell(value, provenance, basis, unit="") -> dict:
    if provenance == ABSENT:
        value = None
    return {
        "value": value,
        "available": value is not None,
        "provenance": provenance,
        "basis": basis,
        "unit": unit,
        "study": None,
    }


def _freed_mib(src: OffloadSources, fraction: float) -> Optional[float]:
    """VRAM released at this resident fraction.

    ``(1 - fraction)`` of the routed-expert pool. The engine keeps
    ``ceil(fraction x count)`` experts per rank, so the real figure is at most
    one expert per rank BELOW this -- stated in the basis rather than folded
    in, because a correction the reader cannot see is a correction they
    cannot check.
    """
    if src.offloadable_mib is None:
        return None
    return max(0.0, float(src.offloadable_mib) * (1.0 - float(fraction)))


def _kv_tokens(src: OffloadSources, freed_mib: Optional[float]) -> Optional[float]:
    if not freed_mib or not src.kv_bytes_per_token:
        return None
    return freed_mib * _MIB / float(src.kv_bytes_per_token)


def _worth_for(src: OffloadSources, fraction: float, freed: Optional[float]) -> str:
    """The condition under which this step is the right trade.

    Three cases, and they are genuinely different advice. When the plan does
    not fit, offload is not a trade at all -- it is the only way to run the
    model, and the question is only how little of it is needed. When the plan
    fits, offload buys context at a decode price nobody has measured for this
    model, and that is a much weaker case.
    """
    if fraction >= 1.0:
        return (
            "the reference arm: nothing is offloaded and nothing crosses "
            "PCIe. Every other step is priced against this one."
        )
    fits = src.fits_without_offload
    if fits is False:
        if freed is not None and src.shortfall_mib:
            if freed >= float(src.shortfall_mib):
                return (
                    f"this step alone closes the {src.shortfall_mib:,.0f} MiB "
                    "gap. Worth it for anyone who wants to run this model at "
                    "all on these cards -- the alternative is not a slower "
                    "server, it is no server."
                )
            return (
                f"not enough on its own: it releases {freed:,.0f} MiB against "
                f"a {src.shortfall_mib:,.0f} MiB shortfall. Go deeper, or "
                "change the card set."
            )
        return (
            "the plan does not fit as it stands, so some depth is required "
            "rather than chosen."
        )
    return (
        "the plan already fits, so this step is elective: it trades decode "
        "throughput for context and concurrency. Worth it when long context "
        "or more sessions is the thing you actually want, and not worth it "
        "for a configuration that is already large enough -- the decode "
        "price is real and has not been measured for this model."
    )


def _step(src: OffloadSources, fraction: float) -> dict:
    freed = _freed_mib(src, fraction)
    kv = _kv_tokens(src, freed)
    resident_pct = int(round(fraction * 100))
    label = (
        "everything resident (offload off)"
        if fraction >= 1.0
        else (
            "nothing resident: the whole routed stack in host RAM"
            if fraction <= 0.0
            else f"{resident_pct} % of the routed experts resident"
        )
    )
    host_fits = None
    if freed is not None and src.host_ram_free_mib is not None:
        host_fits = freed <= float(src.host_ram_free_mib)
    notes: List[str] = []
    if fraction < 1.0:
        if not src.cuda_graph_allowed:
            notes.append(
                "Needs --disable-cuda-graph unless SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1. "
                "Losing the decode graphs is its own cost and is not part of "
                "the PCIe term above."
            )
        if host_fits is False:
            notes.append(
                f"The pinned host pool would need {freed:,.0f} MiB and only "
                f"{float(src.host_ram_free_mib):,.0f} MiB of host RAM is free. "
                "The pool is pinned, so this is a hard wall, not swapping."
            )
    return {
        "fraction": fraction,
        "label": label,
        "vram_freed_mib": _cell(
            freed,
            ESTIMATE if freed is not None else ABSENT,
            (
                "(1 - fraction) of the plan's routed-expert pool "
                f"({float(src.offloadable_mib):,.0f} MiB across "
                f"{src.tp_size} rank(s)). The engine keeps "
                "ceil(fraction x count) experts per rank, so the real figure "
                "is at most one expert per rank lower"
                if freed is not None
                else "the plan reports no offloadable routed-expert pool for "
                "this checkpoint"
            ),
            "MiB",
        ),
        "host_pool_mib": _cell(
            freed,
            ESTIMATE if freed is not None else ABSENT,
            "the same bytes, on the other side: what the pinned host pool has "
            "to hold. Pinned means it is not swappable",
            "MiB",
        ),
        "kv_tokens_gained": _cell(
            kv,
            ESTIMATE if kv is not None else ABSENT,
            (
                "released VRAM divided by the plan's KV bytes per token. It "
                "is what the freed budget could become if it is given to KV; "
                "the working point decides whether it is"
                if kv is not None
                else "needs the plan's KV bytes per token, which the balance "
                "report supplies once the plan resolves"
            ),
            "tokens",
        ),
        "decode_price": _cell(
            None,
            ABSENT,
            "not measured for this model on this rig. The offloaded routed "
            "experts stop being read from the card's own memory and start "
            "being fetched over PCIe on demand, which is a change of one to "
            "two orders of magnitude in the bandwidth term. What that costs "
            "HERE needs one boot at this fraction; the three boots under "
            "'evidence' say what it cost elsewhere",
            "tok/s",
        ),
        "tradeoff": {
            "gain": (
                "nothing released, nothing crossing PCIe"
                if fraction >= 1.0
                else (
                    f"releases about {freed:,.0f} MiB of weight VRAM"
                    + (f" -- roughly {kv:,.0f} KV tokens" if kv else "")
                    if freed is not None
                    else "releases the tail of the routed-expert stack"
                )
            ),
            "price": (
                "none"
                if fraction >= 1.0
                else (
                    f"{int(round((1 - fraction) * 100))} % of the routed "
                    "experts are fetched over PCIe per token instead of read "
                    "from VRAM, plus the decode graphs unless the offload "
                    "graph gate is set. The size of that penalty for this "
                    "model is not measured"
                )
            ),
            "worth_for": _worth_for(src, fraction, freed),
        },
        "current": (
            src.current_fraction is not None
            and abs(float(src.current_fraction) - fraction) < 1e-9
        ),
        "notes": notes,
    }


def offload_dimension(src: OffloadSources) -> dict:
    """The offload depth axis: every step, what it buys, what it costs."""
    if not src.is_moe:
        return {
            "applies": False,
            "available": False,
            "reason": (
                "dense checkpoint: there are no routed experts to offload, so "
                "depth is not a dimension here"
            ),
            "steps": [],
            "evidence": [],
            "coverage": {"measured": 0, "estimate": 0, "absent": 0, "total": 0},
        }
    if src.blocked_verdict:
        return {
            "applies": True,
            "available": False,
            "reason": src.blocked_verdict,
            "source": src.blocked_evidence,
            "steps": [],
            "evidence": [e.to_json() for e in EVIDENCE],
            "coverage": {"measured": 0, "estimate": 0, "absent": 0, "total": 0},
        }

    steps = [_step(src, f) for f in OFFLOAD_STEPS]
    counts = {ESTIMATE: 0, ABSENT: 0}
    for s in steps:
        for key in ("vram_freed_mib", "kv_tokens_gained", "decode_price"):
            prov = s[key]["provenance"]
            counts[prov] = counts.get(prov, 0) + 1
    return {
        "applies": True,
        "available": True,
        "reason": "",
        "current_fraction": src.current_fraction,
        "steps": steps,
        "evidence": [e.to_json() for e in EVIDENCE],
        "counter_reckoning": {
            "buy_side": (
                "computed. Every MiB in the columns above comes from the "
                "plan's own routed-expert pool divided by the step, so it "
                "moves with the checkpoint and the card set."
            ),
            "price_side": (
                "not computed and not borrowed. The decode cost of a given "
                "depth is a measurement, and no boot of THIS model at THIS "
                "fraction exists. Reading the 5-7 tok/s of the 122B boot as "
                "this model's number would be the exact mistake the "
                "provenance labels are here to make impossible."
            ),
            "what_would_settle_it": (
                "one boot at the chosen fraction with the decode window the "
                "split probe already runs. Until then the depth axis states "
                "what it frees and refuses to state what it costs."
            ),
            "second_order": (
                "Two effects ride along and are easy to forget: the decode "
                "CUDA graphs unless the offload graph gate is set, and the "
                "load time -- staging the pool took 2 min 36 s on the AWQ "
                "boot. Neither is in the PCIe term."
            ),
        },
        "wave_order_note": (
            "Ordering the offload waves by expert rather than by token cuts "
            "the per-layer H2D traffic about 3.5x on the boot that measured "
            "it, with byte-identical output. It costs one transient buffer "
            "per layer. If depth is chosen at all, this lever is chosen with "
            "it."
        ),
        "coverage": {
            "measured": 0,
            "estimate": counts.get(ESTIMATE, 0),
            "absent": counts.get(ABSENT, 0),
            "total": counts.get(ESTIMATE, 0) + counts.get(ABSENT, 0),
            "summary": (
                f"{counts.get(ESTIMATE, 0)} derived cells (what each depth "
                f"frees), {counts.get(ABSENT, 0)} absent (what each depth "
                "costs). Nothing on this axis is measured for this model."
            ),
        },
    }
