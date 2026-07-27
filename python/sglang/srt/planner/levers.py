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
"""Levers: one flag combination per direction, with its counter-price.

DESIGN_216 names five directions, and DESIGN_210 explains why they are
separately reachable at all: the fork has several knobs on different axes
(weight shards, MLP units, expert dimension, lm_head width, token ownership),
not one compromise slider.

    lever            maximises                 typical price
    context          max_total_num_tokens      slowest rank carries most tokens
    decode_speed     tok/s at bs=1             less context
    prefill_speed    prompt throughput         decode split off-optimum
    ttft_loaded      time to first token       needs a satellite and a handover
    energy           joules per token          peak throughput drops

"Lever", not "profile": on this fork ``profile`` already names a saved launch
configuration (``planner.flags``) and a catalog card
(``planner.card_library``). A direction that maximises one quantity at a named
price is a third thing, and calling all three the same word in one surface is
how a control panel becomes unreadable.

Two rules keep this from becoming a wish list.

**Staged by evidence.** Before a probe the planner knows STRUCTURE — which
cards, how much VRAM, what fits. After a probe it knows RATES — how fast each
side computes, reads and communicates. Fit questions need no probe; speed
questions do. So an unprobed rig gets a deliberately vague suggestion, and the
detailed one unlocks only once measurements exist. The staging is not a UI
nicety; it is a statement about what is answerable.

The prefill lever adds a third stage, because rates alone do not settle it. A
probe says which rank is compute-strong; it does not say what moving weight
onto that rank costs in decode, and that term is half the decision. Both terms
depend on the card mix, the interconnect, the model and which quantisation
lane each rank runs, so they have to be measured on the rig they are claimed
for (:mod:`sglang.srt.planner.crossover`). Until they are, the lever proposes
nothing and offers the measurement instead — a guessed turning point is worse
than a missing one, and this lever previously handed out a concentration
vector on the strength of a card probe alone.

**Measured and modelled are never printed the same way.** Every figure a lever
shows carries an :class:`Evidence` kind. A measured one names its setup; a
modelled one names its known error. The cost model behind this feature is
calibrated on its decode side and not on its prefill side, so a modelled NET
of the two is not shown at all.

**Gated by the build.** A lever that emits a flag this build does not have is
worse than no lever at all. Every flag a lever wants is checked against
``ServerArgs`` (:func:`flag_available`), and a lever whose central knob is
missing says so and names the branch instead of quietly emitting a command
that will fail to parse. The gate reads the running build, so a lever's state
follows the branch it is executed on rather than a claim written here.

The decode lever is the worked example of why the gate is dynamic. DESIGN_210
records that the WEIGHT split is a near-no-op for decode; the decode lever
sits in the KV-TOKEN split, because under DCP each rank runs attention over
the tokens it owns and at bs=1 the slowest rank sets the pace. That knob —
``--rank-kv-ratio speed`` — landed with #210: measured on 27B FP8, TP=3 uneven
DCP, 120 k resident tokens, bs=1, moving the ownership vector from [2,3,3] to
[2,1,1] cut the context-dependent part of the decode step by 24.5 % against a
boot-to-boot noise floor of 1.07 %.

Counter-reckoning is mandatory: :class:`Lever` has no field for the gain
without a matching field for the cost, so "in all directions" is structural
rather than a habit.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence

from sglang.srt.planner.crossover import (
    EVIDENCE_KINDS,
    MEASURED,
    MODELLED,
    MODELLED_DECODE_NOTE,
    MODELLED_NET_REFUSED,
    MODELLED_PREFILL_NOTE,
    STUDY_TIERS,
    CrossoverFinding,
)

__all__ = [
    "Confidence",
    "Evidence",
    "FlagSpec",
    "Lever",
    "LeverSuggestion",
    "LEVERS",
    "flag_available",
    "missing_flags",
    "suggest_levers",
    "render_levers_text",
    "STRUCTURE_ONLY",
    "RATES_KNOWN",
]

#: Evidence stages.
STRUCTURE_ONLY = "structure"  # no probe: fit questions only
RATES_KNOWN = "rates"  # probe present: speed questions become answerable


class Confidence:
    VAGUE = "vague"
    DETAILED = "detailed"
    BLOCKED = "blocked"
    #: The direction is real and the knob exists, but its SIZE has not been
    #: measured on this rig and cannot be derived from what has. Distinct from
    #: VAGUE, which is "no probe yet", and from BLOCKED, which is "this build
    #: cannot do it at all".
    UNMEASURED = "unmeasured"


@dataclasses.dataclass(frozen=True)
class Evidence:
    """One figure a lever leans on, and where it came from.

    ``kind`` is the whole point. A measured figure has to name the setup it
    was taken on, because a number without its conditions travels further than
    it should; a modelled figure has to name the error it is known to carry,
    because otherwise a reader treats a prediction as a result.
    """

    kind: str
    statement: str
    #: For measured figures: rig, model, method. Required.
    setup: str = ""
    #: For modelled figures: the known bias or the missing calibration.
    #: Required.
    caveat: str = ""

    def __post_init__(self):
        if self.kind not in EVIDENCE_KINDS:
            raise ValueError(
                f"unknown evidence kind {self.kind!r}; one of "
                f"{', '.join(EVIDENCE_KINDS)}"
            )

    def render(self) -> str:
        tail = f" ({self.setup})" if self.setup else ""
        if self.caveat:
            tail += f" — {self.caveat}"
        return f"[{self.kind}] {self.statement}{tail}"

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class FlagSpec:
    """One flag a lever wants to set."""

    flag: str
    value: Optional[str] = None
    #: ``ServerArgs`` field this maps to, for the availability gate.
    field: Optional[str] = None
    why: str = ""
    #: Only emit when the rig actually has non-uniform cards.
    heterogeneous_only: bool = False
    #: Requires measured per-card rates (i.e. a probe).
    needs_probe: bool = False
    #: Requires a crossover measured on THIS rig plus a workload mix that
    #: clears it. Cards alone do not answer the question this flag settles.
    needs_workload: bool = False

    def render(self) -> str:
        return f"{self.flag} {self.value}" if self.value is not None else self.flag


@dataclasses.dataclass
class Lever:
    """A direction, its flags, and what it costs.

    ``gains`` and ``costs`` are both required and both non-empty; a lever
    that only lists what it improves is a sales pitch.
    """

    key: str
    label: str
    maximises: str
    flags: List[FlagSpec]
    gains: List[str]
    costs: List[str]
    #: What the OTHER levers lose if this one is chosen — the "in all
    #: directions" requirement, made explicit per opposing lever.
    tradeoff_against: Dict[str, str] = dataclasses.field(default_factory=dict)
    #: Shown before any probe exists: honest, unquantified.
    vague_statement: str = ""
    #: Preconditions beyond flags (topology, a second node, ...).
    preconditions: List[str] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)
    #: Every figure this lever leans on, labelled measured or modelled.
    evidence: List[Evidence] = dataclasses.field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "maximises": self.maximises,
            "flags": [dataclasses.asdict(f) for f in self.flags],
            "gains": list(self.gains),
            "costs": list(self.costs),
            "tradeoff_against": dict(self.tradeoff_against),
            "vague_statement": self.vague_statement,
            "preconditions": list(self.preconditions),
            "notes": list(self.notes),
            "evidence": [e.to_json() for e in self.evidence],
        }


# ---------------------------------------------------------------------------
# Build gate
# ---------------------------------------------------------------------------


def flag_available(field: str) -> bool:
    """Does this build's ``ServerArgs`` carry that field?"""
    try:
        import dataclasses as _dc

        from sglang.srt.server_args import ServerArgs

        return field in {f.name for f in _dc.fields(ServerArgs)}
    except Exception:
        return False


def missing_flags(lever: Lever) -> List[FlagSpec]:
    return [f for f in lever.flags if f.field and not flag_available(f.field)]


# ---------------------------------------------------------------------------
# The five levers
# ---------------------------------------------------------------------------

LEVERS: Dict[str, Lever] = {}


def _register(p: Lever) -> Lever:
    LEVERS[p.key] = p
    return p


_register(
    Lever(
        key="context",
        label="Context",
        maximises="max_total_num_tokens",
        flags=[
            FlagSpec(
                "--rank-tp-ratio",
                "auto",
                "rank_tp_ratio",
                why="Weight shards proportional to each card's memory budget, "
                "so no card is the odd one out that caps the min-synced pool.",
                heterogeneous_only=True,
            ),
            FlagSpec(
                "--rank-mlp-ratio",
                "auto",
                "rank_mlp_ratio",
                why="Dense-MLP units are the shiftable mass; rebalancing them "
                "grows the min-synced KV pool without touching attention.",
                heterogeneous_only=True,
            ),
            FlagSpec(
                "--rank-kv-ratio",
                "capacity",
                "rank_kv_ratio",
                why="Capacity-weighted KV-token ownership: after the "
                "post-weight-load profiling each rank installs the vector "
                "proportional to its measured free-VRAM-after-weights "
                "capacity, which is what maximises max_total_num_tokens.",
                heterogeneous_only=True,
            ),
            FlagSpec(
                "--rank-perf-loose-ctx-percent",
                "0",
                "rank_perf_loose_ctx_percent",
                why="Zero tolerance: no context may be traded for speed in "
                "this lever.",
            ),
        ],
        gains=[
            "Largest reachable max_total_num_tokens for this model on this rig.",
            "Capacity-directed placement keeps the weakest card from capping "
            "the whole pool.",
        ],
        costs=[
            "Capacity balancing hands the weakest card the largest token "
            "share. Under DCP each rank computes attention over its own "
            "tokens, so the slowest card acquires the most attention work and "
            "the group waits on it in lockstep.",
            "Deep-context decode is where that cost shows up, not short "
            "prompts.",
        ],
        tradeoff_against={
            "decode_speed": "Directly opposed: capacity balancing and speed "
            "balancing pull the token split in opposite directions. This is "
            "the sharpest conflict on the rig.",
            "prefill_speed": "Mild: capacity balancing moves weight mass away "
            "from the compute-strong card.",
        },
        vague_statement=(
            "This rig can hold roughly the context the VRAM sum allows once "
            "weights and pools are subtracted. How much the capacity split "
            "costs in decode speed cannot be stated without a probe."
        ),
        notes=[
            "Solo-draft placement makes this worse: the draft occupies VRAM on "
            "its host card, pushing the capacity-optimal token split further "
            "away from it — the same direction that hurts decode.",
        ],
    )
)

_register(
    Lever(
        key="decode_speed",
        label="Decode speed",
        maximises="tok/s at batch size 1",
        flags=[
            FlagSpec(
                "--rank-vocab-ratio",
                "auto",
                "rank_vocab_ratio",
                why="The lm_head matvec streams the whole shard every decode "
                "step, so an even split is bounded by the slowest card. "
                "Weighting shard width by measured memory bandwidth balances "
                "the read TIME instead.",
                heterogeneous_only=True,
                needs_probe=True,
            ),
            FlagSpec(
                "--rank-kv-ratio",
                "speed",
                "rank_kv_ratio",
                why="Token ownership shifted from the capacity proportion "
                "toward the memory-bandwidth proportion. This is where the "
                "decode lever actually sits: under DCP a rank runs attention "
                "over the tokens it owns, so ownership IS the decode work "
                "split.",
                heterogeneous_only=True,
                needs_probe=True,
            ),
            FlagSpec(
                "--rank-tp-ratio",
                "auto-performance",
                "rank_tp_ratio",
                why="--rank-kv-ratio speed reads the per-rank bandwidth "
                "scores, which only the auto-performance plan resolves. "
                "Without it the mode degrades to 'capacity' — the opposite "
                "direction.",
                heterogeneous_only=True,
                needs_probe=True,
            ),
            FlagSpec(
                "--rank-perf-loose-ctx-percent",
                "10",
                "rank_perf_loose_ctx_percent",
                why="Bounds how much context may be given up for speed.",
            ),
        ],
        gains=[
            "Balances the per-step read time across unequal cards instead of "
            "letting the slowest one set it.",
            "Measured (#210): 27B FP8, TP=3 uneven DCP, 120 k resident tokens, "
            "bs=1, no speculation — moving the ownership vector from [2,3,3] "
            "to [2,1,1] cut the context-dependent part of the decode step by "
            "24.5 % (2.296 -> 1.732 ms) against a 1.07 % boot-to-boot noise "
            "floor. End to end that was -2.5 % step time, and it grows "
            "linearly with resident context.",
        ],
        costs=[
            "Less context: the speed-directed token split holds fewer tokens "
            "on the high-capacity cards.",
            "The gain is bounded by the collectives and by the slowest rank; "
            "at bs=1 those dominate.",
        ],
        tradeoff_against={
            "context": "Directly opposed — see the context lever.",
        },
        vague_statement=(
            "Decode speed on a heterogeneous rig is set by the slowest rank "
            "and by the collectives. Which split helps cannot be derived from "
            "VRAM sizes alone; it needs measured bandwidths."
        ),
        notes=[
            "The measured decode optimum is FLAT across representable splits, "
            "so expect a broad optimum rather than a sharp one — and that "
            "measurement was taken without pinned clocks, so part of the "
            "flatness may be P-state artefact.",
            "Speculation raises the arithmetic intensity of decode: at high "
            "accept length the decode optimum drifts toward the compute "
            "proportion, i.e. toward the prefill lever.",
        ],
        evidence=[
            Evidence(
                kind=MEASURED,
                statement=(
                    "Moving the KV ownership vector from [2,3,3] to [2,1,1] "
                    "cut the context-dependent part of the decode step by "
                    "24.5 % (2.296 -> 1.732 ms), -2.5 % end to end, growing "
                    "linearly with resident context."
                ),
                setup=(
                    "27B FP8, TP=3 uneven DCP, 120 k resident tokens, bs=1, no "
                    "speculation, against a 1.07 % boot-to-boot noise floor"
                ),
            ),
        ],
    )
)

_register(
    Lever(
        key="prefill_speed",
        label="Prefill speed",
        maximises="prompt throughput",
        flags=[
            FlagSpec(
                "--rank-mlp-ratio",
                None,  # filled from the measured crossover, or not at all
                "rank_mlp_ratio",
                why="Dense-MLP units concentrated on the compute-strong rank. "
                "The vector is not a constant: it comes from this rig's "
                "measured crossover and from the workload's prompt:output "
                "mix, and no vector is emitted without both.",
                heterogeneous_only=True,
                needs_probe=True,
                needs_workload=True,
            ),
            FlagSpec(
                "--rank-perf-loose-ctx-percent",
                "15",
                "rank_perf_loose_ctx_percent",
                why="Bounds how much context the concentration may spend. "
                "Gated exactly like the vector it bounds: a context budget "
                "emitted without a concentration to bound is a flag that says "
                "nothing.",
                heterogeneous_only=True,
                needs_probe=True,
                needs_workload=True,
            ),
        ],
        gains=[
            "Concentrates MLP mass on the compute-strong card, which is where "
            "prefill time is spent.",
            "The gain is a change of the prefill SLOPE, so it applies to every "
            "prompt token — but for the same reason it saturates as a "
            "percentage and does not keep growing with prompt length.",
        ],
        costs=[
            "Decode pays per output token: the concentrated rank streams more "
            "weight bytes every step and paces the group in lockstep.",
            "Context is spent: concentrating mass on one card lowers the "
            "min-synced pool.",
            "The trade only turns positive above a prompt:output ratio that is "
            "specific to the rig, the model and the quantisation path, so it "
            "is not a general default.",
        ],
        tradeoff_against={
            "context": "Concentration reduces the min-synced KV pool.",
            "decode_speed": "Directly opposed, per output token. One static "
            "split cannot hold both optima, and chunked prefill mixes the "
            "phases in one batch, so the operating point is a weighted mean "
            "rather than either optimum.",
        },
        vague_statement=(
            "Prefill favours the compute-strong card more strongly than decode "
            "does. By how much requires measured compute throughput per card."
        ),
        notes=[
            "The vector is set directly rather than through "
            "--rank-perf-tune. The optimiser's decode-knee guard rejects "
            "concentration on the decode evidence alone, which is correct for "
            "a decode-balanced objective; a prompt-dominated workload is the "
            "evidence that overrides it, and that evidence lives outside the "
            "optimiser.",
        ],
        evidence=[
            Evidence(
                kind=MODELLED,
                statement=(
                    "The optimiser predicts a prefill gain per candidate "
                    "vector at parse time."
                ),
                caveat=MODELLED_PREFILL_NOTE,
            ),
            Evidence(
                kind=MODELLED,
                statement=(
                    "The optimiser predicts a decode cost per candidate "
                    "vector at parse time."
                ),
                caveat=MODELLED_DECODE_NOTE + " " + MODELLED_NET_REFUSED,
            ),
        ],
    )
)

_register(
    Lever(
        key="ttft_loaded",
        label="TTFT under load",
        maximises="time to first token while the hub is busy",
        flags=[
            FlagSpec(
                "--dcp-size",
                None,
                "dcp_size",
                why="Keep the hub cut for decode; the elasticity sits outside "
                "it, not in this split.",
            ),
        ],
        gains=[
            "Under load TTFT is dominated by queueing, not by prefill "
            "duration — a slower free machine beats a faster busy one.",
            "Lets the hub stay decode-optimal permanently instead of taking a "
            "compromise split, because the prefill elasticity lives outside "
            "it.",
        ],
        costs=[
            "Needs a second node and a handover path; without one this lever "
            "has nothing to offer.",
            "Below roughly 2 300 tokens the length-independent recurrent-state "
            "handover outweighs the saved prefill, so short prompts lose.",
            "Both sides must run comparable weights: a handover that changes "
            "quantisation mid-prompt puts a quality seam inside one sequence.",
        ],
        tradeoff_against={
            "decode_speed": "None on the hub — that is the point of putting "
            "the elasticity outside it.",
        },
        preconditions=[
            "A second node joined to this aggregator.",
            "A transport between the nodes, chosen from the measured pair "
            "matrix rather than from configuration.",
            "Prefill offload support in the running build.",
        ],
        vague_statement=(
            "Whether offloading prefill helps depends on how long requests "
            "actually queue on the hub. Measure the queue first: if it is "
            "small, no transport is cheap enough to win."
        ),
        notes=[
            "Falsify before building: measure hub queueing time under "
            "realistic mixed load. If it stays far below prefill duration, the "
            "premise does not hold.",
        ],
    )
)

_register(
    Lever(
        key="energy",
        label="Energy per token",
        maximises="joules per token (lower is better)",
        flags=[
            FlagSpec(
                "--rank-vocab-ratio",
                "auto",
                "rank_vocab_ratio",
                why="Less waiting means less idle draw across the group; "
                "balancing read time is also an efficiency lever.",
                heterogeneous_only=True,
                needs_probe=True,
            ),
        ],
        gains=[
            "Power capping typically costs throughput sub-linearly, so there "
            "is usually a knee well below the stock limit.",
            "Cutting wait time cuts energy directly: a waiting rank still "
            "draws power.",
        ],
        costs=[
            "Peak throughput drops. This lever trades the top of the range "
            "for the middle of it.",
        ],
        tradeoff_against={
            "decode_speed": "Direct: the power cap that improves J/token "
            "lowers peak tok/s.",
            "prefill_speed": "Direct, and more strongly — prefill is the "
            "compute-heavy phase that the cap bites into first.",
        },
        preconditions=[
            "Power-target control, which needs the collector on the host: an "
            "LXC guest cannot reach the driver's control path regardless of "
            "privilege.",
        ],
        vague_statement=(
            "Energy per token can be improved by capping power, but where the "
            "knee sits is rig-specific and needs a sweep."
        ),
        notes=[
            "Measure with the NVML total-energy counter differenced over the "
            "window, not by integrating power samples — a counter cannot miss "
            "a transient between two samples.",
            "Attribute per card, not as a group sum: power.draw is not "
            "strictly comparable across card generations.",
        ],
    )
)


# ---------------------------------------------------------------------------
# Suggestion
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LeverSuggestion:
    """A lever resolved against this rig, this build and the evidence at hand."""

    lever: Lever
    stage: str
    confidence: str
    statement: str
    command_flags: List[str]
    unavailable_flags: List[Dict[str, str]]
    unmet_preconditions: List[str]
    counter_reckoning: Dict[str, str]
    #: Every figure behind this suggestion, labelled measured or modelled.
    evidence: List[Evidence] = dataclasses.field(default_factory=list)
    #: The crossover resolution: what is known, what is offered, what was
    #: chosen. Empty for levers that do not depend on it.
    crossover: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "key": self.lever.key,
            "label": self.lever.label,
            "maximises": self.lever.maximises,
            "stage": self.stage,
            "confidence": self.confidence,
            "statement": self.statement,
            "command_flags": list(self.command_flags),
            "unavailable_flags": list(self.unavailable_flags),
            "unmet_preconditions": list(self.unmet_preconditions),
            "gains": list(self.lever.gains),
            "costs": list(self.lever.costs),
            "counter_reckoning": dict(self.counter_reckoning),
            "notes": list(self.lever.notes),
            "evidence": [e.to_json() for e in self.evidence],
            "crossover": dict(self.crossover),
        }


# ---------------------------------------------------------------------------
# The crossover, resolved against this rig and this workload
# ---------------------------------------------------------------------------


def _pct(value: Optional[float]) -> str:
    return "unmeasured" if value is None else f"{value:+.1f} %"


def _resolve_crossover(
    finding: Optional[CrossoverFinding],
    prompt_to_output_ratio: Optional[float],
) -> Dict[str, Any]:
    """What may be said about MLP concentration right now.

    Returns a dict with ``usable`` (may a vector be proposed at all),
    ``chosen`` (the vector, or None), ``table`` (every candidate with its two
    slopes and its break-even) and ``offer`` (the study tiers, when there is
    nothing to go on). The three refusals are separate on purpose: no finding,
    a finding from elsewhere, and a workload that does not clear the turn are
    different situations and read differently.
    """
    tiers = [t.to_json() for t in STUDY_TIERS]
    if finding is None:
        return {
            "state": "unmeasured",
            "usable": False,
            "chosen": None,
            "offer": tiers,
            "reason": (
                "This rig has no measured crossover. What the concentration "
                "buys in prefill and costs in decode both depend on the card "
                "mix, the model and the quantisation lane each rank runs, so "
                "neither can be derived from the hardware probe."
            ),
        }
    caveats = finding.caveats()
    if not finding.usable_for_advice():
        return {
            "state": "not_usable",
            "usable": False,
            "chosen": None,
            "rig": finding.rig.label(),
            "provenance": finding.provenance,
            "caveats": caveats,
            "table": finding.break_even_table(),
            "offer": tiers,
            "reason": caveats[0] if caveats else "the finding cannot be used",
        }
    table = finding.break_even_table()
    proposable = [
        r for r in table
        if r["proposable"] and r["break_even_prompt_to_output"] is not None
    ]
    cheapest = min(
        (r["break_even_prompt_to_output"] for r in proposable), default=None
    )
    out: Dict[str, Any] = {
        "state": "measured_here",
        "usable": True,
        "chosen": None,
        "rig": finding.rig.label(),
        "age_days": round(finding.age_s() / 86400.0, 1),
        "caveats": caveats,
        "table": table,
        "cheapest_break_even": cheapest,
        "dropped": [
            {"vector": v, "reason": why} for v, why in finding.pruned()
        ],
        "offer": [],
    }
    if prompt_to_output_ratio is None:
        out["reason"] = (
            "No workload mix given. The turn is known; which side of it this "
            "server runs on is not."
        )
        return out
    out["prompt_to_output_ratio"] = float(prompt_to_output_ratio)
    best = finding.best_for_ratio(float(prompt_to_output_ratio))
    if best is None:
        out["reason"] = (
            f"At {float(prompt_to_output_ratio):.1f}:1 prompt to output every "
            "candidate is a net loss"
            + (f"; the cheapest turns at {cheapest:.1f}:1." if cheapest else ".")
        )
        return out
    out["chosen"] = {
        "vector": best.label(),
        "break_even": best.break_even_ratio,
        "net_ms_per_output_token": best.net_ms_per_output_token(
            float(prompt_to_output_ratio)
        ),
        "prefill_gain_pct": best.prefill_gain_pct,
        "decode_cost_pct": best.decode_cost_pct,
    }
    return out


def _crossover_evidence(res: Dict[str, Any]) -> List[Evidence]:
    """The measured half of the prefill lever's evidence, if it exists."""
    if not res or res.get("state") not in ("measured_here", "not_usable"):
        return []
    rig = res.get("rig", "unknown rig")
    elsewhere = res.get("provenance") == "measured_elsewhere"
    setup = (
        f"another rig: {rig}" if elsewhere else f"{rig}, measured {res.get('age_days')} days ago"
    )
    out = []
    for row in res.get("table") or []:
        be = row["break_even_prompt_to_output"]
        out.append(
            Evidence(
                kind=MEASURED,
                statement=(
                    f"{row['vector']}: prefill {_pct(row['prefill_gain_pct'])}, "
                    f"decode {_pct(row['decode_cost_pct'])}, break-even "
                    + (f"{be:.1f}:1 prompt to output" if be is not None
                       else "never")
                    + ("" if row["proposable"] else " — not proposable, "
                       + row["dropped_because"])
                ),
                setup=setup,
            )
        )
    for c in res.get("caveats") or []:
        out.append(Evidence(kind=MEASURED, statement=c, setup=setup))
    return out


def _counter_reckoning(
    lever: Lever, res: Dict[str, Any]
) -> Dict[str, str]:
    """The per-opposing-lever price, with this rig's numbers where they exist.

    The prose in ``tradeoff_against`` says which direction the conflict runs.
    What it costs is a measurement, and stays absent — explicitly — until one
    exists for this rig.
    """
    out = dict(lever.tradeoff_against)
    if lever.key not in ("prefill_speed", "decode_speed"):
        return out
    rows = [
        r for r in (res.get("table") or [])
        if r["proposable"] and r["break_even_prompt_to_output"] is not None
    ]
    usable = res.get("usable") and rows
    if lever.key == "prefill_speed":
        if usable:
            span = "; ".join(
                f"{r['vector']} costs {_pct(r['decode_cost_pct'])} decode for "
                f"{_pct(r['prefill_gain_pct'])} prefill, turning at "
                f"{r['break_even_prompt_to_output']:.1f}:1"
                for r in rows
            )
            out["decode_speed"] = (
                out.get("decode_speed", "") + " Measured here: " + span + "."
            )
        else:
            out["decode_speed"] = (
                out.get("decode_speed", "")
                + " How much it costs is not measured on this rig; run the "
                "crossover study before spending decode for prefill."
            )
    else:
        if usable:
            best = max(rows, key=lambda r: r["prefill_gain_pct"] or 0.0)
            out["prefill_speed"] = (
                "Staying at the base split forgoes up to "
                f"{_pct(best['prefill_gain_pct'])} prefill "
                f"({best['vector']}), which is only worth taking above "
                f"{best['break_even_prompt_to_output']:.1f}:1 prompt to output."
            )
        else:
            out["prefill_speed"] = (
                "What the base split forgoes in prefill is not measured on "
                "this rig; run the crossover study to put a number on it."
            )
    return out


def suggest_levers(
    heterogeneous: bool = True,
    probe: Optional[dict] = None,
    node_count: int = 1,
    facility_keys_available: Optional[Sequence[str]] = None,
    keys: Optional[Sequence[str]] = None,
    crossover: Optional[CrossoverFinding] = None,
    prompt_to_output_ratio: Optional[float] = None,
) -> List[LeverSuggestion]:
    """Resolve every lever against the evidence available right now.

    ``probe`` is a cached hardware-probe dict (or None). Its presence is what
    moves a lever from a vague statement to a concrete flag set — the
    structure/rates line from DESIGN_216.

    ``crossover`` is this rig's measured MLP-split crossover (or None), and
    ``prompt_to_output_ratio`` the workload's expected prompt tokens per
    output token. The prefill lever needs both: a crossover measured
    elsewhere, or a workload mix with no crossover behind it, each leave it
    unresolved rather than approximately right.
    """
    stage = RATES_KNOWN if probe else STRUCTURE_ONLY
    have_facilities = set(facility_keys_available or [])
    resolution = _resolve_crossover(crossover, prompt_to_output_ratio)
    out: List[LeverSuggestion] = []
    for key, p in LEVERS.items():
        if keys and key not in keys:
            continue
        missing = missing_flags(p)
        unavailable = [
            {
                "flag": f.flag,
                "field": f.field or "",
                "reason": (
                    f"this build has no {f.flag}; the knob lives on a separate "
                    "feature branch"
                ),
                "why_it_mattered": f.why,
            }
            for f in missing
        ]
        chosen = (resolution.get("chosen") or {}).get("vector")
        flags: List[str] = []
        withheld_for_workload = False
        for f in p.flags:
            if f in missing:
                continue
            if f.heterogeneous_only and not heterogeneous:
                continue
            if f.needs_probe and not probe:
                continue
            if f.needs_workload:
                if not chosen:
                    withheld_for_workload = True
                    continue
                if f.value is None:
                    flags.append(f"{f.flag} {chosen}")
                    continue
            if f.value is None:
                continue
            flags.append(f.render())

        unmet = []
        for pre in p.preconditions:
            low = pre.lower()
            if "second node" in low and node_count < 2:
                unmet.append(pre)
            elif "power-target" in low and "power_target" not in have_facilities:
                unmet.append(pre)

        uses_crossover = any(f.needs_workload for f in p.flags)
        lever_res = resolution if uses_crossover else {}
        evidence = list(p.evidence)
        if uses_crossover:
            evidence = _crossover_evidence(resolution) + evidence

        if unavailable and not flags:
            confidence = Confidence.BLOCKED
            statement = (
                f"{p.label}: not settable on this build — "
                + "; ".join(u["reason"] for u in unavailable)
            )
        elif stage == STRUCTURE_ONLY:
            confidence = Confidence.VAGUE
            statement = p.vague_statement
        elif withheld_for_workload and _answered(resolution):
            # The crossover IS measured and the workload IS known; the answer
            # is simply that no concentration pays here. That is a resolved
            # lever with an empty flag set, not an unmeasured one.
            confidence = Confidence.DETAILED
            statement = _base_split_statement(p, resolution)
        elif withheld_for_workload:
            confidence = Confidence.UNMEASURED
            statement = _unmeasured_statement(p, resolution)
        else:
            confidence = Confidence.DETAILED
            statement = (
                f"{p.label} maximises {p.maximises}. "
                + (f"Set: {' '.join(flags)}. " if flags else "")
                + f"Price: {p.costs[0]}"
            )
            if uses_crossover and resolution.get("chosen"):
                ch = resolution["chosen"]
                statement += (
                    f" At {resolution['prompt_to_output_ratio']:.1f}:1 prompt "
                    f"to output, {ch['vector']} nets "
                    f"{ch['net_ms_per_output_token']:.3f} ms per output token "
                    f"over the base split; it turns at {ch['break_even']:.1f}:1."
                )
        out.append(
            LeverSuggestion(
                lever=p,
                stage=stage,
                confidence=confidence,
                statement=statement,
                command_flags=flags,
                unavailable_flags=unavailable,
                unmet_preconditions=unmet,
                counter_reckoning=_counter_reckoning(p, resolution),
                evidence=evidence,
                crossover=lever_res,
            )
        )
    return out


def _answered(res: Dict[str, Any]) -> bool:
    """Is the crossover question settled for this rig and this workload?"""
    return bool(res.get("usable")) and res.get("prompt_to_output_ratio") is not None


def _base_split_statement(lever: Lever, res: Dict[str, Any]) -> str:
    """The answer when the workload sits below every candidate's turn."""
    cheapest = res.get("cheapest_break_even")
    return (
        f"{lever.label}: at {res['prompt_to_output_ratio']:.1f}:1 prompt to "
        "output the base split is the faster configuration on this rig"
        + (f"; the cheapest candidate turns at {cheapest:.1f}:1" if cheapest
           else "")
        + f". Measured on {res['rig']}, {res['age_days']} days ago. "
        "No concentration flag is set."
    )


def _unmeasured_statement(lever: Lever, res: Dict[str, Any]) -> str:
    """What the prefill lever says when it declines to propose a vector."""
    head = (
        f"{lever.label}: concentrating MLP weight on the compute-strong rank "
        "buys prefill per prompt token and costs decode per output token. "
    )
    state = res.get("state")
    if state == "measured_here":
        rows = [
            r for r in res.get("table") or []
            if r["proposable"] and r["break_even_prompt_to_output"] is not None
        ]
        turns = ", ".join(
            f"{r['vector']} at {r['break_even_prompt_to_output']:.1f}:1"
            for r in rows
        )
        return (
            head
            + f"Measured on this rig ({res['rig']}, {res['age_days']} days "
            f"ago), the turn sits at: {turns}. "
            + str(res.get("reason", ""))
            + " Pass the expected prompt:output mix to get a vector, or none."
        )
    tiers = ", ".join(
        f"{t['label']} ~{t['est_runtime_min']} min" for t in res.get("offer") or []
    )
    return (
        head
        + str(res.get("reason", ""))
        + (f" Measure it: {tiers}." if tiers else "")
        + " No vector is proposed until then; the base split stands."
    )


def render_levers_text(suggestions: Sequence[LeverSuggestion]) -> str:
    lines = []
    if suggestions:
        stage = suggestions[0].stage
        lines.append(
            "Evidence: structure only (no probe) — fit questions are "
            "answerable, speed questions are not."
            if stage == STRUCTURE_ONLY
            else "Evidence: measured rates available — speed questions are "
            "answerable."
        )
        lines.append("")
    for s in suggestions:
        lines.append(f"[{s.confidence}] {s.lever.label} -> {s.lever.maximises}")
        lines.append(f"  {s.statement}")
        if s.command_flags:
            lines.append(f"  flags: {' '.join(s.command_flags)}")
        for u in s.unavailable_flags:
            lines.append(f"  unavailable: {u['flag']} — {u['reason']}")
        for pre in s.unmet_preconditions:
            lines.append(f"  precondition unmet: {pre}")
        for e in s.evidence:
            lines.append(f"  {e.render()}")
        for k, v in s.counter_reckoning.items():
            lines.append(f"  costs {k}: {v}")
        for t in s.crossover.get("offer") or []:
            lines.append(
                f"  measure it: {t['label']} ~{t['est_runtime_min']} min "
                f"({t['study_file']}) — {t['what']}"
            )
        lines.append("")
    return "\n".join(lines)
