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
"""Lever profiles: one flag combination per direction, with its counter-price.

DESIGN_216 names five directions, and DESIGN_210 explains why they are
separately reachable at all: the fork has several knobs on different axes
(weight shards, MLP units, expert dimension, lm_head width, token ownership),
not one compromise slider.

    profile          maximises                 typical price
    context          max_total_num_tokens      slowest rank carries most tokens
    decode_speed     tok/s at bs=1             less context
    prefill_speed    prompt throughput         decode split off-optimum
    ttft_loaded      time to first token        needs a satellite and a handover
    energy           joules per token          peak throughput drops

Two rules keep this from becoming a wish list.

**Staged by evidence.** Before a probe the planner knows STRUCTURE — which
cards, how much VRAM, what fits. After a probe it knows RATES — how fast each
side computes, reads and communicates. Fit questions need no probe; speed
questions do. So an unprobed rig gets a deliberately vague suggestion, and the
detailed one unlocks only once measurements exist. The staging is not a UI
nicety; it is a statement about what is answerable.

**Gated by the build.** A profile that emits a flag this build does not have
is worse than no profile at all. Every flag a lever wants is checked against
``ServerArgs`` (:func:`flag_available`), and a lever whose central knob is
missing says so and names the branch instead of quietly emitting a command
that will fail to parse. On this branch, for example, ``--rank-kv-ratio`` is
absent, which is exactly the knob the decode lever wants — and DESIGN_210
already records that ``--rank-perf-tune dec`` is a documented near-no-op
because the decode lever sits in the token split, not in the weight split.
Saying that is more useful than emitting a flag combination that cannot move
the quantity it claims to move.

Counter-reckoning is mandatory: :class:`LeverProfile` has no field for the
gain without a matching field for the cost, so "in all directions" is
structural rather than a habit.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "Confidence",
    "FlagSpec",
    "LeverProfile",
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

    def render(self) -> str:
        return f"{self.flag} {self.value}" if self.value is not None else self.flag


@dataclasses.dataclass
class LeverProfile:
    """A direction, its flags, and what it costs.

    ``gains`` and ``costs`` are both required and both non-empty; a profile
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


def missing_flags(profile: LeverProfile) -> List[FlagSpec]:
    return [f for f in profile.flags if f.field and not flag_available(f.field)]


# ---------------------------------------------------------------------------
# The five levers
# ---------------------------------------------------------------------------

LEVERS: Dict[str, LeverProfile] = {}


def _register(p: LeverProfile) -> LeverProfile:
    LEVERS[p.key] = p
    return p


_register(
    LeverProfile(
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
                "--rank-perf-loose-ctx-percent",
                "0",
                "rank_perf_loose_ctx_percent",
                why="Zero tolerance: no context may be traded for speed in "
                "this profile.",
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
    LeverProfile(
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
                "toward the fast cards. This is where the decode lever "
                "actually sits.",
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
        ],
        costs=[
            "Less context: the speed-directed token split holds fewer tokens "
            "on the high-capacity cards.",
            "The gain is bounded by the collectives and by the slowest rank; "
            "at bs=1 those dominate.",
        ],
        tradeoff_against={
            "context": "Directly opposed — see the context profile.",
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
            "proportion, i.e. toward the prefill profile.",
        ],
    )
)

_register(
    LeverProfile(
        key="prefill_speed",
        label="Prefill speed",
        maximises="prompt throughput",
        flags=[
            FlagSpec(
                "--rank-tp-ratio",
                "auto-performance",
                "rank_tp_ratio",
                why="Runs the split optimiser instead of the plain "
                "VRAM-proportional split.",
                heterogeneous_only=True,
                needs_probe=True,
            ),
            FlagSpec(
                "--rank-perf-tune",
                "enc",
                "rank_perf_tune",
                why="Family-selective MLP concentration toward the "
                "compute-strong ranks — the measured prefill lever.",
                needs_probe=True,
            ),
            FlagSpec(
                "--rank-perf-loose-ctx-percent",
                "15",
                "rank_perf_loose_ctx_percent",
                why="Allows the optimiser to spend some context on compute "
                "concentration.",
            ),
        ],
        gains=[
            "Concentrates MLP mass on the compute-strong card, which is where "
            "prefill time is spent.",
            "The compute ratio between card classes spreads wider than the "
            "bandwidth ratio, so this profile moves further from an even split "
            "than the decode profile does.",
        ],
        costs=[
            "The decode split ends up off its optimum — one static weight "
            "split cannot sit at both.",
            "Context is spent: concentrating mass on one card lowers the "
            "min-synced pool.",
        ],
        tradeoff_against={
            "context": "Concentration reduces the min-synced KV pool.",
            "decode_speed": "One static split cannot hold both optima; "
            "chunked prefill mixes the phases in one batch, so the operating "
            "point is a weighted mean rather than either optimum.",
        },
        vague_statement=(
            "Prefill favours the compute-strong card more strongly than decode "
            "does. By how much requires measured compute throughput per card."
        ),
    )
)

_register(
    LeverProfile(
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
            "Needs a second node and a handover path; without one this profile "
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
    LeverProfile(
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
            "Peak throughput drops. This profile trades the top of the range "
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

    profile: LeverProfile
    stage: str
    confidence: str
    statement: str
    command_flags: List[str]
    unavailable_flags: List[Dict[str, str]]
    unmet_preconditions: List[str]
    counter_reckoning: Dict[str, str]

    def to_json(self) -> dict:
        return {
            "key": self.profile.key,
            "label": self.profile.label,
            "maximises": self.profile.maximises,
            "stage": self.stage,
            "confidence": self.confidence,
            "statement": self.statement,
            "command_flags": list(self.command_flags),
            "unavailable_flags": list(self.unavailable_flags),
            "unmet_preconditions": list(self.unmet_preconditions),
            "gains": list(self.profile.gains),
            "costs": list(self.profile.costs),
            "counter_reckoning": dict(self.counter_reckoning),
            "notes": list(self.profile.notes),
        }


def suggest_levers(
    heterogeneous: bool = True,
    probe: Optional[dict] = None,
    node_count: int = 1,
    facility_keys_available: Optional[Sequence[str]] = None,
    keys: Optional[Sequence[str]] = None,
) -> List[LeverSuggestion]:
    """Resolve every lever against the evidence available right now.

    ``probe`` is a cached hardware-probe dict (or None). Its presence is what
    moves a lever from a vague statement to a concrete flag set — the
    structure/rates line from DESIGN_216.
    """
    stage = RATES_KNOWN if probe else STRUCTURE_ONLY
    have_facilities = set(facility_keys_available or [])
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
        flags: List[str] = []
        for f in p.flags:
            if f in missing:
                continue
            if f.heterogeneous_only and not heterogeneous:
                continue
            if f.needs_probe and not probe:
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

        if unavailable and not flags:
            confidence = Confidence.BLOCKED
            statement = (
                f"{p.label}: not settable on this build — "
                + "; ".join(u["reason"] for u in unavailable)
            )
        elif stage == STRUCTURE_ONLY:
            confidence = Confidence.VAGUE
            statement = p.vague_statement
        else:
            confidence = Confidence.DETAILED
            statement = (
                f"{p.label} maximises {p.maximises}. "
                + (f"Set: {' '.join(flags)}. " if flags else "")
                + f"Price: {p.costs[0]}"
            )
        out.append(
            LeverSuggestion(
                profile=p,
                stage=stage,
                confidence=confidence,
                statement=statement,
                command_flags=flags,
                unavailable_flags=unavailable,
                unmet_preconditions=unmet,
                counter_reckoning=dict(p.tradeoff_against),
            )
        )
    return out


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
        lines.append(f"[{s.confidence}] {s.profile.label} -> {s.profile.maximises}")
        lines.append(f"  {s.statement}")
        if s.command_flags:
            lines.append(f"  flags: {' '.join(s.command_flags)}")
        for u in s.unavailable_flags:
            lines.append(f"  unavailable: {u['flag']} — {u['reason']}")
        for pre in s.unmet_preconditions:
            lines.append(f"  precondition unmet: {pre}")
        for k, v in s.counter_reckoning.items():
            lines.append(f"  costs {k}: {v}")
        lines.append("")
    return "\n".join(lines)
