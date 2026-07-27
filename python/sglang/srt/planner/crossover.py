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
"""Where an MLP weight concentration starts to pay, and on which rig.

Concentrating dense-MLP units on the compute-strong rank buys prefill and
costs decode. Both terms are real and both have been measured, so the question
is not "does it help" but "at what mix of prompt and output tokens does the
first term overtake the second".

    net per output token = (prompt:output ratio) x prefill saving per prompt
                           token  -  decode cost per output token

The break-even ratio is where that expression crosses zero.

**The crossover is a property of a rig, not of the feature.** It depends on
the card mix, on the interconnect, on the model and on the quantisation path
each rank takes -- on the reference rig the fast card runs a native-FP8 lane
and the two slow cards run a Marlin upconvert, and that asymmetry is half the
decode cost. A number carried over from another rig is a guess wearing a
measurement's clothes. This module therefore stores findings with the rig they
came from and refuses to answer for a rig it has no measurement for:
:meth:`CrossoverFinding.usable_for_advice` is false for anything not measured
here, and the caller is expected to offer the measurement instead
(:data:`STUDY_TIERS`).

Three provenances are kept apart everywhere, because they carry different
weight and collapsing them is how a fitted constant becomes a law:

``measured_here``       this rig ran the study; the only source that may
                        select a vector.
``measured_elsewhere``  another rig's finding. Shown for size and shape, with
                        its cards named. Never selects anything.
``modelled``            the parse-time optimizer's prediction. Its decode term
                        is fitted to measurement; its prefill term is not and
                        over-predicts (:data:`MODELLED_PREFILL_NOTE`), so a
                        modelled NET is not reported at all
                        (:data:`MODELLED_NET_REFUSED`).

**Which candidates may be proposed is structural.** A candidate is kept only
if some prompt:output ratio exists at which it is the best positive-net
choice -- the upper envelope of the net lines. A candidate that is beaten
everywhere is dropped with the reason, whatever its numbers happen to be on a
given rig; that rule does not depend on absolute values and so may be applied
generally, unlike the crossover ratios themselves.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "MEASURED",
    "MEASURED_HERE",
    "MEASURED_ELSEWHERE",
    "MODELLED",
    "EVIDENCE_KINDS",
    "PROVENANCE",
    "MODELLED_DECODE_NOTE",
    "MODELLED_PREFILL_NOTE",
    "MODELLED_NET_REFUSED",
    "STALE_AFTER_S",
    "STUDY_KEY",
    "STUDY_TIERS",
    "StudyTier",
    "ConcentrationPoint",
    "RigDescriptor",
    "CrossoverFinding",
    "REFERENCE_FINDING",
    "DEFAULT_FINDING_PATH",
    "load_finding",
    "save_finding",
    "describe_evidence",
]

#: This rig ran the study. The only provenance that may select a vector.
MEASURED_HERE = "measured_here"
#: Another rig's finding: shown with its cards named, never applied here.
MEASURED_ELSEWHERE = "measured_elsewhere"
#: The parse-time cost model's prediction.
MODELLED = "modelled"
PROVENANCE = (MEASURED_HERE, MEASURED_ELSEWHERE, MODELLED)

#: Display kinds. A surface only has to answer "did somebody measure this or
#: did something predict it"; WHICH rig measured it is carried by the rig
#: label next to the number, not by a third kind nobody reads.
MEASURED = "measured"
EVIDENCE_KINDS = (MEASURED, MODELLED)

#: What the optimizer's decode prediction is worth. Fitted against four
#: measured vectors on the reference rig and landing within ~2 points of them,
#: where the model it replaced was 26-38 points out with the wrong sign.
MODELLED_DECODE_NOTE = (
    "Modelled: parse-time prediction from the effective-bandwidth cost model. "
    "Fitted against four measured concentration vectors on the reference rig "
    "and within ~2 points of them there. The fit is one rig wide."
)

#: What it is NOT worth on the other axis. Stated wherever a prefill number
#: from the model is shown, because the two halves are not equally trustworthy
#: and a reader will otherwise assume they are.
MODELLED_PREFILL_NOTE = (
    "Modelled and not calibrated: the prefill term over-predicts by ~1.8x "
    "where it has been checked (23.7 % predicted against 13.0 % measured "
    "slope gain on the reference rig). Read it as a direction, not a value."
)

#: Why no modelled net is printed anywhere.
MODELLED_NET_REFUSED = (
    "A modelled net combines a decode term fitted to measurement with a "
    "prefill term that is not, so the net inherits the uncalibrated error "
    "with no bound. Not reported; measure the crossover instead."
)

#: A crossover finding older than this is not used for advice. Driver
#: versions, clock behaviour and the model's own quantisation path all move.
STALE_AFTER_S = 60 * 60 * 24 * 60  # 60 days

#: The scenario that produces a finding.
STUDY_KEY = "mlp_split_crossover"

DEFAULT_FINDING_PATH = os.path.expanduser("~/.cache/sglang/mlp_crossover.json")

_EPS = 1e-12


# ---------------------------------------------------------------------------
# The two measured slopes per candidate
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ConcentrationPoint:
    """One MLP vector, as the two slopes that decide whether it pays.

    Both are per-token so they can be combined at any prompt:output mix. The
    prefill saving is per PROMPT token (the difference of the prefill slope
    against the base split, which is what saturates past a couple of thousand
    tokens); the decode cost is per OUTPUT token (the per-step difference over
    the base arm's accept length).
    """

    vector: Tuple[int, ...]
    prefill_ms_per_prompt_token_saved: float
    decode_ms_per_output_token_cost: float
    #: Percentages at the longest measured prompt, for display only.
    prefill_gain_pct: Optional[float] = None
    decode_cost_pct: Optional[float] = None

    def net_ms_per_output_token(self, prompt_to_output: float) -> float:
        """Positive = the split wins at this mix."""
        return (
            float(prompt_to_output) * self.prefill_ms_per_prompt_token_saved
            - self.decode_ms_per_output_token_cost
        )

    @property
    def break_even_ratio(self) -> Optional[float]:
        """Prompt tokens per output token at which the two terms cancel.

        ``None`` when the candidate saves nothing in prefill: then no ratio
        makes it pay, however long the prompt.
        """
        if self.prefill_ms_per_prompt_token_saved <= _EPS:
            return None
        if self.decode_ms_per_output_token_cost <= 0.0:
            return 0.0
        return (
            self.decode_ms_per_output_token_cost
            / self.prefill_ms_per_prompt_token_saved
        )

    def label(self) -> str:
        return ",".join(str(v) for v in self.vector)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["vector"] = list(self.vector)
        return d

    @classmethod
    def from_json(cls, d: dict) -> ConcentrationPoint:
        return cls(
            vector=tuple(int(v) for v in d["vector"]),
            prefill_ms_per_prompt_token_saved=float(
                d["prefill_ms_per_prompt_token_saved"]
            ),
            decode_ms_per_output_token_cost=float(
                d["decode_ms_per_output_token_cost"]
            ),
            prefill_gain_pct=(
                None if d.get("prefill_gain_pct") is None
                else float(d["prefill_gain_pct"])
            ),
            decode_cost_pct=(
                None if d.get("decode_cost_pct") is None
                else float(d["decode_cost_pct"])
            ),
        )


# ---------------------------------------------------------------------------
# What the finding is a finding ABOUT
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RigDescriptor:
    """The setup a crossover was measured on. Carried with every finding so a
    number can never be read without the rig it belongs to."""

    cards: Tuple[str, ...]
    model: str
    quant: str
    tp_size: int
    #: The base MLP unit split the candidates were measured against.
    base_vector: Tuple[int, ...] = ()
    interconnect: str = ""

    def label(self) -> str:
        parts = [", ".join(self.cards) if self.cards else "unknown cards"]
        if self.model:
            parts.append(self.model)
        if self.quant:
            parts.append(self.quant)
        parts.append(f"TP={self.tp_size}")
        if self.interconnect:
            parts.append(self.interconnect)
        return " | ".join(parts)

    def matches(self, other: RigDescriptor) -> bool:
        """Same cards in the same rank order, same model, same quant, same TP.

        Deliberately strict. The measured slopes come out of the interaction
        between a card's quantisation lane and its shard size; a finding for
        the same cards under a different quant describes a different machine
        for this purpose.
        """
        return (
            tuple(self.cards) == tuple(other.cards)
            and self.model == other.model
            and self.quant == other.quant
            and int(self.tp_size) == int(other.tp_size)
        )

    def key(self) -> str:
        return "|".join(
            [",".join(self.cards), self.model, self.quant, str(self.tp_size)]
        )

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["cards"] = list(self.cards)
        d["base_vector"] = list(self.base_vector)
        return d

    @classmethod
    def from_json(cls, d: dict) -> RigDescriptor:
        return cls(
            cards=tuple(d.get("cards") or ()),
            model=d.get("model", ""),
            quant=d.get("quant", ""),
            tp_size=int(d.get("tp_size", 0)),
            base_vector=tuple(int(v) for v in (d.get("base_vector") or ())),
            interconnect=d.get("interconnect", ""),
        )


# ---------------------------------------------------------------------------
# The finding
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CrossoverFinding:
    """A measured (or borrowed, or modelled) crossover, with its conditions.

    Treated like a probe result: it has an age, a card state and a set of
    caveats, and a consumer that ignores them is prevented from doing so by
    :meth:`usable_for_advice` returning false rather than by remembering to
    check.
    """

    rig: RigDescriptor
    points: List[ConcentrationPoint]
    provenance: str
    #: Epoch seconds the measurement was taken.
    measured_at: float = 0.0
    #: Boot-to-boot A-vs-A spread per metric, in percent, as (lo, hi).
    noise_floor_pct: Dict[str, Tuple[float, float]] = dataclasses.field(
        default_factory=dict
    )
    #: Every prefill chunk reported ``#cached-token: 0``. Without this the
    #: prefill numbers may be the radix cache and not the split.
    cache_bypass_proven: bool = False
    #: Any card reported a throttle reason during the run.
    throttled: bool = False
    throttle_reason: str = ""
    #: Per-card clock/temperature/throttle records from the run.
    card_state: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    study: str = STUDY_KEY
    tier: str = ""
    note: str = ""

    def __post_init__(self):
        if self.provenance not in PROVENANCE:
            raise ValueError(
                f"unknown provenance {self.provenance!r}; one of "
                f"{', '.join(PROVENANCE)}. A number whose origin is not one of "
                "these cannot be shown next to ones whose origin is."
            )

    # -- age and state ------------------------------------------------------

    def age_s(self, now: Optional[float] = None) -> float:
        return max(0.0, (now if now is not None else time.time()) - self.measured_at)

    def is_stale(
        self, max_age_s: float = STALE_AFTER_S, now: Optional[float] = None
    ) -> bool:
        return self.age_s(now) > max_age_s

    def caveats(self, now: Optional[float] = None) -> List[str]:
        """Everything a reader has to know before using the numbers."""
        out: List[str] = []
        if self.provenance == MEASURED_ELSEWHERE:
            out.append(
                f"Measured on another rig ({self.rig.label()}); shown for size "
                "and shape, not as this rig's crossover."
            )
        elif self.provenance == MODELLED:
            out.append(MODELLED_PREFILL_NOTE)
        if not self.cache_bypass_proven:
            out.append(
                "Prefill cache bypass not proven for this run: without unique "
                "input ids per request the prefill figures can be radix-cache "
                "hits rather than the split."
            )
        if self.is_stale(now=now):
            out.append(
                f"Stale: measured {self.age_s(now) / 86400.0:.0f} days ago, "
                f"older than the {STALE_AFTER_S / 86400.0:.0f}-day limit."
            )
        if self.throttled:
            out.append(
                "Measured under throttling"
                + (f" ({self.throttle_reason})" if self.throttle_reason else "")
                + ": kept and marked, not dropped, but the decode side is the "
                "half that a clock drop distorts most."
            )
        return out

    def usable_for_advice(self, now: Optional[float] = None) -> bool:
        """May this finding select a vector for this rig?

        Throttling does not disqualify -- the run happened and dropping it
        would hide a real measurement -- but it is always reported alongside.
        """
        if self.provenance != MEASURED_HERE:
            return False
        if not self.cache_bypass_proven:
            return False
        return not self.is_stale(now=now)

    # -- which candidates may be proposed -----------------------------------

    def _paying_points(self) -> List[ConcentrationPoint]:
        return [
            p for p in self.points if p.prefill_ms_per_prompt_token_saved > _EPS
        ]

    def _critical_ratios(self) -> List[float]:
        pts = self._paying_points()
        crit = {0.0}
        for p in pts:
            be = p.break_even_ratio
            if be is not None:
                crit.add(be)
        for i, a in enumerate(pts):
            for b in pts[i + 1 :]:
                ds = (
                    a.prefill_ms_per_prompt_token_saved
                    - b.prefill_ms_per_prompt_token_saved
                )
                if abs(ds) > _EPS:
                    r = (
                        a.decode_ms_per_output_token_cost
                        - b.decode_ms_per_output_token_cost
                    ) / ds
                    if r > 0.0:
                        crit.add(r)
        return sorted(crit)

    def envelope(self) -> List[ConcentrationPoint]:
        """The candidates that are the best positive-net choice somewhere.

        Everything else is off the upper envelope of the net lines: for every
        prompt:output mix at which it wins anything, another candidate wins
        more. Proposing such a vector is never right, on any rig, which is why
        this rule is applied generally where the crossover ratios are not.
        """
        pts = self._paying_points()
        if not pts:
            return []
        xs = self._critical_ratios()
        probes = [(lo + hi) / 2.0 for lo, hi in zip(xs, xs[1:])]
        probes.append((xs[-1] if xs else 0.0) + 1.0)
        keep: List[ConcentrationPoint] = []
        for r in probes:
            nets = [(p.net_ms_per_output_token(r), p) for p in pts]
            best = max(n for n, _ in nets)
            if best <= _EPS:
                continue
            for n, p in nets:
                if n >= best - _EPS and p not in keep:
                    keep.append(p)
        keep.sort(key=lambda p: (p.break_even_ratio or 0.0))
        return keep

    def pruned(self) -> List[Tuple[Tuple[int, ...], str]]:
        """Candidates that are never the best choice, and why."""
        on = {id(p) for p in self.envelope()}
        out = []
        for p in self.points:
            if id(p) in on:
                continue
            be = p.break_even_ratio
            if be is None:
                reason = (
                    "saves nothing in prefill against the base split, so no "
                    "prompt:output ratio makes it pay."
                )
            else:
                reason = (
                    f"never the best choice: it needs {be:.1f}:1 to break even, "
                    "and at every ratio where it wins another candidate wins "
                    "more."
                )
            out.append((p.vector, reason))
        return out

    def best_for_ratio(
        self, prompt_to_output: float
    ) -> Optional[ConcentrationPoint]:
        """The candidate with the largest positive net at this mix, or None.

        None is the answer for most workloads and is not a failure: it means
        the base split is the faster configuration at that mix.
        """
        best: Optional[ConcentrationPoint] = None
        best_net = 0.0
        for p in self.envelope():
            n = p.net_ms_per_output_token(prompt_to_output)
            if n <= _EPS:
                continue
            if best is None or n > best_net + _EPS:
                best, best_net = p, n
        return best

    def break_even_table(self) -> List[Dict[str, Any]]:
        """Every candidate with its two slopes and its break-even, envelope
        membership included so a reader sees what was dropped and why."""
        pruned = dict(self.pruned())
        rows = []
        for p in self.points:
            be = p.break_even_ratio
            rows.append(
                {
                    "vector": p.label(),
                    "prefill_ms_saved_per_prompt_token": (
                        p.prefill_ms_per_prompt_token_saved
                    ),
                    "decode_ms_cost_per_output_token": (
                        p.decode_ms_per_output_token_cost
                    ),
                    "prefill_gain_pct": p.prefill_gain_pct,
                    "decode_cost_pct": p.decode_cost_pct,
                    "break_even_prompt_to_output": be,
                    "proposable": p.vector not in pruned,
                    "dropped_because": pruned.get(p.vector, ""),
                }
            )
        return rows

    # -- serialisation ------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "rig": self.rig.to_json(),
            "points": [p.to_json() for p in self.points],
            "provenance": self.provenance,
            "measured_at": self.measured_at,
            "noise_floor_pct": {
                k: list(v) for k, v in self.noise_floor_pct.items()
            },
            "cache_bypass_proven": self.cache_bypass_proven,
            "throttled": self.throttled,
            "throttle_reason": self.throttle_reason,
            "card_state": list(self.card_state),
            "study": self.study,
            "tier": self.tier,
            "note": self.note,
        }

    @classmethod
    def from_json(cls, d: dict) -> CrossoverFinding:
        return cls(
            rig=RigDescriptor.from_json(d.get("rig") or {}),
            points=[ConcentrationPoint.from_json(p) for p in d.get("points") or []],
            provenance=d.get("provenance", MODELLED),
            measured_at=float(d.get("measured_at") or 0.0),
            noise_floor_pct={
                k: (float(v[0]), float(v[1]))
                for k, v in (d.get("noise_floor_pct") or {}).items()
            },
            cache_bypass_proven=bool(d.get("cache_bypass_proven")),
            throttled=bool(d.get("throttled")),
            throttle_reason=d.get("throttle_reason", ""),
            card_state=list(d.get("card_state") or []),
            study=d.get("study", STUDY_KEY),
            tier=d.get("tier", ""),
            note=d.get("note", ""),
        )


# ---------------------------------------------------------------------------
# The reference rig's finding -- an example, never a default
# ---------------------------------------------------------------------------

#: 2026-07-27, the campaign this record comes from.
_REFERENCE_EPOCH = 1785110400.0

#: Measured on ONE rig, over seven interleaved cold boots with the KV
#: ownership vector pinned and only ``--rank-mlp-ratio`` free. Prefill from
#: unique random input ids per request (every prefill chunk reported
#: ``#cached-token: 0``; the positive control, the same prompt twice, went
#: 5505 -> 1319 ms). Decode as ms per speculative step on natural text,
#: because raw ms per output token on random prompts is acceptance-driven.
#:
#: It is here so a reader can see the size and the shape of the effect and so
#: the arithmetic has a fixture. It is NOT this machine's crossover and
#: ``usable_for_advice`` is false for it by construction.
REFERENCE_FINDING = CrossoverFinding(
    rig=RigDescriptor(
        cards=(
            "NVIDIA GeForce RTX 5090",
            "NVIDIA GeForce RTX 3080",
            "NVIDIA GeForce RTX 3080",
        ),
        model="Qwen3.6-27B",
        quant="fp8",
        tp_size=3,
        base_vector=(63, 37, 36),
        interconnect="PCIe, no GPUDirect P2P",
    ),
    points=[
        ConcentrationPoint(
            vector=(3, 1, 1),
            prefill_ms_per_prompt_token_saved=0.0473,
            decode_ms_per_output_token_cost=0.648,
            prefill_gain_pct=7.2,
            decode_cost_pct=6.0,
        ),
        ConcentrationPoint(
            vector=(4, 1, 1),
            prefill_ms_per_prompt_token_saved=0.0649,
            decode_ms_per_output_token_cost=1.444,
            prefill_gain_pct=10.1,
            decode_cost_pct=13.4,
        ),
        ConcentrationPoint(
            vector=(6, 1, 1),
            prefill_ms_per_prompt_token_saved=0.0906,
            decode_ms_per_output_token_cost=1.673,
            prefill_gain_pct=14.7,
            decode_cost_pct=15.5,
        ),
    ],
    provenance=MEASURED_ELSEWHERE,
    measured_at=_REFERENCE_EPOCH,
    noise_floor_pct={
        "ms_per_spec_step": (0.25, 0.84),
        "prefill_ms": (0.03, 2.46),
    },
    cache_bypass_proven=True,
    tier="thorough",
    note=(
        "Prefill gain saturates past ~2000 prompt tokens: prefill is linear in "
        "length in every arm and the split changes the slope, so the "
        "percentage stops growing."
    ),
)


# ---------------------------------------------------------------------------
# Offering the measurement
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StudyTier:
    """One way to run the crossover study. Two exist because a measurement
    nobody starts is not a measurement."""

    key: str
    label: str
    study_file: str
    est_runtime_min: int
    what: str
    vectors: Tuple[str, ...] = ()
    prompt_tokens: Tuple[int, ...] = ()

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["vectors"] = list(self.vectors)
        d["prompt_tokens"] = list(self.prompt_tokens)
        return d


STUDY_TIERS: Tuple[StudyTier, ...] = (
    StudyTier(
        key="quick",
        label="quick crossover",
        study_file="mlp_crossover_quick.json",
        est_runtime_min=55,
        what=(
            "Base split against one concentration vector, three prompt "
            "lengths plus one decode point. 2 floor boots + 1 interleaved "
            "repeat over 2 arms = 4 boots per point, 4 points = 16 boots at "
            "~3.5 min each. Answers whether a crossover exists on this rig "
            "and roughly where; one candidate cannot produce an envelope."
        ),
        vectors=("base", "3,1,1"),
        prompt_tokens=(1000, 4000, 11000),
    ),
    StudyTier(
        key="thorough",
        label="thorough crossover",
        study_file="mlp_crossover_thorough.json",
        est_runtime_min=270,
        what=(
            "Base split against three concentration vectors, six prompt "
            "lengths plus one decode point. 3 floor boots + 2 interleaved "
            "repeats over 4 arms = 11 boots per point, 7 points = 77 boots at "
            "~3.5 min each. Produces the per-candidate break-even table and "
            "the envelope; the prefill sweep may stop early once the slope "
            "saturates."
        ),
        vectors=("base", "3,1,1", "4,1,1", "6,1,1"),
        prompt_tokens=(500, 1000, 2000, 4000, 8000, 11000),
    ),
)


# ---------------------------------------------------------------------------
# The rig-local store
# ---------------------------------------------------------------------------


def load_finding(
    path: Optional[str] = None, rig: Optional[RigDescriptor] = None
) -> Optional[CrossoverFinding]:
    """The locally stored finding, or None.

    ``None`` when the file does not exist, cannot be read, or holds nothing
    for ``rig``. A finding for a different card mix, model or quant is not
    returned at all rather than returned with a warning: this rig's crossover
    is either measured or it is not.
    """
    path = path or DEFAULT_FINDING_PATH
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    entries = data.get("findings") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return None
    best: Optional[CrossoverFinding] = None
    for d in entries:
        try:
            finding = CrossoverFinding.from_json(d)
        except (KeyError, TypeError, ValueError):
            continue
        if rig is not None and not finding.rig.matches(rig):
            continue
        if best is None or finding.measured_at > best.measured_at:
            best = finding
    return best


def save_finding(
    finding: CrossoverFinding, path: Optional[str] = None
) -> str:
    """Store a finding rig-locally, replacing any earlier one for the same rig.

    Earlier findings for OTHER rigs are kept: the same machine can serve
    several models, and each has its own crossover.
    """
    path = path or DEFAULT_FINDING_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    entries: List[dict] = []
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("findings"), list):
            entries = [
                d
                for d in data["findings"]
                if RigDescriptor.from_json(d.get("rig") or {}).key()
                != finding.rig.key()
            ]
    except (OSError, ValueError):
        entries = []
    entries.append(finding.to_json())
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"findings": entries}, f, indent=1)
    os.replace(tmp, path)
    return path


def describe_evidence(
    finding: Optional[CrossoverFinding], now: Optional[float] = None
) -> Dict[str, Any]:
    """What may be said about the crossover right now, in one dict.

    ``state`` is one of ``measured_here``, ``measured_elsewhere_only`` or
    ``unmeasured``; the last two both mean this rig has no answer, and the
    caller is expected to offer :data:`STUDY_TIERS` rather than fill the gap.
    """
    if finding is None:
        return {
            "state": "unmeasured",
            "usable": False,
            "caveats": [],
            "offer": [t.to_json() for t in STUDY_TIERS],
            "reference": REFERENCE_FINDING.rig.label(),
        }
    usable = finding.usable_for_advice(now=now)
    return {
        "state": "measured_here" if usable else (
            "measured_elsewhere_only"
            if finding.provenance == MEASURED_ELSEWHERE
            else "unmeasured"
        ),
        "usable": usable,
        "rig": finding.rig.label(),
        "provenance": finding.provenance,
        "age_days": round(finding.age_s(now) / 86400.0, 1),
        "caveats": finding.caveats(now=now),
        "table": finding.break_even_table(),
        "offer": [] if usable else [t.to_json() for t in STUDY_TIERS],
    }
