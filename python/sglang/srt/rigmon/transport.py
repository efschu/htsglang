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
"""Transport choice, derived from the measured pair matrix.

DESIGN_216 states the rule in one line: *the transport choice falls out of the
pair matrix, not out of the configuration*. This module is that rule, and
nothing more. It ranks the transports a pair could use, and every entry says
which measurement put it there.

Three properties make it honest rather than merely opinionated:

**A missing measurement is not a low score.** If the matrix has no entry for a
pair, the verdict is ``unknown`` and the remedy is to probe, never a default
that looks like a decision. This project's own history is the argument: a
configuration chosen from assumed topology is how "GPU0 is on a x4 link" and
"no pair has P2P" stayed invisible for as long as they did.

**Availability is checked separately from speed.** A transport that would win
on the numbers but is not present on this host is shown, ranked, and marked
unavailable WITH what is missing — not hidden. A hidden option is one the user
cannot decide to install.

**The local optimum is not a verdict about the transport.** On this rig no
pair has GPUDirect P2P, GPU0 sits on four lanes, and every pair goes through
the chipset. Numbers from here are a floor for this box, not a judgement about
the transport in general; a machine with NVLink will rank the same transports
differently, and that is the intended behaviour of a chooser driven by
measurement.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

from sglang.srt.rigmon.probe import LinkRate, ProbeResult

__all__ = [
    "TransportSpec",
    "TransportOption",
    "PairChoice",
    "TRANSPORTS",
    "choose_transport",
    "choose_all_pairs",
    "VERDICT_RECOMMENDED",
    "VERDICT_USABLE",
    "VERDICT_UNAVAILABLE",
    "VERDICT_UNKNOWN",
]

VERDICT_RECOMMENDED = "recommended"
VERDICT_USABLE = "usable"
VERDICT_UNAVAILABLE = "unavailable"
VERDICT_UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class TransportSpec:
    """One transport, and the conditions under which it can carry a pair."""

    key: str
    label: str
    #: Both endpoints on one machine, or across the rig boundary, or either.
    scope: str  # "intra" | "inter" | "any"
    #: Facility keys (:mod:`sglang.srt.rigmon.facilities`) it needs.
    requires: Sequence[str] = ()
    #: Where it sits when the measurement does not separate two candidates.
    #: Lower is preferred. Used ONLY to break ties between transports whose
    #: measured numbers are indistinguishable — never to overrule them.
    tie_break: int = 50
    why: str = ""


#: The transports this fork can actually put behind a pair. Ordered by the
#: layer they sit at, not by preference: preference comes from the matrix.
TRANSPORTS: List[TransportSpec] = [
    TransportSpec(
        "shm",
        "shared memory",
        scope="intra",
        tie_break=10,
        why="Two ranks on one machine that share a physical card exchange "
        "through host memory without a device-to-device hop at all.",
    ),
    TransportSpec(
        "nccl-p2p",
        "NCCL with GPUDirect P2P",
        scope="intra",
        tie_break=20,
        why="Device-to-device without staging through host memory. Visible in "
        "the matrix as a pair bandwidth far above the PCIe staging path.",
    ),
    TransportSpec(
        "nccl-pcie",
        "NCCL staged over PCIe",
        scope="intra",
        tie_break=30,
        why="The fallback when a pair has no P2P: every transfer is staged "
        "through host memory, which is what the measured pair bandwidth on "
        "this rig reflects.",
    ),
    TransportSpec(
        "gloo-tcp",
        "gloo over TCP",
        scope="inter",
        tie_break=60,
        why="The first rung of the cross-rig ladder: no special hardware, "
        "works wherever the pairing connection already works.",
    ),
    TransportSpec(
        "htccl-ucx",
        "HTCCL over UCX",
        scope="inter",
        tie_break=40,
        why="Host-staged, vendor-neutral. The second rung: it needs UCX built "
        "into the image but no fabric hardware.",
    ),
    TransportSpec(
        "rdma",
        "RDMA verbs",
        scope="inter",
        requires=("rdma",),
        tie_break=25,
        why="The third rung: lowest cross-rig latency, but it needs verbs "
        "access, which a container does not get by default.",
    ),
]


@dataclasses.dataclass
class TransportOption:
    """One candidate for one pair, with the evidence behind its rank."""

    key: str
    label: str
    verdict: str
    reason: str
    #: The measured numbers this verdict rests on. Empty means it rests on
    #: nothing measured, which is what ``unknown`` says.
    evidence: Dict[str, float] = dataclasses.field(default_factory=dict)
    missing_facilities: List[str] = dataclasses.field(default_factory=list)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PairChoice:
    """The ranked options for one ordered pair."""

    src: str
    dst: str
    same_node: bool
    options: List[TransportOption] = dataclasses.field(default_factory=list)
    measurement: Optional[dict] = None

    @property
    def chosen(self) -> Optional[TransportOption]:
        for o in self.options:
            if o.verdict == VERDICT_RECOMMENDED:
                return o
        return None

    def to_json(self) -> dict:
        return {
            "src": self.src,
            "dst": self.dst,
            "same_node": self.same_node,
            "options": [o.to_json() for o in self.options],
            "measurement": self.measurement,
            "chosen": self.chosen.key if self.chosen else None,
        }


#: Above this measured pair bandwidth a same-node pair is treated as having a
#: real device-to-device path rather than a staged one. Not a datasheet
#: figure: PCIe 4.0 x16 stages at roughly 13 GB/s in one direction, and every
#: measured pair on this rig sits an order of magnitude below that because
#: nothing here has P2P. A pair well above the staging regime is one where
#: P2P or NVLink is actually carrying the traffic.
P2P_BANDWIDTH_FLOOR_GBS = 20.0

#: Below this, a same-node pair is bandwidth-starved enough that placement
#: matters more than transport selection does. On this rig GPU0 sits on four
#: lanes, which is exactly this case.
STARVED_BANDWIDTH_GBS = 3.0


def choose_transport(
    link: Optional[LinkRate],
    available_facilities: Sequence[str] = (),
    same_node: Optional[bool] = None,
    colocated: bool = False,
) -> PairChoice:
    """Rank the transports for ONE ordered pair from its measurement.

    ``link`` None (or carrying no bandwidth) yields every candidate at
    ``unknown``: a chooser that guesses when it was not told is the failure
    mode this whole probe exists to prevent.
    """
    have = set(available_facilities or ())
    if same_node is None:
        same_node = link.same_node if link is not None else True
    src = link.src if link else ""
    dst = link.dst if link else ""

    choice = PairChoice(src=src, dst=dst, same_node=bool(same_node))
    if link is not None:
        choice.measurement = {
            "latency_us": link.latency_us,
            "bandwidth_gbs": link.bandwidth_gbs,
            "transport_measured_with": link.transport,
            "direction": link.direction,
            "bandwidth_bytes": link.bandwidth_bytes,
        }

    scope = "intra" if same_node else "inter"
    candidates = [t for t in TRANSPORTS if t.scope in (scope, "any")]
    measured_bw = link.bandwidth_gbs if link else None

    options: List[TransportOption] = []
    for spec in candidates:
        missing = [r for r in spec.requires if r not in have]
        evidence: Dict[str, float] = {}
        if measured_bw is not None:
            evidence["bandwidth_gbs"] = measured_bw
        if link is not None and link.latency_us is not None:
            evidence["latency_us"] = link.latency_us

        if missing:
            options.append(
                TransportOption(
                    key=spec.key,
                    label=spec.label,
                    verdict=VERDICT_UNAVAILABLE,
                    reason=(
                        f"{spec.why} Not available here: this host is missing "
                        + ", ".join(missing)
                        + ". Shown so it can be installed rather than "
                        "silently skipped."
                    ),
                    evidence=evidence,
                    missing_facilities=missing,
                )
            )
            continue

        if measured_bw is None:
            options.append(
                TransportOption(
                    key=spec.key,
                    label=spec.label,
                    verdict=VERDICT_UNKNOWN,
                    reason=(
                        "No measurement for this pair, so no transport can be "
                        "recommended for it. Run the short probe; a choice made "
                        "from assumed topology is how a x4 link stays invisible."
                    ),
                    evidence=evidence,
                )
            )
            continue

        options.append(
            TransportOption(
                key=spec.key,
                label=spec.label,
                verdict=VERDICT_USABLE,
                reason=spec.why,
                evidence=evidence,
            )
        )

    # The recommendation, from the measurement.
    usable = [o for o in options if o.verdict == VERDICT_USABLE]
    if usable and measured_bw is not None:
        by_key = {t.key: t for t in candidates}
        if same_node:
            if colocated:
                pick = "shm"
                note = (
                    "Both ranks sit on the same physical card, so there is no "
                    "device-to-device hop to optimise."
                )
            elif measured_bw >= P2P_BANDWIDTH_FLOOR_GBS:
                pick = "nccl-p2p"
                note = (
                    f"The measured {measured_bw:.1f} GB/s is far above the "
                    "PCIe staging regime, so a direct device-to-device path is "
                    "carrying this pair."
                )
            else:
                pick = "nccl-pcie"
                note = (
                    f"The measured {measured_bw:.1f} GB/s is in the staging "
                    "regime: this pair has no GPUDirect P2P, and every "
                    "transfer goes through host memory."
                )
                if measured_bw < STARVED_BANDWIDTH_GBS:
                    note += (
                        f" At {measured_bw:.1f} GB/s the link is starved "
                        "enough that placement matters more than the transport "
                        "does — moving work off this pair beats tuning it."
                    )
        else:
            pick = "rdma" if "rdma" in {o.key for o in usable} else "htccl-ucx"
            note = (
                "Cross-rig: the lowest rung of the ladder that this host can "
                "actually run. "
                + (
                    f"Measured {measured_bw:.2f} GB/s host-to-host bounds it "
                    "from above — no GPU transport undercuts the wire."
                )
            )
        if pick not in {o.key for o in usable}:
            pick = sorted(usable, key=lambda o: by_key[o.key].tie_break)[0].key
        for o in options:
            if o.key == pick:
                o.verdict = VERDICT_RECOMMENDED
                o.reason = f"{note} {by_key[o.key].why}"

    # Recommended first, then usable, then unavailable, then unknown.
    order = {
        VERDICT_RECOMMENDED: 0,
        VERDICT_USABLE: 1,
        VERDICT_UNAVAILABLE: 2,
        VERDICT_UNKNOWN: 3,
    }
    tie = {t.key: t.tie_break for t in TRANSPORTS}
    options.sort(key=lambda o: (order[o.verdict], tie.get(o.key, 99)))
    choice.options = options
    return choice


def choose_all_pairs(
    probe: ProbeResult,
    available_facilities: Sequence[str] = (),
    colocated_pairs: Sequence[tuple] = (),
) -> List[PairChoice]:
    """Run the choice over every ordered pair the probe covers, and over every
    pair it does NOT — the gaps come back as ``unknown`` rather than being
    dropped, because a matrix with holes must not look complete."""
    colocated = {tuple(p) for p in colocated_pairs}
    out: List[PairChoice] = []
    for link in probe.links:
        out.append(
            choose_transport(
                link,
                available_facilities,
                same_node=link.same_node,
                colocated=(link.src, link.dst) in colocated,
            )
        )
    by_id = {c.id: c for c in probe.cards}
    for src, dst in probe.missing_pairs():
        same = (
            by_id[src].node_id == by_id[dst].node_id
            if src in by_id and dst in by_id
            else True
        )
        choice = choose_transport(None, available_facilities, same_node=same)
        choice.src, choice.dst = src, dst
        out.append(choice)
    return out
