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
"""The short probe: four quantities per card, two per ORDERED pair.

DESIGN_216 fixes the scope precisely, and the scope is the point. Per card:
usable VRAM, memory bandwidth, compute throughput in the dtypes that actually
occur. Per ordered pair, including pairs inside one rig: latency for a small
message, bandwidth for a large one. Nothing else. The 30-second budget only
holds while the probe measures exactly what the cost model consumes — every
addition is the first step towards a probe that ends up running half a
benchmark.

**Everything measured, nothing from a datasheet.** This project has been
misled by nameplate numbers repeatedly: GPU0 sits on a x4 link, no pair has
P2P, and NVML's device order does not match torch's.

**What this module binds rather than rebuilds.** ``sglang.srt.uneven_perf``
already runs a stage-0 micro-probe: per-card GEMM and memory bandwidth, a
pairwise NCCL P2P bandwidth matrix, and a group all-reduce latency, cached
under ``~/.cache/sglang/hw_profile-*.json``. That is the intra-host half of
what DESIGN_216 asks for, it is already wired into the boot path, and
duplicating it would mean two probes disagreeing about the same rig. So this
module reads it (:func:`from_hardware_profile`) and adds the three things it
does not have:

1. **Ordered pairs.** ``uneven_perf`` measures a -> b only and stores one
   entry per unordered pair. Routes can be asymmetric, so the reverse
   direction is emitted as a distinct link marked ``mirrored``: present so the
   matrix is complete, labelled so nobody reads it as measured.
2. **Per-pair latency.** ``uneven_perf`` measures ONE all-reduce latency over
   the whole group. That is a group figure, not a pair figure, and it is kept
   as one rather than being spread across the pairs.
3. **The rig boundary.** Nothing in the existing probe crosses a machine.
   :func:`measure_host_link` covers the host-to-host leg over the transport
   that is already open, and says plainly that it BOUNDS the device-to-device
   path without being it.

**State travels with every point.** A short probe measures a MOMENT. Clock,
temperature and throttle state belong in each result, and a point taken under
throttling is KEPT AND MARKED, never dropped — losing it would silently
narrow the sample, and keeping it unmarked would let a hot afternoon become a
permanent configuration. The live evidence for this: GPU2 measured 88 C with
`sw_thermal_slowdown` active at 1695 instead of 1905 MHz.
"""

from __future__ import annotations

import dataclasses
import socket
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CardState",
    "CardRate",
    "LinkRate",
    "GroupLatency",
    "ProbeResult",
    "ProbeBudget",
    "endpoint_id",
    "estimate_budget",
    "from_hardware_profile",
    "measure_host_link",
    "MEASURED",
    "MIRRORED",
    "DEFAULT_MAX_AGE_S",
    "BUDGET_SECONDS",
]

#: DESIGN_216's runtime target for one rig pair.
BUDGET_SECONDS = 30.0
#: Per-endpoint cost the budget is estimated from, taken from the design's own
#: arithmetic (~2-3 s per card, ~1-2 s per pair).
_SECONDS_PER_CARD = 3.0
_SECONDS_PER_PAIR = 2.0

#: A probe older than this is reported as stale rather than used silently.
DEFAULT_MAX_AGE_S = 7 * 24 * 3600.0

#: How far a state sample may sit from the probe and still be called the state
#: the probe was taken in. Beyond it the two are reported as separate facts.
_STATE_CONTEMPORARY_S = 300.0

MEASURED = "measured"
MIRRORED = "mirrored from the reverse direction (routes can be asymmetric)"


def endpoint_id(node_id: str, uuid: str) -> str:
    """A card's identity across the rig pair. UUID, never an index: NVML and
    torch disagree about ordering, and an index means a different card on the
    other machine."""
    return f"{node_id}/{uuid}"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CardState:
    """The condition a measurement was taken under.

    Without this a probe result is a number without a context, and a
    recommendation derived from a throttled card understates it permanently.
    """

    sm_clock_mhz: Optional[int] = None
    sm_clock_max_mhz: Optional[int] = None
    temp_c: Optional[float] = None
    throttle_reasons: Tuple[str, ...] = ()
    #: When this state was read. A cached probe is normally attached to a
    #: state sampled much later, and a state from a different moment is not
    #: the state the measurement was taken in — see
    #: :meth:`ProbeResult.state_warnings`.
    sampled_at: Optional[float] = None

    @property
    def throttled(self) -> bool:
        return bool(self.throttle_reasons)

    @property
    def clock_ratio(self) -> Optional[float]:
        if not self.sm_clock_mhz or not self.sm_clock_max_mhz:
            return None
        return self.sm_clock_mhz / self.sm_clock_max_mhz

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["throttle_reasons"] = list(self.throttle_reasons)
        d["throttled"] = self.throttled
        d["clock_ratio"] = self.clock_ratio
        return d


# ---------------------------------------------------------------------------
# Per card
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CardRate:
    """One card's measured ceiling, plus the state it was measured in."""

    node_id: str
    uuid: str
    name: str
    total_mib: Optional[int] = None
    gemm_tflops: Optional[float] = None
    membw_gbs: Optional[float] = None
    #: The dtype the GEMM figure was measured in. A compute ceiling without
    #: its dtype is not a ceiling — fp8 and bf16 differ by a factor of two on
    #: the cards this fork targets, and Ampere has no fp8 tensor path at all.
    gemm_dtype: str = "bfloat16"
    state: Optional[CardState] = None

    @property
    def id(self) -> str:
        return endpoint_id(self.node_id, self.uuid)

    @property
    def throttled(self) -> bool:
        return bool(self.state and self.state.throttled)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["id"] = self.id
        d["state"] = self.state.to_json() if self.state else None
        d["throttled"] = self.throttled
        return d


# ---------------------------------------------------------------------------
# Per ordered pair
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LinkRate:
    """One ORDERED pair: src -> dst.

    Ordered because a route can be asymmetric — a card on a x4 slot uploads
    and downloads differently, and a pair crossing the chipset is not the pair
    crossing the CPU. ``direction`` says whether this entry was measured or
    mirrored, so a matrix that looks complete cannot be mistaken for one that
    was fully measured.
    """

    src: str
    dst: str
    latency_us: Optional[float] = None
    bandwidth_gbs: Optional[float] = None
    #: How it was measured: "nccl p2p", "tcp (host-to-host)", ...
    transport: str = ""
    direction: str = MEASURED
    same_node: bool = True
    #: Message sizes the two figures came from. A bandwidth without its
    #: message size is not comparable with another bandwidth.
    latency_bytes: Optional[int] = None
    bandwidth_bytes: Optional[int] = None
    note: str = ""

    @property
    def measured(self) -> bool:
        return self.direction == MEASURED

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["measured"] = self.measured
        return d


@dataclasses.dataclass
class GroupLatency:
    """The collective latency over a whole group.

    Kept separate from the pair matrix on purpose: ``uneven_perf`` measures one
    all-reduce over every rank at once, and spreading that figure across the
    pairs would invent per-pair numbers that were never measured.
    """

    node_id: str
    members: List[str] = dataclasses.field(default_factory=list)
    allreduce_10kb_us: Optional[float] = None
    allreduce_1mb_us: Optional[float] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProbeBudget:
    """Whether a probe of this shape can still be called short."""

    cards: int
    ordered_pairs: int
    estimate_s: float
    budget_s: float = BUDGET_SECONDS

    @property
    def fits(self) -> bool:
        return self.estimate_s <= self.budget_s

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["fits"] = self.fits
        return d


def estimate_budget(
    n_cards: int, measure_both_directions: bool = False
) -> ProbeBudget:
    """Estimated runtime for a probe over ``n_cards`` endpoints.

    Pairs are counted as UNORDERED unless both directions are actually
    measured, because mirroring costs nothing. Measuring both directions
    doubles the pair term, which is what makes it a decision rather than a
    default: on four endpoints it is the difference between fitting the budget
    and not.
    """
    unordered = n_cards * (n_cards - 1) // 2
    pairs = unordered * (2 if measure_both_directions else 1)
    estimate = n_cards * _SECONDS_PER_CARD + pairs * _SECONDS_PER_PAIR
    return ProbeBudget(
        cards=n_cards,
        ordered_pairs=unordered * 2,
        estimate_s=round(estimate, 1),
    )


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProbeResult:
    """Cards, ordered pairs, group latencies — with an age and a state."""

    created: float = 0.0
    duration_s: Optional[float] = None
    cards: List[CardRate] = dataclasses.field(default_factory=list)
    links: List[LinkRate] = dataclasses.field(default_factory=list)
    group_latencies: List[GroupLatency] = dataclasses.field(default_factory=list)
    #: Node ids that contributed. Two means the rig boundary was crossed.
    nodes: List[str] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)

    # -- age and state ------------------------------------------------------

    def age_s(self, now: Optional[float] = None) -> Optional[float]:
        if not self.created:
            return None
        return max(0.0, (now if now is not None else time.time()) - self.created)

    def is_stale(
        self, now: Optional[float] = None, max_age_s: float = DEFAULT_MAX_AGE_S
    ) -> bool:
        age = self.age_s(now)
        return age is not None and age > max_age_s

    @property
    def throttled_cards(self) -> List[CardRate]:
        return [c for c in self.cards if c.throttled]

    def state_warnings(self, now: Optional[float] = None) -> List[str]:
        """Everything that changes how these numbers must be read.

        Emitted as data rather than logged, because the surface has to show it:
        a recommendation derived from a throttled or stale probe carries that
        provenance for as long as it is on screen.
        """
        out: List[str] = []
        # Whether the state describes the measurement or merely accompanies
        # it. A cached probe read today carries today's clocks, which is a
        # different statement from "this is how the card was when measured".
        contemporary = True
        for c in self.cards:
            at = c.state.sampled_at if c.state else None
            if at and self.created and abs(at - self.created) > _STATE_CONTEMPORARY_S:
                contemporary = False
                break
        for c in self.throttled_cards:
            reasons = ", ".join(c.state.throttle_reasons) if c.state else ""
            ratio = c.state.clock_ratio if c.state else None
            at = f" at {ratio * 100:.0f} % of its maximum clock" if ratio else ""
            if contemporary:
                out.append(
                    f"{c.name} ({c.id}) was throttled while measured "
                    f"({reasons}){at}. The point is kept and marked; any "
                    "recommendation derived from it understates this card."
                )
            else:
                out.append(
                    f"{c.name} ({c.id}) is throttled RIGHT NOW ({reasons}){at}, "
                    "but this probe came from the cache and was measured at a "
                    "different moment. The state describes the card today, not "
                    "the conditions the numbers were taken under. Re-probe to "
                    "tie the two together."
                )
        if not contemporary and not self.throttled_cards:
            out.append(
                "Card state was read now, not when the cached probe ran, so "
                "the clocks and temperatures shown accompany these numbers "
                "rather than describe them."
            )
        if self.is_stale(now):
            age = self.age_s(now) or 0.0
            out.append(
                f"This probe is {age / 3600:.0f} h old. Hardware, driver and "
                "thermal conditions drift; re-probe before deriving a "
                "configuration from it."
            )
        mirrored = [l for l in self.links if not l.measured]
        if mirrored:
            out.append(
                f"{len(mirrored)} of {len(self.links)} ordered pairs are "
                "mirrored from the opposite direction rather than measured. "
                "Asymmetric routes will not show up in them."
            )
        return out

    # -- lookup -------------------------------------------------------------

    def card(self, endpoint: str) -> Optional[CardRate]:
        for c in self.cards:
            if c.id == endpoint:
                return c
        return None

    def link(self, src: str, dst: str) -> Optional[LinkRate]:
        for l in self.links:
            if l.src == src and l.dst == dst:
                return l
        return None

    def missing_pairs(self) -> List[Tuple[str, str]]:
        """Ordered pairs the matrix does not cover.

        A transport choice made over an incomplete matrix is a guess, so the
        gap is enumerated rather than filled in.
        """
        ids = [c.id for c in self.cards]
        have = {(l.src, l.dst) for l in self.links}
        return [(a, b) for a in ids for b in ids if a != b and (a, b) not in have]

    @property
    def crosses_rig_boundary(self) -> bool:
        return len(set(self.nodes)) > 1

    def to_json(self) -> dict:
        return {
            "created": self.created,
            "duration_s": self.duration_s,
            "cards": [c.to_json() for c in self.cards],
            "links": [l.to_json() for l in self.links],
            "group_latencies": [g.to_json() for g in self.group_latencies],
            "nodes": list(self.nodes),
            "notes": list(self.notes),
            "state_warnings": self.state_warnings(),
            "missing_pairs": [list(p) for p in self.missing_pairs()],
            "crosses_rig_boundary": self.crosses_rig_boundary,
        }

    def merge(self, other: "ProbeResult") -> "ProbeResult":
        """Fold a second node's probe into this one.

        Card and link identities are node-scoped, so a merge cannot collide;
        what it cannot produce is the CROSS-node pairs, which have to be
        measured over the link between the two and are added separately.
        """
        merged = ProbeResult(
            created=min(x for x in (self.created, other.created) if x) if (
                self.created and other.created
            ) else (self.created or other.created),
            duration_s=(
                (self.duration_s or 0.0) + (other.duration_s or 0.0)
                if (self.duration_s or other.duration_s)
                else None
            ),
            cards=list(self.cards) + list(other.cards),
            links=list(self.links) + list(other.links),
            group_latencies=list(self.group_latencies) + list(other.group_latencies),
            nodes=list(dict.fromkeys(list(self.nodes) + list(other.nodes))),
            notes=list(self.notes) + list(other.notes),
        )
        return merged


# ---------------------------------------------------------------------------
# Binding: the existing intra-host probe
# ---------------------------------------------------------------------------


def from_hardware_profile(
    profile: Optional[dict],
    node_id: str,
    card_states: Optional[Dict[str, CardState]] = None,
) -> ProbeResult:
    """Build a :class:`ProbeResult` from ``uneven_perf``'s cached profile.

    ``card_states`` is a UUID-keyed snapshot taken from the collector at the
    same moment; it supplies the clock/temperature/throttle context the
    micro-probe does not record itself.
    """
    result = ProbeResult(nodes=[node_id])
    if not profile:
        result.notes.append(
            "No cached hardware profile. Fit questions are still answerable "
            "from VRAM alone; speed questions are not."
        )
        return result

    states = card_states or {}
    created = profile.get("created")
    if isinstance(created, str):
        try:
            result.created = time.mktime(time.strptime(created, "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            result.created = 0.0
    elif isinstance(created, (int, float)):
        result.created = float(created)
    result.duration_s = profile.get("probe_seconds")

    for uuid, g in (profile.get("gpus") or {}).items():
        result.cards.append(
            CardRate(
                node_id=node_id,
                uuid=str(uuid),
                name=str(g.get("name", "unknown")),
                total_mib=g.get("total_mib"),
                gemm_tflops=g.get("gemm_tflops"),
                membw_gbs=g.get("membw_gbs"),
                state=states.get(str(uuid)),
            )
        )

    links = profile.get("links") or {}
    for key, entry in links.items():
        if key == "__group__":
            result.group_latencies.append(
                GroupLatency(
                    node_id=node_id,
                    members=[c.id for c in result.cards],
                    allreduce_10kb_us=entry.get("ar_10kb_us"),
                    allreduce_1mb_us=entry.get("ar_1mb_us"),
                )
            )
            continue
        parts = key.split("|")
        if len(parts) != 2:
            continue
        a, b = (endpoint_id(node_id, p) for p in parts)
        gbs = entry.get("p2p_gbs")
        # One direction was measured; the reverse is mirrored and says so.
        result.links.append(
            LinkRate(
                src=a,
                dst=b,
                bandwidth_gbs=gbs,
                transport="nccl p2p",
                direction=MEASURED,
                same_node=True,
                bandwidth_bytes=1024 * 1024,
                note="latency per pair is not measured by the stage-0 probe; "
                "the group all-reduce latency is reported separately",
            )
        )
        result.links.append(
            LinkRate(
                src=b,
                dst=a,
                bandwidth_gbs=gbs,
                transport="nccl p2p",
                direction=MIRRORED,
                same_node=True,
                bandwidth_bytes=1024 * 1024,
            )
        )
    return result


# ---------------------------------------------------------------------------
# The rig boundary
# ---------------------------------------------------------------------------


def measure_host_link(
    host: str,
    port: int,
    small_bytes: int = 64,
    large_bytes: int = 8 * 1024 * 1024,
    rounds: int = 20,
    timeout_s: float = 5.0,
    connect=None,
) -> LinkRate:
    """Measure the HOST-to-host leg to another node: RTT and bulk bandwidth.

    This is deliberately not presented as a device-to-device figure. It bounds
    one: a cross-rig tensor transfer cannot beat the wire it rides on, and no
    GPU transport removes the host-to-host latency underneath it. Naming it
    honestly is what keeps a later transport choice from claiming a
    device-to-device number nobody measured.

    ``connect`` is injectable so the measurement is testable over a loopback
    socket without a second machine.
    """
    src = "local"
    dst = f"{host}:{port}"
    opener = connect or (lambda: socket.create_connection((host, port), timeout_s))
    try:
        sock = opener()
    except OSError as e:
        return LinkRate(
            src=src,
            dst=dst,
            transport="tcp (host-to-host)",
            same_node=False,
            note=f"unreachable: {e}",
        )
    try:
        sock.settimeout(timeout_s)
        payload = b"\x00" * small_bytes
        rtts: List[float] = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            sock.sendall(payload)
            _recv_exactly(sock, small_bytes)
            rtts.append((time.perf_counter() - t0) * 1e6)
        rtts.sort()
        # Median, not mean: one scheduling hiccup in twenty must not become
        # the latency of the link.
        latency = rtts[len(rtts) // 2]

        big = b"\x00" * large_bytes
        t0 = time.perf_counter()
        sock.sendall(big)
        _recv_exactly(sock, large_bytes)
        elapsed = time.perf_counter() - t0
        # Round trip, so the bytes crossed the wire twice.
        gbs = (large_bytes * 2 / 1e9) / elapsed if elapsed > 0 else None
    except OSError as e:
        return LinkRate(
            src=src,
            dst=dst,
            transport="tcp (host-to-host)",
            same_node=False,
            note=f"measurement failed: {e}",
        )
    finally:
        try:
            sock.close()
        except OSError:
            pass

    return LinkRate(
        src=src,
        dst=dst,
        latency_us=round(latency, 1),
        bandwidth_gbs=round(gbs, 3) if gbs else None,
        transport="tcp (host-to-host)",
        direction=MEASURED,
        same_node=False,
        latency_bytes=small_bytes,
        bandwidth_bytes=large_bytes,
        note="host-to-host over the pairing connection. It BOUNDS the "
        "device-to-device path across the rig boundary; it is not that path, "
        "and a GPU transport cannot undercut it.",
    )


def _recv_exactly(sock, n: int) -> bytes:
    chunks = []
    got = 0
    while got < n:
        block = sock.recv(min(65536, n - got))
        if not block:
            raise OSError("connection closed mid-measurement")
        chunks.append(block)
        got += len(block)
    return b"".join(chunks)


def echo_server(sock) -> None:
    """The other side of :func:`measure_host_link`: echo whatever arrives.

    Trivially small on purpose. The probe needs a peer that returns the bytes;
    anything more would be a second protocol to keep in step.
    """
    try:
        while True:
            block = sock.recv(65536)
            if not block:
                return
            sock.sendall(block)
    except OSError:
        return
