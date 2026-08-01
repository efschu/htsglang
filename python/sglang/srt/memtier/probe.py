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
"""Per-tier measurement: the catalogue, the contract, and the refusal (#407).

Cut 1 ships the INTERFACE and no card time. Every probe here is declared,
every probe here refuses, and the refusal names the harness that would produce
the number -- which is the point: DESIGN_407 §4 lists six measurements this
rig does not have, three of them for numbers a reader would otherwise assume
exist (the peer-VRAM "1-3 us posted write" class is an assumption; NVMe
latency is unmeasured in any unit; host DRAM bandwidth is derived from an
assumed DDR4-3200 peak). A registry that quietly filled those in would be
worse than one that has them, because a wrong number outranks a missing one in
every comparison it appears in.

The provenance law, enforced by :func:`apply_outcome` being the only way a
measured number enters a descriptor:

*   a value may enter only from a probe outcome whose status is ``ok``;
*   an ``ok`` outcome must carry a ``MEASURED`` rate -- a probe that returns
    an estimate is a bug in the probe, not a cheaper measurement;
*   an existing ``MEASURED`` value is never overwritten by an ``ESTIMATE`` or
    an absence. A re-run that failed does not erase what a run that worked
    established; it is recorded as a failed outcome instead.

Every violation raises :class:`ProvenanceUpgradeRefused` rather than being
logged and dropped, so the failing path cannot be the quiet one.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import msgspec

from sglang.srt.memtier.tiers import TierCaps, TierDescriptor, TierId, TierKind
from sglang.srt.planner.cost_model import Provenance, Rate

__all__ = [
    "PROBES",
    "ProbeOutcome",
    "ProbeSpec",
    "ProbeTarget",
    "ProvenanceUpgradeRefused",
    "TierProbe",
    "UnimplementedProbe",
    "apply_outcome",
    "missing_measurements",
    "probe_by_id",
    "probes_for",
    "require_measured",
    "run_probe",
]


class ProbeTarget(str, enum.Enum):
    """Which field of a tier record a probe fills. One probe, one number."""

    LATENCY = "caps.latency_us"
    BANDWIDTH = "caps.bandwidth_gbs"
    APERTURE = "caps.aperture_bytes"


#: Statuses, identical to ``planner.comm_suite.ArmResult``: ``absent`` and
#: ``error`` stay distinguishable because "nobody measured this" and "the
#: measurement failed" are different findings about a rig.
PROBE_STATUSES: Tuple[str, ...] = ("ok", "warn", "error", "absent")


class ProbeSpec(msgspec.Struct, frozen=True, kw_only=True):
    """One measurement, the question it answers, and who would take it."""

    id: str
    label: str
    question: str
    target: ProbeTarget
    #: Tier kinds this probe applies to.
    kinds: Tuple[TierKind, ...]
    #: The harness that must produce it. Named, because DESIGN_407 §4 found
    #: one promised measurement that CANNOT come from the harness everybody
    #: assumed: barlink implements collectives only and has no send/recv, so
    #: the BAR1 point-latency ladder needs the p2p_readiness d2d path instead.
    harness: str
    #: What stays absent until it is taken.
    unblocks: str
    #: S / M / L, in the cut plan's units.
    cost: str = "S"
    #: Only for tiers carrying this property value, when set.
    requires_property: Tuple[str, str] = ("", "")
    #: Only for tiers currently carrying one of these health verdicts, when
    #: set. M7 uses it: "what would it take to reach the tiers we cannot
    #: reach" is a question about blocked tiers and about nothing else.
    requires_verdict: Tuple[str, ...] = ()

    def applies_to(self, tier: TierDescriptor) -> bool:
        if tier.kind not in self.kinds:
            return False
        if self.requires_verdict and tier.health.verdict not in self.requires_verdict:
            return False
        key, wanted = self.requires_property
        return not key or tier.properties.get(key) == wanted

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "question": self.question,
            "target": self.target.value,
            "kinds": [k.value for k in self.kinds],
            "harness": self.harness,
            "unblocks": self.unblocks,
            "cost": self.cost,
        }


#: DESIGN_407 §4, as data. The table in the design doc is the source; keeping
#: it here as records rather than prose is what lets the dashboard say "this
#: field is absent AND here is the arm that fills it" in one query.
PROBES: Tuple[ProbeSpec, ...] = (
    ProbeSpec(
        id="M1",
        label="BAR1 point-latency ladder, 8 B - 64 KiB",
        question="What does one small peer-VRAM access actually cost?",
        target=ProbeTarget.LATENCY,
        kinds=(TierKind.DEVICE,),
        harness=(
            "p2p_readiness d2d_bench, or the GDR bench against a local target. "
            "NOT the barlink harness: barlink implements collectives only and "
            "has no send/recv (scripts/probe/barlink_vs_nccl.py:4-22), so the "
            "smallest figure it can produce is a 20 KiB three-rank all_reduce "
            "at 45.59 us -- a collective, not a point latency."
        ),
        unblocks=(
            "the peer-VRAM latency class, which is today an ASSUMPTION quoted "
            "as '1-3 us posted write' (EVAL_p2p_prefill_decode_split.md:140)"
        ),
    ),
    ProbeSpec(
        id="M2",
        label="Effective BAR1 aperture per directed pair",
        question="How much of the nominal window actually maps, each direction?",
        target=ProbeTarget.APERTURE,
        kinds=(TierKind.DEVICE,),
        harness=(
            "scripts/p2p_readiness/capability_matrix.py -- the package exists "
            "and has never been run; there is no results/ directory"
        ),
        unblocks=(
            "TierCaps.aperture_bytes, and #286's window policy (reject vs "
            "chunk), which is defaulted conservatively for want of this"
        ),
    ),
    ProbeSpec(
        id="M3",
        label="Direct D2D bandwidth vs host staging, per direction and size",
        question="Is a vram->vram move cheaper than staging through the host?",
        target=ProbeTarget.BANDWIDTH,
        kinds=(TierKind.DEVICE,),
        harness="scripts/p2p_readiness/d2d_bench.py, same run as M2",
        unblocks="the vram -> vram edge weight of the cut-3 staging graph",
    ),
    ProbeSpec(
        id="M4",
        label="Host DRAM sustained read bandwidth",
        question="What does the host tier actually deliver, with all ranks reading?",
        target=ProbeTarget.BANDWIDTH,
        kinds=(TierKind.HOST,),
        harness=(
            "any sustained-read probe, as a new comm-suite arm; note that all "
            "three rank processes contend for one memory controller"
        ),
        unblocks=(
            "the host tier going MEASURED. It is load-bearing: 2.79 GB over "
            "38 GB/s = 73 ms/token is the floor for any RAM-resident expert "
            "tier, and that floor is currently an estimate"
        ),
    ),
    ProbeSpec(
        id="M5",
        label="NVMe read latency",
        question="What does one cold read cost, in microseconds?",
        target=ProbeTarget.LATENCY,
        kinds=(TierKind.FILESYSTEM,),
        harness="fio with an explicit iodepth; nothing of the kind exists in the tree",
        unblocks=(
            "the NVMe tier's latency class. #389's placement rule is a single "
            "size threshold today and would stay one without it"
        ),
        requires_property=("medium", "nvme"),
    ),
    ProbeSpec(
        id="M6",
        label="ZFS write-back behaviour for spill-to-disk",
        question="Is a write durable when the call returns, and is the page cache reclaimable IN TIME?",
        target=ProbeTarget.BANDWIDTH,
        kinds=(TierKind.FILESYSTEM,),
        harness=(
            "extend the a8a2f7bc22 harness, which settled the read side "
            "(posix_fadvise(DONTNEED) is a no-op for mapped pages and on this "
            "pool even post-unmap; madvise(MADV_PAGEOUT) reclaims to 0)"
        ),
        unblocks=(
            "whether an fs: tier may be a spill target on a swapless box. Boot "
            "8 died with memory.current at 98.3/98.5 GiB for seven minutes: "
            "reclaimable is not reclaimable in time"
        ),
        cost="S-M",
    ),
    ProbeSpec(
        id="M7",
        label="Rig-2 point reads, and cross-rig GPU-to-GPU",
        question="Can a byte go from one rig's card to the other's without the host?",
        target=ProbeTarget.LATENCY,
        kinds=(TierKind.DEVICE, TierKind.FILESYSTEM),
        harness="cross-rig bench from the PVE host; not runnable from this container",
        unblocks=(
            "the remote VRAM tier and the remote disk tier entirely. Not on the "
            "#407 critical path: the registry declares them absent and blocked "
            "and remains correct"
        ),
        cost="M",
        requires_verdict=("block",),
    ),
    ProbeSpec(
        id="M8",
        label="Memory-region registration cost",
        question="Can a remote tier be allocated per session, or must it be a static pool?",
        target=ProbeTarget.LATENCY,
        kinds=(TierKind.HOST,),
        harness="time cuMemCreate -> export -> two ioctls -> ibv_reg_dmabuf_mr",
        unblocks=(
            "per-session remote allocation. Amortised to zero for a static "
            "pool; for anything dynamic it is the whole design"
        ),
    ),
)

_PROBE_BY_ID = {p.id: p for p in PROBES}


def probe_by_id(probe_id: str) -> ProbeSpec:
    try:
        return _PROBE_BY_ID[probe_id]
    except KeyError:
        raise KeyError(
            f"no probe {probe_id!r}; declared: {sorted(_PROBE_BY_ID)}"
        ) from None


def probes_for(tier: TierDescriptor) -> Tuple[ProbeSpec, ...]:
    """Every probe that could take a number on this tier."""
    return tuple(p for p in PROBES if p.applies_to(tier))


class ProbeOutcome(msgspec.Struct, frozen=True, kw_only=True):
    """What a probe returned. All four statuses are data."""

    probe_id: str
    tier_id: TierId
    status: str = "absent"
    rate: Optional[Rate] = None
    reason: str = ""
    elapsed_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.status not in PROBE_STATUSES:
            raise ValueError(f"status {self.status!r} is not one of {PROBE_STATUSES}")
        if self.status != "ok" and not self.reason:
            raise ValueError(f"a {self.status!r} outcome must say why in one sentence")

    def to_json(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "tier_id": self.tier_id,
            "status": self.status,
            "value": None if self.rate is None else self.rate.or_none(),
            "provenance": None if self.rate is None else self.rate.provenance.value,
            "reason": self.reason,
            "elapsed_s": self.elapsed_s,
        }


class TierProbe(Protocol):
    """A thing that can take one measurement on one tier."""

    spec: ProbeSpec

    def measure(self, tier: TierDescriptor) -> ProbeOutcome: ...


class UnimplementedProbe:
    """The cut-1 stub: declares the measurement, refuses to invent it.

    It is not a placeholder that returns zero, and not one that raises either.
    An ``absent`` outcome IS the result -- it composes with the registry, it
    renders on the dashboard, and it names the harness that would replace it.
    """

    def __init__(self, spec: ProbeSpec) -> None:
        self.spec = spec

    def measure(self, tier: TierDescriptor) -> ProbeOutcome:
        return ProbeOutcome(
            probe_id=self.spec.id,
            tier_id=tier.id,
            status="absent",
            reason=(
                f"{self.spec.id} ({self.spec.label}) is declared and not "
                f"implemented: cut 1 ships the probe interface and no card "
                f"time. It would come from {self.spec.harness}"
            ),
        )


def run_probe(
    spec: ProbeSpec, tier: TierDescriptor, *, runner: Optional[TierProbe] = None
) -> ProbeOutcome:
    """Run ``spec`` against ``tier``, or record why it did not run.

    ``runner`` is injectable so a later cut adds a real probe without touching
    this dispatcher, and so the tests can drive every status.
    """
    if not spec.applies_to(tier):
        return ProbeOutcome(
            probe_id=spec.id,
            tier_id=tier.id,
            status="absent",
            reason=(
                f"{spec.id} applies to {[k.value for k in spec.kinds]} tiers; "
                f"{tier.id} is {tier.kind.value}"
            ),
        )
    probe = runner if runner is not None else UnimplementedProbe(spec)
    return probe.measure(tier)


class ProvenanceUpgradeRefused(RuntimeError):
    """Something tried to make a number look better measured than it is."""


def apply_outcome(tier: TierDescriptor, outcome: ProbeOutcome) -> TierDescriptor:
    """Fold a probe outcome into a tier record, or refuse and say why.

    The only writer of :class:`~sglang.srt.memtier.tiers.TierCaps` values.
    """
    spec = probe_by_id(outcome.probe_id)
    if outcome.tier_id != tier.id:
        raise ProvenanceUpgradeRefused(
            f"outcome names tier {outcome.tier_id}, record is {tier.id}"
        )
    if outcome.status != "ok":
        if outcome.rate is not None and not outcome.rate.is_absent:
            raise ProvenanceUpgradeRefused(
                f"{outcome.probe_id} on {tier.id}: a {outcome.status!r} outcome "
                f"carries a value ({outcome.rate.value}). A probe that did not "
                "succeed does not get to deliver a number."
            )
        return tier
    if outcome.rate is None:
        raise ProvenanceUpgradeRefused(
            f"{outcome.probe_id} on {tier.id}: an 'ok' outcome with no rate"
        )
    if outcome.rate.provenance is not Provenance.MEASURED:
        raise ProvenanceUpgradeRefused(
            f"{outcome.probe_id} on {tier.id}: an 'ok' outcome carries a "
            f"{outcome.rate.provenance.value} rate. A probe produces "
            "measurements; a formula over measured inputs is an estimate and "
            "belongs in the profile, not in a probe result."
        )
    current = _current_rate(tier.caps, spec.target)
    if current.provenance is Provenance.MEASURED and not _same_source(
        current, outcome.rate
    ):
        raise ProvenanceUpgradeRefused(
            f"{outcome.probe_id} on {tier.id}: {spec.target.value} is already "
            f"MEASURED ({current.source}). Replacing one measurement with "
            "another is a re-measurement and must be recorded as such -- "
            "carry the previous source forward, or clear it deliberately."
        )
    return tier.evolve(caps=_with_rate(tier.caps, spec.target, outcome.rate))


def _same_source(current: Rate, incoming: Rate) -> bool:
    return current.source == incoming.source


def _current_rate(caps: TierCaps, target: ProbeTarget) -> Rate:
    if target is ProbeTarget.LATENCY:
        return caps.latency_us
    if target is ProbeTarget.BANDWIDTH:
        return caps.bandwidth_gbs
    return caps.aperture_bytes


def _with_rate(caps: TierCaps, target: ProbeTarget, rate: Rate) -> TierCaps:
    if target is ProbeTarget.LATENCY:
        return msgspec.structs.replace(caps, latency_us=rate)
    if target is ProbeTarget.BANDWIDTH:
        return msgspec.structs.replace(caps, bandwidth_gbs=rate)
    return msgspec.structs.replace(caps, aperture_bytes=rate)


def require_measured(tier: TierDescriptor, target: ProbeTarget) -> float:
    """The value, or a refusal naming the probe that would produce it.

    This is the "refuse, do not assume" half of the interface: #286's rule
    that an unmeasured path is never assumed usable for parking, expressed as
    a call a consumer cannot forget to make.
    """
    rate = _current_rate(tier.caps, target)
    if not rate.is_absent:
        return rate.require(target.value)
    arms = [p.id for p in probes_for(tier) if p.target is target]
    raise ProvenanceUpgradeRefused(
        f"{tier.id}: {target.value} is absent ({rate.source}). "
        + (
            f"It would come from probe {', '.join(arms)}."
            if arms
            else "No declared probe produces it for this tier kind."
        )
    )


def missing_measurements(
    tiers: Iterable[TierDescriptor],
) -> List[Tuple[TierId, ProbeSpec]]:
    """Every (tier, probe) pair whose number is absent right now.

    The dashboard query: absences as a work list rather than as blank cells.
    """
    gaps: List[Tuple[TierId, ProbeSpec]] = []
    for tier in tiers:
        for spec in probes_for(tier):
            if _current_rate(tier.caps, spec.target).is_absent:
                gaps.append((tier.id, spec))
    return gaps
