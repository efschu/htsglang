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
"""#553 Cut 1 — one answer to "who can give bytes back, and how much".

WHY THIS EXISTS. ``ANALYSE_553_elastic_coresidence.md`` §3 names this as the
cut everything else addresses:

    "``vram_dial``'s ``DialParticipant`` list and #286's asset classes are the
     same question asked twice ('what can give bytes back, and how much').
     Either fold them, or give the register a query that returns dial
     participants as classes."

Today neither registry can see the other — nothing in the tree imports both —
so "free 4 GiB for the video tenant" has **no addressee**. A hot/cold event
has nothing to ask. This module is that addressee.

WHAT IT IS NOT. It does not move anything, does not choose a victim, and does
not fire an actuator. It ENUMERATES and RANKS, and it refuses by name. Keeping
the decision at the call site is deliberate: the actuators have very different
prices (a #704a rung change costs a full ~1575 ms arena refill; a GDN slot
vacate does not), and a module that both priced and pulled would hide that.

REFUSAL IS THE POINT, not a courtesy. Two constraints make a source
unavailable, and both must be visible rather than silently skipped:

  * **VA stability** (#93/#468). A class whose virtual addresses must survive
    the move cannot be moved while a captured graph holds those addresses —
    a park that frees and re-allocates invalidates the capture. The register
    already owns this rule in ``AssetClassDescriptor.va_stability_required``,
    which combines the class's own requirement with the one acquired from the
    route; this module asks it and never re-derives it.
  * **The dial's band.** The VRAM dial works BELOW the captured bound; above
    it, re-capture is not built. Bytes past the band are not reclaimable, they
    are unimplemented, and reporting them would hand a caller a number it
    cannot spend.

An excluded source is returned as an :class:`Unavailable` with its reason, not
dropped. A caller that sees an empty available list and an empty unavailable
list knows the registries were empty; one that sees an empty available list
and three reasons knows why — those are different facts and this module keeps
them apart.

PURE BY CONSTRUCTION. Both registry reads are injectable, so the ranking is
testable without CUDA, a model, or a graph. The defaults pull from the real
registries.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "ProbeUnavailable",
    "ReclaimSource",
    "Unavailable",
    "ReclaimView",
    "enumerate_reclaim_sources",
]

#: Origin registries, kept as strings so a reader of a log line can tell which
#: half of the bridge produced a row without importing either module.
ORIGIN_DIAL = "vram_dial"
ORIGIN_ASSET = "offload_register"


class ProbeUnavailable(RuntimeError):
    """A probe could not measure. Distinct from measuring zero.

    #606: a probe that fails must NOT be reported as 0 bytes available. Zero
    is a measurement ("this source is at its floor / holds nothing"); a failed
    probe is the absence of one. Collapsing them removes a real source from an
    elastic plan while looking like it was considered.
    """


@dataclasses.dataclass(frozen=True)
class ReclaimSource:
    """One thing that can give bytes back, and how many."""

    name: str
    origin: str
    #: Bytes this source could return RIGHT NOW, already clamped to whatever
    #: bound its own registry enforces. Never a hoped-for figure.
    reclaimable_bytes: int
    #: True when returning these bytes is reversible without a re-capture.
    #: A source that cannot be restored is not elastic, it is a one-way spend.
    restorable: bool
    #: Cost hint for ORDERING only. Not a measurement, and named so:
    #: a caller that needs a price measures it.
    cost_rank: int = 0

    def __post_init__(self):
        if self.reclaimable_bytes < 0:
            raise ValueError(
                f"{self.name}: reclaimable_bytes must not be negative "
                f"({self.reclaimable_bytes}); a source that owes bytes is a "
                f"bug in its registry, not a source with a small budget"
            )


@dataclasses.dataclass(frozen=True)
class Unavailable:
    """A source that exists but may not be drawn on, WITH the reason.

    Carried rather than filtered, because "nothing can give bytes" and "three
    things could but none may" are different states and a caller that cannot
    tell them apart will misreport the rig.
    """

    name: str
    origin: str
    reason: str


@dataclasses.dataclass(frozen=True)
class ReclaimView:
    """What the two registries jointly say at one instant."""

    available: Tuple[ReclaimSource, ...]
    unavailable: Tuple[Unavailable, ...]

    @property
    def total_reclaimable_bytes(self) -> int:
        return sum(s.reclaimable_bytes for s in self.available)

    def can_fund(self, want_bytes: int) -> bool:
        """Whether the ask is fundable from AVAILABLE sources alone.

        Deliberately not "fundable if we also moved the unavailable ones":
        that is the silent-partial answer, and #268 forbids it.
        """
        return self.total_reclaimable_bytes >= max(0, int(want_bytes))

    def plan_for(self, want_bytes: int) -> Optional[Tuple[ReclaimSource, ...]]:
        """Cheapest-first sources covering ``want_bytes``, or None.

        None rather than a partial list: a caller that receives fewer bytes
        than it asked for and proceeds anyway is the failure this returns None
        to prevent. The caller may still inspect ``available`` and decide to
        take what there is -- but it has to say so.
        """
        want = max(0, int(want_bytes))
        if want == 0:
            return ()
        if not self.can_fund(want):
            return None
        picked: List[ReclaimSource] = []
        got = 0
        for source in self.available:
            picked.append(source)
            got += source.reclaimable_bytes
            if got >= want:
                break
        return tuple(picked)


def _default_dial_participants():
    from sglang.srt.managers.vram_dial import get_dial_participants

    return get_dial_participants()


def _default_asset_classes():
    from sglang.srt.model_executor.short_term_offload_register import ASSET_CLASSES

    return ASSET_CLASSES


def dial_probe(floor_rows_for: Callable[[object], Optional[int]]):
    """#553 Cut 2: the real dial probe, given a floor authority.

    ``floor_rows_for`` is injected rather than looked up because this module
    must not become a second authority for the floor -- the dial derives it
    from a card-level NVML measurement and #584 says that number has one
    owner. A caller that has no floor gets a refusal, which is the honest
    answer and keeps the bridge's "no probe" path meaningful.

    Returns a callable suitable for ``dial_reclaimable_bytes``. It raises
    :class:`ProbeUnavailable` rather than returning 0 when the live read
    fails, so the caller refuses BY NAME instead of ranking a source at zero
    that was never measured (#606).
    """

    def _probe(participant) -> int:
        from sglang.srt.managers.vram_dial import reclaimable_bytes_for

        nbytes = reclaimable_bytes_for(participant, floor_rows_for(participant))
        if nbytes is None:
            raise ProbeUnavailable(
                "dial reclaimable bytes unreadable (unbooted pool, missing "
                "floor authority, or unmeasurable row width)"
            )
        return int(nbytes)

    return _probe


def asset_probe(register=None):
    """#553 Cut 2: the real per-class probe over the #286 register.

    Delegates to ``OffloadRegister.reclaimable_bytes(class)`` -- resident and
    NOT hot, computed under the register's own lock -- rather than summing
    items here. Same reason as the dial side: the register answers about
    itself.
    """

    def _probe(name: str, descriptor) -> int:
        reg = register
        if reg is None:
            from sglang.srt.model_executor.offload_register import get_global_register

            reg = get_global_register()
        if reg is None:
            raise ProbeUnavailable("no offload register on this process")
        fn = getattr(reg, "reclaimable_bytes", None)
        if fn is None:
            raise ProbeUnavailable(
                "offload register has no reclaimable_bytes accessor"
            )
        return int(fn(name))

    return _probe


def enumerate_reclaim_sources(
    *,
    graph_addressed: bool = False,
    dial_participants: Optional[Sequence] = None,
    asset_classes: Optional[dict] = None,
    dial_reclaimable_bytes: Optional[Callable[[object], int]] = None,
    asset_reclaimable_bytes: Optional[Callable[[str, object], int]] = None,
) -> ReclaimView:
    """Join both registries into one ranked, refusal-carrying view.

    ``graph_addressed`` is the #468 question and it changes answers: under a
    breakable route a captured graph holds the slot arena's device addresses,
    so a class that is freely movable under the eager offload becomes
    VA-pinned. It is a parameter rather than a global read because the caller
    knows which route it is on and this module must not guess.

    The two ``*_reclaimable_bytes`` hooks exist because neither registry
    publishes a byte figure directly -- the dial's is a function of the pool's
    rows above its floor, the register's of the class's live extent. Injecting
    them keeps this module pure and lets a test state the numbers outright.
    """
    dial = _default_dial_participants() if dial_participants is None else dial_participants
    classes = _default_asset_classes() if asset_classes is None else asset_classes

    available: List[ReclaimSource] = []
    unavailable: List[Unavailable] = []

    for idx, participant in enumerate(dial or ()):
        name = f"kv_pool[{idx}]{'/target' if getattr(participant, 'is_target', False) else '/draft'}"
        if dial_reclaimable_bytes is None:
            # No measurement available: refuse rather than assume zero OR
            # assume plenty. Both guesses are wrong in a way that hides.
            unavailable.append(
                Unavailable(
                    name=name,
                    origin=ORIGIN_DIAL,
                    reason=(
                        "no reclaimable-bytes probe supplied; the dial's "
                        "figure is rows-above-floor x row bytes and this "
                        "module will not invent it"
                    ),
                )
            )
            continue
        try:
            nbytes = int(dial_reclaimable_bytes(participant))
        except ProbeUnavailable as e:
            unavailable.append(
                Unavailable(name=name, origin=ORIGIN_DIAL, reason=str(e))
            )
            continue
        if nbytes <= 0:
            unavailable.append(
                Unavailable(
                    name=name,
                    origin=ORIGIN_DIAL,
                    reason="already at its floor inside the captured band",
                )
            )
            continue
        # The dial is reversible within the band by construction: it returns
        # VMM pages and can re-commit them without a re-capture.
        available.append(
            ReclaimSource(
                name=name,
                origin=ORIGIN_DIAL,
                reclaimable_bytes=nbytes,
                restorable=True,
                cost_rank=0,
            )
        )

    for key, descriptor in sorted((classes or {}).items()):
        name = str(key)
        try:
            pinned = bool(
                descriptor.va_stability_required(graph_addressed=graph_addressed)
            )
        except Exception as e:  # pragma: no cover - a descriptor that cannot
            # answer its own stability question is not assumed movable.
            unavailable.append(
                Unavailable(
                    name=name,
                    origin=ORIGIN_ASSET,
                    reason=f"va_stability_required raised ({e}); not assumed movable",
                )
            )
            continue
        if pinned:
            unavailable.append(
                Unavailable(
                    name=name,
                    origin=ORIGIN_ASSET,
                    reason=(
                        "VA-stable: a captured graph holds these addresses, so "
                        "a park that frees and re-allocates would invalidate "
                        "the capture (#93/#468)"
                    ),
                )
            )
            continue
        if asset_reclaimable_bytes is None:
            unavailable.append(
                Unavailable(
                    name=name,
                    origin=ORIGIN_ASSET,
                    reason="no reclaimable-bytes probe supplied for asset classes",
                )
            )
            continue
        try:
            nbytes = int(asset_reclaimable_bytes(name, descriptor))
        except ProbeUnavailable as e:
            unavailable.append(
                Unavailable(name=name, origin=ORIGIN_ASSET, reason=str(e))
            )
            continue
        if nbytes <= 0:
            unavailable.append(
                Unavailable(name=name, origin=ORIGIN_ASSET, reason="nothing resident")
            )
            continue
        available.append(
            ReclaimSource(
                name=name,
                origin=ORIGIN_ASSET,
                reclaimable_bytes=nbytes,
                restorable=True,
                cost_rank=1,
            )
        )

    # Cheapest first, then largest, then name -- the last key only so the
    # order is total and a test is not pinning dict iteration order.
    available.sort(key=lambda s: (s.cost_rank, -s.reclaimable_bytes, s.name))
    unavailable.sort(key=lambda u: (u.origin, u.name))
    return ReclaimView(tuple(available), tuple(unavailable))
