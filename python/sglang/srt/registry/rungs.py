# SPDX-License-Identifier: Apache-2.0
"""#305 cut 1: the residency LADDER view over the engine registry.

The registry (#333-M1) already tracks a residency state per engine
(``TenantState``: HOT / WARM_GPU / WARM_HOST / COLD), a ledger that decides
whether bytes are available, pin as a spec property, and a ``/v1/models``
listing that carries all of it. This module adds only what
``DESIGN_305_multi_model_serving.md`` asks for and the registry does not yet
answer:

1. **The fifth state.** "Registered, nothing staged" must be distinguishable
   from "cold but previously staged", because a client asking "do you know
   this model" and one asking "can you serve it now" are asking different
   questions and COLD answers both the same way.
2. **A promote-cost CLASS a-priori.** The registry fills
   ``measured_promotion_ms`` only after it has OBSERVED a transition, so a
   freshly registered engine tells a client nothing about what a first request
   will cost. The ladder's measured record supplies a class label until a real
   measurement replaces it -- and the two are kept in separate fields on
   purpose, so a class can never be read as a measurement.
3. **Cross-geometry labelling at REGISTRATION time.** An engine whose world
   geometry differs from the active one cannot be promoted by the cheap
   within-geometry instrument (#309); it needs a world re-formation (#329)
   with a 12-20 s floor. The design's rule is that this is known when the
   engine is registered, never discovered when a request arrives.

NOTHING HERE MOVES A MODEL. Rungs are declared and reported; transitions are
the later cuts. Per the #111 rule -- put the seam in, wire it later, and never
advertise what does not execute -- the promote-cost labels are explicitly
a-priori and the transition refusal names the cut that would implement it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

from sglang.srt.registry.ledger import GPU_RESIDENT_STATES, TenantState

__all__ = [
    "LADDER",
    "PromoteCost",
    "Rung",
    "coerce_bytes",
    "cross_geometry_label",
    "pin_refusal_reason",
    "promote_cost_of",
    "rung_extension",
    "rung_of",
    "transition_refusal",
]


class Rung(str):
    """The ladder vocabulary of the design, as reported to clients."""

    HOT = "HOT"
    TEIL_HOT = "TEIL_HOT"
    WARM = "WARM"
    COLD = "COLD"
    REGISTERED = "REGISTERED"


@dataclasses.dataclass(frozen=True)
class PromoteCost:
    """What reaching HOT from this rung costs, as an a-priori CLASS.

    ``seconds`` is the ladder's measured record (see the design's table and
    its sources), not a measurement of THIS engine. ``basis`` names where the
    number came from so a reader can check it, and ``measured`` is always
    False here -- a real per-engine figure lands in the registry's own
    ``measured_promotion_ms`` and must never be conflated with this.
    """

    rung: str
    seconds: str
    latency_class: str
    basis: str
    measured: bool = False


#: The ladder, with the measured entry costs from the design.
LADDER: Mapping[str, PromoteCost] = {
    Rung.HOT: PromoteCost(
        Rung.HOT, "0", "serving latency",
        "already resident; nothing to pay",
    ),
    Rung.TEIL_HOT: PromoteCost(
        Rung.TEIL_HOT, "<1", "one dial re-raise",
        "#297 KV delta move, target <1 s (DESIGN_297_kv_resharding.md:83)",
    ),
    Rung.WARM: PromoteCost(
        Rung.WARM, "3-6", "seconds; a request arrives to a rebuild",
        "graph recapture / weight flip, order 3-6 s (ANALYSE_363:112)",
    ),
    Rung.COLD: PromoteCost(
        Rung.COLD, "12-20", "effectively a boot",
        "#89 resume 8-14 s at uneven TP=3 (DESIGN_201:1635) plus 3-6 s "
        "recapture",
    ),
    Rung.REGISTERED: PromoteCost(
        Rung.REGISTERED, "12-20", "effectively a boot; nothing is staged",
        "same as COLD: a registered engine that was never staged pays the "
        "full cold start",
    ),
}


def coerce_bytes(value: Any) -> int:
    """Bytes as an int, from a scalar or a per-card mapping.

    The registry tracks reservations per card, so ``reserved_bytes`` is a
    Mapping on an instance and a scalar in a test fixture. Both are legitimate
    inputs to "is anything reserved"; anything unreadable is 0, because this
    only ever distinguishes REGISTERED from COLD and guessing "something is
    reserved" from an unparsable value would hide the fifth state.
    """
    if value is None:
        return 0
    if isinstance(value, Mapping):
        total = 0
        for v in value.values():
            try:
                total += int(v)
            except (TypeError, ValueError):
                continue
        return total
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def rung_of(
    state: Any,
    *,
    ever_staged: bool = True,
    reserved_bytes: Any = 0,
) -> str:
    """Map a registry ``TenantState`` onto the design's ladder.

    The registry's four states do not carry the design's fifth one, so it is
    derived: a COLD engine that was NEVER staged is ``REGISTERED``, a COLD
    engine that was staged before is ``COLD``. The distinction is not
    cosmetic -- the second has an image on disk to resume from, the first has
    only a config.

    ``WARM_GPU`` maps to ``TEIL_HOT``: it is the state that still holds device
    memory without serving, which is the design's "weights resident, pools
    reduced" rung. ``WARM_HOST`` maps to ``WARM`` (suspended to host).
    """
    value = getattr(state, "value", state)
    if value == TenantState.HOT.value:
        return Rung.HOT
    if value == TenantState.WARM_GPU.value:
        return Rung.TEIL_HOT
    if value == TenantState.WARM_HOST.value:
        return Rung.WARM
    if value == TenantState.COLD.value:
        if ever_staged or coerce_bytes(reserved_bytes):
            return Rung.COLD
        return Rung.REGISTERED
    raise ValueError(f"unknown residency state {state!r}")


def promote_cost_of(rung: str) -> PromoteCost:
    cost = LADDER.get(rung)
    if cost is None:
        raise ValueError(f"unknown rung {rung!r}")
    return cost


def cross_geometry_label(
    engine_geometry: Optional[str],
    active_geometry: Optional[str],
) -> Optional[dict]:
    """Label an engine whose world geometry differs from the active world.

    Returns ``None`` when the geometries match (or either is unknown -- an
    unknown geometry is NOT declared cross-geometry, because guessing that
    would attach a 12-20 s warning to every engine whose label an operator
    simply did not fill in).

    When they differ the label says so at REGISTRATION time, which is the
    design's rule: promoting such an engine is not a rung change but a world
    re-formation (#329), and a client or operator must be able to see that
    before a request discovers it.
    """
    if not engine_geometry or not active_geometry:
        return None
    if str(engine_geometry) == str(active_geometry):
        return None
    return {
        "cross_geometry": True,
        "engine_geometry": str(engine_geometry),
        "active_geometry": str(active_geometry),
        "instrument": "#329 world re-formation",
        "floor_seconds": "12-20",
        "reason": (
            "this engine's world geometry differs from the active world, so "
            "promoting it rebuilds the process group rather than moving a "
            "rung; the cheap within-geometry instrument (#309) does not apply"
        ),
    }


def pin_refusal_reason(
    *,
    rung: str,
    pinned: bool,
    can_fund: bool,
    ledger_detail: str = "",
    cross_geometry: bool = False,
) -> Optional[str]:
    """Why this pin cannot be honoured, or ``None`` if it can.

    The design's contract: **pin blocks demotion but cannot force promotion
    past the ledger, and an unhonourable pin fails AT PIN TIME.** A pin that
    is accepted and then silently cannot be kept is worse than a refusal --
    the operator believes a model is protected when it is not.

    Unpinning (``pinned=False``) is always honoured: removing a protection
    cannot fail.

    A pin on an engine that is not currently resident is a request to keep it
    resident once promoted, so it is refused when the ledger could not fund
    that promotion. A pin on an already-resident engine is honoured -- its
    bytes are already reserved, and the ledger has nothing to decide.
    """
    if not pinned:
        return None
    if rung in (Rung.HOT, Rung.TEIL_HOT):
        return None
    if cross_geometry:
        return (
            "cannot pin a cross-geometry engine: promoting it requires a "
            "world re-formation (#329), so residency cannot be guaranteed by "
            "a pin. Register it on the active world geometry, or pin it after "
            "the world has been re-formed."
        )
    if not can_fund:
        return (
            "cannot pin: the ledger cannot fund this engine's residency, so "
            "the pin could not be honoured"
            + (f" ({ledger_detail})" if ledger_detail else "")
            + ". A pin that is accepted and then cannot be kept is worse than "
            "a refusal, so this fails at pin time rather than at the first "
            "demotion."
        )
    return None


def transition_refusal(target_rung: str) -> str:
    """The named refusal for a rung change this cut does not implement.

    #111's rule: the seam exists, the movement does not, and the error says
    which cut would provide it rather than failing generically or -- worse --
    reporting success for something that did not happen.
    """
    if target_rung in (Rung.HOT, Rung.TEIL_HOT):
        return (
            f"rung movement to {target_rung} is not implemented in #305 cut 1 "
            "(registry + reporting only). The HOT<->TEIL_HOT transition is "
            "cut 2, via the #330 dial under the #309 quiesce boundary."
        )
    if target_rung == Rung.WARM:
        return (
            "rung movement to WARM is not implemented in #305 cut 1; it is "
            "cut 3, via #89 suspend-to-RAM."
        )
    return (
        f"rung movement to {target_rung} is not implemented in #305 cut 1; "
        "the COLD rung and registry persistence are cut 5."
    )


def rung_extension(
    state: Any,
    *,
    ever_staged: bool = True,
    reserved_bytes: Any = 0,
    pinned: bool = False,
    engine_geometry: Optional[str] = None,
    active_geometry: Optional[str] = None,
) -> dict:
    """The ladder fields for an engine's ``x-htsglang`` block.

    Kept separate from the registry's own fields so nothing here can overwrite
    a measured number: ``promote_cost_class`` is a-priori and says so, while
    the registry's ``measured_promotion_ms`` stays the only measured figure.
    """
    rung = rung_of(state, ever_staged=ever_staged, reserved_bytes=reserved_bytes)
    cost = promote_cost_of(rung)
    block: dict = {
        "rung": rung,
        "gpu_resident": (
            getattr(state, "value", state) in {s.value for s in GPU_RESIDENT_STATES}
        ),
        "pinned": bool(pinned),
        "promote_cost_class": {
            "seconds": cost.seconds,
            "latency_class": cost.latency_class,
            "basis": cost.basis,
            "measured": cost.measured,
        },
    }
    cross = cross_geometry_label(engine_geometry, active_geometry)
    if cross is not None:
        block["geometry"] = cross
    return block
