# SPDX-License-Identifier: Apache-2.0
"""#305: which ladder transitions are REACHABLE, per engine class.

NOTHING HERE MOVES A MODEL -- the same rule ``rungs.py`` states for itself.
This is a declaration of which edges exist, so a caller learns that a
transition is unbuilt BEFORE it drives an actuator, rather than discovering it
from an ``AdapterError`` mid-promotion.

WHY IT IS NEEDED, from the sweep that produced it: **no engine class has all
four rungs**, and the refusals are stated architecture rather than oversight:

* Class 1 (sglang engine) refuses ``WARM_HOST`` -- *"Class 1 has no WARM_HOST
  rung (§4.3) -- sharded, quantised, post-processed weights are not a"*
  reloadable image (``adapters/class1_srt.py:220-224``).
* Class 2 (diffusion) refuses ``WARM_GPU`` -- *"Class 2 exposes no WARM_GPU
  endpoint today (§5.3 names the rung; the upstream server has no route to drop
  just the BCG pool)"* (``adapters/class2_diffusion.py:276-279``).
* Class 3 (utility) refuses ``WARM_GPU`` (``adapters/class3_utility.py:88-93``).

**Consequence, and it is the sharpest fact in the #305 determination:
TEIL-HOT <-> WARM is absent for EVERY class.** Not unwired -- structurally
never attempted, because each class refuses the rung the other implements. A
ladder walk that assumes the four rungs form a chain is wrong on this rig.

TERMINOLOGY, because the promise and the code use different words. #305 promised
HOT / TEIL-HOT / WARM / COLD; the ledger stores ``TenantState``
(``ledger.py:80-84``) as HOT / WARM_GPU / WARM_HOST / COLD, and ``rungs.py``
maps ``WARM_GPU -> TEIL_HOT``, ``WARM_HOST -> WARM``. This module speaks the
promise's vocabulary and converts at the edge, so a reader of #305 can find it.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

from sglang.srt.registry.ledger import TenantState

#: Promise vocabulary -> stored state. One direction only; see the module note.
HOT = "HOT"
TEIL_HOT = "TEIL_HOT"
WARM = "WARM"
COLD = "COLD"

RUNG_TO_STATE: Dict[str, TenantState] = {
    HOT: TenantState.HOT,
    TEIL_HOT: TenantState.WARM_GPU,
    WARM: TenantState.WARM_HOST,
    COLD: TenantState.COLD,
}

#: Descending residency. Adjacency is NOT assumed to imply reachability -- that
#: is exactly the mistake TEIL_HOT <-> WARM invites.
RUNG_ORDER: Tuple[str, ...] = (HOT, TEIL_HOT, WARM, COLD)

#: Which rungs each adapter class actually implements. DECLARED here and PINNED
#: against the adapters' own refusal messages by test, so the table cannot
#: quietly drift from the code it describes.
CLASS_RUNGS: Dict[str, FrozenSet[str]] = {
    "class1_srt": frozenset({HOT, TEIL_HOT, COLD}),
    "class2_diffusion": frozenset({HOT, WARM, COLD}),
    "class3_utility": frozenset({HOT, COLD}),
}

#: Why a class lacks a rung. Named so a refusal can quote the architecture
#: rather than just saying no.
RUNG_ABSENT_BECAUSE: Dict[Tuple[str, str], str] = {
    ("class1_srt", WARM): (
        "Class 1 has no WARM_HOST rung (§4.3): sharded, quantised, "
        "post-processed weights are not a reloadable host image"
    ),
    ("class2_diffusion", TEIL_HOT): (
        "Class 2 exposes no WARM_GPU endpoint (§5.3): the upstream server has "
        "no route to drop just the BCG pool"
    ),
    ("class3_utility", TEIL_HOT): (
        "Class 3 has no WARM_GPU rung (§6.3): its ladder is HOT / COLD"
    ),
    ("class3_utility", WARM): (
        "Class 3 has no WARM_HOST rung (§6.3): its ladder is HOT / COLD"
    ),
}


class LadderRefusal(ValueError):
    """An unbuilt transition, refused before any actuator is driven."""


def rungs_for(klass: str) -> FrozenSet[str]:
    if klass not in CLASS_RUNGS:
        raise LadderRefusal(
            f"unknown engine class {klass!r}; known: {sorted(CLASS_RUNGS)}. A "
            "class with no declared rungs is refused rather than assumed to "
            "have all four."
        )
    return CLASS_RUNGS[klass]


def can(klass: str, src: str, dst: str) -> bool:
    """Is this transition implemented for this class?"""
    have = rungs_for(klass)
    return src in have and dst in have and src != dst


def check_transition(klass: str, src: str, dst: str) -> None:
    """Refuse an unbuilt transition, naming WHICH rung is missing and why."""
    have = rungs_for(klass)
    if src == dst:
        raise LadderRefusal(f"{klass}: {src} -> {dst} is not a transition.")
    for rung, role in ((src, "source"), (dst, "target")):
        if rung not in have:
            why = RUNG_ABSENT_BECAUSE.get((klass, rung), "not implemented")
        else:
            continue
        raise LadderRefusal(
            f"{klass}: {src} -> {dst} is unreachable -- the {role} rung "
            f"{rung} does not exist for this class. {why}. Rungs it does "
            f"have: {sorted(have)}."
        )


def universally_absent() -> Tuple[Tuple[str, str], ...]:
    """Edges no class implements. The #305 determination's headline.

    Computed rather than asserted, so it stays true if a class ever gains a
    rung: the day any adapter implements both TEIL_HOT and WARM, this returns
    empty and the determination's central claim is retired by arithmetic.
    """
    out: List[Tuple[str, str]] = []
    for src in RUNG_ORDER:
        for dst in RUNG_ORDER:
            if src == dst:
                continue
            if not any(can(k, src, dst) for k in CLASS_RUNGS):
                out.append((src, dst))
    return tuple(out)


def reachable_edges(klass: str) -> Tuple[Tuple[str, str], ...]:
    have = rungs_for(klass)
    return tuple(
        (s, d) for s in RUNG_ORDER for d in RUNG_ORDER if s != d and s in have and d in have
    )


def describe(klass: str) -> str:
    have = sorted(rungs_for(klass), key=RUNG_ORDER.index)
    missing = [r for r in RUNG_ORDER if r not in rungs_for(klass)]
    lines = [f"{klass}: rungs {have}"]
    for r in missing:
        lines.append(
            f"  no {r}: {RUNG_ABSENT_BECAUSE.get((klass, r), 'not implemented')}"
        )
    return "\n".join(lines)
