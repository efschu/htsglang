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

#: The inverse. Built from the forward table rather than typed out, so the two
#: vocabularies cannot drift apart by an edit to one of them.
STATE_TO_RUNG: Dict[TenantState, str] = {v: k for k, v in RUNG_TO_STATE.items()}

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


def declare_class(
    name: str,
    rungs: FrozenSet[str] | Tuple[str, ...] | List[str],
    *,
    absent_because: Dict[str, str] | None = None,
    replace: bool = False,
) -> None:
    """Declare the rungs a class outside the three shipped ones implements.

    The three built-in classes are declared literally above and pinned against
    their own refusal text; this is for a fourth class -- or a test double --
    that is not in this file. It exists so the default stays refusal: an
    undeclared class has no rungs and is turned away by :func:`rungs_for`,
    rather than being assumed to have all four, which is precisely the
    assumption the #305 determination found to be wrong on this rig.

    Re-declaring an existing class needs ``replace=True``: a second import
    silently changing what a class can do is how a table stops describing the
    code it names.

    ``absent_because`` must name a reason for EVERY rung the class omits. The
    three shipped classes each state theirs, and a refusal that could only say
    "not implemented" would be the generic no this module exists to avoid.
    """
    wanted = frozenset(rungs)
    unknown = wanted - set(RUNG_ORDER)
    if unknown:
        raise LadderRefusal(
            f"class {name!r} declares rung(s) {sorted(unknown)} that are not on "
            f"the ladder; known rungs: {list(RUNG_ORDER)}"
        )
    if HOT not in wanted or COLD not in wanted:
        raise LadderRefusal(
            f"class {name!r} must declare both ends of the ladder ({HOT} and "
            f"{COLD}); a class that spans neither end is not on the ladder"
        )
    if name in CLASS_RUNGS and not replace and CLASS_RUNGS[name] != wanted:
        raise LadderRefusal(
            f"class {name!r} is already declared with rungs "
            f"{sorted(CLASS_RUNGS[name])}; pass replace=True to change it"
        )
    reasons = dict(absent_because or {})
    missing = [r for r in RUNG_ORDER if r not in wanted and not reasons.get(r)]
    if missing:
        raise LadderRefusal(
            f"class {name!r} omits rung(s) {missing} without stating why; pass "
            "absent_because={rung: reason} for each. A refusal that can only "
            "say 'not implemented' is the generic no this table exists to "
            "replace."
        )
    CLASS_RUNGS[name] = wanted
    for rung, why in reasons.items():
        RUNG_ABSENT_BECAUSE[(name, rung)] = why


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
        (s, d)
        for s in RUNG_ORDER
        for d in RUNG_ORDER
        if s != d and s in have and d in have
    )


def rung_of_state(state: TenantState) -> str:
    """Stored state -> promise vocabulary. The edge conversion, one place."""
    try:
        return STATE_TO_RUNG[state]
    except KeyError:
        raise LadderRefusal(
            f"state {state!r} is on no rung of the #305 ladder; known states: "
            f"{sorted(s.value for s in STATE_TO_RUNG)}"
        ) from None


def step_down_target(klass: str, src: str) -> str | None:
    """The next LOWER rung this class can actually reach from ``src``.

    Not simply ``RUNG_ORDER[i + 1]``: on this rig the adjacent rung is often
    the one the class refuses (Class 1 has no WARM, Class 3 has neither middle
    rung), and a walker that assumed adjacency would stall one rung above the
    bottom forever. So this scans downward for the first rung the class
    declares, and the edge it returns is by construction one :func:`can`
    accepts. ``None`` means there is no lower rung at all -- ``src`` is the
    floor for this class, or the class declares nothing below it.

    It never invents an edge: if the class lacks ``src`` itself, this refuses
    via :func:`rungs_for` rather than guessing where the tenant is.
    """
    have = rungs_for(klass)
    if src not in have:
        raise LadderRefusal(
            f"{klass}: cannot step down from {src}, which is not a rung this "
            f"class has. Rungs it does have: {sorted(have)}."
        )
    for candidate in RUNG_ORDER[RUNG_ORDER.index(src) + 1 :]:
        if candidate in have:
            return candidate
    return None


def skipped_rungs(klass: str, src: str, dst: str) -> Tuple[str, ...]:
    """Rungs strictly between ``src`` and ``dst`` that this class does not have.

    A step that jumps a rung is legitimate here -- the intermediate one does
    not exist for this class -- but it should never be silent, so a caller can
    name what it stepped over and why.
    """
    lo, hi = RUNG_ORDER.index(src), RUNG_ORDER.index(dst)
    if lo > hi:
        lo, hi = hi, lo
    have = rungs_for(klass)
    return tuple(r for r in RUNG_ORDER[lo + 1 : hi] if r not in have)


def describe(klass: str) -> str:
    have = sorted(rungs_for(klass), key=RUNG_ORDER.index)
    missing = [r for r in RUNG_ORDER if r not in rungs_for(klass)]
    lines = [f"{klass}: rungs {have}"]
    for r in missing:
        lines.append(
            f"  no {r}: {RUNG_ABSENT_BECAUSE.get((klass, r), 'not implemented')}"
        )
    return "\n".join(lines)
