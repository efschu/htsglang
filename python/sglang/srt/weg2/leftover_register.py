"""#1494 / Task #51: which D weight bands STAY on a card while P prefills.

THE IDEA (user, 2026-09-19 21:35Z; memory FLIP-27B-LEFTOVER-3080): the P->D
flip spends most of its wake moving D's weight bands back over the BAR1 lanes.
On the 3080s the P layout does not use the whole card, so some of those bands
never had to leave. A band left RESIDENT is a leg that is not run at all --
"liegen lassen" before "BAR1 Diff" before "Host zuletzt".

MEASURED INPUT (boot weg2xsn408, seq=9, the last P->D flip; every figure from
that boot's own lines, see DESIGN_LEFTOVER_27B_0920.md for the derivation):

    card                 D rank  P rank  free while P prefills  D band family
    GPU-5c648f96 (3080)  TP1     PP1     ~8775 MiB              9046 MiB
    GPU-62dbbae1 (3080)  TP2     PP2     ~2945 MiB              8644 MiB
    GPU-31d7ef41 (5090)  TP0     PP0     ~2285 MiB             12382 MiB

The 5090 falls out on its own arithmetic -- it is the card P fills -- which is
why the user's note says 3080s. Nothing here hardcodes a card model: a card
qualifies by fitting, or it does not qualify.

FOUR RULES, all of them one-sided:

1. ONLY P->D. A band stays resident THROUGH P's phase so that D's wake does
   not have to fetch it. The D->P direction is untouched.
2. D's KV IS NEVER A CANDIDATE. "D-KV nie opfern": the kv_cache tag is D's
   serving capacity, it is the largest tag on every rank, and a wake that
   found its weights resident but its KV pool shrunk would have traded the
   thing that matters for the thing that is fast. Weight bands only.
3. THE COST TO P IS A PLANNER TERM, NEVER A RESERVE. A resident leftover is
   subtracted from the budget P is planned with, exactly like any other
   occupant of the card. It is not held back, not rounded up, and nothing is
   kept "just in case" ("keine Korridor-Reserve, nie").
4. IF THAT SUBTRACTION PUSHES P UNDER WHAT THE NEEDLE NEEDS, THE LEFTOVER IS
   REFUSED BY NAME (W116) AND THE BAND TRAVELS. A faster flip that cannot
   prefill 262k is not a faster flip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: The tag that may never be left resident at P's expense (rule 2). Named
#: rather than pattern-matched: a rename must break a test, not a boot.
KV_TAG = "kv_cache"

#: Tags that are not weight bands and never qualify. `cuda_graph` is captured
#: against the layout that is awake, so a resident one is not reusable.
NON_BAND_TAGS = frozenset({KV_TAG, "cuda_graph"})


@dataclass(frozen=True)
class Band:
    """One D weight band on one card, with the leg that carries it today."""

    tag: str
    #: The tag's bytes on D's side -- what staying resident COSTS the card.
    resident_mib: int
    #: Milliseconds this band's collect leg took in the measured flip. The
    #: SAVING, and the reason the register is worth anything.
    leg_ms: int = 0
    #: Which BAR1 lane runs that leg. Legs on one lane are serial, legs on
    #: different lanes are not, so the saving is per LANE, never a sum.
    lane: str = ""


@dataclass(frozen=True)
class Card:
    """One card's arithmetic while P prefills on it."""

    uuid: str
    #: What is free on this card during P's phase, with P's own transient
    #: prefill budget ALREADY taken out. A leftover is placed against this.
    free_while_p_mib: int
    #: What P's planner may not fall below on this card for the needle
    #: (262k context, user order KONTEXT-262K-PFLICHT). 0 disables the check
    #: for this card -- an unknown requirement is not a refusal.
    p_needle_floor_mib: int = 0
    #: P's planned budget on this card BEFORE any leftover is subtracted.
    p_budget_mib: int = 0


@dataclass(frozen=True)
class CardPlan:
    uuid: str
    resident: Tuple[str, ...]
    travelling: Tuple[str, ...]
    resident_mib: int
    #: P's budget after the planner term is subtracted (rule 3).
    p_budget_after_mib: int
    #: Milliseconds off this card's critical path: the per-lane sum of the
    #: resident bands' legs, maxed over lanes and compared to the lane that
    #: now dominates. Never a sum over lanes.
    saved_ms: int
    refusals: Tuple[str, ...] = ()


def _lane_totals(bands: Sequence[Band]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for b in bands:
        out[b.lane] = out.get(b.lane, 0) + int(b.leg_ms)
    return out


def critical_path_ms(bands: Sequence[Band]) -> int:
    """The card's leg time: the BUSIEST lane, not the sum of all of them.

    The measured shape says why this matters. On TP1 at seq=9 the p0 lane
    carried 1282 ms and p5 carried 1190 ms; the rank's own WAKE-TAIL read
    leg_collects=1645 ms. Summing the lanes would have claimed 2472 ms of
    savings available where at most ~1282 exist.
    """
    totals = _lane_totals(bands)
    return max(totals.values()) if totals else 0


def eligible_bands(bands: Sequence[Band]) -> List[Band]:
    """Rule 2 in one place: weight bands only, KV and graph never."""
    return [b for b in bands if b.tag not in NON_BAND_TAGS and int(b.resident_mib) > 0]


def plan_card(card: Card, bands: Sequence[Band]) -> CardPlan:
    """Choose this card's leftovers: most milliseconds saved per MiB first.

    The order is deliberate and it is NOT "biggest band first". A band earns
    its residency by the leg it removes, and the measured legs are nothing
    like proportional to their bytes -- TP1's `weights_7` cost 495 ms for 825
    MiB while `weights_0` cost 248 ms for the same 825 MiB, because the legs
    are WAIT-bound (wait_ms 479 of 495), not bandwidth-bound. Sorting by
    ms/MiB therefore buys roughly twice the flip time per resident MiB that
    sorting by size would.
    """
    refusals: List[str] = []
    chosen: List[Band] = []
    budget = int(card.free_while_p_mib)
    p_after = int(card.p_budget_mib)
    floor = int(card.p_needle_floor_mib)

    def value(b: Band) -> Tuple[float, int]:
        mib = max(1, int(b.resident_mib))
        return (-(float(b.leg_ms) / mib), -int(b.leg_ms))

    for band in sorted(eligible_bands(bands), key=value):
        cost = int(band.resident_mib)
        if cost > budget:
            continue
        if floor and card.p_budget_mib and (p_after - cost) < floor:
            refusals.append(
                f"W116 Weg2LeftoverRefused card={card.uuid} tag={band.tag}: keeping "
                f"{cost} MiB resident would leave P {p_after - cost} MiB, under the "
                f"{floor} MiB the needle needs on this card. The band travels. A "
                f"faster flip that cannot prefill the needle is not a faster flip, "
                f"and this budget is a PLANNER TERM, not a reserve -- nothing is "
                f"held back, the band is simply not left behind."
            )
            continue
        chosen.append(band)
        budget -= cost
        p_after -= cost

    resident = {b.tag for b in chosen}
    before = critical_path_ms(bands)
    after = critical_path_ms([b for b in bands if b.tag not in resident])
    return CardPlan(
        uuid=card.uuid,
        resident=tuple(b.tag for b in chosen),
        travelling=tuple(b.tag for b in bands if b.tag not in resident),
        resident_mib=sum(int(b.resident_mib) for b in chosen),
        p_budget_after_mib=p_after,
        saved_ms=max(0, before - after),
        refusals=tuple(refusals),
    )


def plan_leftovers(cards: Sequence[Card],
                   bands_by_card: Dict[str, Sequence[Band]]) -> Dict[str, CardPlan]:
    """One plan per card. Cards are independent: a band is on the card it is on."""
    return {c.uuid: plan_card(c, bands_by_card.get(c.uuid, ())) for c in cards}


def legs_to_skip(plan: Optional[CardPlan]) -> frozenset:
    """The tags whose P->D collect leg must NOT run (diff instead of transfer).

    An absent plan skips nothing. That is the safe direction in both senses:
    running a leg for a band that happens to be resident costs time, while
    skipping one for a band that is not resident costs the weights.
    """
    return frozenset(plan.resident) if plan else frozenset()


def format_plan(plan: CardPlan) -> str:
    return (
        f"WEG2-LEFTOVER card={plan.uuid} resident={list(plan.resident)} "
        f"({plan.resident_mib} MiB, P planned with that much less) "
        f"travelling={list(plan.travelling)} saves={plan.saved_ms} ms off this "
        f"card's busiest lane (instrument: the measured collect legs of the "
        f"reference flip, maxed per lane -- never summed across lanes, because "
        f"lanes run at the same time); refusals={len(plan.refusals)}"
    )
