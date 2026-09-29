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
"""Lanes: the wizard's unit of "one group with one goal on one card set".

WHY THIS EXISTS BEFORE THE FEATURE DOES
---------------------------------------
Today every configuration the wizard describes has exactly one lane. The
multi-group runtime (#274) is being built alongside it, and DESIGN_201
PRIO-Nachtrag 8 is explicit about the shape it has to arrive into: it is a
MULTI-group runtime, not a dual-group one. N lanes, each with its own key,
its own spread direction and its own priority class, over shared bytes where
the nesting algebra allows.

The requirement that lands here is (a) of that Nachtrag, and it is a
requirement about DATA STRUCTURES rather than about behaviour:

    no two-way hardcoding in structures or signatures -- ``lane_id`` rather
    than "the lane", group LISTS rather than pair parameters.

So this module carries a :class:`LaneSet` that is a list from the first line,
and every wizard payload that describes a topology carries one. A one-lane
configuration is a :class:`LaneSet` of length one, not a special case with a
different shape, and the split control of #258 will add a second entry rather
than a second field.

PRIORITY IS A CLASS, NOT A RANK ORDER
-------------------------------------
Nachtrag 5 gives PD precedence over the main group at compute and at VRAM;
Nachtrag 8 (d) generalises that to priority CLASSES over N lanes. A class is
therefore a named tier -- :data:`PRIORITY_CLASSES`, ordered, most privileged
first -- and not an integer that invites arithmetic. Two lanes may share a
class; that is a statement that neither preempts the other, which is a real
configuration and not an error.

WHAT A LANE IS NOT
------------------
It is not a launch command and it is not a process. A lane says which cards
it may use, what it is spread toward and how it ranks against the others.
Whether that becomes a second process, an in-process second group or (today)
simply the only group is the runtime's business.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "PRIORITY_CLASSES",
    "FOREGROUND",
    "SCAVENGER",
    "LANE_ROLES",
    "Lane",
    "LaneSet",
    "single_lane",
    "lanes_from_card_groups",
]

#: The privileged class: guaranteed compute and reserved VRAM. Nachtrag 5's
#: "PD is always prioritised", stated as a class so it is not tied to PD.
FOREGROUND = "foreground"
#: The work-conserving follower: uses what lies idle, yields at the natural
#: grain boundaries (chunk / decode step) the moment a foreground lane has
#: work. Anything it borrows must be evacuable.
SCAVENGER = "scavenger"

#: Ordered, most privileged first. Membership is checked; the ORDER is what
#: "wins in a conflict" means, and it is read from this tuple rather than
#: from an integer stored per lane.
PRIORITY_CLASSES: Tuple[str, ...] = (FOREGROUND, SCAVENGER)

#: What a lane is FOR. Open vocabulary in spirit -- a spread direction is a
#: goal, not a role -- but these four are the ones the design names, and an
#: unknown role is refused so a typo does not become a new kind of lane.
LANE_ROLES: Tuple[str, ...] = ("main", "pd", "drafter", "side")


@dataclasses.dataclass(frozen=True)
class Lane:
    """One group: which cards, what it optimises for, how it ranks.

    ``goal`` is a wizard target key (``max_decode``, ``max_kv``, ...) or the
    empty string when the lane is not spread toward anything in particular.
    It is deliberately the SAME vocabulary the wizard's targets use, because
    "this lane is tuned for decode" and "I want decode most" have to be the
    same word or the two pages will disagree about what was asked for.

    ``cards`` are card ORDINALS in the wizard's own card set, not CUDA
    indices: the wizard's set is what the user ticked, and CUDA order is not
    stable across ``CUDA_DEVICE_ORDER`` settings. Two lanes naming the same
    ordinal are co-resident on that card, which is a fact the caller can read
    off :meth:`LaneSet.co_residence` rather than having to recompute.
    """

    lane_id: str
    label: str
    cards: Tuple[int, ...]
    goal: str = ""
    priority_class: str = FOREGROUND
    role: str = "main"
    note: str = ""

    def __post_init__(self):
        if not self.lane_id:
            raise ValueError("a lane needs an id: lanes are addressed by id")
        if self.priority_class not in PRIORITY_CLASSES:
            raise ValueError(
                f"unknown priority class {self.priority_class!r}; one of "
                f"{', '.join(PRIORITY_CLASSES)}"
            )
        if self.role not in LANE_ROLES:
            raise ValueError(
                f"unknown lane role {self.role!r}; one of "
                f"{', '.join(LANE_ROLES)}"
            )
        object.__setattr__(self, "cards", tuple(int(c) for c in self.cards))

    @property
    def priority_rank(self) -> int:
        """Position in :data:`PRIORITY_CLASSES`. Lower wins a conflict."""
        return PRIORITY_CLASSES.index(self.priority_class)

    def to_json(self) -> dict:
        return {
            "lane_id": self.lane_id,
            "label": self.label,
            "cards": list(self.cards),
            "goal": self.goal,
            "priority_class": self.priority_class,
            "priority_rank": self.priority_rank,
            "role": self.role,
            "note": self.note,
        }

    @classmethod
    def from_json(cls, d: dict) -> Lane:
        return cls(
            lane_id=str(d.get("lane_id") or ""),
            label=str(d.get("label") or ""),
            cards=tuple(int(c) for c in (d.get("cards") or [])),
            goal=str(d.get("goal") or ""),
            priority_class=str(d.get("priority_class") or FOREGROUND),
            role=str(d.get("role") or "main"),
            note=str(d.get("note") or ""),
        )


@dataclasses.dataclass(frozen=True)
class LaneSet:
    """N lanes. One is the normal case today; nothing here assumes it.

    The invariants are checked once, here, rather than at every consumer:
    ids are unique, the set is non-empty, and every lane is a valid lane.
    """

    lanes: Tuple[Lane, ...]

    def __post_init__(self):
        lanes = tuple(self.lanes)
        if not lanes:
            raise ValueError(
                "a lane set describes at least one group; an empty set is not "
                "a configuration"
            )
        seen: Dict[str, int] = {}
        for lane in lanes:
            if lane.lane_id in seen:
                raise ValueError(
                    f"duplicate lane_id {lane.lane_id!r}: lanes are addressed "
                    "by id, so two lanes cannot share one"
                )
            seen[lane.lane_id] = 1
        object.__setattr__(self, "lanes", lanes)

    # -- list behaviour, so callers iterate rather than unpack a pair -------

    def __len__(self) -> int:
        return len(self.lanes)

    def __iter__(self):
        return iter(self.lanes)

    def __getitem__(self, i):
        return self.lanes[i]

    def by_id(self, lane_id: str) -> Optional[Lane]:
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane
        return None

    def by_priority(self) -> List[Lane]:
        """Most privileged first, insertion order kept within a class."""
        return sorted(
            self.lanes, key=lambda ln: (ln.priority_rank, self.lanes.index(ln))
        )

    def cards(self) -> Tuple[int, ...]:
        """Every card any lane may use, sorted, each once."""
        out: List[int] = []
        for lane in self.lanes:
            for c in lane.cards:
                if c not in out:
                    out.append(c)
        return tuple(sorted(out))

    def co_residence(self) -> Dict[int, List[str]]:
        """``card ordinal -> the lanes on it``, for every shared card.

        Cards used by a single lane are omitted: the interesting fact is
        sharing, and a map where every card appears makes the shared ones
        harder to see, not easier. Elastic borrowing (Nachtrag 4) and
        card-local KV sharing (Nachtrag 6) are defined per lane PAIR on a
        shared card, and this is where such a pair is found -- pairwise over
        a set, never a distinguished pair in a signature.
        """
        by_card: Dict[int, List[str]] = {}
        for lane in self.lanes:
            for c in lane.cards:
                by_card.setdefault(c, []).append(lane.lane_id)
        return {c: ids for c, ids in sorted(by_card.items()) if len(ids) > 1}

    def sharing_pairs(self) -> List[Tuple[str, str, int]]:
        """``(lane_a, lane_b, card)`` for every pair sharing a card.

        Derived from :meth:`co_residence`, so N lanes on one card yield every
        pair rather than the first two.
        """
        out: List[Tuple[str, str, int]] = []
        for card, ids in self.co_residence().items():
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    out.append((a, b, card))
        return out

    def to_json(self) -> dict:
        return {
            "lanes": [ln.to_json() for ln in self.lanes],
            "count": len(self.lanes),
            "priority_classes": list(PRIORITY_CLASSES),
            "cards": list(self.cards()),
            "co_residence": {
                str(c): ids for c, ids in self.co_residence().items()
            },
            "multi_lane": len(self.lanes) > 1,
            "note": (
                "One lane today. The structure is a list because the "
                "multi-group runtime is a MULTI-group runtime (DESIGN_201 "
                "PRIO-Nachtrag 8): the split control adds an entry here, it "
                "does not add a second field."
                if len(self.lanes) == 1
                else (
                    f"{len(self.lanes)} lanes. In a conflict the earlier "
                    "priority class wins; a lane may only borrow from a more "
                    "privileged one what it can evacuate."
                )
            ),
        }

    @classmethod
    def from_json(cls, d) -> LaneSet:
        rows = d.get("lanes") if isinstance(d, dict) else d
        return cls(tuple(Lane.from_json(r) for r in (rows or [])))


def single_lane(
    cards: Sequence[int],
    *,
    goal: str = "",
    label: str = "the serving group",
    lane_id: str = "main",
    role: str = "main",
    note: str = "",
) -> LaneSet:
    """The one-lane set every family the wizard offers today describes.

    A helper rather than a default argument, so the single-lane case is a
    CALL and shows up in a grep for lane construction, instead of being what
    happens when nobody says anything.
    """
    return LaneSet(
        (
            Lane(
                lane_id=lane_id,
                label=label,
                cards=tuple(cards),
                goal=goal,
                priority_class=FOREGROUND,
                role=role,
                note=note,
            ),
        )
    )


def lanes_from_card_groups(
    groups: Iterable[Sequence[int]],
    *,
    goals: Optional[Sequence[str]] = None,
    roles: Optional[Sequence[str]] = None,
    priorities: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
) -> LaneSet:
    """N card groups -> N lanes, ids ``lane0..laneN``.

    Used by the island families, which produce their lane count from the
    topology they were handed rather than from a constant.
    """
    out: List[Lane] = []
    for i, group in enumerate(groups):
        out.append(
            Lane(
                lane_id=f"lane{i}",
                label=(labels[i] if labels and i < len(labels) else f"lane {i}"),
                cards=tuple(group),
                goal=(goals[i] if goals and i < len(goals) else ""),
                priority_class=(
                    priorities[i]
                    if priorities and i < len(priorities)
                    else FOREGROUND
                ),
                role=(roles[i] if roles and i < len(roles) else "main"),
            )
        )
    return LaneSet(tuple(out))
