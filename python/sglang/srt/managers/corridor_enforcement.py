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
"""#584: a boot may not certify a corridor it did not check.

WHAT THIS ENFORCES, AND WHY IT IS NOT THE CORRIDOR GUARD. ``corridor_guard``
polices ALLOCATIONS at runtime: it arms, it reclaims, it refuses one request.
This module polices the CONFIGURATION, once, before serving: given what this
boot predicts each card will rest at, is that inside the band the operator
declared -- and, the question this chain keeps getting wrong, IS THE
PREDICTION WORTH ANYTHING.

THE TWO FAILURE DIRECTIONS ARE BOTH FAILURES. The corridor law reads "~1024
MiB free per card, and best-filled". A boot that rests 300 MiB below the band
breaches it; a boot that rests 2 GiB above it is holding VRAM that buys no
tokens and serves no request. Both are refused here. Measured on this rig:

* ``31800,14000,15600`` rested at 3545/5210/4325 MiB -- ~12.8 GiB idle;
* ``31800,19000,19000`` rests at 1763/2572/2891 MiB -- better, still over.

Neither was ever refused by anything, because nothing was asking.

THE THIRD VERDICT IS THE POINT OF THE MODULE. A configuration whose resting
position cannot be PREDICTED is not thereby compliant. This tree has no
solver that produces a per-rank memory budget across PIPELINE stages under a
live NVML floor: ``planner/key_solver`` and ``distributed/corridor_vector``
both work across the ranks of one TP/DCP group and have no ``pp_size``
concept, ``--rank-kv-ratio corridor`` cannot engage at ``tp_size == 1``, and
``--rank-tp-ratio auto-performance`` is refused outright under ``pp_size > 1``.
So on this topology the honest answer to "where will each card rest" is *we
do not know*, and :data:`Verdict.UNVERIFIABLE` says exactly that.

UNVERIFIABLE REFUSES. It is not a warning and it does not degrade to a pass.
The whole silent-falseness family this corpus collects has one shape --
something that could not answer returned the answer that looks like success:
the corridor audit that never armed and therefore never reported a breach; the
seam-reserve record that made a cold boot look like a measured one; the KV
rung that read a configured row count and called it physical backing. Each was
green while being blind. A gate that passes when it cannot see would be the
next member rather than the thing that catches them.

WHAT A REFUSAL COSTS is a boot that does not start, which is recoverable in
minutes. What a pass costs is a serving instance whose operator believes a law
is being held. That asymmetry is why the default here is strict.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "CORRIDOR-ENFORCE"


class Provenance(enum.Enum):
    """Where a number came from, ordered by how much weight it can bear.

    THE ORDER IS LOAD-BEARING. A verdict is only as good as its weakest
    input, so :func:`weakest` reduces a set of them and the gate reads the
    result -- which is what stops a boot from inheriting the confidence of its
    best-sourced number.
    """

    #: Read from this rig, this boot, this card. NVML's free column.
    MEASURED = "measured"
    #: Produced by a solver that modelled this configuration.
    SOLVED = "solved"
    #: An operator typed it. True by assertion, not by measurement.
    CONFIGURED = "configured"
    #: A constant that nobody chose for this rig.
    DEFAULTED = "defaulted"
    #: Nothing produced it. The honest answer when no solver covers this
    #: topology -- see the module docstring on the PP-budget gap.
    UNKNOWN = "unknown"


_STRENGTH = {
    Provenance.MEASURED: 4,
    Provenance.SOLVED: 3,
    Provenance.CONFIGURED: 2,
    Provenance.DEFAULTED: 1,
    Provenance.UNKNOWN: 0,
}


def weakest(sources: Iterable[Provenance]) -> Provenance:
    """The weakest provenance in a set, or UNKNOWN when the set is empty.

    An empty set is UNKNOWN rather than MEASURED: a verdict with no inputs has
    not been checked, and the empty case is exactly the one a refactor
    produces by accident.
    """
    items = list(sources)
    if not items:
        return Provenance.UNKNOWN
    return min(items, key=lambda p: _STRENGTH[p])


class Verdict(enum.Enum):
    IN_BAND = "in_band"
    BELOW_FLOOR = "below_floor"
    ABOVE_CEILING = "above_ceiling"
    #: The prediction could not be produced, so the configuration has not been
    #: checked. REFUSES; see the module docstring.
    UNVERIFIABLE = "unverifiable"

    @property
    def is_pass(self) -> bool:
        return self is Verdict.IN_BAND


@dataclasses.dataclass(frozen=True)
class CardPrediction:
    """What one card is predicted to rest at, and who says so."""

    card: str
    predicted_free_mib: Optional[int]
    provenance: Provenance
    #: Free-form, and it appears verbatim in a refusal. A refusal that cannot
    #: say WHY it could not check is a refusal an operator will disable.
    detail: str = ""


@dataclasses.dataclass(frozen=True)
class CardVerdict:
    card: str
    verdict: Verdict
    predicted_free_mib: Optional[int]
    provenance: Provenance
    detail: str

    def describe(self, floor: int, ceiling: int) -> str:
        if self.verdict is Verdict.UNVERIFIABLE:
            return f"{self.card}: NOT CHECKED ({self.provenance.value})" + (
                f" -- {self.detail}" if self.detail else ""
            )
        assert self.predicted_free_mib is not None
        where = {
            Verdict.IN_BAND: "in band",
            Verdict.BELOW_FLOOR: f"BELOW the {floor} MiB floor",
            Verdict.ABOVE_CEILING: f"ABOVE the {ceiling} MiB ceiling (idle VRAM)",
        }[self.verdict]
        return (
            f"{self.card}: {self.predicted_free_mib} MiB {where} "
            f"[{self.provenance.value}]"
        )


class CorridorConfigRefused(Exception):
    """A configuration that does not hold the band, or was never checked."""


def classify(
    prediction: CardPrediction, floor_mib: int, ceiling_mib: int, slack_mib: int = 0
) -> CardVerdict:
    """One card's verdict.

    ``slack_mib`` widens the CEILING only. Some headroom above the band is
    declared and legitimate -- a measured cutover draw that has to land
    somewhere -- and a gate with no way to declare it would be turned off the
    first time it refused a configuration the operator had reasons for. It
    does NOT widen the floor: the floor is the band's own tolerance already,
    and a tolerance on a tolerance is how a law stops meaning anything.
    """
    if prediction.provenance is Provenance.UNKNOWN or (
        prediction.predicted_free_mib is None
    ):
        return CardVerdict(
            prediction.card,
            Verdict.UNVERIFIABLE,
            prediction.predicted_free_mib,
            Provenance.UNKNOWN,
            prediction.detail,
        )
    free = int(prediction.predicted_free_mib)
    if free < int(floor_mib):
        verdict = Verdict.BELOW_FLOOR
    elif free > int(ceiling_mib) + max(0, int(slack_mib)):
        verdict = Verdict.ABOVE_CEILING
    else:
        verdict = Verdict.IN_BAND
    return CardVerdict(
        prediction.card, verdict, free, prediction.provenance, prediction.detail
    )


def evaluate(
    predictions: Sequence[CardPrediction],
    *,
    floor_mib: int,
    ceiling_mib: int,
    slack_mib: int = 0,
) -> Tuple[Verdict, List[CardVerdict]]:
    """The whole configuration's verdict, and every card's.

    THE GROUP VERDICT IS THE WORST CARD, not an average and not a majority.
    The corridor law is stated per card and a fleet is only as lawful as the
    card that breaks it; averaging is how five breaches on one card hid behind
    two comfortable ones in the #656 acceptance.

    An empty prediction set is UNVERIFIABLE, never a pass.
    """
    cards = [classify(p, floor_mib, ceiling_mib, slack_mib) for p in predictions]
    if not cards:
        return Verdict.UNVERIFIABLE, cards
    order = [
        Verdict.UNVERIFIABLE,
        Verdict.BELOW_FLOOR,
        Verdict.ABOVE_CEILING,
        Verdict.IN_BAND,
    ]
    for verdict in order:
        if any(c.verdict is verdict for c in cards):
            return verdict, cards
    return Verdict.IN_BAND, cards


def enforce(
    predictions: Sequence[CardPrediction],
    *,
    floor_mib: int,
    ceiling_mib: int,
    slack_mib: int = 0,
    strict: bool = True,
) -> Tuple[Verdict, List[CardVerdict]]:
    """Evaluate and REFUSE anything that is not a pass.

    ``strict=False`` logs the same verdict and returns instead of raising. It
    exists for the one honest case -- a rig bringing a new topology up, where
    refusing every boot would stop the very measurements that would let the
    gate answer. It is not the default, and it says so in the log every time,
    because a permanently non-strict gate is a gate that has been turned off.
    """
    verdict, cards = evaluate(
        predictions,
        floor_mib=floor_mib,
        ceiling_mib=ceiling_mib,
        slack_mib=slack_mib,
    )
    lines = [c.describe(floor_mib, ceiling_mib) for c in cards]
    body = (
        f"corridor band {floor_mib}-{ceiling_mib} MiB"
        + (f" (+{slack_mib} MiB declared slack above)" if slack_mib else "")
        + ": "
        + "; ".join(lines)
    )
    if verdict.is_pass:
        logger.info("%s PASS -- %s", LOG_PREFIX, body)
        return verdict, cards

    why = {
        Verdict.UNVERIFIABLE: (
            "this configuration was NOT CHECKED. A resting position that "
            "cannot be predicted is not thereby compliant, and passing here "
            "would make this gate the next member of the family it exists to "
            "catch -- an instrument that was green because it was blind."
        ),
        Verdict.BELOW_FLOOR: (
            "this configuration rests BELOW the corridor band, so the law is "
            "broken before a single request arrives."
        ),
        Verdict.ABOVE_CEILING: (
            "this configuration rests ABOVE the corridor band: VRAM held free "
            "that buys no tokens and serves no request. 'Best-filled' is half "
            "of the law and too much free is the same defect as too little."
        ),
    }[verdict]
    message = f"{LOG_PREFIX} {verdict.value.upper()}: {why} -- {body}"
    if not strict:
        logger.error("%s (NOT REFUSED: strict=False)", message)
        return verdict, cards
    raise CorridorConfigRefused(message)


# -- the planner side -------------------------------------------------------


def predict_from_planner(server_args, cards: Sequence[str]) -> List[CardPrediction]:
    """Ask the planner where each card will rest. Usually: it cannot say.

    THIS FUNCTION IS MOSTLY A HONEST NO, and that is the finding it encodes
    rather than a placeholder. Searched in the shipping tree:

    * ``planner/key_solver.solve`` (#272) and
      ``distributed/corridor_vector.solve_corridor_vector`` (#602) solve
      across the ranks of ONE TP/DCP group; neither has a ``pp_size`` concept.
    * ``--rank-kv-ratio corridor|capacity|speed`` needs
      ``dcp_size == tp_size > 1``; a ``--tp-size 1 --pp-size 3`` boot makes
      ``uneven_dcp_active()`` false, so the solver is never reached.
    * ``--rank-tp-ratio auto-performance`` is refused outright under
      ``--pp-size > 1``: "the measured hardware ladder plans one TP group, not
      per-stage groups".
    * ``planner/pp_cut.solve_pp_cut`` (#485) IS pipeline-aware and already
      carries ``corridor_mib`` defaulting to 1024 as a hard per-rank floor --
      but it solves the LAYER CUT given budgets, needs a residency census from
      a prior real boot, and is reachable only via ``--pp-solve-cut``.
    * nothing anywhere predicts at-rest free VRAM per card from a config
      offline; ``_corridor_local_capacity`` does it only at a live
      post-weight-load barrier.

    So on a pipeline topology this returns UNKNOWN with the gap named, and the
    gate refuses rather than certifying a corridor nobody computed. That is
    the correct behaviour for a missing capability: the boot stops and someone
    builds the solver, instead of the fleet quietly running unchecked.

    ``solve_pp_cut`` is the natural host for the missing solve, which is why
    the refusal names it.
    """
    pp_size = int(getattr(server_args, "pp_size", 1) or 1)
    if pp_size > 1:
        detail = (
            "no solver in this tree produces a per-rank MEMORY BUDGET across "
            f"PIPELINE stages (pp_size={pp_size}) under a live NVML floor; "
            "key_solver and corridor_vector are TP/DCP-group only, "
            "--rank-kv-ratio corridor cannot engage at tp_size=1, and "
            "auto-performance is refused under pp_size>1. planner/pp_cut."
            "solve_pp_cut is PP-aware and already carries corridor_mib as a "
            "hard floor -- it is the natural home for this solve"
        )
        return [
            CardPrediction(card, None, Provenance.UNKNOWN, detail) for card in cards
        ]
    return [
        CardPrediction(
            card,
            None,
            Provenance.UNKNOWN,
            "no offline at-rest prediction exists for this topology either",
        )
        for card in cards
    ]


def predictions_from_measurement(
    free_mib_by_card: Dict[str, int], detail: str = ""
) -> List[CardPrediction]:
    """Predictions sourced from a real reading of this rig.

    The strongest provenance there is, and the only one this rig can currently
    produce for the resting position -- which is why the acceptance loop is
    boot, measure, judge, rather than solve, certify, boot.
    """
    return [
        CardPrediction(card, int(free), Provenance.MEASURED, detail)
        for card, free in free_mib_by_card.items()
    ]


__all__ = [
    "LOG_PREFIX",
    "CardPrediction",
    "CardVerdict",
    "CorridorConfigRefused",
    "Provenance",
    "Verdict",
    "classify",
    "enforce",
    "evaluate",
    "predict_from_planner",
    "predictions_from_measurement",
    "weakest",
]
