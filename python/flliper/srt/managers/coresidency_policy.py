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
"""#553 Cut 3 — tenant hot/cold events ACTUATE, instead of only pricing.

Cuts 1 and 2 built the bridge and its probes: a hot/cold event finally has an
addressee that can answer "who can give bytes back, and how much". Nothing
called it. This is the caller, and it introduces NO new actuator -- every move
below is an existing authority's own API:

  * ``vram_dial.DialRuntime.apply_budget_request`` -- the replicated grow /
    shrink. It already enforces the floor and its rejections "carry the exact
    floor arithmetic and change nothing", so this module never re-derives a
    floor and never second-guesses a refusal.
  * ``vram_dial.verify_pool_reached_capacity`` -- the read-back. Existing,
    and the reason the hot path can MEASURE rather than assert.
  * ``GdnSlotRuntime.unbind`` -- the slot vacate (#364).
  * ``coresidency_registry.enumerate_reclaim_sources`` -- Cuts 1+2.

THE TWO FAILURE DIRECTIONS ARE NOT SYMMETRIC, and the code is shaped by that:

  * **Shrink must never strand bytes unaccounted.** Every source this module
    draws on is recorded in the returned :class:`ActuationResult` with what it
    was asked for and what it reported giving. A source that was asked and did
    not answer is carried as a failure, never dropped -- bytes that left one
    ledger and entered none are the shape that goes unnoticed for weeks.
  * **Grow must never exceed the floor.** Not enforced here: enforced by the
    dial, which refuses below its floor with the arithmetic. This module's job
    is to not paper over that refusal.

#217 IS THE LESSON THAT SHAPES THE HOT PATH. A restore that "came back" was
measured at 23% of its target. So a grow here is followed by a read-back, and
the result reports the MEASURED capacity, not the requested one. A caller that
wants to know whether the tenant is warm again reads ``reached``, never
``requested``.

#694 COUNTER-VS-ACTUATOR: a count is a promise only if the same call
delivered it. Nothing in this module increments a "reclaimed" total from a
plan; totals come from what each actuator reported.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, List, Optional, Tuple

__all__ = [
    "ActuationStep",
    "ActuationResult",
    "cold_event",
    "hot_event",
]


@dataclasses.dataclass(frozen=True)
class ActuationStep:
    """One actuator call and what it actually reported."""

    name: str
    origin: str
    requested_bytes: int
    #: What the actuator SAID it delivered. None means it was asked and did
    #: not answer -- distinct from 0, which is a delivered nothing.
    delivered_bytes: Optional[int]
    ok: bool
    detail: str = ""

    @property
    def stranded(self) -> bool:
        """Asked, and no accounting came back. The shape that goes unnoticed."""
        return self.ok and self.delivered_bytes is None


@dataclasses.dataclass(frozen=True)
class ActuationResult:
    steps: Tuple[ActuationStep, ...]
    #: Only what actuators REPORTED delivering (#694: a count is a promise
    #: only if the same call delivered it).
    delivered_bytes: int
    #: For a hot event: the capacity a read-back MEASURED, never the request
    #: (#217 -- a restore that "came back" measured 23% of target).
    reached_bytes: Optional[int] = None
    refused: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.refused is None and all(s.ok for s in self.steps)

    @property
    def stranded_steps(self) -> Tuple[ActuationStep, ...]:
        return tuple(s for s in self.steps if s.stranded)


def cold_event(
    want_bytes: int,
    *,
    view,
    release_fn: Callable[[Any], Tuple[bool, Optional[int], str]],
) -> ActuationResult:
    """A tenant went cold: draw ``want_bytes`` from the bridge's ranked plan.

    ``view`` is a :class:`~coresidency_registry.ReclaimView`. ``release_fn``
    performs ONE source's release and returns ``(ok, delivered_bytes, detail)``
    -- ``delivered_bytes=None`` meaning "did not report", which this module
    surfaces as stranded rather than counting as zero.

    REFUSES RATHER THAN PARTIALLY DRAWING. ``plan_for`` returns None when the
    ask is not fundable from AVAILABLE sources, and this returns that refusal
    untouched. Taking what there is would be the silent-partial the bridge was
    built to prevent (#268), and the caller can still inspect ``view`` and ask
    for less on purpose.
    """
    plan = view.plan_for(want_bytes)
    if plan is None:
        return ActuationResult(
            steps=(),
            delivered_bytes=0,
            refused=(
                f"cold event refused: {want_bytes} bytes wanted, only "
                f"{view.total_reclaimable_bytes} available across "
                f"{len(view.available)} source(s); "
                f"{len(view.unavailable)} source(s) are unavailable and their "
                f"bytes do not count toward the ask"
            ),
        )

    steps: List[ActuationStep] = []
    delivered = 0
    for source in plan:
        try:
            ok, got, detail = release_fn(source)
        except Exception as e:  # pragma: no cover - defensive
            steps.append(
                ActuationStep(
                    name=source.name,
                    origin=source.origin,
                    requested_bytes=source.reclaimable_bytes,
                    delivered_bytes=None,
                    ok=False,
                    detail=f"release raised: {e}",
                )
            )
            continue
        steps.append(
            ActuationStep(
                name=source.name,
                origin=source.origin,
                requested_bytes=source.reclaimable_bytes,
                delivered_bytes=got,
                ok=bool(ok),
                detail=detail,
            )
        )
        if ok and got is not None:
            delivered += int(got)
    return ActuationResult(steps=tuple(steps), delivered_bytes=delivered)


def hot_event(
    want_bytes: int,
    *,
    grow_fn: Callable[[int], Tuple[bool, str]],
    measure_fn: Optional[Callable[[], Optional[int]]] = None,
) -> ActuationResult:
    """A tenant went hot: ask the dial to grow, then MEASURE what it reached.

    ``grow_fn`` is the dial's own request (``apply_budget_request``), which
    already refuses below its floor with the arithmetic. A refusal is returned
    as-is: this module does not retry it smaller, because a floor refusal is
    a statement about the rig, not a negotiation.

    ``measure_fn`` is the read-back (``verify_pool_reached_capacity`` or an
    equivalent). Its answer -- not the request -- is what ``reached_bytes``
    reports. #217: a restore that "came back" was measured at 23% of target,
    so "grew" is a claim only a measurement can make.

    A missing ``measure_fn`` leaves ``reached_bytes`` None, which reads as
    "not measured" rather than "reached nothing". Those differ and the caller
    must be able to tell (#606).
    """
    ok, detail = grow_fn(int(want_bytes))
    step = ActuationStep(
        name="dial",
        origin="vram_dial",
        requested_bytes=int(want_bytes),
        delivered_bytes=None,
        ok=bool(ok),
        detail=detail,
    )
    if not ok:
        return ActuationResult(
            steps=(step,),
            delivered_bytes=0,
            refused=f"hot event refused by the dial: {detail}",
        )

    reached = None
    if measure_fn is not None:
        try:
            measured = measure_fn()
            reached = None if measured is None else int(measured)
        except Exception as e:  # pragma: no cover - defensive
            return ActuationResult(
                steps=(step,),
                delivered_bytes=0,
                refused=(
                    f"grow was accepted but the read-back failed ({e}); the "
                    f"tenant must not be reported warm on an unverified grow"
                ),
            )
    return ActuationResult(
        steps=(step,), delivered_bytes=0, reached_bytes=reached
    )
