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
"""Scenario results, per window, and the four-valued verdict over them.

A scenario (:mod:`sglang.srt.planner.scenarios`) says what to measure. This
module holds what came back and decides what may be said about it. Two rules
shape everything here, and both exist because this project has been fooled by
their absence.

**Results are per window, never pooled.** A figure averaged across a state
change describes neither state. The worked example is the spill path: the
window right after a restore contains the speculative-resume backfill, so the
throughput there is a catch-up rate, not a serving rate. Windows the scenario
marked ``exclude_from_headline`` are therefore structurally barred from a
headline figure — :func:`headline` refuses them rather than trusting a caller
to remember, and ``restore_transient`` is the standing case.

**A comparison has four possible answers, not two.** "Faster" and "slower"
leave no room for the two verdicts that are actually most common on a small
rig:

``changed``          the difference exceeds the measured noise floor
``within_noise``     it does not, so nothing may be claimed
``not_comparable``   the two arms did not measure the same thing
``unknown``          no noise floor, or a missing measurement

``within_noise`` and ``unknown`` are different statements, and collapsing them
is how "no effect" gets reported for a run that never had the resolution to
see one. ``not_comparable`` is the verdict this rig needs most: throughput
follows the output content (r = 0.90), so two arms that generated different
text are not comparable at all, and under speculation a round time taken at a
different accept length describes a different amount of work.

The noise floor is an input, never a default. There is no built-in threshold
here: it comes from the ``noise_floor`` scenario's A-vs-A run, and without one
every comparison returns ``unknown`` naming that as the reason.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "CHANGED",
    "WITHIN_NOISE",
    "NOT_COMPARABLE",
    "UNKNOWN",
    "VERDICTS",
    "WindowResult",
    "ArmResult",
    "NoiseFloor",
    "Comparison",
    "headline",
    "compare_metric",
    "compare_arms",
]

CHANGED = "changed"
WITHIN_NOISE = "within_noise"
NOT_COMPARABLE = "not_comparable"
UNKNOWN = "unknown"

#: The complete vocabulary. A verdict outside it is a bug, not a nuance.
VERDICTS = (CHANGED, WITHIN_NOISE, NOT_COMPARABLE, UNKNOWN)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WindowResult:
    """What one measurement window produced.

    ``excluded_from_headline`` is copied from the scenario's window
    declaration rather than decided here, so the rule lives with the
    measurement design and travels with the data.
    """

    window: str
    metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    #: Number of samples behind the figures, per metric where it differs.
    samples: Dict[str, int] = dataclasses.field(default_factory=dict)
    excluded_from_headline: bool = False
    note: str = ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ArmResult:
    """One arm of a comparison: one configuration, measured window by window.

    ``conditions`` carries everything that has to MATCH for two arms to be
    comparable — accept length, chunk size, resident tokens, prompt set, the
    build. :func:`compare_arms` reads it and returns ``not_comparable`` when it
    differs, which is the check that would have caught this project's
    content-variance findings before they were reported.
    """

    label: str
    scenario: str = ""
    windows: List[WindowResult] = dataclasses.field(default_factory=list)
    conditions: Dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def window(self, key: str) -> Optional[WindowResult]:
        for w in self.windows:
            if w.window == key:
                return w
        return None

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "scenario": self.scenario,
            "windows": [w.to_json() for w in self.windows],
            "conditions": dict(self.conditions),
            "provenance": dict(self.provenance),
        }


@dataclasses.dataclass
class NoiseFloor:
    """Per-metric detection thresholds, as relative spread (0.02 = 2 %).

    Per metric on purpose: ms per verify round and raw tok/s differ by 10-30x
    on this rig, and a single floor taken from the coarser one would discard
    real effects on the finer one.
    """

    relative: Dict[str, float] = dataclasses.field(default_factory=dict)
    source: str = ""

    def for_metric(self, key: str) -> Optional[float]:
        return self.relative.get(key)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------


class HeadlineRefused(Exception):
    """Raised when a headline figure was asked for from an excluded window."""


def headline(arm: ArmResult, window: str, metric: str) -> float:
    """One headline figure, from one window, for one metric.

    Refuses an excluded window rather than returning its number with a
    caveat attached: a caveat next to a figure is read less often than the
    figure is, and the transient windows this guards contain a catch-up rate
    that is not a serving rate.
    """
    w = arm.window(window)
    if w is None:
        raise HeadlineRefused(
            f"{arm.label}: no window {window!r} was measured"
        )
    if w.excluded_from_headline:
        raise HeadlineRefused(
            f"{arm.label}: window {window!r} is excluded from headline figures "
            "by the scenario that declared it. It contains a transition, so "
            "its figures describe neither state. Report it beside the steady "
            "windows, not as the result."
        )
    if metric not in w.metrics:
        raise HeadlineRefused(
            f"{arm.label}: window {window!r} has no {metric!r}"
        )
    return w.metrics[metric]


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Comparison:
    """One metric, one window, two arms — and what may be said about it."""

    metric: str
    window: str
    verdict: str
    reason: str
    baseline: Optional[float] = None
    candidate: Optional[float] = None
    relative_delta: Optional[float] = None
    threshold: Optional[float] = None
    direction: str = ""  # "better" | "worse" | ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


#: Conditions whose mismatch makes two arms incomparable rather than merely
#: different. Each is a term that changes the amount of work behind a figure.
BLOCKING_CONDITIONS = (
    "accept_length",
    "chunk_size",
    "resident_tokens",
    "batch_size",
    "prompt_set",
    "model",
    "quant",
)

#: How far a blocking condition may drift and still count as the same
#: condition. Accept length is measured, so it is never bit-identical between
#: two runs; a percent of drift is not a different experiment, ten is.
CONDITION_TOLERANCE = 0.02


def _condition_mismatch(a: ArmResult, b: ArmResult) -> Optional[str]:
    for key in BLOCKING_CONDITIONS:
        if key not in a.conditions or key not in b.conditions:
            continue
        va, vb = a.conditions[key], b.conditions[key]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if va == 0 and vb == 0:
                continue
            base = max(abs(va), abs(vb))
            if base and abs(va - vb) / base > CONDITION_TOLERANCE:
                return (
                    f"{key} differs between the arms ({va} vs {vb}), so the two "
                    "figures describe different amounts of work"
                )
        elif va != vb:
            return (
                f"{key} differs between the arms ({va!r} vs {vb!r}), so the two "
                "runs did not measure the same thing"
            )
    return None


def compare_metric(
    baseline: ArmResult,
    candidate: ArmResult,
    metric: str,
    window: str,
    noise: Optional[NoiseFloor] = None,
    lower_is_better: bool = True,
) -> Comparison:
    """Compare one metric in one window across two arms.

    Order of decision matters: comparability is checked BEFORE the numbers,
    because two arms that measured different work produce a difference that is
    real and meaningless at the same time.
    """
    mismatch = _condition_mismatch(baseline, candidate)
    if mismatch:
        return Comparison(
            metric=metric,
            window=window,
            verdict=NOT_COMPARABLE,
            reason=mismatch,
        )

    wa, wb = baseline.window(window), candidate.window(window)
    if wa is None or wb is None:
        missing = baseline.label if wa is None else candidate.label
        return Comparison(
            metric=metric,
            window=window,
            verdict=UNKNOWN,
            reason=f"{missing} has no window {window!r}",
        )
    if wa.excluded_from_headline or wb.excluded_from_headline:
        return Comparison(
            metric=metric,
            window=window,
            verdict=NOT_COMPARABLE,
            reason=(
                f"window {window!r} is a transition window; the figures in it "
                "describe neither the state before nor the state after, so "
                "they do not compare"
            ),
        )
    if metric not in wa.metrics or metric not in wb.metrics:
        return Comparison(
            metric=metric,
            window=window,
            verdict=UNKNOWN,
            reason=f"{metric!r} was not measured in both arms",
        )

    va, vb = wa.metrics[metric], wb.metrics[metric]
    if not math.isfinite(va) or not math.isfinite(vb) or va == 0:
        return Comparison(
            metric=metric,
            window=window,
            verdict=UNKNOWN,
            reason="a measurement is missing or zero, so no ratio exists",
            baseline=va,
            candidate=vb,
        )

    delta = (vb - va) / abs(va)
    threshold = noise.for_metric(metric) if noise else None
    if threshold is None:
        return Comparison(
            metric=metric,
            window=window,
            verdict=UNKNOWN,
            reason=(
                f"no noise floor for {metric!r}. Run the noise_floor scenario "
                "(A vs A, interleaved) first; without a floor there is no "
                "threshold below which a difference must not be reported, and "
                "'no difference' cannot be told apart from 'no resolution'."
            ),
            baseline=va,
            candidate=vb,
            relative_delta=delta,
        )

    if abs(delta) <= threshold:
        return Comparison(
            metric=metric,
            window=window,
            verdict=WITHIN_NOISE,
            reason=(
                f"{delta * 100:+.2f} % is inside the {threshold * 100:.2f} % "
                f"noise floor for {metric!r}; nothing may be claimed"
            ),
            baseline=va,
            candidate=vb,
            relative_delta=delta,
            threshold=threshold,
        )

    improved = (delta < 0) if lower_is_better else (delta > 0)
    return Comparison(
        metric=metric,
        window=window,
        verdict=CHANGED,
        reason=(
            f"{delta * 100:+.2f} % against a {threshold * 100:.2f} % noise "
            f"floor for {metric!r}"
        ),
        baseline=va,
        candidate=vb,
        relative_delta=delta,
        threshold=threshold,
        direction="better" if improved else "worse",
    )


def compare_arms(
    baseline: ArmResult,
    candidate: ArmResult,
    scenario=None,
    noise: Optional[NoiseFloor] = None,
    metrics: Optional[Sequence[str]] = None,
) -> List[Comparison]:
    """Compare two arms across every window and metric they share.

    When a ``scenario`` is given, the metric directions come from its
    declarations, so a lower-is-better metric is not reported as a regression
    for having gone down. Excluded windows are still compared — and still come
    back ``not_comparable`` — rather than being dropped, so their absence from
    the result cannot be mistaken for their having agreed.
    """
    directions: Dict[str, bool] = {}
    if scenario is not None:
        for m in getattr(scenario, "metrics", []):
            directions[m.key] = m.direction != "higher_better"

    windows = [w.window for w in baseline.windows]
    out: List[Comparison] = []
    for window in windows:
        wa = baseline.window(window)
        wb = candidate.window(window)
        keys = list(metrics or (wa.metrics if wa else {}))
        for metric in keys:
            if wb is not None and metric not in wb.metrics and metrics is None:
                continue
            out.append(
                compare_metric(
                    baseline,
                    candidate,
                    metric,
                    window,
                    noise=noise,
                    lower_is_better=directions.get(metric, True),
                )
            )
    return out
