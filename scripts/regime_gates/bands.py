#!/usr/bin/env python3
"""#363 card gate 3 -- per-signal A-vs-A bands from two identical boots.

Gate 3 of DESIGN_363 section 11.7: measure, on this rig, how much each
classifier input differs between two runs that should be the same, and then
check every provisional constant of section 3.4 against its own signal's
band. A threshold inside its own noise is a coin flip with a number written
next to it.

The judgements go through the CONTROLLER'S OWN ``signal_band`` and
``clears_band`` (``managers.regime_classifier``), so the experiment and the
runtime cannot drift apart -- if the runtime's idea of "clears" ever changes,
this report changes with it.

THE DOMAIN: ACTIVE BOUNDARIES ONLY
----------------------------------
Every signal is read on the boundaries whose window ran a forward. An idle
window is no measurement:

* the shares are ``None`` there (``RegimeSample`` returns ``None`` for both
  when ``forward_rounds == 0``), so dropping the absent samples already does
  it;
* ``occupancy`` and ``queued_prompt_tokens`` report a real ``0.0`` while the
  rig idles, and ``rank_ms_spread_pct`` reports the LAST measured forward, so
  it goes stale rather than absent. All three have to be restricted by hand.

Gate 3 (2026-08-01) is why this is a rule and not a footnote. Its two arms
idled for different lengths -- 19 402 against 15 504 boundaries, because the
workload started at different delays after ready -- so an unrestricted
comparison pairs one arm's quiet stretch against the other's busy one and
reports the signal's whole range as noise. ``rank_ms_spread_pct`` was the last
signal left unrestricted; it is restricted now, which drops its pointwise band
on the post-#388 pair from 0.6815 (the full observed range, flagged
``ARMS_DISSIMILAR``) to 0.5813 against a within-arm swing of 0.675.

THE STATISTIC, PER SIGNAL
-------------------------
A pointwise A-vs-A band -- ``max_i |a_i - b_i|`` over aligned positions --
assumes the value at position ``i`` describes the same thing in both runs.
#388 broke that assumption for four of the five signals and the gate-3
re-record proved it:

    with the phase read off the batch that RAN, the shares are near-BINARY at
    ``window_rounds = 64``. A window is essentially all-prefill or all-decode,
    ``prefill_share`` swings the full 0..1, and the pointwise band is 1
    whenever the two runs' bursts land on different boundary indices -- which
    they MUST, because the same workload produced 41 active boundaries in one
    arm and 56 in the other. All five signals came back ``ARMS_DISSIMILAR``.
    The alignment was designed against traces in which the shares barely
    moved; the fix that made them move invalidated the statistic chosen for
    them (DESIGN 18.2).

So the statistic is chosen per signal, against one criterion: **is the value
at a given active boundary reproducible across boots, or only its rate?**

``STATISTIC_POINTWISE``
    The signal varies slowly relative to the window, so position ``i`` in one
    run describes the same thing as position ``i`` in the other. Band =
    ``signal_band`` over the two active subsequences resampled onto
    ``K = min(len_a, len_b)`` positions of a normalised 0..1 timeline.
    Only ``rank_ms_spread_pct``: it is a within-boundary ratio across ranks,
    not a burst counter, and it moves continuously.

``STATISTIC_DISTRIBUTIONAL``
    The signal is bursty or binary. WHERE a burst lands is not reproducible;
    HOW OFTEN it happens is. Comparing the two runs' distributions over the
    active boundaries is the only thing left that is a measurement, and it is
    also exactly what the gate needs to know -- the gate's question is not
    "did the two runs agree at boundary 37", it is **"would this signal cross
    the threshold consistently across boots?"**. So the compared quantities
    are per-run SUMMARIES and the band is ``|summary_a - summary_b|`` on
    those:

    * **peak** (max over active boundaries) -- the summary the reachability
      check reads, and the signal-level ``band`` reported in the table;
    * **duty cycle at a constant's own value** -- the fraction of active
      boundaries at or above it. This is the summary a THRESHOLD is judged on;
    * **quantiles** p50/p75/p90 -- the shape, reported for information.

    Deliberately NOT used: the max quantile shift (the sup-Wasserstein
    distance, "sort both series and subtract"). On a near-binary signal it is
    maximally sensitive exactly where the distribution is flat-then-jumps: a
    few per cent of mass moving across the jump reads as a 0.7 shift in signal
    units. It measures the binariness, not the noise.

HOW A CONSTANT IS JUDGED, PER STATISTIC
---------------------------------------
Reachability first in both cases -- a threshold the signal never approaches is
dead whatever its band says.

* pointwise, with a hysteresis gap: the gap must clear ``THRESHOLD_MARGIN``
  times the band, unchanged since the first version.
* distributional: the constant's own CROSSING RATE must clear
  ``THRESHOLD_MARGIN`` times the run-to-run disagreement about that rate --
  ``clears_band(mean(duty_a, duty_b), |duty_a - duty_b|)``. A threshold whose
  crossing rate is measured to worse than 50 % relative accuracy across two
  identical boots does not produce a reproducible decision, which is the
  failure ``INSIDE_BAND`` names.

  The hysteresis gap is still REPORTED for a distributional signal (both
  duties, at the enter and the exit value) but does not decide the verdict:
  the gap lives in signal units and the reproducible quantity does not, and
  converting the gap into a duty difference would punish the good case. A
  decisive signal puts almost no windows inside the hysteresis interval, and
  that is the hysteresis doing nothing because it is not needed -- not a
  finding.

THE TWO GUARDS, UNDER THE NEW STATISTIC
---------------------------------------
Neither is dropped; both are re-stated for a distribution.

``UNDERPOWERED``
    Fewer than ``MIN_PAIRED_SAMPLES`` active boundaries in the smaller arm. A
    band -- pointwise or distributional -- from this few is a number, not a
    measurement.

``ARMS_DISSIMILAR``
    The two runs were not doing the same thing, so their difference is not a
    noise floor. Pointwise: the band is as large as the movement WITHIN one
    arm. Distributional: the two runs' duty cycles disagree, at the
    constants' own thresholds, by more than sampling from one workload should
    produce -- a two-proportion difference beyond ``DISSIMILAR_Z`` standard
    errors.

    Evaluated at the CONSTANTS' thresholds and nowhere else, on purpose. A
    sup-over-all-thresholds form (the two-sample Kolmogorov-Smirnov statistic)
    flags any two constant-valued arms held at slightly different levels as
    totally separated, and that case is a real, reproducible bias -- it IS the
    band, and calling it a comparability failure throws the measurement away.
    The gate asks whether the constants' decisions reproduce; the guard asks
    the same question of the same thresholds. A distributional signal that no
    constant reads has no decision to reproduce and is not guarded.

    The active boundaries are autocorrelated, so the nominal significance
    level is optimistic and the test leans toward flagging. ``DISSIMILAR_Z``
    is set at the 1 % two-sided level rather than 5 % to take some of that
    back, and a max over at most two thresholds against a critical value is
    conservative in the same direction.

Resampling on the pointwise path can only INFLATE the band, and inflation is
the conservative direction: it makes a threshold harder to clear, never
easier. The report states both active span lengths so a reader can see how
much work the resampling did.

Card-less checks:

``--smoke``
    the whole pipeline on synthetic pairs with bands known in advance, plus
    the trace of the 2026-08-01 re-run split into two pseudo-arms.
``--falsify``
    the three cases that decide whether this method change was right: a
    same-duty burst shift (the false alarm of record, which the retained
    pointwise path still raises), a genuine duty difference (the guard has to
    survive), and a barely-moving signal (old and new must agree).
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import math
import os
import sys
from typing import Dict, List, Sequence

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
)

from sglang.srt.managers.regime_classifier import (  # noqa: E402
    DEFAULT_ENTER_DECODE,
    DEFAULT_ENTER_PREFILL,
    DEFAULT_EXIT_DECODE,
    DEFAULT_EXIT_PREFILL,
    DEFAULT_WINDOW_ROUNDS,
    KV_ASCEND_MARK,
    KV_DESCEND_MARK,
    clears_band,
    signal_band,
)

#: The margin a THRESHOLD must clear its band by. One band away still leaves
#: the threshold on the wrong side of itself half the time (DESIGN_363 3.4).
THRESHOLD_MARGIN = 2.0

#: Below this many active boundaries in the smaller arm a band is a number,
#: not a measurement -- on either statistic.
MIN_PAIRED_SAMPLES = 8

#: Two-sided normal deviate for the duty-cycle disagreement guard. 2.576 is
#: the 1 % level; see the module docstring for why 1 % and not 5 %.
DISSIMILAR_Z = 2.576

#: Quantiles reported for a distributional signal. Shape, not verdict.
REPORT_QUANTILES = (0.50, 0.75, 0.90)

STATISTIC_POINTWISE = "pointwise"
STATISTIC_DISTRIBUTIONAL = "distributional"

#: The classifier inputs, and where each lives on a verdict record.
SIGNALS = {
    "prefill_share": "prefill_share",
    "decode_share": "decode_share",
    "occupancy": "occupancy",
    "queued_prompt_tokens": "queued_prompt_tokens",
    "rank_ms_spread_pct": "rank_ms_spread_pct",
}

#: Which statistic each signal is compared with, and why. The criterion is in
#: the module docstring: pointwise only where the value at a given active
#: boundary is itself reproducible across boots.
STATISTIC = {
    # Near-binary since #388: a window is all-prefill or all-decode.
    "prefill_share": STATISTIC_DISTRIBUTIONAL,
    "decode_share": STATISTIC_DISTRIBUTIONAL,
    # Bursty: near zero between bursts, a spike while one drains.
    "occupancy": STATISTIC_DISTRIBUTIONAL,
    # The burstiest of the five: median 0, peak 74 802.
    "queued_prompt_tokens": STATISTIC_DISTRIBUTIONAL,
    # Continuous, and a within-boundary ratio across ranks rather than a
    # burst counter: position i still means the same thing in both runs.
    "rank_ms_spread_pct": STATISTIC_POINTWISE,
}


class Constant:
    """One section-3.4 constant, and how to judge it against the data."""

    def __init__(self, name, value, signal, *, exit_value=None, partner=None, note=""):
        self.name = name
        self.value = value
        self.signal = signal
        #: The far side of a hysteresis pair, if the constant has one.
        self.exit_value = exit_value
        self.partner = partner
        self.note = note

    @property
    def gap(self):
        """The interval the constant defends, in the signal's own units.

        ``None`` for a bare threshold: it defends nothing, and on a pointwise
        signal there is then nothing to compare against the band.
        """
        if self.exit_value is None:
            return None
        return self.value - self.exit_value

    @property
    def thresholds(self):
        """Every value of this constant a decision is taken at."""
        if self.exit_value is None:
            return (self.value,)
        return (self.value, self.exit_value)


CONSTANTS = [
    Constant(
        "enter_prefill",
        DEFAULT_ENTER_PREFILL,
        "prefill_share",
        exit_value=DEFAULT_EXIT_PREFILL,
        partner="exit_prefill",
    ),
    Constant(
        "enter_decode",
        DEFAULT_ENTER_DECODE,
        "decode_share",
        exit_value=DEFAULT_EXIT_DECODE,
        partner="exit_decode",
    ),
    Constant(
        "kv_ascend_mark",
        KV_ASCEND_MARK,
        "occupancy",
        exit_value=KV_DESCEND_MARK,
        partner="kv_descend_mark",
        note=(
            "INHERITED from #287 and deliberately not re-derived here: two "
            "independently-chosen thresholds on one physical quantity is how "
            "two controllers end up disagreeing about the same pool. Reported "
            "for information; a failure is a finding for #287, not a licence "
            "to set a second mark."
        ),
    ),
    Constant(
        "spread_veto_pct",
        25.0,
        "rank_ms_spread_pct",
        note="a bare veto threshold: judged on reachability and band only",
    ),
    Constant(
        "PRESTAGE_SINGLE_PROMPT_TOKENS",
        8192,
        "queued_prompt_tokens",
        note="queue mass, not a share; the trace records the total queued",
    ),
]


def thresholds_for(signal: str) -> List[float]:
    """Every threshold any section-3.4 constant reads off ``signal``."""
    out: List[float] = []
    for c in CONSTANTS:
        if c.signal != signal:
            continue
        for t in c.thresholds:
            if t not in out:
                out.append(float(t))
    return out


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_boundaries(path: str) -> List[Dict]:
    """One entry per consensus BOUNDARY, in order.

    A TP group writes one verdict per rank per boundary. The group agrees by
    construction (a disagreement is a desync, which gate 1 catches), so the
    boundary's value is any rank's -- but it must be counted ONCE, or a
    3-rank run looks like three times the samples and every band shrinks by
    the square root of a lie. With a rank stamp we take one rank; without
    one, the first verdict of each round.
    """
    verdicts = []
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("kind") == "verdict":
                verdicts.append(obj)
    ranks = {v.get("rank") for v in verdicts}
    if ranks != {None}:
        pick = sorted(r for r in ranks if r is not None)[0]
        return [v for v in verdicts if v.get("rank") == pick]
    seen, out = set(), []
    for v in verdicts:
        if v.get("round") in seen:
            continue
        seen.add(v["round"])
        out.append(v)
    return out


def is_active(boundary: Dict) -> bool:
    """Did this window run any forward at all?

    Read off the shares rather than off a separate field, because that is how
    the observer defines it: ``RegimeSample`` returns ``None`` for both shares
    exactly when ``forward_rounds == 0``. One definition of "active", held in
    one place, and the analysis inherits the runtime's.
    """
    return (
        boundary.get("prefill_share") is not None
        or boundary.get("decode_share") is not None
    )


def series(boundaries: Sequence[Dict], field: str) -> List[float]:
    """The signal's values on the ACTIVE boundaries, in order.

    Both restrictions, in one place. ``None`` is dropped rather than filled --
    an idle window carries no share, and a zero there would be a different
    claim that would also drag the band -- and the boundaries that ran no
    forward are dropped for every signal, including the three that report a
    real (or stale) value while the rig idles.
    """
    out = []
    for b in boundaries:
        if not is_active(b):
            continue
        v = b.get(field)
        if v is None:
            continue
        out.append(float(v))
    return out


def resample(values: Sequence[float], k: int) -> List[float]:
    """``values`` onto ``k`` positions of a normalised 0..1 timeline.

    Nearest-neighbour on purpose: interpolating between two samples would
    invent a value the run never produced, and the band would then be partly
    a property of the interpolation.
    """
    n = len(values)
    if n == 0 or k <= 0:
        return []
    if n == k:
        return list(values)
    out = []
    for i in range(k):
        pos = 0 if k == 1 else i * (n - 1) / (k - 1)
        out.append(float(values[int(round(pos))]))
    return out


# ---------------------------------------------------------------------------
# Distributional summaries
# ---------------------------------------------------------------------------


def duty_cycle(values: Sequence[float], threshold: float) -> float:
    """Fraction of active boundaries at or above ``threshold``.

    The rate at which the signal crosses a constant. Bursty signals do not
    reproduce their burst POSITIONS across boots, but a workload run twice
    reproduces this.
    """
    if not values:
        return 0.0
    return sum(1 for v in values if v >= threshold) / len(values)


def quantile(values: Sequence[float], p: float) -> float:
    """Nearest-rank quantile: a value the run actually produced."""
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def duty_disagreement(
    a: Sequence[float], b: Sequence[float], thresholds: Sequence[float]
) -> Dict:
    """Worst duty-cycle disagreement over ``thresholds``, and its critical value.

    A two-proportion difference: under "both arms ran the same workload" the
    two duty cycles are two estimates of one rate, and their difference has
    standard error ``sqrt(p(1-p)(1/na + 1/nb))`` at the pooled rate ``p``.
    Beyond ``DISSIMILAR_Z`` of those the arms are not comparable.
    """
    na, nb = len(a), len(b)
    worst = {"threshold": None, "duty_a": None, "duty_b": None, "delta": 0.0, "z": 0.0}
    if not thresholds or na == 0 or nb == 0:
        worst["critical"] = None
        return worst
    for t in thresholds:
        da, db = duty_cycle(a, t), duty_cycle(b, t)
        delta = abs(da - db)
        pooled = (da * na + db * nb) / (na + nb)
        se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / na + 1.0 / nb))
        # se == 0 means both arms are unanimous at this threshold and agree;
        # a difference cannot be nonzero there, so z stays 0.
        z = 0.0 if se == 0.0 else delta / se
        if z > worst["z"] or (worst["threshold"] is None):
            worst = {
                "threshold": float(t),
                "duty_a": da,
                "duty_b": db,
                "delta": delta,
                "z": z,
            }
    worst["critical"] = DISSIMILAR_Z
    return worst


# ---------------------------------------------------------------------------
# The band, per signal
# ---------------------------------------------------------------------------


def _common(field: str, sa: Sequence[float], sb: Sequence[float]) -> Dict:
    pooled = list(sa) + list(sb)
    return {
        "signal": field,
        "statistic": STATISTIC.get(field, STATISTIC_DISTRIBUTIONAL),
        "n_a": len(sa),
        "n_b": len(sb),
        "observed_max": max(pooled) if pooled else None,
        "observed_min": min(pooled) if pooled else None,
    }


def _pointwise(res: Dict, sa: Sequence[float], sb: Sequence[float]) -> Dict:
    k = min(len(sa), len(sb))
    res["paired"] = k
    res["band"] = signal_band(resample(sa, k), resample(sb, k))
    res["span_ratio"] = round(max(len(sa), len(sb)) / min(len(sa), len(sb)), 3)
    # A band that swallows the movement inside a single arm is not a noise
    # measurement: it says the two arms were doing different things at the
    # positions the alignment paired up. Compared against the WITHIN-ARM swing
    # and not the pooled range -- two arms that each hold steady at different
    # levels differ by a real, reproducible bias, and that IS the band.
    within = max(max(sa) - min(sa), max(sb) - min(sb))
    if within > 0 and res["band"] >= 0.9 * within:
        res["status"] = "ARMS_DISSIMILAR"
        res["why"] = (
            f"the pointwise band ({res['band']:g}) is as large as the signal's "
            f"own movement within a single arm ({within:g}): the two arms were "
            f"not doing the same thing at the positions the alignment paired. "
            f"That is a comparability failure, not a noise floor -- check that "
            f"both boots ran the same workload and that neither was truncated."
        )
        return res
    if k < MIN_PAIRED_SAMPLES:
        res["status"] = "UNDERPOWERED"
        res["why"] = (
            f"only {k} paired sample(s); a band from this few is a number, "
            f"not a measurement. Run a workload that produces more windows "
            f"in which this signal exists."
        )
    else:
        res["status"] = "OK"
        res["why"] = f"{k} paired samples after alignment"
    return res


def _distributional(res: Dict, sa: Sequence[float], sb: Sequence[float]) -> Dict:
    field = res["signal"]
    res["paired"] = min(len(sa), len(sb))
    res["peak_a"], res["peak_b"] = max(sa), max(sb)
    # The signal-level band is the PEAK disagreement: |summary_a - summary_b|
    # on the summary the reachability check reads, in the signal's own units.
    # Routed through the controller's own signal_band as a one-element pair so
    # the experiment and the runtime still cannot drift apart.
    res["band"] = signal_band([res["peak_a"]], [res["peak_b"]])
    res["quantiles"] = {
        f"p{int(p * 100)}": {
            "a": quantile(sa, p),
            "b": quantile(sb, p),
            "band": abs(quantile(sa, p) - quantile(sb, p)),
        }
        for p in REPORT_QUANTILES
    }
    res["span_ratio"] = round(max(len(sa), len(sb)) / min(len(sa), len(sb)), 3)
    thresholds = thresholds_for(field)
    worst = duty_disagreement(sa, sb, thresholds)
    res["duty"] = {
        f"{t:g}": {
            "a": duty_cycle(sa, t),
            "b": duty_cycle(sb, t),
            "band": abs(duty_cycle(sa, t) - duty_cycle(sb, t)),
        }
        for t in thresholds
    }
    res["duty_worst"] = worst
    if worst["threshold"] is not None and worst["z"] > DISSIMILAR_Z:
        res["status"] = "ARMS_DISSIMILAR"
        res["why"] = (
            f"the two runs' duty cycles at {worst['threshold']:g} disagree by "
            f"{worst['delta']:.3g} ({worst['duty_a']:.3g} against "
            f"{worst['duty_b']:.3g}), {worst['z']:.3g} standard errors, past "
            f"the {DISSIMILAR_Z:g} allowed: more than sampling one workload "
            f"twice should produce. That is a comparability failure, not a "
            f"noise floor -- check that both boots ran the same workload and "
            f"that neither was truncated."
        )
        return res
    if res["paired"] < MIN_PAIRED_SAMPLES:
        res["status"] = "UNDERPOWERED"
        res["why"] = (
            f"only {res['paired']} active boundary/-ies in the smaller arm; a "
            f"distribution from this few is a number, not a measurement. Run a "
            f"workload that produces more windows in which this signal exists."
        )
    else:
        res["status"] = "OK"
        res["why"] = (
            f"{len(sa)} and {len(sb)} active boundaries; distributions "
            f"compared on {len(thresholds)} threshold(s)"
        )
    return res


def band_for(a: Sequence[Dict], b: Sequence[Dict], field: str) -> Dict:
    sa, sb = series(a, field), series(b, field)
    res = _common(field, sa, sb)
    if not sa or not sb:
        res.update(
            band=None,
            paired=0,
            status="NO_DATA",
            why="the signal is absent in at least one run",
        )
        return res
    if res["statistic"] == STATISTIC_POINTWISE:
        return _pointwise(res, sa, sb)
    return _distributional(res, sa, sb)


def judge_constant(c: Constant, bands: Dict[str, Dict]) -> Dict:
    """Per-constant verdict: does it clear its band, and is it reachable?"""
    b = bands.get(c.signal, {})
    out = {
        "constant": c.name,
        "value": c.value,
        "signal": c.signal,
        "statistic": b.get("statistic"),
        "gap": c.gap,
        "band": b.get("band"),
        "observed_max": b.get("observed_max"),
        "note": c.note,
    }
    if b.get("status") in (None, "NO_DATA"):
        out.update(verdict="NO_DATA", why=b.get("why", "no band"))
        return out

    # Reachability FIRST. A threshold the signal never approaches is dead
    # whatever its band says: the regime it gates cannot be entered, and
    # reporting "clears its band" for it would be true and useless.
    top = b.get("observed_max")
    if top is not None and top < c.value:
        out.update(
            verdict="UNREACHED",
            why=(
                f"the signal peaked at {top:g}, below the constant "
                f"{c.value:g}: nothing this constant gates can ever have "
                f"triggered in these runs. Either the constant is mis-set for "
                f"this rig, or the workload never produced the shape it reads "
                f"-- this report does not choose between those, and does not "
                f"re-tune."
            ),
        )
        return out
    if b.get("status") in ("UNDERPOWERED", "ARMS_DISSIMILAR"):
        out.update(verdict=b["status"], why=b["why"])
        return out

    if b["statistic"] == STATISTIC_DISTRIBUTIONAL:
        # The reproducible quantity for a bursty signal is the CROSSING RATE,
        # so that is what has to clear its own run-to-run band. The hysteresis
        # gap is reported alongside but does not decide: it lives in signal
        # units, and this comparison does not.
        d = b["duty"][f"{c.value:g}"]
        rate = 0.5 * (d["a"] + d["b"])
        out["duty_a"], out["duty_b"], out["duty_band"] = d["a"], d["b"], d["band"]
        if c.exit_value is not None:
            de = b["duty"][f"{c.exit_value:g}"]
            out["exit_duty_a"], out["exit_duty_b"] = de["a"], de["b"]
        ok = clears_band(rate, d["band"], margin=THRESHOLD_MARGIN)
        out.update(
            verdict="CLEARS" if ok else "INSIDE_BAND",
            why=(
                f"crossing rate {rate:.3g} of active boundaries (arm A "
                f"{d['a']:.3g}, arm B {d['b']:.3g}) against "
                f"{THRESHOLD_MARGIN:g}x the run-to-run disagreement "
                f"{d['band']:.3g} = {THRESHOLD_MARGIN * d['band']:.3g}"
            ),
        )
        return out

    if c.gap is None:
        out.update(
            verdict="NO_GAP",
            why=(
                "a bare threshold defends no interval, so there is nothing to "
                f"compare against the pointwise band ({b['band']:g}). Reported "
                "for information."
            ),
        )
        return out
    ok = clears_band(c.gap, b["band"], margin=THRESHOLD_MARGIN)
    out.update(
        verdict="CLEARS" if ok else "INSIDE_BAND",
        why=(
            f"gap {c.gap:g} against {THRESHOLD_MARGIN:g}x the measured band "
            f"{b['band']:g} = {THRESHOLD_MARGIN * b['band']:g}"
        ),
    )
    return out


def report(path_a: str, path_b: str) -> Dict:
    a, b = load_boundaries(path_a), load_boundaries(path_b)
    bands = {name: band_for(a, b, field) for name, field in SIGNALS.items()}
    verdicts = [judge_constant(c, bands) for c in CONSTANTS]
    # Only CLEARS and NO_GAP are non-blocking. UNDERPOWERED belongs in this
    # list for the same reason as the rest: a band from under
    # MIN_PAIRED_SAMPLES samples is a number, not a measurement, so a
    # threshold judged against it has not been checked.
    blocking = [
        v
        for v in verdicts
        if v["verdict"]
        in ("INSIDE_BAND", "UNREACHED", "NO_DATA", "ARMS_DISSIMILAR", "UNDERPOWERED")
    ]
    return {
        "arm_a": {
            "path": path_a,
            "boundaries": len(a),
            "active": sum(1 for x in a if is_active(x)),
        },
        "arm_b": {
            "path": path_b,
            "boundaries": len(b),
            "active": sum(1 for x in b if is_active(x)),
        },
        "window_rounds": DEFAULT_WINDOW_ROUNDS,
        "bands": bands,
        "constants": verdicts,
        "passed": not blocking,
        "blocking": [f"{v['constant']}: {v['verdict']}" for v in blocking],
    }


def evidence_entry(rep: Dict, *, note: str = "") -> Dict:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    measured = ", ".join(
        f"{n}={b['band']:g}"
        for n, b in sorted(rep["bands"].items())
        if b.get("band") is not None
    )
    source = (
        f"gate-3 A-vs-A bands {stamp}: {rep['arm_a']['boundaries']} vs "
        f"{rep['arm_b']['boundaries']} boundaries "
        f"({rep['arm_a']['active']} vs {rep['arm_b']['active']} active); "
        f"{measured}; every section-3.4 constant checked at "
        f"{THRESHOLD_MARGIN:g}x its own band, distributional where the signal "
        f"is bursty "
        f"[{os.path.basename(rep['arm_a']['path'])}; "
        f"{os.path.basename(rep['arm_b']['path'])}]"
    )
    if note:
        source += f" -- {note}"
    return {"f3_bands_measured": {"passed": bool(rep["passed"]), "source": source}}


def render(rep: Dict) -> str:
    lines = [
        f"arm A: {rep['arm_a']['boundaries']} boundaries "
        f"({rep['arm_a'].get('active', '?')} active)  {rep['arm_a']['path']}",
        f"arm B: {rep['arm_b']['boundaries']} boundaries "
        f"({rep['arm_b'].get('active', '?')} active)  {rep['arm_b']['path']}",
        "",
        f"{'signal':<22}{'statistic':>15}{'band':>12}{'n_a':>6}{'n_b':>6}"
        f"{'max':>12}  status",
    ]
    for name, b in sorted(rep["bands"].items()):
        band = "absent" if b.get("band") is None else f"{b['band']:.6g}"
        top = "-" if b.get("observed_max") is None else f"{b['observed_max']:.6g}"
        lines.append(
            f"{name:<22}{b['statistic']:>15}{band:>12}{b.get('n_a', 0):>6}"
            f"{b.get('n_b', 0):>6}{top:>12}  {b['status']}"
        )
    lines += [
        "  every signal is read on the ACTIVE boundaries only",
        "  distributional band = |peak_a - peak_b|; a constant on such a "
        "signal is judged on its duty cycle",
    ]
    duty_rows = []
    for name, b in sorted(rep["bands"].items()):
        for t, d in sorted(b.get("duty", {}).items()):
            duty_rows.append(
                f"  {name} >= {t}: duty A {d['a']:.4g}  B {d['b']:.4g}  "
                f"band {d['band']:.4g}"
            )
    if duty_rows:
        lines += ["", "duty cycles at the constants' own thresholds"] + duty_rows
    lines += ["", f"{'constant':<32}{'value':>10}{'gap':>8}  verdict"]
    for v in rep["constants"]:
        gap = "-" if v["gap"] is None else f"{v['gap']:g}"
        lines.append(f"{v['constant']:<32}{v['value']:>10}{gap:>8}  {v['verdict']}")
        lines.append(f"    {v['why']}")
        if v["note"]:
            lines.append(f"    note: {v['note']}")
    lines += [
        "",
        "PASSED" if rep["passed"] else "NOT PASSED: " + "; ".join(rep["blocking"]),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic traces, shared by the smoke and the falsifier
# ---------------------------------------------------------------------------

_REAL = "/spinning/gpu-battery-results/2026-08-01_363_gates_rerun/regime.jsonl.gz"


def _write(path, rows):
    with open(path, "w") as f:
        f.write(json.dumps({"kind": "header", "mode": "observe", "rank": 0}) + "\n")
        for i, r in enumerate(rows):
            rec = {"kind": "verdict", "rank": 0, "round": (i + 1) * 8}
            rec.update(r)
            f.write(json.dumps(rec) + "\n")


def _bursts(n: int, hot: Sequence[int]) -> List[Dict]:
    """``n`` active windows, binary shares, prefill at the indices in ``hot``.

    The post-#388 shape in miniature: a window is all-prefill or all-decode,
    so ``prefill_share`` is 0 or 1 and never in between.
    """
    hot = set(hot)
    return [
        {
            "prefill_share": 1.0 if i in hot else 0.0,
            "decode_share": 0.0 if i in hot else 1.0,
        }
        for i in range(n)
    ]


def _pair(tmp, rows_a, rows_b):
    a, b = os.path.join(tmp, "fa.jsonl"), os.path.join(tmp, "fb.jsonl")
    _write(a, rows_a)
    _write(b, rows_b)
    return a, b


def _old_pointwise(a_path, b_path, field):
    """The statistic this change replaced, on ``field``, for comparison.

    Not a reimplementation: it calls the pointwise path that is still live for
    ``rank_ms_spread_pct``, so the falsifier is comparing against code that
    still exists and still runs.
    """
    sa = series(load_boundaries(a_path), field)
    sb = series(load_boundaries(b_path), field)
    res = _common(field, sa, sb)
    res["statistic"] = STATISTIC_POINTWISE
    return _pointwise(res, sa, sb)


def falsify() -> int:
    """The three cases that decide whether the method change was right."""
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        # (i) SAME duty cycle, bursts on different boundary indices. This is
        #     the false alarm of record: 10 prefill-heavy windows out of 40 in
        #     both arms, but arm A's fall early and arm B's late. Any two
        #     boots of one workload do this -- the re-record's arms produced
        #     41 and 56 active boundaries, so their bursts CANNOT coincide.
        a, b = _pair(tmp, _bursts(40, range(0, 10)), _bursts(40, range(20, 30)))
        old = _old_pointwise(a, b, "prefill_share")
        new = report(a, b)["bands"]["prefill_share"]
        print("(i)  same duty cycle, shifted bursts -- the false alarm of record")
        print(f"     OLD pointwise : band={old['band']:g}  {old['status']}")
        print(
            f"     NEW distrib.  : band={new['band']:g}  "
            f"duty A {new['duty']['0.35']['a']:g} B "
            f"{new['duty']['0.35']['b']:g}  {new['status']}"
        )
        ok &= old["status"] == "ARMS_DISSIMILAR" and old["band"] == 1.0
        ok &= new["status"] == "OK" and new["duty"]["0.35"]["band"] == 0.0
        vp = next(
            v for v in report(a, b)["constants"] if v["constant"] == "enter_prefill"
        )
        print(f"     enter_prefill -> {vp['verdict']}")
        ok &= vp["verdict"] == "CLEARS"

        # (ii) GENUINELY different duty cycles. The guard has to survive the
        #      reformulation: 10 prefill-heavy windows out of 40 against 30.
        a, b = _pair(tmp, _bursts(40, range(0, 10)), _bursts(40, range(0, 30)))
        new = report(a, b)["bands"]["prefill_share"]
        w = new["duty_worst"]
        print("\n(ii) genuinely different duty cycles -- the guard must survive")
        print(
            f"     NEW distrib.  : duty A {w['duty_a']:g} B {w['duty_b']:g}  "
            f"z={w['z']:.3g} vs {DISSIMILAR_Z:g}  {new['status']}"
        )
        ok &= new["status"] == "ARMS_DISSIMILAR"

        # (iii) A BARELY-MOVING signal: the regime the old statistic was
        #       designed against. Old and new must agree, or this change is a
        #       regression on the traces that motivated the first version.
        a, b = _pair(
            tmp,
            [{"decode_share": 0.50, "prefill_share": None} for _ in range(20)],
            [{"decode_share": 0.53, "prefill_share": None} for _ in range(20)],
        )
        old = _old_pointwise(a, b, "decode_share")
        new = report(a, b)["bands"]["decode_share"]
        print("\n(iii) barely-moving signal -- old and new must agree")
        print(f"     OLD pointwise : band={old['band']:g}  {old['status']}")
        print(f"     NEW distrib.  : band={new['band']:g}  {new['status']}")
        ok &= abs(old["band"] - 0.03) < 1e-9
        ok &= abs(new["band"] - 0.03) < 1e-9
        ok &= old["status"] == new["status"] == "OK"
    print("\nFALSIFIER OK" if ok else "\nFALSIFIER FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def smoke() -> int:
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        # 1. A synthetic pair whose band is known before the code runs. The
        #    distributional band is the peak disagreement, and for a flat
        #    signal that is the same 0.03 the pointwise band reported.
        a = os.path.join(tmp, "a.jsonl")
        b = os.path.join(tmp, "b.jsonl")
        _write(a, [{"decode_share": 0.50, "prefill_share": None} for _ in range(20)])
        _write(b, [{"decode_share": 0.53, "prefill_share": None} for _ in range(20)])
        rep = report(a, b)
        got = rep["bands"]["decode_share"]["band"]
        print(f"[known band ] decode_share band={got:.6g} (expected 0.03)")
        ok &= abs(got - 0.03) < 1e-9

        # 2. Different active-span lengths: the smaller arm decides the power.
        _write(b, [{"decode_share": 0.53, "prefill_share": None} for _ in range(7)])
        rep = report(a, b)
        bb = rep["bands"]["decode_share"]
        print(
            f"[ragged span] paired={bb['paired']} ratio={bb.get('span_ratio')} "
            f"status={bb['status']}"
        )
        ok &= bb["paired"] == 7 and bb["status"] == "UNDERPOWERED"

        # 3. An UNREACHED constant is called out as such, not as a pass.
        _write(a, [{"prefill_share": 0.01, "decode_share": 0.99} for _ in range(20)])
        _write(b, [{"prefill_share": 0.01, "decode_share": 0.99} for _ in range(20)])
        rep = report(a, b)
        vp = next(v for v in rep["constants"] if v["constant"] == "enter_prefill")
        print(f"[unreached  ] enter_prefill -> {vp['verdict']}")
        ok &= vp["verdict"] == "UNREACHED"
        ok &= not rep["passed"]

        # 4. A binary burst signal whose duty cycle DOES reproduce clears its
        #    constant -- the case the pointwise statistic could not express.
        _write(a, _bursts(40, range(0, 12)))
        _write(b, _bursts(40, range(25, 37)))
        rep = report(a, b)
        vp = next(v for v in rep["constants"] if v["constant"] == "enter_prefill")
        print(
            f"[binary duty] enter_prefill -> {vp['verdict']} "
            f"(duty {vp['duty_a']:g} vs {vp['duty_b']:g})"
        )
        ok &= vp["verdict"] == "CLEARS"

        # 5. THE REAL TRACE, split into two pseudo-arms. Not a real A-vs-A
        #    pair -- one boot cannot be two -- but it is the only material
        #    that exercises the pipeline on the real shape, which is what this
        #    script was deferred for.
        if os.path.exists(_REAL):
            bounds = load_boundaries(_REAL)
            half = len(bounds) // 2
            ha, hb = os.path.join(tmp, "ha.jsonl"), os.path.join(tmp, "hb.jsonl")
            for dst, part in ((ha, bounds[:half]), (hb, bounds[half:])):
                with open(dst, "w") as f:
                    f.write(
                        json.dumps({"kind": "header", "mode": "observe", "rank": 0})
                        + "\n"
                    )
                    for r in part:
                        r = dict(r)
                        r["rank"] = 0
                        f.write(json.dumps(r) + "\n")
            rep = report(ha, hb)
            print("\n[real trace, halves as pseudo-arms]")
            print(render(rep))
            ok &= rep["bands"]["occupancy"]["band"] is not None
            # The halves of one run are NOT an A-vs-A pair: the idle first
            # half contributes 2 active windows against the working half's 26,
            # so the distributional summaries are UNDERPOWERED. A refusal, and
            # it names the real problem with this fixture.
            ok &= rep["bands"]["occupancy"]["status"] == "UNDERPOWERED"
            ok &= not rep["passed"]
            print(
                "\n[guard      ] occupancy status="
                f"{rep['bands']['occupancy']['status']} (halves are not a pair)"
            )
            # The POINTWISE guard is still live and still reachable: on this
            # fixture rank_ms_spread_pct -- the one signal that kept the
            # pointwise statistic -- trips it, so both guards stay exercised
            # by the smoke rather than only the new one.
            ok &= rep["bands"]["rank_ms_spread_pct"]["status"] == "ARMS_DISSIMILAR"
            print(
                "[guard      ] rank_ms_spread_pct status="
                f"{rep['bands']['rank_ms_spread_pct']['status']} "
                "(the pointwise guard, still live)"
            )
        else:
            print(f"[real trace ] SKIPPED, {_REAL} not present")
    print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-a", help="verdict trace of boot A")
    ap.add_argument("--arm-b", help="verdict trace of boot B (the repeat)")
    ap.add_argument("--evidence", help="evidence JSON to create/merge into")
    ap.add_argument("--note", default="")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--falsify",
        action="store_true",
        help="the three synthetic cases that justify the distributional statistic",
    )
    args = ap.parse_args(argv)

    if args.smoke:
        return smoke()
    if args.falsify:
        return falsify()
    if not (args.arm_a and args.arm_b):
        ap.error("--arm-a and --arm-b are required, or use --smoke / --falsify")
    if os.path.realpath(args.arm_a) == os.path.realpath(args.arm_b):
        print(
            "arm A and arm B are the same file: a band measured against "
            "itself is zero by construction and every threshold would clear "
            "it for free. Gate 3 needs TWO boots.",
            file=sys.stderr,
        )
        return 2

    rep = report(args.arm_a, args.arm_b)
    print(json.dumps(rep, indent=2) if args.json else render(rep))
    entry = evidence_entry(rep, note=args.note)
    print("\nevidence entry:\n" + json.dumps(entry, indent=2))
    if not rep["passed"]:
        print("\nGATE 3 NOT PASSED -- not written to the evidence file.")
        return 1
    if args.evidence:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from readout import merge_into  # noqa: PLC0415 -- sibling script

        merge_into(args.evidence, entry)
        print(f"\nwritten to {args.evidence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
