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
"""Rolling medians for the landing-page rate tiles -- over PROCESSING windows
only.

A rate tile reads 0 while the server is idle, which is correct and also
useless: the reader cannot tell "this rig does 43 tok/s and is resting" from
"this rig does 3 tok/s". The tiles therefore carry a second number, the median
of the recent past, e.g. ``0 (median 43.2)``.

The whole value of that badge rests on ONE rule: **idle zeros never enter the
window.** A median taken over every poll of a mostly-idle server is 0 by
construction, and a badge that reads ``0 (median 0.0)`` is worse than no badge
-- it looks like a measurement and carries no information. So each tracked
quantity has its own ACTIVITY PREDICATE, evaluated on the same counter deltas
that produced the rate, and a poll window contributes a sample only when that
predicate holds:

===========================  ================================================
key                          contributes when
===========================  ================================================
``decode_tok_s``             generated tokens moved in the window
``decode_tok_s_per_request`` generated tokens moved AND >= 1 running request
``prefill_tok_s``            non-cached prefill tokens were computed
``prefill_tok_s_per_request``  ... AND >= 1 running request
``spec_accept_rate``         spec metrics present AND decode was active
``cache_hit_overall``        the window had a prompt to hit against
===========================  ================================================

The predicates are deliberately expressed on DELTAS, not on the rate value: a
rate is a delta divided by dt, so a zero rate and a zero delta are the same
fact, but the delta is the one that cannot be produced by a degenerate dt.

Window size: :data:`RATE_MEDIAN_WINDOW` = 30 processing samples. The landing
page polls every ``LAND_POLL_MS`` = 2000 ms, so 30 samples is 60 s of ACTUAL
PROCESSING -- the same 60 s horizon the per-card telemetry rings on the same
page already use, expressed in the unit this window counts in. It is short
enough that a configuration change washes out within a minute of load and long
enough that a single slow round does not move the middle value.

Median, not mean: the samples are per-poll rates over a workload that is
bimodal by nature (prefill-heavy vs decode-heavy stretches, see the
content-variance note in ``rigmon/rates.py``). A mean over that mixture tracks
whichever mode happened to be longer; the median reports the typical window.

This module is pure. The window state lives in the opaque ``state`` dict that
``live_metrics.snapshot()`` already hands back to its caller, so nothing is
persisted and nothing is global.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "RATE_MEDIAN_WINDOW",
    "MEDIAN_KEYS",
    "median_of",
    "processing_samples",
    "update_windows",
    "medians",
]

#: Number of PROCESSING samples kept per tracked quantity. See the module
#: docstring for why 30 (60 s of processing at the 2 s landing poll).
RATE_MEDIAN_WINDOW = 30

#: The tracked quantities, in the order the tiles appear on the strip.
MEDIAN_KEYS = (
    "decode_tok_s",
    "decode_tok_s_per_request",
    "prefill_tok_s",
    "prefill_tok_s_per_request",
    "spec_accept_rate",
    "cache_hit_overall",
)


def median_of(values: Sequence[float]) -> Optional[float]:
    """Median of a sample list, or None for an empty one.

    None -- not 0.0 -- is the empty answer on purpose: "nothing has been
    processed yet" and "processing produced zero" are different statements and
    the tile renders them differently.
    """
    vals = sorted(float(v) for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def processing_samples(
    rates: Optional[Dict[str, Any]],
    prev_counters: Optional[Dict[str, Any]],
    cur_counters: Optional[Dict[str, Any]],
    hit_rates: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """The samples this poll window contributes -- possibly none.

    Returns a mapping of :data:`MEDIAN_KEYS` to values, containing ONLY the
    keys whose phase was genuinely active in the window. An idle window
    returns ``{}``, which is what keeps the idle zeros out of the medians.
    """
    out: Dict[str, float] = {}
    if not rates or not prev_counters or not cur_counters:
        return out
    if (rates.get("dt") or 0.0) <= 0:
        return out

    decode_delta = max(
        0.0,
        float(cur_counters.get("generation_tokens_total", 0.0))
        - float(prev_counters.get("generation_tokens_total", 0.0)),
    )
    prompt_delta = max(
        0.0,
        float(cur_counters.get("prompt_tokens_total", 0.0))
        - float(prev_counters.get("prompt_tokens_total", 0.0)),
    )
    cached_delta = max(
        0.0,
        float(cur_counters.get("cached_total", 0.0))
        - float(prev_counters.get("cached_total", 0.0)),
    )
    prefill_delta = max(0.0, prompt_delta - cached_delta)

    running = cur_counters.get("num_running_reqs")
    try:
        running = float(running) if running is not None else None
    except (TypeError, ValueError):
        running = None

    if decode_delta > 0:
        dec = rates.get("decode_tok_s")
        if dec is not None:
            out["decode_tok_s"] = float(dec)
            if running is not None and running >= 1:
                out["decode_tok_s_per_request"] = float(dec) / running
        spec = cur_counters.get("spec")
        if spec and spec.get("accept_rate") is not None:
            out["spec_accept_rate"] = float(spec["accept_rate"])

    if prefill_delta > 0:
        pfx = rates.get("prefill_tok_s")
        if pfx is not None:
            out["prefill_tok_s"] = float(pfx)
            if running is not None and running >= 1:
                out["prefill_tok_s_per_request"] = float(pfx) / running

    # The hit rate is already undefined (None) without a prompt window, so the
    # predicate is the same fact stated twice on purpose -- the badge must not
    # depend on a caller remembering to pass None.
    if prompt_delta > 0 and hit_rates and hit_rates.get("overall") is not None:
        out["cache_hit_overall"] = float(hit_rates["overall"])

    return out


def update_windows(
    prev_windows: Optional[Dict[str, List[float]]],
    samples: Dict[str, float],
    *,
    window: int = RATE_MEDIAN_WINDOW,
) -> Dict[str, List[float]]:
    """Append this window's samples and roll each list to at most ``window``.

    A key that got no sample keeps its list untouched: an idle tick must not
    age a quantity out of its own history, otherwise a server that rests for a
    minute would lose the medians it just earned.
    """
    out: Dict[str, List[float]] = {
        k: list(v) for k, v in (prev_windows or {}).items() if k in MEDIAN_KEYS
    }
    keep = max(1, int(window))
    for key in MEDIAN_KEYS:
        if key not in samples:
            continue
        lst = out.setdefault(key, [])
        lst.append(float(samples[key]))
        if len(lst) > keep:
            del lst[: len(lst) - keep]
    return out


def medians(windows: Optional[Dict[str, List[float]]]) -> Dict[str, dict]:
    """Render the windows as the snapshot's ``rate_medians`` block.

    One entry per key that HAS samples: ``{"median": float, "n": int,
    "window": int}``. Keys with an empty window are omitted entirely, so the
    page renders no badge rather than a zero it would have to explain.
    """
    out: Dict[str, dict] = {}
    for key in MEDIAN_KEYS:
        vals = (windows or {}).get(key) or []
        med = median_of(vals)
        if med is None:
            continue
        out[key] = {"median": med, "n": len(vals), "window": RATE_MEDIAN_WINDOW}
    return out
