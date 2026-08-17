#!/usr/bin/env python3
"""#516 -- does a longer-horizon MISS BUDGET beat swap-every-window?

Replays the recorded #302a expert-stats SERIES (not the single-shot dumps
``simulate_heat.py`` reads) and compares three placement policies over the same
recorded routing.

WHY A SERIES AND NOT THE SINGLE DUMPS. A per-wave budget is already built
(``expert_offload`` scratch slots + ``plan_token_waves``); what #516's third
half actually lacks is a budget over a LONGER horizon, and a horizon needs a
time axis. The single dumps are aggregate histograms with no time in them. The
``stats_series_*`` directories are periodic dumps -- 12 CUMULATIVE snapshots per
rank at 45 s intervals -- so differencing consecutive snapshots yields 11 real
per-window activation deltas. That differencing is the whole reason this
question is answerable at all.

THE THREE ARMS, all causal (a policy may only use windows it has already seen):

  A  static      the load-time set, never re-ranked. The floor.
  B  periodic    re-rank to the previous window's top-R every window. This is
                 the equal-count re-rank #302a ships, and the bar to beat.
  C  budget      re-rank ONLY when the window's miss rate exceeds the budget.

The mechanism C exploits is the one #302a's own config comment names: "small
values re-rank on noise and pay H2D for it". A window whose miss rate is
already fine is a window whose top-R movement is noise, and re-ranking to it
CHASES that noise. C declines those swaps.

Usage:
    simulate_miss_budget.py [RESULTS_DIR]
Defaults to the 2026-08-03 #439 confirm run.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Sequence

DEFAULT_BASE = "/spinning/gpu-battery-results/2026-08-03_439_confirm"
SERIES = ("stats_series_equal", "stats_series_compute", "stats_series_compute-cal")
TAGS = ("tp0ep0", "tp1ep0", "tp2ep0")


def load_series(directory: str, tag: str) -> List[dict]:
    return [
        json.load(open(f))
        for f in sorted(glob.glob(os.path.join(directory, f"*{tag}*.json")))
    ]


def per_window_deltas(snaps: Sequence[dict]) -> Dict[int, dict]:
    """``layer_id -> {E, R, w: [per-window activation vectors]}``.

    The snapshots are CUMULATIVE, so window i is snapshot i+1 minus snapshot i.
    """
    out: Dict[int, dict] = {}
    prev: Dict[int, Sequence[int]] = {}
    for snap in snaps:
        for layer in snap["layers"]:
            lid = int(layer["layer_id"])
            cur = layer["expert_activations"]
            if lid in prev:
                out.setdefault(
                    lid,
                    {
                        "E": int(layer["num_experts"]),
                        "R": int(layer["resident_count"]),
                        "w": [],
                    },
                )
                out[lid]["w"].append([a - b for a, b in zip(cur, prev[lid])])
            prev[lid] = cur
    return out


def static_set(E: int, R: int) -> List[int]:
    """The load-time layout: the #82 pad expert at E-1 is pinned, rest ascend."""
    return [E - 1] + [e for e in range(E) if e != E - 1][: R - 1]


def top_set(acts: Sequence[int], R: int, E: int) -> List[int]:
    """Top-R by observed activation, pad expert kept pinned regardless."""
    order = sorted((e for e in range(E) if e != E - 1), key=lambda e: (-acts[e], e))
    return [E - 1] + order[: R - 1]


def run(directory: str, tag: str, policy) -> tuple:
    """Return (hit_rate, swaps). ``policy`` is None | 'periodic' | float budget."""
    layers = per_window_deltas(load_series(directory, tag))
    hits = total = swaps = 0
    for info in layers.values():
        E, R, windows = info["E"], info["R"], info["w"]
        if R <= 0 or R >= E:
            continue
        resident = set(static_set(E, R))
        for w in windows:
            acts = sum(w)
            if acts == 0:
                continue
            h = sum(w[e] for e in resident if e < len(w))
            hits += h
            total += acts
            if policy == "periodic":
                resident = set(top_set(w, R, E))
                swaps += 1
            elif isinstance(policy, float) and (1 - h / acts) > policy:
                resident = set(top_set(w, R, E))
                swaps += 1
    return (hits / total if total else 0.0), swaps


def main(argv: List[str]) -> int:
    base = argv[1] if len(argv) > 1 else DEFAULT_BASE
    combos = [(s, t) for s in SERIES for t in TAGS]

    print(f"base: {base}\n")
    print(
        f"{'series':>14} {'rank':>7} {'A static':>9} {'B periodic':>11} "
        f"{'C budget .04':>13} {'C-B':>8} {'swaps C/B':>10}"
    )
    deltas = []
    for s, t in combos:
        d = os.path.join(base, s)
        if not os.path.isdir(d):
            print(f"{s:>14} {t:>7}  (absent)")
            continue
        a, _ = run(d, t, None)
        b, bs = run(d, t, "periodic")
        c, cs = run(d, t, 0.04)
        deltas.append(c - b)
        print(
            f"{s[13:]:>14} {t:>7} {a:>9.4f} {b:>11.4f} {c:>13.4f} "
            f"{c-b:>+8.4f} {cs/max(bs,1):>9.1%}"
        )

    if deltas:
        print(
            f"\nwins: {sum(1 for x in deltas if x >= 0)}/{len(deltas)}   "
            f"mean {sum(deltas)/len(deltas):+.4f}   worst {min(deltas):+.4f}"
        )
        print("\nbudget sweep (wins/mean/worst over the same combos):")
        for bud in (0.02, 0.04, 0.06, 0.08, 0.10):
            ds = []
            for s, t in combos:
                d = os.path.join(base, s)
                if not os.path.isdir(d):
                    continue
                b, _ = run(d, t, "periodic")
                c, _ = run(d, t, bud)
                ds.append(c - b)
            print(
                f"  {bud:.2f}: {sum(1 for x in ds if x >= 0)}/{len(ds)} "
                f"mean {sum(ds)/len(ds):+.4f} worst {min(ds):+.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
