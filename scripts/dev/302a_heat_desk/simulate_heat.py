#!/usr/bin/env python3
"""#302a desk falsifier -- can re-ranking the resident expert set help?

Runs entirely off recorded `expert_stats_*.json` artifacts (#390 instrument).
No GPU, no model, no build. Answers three questions:

1. Can the recorded static hit rate be REPRODUCED from the JSON alone?
   (sanity gate: if the reconstruction does not match the recorded number the
   simulation below is measuring something else and must not be trusted)
2. What is the ORACLE ceiling -- the hit rate a resident set of the SAME size
   would have reached if it had been chosen by whole-run activation count?
   This is an upper bound on ANY re-ranking policy, windowed or not.
3. Where does the ceiling come from, per layer, so a weak verdict can name the
   structural reason rather than only the number.

Usage: python3 simulate_heat.py <expert_stats.json> [more.json ...]
"""

from __future__ import annotations

import json
import sys
from typing import List, Sequence


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def static_resident_set(num_experts: int, resident_count: int) -> List[int]:
    """The load-time layout of `plan_load_time_staging` for the recorded runs.

    `pinned_experts` is the #82 expert-dim pad expert at id E-1; pinned ids take
    the lowest slots and the rest fill in ascending id order.
    """
    pinned = [num_experts - 1]
    rest = [e for e in range(num_experts) if e not in set(pinned)]
    return pinned + rest[: resident_count - len(pinned)]


def hits_for(acts: Sequence[int], resident: Sequence[int]) -> int:
    return sum(int(acts[e]) for e in resident if e < len(acts))


def oracle_resident_set(
    acts: Sequence[int], resident_count: int, pinned: Sequence[int] = ()
) -> List[int]:
    """Top-R experts by activation count, pinned ids kept regardless."""
    pinned = sorted({int(e) for e in pinned})
    pin = set(pinned)
    order = sorted(
        (e for e in range(len(acts)) if e not in pin),
        key=lambda e: (-int(acts[e]), e),
    )
    return pinned + order[: resident_count - len(pinned)]


def analyse(path: str) -> dict:
    d = load(path)
    tag = d.get("rank_tag", path)
    rows = []
    tot_act = tot_static = tot_oracle = 0
    tot_static_nopad = tot_oracle_nopad = tot_act_nopad = 0
    for layer in d["layers"]:
        acts = layer["expert_activations"]
        E = int(layer["num_experts"])
        R = int(layer["resident_count"])
        if R <= 0 or R >= E:
            continue
        stat = static_resident_set(E, R)
        orac = oracle_resident_set(acts, R, pinned=[E - 1])
        a = int(layer["activations"])
        hs = hits_for(acts, stat)
        ho = hits_for(acts, orac)
        pad = int(acts[E - 1])
        rows.append(
            dict(
                layer=int(layer["layer_id"]),
                E=E,
                R=R,
                acts=a,
                recorded_hits=int(layer["hit_activations"]),
                recorded_rate=float(layer["hit_rate"]),
                static_hits=hs,
                static_rate=hs / a if a else 0.0,
                oracle_hits=ho,
                oracle_rate=ho / a if a else 0.0,
                pad_share=pad / a if a else 0.0,
                # pad-excluded view: the pad expert is a structural always-hit,
                # it says nothing about placement quality.
                static_rate_nopad=(hs - pad) / (a - pad) if a > pad else 0.0,
                oracle_rate_nopad=(ho - pad) / (a - pad) if a > pad else 0.0,
            )
        )
        tot_act += a
        tot_static += hs
        tot_oracle += ho
        tot_act_nopad += a - pad
        tot_static_nopad += hs - pad
        tot_oracle_nopad += ho - pad
    return dict(
        tag=tag,
        path=path,
        recorded_rate=float(d["totals"]["hit_rate"]),
        recorded_hits=int(d["totals"]["hit_activations"]),
        acts=tot_act,
        static_hits=tot_static,
        static_rate=tot_static / tot_act if tot_act else 0.0,
        oracle_hits=tot_oracle,
        oracle_rate=tot_oracle / tot_act if tot_act else 0.0,
        static_rate_nopad=tot_static_nopad / tot_act_nopad if tot_act_nopad else 0.0,
        oracle_rate_nopad=tot_oracle_nopad / tot_act_nopad if tot_act_nopad else 0.0,
        rows=rows,
    )


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    reports = [analyse(p) for p in argv[1:]]
    print(
        f"{'rank':<8} {'recorded':>9} {'reconstr':>9} {'delta':>8} "
        f"{'oracle':>9} {'lift_pp':>8} {'nopad_st':>9} {'nopad_or':>9} {'nopad_pp':>9}"
    )
    for r in reports:
        print(
            f"{r['tag']:<8} {r['recorded_rate']:>9.4f} {r['static_rate']:>9.4f} "
            f"{r['static_rate'] - r['recorded_rate']:>8.4f} {r['oracle_rate']:>9.4f} "
            f"{100 * (r['oracle_rate'] - r['static_rate']):>8.2f} "
            f"{r['static_rate_nopad']:>9.4f} {r['oracle_rate_nopad']:>9.4f} "
            f"{100 * (r['oracle_rate_nopad'] - r['static_rate_nopad']):>9.2f}"
        )
    print()
    for r in reports:
        print(f"--- {r['tag']} per-layer (first 8 + worst 8 by oracle lift) ---")
        rows = r["rows"]
        worst = sorted(rows, key=lambda x: -(x["oracle_rate"] - x["static_rate"]))[:8]
        shown = rows[:8] + [None] + worst
        print(
            f"{'layer':>5} {'E':>4} {'R':>4} {'acts':>8} {'rec':>7} {'stat':>7} "
            f"{'orac':>7} {'lift_pp':>8} {'pad%':>7}"
        )
        for x in shown:
            if x is None:
                print("   ...")
                continue
            print(
                f"{x['layer']:>5} {x['E']:>4} {x['R']:>4} {x['acts']:>8} "
                f"{x['recorded_rate']:>7.4f} {x['static_rate']:>7.4f} "
                f"{x['oracle_rate']:>7.4f} "
                f"{100 * (x['oracle_rate'] - x['static_rate']):>8.2f} "
                f"{100 * x['pad_share']:>7.2f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
