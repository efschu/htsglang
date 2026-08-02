#!/usr/bin/env python3
"""#302a desk falsifier, part 2 -- does a heat ranking GENERALISE?

`simulate_heat.py` computes the ORACLE ceiling: the hit rate a same-sized
resident set would reach if it had been ranked by the very run it is scored on.
That is an upper bound and is not achievable by any online policy, because an
online policy ranks on PAST traffic and is scored on FUTURE traffic.

This script measures the achievable part instead: rank the resident set from
run A's recorded activations, score it on run B's. A and B are different boots
on different days with different workloads, so this is a HARSHER staleness test
than a re-rank every N decode steps inside one serving session -- it is the
limit case of a completely stale ranking.

It also runs the #302-lookahead question: does layer N's heat predict layer
N+1's? Honest scope note -- the recorded JSONs hold WHOLE-RUN per-expert
activation totals, not a per-token routing trace, so "layer N's top-k predicts
layer N+1's top-k FOR THE SAME TOKEN" is NOT answerable from this artifact.
What is answerable is the aggregate form: how much of layer N's hot set is
layer N+1's hot set, i.e. whether one ranking could serve several layers.

Usage: transfer_heat.py --train <A.json> --test <B.json> [--test <C.json> ...]
       transfer_heat.py --lookahead <run.json>
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence, Tuple


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def layers_by_id(d: dict) -> Dict[int, dict]:
    return {int(x["layer_id"]): x for x in d["layers"]}


def static_set(E: int, R: int) -> List[int]:
    pinned = [E - 1]
    rest = [e for e in range(E) if e != E - 1]
    return pinned + rest[: R - len(pinned)]


def ranked_set(acts: Sequence[int], R: int, pinned: Sequence[int]) -> List[int]:
    pin = sorted({int(e) for e in pinned})
    ps = set(pin)
    order = sorted(
        (e for e in range(len(acts)) if e not in ps), key=lambda e: (-int(acts[e]), e)
    )
    return pin + order[: R - len(pin)]


def score(acts: Sequence[int], resident: Sequence[int]) -> Tuple[int, int]:
    tot = sum(int(a) for a in acts)
    hit = sum(int(acts[e]) for e in resident if e < len(acts))
    return hit, tot


def transfer(train_path: str, test_path: str) -> dict:
    tr, te = load(train_path), load(test_path)
    TR, TE = layers_by_id(tr), layers_by_id(te)
    agg = dict(static=0, trained=0, oracle=0, total=0)
    per_layer = []
    for lid, tl in sorted(TE.items()):
        if lid not in TR:
            continue
        E, R = int(tl["num_experts"]), int(tl["resident_count"])
        if R <= 0 or R >= E:
            continue
        trl = TR[lid]
        if int(trl["num_experts"]) != E:
            continue
        a_te = tl["expert_activations"]
        a_tr = trl["expert_activations"]
        s_static = static_set(E, R)
        s_train = ranked_set(a_tr, R, [E - 1])
        s_oracle = ranked_set(a_te, R, [E - 1])
        h_s, tot = score(a_te, s_static)
        h_t, _ = score(a_te, s_train)
        h_o, _ = score(a_te, s_oracle)
        agg["static"] += h_s
        agg["trained"] += h_t
        agg["oracle"] += h_o
        agg["total"] += tot
        per_layer.append(
            dict(
                layer=lid,
                static=h_s / tot if tot else 0.0,
                trained=h_t / tot if tot else 0.0,
                oracle=h_o / tot if tot else 0.0,
                set_overlap=len(set(s_train) & set(s_oracle)) / R,
            )
        )
    t = agg["total"] or 1
    return dict(
        train=train_path,
        test=test_path,
        static=agg["static"] / t,
        trained=agg["trained"] / t,
        oracle=agg["oracle"] / t,
        per_layer=per_layer,
    )


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")

    def rank(v: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def lookahead(path: str) -> dict:
    d = load(path)
    L = sorted(d["layers"], key=lambda x: int(x["layer_id"]))
    rows = []
    for a, b in zip(L, L[1:]):
        E = int(a["num_experts"])
        R = int(a["resident_count"])
        if int(b["num_experts"]) != E:
            continue
        va = [int(v) for v in a["expert_activations"]]
        vb = [int(v) for v in b["expert_activations"]]
        # exclude the #82 pad expert: it is a structural constant in both
        # layers and would inflate every correlation.
        va_n, vb_n = va[: E - 1], vb[: E - 1]
        sa = set(ranked_set(va, R, [E - 1]))
        sb = set(ranked_set(vb, R, [E - 1]))
        rows.append(
            dict(
                pair=f"{a['layer_id']}->{b['layer_id']}",
                spearman=spearman(va_n, vb_n),
                topR_jaccard=len(sa & sb) / len(sa | sb),
                topR_overlap=len(sa & sb) / R,
                # top-8 EXCLUDING the pad expert. Including it guarantees a
                # 1/8 overlap floor for free (it is the top expert in every
                # layer) and would read as signal where there is none.
                top8_overlap=len(
                    set(ranked_set(va_n, 8, [])) & set(ranked_set(vb_n, 8, []))
                )
                / 8.0,
                top8_chance=8.0 / (E - 1),
            )
        )
    # baseline: same-layer against a random-id ranking -> expected overlap R/E
    E = int(L[0]["num_experts"])
    R = int(L[0]["resident_count"])
    return dict(path=path, rows=rows, chance_overlap=R / E)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--test", action="append", default=[])
    ap.add_argument("--lookahead", action="append", default=[])
    ap.add_argument("--per-layer", action="store_true")
    args = ap.parse_args()

    if args.train:
        print(
            f"{'test run':<62} {'static':>7} {'trained':>8} {'oracle':>7} "
            f"{'lift_pp':>8} {'cap_%':>7}"
        )
        for tp in args.test:
            r = transfer(args.train, tp)
            lift = r["trained"] - r["static"]
            ceil = r["oracle"] - r["static"]
            cap = 100 * lift / ceil if ceil else float("nan")
            name = tp.replace("/spinning/gpu-battery-results/", "")
            print(
                f"{name:<62} {r['static']:>7.4f} {r['trained']:>8.4f} "
                f"{r['oracle']:>7.4f} {100 * lift:>8.2f} {cap:>7.1f}"
            )
            if args.per_layer:
                for x in r["per_layer"]:
                    print(
                        f"    L{x['layer']:<3} static={x['static']:.4f} "
                        f"trained={x['trained']:.4f} oracle={x['oracle']:.4f} "
                        f"set_overlap={x['set_overlap']:.3f}"
                    )
        print()

    for lp in args.lookahead:
        r = lookahead(lp)
        rows = r["rows"]
        n = len(rows)
        print(f"--- lookahead {lp.replace('/spinning/gpu-battery-results/', '')} ---")
        print(
            f"  adjacent-layer pairs: {n}, chance top-R overlap = {r['chance_overlap']:.3f}"
        )
        for key in (
            "spearman",
            "topR_overlap",
            "topR_jaccard",
            "top8_overlap",
            "top8_chance",
        ):
            vals = sorted(x[key] for x in rows)
            mean = sum(vals) / n
            print(
                f"  {key:<14} mean={mean:>7.4f} min={vals[0]:>7.4f} "
                f"p50={vals[n // 2]:>7.4f} max={vals[-1]:>7.4f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
