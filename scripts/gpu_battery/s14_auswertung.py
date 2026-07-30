#!/usr/bin/env python3
"""s14 -- decode_punkte.jsonl to the tables the #294 verdict is written from.

Runs in the container, reads nothing but the run directory. It does not judge:
it prints the floor, the per-point table and the ratios, and the verdict is
written by hand into docs/dev/INTEGRATION_R3_VALIDATION.md.

THE FLOOR IS PRINTED BEFORE THE RATIOS, and every ratio is printed next to it,
because a ratio smaller than the floor of its own arm is not a finding. The
floor here is the spread of REPEATS OF THE SAME ARM at the same batch size --
within one boot and across boots -- expressed as (max-min)/median, which is the
quantity a between-arm difference has to clear.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

# The metric LABELS stay verbatim. `ms/Verify` is the name s12 defines for the
# whole battery, `ms/Schritt` is its sibling, and every table in the validation
# doc is quoted under those two names.
METRICS = (
    ("tick_ms_pro_verify", "ms/Verify"),
    ("tick_gen_tok_s_median", "tok/s (tick)"),
    ("klient_tok_s", "tok/s (client)"),
    ("tick_accept_len_median", "accept"),
    ("tick_ms_pro_schritt", "ms/Schritt"),
)


def load(path: str) -> list:
    points = []
    if not os.path.exists(path):
        return points
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                points.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return points


def _arm_base(arm: str) -> str:
    """`bar1_hi_r2` -> `bar1_hi`. The round belongs to the sample, not the arm."""
    return arm.rsplit("_r", 1)[0] if "_r" in arm else arm


def _spread(values: list) -> dict:
    values = [w for w in values if isinstance(w, (int, float))]
    if not values:
        return {"n": 0}
    med = statistics.median(values)
    out = {
        "n": len(values),
        "median": med,
        "min": min(values),
        "max": max(values),
        "spanne_rel": (max(values) - min(values)) / med if med else None,
    }
    if len(values) > 2:
        out["stdev_rel"] = statistics.stdev(values) / med if med else None
    return out


def _fmt(x, nk=2) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "NO"
    if isinstance(x, float):
        return f"{x:.{nk}f}"
    return str(x)


def table_points(points: list) -> str:
    z = [
        "| Arm | bs | Rep | ms/Verify | ms/Schritt | tok/s tick | tok/s client "
        "| accept (tick) | accept (client) | ticks counted/bs | foreign bs "
        "| Graph |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    counter: dict = {}
    for p in sorted(points, key=lambda q: (q.get("arm", ""), q.get("folge", 0))):
        key = (p.get("arm"), p.get("bs"))
        counter[key] = counter.get(key, 0) + 1
        z.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} |".format(
                p.get("arm"),
                p.get("bs"),
                counter[key],
                _fmt(p.get("tick_ms_pro_verify")),
                _fmt(p.get("tick_ms_pro_schritt"), 1),
                _fmt(p.get("tick_gen_tok_s_median"), 1),
                _fmt(p.get("klient_tok_s"), 1),
                _fmt(p.get("tick_accept_len_median")),
                _fmt(p.get("klient_accept_len_gesamt")),
                _fmt(p.get("tick_ticks_gewertet")),
                _fmt(p.get("tick_ticks_bs")),
                _fmt(p.get("tick_ticks_fremde_bs")),
                _fmt(p.get("tick_cuda_graph")),
            )
        )
    return "\n".join(z)


def table_floor(points: list) -> str:
    """Repeat spread per (arm, bs). This is the floor, and it comes first."""
    groups: dict = {}
    for p in points:
        groups.setdefault((_arm_base(p.get("arm", "")), p.get("bs")), []).append(p)
    z = [
        "| Arm | bs | Rep | Metric | Median | min | max "
        "| Spread (max-min)/median | rel. stdev |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for (arm, bs), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        if len(group) < 2:
            continue
        for field, name in METRICS:
            s = _spread([g.get(field) for g in group])
            if s.get("n", 0) < 2:
                continue
            z.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    arm,
                    bs,
                    s["n"],
                    name,
                    _fmt(s["median"]),
                    _fmt(s["min"]),
                    _fmt(s["max"]),
                    _fmt(100.0 * s["spanne_rel"], 2) + " %" if s["spanne_rel"] is not None else "-",
                    _fmt(100.0 * s["stdev_rel"], 2) + " %" if s.get("stdev_rel") is not None else "-",
                )
            )
    return "\n".join(z) if len(z) > 2 else "(no repeat in this run)"


def table_ratio(points: list, arm_a: str, arm_b: str) -> str:
    """arm_a against arm_b per batch size, with both floors next to the ratio."""
    per: dict = {}
    for p in points:
        per.setdefault((_arm_base(p.get("arm", "")), p.get("bs")), []).append(p)
    bs_values = sorted({bs for (_, bs) in per if bs is not None})
    z = [
        f"| bs | ms/Verify {arm_a} | ms/Verify {arm_b} | Factor | tok/s {arm_a} "
        f"| tok/s {arm_b} | Factor | accept {arm_a} | accept {arm_b} "
        "| floor ms/Verify (max of both arms) | above floor |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for bs in bs_values:
        a = per.get((arm_a, bs)) or []
        b = per.get((arm_b, bs)) or []
        if not a or not b:
            continue
        va = _spread([p.get("tick_ms_pro_verify") for p in a])
        vb = _spread([p.get("tick_ms_pro_verify") for p in b])
        ra = _spread([p.get("tick_gen_tok_s_median") for p in a])
        rb = _spread([p.get("tick_gen_tok_s_median") for p in b])
        aa = _spread([p.get("tick_accept_len_median") for p in a])
        ab = _spread([p.get("tick_accept_len_median") for p in b])
        if not va.get("median") or not vb.get("median"):
            continue
        factor = vb["median"] / va["median"]
        # A single sample has no spread, and _spread reports 0.0 for it. Taking
        # that as a floor would clear every difference against a floor of zero,
        # which is the opposite of what a floor is for: a point without a
        # repetition has NO floor and says so.
        spreads = [
            s["spanne_rel"]
            for s in (va, vb)
            if s.get("n", 0) >= 2 and s.get("spanne_rel") is not None
        ]
        if len(spreads) < 2:
            floor_text, above = "-", "?"
        else:
            # The difference has to clear the floor of the noisier of the two
            # arms; the floor is a relative spread, so the comparison is on
            # |factor - 1|.
            floor = max(spreads)
            floor_text = _fmt(100.0 * floor, 2) + " %"
            above = "yes" if abs(factor - 1.0) > floor else "NO"
        z.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                bs,
                _fmt(va["median"]),
                _fmt(vb["median"]),
                _fmt(factor, 3),
                _fmt(ra.get("median"), 1),
                _fmt(rb.get("median"), 1),
                _fmt((ra["median"] / rb["median"]) if ra.get("median") and rb.get("median") else None, 3),
                _fmt(aa.get("median")),
                _fmt(ab.get("median")),
                floor_text,
                above,
            )
        )
    return "\n".join(z)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step-dir", required=True)
    p.add_argument("--arm-a", default="bar1_hi")
    p.add_argument("--arm-b", default="nccl_hi")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    points = load(os.path.join(args.step_dir, "decode_punkte.jsonl"))
    if not points:
        print("no points in decode_punkte.jsonl", file=sys.stderr)
        return 1

    parts = [
        f"### Noise floor -- repeats of the same arm ({len(points)} points)",
        "",
        table_floor(points),
        "",
        "### The points one by one",
        "",
        table_points(points),
        "",
        f"### {args.arm_a} against {args.arm_b}",
        "",
        table_ratio(points, args.arm_a, args.arm_b),
        "",
    ]
    text = "\n".join(parts)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
