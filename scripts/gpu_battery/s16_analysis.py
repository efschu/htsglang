#!/usr/bin/env python3
"""s16 -- turn the structured-output points into the report tables, task #285.

Reads what the step wrote and NOTHING else: structured_points.jsonl and
proofs/<arm>.txt out of the step directory.

Four rules the tables follow, and each of them is a decision that could have
gone the other way:

* THE CONTENT CLASS IS A ROW, NEVER AN AVERAGE. Code, JSON and lists are three
  different workloads with three different acceptance regimes; a mean over
  them is a number no user ever experiences. Nothing in this file ever
  aggregates across classes.
* THE FLOOR IS THIS RUN'S OWN. floor_a and floor_b are the same NEXTN recipe
  booted twice, so their per-cell spread is the between-boot noise of this
  instrument on this content. A difference smaller than the floor of its own
  cell is printed with `~` and is not a finding. Without a floor round the
  tables still print, marked NO FLOOR, and every delta in them is unverdicted.
* ms/Verify IS THE DECODE MEASURE. It is the quantity the rig rule names, and
  the transport windows put its A-vs-A spread far below the spread of tick
  tok/s (which swings with the accept length). tok/s and accept are carried
  for the record.
* AN UNCOUNTED POINT IS NOT A HOLE, IT IS A RESULT. Points that failed their
  output validation or produced no tick rate are listed by name with their
  reason. A structured-output comparison in which one arm silently stopped
  producing parseable JSON would otherwise look like a throughput win.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

RE_MAXTOK = re.compile(r"max_total_num_tokens[= ](\d+)")
RE_SPEC_ALGO = re.compile(r"speculative_algorithm[=:' ]+([A-Za-z0-9_]+)")

FLOOR_ARMS = ("floor_a", "floor_b")

# Metric key -> (label, source field in the point, "lower is better"?)
METRICS = (
    ("ms_per_verify", "ms/Verify", "tick_ms_per_verify", True),
    ("tok_s", "tick tok/s", "tick_gen_tok_s_median", False),
    ("accept_tick", "accept (tick)", "tick_accept_len_median", False),
    ("accept_client", "accept (meta_info)", "client_accept_len_pooled", False),
    ("valid_ratio", "valid share", "valid_ratio", False),
)


def arm_base(arm: str) -> str:
    """``dflash_r2`` -> ``dflash``. The round is a repetition, not an arm."""
    return re.sub(r"_r\d+$", "", arm or "")


def load_points(path: str) -> list:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_proof(step_dir: str, arm: str) -> dict:
    path = os.path.join(step_dir, "proofs", f"{arm}.txt")
    info = {"max_total_num_tokens": None, "algo_seen": None}
    if not os.path.exists(path):
        return info
    with open(path, errors="replace") as f:
        text = f.read()
    tok = RE_MAXTOK.findall(text)
    if tok:
        info["max_total_num_tokens"] = max(int(t) for t in tok)
    m = RE_SPEC_ALGO.search(text)
    if m:
        info["algo_seen"] = m.group(1)
    return info


def cells(points: list) -> dict:
    """(arm_base, bs, class) -> list of per-round metric dicts.

    Only counted points enter. The uncounted ones are reported separately by
    ``uncounted_table``; folding them in as zeros or skipping them silently
    would both be wrong in the same direction.
    """
    out: dict = {}
    for p in points:
        if not p.get("counted"):
            continue
        key = (arm_base(p.get("arm")), p.get("bs"), p.get("content_class"))
        row = {k: p.get(field) for k, _, field, _ in METRICS}
        row["arm_full"] = p.get("arm")
        out.setdefault(key, []).append(row)
    return out


def median_of(rows: list, key: str):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return statistics.median(vals)


def rel_spread(a, b):
    """|a-b| / mean, in percent. None when either side is missing."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    mean = (a + b) / 2.0
    if mean == 0:
        return None
    return abs(a - b) / mean * 100.0


def floor_table(cell_map: dict) -> tuple:
    """Per-cell A-vs-A floors, and the per-metric floor derived from them.

    The derived floor is the MAXIMUM over the cells of a metric, not the mean:
    the floor is being used as a gate ("is this difference real"), and a gate
    set at the average of the observed noise lets through half of the noise by
    construction.
    """
    per_cell: dict = {}
    for (arm, bs, cls), rows in cell_map.items():
        if arm not in FLOOR_ARMS:
            continue
        other = FLOOR_ARMS[1] if arm == FLOOR_ARMS[0] else FLOOR_ARMS[0]
        partner = cell_map.get((other, bs, cls))
        if not partner:
            continue
        if (bs, cls) in per_cell:
            continue
        entry = {}
        for key, _, _, _ in METRICS:
            entry[key] = rel_spread(median_of(rows, key), median_of(partner, key))
        per_cell[(bs, cls)] = entry
    derived = {}
    for key, _, _, _ in METRICS:
        vals = [e[key] for e in per_cell.values() if isinstance(e.get(key), float)]
        derived[key] = max(vals) if vals else None
    return per_cell, derived


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def delta_pct(nextn, dflash, lower_is_better: bool):
    """DFLASH against NEXTN, signed so that positive always means DFLASH won."""
    if not isinstance(nextn, (int, float)) or not isinstance(dflash, (int, float)):
        return None
    if nextn == 0:
        return None
    raw = (dflash - nextn) / nextn * 100.0
    return -raw if lower_is_better else raw


def comparison_table(cell_map: dict, derived_floor: dict, out) -> None:
    keys = sorted(
        {(bs, cls) for (arm, bs, cls) in cell_map if arm in ("nextn", "dflash")},
        key=lambda k: (k[1], k[0]),
    )
    if not keys:
        print("no nextn/dflash cells", file=out)
        return
    for metric_key, label, _, lower_better in METRICS:
        floor = derived_floor.get(metric_key)
        head = f"floor {fmt(floor)} %" if floor is not None else "NO FLOOR"
        print(f"\n### {label}  ({head}, positive delta = DFLASH better)", file=out)
        print("| class | bs | NEXTN | DFLASH | delta % | verdict |", file=out)
        print("|---|---|---|---|---|---|", file=out)
        for bs, cls in keys:
            n_rows = cell_map.get(("nextn", bs, cls), [])
            d_rows = cell_map.get(("dflash", bs, cls), [])
            n = median_of(n_rows, metric_key)
            d = median_of(d_rows, metric_key)
            delta = delta_pct(n, d, lower_better)
            if delta is None:
                verdict = "incomplete"
            elif floor is None:
                verdict = "unverdicted (no floor)"
            elif abs(delta) <= floor:
                verdict = "~ inside floor"
            else:
                verdict = "DFLASH" if delta > 0 else "NEXTN"
            print(
                f"| {cls} | {bs} | {fmt(n)} | {fmt(d)} | {fmt(delta)} | {verdict} |",
                file=out,
            )


def floor_report(per_cell: dict, derived: dict, out) -> None:
    print("\n## A-vs-A floor (floor_a vs floor_b, same NEXTN recipe twice)", file=out)
    if not per_cell:
        print(
            "NO FLOOR ROUND IN THIS RUN. Every delta below is unverdicted: "
            "without a same-recipe pair there is no way to tell a difference "
            "from the between-boot spread of this instrument.",
            file=out,
        )
        return
    cols = " | ".join(label for _, label, _, _ in METRICS)
    print(f"| class | bs | {cols} |", file=out)
    print("|---|---|" + "---|" * len(METRICS), file=out)
    for bs, cls in sorted(per_cell, key=lambda k: (k[1], k[0])):
        entry = per_cell[(bs, cls)]
        vals = " | ".join(fmt(entry.get(k)) for k, _, _, _ in METRICS)
        print(f"| {cls} | {bs} | {vals} |", file=out)
    derived_txt = ", ".join(
        f"{label} {fmt(derived.get(k))} %" for k, label, _, _ in METRICS
    )
    print(f"\nderived gate (max over cells): {derived_txt}", file=out)


def validation_table(points: list, out) -> None:
    print("\n## Output validation", file=out)
    print(
        "| arm | bs | class | in window | valid | share | counted | reasons |", file=out
    )
    print("|---|---|---|---|---|---|---|---|", file=out)
    for p in sorted(
        points,
        key=lambda p: (
            str(p.get("content_class")),
            p.get("bs") or 0,
            str(p.get("arm")),
        ),
    ):
        reasons = ", ".join(
            f"{k}x{v}" for k, v in sorted((p.get("invalid_reasons") or {}).items())
        )
        print(
            f"| {p.get('arm')} | {p.get('bs')} | {p.get('content_class')} "
            f"| {p.get('requests_in_window')} | {p.get('requests_valid')} "
            f"| {fmt(p.get('valid_ratio'))} | {p.get('counted')} | {reasons or '-'} |",
            file=out,
        )


def uncounted_table(points: list, out) -> None:
    bad = [p for p in points if not p.get("counted")]
    print(f"\n## Points that do NOT count ({len(bad)})", file=out)
    if not bad:
        print("none", file=out)
        return
    for p in bad:
        why = "; ".join(p.get("not_counted_because") or []) or p.get("error") or "?"
        print(
            f"- {p.get('arm')} bs={p.get('bs')} {p.get('content_class')}: {why}",
            file=out,
        )


def capacity_table(step_dir: str, points: list, out) -> None:
    """The KV pool each arm actually got, and whether the arms are comparable.

    The DFLASH drafter costs weights the NEXTN head does not, so at equal
    reserve the pools differ. This table is where that shows; if the run pinned
    --max-total-tokens the two columns are equal and the question is closed.
    """
    arms = sorted({p.get("arm") for p in points if p.get("arm")})
    print("\n## Boot proof per arm", file=out)
    print("| arm | max_total_num_tokens | speculative_algorithm in log |", file=out)
    print("|---|---|---|", file=out)
    caps = {}
    for arm in arms:
        info = load_proof(step_dir, arm)
        caps[arm] = info["max_total_num_tokens"]
        print(
            f"| {arm} | {fmt(info['max_total_num_tokens'])} | "
            f"{info['algo_seen'] or '-'} |",
            file=out,
        )
    known = [v for v in caps.values() if isinstance(v, int)]
    if len(known) >= 2 and max(known) != min(known):
        spread = (max(known) - min(known)) / max(known) * 100.0
        print(
            f"\nKV pool spread across arms: {spread:.1f} %. The arms did not run "
            "the same pool; pin S16_MAX_TOTAL_TOKENS to the smaller capacity and "
            "repeat before quoting a decode verdict.",
            file=out,
        )
    elif known:
        print("\nKV pool identical across arms.", file=out)


def compose(step_dir: str, points: list, out) -> dict:
    cell_map = cells(points)
    per_cell, derived = floor_table(cell_map)

    print("# s16 -- DFLASH vs NEXTN on structured output", file=out)
    print(
        f"\npoints: {len(points)} total, "
        f"{sum(1 for p in points if p.get('counted'))} counted, "
        f"{len({arm_base(p.get('arm')) for p in points})} arms, "
        f"{len({p.get('content_class') for p in points})} content classes",
        file=out,
    )
    print(
        "\nNothing in these tables is averaged across content classes. "
        "The class is the workload.",
        file=out,
    )
    capacity_table(step_dir, points, out)
    floor_report(per_cell, derived, out)
    print("\n## DFLASH against NEXTN, per class and batch size", file=out)
    comparison_table(cell_map, derived, out)
    validation_table(points, out)
    uncounted_table(points, out)

    return {
        "points_total": len(points),
        "points_counted": sum(1 for p in points if p.get("counted")),
        "floor_per_cell": {f"{bs}:{cls}": v for (bs, cls), v in per_cell.items()},
        "floor_derived": derived,
        "cells": {
            f"{arm}:{bs}:{cls}": {key: median_of(rows, key) for key, _, _, _ in METRICS}
            for (arm, bs, cls), rows in cell_map.items()
        },
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step-dir", required=True)
    p.add_argument(
        "--points",
        default="",
        help="points file; default <step-dir>/structured_points.jsonl",
    )
    p.add_argument("--json", default="", help="also write the summary as JSON here")
    args = p.parse_args(argv)

    points_path = args.points or os.path.join(args.step_dir, "structured_points.jsonl")
    points = load_points(points_path)
    if not points:
        print(f"STOP: no points in {points_path}", file=sys.stderr)
        return 2

    summary = compose(args.step_dir, points, sys.stdout)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
