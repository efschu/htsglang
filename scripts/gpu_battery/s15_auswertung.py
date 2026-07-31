#!/usr/bin/env python3
"""s15 -- #296: turn the six arms' raw points into the report tables.

Reads what the step wrote and NOTHING else: punkte.jsonl (prefill + decode per
arm and session count), wait/<arm>.json (the compute/wait split per rank),
proofs/<arm>.txt (the plan the boot actually ran with, and its KV capacity) and
power/<arm>.csv (1 s NVML samples for the joule-per-token approximation).

Three rules the tables follow:

* THE DECODE MEASURE IS ms/Verify. The #294 floor run put the A-vs-A spread of
  ms/Verify at 2.72 %, while tick tok/s and accept length swing together by
  about 7.5 % (accept noise). tok/s is carried in the table for the record and
  is never the basis of a verdict.
* THE PREFILL FLOORS ARE REUSED, not re-measured: 2.71 % at s=1 and 3.18 % at
  s=8, from 2026-07-30_hebel_verif. One pass has no A-vs-A of its own, which is
  exactly why the floors have to come from a run that did.
* ANYTHING INSIDE ITS FLOOR IS MARKED `~` AND IS NOT A FINDING.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys

# Reused noise floors, in percent. Prefill: 2026-07-30_hebel_verif, A-vs-A over
# two rounds of every arm. Decode: the #294 floor for ms/Verify.
FLOOR_PREFILL = {1: 2.71, 8: 3.18}
FLOOR_DECODE_MS_VERIFY = 2.72
FLOOR_DECODE_TOK_S = 7.5

ARM_ORDER = [
    "anchor",
    "prefill_opt",
    "decode_opt",
    "decode_opt_kv111",
    "prefill_opt_nccl",
    "decode_opt_nccl",
]
ARM_LABEL = {
    "anchor": "1 anchor (auto split, KV 7,3,3)",
    "prefill_opt": "2 prefill opt (MLP 10,1,1, KV 7,3,3)",
    "decode_opt": "3 decode opt (MLP 7,3,3, KV 7,3,3)",
    "decode_opt_kv111": "4 decode opt + balanced DCP (KV 1,1,1)",
    "prefill_opt_nccl": "5 prefill opt, NCCL",
    "decode_opt_nccl": "6 decode opt, NCCL",
}

RE_MLP_UNITS = re.compile(r"materialized MLP units \[([0-9,\s]+)\]")
# Two spellings, and the authoritative one is the scheduler's own final
# report `max_total_num_tokens=N`. The weighted-DCP sizing line spells it
# `global max_total_num_tokens N` and is absent on a uniform KV vector
# (arm 4), which is why matching only that form left arm 4 without a capacity.
RE_MAXTOK = re.compile(r"max_total_num_tokens=(\d+)")
RE_MAXTOK_DCP = re.compile(r"global max_total_num_tokens (\d+)")
RE_PINNED = re.compile(r"MLP vector PINNED \(([^)]*)\)")


def load_points(path: str) -> dict:
    """(arm, sessions) -> point."""
    out: dict = {}
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(p.get("arm"), p.get("sessions"))] = p
    return out


def load_proof(step_dir: str, arm: str) -> dict:
    path = os.path.join(step_dir, "proofs", f"{arm}.txt")
    info = {
        "mlp_units": None,
        "max_total_num_tokens": None,
        "pinned": None,
        "barlink_groups": 0,
    }
    if not os.path.exists(path):
        return info
    with open(path, errors="replace") as f:
        text = f.read()
    m = RE_MLP_UNITS.search(text)
    if m:
        info["mlp_units"] = [int(x) for x in m.group(1).split(",")]
    m = RE_PINNED.search(text)
    if m:
        info["pinned"] = m.group(1)
    tok = RE_MAXTOK.findall(text) or RE_MAXTOK_DCP.findall(text)
    if tok:
        info["max_total_num_tokens"] = max(int(t) for t in tok)
    info["barlink_groups"] = text.count("barlink enabled for group")
    return info


def load_power(step_dir: str, arm: str) -> dict:
    """Mean rig power over the measurement window, plus the power states seen.

    The window is the whole arm (both points), so the number is an ARM-level
    average and the joule-per-token figure derived from it is an
    approximation -- it charges idle gaps between the points to the tokens.
    Named as such wherever it is printed.
    """
    path = os.path.join(step_dir, "power", f"{arm}.csv")
    out = {"mean_w": None, "samples": 0, "pstates": ""}
    if not os.path.exists(path):
        return out
    per_ts: dict = {}
    states: dict = {}
    with open(path, errors="replace") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            ts = row[0].strip()
            try:
                watt = float(row[2])
            except ValueError:
                continue
            per_ts[ts] = per_ts.get(ts, 0.0) + watt
            st = row[3].strip()
            states[st] = states.get(st, 0) + 1
    if not per_ts:
        return out
    out["mean_w"] = statistics.mean(per_ts.values())
    out["samples"] = len(per_ts)
    out["pstates"] = ", ".join(
        f"{k}:{v}" for k, v in sorted(states.items(), key=lambda kv: -kv[1])
    )
    return out


def load_wait(step_dir: str, arm: str) -> dict:
    path = os.path.join(step_dir, "wait", f"{arm}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, errors="replace") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def ratio_mark(value, base, floor_pct: float) -> str:
    if value is None or base in (None, 0):
        return "-"
    r = value / base
    inside = abs(r - 1.0) * 100.0 < floor_pct
    return f"{'~' if inside else ''}{r:.3f}"


def fmt(v, digits=1):
    return "-" if v is None else f"{v:.{digits}f}"


def decode_of(point, batch):
    for d in (point or {}).get("decode") or []:
        if d.get("batch") == batch:
            return d
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    step = args.step_dir
    points = load_points(os.path.join(step, "punkte.jsonl"))
    arms = [a for a in ARM_ORDER if any(k[0] == a for k in points)]
    missing = [a for a in ARM_ORDER if a not in arms]

    proofs = {a: load_proof(step, a) for a in arms}
    powers = {a: load_power(step, a) for a in arms}
    waits = {a: load_wait(step, a) for a in arms}

    lines: list = []
    w = lines.append

    w("### Plan actually booted, per arm")
    w("")
    w(
        "| Arm | MLP pin | materialized MLP units | max_total_num_tokens | barlink groups |"
    )
    w("|---|---|---|---:|---:|")
    for a in arms:
        p = proofs[a]
        w(
            f"| {ARM_LABEL[a]} | {p['pinned'] or 'none (auto)'} | "
            f"{p['mlp_units'] or '-'} | {p['max_total_num_tokens'] or '-'} | "
            f"{p['barlink_groups']} |"
        )
    w("")

    base = "anchor"
    w("### Prefill throughput (tok/s) and ratio against the anchor")
    w("")
    w(
        "| Arm | s=1 | vs anchor | s=8 | vs anchor | TTFT p50 ms (s=1) | TTFT p50 ms (s=8) |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|")
    b1 = ((points.get((base, 1)) or {}).get("prefill") or {}).get("prefill_tok_s")
    b8 = ((points.get((base, 8)) or {}).get("prefill") or {}).get("prefill_tok_s")
    for a in arms:
        p1 = (points.get((a, 1)) or {}).get("prefill") or {}
        p8 = (points.get((a, 8)) or {}).get("prefill") or {}
        w(
            f"| {ARM_LABEL[a]} | {fmt(p1.get('prefill_tok_s'))} | "
            f"{ratio_mark(p1.get('prefill_tok_s'), b1, FLOOR_PREFILL[1])} | "
            f"{fmt(p8.get('prefill_tok_s'))} | "
            f"{ratio_mark(p8.get('prefill_tok_s'), b8, FLOOR_PREFILL[8])} | "
            f"{fmt(p1.get('latenz_ms_p50'))} | {fmt(p8.get('latenz_ms_p50'))} |"
        )
    w("")
    w(
        f"Floors reused from 2026-07-30_hebel_verif: s=1 {FLOOR_PREFILL[1]} %, "
        f"s=8 {FLOOR_PREFILL[8]} %. `~` = inside the floor, not a finding."
    )
    w("")

    w("### Decode at the s=8 boot point -- ms/Verify is the measure")
    w("")
    w(
        "| Arm | bs=1 ms/Verify | vs anchor | bs=1 tok/s | bs=1 accept | ticks | "
        "bs=8 ms/Verify | vs anchor | bs=8 tok/s | bs=8 accept | ticks |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    bd1 = decode_of(points.get((base, 8)), 1).get("tick_ms_pro_verify")
    bd8 = decode_of(points.get((base, 8)), 8).get("tick_ms_pro_verify")
    for a in arms:
        d1 = decode_of(points.get((a, 8)), 1)
        d8 = decode_of(points.get((a, 8)), 8)
        w(
            f"| {ARM_LABEL[a]} | {fmt(d1.get('tick_ms_pro_verify'), 2)} | "
            f"{ratio_mark(d1.get('tick_ms_pro_verify'), bd1, FLOOR_DECODE_MS_VERIFY)} | "
            f"{fmt(d1.get('tick_gen_tok_s_median'))} | "
            f"{fmt(d1.get('tick_accept_len_median'), 2)} | "
            f"{d1.get('tick_ticks_gewertet', '-')} | "
            f"{fmt(d8.get('tick_ms_pro_verify'), 2)} | "
            f"{ratio_mark(d8.get('tick_ms_pro_verify'), bd8, FLOOR_DECODE_MS_VERIFY)} | "
            f"{fmt(d8.get('tick_gen_tok_s_median'))} | "
            f"{fmt(d8.get('tick_accept_len_median'), 2)} | "
            f"{d8.get('tick_ticks_gewertet', '-')} |"
        )
    w("")
    w(
        f"ms/Verify floor {FLOOR_DECODE_MS_VERIFY} % (#294); tick tok/s and accept "
        f"swing together by ~{FLOOR_DECODE_TOK_S} %, so tok/s is informational only. "
        "A lower ms/Verify is better; a ratio below 1.000 is a decode WIN."
    )
    w("")

    w("### compute / wait per rank at the s=8 prefill point")
    w("")
    w(
        "| Arm | TP0 comp | TP0 wait | TP1 comp | TP1 wait | TP2 comp | TP2 wait | wait share TP0 |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in arms:
        j = waits.get(a) or {}
        by_rank = {e.get("rang"): e for e in (j.get("wait") or [])}
        cells = []
        share = None
        for r in range(3):
            e = by_rank.get(r, {})
            comp = e.get("compute_ms_median")
            wt = e.get("wait_ms_median")
            cells += [fmt(comp), fmt(wt)]
            if r == 0 and e.get("wait_anteil") is not None:
                share = 100.0 * e["wait_anteil"]
        w(f"| {ARM_LABEL[a]} | " + " | ".join(cells) + f" | {fmt(share)} % |")
    w("")
    w(
        "This is the quantitative test of the reopening thesis: BAR1 lowered the "
        "collective floor, and the design note says the phase-dual gain scales "
        "inversely with it. What is left in `wait` after the transport fix is "
        "the COMPUTE imbalance -- the part a phase-dual split can still take."
    )
    w("")

    w("### KV capacity, power and joule per token (approximation)")
    w("")
    w(
        "| Arm | max_total_num_tokens | mean rig W | samples | power states | J/token (bs=8 decode) |"
    )
    w("|---|---:|---:|---:|---|---:|")
    for a in arms:
        pw = powers[a]
        d8 = decode_of(points.get((a, 8)), 8)
        rate = d8.get("tick_gen_tok_s_median")
        jt = (pw["mean_w"] / rate) if (pw["mean_w"] and rate) else None
        w(
            f"| {ARM_LABEL[a]} | {proofs[a]['max_total_num_tokens'] or '-'} | "
            f"{fmt(pw['mean_w'])} | {pw['samples']} | {pw['pstates'] or '-'} | "
            f"{fmt(jt, 2)} |"
        )
    w("")
    w(
        "J/token is an APPROXIMATION: the power window covers the whole arm "
        "(both points and the gaps between them), while the rate is the bs=8 "
        "decode tick rate. It compares arms against each other, not against an "
        "absolute energy budget."
    )
    w("")

    # --- the three verdicts the window was opened for ----------------------
    def pf(a, s):
        return ((points.get((a, s)) or {}).get("prefill") or {}).get("prefill_tok_s")

    def dv(a, b):
        return decode_of(points.get((a, 8)), b).get("tick_ms_pro_verify")

    w("### Cross costs -- the ceiling of the dynamic ladder (#274/#287)")
    w("")
    p2, p3 = pf("prefill_opt", 8), pf("decode_opt", 8)
    d2, d3 = dv("prefill_opt", 8), dv("decode_opt", 8)
    if p2 and p3:
        w(
            f"* Prefill s=8: prefill optimum {p2:.1f} vs decode optimum {p3:.1f} "
            f"tok/s -- the decode optimum costs "
            f"{100.0 * (p2 - p3) / p2:+.1f} % of prefill "
            f"(floor {FLOOR_PREFILL[8]} %)."
        )
    if d2 and d3:
        w(
            f"* Decode bs=8 ms/Verify: prefill optimum {d2:.2f} vs decode optimum "
            f"{d3:.2f} ms -- the prefill optimum costs "
            f"{100.0 * (d2 - d3) / d3:+.1f} % of the verify round "
            f"(floor {FLOOR_DECODE_MS_VERIFY} %)."
        )
    w("")
    w(
        "The spread between arm 2 and arm 3 is the UPPER BOUND of what a dynamic "
        "phase ladder can harvest: it is what a perfect, cost-free switch between "
        "the two static extrema would win. A real ladder pays switching cost on "
        "top, so it lands strictly below this."
    )
    w("")

    w("### KV placement effect (arm 4 - arm 3)")
    w("")
    d4 = dv("decode_opt_kv111", 8)
    d4a = dv("decode_opt_kv111", 1)
    d3a = dv("decode_opt", 1)
    for label, x, y in (("bs=1", d4a, d3a), ("bs=8", d4, d3)):
        if x and y:
            delta = 100.0 * (x - y) / y
            mark = " (inside the floor)" if abs(delta) < FLOOR_DECODE_MS_VERIFY else ""
            w(
                f"* {label} ms/Verify: balanced DCP {x:.2f} vs 7,3,3 {y:.2f} = "
                f"{delta:+.1f} %{mark}"
            )
    w("")
    w(
        "Both arms carry the identical weight split, so this difference is the KV "
        "token placement alone -- the DCP hop, isolated."
    )
    w("")

    if missing:
        w("### NOT MEASURED")
        w("")
        for a in missing:
            w(f"* {ARM_LABEL.get(a, a)}")
        nm = os.path.join(step, "not_measured.txt")
        if os.path.exists(nm):
            with open(nm, errors="replace") as f:
                for line in f:
                    w(f"  * {line.strip()}")
        w("")

    text = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
