#!/usr/bin/env python3
"""s13 -- the tables for #293 step 2, built from what the run persisted.

Reads three kinds of artifact out of one step directory and prints markdown:

  punkte.jsonl        one line per measured point, arm name carrying the round
                      as a suffix ("bar1pipe_r2")
  wait/<arm>.json     the compute/wait split of that boot's primary point,
                      produced on the host by s12_log_analyse
  belege/<arm>.txt    the ERREICHT lines and the prefill-graph lines of that
                      boot -- evidence, not numbers

THE NOISE FLOOR IS COMPUTED, NOT ASSUMED. Every arm ran in every round, so the
spread of one arm across rounds is an A-vs-A measurement of the same
configuration. The largest such spread over all arms is the floor, and a
difference between two arms that does not clear it is printed with a marker
rather than as a result. Nothing here decides what the levers are worth; it
prints the numbers with their uncertainty attached so the verdict can be
argued with.

Stdlib only: it runs in the container against the run directory, but nothing
stops it running on the host.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

REFERENCE = "nccl"


def arm_and_round(name: str) -> tuple:
    m = re.match(r"^(.*)_r(\d+)$", name or "")
    if not m:
        return (name, 0)
    return (m.group(1), int(m.group(2)))


def load_points(step_dir: str) -> list:
    path = os.path.join(step_dir, "punkte.jsonl")
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


def load_wait(step_dir: str) -> dict:
    """{(arm, round): [per-rank aggregate]} out of wait/*.json."""
    out: dict = {}
    d = os.path.join(step_dir, "wait")
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, name), errors="replace") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for w in payload.get("wait") or []:
            arm, rnd = arm_and_round(w.get("arm"))
            out.setdefault((arm, rnd), []).append(w)
    return out


def load_evidence(step_dir: str) -> dict:
    """{(arm, round): {'bar1_gruppen': n, 'prefill_graph': str}}.

    The keys stay German because they are the artifact's field names, and the
    substrings counted below are log lines the server writes, matched verbatim.
    """
    out: dict = {}
    d = os.path.join(step_dir, "belege")
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".txt"):
            continue
        arm, rnd = arm_and_round(name[:-4])
        try:
            with open(os.path.join(d, name), errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        pg = "off"
        if "prefill CUDA graph end" in text or "prefill CUDA graph begin" in text:
            pg = "ON"
        elif "isabling prefill CUDA graph" in text or "Disable prefill CUDA" in text:
            pg = "off (auto-disable)"
        out[(arm, rnd)] = {
            "bar1_gruppen": text.count("HTCCL-BAR1: Aufbau in"),
            "erreicht": len(re.findall(r"ACHIEVED=", text)),
            "pipe_zeilen": text.count("HTCCL-BAR1-PIPE:"),
            "vorrat_leer": "Graph-Vorrat des Ergebnisrings ist erschoepft" in text,
            "prefill_graph": pg,
        }
    return out


def collect(points: list) -> dict:
    """{(arm, sessions): {round: {...}}}"""
    out: dict = {}
    for p in points:
        arm, rnd = arm_and_round(p.get("arm"))
        sess = p.get("sessions")
        rate = (p.get("prefill") or {}).get("prefill_tok_s")
        entry = {"prefill_tok_s": rate}
        for d in p.get("decode") or []:
            bs = d.get("batch")
            entry[f"tick_tok_s_bs{bs}"] = d.get("tick_gen_tok_s_median")
            entry[f"accept_bs{bs}"] = d.get("tick_accept_len_median")
            entry[f"ms_verify_bs{bs}"] = d.get("tick_ms_pro_verify")
        out.setdefault((arm, sess), {})[rnd] = entry
    return out


def _mean(values: list):
    values = [w for w in values if isinstance(w, (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def _spread_pct(values: list):
    """Relative spread of repeated measurements of the SAME configuration."""
    values = [w for w in values if isinstance(w, (int, float))]
    if len(values) < 2:
        return None
    m = _mean(values)
    if not m:
        return None
    return (max(values) - min(values)) / m * 100.0


def _f(v, nk=1):
    return "-" if not isinstance(v, (int, float)) else format(v, f".{nk}f")


def report(step_dir: str) -> str:
    points = load_points(step_dir)
    data = collect(points)
    wait = load_wait(step_dir)
    evidence = load_evidence(step_dir)

    arms = []
    for arm, _ in sorted(data):
        if arm not in arms:
            arms.append(arm)
    sessions = sorted({s for _, s in data if isinstance(s, int)})
    rounds = sorted({r for d in data.values() for r in d})

    lines = []

    # --- noise floor first, because it decides what may be reported ---------
    # PER SESSION COUNT, not one number for the whole table. The two points
    # are different measurements -- one session is a latency measurement with
    # a single stream feeding it, eight is a saturated pipeline -- and they do
    # not have the same repeatability. Folding them into one maximum would
    # hold the tight point (0,2-0,8 % at eight sessions) to the loose point's
    # floor and throw away real differences.
    spreads = []
    floor_per_sess: dict = {}
    for (arm, sess), per_round in sorted(data.items()):
        s = _spread_pct([e.get("prefill_tok_s") for e in per_round.values()])
        if s is not None:
            spreads.append((s, arm, sess))
            floor_per_sess[sess] = max(floor_per_sess.get(sess, 0.0), s)
    floor = max((s for s, _, _ in spreads), default=None)
    median_spread = statistics.median([s for s, _, _ in spreads]) if spreads else None

    lines.append("### Noise floor (A vs. A, the same arm across the rounds)")
    lines.append("")
    lines.append(
        "| Arm | Sess. | " + " | ".join(f"R{r}" for r in rounds) + " | Spread % |"
    )
    lines.append("|---|---:|" + "---:|" * (len(rounds) + 1))
    for (arm, sess), per_round in sorted(data.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        values = [per_round.get(r, {}).get("prefill_tok_s") for r in rounds]
        lines.append(
            f"| {arm} | {sess} | "
            + " | ".join(_f(w) for w in values)
            + f" | {_f(_spread_pct(values), 2)} |"
        )
    lines.append("")
    lines.append(
        f"Largest A-vs-A spread overall: **{_f(floor, 2)} %**, "
        f"median of the spreads {_f(median_spread, 2)} %. The yardstick, "
        "though, is the floor OF THE POINT IN QUESTION: "
        + ", ".join(
            f"{s} session(s) {_f(b, 2)} %" for s, b in sorted(floor_per_sess.items())
        )
        + ". A ratio that differs from 1.000 by less than that floor is "
        "marked with `~` below and is not a statement."
    )
    lines.append("")

    # --- the main table ----------------------------------------------------
    lines.append("### Arm x sessions: prefill throughput and ratio to NCCL")
    lines.append("")
    head = "| Arm |"
    sep = "|---|"
    for sess in sessions:
        head += f" tok/s (s={sess}) | vs. nccl |"
        sep += "---:|---:|"
    head += " prefill graph | BAR1 groups |"
    sep += "---|---:|"
    lines.append(head)
    lines.append(sep)

    for arm in arms:
        line = f"| {arm} |"
        for sess in sessions:
            per_round = data.get((arm, sess), {})
            m = _mean([e.get("prefill_tok_s") for e in per_round.values()])
            ref = _mean(
                [
                    e.get("prefill_tok_s")
                    for e in data.get((REFERENCE, sess), {}).values()
                ]
            )
            if m is None or not ref:
                line += f" {_f(m)} | - |"
                continue
            v = m / ref
            mark = ""
            limit = floor_per_sess.get(sess)
            if limit is not None and abs(v - 1.0) * 100.0 < limit:
                mark = "~"
            line += f" {_f(m)} | {mark}{v:.3f} |"
        ev = evidence.get((arm, rounds[0] if rounds else 1), {})
        line += f" {ev.get('prefill_graph', '-')} | {ev.get('bar1_gruppen', '-')} |"
        lines.append(line)
    lines.append("")

    # --- compute / wait per rank ------------------------------------------
    lines.append("### compute / wait per rank at the primary point (sessions=8)")
    lines.append("")
    lines.append(
        "| Arm | Round | TP0 comp | TP0 wait | TP1 comp | TP1 wait | "
        "TP2 comp | TP2 wait | gpu-ms TP1 | wait share TP1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        for rnd in rounds:
            rows = wait.get((arm, rnd))
            if not rows:
                continue
            per_rank = {r.get("rang"): r for r in rows}
            cells = []
            for rank in (0, 1, 2):
                r = per_rank.get(rank) or {}
                cells.append(_f(r.get("compute_ms_median")))
                cells.append(_f(r.get("wait_ms_median")))
            tp1 = per_rank.get(1) or {}
            share = tp1.get("wait_anteil")
            lines.append(
                f"| {arm} | {rnd} | "
                + " | ".join(cells)
                + f" | {_f(tp1.get('gpu_ms_median'))} | "
                + (f"{share * 100:.1f} %" if isinstance(share, float) else "-")
                + " |"
            )
    lines.append("")

    # --- decode -----------------------------------------------------------
    lines.append("### Decode ticks from the same boot (sessions=8)")
    lines.append("")
    lines.append(
        "| Arm | bs=1 tok/s | bs=1 accept | bs=16 tok/s | bs=16 accept | "
        "bs=16 ms/Verify |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm in arms:
        per_round = data.get((arm, 8), {})
        if not per_round:
            continue
        def mm(key):
            return _mean([e.get(key) for e in per_round.values()])
        lines.append(
            f"| {arm} | {_f(mm('tick_tok_s_bs1'))} | {_f(mm('accept_bs1'), 2)} | "
            f"{_f(mm('tick_tok_s_bs16'))} | {_f(mm('accept_bs16'), 2)} | "
            f"{_f(mm('ms_verify_bs16'), 2)} |"
        )
    lines.append("")

    # --- evidence ---------------------------------------------------------
    lines.append("### Evidence per boot")
    lines.append("")
    # `ERREICHT` and `Aufbau` name the server log lines that were counted --
    # the marker text itself, not a translated description of it.
    lines.append(
        "| Arm | Round | ERREICHT lines | BAR1 Aufbau | PIPE lines | "
        "graph pool empty | prefill graph |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for arm in arms:
        for rnd in rounds:
            b = evidence.get((arm, rnd))
            if not b:
                continue
            lines.append(
                f"| {arm} | {rnd} | {b['erreicht']} | {b['bar1_gruppen']} | "
                f"{b['pipe_zeilen']} | {'yes' if b['vorrat_leer'] else 'no'} | "
                f"{b['prefill_graph']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    print(report(args.step_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
