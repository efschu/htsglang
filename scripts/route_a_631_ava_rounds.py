#!/usr/bin/env python3
"""#631 A-vs-A gate: the per-rank ms-per-round split, harvested from a boot log.

WHY NOT THE USUAL LINE. The "Prefill rank batch ... gpu-ms (compute, wait)"
split is not installed at pp_size > 1 -- ``_install_rank_prefill_timer``
returns early -- so under a PP=3 boot it carries no numbers, in EITHER tree.
What every PP rank does emit, once ``SGLANG_ENABLE_METRICS_DEVICE_TIMER=1``
is set, is ``fwd occupancy: X%`` on its own ``Prefill batch`` /
``Decode batch`` line: the device timer's GPU-busy time over the wall window.
That is the compute fraction of the rank's round; the remainder is wait plus
host. All three PP ranks log (tp_size 1 => attn_tp_rank 0 on each).

ROUNDS. A ``Prefill batch`` line is one chunked-prefill round; a
``Decode batch`` line is one decode report covering ``decode_log_interval``
rounds. ms/round is therefore taken in AGGREGATE over the measured window --
rounds counted, window measured -- rather than from consecutive log
timestamps, whose resolution is one second and would quantize a ~500 ms
round beyond use.

Read the SPREAD of wait across ranks, not its absolute value: the ranks run
lock-step, so the rank with the most work shows the least wait.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timezone

_LINE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<rank>PP\d+)\] "
    r"(?P<kind>Prefill|Decode) batch,.*?fwd occupancy: (?P<occ>[\d.]+|nan)%"
)
_GEN = re.compile(r"gen throughput \(token/s\): (?P<v>[\d.]+)")


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def parse_window(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def harvest(log: str, t0: datetime, t1: datetime, kind: str) -> dict:
    per_rank: dict[str, list[float]] = {}
    gen: dict[str, list[float]] = {}
    first: dict[str, datetime] = {}
    last: dict[str, datetime] = {}
    with open(log, "rb") as fh:
        for raw in fh:
            line = raw.decode("utf-8", "replace")
            m = _LINE.match(line)
            if m is None or m.group("kind") != kind:
                continue
            ts = parse_ts(m.group("ts"))
            if ts < t0 or ts > t1:
                continue
            rank = m.group("rank")
            occ = m.group("occ")
            if occ != "nan":
                per_rank.setdefault(rank, []).append(float(occ))
            first.setdefault(rank, ts)
            last[rank] = ts
            g = _GEN.search(line)
            if g:
                gen.setdefault(rank, []).append(float(g.group("v")))

    out: dict = {"kind": kind, "ranks": {}}
    for rank in sorted(set(list(first) + list(per_rank))):
        occs = per_rank.get(rank, [])
        rounds = len(occs) if occs else 0
        span = (last[rank] - first[rank]).total_seconds() if rank in first else 0.0
        entry = {
            "rounds_logged": rounds,
            "span_s": span,
            "occupancy_mean_pct": statistics.fmean(occs) if occs else None,
            "occupancy_min_pct": min(occs) if occs else None,
            "occupancy_max_pct": max(occs) if occs else None,
        }
        if gen.get(rank):
            entry["gen_tok_s_mean"] = statistics.fmean(gen[rank])

        # ms PER ROUND, and the two kinds need different arithmetic.
        #
        # Prefill: one ``Prefill batch`` line IS one chunked-prefill round,
        # so the line cadence over the window is the round cadence.
        #
        # Decode: one ``Decode batch`` line covers ``decode_log_interval``
        # rounds, so counting lines would overstate the round by that
        # factor (~40x). The reported gen throughput already carries the
        # per-token rate, and this boot does not speculate -- PP refuses
        # speculation, see the gate boot script -- so one accepted token is
        # exactly one round and 1000/tok_s IS the round.
        ms_round = None
        if kind == "Prefill" and rounds > 1 and span > 0:
            ms_round = 1000.0 * span / (rounds - 1)
        elif kind == "Decode" and entry.get("gen_tok_s_mean"):
            ms_round = 1000.0 / entry["gen_tok_s_mean"]
        if ms_round is not None:
            entry["ms_per_round"] = ms_round
            if entry["occupancy_mean_pct"] is not None:
                entry["compute_ms_per_round"] = (
                    ms_round * entry["occupancy_mean_pct"] / 100.0
                )
                entry["wait_ms_per_round"] = ms_round - entry["compute_ms_per_round"]
        out["ranks"][rank] = entry

    occ_means = [
        v["occupancy_mean_pct"]
        for v in out["ranks"].values()
        if v["occupancy_mean_pct"] is not None
    ]
    if len(occ_means) > 1:
        out["occupancy_spread_pct_points"] = max(occ_means) - min(occ_means)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--gate-json", required=True, help="output of route_a_631_ava_gate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gate = json.load(open(args.gate_json))
    res = {"label": gate["label"], "log": args.log, "rungs": {}}
    for rung, kind in (("prefill", "Prefill"), ("decode", "Decode")):
        w = gate["rungs"][rung]["window"]
        res["rungs"][rung] = {
            "window": w,
            **harvest(args.log, parse_window(w[0]), parse_window(w[1]), kind),
        }
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
