#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 36: judge the rebalance lender's A/B on the axes that
# actually separate the arms.
#
# THREE ARMS, and the third is the control:
#   s34      the green acceptance, lender did not exist       (65 min)
#   s36      the confirmation window, lender ON               (46 min)
#   s36-ab   same commit and same load script, lender OFF     (26 min)
#
# JUDGE AT MATCHED ELAPSED TIME, NOT ON TOTALS. s34's own counters accelerate
# over its window (its soak completes 6 requests by t+2 and 218 by t+64), so a
# total compared against a shorter run says nothing. Every figure below is
# taken over each arm's FIRST 25 MINUTES.
#
# THE DECIDING AXIS IS THE SOAK, because it is the only generator whose
# parameters are identical in all three arms. Agent traffic is real work and
# varies in SIZE even when it matches in rate -- a window whose agents send
# larger prompts holds the instance in PP longer, which is exactly the shape
# under investigation, so it cannot also be the instrument.
#
# Usage: bash scripts/s36_ab_judge.sh <ab-outdir> <ab-serving-log> <ab-start-HH:MM>
set -uo pipefail

AB_DIR="${1:?ab outdir}"
AB_LOG="${2:?ab serving log}"
AB_START="${3:?ab start HH:MM}"

/spinning/htsglang-gpu/.venv/bin/python - "$AB_DIR" "$AB_LOG" "$AB_START" <<'PYEOF'
import datetime, statistics, subprocess, sys

ab_dir, ab_log, ab_start = sys.argv[1], sys.argv[2], sys.argv[3]

ARMS = [
    ("s34   gate only ", "/spinning/evidence-631/s34/serving-run2.log",
     "/spinning/evidence-631/s34/accept2/soak.log", "13:24"),
    ("s36   lender ON ", "/spinning/evidence-631/s36/serving.log",
     "/spinning/evidence-631/s36/confirm/soak.log", "15:47"),
    ("s36ab lender OFF", ab_log, f"{ab_dir}/soak.log", ab_start),
]
SPAN_MIN = 25


def window(start):
    t0 = datetime.datetime.strptime(start, "%H:%M")
    return t0, t0 + datetime.timedelta(minutes=SPAN_MIN)


def count(log, pattern, start):
    t0, t1 = window(start)
    out = subprocess.run(["grep", "-c", pattern, log], capture_output=True, text=True).stdout
    # A cheap total is useless here; re-scan with timestamps instead.
    out = subprocess.run(["grep", "-oE", rf"^\[[0-9-]+ [0-9:]+.*{pattern}", log],
                         capture_output=True, text=True).stdout
    n = 0
    for line in out.splitlines():
        try:
            t = datetime.datetime.strptime(line[12:17], "%H:%M")
        except ValueError:
            continue
        if t0 <= t < t1:
            n += 1
    return n


def dwell(log, start):
    t0, t1 = window(start)
    out = subprocess.run(
        ["grep", "-oE", r"^\[[0-9-]+ [0-9:]+ PP0\].*cutover (pp_to_tp|tp_to_pp)", log],
        capture_output=True, text=True).stdout
    ev = []
    for line in out.splitlines():
        try:
            t = datetime.datetime.strptime(line[12:20], "%H:%M:%S")
        except ValueError:
            continue
        if not (t0 <= t < t1):
            continue
        ev.append((t, "pp_to_tp" if "pp_to_tp" in line else "tp_to_pp"))
    ev.sort()
    tp, pp = [], []
    for (a, da), (b, db) in zip(ev, ev[1:]):
        gap = (b - a).total_seconds()
        if not (0 < gap <= 120):
            continue
        (tp if (da == "pp_to_tp" and db == "tp_to_pp") else
         pp if (da == "tp_to_pp" and db == "pp_to_tp") else []).append(gap)
    med = lambda v: statistics.median(v) if v else 0.0
    return med(tp), med(pp), len(tp)


def soak_at(path, start, minutes):
    """The soak's ok count at t+minutes, read from its own periodic line."""
    t0, _ = window(start)
    want = t0 + datetime.timedelta(minutes=minutes)
    best = None
    try:
        for line in open(path):
            if "ok=" not in line:
                continue
            try:
                t = datetime.datetime.strptime(line.split("Z")[0].strip(), "%H:%M:%S")
            except ValueError:
                continue
            if t <= want:
                best = line.strip()
    except FileNotFoundError:
        return "no soak.log"
    return best or "no sample"


print(f"ALL FIGURES OVER EACH ARM'S FIRST {SPAN_MIN} MINUTES\n")
for label, log, soak, start in ARMS:
    pre = count(log, "Prefill batch", start)
    dec = count(log, "Decode batch", start)
    cut = count(log, "cutover pp_to_tp", start)
    tpd, ppd, ntp = dwell(log, start)
    lends = count(log, "CORRIDOR-REBALANCE device", start)
    print(f"{label}  (start {start})")
    print(f"   prefill batches {pre:6d}   decode batches {dec:5d}   pp->tp cutovers {cut:4d}")
    print(f"   median dwell    TP {tpd:5.1f}s   PP {ppd:5.1f}s   (n={ntp})")
    print(f"   lend summaries  {lends}")
    print(f"   soak t+14  {soak_at(soak, start, 14)}")
    print(f"   soak t+24  {soak_at(soak, start, 24)}")
    print()
print("READ IT THIS WAY: if s36ab (lender OFF) looks like s36 (lender ON),")
print("the decode starvation is NOT the lender and predates this shift. If")
print("s36ab looks like s34, the lender is paying for its corridor gain in")
print("decode throughput and must not ship default-on.")
PYEOF
