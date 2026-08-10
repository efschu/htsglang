#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#656 acceptance evidence extract (user spec item 2), successor 25.

Answers, from ONE unmanned log plus the run's corridor csv, every axis the
spec asks to see AT THE SAME TIME:

  both layouts visited          -- flip DONE counts, per direction
  prefill only in PP            -- purity: prefill batches carrying a CUDA
                                   graph must be ZERO under strict purity
  decode only in TP             -- decode batches exist and carry graphs
  graphs active                 -- decode graph share
  MTP speculation               -- accept length
  per-phase KV pool             -- staging reserved, live slots
  corridor                      -- minimum AND typical free per card
                                   (the law has two halves)
  host RAM                      -- cgroup peak and oom_kill
  real agent traffic            -- /v1/... request lines, counted from the
                                   log rather than asserted

EVERY COUNT IS TAKEN AFTER THE LAST 'PHASE-FLIP armed at boot'.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

FLOOR = 1024
DONE = re.compile(
    r"PHASE-FLIP DONE (\w+) \(epoch \d+\).*?(\d+) live slots,"
    r".*?staging reserved ([\d.]+) MiB"
)
ARMED = re.compile(r"PHASE-FLIP armed at boot")
ABANDON = re.compile(r"FLIP ABANDONED")
PREFILL = re.compile(r"Prefill batch,.*cuda graph: (True|False)")
DECODE = re.compile(r"Decode batch,.*cuda graph: (True|False)")
ACCEPT = re.compile(r"accept[_ ]len[a-z]*[:= ]+([\d.]+)", re.I)
ENDPOINT = re.compile(r"(/v1/[a-zA-Z0-9_/]+)")
TRACE = re.compile(r"Traceback")


def since_last_boot(path, max_bytes=600 * 1024 * 1024):
    size = os.path.getsize(path)
    with open(path, "r", errors="replace") as fh:
        fh.seek(max(0, size - max_bytes))
        lines = fh.readlines()
    last = 0
    for i, ln in enumerate(lines):
        if ARMED.search(ln):
            last = i
    return lines[last:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--log", default="/spinning/serving-30030.boot.log")
    a = ap.parse_args()

    pool_f = os.path.join(a.outdir, "pool")
    pool = int(open(pool_f).read().strip()) if os.path.exists(pool_f) else 0
    print(f"===== #656 ACCEPTANCE EVIDENCE  pool={pool}  {a.outdir}")

    # -- corridor: both halves of the law ----------------------------------
    rows = list(csv.DictReader(open(os.path.join(a.outdir, "corridor.csv"))))
    print(f"\n-- corridor ({len(rows)} samples, floor {FLOOR} MiB/card)")
    breaches = 0
    for c in [c for c in (rows[0] if rows else {}) if c.endswith("_free")]:
        v = sorted(int(r[c]) for r in rows if r.get(c))
        if not v:
            continue
        b = sum(1 for x in v if x < FLOOR)
        breaches += b
        print(
            f"   {c}: MIN={v[0]}  p1={v[len(v)//100]}  TYPICAL(p50)={v[len(v)//2]}"
            f"  max={v[-1]}  breaches={b}  margin={v[0]-FLOOR}"
        )

    lines = since_last_boot(a.log)
    dirs, staging, live = {}, [], []
    abandons = traces = 0
    pg = {"True": 0, "False": 0}
    dg = {"True": 0, "False": 0}
    accepts = []
    endpoints = {}
    for ln in lines:
        m = DONE.search(ln)
        if m:
            dirs[m.group(1)] = dirs.get(m.group(1), 0) + 1
            live.append(int(m.group(2)))
            staging.append(float(m.group(3)))
        if ABANDON.search(ln):
            abandons += 1
        if TRACE.search(ln):
            traces += 1
        m = PREFILL.search(ln)
        if m:
            pg[m.group(1)] += 1
        m = DECODE.search(ln)
        if m:
            dg[m.group(1)] += 1
        m = ACCEPT.search(ln)
        if m:
            accepts.append(float(m.group(1)))
        for e in ENDPOINT.findall(ln):
            endpoints[e] = endpoints.get(e, 0) + 1

    print(f"\n-- flips (lines since last boot: {len(lines)})")
    for d, n in sorted(dirs.items()):
        print(f"   {d}: {n}")
    print(f"   BOTH LAYOUTS VISITED: {len(dirs) >= 2}")
    print(f"   FLIP ABANDONED: {abandons}    tracebacks: {traces}")
    if staging:
        print(
            f"   staging reserved MiB: min={min(staging):.1f} "
            f"max={max(staging):.1f} mean={sum(staging)/len(staging):.1f}"
        )
    if live:
        print(f"   live slots: max={max(live)}", end="")
        if pool:
            print(f"  = {max(live)/pool:.1%} of pool")
        else:
            print()

    print("\n-- phase purity and CUDA graphs")
    print(f"   prefill batches: {pg['True']+pg['False']}  WITH a graph: {pg['True']}")
    print(f"   decode  batches: {dg['True']+dg['False']}  WITH a graph: {dg['True']}")
    pure = pg["True"] == 0
    print(f"   STRICT PURITY (no prefill graph): {pure}")
    if dg["True"] + dg["False"]:
        print(f"   decode graph share: {dg['True']/(dg['True']+dg['False']):.1%}")

    print("\n-- speculation (MTP)")
    if accepts:
        print(f"   accept length: mean={sum(accepts)/len(accepts):.3f} "
              f"n={len(accepts)}")
    else:
        print("   accept length: NOT FOUND in the log -- report as unmeasured, "
              "do NOT report as absent")

    print("\n-- agent / client traffic (counted, not asserted)")
    for e, n in sorted(endpoints.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   {e}: {n}")
    if not endpoints:
        print("   none seen")

    print("\n-- host RAM")
    for f in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory.events"):
        if os.path.exists(f):
            val = open(f).read().strip().replace("\n", " ")
            if f.endswith("peak"):
                val = f"{int(val)/1024**3:.1f} GiB"
            print(f"   {os.path.basename(f)}: {val}")

    ok = breaches == 0 and abandons == 0 and traces == 0 and pure and len(dirs) >= 2
    print(f"\n===== ACCEPTANCE: {'GREEN' if ok else 'NOT GREEN'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
