#!/usr/bin/env python3
"""DUAL-TP3PP3: read one dual boot's logs and answer the questions that decide
whether the double layout RUNS.

  eval_dual_boot.py <evidence-dir-or-prefix>   (the boot_weg2_<tag>_... files)

Answers, each with the lines it counted:
1. Did the dual path engage? (launcher WEG2-DUAL*, front WEG2 DUAL-LAYOUT on,
   P stage assembled, union OWNER/PEER + bound bytes, D ratios published)
2. Were there flips? (WEG2-FLIP begin) -- must be 0.
3. Did P prefill WHILE D decoded? Overlap of PP0 prefill seconds with TP0
   decode seconds (1-s buckets), and D's per-round gpu-ms in buckets with and
   without P activity (the price D pays; risk-1 predicted ~2x at P 50 %).
4. Errors: Traceback / REFUSED / W-codes / CUDA errors, first lines only.
"""
from __future__ import annotations

import collections
import glob
import os
import re
import statistics
import sys

TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _files(arg):
    if os.path.isdir(arg):
        c = sorted(glob.glob(os.path.join(arg, "boot_weg2_*.front.log")), key=os.path.getmtime)
        if not c:
            sys.exit(f"no boot_weg2_*.front.log in {arg}")
        arg = c[-1][: -len(".front.log")]
    return {k: f"{arg}.{k}.log" for k in ("front", "P", "D")}


def _sec(line):
    m = TS.match(line)
    if not m:
        return None
    import datetime

    return int(datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp())


def _grep(path, pats, limit=6):
    out = collections.defaultdict(list)
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            for name, rx in pats.items():
                if rx.search(line):
                    if len(out[name]) < limit:
                        out[name].append(line.rstrip()[:260])
                    out[name + "#"].append(1)
    return out


def main():
    fs = _files(sys.argv[1])
    print("files:", fs)
    pats = {
        "dual_launch": re.compile(r"WEG2-DUAL"),
        "front_dual": re.compile(r"WEG2 DUAL-LAYOUT on"),
        "p_stage": re.compile(r"DUAL-TP3PP3 P stage \d+/\d+ assembled"),
        "p_bound": re.compile(r"DUAL-TP3PP3 P: shared part bound"),
        "union_owner": re.compile(r"WEG2-UNION OWNER"),
        "union_peer": re.compile(r"WEG2-UNION PEER"),
        "d_ratios": re.compile(r"WEG2-UNION D ratios published"),
        "flip": re.compile(r"WEG2-FLIP begin"),
        "error": re.compile(r"Traceback|REFUSED:|CUDA error|out of memory|OutOfMemory|Weg2Stop|DualShareError|UnionShareError|STOP "),
        "mps": re.compile(r"MPS"),
        "duty": re.compile(r"duty throttle armed"),
    }
    for k, p in fs.items():
        g = _grep(p, pats)
        print(f"\n== {k}: " + ", ".join(f"{n}={len(g.get(n + '#', []))}" for n in pats))
        for n in ("dual_launch", "front_dual", "p_stage", "p_bound", "union_owner", "union_peer", "d_ratios",
                  "flip", "error", "duty"):
            for line in g.get(n, [])[:3]:
                print(f"   {n}: {line}")
    # overlap: PP0 prefill seconds vs TP0 decode seconds
    pre = collections.Counter()
    if os.path.exists(fs["P"]):
        with open(fs["P"], errors="replace") as f:
            for line in f:
                if "PP0] Prefill batch" in line:
                    s = _sec(line)
                    m = re.search(r"#new-token: (\d+)", line)
                    if s is not None and m:
                        pre[s] += int(m.group(1))
    dec = collections.defaultdict(list)
    if os.path.exists(fs["D"]):
        with open(fs["D"], errors="replace") as f:
            for line in f:
                if "TP0] Decode rank batch" in line:
                    s = _sec(line)
                    m = re.search(r"gpu-ms: ([\d.]+)", line)
                    if s is not None and m:
                        dec[s].append(float(m.group(1)))
    both = [s for s in dec if s in pre]
    alone = [s for s in dec if s not in pre and (s - 1) not in pre and (s + 1) not in pre]
    print(f"\n== overlap: {len(pre)} s with P prefill ({sum(pre.values())} tokens), {len(dec)} s with D decode, "
          f"{len(both)} s with BOTH")
    for name, secs in (("D round gpu-ms while P prefills", both), ("D round gpu-ms, P idle (+-1 s)", alone)):
        vals = [v for s in secs for v in dec[s]]
        if vals:
            vals.sort()
            print(f"   {name}: n={len(vals)} p50={statistics.median(vals):.1f} "
                  f"p90={vals[int(0.9 * (len(vals) - 1))]:.1f}")
        else:
            print(f"   {name}: n=0")


if __name__ == "__main__":
    main()
