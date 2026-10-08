#!/usr/bin/env python3
"""DUAL-TP3PP3 stage 2: D round gap at admission (metal criterion: <= 2x median).

  join_gap_eval.py <boot prefix>      (reads .D.log)
For every admission on D -- 'PDFLIP DECODE-JOIN' (join path) or a TP0 'Prefill rank batch'
(extend path) -- the gap between the two TP0 'Decode rank batch' rounds around it, placed
by the rounds' own t: field (log stamps are flushed in bursts). Prints median round gap,
per-admission gaps and the verdict gap <= 2 x median.
"""
import bisect
import re
import statistics as st
import sys
import datetime as dt

pre = sys.argv[1]
DR = re.compile(r"TP0\] Decode rank batch, rank: 0, #round: \d+, t: ([\d.]+),")
ADM = re.compile(r"^\[(\S+ \S+) TP0\] (PDFLIP DECODE-JOIN n=\d+|Prefill rank batch, #new-token: (\d+))")
t, adm = [], []
for line in open(pre + ".D.log", errors="replace"):
    m = DR.search(line)
    if m:
        t.append(float(m.group(1)))
        continue
    m = ADM.match(line)
    if m:
        adm.append((m.group(2).split(",")[0], len(t)))  # index of the next round after this admission line
t.sort()
gaps = [b - a for a, b in zip(t, t[1:]) if b - a < 5.0]
if not gaps:
    sys.exit("no decode rounds")
med = st.median(gaps)
print(f"rounds={len(t)} median gap {med * 1000:.1f} ms, p99 {sorted(gaps)[int(0.99 * (len(gaps) - 1))] * 1000:.1f} ms")
bad = idle = 0
for kind, i in adm:
    if 0 < i < len(t):
        g = t[i] - t[i - 1]
        if g >= 5.0:
            print(f"  {kind:42s} gap {g * 1000:7.1f} ms IDLE (no decode running, not counted)")
            idle += 1
            continue
        flag = g <= 2 * med
        bad += not flag
        print(f"  {kind:42s} gap {g * 1000:7.1f} ms {'OK' if flag else '> 2x median'}")
print(f"VERDICT {'OK' if bad == 0 else 'VIOLATED'}: {len(adm) - bad - idle}/{len(adm) - idle} admissions within 2x median "
      f"(note: log-stamp order places an admission line only approximately between rounds)")
