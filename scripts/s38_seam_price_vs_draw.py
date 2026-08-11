#!/usr/bin/env python3
"""Per-cutover: what the gate PRICED against what the card actually gave up.

Successor 38 / #656, register C20 residual 2. HANDOFF_681 §2b says a cutover
entering HIGH still draws ~1040 MiB on the binding card and that
``_staging_bytes`` does not price it. That is a claim about a DIFFERENCE, so
it has to be measured as one: join each seam's own reserved figure (the
``staging reserved`` field of the flip DONE line, which is exactly what the
corridor gate asked for) to the NVML drawdown across the same cutover.

Usage:  s38_seam_price_vs_draw.py <corridor.csv> <serving.log> <rank> <column>

RANK AND COLUMN ARE SEPARATE ARGUMENTS AND THAT IS THE POINT. The log prefix
``PP<rank>`` is a rank id; the corridor sampler's ``gpu<n>_free`` is an
nvidia-smi index; and on this rig they are NOT the same permutation. The boot
runs ``--rank-gpu-id 0,1,2`` in CUDA order, which is FASTEST_FIRST here, so
rank 0 is the 5090 (nvidia-smi index 1) and ranks 1/2 are the two 3080s
(nvidia-smi 0 and 2). Confirmed against the guard's own free readings: PP1
clears at free 1886/2726 MiB match the gpu0 column (p50 2349), PP2's
2968/3236 match gpu2 (p50 2715), and PP0 -- which armed only 10 times in 65
minutes -- is the card with headroom to spare.
"""
import csv
import datetime
import re
import subprocess
import sys

csv_path, log_path = sys.argv[1], sys.argv[2]
rank = int(sys.argv[3])
column = int(sys.argv[4]) if len(sys.argv) > 4 else rank

rows = []
with open(csv_path) as f:
    for d in csv.DictReader(f):
        try:
            rows.append(
                (int(d["ts_ms"]) / 1000.0, int(d[f"gpu{column}_free"]))
            )
        except Exception:
            pass
rows.sort()
day = datetime.datetime.fromtimestamp(rows[0][0])

# The DONE line is emitted AFTER the seam completes and carries the direction
# and the reserved figure. Its timestamp is therefore the seam's END, while
# the cutover line marks its middle; both are needed to place the window.
done = subprocess.run(
    [
        "grep",
        "-oE",
        rf"^\[[0-9-]+ [0-9:]+ PP{rank}\].*DONE (pp_to_tp|tp_to_pp) .*staging reserved [0-9.]+ MiB",
        log_path,
    ],
    capture_output=True,
    text=True,
).stdout


def ep(h):
    return (
        datetime.datetime.strptime(h, "%H:%M:%S")
        .replace(year=day.year, month=day.month, day=day.day)
        .timestamp()
    )


events = []
for line in done.splitlines():
    m = re.match(r"^\[[0-9-]+ ([0-9:]+) PP\d+\].*DONE (\w+) ", line)
    s = re.search(r"staging reserved ([0-9.]+) MiB", line)
    if not (m and s):
        continue
    events.append((ep(m.group(1)), m.group(2), float(s.group(1))))
events.sort()

out = []
for t_end, direction, priced in events:
    # ENTRY: the last second before the seam's window opens. The DONE stamp is
    # a whole second and the seam itself takes O(100 ms), so the entry window
    # sits 2.5..1.0 s before it and the in-seam window spans the stamp.
    # THE SAME WINDOWS successor 37's ``deep_seam_events.py`` used, so the
    # draw column here is comparable to the 1040 MiB figure HANDOFF_681 §2b
    # quotes. The cutover line and the DONE line land in the same second on
    # every seam of that window, so anchoring on DONE costs no alignment.
    pre = [r for r in rows if t_end - 1.5 <= r[0] <= t_end - 0.2]
    ins = [r for r in rows if t_end - 0.2 < r[0] <= t_end + 1.5]
    if not (pre and ins):
        continue
    entry = min(p[1] for p in pre)
    low = min(i[1] for i in ins)
    out.append((t_end, direction, priced, entry, low, entry - low))

if not out:
    print("no joined events -- check the rank prefix or the clock alignment")
    sys.exit(1)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


print(f"rank {rank} on nvidia-smi gpu{column}: joined {len(out)} cutovers of {len(events)} DONE lines\n")
for direction in ("pp_to_tp", "tp_to_pp"):
    sel = [o for o in out if o[1] == direction]
    if not sel:
        continue
    draws = [o[5] for o in sel]
    priced = [o[2] for o in sel]
    over = [o[5] - o[2] for o in sel]
    print(
        f"{direction}: n={len(sel)}\n"
        f"   priced  MiB p50={q(priced,.5):.0f} p90={q(priced,.9):.0f} max={max(priced):.0f}\n"
        f"   drawn   MiB p50={q(draws,.5):.0f} p90={q(draws,.9):.0f} max={max(draws):.0f}\n"
        f"   UNPRICED (drawn-priced) p50={q(over,.5):.0f} p90={q(over,.9):.0f} max={max(over):.0f}"
    )

print("\n15 deepest in-seam minima, with the price that was asked for them:")
print(
    f"{'end':>10}{'dir':>10}{'entry':>7}{'min':>6}{'draw':>6}{'priced':>8}{'unpriced':>9}"
)
for t, d, p, e, m, dr in sorted(out, key=lambda o: o[4])[:15]:
    hh = datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")
    print(f"{hh:>10}{d:>10}{e:>7}{m:>6}{dr:>6}{p:>8.0f}{dr - p:>9.0f}")

# THE QUESTION THE RESIDUAL ACTUALLY ASKS: on the cutovers that entered HIGH,
# is the draw bigger than the price? A high entry is where the seam is free to
# spend, so it is where an unpriced term shows itself.
hi = [o for o in out if o[3] >= 2000]
if hi:
    over = [o[5] - o[2] for o in hi]
    print(
        f"\nentries >= 2000 MiB: n={len(hi)}  unpriced p50={q(over,.5):.0f} "
        f"p90={q(over,.9):.0f} max={max(over):.0f} MiB"
    )
