#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 36: the judged extract for the ITEM 16 CONFIRMATION window.
#
# s34_extract.sh (every acceptance axis, unchanged, so the green state is
# re-judged rather than assumed) plus the one question this window exists to
# answer: did the rebalance lender move the numbers item 16 is written on?
#
# THE COMPARISON IS EXPLICIT AND HARD-CODED. s34's figures are quoted here as
# constants, from /spinning/evidence-631/s34/accept2/EXTRACT.txt, because a
# confirmation window that reports its own numbers without the baseline beside
# them is not a confirmation -- it is a second unjudged run.
#
# Usage: bash scripts/s36_extract.sh <outdir> <serving-log>
set -uo pipefail

OUT="${1:?outdir}"
LOG="${2:?serving log}"
WT=/spinning/wt-631-routea
PY=/spinning/htsglang-gpu/.venv/bin/python
export PYTHONPATH="$WT/python"

EX="$OUT/EXTRACT.txt"
CSV="$OUT/corridor.csv"
SCAN="$OUT/.scan36.txt"
tr -d '\000' < "$LOG" > "$SCAN" 2>/dev/null

c() { grep -c "$1" "$SCAN" 2>/dev/null | head -1; }

{
  bash "$WT/scripts/s34_extract.sh" "$OUT" "$LOG" > /dev/null 2>&1
  cat "$EX" 2>/dev/null

  echo
  echo "===== SUCCESSOR 36 ADDITIONS -- SPEC ITEM 16 ACTUATED"
  echo
  echo "-- the lender's own liveness (armed != spending)"
  if grep -q "CORRIDOR-REBALANCE ARMED" "$SCAN" 2>/dev/null; then
    grep -m 3 "CORRIDOR-REBALANCE ARMED" "$SCAN" | cut -c1-200 | sed 's/^/   /'
  else
    echo "   NO ARM LINE -- the lender was not built on any rank. Every"
    echo "   item-16 number below is then a NO-CHANGE run, not a confirmation."
  fi
  echo
  echo "   lend summaries emitted: $(c 'CORRIDOR-REBALANCE] device')"
  echo "   last three, verbatim:"
  grep "CORRIDOR-REBALANCE] device" "$SCAN" | tail -3 | cut -c1-320 | sed 's/^/     /'

  echo
  echo "-- ITEM 16 A/B: this window vs s34's green window"
  "$PY" - "$CSV" <<'PYEOF'
import sys

sys.path.insert(0, "/spinning/wt-631-routea/python")
from sglang.srt.managers.corridor_guard import (  # noqa: E402
    free_spread_mib,
    water_fill_transfers,
)

MIB = 1024 * 1024
path = sys.argv[1]
rows = []
with open(path) as fh:
    header = fh.readline().strip().split(",")
    # Two spellings exist on disk (gpu{i}_free and free{i}_mib); s35's report
    # script grew the same tolerance and that is why this cross-check exists.
    cols = [i for i, h in enumerate(header) if h.startswith(("gpu", "free"))]
    cols = [i for i in cols if header[i] not in ("free_spread_mib",)]
    for line in fh:
        parts = line.strip().split(",")
        if len(parts) <= max(cols or [0]):
            continue
        try:
            rows.append([int(parts[i]) for i in cols[:3]])
        except ValueError:
            continue

if not rows:
    print("   NO SAMPLES -- corridor.csv unreadable")
    raise SystemExit(0)

mins = [min(r[i] for r in rows) for i in range(3)]
spreads = sorted(max(r) - min(r) for r in rows)
n = len(spreads)
# The BINDING INSTANT: the sample at which the tightest card was tightest.
tight_card = min(range(3), key=lambda i: mins[i])
binding = min(rows, key=lambda r: r[tight_card])
lev = [t / MIB for t in water_fill_transfers([f * MIB for f in binding])]

S34_MINS = [1043, 1922, 1541]
S34_SPREAD = {"mean": 2409, "median": 2723, "worst": 2949, "binding": 879}

print(f"   samples {n}")
print("   per-card MIN free (the corridor's first half, law 1024 MiB)")
for i in range(3):
    d = mins[i] - S34_MINS[i]
    print(
        f"     gpu{i}  min {mins[i]:5d} MiB  margin +{mins[i]-1024:4d}"
        f"   s34 {S34_MINS[i]:5d} (+{S34_MINS[i]-1024})   delta {d:+5d} MiB"
    )
print(
    f"   breaches below 1024: "
    f"{sum(1 for r in rows for f in r if f < 1024)}   (s34: 0)"
)
mean_s = sum(spreads) / n
print("   free-headroom SPREAD (item 16's levelness axis)")
print(
    f"     mean {mean_s:6.0f} MiB (s34 {S34_SPREAD['mean']})   "
    f"median {spreads[n//2]:5d} (s34 {S34_SPREAD['median']})   "
    f"worst {spreads[-1]:5d} (s34 {S34_SPREAD['worst']})"
)
print(
    f"   BINDING INSTANT column {binding} MiB -- tightest card gpu{tight_card}, "
    f"margin {mins[tight_card]-1024:+d} MiB over the law (s34: +19 on gpu0)"
)
print(
    f"     spread at that instant {max(binding)-min(binding)} MiB "
    f"(s34: {S34_SPREAD['binding']})"
)
print(
    "     water-fill there: "
    + ", ".join(f"gpu{i} {v:+.0f}" for i, v in enumerate(lev))
    + " MiB (positive = should shed)"
)
PYEOF

  echo
  echo "-- axes that must NOT have moved (the green state)"
  echo "   pp_to_tp flips:            $(c 'flip pp_to_tp committed')"
  echo "   tp_to_pp flips:            $(c 'flip tp_to_pp committed')"
  echo "   corridor gate cleared:     $(c 'CORRIDOR-GUARD cleared')"
  echo "   corridor gate REFUSED:     $(c 'CORRIDOR-GUARD REFUSED')"
  echo "   host tier forced:          $(c 'spending HOST RAM')"
  echo "   KV-backing shrinks:        $(c 'KV-BACKING released')"
  echo "   provider draft-weights:    $(c 'draft-weights')"
  echo "   tracebacks:                $(c 'Traceback')"
} > "$OUT/EXTRACT36.txt" 2>&1

echo "wrote $OUT/EXTRACT36.txt"
