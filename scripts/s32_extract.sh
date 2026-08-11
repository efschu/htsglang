#!/usr/bin/env bash
# #656 successor 32: the judged extract for an acceptance run.
#
# s25_acceptance_evidence.py already covers corridor, flips, purity, graphs,
# MTP, occupancy, traffic and the relief ladder. Two things it cannot report
# for THIS run are added here:
#
#   * ITEM 16's free-headroom SPREAD. The evidence script reads it from
#     corridor.series.csv, which only exists when the sampler was started
#     with --series. s25_acceptance_run.sh does not pass it, so the spread is
#     computed here from corridor.csv's three free columns instead -- same
#     quantity, different source. A run without a spread figure cannot be
#     judged against item 16 at all, and "the sampler was not asked for it"
#     is not a reason to leave the axis blank.
#
#   * WHY THE GATE DID OR DID NOT ARM. CorridorGuard.ensure_headroom returns
#     early and SILENTLY when free - want already clears the floor, so a log
#     with zero "cleared" and zero "REFUSED" lines means the gate was
#     consulted and never needed to arm -- NOT that it was never called.
#     Those two states have opposite meanings for spec item 12 and the
#     extract must not let a reader confuse them.
#
# Usage: bash scripts/s32_extract.sh <outdir> <serving-log>
set -uo pipefail

OUT="${1:?outdir}"
LOG="${2:?serving log}"
PY=/spinning/htsglang-gpu/.venv/bin/python
export PYTHONPATH=/spinning/wt-631-routea/python

EX="$OUT/EXTRACT.txt"
{
  echo "===== #656 JUDGED EXTRACT  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  cat "$OUT/CONFIG.txt" 2>/dev/null
  echo
  $PY scripts/s25_acceptance_evidence.py "$OUT" --log "$LOG" 2>&1

  echo
  echo "-- item 16: free-headroom spread, computed from corridor.csv"
  awk -F, 'NR>1 && NF>=4 {
      lo=$2; hi=$2
      for (i=3; i<=4; i++) { if ($i+0 < lo) lo=$i+0; if ($i+0 > hi) hi=$i+0 }
      s=hi-lo; n++; sum+=s
      if (mx=="" || s>mx) mx=s
      if (mn=="" || s<mn) mn=s
      a[n]=s
    }
    END {
      if (n==0) { print "   no samples"; exit }
      asort(a)
      printf "   spread over %d samples: mean %.0f MiB, median %.0f, min %d, worst %d\n",
             n, sum/n, a[int(n/2)+1], mn, mx
    }' "$OUT/corridor.csv" 2>/dev/null \
    || awk -F, 'NR>1 && NF>=4 {
         lo=$2; hi=$2
         for (i=3; i<=4; i++) { if ($i+0 < lo) lo=$i+0; if ($i+0 > hi) hi=$i+0 }
         s=hi-lo; n++; sum+=s; if (mx==""||s>mx) mx=s; if (mn==""||s<mn) mn=s
       }
       END { if (n) printf "   spread over %d samples: mean %.0f MiB, min %d, worst %d\n", n, sum/n, mn, mx }' \
       "$OUT/corridor.csv"

  echo
  echo "-- per-card headroom ABOVE the 1024 MiB law (the corridor's second half)"
  awk -F, 'NR>1 && NF>=4 {
      for (i=2; i<=4; i++) if (m[i]=="" || $i+0 < m[i]) m[i]=$i+0
    }
    END { printf "   MIN free  %d / %d / %d MiB  ->  headroom +%d / +%d / +%d\n",
                 m[2], m[3], m[4], m[2]-1024, m[3]-1024, m[4]-1024 }' "$OUT/corridor.csv"

  echo
  echo "-- did the corridor gate ARM at all"
  cl=$(tr -d '\000' < "$LOG" | grep -c "CORRIDOR-GUARD cleared on device")
  rf=$(tr -d '\000' < "$LOG" | grep -c "CORRIDOR-GUARD REFUSED on device")
  echo "   armed-and-logged: $cl cleared, $rf refused"
  if [ "$cl" -eq 0 ] && [ "$rf" -eq 0 ]; then
    echo "   READ THIS AS: the gate was CONSULTED on every seam and never"
    echo "   needed to arm -- ensure_headroom returns silently when free-want"
    echo "   already clears the floor. It is NOT evidence that the gate is"
    echo "   unwired, and it IS evidence that spec item 12's ladder went"
    echo "   unexercised in this run: a ladder that is never reached is"
    echo "   proven not to break, not proven to work."
  fi

  echo
  echo "-- KV rung: shrink / recover, and whether recovery really returns"
  tr -d '\000' < "$LOG" \
    | grep -oE "KV-BACKING released [0-9]+ MiB by backing [0-9]+ rows instead of [0-9]+" \
    | tail -20
  echo "   (each 'instead of N' is the backed span BEFORE that shrink; N"
  echo "    returning to the boot row count between shrinks is the proof that"
  echo "    recovery works and that this is not 'a smaller pool as the fix')"
  echo "   corridor-bounded partial recoveries: $(tr -d '\000' < "$LOG" | grep -c 'corridor-bounded')"
  echo "   deferred recoveries:                 $(tr -d '\000' < "$LOG" | grep -c 'recovery deferred')"
  echo "   cuMemCreate failures anywhere:       $(tr -d '\000' < "$LOG" | grep -c 'cuMemCreate failed')"

  echo
  echo "-- load mix (labelled, per spec item 14)"
  echo "   soak_631_mixed_load: $(tail -1 "$OUT/soak.log" 2>/dev/null)"
  echo "   s26_fill_load:       $(tail -1 "$OUT/fill.log" 2>/dev/null)"
  echo "   qwen agents through router 30099 doing real repository-analysis"
  echo "   tasks: the only load class the spec accepts as the acceptance"
  echo "   carrier; the two above are occupancy top-up and are labelled as such."
} > "$EX" 2>&1

echo "extract -> $EX"
