#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 40: pull the #658 / #661 verdict lines out of a window.
#
# NEVER CAT THE SERVER LOG. It is hundreds of MiB and reading it into an agent
# context is how a shift dies. Every question this window asks has a grep, and
# each grep below is one of the acceptance axes -- so the extract IS the
# judgement, not a convenience.
#
# Usage: bash scripts/s40_extract.sh <outdir>
set -uo pipefail
OUT="${1:?outdir}"
LOG="$OUT/serving.log"
[ -f "$LOG" ] || { echo "no serving.log in $OUT" >&2; exit 2; }

echo "===== #658 / #661 WINDOW EXTRACT -- $OUT"
echo "log size: $(du -h "$LOG" | cut -f1)"

echo
echo "-- (1) did the DIAL arm at boot (#658)"
grep -c "VRAM-DIAL armed at boot" "$LOG" 2>/dev/null || true
grep -m1 "VRAM-DIAL armed at boot" "$LOG" 2>/dev/null || echo "   NOT ARMED -- the dial never built; everything below about #658 is void."

echo
echo "-- (2) the corridor law inside the dial's floor (C18)"
echo "   floors reported at arm (MiB), one per rank:"
grep -m1 -o "floors \[[^]]*\] MiB" "$LOG" 2>/dev/null || echo "   (none)"

echo
echo "-- (3) BUDGET REQUESTS seen by the ranks (the external tenant)"
grep -c "VRAM-DIAL budget rank" "$LOG" 2>/dev/null || true
grep "VRAM-DIAL budget rank" "$LOG" 2>/dev/null | head -12

echo
echo "-- (4) THE JOIN: a budget cut spending the corridor relief ladder"
echo "   (this line existing at all is what #658 built)"
grep -c "budget reduction of" "$LOG" 2>/dev/null || true
grep "budget reduction of" "$LOG" 2>/dev/null | head -12

echo
echo "-- (5) the guard's own ladder lines during the cycle"
grep -c "CORRIDOR-GUARD cleared" "$LOG" 2>/dev/null || true
grep "CORRIDOR-GUARD REFUSED" "$LOG" 2>/dev/null | head -5 || true
grep "external budget reduction" "$LOG" 2>/dev/null | head -8

echo
echo "-- (6) capacity commits (did the residual actually move)"
grep "VRAM-DIAL DONE" "$LOG" 2>/dev/null | head -10
echo "   holds (waiting for a group-idle boundary):"
grep -c "VRAM-DIAL hold" "$LOG" 2>/dev/null || true

echo
echo "-- (7) #661 dynamic-chunk ENGAGEMENT (edge-triggered INFO)"
n=$(grep -c "PP Dynamic Chunk. ENGAGED" "$LOG" 2>/dev/null || echo 0)
echo "   engagement lines: $n"
if [ "$n" = "0" ]; then
  echo "   NOT ENGAGED. Either the flag is absent or the predictor never"
  echo "   returned a width differing from --chunked-prefill-size, so the"
  echo "   dynamic arm measured the STATIC path and the A/B is void."
else
  grep -o "chunk width [0-9]* (static --chunked-prefill-size is [0-9]*, delta [-+][0-9]*" "$LOG" \
    | sed 's/^/   /' | head -20
  echo "   distinct widths:"
  grep -o "chunk width [0-9]*" "$LOG" | sort -u | head -20 | sed 's/^/     /'
fi

echo
echo "-- (8) health of the window: tracebacks / OOM / flip refusals"
echo "   tracebacks:      $(grep -c 'Traceback (most recent call last)' "$LOG" 2>/dev/null || echo 0)"
echo "   CUDA OOM:        $(grep -c 'CUDA_ERROR_OUT_OF_MEMORY\|out of memory' "$LOG" 2>/dev/null || echo 0)"
echo "   KvCapacityError: $(grep -c 'KvCapacityError' "$LOG" 2>/dev/null || echo 0)"
echo "   KvReshardError:  $(grep -c 'KvReshardError' "$LOG" 2>/dev/null || echo 0)"
echo "   seam refusals:   $(grep -c 'CORRIDOR-GUARD REFUSED' "$LOG" 2>/dev/null || echo 0)"

echo
echo "-- (9) the corridor law, at 100 ms, over the whole window"
for csv in "$OUT"/corridor*.csv; do
  [ -f "$csv" ] || continue
  echo "   $(basename "$csv"):"
  awk -F, -v floor=1024 'NR>1 {
      for (i = 2; i <= 4; i++) {
          if (min[i] == 0 || $i < min[i]) min[i] = $i
          if ($i < floor) breach[i]++
      }
      n++
  } END {
      printf "     samples=%d\n", n
      for (i = 2; i <= 4; i++)
          printf "     gpu%d min_free=%6d MiB breaches=%d\n", i-2, min[i], breach[i]+0
  }' "$csv"
done
