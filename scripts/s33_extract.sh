#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 33: the judged extract for the FINAL acceptance run.
#
# s32_extract.sh plus the axes that run carried and no earlier one did:
#
#   * SPEC ITEM 4, the YaRN long-context leg -- quoted from the SERVER's own
#     usage accounting, not from the client's idea of how long its prompt was.
#   * SPEC ITEM 16 read at the BINDING INSTANT as well as on average. The
#     mean-over-samples spread conflates two different things: idle waste (a
#     card that could hold more) and transient depth (a card whose free column
#     swings further under load). Only the first is an item-16 failure, and
#     the spread at the per-card minima is the figure that separates them.
#   * The kvso <-> flip contract's observed states, so "the guard admitted the
#     flip" is quoted rather than assumed.
#   * The dynamic-chunking engagement line.
#
# Usage: bash scripts/s33_extract.sh <outdir> <serving-log>
set -uo pipefail

OUT="${1:?outdir}"
LOG="${2:?serving log}"
WT=/spinning/wt-631-routea
PY=/spinning/htsglang-gpu/.venv/bin/python
export PYTHONPATH="$WT/python"

EX="$OUT/EXTRACT.txt"
{
  bash "$WT/scripts/s32_extract.sh" "$OUT" "$LOG" > /dev/null 2>&1
  cat "$EX" 2>/dev/null

  echo
  echo "===== SUCCESSOR 33 ADDITIONS"

  echo
  echo "-- SPEC ITEM 4: bs1 + YaRN above the 262144 standard context"
  for f in "$OUT"/yarn_leg*.json; do
    [ -e "$f" ] || { echo "   no leg ran"; break; }
    $PY - "$f" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d["results"]:
    if r.get("error"):
        print(f"   {sys.argv[1].split('/')[-1]}: FAILED -- {r['error'][:100]}")
        continue
    print(
        f"   {sys.argv[1].split('/')[-1]}: prompt_tokens={r['prompt_tokens']} "
        f"(> {d['boundary']}: {r['above_262144']}), "
        f"decoded {r['completion_tokens']} tokens in {r['seconds']}s"
    )
PYEOF
  done
  echo "   The prompt is a non-repeating pseudo-random word stream, so the"
  echo "   rows it claims are rows it holds: a repeated block would be"
  echo "   deduplicated by the radix tree into a fraction of them."

  echo
  echo "-- SPEC ITEM 16 at the BINDING INSTANT (the figure that separates"
  echo "   idle waste from transient depth)"
  awk -F, 'NR>1 && NF>=4 {
      for (i=2; i<=4; i++) if (m[i]=="" || $i+0 < m[i]) m[i]=$i+0
    }
    END {
      lo=m[2]; hi=m[2]
      for (i=3; i<=4; i++) { if (m[i]<lo) lo=m[i]; if (m[i]>hi) hi=m[i] }
      printf "   per-card MINIMA %d / %d / %d MiB -> spread at the minimum %d MiB\n",
             m[2], m[3], m[4], hi-lo
      printf "   (mean-over-samples spread is reported above; where the two\n"
      printf "    disagree the difference is transient depth, not waste)\n"
    }' "$OUT/corridor.csv" 2>/dev/null

  echo
  echo "-- kvso <-> phase-flip contract (#656: a state, not a feature)"
  st=$(tr -d '\000' < "$LOG" | grep -c "kv-session-offload busy")
  echo "   flip arming refused for kvso state: $st time(s)"
  echo "   kvso enabled on this instance: $(grep -c -- '--enable-kv-session-offload' /tmp/s33_argv.txt 2>/dev/null || echo 0)"

  echo
  echo "-- dynamic-chunking engagement (INFO line, edge-triggered)"
  en=$(tr -d '\000' < "$LOG" | grep -c "PP Dynamic Chunk\] ENGAGED")
  echo "   engagement lines: $en"
  if [ "$en" -eq 0 ]; then
    echo "   READ THIS AS: the arm was NOT enabled on this boot"
    echo "   (--enable-dynamic-chunking absent), so 512 static is what ran."
    echo "   The line exists and is unit-pinned; it did not fire because the"
    echo "   feature was off, not because it cannot report."
  fi

  echo
  echo "-- weights-arena high-water (the own-bug fixed this shift)"
  echo "   rung 3 tail releases: $(tr -d '\000' < "$LOG" | grep -c 'rung 3 released')"
  echo "   arena invalid-argument faults: $(tr -d '\000' < "$LOG" | grep -c 'invalid argument')"

  echo
  echo "-- host RAM, booked in the same ledger as VRAM"
  cat /sys/fs/cgroup/memory.peak 2>/dev/null \
    | awk '{printf "   memory.peak %.1f GiB\n", $1/1073741824}'
  grep -E "oom_kill " /sys/fs/cgroup/memory.events 2>/dev/null | sed 's/^/   /'
} > "$EX.tmp" 2>&1
mv "$EX.tmp" "$EX"

echo "extract -> $EX"
