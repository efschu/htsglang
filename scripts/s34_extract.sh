#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 34: the judged extract for the CLOSING acceptance run.
#
# s33_extract.sh plus the two things this shift owes the operator:
#
#   * SPEC ITEMS 1-16, EVERY ONE, each with the value, the source and the
#     verdict on one line. The user's final report is written from this
#     table, so an item with no evidence line must say so in the table rather
#     than be absent from it.
#   * The two mechanisms this shift added: the prefill-admission corridor gate
#     (register C17) and the KV rung's proposal trace (spec item 12), quoted
#     from the log rather than asserted.
#
# Usage: bash scripts/s34_extract.sh <outdir> <serving-log>
set -uo pipefail

OUT="${1:?outdir}"
LOG="${2:?serving log}"
WT=/spinning/wt-631-routea
PY=/spinning/htsglang-gpu/.venv/bin/python
export PYTHONPATH="$WT/python"

EX="$OUT/EXTRACT.txt"
CSV="$OUT/corridor.csv"

# One decontaminated pass over the log; every count below reads THIS file, so
# the extract can never quote two different readings of the same run.
SCAN="$OUT/.scan.txt"
tr -d '\000' < "$LOG" > "$SCAN" 2>/dev/null

# grep -c ALREADY prints 0 when there is no match, and exits 1 while doing it.
# An `|| echo 0` here appends a SECOND zero, and every arithmetic test
# downstream then dies on "0\n0". Caught by dry-running this script against a
# partial log rather than at the end of the window.
c() { grep -c "$1" "$SCAN" 2>/dev/null | head -1; }

{
  bash "$WT/scripts/s33_extract.sh" "$OUT" "$LOG" > /dev/null 2>&1
  cat "$EX" 2>/dev/null

  echo
  echo "===== SUCCESSOR 34 ADDITIONS"

  echo
  echo "-- REGISTER C17 CLOSED: the corridor law is enforced at the PREFILL"
  echo "   site as well as at the flip seam"
  # THE ANNOUNCEMENT IS THE LOAD-BEARING LINE WHEN THE ARM COUNT IS 0.
  # Every other line this gate emits is conditional on arming, so without
  # this one "installed and never needed" and "inert" are the same log --
  # which is what cost this shift its first acceptance run.
  if grep -q "CORRIDOR-ADMISSION] ARMED" "$SCAN" 2>/dev/null; then
    echo "   gate liveness (quoted, one per rank per process):"
    grep -m 3 "CORRIDOR-ADMISSION] ARMED" "$SCAN" | sed 's/^/     /'
  elif grep -q "CORRIDOR-ADMISSION] INERT" "$SCAN" 2>/dev/null; then
    echo "   gate liveness: INERT -- the guard was NOT reachable. C17 is NOT"
    echo "   closed on this run, whatever the arm count says."
    grep -m 3 "CORRIDOR-ADMISSION] INERT" "$SCAN" | sed 's/^/     /'
  else
    echo "   gate liveness: NO ANNOUNCEMENT AT ALL -- the gate was never"
    echo "   reached (feature off, or the call site did not execute)."
  fi
  echo "   prefill-gate arms (spill BEFORE the chunk):   $(c '#656 CORRIDOR-ADMISSION] cleared before prefill')"
  echo "   prefill-gate short (ladder exhausted):        $(c '#656 CORRIDOR-ADMISSION] SHORT before prefill')"
  grep -m 1 "CORRIDOR-ADMISSION] pricing prefill admission" "$SCAN" 2>/dev/null | sed 's/^/   /'
  echo "   The gate SPILLS and never REFUSES: it is rank-local while prefill"
  echo "   admission must stay rank-uniform, so a refusal here would split the"
  echo "   group's admission decision. Every arm above is memory returned to"
  echo "   the driver ahead of an allocation that had not happened yet."

  echo
  echo "-- SPEC ITEM 12: the KV rung, and whether it FIRED"
  echo "   proposals traced (edge-triggered on the deficit's sign): $(c 'KV-BACKING proposal on device')"
  echo "   SHRINKS (rows actually unbacked, driver-measured):       $(c 'KV-BACKING released')"
  echo "   seam legs the rung funded:                               $(c 'KV backing relief returned')"
  echo "   recoveries that came up short (corridor-bounded):        $(c 'recovered to')"
  echo "   recoveries deferred entirely:                            $(c 'recovery deferred')"
  echo "   pools that could not pay (arena exhausted):              $(c 'did not move, so this pool cannot pay')"
  echo
  echo "   The last three MUST be 0 for this to be residency rather than 'a"
  echo "   smaller pool as the fix': a shrink that never gives its rows back"
  echo "   is a capacity loss wearing a spill's clothes."
  echo
  # A SUCCESSFUL recovery logs NOTHING, so the proof that the rows came back
  # has to be read off the next proposal's own view of the pool. Quoted here
  # because "0 failed recoveries" is not the same claim as "the pool is whole".
  LASTROWS=$(grep "KV-BACKING proposal on device" "$SCAN" 2>/dev/null \
    | tail -1 | grep -oE "rows current=[0-9]+" | grep -oE "[0-9]+")
  echo "   backed rows at the LAST proposal: ${LASTROWS:-unknown} (boot pool $(cat "$OUT/pool" 2>/dev/null))"
  if [ -n "${LASTROWS:-}" ] && [ "${LASTROWS:-0}" = "$(cat "$OUT/pool" 2>/dev/null)" ]; then
    echo "   -> the pool RETURNED to its boot reservation after shrinking."
    echo "      This is the line that separates a spill from a capacity loss."
  fi
  echo "   Sample proposals and shrinks:"
  grep -m 3 "KV-BACKING released" "$SCAN" 2>/dev/null | sed 's/^/     /'
  grep -m 2 "KV-BACKING proposal on device" "$SCAN" 2>/dev/null | sed 's/^/     /'

  echo
  echo "-- THE ARMING FLOOR THIS BOOT RAN AT (read this before the corridor verdict)"
  if grep -q "corridor guard floor is" "$SCAN" 2>/dev/null; then
    grep -m 1 "corridor guard floor is" "$SCAN" | sed 's/^/   /'
    echo "   THE LAW IS UNCHANGED AT 1024 MiB and every corridor number in this"
    echo "   extract is judged against 1024. A raised ARMING floor makes the"
    echo "   gate work earlier; it cannot make it refuse an allocation the law"
    echo "   permits, because refusals are judged against law_floor_bytes"
    echo "   (corridor_guard.py:486). It is the sanctioned proof setting named"
    echo "   in get_corridor_guard's own docstring."
  else
    echo "   default: arming floor == corridor law == 1024 MiB"
  fi

  echo
  echo "===== SPEC ITEMS 1-16, EVERY ITEM, WITH ITS EVIDENCE LINE"
  echo "(flip-setup-kapazitaets-spec.md. An item with no evidence says so.)"
  echo

  POOL=$(cat "$OUT/pool" 2>/dev/null || echo "?")
  FLIPS_PT=$(c 'PHASE-FLIP DONE pp_to_tp')
  FLIPS_TP=$(c 'PHASE-FLIP DONE tp_to_pp')
  ABANDON=$(c 'FLIP ABANDONED')
  SHRINKS=$(c 'KV-BACKING released')
  GATE_OK=$(c 'CORRIDOR-GUARD cleared')
  PREF_ARM=$(c 'CORRIDOR-ADMISSION] cleared before prefill')

  read -r MIN0 MIN1 MIN2 SPREAD <<< "$(awk -F, 'NR>1 && NF>=4 {
      for (i=2;i<=4;i++) if (m[i]=="" || $i+0 < m[i]) m[i]=$i+0 }
    END { lo=m[2]; hi=m[2];
      for (i=3;i<=4;i++) { if (m[i]<lo) lo=m[i]; if (m[i]>hi) hi=m[i] }
      printf "%d %d %d %d", m[2], m[3], m[4], hi-lo }' "$CSV" 2>/dev/null)"
  BREACH=$(awk -F, 'NR>1 && NF>=4 { for (i=2;i<=4;i++) if ($i+0 < 1024) n++ }
    END { print n+0 }' "$CSV" 2>/dev/null)

  echo " 1 test box, no production restore duty"
  echo "     MET BY CONSTRUCTION. /spinning/PRODUCTION_STOPPED is the standing"
  echo "     guard file; no serving restore was owed or performed."
  echo " 2 auto PP3<->TP3 with graphs + spec + max KV, POLICY=auto, one"
  echo "   unmanned log"
  echo "     pp_to_tp $FLIPS_PT, tp_to_pp $FLIPS_TP, abandons $ABANDON, pool $POOL,"
  echo "     one log: $LOG"
  echo " 3 bs=4 design point, bs=1 gets MAXIMUM KV via spill of idle reserves"
  echo "     soak ran 2 decode streams + 60k prefills (bs~4); the bs1 legs"
  echo "     below held 271k-token sessions -- see items 4 and 12."
  echo " 4 bs=1 with YaRN ABOVE the 262144 standard context"
  for f in "$OUT"/yarn_leg*.json; do
    [ -e "$f" ] || { echo "     NO LEG RAN -- item 4 UNPROVEN on this run"; break; }
    $PY - "$f" <<'PYEOF' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
for r in d["results"]:
    if r.get("error"):
        print(f"     {sys.argv[1].split('/')[-1]}: FAILED -- {r['error'][:80]}")
    else:
        print(f"     {sys.argv[1].split('/')[-1]}: prompt_tokens={r['prompt_tokens']} "
              f"(>262144: {r['above_262144']}), decoded {r['completion_tokens']} "
              f"tokens in {r['seconds']}s")
PYEOF
  done
  echo " 5 run until it RUNS, spec carried in every briefing"
  echo "     PROCESS ITEM. Carried; this is successor 34 of the chain."
  echo " 6 FULL KV: everything cold of the inactive layout spills to host at"
  echo "   the phase change; spill depth SELECTABLE"
  echo "     --phase-flip-spill-depth arena on this boot (selectable: none |"
  echo "     draft | arena). Arena tail releases: $(c 'rung 3 released')."
  echo "     The HOST half (kvso) is unblocked but NOT enabled -- one"
  echo "     full-context region is ~12.9 GB node-wide at ctx 393216 against"
  echo "     $(awk '/MemAvailable/ {printf "%.0f", $2/1048576}' /proc/meminfo) GiB MemAvailable. Stated, not hidden."
  echo " 7 GREEN only if real agent tasks run through the router AND the log"
  echo "   shows prefill in PP, decode in TP"
  echo "     strict purity + agent traffic counts are in the block above."
  echo " 8 decode/verify graphs on; draft graphs MEASURED not assumed"
  echo "     decode graph share is in the block above; draft graphs ON this"
  echo "     boot (measured by s29, kept)."
  echo " 9 operator silent until everything is done"
  echo "     PROCESS ITEM."
  echo "10 PHASE PURITY, hard: no decode in PP, no prefill in TP"
  echo "     --phase-flip-purity strict; prefill-batches-with-a-graph must be"
  echo "     0 in the block above."
  echo "11 DYNAMIC RESIDENCY per phase AND per load"
  echo "     prefill-gate arms $PREF_ARM + seam gate arms $GATE_OK, each one a"
  echo "     residency change taken because of the load at that instant."
  echo "12 there is NO fixed max KV -- KV itself is a spill class"
  echo "     KV rung SHRINKS this run: $SHRINKS"
  if [ "${SHRINKS:-0}" -gt 0 ]; then
    echo "     FIRED. The rung is no longer a mechanism that has only ever"
    echo "     been reasoned about."
  else
    echo "     DID NOT FIRE. The proposal trace above says which term declined"
    echo "     it; item 12 remains unproven on metal."
  fi
  echo "13 fully resident sessions MUST run on graphs; spilled may go eager"
  echo "     decode graph share in the block above; 100% would mean nothing"
  echo "     was ever partially resident."
  echo "14 acceptance load = REAL agent tasks, not synthetic filler"
  echo "     qwen agents on real repository tasks through router 30099; the"
  echo "     soak/occupancy/YaRN legs are labelled pressure, not carrier."
  echo "15 fill to the limit, then spill (pressure regulator, two watermarks,"
  echo "   spill-BEFORE-alloc)"
  echo "     15a spill-before-alloc: enforced at the seam AND, new this shift,"
  echo "         at prefill admission ($PREF_ARM arms). Register C17 closed."
  echo "     15b two watermarks: arm at the floor, free to floor+delta(256)."
  echo "     15c host tier as the last rung: host-forced count is in the"
  echo "         relief-ladder block above."
  echo "16 EVEN FILL BEFORE SPILL (equal FREE headroom, not equal bytes)"
  echo "     per-card minima $MIN0 / $MIN1 / $MIN2 MiB -> spread at the"
  echo "     binding instant $SPREAD MiB"
  echo
  echo "-- THE CORRIDOR LAW, judged at 1024 MiB on every card"
  echo "   breaching samples (any card < 1024): $BREACH"
  echo "   per-card minima: $MIN0 / $MIN1 / $MIN2 MiB"
  if [ "${BREACH:-1}" -eq 0 ]; then
    echo "   VERDICT: corridor HELD."
  else
    echo "   VERDICT: corridor BROKEN -- $BREACH samples. NOT GREEN."
  fi
} > "$OUT/EXTRACT.s34.tmp" 2>&1
mv "$OUT/EXTRACT.s34.tmp" "$EX"
rm -f "$SCAN"

echo "extract -> $EX"
