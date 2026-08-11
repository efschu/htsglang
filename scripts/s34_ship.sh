#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 34: close out an acceptance window in one command.
#
# Runs the judged extract and then prints ONLY the lines a verdict turns on,
# so the close-out is mechanical at the moment when the temptation to skim is
# highest. It judges nothing itself -- every number here is quoted from
# EXTRACT.txt, which is quoted from the log and the corridor CSV.
#
# Usage: bash scripts/s34_ship.sh <outdir> <serving-log>
set -uo pipefail

OUT="${1:?outdir}"
LOG="${2:?serving log}"
WT=/spinning/wt-631-routea

bash "$WT/scripts/s34_extract.sh" "$OUT" "$LOG" >/dev/null 2>&1
EX="$OUT/EXTRACT.txt"
echo "=============== JUDGED SUMMARY  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "extract: $EX"
echo
grep -E "^   (gpu[0-9]_free|VERDICT|breaching samples|per-card minima)" "$EX"
echo
grep -E "STRICT PURITY|decode graph share|accept length|live slots|pp_to_tp:|tp_to_pp:|FLIP ABANDONED" "$EX"
echo
grep -E "SHRINKS \(rows|backed rows at the LAST|-> the pool RETURNED|prefill-gate arms \(|gate liveness|ARMED on device" "$EX"
echo
grep -E "prompt_tokens=[0-9]+" "$EX"
echo
grep -E "spread at the|gate: [0-9]+ cleared" "$EX"
echo
echo "=============== ACCEPTANCE VERDICT LINE"
grep -E "ACCEPTANCE: |VERDICT: corridor" "$EX" | tail -2
