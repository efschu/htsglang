#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 37: close out an acceptance window in one command, with the
# C20 proof line appended to the judged extract rather than reported beside it.
#
# s34_ship.sh's summary is kept verbatim -- it is the shape four shifts have
# been read against. Two blocks are added and both are there because a window
# minimum alone is what let C20 hide for four shifts:
#
#   * the IN-CUTOVER minimum per card, which is where this corpus's depth is
#     made, with the seam-entry gate's own account beside it;
#   * the four-arm judge, on the axes the margin does NOT touch, because a
#     free-memory metric cannot validate a mechanism whose action is to free
#     memory.
#
# The C20 block is APPENDED TO EXTRACT.TXT, so the artefact a successor reads
# carries its own proof instead of pointing at another file.
#
# Usage: bash scripts/s37_ship.sh <outdir> <serving-log> <start-HH:MM>
set -uo pipefail

OUT="${1:?outdir}"
LOG="${2:?serving log}"
START="${3:?window start HH:MM}"
WT=/spinning/wt-631-routea
PY=/spinning/htsglang-gpu/.venv/bin/python
EX="$OUT/EXTRACT.txt"

export SELF="${SELF:-successor 37}"
bash "$WT/scripts/s34_ship.sh" "$OUT" "$LOG"

{
  echo
  echo "===== SUCCESSOR 37 ADDITIONS: REGISTER C20"
  echo
  $PY "$WT/scripts/s37_c20_proof.py" "$OUT/corridor.csv" "$LOG"
  C20=$?
  echo
  echo "   (exit $C20: 0 = the in-cutover minima hold and the margin term ran)"
} >> "$EX" 2>&1

echo
echo "=============== REGISTER C20: THE IN-CUTOVER MINIMUM"
sed -n '/SUCCESSOR 37 ADDITIONS: REGISTER C20/,$p' "$EX"

echo
echo "=============== THE AXES THE MARGIN DOES NOT TOUCH (four arms)"
bash "$WT/scripts/s37_judge.sh" "$OUT" "$LOG" "$START" 2>&1 | tail -32
