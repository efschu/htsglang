#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 CAPACITY RUNG HARNESS: judge one pool/budget rung at the DESIGN POINT.
#
# WHY THIS EXISTS. The capacity ladder up to 540000 tokens was climbed with
# a corridor number taken from ONE bs=1 generation, and bs=1 is not the
# design point. Re-measured at bs=4 (2026-08-09, successor 18), that same
# 540000 rung breached the 1024 MiB floor on ALL THREE cards -- minimum free
# 355 / 42 / 661 MiB. A rung is therefore only real when it is measured
# under concurrency, and this script is the one instrument that does it, so
# that every rung in the ladder is comparable to every other.
#
# THE LOAD IS THE DESIGN POINT, NOT A STRESS TEST. --max-running-requests is
# 4, so the driver keeps exactly 4 requests in flight: 3 long generations
# (the resident decode set that must survive a cutover) plus 1 large-prompt
# worker on a cadence (the arming pressure that makes the policy flip).
# Raising this past 4 would measure a configuration the server refuses to
# run and would make the corridor number pessimistic for no reason.
#
# THE MINIMUM IS A TIME SERIES, NOT A SNAPSHOT. The corridor law is written
# on the NVML FREE column sampled at 100 ms, and judged on the running
# minimum -- never on total-minus-used (the driver carve-out, ~424 MiB on
# the 3080s and ~519 on the 5090, is hidden from both) and never on a single
# reading after the load stops. Note the allocator does NOT hand pages back:
# a card driven to 42 MiB free stays there when the load ends, so a
# post-run snapshot understates nothing but also proves nothing.
#
# Usage:
#   MINUTES=15 bash scripts/route_a_631_bs4_rung.sh <label>
set -uo pipefail

WT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"
PORT="${PORT:-30030}"
MIN="${MINUTES:-15}"
DECODE_STREAMS="${DECODE_STREAMS:-3}"
PREFILL_TOKENS="${PREFILL_TOKENS:-12000}"
PREFILL_PERIOD="${PREFILL_PERIOD:-8}"
FLOOR="${FLOOR:-1024}"
LABEL="${1:-rung}"
OUT="${OUT:-/spinning/evidence-631}"
LOG="${SERVING_LOG:-/spinning/serving-30030.boot.log}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$OUT"
BASE="$OUT/bs4_${LABEL}_${STAMP}"
CORR="$BASE.corridor.csv"
SOAK="$BASE.soak.log"
REPORT="$BASE.report.txt"

export PYTHONPATH="$WT/python"

if ! curl -s -m 5 -o /dev/null "http://127.0.0.1:$PORT/health"; then
    echo "REFUSE: no healthy server on port $PORT" >&2
    exit 1
fi

# Judge THIS window only. The serving log is rotated, never truncated, so
# the byte offset at start is what separates this rung from the boot and
# from every earlier rung in the same file.
START_OFF=$(stat -c %s "$LOG" 2>/dev/null || echo 0)

{
    echo "=== #656 bs=4 capacity rung: $LABEL ==="
    echo "start   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "minutes $MIN  decode_streams $DECODE_STREAMS  prefill ${PREFILL_TOKENS}tok/${PREFILL_PERIOD}s"
    echo "floor   $FLOOR MiB   serving log offset $START_OFF"
    curl -s -m 8 "http://127.0.0.1:$PORT/get_server_info" 2>/dev/null \
      | tr ',' '\n' | grep -E '"(max_total_tokens|max_running_requests|context_length|mem_fraction_static)"' \
      | sed 's/^/  /'
} > "$REPORT"

SECS=$(awk -v m="$MIN" 'BEGIN{printf "%d", m*60}')
bash "$WT/scripts/corridor_sample.sh" "$SECS" "$CORR" > /dev/null 2>&1 &
CORR_PID=$!
"$PY" "$WT/scripts/soak_631_mixed_load.py" --minutes "$MIN" \
    --decode-streams "$DECODE_STREAMS" \
    --prefill-tokens "$PREFILL_TOKENS" \
    --prefill-period "$PREFILL_PERIOD" > "$SOAK" 2>&1
wait "$CORR_PID" 2>/dev/null || true

{
    echo
    echo "--- CORRIDOR (NVML free column, 100 ms, running minimum) ---"
    awk -F, -v floor="$FLOOR" '
    NR>1 { for (i=2;i<=4;i++) { if (min[i]=="" || $i+0<min[i]) min[i]=$i+0
                                if ($i+0<floor) b[i]++ } n++ }
    END { printf "  samples %d\n", n
          for (i=2;i<=4;i++)
              printf "  nvml gpu%d  min_free %6d MiB  breaches %6d  %s\n",
                     i-2, min[i], b[i]+0, (b[i]+0 ? "BREACH" : "held") }' "$CORR"
    echo
    echo "--- LOAD ---"
    tail -3 "$SOAK"
    echo
    echo "--- PHASE EVIDENCE (this window only) ---"
} >> "$REPORT"

WINDOW="$BASE.window.log"
tail -c +"$((START_OFF + 1))" "$LOG" > "$WINDOW" 2>/dev/null || true
bash "$WT/scripts/phase_evidence_extract.sh" "$WINDOW" >> "$REPORT" 2>&1 || true

{
    echo
    echo "--- FLIPS in window ---"
    grep -c 'PHASE-FLIP DONE' "$WINDOW" 2>/dev/null | sed 's/^/  total /'
    grep -o 'PHASE-FLIP DONE [a-z_]*' "$WINDOW" 2>/dev/null | sort | uniq -c | sed 's/^/  /'
    echo
    echo "--- DEATHS in window (empty is the pass) ---"
    grep -nE 'crashed with exit code|out of memory|OutOfMemory' "$WINDOW" 2>/dev/null | tail -5 | cut -c1-200
    echo "end     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$REPORT"

cat "$REPORT"
echo "report: $REPORT"
