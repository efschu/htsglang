#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #631 GREEN CRITERION run: the switch counts green ONLY when serving works
# as a real agent backend while an unmanned >=60 min log evidences prefill
# in PP, decode in TP with graphs, auto flips BOTH directions, zero rank
# deaths, the corridor floor held, and STRICT PHASE PURITY.
#
# THE LOAD IS DELIBERATELY NOT PREFILL-SATURATED. The previous soak driver
# sent 12000 tokens every 4 s, which pins pending prefill above N by
# construction: under that load the policy can never be observed choosing,
# so it can neither prove nor disprove fairness. Here the synthetic part is
# a BACKGROUND trickle (one long prompt per PREFILL_PERIOD, default 25 s)
# whose job is only to guarantee the PP layout has work; the decode side and
# the realistic burstiness come from the live Qwen agent traffic arriving
# through the router.
set -uo pipefail

WT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"
OUT="${OUT:-/spinning/evidence-631}"
MIN="${MINUTES:-65}"
PREFILL_PERIOD="${PREFILL_PERIOD:-25}"
PREFILL_TOKENS="${PREFILL_TOKENS:-12000}"
DECODE_STREAMS="${DECODE_STREAMS:-2}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$OUT"
SOAK="$OUT/green_${STAMP}.soak.log"
CORR="$OUT/green_${STAMP}.corridor.csv"
INFO="$OUT/green_${STAMP}.server_info.json"
EVID="$OUT/green_${STAMP}.phase_evidence.txt"
VERD="$OUT/green_${STAMP}.verdict.txt"

echo "green run $STAMP: waiting for health" | tee "$VERD"
until curl -s -m 3 -o /dev/null http://127.0.0.1:30030/health 2>/dev/null; do
    sleep 5
done
echo "healthy at $(date -u +%H:%M:%SZ)" | tee -a "$VERD"
curl -s -m 8 http://127.0.0.1:30030/get_server_info > "$INFO" 2>/dev/null || true

# The serving log is truncated-by-rotation nowhere, so record the byte
# offset we start at: the verdict must judge THIS window, not the boot.
LOG=/spinning/serving-30030.boot.log
START_OFF=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
echo "serving log start offset $START_OFF" | tee -a "$VERD"

"$PY" -u "$WT/scripts/soak_631_mixed_load.py" \
    --minutes "$MIN" --prefill-tokens "$PREFILL_TOKENS" \
    --prefill-period "$PREFILL_PERIOD" --decode-streams "$DECODE_STREAMS" \
    > "$SOAK" 2>&1 &
SOAK_PID=$!
bash "$WT/scripts/corridor_sample.sh" $((MIN * 60)) "$CORR" \
    > "$OUT/green_${STAMP}.corridor.summary" 2>&1 &
CORR_PID=$!

echo "soak pid $SOAK_PID corridor pid $CORR_PID; window $MIN min" | tee -a "$VERD"
wait $SOAK_PID
wait $CORR_PID 2>/dev/null

echo "=== window closed $(date -u +%H:%M:%SZ) ===" | tee -a "$VERD"

# Judge ONLY the bytes written during this window.
WINDOW_LOG="$OUT/green_${STAMP}.serving_window.log"
tail -c +$((START_OFF + 1)) "$LOG" > "$WINDOW_LOG" 2>/dev/null || true

bash "$WT/scripts/phase_evidence_extract.sh" "$WINDOW_LOG" > "$EVID" 2>&1
PURITY_RC=$?
cat "$EVID" >> "$VERD"

DEATHS=$(grep -ac "Scheduler hit an exception" "$WINDOW_LOG" 2>/dev/null || echo 0)
ERRLINE=$(tail -1 "$SOAK" 2>/dev/null)
{
    echo ""
    echo "=== GREEN CRITERION VERDICT ==="
    echo "purity+both-layouts exit code : $PURITY_RC (0 required)"
    echo "scheduler exceptions in window: $DEATHS (0 required)"
    echo "last soak line               : $ERRLINE"
    echo "corridor summary             : $OUT/green_${STAMP}.corridor.summary"
} | tee -a "$VERD"
