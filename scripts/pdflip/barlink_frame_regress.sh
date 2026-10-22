#!/bin/bash
# barlink_frame_regress.sh -- two-card regression for Fix A (framed BAR1 slots, 22.10.).
#
# Runs scripts/pdflip/barlink_round_bench.py N times as a FRESH process pair (fresh
# transport = fresh slot areas = the first-touch condition of switch-p2p-wurzel-1022
# §2.2) over windows x messages, every all_reduce with the ROUND TERM and verified,
# plus all_gather / broadcast / all_to_all_single per cell. Prints one line per run
# and "<label>: <bad> of <N> runs wrong" at the end. Exit 0 only if 0 of N.
#
# It does NOT book anything and does NOT touch PCI config: pass a RUNNING gpuq
# booking that covers the cards (the bench refuses otherwise), take the card locks
# and MEASURE_WINDOW yourself (rig rules, /spinning/gpu-arb/README.md).
#
# Usage: barlink_frame_regress.sh <label> <booking-id> <flliper-python-tree> [N] [windows] [messages]
#   defaults: N=32, windows=16,40, messages=24,96,128, cards 0,1 (CARDS=...)
# Env: TORCH_EXTENSIONS_DIR is honoured (keep old/new trees in separate dirs, or
#      each switch rebuilds the extension); OUTDIR (default /tmp/barlink_frame_regress).
set -u
L=${1:?label}; BK=${2:?gpuq booking id}; TREE=${3:?flliper python tree}
N=${4:-32}; W=${5:-16,40}; M=${6:-24,96,128}; CARDS=${CARDS:-0,1}
OUT=${OUTDIR:-/tmp/barlink_frame_regress}; mkdir -p "$OUT"
BENCH=$(dirname "$(readlink -f "$0")")/barlink_round_bench.py
bad=0
for i in $(seq 1 "$N"); do
  P=$((29700 + (i % 90))); LOG=$OUT/${L}_$i.log
  CUDA_DEVICE_ORDER=PCI_BUS_ID timeout -k 5 ${RUN_TIMEOUT_S:-150} python3 "$BENCH" --cards "$CARDS" \
    --windows "$W" --messages "$M" --secs "${SECS:-0.5}" --warmup "${WARMUP:-0.2}" \
    --nccl 0 --gloo-rung 0 --max-used-mib "${MAX_USED_MIB:-4096}" --port "$P" \
    --gpuq-booking "$BK" --flliper-python "$TREE" --verify-calls "${VERIFY_CALLS:-4}" \
    --other-ops "${OTHER_OPS:-1}" --out "$OUT/${L}_$i.json" > "$LOG" 2>&1
  rc=$?
  pkill -f "barlink_round_bench.py.*--port $P" 2>/dev/null
  if [ $rc -eq 0 ] && ! grep -q "WRONG RESULT" "$LOG"; then
    echo "$(date -u +%T) $L run $i OK"
  else
    bad=$((bad + 1))
    why=$(grep -m1 -oE "WRONG RESULT: [^.]*" "$LOG" | cut -c1-160)
    [ -z "$why" ] && why="rc=$rc $(grep -m1 -E 'Error|Aborted|REFUSED|status 3' "$LOG" | cut -c1-160)"
    echo "$(date -u +%T) $L run $i FAIL $why"
  fi
done
echo "$L: $bad of $N runs wrong (windows $W MiB, messages $M MiB, cards $CARDS)"
[ $bad -eq 0 ]
