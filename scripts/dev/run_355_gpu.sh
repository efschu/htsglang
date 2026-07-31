#!/usr/bin/env bash
# #355 GPU follow-up: the two card-side proofs the desk work could not run
# because agent-354 held all three cards through this window. Turnkey, one
# command, one GPU, no model, ~6 minutes.
#
#   CARD=0 bash scripts/dev/run_355_gpu.sh
#
# What it proves (numbers, not adjectives):
#   1. bench_masked_kv_bound_check.py -- ns/call for the masked KV writer with
#      the in-kernel bound check OFF (SGLANG_DISABLE_KV_MASKED_BOUND_CHECK=1,
#      the tl.device_assert lowers to nothing) vs ON (default), interleaved
#      A/B with an A-vs-A noise floor established first, 100 reps x 200 launches
#      per point, on the Qwen3.6-27B TP=3 decode row (4 KV heads x head_dim 256).
#      Expectation: the delta is below the A-vs-A noise floor. The script prints
#      "below noise" / "MEASURABLE" per shape so the claim is falsifiable.
#   2. test_masked_kv_bound_falsifier.py -- 4 subprocess cases:
#        a) an in-range write is byte-correct and touches exactly one row;
#        b) an OUT-OF-RANGE loc makes the process die with a CUDA error whose
#           message contains "masked_set_kv_buffer: loc out of range";
#        c) the SAME out-of-range write with the check compiled out returns
#           exit 0 and a silently corrupted row -- the defect #355 closes,
#           asserted as real so a future silent regression fails here too;
#        d) the production pool entry MHATokenToKVPool.set_kv_buffer(
#           dcp_kv_mask=...) reaches the guard, not just a hand-rolled launch.
#   3. test_kv_store_bound_after_growth.py -- the #352 regression on the shared
#      helper (kv_store_bound now feeds store_cache too), proving the refactor
#      did not disturb the raw path.
#
# Discipline: reads /spinning/gpu-arb, refuses a busy card, writes a holder
# with a heartbeat, does a feasibility-arith preflight, signals only its own
# PIDs, and releases on every exit path.

set -uo pipefail

CARD=${CARD:-0}
WT=/spinning/wt-355
PY=/spinning/htsglang-gpu/.venv/bin/python
ARB=/spinning/gpu-arb
SESSION=agent-355
BUDGET_S=600
MIN_FREE_MIB=400
export PYTHONPATH="$WT/python"

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# --- feasibility arith: the segment needs < 500 MiB; refuse a loaded card ---
used=$(CUDA_VISIBLE_DEVICES=$CARD nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
total=$(CUDA_VISIBLE_DEVICES=$CARD nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
name=$(CUDA_VISIBLE_DEVICES=$CARD nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
free=$(( total - used ))
log "card $CARD = $name, free ${free} MiB (used ${used} / ${total})"
if [ "$used" -gt 900 ]; then
  log "REFUSE: card $CARD is busy (${used} MiB > 900). Pick a free card via CARD=."
  exit 3
fi
if [ "$free" -lt "$MIN_FREE_MIB" ]; then
  log "REFUSE: only ${free} MiB free, corridor is >= ${MIN_FREE_MIB}."
  exit 3
fi

# --- holder with heartbeat ---
HOLDER="$ARB/holder"
hb_pid=""
claim() {
  echo "session=$SESSION  cards=$CARD  purpose=#355 masked-bound microbench+falsifier  since=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$HOLDER"
  ( while true; do touch "$HOLDER" 2>/dev/null; sleep 20; done ) &
  hb_pid=$!
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  [$SESSION] BELEGT card=$CARD purpose=#355 microbench+falsifier (budget ${BUDGET_S}s)" >> "$ARB/log"
}
release() {
  [ -n "$hb_pid" ] && kill "$hb_pid" 2>/dev/null
  rm -f "$HOLDER" 2>/dev/null
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  [$SESSION] FREI card=$CARD #355 window done" >> "$ARB/log"
  log "released card $CARD"
}
trap release EXIT INT TERM
claim
log "claimed card $CARD, heartbeat pid $hb_pid"

cd "$WT"
rc=0

log "=== 1/3 microbench: bound check off vs on (fp8) ==="
CUDA_VISIBLE_DEVICES=$CARD timeout 200 "$PY" \
  benchmark/kernels/bench_masked_kv_bound_check.py --dtype fp8 || rc=1

log "=== 1b/3 microbench (bf16) ==="
CUDA_VISIBLE_DEVICES=$CARD timeout 200 "$PY" \
  benchmark/kernels/bench_masked_kv_bound_check.py --dtype bf16 || rc=1

log "=== 2/3 falsifier (4 subprocess cases) ==="
CUDA_VISIBLE_DEVICES=$CARD timeout 200 "$PY" -m pytest \
  test/registered/mem_cache/test_masked_kv_bound_falsifier.py \
  -v --no-header -p no:randomly || rc=1

log "=== 3/3 #352 regression on the shared helper ==="
CUDA_VISIBLE_DEVICES=$CARD timeout 120 "$PY" -m pytest \
  test/registered/mem_cache/test_kv_store_bound_after_growth.py \
  -q -p no:randomly || rc=1

log "DONE rc=$rc (0 = all card-side proofs green)"
exit $rc
