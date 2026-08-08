#!/bin/bash
# #651 window arm 2: EXACT-FILE discriminator — the laptop's Q4_K_M on CUDA,
# TP=2 (5090 + one 3080), laptop-shaped flags otherwise. Closes the
# "Q4_K_M-specific interaction" suspect without the laptop, modulo the TP
# split (TP=2 vs the laptop's TP=1), which the verdict must state honestly.
#
#   ./rig_window_probe_arm2.sh --smoke
#   ./rig_window_probe_arm2.sh
set -u

TREE=/spinning/wt-gguf-q4-651
EXPECTED_BRANCH=feat/gguf-q4-bringup-651
VENV=/spinning/htsglang-gpu/.venv
MODEL=/spinning/llm_stuff/club-3090/models-cache/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
EXPECTED_SHA=0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b
TOKDIR=$(dirname "$MODEL")
PORT=31656
ARB=/spinning/gpu-arb
SESSION=agent-651-gguf
OUT=$TREE/docs/dev/651/p2/results
LOG=/tmp/651_rig_probe_arm2_$(date +%H%M%S).log

fail() { echo "FAIL: $*" >&2; exit 1; }

[ -f "$MODEL" ] || fail "Q4_K_M not downloaded yet: $MODEL"
branch=$(git -C "$TREE" rev-parse --abbrev-ref HEAD)
[ "$branch" = "$EXPECTED_BRANCH" ] || fail "tree on '$branch'"
head=$(git -C "$TREE" rev-parse --short HEAD)
IDX5090=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' '/5090/{print $1; exit}')
IDX3080=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' '/3080/{print $1; exit}')
[ -n "$IDX5090" ] && [ -n "$IDX3080" ] || fail "need a 5090 and a 3080"
CARDS="$IDX5090,$IDX3080"

if [ "${1:-}" = "--smoke" ]; then
  # sha check is expensive (22.6 GB); smoke checks presence + size only.
  sz=$(stat -c%s "$MODEL")
  [ "$sz" = "22663387424" ] || fail "size $sz != expected"
  echo "SMOKE OK: tree=$head cards=$CARDS model size ok (sha checked at download)"
  exit 0
fi

mkdir -p "$OUT"
PGID=$(ps -o pgid= -p $$ | tr -d " ")
mark() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" >> "$ARB/progress.$SESSION"; }

for i in $IDX5090 $IDX3080; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i")
  [ "$used" -lt 500 ] || fail "card $i not free ($used MiB)"
done

cat > "$ARB/holder" << EOF
session=$SESSION cards=$CARDS purpose=651-EXACT-FILE-TP2 since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
pgids=$PGID (port $PORT, tree $head, TP=2 Q4_K_M probe, ~20 min)
note=Exact laptop file on CUDA TP=2. Kill only pgid $PGID on forfeit.
EOF
mark "arm2 start: Q4_K_M TP=2 on cards $CARDS, tree $head, pgid $PGID"

cleanup() {
  mark "arm2 teardown"
  kill -- -"$SERVER_PGID" 2>/dev/null
  for i in $(seq 1 20); do pgrep -g "$SERVER_PGID" >/dev/null 2>&1 || break; sleep 2; done
  pgrep -g "$SERVER_PGID" >/dev/null 2>&1 && kill -9 -- -"$SERVER_PGID" 2>/dev/null
  sleep 3
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$IDX5090")
  u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$IDX3080")
  mark "arm2 teardown done: used=$u,$u2 MiB"
  echo "cards released: $u,$u2 MiB"
}

setsid env CUDA_VISIBLE_DEVICES="$CARDS" PYTHONPATH="$TREE/python" \
  "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path "$TOKDIR" \
  --load-format gguf --quantization gguf \
  --device cuda --tp-size 2 \
  --disable-custom-all-reduce \
  --context-length 8192 --max-total-tokens 8192 \
  --max-running-requests 1 \
  --attention-backend triton --sampling-backend pytorch \
  --disable-cuda-graph --disable-radix-cache \
  --mamba-radix-cache-strategy no_buffer \
  --disable-overlap-schedule --page-size 1 \
  --mem-fraction-static 0.85 --chunked-prefill-size 1024 \
  --host 127.0.0.1 --port $PORT --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
SERVER_PGID=$(ps -o pgid= -p $SERVER_PID | tr -d " ")
trap cleanup EXIT

for i in $(seq 1 90); do
  grep -q "fired up and ready" "$LOG" && break
  grep -qiE "Traceback|Scheduler hit|CUDA error|out of memory" "$LOG" && {
    mark "arm2 BOOT FAULT"; grep -iE "Traceback|Error" "$LOG" | head -5; exit 1; }
  [ $((i % 6)) -eq 0 ] && mark "arm2 boot wait ${i}0s: $(tail -1 "$LOG" | cut -c1-80)"
  sleep 10
done
grep -q "fired up and ready" "$LOG" || { mark "arm2 BOOT TIMEOUT"; exit 1; }
mark "arm2 ready, probing"

STAMP=$(date +%H%M%S)
env PYTHONPATH="$TREE/python" "$VENV/bin/python" "$TREE/docs/dev/651/probe.py" $PORT \
  2>&1 | tee "$OUT/rig_cuda_q4km_tp2_probe_$STAMP.txt"
env PYTHONPATH="$TREE/python" "$VENV/bin/python" "$TREE/docs/dev/651/p2/scripts/det_probe.py" $PORT 8 \
  2>&1 | tee "$OUT/rig_cuda_q4km_tp2_det_$STAMP.txt" | tail -6
mark "arm2 done: $(grep VERDICT "$OUT/rig_cuda_q4km_tp2_probe_$STAMP.txt" | head -1)"
echo "RESULTS: $OUT/rig_cuda_q4km_tp2_probe_$STAMP.txt"
