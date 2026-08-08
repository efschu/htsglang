#!/bin/bash
# #651 rig-window discriminator: 35B GGUF on CUDA (5090, TP=1), laptop-shaped
# flags, content probe. Coherent -> laptop defect is ROCm/gfx1103 device-side.
# Incoherent -> tree runtime regression since dff1ef16c0 (bisect on the rig).
#
# Usage:
#   ./rig_window_probe.sh --smoke   # desk validation, touches no cards
#   ./rig_window_probe.sh           # the real window run (needs the go-signal)
#
# Arbitration: writes /spinning/gpu-arb/holder with this shell's pgid, appends
# progress markers, tears down to 0 MiB, restores holder state on exit.
set -u

TREE=/spinning/wt-gguf-q4-651
EXPECTED_BRANCH=feat/gguf-q4-bringup-651
VENV=/spinning/htsglang-gpu/.venv
MODEL=${MODEL:-/spinning/llm_stuff/club-3090/models-cache/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf}
TOKDIR=$(dirname "$MODEL")
PORT=31655
ARB=/spinning/gpu-arb
SESSION=agent-651-gguf
OUT=$TREE/docs/dev/651/p2/results
LOG=/tmp/651_rig_probe_$(date +%H%M%S).log

fail() { echo "FAIL: $*" >&2; exit 1; }

# ---- desk checks (both modes) ----------------------------------------------
[ -f "$MODEL" ] || fail "model missing: $MODEL"
[ -f "$TOKDIR/config.json" ] || fail "sibling config.json missing"
[ -x "$VENV/bin/python" ] || fail "venv missing"
branch=$(git -C "$TREE" rev-parse --abbrev-ref HEAD)
[ "$branch" = "$EXPECTED_BRANCH" ] || fail "tree on '$branch', expected $EXPECTED_BRANCH"
head=$(git -C "$TREE" rev-parse --short HEAD)
PY="env CUDA_VISIBLE_DEVICES= PYTHONPATH=$TREE/python $VENV/bin/python"
$PY -c "import sglang, sys; p = sglang.__file__; assert p.startswith('$TREE/'), p" \
  || fail "PYTHONPATH does not resolve to the pinned tree"
$PY -c "import ast; ast.parse(open('$TREE/docs/dev/651/probe.py').read())" \
  || fail "probe.py unparseable"
IDX=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' '/5090/{print $1; exit}')
[ -n "$IDX" ] || fail "no 5090 found via nvidia-smi"
echo "desk checks ok: tree=$head 5090=index $IDX model=$(basename "$MODEL")"
# Capacity gate: the rig 5090's CUDA-VISIBLE total is ~19.58 GiB (BAR1-pinned
# environment), NOT the 32 GiB nvidia-smi shows. Weights must fit it TP=1.
CUDA_TOTAL_GIB=$(env CUDA_VISIBLE_DEVICES="$IDX" PYTHONPATH="$TREE/python" "$VENV/bin/python" -c "import torch; print(torch.cuda.mem_get_info(0)[1] / 2**30)")
MODEL_GIB=$(python3 -c "import os; print(os.path.getsize('$MODEL') / 2**30)")
python3 -c "import sys; t, m = float('$CUDA_TOTAL_GIB'), float('$MODEL_GIB'); sys.exit(0 if m + 1.5 < t else 1)" \
  || fail "capacity: model ${MODEL_GIB%.*} GiB + margin exceeds CUDA-visible ${CUDA_TOTAL_GIB%.*} GiB on the 5090 (BAR1-pinned env). Use a smaller quant (Q3_K_XL) or a TP>=2 window."
echo "capacity ok: model ${MODEL_GIB} GiB vs CUDA-visible ${CUDA_TOTAL_GIB} GiB"

if [ "${1:-}" = "--smoke" ]; then
  echo "SMOKE OK (no cards touched)"
  exit 0
fi

# ---- window run -------------------------------------------------------------
mkdir -p "$OUT"
PGID=$(ps -o pgid= -p $$ | tr -d " ")
mark() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" >> "$ARB/progress.$SESSION"; }

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$IDX")
[ "$used" -lt 500 ] || fail "5090 not free ($used MiB used) - window not actually open"

cat > "$ARB/holder" << EOF
session=$SESSION cards=$IDX purpose=651-CUDA-DISCRIMINATOR since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
pgids=$PGID (port $PORT, tree $head, eager TP=1 probe run, ~20 min)
note=35B GGUF coherence discriminator: laptop-shaped flags on CUDA. Kill only pgid $PGID on forfeit.
EOF
mark "window start: booting $(basename "$MODEL") on 5090 idx $IDX, tree $head, pgid $PGID"

cleanup() {
  mark "teardown: killing server"
  kill -- -"$SERVER_PGID" 2>/dev/null
  for i in $(seq 1 20); do
    pgrep -g "$SERVER_PGID" >/dev/null 2>&1 || break
    sleep 2
  done
  pgrep -g "$SERVER_PGID" >/dev/null 2>&1 && kill -9 -- -"$SERVER_PGID" 2>/dev/null
  sleep 3
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$IDX")
  mark "teardown done: 5090 used=${u} MiB (must be ~0)"
  echo "cards released: 5090 used=${u} MiB"
}

setsid env CUDA_VISIBLE_DEVICES="$IDX" PYTHONPATH="$TREE/python" \
  "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path "$TOKDIR" \
  --load-format gguf --quantization gguf \
  --device cuda --tp-size 1 \
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

for i in $(seq 1 60); do
  grep -q "fired up and ready" "$LOG" && break
  grep -qiE "Traceback|Scheduler hit|CUDA error|out of memory" "$LOG" && {
    mark "BOOT FAULT (see $LOG)"; grep -iE "Traceback|Error" "$LOG" | head -5; exit 1; }
  [ $((i % 6)) -eq 0 ] && mark "boot wait ${i}0s: $(tail -1 "$LOG" | cut -c1-80)"
  sleep 10
done
grep -q "fired up and ready" "$LOG" || { mark "BOOT TIMEOUT"; exit 1; }
mark "server ready, running probes"

STAMP=$(date +%H%M%S)
env PYTHONPATH="$TREE/python" "$VENV/bin/python" "$TREE/docs/dev/651/probe.py" $PORT \
  2>&1 | tee "$OUT/rig_cuda_probe_$STAMP.txt"
env PYTHONPATH="$TREE/python" "$VENV/bin/python" "$TREE/docs/dev/651/p2/scripts/det_probe.py" $PORT 8 \
  2>&1 | tee "$OUT/rig_cuda_det_probe_$STAMP.txt" | tail -6
mark "probes done: $(grep VERDICT "$OUT/rig_cuda_probe_$STAMP.txt" | head -1); det: $(grep VERDICT "$OUT/rig_cuda_det_probe_$STAMP.txt" | head -1)"
echo "RESULTS: $OUT/rig_cuda_probe_$STAMP.txt  $OUT/rig_cuda_det_probe_$STAMP.txt"
