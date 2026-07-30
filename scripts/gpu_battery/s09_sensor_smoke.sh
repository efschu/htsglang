#!/usr/bin/env bash
# S9 -- gdn / KV-pressure ladder smoke.
#
# A small model on ONE card, resolved at runtime. Deliberately not the 27B
# vehicles: the question is whether the ladder flags survive a boot and whether
# the sensor eats this rig's own occupancy numbers, and neither question gets a
# better answer from a bigger model -- only a more expensive one.
#
# Time-boxed everywhere: the server gets a bounded wait, the probe gets a
# bounded sampling window, and every curl carries -m.

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh

DIR="${BATTERY_STEP_DIR:?BATTERY_STEP_DIR missing -- start via run_step.sh}"
PORT="${PORT:-30099}"
MODEL="${SMOKE_MODEL:-$MODEL_ROOT/Qwen3.5-4B}"
LOG="$DIR/server.log"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$WT/python:${PYTHONPATH:-}"
export SGLANG_OFFLOAD_REGISTER=1

# The hosting card is resolved, never assumed: the biggest one, found through
# the same PCI join every other step uses.
CUDA_BIG="$("$PY" - <<'PY'
import torch
print(max(range(torch.cuda.device_count()),
          key=lambda i: torch.cuda.get_device_properties(i).total_memory))
PY
)"
echo "smoke card: cuda:$CUDA_BIG"
export CUDA_VISIBLE_DEVICES="$CUDA_BIG"

cd "$WT"
setsid "$VENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size 1 \
    --context-length 8192 \
    --trust-remote-code \
    --max-running-requests 8 \
    --enable-metrics \
    --gdn-state-set-ladder 4,2,1 \
    --gdn-state-set-ladder-hysteresis 2 \
    --kv-pressure-ladder relief:dcp_ratio \
    --kv-pressure-pre-stage \
    --host 127.0.0.1 --port "$PORT" \
    > "$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >> "$DIR/pids"

cleanup() {
    battery_dump_and_kill "$SERVER_PID" "$DIR/pyspy-server.txt"
}
trap cleanup INT TERM

if ! battery_wait_for_server "$PORT" 900 "$SERVER_PID"; then
    echo "server not up -- dump and abort"
    cleanup
    exit 1
fi

"$PY" "$BATTERY_DIR/s09_sensor_smoke.py" \
    --port "$PORT" --out "$DIR/sensor_smoke.json" --sample-seconds 45
RC=$?

curl -sf -m 10 "http://127.0.0.1:$PORT/get_server_info" > "$DIR/server_info.json"

cleanup
echo "rc=$RC"
exit "$RC"
