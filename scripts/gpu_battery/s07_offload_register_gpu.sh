#!/usr/bin/env bash
# S7 -- offload register on GPU: CudaDeviceOps, real item sizes, retrieval
# latency per class (#286 GPU restlist items 1 and 4).

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh

DIR="${BATTERY_STEP_DIR:?BATTERY_STEP_DIR fehlt -- ueber run_step.sh starten}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$WT/python:${PYTHONPATH:-}"
export SGLANG_OFFLOAD_REGISTER=1

# The plan phase is the probe itself against FakeDeviceOps: same policies,
# same register, same three routes, no card. A probe that cannot cycle
# hermetically has no business touching the card in the next line.
echo "== Plan =="
if ! "$PY" "$BATTERY_DIR/s07_offload_register_gpu.py" --dry-run; then
    echo "STOP: die Sonde scheitert schon hermetisch (--dry-run) -- keine Karte anfassen" >&2
    echo "rc=2"
    exit 2
fi

echo "== Messung =="
"$PY" "$BATTERY_DIR/s07_offload_register_gpu.py" --out "$DIR/offload_register_gpu.json"
RC=$?

echo "rc=$RC"
exit "$RC"
