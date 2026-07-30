#!/usr/bin/env bash
# S6 -- NCCL / system-RAM reference measurement, #279 rate source 3.
#
# The measurement the dispatcher's third source has been waiting for. Its
# FORMAT was defined in htccl_path_rates before the run existed, so this step
# writes directly loadable JSON and needs no glue.
#
# NCCL_DEBUG=INFO output goes to a FILE, never to stdout: a debug log in an
# agent's context is a token fire and the check greps the file anyway.

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh

DIR="${BATTERY_STEP_DIR:?BATTERY_STEP_DIR fehlt -- ueber run_step.sh starten}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$WT/python:${PYTHONPATH:-}"

echo "== Plan =="
"$PY" "$BATTERY_DIR/s06_nccl_reference.py" --dry-run

echo "== Messung =="
# --timeout is the budget for ONE pair, both ranks together. Three pairs at 480s
# stay inside the step's 1800s budget with room for the stack dumps and the
# write-out, so a wedged pair is answered by this script and not by the outer
# kill -- which loses the partial result the script writes after every pair.
"$PY" "$BATTERY_DIR/s06_nccl_reference.py" \
    --out "$DIR/nccl_reference.json" \
    --log "$DIR/nccl_debug.log" \
    --timeout 480
RC=$?

echo "rc=$RC"
ls -la "$DIR"
exit "$RC"
