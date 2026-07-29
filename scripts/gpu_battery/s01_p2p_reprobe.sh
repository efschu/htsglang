#!/usr/bin/env bash
# S1 -- P2P re-probe after the driver update.
#
# One call, a few minutes, no sglang boots: capability matrix -> d2d bench ->
# NCCL transport check. Every placement and transport verdict on this rig was
# made on a machine without GPUDirect P2P; this step is what makes those
# verdicts re-examinable.
#
# LOCKS: run_all.sh takes /tmp/gpu-card-N.lock itself. run_step.sh therefore
# only verifies they are free and does NOT hold them -- holding them would make
# run_all.sh abort on its own acquisition.
#
# No expected outcome is encoded anywhere. Whether NCCL picks P2P, and whether
# the 3080's 256-MiB window is fully usable, is what the run determines. The
# check tests that the MEASUREMENT happened and is consumable, not what it says.

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh

DIR="${BATTERY_STEP_DIR:?BATTERY_STEP_DIR fehlt -- ueber run_step.sh starten}"
RESULTS="$DIR/results"
mkdir -p "$RESULTS"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHON="$PY"

# --baseline: the pre-update capture, when one exists. Its absence is not an
# error -- the diff column simply stays empty and verdict_diff.md is filled
# against the SHM rows recorded in INTEGRATION_R3_VALIDATION instead.
BASELINE_ARGS=()
if [ -n "${P2P_BASELINE:-}" ] && [ -f "${P2P_BASELINE}" ]; then
    BASELINE_ARGS=(--baseline "$P2P_BASELINE")
    echo "Baseline: $P2P_BASELINE"
else
    echo "Baseline: keine (Diff-Spalte bleibt leer, das ist zulaessig)"
fi

bash "$WT/scripts/p2p_readiness/run_all.sh" \
    --results-dir "$RESULTS" "${BASELINE_ARGS[@]}"
RC=$?

# The package writes into a dated sub-directory when --results-dir is absent;
# with it, the three artifacts land directly in $RESULTS. Normalise either way
# so the check has one place to look.
for f in capability_matrix.json d2d_bench.json nccl_transport.json run.log verdict_diff.md; do
    if [ ! -f "$RESULTS/$f" ]; then
        found="$(find "$RESULTS" -name "$f" -type f 2>/dev/null | head -1)"
        [ -n "$found" ] && cp "$found" "$RESULTS/$f"
    fi
done

echo "run_all.sh rc=$RC"
echo "Artefakte:"
ls -la "$RESULTS"
exit "$RC"
