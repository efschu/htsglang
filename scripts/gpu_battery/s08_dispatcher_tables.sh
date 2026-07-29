#!/usr/bin/env bash
# S8 -- load the dispatcher rate tables and re-check placeholder neutrality.
#
# CPU-only: no card, no lock, no corridor. It consumes the ARTIFACTS of s01 and
# s06, which is exactly why it can be resumed by itself weeks later without
# re-running a single boot.

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh

DIR="${BATTERY_STEP_DIR:?BATTERY_STEP_DIR fehlt -- ueber run_step.sh starten}"
RUN="$(battery_run_dir)" || exit 2

P2P_DIR="$RUN/s01_p2p_reprobe/results"
NCCL_JSON="$RUN/s06_nccl_reference/nccl_reference.json"

# The #278 GDR TSV is an OPTIONAL fourth source. Its absence is not an error:
# missing sources degrade to placeholders by design, and the neutrality check
# below is precisely what makes that degradation safe.
GDR_ARGS=()
if [ -n "${GDR_TSV:-}" ] && [ -f "$GDR_TSV" ]; then
    GDR_ARGS=(--gdr-tsv "$GDR_TSV")
    echo "GDR-Matrix: $GDR_TSV"
else
    echo "GDR-Matrix: keine (bleibt Platzhalter, zulaessig)"
fi

export PYTHONPATH="$WT/python:${PYTHONPATH:-}"

"$PY" "$BATTERY_DIR/s08_dispatcher_tables.py" \
    --p2p-dir "$P2P_DIR" \
    --nccl "$NCCL_JSON" \
    --out "$DIR/dispatcher_tables.json" \
    "${GDR_ARGS[@]}"
exit $?
