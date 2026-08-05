#!/bin/bash
# Run one arm of the spill matrix end to end and write its RESULTS file.
#
# Sequence: assert the card locks are held -> boot -> wait for the real ready
# condition -> sample the VRAM corridor at 1 Hz for the whole load -> drive the
# load -> read the named signals -> tear the boot down by ITS OWN pgid.
#
# Two safety properties this script is built around:
#   * it never kills anything but the pgid boot.sh returned, and it takes a
#     py-spy stack first -- the stack is the finding, the kill destroys it;
#   * it never waits unbounded. Every wait has a deadline and says so.
#
# Usage: run_arm.sh <recipe> <cell> [cell ...]
#   e.g. run_arm.sh K1 H1 H2 H3 H4 H5
# Env: MTT (pressure knob), STREAMS (default 4), LOADSECS (default 20)
set -u

WT=/spinning/wt-spill-matrix
VENV=/spinning/htsglang-gpu/.venv
HERE="$WT/scripts/dev/spill_matrix"
OUT=/spinning/spill-night-20260804/results
PORT=${PORT:-30041}
STREAMS=${STREAMS:-4}
LOADSECS=${LOADSECS:-20}
READY_DEADLINE=${READY_DEADLINE:-900}

RECIPE=${1:?usage: run_arm.sh <recipe> <cell> [cell ...]}
shift
CELLS=("$@")
LOG=/spinning/spill-matrix-${RECIPE}.boot.log
RES="$OUT/RESULTS_${RECIPE}.md"
CORR=/spinning/spill-matrix-${RECIPE}.corridor.tsv
ARM=/spinning/spill-matrix-${RECIPE}.arm.json

# --- 0. the locks must already be held; this script does not take them ------
for i in 0 1 2; do
    [ -d "/tmp/gpu-card-$i.lock" ] || {
        echo "REFUSING: /tmp/gpu-card-$i.lock is not held. Take the locks and" >&2
        echo "write /spinning/gpu-arb/holder before running an arm." >&2
        exit 2
    }
done

echo "== arm $RECIPE: cells ${CELLS[*]} =="
pgid=$(MTT=${MTT:-8192} PORT="$PORT" bash "$HERE/boot.sh" "$RECIPE" | sed -n 's/.*pgid=\([0-9]*\).*/\1/p')
[ -n "$pgid" ] || { echo "boot.sh returned no pgid" >&2; exit 3; }
echo "booted pgid=$pgid log=$LOG"

cleanup() {
    # Own pgid only. Never a broad pkill.
    if [ -n "${pgid:-}" ] && kill -0 "$pgid" 2>/dev/null; then
        echo "-- py-spy stack before teardown (the stack is the finding) --"
        timeout 60 "$VENV/bin/py-spy" dump --pid "$pgid" 2>&1 | head -40
        kill -TERM -"$pgid" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$pgid" 2>/dev/null || break; sleep 1; done
        kill -9 -"$pgid" 2>/dev/null
    fi
    [ -n "${corr_pid:-}" ] && kill "$corr_pid" 2>/dev/null
    return 0
}
trap cleanup EXIT

# --- 1. wait for the REAL condition, with a stated deadline ----------------
if ! "$VENV/bin/python" "$HERE/drive.py" ready "$PORT" "$READY_DEADLINE"; then
    echo "ARM $RECIPE: server never became ready within ${READY_DEADLINE}s"
    echo "-- last 30 log lines --"
    tail -n 30 "$LOG"
    {
        echo "# RESULTS $RECIPE"
        echo
        echo "Verdict: **BLOCKED** -- server did not reach /health within ${READY_DEADLINE}s."
        echo
        echo '```'
        tail -n 30 "$LOG"
        echo '```'
    } > "$RES"
    exit 1
fi

# --- 2. corridor sampling covers the whole load, not a snapshot ------------
bash "$HERE/cards.sh" corridor "$CORR" $((LOADSECS + 120)) >/dev/null 2>&1 &
corr_pid=$!

# --- 3. drive the load, time-boxed ----------------------------------------
load_out=$("$VENV/bin/python" "$HERE/drive.py" load "$PORT" "$STREAMS" "$LOADSECS" "$ARM" 2>&1)
echo "$load_out"

kill "$corr_pid" 2>/dev/null; corr_pid=""
corridor_out=$(bash "$HERE/cards.sh" verdict "$CORR" 2>&1)
echo "$corridor_out"

# --- 4. read the named signals -------------------------------------------
sig_out=""
for c in "${CELLS[@]}"; do
    s=$("$VENV/bin/python" "$HERE/drive.py" signals "$LOG" "$c" 2>&1)
    echo "$s"
    sig_out="$sig_out$s"$'\n'
done

# --- 5. write the RESULTS file -------------------------------------------
{
    echo "# RESULTS $RECIPE"
    echo
    echo "Cells: ${CELLS[*]}"
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo '## Command'
    echo '```'
    DRY=1 MTT=${MTT:-8192} PORT="$PORT" bash "$HERE/boot.sh" "$RECIPE" | tail -1
    echo '```'
    echo
    echo '## Card identity at run time'
    echo '```'
    bash "$HERE/cards.sh" identity
    echo '```'
    echo
    echo '## Load'
    echo '```'
    echo "$load_out"
    echo '```'
    echo
    echo '## VRAM corridor (1 Hz, floor 400 MiB on every card)'
    echo '```'
    echo "$corridor_out"
    echo '```'
    echo
    echo '## Named signals'
    echo '```'
    echo "$sig_out"
    echo '```'
    echo
    echo '## Verdict'
    echo
    echo '_Fill in per cell. A cell is PASS only if its named signal appeared._'
} > "$RES"
echo "wrote $RES"
