#!/usr/bin/env bash
# The ONE command the executor runs per step.
#
#   BATTERY_RUN=<run dir> bash run_step.sh <step> [--force]
#
# It does, in this order and without asking:
#   1. resolve the step in the step table (prefixes allowed: s02 == s02_boot_a),
#   2. ask the resume gate whether the step should run at all,
#   3. check the VRAM corridor when the step touches a card,
#   4. take /tmp/gpu-card-N.lock when the step table says the battery owns them
#      (and explicitly NOT when the step's own tool takes them),
#   5. run the step script under a HARD timeout from the step table,
#   6. on timeout: py-spy dump every pid the step registered, then kill,
#   7. release the locks,
#   8. run the check and record its verdict in state.json,
#   9. print the verdict line -- and a REPORT-GATE line where the queue asks
#      for a report before the next step.
#
# Exit codes: 0 PASS or SKIP, 1 FAIL, 2 STOP.
#
# The step script's stdout/stderr goes to <step dir>/step.log and NEVER to the
# terminal in full: a server log in an agent's context is a token fire, and the
# check greps the file anyway.

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh

FORCE=""
STEP_TOKEN=""
while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE="--force" ;;
        -*) echo "unknown argument: $1" >&2; exit 2 ;;
        *) STEP_TOKEN="$1" ;;
    esac
    shift
done

if [ -z "$STEP_TOKEN" ]; then
    echo "usage: BATTERY_RUN=<dir> bash run_step.sh <step> [--force]" >&2
    exit 2
fi

RUN="$(battery_run_dir)" || exit 2
STATE="$PY $BATTERY_DIR/battery_state.py --run-dir $RUN"

STEP="$("$PY" "$BATTERY_DIR/battery_state.py" field "$STEP_TOKEN" step_id 2>/dev/null)"
if [ -z "$STEP" ]; then
    echo "BATTERY-STOP $STEP_TOKEN: unknown step (see BATTERY.md)"
    exit 2
fi

field() { "$PY" "$BATTERY_DIR/battery_state.py" field "$STEP" "$1"; }

SCRIPT="$(field script)"
CHECK="$(field check)"
TIMEOUT_S="$(field timeout_s)"
NEEDS_CARDS="$(field needs_cards)"
LOCKS="$(field locks)"
REPORT_GATE="$(field report_gate)"
EXPECTED_MIN="$(field expected_min)"

$STATE init >/dev/null

# --- resume gate ------------------------------------------------------------
GATE_OUT="$($STATE gate "$STEP" $FORCE)"
GATE_RC=$?
case "$GATE_RC" in
    3) echo "$GATE_OUT"; exit 0 ;;   # already green -- nothing to do
    2) echo "$GATE_OUT"; exit 2 ;;   # blocked: a prerequisite is not PASS
esac
echo "$GATE_OUT"

STEP_DIR="$RUN/$STEP"
mkdir -p "$STEP_DIR"
export BATTERY_RUN BATTERY_STEP_DIR="$STEP_DIR" BATTERY_STEP="$STEP"
: > "$STEP_DIR/pids"

STARTED="$(date -Is)"
T0=$(date +%s)

finish() {  # $1 = verdict, $2 = line, $3 = reason
    local dur=$(( $(date +%s) - T0 ))
    $STATE record "$STEP" --verdict "$1" --line "$2" --reason "$3" \
        --started "$STARTED" --duration-s "$dur" >/dev/null
    echo "$2"
    if [ "$1" = "PASS" ] && [ "$REPORT_GATE" = "1" ]; then
        echo "BATTERY-GATE $STEP: REPORT-GATE -- stop and report the result before the next step starts"
    fi
    echo "(took ${dur}s, expected ~$((EXPECTED_MIN * 60))s, budget ${TIMEOUT_S}s, artifacts $STEP_DIR)"
}

# --- corridor ---------------------------------------------------------------
if [ "$NEEDS_CARDS" = "1" ]; then
    if ! battery_assert_corridor 2>"$STEP_DIR/corridor.txt"; then
        finish STOP "BATTERY-STOP $STEP: VRAM corridor red, less than $BATTERY_MIN_FREE_MIB MiB free on at least one card" "corridor"
        exit 2
    fi
fi

# --- locks ------------------------------------------------------------------
case "$LOCKS" in
    battery)
        if ! battery_acquire_locks "$STEP"; then
            finish STOP "BATTERY-STOP $STEP: card lock held by someone else, not broken" "lock"
            exit 2
        fi
        ;;
    tool)
        # The step's own tool takes the locks. Holding them here would make it
        # abort on its own acquisition, so we only verify they are free.
        if ! battery_locks_are_free; then
            finish STOP "BATTERY-STOP $STEP: card lock held by someone else, not broken" "lock"
            exit 2
        fi
        ;;
esac

# --- run --------------------------------------------------------------------
timeout --signal=TERM --kill-after=60 "$TIMEOUT_S" bash "$BATTERY_DIR/$SCRIPT" \
    > "$STEP_DIR/step.log" 2>&1
RC=$?

if [ "$RC" = 124 ] || [ "$RC" = 137 ]; then
    # py-spy BEFORE the kill, for every pid the step registered. A hang whose
    # stack nobody captured has to be reproduced; a dumped one does not.
    n=0
    while read -r pid; do
        [ -z "$pid" ] && continue
        n=$((n + 1))
        battery_dump_and_kill "$pid" "$STEP_DIR/pyspy-$pid.txt"
    done < "$STEP_DIR/pids"
    battery_release_locks
    finish STOP "BATTERY-STOP $STEP: time budget ${TIMEOUT_S}s exceeded, $n process(es) dumped and killed" "timeout"
    exit 2
fi

battery_release_locks

# --- check ------------------------------------------------------------------
LINE="$("$PY" "$BATTERY_DIR/checks/$CHECK" --step-dir "$STEP_DIR" 2>"$STEP_DIR/check.err")"
CRC=$?
if [ -z "$LINE" ]; then
    LINE="BATTERY-STOP $STEP: the check printed no line (see $STEP_DIR/check.err)"
    CRC=2
fi

case "$CRC" in
    0) finish PASS "$LINE" "" ;;
    1) finish FAIL "$LINE" "${LINE#*: }" ;;
    *) finish STOP "$LINE" "${LINE#*: }"; CRC=2 ;;
esac
exit "$CRC"
