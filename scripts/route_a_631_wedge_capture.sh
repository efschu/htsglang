#!/usr/bin/env bash
# #631/#656: wedge evidence capture for the PP/TP phase flip.
#
# WHY THIS EXISTS. The boot-18 wedge was diagnosed from a py-spy that lived
# only in an agent's context, against a serving log that the next boot
# truncated four minutes later. Rank 2's stack -- the one datum that would
# have settled the mechanism -- was never written to disk and is now
# permanently gone. Every later design decision rested on inference about
# it. This script exists so that cannot happen twice.
#
# THE RULE IT ENFORCES: capture ALL THREE ranks, to disk, on an automatic
# trigger, before anything is restarted. Never on manual timing -- a human
# noticing a wedge is already too late, and a wedge that resolves itself
# leaves nothing behind.
#
# Triggers (either fires a capture):
#   liveness  a bounded /health_generate probe fails or times out
#   abandon   the presence-deadline abandonment appears in the serving log
#             (PHASE-FLIP-PRESENCE ... FLIP ABANDONED)
#
# Usage:  bash scripts/route_a_631_wedge_capture.sh [reason-tag]
# Env:    PORT, LOG, EVIDENCE_ROOT, MAX_CAPTURES, PROBE_INTERVAL_S
set -uo pipefail

PORT="${PORT:-30030}"
LOG="${LOG:-/spinning/serving-30030.boot.log}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-/spinning/evidence-631}"
PY_SPY="${PY_SPY:-/spinning/htsglang-gpu/.venv/bin/py-spy}"
MAX_CAPTURES="${MAX_CAPTURES:-4}"
PROBE_INTERVAL_S="${PROBE_INTERVAL_S:-5}"
PROBE_TIMEOUT_S="${PROBE_TIMEOUT_S:-20}"
# Consecutive failed probes before a liveness capture. One timeout under a
# long prefill is normal; three in a row is not.
LIVENESS_STRIKES="${LIVENESS_STRIKES:-3}"

mkdir -p "$EVIDENCE_ROOT"

scheduler_pids() {
    # The scheduler processes retitle themselves sglang::scheduler_PPn.
    pgrep -f 'sglang::scheduler_PP' 2>/dev/null | sort -n
}

capture() {
    local reason="$1"
    local stamp
    stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
    local dir="$EVIDENCE_ROOT/wedge_${stamp}_${reason}"
    mkdir -p "$dir"
    echo "[$(date -u '+%H:%M:%SZ')] CAPTURE reason=$reason -> $dir"

    {
        printf 'reason=%s\n' "$reason"
        printf 'utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        printf 'port=%s\n' "$PORT"
        printf 'tree_commit=%s\n' \
            "$(git -C /spinning/wt-631-routea rev-parse --short HEAD 2>/dev/null)"
    } > "$dir/META.txt"

    # -- the stacks. ALL ranks, each to its own file, failures recorded ----
    # A missing rank is the failure mode this whole script exists to
    # prevent, so an unreachable PID leaves a file saying so rather than
    # silently nothing.
    local pids
    pids="$(scheduler_pids)"
    printf 'scheduler_pids=%s\n' "$(echo "$pids" | tr '\n' ' ')" >> "$dir/META.txt"
    if [ -z "$pids" ]; then
        echo "NO SCHEDULER PIDS FOUND -- the processes are gone" \
            > "$dir/STACKS_MISSING.txt"
    fi
    local idx=0
    for pid in $pids; do
        # Plain Python stacks, all threads. This is the primary artifact.
        timeout 60 "$PY_SPY" dump --pid "$pid" \
            > "$dir/rank${idx}.pid${pid}.py.txt" 2> "$dir/rank${idx}.py.err" \
            || echo "py-spy dump FAILED for pid $pid" >> "$dir/rank${idx}.py.err"
        # Locals: this is where send_req_work depth and the flip's pending
        # state are visible without instrumenting the server.
        timeout 90 "$PY_SPY" dump --pid "$pid" --locals \
            > "$dir/rank${idx}.pid${pid}.locals.txt" 2> "$dir/rank${idx}.locals.err" \
            || echo "py-spy --locals FAILED for pid $pid" >> "$dir/rank${idx}.locals.err"
        # Native frames: "what does :1109 actually wait ON" may live below
        # Python, inside gloo. Allowed to fail without losing the rest.
        timeout 120 "$PY_SPY" dump --pid "$pid" --native \
            > "$dir/rank${idx}.pid${pid}.native.txt" 2> "$dir/rank${idx}.native.err" \
            || echo "py-spy --native FAILED for pid $pid" >> "$dir/rank${idx}.native.err"
        idx=$((idx + 1))
    done

    # -- the rendezvous state ---------------------------------------------
    # Which ranks had announced, for which epoch. The gate's own evidence.
    ls -la /dev/shm/sglang-phase-flip-presence/ > "$dir/presence_markers.txt" 2>&1
    for f in /dev/shm/sglang-phase-flip-presence/*; do
        [ -f "$f" ] && printf '== %s ==\n%s\n' "$f" "$(cat "$f")" \
            >> "$dir/presence_marker_bodies.txt"
    done 2>/dev/null

    # -- surroundings ------------------------------------------------------
    nvidia-smi > "$dir/nvidia-smi.txt" 2>&1
    ps -eo pid,pgid,stat,etimes,cmd | grep -E 'sglang|launch_server' \
        | grep -v grep > "$dir/processes.txt" 2>&1
    # Bounded slice of the log: enough for the flip timeline, small enough
    # to read. The full log is rotated by the boot script, not copied here.
    tail -n 4000 "$LOG" > "$dir/serving_tail.log" 2>&1
    grep -nE 'PHASE-FLIP|FLIP ABANDONED|phase flip|DESYNC|cutover' "$LOG" \
        2>/dev/null | tail -n 400 > "$dir/phase_flip_timeline.txt"

    echo "[$(date -u '+%H:%M:%SZ')] capture complete: $dir"
    ls -la "$dir" | tail -n +2 | wc -l | xargs echo "  files:"
}

# One-shot mode: capture immediately and exit (used to PROVE the capture
# path works against a healthy server before it is needed for real).
if [ "${1:-}" = "--once" ]; then
    capture "${2:-manual}"
    exit 0
fi

echo "watching port $PORT (log $LOG); evidence -> $EVIDENCE_ROOT"
captures=0
strikes=0
log_marker=0
[ -f "$LOG" ] && log_marker="$(wc -l < "$LOG" 2>/dev/null || echo 0)"

while [ "$captures" -lt "$MAX_CAPTURES" ]; do
    sleep "$PROBE_INTERVAL_S"

    # -- trigger 1: the abandonment signature, new lines only -------------
    if [ -f "$LOG" ]; then
        now_lines="$(wc -l < "$LOG" 2>/dev/null || echo 0)"
        if [ "$now_lines" -gt "$log_marker" ]; then
            if sed -n "$((log_marker + 1)),${now_lines}p" "$LOG" 2>/dev/null \
                | grep -qE 'FLIP ABANDONED|no quorum'; then
                capture "abandon"
                captures=$((captures + 1))
            fi
            log_marker="$now_lines"
        fi
    fi

    # -- trigger 2: liveness ----------------------------------------------
    code="$(curl -s -o /dev/null -w '%{http_code}' -m "$PROBE_TIMEOUT_S" \
        "http://127.0.0.1:${PORT}/health_generate" 2>/dev/null || echo 000)"
    if [ "$code" != "200" ]; then
        strikes=$((strikes + 1))
        echo "[$(date -u '+%H:%M:%SZ')] liveness probe $code (strike $strikes/$LIVENESS_STRIKES)"
        if [ "$strikes" -ge "$LIVENESS_STRIKES" ]; then
            capture "liveness"
            captures=$((captures + 1))
            strikes=0
        fi
    else
        strikes=0
    fi
done

echo "capture budget exhausted ($MAX_CAPTURES); watcher exiting"
