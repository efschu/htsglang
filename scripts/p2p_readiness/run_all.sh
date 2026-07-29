#!/usr/bin/env bash
# P2P readiness re-probe: one call, a few minutes, no sglang boots.
#
#   bash run_all.sh [--results-dir DIR] [--baseline OLD_NCCL_JSON] [--dry-run]
#
# Order: capability_matrix -> d2d_bench -> nccl_transport_check.
# Results land in results/<YYYY-MM-DD_HHMM>/ next to this script.
#
# GPU arbitration (cross-session): /tmp/gpu-card-N.lock are DIRECTORIES with
# an info file inside; mkdir is the atomic acquire. We take the locks for ALL
# cards (the probes touch every pair), heartbeat them, and release on exit.
# A held lock is NEVER stolen -- stale or not, we report and abort; breaking
# another session's lock needs the operator (see gpu-arbitrierung notes).

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
DRY_RUN=0
BASELINE=""
RESULTS_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --baseline) BASELINE="$2"; shift ;;
        --results-dir) RESULTS_DIR="$2"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="$SCRIPT_DIR/results/$(date +%Y-%m-%d_%H%M)"
fi

# ---------------------------------------------------------------- locks ----
HELD_LOCKS=()
HEARTBEAT_PID=""

release_locks() {
    [ -n "$HEARTBEAT_PID" ] && kill "$HEARTBEAT_PID" 2>/dev/null
    for d in ${HELD_LOCKS[@]+"${HELD_LOCKS[@]}"}; do
        rm -rf "$d"
    done
    HELD_LOCKS=()
}
trap release_locks EXIT INT TERM

card_count() {
    if [ "$DRY_RUN" = 1 ]; then
        echo 0
        return
    fi
    nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0
}

acquire_locks() {
    local n="$1" i lock info
    for i in $(seq 0 $((n - 1))); do
        lock="/tmp/gpu-card-$i.lock"
        if mkdir "$lock" 2>/dev/null; then
            info="$lock/info"
            {
                echo "holder=p2p_readiness"
                echo "pid=$$"
                echo "acquired=$(date -Is)"
                echo "heartbeat=$(date -Is)"
            } > "$info"
            HELD_LOCKS+=("$lock")
        else
            echo "ABORT: $lock is held:" >&2
            cat "$lock/info" 2>/dev/null | sed 's/^/    /' >&2
            echo "not stealing a lock (stale or not) -- ask the operator." >&2
            release_locks
            exit 3
        fi
    done
    # heartbeat: refresh every 30 s while we run
    (
        while true; do
            for d in ${HELD_LOCKS[@]+"${HELD_LOCKS[@]}"}; do
                sed -i "s/^heartbeat=.*/heartbeat=$(date -Is)/" "$d/info" \
                    2>/dev/null
            done
            sleep 30
        done
    ) &
    HEARTBEAT_PID=$!
}

# ------------------------------------------------------------------ run ----
mkdir -p "$RESULTS_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

DRY_FLAG=""
[ "$DRY_RUN" = 1 ] && DRY_FLAG="--dry-run"

N_CARDS=$(card_count)
if [ "$DRY_RUN" = 0 ]; then
    if [ "$N_CARDS" -lt 2 ]; then
        echo "need >=2 GPUs, found $N_CARDS" >&2
        exit 4
    fi
    acquire_locks "$N_CARDS"
else
    echo "[dry-run] skipping lock acquisition and CUDA entirely"
fi

FAILURES=0
step() {
    echo "=== $1 ==="
    shift
    if ! (cd "$SCRIPT_DIR" && "$@") 2>&1 | tee -a "$RESULTS_DIR/run.log"; then
        echo "step failed (recorded, continuing)" | tee -a "$RESULTS_DIR/run.log"
        FAILURES=$((FAILURES + 1))
    fi
}

step "capability matrix" "$PY" capability_matrix.py \
    --out "$RESULTS_DIR/capability_matrix.json" $DRY_FLAG
step "d2d bench" "$PY" d2d_bench.py \
    --out "$RESULTS_DIR/d2d_bench.json" $DRY_FLAG
if [ -n "$BASELINE" ]; then
    step "nccl transport" "$PY" nccl_transport_check.py \
        --out "$RESULTS_DIR/nccl_transport.json" --baseline "$BASELINE" $DRY_FLAG
else
    step "nccl transport" "$PY" nccl_transport_check.py \
        --out "$RESULTS_DIR/nccl_transport.json" $DRY_FLAG
fi

cp "$SCRIPT_DIR/verdict_diff.md" "$RESULTS_DIR/verdict_diff.md" 2>/dev/null

echo
echo "results: $RESULTS_DIR"
echo "next: fill $RESULTS_DIR/verdict_diff.md (EFFECTIVE aperture values only)"
[ "$FAILURES" -gt 0 ] && echo "$FAILURES step(s) had failures -- see run.log"
exit 0
