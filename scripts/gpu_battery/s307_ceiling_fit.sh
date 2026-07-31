#!/usr/bin/env bash
# #307 card proof: a --max-running-requests-ceiling the cards cannot hold is
# fitted to the budget instead of killing the boot.
#
# Three arms on the TP=3 uneven rig (5090 + 2x 3080), Qwen3.6-27B-FP8:
#
#   A  8/64   the constellation that died on 2026-07-30 (559 MiB over budget
#             on a 3080, before the first KV token). EXPECTATION: HEALTHY,
#             with the fit visible in the log -- "[auto-mamba] the concurrency
#             target does not fit" on at least one rank, and the scheduler's
#             "requested ceiling 64 ... fitted to N" with N < 64. Rank
#             uniformity is part of the verdict: one admission ceiling for the
#             whole group, not one per card.
#   B  raise  on arm A's server: the float starts at 8 and must climb ABOVE
#             its start under load, stop at the fitted ceiling, and keep
#             throttling BEFORE retraction (retract counter stays 0 while the
#             throttle counter moves).
#   C  4/16   the run that carried on 2026-07-30. EXPECTATION: unchanged --
#             100 mamba slots per rank and no fit line anywhere in the log.
#
# Runs from the container; the server runs on the PVE host. Not executed in
# the CPU-only implementation window -- this is the program for the next card
# session. Budget: ~25 min of card time.
set -uo pipefail
BATTERY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BATTERY_DIR"
source ./battery_common.sh
source ./battery_host.sh
source ./_bar1_host_boot.sh

DIR="${DIR:-/spinning/gpu-battery-results/s307_ceiling_fit}"
PORT="${PORT:-30047}"
mkdir -p "$DIR"

# The reserve vector of the FAILING run, not of the carrying one: the point is
# to reproduce the budget that was too small and show it now boots.
RESERVE="${RESERVE:-4500,4200,4200}"
COMMON_ARGS="--rank-auto-reserve-mib $RESERVE --decode-log-interval 1 \
--rank-mlp-ratio 63,37,36 --rank-kv-ratio 7,3,3 \
--admission-throttle-high 0.30 --admission-release-low 0.10"

# Markers the verdict script reads. Kept here so a rename in the emitter that
# is not carried into the consumer fails loudly on the next run instead of
# scoring 0 hits in silence.
M_FIT='[auto-mamba] the concurrency target does not fit'
M_SCHED='does not fit the memory budget'
M_ADMIT='Dynamic admission limit:'
M_POOL='[auto-mamba] demand-driven mamba pool'
M_LEDGER='leaves no GPU memory for the KV cache'
M_RETRACT='Retract requests'

rc_total=0

boot_arm() {  # $1 = arm tag, $2 = extra server args
    local tag="$1" args="$2"
    local hostlog="$BAR1_HOST_LOGDIR/s307_$tag.server.log"
    local hostpid="$BAR1_HOST_LOGDIR/s307_$tag.pid"
    echo "$hostlog" > "$DIR/$tag.hostlog"
    bar1_altlast_pruefen "$PORT" "$DIR/${tag}_blocked.txt" || return 2
    BAR1_EXTRA_ARGS="$COMMON_ARGS $args" \
        bar1_write_boot_script "$DIR/${tag}_boot.sh" bar1 "$hostlog" "$hostpid" "$PORT" \
        || return 2
    local pid
    pid="$(bar1_boot_start "$DIR/${tag}_boot.sh" "$hostpid")" || pid=""
    bar1_pid_ok "$pid" || { echo "$tag: boot did not start"; return 1; }
    echo "$pid" > "$DIR/$tag.pid"
    host_wait_for_server "$PORT" 900
}

harvest() {  # $1 = arm tag
    local tag="$1" hostlog
    hostlog="$(cat "$DIR/$tag.hostlog")"
    host_grep_into "$hostlog" "$DIR/${tag}_markers.txt" \
        "$M_FIT" "$M_SCHED" "$M_ADMIT" "$M_POOL" "$M_LEDGER" "$M_RETRACT"
    host_tail_into "$hostlog" "$DIR/${tag}_tail.txt" 120
}

verdict() {  # $1.. = args to the checker
    python3 "$BATTERY_DIR/s307_ceiling_fit.py" "$@" | tee -a "$DIR/verdict.txt"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc_total=1
}

# ---------------------------------------------------------------- arm A ----
echo "== A: 8/64 -- the constellation that died =="
if boot_arm a64 "--max-running-requests 8 --max-running-requests-ceiling 64"; then
    host_ssh_for 40 "curl -sf -m 15 http://127.0.0.1:$PORT/get_server_info" \
        > "$DIR/a64_server_info.json"
else
    echo "A: FAILED -- the fit did not save the boot"
    rc_total=1
fi
harvest a64
verdict arm_a "$DIR/a64_markers.txt" "$DIR/a64_server_info.json" 64

# ---------------------------------------------------------------- arm B ----
if [ -s "$DIR/a64_server_info.json" ]; then
    echo "== B: the float rises above its start of 8, stops at the fitted ceiling =="
    cat > "$DIR/b_raise.sh" <<EOF
#!/usr/bin/env bash
python3 "$(host_path "$BATTERY_DIR/s307_raise_probe.py")" $PORT
EOF
    host_run_script 1200 "$DIR/b_raise.sh" > "$DIR/b_raise.json" 2>"$DIR/b_raise.err"
    harvest a64
    verdict arm_b "$DIR/b_raise.json" "$DIR/a64_server_info.json" 64 \
        "$DIR/a64_markers.txt"
fi
bar1_kill_host_server "$(cat "$DIR/a64.pid" 2>/dev/null)" "" "$DIR/a64_pyspy.txt"

# ---------------------------------------------------------------- arm C ----
echo "== C: 4/16 -- the run that carried, must be unchanged =="
if boot_arm c16 "--max-running-requests 4 --max-running-requests-ceiling 16"; then
    host_ssh_for 40 "curl -sf -m 15 http://127.0.0.1:$PORT/get_server_info" \
        > "$DIR/c16_server_info.json"
else
    echo "C: FAILED"
    rc_total=1
fi
harvest c16
verdict arm_c "$DIR/c16_markers.txt" "$DIR/c16_server_info.json" 16
bar1_kill_host_server "$(cat "$DIR/c16.pid" 2>/dev/null)" "" "$DIR/c16_pyspy.txt"

echo "s307 exit $rc_total"
exit $rc_total
