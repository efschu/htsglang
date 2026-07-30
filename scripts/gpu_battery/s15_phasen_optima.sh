#!/usr/bin/env bash
# S15 -- #296: the two STATIC phase optima of the weight/KV split, measured as
# the extrema of the dynamic ladder (#274/#287) rather than as a proposal.
#
# Why now. The register entry "phase-dual MLP split" was closed once because
# the predicted gain was small against the collective floor of the day. BAR1
# lowered that floor (#293: 1.170x at s=8), and the design note says the gain
# of a phase-dual split scales INVERSELY with the floor -- a smaller floor
# leaves more of the compute imbalance visible. This run reopens the entry with
# numbers, and it delivers the n>=10 decode sample #294 asked for on the side.
#
# WHAT THE ARMS ARE. All arms share ONE base plan (--rank-tp-ratio
# auto-performance, which on this rig resolves to the plain VRAM-auto split
# 28107,16280,16280 -- every MLP concentration candidate is rejected by the
# decode-knee guard). They differ in exactly two pinned vectors:
#
#   * --rank-mlp-ratio      the WEIGHT split of the dense-MLP family
#   * --rank-kv-ratio       the DCP TOKEN split
#
# An explicit --rank-mlp-ratio takes the pin path in uneven_perf (probe and
# optimizer skipped), so the arms below are not competing with the optimizer;
# they ARE the optimizer's own rejected candidate list, measured.
#
#   anchor           auto split (MLP units 63,37,36) + KV 7,3,3
#   prefill_opt      MLP 10,1,1 + KV 7,3,3
#   decode_opt       MLP 7,3,3  + KV 7,3,3
#   decode_opt_kv111 MLP 7,3,3  + KV 1,1,1
#   prefill_opt_nccl prefill_opt over NCCL instead of BAR1 (s=8 only)
#   decode_opt_nccl  decode_opt over NCCL instead of BAR1 (s=8 only)
#
# WHERE THE TWO VECTORS COME FROM, and why they are not guesses:
#
#   prefill: proportional to MEASURED compute in the checkpoint's ACTUAL quant
#   format. The card probe (#213) times both lanes: 5090 566.88 FP8 TFLOPS
#   against 3080 65.57 bf16 TFLOPS with "compute capability 8.6 has no fp8
#   tensor path (needs 8.9+)" -- a true ratio of 8.64, not the 3.71 a bf16-only
#   probe sees. 8.64:1:1 lands on the 136-unit MLP grid at ~113,11,12, which is
#   the optimizer's own top candidate 10,1,1.
#
#   decode: proportional to MEASURED memory bandwidth, 1664 vs 718 GB/s = 7:3:3
#   -- the same vector the planner computes and calls KV-SPEED WEIGHTS.
#
# KV 7,3,3 is held FIXED across anchor/prefill_opt/decode_opt so the weight
# split is the only moving part; decode_opt_kv111 then changes only the KV
# rule, and the difference against decode_opt is the KV placement effect alone.
#
# The explicit 7,3,3 is used rather than the mode string `speed` on purpose:
# an explicit --rank-mlp-ratio takes the pin path and returns before the block
# that derives the speed weights, so the mode string and the pin would not be
# available on the same arm. 7,3,3 IS what `speed` resolves to on this rig
# (logged by the planner), so the arms stay comparable and the mechanism is
# identical on all four.

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh
source ./battery_host.sh
source ./_bar1_host_boot.sh

DIR="${BATTERY_STEP_DIR:?BATTERY_STEP_DIR missing}"
PORT="${BAR1_PORT}"
POINT_S="${S15_POINT_SECONDS:-12}"
WARMUP_S="${S15_WARMUP_SECONDS:-6}"
PROMPT_TOKENS="${S15_PROMPT_TOKENS:-2048}"
DECODE_BATCHES="${S15_DECODE_BATCHES:-1,8}"

# The pinned reserve of the bar1_hi recipe (#293's best arm). Kept on every
# arm including the NCCL ones, so the reserve never becomes a hidden variable.
RESERVE="--rank-auto-reserve-mib 4500,4200,4200 --decode-log-interval 1"

# WHY --decode-log-interval 1 IS PART OF THE SHARED RECIPE. The decode measure
# is a per-TICK number, and a tick is one scheduler log line. At the default
# interval of 40 steps this rig produces 2-3 ticks per decode point -- the
# first attempt of this step measured exactly that (bs=1 n=1, bs=8 n=3), which
# is not a sample. The point cannot simply run longer: it ends when the 256
# requested tokens are generated (2.8 s at bs=1, 5.0 s at bs=8), not at the
# time box, so the tick COUNT is set by the log interval and nothing else. At
# interval 1 the same points yield ~42 and ~31 ticks, over the n>=10 the window
# asks for. The cost is one formatted log line per decode step, ~0.7 % of a
# 7 ms round, and it is carried IDENTICALLY by every arm including the NCCL
# ones -- so it can shift the absolute level slightly against earlier runs, but
# never the comparison between the arms of this table.

# name | transport | extra env | extra server args | points (sessions:decode)
ARM_TABLE=(
    "anchor|bar1||$RESERVE --rank-kv-ratio 7,3,3|1:0 8:1"
    "prefill_opt|bar1||$RESERVE --rank-mlp-ratio 10,1,1 --rank-kv-ratio 7,3,3|1:0 8:1"
    "decode_opt|bar1||$RESERVE --rank-mlp-ratio 7,3,3 --rank-kv-ratio 7,3,3|1:0 8:1"
    "decode_opt_kv111|bar1||$RESERVE --rank-mlp-ratio 7,3,3 --rank-kv-ratio 1,1,1|1:0 8:1"
    "prefill_opt_nccl|grundlinie||$RESERVE --rank-mlp-ratio 10,1,1 --rank-kv-ratio 7,3,3|8:1"
    "decode_opt_nccl|grundlinie||$RESERVE --rank-mlp-ratio 7,3,3 --rank-kv-ratio 7,3,3|8:1"
)
S15_ONLY="${S15_ONLY:-}"

# Hard wall on card time. The budget starts when the locks are taken, and an
# arm is only STARTED while enough of it is left; a started arm always runs to
# its end so no half-arm lands in the table.
S15_BUDGET_S="${S15_BUDGET_S:-2400}"
S15_ARM_COST_S="${S15_ARM_COST_S:-300}"

mkdir -p "$DIR/proofs" "$DIR/logs" "$DIR/wait" "$DIR/power"
DIR_HOST="$(host_path "$DIR")" || exit 2
DRIVER_HOST="$(host_path "$BATTERY_DIR/s12_prefill_kurve.py")" || exit 2
ANALYSE_HOST="$(host_path "$BATTERY_DIR/s12_log_analyse.py")" || exit 2

if ! host_reachable; then
    echo "STOP: host $BAR1_HOST unreachable"; exit 2
fi
if ! bar1_require_integration; then exit 2; fi
if ! host_locks_acquire "${BATTERY_STEP:-s15}"; then
    echo "STOP: host locks not obtainable"
    echo "host locks held by someone else -- not broken" > "$DIR/blocked.txt"
    exit 2
fi
T_LOCK=$(date +%s)
echo "lock acquired at $(date -Is), budget ${S15_BUDGET_S}s" | tee "$DIR/budget.txt"

SERVER_PID=""
HOSTPID=""
POWER_PID=""

# --- power logging ----------------------------------------------------------
# 1 s samples with the power state column, host-side, one file per arm. Started
# after READY and stopped before the kill, so the window is the measurement and
# not the boot. Own pid, remembered, killed by pid only.
s15_power_start() {  # $1 = arm
    # Two statements, not one `local a=.. b=$a`: under `set -u` bash declares
    # both names before assigning, so the second word expands an unset `arm`.
    local arm="$1"
    local remote_csv="$BAR1_HOST_LOGDIR/s15.$arm.power.csv"
    POWER_PID="$(host_ssh_for 60 "setsid nvidia-smi \
        --query-gpu=timestamp,index,power.draw,pstate,clocks.sm,utilization.gpu,memory.used \
        --format=csv,noheader,nounits -l 1 > $remote_csv 2>/dev/null & echo \$!" 2>/dev/null)"
    POWER_PID="${POWER_PID//[^0-9]/}"
    printf '%s\n' "${POWER_PID:-none}" >> "$DIR/power/pids.txt"
}

s15_power_stop() {  # $1 = arm
    local arm="$1"
    local remote_csv="$BAR1_HOST_LOGDIR/s15.$arm.power.csv"
    if [ -n "$POWER_PID" ]; then
        host_ssh_for 60 "kill $POWER_PID 2>/dev/null; true" >/dev/null 2>&1
        POWER_PID=""
    fi
    host_ssh_for 120 "cat $remote_csv 2>/dev/null" > "$DIR/power/${arm}.csv" 2>/dev/null
}

s15_wait_for_server() {  # $1 = port, $2 = host pid, $3 = budget_s
    local port="$1" pid="$2" budget="${3:-600}" t0
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$budget" ]; do
        if host_ssh_for 40 "curl -sf -m 5 http://127.0.0.1:$port/health >/dev/null" \
            >/dev/null 2>&1; then
            echo "host server up after $(( $(date +%s) - t0 ))s"
            return 0
        fi
        if ! host_ssh_for 40 "kill -0 $pid 2>/dev/null" >/dev/null 2>&1; then
            echo "host server $pid died before answering" >&2
            return 2
        fi
        sleep 10
    done
    echo "host server not up within ${budget}s" >&2
    return 1
}

cleanup() {
    [ -n "$POWER_PID" ] && host_ssh_for 60 "kill $POWER_PID 2>/dev/null; true" >/dev/null 2>&1
    bar1_kill_host_server "$SERVER_PID" "${HOSTPID:-}" \
        "$DIR/pyspy-host-cleanup.txt" || true
    SERVER_PID=""
    host_locks_release
}
trap cleanup EXIT INT TERM

cat > "$DIR/remote_measure.sh" <<EOF
#!/usr/bin/env bash
# GENERATED by s15_phasen_optima.sh. Runs on the PVE host.
set -uo pipefail
/spinning/miniforge3_local_install/bin/python3.12 $DRIVER_HOST \\
  --mode messen --port $PORT --out-dir $DIR_HOST \\
  --point-seconds $POINT_S --warmup-seconds $WARMUP_S \\
  --prompt-tokens $PROMPT_TOKENS --decode-batches $DECODE_BATCHES \\
  --arm "\$1" --sessions "\$2" --folge "\$3" --server-log "\$4" \\
  --with-decode "\$5"
EOF
chmod +x "$DIR/remote_measure.sh"

SEQ=0
ABORT=""
NOT_MEASURED=""

for ROW in "${ARM_TABLE[@]}"; do
    NAME="${ROW%%|*}";       REST="${ROW#*|}"
    TARM="${REST%%|*}";      REST="${REST#*|}"
    EENV="${REST%%|*}";      REST="${REST#*|}"
    EARGS="${REST%%|*}";     POINTS="${REST#*|}"

    if [ -n "$S15_ONLY" ]; then
        case " $S15_ONLY " in *" $NAME "*) ;; *) continue ;; esac
    fi

    SPENT=$(( $(date +%s) - T_LOCK ))
    LEFT=$(( S15_BUDGET_S - SPENT ))
    if [ "$LEFT" -lt "$S15_ARM_COST_S" ]; then
        echo "BUDGET: ${SPENT}s spent, ${LEFT}s left -- $NAME NOT MEASURED"
        NOT_MEASURED="$NOT_MEASURED $NAME"
        printf '%s not measured (budget: %ss spent, %ss left)\n' \
            "$NAME" "$SPENT" "$LEFT" >> "$DIR/not_measured.txt"
        continue
    fi

    ARM="$NAME"
    SEQ=$((SEQ + 1))
    HOSTLOG="$BAR1_HOST_LOGDIR/s15.$ARM.log"
    HOSTPID="$BAR1_HOST_LOGDIR/s15.$ARM.pid"
    echo "== [$SEQ] arm $NAME (transport $TARM), ${SPENT}s of ${S15_BUDGET_S}s spent =="

    if ! bar1_altlast_pruefen "$PORT" "$DIR/blocked.txt"; then
        ABORT="leftovers before $ARM"; break
    fi

    BAR1_EXTRA_ENV="$EENV" BAR1_EXTRA_ARGS="$EARGS" \
        bar1_write_boot_script "$DIR/remote_boot_${ARM}.sh" "$TARM" \
            "$HOSTLOG" "$HOSTPID" "$PORT" \
        || { ABORT="boot script $ARM"; break; }

    SERVER_PID="$(bar1_boot_start "$DIR/remote_boot_${ARM}.sh" "$HOSTPID")" \
        || SERVER_PID=""
    if ! bar1_pid_ok "$SERVER_PID"; then
        SERVER_PID=""
        host_tail_into "$HOSTLOG" "$DIR/logs/${ARM}.tail.txt" 200
        echo "no pid" > "$DIR/logs/${ARM}.booterror.txt"
        echo "  ARM SKIPPED: server start without pid"
        continue
    fi
    echo "$SERVER_PID" >> "$DIR/host_pids"

    if ! s15_wait_for_server "$PORT" "$SERVER_PID" 480; then
        host_grep_into "$HOSTLOG" "$DIR/logs/${ARM}.booterror.txt" \
            "Bar1Unavailable" "ColdBuildWindowError" "CUDA out of memory" \
            "torch.OutOfMemoryError" "NCCL error" "Capture cuda graph failed" \
            "UNBOOTABLE" "Received sigquit"
        host_tail_into "$HOSTLOG" "$DIR/logs/${ARM}.tail.txt" 200
        bar1_kill_host_server "$SERVER_PID" "$HOSTPID" \
            "$DIR/pyspy-host-$SERVER_PID.txt" || true
        SERVER_PID=""
        echo "  ARM SKIPPED: server not up (reason in logs/${ARM}.booterror.txt)"
        sleep 15
        continue
    fi

    # The plan the boot ACTUALLY ran with, and the KV capacity it sized: an arm
    # whose pinned vector was silently not applied is not the arm in the table.
    host_grep_into "$HOSTLOG" "$DIR/proofs/${ARM}.txt" \
        "MLP vector PINNED" \
        "materialized MLP units" \
        "Uneven-DCP token sizing" \
        "max_total_num_tokens" \
        "KV Cache is allocated" \
        "HTCCL enabled for group" \
        "ACHIEVED=" \
        "Disable prefill CUDA graph"

    s15_power_start "$ARM"
    MRC=0
    for P in $POINTS; do
        N="${P%%:*}"
        WD="${P##*:}"
        echo "   point sessions=$N decode=$WD"
        host_run_script 900 "$DIR/remote_measure.sh" "$ARM" "$N" "$SEQ" \
            "$HOSTLOG" "$WD" >> "$DIR/measure.log" 2>&1
        MRC=$?
        [ "$MRC" != 0 ] && break
    done
    s15_power_stop "$ARM"
    echo "  measurement rc=$MRC"

    host_grep_into "$HOSTLOG" "$DIR/logs/${ARM}.fatal.txt" \
        "CUDA out of memory" "torch.OutOfMemoryError" "NCCL error" \
        "Traceback (most recent call last)"

    host_ssh_for 300 "/spinning/miniforge3_local_install/bin/python3.12 \
        $ANALYSE_HOST --log '$ARM:8:$HOSTLOG' --punkte $DIR_HOST/punkte.jsonl \
        --json $DIR_HOST/wait/${ARM}.json" > "$DIR/wait/${ARM}.txt" 2>&1 || true

    host_tail_into "$HOSTLOG" "$DIR/logs/${ARM}.tail.txt" 120

    bar1_kill_host_server "$SERVER_PID" "$HOSTPID" \
        "$DIR/pyspy-host-$SERVER_PID.txt" || true
    SERVER_PID=""
    sleep 15

    if [ "$MRC" != 0 ]; then
        printf 'measurement rc=%s\n' "$MRC" > "$DIR/logs/${ARM}.measureerror.txt"
        echo "  POINT MISSING: measurement rc=$MRC"
    fi
done

cleanup
trap - EXIT INT TERM

printf 'card time used: %ss of %ss\n' "$(( $(date +%s) - T_LOCK ))" "$S15_BUDGET_S" \
    | tee -a "$DIR/budget.txt"
[ -n "$NOT_MEASURED" ] && echo "not measured:$NOT_MEASURED" | tee -a "$DIR/budget.txt"

if [ -n "$ABORT" ]; then
    echo "aborted: $ABORT" | tee "$DIR/abort.txt"
    exit 1
fi
exit 0
