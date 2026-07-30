#!/usr/bin/env bash
# S16 -- task #285: DFLASH against NEXTN on STRUCTURED output, interleaved.
#
# THE CLAIM UNDER TEST. DFLASH is documented in this fork as weak on prose and
# strong on format-constrained text; the crossover suite measured +6 % at short
# context, parity long, and 18.9-20.9 % BEHIND NEXTN in the multiturn regime.
# Every one of those numbers came from prose or from multiturn chat. The claim
# that DFLASH earns its place on code / JSON / tables has never been measured
# on code, JSON or tables. This step measures exactly that, and nothing else.
#
# WHAT IS HELD FIXED, and why each one had to be:
#
#   * THE VEHICLE IS FP8, NOT GGUF. Qwen3.6-27B-FP8 as target, the bf16
#     qwen3.6-27b-dflash checkpoint as drafter. #290 contaminates the GGUF
#     path, so a GGUF vehicle would put a known defect inside the difference
#     this step is trying to read.
#   * THE TWO ARMS DIFFER IN THE SPECULATIVE BLOCK AND IN NOTHING ELSE. Same
#     target, same tp/ratio/reserve, same context, same max-running-requests,
#     same decode log interval, same transport. The boot script is GENERATED
#     from one template here rather than pasted twice, which is the only way
#     that stays true after the third edit.
#   * THE KV POOL CAN BE PINNED. The DFLASH drafter costs weights the NEXTN
#     head does not, so at equal reserve the two arms end up with different KV
#     capacities -- and s14 already paid for the lesson that a comparison with
#     a different pool on each side is not a comparison. Set
#     S16_MAX_TOTAL_TOKENS to the DFLASH arm's own capacity (read
#     `max_total_num_tokens=` out of proofs/dflash_r1.txt after a first
#     calibration round, or out of a single throwaway boot) and both arms run
#     the same pool. Left empty the step still runs, and the analysis prints
#     the two capacities side by side so the reader can see the asymmetry it
#     is then living with.
#   * TEMPERATURE 0 EVERYWHERE, one pinned prompt order per class.
#
# INTERLEAVED, at the granularity a boot flag allows. The speculative algorithm
# is chosen at server start, so "interleaved in the same run" cannot mean
# request-by-request; it means ROUNDS, NOT BLOCKS, exactly as s13/s14 use the
# word: every round walks the arm list in the same order, so a warming card, a
# drifting clock or a background process hits both arms alike, and two rounds
# of one arm are that arm's between-boot spread. Measuring all of NEXTN and
# then all of DFLASH is the one methodological error that cannot be repaired
# afterwards.
#
# (The cross-algo ladder of #156 CAN hold both drafters in one server, but it
# carries a standing 13-15 % tax for the extra hidden-state captures. Using it
# here would put that tax inside the number this step reports.)
#
# THE FLOOR COMES FIRST. Round 0 boots the NEXTN recipe TWICE under two names
# (floor_a, floor_b) and measures both. Their spread per (bs, class) cell IS
# the noise floor of this instrument, and every DFLASH-vs-NEXTN difference
# smaller than it is reported as `~` and is not a finding. A floor measured
# after the fact, or borrowed from another window, would be a floor for another
# instrument.
#
# THE CONTENT AXIS IS A COLUMN. One point is one (arm, bs, class) triple and
# the classes are never averaged together. s16_structured_point.py carries the
# reasoning for the point itself, including the output validation that decides
# whether a point counts at all.
#
# HOW TO RUN IT (the whole invocation, not a sketch):
#
#   export BATTERY_RUN=/spinning/gpu-battery-results/$(date +%F)_dflash_structured
#   export BATTERY_STEP=s16_dflash_structured
#   export BATTERY_STEP_DIR=$BATTERY_RUN/$BATTERY_STEP
#   export WT=/spinning/wt-dflash-structured
#   mkdir -p "$BATTERY_STEP_DIR"
#   bash /spinning/wt-dflash-structured/scripts/gpu_battery/s16_dflash_structured.sh
#
#   # the table:
#   /spinning/htsglang-gpu/.venv/bin/python \
#     /spinning/wt-dflash-structured/scripts/gpu_battery/s16_analysis.py \
#     --step-dir "$BATTERY_STEP_DIR" --json "$BATTERY_STEP_DIR/summary.json"
#
# CALIBRATION FIRST, one boot, before the real run:
#
#   S16_ONLY=dflash S16_FLOOR=0 S16_ROUNDS=1 S16_BS=1 S16_CLASSES=code_completion \
#     bash .../s16_dflash_structured.sh
#   grep max_total_num_tokens "$BATTERY_STEP_DIR/proofs/dflash_r1.txt"
#
# then repeat the full run with S16_MAX_TOTAL_TOKENS set to that number, so both
# arms hold the same KV pool. Budget for the full run: 8 arms (2 floor + 2
# rounds x 2) x (2 batch sizes x 3 classes x ~40 s + boot) -- S16_BUDGET_S
# defaults to 5400 s and arms that no longer fit are named in not_measured.txt
# rather than half-measured.

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh
source ./battery_host.sh
# Sourced for the generic host-process helpers only (bar1_pid_ok,
# bar1_boot_start, bar1_altlast_pruefen, bar1_kill_host_server). The BAR1 boot
# template itself is NOT used: it hardcodes the NEXTN speculative block, and an
# appended second --speculative-algorithm would leave the NEXTN-only step and
# topk arguments standing on the DFLASH arm.
source ./_bar1_host_boot.sh

DIR="${BATTERY_STEP_DIR:?BATTERY_STEP_DIR missing}"
PORT="${S16_PORT:-${BAR1_PORT:-30000}}"

PROMPT_FILE="${S16_PROMPT_FILE:-$BATTERY_DIR/prompts/structured_v1.json}"
CLASSES="${S16_CLASSES:-code_completion json_schema list_table}"
BS_LIST="${S16_BS:-1 8}"
WARMUP_S="${S16_WARMUP_SECONDS:-8}"
RAMP_S="${S16_RAMP_SECONDS:-6}"
WINDOW_S="${S16_WINDOW_SECONDS:-20}"
DRAIN_S="${S16_DRAIN_SECONDS:-5}"
MIN_VALID="${S16_MIN_VALID_RATIO:-0.75}"

TARGET="${S16_TARGET:-$MODEL_ROOT/Qwen3.6-27B-FP8}"
DRAFT="${S16_DRAFT:-$MODEL_ROOT/qwen3.6-27b-dflash}"
CTX="${S16_CONTEXT_LENGTH:-32768}"
MAX_RUNNING="${S16_MAX_RUNNING_REQUESTS:-16}"
# One reserve for BOTH arms, sized for the arm that needs more (the DFLASH
# drafter's sharded weights sit on top of what the NEXTN head costs). The NEXTN
# arm then simply leaves that memory unused, which is the cheap direction of
# the asymmetry: it costs the NEXTN arm KV capacity it does not need, it never
# gives it an advantage.
RESERVE="${S16_RESERVE:-4500,4200,4200}"
# Empty = not pinned. See the header.
MAX_TOTAL_TOKENS="${S16_MAX_TOTAL_TOKENS:-}"
# The tick is the sample and the log interval sets how many ticks a window
# produces. Identical on every arm, so it cannot move the comparison.
LOG_INTERVAL="${S16_LOG_INTERVAL:-1}"

# The speculative block per algorithm, and it is the ONLY thing the arms differ
# in. NEXTN: the 3-step chain of the reference recipe. DFLASH: block size 16,
# the drafter's own verify window, split placement (the default) so the
# comparison is about the ALGORITHM and not about where its drafter lives --
# solo placement is a second, separate question (#153/#155).
SPEC_NEXTN="${S16_SPEC_NEXTN:---speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4}"
SPEC_DFLASH="${S16_SPEC_DFLASH:---speculative-algorithm DFLASH --speculative-draft-model-path $DRAFT --speculative-dflash-block-size 16 --speculative-draft-placement split}"

# name | algorithm | speculative block
ARM_TABLE=(
    "nextn|NEXTN|$SPEC_NEXTN"
    "dflash|DFLASH|$SPEC_DFLASH"
)
# The A-vs-A floor round: the SAME recipe twice, under two names.
FLOOR_TABLE=(
    "floor_a|NEXTN|$SPEC_NEXTN"
    "floor_b|NEXTN|$SPEC_NEXTN"
)

ROUNDS="${S16_ROUNDS:-2}"
ROUND_START="${S16_ROUND_START:-1}"
DO_FLOOR="${S16_FLOOR:-1}"
S16_ONLY="${S16_ONLY:-}"

# Hard wall on card time, same rule as s15: the budget starts when the locks
# are taken, an arm is only STARTED while enough of it is left, and a started
# arm always runs to its end so no half-arm lands in the table.
BUDGET_S="${S16_BUDGET_S:-5400}"
ARM_COST_S="${S16_ARM_COST_S:-900}"

mkdir -p "$DIR/proofs" "$DIR/logs" "$DIR/samples"
DIR_HOST="$(host_path "$DIR")" || exit 2
DRIVER_HOST="$(host_path "$BATTERY_DIR/s16_structured_point.py")" || exit 2
PROMPTS_HOST="$(host_path "$PROMPT_FILE")" || exit 2
SAMPLES_HOST="$(host_path "$DIR/samples")" || exit 2

if [ ! -f "$PROMPT_FILE" ]; then
    echo "STOP: prompt set $PROMPT_FILE missing" >&2
    exit 2
fi
if ! host_reachable; then
    echo "STOP: host $BAR1_HOST unreachable"; exit 2
fi
if ! host_locks_acquire "${BATTERY_STEP:-s16}"; then
    echo "STOP: host locks not obtainable"
    echo "host locks held by someone else -- not broken" > "$DIR/blocked.txt"
    exit 2
fi
T_LOCK=$(date +%s)
echo "lock acquired at $(date -Is), budget ${BUDGET_S}s" | tee "$DIR/budget.txt"

SERVER_PID=""
HOSTPID=""

# s16_write_boot_script <container path> <spec block> <host log> <host pidfile> <port>
#
# One template, and the speculative block is the only interpolated part. The
# environment is the reference recipe of _bar1_host_boot.sh minus the three
# SGLANG_HTCCL* lines: the transport is plain NCCL on every arm here, because
# the question is which DRAFTER is better on structured text and a second
# moving part would have to be defended in every cell of the table.
#
# CUDA_HOME is mandatory (the JIT build fails on ninja without it) and
# CUDA_DEVICE_ORDER is deliberately NOT set: cuda:0 is the 5090 in this recipe
# and the reserve vector is written for that order.
s16_write_boot_script() {
    local out="$1" spec="$2" hostlog="$3" hostpid="$4" port="$5"
    local hv hw hm hcache pin=""
    hv="$(host_path "$VENV")" || return 2
    hw="$(host_path "${S16_HOST_WT:-$WT}")" || return 2
    hm="$(host_path "$TARGET")" || return 2
    hcache="$(host_path "${BAR1_EXTCACHE:-/spinning/torch-ext-cache}")" || return 2
    [ -n "$MAX_TOTAL_TOKENS" ] && pin="--max-total-tokens $MAX_TOTAL_TOKENS"

    cat > "$out" <<EOF
#!/usr/bin/env bash
# GENERATED by s16_dflash_structured.sh. Runs on the PVE host.
set -uo pipefail
mkdir -p "$BAR1_HOST_LOGDIR"
rm -f "$hostpid"
cd $hw
PYTHONPATH=$hw/python:$hv/lib/python3.12/site-packages \\
LD_LIBRARY_PATH=$hv/lib/python3.12/site-packages/nvidia/cu13/lib \\
CUDA_HOME=$hv/lib/python3.12/site-packages/nvidia/cu13 \\
SGLANG_UNEVEN_DCP=1 SGLANG_UNEVEN_DCP_WEIGHTED=1 \\
SGLANG_MAMBA_SSM_DTYPE=bfloat16 FLASHINFER_DISABLE_VERSION_CHECK=1 \\
TORCH_EXTENSIONS_DIR=$hcache \\
TORCH_CUDA_ARCH_LIST="8.6;12.0" MAX_JOBS=4 \\
setsid /spinning/miniforge3_local_install/bin/python3.12 -m sglang.launch_server \\
  --model-path $hm \\
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \\
  --rank-auto-reserve-mib $RESERVE \\
  --kv-cache-dtype fp8_e4m3 --context-length $CTX --trust-remote-code \\
  --max-running-requests $MAX_RUNNING $pin \\
  --decode-log-interval $LOG_INTERVAL \\
  $spec \\
  --enable-metrics --host 127.0.0.1 --port $port \\
  > "$hostlog" 2>&1 &
echo \$! > "$hostpid"
echo "started, pid \$(cat "$hostpid")"
EOF
    chmod +x "$out"
    return 0
}

s16_wait_for_server() {  # $1 = port, $2 = host pid, $3 = budget_s
    local port="$1" pid="$2" budget="${3:-900}" t0
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
    bar1_kill_host_server "$SERVER_PID" "${HOSTPID:-}" \
        "$DIR/pyspy-host-cleanup.txt" || true
    SERVER_PID=""
    host_locks_release
}
trap cleanup EXIT INT TERM

cat > "$DIR/remote_measure.sh" <<EOF
#!/usr/bin/env bash
# GENERATED by s16_dflash_structured.sh. Runs on the PVE host.
set -uo pipefail
/spinning/miniforge3_local_install/bin/python3.12 $DRIVER_HOST \\
  --port $PORT --out-dir $DIR_HOST --prompt-file $PROMPTS_HOST \\
  --samples-dir $SAMPLES_HOST \\
  --warmup-seconds $WARMUP_S --ramp-seconds $RAMP_S \\
  --window-seconds $WINDOW_S --drain-seconds $DRAIN_S \\
  --min-valid-ratio $MIN_VALID \\
  --arm "\$1" --algo "\$2" --bs "\$3" --content-class "\$4" \\
  --seq "\$5" --server-log "\$6"
EOF
chmod +x "$DIR/remote_measure.sh"

SEQ=0
ABORT=""
NOT_MEASURED=""

# s16_run_arm <arm name> <algo> <spec block>
s16_run_arm() {
    local arm="$1" algo="$2" spec="$3"
    local hostlog="$BAR1_HOST_LOGDIR/s16.$arm.log"
    local mrc=0 bs cls
    HOSTPID="$BAR1_HOST_LOGDIR/s16.$arm.pid"

    if ! bar1_altlast_pruefen "$PORT" "$DIR/blocked.txt"; then
        ABORT="leftovers before $arm"
        return 2
    fi

    s16_write_boot_script "$DIR/remote_boot_${arm}.sh" "$spec" \
        "$hostlog" "$HOSTPID" "$PORT" \
        || { ABORT="boot script $arm"; return 2; }

    SERVER_PID="$(bar1_boot_start "$DIR/remote_boot_${arm}.sh" "$HOSTPID")" \
        || SERVER_PID=""
    if ! bar1_pid_ok "$SERVER_PID"; then
        SERVER_PID=""
        host_tail_into "$hostlog" "$DIR/logs/${arm}.tail.txt" 200
        echo "no pid" > "$DIR/logs/${arm}.booterror.txt"
        echo "  ARM SKIPPED: server start without pid"
        return 1
    fi
    echo "$SERVER_PID" >> "$DIR/host_pids"

    if ! s16_wait_for_server "$PORT" "$SERVER_PID" 900; then
        host_grep_into "$hostlog" "$DIR/logs/${arm}.booterror.txt" \
            "CUDA out of memory" "torch.OutOfMemoryError" "NCCL error" \
            "Capture cuda graph failed" "Received sigquit" \
            "Unknown speculative algorithm" "ValueError"
        host_tail_into "$hostlog" "$DIR/logs/${arm}.tail.txt" 200
        bar1_kill_host_server "$SERVER_PID" "$HOSTPID" \
            "$DIR/pyspy-host-$SERVER_PID.txt" || true
        SERVER_PID=""
        echo "  ARM SKIPPED: server not up (reason in logs/${arm}.booterror.txt)"
        sleep 20
        return 1
    fi

    # What the boot ACTUALLY ran with. An arm whose speculative block did not
    # take, or whose KV pool came out different from its partner's, is not the
    # arm in the table -- and the KV capacity is exactly the quantity the
    # optional --max-total-tokens pin exists to equalise.
    host_grep_into "$hostlog" "$DIR/proofs/${arm}.txt" \
        "speculative_algorithm" \
        "speculative_num_draft_tokens" \
        "speculative_draft_model_path" \
        "dflash" \
        "max_total_num_tokens" \
        "KV Cache is allocated" \
        "Capture cuda graph" \
        "Disable cuda graph"

    for bs in $BS_LIST; do
        for cls in $CLASSES; do
            SEQ=$((SEQ + 1))
            echo "   point bs=$bs class=$cls"
            host_run_script 1200 "$DIR/remote_measure.sh" \
                "$arm" "$algo" "$bs" "$cls" "$SEQ" "$hostlog" \
                >> "$DIR/measure.log" 2>&1
            local rc=$?
            [ "$rc" != 0 ] && mrc=$rc
        done
    done
    echo "  measurement rc=$mrc"

    host_grep_into "$hostlog" "$DIR/logs/${arm}.fatal.txt" \
        "CUDA out of memory" "torch.OutOfMemoryError" "NCCL error" \
        "Traceback (most recent call last)"
    host_tail_into "$hostlog" "$DIR/logs/${arm}.tail.txt" 120

    bar1_kill_host_server "$SERVER_PID" "$HOSTPID" \
        "$DIR/pyspy-host-$SERVER_PID.txt" || true
    SERVER_PID=""
    sleep 20

    if [ "$mrc" != 0 ]; then
        printf 'measurement rc=%s\n' "$mrc" > "$DIR/logs/${arm}.measureerror.txt"
        echo "  POINT MISSING or NOT COUNTED: measurement rc=$mrc"
    fi
    return 0
}

# s16_walk <round label> <arm table entries...>
s16_walk() {
    local label="$1"; shift
    local row name algo spec spent left
    for row in "$@"; do
        name="${row%%|*}";  local rest="${row#*|}"
        algo="${rest%%|*}"; spec="${rest#*|}"

        if [ -n "$S16_ONLY" ]; then
            case " $S16_ONLY " in *" $name "*) ;; *) continue ;; esac
        fi

        spent=$(( $(date +%s) - T_LOCK ))
        left=$(( BUDGET_S - spent ))
        if [ "$left" -lt "$ARM_COST_S" ]; then
            echo "BUDGET: ${spent}s spent, ${left}s left -- ${name}_${label} NOT MEASURED"
            NOT_MEASURED="$NOT_MEASURED ${name}_${label}"
            printf '%s not measured (budget: %ss spent, %ss left)\n' \
                "${name}_${label}" "$spent" "$left" >> "$DIR/not_measured.txt"
            continue
        fi

        echo "== arm ${name}_${label} (algo $algo), ${spent}s of ${BUDGET_S}s spent =="
        s16_run_arm "${name}_${label}" "$algo" "$spec"
        [ -n "$ABORT" ] && return 1
    done
    return 0
}

if [ "$DO_FLOOR" = "1" ]; then
    echo "== A-vs-A floor round: the NEXTN recipe twice, under two names =="
    s16_walk "r0" "${FLOOR_TABLE[@]}" || true
fi

if [ -z "$ABORT" ]; then
    for ROUND in $(seq "$ROUND_START" $((ROUND_START + ROUNDS - 1))); do
        echo "== comparison round $ROUND =="
        s16_walk "r$ROUND" "${ARM_TABLE[@]}" || break
    done
fi

cleanup
trap - EXIT INT TERM

printf 'card time used: %ss of %ss\n' "$(( $(date +%s) - T_LOCK ))" "$BUDGET_S" \
    | tee -a "$DIR/budget.txt"
[ -n "$NOT_MEASURED" ] && echo "not measured:$NOT_MEASURED" | tee -a "$DIR/budget.txt"

if [ -n "$ABORT" ]; then
    echo "aborted: $ABORT" | tee "$DIR/abort.txt"
    exit 1
fi
exit 0
