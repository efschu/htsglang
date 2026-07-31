#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# GPU proof program for task #312 -- peer death must end a collective wait.
#
# THE CLAIM UNDER TEST
#   A rank killed with SIGKILL after warmup causes every surviving rank to
#   fail with a clean, NAMED error within 60 s, instead of spinning at 100 %
#   SM and zero PCIe until somebody intervenes by hand.
#
# The two arms:
#   kill  -- boot, warm up past the JIT window, SIGKILL one TP worker,
#            measure the wall time until a named PeerLostError reaches the
#            log on the surviving ranks and the process group exits.
#   ab    -- boot twice, with SGLANG_BARLINK_PEER_LIVENESS=1 and =0, and
#            compare ms/verify and ms/prefill on the SUCCESS path. The fix
#            costs nothing on a collective that completes, and this is the
#            measurement that has to show it.
#
# NOT RUN BY THE AUTHOR OF THE FIX. This script is the program, written to be
# executed by whoever holds the next GPU window. It needs all three cards.
#
# BEFORE YOU RUN THIS: the arms below are right, the environment is not.
# The 2026-07-31 window ran both arms and found four environment facts wrong
# for rig 1 -- each one enough to make the run measure nothing:
#   * VENV here is /spinning/shvllm/.venv, which is torch 2.14.0a0 with no
#     nvidia/cu13 tree. The GPU venv is /spinning/htsglang-gpu/.venv.
#   * SGLANG_BARLINK_GRAPH_ENABLE is not a variable this fork reads. The name is
#     SGLANG_BARLINK_GRAPH_ENABLE.
#   * SGLANG_BARLINK_BAR1_NV_SOURCE and CUDA_HOME are missing. bar1 needs the
#     patched driver headers; the JIT build fails on ninja without CUDA_HOME.
#   * it boots locally, but bar1 runs on the PVE host -- the container has
#     neither the driver source nor a usable NCCL for it.
# The canonical recipe is scripts/gpu_battery/_bar1_host_boot.sh. A driver
# that runs these two arms against it, plus a liveness=0 kill control, is
# /spinning/gpu-battery-results/2026-07-31_312_beleg/run_312.sh. See
# docs/dev/INTEGRATION_R3_VALIDATION.md, "#312-Beleg: Rang-Tod wird laut".
#
# The match patterns below are CORRECT and were confirmed on hardware: the log
# surface carries "peer rank gone: rank N (host, pid M)" and never the
# exception class name. Do not narrow them to class names -- that is exactly
# the false negative the 2026-07-31 window scored before it looked at the log.
#
#   scripts/probe/peer_kill_proof.sh kill
#   scripts/probe/peer_kill_proof.sh ab
#   scripts/probe/peer_kill_proof.sh cards      # NVML index -> name, only
#
# Card assignment is resolved at run time via NVML and never hardcoded:
# nvidia-smi / NVML enumeration order shifts between boots and driver states,
# and CUDA device order is a THIRD order again (see docs/rig-runbook.md 6.1).
set -uo pipefail

# No default arm on purpose. Every arm here boots a real server on all three
# cards; a bare invocation must print usage, not seize the rig.
ARM="${1:-}"

REPO_ROOT="${REPO_ROOT:-/spinning/htsglang}"
WT="${WT:-/spinning/wt-peer-liveness}"
VENV="${VENV:-/spinning/shvllm/.venv}"
MODEL_ROOT="${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}"
MODEL="${MODEL:-$MODEL_ROOT/Qwen3.6-27B-FP8}"
PORT="${PORT:-31712}"
# Logs never land inside the repo and never in an agent's context.
LOGDIR="${LOGDIR:-/tmp/peer-kill-proof}"
# How long the survivors may take. The criterion is 60 s; the harness waits
# longer so a MISS is measured rather than truncated.
CRITERION_S="${CRITERION_S:-60}"
WATCH_S="${WATCH_S:-180}"
BOOT_S="${BOOT_S:-900}"

mkdir -p "$LOGDIR"

# --- card inventory --------------------------------------------------------

cards() {
    "$VENV/bin/python" - <<'EOF'
import pynvml

pynvml.nvmlInit()
for i in range(pynvml.nvmlDeviceGetCount()):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(h)
    name = name.decode() if isinstance(name, bytes) else name
    uuid = pynvml.nvmlDeviceGetUuid(h)
    uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
    total = pynvml.nvmlDeviceGetMemoryInfo(h).total // (1024 * 1024)
    print(f"{i}\t{name}\t{total} MiB\t{uuid}")
EOF
}

if [ "$ARM" = "cards" ]; then
    cards
    exit 0
fi

# --- boot ------------------------------------------------------------------

boot() {
    local tag="$1"
    shift
    local log="$LOGDIR/$tag.log"
    : > "$log"

    export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$WT/python"
    export SGLANG_UNEVEN_DCP=1
    export SGLANG_UNEVEN_DCP_WEIGHTED=1
    export SGLANG_MAMBA_SSM_DTYPE=bfloat16
    # The collective family under test. bar1 is the transport whose spin
    # kernels the abort word reaches; the defect was seen there.
    export SGLANG_BARLINK=1
    export SGLANG_BARLINK_TRANSPORT=bar1
    export SGLANG_BARLINK_GRAPH_ENABLE=1
    # Short probe interval so the proof is about the mechanism, not about
    # waiting out a 1 s default. Production default stays 1 s.
    export SGLANG_BARLINK_PEER_PROBE_S="${SGLANG_BARLINK_PEER_PROBE_S:-0.5}"
    export SGLANG_BARLINK_PEER_TIMEOUT_S="${SGLANG_BARLINK_PEER_TIMEOUT_S:-120}"
    export SGLANG_BARLINK_PEER_LIVENESS="${SGLANG_BARLINK_PEER_LIVENESS:-1}"

    cd "$WT" || exit 1
    setsid "$VENV/bin/python" -u -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
        --rank-auto-reserve-mib 3000,2700,2700 \
        --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
        --max-running-requests 16 \
        --speculative-algorithm NEXTN --speculative-num-steps 3 \
        --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
        --enable-metrics \
        --host 127.0.0.1 --port "$PORT" \
        > "$log" 2>&1 &
    echo $! > "$LOGDIR/$tag.pid"

    local deadline=$((SECONDS + BOOT_S))
    while [ $SECONDS -lt $deadline ] \
        && ! curl -s -m 3 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
        sleep 3
    done
    if ! curl -s -m 3 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "BOOT_FAILED tag=$tag log=$log"
        return 1
    fi
    echo "booted tag=$tag pid=$(cat "$LOGDIR/$tag.pid") log=$log"
}

# One prefill plus one decode stretch, so the JIT cold-build window is closed
# and the CUDA graphs are captured before anything is killed. A kill inside
# the cold-build window measures the WRONG thing: the deadline is 40x there
# by design, and only the liveness check is supposed to carry it.
warmup() {
    curl -s -m 120 "http://127.0.0.1:$PORT/generate" \
        -H 'Content-Type: application/json' \
        -d '{"text":"Summarize the theory of tensor parallelism in detail.",
             "sampling_params":{"max_new_tokens":128,"temperature":0}}' \
        > /dev/null
}

teardown() {
    local tag="$1"
    local pidfile="$LOGDIR/$tag.pid"
    [ -f "$pidfile" ] || return 0
    local pid
    pid="$(cat "$pidfile")"
    # Only our own process group. Never a broad pkill: this box is shared.
    kill -TERM -- "-$pid" 2>/dev/null
    local deadline=$((SECONDS + 60))
    while [ $SECONDS -lt $deadline ] && kill -0 "$pid" 2>/dev/null; do
        sleep 2
    done
    kill -KILL -- "-$pid" 2>/dev/null
    rm -f "$pidfile"
}

# --- arm: kill -------------------------------------------------------------

# The TP workers are children of the launcher. Rank 0 is the process that
# serves HTTP; killing a NON-zero rank is the case under test, because that is
# the one that used to leave the others spinning rather than taking the whole
# server down with it.
worker_pids() {
    local parent="$1"
    "$VENV/bin/python" - "$parent" <<'EOF'
import sys

import psutil

parent = psutil.Process(int(sys.argv[1]))
for child in parent.children(recursive=True):
    try:
        cmd = " ".join(child.cmdline())
    except psutil.Error:
        continue
    if "scheduler" in cmd or "TP" in child.name():
        print(child.pid, child.name(), cmd[:120], sep="\t")
EOF
}

arm_kill() {
    boot kill || return 1
    local log="$LOGDIR/kill.log"
    local parent
    parent="$(cat "$LOGDIR/kill.pid")"

    echo "--- warmup (closes the JIT cold-build window, captures the graphs)"
    warmup
    warmup

    echo "--- worker inventory"
    worker_pids "$parent" | tee "$LOGDIR/kill.workers"

    # Pick the LAST worker: highest rank, never the HTTP server.
    local victim
    victim="$(awk 'END{print $1}' "$LOGDIR/kill.workers")"
    if [ -z "$victim" ]; then
        echo "NO_VICTIM -- could not identify a TP worker; see $LOGDIR/kill.workers"
        teardown kill
        return 1
    fi

    # Put the group into a collective and kill mid-flight.
    curl -s -m 5 "http://127.0.0.1:$PORT/generate" \
        -H 'Content-Type: application/json' \
        -d '{"text":"Write a long essay about heterogeneous tensor parallelism.",
             "sampling_params":{"max_new_tokens":2048,"temperature":0}}' \
        > /dev/null 2>&1 &
    sleep 2

    local t0=$SECONDS
    echo "--- SIGKILL rank victim pid=$victim at t=0"
    kill -KILL "$victim"

    # What counts as a pass: a NAMED error, on the surviving ranks, within
    # CRITERION_S. "The process died" is not enough -- a silent exit is the
    # old silent-corruption failure with extra steps.
    local marker="" hit=0
    local deadline=$((SECONDS + WATCH_S))
    while [ $SECONDS -lt $deadline ] && [ "$hit" -eq 0 ]; do
        if grep -qE "PeerLostError|peer rank gone|no longer exists" "$log"; then
            hit=1
            marker="$(grep -m1 -E 'PeerLostError|peer rank gone|no longer exists' "$log")"
        fi
        sleep 1
    done
    local elapsed=$((SECONDS - t0))

    echo "=== RESULT (arm: kill)"
    echo "victim_pid      $victim"
    echo "named_error     $hit"
    echo "seconds         $elapsed"
    echo "criterion       <= $CRITERION_S s"
    if [ "$hit" -eq 1 ] && [ "$elapsed" -le "$CRITERION_S" ]; then
        echo "verdict         PASS"
    else
        echo "verdict         FAIL"
    fi
    echo "marker          ${marker:-<none>}"
    echo "log             $log"

    # Second half of the criterion: no rank is left burning the card.
    echo "--- residual GPU utilization 10 s after the error"
    sleep 10
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used \
        --format=csv,noheader

    teardown kill
}

# --- arm: ab ---------------------------------------------------------------

# The success path must be unchanged. ms/verify and ms/prefill are the axes
# (not tok/s), read from the metrics endpoint rather than from the log.
measure() {
    local tag="$1"
    warmup
    for _ in 1 2 3; do
        curl -s -m 120 "http://127.0.0.1:$PORT/generate" \
            -H 'Content-Type: application/json' \
            -d '{"text":"Explain speculative decoding and its acceptance model.",
                 "sampling_params":{"max_new_tokens":256,"temperature":0}}' \
            > /dev/null
    done
    curl -s -m 10 "http://127.0.0.1:$PORT/metrics" > "$LOGDIR/$tag.metrics"
    "$VENV/bin/python" - "$LOGDIR/$tag.metrics" "$tag" <<'EOF'
import re
import sys

path, tag = sys.argv[1], sys.argv[2]
text = open(path).read()
wanted = (
    "sglang:e2e_request_latency_seconds_sum",
    "sglang:time_per_output_token_seconds_sum",
    "sglang:time_to_first_token_seconds_sum",
    "sglang:num_generation_tokens_total",
    "sglang:prompt_tokens_total",
)
print(f"--- {tag}")
for key in wanted:
    for line in text.splitlines():
        if line.startswith(key):
            print(" ", line)
            break
EOF
}

arm_ab() {
    for enabled in 1 0; do
        local tag="ab_liveness_$enabled"
        echo "=== arm: ab, SGLANG_BARLINK_PEER_LIVENESS=$enabled"
        SGLANG_BARLINK_PEER_LIVENESS="$enabled" boot "$tag" || return 1
        measure "$tag"
        teardown "$tag"
        # Let the cards settle so the second arm does not inherit the first
        # arm's thermal state.
        sleep 30
    done
    echo "=== compare $LOGDIR/ab_liveness_1.metrics vs $LOGDIR/ab_liveness_0.metrics"
    echo "Report ms/verify and ms/prefill per rank. The claim is that the"
    echo "difference sits inside the A-vs-A noise floor; measure that floor"
    echo "first by running this arm twice with the SAME setting."
}

case "$ARM" in
    kill) arm_kill ;;
    ab)   arm_ab ;;
    *)    echo "usage: $0 [kill|ab|cards]" >&2; exit 2 ;;
esac
