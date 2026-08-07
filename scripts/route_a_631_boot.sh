#!/usr/bin/env bash
# Route A (#631): a PP prefill group handing KV to a TP+NEXTN decode group.
#
# Turnkey boot for the operator's GPU window. Three rungs, cheapest first.
# Each rung is a complete, independently meaningful test; a later rung is
# not a prerequisite fix for an earlier one.
#
#   L1  handover axis   prefill PP=1  ->  decode TP=2 / DCP=2   (venv, 3 cards)
#   L2  pipeline axis   prefill PP=2  ->  decode TP=1           (venv, 3 cards)
#   L3  full Route A    prefill PP=3  ->  decode TP=3 / DCP=3   (Docker + MPS)
#
# WHY THE LADDER, AND NOT ONE COMMAND
# -----------------------------------
# L3 is the shape the ticket asks for, and it needs six ranks on three
# cards, i.e. two processes sharing a physical GPU. That is refused on this
# host's venv, verified by running the probe rather than by reading about
# it (sglang.srt.disaggregation.topology.check_process_colocation_prerequisites):
#
#   NCCL multi-rank per GPU: runtime NCCL is 2.28.9 (from libnccl.so.2),
#     below 2.30 ... Needs NCCL >= 2.30
#   CUDA MPS (co-location scheduler): MPS control directory /tmp/nvidia-mps
#     does not exist ... Needs `nvidia-cuda-mps-control -d`
#
# So L3 requires the Docker image (docker/htsglang.Dockerfile pins NCCL
# 2.30.7) plus an MPS daemon started before the ranks. L1 and L2 need
# neither: they split the three cards between the two arms instead of
# sharing any card, which is why they run today. Between them they cover
# both mechanisms L3 combines -- L1 the DCP owner-rule handover
# (owned_ordinals), L2 the multi-PP-stage sender fan-in.
#
# WHAT EACH FLAG IS FOR
# ---------------------
# --disaggregation-transfer-backend mooncake
#     Load-bearing, not a default restated. Only the mooncake sender
#     implements the owned_ordinals row filter
#     (disaggregation/mooncake/conn.py:1502-1558). The nixl and mori
#     senders refuse a non-None owned_ordinals by name
#     (nixl/conn.py:2514-2518, mori/conn.py:1702-1706). It is also the
#     ServerArgs default, so this line documents an invariant the recipe
#     depends on; the boot-time guard added with this script refuses the
#     other two rather than letting the first transfer discover it.
# SGLANG_UNEVEN_DCP / --rank-tp-ratio on the DECODE arm only
#     Together these resolve dcp_size to tp_size and install the uneven-TP
#     replicated-KV layout. Without the layout, decode.py:1131-1137 refuses
#     stock head-sharded DCP receive -- now hoisted to boot.
# NO --enable-hierarchical-cache and NO hicache storage backend on the
# PREFILL arm
#     Trap #630: PP>1 x Disk-HiCache wedges silently at warmup and the
#     health endpoint returns 503 forever. Both default to off; the point
#     is that the standing "100 GB Disk-HiCache in every serving boot"
#     rule must NOT be applied to a PP prefill arm until #630 is fixed.
#     This is a real conflict between two standing rules and is called out
#     here rather than silently resolved.
# --enable-metrics
#     Mandatory in every boot on this rig, no topology exception.
#
# PREREQUISITE THIS SCRIPT CHECKS AND DOES NOT GUESS
# --------------------------------------------------
# Physical card indices are resolved from NVML at runtime by name. NVML and
# torch enumeration can diverge and the order can shift between boots, so
# the 5090 is never assumed to be a particular index.

set -euo pipefail

RUNG="${1:-}"
if [[ ! "$RUNG" =~ ^(L1|L2|L3|probe)$ ]]; then
    echo "usage: $0 {probe|L1|L2|L3}" >&2
    exit 2
fi

WT="${WT:-/spinning/wt-631-routea}"
PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"
MODEL="${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.5-4B}"
LOGDIR="${LOGDIR:-/tmp/route-a-631}"
PREFILL_PORT="${PREFILL_PORT:-30021}"
DECODE_PORT="${DECODE_PORT:-30022}"
PROXY_PORT="${PROXY_PORT:-8100}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"

NVRTC="/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$NVRTC:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"

mkdir -p "$LOGDIR"

# --- NVML card identity -----------------------------------------------------
# Prints "<index> <name>" per card. Never assume the 5090's index.
card_map() {
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
}

resolve_cards() {
    BIG=""      # the 5090: the only card here that can carry two ranks
    SMALL=()    # the two 3080s
    while IFS=, read -r idx name _total; do
        idx="$(echo "$idx" | tr -d ' ')"
        name="$(echo "$name" | xargs)"
        case "$name" in
            *5090*) BIG="$idx" ;;
            *3080*) SMALL+=("$idx") ;;
        esac
    done < <(card_map)
    if [[ -z "$BIG" || ${#SMALL[@]} -lt 2 ]]; then
        echo "FATAL: expected one 5090 and two 3080s; NVML reports:" >&2
        card_map >&2
        exit 1
    fi
}

if [[ "$RUNG" == "probe" ]]; then
    echo "== NVML card map =="
    card_map
    resolve_cards
    echo "resolved: 5090=$BIG  3080s=${SMALL[0]},${SMALL[1]}"
    echo
    echo "== colocated-process prerequisites (gates L3) =="
    CUDA_VISIBLE_DEVICES=99 "$PY" - <<'EOF'
from sglang.srt.disaggregation.topology import check_process_colocation_prerequisites
try:
    check_process_colocation_prerequisites()
    print("ACCEPTED: L3 can run in this environment")
except Exception as exc:
    print("REFUSED:", exc)
EOF
    echo
    echo "== free VRAM (L1/L2 need the live serving stack stopped) =="
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
    exit 0
fi

resolve_cards

wait_healthy() {
    local url="$1" name="$2" deadline=$((SECONDS + 900))
    while (( SECONDS < deadline )); do
        if curl -fsS -m 5 "$url/health" >/dev/null 2>&1; then
            echo "$name healthy after ${SECONDS}s"
            return 0
        fi
        sleep 5
    done
    echo "FATAL: $name never became healthy (see $LOGDIR). If this is a PP" >&2
    echo "prefill arm stuck at warmup, check for disk HiCache first (#630)." >&2
    return 1
}

case "$RUNG" in
L1)
    # Handover axis. Prefill is a single rank on one 3080; decode is TP=2
    # over the 5090 + the other 3080 with dcp_size=2, so the decode pool is
    # token-sharded and owned_ordinals is exercised end to end.
    PREFILL_CARDS="${SMALL[0]}"
    DECODE_CARDS="$BIG,${SMALL[1]}"
    # Token ratio follows capacity, not card count: the 5090 carries more.
    RATIO="8,5"
    CUDA_VISIBLE_DEVICES="$PREFILL_CARDS" setsid "$PY" -m sglang.launch_server \
        --model-path "$MODEL" --trust-remote-code \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend mooncake \
        --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
        --tp 1 \
        --enable-metrics --host 127.0.0.1 --port "$PREFILL_PORT" \
        > "$LOGDIR/prefill_L1.log" 2>&1 &

    SGLANG_UNEVEN_DCP=1 SGLANG_UNEVEN_DCP_WEIGHTED=1 \
    CUDA_VISIBLE_DEVICES="$DECODE_CARDS" setsid "$PY" -m sglang.launch_server \
        --model-path "$MODEL" --trust-remote-code \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend mooncake \
        --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
        --tp 2 --rank-tp-ratio "$RATIO" \
        --page-size 1 \
        --enable-metrics --host 127.0.0.1 --port "$DECODE_PORT" \
        > "$LOGDIR/decode_L1.log" 2>&1 &
    ;;
L2)
    # Pipeline axis. Prefill is PP=2 layer-split over the two 3080s, decode
    # is a single rank on the 5090. Exercises the pp_size 2 -> 1 asymmetry
    # in the handshake and the multi-stage sender fan-in.
    PREFILL_CARDS="${SMALL[0]},${SMALL[1]}"
    DECODE_CARDS="$BIG"
    CUDA_VISIBLE_DEVICES="$PREFILL_CARDS" setsid "$PY" -m sglang.launch_server \
        --model-path "$MODEL" --trust-remote-code \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend mooncake \
        --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
        --tp 1 --pp 2 \
        --enable-metrics --host 127.0.0.1 --port "$PREFILL_PORT" \
        > "$LOGDIR/prefill_L2.log" 2>&1 &

    CUDA_VISIBLE_DEVICES="$DECODE_CARDS" setsid "$PY" -m sglang.launch_server \
        --model-path "$MODEL" --trust-remote-code \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend mooncake \
        --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
        --tp 1 \
        --enable-metrics --host 127.0.0.1 --port "$DECODE_PORT" \
        > "$LOGDIR/decode_L2.log" 2>&1 &
    ;;
L3)
    echo "L3 requires the Docker image (NCCL >= 2.30) and a running MPS" >&2
    echo "daemon. Run '$0 probe' first: it refuses with the exact reason" >&2
    echo "when this environment cannot host two processes per card." >&2
    CUDA_VISIBLE_DEVICES=99 "$PY" - <<'EOF'
from sglang.srt.disaggregation.topology import check_process_colocation_prerequisites
check_process_colocation_prerequisites()
print("colocated-process prerequisites satisfied")
EOF
    echo "Prerequisites satisfied; L3 launch is intentionally not automated" >&2
    echo "here -- it has never been executed, and a script that pretends" >&2
    echo "otherwise is worse than none." >&2
    exit 3
    ;;
esac

wait_healthy "http://127.0.0.1:$PREFILL_PORT" prefill
wait_healthy "http://127.0.0.1:$DECODE_PORT" decode

setsid "$PY" -m sglang.srt.disaggregation.local_proxy \
    --prefill "http://127.0.0.1:$PREFILL_PORT" \
    --decode "http://127.0.0.1:$DECODE_PORT" \
    --bootstrap-port "$BOOTSTRAP_PORT" \
    --host 127.0.0.1 --port "$PROXY_PORT" \
    > "$LOGDIR/proxy_$RUNG.log" 2>&1 &

wait_healthy "http://127.0.0.1:$PROXY_PORT" proxy

echo
echo "Route A $RUNG up on http://127.0.0.1:$PROXY_PORT (logs in $LOGDIR)."
echo "Correctness probe -- a token-sharded handover that dropped or"
echo "misfiled rows produces fluent nonsense, not an error, so read the"
echo "output rather than the status code:"
cat <<EOF

  curl -s http://127.0.0.1:$PROXY_PORT/generate \\
    -H 'Content-Type: application/json' \\
    -d '{"text":"Count from one to twenty in words, comma separated:",
         "sampling_params":{"temperature":0,"max_new_tokens":96}}' | jq -r .text

Expected: the full ordered sequence. A run whose owner rule was bypassed
typically stays grammatical and loses or repeats items in the middle.
EOF
