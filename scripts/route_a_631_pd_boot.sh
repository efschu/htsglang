#!/usr/bin/env bash
# #631 Route A -- a PD pair on this rig: prefill arm + TP/DCP decode arm.
#
#   pair    prefill TP=1        ->  decode TP=2 / DCP=2        (no speculation)
#   nextn   prefill TP=1        ->  decode TP=2 / DCP=2 + NEXTN
#
# DEVICE SELECTION IS BY UUID, NOT BY INDEX. nvidia-smi/NVML and CUDA
# enumerate this host in DIFFERENT orders -- measured here, not assumed:
#   NVML 0 = 3080 = cuda:1     NVML 1 = 5090 = cuda:0     NVML 2 = 3080 = cuda:2
# so passing an NVML index to CUDA_VISIBLE_DEVICES selects the wrong card. A
# first boot of this script did exactly that and ran the "3080 prefill arm" on
# the 5090 (the log said avail mem=30.66 GB on a 20 GB card). CUDA accepts
# "GPU-<uuid>" in CUDA_VISIBLE_DEVICES, which is ordering-independent, so that
# is what is used.
#
# ENV. SGLANG_BARLINK=0 is kept only to match production and explains
# NOTHING: environ.py:688 declares SGLANG_BARLINK = EnvBool(False), so it is
# already the default, and on this rig barlink was additionally forced off
# rig-wide by /spinning/COUNTERTEST_NCCL. An earlier version of this header
# credited it with clearing a decode wedge. That was wrong.
#
# The wedge was rank-divergent CUDA-graph capture shapes: capture_bs is clamped
# by the RANK-LOCAL req_to_token_pool.size, capture replays a collective per
# shape, and two ranks with different lists run different collective counts
# (TP0 [..,16,19] vs TP1 [..,16,24] -> hang; identical lists -> served). Fixed
# in base_cuda_graph_runner.py by min-reducing that bound across the TP group.
#
# SGLANG_UNEVEN_DCP_WEIGHTED=1 does have a mechanism -- the WEIGHTED owner rule
# never reaches dcp_even_write_mask -- and is required by --draft-kv-layout dcp.
# SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 is required because a decode TP
# group spanning a 5090 and a 3080 is deliberately unbalanced.
set -euo pipefail

RUNG="${1:-pair}"
[[ "$RUNG" =~ ^(pair|nextn)$ ]] || { echo "usage: $0 {pair|nextn}" >&2; exit 2; }

WT="${WT:-/spinning/wt-631-routea}"
PY=/spinning/htsglang-gpu/.venv/bin/python
LOGDIR=/tmp/route-a-631
PREFILL_PORT=30021
DECODE_PORT=30022
PROXY_PORT=8100
BOOTSTRAP_PORT=8998

export PYTHONPATH="$WT/python"
export LD_LIBRARY_PATH=/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_BARLINK=0
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
# The decode arm's TP group deliberately spans a 5090 and a 3080. sglang's
# TP memory-balance check (model_runner.py:1880-1890) treats that as "some
# GPUs may be occupied by other processes" and aborts -- it assumes a TP group
# is homogeneous, which is the assumption this fork's uneven-TP path exists to
# break. The imbalance is the CONFIGURATION here, expressed by
# --rank-tp-ratio, not a symptom, so the check is downgraded to its warning
# form via its own documented toggle rather than worked around.
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

mkdir -p "$LOGDIR"

# --- resolve cards by NAME -> UUID -------------------------------------------
BIG_UUID=""; SMALL_UUID=()
while IFS=, read -r _idx name uuid _tot; do
    name="$(echo "$name" | xargs)"; uuid="$(echo "$uuid" | xargs)"
    case "$name" in *5090*) BIG_UUID="$uuid" ;; *3080*) SMALL_UUID+=("$uuid") ;; esac
done < <(nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader)
[[ -n "$BIG_UUID" && ${#SMALL_UUID[@]} -ge 2 ]] || {
    echo "FATAL: expected one 5090 and two 3080s from NVML" >&2; exit 1; }
echo "5090=$BIG_UUID"
echo "3080a=${SMALL_UUID[0]}"
echo "3080b=${SMALL_UUID[1]}"

if [[ "$RUNG" == "nextn" ]]; then
    # An MTP-bearing 27B. Prefill goes on the 5090 because a 16 GB weight set
    # plus a prompt-sized KV pool does not leave workable headroom on a 20 GB
    # 3080; decode TP=2 splits those weights ~8 GB per rank, which the two
    # 3080s carry comfortably. The decode ratio stays NON-UNIFORM because
    # --draft-kv-layout dcp requires a non-uniform --rank-tp-ratio.
    MODEL="${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4}"
    PREFILL_DEV="$BIG_UUID"
    DECODE_DEV="${SMALL_UUID[0]},${SMALL_UUID[1]}"
    RATIO="8,7"
    # EAGLE, not NEXTN: NEXTN is an ALIAS that handle_speculative_decoding
    # resolves to EAGLE at server_args.py:6067, but _handle_dcp_validation
    # parses the algorithm at :5922 -- before that -- so the raw string reaches
    # SpeculativeAlgorithm.from_string, which has no NEXTN member, whenever
    # --dcp-size is also given explicitly. EAGLE is the resolved value and a
    # real enum member, so it is the same speculation with the alias hazard
    # removed. The MTP head comes from the checkpoint (mtp.* weights), not
    # from the algorithm name.
    SPEC=(--speculative-algorithm EAGLE
          --speculative-num-steps 3
          --speculative-eagle-topk 1
          --speculative-num-draft-tokens 4
          --draft-kv-layout dcp)
else
    # Plumbing rung: small model, fast load, no speculation. Decode spans the
    # 5090 + one 3080 so the token split is genuinely heterogeneous (8:5).
    MODEL="${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.5-4B}"
    PREFILL_DEV="${SMALL_UUID[0]}"
    DECODE_DEV="$BIG_UUID,${SMALL_UUID[1]}"
    RATIO="8,5"
    SPEC=()
fi
echo "rung=$RUNG model=$MODEL ratio=$RATIO"

wait_healthy() {
    local url="$1" name="$2" log="$3" deadline=$((SECONDS + 1200))
    while (( SECONDS < deadline )); do
        curl -fsS -m 5 "$url/health" >/dev/null 2>&1 && { echo "$name healthy after ${SECONDS}s"; return 0; }
        sleep 5
    done
    echo "FATAL: $name never healthy; tail of $log:" >&2; tail -40 "$log" >&2; return 1
}

echo "== prefill arm =="
CUDA_VISIBLE_DEVICES="$PREFILL_DEV" setsid "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --trust-remote-code \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --tp 1 --page-size 1 \
    --enable-metrics --host 127.0.0.1 --port "$PREFILL_PORT" \
    > "$LOGDIR/prefill_$RUNG.log" 2>&1 &

echo "== decode arm TP=2 / DCP=2 =="
CUDA_VISIBLE_DEVICES="$DECODE_DEV" setsid "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --trust-remote-code \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --tp 2 --rank-tp-ratio "$RATIO" --dcp-size 2 --page-size 1 \
    "${SPEC[@]+"${SPEC[@]}"}" \
    --enable-metrics --host 127.0.0.1 --port "$DECODE_PORT" \
    > "$LOGDIR/decode_$RUNG.log" 2>&1 &

wait_healthy "http://127.0.0.1:$PREFILL_PORT" prefill "$LOGDIR/prefill_$RUNG.log"
wait_healthy "http://127.0.0.1:$DECODE_PORT" decode "$LOGDIR/decode_$RUNG.log"

echo "== PD proxy =="
setsid "$PY" -m sglang.srt.disaggregation.local_proxy \
    --prefill "http://127.0.0.1:$PREFILL_PORT" \
    --decode "http://127.0.0.1:$DECODE_PORT" \
    --bootstrap-port "$BOOTSTRAP_PORT" \
    --host 127.0.0.1 --port "$PROXY_PORT" \
    > "$LOGDIR/proxy_$RUNG.log" 2>&1 &

wait_healthy "http://127.0.0.1:$PROXY_PORT" proxy "$LOGDIR/proxy_$RUNG.log"
echo "Route A $RUNG up on http://127.0.0.1:$PROXY_PORT"

# NOTE (found 2026-08-07, separate slice): dcp_size is auto-set from
# SGLANG_UNEVEN_DCP=1 + a non-uniform --rank-tp-ratio rather than passed as
# --dcp-size. That is production's path, and it also SIDESTEPS a latent
# ordering bug this rung exposed:
#
#   server_args.__post_init__ calls _handle_dcp_validation at :5922, which
#   reads BOTH speculative_algorithm and dcp_size before either is resolved --
#   the NEXTN -> EAGLE alias lands at :6067 (handle_speculative_decoding) and
#   dcp_size is auto-set at :5957 (_handle_uneven_tp). Passing --dcp-size
#   explicitly TOGETHER with --speculative-algorithm NEXTN therefore reaches
#   SpeculativeAlgorithm.from_string("NEXTN") and dies with "Unknown
#   speculative algorithm name: NEXTN". With dcp_size still 1 at that point
#   (the env path) the gate returns early instead -- which is why no existing
#   boot hits it, and equally why the #229 refusal it implements is currently
#   dead on the auto-set path.
#
# Not fixed here on purpose. The fix is an ORDERING change (run the gate after
# resolution, like the #636/#642/#631b gates do), not an alias in from_string:
# the hook maps NEXTN -> EAGLE except for a gemma4 draft, where it becomes
# FROZEN_KV_MTP -- exactly the case this gate refuses under dcp_size > 1. An
# alias in from_string would silently disable that refusal.
