#!/usr/bin/env bash
# #631 Route A -- L1: a PD pair on this rig, prefill arm and decode arm.
#
#   prefill  1 rank   on one 3080
#   decode   TP=2 / DCP=2  on the 5090 + the other 3080  (token-sharded)
#
# Speculation is NOT requested here: pd_disaggregation_hook.py:195-229 (#631a)
# refuses --speculative-algorithm on either PD arm today. Lifting that refusal
# is the next slice; this rung proves the pair boots, hands over KV and returns
# coherent text without it.
#
# Card indices come from NVML by NAME at runtime. torch and NVML enumeration
# diverge on this host, so the 5090 is never assumed to be a fixed index.
set -euo pipefail

WT=/spinning/wt-631-routea
PY=/spinning/htsglang-gpu/.venv/bin/python
MODEL="${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.5-4B}"
LOGDIR=/tmp/route-a-631
PREFILL_PORT=30021
DECODE_PORT=30022
PROXY_PORT=8100
BOOTSTRAP_PORT=8998

export PYTHONPATH="$WT/python"
export LD_LIBRARY_PATH=/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

mkdir -p "$LOGDIR"

BIG=""; SMALL=()
while IFS=, read -r idx name _t; do
    idx="$(echo "$idx" | tr -d ' ')"; name="$(echo "$name" | xargs)"
    case "$name" in *5090*) BIG="$idx" ;; *3080*) SMALL+=("$idx") ;; esac
done < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader)
[[ -n "$BIG" && ${#SMALL[@]} -ge 2 ]] || { echo "FATAL: NVML did not report one 5090 and two 3080s" >&2; exit 1; }
echo "NVML: 5090=$BIG  3080s=${SMALL[0]},${SMALL[1]}"

wait_healthy() {
    local url="$1" name="$2" log="$3" deadline=$((SECONDS + 900))
    while (( SECONDS < deadline )); do
        curl -fsS -m 5 "$url/health" >/dev/null 2>&1 && { echo "$name healthy after ${SECONDS}s"; return 0; }
        if ! pgrep -f "port $((${url##*:}))" >/dev/null 2>&1; then :; fi
        sleep 5
    done
    echo "FATAL: $name never healthy; tail of $log:" >&2
    tail -40 "$log" >&2
    return 1
}

echo "== launching prefill arm on GPU ${SMALL[0]} =="
CUDA_VISIBLE_DEVICES="${SMALL[0]}" setsid "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --trust-remote-code \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --tp 1 --page-size 1 \
    --enable-metrics --host 127.0.0.1 --port "$PREFILL_PORT" \
    > "$LOGDIR/prefill_L1.log" 2>&1 &

echo "== launching decode arm TP=2/DCP=2 on GPUs $BIG,${SMALL[1]} =="
SGLANG_UNEVEN_DCP=1 CUDA_VISIBLE_DEVICES="$BIG,${SMALL[1]}" setsid "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --trust-remote-code \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --tp 2 --rank-tp-ratio 8,5 --dcp-size 2 --page-size 1 \
    --enable-metrics --host 127.0.0.1 --port "$DECODE_PORT" \
    > "$LOGDIR/decode_L1.log" 2>&1 &

wait_healthy "http://127.0.0.1:$PREFILL_PORT" prefill "$LOGDIR/prefill_L1.log"
wait_healthy "http://127.0.0.1:$DECODE_PORT" decode "$LOGDIR/decode_L1.log"

echo "== launching PD proxy =="
setsid "$PY" -m sglang.srt.disaggregation.local_proxy \
    --prefill "http://127.0.0.1:$PREFILL_PORT" \
    --decode "http://127.0.0.1:$DECODE_PORT" \
    --bootstrap-port "$BOOTSTRAP_PORT" \
    --host 127.0.0.1 --port "$PROXY_PORT" \
    > "$LOGDIR/proxy_L1.log" 2>&1 &

wait_healthy "http://127.0.0.1:$PROXY_PORT" proxy "$LOGDIR/proxy_L1.log"
echo "Route A L1 up on http://127.0.0.1:$PROXY_PORT"
