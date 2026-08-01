#!/usr/bin/env bash
# One arm of the #361 A/B: MoE expert dispatch over the BAR1 direct path
# (--moe-a2a-backend bar1ep) against the stock path (--moe-a2a-backend none).
#
# EAGER vs EAGER, and that is not a concession. bar1ep exchanges the per-rank
# token counts over a HOST collective before the data path (an all_gather on
# the CPU group), so server_args turns CUDA graphs off for it by construction.
# Comparing it against a graph-captured stock run would measure the capture,
# not the dispatch, so the control arm is forced eager too.
#
# TP=2 on the two 3080s, not 3 cards: bar1ep maps expert e onto rank
# e // num_local_experts, so num_experts must divide by the world size. The
# vehicle has 256 experts -- 256/3 is not an integer, and the mixed 5090+3080
# pair is refused by the stock memory-balance guard at even TP. Two identical
# 3080s is the only geometry on this rig that satisfies both.
#
# Usage: bar1ep_vs_nccl_arm.sh <arm> <backend: bar1ep|none> [port]
set -uo pipefail

ARM="${1:?arm name}"; BACKEND="${2:?bar1ep|none}"; PORT="${3:-30361}"

HOST="${BAR1_HOST:-192.168.0.1}"
KEY="${BAR1_HOST_KEY:-/root/.ssh/id_root@proxmox}"
SUB="${BAR1_HOST_SUBVOL:-/spinning/subvol-999-disk-0}"
IMG="${BAR1_IMG:-htsglang:cu130-nccl2307}"
WT="${WT:-/spinning/wt-361b-ab}"
MODEL_SUB="${MODEL_SUB:-Qwen3.6-35B-A3B-FP8}"
OUT="${OUT:-/spinning/gpu-battery-results/2026-08-01_361_bar1ep_ab}"
H_LOG="/root/battery-bar1/361.$ARM.log"
NAME="a361$ARM"

mkdir -p "$OUT"
hssh() { timeout "${1:?t}" ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 \
         -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$HOST" "$2"; }

# bar1ep rides on the BAR1 transport, so that arm needs the holder device and
# the barlink env. The control arm gets neither -- barlink unset is the stock
# NCCL path, which is exactly what it is the control for.
if [ "$BACKEND" = bar1ep ]; then
    DEV_ARGS="--device /dev/dmabuf_holder --cap-add SYS_ADMIN --security-opt apparmor=unconfined -v /sys:/sys"
    BARLINK_ENV="-e SGLANG_BARLINK=1 -e SGLANG_BARLINK_TRANSPORT=bar1 \
 -e SGLANG_BARLINK_BAR1_NV_SOURCE=/nvsrc -e SGLANG_BARLINK_BAR1_WINDOW_MIB=64"
else
    DEV_ARGS=""; BARLINK_ENV=""
fi

echo "=== $(date -u +%H:%M:%SZ) arm=$ARM backend=$BACKEND port=$PORT ==="
hssh 60 "docker rm -f $NAME 2>/dev/null; rm -f $H_LOG" >/dev/null 2>&1

BOOT_T0=$(date +%s)
hssh 120 "setsid bash -c 'docker run --rm --name $NAME --network host \
 --gpus all $DEV_ARGS --shm-size=8g \
 -v $SUB$WT:/wt:ro \
 -v $SUB/spinning/nvidia-open-595:/nvsrc:ro \
 -v /root/battery-bar1/extcache_docker:/extcache \
 -v /root/.cache:/root/.cache -v /root/.triton:/root/.triton \
 -v $SUB/spinning/llm_stuff/club-3090/models-cache/$MODEL_SUB:/model:ro \
 -e PYTHONPATH=/wt/python -e TORCH_EXTENSIONS_DIR=/extcache \
 -e TORCH_CUDA_ARCH_LIST=8.6 \
 -e CUDA_VISIBLE_DEVICES=1,2 \
 $BARLINK_ENV \
 --entrypoint bash $IMG -c \"cd /wt && python3 -m sglang.launch_server \
   --model-path /model --tp-size 2 \
   --moe-a2a-backend $BACKEND --deepep-mode normal \
   --mem-fraction-static ${MEMFRAC:-0.94} --context-length ${CTX:-2048} --trust-remote-code \
   --max-running-requests ${MAXREQ:-1} --disable-cuda-graph \
   --enable-metrics --host 127.0.0.1 --port $PORT\"' > $H_LOG 2>&1 &" \
  >/dev/null 2>&1

# Bounded readiness poll. Never one unbounded wait: a wedged boot must show up
# as a bounded failure, not as a silent agent.
UP=0
for _ in $(seq 1 60); do
    sleep 10
    R=$(hssh 30 "curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health 2>/dev/null; echo; \
                 docker ps -q --filter name=$NAME | head -1")
    case "$R" in 200*) UP=1; break ;; esac
    echo "$R" | tail -1 | grep -q . || { echo "CONTAINER GONE arm=$ARM"; break; }
done
BOOT_S=$(( $(date +%s) - BOOT_T0 ))

if [ "$UP" != 1 ]; then
    echo "BOOT FAILED arm=$ARM after ${BOOT_S}s"
    hssh 60 "grep -nE 'Error|error|Traceback|not available|NotImplementedError|out of memory' $H_LOG | tail -12"
    hssh 60 "cp $H_LOG $H_LOG.kept; docker stop -t 10 $NAME 2>/dev/null; true" >/dev/null 2>&1
    exit 2
fi
echo "arm=$ARM UP in ${BOOT_S}s"

# --- THE GATE: is this arm really the path it claims to be? -----------------
# A measurement of a silently different path is worse than no measurement.
hssh 60 "grep -cE 'bar1ep: byte proof passed|bar1ep: Byte' $H_LOG; echo '|'; \
         grep -cE 'ACHIEVED=bar1' $H_LOG" > "$OUT/gate_$ARM.txt" 2>&1
cat "$OUT/gate_$ARM.txt"
echo "GATE arm=$ARM backend=$BACKEND (bar1ep needs both counts > 0; none needs both 0)"
