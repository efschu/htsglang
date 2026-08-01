#!/usr/bin/env bash
# Arm K_bar1_graphs of the #349 boot matrix, run where it CAN run (#369).
#
# The sweep driver composes a plain `python -m sglang.launch_server` command
# and runs it locally. Every other arm is fine with that; K is not. K crosses
# the bar1 transport with CUDA graphs, and bar1 needs /dev/dmabuf_holder,
# which CT999 has no device-cgroup entry for and cannot mknod (#361). So K
# boots in the htsglang image on the Proxmox host, which has both the device
# node and a working toolchain (#369, runbook 4.15).
#
# The artifacts are written in the sweep's own contract -- arm.json,
# server.log, probes.json in one directory -- so the verdict comes from the
# SAME check_arm core as the other sixteen arms. A separate judgement for one
# arm would be a second standard, and the matrix would stop meaning one thing.
set -uo pipefail

OUT="${1:?artifact dir for this arm}"
PORT="${PORT:-31349}"
HOST="${BAR1_HOST:-192.168.0.1}"
KEY="${BAR1_HOST_KEY:-/root/.ssh/id_root@proxmox}"
SUB="${BAR1_HOST_SUBVOL:-/spinning/subvol-999-disk-0}"
IMG="${BAR1_IMG:-htsglang:cu130-nccl2307}"
WT="${WT:-/spinning/wt-349-sweep}"
MODEL_SUB="${MODEL_SUB:-Qwen3.6-27B-FP8}"
CEILING="${CEILING:-1200}"
NAME=a349K
H_LOG=/root/battery-bar1/349.K.log

mkdir -p "$OUT"
hssh() { timeout "${1:?t}" ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 \
         -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$HOST" "$2"; }

hssh 60 "docker rm -f $NAME 2>/dev/null; rm -f $H_LOG" >/dev/null 2>&1

# The arm's env and flags, verbatim from `sweep.py --dry-run K_bar1_graphs`.
# Graphs are deliberately NOT disabled: the crossing IS bar1 under capture.
T0=$(date +%s)
hssh 120 "setsid bash -c 'docker run --rm --name $NAME --network host \
 --gpus all --device /dev/dmabuf_holder \
 --cap-add SYS_ADMIN --security-opt apparmor=unconfined \
 -v /sys:/sys --shm-size=8g \
 -v $SUB$WT:/wt:ro \
 -v $SUB/spinning/nvidia-open-595:/nvsrc:ro \
 -v /root/battery-bar1/extcache_docker:/extcache \
 -v /root/.cache:/root/.cache -v /root/.triton:/root/.triton \
 -v $SUB/spinning/llm_stuff/club-3090/models-cache/$MODEL_SUB:/model:ro \
 -e PYTHONPATH=/wt/python -e TORCH_EXTENSIONS_DIR=/extcache \
 -e TORCH_CUDA_ARCH_LIST=8.6\;12.0 \
 -e SGLANG_BARLINK_BAR1_NV_SOURCE=/nvsrc \
 -e SGLANG_UNEVEN_DCP=1 -e SGLANG_UNEVEN_DCP_WEIGHTED=1 \
 -e SGLANG_MAMBA_SSM_DTYPE=bfloat16 \
 -e SGLANG_BARLINK=1 -e SGLANG_BARLINK_TRANSPORT=bar1 \
 -e SGLANG_BARLINK_GRAPH_ENABLE=1 \
 -e SGLANG_BARLINK_BAR1_WINDOW_MIB=64 -e SGLANG_BARLINK_BAR1_WINDOW_MIB_DCP=32 \
 --entrypoint bash $IMG -c \"cd /wt && python3 -m sglang.launch_server \
   --model-path /model --host 127.0.0.1 --port $PORT \
   --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
   --rank-auto-reserve-mib 3000,2700,2700 \
   --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
   --max-running-requests 16 \
   --speculative-algorithm NEXTN --speculative-num-steps 3 \
   --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
   --enable-metrics\"' > $H_LOG 2>&1 &" >/dev/null 2>&1

# Bounded to the arm's own ceiling. Reaching it is a FINDING, not a timeout to
# be papered over: warm bar1 capture is ~66 s (#370), so a K that spends 20
# minutes is telling us the cache went cold or the crossing is pathological.
STATUS=hang
for _ in $(seq 1 $((CEILING / 10))); do
    sleep 10
    R=$(hssh 30 "curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health 2>/dev/null; echo; docker ps -q --filter name=$NAME | head -1")
    case "$R" in 200*) STATUS=ready; break ;; esac
    echo "$R" | tail -1 | grep -q . || { STATUS=crashed; break; }
done
BOOT_S=$(( $(date +%s) - T0 ))
echo "arm K_bar1_graphs: status=$STATUS after ${BOOT_S}s (ceiling ${CEILING}s)"

# Probes, only if it came up. Same two tiers the sweep uses.
PROBES="[]"
if [ "$STATUS" = ready ]; then
    BYTE=$(hssh 60 "curl -s -m 40 http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' -d '{\"text\":\"Count up by ones, one number per line, no words:\n1\n2\n3\n\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":24}}'")
    GRAD=$(hssh 60 "curl -s -m 40 http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' -d '{\"text\":\"Continue the sequence, one letter per line:\nt\nu\nv\n\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":32}}'")
    printf '%s\n' "$BYTE" > "$OUT/probe_byte_raw.json"
    printf '%s\n' "$GRAD" > "$OUT/probe_graded_raw.json"
fi

hssh 90 "cp $H_LOG $H_LOG.kept 2>/dev/null; docker stop -t 15 $NAME 2>/dev/null; sleep 3; \
         docker ps -a --filter name=$NAME --format '{{.Names}}'; \
         nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
hssh 90 "cat $H_LOG.kept" > "$OUT/server.log" 2>&1

python3 - "$OUT" "$STATUS" "$BOOT_S" <<'PY'
import json, sys
out, status, boot_s = sys.argv[1], sys.argv[2], int(sys.argv[3])
json.dump({"name": "K_bar1_graphs", "kind": "boot", "axis": "bar1 x graphs",
           "boot_status": status, "expected_seconds": 1200,
           "capture_note": f"booted in the htsglang image on the PVE host "
                           f"(#369 route); reached {status} in {boot_s}s",
           "observed_seconds": boot_s},
          open(f"{out}/arm.json", "w"), indent=2)
PY
echo "artifacts in $OUT"
