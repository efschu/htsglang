#!/usr/bin/env bash
# One #366 measurement ARM over the barlink BAR1 direct path, served from the
# htsglang Docker image ON THE PVE HOST.
#
# WHY DOCKER, NOT THE venv-SHIM HOST ROUTE (#366 window 1's base). bar1 needs a
# JIT-compiled CUDA extension, and the host toolchain cannot build it (gcc 12.4
# vs CUDA 12.9 libcu++: "the global scope has no islessgreater"). Window 1
# worked around that by building the extension in the container and reusing the
# .so on the host -- but torch keys its ninja command line by the PYTHON-include
# path, which resolves differently in the container (/spinning/subvol-.../usr...
# via the alias tree) than on the host (/usr/include/python3.12), so torch
# regenerates build.ninja, ninja sees a different command hash, and rebuilds --
# straight back into the broken host toolchain. The Docker image sidesteps the
# whole trap: its paths are identical on every run (/wt, /model, /extcache), it
# carries a compiler that builds the extension, and #369 already proved a full
# bar1 serving boot in it (world/tp/dcp all ACHIEVED=bar1). This is the robust
# serving route.
#
# WHY THE CACHES ARE MOUNTED. The container is --rm, so anything it writes to an
# unmounted path is gone at teardown. Three caches must survive across the four
# boots or every boot pays the full cold capture (>18 min in #369):
#   /extcache       -> TORCH_EXTENSIONS_DIR : the bar1 JIT .so (agent-369 warm)
#   /root/.cache    -> flashinfer / nvrtc / inductor
#   /root/.triton   -> triton autotune
# Boot 1 fills them cold; boots 2-4 read them. The boot-to-ready time is
# recorded per arm, which is the direct measurement of the warm-cache question.
#
# WHY INT8 INJECTS A WHEEL. The image ships the PyPI sgl_kernel, which has no
# sm120 INT8 arm (fork=0 pypi=1 in common_ops.abi3.so), so an INT8-W8A8
# checkpoint would crash the 5090 rank at its first forward. The fork wheel
# (abi3, installs into the image's py3.12) is force-reinstalled before launch on
# INT8 arms only. FP8 arms leave the image untouched.
#
# Usage: run_eager_arm.sh <arm> <model-subdir> <mlp|auto> <reserve> <fmt> <transport>
#   fmt       = fp8 | int8       (int8 triggers the fork-wheel inject)
#   transport = bar1 | nccl      (bar1 -> barlink direct path; nccl -> baseline)
#
# EAGER by construction (--disable-cuda-graph). #366 window-2 established that
# bar1 + CUDA graphs + NEXTN cold-captures for >35 min on this 27B FP8 vehicle
# because the 5090 has no tuned W8A8 block-fp8 configs (every GEMM shape falls
# to a cold default triton autotune -- the #255/#368 config gap). That does not
# fit four boots, let alone eight. Running EAGER skips capture entirely, so both
# transports are measured in the same regime and the bar1-vs-NCCL question is
# answered honestly. This is NOT comparable to #354's graph-mode NCCL numbers;
# it is a self-contained eager A/B, and the table labels every cell 'eager'.
set -uo pipefail

ARM="${1:?arm}"; MODEL_SUB="${2:?model subdir}"; MLP="${3:?mlp or auto}"
RESERVE="${4:?reserve list}"; FMT="${5:?fp8|int8}"; TRANSPORT="${6:?bar1|nccl}"

HOST="${BAR1_HOST:-192.168.0.1}"
KEY="${BAR1_HOST_KEY:-/root/.ssh/id_root@proxmox}"
SUB="${BAR1_HOST_SUBVOL:-/spinning/subvol-999-disk-0}"
IMG="${BAR1_IMG:-htsglang:cu130-nccl2307}"
PORT="${PORT:-30366}"
POINT_S="${POINT_S:-12}"
READY_TRIES="${READY_TRIES:-200}"

OUT="${OUT:-/spinning/gpu-battery-results/2026-08-01_366_bar1_formats}"
WT=/spinning/wt-final
VENV=/spinning/htsglang-gpu/.venv
H_OUT="$SUB$OUT"
H_WT="$SUB$WT"
H_MODEL="$SUB/spinning/llm_stuff/club-3090/models-cache/$MODEL_SUB"
H_NVSRC="$SUB/spinning/nvidia-open-595"
H_WHEEL="$SUB/spinning/wt-327a-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl"
CN="a366_$ARM"
H_LOG="/root/battery-bar1/366d.$ARM.log"
# Host python for the pure-stdlib driver scripts (s12/s14): the venv's
# bin/python3.12 is a dangling symlink on the host, but the scripts import only
# the standard library, so the host's own python3 runs them unchanged.
H_DRVPY="${H_DRVPY:-/usr/bin/python3}"

mkdir -p "$OUT"
hssh() { timeout "${1:?}" ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$HOST" "$2"; }

MLP_FLAG=""; [ "$MLP" != auto ] && MLP_FLAG="--rank-mlp-ratio $MLP"
# INT8: force-reinstall the fork sgl_kernel wheel inside the container first.
# No nested single quotes here: the whole LAUNCH is passed to docker as
# -c '$LAUNCH', and a single quote inside INJECT would close that wrapper and
# break `docker run` before the container ever starts (observed: INT8 arms
# died in 2-3 s with "No such container"). Keep it quote-free.
INJECT=""
INT8_LIBS=""
if [ "$FMT" = int8 ]; then
    INJECT="pip install --force-reinstall --no-deps /wheel/$(basename "$H_WHEEL") >/tmp/wheel.log 2>&1;"
    # The fork sgl_kernel wheel was built against CUDA 12.9, so its
    # common_ops.abi3.so links the whole CUDA-12 runtime set (libcudart.so.12,
    # libcublas.so.12, libcublasLt.so.12, ...). The image is cu13 and has none
    # of them, so installing the wheel BREAKS sgl_kernel outright: common_ops
    # fails to import, sglang falls back to "no int8_scaled_mm for this
    # device", and every INT8 rank dies at model load. /spinning/cu12libs is a
    # directory of symlinks to every *.so.12* the CT999 venv ships (targets
    # written with the host's subvol prefix so they resolve on both sides).
    # It is APPENDED to LD_LIBRARY_PATH, never prepended: the SONAMEs differ
    # from the image's .so.13, so torch keeps using its own cu13 libs and only
    # the fork wheel resolves against these.
    # The image's own default must be preserved verbatim and /cublas12 only
    # APPENDED -- overriding LD_LIBRARY_PATH with a hand-written list would
    # drop the image's cu13 torch libs and break the boot for a different
    # reason than the one being fixed.
    IMG_LDPATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
    INT8_LIBS="-v $SUB/spinning/cu12libs:/cu12libs:ro -v $SUB$VENV:$SUB$VENV:ro -e LD_LIBRARY_PATH=$IMG_LDPATH:/cu12libs"
fi

echo "=== $(date -u +%H:%M:%SZ) arm=$ARM fmt=$FMT transport=$TRANSPORT model=$MODEL_SUB mlp=$MLP reserve=$RESERVE ==="

# In-container launch command. --network host so the host loopback reaches it.
# --disable-cuda-graph: EAGER regime, no capture (see the header).
LAUNCH="cd /wt && $INJECT python3 -m sglang.launch_server \
  --model-path /model --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto \
  --rank-auto-reserve-mib $RESERVE $MLP_FLAG \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 --disable-cuda-graph \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics --host 127.0.0.1 --port $PORT"

# Transport-conditional docker args. bar1 needs the holder device + the barlink
# env; nccl is the stock path (barlink OFF) and needs neither the device nor
# SYS_ADMIN. Both are eager.
if [ "$TRANSPORT" = bar1 ]; then
    DEV_ARGS="--device /dev/dmabuf_holder --cap-add SYS_ADMIN --security-opt apparmor=unconfined -v /sys:/sys"
    BARLINK_ENV="-e SGLANG_BARLINK_BAR1_NV_SOURCE=/nvsrc \
  -e SGLANG_BARLINK=1 -e SGLANG_BARLINK_TRANSPORT=bar1 -e SGLANG_BARLINK_GRAPH_ENABLE=0 \
  -e SGLANG_BARLINK_BAR1_WINDOW_MIB=64 -e SGLANG_BARLINK_BAR1_WINDOW_MIB_DCP=32"
else
    DEV_ARGS=""
    BARLINK_ENV=""   # barlink unset -> stock NCCL TP collectives
fi

BOOT_T0=$(date +%s)
hssh 120 "mkdir -p /root/battery-bar1 /root/battery-bar1/dockercache /root/battery-bar1/tritoncache /root/battery-bar1/extcache_docker
docker rm -f $CN >/dev/null 2>&1 || true
docker run -d --rm --name $CN --network host --gpus all \
  $DEV_ARGS --shm-size=2g \
  -v $H_WT:/wt:ro -v $H_MODEL:/model:ro -v $H_NVSRC:/nvsrc:ro \
  -v $(dirname "$H_WHEEL"):/wheel:ro \
  -v /root/battery-bar1/extcache_docker:/extcache \
  -v /root/battery-bar1/dockercache:/root/.cache \
  -v /root/battery-bar1/tritoncache:/root/.triton \
  -e PYTHONPATH=/wt/python -e TORCH_EXTENSIONS_DIR=/extcache \
  -e TORCH_CUDA_ARCH_LIST='8.6;12.0' \
  $INT8_LIBS \
  $BARLINK_ENV \
  -e SGLANG_UNEVEN_DCP=1 -e SGLANG_UNEVEN_DCP_WEIGHTED=1 -e SGLANG_MAMBA_SSM_DTYPE=bfloat16 \
  --entrypoint bash $IMG -c '$LAUNCH'
echo started" || { echo "DOCKER RUN FAILED arm=$ARM"; exit 2; }

# Stream container logs to a host file for readiness + the gate + tick parsing.
# setsid + own session so it survives the ssh call returning; pid saved for
# teardown. s14 parses the Decode-batch ticks out of this file during the
# measured window, so it must keep growing for the whole arm.
hssh 30 "setsid bash -c 'docker logs -f $CN > $H_LOG 2>&1' </dev/null >/dev/null 2>&1 & echo \$! > /root/battery-bar1/366d.$ARM.logpid; echo logfollower \$(cat /root/battery-bar1/366d.$ARM.logpid)" || true

UP=0
for _ in $(seq 1 "$READY_TRIES"); do
    R=$(hssh 30 "curl -s -m 5 http://127.0.0.1:$PORT/health_generate >/dev/null 2>&1 && echo UP || echo NO; \
                 docker inspect -f '{{.State.Running}}' $CN 2>/dev/null || echo NOCON")
    case "$R" in
        *UP*)      UP=1; break ;;
        *"false"*|*NOCON*) echo "CONTAINER EXITED arm=$ARM"; break ;;
    esac
    sleep 10
done
BOOT_SECS=$(( $(date +%s) - BOOT_T0 ))
echo "arm=$ARM boot_to_ready_seconds=$BOOT_SECS" | tee "$OUT/boot_secs_$ARM.txt"
if [ "$UP" != 1 ]; then
    echo "BOOT FAILED arm=$ARM -- log tail:"
    hssh 60 "grep -nE 'ACHIEVED|out of memory|OutOfMemory|Traceback|Error|int8_scaled_mm|Bar1' $H_LOG | tail -25"
    hssh 60 "tail -30 $H_LOG"
    hssh 60 "docker rm -f $CN >/dev/null 2>&1 || true"
    exit 2
fi
echo "arm=$ARM UP"

# --- THE GATE (transport-aware) ---------------------------------------------
# bar1: every barlink group must log ACHIEVED=bar1 -- a fallback to gloo/device
#       makes the run mixed and it must not be reported as a bar1 number.
# nccl: the OPPOSITE gate -- there must be NO barlink group line at all, so the
#       baseline is genuinely stock NCCL and not an accidental barlink run.
hssh 60 "grep -E 'barlink enabled for group' $H_LOG" > "$OUT/gate_$ARM.txt" 2>&1
cat "$OUT/gate_$ARM.txt"
if [ "$TRANSPORT" = bar1 ]; then
    NGROUP=$(grep -c "ACHIEVED=" "$OUT/gate_$ARM.txt" 2>/dev/null || echo 0)
    NBAD=$(grep -E "ACHIEVED=" "$OUT/gate_$ARM.txt" | grep -vc "ACHIEVED=bar1" || true)
    echo "GATE arm=$ARM groups=$NGROUP not-bar1=$NBAD"
    if [ "$NGROUP" -eq 0 ] || [ "$NBAD" -ne 0 ]; then
        echo "GATE FAILED arm=$ARM -- not a clean bar1 run, not measuring."
        hssh 60 "docker rm -f $CN >/dev/null 2>&1 || true"; exit 3
    fi
    echo "GATE PASSED arm=$ARM (all $NGROUP group(s) ACHIEVED=bar1)"
else
    NBAR=$(grep -c "barlink enabled" "$OUT/gate_$ARM.txt" 2>/dev/null || echo 0)
    echo "GATE arm=$ARM barlink_lines=$NBAR (expect 0 for NCCL)"
    if [ "$NBAR" -ne 0 ]; then
        echo "GATE FAILED arm=$ARM -- NCCL baseline unexpectedly ran barlink."
        hssh 60 "docker rm -f $CN >/dev/null 2>&1 || true"; exit 3
    fi
    echo "GATE PASSED arm=$ARM (stock NCCL, no barlink group)"
fi

# --- the four points (prefill from prefopt boots, decode from auto boots) ----
for N in 1 8; do
    hssh 900 "'$H_DRVPY' '$H_WT/scripts/gpu_battery/s12_prefill_kurve.py' --mode messen \
        --port $PORT --out-dir '$H_OUT' --arm '$ARM' --sessions $N --folge $N \
        --point-seconds $POINT_S --warmup-seconds 6 --prompt-tokens 2048 \
        --with-decode 0 --server-log '$H_LOG'" >> "$OUT/messen_$ARM.log" 2>&1
    echo "  prefill s=$N rc=$?"
done
for B in 1 8; do
    hssh 900 "'$H_DRVPY' '$H_WT/scripts/gpu_battery/s14_decode_punkt.py' \
        --port $PORT --out-dir '$H_OUT' --arm '$ARM' --bs $B --folge $B \
        --context-tokens 2048 --model-context-tokens 32768 --ramp-seconds 6 \
        --window-seconds $POINT_S --server-log '$H_LOG'" >> "$OUT/messen_$ARM.log" 2>&1
    echo "  decode bs=$B rc=$?"
done

hssh 60 "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader" > "$OUT/vram_$ARM.txt" 2>&1
cat "$OUT/vram_$ARM.txt"
hssh 60 "grep -E 'ACHIEVED|derived memory budgets|MLP vector|max_total_num_tokens=|CHOSEN' $H_LOG | head -20" > "$OUT/plan_$ARM.txt" 2>&1
hssh 120 "kill \$(cat /root/battery-bar1/366d.$ARM.logpid 2>/dev/null) 2>/dev/null; \
          docker stop -t 20 $CN >/dev/null 2>&1; docker rm -f $CN >/dev/null 2>&1 || true"
echo "arm=$ARM DONE"
