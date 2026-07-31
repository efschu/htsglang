#!/usr/bin/env bash
# One #366 measurement ARM: boot Qwen3.6-27B TP=3 over the barlink BAR1 direct
# path ON THE PVE HOST, gate the transport, then take four throughput points.
#
# WHY THE HOST. bar1 needs /dev/dmabuf_holder (char 10:267). CT999 has no
# device-cgroup allow for it and cannot mknod, so inside the container the
# transport declines and lands on gloo -- #361 proved that today, and the
# decline now logs loudly instead of silently. A bar1 number can therefore
# only be produced on the host. Everything else stays where it belongs: this
# script runs IN THE CONTAINER, drives the host over bounded ssh, and the
# artifacts land in the container's results directory.
#
# WHY THE CONTAINER'S INTERPRETER, BEHIND A SHIM. The shared venv cannot run on
# the host: .venv/bin/python3.12 is a symlink to /bin/python3.12 and the host
# has only 3.11 and 3.13. Nothing is installed on the host to fix that -- the
# container's own interpreter at <subvol>/usr/bin/python3.12 executes natively
# on the host kernel and supplies torch 2.11.0+cu130, sglang and sgl_kernel.
# Running it bare is NOT enough though; see the H_SHIM comment below.
#
# WHY THE PATHS ARE DOUBLED. The container root is the host's
# /spinning/subvol-999-disk-0, so every container path has a host path. The
# server runs on the host and therefore needs HOST paths for the model, the
# driver headers and the repo; the measurement JSONL it writes goes to the
# host path of the container results directory, which is the same file.
#
# Usage: run_bar1_arm.sh <arm> <model-subdir> <mlp-vector|auto> <reserve-list>
set -uo pipefail

ARM="${1:?arm name}"
MODEL_SUB="${2:?model subdir under models-cache}"
MLP="${3:?mlp vector or 'auto'}"
RESERVE="${4:?rank-auto-reserve-mib list}"

HOST="${BAR1_HOST:-192.168.0.1}"
KEY="${BAR1_HOST_KEY:-/root/.ssh/id_root@proxmox}"
SUB="${BAR1_HOST_SUBVOL:-/spinning/subvol-999-disk-0}"

# Container paths.
OUT=/spinning/gpu-battery-results/2026-07-31_366_bar1_formats
WT=/spinning/wt-final
VENV=/spinning/htsglang-gpu/.venv
# Host views of the same things.
H_OUT="$SUB$OUT"
H_WT="$SUB$WT"
H_SP="$SUB$VENV/lib/python3.12/site-packages"
# The shared venv cannot be RUN on the host (its bin/python3.12 points at
# /bin/python3.12 and the host has only 3.11/3.13), and invoking the container's
# bare interpreter instead silently loses the venv's
# "include-system-site-packages = false": the container's
# /usr/local/lib/python3.12/dist-packages reappears on sys.path, flashinfer_cubin
# 0.6.12 becomes importable next to flashinfer 0.6.14, and the boot dies in
# flashinfer's version check -- a failure the container never sees.
# This shim is a venv-shaped directory (bin/python3.12 symlink + pyvenv.cfg with
# include-system-site-packages=false) in HOST-LOCAL scratch. It installs nothing:
# it only makes the interpreter apply the same isolation the container venv
# applies, so the host boot imports exactly the packages the container boot does.
H_SHIM="/root/battery-bar1/venvshim"
H_PY="$H_SHIM/bin/python3.12"
H_MODEL="$SUB/spinning/llm_stuff/club-3090/models-cache/$MODEL_SUB"
H_NVSRC="$SUB/spinning/nvidia-open-595"
# JIT cache on the pool, not on /root: rpool is at 95% with 11G free.
H_EXTCACHE="$SUB/spinning/barlink_extcache_host"
H_LOG="/root/battery-bar1/366.$ARM.log"

PORT="${PORT:-30366}"
POINT_S="${POINT_S:-12}"

mkdir -p "$OUT"

# One place, one policy: every ssh is bounded, so a hung call can never wedge
# the agent without anyone seeing it.
hssh() {
    timeout "${1:?timeout}" ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$HOST" "$2"
}

MLP_FLAG=""
[ "$MLP" != "auto" ] && MLP_FLAG="--rank-mlp-ratio $MLP"

echo "=== $(date -u +%H:%M:%SZ) arm=$ARM model=$MODEL_SUB mlp=$MLP reserve=$RESERVE ==="

# --- boot on the host -------------------------------------------------------
# --rank-tp-ratio auto (not auto-performance): with an explicit --rank-mlp-ratio
# the planner takes its pin path and skips the hardware probe anyway, and plain
# auto resolves to the same VRAM-auto vector #354's auto arm chose. That keeps
# the boot independent of the host's hw-profile cache, which is a DIFFERENT
# file from the container's (different HOME).
BOOT=$(cat <<EOF
mkdir -p /root/battery-bar1 "$H_EXTCACHE" "$H_SHIM/bin" "$H_SHIM/lib/python3.12/site-packages"
ln -sf "$SUB/usr/bin/python3.12" "$H_SHIM/bin/python3.12"
printf 'home = %s/usr/bin\ninclude-system-site-packages = false\nversion = 3.12.3\n' "$SUB" > "$H_SHIM/pyvenv.cfg"
cd "$H_WT"
export PYTHONPATH="$H_WT/python:$H_SP"
export LD_LIBRARY_PATH="$H_SP/nvidia/cu13/lib"
export PATH="$SUB/usr/local/cuda-12.9/bin:$SUB$VENV/bin:\$PATH"
export CUDA_HOME="$SUB/usr/local/cuda-12.9"
export TORCH_EXTENSIONS_DIR="$H_EXTCACHE"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_BARLINK=1
export SGLANG_BARLINK_TRANSPORT=bar1
export SGLANG_BARLINK_BAR1_NV_SOURCE="$H_NVSRC"
# Two communicator groups exist under SGLANG_UNEVEN_DCP (tp and dcp) and they
# share one aperture. The 3080s have 256 MiB of BAR1 gross; at the 96 MiB
# default the tp group takes its window first and dcp gets a bare ENOMEM from
# the holder and silently falls back -- which is exactly the mixed run the
# gate below exists to catch. 64 + 32 MiB fits both with room for RM's own
# use, and neither window throttles: the tp group's actual prefill payload is
# ~20 MiB (chunked_prefill_size 2048 x hidden 5120 x 2 B) and dcp carries less.
export SGLANG_BARLINK_BAR1_WINDOW_MIB=64
export SGLANG_BARLINK_BAR1_WINDOW_MIB_DCP=32
setsid "$H_PY" -m sglang.launch_server \
  --model-path "$H_MODEL" \
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto \
  --rank-auto-reserve-mib "$RESERVE" \
  $MLP_FLAG \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics \
  --host 127.0.0.1 --port $PORT \
  > "$H_LOG" 2>&1 &
echo \$! > /root/battery-bar1/366.$ARM.pid
echo "started pid \$(cat /root/battery-bar1/366.$ARM.pid)"
EOF
)
hssh 120 "$BOOT"
[ $? -ne 0 ] && { echo "BOOT SSH FAILED arm=$ARM"; exit 2; }

# Readiness, bounded, polled from the host over its own loopback. The JIT build
# of the BAR1 extension is paid on the FIRST arm only (shared cache), so this
# wait is deliberately long.
UP=0
for _ in $(seq 1 100); do
    R=$(hssh 30 "curl -s -m 5 http://127.0.0.1:$PORT/health_generate >/dev/null 2>&1 && echo UP || echo NO; \
                 kill -0 \$(cat /root/battery-bar1/366.$ARM.pid 2>/dev/null) 2>/dev/null || echo DEAD")
    case "$R" in
        *UP*)   UP=1; break ;;
        *DEAD*) echo "SERVER DIED arm=$ARM"; break ;;
    esac
    sleep 10
done
if [ "$UP" != 1 ]; then
    echo "BOOT FAILED arm=$ARM -- log tail:"
    hssh 60 "grep -nE 'ACHIEVED|barlink|out of memory|OutOfMemory|Traceback|Error' '$H_LOG' | tail -25"
    hssh 60 "tail -25 '$H_LOG'" | tail -25
    exit 2
fi
echo "arm=$ARM UP"

# --- THE GATE ---------------------------------------------------------------
# The whole reason #366 exists apart from #354. Every barlink group must say
# ACHIEVED=bar1. A group that fell back to gloo or device makes the run mixed,
# and a mixed run must not be reported as a bar1 number.
hssh 60 "grep -nE 'barlink (enabled for group|group)' '$H_LOG'" > "$OUT/gate_$ARM.txt" 2>&1
cat "$OUT/gate_$ARM.txt"
NGROUP=$(grep -c "ACHIEVED=" "$OUT/gate_$ARM.txt" 2>/dev/null || echo 0)
NBAR1=$(grep -c "ACHIEVED=bar1" "$OUT/gate_$ARM.txt" 2>/dev/null || echo 0)
NBAD=$(grep -E "ACHIEVED=" "$OUT/gate_$ARM.txt" | grep -vc "ACHIEVED=bar1" || true)
echo "GATE arm=$ARM groups=$NGROUP bar1=$NBAR1 not-bar1=$NBAD"
if [ "$NGROUP" -eq 0 ] || [ "$NBAD" -ne 0 ]; then
    echo "GATE FAILED arm=$ARM -- NOT measuring, this would not be a bar1 number."
    hssh 60 "'$SUB$VENV/bin/py-spy' dump --pid \$(cat /root/battery-bar1/366.$ARM.pid)" \
        > "$OUT/pyspy_$ARM.txt" 2>&1 || true
    hssh 60 "P=\$(cat /root/battery-bar1/366.$ARM.pid); kill -TERM -- -\$P 2>/dev/null; sleep 8; kill -KILL -- -\$P 2>/dev/null; rm -f /root/battery-bar1/366.$ARM.pid"
    exit 3
fi
echo "GATE PASSED arm=$ARM (all $NGROUP group(s) ACHIEVED=bar1)"

# --- the four points --------------------------------------------------------
# Prefill from the concentrated boots, decode from the VRAM-auto boots (the
# #354 phase recipe); this script takes all four on every arm and the table
# picks the phase-correct column, so a boot is never repeated for one number.
for N in 1 8; do
    hssh 900 "cd '$H_WT' && '$H_PY' '$H_WT/scripts/gpu_battery/s12_prefill_kurve.py' --mode messen \
        --port $PORT --out-dir '$H_OUT' --arm '$ARM' --sessions $N --folge $N \
        --point-seconds $POINT_S --warmup-seconds 6 --prompt-tokens 2048 \
        --with-decode 0 --server-log '$H_LOG'" >> "$OUT/messen_$ARM.log" 2>&1
    echo "  prefill s=$N rc=$?"
done
for B in 1 8; do
    hssh 900 "cd '$H_WT' && '$H_PY' '$H_WT/scripts/gpu_battery/s14_decode_punkt.py' \
        --port $PORT --out-dir '$H_OUT' --arm '$ARM' --bs $B --folge $B \
        --context-tokens 2048 --model-context-tokens 32768 --ramp-seconds 6 \
        --window-seconds $POINT_S --server-log '$H_LOG'" >> "$OUT/messen_$ARM.log" 2>&1
    echo "  decode bs=$B rc=$?"
done

# --- corridor + teardown ----------------------------------------------------
hssh 60 "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader" \
    > "$OUT/vram_$ARM.txt" 2>&1
cat "$OUT/vram_$ARM.txt"
hssh 60 "'$SUB$VENV/bin/py-spy' dump --pid \$(cat /root/battery-bar1/366.$ARM.pid)" \
    > "$OUT/pyspy_$ARM.txt" 2>&1 || true
hssh 60 "grep -E 'ACHIEVED|derived memory budgets|MLP vector|max_total_num_tokens=' '$H_LOG' | head -20" \
    > "$OUT/plan_$ARM.txt" 2>&1
hssh 120 "P=\$(cat /root/battery-bar1/366.$ARM.pid); kill -TERM -- -\$P 2>/dev/null; \
          for i in \$(seq 1 30); do kill -0 \$P 2>/dev/null || break; sleep 2; done; \
          kill -KILL -- -\$P 2>/dev/null; rm -f /root/battery-bar1/366.$ARM.pid"
echo "arm=$ARM DONE"
