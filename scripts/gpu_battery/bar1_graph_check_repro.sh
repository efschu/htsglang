#!/usr/bin/env bash
# Reproduce the bar1 CUDA-graph gate run on the PVE host and capture stacks
# from EVERY process in the tree before anything is killed (#369).
#
# WHY THIS SCRIPT EXISTS. `benchmark/bar1_graph_check.py` hangs on the host
# (agent-366: ~3.5 min, 0% GPU, spawn parent in sigsuspend, workers in
# poll/wait). A hang is only diagnosable from the stacks of ALL ranks at the
# same moment -- one rank's stack says where it waits, and only the others
# say why nobody arrives. So: bounded run, stack sweep over the whole process
# tree while it is still wedged, then kill our own process group and nothing
# else.
#
# WHY IT RUNS ON THE HOST. bar1 needs /dev/dmabuf_holder (char 10:267); CT999
# has no device-cgroup entry for it and cannot mknod (#361). The container
# drives, the host executes, the artifacts land back in the container.
#
# Usage: bar1_graph_check_repro.sh <out-dir> [devs] [cases] [budget-s]
#   devs   comma-separated CUDA device list, default 0,1,2
#   cases  comma-separated case names, default all
set -uo pipefail

OUT="${1:?output directory}"
DEVS="${2:-0,1,2}"
CASES="${3:-}"
BUDGET="${4:-420}"

HOST="${BAR1_HOST:-192.168.0.1}"
KEY="${BAR1_HOST_KEY:-/root/.ssh/id_root@proxmox}"
SUB="${BAR1_HOST_SUBVOL:-/spinning/subvol-999-disk-0}"

WT=/spinning/wt-369-bar1graph
VENV=/spinning/htsglang-gpu/.venv
H_WT="$SUB$WT"
H_SP="$SUB$VENV/lib/python3.12/site-packages"
H_SHIM=/root/battery-bar1/venvshim
H_PY="$H_SHIM/bin/python3.12"
# The SHARED cache, not the host one: it is the only tree that carries both
# barlink_bar1_ext and barlink_bar1_dmabuf_ext, built in the container under
# the host's path spelling (torch keys ninja by path string).
H_EXTCACHE="$SUB/spinning/barlink_extcache_shared"
H_NVSRC="$SUB/spinning/nvidia-open-595"
H_LOG=/root/battery-bar1/369.gate.log
H_PIDF=/root/battery-bar1/369.gate.pid
PORT="${PORT:-29700}"

mkdir -p "$OUT"

# Every ssh is bounded. A hung call must never wedge the caller unseen.
hssh() {
    timeout "${1:?timeout}" ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "root@$HOST" "$2"
}

echo "=== $(date -u +%H:%M:%SZ) #369 gate repro: devs=$DEVS cases=${CASES:-<all>} budget=${BUDGET}s ==="

ARGS="$DEVS $PORT"
[ -n "$CASES" ] && ARGS="$ARGS $CASES"

START=$(cat <<EOF
mkdir -p /root/battery-bar1
rm -f "$H_LOG" "$H_PIDF"
cd "$H_WT"
export PYTHONPATH="$H_WT/python:$H_SP"
export LD_LIBRARY_PATH="$H_SP/nvidia/cu13/lib"
export PATH="$SUB/usr/local/cuda-12.9/bin:\$PATH"
export CUDA_HOME="$SUB/usr/local/cuda-12.9"
export TORCH_EXTENSIONS_DIR="$H_EXTCACHE"
export SGLANG_BARLINK=1
export SGLANG_BARLINK_TRANSPORT=bar1
export SGLANG_BARLINK_BAR1_NV_SOURCE="$H_NVSRC"
setsid "$H_PY" "$H_WT/benchmark/bar1_graph_check.py" $ARGS > "$H_LOG" 2>&1 &
echo \$! > "$H_PIDF"
echo "started pid \$(cat $H_PIDF)"
EOF
)
hssh 120 "$START" || { echo "START SSH FAILED"; exit 2; }

# Bounded progress poll. "Progress" is the log growing; a case boundary is the
# only thing that matters for attributing a hang to a case.
LAST=""; STALL=0; DONE=0; SWEPT=0
for i in $(seq 1 $((BUDGET / 10))); do
    sleep 10
    R=$(hssh 30 "wc -c < '$H_LOG' 2>/dev/null || echo 0; \
                 kill -0 \$(cat $H_PIDF 2>/dev/null) 2>/dev/null && echo RUN || echo GONE; \
                 grep -c '^--- case' '$H_LOG' 2>/dev/null || echo 0")
    SIZE=$(echo "$R" | sed -n 1p); STATE=$(echo "$R" | sed -n 2p); NCASE=$(echo "$R" | sed -n 3p)
    if [ "$STATE" = "GONE" ]; then DONE=1; echo "t=$((i*10))s: process exited"; break; fi
    if [ "$SIZE" = "$LAST" ]; then STALL=$((STALL + 10)); else STALL=0; fi
    LAST="$SIZE"
    echo "t=$((i*10))s log=${SIZE}B cases_started=$NCASE stalled=${STALL}s"
    # One sweep, once, after the run has clearly stopped moving. Doing it
    # while wedged is the whole point -- a stack taken after the kill says
    # nothing.
    if [ "$STALL" -ge 60 ] && [ "$SWEPT" -eq 0 ]; then
        SWEPT=1
        echo "--- stalled ${STALL}s: sweeping stacks over the whole tree ---"
        hssh 180 "
            P=\$(cat $H_PIDF)
            TREE=\$(pstree -p \$P 2>/dev/null | grep -oE '\([0-9]+\)' | tr -d '()' )
            echo \"=== tree of \$P: \$TREE\"
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
            for q in \$TREE; do
                echo \"########## pid \$q ##########\"
                echo \"--- cmdline: \$(tr '\\\\0' ' ' < /proc/\$q/cmdline 2>/dev/null)\"
                echo \"--- state:   \$(awk '/^State:/{print \$2,\$3}' /proc/\$q/status 2>/dev/null)\"
                echo \"--- wchan:   \$(cat /proc/\$q/wchan 2>/dev/null)\"
                echo \"--- syscall: \$(cat /proc/\$q/syscall 2>/dev/null)\"
                echo \"--- py-spy:\"
                timeout 25 '$SUB$VENV/bin/py-spy' dump --pid \$q --nonblocking 2>&1 | head -60
            done
        " > "$OUT/stacks_$(date -u +%H%M%SZ).txt" 2>&1
        echo "    stacks -> $OUT/"
    fi
done

hssh 60 "tail -60 '$H_LOG'" > "$OUT/gate_repro.tail.log" 2>&1
hssh 60 "cat '$H_LOG'" > "$OUT/gate_repro.full.log" 2>&1

if [ "$DONE" != 1 ]; then
    echo "=== budget spent, killing OUR process group only ==="
    hssh 90 "P=\$(cat $H_PIDF); kill -TERM -- -\$P 2>/dev/null; sleep 8; \
             kill -KILL -- -\$P 2>/dev/null; rm -f $H_PIDF; sleep 2; \
             nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
fi

echo "=== summary ==="
grep -E "^--- case|=> PASSED|=> FAILED|^  (PASSED|FAILED)" "$OUT/gate_repro.full.log" 2>/dev/null | tail -30
