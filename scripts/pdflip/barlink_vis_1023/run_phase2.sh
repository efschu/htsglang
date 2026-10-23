#!/bin/bash
# barlink-visibility-1023 PHASE 2 runner. Do NOT start before the caller says the cards are free.
# Books its own gpuq window (cards 0,1, 4g each), takes the card locks, sets MEASURE_WINDOW, reads
# (never writes) ACS/DevCtl on the host, runs the visibility probe once under the RM/UVM ioctl
# tracer and once plain, releases everything on exit.
# Usage: run_phase2.sh <label> [extra vis_probe args]
set -u
L=${1:-p2}; shift || true
D=$(cd "$(dirname "$0")" && pwd); OUT=/tmp/bvis_$L; mkdir -p "$OUT"
NVO=/spinning/nvidia-open-595; SP=/spinning/nvidia-smallbar-p2p
GPUQ=http://127.0.0.1:8770/api/v1; HOST="ssh -n -o BatchMode=yes proxmox"
/usr/local/cuda/bin/nvcc -O3 -std=c++14 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 \
  -I$SP -I$SP/sonden -I$NVO/src/common/sdk/nvidia/inc -I$NVO/kernel-open/common/inc \
  -I$NVO/src/nvidia/arch/nvalloc/unix/include -o "$OUT/vis_probe" "$D/vis_probe.cu" -lcuda 2>"$OUT/build.err" || { cat "$OUT/build.err"; exit 2; }
gcc -O2 -shared -fPIC -o "$OUT/rmtrace.so" "$D/rmtrace.c" -ldl -I$NVO/src/common/sdk/nvidia/inc -I$NVO/kernel-open/common/inc \
  -I$NVO/src/nvidia/arch/nvalloc/unix/include -I$NVO/kernel-open/nvidia-uvm || exit 2
R=$(curl -s -X POST $GPUQ/bookings -H 'Content-Type: application/json' \
  -d '{"owner":"barlink-visibility-1023","purpose":"barlink BAR1 visibility probe (sender/fabric/receiver)","cards":[0,1],"duration":"40m","mib":"4g"}')
ID=$(echo "$R" | python3 -c "import sys,json;print(json.load(sys.stdin).get('id',''))")
TOK=$(echo "$R" | python3 -c "import sys,json;print(json.load(sys.stdin).get('token',''))")
ST=$(echo "$R" | python3 -c "import sys,json;print(json.load(sys.stdin).get('state',''))")
echo "gpuq: $R" > "$OUT/gpuq.txt"
trap 'rmdir /tmp/gpu-card-0.lock /tmp/gpu-card-1.lock 2>/dev/null; rm -f /spinning/gpu-arb/MEASURE_WINDOW; [ -n "$ID" ] && curl -s -X DELETE -H "X-Gpuq-Token: $TOK" $GPUQ/bookings/$ID >/dev/null && echo "gpuq $ID released"' EXIT
[ "$ST" = running ] || { echo "window not running (state=$ST) -- not waiting, booking stays queued: $ID"; trap - EXIT; exit 3; }
mkdir /tmp/gpu-card-0.lock && mkdir /tmp/gpu-card-1.lock || { echo "card lock busy"; exit 2; }
echo "barlink-visibility-1023 $ID" > /spinning/gpu-arb/MEASURE_WINDOW
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_bus_id --format=csv > "$OUT/nvsmi_before.csv"
$HOST 'for p in 0e:04.0 0e:08.0; do echo -n "ACS $p=$(setpci -s $p ECAP_ACS+6.w) "; done; for g in 15:00.0 18:00.0; do echo -n "DevCtl $g=$(setpci -s $g CAP_EXP+8.w) "; done; echo' > "$OUT/pci_state.txt" 2>&1
cat "$OUT/pci_state.txt"
cd "$OUT"
# 1) mapping attributes as CUDA requests them (one short trial)
RMTRACE_LOG=$OUT/rmtrace.log LD_PRELOAD=$OUT/rmtrace.so timeout -k 5 120 ./vis_probe --arms=E --n=1 "$@" > "$OUT/trace_run.log" 2>&1
# 2) the decision matrix
timeout -k 5 1500 ./vis_probe --arms=ESXMRAFB --n=6 "$@" > "$OUT/matrix.log" 2>&1; echo "matrix rc=$?"
grep -E '^(ARM|PAIR)' "$OUT/matrix.log" | cut -c1-260
grep -E 'OS_DESCRIPTOR|MAP_MEMORY_DMA|MAP_EXTERNAL|cachingType|status' "$OUT/rmtrace.log" | head -40
