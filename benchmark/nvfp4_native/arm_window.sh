#!/usr/bin/env bash
# N4B micro-bench window: card 1 (5090) and card 2 (3080) in parallel, then done.
# Usage: arm_window.sh <outdir>. Caller holds the gpuq window and releases it after.
set -uo pipefail
OUT=${1:?outdir}
mkdir -p "$OUT"
WT=/spinning/wt-27b-nvfp4-native
R=$WT/benchmark/nvfp4_native/run_bench.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# safety net: the two cards must be ours (<500 MiB used) and be what we think they are
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tee "$OUT/nvsmi_before.txt"
n1=$(nvidia-smi -i 1 --query-gpu=name --format=csv,noheader); n2=$(nvidia-smi -i 2 --query-gpu=name --format=csv,noheader)
u1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits); u2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
case "$n1" in *5090*) ;; *) echo "ABORT card1 is $n1"; exit 3;; esac
case "$n2" in *3080*) ;; *) echo "ABORT card2 is $n2"; exit 3;; esac
[ "$u1" -lt 500 ] && [ "$u2" -lt 500 ] || { echo "ABORT cards busy ($u1/$u2 MiB)"; exit 4; }
(
  CUDA_VISIBLE_DEVICES=1 timeout 60 $WT/.jit/mma/probe_sm120a > "$OUT/mma_int4_5090.txt" 2>&1; echo "5090 mma rc=$?" >> "$OUT/status.txt"
  CUDA_VISIBLE_DEVICES=1 timeout 780 "$R" --out "$OUT/b5090_lanes.json" --fi-backends cutlass,cudnn > "$OUT/b5090_lanes.log" 2>&1
  echo "5090 lanes rc=$?" >> "$OUT/status.txt"
  CUDA_VISIBLE_DEVICES=1 BENCH_SCRIPT=bench_layer_components.py timeout 240 "$R" --out "$OUT/b5090_comp.json" > "$OUT/b5090_comp.log" 2>&1
  echo "5090 comp rc=$?" >> "$OUT/status.txt"
  CUDA_VISIBLE_DEVICES=1 timeout 240 "$R" --out "$OUT/b5090_b12x.json" --lanes fi --fi-backends b12x --shapes P.gate_up,P.down,D.gate_up,D.down --ms 8,48,512,4096 --fp8-shapes none > "$OUT/b5090_b12x.log" 2>&1; echo "5090 b12x rc=$?" >> "$OUT/status.txt"
) &
P1=$!
(
  CUDA_VISIBLE_DEVICES=2 timeout 60 $WT/.jit/mma/probe_sm86 > "$OUT/mma_int4_3080.txt" 2>&1; echo "3080 mma rc=$?" >> "$OUT/status.txt"
  CUDA_VISIBLE_DEVICES=2 FLASHINFER_CUDA_ARCH_LIST=8.6 BENCH_SCRIPT=bench_layer_components.py timeout 300 "$R" --out "$OUT/b3080_comp.json" > "$OUT/b3080_comp.log" 2>&1
  echo "3080 comp rc=$?" >> "$OUT/status.txt"
  CUDA_VISIBLE_DEVICES=2 FLASHINFER_CUDA_ARCH_LIST=8.6 timeout 780 "$R" --out "$OUT/b3080_lanes.json" --shapes P.gate_up,P.down,D.gate_up,D.down --ms 1,8,16,48,512,2048 > "$OUT/b3080_lanes.log" 2>&1
  echo "3080 lanes rc=$?" >> "$OUT/status.txt"
) &
P2=$!
wait $P1 $P2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/nvsmi_after.txt"
echo "ARM DONE" >> "$OUT/status.txt"
