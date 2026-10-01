#!/bin/bash
# efeu-TP14: PP2 over iGPU + CPU on the SAME GGUF (llama.cpp HIP build, -ngl k).
# -ngl 41 = everything on the iGPU (41 = 40 layers + output); smaller k moves
# the first 41-k layers to the CPU stage. --mmap 1: the CPU stage reads its
# layers straight out of the page cache (no copy); the iGPU stage holds its own
# copy in device memory. Coherence: greedy text of a fixed prompt per k vs k=41
# and vs the CPU-only run (k=0).  Balanced profile; sglang must be stopped.
set -u
cd /root/efeu35q3
B=/root/efeu35q3/llama-hip-build/bin
M=models/Qwen3.8-35B-A3B-Q3_K_M.gguf
OUT=results/pp2
mkdir -p $OUT
export HSA_OVERRIDE_GFX_VERSION=11.0.0
powerprofilesctl set balanced
echo "=== pp2 llama.cpp $(date -Is) resets=$(dmesg | grep -c 'GPU reset')"
for NGL in ${NGLS:-41 37 33 29 25 0}; do
  echo "--- ngl=$NGL"
  timeout 1800 $B/llama-bench -m $M -ngl $NGL -t 8 -p 2048 -n 128 -b 512 -ub 512 -r 2 -fa 0 -mmp 1 -o json \
    > $OUT/bench_ngl$NGL.json 2> $OUT/bench_ngl$NGL.err
  python3 - "$OUT/bench_ngl$NGL.json" <<'PY'
import json, sys
for r in json.load(open(sys.argv[1])):
    kind = "prefill" if r["n_prompt"] else "decode"
    print(f"   {kind:8s} n={r['n_prompt'] or r['n_gen']:5d} {r['avg_ts']:7.2f} tok/s (sd {r['stddev_ts']:.2f})")
PY
  # coherence: the probe_q38 prompts, greedy, against the sglang outputs
  $B/llama-server -m $M -ngl $NGL -t 8 --jinja -c 4096 --port 31690 --host 127.0.0.1 > $OUT/server_ngl$NGL.log 2>&1 &
  SP=$!
  for i in $(seq 1 120); do curl -sf -m 3 localhost:31690/health >/dev/null && break; sleep 3; done
  /root/lh/venv/bin/python llama_ref_compare.py 31690 results/probe_q38_outperm.json --json $OUT/coh_ngl$NGL.json | tail -1
  kill $SP; wait $SP 2>/dev/null
  echo "   resets=$(dmesg | grep -c 'GPU reset')"
done
powerprofilesctl set power-saver
echo "=== PP2 DONE $(date -Is)"
