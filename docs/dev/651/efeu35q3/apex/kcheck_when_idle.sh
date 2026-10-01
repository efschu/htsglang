#!/bin/bash
# efeu-TP14: APEX kernel check, gated: download done + live service idle >= 300 s
# with inflight 0. The check itself also stops between ops if a user request
# arrives. Results: results/apex/kcheck_*.txt
set -u
cd /root/efeu35q3
mkdir -p results/apex
st() { curl -s -m5 localhost:31651/ondemand/status; }
quiet() { st | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d["inflight"]==0 and d["idle_seconds"]>=300 else 1)' 2>/dev/null; }
until grep -q "DL-DONE" logs/dl_apex.log; do sleep 20; done
A=models_apex/Qwen3.8-35B-A3B-Distill.APEX-I-MiniPlus-V2.1-Abliterated.gguf
M=models_apex/mtp-Qwen3.8-35B-A3B-Distill-Q4_0.gguf
export HSA_OVERRIDE_GFX_VERSION=11.0.0 GGUF_EXT_DIR=/root/efeu35q3/ext_v2
PY=/root/lh/venv/bin/python
run() {  # $1 label, rest args
  local L=$1; shift
  until quiet; do sleep 30; done
  echo "=== $L $(date -Is) $(st)" >> results/apex/kcheck_$L.txt
  systemd-run --scope -q -p MemoryMax=4G nice -n 10 $PY apex/apex_kernel_check.py "$@" >> results/apex/kcheck_$L.txt 2>&1
  echo "=== rc=$? $(date -Is)" >> results/apex/kcheck_$L.txt
}
run quiet $A $M
# ddr5load run REMOVED: its 6 hogs caused the 21:16 global OOM (user session killed)
echo "KCHECK DONE $(date -Is)" >> results/apex/kcheck_done.txt
