#!/bin/bash
# #655 finalize: pin the chosen context, then prove the stack at that setting.
#
#   finalize655.sh <CTX>
#
# Order matters. The temporary 50-ctxtest.conf written by ctxwall655.sh is
# removed FIRST, so the value that ends up serving is the one written into
# 40-kt655.conf and nothing shadows it. Then:
#   1. warm the service (a cold load is ~195 s and would otherwise be counted
#      as prefill time in the probe -- that is exactly how the ctxwall probes
#      ended up reporting a nonsense 4.8 tok/s prefill)
#   2. a WARM 10k needle probe, which is the honest prefill/decode measurement
#   3. the coding agent end to end, as user efeu
set -u
CTX_FINAL=${1:?usage: finalize655.sh <CTX>}
TS=$(date +%H%M%S)
LOG=/root/651-p2/results/finalize655_${TS}.log
exec > >(tee -a "$LOG") 2>&1

D=/etc/systemd/system/htsglang-ondemand.service.d
mark() { echo "[$(date -Is)] $*"; }

mark "finalizing at CTX=$CTX_FINAL"
rm -f "$D/50-ctxtest.conf"
# Rewrite only the CTX line in the kt drop-in; the documentation around it stays.
sed -i "s/^Environment=CTX=.*/Environment=CTX=${CTX_FINAL}/" "$D/40-kt655.conf"
grep -n "^Environment=" "$D/40-kt655.conf"
systemctl daemon-reload
systemctl restart htsglang-ondemand
sleep 3

mark "warming (cold load ~195 s, kept out of the probe timings)"
curl -sS -m 1800 -X POST localhost:31651/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen36-35b-a3b","messages":[{"role":"user","content":"What is 17 times 23? Answer with just the number."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | head -c 400
echo
mark "warm. status: $(curl -s -m 10 localhost:31651/ondemand/status)"
mark "resets: $(dmesg -T | grep -c 'GPU reset(')"

mark "=== 10k needle probe, WARM ==="
R0=$(dmesg -T | grep -c 'GPU reset(')
timeout 1800 python3 /root/651-p2/scripts/needle655.py 10000 \
  "/root/651-p2/results/needle655_10k_final_${TS}.txt"
mark "10k probe rc=$? resets $R0 -> $(dmesg -T | grep -c 'GPU reset(')"

mark "=== coding agent end to end ==="
/root/651-p2/scripts/accept_omp655b.sh
mark "agent rc=$?"

mark "final status: $(curl -s -m 10 localhost:31651/ondemand/status)"
mark "final resets: $(dmesg -T | grep -c 'GPU reset(')"
mark "log=$LOG"
