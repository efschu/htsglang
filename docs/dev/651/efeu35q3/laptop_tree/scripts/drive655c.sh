#!/bin/bash
# Warm the service, then run the minimal coding-agent acceptance.
#
# Warming separately matters: a cold load is ~195 s and the agent's own HTTP
# client would otherwise sit through it on its first turn and may time out.
set -u
LOG=/root/651-p2/results/drive655c_$(date +%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
mark() { echo "[$(date -Is)] $*"; }

mark "CTX now: $(systemctl show htsglang-ondemand -p Environment | tr ' ' '\n' | grep '^CTX=')"
mark "warming (cold load ~195 s)"
curl -sS -m 1800 -X POST localhost:31651/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen36-35b-a3b","messages":[{"role":"user","content":"What is 17 times 23? Answer with just the number."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | head -c 300
echo
mark "status: $(curl -s -m 10 localhost:31651/ondemand/status)"
mark "resets after warm: $(dmesg -T | grep -c 'GPU reset(')"

mark "=== coding agent acceptance ==="
/root/651-p2/scripts/accept_omp655c.sh
mark "agent rc=$?"
mark "final resets: $(dmesg -T | grep -c 'GPU reset(')"
mark "log=$LOG"
