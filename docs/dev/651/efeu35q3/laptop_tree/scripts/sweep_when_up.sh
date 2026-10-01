#!/bin/bash
# #651: get ONE healthy Q4 load, then immediately run the prefill sweep.
#
# Loads on this machine are non-deterministic (KV lottery, scratch shortfall),
# so the sweep has to be opportunistic: keep asking for the model until a load
# takes, then spend that window on the measurement rather than on more retries.
set -u
BASE=http://127.0.0.1:31651
LOG=/root/651-p2/results/sweep_when_up_$(date +%H%M%S).txt
exec > >(tee -a "$LOG") 2>&1

for try in 1 2 3 4 5 6; do
  echo "=== wake attempt $try $(date +%T) ==="
  curl -s -m 900 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen36-35b-a3b","messages":[{"role":"user","content":"hi"}],
         "chat_template_kwargs":{"enable_thinking":false},"max_tokens":4}' \
    -o /tmp/wake.json -w "  http=%{http_code}\n" || true
  ST=$(curl -s -m 8 "$BASE/ondemand/status" | python3 -c 'import json,sys;print(json.load(sys.stdin)["state"])' 2>/dev/null || echo unknown)
  echo "  state=$ST"
  if [ "$ST" = "up" ]; then
    echo "=== MODEL UP -- running prefill sweep $(date +%T) ==="
    bash /root/651-p2/scripts/prefill_sweep.sh
    exit $?
  fi
  sleep 5
done
echo "=== NEVER GOT A HEALTHY LOAD IN 6 ATTEMPTS ==="
exit 2
