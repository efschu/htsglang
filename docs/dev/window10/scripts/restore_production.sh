#!/usr/bin/env bash
# Close the window: production back up VERBATIM, then prove it -- health 200 is
# not proof that the server can generate, so a smoke generation is part of the
# check, and the holder is only released after both pass.
set -u
D=/spinning/gpu-battery-results/2026-08-05_window10

echo "=== 1. nothing of ours may still hold a card ==="
for f in "$D"/raw/pgid_*; do
  [ -e "$f" ] || continue
  P=$(cat "$f")
  kill -TERM -- -"$P" 2>/dev/null && echo "  TERM -$P"
done
sleep 8
for f in "$D"/raw/pgid_*; do
  [ -e "$f" ] || continue
  kill -KILL -- -"$(cat "$f")" 2>/dev/null
done
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== 2. stop the heartbeat BEFORE anything else touches the holder ==="
pkill -f "while true; do touch /spinning/gpu-arb/holder" && echo "  heartbeat stopped"
pkill -f "nvidia-smi --query-gpu=index,power.limit" 2>/dev/null
sleep 2

echo "=== 3. boot production verbatim ==="
bash /root/bin/start-serving-30030.sh
sleep 5

echo "=== 4. wait for health 200 ==="
UP=0
for i in $(seq 1 90); do
  C=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:30030/health || true)
  if [ "$C" = "200" ]; then UP=1; echo "  health 200 after ~$((i*5))s"; break; fi
  sleep 5
done
[ "$UP" = 1 ] || { echo "PRODUCTION DID NOT COME UP -- log tail:"; tail -40 /spinning/serving-30030.boot.log; exit 2; }

echo "=== 5. smoke generation (content must be sane, not merely 200) ==="
curl -s -m 120 http://127.0.0.1:30030/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B","messages":[{"role":"user","content":"Was ist die Hauptstadt von Norwegen? Antworte in genau einem Wort."}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("SMOKE:", repr(d["choices"][0]["message"]["content"]))'
curl -s -m 120 http://127.0.0.1:30030/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B","messages":[{"role":"user","content":"Wieviel ist 14 mal 3? Nur die Zahl."}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("SMOKE:", repr(d["choices"][0]["message"]["content"]))'

echo "=== 6. corridor ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
grep -o "max_total_num_tokens=[0-9]*" /spinning/serving-30030.boot.log | tail -1
