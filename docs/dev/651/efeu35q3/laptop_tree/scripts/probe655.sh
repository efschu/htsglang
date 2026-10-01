#!/bin/bash
# #655 correctness gate: two prompts of different shape against a backend port.
P="${1:-31661}"; T="${2:-120}"
ask() {
  printf '{"model":"qwen36-35b-a3b","messages":[{"role":"user","content":%s}],"max_tokens":%s,"temperature":0}' "$1" "$2" > /tmp/probe655.json
  local r
  r=$(curl -s -m "$T" -w '\nHTTP=%{http_code} T=%{time_total}s' "localhost:$P/v1/chat/completions" \
        -H "Content-Type: application/json" -d @/tmp/probe655.json)
  echo "$r" | python3 -c '
import sys,json
raw=sys.stdin.read()
tail=raw.rsplit("\n",1)[-1]
body=raw.rsplit("\n",1)[0]
try:
    d=json.loads(body); print("TEXT:", repr(d["choices"][0]["message"]["content"])[:600])
except Exception as e:
    print("NO-JSON:", body[:300])
print(tail)'
}
echo "--- Q1 fibonacci"
ask '"List the first 10 Fibonacci numbers, comma separated."' 60
echo "--- Q2 sea"
ask '"Describe the sea in exactly 3 sentences."' 120
