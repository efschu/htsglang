#!/bin/bash
# #655 correctness gate, thinking OFF so the token budget buys the ANSWER
# rather than a reasoning preamble.
P="${1:-31651}"
ask() {
  printf '{"model":"qwen36-35b-a3b","messages":[{"role":"user","content":%s}],"max_tokens":300,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' "$1" > /tmp/p655.json
  curl -s -m 600 "localhost:$P/v1/chat/completions" -H "Content-Type: application/json" -d @/tmp/p655.json \
  | python3 -c '
import sys,json
try:
    d=json.load(sys.stdin); print("ANSWER:", d["choices"][0]["message"]["content"].strip()[:700])
except Exception as e: print("ERR:", e)'
}
echo "== Q1 fibonacci"; ask '"List the first 10 Fibonacci numbers, comma separated. Answer only with the numbers."'
echo "== Q2 sea";       ask '"Describe the sea in exactly 3 sentences."'
echo "== Q3 arithmetic";ask '"What is 17 times 23? Answer with just the number."'
