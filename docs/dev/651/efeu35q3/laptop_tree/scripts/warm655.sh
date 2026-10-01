#!/bin/bash
# Wake the on-demand service and record the first reply. Detached on purpose:
# a cold load is ~150 s and must not sit inside an interactive call.
OUT=/root/651-p2/logs/warm655_$(date +%H%M%S).json
curl -sS -m 1800 -X POST localhost:31651/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen36-35b-a3b\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23? Answer with just the number.\"}],\"max_tokens\":16,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  > "$OUT" 2>&1
echo "exit=$? out=$OUT" >> "$OUT"
