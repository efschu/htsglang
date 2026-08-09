#!/bin/bash
# #655 quality gate. Two factual probes with deterministic sampling, quoted
# verbatim. Run identically with and without kt so a numerical regression from
# CPU experts shows up as a changed answer rather than as a vibe.
set -u
BASE=http://127.0.0.1:31651
TAG=${TAG:-untagged}
STAMP=$(date +%H%M%S)
RESULT=/root/651-p2/results/qual655_${TAG}_${STAMP}.txt
mkdir -p /root/651-p2/results
exec > >(tee -a "$RESULT") 2>&1
echo "=== quality $TAG $STAMP ==="

ask() {
  python3 - "$BASE" "$1" "$2" <<'PY'
import json, sys, urllib.request
base, prompt, maxtok = sys.argv[1], sys.argv[2], int(sys.argv[3])
req = {"model": "qwen36-35b-a3b",
       "messages": [{"role": "user", "content": prompt}],
       "chat_template_kwargs": {"enable_thinking": False},
       "temperature": 0, "max_tokens": maxtok}
r = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(req).encode(),
                           headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=600))
print("PROMPT:", prompt)
print("ANSWER:", d["choices"][0]["message"]["content"])
print("---")
PY
}

ask "What is 17 times 23? Answer with just the number." 48
ask "List the first 12 Fibonacci numbers starting from 1, comma separated. Answer with just the list." 96
ask "Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters. Code only." 192
echo "=== quality done $TAG ==="
