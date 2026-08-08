#!/bin/bash
# #655: axis B redo (real decode length) plus axis C (exact agent-shaped request).
#
# WHY A REDO. The first axis B was invalid: it raised max_tokens but the model
# reached EOS after ~10 tokens every time, so completion_tokens never exceeded
# 10 and long decode was never exercised. `ignore_eos` forces the sampler to run
# the full budget, which is what actually tests a long dispatch chain.
#
# Axis C replays the request shape that is KNOWN to have wedged this GPU: the
# efeu-code system prompt plus a task, at max_tokens=700. Axis A having stayed
# clean to 1939 prompt tokens means length alone does not explain that wedge, so
# the remaining suspects are decode length, the system role, and turn growth.
set -u

BASE=http://127.0.0.1:31651
MODEL=qwen36-35b-a3b
STAMP=$(date +%H%M%S)
RESULT=/root/651-p2/results/wedge_axis_bc_${STAMP}.txt
mkdir -p /root/651-p2/results
exec > >(tee -a "$RESULT") 2>&1

resets() { dmesg -T 2>/dev/null | grep -cE "GPU reset\("; }
R0=$(resets)
echo "=== axis B/C ${STAMP}, baseline GPU resets=${R0} ==="

# $1 = python payload builder mode, $2 = numeric arg, $3 = label
run_arm() {
  local mode="$1" n="$2" label="$3"
  local body t0 t1 ptok ctok r health
  t0=$(date +%s)
  body=$(python3 - "$mode" "$n" "$MODEL" <<'PY' | curl -s -m 900 "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' --data-binary @-
import json, sys
mode, n, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]

SYSTEM_PROMPT = """You are a coding assistant on a Linux machine.
Reply with EXACTLY ONE action per message, in one of these forms:

WRITE <path>
```
<full file content>
```

RUN <shell command>

DONE <one line summary>

Rules: no explanations outside the action. Use WRITE to create or replace a
file. Use RUN to execute a command and see its output. Use DONE when the task
is finished."""

req = {"model": model, "temperature": 0,
       "chat_template_kwargs": {"enable_thinking": False}}
if mode == "decode":
    # ignore_eos makes the budget binding, so completion_tokens == max_tokens.
    req["messages"] = [{"role": "user", "content": "Write a long description of the sea."}]
    req["max_tokens"] = n
    req["ignore_eos"] = True
elif mode == "agent":
    req["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Write a Python script fib.py that prints the first 20 Fibonacci numbers, then run it."},
    ]
    req["max_tokens"] = n
elif mode == "sysonly":
    # System role present, decode pinned tiny: isolates the role itself.
    req["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 6 times 7?"},
    ]
    req["max_tokens"] = n
print(json.dumps(req))
PY
)
  t1=$(date +%s)
  read -r ptok ctok <<<"$(printf '%s' "$body" | python3 -c "
import json,sys
try:
    u = json.load(sys.stdin).get('usage', {})
    print(u.get('prompt_tokens','?'), u.get('completion_tokens','?'))
except Exception:
    print('ERR ERR')")"
  r=$(resets)
  health=$(curl -s -m 15 -o /dev/null -w "%{http_code}" "$BASE/health" 2>/dev/null)
  echo "${label} mode=${mode} n=${n} prompt_tokens=${ptok} completion_tokens=${ctok} elapsed=$((t1-t0))s resets=${r} health=${health}"
  if [ "$r" != "$R0" ]; then
    echo ">>> WEDGE on ${label} (mode=${mode} n=${n}); resets ${R0} -> ${r}"
    dmesg -T | tail -30
    return 1
  fi
  if [ "$health" != "200" ]; then
    echo ">>> UNHEALTHY on ${label} (mode=${mode} n=${n}) health=${health}"
    printf '%s\n' "$body" | head -c 600
    return 1
  fi
  return 0
}

echo "--- axis B: real decode length, ignore_eos ---"
for n in 64 256 512 700 1024 2048; do
  run_arm decode "$n" "B" || exit 1
done
echo "--- axis C1: system role present, tiny decode ---"
run_arm sysonly 8 "C1" || exit 1
echo "--- axis C2: exact agent-shaped first request ---"
run_arm agent 700 "C2" || exit 1
echo "=== axis B/C ${STAMP} complete; final resets=$(resets) ==="
