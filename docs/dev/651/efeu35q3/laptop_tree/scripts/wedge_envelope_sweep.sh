#!/bin/bash
# #655: bisect the amdgpu MES wedge envelope on gfx1103, two axes.
#
# WHY TWO AXES. The datum this replaces -- "a ~400-token agent prompt wedged on
# its first request while a 20-token probe survived" -- is confounded. The agent
# request differs from the surviving probe in TWO variables, not one: it carries
# a ~200-token system prompt AND asks for max_tokens=700. A 700-token decode is
# 700 sequential dispatches, which is at least as plausible an MES trigger as a
# long prefill. Sweeping only prefill would return a clean table and be read as
# "no envelope problem", which would be wrong.
#
# So: axis A walks PREFILL length with the decode pinned tiny; axis B walks
# DECODE length with the prompt pinned tiny. Whichever axis faults, faults alone.
#
# Each arm is bounded (curl -m), records the usage-reported prompt_tokens rather
# than trusting the word count, and re-reads the dmesg reset counter afterwards.
# A wedge ends its axis -- the safe envelope is below the faulting rung.
set -u

BASE=http://127.0.0.1:31651
MODEL=qwen36-35b-a3b
STAMP=$(date +%H%M%S)
RESULT=/root/651-p2/results/wedge_envelope_${STAMP}.txt
mkdir -p /root/651-p2/results
exec > >(tee -a "$RESULT") 2>&1

resets() { dmesg -T 2>/dev/null | grep -cE "GPU reset\("; }
R0=$(resets)
echo "=== wedge envelope sweep ${STAMP}, baseline GPU resets=${R0} ==="
echo "=== axis A: prefill length, max_tokens pinned to 8 ==="

# One arm. $1 = filler word count, $2 = max_tokens, $3 = label.
arm() {
  local words="$1" maxtok="$2" label="$3"
  local body t0 t1 ptok ctok r health
  t0=$(date +%s)
  body=$(python3 - "$words" "$maxtok" "$MODEL" <<'PY' | curl -s -m 600 "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' --data-binary @-
import json, sys
words, maxtok, model = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
filler = " ".join(["alpha bravo charlie delta echo"] * max(1, words // 5))
prompt = ("Repeat back only the word OK. " + filler) if words else "What is 6 times 7?"
print(json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "chat_template_kwargs": {"enable_thinking": False},
    "temperature": 0,
    "max_tokens": maxtok,
}))
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
  echo "${label} words=${words} max_tokens=${maxtok} prompt_tokens=${ptok} completion_tokens=${ctok} elapsed=$((t1-t0))s resets=${r} health=${health}"
  if [ "$r" != "$R0" ]; then
    echo ">>> WEDGE on ${label}: prompt_tokens=${ptok} completion_tokens=${ctok} (resets ${R0} -> ${r})"
    dmesg -T | tail -25
    return 1
  fi
  if [ "$health" != "200" ]; then
    echo ">>> UNHEALTHY on ${label} without a reset (health=${health})"
    return 1
  fi
  return 0
}

AXIS_A_LIMIT=""
for words in 0 20 50 100 200 400 800 1600; do
  arm "$words" 8 "A" || { AXIS_A_LIMIT="$words"; break; }
done
if [ -z "$AXIS_A_LIMIT" ]; then
  echo "=== axis A clean through 1600 filler words at max_tokens=8 ==="
else
  echo "=== axis A faulted at ${AXIS_A_LIMIT} filler words ==="
fi

echo "=== axis B: decode length, prompt pinned tiny ==="
R0=$(resets)
AXIS_B_LIMIT=""
for maxtok in 8 64 256 512 700 1024; do
  arm 0 "$maxtok" "B" || { AXIS_B_LIMIT="$maxtok"; break; }
done
if [ -z "$AXIS_B_LIMIT" ]; then
  echo "=== axis B clean through max_tokens=1024 ==="
else
  echo "=== axis B faulted at max_tokens=${AXIS_B_LIMIT} ==="
fi

echo "=== sweep ${STAMP} complete; final resets=$(resets) ==="
