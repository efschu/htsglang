#!/bin/bash
# #651: at chunked-prefill 256, how LONG a prompt can this GPU survive?
#
# The chunk cap is already armed and still wedged on an agent-sized prompt, so
# the uncovered dimension is total prefill LENGTH -- i.e. how many consecutive
# chunk GEMMs run back to back, not how big one is. This walks the length up
# and stops at the first fault, so the safe envelope is measured rather than
# guessed.
#
# max_tokens is tiny throughout: we are testing prefill, and generation would
# only add time and confounds.
set -u
BASE=http://127.0.0.1:31651
RESULT=/root/651-p2/results/prefill_sweep_$(date +%H%M%S).txt
exec > >(tee -a "$RESULT") 2>&1

resets() { dmesg -T 2>/dev/null | grep -cE "GPU reset\("; }
R0=$(resets)
echo "=== prefill length sweep, cp=256, baseline GPU resets=$R0 ==="

for words in 200 500 1000 1500 2000 3000 4000 5000; do
  P=$(python3 -c "print(' '.join(['alpha bravo charlie delta echo'] * ($words // 5)))")
  T0=$(date +%s)
  BODY=$(curl -s -m 400 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-raw "$(python3 -c "
import json,sys
p = sys.argv[1]
print(json.dumps({'model':'qwen36-35b-a3b',
                  'messages':[{'role':'user','content':'Repeat back only the word OK. ' + p}],
                  'chat_template_kwargs':{'enable_thinking':False},
                  'temperature':0,'max_tokens':8}))" "$P")")
  T1=$(date +%s)
  PT=$(printf '%s' "$BODY" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); print(d.get('usage',{}).get('prompt_tokens','?'))
except Exception: print('ERR')")
  R=$(resets)
  UP=$(curl -s -m 10 -o /dev/null -w "%{http_code}" "$BASE/health" 2>/dev/null)
  echo "words=$words prompt_tokens=$PT elapsed=$((T1-T0))s resets=$R health=$UP"
  if [ "$R" != "$R0" ]; then
    echo ">>> WEDGED at prompt_tokens=$PT (resets $R0 -> $R). Safe envelope is BELOW this."
    exit 1
  fi
  if [ "$UP" != "200" ]; then
    echo ">>> server unhealthy at prompt_tokens=$PT without a reset"
    exit 1
  fi
done
echo "=== no wedge up to the largest prompt tested ==="
