#!/usr/bin/env bash
# Correctness probe for the #631 Route A PD pair.
#
# JUDGED BY CONTENT, NOT BY STATUS CODE. A token-sharded handover that drops or
# misfiles rows does not error: the decode arm keeps sampling from whatever is
# in its compact rows and returns fluent, grammatical, WRONG text. A 200 here
# means "the HTTP path works", nothing more. So every probe below has an answer
# that is checkable without judgement -- an ordered sequence, an exact string,
# an arithmetic fact -- and the failure mode being hunted (a middle that loses
# or repeats items while the sentence stays well-formed) is visible in it.
set -uo pipefail

PORT="${1:-8100}"
URL="http://127.0.0.1:$PORT"

ask() {
    local name="$1" prompt="$2" maxtok="${3:-96}"
    echo "=================================================================="
    echo "PROBE: $name"
    echo "PROMPT: $prompt"
    local out
    out=$(curl -s -m 180 "$URL/generate" -H 'Content-Type: application/json' \
        -d "{\"text\": $(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
             \"sampling_params\": {\"temperature\": 0, \"max_new_tokens\": $maxtok}}" 2>&1)
    echo "RAW: $(printf '%s' "$out" | head -c 400)"
    echo "TEXT: $(printf '%s' "$out" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin)
    print(d["text"] if isinstance(d,dict) else d[0]["text"])
except Exception as e:
    print("<<unparseable:",e,">>")' 2>&1)"
    echo
}

echo "### health ###"
curl -s -m 5 "$URL/health" -o /dev/null -w 'proxy health: %{http_code}\n'

# 1. Ordered sequence. The canonical detector for a misfiled token shard: the
#    sentence stays grammatical while the MIDDLE loses or repeats items.
ask "counting" "Count from one to twenty in words, comma separated:" 120

# 2. Long-prefix recall. Forces a prompt long enough to span many KV rows, then
#    asks for one fact from the middle of it -- a handover that truncates or
#    mis-addresses rows answers confidently and wrongly.
ask "needle" "Here is a list of pairs. apple=17, banana=42, cherry=88, date=5, elderberry=61, fig=23, grape=94, honeydew=36. Question: what number is paired with elderberry? Answer with just the number." 24

# 3. Determinism at temperature 0: the same prompt twice must give the same
#    text. A rank-divergent decode group is a known failure family on this rig.
ask "determinism-a" "Name the seven days of the week in order:" 60
ask "determinism-b" "Name the seven days of the week in order:" 60
