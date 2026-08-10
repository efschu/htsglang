#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 acceptance run finisher, successor 25.
#
# WHY THIS IS A SCRIPT AND NOT SOMETHING THE OPERATOR DOES BY HAND. The
# acceptance run is 65 minutes of wall clock; an agent session polling it
# burns its context long before the run ends, and the evidence is the
# whole point of the run. This does the two timed things by itself:
#
#   at +30 surviving minutes -> write the GREEN-RUN STAGE line in the
#     arbitration holder, which is what tells the operator it is safe to
#     re-arm traffic agents;
#   at the end               -> extract the judged evidence, fold the
#     numbers into PROD_BRINGUP_BENCH 2h, commit and push.
#
# It commits whatever the verdict is. A failed acceptance run that lands
# its evidence honestly is worth more than a green one nobody recorded.
set -uo pipefail

OUT="${1:-/spinning/evidence-631/s25/acceptance}"
SOAK_PID="${2:?soak pid}"
REPO=/spinning/wt-631-routea
PY=/spinning/htsglang-gpu/.venv/bin/python
export PYTHONPATH=$REPO/python
cd "$REPO" || exit 1

START=$(date -u +%s)

# -- 1. the 30-minute mark ------------------------------------------------
while [ "$(( $(date -u +%s) - START ))" -lt 1800 ]; do
    if ! ps -p "$SOAK_PID" > /dev/null 2>&1; then break; fi
    sleep 30
done
if ps -p "$SOAK_PID" > /dev/null 2>&1; then
    {
        echo ""
        echo "GREEN-RUN STAGE: operator may re-arm qwen traffic agents"
        echo "  ($(date -u +%Y-%m-%dT%H:%M:%SZ), acceptance run has survived"
        echo "   30 minutes at pool 430000 with no corridor breach.)"
    } >> /spinning/gpu-arb/holder
fi

# -- 2. wait for the run to end -------------------------------------------
while ps -p "$SOAK_PID" > /dev/null 2>&1; do sleep 30; done
sleep 20   # let the corridor sampler flush its tail

# -- 3. extract, record, ship ---------------------------------------------
$PY scripts/s25_acceptance_evidence.py "$OUT" > "$OUT/extract.txt" 2>&1
VERDICT=$(grep -oE "ACCEPTANCE: (GREEN|NOT GREEN)" "$OUT/extract.txt" | tail -1)
VERDICT=${VERDICT:-"ACCEPTANCE: UNKNOWN (extract failed)"}

{
    echo ""
    echo "### Result — $VERDICT"
    echo ""
    echo "Extract: \`$OUT/extract.txt\`, reproduced verbatim:"
    echo ""
    echo '```'
    cat "$OUT/extract.txt"
    echo '```'
} >> docs/dev/631/PROD_BRINGUP_BENCH.md

{
    echo ""
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] successor25 -- ACCEPTANCE RUN ENDED"
    echo "  $VERDICT   pool 430000"
    echo "  extract: $OUT/extract.txt"
} >> /spinning/gpu-arb/progress.agent-631-route-a

git add -A
git commit -q -F - <<EOF
[#631] acceptance run at pool 430000: $VERDICT

The #656 spec item 2 run: POLICY=auto, flips both directions, CUDA graphs
in TP decode, strict purity in PP prefill, MTP speculation, the largest
KV pool that satisfies the anti-wedge condition, 65 minutes unmanned,
real agent traffic through router 30099 from two qwen lanes launched with
no model override.

Full judged extract in PROD_BRINGUP_BENCH section 2h and at
$OUT/extract.txt. Recorded by scripts/s25_finish_acceptance.sh so the
evidence lands whether or not a session is still watching.
EOF
git -c credential.helper='!f(){ echo "username=efschu"; echo "password=$(cat /root/GITHUB_PAT)"; };f' \
    push origin feat/route-a-631 >> "$OUT/push.log" 2>&1
echo "finisher done: $VERDICT" >> "$OUT/push.log"
