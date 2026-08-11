#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 FINAL ACCEPTANCE RUN, successor 33.
#
# s25_acceptance_run.sh plus the two spec legs that no run in this chain has
# carried, scheduled INSIDE the window rather than beside it:
#
#   * the bs1 leg (spec items 3+4): one session whose context passes 262144,
#     which is simultaneously the YaRN long-context proof and the largest KV
#     residency a single request can demand. It is fired twice, spaced, so
#     that the corridor sampler covers a bs1 regime in BOTH phases.
#   * the KV rung (spec item 12): the rung fires when the corridor gate arms
#     and the cheaper tiers cannot cover the deficit. The bs1 leg is what
#     creates that deficit -- a 270k-token session holds rows the allocator
#     cache cannot hand back, so the hoard the cheap tier lives on is small
#     exactly when the gate arms.
#
# The legs are DELIBERATELY not the acceptance carrier. Spec item 14 says the
# carrier is real agent traffic through router 30099; these are pressure and
# proof, and the extract labels them as such.
#
# Usage: bash scripts/s33_acceptance_run.sh <minutes> <outdir>
set -uo pipefail

MINS="${1:-65}"
OUT="${2:?outdir}"
PY=/spinning/htsglang-gpu/.venv/bin/python
WT=/spinning/wt-631-routea
export PYTHONPATH="$WT/python"

mkdir -p "$OUT"
GRACE=$(python3 -c "print(int($MINS*60)+300)")

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/started_at"
curl -s -m 6 http://127.0.0.1:30030/get_server_info > "$OUT/server_info.json" 2>/dev/null
grep -oE '"max_total_num_tokens":[0-9]+' "$OUT/server_info.json" | head -1 | cut -d: -f2 > "$OUT/pool"
echo "acceptance: ${MINS} min, pool $(cat "$OUT/pool") -> $OUT"

# Corridor first so it covers every other leg including its ramp.
setsid nohup bash "$WT/scripts/corridor_sample.sh" "$GRACE" "$OUT/corridor.csv" \
    > "$OUT/corridor.stderr" 2>&1 < /dev/null &
echo "corridor pid $!"

# bs=4 mixed load: the steady load the flip policy oscillates against.
setsid nohup $PY "$WT/scripts/soak_631_mixed_load.py" \
    --minutes "$MINS" --decode-streams 2 \
    --prefill-tokens 60000 --prefill-period 6 \
    > "$OUT/soak.log" 2>&1 < /dev/null &
echo "soak pid $!"

# The bs1 legs, fired from a shepherd so the window stays unmanned. The first
# is late enough that the steady load has established a phase rhythm, the
# second far enough after it to land in a different phase.
setsid nohup bash -c "
  sleep 600
  $PY $WT/scripts/s33_yarn_bs1_leg.py --tokens 272000 --max-tokens 48 \
      --out $OUT/yarn_leg1.json > $OUT/yarn_leg1.log 2>&1
  sleep 900
  $PY $WT/scripts/s33_yarn_bs1_leg.py --tokens 272000 --max-tokens 48 \
      --out $OUT/yarn_leg2.json > $OUT/yarn_leg2.log 2>&1
" > "$OUT/legs.log" 2>&1 < /dev/null &
echo "bs1/YaRN leg shepherd pid $!"

echo "legs launched at $(date -u +%H:%M:%SZ). Agent traffic starts now, NO model override."
