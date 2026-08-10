#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 FINAL ACCEPTANCE RUN (user spec item 2), successor 25.
#
# All axes at once, unmanned, on ONE log:
#   POLICY=auto, flips in BOTH directions
#   CUDA graphs active in TP decode, ABSENT in PP prefill (strict purity)
#   MTP speculation on
#   the largest KV pool that holds the corridor
#   corridor 1024 MiB per card, never breached AND well filled
#   real agent traffic through the router (launched separately, from the
#   operator session, so "did it carry traffic" is answered from the
#   serving log rather than from this script's intentions)
#
# Usage: bash scripts/s25_acceptance_run.sh <minutes> <outdir>
set -uo pipefail

MINS="${1:-65}"
OUT="${2:?outdir}"
PY=/spinning/htsglang-gpu/.venv/bin/python
export PYTHONPATH=/spinning/wt-631-routea/python

mkdir -p "$OUT"
GRACE=$(python3 -c "print(int($MINS*60)+300)")

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/started_at"
curl -s -m 6 http://127.0.0.1:30030/get_server_info > "$OUT/server_info.json" 2>/dev/null
grep -oE '"max_total_num_tokens":[0-9]+' "$OUT/server_info.json" | head -1 | cut -d: -f2 > "$OUT/pool"
echo "acceptance run: ${MINS} min, pool $(cat "$OUT/pool") -> $OUT"

# Corridor first so it covers every other leg including its ramp.
setsid nohup bash scripts/corridor_sample.sh "$GRACE" "$OUT/corridor.csv" \
    > "$OUT/corridor.stderr" 2>&1 < /dev/null &
echo "corridor pid $!"

# bs=4 mixed load: decode streams plus periodic long prefills, the steady
# load the flip policy oscillates against.
setsid nohup $PY scripts/soak_631_mixed_load.py \
    --minutes "$MINS" --decode-streams 2 \
    --prefill-tokens 60000 --prefill-period 6 \
    > "$OUT/soak.log" 2>&1 < /dev/null &
echo "soak pid $!"

echo "legs launched. Start the agent traffic now, with NO model override."
