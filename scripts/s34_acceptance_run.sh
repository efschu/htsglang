#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 FINAL ACCEPTANCE RUN, successor 34.
#
# s33_acceptance_run.sh, plus the one leg that chain never got to run: the
# occupancy pressure that spec item 12's KV rung needs in order to be the
# funder rather than a spectator.
#
# WHY THE OCCUPANCY LEG IS INSIDE THE WINDOW AND NOT BESIDE IT. The rung
# declined on every one of ~324 seam legs across two acceptance runs, and the
# reconstruction (HANDOFF_678 1b) shows why: its deficit is
#
#     floor + delta + want - free - cheap_relief
#
# and ``cheap_relief`` -- torch's reserved-minus-allocated hoard, 766 MiB
# median -- was larger than the gap on 100% of arms. Two things this boot
# does change that, and BOTH of them are properties of the run rather than
# of a switch:
#
#   * the arming floor is raised to 1536 MiB (the corridor LAW stays 1024 and
#     the verdict is still read against 1024 -- see get_corridor_guard, which
#     passes law_floor_mib explicitly for exactly this reason), which lifts
#     the ``floor`` term over the hoard on 91 of s33's 93 measured arms;
#   * the new prefill-admission gate drains that same hoard continuously on
#     the hot path, so the ``cheap_relief`` the seam discounts against is
#     smaller by the time the seam asks.
#
# The legs are DELIBERATELY not the acceptance carrier. Spec item 14 says the
# carrier is real agent traffic through router 30099; these are pressure and
# proof, and the extract labels them as such.
#
# Usage: bash scripts/s34_acceptance_run.sh <minutes> <outdir>
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

# The pressure and proof legs, from a shepherd so the window stays unmanned.
#
#   t+5   occupancy: drive live slots up, which raises the seam's staging
#         term and lowers free -- both terms of the KV rung's deficit, in the
#         direction that makes it fire.
#   t+14  bs1/YaRN leg 1 (spec items 3+4), above 262144.
#   t+26  occupancy again, in a different phase.
#   t+38  bs1/YaRN leg 2, so the sampler covers a bs1 regime in both phases.
setsid nohup bash -c "
  sleep 300
  $PY $WT/scripts/s33_occupancy_leg.py --sessions 3 --tokens 130000 \
      --rounds 1 --out $OUT/occupancy1.json > $OUT/occupancy1.log 2>&1
  sleep 240
  $PY $WT/scripts/s33_yarn_bs1_leg.py --tokens 272000 --max-tokens 48 \
      --out $OUT/yarn_leg1.json > $OUT/yarn_leg1.log 2>&1
  sleep 300
  $PY $WT/scripts/s33_occupancy_leg.py --sessions 3 --tokens 130000 \
      --rounds 1 --out $OUT/occupancy2.json > $OUT/occupancy2.log 2>&1
  sleep 300
  $PY $WT/scripts/s33_yarn_bs1_leg.py --tokens 272000 --max-tokens 48 \
      --out $OUT/yarn_leg2.json > $OUT/yarn_leg2.log 2>&1
" > "$OUT/legs.log" 2>&1 < /dev/null &
echo "leg shepherd pid $!"

echo "legs launched at $(date -u +%H:%M:%SZ). Agent traffic starts now, NO model override."
