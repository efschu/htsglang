#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 37: make the C20 DELAY and YIELD branches fire ON METAL.
#
# WHY THIS EXISTS. The acceptance window funds the margin on every seam, which
# is the outcome the feature wants and the worst possible coverage: the two
# branches that make the mechanism SAFE -- delay when the margin is short,
# yield to the law when the budget is spent -- never execute. They are pinned
# on CPU and mutation-checked, and this corpus has shipped seven mechanisms
# that were green in exactly that state and inert in production.
#
# So: boot with a margin no ladder on this rig can fund (8 GiB) and watch what
# the gate does. The pass conditions are the falsifier, not the feature:
#
#   * seams are DELAYED  (the gate refuses to enter without headroom)
#   * seams then YIELD   (the budget is spent, the law governs, decode lives)
#   * corridor breaches remain ZERO
#   * /health stays 200 and requests keep completing -- an unbounded refusal
#     of pp->tp is the 411-abandon wedge and the budget is what prevents it
#
# It is a PROBE, not an acceptance: a deliberately impossible margin is not a
# shipping configuration, and its corridor numbers describe that margin rather
# than the shipped one.
#
# Usage: bash scripts/s37_delay_probe.sh <minutes> <outdir>
set -uo pipefail

MINS="${1:-6}"
OUT="${2:?outdir}"
PY=/spinning/htsglang-gpu/.venv/bin/python
WT=/spinning/wt-631-routea
export PYTHONPATH="$WT/python"
LOG="$OUT/serving.log"

mkdir -p "$OUT"

LOG="$LOG" SELF=656-successor37 \
ARGV_SRC=/tmp/s33_argv.txt ENV_SRC=/tmp/s30_env.txt \
EXTRA_ENV='SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8
SGLANG_CORRIDOR_FLOOR_MIB=1536
SGLANG_KV_BACKING_RELIEF=1
SGLANG_FLIP_SEAM_CHUNK_MIB=8
SGLANG_CORRIDOR_REBALANCE=0
SGLANG_SEAM_ENTRY_MARGIN_MIB=8192
SGLANG_SEAM_ENTRY_DELAY_BUDGET=2' bash "$WT/scripts/s33_boot_from_capture.sh" || exit 3

for _ in $(seq 1 60); do
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:30030/health)" = "200" ] && break
  sleep 10
done

setsid nohup bash "$WT/scripts/corridor_sample.sh" $((MINS * 60 + 120)) "$OUT/corridor.csv" \
    > "$OUT/corridor.stderr" 2>&1 < /dev/null &
setsid nohup $PY "$WT/scripts/soak_631_mixed_load.py" \
    --minutes "$MINS" --decode-streams 2 \
    --prefill-tokens 60000 --prefill-period 6 \
    > "$OUT/soak.log" 2>&1 < /dev/null &

echo "probe running ${MINS} min -> $OUT"
sleep $((MINS * 60))

echo
echo "===== C20 DELAY PROBE, margin 8192 MiB (deliberately unfundable)"
for p in "C20 entry margin" "seam entry DELAYED" "seam entry margin YIELDED" \
         "corridor gate refused the seam staging" "cutover pp_to_tp" "Traceback"; do
  printf '   %-42s %s\n' "$p" "$(grep -oF "$p" "$LOG" | wc -l)"
done
echo "   health $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:30030/health)"
tail -1 "$OUT/soak.log"
$PY - "$OUT/corridor.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if rows:
    m = [min(int(r[f"gpu{g}_free"]) for r in rows) for g in (0, 1, 2)]
    b = sum(1 for r in rows if min(int(r[f"gpu{g}_free"]) for g in (0, 1, 2)) < 1024)
    print(f"   corridor MIN {m[0]}/{m[1]}/{m[2]} MiB   breaches below 1024: {b}")
PYEOF
echo
echo "PASS means: delays > 0 AND yields > 0 AND breaches 0 AND health 200."
echo "A run with 0 delays has not exercised the branch and proves nothing."
