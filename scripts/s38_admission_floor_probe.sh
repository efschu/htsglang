#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 38: the SAME unfundable margin that killed the instance,
# against the admission floor that is supposed to make it survivable.
#
# THE FAILURE THIS REPRODUCES (HANDOFF_681 §1a, metal, 2026-08-11). With
# SGLANG_SEAM_ENTRY_MARGIN_MIB=8192 the seam gate's ask flows into the
# collective KV rung, whose floor was ``max_live + 1`` -- it protects the rows
# that EXIST and reserves nothing to admit with. 42 cutovers later, on all
# three ranks:
#
#   RuntimeError: Out of memory. Try to allocate 512 tokens.
#   Available full tokens: 0 (full_available_size=0 + full_evictable_size=0)
#   scheduler.py:5429 _get_new_batch_prefill_raw -> alloc_for_extend
#
# The delay and yield branches were exonerated by their own timeline (first
# delay 19:23:03, first yield 19:23:09, first traceback 19:25:15).
#
# PASS CONDITIONS, and the first one is the whole point:
#
#   * health 200 at the END, with the soak still completing requests -- the
#     run that produced this probe was health 000 at this point;
#   * ZERO "Available full tokens: 0" and zero tracebacks;
#   * delays > 0 AND yields > 0 -- the branches still execute, so the survival
#     is not the survival of a mechanism that failed to arm;
#   * corridor breaches 0.
#
# The bounded ask is visible too: "KV rung asked for N of 8192 MiB
# discretionary" is the gate declining to demand what the rung cannot fund.
#
# It is a PROBE, not an acceptance: 8 GiB is not a shipping margin, and its
# corridor numbers describe that margin rather than the shipped one.
#
# Usage: bash scripts/s38_admission_floor_probe.sh <minutes> <outdir>
set -uo pipefail

MINS="${1:-10}"
OUT="${2:?outdir}"
PY=/spinning/htsglang-gpu/.venv/bin/python
WT=/spinning/wt-631-routea
export PYTHONPATH="$WT/python"
LOG="$OUT/serving.log"

mkdir -p "$OUT"

LOG="$LOG" SELF=656-successor38 \
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
echo "===== ADMISSION-FLOOR PROBE, margin 8192 MiB (deliberately unfundable)"
for p in "C20 entry margin" "seam entry DELAYED" "seam entry margin YIELDED" \
         "KV rung asked for" "corridor gate refused the seam staging" \
         "cutover pp_to_tp" "Available full tokens: 0" "Traceback"; do
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
    print(f"   corridor MIN {m[0]}/{m[1]}/{m[2]} MiB   breaches below 1024: {b} "
          f"({len(rows)} samples)")
PYEOF
echo
echo "PASS: health 200 AND 0 'Available full tokens: 0' AND 0 tracebacks AND"
echo "      delays > 0 AND yields > 0 AND breaches 0."
echo "A run with 0 delays has not exercised the branch and proves nothing."
