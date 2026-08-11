#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #657 successor 39: the confirmation window for ALLOCATION STEERING.
#
# The ship configuration, byte-identical to s38's, PLUS one armed mechanism:
# SGLANG_CORRIDOR_STEERING=1. Everything else -- margin 512, delay budget 2,
# lender OFF, arming floor 1536, corridor law 1024 -- is what s38 shipped, so
# any axis that moves is attributable to the steer and to nothing else.
#
# WHAT THIS WINDOW IS FOR, stated before it runs so the verdict cannot be
# fitted to it afterwards. The steer's own effect on the free column is
# predicted to be NIL: the scheduler holds one allocator (the PP stack's,
# where a slot id IS the row on every rank), the TP pools are pre-sized at
# boot, and the only residency lever is the PP pool's backing watermark,
# floored by the MAXIMUM LIVE SLOT ID. A residue-class steer changes which
# ids are handed out, not how many rows a card commits. So this window tests
# three things:
#
#   1. SAFETY, which is the real question. The steer reorders replicated
#      scheduler state, and its decision must be identical on all three
#      ranks. The seam's MIN reduction agrees it, and a checksum of the free
#      list rides the same reduction: a disagreement DISARMS the mechanism.
#      A window with 0 disarms and a resolved UUID permutation is the first
#      metal evidence that the free list really is replicated.
#   2. NO REGRESSION on every axis s34/s37/s38 hold.
#   3. The PLACEMENT measurement -- per-rank share of the decisions -- and
#      the free-headroom spread series, judged against s38's own window.
#
# Usage: bash scripts/s39_steer_window.sh <minutes> <outdir>
set -uo pipefail

MINS="${1:-50}"
OUT="${2:?outdir}"
WT=/spinning/wt-631-routea
export PYTHONPATH="$WT/python"
LOG="$OUT/serving.log"

mkdir -p "$OUT"

LOG="$LOG" SELF=656-successor39 \
ARGV_SRC=/tmp/s33_argv.txt ENV_SRC=/tmp/s30_env.txt \
EXTRA_ENV='SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8
SGLANG_CORRIDOR_FLOOR_MIB=1536
SGLANG_KV_BACKING_RELIEF=1
SGLANG_FLIP_SEAM_CHUNK_MIB=8
SGLANG_CORRIDOR_REBALANCE=0
SGLANG_SEAM_ENTRY_MARGIN_MIB=512
SGLANG_SEAM_ENTRY_DELAY_BUDGET=2
SGLANG_CORRIDOR_STEERING=1' bash "$WT/scripts/s33_boot_from_capture.sh" || exit 3

for _ in $(seq 1 90); do
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:30030/health)" = "200" ] && break
  sleep 10
done
if [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:30030/health)" != "200" ]; then
  echo "STEER BOOT FAILED: health never reached 200" >&2
  exit 4
fi
echo "steer config UP at $(date -u +%H:%M:%SZ); serving is on 30030"

bash "$WT/scripts/s34_acceptance_run.sh" "$MINS" "$OUT"
