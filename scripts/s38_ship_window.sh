#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 38: restore the SHIP configuration and run a confirmation
# window on it.
#
# THIS IS A REGRESSION CHECK, NOT A STAMP HUNT. s37's acceptance
# (/spinning/evidence-631/s37/accept2/EXTRACT.txt) is the standing green and
# this shift changed two things underneath it: the KV rung now keeps an
# admission reserve, and the seam gate bounds what it asks that rung for. On
# the shipped margin (512 MiB) both terms should be nearly INERT -- the rung
# has hundreds of thousands of rows of slack above a 512-row reserve, and the
# margin is fundable on every seam. The window exists to prove exactly that:
# same axes, no regression.
#
# Ship config, byte-identical to s37's: margin 512, delay budget 2, lender
# OFF, arming floor 1536, corridor law 1024, rebalance off.
#
# Usage: bash scripts/s38_ship_window.sh <minutes> <outdir>
set -uo pipefail

MINS="${1:-30}"
OUT="${2:?outdir}"
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
SGLANG_SEAM_ENTRY_MARGIN_MIB=512
SGLANG_SEAM_ENTRY_DELAY_BUDGET=2' bash "$WT/scripts/s33_boot_from_capture.sh" || exit 3

for _ in $(seq 1 90); do
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:30030/health)" = "200" ] && break
  sleep 10
done
if [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:30030/health)" != "200" ]; then
  echo "SHIP BOOT FAILED: health never reached 200" >&2
  exit 4
fi
echo "ship config UP at $(date -u +%H:%M:%SZ); serving is restored on 30030"

bash "$WT/scripts/s34_acceptance_run.sh" "$MINS" "$OUT"
