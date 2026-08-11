#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 40: the shared confirmation window for #658 and #661.
#
# ONE BOOT, TWO QUESTIONS, IN AN ORDER THAT KEEPS THEM APART.
#
# The ship config plus exactly two added flags:
#   --enable-vram-dial        (#658: the budget dial, wired to the corridor
#                              guard's relief ladder by this shift)
#   --enable-dynamic-chunking (#661: the arm whose only value hypothesis is
#                              downward adaptivity under mixed load)
#
# WHY BOTH IN ONE BOOT, AND WHY THAT IS NOT A CONFOUND. Each is a boot-time
# flag, so each costs a boot; the rig has one window. They are kept apart in
# TIME instead: the chunk arm runs FIRST, while the dial sits armed but never
# dialled (it changes nothing until a budget request arrives -- the endpoint
# is the only mutation door), and the budget cycle runs AFTER the chunk arm
# has written its result. So the chunk measurement never overlaps a capacity
# commit, and the budget cycle inherits a warm, loaded instance, which is the
# state an external tenant would actually arrive in.
#
# THE STATIC ARM IS NOT RUN HERE. It was measured on the SHIPPED instance
# before this boot (/spinning/evidence-631/s40/phaseA), which is the honest
# baseline: the ship config, unmodified, with the same harness and the same
# mixed load. The cross-boot caveat is real and is recorded in the handoff --
# a boot-time flag cannot be A/B'd within one boot, so each arm carries its
# OWN A-vs-A noise floor and the comparison is judged against those.
#
# Usage: bash scripts/s40_window.sh <outdir>
set -uo pipefail

OUT="${1:?outdir}"
WT=/spinning/wt-631-routea
PY=/spinning/htsglang-gpu/.venv/bin/python
export PYTHONPATH="$WT/python"
mkdir -p "$OUT"

echo "== host RAM before boot (the session has been OOM-killed by probe boots twice)"
free -g | sed -n '2p'
AVAIL=$(free -g | awk 'NR==2{print $7}')
if [ "$AVAIL" -lt 15 ]; then
  echo "REFUSE: only ${AVAIL} GiB available, the floor is 15" >&2
  exit 2
fi

LOG="$OUT/serving.log" SELF=656-successor40 \
ARGV_SRC="${S40_ARGV_SRC:-/tmp/s40_ship_argv.txt}" ENV_SRC=/tmp/s30_env.txt \
ARGV_SET="${S40_ARGV_SET:---enable-dynamic-chunking}" \
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
  echo "BOOT FAILED: health never reached 200" >&2
  exit 4
fi
echo "s40 window UP at $(date -u +%H:%M:%SZ)"

# The dial must be ARMED before anything is judged: a window that dialled a
# disabled dial would record a polite refusal as a null result.
curl -s -m 20 -X POST http://127.0.0.1:30030/vram_budget \
  -H 'Content-Type: application/json' -d '{"query":true}' \
  | tee "$OUT/dial_status_boot.json"; echo
