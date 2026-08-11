#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 36: the CONTROLLED arm of the rebalance lender's A/B.
#
# WHY THIS EXISTS. The confirmation window showed the lender doing exactly
# what it claims (a non-seam floor at its own configured watermark), and it
# also showed the soak generator completing 21 requests by t+24 where s34
# completed 59. Two windows, two loads, two boots -- so the honest position
# is that neither the corridor gain nor the throughput gap is attributed
# until the same recipe runs with the lender OFF.
#
# The gap is visible from t+2, before the lender had done much, which is an
# argument against the lender causing it. An argument is not a measurement.
#
# WHAT IS AND IS NOT CONTROLLED HERE:
#   controlled   argv (byte-identical capture), env (identical but for the
#                one switch), load script and its parameters, the code
#                (same commit, both arms of THIS comparison)
#   NOT          the agent traffic, which is real work and therefore varies;
#                and boot-to-boot state (prefix cache, thermals). Judge the
#                SOAK counters, which are the only generator with identical
#                parameters in both arms.
#
# Usage: bash scripts/s36_ab_lender_off.sh <minutes> <outdir>
set -uo pipefail

MINS="${1:-25}"
OUT="${2:?outdir}"
WT=/spinning/wt-631-routea

mkdir -p "$OUT"

echo "[ab] booting with SGLANG_CORRIDOR_REBALANCE=0"
LOG="$OUT/serving.log" SELF=656-successor36 \
ARGV_SRC=/tmp/s33_argv.txt ENV_SRC=/tmp/s30_env.txt \
EXTRA_ENV='SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8
SGLANG_CORRIDOR_FLOOR_MIB=1536
SGLANG_KV_BACKING_RELIEF=1
SGLANG_FLIP_SEAM_CHUNK_MIB=8
SGLANG_CORRIDOR_REBALANCE=0' \
    bash "$WT/scripts/s33_boot_from_capture.sh" || exit 1

for i in $(seq 1 90); do
    code=$(curl -s -m 4 -o /dev/null -w "%{http_code}" http://127.0.0.1:30030/health 2>/dev/null)
    [ "$code" = "200" ] && { echo "[ab] healthy after ${i}0s"; break; }
    sleep 10
done

# THE LENDER MUST BE PROVABLY OFF, not assumed off. An arm line here would
# invalidate the whole comparison, so the run refuses rather than produce a
# number nobody can trust.
if grep -q "CORRIDOR-REBALANCE ARMED" "$OUT/serving.log" 2>/dev/null; then
    echo "[ab] REFUSE: the lender armed despite SGLANG_CORRIDOR_REBALANCE=0" >&2
    exit 2
fi
echo "[ab] confirmed: no lender arm line"

bash "$WT/scripts/s34_acceptance_run.sh" "$MINS" "$OUT"
