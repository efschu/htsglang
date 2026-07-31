#!/usr/bin/env bash
# The #332 + families card window: four arms, hard-budgeted, most decisive
# first. Order is deliberate and the tail is the part that gets cut:
#
#   1. v4_tp3_nextn      -- #332 posten 1 + 2. Only a card can say whether the
#                           dequant fallback carries the 96-row GDN gates and
#                           whether the drafter is now built on real weights.
#   2. v4_solo_res2048   -- #332 posten 3. Reserve-based sizing against the
#                           beleg's 153,007 tokens at the same weights.
#   3. dense2_noradix    -- the families follow-up: was the VOID serving floor
#                           the prefix cache?
#   4. fp82_lowbudget    -- the families follow-up: budget or real defect?
#
# Each arm gets a wall-clock share. An arm that overruns is killed and reported
# as such; an arm that never started is reported as NOT RUN, not as a failure.

set -uo pipefail
source /spinning/gpu-battery-results/2026-07-31_332_fam_beleg/env.sh
source "$WT/scripts/dual_group/r7c/common.sh"

OWNER="agent-332-fam-beleg"
BUDGET_S="${BUDGET_S:-2100}"
ARMS="${ARMS:-v4tp3,v4solo,dense2,fp82}"
SUMMARY="${SUMMARY:-$OUT/window_summary.txt}"
declare -A SHARE_S=([v4tp3]=820 [v4solo]=430 [dense2]=380 [fp82]=470 [falsi]=300)

assert_cards_free || exit 1
claim_cards "332-fam-beleg (Budget ${BUDGET_S}s)"
trap 'release_cards "332-fam-beleg window aborted"; exit 1' INT TERM

eval "$(resolve_uuids | tee "$OUT/nvml_inventory.txt" | grep -E '^(FIVE|THREES)=')"
IFS=, read -r T0 T1 <<< "$THREES"
# Each arm runs in its own `bash -c` subshell, so anything an arm reads has to
# be EXPORTED, not merely assigned. An unexported FIVE would reach the solo arm
# as an empty CVD, which arm.sh now treats as "all cards" -- the one silent way
# this driver could run the wrong experiment.
export FIVE THREES T0 T1 OWNER SOLO_RESERVE_MIB="${SOLO_RESERVE_MIB:-2048}"
# Immutable copy of the run directory: the arms reassign OUT for themselves.
export RUNDIR="$OUT"
echo "resolved: 5090 -> ${FIVE}; 3080s -> ${T0} ${T1}" | tee "$OUT/cards_resolved.txt"

run_v4tp3() {
  # #332 posten 1 + 2. Uneven TP=3 (`--rank-tp-ratio auto`) with NEXTN, at the
  # anchor's KV dtype and context so the prefill/decode points are readable
  # against the fp8 TP=3 anchor of the 06:40 window.
  #
  # Placement is --rank-gpu-id, NOT CUDA_VISIBLE_DEVICES. scripts/nvfp4/
  # v4_boot_proof.sh pins the cards by UUID and passes --rank-tp-ratio auto
  # alone; that combination cannot start, because `auto` derives its per-rank
  # budgets from NVML totals and so requires --rank-gpu-id ("--rank-tp-ratio
  # auto requires --rank-gpu-id"). Checked off-card before this window. With
  # all three cards in play there is nothing to hide anyway, and index 0 here
  # resolves to the 5090 (budgets [29607, 17780, 17780] MiB), which is what the
  # 3000/2700/2700 reserve list assumes.
  bash "$OUT/arm.sh" v4_tp3_nextn "$V4_MODEL" 1 -- \
    --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto \
    --rank-auto-reserve-mib 3000,2700,2700 \
    --kv-cache-dtype fp8_e4m3 --context-length 32768 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
}

run_v4solo() {
  # #332 posten 3. Identical to the beleg's solo arm in every respect EXCEPT
  # the sizing knob: --mem-fraction-static 0.90 becomes a named reserve, so the
  # weight figure must not move and only the pool may grow.
  CVD="${FIVE}" bash "$OUT/arm.sh" v4_solo_res2048 "$V4_MODEL" 1,8 -- \
    --tp-size 1 --rank-auto-reserve-mib "${SOLO_RESERVE_MIB:-2048}" \
    --kv-cache-dtype fp8_e4m3 --context-length 32768
}

run_dense2() {
  # LOG is built from RUNDIR, not from OUT. In a command prefix bash expands
  # the assignments left to right and later ones already see the earlier ones,
  # so "OUT=$OUT/dense2_noradix LOG=$OUT/logs/..." put the log inside the arm's
  # own directory, where no logs/ exists -- the redirect failed and the server
  # died at once. Round 1 lost both family arms to exactly that.
  WT="$WT" OWNER="$OWNER" OUT="$RUNDIR/dense2_noradix" \
  LOG="$RUNDIR/logs/dense2_noradix.server.log" \
  EXTRA_ARGS="--disable-radix-cache" \
    bash "$WT/scripts/dual_group/fam2/boot_dense_twocard.sh"
}

run_fp82() {
  # Budget hypothesis, one boot: more air on BOTH cards. The ratio moves weight
  # onto the 5090 so the 3080 can carry the foreign lane part with room to
  # spare, the serving budget on the 3080 drops by 1000 MiB, and the lane pool
  # on the 5090 drops by 400 MiB. If the illegal access survives all of that,
  # the budget candidate is dead and the defect is real.
  WT="$WT" OWNER="$OWNER" OUT="$RUNDIR/fp82_lowbudget" \
  LOG="$RUNDIR/logs/fp82_lowbudget.server.log" \
  RATIO="6,1" RANK_MIB="27000,9500" LANE_BUDGET="600" \
    bash "$WT/scripts/dual_group/fam2/boot_fp8_twocard.sh"
}

run_falsi() {
  # Falsifier for ARM 1's stopper; the prediction is registered in the script.
  bash "$RUNDIR/falsifier_even.sh"
}

declare -A PIDFILE=(
  [v4tp3]="$OUT/logs/v4_tp3_nextn.pid"
  [v4solo]="$OUT/logs/v4_solo_res2048.pid"
  [falsi]="$OUT/logs/v4_tp3_even.pid"
  [dense2]=/tmp/fam2-dense2.pid
  [fp82]=/tmp/fam2-fp8.pid
)

T0S=$(date +%s)
: > "$SUMMARY"
for arm in ${ARMS//,/ }; do
  left=$(( BUDGET_S - ($(date +%s) - T0S) ))
  share=${SHARE_S[$arm]}
  if [ "$left" -lt 150 ]; then
    echo "$arm: NOT RUN (window budget exhausted, ${left}s left)" | tee -a "$SUMMARY"
    continue
  fi
  [ "$share" -gt "$left" ] && share=$left
  echo "=== arm $arm, share ${share}s, start $(date -u +%H:%M:%S) ===" | tee -a "$SUMMARY"
  timeout -k 30 "$share" bash -c "$(declare -f run_$arm); run_$arm"
  rc=$?
  # No server from one arm may survive into the next: two large boots on one
  # card is the failure mode that eats a whole window.
  pf="${PIDFILE[$arm]}"
  if [ -f "$pf" ]; then
    pid="$(cat "$pf" 2>/dev/null || true)"
    if [ -n "$pid" ]; then
      kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 1 30); do _pid_alive "$pid" || break; sleep 1; done
      _pid_alive "$pid" && { kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; }
    fi
    rm -f "$pf"
  fi
  sleep 6
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
    >> "$OUT/proofs/between_arms.csv"
  echo "$arm: rc=$rc  end $(date -u +%H:%M:%S)" | tee -a "$SUMMARY"
done

echo "--- window done after $(( $(date +%s) - T0S ))s of ${BUDGET_S}s ---" | tee -a "$SUMMARY"
release_cards "332-fam-beleg done ($(tr '\n' ' ' < "$SUMMARY" | tail -c 220))"
