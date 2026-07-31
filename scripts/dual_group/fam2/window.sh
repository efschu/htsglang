#!/usr/bin/env bash
# #274 families slice 2: the ONE card window, three arms, budgeted.
#
# Order is by what only a card can decide, most first:
#   1. MoE lane  -- the expert shell has never met a real expert tensor.
#   2. dense two-card -- the cheapest possible proof of the cross-card path.
#   3. FP8 two-card   -- the same mechanism on the vehicle it was built for.
#
# Each arm gets a hard wall-clock share. An arm that overruns is killed and
# reported as such; the window never runs past its budget, and an arm that
# did not run is reported as NOT RUN rather than as a failure.

set -uo pipefail
cd "$(dirname "$0")"
source ../r7c/common.sh

OWNER="${OWNER:-agent-fam2}"
BUDGET_S="${BUDGET_S:-1800}"
ARMS="${ARMS:-moe,dense2,fp82}"
SUMMARY="${SUMMARY:-/tmp/fam2-window-summary.txt}"

declare -A SCRIPT=(
  [moe]=boot_moe_lane.sh
  [dense2]=boot_dense_twocard.sh
  [fp82]=boot_fp8_twocard.sh
)
declare -A SHARE_S=([moe]=720 [dense2]=420 [fp82]=900)
declare -A PIDFILE=(
  [moe]=/tmp/fam2-moe.pid
  [dense2]=/tmp/fam2-dense2.pid
  [fp82]=/tmp/fam2-fp8.pid
)

assert_cards_free || exit 1
claim_cards "274-families-2 (Budget ${BUDGET_S}s)"
trap 'release_cards "fam2 window abgebrochen"; exit 1' INT TERM

T0=$(date +%s)
: > "$SUMMARY"
for arm in ${ARMS//,/ }; do
  now=$(date +%s)
  left=$(( BUDGET_S - (now - T0) ))
  share=${SHARE_S[$arm]}
  if [ "$left" -lt 120 ]; then
    echo "$arm: NOT RUN (window budget exhausted, ${left}s left)" | tee -a "$SUMMARY"
    continue
  fi
  [ "$share" -gt "$left" ] && share=$left
  echo "=== arm $arm, share ${share}s ===" | tee -a "$SUMMARY"
  timeout -k 30 "$share" bash "${SCRIPT[$arm]}"
  rc=$?
  # Whatever happened, the server this arm started must not survive into the
  # next one: two 20+ GiB boots on one card is the failure mode that eats a
  # whole window.
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
  sleep 5
  echo "$arm: rc=$rc" | tee -a "$SUMMARY"
done

echo "--- window done after $(( $(date +%s) - T0 ))s of ${BUDGET_S}s ---" | tee -a "$SUMMARY"
release_cards "fam2 window beendet ($(tr '\n' ' ' < "$SUMMARY" | tail -c 200))"
