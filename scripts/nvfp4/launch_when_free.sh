#!/usr/bin/env bash
# Waits for the cards, then starts the window -- once, and only on evidence.
#
# Two conditions must hold together, because either one alone lies: the
# arbitration `holder` must be gone (or name us), AND every card must be under
# 500 MiB. The file can go stale after a crash, and the hardware can still be
# in teardown seconds after a holder is deleted. Both are checked, and the
# hardware check is repeated after a settle pause so a card that is still
# releasing does not read as free.
#
# Polls on a fixed interval with a hard deadline; it never waits unbounded.
set -uo pipefail
OUT=/spinning/gpu-battery-results/2026-07-31_332_fam_beleg
ARB=/spinning/gpu-arb
DEADLINE_S="${DEADLINE_S:-8400}"
POLL_S="${POLL_S:-20}"
STATE="$OUT/waiter.log"

say() { echo "$(date -u +%H:%M:%S) $*" >> "$STATE"; }
: > "$STATE"

cards_free() {
  local busy
  busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          | awk -F', *' '$2 > 500 {print $1}')"
  [ -z "$busy" ]
}

holder_clear() {
  [ -f "$ARB/holder" ] || return 0
  grep -q 'session=agent-332-fam-beleg' "$ARB/holder" && return 0
  return 1
}

T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt "$DEADLINE_S" ]; do
  if holder_clear && cards_free; then
    say "candidate: holder clear and all cards under 500 MiB -- settling"
    sleep 15
    if holder_clear && cards_free; then
      say "CONFIRMED free, starting window"
      exec bash "$OUT/window.sh" >> "$OUT/window_driver.log" 2>&1
    fi
    say "settle check failed, back to polling"
  fi
  sleep "$POLL_S"
done
say "DEADLINE reached without a free window -- nothing was started"
exit 2
